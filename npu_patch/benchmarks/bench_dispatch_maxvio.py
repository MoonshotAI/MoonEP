"""MoonEP-Ascend dispatch-only benchmark with a MaxVio sweep.

This is the dispatch counterpart of ``bench_grad_reduce_maxvio.py``.  Every
case first runs the real planning kernel to materialize ``MoonEPCommPlan``;
the timed loop then reuses that plan and contains only the fused dispatch plus
local duplicate-expansion kernel.

The report contains two bandwidth views:

* ``logical_hidden_gbps`` uses ``S*K*H*2 / latency``, matching the MoonEP
  communication benchmark convention.
* ``aggregate_remote_payload_gbps`` decodes ``plan.dst`` and counts bytes that
  all EP ranks actually send to another rank.  Canonical hidden rows and fp32
  route weights are counted; local writes and local duplicate expansion are
  excluded.  This is aggregate EP traffic divided by slowest-rank latency.

EP group = the whole torchrun world.  Examples:

    # One node, 8 ranks (UID bootstrap)
    torchrun --nproc-per-node=8 npu_patch/benchmarks/bench_dispatch_maxvio.py

    # One node, 8 ranks, torchrun launcher without a process group
    bash npu_patch/benchmarks/run_bench.sh bench_dispatch_maxvio.py "--no-dist"

    # Two nodes, 8 ranks per node (run on each node with NODE_RANK=0/1)
    MASTER_ADDR=192.168.1.10 NODES=2 NODE_RANK=0 \
        bash npu_patch/benchmarks/run_bench.sh bench_dispatch_maxvio.py

    # Explicit SHMEM endpoint
    torchrun --nnodes=2 --nproc-per-node=8 --node-rank=0 \
        --master-addr=192.168.1.10 --master-port=29500 \
        npu_patch/benchmarks/bench_dispatch_maxvio.py \
        --ip-port tcp://192.168.1.10:29501
"""

import argparse
import csv
import os

import torch
import torch_npu
import torch.distributed as dist

from _bench_common import (
    add_engine_args,
    apply_engine,
    generate_topk_routing,
    setup,
)

from npu_patch import Buffer, ShmemRuntime
from npu_patch._common import sym_put_from
from npu_patch._device import local_device_index
from npu_patch.buffer_c import (
    create_sym_tensor_from_source,
    create_vmm_physical,
    reset_symmetric_descriptors,
)
from npu_patch.launchers import launch_dispatch, launch_planning
from npu_patch.planning import allocate_planning_outputs
from npu_patch.shmem_runtime import default_shmem_ip_port


def gbps(nbytes: int, us: float) -> float:
    return nbytes / (us * 1e-6) / 1e9 if us > 0 else 0.0


def time_op(fn, group, rank, warmup=20, iters=50, stats=None,
            remote_counts=(0, 0)):
    """Return per-rank, mean, and slowest-rank eager NPU-event latency."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    if stats is None:
        dist.barrier(group=group)
    else:
        ShmemRuntime.barrier()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    local_us = start.elapsed_time(end) / iters * 1e3

    if stats is not None:
        return stats.collect(local_us, *remote_counts)

    dev = torch.device(f"npu:{torch.npu.current_device()}")
    world_size = dist.get_world_size(group=group)
    local = torch.tensor([local_us], dtype=torch.float64, device=dev)
    gathered = [
        torch.empty(1, dtype=torch.float64, device=dev)
        for _ in range(world_size)
    ]
    dist.all_gather(gathered, local, group=group)
    times = [value.item() for value in gathered]
    mean_us = sum(times) / len(times)
    max_us = max(times)

    if rank == 0:
        print(
            f"  [time_op] per-rank: {[f'{value:.1f}' for value in times]} "
            f"max={max_us:.1f} mean={mean_us:.1f} us",
            flush=True,
        )

    dist.barrier(group=group)
    return {"per_rank_us": times, "mean_us": mean_us, "max_us": max_us}


def _maxvio_for_sigma(sigma, S, K, E, R, dev):
    total = torch.zeros(E, dtype=torch.int64, device=dev)
    for rank in range(R):
        _, tpe = generate_topk_routing(
            S, K, E, R, sigma, dev, 1234, rank=rank
        )
        total += tpe.to(torch.int64)
    return total.max().item() / (total.sum().item() / E) - 1.0


def solve_sigma_for_maxvio(target, S, K, E, R, dev):
    """Find a deterministic routing sigma whose realized MaxVio is closest."""
    lo, hi = 1e-5, 10.0
    mv_lo = _maxvio_for_sigma(lo, S, K, E, R, dev)
    if mv_lo >= target:
        return lo, mv_lo
    best_sigma, best_mv = lo, mv_lo
    for _ in range(60):
        mid = (lo * hi) ** 0.5
        mv = _maxvio_for_sigma(mid, S, K, E, R, dev)
        if abs(mv - target) < abs(best_mv - target):
            best_sigma, best_mv = mid, mv
        if abs(mv - target) <= max(0.02 * target, 0.005):
            return mid, mv
        if mv < target:
            lo = mid
        else:
            hi = mid
    return best_sigma, best_mv


class DispatchRunner:
    """Own the Buffer, plan, and fixed dispatch inputs for one benchmark."""

    def __init__(self, group, R, S, K, E, H, num_sms, local_rank, with_weights,
                 rank=None, use_udma=False):
        self.group = group
        self.rank = rank if rank is not None else dist.get_rank(group=group)
        self.R = R
        self.S = S
        self.K = K
        self.H = H
        self.with_weights = with_weights
        self.use_udma = use_udma
        self.buffer = Buffer(
            S,
            H,
            K,
            E,
            R,
            num_sms=num_sms,
            group=group,
            use_udma=use_udma,
            explicitly_destroy=True,
            rank=rank,
        )
        self.ctx = self.buffer._require_ctx()
        self.plan, self.cu_seqlens = allocate_planning_outputs(self.ctx)

        dev = f"npu:{local_rank}"
        generator = torch.Generator(device=dev).manual_seed(7777 + self.rank)
        self.hidden = torch.randn(
            S, H, dtype=torch.bfloat16, device=dev, generator=generator
        )
        self.weights = None
        if with_weights:
            self.weights = torch.rand(
                S, K, dtype=torch.float32, device=dev, generator=generator
            )

    def activate_symmetric_views(self):
        self.buffer._ensure_shmem_views()
        torch.npu.synchronize()
        ShmemRuntime.barrier(group=self.group)

    def prepare(self, topk, tokens_per_expert):
        """Run real planning once; no planning work is in ``dispatch``."""
        topk_flat = topk.reshape(-1).contiguous()
        launch_planning(
            self.ctx,
            topk_flat,
            tokens_per_expert,
            self.cu_seqlens,
            self.plan,
        )
        torch.npu.synchronize()
        ShmemRuntime.barrier(group=self.group)

    def dispatch(self):
        launch_dispatch(
            self.ctx,
            self.hidden,
            self.weights,
            self.plan,
            build_dedup_map=False,
        )

    def local_remote_traffic(self):
        """Decode this rank's plan.dst and count its inter-rank payload."""
        NvS = int(self.ctx["NvS"])
        encoded = self.plan.dst.to(torch.int64)
        canonical = encoded >= 0
        raw_dst = torch.where(canonical, encoded, -encoded - 1)
        dest_rank = torch.div(raw_dst, NvS, rounding_mode="floor")
        is_remote = dest_rank != self.rank

        remote_hidden_rows = (canonical & is_remote).sum(dtype=torch.int64)
        remote_weight_values = is_remote.sum(dtype=torch.int64)
        if not self.with_weights:
            remote_weight_values.zero_()
        return int(remote_hidden_rows.item()), int(remote_weight_values.item())

    def destroy(self):
        self.buffer.destroy()


class ShmemStats:
    """Gather one int32 stats row per rank without torch.distributed."""

    def __init__(self, rank, world_size, local_rank):
        self.rank = rank
        self.world_size = world_size
        self.device = f"npu:{local_rank}"
        self.source, _, _ = create_vmm_physical(
            [world_size, 3], torch.int32, local_rank
        )
        self.tensor = None

    def activate(self):
        self.tensor = create_sym_tensor_from_source(
            self.source.data_ptr(),
            [self.world_size, 3],
            torch.int32,
            self.device,
            "dispatch_benchmark_stats",
        )

    def collect(self, local_us, remote_hidden_rows, remote_weight_values):
        payload = torch.tensor(
            [round(local_us * 1000), remote_hidden_rows, remote_weight_values],
            dtype=torch.int32,
            device=self.device,
        )
        stream = torch.npu.current_stream().npu_stream
        sym_put_from(self.tensor, self.rank * 3 * 4, payload, 0, stream)
        torch.npu.synchronize()
        ShmemRuntime.barrier()

        if self.rank == 0:
            gathered = self.tensor.cpu()
            times = [value / 1000.0 for value in gathered[:, 0].tolist()]
            hidden_rows = int(gathered[:, 1].sum().item())
            weight_values = int(gathered[:, 2].sum().item())
        else:
            times = [local_us]
            hidden_rows = remote_hidden_rows
            weight_values = remote_weight_values
        ShmemRuntime.barrier()

        if self.rank == 0:
            print(
                f"  [time_op] per-rank: {[f'{value:.1f}' for value in times]} "
                f"max={max(times):.1f} mean={sum(times) / len(times):.1f} us",
                flush=True,
            )

        return {
            "per_rank_us": times,
            "mean_us": sum(times) / len(times),
            "max_us": max(times),
            "remote_hidden_rows": hidden_rows,
            "remote_weight_values": weight_values,
        }

    def destroy(self):
        self.tensor = None
        allocation = getattr(self.source, "_vmm_allocation", None)
        if allocation is not None:
            allocation.destroy()
        self.source = None


def setup_no_dist():
    """Read torchrun rank metadata without creating a process group."""
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(
            f"--no-dist requires torchrun environment variables: {missing}"
        )
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_device_index(local_rank))
    return rank, world_size, local_rank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_engine_args(parser)
    parser.add_argument(
        "--out",
        default="bench_dispatch_maxvio_results.csv",
        help="CSV output path on rank 0 (default: %(default)s)",
    )
    parser.add_argument(
        "--plot",
        default=None,
        help="Optional latency figure path on rank 0",
    )
    parser.add_argument(
        "--maxvios",
        default="0.2,1,10,20",
        help="Comma-separated target MaxVio values (default: %(default)s)",
    )
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num-sms", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--no-weights",
        action="store_true",
        help="Benchmark hidden-only dispatch instead of forward dispatch",
    )
    parser.add_argument(
        "--ip-port",
        default=None,
        help=(
            "SHMEM DEFAULT-bootstrap endpoint. Multi-node runs derive it from "
            "MASTER_ADDR and MASTER_PORT (+1) when omitted"
        ),
    )
    parser.add_argument(
        "--no-dist",
        action="store_true",
        help="Use torchrun only as a process launcher; bootstrap and statistics use SHMEM",
    )
    args = parser.parse_args()

    if args.no_dist:
        rank, world_size, local_rank = setup_no_dist()
        group = None
    else:
        rank, world_size, local_rank = setup()
        group = dist.group.WORLD
    local_rank = local_device_index(local_rank)
    use_udma = apply_engine(args.engine)
    R = world_size
    S, K, E, H = args.tokens, args.topk, args.experts, args.hidden
    dev = f"npu:{local_rank}"
    assert R >= 2, f"dispatch benchmark requires at least 2 ranks, got {R}"
    assert E % R == 0, f"E ({E}) must be divisible by world size ({R})"
    assert K <= E, f"K ({K}) must be <= E ({E})"
    targets = [float(value) for value in args.maxvios.split(",")]

    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    multi_node = world_size > local_world_size

    # Construct Buffer first so all VMM descriptors exist before SHMEM init.
    stats = ShmemStats(rank, R, local_rank) if args.no_dist else None
    runner = DispatchRunner(
        group,
        R,
        S,
        K,
        E,
        H,
        args.num_sms,
        local_rank,
        with_weights=not args.no_weights,
        rank=rank if args.no_dist else None,
        use_udma=use_udma,
    )
    if not ShmemRuntime.is_initialized():
        if args.no_dist:
            ip_port = args.ip_port or default_shmem_ip_port(
                os.environ.get("MASTER_ADDR", "127.0.0.1"),
                int(os.environ.get("MASTER_PORT", "29500")),
            )
            ShmemRuntime.init_with_ip_port(
                ip_port=ip_port,
                rank=rank,
                world_size=R,
                enable_udma=use_udma,
            )
        elif args.ip_port is not None or multi_node:
            ShmemRuntime.init_with_ip_port(group=group, ip_port=args.ip_port,
                                           enable_udma=use_udma)
        else:
            ShmemRuntime.init_with_buffer(group=group, enable_udma=use_udma)
    runner.activate_symmetric_views()
    if stats is not None:
        stats.activate()

    rows = []
    try:
        for target in targets:
            if args.no_dist:
                sigma, realized = solve_sigma_for_maxvio(
                    target, S, K, E, R, dev
                )
            elif rank == 0:
                sigma, realized = solve_sigma_for_maxvio(
                    target, S, K, E, R, dev
                )
            else:
                sigma, realized = None, None
            if not args.no_dist:
                solved = [sigma, realized]
                dist.broadcast_object_list(solved, src=0, group=group)
                sigma, realized = solved

            topk, tpe = generate_topk_routing(
                S, K, E, R, sigma, dev, 1234, rank=rank
            )
            if args.no_dist:
                maxvio = realized
            else:
                global_tpe = tpe.to(torch.int64)
                dist.all_reduce(global_tpe, group=group)
                maxvio = (
                    global_tpe.max().item()
                    / (global_tpe.sum().item() / E)
                    - 1.0
                )
            if rank == 0:
                note = (
                    ""
                    if abs(maxvio - target) <= max(0.05 * target, 0.01)
                    else f" (target {target} unreachable, floored)"
                )
                print(
                    f"--- target MaxVio={target}: sigma={sigma:.5f}, "
                    f"realized MaxVio={maxvio:.3f}{note}",
                    flush=True,
                )

            runner.prepare(topk, tpe)
            local_hidden_rows, local_weight_values = runner.local_remote_traffic()
            if not args.no_dist:
                counts = torch.tensor(
                    [local_hidden_rows, local_weight_values],
                    dtype=torch.int64,
                    device=dev,
                )
                dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=group)
                remote_hidden_rows, remote_weight_values = [
                    int(value) for value in counts.cpu()
                ]
            timing = time_op(
                runner.dispatch,
                group,
                rank,
                warmup=args.warmup,
                iters=args.iters,
                stats=stats,
                remote_counts=(local_hidden_rows, local_weight_values),
            )
            if args.no_dist:
                remote_hidden_rows = timing["remote_hidden_rows"]
                remote_weight_values = timing["remote_weight_values"]
            remote_bytes = (
                remote_hidden_rows * H * 2 + remote_weight_values * 4
            )

            logical_hidden_bytes = S * K * H * 2
            mean_us = timing["mean_us"]
            max_us = timing["max_us"]
            rows.append(
                {
                    "lib": "moonep",
                    "ep": R,
                    "S": S,
                    "E": E,
                    "H": H,
                    "K": K,
                    "with_weights": not args.no_weights,
                    "sigma": sigma,
                    "target_maxvio": target,
                    "maxvio": maxvio,
                    "dispatch_mean_us": mean_us,
                    "dispatch_max_us": max_us,
                    "logical_hidden_gbps": gbps(logical_hidden_bytes, mean_us),
                    "aggregate_remote_payload_gbps": gbps(remote_bytes, max_us),
                    "aggregate_remote_data_mb": remote_bytes / 1e6,
                    "remote_hidden_rows": remote_hidden_rows,
                    "remote_weight_values": remote_weight_values,
                    "per_rank_us": ";".join(
                        f"{value:.3f}" for value in timing["per_rank_us"]
                    ),
                }
            )
            if rank == 0:
                print(
                    f"{'moonep':>8} ep={R} S={S} E={E} H={H} K={K} "
                    f"sigma={sigma:.5f}: maxvio={maxvio:6.2f} | "
                    f"dispatch mean={mean_us:7.1f} max={max_us:7.1f} us | "
                    f"logical hidden={gbps(logical_hidden_bytes, mean_us):5.1f} "
                    f"GB/s | aggregate remote={gbps(remote_bytes, max_us):5.1f} "
                    f"GB/s ({remote_bytes / 1e6:.1f} MB)",
                    flush=True,
                )
    finally:
        if ShmemRuntime.is_initialized():
            ShmemRuntime.finalize()
        runner.destroy()
        if stats is not None:
            stats.destroy()
        reset_symmetric_descriptors()

    if rank == 0 and args.out:
        with open(args.out, "w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.out}")
    if rank == 0 and args.plot:
        plot_rows(rows, args.plot)
        print(f"wrote {args.plot}")

    if not args.no_dist:
        dist.destroy_process_group()


def plot_rows(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [f"{row['maxvio']:.2g}" for row in rows]
    xpos = np.arange(len(rows))
    means = [row["dispatch_mean_us"] for row in rows]
    maxima = [row["dispatch_max_us"] for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.36
    ax.bar(xpos - width / 2, means, width, label="rank mean")
    ax.bar(xpos + width / 2, maxima, width, label="slowest rank")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_xlabel("realized MaxVio")
    ax.set_ylabel("dispatch latency (us)")
    ax.set_title(
        f"MoonEP-Ascend dispatch, EP={rows[0]['ep']}, "
        f"S={rows[0]['S']}, K={rows[0]['K']}, H={rows[0]['H']}"
    )
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
