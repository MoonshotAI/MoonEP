"""
MoonEP-Ascend grad_reduce benchmark: MaxVio sweep.

Ascend port of ``MoonEP/benchmarks/bench_grad_reduce.py`` with the same CLI and
methodology, extended with a MaxVio sweep to measure grad_reduce performance
under different load balance conditions.

Reported ops:
  - gr: grad_reduce (1 projection, representative of all 3: gate/up/down)

Bandwidth = bytes_transferred / t.
Results are saved to a CSV on rank 0 (--out) for later plotting.

EP group = whole world (R == world_size). Single- and multi-node both work:

    Single node (default; UID bootstrap, no ip_port needed):
        torchrun --nproc_per_node=8 npu_patch/benchmarks/bench_grad_reduce_maxvio.py [--out r.csv]
    Multi node (ip_port/DEFAULT bootstrap; when ip_port is omitted it is
    derived from MASTER_ADDR/MASTER_PORT, port + 1):
        # node 0
        MASTER_ADDR=192.168.1.10 bash npu_patch/benchmarks/run_bench.sh bench_grad_reduce_maxvio.py   # NODES=2 NODE_RANK=0
        # node 1
        MASTER_ADDR=192.168.1.10 NODES=2 NODE_RANK=1 bash npu_patch/benchmarks/run_bench.sh bench_grad_reduce_maxvio.py
    Explicit shmem ip_port:
        torchrun --nnodes=2 --nproc-per-node=8 --node-rank=0 \
            --master-addr=192.168.1.10 --master-port=29500 \
            npu_patch/benchmarks/bench_grad_reduce_maxvio.py --ip-port tcp://192.168.1.10:29501
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
    setup,
    time_kernel,
)
from _bench_common import generate_topk_routing  # re-export

from npu_patch import Buffer, ShmemRuntime
from npu_patch._device import local_device_index
from npu_patch.buffer_c import (
    alloc_sym_chunk,
    reset_symmetric_descriptors,
    sym_tensor,
)
from npu_patch.launchers import (
    launch_grad_reduce,
    launch_planning,
)
from npu_patch.planning import allocate_planning_outputs


# ---------------------------------------------------------------------------
# Shared timing harness — eager npu-event timing, cross-rank mean us
# ---------------------------------------------------------------------------

def time_op(fn, group, rank, warmup=20, iters=50, cudagraph=True):
    """Time a kernel and print per-rank latency."""
    dev = torch.device(f"npu:{torch.npu.current_device()}")
    ws = dist.get_world_size(group=group)

    # Warmup
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    dist.barrier(group=group)

    # Measure
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    local_us = start.elapsed_time(end) / iters * 1e3

    # Gather per-rank times
    t = torch.tensor([local_us], dtype=torch.float64, device=dev)
    outs = [torch.empty(1, dtype=torch.float64, device=dev) for _ in range(ws)]
    dist.all_gather(outs, t, group=group)
    cat = torch.cat(outs)
    times = [o.item() for o in outs]

    if rank == 0:
        print(f"  [time_op] per-rank: {[f'{t:.1f}' for t in times]} "
              f"max={max(times):.1f} mean={sum(times)/len(times):.1f} us",
              flush=True)

    dist.barrier(group=group)
    return cat.mean().item()


def gbps(bytes_transferred, us):
    return bytes_transferred / (us * 1e-6) / 1e9 if us > 0 else 0.0


def _maxvio_for_sigma(sigma, S, K, E, R, dev):
    """Realized MaxVio of the lognormal routing at the given sigma, using the
    benchmark's fixed seeds (1234 shared, per-rank local) — deterministic."""
    total = torch.zeros(E, dtype=torch.int64, device=dev)
    for r in range(R):
        _, tpe = generate_topk_routing(S, K, E, R, sigma, dev, 1234, rank=r)
        total += tpe.to(torch.int64)
    return total.max().item() / (total.sum().item() / E) - 1.0


def solve_sigma_for_maxvio(target, S, K, E, R, dev):
    """Log-scale bisection for the lognormal sigma whose realized MaxVio best
    matches ``target``. Returns ``(sigma, realized_maxvio)``; if the target is
    below the uniform-sampling noise floor, returns the smallest sigma tried."""
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


# ---------------------------------------------------------------------------
# MoonEP grad_reduce runner
# ---------------------------------------------------------------------------

class GradReduceRunner:
    """Grad-reduce-only runner for benchmarking."""

    NAME = 'moonep'

    def __init__(self, group, R, S, K, E, H, num_sms, local_rank, hp=2048,
                 use_udma=False):
        self.group = group
        self.num_sms = num_sms
        self.rank = dist.get_rank(group)
        self.R = R
        self.local_rank = local_rank
        self.hp = hp
        self.E = E
        self.epn = E // R
        self.B = self.epn  # B == epn in MoonEP
        self.use_udma = use_udma

        # We need a minimal Buffer for the plan (experts_to_copy).
        # ShmemRuntime must already be initialized.
        self.buffer = Buffer(S, H, K, E, R, num_sms=num_sms, group=group,
                             use_udma=use_udma, explicitly_destroy=True)
        self.ctx = self.buffer._require_ctx()
        self.plan, self.cu_seqlens = allocate_planning_outputs(self.ctx)

        # Grad tensors for grad_reduce (1 projection, representative of all 3).
        # In SHMEM symmetric memory, each rank allocates only its own chunk:
        #   expert_grads: [epn, H, Hp] fp32 — this rank's expert grads
        #   reduce_buffers: [B, H, Hp] fp32 — this rank's reduce buffer
        self.expert_grads_src = alloc_sym_chunk(
            [self.epn, H, hp], torch.float32, local_rank)
        self.reduce_buffers_src = alloc_sym_chunk(
            [self.B, H, hp], torch.float32, local_rank)

        self.expert_grads_src.normal_()
        self.reduce_buffers_src.normal_()

        torch.npu.synchronize()
        dist.barrier(group=self.group)

    def activate_symmetric_views(self):
        self.buffer._ensure_shmem_views()
        self.expert_grads = sym_tensor(self.expert_grads_src)
        self.reduce_buffers = sym_tensor(self.reduce_buffers_src)
        torch.npu.synchronize()
        ShmemRuntime.barrier(group=self.group)

    def prepare(self, topk, tpe):
        """Materialize the plan (experts_to_copy) for grad_reduce."""
        self._topk_flat = topk.reshape(-1).contiguous()
        launch_planning(self.ctx, self._topk_flat, tpe, self.cu_seqlens,
                        self.plan)
        # Byte accounting from the plan's slot table (global expert ids).
        etc_cpu = self.plan.experts_to_copy.cpu()   # [R, B] int32, -1 = idle
        self.max_recv = int((etc_cpu >= 0).sum(dim=1).max().item())
        torch.npu.synchronize()
        ShmemRuntime.barrier(group=self.group)

    def grad_reduce(self):
        """Run grad_reduce for 1 projection."""
        launch_grad_reduce(
            self.expert_grads,
            self.reduce_buffers,
            self.plan.experts_to_copy,
            rank=self.rank,
            num_sms=self.num_sms,
            meta_buf=None,
            meta_stride=0,
            barrier_off=0,
            grid_sync_bar=None,
        )

    def destroy(self):
        for tensor in (self.expert_grads_src, self.reduce_buffers_src):
            allocation = getattr(tensor, '_vmm_allocation', None)
            if allocation is not None:
                allocation.destroy()
        self.buffer.destroy()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_engine_args(ap)
    ap.add_argument('--out', default='bench_grad_reduce_maxvio_results.csv',
                    help='CSV output path (rank 0). Default: %(default)s')
    ap.add_argument('--plot', default=None,
                    help='If set, save the figure to this path (rank 0 only).')
    ap.add_argument('--maxvios', default='0.2,1,10,20',
                    help='Comma-separated target MaxVio values; the lognormal '
                         'sigma is reverse-solved for each (default: %(default)s)')
    ap.add_argument('--experts', type=int, default=384)
    ap.add_argument('--hidden', type=int, default=7168)
    ap.add_argument('--topk', type=int, default=8)
    ap.add_argument('--hp', type=int, default=2048,
                    help='Expert weight inner dim H\' (MoE intermediate size) '
                         'for the MoonEP prefetch op (default: %(default)s)')
    ap.add_argument('--num-sms', type=int, default=32)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--iters', type=int, default=50)
    ap.add_argument('--ip-port', default=None,
                    help='shmem DEFAULT-bootstrap ip_port URL for the '
                         'buffer-backed shmem init ("tcp://<rank0_ip>:<port>" '
                         'or "tcp6://[ipv6]:<port>"). Required for multi-node; '
                         'when omitted in a multi-node launch it is derived '
                         'from MASTER_ADDR/MASTER_PORT (port + 1). Single-node '
                         'launches keep the UID bootstrap and ignore this.')
    ap.add_argument('--no-cudagraph', dest='cudagraph', action='store_false',
                    help='Use the eager timing loop instead of CUDA-graph '
                         'capture (CUDA-graph removes per-launch host overhead).')
    ap.set_defaults(cudagraph=True)
    args = ap.parse_args()

    rank, world_size, local_rank = setup()
    local_rank = local_device_index(local_rank)
    use_udma = apply_engine(args.engine)
    R = world_size
    assert R >= 2, f'bench_grad_reduce requires at least 2 NPUs, got {R}'
    group = dist.group.WORLD
    dev = f"npu:{local_rank}"

    S, E = 8192, args.experts
    H, K = args.hidden, args.topk
    targets = [float(x) for x in args.maxvios.split(',')]

    # Multi-node detection: under torchrun, WORLD_SIZE > LOCAL_WORLD_SIZE
    # (per-node process count) means multi-node.
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    multi_node = world_size > local_world_size

    runner = GradReduceRunner(group, R, S, K, E, H, args.num_sms, local_rank,
                              hp=args.hp, use_udma=use_udma)
    if not ShmemRuntime.is_initialized():
        if args.ip_port is not None or multi_node:
            # Multi-node: ip_port/DEFAULT bootstrap. When ip_port is omitted it
            # is derived from MASTER_ADDR/MASTER_PORT (port + 1; see
            # ShmemRuntime.init_with_ip_port).
            ShmemRuntime.init_with_ip_port(group=group, ip_port=args.ip_port,
                                           enable_udma=use_udma)
        else:
            ShmemRuntime.init_with_buffer(group=group, enable_udma=use_udma)
    runner.activate_symmetric_views()

    rows = []
    try:
        for target in targets:
            # Rank 0 reverse-solves the lognormal sigma for the target MaxVio
            # (deterministic given the fixed seeds), then broadcasts it.
            if rank == 0:
                sigma, realized = solve_sigma_for_maxvio(target, S, K, E, R, dev)
            else:
                sigma, realized = None, None
            obj = [sigma, realized]
            dist.broadcast_object_list(obj, src=0, group=group)
            sigma, realized = obj
            topk, tpe = generate_topk_routing(S, K, E, R, sigma, dev, 1234,
                                              rank=rank)
            global_tpe = tpe.to(torch.int64)
            dist.all_reduce(global_tpe, group=group)
            maxvio = global_tpe.max().item() / (global_tpe.sum().item() / E) - 1.0
            if rank == 0:
                note = '' if abs(maxvio - target) <= max(0.05 * target, 0.01) \
                    else f' (target {target} unreachable, floored)'
                print(f'--- target MaxVio={target}: sigma={sigma:.5f}, '
                      f'realized MaxVio={maxvio:.3f}{note}', flush=True)

            runner.prepare(topk, tpe)

            # Time grad_reduce
            gr_us = time_op(runner.grad_reduce, group, rank, args.warmup,
                            args.iters, cudagraph=args.cudagraph)

            # Bandwidth: 1 projection
            #   remote reads: max_recv reduce slots
            #   local read+write: epn expert grad rows
            gr_bytes = (runner.max_recv * H * runner.hp * 4 +   # remote reads
                        runner.epn * H * runner.hp * 4 * 2)     # local read+write
            gr_gbps = gbps(gr_bytes, gr_us)

            rows.append(dict(lib='moonep', ep=R, E=E, H=H, K=K, sigma=sigma,
                             target_maxvio=target, maxvio=maxvio,
                             gr_us=gr_us, gr_gbps=gr_gbps,
                             slots=runner.max_recv, active=runner.epn,
                             data_mb=gr_bytes / 1e6))
            if rank == 0:
                print(f'{"moonep":>8} ep={R} E={E} H={H} K={K} sigma={sigma:.5f}: '
                      f'maxvio={maxvio:6.2f} | '
                      f'gr {gr_us:7.1f} us ({gr_gbps:5.0f} GB/s)', flush=True)
    finally:
        # Finalize (tear down symmetric mappings) before reversing the VMM
        # allocations; clear the registry last. In finally so a failure on any
        # target still frees shmem/VMM resources.
        if ShmemRuntime.is_initialized():
            ShmemRuntime.finalize()
        runner.destroy()
        reset_symmetric_descriptors()

    if rank == 0 and args.out:
        with open(args.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'wrote {args.out}')
    if rank == 0 and args.plot:
        plot_rows(rows, args.plot)
        print(f'wrote {args.plot}')

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Plotting (inline): single-bar figure
# ---------------------------------------------------------------------------

_COLORS = {'comm': '#2563eb'}


def plot_rows(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    xs = sorted({r['maxvio'] for r in rows})
    xpos = np.arange(len(xs))
    labels = [f'{x:g}' for x in xs]
    C = _COLORS

    def get(op, x):
        return next(r[f'{op}_us'] for r in rows if r['maxvio'] == x)

    fig, ax = plt.subplots(figsize=(9, 5))

    ys = [get('gr', x) for x in xs]
    bars = ax.bar(xpos, ys, 0.5, color=C['comm'])
    ax.bar_label(bars, fmt='%.0f', fontsize=7.5, padding=2)

    ax.set_title('grad_reduce', fontsize=11)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylabel('latency (us)')
    ax.set_xlabel('MaxVio (0 = balanced, k = hottest expert at (k+1)x load)')

    E = rows[0]['E']
    H = rows[0]['H']
    K = rows[0]['K']
    fig.suptitle(f'MoonEP-Ascend grad_reduce — ep=8, E={E}, H={H}, K={K}, S=8192/rank',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == '__main__':
    main()
