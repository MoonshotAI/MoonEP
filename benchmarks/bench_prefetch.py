"""
Remote expert prefetch benchmark.

Run with:
    torchrun --nproc_per_node=8 benchmarks/bench_prefetch.py
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

from moonep.buffer import create_nvl_single_owner_tensor, pad_dim0_for_alignment
from moonep.prefetch import launch_prefetch


# Expert shape fixed at 3584x3072 (3583 padded up to a 128 multiple).
EXP_H = 3584
EXP_HP = 3072

K3_BF16_LAYER = [
    {"transposed": True},                 # gate  -> (I, H)
    {"transposed": True},                 # up    -> (I, H)
    {},                                   # down  -> (H, I)
]
K3_MXFP4_LAYER = [
    {"transposed": True, "pack": 2},      # gate packed -> (I, H/2)
    {"transposed": True, "pack": 2},      # up packed   -> (I, H/2)
    {"pack": 2},                          # down packed -> (H, I/2)
    {"scale": True},                      # gate scale  -> (128, 2688)
    {"scale": True},                      # up scale
    {"scale": True},                      # down scale
]

_DT_LABEL = {
    torch.bfloat16: "bf16",
    torch.int8: "i8",
    torch.uint8: "u8",
}
_DT_LABEL_TO_TORCH = {v: k for k, v in _DT_LABEL.items()}

# All cases: rank0-initiated, num_sms == 32. Each remote expert is prefetched
# a *different* number of times in 0..3 (R-1=3 is the max), via `counts[e]`.
# 0 means that expert is skipped. base_B > epn leaves idle columns so the
# slot layout is sparse. Default shape is 3584x3072; later cases vary the
# expert shape, epn and slot-column density.
NUM_SMS = 32
DEFAULT_CASES = [
    # small-slot cases: only a few experts prefetched at all.
    {"label": "slots_1",  "epn": 8, "H": EXP_H, "Hp": EXP_HP, "base_B": 14,
     "counts": [1, 0, 0, 0, 0, 0, 0, 0]},
    {"label": "slots_2",  "epn": 8, "H": EXP_H, "Hp": EXP_HP, "base_B": 14,
     "counts": [1, 1, 0, 0, 0, 0, 0, 0]},
    {"label": "slots_3",  "epn": 8, "H": EXP_H, "Hp": EXP_HP, "base_B": 14,
     "counts": [2, 1, 0, 0, 0, 0, 0, 0]},
    {"label": "slots_5",  "epn": 8, "H": EXP_H, "Hp": EXP_HP, "base_B": 14,
     "counts": [3, 2, 0, 0, 0, 0, 0, 0]},
    {"label": "ramp_0_3", "epn": 8, "H": EXP_H, "Hp": EXP_HP, "base_B": 14,
     "counts": [0, 1, 2, 3, 0, 1, 2, 3]},
    {"label": "mixed",    "epn": 8, "H": EXP_H, "Hp": EXP_HP, "base_B": 14,
     "counts": [3, 0, 2, 1, 3, 0, 2, 1]},
    {"label": "heavy",    "epn": 8, "H": EXP_H, "Hp": EXP_HP, "base_B": 14,
     "counts": [3, 3, 2, 3, 1, 0, 2, 3]},
    # saturated: every expert contributed by every remote rank (24 slots).
    {"label": "full_3x8", "epn": 8, "H": EXP_H, "Hp": EXP_HP, "base_B": 14,
     "counts": [3, 3, 3, 3, 3, 3, 3, 3]},
    # dense slot columns: base_B == epn, no idle columns.
    {"label": "dense_B8", "epn": 8, "H": EXP_H, "Hp": EXP_HP, "base_B": 8,
     "counts": [2, 1, 3, 1, 2, 3, 1, 2]},
    # fewer / more experts per rank at the same shape.
    {"label": "epn4_full",  "epn": 4, "H": EXP_H, "Hp": EXP_HP, "base_B": 14,
     "counts": [3, 3, 3, 3]},
    {"label": "epn16_mixed", "epn": 16, "H": EXP_H, "Hp": EXP_HP, "base_B": 16,
     "counts": [3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0]},
    # different expert shapes (same mixed/heavy count patterns).
    {"label": "thin_7168x128", "epn": 8, "H": 7168, "Hp": 128, "base_B": 14,
     "counts": [3, 0, 2, 1, 3, 0, 2, 1]},
    {"label": "tall_1024x3072", "epn": 8, "H": 1024, "Hp": 3072, "base_B": 14,
     "counts": [3, 0, 2, 1, 3, 0, 2, 1]},
    {"label": "tiny_512x512", "epn": 8, "H": 512, "Hp": 512, "base_B": 14,
     "counts": [3, 3, 2, 3, 1, 0, 2, 3]},
    # --- Quantized experts -------------------------------------------------
    {"label": "mxfp4_gate_up", "epn": 8, "H": EXP_H, "Hp": EXP_HP,
     "transposed": True, "pack": 2,
     "dtype": torch.uint8, "base_B": 14, "counts": [3, 0, 2, 1, 3, 0, 2, 1]},
    {"label": "mxfp4_down", "epn": 8, "H": EXP_H, "Hp": EXP_HP, "pack": 2,
     "dtype": torch.uint8, "base_B": 14, "counts": [3, 0, 2, 1, 3, 0, 2, 1]},
    {"label": "mxfp4_scale", "epn": 8, "H": EXP_H, "Hp": EXP_HP, "scale": True,
     "dtype": torch.uint8, "base_B": 14, "counts": [3, 0, 2, 1, 3, 0, 2, 1]},
    {"label": "k3_layer_bf16", "epn": 8, "H": EXP_H, "Hp": EXP_HP, "base_B": 14,
     "parts": K3_BF16_LAYER, "counts": [3, 0, 2, 1, 3, 0, 2, 1]},
    {"label": "k3_layer_mxfp4", "epn": 8, "H": EXP_H, "Hp": EXP_HP, "base_B": 14,
     "dtype": torch.uint8,
     "parts": K3_MXFP4_LAYER, "counts": [3, 0, 2, 1, 3, 0, 2, 1]},
]


def derive_extents(H, Hp, part):
    if part.get("scale"):
        # One ue8m0 byte per 32 values, re-cut into whole 128x128 tiles.
        nbytes = H * Hp // 32
        return 128, nbytes // 128
    out, contracted = (Hp, H) if part.get("transposed") else (H, Hp)
    return out, contracted // int(part.get("pack", 1))


def resolve_parts(case):
    H, Hp = int(case["H"]), int(case["Hp"])
    default_dt = case.get("dtype", torch.bfloat16)
    resolved = []
    for part in case.get("parts") or [case]:
        th, thp = derive_extents(H, Hp, part)
        resolved.append((th, thp, part.get("dtype", default_dt)))
    return resolved


def fill_random(shape, dtype, gen, dev="cuda"):
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=dev, generator=gen)
    info = torch.iinfo(dtype)
    return torch.randint(info.min, info.max, shape, dtype=dtype, device=dev,
                         generator=gen)


def setup():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    return rank, dist.get_world_size()


def expert_plan(R, B, epn, counts, dev):
    """rank0 prefetches; expert column e is fetched ``counts[e]`` times.

    The plan mirrors grad_reduce's (R, B) slot grid: for expert e the
    contributing source rows are 1..counts[e], each placing a distinct remote
    expert id (src_rank-1)*epn + e in slot column e. All other slots stay
    idle (-1). counts[e] must be < R so every id fits in the remote table.
    epn <= B so column e exists. The grid flattens to the R*B prefetch slots.
    """
    plan = torch.full((R, B), -1, dtype=torch.int32, device=dev)
    for e in range(epn):
        for src_rank in range(1, counts[e] + 1):
            plan[src_rank, e] = (src_rank - 1) * epn + e
    return plan


def bench_case(case, args, rank, R):
    dev = "cuda"
    epn = int(case["epn"])
    E = R * epn
    parts = resolve_parts(case)
    th0, thp0, dt0 = parts[0]
    B = pad_dim0_for_alignment([int(case["base_B"]), th0, thp0], dt0)
    counts = case.get("counts") or [int(case.get("nremote", 1))] * epn
    num_sms = NUM_SMS

    for th, thp, _dt in parts:
        assert th % 128 == 0 and thp % 128 == 0, \
            f"{case['label']}: derived extents must be multiples of 128, " \
            f"got ({th}, {thp})"
    assert len(counts) == epn, f"{case['label']}: counts must have epn={epn} entries"
    assert all(0 <= c < R for c in counts), f"{case['label']}: each count must be in [0, R)"

    # The remote expert table lives on one owner; rank0 reads it over NVLink
    owner_rank = int(args.owner_rank)
    remote_experts, prefetch_buffers = [], []
    for i, (th, thp, dt) in enumerate(parts):
        padded_E = pad_dim0_for_alignment([E, th, thp], dt)
        mapped = create_nvl_single_owner_tensor(
            [padded_E, th, thp], dt, owner_rank=owner_rank, local_rank=rank
        )
        remote = mapped[:E]
        if rank == owner_rank:
            gen = torch.Generator(device=dev).manual_seed(321 + rank + i)
            remote.copy_(fill_random((E, th, thp), dt, gen, dev))
        remote_experts.append(remote)
        prefetch_buffers.append(
            torch.empty(R * B, th, thp, dtype=dt, device=dev)
        )

    plan = expert_plan(R, B, epn, counts, dev)
    experts_to_copy = plan.flatten() if rank == 0 else \
        torch.full((R * B,), -1, dtype=torch.int32, device=dev)
    torch.cuda.synchronize()
    dist.barrier(device_ids=[torch.cuda.current_device()])

    def prefetch_once():
        for remote, buf in zip(remote_experts, prefetch_buffers):
            launch_prefetch(
                remote,
                buf,
                experts_to_copy,
                num_sms=num_sms,
            )

    # Warmup (also JIT-compiles the kernel) then capture the iters loop into a
    # single CUDA graph to strip launch/python overhead.
    for _ in range(args.warmup):
        prefetch_once()
    torch.cuda.synchronize()
    dist.barrier(device_ids=[torch.cuda.current_device()])

    # --no-graph keeps plain stream launches so tools like NCU can intercept
    # each kernel (graph capture/replay hides launches from kernel filters).
    graph = None
    if not args.no_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.iters):
                prefetch_once()
        torch.cuda.synchronize()
        dist.barrier(device_ids=[torch.cuda.current_device()])

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    if graph is not None:
        graph.replay()
    else:
        for _ in range(args.iters):
            prefetch_once()
    end.record()
    end.synchronize()

    # Time rank0 only — it issues every prefetch here, so it is the worst rank
    # and the idle ranks just run empty plans.
    worst_us = start.elapsed_time(end) * 1e3 / args.iters if rank == 0 else 0.0

    # Critical-path traffic for the prefetching rank: read every consumed slot
    # from the remote table over NVLink, then write it to the local prefetch
    # buffer.
    slots = int(sum(counts))
    per_slot = sum(th * thp * dt.itemsize for th, thp, dt in parts)
    bytes_per_rank = slots * per_slot * 2
    bw_gbs = bytes_per_rank / worst_us * 1e6 / 1e9 if worst_us > 0 else 0.0
    # pure NVLink read traffic: only the remote expert-table reads (buffer
    # writes are local HBM, off the NVLink path).
    comm_gbs = slots * per_slot / worst_us * 1e6 / 1e9 if worst_us > 0 else 0.0

    dist.barrier(device_ids=[torch.cuda.current_device()])
    dt_label = (_DT_LABEL.get(dt0, str(dt0))
                if len({dt for _, _, dt in parts}) == 1 else "mix")
    return (worst_us, bytes_per_rank / 1e6, bw_gbs, comm_gbs, E, B, slots,
            len(parts), dt_label, int(case["H"]), int(case["Hp"]))


def explicit_single_config_requested(argv):
    shape_flags = ("--epn", "--H", "--Hp", "--B", "--nremote", "--dtype")
    for arg in argv:
        for flag in shape_flags:
            if arg == flag or arg.startswith(flag + "="):
                return True
    return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", action="store_true",
                        help="Run the default multi-case benchmark suite")
    parser.add_argument("--single", action="store_true",
                        help="Run exactly the config described by --epn/--H/--Hp/--B/--nremote")
    parser.add_argument("--epn", type=int, default=8)
    parser.add_argument("--H", type=int, default=3584)
    parser.add_argument("--Hp", type=int, default=3072)
    parser.add_argument("--B", type=int, default=14)
    parser.add_argument("--nremote", type=int, default=2)
    parser.add_argument("--dtype", choices=sorted(_DT_LABEL_TO_TORCH),
                        default="bf16",
                        help="Element type of the prefetched tensor "
                             "(u8 = packed MXFP4 or its ue8m0 scales)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--owner-rank", type=int, default=1,
                        help="Rank whose GPU physically holds the remote expert "
                             "rank0 is always the reader. An owner on rank0's own node "
                             "measures intra-node NVLink, one on another node "
                             "measures cross-node MNNVL fabric.")
    parser.add_argument("--no-graph", action="store_true",
                        help="Time plain launches instead of a CUDA graph (for NCU)")
    return parser.parse_args()


def main():
    args = parse_args()
    rank, R = setup()
    assert R >= 2, "bench_prefetch requires at least 2 GPUs"

    if args.single and args.suite:
        raise ValueError("Use only one of --single or --suite")
    if not 0 <= args.owner_rank < R:
        raise ValueError(
            f"--owner-rank must be in [0, {R}), got {args.owner_rank}"
        )

    run_single = args.single or (
        not args.suite and explicit_single_config_requested(sys.argv[1:])
    )
    if run_single:
        cases = [{
            "label": "custom",
            "epn": args.epn,
            "H": args.H,
            "Hp": args.Hp,
            "dtype": _DT_LABEL_TO_TORCH[args.dtype],
            "base_B": args.B,
            "nremote": args.nremote,
        }]
    else:
        cases = DEFAULT_CASES

    if rank == 0:
        print(
            f"MoonEP Prefetch Benchmark (R={R}, warmup={args.warmup}, "
            f"iters={args.iters}, reader=rank0, owner=rank{args.owner_rank}, "
            f"num_sms={NUM_SMS})"
        )
        print(
            f"{'Config':<20} {'E':>5} {'B':>4} {'dt':>5} {'N':>3} {'H':>7} {'Hp':>6} "
            f"{'SMs':>5} {'Slots':>6} {'Data(MB)':>10} {'Worst(us)':>10} {'BW(GB/s)':>9} {'CommBW':>8}"
        )
        print("-" * 111)

    for case in cases:
        (worst_us, mb, bw_gbs, comm_gbs, E, B, slots,
         ntensor, dt_label, H, Hp) = bench_case(case, args, rank, R)
        if rank == 0:
            print(
                f"{case['label']:<20} {E:>5} {B:>4} {dt_label:>5} {ntensor:>3} "
                f"{H:>7} {Hp:>6} {NUM_SMS:>5} {slots:>6} "
                f"{mb:>10.2f} {worst_us:>10.2f} {bw_gbs:>9.2f} {comm_gbs:>8.2f}"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
