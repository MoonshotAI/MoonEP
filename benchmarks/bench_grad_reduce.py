"""
Remote expert grad reduce benchmark.

Run with:
    torchrun --nproc_per_node=8 benchmarks/bench_grad_reduce.py
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

from moonep import Buffer
from moonep.buffer import create_nvl_dist_tensor, pad_dim0_for_alignment
from moonep.grad_reduce import launch_grad_reduce


# Expert shape fixed at 3584x3072 (3583 padded up to a 128 multiple).
EXP_H = 3584
EXP_HP = 3072

# All cases: rank0-initiated, num_sms == 32. Each local expert is reduced
# a *different* number of times in 0..3 (R-1=3 is the max), via `counts[e]`.
# 0 means that expert is skipped. base_B > epn leaves idle columns so the
# slot layout is sparse. Default shape is 3584x3072; later cases vary the
# expert shape, epn and slot-column density.
NUM_SMS = 32
DEFAULT_CASES = [
    # small-slot cases: only a few experts reduced at all.
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
]


def setup():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    return rank, dist.get_world_size()


def expert_plan(R, B, epn, counts, dev):
    """rank0 owns experts 0..epn-1; expert e is reduced ``counts[e]`` times.

    For expert e the contributing remote ranks are 1..counts[e], each placing
    its grad in slot column e. All other slots stay idle (-1). counts[e] must
    be < R (only R-1 remote ranks exist). epn <= B so column e exists.
    """
    plan = torch.full((R, B), -1, dtype=torch.int32, device=dev)
    for e in range(epn):
        for src_rank in range(1, counts[e] + 1):
            plan[src_rank, e] = e  # owner rank0, local expert e
    return plan


def bench_case(case, args, rank, R):
    dev = "cuda"
    epn = int(case["epn"])
    E = R * epn
    H = int(case["H"])
    Hp = int(case["Hp"])
    B = pad_dim0_for_alignment([int(case["base_B"]), H, Hp], torch.float32)
    counts = case.get("counts") or [int(case.get("nremote", 1))] * epn
    num_sms = NUM_SMS

    assert H % 128 == 0 and Hp % 128 == 0, \
        f"{case['label']}: H and Hp must be multiples of 128"
    assert len(counts) == epn, f"{case['label']}: counts must have epn={epn} entries"
    assert all(0 <= c < R for c in counts), f"{case['label']}: each count must be in [0, R)"

    reduce_full = create_nvl_dist_tensor([B, H, Hp], torch.float32, rank, R)
    reduce_buffers = reduce_full.view(R, B, H, Hp)
    experts_to_copy = expert_plan(R, B, epn, counts, dev)

    # grad_reduce needs the cross-rank barrier slots from Buffer.
    buffer = Buffer(S=128, H=H, K=1, E=E, num_ep_ranks=R, num_sms=num_sms, B=B)
    ctx = buffer._require_ctx()

    gen = torch.Generator(device=dev).manual_seed(321 + rank)
    remote_expert_grads = torch.randn(
        E, H, Hp, dtype=torch.float32, device=dev, generator=gen
    )
    reduce_buffers[rank].copy_(torch.randn_like(reduce_buffers[rank]))
    torch.cuda.synchronize()
    dist.barrier(device_ids=[torch.cuda.current_device()])

    def reduce_once():
        launch_grad_reduce(
            remote_expert_grads,
            reduce_buffers,
            experts_to_copy,
            rank=rank,
            num_sms=num_sms,
            meta_buf=ctx['meta_buf'],
            meta_stride=ctx['meta_chunk_padded'],
            barrier_off=ctx['BARRIER_OFF'],
            grid_sync_bar=ctx['grid_sync_bar'],
        )

    # Warmup (also JIT-compiles + primes the cross-rank barrier) then capture
    # the iters loop into a single CUDA graph to strip launch/python overhead.
    for _ in range(args.warmup):
        reduce_once()
    torch.cuda.synchronize()
    dist.barrier(device_ids=[torch.cuda.current_device()])

    # --no-graph keeps plain stream launches so tools like NCU can intercept
    # each kernel (graph capture/replay hides launches from kernel filters).
    graph = None
    if not args.no_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.iters):
                reduce_once()
        torch.cuda.synchronize()
        dist.barrier(device_ids=[torch.cuda.current_device()])

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    if graph is not None:
        graph.replay()
    else:
        for _ in range(args.iters):
            reduce_once()
    end.record()
    end.synchronize()

    # Time rank0 only — it owns every expert here, so it is the worst rank and
    # the idle ranks just wait at the barrier.
    worst_us = start.elapsed_time(end) * 1e3 / args.iters if rank == 0 else 0.0

    # Critical-path traffic for the owner rank: read every consumed reduce
    # slot (4B/elem), then read+write the grad of each active expert (8B/elem).
    # Slot clears happen locally on each source rank in parallel, off the
    # owner's critical path, so they are not counted.
    slots = int(sum(counts))
    active = int(sum(1 for c in counts if c > 0))
    tile = H * Hp * 4
    bytes_per_rank = slots * tile + active * tile * 2
    bw_gbs = bytes_per_rank / worst_us * 1e6 / 1e9 if worst_us > 0 else 0.0
    # pure NVLink read traffic: only the remote reduce-slot reads (grad rw and
    # clears are local HBM, off the NVLink path).
    comm_gbs = slots * tile / worst_us * 1e6 / 1e9 if worst_us > 0 else 0.0

    dist.barrier(device_ids=[torch.cuda.current_device()])
    buffer.destroy()
    return worst_us, bytes_per_rank / 1e6, bw_gbs, comm_gbs, E, B, slots


def explicit_single_config_requested(argv):
    shape_flags = ("--epn", "--H", "--Hp", "--B", "--nremote")
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
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--no-graph", action="store_true",
                        help="Time plain launches instead of a CUDA graph (for NCU)")
    return parser.parse_args()


def main():
    args = parse_args()
    rank, R = setup()
    assert R >= 2, "bench_grad_reduce requires at least 2 GPUs"

    if args.single and args.suite:
        raise ValueError("Use only one of --single or --suite")

    run_single = args.single or (
        not args.suite and explicit_single_config_requested(sys.argv[1:])
    )
    if run_single:
        cases = [{
            "label": "custom",
            "epn": args.epn,
            "H": args.H,
            "Hp": args.Hp,
            "base_B": args.B,
            "nremote": args.nremote,
        }]
    else:
        cases = DEFAULT_CASES

    if rank == 0:
        print(
            f"MoonEP Grad Reduce Benchmark (R={R}, warmup={args.warmup}, "
            f"iters={args.iters})"
        )
        print(
            f"{'Config':<20} {'E':>5} {'B':>4} {'H':>7} {'Hp':>6} "
            f"{'SMs':>5} {'Slots':>6} {'Data(MB)':>10} {'Worst(us)':>10} {'BW(GB/s)':>9} {'CommBW':>8}"
        )
        print("-" * 101)

    for case in cases:
        worst_us, mb, bw_gbs, comm_gbs, E, B, slots = bench_case(case, args, rank, R)
        if rank == 0:
            print(
                f"{case['label']:<20} {E:>5} {B:>4} "
                f"{case['H']:>7} {case['Hp']:>6} {NUM_SMS:>5} {slots:>6} "
                f"{mb:>10.2f} {worst_us:>10.2f} {bw_gbs:>9.2f} {comm_gbs:>8.2f}"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
