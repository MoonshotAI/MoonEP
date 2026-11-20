"""
MoonEP-Ascend remote expert weight-grad reduce benchmark.

Ascend port of ``MoonEP/benchmarks/bench_grad_reduce.py``, driven through the
low-level ``launch_grad_reduce`` exactly like ``test/test_grad_reduce.py``.
On Ascend each rank owns a local ``[E, H, H']`` fp32 home-grad chunk and a
``[B, H, H']`` fp32 reduce-slot chunk (symmetric memory); duplicated experts'
grads flow back to their home ranks and the consumed slots are cleared.

The reported latency is the worst rank's (``agg=max``), matching the CUDA
reference's owner-rank timing. ``DEFAULT_CASES`` (labels + settings) are kept
byte-for-byte identical to the origin so results compare directly; the
production 8-card shape (H=7168, Hp=3072, E=896, B=112) is run via ``--single``.

Run:
    bash npu_patch/benchmarks/run_bench.sh bench_grad_reduce.py                # default suite
    bash npu_patch/benchmarks/run_bench.sh bench_grad_reduce.py "--single --epn 112 --H 7168 --Hp 3072 --B 112 --nremote 2"
"""

import argparse
import importlib
import os
import sys

import torch
import torch_npu
import torch.distributed as dist

from _bench_common import (
    add_common_args,
    apply_engine,
    setup,
    time_kernel,
    time_text,
)
from npu_patch._device import local_device_index
from npu_patch.buffer_c import (
    create_vmm_physical,
    reset_symmetric_descriptors,
    sym_tensor,
)
from npu_patch.launchers import launch_grad_reduce
from npu_patch.shmem_runtime import ShmemRuntime

NUM_SMS = 32
EXP_H = 3584
EXP_HP = 3072

DEFAULT_CASES = [
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
    {"label": "full_3x8", "epn": 8, "H": EXP_H, "Hp": EXP_HP, "base_B": 14,
     "counts": [3, 3, 3, 3, 3, 3, 3, 3]},
    {"label": "dense_B8", "epn": 8, "H": EXP_H, "Hp": EXP_HP, "base_B": 8,
     "counts": [2, 1, 3, 1, 2, 3, 1, 2]},
    {"label": "epn4_full",  "epn": 4, "H": EXP_H, "Hp": EXP_HP, "base_B": 14,
     "counts": [3, 3, 3, 3]},
    {"label": "epn16_mixed", "epn": 16, "H": EXP_H, "Hp": EXP_HP, "base_B": 16,
     "counts": [3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0]},
    {"label": "thin_7168x128", "epn": 8, "H": 7168, "Hp": 128, "base_B": 14,
     "counts": [3, 0, 2, 1, 3, 0, 2, 1]},
    {"label": "tall_1024x3072", "epn": 8, "H": 1024, "Hp": 3072, "base_B": 14,
     "counts": [3, 0, 2, 1, 3, 0, 2, 1]},
    {"label": "tiny_512x512", "epn": 8, "H": 512, "Hp": 512, "base_B": 14,
     "counts": [3, 3, 2, 3, 1, 0, 2, 3]},
]


def grad_reduce_plan(R, B, epn, counts, dev):
    """Expert e (owned by rank 0's ``epn`` share) is reduced ``counts[e]`` times.

    The contributing remote ranks are ``1..counts[e]``, each placing its grad in
    slot column e; all other slots stay idle (-1). Mirrors the CUDA reference.
    """
    plan = torch.full((R, B), -1, dtype=torch.int32, device=dev)
    for e in range(epn):
        for src_rank in range(1, counts[e] + 1):
            plan[src_rank, e] = e
    return plan


def explicit_single_config_requested(argv):
    flags = ("--epn", "--H", "--Hp", "--B", "--nremote")
    return any(arg == f or arg.startswith(f + "=") for arg in argv for f in flags)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--suite", action="store_true", help="Run the default case suite.")
    ap.add_argument("--single", action="store_true",
                    help="Run exactly the config given by --epn/--H/--Hp/--B/--nremote.")
    ap.add_argument("--epn", type=int, default=8)
    ap.add_argument("--H", type=int, default=EXP_H)
    ap.add_argument("--Hp", type=int, default=EXP_HP)
    ap.add_argument("--B", type=int, default=14)
    ap.add_argument("--nremote", type=int, default=2)
    return ap.parse_args()


def bench_case(case, args, rank, group, R, local_rank):
    dev = f"npu:{local_rank}"
    epn = int(case["epn"])
    E = R * epn
    H = int(case["H"])
    Hp = int(case["Hp"])
    B = int(case["base_B"])
    counts = case.get("counts") or [int(case.get("nremote", 1))] * epn
    num_sms = args.num_sms or NUM_SMS

    assert H % 128 == 0 and Hp % 128 == 0, \
        f"{case['label']}: H and Hp must be multiples of 128"
    assert len(counts) == epn, f"{case['label']}: counts must have epn={epn} entries"
    assert all(0 <= c < R for c in counts), \
        f"{case['label']}: each count must be in [0, R)"

    plan = grad_reduce_plan(R, B, epn, counts, dev)
    # Buffer-backed only: allocate/register VMM chunks; after init_with_buffer
    # they become the local symmetric tensors the kernel expects (symmetric base).
    expert_grads_src, _, _ = create_vmm_physical(
        [E, H, Hp], torch.float32, local_rank)
    reduce_buffers_src, _, _ = create_vmm_physical(
        [B, H, Hp], torch.float32, local_rank)
    expert_grads_src.zero_()
    reduce_buffers_src.zero_()
    torch.npu.synchronize()
    dist.barrier(group=group)
    if not ShmemRuntime.is_initialized():
        # This script calls `launch_grad_reduce` directly (no Buffer): the
        # engine policy comes from env set by apply_engine, and enable_udma is
        # passed explicitly to keep the symmetric heap and op choice aligned.
        ShmemRuntime.init_with_buffer(group=group, tail_bytes=64 * 1024 * 1024,
                                      enable_udma=args.use_udma)
    expert_grads = sym_tensor(expert_grads_src)
    reduce_buffers = sym_tensor(reduce_buffers_src)

    try:
        res = time_kernel(
            lambda: launch_grad_reduce(
                expert_grads, reduce_buffers, plan,
                rank=rank, num_sms=num_sms,
                meta_buf=None, meta_stride=0, barrier_off=0, grid_sync_bar=None),
            group, rank, warmup=args.warmup, iters=args.iters, agg="max",
            cudagraph=args.cudagraph)
    finally:
        # Re-init per case: finalize (tear down symmetric mappings) first, then
        # reverse the VMM allocations, then clear the descriptor registry.
        dist.barrier(group=group)
        if ShmemRuntime.is_initialized():
            ShmemRuntime.finalize()
        for t in (expert_grads_src, reduce_buffers_src):
            alloc = getattr(t, '_vmm_allocation', None)
            if alloc is not None:
                alloc.destroy()
        reset_symmetric_descriptors()
    return res, E, B


def main():
    args = parse_args()
    rank, world_size, local_rank = setup()
    local_rank = local_device_index(local_rank)
    args.use_udma = apply_engine(args.engine)
    R = args.ep_size if args.ep_size is not None else world_size
    assert R >= 2, "grad-reduce benchmark requires at least 2 ranks"
    assert R <= world_size, f"--ep-size {R} exceeds world size {world_size}"
    in_group = rank < R
    group = dist.new_group(ranks=list(range(R)))

    run_single = args.single or (not args.suite and explicit_single_config_requested(sys.argv[1:]))
    if run_single:
        cases = [{"label": "custom", "epn": args.epn, "H": args.H, "Hp": args.Hp,
                  "base_B": args.B, "nremote": args.nremote}]
    else:
        cases = DEFAULT_CASES
        # Suite cases need R > max(counts) (a count c needs contributing ranks 1..c).
        skipped = [c["label"] for c in cases
                   if max(c.get("counts") or [c.get("nremote", 1)] * c["epn"]) >= R]
        cases = [c for c in cases
                 if max(c.get("counts") or [c.get("nremote", 1)] * c["epn"]) < R]
        if rank == 0 and skipped:
            print(f"NOTE: skipping {len(skipped)} cases needing >{R} ranks: "
                  f"{', '.join(skipped)}", flush=True)

    if rank == 0:
        print(f"=== bench_grad_reduce: world={world_size} R={R} cases={len(cases)} "
              f"=== ", flush=True)

    dist.barrier()

    if rank == 0:
        print(f"{'Config':<20} {'E':>5} {'B':>4} {'H':>7} {'Hp':>6} {'Slots':>6} "
              f"{'Worst(us)':>10} {'Status':>8}")
        print("-" * 78)

    failures = 0
    for case in cases:
        res = None
        if in_group:
            try:
                res, E, B = bench_case(case, args, rank, group, R, local_rank)
            except Exception:
                import traceback
                traceback.print_exc()
                res = None
        dist.barrier()

        if rank == 0:
            if res is None:
                failures += 1
                print(f"{case['label']:<20} FAILED", flush=True)
                continue
            slots = sum(case.get("counts") or [case.get("nremote", 1)] * case["epn"])
            st = "ok" if res["status"] == "ok" else "stub"
            print(f"{case['label']:<20} {E:>5} {B:>4} {case['H']:>7} {case['Hp']:>6} "
                  f"{slots:>6} {time_text(res):>10} {st:>8}", flush=True)

    if in_group:
        ShmemRuntime.finalize()
    dist.destroy_process_group()
    if rank == 0 and failures:
        print(f"{failures}/{len(cases)} cases FAILED", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
