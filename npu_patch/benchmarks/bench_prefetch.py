"""
MoonEP-Ascend remote expert weight prefetch benchmark.

Ascend port of ``MoonEP/benchmarks/bench_prefetch.py``, driven through the
low-level ``launch_prefetch`` exactly like ``test/test_prefetch.py``. Each rank
owns a per-rank symmetric chunk ``[epn + B, H, H']`` bf16; ``remote_expert`` is
the logical ``[E, H, H']`` view over all ranks' home rows and the local
``[B, H, H']`` slice is the prefetch-slot buffer. The prefetch pulls the expert
ids selected by this rank's ``experts_to_copy`` row.

The reported latency is the worst rank's (``agg=max``). ``DEFAULT_CASES``
(labels + settings) are kept byte-for-byte identical to the origin so results
compare directly; the production 8-card shape (H=7168, Hp=3072, E=896, B=112)
is run via ``--single``.

Run:
    bash npu_patch/benchmarks/run_bench.sh bench_prefetch.py                # default suite
    bash npu_patch/benchmarks/run_bench.sh bench_prefetch.py "--single --epn 112 --H 7168 --Hp 3072 --B 112 --nremote 2"
"""

import argparse
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
from npu_patch import ShmemRuntime, launch_prefetch
from npu_patch.buffer_c import (
    create_sym_tensor_from_ptr,
    create_vmm_physical,
    map_to_sym_ptr,
    reset_symmetric_descriptors,
)

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


def make_symmetric_experts(rank, R, E, H, Hp, B):
    """Allocate the per-rank [epn + B, H, Hp] VMM chunk registered with the
    buffer-backed init. Returns the source tensor (``data_ptr`` = source VA);
    kernel args are mapped to local symmetric tensors via ``map_to_sym_ptr``
    in ``bench_case``. ``rank`` is only used for shape math: the VMM physical
    allocation must land on the physical card of the current logical device
    (in multi-node the global rank is not a local card id), hence
    ``torch.npu.current_device()``.
    """
    epn = (E + R - 1) // R
    return create_vmm_physical(
        [epn + B, H, Hp], torch.bfloat16,
        torch.npu.current_device())[0]


def prefetch_plan(R, B, epn, counts, dev):
    """Rank r pulls remote expert ``(r-1)*epn + e`` for column e when r <= counts[e].

    Mirrors the CUDA reference's slot grid: idle slots stay -1. ``counts[e]``
    must be < R so every id fits in the remote table; ``epn <= B`` so column e
    exists.
    """
    plan = torch.full((R, B), -1, dtype=torch.int32, device=dev)
    for e in range(epn):
        for src_rank in range(1, counts[e] + 1):
            plan[src_rank, e] = (src_rank - 1) * epn + e
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


def bench_case(case, args, rank, group, R):
    dev = f"npu:{torch.npu.current_device()}"
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

    keepalive = make_symmetric_experts(rank, R, E, H, Hp, B)
    local_expert = keepalive[:epn]
    prefetch_src = keepalive[epn:]
    local_expert.zero_()
    prefetch_src.zero_()
    experts_to_copy = prefetch_plan(R, B, epn, counts, dev)[rank].contiguous()
    torch.npu.synchronize()
    dist.barrier(group=group)

    # Buffer-backed only: map to symmetric tensors after init_with_buffer (EP
    # group). This script calls launch_prefetch directly (not via Buffer);
    # apply_engine writes the engine policy into env and enable_udma is passed
    # through explicitly, so the symmetric heap and the op select the same
    # engine.
    if not ShmemRuntime.is_initialized():
        ShmemRuntime.init_with_buffer(group=group, enable_udma=args.use_udma)
    sym_base = map_to_sym_ptr(keepalive.data_ptr(), 'prefetch_chunk')
    row_bytes = H * Hp * 2
    remote_expert = create_sym_tensor_from_ptr(
        sym_base, [E, H, Hp], torch.bfloat16, dev)
    prefetch_buffers = create_sym_tensor_from_ptr(
        sym_base + epn * row_bytes, [B, H, Hp], torch.bfloat16, dev)

    try:
        res = time_kernel(
            lambda: launch_prefetch(
                remote_expert, prefetch_buffers, experts_to_copy, num_sms=num_sms),
            group, rank, warmup=args.warmup, iters=args.iters, agg="max",
            cudagraph=args.cudagraph)
    finally:
        # Re-init per case: finalize first (drop symmetric mappings so shmem no
        # longer references the chunk VA), then the inverse VMM ops
        # (unmap/release/free), then clear the registry. Reversing the order
        # makes the re-init's get_buffer_ptr miss the newly registered chunk.
        if ShmemRuntime.is_initialized():
            ShmemRuntime.finalize()
        alloc = getattr(keepalive, '_vmm_allocation', None)
        if alloc is not None:
            alloc.destroy()
        reset_symmetric_descriptors()
    return res, E, B


def main():
    args = parse_args()
    rank, world_size, _ = setup()
    args.use_udma = apply_engine(args.engine)
    R = args.ep_size if args.ep_size is not None else world_size
    assert R >= 2, "prefetch benchmark requires at least 2 ranks (a source and a sink)"
    assert R <= world_size, f"--ep-size {R} exceeds world size {world_size}"
    in_group = rank < R
    group = dist.new_group(ranks=list(range(R)))

    run_single = args.single or (not args.suite and explicit_single_config_requested(sys.argv[1:]))
    if run_single:
        cases = [{"label": "custom", "epn": args.epn, "H": args.H, "Hp": args.Hp,
                  "base_B": args.B, "nremote": args.nremote}]
    else:
        cases = DEFAULT_CASES
        skipped = [c["label"] for c in cases
                   if max(c.get("counts") or [c.get("nremote", 1)] * c["epn"]) >= R]
        cases = [c for c in cases
                 if max(c.get("counts") or [c.get("nremote", 1)] * c["epn"]) < R]
        if rank == 0 and skipped:
            print(f"NOTE: skipping {len(skipped)} cases needing >{R} ranks: "
                  f"{', '.join(skipped)}", flush=True)

    if rank == 0:
        print(f"=== bench_prefetch: world={world_size} R={R} cases={len(cases)} "
              f"=== ", flush=True)

    # shmem init lives inside each case's bench_case (buffer-backed only).
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
                res, E, B = bench_case(case, args, rank, group, R)
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

    dist.destroy_process_group()
    if rank == 0 and failures:
        print(f"{failures}/{len(cases)} cases FAILED", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
