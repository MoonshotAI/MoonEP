"""
MoonEP-Ascend remote expert weight-grad reduce benchmark (canonical targets).

Matches ``bench_grad_reduce.py``'s cases and timing, driven through the
low-level ``launch_grad_reduce_targets`` instead of ``launch_grad_reduce``.
On Ascend each rank owns local fp32 ``[H, H']`` target tensors (ordinary memory,
handed to the op as a pointer table) and a ``[B, H, H']`` fp32 reduce-slot chunk
(symmetric memory); duplicated experts' grads are reduced into those targets and
the consumed slots are cleared. This variant passes ``epn`` local target
addresses. The unchanged dense reference currently allocates ``E = R * epn``
home rows; matching case fields does not remove that workload difference.
``epn`` counts local targets; ``B`` is independent slot capacity. Both the
default suite's B values and the ``--B`` default (14) match the dense benchmark.

The reported latency is the worst rank's (``agg=max``), matching the CUDA
reference's owner-rank timing. ``DEFAULT_CASES`` (labels + settings) are kept
identical to ``bench_grad_reduce.py``, including slot capacity and contribution
counts; the production 8-card shape (H=7168, Hp=3072, E=896, B=112) is run via
``--single``.

Run:
    bash npu_patch/benchmarks/run_bench.sh bench_grad_reduce_targets.py                # default suite
    bash npu_patch/benchmarks/run_bench.sh bench_grad_reduce_targets.py "--single --epn 112 --H 7168 --Hp 3072 --B 112 --nremote 2"
"""

import argparse
import csv
import os
import sys

import torch
import torch_npu
import torch.distributed as dist

if __package__:
    from ._bench_common import (
        add_common_args,
        apply_engine,
        setup,
        time_kernel,
        time_text,
    )
else:
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
from npu_patch.launchers import launch_grad_reduce_targets
from npu_patch.shmem_runtime import ShmemRuntime

NUM_SMS = 32
EXP_H = 3584
EXP_HP = 3072

# Match every dense case field, including B: pointer targets do not require
# one slot per local expert. The two scripts use the same contribution grid.
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


def check_payload(checks, group):
    passed = torch.stack(checks).all().to(torch.int32)
    dist.all_reduce(passed, op=dist.ReduceOp.MIN, group=group)
    if not bool(passed.item()):
        raise RuntimeError("collective payload/reference or slot-clear check failed")


def explicit_single_config_requested(argv):
    flags = ("--epn", "--H", "--Hp", "--B", "--nremote")
    return any(arg == f or arg.startswith(f + "=") for arg in argv for f in flags)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--suite", action="store_true", help="Run the default case suite.")
    ap.add_argument("--single", action="store_true",
                    help="Run exactly the config given by --epn/--H/--Hp/--B/--nremote.")
    ap.add_argument("--epn", type=int, default=8)
    ap.add_argument("--H", type=int, default=EXP_H)
    ap.add_argument("--Hp", type=int, default=EXP_HP)
    ap.add_argument("--B", type=int, default=14,
                    help="Reduction slot capacity, independent of epn (default: 14).")
    ap.add_argument("--nremote", type=int, default=2)
    ap.add_argument("--ip-port", help="SHMEM bootstrap URL for multi-node runs")
    args = ap.parse_args(argv)
    # This benchmark places expert e in column e, matching the dense plan
    # builder. Extra B columns remain idle; the operator itself accepts plans
    # with arbitrary mappings between slots and local experts.
    if min(args.epn, args.B, args.H, args.Hp, args.iters) <= 0 or args.warmup < 0 or args.nremote < 0:
        ap.error("dimensions/iters must be positive; warmup/nremote must be nonnegative")
    if args.num_sms is not None and not 8 <= args.num_sms <= 56:
        ap.error("num-sms must be in 8..56")
    return args


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
    # Canonical targets: this rank's ``epn`` local fp32 ``[H, H']`` tensors,
    # handed to the op as a pointer table (ordinary memory, not symmetric).
    targets = tuple(torch.empty((H, Hp), dtype=torch.float32, device=dev)
                    for _ in range(epn))
    target_ptrs = torch.tensor([t.data_ptr() for t in targets],
                               dtype=torch.int64, device=dev)
    # Buffer-backed only: allocate/register the VMM slot chunk; after
    # init_with_buffer it becomes the local symmetric tensor the kernel expects
    # (symmetric base).
    slots_src, _, _ = create_vmm_physical([B, H, Hp], torch.float32, local_rank)
    slots_src.zero_()
    workspace = (torch.empty(512 * 1024 * max(2 * min(num_sms, 64), R * B),
                             dtype=torch.float32, device=dev)
                 if args.use_udma else None)
    torch.npu.synchronize()
    dist.barrier(group=group)
    if not ShmemRuntime.is_initialized():
        # This script calls `launch_grad_reduce_targets` directly (no Buffer):
        # the engine policy comes from env set by apply_engine, and enable_udma
        # is passed explicitly to keep the symmetric heap and op choice aligned.
        ShmemRuntime.init_with_buffer(group=group, tail_bytes=64 * 1024 * 1024,
                                      enable_udma=args.use_udma,
                                      ip_port=args.ip_port)
    slots = sym_tensor(slots_src)
    row = plan[rank].tolist()

    def fill_inputs():
        for expert, target in enumerate(targets):
            target.fill_(3 + expert)
        slots.fill_(-17)
        for slot, expert in enumerate(row):
            if expert >= 0:
                slots[slot].fill_((rank + 1) * (slot + 1))

    def submit():
        launch_grad_reduce_targets(target_ptrs, slots, plan, rank=rank,
                                   num_sms=num_sms, udma_workspace=workspace)

    try:
        # Verify the payload / slot-clear contract once, on a single application.
        fill_inputs()
        torch.npu.synchronize()
        dist.barrier(group=group)
        submit()
        torch.npu.synchronize()
        checks = []
        for expert, target in enumerate(targets):
            contribution = (sum((peer + 1) * (expert + 1)
                                for peer in range(1, counts[expert] + 1))
                            if rank == 0 else 0)
            checks.append(torch.all(target == 3 + expert + contribution))
        for slot, expert in enumerate(row):
            checks.append(torch.all(slots[slot] == (0 if expert >= 0 else -17)))
        check_payload(checks, group)

        # The nonzero preflight above is outside timing. Reset both operands
        # to match the dense benchmark's zero-input timed loop.
        for target in targets:
            target.zero_()
        slots_src.zero_()
        torch.npu.synchronize()
        dist.barrier(group=group)
        res = time_kernel(
            submit, group, rank, warmup=args.warmup, iters=args.iters,
            agg="max", cudagraph=args.cudagraph)
    finally:
        # Re-init per case: finalize (tear down symmetric mappings) first, then
        # reverse the VMM allocations, then clear the descriptor registry.
        dist.barrier(group=group)
        if ShmemRuntime.is_initialized():
            ShmemRuntime.finalize()
        alloc = getattr(slots_src, "_vmm_allocation", None)
        if alloc is not None:
            alloc.destroy()
        reset_symmetric_descriptors()
    return res, E, B


def main():
    args = parse_args()
    if args.out and os.path.exists(args.out):
        raise FileExistsError("use a new --out CSV path")
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
        print(f"=== bench_grad_reduce_targets: world={world_size} R={R} "
              f"cases={len(cases)} === ", flush=True)

    dist.barrier()

    if rank == 0:
        print(f"{'Config':<20} {'E':>5} {'B':>4} {'H':>7} {'Hp':>6} {'Slots':>6} "
              f"{'Worst(us)':>10} {'Status':>8}")
        print("-" * 78)

    failures = 0
    results = []
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
            results.append(dict(Config=case["label"], E=E, B=B, H=case["H"],
                                Hp=case["Hp"], Slots=slots,
                                **{"Worst(us)": res["us"], "Status": res["status"]}))

    if rank == 0 and args.out:
        with open(args.out, "x", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=["Config", "E", "B", "H", "Hp",
                                                        "Slots", "Worst(us)", "Status"])
            writer.writeheader()
            writer.writerows(results)

    if in_group:
        ShmemRuntime.finalize()
    dist.destroy_process_group()
    if rank == 0 and failures:
        print(f"{failures}/{len(cases)} cases FAILED", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
