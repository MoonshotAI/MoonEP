#!/usr/bin/env python3
"""Joint kernel test: dispatch → combine (no GMM / expert FFN).

Buffer-backed: the symmetric shards are built with ``create_vmm_physical`` + ``sym_tensor``,
``init_shmem`` (single-node default goes through ``ShmemRuntime.init_with_buffer``, UID bootstrap) builds
the symmetric heap in one place, and the kernels run through ``torch.classes.AscendMoonEP.Dispatch/Combine``.

Plan / tensor helpers come from ``test_moonep_dispatch_with_plan.py``.
Golden (no intermediate compute):

  - hidden_sk  ≈  K * hidden_sh   (K-sum of identical routed copies)
  - route_weights_sk ≈ route_weights_sk input  (per-route gather)

Usage:
  python test_npu/mte/test_dispatch_combine_joint.py --R 8 --S 16 --K 4 --H 128
  python test_npu/mte/test_dispatch_combine_joint.py --R 8 --S 64 --K 8 --H 128 --no-weights
  python test_npu/mte/test_dispatch_combine_joint.py --R 8 --prod-shape            # checks 64
  python test_npu/mte/test_dispatch_combine_joint.py --R 8 --prod-shape --max-checks 0  # all
  # prod-shape => S=4096, K=16, E=896, H=7168
"""
from __future__ import annotations


import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel_test_utils import init_shmem
import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from npu_patch._device import local_device_index
from npu_patch import ShmemRuntime, _C
from npu_patch.buffer_c import (
    create_vmm_physical,
    reset_symmetric_descriptors,
    sym_tensor,
)

# Reuse plan / tensor helpers from the dispatch single-op test.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_moonep_dispatch_with_plan as dispatch_test


PRECISION_THR = 1e-2  # fail if both abs and rel error ≥ 1% (CPU golden; bf16 cast)
# Production 8-card EP shape: S=4096, K=16, E=896, H=7168.
PROD_S = 4096
PROD_H = 7168
PROD_K = 16
PROD_E = 896


def validate_joint_args(args) -> None:
    dispatch_test.validate_args(args)
    if args.H % 16 != 0:
        raise ValueError("joint test requires H multiple of 16 (combine UB)")
    if args.K > 16:
        raise ValueError("joint test requires K <= 16 (combine limit)")
    if args.H * 10 + 256 > 192 * 1024:
        raise ValueError("H too large for combine single-pass UB")


def check_precision(real: torch.Tensor, golden: torch.Tensor, thr: float = PRECISION_THR):
    ae = (real.float().cpu() - golden.float().cpu()).abs()
    re = ae / (golden.float().cpu().abs() + 1e-8)
    bad = (ae >= thr) & (re >= thr)
    return {
        "ok": not bool(bad.any().item()),
        "max_ae": float(ae.max().item()),
        "fail_both": int(bad.sum().item()),
    }


def select_token_indices(s: int, max_checks: int, seed: int) -> torch.Tensor:
    if max_checks <= 0 or max_checks >= s:
        return torch.arange(s)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed + 4242)
    perm = torch.randperm(s, generator=rng)
    return perm[:max_checks].sort().values


def run_rank(local_rank: int, args) -> None:
    validate_joint_args(args)
    rank = dispatch_test.global_rank_for_local(local_rank, args)
    device_id = local_device_index(local_rank)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(args.R)

    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"

    s, h, k = int(args.S), int(args.H), int(args.K)
    n = dispatch_test.num_output_slots(args)
    with_weight = not bool(args.no_weights)

    src_chunks = []
    failed = False
    try:
        assert _C is not None
        dispatch_op = torch.classes.AscendMoonEP.Dispatch()
        combine_op = torch.classes.AscendMoonEP.Combine()
        # Buffer-backed only: needs dist (init_with_buffer broadcasts the UID over dist).
        if not dist.is_initialized():
            dist.init_process_group(
                backend="hccl",
                init_method=f"tcp://{args.master_addr}:{args.master_port}",
                rank=rank, world_size=args.R,
            )

        # Symmetric shards: register the VMM chunk -> init_with_buffer -> sym_tensor translation.
        hidden_src, _, _ = create_vmm_physical([n, h], torch.bfloat16, device_id)
        hidden_src.zero_()
        src_chunks.append(hidden_src)
        if with_weight:
            weight_src, _, _ = create_vmm_physical([n], torch.float32, device_id)
            weight_src.zero_()
            src_chunks.append(weight_src)
        else:
            weight_src = None
        # Dispatch sync_buffer: per-rank peer completion/credit flag cacheline pool
        # for group flow control [R * kMaxDispatchGroupWidth(16) *
        # kMaxCreatedUdmaQpCount(6) * 512] (mirrors Buffer layout in api.py).
        # Dispatch::compute requires this tensor in its C++ ABI even on the
        # non-UDMA path; the kernel only consumes it when use_udma is true.
        dispatch_sync_src, _, _ = create_vmm_physical(
            [args.R * 16 * 6 * 512], torch.uint8, device_id)
        dispatch_sync_src.zero_()
        src_chunks.append(dispatch_sync_src)
        if not ShmemRuntime.is_initialized():
            init_shmem(args=args, group=None)
        hidden_buf = sym_tensor(hidden_src)
        weight_buf = sym_tensor(weight_src) if weight_src is not None else None
        dispatch_sync_buffer = sym_tensor(dispatch_sync_src)

        dst, dup_groups, dup_loffs, dup_counts, _expected = dispatch_test.make_plan(
            args, rank
        )
        hidden = dispatch_test.make_hidden(rank, args)
        hidden_npu = hidden.to(device=device)
        weights_npu = (
            dispatch_test.make_weights(rank, args).to(device=device)
            if with_weight
            else None
        )

        dst_npu = dst.to(device=device)
        dup_groups_npu = dup_groups.to(device=device)
        dup_loffs_npu = dup_loffs.to(device=device)
        dup_counts_npu = dup_counts.to(device=device)

        sk_stride = ((k + 7) // 8) * 8
        hidden_sk = torch.empty(s, h, dtype=torch.bfloat16, device=device)
        weights_sk_launch = (
            torch.empty(s, sk_stride, dtype=torch.float32, device=device)
            if with_weight
            else None
        )

        ShmemRuntime.barrier()
        dispatch_op.compute(
            hidden_npu,
            weights_npu,
            dst_npu,
            dup_groups_npu,
            dup_loffs_npu,
            dup_counts_npu,
            hidden_buf,
            weight_buf,
            dispatch_sync_buffer,
            s,
            h,
            k,
            args.R,
            rank,
            False,
        )
        torch.npu.synchronize()
        ShmemRuntime.barrier()

        # No GMM: combine reads the dispatch-filled symmetric shard in place.
        combine_op.compute(
            hidden_buf,
            weight_buf,
            dst_npu,
            dup_groups_npu,
            dup_loffs_npu,
            dup_counts_npu,
            hidden_sk,
            weights_sk_launch,
            s,
            h,
            k,
            args.R,
            rank,
            False,
        )
        torch.npu.synchronize()
        ShmemRuntime.barrier()

        check_idx = select_token_indices(s, args.max_checks, args.seed)
        # Hidden: each of K routes contributes one copy of the source token
        # (dup routes are folded into the primary by combine prologue; combine
        # only loads dst>=0, so the net factor is still K).
        golden_h = (hidden.float() * float(k)).to(torch.bfloat16)
        hstats = check_precision(hidden_sk[check_idx], golden_h[check_idx])
        print(
            f"[rank {rank}] joint hidden: ok={hstats['ok']} "
            f"max_ae={hstats['max_ae']:.5f} fail_both={hstats['fail_both']} "
            f"checked={check_idx.numel()}/{s}",
            flush=True,
        )
        if not hstats["ok"]:
            failed = True

        if with_weight:
            assert weights_sk_launch is not None and weights_npu is not None
            route_weights_sk = weights_sk_launch[:, :k].contiguous()
            wstats = check_precision(
                route_weights_sk[check_idx], weights_npu[check_idx]
            )
            print(
                f"[rank {rank}] joint weights: ok={wstats['ok']} "
                f"max_ae={wstats['max_ae']:.5f} fail_both={wstats['fail_both']} "
                f"checked={check_idx.numel()}/{s}",
                flush=True,
            )
            if not wstats["ok"]:
                failed = True

        print(f"[rank {rank}] {'PASS' if not failed else 'FAIL'}", flush=True)
    finally:
        # Buffer-backed only: finalize first, then the VMM inverse operations, and clear the registry last.
        if ShmemRuntime.is_initialized():
            ShmemRuntime.finalize()
        for t in src_chunks:
            alloc = getattr(t, '_vmm_allocation', None)
            if alloc is not None:
                alloc.destroy()
        reset_symmetric_descriptors()

    if failed:
        raise RuntimeError(f"rank {rank} dispatch+combine joint precision failed")


def parse_args():
    p = argparse.ArgumentParser(
        description="dispatch→combine joint kernel test (buffer-backed, no GMM)"
    )
    p.add_argument("--R", type=int, default=8)
    p.add_argument("--S", type=int, default=16)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--H", type=int, default=128)
    p.add_argument("--E", type=int, default=None)
    p.add_argument("--seed", type=int, default=dispatch_test.SEED)
    p.add_argument("--first-device", type=int, default=0, help="Deprecated and ignored: the per-rank device index is resolved from ASCEND_RT_VISIBLE_DEVICES via npu_patch._device.")
    p.add_argument("--master-addr", type=str, default="127.0.0.1")
    p.add_argument("--master-port", type=int, default=29521)
    p.add_argument("--shmem-ip-port", type=str, default="")
    p.add_argument("--nproc-per-node", type=int, default=None)
    p.add_argument("--node-rank", type=int, default=0)
    p.add_argument(
        "--max-checks",
        type=int,
        default=None,
        help=(
            "token rows to check; 0 means all. "
            "Default: all; with --prod-shape alone: 64."
        ),
    )
    p.add_argument("--no-weights", action="store_true")
    p.add_argument(
        "--prod-shape",
        action="store_true",
        help=f"Override S/H/K to {PROD_S}/{PROD_H}/{PROD_K}",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    # Fields required by dispatch_test.validate_args / helpers.
    args.mode = "kernel"
    args.profile = False
    args.profile_iters = 1
    args.profile_warmup = 0
    args.output_mode = "zero-copy"
    args.clear_symmetric = False
    args.no_weights = bool(args.no_weights)
    # Buffer-backed only: the UID goes over hccl broadcast (inside init_with_buffer, the single-node default path),
    # and the bootstrap attribute is only used by dispatch_test to pick its dist init backend.
    args.bootstrap = "uid"

    if args.prod_shape:
        args.S = PROD_S
        args.H = PROD_H
        args.K = PROD_K
        args.E = PROD_E
        if args.max_checks is None:
            args.max_checks = 64
    if args.max_checks is None:
        args.max_checks = 0  # all tokens

    validate_joint_args(args)
    local_count = dispatch_test.local_process_count(args)
    print(
        f"dispatch+combine joint: R={args.R} S={args.S} H={args.H} K={args.K} "
        f"E={dispatch_test.num_experts(args)} weights={not args.no_weights} "
        f"shmem=buffer-backed max_checks={args.max_checks or 'all'}",
        flush=True,
    )

    if local_count == 1:
        run_rank(0, args)
    else:
        mp.spawn(run_rank, args=(args,), nprocs=local_count, join=True)

    print("ALL RANKS PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
