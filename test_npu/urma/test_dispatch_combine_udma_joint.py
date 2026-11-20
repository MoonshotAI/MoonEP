#!/usr/bin/env python3
"""Joint UDMA test: dispatch -> combine without an expert/GMM stage.

The dispatch result stays in Buffer's symmetric storage and is passed to
``Buffer.combine`` in zero-copy mode. The combine launcher selects UDMA through
``ASCEND_MOONEP_COMBINE_USE_UDMA``. Therefore this test covers the framework-level
dispatch and UDMA combine path as one operation chain.

Golden result (there is no expert computation between the two operations):

* every token is dispatched K times, so combined hidden is ``K * hidden``;
* every route weight is gathered back to its original ``[S, K]`` position.

Examples:

  python test_npu/urma/test_dispatch_combine_udma_joint.py --R 8 --S 16 --K 4 --H 128
  python test_npu/urma/test_dispatch_combine_udma_joint.py --R 8 --S 64 --K 8 --H 128 --no-weights
  python test_npu/urma/test_dispatch_combine_udma_joint.py --prod-shape --max-checks 64
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

# Reuse exactly the same data and communication-plan generation as the UDMA
# dispatch single-op test.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_moonep_dispatch_udma_with_plan as dispatch_test


PRECISION_THR = 1e-2
UDMA_QP_COUNT = 2
PROD_R = 8
PROD_S = 4096
PROD_H = 7168
PROD_K = 16
PROD_E = 896


def validate_joint_args(args) -> None:
    dispatch_test.validate_args(args)
    if args.K > 16:
        raise ValueError("UDMA combine requires K <= 16")


def check_precision(
    real: torch.Tensor,
    golden: torch.Tensor,
    threshold: float = PRECISION_THR,
) -> dict[str, int | float | bool]:
    actual = real.float().cpu()
    expected = golden.float().cpu()
    absolute_error = (actual - expected).abs()
    relative_error = absolute_error / (expected.abs() + 1e-8)
    failed = (absolute_error >= threshold) & (relative_error >= threshold)
    return {
        "ok": not bool(failed.any().item()),
        "max_ae": float(absolute_error.max().item()),
        "fail_both": int(failed.sum().item()),
    }


def select_token_indices(s: int, max_checks: int, seed: int) -> torch.Tensor:
    if max_checks <= 0 or max_checks >= s:
        return torch.arange(s)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 4242)
    return torch.randperm(s, generator=generator)[:max_checks].sort().values


def make_comm_plan(args, rank: int, device: torch.device):
    from npu_patch import MoonEPCommPlan

    dst, dup_groups, dup_loffs, dup_counts, _ = dispatch_test.make_plan(
        args, rank
    )
    E = dispatch_test.num_experts(args)
    B = dispatch_test.num_reserved_slots(args)
    NvS = dispatch_test.num_output_slots(args)
    return MoonEPCommPlan(
        dst=dst.to(device=device),
        experts_to_copy=torch.full(
            (args.R, B), -1, dtype=torch.int32, device=device
        ),
        zero_fill_ranges=torch.zeros(
            (E + B, 2), dtype=torch.int32, device=device
        ),
        remote_stats=torch.zeros(2, dtype=torch.int32, device=device),
        dup_groups=dup_groups.to(device=device),
        dup_loffs=dup_loffs.to(device=device),
        dup_counts=dup_counts.to(device=device),
        N=NvS,
        R=args.R,
        E=E,
        B=B,
        NvS=NvS,
        K=args.K,
    )


def run_rank(local_rank: int, args) -> None:
    validate_joint_args(args)
    rank = dispatch_test.global_rank_for_local(local_rank, args)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(args.R)
    os.environ["ASCEND_MOONEP_COMBINE_USE_UDMA"] = "1"

    import torch_npu
    from npu_patch._device import local_device_index
    from npu_patch import Buffer, ShmemRuntime
    # The dedicated ``npu_patch.dispatch_udma`` module was removed; the
    # UDMA dispatch path is now driven by ``ASCEND_MOONEP_DISPATCH_USE_UDMA``
    # (set by Buffer construction on UDMA-capable hardware) through the shared
    # ``npu_patch.dispatch`` launcher, which is what the sibling
    # ``test_moonep_dispatch_udma_with_plan`` already uses.
    from npu_patch.dispatch import (
        launch_dispatch,
        materialize_dispatch_outputs,
    )

    # Must come *after* the import above: a function-local ``import`` makes the
    # name local to this whole scope, so any earlier call would raise
    # UnboundLocalError.
    device_id = local_device_index(local_rank)
    torch.npu.set_device(device_id)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="hccl",
            init_method=f"tcp://{args.master_addr}:{args.master_port}",
            rank=rank,
            world_size=args.R,
            device_id=torch.device(f"npu:{device_id}"),
        )

    buffer = None
    hidden_nvsh = None
    route_weights_nvs = None
    hidden_sk = None
    route_weights_sk = None
    plan = None
    ctx = None
    failed = False
    try:
        E = dispatch_test.num_experts(args)
        B = dispatch_test.num_reserved_slots(args)
        buffer = Buffer(
            args.S,
            args.H,
            args.K,
            E,
            args.R,
            B=B,
            explicitly_destroy=True,
        )
        if not ShmemRuntime.is_initialized():
            init_shmem(args=args, 
                group=None,
                enable_udma=True,
                qp_count=UDMA_QP_COUNT,
            )

        buffer._ensure_shmem_views()
        ctx = buffer._require_ctx()
        ctx["active_qp_count"] = UDMA_QP_COUNT

        hidden_cpu = dispatch_test.make_hidden(rank, args)
        hidden = hidden_cpu.to(device=f"npu:{device_id}")
        route_weights = (
            dispatch_test.make_weights(rank, args).to(
                device=f"npu:{device_id}"
            )
            if not args.no_weights
            else None
        )
        plan = make_comm_plan(args, rank, hidden.device)

        ShmemRuntime.barrier()
        launch_dispatch(ctx, hidden, route_weights, plan)
        torch.npu.synchronize()
        ShmemRuntime.barrier()

        # The combine input must be the exact symmetric shard populated by
        # dispatch.  Pointer checks make accidental copy-path coverage fail.
        hidden_nvsh, route_weights_nvs = materialize_dispatch_outputs(
            ctx,
            with_weights=route_weights is not None,
            zero_copy=True,
        )
        if hidden_nvsh.data_ptr() != ctx["hidden_buf_local"].data_ptr():
            raise AssertionError("UDMA dispatch hidden output is not zero-copy")
        if route_weights_nvs is not None and (
            route_weights_nvs.data_ptr()
            != ctx["weights_buf_local"].data_ptr()
        ):
            raise AssertionError("UDMA dispatch weight output is not zero-copy")

        # On the dispatch side, materialize_dispatch_outputs (which does not support this switch)
        # produces the aliased tensors; the combine side declares router_weights_zero_copy explicitly to skip the self-copy.
        hidden_sk, route_weights_sk, done = buffer.combine(
            plan=plan,
            hidden_nvsh=hidden_nvsh,
            route_weights_nvs=route_weights_nvs,
            async_finish=False,
            inter_rank_sync=True,
            zero_copy=True,
            router_weights_zero_copy=True,
        )
        if done is not None:
            raise AssertionError("synchronous UDMA combine must return done=None")
        torch.npu.synchronize()
        ShmemRuntime.barrier()

        check_idx = select_token_indices(args.S, args.max_checks, args.seed)
        golden_hidden = (hidden_cpu.float() * float(args.K)).to(torch.bfloat16)
        hidden_stats = check_precision(
            hidden_sk[check_idx], golden_hidden[check_idx]
        )
        print(
            f"[rank {rank}] UDMA joint hidden: ok={hidden_stats['ok']} "
            f"max_ae={hidden_stats['max_ae']:.5f} "
            f"fail_both={hidden_stats['fail_both']} "
            f"checked={check_idx.numel()}/{args.S}",
            flush=True,
        )
        failed = not bool(hidden_stats["ok"])

        if route_weights is None:
            if route_weights_sk is not None:
                raise AssertionError(
                    "UDMA combine returned weights when input weights are absent"
                )
        else:
            if route_weights_sk is None:
                raise AssertionError("UDMA combine did not return route weights")
            weight_stats = check_precision(
                route_weights_sk[check_idx], route_weights[check_idx]
            )
            print(
                f"[rank {rank}] UDMA joint weights: ok={weight_stats['ok']} "
                f"max_ae={weight_stats['max_ae']:.5f} "
                f"fail_both={weight_stats['fail_both']} "
                f"checked={check_idx.numel()}/{args.S}",
                flush=True,
            )
            failed = failed or not bool(weight_stats["ok"])

        print(
            f"[rank {rank}] UDMA dispatch+combine "
            f"{'PASS' if not failed else 'FAIL'}",
            flush=True,
        )
    finally:
        # Release all aliases before Buffer tears down its symmetric VMM.
        hidden_nvsh = None
        route_weights_nvs = None
        hidden_sk = None
        route_weights_sk = None
        plan = None
        ctx = None
        # Finalize before destroying: the R4 contract requires ShmemRuntime.finalize() to run
        # before Buffer.destroy() (tear down the symmetric mapping first, then release Buffer's VMM
        # physical memory; freeing the memory before finalize would be a use-after-free). This
        # matches the release order of test_moonep_dispatch_udma_with_plan and of
        # finalize_shmem_and_release in kernel_test_utils.
        try:
            if ShmemRuntime.is_initialized():
                ShmemRuntime.finalize()
        finally:
            if buffer is not None:
                buffer.destroy()
            if dist.is_initialized():
                dist.destroy_process_group()

    if failed:
        raise RuntimeError(
            f"rank {rank} UDMA dispatch+combine joint precision failed"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="UDMA dispatch -> UDMA combine joint test (no GMM)"
    )
    parser.add_argument("--R", type=int, default=8)
    parser.add_argument("--S", type=int, default=16)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--H", type=int, default=128)
    parser.add_argument("--E", type=int, default=None)
    parser.add_argument("--seed", type=int, default=dispatch_test.SEED)
    parser.add_argument("--first-device", type=int, default=0, help="Deprecated and ignored: the per-rank device index is resolved from ASCEND_RT_VISIBLE_DEVICES via npu_patch._device.")
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29521)
    parser.add_argument("--nproc-per-node", type=int, default=None)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument(
        "--plan-mode",
        choices=("random", "cyclic"),
        default="random",
    )
    parser.add_argument(
        "--max-checks",
        type=int,
        default=None,
        help="token rows to check; 0 means all (prod-shape default: 64)",
    )
    parser.add_argument("--no-weights", action="store_true")
    parser.add_argument(
        "--prod-shape",
        action="store_true",
        help="use R=8, S=4096, K=16, H=7168, E=896",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # Compatibility fields consumed by dispatch_test.validate_args.
    args.mode = "kernel"
    args.bootstrap = "uid"
    args.shmem_ip_port = ""
    args.profile = False
    args.profile_iters = 1
    args.profile_warmup = 0
    args.active_qp_count = UDMA_QP_COUNT

    if args.prod_shape:
        args.R = PROD_R
        args.S = PROD_S
        args.H = PROD_H
        args.K = PROD_K
        args.E = PROD_E
        if args.max_checks is None:
            args.max_checks = 64
    if args.max_checks is None:
        args.max_checks = 0

    validate_joint_args(args)
    local_count = dispatch_test.local_process_count(args)
    print(
        f"UDMA dispatch+combine joint: R={args.R} S={args.S} H={args.H} "
        f"K={args.K} E={dispatch_test.num_experts(args)} "
        f"weights={not args.no_weights} plan_mode={args.plan_mode} "
        f"qp_count={UDMA_QP_COUNT} max_checks={args.max_checks or 'all'}",
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
