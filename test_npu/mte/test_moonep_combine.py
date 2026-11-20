#!/usr/bin/env python3
"""Multi-rank smoke / precision test for moonEP_combine (standalone, no dispatch).

By default all ranks spawn on one host (same habit as before)::

  python test_npu/mte/test_moonep_combine.py --ep-size 8
  python test_npu/mte/test_moonep_combine.py --ep-size 8 --prod-shape
  python test_npu/mte/test_moonep_combine.py --ep-size 8 --with-weight

Multi-node (aligned with ``test_moonep_dispatch_with_plan.py``): launch once
per node with the same ``--master-addr``/``--master-port`` (buffer-backed
init broadcasts the UID over dist), a unique ``--node-rank``, and ``--nproc-per-node``
local NPUs. Example 8 hosts × 8 cards = EP64::

  # on node ${NODE_RANK}=0..7
  python test_npu/mte/test_moonep_combine.py \\
    --ep-size 64 --nproc-per-node 8 --node-rank ${NODE_RANK} \\
    --master-addr <node0_ip> --master-port 29521 --prod-shape

Each rank fills its SHMEM-symmetric shard [N,H]; combine reads peers via shmem_ptr.
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

PRECISION_THR = 1e-2  # fail if both abs and rel error ≥ 1% (CPU golden; bf16 cast)

# Production 8-card EP combine shape (per rank): S=4096, K=16, H=7168.
PROD_NUM_TOKENS = 4096
PROD_HIDDEN = 7168
PROD_NUM_TOPK = 16


def global_rank_for_local(local_rank: int, args) -> int:
    if args.nproc_per_node is None:
        return local_rank
    return args.node_rank * args.nproc_per_node + local_rank


def local_process_count(args) -> int:
    return args.ep_size if args.nproc_per_node is None else args.nproc_per_node


def validate_args(args) -> None:
    if args.ep_size < 1:
        raise ValueError("--ep-size must be >= 1")
    if args.hidden % 16 != 0:
        raise ValueError("--hidden must be multiple of 16")
    if args.num_topk > 16:
        raise ValueError("--num-topk must be <= 16")
    if not args.master_addr:
        raise ValueError("--master-addr must not be empty")
    if args.master_port <= 0 or args.master_port > 65535:
        raise ValueError("--master-port must be in [1, 65535]")
    if args.nproc_per_node is None:
        if args.node_rank != 0:
            raise ValueError("--node-rank requires --nproc-per-node")
        return
    if args.nproc_per_node <= 0:
        raise ValueError("--nproc-per-node must be positive")
    if args.ep_size % args.nproc_per_node != 0:
        raise ValueError("--ep-size must be divisible by --nproc-per-node")
    num_nodes = args.ep_size // args.nproc_per_node
    if args.node_rank < 0 or args.node_rank >= num_nodes:
        raise ValueError(f"--node-rank must be in [0, {num_nodes})")


def make_local_shard(rank: int, n: int, h: int, device: str) -> torch.Tensor:
    """Deterministic local hidden: hidden[loff, :] = (rank+1)*100 + loff (bf16)."""
    loff = torch.arange(n, device=device, dtype=torch.float32)
    row = (rank + 1) * 100.0 + loff
    return row.unsqueeze(1).expand(n, h).contiguous().to(torch.bfloat16)


def make_local_weight(rank: int, n: int, device: str) -> torch.Tensor:
    loff = torch.arange(n, device=device, dtype=torch.float32)
    return ((rank + 1) * 1000.0 + loff).contiguous()


def make_dst(s: int, k: int, n: int, world: int, device: str) -> torch.Tensor:
    """Cross-rank plan: dst[s,k] = drank*N + loff, drank=(s+k)%W, loff=(s*K+k)%N."""
    ss = torch.arange(s, device=device).unsqueeze(1).expand(s, k)
    kk = torch.arange(k, device=device).unsqueeze(0).expand(s, k)
    drank = (ss + kk) % world
    loff = (ss * k + kk) % n
    return (drank * n + loff).to(torch.int32).contiguous()


def cpu_golden_hidden(dst: torch.Tensor, n: int, h: int) -> torch.Tensor:
    """Vectorized CPU reference (row value is scalar (drank+1)*100+loff)."""
    d = dst.cpu()
    drank = torch.div(d, n, rounding_mode="floor")
    loff = torch.remainder(d, n)
    contrib = (drank.to(torch.float32) + 1.0) * 100.0 + loff.to(torch.float32)
    contrib = torch.where(d >= 0, contrib, torch.zeros_like(contrib))
    row_sum = contrib.sum(dim=1)
    return row_sum.unsqueeze(1).expand(d.size(0), h).contiguous()


def cpu_golden_weights(dst: torch.Tensor, n: int) -> torch.Tensor:
    d = dst.cpu()
    raw = torch.where(d >= 0, d, -d - 1)
    drank = torch.div(raw, n, rounding_mode="floor")
    loff = torch.remainder(raw, n)
    return (drank.to(torch.float32) + 1.0) * 1000.0 + loff.to(torch.float32)


def select_check_indices(s: int, check_tokens: int) -> torch.Tensor:
    """Indices for precision check; full range when check_tokens<=0 or >=S."""
    if check_tokens <= 0 or check_tokens >= s:
        return torch.arange(s)
    n = check_tokens
    head = max(1, n // 3)
    tail = max(1, n // 3)
    mid = n - head - tail
    idx = []
    idx.extend(range(0, head))
    if mid > 0:
        start = max(head, (s - mid) // 2)
        idx.extend(range(start, min(s, start + mid)))
    idx.extend(range(max(0, s - tail), s))
    return torch.tensor(sorted(set(idx)), dtype=torch.long)


def check_precision(real: torch.Tensor, golden: torch.Tensor, thr: float = PRECISION_THR):
    ae = (real.float().cpu() - golden.float()).abs()
    re = ae / (golden.float().abs() + 1e-8)
    bad = (ae >= thr) & (re >= thr)
    return {
        "ok": not bool(bad.any().item()),
        "max_ae": float(ae.max().item()),
        "fail_both": int(bad.sum().item()),
    }


def run_rank(local_rank: int, args) -> None:
    world = int(args.ep_size)
    rank = global_rank_for_local(local_rank, args)
    device_id = local_device_index(local_rank)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world)

    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"

    s, h, k = args.num_tokens, args.hidden, args.num_topk
    n = s * k
    with_weight = bool(args.with_weight)

    src_chunks = []
    failed = False
    try:
        assert _C is not None
        combine_op = torch.classes.AscendMoonEP.Combine()
        # Buffer-backed only: needs dist (init_with_buffer broadcasts the UID over dist).
        if not dist.is_initialized():
            dist.init_process_group(
                backend="hccl",
                init_method=f"tcp://{args.master_addr}:{args.master_port}",
                rank=rank, world_size=world,
            )

        # Symmetric shards: register the VMM chunk -> init_with_buffer -> sym_tensor translation.
        hidden_src, _, _ = create_vmm_physical([n, h], torch.bfloat16, device_id)
        hidden_src.copy_(make_local_shard(rank, n, h, device))
        src_chunks.append(hidden_src)
        sk_stride = ((k + 7) // 8) * 8
        if with_weight:
            weight_src, _, _ = create_vmm_physical([n], torch.float32, device_id)
            weight_src.copy_(make_local_weight(rank, n, device))
            src_chunks.append(weight_src)
            weights_sk_launch = torch.empty(
                s, sk_stride, dtype=torch.float32, device=device
            )
        else:
            weight_src = None
            weights_sk_launch = None
        if not ShmemRuntime.is_initialized():
            init_shmem(args=args, group=None)
        hidden_buf = sym_tensor(hidden_src)
        weight_buf = sym_tensor(weight_src) if weight_src is not None else None

        dst = make_dst(s, k, n, world, device).view(-1)
        dup_groups = torch.zeros(n, 3, dtype=torch.int32, device=device)
        dup_loffs = torch.zeros(n, dtype=torch.int32, device=device)
        dup_counts = torch.tensor([0, 0], dtype=torch.int32, device=device)
        hidden_sk = torch.empty(s, h, dtype=torch.bfloat16, device=device)

        ShmemRuntime.barrier()
        combine_op.compute(
            hidden_buf,
            weight_buf,
            dst,
            dup_groups,
            dup_loffs,
            dup_counts,
            hidden_sk,
            weights_sk_launch,
            s,
            h,
            k,
            world,
            rank,
            False,
        )
        torch.npu.synchronize()
        ShmemRuntime.barrier()

        check_idx = select_check_indices(s, int(args.check_tokens))
        dst_cpu = dst.view(s, k).cpu()
        golden_h = cpu_golden_hidden(dst_cpu, n, h)
        stats = check_precision(hidden_sk[check_idx], golden_h[check_idx])
        print(
            f"[rank {rank}] hidden: ok={stats['ok']} max_ae={stats['max_ae']:.5f} "
            f"fail_both={stats['fail_both']} shape={tuple(hidden_sk.shape)} "
            f"checked={check_idx.numel()}/{s}",
            flush=True,
        )
        if not stats["ok"]:
            failed = True

        if with_weight:
            weights_sk = weights_sk_launch[:, :k].contiguous()
            golden_w = cpu_golden_weights(dst_cpu, n)
            wstats = check_precision(weights_sk[check_idx], golden_w[check_idx])
            print(
                f"[rank {rank}] weights: ok={wstats['ok']} max_ae={wstats['max_ae']:.5f} "
                f"fail_both={wstats['fail_both']} shape={tuple(weights_sk.shape)} "
                f"checked={check_idx.numel()}/{s}",
                flush=True,
            )
            if not wstats["ok"]:
                failed = True

        if not failed:
            print(f"[rank {rank}] PASS", flush=True)
        else:
            print(f"[rank {rank}] FAIL", flush=True)

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
        raise RuntimeError(f"rank {rank} moonEP_combine precision failed")


def parse_args():
    p = argparse.ArgumentParser(description="moonEP_combine multi-rank test")
    p.add_argument("--ep-size", type=int, default=8, help="world size / EP size")
    p.add_argument("--first-device", type=int, default=0, help="Deprecated and ignored: the per-rank device index is resolved from ASCEND_RT_VISIBLE_DEVICES via npu_patch._device.")
    p.add_argument(
        "--node-rank",
        type=int,
        default=0,
        help="zero-based node index; requires --nproc-per-node",
    )
    p.add_argument(
        "--nproc-per-node",
        type=int,
        default=None,
        help="multi-node mode: spawn this many local NPU processes",
    )
    p.add_argument(
        "--master-addr",
        default=os.environ.get("MASTER_ADDR", "127.0.0.1"),
        help="rank-0 host address (dist UID bootstrap)",
    )
    p.add_argument(
        "--master-port",
        type=int,
        default=int(os.environ.get("MASTER_PORT", "29521")),
        help="rank-0 TCP port (dist UID bootstrap)",
    )
    p.add_argument("--num-tokens", type=int, default=16)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num-topk", type=int, default=4)
    p.add_argument(
        "--prod-shape",
        action="store_true",
        help=(
            f"Override S/H/K to production-like "
            f"{PROD_NUM_TOKENS}/{PROD_HIDDEN}/{PROD_NUM_TOPK}"
        ),
    )
    p.add_argument(
        "--check-tokens",
        type=int,
        default=None,
        help="Precision-check this many tokens (0=all). Default: all; with --prod-shape: 64.",
    )
    p.add_argument("--with-weight", action="store_true", default=None)
    p.add_argument("--no-weight", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.with_weight is None:
        args.with_weight = True  # weights included by default
    if args.no_weight:
        args.with_weight = False
    if args.prod_shape:
        args.num_tokens = PROD_NUM_TOKENS
        args.hidden = PROD_HIDDEN
        args.num_topk = PROD_NUM_TOPK
        if args.check_tokens is None:
            args.check_tokens = 64
    if args.check_tokens is None:
        args.check_tokens = 0  # all tokens
    validate_args(args)

    n = args.num_tokens * args.num_topk
    local_count = local_process_count(args)
    first_rank = global_rank_for_local(0, args)
    last_rank = global_rank_for_local(local_count - 1, args)
    print(
        f"moonEP_combine test: ep={args.ep_size} S={args.num_tokens} H={args.hidden} "
        f"K={args.num_topk} N={n} with_weight={args.with_weight} "
        f"check_tokens={args.check_tokens or 'all'} "
        f"master={args.master_addr}:{args.master_port} "
        f"node_rank={args.node_rank} local_processes={local_count} "
        f"global_ranks={first_rank}-{last_rank}",
        flush=True,
    )

    if local_count == 1:
        run_rank(0, args)
    else:
        mp.spawn(run_rank, args=(args,), nprocs=local_count, join=True)

    if args.nproc_per_node is None:
        print("ALL RANKS PASS", flush=True)
    else:
        print(f"NODE {args.node_rank} LOCAL RANKS PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
