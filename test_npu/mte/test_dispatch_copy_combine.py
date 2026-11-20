# SPDX-License-Identifier: Apache-2.0
"""Regression: MTE dispatch's independent output must be complete.

``Buffer.dispatch`` with the default ``zero_copy=False`` snapshots
``ctx['hidden_buf_local']`` / ``ctx['weights_buf_local']`` on the current stream
right after the kernel. The MTE dispatch kernel scatters the payload with
``aclshmemx_mte_put_nbi``; its trailing ``aclshmemx_barrier_all_vec`` is an
alias of ``aclshmemx_sync_vec_all``, which does not complete outstanding NBI
communication. Without an ``aclshmemx_mte_quiet`` drain before that barrier the
copy can observe a shard the peer ranks have not finished writing (the reported
12-of-32 zero route weights / hidden mismatch). The kernel now quiets first.

Fixture: ``S == B`` tokens per rank, each routed to one local and one next-rank
expert (``K=2``, no duplicate rows), identity experts, both route weights 0.5.
``dispatch -> weight -> combine`` must reproduce the original hidden and route
scores exactly, so any stale/zero shard slot shows up as a mismatch.

Run with:
    torchrun --nproc_per_node=2 -m pytest -s test_npu/mte/test_dispatch_copy_combine.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel_test_utils import init_shmem

import pytest
import torch
import torch.distributed as dist

import torch_npu

from npu_patch._device import local_device_index
from npu_patch import Buffer, ShmemRuntime
from npu_patch.buffer_c import reset_symmetric_descriptors


CASES = [
    # B=16/S=16/K=2 mirrors the reported per-shard shape (NvS=32).
    {"name": "h128", "H": 128, "rounds": 8},
    {"name": "h1024", "H": 1024, "rounds": 4},
]


@pytest.fixture(scope="module")
def dist_env():
    if "RANK" not in os.environ:
        pytest.skip(
            "test_dispatch_copy_combine requires torchrun with at least 2 NPUs")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.npu.set_device(local_device_index(local_rank))
    owns_process_group = not dist.is_initialized()
    if owns_process_group:
        dist.init_process_group(
            backend="hccl", device_id=torch.device(f"npu:{local_device_index(local_rank)}"))
    R = dist.get_world_size()
    if R < 2:
        if owns_process_group:
            dist.destroy_process_group()
        pytest.skip("test_dispatch_copy_combine requires at least 2 NPUs")

    yield dist.get_rank(), local_rank, R

    dist.barrier()


def _source_row(src_rank: int, S: int, iteration: int) -> torch.Tensor:
    """One source rank's [S] hidden values, exactly representable in bf16.

    Multiples of 0.25 below 32 are exact in bf16, so 0.5 * value and the K-sum
    of two identical copies are exact too and the check can use torch.equal.
    """
    return (torch.arange(S, dtype=torch.float32) * 0.5
            + float(src_rank + 1) + iteration * 0.25)


def _source_hidden(src_rank: int, S: int, H: int, iteration: int,
                   device) -> torch.Tensor:
    row = _source_row(src_rank, S, iteration).to(torch.bfloat16)
    return row.view(S, 1).expand(S, H).contiguous().to(device)


def _expected_shard(rank, R, S, K, H, iteration, dst_by_src):
    """CPU model of the dispatch shard for this rank (no duplicate rows)."""
    NvS = S * K
    expected_hidden = torch.zeros(NvS, H, dtype=torch.bfloat16)
    expected_weights = torch.zeros(NvS, dtype=torch.float32)
    covered = torch.zeros(NvS, dtype=torch.bool)
    for src in range(R):
        row = _source_row(src, S, iteration).to(torch.bfloat16)
        for route in range(NvS):
            raw = int(dst_by_src[src][route])
            assert raw >= 0, (
                "fixture must not produce duplicate routes (negative dst)")
            dest, loff = divmod(raw, NvS)
            if dest != rank:
                continue
            expected_hidden[loff] = row[route // K]
            expected_weights[loff] = 0.5
            covered[loff] = True
    return expected_hidden, expected_weights, covered


def _collective_ok(device, local_ok: bool) -> bool:
    flag = torch.tensor([int(local_ok)], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def run_case(rank: int, local_rank: int, R: int, case: dict) -> None:
    dev = f"npu:{local_device_index(local_rank)}"
    H = case["H"]
    rounds = case["rounds"]
    K = 2
    B = 16
    S = B
    E = R * B

    buffer = None
    try:
        buffer = Buffer(S, H, K, E, R, B=B, explicitly_destroy=True,
                        use_udma=False)
        if not ShmemRuntime.is_initialized():
            init_shmem(group=None, enable_udma=False)
        ctx = buffer._require_ctx()
        hidden_shared = ctx["hidden_buf_local"]

        token = torch.arange(S, device=dev, dtype=torch.int32) % B
        topk = torch.stack(
            [token + rank * B, token + ((rank + 1) % R) * B], dim=1
        ).contiguous()
        weights = torch.full((S, K), 0.5, dtype=torch.float32, device=dev)
        counts = torch.bincount(topk.flatten().long(), minlength=E).int()

        worst_hidden = 0.0
        worst_score = 0.0
        for iteration in range(rounds):
            hidden = _source_hidden(rank, S, H, iteration, dev)

            value = buffer.dispatch(hidden, weights, topk, counts)
            dispatched, routed, cu_seqlens, plan = value
            assert cu_seqlens is not None, "fresh planning must run"
            assert dispatched.data_ptr() != hidden_shared.data_ptr(), (
                "zero_copy defaults to False; the output must be an ordinary copy")
            torch.npu.synchronize()
            dist.barrier()

            # Re-read the plan every round: planning may reorder which route
            # owns which loff even though the routing input is identical.
            # plan.dst is per-source-rank, so every rank's row is needed.
            gathered = [torch.zeros_like(plan.dst) for _ in range(R)]
            dist.all_gather(gathered, plan.dst)
            dst_by_src = [g.cpu().tolist() for g in gathered]

            expected_hidden, expected_weights, covered = _expected_shard(
                rank, R, S, K, H, iteration, dst_by_src)
            assert bool(covered.all()), (
                f"rank {rank}: fixture only covered {int(covered.sum())}/{covered.numel()} slots")

            hidden_error = (dispatched.cpu() - expected_hidden).abs().max().item()
            score_error = (routed.cpu() - expected_weights).abs().max().item()

            weighted = (dispatched.float() * routed[:, None]).to(
                dispatched.dtype).contiguous()
            actual, scores, done = buffer.combine(
                plan=plan, hidden_nvsh=weighted, route_weights_nvs=routed)
            assert done is None, "sync combine must not return an event"
            torch.npu.synchronize()

            combine_hidden_error = (actual.float() - hidden.float()).abs().max().item()
            combine_score_error = (scores - weights).abs().max().item()

            worst_hidden = max(worst_hidden, hidden_error, combine_hidden_error)
            worst_score = max(worst_score, score_error, combine_score_error)
            local_ok = (hidden_error == 0.0 and score_error == 0.0
                        and combine_hidden_error == 0.0 and combine_score_error == 0.0)
            if not local_ok:
                print(
                    f"[rank {rank}] {case['name']} round {iteration}: "
                    f"dispatch_hidden_err={hidden_error} "
                    f"dispatch_score_err={score_error} "
                    f"combine_hidden_err={combine_hidden_error} "
                    f"combine_score_err={combine_score_error}",
                    flush=True,
                )
            if not _collective_ok(dev, local_ok):
                raise AssertionError(
                    f"{case['name']} dispatch independent output incomplete "
                    f"(round {iteration}): worst hidden={worst_hidden} "
                    f"score={worst_score}")
            dist.barrier()

        if rank == 0:
            print(
                f"  [PASS] {case['name']}: R={R}, S={S}, K={K}, H={H}, "
                f"E={E}, rounds={rounds}", flush=True)
    finally:
        if ShmemRuntime.is_initialized():
            ShmemRuntime.finalize()
        if buffer is not None:
            buffer.destroy()
        reset_symmetric_descriptors()


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_dispatch_copy_combine(dist_env, case):
    rank, local_rank, R = dist_env
    if rank == 0:
        print(f"[test_dispatch_copy_combine] case={case['name']} R={R} use_udma=False")
    run_case(rank, local_rank, R, case)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s"]))
