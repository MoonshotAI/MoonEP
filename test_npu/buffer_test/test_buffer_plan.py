"""Buffer.dispatch plan-correctness test (planning kernel driven by Buffer API).

Unlike ``test_buffer_dispatch_with_reference_plan.py`` (which supplies a
prebuilt plan and skips the planning kernel), this test calls
``buffer.dispatch(..., plan=None)`` so the **planning operator itself** runs
through the public Buffer path. It then asserts two things:

  * the produced ``MoonEPCommPlan`` matches the torch planning reference
    (``planing_reference.reference_plan_and_cu_seqlens``) semantically;
  * the dispatched symmetric shard matches the CPU decode model, exactly as
    the dispatch-kernel test does.

Run with (Ascend host, CANN + torch_npu + shmem + built npu_patch):

    torchrun --nproc_per_node=8 -m pytest -s test_npu/buffer_test/test_buffer_plan.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel_test_utils import init_shmem

import pytest
import torch
import torch.distributed as dist

import torch_npu

from generate_topk_routing import generate_topk_routing
from planing_reference import reference_plan_and_cu_seqlens

from npu_patch._device import local_device_index
from npu_patch import Buffer, ShmemRuntime
from npu_patch.buffer_c import reset_symmetric_descriptors
from kernel_test_utils import (
    assert_dedup_plan_semantic_equal_all_ranks,
    dedup_plan_invariant_errors,
)


# ---------------------------------------------------------------------------
# deterministic per-rank data fixtures (same contract as the dispatch test)
# ---------------------------------------------------------------------------

def make_hidden(src_rank: int, token_id: int, S: int, H: int) -> torch.Tensor:
    base = float(src_rank * S + token_id)
    col = (torch.arange(H, dtype=torch.float32) % 16) / 64.0
    return (base + col).to(torch.bfloat16)


def hidden_sh_for(rank: int, S: int, H: int) -> torch.Tensor:
    return torch.stack([make_hidden(rank, t, S, H) for t in range(S)]).contiguous()


def make_weight(src_rank: int, token_id: int, topk_id: int, S: int, K: int) -> float:
    return float(src_rank * S * K + token_id * K + topk_id)


def weights_sk_for(rank: int, S: int, K: int) -> torch.Tensor:
    base = float(rank * S * K)
    return (torch.arange(S * K, dtype=torch.float32) + base).reshape(S, K).contiguous()


# ---------------------------------------------------------------------------
# CPU expected model (inverts the plan's dst encoding exactly as the kernel)
# ---------------------------------------------------------------------------

def expected_dispatch(rank, R, S, K, H, plan):
    NvS = S * K
    N = NvS
    gathered = [torch.zeros_like(plan.dst) for _ in range(R)]
    dist.all_gather(gathered, plan.dst)
    dst_by_rank = torch.stack(gathered).cpu()  # [R, N] int32

    expected_hidden = torch.zeros(NvS, H, dtype=torch.bfloat16)
    expected_weight = torch.zeros(NvS, dtype=torch.float32)
    covered = torch.zeros(NvS, dtype=torch.bool)

    for src in range(R):
        for i in range(N):
            raw = int(dst_by_rank[src, i])
            if raw >= 0:
                dest, loff = divmod(raw, NvS)
            else:
                dest, loff = divmod(-raw - 1, NvS)
            if dest == rank:
                token_id, topk_id = divmod(i, K)
                expected_hidden[loff] = make_hidden(src, token_id, S, H)
                expected_weight[loff] = make_weight(src, token_id, topk_id, S, K)
                covered[loff] = True

    n_groups = int(plan.dup_counts.cpu()[0])
    dup_groups = plan.dup_groups.cpu()
    dup_loffs = plan.dup_loffs.cpu()
    for gid in range(n_groups):
        primary, dup_start, dup_n = (int(v) for v in dup_groups[gid])
        if primary < 0 or dup_n <= 0 or dup_start < 0:
            continue
        for j in range(dup_n):
            loff = int(dup_loffs[dup_start + j])
            if 0 <= loff < NvS:
                expected_hidden[loff] = expected_hidden[primary]
                covered[loff] = True

    return expected_hidden, expected_weight, covered


CASES = [
    {"name": "balanced", "S": 64, "K": 4, "H": 64, "E": 32,
     "bias": 0.0, "weights": True},
    {"name": "biased", "S": 64, "K": 4, "H": 64, "E": 32,
     "bias": 1.0, "weights": True},
    {"name": "no_weights", "S": 32, "K": 2, "H": 128, "E": 32,
     "bias": 0.0, "weights": False},
]


@pytest.fixture(scope="module")
def dist_env():
    if "RANK" not in os.environ:
        pytest.skip("test_buffer_plan requires torchrun with at least 2 NPUs")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.npu.set_device(local_device_index(local_rank))
    owns = not dist.is_initialized()
    if owns:
        dist.init_process_group(backend="hccl", device_id=torch.device(f"npu:{local_device_index(local_rank)}"))
    R = dist.get_world_size()
    if R < 2:
        if owns:
            dist.destroy_process_group()
        pytest.skip("test_buffer_plan requires at least 2 NPUs")
    yield rank, local_rank, R
    dist.barrier()
    if owns:
        dist.destroy_process_group()


def run_case(rank, local_rank, R, case, use_udma: bool = True):
    dev = f"npu:{local_device_index(local_rank)}"
    S, H, K, E = case["S"], case["H"], case["K"], case["E"]
    bias = case["bias"]
    with_weights = case["weights"]
    seed = case.get("seed", 1234)
    assert E % R == 0 and K <= E

    buffer = None
    try:
        buffer = Buffer(S, H, K, E, R, explicitly_destroy=True, use_udma=use_udma)
        if not ShmemRuntime.is_initialized():
            init_shmem(group=None, enable_udma=use_udma)
        ctx = buffer._require_ctx()

        topk_all, tpe_all = generate_topk_routing(S, K, E, R, bias, "cpu", seed, all_ranks=True)
        topk_this = topk_all[rank].to(dev)
        tpe_this = tpe_all[rank].to(dev)

        hidden_sh = hidden_sh_for(rank, S, H).to(dev)
        route_weights_sk = weights_sk_for(rank, S, K).to(dev) if with_weights else None

        label = f"{case['name']} (R={R}, use_udma={use_udma})"

        # auto-planning path: planning kernel runs inside dispatch
        hidden_nvsh, weights_nvs, cu_seqlens, plan_out = buffer.dispatch(
            hidden_sh, route_weights_sk, plan=None,
            topk_experts_sk=topk_this, tokens_per_expert=tpe_this)
        torch.npu.synchronize()
        ShmemRuntime.barrier()

        # 1) produced plan must satisfy kernel invariants
        inv_errs = dedup_plan_invariant_errors(
            type("C", (), {"E": E, "K": K, "R": R})(), ctx, plan_out)
        assert not inv_errs, f"{label}: plan invariants: {inv_errs}"

        # 2) produced plan must match the torch planning reference (semantic)
        ref_plan, _ = reference_plan_and_cu_seqlens(ctx, topk_this, tpe_this)
        assert_dedup_plan_semantic_equal_all_ranks(
            "plan", plan_out, ref_plan, rank, R)

        # 3) dispatched shard must match the CPU decode model
        exp_h, exp_w, covered = expected_dispatch(rank, R, S, K, H, plan_out)
        assert bool(covered.all()), f"{label}: only {int(covered.sum())} slots covered"
        ok_h = torch.equal(hidden_nvsh.cpu(), exp_h)
        ok_w = True if not with_weights else torch.equal(weights_nvs.cpu(), exp_w)
        ok = bool(ok_h and ok_w)
        if not ok:
            diff = int((hidden_nvsh.cpu() != exp_h).sum().item())
            print(f"[rank {rank}] {label}: hidden={ok_h} weights={ok_w} mism={diff}")
        ok_t = torch.tensor([int(ok)], dtype=torch.int32, device=hidden_nvsh.device)
        dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
        assert int(ok_t.item()) == 1, label

        if rank == 0:
            print(f"  [PASS] {label}")
        dist.barrier()
    finally:
        ShmemRuntime.finalize()
        if buffer is not None:
            buffer.destroy()
        reset_symmetric_descriptors()


@pytest.mark.parametrize("use_udma", [True, False], ids=["urma", "mte"])
@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_buffer_plan(dist_env, case, use_udma):
    rank, local_rank, R = dist_env
    run_case(rank, local_rank, R, case, use_udma)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s"]))
