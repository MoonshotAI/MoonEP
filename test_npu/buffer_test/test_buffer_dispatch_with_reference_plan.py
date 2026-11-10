"""Buffer.dispatch correctness test driven by the torch planning reference.

Structure follows ``test_grad_reduce.py`` (torchrun + pytest, collective
verify via ``all_reduce``) and ``test_moonep_dispatch_with_plan.py``
(deterministic per-rank hidden/weight fixtures, plan-driven dispatch). The
difference: the plan is built by ``planing_reference`` (the torch planning
reference), so the planning *kernel* is skipped and only the dispatch kernel
plus the public ``Buffer.dispatch`` plumbing is under test.

Each rank builds the same global routing with ``generate_topk_routing``, runs
``planing_reference.reference_plan_and_cu_seqlens`` to get a
``MoonEPCommPlan`` (collective: it all-gathers ``dst``/``tokens_per_expert``),
then calls ``buffer.dispatch(hidden_sh, route_weights_sk, plan=plan)``. The
output shard is verified against a CPU model that inverts the plan's ``dst``
encoding (``dest_rank * NvS + loff``) exactly as the dispatch kernel decodes
it, including the duplicate-row epilogue.

Run with (Ascend host, CANN + torch_npu + shmem + built npu_patch):

    torchrun --nproc_per_node=8 -m pytest -s test_npu/buffer_test/test_buffer_dispatch_with_reference_plan.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel_test_utils import init_shmem
import os
import sys

import pytest
import torch
import torch.distributed as dist

import torch_npu

sys.path.insert(0, os.path.dirname(__file__))
from generate_topk_routing import generate_topk_routing
from planing_reference import reference_plan_and_cu_seqlens

from npu_patch._device import local_device_index
from npu_patch import Buffer, ShmemRuntime
from npu_patch.buffer_c import reset_symmetric_descriptors


# ---------------------------------------------------------------------------
# deterministic per-rank data fixtures (same shape contract as
# test_moonep_dispatch_with_plan.py)
# ---------------------------------------------------------------------------

def make_hidden(src_rank: int, token_id: int, S: int, H: int) -> torch.Tensor:
    """A single [H] bf16 hidden row as a pure function of (src_rank, token_id)."""
    base = float(src_rank * S + token_id)
    col = (torch.arange(H, dtype=torch.float32) % 16) / 64.0
    return (base + col).to(torch.bfloat16)


def hidden_sh_for(rank: int, S: int, H: int) -> torch.Tensor:
    """This rank's [S, H] bf16 input tensor."""
    return torch.stack([make_hidden(rank, t, S, H) for t in range(S)]).contiguous()


def make_weight(src_rank: int, token_id: int, topk_id: int, S: int, K: int) -> float:
    """Route weight as a pure function of (src_rank, token_id, topk_id)."""
    return float(src_rank * S * K + token_id * K + topk_id)


def weights_sk_for(rank: int, S: int, K: int) -> torch.Tensor:
    """This rank's [S, K] fp32 routing weights."""
    base = float(rank * S * K)
    return (torch.arange(S * K, dtype=torch.float32) + base).reshape(S, K).contiguous()


# ---------------------------------------------------------------------------
# CPU expected model (inverts the plan's dst encoding exactly as the kernel)
# ---------------------------------------------------------------------------

def expected_dispatch(
    rank: int,
    R: int,
    S: int,
    K: int,
    H: int,
    plan,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU model of the dispatch kernel output for this rank.

    The kernel decodes every route's ``dst`` as ``dest_rank * NvS + loff``
    (negative entries ``-raw - 1`` for duplicate top-k rows) and:
      - canonical rows (raw >= 0) MTE-put the source hidden row to the remote
        shard slot and write the route weight there;
      - duplicate rows write only the weight to the slot their raw dst decodes
        to, and the plan epilogue copies the primary hidden row to them.
    Since a duplicate slot's owner route belongs to the same (src, token) as
    its primary, every slot's final hidden row is exactly
    ``make_hidden(src, token)`` of the route that owns it — so the unified
    model below matches the kernel including the epilogue.

    Returns (expected_hidden [NvS, H] bf16, expected_weight [NvS] fp32,
    covered [NvS] bool).
    """
    NvS = S * K
    N = NvS

    # Gather every rank's canonical dst (the reference already all-gathers it
    # internally; re-gather here so the expected model needs no assumption).
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

    # Duplicate-row epilogue: copy primary hidden row to the duplicate slots.
    # (Redundant with the loop above for correctness — same (src, token) — but
    # kept to mirror the kernel's epilogue exactly.)
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


def verify_dispatch(
    rank: int,
    R: int,
    S: int,
    K: int,
    H: int,
    plan,
    hidden_nvsh: torch.Tensor,
    weights_nvs: torch.Tensor | None,
    with_weights: bool,
    label: str,
) -> None:
    """Collectively verify one dispatch output against the CPU model."""
    expected_hidden, expected_weight, covered = expected_dispatch(
        rank, R, S, K, H, plan)
    assert bool(covered.all()), (
        f"rank {rank}: {label}: only {int(covered.sum())}/{covered.numel()} "
        f"output slots covered by the plan"
    )

    actual_hidden = hidden_nvsh.cpu()
    ok_hidden = torch.equal(actual_hidden, expected_hidden)
    ok_weights = True
    if with_weights:
        ok_weights = torch.equal(weights_nvs.cpu(), expected_weight)

    ok = bool(ok_hidden and ok_weights)
    if not ok:
        diff = int((actual_hidden != expected_hidden).sum().item())
        print(
            f"[rank {rank}] {label}: hidden={ok_hidden} weights={ok_weights} "
            f"mismatch_elems={diff}/{actual_hidden.numel()}"
        )
    ok_tensor = torch.tensor([int(ok)], dtype=torch.int32, device=hidden_nvsh.device)
    dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
    assert int(ok_tensor.item()) == 1, f"{label}"


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------

CASES = [
    # bias=0 -> deterministic round-robin, no duplicate rows (K distinct dest
    # ranks per token when K <= R).
    {"name": "balanced_weights", "S": 64, "K": 4, "H": 64, "E": 32,
     "bias": 0.0, "weights": True},
    {"name": "balanced_no_weights", "S": 32, "K": 2, "H": 128, "E": 32,
     "bias": 0.0, "weights": False},
    # bias=1 -> skewed routing that produces duplicate rows (epilogue active).
    {"name": "biased_weights", "S": 64, "K": 4, "H": 64, "E": 32,
     "bias": 1.0, "weights": True},
    {"name": "biased_no_weights", "S": 32, "K": 4, "H": 64, "E": 32,
     "bias": 1.0, "weights": False},
    {"name": "balanced_small_H", "S": 16, "K": 4, "H": 32, "E": 32,
     "bias": 0.0, "weights": True},
    # Production 8-card scale: S=4096, K=16, E=896, H=7168.
    # CPU reference `expected_dispatch` is O(R*S*K) Python — slow but bounded.
    {"name": "prod_4096x16_896", "S": 4096, "K": 16, "H": 7168, "E": 896,
     "bias": 0.0, "weights": True},
]


@pytest.fixture(scope="module")
def dist_env():
    if "RANK" not in os.environ:
        pytest.skip(
            "test_buffer_dispatch_with_reference_plan requires torchrun with "
            "at least 2 NPUs")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.npu.set_device(local_device_index(local_rank))
    owns_process_group = not dist.is_initialized()
    if owns_process_group:
        dist.init_process_group(
            backend="hccl", device_id=torch.device(f"npu:{local_device_index(local_rank)}"))
    rank = dist.get_rank()
    R = dist.get_world_size()
    if R < 2:
        if owns_process_group:
            dist.destroy_process_group()
        pytest.skip("test_buffer_dispatch_with_reference_plan requires at least 2 NPUs")

    yield rank, local_rank, R

    dist.barrier()
    if owns_process_group:
        dist.destroy_process_group()


def run_case(rank: int, local_rank: int, R: int, case: dict, use_udma: bool = True) -> None:
    dev = f"npu:{local_device_index(local_rank)}"
    S, H, K, E = case["S"], case["H"], case["K"], case["E"]
    bias = case["bias"]
    with_weights = case["weights"]
    seed = case.get("seed", 1234)
    assert E % R == 0, f"E={E} must be divisible by R={R}"
    assert K <= E
    assert H % 8 == 0 and H * 2 <= 128 * 1024

    # buffer-backed only: construct the Buffer first (it registers the VMM descriptors), then
    # call init_shmem (single-node defaults to init_with_buffer). use_udma selects the urma(UDMA)/mte operator.
    buffer = None
    try:
        buffer = Buffer(S, H, K, E, R, explicitly_destroy=True, use_udma=use_udma)
        if not ShmemRuntime.is_initialized():
            init_shmem(group=None, enable_udma=use_udma)
        ctx = buffer._require_ctx()
        NvS = int(ctx["NvS"])

        # Deterministic global routing; each rank passes its own slice into the
        # torch planning reference (which all-gathers the rest).
        topk_all, tpe_all = generate_topk_routing(
            S, K, E, R, bias, "cpu", seed, all_ranks=True)
        topk_this = topk_all[rank].to(dev)
        tpe_this = tpe_all[rank].to(dev)

        plan, _cu_seqlens = reference_plan_and_cu_seqlens(
            ctx, topk_this, tpe_this)

        hidden_sh = hidden_sh_for(rank, S, H).to(dev)
        route_weights_sk = (
            weights_sk_for(rank, S, K).to(dev) if with_weights else None
        )

        label = (f"{case['name']} (R={R}, S={S}, K={K}, H={H}, E={E}, "
                 f"bias={bias}, weights={with_weights})")

        # ---- sync, zero_copy=False (boundary copy path) --------------------
        hidden_nvsh, weights_nvs, cu_seqlens_out, plan_out = buffer.dispatch(
            hidden_sh, route_weights_sk, plan=plan)
        assert cu_seqlens_out is None, "plan-reuse dispatch must not run planning"
        torch.npu.synchronize()
        ShmemRuntime.barrier()
        verify_dispatch(rank, R, S, K, H, plan_out, hidden_nvsh, weights_nvs,
                        with_weights, f"{label} sync/copy")
        del hidden_nvsh, weights_nvs, cu_seqlens_out, plan_out

        # ---- sync, zero_copy=True (aliased shard views) --------------------
        # hidden and router weights each have their own switch; both must be on to get the fully aliased path.
        hidden_nvsh, weights_nvs, _cu, _plan = buffer.dispatch(
            hidden_sh, route_weights_sk, plan=plan,
            zero_copy=True, router_weights_zero_copy=True)
        torch.npu.synchronize()
        ShmemRuntime.barrier()
        assert hidden_nvsh.data_ptr() == ctx["hidden_buf_local"].data_ptr(), (
            "zero-copy hidden must alias the SHMEM shard")
        if with_weights:
            assert weights_nvs.data_ptr() == ctx["weights_buf_local"].data_ptr(), (
                "zero-copy weights must alias the SHMEM shard")
        verify_dispatch(rank, R, S, K, H, plan, hidden_nvsh, weights_nvs,
                        with_weights, f"{label} sync/zero-copy")
        del hidden_nvsh, weights_nvs, _cu, _plan

        # ---- async_finish=True (comm-stream orchestration) -----------------
        hidden_nvsh, weights_nvs, _cu, _plan, done = buffer.dispatch(
            hidden_sh, route_weights_sk, plan=plan, async_finish=True)
        done.synchronize()
        torch.npu.synchronize()
        ShmemRuntime.barrier()
        verify_dispatch(rank, R, S, K, H, plan, hidden_nvsh, weights_nvs,
                        with_weights, f"{label} async")
        del hidden_nvsh, weights_nvs, _cu, _plan, done

        if rank == 0:
            print(f"  [PASS] {case['name']}: R={R}, S={S}, K={K}, H={H}, "
                  f"E={E}, bias={bias}, weights={with_weights}")
        dist.barrier()
    finally:
        # buffer-backed only: finalize first (tear down the symmetric mappings), then the VMM undo ops,
        # then clear the registry. Parametrized cases run sequentially in one process; skipping this accumulates descriptors into the next case.
        ShmemRuntime.finalize()
        if buffer is not None:
            buffer.destroy()
        reset_symmetric_descriptors()


@pytest.mark.parametrize("use_udma", [True, False], ids=["urma", "mte"])
@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_dispatch_with_reference_plan(dist_env, case, use_udma):
    rank, local_rank, R = dist_env
    if rank == 0:
        print(f"[test_dispatch_with_reference_plan] case={case['name']} R={R} "
              f"use_udma={use_udma}")
    run_case(rank, local_rank, R, case, use_udma)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s"]))
