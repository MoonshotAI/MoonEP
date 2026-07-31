"""Remote expert grad reduce correctness test.

Run with:
    torchrun --nproc_per_node=8 -m pytest -s tests/test_grad_reduce.py

Coverage axes:
  - shapes: single tile, multi tile, production 7168x3072 (>2^31-element
    expert offsets), epn=1, num_sms=1.
  - plans: ring (default), high fan-in (one expert receives 2R slots,
    > pipeline STAGES, with self-sends and same-src duplicates), empty
    (all -1), asymmetric (a rank that only sends, a rank that only
    receives), random multi-seed.
  - data: affine (all fp32 adds are exact, so expectations are
    order-independent) and randn (adds round, pinning the kernel's
    rb-ascending accumulation order bitwise). randn relies on torch's
    Philox generator producing identical bits for the same seed on every
    rank, which holds on homogeneous nodes.
  - lifecycle: repeated launches on a single Buffer (barrier slots
    and the cooperative grid counter must self-reset between launches).
"""

import os

import pytest
import torch
import torch.distributed as dist

from moonep import Buffer
from moonep.buffer import create_nvl_dist_tensor, pad_dim0_for_alignment
from moonep.grad_reduce import launch_grad_reduce
from tests.kernel_test_utils import local_device_index


# --------------------------------------------------------------------------
# plan builders. Plans are global [R, B] tables, so every builder must be a
# deterministic function of (R, B, epn, seed) only — never of the local rank.
# --------------------------------------------------------------------------


def _plan_ring(R, B, epn, dev, seed=0):
    plan = torch.full((R, B), -1, dtype=torch.int32)
    for src_rank in range(R):
        dst_rank = (src_rank + 1) % R
        plan[src_rank, 0] = dst_rank * epn + (src_rank % epn)
        if B > 1:
            dst_rank = (src_rank + 2) % R
            plan[src_rank, 1] = dst_rank * epn + ((src_rank + 1) % epn)
        if B > 2:
            dst_rank = (src_rank + R - 1) % R
            plan[src_rank, 2] = dst_rank * epn + ((src_rank + 2) % epn)
        if B > 3:
            # target the TOP of the dst rank's expert range so the highest
            # global expert ids (deepest gmem offsets) are always exercised.
            dst_rank = (src_rank + 1) % R
            plan[src_rank, 3] = dst_rank * epn + (epn - 1 - (src_rank % epn))
    return plan.to(dev)


def _plan_fan_in(R, B, epn, dev, seed=0):
    """One expert per dst rank receives 2 slots from EVERY rank.

    nslot = 2R per active expert exceeds the load pipeline depth (STAGES=3),
    forcing producer/consumer stage wraparound within a single work item.
    Also covers self-sends (src == dst) and same-src duplicate experts.
    """
    assert 2 * R <= B, f"fan_in plan needs B >= 2R, got B={B}, R={R}"
    plan = torch.full((R, B), -1, dtype=torch.int32)
    for dst in range(R):
        expert = dst * epn + (dst % epn)
        for src in range(R):
            plan[src, 2 * dst] = expert
            plan[src, 2 * dst + 1] = expert
    return plan.to(dev)


def _plan_empty(R, B, epn, dev, seed=0):
    return torch.full((R, B), -1, dtype=torch.int32).to(dev)


def _plan_asymmetric(R, B, epn, dev, seed=0):
    """Rank 0 is never a dst (send-only: total_work=0, clear-only); rank 1
    never sends (recv-only: nothing to clear). Both must still pass the
    cross-rank barrier in lockstep with the busy ranks."""
    plan = torch.full((R, B), -1, dtype=torch.int32)
    for src in range(R):
        if src == 1:
            continue
        dst = src + 1 if src + 1 < R else 1
        plan[src, 0] = dst * epn + (src % epn)
        if B > 1:
            plan[src, 1] = dst * epn + ((src + 1) % epn)
    return plan.to(dev)


def _plan_random(R, B, epn, dev, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    live = torch.rand((R, B), generator=g) < 0.6
    ids = torch.randint(0, R * epn, (R, B), generator=g, dtype=torch.int32)
    plan = torch.where(live, ids, torch.full_like(ids, -1))
    return plan.to(dev)


PLAN_BUILDERS = {
    "ring": _plan_ring,
    "fan_in": _plan_fan_in,
    "empty": _plan_empty,
    "asymmetric": _plan_asymmetric,
    "random": _plan_random,
}


# --------------------------------------------------------------------------
# data generators. slot values must be regenerable bitwise-identically on
# every rank (the reference rebuilds remote ranks' slots locally); grads are
# purely local. The affine patterns keep every fp32 add exact (all values
# fit in 24 mantissa bits), so affine expectations hold under ANY
# accumulation order; randn sums do round and pin the rb-ascending order.
# --------------------------------------------------------------------------


def _grad_values(E, H, Hp, dev):
    expert = torch.arange(E, dtype=torch.float32, device=dev).view(E, 1, 1)
    row = torch.arange(H, dtype=torch.float32, device=dev).view(1, H, 1)
    col = torch.arange(Hp, dtype=torch.float32, device=dev).view(1, 1, Hp)
    return expert * 17.0 + row * 0.125 + col * 0.0078125


def _slot_values(rank, B, H, Hp, dev):
    slot = torch.arange(B, dtype=torch.float32, device=dev).view(B, 1, 1)
    row = torch.arange(H, dtype=torch.float32, device=dev).view(1, H, 1)
    col = torch.arange(Hp, dtype=torch.float32, device=dev).view(1, 1, Hp)
    return (rank + 1) * 101.0 + slot * 11.0 + row * 0.03125 + col * 0.00390625


def _randn(shape, seed, dev):
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    return torch.randn(shape, generator=g, device=dev, dtype=torch.float32)


def _make_data_fns(mode, rank, R, E, B, H, Hp, dev, seed):
    """Return (grads_fn, slot_fn). Both regenerate identical bits per call."""
    if mode == "affine":
        return (
            lambda: _grad_values(E, H, Hp, dev),
            lambda src: _slot_values(src, B, H, Hp, dev),
        )
    assert mode == "randn", mode
    return (
        lambda: _randn((E, H, Hp), 200_000 + 1_000 * seed + rank, dev),
        lambda src: _randn((B, H, Hp), 100_000 + 1_000 * seed + src, dev),
    )


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------


def _expected_for_rank(rank, R, E, B, plan_cpu, grads_fn, slot_fn):
    expected = grads_fn()
    for src in range(R):
        if not bool((plan_cpu[src] >= 0).any()):
            continue
        slot_vals = slot_fn(src)
        for b in range(B):
            expert = int(plan_cpu[src, b])
            if expert >= 0:
                expected[expert].add_(slot_vals[b])
    epn = E // R
    return expected[rank * epn : (rank + 1) * epn].contiguous()


def _assert_all_ranks(ok, rank, label):
    ok_tensor = torch.tensor([int(ok)], dtype=torch.int32, device="cuda")
    dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
    assert int(ok_tensor.item()) == 1, label


def _verify(rank, R, E, B, plan, grads_fn, slot_fn,
            remote_expert_grads, reduce_buffers, label):
    epn = E // R
    plan_cpu = plan.cpu()
    local_start = rank * epn
    local_end = local_start + epn

    expected_local = _expected_for_rank(rank, R, E, B, plan_cpu, grads_fn, slot_fn)
    actual_local = remote_expert_grads[local_start:local_end]
    # Bitwise: the kernel accumulates each element serially in rb-ascending
    # slot order seeded from the local grad value — the same fp32 add
    # sequence as _expected_for_rank. Affine cases are exact regardless of
    # order; the randn cases round and so genuinely pin that order. Loosen
    # to allclose if the kernel ever reorders accumulation (atomics,
    # multi-CTA slot splits, ...).
    ok_grads = torch.equal(actual_local, expected_local)

    pristine = grads_fn()
    ok_nonlocal = torch.equal(
        remote_expert_grads[:local_start], pristine[:local_start]
    ) and torch.equal(remote_expert_grads[local_end:], pristine[local_end:])

    zero_ok = True
    unchanged_ok = True
    for src in range(R):
        slot_vals = None
        for b in range(B):
            expert = int(plan_cpu[src, b])
            if expert >= 0:
                if not torch.equal(reduce_buffers[src, b],
                                   torch.zeros_like(reduce_buffers[src, b])):
                    zero_ok = False
            else:
                if slot_vals is None:
                    slot_vals = slot_fn(src)
                if not torch.equal(reduce_buffers[src, b], slot_vals[b]):
                    unchanged_ok = False

    ok = bool(ok_grads and ok_nonlocal and zero_ok and unchanged_ok)
    if not ok:
        print(
            f"[rank {rank}] {label}: grads={ok_grads} nonlocal={ok_nonlocal} "
            f"zero={zero_ok} unchanged={unchanged_ok}"
        )
    _assert_all_ranks(ok, rank, label)


# --------------------------------------------------------------------------
# cases
# --------------------------------------------------------------------------

CASES = [
    {
        "name": "single_tile",
        "epn": 4,
        "H": 128,
        "Hp": 128,
        "base_B": 4,
        "num_sms": 8,
    },
    {
        "name": "multi_tile",
        "epn": 4,
        "H": 256,
        "Hp": 128,
        "base_B": 5,
        "num_sms": 16,
    },
    # Production-like expert shape where expert_id * H * Hp crosses 2^31
    # elements (expert ids >= 98 at 7168x3072). Catches 32-bit offset
    # arithmetic in the kernel's gmem addressing (would fault or corrupt).
    {
        "name": "i64_offset_7168x3072",
        "epn": 32,
        "H": 7168,
        "Hp": 3072,
        "base_B": 4,
        "num_sms": 32,
    },
    # nslot = 2R > STAGES on a single expert: pipeline wraparound within one
    # work item, plus self-sends and same-src duplicate experts.
    {
        "name": "fan_in_gt_stages",
        "epn": 4,
        "H": 256,
        "Hp": 128,
        "base_B": 8,
        "num_sms": 8,
        "plan": "fan_in",
    },
    # All -1: total_work=0 and zero consumed slots on every rank. The kernel
    # must cross both barriers without touching any memory.
    {
        "name": "empty_plan",
        "epn": 4,
        "H": 128,
        "Hp": 128,
        "base_B": 4,
        "num_sms": 8,
        "plan": "empty",
    },
    # Rank 0 only sends (accumulates nothing), rank 1 only receives (clears
    # nothing): barriers must hold under maximally unbalanced work.
    {
        "name": "asymmetric",
        "epn": 4,
        "H": 256,
        "Hp": 128,
        "base_B": 4,
        "num_sms": 16,
        "plan": "asymmetric",
    },
    # Degenerate extents: one expert per rank (EPN=1 prescan bounds, and the
    # ring plan then gives nslot=4 > STAGES) on a single cooperative CTA.
    {
        "name": "epn1_num_sms1",
        "epn": 1,
        "H": 128,
        "Hp": 128,
        "base_B": 4,
        "num_sms": 1,
    },
    # randn data on the known ring plan: first case where fp32 adds round,
    # so the bitwise check pins the kernel's accumulation order.
    {
        "name": "ring_randn",
        "epn": 4,
        "H": 256,
        "Hp": 128,
        "base_B": 5,
        "num_sms": 16,
        "data": "randn",
        "seed": 7,
    },
    # Random plans (holes, self-sends, duplicates, uneven fan-in) at
    # R*B=64 (two 32-lane prescan chunks at R=4) with rounding randn data.
    {
        "name": "random_plan_s1",
        "epn": 8,
        "H": 256,
        "Hp": 128,
        "base_B": 16,
        "num_sms": 16,
        "plan": "random",
        "data": "randn",
        "seed": 1,
    },
    {
        "name": "random_plan_s2",
        "epn": 8,
        "H": 256,
        "Hp": 128,
        "base_B": 16,
        "num_sms": 16,
        "plan": "random",
        "data": "randn",
        "seed": 2,
    },
    {
        "name": "random_plan_s3",
        "epn": 8,
        "H": 256,
        "Hp": 128,
        "base_B": 16,
        "num_sms": 16,
        "plan": "random",
        "data": "randn",
        "seed": 3,
    },
]


@pytest.fixture(scope="module")
def dist_env():
    if "RANK" not in os.environ:
        pytest.skip("test_grad_reduce requires torchrun with at least 2 GPUs")

    owns_process_group = not dist.is_initialized()
    if owns_process_group:
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(local_device_index())
    R = dist.get_world_size()
    if R < 2:
        if owns_process_group:
            dist.destroy_process_group()
        pytest.skip("test_grad_reduce requires at least 2 GPUs")

    yield rank, R

    dist.barrier(device_ids=[local_device_index()])
    if owns_process_group:
        dist.destroy_process_group()


def run_case(rank, R, case):
    dev = "cuda"
    epn = case["epn"]
    E = R * epn
    H = case["H"]
    Hp = case["Hp"]
    B = pad_dim0_for_alignment([case["base_B"], H, Hp], torch.float32)
    num_sms = case["num_sms"]
    seed = case.get("seed", 0)

    reduce_full = create_nvl_dist_tensor([B, H, Hp], torch.float32, rank, R)
    reduce_buffers = reduce_full.view(R, B, H, Hp)
    plan = PLAN_BUILDERS[case.get("plan", "ring")](R, B, epn, dev, seed)
    grads_fn, slot_fn = _make_data_fns(
        case.get("data", "affine"), rank, R, E, B, H, Hp, dev, seed)

    buffer = Buffer(S=128, H=H, K=1, E=E, num_ep_ranks=R, num_sms=num_sms, B=B)
    try:
        # standalone launch_grad_reduce needs the cross-rank barrier handles
        # built by Buffer; the production path passes them in from inside
        # Buffer.reduce_grad.
        ctx = buffer._require_ctx()

        remote_expert_grads = grads_fn().contiguous()
        reduce_buffers[rank].copy_(slot_fn(rank))
        torch.cuda.synchronize()
        dist.barrier(device_ids=[local_device_index()])

        launch_grad_reduce(
            remote_expert_grads,
            reduce_buffers,
            plan,
            rank=rank,
            num_sms=num_sms,
            meta_buf=ctx['meta_buf'],
            meta_stride=ctx['meta_chunk_padded'],
            barrier_off=ctx['BARRIER_OFF'],
            grid_sync_bar=ctx['grid_sync_bar'],
        )
        torch.cuda.synchronize()
        dist.barrier(device_ids=[local_device_index()])

        _verify(rank, R, E, B, plan, grads_fn, slot_fn,
                remote_expert_grads, reduce_buffers,
                f"{case['name']} grad_reduce failed")

        if rank == 0:
            print(
                f"  [PASS] {case['name']}: R={R}, E={E}, B={B}, "
                f"H={H}, Hp={Hp}, num_sms={num_sms}"
            )

        dist.barrier(device_ids=[local_device_index()])
    finally:
        buffer.destroy()


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_grad_reduce_case(dist_env, case):
    rank, R = dist_env
    if rank == 0:
        print(f"[test_grad_reduce] Running case={case['name']} with R={R}")
    run_case(rank, R, case)


def test_grad_reduce_repeated_launch(dist_env):
    """Three launches on ONE Buffer: the cross-rank barrier slots and
    the cooperative grid counter must self-reset between launches (the
    production pattern is one launch per training step). The middle round is
    an empty plan so a zero-work launch is also proven not to wedge or
    desync the barrier state for the round after it."""
    rank, R = dist_env
    dev = "cuda"
    epn, H, Hp, base_B, num_sms = 4, 256, 128, 5, 16
    E = R * epn
    B = pad_dim0_for_alignment([base_B, H, Hp], torch.float32)

    reduce_full = create_nvl_dist_tensor([B, H, Hp], torch.float32, rank, R)
    reduce_buffers = reduce_full.view(R, B, H, Hp)

    buffer = Buffer(S=128, H=H, K=1, E=E, num_ep_ranks=R, num_sms=num_sms, B=B)
    try:
        # standalone launch_grad_reduce needs the cross-rank barrier handles
        # built by Buffer; the production path passes them in from inside
        # Buffer.reduce_grad.
        ctx = buffer._require_ctx()

        rounds = [("random", 11), ("empty", 0), ("random", 12)]
        for it, (plan_name, plan_seed) in enumerate(rounds):
            plan = PLAN_BUILDERS[plan_name](R, B, epn, dev, plan_seed)
            grads_fn, slot_fn = _make_data_fns(
                "randn", rank, R, E, B, H, Hp, dev, seed=50 + it)

            remote_expert_grads = grads_fn()
            reduce_buffers[rank].copy_(slot_fn(rank))
            torch.cuda.synchronize()
            dist.barrier(device_ids=[local_device_index()])

            launch_grad_reduce(
                remote_expert_grads,
                reduce_buffers,
                plan,
                rank=rank,
                num_sms=num_sms,
                meta_buf=ctx['meta_buf'],
                meta_stride=ctx['meta_chunk_padded'],
                barrier_off=ctx['BARRIER_OFF'],
                grid_sync_bar=ctx['grid_sync_bar'],
            )
            torch.cuda.synchronize()
            dist.barrier(device_ids=[local_device_index()])

            _verify(rank, R, E, B, plan, grads_fn, slot_fn,
                    remote_expert_grads, reduce_buffers,
                    f"repeated_launch round {it} ({plan_name}) failed")

        if rank == 0:
            print(f"  [PASS] repeated_launch: R={R}, rounds={len(rounds)}")
        dist.barrier(device_ids=[local_device_index()])
    finally:
        buffer.destroy()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s"]))
