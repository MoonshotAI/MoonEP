# SPDX-License-Identifier: Apache-2.0
"""Two independent shmem domains on one node: grouped Buffer + grouped shmem e2e.

Implements "Path B" of the grouped-initialisation analysis (``transfer_report.md``
§9). The ``world`` ranks are split into ``world_size // group_size`` EP subgroups
and **each subgroup bootstraps its own buffer-backed shmem domain**. With

    ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ...

and the default ``MOONEP_GROUP_SIZE=2`` this is exactly "rank 0-1 and rank 2-3
each get their own Buffer + shmem + e2e".

What this covers that no other test does:

  * ``Buffer(group=...)`` + ``ShmemRuntime.init_with_buffer(group=...)`` produce a
    *domain-scoped* symmetric heap: ``aclshmem_my_pe()`` / ``aclshmem_n_pes()``
    (and the planning ``R`` the kernel walks) are the subgroup's, not the world's.
  * Cross-PE traffic stays inside the subgroup. With ``R=2`` every remote expert a
    rank prefetches lives on its single in-group peer, so an accidental use of the
    global domain would read a rank from the *other* subgroup and fail the
    deterministic ``_expert_value`` / ``expected_local_grad`` references.
  * Two shmem domains coexist on one node: their UID bootstraps are independent
    (one broadcast per subgroup), so the second domain must not disturb the first.

Device selection uses the **global** ``LOCAL_RANK`` (``local_device_index()``),
never the subgroup rank -- deriving it from ``ep_rank`` would put both subgroups
on cards 0/1 and hit the ``aclrtMallocPhysical`` 107001 failure this migration
fixed. ``_assert_disjoint_devices`` checks over the *global* group that all
``world_size`` processes hold distinct cards, so that mistake fails loudly here
instead of silently aliasing memory.

Run with:
    ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \\
        torchrun --nproc_per_node=4 -m pytest -s \\
        test_npu/buffer_test/test_buffer_grouped_e2e.py

    # two subgroups of four, out of eight cards:
    MOONEP_GROUP_SIZE=4 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\
        torchrun --nproc_per_node=8 -m pytest -s ...
"""

import os
import sys

_TEST_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_ROOT)
sys.path.insert(0, os.path.join(_TEST_ROOT, "mte"))  # for the shared e2e helpers

import torch
import torch.distributed as dist

import torch_npu

from npu_patch import Buffer, MoonEPCommPlan, ShmemRuntime

from kernel_test_utils import (
    clone_dedup_plan_fields,
    dedup_plan_fields_equal,
    init_shmem,
    local_device_index,
)
from moonep.buffer import release_owned

# Reuse the MTE e2e references verbatim: identical shapes and, apart from the
# domain size R, identical rank-parametric semantics. Keeping one copy means the
# grouped case cannot drift from the reference arithmetic the MTE case validates.
from test_e2e import (
    assert_dedup_plan_semantic_equal,
    assert_grad_reduced,
    assert_prefetch_projection_readback,
    assert_prefetch_slots_equal,
    assert_prefetched,
    assert_raises_assertion,
    fill_prefetch_slots,
    make_grad_reduce_args,
    make_inputs,
    make_prefetch_args,
)


def _setup():
    """Init the *global* process group and select this process' NPU card."""
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    torch.npu.set_device(local_device_index())
    return rank, dist.get_world_size()


def _make_subgroup(world_size, group_size, rank):
    """Create every EP subgroup on every rank, return this rank's ``(group, ep_rank, R)``.

    ``dist.new_group`` is collective: all ranks must create the same groups in the
    same order, so we build the full list everywhere and only *keep* our own.
    """
    if group_size <= 0:
        raise ValueError(f"MOONEP_GROUP_SIZE must be positive, got {group_size}")
    if world_size % group_size:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by group_size "
            f"({group_size})")
    n_groups = world_size // group_size
    if n_groups < 2:
        raise ValueError(
            f"need at least 2 subgroups to exercise domain isolation, got "
            f"{n_groups} (world_size={world_size}, group_size={group_size})")
    mine = None
    ep_rank = None
    for g in range(n_groups):
        ranks = list(range(g * group_size, (g + 1) * group_size))
        pg = dist.new_group(ranks=ranks)
        if rank in ranks:
            mine, ep_rank = pg, ranks.index(rank)
    assert mine is not None and ep_rank is not None
    return mine, ep_rank, group_size


def _assert_disjoint_devices(world_size, device, dev):
    """Every process must hold a distinct card (global all-gather, once)."""
    local = torch.tensor([int(device)], dtype=torch.int32, device=dev)
    gathered = [torch.zeros_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    ids = [int(t.item()) for t in gathered]
    assert sorted(ids) == list(range(world_size)), (
        f"processes must hold distinct NPU cards, got device ids {ids} for "
        f"world_size={world_size}. Select the device from the *global* LOCAL_RANK "
        "via local_device_index(), never from the subgroup rank/ep_rank -- two "
        "subgroups would otherwise both land on cards 0..group_size-1."
    )
    return ids


def test_grouped_e2e():
    """Two subgroups, each with its own Buffer + shmem domain: full e2e."""
    rank, world_size = _setup()
    group_size = int(os.environ.get("MOONEP_GROUP_SIZE", "2"))
    device = local_device_index()
    dev = f"npu:{device}"
    group, ep_rank, R = _make_subgroup(world_size, group_size, rank)
    device_ids = _assert_disjoint_devices(world_size, device, dev)

    # S/H mirror test_npu/mte/test_e2e.py (stable on this hardware; larger S/H can
    # trip the planning kernel's MaskedSelectV3 OOB, see the operator test notes).
    S, H, K = 4096, 256, 16
    epn = 8
    E = R * epn
    # The Ascend planning kernel requires num_prefetch_slots == E / R.
    B = E // R
    # epn * H * Hp * 2B == 2 MiB == the VMM granule, so the full_weight projection
    # can place the home segment exactly at global expert rows [ep_rank*epn, ...).
    Hp = 512
    num_sms = 32

    print(
        f"[grouped-e2e rank {rank}] world={world_size} group_size={group_size} "
        f"ep_rank={ep_rank}/{R} device=npu:{device} devices={device_ids} "
        f"S={S} H={H} K={K} E={E} B={B} Hp={Hp}",
        flush=True,
    )

    buffer = None

    # Guard *before* anything is allocated: the shmem runtime is a process-global
    # singleton, and this test needs a clean one so that the domain it bootstraps
    # is exclusively its own (a leaked initialization from an earlier test would
    # make init_shmem() a silent no-op and leave our chunks unregistered).
    if ShmemRuntime.is_initialized():
        raise RuntimeError(
            "test_buffer_grouped_e2e bootstraps its own shmem domain and needs a "
            "clean runtime; run it as its own torchrun job, or make sure the "
            "preceding test finalised shmem.")

    try:
        # Buffer first: it registers this subgroup's VMM descriptors. The shmem
        # domain is bootstrapped after every chunk exists (init is one-shot).
        buffer = Buffer(
            S, H, K, E, R, B=B, num_sms=num_sms, group=group, use_udma=False,
        )

        # All full_weight / full_grad / reduce_buffer must be created before init_shmem
        # (the VMM chunks registered by the constructor have to reach the descriptor registry; init is one-shot).
        sync_pargs, sync_pmeta = make_prefetch_args(
            ep_rank, R, E, B, H, Hp, device)
        async_pargs, async_pmeta = make_prefetch_args(
            ep_rank, R, E, B, H, Hp, device)
        reuse_pargs, reuse_pmeta = make_prefetch_args(
            ep_rank, R, E, B, H, Hp, device)
        nopre_pargs, nopre_pmeta = make_prefetch_args(
            ep_rank, R, E, B, H, Hp, device)

        grad_offsets = (1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0)
        grad_args, grad_home, grad_slots = make_grad_reduce_args(
            ep_rank, R, E, B, H, Hp, grad_offsets, device)
        async_grad_offsets = (11000.0, 12000.0, 13000.0, 14000.0, 15000.0, 16000.0)
        async_grad_args, async_grad_home, async_grad_slots = make_grad_reduce_args(
            ep_rank, R, E, B, H, Hp, async_grad_offsets, device)

        # Bootstrap *this subgroup's* domain: UID generated and broadcast inside
        # the group, so n_pes() == R and my_pe() == ep_rank.
        init_shmem(group=group, enable_udma=False)

        hidden, weights, topk, tpe = make_inputs(ep_rank, S, H, K, E)

        # --- sync dispatch + prefetch ----------------------------------------
        (h_sync, w_sync, cu_sync, plan_sync) = buffer.dispatch(
            hidden, weights, topk, tpe,
        )
        h_sync_cpu = h_sync.clone()
        w_sync_cpu = w_sync.clone() if w_sync is not None else None
        cu_sync_cpu = cu_sync.clone()
        assert isinstance(plan_sync, MoonEPCommPlan)
        plan_snapshot = plan_sync.clone()
        buffer.prefetch_weight(plan=plan_snapshot, **sync_pargs)
        torch.npu.synchronize()
        assert_prefetched(
            sync_pargs, sync_pmeta, plan_snapshot.experts_to_copy[ep_rank], R)
        for name, (full, offset) in sync_pmeta.items():
            assert_prefetch_projection_readback(
                sync_pargs[name], ep_rank, epn, B, offset,
                plan_snapshot.experts_to_copy[ep_rank], dev)

        # --- async dispatch + prefetch (fresh inputs, no symmetric aliasing) ---
        hidden2, weights2, topk2, tpe2 = make_inputs(ep_rank, S, H, K, E)
        assert torch.equal(hidden, hidden2)
        assert torch.equal(weights, weights2)
        (h_a, w_a, cu_a, plan_a, _dispatch_event) = buffer.dispatch(
            hidden2, weights2, topk2, tpe2, async_finish=True,
        )
        prefetch_event = buffer.prefetch_weight(
            plan=plan_a, async_finish=True, **async_pargs,
        )
        prefetch_event.wait(torch.npu.current_stream())
        h_a_snap = h_a.clone()
        w_a_snap = w_a.clone() if w_a is not None else None
        assert_prefetched(
            async_pargs, async_pmeta, plan_a.experts_to_copy[ep_rank], R)
        for name, (full, offset) in async_pmeta.items():
            assert_prefetch_projection_readback(
                async_pargs[name], ep_rank, epn, B, offset,
                plan_a.experts_to_copy[ep_rank], dev)

        assert torch.equal(h_sync_cpu, h_a_snap), "dispatch hidden mismatch"
        assert torch.equal(w_sync_cpu, w_a_snap), "dispatch weights mismatch"
        assert torch.equal(cu_sync_cpu, cu_a), "cu_seqlens mismatch"
        # zero_fill_ranges / remote_stats are compat-only fields the Ascend
        # planning kernel does not write; only the real plan fields must match.
        for name in ("dst", "experts_to_copy"):
            assert torch.equal(getattr(plan_snapshot, name), getattr(plan_a, name)), \
                f"plan.{name} mismatch"
        assert_dedup_plan_semantic_equal(plan_a, plan_snapshot)

        # --- public plan-reuse path (planning skipped) ------------------------
        reuse_dedup_before = clone_dedup_plan_fields(plan_snapshot)
        (h_reuse, w_reuse, cu_reuse, plan_reuse) = buffer.dispatch(
            hidden, plan=plan_snapshot,
        )
        buffer.prefetch_weight(plan=plan_snapshot, **reuse_pargs)
        torch.npu.synchronize()
        assert torch.equal(h_reuse.clone(), h_sync_cpu), "plan-reuse hidden mismatch"
        assert w_reuse is None, "hidden-only dispatch should not return weights"
        assert cu_reuse is None, "plan-reuse path should skip planning outputs"
        assert plan_reuse is plan_snapshot, "plan-reuse should echo the input plan"
        assert dedup_plan_fields_equal(plan_snapshot, reuse_dedup_before), \
            "plan-reuse dispatch must not rebuild or mutate dedup structures"
        assert_prefetched(
            reuse_pargs, reuse_pmeta, plan_snapshot.experts_to_copy[ep_rank], R)
        for name, (full, offset) in reuse_pmeta.items():
            assert_prefetch_projection_readback(
                reuse_pargs[name], ep_rank, epn, B, offset,
                plan_snapshot.experts_to_copy[ep_rank], dev)

        # --- plan-reuse without prefetch: slots stay untouched ----------------
        sentinel = 7.0
        fill_prefetch_slots(nopre_pmeta, B, sentinel)
        nopre_dedup_before = clone_dedup_plan_fields(plan_snapshot)
        (h_nopre, w_nopre, cu_nopre, plan_nopre) = buffer.dispatch(
            hidden, plan=plan_snapshot,
        )
        torch.npu.synchronize()
        assert torch.equal(h_nopre.clone(), h_sync_cpu), "no-prefetch hidden mismatch"
        assert w_nopre is None
        assert cu_nopre is None
        assert plan_nopre is plan_snapshot
        assert dedup_plan_fields_equal(plan_snapshot, nopre_dedup_before)
        assert_prefetch_slots_equal(nopre_pmeta, B, sentinel)

        # --- combine: sync / with weights / async -----------------------------
        h_for_combine, w_for_combine, _, _ = buffer.dispatch(
            hidden, weights, topk, tpe,
        )
        out_sync, _, _ = buffer.combine(plan=plan_snapshot, hidden_nvsh=h_for_combine)
        out_sync_snap = out_sync.clone()
        torch.npu.synchronize()

        h_for_combine, w_for_combine, _, _ = buffer.dispatch(
            hidden, weights, topk, tpe,
        )
        out_with_weights, gathered_weights, _ = buffer.combine(
            plan=plan_snapshot,
            hidden_nvsh=h_for_combine,
            route_weights_nvs=w_for_combine,
        )
        torch.npu.synchronize()
        assert torch.equal(out_sync_snap, out_with_weights), \
            "combine weight-gather mismatch"
        assert torch.equal(gathered_weights, weights), \
            "combine route_weights_sk gather mismatch"

        # --- zero_copy round-trip (hidden and router weights both aliased) -----
        # Independent switches: zero_copy covers hidden, router_weights_zero_copy
        # covers route_weights_nvs. Enabling only the first still copies weights.
        h_zc, w_zc, _, plan_zc = buffer.dispatch(
            hidden, weights, topk, tpe,
            zero_copy=True,
            router_weights_zero_copy=True,
        )
        assert h_zc.data_ptr() == buffer._require_ctx()['hidden_buf_local'].data_ptr(), \
            "dispatch(zero_copy=True) must return the symmetric shard view"
        assert w_zc.data_ptr() == buffer._require_ctx()['weights_buf_local'].data_ptr(), \
            "dispatch(router_weights_zero_copy=True) must return the weights shard view"
        assert torch.equal(h_zc, h_for_combine), "zero_copy dispatch hidden mismatch"
        assert torch.equal(w_zc, w_for_combine), "zero_copy dispatch weights mismatch"
        out_zc, gathered_weights_zc, _ = buffer.combine(
            plan=plan_zc,
            hidden_nvsh=h_zc,
            route_weights_nvs=w_zc,
            zero_copy=True,
            router_weights_zero_copy=True,
        )
        torch.npu.synchronize()
        assert torch.equal(out_sync_snap, out_zc), "zero_copy combine hidden mismatch"
        assert torch.equal(gathered_weights_zc, weights), \
            "zero_copy combine route_weights_sk gather mismatch"
        assert_raises_assertion(
            "alias",
            lambda: buffer.combine(
                plan=plan_zc,
                hidden_nvsh=h_for_combine,
                zero_copy=True,
            ),
        )

        # --- combine + sync reduce_grad ---------------------------------------
        h_for_combine, _, _, _ = buffer.dispatch(hidden, weights, topk, tpe)
        out_grad_sync, _, _ = buffer.combine(
            plan=plan_snapshot, hidden_nvsh=h_for_combine,
        )
        buffer.reduce_grad(plan=plan_snapshot, **grad_args)
        torch.npu.synchronize()
        assert torch.equal(out_sync_snap, out_grad_sync), \
            "sync reduce_grad combine mismatch"
        assert_grad_reduced(
            ep_rank, R, E, H, Hp, plan_snapshot.experts_to_copy,
            grad_home, grad_slots, grad_offsets, device)

        # --- combine + async reduce_grad --------------------------------------
        h_for_combine, _, _, _ = buffer.dispatch(hidden, weights, topk, tpe)
        out_async, _, _combine_ev = buffer.combine(
            plan=plan_snapshot, hidden_nvsh=h_for_combine, async_finish=True,
        )
        reduce_ev = buffer.reduce_grad(
            plan=plan_snapshot, async_finish=True, **async_grad_args,
        )
        reduce_ev.wait(torch.npu.current_stream())
        torch.npu.synchronize()
        assert torch.equal(out_sync_snap, out_async), "combine output mismatch"
        assert_grad_reduced(
            ep_rank, R, E, H, Hp, plan_snapshot.experts_to_copy,
            async_grad_home, async_grad_slots, async_grad_offsets, device)

        print(
            f"[grouped-e2e rank {rank}] PASS: subgroup {ep_rank}/{R} "
            f"(device npu:{device}) ran dispatch/prefetch/combine/reduce_grad "
            f"e2e in its own shmem domain.",
            flush=True,
        )
    finally:
        # Same release order as test_npu/mte/test_e2e.py, via release_owned:
        # settle pending device work first, then finalize the domain (tear down
        # symmetric mappings), destroy the constructor-registered home/slot VMM
        # chunks + projection VAs and the Buffer, and reset the registry.
        torch.npu.synchronize()
        release_owned(buffers=[buffer] if buffer is not None else [])


if __name__ == "__main__":
    test_grouped_e2e()
