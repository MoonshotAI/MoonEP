# SPDX-License-Identifier: Apache-2.0
"""End-to-end smoke test for the public Buffer dispatch/combine API (MTE).

Ascend port of ``MoonEP/tests/test_e2e.py`` (MTE-only version): the original
``test_npu/buffer_test/test_e2e.py`` urma part has moved into
``test_npu/urma/test_e2e_urma.py``. Verifies sync/async dispatch, separate prefetch,
combine behavior, the public plan-reuse dispatch path, zero-copy aliasing, and
sync/async reduce_grad against affine fp32 references, all through the MTE
(``use_udma=False``) path.

Ascend adaptations:
  - HCCL process group + NPU devices; buffer-backed only: Buffer is constructed
    first (registers the CANN VMM descriptor), then ``init_shmem`` (the single-node
    default uses ``ShmemRuntime.init_with_buffer``, UID bootstrap).
  - Cross-rank data visibility uses the shmem barrier
    (``ShmemRuntime.barrier``) instead of ``dist.barrier``.
  - Prefetch weight pools and grad/reduce buffers are built by the public
    high-level constructors ``create_full_weight`` /
    ``create_full_grad_and_reduce_buffer`` (the moonep.buffer NPU path): they create
    the local home/slot VMM chunk + ``[E+B]`` projection and attach the source VAs; reduce_buffer
    is a pre-initialized ``[R, B, H, H']`` carrier (reduce_grad only consumes its slot source VA).
  - Assertions are local (no collective all-gathers); any rank that fails
    raises its own AssertionError.

Run with:
    torchrun --nproc_per_node=8 -m pytest -s test_npu/mte/test_e2e.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.distributed as dist

import torch_npu

from npu_patch import Buffer, MoonEPCommPlan
from kernel_test_utils import (
    clone_dedup_plan_fields,
    dedup_plan_fields_equal,
    dedup_plan_semantic_errors,
    init_shmem,
    local_device_index,
)
from moonep.buffer import (
    create_full_grad_and_reduce_buffer,
    create_full_weight,
    projection_home_slice,
    projection_slot_slice,
    release_owned,
)


def make_inputs(rank, S, H, K, E, seed=0):
    dev = "npu"
    # Deterministic round-robin routing (same as generate_topk_routing bias=0).
    g = torch.Generator(device=dev).manual_seed(seed + rank)
    hidden = torch.randn(S, H, dtype=torch.bfloat16, device=dev, generator=g)
    weights = torch.rand(S, K, dtype=torch.float32, device=dev, generator=g)
    topk = torch.randint(0, E, (S, K), dtype=torch.int32, device=dev, generator=g)
    tpe = torch.bincount(topk.flatten(), minlength=E).to(torch.int32)
    return hidden, weights, topk, tpe


def _expert_value(expert, H, Hp, dev):
    e = torch.full((H, Hp), float(expert), dtype=torch.bfloat16, device=dev)
    e += (torch.arange(Hp, dtype=torch.bfloat16, device=dev) * 0.01).view(1, Hp)
    e += (torch.arange(H, dtype=torch.bfloat16, device=dev) * 0.0001).view(H, 1)
    return e


def make_full_weight(rank, R, E, B, H, Hp, device, source_offset):
    """Return ``(full_weight, offset)``.

    full_weight is built by ``create_full_weight``: it creates this rank's home/slot
    VMM chunk and ``[E+B]`` projection, copies ``expert_table[rank*epn:(rank+1)*epn]``
    into the home rows plus offset, zeroes the slot rows, and attaches the home/slot
    source VAs (the Buffer prefetch_weight VMM path translates through them). expert_table
    is generated in full by a deterministic formula and is identical on all ranks (rank-invariant).
    """
    dev = f"npu:{device}"
    expert_table = torch.stack(
        [_expert_value(e, H, Hp, dev) for e in range(E)])
    full = create_full_weight(
        E, B, H, Hp, rank, R, expert_table, source_offset, device=device)
    return full, source_offset


def make_prefetch_args(rank, R, E, B, H, Hp, device):
    """Return ``(args_dict, meta)``.

    ``args_dict`` values are the full_weight **tensors** (spread into ``buffer.prefetch_weight(**args)``);
    ``meta[name] = (full_weight, offset)`` is for the assertions (slot rows read via
    ``projection_slot_slice``).
    """
    args = {}
    meta = {}
    for name, offset in (("full_gate_weight", 0.0),
                         ("full_up_weight", 1.0),
                         ("full_down_weight", 2.0)):
        full, off = make_full_weight(rank, R, E, B, H, Hp, device, offset)
        args[name] = full
        meta[name] = (full, off)
    return args, meta


def assert_prefetched(prefetch_arge, prefetch_meta, experts_to_copy, R):
    """``R`` is the *domain* EP size (== group world size), not the global one.

    The prefetch projection lays ``[E+B]`` global expert rows where ``E =
    epn * R``; taking ``R`` from the caller keeps this correct inside an EP
    subgroup (see ``test_npu/buffer_test/test_buffer_grouped_e2e.py``).
    """
    B = experts_to_copy.numel()
    for b in range(B):
        expert = int(experts_to_copy[b].item())
        if expert < 0:
            continue
        for name, (full, offset) in prefetch_meta.items():
            H, Hp = int(full.shape[1]), int(full.shape[2])
            expected = _expert_value(expert, H, Hp, full.device)
            if offset:
                expected = expected + offset
            slots = projection_slot_slice(full)
            assert torch.equal(slots[b], expected), (
                f"{name} prefetch buffer {b} does not match expert {expert}"
            )


def assert_prefetch_projection_readback(
    full, rank, epn, B, offset, experts_to_copy, dev,
):
    """Read back through the full_weight projection VA and verify against an independent reference (§6.4 projection data correctness).

    1) home rows [rank*epn, (rank+1)*epn) are unchanged across prefetch (local expert value + offset);
    2) slot rows [E, E+B) have been written by prefetch with remote expert value + offset.
    Reference values are generated independently by ``_expert_value`` (not read back from the slots, to avoid self-reflection), checked by exact
    ``torch.equal`` (the bf16 copy does no arithmetic, so the reference is reproduced exactly).
    """
    H, Hp = int(full.shape[1]), int(full.shape[2])
    home_start = full._projection.home_start
    for i in range(epn):
        expected = _expert_value(rank * epn + i, H, Hp, dev) + offset
        assert torch.equal(full[home_start + i], expected), (
            f"projection home row {home_start + i} changed after prefetch")
    slot_start = full._projection.slot_start
    for b in range(B):
        expert = int(experts_to_copy[b].item())
        if expert < 0:
            continue
        expected = _expert_value(expert, H, Hp, dev) + offset
        assert torch.equal(full[slot_start + b], expected), (
            f"projection slot row {slot_start + b} != expert {expert}")


def fill_prefetch_slots(prefetch_meta, B, value):
    for (full, offset) in prefetch_meta.values():
        projection_slot_slice(full).fill_(value)


def assert_prefetch_slots_equal(prefetch_meta, B, value):
    for name, (full, offset) in prefetch_meta.items():
        slots = projection_slot_slice(full)
        expected = torch.full_like(slots, value)
        assert torch.equal(slots, expected), (
            f"{name} prefetch slots changed without prefetch_weight"
        )


def assert_dedup_plan_semantic_equal(actual, expected):
    errors = dedup_plan_semantic_errors("dedup plan", actual, expected)
    assert not errors, "; ".join(errors[:5])


def grad_base(E, H, Hp, offset, dev):
    expert = torch.arange(E, dtype=torch.float32, device=dev).view(E, 1, 1)
    row = torch.arange(H, dtype=torch.float32, device=dev).view(1, H, 1)
    col = torch.arange(Hp, dtype=torch.float32, device=dev).view(1, 1, Hp)
    return offset + expert * 17.0 + row * 0.125 + col * 0.0078125


def reduce_base(rank, B, H, Hp, offset, dev):
    """This rank's [B, H, H'] reduce-slot pattern (src_rank == rank)."""
    slot = torch.arange(B, dtype=torch.float32, device=dev).view(B, 1, 1)
    row = torch.arange(H, dtype=torch.float32, device=dev).view(1, H, 1)
    col = torch.arange(Hp, dtype=torch.float32, device=dev).view(1, 1, Hp)
    return offset + (rank + 1) * 101.0 + slot * 11.0 + row * 0.03125 + col * 0.00390625


def reduce_base_all(R, B, H, Hp, offset, dev):
    """All ranks' reduce-slot patterns stacked as the rank-invariant [R,B,H,H'] table."""
    return torch.stack(
        [reduce_base(r, B, H, Hp, offset, dev) for r in range(R)])


def make_grad_reduce_args(rank, R, E, B, H, Hp, offsets, device):
    """Return ``(args_dict, home_grads, slot_chunks)``.

    full_grad and reduce_buffer are built by ``create_full_grad_and_reduce_buffer``:
    it creates the home grad chunk ([epn,H,Hp]) and the reduce slot chunk ([B,H,Hp]) as two
    local VMM chunks (fp32), the ``[E+B]`` projection and the pre-initialized ``[R,B,H,H']``
    carrier (reduce_grad only consumes its slot source VA), and attaches the source VAs. grad_table takes
    the full ``[E,H,Hp]`` affine table (rank-invariant), reduce_table the full
    ``[R,B,H,Hp]`` table; the constructors copy only this rank's home/slot slices.
    home_grads / slot_chunks are views of the projection home/slot rows (same physical memory as the chunks),
    used by ``assert_grad_reduced`` for the checks.
    """
    dev = f"npu:{device}"
    args = {}
    home_grads = []
    slot_chunks = []
    for idx, name in enumerate(["gate", "up", "down"]):
        grad_table = grad_base(E, H, Hp, offsets[idx], dev)
        reduce_table = reduce_base_all(R, B, H, Hp, offsets[3 + idx], dev)
        full_grad, rb = create_full_grad_and_reduce_buffer(
            E, B, H, Hp, rank, R, grad_table, reduce_table, device=device)
        args[f"full_{name}_grad"] = full_grad
        args[f"{name}_reduce_buffer"] = rb
        home_grads.append(projection_home_slice(full_grad))
        slot_chunks.append(projection_slot_slice(full_grad))
    return args, home_grads, slot_chunks


def expected_local_grad(rank, R, E, H, Hp, full_offset, reduce_offset, experts_to_copy):
    dev = "npu"
    expected = grad_base(E, H, Hp, full_offset, dev)
    B = experts_to_copy.shape[1]
    for src_rank in range(R):
        slot_values = reduce_base(src_rank, B, H, Hp, reduce_offset, dev)
        for b in range(B):
            expert = int(experts_to_copy[src_rank, b].item())
            if expert >= 0:
                expected[expert].add_(slot_values[b])
    epn = E // R
    return expected[rank * epn:(rank + 1) * epn].contiguous()


def assert_grad_reduced(rank, R, E, H, Hp, experts_to_copy, home_grads,
                        slot_chunks, offsets, device):
    B = E // R
    epn = E // R
    dev = f"npu:{device}"
    names = ["gate", "up", "down"]
    for idx, name in enumerate(names):
        home_grad = home_grads[idx]
        slot_chunk = slot_chunks[idx]
        full_offset = offsets[idx]
        reduce_offset = offsets[3 + idx]
        expected = expected_local_grad(
            rank, R, E, H, Hp, full_offset, reduce_offset, experts_to_copy
        )
        assert torch.allclose(
            home_grad, expected, rtol=0.0, atol=1e-5
        ), f"{name} local grad reduce mismatch"

        original = reduce_base(rank, B, H, Hp, reduce_offset, dev)
        for b in range(B):
            expert = int(experts_to_copy[rank, b].item())
            if expert >= 0:
                assert torch.equal(
                    slot_chunk[b], torch.zeros_like(slot_chunk[b]),
                ), f"{name} consumed reduce slot {b} was not cleared"
            else:
                assert torch.equal(
                    slot_chunk[b], original[b],
                ), f"{name} non-local reduce slot {b} changed"


def assert_raises_assertion(expected_substr, fn):
    try:
        fn()
    except AssertionError as exc:
        assert expected_substr in str(exc), (
            f"expected assertion containing {expected_substr!r}, got {exc!r}"
        )
    else:
        raise AssertionError(f"expected AssertionError containing {expected_substr!r}")


def test_e2e():
    """MTE-only end-to-end: the full dispatch / prefetch / combine / reduce_grad chain."""
    rank, R = setup()
    # S=128/H=64 mirrors test_npu/buffer_test/test_buffer_plan.py's default config, which is
    # stable on this hardware; larger S/H intermittently trigger a MaskedSelectV3
    # OOB (507035) in the planning kernel (see docs/planning_operator_test_notes.md).
    S, H, K, E = 4096, 256, 16, 64
    # The Ascend planning kernel requires num_prefetch_slots == E / R.
    B = E // R
    epn = E // R
    # Hp=512 makes epn*row = 8*256*512*2 = 2 MiB an exact multiple of the VMM granularity,
    # so the full_weight projection can place the home rows exactly at [rank*epn, (rank+1)*epn).
    Hp = 512
    num_sms = 32
    device = local_device_index()
    dev = f"npu:{device}"

    buffer = None
    try:
        # Buffer-backed only: construct Buffer first (registers the VMM descriptor), then init_shmem.
        # The MTE path pins use_udma=False (for urma see test_npu/urma/test_e2e_urma.py).
        buffer = Buffer(S, H, K, E, R, B=B, num_sms=num_sms, use_udma=False)

        # All full_weight / full_grad / reduce_buffer must be created before init_shmem
        # (the VMM chunks the constructors register go into the descriptor registry, and init is one-shot).
        sync_prefetch_args, sync_prefetch_meta = make_prefetch_args(
            rank, R, E, B, H, Hp, device)
        async_prefetch_args, async_prefetch_meta = make_prefetch_args(
            rank, R, E, B, H, Hp, device)
        reuse_prefetch_args, reuse_prefetch_meta = make_prefetch_args(
            rank, R, E, B, H, Hp, device)
        no_prefetch_args, no_prefetch_meta = make_prefetch_args(
            rank, R, E, B, H, Hp, device)

        grad_offsets = (1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0)
        grad_args, grad_home, grad_slots = make_grad_reduce_args(
            rank, R, E, B, H, Hp, grad_offsets, device)
        async_grad_offsets = (11000.0, 12000.0, 13000.0, 14000.0, 15000.0, 16000.0)
        async_grad_args, async_grad_home, async_grad_slots = make_grad_reduce_args(
            rank, R, E, B, H, Hp, async_grad_offsets, device)

        # The MTE path pins enable_udma=False (init_shmem is idempotent; it skips when already initialized).
        init_shmem(group=None, enable_udma=False)

        hidden, weights, topk, tpe = make_inputs(rank, S, H, K, E)

        def _dbg(step):
            if rank == 0:
                print(f"[dbg] step={step}", flush=True)

        # --- Sync reference ---
        _dbg("sync-dispatch")
        (h_sync, w_sync, cu_sync, plan_sync) = buffer.dispatch(
            hidden, weights, topk, tpe,
        )
        h_sync_cpu = h_sync.clone()
        w_sync_cpu = w_sync.clone() if w_sync is not None else None
        cu_sync_cpu = cu_sync.clone()
        assert isinstance(plan_sync, MoonEPCommPlan)
        plan_snapshot = plan_sync.clone()
        buffer.prefetch_weight(plan=plan_snapshot, **sync_prefetch_args)
        torch.npu.synchronize()
        assert_prefetched(sync_prefetch_args, sync_prefetch_meta, plan_snapshot.experts_to_copy[rank], R)
        for name, (full, offset) in sync_prefetch_meta.items():
            assert_prefetch_projection_readback(
                sync_prefetch_args[name], rank, epn, B, offset,
                plan_snapshot.experts_to_copy[rank], dev)

        _dbg("sync-prefetch")
        # --- Async variant (fresh inputs to avoid symmetric aliasing) ---
        _dbg("async-dispatch")
        hidden2, weights2, topk2, tpe2 = make_inputs(rank, S, H, K, E)
        assert torch.equal(hidden, hidden2)
        assert torch.equal(weights, weights2)

        (h_a, w_a, cu_a, plan_a, _dispatch_event) = buffer.dispatch(
            hidden2, weights2, topk2, tpe2, async_finish=True,
        )
        _dbg("async-prefetch_weight")
        prefetch_event = buffer.prefetch_weight(
            plan=plan_a, async_finish=True, **async_prefetch_args,
        )
        _dbg("wait-prefetch_weight")
        prefetch_event.wait(torch.npu.current_stream())
        h_a_snap = h_a.clone()
        w_a_snap = w_a.clone() if w_a is not None else None
        assert_prefetched(async_prefetch_args, async_prefetch_meta, plan_a.experts_to_copy[rank], R)
        for name, (full, offset) in async_prefetch_meta.items():
            assert_prefetch_projection_readback(
                async_prefetch_args[name], rank, epn, B, offset,
                plan_a.experts_to_copy[rank], dev)

        assert torch.equal(h_sync_cpu, h_a_snap), "dispatch hidden mismatch"
        assert torch.equal(w_sync_cpu, w_a_snap), "dispatch weights mismatch"
        assert torch.equal(cu_sync_cpu, cu_a), "cu_seqlens mismatch"
        # zero_fill_ranges / remote_stats are compat-only fields the Ascend
        # planning kernel does not write (cf. test_planning.py); only the real
        # plan fields must be bit-identical between sync and async runs.
        for name in ("dst", "experts_to_copy"):
            assert torch.equal(getattr(plan_snapshot, name), getattr(plan_a, name)), \
                f"plan.{name} mismatch"
        assert_dedup_plan_semantic_equal(plan_a, plan_snapshot)

        # --- Public plan-reuse path: planning is skipped. ---
        reuse_dedup_before = clone_dedup_plan_fields(plan_snapshot)
        (h_reuse, w_reuse, cu_reuse, plan_reuse) = buffer.dispatch(
            hidden, plan=plan_snapshot,
        )
        buffer.prefetch_weight(plan=plan_snapshot, **reuse_prefetch_args)
        torch.npu.synchronize()
        assert torch.equal(h_reuse.clone(), h_sync_cpu), "plan-reuse hidden mismatch"
        assert w_reuse is None, "plan-reuse hidden-only dispatch should not return weights"
        assert cu_reuse is None, "plan-reuse path should skip planning outputs"
        assert plan_reuse is plan_snapshot, "plan-reuse should echo the input plan back"
        assert dedup_plan_fields_equal(plan_snapshot, reuse_dedup_before), \
            "plan-reuse dispatch should not rebuild or mutate dedup structures"
        assert_prefetched(reuse_prefetch_args, reuse_prefetch_meta, plan_snapshot.experts_to_copy[rank], R)
        for name, (full, offset) in reuse_prefetch_meta.items():
            assert_prefetch_projection_readback(
                reuse_prefetch_args[name], rank, epn, B, offset,
                plan_snapshot.experts_to_copy[rank], dev)

        # --- Plan-reuse without prefetch: prefetch slots stay untouched. ---
        sentinel = 7.0
        fill_prefetch_slots(no_prefetch_meta, B, sentinel)
        no_prefetch_dedup_before = clone_dedup_plan_fields(plan_snapshot)
        (h_no_prefetch, w_no_prefetch, cu_no_prefetch, plan_no_prefetch) = buffer.dispatch(
            hidden, plan=plan_snapshot,
        )
        torch.npu.synchronize()
        assert torch.equal(h_no_prefetch.clone(), h_sync_cpu), "no-prefetch hidden mismatch"
        assert w_no_prefetch is None
        assert cu_no_prefetch is None
        assert dedup_plan_fields_equal(plan_snapshot, no_prefetch_dedup_before)
        assert plan_no_prefetch is plan_snapshot
        assert_prefetch_slots_equal(no_prefetch_meta, B, sentinel)

        (h_no_weights, _, _, _) = buffer.dispatch(
            hidden, plan=plan_snapshot,
        )
        torch.npu.synchronize()
        assert torch.equal(h_no_weights.clone(), h_sync_cpu), "dispatch without weights mismatch"

        (h_no_prefetch_async, _, _, _, ev_no_prefetch) = buffer.dispatch(
            hidden, plan=plan_snapshot, async_finish=True,
        )
        ev_no_prefetch.wait(torch.npu.current_stream())
        torch.npu.synchronize()
        assert torch.equal(h_no_prefetch_async.clone(), h_sync_cpu), \
            "async no-prefetch hidden mismatch"

        # --- Combine: sync vs async, with and without weight gather. ---
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
        assert torch.equal(out_sync_snap, out_with_weights), "combine weight-gather mismatch"
        assert torch.equal(gathered_weights, weights), "combine route_weights_sk gather mismatch"

        # --- zero_copy round-trip (both hidden and router weights aliased). ---
        # The two switches are independent: zero_copy governs hidden, router_weights_zero_copy governs
        # route_weights_nvs. With only the former enabled, weights are still a fresh torch.empty + copy tensor.
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

        # --- Combine + sync reduce_grad. ---
        h_for_combine, _, _, _ = buffer.dispatch(
            hidden, weights, topk, tpe,
        )
        out_grad_sync, _, _ = buffer.combine(
            plan=plan_snapshot,
            hidden_nvsh=h_for_combine,
        )
        buffer.reduce_grad(plan=plan_snapshot, **grad_args)
        torch.npu.synchronize()
        assert torch.equal(out_sync_snap, out_grad_sync), "sync reduce_grad combine mismatch"
        assert_grad_reduced(
            rank, R, E, H, Hp, plan_snapshot.experts_to_copy,
            grad_home, grad_slots, grad_offsets, device)

        # --- Combine + async reduce_grad. ---
        h_for_combine, _, _, _ = buffer.dispatch(
            hidden, weights, topk, tpe,
        )
        out_async, _, _combine_ev = buffer.combine(
            plan=plan_snapshot,
            hidden_nvsh=h_for_combine,
            async_finish=True,
        )
        reduce_ev = buffer.reduce_grad(
            plan=plan_snapshot,
            async_finish=True,
            **async_grad_args,
        )
        reduce_ev.wait(torch.npu.current_stream())
        torch.npu.synchronize()
        assert torch.equal(out_sync_snap, out_async), "combine output mismatch"
        assert_grad_reduced(
            rank, R, E, H, Hp, plan_snapshot.experts_to_copy,
            async_grad_home, async_grad_slots, async_grad_offsets, device)

        if rank == 0:
            print("[test_e2e] PASS: MTE public API sync/async, separate prefetch, and plan reuse match.")
    finally:
        # release_owned does the unified teardown: finalize first (tears down the symmetric mapping),
        # then destroy the home/slot VMM chunks and projection VAs registered by the constructors,
        # finally destroy the Buffer and clear the descriptor registry (the order must not be swapped, to avoid get_buffer_ptr failing on re-init).
        release_owned(buffers=[buffer] if buffer is not None else [])

def setup():
    # Skip if the process group was already initialized by an earlier pytest file in this process (several files in one torchrun run).
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    torch.npu.set_device(local_device_index())
    return rank, dist.get_world_size()


if __name__ == "__main__":
    test_e2e()
