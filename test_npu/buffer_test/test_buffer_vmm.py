"""
End-to-end Buffer test for VMM mode (buffer-backed shmem init).

Covers the new paths of the analysis doc §4:
  - Buffer construction does **not** initialize shmem (the communication chunks are CANN VMM,
    registered into the global descriptor registry by ``create_vmm_physical``);
  - ``init_shmem`` (single node defaults to ``ShmemRuntime.init_with_buffer``, UID bootstrap)
    uses these descriptors for one buffer-backed symmetric heap initialization;
  - ``Buffer.prefetch_weight`` / ``reduce_grad`` read the local source VA attached by the framework
    (``_moonep_owner_source_va`` / ``_moonep_slot_source_va``), and use
    ``get_buffer_ptr`` to translate it into local [epn]/[B] shmem tensors for the operators.

Run with:
    ASCEND_RT_VISIBLE_DEVICES=8,9 \
    torchrun --nproc-per-node=8 -m pytest -s test_npu/buffer_test/test_buffer_vmm.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel_test_utils import init_shmem
import os

import pytest
import torch
import torch.distributed as dist
import torch_npu

from npu_patch import Buffer, ShmemRuntime
from npu_patch._device import local_device_index
from moonep.buffer import (
    create_full_grad_and_reduce_buffer,
    create_full_weight,
    projection_home_slice,
    projection_slot_slice,
    release_owned,
)
from npu_patch.planning import allocate_planning_outputs


def _expert_value(expert, H, Hp, dev, dtype=torch.bfloat16):
    gen = torch.Generator(device=dev).manual_seed(2026 + expert)
    return torch.randn(H, Hp, dtype=dtype, device=dev, generator=gen)


def _prefetch_plan(R, B, epn, rank, dev):
    """Rank ``rank`` pulls the experts owned by ``(rank+1) % R`` (remote)."""
    plan = torch.full((R, B), -1, dtype=torch.int32, device=dev)
    remote = (rank + 1) % R
    for e in range(min(epn, B)):
        plan[rank, e] = remote * epn + e
    return plan


@pytest.fixture(scope="module")
def dist_env():
    if "RANK" not in os.environ:
        pytest.skip("test_buffer_vmm requires torchrun with >= 2 NPUs")
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    R = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.npu.set_device(local_device_index(local_rank))
    if R < 2:
        dist.destroy_process_group()
        pytest.skip("test_buffer_vmm requires at least 2 NPUs")
    yield rank, R
    # Do not destroy the process group: several pytest files then run sequentially in one torchrun
    # (the next file's init_process_group would hang once it has been destroyed). shmem is finalized
    # by each case's own finally.
    del rank, R


def _make_weight(rank, R, epn, B, E, H, Hp, dev):
    """Build the ``[E+B]`` VMM projection full_weight via ``create_full_weight``,
    returning ``(full_weight, home_chunk, slot_chunk)``.

    expert_table is generated in full by ``_expert_value`` (seed=2026+expert) and its content is
    identical on every rank (rank-invariant); the constructor creates this rank's home/slot VMM
    chunks (registered for buffer-backed init), copies the matching expert rows into the home
    segment, zeroes the slot segment and sets up the source VA. The real home/slot row indices come
    from ``_projection.home_start``/``slot_start`` (this file uses only gran-aligned Hp, home lands on global
    expert rows ``[rank*epn, …)``; a non-gran-multiple epn*row raises ValueError outright -- there is no "unaligned fallback layout").
    """
    device = local_device_index(int(os.environ.get("LOCAL_RANK", rank)))
    expert_table = torch.stack(
        [_expert_value(e, H, Hp, dev) for e in range(E)])
    full = create_full_weight(
        E, B, H, Hp, rank, R, expert_table, 0.0, device=device)
    # Projection alias check: two address identities for the same physical memory (projection VA vs chunk source VA).
    home_start = full._projection.home_start
    assert torch.equal(
        full[home_start:home_start + epn], full._npu_home_chunk), \
        "projection home segment must alias home chunk"
    slot_start = full._projection.slot_start
    assert torch.equal(
        full[slot_start:slot_start + B], full._npu_slot_chunk), \
        "projection slot segment must alias slot chunk"
    return full, full._npu_home_chunk, full._npu_slot_chunk


@pytest.mark.parametrize("use_udma", [True, False], ids=["urma", "mte"])
# The [E+B] projection build (including the internal path of create_full_weight) strictly requires
# epn*row to be an integer multiple of VMM gran: a non-multiple raises ValueError directly, there is no
# "unaligned fallback layout" (see the gran check in buffer_c.py). An unaligned_epn_row case with
# Hp=512 used to sit here; it errored on that constraint and was removed.
@pytest.mark.parametrize("Hp", [1024], ids=["gran_aligned"])
def test_vmm_prefetch(dist_env, Hp, use_udma):
    rank, R = dist_env
    dev = f"npu:{local_device_index(int(os.environ.get('LOCAL_RANK', rank)))}"
    # Hp=1024 → epn*row=2MiB(bf16), a gran multiple → the projection's global expert-row layout.
    epn, B, H = 2, 2, 512
    E = R * epn
    S, K = 32, 1

    # VMM mode: construct Buffer while shmem is not initialized (communication chunks registered as descriptors).
    buffer = Buffer(S, H, K, E, R, B=B, num_sms=8, use_udma=use_udma)
    ctx = buffer._require_ctx()
    assert ctx.get('vmm_chunks'), "Buffer must run in VMM mode"

    plan, _ = allocate_planning_outputs(ctx)
    plan.experts_to_copy.copy_(_prefetch_plan(R, B, epn, rank, dev))

    weights = [_make_weight(rank, R, epn, B, E, H, Hp, dev) for _ in range(3)]

    # buffer-backed symmetric heap initialization (collective).
    if not ShmemRuntime.is_initialized():
        init_shmem(group=None, enable_udma=use_udma)

    try:
        buffer.prefetch_weight(
            plan=plan,
            full_gate_weight=weights[0][0],
            full_up_weight=weights[1][0],
            full_down_weight=weights[2][0],
        )
        torch.npu.synchronize()
        dist.barrier()

        remote = (rank + 1) % R
        ok = True
        for weight_index, (full, home, slot) in enumerate(weights):
            for b in range(B):
                expected = _expert_value(remote * epn + b, H, Hp, dev)
                if not torch.equal(slot[b], expected):
                    ok = False
                    diff = (slot[b].float() - expected.float()).abs().max().item()
                    print(
                        f"[rank {rank}] weight={weight_index} slot={b} "
                        f"expected remote expert {remote * epn + b}, "
                        f"max_abs_diff={diff}")
        ok_t = torch.tensor([int(ok)], dtype=torch.int32, device=dev)
        dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
        assert ok_t.item() == 1, "VMM prefetch produced mismatched slots"
    finally:
        # Clean up even on failure so the next case in the same process is not polluted.
        # release_owned unified teardown: finalize first (tears down the symmetric mapping), then destroy the
        # constructor-registered projection VA and home/slot VMM allocations, then the Buffer and the registry.
        release_owned(buffers=[buffer] if buffer is not None else [])
    if rank == 0:
        print(f"  [PASS] test_vmm_prefetch: R={R}, E={E}, B={B}, H={H}, Hp={Hp}")


@pytest.mark.parametrize("use_udma", [True, False], ids=["urma", "mte"])
# Same as above: keep only the Hp values satisfying epn*row % gran == 0.
@pytest.mark.parametrize("Hp", [1024], ids=["gran_aligned"])
def test_vmm_reduce_grad_launch(dist_env, Hp, use_udma):
    """Launch validation of the reduce_grad VMM path (counterpart of test_buffer_grad_reduce.py)."""
    rank, R = dist_env
    device = local_device_index(int(os.environ.get("LOCAL_RANK", rank)))
    dev = f"npu:{device}"
    # Hp=1024 → epn*row=4MiB(fp32), a gran multiple → the projection's global expert-row layout.
    epn, B, H = 2, 2, 512
    E = R * epn
    S, K = 32, 1

    buffer = None
    try:
        buffer = Buffer(S, H, K, E, R, B=B, num_sms=8, use_udma=use_udma)
        ctx = buffer._require_ctx()

        plan, _ = allocate_planning_outputs(ctx)
        plan.experts_to_copy.copy_(_prefetch_plan(R, B, epn, rank, dev))

        # Local home grad chunk [epn] fp32 + local reduce slot chunk [B] fp32,
        # created together by create_full_grad_and_reduce_buffer (initialized from the zero table).
        # The case requires home grad content rank+i (rank-dependent, no rank-invariant table can express it), so
        # it is overwritten via the projection home segment after creation -- same physical memory as the home chunk.
        # reduce_buffer (the [R,B,H,Hp] pre-init carrier) only carries the slot source VA; the
        # kernel does not dereference it.
        zero_grad = torch.zeros(E, H, Hp, dtype=torch.float32, device=dev)
        zero_reduce = torch.zeros(R, B, H, Hp, dtype=torch.float32, device=dev)
        full_grads = []
        for i in range(3):
            full, rb = create_full_grad_and_reduce_buffer(
                E, B, H, Hp, rank, R, zero_grad, zero_reduce, device=device)
            projection_home_slice(full).fill_(float(rank + i))
            full_grads.append((full, rb))

        if not ShmemRuntime.is_initialized():
            init_shmem(group=None, enable_udma=use_udma)

        buffer.reduce_grad(
            plan=plan,
            full_gate_grad=full_grads[0][0], full_up_grad=full_grads[1][0],
            full_down_grad=full_grads[2][0],
            gate_reduce_buffer=full_grads[0][1], up_reduce_buffer=full_grads[1][1],
            down_reduce_buffer=full_grads[2][1],
        )
        torch.npu.synchronize()
        dist.barrier()
    finally:
        # release_owned unified teardown: finalize first (tears down the symmetric mapping), then destroy the
        # constructor-registered projection VA and home/slot VMM allocations, then the Buffer and the registry.
        release_owned(buffers=[buffer] if buffer is not None else [])
    if rank == 0:
        print(f"  [PASS] test_vmm_reduce_grad_launch: R={R}, E={E}, B={B}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s"]))
