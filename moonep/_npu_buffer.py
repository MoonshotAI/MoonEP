"""
Ascend-only part of ``moonep.buffer``'s public surface.

This module is NPU-only: it is imported exclusively from the ``if _NPU`` binding
block at the bottom of ``moonep.buffer`` (which re-exports the NPU-only names
defined here), so ``npu_patch`` / ``torch_npu`` are imported once at module
level and a pure-GPU process never loads this module at all (``npu_patch``
itself never imports ``moonep``, so there is no import cycle).

Every name here is NPU-only — pool builders, point-visit cross-rank accessors
and ``release_owned``; ``moonep.buffer`` publishes them only on the Ascend
backend, and a GPU process never sees them (a GPU test builds the equivalent
tensors with plain ``torch.empty`` slices). See each docstring for the home/slot
VMM layout it builds or reads. The GPU NVLink allocation helpers
(``create_nvl_dist_tensor``, ``create_nvl_single_owner_tensor``,
``pad_dim0_for_alignment``) are deliberately not adapted: no NPU code path calls
them.

Builders (``create_full_weight`` / ``create_full_grad_and_reduce_buffer``)
allocate and register the CANN VMM descriptors *before*
``ShmemRuntime.init_with_buffer`` runs — init consumes the frozen descriptor
set, so chunks created afterwards never enter the current symmetric heap. The
point-visit accessors (``remote_home_rows`` / ``read_reduce_slot`` /
``expert_pool_row``) instead require init to have run already.
"""

from __future__ import annotations

import torch
import torch_npu

from npu_patch import ShmemRuntime, buffer_c


def _require_npu():
    if not ShmemRuntime.is_initialized():
        raise RuntimeError(
            "moonep.buffer (NPU): buffer-backed shmem heap is not initialized. "
            "Construct all Buffers first, then call "
            "moonep.init_shmem() / ShmemRuntime.init_with_buffer()."
        )
    return None


def _current_device() -> int:
    return int(torch.npu.current_device())


# ---------------------------------------------------------------------------
# Public pool / reduce-buffer builders (NPU-only names).
#
# These are the constructs the framework/test feeds straight into
# ``Buffer.prefetch_weight`` / ``Buffer.reduce_grad``. They build the registered
# home/slot VMM chunks, form the [E+B] projection view, and attach the source-VA
# tags the ascend Buffer consumes — so the framework never needs to touch VMM
# home/slot or ``attach_local_source_vas`` itself.
# ---------------------------------------------------------------------------


def _tag_layout(tensor, local_rank, num_ep_ranks, num_home, num_slots):
    """Record the container layout on ``tensor`` (``_moonep_*`` tags)."""
    tensor._moonep_local_rank = int(local_rank)
    tensor._moonep_num_ep_ranks = int(num_ep_ranks)
    tensor._moonep_num_home = int(num_home)
    tensor._moonep_num_slots = int(num_slots)
    return tensor


# ---------------------------------------------------------------------------
# Owned-resource registry.
#
# Every VMM chunk / projection the pool & reduce-buffer builders hand out is
# tracked here, so ``release_owned(buffers=(...))`` can tear everything down
# without the caller collecting projections and chunks by hand. Teardown order
# (finalize -> projection -> VMM inverse ops -> reset) stays in ``release_owned``.
# ---------------------------------------------------------------------------

_OWNED_RESOURCES: list = []


def _track_owned(*tensors) -> None:
    for t in tensors:
        if t is not None:
            _OWNED_RESOURCES.append(t)


def _take_owned() -> list:
    owned = list(_OWNED_RESOURCES)
    _OWNED_RESOURCES.clear()
    return owned


def _home_slot_projection(E, B, R, H, Hp, dtype, device, local_rank):
    epn = int(E) // int(R)
    home, _, _ = buffer_c.create_vmm_physical([epn, H, Hp], dtype, device)
    slot, _, _ = buffer_c.create_vmm_physical([B, H, Hp], dtype, device)
    full = buffer_c.create_full_projection_tensor(
        home, slot, int(E), int(B), int(local_rank), device)
    buffer_c.attach_local_source_vas(full, home.data_ptr(), slot.data_ptr())
    full._npu_home_chunk = home
    full._npu_slot_chunk = slot
    _track_owned(full, home, slot)
    return full, home, slot


def create_full_weight(
    num_experts: int,
    slot_rows: int,
    hidden: int,
    hidden_prime: int,
    local_rank: int,
    num_ep_ranks: int,
    expert_table: torch.Tensor,
    source_offset: float = 0.0,
    group=None,
    dtype: torch.dtype = torch.bfloat16,
    device: int | None = None,
) -> torch.Tensor:
    """Backend-specific high-level constructor for one ``[E+B]`` weight pool.

    Only this rank's home rows and local prefetch slots have physical backing
    in the returned projection. Remote expert rows are observable after shmem
    initialization through :func:`remote_home_rows` / :func:`expert_pool_row`
    (bind-free point-visit).
    """
    E, B, R = int(num_experts), int(slot_rows), int(num_ep_ranks)
    rank = int(local_rank)
    epn = E // R
    device = _current_device() if device is None else int(device)
    pool, home, slot = _home_slot_projection(
        E, B, R, int(hidden), int(hidden_prime), dtype, device, rank)
    home.copy_(expert_table[rank * epn:(rank + 1) * epn])
    if source_offset:
        home.add_(source_offset)
    slot.zero_()
    pool._npu_expert_reference = expert_table
    pool._npu_reference_offset = source_offset
    return _tag_layout(pool, rank, R, E, B)


def create_full_grad_and_reduce_buffer(
    num_experts: int,
    slot_rows: int,
    hidden: int,
    hidden_prime: int,
    local_rank: int,
    num_ep_ranks: int,
    grad_table: torch.Tensor,
    reduce_table: torch.Tensor,
    group=None,
    dtype: torch.dtype = torch.float32,
    device: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create one local full-grad projection and its pre-init reduce carrier.

    ``grad_table`` is the rank-invariant ``[E, H, H']`` reference; this rank
    copies only its home slice. ``reduce_table`` is ``[R, B, H, H']``; this
    rank copies only its own slot slice. Cross-rank slot reads go through
    :func:`read_reduce_slot` (bind-free point-visit).
    """
    E, B, R = int(num_experts), int(slot_rows), int(num_ep_ranks)
    rank = int(local_rank)
    epn = E // R
    H, Hp = int(hidden), int(hidden_prime)
    device = _current_device() if device is None else int(device)

    full_grad, home, slot = _home_slot_projection(
        E, B, R, H, Hp, dtype, device, rank)
    home.copy_(grad_table[rank * epn:(rank + 1) * epn])
    slot.copy_(reduce_table[rank])
    # Pre-init [R,B,H,H'] carrier. Its storage is not read by reduce_grad; only
    # the attached slot source VA is consumed. read_reduce_slot() resolves each
    # rank's live slot chunk from it.
    reduce_buffer = torch.empty(
        R, B, H, Hp, dtype=dtype, device=torch.device(f"npu:{device}"))
    buffer_c.attach_local_source_vas(
        reduce_buffer, home.data_ptr(), slot.data_ptr())
    reduce_buffer._npu_slot_chunk = slot
    _track_owned(reduce_buffer)
    return (_tag_layout(full_grad, rank, R, E, B),
            _tag_layout(reduce_buffer, rank, R, E, B))


def projection_home_slice(projection: torch.Tensor) -> torch.Tensor:
    """Directly readable local home slice of an ``[E+B]`` projection."""
    epn = int(projection._moonep_num_home) // int(projection._moonep_num_ep_ranks)
    start = int(projection._projection.home_start)
    return projection[start:start + epn]


def projection_slot_slice(projection: torch.Tensor) -> torch.Tensor:
    """Directly readable local slot slice of an ``[E+B]`` projection."""
    B = int(projection._moonep_num_slots)
    start = int(projection._projection.slot_start)
    return projection[start:start + B]


# ---------------------------------------------------------------------------
# Point-visit cross-rank accessors (no pre-binding required).
#
# Each call resolves one peer's symmetric address (``buffer_c.remote_view``) and
# returns a live zero-copy view. Both require buffer-backed shmem init; callers
# synchronize before reading so peer writes are visible.
# ---------------------------------------------------------------------------


def remote_home_rows(projection: torch.Tensor, pe: int) -> torch.Tensor:
    """Live view of rank ``pe``'s home chunk ``[E/R, H, H']``."""
    _require_npu()
    return buffer_c.remote_view(projection._npu_home_chunk, int(pe))


def read_reduce_slot(reduce_buffer: torch.Tensor, pe: int) -> torch.Tensor:
    """Live view of rank ``pe``'s reduce slot chunk ``[B, H, H']``.

    Bind-free cross-rank read for the reduce carrier: resolves ``pe``'s
    address from the attached slot chunk's source VA on demand. Accepts the
    pre-init ``[R,B,...]`` carrier or a bare slot chunk — anything carrying
    ``_npu_slot_chunk`` (or being the chunk itself).
    """
    _require_npu()
    chunk = getattr(reduce_buffer, "_npu_slot_chunk", None)
    if chunk is None:
        chunk = reduce_buffer
    return buffer_c.remote_view(chunk, int(pe))


# ---------------------------------------------------------------------------
# Pool read-back accessor.
#
# The Ascend kernels read each rank's *own* home chunk through the symmetric
# mapping, so a rank only ever materialises the rows it owns; the [E+B] view is
# the projection whose home/slot segments alias the local chunks. Read-back for
# remote experts resolves the owner's chunk on demand (point-visit) after shmem
# init; pre-init it falls back to the rank-invariant reference table captured
# by ``create_full_weight``.
# ---------------------------------------------------------------------------


def expert_pool_row(pool, expert):
    """Read expert ``expert`` from the owner's home chunk.

    Resolution order: locally owned row (home chunk) → live point-visit view of
    the owner's chunk (requires shmem init, no binding) → rank-invariant
    reference table (pre-init fallback captured by :func:`create_full_weight`).
    """
    rank = int(pool._moonep_local_rank)
    R = int(pool._moonep_num_ep_ranks)
    epn = int(pool._moonep_num_home) // R
    expert = int(expert)
    owner, local = expert // epn, expert % epn
    if owner == rank:
        return pool._npu_home_chunk[local]
    if ShmemRuntime.is_initialized():
        return buffer_c.remote_view(pool._npu_home_chunk, owner)[local]
    row = pool._npu_expert_reference[expert]
    offset = getattr(pool, "_npu_reference_offset", 0.0)
    return row + offset if offset else row


def release_owned(projections=(), vmm_tensors=(), buffers=()):
    """Tear down buffer-backed shmem heap + caller-owned VMM/projection resources.

    Teardown order (must not be inverted):
      1. ``ShmemRuntime.finalize()`` --- unmap the symmetric heap first. Freeing /
         reusing the backing VMM physical memory while shmem still holds mappings
         to those addresses would be a use-after-free hazard.
      2. Destroy the VMM ``_vmm_allocation`` objects (reverse VMM op) and destroy
         ``_projection`` handles. Physical memory is owned by the caller; it may
         only be freed after the symmetric mapping is gone.
      3. ``reset_symmetric_descriptors()`` --- clear the process-wide descriptor
         registry so the next case can re-init with fresh Buffers/chunks.

    ``projections`` are the ``[E+B]`` full/full_grad projection tensors (each may
    carry a ``_projection`` handle); ``vmm_tensors`` are the home/slot VMM chunks
    (their ``_vmm_allocation``); ``buffers`` are ``Buffer`` objects. Dedup of a
    shared ``VmmAllocation`` is idempotent (same allocation destroyed once).
    Everything ``create_full_weight`` / ``create_full_grad_and_reduce_buffer``
    allocated is merged in automatically (see ``_OWNED_RESOURCES``), so callers
    normally only pass ``buffers``; explicit ``projections`` / ``vmm_tensors``
    still work.

    This name only exists on the Ascend backend. A GPU process has no shmem
    runtime / VMM heap to unwind, so it destroys each ``Buffer`` directly and
    never calls a release helper.
    """
    if ShmemRuntime.is_initialized():
        ShmemRuntime.finalize()

    owned = _take_owned()
    owned_projections = [
        t for t in owned if getattr(t, "_projection", None) is not None]
    owned_chunks = [
        t for t in owned if getattr(t, "_vmm_allocation", None) is not None]

    destroyed_projections = set()
    for t in list(projections) + owned_projections:
        proj = getattr(t, "_projection", None)
        if proj is None or id(proj) in destroyed_projections:
            continue
        destroyed_projections.add(id(proj))
        try:
            proj.destroy()
        except Exception:  # best-effort teardown
            pass

    destroyed = set()
    for t in list(vmm_tensors) + owned_chunks:
        alloc = getattr(t, "_vmm_allocation", None)
        if alloc is None or id(alloc) in destroyed:
            continue
        destroyed.add(id(alloc))
        try:
            alloc.destroy()
        except Exception:  # best-effort teardown
            pass

    for b in buffers:
        if not b.destroyed:
            try:
                b.destroy()
            except Exception:  # best-effort teardown
                pass

    buffer_c.reset_symmetric_descriptors()
