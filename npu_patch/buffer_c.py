"""Symmetric-memory buffer methods backed by the C API (drop-in replacement
for the shmem Python API version in buffer.py).

Uses the ``acl_*`` interfaces of ``npu_patch._C`` (a self-built C extension
sharing one libshmem.so instance with _C) instead of the shmem Python package
(``shmem.aclshmem_create_tensor`` / ``core.*``). Methods share names and
signatures with ``buffer.py``; callers only change the import.

  - ``shmem_finalize``: buffer-backed heap teardown
  - ``shmem_init_with_buffer`` / ``shmem_init_with_ip_port``: buffer-backed
    bootstrap (UID / ip_port)
  - ``create_sym_dist_tensor``: allocate one symmetric chunk

Difference from the Python-package version: RAII -- the deleter runs
``aclshmem_free`` automatically when the tensor is destroyed, no manual
``core.free``.
"""

from __future__ import annotations

import math

import torch

from ._common import ACLSHMEM_DATA_OP_MTE
from ._device import local_physical_id

from ._C import (
    acl_dist_alloc,
    acl_shmem_finalize,
    acl_shmem_ptr,
    acl_shmem_getmem,
    # CANN VMM (aclrt* physical alloc + shareable VA)
    acl_vmm_granularity,
    acl_vmm_reserve,
    acl_vmm_release,
    acl_vmm_map,
    acl_vmm_unmap,
    acl_vmm_set_access,
    acl_vmm_alloc_source,
    acl_vmm_free_source,
    # shmem buffer-heap (PR-653)
    acl_shmem_get_buffer_ptr,
    acl_shmem_init_with_buffers,
    acl_shmem_init_with_buffers_ip_port,
)

_ELEM_SIZE = {
    torch.float32: 4,
    torch.bfloat16: 2,
    torch.int32: 4,
    torch.uint8: 1,
}

# Caller-owned descriptor backing is pinned from successful buffer-backed init
# until successful collective finalize.  A failed finalize keeps this latch set
# so Python teardown cannot invalidate handles still referenced by ACLSHMEM.
_SHMEM_BACKING_LOCKED = False


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _align_up(value: int, alignment: int) -> int:
    """Round ``value`` up to a multiple of ``alignment``."""
    return (value + alignment - 1) // alignment * alignment


def _lcm(a: int, b: int) -> int:
    """Least common multiple of two positive integers."""
    return a // math.gcd(a, b) * b


def _chunk_nbytes(chunk_shape, dtype: torch.dtype) -> int:
    nbytes = _numel(chunk_shape) * _ELEM_SIZE[dtype]
    assert nbytes > 0, "chunk must have at least one element"
    return nbytes


# ---------------------------------------------------------------------------
# shmem finalize (init lives in shmem_runtime / the *_with_buffer helpers)
# ---------------------------------------------------------------------------

def shmem_finalize() -> int:
    """Finalize the shmem symmetric heap and return the native status code.

    A non-zero return means native teardown did not complete.  Callers must
    retain every registered VMM allocation/handle and treat the process as
    still initialized; freeing caller-owned backing after such a failure is a
    use-after-free hazard for ACLSHMEM.
    """
    global _SHMEM_BACKING_LOCKED
    rc = int(acl_shmem_finalize())
    if rc == 0:
        _SHMEM_BACKING_LOCKED = False
    return rc


def shmem_init_with_ip_port(my_pe: int, n_pes: int, ip_port: str,
                            descriptors, tail_bytes: int = 0,
                            data_op_engine_type: int = ACLSHMEM_DATA_OP_MTE,
                            device_id: int | None = None,
                            enable_udma: bool | None = None, qp_count: int = 0) -> None:
    """Buffer-backed shmem symmetric-heap init via ip_port/DEFAULT bootstrap
    (collective).

    Equivalent to the shmem package's ``aclshmem_init(InitAttr(ip_port=...))``
    but buffer-backed (PR-653 ``aclshmemx_init_attr_with_buffers``; see
    ``shmem_init_with_buffer`` for the descriptor layout). This is the
    multi-node mode: all ranks pass the same ``ip_port``
    (``"tcp://<rank0_ip>:<port>"`` or ``"tcp6://[ipv6]:<port>"``); rank0
    (my_pe=0) starts the config store server and the other ranks connect as
    clients, so no dist-broadcast UID is needed and the symmetric space can
    span nodes. Non-zero ``tail_bytes`` makes shmem reserve a private malloc
    heap.

    ``my_pe`` / ``n_pes`` must match ``aclshmem_my_pe()`` /
    ``aclshmem_n_pes()`` (callers pass rank/world_size from dist).
    """
    global _SHMEM_BACKING_LOCKED
    if device_id is None:
        device_id = int(torch.npu.current_device())
    # Match the UID-backed path's ownership transaction: seal the registry
    # before descriptor conversion/native collectives and roll the seal back
    # only when initialization did not succeed.
    was_locked = _SHMEM_BACKING_LOCKED
    _SHMEM_BACKING_LOCKED = True
    try:
        native_descriptors = [
            (int(addr), int(size), int(mh),
             fb if isinstance(fb, bytes) else bytes(fb or b""))
            for (addr, size, mh, fb) in descriptors
        ]
        rc = acl_shmem_init_with_buffers_ip_port(
            int(my_pe), int(n_pes), int(tail_bytes),
            int(data_op_engine_type), ip_port, native_descriptors,
            int(device_id), bool(enable_udma), int(qp_count))
    except Exception:
        if not was_locked:
            _SHMEM_BACKING_LOCKED = False
        raise
    if rc != 0:
        if not was_locked:
            _SHMEM_BACKING_LOCKED = False
        raise RuntimeError(
            f"acl_shmem_init_with_buffers_ip_port failed with {rc}")


# ---------------------------------------------------------------------------
# Symmetric chunk allocation (same names/signatures as buffer.py)
# ---------------------------------------------------------------------------

def create_sym_dist_tensor(chunk_shape, dtype: torch.dtype, device: int) -> torch.Tensor:
    """Allocate one symmetric chunk and wrap it as an NPU tensor.

    ``tensor.data_ptr()`` is the local symmetric base (used for peer
    addressing / cross-rank access).

    ``device`` is kept only for signature compatibility (``acl_dist_alloc``'s
    from_blob uses the current NPU device; callers should already have run
    ``torch.npu.set_device``).

    RAII: ``aclshmem_free`` runs automatically when the tensor is destroyed.
    Allocation is collective: every rank allocates in the same order and size.
    """
    _chunk_nbytes(chunk_shape, dtype)
    keepalive, _, _ = acl_dist_alloc(shape=list(chunk_shape), dtype=dtype)
    return keepalive


# ---------------------------------------------------------------------------
# Point-visit cross-rank access.
#
# These resolve ONE peer's address at a time -- the same ``aclshmem_ptr``
# address identity an eagerly mapped ``[world_size]``-strided view would hand
# out at index ``pe`` -- so callers can read remote data without any
# pre-binding step. Both require buffer-backed shmem init (the VA ->
# symmetric-address translation only exists after init).
# ---------------------------------------------------------------------------


def sym_ptr_for_pe(source_va, pe: int) -> int:
    """Resolve ``pe``'s symmetric address for a registered local source VA.

    Two-step translation: ``get_buffer_ptr`` maps the registered VA to this
    PE's local symmetric address, then ``aclshmem_ptr`` resolves the same
    symmetric object as dereferenceable for ``pe``. The result equals the
    address ``acl_dist_map``'s ``[world_size]``-strided view exposes at index
    ``pe`` (``ptr0 = local - my_pe*stride`` arithmetic), so tensors built from
    it alias that view bit-for-bit.
    """
    local_sym = map_to_sym_ptr(int(source_va), "sym_ptr_for_pe")
    return int(acl_shmem_ptr(local_sym, int(pe)))


def remote_view(source_tensor: torch.Tensor, pe: int) -> torch.Tensor:
    """Live single-chunk view of ``pe``'s copy of a registered VMM chunk.

    The returned tensor wraps ``sym_ptr_for_pe(source_tensor, pe)`` — a pure
    address alias, no copy and no ownership. Device reads through it observe
    ``pe``'s memory at execution time, exactly like ``view[pe]`` of an
    ``acl_dist_map`` strided view; caller-side stream/barrier synchronization
    decides which peer writes are visible. Keeps ``source_tensor`` alive.
    """
    ptr = sym_ptr_for_pe(source_tensor.data_ptr(), pe)
    view = create_sym_tensor_from_ptr(
        ptr, list(source_tensor.shape), source_tensor.dtype,
        source_tensor.device)
    view._keepalive = source_tensor
    return view


def read_remote(source_tensor: torch.Tensor, pe: int,
                out: torch.Tensor | None = None,
                stream: int | None = None) -> torch.Tensor:
    """Snapshot ``pe``'s copy of a registered VMM chunk into a local tensor.

    Stream-ordered RMA get (``aclshmemx_getmem_on_stream``); NOT quieted —
    callers synchronize ``stream`` before consuming the result. ``stream``
    defaults to the current NPU stream's raw handle. Returns ``out`` (a fresh
    local tensor when omitted).
    """
    import torch_npu

    local_sym = map_to_sym_ptr(source_tensor.data_ptr(), "read_remote")
    if out is None:
        out = torch.empty(
            list(source_tensor.shape), dtype=source_tensor.dtype,
            device=source_tensor.device)
    if stream is None:
        stream = int(torch_npu.npu.current_stream().npu_stream)
    acl_shmem_getmem(
        out.data_ptr(), local_sym,
        out.numel() * out.element_size(), int(pe), int(stream))
    return out


# ---------------------------------------------------------------------------
# CANN VMM (aclrt* physical alloc + shareable VA) -- prerequisite for
# buffer-backed shmem init
# ---------------------------------------------------------------------------

def _require_current_device(device_id: int) -> None:
    """Fail fast when the VMM target device is not the current device.

    Contract: symmetric chunks are always allocated on the process's current
    device (each rank calls ``torch.npu.set_device`` first). ACL itself only
    requires the device to have been selected, but that is invisible on the
    Python side -- allocating on a device this process never selected only
    produces an opaque 107001, so give an actionable error here.
    """
    current = int(torch.npu.current_device())
    if int(device_id) != current:
        raise ValueError(
            f"VMM allocation targets device {int(device_id)}, but the current "
            f"device is {current}: call torch.npu.set_device({int(device_id)}) "
            "first (aclrtMallocPhysical needs the device to have been selected)")


class VmmAllocation:
    """A caller-owned CANN VMM physical allocation mapped at a unique source VA.

    Counterpart of ``moonep.ascend_vmm.AscendPhysicalAllocation``:
      - ``source_va``        : mapped VA registered into the symmetric arena
                               (registered once)
      - ``mem_handle``       : raw ``aclrtMallocPhysical`` handle (aka
                               ``owned_handle``)
      - ``shareable_handle`` : legacy V1 shareable handle exported for the
                               descriptor
      - ``fabric_handle``    : Fabric V2 handle (None on the legacy-V1 path)
      - ``size``             : size in bytes
    ``destroy()`` releases in dependency order: unmap -> release ->
    free_physical (the VMM inverse operations, not ``aclshmem_free``).
    """

    def __init__(self, source_va, mem_handle, shareable_handle, size,
                 fabric_handle_bytes=b"", device_id=0):
        self.source_va = int(source_va)
        self.mem_handle = mem_handle
        self.shareable_handle = int(shareable_handle) if shareable_handle else None
        # Raw bytes of the 64B aclrtMemFabricHandle (Ascend950); shmem
        # init_with_buffers consumes it directly without re-exporting. Empty
        # on 910/legacy.
        self.fabric_handle_bytes = bytes(fabric_handle_bytes or b"")
        self.size = int(size)
        self.device_id = int(device_id)
        self._destroyed = False
        self._registered = False

    @property
    def owned_handle(self):
        """Alias used by the arena descriptor (same as ``mem_handle``)."""
        return self.mem_handle

    def destroy(self) -> None:
        if self._destroyed:
            return
        if self._registered and _SHMEM_BACKING_LOCKED:
            raise RuntimeError(
                "cannot release VMM descriptor backing before a successful "
                "collective shmem finalize")
        acl_vmm_free_source(self.source_va, self.mem_handle)
        self._destroyed = True


# Global VMM descriptor registry: collects every VMM range of this process
# that must enter the symmetric space (Buffer communication chunks + op
# workspace + framework full tensors) before the framework's single
# ``init_with_buffer``.
_VMM_REGISTRY: list[VmmAllocation] = []


def register_symmetric_descriptor(allocation: VmmAllocation) -> None:
    """Register a VMM range for buffer-backed symmetric-space init."""
    if _SHMEM_BACKING_LOCKED:
        raise RuntimeError(
            "cannot register a symmetric descriptor after buffer-backed "
            "shmem initialization; it would not belong to the active heap")
    _VMM_REGISTRY.append(allocation)
    allocation._registered = True


def symmetric_descriptors():
    """Return ``(addr, size, mem_handle, fabric_bytes)`` for every live VMM
    range.

    Matches the descriptor format of ``acl_shmem_init_with_buffers``: mem_handle
    0 with empty fabric_bytes means VA-only (shmem reserves the mapping handle
    from addr); non-empty fabric_bytes is consumed directly by shmem (no
    re-export). Already ``destroy()``-ed allocations are skipped.
    """
    return tuple(
        (a.source_va, a.size,
         int(a.mem_handle) if a.mem_handle else 0,
         a.fabric_handle_bytes)
        for a in _VMM_REGISTRY
        if not getattr(a, '_destroyed', False)
    )


def reset_symmetric_descriptors() -> None:
    """Clear the global VMM descriptor registry (reset between test cases).

    shmem init is one-shot per process; when multiple cases reuse one process
    (pytest modules), ``destroy_active_buffers`` calls this after finalize so
    the next case can init again with the new Buffer's VMM chunks.
    """
    if _SHMEM_BACKING_LOCKED:
        raise RuntimeError(
            "cannot reset symmetric descriptors before a successful "
            "collective shmem finalize")
    _VMM_REGISTRY.clear()


def _contiguous_strides(shape):
    if len(shape) == 0:
        return ()
    strides = [1]
    for d in reversed(shape[1:]):
        strides.insert(0, strides[0] * d)
    return tuple(strides)


def create_sym_tensor_from_ptr(ptr, shape, dtype: torch.dtype, device,
                               stride=None, storage_offset=None) -> torch.Tensor:
    """Build a non-owning NPU tensor from an existing device address
    (``data_ptr()==ptr``).

    Counterpart of ``shmem.construct_tensor_from_ptr`` (torch_npu._C storage
    construction). The returned tensor does not own ``ptr``: the caller holds
    the address lifetime (a VMM ``source_va``, or a local symmetric address
    from ``map_to_sym_ptr``).
    """
    import torch_npu

    if not isinstance(ptr, int):
        raise TypeError(f"ptr must be an int address, got {type(ptr).__name__}")
    if ptr <= 0:
        raise ValueError(f"ptr must be positive, got {ptr}")
    shape = tuple(int(d) for d in shape)
    if any(d < 0 for d in shape):
        raise ValueError(f"shape dims must be non-negative, got {shape}")
    if stride is None:
        stride = _contiguous_strides(shape)
    if storage_offset is None:
        storage_offset = 0
    nbytes = _numel(shape) * torch.empty((), dtype=dtype).element_size()
    # torch_npu._C._construct_storage_from_data_pointer needs a torch.device;
    # normalize strings (e.g. 'npu:1') to torch.device.
    if not isinstance(device, torch.device):
        device = torch.device(device)
    metadata = {
        "data_ptr": int(ptr),
        "device": device,
        "nbytes": nbytes,
        "dtype": dtype,
        "size": list(shape),
        "stride": [int(s) for s in stride],
        "storage_offset": int(storage_offset),
    }
    storage = torch_npu._C._construct_storage_from_data_pointer(
        metadata["data_ptr"], metadata["device"], metadata["nbytes"])
    return torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(
        metadata, storage)


def create_vmm_physical(shape, dtype: torch.dtype, device: int):
    """Allocate a CANN VMM physical chunk and wrap its source_va as an NPU
    tensor.

    Returns ``(tensor, mem_handle, shareable_va)``; ``tensor.data_ptr()`` is
    the registered source VA. CANN VMM physical allocations must be aligned to
    the allocation granularity (2MiB), so the actual allocation is rounded up
    to gran (the extra padding is only spare physical/symmetric VA; the tensor
    still covers only the requested shape). Allocation also:
      - builds a ``VmmAllocation`` attached as ``tensor._vmm_allocation``
        (destroy/GC release via the VMM inverse ops, not aclshmem_free);
      - registers it in the global descriptor registry
        (``register_symmetric_descriptor``) for the framework to collect and
        run a single ``shmem_init_with_buffer``.

    ``device`` is the current logical device id (the visible 0..N-1 index from
    the torch.npu / ``aclrtSetDevice`` viewpoint) and must equal
    ``torch.npu.current_device()``. Note VMM has two id spaces (measured
    different on Ascend950 with ``ASCEND_RT_VISIBLE_DEVICES=4,5``):
      - ``aclrtMemGetAllocationGranularity`` / ``aclrtMallocPhysical`` /
        ``aclrtMapMem`` use the visible index ``device`` (passing 4/5 raises
        107001);
      - ``aclrtMemSetAccess``'s ``desc.location.id`` uses the physical card id,
        i.e. the entry in ``ASCEND_RT_VISIBLE_DEVICES`` (4/5 under mask "4,5";
        passing 0/1 raises 507899), obtained via ``_device.local_physical_id``.
    The returned tensor lives on that logical device.
    """
    nbytes = _chunk_nbytes(shape, dtype)
    device_id = int(device)
    _require_current_device(device_id)
    # Alloc uses the visible index; set_access uses the physical card id
    # (see docstring).
    access_device_id = local_physical_id(device_id)
    gran = int(acl_vmm_granularity(device_id))
    alloc_size = ((nbytes + gran - 1) // gran) * gran
    # On Ascend950 fabric_bytes is non-empty (acl_vmm_alloc_source exports
    # Fabric V2 internally).
    va, handle, shareable, size, fabric_bytes = acl_vmm_alloc_source(
        alloc_size, device_id, access_device_id)
    allocation = VmmAllocation(
        va, handle, shareable, size, fabric_handle_bytes=fabric_bytes,
        device_id=device_id)
    try:
        tensor = create_sym_tensor_from_ptr(
            va, list(shape), dtype, torch.device(f"npu:{device_id}"))
        tensor._vmm_allocation = allocation
        register_symmetric_descriptor(allocation)
    except Exception:
        # Registration can fail when the active heap has already frozen its
        # descriptor set.  This allocation was never registered/referenced by
        # SHMEM, so release its source mapping/handle transactionally.
        allocation.destroy()
        raise
    return tensor, handle, shareable


def map_to_sym_ptr(va, name: str) -> int:
    """Translate a registered local VA to its local shmem symmetric address
    (PR-653 ``get_buffer_ptr``).

    Requires an initialized buffer-backed symmetric heap
    (``shmem_init_with_buffer``) and a registered ``va``; raises otherwise.
    """
    sym = int(acl_shmem_get_buffer_ptr(int(va)))
    if sym == 0:
        raise RuntimeError(
            f"{name}: VA 0x{int(va):x} is not registered in the buffer-backed "
            f"shmem heap; call shmem_init_with_buffer with its descriptor first")
    return sym


# Local source-VA attribute names agreed with the framework side (correspond
# to MindSpeed's ``symmetric_owner_weight_source_vas`` /
# ``symmetric_weight_slots`` mappings):
#   ``_moonep_owner_source_va`` -- source VA of this rank's local home chunk
#   ``_moonep_slot_source_va``  -- source VA of this rank's local slot chunk
# Buffer reads these attributes on the VMM path of prefetch/reduce_grad and
# translates them to the local symmetric base via ``map_to_sym_ptr``. The
# framework attaches them with ``attach_local_source_vas`` when building the
# full weight.
_OWNER_SOURCE_VA_ATTR = "_moonep_owner_source_va"
_SLOT_SOURCE_VA_ATTR = "_moonep_slot_source_va"


def attach_local_source_vas(tensor, owner_source_va, slot_source_va):
    """Attach the source VAs of this rank's local home/slot chunks to the full
    weight tensor.

    Called by the framework/tests after building the full weight; Buffer's
    prefetch/reduce_grad VMM path uses them with ``get_buffer_ptr`` to build
    local shmem tensors.
    """
    setattr(tensor, _OWNER_SOURCE_VA_ATTR, int(owner_source_va))
    setattr(tensor, _SLOT_SOURCE_VA_ATTR, int(slot_source_va))
    return tensor


def create_sym_tensor_from_source(source_va, shape, dtype: torch.dtype, device,
                                  name: str, *, allow_logical_extent: bool = False
                                  ) -> torch.Tensor:
    """Build a local shmem tensor from a registered local source VA
    (translated via get_buffer_ptr).

    ``source_va`` must identify exactly one registered, non-destroyed VMM
    allocation. By default the requested view must not exceed that
    allocation's physical bytes; otherwise non-owning tensor metadata would
    advertise a larger kernel-accessible range than the caller-owned backing,
    overrunning in async MTE.

    ``allow_logical_extent`` exists only for the prefetch home ABI: it derives
    ``epn=E/R`` from the logical ``E``, but the kernel only touches the local
    ``epn`` rows. GradReduce must not use this exemption because its native ABI
    interprets dim 0 directly as ``epn``.
    The returned tensor's ``data_ptr()`` is the local symmetric base (the
    local part of the unified symmetric slots).
    """
    source_va = int(source_va)
    allocation = next(
        (
            item for item in _VMM_REGISTRY
            if not getattr(item, '_destroyed', False)
            and int(item.source_va) == source_va
        ),
        None,
    )
    if allocation is None:
        raise RuntimeError(
            f"{name}: source VA 0x{source_va:x} does not identify an active "
            "registered VMM allocation")
    requested_bytes = _chunk_nbytes(shape, dtype)
    if requested_bytes > int(allocation.size) and not allow_logical_extent:
        raise RuntimeError(
            f"{name}: requested symmetric view requires {requested_bytes} bytes, "
            f"but source allocation 0x{source_va:x} owns only "
            f"{int(allocation.size)} bytes")
    sym = map_to_sym_ptr(source_va, name)
    return create_sym_tensor_from_ptr(sym, shape, dtype, device)


class VmmProjection:
    """Projection view over a reserved VA plus multiple physical-handle
    segments (the ``[E+B]`` full_weight of section 6.4).

    Segments may be mapped at gran-aligned offsets inside a larger reserved
    VA (the ``[E+B]`` view only maps this rank's home/slot segments; the other
    rows are reserved-unmapped). ``destroy()`` only unmaps+releases the
    reserved VA, not the physical memory (owned by each segment's
    ``_vmm_allocation``).
    """

    def __init__(self, base, segments):
        self.base = int(base)
        self._segments = list(segments)  # (va, size, handle)
        self._destroyed = False

    def destroy(self) -> None:
        if self._destroyed:
            return
        for va, _, _ in self._segments:
            acl_vmm_unmap(va)
        if self.base:
            acl_vmm_release(self.base)
            self.base = 0
        self._destroyed = True


def create_full_projection_tensor(home, slot, E, B, rank, device):
    """Build the real ``[E+B, H, H']`` VMM projection view of section 6.4
    (replaces ``torch.empty``).

    ``home`` / ``slot`` must be registered VMM chunks
    (``create_vmm_physical``); their ``_vmm_allocation`` provides the
    gran-aligned physical segments. Mapping:
      - when ``epn*row % gran == 0``, keep the "global expert row" layout: the
        home segment maps to logical global expert rows
        ``[rank*epn, (rank+1)*epn)`` (``aclrtMapMem`` requires the mapped VA to
        be gran-aligned; ``epn*row`` being a gran multiple guarantees every
        rank's ``rank*epn*row`` is aligned);
      - otherwise fall back: the home segment starts at the projection base
        (``[0, epn*row)``) and the slot segment starts at
        ``ALIGN_UP(epn*row, lcm(gran, row))`` -- ``lcm(gran, row)`` satisfies
        both aclrtMapMem's gran alignment and the tensor row-boundary
        (``row``) alignment, so ``slot_start`` equals exactly
        ``slot_off // row`` (otherwise slot_off lands mid-row and the alias
        breaks). If ``slot_start > E`` (equivalently
        ``lcm(gran,row)/row > E``), the config cannot be represented as a
        ``[E+B]`` row-aligned projection and ValueError is raised.
    In the "global" layout the slot segment starts at ``ALIGN_UP(E*row, gran)``
    (logical rows ``[E, E+B)`` when ``E*row`` is a gran multiple). The reserved
    VA covers only the two mapped segments; the other rows are
    reserved-unmapped -- the kernel never reads the projection view (remote
    rows use symmetric addressing, see sections 6.3/6.4).

    The returned tensor's ``data_ptr()`` is the projection VA base; this base
    is NOT registered into the buffer-backed shmem init and cannot be passed to
    ``map_to_sym_ptr``. ``tensor._projection`` holds the destroyable handles
    (``home_start`` / ``slot_start`` give the home/slot segment positions as
    ``[E+B]`` logical row numbers, for tests to verify aliasing). The same
    physical handle is mapped into two VAs (source_va and the projection
    base); physical memory stays owned by home/slot's ``_vmm_allocation``.

    ``device`` is the current logical device id (visible index, must equal
    ``torch.npu.current_device()``): used by ``acl_vmm_granularity`` /
    ``acl_vmm_map``, while ``acl_vmm_set_access`` uses the corresponding
    physical card id (``_device.local_physical_id(device)``; 4/5 under mask
    "4,5").
    """
    H, Hp = int(home.shape[1]), int(home.shape[2])
    row = H * Hp * int(home.element_size())
    _require_current_device(int(device))
    gran = int(acl_vmm_granularity(int(device)))
    epn = int(home.shape[0])
    home_alloc = getattr(home, '_vmm_allocation', None)
    slot_alloc = getattr(slot, '_vmm_allocation', None)
    if home_alloc is None or slot_alloc is None:
        raise ValueError(
            "home/slot must be VMM chunks (create_vmm_physical)")
    home_size = int(home_alloc.size)
    slot_size = int(slot_alloc.size)
    step = _lcm(gran, row)
    if (epn * row) % gran != 0:
        raise ValueError(f'epn * row % gran = {epn * row % gran}, while epn * row should be a multiple of gran')
    home_off = int(rank) * epn * row
    slot_off = _align_up(E * row, gran)
    home_start = home_off // row
    slot_start = slot_off // row
    if slot_start + B > E + B:
        raise ValueError(
            f"config not representable as a [E+B] row-aligned projection: "
            f"lcm(gran,row)/row = {step // row} forces slot_start={slot_start} "
            f"> E={E}; choose H*Hp so that lcm(2MiB, H*Hp*elemsize) <= E*row")
    # The reserved VA covers only the two actually mapped segments
    # (``ALIGN_UP(a)+ALIGN_UP(b)`` may exceed ``ALIGN_UP((E+B)*row)``; do not
    # estimate from the logical total size).
    total = _align_up(max(home_off + home_size, slot_off + slot_size), gran)
    base = int(acl_vmm_reserve(total))
    segments = []
    try:
        for alloc, off in ((home_alloc, home_off), (slot_alloc, slot_off)):
            va = base + off
            acl_vmm_map(va, int(alloc.size), alloc.mem_handle, 0)
            acl_vmm_set_access(va, int(alloc.size),
                               local_physical_id(int(device)))
            segments.append((va, int(alloc.size), alloc.mem_handle))
    except Exception:
        for va, _, _ in segments:
            try:
                acl_vmm_unmap(va)
            except Exception:
                pass
        try:
            acl_vmm_release(base)
        except Exception:
            pass
        raise
    tensor = create_sym_tensor_from_ptr(
        base, [E + B, H, Hp], home.dtype, torch.device(f"npu:{int(device)}"))
    proj = VmmProjection(base, segments)
    proj.home_start = home_start
    proj.slot_start = slot_start
    tensor._projection = proj
    return tensor


def shmem_init_with_buffer(descriptors, my_pe: int, n_pes: int, uid_bytes,
                           tail_bytes: int = 0,
                           data_op_engine_type: int = ACLSHMEM_DATA_OP_MTE,
                           device_id: int | None = None,
                           enable_udma: bool | None = None, qp_count: int = 0) -> None:
    """Initialize the buffer-backed shmem symmetric heap from caller-mapped VMM
    buffers (collective).

    ``descriptors`` is a list of ``(addr, size, mem_handle, fabric_handle)``,
    one per registered VMM buffer (see ``symmetric_descriptors()``). All ranks
    pass the same descriptor count/sizes. ``uid_bytes`` is rank0's
    ``acl_shmem_get_unique_id()`` UID broadcast via dist. Non-zero
    ``tail_bytes`` makes shmem keep a private malloc heap (usable by
    ``aclshmem_malloc``).
    """
    global _SHMEM_BACKING_LOCKED
    if device_id is None:
        device_id = int(torch.npu.current_device())
    # Freeze the registry before taking/converting the descriptor snapshot and
    # entering the collective.  Otherwise a second Python thread could append
    # an allocation after the snapshot but before native init returns.
    was_locked = _SHMEM_BACKING_LOCKED
    _SHMEM_BACKING_LOCKED = True
    try:
        native_descriptors = [
            (int(addr), int(size), int(mh),
             fb if isinstance(fb, bytes) else bytes(fb or b""))
            for (addr, size, mh, fb) in descriptors
        ]
        rc = acl_shmem_init_with_buffers(
            int(my_pe), int(n_pes), int(tail_bytes), int(data_op_engine_type),
            uid_bytes, native_descriptors, int(device_id), bool(enable_udma), int(qp_count))
    except Exception:
        if not was_locked:
            _SHMEM_BACKING_LOCKED = False
        raise
    if rc != 0:
        if not was_locked:
            _SHMEM_BACKING_LOCKED = False
        raise RuntimeError(f"acl_shmem_init_with_buffers failed with {rc}")


def alloc_sym_chunk(shape, dtype: torch.dtype, device: int) -> torch.Tensor:
    """Allocate a VMM chunk registered for buffer-backed init and return the
    source tensor.

    For low-level tests that bypass Buffer/ShmemRuntime: after allocation
    (already registered in the global descriptor registry) the caller runs
    init and then uses ``sym_tensor()`` to translate the tensor into the local
    symmetric tensor expected by kernels.

    ``device`` is the current logical device id (same as
    ``create_vmm_physical``; must equal ``torch.npu.current_device()``).
    """
    return create_vmm_physical(shape, dtype, device)[0]


def sym_tensor(source_tensor: torch.Tensor) -> torch.Tensor:
    """Translate a registered source tensor into the local symmetric tensor
    expected by kernels (get_buffer_ptr).

    ``source_tensor.data_ptr()`` must be a source VA registered into the
    buffer-backed heap; the returned tensor's ``data_ptr()`` is the local
    symmetric base (the local part of the unified symmetric slots).
    """
    return create_sym_tensor_from_ptr(
        map_to_sym_ptr(source_tensor.data_ptr(), 'sym_tensor'),
        source_tensor.shape, source_tensor.dtype, source_tensor.device)
