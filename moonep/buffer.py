import os
import shutil
import socket
import struct
import tempfile

import torch
import torch.distributed as dist

from moonep._C import (
    FABRIC_HANDLE_BYTES as _FABRIC_HANDLE_BYTES,
    nvl_dist_alloc,
    nvl_dist_map,
    nvl_fabric_supported,
    nvl_release_mem_handle,
    get_vmm_granularity,
    get_multicast_granularity,
    nvl_multicast_supported,
    nvl_multicast_create,
    nvl_multicast_import,
    nvl_multicast_add_device,
    nvl_multicast_bind_map,
)

_ELEM_SIZE = {
    torch.float32: 4,
    torch.bfloat16: 2,
    torch.int32: 4,
}

# How VMM allocations are shared between the ranks of an EP group. "auto" (the
# default) picks fabric handles when the group spans more than one node and the
# device supports them, and POSIX fds otherwise; "fabric" / "fd" force one.
_HANDLE_TYPE_ENV = "MOONEP_MEM_HANDLE_TYPE"


def _use_fabric_for_group(group: dist.ProcessGroup | None) -> bool:
    mode = os.environ.get(_HANDLE_TYPE_ENV, "auto").lower()
    assert mode in ("auto", "fabric", "fd"), (
        f"{_HANDLE_TYPE_ENV} must be one of auto/fabric/fd, got {mode!r}"
    )
    if mode == "fd":
        return False

    supported = bool(nvl_fabric_supported())
    world_size = (dist.get_world_size(group=group)
                  if dist.is_available() and dist.is_initialized() else 1)
    if world_size == 1:
        unsupported = [] if supported else [0]
    else:
        local = torch.tensor([supported], dtype=torch.uint8, device="cuda")
        gathered = torch.empty(world_size, dtype=torch.uint8, device="cuda")
        dist.all_gather_into_tensor(gathered, local, group=group)
        unsupported = (gathered == 0).nonzero().flatten().tolist()

    if mode == "fabric":
        assert not unsupported, (
            f"{_HANDLE_TYPE_ENV}=fabric, but fabric memory handles are "
            f"unsupported on group ranks {unsupported}."
        )
        return True

    return not unsupported


def _all_gather_shareables(
    local_handle: torch.Tensor,
    group: dist.ProcessGroup | None,
) -> torch.Tensor:
    """Gather every rank's fabric handle into a uint8[world_size, 64] CPU tensor."""
    world_size = dist.get_world_size(group=group)
    gathered = torch.empty(world_size, _FABRIC_HANDLE_BYTES,
                           dtype=torch.uint8, device="cuda")
    dist.all_gather_into_tensor(
        gathered, local_handle.cuda().view(1, _FABRIC_HANDLE_BYTES), group=group)
    return gathered.cpu()


def _broadcast_shareable(
    local_handle: torch.Tensor | None,
    owner_rank: int,
    group: dist.ProcessGroup | None,
) -> torch.Tensor:
    """Broadcast one rank's fabric handle to the whole group."""
    if local_handle is not None:
        buf = local_handle.cuda()
    else:
        buf = torch.empty(_FABRIC_HANDLE_BYTES, dtype=torch.uint8, device="cuda")
    src = dist.get_global_rank(group, owner_rank) if group is not None \
        else owner_rank
    dist.broadcast(buf, src=src, group=group)
    return buf.cpu()


def pad_to_granularity(nbytes: int) -> int:
    """Round up nbytes to VMM granularity."""
    gran = get_vmm_granularity()
    return ((nbytes + gran - 1) // gran) * gran


def pad_dim0_for_alignment(chunk_shape: list[int], dtype: torch.dtype) -> int:
    """Compute the padded dim0 so that chunk bytes are aligned to VMM granularity.

    Returns the padded dim0 value (>= chunk_shape[0]).
    """
    elem_size = _ELEM_SIZE[dtype]
    inner_size = elem_size
    for d in chunk_shape[1:]:
        inner_size *= d  # bytes per row

    nbytes = chunk_shape[0] * inner_size
    padded_bytes = pad_to_granularity(nbytes)
    padded_dim0 = padded_bytes // inner_size
    # Ensure exact alignment
    while padded_dim0 * inner_size % get_vmm_granularity() != 0:
        padded_dim0 += 1
    return padded_dim0


def _exchange_ipc_fds(
    local_fd: int | None,
    sender_ranks: list[int],
    local_rank: int,
    world_size: int,
    group: dist.ProcessGroup | None,
) -> dict[int, int]:
    """Pass POSIX fds between ranks via per-rank unix datagram sockets.

    Ranks listed in ``sender_ranks`` must pass their exported fd as
    ``local_fd`` (other ranks pass None). Every rank receives one fd from
    each sender; the kernel dups the fd into the receiving process
    (SCM_RIGHTS), so the returned fds are owned by this process and must be
    closed by the caller after import. The sender's own fd may be closed as
    soon as this function returns — a trailing barrier guarantees all peers
    have already received their copy.
    """
    # Group-rank 0 creates a shared dir; broadcast the path to the group.
    if local_rank == 0:
        dir_path = tempfile.mkdtemp(prefix="moonep_ipc_")
    else:
        dir_path = None
    obj = [dir_path]
    src = dist.get_global_rank(group, 0) if group is not None else 0
    dist.broadcast_object_list(obj, src=src, group=group)
    dir_path = obj[0]

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.bind(os.path.join(dir_path, f"rank_{local_rank}"))
    sock.settimeout(120)
    # All sockets must be bound before any send.
    dist.barrier(group=group)

    try:
        if local_fd is not None:
            payload = struct.pack("<i", local_rank)
            anc = [(socket.SOL_SOCKET, socket.SCM_RIGHTS,
                    struct.pack("<i", local_fd))]
            for dst in range(world_size):
                sock.sendmsg([payload], anc, 0,
                             os.path.join(dir_path, f"rank_{dst}"))

        fds = {}
        while len(fds) < len(sender_ranks):
            msg, ancdata, _flags, _addr = sock.recvmsg(16, socket.CMSG_SPACE(4))
            src_rank = struct.unpack("<i", msg[:4])[0]
            for level, ctype, cdata in ancdata:
                if level == socket.SOL_SOCKET and ctype == socket.SCM_RIGHTS:
                    fds[src_rank] = struct.unpack("<i", cdata[:4])[0]
                    break
            else:
                raise RuntimeError("received IPC message without an fd")
    finally:
        sock.close()
        # Everyone has received their fds; safe to tear down the sockets and
        # for senders to close their original fd.
        dist.barrier(group=group)
        if local_rank == 0:
            shutil.rmtree(dir_path, ignore_errors=True)
    return fds


def _map_nvl_dist_tensor(
    chunk_shape: list[int],
    dtype: torch.dtype,
    shareable: torch.Tensor,
    keepalive: torch.Tensor,
    local_rank: int,
    world_size: int,
    group: dist.ProcessGroup | None,
    use_fabric: bool,
) -> torch.Tensor:
    if use_fabric:
        shareables = _all_gather_shareables(shareable, group)
        full_tensor = nvl_dist_map(
            chunk_shape=chunk_shape,
            dtype=dtype,
            shareables=shareables,
            local_rank=local_rank,
            world_size=world_size,
            use_fabric=True,
        )
    else:
        local_fd = int(shareable.item())
        fds = _exchange_ipc_fds(local_fd, list(range(world_size)),
                                local_rank, world_size, group)
        os.close(local_fd)
        all_fds = [fds[r] for r in range(world_size)]
        try:
            full_tensor = nvl_dist_map(
                chunk_shape=chunk_shape,
                dtype=dtype,
                shareables=torch.tensor(all_fds, dtype=torch.int64),
                local_rank=local_rank,
                world_size=world_size,
                use_fabric=False,
            )
        finally:
            for fd in all_fds:
                os.close(fd)
    full_tensor._keepalive = keepalive
    return full_tensor


def create_nvl_dist_tensor(
    chunk_shape: list[int],
    dtype: torch.dtype,
    local_rank: int,
    world_size: int,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Allocate an NVLink distributed tensor with all-RW access.

    chunk_shape MUST already be padded to VMM granularity alignment.
    Use pad_dim0_for_alignment() to compute the padded dim0.

    `local_rank` and `world_size` must match the given `group` (or the default
    group when `group is None`). All ranks in the group exchange memory handles.
    """
    use_fabric = _use_fabric_for_group(group)
    keepalive, shareable, owned_handle = nvl_dist_alloc(
        shape=chunk_shape, dtype=dtype, use_fabric=use_fabric)
    try:
        return _map_nvl_dist_tensor(
            chunk_shape, dtype, shareable, keepalive,
            local_rank, world_size, group, use_fabric,
        )
    finally:
        nvl_release_mem_handle(owned_handle)


def create_nvl_dist_multicast_tensor(
    chunk_shape: list[int],
    dtype: torch.dtype,
    local_rank: int,
    world_size: int,
    group: dist.ProcessGroup | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate an NVLink distributed tensor and its multicast view together.

    Returns `(full_tensor, mc_view)`. `full_tensor` is the regular all-rank RW
    VMM mapping; `mc_view` is a single-chunk multicast VA whose `.data_ptr()` can
    be used by `multimem.st` to fan out writes to every rank's local chunk.

    The owned allocation handle needed by `cuMulticastBindMem` stays internal to
    this helper and is released after both mappings have been created.
    """
    use_fabric = _use_fabric_for_group(group)
    keepalive, shareable, owned_handle = nvl_dist_alloc(
        shape=chunk_shape, dtype=dtype, use_fabric=use_fabric)
    try:
        full_tensor = _map_nvl_dist_tensor(
            chunk_shape, dtype, shareable, keepalive,
            local_rank, world_size, group, use_fabric,
        )
        mc_view = _create_nvl_multicast_view(
            full_tensor, owned_handle, local_rank, world_size, group,
            use_fabric,
        )
        return full_tensor, mc_view
    finally:
        nvl_release_mem_handle(owned_handle)


def _create_nvl_multicast_view(
    meta_buf: torch.Tensor,
    owned_handle: int,
    local_rank: int,
    world_size: int,
    group: dist.ProcessGroup | None = None,
    use_fabric: bool = False,
) -> torch.Tensor:
    """Overlay a multicast (NVSwitch SHARP) mapping on an existing NVL chunk.

    Binds each rank's own chunk physical memory (the chunk it `cuMemCreate`d
    inside `meta_buf`) to a single multicast object, then maps a multicast VA.
    A `multimem.st` to the returned VA fan-outs the hardware-replicated write to
    every rank's local chunk at the same offset.

    Registration is one-shot and persistent — the caller keeps the returned
    tensor alive (same lifetime as `meta_buf`). No extra device memory is used
    beyond an additional virtual address mapping over the existing chunk.

    The returned tensor's `.data_ptr()` is the multimem address, laid out as a
    single chunk (write `mc[off]` hits every rank's chunk `[off]`).
    """
    assert nvl_multicast_supported(), "Multicast not supported on this device"
    chunk_elems = meta_buf.numel() // world_size
    size_bytes = chunk_elems * meta_buf.element_size()
    is_root = local_rank == 0

    # Root creates the multicast object and shares it with all ranks.
    if is_root:
        mc_handle, mc_shareable = nvl_multicast_create(
            size_bytes, world_size, use_fabric=use_fabric)
    else:
        mc_handle, mc_shareable = 0, None

    if use_fabric:
        root_handle = _broadcast_shareable(mc_shareable, 0, group)
        if not is_root:
            mc_handle = nvl_multicast_import(root_handle, use_fabric=True)
    else:
        local_fd = int(mc_shareable.item()) if is_root else None
        fds = _exchange_ipc_fds(local_fd, [0], local_rank, world_size, group)
        if is_root:
            os.close(local_fd)
        root_fd = fds[0]
        try:
            if not is_root:
                mc_handle = nvl_multicast_import(
                    torch.tensor(root_fd, dtype=torch.int64), use_fabric=False)
        finally:
            os.close(root_fd)

    # All ranks add their device before any bind, then barrier.
    nvl_multicast_add_device(mc_handle)
    dist.barrier(group=group)

    mc_view = nvl_multicast_bind_map(
        mc_handle, owned_handle, size_bytes, world_size)
    dist.barrier(group=group)
    return mc_view


def create_nvl_single_owner_tensor(
    shape: list[int],
    dtype: torch.dtype,
    owner_rank: int,
    local_rank: int,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Allocate a VMM tensor on one GPU, visible to all ranks via NVLink.

    The physical memory resides on owner_rank's GPU.  All ranks get an RW
    mapping so the resulting tensor can be read/written from any rank (remote
    accesses go over NVLink).  shape must already be padded to VMM granularity
    (use pad_dim0_for_alignment).
    """
    world_size = dist.get_world_size(group=group)
    use_fabric = _use_fabric_for_group(group)
    is_owner = local_rank == owner_rank

    if is_owner:
        keepalive, shareable, owned_handle = nvl_dist_alloc(
            shape=shape, dtype=dtype, use_fabric=use_fabric)
        nvl_release_mem_handle(owned_handle)
    else:
        shareable = None

    if use_fabric:
        owner_handle = _broadcast_shareable(shareable, owner_rank, group)
        tensor = nvl_dist_map(
            chunk_shape=shape, dtype=dtype,
            shareables=owner_handle.view(1, _FABRIC_HANDLE_BYTES),
            local_rank=0, world_size=1, use_fabric=True,
        )
    else:
        local_fd = int(shareable.item()) if is_owner else None
        fds = _exchange_ipc_fds(local_fd, [owner_rank], local_rank,
                                world_size, group)
        if is_owner:
            os.close(local_fd)
        owner_fd = fds[owner_rank]
        try:
            tensor = nvl_dist_map(
                chunk_shape=shape, dtype=dtype,
                shareables=torch.tensor([owner_fd], dtype=torch.int64),
                local_rank=0, world_size=1, use_fabric=False,
            )
        finally:
            os.close(owner_fd)

    if is_owner:
        tensor._keepalive = keepalive
    return tensor
