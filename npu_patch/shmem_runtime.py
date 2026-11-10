"""
ShmemRuntime: module-level wrapper over the shmem symmetric-heap lifecycle.

Plan B: init/finalize reuse the C-interface wrappers in
``npu_patch.buffer_c`` (``shmem_init`` / ``shmem_finalize``), so ShmemRuntime
and ``_C`` share one libshmem.so instance (no dependency on the shmem Python
package). rank/world_size/barrier use ``torch.distributed``. The original
shmem-Python-package implementation is kept in comments for rollback.

Buffer-backed init supports two bootstraps:
  - ``init_with_buffer`` (default) -- UID bootstrap (single node): rank0
    generates the UID and broadcasts it over dist; the symmetric-heap scope
    follows the broadcast scope.
  - ``init_with_ip_port`` (or ``init_with_buffer(ip_port=...)``) --
    ip_port/DEFAULT bootstrap (multi-node): every rank passes the same
    ``tcp://<rank0_ip>:<port>``; rank0 runs the config store server and the
    others connect as clients, so no UID broadcast is needed.

Legacy plan A note: the shmem Python package loads its own libshmem.so
instance, separate from the copy ``_C`` uses; the two paths must not be
mixed (do not mix).
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from ._common import (
    ACLSHMEM_DATA_OP_MTE,
    ACLSHMEM_DATA_OP_UDMA,
    compute_shmem_mem_size,
)
from .buffer_c import (
    shmem_finalize,
    shmem_init_with_ip_port,
    shmem_init_with_buffer,
    symmetric_descriptors,
)
from ._C import acl_shmem_barrier_all, acl_shmem_get_unique_id
from .runtime_audit import audit_loaded_shmem_library


def _env_udma_enabled() -> bool:
    return os.environ.get("ASCEND_MOONEP_USE_UDMA", "0").strip().lower() in (
        "1", "true", "yes", "on"
    )


def default_shmem_ip_port(master_addr: str, master_port: int) -> str:
    """Derive the shmem DEFAULT-bootstrap ip_port URL from the torchrun
    rendezvous address.

    The shmem config store is a *separate* TCP server and cannot reuse the
    torch-held ``MASTER_PORT``, so the port is +1 (avoids the torch
    rendezvous port). ``ip_port`` must be identical on every rank:
    ``MASTER_ADDR``/``MASTER_PORT`` are identical across ranks under a
    multi-node torchrun launch, so deriving from them is naturally
    consistent.
    """
    return f"tcp://{master_addr}:{int(master_port) + 1}"


def _multi_node_launch() -> bool:
    """Whether the torchrun launch spans multiple nodes
    (``WORLD_SIZE`` > per-node process count)."""
    world = os.environ.get("WORLD_SIZE")
    local = os.environ.get("LOCAL_WORLD_SIZE")
    if world is None or local is None:
        return False
    return int(world) > int(local)


def _env_shmem_ip_port() -> str | None:
    """Derive the shmem ip_port from the torchrun environment; None unless
    the launch is multi-node and both vars are present."""
    master_addr = os.environ.get("MASTER_ADDR")
    master_port = os.environ.get("MASTER_PORT")
    if not _multi_node_launch() or not master_addr or not master_port:
        return None
    return default_shmem_ip_port(master_addr, int(master_port))

# --- Original implementation (plan A: shmem Python package) ---
# import shmem.core as core
# from shmem.core.direct import my_pe, n_pes
# from shmem.core.utils import Buffer as SymBuffer


class ShmemRuntime:
    """Thin facade over the shmem symmetric-memory runtime lifecycle."""

    _initialized = False
    _poisoned = False
    _runtime_library_record = None

    @staticmethod
    def compute_mem_size(
        S: int,
        H: int,
        K: int,
        E: int,
        H_prime: int,
        num_ep_ranks: int,
        **kwargs,
    ) -> int:
        """Per-PE symmetric heap bytes needed for a given MoonEP config.

        Convenience over ``_common.compute_shmem_mem_size``: pass the model
        config (S seqlen, K topk count, E expert count, H / H' hidden sizes,
        num_ep_ranks) and get the 2 MiB-aligned ``mem_size`` to feed to
        ``ShmemRuntime.init``. The calculation covers the Buffer's symmetric
        chunks, the RMA op workspaces, the grad-reduce buffers and the prefetch
        weight slots; extra framework-side symmetric tensors can be added via
        ``owner_weight_bytes`` / the ``*_buffers`` / ``*_slots`` counts.
        """
        return compute_shmem_mem_size(S, H, K, E, H_prime, num_ep_ranks, **kwargs)

    @classmethod
    def init(cls, group=None, mem_size: int = None, uid=None) -> None:
        """[Removed] Legacy UID + ``local_mem_size`` symmetric-heap init.

        Only buffer-backed init is supported: construct Buffers /
        ``create_vmm_physical`` to register descriptors first, then call
        ``ShmemRuntime.init_with_buffer``. The legacy path is mutually
        exclusive with this process's buffer-backed heap; the symbol stays but
        raises explicitly so misuse is not silently swallowed by the C-side
        idempotency check.
        """
        raise RuntimeError(
            "ShmemRuntime.init(local_mem_size) has been removed: only "
            "buffer-backed ShmemRuntime.init_with_buffer is supported (PR-653)")

    @classmethod
    def init_with_buffer(cls, group=None, descriptors=None, uid_bytes=None,
                         tail_bytes: int = 0, ip_port: str | None = None,
                         enable_udma: bool | None = None, qp_count: int = 0) -> None:
        """Initialize the buffer-backed shmem symmetric heap (PR-653, collective).

        Uses caller-mapped CANN VMM buffers instead of ``local_mem_size``. Two
        bootstraps:

          - ``ip_port is None`` (default): UID bootstrap (single node). rank0
            generates the UID and broadcasts it within ``group``; every rank
            calls ``buffer_c.shmem_init_with_buffer`` with the same UID, and
            the symmetric-heap scope follows the broadcast scope.
          - ``ip_port`` is not None: ip_port/DEFAULT bootstrap (multi-node).
            Every rank passes the same ``ip_port``
            (``tcp://<rank0_ip>:<port>``); rank0 runs the config store server
            and the others connect as clients, no UID broadcast -- see
            ``ShmemRuntime.init_with_ip_port``.

        ``descriptors`` defaults to the process-wide registered VMM allocations
        (``buffer_c.symmetric_descriptors()``, populated by ``Buffer`` /
        ``create_vmm_physical`` at construction). Must run AFTER every Buffer is
        constructed and BEFORE the first dispatch/combine/prefetch/reduce_grad.
        """
        if cls._poisoned:
            raise RuntimeError(
                "shmem runtime is poisoned by a prior finalize failure; "
                "reinitialization is forbidden")
        if enable_udma is None:
            enable_udma = _env_udma_enabled()
        if cls._initialized:
            return
        if enable_udma and not qp_count:
            qp_count = 2
        if ip_port is not None:
            cls.init_with_ip_port(group=group, descriptors=descriptors,
                                  tail_bytes=tail_bytes, ip_port=ip_port,
                                  enable_udma=enable_udma, qp_count=qp_count)
            return
        # Fail closed before any SHMEM-native call (including UID generation).
        # Loading two libshmem images creates independent process-global
        # PE/heap state.
        # cls._runtime_library_record = audit_loaded_shmem_library()
        rank = dist.get_rank(group=group)
        world_size = dist.get_world_size(group=group)
        if uid_bytes is None:
            uid = acl_shmem_get_unique_id() if rank == 0 else None
            # ``src`` is always a global rank even when ``group`` is an EP
            # subgroup.  EP rank 0 is not necessarily global rank 0.
            uid_root = 0 if group is None else dist.get_global_rank(group, 0)
            # HCCL process groups do not provide a CPU/object transport.  Send
            # the binary UID as NPU tensors, including its ABI-derived length,
            # so embedded NUL bytes are preserved without pickle or Gloo.
            device = torch.device("npu", torch.npu.current_device())
            uid_length = torch.tensor(
                [len(uid) if uid is not None else 0],
                dtype=torch.int64,
                device=device,
            )
            dist.broadcast(uid_length, src=uid_root, group=group)
            length = int(uid_length.item())
            if length <= 0:
                raise RuntimeError("aclshmem UID broadcast produced an invalid byte length")
            uid_tensor = (
                torch.tensor(list(uid), dtype=torch.uint8, device=device)
                if uid is not None
                else torch.empty(length, dtype=torch.uint8, device=device)
            )
            if int(uid_tensor.numel()) != length:
                raise RuntimeError(
                    "aclshmem UID length changed while preparing subgroup broadcast"
                )
            dist.broadcast(uid_tensor, src=uid_root, group=group)
            uid_bytes = bytes(uid_tensor.cpu().tolist())
        if descriptors is None:
            descriptors = symmetric_descriptors()
        shmem_init_with_buffer(
            descriptors, int(rank), int(world_size), uid_bytes,
            tail_bytes=int(tail_bytes),
            data_op_engine_type=(ACLSHMEM_DATA_OP_UDMA | ACLSHMEM_DATA_OP_MTE if enable_udma
                                 else ACLSHMEM_DATA_OP_MTE),
            device_id=int(torch.npu.current_device()),
            enable_udma=enable_udma, qp_count=qp_count)
        cls._initialized = True

    @classmethod
    def init_with_ip_port(cls, group=None, ip_port=None, descriptors=None,
                          tail_bytes: int = 0, rank: int | None = None,
                          world_size: int | None = None,
                          enable_udma: bool | None = None, qp_count: int = 0) -> None:
        """Initialize the buffer-backed shmem heap via ip_port/DEFAULT
        bootstrap (multi-node, collective).

        rank/world_size default to ``dist``. Callers without a process group
        may pass ``rank`` and ``world_size`` explicitly, in which case dist is
        never touched. Every rank passes the same ``ip_port``
        (``"tcp://<rank0_ip>:<port>"`` or ``"tcp6://[ipv6]:<port>"``):
        rank0 (my_pe=0) runs the config store server and the others connect
        as clients, so the shmem symmetric heap spans nodes without a UID
        broadcast (equivalent to the shmem Python package's
        ``aclshmem_init(InitAttr(ip_port=...))``).

        ``ip_port`` defaults to the torchrun environment
        (``MASTER_ADDR``/``MASTER_PORT``, port +1 to avoid torch rendezvous;
        see ``default_shmem_ip_port``), so multi-node torchrun launches may
        omit it. ``descriptors`` defaults to the process-wide registered VMM
        allocations (``buffer_c.symmetric_descriptors()``). Timing is the
        same as ``init_with_buffer``.
        """
        if cls._poisoned:
            raise RuntimeError(
                "shmem runtime is poisoned by a prior finalize failure; "
                "reinitialization is forbidden")
        if enable_udma is None:
            enable_udma = _env_udma_enabled()
        if cls._initialized:
            return
        # The IP/port path owns the same process-global runtime and backing as
        # the UID path, so it must pass the identical pre-native audit.
        # cls._runtime_library_record = audit_loaded_shmem_library()
        if ip_port is None:
            ip_port = _env_shmem_ip_port()
        if ip_port is None:
            raise RuntimeError(
                "ShmemRuntime.init_with_ip_port requires ip_port "
                "('tcp://<rank0_ip>:<port>' or 'tcp6://[ipv6]:<port>'); pass it "
                "explicitly, or set MASTER_ADDR/MASTER_PORT for a multi-node "
                "torchrun launch")
        if rank is None and world_size is None:
            rank = dist.get_rank(group=group)
            world_size = dist.get_world_size(group=group)
        else:
            if rank is None or world_size is None:
                raise ValueError("rank and world_size must be provided together")
            if group is not None:
                raise ValueError("group must be None when rank is provided explicitly")
            if not isinstance(rank, int) or not isinstance(world_size, int):
                raise TypeError("rank and world_size must be integers")
            if world_size <= 0 or rank < 0 or rank >= world_size:
                raise ValueError(
                    f"invalid rank/world_size: rank={rank}, world_size={world_size}"
                )
        if descriptors is None:
            descriptors = symmetric_descriptors()
        if enable_udma and not qp_count:
            qp_count = 2
        shmem_init_with_ip_port(
            int(rank), int(world_size), ip_port, descriptors,
            tail_bytes=int(tail_bytes),
            data_op_engine_type=(ACLSHMEM_DATA_OP_UDMA | ACLSHMEM_DATA_OP_MTE if enable_udma
                                 else ACLSHMEM_DATA_OP_MTE),
            device_id=int(torch.npu.current_device()),
            enable_udma=enable_udma, qp_count=qp_count)
        cls._initialized = True

    @classmethod
    def finalize(cls) -> int:
        """Finalize collectively and return the native status code.

        On failure ``_initialized`` deliberately remains true, which prevents
        framework teardown from treating registered caller-owned VMM backing as
        releasable.  The process should be considered poisoned and terminated
        only after diagnostics are captured.
        """
        if cls._initialized:
            # Plan B: reuse the buffer_c wrapper.
            rc = int(shmem_finalize())
            if rc != 0:
                cls._poisoned = True
                raise RuntimeError(
                    f"aclshmem_finalize failed with {rc}; runtime state and "
                    "caller-owned VMM backing remain live")
            # --- Original implementation (plan A) ---
            # core.finalize()
            cls._initialized = False
            cls._poisoned = False
            return rc
        return 0

    @classmethod
    def is_initialized(cls) -> bool:
        # Plan B: the C interface exposes no init_status; use the internal flag.
        return cls._initialized
        # --- Original implementation (plan A) ---
        # try:
        #     from shmem.core.direct import InitStatus, init_status
        #     return init_status() == InitStatus.INITIALIZED
        # except Exception:
        #     return cls._initialized

    @classmethod
    def runtime_library_info(cls) -> dict[str, object] | None:
        """Return a copy of the last successful pre-init runtime identity audit."""
        if cls._runtime_library_record is None:
            return None
        return dict(cls._runtime_library_record)

    @staticmethod
    def rank() -> int:
        # Plan B: under the C-interface path the Python package instance is
        # never init'd; use dist.
        return int(dist.get_rank())
        # --- Original implementation (plan A) ---
        # return int(my_pe())

    @staticmethod
    def world_size() -> int:
        return int(dist.get_world_size())
        # --- Original implementation (plan A) ---
        # return int(n_pes())

    @staticmethod
    def barrier(group=None) -> None:
        # Plan B: v1.3 aclshmem_barrier_all is a global barrier that also
        # waits for pending RMA to complete.
        acl_shmem_barrier_all()
        # --- Original implementation (plan A) ---
        # """Best-effort shmem barrier; falls back to ``dist.barrier`` when the
        # python binding does not expose one."""
        # barrier = getattr(core, "barrier", None)
        # if callable(barrier):
        #     barrier()
        # else:
        #     dist.barrier(group=group)


def runtime_library_info() -> dict[str, object] | None:
    """Return the cached ``libshmem.so`` realpath/dev/inode/SHA256 evidence."""
    return ShmemRuntime.runtime_library_info()
