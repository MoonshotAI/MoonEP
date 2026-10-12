"""
Ascend port of ``moonep/api.py``.

Same public API (``Buffer`` / ``dispatch`` / ``prefetch_weight`` / ``combine`` /
``reduce_grad``) and the same ``ctx`` layout contract, but the communication
memory is shmem symmetric memory allocated via the shmem Python API, and the
kernels are invoked through op objects created at Buffer init (ascend_deepep
-style architecture: op constructed at init, ``compute`` on demand).
"""

import logging
import os
import warnings
from collections.abc import Sequence

import torch
import torch_npu
import torch.distributed as dist

# Plan B: the C-interface path no longer depends on the shmem Python package.
# import shmem.core as core

from ._common import (
    BARRIER_SLOTS,
    OP_WS_BYTES_DEFAULT,
    cross_rank_barrier,
    record_streams,
)
from .buffer_c import (
    _OWNER_SOURCE_VA_ATTR,
    _SLOT_SOURCE_VA_ATTR,
    create_sym_tensor_from_ptr,
    create_sym_tensor_from_source,
    create_vmm_physical,
    map_to_sym_ptr,
)
from .planning import MoonEPCommPlan, allocate_planning_outputs
from .ops import (
    CombineOp,
    DispatchOp,
    GradReduceOp,
    PlanningOp,
    PrefetchOp,
)
from .launchers import planning_workspace_sizes
from .shmem_runtime import ShmemRuntime

logger = logging.getLogger(__name__)

_COMBINE_SYNC_BYTES_PER_RANK = 16 * 2 * 512

def _is_ascend950(device: int, props) -> bool:
    names = [os.environ.get("SOC_VERSION", ""), os.environ.get("ASCEND_SOC_VERSION", ""),
             getattr(props, "name", ""), getattr(props, "soc_name", "")]
    get_device_name = getattr(torch.npu, "get_device_name", None)
    if get_device_name is not None:
        try:
            names.append(get_device_name(device))
        except TypeError:
            try:
                names.append(get_device_name())
            except Exception:
                pass
        except Exception:
            pass
    return any("ascend950" in "".join(c for c in str(name).lower() if c.isalnum()) for name in names)


def _num_sms_dedup_from_env(max_sms: int) -> int:
    """Resolve the local epilogue/prologue SM count (same env override as CUDA)."""
    raw = os.environ.get("MOONEP_NUM_SMS_DEDUP")
    if raw is None or raw == "":
        return max_sms
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"MOONEP_NUM_SMS_DEDUP must be an integer in [1, {max_sms}], got {raw!r}"
        ) from exc
    if not (1 <= value <= max_sms):
        raise ValueError(
            f"MOONEP_NUM_SMS_DEDUP must be in [1, {max_sms}], got {value}"
        )
    return value


def _op_ws_bytes_from_env(default: int = OP_WS_BYTES_DEFAULT) -> int:
    """Per-op symmetric workspace bytes (tunable for benchmarks).

    v1 uses a single generous workspace per RMA-capable op; M4 tunes it down to
    the kernels' actual scratch needs. No 2 MiB alignment is required on Ascend
    (individual symmetric allocations are aligned by the allocator internally).
    """
    raw = os.environ.get("MOONEP_ASCEND_OP_WS_BYTES")
    if raw is None or raw == "":
        return default
    return int(raw)


def _log_context_buffer_size(ctx: dict) -> None:
    def tensor_nbytes(tensor: torch.Tensor) -> int:
        return int(tensor.numel()) * int(tensor.element_size())

    def format_nbytes(nbytes: int) -> str:
        if nbytes >= 1 << 30:
            return f"{nbytes / (1 << 30):.2f} GiB"
        if nbytes >= 1 << 20:
            return f"{nbytes / (1 << 20):.2f} MiB"
        if nbytes >= 1 << 10:
            return f"{nbytes / (1 << 10):.2f} KiB"
        return f"{nbytes} B"

    if int(ctx['rank']) != 0:
        return

    R = int(ctx['R'])
    hidden_chunk_bytes = tensor_nbytes(ctx['hidden_buf'])
    meta_vars_bytes = sum(tensor_nbytes(ctx[name]) for name in (
        'weights_buf',
        'barrier_buf',
        'combine_sync_buffer',
        'dispatch_sync_buffer',
    ))
    local_temp_bytes = tensor_nbytes(ctx['planning_workspace'])
    op_workspace_bytes = sum(op._ws_bytes for op in ctx['kernels'].values())
    combine_udma_payload_bytes = (
        int(ctx['NvS']) * int(ctx['H']) * torch.bfloat16.itemsize
        + int(ctx['R']) * int(ctx['NvS']) * torch.float32.itemsize
    )
    combine_udma_local_workspace_bytes = (
        (combine_udma_payload_bytes + 63) // 64 * 64
        + 32 * 64
    )

    log_lines = [
        "MoonEP-Ascend comm buffer sizes (per rank):",
        "  config                 : "
        f"R={ctx['R']}, S={ctx['S']}, H={ctx['H']}, E={ctx['E']}, "
        f"K={ctx['K']}, NvS={ctx['NvS']}",
        "  sm_count               : "
        f"num_sms={ctx['num_sms']}, num_sms_dedup={ctx['num_sms_dedup']}",
        f"  hidden_chunk           : {format_nbytes(hidden_chunk_bytes)}",
        f"  meta_vars              : {format_nbytes(meta_vars_bytes)}",
        f"  local_temp             : {format_nbytes(local_temp_bytes)}",
        f"  op_symmetric_workspace : {format_nbytes(op_workspace_bytes)}",
        f"  combine_udma_ws (lazy) : "
        f"{format_nbytes(combine_udma_local_workspace_bytes)}",
        f"  total_shmem+local      : "
        f"{format_nbytes(hidden_chunk_bytes + meta_vars_bytes + local_temp_bytes + op_workspace_bytes + combine_udma_local_workspace_bytes)}",
    ]
    logger.info("\n".join(log_lines))


def _create_context(
    S: int,
    H: int,
    K: int,
    E: int,
    num_ep_ranks: int,
    num_sms: int | None = None,
    token_padding: int = 128,
    B: int | None = None,
    group: "dist.ProcessGroup | None" = None,
    rank: int | None = None,
    use_udma: bool = True,
    udma_flags: dict | None = None,
) -> dict:
    """Pre-allocate all shmem symmetric chunks and local temp buffers.

    Ascend applies no token/expert padding and no per-allocation 2 MiB padding:
    ``NvS`` is exactly ``S*K``. Planning owns one exact-size symmetric workspace
    for device communication and the context owns one contiguous local int32
    workspace whose aligned subregions are derived by the C++ launcher.
    There is no cross-rank contiguous VA: remote chunk access goes through the
    peer symmetric address (host put/get, or device aclshmem_ptr in kernels).
    """
    if num_sms is None:
        num_sms = 32
    assert isinstance(num_sms, int) and num_sms > 0, \
        f"num_sms must be a positive int, got {num_sms}"
    R = num_ep_ranks
    uses_process_group = rank is None
    if uses_process_group:
        rank = dist.get_rank(group=group)
        assert R == dist.get_world_size(group=group), (
            f"num_ep_ranks ({R}) must equal group world size "
            f"({dist.get_world_size(group=group)})"
        )
    else:
        assert group is None, "group must be None when rank is provided explicitly"
        assert isinstance(rank, int) and 0 <= rank < R, (
            f"rank must be an int in [0, {R}), got {rank}"
        )
    N = S * K
    device = torch.npu.current_device()
    dev = f"npu:{device}"
    props = torch.npu.get_device_properties(device)
    max_sms = getattr(props, "multi_processor_count", 32)
    num_sms_dedup = _num_sms_dedup_from_env(max_sms)
    udma_supported = _is_ascend950(device, props)

    epn = E // R
    assert E % R == 0, f"E ({E}) must be divisible by R ({R})"
    assert isinstance(token_padding, int) and token_padding > 0, \
        f"token_padding must be a positive int, got {token_padding}"
    if B is None:
        B = epn
    assert isinstance(B, int) and B > 0, f"B must be a positive int, got {B}"

    # Ascend: no token-padding / expert-alignment padding and no 2 MiB-aligned
    # padding. shmem cross-rank transfer needs no padding to a fixed length, and
    # individual symmetric allocations need no 2 MiB alignment (only the per-PE
    # heap ``mem_size`` must be 2 MiB-aligned). The logical NvS is exactly the
    # route count S*K, and the hidden chunk is allocated at that exact shape.
    NvS = S * K

    # Planning int32 range guard (there is no cross-rank contiguous VA).
    int32_max = 2**31 - 1
    assert 0 < N < int32_max, (
        "planning requires 0 < S*K < int32_max: "
        f"S={S}, K={K}, S*K={N}, int32_max={int32_max}"
    )
    planning_local_elems, planning_symmetric_bytes = planning_workspace_sizes(
        S, K, E, NvS, R, num_sms
    )
    # ================================================================
    # Allocate communication chunks as CANN VMM physical chunks (collective).
    # Buffer-backed path only: Buffers are constructed while shmem is still
    # uninitialized; the comm chunks are CANN VMM (source VA) allocations
    # registered in the global descriptor registry. After constructing every
    # Buffer the caller runs ShmemRuntime.init_with_buffer, and
    # Buffer._ensure_shmem_views then maps each chunk to the symmetric
    # address the kernels use (not the source VA).
    # ================================================================
    hidden_buf, _, _ = create_vmm_physical([NvS, H], torch.bfloat16, device)
    # Cross-rank (symmetric) dispatch metadata.
    weights_buf, _, _ = create_vmm_physical([NvS], torch.int32, device)
    barrier_buf, _, _ = create_vmm_physical(
        [BARRIER_SLOTS], torch.int32, device)
    vmm_chunks = {
        'hidden_buf': hidden_buf._vmm_allocation,
        'weights_buf': weights_buf._vmm_allocation,
        'barrier_buf': barrier_buf._vmm_allocation,
    }

    # Combine owns a separate grant/done cacheline pool.  Keeping this distinct
    # from dispatch prevents the two protocols from sharing flag generations or
    # becoming coupled through their unrelated layouts.
    combine_sync_buffer, _, _ = create_vmm_physical(
        [R * _COMBINE_SYNC_BYTES_PER_RANK], torch.uint8, device)
    combine_sync_buffer.zero_()
    vmm_chunks['combine_sync_buffer'] = combine_sync_buffer._vmm_allocation

    # Dispatch sync_buffer: Token control region followed by the four-round
    # local sync region for the 16 Token AIVs.
    dispatch_sync_buffer, _, _ = create_vmm_physical(
        [R * 16 * 6 * 512 + 16 * 4 * 512], torch.uint8, device)
    dispatch_sync_buffer.zero_()
    vmm_chunks['dispatch_sync_buffer'] = dispatch_sync_buffer._vmm_allocation
    # Zero this rank's barrier words (writes through to the VMM source VA).
    barrier_buf.zero_()

    planning_workspace = torch.empty(
        planning_local_elems, dtype=torch.int32, device=dev
    )

    # ================================================================
    # Kernel ops (ascend_deepep-style: created at Buffer init)
    # ================================================================
    ws = _op_ws_bytes_from_env()
    kernels = {
        'planning': PlanningOp(ws_bytes=planning_symmetric_bytes, num_sms=num_sms),
        'dispatch': DispatchOp(ws_bytes=ws, num_sms=num_sms),
        'combine': CombineOp(ws_bytes=ws, num_sms=num_sms),
        'prefetch': PrefetchOp(ws_bytes=0, num_sms=num_sms),
        'grad_reduce': GradReduceOp(ws_bytes=0, num_sms=num_sms),
    }
    ctx_udma_flags = {'planning': False, 'dispatch': True, 'combine': True,
                      'prefetch': True, 'grad_reduce': True}
    requested_udma = {name: bool(use_udma) for name in ctx_udma_flags}
    if udma_flags:
        requested_udma.update({name: bool(value) for name, value in udma_flags.items()})
    effective_udma = {name: bool(udma_supported and ctx_udma_flags[name] and requested_udma[name])
                      for name in ctx_udma_flags}
    os.environ['ASCEND_MOONEP_USE_UDMA'] = '1' if any(effective_udma.values()) else '0'
    for name, enabled in effective_udma.items():
        os.environ[f'ASCEND_MOONEP_{name.upper()}_USE_UDMA'] = '1' if enabled else '0'

    ctx = {
        'rank': rank,
        'group': group,
        '_uses_process_group': uses_process_group,
        'R': R, 'E': E, 'S': S, 'K': K, 'H': H,
        'B': B,
        'N': N, 'NvS': NvS,
        'num_sms': num_sms,
        'num_sms_dedup': num_sms_dedup,
        'device': device,
        '_udma_supported': udma_supported,
        '_udma_ctx_flags': ctx_udma_flags,
        '_udma_user_flags': requested_udma,
        '_planning_use_udma': effective_udma['planning'],
        '_dispatch_use_udma': effective_udma['dispatch'],
        '_combine_use_udma': effective_udma['combine'],
        '_prefetch_use_udma': effective_udma['prefetch'],
        '_grad_reduce_use_udma': effective_udma['grad_reduce'],
        # shmem symmetric chunks (tensors; tensor.data_ptr() is the symmetric
        # base for peer addressing / device RMA — no separate Buffer handles).
        'hidden_buf': hidden_buf,
        'weights_buf': weights_buf,
        'barrier_buf': barrier_buf,
        'combine_sync_buffer': combine_sync_buffer,
        'dispatch_sync_buffer': dispatch_sync_buffer,
        'shmem_chunks': [
            hidden_buf,
            weights_buf,
            barrier_buf,
            combine_sync_buffer,
            dispatch_sync_buffer,
        ],
        # CANN VMM allocations (source_va + mem_handle + shareable_va):
        # collected by the framework for buffer-backed shmem init
        # (ShmemRuntime.init_with_buffer); Buffer.destroy reverses the VMM ops.
        'vmm_chunks': vmm_chunks,
        # One local allocation; C++ slices every 32-byte-aligned subregion.
        'planning_workspace': planning_workspace,
        # Local views (rebuilt over translated shmem addresses once
        # _ensure_shmem_views runs after init_with_buffer).
        'hidden_buf_local': hidden_buf[:NvS],
        'weights_buf_local': weights_buf[:NvS],
        # Kernel ops + barrier state
        'kernels': kernels,
        '_barrier_round': 0,
        # Buffer-backed: translated by _ensure_shmem_views after
        # init_with_buffer.
        '_shmem_views_ready': False,
        # Total VA bytes registered with buffer-backed init (actual 2 MiB-gran
        # aligned sizes of the VMM chunks / op workspaces); used by the
        # framework to budget descriptor size.
        'shmem_footprint_bytes': (
            sum(a.size for a in vmm_chunks.values())
            + sum(
                int(op._ws_tensor._vmm_allocation.size)
                if op._ws_tensor is not None
                and getattr(op._ws_tensor, '_vmm_allocation', None) is not None
                else 0
                for op in kernels.values()
            )
        ),
    }
    _log_context_buffer_size(ctx)
    return ctx


class Buffer:
    """Ascend MoonEP communication buffer owner (buffer-backed path only).

    A Buffer owns the communication chunks, local scratch tensors, the kernel
    op objects (created at init), and the async communication stream.

    Buffer-backed only (see analysis doc §4): the comm chunks are CANN VMM
    allocations (source VA + shareable VA) registered in the global descriptor
    registry at construction; after constructing all Buffers the caller runs
    ``ShmemRuntime.init_with_buffer`` once, and on the first
    dispatch/combine/prefetch/reduce_grad the Buffer maps the chunks to the
    symmetric addresses the kernels use via ``_ensure_shmem_views``.
    Call ``destroy()`` before tearing down the process group; shmem runtime
    finalize is owned by the framework/caller (Buffer never finalizes itself).
    """

    def __init__(
        self,
        S: int,
        H: int,
        K: int,
        E: int,
        num_ep_ranks: int,
        num_sms: int | None = None,
        token_padding: int = 128,
        B: int | None = None,
        group: "dist.ProcessGroup | None" = None,
        comm_stream_priority: int = -1,
        enable_pdl: bool = True,
        explicitly_destroy: bool = False,
        rank: int | None = None,
        **kwargs,
    ):
        """Allocate and hold all communication buffers.

        Args: same as ``moonep.api.Buffer``. ``token_padding`` is accepted for
        moonep API compatibility but ignored on Ascend (no token-padding in the
        NvS layout). ``enable_pdl`` is reserved on Ascend (CUDA programmatic
        dependent launch has no equivalent) and kept only for API compatibility.
        ``rank`` may be supplied by bootstrap paths that do not create a
        torch.distributed process group.
        """
        # Buffer-backed only: at construction shmem is uninitialized; the comm
        # chunks are CANN VMM allocations registered in the global descriptor
        # registry, awaiting ShmemRuntime.init_with_buffer from the
        # framework/tests. Call sites that need shmem addresses
        # (dispatch/combine/prefetch/reduce_grad) verify and map them through
        # _ensure_shmem_views.
        allowed_udma = {'planning', 'dispatch', 'combine', 'prefetch', 'grad_reduce'}
        use_udma = kwargs.pop('use_udma', False)
        if not isinstance(use_udma, bool):
            raise TypeError('use_udma must be a bool')
        udma_flags = {}
        for name in allowed_udma:
            for key in (f'{name}_use_udma', f'use_udma_{name}'):
                if key in kwargs:
                    value = kwargs.pop(key)
                    if not isinstance(value, bool):
                        raise TypeError(f'{key} must be a bool')
                    udma_flags[name] = value
                    break
        if kwargs:
            raise TypeError(f'unexpected Buffer kwargs: {sorted(kwargs)}')
        self._shmem_ready = False
        assert isinstance(comm_stream_priority, int), (
            f"comm_stream_priority must be an int, got "
            f"{type(comm_stream_priority).__name__}"
        )
        assert isinstance(enable_pdl, bool), (
            f"enable_pdl must be a bool, got {type(enable_pdl).__name__}"
        )
        self.explicitly_destroy = explicitly_destroy
        self.comm_stream_priority = comm_stream_priority
        self.enable_pdl = enable_pdl
        self._comm_stream: torch_npu.npu.Stream | None = None
        self._destroyed = False
        self._ctx = _create_context(
            S, H, K, E, num_ep_ranks,
            num_sms=num_sms,
            token_padding=token_padding,
            B=B,
            group=group,
            rank=rank,
            use_udma=use_udma,
            udma_flags=udma_flags,
        )
        # Buffer-backed: set True by _ensure_shmem_views after init_with_buffer.
        self._shmem_ready = False
        self._comm_stream = torch_npu.npu.Stream(
            device=int(self._ctx['device']),
            priority=self.comm_stream_priority,
        )

    @property
    def destroyed(self) -> bool:
        return self._destroyed

    @property
    def symmetric_footprint(self) -> int:
        """Total CANN VMM bytes of this Buffer's comm chunks + RMA op workspaces
        (i.e. the descriptor size registered with buffer-backed shmem init).

        The framework-side local physical segments of full weight/grad/reduce
        buffers are not counted; per §6 the framework collects
        ``buffer_c.symmetric_descriptors()`` and calls init_with_buffer once.
        """
        ctx = self._ctx
        assert ctx is not None, "Buffer is destroyed"
        return int(ctx['shmem_footprint_bytes'])

    def _require_ctx(self) -> dict:
        assert not self._destroyed, "MoonEP-Ascend Buffer has been destroyed"
        assert self._ctx is not None, "MoonEP-Ascend Buffer is not initialized"
        return self._ctx

    def _ensure_shmem_views(self) -> None:
        """Translate the VMM communication chunks to their kernel-facing shmem
        addresses once the buffer-backed shmem heap is initialized.

        After ``ShmemRuntime.init_with_buffer`` each registered source VA maps
        to a local shmem symmetric address (``map_to_sym_ptr``); the kernels
        address the chunks via that symmetric address (``aclshmem_ptr`` / device
        RMA), so the ctx chunk tensors are rebuilt over the translated pointers.
        Idempotent; raises if shmem is not yet buffer-backedly initialized.
        """
        ctx = self._require_ctx()
        if ctx.get('_shmem_views_ready'):
            return
        if not ShmemRuntime.is_initialized():
            raise RuntimeError(
                "MoonEP-Ascend Buffer requires the buffer-backed shmem heap "
                "(ShmemRuntime.init_with_buffer) before dispatch/combine/"
                "prefetch/reduce_grad; construct all Buffers first, then call "
                "ShmemRuntime.init_with_buffer")
        for name in (
            'hidden_buf',
            'weights_buf',
            'barrier_buf',
            'combine_sync_buffer',
            'dispatch_sync_buffer',
        ):
            allocation = ctx['vmm_chunks'][name]
            sym = map_to_sym_ptr(allocation.source_va, name)
            src = ctx[name]
            ctx[name] = create_sym_tensor_from_ptr(
                sym, src.shape, src.dtype, src.device)
        ctx['hidden_buf_local'] = ctx['hidden_buf'][:ctx['NvS']]
        ctx['weights_buf_local'] = ctx['weights_buf'][:ctx['NvS']]
        ctx['_shmem_views_ready'] = True
        self._shmem_ready = True

    def destroy(self) -> None:
        """Synchronize and release all shmem symmetric resources.

        Collective: every rank must call ``destroy`` together. Must be called
        before tearing down the process group. Idempotent.

        Ordering contract (R4): ``ShmemRuntime.finalize()`` MUST run before
        this method. The Buffer never finalizes shmem itself (that belongs to
        the framework/caller); the VMM physical handles must stay alive until
        ``aclshmem_finalize`` completes (shmem_host_init.h) — freeing physical
        memory before finalize is a use-after-free. See
        kernel_test_utils.destroy_active_buffers.
        """
        if self._destroyed:
            return
        ctx = self._ctx
        if ctx is None:
            self._destroyed = True
            return

        # torch_npu.npu.synchronize() below already drains every stream on the
        # device, including the comm stream, so no separate comm-stream sync.
        torch_npu.npu.synchronize()
        group = ctx.get('group')
        uses_process_group = bool(ctx.get('_uses_process_group', True))
        if uses_process_group and dist.is_initialized():
            dist.barrier(group=group)

        # Collective teardown: kernel ops first (their workspaces), then the
        # VMM communication chunks (unmap -> release -> free_physical), then a
        # trailing barrier so no rank frees memory another rank is still using.
        # Global shmem init/finalize belongs to the framework/caller (the
        # Buffer does not shmem_finalize itself).
        for op in ctx['kernels'].values():
            op.destroy()
        for name in (
            'hidden_buf',
            'weights_buf',
            'barrier_buf',
            'combine_sync_buffer',
            'dispatch_sync_buffer',
        ):
            allocation = ctx['vmm_chunks'].get(name)
            if allocation is not None:
                allocation.destroy()
        if uses_process_group:
            dist.barrier(group=group)

        # Drop views before owners. Tensor deleters must NOT release the shmem
        # allocations — release happened via core.free above.
        for key in (
            'hidden_buf_local',
            'weights_buf_local',
            'hidden_buf',
            'weights_buf',
            'barrier_buf',
            'combine_sync_buffer',
            'planning_workspace',
            'dispatch_sync_buffer',
        ):
            ctx.pop(key, None)
        ctx.clear()
        self._ctx = None
        self._comm_stream = None
        self._destroyed = True

    def __del__(self):
        if getattr(self, "_destroyed", True):
            return
        if getattr(self, "explicitly_destroy", False):
            warnings.warn(
                "MoonEP-Ascend Buffer was not destroyed explicitly; resources may leak.",
                ResourceWarning,
            )
            return
        try:
            self.destroy()
        except Exception as exc:
            warnings.warn(
                f"MoonEP-Ascend Buffer destructor failed to destroy resources: {exc!r}",
                ResourceWarning,
            )

    @staticmethod
    def _record_streams(tensors, stream: torch_npu.npu.Stream) -> None:
        record_streams(tensors, stream)

    @staticmethod
    def _plan_runtime_tensors(plan: MoonEPCommPlan):
        return (
            plan.dst,
            plan.experts_to_copy,
            plan.zero_fill_ranges,
            plan.remote_stats,
            plan.dup_groups,
            plan.dup_loffs,
            plan.dup_counts,
        )

    def _run_dispatch_on_current_stream(
        self,
        ctx: dict,
        hidden_sh: torch.Tensor,
        route_weights_sk: torch.Tensor | None,
        planning_args,
        plan: MoonEPCommPlan,
        hidden_nvsh: torch.Tensor,
        route_weights_nvs: torch.Tensor | None,
        *,
        inter_rank_sync: bool,
        zero_copy: bool,
        router_weights_zero_copy: bool = False,
    ) -> None:
        stream = torch_npu.npu.current_stream().npu_stream
        if inter_rank_sync:
            cross_rank_barrier(ctx, stream)

        if planning_args is not None:
            topk_flat, tokens_per_expert, cu_seqlens = planning_args
            ctx['kernels']['planning'].compute(
                ctx, topk_flat, tokens_per_expert, cu_seqlens, plan,
                shmem_instance_id=0)

        ctx['kernels']['dispatch'].compute(
            ctx,
            hidden_sh,
            route_weights_sk,
            plan,
            build_dedup_map=planning_args is not None,
            shmem_instance_id=0,
        )
        if not zero_copy:
            # master-style boundary copy of the hidden shard (same stream).
            hidden_nvsh.copy_(ctx['hidden_buf_local'])
        if route_weights_nvs is not None and not router_weights_zero_copy:
            route_weights_nvs.copy_(
                ctx['weights_buf_local'].view(torch.float32)
            )

    def _run_combine_on_current_stream(
        self,
        ctx: dict,
        hidden_sh: torch.Tensor,
        plan: MoonEPCommPlan,
        hidden_nvsh: torch.Tensor,
        route_weights_nvs: torch.Tensor | None,
        route_weights_sk: torch.Tensor | None,
        *,
        inter_rank_sync: bool,
        zero_copy: bool,
        router_weights_zero_copy: bool = False,
    ) -> None:
        stream = torch_npu.npu.current_stream().npu_stream
        if not zero_copy:
            # master-style boundary copy of the hidden shard (same stream).
            ctx['hidden_buf_local'].copy_(hidden_nvsh)
        if route_weights_nvs is not None and not router_weights_zero_copy:
            ctx['weights_buf_local'].copy_(
                route_weights_nvs.view(torch.int32)
            )
        # Publish the local shard before either implementation starts reading
        # peer shards.  Keeping this boundary common lets CombineOp own the
        # implementation selection without duplicating Buffer call paths.
        if inter_rank_sync:
            cross_rank_barrier(ctx, stream)
        # Prologue is folded into moonEP_combine (cf. dispatch_epilogue).
        ctx['kernels']['combine'].compute(
            ctx,
            hidden_sh,
            plan,
            output_sk=route_weights_sk,
            shmem_instance_id=0,
        )

    def _run_prefetch_weight_on_current_stream(
        self,
        ctx: dict,
        experts_to_copy: torch.Tensor,
        weight_prefetch_args,
    ) -> None:
        """Prefetch expert weights into the local prefetch slots.

        Buffer-backed only: each ``full_weight`` is a framework-side CANN VMM
        ``[E+B, H, H']`` tensor. Map it to the local symmetric base
        (``map_to_sym_ptr``) and build the op's *local* shmem tensors:
          - ``weight_home``  = this rank's local home chunk (logical shape
            ``[E]``; data_ptr is the local home symmetric base, only rows
            [0, epn) are backed/read);
          - ``weight_slots`` = this rank's local slot chunk, shape
            ``[B, H, H']``.
        This matches the kernel addressing ``owner = expert/epn`` and
        ``source = (expert%epn)*H*Hp`` (analysis §4.2).
        """
        self._ensure_shmem_views()
        E = int(ctx['E'])
        local_experts = experts_to_copy[int(ctx['rank'])]  # this rank's [B] row
        # The op layer requires shmem tensors pointing at the "local part of
        # symmetric memory": this rank's local home chunk / [B] slot chunk,
        # with every rank's chunks occupying the same symmetric slots.
        # The kernel resolves peers via aclshmem_ptr with
        # owner=expert/epn, source=(expert%epn)*H*Hp, where epn is derived
        # from weight_home.shape[0] (logical E), while data_ptr is the local
        # home symmetric base (only rows [0, epn) are backed). Per §6 the
        # framework registers the full weight's *local physical segments* and
        # attaches this rank's source VAs to the tensor:
        #   _moonep_owner_source_va -- source VA of the local home chunk
        #   _moonep_slot_source_va  -- source VA of the local slot chunk
        # Buffer maps them to local symmetric bases via get_buffer_ptr and
        # builds the shmem tensors.
        B = int(ctx['B'])
        for full_weight in weight_prefetch_args:
            H = int(full_weight.shape[1])
            Hp = int(full_weight.shape[2])
            owner_src = getattr(full_weight, _OWNER_SOURCE_VA_ATTR, None)
            slot_src = getattr(full_weight, _SLOT_SOURCE_VA_ATTR, None)
            if owner_src is None or slot_src is None:
                raise RuntimeError(
                    "full_weight is missing _moonep_owner_source_va / "
                    "_moonep_slot_source_va: the framework must attach this "
                    "rank's local home/slot chunk source VAs (buffer_c."
                    "attach_local_source_vas) so the Buffer can translate them "
                    "via get_buffer_ptr (analysis §4.2 / §6)")
            # Logical shape [E] (the kernel derives epn from it); data_ptr is
            # the local home base.
            weight_home = create_sym_tensor_from_source(
                owner_src, [E, H, Hp], torch.bfloat16, full_weight.device,
                'weight_home', allow_logical_extent=True)
            weight_slots = create_sym_tensor_from_source(
                slot_src, [B, H, Hp], torch.bfloat16, full_weight.device,
                'weight_slots')
            ctx['kernels']['prefetch'].compute(
                ctx,
                weight_home,
                weight_slots,
                local_experts,
                shmem_instance_id=0,
            )

    def _run_reduce_grad_on_current_stream(
        self,
        ctx: dict,
        experts_to_copy: torch.Tensor,
        projection_grads,
        reduce_buffers,
    ) -> None:
        """Reduce duplicated experts' weight grads back to their home ranks.

        Buffer-backed only: ``full_grad`` (``[E+B,H,H']`` fp32) and
        ``reduce_buffer`` (the ``[R,B,H,H']`` compatibility view or a
        ``[B,H,H']`` local slot carrier) are framework-side VMM tensors; map
        them to local symmetric bases and build the op's *local* shmem tensors
        (analysis §4.3):
          - ``full_grad_home`` = this rank's home grad rows (physical
            ``[E/R]``; data_ptr points at the local home base);
          - ``reduce_slots``   = this rank's local reduction slots
            ``[B, H, H']``.
        ``experts_to_copy`` still carries the full ``[R, B]`` plan.
        """
        self._ensure_shmem_views()
        rank = int(ctx['rank'])
        E = int(ctx['E'])
        R = int(ctx['R'])
        if R <= 0 or E % R:
            raise RuntimeError(
                f'invalid expert layout for grad reduce: E={E}, R={R}')
        epn = E // R
        B = int(ctx['B'])
        num_sms = int(ctx['num_sms'])
        for full_grad, reduce_buffer in zip(projection_grads, reduce_buffers):
            # Buffer-backed only: full_grad / reduce_buffer -> local shmem
            # tensors. The framework attaches this rank's local physical
            # segment source VAs (_moonep_owner_source_va /
            # _moonep_slot_source_va), which the Buffer maps to local symmetric
            # bases via get_buffer_ptr:
            #   full_grad_home = symmetric base of this rank's home grad chunk
            #     (physical shape [epn=E/R], data_ptr on the local home
            #     backing). The GradReduce native ABI accepts epn directly
            #     since 43aba6a; faking this view as [E] makes the kernel run
            #     past the caller-owned physical allocation when EP > 1.
            #   reduce_slots   = symmetric base of this rank's local reduce
            #     slot chunk.
            H = int(full_grad.shape[1])
            Hp = int(full_grad.shape[2])
            owner_src = getattr(full_grad, _OWNER_SOURCE_VA_ATTR, None)
            slot_src = getattr(reduce_buffer, _SLOT_SOURCE_VA_ATTR, None)
            if owner_src is None or slot_src is None:
                raise RuntimeError(
                    "full_grad / reduce_buffer missing "
                    "_moonep_owner_source_va / _moonep_slot_source_va: the "
                    "framework must attach this rank's local home/slot chunk "
                    "source VAs (buffer_c.attach_local_source_vas) so the "
                    "Buffer can translate them via get_buffer_ptr (analysis "
                    "§4.3 / §6)")
            full_grad_home = create_sym_tensor_from_source(
                owner_src, [epn, H, Hp], torch.float32, full_grad.device,
                'full_grad_home')
            reduce_slots = create_sym_tensor_from_source(
                slot_src, [B, H, Hp], torch.float32, reduce_buffer.device,
                'reduce_slots')
            ctx['kernels']['grad_reduce'].compute(
                ctx,
                full_grad_home,
                reduce_slots,
                experts_to_copy,
                rank=rank,
                num_sms=num_sms,
                shmem_instance_id=0,
            )

    # ------------------------------------------------------------------
    # Public API (identical signatures to moonep.api.Buffer)
    # ------------------------------------------------------------------
    def dispatch(
        self,
        hidden_sh: torch.Tensor,
        route_weights_sk: torch.Tensor | None = None,
        topk_experts_sk: torch.Tensor | None = None,
        tokens_per_expert: torch.Tensor | None = None,
        plan: MoonEPCommPlan | None = None,
        async_finish: bool = False,
        *,
        inter_rank_sync: bool = True,
        zero_copy: bool = False,
        router_weights_zero_copy: bool = False,
    ):
        """dispatch fwd: run planning (unless reusing a plan) and scatter tokens
        to their expert-grouped positions on remote ranks.

        Same contract as ``moonep.api.Buffer.dispatch``; ``zero_copy`` returns a
        hidden view of the local symmetric shard (which remote ranks have written
        into via the peer address), and ``router_weights_zero_copy`` independently
        returns the ``route_weights_nvs`` fp32 view of ``weights_buf_local``
        instead of a fresh copy (default False, matching the CUDA API).
        """
        ctx = self._require_ctx()
        self._ensure_shmem_views()

        if plan is None:
            assert topk_experts_sk is not None and tokens_per_expert is not None
            topk_flat = topk_experts_sk.reshape(-1)
            assert topk_flat.dtype == torch.int32 and topk_flat.numel() == int(ctx['N'])
            assert tokens_per_expert.dtype == torch.int32
            assert tokens_per_expert.numel() == int(ctx['E']) and tokens_per_expert.is_contiguous()
            plan, cu_seqlens = allocate_planning_outputs(ctx)
            planning_args = (topk_flat, tokens_per_expert, cu_seqlens)
        else:
            cu_seqlens = None
            planning_args = None
            assert isinstance(plan, MoonEPCommPlan)

        if zero_copy:
            hidden_nvsh = ctx['hidden_buf_local']
        else:
            hidden_nvsh = torch.empty_like(ctx['hidden_buf_local'])
        if route_weights_sk is None:
            route_weights_nvs = None
        elif router_weights_zero_copy:
            route_weights_nvs = ctx['weights_buf_local'].view(torch.float32)
        else:
            route_weights_nvs = torch.empty(
                ctx['NvS'], dtype=torch.float32, device=ctx['weights_buf'].device)

        if not async_finish:
            self._run_dispatch_on_current_stream(
                ctx,
                hidden_sh,
                route_weights_sk,
                planning_args,
                plan,
                hidden_nvsh,
                route_weights_nvs,
                inter_rank_sync=inter_rank_sync,
                zero_copy=zero_copy,
                router_weights_zero_copy=router_weights_zero_copy,
            )
            return hidden_nvsh, route_weights_nvs, cu_seqlens, plan

        # True async: launch the kernels on a dedicated comm_stream and return
        # its event without blocking the host. Cross-rank sync runs on
        # comm_stream via cross_rank_barrier's on-stream primitive
        # (acl_shmem_barrier_all_on_stream); no torch.npu.synchronize +
        # dist.barrier needed. Back-to-back async ops use _barrier_round
        # generations to avoid deadlocking on the shared sync_counter.
        tensors = (
            hidden_sh, route_weights_sk, hidden_nvsh, route_weights_nvs,
            plan.dst, plan.experts_to_copy, plan.zero_fill_ranges,
            plan.remote_stats, plan.dup_groups, plan.dup_loffs,
            plan.dup_counts,
        )
        # First make comm_stream wait for sync work already queued on the
        # current stream (e.g. a preceding dispatch) to drain, then launch
        # async on comm_stream; otherwise the two streams race over the same
        # dependent data and the device-side barrier desyncs and hangs.
        # wait_stream is an inter-stream event handshake (no host block).
        prev_stream = torch_npu.npu.current_stream()
        with torch_npu.npu.stream(self._comm_stream):
            self._comm_stream.wait_stream(prev_stream)
            self._run_dispatch_on_current_stream(
                ctx,
                hidden_sh,
                route_weights_sk,
                planning_args,
                plan,
                hidden_nvsh,
                route_weights_nvs,
                inter_rank_sync=inter_rank_sync,
                zero_copy=zero_copy,
                router_weights_zero_copy=router_weights_zero_copy,
            )
            self._record_streams(tensors, self._comm_stream)
        done = self._comm_stream.record_event()

        return hidden_nvsh, route_weights_nvs, cu_seqlens, plan, done

    def prefetch_weight(
        self,
        plan: MoonEPCommPlan | None = None,
        async_finish: bool = False,
        *,
        full_gate_weight: torch.Tensor | None = None,
        full_up_weight: torch.Tensor | None = None,
        full_down_weight: torch.Tensor | None = None,
    ):
        """Prefetch the remote expert weights selected by ``plan`` into the
        local prefetch slots (dispatch fwd, weight side)."""
        weight_prefetch_args = (full_gate_weight, full_up_weight, full_down_weight)
        assert all(w is not None for w in weight_prefetch_args), \
            "prefetch_weight tensors must be provided together"
        return self.prefetch_projections(
            plan=plan,
            projections=weight_prefetch_args,
            async_finish=async_finish,
        )

    def prefetch_projections(
        self,
        plan: MoonEPCommPlan | None,
        projections: Sequence[torch.Tensor],
        async_finish: bool = False,
    ):
        """Prefetch an arbitrary non-empty sequence of expert projections.

        Every projection is a contiguous bf16 ``[E+B, M, N]`` tensor with
        ``_moonep_owner_source_va`` and ``_moonep_slot_source_va`` attached.
        ``M`` and ``N`` are derived per projection, so FC1 ``[hidden, ffn]``
        and FC2 ``[ffn, hidden]`` may be supplied in the same call.
        This is the minimal framework-facing API; ``prefetch_weight`` remains
        the three-projection compatibility wrapper.
        """
        ctx = self._require_ctx()
        assert isinstance(plan, MoonEPCommPlan), \
            "Buffer.prefetch_projections: plan is required"
        assert isinstance(projections, Sequence) and len(projections) > 0, \
            "prefetch_projections requires a non-empty projection sequence"
        projections = tuple(projections)
        for projection in projections:
            assert projection.dtype == torch.bfloat16 and projection.is_contiguous()
            assert projection.ndim == 3
            assert int(projection.shape[0]) == int(ctx['E']) + int(ctx['B'])
            assert all(int(dimension) > 0 for dimension in projection.shape[1:])

        if not async_finish:
            self._run_prefetch_weight_on_current_stream(
                ctx,
                plan.experts_to_copy,
                projections,
            )
            return None

        # True async: launch on the dedicated comm_stream and return its event
        # without blocking the host.
        prev_stream = torch_npu.npu.current_stream()
        with torch_npu.npu.stream(self._comm_stream):
            self._comm_stream.wait_stream(prev_stream)
            self._run_prefetch_weight_on_current_stream(
                ctx,
                plan.experts_to_copy,
                projections,
            )
            self._record_streams(
                (*projections, plan.experts_to_copy), self._comm_stream)
        done = self._comm_stream.record_event()

        return done

    def combine(
        self,
        plan: MoonEPCommPlan | None = None,
        hidden_nvsh: torch.Tensor | None = None,
        route_weights_nvs: torch.Tensor | None = None,
        async_finish: bool = False,
        inter_rank_sync: bool = True,
        *,
        zero_copy: bool = False,
        router_weights_zero_copy: bool = False,
    ):
        """combine fwd: gather expert outputs from the symmetric shard and
        K-sum back to token-major [S, H]. Also serves as dispatch bwd."""
        ctx = self._require_ctx()
        self._ensure_shmem_views()

        assert isinstance(plan, MoonEPCommPlan), "Buffer.combine: plan is required"

        assert hidden_nvsh is not None
        assert hidden_nvsh.dtype == torch.bfloat16
        assert hidden_nvsh.is_contiguous()
        assert tuple(hidden_nvsh.shape) == (int(ctx['NvS']), int(ctx['H']))
        if route_weights_nvs is not None:
            assert route_weights_nvs.dtype == torch.float32
            assert route_weights_nvs.is_contiguous()
            assert tuple(route_weights_nvs.shape) == (int(ctx['NvS']),)
        if zero_copy:
            assert hidden_nvsh.data_ptr() == ctx['hidden_buf_local'].data_ptr(), (
                "combine(zero_copy=True): hidden_nvsh must alias the shard view "
                "returned by dispatch(zero_copy=True)"
            )
        if router_weights_zero_copy and route_weights_nvs is not None:
            assert route_weights_nvs.data_ptr() == \
                ctx['weights_buf_local'].data_ptr(), (
                "combine(router_weights_zero_copy=True): route_weights_nvs must "
                "alias the weights view returned by "
                "dispatch(router_weights_zero_copy=True)"
            )

        hidden_sh = torch.empty(
            int(ctx['S']),
            int(ctx['H']),
            dtype=hidden_nvsh.dtype,
            device=hidden_nvsh.device,
        )
        route_weights_sk = (
            torch.empty(
                int(ctx['S']),
                int(ctx['K']),
                dtype=torch.float32,
                device=hidden_nvsh.device,
            )
            if route_weights_nvs is not None else None
        )

        if not async_finish:
            self._run_combine_on_current_stream(
                ctx,
                hidden_sh,
                plan,
                hidden_nvsh,
                route_weights_nvs,
                route_weights_sk,
                inter_rank_sync=inter_rank_sync,
                zero_copy=zero_copy,
                router_weights_zero_copy=router_weights_zero_copy,
            )
            return hidden_sh, route_weights_sk, None

        # True async: launch on the dedicated comm_stream and return its event
        # without blocking the host.
        tensors = (
            hidden_nvsh, route_weights_nvs, hidden_sh, route_weights_sk,
            plan.dst, plan.experts_to_copy, plan.zero_fill_ranges,
            plan.remote_stats, plan.dup_groups, plan.dup_loffs,
            plan.dup_counts,
        )
        prev_stream = torch_npu.npu.current_stream()
        with torch_npu.npu.stream(self._comm_stream):
            self._comm_stream.wait_stream(prev_stream)
            self._run_combine_on_current_stream(
                ctx,
                hidden_sh,
                plan,
                hidden_nvsh,
                route_weights_nvs,
                route_weights_sk,
                inter_rank_sync=inter_rank_sync,
                zero_copy=zero_copy,
                router_weights_zero_copy=router_weights_zero_copy,
            )
            self._record_streams(tensors, self._comm_stream)
        done = self._comm_stream.record_event()

        return hidden_sh, route_weights_sk, done

    def reduce_grad(
        self,
        plan: MoonEPCommPlan | None = None,
        async_finish: bool = False,
        full_gate_grad: torch.Tensor | None = None,
        full_up_grad: torch.Tensor | None = None,
        full_down_grad: torch.Tensor | None = None,
        gate_reduce_buffer: torch.Tensor | None = None,
        up_reduce_buffer: torch.Tensor | None = None,
        down_reduce_buffer: torch.Tensor | None = None,
    ):
        """Reduce duplicated experts' weight grads back to their home ranks
        (dispatch bwd, weight side).

        Same layout contract as ``moonep.api.Buffer.reduce_grad``: each
        ``full_*_grad`` is ``[E+B, H, H']`` fp32 — rows ``[0, E)`` are this
        rank's home grads, rows ``[E, E+B)`` are backed by the reduce buffer —
        and each ``*_reduce_buffer`` is the full ``[R, B, H, H']`` fp32 view
        (all R ranks' reduce buffers mapped as one).
        """
        projection_grads = (full_gate_grad, full_up_grad, full_down_grad)
        reduce_buffers = (
            gate_reduce_buffer, up_reduce_buffer, down_reduce_buffer)
        assert all(t is not None for t in (*projection_grads, *reduce_buffers)), \
            "reduce_grad tensors must be provided together"
        # Preserve the legacy entry point and its historical [R,B,H,H'] view;
        # only the generic entry point exposes local [B,H,H'] carriers.
        ctx = self._require_ctx()
        for reduce_buffer in reduce_buffers:
            assert reduce_buffer.ndim == 4
            assert tuple(reduce_buffer.shape[:2]) == (
                int(ctx['R']), int(ctx['B']))
        return self.reduce_projection_grads(
            plan=plan,
            projection_grads=tuple(zip(projection_grads, reduce_buffers)),
            async_finish=async_finish,
        )

    def reduce_projection_grads(
        self,
        plan: MoonEPCommPlan | None,
        projection_grads: (
            Sequence[tuple[torch.Tensor, torch.Tensor]] |
            Sequence[torch.Tensor]
        ),
        reduce_buffers: Sequence[torch.Tensor] | None = None,
        async_finish: bool = False,
    ):
        """Reduce an arbitrary non-empty sequence of projection gradients.

        The primary contract is ``projection_grads=Sequence[(full_grad,
        reduce_buffer)]``. ``full_grad`` is contiguous fp32 ``[E+B,M,N]`` and
        carries this rank's ``_moonep_owner_source_va``. ``reduce_buffer`` may be
        either the historical cross-rank compatibility view
        ``[R,B,M,N]`` or a contiguous local ``[B,M,N]`` carrier.  In both
        cases it must carry this rank's ``_moonep_slot_source_va``; the kernel
        performs peer addressing from that local symmetric slot VA, so a
        framework does not need to construct a secondary peer-import alias.

        For source compatibility, callers may alternatively pass separate
        ``projection_grads`` and ``reduce_buffers`` sequences.
        """
        ctx = self._require_ctx()
        assert isinstance(plan, MoonEPCommPlan), \
            "Buffer.reduce_projection_grads: plan is required"
        assert isinstance(projection_grads, Sequence) and len(projection_grads) > 0, \
            "reduce_projection_grads requires a non-empty projection sequence"
        if reduce_buffers is None:
            projection_pairs = tuple(projection_grads)
            assert all(
                isinstance(pair, Sequence) and len(pair) == 2
                for pair in projection_pairs
            ), "projection_grads must contain (full_grad, reduce_buffer) pairs"
            projection_grads = tuple(pair[0] for pair in projection_pairs)
            reduce_buffers = tuple(pair[1] for pair in projection_pairs)
        else:
            assert isinstance(reduce_buffers, Sequence), \
                "reduce_buffers must be a sequence"
            assert len(projection_grads) == len(reduce_buffers), \
                "projection_grads and reduce_buffers must have equal length"
            projection_grads = tuple(projection_grads)
            reduce_buffers = tuple(reduce_buffers)
        for full_grad, reduce_buffer in zip(projection_grads, reduce_buffers):
            assert full_grad.dtype == torch.float32 and full_grad.is_contiguous()
            assert full_grad.ndim == 3
            assert int(full_grad.shape[0]) == int(ctx['E']) + int(ctx['B'])
            assert all(int(dimension) > 0 for dimension in full_grad.shape[1:])
            matrix_shape = tuple(int(dimension) for dimension in full_grad.shape[1:])
            assert reduce_buffer.dtype == torch.float32
            local_shape = (int(ctx['B']), *matrix_shape)
            compatibility_shape = (int(ctx['R']), *local_shape)
            assert tuple(reduce_buffer.shape) in (local_shape, compatibility_shape), \
                ("reduce buffer must be local [B,M,N] or compatibility "
                 "[R,B,M,N]")
            if reduce_buffer.ndim == 3:
                assert reduce_buffer.is_contiguous(), \
                    "local [B,M,N] reduce-buffer carrier must be contiguous"
            assert getattr(reduce_buffer, _SLOT_SOURCE_VA_ATTR, None) is not None, \
                "reduce buffer must carry _moonep_slot_source_va"

        if not async_finish:
            self._run_reduce_grad_on_current_stream(
                ctx,
                plan.experts_to_copy,
                projection_grads,
                reduce_buffers,
            )
            return None

        # True async: launch on the dedicated comm_stream and return its event
        # without blocking the host.
        prev_stream = torch_npu.npu.current_stream()
        with torch_npu.npu.stream(self._comm_stream):
            self._comm_stream.wait_stream(prev_stream)
            self._run_reduce_grad_on_current_stream(
                ctx,
                plan.experts_to_copy,
                projection_grads,
                reduce_buffers,
            )
            recordable_reduce_buffers = tuple(
                getattr(reduce_buffer, "_npu_slot_chunk", reduce_buffer)
                for reduce_buffer in reduce_buffers
            )
            self._record_streams(
                (*projection_grads, *recordable_reduce_buffers,
                 plan.experts_to_copy),
                self._comm_stream,
            )
        done = self._comm_stream.record_event()

        return done
