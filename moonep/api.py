"""
MoonEP Top-level API.

Notation: S = input tokens per rank, K = routed top-k per token, E = total routed
experts in the EP group, R = number of EP ranks (EP comm size), B = weight
prefetch slots per rank, NvS = dispatched token slots per rank (S*K real
tokens plus per-VM-group padding), H = hidden size, H' = expert FFN
intermediate size.

Usage:
    buffer = Buffer(S, H, K, E, num_ep_ranks, num_sms=None, token_padding=128)

    # dispatch fwd
    hidden_nvsh, route_weights_nvs, cu_seqlens, plan = buffer.dispatch(
        hidden_sh, route_weights_sk, topk_experts_sk, tokens_per_expert,
    )
    buffer.prefetch_weight(
        plan=plan,
        full_gate_weight=full_gate_weight,
        full_up_weight=full_up_weight,
        full_down_weight=full_down_weight,
    )

    # combine fwd
    output_sh, gathered_route_weights_sk, _ = buffer.combine(
        plan=plan,
        hidden_nvsh=expert_output_nvsh,
        route_weights_nvs=route_weights_nvs,
    )

    # combine bwd: re-dispatch the output grad with the saved plan
    grad_expert_output_nvsh, _, _, _ = buffer.dispatch(grad_output_sh, plan=plan)

    # dispatch bwd: combine the hidden grad back to token-major, and reduce
    # duplicated experts' weight grads to their home ranks
    grad_hidden_sh, _, _ = buffer.combine(plan=plan, hidden_nvsh=grad_hidden_nvsh)
    buffer.reduce_grad(
        plan=plan,
        full_gate_grad=full_gate_grad,
        full_up_grad=full_up_grad,
        full_down_grad=full_down_grad,
        gate_reduce_buffer=gate_reduce_buffer,
        up_reduce_buffer=up_reduce_buffer,
        down_reduce_buffer=down_reduce_buffer,
    )

    buffer.destroy()
"""

import logging
import os
import warnings

import torch
import torch.distributed as dist

from .buffer import (
    create_nvl_dist_tensor,
    create_nvl_dist_multicast_tensor,
    pad_dim0_for_alignment,
    get_vmm_granularity,
    get_multicast_granularity,
)
from .constants import DEDUP_BUILDER_WARPS
from .planning import MoonEPCommPlan, allocate_planning_outputs, launch_planning
from .inter_rank_sync import launch_inter_rank_sync
from .dispatch import launch_dispatch
from .dispatch_epilogue import launch_dispatch_epilogue
from .combine import launch_combine
from .combine_prologue import launch_combine_prologue
from .prefetch import launch_prefetch
from .grad_reduce import launch_grad_reduce

logger = logging.getLogger(__name__)


def _align_up(x: int, alignment: int) -> int:
    """Round x up to a multiple of alignment."""
    return ((x + alignment - 1) // alignment) * alignment


def _num_sms_dedup_from_env(max_sms: int) -> int:
    """Resolve the local epilogue/prologue SM count.

    ``MOONEP_NUM_SMS_DEDUP`` is intentionally an environment override rather
    than a Buffer argument so benchmark jobs can sweep it without touching the
    public API surface.
    """
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
    hidden_mapped_bytes = tensor_nbytes(ctx['hidden_buf'])
    meta_mapped_bytes = tensor_nbytes(ctx['meta_buf'])
    hidden_chunk_bytes = hidden_mapped_bytes // R
    meta_chunk_bytes = meta_mapped_bytes // R
    local_temp_bytes = sum(
        tensor_nbytes(ctx[name])
        for name in (
            'alloc',
            'group_tokens',
            'z',
            'local_hist',
            'grid_sync_bar',
            'primary_packed',
            'kmask',
            'kidx_to_loff',
            'builder_bar',
        )
    )

    total_buffer_bytes = hidden_chunk_bytes + meta_chunk_bytes + local_temp_bytes

    log_lines = [
        "MoonEP comm buffer sizes:",
        "  config                 : "
        f"R={ctx['R']}, S={ctx['S']}, H={ctx['H']}, E={ctx['E']}, "
        f"K={ctx['K']}, NvS={ctx['NvS']}, NvS_padded={ctx['NvS_padded']}",
        "  sm_count               : "
        f"num_sms={ctx['num_sms']}, num_sms_dedup={ctx['num_sms_dedup']}",
        f"  hidden_buffer_size     : {format_nbytes(hidden_chunk_bytes)}",
        f"  meta_buffer_size       : {format_nbytes(meta_chunk_bytes)}",
        f"  local_temp_buffer_size : {format_nbytes(local_temp_bytes)}",
        f"  total_size             : {format_nbytes(total_buffer_bytes)}",
    ]
    logger.info("\n".join(log_lines))


def _launch_full_weight_prefetches(
    ctx,
    full_gate_weight: torch.Tensor,
    full_up_weight: torch.Tensor,
    full_down_weight: torch.Tensor,
    experts_to_copy: torch.Tensor,
) -> None:
    E = int(ctx['E'])
    num_sms = int(ctx['num_sms'])
    for full_weight in (full_gate_weight, full_up_weight, full_down_weight):
        launch_prefetch(
            full_weight[:E],
            full_weight[E:],
            experts_to_copy,
            num_sms=num_sms,
        )


def _launch_full_grad_reduces(
    ctx,
    experts_to_copy: torch.Tensor,
    full_gate_grad: torch.Tensor,
    full_up_grad: torch.Tensor,
    full_down_grad: torch.Tensor,
    gate_reduce_buffer: torch.Tensor,
    up_reduce_buffer: torch.Tensor,
    down_reduce_buffer: torch.Tensor,
) -> None:
    E = int(ctx['E'])
    B = int(ctx['B'])
    rank = int(ctx['rank'])
    num_sms = int(ctx['num_sms'])
    for name, full_grad, reduce_buffer in (
        ("gate", full_gate_grad, gate_reduce_buffer),
        ("up", full_up_grad, up_reduce_buffer),
        ("down", full_down_grad, down_reduce_buffer),
    ):
        assert full_grad is not None, f"full_{name}_grad is required"
        assert full_grad.dtype == torch.float32 and full_grad.is_contiguous(), \
            f"full_{name}_grad must be contiguous fp32 [E+B, H, H']"
        assert full_grad.ndim == 3 and int(full_grad.shape[0]) == E + B, \
            f"full_{name}_grad first dim must be E+B"
        launch_grad_reduce(
            full_grad[:E],
            reduce_buffer,
            experts_to_copy,
            rank=rank,
            num_sms=num_sms,
            meta_buf=ctx['meta_buf'],
            meta_stride=int(ctx['meta_chunk_padded']),
            barrier_off=int(ctx['BARRIER_OFF']),
            grid_sync_bar=ctx['grid_sync_bar'],
        )


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
) -> dict:
    """Pre-allocate all NVLink shared buffers and local temp buffers.

    The extra logical NvS slots from token_padding are allocated as
    (token_padding - 1) * 2 * E / num_ep_ranks; the physical allocation is
    still aligned to VMM granularity.
    NVLink shared buffer layout:
      - hidden_buf: [NvS_padded, H] bf16 — separate allocation, padded for VMM alignment
      - meta_buf:   [meta_chunk_padded] int32 — merged allocation containing:
          [0, NvS):                weights (fp32, alias as int32)
          [NvS, NvS + R*E):       tpe gather (only rank 0 is the read/write target; symmetric reserve)
          [NvS + R*E, ...):       planning internal scratch/staging
          [..., +N4):             topk0 offload scratch
          [..., +N4):             ORDER, c1 order consumed by c2/passB
          [..., +N4):             ORDER0, rank1 rank0-c1 scratch
          [..., +3):              cross-rank phase barrier
          [..., +NvS):            src_info scratch published by planning and
                                  consumed by the fresh dispatch dedup builder;
                                  non-negative entries encode src_rank * NvS + offv,
                                  -1 marks empty slots
    """
    if num_sms is None:
        num_sms = 32
    assert isinstance(num_sms, int) and num_sms > 0, \
        f"num_sms must be a positive int, got {num_sms}"
    rank = dist.get_rank(group=group)
    R = num_ep_ranks
    assert R == dist.get_world_size(group=group), (
        f"num_ep_ranks ({R}) must equal group world size "
        f"({dist.get_world_size(group=group)})"
    )
    N = S * K
    device = torch.cuda.current_device()
    dev = f"cuda:{device}"
    max_sms = torch.cuda.get_device_properties(device).multi_processor_count
    num_sms_dedup = _num_sms_dedup_from_env(max_sms)

    epn = E // R
    assert E % R == 0, f"E ({E}) must be divisible by R ({R})"
    assert isinstance(token_padding, int) and token_padding > 0, \
        f"token_padding must be a positive int, got {token_padding}"
    if B is None:
        B = epn
    assert isinstance(B, int) and B > 0, f"B must be a positive int, got {B}"

    NvS_capacity = S * K

    # The current constructive planner lets each destination rank receive
    # tokens from at most one remote home group. Each home group has E/R
    # experts, so each rank has at most E/R remote expert segments, plus at
    # most E/R local expert segments. Each non-empty segment is padded up
    # from at least 1 real token slot to a multiple of token_padding, so the
    # extra logical NvS slot bound is (token_padding - 1) * 2 * E/R. The
    # physical allocation below is still aligned to VMM granularity via
    # pad_dim0_for_alignment() / pad_to_granularity().
    token_padding_extra = (token_padding - 1) * 2 * epn
    NvS = NvS_capacity + token_padding_extra

    # ================================================================
    # Planning constexpr computation and int32 range guards
    # ================================================================
    BLOCK_SIZE_P2 = 2048

    int32_max = 2**31 - 1
    assert 0 < N < int32_max, (
        "planning requires 0 < S*K < int32_max: "
        f"S={S}, K={K}, S*K={N}, int32_max={int32_max}"
    )
    num_vblocks = (N + BLOCK_SIZE_P2 - 1) // BLOCK_SIZE_P2

    # ================================================================
    # Compute padded dimensions for VMM alignment
    # ================================================================
    NvS_padded = pad_dim0_for_alignment([NvS, H], torch.bfloat16)

    # Meta offsets are int32 elements. Vectorized regions are 16B-aligned.
    WEIGHTS_OFF = 0
    TPE_OFF = _align_up(NvS, 4)
    PLAN_OFF = _align_up(TPE_OFF + R * E, 4)
    broadcast_elems = 3 * E * R
    assert broadcast_elems % 4 == 0, (
        f"broadcast_elems ({broadcast_elems}) must be divisible by 4"
    )
    planning_out_elems = (
        broadcast_elems
        + R * (E + B)
        + 2 * R * (E + B)
        + B * R
        + 2 * R
    )

    # C1 scratch segments are padded so bulk copies can stay vectorized.
    N4 = _align_up(N, 4)
    TOPK0_OFF = _align_up(PLAN_OFF + planning_out_elems, 4)
    ORDER_OFF = TOPK0_OFF + N4
    ORDER0_OFF = ORDER_OFF + N4
    BARRIER_OFF = ORDER0_OFF + N4
    # cross_rank_barrier uses 3 int32 per rank: 2 phase signals + 1 phase/sign
    # counter (independent of rank count, self-resetting double buffer).
    BARRIER_SLOTS = 3
    SRC_INFO_OFF = BARRIER_OFF + BARRIER_SLOTS
    meta_chunk_logical = SRC_INFO_OFF + NvS

    # Chunk size must satisfy both VMM mapping and multicast binding.
    gran = get_vmm_granularity()
    mc_gran = get_multicast_granularity(R)
    chunk_align_bytes = max(gran, mc_gran)
    meta_chunk_bytes = meta_chunk_logical * 4
    meta_chunk_padded = _align_up(meta_chunk_bytes, chunk_align_bytes) // 4

    # Some CuTe DSL address expressions multiply a runtime Int32 rank/drank by
    # these constexpr strides, so guard the largest reachable nonnegative index.
    max_meta_index = (R - 1) * meta_chunk_padded + meta_chunk_logical - 1
    assert max_meta_index <= int32_max, (
        "meta_buf rank-stride indexing would overflow CuTe Int32 arithmetic: "
        f"max_index={max_meta_index}, R={R}, S={S}, K={K}, N={N}, "
        f"NvS={NvS}, meta_chunk_logical={meta_chunk_logical}, "
        f"meta_chunk_padded={meta_chunk_padded}, int32_max={int32_max}"
    )
    max_hidden_index = (R - 1) * NvS_padded + NvS - 1
    assert max_hidden_index <= int32_max, (
        "hidden_buf/dst rank-stride indexing would overflow CuTe Int32 arithmetic: "
        f"max_index={max_hidden_index}, R={R}, S={S}, K={K}, N={N}, "
        f"NvS={NvS}, NvS_padded={NvS_padded}, int32_max={int32_max}"
    )

    # ================================================================
    # Allocate NVLink shared buffers
    # ================================================================
    hidden_buf = create_nvl_dist_tensor([NvS_padded, H], torch.bfloat16, rank, R, group=group)
    meta_buf, meta_mc = create_nvl_dist_multicast_tensor(
        [meta_chunk_padded], torch.int32, rank, R, group=group,
    )

    # Zero the barrier region on all ranks
    for r in range(R):
        meta_buf[r * meta_chunk_padded + BARRIER_OFF:
                 r * meta_chunk_padded + BARRIER_OFF + BARRIER_SLOTS].zero_()
    dist.barrier(group=group)


    # ================================================================
    # Local buffers
    # ================================================================
    alloc = torch.empty(E * R, dtype=torch.int32, device=dev)
    group_tokens = torch.empty(R, dtype=torch.int32, device=dev)
    z = torch.empty(R * R, dtype=torch.int32, device=dev)
    local_hist = torch.empty(E * num_vblocks, dtype=torch.int32, device=dev)
    # Global arrive counter (1 int32) for the software grid barrier: the
    # equivalent of cudaLaunchCooperative's grid workspace. Self-resetting
    # (sentinel-bit flip), allocated once and shared by planning/dispatch/
    # combine/inter_rank, with no per-launch zeroing needed.
    grid_sync_bar = torch.zeros(1, dtype=torch.int32, device=dev)
    # Dispatch builder scratch for fresh planning. ``kidx_to_loff`` maps each
    # (source rank, token, topk index) key to the local NvS slot used later when
    # expanding duplicate groups. ``builder_bar`` is a self-resetting TAG
    # counter, so it is initialized once like grid_sync_bar.
    primary_packed = torch.empty(R * S, dtype=torch.int32, device=dev)
    kmask = torch.empty(R * S, dtype=torch.int32, device=dev)
    kidx_to_loff = torch.empty(R * S * K, dtype=torch.int32, device=dev)
    builder_bar = torch.zeros(1, dtype=torch.int32, device=dev)

    ctx = {
        'rank': rank,
        'group': group,
        'R': R, 'E': E, 'S': S, 'K': K, 'H': H,
        'B': B,
        'N': N, 'NvS': NvS,
        'NvS_capacity': NvS_capacity,
        'NvS_padded': NvS_padded,
        'num_sms': num_sms,
        'num_sms_dedup': num_sms_dedup,
        'token_padding': token_padding,
        'device': device,
        'token_padding_extra': token_padding_extra,
        # NVLink shared
        'hidden_buf': hidden_buf,
        'meta_buf': meta_buf,
        'meta_mc': meta_mc,
        # Meta layout
        'meta_chunk_padded': meta_chunk_padded,
        'WEIGHTS_OFF': WEIGHTS_OFF,
        'TPE_OFF': TPE_OFF,
        'PLAN_OFF': PLAN_OFF,
        'TOPK0_OFF': TOPK0_OFF,
        'ORDER_OFF': ORDER_OFF,
        'ORDER0_OFF': ORDER0_OFF,
        'BARRIER_OFF': BARRIER_OFF,
        'SRC_INFO_OFF': SRC_INFO_OFF,
        # Local temps
        'alloc': alloc, 'group_tokens': group_tokens,
        'z': z,
        'local_hist': local_hist,
        'grid_sync_bar': grid_sync_bar,
        'primary_packed': primary_packed,
        'kmask': kmask,
        'kidx_to_loff': kidx_to_loff,
        'builder_bar': builder_bar,
        'num_vblocks': num_vblocks,
        # Local views
        'hidden_buf_local': hidden_buf[rank * NvS_padded: rank * NvS_padded + NvS],
        'weights_buf_local': meta_buf[rank * meta_chunk_padded + WEIGHTS_OFF:
                                      rank * meta_chunk_padded + WEIGHTS_OFF + NvS],
    }
    _log_context_buffer_size(ctx)
    return ctx


class Buffer:
    """MoonEP communication buffer owner.

    A Buffer owns the NVLink/VMM communication buffers, local scratch tensors,
    and the async communication stream used by dispatch/prefetch/combine/reduce. Call
    ``destroy()`` before tearing down the process group.
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
    ):
        """Allocate and hold all communication buffers.

        Args:
            S: input tokens per rank.
            H: hidden size.
            K: routed top-k per token.
            E: total routed experts in the EP group; must be divisible by
                ``num_ep_ranks``.
            num_ep_ranks: EP group size (R); must equal the group world size.
            num_sms: SM count for the comm kernels; None defaults to 32.
            token_padding: each non-empty VM group is padded up to a multiple
                of this token count.
            B: weight prefetch slots per rank; None defaults to
                ``E // num_ep_ranks``.
            group: torch.distributed process group; None uses the default
                group. Must be called after ``init_process_group``.
            comm_stream_priority: priority of the async comm stream; the
                default -1 gives it higher priority than the main stream.
            enable_pdl: launch kernels with programmatic dependent launch;
                False falls back to plain same-stream serial launches.
            explicitly_destroy: if True, warn (instead of auto-destroying)
                when the Buffer is garbage-collected without ``destroy()``.
        """
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
        self._comm_stream: torch.cuda.Stream | None = None
        self._destroyed = False
        self._ctx = _create_context(
            S, H, K, E, num_ep_ranks,
            num_sms=num_sms,
            token_padding=token_padding,
            B=B,
            group=group,
        )
        self._comm_stream = torch.cuda.Stream(
            device=int(self._ctx['device']),
            priority=self.comm_stream_priority,
        )

    @property
    def destroyed(self) -> bool:
        return self._destroyed

    @property
    def hidden_nvsh_buffer_view(self) -> torch.Tensor:
        """The local rank's [NvS, H] bf16 communication buffer.

        Exposed for zero-copy integration: callers may write expert outputs
        into this view and hand it back to ``combine(zero_copy=True)``. The
        view aliases persistent comm state that every dispatch/combine on
        this Buffer overwrites — never let it (or any tensor sharing its
        storage) cross into autograd-saved state.
        """
        return self._require_ctx()['hidden_buf_local']

    @property
    def router_weight_buffer_view(self) -> torch.Tensor:
        """fp32 view of the local rank's [NvS] route-weights comm buffer.

        Same aliasing/lifetime rules as ``hidden_nvsh_buffer_view``.
        """
        return self._require_ctx()['weights_buf_local'].view(torch.float32)

    def _require_ctx(self) -> dict:
        assert not self._destroyed, "MoonEP Buffer has been destroyed"
        assert self._ctx is not None, "MoonEP Buffer is not initialized"
        return self._ctx

    def destroy(self) -> None:
        """Synchronize and release all VMM/NVLink resources held by this Buffer.

        Must be called before tearing down the process group. Idempotent.
        """
        if self._destroyed:
            return
        ctx = self._ctx
        if ctx is None:
            self._destroyed = True
            return

        if self._comm_stream is not None:
            self._comm_stream.synchronize()
        torch.cuda.synchronize()
        group = ctx.get('group')
        if dist.is_initialized():
            dist.barrier(group=group)

        # Drop views before owners. Tensor deleters release VMM mappings and
        # multicast handles once the Python references disappear.
        for key in (
            'hidden_buf_local',
            'weights_buf_local',
            'meta_mc',
            'meta_buf',
            'hidden_buf',
            'alloc',
            'group_tokens',
            'z',
            'local_hist',
            'grid_sync_bar',
            'primary_packed',
            'kmask',
            'kidx_to_loff',
            'builder_bar',
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
                "MoonEP Buffer was not destroyed explicitly; resources may leak.",
                ResourceWarning,
            )
            return
        try:
            self.destroy()
        except Exception as exc:
            warnings.warn(
                f"MoonEP Buffer destructor failed to destroy resources: {exc!r}",
                ResourceWarning,
            )

    @staticmethod
    def _record_streams(tensors, stream: torch.cuda.Stream) -> None:
        for t in tensors:
            if t is not None:
                t.record_stream(stream)

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
        route_weights_zero_copy: bool,
    ) -> None:
        if inter_rank_sync:
            launch_inter_rank_sync(ctx)

        if planning_args is not None:
            topk_flat, tokens_per_expert, cu_seqlens = planning_args
            launch_planning(ctx, topk_flat, tokens_per_expert, cu_seqlens, plan)

        # Fresh planning publishes dst/src_info immediately before dispatch, so
        # dispatch can safely materialize the plan-owned dedup structures from
        # those temporaries. Plan-reuse paths must keep the saved structures
        # untouched. Dispatch also zero-fills the segment-padding rows of the
        # local shard (zero warp, plan.zero_fill_ranges).
        launch_dispatch(
            ctx,
            hidden_sh,
            route_weights_sk,
            plan,
            build_dedup_map=planning_args is not None,
            pdl_trigger=self.enable_pdl,
        )
        # In-place duplicate expansion on the NVL shard: after this the shard
        # holds the full user-visible [NvS, H] layout.
        launch_dispatch_epilogue(ctx, plan, pdl_launch=self.enable_pdl)
        # master-style boundary copies (same stream, plain SM copies), gated
        # independently per output tensor.
        if not zero_copy:
            hidden_nvsh.copy_(ctx['hidden_buf_local'])
        if route_weights_nvs is not None and not route_weights_zero_copy:
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
        router_weights_zero_copy: bool,
    ) -> None:
        # Pre-staging sync keeps its master-era position; combine's own entry
        # cross_rank_barrier publishes the staged + accumulated NVL writes.
        if inter_rank_sync:
            launch_inter_rank_sync(ctx)
        # master-style boundary copies into the shard (same stream), gated
        # independently per input tensor.
        if not zero_copy:
            ctx['hidden_buf_local'].copy_(hidden_nvsh)
        if route_weights_nvs is not None and not router_weights_zero_copy:
            ctx['weights_buf_local'].copy_(
                route_weights_nvs.view(torch.int32)
            )
        # In-place fp32 accumulation of duplicate rows into their primary.
        launch_combine_prologue(ctx, plan, pdl_trigger=self.enable_pdl)
        launch_combine(
            ctx,
            hidden_sh,
            plan.dst,
            output_sk=route_weights_sk,
            pdl_launch=self.enable_pdl,
        )

    def _run_reduce_grad_on_current_stream(
        self,
        ctx: dict,
        experts_to_copy: torch.Tensor,
        grad_reduce_args,
    ) -> None:
        _launch_full_grad_reduces(ctx, experts_to_copy, *grad_reduce_args)

    def _run_prefetch_weight_on_current_stream(
        self,
        ctx: dict,
        experts_to_copy: torch.Tensor,
        weight_prefetch_args,
    ) -> None:
        _launch_full_weight_prefetches(
            ctx,
            *weight_prefetch_args,
            experts_to_copy[int(ctx['rank'])],
        )

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

        Args:
            hidden_sh: [S, H] bf16 input tokens.
            route_weights_sk: [S, K] fp32 routing weights; None skips the
                weights buffer entirely.
            topk_experts_sk: [S, K] int32 expert ids; required when ``plan``
                is None.
            tokens_per_expert: [E] int32 local token count per expert;
                required when ``plan`` is None.
            plan: saved MoonEPCommPlan to reuse — planning is skipped and
                ``topk_experts_sk`` / ``tokens_per_expert`` are ignored. This
                is the combine bwd path: re-dispatching ``grad_output_sh``
                with the fwd plan scatters the output grad back to VM group
                order.
            async_finish: run on the comm stream and append a CUDA event to
                the return value.
            inter_rank_sync: run a CuTe DSL rank sync before planning
                (default True).
            zero_copy: return a view of the communication buffer
                (``hidden_buf_local``) instead of a fresh tensor. The view
                aliases state that the next dispatch/combine on this Buffer
                overwrites, so callers must not keep it across communication
                calls (in particular autograd must not save it for backward
                — that is exactly the case that requires ``zero_copy=False``).
                Row content is only defined within the ``cu_seqlens``-covered
                padded segments.
            router_weights_zero_copy: like ``zero_copy`` but for
                ``route_weights_nvs`` (the fp32 view of ``weights_buf_local``).
                Defaults to False even when ``zero_copy=True``: the weights
                view is the dangerous special case (tiny tensor, commonly
                saved into autograd state by training frameworks), so callers
                that only consume it before the next dispatch — e.g.
                inference — opt in explicitly.

        Returns:
            ``(hidden_nvsh, route_weights_nvs, cu_seqlens, plan)``, plus a
            comm-stream CUDA event when ``async_finish=True``:

            - hidden_nvsh: [NvS, H] bf16 dispatched tokens in physical VM
              group order.
            - route_weights_nvs: [NvS] fp32, or None when ``route_weights_sk``
              is None.
            - cu_seqlens: [E+B] int32 padded token end offset per VM group
              row; None on the plan-reuse path.
            - plan: MoonEPCommPlan; save it for prefetch/combine and both
              backward passes.
        """
        ctx = self._require_ctx()

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
                ctx['NvS'], dtype=torch.float32, device=ctx['meta_buf'].device
            )

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
                route_weights_zero_copy=router_weights_zero_copy,
            )
            return hidden_nvsh, route_weights_nvs, cu_seqlens, plan

        main_stream = torch.cuda.current_stream()
        comm = self._comm_stream
        assert comm is not None, "MoonEP Buffer communication stream is not initialized"

        tensors_to_record = [
            hidden_sh,
            route_weights_sk,
            hidden_nvsh,
            route_weights_nvs,
            *self._plan_runtime_tensors(plan),
        ]
        if planning_args is not None:
            tensors_to_record.extend(planning_args)
        self._record_streams(tensors_to_record, comm)

        input_ready = main_stream.record_event()
        comm.wait_event(input_ready)

        with torch.cuda.stream(comm):
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
                route_weights_zero_copy=router_weights_zero_copy,
            )
            done = comm.record_event()

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
        local prefetch slots (dispatch fwd, weight side).

        Args:
            plan: MoonEPCommPlan returned by ``dispatch``.
            async_finish: run on the comm stream and return a CUDA event.
            full_gate_weight / full_up_weight / full_down_weight:
                [E+B, H, H'] bf16 contiguous weight tensors; rows [0, E) are
                source expert weights, rows [E, E+B) are the prefetch slots
                filled by this call.

        Returns:
            None in synchronous mode, or the comm-stream CUDA event when
            ``async_finish=True``.

        Kept separate from ``dispatch`` so the plan-reuse path (combine bwd)
        can skip re-prefetching. Calling this after
        ``dispatch(async_finish=True)`` preserves stream ordering, and the
        returned event also covers the queued dispatch.
        """
        ctx = self._require_ctx()

        assert isinstance(plan, MoonEPCommPlan), "Buffer.prefetch_weight: plan is required"
        weight_prefetch_args = (full_gate_weight, full_up_weight, full_down_weight)
        assert all(w is not None for w in weight_prefetch_args), \
            "prefetch_weight tensors must be provided together"
        for w in weight_prefetch_args:
            assert w.dtype == torch.bfloat16 and w.is_contiguous()
            assert w.ndim == 3 and int(w.shape[0]) == int(ctx['E']) + int(ctx['B'])

        if not async_finish:
            self._run_prefetch_weight_on_current_stream(
                ctx,
                plan.experts_to_copy,
                weight_prefetch_args,
            )
            return None

        main_stream = torch.cuda.current_stream()
        comm = self._comm_stream
        assert comm is not None, "MoonEP Buffer communication stream is not initialized"

        self._record_streams((plan.experts_to_copy, *weight_prefetch_args), comm)
        input_ready = main_stream.record_event()
        comm.wait_event(input_ready)

        with torch.cuda.stream(comm):
            self._run_prefetch_weight_on_current_stream(
                ctx,
                plan.experts_to_copy,
                weight_prefetch_args,
            )
            done = comm.record_event()

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
        """combine fwd: gather expert outputs from the NVL buffer and K-sum
        back to token-major [S, H].

        Also serves as dispatch bwd: combining ``grad_hidden_nvsh`` sums each
        token's K dispatched grad copies back to its token-major grad.

        Args:
            plan: MoonEPCommPlan returned by ``dispatch``.
            hidden_nvsh: [NvS, H] bf16 expert outputs in physical VM group
                order.
            route_weights_nvs: [NvS] fp32, optional; pass the
                dispatch-returned weights to also gather them back to
                token-major.
            async_finish: run on the comm stream and return a CUDA event.
            inter_rank_sync: run a CuTe DSL rank sync before staging
                (default True).
            zero_copy: ``hidden_nvsh`` must be exactly the view returned by a
                ``zero_copy=True`` dispatch — the caller's FFN writes its
                output in place on the shard and no boundary copy is performed
                (asserted via ``data_ptr()``). With ``zero_copy=False`` the
                input is an ordinary tensor that is first copied into the
                shard.
            router_weights_zero_copy: same contract for ``route_weights_nvs``
                (the fp32 view of ``weights_buf_local``); default False, i.e.
                an ordinary tensor that is copied into the shard first.

        Returns:
            ``(hidden_sh, route_weights_sk, event)``:

            - hidden_sh: [S, H] bf16 combined token-major output, allocated
              by MoonEP.
            - route_weights_sk: [S, K] fp32 gathered routing weights, or None
              when ``route_weights_nvs`` is None.
            - event: comm-stream CUDA event when ``async_finish=True``, else
              None.
        """
        ctx = self._require_ctx()

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
                "combine(zero_copy=True): hidden_nvsh must alias the NVL shard "
                "view returned by dispatch(zero_copy=True)"
            )
        if router_weights_zero_copy and route_weights_nvs is not None:
            assert route_weights_nvs.data_ptr() == \
                ctx['weights_buf_local'].data_ptr(), (
                "combine(router_weights_zero_copy=True): route_weights_nvs must "
                "alias the NVL weights view returned by "
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

        main_stream = torch.cuda.current_stream()
        comm = self._comm_stream
        assert comm is not None, "MoonEP Buffer communication stream is not initialized"

        self._record_streams(
            (
                hidden_sh,
                *self._plan_runtime_tensors(plan),
                hidden_nvsh,
                route_weights_nvs,
                route_weights_sk,
            ),
            comm,
        )
        input_ready = main_stream.record_event()
        comm.wait_event(input_ready)

        with torch.cuda.stream(comm):
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
            done = comm.record_event()

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

        Each rank remote-reads the prefetch-slot grads of its own experts
        from every rank's reduce buffer, accumulates them into its local
        grad rows, then zeros its own consumed slots for the next
        microbatch.

        Args:
            plan: MoonEPCommPlan returned by ``dispatch``.
            async_finish: run on the comm stream and return a CUDA event.
            full_gate_grad / full_up_grad / full_down_grad:
                [E+B, H, H'] fp32 contiguous grad tensors, same layout as the
                weights; rows [E, E+B) are backed by the reduce buffer.
            gate_reduce_buffer / up_reduce_buffer / down_reduce_buffer:
                [R, B, H, H'] fp32 — all R ranks' reduce buffers mapped as
                one view.

        Returns:
            None in synchronous mode, or the comm-stream CUDA event when
            ``async_finish=True``.

        Kept separate from ``combine``; running on the shared comm stream
        keeps the Buffer's barrier/meta resources serialized.
        """
        ctx = self._require_ctx()

        grad_reduce_args = (
            full_gate_grad,
            full_up_grad,
            full_down_grad,
            gate_reduce_buffer,
            up_reduce_buffer,
            down_reduce_buffer,
        )
        assert all(t is not None for t in grad_reduce_args), \
            "reduce_grad tensors must be provided together"
        assert isinstance(plan, MoonEPCommPlan), "Buffer.reduce_grad: plan is required"

        if not async_finish:
            self._run_reduce_grad_on_current_stream(
                ctx,
                plan.experts_to_copy,
                grad_reduce_args,
            )
            return None

        main_stream = torch.cuda.current_stream()
        comm = self._comm_stream
        assert comm is not None, "MoonEP Buffer communication stream is not initialized"

        self._record_streams((plan.experts_to_copy, *grad_reduce_args), comm)
        input_ready = main_stream.record_event()
        comm.wait_event(input_ready)

        with torch.cuda.stream(comm):
            self._run_reduce_grad_on_current_stream(
                ctx,
                plan.experts_to_copy,
                grad_reduce_args,
            )
            done = comm.record_event()

        return done
