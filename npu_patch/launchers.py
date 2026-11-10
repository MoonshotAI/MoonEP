"""
``launch_*`` compatibility shims for the Ascend port.

The moonep stage tests call module-level launchers directly
(``launch_planning``, ``launch_dispatch``, ``launch_combine``,
``launch_inter_rank_sync``) rather than going through ``Buffer``. These shims
keep the ported Ascend tests minimal: they resolve the op object from
``ctx['kernels']`` (created at Buffer init) and call its ``compute``. The former
``launch_dispatch_epilogue`` / ``launch_combine_prologue`` steps are folded into
``launch_dispatch`` / ``launch_combine`` and no longer have their own shims.

Two launchers are intentionally *not* shimmed through a Buffer ctx because the
standalone stage tests call them without one:
  - ``launch_prefetch(remote_expert, prefetch_buffers, experts_to_copy, num_sms)``
  - ``launch_grad_reduce(remote_expert_grads, reduce_buffers, plan, rank, ...)``
Their Ascend contracts are described in ``ops.py``.
"""

import torch
import torch_npu
import os

from ._common import cross_rank_barrier

_PLANNING = None
_DISPATCH = None
_COMBINE = None
_COMBINE_UDMA = None
_GRAD_REDUCE = None
_PREFETCH_WEIGHT = None


def _env_use_udma(name: str | None = None) -> bool:
    key = 'ASCEND_MOONEP_USE_UDMA' if name is None else f'ASCEND_MOONEP_{name.upper()}_USE_UDMA'
    return os.environ.get(key, '0').strip().lower() in ('1', 'true', 'yes', 'on')


def planning_workspace_sizes(
    num_tokens: int,
    num_topk: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    world_size: int,
    core_num: int,
) -> tuple[int, int]:
    global _PLANNING
    if _PLANNING is None:
        _PLANNING = torch.classes.AscendMoonEP.Planning()
    local_elements = int(
        _PLANNING.local_workspace_elements(
            num_tokens,
            num_topk,
            num_experts,
            world_size,
            core_num,
        )
    )
    symmetric_bytes = int(
        _PLANNING.symmetric_workspace_bytes(
            num_experts,
            num_max_tokens_per_rank,
            world_size,
        )
    )
    return local_elements, symmetric_bytes


def launch_planning_op(
    topk_idx: torch.Tensor,
    dst: torch.Tensor,
    cu_seqlens: torch.Tensor,
    experts_to_copy: torch.Tensor,
    dup_groups: torch.Tensor,
    dup_loffs: torch.Tensor,
    dup_counts: torch.Tensor,
    local_workspace: torch.Tensor,
    symmetric_workspace_addr: int,
    symmetric_workspace_bytes: int,
    num_tokens: int,
    num_topk: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_prefetch_slots: int,
    core_num: int,
    rank: int,
    world_size: int,
) -> None:
    global _PLANNING
    if _PLANNING is None:
        _PLANNING = torch.classes.AscendMoonEP.Planning()
    _PLANNING.compute(
        topk_idx,
        dst,
        cu_seqlens,
        experts_to_copy,
        dup_groups,
        dup_loffs,
        dup_counts,
        local_workspace,
        symmetric_workspace_addr,
        symmetric_workspace_bytes,
        num_tokens,
        num_topk,
        num_experts,
        num_max_tokens_per_rank,
        num_prefetch_slots,
        core_num,
        rank,
        world_size,
        _env_use_udma('planning'),
    )


def launch_dispatch_op(
    hidden_sh: torch.Tensor,
    route_weights: torch.Tensor | None,
    dst: torch.Tensor,
    dup_groups: torch.Tensor,
    dup_loffs: torch.Tensor,
    dup_counts: torch.Tensor,
    hidden_buf: torch.Tensor,
    route_weights_buf: torch.Tensor | None,
    sync_buffer: torch.Tensor,
    num_tokens: int,
    hidden_size: int,
    num_topk: int,
    world_size: int,
    rank: int,
) -> None:
    global _DISPATCH
    if _DISPATCH is None:
        _DISPATCH = torch.classes.AscendMoonEP.Dispatch()
    _DISPATCH.compute(
        hidden_sh,
        route_weights,
        dst,
        dup_groups,
        dup_loffs,
        dup_counts,
        hidden_buf,
        route_weights_buf,
        sync_buffer,
        num_tokens,
        hidden_size,
        num_topk,
        world_size,
        rank,
        _env_use_udma('dispatch'),
    )


def launch_dispatch_profile_op(
    hidden_sh: torch.Tensor,
    route_weights: torch.Tensor | None,
    dst: torch.Tensor,
    dup_groups: torch.Tensor,
    dup_loffs: torch.Tensor,
    dup_counts: torch.Tensor,
    hidden_buf: torch.Tensor,
    route_weights_buf: torch.Tensor | None,
    sync_buffer: torch.Tensor,
    profile_cycles: torch.Tensor,
    num_tokens: int,
    hidden_size: int,
    num_topk: int,
    world_size: int,
    rank: int,
) -> None:
    global _DISPATCH
    if _DISPATCH is None:
        _DISPATCH = torch.classes.AscendMoonEP.Dispatch()
    _DISPATCH.compute_profile(
        hidden_sh,
        route_weights,
        dst,
        dup_groups,
        dup_loffs,
        dup_counts,
        hidden_buf,
        route_weights_buf,
        sync_buffer,
        profile_cycles,
        num_tokens,
        hidden_size,
        num_topk,
        world_size,
        rank,
        _env_use_udma('dispatch'),
    )


def launch_combine_op(
    hidden_buf: torch.Tensor,
    weight_buf: torch.Tensor | None,
    sync_buffer: torch.Tensor,
    dst: torch.Tensor,
    dup_groups: torch.Tensor,
    dup_loffs: torch.Tensor,
    dup_counts: torch.Tensor,
    hidden_sk_out: torch.Tensor,
    route_weights_sk_out: torch.Tensor | None,
    num_tokens: int,
    hidden_size: int,
    num_topk: int,
    world_size: int,
    rank: int,
) -> None:
    """Select native/UDMA combine from the process environment."""
    global _COMBINE, _COMBINE_UDMA
    if _env_use_udma("combine"):
        if _COMBINE_UDMA is None:
            _COMBINE_UDMA = torch.classes.AscendMoonEP.CombineUrma()
        _COMBINE_UDMA.compute(
            hidden_buf,
            weight_buf,
            sync_buffer,
            dst,
            dup_groups,
            dup_loffs,
            dup_counts,
            hidden_sk_out,
            route_weights_sk_out,
            num_tokens,
            hidden_size,
            num_topk,
            world_size,
            rank,
        )
        return

    sk_stride = ((num_topk + 7) // 8) * 8
    route_weights_sk_launch = route_weights_sk_out
    if route_weights_sk_out is not None and sk_stride != num_topk:
        route_weights_sk_launch = torch.empty(
            (num_tokens, sk_stride),
            dtype=torch.float32,
            device=route_weights_sk_out.device,
        )

    if _COMBINE is None:
        _COMBINE = torch.classes.AscendMoonEP.Combine()
    _COMBINE.compute(
        hidden_buf,
        weight_buf,
        dst,
        dup_groups,
        dup_loffs,
        dup_counts,
        hidden_sk_out,
        route_weights_sk_launch,
        num_tokens,
        hidden_size,
        num_topk,
        world_size,
        rank,
        False,
    )

    if route_weights_sk_out is not None and sk_stride != num_topk:
        route_weights_sk_out.copy_(route_weights_sk_launch[:, :num_topk])


def launch_planning(ctx: dict, topk_flat, tokens_per_expert, cu_seqlens, plan) -> None:
    """Mirror ``moonep.launch_planning``; returns None (the tests assert this)."""
    ctx['kernels']['planning'].compute(
        ctx, topk_flat, tokens_per_expert, cu_seqlens, plan,
        shmem_instance_id=0)
    return None


def launch_dispatch(
    ctx: dict,
    hidden_sh,
    route_weights_sk,
    plan,
    *,
    build_dedup_map: bool = True,
    pdl_trigger: bool = False,
) -> None:
    """Mirror ``moonep.launch_dispatch``. ``pdl_trigger`` is CUDA-only and ignored."""
    ctx['kernels']['dispatch'].compute(
        ctx, hidden_sh, route_weights_sk, plan,
        build_dedup_map=build_dedup_map, shmem_instance_id=0)
    return None


def launch_dispatch_profile(
    ctx: dict,
    hidden_sh,
    route_weights_sk,
    plan,
    profile_cycles,
    build_dedup_map: bool = True,
    pdl_trigger: bool = False,
) -> None:
    """Mirror ``moonep.launch_dispatch_profile``. ``pdl_trigger`` is CUDA-only and ignored."""
    ctx['kernels']['dispatch'].compute_profile(
        ctx, hidden_sh, route_weights_sk, plan, profile_cycles,
        build_dedup_map=build_dedup_map, shmem_instance_id=0)
    return None


def launch_combine(
    ctx: dict,
    output_sh,
    plan,
    output_sk=None,
    *,
    pdl_launch: bool = False,
) -> None:
    """Mirror ``moonep.launch_combine`` (Ascend: moonEP_combine kernel)."""
    ctx['kernels']['combine'].compute(
        ctx, output_sh, plan, output_sk=output_sk, shmem_instance_id=0)
    return None


def launch_inter_rank_sync(ctx: dict) -> None:
    """Ascend replacement for the CuTe DSL rank sync kernel: a host-side
    cross-rank signal barrier on the current stream."""
    from ._common import cross_rank_barrier

    cross_rank_barrier(ctx, torch_npu.npu.current_stream().npu_stream)
    return None


def launch_prefetch(
    remote_expert: torch.Tensor,
    prefetch_buffers: torch.Tensor,
    experts_to_copy: torch.Tensor,
    num_sms: int,
) -> torch.Tensor:
    """Prefetch selected expert weights using the registered custom class."""
    assert remote_expert.dtype == torch.bfloat16
    assert remote_expert.ndim == 3
    assert prefetch_buffers.dtype == torch.bfloat16
    assert prefetch_buffers.is_contiguous() and prefetch_buffers.ndim == 3
    assert experts_to_copy.dtype == torch.int32
    assert experts_to_copy.is_contiguous() and experts_to_copy.ndim == 1
    assert remote_expert.device == prefetch_buffers.device
    assert remote_expert.device == experts_to_copy.device
    num_experts = int(remote_expert.shape[0])
    assert prefetch_buffers.shape[0] == experts_to_copy.numel()
    assert remote_expert.shape[1:] == prefetch_buffers.shape[1:]
    assert isinstance(num_sms, int) and num_sms > 0

    global _PREFETCH_WEIGHT
    if _PREFETCH_WEIGHT is None:
        _PREFETCH_WEIGHT = torch.classes.AscendMoonEP.PrefetchWeight(0)
    return _PREFETCH_WEIGHT.compute(
        remote_expert, prefetch_buffers, experts_to_copy,
        num_experts, num_sms, 0, _env_use_udma('prefetch'))


def launch_grad_reduce(
    remote_expert_grads: torch.Tensor,
    remote_reduce_buffers: torch.Tensor,
    experts_to_copy: torch.Tensor,
    rank: int,
    num_sms: int,
    meta_buf: torch.Tensor,
    meta_stride: int,
    barrier_off: int,
    grid_sync_bar: torch.Tensor,
) -> None:
    """Accumulate symmetric reduce slots into owner gradients.

    ``remote_expert_grads`` and ``remote_reduce_buffers`` must be separate
    symmetric allocations from the active CANN-SHMEM heap. Every PE owns one
    local ``remote_reduce_buffers[B, H, H']`` chunk; the ``[R, B]`` plan selects
    the source PE and its local slot. The allocations do not need to be
    adjacent in virtual address space. ``meta_buf``, ``meta_stride``,
    ``barrier_off`` and ``grid_sync_bar`` are retained only for MoonEP API
    compatibility; the Ascend implementation ignores them.
    """
    del meta_buf, meta_stride, barrier_off, grid_sync_bar

    if remote_reduce_buffers.numel() == 0 or experts_to_copy.numel() == 0:
        return

    assert remote_expert_grads.dtype == torch.float32
    assert remote_expert_grads.is_contiguous()
    assert remote_expert_grads.ndim == 3
    assert remote_reduce_buffers.dtype == torch.float32
    assert remote_reduce_buffers.is_contiguous()
    assert remote_reduce_buffers.ndim == 3
    assert experts_to_copy.dtype == torch.int32
    assert experts_to_copy.is_contiguous()
    assert experts_to_copy.ndim == 2
    assert isinstance(rank, int)
    assert isinstance(num_sms, int) and num_sms > 0

    epn, H, Hp = (int(value) for value in remote_expert_grads.shape)
    B, buf_H, buf_Hp = (int(value) for value in remote_reduce_buffers.shape)
    R, plan_B = (int(value) for value in experts_to_copy.shape)
    assert (buf_H, buf_Hp) == (H, Hp)
    assert plan_B == B
    assert 0 <= rank < R
    device = remote_expert_grads.device
    assert device.type == "npu"
    assert remote_reduce_buffers.device == device
    assert experts_to_copy.device == device

    global _GRAD_REDUCE
    if _GRAD_REDUCE is None:
        _GRAD_REDUCE = torch.classes.AscendMoonEP.GradReduce()
    _GRAD_REDUCE.compute(
        remote_expert_grads,
        remote_reduce_buffers,
        experts_to_copy,
        epn,
        H,
        Hp,
        R,
        B,
        rank,
        min(num_sms, 64),
        _env_use_udma('grad_reduce'),
    )
