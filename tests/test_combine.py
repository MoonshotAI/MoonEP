"""Combine kernel correctness tests.

Run with:
    torchrun --nproc_per_node=8 -m pytest -s tests/test_combine.py
"""

import pytest
import torch

from tests.kernel_test_utils import (
    KernelCase,
    assert_close_all_ranks,
    assert_tensor_equal_all_ranks,
    assert_ulp_all_ranks,
    case_params,
    gather_tensor,
    init_case,
    make_topk,
)


COMBINE_CASES = [
    KernelCase("balanced", S=256, K=8, epn=16, H=128, num_sms=32),
    KernelCase("k1_direct_gather", S=32, K=1, epn=4, H=128, num_sms=8),
    KernelCase(
        "all_remote",
        S=64,
        K=4,
        epn=8,
        H=128,
        num_sms=8,
        routing="all_remote",
        min_R=2,
    ),
    KernelCase(
        "duplicate_topk",
        S=32,
        K=4,
        epn=4,
        H=128,
        num_sms=8,
        routing="duplicate_topk",
        min_R=2,
    ),
    KernelCase(
        "duplicate_topk_cross_stage",
        S=64,
        K=3,
        epn=4,
        H=128,
        num_sms=8,
        routing="duplicate_topk",
        min_R=2,
    ),
]

GLOBAL_REF_CASES = [
    KernelCase("global_balanced", S=64, K=4, epn=8, H=128, num_sms=8, B=2),
    KernelCase(
        "global_all_remote",
        S=64,
        K=4,
        epn=8,
        H=128,
        num_sms=8,
        B=2,
        routing="all_remote",
        min_R=2,
    ),
]

LARGE_COMBINE_CASES = [
    KernelCase(
        "large_hidden_stride_identity",
        S=8192,
        K=16,
        epn=14,
        H=7168,
        num_sms=32,
        B=4,
    )
]


def _random_inputs(case, rank, R, seed=0):
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(seed + rank)
    hidden = torch.randn(case.S, case.H, dtype=torch.bfloat16, device=dev, generator=gen)
    weights = torch.rand(case.S, case.K, dtype=torch.float32, device=dev, generator=gen)
    topk, tpe = make_topk(case, rank, R)
    return hidden, weights, topk, tpe


def _dispatch_inputs(ctx, case, rank, R, seed=0):
    from moonep.dispatch import launch_dispatch
    from moonep.dispatch_epilogue import launch_dispatch_epilogue
    from moonep.planning import allocate_planning_outputs, launch_planning

    hidden, weights, topk, tpe = _random_inputs(case, rank, R, seed)
    plan, cu_seqlens = allocate_planning_outputs(ctx)
    launch_planning(ctx, topk.reshape(-1).contiguous(), tpe, cu_seqlens, plan)
    dst = plan.dst
    experts_to_copy = plan.experts_to_copy
    expert_ids = _expert_ids_from_experts_to_copy(ctx, cu_seqlens, experts_to_copy[rank])
    launch_dispatch(ctx, hidden, weights, plan, build_dedup_map=True)
    launch_dispatch_epilogue(ctx, plan)
    hidden_user = torch.empty_like(ctx["hidden_buf_local"])
    hidden_user.copy_(ctx["hidden_buf_local"])
    weights_user = torch.empty(
        (int(ctx["NvS"]),), dtype=torch.float32, device=hidden_user.device
    )
    weights_user.copy_(ctx["weights_buf_local"].view(torch.float32))
    return hidden, weights, dst, cu_seqlens, expert_ids, plan, hidden_user, weights_user


def _combine_full(ctx, case, plan, hidden_user, weights_user=None):
    from moonep.combine import launch_combine
    from moonep.combine_prologue import launch_combine_prologue
    from moonep.inter_rank_sync import launch_inter_rank_sync

    output = torch.empty(
        case.S, case.H, dtype=torch.bfloat16, device=hidden_user.device
    )
    launch_inter_rank_sync(ctx)
    # zero_copy=False boundary: stage the combine inputs into the NVL shard,
    # then accumulate duplicate rows into their primary in place.
    ctx["hidden_buf_local"].copy_(hidden_user)
    if weights_user is not None:
        ctx["weights_buf_local"].copy_(weights_user.view(torch.int32))
    launch_combine_prologue(ctx, plan)
    if weights_user is None:
        launch_combine(ctx, output, plan.dst)
        return output, None

    output_sk = torch.empty(
        case.S, case.K, dtype=torch.float32, device=hidden_user.device
    )
    launch_combine(ctx, output, plan.dst, output_sk=output_sk)
    return output, output_sk


def _expert_ids_from_experts_to_copy(ctx, cu_seqlens, experts_to_copy_row):
    E = int(ctx["E"])
    B = int(ctx["B"])
    expert_ids = torch.full((E + B,), -1, dtype=torch.int32, device=cu_seqlens.device)
    prev = 0
    for group_id in range(E + B):
        cur = int(cu_seqlens[group_id].item())
        if cur > prev:
            if group_id < E:
                expert_ids[group_id] = group_id
            else:
                expert_ids[group_id] = experts_to_copy_row[group_id - E]
        prev = cur
    return expert_ids


def _uniform_scale(buf, cu_seqlens, expert_ids):
    total = int(cu_seqlens[-1].item())
    buf[:total].mul_(3.0)


def _per_expert_scale(buf, cu_seqlens, expert_ids):
    prev = 0
    for group_id in range(cu_seqlens.numel()):
        cur = int(cu_seqlens[group_id].item())
        if cur > prev:
            expert_id = int(expert_ids[group_id].item())
            if expert_id >= 0:
                buf[prev:cur].mul_(float(expert_id + 1))
        prev = cur


def _combine_global_reference(ctx, case, rank, R, hidden, dst, cu_seqlens,
                              experts_to_copy, expert_fn):
    nvs_stride = int(ctx["NvS"])
    NvS_padded = int(ctx["NvS_padded"])
    all_hidden = gather_tensor(hidden.contiguous(), R)
    all_dst = gather_tensor(dst.reshape(case.S, case.K).contiguous(), R)
    all_cu = gather_tensor(cu_seqlens.contiguous(), R)

    global_buf = torch.zeros(R, NvS_padded, case.H, dtype=torch.bfloat16,
                             device="cuda")
    for src_r in range(R):
        for s in range(case.S):
            for k in range(case.K):
                dst_val = int(all_dst[src_r, s, k].item())
                raw_dst = -dst_val - 1 if dst_val < 0 else dst_val
                dest_rank = raw_dst // nvs_stride
                local_off = raw_dst % nvs_stride
                global_buf[dest_rank, local_off] = all_hidden[src_r, s]

    for dest_r in range(R):
        expert_ids = _expert_ids_from_experts_to_copy(
            ctx, all_cu[dest_r], experts_to_copy[dest_r]
        )
        expert_fn(global_buf[dest_r], all_cu[dest_r], expert_ids)

    ref = torch.zeros(case.S, case.H, dtype=torch.float32, device="cuda")
    local_dst = all_dst[rank]
    for s in range(case.S):
        for k in range(case.K):
            dst_val = int(local_dst[s, k].item())
            raw_dst = -dst_val - 1 if dst_val < 0 else dst_val
            dest_rank = raw_dst // nvs_stride
            local_off = raw_dst % nvs_stride
            ref[s] += global_buf[dest_rank, local_off].float()
    return ref.to(torch.bfloat16)


@pytest.mark.parametrize("case", case_params(COMBINE_CASES))
def test_combine_identity_round_trip(dist_env, case):
    rank, R = dist_env
    ctx = init_case(case, R)
    hidden, _weights, _dst, _cu, _expert_ids, plan, hidden_user, _weights_user = _dispatch_inputs(
        ctx, case, rank, R
    )

    output, _ = _combine_full(ctx, case, plan, hidden_user)
    ref = (hidden.float() * case.K).to(torch.bfloat16)
    assert_close_all_ranks(f"{case.name} identity combine", output, ref, rank, R)


@pytest.mark.parametrize("case", case_params(GLOBAL_REF_CASES))
@pytest.mark.parametrize(
    "expert_name,expert_fn,use_ulp",
    [
        ("uniform_scale", _uniform_scale, False),
        ("per_expert_scale", _per_expert_scale, True),
    ],
)
def test_combine_matches_global_reference(dist_env, case, expert_name, expert_fn, use_ulp):
    rank, R = dist_env
    ctx = init_case(case, R)
    hidden, _weights, dst, cu_seqlens, expert_ids, plan, hidden_user, _weights_user = _dispatch_inputs(
        ctx, case, rank, R
    )

    expert_fn(hidden_user, cu_seqlens, expert_ids)
    torch.cuda.synchronize()
    ref = _combine_global_reference(
        ctx, case, rank, R, hidden, dst, cu_seqlens, plan.experts_to_copy, expert_fn
    )

    output, _ = _combine_full(ctx, case, plan, hidden_user)
    if use_ulp:
        assert_ulp_all_ranks(
            f"{case.name} {expert_name} combine", output, ref, rank, R, max_ulps=1
        )
    else:
        assert_close_all_ranks(f"{case.name} {expert_name} combine", output, ref, rank, R)


def test_combine_output_sk_gathers_route_weights(dist_env):
    rank, R = dist_env
    case = KernelCase("output_sk", S=128, K=4, epn=8, H=128, num_sms=8, B=2)
    ctx = init_case(case, R)
    hidden, weights, _dst, _cu, _expert_ids, plan, hidden_user, _weights_user = _dispatch_inputs(
        ctx, case, rank, R, seed=100
    )

    out_ref, _ = _combine_full(ctx, case, plan, hidden_user)

    # Re-dispatch because combine reads the mutable NVL buffer.
    _hidden, _weights, _dst, _cu, _expert_ids, plan, hidden_user, weights_user = _dispatch_inputs(
        ctx, case, rank, R, seed=100
    )
    out, out_weights = _combine_full(ctx, case, plan, hidden_user, weights_user)

    assert_tensor_equal_all_ranks("output_sk hidden", out, out_ref, rank, R)
    assert_tensor_equal_all_ranks("output_sk weights", out_weights, weights, rank, R)
    assert_tensor_equal_all_ranks("output_sk hidden input", _hidden, hidden, rank, R)
    assert_tensor_equal_all_ranks("output_sk weight input", _weights, weights, rank, R)


def test_buffer_combine_stages_external_buffers_and_gathers_weights(dist_env):
    rank, R = dist_env
    case = KernelCase("public_staging", S=128, K=4, epn=8, H=128, num_sms=8, B=2)
    ctx = init_case(case, R)
    buffer = ctx["_buffer"]
    _hidden, weights, _dst, _cu, _expert_ids, plan, hidden_user, weights_user = _dispatch_inputs(
        ctx, case, rank, R, seed=200
    )

    ref_out, ref_weights = _combine_full(ctx, case, plan, hidden_user, weights_user)

    staged_hidden = hidden_user.clone()
    staged_weights = weights_user.clone()
    ctx["hidden_buf_local"].zero_()
    ctx["weights_buf_local"].zero_()

    out, out_weights, _ = buffer.combine(
        plan=plan,
        hidden_nvsh=staged_hidden,
        route_weights_nvs=staged_weights,
    )

    assert_tensor_equal_all_ranks("public staging hidden", out, ref_out, rank, R)
    assert_tensor_equal_all_ranks("public staging weights", out_weights, weights, rank, R)
    assert_tensor_equal_all_ranks("public staging ref weights", ref_weights, weights, rank, R)


def test_buffer_combine_async_gathers_weights(dist_env):
    rank, R = dist_env
    case = KernelCase("async_weights", S=128, K=4, epn=8, H=128, num_sms=8, B=2)
    ctx = init_case(case, R)
    buffer = ctx["_buffer"]
    _hidden, weights, _dst, _cu, _expert_ids, plan, hidden_user, weights_user = _dispatch_inputs(
        ctx, case, rank, R, seed=300
    )

    ref_out, ref_weights, _ = buffer.combine(
        plan=plan,
        hidden_nvsh=hidden_user,
        route_weights_nvs=weights_user,
    )

    _hidden, _weights, _dst, _cu, _expert_ids, plan, hidden_user, weights_user = _dispatch_inputs(
        ctx, case, rank, R, seed=300
    )
    out, out_weights, event = buffer.combine(
        plan=plan,
        hidden_nvsh=hidden_user,
        route_weights_nvs=weights_user,
        async_finish=True,
    )
    event.wait(torch.cuda.current_stream())
    torch.cuda.synchronize()

    assert_tensor_equal_all_ranks("async combine hidden", out, ref_out, rank, R)
    assert_tensor_equal_all_ranks("async combine weights", out_weights, weights, rank, R)


@pytest.mark.parametrize("case", case_params(LARGE_COMBINE_CASES))
def test_combine_large_hidden_stride_identity(dist_env, case):
    rank, R = dist_env
    ctx = init_case(case, R)
    hidden, _weights, _dst, _cu, _expert_ids, plan, hidden_user, _weights_user = _dispatch_inputs(
        ctx, case, rank, R
    )

    output, _ = _combine_full(ctx, case, plan, hidden_user)
    ref = (hidden.float() * case.K).to(torch.bfloat16)
    assert_close_all_ranks(f"{case.name} combine", output, ref, rank, R)


def test_combine_rejects_bad_inputs(dist_env):
    from moonep.combine import launch_combine

    rank, R = dist_env
    case = KernelCase("bad_inputs", S=4, K=2, epn=2, H=128, num_sms=1)
    ctx = init_case(case, R)
    buffer = ctx["_buffer"]
    _hidden, _weights, dst, _cu, _expert_ids, plan, hidden_user, _weights_user = _dispatch_inputs(
        ctx, case, rank, R
    )
    output = torch.empty(case.S, case.H, dtype=torch.bfloat16, device="cuda")

    with pytest.raises(TypeError, match="hidden_sh"):
        buffer.combine(hidden_sh=output, plan=plan, hidden_nvsh=hidden_user)
    with pytest.raises(AssertionError, match="plan is required"):
        buffer.combine(hidden_nvsh=hidden_user)
    with pytest.raises(AssertionError):
        buffer.combine(plan=plan)
    with pytest.raises(AssertionError):
        buffer.combine(
            plan=plan,
            hidden_nvsh=hidden_user,
            route_weights_nvs=torch.empty(
                case.S, case.K, dtype=torch.float32, device="cuda"
            ),
        )
    with pytest.raises(AssertionError, match="output_sk"):
        bad_output_sk = torch.empty(case.S, case.K, dtype=torch.bfloat16,
                                    device="cuda")
        launch_combine(ctx, output, dst, output_sk=bad_output_sk)
