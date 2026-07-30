"""Dispatch kernel correctness tests.

Run with:
    torchrun --nproc_per_node=8 -m pytest -s tests/test_dispatch.py
"""

from dataclasses import replace

import pytest
import torch

from tests.kernel_test_utils import (
    KernelCase,
    assert_all_ranks,
    assert_dedup_plan_semantic_equal_all_ranks,
    case_params,
    clone_dedup_plan_fields,
    dedup_plan_fields_equal,
    dedup_plan_invariant_errors,
    gather_tensor,
    init_case,
    make_topk,
)
from tests.planning_reference import launch_planning_torch_reference


DISPATCH_CASES = [
    KernelCase("balanced", S=256, K=8, epn=16, H=128, num_sms=32, B=4),
    KernelCase(
        "tiny_k1_no_padding",
        S=1,
        K=1,
        epn=1,
        H=8,
        num_sms=1,
        token_padding=1,
    ),
    KernelCase(
        "all_local",
        S=64,
        K=4,
        epn=8,
        H=64,
        num_sms=8,
        B=2,
        token_padding=16,
        routing="all_local",
    ),
    KernelCase(
        "all_remote",
        S=64,
        K=4,
        epn=8,
        H=64,
        num_sms=8,
        B=2,
        token_padding=16,
        routing="all_remote",
        min_R=2,
    ),
    KernelCase(
        "duplicate_topk",
        S=32,
        K=4,
        epn=4,
        H=64,
        num_sms=8,
        B=2,
        token_padding=8,
        routing="duplicate_topk",
        min_R=2,
    ),
    KernelCase(
        "duplicate_topk_k32",
        S=8,
        K=32,
        epn=4,
        H=64,
        num_sms=8,
        B=2,
        token_padding=8,
        routing="duplicate_topk",
        min_R=2,
    ),
    KernelCase(
        "single_expert_split",
        S=9,
        K=2,
        epn=4,
        H=64,
        num_sms=8,
        B=1,
        token_padding=8,
        routing="single_expert",
        min_R=2,
    ),
]

ZERO_FILL_CASES = [
    KernelCase(
        "local_padding",
        S=17,
        K=3,
        epn=8,
        H=64,
        num_sms=8,
        B=2,
        token_padding=16,
        routing="all_local",
    ),
    KernelCase(
        "remote_padding",
        S=17,
        K=3,
        epn=8,
        H=64,
        num_sms=8,
        B=3,
        token_padding=16,
        routing="all_remote",
        min_R=2,
    ),
]

LARGE_DISPATCH_CASES = [
    KernelCase(
        "large_hidden_stride_spotcheck",
        S=8192,
        K=16,
        epn=14,
        H=7168,
        num_sms=32,
        B=4,
    )
]


def _traceable_hidden(rank, S, H):
    hidden = torch.zeros(S, H, dtype=torch.bfloat16, device=f"cuda:{rank}")
    hidden_i16 = hidden.view(torch.int16)
    s_idx = torch.arange(S, dtype=torch.int32, device=f"cuda:{rank}")
    hidden_i16[:, 0] = s_idx.to(torch.int16)
    if H > 1:
        hidden_i16[:, 1] = torch.full((S,), rank, dtype=torch.int16, device=f"cuda:{rank}")
    return hidden


def _traceable_weights(rank, S, K):
    weights_i32 = (
        torch.arange(S * K, dtype=torch.int32, device=f"cuda:{rank}")
        .reshape(S, K)
        .add_(rank * S * K)
    )
    return weights_i32.view(torch.float32)


def _plan_and_dispatch(
    ctx,
    case,
    rank,
    R,
    hidden,
    weights,
    *,
    hidden_user=None,
    weights_user=None,
):
    from moonep.dispatch import launch_dispatch
    from moonep.dispatch_epilogue import launch_dispatch_epilogue
    from moonep.planning import allocate_planning_outputs, launch_planning

    topk, tpe = make_topk(case, rank, R)
    plan, _cu = allocate_planning_outputs(ctx)
    launch_planning(ctx, topk.reshape(-1).contiguous(), tpe, _cu, plan)
    _ref_dst, _ref_cu, _ref_etc, _ref_stats, _ref_zfr, ref_dedup_plan = (
        launch_planning_torch_reference(ctx, topk, tpe)
    )
    launch_dispatch(ctx, hidden, weights, plan, build_dedup_map=True)
    assert_dedup_plan_semantic_equal_all_ranks(
        "dispatch dedup plan", plan, ref_dedup_plan, rank, R
    )
    dedup_errors = dedup_plan_invariant_errors(case, ctx, plan)
    assert_all_ranks(
        not dedup_errors,
        rank,
        R,
        f"{case.name} dispatch dedup plan invariants",
        "; ".join(dedup_errors[:5]),
    )
    # In-place duplicate expansion on the NVL shard, then the master-style
    # zero_copy=False boundary copies into user tensors.
    launch_dispatch_epilogue(ctx, plan)
    if hidden_user is None:
        hidden_user = torch.empty_like(ctx["hidden_buf_local"])
    hidden_user.copy_(ctx["hidden_buf_local"])
    if weights is not None:
        if weights_user is None:
            weights_user = torch.empty(
                (int(ctx["NvS"]),),
                dtype=torch.float32,
                device=hidden_user.device,
            )
        weights_user.copy_(ctx["weights_buf_local"].view(torch.float32))
    else:
        weights_user = None
    return plan.dst, plan, hidden_user, weights_user


def _verify_dispatch_by_dst(ctx, case, rank, R, dst, hidden_user,
                            check_weights=True, weights_user=None,
                            max_checks=None, copy_to_cpu=True):
    nvs_stride = int(ctx["NvS"])
    all_dst = gather_tensor(dst.reshape(case.S, case.K).contiguous(), R).cpu()

    if copy_to_cpu:
        hidden_i16 = hidden_user.view(torch.int16).cpu()
        weights_i32 = weights_user.view(torch.int32).cpu() if weights_user is not None else None
    else:
        hidden_i16 = hidden_user.view(torch.int16)
        weights_i32 = weights_user.view(torch.int32) if weights_user is not None else None

    errors = []
    checked = 0
    for src_r in range(R):
        for s in range(case.S):
            for k in range(case.K):
                dst_val = int(all_dst[src_r, s, k].item())
                raw_dst = -dst_val - 1 if dst_val < 0 else dst_val
                dest_rank = raw_dst // nvs_stride
                local_off = raw_dst % nvs_stride
                if dest_rank != rank:
                    continue

                expected_s = s & 0xFFFF
                if expected_s >= 0x8000:
                    expected_s -= 0x10000
                actual_s = int(hidden_i16[local_off, 0].item())
                actual_r = int(hidden_i16[local_off, 1].item()) if case.H > 1 else src_r
                if actual_s != expected_s or actual_r != src_r:
                    errors.append(
                        f"hidden src=({src_r},{s},{k}) loff={local_off}: "
                        f"expected s/r={expected_s}/{src_r}, got {actual_s}/{actual_r}"
                    )

                if check_weights:
                    if weights_i32 is None:
                        errors.append("weights_user missing while check_weights=True")
                        continue
                    expected_w = src_r * case.S * case.K + s * case.K + k
                    actual_w = int(weights_i32[local_off].item())
                    if actual_w != expected_w:
                        errors.append(
                            f"weight src=({src_r},{s},{k}) loff={local_off}: "
                            f"expected {expected_w}, got {actual_w}"
                        )

                checked += 1
                if max_checks is not None and checked >= max_checks:
                    return errors, checked
    return errors, checked


def _zero_row_errors(zero_fill_ranges, hidden_user, weights_user=None):
    """Every row covered by plan.zero_fill_ranges must be zero (hidden and,
    when given, the matching weight slot). The [last_padded_end, NvS) tail is
    deliberately NOT covered (master behavior: undefined, read via
    cu_seqlens)."""
    hidden_cpu = hidden_user.cpu()
    weights_cpu = weights_user.cpu() if weights_user is not None else None
    ranges = zero_fill_ranges.cpu()

    errors = []
    rows = 0
    for g in range(ranges.shape[0]):
        pad_start = int(ranges[g, 0].item())
        n_pad = int(ranges[g, 1].item())
        for j in range(n_pad):
            local_off = pad_start + j
            rows += 1
            if not torch.all(hidden_cpu[local_off] == 0):
                errors.append(f"hidden padding loff={local_off} is nonzero")
            if weights_cpu is not None and float(weights_cpu[local_off].item()) != 0.0:
                value = int(weights_cpu.view(torch.int32)[local_off].item()) & 0xFFFFFFFF
                errors.append(
                    f"weight padding loff={local_off} is 0x{value:08x}"
                )
    return errors, rows


@pytest.mark.parametrize("case", case_params(DISPATCH_CASES))
def test_dispatch_scatters_hidden_and_weights_by_dst(dist_env, case):
    rank, R = dist_env
    ctx = init_case(case, R)
    hidden = _traceable_hidden(rank, case.S, case.H)
    weights = _traceable_weights(rank, case.S, case.K)

    dst, _plan, hidden_user, weights_user = _plan_and_dispatch(
        ctx, case, rank, R, hidden, weights
    )
    torch.cuda.synchronize()

    errors, checked = _verify_dispatch_by_dst(
        ctx, case, rank, R, dst, hidden_user=hidden_user, weights_user=weights_user
    )
    assert_all_ranks(
        checked > 0 and not errors,
        rank,
        R,
        f"{case.name} dispatch scatter",
        "; ".join(errors[:5]) or f"checked={checked}",
    )


@pytest.mark.parametrize("case", case_params(ZERO_FILL_CASES))
def test_dispatch_dedup_plan_clears_padding_with_weights(dist_env, case):
    rank, R = dist_env
    ctx = init_case(case, R)
    hidden = torch.randn(case.S, case.H, dtype=torch.bfloat16, device=f"cuda:{rank}")
    weights = torch.rand(case.S, case.K, dtype=torch.float32, device=f"cuda:{rank}")

    ctx["hidden_buf_local"].fill_(7)
    ctx["weights_buf_local"].fill_(0x55555555)
    hidden_user = torch.empty(
        (int(ctx["NvS"]), case.H),
        dtype=torch.bfloat16,
        device=f"cuda:{rank}",
    )
    hidden_user.fill_(7)
    weights_user = torch.empty(
        (int(ctx["NvS"]),),
        dtype=torch.float32,
        device=f"cuda:{rank}",
    )
    weights_user.view(torch.int32).fill_(0x55555555)
    torch.cuda.synchronize()

    _dst, plan, hidden_user, weights_user = _plan_and_dispatch(
        ctx,
        case,
        rank,
        R,
        hidden,
        weights,
        hidden_user=hidden_user,
        weights_user=weights_user,
    )
    torch.cuda.synchronize()

    errors, rows = _zero_row_errors(plan.zero_fill_ranges, hidden_user, weights_user)
    assert_all_ranks(
        rows > 0 and not errors,
        rank,
        R,
        f"{case.name} dispatch padding zero-fill",
        "; ".join(errors[:5]) or f"padding_rows={rows}",
    )


def test_dispatch_saved_plan_hidden_only_reuses_dst_and_skips_weights(dist_env):
    from moonep.dispatch import launch_dispatch
    from moonep.dispatch_epilogue import launch_dispatch_epilogue
    from moonep.planning import allocate_planning_outputs, launch_planning

    rank, R = dist_env
    case = KernelCase(
        "saved_plan",
        S=17,
        K=3,
        epn=8,
        H=64,
        num_sms=8,
        B=2,
        token_padding=16,
        routing="all_remote",
        min_R=2,
    )
    ctx = init_case(case, R)

    hidden_a = _traceable_hidden(rank, case.S, case.H)
    weights_a = _traceable_weights(rank, case.S, case.K)
    topk_a, tpe_a = make_topk(case, rank, R)
    plan_a, _cu = allocate_planning_outputs(ctx)
    launch_planning(ctx, topk_a.reshape(-1).contiguous(), tpe_a, _cu, plan_a)
    dst_a = plan_a.dst
    launch_dispatch(ctx, hidden_a, weights_a, plan_a, build_dedup_map=True)
    launch_dispatch_epilogue(ctx, plan_a)
    torch.cuda.synchronize()
    dedup_a_snapshot = clone_dedup_plan_fields(plan_a)

    case_b = replace(case, routing="all_local")
    hidden_b = torch.randn(case.S, case.H, dtype=torch.bfloat16, device=f"cuda:{rank}")
    weights_b = torch.rand(case.S, case.K, dtype=torch.float32, device=f"cuda:{rank}")
    topk_b, tpe_b = make_topk(case_b, rank, R)
    plan_b, _cu = allocate_planning_outputs(ctx)
    launch_planning(ctx, topk_b.reshape(-1).contiguous(), tpe_b, _cu, plan_b)
    launch_dispatch(ctx, hidden_b, weights_b, plan_b, build_dedup_map=True)
    launch_dispatch_epilogue(ctx, plan_b)
    torch.cuda.synchronize()
    weights_after_b = ctx["weights_buf_local"].clone()

    launch_dispatch(ctx, hidden_a, None, plan_a, build_dedup_map=False)
    launch_dispatch_epilogue(ctx, plan_a)
    hidden_user = torch.empty_like(ctx["hidden_buf_local"])
    hidden_user.copy_(ctx["hidden_buf_local"])
    torch.cuda.synchronize()

    scatter_errors, checked = _verify_dispatch_by_dst(
        ctx, case, rank, R, dst_a, check_weights=False, hidden_user=hidden_user
    )
    padding_errors, padding_rows = _zero_row_errors(plan_a.zero_fill_ranges, hidden_user)
    weights_unchanged = torch.equal(ctx["weights_buf_local"], weights_after_b)
    dedup_unchanged = dedup_plan_fields_equal(plan_a, dedup_a_snapshot)
    errors = scatter_errors + padding_errors
    if not weights_unchanged:
        errors.append("weights_buf_local changed during hidden-only dispatch")
    if not dedup_unchanged:
        errors.append("saved plan dedup structures changed during reuse dispatch")

    assert_all_ranks(
        checked > 0 and padding_rows > 0 and not errors,
        rank,
        R,
        "saved-plan hidden-only dispatch",
        "; ".join(errors[:5]) or f"checked={checked}, padding_rows={padding_rows}",
    )


def test_prepared_dispatch_object_publishes_external_payload(dist_env):
    """Emulate a producer direct-out, then run MoonEP's metadata-only publish.

    The first payload-only launch is a correctness oracle for the external
    producer contract: it writes representative rows according to ``dst`` but
    deliberately skips dedup construction and route metadata. The public
    publish path must complete those pieces without recopying hidden payloads.
    """
    from moonep.dispatch import launch_dispatch

    rank, R = dist_env
    case = KernelCase(
        "prepared_object",
        S=17,
        K=3,
        epn=8,
        H=64,
        num_sms=8,
        B=2,
        token_padding=16,
        routing="all_remote",
        min_R=2,
    )
    ctx = init_case(case, R)
    buffer = ctx["_buffer"]
    topk, tpe = make_topk(case, rank, R)
    hidden = _traceable_hidden(rank, case.S, case.H)
    weights = _traceable_weights(rank, case.S, case.K)

    target, cu_seqlens, plan, planning_done = buffer.prepare_dispatch(
        topk,
        tpe,
        async_finish=True,
    )
    torch.cuda.current_stream().wait_event(planning_done)
    assert target.capability == "remote_final_slot_store_v1"
    assert target.mapped_hidden.data_ptr() == ctx["hidden_buf"].data_ptr()
    assert target.local_hidden.data_ptr() == ctx["hidden_buf_local"].data_ptr()
    assert target.plan is plan
    assert tuple(cu_seqlens.shape) == (case.E(R) + int(ctx["B"]),)

    # Emulate a compatible upstream producer. This call writes payload only;
    # publish_dispatch below owns the fresh dedup build, route metadata,
    # visibility publication, and local duplicate expansion.
    launch_dispatch(
        ctx,
        hidden,
        None,
        plan,
        write_payload=True,
        build_dedup_map=False,
    )
    hidden_view, weights_view, event = buffer.publish_dispatch(
        target,
        plan,
        route_weights_sk=weights,
        zero_copy=True,
    )
    assert event is None
    assert hidden_view.data_ptr() == target.local_hidden.data_ptr()
    assert weights_view.data_ptr() == ctx["weights_buf_local"].data_ptr()
    torch.cuda.synchronize()

    scatter_errors, checked = _verify_dispatch_by_dst(
        ctx,
        case,
        rank,
        R,
        plan.dst,
        hidden_user=hidden_view,
        weights_user=weights_view,
    )
    padding_errors, padding_rows = _zero_row_errors(
        plan.zero_fill_ranges,
        hidden_view,
        weights_view,
    )
    errors = scatter_errors + padding_errors
    assert_all_ranks(
        checked > 0 and padding_rows > 0 and not errors,
        rank,
        R,
        "prepared dispatch object publish",
        "; ".join(errors[:5])
        or f"checked={checked}, padding_rows={padding_rows}",
    )


def test_prepared_dispatch_object_overlaps_metadata_publication(dist_env):
    """Two-phase publication preserves payload, weights, and padding."""
    from moonep.dispatch import launch_dispatch

    rank, R = dist_env
    case = KernelCase(
        "prepared_overlap",
        S=17,
        K=3,
        epn=8,
        H=64,
        num_sms=8,
        B=2,
        token_padding=16,
        routing="all_remote",
        min_R=2,
    )
    ctx = init_case(case, R)
    buffer = ctx["_buffer"]
    topk, tpe = make_topk(case, rank, R)
    hidden = _traceable_hidden(rank, case.S, case.H)
    weights = _traceable_weights(rank, case.S, case.K)

    target, _cu_seqlens, plan = buffer.prepare_dispatch(topk, tpe)
    prepared = buffer.prepare_dispatch_publication(
        target,
        plan,
        route_weights_sk=weights,
    )
    with pytest.raises(RuntimeError, match="already been prepared"):
        buffer.prepare_dispatch_publication(
            target,
            plan,
            route_weights_sk=weights,
        )
    with pytest.raises(RuntimeError, match="pass the returned"):
        buffer.publish_dispatch(
            target,
            plan,
            route_weights_sk=weights,
        )

    # The generic test producer shares MoonEP's barrier scratch, unlike a real
    # direct-out epilogue, so order it after metadata here. The research CUDA
    # Tile benchmark covers true concurrent execution.
    prepared.metadata_done.synchronize()
    launch_dispatch(
        ctx,
        hidden,
        None,
        plan,
        write_payload=True,
        build_dedup_map=False,
    )
    hidden_view, weights_view, done = buffer.publish_dispatch(
        target,
        plan,
        async_finish=True,
        zero_copy=True,
        prepared_publication=prepared,
    )
    assert done is not None
    done.synchronize()

    scatter_errors, checked = _verify_dispatch_by_dst(
        ctx,
        case,
        rank,
        R,
        plan.dst,
        hidden_user=hidden_view,
        weights_user=weights_view,
    )
    padding_errors, padding_rows = _zero_row_errors(
        plan.zero_fill_ranges,
        hidden_view,
        weights_view,
    )
    errors = scatter_errors + padding_errors
    assert_all_ranks(
        checked > 0 and padding_rows > 0 and not errors,
        rank,
        R,
        "overlapped prepared dispatch publication",
        "; ".join(errors[:5])
        or f"checked={checked}, padding_rows={padding_rows}",
    )


def test_prepared_dispatch_object_rejects_stale_generation(dist_env):
    rank, R = dist_env
    case = KernelCase(
        "stale_prepared_object",
        S=8,
        K=2,
        epn=4,
        H=32,
        num_sms=4,
        B=1,
        token_padding=8,
        routing="all_remote",
        min_R=2,
    )
    ctx = init_case(case, R)
    buffer = ctx["_buffer"]
    topk, tpe = make_topk(case, rank, R)

    old_target, _old_cu, old_plan = buffer.prepare_dispatch(topk, tpe)
    buffer.prepare_dispatch(topk, tpe)
    with pytest.raises(RuntimeError, match="stale"):
        buffer.publish_dispatch(old_target, old_plan)


def test_owner_pull_materializes_source_object(dist_env):
    """Explicit peer-VMM loads materialize the one-copy source view correctly."""
    rank, R = dist_env
    case = KernelCase(
        "owner_pull_source",
        S=31,
        K=4,
        epn=8,
        H=64,
        num_sms=8,
        B=2,
        token_padding=16,
        routing="duplicate_topk",
        min_R=2,
        enable_source_object=True,
    )
    ctx = init_case(case, R)
    buffer = ctx["_buffer"]
    topk, tpe = make_topk(case, rank, R)
    hidden = _traceable_hidden(rank, case.S, case.H)
    weights = _traceable_weights(rank, case.S, case.K)

    storage = buffer.source_storage()
    storage.local_hidden.copy_(hidden)
    storage.local_route_weights.copy_(weights)
    view, _cu_seqlens, plan, planning_done = buffer.prepare_source_dispatch(
        storage,
        topk,
        tpe,
        async_finish=True,
    )
    torch.cuda.current_stream().wait_event(planning_done)
    hidden_view, weights_view, event = buffer.materialize_source_dispatch(
        view,
        plan,
        with_route_weights=True,
        zero_copy=True,
    )
    assert event is None
    assert hidden_view.data_ptr() == ctx["hidden_buf_local"].data_ptr()
    assert weights_view.data_ptr() == ctx["weights_buf_local"].data_ptr()
    torch.cuda.synchronize()

    scatter_errors, checked = _verify_dispatch_by_dst(
        ctx,
        case,
        rank,
        R,
        plan.dst,
        hidden_user=hidden_view,
        weights_user=weights_view,
    )
    padding_errors, padding_rows = _zero_row_errors(
        plan.zero_fill_ranges,
        hidden_view,
        weights_view,
    )
    errors = scatter_errors + padding_errors
    assert_all_ranks(
        checked > 0 and padding_rows > 0 and not errors,
        rank,
        R,
        "owner-pull source materialization",
        "; ".join(errors[:5])
        or f"checked={checked}, padding_rows={padding_rows}",
    )

    hidden_copy, weights_copy, done = buffer.materialize_source_dispatch(
        view,
        plan,
        with_route_weights=True,
        async_finish=True,
        zero_copy=False,
    )
    assert done is not None
    done.synchronize()
    async_scatter_errors, async_checked = _verify_dispatch_by_dst(
        ctx,
        case,
        rank,
        R,
        plan.dst,
        hidden_user=hidden_copy,
        weights_user=weights_copy,
    )
    async_padding_errors, async_padding_rows = _zero_row_errors(
        plan.zero_fill_ranges,
        hidden_copy,
        weights_copy,
    )
    async_errors = async_scatter_errors + async_padding_errors
    assert_all_ranks(
        async_checked > 0 and async_padding_rows > 0 and not async_errors,
        rank,
        R,
        "async owner-pull source materialization",
        "; ".join(async_errors[:5])
        or f"checked={async_checked}, padding_rows={async_padding_rows}",
    )


@pytest.mark.parametrize("case", case_params(LARGE_DISPATCH_CASES))
def test_dispatch_large_hidden_stride_spotcheck(dist_env, case):
    rank, R = dist_env
    ctx = init_case(case, R)
    hidden = _traceable_hidden(rank, case.S, case.H)
    weights = _traceable_weights(rank, case.S, case.K)

    dst, _plan, hidden_user, weights_user = _plan_and_dispatch(
        ctx, case, rank, R, hidden, weights
    )
    torch.cuda.synchronize()

    errors, checked = _verify_dispatch_by_dst(
        ctx, case, rank, R, dst, hidden_user=hidden_user,
        weights_user=weights_user, max_checks=256, copy_to_cpu=False
    )
    assert_all_ranks(
        checked > 0 and not errors,
        rank,
        R,
        f"{case.name} dispatch large-stride spotcheck",
        "; ".join(errors[:5]) or f"checked={checked}",
    )


def test_dispatch_rejects_bad_inputs(dist_env):
    from moonep.dispatch import launch_dispatch
    from moonep.dispatch_epilogue import launch_dispatch_epilogue
    from moonep.planning import allocate_planning_outputs, launch_planning

    rank, R = dist_env
    case = KernelCase("bad_inputs", S=4, K=2, epn=2, H=8, num_sms=1, token_padding=1)
    ctx = init_case(case, R)
    topk, tpe = make_topk(case, rank, R)
    hidden = _traceable_hidden(rank, case.S, case.H)
    weights = torch.rand(case.S, case.K, dtype=torch.float32, device=f"cuda:{rank}")
    plan, _cu = allocate_planning_outputs(ctx)
    launch_planning(ctx, topk.reshape(-1).contiguous(), tpe, _cu, plan)

    with pytest.raises(AssertionError, match="hidden_sh"):
        launch_dispatch(ctx, hidden.float(), weights, plan, build_dedup_map=True)
    with pytest.raises(AssertionError, match="must be None"):
        launch_dispatch(
            ctx,
            hidden,
            weights,
            plan,
            write_payload=False,
            build_dedup_map=True,
        )
    with pytest.raises(AssertionError, match="hidden_sh"):
        launch_dispatch(
            ctx,
            hidden[: case.S // 2].contiguous(),
            weights,
            plan,
            build_dedup_map=True,
        )
    with pytest.raises(AssertionError, match="route_weights_sk"):
        launch_dispatch(ctx, hidden, weights.double(), plan, build_dedup_map=True)
    with pytest.raises(AssertionError, match="route_weights_sk"):
        launch_dispatch(
            ctx,
            hidden,
            weights[:, : case.K - 1].contiguous(),
            plan,
            build_dedup_map=True,
        )
    bad_plan = plan.clone()
    object.__setattr__(bad_plan, "dst", plan.dst.long())
    with pytest.raises(AssertionError, match="dst"):
        launch_dispatch(ctx, hidden, weights, bad_plan, build_dedup_map=True)
    bad_epilogue_plan = plan.clone()
    object.__setattr__(
        bad_epilogue_plan,
        "dup_loffs",
        plan.dup_loffs.flatten()[:-1].contiguous(),
    )
    with pytest.raises(AssertionError, match="dup_loffs"):
        launch_dispatch_epilogue(ctx, bad_epilogue_plan)
