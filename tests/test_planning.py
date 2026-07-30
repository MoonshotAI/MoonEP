"""Planning kernel correctness tests.

Run with:
    torchrun --nproc_per_node=8 -m pytest -s tests/test_planning.py
"""

import pytest
import torch
from tests.kernel_test_utils import (
    DEFAULT_TOKEN_PADDING,
    KernelCase,
    assert_all_ranks,
    assert_tensor_equal_all_ranks,
    case_params,
    gather_tensor,
    init_case,
    make_topk,
    planning_invariant_errors,
    skip_if_unsupported_world_size,
)
from tests.planning_reference import launch_planning_torch_reference


PLANNING_CASES = [
    KernelCase(
        "balanced_epn16",
        S=256,
        K=8,
        epn=16,
        H=128,
        num_sms=32,
        token_padding=DEFAULT_TOKEN_PADDING,
    ),
    KernelCase("tiny_s1_k1", S=1, K=1, epn=1, H=16, num_sms=1, token_padding=8),
    KernelCase(
        "tiny_biased_with_prefetch",
        S=3,
        K=5,
        epn=3,
        H=16,
        num_sms=1,
        B=1,
        token_padding=32,
        routing="biased",
        bias_ratio=1.0,
        seed=9002,
        min_R=2,
    ),
    KernelCase(
        "no_padding",
        S=33,
        K=2,
        epn=2,
        H=16,
        num_sms=1,
        B=1,
        token_padding=1,
        routing="biased",
        bias_ratio=1.0,
        seed=10001,
    ),
    KernelCase(
        "non_power_mild_bias",
        S=17,
        K=3,
        epn=3,
        H=16,
        num_sms=1,
        B=2,
        token_padding=16,
        routing="biased",
        bias_ratio=0.5,
        seed=10501,
    ),
    KernelCase(
        "small_balanced_with_prefetch",
        S=17,
        K=3,
        epn=3,
        H=16,
        num_sms=1,
        B=2,
        token_padding=16,
        seed=11001,
    ),
    KernelCase(
        "near_degenerate_bias",
        S=64,
        K=2,
        epn=5,
        H=32,
        num_sms=32,
        B=3,
        routing="biased",
        bias_ratio=5.0,
        seed=12001,
    ),
    KernelCase(
        "typical_bias",
        S=32,
        K=4,
        epn=4,
        H=32,
        num_sms=32,
        B=2,
        token_padding=64,
        routing="biased",
        bias_ratio=1.0,
        seed=13001,
    ),
    KernelCase(
        "heavy_bias",
        S=96,
        K=3,
        epn=7,
        H=48,
        num_sms=1,
        B=4,
        routing="biased",
        bias_ratio=2.0,
        seed=14001,
    ),
    KernelCase(
        "step1_cross_cta_group",
        S=1024,
        K=1,
        epn=256,
        H=16,
        num_sms=8,
        B=1,
        token_padding=1,
    ),
    KernelCase(
        "step1_multi_chunk_full_tile",
        S=2048,
        K=1,
        epn=320,
        H=16,
        num_sms=1,
        B=1,
        token_padding=1,
    ),
    KernelCase(
        "step1_segment_tail_full_tile",
        S=1536,
        K=1,
        epn=320,
        H=16,
        num_sms=2,
        B=1,
        token_padding=1,
        min_R=4,
    ),
    KernelCase(
        "experts_gt_block_size",
        S=256,
        K=1,
        epn=1025,
        H=16,
        num_sms=8,
        B=1,
        token_padding=1,
        max_R=2,
    ),
    KernelCase(
        "all_local",
        S=32,
        K=4,
        epn=8,
        H=32,
        num_sms=8,
        token_padding=16,
        routing="all_local",
    ),
    KernelCase(
        "all_remote_with_prefetch_slots",
        S=32,
        K=4,
        epn=8,
        H=32,
        num_sms=8,
        B=3,
        token_padding=16,
        routing="all_remote",
        min_R=2,
    ),
    KernelCase(
        "duplicate_topk",
        S=16,
        K=4,
        epn=4,
        H=32,
        num_sms=8,
        B=2,
        token_padding=8,
        routing="duplicate_topk",
        min_R=2,
    ),
    KernelCase(
        "single_expert_exact_padding",
        S=8,
        K=2,
        epn=4,
        H=16,
        num_sms=1,
        B=1,
        token_padding=8,
        routing="single_expert",
    ),
    KernelCase(
        "single_expert_padding_tail",
        S=9,
        K=2,
        epn=4,
        H=16,
        num_sms=1,
        B=1,
        token_padding=8,
        routing="single_expert",
    ),
]


def _align_up(x, alignment):
    return ((x + alignment - 1) // alignment) * alignment


def _step1_params(case, R):
    E = case.E(R)
    seg_raw = (E + case.num_sms - 1) // case.num_sms
    experts_per_block = _align_up(seg_raw, 32)
    s1_cols = min(experts_per_block, 512)
    work_ctas = (E + experts_per_block - 1) // experts_per_block
    group_spans_ctas = experts_per_block < case.epn
    has_segment_tail = experts_per_block > s1_cols and experts_per_block % s1_cols != 0
    return {
        "E": E,
        "experts_per_block": experts_per_block,
        "s1_cols": s1_cols,
        "work_ctas": work_ctas,
        "group_spans_ctas": group_spans_ctas,
        "has_segment_tail": has_segment_tail,
    }


def test_planning_step1_case_coverage():
    params = [
        _step1_params(case, R)
        for R in (2, 4)
        for case in PLANNING_CASES
        if R >= case.min_R and (case.max_R is None or R <= case.max_R)
    ]

    assert any(p["E"] > 2048 for p in params)
    assert any(p["s1_cols"] > 32 for p in params)
    assert any(p["s1_cols"] == 512 for p in params)
    assert any(p["experts_per_block"] > p["s1_cols"] for p in params)
    assert any(p["has_segment_tail"] for p in params)
    assert any(p["work_ctas"] > 1 and p["group_spans_ctas"] for p in params)


@pytest.mark.parametrize("case", case_params(PLANNING_CASES))
def test_planning_matches_reference_and_invariants(dist_env, case):
    from moonep.planning import allocate_planning_outputs, launch_planning

    rank, R = dist_env
    skip_if_unsupported_world_size(case, R)

    ctx = init_case(case, R)
    assert (ctx["source_primary_local"] is not None) == \
        case.enable_source_object
    assert (ctx["source_next_local"] is not None) == case.enable_source_object
    topk, tpe = make_topk(case, rank, R)

    plan, cu_seqlens = allocate_planning_outputs(ctx)
    assert (
        launch_planning(ctx, topk.reshape(-1).contiguous(), tpe, cu_seqlens, plan)
        is None
    )
    dst = plan.dst
    experts_to_copy = plan.experts_to_copy
    remote_stats = plan.remote_stats
    (
        ref_dst,
        ref_cu_seqlens,
        ref_experts_to_copy,
        ref_remote_stats,
        ref_zero_fill_ranges,
        _ref_dedup_plan,
    ) = launch_planning_torch_reference(ctx, topk, tpe)

    assert_tensor_equal_all_ranks("cu_seqlens", cu_seqlens, ref_cu_seqlens, rank, R)
    assert_tensor_equal_all_ranks(
        "zero_fill_ranges", plan.zero_fill_ranges, ref_zero_fill_ranges, rank, R
    )
    assert_tensor_equal_all_ranks(
        "experts_to_copy", experts_to_copy, ref_experts_to_copy, rank, R
    )
    assert_tensor_equal_all_ranks(
        "remote_stats", remote_stats, ref_remote_stats, rank, R
    )
    assert_tensor_equal_all_ranks(
        "dst", dst.reshape(case.S, case.K), ref_dst.reshape(case.S, case.K), rank, R
    )

    errors = planning_invariant_errors(case, ctx, dst, cu_seqlens, experts_to_copy)
    assert_all_ranks(
        not errors,
        rank,
        R,
        f"{case.name} planning invariants",
        "; ".join(errors[:5]),
    )


def test_owner_sharded_source_view_matches_planner_provenance(dist_env):
    """The direct-consume view keeps one source row and exposes its slot map."""
    rank, R = dist_env
    case = KernelCase(
        "owner_sharded_source",
        S=17,
        K=3,
        epn=8,
        H=64,
        num_sms=8,
        B=2,
        token_padding=16,
        routing="all_remote",
        min_R=2,
        enable_source_object=True,
    )
    ctx = init_case(case, R)
    buffer = ctx["_buffer"]
    topk, tpe = make_topk(case, rank, R)

    storage = buffer.source_storage()
    hidden = (
        torch.arange(case.S * case.H, device=f"cuda:{rank}")
        .reshape(case.S, case.H)
        .to(torch.bfloat16)
        + rank * 1024
    )
    storage.local_hidden.copy_(hidden)
    view, cu_seqlens, plan = buffer.prepare_source_dispatch(
        storage,
        topk,
        tpe,
    )
    assert view.capability == "peer_indexed_source_v1"
    assert view.storage is storage
    assert view.plan is plan
    assert view.slot_to_source.data_ptr() == ctx["source_info_local"].data_ptr()
    assert view.slot_to_primary.data_ptr() == \
        ctx["source_primary_local"].data_ptr()
    assert view.slot_to_next.data_ptr() == ctx["source_next_local"].data_ptr()
    assert torch.equal(storage.local_hidden, hidden)
    assert tuple(cu_seqlens.shape) == (case.E(R) + int(ctx["B"]),)

    all_dst = gather_tensor(plan.dst, R).cpu()
    source_info = view.slot_to_source.cpu()
    source_primary = view.slot_to_primary.cpu()
    source_next = view.slot_to_next.cpu()
    expected = torch.full_like(source_info, -1)
    expected_primary = torch.full_like(source_primary, -1)
    expected_next = torch.full_like(source_next, -1)
    NvS = int(ctx["NvS"])
    for src_rank in range(R):
        src_dst = all_dst[src_rank].tolist()
        for source_token in range(case.S):
            primary_by_rank = {}
            previous_by_rank = {}
            for kidx in range(case.K):
                source_topk_offset = source_token * case.K + kidx
                encoded_dst = src_dst[source_topk_offset]
                raw_dst = encoded_dst if encoded_dst >= 0 else -encoded_dst - 1
                dest_rank = raw_dst // NvS
                loff = raw_dst % NvS
                if encoded_dst >= 0:
                    primary_by_rank[dest_rank] = loff
                if dest_rank == rank:
                    expected[loff] = src_rank * NvS + source_topk_offset
                    expected_primary[loff] = primary_by_rank[dest_rank]
                    if dest_rank in previous_by_rank:
                        expected_next[previous_by_rank[dest_rank]] = loff
                    previous_by_rank[dest_rank] = loff

    errors = []
    if not torch.equal(source_info, expected):
        mismatch = (source_info != expected).nonzero().flatten()
        for loff in mismatch[:5].tolist():
            errors.append(
                f"loff={loff}: actual={int(source_info[loff])}, "
                f"expected={int(expected[loff])}"
            )
    if not torch.equal(source_primary, expected_primary):
        mismatch = (source_primary != expected_primary).nonzero().flatten()
        for loff in mismatch[:5].tolist():
            errors.append(
                f"primary loff={loff}: actual={int(source_primary[loff])}, "
                f"expected={int(expected_primary[loff])}"
            )
    if not torch.equal(source_next, expected_next):
        mismatch = (source_next != expected_next).nonzero().flatten()
        for loff in mismatch[:5].tolist():
            errors.append(
                f"next loff={loff}: actual={int(source_next[loff])}, "
                f"expected={int(expected_next[loff])}"
            )
    valid = source_info >= 0
    if not bool(valid.any()):
        errors.append("source view has no valid local expert slots")
    else:
        encoded = source_info[valid]
        source_ranks = encoded // NvS
        source_tokens = (encoded % NvS) // case.K
        if not bool(((source_ranks >= 0) & (source_ranks < R)).all()):
            errors.append("decoded source rank is out of range")
        if not bool(((source_tokens >= 0) & (source_tokens < case.S)).all()):
            errors.append("decoded source token is out of range")

    assert_all_ranks(
        not errors,
        rank,
        R,
        "owner-sharded source provenance",
        "; ".join(errors),
    )
