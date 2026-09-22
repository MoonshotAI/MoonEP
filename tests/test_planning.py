"""Planning kernel correctness tests.

Run with:
    torchrun --nproc_per_node=8 -m pytest -s tests/test_planning.py
"""

import pytest
from tests.kernel_test_utils import (
    DEFAULT_TOKEN_PADDING,
    KernelCase,
    assert_all_ranks,
    assert_tensor_equal_all_ranks,
    case_params,
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
        token_padding=1,
    ),
    KernelCase(
        "step1_multi_chunk_full_tile",
        S=2048,
        K=1,
        epn=320,
        H=16,
        num_sms=1,
        token_padding=1,
    ),
    KernelCase(
        "step1_segment_tail_full_tile",
        S=1536,
        K=1,
        epn=320,
        H=16,
        num_sms=2,
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
        token_padding=1,
        max_R=2,
    ),
    KernelCase(
        "smem_boundary_e2880",
        S=64,
        K=2,
        epn=360,
        H=16,
        num_sms=32,
        token_padding=1,
        min_R=8,
        max_R=8,
    ),
    KernelCase(
        "smem_overflow_e2888",
        S=64,
        K=2,
        epn=361,
        H=16,
        num_sms=32,
        token_padding=1,
        min_R=8,
        max_R=8,
    ),
    KernelCase(
        "smem_overflow_e3072",
        S=64,
        K=2,
        epn=384,
        H=16,
        num_sms=32,
        token_padding=1,
        min_R=8,
        max_R=8,
    ),
    KernelCase(
        "smem_overflow_e4608",
        S=64,
        K=2,
        epn=576,
        H=16,
        num_sms=32,
        token_padding=1,
        min_R=8,
        max_R=8,
    ),
    KernelCase(
        "smem_overflow_e6144",
        S=64,
        K=2,
        epn=768,
        H=16,
        num_sms=32,
        token_padding=1,
        min_R=8,
        max_R=8,
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
