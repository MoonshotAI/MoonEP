"""FP8 dispatch, scale routing, BF16 combine, and plan reuse.

Quick check: torchrun --standalone --nproc_per_node=2 -m pytest -q tests/test_dispatch_fp8.py
Full topology: use --nproc_per_node=8 (one cooperating pytest worker per GPU).
"""

import pytest
import torch

from tests.kernel_test_utils import (
    KernelCase, assert_all_ranks, assert_ulp_all_ranks, case_params,
    clone_dedup_plan_fields, dedup_plan_fields_equal, gather_tensor,
    init_case, make_topk,
)
from tests.quantization_reference import dequantize_fp8, quantize_fp8


FP8_CASES = [
    KernelCase("one_tile", S=1, K=1, epn=2, H=128, num_sms=1, token_padding=1),
    KernelCase("scale_tail", S=33, K=4, epn=4, H=384, num_sms=8,
               token_padding=16, routing="all_remote", min_R=2),
    KernelCase("local_padding", S=17, K=3, epn=4, H=512, num_sms=8,
               token_padding=16, routing="all_local"),
    KernelCase("duplicate_topk", S=32, K=4, epn=4, H=3584, num_sms=8,
               token_padding=8, routing="duplicate_topk", min_R=2),
    KernelCase("single_expert_split", S=9, K=2, epn=4, H=7168, num_sms=8,
               token_padding=8, routing="single_expert", min_R=2),
    KernelCase("balanced", S=64, K=8, epn=8, H=7168, num_sms=8,
               token_padding=16),
    KernelCase("pipeline_reuse", S=1025, K=4, epn=4, H=7168, num_sms=8,
               token_padding=16, routing="all_remote", min_R=2),
]

# Independent hidden/router-weight zero-copy controls, optional weights,
# async completion, PDL on/off, and input scale layouts. Row-major 16-B
# aligned scales take the TMA path when H % 512 == 0; column-major and
# misaligned scales take the generic-load path.
MODES = [
    pytest.param(False, False, True, False, True, "row", id="copy"),
    pytest.param(True, False, True, True, True, "column", id="view_async_column"),
    pytest.param(False, True, True, True, False, "column", id="weights_view_no_pdl"),
    pytest.param(True, False, False, False, False, "row", id="hidden_only_view"),
    pytest.param(False, False, True, False, True, "misaligned", id="copy_misaligned"),
]


def _input(case, rank, seed, scale_layout="row"):
    torch.manual_seed(seed + rank)
    hidden = torch.randn(case.S, case.H, dtype=torch.bfloat16, device="cuda")
    # Different scales across tiles exercise both the tile axis and scale tail.
    magnitudes = 2.0 ** (torch.arange(case.H // 128, device="cuda") % 17 - 8)
    hidden = (hidden.reshape(case.S, -1, 128) * magnitudes[None, :, None]).reshape_as(hidden)
    if case.S > 1:
        hidden[0].zero_()
    data, scales = quantize_fp8(hidden)
    if scale_layout == "column":
        scales = scales.t().contiguous().t()
    elif scale_layout == "misaligned":
        # Contiguous, but 4 B past a 16-B boundary.
        storage = torch.empty(scales.numel() + 1, dtype=scales.dtype, device=scales.device)
        scales = storage[1:].view_as(scales).copy_(scales)
        assert scales.is_contiguous() and scales.data_ptr() % 16 == 4
    return data, scales


def _finish(result, async_finish):
    if async_finish:
        result[-1].wait()
        return result[:-1]
    return result


def _verify(ctx, case, rank, R, plan, source, received, weights=None, recv_weights=None):
    if isinstance(source, tuple):
        source_bits = (source[0].view(torch.uint8), source[1].view(torch.int32))
        received_bits = (received[0].view(torch.uint8), received[1].view(torch.int32))
    else:
        # NCCL supports byte tensors; int16 is not a collective dtype.
        source_bits = (source.view(torch.uint8),)
        received_bits = (received.view(torch.uint8),)
    dst = gather_tensor(plan.dst, R).reshape(R, case.S, case.K).long()
    raw = torch.where(dst < 0, -dst - 1, dst)
    mine = raw // ctx['NvS'] == rank
    slots = raw[mine] % ctx['NvS']
    sources = torch.arange(R * case.S, device="cuda").reshape(R, case.S, 1)
    sources = sources.expand_as(raw)[mine]
    rows = torch.arange(ctx['NvS'], device="cuda")[:, None]
    ranges = plan.zero_fill_ranges
    padding = ((rows >= ranges[:, 0]) & (rows < ranges[:, 0] + ranges[:, 1])).any(dim=1)
    ok = True
    for sent, recv in zip(source_bits, received_bits):
        all_sent = gather_tensor(sent.contiguous(), R).reshape(-1, sent.shape[1])
        ok &= torch.equal(recv[slots], all_sent[sources])
        ok &= bool((recv[padding] == 0).all().item())
    if weights is not None:
        all_weights = gather_tensor(weights, R)
        ok &= torch.equal(recv_weights[slots], all_weights[mine])
    else:
        ok &= recv_weights is None
    if recv_weights is not None:
        ok &= bool((recv_weights[padding] == 0).all().item())
    assert_all_ranks(ok, rank, R, "dispatched data/scales/weights and padding match routing bitwise")


@pytest.mark.parametrize("case", case_params(FP8_CASES))
@pytest.mark.parametrize("zero_copy,weights_view,with_weights,async_finish,pdl,scale_layout", MODES)
def test_fp8_dispatch(dist_env, monkeypatch, case, zero_copy, weights_view, with_weights,
                      async_finish, pdl, scale_layout):
    rank, R = dist_env
    if case.name == "pipeline_reuse":
        # 129 tokens per dispatch CTA exceed its maximum 16 stages. Limit the
        # epilogue grid too, so duplicate expansion also reuses pipeline slots.
        monkeypatch.setenv("MOONEP_NUM_SMS_DEDUP", "2")
    ctx = init_case(case, R)
    buffer = ctx['_buffer']
    buffer.enable_pdl = pdl
    topk, tpe = make_topk(case, rank, R)
    source = _input(case, rank, 123, scale_layout)
    weights = torch.rand(case.S, case.K, device="cuda") if with_weights else None
    ctx['hidden_fp8_buf_local'].view(torch.uint8).fill_(0x55)
    ctx['scales_buf_local'].fill_(17)
    received, recv_weights, cu, plan = _finish(buffer.dispatch(
        source, weights, topk, tpe, zero_copy=zero_copy,
        router_weights_zero_copy=weights_view, async_finish=async_finish,
    ), async_finish)
    assert cu is not None
    assert isinstance(received, tuple)
    assert received[0].dtype == torch.float8_e4m3fn
    assert received[1].dtype == torch.float32
    assert all(t.is_contiguous() for t in received)
    for output, key in zip(received, ('hidden_fp8_buf_local', 'scales_buf_local')):
        assert (output.data_ptr() == ctx[key].data_ptr()) == zero_copy
    if with_weights:
        assert (recv_weights.data_ptr() == buffer.router_weight_buffer_view.data_ptr()) == weights_view
    _verify(ctx, case, rank, R, plan, source, received, weights, recv_weights)
    if case.name == "pipeline_reuse":
        # Even the largest epilogue geometry (32 rows x 16 stages) must wrap.
        # Check the actual routing produced enough work for the two-CTA grid.
        assert_all_ranks(int(plan.dup_counts[0].item()) > 32 * 16 * ctx['num_sms_dedup'],
                         rank, R, "enough duplicate groups to reuse epilogue stages")
    snapshot = clone_dedup_plan_fields(plan)
    saved = tuple(t.clone() for t in received)

    # The same allocation and saved routing plan can switch between bf16 and
    # FP8 (including the backward dispatch path). Owned outputs must survive.
    bf16 = dequantize_fp8(*source)
    bf16_received, bf16_weights, bf16_cu, bf16_plan = _finish(buffer.dispatch(
        bf16, weights, plan=plan, zero_copy=True,
        router_weights_zero_copy=weights_view, async_finish=async_finish,
    ), async_finish)
    assert bf16_received.dtype == torch.bfloat16
    assert bf16_plan is plan and bf16_cu is None
    _verify(ctx, case, rank, R, plan, bf16, bf16_received, weights, bf16_weights)
    if not zero_copy:
        assert_all_ranks(
            torch.equal(received[0].view(torch.uint8), saved[0].view(torch.uint8))
            and torch.equal(received[1].view(torch.int32), saved[1].view(torch.int32)),
            rank, R, "owned FP8 outputs survive subsequent communication",
        )

    source = _input(case, rank, 456, scale_layout)
    received, recv_weights, reused_cu, reused = _finish(buffer.dispatch(
        source, plan=plan, zero_copy=zero_copy, async_finish=async_finish,
    ), async_finish)
    assert reused is plan and reused_cu is None
    _verify(ctx, case, rank, R, plan, source, received)
    assert_all_ranks(dedup_plan_fields_equal(plan, snapshot), rank, R, "FP8 plan reuse")

    # Model an identity expert that consumes FP8 and produces BF16. Cover the
    # explicit BF16 output view too, as FP8 views share the combine allocation.
    expert_output = dequantize_fp8(*received)
    if zero_copy:
        buffer.hidden_nvsh_buffer_view.copy_(expert_output)
        expert_output = buffer.hidden_nvsh_buffer_view
    combined, _, event = buffer.combine(
        plan=plan, hidden_nvsh=expert_output, zero_copy=zero_copy,
        async_finish=async_finish,
    )
    if event is not None:
        event.wait()
    expected = (dequantize_fp8(*source).float() * case.K).to(torch.bfloat16)
    assert_ulp_all_ranks("FP8 dispatch -> BF16 combine", combined, expected, rank, R, max_ulps=2)


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("case,scale_layout", [
    pytest.param(FP8_CASES[1], "column", id="generic_scales"),
    pytest.param(FP8_CASES[4], "row", id="tma_scales"),
])
def test_fp8_cuda_graph(dist_env, pdl, case, scale_layout):
    rank, R = dist_env
    ctx = init_case(case, R)
    buffer = ctx['_buffer']
    buffer.enable_pdl = pdl
    topk, tpe = make_topk(case, rank, R)
    source = _input(case, rank, 789, scale_layout)
    _, _, _, plan = buffer.dispatch(source, None, topk, tpe, zero_copy=True)
    for reuse in (False, True):
        def dispatch():
            return buffer.dispatch(source, None, topk, tpe,
                                   plan=plan if reuse else None, zero_copy=True)
        for _ in range(3):
            dispatch()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            received, _, _, captured_plan = dispatch()
        next_source = _input(case, rank, 999 + int(reuse), scale_layout)
        for dst, src in zip(source, next_source):
            dst.copy_(src)
        graph.replay()
        _verify(ctx, case, rank, R, captured_plan, source, received)


@pytest.mark.parametrize("invalid,match", [
    ("bare_fp8", "tuple"), ("tuple_length", "tuple"),
    ("fp8_dtype", "float8_e4m3fn"), ("scale_dtype", "float32"),
    ("scale_shape", "scales must be shape"), ("scale_device", "same device"),
    ("data_shape", "hidden_sh must be shape"), ("data_stride", "contiguous"),
    ("data_alignment", "16-byte aligned"), ("hidden_size", "divisible by 128"),
])
def test_fp8_invalid_input(dist_env, invalid, match):
    _, R = dist_env
    # 192 is BF16-compatible but not divisible by 128. Its shorter VMM
    # alignment period keeps this argument check cheap at large world sizes.
    H = 192 if invalid == "hidden_size" else 384
    case = KernelCase("invalid", S=2, K=1, epn=2, H=H, num_sms=1)
    ctx = init_case(case, R)
    data = torch.zeros(2, H, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.ones(2, H // 128, device="cuda")
    if invalid == "fp8_dtype":
        data = data.to(torch.float8_e5m2)
    elif invalid == "scale_dtype":
        scales = scales.to(torch.bfloat16)
    elif invalid == "scale_shape":
        scales = scales[:, :-1]
    elif invalid == "scale_device":
        scales = scales.cpu()
    elif invalid == "data_shape":
        data = data[:1]
    elif invalid == "data_stride":
        data = torch.empty(2, H * 2, device="cuda", dtype=torch.float8_e4m3fn)[:, ::2]
    elif invalid == "data_alignment":
        data = torch.empty(2 * H + 1, device="cuda", dtype=torch.float8_e4m3fn)[1:].view(2, H)
    source = data if invalid == "bare_fp8" else (data, scales)
    if invalid == "tuple_length":
        source = (data,)
    # No routing inputs: invalid activations must fail before planning starts.
    with pytest.raises(AssertionError, match=match):
        ctx['_buffer'].dispatch(source)
