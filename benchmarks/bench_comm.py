"""
MoonEP intranode dispatch/combine operator benchmark.

Sweeps:
  - hidden_size: 3584, 7168
  - topk:        8, 16
  - ep_size:     4, 8  (sub-groups carved out of the global world)
  - seq_length:  8192
  - bias_ratio:  0.1, 1, 5
                 sigma of the lognormal expert-logit distribution (mirrors
                 the unbalance forcing used in training). 0.1 -> near
                 balanced; 1 -> typical dropless-MoE skew; 5 -> near-degenerate.

E is fixed at 896 total experts; experts-per-rank (epn = E / ep) shrinks as
the EP group grows.

Reports the MoonEP communication operators used by Megatron:
  - dispatch_fwd: dispatch kernel with route-weight scatter (dedup: only
    representative ``dst >= 0`` rows are transferred).
  - publish_meta: producer-direct-out completion with the hidden payload
    already in final owner placement. It scatters route metadata, zero-fills
    padding, rebuilds dedup metadata, and publishes visibility. The difference
    from dispatch_fwd is an upper bound on removable standalone payload-path
    cost, not an end-to-end fused-producer speedup.
  - source_pull: runtime materialization from the one-physical-copy
    owner-sharded source object. It includes route metadata, padding, and
    explicit peer-VMM row loads. Because it writes duplicate slots directly,
    compare it with dispatch_fwd + epilogue_fwd for the same ready-to-consume
    contiguous expert view.
  - dispatch_bwd: hidden-only dispatch kernel with the saved plan.
  - epilogue_fwd: in-place duplicate expansion on the NVL shard
    (dispatch_epilogue kernel), timed after dispatch_fwd.
  - epilogue_bwd: same expansion timed after the saved-plan dispatch_bwd.
  - combine_prologue_fwd: in-place fp32 accumulation of duplicate rows into
    their primary (combine_prologue kernel).
  - combine_prologue_bwd: same accumulation timed on the backward staging
    state.
  - combine_fwd:  dedup-aware combine kernel after the prologue.
  - combine_bwd:  dedup-aware combine kernel with droute_weights_sk gather
                  after the prologue.
  - prefetch:     remote expert-weight prefetch driven by the planner's
                  experts_to_copy. The kernel has no inter-rank sync of its
                  own, so each timed iteration is prefetch + a small
                  inter-rank sync kernel to keep ranks serialized.
  - grad_reduce:  remote expert-grad reduction (cross-rank barrier built in).

Expert weights/grads use one [Hp]-wide proxy tensor per rank, NVL-distributed
so each rank physically owns its epn experts (remote rows are NVLink-mapped).
max_recv is the max over ranks of prefetched expert count (slots with
experts_to_copy >= 0); max_send is the max over owner ranks of slots whose
grads flow back to that owner. Both prefetch and grad_reduce bandwidth use
the bottleneck rank's traffic max(max_send, max_recv) * H * Hp as the byte
count (bf16 for prefetch, fp32 for grad_reduce). mx/mean is the global
expert-load imbalance
(max over experts of routed token count, divided by the mean).

The epilogue/prologue bandwidth columns use total HBM traffic (read + write
bytes divided by time; route weights are negligible and ignored):
  epilogue = (dup_group_count + dup_loff_count) * H * 2  (read primaries,
             write duplicates)
  prologue = (2 * dup_group_count + dup_loff_count) * H * 2  (read primary +
             duplicates, write primary)

Planning is run once per config to populate `dst` and src_info. Fresh forward
dispatch materializes the plan-owned dedup structures with its builder warps
(time included in dispatch_fwd) and zero-fills padding with the zero warp.
Saved-plan backward dispatch reuses the existing dedup structures. The
zero_copy=False boundary ``tensor.copy_`` calls are plain torch ops, not
MoonEP kernels, and are not benchmarked here (with zero_copy=True they
disappear entirely).

Set ``MOONEP_NUM_SMS_DEDUP=<n>`` before launching to override the SM count used
by the local dispatch_epilogue and combine_prologue kernels. By default they
use all SMs on the current device.
"""

import argparse
import csv
import gc
import math
import os
import time
import traceback

import torch
import torch.distributed as dist

from tests.generate_topk_routing import generate_topk_routing

from moonep import Buffer
from moonep.buffer import create_nvl_dist_tensor, pad_dim0_for_alignment
from moonep.dispatch import launch_dispatch
from moonep.dispatch_epilogue import launch_dispatch_epilogue
from moonep.combine import launch_combine
from moonep.combine_prologue import launch_combine_prologue
from moonep.grad_reduce import launch_grad_reduce
from moonep.inter_rank_sync import launch_inter_rank_sync
from moonep.planning import allocate_planning_outputs, launch_planning
from moonep.prefetch import launch_prefetch


def fmt4(x: float) -> str:
    """Format float to exactly 4 significant digits, keeping trailing zeros.

    Examples: 181.0 -> '181.0', 1486.65 -> '1487', 856.71 -> '856.7', 0.5 -> '0.5000'.
    """
    if x == 0:
        return "0.000"
    d = math.floor(math.log10(abs(x))) + 1  # digits before the decimal point
    return f"{x:.{max(0, 4 - d)}f}"


# -------------------------------------------------------------------------
# Distributed setup
# -------------------------------------------------------------------------

def setup():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    world_rank = dist.get_rank()
    world_size = dist.get_world_size()
    return world_rank, world_size, local_rank


# -------------------------------------------------------------------------
# Benchmark a single config (called only on ranks in the EP subgroup)
# -------------------------------------------------------------------------

def time_gpu_op(launch_fn, warmup, iters, group, cudagraph=True):
    """Run launch_fn() warmup+iters times; return cross-rank mean per-iter us.

    Pure GPU-side timing: no host syncs inside the loop. The kernel's own
    cross-rank barrier serializes successive calls. Each rank in `group`
    measures its own elapsed time; we all_gather the per-rank means and return
    the average across the subgroup.

    cudagraph=True (default): capture `iters` launches and time the replay,
    removing the per-launch Python overhead (cute JIT orchestration +
    make_ptr) so the timing reflects the kernels alone. cooperative launch +
    the cross-rank NVLink barrier are capturable; output tensors are freshly
    allocated each round and the replay reuses the captured addresses.
    cudagraph=False: eager loop (includes host launch overhead).
    """
    for _ in range(warmup):
        launch_fn()
    torch.cuda.synchronize()
    dist.barrier(group=group)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if cudagraph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(iters):
                launch_fn()
        torch.cuda.synchronize()
        dist.barrier(group=group)
        start.record()
        g.replay()
        end.record()
    else:
        start.record()
        for _ in range(iters):
            launch_fn()
        end.record()
    end.synchronize()
    local_us = start.elapsed_time(end) / iters * 1e3  # ms -> us

    world_size = dist.get_world_size(group=group)
    dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    t = torch.tensor([local_us], dtype=torch.float64, device=dev)
    outs = [torch.empty(1, dtype=torch.float64, device=dev) for _ in range(world_size)]
    dist.all_gather(outs, t, group=group)
    return torch.cat(outs).mean().item()


def bench_one(group, group_rank, R, S, K, E, H, bias_ratio, Hp,
              num_sms=32, warmup=5, iters=20,
              cudagraph: bool = True,
              source_pull_blocks: int = 0,
              source_pull_threads: int = 64):
    """Run one (R, S, K, E, H, Hp, bias_ratio) configuration on the given subgroup."""
    dev = torch.device(f"cuda:{torch.cuda.current_device()}")

    buffer = Buffer(
        S, H, K, E, R, num_sms=num_sms, group=group,
        enable_source_object=True,
        explicitly_destroy=True,
    )
    ctx = buffer._require_ctx()

    # Shared seed -> rank-shared expert popularity (the gate is global in
    # real training); rank -> independent per-token draws.
    topk, tpe = generate_topk_routing(S, K, E, R, bias_ratio, dev, 1234,
                                      rank=group_rank)

    # Global expert-load imbalance:
    # per-expert routed token count summed over ranks, max / mean.
    global_tpe = tpe.to(torch.int64)
    dist.all_reduce(global_tpe, group=group)
    max_load = int(global_tpe.max().item())
    mean_load = float(global_tpe.sum().item()) / E
    load_max_mean = max_load / mean_load if mean_load else 0.0

    hidden = torch.randn(S, H, dtype=torch.bfloat16, device=dev)
    weights = torch.rand(S, K, dtype=torch.float32, device=dev)
    output = torch.empty(S, H, dtype=torch.bfloat16, device=dev)
    topk_flat = topk.reshape(-1).contiguous()

    plan, cu_seqlens = allocate_planning_outputs(ctx)

    def _plan_call():
        launch_planning(ctx, topk_flat, tpe, cu_seqlens, plan)

    planning_us = time_gpu_op(_plan_call, warmup, iters, group, cudagraph=cudagraph)

    # Capture deduped `dst` plus src_info. The first forward dispatch below
    # materializes the plan-owned dedup structures (dup_groups / dup_loffs /
    # dup_counts), which then drive the dispatch epilogue / combine prologue.
    # The plan carries the full [R, B] experts_to_copy table needed by
    # prefetch/grad_reduce.
    launch_planning(ctx, topk_flat, tpe, cu_seqlens, plan)
    source_storage = buffer.source_storage()
    source_storage.local_hidden.copy_(hidden)
    source_storage.local_route_weights.copy_(weights)
    source_view, source_cu_seqlens, source_plan = \
        buffer.prepare_source_dispatch(
            source_storage,
            topk,
            tpe,
            inter_rank_sync=True,
        )
    # Use the exact plan that carries the source provenance for every
    # following comparison. The routing inputs are identical, so it is
    # semantically equal to the plan used by the planning timing above.
    plan = source_plan
    cu_seqlens = source_cu_seqlens
    dst = plan.dst
    experts_to_copy = plan.experts_to_copy
    NvS = int(ctx['NvS'])
    torch.cuda.synchronize()
    dist.barrier(group=group)

    # ---- dispatch_fwd: forward dispatch writes hidden + route weights
    # (representative rows only under dedup), zero-fills the padding rows
    # (zero warp) and materializes the dedup structures (builder warps).
    def _dispatch_fwd_call():
        launch_dispatch(ctx, hidden, weights, plan, build_dedup_map=True)

    dispatch_fwd_us = time_gpu_op(
        _dispatch_fwd_call, warmup, iters, group, cudagraph=cudagraph
    )

    # ---- publish_meta: prepared-object completion after a compatible
    # producer has already written representative hidden rows into final
    # owner placement. Reuse the payload populated above as untimed setup.
    # This deliberately rebuilds the same fresh-plan dedup structures as
    # dispatch_fwd, so the only omitted work is the hidden source read and
    # representative remote stores.
    def _publish_meta_call():
        launch_dispatch(
            ctx,
            None,
            weights,
            plan,
            write_payload=False,
            build_dedup_map=True,
        )

    publish_meta_us = time_gpu_op(
        _publish_meta_call, warmup, iters, group, cudagraph=cudagraph
    )

    def _publish_saved_meta_call():
        launch_dispatch(
            ctx,
            None,
            weights,
            plan,
            write_payload=False,
            build_dedup_map=False,
        )

    publish_saved_meta_us = time_gpu_op(
        _publish_saved_meta_call, warmup, iters, group, cudagraph=cudagraph
    )

    # ---- epilogue_fwd: in-place duplicate expansion on the NVL shard
    # (load primary once, store to every duplicate slot). Timed after
    # dispatch_fwd so it reads the freshly written NVL state, matching the
    # real dispatch -> epilogue order. The zero_copy=False boundary
    # ``tensor.copy_`` is a plain torch op, not a MoonEP kernel, and is
    # deliberately not benchmarked here.
    def _epilogue_fwd_call():
        launch_dispatch_epilogue(ctx, plan)

    epilogue_fwd_us = time_gpu_op(
        _epilogue_fwd_call, warmup, iters, group, cudagraph=cudagraph
    )

    # ---- source_pull: explicit owner-pull of one representative row per
    # source-token/consumer-rank. Each loaded vector fans directly from
    # registers through the planner-published duplicate chain; a compact
    # range kernel initializes padding. The result is ready for the ordinary
    # contiguous expert consumer without the dispatch dedup builder.
    def _source_pull_call():
        buffer.materialize_source_dispatch(
            source_view,
            plan,
            with_route_weights=True,
            zero_copy=True,
            num_blocks=source_pull_blocks,
            num_threads=source_pull_threads,
        )

    source_pull_us = time_gpu_op(
        _source_pull_call, warmup, iters, group, cudagraph=cudagraph
    )

    # ---- dispatch_bwd: hidden-only dispatch with the saved plan.
    grad_output = torch.randn(S, H, dtype=torch.bfloat16, device=dev)

    def _dispatch_bwd_call():
        launch_dispatch(ctx, grad_output, None, plan, build_dedup_map=False)

    dispatch_bwd_us = time_gpu_op(
        _dispatch_bwd_call, warmup, iters, group, cudagraph=cudagraph
    )

    # ---- epilogue_bwd: same in-place expansion timed after dispatch_bwd
    # (saved-plan backward state).
    def _epilogue_bwd_call():
        launch_dispatch_epilogue(ctx, plan)

    epilogue_bwd_us = time_gpu_op(
        _epilogue_bwd_call, warmup, iters, group, cudagraph=cudagraph
    )

    # ---- Combine-side paths. The combine input is staged into the NVL
    # shard once as untimed setup (zero_copy=True callers write it in place;
    # the zero_copy=False boundary copy is a plain torch op). The prologue
    # then reduces duplicate rows into their primary in place; repeated timed
    # iterations keep the exact same memory work regardless of the
    # accumulated values.
    hidden_nvsh = torch.randn(NvS, H, dtype=torch.bfloat16, device=dev)
    droute_weights_sk = torch.empty(S, K, dtype=torch.float32, device=dev)
    ctx['hidden_buf_local'].copy_(hidden_nvsh)
    torch.cuda.synchronize()
    dist.barrier(group=group)

    # ---- combine_prologue_fwd: in-place fp32 duplicate accumulation.
    def _combine_prologue_fwd_call():
        launch_combine_prologue(ctx, plan)

    combine_prologue_fwd_us = time_gpu_op(
        _combine_prologue_fwd_call, warmup, iters, group, cudagraph=cudagraph
    )

    # ---- combine_fwd: forward combine gathers/sums K rows back to [S, H].
    def _combine_fwd_call():
        launch_combine(ctx, output, dst)

    combine_fwd_us = time_gpu_op(
        _combine_fwd_call, warmup, iters, group, cudagraph=cudagraph
    )

    # ---- combine_prologue_bwd: same kernel on the backward staging state.
    def _combine_prologue_bwd_call():
        launch_combine_prologue(ctx, plan)

    combine_prologue_bwd_us = time_gpu_op(
        _combine_prologue_bwd_call, warmup, iters, group, cudagraph=cudagraph
    )

    # ---- combine_bwd: combine kernel with droute_weights_sk gather.
    def _combine_bwd_call():
        launch_combine(ctx, output, dst, output_sk=droute_weights_sk)

    combine_bwd_us = time_gpu_op(
        _combine_bwd_call, warmup, iters, group, cudagraph=cudagraph
    )

    # ---- prefetch / grad_reduce: remote expert weight movement driven by the
    # planner's experts_to_copy ([R, B] global expert ids, -1 = idle slot).
    B = int(ctx['B'])  # prefetch slots per rank, == epn by default
    epn = E // R
    etc_cpu = experts_to_copy.cpu()
    valid = etc_cpu >= 0
    # max_recv: worst rank's prefetched (received) expert count.
    max_recv = int(valid.sum(dim=1).max().item())
    # max_send: worst owner rank's slot count across all ranks — these slots'
    # grads flow back to (are pulled by) the owner in grad_reduce.
    if bool(valid.any()):
        owners = etc_cpu[valid].long() // epn
        max_send = int(torch.bincount(owners, minlength=R).max().item())
    else:
        max_send = 0

    # NVL-dist expert weight/grad pools: each rank physically owns epn experts;
    # remote rows are NVLink-mapped. Chunk dim0 is padded to VMM granularity,
    # so global expert e lives at padded id (e // epn) * padded_epn + e % epn.
    # bf16 padding also satisfies fp32 (rows are 2x the bytes).
    padded_epn = pad_dim0_for_alignment([epn, H, Hp], torch.bfloat16)
    E_pad = R * padded_epn
    Bp = pad_dim0_for_alignment([B, H, Hp], torch.float32)

    etc_pad = torch.full((R, Bp), -1, dtype=torch.int32, device=dev)
    remapped = ((experts_to_copy.long() // epn) * padded_epn
                + experts_to_copy.long() % epn).to(torch.int32)
    etc_pad[:, :B] = torch.where(experts_to_copy >= 0, remapped, experts_to_copy)

    weights_full = create_nvl_dist_tensor(
        [padded_epn, H, Hp], torch.bfloat16, group_rank, R, group=group)
    assert weights_full.shape[0] == E_pad
    weights_full[group_rank * padded_epn:(group_rank + 1) * padded_epn].normal_()
    prefetch_buf = torch.empty(Bp, H, Hp, dtype=torch.bfloat16, device=dev)

    grads_full = create_nvl_dist_tensor(
        [padded_epn, H, Hp], torch.float32, group_rank, R, group=group)
    grads_full[group_rank * padded_epn:(group_rank + 1) * padded_epn].zero_()
    reduce_full = create_nvl_dist_tensor(
        [Bp, H, Hp], torch.float32, group_rank, R, group=group)
    reduce_buffers = reduce_full.view(R, Bp, H, Hp)
    reduce_buffers[group_rank].normal_()
    torch.cuda.synchronize()
    dist.barrier(group=group)

    # prefetch has no inter-rank sync of its own; pair every launch with the
    # small inter-rank sync kernel so successive iterations stay serialized
    # across ranks (same role the built-in barrier plays for the other ops).
    def _prefetch_call():
        launch_prefetch(weights_full, prefetch_buf, etc_pad[group_rank],
                        num_sms=num_sms)
        launch_inter_rank_sync(ctx)

    prefetch_us = time_gpu_op(_prefetch_call, warmup, iters, group, cudagraph=cudagraph)

    def _grad_reduce_call():
        launch_grad_reduce(
            grads_full, reduce_buffers, etc_pad,
            rank=group_rank, num_sms=num_sms,
            meta_buf=ctx['meta_buf'],
            meta_stride=int(ctx['meta_chunk_padded']),
            barrier_off=int(ctx['BARRIER_OFF']),
            grid_sync_bar=ctx['grid_sync_bar'],
        )

    grad_reduce_us = time_gpu_op(_grad_reduce_call, warmup, iters, group, cudagraph=cudagraph)

    # Byte accounting follows the timed operator work above.
    dispatch_fwd_bytes = S * K * H * 2
    dispatch_bwd_bytes = S * K * H * 2
    combine_fwd_bytes = S * K * H * 2
    combine_bwd_bytes = S * K * H * 2

    # Local-op bandwidth accounting (route weights ignored: [NvS] fp32 is
    # negligible next to [NvS, H] bf16).
    n_entries = dst.numel()
    group_size = dist.get_world_size(group=group)
    dup_counts_cpu = plan.dup_counts.cpu()
    dup_group_count = int(dup_counts_cpu[0].item())
    dup_loff_count = int(dup_counts_cpu[1].item())
    zero_pad_rows = int(plan.zero_fill_ranges[:, 1].sum().item())
    stats_t = torch.tensor(
        [
            float((dst < 0).sum().item()) / n_entries,
            float(dup_group_count),
            float(dup_loff_count),
        ],
        dtype=torch.float64,
        device=dev,
    )
    stats_outs = [
        torch.empty(3, dtype=torch.float64, device=dev) for _ in range(group_size)
    ]
    dist.all_gather(stats_outs, stats_t, group=group)
    stats_mean = torch.stack(stats_outs).mean(dim=0)
    dedup_ratio = stats_mean[0].item()
    mean_groups = stats_mean[1].item()
    mean_dups = stats_mean[2].item()

    # epilogue: read each group's primary once, write every duplicate slot.
    epilogue_total_bytes = (mean_groups + mean_dups) * H * 2
    # prologue: read primary + duplicates, write primary back.
    combine_prologue_total_bytes = (2 * mean_groups + mean_dups) * H * 2

    epilogue_fwd_total_gbps = epilogue_total_bytes / epilogue_fwd_us * 1e6 / 1e9
    epilogue_bwd_total_gbps = epilogue_total_bytes / epilogue_bwd_us * 1e6 / 1e9
    combine_prologue_fwd_total_gbps = (
        combine_prologue_total_bytes / combine_prologue_fwd_us * 1e6 / 1e9
    )
    combine_prologue_bwd_total_gbps = (
        combine_prologue_total_bytes / combine_prologue_bwd_us * 1e6 / 1e9
    )

    # Bottleneck rank traffic: each op's worst rank moves
    # max(max_send, max_recv) expert rows over NVLink.
    mx_experts = max(max_send, max_recv)
    prefetch_bytes = mx_experts * H * Hp * 2     # bf16 weights
    grad_reduce_bytes = mx_experts * H * Hp * 4  # fp32 grads

    result = {
        'planning_us': planning_us,
        'dispatch_fwd_us': dispatch_fwd_us,
        'publish_meta_us': publish_meta_us,
        'publish_saved_meta_us': publish_saved_meta_us,
        'dispatch_payload_delta_us': dispatch_fwd_us - publish_meta_us,
        'dispatch_ready_us': dispatch_fwd_us + epilogue_fwd_us,
        'source_pull_us': source_pull_us,
        'source_pull_delta_us': (
            dispatch_fwd_us + epilogue_fwd_us - source_pull_us
        ),
        'dispatch_bwd_us': dispatch_bwd_us,
        'epilogue_fwd_us': epilogue_fwd_us,
        'epilogue_bwd_us': epilogue_bwd_us,
        'combine_prologue_fwd_us': combine_prologue_fwd_us,
        'combine_prologue_bwd_us': combine_prologue_bwd_us,
        'combine_fwd_us': combine_fwd_us,
        'combine_bwd_us': combine_bwd_us,
        'NvS': NvS,
        'dedup_ratio': dedup_ratio,
        'dup_group_count': dup_group_count,
        'dup_loff_count': dup_loff_count,
        'zero_pad_rows': zero_pad_rows,
        'prefetch_us': prefetch_us,
        'grad_reduce_us': grad_reduce_us,
        'max_recv': max_recv,
        'max_send': max_send,
        'load_max_mean': load_max_mean,
        'dispatch_fwd_GBps': dispatch_fwd_bytes / dispatch_fwd_us * 1e6 / 1e9,
        'dispatch_bwd_GBps': dispatch_bwd_bytes / dispatch_bwd_us * 1e6 / 1e9,
        'combine_fwd_GBps': combine_fwd_bytes / combine_fwd_us * 1e6 / 1e9,
        'combine_bwd_GBps': combine_bwd_bytes / combine_bwd_us * 1e6 / 1e9,
        'epilogue_fwd_total_GBps': epilogue_fwd_total_gbps,
        'epilogue_bwd_total_GBps': epilogue_bwd_total_gbps,
        'combine_prologue_fwd_total_GBps': combine_prologue_fwd_total_gbps,
        'combine_prologue_bwd_total_GBps': combine_prologue_bwd_total_gbps,
        'prefetch_GBps': prefetch_bytes / prefetch_us * 1e6 / 1e9,
        'grad_reduce_GBps': grad_reduce_bytes / grad_reduce_us * 1e6 / 1e9,
    }
    buffer.destroy()
    return result


def cleanup_ctx():
    gc.collect()
    torch.cuda.empty_cache()


# -------------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------------

EP_SIZES = [4, 8]


def build_configs():
    HIDDENS = [3584, 7168]
    TOPKS = [8, 16]
    SEQLENS = [8192]
    # bias_ratio sweep — sigma of the lognormal expert-logit distribution.
    # Three regimes: near-balanced, typical dropless-MoE skew, pathological.
    RATIOS = [0.1, 1.0, 5.0]
    # Total expert count is fixed; experts-per-rank (epn = E / ep)
    # shrinks as the EP group grows (e.g. 224 at ep=4, 112 at ep=8).
    TOTAL_EXPERTS = 896

    configs = []
    for ep in EP_SIZES:
        E = TOTAL_EXPERTS
        assert E % ep == 0, f"E ({E}) must be divisible by ep ({ep})"
        for H in HIDDENS:
            for K in TOPKS:
                for S in SEQLENS:
                    for ratio in RATIOS:
                        if K > E:
                            continue
                        configs.append(dict(
                            ep=ep, H=H, K=K, S=S, E=E,
                            unbalance_ratio=ratio,
                        ))
    return configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="Output CSV path (rank 0 only). Defaults to stdout-only.")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--num-sms", type=int, default=32)
    # Expert weight inner dim H' for prefetch/grad_reduce ([E, H, Hp] proxy
    # weights).
    ap.add_argument("--hp", type=int, default=3072,
                    help="Expert weight inner dim H' (multiple of 128).")
    # CUDA-graph timing is on by default, removing per-launch overhead;
    # --no-cudagraph falls back to the eager loop.
    ap.add_argument("--no-cudagraph", dest="cudagraph", action="store_false",
                    help="Disable CUDA-graph timing (eager launch loop instead).")
    ap.set_defaults(cudagraph=True)
    # Per-dimension filters. If set, restrict the sweep to matching configs.
    # Useful for profiling (ncu/nsys) a single point.
    ap.add_argument("--ep", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--unbalance-ratio", type=float, default=None)
    # Override total expert count E on the selected config(s). Useful for
    # profiling unusual (ep, E) combos that the default ep*14 sweep doesn't
    # produce; the planning CuTe DSL path must have a matching specialization.
    ap.add_argument("--experts", type=int, default=None)
    ap.add_argument(
        "--source-pull-blocks",
        type=int,
        default=0,
        help=(
            "Owner-pull materializer grid size. Zero uses the measured "
            "49152-block cap."
        ),
    )
    ap.add_argument(
        "--source-pull-threads",
        type=int,
        choices=(64, 128, 256, 512),
        default=64,
        help="Threads per owner-pull materializer block.",
    )
    args = ap.parse_args()
    assert args.hp % 128 == 0, f"--hp must be a multiple of 128, got {args.hp}"
    assert args.source_pull_blocks >= 0

    world_rank, world_size, local_rank = setup()

    if world_rank == 0:
        print(f"=== bench_comm starting: world_size={world_size} ===", flush=True)
        print(f"hostname={os.uname().nodename} local_rank={local_rank}", flush=True)

    # Pre-create one subgroup per ep_size we'll visit. Subgroups must be
    # created collectively by all ranks (those not in `ranks=` get NULL).
    ep_groups = {}
    for ep in EP_SIZES:
        if ep > world_size:
            continue
        g = dist.new_group(ranks=list(range(ep)))
        ep_groups[ep] = g

    configs = build_configs()
    configs = [c for c in configs if (
        (args.ep is None or c['ep'] == args.ep)
        and (args.hidden is None or c['H'] == args.hidden)
        and (args.topk is None or c['K'] == args.topk)
        and (args.unbalance_ratio is None or c['unbalance_ratio'] == args.unbalance_ratio)
    )]
    if args.experts is not None:
        for c in configs:
            c['E'] = args.experts
        configs = [c for c in configs if c['K'] <= c['E']]
    if world_rank == 0:
        print(f"Sweeping {len(configs)} configs...", flush=True)

    results = []
    csv_writer = None
    csv_file = None
    csv_header = [
        "ep", "H", "K", "S", "E", "Hp", "unbalance_ratio",
        "max_recv", "max_send", "load_max_mean",
        "planning_us",
        "dispatch_fwd_us", "publish_meta_us", "publish_saved_meta_us",
        "dispatch_payload_delta_us",
        "dispatch_ready_us", "source_pull_us", "source_pull_delta_us",
        "dispatch_bwd_us",
        "epilogue_fwd_us", "epilogue_bwd_us",
        "combine_prologue_fwd_us", "combine_prologue_bwd_us",
        "combine_fwd_us", "combine_bwd_us",
        "NvS", "dedup_ratio",
        "dup_group_count", "dup_loff_count", "zero_pad_rows",
        "prefetch_us", "grad_reduce_us",
        "dispatch_fwd_GBps", "dispatch_bwd_GBps",
        "combine_fwd_GBps", "combine_bwd_GBps",
        "epilogue_fwd_total_GBps", "epilogue_bwd_total_GBps",
        "combine_prologue_fwd_total_GBps", "combine_prologue_bwd_total_GBps",
        "prefetch_GBps", "grad_reduce_GBps",
    ]
    if world_rank == 0 and args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        csv_file = open(args.out, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(csv_header)

    if world_rank == 0:
        header = (f"{'ep':>3} {'H':>5} {'K':>3} {'S':>5} {'E':>4} {'Hp':>5} "
                  f"{'unb_r':>5} "
                  f"{'mx_rcv':>6} {'mx_snd':>6} {'mx/mean':>7} "
                  f"{'plan_us':>9} "
                  f"{'d_fwd':>9} {'pub_meta':>9} {'pub_saved':>9} "
                  f"{'pay_delta':>9} "
                  f"{'d_ready':>9} {'src_pull':>9} {'src_delta':>9} "
                  f"{'d_bwd':>9} "
                  f"{'e_fwd':>9} {'e_bwd':>9} "
                  f"{'cp_fwd':>9} {'cp_bwd':>9} "
                  f"{'c_fwd':>9} {'c_bwd':>9} "
                  f"{'NvS':>7} {'dedup%':>7} "
                  f"{'grps':>7} {'dups':>7} {'zpad':>7} "
                  f"{'pf':>9} {'gr':>9} "
                  f"{'DfGB/s':>8} {'DbGB/s':>8} "
                  f"{'CfGB/s':>8} {'CbGB/s':>8} "
                  f"{'EfTotGB':>8} {'EbTotGB':>8} "
                  f"{'CpFTot':>8} {'CpBTot':>8} "
                  f"{'PfGB/s':>8} {'GrGB/s':>8}")
        print(header, flush=True)
        print("-" * len(header), flush=True)

    for cfg in configs:
        ep = cfg['ep']
        if ep > world_size:
            continue
        group = ep_groups.get(ep)
        in_group = (world_rank < ep)

        res = None
        if in_group:
            try:
                group_rank = world_rank  # subgroup ranks 0..ep-1 == world ranks 0..ep-1
                res = bench_one(
                    group=group, group_rank=group_rank,
                    R=ep, S=cfg['S'], K=cfg['K'], E=cfg['E'], H=cfg['H'],
                    bias_ratio=cfg['unbalance_ratio'], Hp=args.hp,
                    num_sms=args.num_sms,
                    warmup=args.warmup, iters=args.iters,
                    cudagraph=args.cudagraph,
                    source_pull_blocks=args.source_pull_blocks,
                    source_pull_threads=args.source_pull_threads,
                )
            except Exception:
                if world_rank == 0:
                    traceback.print_exc()
            finally:
                cleanup_ctx()

        # Hard global sync so idle ranks don't race ahead and so failures
        # in one group don't leave the next config wedged.
        dist.barrier()

        if world_rank == 0:
            if res is not None:
                line = (f"{ep:>3} {cfg['H']:>5} {cfg['K']:>3} {cfg['S']:>5} {cfg['E']:>4} "
                        f"{args.hp:>5} "
                        f"{cfg['unbalance_ratio']:>5.2f} "
                        f"{res['max_recv']:>6d} "
                        f"{res['max_send']:>6d} "
                        f"{res['load_max_mean']:>7.3f} "
                        f"{fmt4(res['planning_us']):>9} "
                        f"{fmt4(res['dispatch_fwd_us']):>9} "
                        f"{fmt4(res['publish_meta_us']):>9} "
                        f"{fmt4(res['publish_saved_meta_us']):>9} "
                        f"{fmt4(res['dispatch_payload_delta_us']):>9} "
                        f"{fmt4(res['dispatch_ready_us']):>9} "
                        f"{fmt4(res['source_pull_us']):>9} "
                        f"{fmt4(res['source_pull_delta_us']):>9} "
                        f"{fmt4(res['dispatch_bwd_us']):>9} "
                        f"{fmt4(res['epilogue_fwd_us']):>9} "
                        f"{fmt4(res['epilogue_bwd_us']):>9} "
                        f"{fmt4(res['combine_prologue_fwd_us']):>9} "
                        f"{fmt4(res['combine_prologue_bwd_us']):>9} "
                        f"{fmt4(res['combine_fwd_us']):>9} "
                        f"{fmt4(res['combine_bwd_us']):>9} "
                        f"{res['NvS']:>7d} "
                        f"{res['dedup_ratio'] * 100:>6.2f}% "
                        f"{res['dup_group_count']:>7d} "
                        f"{res['dup_loff_count']:>7d} "
                        f"{res['zero_pad_rows']:>7d} "
                        f"{fmt4(res['prefetch_us']):>9} "
                        f"{fmt4(res['grad_reduce_us']):>9} "
                        f"{fmt4(res['dispatch_fwd_GBps']):>8} "
                        f"{fmt4(res['dispatch_bwd_GBps']):>8} "
                        f"{fmt4(res['combine_fwd_GBps']):>8} "
                        f"{fmt4(res['combine_bwd_GBps']):>8} "
                        f"{fmt4(res['epilogue_fwd_total_GBps']):>8} "
                        f"{fmt4(res['epilogue_bwd_total_GBps']):>8} "
                        f"{fmt4(res['combine_prologue_fwd_total_GBps']):>8} "
                        f"{fmt4(res['combine_prologue_bwd_total_GBps']):>8} "
                        f"{fmt4(res['prefetch_GBps']):>8} "
                        f"{fmt4(res['grad_reduce_GBps']):>8}")
                print(line, flush=True)
                if csv_writer:
                    csv_writer.writerow([
                        ep, cfg['H'], cfg['K'], cfg['S'], cfg['E'], args.hp,
                        f"{cfg['unbalance_ratio']:.2f}",
                        res['max_recv'],
                        res['max_send'],
                        f"{res['load_max_mean']:.3f}",
                        fmt4(res['planning_us']),
                        fmt4(res['dispatch_fwd_us']),
                        fmt4(res['publish_meta_us']),
                        fmt4(res['publish_saved_meta_us']),
                        fmt4(res['dispatch_payload_delta_us']),
                        fmt4(res['dispatch_ready_us']),
                        fmt4(res['source_pull_us']),
                        fmt4(res['source_pull_delta_us']),
                        fmt4(res['dispatch_bwd_us']),
                        fmt4(res['epilogue_fwd_us']),
                        fmt4(res['epilogue_bwd_us']),
                        fmt4(res['combine_prologue_fwd_us']),
                        fmt4(res['combine_prologue_bwd_us']),
                        fmt4(res['combine_fwd_us']),
                        fmt4(res['combine_bwd_us']),
                        res['NvS'],
                        f"{res['dedup_ratio']:.4f}",
                        res['dup_group_count'],
                        res['dup_loff_count'],
                        res['zero_pad_rows'],
                        fmt4(res['prefetch_us']),
                        fmt4(res['grad_reduce_us']),
                        fmt4(res['dispatch_fwd_GBps']),
                        fmt4(res['dispatch_bwd_GBps']),
                        fmt4(res['combine_fwd_GBps']),
                        fmt4(res['combine_bwd_GBps']),
                        fmt4(res['epilogue_fwd_total_GBps']),
                        fmt4(res['epilogue_bwd_total_GBps']),
                        fmt4(res['combine_prologue_fwd_total_GBps']),
                        fmt4(res['combine_prologue_bwd_total_GBps']),
                        fmt4(res['prefetch_GBps']),
                        fmt4(res['grad_reduce_GBps']),
                    ])
                    csv_file.flush()
            else:
                line = (f"{ep:>3} {cfg['H']:>5} {cfg['K']:>3} {cfg['S']:>5} {cfg['E']:>4} "
                        f"{args.hp:>5} "
                        f"{cfg['unbalance_ratio']:>5.2f} FAILED")
                print(line, flush=True)
                if csv_writer:
                    row = [
                        ep, cfg['H'], cfg['K'], cfg['S'], cfg['E'], args.hp,
                        f"{cfg['unbalance_ratio']:.2f}",
                        "", "", "", "", "", "", "", "", "", "", "", "", "", "",
                    ]
                    row.extend([""] * (len(csv_header) - len(row)))
                    csv_writer.writerow(row)
                    csv_file.flush()
            results.append((cfg, res))

    if world_rank == 0:
        print(f"\n=== bench_comm done: {len(results)} configs ===", flush=True)
        if csv_file:
            csv_file.close()
            print(f"Wrote CSV to {args.out}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
