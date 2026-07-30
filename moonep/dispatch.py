"""
MoonEP Dispatch — CuTe DSL implementation.

Warp-specialized G2S/S2G ring buffer with cp.async.bulk per-row TMA, expressed
entirely in the CUTLASS Python DSL. ``launch_dispatch`` writes representative
payload rows (``dst >= 0``) into the NVL buffer and scatters the per-topk
route weights, zero-fills the segment-padding rows of the local NVL shard
(warp 2, driven by ``plan.zero_fill_ranges``), and on fresh planning paths
materializes the plan-owned dedup structures (``dup_groups`` / ``dup_loffs``
/ ``dup_counts``) with builder warps; plan-reuse paths keep those tensors
untouched. The in-place duplicate expansion on the NVL shard lives in
``moonep/dispatch_epilogue.py`` (``launch_dispatch_epilogue``).
"""

import functools

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import BFloat16, Int32, Int64, Uint32
from cutlass.cute.runtime import make_ptr

from moonep._common import (
    cp_async_bulk_g2s,
    cp_async_bulk_s2g,
    cross_rank_barrier,
    cross_warp_sync,
    atom_add_relaxed_gpu_s32,
    atom_min_relaxed_gpu_s32,
    atom_or_relaxed_gpu_b32,
    popc_b32,
    ctz_b32,
    pdl_trigger_dependents,
)
from moonep.constants import (
    KIDX_BITS,
    DEDUP_BUILDER_WARPS,
)
from moonep.planning import MoonEPCommPlan, warp_inclusive_scan


# ============================================================================
# Dispatch kernel
# ============================================================================

class DispatchKernel:
    """Warp-specialized dispatch.

    Layout:
      - Warp 0 G2S producer / warp 1 S2G consumer (data path, single thread
        of each drives the TMA copies; pipeline depth auto-picked at JIT).
      - Warp 2 zero warp: per-expert segment-padding zeroing of the local NVL
        shard (both hidden rows and the matching int32 weight slots when
        ``with_weights`` is set), driven by ``plan.zero_fill_ranges``. Runs on
        reuse paths too.
      - Warps 3..: DEDUP_BUILDER_WARPS dedup builder warps, fresh-planning
        paths only (build_dedup_map=True). They materialize the plan-owned
        ``dup_groups`` / ``dup_loffs`` / ``dup_counts`` from the slot_info
        provenance published by planning.
      - Exit barrier is cross_rank_barrier: dispatch writes to NVL hidden_buf /
        weights, then publishes them through grid_sync plus system-scope
        release/acquire atomics before peer ranks consume the rows.

    Dedup contract: non-negative ``dst`` entries copy the payload row to NVL.
    Negative entries encode the same raw destination for duplicate top-k
    entries; they skip the payload copy but still scatter the per-topk weight.
    Duplicate rows are expanded in place on the NVL shard afterwards by
    ``launch_dispatch_epilogue``.
    """

    # 3 fixed warps (producer + consumer + zero) + DEDUP_BUILDER_WARPS
    # dedup builder warps on fresh-planning paths.
    num_threads = 96 + 32 * DEDUP_BUILDER_WARPS
    PRODUCER_WARP = 0
    CONSUMER_WARP = 1
    ZERO_WARP = 2
    DEDUP_BUILDER_WARP = 3

    def __init__(
        self,
        H: int,
        R: int,
        S: int,
        K: int,
        zero_groups: int,       # zero_fill_ranges entries = E + B
        NvS: int,
        NvS_padded: int,
        SRC_INFO_OFF: int,
        meta_stride: int,
        num_sms: int,
        with_weights: bool,
        write_payload: bool,
        build_dedup_map: bool,
        smem_budget: int,
        pdl_trigger: bool,
    ):
        self.H = H
        self.R = R
        self.S = S
        self.K = K
        self.zero_groups = zero_groups
        self.NvS = NvS
        self.NvS_padded = NvS_padded
        self.meta_stride = meta_stride
        self.SRC_INFO_OFF = SRC_INFO_OFF
        self.num_sms = num_sms
        self.with_weights = with_weights
        self.write_payload = write_payload
        self.pdl_trigger = pdl_trigger
        self.build_dedup_map = build_dedup_map
        self.num_threads = (
            96 + 32 * DEDUP_BUILDER_WARPS if build_dedup_map else 96
        )
        # Metadata-only publication has no G2S/S2G payload pipeline. Keep the
        # minimum allocation accepted by the shared kernel shape instead of
        # reserving the maximum stage depth for storage it never touches.
        self.stages = (
            self._pick_stages(H, smem_budget)
            if write_payload
            else 2
        )
        if self.stages == 0:
            raise RuntimeError(
                f"dispatch: H={H} too large for per-block smem budget "
                f"{smem_budget} B (need at least "
                f"{self._smem_bytes(H, 2)} B)"
            )

    @classmethod
    def _smem_bytes(cls, H: int, stages: int) -> int:
        # stage_smem (bf16, kStages deep) + 2 mbar i64 per stage (full+empty).
        # Pad stage_smem to 128 B for byte_alignment, mbar block
        # to 16 B, plus 256 B headroom for any cutlass-internal static smem.
        def _round_up(n: int, a: int) -> int:
            return (n + a - 1) // a * a
        return (
            _round_up(stages * H * 2, 128)
            + _round_up(H * 2, 128)
            + _round_up(stages * 2 * 8, 16)
            + 256
        )

    @classmethod
    def _pick_stages(cls, H: int, smem_budget: int) -> int:
        for s in (16, 14, 12, 10, 8, 6, 4, 2):
            if cls._smem_bytes(H, s) <= smem_budget:
                return s
        return 0

    # ------------------------------------------------------------------ host

    @cute.jit
    def __call__(
        self,
        hidden_sh_ptr: cute.Pointer,          # bf16 [S, H]
        hidden_buf_ptr: cute.Pointer,         # bf16 [R*NvS_padded, H]
        weights_ptr: cute.Pointer,            # int32 view of fp32 [S, K] (or placeholder)
        dst_ptr: cute.Pointer,                # int32 [N=S*K]
        meta_ptr: cute.Pointer,               # int32 [R*meta_stride]
        zero_fill_ranges_ptr: cute.Pointer,   # int32 [E+B, 2] (col0=pad_start, col1=n_pad)
        bar_ptr: cute.Pointer,                # int32 [1] grid barrier counter
        primary_packed_ptr: cute.Pointer,
        kmask_ptr: cute.Pointer,
        kidx_to_loff_ptr: cute.Pointer,
        dup_groups_ptr: cute.Pointer,
        dup_loffs_ptr: cute.Pointer,
        dup_counts_ptr: cute.Pointer,
        builder_bar_ptr: cute.Pointer,
        rank: Int32,
        weights_off: Int32,
        barrier_off: Int32,
        stream: cuda.CUstream,
    ):
        H = cutlass.const_expr(self.H)
        S = cutlass.const_expr(self.S)
        K = cutlass.const_expr(self.K)
        NvS = cutlass.const_expr(self.NvS)
        NvS_padded = cutlass.const_expr(self.NvS_padded)
        meta_stride = cutlass.const_expr(self.meta_stride)
        R = cutlass.const_expr(self.R)
        stages = cutlass.const_expr(self.stages)

        # 2-D row-major views with int64 stride. R*NvS_padded*H can exceed
        # 2^31 in production, so the row-offset arithmetic must be evaluated
        # as int64.
        out_rows = R * NvS_padded
        H64 = cutlass.Int64(H)
        gmem_src = cute.make_tensor(
            hidden_sh_ptr,
            cute.make_layout((S, H), stride=(H64, cutlass.Int64(1))),
        )
        gmem_dst = cute.make_tensor(
            hidden_buf_ptr,
            cute.make_layout((out_rows, H), stride=(H64, cutlass.Int64(1))),
        )

        dst_tensor = cute.make_tensor(
            dst_ptr, cute.make_layout((S * K,))
        )
        meta_tensor = cute.make_tensor(
            meta_ptr, cute.make_layout((R * meta_stride,))
        )
        # weights: int32 view of fp32 [S, K] when with_weights, else placeholder.
        if cutlass.const_expr(self.with_weights):
            w_tensor = cute.make_tensor(
                weights_ptr, cute.make_layout((S * K,))
            )
        else:
            w_tensor = cute.make_tensor(
                weights_ptr, cute.make_layout((1,))
            )
        # zero_fill_ranges: int32 [E+B, 2] — col 0 pad_start_loff, col 1 n_pad_rows.
        # Linearized as length 2*(E+B) so warp 2's per-group read pulls both
        # values via a single contiguous int2 load.
        zero_fill_ranges_tensor = cute.make_tensor(
            zero_fill_ranges_ptr,
            cute.make_layout((2 * cutlass.const_expr(self.zero_groups),)),
        )
        bar_tensor = cute.make_tensor(bar_ptr, cute.make_layout((1,)))

        primary_packed_tensor = cute.make_tensor(
            primary_packed_ptr, cute.make_layout((R * S,))
        )
        kmask_tensor = cute.make_tensor(
            kmask_ptr, cute.make_layout((R * S,))
        )
        kidx_to_loff_tensor = cute.make_tensor(
            kidx_to_loff_ptr, cute.make_layout((R * S * K,))
        )
        dup_groups_tensor = cute.make_tensor(
            dup_groups_ptr, cute.make_layout((NvS * 3,))
        )
        dup_loffs_tensor = cute.make_tensor(
            dup_loffs_ptr, cute.make_layout((NvS,))
        )
        dup_counts_tensor = cute.make_tensor(
            dup_counts_ptr, cute.make_layout((2,))
        )
        builder_bar_tensor = cute.make_tensor(
            builder_bar_ptr, cute.make_layout((1,))
        )

        smem_bytes = self._smem_bytes(H, stages)

        self.kernel(
            gmem_src,
            gmem_dst,
            dst_tensor,
            meta_tensor,
            w_tensor,
            zero_fill_ranges_tensor,
            bar_tensor,
            primary_packed_tensor,
            kmask_tensor,
            kidx_to_loff_tensor,
            dup_groups_tensor,
            dup_loffs_tensor,
            dup_counts_tensor,
            builder_bar_tensor,
            rank,
            weights_off,
            barrier_off,
        ).launch(
            grid=(self.num_sms, 1, 1),
            block=(self.num_threads, 1, 1),
            smem=smem_bytes,
            stream=stream,
            cooperative=True,
        )

    # ----------------------------------------------------------------- kernel

    @cute.kernel
    def kernel(
        self,
        gmem_src: cute.Tensor,
        gmem_dst: cute.Tensor,
        dst_tensor: cute.Tensor,
        meta_tensor: cute.Tensor,
        w_tensor: cute.Tensor,
        zero_fill_ranges_tensor: cute.Tensor,
        bar_tensor: cute.Tensor,
        primary_packed_tensor: cute.Tensor,
        kmask_tensor: cute.Tensor,
        kidx_to_loff_tensor: cute.Tensor,
        dup_groups_tensor: cute.Tensor,
        dup_loffs_tensor: cute.Tensor,
        dup_counts_tensor: cute.Tensor,
        builder_bar_tensor: cute.Tensor,
        rank: Int32,
        weights_off: Int32,
        barrier_off: Int32,
    ):
        R = cutlass.const_expr(self.R)
        H = cutlass.const_expr(self.H)
        S = cutlass.const_expr(self.S)
        K = cutlass.const_expr(self.K)
        zero_groups = cutlass.const_expr(self.zero_groups)
        NvS = cutlass.const_expr(self.NvS)
        NvS_padded = cutlass.const_expr(self.NvS_padded)
        meta_stride = cutlass.const_expr(self.meta_stride)
        stages = cutlass.const_expr(self.stages)
        num_sms = cutlass.const_expr(self.num_sms)
        num_threads = cutlass.const_expr(self.num_threads)
        num_ranks = cutlass.const_expr(self.R)
        H_BYTES = cutlass.const_expr(H * 2)
        # Offset of the per-rank src_info scratch written by planning.
        SRC_INFO_OFF = cutlass.const_expr(self.SRC_INFO_OFF)
        # src_info mirrors dst's rank-stride encoding:
        # src_rank * NvS + offv, with -1 for empty slots.
        NvS_BITS = cutlass.const_expr(32 - 1 - KIDX_BITS)
        NvS_MASK = cutlass.const_expr((1 << NvS_BITS) - 1)
        INT32_MAX = cutlass.const_expr(0x7FFFFFFF)

        bidx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        # ----- shared memory layout
        smem = utils.SmemAllocator()
        load_mbar = smem.allocate_array(Int64, num_elems=2 * stages)
        stage_smem = smem.allocate_tensor(
            BFloat16,
            cute.make_ordered_layout((H, stages), order=(0, 1)),
            byte_alignment=128,
        )
        zero_smem = smem.allocate_tensor(
            BFloat16,
            cute.make_layout((H,)),
            byte_alignment=128,
        )

        # Load pipeline (TMA G2S):
        #   Producer = warp 0 (1 effective thread, lane 0).
        #   Consumer = warp 1 (1 effective thread, lane 0).
        load_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=load_mbar,
            num_stages=stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=H * 2,
        )

        # ----- zero_smem one-shot init (reused by every padding s2g from
        # warp 2). All NUM_THREADS cooperatively zero H bf16 elements, then
        # a single fence publishes the cta-scoped smem writes to the
        # cp.async.bulk async proxy for the rest of the kernel.
        H_PER_THR_ZERO = cutlass.const_expr((H + num_threads - 1) // num_threads)
        for j in cutlass.range_constexpr(H_PER_THR_ZERO):
            idx = j * num_threads + tidx
            if idx < Int32(H):
                zero_smem[idx] = BFloat16(0)
        cute.arch.barrier()
        cute.arch.fence_view_async_shared()

        # ----- per-block token range
        tpb = (S + self.num_sms - 1) // self.num_sms
        s_beg = bidx * tpb
        s_end = cutlass.min(s_beg + tpb, S)
        n_tok = s_end - s_beg

        # ============================================
        # Warp 0 — G2S producer
        # ============================================
        if warp_idx == self.PRODUCER_WARP:
            if cutlass.const_expr(self.write_payload):
                load_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, stages
                )
                for li in cutlass.range(n_tok, unroll=1):
                    s = s_beg + li
                    load_pipe.producer_acquire(load_state)
                    # cp.async.bulk is single-thread; only lane 0 issues it.
                    if cute.arch.lane_idx() == 0:
                        g_row_int = (
                            gmem_src.iterator + cutlass.Int64(s) * cutlass.Int64(H)
                        ).toint()
                        s_row_int = (
                            stage_smem.iterator + load_state.index * H
                        ).toint()
                        mbar_int = load_pipe.producer_get_barrier(load_state).toint()
                        cp_async_bulk_g2s(
                            s_row_int.ir_value(),
                            g_row_int.ir_value(),
                            Int32(H_BYTES).ir_value(),
                            mbar_int.ir_value(),
                        )
                    load_state.advance()

        # ============================================
        # Warp 1 — S2G consumer (K stores per token + optional weight scatter)
        # ============================================
        elif warp_idx == self.CONSUMER_WARP:
            if cutlass.const_expr(self.write_payload):
                use_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, stages
                )
                rel_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, stages
                )
                for li in cutlass.range(n_tok, unroll=1):
                    s = s_beg + li
                    sK = s * K

                    load_pipe.consumer_wait(use_state)

                    # All K stores + commit + optional weight scatter run on lane 0;
                    # 32 lanes would over-decrement the bulk_group / re-scatter.
                    if cute.arch.lane_idx() == 0:
                        s_row_int = (
                            stage_smem.iterator + use_state.index * H
                        ).toint()
                        for k in cutlass.range(K, unroll=1):
                            dst_val = dst_tensor[sK + k]
                            # Negative dst keeps the raw destination for the
                            # weight scatter, but marks the payload as duplicate.
                            store_token = dst_val >= Int32(0)
                            raw_dst = dst_val
                            if not store_token:
                                raw_dst = - dst_val - Int32(1)
                            drank = raw_dst // NvS
                            loff = raw_dst % NvS

                            if store_token:
                                drow = drank * NvS_padded + loff
                                g_row_int = (
                                    gmem_dst.iterator
                                    + cutlass.Int64(drow) * cutlass.Int64(H)
                                ).toint()
                                cp_async_bulk_s2g(
                                    s_row_int.ir_value(),
                                    g_row_int.ir_value(),
                                    Int32(H_BYTES).ir_value(),
                                )

                            if cutlass.const_expr(self.with_weights):
                                wb = w_tensor[sK + k]
                                meta_tensor[
                                    drank * meta_stride + weights_off + loff
                                ] = wb
                        cute.arch.cp_async_bulk_commit_group()

                    use_state.advance()

                    # Throttle in-flight bulk_groups to <= STAGES-1.
                    if li >= Int32(stages - 1):
                        cute.arch.cp_async_bulk_wait_group(stages - 1)
                        load_pipe.consumer_release(rel_state)
                        rel_state.advance()

                # Drain the trailing in-flight stores. consumer_release for the
                # last (kStages-1) stages isn't strictly needed since the kernel
                # exits below, but we keep it symmetric so the empty mbars are
                # in a known state if this kernel is replayed.
                cute.arch.cp_async_bulk_wait_group(0)
            elif cutlass.const_expr(self.with_weights):
                # The external producer already wrote every representative
                # payload row into gmem_dst. Use all lanes of this warp to
                # scatter only the per-topk route metadata before publishing
                # the object. Negative dst values still decode to their unique
                # metadata slots.
                lane = cute.arch.lane_idx()
                n = S * K
                begin = bidx * Int32(32) + lane
                stride = Int32(self.num_sms * 32)
                for idx in cutlass.range(begin, n, stride):
                    dst_val = dst_tensor[idx]
                    raw_dst = dst_val
                    if dst_val < Int32(0):
                        raw_dst = -dst_val - Int32(1)
                    drank = raw_dst // NvS
                    loff = raw_dst % NvS
                    meta_tensor[
                        drank * meta_stride + weights_off + loff
                    ] = w_tensor[idx]

        # ============================================
        # Warp 2 — per-expert zero-fill loop (runs concurrently with
        # producer/consumer dispatch). For each expert e on this rank the
        # planning kernel wrote zero_fill_ranges[e] = (pad_start_loff, n_pad_rows)
        # for the segment-padding rows DeepGEMM will read but the dispatch
        # consumer does not write. Each CTA strides through E experts; for
        # each non-empty range it issues n_pad_rows back-to-back
        # cp.async.bulk_s2g zeros from zero_smem and, when with_weights is
        # set, a paired int32 zero store into the matching slot of meta_buf
        # (fp32 0.0 shares the all-zero bit pattern, so int32 0 suffices).
        #
        # Each row is written by at most one source (dispatch consumer OR
        # this warp), so the local rank's zero writes do not race with peer
        # NVL writes. cross_rank_barrier below uses grid_sync plus
        # system-scope release/acquire atomics to publish both kinds of writes
        # before peer ranks consume the rows.
        # ============================================
        if warp_idx == self.ZERO_WARP:
            issued = Int32(0)
            e = Int32(bidx)
            while e < Int32(zero_groups):
                pad_start = zero_fill_ranges_tensor[e * Int32(2) + Int32(0)]
                n_pad = zero_fill_ranges_tensor[e * Int32(2) + Int32(1)]
                if n_pad > Int32(0):
                    for j in cutlass.range(n_pad, unroll=1):
                        loff = pad_start + Int32(j)
                        local_row = Int32(rank) * Int32(NvS_padded) + loff
                        if cute.arch.lane_idx() == 0:
                            g_row_int = (
                                gmem_dst.iterator
                                + cutlass.Int64(local_row) * cutlass.Int64(H)
                            ).toint()
                            z_smem_int = zero_smem.iterator.toint()
                            cp_async_bulk_s2g(
                                z_smem_int.ir_value(),
                                g_row_int.ir_value(),
                                Int32(H_BYTES).ir_value(),
                            )
                            cute.arch.cp_async_bulk_commit_group()
                            if cutlass.const_expr(self.with_weights):
                                meta_tensor[
                                    Int32(rank) * meta_stride + weights_off + loff
                                ] = Int32(0)
                        if issued >= Int32(stages - 1):
                            cute.arch.cp_async_bulk_wait_group(stages - 1)
                        issued += Int32(1)
                e += Int32(self.num_sms)
            cute.arch.cp_async_bulk_wait_group(0)

        # ============================================
        # Warps 3.. — dedup structure builder (fresh planning only)
        # ============================================
        elif warp_idx >= self.DEDUP_BUILDER_WARP:
            if cutlass.const_expr(self.build_dedup_map):
                lane = cute.arch.lane_idx()
                # Global builder index: DEDUP_BUILDER_WARPS warps per CTA each
                # own one chunk, dividing the latency-bound scan wall time.
                n_builders = cutlass.const_expr(
                    self.num_sms * DEDUP_BUILDER_WARPS
                )
                gb = (
                    bidx * Int32(DEDUP_BUILDER_WARPS)
                    + (warp_idx - self.DEDUP_BUILDER_WARP)
                )
                is_leader = Int32(0)
                if gb == 0:
                    is_leader = Int32(1)
                # Hierarchical phase barrier: builder warps rendezvous inside
                # the CTA on a named barrier, then only the first builder warp
                # of each CTA joins the global release/acquire barrier. This
                # keeps the global participant count at num_sms regardless of
                # DEDUP_BUILDER_WARPS (a flat num_sms*W-warp global barrier
                # measurably ate the multi-warp speedup).
                # Use HW named barrier 8; id 0..7 are reserved by the
                # CUTLASS pipelines / __syncthreads.
                cta_bar = pipeline.NamedBarrier(8, 32 * DEDUP_BUILDER_WARPS)

                # Initialize the per-token primary-election scratch every
                # fresh dispatch (a tail-clear scheme was measured neutral and
                # rejected: callers typically reuse one Buffer across many
                # iterations and cannot guarantee the previous dispatch fully
                # completed). The leader also clears the dup_counts output
                # counters that the warp-aggregated atomicAdds below allocate
                # from; phase 1 publishes both.
                seg_pk = cute.ceil_div(R * S, n_builders)
                sbeg_pk = seg_pk * gb
                send_pk = cutlass.min(sbeg_pk + seg_pk, R * S)
                for base_pk in cutlass.range(sbeg_pk + lane, send_pk, Int32(32)):
                    primary_packed_tensor[base_pk] = INT32_MAX
                    kmask_tensor[base_pk] = Uint32(0)
                if gb == 0:
                    if lane < 2:
                        dup_counts_tensor[lane] = Int32(0)

                # Phase 1: init published.
                cta_bar.arrive_and_wait()
                if warp_idx == self.DEDUP_BUILDER_WARP:
                    cross_warp_sync(
                        builder_bar_tensor.iterator, num_sms, is_leader
                    )
                cta_bar.arrive_and_wait()

                # Pass 1: elect one primary local slot per (source rank, token)
                # and record the kidx -> loff mapping for duplicate expansion.
                seg_src = cute.ceil_div(NvS, n_builders)
                sbeg_src = gb * seg_src
                send_src = cutlass.min(sbeg_src + seg_src, NvS)
                for loff in cutlass.range(sbeg_src + lane, send_src, Int32(32)):
                    info = meta_tensor[rank * meta_stride + SRC_INFO_OFF + loff]
                    if info >= 0:
                        src_rank = info // NvS
                        offv = info - src_rank * NvS
                        token = offv // K
                        kidx = offv - token * K
                        key = src_rank * S + token
                        packed = (kidx << NvS_BITS) | loff
                        primary_packed_ptr = (
                            primary_packed_tensor.iterator + key
                        ).toint()
                        atom_min_relaxed_gpu_s32(primary_packed_ptr.ir_value(), packed)
                        kmask_ptr = (
                            kmask_tensor.iterator + key
                        ).toint()
                        atom_or_relaxed_gpu_b32(
                            kmask_ptr.ir_value(), Uint32(1) << kidx
                        )
                        kidx_to_loff_tensor[key * K + kidx] = loff

                # Phase 2: primary election published.
                cta_bar.arrive_and_wait()
                if warp_idx == self.DEDUP_BUILDER_WARP:
                    cross_warp_sync(
                        builder_bar_tensor.iterator, num_sms, is_leader
                    )
                cta_bar.arrive_and_wait()

                # Pass 2a (per-lane, straight-line): classify this chunk's
                # slots from the election results and tally each lane's
                # primary-with-duplicates group / duplicate totals in
                # registers. Keeping this pass free of warp-collective ops
                # lets the compiler pipeline the dependent gmem loads (info
                # -> election) across iterations; keep the classification in
                # sync with Pass 2b below.
                lane_grp_n = Int32(0)
                lane_dup_n = Int32(0)
                for loff in cutlass.range(sbeg_src + lane, send_src, Int32(32)):
                    info = meta_tensor[rank * meta_stride + SRC_INFO_OFF + loff]
                    if info >= 0:
                        src_rank = info // NvS
                        offv = info - src_rank * NvS
                        token = offv // K
                        key = src_rank * S + token
                        packed = primary_packed_tensor[key]
                        mask = Uint32(kmask_tensor[key])
                        primary_loff = packed & NvS_MASK
                        dup_count = popc_b32(mask) - 1

                        if loff == primary_loff:
                            if dup_count > Int32(0):
                                lane_grp_n += Int32(1)
                                lane_dup_n += dup_count

                # Per-lane exclusive offsets within this warp's output range,
                # then one warp-aggregated atomicAdd per output array reserves
                # the warp's compact prefix range (2 atomics per warp,
                # 2*num_sms*DEDUP_BUILDER_WARPS total — a per-record atomicAdd
                # scheme serialized and dominated dispatch latency in the
                # pieces-era builder).
                incl_grp = warp_inclusive_scan(lane_grp_n, lane)
                incl_dup = warp_inclusive_scan(lane_dup_n, lane)
                grp_total = cute.arch.shuffle_sync(incl_grp, Int32(31))
                dup_total = cute.arch.shuffle_sync(incl_dup, Int32(31))
                lane_grp_off = incl_grp - lane_grp_n
                lane_dup_off = incl_dup - lane_dup_n

                grp_base_w = Int32(0)
                dup_base_w = Int32(0)
                if lane == 0:
                    grp_count_ptr = (
                        dup_counts_tensor.iterator + 0
                    ).toint()
                    grp_base_w = atom_add_relaxed_gpu_s32(
                        grp_count_ptr.ir_value(), grp_total
                    )
                    dup_count_ptr = (
                        dup_counts_tensor.iterator + 1
                    ).toint()
                    dup_base_w = atom_add_relaxed_gpu_s32(
                        dup_count_ptr.ir_value(), dup_total
                    )
                grp_base_w = cute.arch.shuffle_sync(grp_base_w, Int32(0))
                dup_base_w = cute.arch.shuffle_sync(dup_base_w, Int32(0))

                # Pass 2b (per-lane, straight-line): re-classify and emit
                # dup_groups / dup_loffs at the reserved positions. The
                # compact-prefix order across warps follows atomicAdd arrival
                # order and is NOT stable run-to-run; consumers iterate by
                # index and tests must compare group sets, not element order.
                my_grp = grp_base_w + lane_grp_off
                my_dup = dup_base_w + lane_dup_off
                for loff in cutlass.range(sbeg_src + lane, send_src, Int32(32)):
                    info = meta_tensor[rank * meta_stride + SRC_INFO_OFF + loff]
                    if info >= 0:
                        src_rank = info // NvS
                        offv = info - src_rank * NvS
                        token = offv // K
                        key = src_rank * S + token
                        packed = primary_packed_tensor[key]
                        mask = Uint32(kmask_tensor[key])
                        primary_kidx = packed >> NvS_BITS
                        primary_loff = packed & NvS_MASK
                        dup_count = popc_b32(mask) - 1

                        if loff == primary_loff:
                            if dup_count > Int32(0):
                                dup_group_key = my_grp * 3
                                dup_groups_tensor[dup_group_key] = loff
                                dup_groups_tensor[dup_group_key + 1] = my_dup
                                dup_groups_tensor[dup_group_key + 2] = dup_count
                                key_row = key * K
                                dup_mask = mask & (~(Uint32(1) << primary_kidx))
                                pos = Int32(0)
                                while dup_mask != Uint32(0):
                                    cur_dup_kidx = ctz_b32(dup_mask)
                                    cur_dup_loff = kidx_to_loff_tensor[
                                        key_row + cur_dup_kidx
                                    ]
                                    dup_loffs_tensor[my_dup + pos] = cur_dup_loff
                                    pos += Int32(1)
                                    dup_mask = dup_mask & (dup_mask - Uint32(1))
                                my_grp += Int32(1)
                                my_dup += dup_count

        # ----- exit barrier: dispatch wrote NVL hidden_buf / weight slots.
        # cross_rank_barrier uses grid_sync plus system-scope release/acquire
        # atomics to publish the whole grid's writes to peer ranks before the
        # local epilogue reads those rows.
        cross_rank_barrier(
            meta_tensor, meta_stride, barrier_off, rank, num_ranks,
            bar_tensor.iterator, Int32(self.num_sms), num_threads, tidx,
        )
        if cutlass.const_expr(self.pdl_trigger):
            pdl_trigger_dependents(tidx)


# ============================================================================
# Host launcher with per-shape compile cache
# ============================================================================

@functools.lru_cache(maxsize=None)
def _max_smem_per_block_optin(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).shared_memory_per_block_optin


@functools.lru_cache(maxsize=None)
def _get_compiled(
    H: int,
    R: int,
    S: int,
    K: int,
    zero_groups: int,
    NvS: int,
    NvS_padded: int,
    meta_stride: int,
    SRC_INFO_OFF: int,
    num_sms: int,
    with_weights: bool,
    write_payload: bool,
    build_dedup_map: bool,
    device_index: int,
    pdl_trigger: bool,
):
    smem_budget = _max_smem_per_block_optin(device_index) - 1024
    kernel = DispatchKernel(
        H=H, R=R, S=S, K=K, zero_groups=zero_groups,
        NvS=NvS, NvS_padded=NvS_padded,
        SRC_INFO_OFF=SRC_INFO_OFF, meta_stride=meta_stride,
        num_sms=num_sms,
        with_weights=with_weights,
        write_payload=write_payload,
        build_dedup_map=build_dedup_map,
        smem_budget=smem_budget,
        pdl_trigger=pdl_trigger,
    )

    bf16_ptr = make_ptr(BFloat16, 0, cute.AddressSpace.gmem, assumed_align=16)
    i32_ptr = make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=16)
    u32_ptr = make_ptr(Uint32, 0, cute.AddressSpace.gmem, assumed_align=16)
    stream_arg = cuda.CUstream(0)

    return cute.compile(
        kernel,
        bf16_ptr,  # hidden_sh
        bf16_ptr,  # hidden_buf
        i32_ptr,   # weights (int32 view of fp32)
        i32_ptr,   # dst
        i32_ptr,   # meta
        i32_ptr,   # zero_fill_ranges
        i32_ptr,   # bar (grid barrier counter)
        i32_ptr,   # primary_packed
        u32_ptr,   # kmask
        i32_ptr,   # kidx_to_loff
        i32_ptr,   # dup_groups
        i32_ptr,   # dup_loffs
        i32_ptr,   # dup_counts
        i32_ptr,   # builder_bar
        Int32(0),  # rank
        Int32(0),  # weights_off
        Int32(0),  # barrier_off
        stream_arg,
    )


def _check_dedup_builder_bounds(ctx: dict) -> None:
    R = int(ctx['R'])
    S = int(ctx['S'])
    K = int(ctx['K'])
    NvS = int(ctx['NvS'])
    NvS_BITS = 32 - 1 - KIDX_BITS
    N = S * K
    int32_max = 2**31 - 1
    assert N <= NvS, (
        f"src_info NvS-stride encoding requires S*K <= NvS, got S*K={N}, NvS={NvS}"
    )
    assert R * NvS <= int32_max, (
        "src_info linear encoding requires R*NvS <= int32_max: "
        f"R={R}, NvS={NvS}, R*NvS={R * NvS}, int32_max={int32_max}"
    )
    assert K <= (1 << KIDX_BITS) - 1, (
        f"primary_packed encoding requires K <= "
        f"{(1 << KIDX_BITS) - 1}, got {K}"
    )
    assert NvS <= (1 << NvS_BITS) - 1, (
        f"primary_packed encoding requires NvS <= "
        f"{(1 << NvS_BITS) - 1}, got {NvS}"
    )
    assert K <= 32, f"kmask bitmask requires K <= 32, got {K}"


def _check_dispatch_plan(ctx: dict, hidden_sh: torch.Tensor, plan: MoonEPCommPlan) -> None:
    S = int(ctx['S'])
    K = int(ctx['K'])
    N = S * K
    R = int(ctx['R'])
    E = int(ctx['E'])
    B = int(ctx.get('B', 0))
    NvS = int(ctx['NvS'])
    dev = hidden_sh.device

    assert plan.N == N, f"plan.N must be S*K={N}, got {plan.N}"
    assert plan.R == R, f"plan.R must match ctx R={R}, got {plan.R}"
    assert plan.K == K, f"plan.K must match ctx K={K}, got {plan.K}"
    assert plan.NvS == NvS, f"plan.NvS must match ctx NvS={NvS}, got {plan.NvS}"

    def _check_tensor(t: torch.Tensor, name: str, shape: tuple[int, ...]) -> None:
        assert t.dtype == torch.int32 and t.is_contiguous(), \
            f"{name} must be contiguous int32"
        assert tuple(t.shape) == shape, \
            f"{name} must be shape {shape}, got {tuple(t.shape)}"
        assert t.device == dev, \
            f"{name} must be on {dev}, got {t.device}"

    _check_tensor(plan.dst, "dst", (N,))
    _check_tensor(plan.zero_fill_ranges, "zero_fill_ranges", (E + B, 2))
    _check_tensor(plan.dup_groups, "dup_groups", (NvS, 3))
    _check_tensor(plan.dup_loffs, "dup_loffs", (NvS,))
    _check_tensor(plan.dup_counts, "dup_counts", (2,))


def _check_dedup_builder_tensors(ctx: dict, dev: torch.device) -> None:
    R = int(ctx['R'])
    S = int(ctx['S'])
    K = int(ctx['K'])
    NvS = int(ctx['NvS'])

    def _check_scratch(name: str, numel: int) -> None:
        assert name in ctx, f"ctx missing {name}"
        t = ctx[name]
        assert t.dtype == torch.int32 and t.is_contiguous(), \
            f"{name} must be contiguous int32"
        assert t.numel() == numel, \
            f"{name} must have {numel} elements, got {t.numel()}"
        assert t.device == dev, \
            f"{name} must be on {dev}, got {t.device}"

    _check_scratch("primary_packed", R * S)
    _check_scratch("kmask", R * S)
    _check_scratch("kidx_to_loff", R * S * K)
    _check_scratch("builder_bar", 1)


def launch_dispatch(
    ctx: dict,
    hidden_sh,
    route_weights_sk,
    plan,
    *,
    write_payload: bool = True,
    build_dedup_map: bool = True,
    pdl_trigger: bool = False,
):
    """Launch the dispatch kernel.

    Args:
        hidden_sh: [S, H] bf16 source hidden states when
            ``write_payload=True``. Pass None when an external producer has
            already written every representative row into ``ctx['hidden_buf']``.
        route_weights_sk: [S, K] fp32 route weights, or None to skip the
            weights scatter (placeholder tensor is passed to satisfy the
            non-null pointer constraint; the kernel ignores it when
            with_weights=False).
        plan: communication plan carrying ``dst``, ``zero_fill_ranges`` and
            the plan-owned dedup structures. Non-negative ``dst`` entries
            encode ``dest_rank * NvS + local_offset`` and copy the payload.
            Negative entries encode ``-raw_dst - 1`` and only scatter the
            corresponding weight.
        build_dedup_map: true only immediately after fresh planning. Reuse and
            backward paths pass false so the saved dedup structures
            (``dup_groups`` / ``dup_loffs`` / ``dup_counts``) are not rebuilt
            from stale ``src_info`` scratch. The zero warp runs on both paths.
        write_payload: copy hidden payload rows from ``hidden_sh`` when true.
            When false, this launch only scatters route metadata, zero-fills
            padding, builds fresh dedup metadata when requested, and publishes
            the external producer's writes through the cross-rank barrier.

    The in-place duplicate expansion on the NVL shard is a separate kernel —
    call ``moonep.dispatch_epilogue.launch_dispatch_epilogue`` afterwards on the same
    stream.
    """
    assert isinstance(plan, MoonEPCommPlan)
    with_weights = route_weights_sk is not None
    if not with_weights:
        # Placeholder; kernel will not dereference.
        route_weights_sk = plan.dst

    H = int(ctx['H'])
    R = int(ctx['R'])
    S = int(ctx['S'])
    K = int(ctx['K'])
    E = int(ctx['E'])
    B = int(ctx.get('B', 0))

    assert isinstance(write_payload, bool), \
        f"write_payload must be bool, got {type(write_payload).__name__}"
    if write_payload:
        assert hidden_sh is not None, "hidden_sh is required when write_payload=True"
        assert hidden_sh.dtype == torch.bfloat16 and hidden_sh.is_contiguous(), \
            "hidden_sh must be contiguous bf16"
        assert hidden_sh.is_cuda, "hidden_sh must be a CUDA tensor"
        assert tuple(hidden_sh.shape) == (S, H), \
            f"hidden_sh must be shape [S={S}, H={H}], got {tuple(hidden_sh.shape)}"
        source_hidden = hidden_sh
    else:
        assert hidden_sh is None, \
            "hidden_sh must be None when write_payload=False"
        source_hidden = ctx['hidden_buf_local']
    _check_dispatch_plan(ctx, source_hidden, plan)
    assert ctx['hidden_buf'].dtype == torch.bfloat16 and ctx['hidden_buf'].is_contiguous()
    assert ctx['hidden_buf'].device == source_hidden.device
    assert ctx['meta_buf'].dtype == torch.int32 and ctx['meta_buf'].is_contiguous()
    assert ctx['meta_buf'].device == source_hidden.device
    assert ctx['grid_sync_bar'].dtype == torch.int32 and ctx['grid_sync_bar'].is_contiguous()
    assert ctx['grid_sync_bar'].numel() == 1
    assert ctx['grid_sync_bar'].device == source_hidden.device
    if with_weights:
        assert route_weights_sk.dtype == torch.float32 and route_weights_sk.is_contiguous(), \
            "route_weights_sk must be contiguous fp32"
        assert tuple(route_weights_sk.shape) == (S, K), \
            f"route_weights_sk must be shape [S={S}, K={K}], got {tuple(route_weights_sk.shape)}"
        assert route_weights_sk.device == source_hidden.device
    assert ctx['H'] % 8 == 0, "H must be multiple of 8 for 16-B bulk-copy alignment"
    if build_dedup_map:
        _check_dedup_builder_bounds(ctx)
        _check_dedup_builder_tensors(ctx, source_hidden.device)

    NvS = int(ctx['NvS'])
    NvS_padded = int(ctx['NvS_padded'])
    meta_stride = int(ctx['meta_chunk_padded'])
    SRC_INFO_OFF = int(ctx['SRC_INFO_OFF'])
    num_sms = int(ctx['num_sms'])
    device_index = source_hidden.device.index

    dispatch_compiled = _get_compiled(
        H, R, S, K, E + B, NvS, NvS_padded, meta_stride, SRC_INFO_OFF,
        num_sms, with_weights, write_payload, build_dedup_map, device_index,
        bool(pdl_trigger),
    )

    # int32 view onto fp32 weights so the kernel always sees an int32 ptr
    # (matches the C++ approach: a 4-byte gather, no fp32 arithmetic).
    w_int = (
        route_weights_sk.view(torch.int32) if with_weights else route_weights_sk
    )

    hsh_ptr = make_ptr(BFloat16, source_hidden.data_ptr(), cute.AddressSpace.gmem,
                       assumed_align=16)
    hbf_ptr = make_ptr(BFloat16, ctx['hidden_buf'].data_ptr(),
                       cute.AddressSpace.gmem, assumed_align=16)
    w_ptr = make_ptr(Int32, w_int.data_ptr(), cute.AddressSpace.gmem,
                     assumed_align=16)
    dst_ptr = make_ptr(Int32, plan.dst.data_ptr(), cute.AddressSpace.gmem,
                      assumed_align=16)
    meta_ptr = make_ptr(Int32, ctx['meta_buf'].data_ptr(),
                        cute.AddressSpace.gmem, assumed_align=16)
    zfr_ptr = make_ptr(Int32, plan.zero_fill_ranges.data_ptr(),
                       cute.AddressSpace.gmem, assumed_align=8)
    bar_ptr = make_ptr(Int32, ctx['grid_sync_bar'].data_ptr(),
                       cute.AddressSpace.gmem, assumed_align=16)

    # Builder scratch is only dereferenced on fresh-planning launches; the
    # reuse path passes a harmless placeholder to satisfy the pointer args.
    primary_packed = ctx['primary_packed'] if build_dedup_map else plan.dst
    kmask = ctx['kmask'] if build_dedup_map else plan.dst
    kidx_to_loff = ctx['kidx_to_loff'] if build_dedup_map else plan.dst
    builder_bar = ctx['builder_bar'] if build_dedup_map else plan.dst
    primary_packed_ptr = make_ptr(Int32, primary_packed.data_ptr(),
                                  cute.AddressSpace.gmem, assumed_align=16)
    kmask_ptr = make_ptr(Uint32, kmask.data_ptr(),
                         cute.AddressSpace.gmem, assumed_align=16)
    kidx_to_loff_ptr = make_ptr(Int32, kidx_to_loff.data_ptr(),
                                cute.AddressSpace.gmem, assumed_align=16)
    builder_bar_ptr = make_ptr(Int32, builder_bar.data_ptr(),
                               cute.AddressSpace.gmem, assumed_align=16)

    dup_groups_ptr = make_ptr(Int32, plan.dup_groups.data_ptr(),
                              cute.AddressSpace.gmem, assumed_align=16)
    dup_loffs_ptr = make_ptr(Int32, plan.dup_loffs.data_ptr(),
                             cute.AddressSpace.gmem, assumed_align=16)
    dup_counts_ptr = make_ptr(Int32, plan.dup_counts.data_ptr(),
                              cute.AddressSpace.gmem, assumed_align=8)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    dispatch_compiled(
        hsh_ptr,
        hbf_ptr,
        w_ptr,
        dst_ptr,
        meta_ptr,
        zfr_ptr,
        bar_ptr,
        primary_packed_ptr,
        kmask_ptr,
        kidx_to_loff_ptr,
        dup_groups_ptr,
        dup_loffs_ptr,
        dup_counts_ptr,
        builder_bar_ptr,
        Int32(int(ctx['rank'])),
        Int32(int(ctx['WEIGHTS_OFF'])),
        Int32(int(ctx['BARRIER_OFF'])),
        stream,
    )
