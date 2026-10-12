"""CuTe DSL implementation of the MoonEP planning kernel.

A single cooperative grid launch (num_sms blocks) runs Phase A/B/C/D in one
kernel and produces the canonical ``dst`` (with negative encoding for
duplicates), cu_seqlens, experts_to_copy, zero_fill_ranges and remote_stats.
Fresh dispatch then builds the plan-owned dedup structures from ``dst`` and
the ``src_info`` scratch; plan reuse reuses these structures directly.
Inter-block sync uses a software grid barrier (cooperative launch keeps all
blocks resident); cross-rank sync uses a system-scope atomic self-resetting
barrier on the NVLink meta_buf.
"""

import functools
from dataclasses import dataclass

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.runtime import make_ptr

from moonep._common import cp_async_bulk_g2s, cross_rank_barrier, grid_sync
from moonep.constants import KIDX_BITS


@dataclass(frozen=True, slots=True)
class MoonEPCommPlan:
    dst: torch.Tensor
    experts_to_copy: torch.Tensor
    zero_fill_ranges: torch.Tensor
    remote_stats: torch.Tensor
    N: int
    R: int
    E: int
    NvS: int
    K: int

    # Dedup structures written by the dispatch builder and consumed by the
    # dispatch epilogue / combine prologue. dup_counts = [n_groups, n_dup_loffs];
    # only the compact prefix of dup_groups/dup_loffs is valid, and the ordering
    # is decided by the builder's atomicAdd — not guaranteed to be stable.
    dup_groups: torch.Tensor
    dup_loffs: torch.Tensor
    dup_counts: torch.Tensor

    def __post_init__(self) -> None:
        N = int(self.N)
        R = int(self.R)
        E = int(self.E)
        epn = E // R
        NvS = int(self.NvS)
        assert self.dst.dtype == torch.int32 and self.dst.is_contiguous()
        assert self.dst.numel() == N
        assert self.experts_to_copy.dtype == torch.int32 and self.experts_to_copy.is_contiguous()
        assert tuple(self.experts_to_copy.shape) == (R, epn)
        assert self.remote_stats.dtype == torch.int32 and self.remote_stats.is_contiguous()
        assert tuple(self.remote_stats.shape) == (2,)
        assert self.zero_fill_ranges.dtype == torch.int32 and self.zero_fill_ranges.is_contiguous()
        assert tuple(self.zero_fill_ranges.shape) == (2 * epn, 2)
        assert self.dup_groups.dtype == torch.int32 and self.dup_groups.is_contiguous()
        assert tuple(self.dup_groups.shape) == (NvS, 3)
        assert self.dup_loffs.dtype == torch.int32 and self.dup_loffs.is_contiguous()
        assert tuple(self.dup_loffs.shape) == (NvS,)
        assert self.dup_counts.dtype == torch.int32 and self.dup_counts.is_contiguous()
        assert tuple(self.dup_counts.shape) == (2,)

    def clone(self) -> "MoonEPCommPlan":
        return type(self)(
            dst=self.dst.clone(),
            experts_to_copy=self.experts_to_copy.clone(),
            zero_fill_ranges=self.zero_fill_ranges.clone(),
            remote_stats=self.remote_stats.clone(),
            dup_groups=self.dup_groups.clone(),
            dup_loffs=self.dup_loffs.clone(),
            dup_counts=self.dup_counts.clone(),
            N=self.N,
            R=self.R,
            E=self.E,
            NvS=self.NvS,
            K=self.K,
        )


# ============================================================
# Compile-time constants
# ============================================================
BLOCK_SIZE_P2 = 2048
BLOCK_DIM_P2 = 512
ITEMS_PER_THREAD_P2 = BLOCK_SIZE_P2 // BLOCK_DIM_P2  # 4


def ceil_div(x, y):
    """Return ceil(x / y), for compile-time integer tiling."""
    return (x + y - 1) // y


def align_up(x, alignment):
    """Round x up to a multiple of alignment."""
    return ceil_div(x, alignment) * alignment


def ceil_pow2(x):
    """Return the smallest power of 2 >= x, for shared-memory/layout padding."""
    return 1 << max(x - 1, 0).bit_length()


def log2_r(R):
    """Return max(ceil(log2(R + 1)), 1), for fixed-trip-count rank binary search."""
    return max(R.bit_length(), 1)


# ============================================================
# Low-level inline-PTX helpers
# ============================================================


@dsl_user_op
def match_any_b32(val, *, loc=None, ip=None) -> Uint32:
    """__match_any_sync(0xffffffff, val) -> peer mask."""
    return Uint32(llvm.inline_asm(
        T.i32(), [Uint32(val).ir_value(loc=loc, ip=ip),
                  Uint32(0xFFFFFFFF).ir_value(loc=loc, ip=ip)],
        "match.any.sync.b32 $0, $1, $2;", "=r,r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def st_global_v4_s32(addr_i64, x, y, z, w, *, loc=None, ip=None) -> None:
    """st.global.v4.s32 [addr], {x,y,z,w}: one 128bit store, NVLink transactions /4."""
    llvm.inline_asm(
        None,
        [addr_i64,
         Int32(x).ir_value(loc=loc, ip=ip), Int32(y).ir_value(loc=loc, ip=ip),
         Int32(z).ir_value(loc=loc, ip=ip), Int32(w).ir_value(loc=loc, ip=ip)],
        "st.global.v4.s32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def multimem_st_v4(addr_i64, x, y, z, w, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [addr_i64,
         Uint32(x).ir_value(loc=loc, ip=ip), Uint32(y).ir_value(loc=loc, ip=ip),
         Uint32(z).ir_value(loc=loc, ip=ip), Uint32(w).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .u64 g;\n\t cvta.to.global.u64 g, $0;\n\t"
        "multimem.st.relaxed.sys.global.v4.f32 [g], {$1, $2, $3, $4};\n\t}",
        "l,r,r,r,r", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def multimem_st_b32(addr_i64, x, *, loc=None, ip=None) -> None:
    """One 32-bit multimem store: the scalar tail of the vectorized publish.

    Same target, scope and relaxed ordering as ``multimem_st_v4``; the value is
    passed through as raw bits, so int32 payloads are not reinterpreted.
    """
    llvm.inline_asm(
        None,
        [addr_i64, Uint32(x).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .u64 g;\n\t cvta.to.global.u64 g, $0;\n\t"
        "multimem.st.relaxed.sys.global.b32 [g], $1;\n\t}",
        "l,r", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@cute.jit
def warp_inclusive_scan(v, lane):
    offset = 1
    for scan_step in cutlass.range_constexpr(5):
        y = cute.arch.shuffle_sync(v, lane - offset)
        if lane >= offset:
            v += y
        offset <<= 1
    return v


@cute.jit
def warp_exclusive_scan_e(s_hist, E: cutlass.Constexpr, tid):
    CHUNK = cutlass.const_expr(ceil_div(E, 32))
    if tid < 32:
        lane = tid
        csum = 0
        for j in cutlass.range_constexpr(CHUNK):
            idx = lane * CHUNK + j
            v = 0
            if idx < E:
                v = s_hist[idx]; s_hist[idx] = csum
            csum += v
        x = warp_inclusive_scan(csum, lane)
        offset = x - csum
        for j in cutlass.range_constexpr(CHUNK):
            idx = lane * CHUNK + j
            if idx < E: s_hist[idx] += offset
    cute.arch.barrier()


@cute.jit
def elem_ptr(tensor, coord):
    return tensor.iterator + tensor.layout(coord)


# warp argmax/argmin: instead of packing value/idx into one int32, run two
# redux passes over value and tie-break idx separately. value takes max/min;
# on ties idx takes the min (sentinel s32_max) or the max (sentinel -1).
# The returned (best_val, best_idx) is already broadcast across the warp.
# idx is >= 0 by convention, so the sentinel never wins.
@cute.jit
def warp_argmax_min_idx(v, i):
    m = cute.arch.warp_redux_sync(v, "max")
    j = cute.arch.warp_redux_sync(i if v == m else 2147483647, "min")
    return m, j

@cute.jit
def warp_argmin_min_idx(v, i):
    m = cute.arch.warp_redux_sync(v, "min")
    j = cute.arch.warp_redux_sync(i if v == m else 2147483647, "min")
    return m, j

@cute.jit
def warp_argmin_max_idx(v, i):
    m = cute.arch.warp_redux_sync(v, "min")
    j = cute.arch.warp_redux_sync(i if v == m else -1, "max")
    return m, j

# Scan version: one warp strides over s[0:n] taking the extremum; n is
# arbitrary (no longer limited to 32/64).
@cute.jit
def warp_scan_argmax_min_idx(s, n, lane):
    bv = -2147483648; bi = 2147483647
    for k in cutlass.range(lane, n, 32):
        x = s[k]
        if x > bv: bv = x; bi = k
    return warp_argmax_min_idx(bv, bi)

@cute.jit
def warp_scan_argmin_min_idx(s, n, lane):
    bv = 2147483647; bi = 2147483647
    for k in cutlass.range(lane, n, 32):
        x = s[k]
        if x < bv: bv = x; bi = k
    return warp_argmin_min_idx(bv, bi)


# Pure-register scan version: one warp splits [0:N) into CHUNK=ceil(N/32)
# segments, lane holds reg[j]=s[lane+j*32]; the whole segment stays in
# registers (CHUNK constexpr, fully unrolled), never touching smem. N is
# arbitrary, bounded by register count rather than 32/64.
@cute.jit
def reg_scan_argmax_min_idx(reg, N: cutlass.Constexpr, lane):
    bv = -2147483648; bi = 2147483647
    for j in cutlass.range_constexpr(ceil_div(N, 32)):
        k = lane + j * 32
        if k < N and reg[j] > bv: bv = reg[j]; bi = k
    return warp_argmax_min_idx(bv, bi)

@cute.jit
def reg_scan_argmin_min_idx(reg, N: cutlass.Constexpr, lane):
    bv = 2147483647; bi = 2147483647
    for j in cutlass.range_constexpr(ceil_div(N, 32)):
        k = lane + j * 32
        if k < N and reg[j] < bv: bv = reg[j]; bi = k
    return warp_argmin_min_idx(bv, bi)


@cute.jit
def copy_v4_remote(dst, dst_off, src, n: cutlass.Constexpr,
                   pid, tid, nth, num_sms):
    # src[n] -> dst[dst_off:]: scalar head pads to 16B, int4 body does one
    # 128bit store (transactions /4), scalar tail.
    # dst_off may be arbitrarily aligned (dst.iterator is 16B aligned); no src
    # padding needed.
    head = (-dst_off) & 3                      # make dst_off+head ≡0 mod4 so v4 addresses are 16B aligned
    nv = (n - head) >> 2                       # number of vectorizable 4-tuples

    for h in cutlass.range(pid * nth + tid, head, num_sms * nth):
        dst[dst_off + h] = src[h]

    for i in cutlass.range(pid * nth + tid, nv, num_sms * nth):
        j = head + i * 4
        st_global_v4_s32((dst.iterator + (dst_off + j)).toint().ir_value(),
                         src[j], src[j + 1], src[j + 2], src[j + 3])

    for off in cutlass.range(head + nv * 4 + pid * nth + tid, n, num_sms * nth):
        dst[dst_off + off] = src[off]


@cute.jit
def _pd_cta_slice(total: cutlass.Constexpr, pid, num_sms, tile: cutlass.Constexpr):
    # Phase D outputs are small: slice contiguous segments per CTA to keep
    # gmem writeback coalesced; the copy_begin clamp only handles empty CTAs
    # whose pid falls past the valid segments, so copy_count never goes negative.
    per_cta = cute.round_up(cute.ceil_div(total, num_sms), tile)
    begin = pid * per_cta
    end = cutlass.min(begin + per_cta, total)
    copy_begin = cutlass.min(begin, total)
    copy_count = end - copy_begin
    return begin, end, copy_begin, copy_count


@cute.jit
def _pd_stage_bias(src_begin):
    # src_begin is in int32 units; its low 2 bits are the offset of the logical
    # slice inside the 16B envelope, used to realign the SMEM stage to the
    # logical start on Phase D writeback.
    return src_begin & 3


@cute.jit
def _pd_aligned_ints(src_begin, logical_count):
    # cp.async.bulk requires 16B-aligned src/dst and a size that is a multiple
    # of 16B. Copy from floor(src/16)*16 here; the extra ints only land in the
    # staging padding.
    aligned_ints = 0
    if logical_count > 0:
        aligned_ints = cute.round_up(_pd_stage_bias(src_begin) + logical_count, 4)
    return aligned_ints


@cute.jit
def _pd_issue_g2s(meta, smem_stage, src_begin, logical_count, mbar):
    if logical_count > 0:
        # Bulk-copy from the 16B-aligned gmem address into the 16B-aligned
        # SMEM stage. The extra head/tail ints stay in the stage padding and
        # are not part of the final writeback.
        src_aligned = src_begin - _pd_stage_bias(src_begin)
        aligned_ints = _pd_aligned_ints(src_begin, logical_count)
        cp_async_bulk_g2s(
            smem_stage.iterator.toint().ir_value(),
            (meta.iterator + src_aligned).toint().ir_value(),
            Int32(aligned_ints * 4).ir_value(),
            mbar.toint().ir_value(),
        )


class PlanningKernel:
    def __init__(self, R, E, S, K, NvS_capacity, NvS, num_vblocks, meta_stride,
                 TPE_OFF, PLAN_OFF, BARRIER_OFF, TOPK0_OFF, ORDER_OFF, ORDER0_OFF,
                 token_padding, num_sms):
        self.R, self.E, self.S, self.K = R, E, S, K
        self.N = self.S * self.K
        self.NvS_capacity, self.NvS, self.num_vblocks = NvS_capacity, NvS, num_vblocks
        self.meta_stride = meta_stride
        self.TPE_OFF, self.PLAN_OFF, self.BARRIER_OFF = TPE_OFF, PLAN_OFF, BARRIER_OFF
        self.TOPK0_OFF, self.ORDER_OFF, self.ORDER0_OFF = TOPK0_OFF, ORDER_OFF, ORDER0_OFF
        self.token_padding, self.num_sms = token_padding, num_sms

    @cute.jit
    def __call__(self, tpe, topk, meta, mc, dst, cu_seqlens,
                 experts_to_copy, zero_fill, remote_stats, alloc, group_tokens, z,
                 local_hist, bar,
                 rank: Int32, stream: cuda.CUstream):
        R = cutlass.const_expr(self.R)
        ms = cutlass.const_expr(self.meta_stride)
        N = cutlass.const_expr(self.N)
        epn = cutlass.const_expr(self.E // self.R)
        num_sms = cutlass.const_expr(self.num_sms)
        meta_t = cute.make_tensor(meta, cute.make_layout((R * ms,)))
        mc_t = cute.make_tensor(mc, cute.make_layout((R * ms,)))
        tpe_t = cute.make_tensor(tpe, cute.make_layout((self.E,)))
        topk_t = cute.make_tensor(topk, cute.make_layout((N,)))
        dst_t = cute.make_tensor(dst, cute.make_layout((N,)))
        cu_t = cute.make_tensor(cu_seqlens, cute.make_layout((2 * epn,)))
        etc_t = cute.make_tensor(experts_to_copy, cute.make_layout((self.R * epn,)))
        zfr_t = cute.make_tensor(zero_fill, cute.make_layout((4 * epn,)))
        stats_t = cute.make_tensor(remote_stats, cute.make_layout((2,)))
        alloc_t = cute.make_tensor(alloc, cute.make_layout((R * self.E,)))
        gt_t = cute.make_tensor(group_tokens, cute.make_layout((R,)))
        z_t = cute.make_tensor(z, cute.make_layout((R * R,)))
        lh_t = cute.make_tensor(local_hist, cute.make_layout((self.num_vblocks * self.E,)))
        bar_t = cute.make_tensor(bar, cute.make_layout((1,)))

        self.kernel(tpe_t, topk_t, meta_t, mc_t, dst_t, cu_t, etc_t,
                    zfr_t, stats_t, alloc_t, gt_t, z_t, lh_t, bar_t,
                    rank).launch(
            grid=(num_sms, 1, 1), block=(BLOCK_DIM_P2, 1, 1),
            stream=stream, cooperative=True)

    # =========================================================
    # run_c1: 1a histogram / 1b vblock prefix / expoff / passA scatter
    # =========================================================
    @cute.jit
    def run_c1(self, topk_src, order_dst, tpe_src, local_hist, s_hist, s_bp, s_wcount,
               bar_ptr, num_sms, pid, tid):
        R = cutlass.const_expr(self.R)
        E = cutlass.const_expr(self.E)
        NUM_WARPS = cutlass.const_expr(BLOCK_DIM_P2 // 32)
        WST = cutlass.const_expr(NUM_WARPS + 1)
        IPT = cutlass.const_expr(ITEMS_PER_THREAD_P2)
        N = cutlass.const_expr(self.N)
        num_vblocks = cutlass.const_expr(self.num_vblocks)
        num_threads = BLOCK_DIM_P2
        warp = tid >> 5
        lane = tid & 31

        topk_in = cute.make_tensor(topk_src.iterator, cute.make_layout((N,)))
        order_out = cute.make_tensor(order_dst.iterator, cute.make_layout((N,)))
        tpe_counts = cute.make_tensor(tpe_src.iterator, cute.make_layout((E,)))
        vblocks_histogram = cute.make_tensor(
            local_hist.iterator,
            cute.make_layout((num_vblocks, E), stride=(E, 1)),
        )
        s_histogram = cute.make_tensor(s_hist.iterator, cute.make_layout((E,)))
        s_block_prefix = cute.make_tensor(s_bp.iterator, cute.make_layout((E,)))
        s_warp_counts = cute.make_tensor(
            s_wcount.iterator,
            cute.make_layout((E + 1, WST), stride=(WST, 1)),
        )

        # 1a
        for vb in cutlass.range(pid, num_vblocks, num_sms):
            for e in cutlass.range(tid, E, num_threads):
                s_histogram[e] = 0

            cute.arch.barrier()

            chunk = vb * BLOCK_SIZE_P2
            for p in cutlass.range(tid, BLOCK_SIZE_P2, num_threads):
                off = chunk + p
                if off < N:
                    expert = topk_in[off]
                    cute.arch.atomic_add(elem_ptr(s_histogram, expert), 1, scope="cta")

            cute.arch.barrier()

            for e in cutlass.range(tid, E, num_threads):
                vblocks_histogram[vb, e] = s_histogram[e]
            cute.arch.barrier()

        grid_sync(bar_ptr, num_sms, tid)
        E_SEG = 32
        seg_raw = cute.ceil_div(E, num_sms)
        experts_per_block = cute.round_up(seg_raw, E_SEG)
        e_lo = pid * experts_per_block
        e_hi = cutlass.min(e_lo + experts_per_block, E)
        for e in cutlass.range(e_lo + tid, e_hi, num_threads):
            cumsum = 0
            for vb in cutlass.range_constexpr(num_vblocks):
                v = vblocks_histogram[vb, e]
                vblocks_histogram[vb, e] = cumsum
                cumsum += v

        grid_sync(bar_ptr, num_sms, tid)
        for e in cutlass.range(tid, E, num_threads):
            s_histogram[e] = tpe_counts[e]

        cute.arch.barrier()
        warp_exclusive_scan_e(s_histogram, E, tid)
        lanes_lt = (Uint32(1) << lane) - Uint32(1)
        for vb in cutlass.range(pid, num_vblocks, num_sms):
            chunk = vb * BLOCK_SIZE_P2
            my_e = []; my_p = []
            for i in cutlass.range_constexpr(IPT):
                p = warp * (32 * IPT) + i * 32 + lane
                off = chunk + p
                my_p.append(p)
                ev = E
                if off < N:
                    ev = topk_in[off]
                my_e.append(ev)

            for idx in cutlass.range(tid, (E + 1) * WST, num_threads):
                expert_idx = idx // WST
                warp_slot = idx - expert_idx * WST
                s_warp_counts[expert_idx, warp_slot] = 0

            cute.arch.barrier()

            for e in cutlass.range(tid, E, num_threads):
                s_block_prefix[e] = vblocks_histogram[vb, e]

            ww = []
            for i in cutlass.range_constexpr(IPT):
                peers = match_any_b32(my_e[i])
                cell = elem_ptr(s_warp_counts, (my_e[i], warp))
                base = cute.arch.load(cell, Int32)
                ww.append(base + Int32(cute.arch.popc(peers & lanes_lt)))
                cute.arch.sync_warp()
                if (peers & lanes_lt) == Uint32(0):
                    cute.arch.store(cell, base + Int32(cute.arch.popc(peers)))
                cute.arch.sync_warp()
            cute.arch.barrier()

            for e in cutlass.range(tid, E, num_threads):
                cumsum = 0
                for w in cutlass.range_constexpr(NUM_WARPS):
                    c = s_warp_counts[e, w]
                    s_warp_counts[e, w] = cumsum
                    cumsum += c

            cute.arch.barrier()

            for i in cutlass.range_constexpr(IPT):
                ei = my_e[i]
                if ei < E:
                    within = s_warp_counts[ei, warp] + ww[i]
                    sp = s_histogram[ei] + s_block_prefix[ei] + within
                    order_out[sp] = chunk + my_p[i]
            cute.arch.barrier()

        grid_sync(bar_ptr, num_sms, tid)

    @cute.kernel
    def kernel(self, tpe, topk, meta, mc, dst, cu_seqlens,
               experts_to_copy, zfr, remote_stats, alloc, group_tokens, z, lh, bar,
               rank: Int32):
        R = cutlass.const_expr(self.R)
        E = cutlass.const_expr(self.E)
        S = cutlass.const_expr(self.S)
        K = cutlass.const_expr(self.K)
        epn = cutlass.const_expr(E // R)
        LOG2_R = cutlass.const_expr(log2_r(R))
        EB_PAD = cutlass.const_expr(ceil_pow2(2 * epn))
        IPT_EB = cutlass.const_expr(ceil_div(EB_PAD, BLOCK_DIM_P2))
        ms = cutlass.const_expr(self.meta_stride)
        N = cutlass.const_expr(self.N)
        NvS = cutlass.const_expr(self.NvS)
        CAP = cutlass.const_expr(self.NvS_capacity)
        tp = cutlass.const_expr(self.token_padding)
        num_sms = cutlass.const_expr(self.num_sms)
        TPE_OFF = cutlass.const_expr(self.TPE_OFF)
        PLAN_OFF = cutlass.const_expr(self.PLAN_OFF)
        BARRIER_OFF = cutlass.const_expr(self.BARRIER_OFF)
        TOPK0_OFF = cutlass.const_expr(self.TOPK0_OFF)
        ORDER_OFF = cutlass.const_expr(self.ORDER_OFF)
        ORDER0_OFF = cutlass.const_expr(self.ORDER0_OFF)
        BARRIER_SLOTS = 3
        SRC_INFO_OFF = cutlass.const_expr(self.BARRIER_OFF + BARRIER_SLOTS)
        ALLOC_SUB = 0
        TPE_SUB = E * R
        EOFF_SUB = 2 * E * R
        CU_SUB = 3 * E * R
        ZFR_SUB = CU_SUB + R * (2 * epn)
        ETC_SUB = ZFR_SUB + 2 * R * (2 * epn)
        STATS_SUB = ETC_SUB + R * epn
        PB = PLAN_OFF
        num_threads = BLOCK_DIM_P2
        NUM_WARPS = cutlass.const_expr(BLOCK_DIM_P2 // 32)
        S1_TILE = 32
        S1_COLS = cutlass.const_expr(min(
            align_up((E + num_sms - 1) // num_sms, S1_TILE),
            BLOCK_DIM_P2,
        ))
        pid = cute.arch.block_idx()[0]
        tid = cute.arch.thread_idx()[0]

        smem = utils.SmemAllocator()
        def sa(n):
            align_elems = 16
            aligned_n = align_up(n, align_elems)
            return smem.allocate_tensor(Int32, cute.make_layout((aligned_n,)), byte_alignment=16)
        PHASE_D_TILE = 32
        PHASE_D_GROUPS_PER_CTA = cutlass.const_expr(
            align_up((2 * epn + num_sms - 1) // num_sms, PHASE_D_TILE)
        )
        PHASE_D_ETC_PER_CTA = cutlass.const_expr(
            align_up((R * epn + num_sms - 1) // num_sms, PHASE_D_TILE)
        )
        # Each CTA only stages the Phase D slice it owns. Each segment gets an
        # extra +4 ints to hold the head/tail elements the 16B-aligned envelope
        # may pull in.
        PD_CU_OFF = 0
        PD_CU_LEN = cutlass.const_expr(align_up(PHASE_D_GROUPS_PER_CTA + 4, 4))
        PD_ZFR_OFF = cutlass.const_expr(PD_CU_OFF + PD_CU_LEN)
        PD_ZFR_LEN = cutlass.const_expr(align_up(2 * PHASE_D_GROUPS_PER_CTA + 4, 4))
        PD_ETC_OFF = cutlass.const_expr(PD_ZFR_OFF + PD_ZFR_LEN)
        PD_ETC_LEN = cutlass.const_expr(align_up(PHASE_D_ETC_PER_CTA + 4, 4))
        PD_SCRATCH_INTS = cutlass.const_expr(PD_ETC_OFF + PD_ETC_LEN)
        scratch_ints = cutlass.const_expr(max(
            R * S1_COLS,
            E + E // R + R,
            (E + 1) * (BLOCK_DIM_P2 // 32 + 1),
            PD_SCRATCH_INTS,
        ))
        scratch = sa(scratch_ints)
        pd_mbar = smem.allocate_array(Int64, num_elems=1)
        s_hist = sa(E)
        s_bp = sa(E)
        s_col = sa(E)
        s_chosen = sa(epn)
        s_wmax = sa(64)
        bar_p = bar.iterator
        # Phase A
        # tpe gather -> rank0 chunk (helper handles head/tail alignment itself)
        copy_v4_remote(meta, TPE_OFF + rank * E, tpe, E, pid, tid, num_threads, num_sms)
        if cutlass.const_expr(R > 1):
            if rank == 0:
                # The TOPK0/TPE region of alloc is guaranteed 16B aligned: topk/tpe push goes v4 (tail padded internally).
                copy_v4_remote(meta, ms + TOPK0_OFF, topk, N, pid, tid, num_threads, num_sms)
                copy_v4_remote(meta, ms + TPE_OFF, tpe, E, pid, tid, num_threads, num_sms)
        cross_rank_barrier(meta, ms, BARRIER_OFF, rank, R, bar_p, num_sms, num_threads, tid)
        if rank == 0:
            z_tensor = cute.make_tensor(
                z.iterator,
                cute.make_layout((R, R), stride=(R, 1)),
            )
            for i in cutlass.range(
                pid * num_threads + tid, R * R, num_sms * num_threads
            ):
                z[i] = 0
            for i in cutlass.range(
                pid * num_threads + tid, R, num_sms * num_threads
            ):
                group_tokens[i] = 0
            grid_sync(bar_p, num_sms, tid)
            # Split the expert dimension across blocks; S1_TILE alignment keeps
            # each round processing a fixed number of columns.
            seg_raw = cute.ceil_div(E, num_sms)
            experts_per_block = cute.round_up(seg_raw, S1_TILE)
            start_idx = pid * experts_per_block; end_idx = cutlass.min(start_idx + experts_per_block, E)
            tpe_gather = cute.make_tensor(
                meta.iterator + TPE_OFF,
                cute.make_layout((R, E), stride=(E, 1)),
            )
            tpe_cumsum = cute.make_tensor(
                meta.iterator + (PB + TPE_SUB),
                cute.make_layout((R, E), stride=(E, 1)),
            )
            s_tpe = cute.make_tensor(
                scratch.iterator,
                cute.make_layout((R, S1_COLS), stride=(S1_COLS, 1)),
            )
            for e0 in cutlass.range(start_idx, end_idx, S1_COLS):
                # All threads copy the current expert tile from gmem into s_tpe.
                for idx in cutlass.range(tid, R * S1_COLS, num_threads):
                    r = idx // S1_COLS; col = idx - r * S1_COLS; expert_idx = e0 + col
                    v = 0
                    if expert_idx < end_idx:
                        v = tpe_gather[r, expert_idx]
                    s_tpe[r, col] = v

                cute.arch.barrier()

                if tid < S1_COLS:
                    expert_idx = e0 + tid
                    if expert_idx < end_idx:
                        # Rank prefix sum within the column; the final run accumulates
                        # the total tokens of the target rank group.
                        run = 0
                        for r in cutlass.range_constexpr(R):
                            run += s_tpe[r, tid]; s_tpe[r, tid] = run
                        cute.arch.atomic_add(group_tokens.iterator + expert_idx // epn, run, scope="gpu")
                cute.arch.barrier()

                # Write the prefix sums back to meta; step3 queries them by rank/expert.
                for idx in cutlass.range(tid, R * S1_COLS, num_threads):
                    r = idx // S1_COLS; col = idx - r * S1_COLS; expert_idx = e0 + col
                    if expert_idx < end_idx:
                        tpe_cumsum[r, expert_idx] = s_tpe[r, col]

                cute.arch.barrier()
            grid_sync(bar_p, num_sms, tid)
            if pid == 0:
                if tid < 32:
                    lane = tid
                    # balance stays in registers throughout: lane holds
                    # bal[j]=group_tokens[lane+j*32]-CAP, CHUNK=ceil(R/32).
                    CHUNK = cutlass.const_expr(ceil_div(R, 32))
                    balance = cute.make_rmem_tensor(CHUNK, Int32)
                    for j in cutlass.range_constexpr(CHUNK):
                        k = lane + j * 32
                        balance[j] = 0
                        if k < R: balance[j] = group_tokens[k] - CAP
                    keep_balancing = True
                    while keep_balancing:
                        # surplus takes max (larger balance first, smaller rank
                        # on ties); deficit takes min (larger shortfall first,
                        # smaller rank on ties).
                        surplus, surplus_rank = reg_scan_argmax_min_idx(balance, R, lane)
                        deficit, deficit_rank = reg_scan_argmin_min_idx(balance, R, lane)
                        if surplus <= 0 or deficit >= 0:
                            keep_balancing = False
                        else:
                            # The move amount is limited by the receiver's
                            # shortfall; refill deficit_rank back to CAP in one shot.
                            move_tokens = -deficit
                            for j in cutlass.range_constexpr(CHUNK):
                                k = lane + j * 32
                                if k == surplus_rank: balance[j] -= move_tokens
                                elif k == deficit_rank: balance[j] = 0
                            if lane == 0:
                                z_tensor[surplus_rank, deficit_rank] = move_tokens
                            cute.arch.sync_warp()
            grid_sync(bar_p, num_sms, tid)
            # alloc_by_rank[rank, expert] feeds step4; alloc_prefix[expert, rank]
            # stores the cumulative counts consumed by C2 binary search.
            alloc_cumsum = cute.make_tensor(
                meta.iterator + (PB + ALLOC_SUB),
                cute.make_layout((E, R), stride=(R, 1)),
            )
            alloc_tensor = cute.make_tensor(
                alloc.iterator,
                cute.make_layout((R, E), stride=(E, 1)),
            )
            s_alloc = cute.make_tensor(
                scratch.iterator,
                cute.make_layout((R, epn), stride=(epn, 1)),
            )
            for owner_rank in cutlass.range(pid, R, num_sms):
                expert_base = owner_rank * epn

                for idx in cutlass.range(tid, epn * R, num_threads):
                    local_expert_id = idx // R
                    rank_idx = idx - local_expert_id * R
                    global_expert = expert_base + local_expert_id
                    s_alloc[rank_idx, local_expert_id] = (
                        tpe_cumsum[R - 1, global_expert] if rank_idx == owner_rank else 0
                    )

                cute.arch.barrier()

                if tid < 32:
                    lane = tid
                    R_CHUNK = cutlass.const_expr(ceil_div(R, 32))
                    EPN_CHUNK = cutlass.const_expr(ceil_div(epn, 32))
                    quotas = cute.make_rmem_tensor(R_CHUNK, Int32)
                    owner_remaining = cute.make_rmem_tensor(EPN_CHUNK, Int32)
                    for j in cutlass.range_constexpr(R_CHUNK):
                        rank_idx = lane + j * 32
                        quotas[j] = 0
                        if rank_idx < R: quotas[j] = z_tensor[owner_rank, rank_idx]
                    for j in cutlass.range_constexpr(EPN_CHUNK):
                        local_expert_id = lane + j * 32
                        owner_remaining[j] = 0
                        if local_expert_id < epn:
                            owner_remaining[j] = s_alloc[owner_rank, local_expert_id]

                    keep_balancing = cutlass.Boolean(True)
                    while keep_balancing:
                        max_quota, target_rank = reg_scan_argmax_min_idx(quotas, R, lane)
                        if max_quota <= 0:
                            keep_balancing = cutlass.Boolean(False)
                        else:
                            max_remaining, selected_expert_id = reg_scan_argmax_min_idx(
                                owner_remaining, epn, lane)
                            if max_remaining <= 0:
                                keep_balancing = cutlass.Boolean(False)
                            else:
                                take = cutlass.min(max_remaining, max_quota)
                                for j in cutlass.range_constexpr(R_CHUNK):
                                    rank_idx = lane + j * 32
                                    if rank_idx == target_rank: quotas[j] = max_quota - take
                                for j in cutlass.range_constexpr(EPN_CHUNK):
                                    local_expert_id = lane + j * 32
                                    if local_expert_id == selected_expert_id:
                                        owner_remaining[j] = max_remaining - take
                                if tid == 0:
                                    s_alloc[target_rank, selected_expert_id] += take
                                    s_alloc[owner_rank, selected_expert_id] = max_remaining - take
                                cute.arch.sync_warp()
                cute.arch.barrier()

                for idx in cutlass.range(tid, epn * R, num_threads):
                    rank_idx = idx // epn
                    local_expert_id = idx - rank_idx * epn
                    global_expert = expert_base + local_expert_id
                    alloc_tensor[rank_idx, global_expert] = (
                        s_alloc[rank_idx, local_expert_id]
                    )

                cute.arch.barrier()

                for local_expert_id in cutlass.range(tid, epn, num_threads):
                    cum = 0
                    for rank_idx in cutlass.range_constexpr(R):
                        cum += s_alloc[rank_idx, local_expert_id]
                        s_alloc[rank_idx, local_expert_id] = cum

                cute.arch.barrier()

                for idx in cutlass.range(tid, epn * R, num_threads):
                    local_expert_id = idx // R
                    rank_idx = idx - local_expert_id * R
                    global_expert = expert_base + local_expert_id
                    alloc_cumsum[global_expert, rank_idx] = (
                        s_alloc[rank_idx, local_expert_id]
                    )

                cute.arch.barrier()
            grid_sync(bar_p, num_sms, tid)
            expert_offsets = cute.make_tensor(
                meta.iterator + (PB + EOFF_SUB),
                cute.make_layout((R, E), stride=(E, 1)),
            )
            all_cu_seqlens = cute.make_tensor(
                meta.iterator + (PB + CU_SUB),
                cute.make_layout((R, 2 * epn), stride=(2 * epn, 1)),
            )
            zero_fill_start = cute.make_tensor(
                meta.iterator + (PB + ZFR_SUB),
                cute.make_layout((R, 2 * epn), stride=(4 * epn, 2)),
            )
            zero_fill_count = cute.make_tensor(
                meta.iterator + (PB + ZFR_SUB + 1),
                cute.make_layout((R, 2 * epn), stride=(4 * epn, 2)),
            )
            all_experts_to_copy = cute.make_tensor(
                meta.iterator + (PB + ETC_SUB),
                cute.make_layout((R, epn), stride=(epn, 1)),
            )
            all_remote_stats = cute.make_tensor(
                meta.iterator + (PB + STATS_SUB),
                cute.make_layout((R, 2), stride=(2, 1)),
            )
            s_expert_counts = cute.make_tensor(s_col.iterator, cute.make_layout((E,)))
            s_selected_experts = cute.make_tensor(s_chosen.iterator, cute.make_layout((epn,)))
            s_scan_warp_prefix = cute.make_tensor(s_wmax.iterator, cute.make_layout((NUM_WARPS,)))
            for idx in cutlass.range(
                pid * num_threads + tid, R * 2, num_sms * num_threads
            ):
                stat_rank = idx // 2
                stat_idx = idx - stat_rank * 2
                all_remote_stats[stat_rank, stat_idx] = 0
            grid_sync(bar_p, num_sms, tid)
            for dest_rank in cutlass.range(pid, R, num_sms):
                local_start = dest_rank * epn
                local_end = local_start + epn
                for expert_idx in cutlass.range(tid, E, num_threads):
                    cnt = alloc_tensor[dest_rank, expert_idx]
                    s_expert_counts[expert_idx] = cnt
                for slot in cutlass.range(tid, epn, num_threads):
                    s_selected_experts[slot] = -1
                    all_experts_to_copy[dest_rank, slot] = -1
                cute.arch.barrier()

                # With epn slots every remote expert fits. Compact all positive
                # remote experts once in global-expert order.
                if tid < 32:
                    lane = tid
                    E_CHUNK = cutlass.const_expr(ceil_div(E, 32))
                    lanes_lt = (Uint32(1) << lane) - Uint32(1)
                    remote_expert_count = Int32(0)
                    for chunk in cutlass.range(E_CHUNK, unroll=1):
                        expert_idx = chunk * Int32(32) + lane
                        selected = cutlass.Boolean(False)
                        if expert_idx < E:
                            cnt = s_expert_counts[expert_idx]
                            selected = (cnt > 0) & (
                                (expert_idx < local_start) | (expert_idx >= local_end)
                            )
                        peers = Uint32(cute.arch.vote_ballot_sync(selected))
                        if selected:
                            slot = remote_expert_count + Int32(
                                cute.arch.popc(peers & lanes_lt)
                            )
                            s_selected_experts[slot] = expert_idx
                            all_experts_to_copy[dest_rank, slot] = expert_idx
                            owner_rank = expert_idx // epn
                            cute.arch.atomic_add(
                                elem_ptr(all_remote_stats, (owner_rank, 1)),
                                1,
                                scope="gpu",
                            )
                        remote_expert_count += Int32(cute.arch.popc(peers))
                    if lane == 0:
                        all_remote_stats[dest_rank, 0] = remote_expert_count
                cute.arch.barrier()

                count_values = []
                expert_values = []
                padded_values = []
                for i in cutlass.range_constexpr(IPT_EB):
                    group_idx = tid * IPT_EB + i
                    token_count = 0
                    expert_id = -1
                    if group_idx < 2 * epn:
                        if group_idx < epn:
                            expert_id = local_start + group_idx
                            token_count = s_expert_counts[expert_id]
                        else:
                            selected_expert = s_selected_experts[group_idx - epn]
                            if selected_expert >= 0:
                                token_count = s_expert_counts[selected_expert]
                                expert_id = selected_expert
                    padded_count = 0
                    if token_count > 0:
                        if cutlass.const_expr(tp > 1):
                            padded_count = cute.round_up(token_count, tp)
                        else:
                            padded_count = token_count
                    count_values.append(token_count)
                    expert_values.append(expert_id)
                    padded_values.append(padded_count)

                total_padded = 0
                for i in cutlass.range_constexpr(IPT_EB):
                    total_padded += padded_values[i]
                # Block-wide exclusive prefix (cub BlockScan equivalent):
                # intra-warp shfl scan + warp-segment combine.
                lane = tid & 31
                warp_id = tid >> 5
                inclusive = warp_inclusive_scan(total_padded, lane)
                if lane == 31:
                    s_scan_warp_prefix[warp_id] = inclusive
                cute.arch.barrier()
                if tid == 0:
                    acc = 0
                    for warp_idx in cutlass.range_constexpr(NUM_WARPS):
                        warp_total = s_scan_warp_prefix[warp_idx]
                        s_scan_warp_prefix[warp_idx] = acc
                        acc += warp_total
                cute.arch.barrier()
                base = s_scan_warp_prefix[warp_id] + inclusive - total_padded
                for i in cutlass.range_constexpr(IPT_EB):
                    group_idx = tid * IPT_EB + i
                    if group_idx < 2 * epn:
                        padded_end = base + padded_values[i]
                        token_count = count_values[i]
                        expert_id = expert_values[i]
                        if token_count > 0:
                            expert_offsets[dest_rank, expert_id] = base
                        all_cu_seqlens[dest_rank, group_idx] = padded_end
                        pad_start = 0
                        pad_count = 0
                        if token_count > 0:
                            pad_extra = padded_values[i] - token_count
                            if pad_extra > 0:
                                pad_start = base + token_count
                                pad_count = pad_extra
                        zero_fill_start[dest_rank, group_idx] = pad_start
                        zero_fill_count[dest_rank, group_idx] = pad_count
                        base += padded_values[i]
                cute.arch.barrier()
            grid_sync(bar_p, num_sms, tid)
            # Replicate ALLOC | TPE | EOFF into every peer chunk: each rank
            # reads those three sub-regions from its own chunk (plo = rank * ms
            # + PLAN_OFF below). CU | ZFR | ETC | STATS are read from rank 0's
            # chunk by every rank, so this publish is the only writer of a peer
            # chunk's plan and nothing republishes an element it drops. The v4
            # body covers whole 16B groups; nb = 3 * E * R is not necessarily a
            # multiple of 4, so the trailing nb % 4 entries -- the last EOFF
            # entries, i.e. the top experts' destination offsets -- need the
            # scalar store. Both loops share one grid-wide worker numbering, so
            # every element is written exactly once and the scalar region never
            # overlaps the vector body. R = 1 has no peer chunk and the plan is
            # already in rank 0's own chunk, where the publish is a self-copy.
            if cutlass.const_expr(R > 1):
                nb = 3 * E * R
                nvec = nb // 4
                worker = pid * num_threads + tid
                workers = num_sms * num_threads
                for i in cutlass.range(worker, nvec, workers):
                    a0 = meta[PB + i * 4 + 0]; a1 = meta[PB + i * 4 + 1]
                    a2 = meta[PB + i * 4 + 2]; a3 = meta[PB + i * 4 + 3]
                    addr = (mc.iterator + (PLAN_OFF + i * 4)).toint()
                    multimem_st_v4(addr.ir_value(), a0, a1, a2, a3)
                if cutlass.const_expr(nb % 4 != 0):
                    for i in cutlass.range(nvec * 4 + worker, nb, workers):
                        addr = (mc.iterator + (PLAN_OFF + i)).toint()
                        multimem_st_b32(addr.ir_value(), meta[PB + i])
        order = cute.make_tensor(meta.iterator + (rank * ms + ORDER_OFF), cute.make_layout((N,)))
        if cutlass.const_expr(R > 1):
            if rank != 0:
                self.run_c1(topk, order, tpe, lh, s_hist, s_bp, scratch, bar_p, num_sms, pid, tid)
                if rank == 1:
                    tk0 = cute.make_tensor(meta.iterator + (rank * ms + TOPK0_OFF), cute.make_layout((N,)))
                    tp0 = cute.make_tensor(meta.iterator + (rank * ms + TPE_OFF), cute.make_layout((E,)))
                    order0 = cute.make_tensor(meta.iterator + (rank * ms + ORDER0_OFF), cute.make_layout((N,)))
                    self.run_c1(tk0, order0, tp0, lh, s_hist, s_bp, scratch, bar_p, num_sms, pid, tid)
                    copy_v4_remote(meta, ORDER_OFF, order0, N, pid, tid, num_threads, num_sms)
        else:
            self.run_c1(topk, order, tpe, lh, s_hist, s_bp, scratch, bar_p, num_sms, pid, tid)
        # Clear this rank's src_info slice before all ranks publish fresh slot
        # provenance into destination-rank slices below. src_info mirrors dst's
        # rank-stride encoding: src_rank * NvS + offv; -1 is the empty-slot
        # sentinel. offv is always in [0, N), and NvS >= N.
        for idx in cutlass.range(pid * num_threads + tid, NvS, num_sms * num_threads):
            meta[rank * ms + SRC_INFO_OFF + idx] = Int32(-1)
        cross_rank_barrier(meta, ms, BARRIER_OFF, rank, R, bar_p, num_sms, num_threads, tid)
        s_expoff = cute.make_tensor(s_hist.iterator, cute.make_layout((E,)))
        for e in cutlass.range(tid, E, num_threads):
            s_expoff[e] = tpe[e]

        cute.arch.barrier()
        warp_exclusive_scan_e(s_expoff, E, tid)
        plo = rank * ms + PLAN_OFF
        order_in = cute.make_tensor(meta.iterator + (rank * ms + ORDER_OFF), cute.make_layout((N,)))
        topk_by_off = cute.make_tensor(topk.iterator, cute.make_layout((N,)))
        dst_out = cute.make_tensor(dst.iterator, cute.make_layout((N,)))
        tpe_cumsum_view = cute.make_tensor(
            meta.iterator + (plo + TPE_SUB),
            cute.make_layout((R, E), stride=(E, 1)),
        )
        alloc_cumsum_view = cute.make_tensor(
            meta.iterator + (plo + ALLOC_SUB),
            cute.make_layout((E, R), stride=(R, 1)),
        )
        expert_off_view = cute.make_tensor(
            meta.iterator + (plo + EOFF_SUB),
            cute.make_layout((R, E), stride=(E, 1)),
        )

        pd_group_count = cutlass.const_expr(2 * epn)
        group_begin, group_end, group_copy_begin, group_copy_count = _pd_cta_slice(
            pd_group_count, pid, num_sms, PHASE_D_TILE
        )
        etc_begin, etc_end, etc_copy_begin, etc_copy_count = _pd_cta_slice(
            R * epn, pid, num_sms, PHASE_D_TILE
        )
        # Index 0 of the stage tensors is the aligned envelope start; the
        # logical start is pointed inside the envelope by *_stage_bias, so the
        # Phase D writeback still reads by group/ETC-local indices.
        s_pd_cu_stage = cute.make_tensor(
            scratch.iterator + PD_CU_OFF,
            cute.make_layout((PD_CU_LEN,)),
        )
        s_pd_zfr_stage = cute.make_tensor(
            scratch.iterator + PD_ZFR_OFF,
            cute.make_layout((PD_ZFR_LEN,)),
        )
        s_pd_etc_stage = cute.make_tensor(
            scratch.iterator + PD_ETC_OFF,
            cute.make_layout((PD_ETC_LEN,)),
        )
        cu_src_begin = PB + CU_SUB + rank * pd_group_count + group_copy_begin
        zfr_src_begin = PB + ZFR_SUB + (rank * pd_group_count + group_copy_begin) * 2
        etc_src_begin = PB + ETC_SUB + etc_copy_begin
        zfr_copy_count = group_copy_count * 2

        if tid == 0:
            cute.arch.mbarrier_init(pd_mbar, 1)
        cute.arch.mbarrier_init_fence()
        # The scratch smem was previously written through the generic proxy
        # (Phase B s_tpe / run_c1 s_wcount). Bridge those writes into the
        # async proxy before the Phase-D bulk G2S copies overwrite the same
        # storage (every thread fences its own prior writes before the
        # rendezvous).
        cute.arch.fence_view_async_shared()
        cute.arch.barrier()
        if tid == 0:
            # Kick off the three G2S bulk copies before C2 starts; the
            # mbarrier's expected bytes must cover all actual bulk envelope
            # bytes.
            pd_stage_bytes = (
                _pd_aligned_ints(cu_src_begin, group_copy_count)
                + _pd_aligned_ints(zfr_src_begin, zfr_copy_count)
                + _pd_aligned_ints(etc_src_begin, etc_copy_count)
            ) * 4
            cute.arch.mbarrier_arrive_and_expect_tx(pd_mbar, pd_stage_bytes)
            _pd_issue_g2s(meta, s_pd_cu_stage, cu_src_begin, group_copy_count, pd_mbar)
            _pd_issue_g2s(meta, s_pd_zfr_stage, zfr_src_begin, zfr_copy_count, pd_mbar)
            _pd_issue_g2s(meta, s_pd_etc_stage, etc_src_begin, etc_copy_count, pd_mbar)
        cute.arch.barrier()
        seg = cute.ceil_div(N, num_sms)
        sbeg = pid * seg; send = cutlass.min(sbeg + seg, N)
        for base in cutlass.range(sbeg + tid, send, num_threads * ITEMS_PER_THREAD_P2):
            for i in cutlass.range_constexpr(ITEMS_PER_THREAD_P2):
                idx = base + i * BLOCK_DIM_P2
                if idx < send:
                    offv = order_in[idx]
                    expert_idx = topk_by_off[offv]
                    prev = 0
                    if rank > 0:
                        prev = tpe_cumsum_view[rank - 1, expert_idx]
                    global_rank = prev + (idx - s_expoff[expert_idx])
                    lo = 0; hi = R; pc = 0
                    for bin_step in cutlass.range_constexpr(LOG2_R):
                        mid = (lo + hi) >> 1
                        ac = alloc_cumsum_view[expert_idx, mid]
                        if ac > global_rank: hi = mid
                        else: lo = mid + 1; pc = ac
                    bo = expert_off_view[lo, expert_idx]
                    dst_out[offv] = lo * NvS + bo + (global_rank - pc)
                    # Publish source provenance for the dispatch builder. This
                    # mirrors dst's rank-stride encoding, but points back to the
                    # source rank and source flat top-k offset.
                    src_val = rank * NvS + offv
                    loff = bo + (global_rank - pc)
                    meta[lo * ms + SRC_INFO_OFF + loff] = src_val
        # Dispatch may start independently on each rank after planning
        # returns. This barrier makes all peer src_info writes visible before
        # any rank's fresh dispatch builder reads its local src_info slice.
        cross_rank_barrier(meta, ms, BARRIER_OFF, rank, R, bar_p, num_sms, num_threads, tid)

        # Canonicalize dst duplicate entries. The first top-k entry per
        # destination rank stays non-negative and copies the payload; later
        # entries encode -raw_dst - 1 and only carry weights. Fresh dispatch
        # materializes the dedup structures from src_info.
        seg_dst = cute.ceil_div(S, num_sms)
        sbeg_dst = pid * seg_dst; send_dst = cutlass.min(sbeg_dst + seg_dst, S)
        for base in cutlass.range(sbeg_dst + tid, send_dst, num_threads):
            s = base
            base_idx = s * K
            dst_vals = []
            dests = []
            for k in cutlass.range_constexpr(K):
                v = dst_out[base_idx + k]
                d = v // NvS
                dst_vals.append(v)
                dests.append(d)
            # Detect duplicate destination ranks in registers. Two i64 masks
            # cover up to 128 ranks; host-side bounds guard larger configs.
            seen_lo = Int64(0)
            seen_hi = Int64(0)
            for k in cutlass.range_constexpr(K):
                d = dests[k]
                dup = cutlass.Boolean(False)
                if d < Int32(64):
                    shift = Int64(d)
                    bit = Int64(1) << shift
                    dup = (seen_lo & bit) != Int64(0)
                    seen_lo = seen_lo | bit
                else:
                    shift = Int64(d - Int32(64))
                    bit = Int64(1) << shift
                    dup = (seen_hi & bit) != Int64(0)
                    seen_hi = seen_hi | bit
                if dup:
                    dst_out[base_idx + k] = -(dst_vals[k]) - 1
        cute.arch.mbarrier_wait(pd_mbar, 0)
        cu_stage_bias = _pd_stage_bias(cu_src_begin)
        zfr_stage_bias = _pd_stage_bias(zfr_src_begin)
        etc_stage_bias = _pd_stage_bias(etc_src_begin)
        if pid == 0:
            for i in cutlass.range(tid, 2, num_threads):
                remote_stats[i] = meta[PB + STATS_SUB + rank * 2 + i]
        for group_idx in cutlass.range(group_begin + tid, group_end, num_threads):
            local_group = group_idx - group_begin
            cu_seqlens[group_idx] = s_pd_cu_stage[cu_stage_bias + local_group]
            zfr[group_idx * 2] = s_pd_zfr_stage[zfr_stage_bias + local_group * 2]
            zfr[group_idx * 2 + 1] = s_pd_zfr_stage[zfr_stage_bias + local_group * 2 + 1]

        for etc_idx in cutlass.range(etc_begin + tid, etc_end, num_threads):
            experts_to_copy[etc_idx] = s_pd_etc_stage[
                etc_stage_bias + (etc_idx - etc_begin)
            ]
        # grid_sync self-resets: the counter's low 31 bits always return to 0
        # and the top bit alternates between two phases, so no cleanup zeroing
        # is needed.

# ============================================================
# Host side: compile cache + launch
# ============================================================
@functools.lru_cache(maxsize=None)
def _get_compiled(R, E, S, K, NvS_capacity, NvS, num_vblocks, meta_stride,
                  TPE_OFF, PLAN_OFF, BARRIER_OFF, TOPK0_OFF, ORDER_OFF, ORDER0_OFF,
                  token_padding, num_sms):
    k = PlanningKernel(R, E, S, K, NvS_capacity, NvS, num_vblocks, meta_stride,
                       TPE_OFF, PLAN_OFF, BARRIER_OFF, TOPK0_OFF, ORDER_OFF, ORDER0_OFF,
                       token_padding, num_sms)
    i32 = make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=16)
    return cute.compile(k, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
                        i32, i32, i32, i32, Int32(0), cuda.CUstream(0))


def _launch_planning_kernel(ctx, topk, tpe, dst, cu_seqlens,
                            experts_to_copy, zero_fill_ranges, remote_stats):
    comp = _get_compiled(
        int(ctx['R']),
        int(ctx['E']),
        int(ctx['S']),
        int(ctx['K']),
        int(ctx['NvS_capacity']),
        int(ctx['NvS']),
        int(ctx['num_vblocks']),
        int(ctx['meta_chunk_padded']),
        int(ctx['TPE_OFF']),
        int(ctx['PLAN_OFF']),
        int(ctx['BARRIER_OFF']),
        int(ctx['TOPK0_OFF']),
        int(ctx['ORDER_OFF']),
        int(ctx['ORDER0_OFF']),
        int(ctx['token_padding']),
        int(ctx['num_sms']),
    )

    def p16(t):  # large buffers 16B aligned -> allows coalesced/vectorized access
        return make_ptr(Int32, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    def p4(t):  # odd-length outputs; avoids out-of-bounds vector writes
        return make_ptr(Int32, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    comp(
        p16(tpe),
        p16(topk),
        p16(ctx['meta_buf']),
        p16(ctx['meta_mc']),
        p16(dst),
        p4(cu_seqlens),
        p4(experts_to_copy),
        p4(zero_fill_ranges),
        p4(remote_stats),
        p16(ctx['alloc']),
        p16(ctx['group_tokens']),
        p16(ctx['z']),
        p16(ctx['local_hist']),
        p16(ctx['grid_sync_bar']),
        Int32(int(ctx['rank'])),
        stream,
    )


def _round4(n):
    return (n + 3) & ~3


def allocate_planning_outputs(ctx: dict):
    """Allocate a ``(MoonEPCommPlan, cu_seqlens)`` pair on the current stream.

    The plan-owned dedup tensors are allocated here so the returned plan is
    complete, but fresh planning leaves their contents for the dispatch builder
    to materialize.
    """
    E = ctx['E']
    epn = E // ctx['R']
    S = ctx['S']
    K = ctx['K']
    N = S * K
    NvS = ctx['NvS']
    dev = ctx['meta_buf'].device

    # CuTe DSL kernel grid-stride writes get vectorized (up to 4×int32 = 16B).
    # Over-allocate output tensors to a multiple of 4 and then slice, avoiding
    # an out-of-bounds full-vector write at the tail (returned shapes unchanged).
    dst = torch.empty(_round4(N), dtype=torch.int32, device=dev)[:N]
    cu_seqlens = torch.empty(_round4(2 * epn), dtype=torch.int32, device=dev)[:2 * epn]
    experts_to_copy = torch.empty(
        _round4(ctx['R'] * epn), dtype=torch.int32, device=dev
    )[:ctx['R'] * epn].view(ctx['R'], epn)
    zero_fill_ranges = torch.empty(
        _round4(4 * epn), dtype=torch.int32, device=dev
    )[:4 * epn].view(2 * epn, 2)
    remote_stats = torch.empty(_round4(2), dtype=torch.int32, device=dev)[:2]
    dup_groups = torch.empty(
        _round4(NvS * 3), dtype=torch.int32, device=dev
    )[:NvS * 3].view(NvS, 3)
    dup_loffs = torch.empty(_round4(NvS), dtype=torch.int32, device=dev)[:NvS]
    dup_counts = torch.empty(_round4(2), dtype=torch.int32, device=dev)[:2]

    plan = MoonEPCommPlan(
        dst=dst,
        experts_to_copy=experts_to_copy,
        zero_fill_ranges=zero_fill_ranges,
        remote_stats=remote_stats,
        dup_groups=dup_groups,
        dup_loffs=dup_loffs,
        dup_counts=dup_counts,
        N=N,
        R=ctx['R'],
        E=E,
        NvS=NvS,
        K=K,
    )
    return plan, cu_seqlens


def _check_planning_outputs(ctx: dict, cu_seqlens, plan) -> None:
    assert isinstance(plan, MoonEPCommPlan)
    E = ctx['E']
    epn = E // ctx['R']
    assert plan.N == ctx['S'] * ctx['K']
    assert plan.R == ctx['R']
    assert plan.E == E
    assert plan.NvS == ctx['NvS']
    assert plan.K == ctx['K']
    assert cu_seqlens.dtype == torch.int32 and cu_seqlens.is_contiguous()
    assert tuple(cu_seqlens.shape) == (2 * epn,)


def _check_dedup_encoding_bounds(ctx: dict) -> None:
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
    assert R <= 128, (
        f"dst duplicate canonicalization rank bitset requires R <= 128, got {R}"
    )
    assert K <= (1 << KIDX_BITS) - 1, (
        f"primary_packed encoding requires K <= {(1 << KIDX_BITS) - 1}, got {K}"
    )
    assert NvS <= (1 << NvS_BITS) - 1, (
        f"primary_packed encoding requires NvS <= {(1 << NvS_BITS) - 1}, got {NvS}"
    )
    assert K <= 32, f"kmask bitmask requires K <= 32, got {K}"


def launch_planning(
    ctx: dict,
    topk_experts_flat,
    tokens_per_expert,
    cu_seqlens,
    plan,
) -> None:
    """Run the planning kernel; may reuse caller-allocated output objects.

    Results are written in-place into ``plan.dst``, ``plan.experts_to_copy``,
    ``plan.zero_fill_ranges``, ``plan.remote_stats`` and ``cu_seqlens``.
    The plan-owned dedup structures are materialized in the fresh dispatch
    builder; the reuse path with an existing plan reuses these tensors
    directly.
    """
    _check_planning_outputs(ctx, cu_seqlens, plan)
    _check_dedup_encoding_bounds(ctx)

    _launch_planning_kernel(
        ctx, topk_experts_flat, tokens_per_expert,
        plan.dst, cu_seqlens, plan.experts_to_copy,
        plan.zero_fill_ranges, plan.remote_stats,
    )
