"""Remote expert grad reduce kernel.

Accumulates fp32 gradients from per-rank prefetch reduce buffers back into the
owner rank's local expert gradient buffer.

Warp-specialized, tile-partitioned design:
  - 5 warps/CTA: 1 load warp (G2S TMA), 4 fp32 ACC warps (acc + store).
  - The TMA pipeline streams only the remote reduce tiles. The acc warps
    seed their accumulator with the local grad tile through a vectorized
    16B autovec_copy whose latency hides under the first remote-stage wait,
    then store the result back with 16B stores. The seed/store MUST go
    through autovec_copy over views with *static* strides: per-element
    fragment assignments from gmem (or a dynamic-stride view) demote the
    128-reg accumulator to a stack alloca or fall back to scalar LDG/STG.
  - Work is split by 128x128 tile across all SMs over a prescan-compacted
    list of active local experts (experts with no remote slot cost nothing).
    The prescan itself is cooperative (smem stage + cta-atomic count +
    match_any warp fill): a single-thread gmem scan of the R*B plan is
    bottlenecked by serial load latency.
  - Accumulation never writes the reduce buffers. After all tiles are summed
    a cross-rank barrier fences all peers, then each rank clears only its own
    consumed slots locally with grid-strided 16B vector stores. Remote-write
    clears (scalar or bulk S2G) were tried and rejected: simultaneous remote
    reads + remote writes share a single NVLink budget per GPU (each
    direction gets roughly half), so riding the "idle" write
    direction stretches phase 1 far more than the local clear costs.
"""

import functools

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.cute.nvgpu.cpasync as cpasync
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_ptr

from moonep._common import cross_rank_barrier
from moonep.planning import match_any_b32, st_global_v4_s32


class GradReduceKernel:
    """Persistent fp32 tile reducer for remote expert grads."""

    M_BLOCK = 128
    N_BLOCK = 128
    STAGES = 3                 # load-pipeline depth
    ACC_THREADS = 128          # warps 1..4
    NUM_THREADS = 160          # 5 warps: 1 load + 4 fp32 acc (acc also stores)

    def __init__(
        self,
        E: int,
        H: int,
        Hp: int,
        R: int,
        B: int,
        meta_stride: int,
        num_sms: int,
        smem_budget: int,
    ):
        self.E = E
        self.H = H
        self.Hp = Hp
        self.R = R
        self.B = B
        self.meta_stride = meta_stride
        self.num_sms = num_sms
        need = self._smem_bytes()
        if need > smem_budget:
            raise RuntimeError(
                "grad_reduce: not enough per-block shared memory for one "
                f"{self.M_BLOCK}x{self.N_BLOCK} fp32 acc tile: need {need} B, "
                f"budget {smem_budget} B"
            )

    def _smem_bytes(self) -> int:
        def _round_up(n: int, a: int) -> int:
            return (n + a - 1) // a * a

        tile_bytes = self.M_BLOCK * self.N_BLOCK * 4
        stage = _round_up(self.STAGES * tile_bytes, 128)
        mbar = _round_up(self.STAGES * 2 * 8, 16)
        # off[EPN+1] + alist[EPN] + acnt[1] + slist[R*B] + sexp[R*B]
        # + cur[EPN+1] + clist[B] + ccnt[1]
        epn = self.E // self.R
        scan = _round_up(
            (3 * epn + 4 + 2 * self.R * self.B + self.B) * 4, 128)
        return stage + mbar + scan + 256

    @cute.jit
    def __call__(
        self,
        expert_grad_ptr: cute.Pointer,    # fp32 [E, H, H']
        reduce_buf_ptr: cute.Pointer,     # fp32 [R, B, H, H']
        experts_ptr: cute.Pointer,        # int32 [R, B]
        meta_ptr: cute.Pointer,           # int32 [R*meta_stride] (barrier)
        bar_ptr: cute.Pointer,            # int32 [1] grid barrier counter
        rank: Int32,
        barrier_off: Int32,
        stream: cuda.CUstream,
    ):
        E = cutlass.const_expr(self.E)
        H = cutlass.const_expr(self.H)
        Hp = cutlass.const_expr(self.Hp)
        R = cutlass.const_expr(self.R)
        B = cutlass.const_expr(self.B)
        meta_stride = cutlass.const_expr(self.meta_stride)
        M_BLOCK = cutlass.const_expr(self.M_BLOCK)
        N_BLOCK = cutlass.const_expr(self.N_BLOCK)
        Hp64 = cutlass.Int64(Hp)

        expert_grad = cute.make_tensor(
            expert_grad_ptr,
            cute.make_layout((E * H, Hp), stride=(Hp64, cutlass.Int64(1))),
        )
        reduce_buf = cute.make_tensor(
            reduce_buf_ptr,
            cute.make_layout((R * B * H, Hp), stride=(Hp64, cutlass.Int64(1))),
        )
        experts = cute.make_tensor(experts_ptr, cute.make_layout((R * B,)))
        meta = cute.make_tensor(meta_ptr, cute.make_layout((R * meta_stride,)))
        bar = cute.make_tensor(bar_ptr, cute.make_layout((1,)))

        # 2D TMA atom for the remote reduce tiles; the local grad tile is
        # loaded by the acc warps with vectorized 16B reads.
        smem_tile_layout = cute.make_ordered_layout((M_BLOCK, N_BLOCK), order=(1, 0))
        cta_tiler = (M_BLOCK, N_BLOCK)
        tma_g2s_red, tma_red = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            reduce_buf,
            smem_tile_layout,
            cta_tiler,
        )

        self.kernel(expert_grad, reduce_buf,
                    tma_g2s_red, tma_red, experts, meta, bar,
                    rank, barrier_off).launch(
            grid=(self.num_sms, 1, 1),
            block=(self.NUM_THREADS, 1, 1),
            smem=self._smem_bytes(),
            stream=stream,
            cooperative=True,
        )

    @cute.kernel
    def kernel(
        self,
        expert_grad: cute.Tensor,
        reduce_buf: cute.Tensor,
        tma_g2s_red: cute.CopyAtom,
        tma_red: cute.Tensor,
        experts: cute.Tensor,
        meta: cute.Tensor,
        bar: cute.Tensor,
        rank: Int32,
        barrier_off: Int32,
    ):
        E = cutlass.const_expr(self.E)
        H = cutlass.const_expr(self.H)
        Hp = cutlass.const_expr(self.Hp)
        R = cutlass.const_expr(self.R)
        B = cutlass.const_expr(self.B)
        M_BLOCK = cutlass.const_expr(self.M_BLOCK)
        N_BLOCK = cutlass.const_expr(self.N_BLOCK)
        meta_stride = cutlass.const_expr(self.meta_stride)
        num_ranks = cutlass.const_expr(self.R)
        TILE_ELEMS = cutlass.const_expr(M_BLOCK * N_BLOCK)
        MTILES = cutlass.const_expr(H // M_BLOCK)
        NTILES = cutlass.const_expr(Hp // N_BLOCK)
        TILES_PER_EXPERT = cutlass.const_expr(MTILES * NTILES)
        EPN = cutlass.const_expr(E // R)

        STAGES = cutlass.const_expr(self.STAGES)
        ACC_THREADS = cutlass.const_expr(self.ACC_THREADS)
        num_threads = cutlass.const_expr(self.NUM_THREADS)
        TILE_BYTES = cutlass.const_expr(TILE_ELEMS * 4)

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        smem = utils.SmemAllocator()
        load_mbar = smem.allocate_array(Int64, num_elems=2 * STAGES)
        stage_smem = smem.allocate_tensor(
            Float32,
            cute.make_ordered_layout((M_BLOCK, N_BLOCK, STAGES), order=(1, 0, 2)),
            byte_alignment=128,
        )
        # flat per-stage view of the same smem for the acc warps.
        stage_flat = cute.make_tensor(
            stage_smem.iterator,
            cute.make_ordered_layout((TILE_ELEMS, STAGES), order=(0, 1)),
        )
        # prescan: cnt[le] and a per-expert-grouped slot list. cnt[EPN] holds
        # running prefix; slist[off[le] .. off[le]+cnt[le]) are the matching rb.
        off = smem.allocate_tensor(Int32, cute.make_layout((EPN + 1,)), byte_alignment=16)
        # compacted list of active local experts (off[le+1] > off[le]).
        alist = smem.allocate_tensor(Int32, cute.make_layout((EPN,)), byte_alignment=16)
        acnt = smem.allocate_tensor(Int32, cute.make_layout((1,)), byte_alignment=16)
        slist = smem.allocate_tensor(Int32, cute.make_layout((R * B,)), byte_alignment=16)
        # smem stage of experts[] (pre-biased to local ids) for the prescan.
        sexp = smem.allocate_tensor(Int32, cute.make_layout((R * B,)), byte_alignment=16)
        # slist fill cursors; cur[EPN] is a dummy bucket for invalid lanes.
        cur = smem.allocate_tensor(Int32, cute.make_layout((EPN + 1,)), byte_alignment=16)
        # this rank's own consumed slots (experts[rank,b] >= 0) for clearing.
        clist = smem.allocate_tensor(Int32, cute.make_layout((B,)), byte_alignment=16)
        ccnt = smem.allocate_tensor(Int32, cute.make_layout((1,)), byte_alignment=16)

        rank_epn = rank * EPN

        # ---- prescan experts once (parallel): bucket slots per local expert.
        # A single-thread gmem scan of all R*B entries is bottlenecked by a
        # serial load latency chain, so it must be cooperative: stage the
        # table to smem with all threads, count with cta atomics, then have
        # warp 0 fill slist chunk-by-chunk with match_any so the final order
        # stays rb-ascending (bitwise-identical to the serial scan).
        for i in cutlass.range(tidx, R * B, self.NUM_THREADS, unroll=1):
            sexp[i] = experts[i] - rank_epn
        for i in cutlass.range(tidx, EPN + 1, self.NUM_THREADS, unroll=1):
            off[i] = Int32(0)
        if tidx == 0:
            ccnt[0] = Int32(0)
        cute.arch.sync_threads()
        for i in cutlass.range(tidx, R * B, self.NUM_THREADS, unroll=1):
            e = sexp[i]
            if e >= Int32(0) and e < Int32(EPN):
                cute.arch.atomic_add(off.iterator + (e + Int32(1)), Int32(1),
                                     scope="cta")
        # this rank's own consumed slots (experts[rank,b] >= 0), unordered.
        for b in cutlass.range(tidx, B, self.NUM_THREADS, unroll=1):
            if sexp[rank * B + b] >= Int32(0) - rank_epn:
                pos = cute.arch.atomic_add(ccnt.iterator, Int32(1), scope="cta")
                clist[pos] = Int32(b)
        cute.arch.sync_threads()
        # serial prefix + active-expert compaction over EPN (smem-only).
        if tidx == 0:
            run = Int32(0)
            ac = Int32(0)
            for le in cutlass.range(EPN, unroll=1):
                c = off[le + 1]
                if c > Int32(0):
                    alist[ac] = Int32(le)
                    ac = ac + Int32(1)
                run = run + c
                off[le + 1] = run
            acnt[0] = ac
        cute.arch.sync_threads()
        for i in cutlass.range(tidx, EPN + 1, self.NUM_THREADS, unroll=1):
            cur[i] = off[i]
        cute.arch.sync_threads()
        if warp_idx == 0:
            lane = cute.arch.lane_idx()
            lanes_lt = (Uint32(1) << lane) - Uint32(1)
            NCHUNK = cutlass.const_expr((R * B + 31) // 32)
            for c0 in cutlass.range(NCHUNK, unroll=1):
                rb = c0 * 32 + lane
                e = Int32(EPN)
                if rb < Int32(R * B):
                    ee = sexp[rb]
                    if ee >= Int32(0) and ee < Int32(EPN):
                        e = ee
                peers = match_any_b32(e)
                cell = cur.iterator + e
                base = cute.arch.load(cell, Int32)
                pos = base + Int32(cute.arch.popc(peers & lanes_lt))
                cute.arch.sync_warp()
                if (peers & lanes_lt) == Uint32(0):
                    cute.arch.store(cell, base + Int32(cute.arch.popc(peers)))
                cute.arch.sync_warp()
                if e < Int32(EPN):
                    slist[pos] = Int32(rb)
        cute.arch.sync_threads()

        cta_tiler = (M_BLOCK, N_BLOCK)
        smem_for_tma = cute.group_modes(stage_smem, 0, 2)
        red_tiles = cute.zipped_divide(tma_red, cta_tiler)
        tRsR, tRgR = cpasync.tma_partition(
            tma_g2s_red, 0, cute.make_layout(1), smem_for_tma, red_tiles,
        )
        load_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=load_mbar, num_stages=STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 4),
            tx_count=TILE_BYTES,
        )
        ld_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, STAGES)
        cd_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, STAGES)
        acc_tid = tidx - 32
        H_PER = cutlass.const_expr(TILE_ELEMS // ACC_THREADS)
        acc_reg = cute.make_fragment((H_PER,), Float32)

        # ============ phase 1: warp-specialized accumulate, no clears ========
        total_work = acnt[0] * TILES_PER_EXPERT
        for work_idx in cutlass.range(bidx, total_work, self.num_sms, unroll=1):
            local_expert = alist[work_idx // TILES_PER_EXPERT]
            rem = work_idx % TILES_PER_EXPERT
            mt = rem // NTILES
            nt = rem % NTILES
            expert_id = rank_epn + local_expert
            base_row = mt * M_BLOCK
            base_col = nt * N_BLOCK
            beg = off[local_expert]
            nslot = off[local_expert + 1] - beg
            # load warp (0): stream the nslot remote reduce tiles.
            if warp_idx == 0:
                for s in cutlass.range(nslot, unroll=1):
                    load_pipe.producer_acquire(ld_state)
                    src_mt = slist[beg + s] * MTILES + mt
                    cute.copy(
                        tma_g2s_red,
                        tRgR[(None, (src_mt, nt))],
                        tRsR[(None, ld_state.index)],
                        tma_bar_ptr=load_pipe.producer_get_barrier(ld_state),
                    )
                    ld_state.advance()
            # acc warps (1..4): seed the accumulator with the local grad tile
            # (16B vector loads, hidden under the first remote-stage wait),
            # sum the remote stages, then write back with 16B stores. Each
            # thread owns groups of 4 consecutive elements:
            #   e = (j//4)*4*ACC_THREADS + acc_tid*4 + j%4.
            elif warp_idx >= 1:
                grad0 = expert_id * H + base_row
                row_off = acc_tid // Int32(32)
                col0 = (acc_tid % Int32(32)) * Int32(4)
                base_t = row_off * N_BLOCK + col0
                # per-thread (32 groups x 4 consecutive elems) view of the
                # grad tile. The strides must stay static and the offset 16B
                # aligned for autovec_copy to emit 128-bit accesses. The row
                # offset must be widened to i64 BEFORE multiplying by Hp:
                # expert_id * H * Hp exceeds 2^31 at production shapes.
                goff = (cutlass.Int64(grad0 + row_off) * cutlass.Int64(Hp)
                        + cutlass.Int64(base_col + col0))
                gview = cute.make_tensor(
                    expert_grad.iterator + cute.assume(goff, divby=4),
                    cute.make_layout(
                        (H_PER // 4, 4),
                        stride=(4 * self.Hp, 1),
                    ),
                )
                acc_view = cute.make_tensor(
                    acc_reg.iterator,
                    cute.make_layout((H_PER // 4, 4), stride=(4, 1)),
                )
                cute.autovec_copy(gview, acc_view)
                for _ in cutlass.range(nslot, unroll=1):
                    load_pipe.consumer_wait(cd_state)
                    for j in cutlass.range_constexpr(H_PER):
                        acc_reg[j] = acc_reg[j] + stage_flat[
                            (j // 4) * (self.ACC_THREADS * 4) + (j % 4) + base_t,
                            cd_state.index,
                        ]
                    # Stage may be overwritten immediately after release.
                    # consumer_release signals from each warp's lane 0 only
                    # (PipelineTmaAsync signalling thread; consumer_group=4),
                    # so sync_warp makes that arrive cover all 32 lanes' reads;
                    # cross-warp completion is gated by the 4-count empty mbarrier.
                    cute.arch.fence_acq_rel_cta()
                    cute.arch.sync_warp()
                    load_pipe.consumer_release(cd_state)
                    cd_state.advance()
                cute.autovec_copy(acc_view, gview)

        # ============ phase 2: cross-rank barrier then local clear ============
        cross_rank_barrier(
            meta, meta_stride, barrier_off, rank, num_ranks,
            bar.iterator, Int32(self.num_sms), num_threads, tidx,
        )
        # Each rank zeroes only its own consumed slots (experts[rank,b] >= 0).
        # A slot is one contiguous [H*Hp] fp32 region; clear it with
        # grid-strided 16B vector stores from every thread. This is bound by
        # the SM<->L2 write throughput, which vector stores already saturate
        # (scalar 4B stores fall short, and wider machinery such as TMA bulk
        # S2G zeros buys nothing).
        cons = ccnt[0]
        VECS = cutlass.const_expr(H * Hp // 4)
        for k in cutlass.range(cons, unroll=1):
            sbase = (cutlass.Int64(rank * B + clist[k])
                     * cutlass.Int64(H) * cutlass.Int64(Hp))
            for v in cutlass.range(bidx * self.NUM_THREADS + tidx, VECS,
                                   self.num_sms * self.NUM_THREADS, unroll=1):
                addr = (reduce_buf.iterator
                        + (sbase + cutlass.Int64(v) * 4)).toint()
                st_global_v4_s32(addr.ir_value(),
                                 Int32(0), Int32(0), Int32(0), Int32(0))


@functools.lru_cache(maxsize=None)
def _max_smem_per_block_optin(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).shared_memory_per_block_optin


@functools.lru_cache(maxsize=None)
def _get_compiled(
    E: int,
    H: int,
    Hp: int,
    R: int,
    B: int,
    meta_stride: int,
    num_sms: int,
    device_index: int,
):
    smem_budget = _max_smem_per_block_optin(device_index) - 1024
    kernel = GradReduceKernel(
        E=E,
        H=H,
        Hp=Hp,
        R=R,
        B=B,
        meta_stride=meta_stride,
        num_sms=num_sms,
        smem_budget=smem_budget,
    )

    f32_ptr = make_ptr(Float32, 0, cute.AddressSpace.gmem, assumed_align=16)
    i32_ptr = make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=16)
    stream_arg = cuda.CUstream(0)

    return cute.compile(
        kernel,
        f32_ptr,
        f32_ptr,
        i32_ptr,
        i32_ptr,
        i32_ptr,
        Int32(0),
        Int32(0),
        stream_arg,
    )


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
    """Launch remote expert grad reduction.

    Args:
        remote_expert_grads: contiguous fp32 tensor shaped [E, H, H'].
            The current rank owns expert ids
            ``rank * (E // R) : (rank + 1) * (E // R)``; only that range is
            updated.
        remote_reduce_buffers: contiguous fp32 tensor shaped [R, B, H, H'].
            Slots whose ``experts_to_copy[r, b]`` belongs to this rank's owner
            range are accumulated into ``remote_expert_grads``; afterwards a
            cross-rank barrier fences peers and each rank clears its own
            consumed slots locally.
        experts_to_copy: contiguous int32 tensor shaped [R, B].
        rank: current EP rank.
        num_sms: number of persistent CTAs to launch.
        meta_buf: int32 NVL-distributed meta buffer holding the barrier slots.
        meta_stride: per-rank stride (meta_chunk_padded) into ``meta_buf``.
        barrier_off: offset of the barrier slots within each rank's chunk.
        grid_sync_bar: int32 [1] grid-barrier counter.

    The caller must ensure all ranks have finished writing
    ``remote_reduce_buffers`` before launching this kernel on any rank.
    """
    if remote_reduce_buffers.numel() == 0 or experts_to_copy.numel() == 0:
        return

    assert remote_expert_grads.dtype == torch.float32 and remote_expert_grads.is_contiguous(), \
        "remote_expert_grads must be contiguous fp32 [E, H, H']"
    assert remote_reduce_buffers.dtype == torch.float32 and remote_reduce_buffers.is_contiguous(), \
        "remote_reduce_buffers must be contiguous fp32 [R, B, H, H']"
    assert experts_to_copy.dtype == torch.int32 and experts_to_copy.is_contiguous(), \
        "experts_to_copy must be contiguous int32 [R, B]"
    assert remote_expert_grads.ndim == 3, \
        f"remote_expert_grads must have rank 3, got shape={tuple(remote_expert_grads.shape)}"
    assert remote_reduce_buffers.ndim == 4, \
        f"remote_reduce_buffers must have rank 4, got shape={tuple(remote_reduce_buffers.shape)}"
    assert experts_to_copy.ndim == 2, \
        f"experts_to_copy must have rank 2, got shape={tuple(experts_to_copy.shape)}"

    E, H, Hp = (int(x) for x in remote_expert_grads.shape)
    R, B, buf_H, buf_Hp = (int(x) for x in remote_reduce_buffers.shape)
    assert tuple(experts_to_copy.shape) == (R, B), \
        f"experts_to_copy shape {tuple(experts_to_copy.shape)} must match [R, B]=[{R}, {B}]"
    assert buf_H == H and buf_Hp == Hp, (
        f"remote_reduce_buffers shape {tuple(remote_reduce_buffers.shape)} "
        f"incompatible with remote_expert_grads {tuple(remote_expert_grads.shape)}"
    )
    assert E % R == 0, f"E ({E}) must be divisible by R ({R})"
    assert 0 <= int(rank) < R, f"rank must be in [0, {R}), got {rank}"
    assert H % GradReduceKernel.M_BLOCK == 0 and Hp % GradReduceKernel.N_BLOCK == 0, \
        f"H and H' must be multiples of ({GradReduceKernel.M_BLOCK}, {GradReduceKernel.N_BLOCK}), got ({H}, {Hp})"
    assert isinstance(num_sms, int) and num_sms > 0, \
        f"num_sms must be a positive int, got {num_sms}"

    device_index = remote_expert_grads.device.index
    assert device_index is not None, "remote_expert_grads must be a CUDA tensor"
    assert remote_reduce_buffers.device.index == device_index, \
        "remote_reduce_buffers must be on the same CUDA device as remote_expert_grads"
    assert experts_to_copy.device.index == device_index, \
        "experts_to_copy must be on the same CUDA device as remote_expert_grads"

    compiled = _get_compiled(E, H, Hp, R, B, int(meta_stride), int(num_sms),
                             int(device_index))

    grad_ptr = make_ptr(
        Float32,
        remote_expert_grads.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    reduce_ptr = make_ptr(
        Float32,
        remote_reduce_buffers.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    experts_ptr = make_ptr(
        Int32,
        experts_to_copy.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    meta_ptr = make_ptr(
        Int32, meta_buf.data_ptr(), cute.AddressSpace.gmem, assumed_align=16,
    )
    bar_ptr = make_ptr(
        Int32, grid_sync_bar.data_ptr(), cute.AddressSpace.gmem, assumed_align=16,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compiled(grad_ptr, reduce_ptr, experts_ptr, meta_ptr, bar_ptr,
             Int32(int(rank)), Int32(int(barrier_off)), stream)
