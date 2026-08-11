"""
Remote expert prefetch kernel.

Copies selected remote expert weights from ``remote_expert[E, H, H']`` into
``prefetch_buffers[B, H, H']`` using a persistent, warp-specialized 2D TMA
pipeline:

  - warp 0: GMEM -> SMEM 2D TMA load
  - warp 1: SMEM -> GMEM 2D TMA store

The initial tile shape is fixed at 128 x 128 elements.  H and H' are
therefore required to be multiples of 128 for this first implementation.
For a scale tensor whose natural trailing extent is K/32 (never a
multiple of 128), re-tile its contiguous per-expert byte range to
``[128, nbytes // 128]`` before calling in.
"""

import functools

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.cute.nvgpu.cpasync as cpasync
from cutlass import BFloat16, Int8, Int32, Int64, Uint8
from cutlass.cute.runtime import make_ptr


_ELEM_TYPES = {
    torch.bfloat16: (BFloat16, 2),
    torch.int8: (Int8, 1),
    torch.uint8: (Uint8, 1),
}


class PrefetchKernel:
    """Persistent 2D TMA remote-expert prefetch."""

    NUM_THREADS = 64
    PRODUCER_WARP = 0
    CONSUMER_WARP = 1
    M_BLOCK = 128
    N_BLOCK = 128

    def __init__(
        self,
        E: int,
        H: int,
        Hp: int,
        B: int,
        num_sms: int,
        smem_budget: int,
        elem_ty=BFloat16,
        elem_bytes: int = 2,
    ):
        self.E = E
        self.H = H
        self.Hp = Hp
        self.B = B
        self.num_sms = num_sms
        self.elem_ty = elem_ty
        self.elem_bytes = elem_bytes
        self.stages = self._pick_stages(smem_budget)
        if self.stages == 0:
            raise RuntimeError(
                "prefetch: not enough per-block shared memory for one "
                f"{self.M_BLOCK}x{self.N_BLOCK} x {elem_bytes}B tile under "
                f"budget {smem_budget} B"
            )

    def _smem_bytes(self, stages: int) -> int:
        def _round_up(n: int, a: int) -> int:
            return (n + a - 1) // a * a

        tile_bytes = self.M_BLOCK * self.N_BLOCK * self.elem_bytes
        return (
            _round_up(stages * tile_bytes, 128)
            + _round_up(stages * 2 * 8, 16)
            + _round_up(2 * self.B * 4, 16)  # expert/slot compaction tables
            + 256
        )

    def _pick_stages(self, smem_budget: int) -> int:
        for stages in (6, 5, 4, 3, 2):
            if self._smem_bytes(stages) <= smem_budget:
                return stages
        return 0

    @cute.jit
    def __call__(
        self,
        remote_expert_ptr: cute.Pointer,    # elem_ty [E, H, H']
        prefetch_buf_ptr: cute.Pointer,     # elem_ty [B, H, H']
        experts_ptr: cute.Pointer,          # int32 [B]
        stream: cuda.CUstream,
    ):
        E = cutlass.const_expr(self.E)
        H = cutlass.const_expr(self.H)
        Hp = cutlass.const_expr(self.Hp)
        B = cutlass.const_expr(self.B)
        M_BLOCK = cutlass.const_expr(self.M_BLOCK)
        N_BLOCK = cutlass.const_expr(self.N_BLOCK)

        # View the 3D tensors as 2D row-major matrices so each expert tile is
        # a 2D TMA tile.  Row coordinates are expert-major:
        #   remote row = expert_id * H + h
        #   buffer row = buffer_id * H + h
        Hp64 = cutlass.Int64(Hp)
        src_rows = E * H
        dst_rows = B * H
        gmem_src = cute.make_tensor(
            remote_expert_ptr,
            cute.make_layout((src_rows, Hp), stride=(Hp64, cutlass.Int64(1))),
        )
        gmem_dst = cute.make_tensor(
            prefetch_buf_ptr,
            cute.make_layout((dst_rows, Hp), stride=(Hp64, cutlass.Int64(1))),
        )
        experts = cute.make_tensor(experts_ptr, cute.make_layout((B,)))

        smem_tile_layout = cute.make_ordered_layout(
            (M_BLOCK, N_BLOCK),
            order=(1, 0),
        )
        cta_tiler = (M_BLOCK, N_BLOCK)
        tma_g2s, tma_src = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            gmem_src,
            smem_tile_layout,
            cta_tiler,
        )
        tma_s2g, tma_dst = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            gmem_dst,
            smem_tile_layout,
            cta_tiler,
        )

        smem_bytes = self._smem_bytes(self.stages)

        self.kernel(tma_g2s, tma_src, tma_s2g, tma_dst, experts).launch(
            grid=(self.num_sms, 1, 1),
            block=(self.NUM_THREADS, 1, 1),
            smem=smem_bytes,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tma_g2s: cute.CopyAtom,
        tma_src: cute.Tensor,
        tma_s2g: cute.CopyAtom,
        tma_dst: cute.Tensor,
        experts: cute.Tensor,
    ):
        H = cutlass.const_expr(self.H)
        Hp = cutlass.const_expr(self.Hp)
        B = cutlass.const_expr(self.B)
        stages = cutlass.const_expr(self.stages)
        num_sms = cutlass.const_expr(self.num_sms)
        M_BLOCK = cutlass.const_expr(self.M_BLOCK)
        N_BLOCK = cutlass.const_expr(self.N_BLOCK)
        TILE_ELEMS = cutlass.const_expr(M_BLOCK * N_BLOCK)
        TILE_BYTES = cutlass.const_expr(TILE_ELEMS * self.elem_bytes)
        MTILES = cutlass.const_expr(H // M_BLOCK)
        NTILES = cutlass.const_expr(Hp // N_BLOCK)
        TILES_PER_EXPERT = cutlass.const_expr(MTILES * NTILES)

        bidx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        smem = utils.SmemAllocator()
        load_mbar = smem.allocate_array(Int64, num_elems=2 * stages)
        exp_tab = smem.allocate_tensor(Int32, cute.make_layout((B,)), byte_alignment=4)
        slot_tab = smem.allocate_tensor(Int32, cute.make_layout((B,)), byte_alignment=4)
        stage_smem = smem.allocate_tensor(
            self.elem_ty,
            cute.make_ordered_layout(
                (M_BLOCK, N_BLOCK, stages),
                order=(1, 0, 2),
            ),
            byte_alignment=128,
        )

        cta_tiler = (M_BLOCK, N_BLOCK)
        smem_for_tma = cute.group_modes(stage_smem, 0, 2)
        src_tiles = cute.zipped_divide(tma_src, cta_tiler)
        dst_tiles = cute.zipped_divide(tma_dst, cta_tiler)
        tSsS, tSgS = cpasync.tma_partition(
            tma_g2s,
            0,
            cute.make_layout(1),
            smem_for_tma,
            src_tiles,
        )
        tDsD, tDgD = cpasync.tma_partition(
            tma_s2g,
            0,
            cute.make_layout(1),
            smem_for_tma,
            dst_tiles,
        )

        load_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=load_mbar,
            num_stages=stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=TILE_BYTES,
        )

        # Compact the sparse plan so the tile loops only visit active slots
        # instead of skipping over B * TILES_PER_EXPERT indices per CTA.
        # Every thread runs the same scan and writes identical values, so the
        # tables it reads back are its own writes and no block barrier is
        # needed before the warp split.
        n_active = Int32(0)
        for b in cutlass.range_constexpr(B):
            e = experts[b]
            if e >= Int32(0):
                exp_tab[n_active] = e
                slot_tab[n_active] = Int32(b)
                n_active += Int32(1)
        total_tiles = n_active * TILES_PER_EXPERT

        if warp_idx == self.PRODUCER_WARP:
            load_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, stages
            )
            for tile in cutlass.range(bidx, total_tiles, num_sms, unroll=1):
                i = tile // TILES_PER_EXPERT
                rem = tile % TILES_PER_EXPERT
                mt = rem // NTILES
                nt = rem % NTILES

                expert = exp_tab[i]
                load_pipe.producer_acquire(load_state)

                src_mt = expert * MTILES + mt
                cute.copy(
                    tma_g2s,
                    tSgS[(None, (src_mt, nt))],
                    tSsS[(None, load_state.index)],
                    tma_bar_ptr=load_pipe.producer_get_barrier(load_state),
                )

                load_state.advance()

        elif warp_idx == self.CONSUMER_WARP:
            use_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, stages
            )
            rel_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, stages
            )
            issued = Int32(0)
            for tile in cutlass.range(bidx, total_tiles, num_sms, unroll=1):
                i = tile // TILES_PER_EXPERT
                rem = tile % TILES_PER_EXPERT
                mt = rem // NTILES
                nt = rem % NTILES

                b = slot_tab[i]
                load_pipe.consumer_wait(use_state)

                dst_mt = b * MTILES + mt
                cute.copy(
                    tma_s2g,
                    tDsD[(None, use_state.index)],
                    tDgD[(None, (dst_mt, nt))],
                )
                cute.arch.cp_async_bulk_commit_group()

                use_state.advance()
                issued += Int32(1)

                # Release a stage as soon as its smem read (the S2G store)
                # has drained, holding at most one store group in flight.
                # Lagging by stages-1 groups instead would keep stages-1
                # slots hostage and cap the producer at ~1 in-flight load,
                # making per-SM bandwidth tile_bytes / remote latency.
                if issued >= Int32(2):
                    cute.arch.cp_async_bulk_wait_group(1, read=True)
                    load_pipe.consumer_release(rel_state)
                    rel_state.advance()

            cute.arch.cp_async_bulk_wait_group(0, read=True)


@functools.lru_cache(maxsize=None)
def _max_smem_per_block_optin(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).shared_memory_per_block_optin


@functools.lru_cache(maxsize=None)
def _get_compiled(
    E: int,
    H: int,
    Hp: int,
    B: int,
    num_sms: int,
    device_index: int,
    torch_dtype: torch.dtype,
):
    elem_ty, elem_bytes = _ELEM_TYPES[torch_dtype]
    smem_budget = _max_smem_per_block_optin(device_index) - 1024
    kernel = PrefetchKernel(
        E=E,
        H=H,
        Hp=Hp,
        B=B,
        num_sms=num_sms,
        smem_budget=smem_budget,
        elem_ty=elem_ty,
        elem_bytes=elem_bytes,
    )

    data_ptr = make_ptr(elem_ty, 0, cute.AddressSpace.gmem, assumed_align=16)
    i32_ptr = make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=4)
    stream_arg = cuda.CUstream(0)

    return cute.compile(
        kernel,
        data_ptr,
        data_ptr,
        i32_ptr,
        stream_arg,
    )


def prefetch_retile_nbytes(per_expert_nbytes: int) -> int:
    """Round a per-expert byte count up to what ``retile_for_prefetch`` needs.
    """
    tile = PrefetchKernel.M_BLOCK * PrefetchKernel.N_BLOCK
    return (per_expert_nbytes + tile - 1) // tile * tile


def retile_for_prefetch(t: torch.Tensor) -> torch.Tensor:
    """View a contiguous ``[N, ...]`` expert tensor as ``[N, 128, X]`` uint8.
    """
    assert t.is_contiguous(), "retile_for_prefetch requires a contiguous tensor"
    n = int(t.shape[0])
    per_expert = t.nbytes // n if n else 0
    tile = PrefetchKernel.M_BLOCK * PrefetchKernel.N_BLOCK
    assert per_expert % tile == 0, (
        f"retile_for_prefetch: per-expert extent {per_expert} bytes is not a "
        f"multiple of {tile}; allocate {prefetch_retile_nbytes(per_expert)} "
        f"bytes per expert instead"
    )
    return t.view(torch.uint8).reshape(n, PrefetchKernel.M_BLOCK, -1)


def launch_prefetch(
    remote_expert: torch.Tensor,
    prefetch_buffers: torch.Tensor,
    experts_to_copy: torch.Tensor,
    num_sms: int,
):
    """Launch remote expert prefetch.

    Args:
        remote_expert: contiguous tensor shaped [E, H, H'].  dtype must be one
            of ``_ELEM_TYPES`` -- the copy is type-agnostic, so packed MXFP4
            (uint8) and ue8m0 scales (uint8) go through the same path as bf16.
        prefetch_buffers: contiguous tensor shaped [B, H, H'], same dtype.
        experts_to_copy: contiguous int32 tensor shaped [B].  Entries are
            expert ids in [0, E), or -1 for unused slots.  Unused slots are
            not written by this kernel.
        num_sms: number of persistent CTAs to launch.
    """
    if prefetch_buffers.numel() == 0 or experts_to_copy.numel() == 0:
        return

    dtype = remote_expert.dtype
    assert dtype in _ELEM_TYPES, (
        f"prefetch: unsupported dtype {dtype}; "
        f"supported: {sorted(str(d) for d in _ELEM_TYPES)}"
    )
    assert remote_expert.is_contiguous(), \
        "remote_expert must be contiguous [E, H, H']"
    assert prefetch_buffers.dtype == dtype and prefetch_buffers.is_contiguous(), \
        f"prefetch_buffers must be contiguous [B, H, H'] with dtype {dtype}, " \
        f"got {prefetch_buffers.dtype}"
    assert experts_to_copy.dtype == torch.int32 and experts_to_copy.is_contiguous(), \
        "experts_to_copy must be contiguous int32 [B]"
    assert remote_expert.ndim == 3, \
        f"remote_expert must have rank 3, got shape={tuple(remote_expert.shape)}"
    assert prefetch_buffers.ndim == 3, \
        f"prefetch_buffers must have rank 3, got shape={tuple(prefetch_buffers.shape)}"

    E, H, Hp = (int(x) for x in remote_expert.shape)
    B, out_H, out_Hp = (int(x) for x in prefetch_buffers.shape)
    assert out_H == H and out_Hp == Hp, \
        f"prefetch_buffers shape {tuple(prefetch_buffers.shape)} incompatible with remote_expert {tuple(remote_expert.shape)}"
    assert experts_to_copy.numel() == B, \
        f"experts_to_copy length {experts_to_copy.numel()} must equal B={B}"
    assert H % PrefetchKernel.M_BLOCK == 0 and Hp % PrefetchKernel.N_BLOCK == 0, \
        f"H and H' must be multiples of ({PrefetchKernel.M_BLOCK}, {PrefetchKernel.N_BLOCK}), got ({H}, {Hp})"
    assert isinstance(num_sms, int) and num_sms > 0, \
        f"num_sms must be a positive int, got {num_sms}"

    device_index = prefetch_buffers.device.index
    assert device_index is not None, "prefetch_buffers must be a CUDA tensor"
    assert experts_to_copy.device.index == device_index, \
        "experts_to_copy must be on the same CUDA device as prefetch_buffers"

    compiled = _get_compiled(E, H, Hp, B, int(num_sms), int(device_index), dtype)

    elem_ty, _ = _ELEM_TYPES[dtype]
    src_ptr = make_ptr(
        elem_ty,
        remote_expert.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    dst_ptr = make_ptr(
        elem_ty,
        prefetch_buffers.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    experts_ptr = make_ptr(
        Int32,
        experts_to_copy.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compiled(src_ptr, dst_ptr, experts_ptr, stream)
