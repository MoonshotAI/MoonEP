"""
MoonEP Dispatch Epilogue — CuTe DSL implementation.

In-place duplicate expansion on the local NVL shard after dispatch: for each
duplicate group, the primary row (the only copy dispatch sent over NVLink) is
staged once through smem and stored to every duplicate slot of the same
shard. No user tensor is involved — the ``zero_copy=False`` boundary copies
the whole shard afterwards with a plain ``tensor.copy_`` (master-style), and
``zero_copy=True`` returns the shard view directly. Padding rows are already
zero-filled by the dispatch zero warp (``plan.zero_fill_ranges``), and the
per-topk route weights are already scattered by the dispatch consumer.

Runs entirely on local memory — no cross-rank communication; it must be
launched after the dispatch kernel on the same stream (dispatch's exit
cross_rank_barrier publishes the remote NVL writes it reads). Its outputs are
consumed locally (GEMM) or published later by combine's entry barrier, so it
needs no cross-rank barrier of its own.
"""

import functools

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import BFloat16, Int32, Int64, Uint8
from cutlass.cute.runtime import make_ptr

from moonep._common import (
    cp_async_bulk_g2s,
    cp_async_bulk_s2g,
    pdl_wait_predecessor,
)
from moonep.planning import MoonEPCommPlan
from moonep.constants import FP8_TILE_SIZE


# ============================================================================
# Dispatch epilogue kernel
# ============================================================================

class DispatchEpilogueKernel:
    """Round-robin duplicate-group expansion on the local NVL shard.

    Two warps per CTA mirror the dispatch data path: warp 0 stages a batch of
    ``B`` primary rows G2S into each smem pipeline stage (one mbarrier
    handshake per batch), warp 1 stores every staged row S2G to all duplicate
    slots of its group (the primary row is read from gmem exactly once
    however many copies it fans out to).

    The plan-owned dedup structures drive all work: ``dup_groups`` lists the
    ``(primary_loff, dup_start, dup_n)`` headers, ``dup_loffs`` the compact
    flat duplicate slots, and ``dup_counts[0]`` the valid ``dup_groups``
    prefix length (read on device — no host synchronization).

    Blocks take group batches round-robin (batch ``i`` = groups
    ``[i*B, i*B+B)`` -> block ``i % num_sms``), so the launch grid must
    equal ``num_sms``.
    """

    num_threads = 64
    PRODUCER_WARP = 0
    CONSUMER_WARP = 1

    # Handshake amortization dominates and pipeline depth beyond 2 is
    # measurably irrelevant, so pick the largest per-stage group batch B
    # that still double-buffers.
    _B_CANDIDATES = (32, 16, 8, 4, 2, 1)
    _MIN_STAGES = 2

    def __init__(
        self,
        H: int,
        NvS: int,
        num_sms: int,
        smem_budget: int,
        pdl_launch: bool,
        use_fp8: bool = False,
    ):
        self.H = H
        self.use_fp8 = use_fp8
        self.elem_type = Uint8 if use_fp8 else BFloat16
        self.elem_bytes = 1 if use_fp8 else 2
        self.NvS = NvS
        self.num_sms = num_sms
        self.pdl_launch = pdl_launch
        # FP8 scale rows that are 16-B multiples are staged by TMA with their
        # payload rows, so the consumer reads primary scales from smem.
        self.scale_bulk = use_fp8 and (H // FP8_TILE_SIZE * 4) % 16 == 0
        self.scale_row_bytes = H // FP8_TILE_SIZE * 4 if self.scale_bulk else 0
        self.B, self.stages = self._pick_geometry(
            H, smem_budget, self.elem_bytes, self.scale_row_bytes,
        )
        # Consumer reads the B batch headers with one lane each + shuffle.
        assert self.B <= 32, "batch size B must fit in one warp"
        if self.stages == 0:
            raise RuntimeError(
                f"dispatch epilogue: H={H} too large for per-block smem budget "
                f"{smem_budget} B (need at least {self._smem_bytes(H, 2, 1, self.elem_bytes, self.scale_row_bytes)} B)"
            )

    @staticmethod
    def _smem_bytes(
        H: int, stages: int, B: int, elem_bytes: int = 2, scale_row_bytes: int = 0,
    ) -> int:
        # stage_smem (payload, B rows x kStages deep) + optional scale rows +
        # 2 mbar i64 per stage (full+empty), padded like the dispatch kernel,
        # plus 256 B cutlass headroom.
        def _round_up(n: int, a: int) -> int:
            return (n + a - 1) // a * a
        return (
            _round_up(stages * B * H * elem_bytes, 128)
            + _round_up(stages * B * scale_row_bytes, 16)
            + _round_up(stages * 2 * 8, 16)
            + 256
        )

    @classmethod
    def _pick_stages(
        cls, H: int, smem_budget: int, B: int, elem_bytes: int = 2,
        scale_row_bytes: int = 0,
    ) -> int:
        for s in (16, 14, 12, 10, 8, 6, 4, 2):
            if cls._smem_bytes(H, s, B, elem_bytes, scale_row_bytes) <= smem_budget:
                return s
        return 0

    @classmethod
    def _pick_geometry(
        cls, H: int, smem_budget: int, elem_bytes: int = 2, scale_row_bytes: int = 0,
    ) -> tuple[int, int]:
        for b in cls._B_CANDIDATES:
            s = cls._pick_stages(H, smem_budget, b, elem_bytes, scale_row_bytes)
            if s >= cls._MIN_STAGES:
                return b, s
        return 1, cls._pick_stages(H, smem_budget, 1, elem_bytes, scale_row_bytes)

    # ------------------------------------------------------------------ host

    @cute.jit
    def __call__(
        self,
        hidden_buf_local_ptr: cute.Pointer,  # bf16 or fp8 [NvS, H] local NVL shard
        scales_buf_local_ptr: cute.Pointer,  # int32 bits of fp32 [NvS, H/128]
        dup_groups_ptr: cute.Pointer,        # int32 [NvS, 3]
        dup_loffs_ptr: cute.Pointer,         # int32 [NvS,]
        dup_counts_ptr: cute.Pointer,        # int32 [2,]
        stream: cuda.CUstream,
    ):
        H = cutlass.const_expr(self.H)
        NvS = cutlass.const_expr(self.NvS)

        hidden = cute.make_tensor(
            hidden_buf_local_ptr,
            cute.make_layout((NvS, H), stride=(Int64(H), Int64(1))),
        )
        num_scales = max(H // FP8_TILE_SIZE, 1)
        scales = cute.make_tensor(
            scales_buf_local_ptr,
            cute.make_layout((NvS, num_scales), stride=(Int64(num_scales), Int64(1))),
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

        smem_bytes = self._smem_bytes(
            H, self.stages, self.B, self.elem_bytes, self.scale_row_bytes,
        )

        self.kernel(
            hidden,
            scales,
            dup_groups_tensor,
            dup_loffs_tensor,
            dup_counts_tensor,
        ).launch(
            grid=(self.num_sms, 1, 1),
            block=(self.num_threads, 1, 1),
            smem=smem_bytes,
            stream=stream,
            use_pdl=self.pdl_launch,
        )

    # ----------------------------------------------------------------- kernel

    @cute.kernel
    def kernel(
        self,
        hidden: cute.Tensor,
        scales: cute.Tensor,
        dup_groups_tensor: cute.Tensor,
        dup_loffs_tensor: cute.Tensor,
        dup_counts_tensor: cute.Tensor,
    ):
        H = cutlass.const_expr(self.H)
        H_BYTES = cutlass.const_expr(H * self.elem_bytes)
        SCALE_BYTES = cutlass.const_expr(self.scale_row_bytes)
        NUM_SCALES = cutlass.const_expr(H // FP8_TILE_SIZE)
        stages = cutlass.const_expr(self.stages)
        num_sms = cutlass.const_expr(self.num_sms)
        B = cutlass.const_expr(self.B)

        if cutlass.const_expr(self.pdl_launch):
            pdl_wait_predecessor()

        bidx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # ----- shared memory layout
        smem = utils.SmemAllocator()
        load_mbar = smem.allocate_array(Int64, num_elems=2 * stages)
        stage_smem = smem.allocate_tensor(
            self.elem_type,
            cute.make_ordered_layout((B, H, stages), order=(1, 0, 2)),
            byte_alignment=128,
        )
        if cutlass.const_expr(self.scale_bulk):
            scale_smem = smem.allocate_tensor(
                Int32,
                cute.make_ordered_layout((NUM_SCALES, B, stages), order=(0, 1, 2)),
                byte_alignment=16,
            )

        # Load pipeline (TMA G2S):
        #   Producer = warp 0 (1 effective thread, lane 0).
        #   Consumer = warp 1 (1 effective thread, lane 0).
        load_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=load_mbar,
            num_stages=stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=0,
        )

        n_groups = dup_counts_tensor[0]

        # ============================================
        # Warp 0 — G2S producer: one primary row per pipeline stage
        # ============================================
        if warp_idx == self.PRODUCER_WARP:
            load_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, stages
            )
            gi = Int32(bidx * B)
            lane = cute.arch.lane_idx()
            while gi < n_groups:
                n = Int32(B)
                if gi + n > n_groups:
                    n = n_groups - gi

                load_pipe.sync_object_empty.wait(
                    load_state.index, load_state.phase
                )
                # cp.async.bulk is single-thread; only lane 0 issues it.
                if lane == 0:
                    mbar = load_pipe.producer_get_barrier(load_state)
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        mbar, n * (H_BYTES + SCALE_BYTES),
                    )

                    for idx in cutlass.range(n, unroll=1):
                        loff = dup_groups_tensor[(gi + idx) * 3]
                        g_row_int = (
                            hidden.iterator + Int64(loff) * Int64(H)
                        ).toint()
                        s_row_int = (
                            stage_smem.iterator + load_state.index * B * H + idx * H
                        ).toint()
                        cp_async_bulk_g2s(
                            s_row_int.ir_value(),
                            g_row_int.ir_value(),
                            Int32(H_BYTES).ir_value(),
                            mbar.toint().ir_value(),
                        )
                        if cutlass.const_expr(self.scale_bulk):
                            g_scale_int = (
                                scales.iterator + Int64(loff) * Int64(NUM_SCALES)
                            ).toint()
                            s_scale_int = (
                                scale_smem.iterator
                                + (load_state.index * B + idx) * NUM_SCALES
                            ).toint()
                            cp_async_bulk_g2s(
                                s_scale_int.ir_value(),
                                g_scale_int.ir_value(),
                                Int32(SCALE_BYTES).ir_value(),
                                mbar.toint().ir_value(),
                            )
                load_state.advance()
                gi += Int32(num_sms * B)

        # ============================================
        # Warp 1 — S2G consumer: staged primary row -> every duplicate slot
        # ============================================
        elif warp_idx == self.CONSUMER_WARP:
            use_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, stages
            )
            rel_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, stages
            )

            issued = Int32(0)
            gi = Int32(bidx * B)
            lane = cute.arch.lane_idx()
            while gi < n_groups:
                n = Int32(B)
                if gi + n > n_groups:
                    n = n_groups - gi

                dup_start = Int32(0)
                dup_n = Int32(0)
                if lane < n:
                    dup_start = dup_groups_tensor[(gi + lane) * 3 + 1]
                    dup_n = dup_groups_tensor[(gi + lane) * 3 + 2]

                load_pipe.consumer_wait(use_state)
                for idx in cutlass.range(n, unroll=1):
                    cur_dup_start = cute.arch.shuffle_sync(dup_start, Int32(idx))
                    cur_dup_n = cute.arch.shuffle_sync(dup_n, Int32(idx))
                    if lane == 0:
                        s_row_int = (
                            stage_smem.iterator + use_state.index * B * H + idx * H
                        ).toint()
                        for k in cutlass.range(cur_dup_n, unroll=1):
                            slot = dup_loffs_tensor[cur_dup_start + k]
                            g_dup_int = (
                                hidden.iterator + Int64(slot) * Int64(H)
                            ).toint()
                            cp_async_bulk_s2g(
                                s_row_int.ir_value(),
                                g_dup_int.ir_value(),
                                Int32(H_BYTES).ir_value(),
                            )
                    if cutlass.const_expr(self.use_fp8):
                        if cutlass.const_expr(not self.scale_bulk):
                            primary = dup_groups_tensor[(gi + idx) * 3]
                        for tile in cutlass.range_constexpr((NUM_SCALES + 31) // 32):
                            col = tile * 32 + lane
                            if col < NUM_SCALES:
                                if cutlass.const_expr(self.scale_bulk):
                                    scale = scale_smem[col, idx, use_state.index]
                                else:
                                    scale = scales[primary, col]
                                for k in cutlass.range(cur_dup_n, unroll=1):
                                    slot = dup_loffs_tensor[cur_dup_start + k]
                                    scales[slot, col] = scale
                if lane == 0:
                    cute.arch.cp_async_bulk_commit_group()
                use_state.advance()

                # Throttle in-flight bulk_groups to <= STAGES-1.
                if issued >= Int32(stages - 1):
                    cute.arch.cp_async_bulk_wait_group(stages - 1)
                    load_pipe.consumer_release(rel_state)
                    rel_state.advance()
                issued += Int32(1)
                gi += Int32(num_sms * B)

            # Drain the trailing in-flight stores.
            cute.arch.cp_async_bulk_wait_group(0)


# ============================================================================
# Host launcher with per-shape compile cache
# ============================================================================

@functools.lru_cache(maxsize=None)
def _max_smem_per_block_optin(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).shared_memory_per_block_optin


def _check_epilogue_plan(
    ctx: dict,
    plan: MoonEPCommPlan,
    dev: torch.device,
) -> None:
    NvS = int(ctx['NvS'])
    assert isinstance(plan, MoonEPCommPlan)
    assert plan.NvS == NvS, f"plan.NvS must match ctx NvS={NvS}, got {plan.NvS}"

    def _check_tensor(t: torch.Tensor, name: str, shape: tuple[int, ...]) -> None:
        assert t.dtype == torch.int32 and t.is_contiguous(), \
            f"{name} must be contiguous int32"
        assert tuple(t.shape) == shape, \
            f"{name} must be shape {shape}, got {tuple(t.shape)}"
        assert t.device == dev, \
            f"{name} must be on {dev}, got {t.device}"

    _check_tensor(plan.dup_groups, "dup_groups", (NvS, 3))
    _check_tensor(plan.dup_loffs, "dup_loffs", (NvS,))
    _check_tensor(plan.dup_counts, "dup_counts", (2,))


@functools.lru_cache(maxsize=None)
def _get_compiled(
    H: int,
    NvS: int,
    num_sms: int,
    device_index: int,
    pdl_launch: bool,
    use_fp8: bool = False,
):
    smem_budget = _max_smem_per_block_optin(device_index) - 1024
    kernel = DispatchEpilogueKernel(
        H=H, NvS=NvS, num_sms=num_sms, smem_budget=smem_budget,
        pdl_launch=pdl_launch, use_fp8=use_fp8,
    )

    payload_ptr = make_ptr(Uint8 if use_fp8 else BFloat16, 0, cute.AddressSpace.gmem, assumed_align=16)
    i32_ptr = make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=16)
    stream_arg = cuda.CUstream(0)

    return cute.compile(
        kernel,
        payload_ptr,  # hidden_buf_local
        i32_ptr,      # scales_buf_local
        i32_ptr,   # dup_groups
        i32_ptr,   # dup_loffs
        i32_ptr,   # dup_counts
        stream_arg,
    )


def launch_dispatch_epilogue(
    ctx: dict,
    plan: MoonEPCommPlan,
    *,
    pdl_launch: bool = False,
    use_fp8: bool = False,
):
    """Expand duplicate rows in place on the local NVL shard.

    Must run after ``launch_dispatch`` on the same stream: dispatch's exit
    cross_rank_barrier publishes the remote NVL writes this kernel reads.
    Consumes the plan-owned ``dup_groups`` / ``dup_loffs`` / ``dup_counts``
    written by a fresh dispatch builder (reuse paths pass the saved tensors
    unchanged). Purely local — no cross-rank communication. Set use_fp8=True
    after an FP8 dispatch to expand both data and its fp32 scale rows.
    """
    H = int(ctx['H'])
    NvS = int(ctx['NvS'])
    num_sms = int(ctx['num_sms_dedup'])
    hidden = ctx['hidden_fp8_buf_local'] if use_fp8 else ctx['hidden_buf_local']
    scales = ctx['scales_buf_local'] if use_fp8 else plan.dst
    dev = hidden.device
    device_index = dev.index

    assert ctx['H'] % 8 == 0, "H must be multiple of 8 for 16-B bulk-copy alignment"
    assert hidden.dtype == (torch.float8_e4m3fn if use_fp8 else torch.bfloat16)
    assert hidden.is_contiguous() and hidden.is_cuda
    assert tuple(hidden.shape) == (NvS, H)
    if use_fp8:
        assert H % FP8_TILE_SIZE == 0
        assert scales.dtype == torch.float32 and scales.is_contiguous()
        assert tuple(scales.shape) == (NvS, H // FP8_TILE_SIZE)
        assert scales.device == dev
    assert device_index is not None, "dispatch epilogue requires a CUDA device"
    _check_epilogue_plan(ctx, plan, dev)

    epilogue_compiled = _get_compiled(H, NvS, num_sms, device_index, bool(pdl_launch), use_fp8)

    hbf_local_ptr = make_ptr(Uint8 if use_fp8 else BFloat16, hidden.data_ptr(),
                             cute.AddressSpace.gmem, assumed_align=16)
    scales_local_ptr = make_ptr(Int32, scales.data_ptr(),
                                cute.AddressSpace.gmem, assumed_align=16)
    dup_groups_ptr = make_ptr(Int32, plan.dup_groups.data_ptr(),
                              cute.AddressSpace.gmem, assumed_align=16)
    dup_loffs_ptr = make_ptr(Int32, plan.dup_loffs.data_ptr(),
                             cute.AddressSpace.gmem, assumed_align=16)
    dup_counts_ptr = make_ptr(Int32, plan.dup_counts.data_ptr(),
                              cute.AddressSpace.gmem, assumed_align=8)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    epilogue_compiled(
        hbf_local_ptr,
        scales_local_ptr,
        dup_groups_ptr,
        dup_loffs_ptr,
        dup_counts_ptr,
        stream,
    )
