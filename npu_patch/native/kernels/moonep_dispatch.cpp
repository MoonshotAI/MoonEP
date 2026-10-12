// MoonEP-style plan-driven dispatch kernel.
// UBMEM implementation — uses SHMEM symmetric window and DataCopyPad + UB/MTE.
// No URMA/UDMA, no zero-fill.
//
// Architecture: A2/A5, fixed kMoonEPAivNum AIV cores.
//
// Single-kernel design:
//   Phase 1 (Scatter): cores split routed slots N=S*K.
//       For each routed slot, decode dst -> (dest_rank, loff).
//       dst >= 0: SHMEM MTE put hidden row + write weight.
//       dst <  0: write weight only (duplicate hidden).
//   Phase 2 (Plan epilogue): expand duplicate hidden rows from the plan-provided
//       dup_groups / dup_loffs / dup_counts directly in the local SHMEM tensor.

#include "kernel_operator.h"
#include "ascend_moonep/kernels/types.h"
#include "shmem.h"

using namespace AscendC;

using bf16_t = bfloat16_t;
using ascend_moonep::kMoonEPAivNum;

// UB buffer sizing: one bf16 hidden row, weight ping-pong, and route metadata.
// Safety margin: allocate up to 128KB for hidden row (H <= 65536).
static constexpr uint32_t kUbHiddenRowBytes = 128 * 1024;
// DataCopyParams::blockLen is uint16_t. Keep the largest 32-byte-aligned
// payload it can represent so staged GM/UB copies need the fewest MTE tiles.
static constexpr uint32_t kDispatchMaxDataCopyPadBytes = 65504;
static constexpr uint32_t kMaxBatchedWeights = 16;
static constexpr uint32_t kWeightAlignedSlotBytes = 512;
static constexpr uint32_t kWeightBatchStageBytes =
    kMaxBatchedWeights * kWeightAlignedSlotBytes;
static constexpr uint32_t kWeightStagingBytes = 2 * kWeightBatchStageBytes;
static constexpr uint32_t kDstStagingOffset = kUbHiddenRowBytes + kWeightStagingBytes;
#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510
static constexpr uint32_t kDstStagingBytes = 64 * 1024;
#else
static constexpr uint32_t kDstStagingBytes = 32 * 1024;
#endif
static constexpr uint32_t kDispatchUbBytes = kDstStagingOffset + kDstStagingBytes;
static constexpr uint32_t kScatterHiddenStageBytes = kUbHiddenRowBytes / 2;
static constexpr uint32_t kDupGroupFields = 3;
static constexpr uint32_t kEpilogueGroupStagingOffset = kUbHiddenRowBytes;
static constexpr uint32_t kEpilogueGroupStagingBytes = kWeightStagingBytes;
static constexpr uint32_t kEpilogueLoffStagingOffset = kDstStagingOffset;
static constexpr uint32_t kEpilogueLoffStagingBytes = kDstStagingBytes;
static constexpr uint32_t kEpilogueGroupsPerTile =
    kEpilogueGroupStagingBytes / (kDupGroupFields * sizeof(int32_t));
static_assert(2 * kDispatchMaxDataCopyPadBytes <= kUbHiddenRowBytes,
              "MoonEP GM/UB ping-pong buffers exceed reserved UB");
static_assert(2 * kScatterHiddenStageBytes == kUbHiddenRowBytes,
              "MoonEP scatter ping-pong buffers must fill the hidden UB region");
static_assert(kDstStagingBytes % sizeof(int32_t) == 0,
              "MoonEP dst staging must contain whole int32 elements");
static_assert(kEpilogueGroupsPerTile > 0,
              "MoonEP epilogue group staging must hold at least one group");

__forceinline__ __aicore__ void DispatchSplitRange(
    uint32_t total,
    uint32_t parts,
    uint32_t part_idx,
    uint32_t& start,
    uint32_t& end) {
    if (total == 0 || parts == 0 || part_idx >= parts) {
        start = 0;
        end = 0;
        return;
    }
    uint32_t base = total / parts;
    uint32_t rem = total - base * parts;
    start = part_idx * base + ((part_idx < rem) ? part_idx : rem);
    end = start + base + ((part_idx < rem) ? 1U : 0U);
}

template <typename T>
__forceinline__ __aicore__ LocalTensor<T> DispatchMakeUbTensor(
    LocalTensor<uint8_t> ub_base,
    uint32_t byte_offset,
    uint32_t elements) {
    LocalTensor<T> tensor = ub_base[byte_offset].template ReinterpretCast<T>();
    tensor.address_.logicPos = static_cast<uint8_t>(TPosition::VECIN);
    tensor.address_.dataLen = elements * sizeof(T);
    return tensor;
}

template <typename T>
__forceinline__ __aicore__ void DispatchCopyGmToUb(
    LocalTensor<T> ub,
    const __gm__ T* gm,
    uint32_t elements,
    TEventID event_id) {
    uint32_t copied = 0;
    while (copied < elements) {
        uint32_t chunk_elements = elements - copied;
        uint32_t max_chunk_elements = kDispatchMaxDataCopyPadBytes / sizeof(T);
        if (chunk_elements > max_chunk_elements) {
            chunk_elements = max_chunk_elements;
        }
        uint32_t chunk_bytes = chunk_elements * sizeof(T);
        GlobalTensor<T> srcGM;
        srcGM.SetGlobalBuffer(const_cast<__gm__ T*>(gm) + copied, chunk_elements);
        DataCopyParams cp {1, static_cast<uint16_t>(chunk_bytes), 0, 0};
        DataCopyPad(ub[copied], srcGM, cp, DataCopyPadParams{});
        SetFlag<HardEvent::MTE2_MTE3>(event_id);
        WaitFlag<HardEvent::MTE2_MTE3>(event_id);
        copied += chunk_elements;
    }
}

template <typename T>
__forceinline__ __aicore__ void DispatchCopyGmToUbForScalar(
    LocalTensor<T> ub,
    const __gm__ T* gm,
    uint32_t elements) {
    uint32_t copied = 0;
    while (copied < elements) {
        uint32_t chunk_elements = elements - copied;
        uint32_t max_chunk_elements = kDispatchMaxDataCopyPadBytes / sizeof(T);
        if (chunk_elements > max_chunk_elements) {
            chunk_elements = max_chunk_elements;
        }
        uint32_t chunk_bytes = chunk_elements * sizeof(T);
        GlobalTensor<T> srcGM;
        srcGM.SetGlobalBuffer(const_cast<__gm__ T*>(gm) + copied, chunk_elements);
        DataCopyParams cp {1, static_cast<uint16_t>(chunk_bytes), 0, 0};
        DataCopyPad(ub[copied], srcGM, cp, DataCopyPadParams{});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        copied += chunk_elements;
    }
}

template <typename T>
__forceinline__ __aicore__ void DispatchIssueUbToGm(
    __gm__ T* gm,
    LocalTensor<T> ub,
    uint32_t elements) {
    uint32_t copied = 0;
    while (copied < elements) {
        uint32_t chunk_elements = elements - copied;
        uint32_t max_chunk_elements = kDispatchMaxDataCopyPadBytes / sizeof(T);
        if (chunk_elements > max_chunk_elements) {
            chunk_elements = max_chunk_elements;
        }
        uint32_t chunk_bytes = chunk_elements * sizeof(T);
        GlobalTensor<T> dstGM;
        dstGM.SetGlobalBuffer(gm + copied, chunk_elements);
        DataCopyParams cp {1, static_cast<uint16_t>(chunk_bytes), 0, 0};
        DataCopyPad(dstGM, ub[copied], cp);
        copied += chunk_elements;
    }
}

template <typename T>
__forceinline__ __aicore__ __ubuf__ T* DispatchUbPtr(LocalTensor<T> ub) {
    return reinterpret_cast<__ubuf__ T*>(static_cast<uintptr_t>(ub.address_.bufferAddr));
}

__forceinline__ __aicore__ void DispatchCopyWeightsToAlignedUb(
    LocalTensor<float> ub,
    const __gm__ float* gm,
    uint32_t elements,
    TEventID event_id) {
    GlobalTensor<float> srcGM;
    srcGM.SetGlobalBuffer(const_cast<__gm__ float*>(gm), elements);
    // Each 4-byte GM block is placed at the start of a 512-byte-aligned UB slot.
    constexpr uint16_t kWeightDstStrideBlocks =
        static_cast<uint16_t>(kWeightAlignedSlotBytes / 32U - 1U);
    DataCopyParams cp {
        static_cast<uint16_t>(elements),
        static_cast<uint16_t>(sizeof(float)),
        0,
        kWeightDstStrideBlocks};
    DataCopyPad(ub, srcGM, cp, DataCopyPadParams{});
    SetFlag<HardEvent::MTE2_MTE3>(event_id);
    WaitFlag<HardEvent::MTE2_MTE3>(event_id);
}

template <typename T>
__forceinline__ __aicore__ void DispatchIssueShmemUbPut(
    __gm__ T* symmetric_dst,
    LocalTensor<T> ub,
    uint32_t elements,
    int32_t pe,
    TEventID event_id) {
    aclshmemx_mte_put_nbi(
        symmetric_dst,
        DispatchUbPtr(ub),
        elements,
        pe,
        event_id);
}

template <typename T>
__forceinline__ __aicore__ void DispatchIssueShmemMtePut(
    __gm__ T* symmetric_dst,
    const __gm__ T* local_src,
    LocalTensor<T> ub,
    uint32_t ub_bytes,
    uint32_t elements,
    int32_t pe,
    TEventID event_id) {
    aclshmemx_mte_put_nbi(
        symmetric_dst,
        const_cast<__gm__ T*>(local_src),
        DispatchUbPtr(ub),
        ub_bytes,
        elements,
        pe,
        event_id);
}

template <typename T>
__forceinline__ __aicore__ void DispatchShmemMtePut(
    __gm__ T* symmetric_dst,
    const __gm__ T* local_src,
    LocalTensor<T> ub,
    uint32_t ub_bytes,
    uint32_t elements,
    int32_t pe,
    TEventID event_id) {
    DispatchIssueShmemMtePut(
        symmetric_dst, local_src, ub, ub_bytes, elements, pe, event_id);
    SetFlag<HardEvent::MTE3_MTE2>(event_id);
    WaitFlag<HardEvent::MTE3_MTE2>(event_id);
}

// ============================================================
// Phase 1: Scatter — SHMEM MTE put path
// ============================================================

template <typename HiddenT>
__aicore__ void ScatterPhase(
    LocalTensor<uint8_t> ub_base,
    const __gm__ HiddenT* hidden_sh,      // [S, H]
    const __gm__ float* route_weights,    // [S, K]
    const __gm__ int32_t* dst,            // [S*K]
    GM_ADDR hidden_symmetric,             // SHMEM bf16 [NvS, H]
    GM_ADDR weights_symmetric,            // SHMEM fp32 [NvS]
    uint32_t S, uint32_t H, uint32_t K,
    uint32_t R,
    uint32_t NvS,
    bool with_weights,
    uint32_t core_idx, uint32_t core_num) {

    uint32_t N = S * K;
    uint32_t start = 0;
    uint32_t end = 0;
    DispatchSplitRange(N, core_num, core_idx, start, end);
    if (start >= N) return;

    // UB buffer for one hidden row (bf16)
    uint32_t row_elements = H;
    uint32_t row_bytes = row_elements * sizeof(HiddenT);
    if (row_bytes > kUbHiddenRowBytes) {
        return;
    }

    LocalTensor<HiddenT> localRow = DispatchMakeUbTensor<HiddenT>(
        ub_base, 0, kUbHiddenRowBytes / sizeof(HiddenT));
    LocalTensor<float> localWt = DispatchMakeUbTensor<float>(
        ub_base, kUbHiddenRowBytes, kWeightStagingBytes / sizeof(float));
    LocalTensor<int32_t> localDst = DispatchMakeUbTensor<int32_t>(
        ub_base, kDstStagingOffset, kDstStagingBytes / sizeof(int32_t));
    uint32_t route_count = end - start;
    bool stage_dst = route_count <= kDstStagingBytes / sizeof(int32_t);
    if (stage_dst) {
        DispatchCopyGmToUbForScalar(localDst, dst + start, route_count);
    }
    bool pipeline_hidden = row_bytes <= kScatterHiddenStageBytes;
    uint32_t hidden_stage_elements = kScatterHiddenStageBytes / sizeof(HiddenT);
    uint32_t hidden_stage_idx = 0;
    if (pipeline_hidden) {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
    bool batch_weights = with_weights && K <= kMaxBatchedWeights;
    uint32_t weight_stage_idx = 0;
    if (with_weights) {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
    }

    uint32_t weight_stage_elements = kWeightBatchStageBytes / sizeof(float);
    uint32_t weight_slot_elements = kWeightAlignedSlotBytes / sizeof(float);
    bool nvs_is_power_of_two = (NvS & (NvS - 1U)) == 0;
    uint32_t nvs_shift = 0;
    if (nvs_is_power_of_two) {
        uint32_t value = NvS;
        while (value > 1U) {
            value >>= 1U;
            ++nvs_shift;
        }
    }
    uint32_t nvs_mask = NvS - 1U;
    uint32_t i = start;
    while (i < end) {
        uint32_t token_id = i / K;
        uint32_t topk_id = i - token_id * K;
        uint32_t segment_count = K - topk_id;
        if (segment_count > end - i) {
            segment_count = end - i;
        }

        TEventID batch_event = weight_stage_idx == 0 ? EVENT_ID2 : EVENT_ID3;
        LocalTensor<float> batch_stage = localWt[
            weight_stage_idx * weight_stage_elements];
        if (batch_weights) {
            WaitFlag<HardEvent::MTE3_MTE2>(batch_event);
            DispatchCopyWeightsToAlignedUb(
                batch_stage,
                route_weights + i,
                segment_count,
                batch_event);
        }

        for (uint32_t segment_idx = 0; segment_idx < segment_count; ++segment_idx) {
            uint32_t route_idx = i + segment_idx;
            int32_t dst_val = stage_dst
                ? localDst.GetValue(route_idx - start)
                : dst[route_idx];

            bool should_copy_hidden = (dst_val >= 0);
            uint32_t raw_dst = should_copy_hidden
                ? static_cast<uint32_t>(dst_val)
                : static_cast<uint32_t>(-static_cast<int64_t>(dst_val) - 1);

            uint32_t dest_rank = nvs_is_power_of_two
                ? raw_dst >> nvs_shift
                : raw_dst / NvS;
            uint32_t loff = nvs_is_power_of_two
                ? raw_dst & nvs_mask
                : raw_dst - dest_rank * NvS;
            if (dest_rank < R) {
                if (should_copy_hidden) {
                    uint64_t hidden_off_bytes = static_cast<uint64_t>(loff) * H * sizeof(HiddenT);
                    if (pipeline_hidden) {
                        TEventID event_id = hidden_stage_idx == 0 ? EVENT_ID0 : EVENT_ID1;
                        LocalTensor<HiddenT> hidden_stage = localRow[
                            hidden_stage_idx * hidden_stage_elements];
                        WaitFlag<HardEvent::MTE3_MTE2>(event_id);
                        DispatchIssueShmemMtePut(
                            reinterpret_cast<__gm__ HiddenT*>(hidden_symmetric + hidden_off_bytes),
                            hidden_sh + static_cast<uint64_t>(token_id) * H,
                            hidden_stage,
                            kScatterHiddenStageBytes,
                            row_elements,
                            static_cast<int32_t>(dest_rank),
                            event_id);
                        SetFlag<HardEvent::MTE3_MTE2>(event_id);
                        hidden_stage_idx ^= 1U;
                    } else {
                        DispatchShmemMtePut(
                            reinterpret_cast<__gm__ HiddenT*>(hidden_symmetric + hidden_off_bytes),
                            hidden_sh + static_cast<uint64_t>(token_id) * H,
                            localRow,
                            kUbHiddenRowBytes,
                            row_elements,
                            static_cast<int32_t>(dest_rank),
                            EVENT_ID0);
                    }
                }

                if (with_weights) {
                    uint64_t weight_off_bytes = static_cast<uint64_t>(loff) * sizeof(float);
                    if (batch_weights) {
                        DispatchIssueShmemUbPut(
                            reinterpret_cast<__gm__ float*>(weights_symmetric + weight_off_bytes),
                            batch_stage[segment_idx * weight_slot_elements],
                            1,
                            static_cast<int32_t>(dest_rank),
                            batch_event);
                    } else {
                        TEventID event_id = weight_stage_idx == 0 ? EVENT_ID2 : EVENT_ID3;
                        LocalTensor<float> weight_stage = localWt[
                            weight_stage_idx * weight_stage_elements];
                        WaitFlag<HardEvent::MTE3_MTE2>(event_id);
                        DispatchIssueShmemMtePut(
                            reinterpret_cast<__gm__ float*>(weights_symmetric + weight_off_bytes),
                            route_weights + route_idx,
                            weight_stage,
                            kWeightBatchStageBytes,
                            1,
                            static_cast<int32_t>(dest_rank),
                            event_id);
                        SetFlag<HardEvent::MTE3_MTE2>(event_id);
                        weight_stage_idx ^= 1U;
                    }
                }
            }
        }

        if (batch_weights) {
            SetFlag<HardEvent::MTE3_MTE2>(batch_event);
            weight_stage_idx ^= 1U;
        }
        i += segment_count;
    }

    if (pipeline_hidden) {
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
    if (with_weights) {
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
    }
}

template <typename HiddenT>
__aicore__ void PlanEpiloguePhase(
    LocalTensor<uint8_t> ub_base,
    __gm__ HiddenT* hidden_buf_local,
    const __gm__ int32_t* dup_groups,
    const __gm__ int32_t* dup_loffs,
    const __gm__ int32_t* dup_counts,
    uint32_t H,
    uint32_t NvS,
    uint32_t core_idx,
    uint32_t core_num) {

    GlobalTensor<int32_t> cntGM;
    cntGM.SetGlobalBuffer(const_cast<__gm__ int32_t*>(dup_counts), 2);

    int32_t n_groups_i = cntGM.GetValue(0);
    int32_t n_dup_loffs_i = cntGM.GetValue(1);
    if (n_groups_i <= 0 || n_dup_loffs_i <= 0) {
        return;
    }
    uint32_t n_groups = static_cast<uint32_t>(n_groups_i);
    uint32_t n_dup_loffs = static_cast<uint32_t>(n_dup_loffs_i);
    if (n_groups > NvS) {
        n_groups = NvS;
    }
    if (n_dup_loffs > NvS) {
        n_dup_loffs = NvS;
    }

    uint32_t row_elements = H;
    uint32_t row_bytes = row_elements * sizeof(HiddenT);
    if (row_bytes > kUbHiddenRowBytes) {
        return;
    }

    uint32_t group_start = 0;
    uint32_t group_end = 0;
    DispatchSplitRange(n_groups, core_num, core_idx, group_start, group_end);
    if (group_start >= group_end) {
        return;
    }

    LocalTensor<HiddenT> localRow = DispatchMakeUbTensor<HiddenT>(
        ub_base, 0, kUbHiddenRowBytes / sizeof(HiddenT));
    LocalTensor<int32_t> groupStage = DispatchMakeUbTensor<int32_t>(
        ub_base,
        kEpilogueGroupStagingOffset,
        kEpilogueGroupStagingBytes / sizeof(int32_t));
    LocalTensor<int32_t> loffStage = DispatchMakeUbTensor<int32_t>(
        ub_base,
        kEpilogueLoffStagingOffset,
        kEpilogueLoffStagingBytes / sizeof(int32_t));
    bool pipeline_hidden = row_bytes <= kScatterHiddenStageBytes;
    uint32_t hidden_stage_elements = kScatterHiddenStageBytes / sizeof(HiddenT);
    GlobalTensor<int32_t> loffGM;
    loffGM.SetGlobalBuffer(const_cast<__gm__ int32_t*>(dup_loffs), NvS);

    for (uint32_t tile_start = group_start; tile_start < group_end;) {
        uint32_t tile_count = group_end - tile_start;
        if (tile_count > kEpilogueGroupsPerTile) {
            tile_count = kEpilogueGroupsPerTile;
        }
        DispatchCopyGmToUbForScalar(
            groupStage,
            dup_groups + static_cast<uint64_t>(tile_start) * kDupGroupFields,
            tile_count * kDupGroupFields);

        uint32_t loff_span_start = n_dup_loffs;
        uint32_t loff_span_end = 0;
        for (uint32_t local_gid = 0; local_gid < tile_count; ++local_gid) {
            uint32_t group_offset = local_gid * kDupGroupFields;
            int32_t primary_i = groupStage.GetValue(group_offset);
            int32_t dup_start_i = groupStage.GetValue(group_offset + 1);
            int32_t dup_n_i = groupStage.GetValue(group_offset + 2);
            if (primary_i < 0 || dup_start_i < 0 || dup_n_i <= 0) {
                continue;
            }
            uint32_t primary_loff = static_cast<uint32_t>(primary_i);
            uint32_t dup_start = static_cast<uint32_t>(dup_start_i);
            uint32_t dup_n = static_cast<uint32_t>(dup_n_i);
            if (primary_loff >= NvS || dup_start >= n_dup_loffs) {
                continue;
            }
            uint32_t available_dups = n_dup_loffs - dup_start;
            if (dup_n > available_dups) {
                dup_n = available_dups;
            }
            if (dup_start < loff_span_start) {
                loff_span_start = dup_start;
            }
            uint32_t dup_end = dup_start + dup_n;
            if (dup_end > loff_span_end) {
                loff_span_end = dup_end;
            }
        }

        uint32_t loff_span = loff_span_end > loff_span_start
            ? loff_span_end - loff_span_start
            : 0;
        bool stage_loffs = loff_span > 0 &&
            loff_span <= kEpilogueLoffStagingBytes / sizeof(int32_t);
        if (stage_loffs) {
            DispatchCopyGmToUbForScalar(
                loffStage, dup_loffs + loff_span_start, loff_span);
        }

        uint32_t hidden_stage_idx = 0;
        if (pipeline_hidden) {
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
        for (uint32_t local_gid = 0; local_gid < tile_count; ++local_gid) {
            uint32_t group_offset = local_gid * kDupGroupFields;
            int32_t primary_i = groupStage.GetValue(group_offset);
            int32_t dup_start_i = groupStage.GetValue(group_offset + 1);
            int32_t dup_n_i = groupStage.GetValue(group_offset + 2);
            if (primary_i < 0 || dup_start_i < 0 || dup_n_i <= 0) {
                continue;
            }
            uint32_t primary_loff = static_cast<uint32_t>(primary_i);
            uint32_t dup_start = static_cast<uint32_t>(dup_start_i);
            uint32_t dup_n = static_cast<uint32_t>(dup_n_i);
            if (primary_loff >= NvS || dup_start >= n_dup_loffs) {
                continue;
            }
            uint32_t available_dups = n_dup_loffs - dup_start;
            if (dup_n > available_dups) {
                dup_n = available_dups;
            }

            TEventID event_id = pipeline_hidden && hidden_stage_idx != 0
                ? EVENT_ID1
                : EVENT_ID0;
            LocalTensor<HiddenT> hidden_stage = pipeline_hidden
                ? localRow[hidden_stage_idx * hidden_stage_elements]
                : localRow;
            if (pipeline_hidden) {
                WaitFlag<HardEvent::MTE3_MTE2>(event_id);
            }
            DispatchCopyGmToUb(
                hidden_stage,
                hidden_buf_local + static_cast<uint64_t>(primary_loff) * H,
                row_elements,
                event_id);

            bool has_pending_writes = false;
            for (uint32_t j = 0; j < dup_n; ++j) {
                uint32_t loff_idx = dup_start + j;
                int32_t dst_i = stage_loffs
                    ? loffStage.GetValue(loff_idx - loff_span_start)
                    : loffGM.GetValue(loff_idx);
                if (dst_i < 0) {
                    continue;
                }
                uint32_t dst_loff = static_cast<uint32_t>(dst_i);
                if (dst_loff >= NvS) {
                    continue;
                }

                DispatchIssueUbToGm(
                    hidden_buf_local + static_cast<uint64_t>(dst_loff) * H,
                    hidden_stage,
                    row_elements);
                has_pending_writes = true;
            }
            if (pipeline_hidden) {
                SetFlag<HardEvent::MTE3_MTE2>(event_id);
                hidden_stage_idx ^= 1U;
            } else if (has_pending_writes) {
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        }

        if (pipeline_hidden) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
        tile_start += tile_count;
    }

    PipeBarrier<PIPE_ALL>();
}

// ============================================================
// AscendC kernel entry point
// ============================================================

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void moonep_dispatch_kernel(
    GM_ADDR hidden_sh_gm,         // bf16 [S, H]
    GM_ADDR route_weights_gm,     // fp32 [S, K]
    GM_ADDR dst_gm,               // int32 [S*K]
    GM_ADDR dup_groups_gm,        // int32 [NvS, 3]
    GM_ADDR dup_loffs_gm,         // int32 [NvS]
    GM_ADDR dup_counts_gm,        // int32 [2]
    GM_ADDR hidden_buf_gm,        // SHMEM bf16 [NvS, H]
    GM_ADDR route_weights_buf_gm, // SHMEM fp32 [NvS]
    ascend_moonep::MoonEPDispatchTilingData tiling)
{
    uint32_t S = tiling.S;
    uint32_t H = tiling.H;
    uint32_t K = tiling.K;
    uint32_t NvS = S * K;
    uint32_t with_weights = tiling.with_weights;

    uint32_t core_idx = get_block_idx();

    LocalTensor<uint8_t> ub_base;
    ub_base.address_.logicPos = static_cast<uint8_t>(TPosition::VECIN);
    ub_base.address_.bufferAddr = 0;
    ub_base.address_.dataLen = kDispatchUbBytes;

    const __gm__ bf16_t* hidden_sh = reinterpret_cast<const __gm__ bf16_t*>(hidden_sh_gm);
    const __gm__ float* rw = reinterpret_cast<const __gm__ float*>(route_weights_gm);
    const __gm__ int32_t* dst_ptr = reinterpret_cast<const __gm__ int32_t*>(dst_gm);
    const __gm__ int32_t* dgrp = reinterpret_cast<const __gm__ int32_t*>(dup_groups_gm);
    const __gm__ int32_t* dloff = reinterpret_cast<const __gm__ int32_t*>(dup_loffs_gm);
    const __gm__ int32_t* dcnt = reinterpret_cast<const __gm__ int32_t*>(dup_counts_gm);
    ScatterPhase<bf16_t>(
        ub_base, hidden_sh, rw, dst_ptr,
        hidden_buf_gm, route_weights_buf_gm,
        S, H, K, tiling.R, NvS,
        with_weights != 0,
        core_idx, kMoonEPAivNum);

    // Completes the scatter RMA updates and synchronizes all vector cores on
    // every rank before the local duplicate expansion reads the receive buffer.
    aclshmemx_barrier_all_vec();

    __gm__ bf16_t* hbuf = reinterpret_cast<__gm__ bf16_t*>(hidden_buf_gm);

    PlanEpiloguePhase<bf16_t>(
        ub_base, hbuf, dgrp, dloff, dcnt,
        H, NvS, core_idx, kMoonEPAivNum);
}


// ============================================================
// Host-side launcher (C-linkage)
// ============================================================

extern "C" __attribute__((visibility("default")))
void moonep_dispatch_kernel_do(
    void* stream,
    uint8_t* hidden_sh,
    uint8_t* route_weights,
    uint8_t* dst,
    uint8_t* dup_groups,
    uint8_t* dup_loffs,
    uint8_t* dup_counts,
    uint8_t* hidden_buf,
    uint8_t* route_weights_buf,
    uint8_t* tiling_host)
{
    auto* td = reinterpret_cast<ascend_moonep::MoonEPDispatchTilingData*>(tiling_host);

    moonep_dispatch_kernel<<<kMoonEPAivNum, nullptr, stream>>>(
        hidden_sh,
        route_weights,
        dst,
        dup_groups,
        dup_loffs,
        dup_counts,
        hidden_buf,
        route_weights_buf,
        *td);
}
