// MoonEP-style plan-driven combine kernel.
// UBMEM implementation — uses the SHMEM symmetric window for publication and
// batched UDMA Get submission across two explicit QPs for the remote gather.
//
// Architecture: A5/Ascend950 (UDMA), fixed kCombineAivNum AIV cores.
//
// Inputs are already published in SHMEM symmetric allocations.
// Single-kernel design (inverse of moonep_dispatch):
//   Phase 1 (Prologue): for each dup group, fp32-sum primary + dups and
//       store the reduced row back to the primary loff (in-place on SHMEM).
//   Barrier: publish reduced primary rows to peers.
//   Phase 2 (Gather): dispatch-style group flow limits the readers admitted by
//       each source PE. One AIV owns a peer for the whole group and submits both
//       QPs for that peer. The reader returns done only after both QPs and the
//       optional weight GET are complete, then the source admits the next group.
//   Phase 3 (Reduce): fp32-accumulate staged primary rows into output [S,H].

#include "kernel_operator.h"
#include "ascend_moonep/kernels/types.h"
#include "shmem.h"

using namespace AscendC;

using bf16_t = bfloat16_t;

#define MOONEP_UDMA_CQE_DEBUG 0
#define MOONEP_UDMA_PROFILE_STATISTIC 0
static constexpr uint32_t kCombineAivNum =
    ascend_moonep::kMoonEPAivNum;
static constexpr uint32_t kCombineUdmaQpNum =
    ascend_moonep::kMoonEPUdmaCombineQpCount;
static_assert(kCombineUdmaQpNum == 2,
              "MoonEP weighted UDMA GET assignment requires exactly two QPs");
static constexpr uint32_t kCombineUbHiddenRowBytes = 128 * 1024;
static constexpr uint32_t kCombineAccBytesPerElement =
    sizeof(bf16_t) + 2 * sizeof(float);
static constexpr uint32_t kCombineAccTileAlignElements =
    32 / sizeof(bf16_t);
static constexpr uint32_t kCombineMaxAccTileElements =
    (kCombineUbHiddenRowBytes /
     kCombineAccBytesPerElement /
     kCombineAccTileAlignElements) *
    kCombineAccTileAlignElements;
static constexpr uint32_t kCombineAccTileBf16Bytes =
    kCombineMaxAccTileElements * sizeof(bf16_t);
static constexpr uint32_t kCombineAccTileFp32Bytes =
    kCombineMaxAccTileElements * sizeof(float);
static constexpr uint32_t kCombineAccTileBytes =
    kCombineAccTileBf16Bytes + 2 * kCombineAccTileFp32Bytes;
static constexpr uint32_t kCombineMaxDataCopyPadBytes = 65504;
static constexpr uint32_t kCombineWeightStageBytes = 32;
static constexpr uint32_t kCombineWeightStagingBytes = 2 * kCombineWeightStageBytes;
// Group credit caps the readers admitted by each source, so restore dispatch's
// 128-WQE doorbell batch without recreating the unbounded cross-node fan-in.
static constexpr uint32_t kCombineUdmaWqeBatchSize = 128;
static constexpr uint32_t kCombineUdmaWqeBytes = 64;
static constexpr uint32_t kCombineUdmaStagingBytes =
    kCombineUdmaWqeBatchSize * kCombineUdmaWqeBytes;
// Keep the grouped protocol aligned with dispatch: at most 16 readers are
// admitted per source in one group.
static constexpr uint32_t kCombineMaxGroupWidth = 16;
static constexpr uint32_t kCombineGroupSignalBytes = 512;
static constexpr uint32_t kCombineGroupSignalSlots = 2;
static constexpr uint32_t kCombineGrantSignalSlot = 0;
static constexpr uint32_t kCombineDoneSignalSlot = 1;
static constexpr uint64_t kCombineGroupMagicPrefix = 0x434F4D4200000000ULL;
// Match the published Dispatch/PUT vector selection granularity. The selection
// buffers are phase-local and may exceed the hidden-row accumulator, so place
// the persistent weight/WQE staging after the larger scratch requirement.
static constexpr uint32_t kCombineSlotChunkCount = 8192;
static constexpr uint32_t kCombineVectorSlotSelectCountAlign = 64;
static constexpr uint32_t kCombineSlotTensorBytes =
    kCombineSlotChunkCount * sizeof(int32_t);
static constexpr uint32_t kCombineSlotMaskBytes =
    ((kCombineSlotChunkCount / 8U + 31U) / 32U) * 32U;
static constexpr uint32_t kCombineSlotSelectBytes =
    4U * kCombineSlotTensorBytes + kCombineSlotMaskBytes;
static constexpr uint32_t kCombineWeightOutputStageBytes =
    kCombineSlotChunkCount * sizeof(float);
// Prologue / reduce full-row DB (UBMem style): 2 load + 2 store + fp32 acc/tmp.
static constexpr uint32_t kCombinePrologueGroupFields = 3;
static constexpr uint32_t kCombinePrologueEntryWindow = 1024;
static constexpr uint32_t kCombinePrologueMaxSrcs = 16;
static constexpr uint32_t kCombinePrologueDbBytesPerElement =
    4U * sizeof(bf16_t) + 2U * sizeof(float);
static constexpr uint32_t kCombinePrologueDbMaxBytes =
    kCombineUbHiddenRowBytes;
static constexpr uint32_t kCombineProloguePlanBytes =
    kCombinePrologueEntryWindow * kCombinePrologueGroupFields *
    sizeof(int32_t);
static constexpr uint32_t kCombineProloguePlanOffset =
    kCombinePrologueDbMaxBytes;
static constexpr uint32_t kCombinePrologueScratchBytes =
    kCombineProloguePlanOffset + kCombineProloguePlanBytes;
static constexpr uint32_t kCombinePhaseScratchBytes =
    kCombineSlotSelectBytes > kCombineUbHiddenRowBytes
        ? (kCombineSlotSelectBytes > kCombinePrologueScratchBytes
               ? kCombineSlotSelectBytes
               : kCombinePrologueScratchBytes)
        : (kCombineUbHiddenRowBytes > kCombinePrologueScratchBytes
               ? kCombineUbHiddenRowBytes
               : kCombinePrologueScratchBytes);
static constexpr uint32_t kCombineWeightStagingOffset =
    kCombinePhaseScratchBytes;
static constexpr uint32_t kCombineUdmaStagingOffset =
    kCombineWeightStagingOffset + kCombineWeightStagingBytes;
static constexpr uint32_t kCombineUbBytes = kCombineUdmaStagingOffset + kCombineUdmaStagingBytes;
// Keep each AIV's bandwidth and phase counters on a separate 64-byte line.
static constexpr uint32_t kCombineBandwidthStatsStride =
    64U / sizeof(int64_t);
static constexpr float kCombineCyclesPerMicrosecond = 1000.0F;
static constexpr uint32_t kCombineStatHiddenBytes = 0U;
static constexpr uint32_t kCombineStatGatherCycles = 1U;
static constexpr uint32_t kCombineStatHiddenSubmitCycles = 2U;
static constexpr uint32_t kCombineStatWeightGetCycles = 3U;
static constexpr uint32_t kCombineStatWeightSyncCycles = 4U;
static constexpr uint32_t kCombineStatWeightAssembleCycles = 5U;
static constexpr uint32_t kCombineStatHiddenQuietCycles = 6U;
static constexpr uint32_t kCombineStatFinalSyncCycles = 7U;
static_assert(kCombineBandwidthStatsStride > kCombineStatFinalSyncCycles,
              "MoonEP combine profile counters must fit one cache line");

struct CombineGatherResult {
    uint64_t hidden_bytes;
#if MOONEP_UDMA_PROFILE_STATISTIC
    uint64_t hidden_submit_cycles;
    uint64_t weight_get_cycles;
    uint64_t weight_sync_cycles;
    uint64_t weight_assemble_cycles;
    uint64_t hidden_quiet_cycles;
    uint64_t final_sync_cycles;
#endif
};
static_assert(kCombineMaxAccTileElements > 0,
              "MoonEP accumulation tile must contain at least one aligned block");
static_assert(kCombineAccTileBytes <= kCombineUbHiddenRowBytes,
              "MoonEP accumulation tile must fit the hidden UB region");
static_assert(kCombineSlotSelectBytes <= kCombinePhaseScratchBytes,
              "MoonEP vector slot-select buffers must fit the phase scratch region");
static_assert(2U * kCombineWeightOutputStageBytes + kCombineSlotTensorBytes <=
                  kCombinePhaseScratchBytes,
              "MoonEP direct weight gather buffers must fit phase scratch");
static_assert(kCombineUdmaStagingBytes >= ACLSHMEM_UDMA_MTE_STAGING_UB_SIZE,
              "MoonEP aggregate WQE staging must satisfy SHMEM's minimum size");
static_assert(
    kCombinePrologueDbMaxBytes <= kCombineUbHiddenRowBytes,
    "Prologue/reduce full-row double-buffer must fit the hidden UB region");
static_assert(
    kCombineProloguePlanOffset + kCombineProloguePlanBytes <=
        kCombinePhaseScratchBytes,
    "Prologue plan AoS must sit in phase scratch above the reduce DB");

__forceinline__ __aicore__ void CombineSplitRange(
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

// Match dispatch's 6:2 CLOS port ratio: each four route chunks use
// QP0, QP0, QP0, and QP1. A single AIV owns both QPs for a remote PE.
__forceinline__ __aicore__ uint32_t CombineGetChunkQpIndex(
    uint32_t chunk_index) {
    return chunk_index % 4U == 3U ? 1U : 0U;
}

__forceinline__ __aicore__ uint32_t CombineGetGroupWidth(uint32_t world_size) {
    return world_size < kCombineMaxGroupWidth
        ? world_size
        : kCombineMaxGroupWidth;
}

__forceinline__ __aicore__ uint32_t CombineGetGroupRankIndex(
    uint32_t rank,
    uint32_t world_size) {
    const uint32_t half_world_size = world_size / 2U;
    const uint32_t half_group_width = CombineGetGroupWidth(world_size) / 2U;
    return rank / half_world_size * half_group_width +
        rank % half_group_width;
}

// Same balanced rank permutation as dispatch: each group contains half of its
// peers from each node half. For R=16/group_width=8, every phase therefore has
// four local-half and four remote-half peers.
__forceinline__ __aicore__ uint32_t CombineGetGroupSourceRank(
    uint32_t rank,
    uint32_t world_size,
    uint32_t group,
    uint32_t group_rank_index) {
    const uint32_t half_world_size = world_size / 2U;
    const uint32_t group_width = CombineGetGroupWidth(world_size);
    const uint32_t half_group_width = group_width / 2U;
    const uint32_t group_count = world_size / group_width;
    const uint32_t rank_group =
        rank % half_world_size / half_group_width;
    const uint32_t source_group = (rank_group + group) % group_count;
    return group_rank_index / half_group_width * half_world_size +
        source_group * half_group_width +
        group_rank_index % half_group_width;
}

// Inverse of CombineGetGroupSourceRank: reader_rank GETs from rank in group.
__forceinline__ __aicore__ uint32_t CombineGetGroupReaderRank(
    uint32_t rank,
    uint32_t world_size,
    uint32_t group,
    uint32_t group_rank_index) {
    const uint32_t half_world_size = world_size / 2U;
    const uint32_t group_width = CombineGetGroupWidth(world_size);
    const uint32_t half_group_width = group_width / 2U;
    const uint32_t group_count = world_size / group_width;
    const uint32_t rank_group =
        rank % half_world_size / half_group_width;
    const uint32_t reader_group =
        (rank_group + group_count - group % group_count) % group_count;
    return group_rank_index / half_group_width * half_world_size +
        reader_group * half_group_width +
        group_rank_index % half_group_width;
}

__forceinline__ __aicore__ __gm__ uint64_t* CombineGetGroupSignal(
    __gm__ uint8_t* sync_buffer,
    uint32_t publisher_rank,
    uint32_t receiver_group_rank_index,
    uint32_t signal_slot) {
    const uint64_t rank_stride =
        static_cast<uint64_t>(kCombineMaxGroupWidth) *
        kCombineGroupSignalSlots * kCombineGroupSignalBytes;
    const uint64_t receiver_stride =
        static_cast<uint64_t>(kCombineGroupSignalSlots) *
        kCombineGroupSignalBytes;
    return reinterpret_cast<__gm__ uint64_t*>(
        sync_buffer + static_cast<uint64_t>(publisher_rank) * rank_stride +
        static_cast<uint64_t>(receiver_group_rank_index) * receiver_stride +
        static_cast<uint64_t>(signal_slot) * kCombineGroupSignalBytes);
}

__forceinline__ __aicore__ uint64_t CombineGetGroupToken(
    uint64_t invocation_magic,
    uint32_t group,
    uint32_t group_count) {
    return kCombineGroupMagicPrefix |
        (invocation_magic * (static_cast<uint64_t>(group_count) + 1U) +
         group + 1U);
}

template <typename T>
__forceinline__ __aicore__ LocalTensor<T> CombineMakeUbTensor(
    LocalTensor<uint8_t> ub_base,
    uint32_t byte_offset,
    uint32_t elements) {
    LocalTensor<T> tensor = ub_base[byte_offset].template ReinterpretCast<T>();
    tensor.address_.logicPos = static_cast<uint8_t>(TPosition::VECIN);
    tensor.address_.dataLen = elements * sizeof(T);
    return tensor;
}

template <typename T>
__forceinline__ __aicore__ __ubuf__ T* CombineUbPtr(LocalTensor<T> ub) {
    return reinterpret_cast<__ubuf__ T*>(static_cast<uintptr_t>(ub.address_.bufferAddr));
}

__forceinline__ __aicore__ void CombinePublishGroupSignal(
    LocalTensor<uint8_t> ub_base,
    __gm__ uint8_t* sync_buffer,
    uint32_t publisher_rank,
    uint32_t receiver_rank,
    uint32_t receiver_group_rank_index,
    uint32_t signal_slot,
    uint64_t token) {
    if (publisher_rank == receiver_rank) {
        return;
    }

    constexpr uint32_t word_count =
        kCombineGroupSignalBytes / sizeof(uint64_t);
    LocalTensor<uint64_t> signal_local = CombineMakeUbTensor<uint64_t>(
        ub_base, kCombineUdmaStagingOffset, word_count);
    for (uint32_t word = 0; word < word_count; ++word) {
        signal_local.SetValue(word, word == 0U ? token : 0U);
    }
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);

    __gm__ uint64_t* local_signal = CombineGetGroupSignal(
        sync_buffer,
        publisher_rank,
        receiver_group_rank_index,
        signal_slot);
    __gm__ uint64_t* remote_signal = reinterpret_cast<__gm__ uint64_t*>(
        aclshmem_ptr(local_signal, static_cast<int32_t>(receiver_rank)));
    GlobalTensor<uint64_t> remote_signal_tensor;
    remote_signal_tensor.SetGlobalBuffer(remote_signal, word_count);
    DataCopy(remote_signal_tensor, signal_local, word_count);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
}

__forceinline__ __aicore__ void CombineWaitGroupSignal(
    __gm__ uint8_t* sync_buffer,
    uint32_t publisher_rank,
    uint32_t receiver_rank,
    uint32_t receiver_group_rank_index,
    uint32_t signal_slot,
    uint64_t token) {
    if (publisher_rank == receiver_rank) {
        return;
    }
    aclshmem_uint64_wait_until(
        CombineGetGroupSignal(
            sync_buffer,
            publisher_rank,
            receiver_group_rank_index,
            signal_slot),
        ACLSHMEM_CMP_EQ,
        token);
}

template <typename T>
__forceinline__ __aicore__ void CombineCopyGmToUb(
    LocalTensor<T> ub,
    const __gm__ T* gm,
    uint32_t elements) {
    uint32_t total_bytes = elements * sizeof(T);
    if (total_bytes != 0 &&
        total_bytes <= kCombineMaxDataCopyPadBytes &&
        total_bytes % 32U == 0) {
        GlobalTensor<T> srcGM;
        srcGM.SetGlobalBuffer(const_cast<__gm__ T*>(gm), elements);
        DataCopy(ub, srcGM, elements);
        return;
    }

    uint32_t copied = 0;
    while (copied < elements) {
        uint32_t chunk_elements = elements - copied;
        uint32_t max_chunk_elements = kCombineMaxDataCopyPadBytes / sizeof(T);
        if (chunk_elements > max_chunk_elements) {
            chunk_elements = max_chunk_elements;
        }
        uint32_t chunk_bytes = chunk_elements * sizeof(T);
        GlobalTensor<T> srcGM;
        srcGM.SetGlobalBuffer(const_cast<__gm__ T*>(gm) + copied, chunk_elements);
        DataCopyParams cp {1, static_cast<uint16_t>(chunk_bytes), 0, 0};
        DataCopyPad(ub[copied], srcGM, cp, DataCopyPadParams{});
        copied += chunk_elements;
    }
}

__forceinline__ __aicore__ void CombineLocalAivBarrier() {
    // SHMEM's internal core barrier selects hardware SyncAll for MIX builds
    // and its soft AIV barrier otherwise.  Unlike barrier_all_vec this never
    // waits for a peer rank.
    aclshmemi_sync_core<true>();
}

__forceinline__ __aicore__ void CombineCopyGmToUbForScalar(
    LocalTensor<int32_t> ub,
    const __gm__ int32_t* gm,
    uint32_t elements) {
    uint32_t copied = 0;
    while (copied < elements) {
        uint32_t chunk_elements = elements - copied;
        uint32_t max_chunk_elements =
            kCombineMaxDataCopyPadBytes / sizeof(int32_t);
        if (chunk_elements > max_chunk_elements) {
            chunk_elements = max_chunk_elements;
        }
        uint32_t chunk_bytes = chunk_elements * sizeof(int32_t);
        GlobalTensor<int32_t> srcGM;
        srcGM.SetGlobalBuffer(
            const_cast<__gm__ int32_t*>(gm) + copied,
            chunk_elements);
        DataCopyParams cp {1, static_cast<uint16_t>(chunk_bytes), 0, 0};
        DataCopyPad(ub[copied], srcGM, cp, DataCopyPadParams{});
        copied += chunk_elements;
    }
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
}

// Like CombineCopyGmToUbForScalar, but MTE2_S waits on `event_id` so a
// preceding large MTE2 (e.g. hidden row on ID0) can keep MTE2_V ownership.
__forceinline__ __aicore__ void CombineCopyGmToUbForScalarOnEvent(
    LocalTensor<int32_t> ub,
    const __gm__ int32_t* gm,
    uint32_t elements,
    TEventID event_id) {
    uint32_t copied = 0;
    while (copied < elements) {
        uint32_t chunk_elements = elements - copied;
        uint32_t max_chunk_elements =
            kCombineMaxDataCopyPadBytes / sizeof(int32_t);
        if (chunk_elements > max_chunk_elements) {
            chunk_elements = max_chunk_elements;
        }
        uint32_t chunk_bytes = chunk_elements * sizeof(int32_t);
        GlobalTensor<int32_t> srcGM;
        srcGM.SetGlobalBuffer(
            const_cast<__gm__ int32_t*>(gm) + copied,
            chunk_elements);
        DataCopyParams cp {1, static_cast<uint16_t>(chunk_bytes), 0, 0};
        DataCopyPad(ub[copied], srcGM, cp, DataCopyPadParams{});
        copied += chunk_elements;
    }
    SetFlag<HardEvent::MTE2_S>(event_id);
    WaitFlag<HardEvent::MTE2_S>(event_id);
}

template <typename T>
__forceinline__ __aicore__ void CombineIssueUbToGm(
    __gm__ T* gm,
    LocalTensor<T> ub,
    uint32_t elements) {
    uint32_t total_bytes = elements * sizeof(T);
    if (total_bytes != 0 &&
        total_bytes <= kCombineMaxDataCopyPadBytes &&
        total_bytes % 32U == 0) {
        GlobalTensor<T> dstGM;
        dstGM.SetGlobalBuffer(gm, elements);
        DataCopy(dstGM, ub, elements);
        return;
    }

    uint32_t copied = 0;
    while (copied < elements) {
        uint32_t chunk_elements = elements - copied;
        uint32_t max_chunk_elements = kCombineMaxDataCopyPadBytes / sizeof(T);
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

__forceinline__ __aicore__ void CombineZeroFp32(
    LocalTensor<float> acc,
    uint32_t elements) {
    Duplicate(acc, 0.0f, elements);
}

__forceinline__ __aicore__ void CombineAccumulateBf16Row(
    LocalTensor<float> acc,
    LocalTensor<float> tmp,
    LocalTensor<bf16_t> row,
    uint32_t elements,
    bool first) {
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    if (first) {
        Cast(acc, row, RoundMode::CAST_NONE, static_cast<int32_t>(elements));
    } else {
        Cast(tmp, row, RoundMode::CAST_NONE, static_cast<int32_t>(elements));
        PipeBarrier<PIPE_V>();
        Add(acc, acc, tmp, static_cast<int32_t>(elements));
    }

    // The next CombineCopyGmToUb reuses row through MTE2. Do not let MTE2
    // overwrite it until Vector has finished consuming the current contents.
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
}

__forceinline__ __aicore__ void CombineStoreAccAsBf16(
    __gm__ bf16_t* dst,
    LocalTensor<float> acc,
    LocalTensor<bf16_t> out_row,
    uint32_t elements) {
    Cast(out_row, acc, RoundMode::CAST_RINT, static_cast<int32_t>(elements));
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    CombineIssueUbToGm(dst, out_row, elements);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

// ============================================================
// Phase 1: Prologue — reduce duplicates into primary (inverse of epilogue)
// ============================================================
// UBMem-style full-row load ping-pong (ID0/ID1) + dual store credits (ID2/ID3).
// GET plans stay 3-field (primary, dup_start, dup_n). Contiguous AIV SplitRange
// replaces the old strided gid walk so windowed group meta loads stay coherent.
// Drain store credits before the cross-rank barrier / gather.

template <typename HiddenT>
__forceinline__ __aicore__ void PlanPrologueIssueHiddenLoad(
    LocalTensor<HiddenT> row,
    __gm__ HiddenT* local_hidden,
    uint32_t loff,
    uint32_t H,
    TEventID event_id) {
    CombineCopyGmToUb(
        row, local_hidden + static_cast<uint64_t>(loff) * H, H);
    SetFlag<HardEvent::MTE2_V>(event_id);
}

template <typename HiddenT>
__aicore__ void PlanPrologueReduceGroup(
    LocalTensor<uint8_t> ub_base,
    LocalTensor<int32_t> metaLocal,
    __gm__ HiddenT* local_hidden,
    const __gm__ int32_t* dup_loffs,
    int32_t primary_loff,
    int32_t dup_start,
    int32_t dup_n,
    uint32_t H,
    uint32_t NvS,
    uint32_t kMetaStageElements,
    uint32_t& store_stage) {
    if (H == 0U ||
        H * kCombinePrologueDbBytesPerElement > kCombineUbHiddenRowBytes) {
        return;
    }

    const uint32_t row_bytes = H * sizeof(HiddenT);
    LocalTensor<HiddenT> row0 =
        CombineMakeUbTensor<HiddenT>(ub_base, 0, H);
    LocalTensor<HiddenT> row1 =
        CombineMakeUbTensor<HiddenT>(ub_base, row_bytes, H);
    LocalTensor<HiddenT> store0 =
        CombineMakeUbTensor<HiddenT>(ub_base, 2U * row_bytes, H);
    LocalTensor<HiddenT> store1 =
        CombineMakeUbTensor<HiddenT>(ub_base, 3U * row_bytes, H);
    LocalTensor<float> acc =
        CombineMakeUbTensor<float>(ub_base, 4U * row_bytes, H);
    LocalTensor<float> tmp = CombineMakeUbTensor<float>(
        ub_base, 4U * row_bytes + H * sizeof(float), H);

    uint32_t src_loff[kCombinePrologueMaxSrcs];
    uint32_t n_src = 0;
    src_loff[n_src++] = static_cast<uint32_t>(primary_loff);

    SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
    PlanPrologueIssueHiddenLoad<HiddenT>(
        row0, local_hidden, src_loff[0], H, EVENT_ID0);

    // Small dup_loffs MTE2 tucked behind the first large hidden load (ID1).
    for (int32_t j = 0; j < dup_n && n_src < kCombinePrologueMaxSrcs;) {
        uint32_t loff_count = static_cast<uint32_t>(dup_n - j);
        if (loff_count > kMetaStageElements) {
            loff_count = kMetaStageElements;
        }
        if (n_src + loff_count > kCombinePrologueMaxSrcs) {
            loff_count = kCombinePrologueMaxSrcs - n_src;
        }
        CombineCopyGmToUbForScalarOnEvent(
            metaLocal, dup_loffs + dup_start + j, loff_count, EVENT_ID1);
        for (uint32_t loff_i = 0; loff_i < loff_count; ++loff_i) {
            int32_t dst_i = metaLocal.GetValue(loff_i);
            if (dst_i < 0) {
                continue;
            }
            uint32_t dst_loff = static_cast<uint32_t>(dst_i);
            if (dst_loff >= NvS) {
                continue;
            }
            src_loff[n_src++] = dst_loff;
            if (n_src >= kCombinePrologueMaxSrcs) {
                break;
            }
        }
        j += static_cast<int32_t>(loff_count);
    }

    SetFlag<HardEvent::S_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::S_MTE2>(EVENT_ID1);

    bool ping_pending = false;
    bool pong_pending = false;
    int32_t cur = 0;
    for (uint32_t i = 0; i < n_src; ++i) {
        TEventID cur_ev = cur == 0 ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<HiddenT> cur_row = cur == 0 ? row0 : row1;
        WaitFlag<HardEvent::MTE2_V>(cur_ev);

        uint32_t next_i = i + 1U;
        if (next_i < n_src) {
            int32_t next_buf = 1 - cur;
            TEventID next_ev = next_buf == 0 ? EVENT_ID0 : EVENT_ID1;
            LocalTensor<HiddenT> next_row = next_buf == 0 ? row0 : row1;
            if (next_buf == 0 && ping_pending) {
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                ping_pending = false;
            } else if (next_buf == 1 && pong_pending) {
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
                pong_pending = false;
            }
            PlanPrologueIssueHiddenLoad<HiddenT>(
                next_row, local_hidden, src_loff[next_i], H, next_ev);
        }

        if (i == 0U) {
            Cast(acc, cur_row, RoundMode::CAST_NONE, static_cast<int32_t>(H));
        } else {
            Cast(tmp, cur_row, RoundMode::CAST_NONE, static_cast<int32_t>(H));
            PipeBarrier<PIPE_V>();
            Add(acc, acc, tmp, static_cast<int32_t>(H));
        }
        SetFlag<HardEvent::V_MTE2>(cur_ev);
        if (cur == 0) {
            ping_pending = true;
        } else {
            pong_pending = true;
        }
        cur = 1 - cur;
    }

    if (ping_pending) {
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }
    if (pong_pending) {
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    }

    TEventID store_ev = store_stage == 0U ? EVENT_ID2 : EVENT_ID3;
    LocalTensor<HiddenT> store_row = store_stage == 0U ? store0 : store1;
    WaitFlag<HardEvent::MTE3_MTE2>(store_ev);
    Cast(store_row, acc, RoundMode::CAST_RINT, static_cast<int32_t>(H));
    SetFlag<HardEvent::V_MTE3>(store_ev);
    WaitFlag<HardEvent::V_MTE3>(store_ev);
    CombineIssueUbToGm(
        local_hidden + static_cast<uint64_t>(primary_loff) * H,
        store_row,
        H);
    SetFlag<HardEvent::MTE3_MTE2>(store_ev);
    store_stage ^= 1U;
}

template <typename HiddenT>
__aicore__ void PlanProloguePhase(
    LocalTensor<uint8_t> ub_base,
    __gm__ HiddenT* local_hidden,
    const __gm__ int32_t* dup_groups,
    const __gm__ int32_t* dup_loffs,
    const __gm__ int32_t* dup_counts,
    uint32_t H,
    uint32_t NvS,
    uint32_t core_idx,
    uint32_t core_num) {

    constexpr uint32_t kMetaStageElements =
        kCombineWeightStagingBytes / sizeof(int32_t);
    LocalTensor<int32_t> metaLocal = CombineMakeUbTensor<int32_t>(
        ub_base, kCombinePhaseScratchBytes, kMetaStageElements);
    CombineCopyGmToUbForScalar(metaLocal, dup_counts, 2);

    int32_t n_groups_i = metaLocal.GetValue(0);
    int32_t n_dup_loffs_i = metaLocal.GetValue(1);
    if (n_groups_i <= 0) {
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::DDR>();
        CombineLocalAivBarrier();
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

    uint32_t group_beg = 0;
    uint32_t group_end = 0;
    CombineSplitRange(n_groups, core_num, core_idx, group_beg, group_end);
    if (group_beg >= group_end) {
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::DDR>();
        CombineLocalAivBarrier();
        return;
    }

    LocalTensor<int32_t> planChunkLocal = CombineMakeUbTensor<int32_t>(
        ub_base, kCombineProloguePlanOffset,
        kCombinePrologueEntryWindow * kCombinePrologueGroupFields);

    uint32_t store_stage = 0;
    bool store_db_live = false;

    for (uint32_t window_start = group_beg;
         window_start < group_end;
         window_start += kCombinePrologueEntryWindow) {
        uint32_t window_count = group_end - window_start;
        if (window_count > kCombinePrologueEntryWindow) {
            window_count = kCombinePrologueEntryWindow;
        }
        const uint32_t flat_count =
            window_count * kCombinePrologueGroupFields;
        CombineCopyGmToUb(
            planChunkLocal,
            dup_groups +
                static_cast<uint64_t>(window_start) * kCombinePrologueGroupFields,
            flat_count);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);

        for (uint32_t li = 0; li < window_count; ++li) {
            const uint32_t gid = window_start + li;
            const uint32_t base = li * kCombinePrologueGroupFields;
            int32_t primary_loff = planChunkLocal.GetValue(base);
            int32_t dup_start = planChunkLocal.GetValue(base + 1);
            int32_t dup_n = planChunkLocal.GetValue(base + 2);
            bool group_ok =
                primary_loff >= 0 &&
                primary_loff < static_cast<int32_t>(NvS) &&
                dup_start >= 0 &&
                dup_n > 0;
            if (group_ok) {
                if (dup_start > static_cast<int32_t>(n_dup_loffs)) {
                    group_ok = false;
                } else {
                    int32_t available_dups =
                        static_cast<int32_t>(n_dup_loffs) - dup_start;
                    if (dup_n > available_dups) {
                        dup_n = available_dups;
                    }
                    if (dup_n <= 0) {
                        group_ok = false;
                    }
                }
            }

            if (group_ok) {
                if (!store_db_live) {
                    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
                    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
                    store_db_live = true;
                }
                PlanPrologueReduceGroup<HiddenT>(
                    ub_base, metaLocal, local_hidden, dup_loffs,
                    primary_loff, dup_start, dup_n, H, NvS, kMetaStageElements,
                    store_stage);
            }
        }
    }

    if (store_db_live) {
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
    }

    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::DDR>();
    CombineLocalAivBarrier();
}

// ============================================================
// Phase 2: basic UDMA Get gather into route-major local GM
// ============================================================

__forceinline__ __aicore__ uint32_t CombineDecodeRank(
    int32_t dst_val,
    uint32_t NvS) {
    uint32_t raw_dst = dst_val >= 0
        ? static_cast<uint32_t>(dst_val)
        : static_cast<uint32_t>(-static_cast<int64_t>(dst_val) - 1);
    return raw_dst / NvS;
}

__forceinline__ __aicore__ uint32_t CombineDecodeLoff(
    int32_t dst_val,
    uint32_t NvS) {
    uint32_t raw_dst = dst_val >= 0
        ? static_cast<uint32_t>(dst_val)
        : static_cast<uint32_t>(-static_cast<int64_t>(dst_val) - 1);
    return raw_dst % NvS;
}

__forceinline__ __aicore__ bool CombineCanUseVectorSlotSelect(
    uint32_t route_count,
    uint32_t& slot_shift) {
    if (route_count == 0 ||
        (route_count & (route_count - 1U)) != 0) {
        return false;
    }
    slot_shift = 0;
    for (uint32_t value = route_count; value > 1U; value >>= 1U) {
        ++slot_shift;
    }
    return slot_shift <= 31U;
}

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
// GET does not use the RO/SO ordering domains. Fence every aggregate WQE;
// deferred WQEs stay CQE-free and the final submit WQE publishes the CQE.
inline constexpr aclshmemx_udma_op_config_t
    kCombineUdmaOpConfigDeferFence{
        0U, 0U, 1U, ACLSHMEMX_UDMA_ODR_NO};
inline constexpr aclshmemx_udma_op_config_t
    kCombineUdmaOpConfigSubmitFence{
        1U, 0U, 1U, ACLSHMEMX_UDMA_ODR_NO};

template <typename T>
__forceinline__ __aicore__ void CombineUdmaGet(
    __gm__ T* local_dst,
    __gm__ T* symmetric_src,
    __ubuf__ uint8_t* wqe_staging,
    uint32_t elements,
    uint32_t dest_rank,
    uint32_t qp_idx) {
    aclshmemx_udma_qp_get_nbi<T, PIPE_MTE3>(
        local_dst,
        symmetric_src,
        reinterpret_cast<__ubuf__ T*>(wqe_staging),
        elements,
        static_cast<int32_t>(dest_rank),
        qp_idx,
        EVENT_ID0);
}

template <typename T>
__forceinline__ __aicore__ void CombineUdmaGet(
    __gm__ T* local_dst,
    __gm__ T* symmetric_src,
    __ubuf__ uint8_t* wqe_staging,
    uint32_t elements,
    uint32_t dest_rank,
    uint32_t qp_idx,
    aclshmemx_defer_t action) {
    aclshmemx_udma_qp_get_nbi<
        T, PIPE_MTE3, kCombineUdmaOpConfigDeferFence>(
        local_dst,
        symmetric_src,
        reinterpret_cast<__ubuf__ T*>(wqe_staging),
        elements,
        static_cast<int32_t>(dest_rank),
        qp_idx,
        EVENT_ID0,
        action);
}

template <typename T>
__forceinline__ __aicore__ void CombineUdmaGet(
    __gm__ T* local_dst,
    __gm__ T* symmetric_src,
    __ubuf__ uint8_t* wqe_staging,
    uint32_t elements,
    uint32_t dest_rank,
    uint32_t qp_idx,
    aclshmemx_submit_t action) {
    aclshmemx_udma_qp_get_nbi<
        T, PIPE_MTE3, kCombineUdmaOpConfigSubmitFence>(
        local_dst,
        symmetric_src,
        reinterpret_cast<__ubuf__ T*>(wqe_staging),
        elements,
        static_cast<int32_t>(dest_rank),
        qp_idx,
        EVENT_ID0,
        action);
}
#endif

__aicore__ void AssembleWeightsFromStagingPhase(
    LocalTensor<uint8_t> ub_base,
    __gm__ float* published_weights,
    const __gm__ float* gathered_weights,
    const __gm__ int32_t* dst,
    __gm__ float* output_sk,
    uint32_t R,
    uint32_t NvS,
    uint32_t rank,
    uint32_t worker_idx,
    uint32_t worker_num) {
    uint32_t route_start = 0;
    uint32_t route_end = 0;
    CombineSplitRange(
        NvS, worker_num, worker_idx, route_start, route_end);
    if (route_start >= route_end) {
        return;
    }

    LocalTensor<int32_t> dst_local = CombineMakeUbTensor<int32_t>(
        ub_base,
        2U * kCombineWeightOutputStageBytes,
        kCombineSlotChunkCount);

    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
    uint32_t output_stage_idx = 0;

    for (uint32_t chunk_start = route_start;
         chunk_start < route_end;
         chunk_start += kCombineSlotChunkCount) {
        uint32_t chunk_count = route_end - chunk_start;
        if (chunk_count > kCombineSlotChunkCount) {
            chunk_count = kCombineSlotChunkCount;
        }

        const TEventID store_ev =
            output_stage_idx == 0U ? EVENT_ID2 : EVENT_ID3;
        LocalTensor<float> output_local = CombineMakeUbTensor<float>(
            ub_base,
            output_stage_idx * kCombineWeightOutputStageBytes,
            kCombineSlotChunkCount);
        WaitFlag<HardEvent::MTE3_MTE2>(store_ev);
        CombineCopyGmToUbForScalarOnEvent(
            dst_local, dst + chunk_start, chunk_count, EVENT_ID0);

        for (uint32_t route_i = 0; route_i < chunk_count; ++route_i) {
            const int32_t dst_val = dst_local.GetValue(route_i);
            const uint32_t dest_rank = CombineDecodeRank(dst_val, NvS);
            const uint32_t loff = CombineDecodeLoff(dst_val, NvS);
            float weight = 0.0f;
            if (dest_rank >= R || loff >= NvS) {
                output_local.SetValue(route_i, weight);
                continue;
            }
            weight = dest_rank == rank
                ? published_weights[loff]
                : gathered_weights[
                      static_cast<uint64_t>(dest_rank) * NvS + loff];
            output_local.SetValue(route_i, weight);
        }

        SetFlag<HardEvent::S_MTE3>(store_ev);
        WaitFlag<HardEvent::S_MTE3>(store_ev);
        CombineIssueUbToGm(
            output_sk + chunk_start, output_local, chunk_count);
        SetFlag<HardEvent::MTE3_MTE2>(store_ev);
        output_stage_idx ^= 1U;
    }

    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
}

template <typename HiddenT>
__aicore__ CombineGatherResult GatherToStagingPhase(
    LocalTensor<uint8_t> ub_base,
    __gm__ HiddenT* published_hidden,
    __gm__ float* published_weights,
    __gm__ uint8_t* sync_buffer,
    GM_ADDR workspace,
    const __gm__ int32_t* dst,
    __gm__ float* output_sk,
    uint32_t H,
    uint32_t R,
    uint32_t NvS,
    bool with_weights,
    uint64_t invocation_magic,
    uint32_t rank,
    uint32_t core_idx,
    uint32_t core_num) {

    __gm__ HiddenT* gathered_hidden = reinterpret_cast<__gm__ HiddenT*>(
        workspace);
    const uint64_t gathered_weights_off_bytes =
        static_cast<uint64_t>(NvS) * H * sizeof(HiddenT);
    __gm__ float* gathered_weights = reinterpret_cast<__gm__ float*>(
        workspace + gathered_weights_off_bytes);
    __ubuf__ uint8_t* wqe_staging = reinterpret_cast<__ubuf__ uint8_t*>(
        CombineUbPtr(CombineMakeUbTensor<uint8_t>(
            ub_base, kCombineUdmaStagingOffset, kCombineUdmaStagingBytes)));

    CombineGatherResult result{};
    CombineLocalAivBarrier();
#if MOONEP_UDMA_PROFILE_STATISTIC
    const uint64_t hidden_submit_start_cycle = AscendC::GetSystemCycle();
#endif

    uint32_t slot_shift = 0;
    const bool use_vector_slot_select =
        CombineCanUseVectorSlotSelect(NvS, slot_shift);
    LocalTensor<int32_t> dstLocal = CombineMakeUbTensor<int32_t>(
        ub_base, 0, kCombineSlotChunkCount);
    LocalTensor<int32_t> dstRankLocal = CombineMakeUbTensor<int32_t>(
        ub_base, kCombineSlotTensorBytes, kCombineSlotChunkCount);
    LocalTensor<int32_t> routeIndices = CombineMakeUbTensor<int32_t>(
        ub_base, 2U * kCombineSlotTensorBytes, kCombineSlotChunkCount);
    LocalTensor<int32_t> selectedRouteIndices = CombineMakeUbTensor<int32_t>(
        ub_base, 3U * kCombineSlotTensorBytes, kCombineSlotChunkCount);
    LocalTensor<uint8_t> compareMask = CombineMakeUbTensor<uint8_t>(
        ub_base, 4U * kCombineSlotTensorBytes, kCombineSlotMaskBytes);
    GlobalTensor<int32_t> dstGlobal;
    dstGlobal.SetGlobalBuffer(
        const_cast<__gm__ int32_t*>(dst), NvS);

    // Dispatch-style group ownership. The active AIV set is group_width, and
    // each active AIV owns one source PE (and both of its QPs) per group.
    const uint32_t group_width = CombineGetGroupWidth(R);
    const uint32_t group_count = R > 1U ? R / group_width : 0U;
    const uint32_t group_rank_index = core_idx % group_width;
    const uint32_t current_rank_index = R > 1U
        ? CombineGetGroupRankIndex(rank, R)
        : 0U;
    const bool reader_active = R > 1U && core_idx < group_width;
    uint64_t hidden_rows = 0;
    for (uint32_t group = 0; group < group_count && reader_active; ++group) {
        const uint32_t dest_rank = CombineGetGroupSourceRank(
            rank, R, group, group_rank_index);
        const uint32_t incoming_reader = CombineGetGroupReaderRank(
            rank, R, group, group_rank_index);
        const uint64_t group_token = CombineGetGroupToken(
            invocation_magic, group, group_count);

        // This rank is a source for incoming_reader while it is a reader of
        // dest_rank. Publish every grant before waiting, exactly dual to
        // dispatch's receiver-credit flow, so the phase cannot deadlock.
        CombinePublishGroupSignal(
            ub_base,
            sync_buffer,
            rank,
            incoming_reader,
            group_rank_index,
            kCombineGrantSignalSlot,
            group_token);
        CombineWaitGroupSignal(
            sync_buffer,
            dest_rank,
            rank,
            current_rank_index,
            kCombineGrantSignalSlot,
            group_token);

        // Self occupies one balanced group slot but does not consume UDMA.
        if (dest_rank != rank) {
        if (use_vector_slot_select) {
            // NvS is a power of two, so encoded rank is dst >> log2(NvS).
            // Negative duplicate encodings remain negative after the shift
            // and therefore cannot match the non-negative destination rank.
            for (uint32_t chunk_start = 0;
                 chunk_start < NvS;
                 chunk_start += kCombineSlotChunkCount) {
                const uint32_t chunk_index =
                    chunk_start / kCombineSlotChunkCount;
                const uint32_t qp_idx =
                    CombineGetChunkQpIndex(chunk_index);
                uint32_t chunk_count = NvS - chunk_start;
                if (chunk_count > kCombineSlotChunkCount) {
                    chunk_count = kCombineSlotChunkCount;
                }
                const uint32_t padded_count =
                    ((chunk_count + kCombineVectorSlotSelectCountAlign - 1U) /
                     kCombineVectorSlotSelectCountAlign) *
                    kCombineVectorSlotSelectCountAlign;

                SetFlag<HardEvent::S_MTE2>(EVENT_ID1);
                WaitFlag<HardEvent::S_MTE2>(EVENT_ID1);
                DataCopyExtParams dst_copy_params {
                    1,
                    static_cast<uint32_t>(chunk_count * sizeof(int32_t)),
                    0,
                    0,
                    0};
                DataCopyPadExtParams<int32_t> dst_pad_params {
                    false, 0, 0, 0};
                DataCopyPad(
                    dstLocal,
                    dstGlobal[chunk_start],
                    dst_copy_params,
                    dst_pad_params);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

                ShiftRight(
                    dstRankLocal,
                    dstLocal,
                    static_cast<int32_t>(slot_shift),
                    padded_count);
                CreateVecIndex(
                    routeIndices,
                    static_cast<int32_t>(chunk_start),
                    padded_count);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::S_V>(EVENT_ID1);
                WaitFlag<HardEvent::S_V>(EVENT_ID1);
                Compares(
                    compareMask,
                    dstRankLocal,
                    static_cast<int32_t>(dest_rank),
                    CMPMODE::EQ,
                    padded_count);
                PipeBarrier<PIPE_V>();
                uint64_t selected_route_count = 0;
                GatherMask(
                    selectedRouteIndices,
                    routeIndices,
                    compareMask.ReinterpretCast<uint32_t>(),
                    true,
                    chunk_count,
                    {1U, 1U, 0U, 0U},
                    selected_route_count);
                SetFlag<HardEvent::V_S>(EVENT_ID1);
                WaitFlag<HardEvent::V_S>(EVENT_ID1);
                hidden_rows += selected_route_count;
                aclshmemx_submit_state_t submit_state{};
                aclshmemx_defer_t defer_action(submit_state);
                aclshmemx_submit_t submit_action(submit_state);
                for (uint64_t batch_start = 0;
                     batch_start < selected_route_count;
                     batch_start += kCombineUdmaWqeBatchSize) {
                    uint64_t batch_end = selected_route_count;
                    if (batch_end - batch_start > kCombineUdmaWqeBatchSize) {
                        batch_end = batch_start + kCombineUdmaWqeBatchSize;
                    }
                    // Defer the first N-1 requests without CQEs, then submit
                    // the final request as the batch's only CQE checkpoint.
                    for (uint64_t selected = batch_start;
                         selected + 1U < batch_end;
                         ++selected) {
                        const uint32_t route_idx = static_cast<uint32_t>(
                            selectedRouteIndices.GetValue(selected));
                        const int32_t dst_val =
                            dstLocal.GetValue(route_idx - chunk_start);
                        const uint32_t loff =
                            static_cast<uint32_t>(dst_val) & (NvS - 1U);
                        __gm__ HiddenT* remote_hidden =
                            published_hidden + static_cast<uint64_t>(loff) * H;
                        CombineUdmaGet(
                            gathered_hidden +
                                static_cast<uint64_t>(route_idx) * H,
                            remote_hidden,
                            wqe_staging,
                            H,
                            dest_rank,
                            qp_idx,
                            defer_action);
                    }

                    const uint64_t selected = batch_end - 1U;
                    const uint32_t route_idx = static_cast<uint32_t>(
                        selectedRouteIndices.GetValue(selected));
                    const int32_t dst_val =
                        dstLocal.GetValue(route_idx - chunk_start);
                    const uint32_t loff =
                        static_cast<uint32_t>(dst_val) & (NvS - 1U);
                    __gm__ HiddenT* remote_hidden =
                        published_hidden + static_cast<uint64_t>(loff) * H;
                    CombineUdmaGet(
                        gathered_hidden +
                            static_cast<uint64_t>(route_idx) * H,
                        remote_hidden,
                        wqe_staging,
                        H,
                        dest_rank,
                        qp_idx,
                        submit_action);
                }
            }
        } else {
            // Keep every aggregate batch on one QP. A submit state must never
            // span a chunk boundary where the 3:1 QP selection may change.
            for (uint32_t chunk_start = 0;
                 chunk_start < NvS;
                 chunk_start += kCombineSlotChunkCount) {
                uint32_t chunk_end = chunk_start + kCombineSlotChunkCount;
                if (chunk_end > NvS) {
                    chunk_end = NvS;
                }
                const uint32_t chunk_index =
                    chunk_start / kCombineSlotChunkCount;
                const uint32_t qp_idx =
                    CombineGetChunkQpIndex(chunk_index);
                aclshmemx_submit_state_t submit_state{};
                aclshmemx_defer_t defer_action(submit_state);
                aclshmemx_submit_t submit_action(submit_state);
                bool has_held_final = false;
                bool has_held_unsignaled = false;
                uint32_t held_final_route_idx = 0;
                uint32_t held_final_loff = 0;
                uint32_t held_unsignaled_route_idx = 0;
                uint32_t held_unsignaled_loff = 0;
                uint32_t deferred_in_batch = 0;
                for (uint32_t route_idx = chunk_start;
                     route_idx < chunk_end;
                     ++route_idx) {
                    const int32_t dst_val = dst[route_idx];
                    if (dst_val < 0 ||
                        CombineDecodeRank(dst_val, NvS) != dest_rank) {
                        continue;
                    }
                    ++hidden_rows;
                    const uint32_t loff = CombineDecodeLoff(dst_val, NvS);
                    if (!has_held_final) {
                        held_final_route_idx = route_idx;
                        held_final_loff = loff;
                        has_held_final = true;
                        continue;
                    }

                    if (has_held_unsignaled) {
                        __gm__ HiddenT* held_remote_hidden = published_hidden +
                            static_cast<uint64_t>(held_unsignaled_loff) * H;
                        CombineUdmaGet(
                            gathered_hidden + static_cast<uint64_t>(
                                held_unsignaled_route_idx) * H,
                            held_remote_hidden,
                            wqe_staging,
                            H,
                            dest_rank,
                            qp_idx,
                            defer_action);
                        ++deferred_in_batch;
                    }
                    held_unsignaled_route_idx = held_final_route_idx;
                    held_unsignaled_loff = held_final_loff;
                    has_held_unsignaled = true;
                    held_final_route_idx = route_idx;
                    held_final_loff = loff;

                    if (deferred_in_batch + 2U ==
                        kCombineUdmaWqeBatchSize) {
                        __gm__ HiddenT* held_remote_hidden = published_hidden +
                            static_cast<uint64_t>(held_unsignaled_loff) * H;
                        CombineUdmaGet(
                            gathered_hidden + static_cast<uint64_t>(
                                held_unsignaled_route_idx) * H,
                            held_remote_hidden,
                            wqe_staging,
                            H,
                            dest_rank,
                            qp_idx,
                            defer_action);
                        __gm__ HiddenT* final_remote_hidden = published_hidden +
                            static_cast<uint64_t>(held_final_loff) * H;
                        CombineUdmaGet(
                            gathered_hidden + static_cast<uint64_t>(
                                held_final_route_idx) * H,
                            final_remote_hidden,
                            wqe_staging,
                            H,
                            dest_rank,
                            qp_idx,
                            submit_action);
                        deferred_in_batch = 0;
                        has_held_final = false;
                        has_held_unsignaled = false;
                    }
                }
                if (has_held_unsignaled) {
                    __gm__ HiddenT* held_remote_hidden = published_hidden +
                        static_cast<uint64_t>(held_unsignaled_loff) * H;
                    CombineUdmaGet(
                        gathered_hidden + static_cast<uint64_t>(
                            held_unsignaled_route_idx) * H,
                        held_remote_hidden,
                        wqe_staging,
                        H,
                        dest_rank,
                        qp_idx,
                        defer_action);
                }
                if (has_held_final) {
                    __gm__ HiddenT* held_remote_hidden = published_hidden +
                        static_cast<uint64_t>(held_final_loff) * H;
                    CombineUdmaGet(
                        gathered_hidden + static_cast<uint64_t>(
                            held_final_route_idx) * H,
                        held_remote_hidden,
                        wqe_staging,
                        H,
                        dest_rank,
                        qp_idx,
                        submit_action);
                }
            }
        }
        }

        // Weight GET overlaps the already submitted hidden GETs for this peer.
        // done is not returned until both UDMA QPs and MTE are complete.
        if (dest_rank != rank && with_weights && NvS != 0U) {
            aclshmemx_mte_get_nbi<float>(
                gathered_weights + static_cast<uint64_t>(dest_rank) * NvS,
                published_weights,
                reinterpret_cast<__ubuf__ float*>(wqe_staging),
                kCombineUdmaStagingBytes,
                NvS,
                static_cast<int32_t>(dest_rank),
                EVENT_ID0);
        }
        if (dest_rank != rank) {
            for (uint32_t qp_idx = 0;
                 qp_idx < kCombineUdmaQpNum;
                 ++qp_idx) {
                aclshmemx_udma_qp_quiet(
                    static_cast<int32_t>(dest_rank), qp_idx);
            }
            if (with_weights && NvS != 0U) {
                aclshmemx_mte_quiet();
            }
        }

        CombinePublishGroupSignal(
            ub_base,
            sync_buffer,
            rank,
            dest_rank,
            group_rank_index,
            kCombineDoneSignalSlot,
            group_token);
        CombineWaitGroupSignal(
            sync_buffer,
            incoming_reader,
            rank,
            current_rank_index,
            kCombineDoneSignalSlot,
            group_token);
    }

    // Group completion above already includes hidden QP quiet and the optional
    // weight MTE quiet. Keep the legacy profile fields ABI-stable: the complete
    // grouped transfer is reported in hidden_submit.
#if MOONEP_UDMA_PROFILE_STATISTIC
    const uint64_t hidden_submit_end_cycle = AscendC::GetSystemCycle();
#endif
#if MOONEP_UDMA_PROFILE_STATISTIC
    const uint64_t weight_get_end_cycle = AscendC::GetSystemCycle();
#endif
    CombineLocalAivBarrier();
#if MOONEP_UDMA_PROFILE_STATISTIC
    const uint64_t weight_sync_end_cycle = AscendC::GetSystemCycle();
#endif
    if (with_weights) {
        AssembleWeightsFromStagingPhase(
            ub_base,
            published_weights,
            gathered_weights,
            dst,
            output_sk,
            R,
            NvS,
            rank,
            core_idx,
            core_num);
    }
#if MOONEP_UDMA_PROFILE_STATISTIC
    const uint64_t weight_assemble_end_cycle = AscendC::GetSystemCycle();
#endif

#if MOONEP_UDMA_PROFILE_STATISTIC
    const uint64_t hidden_quiet_end_cycle = AscendC::GetSystemCycle();
#endif
    CombineLocalAivBarrier();
#if MOONEP_UDMA_PROFILE_STATISTIC
    const uint64_t final_sync_end_cycle = AscendC::GetSystemCycle();
    result.hidden_submit_cycles =
        hidden_submit_end_cycle - hidden_submit_start_cycle;
    result.weight_get_cycles =
        weight_get_end_cycle - hidden_submit_end_cycle;
    result.weight_sync_cycles =
        weight_sync_end_cycle - weight_get_end_cycle;
    result.weight_assemble_cycles =
        weight_assemble_end_cycle - weight_sync_end_cycle;
    result.hidden_quiet_cycles =
        hidden_quiet_end_cycle - weight_assemble_end_cycle;
    result.final_sync_cycles =
        final_sync_end_cycle - hidden_quiet_end_cycle;
#endif
    result.hidden_bytes =
        hidden_rows * static_cast<uint64_t>(H) * sizeof(HiddenT);
    return result;
}

// ============================================================
// Phase 3: K-sum staged rows into token-major output
// ============================================================
// Token-parallel AIV split. Weights are already in output_sk from Phase 2.
// Hidden: full-row load ping-pong (ID0/ID1) and dual store credits (ID2/ID3).
// Ported from get_opt: dst window MTE2'd into UB; load credits live across
// tokens; next-token decode+preload overlaps store Cast; Cast → V_MTE2 → Add
// with no Cast↔Add PipeBarrier so MTE2 overlaps accumulate.

template <typename HiddenT>
__forceinline__ __aicore__ void Phase3IssueHiddenLoad(
    LocalTensor<HiddenT> row,
    __gm__ HiddenT* published_hidden,
    const __gm__ HiddenT* gathered_hidden,
    uint32_t dest_rank,
    uint32_t loff,
    uint32_t route_idx,
    uint32_t H,
    uint32_t rank,
    TEventID event_id) {
    const __gm__ HiddenT* source_row =
        dest_rank == rank
            ? published_hidden + static_cast<uint64_t>(loff) * H
            : gathered_hidden + static_cast<uint64_t>(route_idx) * H;
    CombineCopyGmToUb(row, source_row, H);
    SetFlag<HardEvent::MTE2_V>(event_id);
}

__forceinline__ __aicore__ uint32_t Phase3DecodeTokenSrcs(
    LocalTensor<int32_t> ub_dst,
    uint32_t local_base,
    uint32_t token_id,
    uint32_t K,
    uint32_t R,
    uint32_t NvS,
    uint32_t* src_rank,
    uint32_t* src_loff,
    uint32_t* src_route) {
    uint32_t n_src = 0;
    const uint32_t route_base = token_id * K;
    for (uint32_t topk_id = 0; topk_id < K; ++topk_id) {
        int32_t dst_val = ub_dst.GetValue(local_base + topk_id);
        if (dst_val < 0) {
            continue;
        }
        uint32_t dest_rank = CombineDecodeRank(dst_val, NvS);
        if (dest_rank >= R) {
            continue;
        }
        uint32_t loff = CombineDecodeLoff(dst_val, NvS);
        if (loff >= NvS) {
            continue;
        }
        src_rank[n_src] = dest_rank;
        src_loff[n_src] = loff;
        src_route[n_src] = route_base + topk_id;
        ++n_src;
    }
    return n_src;
}

template <typename HiddenT>
__aicore__ void GatherAccumulatePhase(
    LocalTensor<uint8_t> ub_base,
    __gm__ HiddenT* published_hidden,
    GM_ADDR workspace,
    const __gm__ int32_t* dst,
    __gm__ HiddenT* output_sh,
    uint32_t S,
    uint32_t H,
    uint32_t K,
    uint32_t R,
    uint32_t NvS,
    uint32_t rank,
    uint32_t core_idx,
    uint32_t core_num) {

    uint32_t tok_start = 0;
    uint32_t tok_end = 0;
    CombineSplitRange(S, core_num, core_idx, tok_start, tok_end);
    if (tok_start >= tok_end) {
        return;
    }
    if (H == 0U || K == 0U ||
        H * kCombinePrologueDbBytesPerElement > kCombineUbHiddenRowBytes ||
        K > kCombinePrologueMaxSrcs) {
        return;
    }

    const uint32_t row_bytes = H * sizeof(HiddenT);
    LocalTensor<HiddenT> row0 =
        CombineMakeUbTensor<HiddenT>(ub_base, 0, H);
    LocalTensor<HiddenT> row1 =
        CombineMakeUbTensor<HiddenT>(ub_base, row_bytes, H);
    LocalTensor<HiddenT> store0 =
        CombineMakeUbTensor<HiddenT>(ub_base, 2U * row_bytes, H);
    LocalTensor<HiddenT> store1 =
        CombineMakeUbTensor<HiddenT>(ub_base, 3U * row_bytes, H);
    LocalTensor<float> acc =
        CombineMakeUbTensor<float>(ub_base, 4U * row_bytes, H);
    LocalTensor<float> tmp = CombineMakeUbTensor<float>(
        ub_base, 4U * row_bytes + H * sizeof(float), H);

    constexpr uint32_t kDstUbCap =
        kCombineProloguePlanBytes / sizeof(int32_t);
    const uint32_t max_tok_window = kDstUbCap / K;
    if (max_tok_window == 0U) {
        return;
    }
    LocalTensor<int32_t> ubDst = CombineMakeUbTensor<int32_t>(
        ub_base, kCombineProloguePlanOffset, kDstUbCap);

    const __gm__ HiddenT* gathered_hidden = reinterpret_cast<const __gm__ HiddenT*>(
        workspace);

    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
    uint32_t store_stage = 0;
    bool ping_pending = false;
    bool pong_pending = false;
    bool preload_issued = false;

    uint32_t src_rank[kCombinePrologueMaxSrcs];
    uint32_t src_loff[kCombinePrologueMaxSrcs];
    uint32_t src_route[kCombinePrologueMaxSrcs];
    uint32_t n_src = 0;

    for (uint32_t win_start = tok_start; win_start < tok_end;
         win_start += max_tok_window) {
        uint32_t win_end = win_start + max_tok_window;
        if (win_end > tok_end) {
            win_end = tok_end;
        }
        if (ping_pending) {
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            ping_pending = false;
        }
        if (pong_pending) {
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
            pong_pending = false;
        }
        preload_issued = false;

        const uint32_t win_elems = (win_end - win_start) * K;
        CombineCopyGmToUbForScalar(
            ubDst, dst + static_cast<uint64_t>(win_start) * K, win_elems);

        n_src = Phase3DecodeTokenSrcs(
            ubDst, 0, win_start, K, R, NvS,
            src_rank, src_loff, src_route);

        for (uint32_t token_id = win_start; token_id < win_end; ++token_id) {
            if (n_src == 0U) {
                Duplicate(acc, 0.0f, static_cast<int32_t>(H));
                PipeBarrier<PIPE_V>();
            } else {
                if (!preload_issued) {
                    if (ping_pending) {
                        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                        ping_pending = false;
                    }
                    SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
                    WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
                    Phase3IssueHiddenLoad<HiddenT>(
                        row0, published_hidden, gathered_hidden,
                        src_rank[0], src_loff[0], src_route[0], H, rank,
                        EVENT_ID0);
                }
                preload_issued = false;

                int32_t cur = 0;
                for (uint32_t i = 0; i < n_src; ++i) {
                    TEventID cur_ev = cur == 0 ? EVENT_ID0 : EVENT_ID1;
                    LocalTensor<HiddenT> cur_row = cur == 0 ? row0 : row1;
                    WaitFlag<HardEvent::MTE2_V>(cur_ev);

                    uint32_t next_i = i + 1U;
                    if (next_i < n_src) {
                        int32_t next_buf = 1 - cur;
                        TEventID next_ev =
                            next_buf == 0 ? EVENT_ID0 : EVENT_ID1;
                        LocalTensor<HiddenT> next_row =
                            next_buf == 0 ? row0 : row1;
                        if (next_buf == 0 && ping_pending) {
                            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                            ping_pending = false;
                        } else if (next_buf == 1 && pong_pending) {
                            WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
                            pong_pending = false;
                        }
                        Phase3IssueHiddenLoad<HiddenT>(
                            next_row, published_hidden, gathered_hidden,
                            src_rank[next_i], src_loff[next_i],
                            src_route[next_i], H, rank, next_ev);
                    }

                    if (i == 0U) {
                        Cast(acc, cur_row, RoundMode::CAST_NONE,
                             static_cast<int32_t>(H));
                        SetFlag<HardEvent::V_MTE2>(cur_ev);
                        if (cur == 0) {
                            ping_pending = true;
                        } else {
                            pong_pending = true;
                        }
                    } else {
                        Cast(tmp, cur_row, RoundMode::CAST_NONE,
                             static_cast<int32_t>(H));
                        SetFlag<HardEvent::V_MTE2>(cur_ev);
                        if (cur == 0) {
                            ping_pending = true;
                        } else {
                            pong_pending = true;
                        }
                        Add(acc, acc, tmp, static_cast<int32_t>(H));
                    }
                    cur = 1 - cur;
                }
            }

            PipeBarrier<PIPE_V>();
            TEventID store_ev =
                store_stage == 0U ? EVENT_ID2 : EVENT_ID3;
            LocalTensor<HiddenT> store_row =
                store_stage == 0U ? store0 : store1;
            WaitFlag<HardEvent::MTE3_MTE2>(store_ev);
            Cast(store_row, acc, RoundMode::CAST_RINT, static_cast<int32_t>(H));
            SetFlag<HardEvent::V_MTE3>(store_ev);

            const uint32_t next_token = token_id + 1U;
            if (next_token < win_end) {
                n_src = Phase3DecodeTokenSrcs(
                    ubDst, (next_token - win_start) * K, next_token, K, R, NvS,
                    src_rank, src_loff, src_route);
                if (n_src > 0U) {
                    if (ping_pending) {
                        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                        ping_pending = false;
                    }
                    SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
                    WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
                    Phase3IssueHiddenLoad<HiddenT>(
                        row0, published_hidden, gathered_hidden,
                        src_rank[0], src_loff[0], src_route[0], H, rank,
                        EVENT_ID0);
                    preload_issued = true;
                }
            } else {
                n_src = 0;
            }

            WaitFlag<HardEvent::V_MTE3>(store_ev);
            CombineIssueUbToGm(
                output_sh + static_cast<uint64_t>(token_id) * H,
                store_row,
                H);
            SetFlag<HardEvent::MTE3_MTE2>(store_ev);
            store_stage ^= 1U;
        }
    }

    if (ping_pending) {
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }
    if (pong_pending) {
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
}

#if MOONEP_UDMA_PROFILE_STATISTIC
__aicore__ inline void MoonEPUdmaCombinePrintHiddenBandwidth(
    GM_ADDR bandwidth_scratch,
    const CombineGatherResult& gather_result,
    uint64_t gather_cycles,
    uint32_t rank,
    uint32_t world_size,
    uint32_t core_idx,
    uint32_t core_num)
{
    if (bandwidth_scratch == nullptr) {
        return;
    }

    AscendC::GlobalTensor<int64_t> statsGlobal;
    statsGlobal.SetGlobalBuffer(
        reinterpret_cast<__gm__ int64_t*>(bandwidth_scratch));
    AscendC::GlobalTensor<int64_t> localStats =
        statsGlobal[core_idx * kCombineBandwidthStatsStride];
    localStats(kCombineStatHiddenBytes) =
        static_cast<int64_t>(gather_result.hidden_bytes);
    localStats(kCombineStatGatherCycles) =
        static_cast<int64_t>(gather_cycles);
    localStats(kCombineStatHiddenSubmitCycles) =
        static_cast<int64_t>(gather_result.hidden_submit_cycles);
    localStats(kCombineStatWeightGetCycles) =
        static_cast<int64_t>(gather_result.weight_get_cycles);
    localStats(kCombineStatWeightSyncCycles) =
        static_cast<int64_t>(gather_result.weight_sync_cycles);
    localStats(kCombineStatWeightAssembleCycles) =
        static_cast<int64_t>(gather_result.weight_assemble_cycles);
    localStats(kCombineStatHiddenQuietCycles) =
        static_cast<int64_t>(gather_result.hidden_quiet_cycles);
    localStats(kCombineStatFinalSyncCycles) =
        static_cast<int64_t>(gather_result.final_sync_cycles);
    AscendC::PipeBarrier<PIPE_ALL>();
    __asm__ __volatile__("");
    AscendC::DataCacheCleanAndInvalid<
        int64_t,
        AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(localStats[0]);
    __asm__ __volatile__("");

    CombineLocalAivBarrier();
    if (core_idx == 0U) {
        uint64_t rank_hidden_bytes = 0;
        uint64_t max_gather_cycles = 0;
        uint64_t max_phase_cycles[6] = {0, 0, 0, 0, 0, 0};
        uint32_t max_phase_aiv[6] = {0, 0, 0, 0, 0, 0};
        for (uint32_t aiv_idx = 0; aiv_idx < core_num; ++aiv_idx) {
            AscendC::GlobalTensor<int64_t> coreStats =
                statsGlobal[aiv_idx * kCombineBandwidthStatsStride];
            __asm__ __volatile__("");
            AscendC::DataCacheCleanAndInvalid<
                int64_t,
                AscendC::CacheLine::SINGLE_CACHE_LINE,
                AscendC::DcciDst::CACHELINE_OUT>(coreStats[0]);
            __asm__ __volatile__("");
            const uint64_t core_hidden_bytes =
                static_cast<uint64_t>(
                    coreStats.GetValue(kCombineStatHiddenBytes));
            const uint64_t core_gather_cycles =
                static_cast<uint64_t>(
                    coreStats.GetValue(kCombineStatGatherCycles));
            rank_hidden_bytes += core_hidden_bytes;
            if (core_gather_cycles > max_gather_cycles) {
                max_gather_cycles = core_gather_cycles;
            }
            for (uint32_t phase_idx = 0; phase_idx < 6U; ++phase_idx) {
                const uint64_t phase_cycles = static_cast<uint64_t>(
                    coreStats.GetValue(
                        kCombineStatHiddenSubmitCycles + phase_idx));
                if (phase_cycles > max_phase_cycles[phase_idx]) {
                    max_phase_cycles[phase_idx] = phase_cycles;
                    max_phase_aiv[phase_idx] = aiv_idx;
                }
            }
        }

        const float gather_time_us =
            static_cast<float>(max_gather_cycles) /
            kCombineCyclesPerMicrosecond;
        const float hidden_bandwidth_gbps = gather_time_us > 0.0F
            ? static_cast<float>(rank_hidden_bytes) /
                gather_time_us / 1000.0F
            : 0.0F;
        AscendC::printf(
            "[moonep_combine][dispatch-group-2qp-v1] rank=%u/%u "
            "GatherToStagingPhase hidden: "
            "data_bytes=%lld, time_us=%f, bandwidth=%f GB/s\n",
            rank,
            world_size,
            static_cast<long long>(rank_hidden_bytes),
            gather_time_us,
            hidden_bandwidth_gbps);

        AscendC::printf(
            "[moonep_combine] rank=%u phase_max_us: "
            "hidden_submit=%f(aiv=%u), weight_get=%f(aiv=%u), "
            "weight_sync=%f(aiv=%u)\n",
            rank,
            static_cast<float>(max_phase_cycles[0]) /
                kCombineCyclesPerMicrosecond,
            max_phase_aiv[0],
            static_cast<float>(max_phase_cycles[1]) /
                kCombineCyclesPerMicrosecond,
            max_phase_aiv[1],
            static_cast<float>(max_phase_cycles[2]) /
                kCombineCyclesPerMicrosecond,
            max_phase_aiv[2]);
        AscendC::printf(
            "[moonep_combine] rank=%u phase_max_us: "
            "weight_assemble=%f(aiv=%u), hidden_quiet=%f(aiv=%u), "
            "final_sync=%f(aiv=%u)\n",
            rank,
            static_cast<float>(max_phase_cycles[3]) /
                kCombineCyclesPerMicrosecond,
            max_phase_aiv[3],
            static_cast<float>(max_phase_cycles[4]) /
                kCombineCyclesPerMicrosecond,
            max_phase_aiv[4],
            static_cast<float>(max_phase_cycles[5]) /
                kCombineCyclesPerMicrosecond,
            max_phase_aiv[5]);

        AscendC::printf(
            "[moonep_combine] rank=%u group_width=%u group_count=%u "
            "slowest_group_aiv=%u\n",
            rank,
            CombineGetGroupWidth(world_size),
            world_size / CombineGetGroupWidth(world_size),
            max_phase_aiv[0]);
    }
    // Keep other AIVs from reusing the buffer while AIV0 reads the reduction.
    CombineLocalAivBarrier();
}
#endif

// Args order: buffers, workspace, bandwidth scratch, tiling.
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void moonep_udma_combine_kernel(
    GM_ADDR hidden_nvsh_gm,       // bf16 [NvS, H]
    GM_ADDR route_weights_nvs_gm, // fp32 [NvS]
    GM_ADDR dst_gm,               // int32 [S*K]
    GM_ADDR dup_groups_gm,        // int32 [NvS, 3]
    GM_ADDR dup_loffs_gm,         // int32 [NvS]
    GM_ADDR dup_counts_gm,        // int32 [2]
    GM_ADDR output_sh_gm,         // bf16 [S, H]
    GM_ADDR output_sk_gm,         // fp32 [S, K]
    GM_ADDR sync_buffer_gm,       // symmetric grant/done cacheline pool
    GM_ADDR workspace_gm,         // rank-local GM gather workspace
    GM_ADDR bandwidth_scratch,    // int64 bandwidth scratch, 1 line per AIV
    ascend_moonep::MoonEPUdmaCombineTilingData tiling,
    uint64_t invocation_magic)
{
    uint32_t S = tiling.S;
    uint32_t H = tiling.H;
    uint32_t K = tiling.K;
    uint32_t NvS = S * K;
    uint32_t with_weights = tiling.with_weights;
    uint32_t rank = tiling.rank;

    uint32_t core_idx = get_block_idx();

    LocalTensor<uint8_t> ub_base;
    ub_base.address_.logicPos = static_cast<uint8_t>(TPosition::VECIN);
    ub_base.address_.bufferAddr = 0;
    ub_base.address_.dataLen = kCombineUbBytes;

    const __gm__ bf16_t* hidden_nvsh = reinterpret_cast<const __gm__ bf16_t*>(hidden_nvsh_gm);
    const __gm__ float* rw = reinterpret_cast<const __gm__ float*>(route_weights_nvs_gm);
    const __gm__ int32_t* dst_ptr = reinterpret_cast<const __gm__ int32_t*>(dst_gm);
    const __gm__ int32_t* dgrp = reinterpret_cast<const __gm__ int32_t*>(dup_groups_gm);
    const __gm__ int32_t* dloff = reinterpret_cast<const __gm__ int32_t*>(dup_loffs_gm);
    const __gm__ int32_t* dcnt = reinterpret_cast<const __gm__ int32_t*>(dup_counts_gm);
    __gm__ bf16_t* output_sh = reinterpret_cast<__gm__ bf16_t*>(output_sh_gm);
    __gm__ float* output_sk = reinterpret_cast<__gm__ float*>(output_sk_gm);

    // Inputs are always SHMEM symmetric. Buffer(use_udma=True).combine() publishes
    // ordinary tensors into its symmetric shard before launching this kernel.
    __gm__ bf16_t* published_hidden =
        const_cast<__gm__ bf16_t*>(hidden_nvsh);
    __gm__ float* published_weights = const_cast<__gm__ float*>(rw);

    PlanProloguePhase<bf16_t>(
        ub_base, published_hidden, dgrp, dloff, dcnt,
        H, NvS, core_idx, kCombineAivNum);

    // Every prologue GM write is already completed by the MTE3 dependency in
    // CombineStoreAccAsBf16, and PlanProloguePhase ends with PIPE_ALL. Avoid
    // aclshmemx_mte_quiet here: it repeats the pipe drain and flushes the whole
    // data cache on every AIV before the cross-rank barrier.
    aclshmemx_barrier_all_vec();

#if MOONEP_UDMA_PROFILE_STATISTIC
    for (int i = 0; i < 1; i++) {
        (void)GatherToStagingPhase<bf16_t>(
        ub_base, published_hidden, published_weights,
        reinterpret_cast<__gm__ uint8_t*>(sync_buffer_gm),
        workspace_gm, dst_ptr, output_sk,
        H, tiling.R, NvS, false,
        invocation_magic * 2U,
        rank, core_idx, kCombineAivNum);
    }
    const uint64_t gather_start_cycle = AscendC::GetSystemCycle();
#endif

    CombineGatherResult gather_result{};
    for (int i = 0; i < 1; i++) {
        const CombineGatherResult iter_result =
            GatherToStagingPhase<bf16_t>(
                ub_base, published_hidden, published_weights,
                reinterpret_cast<__gm__ uint8_t*>(sync_buffer_gm),
                workspace_gm, dst_ptr, output_sk,
                H, tiling.R, NvS, with_weights != 0,
                invocation_magic * 2U + 1U,
                rank, core_idx, kCombineAivNum);
        gather_result.hidden_bytes += iter_result.hidden_bytes;
#if MOONEP_UDMA_PROFILE_STATISTIC
        gather_result.hidden_submit_cycles +=
            iter_result.hidden_submit_cycles;
        gather_result.weight_get_cycles += iter_result.weight_get_cycles;
        gather_result.weight_sync_cycles += iter_result.weight_sync_cycles;
        gather_result.weight_assemble_cycles +=
            iter_result.weight_assemble_cycles;
        gather_result.hidden_quiet_cycles += iter_result.hidden_quiet_cycles;
        gather_result.final_sync_cycles += iter_result.final_sync_cycles;
#endif
    }

#if MOONEP_UDMA_PROFILE_STATISTIC
    const uint64_t gather_end_cycle = AscendC::GetSystemCycle();
    MoonEPUdmaCombinePrintHiddenBandwidth(
        bandwidth_scratch,
        gather_result,
        gather_end_cycle - gather_start_cycle,
        rank,
        tiling.R,
        core_idx,
        kCombineAivNum);
#endif

    GatherAccumulatePhase<bf16_t>(
        ub_base, published_hidden,
        workspace_gm, dst_ptr, output_sh,
        S, H, K, tiling.R, NvS,
        rank, core_idx, kCombineAivNum);
}


// ============================================================
// Host-side launcher (C-linkage)
// ============================================================

extern "C" __attribute__((visibility("default")))
void moonep_udma_combine_kernel_do(
    void* stream,
    uint8_t* hidden_nvsh,
    uint8_t* route_weights_nvs,
    uint8_t* sync_buffer,
    uint8_t* dst,
    uint8_t* dup_groups,
    uint8_t* dup_loffs,
    uint8_t* dup_counts,
    uint8_t* output_sh,
    uint8_t* output_sk,
    uint8_t* local_workspace,
    uint8_t* bandwidth_scratch,
    uint8_t* tiling_host,
    uint64_t invocation_magic)
{
    auto* td = reinterpret_cast<ascend_moonep::MoonEPUdmaCombineTilingData*>(tiling_host);

    moonep_udma_combine_kernel<<<kCombineAivNum, nullptr, stream>>>(
        hidden_nvsh,
        route_weights_nvs,
        dst,
        dup_groups,
        dup_loffs,
        dup_counts,
        output_sh,
        output_sk,
        sync_buffer,
        local_workspace,
        bandwidth_scratch,
        *td,
        invocation_magic);
}
