#pragma once

// Shared, architecture-neutral declarations for the planning kernel.
//
// The planning kernel has one backend per instruction-set family, selected at
// build time (see native/CMakeLists.txt):
//   planning_simt.cpp    A5 / Ascend950 / 3510 / SIMT
//   planning_aicore.cpp  A2/A3 / Ascend910B / 910_93 / 2201 / non-SIMT
// Both include this header for the pieces that are byte-identical across
// families: tuning constants, UB-layout structs, the DstSimtContextIndex enum,
// and the arch-independent __aicore__ helpers below.
//
// Only declarations that are truly family-independent live here. Anything that
// touches Catlass::Arch::Resource<ArchTag> (a family-specific type) either is a
// template on the resource type (so the family .cpp deduces it) or stays in the
// family .cpp. The family .cpp files keep their own arch-specific kernels
// (*Simt / *Reg for SIMT, *A2 for non-SIMT) and the orchestration helpers whose
// bodies diverge between families.

#include "acl/acl.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "kernel_operator.h"
#include "shmem.h"

#include "ascend_moonep/kernels/arch_config.h"
#include "ascend_moonep/kernels/launch.h"
#include "ascend_moonep/kernels/types.h"

namespace ascend_moonep {
namespace planning {

constexpr int32_t kFillTileElems = 1024; // 需要32B对齐
constexpr int32_t kInt32Bytes = 4;
constexpr int32_t kMaxPlanningCores = 56;
constexpr int32_t kMaxPlanningRanks = 1024;
constexpr int32_t kPlanningUbSizeBytes =
    static_cast<int32_t>(ascend_moonep::kPlanningUbSizeBytes);
constexpr int32_t kCoreCursorSimtMaxThreads = 256;
constexpr int32_t kCoreCursorThreadPrefixBytes =
    kCoreCursorSimtMaxThreads * kInt32Bytes;
constexpr int32_t kCoreCursorCarryBytes = 32;
constexpr int32_t kCursorWorkerScratchRowElems = 32 / kInt32Bytes;
constexpr int32_t kDupSrcTileElems = 1024;
constexpr int32_t kSourceDupSimtMaxThreads = 1024;
constexpr int32_t kSourceDupInboxSlotElems = 8;
constexpr int32_t kPrefetchSimtMaxThreads = 1024;
constexpr int32_t kPrefetchSimtItemsPerThread = 32;
constexpr int32_t kRouteHistogramTileElems = 4096;
constexpr int32_t kTpeReduceExpertsPerCore = 64;
constexpr int32_t kOwnerReduceTotalBytes = 32;
constexpr int32_t kTpeReduceRegRowBytes =
    kTpeReduceExpertsPerCore * kInt32Bytes;
constexpr int32_t kSourceDupPackTinyChunkGroups = 16;
constexpr int32_t kSourceDupPackSmallChunkGroups = 32;
constexpr int32_t kSourceDupPackMediumChunkGroups = 64;
constexpr int32_t kSourceDupPackLargeChunkGroups = 128;
constexpr int32_t kSourceDupPackXLargeChunkGroups = 256;
constexpr int32_t kDstSimtMaxThreads = 1024;
constexpr int32_t kDstSimtMinChunkElems = 32;
constexpr int32_t kDstSimtMaxChunkElems = 128;
constexpr int32_t kDstSimtContextElems = 16;
constexpr int32_t kDstSimtContextBytes = kDstSimtContextElems * kInt32Bytes;
constexpr int32_t kMaxDuplicateInt32Elems =
    kPlanningUbSizeBytes / kInt32Bytes;
static_assert((kDupSrcTileElems * kInt32Bytes) % 32 == 0);
static_assert(
    (kPrefetchSimtMaxThreads & (kPrefetchSimtMaxThreads - 1)) == 0);
static_assert(kDstSimtContextBytes % 32 == 0);
static_assert((kRouteHistogramTileElems * kInt32Bytes) % 32 == 0);
static_assert(
    kRouteHistogramTileElems * kInt32Bytes < kPlanningUbSizeBytes);
static_assert(kOwnerReduceTotalBytes % 32 == 0);
static_assert(kTpeReduceRegRowBytes == 256);
static_assert(
    kTpeReduceRegRowBytes + kMaxPlanningRanks * 32 <=
    kPlanningUbSizeBytes);
static_assert(
    (kMaxPlanningCores + 1) * 32 + kCoreCursorThreadPrefixBytes +
            kCoreCursorCarryBytes <=
        kPlanningUbSizeBytes);
static_assert(kCoreCursorThreadPrefixBytes % 32 == 0);
static_assert(kCoreCursorCarryBytes % 32 == 0);
static_assert(kCursorWorkerScratchRowElems * kInt32Bytes == 32);
static_assert(kSourceDupInboxSlotElems * kInt32Bytes == 32);
static_assert(
    kMaxPlanningRanks * kSourceDupInboxSlotElems * kInt32Bytes + 32 <=
    kPlanningUbSizeBytes);

// Pointer fields are int32 element offsets from the single UB workspace base.

enum DstSimtContextIndex : int32_t {
    kDstSimtTopkOffset = 0,
    kDstSimtHistOffset,
    kDstSimtOrdinalOffset,
    kDstSimtAllocCumsumOffset,
    kDstSimtExpertOffsetsOffset,
    kDstSimtDstOffset,
    kDstSimtChunkSize = 9,
    kDstSimtHistStride,
    kDstSimtRankCount,
    kDstSimtExpertCount,
    kDstSimtMaxTokensPerRank,
};

struct DstUbLayout {
    int32_t context_offset;
    int32_t dst_tile_offset;
    int32_t lookup_offset;
    int32_t gm_simt_topk_offset;
    int32_t lookup_table_elems;
    int32_t lookup_table_bytes;
    int32_t simt_topk_offset;
    int32_t simt_hist_offset;
    int32_t simt_hist_stride;
    int32_t simt_chunk_size;
    bool use_simt;
};

struct SourceDupConfig {
    int32_t rank;
    int32_t rank_count;
    int32_t token_count;
    int32_t topk;
    int32_t max_tokens_per_rank;
    int32_t core_idx;
    int32_t core_num;
    int32_t tokens_per_tile;
};

struct SourceDupPartition {
    int32_t token_begin;
    int32_t token_end;
    int32_t max_tokens_per_core;
    int32_t max_tiles_per_core;
};

struct SourceDupUbLayout {
    int32_t route_tile_elems;
    int32_t count_bytes;
    int32_t staging_bytes;
    int32_t prefix_offset;
    int32_t meta_offset;
    int32_t raw_group_offset;
    int32_t raw_loff_offset;
    int32_t group_send_offset;
    int32_t tile_count_offset;
    int32_t tile_count_row_elems;
    int32_t token_group_count_offset;
    int32_t token_loff_count_offset;
    int32_t token_group_prefix_offset;
    int32_t token_loff_prefix_offset;
    int32_t compact_loff_offset;
    int32_t pack_hist_offset;
    int32_t pack_hist_row_elems;
    int32_t pack_chunk_groups;
    int32_t total_bytes;
};

struct SourceDupWorkspaceLayout {
    int32_t tile_count_offset;
    int32_t group_stride;
    int32_t loff_stride;
    int32_t group_offset;
    int32_t loff_offset;
    int32_t prebuilt_group_stride;
    int32_t prebuilt_loff_stride;
    int32_t prebuilt_group_offset;
    int32_t prebuilt_loff_offset;
};

struct SourceDupCountTileLayout {
    int32_t max_ranks_per_core;
    int32_t row_elems;
    int32_t core_counts_offset;
    int32_t slots_offset;
    int32_t result_offset;
    int32_t total_bytes;
};

struct PrefetchUbLayout {
    int32_t group_counts_offset;
    int32_t group_counts_bytes;
    int32_t selected_experts_offset;
    int32_t selected_experts_bytes;
    int32_t scratch_counts_offset;
    int32_t scratch_experts_offset;
    int32_t scratch_row_elems;
    int32_t scratch_row_bytes;
    int32_t total_bytes;
};

// Kernel-local metadata and GM views for this architecture block.
// Keeping this state in one descriptor separates pointer wiring from stages.
struct PlanningKernelState {
    int32_t core_idx;
    int32_t core_num;
    int32_t rank;
    int32_t rank_count;
    int32_t token_count;
    int32_t topk;
    int32_t expert_count;
    int32_t max_tokens_per_rank;
    int32_t prefetch_count;
    int32_t experts_per_rank;
    int32_t route_items;
    int32_t tpe_offset;
    int32_t count_inbox_offset;
    int32_t prefix_inbox_offset;
    int32_t payload_done_flag_offset;
    int32_t groups_scratch_offset;
    int32_t loffs_scratch_offset;
    int32_t source_dup_tokens_per_tile;
    int32_t owner_reduce_tile_elems;
    int32_t route_core_num;
    int32_t items_per_core;
    int32_t begin;
    int32_t end;

    AscendC::GlobalTensor<int32_t> topk_gm;
    AscendC::GlobalTensor<int32_t> tpe_gm;
    AscendC::GlobalTensor<int32_t> dst_gm;
    AscendC::GlobalTensor<int32_t> cu_gm;
    AscendC::GlobalTensor<int32_t> experts_to_copy_gm;
    AscendC::GlobalTensor<int32_t> dup_groups_gm;
    AscendC::GlobalTensor<int32_t> dup_loffs_gm;
    AscendC::GlobalTensor<int32_t> dup_counts_gm;
    AscendC::GlobalTensor<int32_t> tpe_all_gm;
    AscendC::GlobalTensor<int32_t> tpe_prefix_gm;
    AscendC::GlobalTensor<int32_t> balance_gm;
    AscendC::GlobalTensor<int32_t> z_gm;
    AscendC::GlobalTensor<int32_t> alloc_gm;
    AscendC::GlobalTensor<int32_t> alloc_cumsum_gm;
    AscendC::GlobalTensor<int32_t> expert_offsets_gm;
    AscendC::GlobalTensor<int32_t> core_expert_counts_gm;
    AscendC::GlobalTensor<int32_t> cursor_worker_scratch_gm;
    AscendC::GlobalTensor<int32_t> dup_workspace_gm;
    AscendC::GlobalTensor<int32_t> dup_key_workspace_gm;
    AscendC::GlobalTensor<int32_t> local_shmem_gm;
    __gm__ int32_t* shmem_base;
};

__aicore__ inline int32_t IntMin(int32_t a, int32_t b) {
    return a < b ? a : b;
}

__aicore__ inline int32_t AlignUpBytes(int32_t value, int32_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

__aicore__ inline int32_t AlignUpInt32(int32_t value) {
    return ((value + 7) / 8) * 8;
}

__aicore__ inline void InitDstUbLayout(
    int32_t rank_count,
    int32_t expert_count,
    int32_t expert_row_bytes,
    DstUbLayout& layout) {
    layout.context_offset = expert_row_bytes;
    const int32_t dst_tile_bytes =
        AlignUpBytes(kFillTileElems * kInt32Bytes, 32);
    layout.dst_tile_offset =
        layout.context_offset + kDstSimtContextBytes;
    layout.lookup_offset = layout.dst_tile_offset + dst_tile_bytes;
    layout.gm_simt_topk_offset = layout.lookup_offset;

    const int64_t lookup_table_elems =
        static_cast<int64_t>(rank_count) * expert_count;
    const int64_t lookup_table_data_bytes =
        lookup_table_elems * kInt32Bytes;
    const int64_t lookup_table_bytes =
        ((lookup_table_data_bytes + 31) / 32) * 32;
    const int64_t lookup_end_offset =
        static_cast<int64_t>(layout.lookup_offset) +
        lookup_table_bytes * 2;
    const bool lookup_fits = lookup_end_offset <= kPlanningUbSizeBytes;
    layout.lookup_table_elems = lookup_fits
        ? static_cast<int32_t>(lookup_table_elems)
        : 0;
    layout.lookup_table_bytes = lookup_fits
        ? static_cast<int32_t>(lookup_table_bytes)
        : 0;

    const int32_t simt_topk_bytes =
        AlignUpBytes(kFillTileElems * kInt32Bytes, 32);
    layout.simt_topk_offset = lookup_fits
        ? AlignUpBytes(static_cast<int32_t>(lookup_end_offset), 32)
        : 0;
    layout.simt_hist_offset = lookup_fits
        ? layout.simt_topk_offset + simt_topk_bytes
        : 0;
    layout.simt_hist_stride = AlignUpInt32(expert_count);
    layout.simt_chunk_size = kDstSimtMinChunkElems;
    if (lookup_fits) {
        int64_t simt_end_offset = 0;
        while (true) {
            const int32_t chunk_count =
                (kFillTileElems + layout.simt_chunk_size - 1) /
                layout.simt_chunk_size;
            simt_end_offset =
                static_cast<int64_t>(layout.simt_hist_offset) +
                static_cast<int64_t>(chunk_count) *
                    layout.simt_hist_stride * kInt32Bytes;
            if (simt_end_offset <= kPlanningUbSizeBytes ||
                layout.simt_chunk_size >= kDstSimtMaxChunkElems) {
                break;
            }
            layout.simt_chunk_size *= 2;
        }
        layout.use_simt = simt_end_offset <= kPlanningUbSizeBytes;
    } else {
        layout.use_simt = false;
    }
}

__aicore__ inline int32_t SelectPrefetchSimtThreadCount(int32_t expert_count) {
    int32_t required_threads =
        (expert_count + kPrefetchSimtItemsPerThread - 1) /
        kPrefetchSimtItemsPerThread;
    int32_t thread_count = 32;
    while (thread_count < required_threads &&
           thread_count < kPrefetchSimtMaxThreads) {
        thread_count <<= 1;
    }
    return thread_count;
}

__aicore__ inline void InitPrefetchUbLayout(
    int32_t expert_count,
    int32_t prefetch_count,
    PrefetchUbLayout& layout) {
    layout.group_counts_offset = 0;
    layout.group_counts_bytes =
        AlignUpBytes((expert_count + prefetch_count) * kInt32Bytes, 32);
    layout.selected_experts_offset = layout.group_counts_bytes;
    layout.selected_experts_bytes =
        AlignUpBytes(prefetch_count * kInt32Bytes, 32);
    layout.scratch_counts_offset =
        layout.selected_experts_offset + layout.selected_experts_bytes;
    const int32_t scratch_elems =
        prefetch_count > kPrefetchSimtMaxThreads
            ? prefetch_count
            : kPrefetchSimtMaxThreads;
    layout.scratch_row_elems = AlignUpInt32(scratch_elems);
    layout.scratch_row_bytes = layout.scratch_row_elems * kInt32Bytes;
    layout.scratch_experts_offset =
        layout.scratch_counts_offset + layout.scratch_row_bytes;
    layout.total_bytes =
        layout.scratch_experts_offset + layout.scratch_row_bytes;
}

__aicore__ inline void InitSourceDupCountTileLayout(
    const SourceDupConfig& config,
    SourceDupCountTileLayout& layout) {
    layout.max_ranks_per_core =
        (config.rank_count + config.core_num - 1) / config.core_num;
    layout.row_elems = AlignUpInt32(layout.max_ranks_per_core * 2);
    layout.core_counts_offset = 0;
    layout.slots_offset =
        config.core_num * layout.row_elems * kInt32Bytes;
    layout.result_offset = layout.slots_offset + AlignUpBytes(
        layout.max_ranks_per_core * kSourceDupInboxSlotElems * kInt32Bytes,
        32);
    layout.total_bytes = layout.result_offset + 32;
}

__aicore__ inline void GetSourceDupRankPartition(
    const SourceDupConfig& config,
    int32_t& rank_begin,
    int32_t& rank_count) {
    const int32_t ranks_per_core = config.rank_count / config.core_num;
    const int32_t tail_cores =
        config.rank_count - ranks_per_core * config.core_num;
    rank_begin =
        config.core_idx * ranks_per_core + IntMin(config.core_idx, tail_cores);
    rank_count = ranks_per_core + (config.core_idx < tail_cores ? 1 : 0);
}

__aicore__ inline void LoadSourceDupCoreCountTile(
    const SourceDupConfig& config,
    const SourceDupCountTileLayout& layout,
    int32_t dest_begin,
    int32_t dest_count,
    AscendC::LocalTensor<int32_t>& core_counts_ub,
    AscendC::GlobalTensor<int32_t>& core_counts_gm) {
    const int32_t active_elems = dest_count * 2;
    AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
    AscendC::DataCopyExtParams copy_params(
        1,
        static_cast<uint32_t>(active_elems * kInt32Bytes),
        0,
        0,
        0);
    for (int32_t source_core = 0;
         source_core < config.core_num;
         ++source_core) {
        AscendC::DataCopyPad(
            core_counts_ub[source_core * layout.row_elems],
            core_counts_gm[
                (source_core * config.rank_count + dest_begin) * 2],
            copy_params,
            pad_params);
    }
}

__aicore__ inline void StoreSourceDupCoreCountTile(
    const SourceDupConfig& config,
    const SourceDupCountTileLayout& layout,
    int32_t dest_begin,
    int32_t dest_count,
    AscendC::GlobalTensor<int32_t>& core_counts_gm,
    AscendC::LocalTensor<int32_t>& core_counts_ub) {
    const int32_t active_elems = dest_count * 2;
    AscendC::DataCopyExtParams copy_params(
        1,
        static_cast<uint32_t>(active_elems * kInt32Bytes),
        0,
        0,
        0);
    for (int32_t source_core = 0;
         source_core < config.core_num;
         ++source_core) {
        AscendC::DataCopyPad(
            core_counts_gm[
                (source_core * config.rank_count + dest_begin) * 2],
            core_counts_ub[source_core * layout.row_elems],
            copy_params);
    }
}

__aicore__ inline int32_t SelectSourceDupPackChunkGroups(
    int32_t rank_count,
    int32_t topk) {
    if (rank_count <= 32 && topk <= 8) {
        return kSourceDupPackTinyChunkGroups;
    }
    if (rank_count <= 32) {
        return kSourceDupPackSmallChunkGroups;
    }
    if (rank_count <= 128) {
        return kSourceDupPackMediumChunkGroups;
    }
    if (rank_count <= 256) {
        return kSourceDupPackLargeChunkGroups;
    }
    return kSourceDupPackXLargeChunkGroups;
}

__aicore__ inline void InitSourceDupPartition(
    const SourceDupConfig& config,
    SourceDupPartition& partition) {
    const int32_t tokens_per_core = config.token_count / config.core_num;
    const int32_t token_tail_cores =
        config.token_count - tokens_per_core * config.core_num;
    partition.token_begin =
        config.core_idx * tokens_per_core + IntMin(config.core_idx, token_tail_cores);
    const int32_t token_assigned =
        tokens_per_core + (config.core_idx < token_tail_cores ? 1 : 0);
    partition.token_end = partition.token_begin + token_assigned;
    partition.max_tokens_per_core =
        (config.token_count + config.core_num - 1) / config.core_num;
    partition.max_tiles_per_core =
        (partition.max_tokens_per_core + config.tokens_per_tile - 1) /
        config.tokens_per_tile;

}

__aicore__ inline void InitSourceDupUbLayout(
    const SourceDupConfig& config,
    SourceDupUbLayout& layout) {
    const int32_t rank_count = config.rank_count;
    const int32_t tokens_per_tile = config.tokens_per_tile;
    layout.route_tile_elems = tokens_per_tile * config.topk;
    layout.count_bytes = AlignUpBytes(rank_count * 2 * kInt32Bytes, 32);

    const int32_t staging_elems =
        layout.route_tile_elems + rank_count * 7 > rank_count * 2
            ? layout.route_tile_elems + rank_count * 7
            : rank_count * 2;
    layout.staging_bytes = AlignUpBytes(staging_elems * kInt32Bytes, 32);
    layout.prefix_offset = layout.count_bytes + layout.staging_bytes;

    const int32_t prefix_bytes =
        AlignUpBytes(rank_count * 2 * kInt32Bytes, 32);
    layout.meta_offset = layout.prefix_offset + prefix_bytes;
    const int32_t meta_bytes =
        AlignUpBytes(rank_count * 5 * kInt32Bytes, 32);
    layout.raw_group_offset = layout.meta_offset + meta_bytes;

    const int32_t max_group_count = layout.route_tile_elems / 2;
    const int32_t group_send_elems =
        max_group_count * 3 + rank_count * 7 > rank_count * 2
            ? max_group_count * 3 + rank_count * 7
            : rank_count * 2;
    const int32_t raw_group_elems =
        max_group_count * 4 > group_send_elems
            ? max_group_count * 4
            : group_send_elems;
    const int32_t raw_group_bytes =
        AlignUpBytes(raw_group_elems * kInt32Bytes, 32);
    layout.raw_loff_offset = layout.raw_group_offset + raw_group_bytes;
    const int32_t raw_loff_elems = layout.route_tile_elems + rank_count * 7;
    const int32_t raw_loff_bytes =
        AlignUpBytes(raw_loff_elems * kInt32Bytes, 32);
    layout.group_send_offset = layout.raw_loff_offset + raw_loff_bytes;
    const int32_t group_send_bytes =
        AlignUpBytes(group_send_elems * kInt32Bytes, 32);
    layout.tile_count_offset = layout.group_send_offset + group_send_bytes;
    layout.tile_count_row_elems = AlignUpInt32(rank_count * 2);
    const int32_t tile_count_bytes =
        layout.tile_count_row_elems * kInt32Bytes;
    layout.token_group_count_offset = layout.tile_count_offset + tile_count_bytes;

    const int32_t token_row_bytes =
        AlignUpBytes(tokens_per_tile * kInt32Bytes, 32);
    layout.token_loff_count_offset =
        layout.token_group_count_offset + token_row_bytes;
    layout.token_group_prefix_offset =
        layout.token_loff_count_offset + token_row_bytes;
    layout.token_loff_prefix_offset =
        layout.token_group_prefix_offset + token_row_bytes;
    layout.compact_loff_offset =
        layout.token_loff_prefix_offset + token_row_bytes;
    layout.pack_hist_offset =
        layout.compact_loff_offset +
        AlignUpBytes(layout.route_tile_elems * kInt32Bytes, 32);
    layout.pack_hist_row_elems = AlignUpInt32(rank_count * 2);
    layout.pack_chunk_groups =
        SelectSourceDupPackChunkGroups(rank_count, config.topk);
    const int32_t max_pack_chunk_count =
        (max_group_count + layout.pack_chunk_groups - 1) /
        layout.pack_chunk_groups;
    layout.total_bytes =
        layout.pack_hist_offset +
        max_pack_chunk_count * layout.pack_hist_row_elems * kInt32Bytes;
}

__aicore__ inline void InitSourceDupWorkspaceLayout(
    const SourceDupConfig& config,
    const SourceDupPartition& partition,
    const SourceDupUbLayout& ub_layout,
    SourceDupWorkspaceLayout& layout) {
    layout.tile_count_offset = kMaxPlanningCores * config.rank_count * 2;
    const int32_t max_routes_per_core =
        partition.max_tokens_per_core * config.topk;
    layout.group_stride = (max_routes_per_core / 2) * 3;
    layout.loff_stride = max_routes_per_core;
    const int32_t prefix_elems = kMaxPlanningCores * config.rank_count * 2;
    layout.group_offset = prefix_elems;
    layout.loff_offset =
        layout.group_offset + config.core_num * layout.group_stride;
    const int32_t payload_elems =
        layout.loff_offset + config.core_num * layout.loff_stride;
    layout.prebuilt_group_stride = AlignUpInt32(
        (ub_layout.route_tile_elems / 2) * 3 + config.rank_count * 7);
    layout.prebuilt_loff_stride =
        AlignUpInt32(ub_layout.route_tile_elems + config.rank_count * 7);
    layout.prebuilt_group_offset = AlignUpInt32(payload_elems);
    const int32_t tile_slots =
        config.core_num * partition.max_tiles_per_core;
    layout.prebuilt_loff_offset =
        layout.prebuilt_group_offset +
        tile_slots * layout.prebuilt_group_stride;
}

__aicore__ inline void CleanInvalidDataCache() {
    AscendC::GlobalTensor<uint8_t> global;
    __asm__ __volatile__("");
    AscendC::DataCacheCleanAndInvalid<
        uint8_t,
        AscendC::CacheLine::ENTIRE_DATA_CACHE,
        AscendC::DcciDst::CACHELINE_OUT>(global);
    __asm__ __volatile__("");
}

__aicore__ inline void StoreContiguousGmThroughUb(
    AscendC::GlobalTensor<int32_t>& gm,
    AscendC::LocalTensor<int32_t>& ub,
    int32_t gm_offset,
    int32_t elem_count) {
    // Submission only: the caller owns producer and UB-reuse synchronization.
    if (elem_count <= 0) {
        return;
    }

    AscendC::DataCopyExtParams copy_params(
        1, static_cast<uint32_t>(elem_count * sizeof(int32_t)), 0, 0, 0);
    AscendC::DataCopyPad(gm[gm_offset], ub, copy_params);
}

__aicore__ inline void FillContiguousGm(
    AscendC::GlobalTensor<int32_t>& gm,
    AscendC::LocalTensor<int32_t>& fill_ub,
    int32_t gm_offset,
    int32_t elem_count,
    int32_t value,
    int32_t core_idx,
    int32_t core_num) {
    (void)fill_ub;
    if (elem_count <= 0) {
        return;
    }

    constexpr int32_t kElemsPerDataBlock = 32 / kInt32Bytes;
    const int32_t block_count =
        (elem_count + kElemsPerDataBlock - 1) / kElemsPerDataBlock;
    const int32_t blocks_per_core = block_count / core_num;
    const int32_t tail_cores = block_count - blocks_per_core * core_num;
    const int32_t block_begin =
        core_idx * blocks_per_core + IntMin(core_idx, tail_cores);
    const int32_t assigned_blocks =
        blocks_per_core + (core_idx < tail_cores ? 1 : 0);
    if (assigned_blocks <= 0) {
        return;
    }
    const int32_t begin = block_begin * kElemsPerDataBlock;
    const int32_t end = IntMin(
        elem_count,
        (block_begin + assigned_blocks) * kElemsPerDataBlock);

    for (int32_t base = begin; base < end; base += kFillTileElems) {
        int32_t len = IntMin(kFillTileElems, end - base);
        AscendC::Duplicate(fill_ub, value, len);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        StoreContiguousGmThroughUb(gm, fill_ub, gm_offset + base, len);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }
}

template <class ResourceT>
__aicore__ inline void InitPlanningKernelState(
    GM_ADDR topk_idx,
    GM_ADDR symmetric_buffer,
    GM_ADDR dst,
    GM_ADDR cu_seqlens,
    GM_ADDR experts_to_copy,
    GM_ADDR dup_groups,
    GM_ADDR dup_loffs,
    GM_ADDR dup_counts,
    GM_ADDR tpe_all,
    GM_ADDR tpe_prefix,
    GM_ADDR balance,
    GM_ADDR z_tensor,
    GM_ADDR alloc_tensor,
    GM_ADDR alloc_cumsum,
    GM_ADDR expert_offsets,
    GM_ADDR core_expert_counts,
    GM_ADDR dup_workspace,
    GM_ADDR dup_key_workspace,
    const ascend_moonep::PlanningTilingData& tiling,
    uint64_t ffts_addr,
    ResourceT& resource,
    AscendC::LocalTensor<int32_t>& work_ub,
    PlanningKernelState& state) {
    util_set_ffts_config(ffts_addr);

    state.core_idx =
        get_block_idx() * get_subblockdim() + get_subblockid();
    state.core_num = get_block_num() * get_subblockdim();
    state.rank = static_cast<int32_t>(tiling.rank);
    state.rank_count = static_cast<int32_t>(tiling.world_size);
    state.token_count = static_cast<int32_t>(tiling.num_tokens);
    state.topk = static_cast<int32_t>(tiling.num_topk);
    state.expert_count = static_cast<int32_t>(tiling.num_experts);
    state.max_tokens_per_rank =
        static_cast<int32_t>(tiling.num_max_tokens_per_rank);
    state.prefetch_count = static_cast<int32_t>(tiling.num_prefetch_slots);
    state.experts_per_rank = state.expert_count / state.rank_count;
    state.route_items = state.token_count * state.topk;
    state.tpe_offset = static_cast<int32_t>(tiling.symmetric_tpe_offset);
    state.count_inbox_offset = static_cast<int32_t>(
        tiling.symmetric_dup_rank_counts_offset);
    state.prefix_inbox_offset =
        state.count_inbox_offset + state.rank_count * kSourceDupInboxSlotElems;
    state.payload_done_flag_offset =
        state.prefix_inbox_offset + state.rank_count * kSourceDupInboxSlotElems;
    state.groups_scratch_offset =
        state.payload_done_flag_offset + ((state.rank_count + 7) / 8) * 8;
    state.loffs_scratch_offset =
        state.groups_scratch_offset +
        ((state.max_tokens_per_rank * 3 + 7) / 8) * 8;
    state.source_dup_tokens_per_tile =
        static_cast<int32_t>(tiling.source_dup_tokens_per_tile);
    state.owner_reduce_tile_elems =
        static_cast<int32_t>(tiling.owner_reduce_tile_elems);
    state.route_core_num = IntMin(state.core_num, kMaxPlanningCores);
    state.items_per_core =
        (state.route_items + state.route_core_num - 1) / state.route_core_num;
    state.begin = state.core_idx * state.items_per_core;
    state.end = IntMin(
        state.begin + state.items_per_core,
        state.route_items);

    state.topk_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(topk_idx));
    state.dst_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(dst));
    state.cu_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(cu_seqlens));
    state.experts_to_copy_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(experts_to_copy));
    state.dup_groups_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(dup_groups));
    state.dup_loffs_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(dup_loffs));
    state.dup_counts_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(dup_counts));
    state.tpe_all_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(tpe_all));
    state.tpe_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(tpe_all) +
        state.rank * state.expert_count);
    state.tpe_prefix_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(tpe_prefix));
    state.balance_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(balance));
    state.z_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(z_tensor));
    state.alloc_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(alloc_tensor));
    state.alloc_cumsum_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(alloc_cumsum));
    state.expert_offsets_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(expert_offsets));
    state.core_expert_counts_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(core_expert_counts));
    state.cursor_worker_scratch_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(core_expert_counts) +
        kMaxPlanningCores * state.expert_count);
    state.dup_workspace_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(dup_workspace));
    state.dup_key_workspace_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(dup_key_workspace));

    state.shmem_base = reinterpret_cast<__gm__ int32_t*>(symmetric_buffer);
    state.local_shmem_gm.SetGlobalBuffer(state.shmem_base);

    FillContiguousGm(
        state.local_shmem_gm,
        work_ub,
        state.count_inbox_offset,
        state.rank_count * kSourceDupInboxSlotElems * 2,
        -1,
        state.core_idx,
        state.core_num);
    FillContiguousGm(
        state.local_shmem_gm,
        work_ub,
        state.payload_done_flag_offset,
        state.rank_count,
        0,
        state.core_idx,
        state.core_num);
}

__aicore__ inline bool ExpertSelectedInUb(
    AscendC::LocalTensor<int32_t>& selected_ub,
    int32_t expert,
    int32_t B) {
    for (int32_t b = 0; b < B; ++b) {
        if (selected_ub.GetValue(b) == expert) {
            return true;
        }
    }
    return false;
}

template <class ResourceT>
__aicore__ inline void PublishSourceDupPrefixes(
    const SourceDupConfig& config,
    ResourceT& resource,
    __gm__ int32_t* shmem_base,
    AscendC::GlobalTensor<int32_t>& dup_key_workspace_gm,
    int32_t prefix_inbox_off) {
    int32_t source_begin = 0;
    int32_t source_count = 0;
    GetSourceDupRankPartition(config, source_begin, source_count);
    if (source_count <= 0) {
        return;
    }

    const int32_t slot_elems = source_count * kSourceDupInboxSlotElems;
    AscendC::LocalTensor<int32_t> prefix_slots_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(0);
    AscendC::DataCopyExtParams load_params(
        1,
        static_cast<uint32_t>(slot_elems * kInt32Bytes),
        0,
        0,
        0);
    AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
    AscendC::DataCopyPad(
        prefix_slots_ub,
        dup_key_workspace_gm[
            source_begin * kSourceDupInboxSlotElems],
        load_params,
        pad_params);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

    AscendC::DataCopyExtParams slot_copy_params(
        1,
        static_cast<uint32_t>(kSourceDupInboxSlotElems * kInt32Bytes),
        0,
        0,
        0);
    for (int32_t local_source = 0;
         local_source < source_count;
         ++local_source) {
        const int32_t source = source_begin + local_source;
        AscendC::GlobalTensor<int32_t> remote_shmem_gm;
        remote_shmem_gm.SetGlobalBuffer(
            static_cast<__gm__ int32_t*>(aclshmem_ptr(shmem_base, source)));
        AscendC::DataCopyPad(
            remote_shmem_gm[
                prefix_inbox_off +
                config.rank * kSourceDupInboxSlotElems],
            prefix_slots_ub[local_source * kSourceDupInboxSlotElems],
            slot_copy_params);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

template <class ResourceT>
__aicore__ inline void MaterializeSourceDupOutputs(
    const SourceDupConfig& config,
    const SourceDupUbLayout& ub_layout,
    ResourceT& resource,
    AscendC::GlobalTensor<int32_t>& local_shmem_gm,
    AscendC::GlobalTensor<int32_t>& dup_counts_gm,
    AscendC::GlobalTensor<int32_t>& dup_groups_gm,
    AscendC::GlobalTensor<int32_t>& dup_loffs_gm,
    int32_t dup_groups_scratch_off,
    int32_t dup_loffs_scratch_off) {
    AscendC::LocalTensor<int32_t> count_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(0);
    AscendC::LocalTensor<int32_t> transfer_pingpong_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.count_bytes));
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
    AscendC::DataCopyExtParams count_copy_params(
        1,
        static_cast<uint32_t>(2 * kInt32Bytes),
        0,
        0,
        0);
    AscendC::DataCopyPadExtParams<int32_t> count_pad_params{false, 0, 0, 0};
    AscendC::DataCopyPad(
        count_ub,
        dup_counts_gm[0],
        count_copy_params,
        count_pad_params);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

    const int32_t total_group_elems = count_ub.GetValue(0) * 3;
    const int32_t group_elems_per_core = total_group_elems / config.core_num;
    const int32_t group_tail_cores =
        total_group_elems - group_elems_per_core * config.core_num;
    const int32_t group_begin =
        config.core_idx * group_elems_per_core +
        IntMin(config.core_idx, group_tail_cores);
    const int32_t group_assigned =
        group_elems_per_core + (config.core_idx < group_tail_cores ? 1 : 0);
    const int32_t group_end = group_begin + group_assigned;
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    int32_t group_tile_idx = 0;
    for (int32_t base = group_begin;
         base < group_end;
         base += kDupSrcTileElems, ++group_tile_idx) {
        const int32_t len = IntMin(kDupSrcTileElems, group_end - base);
        const int32_t buffer_idx = group_tile_idx & 1;
        const auto event_id = buffer_idx == 0 ? EVENT_ID0 : EVENT_ID1;
        AscendC::LocalTensor<int32_t> transfer_ub =
            transfer_pingpong_ub[buffer_idx * kDupSrcTileElems];
        AscendC::DataCopyExtParams copy_params(
            1,
            static_cast<uint32_t>(len * kInt32Bytes),
            0,
            0,
            0);
        AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
        AscendC::DataCopyPad(
            transfer_ub,
            local_shmem_gm[dup_groups_scratch_off + base],
            copy_params,
            pad_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
        AscendC::DataCopyPad(dup_groups_gm[base], transfer_ub, copy_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

    const int32_t total_loffs = count_ub.GetValue(1);
    const int32_t loffs_per_core = total_loffs / config.core_num;
    const int32_t loff_tail_cores = total_loffs - loffs_per_core * config.core_num;
    const int32_t loff_begin =
        config.core_idx * loffs_per_core + IntMin(config.core_idx, loff_tail_cores);
    const int32_t loff_assigned =
        loffs_per_core + (config.core_idx < loff_tail_cores ? 1 : 0);
    const int32_t loff_end = loff_begin + loff_assigned;
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    int32_t loff_tile_idx = 0;
    for (int32_t base = loff_begin;
         base < loff_end;
         base += kDupSrcTileElems, ++loff_tile_idx) {
        const int32_t len = IntMin(kDupSrcTileElems, loff_end - base);
        const int32_t buffer_idx = loff_tile_idx & 1;
        const auto event_id = buffer_idx == 0 ? EVENT_ID0 : EVENT_ID1;
        AscendC::LocalTensor<int32_t> transfer_ub =
            transfer_pingpong_ub[buffer_idx * kDupSrcTileElems];
        AscendC::DataCopyExtParams copy_params(
            1,
            static_cast<uint32_t>(len * kInt32Bytes),
            0,
            0,
            0);
        AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
        AscendC::DataCopyPad(
            transfer_ub,
            local_shmem_gm[dup_loffs_scratch_off + base],
            copy_params,
            pad_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
        AscendC::DataCopyPad(dup_loffs_gm[base], transfer_ub, copy_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <class ResourceT>
__aicore__ inline void CountAndPrebuildSourceDupTiles(
    const SourceDupConfig& config,
    const SourceDupPartition& partition,
    const SourceDupUbLayout& ub_layout,
    const SourceDupWorkspaceLayout& workspace_layout,
    ResourceT& resource,
    AscendC::GlobalTensor<int32_t>& dst_gm,
    AscendC::GlobalTensor<int32_t>& dup_workspace_gm,
    AscendC::GlobalTensor<int32_t>& dup_key_workspace_gm) {
    AscendC::LocalTensor<int32_t> count_ub = resource.ubBuf.template GetBufferByByte<int32_t>(0);
    AscendC::Duplicate(
        count_ub,
        static_cast<int32_t>(0),
        config.rank_count * 2);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

    int32_t tile_idx = 0;
    for (int32_t token_base = partition.token_begin;
         token_base < partition.token_end;
         token_base += config.tokens_per_tile, ++tile_idx) {
        ProcessSourceDupTile(
            config,
            partition,
            ub_layout,
            workspace_layout,
            resource,
            dst_gm,
            dup_workspace_gm,
            dup_key_workspace_gm,
            token_base,
            tile_idx);
    }

    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    StoreContiguousGmThroughUb(
        dup_workspace_gm,
        count_ub,
        config.core_idx * config.rank_count * 2,
        config.rank_count * 2);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

} // namespace planning
} // namespace ascend_moonep
