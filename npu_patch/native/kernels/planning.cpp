#include "acl/acl.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "kernel_operator.h"
#include "shmem.h"
#include "simt_api/common_functions.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"

#include "ascend_moonep/kernels/launch.h"
#include "ascend_moonep/kernels/types.h"

using namespace AscendC;

namespace {

constexpr int32_t kFillTileElems = 1024; // must be 32B aligned
constexpr int32_t kInt32Bytes = 4;
constexpr int32_t kMaxPlanningCores = 56;
constexpr int32_t kMaxPlanningRanks = 1024;
constexpr int32_t kAscend950MixedUbSizeBytes = 216 * 1024;
constexpr int32_t kPlanningUbSizeBytes = 216 * 1024;
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
static_assert(kPlanningUbSizeBytes <= kAscend950MixedUbSizeBytes);
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

struct PlanningAscend950 : Catlass::Arch::Ascend950 {
    static constexpr uint32_t UB_SIZE = kPlanningUbSizeBytes;
};

using PlanningResource = Catlass::Arch::Resource<PlanningAscend950>;

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

__simd_vf__ inline void ReduceTpeRowsReg(
    __ubuf__ int32_t* dst,
    __ubuf__ int32_t* src,
    uint32_t row_count,
    uint32_t column_count,
    uint32_t row_stride_elems) {
    AscendC::Reg::RegTensor<int32_t> accumulator;
    AscendC::Reg::RegTensor<int32_t> row_value;
    AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<int32_t>(column_count);

    AscendC::Reg::LoadAlign(accumulator, src);
    for (uint32_t row = 1; row < row_count; ++row) {
        AscendC::Reg::LoadAlign(row_value, src + row * row_stride_elems);
        AscendC::Reg::Add(accumulator, accumulator, row_value, mask);
    }
    AscendC::Reg::StoreAlign(dst, accumulator, mask);
}

__simd_vf__ inline void ReduceTpeOwnerRowsReg(
    __ubuf__ int32_t* dst,
    __ubuf__ int32_t* total_dst,
    __ubuf__ int32_t* src,
    uint32_t row_count,
    uint32_t column_count,
    uint32_t row_stride_elems) {
    AscendC::Reg::RegTensor<int32_t> accumulator;
    AscendC::Reg::RegTensor<int32_t> row_value;
    AscendC::Reg::RegTensor<int32_t> total_value;
    AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<int32_t>(column_count);

    AscendC::Reg::LoadAlign(accumulator, src);
    for (uint32_t row = 1; row < row_count; ++row) {
        AscendC::Reg::LoadAlign(row_value, src + row * row_stride_elems);
        AscendC::Reg::Add(accumulator, accumulator, row_value, mask);
    }
    AscendC::Reg::StoreAlign(dst, accumulator, mask);
    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(
        total_value, accumulator, mask);
    uint32_t total_count = 1;
    AscendC::Reg::MaskReg total_mask = AscendC::Reg::UpdateMask<int32_t>(total_count);
    AscendC::Reg::StoreAlign(total_dst, total_value, total_mask);
}

__simt_vf__ __launch_bounds__(kSourceDupSimtMaxThreads) inline void BuildRouteHistogramSimt(
    __ubuf__ int32_t* routes,
    __ubuf__ int32_t* expert_counts,
    int32_t route_count) {
    for (int32_t route = static_cast<int32_t>(threadIdx.x);
         route < route_count;
         route += kSourceDupSimtMaxThreads) {
        const int32_t expert = routes[route];
        asc_atomic_add(expert_counts + expert, 1);
    }
}

__simt_vf__ __launch_bounds__(kSourceDupSimtMaxThreads) inline void BuildAllocCumsumSimt(
    __ubuf__ int32_t* alloc,
    __gm__ int32_t* alloc_cumsum,
    int32_t rank_count,
    int32_t experts_per_rank,
    int32_t alloc_row_elems,
    int32_t expert_base,
    int32_t thread_count) {
    const int32_t thread = static_cast<int32_t>(threadIdx.x);

    for (int32_t local_expert = thread;
         local_expert < experts_per_rank;
         local_expert += thread_count) {
        int32_t prefix = 0;
        for (int32_t dest = 0; dest < rank_count; ++dest) {
            const int32_t offset = dest * alloc_row_elems + local_expert;
            prefix += alloc[offset];
            alloc[offset] = prefix;
        }
    }
    asc_syncthreads();

    const int32_t total_elems = rank_count * experts_per_rank;
    for (int32_t index = thread; index < total_elems; index += thread_count) {
        const int32_t local_expert = index / rank_count;
        const int32_t dest = index - local_expert * rank_count;
        alloc_cumsum[(expert_base + local_expert) * rank_count + dest] =
            alloc[dest * alloc_row_elems + local_expert];
    }
}

__simt_vf__ __launch_bounds__(kPrefetchSimtMaxThreads) inline void SelectPrefetchExpertsSimt(
    __ubuf__ int32_t* group_counts,
    __ubuf__ int32_t* selected_experts,
    __ubuf__ int32_t* candidate_counts,
    __ubuf__ int32_t* candidate_experts,
    int32_t expert_count,
    int32_t prefetch_count,
    int32_t local_expert_begin,
    int32_t local_expert_end,
    int32_t thread_count) {
    const int32_t thread = static_cast<int32_t>(threadIdx.x);

    for (int32_t slot = thread; slot < prefetch_count; slot += thread_count) {
        selected_experts[slot] = -1;
        group_counts[expert_count + slot] = 0;
    }
    asc_syncthreads();

    for (int32_t slot = 0; slot < prefetch_count; ++slot) {
        int32_t best_count = 0;
        int32_t best_expert = -1;
        for (int32_t expert = thread;
             expert < expert_count;
             expert += thread_count) {
            if (expert >= local_expert_begin && expert < local_expert_end) {
                continue;
            }
            const int32_t count = group_counts[expert];
            if (count > best_count ||
                (count == best_count && count > 0 && expert > best_expert)) {
                best_count = count;
                best_expert = expert;
            }
        }
        candidate_counts[thread] = best_count;
        candidate_experts[thread] = best_expert;
        asc_syncthreads();

        for (int32_t stride = thread_count >> 1; stride > 0; stride >>= 1) {
            if (thread < stride) {
                const int32_t other_count = candidate_counts[thread + stride];
                const int32_t other_expert = candidate_experts[thread + stride];
                const int32_t current_count = candidate_counts[thread];
                const int32_t current_expert = candidate_experts[thread];
                if (other_count > current_count ||
                    (other_count == current_count && other_count > 0 &&
                     other_expert > current_expert)) {
                    candidate_counts[thread] = other_count;
                    candidate_experts[thread] = other_expert;
                }
            }
            asc_syncthreads();
        }

        if (thread == 0) {
            const int32_t selected_count = candidate_counts[0];
            const int32_t selected_expert = candidate_experts[0];
            if (selected_count > 0 && selected_expert >= 0) {
                selected_experts[slot] = selected_expert;
                group_counts[expert_count + slot] = selected_count;
                group_counts[selected_expert] = 0;
            }
        }
        asc_syncthreads();
        if (selected_experts[slot] < 0) {
            break;
        }
    }
}

__simt_vf__ __launch_bounds__(kPrefetchSimtMaxThreads) inline void BuildGroupPrefixSimt(
    __ubuf__ int32_t* group_counts,
    __ubuf__ int32_t* thread_prefix,
    int32_t group_count,
    int32_t thread_count) {
    const int32_t thread = static_cast<int32_t>(threadIdx.x);
    const int32_t elems_per_thread =
        (group_count + thread_count - 1) / thread_count;
    const int32_t begin = thread * elems_per_thread;
    int32_t end = begin + elems_per_thread;
    if (end > group_count) {
        end = group_count;
    }

    int32_t local_prefix = 0;
    for (int32_t group = begin; group < end; ++group) {
        local_prefix += group_counts[group];
        group_counts[group] = local_prefix;
    }
    thread_prefix[thread] = local_prefix;
    asc_syncthreads();

    if (thread == 0) {
        int32_t block_prefix = 0;
        for (int32_t source_thread = 0;
             source_thread < thread_count;
             ++source_thread) {
            const int32_t thread_total = thread_prefix[source_thread];
            thread_prefix[source_thread] = block_prefix;
            block_prefix += thread_total;
        }
    }
    asc_syncthreads();

    const int32_t block_prefix = thread_prefix[thread];
    for (int32_t group = begin; group < end; ++group) {
        group_counts[group] += block_prefix;
    }
    asc_syncthreads();
}

__simt_vf__ __launch_bounds__(kCoreCursorSimtMaxThreads) inline void ReduceExpertTileTotalSimt(
    __ubuf__ int32_t* expert_counts,
    __ubuf__ int32_t* thread_totals,
    __ubuf__ int32_t* tile_total,
    int32_t expert_count,
    int32_t thread_count) {
    const int32_t thread = static_cast<int32_t>(threadIdx.x);
    int32_t local_total = 0;
    for (int32_t expert = thread; expert < expert_count;
         expert += thread_count) {
        local_total += expert_counts[expert];
    }
    thread_totals[thread] = local_total;
    asc_syncthreads();

    if (thread == 0) {
        int32_t total = 0;
        for (int32_t source_thread = 0; source_thread < thread_count;
             ++source_thread) {
            total += thread_totals[source_thread];
        }
        tile_total[0] = total;
    }
}

__simt_vf__ __launch_bounds__(kCoreCursorSimtMaxThreads) inline void BuildCoreExpertCursorsSimt(
    __ubuf__ int32_t* core_counts,
    __ubuf__ int32_t* tpe_counts,
    __ubuf__ int32_t* thread_prefix,
    __ubuf__ int32_t* expert_carry,
    int32_t core_count,
    int32_t expert_count,
    int32_t row_stride,
    int32_t thread_count) {
    const int32_t thread = static_cast<int32_t>(threadIdx.x);
    const int32_t experts_per_thread =
        (expert_count + thread_count - 1) / thread_count;
    const int32_t begin = thread * experts_per_thread;
    int32_t end = begin + experts_per_thread;
    if (end > expert_count) {
        end = expert_count;
    }

    int32_t local_total = 0;
    for (int32_t expert = begin; expert < end; ++expert) {
        local_total += tpe_counts[expert];
        tpe_counts[expert] = local_total;
    }
    thread_prefix[thread] = local_total;
    asc_syncthreads();

    if (thread == 0) {
        int32_t prefix = expert_carry[0];
        for (int32_t source_thread = 0;
             source_thread < thread_count;
             ++source_thread) {
            const int32_t thread_total = thread_prefix[source_thread];
            thread_prefix[source_thread] = prefix;
            prefix += thread_total;
        }
        expert_carry[0] = prefix;
    }
    asc_syncthreads();

    const int32_t thread_base = thread_prefix[thread];
    for (int32_t expert = begin; expert < end; ++expert) {
        const int32_t expert_base =
            thread_base + (expert == begin ? 0 : tpe_counts[expert - 1]);
        int32_t cursor = expert_base;
        for (int32_t source_core = 0;
             source_core < core_count;
             ++source_core) {
            const int32_t offset = source_core * row_stride + expert;
            const int32_t count = core_counts[offset];
            core_counts[offset] = cursor;
            cursor += count;
        }
    }
    asc_syncthreads();
}

__simt_vf__ __launch_bounds__(kPrefetchSimtMaxThreads) inline void
ConvertGroupPrefixToExpertOffsetsSimt(
    __ubuf__ int32_t* group_prefix,
    __ubuf__ int32_t* selected_experts,
    __ubuf__ int32_t* selected_offsets,
    int32_t expert_count,
    int32_t prefetch_count,
    int32_t thread_count) {
    const int32_t thread = static_cast<int32_t>(threadIdx.x);

    for (int32_t slot = thread; slot < prefetch_count; slot += thread_count) {
        const int32_t group = expert_count + slot;
        const int32_t begin = group == 0 ? 0 : group_prefix[group - 1];
        const int32_t end = group_prefix[group];
        selected_offsets[slot] = end > begin ? begin : -1;
    }

    const int32_t experts_per_thread =
        (expert_count + thread_count - 1) / thread_count;
    const int32_t begin = thread * experts_per_thread;
    int32_t end = begin + experts_per_thread;
    if (end > expert_count) {
        end = expert_count;
    }
    const int32_t preceding_prefix =
        begin > 0 && begin < expert_count ? group_prefix[begin - 1] : 0;
    asc_syncthreads();

    for (int32_t expert = end - 1; expert >= begin; --expert) {
        const int32_t group_begin =
            expert == begin ? preceding_prefix : group_prefix[expert - 1];
        const int32_t group_end = group_prefix[expert];
        group_prefix[expert] = group_end > group_begin ? group_begin : -1;
    }
    asc_syncthreads();

    for (int32_t slot = thread; slot < prefetch_count; slot += thread_count) {
        const int32_t expert = selected_experts[slot];
        const int32_t offset = selected_offsets[slot];
        if (expert >= 0 && offset >= 0) {
            group_prefix[expert] = offset;
        }
    }
    asc_syncthreads();
}

__simt_vf__ __launch_bounds__(kSourceDupSimtMaxThreads) inline void DedupRouteTileSimt(
    __ubuf__ int32_t* dst,
    __ubuf__ int32_t* token_groups,
    __ubuf__ int32_t* token_loffs,
    __ubuf__ int32_t* token_group_counts,
    __ubuf__ int32_t* token_loff_counts,
    __ubuf__ int32_t* tile_counts,
    int32_t tile_tokens,
    int32_t topk,
    int32_t max_tokens_per_rank) {
    const int32_t token = static_cast<int32_t>(threadIdx.x);
    if (token >= tile_tokens) {
        return;
    }

    const int32_t route_base = token * topk;
    const int32_t group_base = token * (topk / 2) * 4;
    const int32_t loff_base = token * topk;
    int32_t group_count = 0;
    int32_t loff_count = 0;

    for (int32_t kidx = 0; kidx < topk; ++kidx) {
        const int32_t raw = dst[route_base + kidx];
        if (raw < 0) {
            continue;
        }

        const int32_t dest = raw / max_tokens_per_rank;
        const int32_t dest_base = dest * max_tokens_per_rank;
        const int32_t dest_end = dest_base + max_tokens_per_rank;
        const int32_t group_loff_start = loff_count;
        int32_t duplicate_count = 0;
        for (int32_t next_kidx = kidx + 1; next_kidx < topk; ++next_kidx) {
            const int32_t next_raw = dst[route_base + next_kidx];
            if (next_raw < dest_base || next_raw >= dest_end) {
                continue;
            }
            token_loffs[loff_base + loff_count] = next_raw - dest_base;
            ++loff_count;
            ++duplicate_count;
            dst[route_base + next_kidx] = -next_raw - 1;
        }

        if (duplicate_count > 0) {
            const int32_t group_pos = group_base + group_count * 4;
            token_groups[group_pos] = dest;
            token_groups[group_pos + 1] = raw - dest_base;
            token_groups[group_pos + 2] = loff_base + group_loff_start;
            token_groups[group_pos + 3] = duplicate_count;
            asc_atomic_add(tile_counts + dest * 2, 1);
            asc_atomic_add(tile_counts + dest * 2 + 1, duplicate_count);
            ++group_count;
        }
    }

    token_group_counts[token] = group_count;
    token_loff_counts[token] = loff_count;
}

__simt_vf__ __launch_bounds__(kSourceDupSimtMaxThreads) inline void CompactRouteTileSimt(
    __ubuf__ int32_t* token_groups,
    __ubuf__ int32_t* token_loffs,
    __ubuf__ int32_t* token_group_counts,
    __ubuf__ int32_t* token_loff_counts,
    __ubuf__ int32_t* token_group_prefix,
    __ubuf__ int32_t* token_loff_prefix,
    __ubuf__ int32_t* compact_groups,
    __ubuf__ int32_t* compact_loffs,
    int32_t tile_tokens,
    int32_t topk) {
    const int32_t token = static_cast<int32_t>(threadIdx.x);
    if (token >= tile_tokens) {
        return;
    }

    const int32_t group_count = token_group_counts[token];
    const int32_t loff_count = token_loff_counts[token];
    const int32_t source_group_base = token * (topk / 2) * 4;
    const int32_t source_loff_base = token * topk;
    const int32_t compact_group_base = token_group_prefix[token] * 3;
    const int32_t compact_loff_base = token_loff_prefix[token];

    for (int32_t group = 0; group < group_count; ++group) {
        const int32_t source_group_pos = source_group_base + group * 4;
        const int32_t compact_group_pos = compact_group_base + group * 3;
        const int32_t source_loff_start = token_groups[source_group_pos + 2];
        compact_groups[compact_group_pos] = token_groups[source_group_pos];
        compact_groups[compact_group_pos + 1] = token_groups[source_group_pos + 1];
        compact_groups[compact_group_pos + 2] =
            compact_loff_base + source_loff_start - source_loff_base;
    }
    for (int32_t i = 0; i < loff_count; ++i) {
        compact_loffs[compact_loff_base + i] = token_loffs[source_loff_base + i];
    }
}

__simt_vf__ __launch_bounds__(kSourceDupSimtMaxThreads) inline void
PackSourceDupByDestSimt(
    __ubuf__ int32_t* compact_groups,
    __ubuf__ int32_t* compact_loffs,
    __ubuf__ int32_t* dest_meta,
    __ubuf__ int32_t* packed_groups,
    __ubuf__ int32_t* packed_loffs,
    __ubuf__ int32_t* chunk_hist,
    int32_t group_count,
    int32_t loff_count,
    int32_t rank_count,
    int32_t groups_per_chunk,
    int32_t hist_row_elems,
    int32_t thread_count) {
    const int32_t thread = static_cast<int32_t>(threadIdx.x);
    const int32_t chunk_count =
        (group_count + groups_per_chunk - 1) / groups_per_chunk;

    // One thread owns each chunk, so its histogram and the later cursor
    // updates are deterministic without atomics.
    for (int32_t chunk = thread; chunk < chunk_count; chunk += thread_count) {
        __ubuf__ int32_t* hist_row = chunk_hist + chunk * hist_row_elems;
        const int32_t group_begin = chunk * groups_per_chunk;
        const int32_t group_end =
            group_begin + groups_per_chunk < group_count
                ? group_begin + groups_per_chunk
                : group_count;
        for (int32_t group = group_begin; group < group_end; ++group) {
            const int32_t compact_pos = group * 3;
            const int32_t dest = compact_groups[compact_pos];
            const int32_t raw_loff_start = compact_groups[compact_pos + 2];
            const int32_t next_raw_loff_start =
                group + 1 < group_count
                    ? compact_groups[compact_pos + 5]
                    : loff_count;
            const int32_t hist_pos = dest * 2;
            hist_row[hist_pos] = hist_row[hist_pos] + 1;
            hist_row[hist_pos + 1] = hist_row[hist_pos + 1] +
                next_raw_loff_start - raw_loff_start;
        }
    }
    asc_syncthreads();

    // Convert every destination's chunk histogram to exclusive chunk bases.
    for (int32_t dest = thread; dest < rank_count; dest += thread_count) {
        int32_t group_prefix = 0;
        int32_t loff_prefix = 0;
        const int32_t hist_pos = dest * 2;
        for (int32_t chunk = 0; chunk < chunk_count; ++chunk) {
            __ubuf__ int32_t* hist_row = chunk_hist + chunk * hist_row_elems;
            const int32_t chunk_groups = hist_row[hist_pos];
            const int32_t chunk_loffs = hist_row[hist_pos + 1];
            hist_row[hist_pos] = group_prefix;
            hist_row[hist_pos + 1] = loff_prefix;
            group_prefix += chunk_groups;
            loff_prefix += chunk_loffs;
        }
    }
    asc_syncthreads();

    // Re-scan each chunk in source order. The exclusive bases above place
    // chunks in order, while this serial per-chunk scan preserves group order.
    for (int32_t chunk = thread; chunk < chunk_count; chunk += thread_count) {
        __ubuf__ int32_t* cursor_row = chunk_hist + chunk * hist_row_elems;
        const int32_t group_begin = chunk * groups_per_chunk;
        const int32_t group_end =
            group_begin + groups_per_chunk < group_count
                ? group_begin + groups_per_chunk
                : group_count;
        for (int32_t group = group_begin; group < group_end; ++group) {
            const int32_t compact_pos = group * 3;
            const int32_t dest = compact_groups[compact_pos];
            const int32_t primary_loff = compact_groups[compact_pos + 1];
            const int32_t raw_loff_start = compact_groups[compact_pos + 2];
            const int32_t next_raw_loff_start =
                group + 1 < group_count
                    ? compact_groups[compact_pos + 5]
                    : loff_count;
            const int32_t duplicate_count =
                next_raw_loff_start - raw_loff_start;
            const int32_t hist_pos = dest * 2;
            const int32_t group_ordinal = cursor_row[hist_pos];
            cursor_row[hist_pos] = group_ordinal + 1;
            const int32_t loff_ordinal = cursor_row[hist_pos + 1];
            cursor_row[hist_pos + 1] = loff_ordinal + duplicate_count;

            const int32_t meta_pos = dest * 5;
            const int32_t group_pos =
                dest_meta[meta_pos] + group_ordinal * 3;
            const int32_t loff_base = dest_meta[meta_pos + 3];
            const int32_t loff_pos = loff_base + loff_ordinal;
            packed_groups[group_pos] = primary_loff;
            packed_groups[group_pos + 1] = loff_ordinal;
            packed_groups[group_pos + 2] = duplicate_count;
            for (int32_t i = 0; i < duplicate_count; ++i) {
                packed_loffs[loff_pos + i] =
                    compact_loffs[raw_loff_start + i];
            }
        }
    }
}

__simt_vf__ __launch_bounds__(kSourceDupSimtMaxThreads) inline void
AdjustSourceDupGroupLoffsSimt(
    __ubuf__ int32_t* groups,
    __ubuf__ int32_t* tile_counts,
    __ubuf__ int32_t* dest_meta,
    int32_t rank_count,
    int32_t threads_per_dest) {
    const int32_t thread = static_cast<int32_t>(threadIdx.x);
    const int32_t logical_thread_count = rank_count * threads_per_dest;
    for (int32_t logical_thread = thread;
         logical_thread < logical_thread_count;
         logical_thread += kSourceDupSimtMaxThreads) {
        const int32_t dest = logical_thread / threads_per_dest;
        const int32_t dest_lane =
            logical_thread - dest * threads_per_dest;
        const int32_t group_count = tile_counts[dest * 2];
        const int32_t meta_pos = dest * 5;
        const int32_t group_start = dest_meta[meta_pos];
        const int32_t global_loff_base = dest_meta[meta_pos + 1];
        for (int32_t group = dest_lane;
             group < group_count;
             group += threads_per_dest) {
            const int32_t loff_field = group_start + group * 3 + 1;
            groups[loff_field] = global_loff_base + groups[loff_field];
        }
    }
}

__simt_vf__ __launch_bounds__(kSourceDupSimtMaxThreads) inline void
PackSourceDupCountSlotsSimt(
    __ubuf__ int32_t* core_counts,
    __ubuf__ int32_t* count_slots,
    int32_t core_count,
    int32_t dest_count,
    int32_t core_row_elems) {
    const int32_t slot_element = static_cast<int32_t>(threadIdx.x);
    const int32_t slot_elems = dest_count * kSourceDupInboxSlotElems;
    if (slot_element >= slot_elems) {
        return;
    }

    const int32_t slot_lane = slot_element % kSourceDupInboxSlotElems;
    if (slot_lane >= 2) {
        count_slots[slot_element] = 0;
        return;
    }

    const int32_t local_dest = slot_element / kSourceDupInboxSlotElems;
    const int32_t count_element = local_dest * 2 + slot_lane;
    int32_t total = 0;
    for (int32_t source_core = 0; source_core < core_count; ++source_core) {
        total += core_counts[source_core * core_row_elems + count_element];
    }
    count_slots[slot_element] = total;
}

__simt_vf__ __launch_bounds__(32) inline void
BuildSourceDupDestinationPrefixesSimt(
    __ubuf__ int32_t* count_slots,
    __ubuf__ int32_t* totals,
    int32_t rank_count) {
    const int32_t count_lane = static_cast<int32_t>(threadIdx.x);
    if (count_lane >= 2) {
        return;
    }

    int32_t prefix = 0;
    for (int32_t source_rank = 0; source_rank < rank_count; ++source_rank) {
        const int32_t offset =
            source_rank * kSourceDupInboxSlotElems + count_lane;
        const int32_t count = count_slots[offset];
        count_slots[offset] = prefix;
        prefix += count;
    }
    totals[count_lane] = prefix;
}

__simt_vf__ __launch_bounds__(kSourceDupSimtMaxThreads) inline void
BuildSourceDupCorePrefixesSimt(
    __ubuf__ int32_t* core_counts,
    __ubuf__ int32_t* rank_prefix_slots,
    int32_t core_count,
    int32_t dest_count,
    int32_t core_row_elems) {
    const int32_t count_element = static_cast<int32_t>(threadIdx.x);
    const int32_t count_elems = dest_count * 2;
    if (count_element >= count_elems) {
        return;
    }

    const int32_t local_dest = count_element / 2;
    const int32_t count_lane = count_element - local_dest * 2;
    int32_t prefix = rank_prefix_slots[
        local_dest * kSourceDupInboxSlotElems + count_lane];
    for (int32_t source_core = 0; source_core < core_count; ++source_core) {
        const int32_t offset =
            source_core * core_row_elems + count_element;
        const int32_t count = core_counts[offset];
        core_counts[offset] = prefix;
        prefix += count;
    }
}

__simt_vf__ __launch_bounds__(kDstSimtMaxThreads) inline void BuildDstTileSimt(
    __ubuf__ int32_t* workspace,
    int32_t context_offset,
    int32_t tile_base,
    int32_t tile_len) {
    (void)tile_base;
    const int32_t thread = static_cast<int32_t>(threadIdx.x);
    __ubuf__ int32_t* context = workspace + context_offset;
    __ubuf__ int32_t* topk_tile = workspace + context[kDstSimtTopkOffset];
    __ubuf__ int32_t* chunk_prefix = workspace + context[kDstSimtHistOffset];
    __ubuf__ int32_t* ordinal_base = workspace + context[kDstSimtOrdinalOffset];
    __ubuf__ int32_t* alloc_cumsum = workspace + context[kDstSimtAllocCumsumOffset];
    __ubuf__ int32_t* expert_offsets = workspace + context[kDstSimtExpertOffsetsOffset];
    __ubuf__ int32_t* dst_tile = workspace + context[kDstSimtDstOffset];
    const int32_t chunk_size = context[kDstSimtChunkSize];
    const int32_t hist_stride = context[kDstSimtHistStride];
    const int32_t rank_count = context[kDstSimtRankCount];
    const int32_t expert_count = context[kDstSimtExpertCount];
    const int32_t max_tokens_per_rank = context[kDstSimtMaxTokensPerRank];
    const int32_t chunk_count = (tile_len + chunk_size - 1) / chunk_size;

    if (thread < chunk_count) {
        const int32_t route_begin = thread * chunk_size;
        const int32_t route_end =
            route_begin + chunk_size < tile_len ? route_begin + chunk_size : tile_len;
        __ubuf__ int32_t* hist_row = chunk_prefix + thread * hist_stride;
        for (int32_t route = route_begin; route < route_end; ++route) {
            const int32_t expert = topk_tile[route];
            // Reuse the output tile for the stable ordinal inside this chunk.
            const int32_t rank_in_chunk = hist_row[expert];
            hist_row[expert] = rank_in_chunk + 1;
            dst_tile[route] = rank_in_chunk;
        }
    }
    asc_syncthreads();

    for (int32_t expert = thread; expert < expert_count;
         expert += kDstSimtMaxThreads) {
        int32_t prefix = 0;
        for (int32_t chunk = 0; chunk < chunk_count; ++chunk) {
            const int32_t offset = chunk * hist_stride + expert;
            prefix += chunk_prefix[offset];
            chunk_prefix[offset] = prefix;
        }
    }
    asc_syncthreads();

    if (thread < tile_len) {
        const int32_t route = thread;
        const int32_t expert = topk_tile[route];
        const int32_t chunk = route / chunk_size;
        const int32_t rank_in_chunk = dst_tile[route];

        const int32_t prior_chunk_count =
            chunk == 0 ? 0 : chunk_prefix[(chunk - 1) * hist_stride + expert];
        const int32_t global_ordinal =
            ordinal_base[expert] + prior_chunk_count + rank_in_chunk;

        int32_t dst_rank = rank_count - 1;
        int32_t previous_boundary = 0;
        for (int32_t dst = 0; dst < rank_count; ++dst) {
            const int32_t current_boundary = alloc_cumsum[expert * rank_count + dst];
            if (global_ordinal < current_boundary) {
                dst_rank = dst;
                break;
            }
            previous_boundary = current_boundary;
        }
        const int32_t dst_base = expert_offsets[dst_rank * expert_count + expert];
        dst_tile[route] = dst_rank * max_tokens_per_rank + dst_base +
            (global_ordinal - previous_boundary);
    }
    asc_syncthreads();
}

__simt_vf__ __launch_bounds__(kDstSimtMaxThreads) inline void BuildDstTileFromGmSimt(
    __ubuf__ int32_t* topk_tile,
    __ubuf__ int32_t* expert_ordinal,
    __ubuf__ int32_t* dst_tile,
    __gm__ int32_t* alloc_cumsum,
    __gm__ int32_t* expert_offsets,
    int32_t tile_len,
    int32_t rank_count,
    int32_t expert_count,
    int32_t max_tokens_per_rank) {
    const int32_t thread = static_cast<int32_t>(threadIdx.x);

    // A single thread assigns stable ordinals in input-route order. The
    // expensive rank lookup below remains parallel across the whole tile.
    if (thread == 0) {
        for (int32_t route = 0; route < tile_len; ++route) {
            const int32_t expert = topk_tile[route];
            const int32_t ordinal = expert_ordinal[expert];
            expert_ordinal[expert] = ordinal + 1;
            dst_tile[route] = ordinal;
        }
    }
    asc_syncthreads();

    for (int32_t route = thread; route < tile_len;
         route += kDstSimtMaxThreads) {
        const int32_t expert = topk_tile[route];
        const int32_t global_ordinal = dst_tile[route];
        const int32_t cumsum_base = expert * rank_count;

        int32_t low = 0;
        int32_t high = rank_count;
        while (low < high) {
            const int32_t mid = low + (high - low) / 2;
            if (global_ordinal < alloc_cumsum[cumsum_base + mid]) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }

        const int32_t dst_rank = low < rank_count ? low : rank_count - 1;
        const int32_t previous_boundary = dst_rank == 0
            ? 0
            : alloc_cumsum[cumsum_base + dst_rank - 1];
        const int32_t dst_base =
            expert_offsets[dst_rank * expert_count + expert];
        dst_tile[route] = dst_rank * max_tokens_per_rank + dst_base +
            (global_ordinal - previous_boundary);
    }
    asc_syncthreads();
}

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

// Partition by 32-byte data blocks so concurrent cores never update different
// parts of the same GM block. Only the final owner may issue a short tail.
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

__aicore__ inline void WaitForSourceDupInbox(
    AscendC::GlobalTensor<int32_t>& local_shmem_gm,
    AscendC::LocalTensor<int32_t>& inbox_ub,
    AscendC::LocalTensor<int32_t>& result_ub,
    int32_t inbox_offset,
    int32_t slot_count) {
    if (slot_count <= 0) {
        return;
    }

    const int32_t inbox_elems = slot_count * kSourceDupInboxSlotElems;
    AscendC::DataCopyExtParams copy_params(
        1,
        static_cast<uint32_t>(inbox_elems * kInt32Bytes),
        0,
        0,
        0);
    AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
    while (true) {
        AscendC::DataCopyPad(
            inbox_ub,
            local_shmem_gm[inbox_offset],
            copy_params,
            pad_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::ReduceMin<int32_t>(
            result_ub,
            inbox_ub,
            inbox_ub,
            inbox_elems,
            false);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        if (result_ub.GetValue(0) >= 0) {
            break;
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
    }
}

__aicore__ inline void PublishSourceDupPayloadDoneFlags(
    const SourceDupConfig& config,
    __gm__ int32_t* shmem_base,
    AscendC::LocalTensor<int32_t>& signal_ub,
    int32_t payload_done_flag_off) {
    signal_ub.SetValue(0, 1);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);

    int32_t dest_begin = 0;
    int32_t dest_count = 0;
    GetSourceDupRankPartition(config, dest_begin, dest_count);
    if (dest_count <= 0) {
        return;
    }

    AscendC::DataCopyExtParams flag_store_params(
        1,
        static_cast<uint32_t>(kInt32Bytes),
        0,
        0,
        0);
    for (int32_t local_dest = 0; local_dest < dest_count; ++local_dest) {
        const int32_t dest = dest_begin + local_dest;
        AscendC::GlobalTensor<int32_t> remote_shmem_gm;
        remote_shmem_gm.SetGlobalBuffer(
            static_cast<__gm__ int32_t*>(aclshmem_ptr(shmem_base, dest)));
        AscendC::DataCopyPad(
            remote_shmem_gm[payload_done_flag_off + config.rank],
            signal_ub,
            flag_store_params);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

__aicore__ inline void WaitForSourceDupPayloadDoneFlags(
    const SourceDupConfig& config,
    PlanningResource& resource,
    AscendC::GlobalTensor<int32_t>& local_shmem_gm,
    int32_t payload_done_flag_off) {
    const int32_t ranks_per_core = config.rank_count / config.core_num;
    const int32_t tail_cores =
        config.rank_count - ranks_per_core * config.core_num;
    const int32_t rank_begin =
        config.core_idx * ranks_per_core + IntMin(config.core_idx, tail_cores);
    const int32_t rank_count =
        ranks_per_core + (config.core_idx < tail_cores ? 1 : 0);
    if (rank_count <= 0) {
        return;
    }

    const int32_t flag_bytes = AlignUpBytes(rank_count * kInt32Bytes, 32);
    AscendC::LocalTensor<int32_t> flag_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(0);
    AscendC::LocalTensor<int32_t> result_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(flag_bytes));
    AscendC::DataCopyExtParams flag_load_params(
        1,
        static_cast<uint32_t>(rank_count * kInt32Bytes),
        0,
        0,
        0);
    AscendC::DataCopyPadExtParams<int32_t> flag_pad_params{false, 0, 0, 0};
    while (true) {
        AscendC::DataCopyPad(
            flag_ub,
            local_shmem_gm[payload_done_flag_off + rank_begin],
            flag_load_params,
            flag_pad_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::ReduceMin<int32_t>(
            result_ub,
            flag_ub,
            flag_ub,
            rank_count,
            false);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        if (result_ub.GetValue(0) >= 1) {
            break;
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
    }
}

__aicore__ inline void PublishSourceDupCounts(
    const SourceDupConfig& config,
    PlanningResource& resource,
    __gm__ int32_t* shmem_base,
    AscendC::GlobalTensor<int32_t>& dup_workspace_gm,
    int32_t count_inbox_off) {
    SourceDupCountTileLayout layout{};
    InitSourceDupCountTileLayout(config, layout);
    AscendC::LocalTensor<int32_t> core_counts_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(layout.core_counts_offset));
    AscendC::LocalTensor<int32_t> count_slots_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(layout.slots_offset));

    int32_t dest_begin = 0;
    int32_t dest_count = 0;
    GetSourceDupRankPartition(config, dest_begin, dest_count);
    if (dest_count <= 0) {
        return;
    }

    LoadSourceDupCoreCountTile(
        config,
        layout,
        dest_begin,
        dest_count,
        core_counts_ub,
        dup_workspace_gm);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    const int32_t slot_elems = dest_count * kSourceDupInboxSlotElems;
    const uint32_t thread_count = static_cast<uint32_t>(
        ((slot_elems + 31) / 32) * 32);
    asc_vf_call<PackSourceDupCountSlotsSimt>(
        dim3(thread_count),
        (__ubuf__ int32_t*)core_counts_ub.GetPhyAddr(),
        (__ubuf__ int32_t*)count_slots_ub.GetPhyAddr(),
        config.core_num,
        dest_count,
        layout.row_elems);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

    AscendC::DataCopyExtParams slot_copy_params(
        1,
        static_cast<uint32_t>(kSourceDupInboxSlotElems * kInt32Bytes),
        0,
        0,
        0);
    for (int32_t local_dest = 0; local_dest < dest_count; ++local_dest) {
        const int32_t dest = dest_begin + local_dest;
        AscendC::GlobalTensor<int32_t> remote_shmem_gm;
        remote_shmem_gm.SetGlobalBuffer(
            static_cast<__gm__ int32_t*>(aclshmem_ptr(shmem_base, dest)));
        AscendC::DataCopyPad(
            remote_shmem_gm[
                count_inbox_off +
                config.rank * kSourceDupInboxSlotElems],
            count_slots_ub[local_dest * kSourceDupInboxSlotElems],
            slot_copy_params);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

__aicore__ inline void BuildSourceDupDestinationPrefixes(
    const SourceDupConfig& config,
    PlanningResource& resource,
    AscendC::GlobalTensor<int32_t>& local_shmem_gm,
    AscendC::GlobalTensor<int32_t>& dup_key_workspace_gm,
    AscendC::GlobalTensor<int32_t>& dup_counts_gm,
    int32_t count_inbox_off) {
    const int32_t inbox_elems =
        config.rank_count * kSourceDupInboxSlotElems;
    const int32_t inbox_bytes = inbox_elems * kInt32Bytes;
    AscendC::LocalTensor<int32_t> count_slots_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(0);
    AscendC::LocalTensor<int32_t> totals_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(inbox_bytes));
    WaitForSourceDupInbox(
        local_shmem_gm,
        count_slots_ub,
        totals_ub,
        count_inbox_off,
        config.rank_count);

    asc_vf_call<BuildSourceDupDestinationPrefixesSimt>(
        dim3(32),
        (__ubuf__ int32_t*)count_slots_ub.GetPhyAddr(),
        (__ubuf__ int32_t*)totals_ub.GetPhyAddr(),
        config.rank_count);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::DataCopyExtParams prefix_copy_params(
        1,
        static_cast<uint32_t>(inbox_bytes),
        0,
        0,
        0);
    AscendC::DataCopyPad(
        dup_key_workspace_gm[0],
        count_slots_ub,
        prefix_copy_params);
    AscendC::DataCopyExtParams total_copy_params(
        1,
        static_cast<uint32_t>(2 * kInt32Bytes),
        0,
        0,
        0);
    AscendC::DataCopyPad(dup_counts_gm[0], totals_ub, total_copy_params);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

__aicore__ inline void PublishSourceDupPrefixes(
    const SourceDupConfig& config,
    PlanningResource& resource,
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

__aicore__ inline void BuildSourceDupCorePrefixes(
    const SourceDupConfig& config,
    PlanningResource& resource,
    AscendC::GlobalTensor<int32_t>& local_shmem_gm,
    AscendC::GlobalTensor<int32_t>& dup_workspace_gm,
    AscendC::GlobalTensor<int32_t>& dup_key_workspace_gm,
    int32_t prefix_inbox_off) {
    SourceDupCountTileLayout layout{};
    InitSourceDupCountTileLayout(config, layout);
    AscendC::LocalTensor<int32_t> core_counts_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(layout.core_counts_offset));
    AscendC::LocalTensor<int32_t> prefix_slots_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(layout.slots_offset));
    AscendC::LocalTensor<int32_t> result_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(layout.result_offset));

    int32_t dest_begin = 0;
    int32_t dest_count = 0;
    GetSourceDupRankPartition(config, dest_begin, dest_count);
    if (dest_count <= 0) {
        return;
    }

    WaitForSourceDupInbox(
        local_shmem_gm,
        prefix_slots_ub,
        result_ub,
        prefix_inbox_off + dest_begin * kSourceDupInboxSlotElems,
        dest_count);
    LoadSourceDupCoreCountTile(
        config,
        layout,
        dest_begin,
        dest_count,
        core_counts_ub,
        dup_workspace_gm);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    const int32_t count_elems = dest_count * 2;
    const uint32_t thread_count = static_cast<uint32_t>(
        ((count_elems + 31) / 32) * 32);
    asc_vf_call<BuildSourceDupCorePrefixesSimt>(
        dim3(thread_count),
        (__ubuf__ int32_t*)core_counts_ub.GetPhyAddr(),
        (__ubuf__ int32_t*)prefix_slots_ub.GetPhyAddr(),
        config.core_num,
        dest_count,
        layout.row_elems);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    StoreSourceDupCoreCountTile(
        config,
        layout,
        dest_begin,
        dest_count,
        dup_key_workspace_gm,
        core_counts_ub);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

__aicore__ inline void MaterializeSourceDupOutputs(
    const SourceDupConfig& config,
    const SourceDupUbLayout& ub_layout,
    PlanningResource& resource,
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

__aicore__ inline void ProcessSourceDupTile(
    const SourceDupConfig& config,
    const SourceDupPartition& partition,
    const SourceDupUbLayout& ub_layout,
    const SourceDupWorkspaceLayout& workspace_layout,
    PlanningResource& resource,
    AscendC::GlobalTensor<int32_t>& dst_gm,
    AscendC::GlobalTensor<int32_t>& dup_workspace_gm,
    AscendC::GlobalTensor<int32_t>& dup_key_workspace_gm,
    int32_t token_base,
    int32_t tile_idx) {
    AscendC::LocalTensor<int32_t> count_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(0);
    AscendC::LocalTensor<int32_t> dst_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.count_bytes));
    AscendC::LocalTensor<int32_t> meta_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.meta_offset));
    AscendC::LocalTensor<int32_t> raw_group_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.raw_group_offset));
    AscendC::LocalTensor<int32_t> raw_loff_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.raw_loff_offset));
    AscendC::LocalTensor<int32_t> group_send_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.group_send_offset));
    AscendC::LocalTensor<int32_t> tile_count_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.tile_count_offset));
    AscendC::LocalTensor<int32_t> token_group_count_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.token_group_count_offset));
    AscendC::LocalTensor<int32_t> token_loff_count_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.token_loff_count_offset));
    AscendC::LocalTensor<int32_t> token_group_prefix_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.token_group_prefix_offset));
    AscendC::LocalTensor<int32_t> token_loff_prefix_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.token_loff_prefix_offset));
    AscendC::LocalTensor<int32_t> compact_loff_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.compact_loff_offset));
    AscendC::LocalTensor<int32_t> pack_hist_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.pack_hist_offset));
    const int32_t tile_tokens = IntMin(config.tokens_per_tile, partition.token_end - token_base);
    const int32_t tile_routes = tile_tokens * config.topk;
    AscendC::Duplicate(tile_count_ub, static_cast<int32_t>(0), ub_layout.tile_count_row_elems);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::DataCopyExtParams copy_params(
        1,
        static_cast<uint32_t>(tile_routes * kInt32Bytes),
        0,
        0,
        0);
    AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
    AscendC::DataCopyPad(
        dst_ub,
        dst_gm[token_base * config.topk],
        copy_params,
        pad_params);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    const uint32_t simt_thread_count = static_cast<uint32_t>(
        ((tile_tokens + 31) / 32) * 32);
    asc_vf_call<DedupRouteTileSimt>(
        dim3(simt_thread_count),
        (__ubuf__ int32_t*)dst_ub.GetPhyAddr(),
        (__ubuf__ int32_t*)raw_group_ub.GetPhyAddr(),
        (__ubuf__ int32_t*)raw_loff_ub.GetPhyAddr(),
        (__ubuf__ int32_t*)token_group_count_ub.GetPhyAddr(),
        (__ubuf__ int32_t*)token_loff_count_ub.GetPhyAddr(),
        (__ubuf__ int32_t*)tile_count_ub.GetPhyAddr(),
        tile_tokens,
        config.topk,
        config.max_tokens_per_rank);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    int32_t raw_group_count = 0;
    int32_t raw_loff_count = 0;
    for (int32_t local_token = 0; local_token < tile_tokens; ++local_token) {
        const int32_t token_group_count =
            token_group_count_ub.GetValue(local_token);
        const int32_t token_loff_count =
            token_loff_count_ub.GetValue(local_token);
        token_group_prefix_ub.SetValue(local_token, raw_group_count);
        token_loff_prefix_ub.SetValue(local_token, raw_loff_count);
        raw_group_count += token_group_count;
        raw_loff_count += token_loff_count;
    }

    const bool tile_has_duplicates = raw_group_count > 0;
    if (tile_has_duplicates) {
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        asc_vf_call<CompactRouteTileSimt>(
            dim3(simt_thread_count),
            (__ubuf__ int32_t*)raw_group_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)raw_loff_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)token_group_count_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)token_loff_count_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)token_group_prefix_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)token_loff_prefix_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)group_send_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)compact_loff_ub.GetPhyAddr(),
            tile_tokens,
            config.topk);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    }
    if (tile_has_duplicates) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopyExtParams dst_copy_params(
            1,
            static_cast<uint32_t>(tile_routes * kInt32Bytes),
            0,
            0,
            0);
        AscendC::DataCopyPad(
            dst_gm[token_base * config.topk],
            dst_ub,
            dst_copy_params);
    }
    int32_t group_tile_prefix = 0;
    int32_t loff_tile_prefix = 0;
    for (int32_t dest = 0; dest < config.rank_count; ++dest) {
        const int32_t meta_offset = dest * 5;
        const int32_t count_offset = dest * 2;
        const int32_t group_count = tile_count_ub.GetValue(count_offset);
        const int32_t loff_count = tile_count_ub.GetValue(count_offset + 1);
        if (group_count > 0) {
            group_tile_prefix = AlignUpInt32(group_tile_prefix);
            loff_tile_prefix = AlignUpInt32(loff_tile_prefix);
        }
        meta_ub.SetValue(meta_offset, group_tile_prefix);
        meta_ub.SetValue(meta_offset + 1, group_tile_prefix);
        meta_ub.SetValue(meta_offset + 3, loff_tile_prefix);
        meta_ub.SetValue(meta_offset + 4, loff_tile_prefix);
        group_tile_prefix += group_count * 3;
        loff_tile_prefix += loff_count;
    }
    if (tile_has_duplicates) {
        const int32_t pack_chunk_count =
            (raw_group_count + ub_layout.pack_chunk_groups - 1) /
            ub_layout.pack_chunk_groups;
        const int32_t pack_hist_elems =
            pack_chunk_count * ub_layout.pack_hist_row_elems;
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::Duplicate(pack_hist_ub, static_cast<int32_t>(0), pack_hist_elems);
        AscendC::PipeBarrier<PIPE_V>();

        const int32_t pack_work_items =
            pack_chunk_count > config.rank_count
                ? pack_chunk_count
                : config.rank_count;
        const uint32_t pack_thread_count = static_cast<uint32_t>(IntMin(
            kSourceDupSimtMaxThreads, ((pack_work_items + 31) / 32) * 32));
        asc_vf_call<PackSourceDupByDestSimt>(
            dim3(pack_thread_count),
            (__ubuf__ int32_t*)group_send_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)compact_loff_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)meta_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)raw_group_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)raw_loff_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)pack_hist_ub.GetPhyAddr(),
            raw_group_count,
            raw_loff_count,
            config.rank_count,
            ub_layout.pack_chunk_groups,
            ub_layout.pack_hist_row_elems,
            static_cast<int32_t>(pack_thread_count));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    }
    for (int32_t i = 0; i < config.rank_count * 2; ++i) {
        count_ub.SetValue(
            i,
            count_ub.GetValue(i) + tile_count_ub.GetValue(i));
    }
    const int32_t tile_count_gm_offset =
        workspace_layout.tile_count_offset +
        (config.core_idx * partition.max_tiles_per_core + tile_idx) *
            ub_layout.tile_count_row_elems;
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::DataCopyExtParams tile_count_copy_params(
        1,
        static_cast<uint32_t>(config.rank_count * 2 * kInt32Bytes),
        0,
        0,
        0);
    AscendC::DataCopyPad(
        dup_workspace_gm[tile_count_gm_offset],
        tile_count_ub,
        tile_count_copy_params);

    if (tile_has_duplicates) {
        const int32_t tile_slot =
            config.core_idx * partition.max_tiles_per_core + tile_idx;
        AscendC::DataCopyExtParams group_copy_params(
            1,
            static_cast<uint32_t>(group_tile_prefix * kInt32Bytes),
            0,
            0,
            0);
        AscendC::DataCopyPad(
            dup_key_workspace_gm[
                workspace_layout.prebuilt_group_offset +
                tile_slot * workspace_layout.prebuilt_group_stride],
            raw_group_ub,
            group_copy_params);
        AscendC::DataCopyExtParams loff_copy_params(
            1,
            static_cast<uint32_t>(loff_tile_prefix * kInt32Bytes),
            0,
            0,
            0);
        AscendC::DataCopyPad(
            dup_key_workspace_gm[
                workspace_layout.prebuilt_loff_offset +
                tile_slot * workspace_layout.prebuilt_loff_stride],
            raw_loff_ub,
            loff_copy_params);
    }
    // The following tile reuses these UB regions from both V and MTE2. Wait
    // until all count, payload, and dst writebacks have consumed their input.
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
}

__aicore__ inline void CountAndPrebuildSourceDupTiles(
    const SourceDupConfig& config,
    const SourceDupPartition& partition,
    const SourceDupUbLayout& ub_layout,
    const SourceDupWorkspaceLayout& workspace_layout,
    PlanningResource& resource,
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

__aicore__ inline void PublishSourceDupPayloads(
    const SourceDupConfig& config,
    const SourceDupPartition& partition,
    const SourceDupUbLayout& ub_layout,
    const SourceDupWorkspaceLayout& workspace_layout,
    PlanningResource& resource,
    __gm__ int32_t* shmem_base,
    AscendC::GlobalTensor<int32_t>& dup_workspace_gm,
    AscendC::GlobalTensor<int32_t>& dup_key_workspace_gm,
    int32_t payload_done_flag_off,
    int32_t dup_groups_scratch_off,
    int32_t dup_loffs_scratch_off) {
    AscendC::LocalTensor<int32_t> count_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(0);
    AscendC::LocalTensor<int32_t> staging_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.count_bytes));
    AscendC::LocalTensor<int32_t> meta_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.meta_offset));
    AscendC::LocalTensor<int32_t> group_send_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.group_send_offset));
    AscendC::LocalTensor<int32_t> tile_count_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.tile_count_offset));

    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
    AscendC::DataCopyExtParams count_copy_params(
        1,
        static_cast<uint32_t>(config.rank_count * 2 * kInt32Bytes),
        0,
        0,
        0);
    AscendC::DataCopyPadExtParams<int32_t> count_pad_params{false, 0, 0, 0};
    AscendC::DataCopyPad(
        count_ub,
        dup_key_workspace_gm[config.core_idx * config.rank_count * 2],
        count_copy_params,
        count_pad_params);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

    int32_t tile_idx = 0;
    for (int32_t token_base = partition.token_begin;
         token_base < partition.token_end;
         token_base += config.tokens_per_tile, ++tile_idx) {
        const int32_t tile_count_gm_offset =
            workspace_layout.tile_count_offset +
            (config.core_idx * partition.max_tiles_per_core + tile_idx) *
                ub_layout.tile_count_row_elems;
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
        AscendC::DataCopyPad(
            tile_count_ub,
            dup_workspace_gm[tile_count_gm_offset],
            count_copy_params,
            count_pad_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

        int32_t group_tile_prefix = 0;
        int32_t loff_tile_prefix = 0;
        for (int32_t dest = 0; dest < config.rank_count; ++dest) {
            const int32_t meta_offset = dest * 5;
            const int32_t count_offset = dest * 2;
            const int32_t group_count = tile_count_ub.GetValue(count_offset);
            const int32_t loff_count = tile_count_ub.GetValue(count_offset + 1);
            if (group_count > 0) {
                group_tile_prefix = AlignUpInt32(group_tile_prefix);
                loff_tile_prefix = AlignUpInt32(loff_tile_prefix);
            }
            meta_ub.SetValue(meta_offset, group_tile_prefix);
            meta_ub.SetValue(meta_offset + 1, group_tile_prefix);
            meta_ub.SetValue(meta_offset + 3, loff_tile_prefix);
            meta_ub.SetValue(meta_offset + 4, loff_tile_prefix);
            group_tile_prefix += group_count * 3;
            loff_tile_prefix += loff_count;
        }

        const int32_t tile_slot =
            config.core_idx * partition.max_tiles_per_core + tile_idx;
        if (group_tile_prefix <= 0) {
            continue;
        }

        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
        AscendC::DataCopyExtParams group_load_params(
            1,
            static_cast<uint32_t>(group_tile_prefix * kInt32Bytes),
            0,
            0,
            0);
        AscendC::DataCopyPadExtParams<int32_t> group_pad_params{false, 0, 0, 0};
        AscendC::DataCopyPad(
            group_send_ub,
            dup_key_workspace_gm[
                workspace_layout.prebuilt_group_offset +
                tile_slot * workspace_layout.prebuilt_group_stride],
            group_load_params,
            group_pad_params);
        AscendC::DataCopyExtParams loff_load_params(
            1,
            static_cast<uint32_t>(loff_tile_prefix * kInt32Bytes),
            0,
            0,
            0);
        AscendC::DataCopyPadExtParams<int32_t> loff_pad_params{false, 0, 0, 0};
        AscendC::DataCopyPad(
            staging_ub,
            dup_key_workspace_gm[
                workspace_layout.prebuilt_loff_offset +
                tile_slot * workspace_layout.prebuilt_loff_stride],
            loff_load_params,
            loff_pad_params);

        for (int32_t dest = 0; dest < config.rank_count; ++dest) {
            const int32_t meta_offset = dest * 5;
            const int32_t count_offset = dest * 2;
            const int32_t group_count = tile_count_ub.GetValue(count_offset);
            if (group_count <= 0) {
                continue;
            }
            const int32_t global_loff_base =
                count_ub.GetValue(count_offset + 1);
            meta_ub.SetValue(meta_offset + 1, global_loff_base);
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        const int32_t threads_per_dest = IntMin(
            32,
            config.rank_count < kSourceDupSimtMaxThreads
                ? kSourceDupSimtMaxThreads / config.rank_count
                : 1);
        const int32_t logical_adjust_threads =
            config.rank_count * threads_per_dest;
        const uint32_t adjust_thread_count = static_cast<uint32_t>(IntMin(
            kSourceDupSimtMaxThreads,
            ((logical_adjust_threads + 31) / 32) * 32));
        asc_vf_call<AdjustSourceDupGroupLoffsSimt>(
            dim3(adjust_thread_count),
            (__ubuf__ int32_t*)group_send_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)tile_count_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)meta_ub.GetPhyAddr(),
            config.rank_count,
            threads_per_dest);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);

        for (int32_t dest = 0; dest < config.rank_count; ++dest) {
            const int32_t meta_offset = dest * 5;
            const int32_t group_start = meta_ub.GetValue(meta_offset);
            const int32_t count_offset = dest * 2;
            const int32_t group_count = tile_count_ub.GetValue(count_offset);
            if (group_count <= 0) {
                continue;
            }

            const int32_t group_elems = group_count * 3;
            const int32_t global_group_pos =
                count_ub.GetValue(count_offset);
            AscendC::DataCopyExtParams group_store_params(
                1,
                static_cast<uint32_t>(group_elems * kInt32Bytes),
                0,
                0,
                0);
            AscendC::GlobalTensor<int32_t> remote_shmem_gm;
            remote_shmem_gm.SetGlobalBuffer(
                static_cast<__gm__ int32_t*>(
                    aclshmem_ptr(shmem_base, dest)));
            AscendC::DataCopyPad(
                remote_shmem_gm[
                    dup_groups_scratch_off + global_group_pos * 3],
                group_send_ub[group_start],
                group_store_params);
            count_ub.SetValue(
                count_offset,
                global_group_pos + group_count);

            const int32_t loff_start = meta_ub.GetValue(meta_offset + 3);
            const int32_t loff_count = tile_count_ub.GetValue(count_offset + 1);
            const int32_t global_loff_pos =
                count_ub.GetValue(count_offset + 1);
            AscendC::DataCopyExtParams loff_copy_params(
                1,
                static_cast<uint32_t>(loff_count * kInt32Bytes),
                0,
                0,
                0);
            AscendC::DataCopyPad(
                remote_shmem_gm[dup_loffs_scratch_off + global_loff_pos],
                staging_ub[loff_start],
                loff_copy_params);
            count_ub.SetValue(
                count_offset + 1,
                global_loff_pos + loff_count);
        }
        // The next tile reloads the payload into the same UB regions.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    AscendC::SyncAll<true>();
    PublishSourceDupPayloadDoneFlags(
        config,
        shmem_base,
        count_ub,
        payload_done_flag_off);
}

}  // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void planning_kernel(
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
    ascend_moonep::PlanningTilingData tiling,
    uint64_t ffts_addr) {
    (void)ffts_addr;

    const int32_t core_idx = get_block_idx() * get_subblockdim() + get_subblockid();
    const int32_t core_num = get_block_num() * get_subblockdim();

    const int32_t rank = static_cast<int32_t>(tiling.rank);
    const int32_t R = static_cast<int32_t>(tiling.world_size);
    const int32_t S = static_cast<int32_t>(tiling.num_tokens);
    const int32_t K = static_cast<int32_t>(tiling.num_topk);
    const int32_t E = static_cast<int32_t>(tiling.num_experts);
    const int32_t NvS =
        static_cast<int32_t>(tiling.num_max_tokens_per_rank);
    const int32_t B = static_cast<int32_t>(tiling.num_prefetch_slots);
    const int32_t epn = E / R;
    const int32_t N = S * K;
    const int32_t tpe_off =
        static_cast<int32_t>(tiling.symmetric_tpe_offset);
    const int32_t dup_count_inbox_off =
        static_cast<int32_t>(tiling.symmetric_dup_rank_counts_offset);
    const int32_t dup_prefix_inbox_off =
        dup_count_inbox_off + R * kSourceDupInboxSlotElems;
    const int32_t dup_payload_done_flag_off =
        dup_prefix_inbox_off + R * kSourceDupInboxSlotElems;
    const int32_t dup_groups_scratch_off =
        dup_payload_done_flag_off + ((R + 7) / 8) * 8;
    const int32_t dup_loffs_scratch_off =
        dup_groups_scratch_off + ((NvS * 3 + 7) / 8) * 8;
    const int32_t source_dup_tokens_per_tile =
        static_cast<int32_t>(tiling.source_dup_tokens_per_tile);
    const int32_t owner_reduce_tile_elems =
        static_cast<int32_t>(tiling.owner_reduce_tile_elems);
    const int32_t route_core_num = IntMin(core_num, kMaxPlanningCores);
    const int32_t items_per_core = (N + route_core_num - 1) / route_core_num;
    const int32_t begin = core_idx * items_per_core;
    const int32_t end = IntMin(begin + items_per_core, N);
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

    topk_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(topk_idx));
    dst_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(dst));
    cu_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cu_seqlens));
    experts_to_copy_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(experts_to_copy));
    dup_groups_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(dup_groups));
    dup_loffs_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(dup_loffs));
    dup_counts_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(dup_counts));
    tpe_all_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tpe_all));
    tpe_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(tpe_all) + rank * E);
    tpe_prefix_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tpe_prefix));
    balance_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(balance));
    z_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(z_tensor));
    alloc_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(alloc_tensor));
    alloc_cumsum_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(alloc_cumsum));
    expert_offsets_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(expert_offsets));
    core_expert_counts_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(core_expert_counts));
    cursor_worker_scratch_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(core_expert_counts) +
        kMaxPlanningCores * E);
    dup_workspace_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(dup_workspace));
    dup_key_workspace_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(dup_key_workspace));

    __gm__ int32_t* shmem_base = reinterpret_cast<__gm__ int32_t*>(symmetric_buffer);
    local_shmem_gm.SetGlobalBuffer(shmem_base);

    using ArchTag = PlanningAscend950;
    Catlass::Arch::Resource<ArchTag> resource;
    AscendC::LocalTensor<int32_t> work_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(0);

    // The first cross-rank barrier also publishes the mailbox initialization.
    // Count and prefix messages each own one 32-byte slot per peer; payload
    // completion uses one flag per source rank after all local cores finish.
    FillContiguousGm(
        local_shmem_gm,
        work_ub,
        dup_count_inbox_off,
        R * kSourceDupInboxSlotElems * 2,
        -1,
        core_idx,
        core_num);
    FillContiguousGm(
        local_shmem_gm,
        work_ub,
        dup_payload_done_flag_off,
        R,
        0,
        core_idx,
        core_num);

    int32_t tpe_elems_per_core = E / core_num;
    int32_t tpe_tail_cores = E - tpe_elems_per_core * core_num;
    int32_t tpe_begin = core_idx * tpe_elems_per_core + IntMin(core_idx, tpe_tail_cores);
    int32_t tpe_assigned = tpe_elems_per_core + (core_idx < tpe_tail_cores ? 1 : 0);
    int32_t tpe_end = tpe_begin + tpe_assigned;

    // Stage 1: build one expert histogram per route-processing core. The same
    // rows later become exclusive per-core expert prefixes for dst ordinals.
    if (core_idx < route_core_num) {
        const int32_t histogram_bytes = AlignUpBytes(E * kInt32Bytes, 32);
        AscendC::LocalTensor<int32_t> route_tile_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(
                static_cast<uint32_t>(histogram_bytes));
        AscendC::Duplicate(work_ub, static_cast<int32_t>(0), E);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        for (int32_t base = begin; base < end; base += kRouteHistogramTileElems) {
            const int32_t len =
                IntMin(kRouteHistogramTileElems, end - base);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::DataCopyExtParams copy_params(
                1,
                static_cast<uint32_t>(len * kInt32Bytes),
                0,
                0,
                0);
            AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
            AscendC::DataCopyPad(
                route_tile_ub,
                topk_gm[base],
                copy_params,
                pad_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            const uint32_t thread_count = static_cast<uint32_t>(IntMin(
                ((len + 31) / 32) * 32,
                kSourceDupSimtMaxThreads));
            asc_vf_call<BuildRouteHistogramSimt>(
                dim3(thread_count),
                (__ubuf__ int32_t*)route_tile_ub.GetPhyAddr(),
                (__ubuf__ int32_t*)work_ub.GetPhyAddr(),
                len);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        for (int32_t base = 0; base < E; base += kFillTileElems) {
            const int32_t len = IntMin(kFillTileElems, E - base);
            AscendC::DataCopyExtParams copy_params(
                1,
                static_cast<uint32_t>(len * kInt32Bytes),
                0,
                0,
                0);
            AscendC::DataCopyPad(
                core_expert_counts_gm[core_idx * E + base],
                work_ub[base],
                copy_params);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    }
    AscendC::SyncAll<true>();

    // Reduce [route_core_num, E] by expert slices. A 64-expert slice is one
    // 256-byte vector width, while multiple cores remove the core-0 bottleneck.
    const int32_t tpe_reduce_core_num = IntMin(
        core_num, (E + kTpeReduceExpertsPerCore - 1) / kTpeReduceExpertsPerCore);

    if (core_idx < tpe_reduce_core_num) {
        constexpr int32_t kTpeReduceTmpOffsetBytes =
            ((kFillTileElems * kInt32Bytes + 31) / 32) * 32;
        AscendC::LocalTensor<int32_t> tpe_reduce_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(
                static_cast<uint32_t>(kTpeReduceTmpOffsetBytes));
        static_assert(
            kTpeReduceTmpOffsetBytes +
                    kMaxPlanningCores * kTpeReduceRegRowBytes <=
                kPlanningUbSizeBytes);

        // EVENT_ID1 owns tpe_reduce_ub. The next MTE2 load must not overwrite
        // it until the previous Reg VF has finished reading the matrix.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        for (int32_t base = core_idx * kTpeReduceExpertsPerCore;
             base < E;
             base += tpe_reduce_core_num * kTpeReduceExpertsPerCore) {
            const int32_t len = IntMin(kTpeReduceExpertsPerCore, E - base);
            AscendC::DataCopyExtParams copy_params(
                1,
                static_cast<uint32_t>(len * kInt32Bytes),
                0,
                0,
                0);
            AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};

            const int32_t aligned_row_bytes =
                ((len * kInt32Bytes + 31) / 32) * 32;
            AscendC::DataCopyExtParams matrix_copy_params(
                static_cast<uint16_t>(route_core_num),
                static_cast<uint32_t>(len * kInt32Bytes),
                static_cast<uint32_t>((E - len) * kInt32Bytes),
                static_cast<uint32_t>(
                    (kTpeReduceRegRowBytes - aligned_row_bytes) / 32),
                0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::DataCopyPad(
                tpe_reduce_ub,
                core_expert_counts_gm[base],
                matrix_copy_params,
                pad_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            asc_vf_call<ReduceTpeRowsReg>(
                (__ubuf__ int32_t*)work_ub.GetPhyAddr(),
                (__ubuf__ int32_t*)tpe_reduce_ub.GetPhyAddr(),
                static_cast<uint32_t>(route_core_num),
                static_cast<uint32_t>(len),
                static_cast<uint32_t>(kTpeReduceExpertsPerCore));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopyPad(
                local_shmem_gm[tpe_off + base],
                work_ub,
                copy_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    }
    aclshmemx_barrier_all_vec();

    // Stage 2: allgather tpe by contiguous (peer, expert-tile) tasks. This
    // avoids issuing one tiny copy per peer from every core when E is small.
    constexpr int32_t kTpeGatherTileBytes = ((kFillTileElems * kInt32Bytes + 31) / 32) * 32;
    static_assert(kTpeGatherTileBytes * 2 <= kPlanningUbSizeBytes);
    AscendC::LocalTensor<int32_t> tpe_gather_pingpong_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(0);
    const int32_t tpe_tiles_per_peer = (E + kFillTileElems - 1) / kFillTileElems;
    const int32_t tpe_gather_task_count = R * tpe_tiles_per_peer;
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    int32_t local_task_idx = 0;
    for (int32_t task_idx = core_idx; task_idx < tpe_gather_task_count;
         task_idx += core_num, ++local_task_idx) {
        const int32_t peer = task_idx / tpe_tiles_per_peer;
        const int32_t tile_idx = task_idx - peer * tpe_tiles_per_peer;
        const int32_t base = tile_idx * kFillTileElems;
        const int32_t len = IntMin(kFillTileElems, E - base);
        const int32_t buffer_idx = local_task_idx & 1;
        const auto event_id = buffer_idx == 0 ? EVENT_ID0 : EVENT_ID1;
        AscendC::LocalTensor<int32_t> tpe_gather_ub =
            tpe_gather_pingpong_ub[buffer_idx * kFillTileElems];
        AscendC::GlobalTensor<int32_t> peer_shmem_gm;
        peer_shmem_gm.SetGlobalBuffer(static_cast<__gm__ int32_t*>(aclshmem_ptr(shmem_base, peer)));
        AscendC::DataCopyExtParams copy_params(
            1,
            static_cast<uint32_t>(len * kInt32Bytes),
            0,
            0,
            0);
        AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
        AscendC::DataCopyPad(
            tpe_gather_ub,
            peer_shmem_gm[tpe_off + base],
            copy_params,
            pad_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);

        AscendC::DataCopyPad(
            tpe_all_gm[peer * E + base],
            tpe_gather_ub,
            copy_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

    // Keep the V-produced fill tile separate from both gather buffers. The
    // gather completion events order MTE3 against MTE2, not against PIPE_V.
    constexpr int32_t kZFillUbOffsetBytes = kTpeGatherTileBytes * 2;
    static_assert(kZFillUbOffsetBytes + kFillTileElems * kInt32Bytes <= kPlanningUbSizeBytes);
    AscendC::LocalTensor<int32_t> z_fill_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(static_cast<uint32_t>(kZFillUbOffsetBytes));
    FillContiguousGm(z_gm, z_fill_ub, 0, R * R, 0, core_idx, core_num);

    AscendC::SyncAll<true>();

    // balance[owner] and the home alloc row are independent per owner rank.
    {
        const int32_t reduce_tile_elems = owner_reduce_tile_elems;
        const int32_t reduce_row_bytes = ((reduce_tile_elems * kInt32Bytes + 31) / 32) * 32;
        const int32_t reduce_row_elems = reduce_row_bytes / kInt32Bytes;
        const int32_t reduce_matrix_bytes = R * reduce_row_bytes;
        // Host selects the tile after reserving the tile total and maximum
        // per-core balance output, keeping all three regions disjoint.
        AscendC::LocalTensor<int32_t> tpe_owner_matrix_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(0);
        AscendC::LocalTensor<int32_t> tile_total_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(
                static_cast<uint32_t>(reduce_matrix_bytes));
        AscendC::LocalTensor<int32_t> balance_out_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(
                static_cast<uint32_t>(
                    reduce_matrix_bytes + kOwnerReduceTotalBytes));

        const int32_t owners_per_core = R / core_num;
        const int32_t owner_tail_cores = R - owners_per_core * core_num;
        const int32_t owner_begin =
            core_idx * owners_per_core + IntMin(core_idx, owner_tail_cores);
        const int32_t owner_assigned = owners_per_core + (core_idx < owner_tail_cores ? 1 : 0);
        const int32_t owner_end = owner_begin + owner_assigned;

        // Each core accumulates token counts for its owner_assigned ranks.
        for (int32_t owner = owner_begin; owner < owner_end; ++owner) {
            int32_t total = 0;
            int32_t e_begin = owner * epn;
            int32_t e_end = e_begin + epn;

            for (int32_t base = e_begin; base < e_end; base += reduce_tile_elems) {
                const int32_t len = IntMin(reduce_tile_elems, e_end - base);
                const int32_t aligned_row_bytes =
                    ((len * kInt32Bytes + 31) / 32) * 32;
                AscendC::DataCopyExtParams matrix_copy_params(
                    static_cast<uint16_t>(R),
                    static_cast<uint32_t>(len * kInt32Bytes),
                    static_cast<uint32_t>((E - len) * kInt32Bytes),
                    static_cast<uint32_t>(
                        (reduce_row_bytes - aligned_row_bytes) / 32),
                    0);
                AscendC::DataCopyPadExtParams<int32_t> pad_params{
                    false, 0, 0, 0};
                AscendC::DataCopyPad(
                    tpe_owner_matrix_ub,
                    tpe_all_gm[base],
                    matrix_copy_params,
                    pad_params);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                asc_vf_call<ReduceTpeOwnerRowsReg>(
                    (__ubuf__ int32_t*)tpe_owner_matrix_ub.GetPhyAddr(),
                    (__ubuf__ int32_t*)tile_total_ub.GetPhyAddr(),
                    (__ubuf__ int32_t*)tpe_owner_matrix_ub.GetPhyAddr(),
                    static_cast<uint32_t>(R),
                    static_cast<uint32_t>(len),
                    static_cast<uint32_t>(reduce_row_elems));

                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                total += tile_total_ub.GetValue(0);
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
                StoreContiguousGmThroughUb(
                    alloc_gm,
                    tpe_owner_matrix_ub,
                    owner * E + base,
                    len);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
            const int32_t owner_local = owner - owner_begin;
            balance_out_ub.SetValue(owner_local, total - N);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        StoreContiguousGmThroughUb(balance_gm, balance_out_ub, owner_begin, owner_assigned);
    }
    AscendC::SyncAll<true>();

    if (core_idx == 0) {
        // The previous work_ub users have completed and synchronized here, so
        // the balance planner can reuse UB from offset 0 with a semantic name.
        AscendC::LocalTensor<int32_t> balance_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(0);
        {
            AscendC::DataCopyExtParams copy_params(
                1,
                static_cast<uint32_t>(R * kInt32Bytes),
                0,
                0,
                0);
            AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
            AscendC::DataCopyPad(balance_ub, balance_gm[0], copy_params, pad_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        }

        for (int32_t iter = 0; iter < R * R; ++iter) {
            int32_t surplus_rank = 0;
            int32_t deficit_rank = 0;
            int32_t max_balance = balance_ub.GetValue(0);
            int32_t min_balance = balance_ub.GetValue(0);
            for (int32_t r = 1; r < R; ++r) {
                int32_t v = balance_ub.GetValue(r);
                if (v > max_balance) {
                    max_balance = v;
                    surplus_rank = r;
                }
                if (v < min_balance) {
                    min_balance = v;
                    deficit_rank = r;
                }
            }
            if (max_balance <= 0 || min_balance >= 0) {
                break;
            }
            int32_t move_tokens = -min_balance;
            AscendC::WriteGmByPassDCache<int32_t>(
                z_gm.GetPhyAddr(surplus_rank * R + deficit_rank),
                move_tokens);
            balance_ub.SetValue(surplus_rank, max_balance - move_tokens);
            balance_ub.SetValue(deficit_rank, 0);
        }
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }
    AscendC::SyncAll<true>();

    {
        const int32_t s_alloc_row_bytes = ((epn * kInt32Bytes + 31) / 32) * 32;
        const int32_t s_alloc_row_elems = s_alloc_row_bytes / kInt32Bytes;
        const int32_t s_alloc_elems = s_alloc_row_elems * R;
        const int32_t s_alloc_bytes = s_alloc_row_bytes * R;
        AscendC::LocalTensor<int32_t> s_alloc_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(0);
        AscendC::LocalTensor<int32_t> quota_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(static_cast<uint32_t>(s_alloc_bytes));

        for (int32_t owner = core_idx; owner < R; owner += core_num) {
            const int32_t expert_base = owner * epn;
            const int32_t owner_row_offset = owner * s_alloc_row_elems;

            AscendC::Duplicate(s_alloc_ub, static_cast<int32_t>(0), s_alloc_elems);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

            {
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopyExtParams copy_params(
                    1,
                    static_cast<uint32_t>(epn * kInt32Bytes),
                    0,
                    0,
                    0);
                AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
                AscendC::DataCopyPad(
                    s_alloc_ub[owner_row_offset],
                    alloc_gm[owner * E + expert_base],
                    copy_params,
                    pad_params);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            }

            {
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopyExtParams copy_params(
                    1,
                    static_cast<uint32_t>(R * kInt32Bytes),
                    0,
                    0,
                    0);
                AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
                AscendC::DataCopyPad(quota_ub, z_gm[owner * R], copy_params, pad_params);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            }

            // Each owner owns a disjoint expert range, so different cores write
            // disjoint alloc_gm columns.  Inside one owner, follow MoonEP's
            // greedy policy: largest remaining destination quota first, then
            // largest remaining local expert segment.
            for (int32_t iter = 0; iter < R + epn; ++iter) {
                int32_t target_rank = 0;
                int32_t max_quota = quota_ub.GetValue(0);
                for (int32_t r = 1; r < R; ++r) {
                    const int32_t quota = quota_ub.GetValue(r);
                    if (quota > max_quota) {
                        max_quota = quota;
                        target_rank = r;
                    }
                }
                if (max_quota <= 0) {
                    break;
                }

                int32_t selected_local_expert = 0;
                int32_t max_remaining = s_alloc_ub.GetValue(owner_row_offset);
                for (int32_t local_expert = 1; local_expert < epn; ++local_expert) {
                    const int32_t remaining = s_alloc_ub.GetValue(owner_row_offset + local_expert);
                    if (remaining > max_remaining) {
                        max_remaining = remaining;
                        selected_local_expert = local_expert;
                    }
                }
                if (max_remaining <= 0) {
                    break;
                }

                const int32_t take = IntMin(max_remaining, max_quota);
                const int32_t owner_offset = owner_row_offset + selected_local_expert;
                const int32_t target_offset =
                    target_rank * s_alloc_row_elems + selected_local_expert;
                s_alloc_ub.SetValue(
                    target_offset,
                    s_alloc_ub.GetValue(target_offset) + take);
                s_alloc_ub.SetValue(owner_offset, max_remaining - take);
                quota_ub.SetValue(target_rank, max_quota - take);
            }

            // One owner contributes one compact expert segment to every
            // destination row. UB rows are 32-byte aligned; DataCopyPad keeps
            // only the logical bytes when scattering them into compact GM.
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyExtParams alloc_store_params(
                static_cast<uint16_t>(R),
                static_cast<uint32_t>(epn * kInt32Bytes),
                0,
                static_cast<uint32_t>((E - epn) * kInt32Bytes),
                0);
            AscendC::DataCopyPad(
                alloc_gm[expert_base],
                s_alloc_ub,
                alloc_store_params);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            const uint32_t cumsum_thread_count = static_cast<uint32_t>(
                IntMin(((E + 31) / 32) * 32, kSourceDupSimtMaxThreads));
            asc_vf_call<BuildAllocCumsumSimt>(
                dim3(cumsum_thread_count),
                (__ubuf__ int32_t*)s_alloc_ub.GetPhyAddr(),
                reinterpret_cast<__gm__ int32_t*>(alloc_cumsum),
                R,
                epn,
                s_alloc_row_elems,
                expert_base,
                static_cast<int32_t>(cumsum_thread_count));
        }
    }
    AscendC::SyncAll<true>();
    // The selected experts are moved from their original E slots to the B
    // trailing slots in one in-place E+B count row. The scratch rows hold
    // Top-B reduction candidates, then are reused by prefix/offset building.
    PrefetchUbLayout prefetch_layout;
    InitPrefetchUbLayout(E, B, prefetch_layout);
    AscendC::LocalTensor<int32_t> group_counts_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(prefetch_layout.group_counts_offset));
    AscendC::LocalTensor<int32_t> selected_experts_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(prefetch_layout.selected_experts_offset));
    AscendC::LocalTensor<int32_t> scratch_counts_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(prefetch_layout.scratch_counts_offset));
    AscendC::LocalTensor<int32_t> scratch_experts_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(prefetch_layout.scratch_experts_offset));
    const int32_t prefetch_thread_count = SelectPrefetchSimtThreadCount(E);
    for (int32_t d = core_idx; d < R; d += core_num) {
        AscendC::DataCopyExtParams copy_params(
            1,
            static_cast<uint32_t>(E * kInt32Bytes),
            0,
            0,
            0);
        AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
        AscendC::DataCopyPad(
            group_counts_ub,
            alloc_gm[d * E],
            copy_params,
            pad_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        const int32_t local_expert_begin = d * epn;
        const int32_t local_expert_end = local_expert_begin + epn;
        asc_vf_call<SelectPrefetchExpertsSimt>(
            dim3(static_cast<uint32_t>(prefetch_thread_count)),
            (__ubuf__ int32_t*)group_counts_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)selected_experts_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)scratch_counts_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)scratch_experts_ub.GetPhyAddr(),
            E,
            B,
            local_expert_begin,
            local_expert_end,
            prefetch_thread_count);
        if ((B * kInt32Bytes) % 32 == 0) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            StoreContiguousGmThroughUb(
                experts_to_copy_gm,
                selected_experts_ub,
                d * B,
                B);
        } else {
            // Packed B-wide rows may share a 32-byte GM block. Concurrent
            // padded MTE3 stores from different cores would then overwrite
            // neighboring rows, so use exact scalar stores for this tail case.
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            for (int32_t slot = 0; slot < B; ++slot) {
                AscendC::WriteGmByPassDCache<int32_t>(
                    experts_to_copy_gm.GetPhyAddr(d * B + slot),
                    selected_experts_ub.GetValue(slot));
            }
        }

        asc_vf_call<BuildGroupPrefixSimt>(
            dim3(static_cast<uint32_t>(prefetch_thread_count)),
            (__ubuf__ int32_t*)group_counts_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)scratch_counts_ub.GetPhyAddr(),
            E + B,
            prefetch_thread_count);
        if (d == rank) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            StoreContiguousGmThroughUb(cu_gm, group_counts_ub, 0, E + B);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }

        asc_vf_call<ConvertGroupPrefixToExpertOffsetsSimt>(
            dim3(static_cast<uint32_t>(prefetch_thread_count)),
            (__ubuf__ int32_t*)group_counts_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)selected_experts_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)scratch_counts_ub.GetPhyAddr(),
            E,
            B,
            prefetch_thread_count);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        StoreContiguousGmThroughUb(
            expert_offsets_gm,
            group_counts_ub,
            d * E,
            E);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    if ((B * kInt32Bytes) % 32 != 0) {
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    CleanInvalidDataCache();

    // tpe_prefix[e] = sum_{r < rank} tpe_all[r, e]. Load all contributing
    // rows for one expert tile and keep the cross-rank accumulation in vector
    // registers, so the loop does not materialize every intermediate sum.
    if (rank == 0) {
        FillContiguousGm(
            tpe_prefix_gm,
            work_ub,
            0,
            E,
            0,
            core_idx,
            core_num);
    } else {
        constexpr int32_t kTpePrefixMatrixOffsetBytes = kTpeReduceRegRowBytes;
        const int32_t tpe_prefix_row_bytes = IntMin(
            kTpeReduceRegRowBytes,
            ((kPlanningUbSizeBytes - kTpePrefixMatrixOffsetBytes) / R / 32) * 32);
        const int32_t tpe_prefix_tile_elems = tpe_prefix_row_bytes / kInt32Bytes;
        AscendC::LocalTensor<int32_t> tpe_prefix_matrix_ub = resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(kTpePrefixMatrixOffsetBytes));
        AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        for (int32_t base = tpe_begin;
             base < tpe_end;
             base += tpe_prefix_tile_elems) {
            const int32_t len =
                IntMin(tpe_prefix_tile_elems, tpe_end - base);
            const int32_t aligned_row_bytes =
                AlignUpBytes(len * kInt32Bytes, 32);
            AscendC::DataCopyExtParams matrix_copy_params(
                static_cast<uint16_t>(rank),
                static_cast<uint32_t>(len * kInt32Bytes),
                static_cast<uint32_t>((E - len) * kInt32Bytes),
                static_cast<uint32_t>(
                    (tpe_prefix_row_bytes - aligned_row_bytes) / 32),
                0);
            AscendC::DataCopyExtParams output_copy_params(
                1,
                static_cast<uint32_t>(len * kInt32Bytes),
                0,
                0,
                0);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::DataCopyPad(
                tpe_prefix_matrix_ub,
                tpe_all_gm[base],
                matrix_copy_params,
                pad_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            asc_vf_call<ReduceTpeRowsReg>(
                (__ubuf__ int32_t*)work_ub.GetPhyAddr(),
                (__ubuf__ int32_t*)tpe_prefix_matrix_ub.GetPhyAddr(),
                static_cast<uint32_t>(rank),
                static_cast<uint32_t>(len),
                static_cast<uint32_t>(tpe_prefix_row_bytes / kInt32Bytes));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopyPad(
                tpe_prefix_gm[base],
                work_ub,
                output_copy_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    }
    AscendC::SyncAll<true>();

    // Split 32-byte-aligned expert columns across AIV cores. If an expert row
    // itself is not 32-byte aligned, keep one worker so two MTE3 writers cannot
    // touch the same GM cache line at a row boundary.
    constexpr int32_t kExpertsPerDataBlock = 32 / kInt32Bytes;
    const int32_t cursor_expert_block_count =
        (E + kExpertsPerDataBlock - 1) / kExpertsPerDataBlock;
    const int32_t cursor_worker_num = E % kExpertsPerDataBlock == 0
        ? IntMin(route_core_num, cursor_expert_block_count)
        : 1;
    const int32_t cursor_blocks_per_worker =
        cursor_expert_block_count / cursor_worker_num;
    const int32_t cursor_block_tail_workers =
        cursor_expert_block_count -
        cursor_blocks_per_worker * cursor_worker_num;
    const int32_t cursor_block_begin =
        core_idx * cursor_blocks_per_worker +
        IntMin(core_idx, cursor_block_tail_workers);
    const int32_t cursor_block_count =
        cursor_blocks_per_worker +
        (core_idx < cursor_block_tail_workers ? 1 : 0);
    const int32_t cursor_expert_begin = IntMin(
        E,
        cursor_block_begin * kExpertsPerDataBlock);
    const int32_t cursor_expert_end = IntMin(
        E,
        (cursor_block_begin + cursor_block_count) *
            kExpertsPerDataBlock);
    const int32_t cursor_expert_count =
        cursor_expert_end - cursor_expert_begin;

    // Publish one total per worker into a dedicated, 32-byte-strided GM row.
    // This scratch is outside the source-dup workspace and is fully overwritten
    // on every launch before any worker reads it.
    if (core_idx < cursor_worker_num) {
        constexpr int32_t kTotalTileBytes =
            kFillTileElems * kInt32Bytes;
        constexpr int32_t kThreadTotalsOffset = kTotalTileBytes;
        constexpr int32_t kTileTotalOffset =
            kThreadTotalsOffset + kCoreCursorThreadPrefixBytes;
        static_assert(
            kTileTotalOffset + kCoreCursorCarryBytes <=
            kPlanningUbSizeBytes);

        AscendC::LocalTensor<int32_t> tpe_tile_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(0);
        AscendC::LocalTensor<int32_t> thread_totals_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(
                static_cast<uint32_t>(kThreadTotalsOffset));
        AscendC::LocalTensor<int32_t> tile_total_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(
                static_cast<uint32_t>(kTileTotalOffset));
        AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};

        const bool accumulate_total_tiles =
            cursor_expert_count > kFillTileElems;
        int32_t worker_total = 0;
        for (int32_t base = cursor_expert_begin;
             base < cursor_expert_end;
             base += kFillTileElems) {
            const int32_t len =
                IntMin(kFillTileElems, cursor_expert_end - base);
            AscendC::DataCopyExtParams copy_params(
                1,
                static_cast<uint32_t>(len * kInt32Bytes),
                0,
                0,
                0);
            AscendC::DataCopyPad(
                tpe_tile_ub,
                tpe_gm[base],
                copy_params,
                pad_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            const int32_t thread_count = IntMin(
                kCoreCursorSimtMaxThreads,
                ((len + 31) / 32) * 32);
            asc_vf_call<ReduceExpertTileTotalSimt>(
                dim3(static_cast<uint32_t>(thread_count)),
                (__ubuf__ int32_t*)tpe_tile_ub.GetPhyAddr(),
                (__ubuf__ int32_t*)thread_totals_ub.GetPhyAddr(),
                (__ubuf__ int32_t*)tile_total_ub.GetPhyAddr(),
                len,
                thread_count);
            if (accumulate_total_tiles) {
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                worker_total += tile_total_ub.GetValue(0);
            }
        }

        if (accumulate_total_tiles) {
            tile_total_ub.SetValue(0, worker_total);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        }
        AscendC::DataCopyExtParams total_store_params(
            1,
            32,
            0,
            0,
            0);
        AscendC::DataCopyPad(
            cursor_worker_scratch_gm[
                core_idx * kCursorWorkerScratchRowElems],
            tile_total_ub,
            total_store_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    }
    AscendC::SyncAll<true>();

    // Convert [route_core, expert] counts to expert-major cursors. A worker
    // starts from the sum of all preceding expert partitions and advances its
    // carry across local tiles.
    if (core_idx < cursor_worker_num) {
        const int32_t fixed_ub_bytes =
            kCoreCursorThreadPrefixBytes + kCoreCursorCarryBytes;
        const int32_t max_cursor_row_bytes =
            (((kPlanningUbSizeBytes - fixed_ub_bytes) /
              (route_core_num + 1)) /
             32) *
            32;
        const int32_t cursor_tile_elems = IntMin(
            IntMin(kFillTileElems, cursor_expert_count),
            max_cursor_row_bytes / kInt32Bytes);
        const int32_t cursor_row_bytes =
            AlignUpBytes(cursor_tile_elems * kInt32Bytes, 32);
        const int32_t cursor_matrix_bytes =
            route_core_num * cursor_row_bytes;
        const int32_t tpe_tile_offset = cursor_matrix_bytes;
        const int32_t thread_prefix_offset =
            tpe_tile_offset + cursor_row_bytes;
        const int32_t expert_carry_offset =
            thread_prefix_offset + kCoreCursorThreadPrefixBytes;

        AscendC::LocalTensor<int32_t> cursor_matrix_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(0);
        AscendC::LocalTensor<int32_t> tpe_tile_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(
                static_cast<uint32_t>(tpe_tile_offset));
        AscendC::LocalTensor<int32_t> thread_prefix_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(
                static_cast<uint32_t>(thread_prefix_offset));
        AscendC::LocalTensor<int32_t> expert_carry_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(
                static_cast<uint32_t>(expert_carry_offset));

        AscendC::DataCopyExtParams totals_load_params(
            1,
            static_cast<uint32_t>(
                cursor_worker_num * kCursorWorkerScratchRowElems *
                kInt32Bytes),
            0,
            0,
            0);
        AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
        AscendC::DataCopyPad(
            cursor_matrix_ub,
            cursor_worker_scratch_gm[0],
            totals_load_params,
            pad_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        int32_t expert_base = 0;
        for (int32_t worker = 0; worker < core_idx; ++worker) {
            expert_base += cursor_matrix_ub.GetValue(
                worker * kCursorWorkerScratchRowElems);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
        AscendC::Duplicate(
            expert_carry_ub,
            expert_base,
            kCoreCursorCarryBytes / kInt32Bytes);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        for (int32_t base = cursor_expert_begin;
             base < cursor_expert_end;
             base += cursor_tile_elems) {
            const int32_t len =
                IntMin(cursor_tile_elems, cursor_expert_end - base);
            const int32_t active_row_bytes = len * kInt32Bytes;
            const int32_t aligned_active_row_bytes =
                AlignUpBytes(active_row_bytes, 32);
            AscendC::DataCopyExtParams matrix_load_params(
                static_cast<uint16_t>(route_core_num),
                static_cast<uint32_t>(active_row_bytes),
                static_cast<uint32_t>((E - len) * kInt32Bytes),
                static_cast<uint32_t>(
                    (cursor_row_bytes - aligned_active_row_bytes) / 32),
                0);
            AscendC::DataCopyExtParams tpe_load_params(
                1,
                static_cast<uint32_t>(active_row_bytes),
                0,
                0,
                0);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::DataCopyPad(
                cursor_matrix_ub,
                core_expert_counts_gm[base],
                matrix_load_params,
                pad_params);
            AscendC::DataCopyPad(
                tpe_tile_ub,
                tpe_gm[base],
                tpe_load_params,
                pad_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            const int32_t thread_count = IntMin(
                kCoreCursorSimtMaxThreads,
                ((len + 31) / 32) * 32);
            asc_vf_call<BuildCoreExpertCursorsSimt>(
                dim3(static_cast<uint32_t>(thread_count)),
                (__ubuf__ int32_t*)cursor_matrix_ub.GetPhyAddr(),
                (__ubuf__ int32_t*)tpe_tile_ub.GetPhyAddr(),
                (__ubuf__ int32_t*)thread_prefix_ub.GetPhyAddr(),
                (__ubuf__ int32_t*)expert_carry_ub.GetPhyAddr(),
                route_core_num,
                len,
                cursor_row_bytes / kInt32Bytes,
                thread_count);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopyExtParams matrix_store_params(
                static_cast<uint16_t>(route_core_num),
                static_cast<uint32_t>(active_row_bytes),
                static_cast<uint32_t>(
                    (cursor_row_bytes - aligned_active_row_bytes) / 32),
                static_cast<uint32_t>((E - len) * kInt32Bytes),
                0);
            AscendC::DataCopyPad(
                core_expert_counts_gm[base],
                cursor_matrix_ub,
                matrix_store_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    AscendC::SyncAll<true>();
    const int32_t expert_row_bytes = AlignUpBytes(E * kInt32Bytes, 32);
    AscendC::LocalTensor<int32_t> expert_ordinal_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(0);
    if (core_idx < route_core_num) {
        static_assert((kFillTileElems * kInt32Bytes) % 32 == 0);
        constexpr int32_t kOrdinalScratchBytes =
            kFillTileElems * kInt32Bytes;
        AscendC::LocalTensor<int32_t> current_core_count_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(
                static_cast<uint32_t>(expert_row_bytes));
        AscendC::LocalTensor<int32_t> core_zero_count_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(
                static_cast<uint32_t>(
                    expert_row_bytes + kOrdinalScratchBytes));
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int32_t base = 0; base < E; base += kFillTileElems) {
            const int32_t len = IntMin(kFillTileElems, E - base);
            AscendC::DataCopyExtParams copy_params(
                1,
                static_cast<uint32_t>(len * kInt32Bytes),
                0,
                0,
                0);
            AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
            AscendC::DataCopyPad(
                expert_ordinal_ub[base],
                tpe_prefix_gm[base],
                copy_params,
                pad_params);
            AscendC::DataCopyPad(
                current_core_count_ub,
                core_expert_counts_gm[core_idx * E + base],
                copy_params,
                pad_params);
            AscendC::DataCopyPad(
                core_zero_count_ub,
                core_expert_counts_gm[base],
                copy_params,
                pad_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            AscendC::Sub(
                current_core_count_ub,
                current_core_count_ub,
                core_zero_count_ub,
                len);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(
                expert_ordinal_ub[base],
                expert_ordinal_ub[base],
                current_core_count_ub,
                len);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        DstUbLayout dst_layout{};
        InitDstUbLayout(R, E, expert_row_bytes, dst_layout);
        AscendC::LocalTensor<int32_t> dst_tile_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(
                static_cast<uint32_t>(dst_layout.dst_tile_offset));

        if (dst_layout.use_simt) {
            AscendC::LocalTensor<int32_t> alloc_cumsum_lookup_ub =
                resource.ubBuf.template GetBufferByByte<int32_t>(
                    static_cast<uint32_t>(dst_layout.lookup_offset));
            AscendC::LocalTensor<int32_t> expert_offsets_lookup_ub =
                resource.ubBuf.template GetBufferByByte<int32_t>(
                    static_cast<uint32_t>(
                        dst_layout.lookup_offset +
                        dst_layout.lookup_table_bytes));
            {
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopyExtParams copy_params(
                    1,
                    static_cast<uint32_t>(
                        dst_layout.lookup_table_elems * kInt32Bytes),
                    0,
                    0,
                    0);
                AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
                AscendC::DataCopyPad(alloc_cumsum_lookup_ub, alloc_cumsum_gm[0], copy_params, pad_params);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            }
            {
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopyExtParams copy_params(
                    1,
                    static_cast<uint32_t>(
                        dst_layout.lookup_table_elems * kInt32Bytes),
                    0,
                    0,
                    0);
                AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
                AscendC::DataCopyPad(expert_offsets_lookup_ub, expert_offsets_gm[0], copy_params, pad_params);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            }
            {
                AscendC::LocalTensor<int32_t> dst_simt_context_ub =
                    resource.ubBuf.template GetBufferByByte<int32_t>(
                        static_cast<uint32_t>(dst_layout.context_offset));
                AscendC::LocalTensor<int32_t> dst_simt_topk_ub =
                    resource.ubBuf.template GetBufferByByte<int32_t>(
                        static_cast<uint32_t>(dst_layout.simt_topk_offset));
                AscendC::LocalTensor<int32_t> dst_simt_hist_ub =
                    resource.ubBuf.template GetBufferByByte<int32_t>(
                        static_cast<uint32_t>(dst_layout.simt_hist_offset));

                dst_simt_context_ub.SetValue(
                    kDstSimtTopkOffset,
                    dst_layout.simt_topk_offset / kInt32Bytes);
                dst_simt_context_ub.SetValue(
                    kDstSimtHistOffset,
                    dst_layout.simt_hist_offset / kInt32Bytes);
                dst_simt_context_ub.SetValue(
                    kDstSimtOrdinalOffset,
                    0);
                dst_simt_context_ub.SetValue(
                    kDstSimtAllocCumsumOffset,
                    dst_layout.lookup_offset / kInt32Bytes);
                dst_simt_context_ub.SetValue(
                    kDstSimtExpertOffsetsOffset,
                    (dst_layout.lookup_offset +
                     dst_layout.lookup_table_bytes) /
                        kInt32Bytes);
                dst_simt_context_ub.SetValue(
                    kDstSimtDstOffset,
                    dst_layout.dst_tile_offset / kInt32Bytes);
                dst_simt_context_ub.SetValue(
                    kDstSimtChunkSize,
                    dst_layout.simt_chunk_size);
                dst_simt_context_ub.SetValue(
                    kDstSimtHistStride,
                    dst_layout.simt_hist_stride);
                dst_simt_context_ub.SetValue(kDstSimtRankCount, R);
                dst_simt_context_ub.SetValue(kDstSimtExpertCount, E);
                dst_simt_context_ub.SetValue(kDstSimtMaxTokensPerRank, NvS);

                AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);

                // MTE2 reuses dst_simt_topk_ub on every tile. Keep the next
                // load behind the preceding SIMT reader of that same buffer.
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                for (int32_t base = begin; base < end; base += kFillTileElems) {
                    const int32_t len = IntMin(kFillTileElems, end - base);
                    const int32_t chunk_count =
                        (len + dst_layout.simt_chunk_size - 1) /
                        dst_layout.simt_chunk_size;
                    const int32_t hist_elems =
                        chunk_count * dst_layout.simt_hist_stride;

                    for (int32_t clear_base = 0; clear_base < hist_elems;
                         clear_base += kMaxDuplicateInt32Elems) {
                        const int32_t clear_len = IntMin(
                            kMaxDuplicateInt32Elems,
                            hist_elems - clear_base);
                        AscendC::Duplicate(
                            dst_simt_hist_ub[clear_base],
                            static_cast<int32_t>(0),
                            clear_len);
                    }
                    AscendC::PipeBarrier<PIPE_V>();

                    AscendC::DataCopyExtParams copy_params(
                        1,
                        static_cast<uint32_t>(len * kInt32Bytes),
                        0,
                        0,
                        0);
                    AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                    AscendC::DataCopyPad(
                        dst_simt_topk_ub,
                        topk_gm[base],
                        copy_params,
                        pad_params);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

                    asc_vf_call<BuildDstTileSimt>(
                        dim3(static_cast<uint32_t>(kDstSimtMaxThreads)),
                        (__ubuf__ int32_t*)work_ub.GetPhyAddr(),
                        dst_layout.context_offset / kInt32Bytes,
                        base,
                        len);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

                    const int32_t total_row =
                        (chunk_count - 1) * dst_layout.simt_hist_stride;
                    AscendC::Add(
                        expert_ordinal_ub,
                        expert_ordinal_ub,
                        dst_simt_hist_ub[total_row],
                        E);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

                    AscendC::DataCopyPad(
                        dst_gm[base],
                        dst_tile_ub,
                        copy_params);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                }
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            }
        } else {
            // Host validation guarantees that the ordinal row and both route
            // tiles fit when the complete lookup-SIMT layout does not.
            AscendC::LocalTensor<int32_t> dst_gm_simt_topk_ub =
                resource.ubBuf.template GetBufferByByte<int32_t>(
                    static_cast<uint32_t>(
                        dst_layout.gm_simt_topk_offset));

            // alloc_cumsum and expert_offsets are produced by multiple cores.
            // The preceding SyncAll joins those producers, while this fence
            // makes their GM stores visible before SIMT starts direct loads.
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
            CleanInvalidDataCache();

            for (int32_t base = begin; base < end;
                 base += kFillTileElems) {
                const int32_t len = IntMin(kFillTileElems, end - base);
                AscendC::DataCopyExtParams copy_params(
                    1,
                    static_cast<uint32_t>(len * kInt32Bytes),
                    0,
                    0,
                    0);
                AscendC::DataCopyPadExtParams<int32_t> pad_params{
                    false, 0, 0, 0};
                AscendC::DataCopyPad(
                    dst_gm_simt_topk_ub,
                    topk_gm[base],
                    copy_params,
                    pad_params);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

                asc_vf_call<BuildDstTileFromGmSimt>(
                    dim3(static_cast<uint32_t>(kDstSimtMaxThreads)),
                    (__ubuf__ int32_t*)dst_gm_simt_topk_ub.GetPhyAddr(),
                    (__ubuf__ int32_t*)expert_ordinal_ub.GetPhyAddr(),
                    (__ubuf__ int32_t*)dst_tile_ub.GetPhyAddr(),
                    reinterpret_cast<__gm__ int32_t*>(alloc_cumsum),
                    reinterpret_cast<__gm__ int32_t*>(expert_offsets),
                    len,
                    R,
                    E,
                    NvS);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

                AscendC::DataCopyPad(
                    dst_gm[base],
                    dst_tile_ub,
                    copy_params);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();

    // Source-dup immediately reads back dst through MTE2. In the aligned
    // partition case each core consumes only the slice it produced, so a DDR
    // visibility barrier is sufficient and avoids an unnecessary grid sync.
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();

    // When tokens divide evenly across cores, dst production and source-dup
    // consumption use the same per-core route interval. Each core can consume
    // its own completed dst slice immediately; uneven partitions retain the
    // grid barrier because their route and token boundaries may differ.
    const bool source_dup_is_core_local = (S % core_num) == 0;
    if (!source_dup_is_core_local) {
        AscendC::SyncAll<true>();
    }
    // Source-side duplicate stages use one shared layout. Cross-core and
    // cross-rank synchronization remains explicit between the phase calls.
    {
        const SourceDupConfig source_dup_config{
            rank,
            R,
            S,
            K,
            NvS,
            core_idx,
            core_num,
            source_dup_tokens_per_tile};
        SourceDupPartition source_dup_partition{};
        SourceDupUbLayout source_dup_ub_layout{};
        SourceDupWorkspaceLayout source_dup_workspace_layout{};
        InitSourceDupPartition(source_dup_config, source_dup_partition);
        InitSourceDupUbLayout(source_dup_config, source_dup_ub_layout);
        InitSourceDupWorkspaceLayout(
            source_dup_config,
            source_dup_partition,
            source_dup_ub_layout,
            source_dup_workspace_layout);

        CountAndPrebuildSourceDupTiles(
            source_dup_config,
            source_dup_partition,
            source_dup_ub_layout,
            source_dup_workspace_layout,
            resource,
            dst_gm,
            dup_workspace_gm,
            dup_key_workspace_gm);

        // Per-core counts are local. Finish all rows before reducing and
        // actively publishing one compact count slot to each destination.
        AscendC::SyncAll<true>();
        PublishSourceDupCounts(
            source_dup_config,
            resource,
            shmem_base,
            dup_workspace_gm,
            dup_count_inbox_off);

        // Each destination waits on its local inbox, computes source-rank
        // prefixes, and stages them for a parallel return to the sources.
        if (core_idx == 0) {
            BuildSourceDupDestinationPrefixes(
                source_dup_config,
                resource,
                local_shmem_gm,
                dup_key_workspace_gm,
                dup_counts_gm,
                dup_count_inbox_off);
        }
        AscendC::SyncAll<true>();
        PublishSourceDupPrefixes(
            source_dup_config,
            resource,
            shmem_base,
            dup_key_workspace_gm,
            dup_prefix_inbox_off);

        // Prefix senders read the staging table above. Do not reuse it for the
        // per-core prefix table until every local sender has completed MTE3.
        AscendC::SyncAll<true>();
        BuildSourceDupCorePrefixes(
            source_dup_config,
            resource,
            local_shmem_gm,
            dup_workspace_gm,
            dup_key_workspace_gm,
            dup_prefix_inbox_off);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        AscendC::SyncAll<true>();

        PublishSourceDupPayloads(
            source_dup_config,
            source_dup_partition,
            source_dup_ub_layout,
            source_dup_workspace_layout,
            resource,
            shmem_base,
            dup_workspace_gm,
            dup_key_workspace_gm,
            dup_payload_done_flag_off,
            dup_groups_scratch_off,
            dup_loffs_scratch_off);

        WaitForSourceDupPayloadDoneFlags(
            source_dup_config,
            resource,
            local_shmem_gm,
            dup_payload_done_flag_off);
        AscendC::SyncAll<true>();

        MaterializeSourceDupOutputs(
            source_dup_config,
            source_dup_ub_layout,
            resource,
            local_shmem_gm,
            dup_counts_gm,
            dup_groups_gm,
            dup_loffs_gm,
            dup_groups_scratch_off,
            dup_loffs_scratch_off);
    }

}

extern "C" ASCEND_MOONEP_EXPORT void planning_kernel_do(
    void* stream,
    uint8_t* topk_idx,
    uint8_t* symmetric_buffer,
    uint8_t* dst,
    uint8_t* cu_seqlens,
    uint8_t* experts_to_copy,
    uint8_t* dup_groups,
    uint8_t* dup_loffs,
    uint8_t* dup_counts,
    uint8_t* tpe_all,
    uint8_t* tpe_prefix,
    uint8_t* balance,
    uint8_t* z_tensor,
    uint8_t* alloc_tensor,
    uint8_t* alloc_cumsum,
    uint8_t* expert_offsets,
    uint8_t* core_expert_counts,
    uint8_t* dup_workspace,
    uint8_t* dup_key_workspace,
    uint8_t* tiling_host,
    uint64_t ffts_addr) {
    auto* td = reinterpret_cast<ascend_moonep::PlanningTilingData*>(tiling_host);
    planning_kernel<<<td->block_dim, kPlanningUbSizeBytes, stream>>>(
        topk_idx,
        symmetric_buffer,
        dst,
        cu_seqlens,
        experts_to_copy,
        dup_groups,
        dup_loffs,
        dup_counts,
        tpe_all,
        tpe_prefix,
        balance,
        z_tensor,
        alloc_tensor,
        alloc_cumsum,
        expert_offsets,
        core_expert_counts,
        dup_workspace,
        dup_key_workspace,
        *td,
        ffts_addr);
}
