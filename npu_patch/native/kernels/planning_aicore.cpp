// ============================================================================
// planning kernel - A2/A3 backend: Ascend910B / Ascend910_93 / 2201 / non-SIMT
// ============================================================================
// This is the non-SIMT-family backend of the planning kernel. Everything
// shared with the SIMT (A5) backend lives in planning_common.h; this file
// keeps only the non-SIMT pieces: the *A2 kernels, the orchestration helpers
// whose bodies diverge from A5, and the planning_kernel / planning_kernel_do
// entry points.
//
// Selected at build time by native/CMakeLists.txt (non-SIMT). It and
// planning_simt.cpp are never compiled into the same target, so both export
// the same planning_kernel_do symbol.
//
// Reading map:
//   Source-duplicate exchange helpers: inbox/count/prefix/payload protocol.
//   Local compute helpers: scalar UB algorithms and A2/A3 vector reductions.
//   planning_kernel: numbered stages 1-10, from histograms to duplicate output.
//
// Synchronization convention for local compute helpers below:
//   Raw __ubuf__ pointer helpers operate on PIPE_S and issue no events unless
//   documented otherwise. The caller must finish producers (MTE2_S / V_S)
//   before scalar reads, and finish consumers before overwriting their inputs.
//   Before a later vector or MTE3 consumer, the caller supplies S_V / S_MTE3.
//   LocalTensor helpers document their internal events at their definitions.
//   These pipe events do not replace cross-core or cross-rank synchronization;
//   the orchestration code owns SyncAll and the SHMEM inbox/barrier protocol.

#include "planning_common.h"

using namespace AscendC;
using namespace ascend_moonep::planning;

namespace {

static_assert(kPlanningUbSizeBytes <= Catlass::Arch::AtlasA2::UB_SIZE);

struct PlanningArch : Catlass::Arch::AtlasA2 {
    static constexpr uint32_t UB_SIZE = kPlanningUbSizeBytes;
};

using PlanningResource = Catlass::Arch::Resource<PlanningArch>;

__aicore__ inline void PackSourceDupByDestA2(
    __ubuf__ int32_t* compact_groups,
    __ubuf__ int32_t* compact_loffs,
    __ubuf__ int32_t* dest_meta,
    __ubuf__ int32_t* packed_groups,
    __ubuf__ int32_t* packed_loffs,
    int32_t group_count,
    int32_t loff_count);

__aicore__ inline void AdjustSourceDupGroupLoffsA2(
    __ubuf__ int32_t* groups,
    __ubuf__ int32_t* tile_counts,
    __ubuf__ int32_t* dest_meta,
    int32_t rank_count);

__aicore__ inline void PackSourceDupCountSlotsA2(
    __ubuf__ int32_t* core_counts,
    __ubuf__ int32_t* count_slots,
    int32_t core_count,
    int32_t dest_count,
    int32_t core_row_elems);

__aicore__ inline void BuildSourceDupDestinationPrefixesA2(
    __ubuf__ int32_t* count_slots,
    __ubuf__ int32_t* totals,
    int32_t rank_count);

__aicore__ inline void BuildSourceDupCorePrefixesA2(
    __ubuf__ int32_t* core_counts,
    __ubuf__ int32_t* rank_prefix_slots,
    int32_t core_count,
    int32_t dest_count,
    int32_t core_row_elems);

__aicore__ inline void DedupRouteTileA2(
    __ubuf__ int32_t* dst,
    __ubuf__ int32_t* compact_groups,
    __ubuf__ int32_t* compact_loffs,
    __ubuf__ int32_t* tile_counts,
    int32_t tile_tokens,
    int32_t topk,
    int32_t max_tokens_per_rank,
    int32_t& group_count,
    int32_t& loff_count);


__aicore__ inline void DedupRouteTileMaskA2(
    __ubuf__ int32_t* dst,
    __ubuf__ int32_t* compact_groups,
    __ubuf__ int32_t* compact_loffs,
    __ubuf__ int32_t* tile_counts,
    __ubuf__ int32_t* rank_scratch,
    __ubuf__ int32_t* first_route_destinations,
    int32_t tile_tokens,
    int32_t topk,
    int32_t rank_count,
    int32_t max_tokens_per_rank,
    int32_t& group_count,
    int32_t& loff_count);


// Partition by 32-byte data blocks so concurrent cores never update different
// parts of the same GM block. Only the final owner may issue a short tail.
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
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        bool ready = true;
        for (int32_t index = 0; index < inbox_elems; ++index) {
            if (inbox_ub.GetValue(index) < 0) {
                ready = false;
                break;
            }
        }
        if (ready) {
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
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        bool ready = true;
        for (int32_t index = 0; index < rank_count; ++index) {
            if (flag_ub.GetValue(index) < 1) {
                ready = false;
                break;
            }
        }
        if (ready) {
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
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    PackSourceDupCountSlotsA2(
        (__ubuf__ int32_t*)core_counts_ub.GetPhyAddr(),
        (__ubuf__ int32_t*)count_slots_ub.GetPhyAddr(),
        config.core_num,
        dest_count,
        layout.row_elems);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);

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

    BuildSourceDupDestinationPrefixesA2(
        (__ubuf__ int32_t*)count_slots_ub.GetPhyAddr(),
        (__ubuf__ int32_t*)totals_ub.GetPhyAddr(),
        config.rank_count);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
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
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    BuildSourceDupCorePrefixesA2(
        (__ubuf__ int32_t*)core_counts_ub.GetPhyAddr(),
        (__ubuf__ int32_t*)prefix_slots_ub.GetPhyAddr(),
        config.core_num,
        dest_count,
        layout.row_elems);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
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
    AscendC::LocalTensor<int32_t> compact_loff_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.compact_loff_offset));
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
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    int32_t raw_group_count = 0;
    int32_t raw_loff_count = 0;
    if (config.topk >= 8 && config.topk <= 32) {
        DedupRouteTileMaskA2(
            (__ubuf__ int32_t*)dst_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)group_send_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)compact_loff_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)tile_count_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)meta_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)raw_group_ub.GetPhyAddr(),
            tile_tokens, config.topk, config.rank_count,
            config.max_tokens_per_rank, raw_group_count, raw_loff_count);
    } else {
        // Small K avoids bucket setup; K > 32 exceeds the bit-mask width.
        DedupRouteTileA2(
            (__ubuf__ int32_t*)dst_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)group_send_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)compact_loff_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)tile_count_ub.GetPhyAddr(),
            tile_tokens,
            config.topk,
            config.max_tokens_per_rank,
            raw_group_count,
            raw_loff_count);
    }
    AscendC::PipeBarrier<PIPE_ALL>();

    const bool tile_has_duplicates = raw_group_count > 0;
    if (tile_has_duplicates) {
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
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
        PackSourceDupByDestA2(
            (__ubuf__ int32_t*)group_send_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)compact_loff_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)meta_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)raw_group_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)raw_loff_ub.GetPhyAddr(),
            raw_group_count,
            raw_loff_count);
        AscendC::PipeBarrier<PIPE_ALL>();
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

        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AdjustSourceDupGroupLoffsA2(
            (__ubuf__ int32_t*)group_send_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)tile_count_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)meta_ub.GetPhyAddr(),
            config.rank_count);
        AscendC::PipeBarrier<PIPE_ALL>();
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
    // Payload publication must be complete on every rank before any core raises
    // a done flag: a peer that observes the flag is allowed to materialize the
    // payload bytes. mte_quiet drains this core's remote stores (PIPE_ALL +
    // dcci); the cross-rank barrier turns "every rank published" into a global
    // condition, which a local SyncAll cannot express because the flags travel
    // from other cores on their own queues.
    aclshmemx_mte_quiet();
    aclshmemx_barrier_all_vec();
    PublishSourceDupPayloadDoneFlags(
        config,
        shmem_base,
        count_ub,
        payload_done_flag_off);
}

// Keep the source-duplicate exchange explicit in this architecture block.
// Its phase order follows the Planning protocol for the selected backend.
struct SourceDupStageArgs {
    SourceDupConfig config;
    int32_t count_inbox_offset;
    int32_t prefix_inbox_offset;
    int32_t payload_done_flag_offset;
    int32_t groups_scratch_offset;
    int32_t loffs_scratch_offset;
    PlanningResource* resource;
    AscendC::GlobalTensor<int32_t>* dst_gm;
    AscendC::GlobalTensor<int32_t>* dup_workspace_gm;
    AscendC::GlobalTensor<int32_t>* dup_key_workspace_gm;
    AscendC::GlobalTensor<int32_t>* local_shmem_gm;
    AscendC::GlobalTensor<int32_t>* dup_counts_gm;
    AscendC::GlobalTensor<int32_t>* dup_groups_gm;
    AscendC::GlobalTensor<int32_t>* dup_loffs_gm;
    __gm__ int32_t* shmem_base;
};

// Collective entry: all participating planning cores execute this protocol in
// the same order. The caller finishes its local dst writes/UB users first.
// This helper starts with a DDR fence and, for uneven token partitions, joins
// dst producers with SyncAll before repartitioned readers consume dst. Later
// count/prefix/payload phases own their grid barriers and inbox waits below.
__aicore__ inline void RunSourceDupStages(
    const SourceDupStageArgs& args) {
    auto& resource = *args.resource;
    auto& dst_gm = *args.dst_gm;
    auto& dup_workspace_gm = *args.dup_workspace_gm;
    auto& dup_key_workspace_gm = *args.dup_key_workspace_gm;
    auto& local_shmem_gm = *args.local_shmem_gm;
    auto& dup_counts_gm = *args.dup_counts_gm;
    auto& dup_groups_gm = *args.dup_groups_gm;
    auto& dup_loffs_gm = *args.dup_loffs_gm;
    const auto& config = args.config;

    // Source-dup immediately reads back dst through MTE2. Make the preceding
    // destination stores visible before deciding whether a grid barrier can
    // be skipped for an even per-core token partition.
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    const bool source_dup_is_core_local =
        (config.token_count % config.core_num) == 0;
    if (!source_dup_is_core_local) {
        AscendC::SyncAll<true>();
    }

    SourceDupPartition partition{};
    SourceDupUbLayout ub_layout{};
    SourceDupWorkspaceLayout workspace_layout{};
    InitSourceDupPartition(config, partition);
    InitSourceDupUbLayout(config, ub_layout);
    InitSourceDupWorkspaceLayout(
        config,
        partition,
        ub_layout,
        workspace_layout);

    CountAndPrebuildSourceDupTiles(
        config,
        partition,
        ub_layout,
        workspace_layout,
        resource,
        dst_gm,
        dup_workspace_gm,
        dup_key_workspace_gm);

    // Per-core counts are local. Finish all rows before reducing and
    // actively publishing one compact count slot to each destination.
    AscendC::SyncAll<true>();
    PublishSourceDupCounts(
        config,
        resource,
        args.shmem_base,
        dup_workspace_gm,
        args.count_inbox_offset);

    // Each destination waits on its local inbox, computes source-rank
    // prefixes, and stages them for a parallel return to the sources.
    if (config.core_idx == 0) {
        BuildSourceDupDestinationPrefixes(
            config,
            resource,
            local_shmem_gm,
            dup_key_workspace_gm,
            dup_counts_gm,
            args.count_inbox_offset);
    }
    AscendC::SyncAll<true>();
    PublishSourceDupPrefixes(
        config,
        resource,
        args.shmem_base,
        dup_key_workspace_gm,
        args.prefix_inbox_offset);

    // Prefix senders read the staging table above. Do not reuse it for the
    // per-core prefix table until every local sender has completed MTE3.
    AscendC::SyncAll<true>();
    BuildSourceDupCorePrefixes(
        config,
        resource,
        local_shmem_gm,
        dup_workspace_gm,
        dup_key_workspace_gm,
        args.prefix_inbox_offset);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    AscendC::SyncAll<true>();

    PublishSourceDupPayloads(
        config,
        partition,
        ub_layout,
        workspace_layout,
        resource,
        args.shmem_base,
        dup_workspace_gm,
        dup_key_workspace_gm,
        args.payload_done_flag_offset,
        args.groups_scratch_offset,
        args.loffs_scratch_offset);

    WaitForSourceDupPayloadDoneFlags(
        config,
        resource,
        local_shmem_gm,
        args.payload_done_flag_offset);
    AscendC::SyncAll<true>();
    aclshmemx_mte_quiet();

    MaterializeSourceDupOutputs(
        config,
        ub_layout,
        resource,
        local_shmem_gm,
        dup_counts_gm,
        dup_groups_gm,
        dup_loffs_gm,
        args.groups_scratch_offset,
        args.loffs_scratch_offset);
    aclshmemx_mte_quiet();
    aclshmemx_barrier_all_vec();
}

}  // namespace

namespace {
// Caller: aligned source rows are staged, output storage is no longer in use,
// and EVENT_ID0 is available for this helper's MTE2_V and V_S pairs.
// Helper: waits for MTE2 inputs, orders vector accumulation, and returns with
// vector output visible to PIPE_S. Caller owns any subsequent store/reuse fence.
__aicore__ inline void ReduceTpeRowsA2(
    AscendC::LocalTensor<int32_t>& dst,
    AscendC::LocalTensor<int32_t>& src,
    uint32_t row_count,
    uint32_t column_count,
    uint32_t row_stride_elems) {
    // All callers stage aligned rows. Sum across rows using the A3 vector
    // pipeline instead of scalar, column-major UB loads. Exact src/dst alias
    // is supported by starting from row zero rather than zeroing the output.
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    if (row_count == 0) {
        AscendC::Duplicate(dst, int32_t{0}, column_count);
    } else {
        AscendC::Adds(dst, src, int32_t{0}, column_count);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t row = 1; row < row_count; ++row) {
            AscendC::Add(dst, dst, src[row * row_stride_elems], column_count);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
}

// Same entry contract as ReduceTpeRowsA2; it supplies the vector-to-scalar wait.
// total_dst is then written by PIPE_S. Caller orders its MTE3 consumers.
__aicore__ inline void ReduceTpeOwnerRowsA2(
    AscendC::LocalTensor<int32_t>& dst,
    AscendC::LocalTensor<int32_t>& total_dst,
    AscendC::LocalTensor<int32_t>& src,
    uint32_t row_count,
    uint32_t column_count,
    uint32_t row_stride_elems) {
    ReduceTpeRowsA2(dst, src, row_count, column_count, row_stride_elems);
    int32_t tile_total = 0;
    for (uint32_t column = 0; column < column_count; ++column) {
        tile_total += dst.GetValue(column);
    }
    total_dst.SetValue(0, tile_total);
}

// Scalar-only: caller makes routes/counts visible to PIPE_S and initializes
// expert_counts before the first tile; subsequent tiles accumulate in place.
__aicore__ inline void BuildRouteHistogramA2(
    __ubuf__ int32_t* routes,
    __ubuf__ int32_t* expert_counts,
    int32_t route_count) {
    for (int32_t route = 0; route < route_count; ++route) {
        const int32_t expert = routes[route];
        expert_counts[expert] = expert_counts[expert] + 1;
    }
}

// Scalar-only, in-place inclusive scan over destination ranks for each expert.
// Caller must finish MTE3 readers of alloc before replacing counts with prefixes.
__aicore__ inline void BuildAllocCumsumA2(
    __ubuf__ int32_t* alloc,
    int32_t rank_count,
    int32_t experts_per_rank,
    int32_t alloc_row_elems) {
    for (int32_t local_expert = 0;
         local_expert < experts_per_rank;
         ++local_expert) {
        int32_t prefix = 0;
        for (int32_t dest = 0; dest < rank_count; ++dest) {
            const int32_t offset = dest * alloc_row_elems + local_expert;
            prefix += alloc[offset];
            alloc[offset] = prefix;
        }
    }
}

// Scalar-only; caller supplies an E+B count row and B selected-expert slots.
// Entry: the first E counts are ready for PIPE_S. Exit: the trailing B slots
// hold selected counts; only the final selected experts have their E slots zeroed.
__aicore__ inline void SelectPrefetchExpertsA2(
    __ubuf__ int32_t* group_counts,
    __ubuf__ int32_t* selected_experts,
    int32_t expert_count,
    int32_t prefetch_count,
    int32_t local_expert_begin,
    int32_t local_expert_end) {
    for (int32_t slot = 0; slot < prefetch_count; ++slot) {
        selected_experts[slot] = -1;
        group_counts[expert_count + slot] = 0;
    }
    int32_t selected_count = 0;
    for (int32_t expert = 0; expert < expert_count; ++expert) {
        if (expert >= local_expert_begin && expert < local_expert_end) {
            continue;
        }
        const int32_t count = group_counts[expert];
        if (count <= 0) {
            continue;
        }
        int32_t pos = selected_count < prefetch_count
            ? selected_count : prefetch_count;
        while (pos > 0) {
            const int32_t previous_count = group_counts[expert_count + pos - 1];
            // Experts are visited in ascending order: ties prefer the new one.
            if (previous_count > count) {
                break;
            }
            if (pos < prefetch_count) {
                selected_experts[pos] = selected_experts[pos - 1];
                group_counts[expert_count + pos] = previous_count;
            }
            --pos;
        }
        if (pos < prefetch_count) {
            selected_experts[pos] = expert;
            group_counts[expert_count + pos] = count;
            if (selected_count < prefetch_count) {
                ++selected_count;
            }
        }
    }
    // Defer clearing until selection is complete: an evicted candidate keeps
    // its original expert slot, exactly as in repeated max selection.
    for (int32_t slot = 0; slot < selected_count; ++slot) {
        group_counts[selected_experts[slot]] = 0;
    }
}

// Assign complete 32-byte-aligned groups of packed prefetch rows to each core.
// L = 8 / gcd(prefetch_count, 8) rows span whole GM blocks. Since E = R * B,
// the same boundaries also align the expert-offset rows. No UB is allocated.
// This helper only partitions work: callers retain their MTE3 completion and
// UB-reuse waits, including between short rows owned by the same core.
__aicore__ inline void PartitionPrefetchRowsA2(
    int32_t rank_count,
    int32_t prefetch_count,
    int32_t core_idx,
    int32_t core_count,
    int32_t& begin,
    int32_t& end,
    int32_t& step) {
    if ((prefetch_count & 7) == 0) {
        begin = core_idx;
        end = rank_count;
        step = core_count;
        return;
    }
    const int32_t ranks_per_group = (prefetch_count & 3) == 0 ? 2
        : (prefetch_count & 1) == 0 ? 4 : 8;
    const int32_t group_count =
        (rank_count + ranks_per_group - 1) / ranks_per_group;
    const int32_t groups_per_core = group_count / core_count;
    const int32_t tail_cores = group_count - groups_per_core * core_count;
    const int32_t group_begin = core_idx * groups_per_core +
        (core_idx < tail_cores ? core_idx : tail_cores);
    const int32_t assigned_groups =
        groups_per_core + (core_idx < tail_cores ? 1 : 0);
    begin = group_begin * ranks_per_group;
    const int32_t row_end = (group_begin + assigned_groups) * ranks_per_group;
    end = row_end < rank_count ? row_end : rank_count;
    step = 1;
}

__aicore__ inline void BuildGroupPrefixA2(
    __ubuf__ int32_t* group_counts,
    int32_t group_count) {
    int32_t prefix = 0;
    for (int32_t group = 0; group < group_count; ++group) {
        prefix += group_counts[group];
        group_counts[group] = prefix;
    }
}

__aicore__ inline int32_t ReduceExpertTileTotalA2(
    __ubuf__ int32_t* expert_counts,
    int32_t expert_count) {
    int32_t total = 0;
    for (int32_t expert = 0; expert < expert_count; ++expert) {
        total += expert_counts[expert];
    }
    return total;
}

__aicore__ inline void AdjustExpertOrdinalsA2(
    __ubuf__ int32_t* expert_ordinal,
    __ubuf__ int32_t* current_core_count,
    __ubuf__ int32_t* core_zero_count,
    int32_t expert_count) {
    for (int32_t expert = 0; expert < expert_count; ++expert) {
        expert_ordinal[expert] +=
            current_core_count[expert] - core_zero_count[expert];
    }
}

// Caller: core_counts/tpe_counts are loaded and expert_carry is initialized;
// their MTE2/vector producers must complete before the helper's scalar reads,
// and previous MTE3 readers must finish before these buffers are overwritten.
// Helper: owns S_V and V_S EVENT_ID0 pairs; returns scalar-visible vector output.
// UB: core_counts[core_count, row_stride] becomes exclusive core prefixes;
// tpe_counts[expert_count] is overwritten with expert bases; expert_carry[0]
// advances across tiles. Caller supplies the subsequent S_MTE3 store fence.
__aicore__ inline void BuildCoreExpertCursorsA2(
    AscendC::LocalTensor<int32_t>& core_counts,
    AscendC::LocalTensor<int32_t>& tpe_counts,
    AscendC::LocalTensor<int32_t>& expert_carry,
    int32_t core_count,
    int32_t expert_count,
    int32_t row_stride) {
    // Build per-expert bases once. Scan contiguous core rows with vector Add,
    // then shift backwards to exclusive prefixes without another UB matrix.
    // The caller reloads tpe_counts on the next tile; it is scratch here.
    int32_t expert_base = expert_carry.GetValue(0);
    for (int32_t expert = 0; expert < expert_count; ++expert) {
        const int32_t count = tpe_counts.GetValue(expert);
        tpe_counts.SetValue(expert, expert_base);
        expert_base += count;
    }
    expert_carry.SetValue(0, expert_base);
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    for (int32_t row = 1; row < core_count; ++row) {
        AscendC::Add(core_counts[row * row_stride], core_counts[row * row_stride],
            core_counts[(row - 1) * row_stride], expert_count);
        AscendC::PipeBarrier<PIPE_V>();
    }
    for (int32_t row = core_count - 1; row > 0; --row) {
        AscendC::Add(core_counts[row * row_stride], core_counts[(row - 1) * row_stride],
            tpe_counts, expert_count);
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::Adds(core_counts, tpe_counts, int32_t{0}, expert_count);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
}

__aicore__ inline void ConvertGroupPrefixToExpertOffsetsA2(
    __ubuf__ int32_t* group_prefix,
    __ubuf__ int32_t* selected_experts,
    __ubuf__ int32_t* selected_offsets,
    int32_t expert_count,
    int32_t prefetch_count) {
    for (int32_t slot = 0; slot < prefetch_count; ++slot) {
        const int32_t group = expert_count + slot;
        const int32_t begin = group == 0 ? 0 : group_prefix[group - 1];
        const int32_t end = group_prefix[group];
        selected_offsets[slot] = end > begin ? begin : -1;
    }
    for (int32_t expert = expert_count - 1; expert >= 0; --expert) {
        const int32_t begin = expert == 0 ? 0 : group_prefix[expert - 1];
        const int32_t end = group_prefix[expert];
        group_prefix[expert] = end > begin ? begin : -1;
    }
    for (int32_t slot = 0; slot < prefetch_count; ++slot) {
        const int32_t expert = selected_experts[slot];
        const int32_t offset = selected_offsets[slot];
        if (expert >= 0 && offset >= 0) {
            group_prefix[expert] = offset;
        }
    }
}

__aicore__ inline void DedupRouteTileA2(
    __ubuf__ int32_t* dst,
    __ubuf__ int32_t* compact_groups,
    __ubuf__ int32_t* compact_loffs,
    __ubuf__ int32_t* tile_counts,
    int32_t tile_tokens,
    int32_t topk,
    int32_t max_tokens_per_rank,
    int32_t& group_count,
    int32_t& loff_count) {
    // Tokens are serial on A2/A3: append in token/route order directly,
    // avoiding per-token slots, a prefix pass and a compaction pass.
    // Keep the shared UB layout and downstream compact format unchanged.
    group_count = 0;
    loff_count = 0;
    for (int32_t token = 0; token < tile_tokens; ++token) {
        const int32_t route_base = token * topk;
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
            for (int32_t next_kidx = kidx + 1;
                 next_kidx < topk;
                 ++next_kidx) {
                const int32_t next_raw = dst[route_base + next_kidx];
                if (next_raw < dest_base || next_raw >= dest_end) {
                    continue;
                }
                compact_loffs[loff_count] = next_raw - dest_base;
                ++loff_count;
                ++duplicate_count;
                dst[route_base + next_kidx] = -next_raw - 1;
            }
            if (duplicate_count > 0) {
                const int32_t group_pos = group_count * 3;
                compact_groups[group_pos] = dest;
                compact_groups[group_pos + 1] = raw - dest_base;
                compact_groups[group_pos + 2] = group_loff_start;
                tile_counts[dest * 2] = tile_counts[dest * 2] + 1;
                tile_counts[dest * 2 + 1] =
                    tile_counts[dest * 2 + 1] + duplicate_count;
                ++group_count;
            }
        }
    }
}


__aicore__ inline void DedupRouteTileMaskA2(
    __ubuf__ int32_t* dst,
    __ubuf__ int32_t* compact_groups,
    __ubuf__ int32_t* compact_loffs,
    __ubuf__ int32_t* tile_counts,
    __ubuf__ int32_t* rank_scratch,
    __ubuf__ int32_t* first_route_destinations,
    int32_t tile_tokens,
    int32_t topk,
    int32_t rank_count,
    int32_t max_tokens_per_rank,
    int32_t& group_count,
    int32_t& loff_count) {
    // Scalar-only: caller completes dst producers and zeroes tile_counts.
    // Scratch: rank_scratch has 2*rank_count entries (token tag, route mask);
    // first_route_destinations has topk entries. Both may be reused after return.
    // The caller restricts K to [8, 32]. Each rank's mask records route
    // positions; consuming least-significant bits preserves original order.
    // Reuse packing metadata for per-rank masks and raw-group storage for
    // K first-route destinations. duplicate_group_leader_mask contains only groups that have
    // duplicates; singleton destinations never enter the emission loop.
    for (int32_t rank = 0; rank < rank_count; ++rank) {
        rank_scratch[rank * 2] = -1;
    }
    group_count = 0;
    loff_count = 0;
    for (int32_t token = 0; token < tile_tokens; ++token) {
        const int32_t base = token * topk;
        uint32_t duplicate_group_leader_mask = 0;
        for (int32_t k = 0; k < topk; ++k) {
            const int32_t raw = dst[base + k];
            if (raw < 0) {
                continue;
            }
            const int32_t bucket = (raw / max_tokens_per_rank) * 2;
            const uint32_t bit = uint32_t{1} << k;
            if (rank_scratch[bucket] != token) {
                rank_scratch[bucket] = token;
                rank_scratch[bucket + 1] = static_cast<int32_t>(bit);
                first_route_destinations[k] = bucket / 2;
            } else {
                const uint32_t mask = static_cast<uint32_t>(rank_scratch[bucket + 1]);
                duplicate_group_leader_mask |= mask & (uint32_t{0} - mask);
                rank_scratch[bucket + 1] = static_cast<int32_t>(mask | bit);
            }
        }
        while (duplicate_group_leader_mask != 0) {
            const int32_t k = __builtin_ctz(duplicate_group_leader_mask);
            duplicate_group_leader_mask &= duplicate_group_leader_mask - 1;
            const int32_t raw = dst[base + k];
            const int32_t dest = first_route_destinations[k];
            uint32_t mask = static_cast<uint32_t>(rank_scratch[dest * 2 + 1]);
            mask &= mask - 1;
            const int32_t dest_base = dest * max_tokens_per_rank;
            compact_groups[group_count * 3] = dest;
            compact_groups[group_count * 3 + 1] = raw - dest_base;
            compact_groups[group_count * 3 + 2] = loff_count;
            ++group_count;
            int32_t count = 0;
            while (mask != 0) {
                const int32_t next = __builtin_ctz(mask);
                mask &= mask - 1;
                const int32_t duplicate = dst[base + next];
                compact_loffs[loff_count++] = duplicate - dest_base;
                dst[base + next] = -duplicate - 1;
                ++count;
            }
            tile_counts[dest * 2] += 1;
            tile_counts[dest * 2 + 1] += count;
        }
    }
}

__aicore__ inline void PackSourceDupByDestA2(
    __ubuf__ int32_t* compact_groups,
    __ubuf__ int32_t* compact_loffs,
    __ubuf__ int32_t* dest_meta,
    __ubuf__ int32_t* packed_groups,
    __ubuf__ int32_t* packed_loffs,
    int32_t group_count,
    int32_t loff_count) {
    // meta[0]/meta[3] retain aligned bucket starts; meta[1]/meta[4] were
    // initialized to those starts by the caller and now serve as write cursors.
    // Stable scatter visits each group once, rather than once per rank.
    for (int32_t group = 0; group < group_count; ++group) {
        const int32_t compact_pos = group * 3;
        const int32_t dest = compact_groups[compact_pos];
        const int32_t meta_pos = dest * 5;
        const int32_t raw_loff_start = compact_groups[compact_pos + 2];
        const int32_t next_raw_loff_start = group + 1 < group_count
            ? compact_groups[compact_pos + 5]
            : loff_count;
        const int32_t duplicate_count = next_raw_loff_start - raw_loff_start;
        const int32_t group_pos = dest_meta[meta_pos + 1];
        const int32_t loff_pos = dest_meta[meta_pos + 4];
        packed_groups[group_pos] = compact_groups[compact_pos + 1];
        packed_groups[group_pos + 1] = loff_pos - dest_meta[meta_pos + 3];
        packed_groups[group_pos + 2] = duplicate_count;
        for (int32_t i = 0; i < duplicate_count; ++i) {
            packed_loffs[loff_pos + i] = compact_loffs[raw_loff_start + i];
        }
        dest_meta[meta_pos + 1] = group_pos + 3;
        dest_meta[meta_pos + 4] = loff_pos + duplicate_count;
    }
}

__aicore__ inline void AdjustSourceDupGroupLoffsA2(
    __ubuf__ int32_t* groups,
    __ubuf__ int32_t* tile_counts,
    __ubuf__ int32_t* dest_meta,
    int32_t rank_count) {
    for (int32_t dest = 0; dest < rank_count; ++dest) {
        const int32_t group_count = tile_counts[dest * 2];
        const int32_t meta_pos = dest * 5;
        const int32_t group_start = dest_meta[meta_pos];
        const int32_t global_loff_base = dest_meta[meta_pos + 1];
        for (int32_t group = 0; group < group_count; ++group) {
            const int32_t loff_field = group_start + group * 3 + 1;
            groups[loff_field] = global_loff_base + groups[loff_field];
        }
    }
}

__aicore__ inline void PackSourceDupCountSlotsA2(
    __ubuf__ int32_t* core_counts,
    __ubuf__ int32_t* count_slots,
    int32_t core_count,
    int32_t dest_count,
    int32_t core_row_elems) {
    for (int32_t local_dest = 0; local_dest < dest_count; ++local_dest) {
        for (int32_t lane = 0; lane < kSourceDupInboxSlotElems; ++lane) {
            const int32_t slot =
                local_dest * kSourceDupInboxSlotElems + lane;
            if (lane >= 2) {
                count_slots[slot] = 0;
                continue;
            }
            int32_t total = 0;
            for (int32_t source_core = 0;
                 source_core < core_count;
                 ++source_core) {
                total += core_counts[
                    source_core * core_row_elems + local_dest * 2 + lane];
            }
            count_slots[slot] = total;
        }
    }
}

__aicore__ inline void BuildSourceDupDestinationPrefixesA2(
    __ubuf__ int32_t* count_slots,
    __ubuf__ int32_t* totals,
    int32_t rank_count) {
    for (int32_t lane = 0; lane < 2; ++lane) {
        int32_t prefix = 0;
        for (int32_t source_rank = 0;
             source_rank < rank_count;
             ++source_rank) {
            const int32_t offset =
                source_rank * kSourceDupInboxSlotElems + lane;
            const int32_t count = count_slots[offset];
            count_slots[offset] = prefix;
            prefix += count;
        }
        totals[lane] = prefix;
    }
}

__aicore__ inline void BuildSourceDupCorePrefixesA2(
    __ubuf__ int32_t* core_counts,
    __ubuf__ int32_t* rank_prefix_slots,
    int32_t core_count,
    int32_t dest_count,
    int32_t core_row_elems) {
    for (int32_t local_dest = 0; local_dest < dest_count; ++local_dest) {
        for (int32_t lane = 0; lane < 2; ++lane) {
            int32_t prefix = rank_prefix_slots[
                local_dest * kSourceDupInboxSlotElems + lane];
            for (int32_t source_core = 0;
                 source_core < core_count;
                 ++source_core) {
                const int32_t offset = source_core * core_row_elems +
                    local_dest * 2 + lane;
                const int32_t count = core_counts[offset];
                core_counts[offset] = prefix;
                prefix += count;
            }
        }
    }
}

// Per-expert ordinals only increase, including across route tiles. Find the
// initial destination once, then advance its cursor at quota boundaries.
// dst_cursor_cache is packed as three E-element rows (no per-row padding):
//   [0, E): destination rank; [E, 2E): exclusive segment end;
//   [2E, 3E): bias, so encoded_dst = bias + ordinal within that segment.
// Scalar-only: caller completes MTE2 loads and initializes segment ends to zero
// once before the first route tile. Cache rows persist across tiles; rank/bias
// are read only after a positive segment end makes them valid. Caller fences
// dst_tile for MTE3 and waits for its store before reusing the route buffers.
__aicore__ inline void BuildDstTileCachedA2(
    __ubuf__ int32_t* topk_tile,
    __ubuf__ int32_t* expert_ordinal,
    __ubuf__ int32_t* dst_tile,
    __ubuf__ int32_t* cumsum,
    __ubuf__ int32_t* offsets,
    __ubuf__ int32_t* dst_cursor_cache,
    int32_t tile_len,
    int32_t rank_count,
    int32_t expert_count,
    int32_t max_tokens_per_rank) {
    auto* destination_limits = dst_cursor_cache + expert_count;
    auto* destination_biases = destination_limits + expert_count;
    for (int32_t route = 0; route < tile_len; ++route) {
        const int32_t expert = topk_tile[route];
        const int32_t ordinal = expert_ordinal[expert]++;
        const int32_t row = expert * rank_count;
        // Zero is the uninitialized sentinel: every valid ordinal is >= 0,
        // and the segment containing it must have a strictly positive limit.
        const int32_t limit = destination_limits[expert];
        if (ordinal < limit) {
            // Most routes stay within the same expert/destination segment.
            dst_tile[route] = destination_biases[expert] + ordinal;
            continue;
        }
        int32_t dest = limit == 0 ? -1 : dst_cursor_cache[expert];
        if (dest < 0) {
            // Most experts retain a home-rank segment. Probe that interval
            // first, then narrow the binary search on a miss.
            const int32_t home = expert * rank_count / expert_count;
            const int32_t home_begin = home == 0 ? 0 : cumsum[row + home - 1];
            const int32_t home_end = cumsum[row + home];
            int32_t low = ordinal < home_begin ? 0 : home + 1;
            int32_t high = ordinal < home_begin ? home : rank_count;
            if (ordinal >= home_begin && ordinal < home_end) {
                low = home;
                high = home;
            }
            while (low < high) {
                const int32_t mid = low + (high - low) / 2;
                if (ordinal < cumsum[row + mid]) {
                    high = mid;
                } else {
                    low = mid + 1;
                }
            }
            dest = low < rank_count ? low : rank_count - 1;
        } else {
            while (dest + 1 < rank_count && ordinal >= cumsum[row + dest]) {
                ++dest;
            }
        }
        dst_cursor_cache[expert] = dest;
        destination_limits[expert] = cumsum[row + dest];
        const int32_t previous = dest == 0 ? 0 : cumsum[row + dest - 1];
        const int32_t bias = dest * max_tokens_per_rank +
            offsets[dest * expert_count + expert] - previous;
        destination_biases[expert] = bias;
        dst_tile[route] = bias + ordinal;
    }
}

// A2/A3 dst-tile builder. alloc_cumsum is [E,R] expert-major (contiguous per
// expert row); expert_offsets is [R,E] rank-major. The two [R,E] tables may
// exceed the 192KB A2/A3 UB at large world_size, so stage one contiguous
// expert-block at a time through MTE2 -- a full cumsum block (B*R int32) in
// one copy plus one B-int32 row per rank for expert_offsets -- then serve only
// the routes whose expert falls in the current block. The per-expert ordinal
// counter is independent per expert, so block-ordered processing yields dst
// encodings identical to a single in-order pass. expert_block_size is chosen
// by the caller so a single cumsum-block DataCopyPad fits the uint16 blockLen
// and both blocks stay below the UB cap.
// Caller: topk/ordinal inputs are ready for PIPE_S, shared GM lookup producers
// have completed, and scratch buffers have no pending consumers. This helper
// owns each lookup-block MTE2_S EVENT_ID0 pair; outputs are scalar UB writes.
// Caller still owns the S_MTE3 fence and completion wait for dst_tile writeback.
__aicore__ inline void BuildDstTileFromGmA2Oversized(
    __ubuf__ int32_t* topk_tile,
    __ubuf__ int32_t* expert_ordinal,
    __ubuf__ int32_t* dst_tile,
    AscendC::GlobalTensor<int32_t>& alloc_cumsum_gm,
    AscendC::GlobalTensor<int32_t>& expert_offsets_gm,
    AscendC::LocalTensor<int32_t>& cumsum_blk_ub,
    AscendC::LocalTensor<int32_t>& offsets_blk_ub,
    int32_t tile_len,
    int32_t rank_count,
    int32_t expert_count,
    int32_t max_tokens_per_rank,
    int32_t expert_block_size) {
    for (int32_t e0 = 0; e0 < expert_count; e0 += expert_block_size) {
        const int32_t e_end = IntMin(e0 + expert_block_size, expert_count);
        const int32_t blk_len = e_end - e0;
        const int32_t cumsum_blk_ints = blk_len * rank_count;
        // Stage the cumsum block (expert-major, contiguous) and the per-rank
        // expert_offsets rows for this expert block in one MTE2 batch, then make
        // them visible to scalar reads with a single MTE2->S fence.
        {
            AscendC::DataCopyExtParams cumsum_cp(
                1,
                static_cast<uint32_t>(cumsum_blk_ints * kInt32Bytes),
                0, 0, 0);
            AscendC::DataCopyPadExtParams<int32_t> pp{false, 0, 0, 0};
            AscendC::DataCopyPad(
                cumsum_blk_ub,
                alloc_cumsum_gm[e0 * rank_count],
                cumsum_cp, pp);
            AscendC::DataCopyExtParams off_cp(
                1,
                static_cast<uint32_t>(blk_len * kInt32Bytes),
                0, 0, 0);
            for (int32_t r = 0; r < rank_count; ++r) {
                AscendC::DataCopyPad(
                    offsets_blk_ub[r * expert_block_size],
                    expert_offsets_gm[r * expert_count + e0],
                    off_cp, pp);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        }
        for (int32_t route = 0; route < tile_len; ++route) {
            const int32_t expert = topk_tile[route];
            if (expert < e0 || expert >= e_end) {
                continue;
            }
            const int32_t local = expert - e0;
            const int32_t global_ordinal = expert_ordinal[expert];
            expert_ordinal[expert] = global_ordinal + 1;
            const int32_t cumsum_base = local * rank_count;
            int32_t low = 0;
            int32_t high = rank_count;
            while (low < high) {
                const int32_t mid = low + (high - low) / 2;
                if (global_ordinal < cumsum_blk_ub.GetValue(cumsum_base + mid)) {
                    high = mid;
                } else {
                    low = mid + 1;
                }
            }
            const int32_t dst_rank = low < rank_count ? low : rank_count - 1;
            const int32_t previous_boundary = dst_rank == 0
                ? 0
                : cumsum_blk_ub.GetValue(cumsum_base + dst_rank - 1);
            const int32_t dst_base =
                offsets_blk_ub.GetValue(dst_rank * expert_block_size + local);
            dst_tile[route] = dst_rank * max_tokens_per_rank + dst_base +
                global_ordinal - previous_boundary;
        }
    }
}

}

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
    // Stage map (UB is reused between stages; offsets below are byte offsets):
    // 1 histograms/local totals -> 2 gathered rank totals -> 3 owner balances
    // -> 4 migration quotas -> 5 expert allocation/cumsum -> 6 prefetch/offsets
    // -> 7 source-rank prefixes -> 8 core prefixes -> 9 dst -> 10 source dedup.
    // Layout reuse relies on the explicit pipe waits and grid joins at each
    // transition; overlapping LocalTensor views do not imply independent storage.
    PlanningResource resource;
    AscendC::LocalTensor<int32_t> work_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(0);
    PlanningKernelState state{};
    InitPlanningKernelState(
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
        tiling,
        ffts_addr,
        resource,
        work_ub,
        state);

    const int32_t core_idx = state.core_idx;
    const int32_t core_num = state.core_num;
    const int32_t rank = state.rank;
    const int32_t R = state.rank_count;
    const int32_t S = state.token_count;
    const int32_t K = state.topk;
    const int32_t E = state.expert_count;
    const int32_t NvS = state.max_tokens_per_rank;
    const int32_t B = state.prefetch_count;
    const int32_t epn = state.experts_per_rank;
    const int32_t N = state.route_items;
    const int32_t tpe_off = state.tpe_offset;
    const int32_t dup_count_inbox_off = state.count_inbox_offset;
    const int32_t dup_prefix_inbox_off = state.prefix_inbox_offset;
    const int32_t dup_payload_done_flag_off = state.payload_done_flag_offset;
    const int32_t dup_groups_scratch_off = state.groups_scratch_offset;
    const int32_t dup_loffs_scratch_off = state.loffs_scratch_offset;
    const int32_t source_dup_tokens_per_tile = state.source_dup_tokens_per_tile;
    const int32_t owner_reduce_tile_elems = state.owner_reduce_tile_elems;
    const int32_t route_core_num = state.route_core_num;
    const int32_t items_per_core = state.items_per_core;
    const int32_t begin = state.begin;
    const int32_t end = state.end;
    auto& topk_gm = state.topk_gm;
    auto& tpe_gm = state.tpe_gm;
    auto& dst_gm = state.dst_gm;
    auto& cu_gm = state.cu_gm;
    auto& experts_to_copy_gm = state.experts_to_copy_gm;
    auto& dup_groups_gm = state.dup_groups_gm;
    auto& dup_loffs_gm = state.dup_loffs_gm;
    auto& dup_counts_gm = state.dup_counts_gm;
    auto& tpe_all_gm = state.tpe_all_gm;
    auto& tpe_prefix_gm = state.tpe_prefix_gm;
    auto& balance_gm = state.balance_gm;
    auto& z_gm = state.z_gm;
    auto& alloc_gm = state.alloc_gm;
    auto& alloc_cumsum_gm = state.alloc_cumsum_gm;
    auto& expert_offsets_gm = state.expert_offsets_gm;
    auto& core_expert_counts_gm = state.core_expert_counts_gm;
    auto& cursor_worker_scratch_gm = state.cursor_worker_scratch_gm;
    auto& dup_workspace_gm = state.dup_workspace_gm;
    auto& dup_key_workspace_gm = state.dup_key_workspace_gm;
    auto& local_shmem_gm = state.local_shmem_gm;
    __gm__ int32_t* shmem_base = state.shmem_base;

    int32_t tpe_elems_per_core = E / core_num;
    int32_t tpe_tail_cores = E - tpe_elems_per_core * core_num;
    int32_t tpe_begin = core_idx * tpe_elems_per_core + IntMin(core_idx, tpe_tail_cores);
    int32_t tpe_assigned = tpe_elems_per_core + (core_idx < tpe_tail_cores ? 1 : 0);
    int32_t tpe_end = tpe_begin + tpe_assigned;

    // Stage 1: build one expert histogram per route-processing core. The same
    // rows later become exclusive per-core expert prefixes for dst ordinals.
    if (core_idx >= route_core_num) {
        return;
    }
    if (core_idx < route_core_num) {
        const int32_t histogram_bytes = AlignUpBytes(E * kInt32Bytes, 32);
        AscendC::LocalTensor<int32_t> route_tile_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(
                static_cast<uint32_t>(histogram_bytes));
        AscendC::Duplicate(work_ub, static_cast<int32_t>(0), E);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
        for (int32_t base = begin; base < end; base += kRouteHistogramTileElems) {
            const int32_t len =
                IntMin(kRouteHistogramTileElems, end - base);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
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
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            BuildRouteHistogramA2(
                (__ubuf__ int32_t*)route_tile_ub.GetPhyAddr(),
                (__ubuf__ int32_t*)work_ub.GetPhyAddr(),
                len);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
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
        // it until the previous vector reduction and its completion fence
        // have finished reading the matrix.
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID1);
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
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID1);
            AscendC::DataCopyPad(
                tpe_reduce_ub,
                core_expert_counts_gm[base],
                matrix_copy_params,
                pad_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            ReduceTpeRowsA2(
                work_ub,
                tpe_reduce_ub,
                static_cast<uint32_t>(route_core_num),
                static_cast<uint32_t>(len),
                static_cast<uint32_t>(kTpeReduceExpertsPerCore));
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID1);

            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::DataCopyPad(
                local_shmem_gm[tpe_off + base],
                work_ub,
                copy_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID1);
    }
    // A one-rank run has no remote SHMEM peer to synchronize. Avoid entering
    // the vector SHMEM barrier implementation, which is not valid on A2 when
    // the symmetric VMM window is addressed by scalar fallback code.
    if (R > 1) {
        aclshmemx_barrier_all_vec();
    }

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
    constexpr int32_t kMigrationMatrixScratchOffsetBytes = kTpeGatherTileBytes * 2;
    static_assert(kMigrationMatrixScratchOffsetBytes + kFillTileElems * kInt32Bytes <= kPlanningUbSizeBytes);
    AscendC::LocalTensor<int32_t> migration_matrix_scratch_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(static_cast<uint32_t>(kMigrationMatrixScratchOffsetBytes));
    FillContiguousGm(z_gm, migration_matrix_scratch_ub, 0, R * R, 0, core_idx, core_num);

    AscendC::SyncAll<true>();

    // Stage 3: reduce gathered counts into home allocation rows and balances.
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

        // 每个core负责owner_assigned个rank的token数统计
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
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
                ReduceTpeOwnerRowsA2(
                    tpe_owner_matrix_ub,
                    tile_total_ub,
                    tpe_owner_matrix_ub,
                    static_cast<uint32_t>(R),
                    static_cast<uint32_t>(len),
                    static_cast<uint32_t>(reduce_row_elems));
                AscendC::PipeBarrier<PIPE_ALL>();

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

    // Stage 4: core 0 greedily matches surplus/deficit ranks into z[src, dst].
    // UB: balance starts at 0; migration scratch starts after the two gather
    // tiles. It holds either all R*R entries or one aligned GM update block.
    if (core_idx == 0) {
        // The previous work_ub users have completed and synchronized here, so
        // the balance planner can reuse UB from offset 0 with a semantic name.
        AscendC::LocalTensor<int32_t> balance_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(0);
        // The R*R migration matrix can exceed UB at large EP sizes. Keep it
        // resident only when the full byte extent fits; otherwise update one
        // aligned GM block at a time using this same scratch buffer.
        const bool cache_migration_matrix = static_cast<int64_t>(kMigrationMatrixScratchOffsetBytes) +
            static_cast<int64_t>(R) * R * kInt32Bytes <= kPlanningUbSizeBytes;
        if (cache_migration_matrix) {
            AscendC::DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
            for (int32_t pos = 0; pos < R * R; pos += kFillTileElems) {
                const int32_t len = IntMin(kFillTileElems, R * R - pos);
                AscendC::DataCopyExtParams cp(1, len * kInt32Bytes, 0, 0, 0);
                AscendC::DataCopyPad(migration_matrix_scratch_ub[pos], z_gm[pos], cp, pad);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        }
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
            const int32_t z_index = surplus_rank * R + deficit_rank;
            if (cache_migration_matrix) {
                migration_matrix_scratch_ub.SetValue(z_index, move_tokens);
            } else {
                // Other cores finished zero-filling z before entering this
                // core-0 stage. Preserve neighboring entries in the GM block.
                const int32_t block_begin = (z_index / 8) * 8;
                const int32_t block_elems = IntMin(8, R * R - block_begin);
                AscendC::DataCopyExtParams block_copy(
                    1, block_elems * kInt32Bytes, 0, 0, 0);
                AscendC::DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
                AscendC::DataCopyPad(migration_matrix_scratch_ub, z_gm[block_begin], block_copy, pad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
                migration_matrix_scratch_ub.SetValue(z_index - block_begin, move_tokens);
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
                AscendC::DataCopyPad(z_gm[block_begin], migration_matrix_scratch_ub, block_copy);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
            }
            balance_ub.SetValue(surplus_rank, max_balance - move_tokens);
            balance_ub.SetValue(deficit_rank, 0);
        }
        if (cache_migration_matrix) {
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            for (int32_t pos = 0; pos < R * R; pos += kFillTileElems) {
                const int32_t len = IntMin(kFillTileElems, R * R - pos);
                AscendC::DataCopyExtParams cp(1, len * kInt32Bytes, 0, 0, 0);
                AscendC::DataCopyPad(z_gm[pos], migration_matrix_scratch_ub[pos], cp);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        }
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }
    AscendC::SyncAll<true>();

    // Stage 5: distribute each owner's experts according to migration quotas,
    // then publish alloc[dest, expert] and alloc_cumsum[expert, dest].
    // UB: [0, s_alloc_bytes) is the padded [R, epn] allocation matrix;
    // quota scratch follows it. After allocation, that scratch is reused for
    // a whole cumsum transpose if it fits, otherwise one R-element expert row.
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

            // 把结果搬运到alloc_gm
            const bool use_aligned_alloc_staging =
                (epn * kInt32Bytes) % 32 != 0;
            for (int32_t dest = 0; dest < R; ++dest) {
                const int32_t dest_row_offset = dest * s_alloc_row_elems;
                const int32_t store_elems = use_aligned_alloc_staging
                    ? s_alloc_row_elems
                    : epn;
                const int32_t gm_row_offset = use_aligned_alloc_staging
                    ? (dest * R + owner) * s_alloc_row_elems
                    : dest * E + expert_base;
                for (int32_t local_base = 0; local_base < store_elems;
                     local_base += kFillTileElems) {
                    const int32_t len =
                        IntMin(kFillTileElems, store_elems - local_base);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::DataCopyExtParams copy_params(
                        1,
                        static_cast<uint32_t>(len * kInt32Bytes),
                        0,
                        0,
                        0);
                    if (use_aligned_alloc_staging) {
                        AscendC::DataCopyPad(
                            dup_key_workspace_gm[gm_row_offset + local_base],
                            s_alloc_ub[dest_row_offset + local_base],
                            copy_params);
                    } else {
                        AscendC::DataCopyPad(
                            alloc_gm[gm_row_offset + local_base],
                            s_alloc_ub[dest_row_offset + local_base],
                            copy_params);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                }
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            BuildAllocCumsumA2(
                (__ubuf__ int32_t*)s_alloc_ub.GetPhyAddr(),
                R,
                epn,
                s_alloc_row_elems);
            AscendC::PipeBarrier<PIPE_ALL>();
            // Batch the expert-major cumsum transpose when the temporary
            // matrix fits in the remaining UB. Each row is data-block aligned.
            const int64_t transpose_bytes = static_cast<int64_t>(epn) * R * kInt32Bytes;
            if (R % 8 == 0 && s_alloc_bytes + transpose_bytes <= kPlanningUbSizeBytes) {
                for (int32_t local_expert = 0; local_expert < epn; ++local_expert) {
                    for (int32_t dest = 0; dest < R; ++dest) {
                        quota_ub.SetValue(local_expert * R + dest,
                            s_alloc_ub.GetValue(dest * s_alloc_row_elems + local_expert));
                    }
                }
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
                constexpr int32_t kTransposeCopyElems = 16376;
                for (int32_t pos = 0; pos < epn * R; pos += kTransposeCopyElems) {
                    const int32_t len = IntMin(kTransposeCopyElems, epn * R - pos);
                    AscendC::DataCopyExtParams cp(1, len * kInt32Bytes, 0, 0, 0);
                    AscendC::DataCopyPad(alloc_cumsum_gm[expert_base * R + pos], quota_ub[pos], cp);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            } else {
                // Scalar GM addressing is not available for A2 VMM pointers.
                // Transpose one expert row at a time into the quota UB tile and
                // publish it through the MTE3 path instead.
                for (int32_t local_expert = 0;
                     local_expert < epn;
                     ++local_expert) {
                    for (int32_t dest = 0; dest < R; ++dest) {
                        quota_ub.SetValue(
                            dest,
                            s_alloc_ub.GetValue(
                                dest * s_alloc_row_elems + local_expert));
                    }
                    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
                    AscendC::DataCopyExtParams row_copy_params(
                        1,
                        static_cast<uint32_t>(R * kInt32Bytes),
                        0,
                        0,
                        0);
                    AscendC::DataCopyPad(
                        alloc_cumsum_gm[(expert_base + local_expert) * R],
                        quota_ub,
                        row_copy_params);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
                }
            }
        }
    }
    AscendC::SyncAll<true>();

    // Stage 5b: compact padded staging rows when owners share GM data blocks.
    if ((epn * kInt32Bytes) % 32 != 0) {
        const int32_t alloc_staging_row_elems =
            AlignUpBytes(epn * kInt32Bytes, 32) / kInt32Bytes;
        AscendC::LocalTensor<int32_t> alloc_compact_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(0);
        AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        for (int32_t dest = core_idx; dest < R; dest += core_num) {
            for (int32_t owner = 0; owner < R; ++owner) {
                for (int32_t local_base = 0; local_base < epn;
                     local_base += kFillTileElems) {
                    const int32_t len =
                        IntMin(kFillTileElems, epn - local_base);
                    AscendC::DataCopyExtParams copy_params(
                        1,
                        static_cast<uint32_t>(len * kInt32Bytes),
                        0,
                        0,
                        0);
                    const int32_t staging_offset =
                        (dest * R + owner) * alloc_staging_row_elems +
                        local_base;
                    const int32_t compact_offset =
                        dest * E + owner * epn + local_base;

                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    AscendC::DataCopyPad(
                        alloc_compact_ub,
                        dup_key_workspace_gm[staging_offset],
                        copy_params,
                        pad_params);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                    AscendC::DataCopyPad(
                        alloc_gm[compact_offset],
                        alloc_compact_ub,
                        copy_params);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                }
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SyncAll<true>();
    }
    // Stage 6: select prefetch experts and build cu_seqlens/expert offsets.
    // PrefetchUbLayout separates group_counts[E+B], selected_experts[B] and
    // offset scratch. Selection keeps its sorted candidate counts in the B
    // trailing group_counts slots; scratch_counts_ub holds selected offsets
    // during prefix conversion. The shared layout also reserves A5 scratch.
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
    // Different cores own disjoint GM blocks in both packed output tables.
    // Aligned rows retain the original strided assignment; short rows are
    // processed in contiguous groups, with each group owned by one core.
    int32_t prefetch_d_begin = 0;
    int32_t prefetch_d_end = 0;
    int32_t prefetch_d_step = 1;
    PartitionPrefetchRowsA2(
        R, B, core_idx, core_num,
        prefetch_d_begin, prefetch_d_end, prefetch_d_step);
    for (int32_t d = prefetch_d_begin; d < prefetch_d_end; d += prefetch_d_step) {
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
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

        const int32_t local_expert_begin = d * epn;
        const int32_t local_expert_end = local_expert_begin + epn;
        SelectPrefetchExpertsA2(
            (__ubuf__ int32_t*)group_counts_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)selected_experts_ub.GetPhyAddr(),
            E,
            B,
            local_expert_begin,
            local_expert_end);
        AscendC::PipeBarrier<PIPE_ALL>();
        if ((B * kInt32Bytes) % 32 == 0) {
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            StoreContiguousGmThroughUb(
                experts_to_copy_gm,
                selected_experts_ub,
                d * B,
                B);
        } else {
            // Packed B-wide rows may share a 32-byte GM block. Their group
            // belongs to one core; complete each short store before reusing
            // its UB or issuing a later neighboring-row update on this core.
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            StoreContiguousGmThroughUb(
                experts_to_copy_gm,
                selected_experts_ub,
                d * B,
                B);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        }

        BuildGroupPrefixA2(
            (__ubuf__ int32_t*)group_counts_ub.GetPhyAddr(),
            E + B);
        AscendC::PipeBarrier<PIPE_ALL>();
        if (d == rank) {
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            StoreContiguousGmThroughUb(cu_gm, group_counts_ub, 0, E + B);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        }

        ConvertGroupPrefixToExpertOffsetsA2(
            (__ubuf__ int32_t*)group_counts_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)selected_experts_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)scratch_counts_ub.GetPhyAddr(),
            E,
            B);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
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

    // Stage 7: tpe_prefix[e] = sum_{r < rank} tpe_all[r, e].
    // UB: the output row starts at 0; the staged rank-by-expert matrix starts
    // at kTpePrefixMatrixOffsetBytes. Vector Add accumulates rows into the UB
    // output; ReduceTpeRowsA2 orders the vector operations and scalar consumers.
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

        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
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

            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
            AscendC::DataCopyPad(
                tpe_prefix_matrix_ub,
                tpe_all_gm[base],
                matrix_copy_params,
                pad_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            ReduceTpeRowsA2(
                work_ub,
                tpe_prefix_matrix_ub,
                static_cast<uint32_t>(rank),
                static_cast<uint32_t>(len),
                static_cast<uint32_t>(tpe_prefix_row_bytes / kInt32Bytes));
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::DataCopyPad(
                tpe_prefix_gm[base],
                work_ub,
                output_copy_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
    }
    AscendC::SyncAll<true>();

    // Stage 8: convert per-core histograms into exclusive expert/core cursors.
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
        // Preserve the shared-layout reservation; this A2/A3 scalar reduction
        // does not use per-thread totals in the reserved interval.
        constexpr int32_t kReservedThreadTotalsOffsetBytes = kTotalTileBytes;
        constexpr int32_t kTileTotalOffset =
            kReservedThreadTotalsOffsetBytes + kCoreCursorThreadPrefixBytes;
        static_assert(
            kTileTotalOffset + kCoreCursorCarryBytes <=
            kPlanningUbSizeBytes);

        AscendC::LocalTensor<int32_t> tpe_tile_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(0);
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
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

            tile_total_ub.SetValue(
                0,
                ReduceExpertTileTotalA2(
                    (__ubuf__ int32_t*)tpe_tile_ub.GetPhyAddr(), len));
            if (accumulate_total_tiles) {
                worker_total += tile_total_ub.GetValue(0);
            }
        }

        if (accumulate_total_tiles) {
            tile_total_ub.SetValue(0, worker_total);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
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

    // Stage 8b: replace counts in the same [route_core, expert] matrix with
    // expert-major cursors (the matrix's physical layout does not change). A worker
    // starts from the sum of all preceding expert partitions and advances its
    // carry across local tiles.
    if (core_idx < cursor_worker_num) {
        // UB layout: core-count matrix, one tpe row, reserved thread-prefix
        // storage, then expert carry. A2/A3 keeps the shared reservation but
        // does not use it; the tpe row itself becomes expert bases in the scan.
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
        const int32_t reserved_thread_prefix_offset_bytes =
            tpe_tile_offset + cursor_row_bytes;
        const int32_t expert_carry_offset =
            reserved_thread_prefix_offset_bytes + kCoreCursorThreadPrefixBytes;

        AscendC::LocalTensor<int32_t> cursor_matrix_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(0);
        AscendC::LocalTensor<int32_t> tpe_tile_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(
                static_cast<uint32_t>(tpe_tile_offset));
        AscendC::LocalTensor<int32_t> reserved_thread_prefix_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(
                static_cast<uint32_t>(reserved_thread_prefix_offset_bytes));
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
            // A2's scalar path is sensitive to short, strided matrix copies.
            // Load each route-core row as one contiguous transfer so the
            // cursor scan sees the exact [core, expert] layout.
            AscendC::DataCopyExtParams row_copy_params(
                1,
                static_cast<uint32_t>(active_row_bytes),
                0,
                0,
                0);
            for (int32_t source_core = 0;
                 source_core < route_core_num;
                 ++source_core) {
                AscendC::DataCopyPad(
                    cursor_matrix_ub[source_core * (cursor_row_bytes / kInt32Bytes)],
                    core_expert_counts_gm[base + source_core * E],
                    row_copy_params,
                    pad_params);
            }
            AscendC::DataCopyPad(
                tpe_tile_ub,
                tpe_gm[base],
                tpe_load_params,
                pad_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

            BuildCoreExpertCursorsA2(
                cursor_matrix_ub,
                tpe_tile_ub,
                expert_carry_ub,
                route_core_num,
                len,
                cursor_row_bytes / kInt32Bytes);

            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::DataCopyExtParams matrix_store_params(
                static_cast<uint16_t>(route_core_num),
                static_cast<uint32_t>(active_row_bytes),
                static_cast<uint32_t>(
                    (cursor_row_bytes - aligned_active_row_bytes) / 32),
                static_cast<uint32_t>((E - len) * kInt32Bytes),
                0);
            AscendC::DataCopyExtParams row_store_params(
                1,
                static_cast<uint32_t>(active_row_bytes),
                0,
                0,
                0);
            for (int32_t source_core = 0;
                 source_core < route_core_num;
                 ++source_core) {
                AscendC::DataCopyPad(
                    core_expert_counts_gm[base + source_core * E],
                    cursor_matrix_ub[source_core * (cursor_row_bytes / kInt32Bytes)],
                    row_store_params);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    AscendC::SyncAll<true>();
    // Stage 9: combine source-rank and core prefixes into expert ordinals,
    // then map each route to its destination rank and destination-local slot.
    // UB ordinal setup: E ordinals at 0, followed by two kFillTileElems scratch
    // rows. Those scratch rows expire before DstUbLayout is used below.
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
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            // Adjust the same expert tile loaded above, including E > 1024.
            AdjustExpertOrdinalsA2(
                (__ubuf__ int32_t*)expert_ordinal_ub.GetPhyAddr() + base,
                (__ubuf__ int32_t*)current_core_count_ub.GetPhyAddr(),
                (__ubuf__ int32_t*)core_zero_count_ub.GetPhyAddr(),
                len);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        DstUbLayout dst_layout{};
        InitDstUbLayout(R, E, expert_row_bytes, dst_layout);
        AscendC::LocalTensor<int32_t> dst_tile_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(
                static_cast<uint32_t>(dst_layout.dst_tile_offset));

        // DstUbLayout is shared with A5, so its field names retain "simt";
        // this backend uses those regions for scalar UB processing only.
        // Resident-path UB, in increasing byte-address order:
        //   ordinal[E] at 0; shared context reservation;
        //   dst[kFillTileElems]; cumsum[E,R]; offsets[R,E];
        //   topk[kFillTileElems]; packed cursor cache[3,E].
        // Lookup tables and cache persist across route tiles. The cache budget
        // reserves three aligned expert rows, though its payload is packed.
        const int64_t dst_cursor_cache_offset_bytes =
            static_cast<int64_t>(dst_layout.simt_topk_offset) +
            kFillTileElems * kInt32Bytes;
        const bool use_cached_lookup = dst_layout.lookup_table_elems > 0 &&
            dst_cursor_cache_offset_bytes + static_cast<int64_t>(expert_row_bytes) * 3 <=
                kPlanningUbSizeBytes;
        if (use_cached_lookup) {
            // Keep both tables resident across route tiles. On A3, repeatedly
            // staging the oversized fallback tables costs more than the data
            // volume suggests because each load also fences the scalar pipe.
            auto cumsum_ub = resource.ubBuf.template GetBufferByByte<int32_t>(
                static_cast<uint32_t>(dst_layout.lookup_offset));
            auto offsets_ub = resource.ubBuf.template GetBufferByByte<int32_t>(
                static_cast<uint32_t>(dst_layout.lookup_offset +
                                      dst_layout.lookup_table_bytes));
            auto topk_ub = resource.ubBuf.template GetBufferByByte<int32_t>(
                static_cast<uint32_t>(dst_layout.simt_topk_offset));
            auto dst_cursor_cache_ub = resource.ubBuf.template GetBufferByByte<int32_t>(
                static_cast<uint32_t>(dst_cursor_cache_offset_bytes));
            // Reserve all three rows. Initialize only segment ends to zero; rank and
            // bias become valid on the first route through each expert.
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
            // Keep each transfer below the fallback's 64 KiB copy limit.
            constexpr int32_t kLookupCopyElems = 16376;
            for (int32_t pos = 0; pos < dst_layout.lookup_table_elems;
                 pos += kLookupCopyElems) {
                const int32_t len = IntMin(
                    kLookupCopyElems, dst_layout.lookup_table_elems - pos);
                AscendC::DataCopyExtParams cp(
                    1, static_cast<uint32_t>(len * kInt32Bytes), 0, 0, 0);
                AscendC::DataCopyPad(cumsum_ub[pos], alloc_cumsum_gm[pos], cp, pad);
                AscendC::DataCopyPad(offsets_ub[pos], expert_offsets_gm[pos], cp, pad);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            auto* dst_cursor_cache = (__ubuf__ int32_t*)dst_cursor_cache_ub.GetPhyAddr();
            for (int32_t expert = 0; expert < E; ++expert) {
                dst_cursor_cache[E + expert] = 0;
            }
            for (int32_t base = begin; base < end; base += kFillTileElems) {
                const int32_t len = IntMin(kFillTileElems, end - base);
                AscendC::DataCopyExtParams cp(
                    1, static_cast<uint32_t>(len * kInt32Bytes), 0, 0, 0);
                AscendC::DataCopyPad(topk_ub, topk_gm[base], cp, pad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
                BuildDstTileCachedA2(
                    (__ubuf__ int32_t*)topk_ub.GetPhyAddr(),
                    (__ubuf__ int32_t*)expert_ordinal_ub.GetPhyAddr(),
                    (__ubuf__ int32_t*)dst_tile_ub.GetPhyAddr(),
                    (__ubuf__ int32_t*)cumsum_ub.GetPhyAddr(),
                    (__ubuf__ int32_t*)offsets_ub.GetPhyAddr(),
                    dst_cursor_cache, len, R, E, NvS);
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
                AscendC::DataCopyPad(dst_gm[base], dst_tile_ub, cp);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        } else {
            // Host validation guarantees that the ordinal row and both route
            // tiles fit when the resident lookup/cache layout does not.
            AscendC::LocalTensor<int32_t> fallback_route_tile_ub =
                resource.ubBuf.template GetBufferByByte<int32_t>(
                    static_cast<uint32_t>(
                        dst_layout.gm_simt_topk_offset));

            // alloc_cumsum and expert_offsets are produced by multiple cores.
            // The preceding SyncAll joins those producers, while this fence
            // makes their GM stores visible before fallback MTE2 lookup loads.
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
            CleanInvalidDataCache();

            const int32_t fallback_route_tile_bytes = AlignUpBytes(
                kFillTileElems * kInt32Bytes, 32);
            // Fallback UB after ordinal/context/dst: topk tile, cumsum block
            // [expert_block_size, R], then offsets block [R, expert_block_size].
            // Blocks are reloaded per expert range; no resident cursor cache.
            const int32_t fallback_cumsum_off = AlignUpBytes(
                dst_layout.gm_simt_topk_offset + fallback_route_tile_bytes, 32);
            const int32_t fallback_ub_budget =
                kPlanningUbSizeBytes - fallback_cumsum_off;
            const int32_t block_cap = IntMin(
                fallback_ub_budget / (2 * R * kInt32Bytes),
                65504 / (R * kInt32Bytes));
            const int32_t fallback_expert_block_size = (block_cap / 8) * 8;
            const int32_t fallback_blk_ints =
                fallback_expert_block_size * R;
            const int32_t fallback_offsets_off = AlignUpBytes(
                fallback_cumsum_off + fallback_blk_ints * kInt32Bytes, 32);
            AscendC::LocalTensor<int32_t> fallback_cumsum_blk_ub =
                resource.ubBuf.template GetBufferByByte<int32_t>(
                    static_cast<uint32_t>(fallback_cumsum_off));
            AscendC::LocalTensor<int32_t> fallback_offsets_blk_ub =
                resource.ubBuf.template GetBufferByByte<int32_t>(
                    static_cast<uint32_t>(fallback_offsets_off));

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
                    fallback_route_tile_ub,
                    topk_gm[base],
                    copy_params,
                    pad_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

                BuildDstTileFromGmA2Oversized(
                    (__ubuf__ int32_t*)fallback_route_tile_ub.GetPhyAddr(),
                    (__ubuf__ int32_t*)expert_ordinal_ub.GetPhyAddr(),
                    (__ubuf__ int32_t*)dst_tile_ub.GetPhyAddr(),
                    alloc_cumsum_gm,
                    expert_offsets_gm,
                    fallback_cumsum_blk_ub,
                    fallback_offsets_blk_ub,
                    len,
                    R,
                    E,
                    NvS,
                    fallback_expert_block_size);
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);

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

    // Stage 10: deduplicate source routes and exchange counts/prefixes/payloads.
    // Source-dup immediately reads back dst through MTE2. Keep the complete
    // exchange in this architecture block so its synchronization order stays
    // explicit and local to the selected backend.
    const SourceDupStageArgs source_dup_args{
        SourceDupConfig{
            rank,
            R,
            S,
            K,
            NvS,
            core_idx,
            core_num,
            source_dup_tokens_per_tile},
        dup_count_inbox_off,
        dup_prefix_inbox_off,
        dup_payload_done_flag_off,
        dup_groups_scratch_off,
        dup_loffs_scratch_off,
        &resource,
        &dst_gm,
        &dup_workspace_gm,
        &dup_key_workspace_gm,
        &local_shmem_gm,
        &dup_counts_gm,
        &dup_groups_gm,
        &dup_loffs_gm,
        shmem_base};
    RunSourceDupStages(source_dup_args);

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
    planning_kernel<<<td->block_dim, nullptr, stream>>>(
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
