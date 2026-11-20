#include "csrc/ops/planning.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "ascend_moonep/kernels/types.h"
#include "ascend_moonep/kernels/arch_config.h"
#include "csrc/launcher.h"
#include "shmem.h"

namespace {

constexpr int64_t kMinPlanningCores = 8;
constexpr int64_t kMaxPlanningCores = 56;
constexpr int64_t kMaxPlanningRanks = 1024;
constexpr int64_t kInt32Bytes = 4;
constexpr int64_t kUbAlignBytes = 32;
constexpr int64_t kAlignElems = kUbAlignBytes / kInt32Bytes;
constexpr int64_t kDupSrcTileElems = 1024;
constexpr int64_t kDstTileElems = 1024;
constexpr int64_t kRouteHistogramTileElems = 4096;
constexpr int64_t kDstSimtContextBytes = 64;
constexpr int64_t kSourceDupSimtMaxThreads = 1024;
constexpr int64_t kSourceDupInboxSlotElems = kAlignElems;
constexpr int64_t kPrefetchSimtMaxThreads = 1024;
constexpr int64_t kSourceDupPreferredTokens = 256;
constexpr int64_t kSourceDupThreadAlign = 32;
constexpr int64_t kSourceDupPackTinyChunkGroups = 16;
constexpr int64_t kSourceDupPackSmallChunkGroups = 32;
constexpr int64_t kSourceDupPackMediumChunkGroups = 64;
constexpr int64_t kSourceDupPackLargeChunkGroups = 128;
constexpr int64_t kSourceDupPackXLargeChunkGroups = 256;
constexpr int64_t kOwnerReduceMaxExperts = 64;
constexpr int64_t kOwnerReduceTotalBytes = kUbAlignBytes;
constexpr uint64_t kPlanningUbSizeBytes = ascend_moonep::kPlanningUbSizeBytes;
constexpr int64_t kCursorWorkerScratchRowElems = kAlignElems;

struct WorkspaceSlice {
    int64_t offset;
    int64_t size;
};

struct PlanningWorkspaceLayout {
    WorkspaceSlice tpe_all;
    WorkspaceSlice tpe_prefix;
    WorkspaceSlice balance;
    WorkspaceSlice z;
    WorkspaceSlice alloc;
    WorkspaceSlice alloc_cumsum;
    WorkspaceSlice expert_offsets;
    WorkspaceSlice core_expert_counts;
    WorkspaceSlice dup_workspace;
    WorkspaceSlice dup_key_workspace;
    int64_t total_elems;
};

int64_t AlignUp(int64_t value, int64_t alignment) {
    TORCH_CHECK(value >= 0 && alignment > 0, "invalid alignment operands");
    TORCH_CHECK(
        value <= std::numeric_limits<int64_t>::max() - (alignment - 1),
        "planning workspace size overflow");
    return ((value + alignment - 1) / alignment) * alignment;
}

int64_t AlignDown(int64_t value, int64_t alignment) {
    return (value / alignment) * alignment;
}

int64_t CheckedMul(int64_t lhs, int64_t rhs, const char* label) {
    TORCH_CHECK(lhs >= 0 && rhs >= 0, label, " dimensions must be non-negative");
    if (lhs == 0 || rhs == 0) {
        return 0;
    }
    TORCH_CHECK(
        lhs <= std::numeric_limits<int64_t>::max() / rhs,
        label,
        " size overflow");
    return lhs * rhs;
}

int64_t SelectOwnerReduceTileElems(
    int64_t world_size,
    int64_t experts_per_rank,
    int64_t core_num,
    int64_t ub_size_bytes) {
    const int64_t max_owners_per_core =
        (world_size + core_num - 1) / core_num;
    const int64_t balance_bytes = AlignUp(
        max_owners_per_core * kInt32Bytes, kUbAlignBytes);
    const int64_t auxiliary_bytes =
        kOwnerReduceTotalBytes + balance_bytes;
    TORCH_CHECK(
        auxiliary_bytes < ub_size_bytes,
        "planning owner-reduce auxiliary buffers exceed UB capacity");

    const int64_t row_budget_bytes = AlignDown(
        (ub_size_bytes - auxiliary_bytes) / world_size,
        kUbAlignBytes);
    TORCH_CHECK(
        row_budget_bytes >= kUbAlignBytes,
        "planning owner-reduce UB cannot hold one aligned expert tile for ",
        world_size,
        " ranks");

    return std::min({
        kOwnerReduceMaxExperts,
        experts_per_rank,
        row_budget_bytes / kInt32Bytes});
}

int64_t SourceDupPackChunkGroups(int64_t world_size, int64_t num_topk) {
    if (world_size <= 32 && num_topk <= 8) {
        return kSourceDupPackTinyChunkGroups;
    }
    if (world_size <= 32) {
        return kSourceDupPackSmallChunkGroups;
    }
    if (world_size <= 128) {
        return kSourceDupPackMediumChunkGroups;
    }
    if (world_size <= 256) {
        return kSourceDupPackLargeChunkGroups;
    }
    return kSourceDupPackXLargeChunkGroups;
}

int64_t SourceDupUbBytes(
    int64_t world_size,
    int64_t num_topk,
    int64_t tokens_per_tile) {
    const int64_t route_tile_elems = CheckedMul(tokens_per_tile, num_topk, "route tile");
    const int64_t staging_elems = std::max(route_tile_elems + world_size * 7, world_size * 2);
    const int64_t prefix_bytes =
        AlignUp(world_size * 2 * kInt32Bytes, kUbAlignBytes) +
        AlignUp(staging_elems * kInt32Bytes, kUbAlignBytes) +
        AlignUp(world_size * 2 * kInt32Bytes, kUbAlignBytes);

    const int64_t max_group_count = route_tile_elems / 2;
    const int64_t group_send_elems = std::max(max_group_count * 3 + world_size * 7, world_size * 2);
    const int64_t raw_group_tile_elems = std::max(max_group_count * 4, group_send_elems);
    const int64_t raw_loff_tile_elems = route_tile_elems + world_size * 7;
    const int64_t pack_chunk_groups =
        SourceDupPackChunkGroups(world_size, num_topk);
    const int64_t pack_chunk_count =
        (max_group_count + pack_chunk_groups - 1) / pack_chunk_groups;
    const int64_t pack_hist_row_bytes =
        AlignUp(world_size * 2 * kInt32Bytes, kUbAlignBytes);
    const int64_t pack_hist_bytes =
        CheckedMul(pack_chunk_count, pack_hist_row_bytes, "source dup pack histogram");
    const int64_t group_bytes =
        AlignUp(world_size * 2 * kInt32Bytes, kUbAlignBytes) +
        AlignUp(staging_elems * kInt32Bytes, kUbAlignBytes) +
        AlignUp(world_size * 2 * kInt32Bytes, kUbAlignBytes) +
        AlignUp(world_size * 5 * kInt32Bytes, kUbAlignBytes) +
        AlignUp(raw_group_tile_elems * kInt32Bytes, kUbAlignBytes) +
        AlignUp(raw_loff_tile_elems * kInt32Bytes, kUbAlignBytes) +
        AlignUp(group_send_elems * kInt32Bytes, kUbAlignBytes) +
        AlignUp(world_size * 2 * kInt32Bytes, kUbAlignBytes) +
        4 * AlignUp(tokens_per_tile * kInt32Bytes, kUbAlignBytes) +
        AlignUp(route_tile_elems * kInt32Bytes, kUbAlignBytes) +
        pack_hist_bytes;
    const int64_t materialize_bytes =
        AlignUp(world_size * 2 * kInt32Bytes, kUbAlignBytes) +
        2 * AlignUp(kDupSrcTileElems * kInt32Bytes, kUbAlignBytes);
    return std::max({prefix_bytes, group_bytes, materialize_bytes});
}

uint32_t SelectSourceDupTokensPerTile(
    int64_t world_size,
    int64_t num_topk,
    uint64_t ub_size) {
    int64_t preferred_tokens = std::max(kSourceDupPreferredTokens, kDupSrcTileElems / num_topk);
    preferred_tokens = std::min(preferred_tokens, kSourceDupSimtMaxThreads);
    if (preferred_tokens >= kSourceDupThreadAlign) {
        preferred_tokens = AlignDown(preferred_tokens, kSourceDupThreadAlign);
    }
    for (int64_t tokens = preferred_tokens;
         tokens >= kSourceDupThreadAlign;
         tokens -= kSourceDupThreadAlign) {
        if (SourceDupUbBytes(world_size, num_topk, tokens) <= static_cast<int64_t>(ub_size)) {
            return static_cast<uint32_t>(tokens);
        }
    }
    for (int64_t tokens = std::min(preferred_tokens, kSourceDupThreadAlign - 1);
         tokens > 0;
         --tokens) {
        if (SourceDupUbBytes(world_size, num_topk, tokens) <= static_cast<int64_t>(ub_size)) {
            return static_cast<uint32_t>(tokens);
        }
    }
    TORCH_CHECK(
        false,
        "planning source duplicate UB workspace cannot hold one token when world_size=",
        world_size,
        ", num_topk=",
        num_topk);
    return 0;
}

PlanningWorkspaceLayout MakePlanningWorkspaceLayout(
    int64_t num_tokens,
    int64_t num_topk,
    int64_t num_experts,
    int64_t world_size,
    int64_t core_num) {
    TORCH_CHECK(num_tokens > 0, "num_tokens must be positive");
    TORCH_CHECK(num_topk > 0, "num_topk must be positive");
    TORCH_CHECK(num_experts > 0, "num_experts must be positive");
    TORCH_CHECK(world_size > 0, "world_size must be positive");
    TORCH_CHECK(world_size <= kMaxPlanningRanks,
                "planning world_size must be at most ", kMaxPlanningRanks,
                ", got ", world_size);
    TORCH_CHECK(num_experts % world_size == 0,
                "num_experts must be divisible by world_size");
    TORCH_CHECK(core_num >= kMinPlanningCores && core_num <= kMaxPlanningCores,
                "planning core_num must be in [8, 56], got ", core_num);

    const int64_t route_count = CheckedMul(num_tokens, num_topk, "route count");
    const int64_t rank_experts = CheckedMul(world_size, num_experts, "rank-expert table");
    TORCH_CHECK(route_count <= std::numeric_limits<int32_t>::max(),
                "num_tokens * num_topk must fit int32 indexing");
    TORCH_CHECK(rank_experts <= std::numeric_limits<int32_t>::max(),
                "world_size * num_experts must fit int32 indexing");
    TORCH_CHECK(
        CheckedMul(world_size, route_count, "encoded dst range") <
            (int64_t{1} << 31),
        "world_size * num_tokens * num_topk must be less than 2^31");
    const int64_t epn = num_experts / world_size;
    const int64_t prefetch_scratch_elems =
        std::max(epn, kPrefetchSimtMaxThreads);
    const int64_t prefetch_layout_ub_bytes =
        AlignUp(
            CheckedMul(
                num_experts + epn,
                kInt32Bytes,
                "prefetch group-count row"),
            kUbAlignBytes) +
        AlignUp(
            CheckedMul(epn, kInt32Bytes, "prefetch selected-expert row"),
            kUbAlignBytes) +
        2 * AlignUp(
            CheckedMul(
                prefetch_scratch_elems,
                kInt32Bytes,
                "prefetch scratch row"),
            kUbAlignBytes);
    TORCH_CHECK(
        prefetch_layout_ub_bytes <= static_cast<int64_t>(kPlanningUbSizeBytes),
        "planning prefetch-layout UB workspace is too large: need ",
        prefetch_layout_ub_bytes,
        " bytes, capacity is ",
        kPlanningUbSizeBytes);
    const int64_t expert_row_bytes = AlignUp(
        CheckedMul(
            num_experts,
            kInt32Bytes,
            "dst ordinal expert row"),
        kUbAlignBytes);
    const int64_t ordinal_init_ub_bytes =
        expert_row_bytes +
        2 * AlignUp(kDstTileElems * kInt32Bytes, kUbAlignBytes);
    const int64_t dst_gm_simt_ub_bytes =
        expert_row_bytes + kDstSimtContextBytes +
        2 * AlignUp(kDstTileElems * kInt32Bytes, kUbAlignBytes);
    const int64_t route_histogram_ub_bytes =
        expert_row_bytes +
        AlignUp(kRouteHistogramTileElems * kInt32Bytes, kUbAlignBytes);
    const int64_t dst_ordinal_ub_bytes = std::max(
        route_histogram_ub_bytes,
        std::max(ordinal_init_ub_bytes, dst_gm_simt_ub_bytes));
    TORCH_CHECK(
        dst_ordinal_ub_bytes <= static_cast<int64_t>(kPlanningUbSizeBytes),
        "planning dst-ordinal UB workspace is too large: need ",
        dst_ordinal_ub_bytes,
        " bytes, capacity is ",
        kPlanningUbSizeBytes);
    const int64_t alloc_migration_ub_bytes =
        world_size * AlignUp(epn * kInt32Bytes, kUbAlignBytes) +
        AlignUp(world_size * kInt32Bytes, kUbAlignBytes);
    TORCH_CHECK(
        alloc_migration_ub_bytes <= static_cast<int64_t>(kPlanningUbSizeBytes),
        "planning alloc migration UB workspace is too large: need ",
        alloc_migration_ub_bytes,
        " bytes, capacity is ",
        kPlanningUbSizeBytes);
    (void)SelectOwnerReduceTileElems(
        world_size,
        epn,
        core_num,
        static_cast<int64_t>(kPlanningUbSizeBytes));
    const int64_t source_dup_tokens_per_tile =
        SelectSourceDupTokensPerTile(world_size, num_topk, kPlanningUbSizeBytes);
    const int64_t max_tokens_per_core = (num_tokens + core_num - 1) / core_num;
    const int64_t max_tiles_per_core =
        (max_tokens_per_core + source_dup_tokens_per_tile - 1) /
        source_dup_tokens_per_tile;
    const int64_t dup_count_elems = kMaxPlanningCores * world_size * 2;
    const int64_t tile_count_row_elems = AlignUp(world_size * 2, kAlignElems);
    const int64_t source_dup_workspace_elems =
        dup_count_elems + core_num * max_tiles_per_core * tile_count_row_elems;
    const int64_t dup_workspace_elems = source_dup_workspace_elems;

    const int64_t max_routes_per_core = CheckedMul(max_tokens_per_core, num_topk, "routes per core");
    const int64_t payload_prefix_elems = kMaxPlanningCores * world_size * 2;
    const int64_t group_workspace_stride = (max_routes_per_core / 2) * 3;
    const int64_t loff_workspace_stride = max_routes_per_core;
    const int64_t payload_workspace_elems =
        payload_prefix_elems + core_num * (group_workspace_stride + loff_workspace_stride);
    const int64_t route_tile_elems = source_dup_tokens_per_tile * num_topk;
    const int64_t prebuilt_group_stride =
        AlignUp((route_tile_elems / 2) * 3 + world_size * 7, kAlignElems);
    const int64_t prebuilt_loff_stride =
        AlignUp(route_tile_elems + world_size * 7, kAlignElems);
    const int64_t prebuilt_group_off = AlignUp(payload_workspace_elems, kAlignElems);
    const int64_t prebuilt_tile_slots = core_num * max_tiles_per_core;
    const int64_t prebuilt_loff_off =
        prebuilt_group_off + prebuilt_tile_slots * prebuilt_group_stride;
    const int64_t source_dup_key_workspace_elems =
        prebuilt_loff_off + prebuilt_tile_slots * prebuilt_loff_stride;
#if defined(ASCEND_MOONEP_ARCH_A5)
    const int64_t dup_key_workspace_elems = source_dup_key_workspace_elems;
#else
    const int64_t alloc_staging_elems = epn % kAlignElems == 0
        ? 0
        : CheckedMul(
              CheckedMul(world_size, world_size, "alloc staging rank table"),
              AlignUp(epn, kAlignElems),
              "alloc staging workspace");
    const int64_t dup_key_workspace_elems =
        std::max(source_dup_key_workspace_elems, alloc_staging_elems);
#endif

    PlanningWorkspaceLayout layout{};
    int64_t cursor = 0;
    auto add = [&](WorkspaceSlice& slice, int64_t size) {
        cursor = AlignUp(cursor, kAlignElems);
        slice = {cursor, size};
        TORCH_CHECK(cursor <= std::numeric_limits<int64_t>::max() - size,
                    "planning local workspace size overflow");
        cursor += size;
    };

    // Every region starts at a 32-byte-aligned int32 element offset. The
    // regions share one allocation because their lifetime is the full launch.
    add(layout.tpe_all, rank_experts);
    add(layout.tpe_prefix, num_experts);
    add(layout.balance, world_size);
    add(layout.z, CheckedMul(world_size, world_size, "z table"));
    add(layout.alloc, rank_experts);
    add(layout.alloc_cumsum, rank_experts);
    add(layout.expert_offsets, rank_experts);
    const int64_t core_expert_count_elems =
        CheckedMul(kMaxPlanningCores, num_experts, "core expert count table");
    const int64_t cursor_worker_scratch_elems = CheckedMul(
        kMaxPlanningCores,
        kCursorWorkerScratchRowElems,
        "core cursor worker scratch");
    TORCH_CHECK(
        core_expert_count_elems <=
            std::numeric_limits<int64_t>::max() - cursor_worker_scratch_elems,
        "core expert count workspace size overflow");
    // The tail stores one private 32-byte total per cursor worker. Keeping it
    // with the cursor table avoids reusing source-dup workspace across stages.
    add(
        layout.core_expert_counts,
        core_expert_count_elems + cursor_worker_scratch_elems);
    add(layout.dup_workspace, dup_workspace_elems);
    add(layout.dup_key_workspace, dup_key_workspace_elems);
    layout.total_elems = AlignUp(cursor, kAlignElems);
    TORCH_CHECK(
        layout.total_elems <= std::numeric_limits<int32_t>::max(),
        "planning local workspace exceeds int32 device offsets: ",
        layout.total_elems,
        " elements");
    return layout;
}

void CheckNpuInt32(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
                name, " must be an NPU tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == torch::kInt32, name, " must be int32");
}

void CheckSameDevice(
    const torch::Tensor& tensor,
    const torch::Tensor& reference,
    const char* name) {
    TORCH_CHECK(tensor.device() == reference.device(),
                name, " must be on the same device as topk_idx");
}

torch::Tensor SliceWorkspace(
    const torch::Tensor& workspace,
    const WorkspaceSlice& slice) {
    return workspace.narrow(0, slice.offset, slice.size);
}

}  // namespace

namespace ascend_moonep::ops {

int64_t PlanningSymmetricWorkspaceBytes(
    int64_t num_experts,
    int64_t num_max_tokens_per_rank,
    int64_t world_size) {
    TORCH_CHECK(num_experts > 0, "num_experts must be positive");
    TORCH_CHECK(num_max_tokens_per_rank > 0, "num_max_tokens_per_rank must be positive");
    TORCH_CHECK(world_size > 0, "world_size must be positive");
    const int64_t dup_rank_counts_off = AlignUp(num_experts, kAlignElems);
    const int64_t dup_payload_done_off = dup_rank_counts_off +
        CheckedMul(
            CheckedMul(world_size, kSourceDupInboxSlotElems, "source dup count inbox"),
            2,
            "source dup exchange inboxes");
    const int64_t dup_groups_off =
        dup_payload_done_off + AlignUp(world_size, kAlignElems);
    const int64_t dup_loffs_off =
        dup_groups_off + AlignUp(num_max_tokens_per_rank * 3, kAlignElems);
    const int64_t required_ints =
        dup_loffs_off + AlignUp(num_max_tokens_per_rank, kAlignElems);
    return CheckedMul(required_ints, kInt32Bytes, "planning symmetric workspace");
}

int64_t PlanningLocalWorkspaceElements(
    int64_t num_tokens,
    int64_t num_topk,
    int64_t num_experts,
    int64_t world_size,
    int64_t core_num) {
    return MakePlanningWorkspaceLayout(
        num_tokens, num_topk, num_experts, world_size, core_num).total_elems;
}

void Planning(
    torch::Tensor topk_idx,
    torch::Tensor dst,
    torch::Tensor cu_seqlens,
    torch::Tensor experts_to_copy,
    torch::Tensor dup_groups,
    torch::Tensor dup_loffs,
    torch::Tensor dup_counts,
    torch::Tensor local_workspace,
    int64_t symmetric_workspace_addr,
    int64_t symmetric_workspace_bytes,
    int64_t num_tokens,
    int64_t num_topk,
    int64_t num_experts,
    int64_t num_max_tokens_per_rank,
    int64_t num_prefetch_slots,
    int64_t core_num,
    int64_t rank,
    int64_t world_size,
    bool use_udma) {
    (void)use_udma;
    CheckNpuInt32(topk_idx, "topk_idx");
    CheckNpuInt32(dst, "dst");
    CheckNpuInt32(cu_seqlens, "cu_seqlens");
    CheckNpuInt32(experts_to_copy, "experts_to_copy");
    CheckNpuInt32(dup_groups, "dup_groups");
    CheckNpuInt32(dup_loffs, "dup_loffs");
    CheckNpuInt32(dup_counts, "dup_counts");
    CheckNpuInt32(local_workspace, "local_workspace");
    for (const auto* entry : {
             &dst, &cu_seqlens, &experts_to_copy, &dup_groups,
             &dup_loffs, &dup_counts, &local_workspace}) {
        CheckSameDevice(*entry, topk_idx, "planning tensor");
    }

    TORCH_CHECK(num_tokens > 0, "num_tokens must be positive");
    TORCH_CHECK(num_topk > 0, "num_topk must be positive");
    TORCH_CHECK(num_experts > 0, "num_experts must be positive");
    TORCH_CHECK(world_size > 0, "world_size must be positive");
    TORCH_CHECK(num_max_tokens_per_rank > 0,
                "num_max_tokens_per_rank must be positive");
    TORCH_CHECK(num_prefetch_slots > 0, "num_prefetch_slots must be positive");
    TORCH_CHECK(num_experts % world_size == 0,
                "num_experts must be divisible by world_size");
    TORCH_CHECK(rank >= 0 && rank < world_size, "rank must be in [0, world_size)");
    TORCH_CHECK(num_prefetch_slots == num_experts / world_size,
                "planning requires num_prefetch_slots == num_experts / world_size");
    const int64_t route_count = CheckedMul(num_tokens, num_topk, "route count");
    for (const auto value : {
             num_tokens, num_topk, num_experts, num_max_tokens_per_rank,
             num_prefetch_slots, core_num, rank, world_size}) {
        TORCH_CHECK(
            value >= 0 &&
                value <= std::numeric_limits<int32_t>::max(),
            "planning dimensions must fit signed int32 device indices");
    }
    TORCH_CHECK(
        CheckedMul(world_size, num_max_tokens_per_rank, "encoded dst range") <
            (int64_t{1} << 31),
        "planning requires world_size * num_max_tokens_per_rank < 2^31");
    TORCH_CHECK(
        CheckedMul(world_size, num_experts, "rank-expert table") <=
            std::numeric_limits<int32_t>::max(),
        "planning requires world_size * num_experts to fit int32 indexing");
    TORCH_CHECK(num_max_tokens_per_rank >= route_count,
                "num_max_tokens_per_rank must be at least num_tokens * num_topk");
    TORCH_CHECK(topk_idx.numel() == route_count,
                "topk_idx numel must equal num_tokens * num_topk");
    TORCH_CHECK(dst.numel() == route_count, "dst shape mismatch");
    TORCH_CHECK(cu_seqlens.numel() == num_experts + num_prefetch_slots,
                "cu_seqlens shape mismatch");
    TORCH_CHECK(experts_to_copy.numel() == world_size * num_prefetch_slots,
                "experts_to_copy shape mismatch");
    TORCH_CHECK(dup_groups.numel() == num_max_tokens_per_rank * 3,
                "dup_groups shape mismatch");
    TORCH_CHECK(dup_loffs.numel() == num_max_tokens_per_rank,
                "dup_loffs shape mismatch");
    TORCH_CHECK(dup_counts.numel() == 2, "dup_counts shape mismatch");
    TORCH_CHECK(symmetric_workspace_addr != 0,
                "planning symmetric workspace address must not be zero");
    TORCH_CHECK(
        reinterpret_cast<uintptr_t>(local_workspace.data_ptr()) % kUbAlignBytes == 0,
        "planning local workspace base must be 32-byte aligned");

    const auto layout = MakePlanningWorkspaceLayout(
        num_tokens, num_topk, num_experts, world_size, core_num);
    TORCH_CHECK(local_workspace.numel() >= layout.total_elems,
                "planning local workspace is too small: need ",
                layout.total_elems, " int32 elements, got ", local_workspace.numel());
    const int64_t required_symmetric_bytes = PlanningSymmetricWorkspaceBytes(
        num_experts, num_max_tokens_per_rank, world_size);
    TORCH_CHECK(symmetric_workspace_bytes >= required_symmetric_bytes,
                "planning symmetric workspace is too small: need ",
                required_symmetric_bytes, " bytes, got ", symmetric_workspace_bytes);

    PlanningTilingData tiling{};
    tiling.rank = static_cast<uint32_t>(rank);
    tiling.world_size = static_cast<uint32_t>(world_size);
    tiling.num_tokens = static_cast<uint32_t>(num_tokens);
    tiling.num_topk = static_cast<uint32_t>(num_topk);
    tiling.num_experts = static_cast<uint32_t>(num_experts);
    tiling.num_max_tokens_per_rank = static_cast<uint32_t>(num_max_tokens_per_rank);
    tiling.num_prefetch_slots = static_cast<uint32_t>(num_prefetch_slots);
    tiling.block_dim = static_cast<uint32_t>(core_num);
    tiling.symmetric_tpe_offset = 0;
    tiling.symmetric_dup_rank_counts_offset =
        static_cast<uint32_t>(AlignUp(num_experts, kAlignElems));
    tiling.source_dup_tokens_per_tile =
        SelectSourceDupTokensPerTile(world_size, num_topk, kPlanningUbSizeBytes);
    tiling.owner_reduce_tile_elems = static_cast<uint32_t>(
        SelectOwnerReduceTileElems(
            world_size,
            num_experts / world_size,
            core_num,
            static_cast<int64_t>(kPlanningUbSizeBytes)));
    tiling.reserved = 0;

    auto tpe_all = SliceWorkspace(local_workspace, layout.tpe_all);
    auto tpe_prefix = SliceWorkspace(local_workspace, layout.tpe_prefix);
    auto balance = SliceWorkspace(local_workspace, layout.balance);
    auto z = SliceWorkspace(local_workspace, layout.z);
    auto alloc = SliceWorkspace(local_workspace, layout.alloc);
    auto alloc_cumsum = SliceWorkspace(local_workspace, layout.alloc_cumsum);
    auto expert_offsets = SliceWorkspace(local_workspace, layout.expert_offsets);
    auto core_expert_counts = SliceWorkspace(local_workspace, layout.core_expert_counts);
    auto dup_workspace = SliceWorkspace(local_workspace, layout.dup_workspace);
    auto dup_key_workspace = SliceWorkspace(local_workspace, layout.dup_key_workspace);

    // SHMEM initializes FFTS where required; A5 legitimately returns zero.
    const uint64_t ffts_addr = shmemx_get_ffts_config();

    LaunchPlanningKernel(
        topk_idx,
        dst,
        cu_seqlens,
        experts_to_copy,
        dup_groups,
        dup_loffs,
        dup_counts,
        tpe_all,
        tpe_prefix,
        balance,
        z,
        alloc,
        alloc_cumsum,
        expert_offsets,
        core_expert_counts,
        dup_workspace,
        dup_key_workspace,
        tiling,
        reinterpret_cast<void*>(static_cast<uintptr_t>(symmetric_workspace_addr)),
        ffts_addr);
}

}  // namespace ascend_moonep::ops
