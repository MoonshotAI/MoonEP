// MoonEP dispatch UDMA kernel with SIMT weight scatter — migrated from shmem_dev_moonep dispatch_moonep.
// Multi-QP UDMA path: scatter hidden via aclshmemx_udma_qp_put_nbi directly
// into hidden_buf; weights are scattered directly to the destination rank's
// weights_buf by SIMT threads. DispatchEpilogue expands duplicates in place on
// hidden_buf; QP signals use a dedicated sync_buffer.
//
// Token keeps group flow control; the last 16 non-Token AIVs run the Weight
// SIMT scatter in parallel or serially per the in-kernel constant.

#include "kernel_operator.h"
#include "ascend_moonep/kernels/types.h"
#include "ascend_moonep/kernels/launch.h"
#include "shmem.h"
#include "device/shmem_def.h"
#include "simt_api/device_sync_functions.h"

namespace DispatchMoonep {

constexpr uint32_t kUdmaWqeBatchSize = 128;
constexpr uint32_t kUdmaWqeBytes = 64;
// Perf path uses QP0/1/2/4 by runtime count; transport QP3/QP5 stay idle, slot 5 is the peer credit.
constexpr uint32_t kMaxCreatedUdmaQpCount = 6;
constexpr uint32_t kMaxActiveUdmaQpCount = 4;
constexpr uint32_t kActiveUdmaQpIndices[kMaxActiveUdmaQpCount] = {0, 1, 2, 4};
constexpr uint32_t kPeerCreditFlagSlot = 5;
static_assert(kActiveUdmaQpIndices[kMaxActiveUdmaQpCount - 1] < kMaxCreatedUdmaQpCount);
static_assert(kPeerCreditFlagSlot < kMaxCreatedUdmaQpCount);
static_assert(kPeerCreditFlagSlot > kActiveUdmaQpIndices[kMaxActiveUdmaQpCount - 1]);

// Data puts always use Relax Order: defer produces no cqe, submit/immediate does.
// CQEs from submit/immediate are reclaimed together at the current peer completion phase.
// Flag puts alone use Strong Order (see PublishPeerCompletionSignal).
// aclshmemx_udma_op_config_t is visible only in the device compile pass (shmem.h
// guards it with __CCE_AICORE__); these constants are used only inside __aicore__
// functions, so guard them identically to avoid host-pass instantiation failures.
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
inline constexpr aclshmemx_udma_op_config_t kUdmaDataRoNoCqeConfig{0U, 0U, 0U, ACLSHMEMX_UDMA_ODR_RO};
inline constexpr aclshmemx_udma_op_config_t kUdmaDataRoCqeConfig{1U, 0U, 0U, ACLSHMEMX_UDMA_ODR_RO};
inline constexpr aclshmemx_udma_op_config_t kUdmaSyncSoCqeConfig{1U, 0U, 0U, ACLSHMEMX_UDMA_ODR_SO};
#endif

// Profiling is controlled by the bool template parameter (cf. the UBMEM version
// kernels/moonep_dispatch.cpp on the ascend-moonep-timing branch):
// MoonEPDispatchUdmaKernelImpl<Profile> instantiates both kernels at once —
//   moonep_dispatch_udma_kernel          (Profile=false, zero timing overhead)
//   moonep_dispatch_udma_profile_kernel  (Profile=true, per-phase cycles + bytes)
// One build gives both launchers, selected at runtime; no rebuild to flip a switch.
// With Profile=true results go to GM profile_cycles_gm
// (int64 [kMoonEPAivNum, kMoonEPDispatchUdmaProfileFields]), read and printed by
// host; no AscendC::printf in the kernel, avoiding 32 AIV x R ranks of log spam.

// UB allocation
constexpr uint32_t kUdmaWqeScratchBytes = kUdmaWqeBatchSize * kUdmaWqeBytes;
// One SIMT thread block maps to one 512B weight GM conflict window.
constexpr uint32_t kWeightGmConflictGroupBytes = 512;
constexpr uint32_t kWeightsPerGmConflictGroup = kWeightGmConflictGroupBytes / sizeof(float);
constexpr uint32_t kWeightSimtThreadCount = kWeightsPerGmConflictGroup;
// Epilogue metadata tile stays a fixed 16KiB, independent of Weight slot size.
constexpr uint32_t kMteUbBytes          = 16 * 1024;
constexpr uint32_t kLocalCopyTileBytes  = 16 * 1024;
constexpr uint32_t slotChunkCount = 8192;
constexpr uint32_t kSlotTensorBytes = slotChunkCount * sizeof(int32_t);
constexpr uint32_t kSlotCompareMaskBytes = ((slotChunkCount / 8 + 31) / 32) * 32;
constexpr uint32_t kSlotSelectUbBytes = 4 * kSlotTensorBytes + kSlotCompareMaskBytes;
constexpr uint32_t kDuplicateHiddenStageBytes = 65504;
constexpr uint32_t kDuplicateHiddenStagingBytes = 2 * kDuplicateHiddenStageBytes;
constexpr uint32_t kDuplicateGroupFields = 3;
constexpr uint32_t kDuplicateGroupStagingBytes = kMteUbBytes;
constexpr uint32_t kDuplicateLoffStagingBytes = kLocalCopyTileBytes;
constexpr uint32_t kDuplicateGroupsPerTile =
    kDuplicateGroupStagingBytes / (kDuplicateGroupFields * sizeof(int32_t));
constexpr uint32_t kAivUbCapacityBytes = 192 * 1024;
constexpr uint32_t kMaxDispatchGroupWidth = 16;
// 128P production topology: two 64P frames, each laid out as 8 direct-link
// rows x 8 columns.  A red-box dispatch group is one 4-row x 4-column tile,
// so each frame contains two row blocks x two left/right sides = four groups.
constexpr uint32_t kDispatchFrameCount = 2;
constexpr uint32_t kDispatchRanksPerFrame = 64;
constexpr uint32_t kDispatchRanksPerDirectRow = 8;
constexpr uint32_t kDispatchRowsPerFrame = 8;
constexpr uint32_t kDispatchGroupRows = 4;
constexpr uint32_t kDispatchGroupColumns = 4;
constexpr uint32_t kDispatchGroupRowBlocks =
    kDispatchRowsPerFrame / kDispatchGroupRows;
constexpr uint32_t kDispatchGroupSides =
    kDispatchRanksPerDirectRow / kDispatchGroupColumns;
constexpr uint32_t kDispatchGroupsPerFrame =
    kDispatchGroupRowBlocks * kDispatchGroupSides;
constexpr uint32_t kTwoFrameDispatchWorldSize =
    kDispatchFrameCount * kDispatchRanksPerFrame;

enum class DispatchGroupSchedule : uint32_t {
    // Keep PR 44's original half-world mapping for every supported world size.
    Legacy = 0,
    // Four rounds to one frame, then four rounds to the other frame.
    Staged = 1,
    // Alternate local/remote frame on every round while retaining a bijective
    // source-group -> destination-group mapping.
    Alternating = 2,
};

// Compile-time schedule selection; changing it requires rebuilding the kernel.
// Staged/Alternating affect exactly 128P. Other supported sizes always retain
// the Legacy mapping. Default to Staged so 128P uses frame-local groups.
constexpr DispatchGroupSchedule kDispatchGroupSchedule =
    DispatchGroupSchedule::Staged;
constexpr int64_t kPeerSyncFlagBytes = 512;
constexpr int64_t kPeerSyncSlotBytes = kMaxCreatedUdmaQpCount * kPeerSyncFlagBytes;
constexpr int64_t kRankSyncSlotBytes = kMaxDispatchGroupWidth * kPeerSyncSlotBytes;
constexpr uint32_t kMaxDispatchAivCount = 32;
constexpr bool kEnableParallelTokenWeight = true;
constexpr int64_t kGroupSyncFlagBytes = 512;
constexpr uint32_t kGroupSyncRoundCount = 4;
constexpr uint32_t kGroupSyncPhasesPerGroup = 2;
constexpr uint32_t kGroupSyncCompletionPhase = 0;
constexpr uint32_t kGroupSyncDonePhase = 1;
constexpr int32_t kStrictGroupSyncMaxWorldSize = 64;
static_assert(kPeerSyncFlagBytes >= sizeof(uint64_t));
static_assert(kPeerSyncFlagBytes % sizeof(uint64_t) == 0);
static_assert(kGroupSyncFlagBytes >= sizeof(uint64_t));
static_assert(kGroupSyncFlagBytes % sizeof(uint64_t) == 0);
static_assert(kMaxDispatchGroupWidth % 2 == 0);
static_assert(kMaxDispatchAivCount > kMaxDispatchGroupWidth);
static_assert((1U << kGroupSyncRoundCount) >= kMaxDispatchGroupWidth);
static_assert(kDispatchGroupRows * kDispatchGroupColumns == kMaxDispatchGroupWidth);
static_assert(
    kDispatchRowsPerFrame * kDispatchRanksPerDirectRow == kDispatchRanksPerFrame);
static_assert(
    kDispatchGroupsPerFrame * kMaxDispatchGroupWidth == kDispatchRanksPerFrame);
static_assert(kDispatchGroupRowBlocks == 2);
static_assert(kDispatchGroupSides == 2);
static_assert(kDispatchGroupsPerFrame == 4);
static_assert(kTwoFrameDispatchWorldSize == 128);
static_assert(kWeightGmConflictGroupBytes % sizeof(float) == 0);
static_assert(kWeightSimtThreadCount == 128);
static_assert(kDuplicateHiddenStagingBytes <= kSlotSelectUbBytes);
// SIMT VF also needs the framework-reserved 8KiB immediate data UB.
static_assert(kUdmaWqeScratchBytes + kMteUbBytes + kLocalCopyTileBytes + kSlotSelectUbBytes <=
    kAivUbCapacityBytes - 8 * 1024);

// Alignment of the routes processed per chunk.
constexpr uint32_t kVectorSlotSelectCountAlign = 64;
// hidden is moved as raw BF16 bytes; row width and nominal byte count derive from DataType.
using DataType = uint16_t;
// Timing helpers gated by the Profile template parameter (cf. DispatchProfileStart/
// DispatchProfileAccumulate in the timing branch's UBMEM version): compile to no-op
// with zero hot-path cost when Profile=false; call GetSystemCycle and accumulate
// cycles only when Profile=true.
template <bool Profile>
__forceinline__ __aicore__ uint64_t UdmaProfileStart() {
    if constexpr (Profile) {
        return AscendC::GetSystemCycle();
    }
    return 0;
}

template <bool Profile>
__forceinline__ __aicore__ void UdmaProfileAccumulate(
    uint64_t start,
    uint64_t& cycles) {
    if constexpr (Profile) {
        cycles += AscendC::GetSystemCycle() - start;
    }
}

// Inner detailed timing switch (GetSystemCycle for wqe_submit / sync).
// Default false: outer brackets (token_scatter / weight_scatter / duplicate / total)
// stay free of inner GetSystemCycle overhead, so per-phase latency and the two bw
// figures are exact. Set true + rebuild to get non-zero per-core wqe_submit/sync
// breakdowns (outer brackets then include that overhead and read slightly high —
// cannot have both). Affects only GetSystemCycle calls; byte/packet counters stay
// gated by the outer Profile, so the data behind the two bw figures is unchanged.
constexpr bool kEnableDetailedProfile = false;

// Inner-detailed-only helpers: gate on Profile && kEnableDetailedProfile.
// Called only for wqe_submit / sync timing inside DispatchScatterToken.
template <bool Profile>
__forceinline__ __aicore__ uint64_t UdmaDetailedProfileStart() {
    if constexpr (Profile && kEnableDetailedProfile) {
        return AscendC::GetSystemCycle();
    }
    return 0;
}

template <bool Profile>
__forceinline__ __aicore__ void UdmaDetailedProfileAccumulate(
    uint64_t start,
    uint64_t& cycles) {
    if constexpr (Profile && kEnableDetailedProfile) {
        cycles += AscendC::GetSystemCycle() - start;
    }
}

// 2-QP perf experiment: per group of four chunks, the first three go to QP0 and the last to QP1.
__aicore__ inline uint32_t GetChunkQpSlot(int64_t chunk_index, uint32_t active_qp_count)
{
    // 3:1 weighted split only in 2-QP mode; other modes keep the uniform round-robin.
    if (active_qp_count == 2) {
        return chunk_index % 4 == 3 ? 1U : 0U;
    }
    return static_cast<uint32_t>(chunk_index % active_qp_count);
}

// Each (active AIV, QP) completion flag owns a dedicated 512B cacheline.
__aicore__ inline __gm__ uint64_t *GetPeerCompletionFlag(
    int32_t src_rank, uint32_t peer_rank_index, uint32_t qp_idx, __gm__ uint8_t *sync_buffer)
{
    return reinterpret_cast<__gm__ uint64_t *>(
        sync_buffer +
        static_cast<int64_t>(src_rank) * kRankSyncSlotBytes +
        static_cast<int64_t>(peer_rank_index) * kPeerSyncSlotBytes +
        qp_idx * kPeerSyncFlagBytes);
}

// Each active AIV's peer credit owns a dedicated 512B cacheline (slot 5).
__aicore__ inline __gm__ uint64_t *GetPeerCreditFlag(
    int32_t src_rank, uint32_t peer_rank_index, __gm__ uint8_t *sync_buffer)
{
    return reinterpret_cast<__gm__ uint64_t *>(
        sync_buffer +
        static_cast<int64_t>(src_rank) * kRankSyncSlotBytes +
        static_cast<int64_t>(peer_rank_index) * kPeerSyncSlotBytes +
        kPeerCreditFlagSlot * kPeerSyncFlagBytes);
}

// The credit encodes both kernel round and group generation, so async peers never consume a stale value.
__aicore__ inline uint64_t GetPeerCreditToken(uint64_t magic, uint32_t group, uint32_t group_count)
{
    return magic * (static_cast<uint64_t>(group_count) + 1U) + group + 1U;
}

// Each (active AIV, round) barrier flag owns a dedicated 512B region.
__aicore__ inline __gm__ uint64_t *GetGroupSyncFlag(
    uint32_t aiv_index, uint32_t round, __gm__ uint8_t *group_sync_buffer)
{
    const int64_t flag_index = static_cast<int64_t>(aiv_index) * kGroupSyncRoundCount + round;
    return reinterpret_cast<__gm__ uint64_t *>(group_sync_buffer + flag_index * kGroupSyncFlagBytes);
}

// Group sync tokens increase monotonically: init, then per-group completion, then per-group done.
__aicore__ inline uint64_t GetGroupSyncInitToken(uint64_t magic, uint32_t group_count)
{
    const uint64_t token_count = static_cast<uint64_t>(group_count) * kGroupSyncPhasesPerGroup + 1U;
    return magic * token_count + 1U;
}

__aicore__ inline uint64_t GetGroupSyncPhaseToken(
    uint64_t magic, uint32_t group, uint32_t group_count, uint32_t phase)
{
    const uint64_t token_count = static_cast<uint64_t>(group_count) * kGroupSyncPhasesPerGroup + 1U;
    const uint64_t group_token_offset =
        static_cast<uint64_t>(group) * kGroupSyncPhasesPerGroup + phase + 2U;
    return magic * token_count + group_token_offset;
}

// Active AIVs use a four-round dissemination barrier; every flag always has one writer and one reader.
__aicore__ inline void SyncActiveAivs(uint32_t aiv_index, uint32_t active_aiv_count,
    uint64_t sync_token, __gm__ uint8_t *group_sync_buffer)
{
    uint32_t distance = 1U;
    uint32_t round = 0U;
    // Reach doubles each round; 16 active AIVs finish strict sync in at most four rounds.
    while (distance < active_aiv_count) {
        const uint32_t next_aiv = (aiv_index + distance) % active_aiv_count;
        __gm__ uint64_t *next_flag = GetGroupSyncFlag(next_aiv, round, group_sync_buffer);
        *next_flag = sync_token;
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(next_flag));

        // Each AIV polls only its own dedicated flag; GE lets it safely consume a token already advanced to the next generation.
        aclshmem_uint64_wait_until(
            GetGroupSyncFlag(aiv_index, round, group_sync_buffer), ACLSHMEM_CMP_GE, sync_token);
        distance <<= 1U;
        ++round;
    }
}

// Below 16P all ranks form one group; larger sizes keep splitting into 16-rank groups.
__aicore__ inline uint32_t GetDispatchGroupWidth(int32_t world_size)
{
    return world_size < static_cast<int32_t>(kMaxDispatchGroupWidth) ?
        static_cast<uint32_t>(world_size) : kMaxDispatchGroupWidth;
}

// Group flow control requires an even rank count and an exact partition into equal-width
// groups; the kernel always launches 32 AIVs: the first group_width run Hidden, the
// rest run Weight in parallel with Hidden.
__aicore__ inline bool CanUseGroupFlowControl(int32_t world_size, int64_t aiv_count)
{
    const uint32_t group_width = GetDispatchGroupWidth(world_size);
    return world_size >= 2 && world_size % 2 == 0 &&
        static_cast<uint32_t>(world_size) % group_width == 0 &&
        aiv_count == static_cast<int64_t>(kMaxDispatchAivCount) &&
        static_cast<int64_t>(group_width) < aiv_count;
}

__aicore__ inline bool UseTwoFrameDispatchSchedule(int32_t world_size)
{
    return kDispatchGroupSchedule != DispatchGroupSchedule::Legacy &&
        world_size == static_cast<int32_t>(kTwoFrameDispatchWorldSize);
}

// Split the eight 128P rounds into "flip destination frame or not" and "in-frame
// subround". The two schedules only change how the two interleave:
//   Alternating: frame_toggle = 0,1,0,1,...; subround = 0,0,1,1,2,2,3,3.
//   Staged     : frame_toggle = 0,0,0,0,1,1,1,1; subround = 0,1,2,3,0,1,2,3.
// group is the logical send round (0..7), not the source/destination red-box group id.
// frame_toggle combines with source_group_side below; 0/1 alone is not local/remote.
// No global barrier here: the PR 44 completion/credit chain remains, AIVs advance async.
__aicore__ inline uint32_t GetDispatchFrameToggle(uint32_t group)
{
    if constexpr (kDispatchGroupSchedule == DispatchGroupSchedule::Alternating) {
        // Round parity is frame_toggle, so the destination frame alternates every round.
        return group % kDispatchFrameCount;
    }
    // frame_toggle switches every four rounds, so one destination frame takes four consecutive in-frame subrounds.
    return group / kDispatchGroupsPerFrame;
}

__aicore__ inline uint32_t GetDispatchFrameSubround(uint32_t group)
{
    if constexpr (kDispatchGroupSchedule == DispatchGroupSchedule::Alternating) {
        // Adjacent rounds share one subround; only frame_toggle changes.
        return group / kDispatchFrameCount;
    }
    // Enumerate subround 0..3 across four rounds, then repeat under the other frame_toggle.
    return group % kDispatchGroupsPerFrame;
}

__aicore__ inline int32_t BuildTwoFrameDispatchRank(
    uint32_t frame, uint32_t group_row, uint32_t group_side, uint32_t group_rank_index)
{
    const uint32_t lane_row = group_rank_index / kDispatchGroupColumns;
    const uint32_t lane_column = group_rank_index % kDispatchGroupColumns;
    const uint32_t row = group_row * kDispatchGroupRows + lane_row;
    const uint32_t column = group_side * kDispatchGroupColumns + lane_column;
    return static_cast<int32_t>(
        frame * kDispatchRanksPerFrame + row * kDispatchRanksPerDirectRow + column);
}

// Return the rank's index within the current 16-rank group. 128P uses the in-frame
// 4x4 red-box layout; other sizes keep PR 44's original cross-half-world group mapping.
__aicore__ inline uint32_t GetGroupRankIndex(int32_t rank, int32_t world_size)
{
    if (UseTwoFrameDispatchSchedule(world_size)) {
        const uint32_t local_rank = static_cast<uint32_t>(rank) % kDispatchRanksPerFrame;
        const uint32_t row = local_rank / kDispatchRanksPerDirectRow;
        const uint32_t column = local_rank % kDispatchRanksPerDirectRow;
        // Row-major flatten the rank's (row, column) offset inside the 4x4 red box to
        // 0..15; this index locates peer sync slots and matches the indexing used by
        // BuildTwoFrameDispatchRank.
        return (row % kDispatchGroupRows) * kDispatchGroupColumns +
            column % kDispatchGroupColumns;
    }

    const int32_t half_world_size = world_size / 2;
    const uint32_t half_group_width = GetDispatchGroupWidth(world_size) / 2;
    return static_cast<uint32_t>(rank / half_world_size) * half_group_width +
        static_cast<uint32_t>(rank % static_cast<int32_t>(half_group_width));
}

// Return the global dst rank this core actually targets, from the relative group and in-group index.
// 128P alternating policy:
//   left source group  : local, remote, local, remote, ...
//   right source group : remote, local, remote, local, ...
// For every round the transform is bijective, so every destination red-box
// group receives from exactly one source red-box group.
__aicore__ inline int32_t GetGroupDstRank(
    int32_t rank, int32_t world_size, uint32_t group, uint32_t group_rank_index)
{
    if (UseTwoFrameDispatchSchedule(world_size)) {
        const uint32_t source_frame = static_cast<uint32_t>(rank) / kDispatchRanksPerFrame;
        const uint32_t local_rank = static_cast<uint32_t>(rank) % kDispatchRanksPerFrame;
        const uint32_t source_row = local_rank / kDispatchRanksPerDirectRow;
        const uint32_t source_column = local_rank % kDispatchRanksPerDirectRow;
        const uint32_t source_group_row = source_row / kDispatchGroupRows;
        const uint32_t source_group_side = source_column / kDispatchGroupColumns;

        // frame_toggle: 0/1 frame-select phase; combine with source_group_side to tell local vs remote.
        // subround: in-frame target index 0..3, enumerating the four red-box groups in the destination frame.
        const uint32_t frame_toggle = GetDispatchFrameToggle(group);
        const uint32_t subround = GetDispatchFrameSubround(group);

        // Read subround as two bits: high bit row_delta, low bit side_delta.
        //
        //   subround (binary)  row_delta  side_delta  relative target group
        //          0 (00)         0           0       same row block, same side
        //          1 (01)         0           1       same row block, other side
        //          2 (10)         1           0       other row block, same side
        //          3 (11)         1           1       other row block, other side
        const uint32_t group_row_delta = subround / 2U;
        const uint32_t group_side_delta = subround % 2U;

        // source_frame, source_group_side and frame_toggle are all 0/1. The XOR is
        // not a hash but "flip the destination frame once per 1"; two 1s cancel:
        //
        //   source group side   toggle=0   toggle=1
        //   left (0)            local      remote
        //   right (1)           remote     local
        //
        // So each round the two sides take opposite local/remote directions, staggering
        // the use of the two link levels.
        const uint32_t destination_frame =
            source_frame ^ source_group_side ^ frame_toggle;
        // row_delta=0 keeps the row block; row_delta=1 swaps between the two row blocks.
        const uint32_t destination_group_row =
            (source_group_row + group_row_delta) % kDispatchGroupRowBlocks;
        // side_delta=0 keeps the side; side_delta=1 swaps between left and right groups.
        const uint32_t destination_group_side = source_group_side ^ group_side_delta;

        return BuildTwoFrameDispatchRank(
            destination_frame, destination_group_row, destination_group_side, group_rank_index);
    }

    const int32_t half_world_size = world_size / 2;
    const uint32_t group_width = GetDispatchGroupWidth(world_size);
    const uint32_t half_group_width = group_width / 2;
    const uint32_t group_count = static_cast<uint32_t>(world_size) / group_width;
    const uint32_t rank_group = static_cast<uint32_t>(rank % half_world_size) /
        half_group_width;
    const uint32_t peer_group = (rank_group + group) % group_count;
    return static_cast<int32_t>(group_rank_index / half_group_width) * half_world_size +
        static_cast<int32_t>(peer_group * half_group_width + group_rank_index % half_group_width);
}

// Return the real source rank that sends to this rank in the current group.
// The 128P path is the exact inverse of GetGroupDstRank, keeping completion/credit slots 1:1.
__aicore__ inline int32_t GetGroupIncomingPeer(
    int32_t rank, int32_t world_size, uint32_t group, uint32_t group_rank_index)
{
    if (UseTwoFrameDispatchSchedule(world_size)) {
        const uint32_t destination_frame = static_cast<uint32_t>(rank) / kDispatchRanksPerFrame;
        const uint32_t local_rank = static_cast<uint32_t>(rank) % kDispatchRanksPerFrame;
        const uint32_t destination_row = local_rank / kDispatchRanksPerDirectRow;
        const uint32_t destination_column = local_rank % kDispatchRanksPerDirectRow;
        const uint32_t destination_group_row = destination_row / kDispatchGroupRows;
        const uint32_t destination_group_side = destination_column / kDispatchGroupColumns;

        // Use the same round decoding as the forward map, then undo side, row, frame in order.
        const uint32_t frame_toggle = GetDispatchFrameToggle(group);
        const uint32_t subround = GetDispatchFrameSubround(group);
        const uint32_t group_row_delta = subround / 2U;
        const uint32_t group_side_delta = subround % 2U;
        // XOR is its own inverse: x ^ b ^ b == x. Recover the source group side first,
        // then apply the same formula backwards to derive the source frame.
        const uint32_t source_group_side = destination_group_side ^ group_side_delta;
        // Undo the cyclic shift applied to the two row blocks in GetGroupDstRank.
        const uint32_t source_group_row =
            (destination_group_row + kDispatchGroupRowBlocks - group_row_delta) %
            kDispatchGroupRowBlocks;
        const uint32_t source_frame =
            destination_frame ^ source_group_side ^ frame_toggle;

        return BuildTwoFrameDispatchRank(
            source_frame, source_group_row, source_group_side, group_rank_index);
    }

    const int32_t half_world_size = world_size / 2;
    const uint32_t group_width = GetDispatchGroupWidth(world_size);
    const uint32_t half_group_width = group_width / 2;
    const uint32_t group_count = static_cast<uint32_t>(world_size) / group_width;

    const uint32_t rank_group = static_cast<uint32_t>(rank % half_world_size) / half_group_width;
    const uint32_t incoming_group = (rank_group + group_count - group % group_count) % group_count;

    return static_cast<int32_t>(group_rank_index / half_group_width) * half_world_size +
        static_cast<int32_t>(incoming_group * half_group_width + group_rank_index % half_group_width);
}

// After the current group completes, send credit to the source rank that will send to this rank in the next relative group.
__aicore__ inline int32_t GetNextGroupCreditPeer(
    int32_t rank, int32_t world_size, uint32_t group, uint32_t group_rank_index)
{
    return GetGroupIncomingPeer(rank, world_size, group + 1U, group_rank_index);
}

__aicore__ inline void CopyLocalBytes(GM_ADDR destination_gm, GM_ADDR source_gm, int64_t bytes,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &copy_buffer)
{
    AscendC::LocalTensor<uint8_t> local_tensor = copy_buffer.Get<uint8_t>();
    AscendC::GlobalTensor<uint8_t> source_tensor;
    AscendC::GlobalTensor<uint8_t> destination_tensor;
    source_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(source_gm), bytes);
    destination_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(destination_gm), bytes);

    for (int64_t copied_bytes = 0; copied_bytes < bytes; copied_bytes += kLocalCopyTileBytes) {
        int64_t current_tile_bytes = bytes - copied_bytes;
        if (current_tile_bytes > static_cast<int64_t>(kLocalCopyTileBytes)) {
            current_tile_bytes = kLocalCopyTileBytes;
        }
        AscendC::DataCopyPadExtParams<uint8_t> pad_parameters {false, 0, 0, 0};
        AscendC::DataCopyExtParams copy_parameters {
            1, static_cast<uint32_t>(current_tile_bytes), 0, 0, 0};
        AscendC::DataCopyPad(local_tensor, source_tensor[copied_bytes], copy_parameters, pad_parameters);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(destination_tensor[copied_bytes], local_tensor, copy_parameters);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void DispatchSplitRange(int64_t total, int64_t part_count, int64_t part_index,
    int64_t &range_begin, int64_t &range_end)
{
    if (total <= 0 || part_count <= 0 || part_index < 0 || part_index >= part_count) {
        range_begin = 0;
        range_end = 0;
        return;
    }
    const int64_t base_count = total / part_count;
    const int64_t remainder = total - base_count * part_count;
    range_begin = part_index * base_count + (part_index < remainder ? part_index : remainder);
    range_end = range_begin + base_count + (part_index < remainder ? 1 : 0);
}

// Split weights per 512B GM conflict window so adjacent AIVs never share one 128-float route group.
__aicore__ inline void DispatchSplitWeightRange(int64_t total, int64_t part_count, int64_t part_index,
    int64_t &range_begin, int64_t &range_end)
{
    const int64_t group_size = static_cast<int64_t>(kWeightsPerGmConflictGroup);
    const int64_t group_count = total > 0 ? total / group_size + (total % group_size != 0 ? 1 : 0) : 0;
    int64_t group_begin = 0;
    int64_t group_end = 0;
    DispatchSplitRange(group_count, part_count, part_index, group_begin, group_end);
    range_begin = group_begin * group_size;
    range_end = group_end * group_size;
    if (range_begin > total) {
        range_begin = total;
    }
    if (range_end > total) {
        range_end = total;
    }
}

__aicore__ inline void CopyGmInt32ToUb(AscendC::LocalTensor<int32_t> destination_ub,
    __gm__ int32_t *source_gm, uint32_t element_count)
{
    AscendC::GlobalTensor<int32_t> source_tensor;
    source_tensor.SetGlobalBuffer(source_gm, element_count);
    AscendC::DataCopyParams copy_parameters {
        1, static_cast<uint16_t>(element_count * sizeof(int32_t)), 0, 0};
    AscendC::DataCopyPad(
        destination_ub, source_tensor, copy_parameters, AscendC::DataCopyPadParams {});
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID2);
}

__aicore__ inline void CopyGmBytesToUb(AscendC::LocalTensor<uint8_t> destination_ub,
    GM_ADDR source_gm, uint32_t bytes, AscendC::TEventID event_id)
{
    AscendC::GlobalTensor<uint8_t> source_tensor;
    source_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(source_gm), bytes);
    AscendC::DataCopyPadExtParams<uint8_t> pad_parameters {false, 0, 0, 0};
    AscendC::DataCopyExtParams copy_parameters {1, bytes, 0, 0, 0};
    AscendC::DataCopyPad(destination_ub, source_tensor, copy_parameters, pad_parameters);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
}

__aicore__ inline void IssueUbBytesToGm(GM_ADDR destination_gm,
    AscendC::LocalTensor<uint8_t> source_ub, uint32_t bytes)
{
    AscendC::GlobalTensor<uint8_t> destination_tensor;
    destination_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(destination_gm), bytes);
    AscendC::DataCopyExtParams copy_parameters {1, bytes, 0, 0, 0};
    AscendC::DataCopyPad(destination_tensor, source_ub, copy_parameters);
}

// Each Token AIV initializes its own completion/credit cachelines, then does one local strict sync.
__aicore__ inline void PrepareTokenScatterPeerSignals(int32_t current_rank,
    uint32_t group_rank_index, uint64_t magic, uint32_t active_qp_count,
    uint32_t active_aiv_count, uint32_t group_count,
    __gm__ uint8_t *sync_buffer, __gm__ uint8_t *group_sync_buffer)
{
    // Each active QP gets its own completion cacheline.
    for (uint32_t i = 0; i < active_qp_count; ++i) {
        const uint32_t qp_idx = kActiveUdmaQpIndices[i];
        __gm__ uint64_t *signal_flag = GetPeerCompletionFlag(current_rank, group_rank_index, qp_idx, sync_buffer);
        *signal_flag = magic;
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(signal_flag));
    }
    __gm__ uint64_t *credit_flag = GetPeerCreditFlag(current_rank, group_rank_index, sync_buffer);
    *credit_flag = 0;
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(credit_flag));

    SyncActiveAivs(group_rank_index, active_aiv_count,
        GetGroupSyncInitToken(magic, group_count), group_sync_buffer);
}

// Each active AIV polls only the credit its current target peer published for this source rank.
__aicore__ inline void WaitPeerCredit(uint32_t group, int32_t dst_rank,
    int32_t current_rank, uint32_t current_rank_index, uint64_t magic,
    uint32_t group_count, __gm__ uint8_t *sync_buffer)
{
    // self consumes no remote UDMA receive resources, so no remote credit wait is needed.
    if (dst_rank == current_rank) {
        return;
    }
    const uint64_t credit_token = GetPeerCreditToken(magic, group, group_count);
    aclshmem_uint64_wait_until(
        GetPeerCreditFlag(dst_rank, current_rank_index, sync_buffer), ACLSHMEM_CMP_EQ, credit_token);
}

// The receiver polls only the dedicated completion cachelines of the current incoming peer.
__aicore__ inline void WaitPeerCompletionSignal(int32_t incoming_peer,
    uint32_t current_rank_index, uint64_t magic, uint32_t active_qp_count,
    __gm__ uint8_t *sync_buffer)
{
    for (uint32_t i = 0; i < active_qp_count; ++i) {
        const uint32_t qp_idx = kActiveUdmaQpIndices[i];
        aclshmem_uint64_wait_until(
            GetPeerCompletionFlag(incoming_peer, current_rank_index, qp_idx, sync_buffer),
            ACLSHMEM_CMP_EQ, magic);
    }
}

// This AIV appends SO on every active QP of the current dst using its own completion cachelines.
__aicore__ inline void PublishPeerCompletionSignal(int32_t dst_rank,
    int32_t current_rank, uint32_t group_rank_index, uint32_t active_qp_count,
    __gm__ uint8_t *sync_buffer, __ubuf__ uint8_t *udma_wqe_buffer)
{
    // Append one SO per active QP so that QP's RO puts become visible to the destination peer.
    for (uint32_t i = 0; i < active_qp_count; ++i) {
        const uint32_t qp_idx = kActiveUdmaQpIndices[i];
        __gm__ uint64_t *signal_flag = GetPeerCompletionFlag(
            current_rank, group_rank_index, qp_idx, sync_buffer);

        // UDMA has no self put; local data is already done by this AIV.
        if (dst_rank == current_rank) {
            continue;
        }
        aclshmemx_udma_qp_put_nbi<uint64_t, PIPE_MTE3, kUdmaSyncSoCqeConfig>(
            signal_flag, signal_flag, reinterpret_cast<__ubuf__ uint64_t *>(udma_wqe_buffer),
            1, dst_rank, qp_idx, EVENT_ID0);
    }
}

// After seeing completion, this AIV grants credit to the next incoming peer via its own credit cacheline.
__aicore__ inline void PublishNextPeerCredit(uint32_t group,
    int32_t current_rank, int32_t world_size, uint32_t group_rank_index,
    uint64_t magic, uint32_t group_count, __gm__ uint8_t *sync_buffer,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &local_copy_buffer)
{
    const int32_t next_peer = GetNextGroupCreditPeer(
        current_rank, world_size, group, group_rank_index);
    // self's local copy uses no remote receive resources; no cross-rank credit publish is needed.
    if (next_peer == current_rank) {
        return;
    }

    constexpr uint32_t credit_word_count = kPeerSyncFlagBytes / sizeof(uint64_t);
    const uint64_t next_credit_token = GetPeerCreditToken(magic, group + 1U, group_count);
    // Reuse the local copy UB to build the credit cacheline, avoiding a dedicated buffer.
    AscendC::LocalTensor<uint64_t> credit_local = local_copy_buffer.Get<uint64_t>();
    // Send the full 512B cacheline to avoid read/write conflicts with neighboring peers' sync bits during remote polling.
    for (uint32_t i = 0; i < credit_word_count; ++i) {
        credit_local.SetValue(i, i == 0U ? next_credit_token : 0U);
    }
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);

    __gm__ uint64_t *local_credit = GetPeerCreditFlag(current_rank, group_rank_index, sync_buffer);
    __gm__ uint64_t *remote_credit = reinterpret_cast<__gm__ uint64_t *>(
        aclshmem_ptr(local_credit, next_peer));
    AscendC::GlobalTensor<uint64_t> remote_credit_tensor;
    remote_credit_tensor.SetGlobalBuffer(remote_credit, credit_word_count);
    AscendC::DataCopy(remote_credit_tensor, credit_local, credit_word_count);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

// Each active AIV completes its current peer; at <=64P all lock-step at completion and group end.
__aicore__ inline void CompleteTokenScatterPeerWithSignal(uint32_t group, int32_t dst_rank,
    int32_t current_rank, int32_t world_size, uint32_t group_rank_index,
    uint32_t current_rank_index, uint64_t magic, uint32_t active_qp_count, bool enable_strict_group_sync,
    __gm__ uint8_t *sync_buffer, __gm__ uint8_t *group_sync_buffer,
    __ubuf__ uint8_t *udma_wqe_buffer,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &local_copy_buffer)
{
    const uint32_t active_aiv_count = GetDispatchGroupWidth(world_size);
    const uint32_t group_count = static_cast<uint32_t>(world_size) / active_aiv_count;
    PublishPeerCompletionSignal(
        dst_rank, current_rank, group_rank_index, active_qp_count, sync_buffer, udma_wqe_buffer);

    // This AIV's index determines the incoming source peer; the current rank's in-group
    // index determines the completion slot the remote sender uses.
    const int32_t incoming_peer = GetGroupIncomingPeer(current_rank, world_size, group, group_rank_index);
    WaitPeerCompletionSignal(
        incoming_peer, current_rank_index, magic, active_qp_count, sync_buffer);

    // At <=64P, wait for all of this rank's Token AIVs before any may publish the next-round credit.
    if (enable_strict_group_sync) {
        SyncActiveAivs(group_rank_index, active_aiv_count,
            GetGroupSyncPhaseToken(magic, group, group_count, kGroupSyncCompletionPhase),
            group_sync_buffer);
    }

    // The last group starts no next round, so only the current incoming SO acknowledgement is needed.
    if (group + 1U < group_count) {
        PublishNextPeerCredit(
            group, current_rank, world_size, group_rank_index, magic, group_count,
            sync_buffer, local_copy_buffer);
    }

    // The self path has no UDMA WQE and never touches self QP context.
    if (dst_rank != current_rank) {
        // Incoming completion is done and the next peer released when needed; now reclaim
        // all CQEs of the current outgoing peer at once.
        for (uint32_t qp_i = 0; qp_i < active_qp_count; ++qp_i) {
            aclshmemx_udma_qp_quiet(dst_rank, kActiveUdmaQpIndices[qp_i]);
        }
    }

    // At <=64P, align credit publish and CQE reclaim at the group tail; larger sizes keep AIVs advancing async.
    if (enable_strict_group_sync) {
        SyncActiveAivs(group_rank_index, active_aiv_count,
            GetGroupSyncPhaseToken(magic, group, group_count, kGroupSyncDonePhase),
            group_sync_buffer);
    }
}

// Enable chunked vector rank select only when the encoded slot count satisfies the shift-decode constraint.
__aicore__ inline bool CanUseVectorSlotSelect(int64_t route_count)
{
    if (route_count <= 0 || (route_count & (route_count - 1)) != 0) {
        return false;
    }
    uint32_t slot_shift = 0;
    for (int64_t value = route_count; value > 1; value >>= 1) {
        ++slot_shift;
    }
    return slot_shift <= 31;
}

// Phase 1: vector-select the primary dst and batch-issue token WQEs.
// template <bool Profile>, matching the UBMEM version, replaces the constexpr switch;
// with Profile=false the GetSystemCycle timing compiles to no-op, counters come via reference params.
template <bool Profile>
__aicore__ inline uint64_t DispatchScatterToken(
    GM_ADDR hidden_gm,
    GM_ADDR dst_gm,
    GM_ADDR hidden_buf_gm,
    GM_ADDR sync_buffer_gm,
    int64_t num_tokens, int64_t hidden_dim, int64_t num_topk, uint64_t magic,
    uint32_t active_qp_count,
    uint64_t &wqe_submit_cycles,
    uint64_t &wqe_submit_packets,
    uint64_t &sync_cycles,
    __ubuf__ uint8_t *udma_wqe_buffer,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &local_copy_buffer,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &slot_select_buffer)
{
    __gm__ uint8_t *hidden = reinterpret_cast<__gm__ uint8_t *>(hidden_gm);
    __gm__ int32_t *encoded_destinations = reinterpret_cast<__gm__ int32_t *>(dst_gm);
    const int64_t route_count = num_tokens * num_topk;
    const int64_t hidden_row_bytes = hidden_dim * sizeof(DataType);
    __gm__ uint8_t *hidden_window = reinterpret_cast<__gm__ uint8_t *>(hidden_buf_gm);
    __gm__ uint8_t *sync_buffer = reinterpret_cast<__gm__ uint8_t *>(sync_buffer_gm);
    const int64_t aiv_index = AscendC::GetBlockIdx();
    const int32_t current_rank = aclshmem_my_pe();
    const int32_t world_size = aclshmem_n_pes();
    const uint32_t group_width = GetDispatchGroupWidth(world_size);
    const uint32_t group_count = static_cast<uint32_t>(world_size) / group_width;
    // The first 32KiB after the Token control region holds this rank's local sync flags.
    __gm__ uint8_t *group_sync_buffer =
        sync_buffer + static_cast<uint64_t>(world_size) * kRankSyncSlotBytes;
    const uint32_t group_rank_index = static_cast<uint32_t>(aiv_index) % group_width;
    const uint32_t current_rank_index = GetGroupRankIndex(current_rank, world_size);
    // Rank count is fixed for the kernel run, so compute the strict group sync switch once.
    const bool enable_strict_group_sync = world_size <= kStrictGroupSyncMaxWorldSize;
    uint64_t transferred_bytes = 0;

    // Step 0: Token AIVs init their dedicated signals and start issuing after the local strict sync.
    PrepareTokenScatterPeerSignals(
        current_rank, group_rank_index, magic, active_qp_count,
        group_width, group_count, sync_buffer, group_sync_buffer);

    if (CanUseVectorSlotSelect(route_count)) {
        // Step 1: load dst into UB in chunks, keeping a separate index region for select results so raw dst is not overwritten.
        const uint32_t tensor_bytes = slotChunkCount * sizeof(int32_t);
        const uint32_t compare_mask_bytes = ((slotChunkCount / 8 + 31) / 32) * 32;
        AscendC::LocalTensor<int32_t> dst_local =
            slot_select_buffer.GetWithOffset<int32_t>(slotChunkCount, 0);
        AscendC::LocalTensor<int32_t> dst_rank_tensor =
            slot_select_buffer.GetWithOffset<int32_t>(slotChunkCount, tensor_bytes);
        AscendC::LocalTensor<int32_t> route_indices =
            slot_select_buffer.GetWithOffset<int32_t>(slotChunkCount, tensor_bytes * 2);
        AscendC::LocalTensor<int32_t> selected_route_indices =
            slot_select_buffer.GetWithOffset<int32_t>(slotChunkCount, tensor_bytes * 3);
        AscendC::LocalTensor<uint8_t> compare_mask =
            slot_select_buffer.GetWithOffset<uint8_t>(compare_mask_bytes, tensor_bytes * 4);

        AscendC::GlobalTensor<int32_t> dst_global;
        dst_global.SetGlobalBuffer(encoded_destinations, route_count);
        uint32_t slot_shift = 0;
        for (int64_t value = route_count; value > 1; value >>= 1) {
            ++slot_shift;
        }

        // Step 2: each active AIV walks the peer sequence independently; group 0 sends directly, later peers wait for the dedicated credit.
        for (uint32_t group = 0; group < group_count; ++group) {
            const int32_t dst_rank = GetGroupDstRank(current_rank, world_size, group, group_rank_index);

            // Later peers consume only the current-generation credit the destination published for this source rank.
            if (group > 0U) {
                WaitPeerCredit(
                    group, dst_rank, current_rank, current_rank_index, magic, group_count, sync_buffer);
            }

            // This AIV finishes all chunks of the current peer before moving to its next peer.
            for (int64_t chunk_start = 0; chunk_start < route_count; chunk_start += slotChunkCount) {
                const int64_t chunk_index = chunk_start / static_cast<int64_t>(slotChunkCount);
                const uint32_t qp_slot = GetChunkQpSlot(chunk_index, active_qp_count);
                const uint32_t qp_idx = kActiveUdmaQpIndices[qp_slot];

                const int64_t remaining_count = route_count - chunk_start;
                const uint32_t chunk_count = static_cast<uint32_t>(
                    remaining_count > static_cast<int64_t>(slotChunkCount) ? slotChunkCount : remaining_count);
                const uint32_t padded_count =
                    ((chunk_count + kVectorSlotSelectCountAlign - 1) /
                    kVectorSlotSelectCountAlign) * kVectorSlotSelectCountAlign;

                AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID1);
                AscendC::DataCopyExtParams dst_copy_parameters {
                    1, static_cast<uint32_t>(chunk_count * sizeof(int32_t)), 0, 0, 0};
                AscendC::DataCopyPadExtParams<int32_t> dst_pad_parameters {false, 0, 0, 0};
                AscendC::DataCopyPad(
                    dst_local, dst_global[chunk_start], dst_copy_parameters, dst_pad_parameters);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);

                // A negative dst cannot shift-match a non-negative rank, so the list contains primary routes only.
                AscendC::ShiftRight(
                    dst_rank_tensor, dst_local, static_cast<int32_t>(slot_shift), padded_count);
                AscendC::CreateVecIndex(
                    route_indices, static_cast<int32_t>(chunk_start), padded_count);
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID1);

                AscendC::Compares(compare_mask, dst_rank_tensor, dst_rank, AscendC::CMPMODE::EQ, padded_count);
                AscendC::PipeBarrier<PIPE_V>();
                uint64_t selected_route_count = 0;
                AscendC::GatherMask(selected_route_indices, route_indices,
                    compare_mask.ReinterpretCast<uint32_t>(), true, chunk_count,
                    {1U, 1U, 0U, 0U}, selected_route_count);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID1);

                // One fixed sender handles the full select list per peer, preserving the global QP traffic ratio and full WQE batches.
                const uint64_t local_selected_count = selected_route_count;
                // This core's byte count accumulates the primary routes it actually owns, covering both local copies and remote sends.
                transferred_bytes += local_selected_count * static_cast<uint64_t>(hidden_row_bytes);

                // Step 4: one QP for the whole chunk; the select list only generates and batch-submits WQEs.
                const bool profile_wqe_submit = dst_rank != current_rank && local_selected_count > 0;
                uint64_t wqe_submit_start_cycle = UdmaDetailedProfileStart<Profile>();
                if constexpr (Profile && kEnableDetailedProfile) {
                    // Count only remote WQE generation/submission; local copies are not on the send path.
                    if (profile_wqe_submit) {
                        wqe_submit_start_cycle = AscendC::GetSystemCycle();
                    }
                }

                // dst_rank is constant across the select list; branch once instead of per route.
                if (dst_rank == current_rank) {
                    // Local copy. Could be folded into the duplicate phase, but kept separate:
                    // single-core UB copies there would drag bandwidth down.
                    for (uint64_t selected = 0; selected < local_selected_count; ++selected) {
                        const int64_t route_index = selected_route_indices.GetValue(selected);
                        const int64_t token_index = route_index / num_topk;
                        // Output position is always decoded from the current route's real dst.
                        const int64_t output_offset =
                            dst_local.GetValue(route_index - chunk_start) % route_count;
                        __gm__ uint8_t *hidden_destination =
                            hidden_window + output_offset * hidden_row_bytes;
                        // The hidden memory type depends on host allocation; the kernel always reads the real token row.
                        __gm__ uint8_t *hidden_source = hidden + token_index * hidden_row_bytes;
                        CopyLocalBytes(reinterpret_cast<GM_ADDR>(hidden_destination),
                            reinterpret_cast<GM_ADDR>(hidden_source), hidden_row_bytes, local_copy_buffer);
                    }
                } else {
                    aclshmemx_submit_state_t udma_submit_state{};
                    aclshmemx_defer_t defer_action(udma_submit_state);
                    aclshmemx_submit_t submit_action(udma_submit_state);

                    // The first N-1 entries of each batch only fill WQEs; the last always submits, removing the per-WQE submit check.
                    for (uint64_t batch_start = 0; batch_start < local_selected_count; batch_start += kUdmaWqeBatchSize) {
                        const uint64_t batch_end = local_selected_count - batch_start > kUdmaWqeBatchSize ?
                            batch_start + kUdmaWqeBatchSize : local_selected_count;

                        // Defer every entry of the current batch except the last.
                        for (uint64_t selected = batch_start; selected + 1 < batch_end; ++selected) {
                            const int64_t route_index = selected_route_indices.GetValue(selected);
                            const int64_t token_index = route_index / num_topk;
                            const int64_t output_offset =
                                dst_local.GetValue(route_index - chunk_start) % route_count;
                            __gm__ uint8_t *hidden_destination =
                                hidden_window + output_offset * hidden_row_bytes;
                            // The hidden memory type depends on host allocation; the kernel always reads the real token row.
                            __gm__ uint8_t *hidden_source = hidden + token_index * hidden_row_bytes;
                            aclshmemx_udma_qp_put_nbi<uint8_t, PIPE_MTE3, kUdmaDataRoNoCqeConfig>(hidden_destination, hidden_source,
                                udma_wqe_buffer, static_cast<uint32_t>(hidden_row_bytes), dst_rank, qp_idx,
                                EVENT_ID0, defer_action);
                        }

                        // Once the last entry of the current batch is filled, submit the whole batch immediately.
                        const uint64_t selected = batch_end - 1;
                        const int64_t route_index = selected_route_indices.GetValue(selected);
                        const int64_t token_index = route_index / num_topk;
                        const int64_t output_offset =
                            dst_local.GetValue(route_index - chunk_start) % route_count;
                        __gm__ uint8_t *hidden_destination =
                            hidden_window + output_offset * hidden_row_bytes;
                        // The hidden memory type depends on host allocation; the kernel always reads the real token row.
                        __gm__ uint8_t *hidden_source = hidden + token_index * hidden_row_bytes;
                        aclshmemx_udma_qp_put_nbi<uint8_t, PIPE_MTE3, kUdmaDataRoCqeConfig>(hidden_destination, hidden_source,
                            udma_wqe_buffer, static_cast<uint32_t>(hidden_row_bytes), dst_rank, qp_idx,
                            EVENT_ID0, submit_action);
                    }
                }
                if constexpr (Profile) {
                    // selected_route_count is also this round's remote WQE count.
                    if (profile_wqe_submit) {
                        wqe_submit_packets += local_selected_count;
                    }
                }
                if constexpr (Profile && kEnableDetailedProfile) {
                    if (profile_wqe_submit) {
                        wqe_submit_cycles += AscendC::GetSystemCycle() - wqe_submit_start_cycle;
                    }
                }
            }

            // Step 5: append completion SO on the same dst/QP; after waiting the incoming
            // completion and forwarding the next-round credit, reclaim CQEs together.
            uint64_t sync_start_cycle = UdmaDetailedProfileStart<Profile>();
            CompleteTokenScatterPeerWithSignal(group, dst_rank,
                current_rank, world_size, group_rank_index, current_rank_index,
                magic, active_qp_count, enable_strict_group_sync, sync_buffer, group_sync_buffer,
                udma_wqe_buffer, local_copy_buffer);
            UdmaDetailedProfileAccumulate<Profile>(sync_start_cycle, sync_cycles);
        }
        return transferred_bytes;
    }

    // Non-power-of-two route_count cannot extract rank by shift; the scalar fallback keeps the same peer flow control.
    // The scalar path likewise has each active AIV walk its target peers independently.
    for (uint32_t group = 0; group < group_count; ++group) {
        const int32_t dst_rank = GetGroupDstRank(current_rank, world_size, group, group_rank_index);
        // Later peers must first consume the dedicated credit the destination forwarded after finishing the previous incoming round.
        if (group > 0U) {
            WaitPeerCredit(
                group, dst_rank, current_rank, current_rank_index, magic, group_count, sync_buffer);
        }

        for (int64_t route_index = 0; route_index < route_count; ++route_index) {
            const int64_t encoded_destination = encoded_destinations[route_index];
            if (encoded_destination < 0) {
                continue;
            }
            const int64_t flat_destination = encoded_destination;
            const int32_t route_dst_rank = static_cast<int32_t>(flat_destination / route_count);
            if (route_dst_rank != dst_rank) {
                continue;
            }

            const int64_t destination_local_offset = flat_destination % route_count;
            const int64_t token_index = route_index / num_topk;
            const uint32_t qp_slot = GetChunkQpSlot(route_index, active_qp_count);
            const uint32_t qp_idx = kActiveUdmaQpIndices[qp_slot];
            // Scalar path: once the target rank owned by this core is hit, accumulate the bytes this core actually handles.
            transferred_bytes += static_cast<uint64_t>(hidden_row_bytes);

            __gm__ uint8_t *hidden_destination =
                hidden_window + destination_local_offset * hidden_row_bytes;
            __gm__ uint8_t *hidden_source = hidden + token_index * hidden_row_bytes;
            if (route_dst_rank == current_rank) {
                CopyLocalBytes(reinterpret_cast<GM_ADDR>(hidden_destination),
                    reinterpret_cast<GM_ADDR>(hidden_source), hidden_row_bytes, local_copy_buffer);
            } else {
                uint64_t wqe_submit_start_cycle = UdmaDetailedProfileStart<Profile>();
                aclshmemx_udma_qp_put_nbi<uint8_t, PIPE_MTE3, kUdmaDataRoCqeConfig>(hidden_destination, hidden_source,
                    udma_wqe_buffer, static_cast<uint32_t>(hidden_row_bytes), route_dst_rank, qp_idx, EVENT_ID0);
                if constexpr (Profile) {
                    ++wqe_submit_packets;
                }
                if constexpr (Profile && kEnableDetailedProfile) {
                    wqe_submit_cycles += AscendC::GetSystemCycle() - wqe_submit_start_cycle;
                }
            }
        }

        // The scalar path keeps the same two-phase sync: completion SO plus dedicated credit.
        uint64_t sync_start_cycle = UdmaDetailedProfileStart<Profile>();
        CompleteTokenScatterPeerWithSignal(group, dst_rank,
            current_rank, world_size, group_rank_index, current_rank_index,
            magic, active_qp_count, enable_strict_group_sync, sync_buffer, group_sync_buffer,
            udma_wqe_buffer, local_copy_buffer);
        UdmaDetailedProfileAccumulate<Profile>(sync_start_cycle, sync_cycles);
    }
    return transferred_bytes;
}

// Each SIMT thread owns one weight in the conflict window and writes it directly to the destination rank's final output.
__simt_vf__ __launch_bounds__(kWeightSimtThreadCount) inline void
DispatchScatterWeightsSimtVf(__gm__ float *weights, __gm__ int32_t *encoded_destinations,
    __gm__ float *weight_window, uint32_t route_begin, uint32_t local_route_count,
    uint32_t route_count, uint32_t route_count_is_power_of_two, uint32_t route_shift,
    uint32_t first_group)
{
    const uint32_t thread_index = static_cast<uint32_t>(threadIdx.x);
    const uint32_t local_group_count =
        (local_route_count + kWeightsPerGmConflictGroup - 1U) / kWeightsPerGmConflictGroup;
    // Threads round-robin over 512B groups, staggering how ranks/AIVs hit destination GM conflict windows.
    for (uint32_t group_order = 0; group_order < local_group_count; ++group_order) {
        uint32_t group_index = first_group + group_order;
        // Wrap back to the first group once the index passes the local range.
        if (group_index >= local_group_count) {
            group_index -= local_group_count;
        }
        const uint32_t local_route_index = group_index * kWeightsPerGmConflictGroup + thread_index;
        // When the tail has fewer than 128 routes, idle threads touch neither input nor output.
        if (local_route_index >= local_route_count) {
            continue;
        }

        const uint32_t route_index = route_begin + local_route_index;
        const uint32_t encoded_bits = static_cast<uint32_t>(encoded_destinations[route_index]);
        const uint32_t sign_mask = 0U - (encoded_bits >> 31U);
        const uint32_t flat_destination = encoded_bits ^ sign_mask;
        uint32_t destination_rank = 0;
        uint32_t destination_local_offset = 0;
        // Power-of-two route_count decodes with shifts; other sizes keep division semantics.
        if (route_count_is_power_of_two != 0U) {
            destination_rank = flat_destination >> route_shift;
            destination_local_offset = flat_destination & (route_count - 1U);
        } else {
            destination_rank = flat_destination / route_count;
            destination_local_offset = flat_destination - destination_rank * route_count;
        }

        // dst is globally unique, so no atomic is needed; self rank also goes through the same SIMT aclshmem_float_p path.
        ::simt::aclshmem_float_p(weight_window + destination_local_offset, weights[route_index],
            static_cast<int32_t>(destination_rank));
    }
    // Every thread completes its remote store fence before the VF returns.
    asc_threadfence();
}

// Each AIV launches the VF once; 128 threads cover this core's full route range with group stride.
template <bool Profile>
__aicore__ inline uint64_t DispatchScatterWeightsSimt(GM_ADDR weights_gm, GM_ADDR dst_gm,
    GM_ADDR weights_buf_gm, int64_t route_count, int64_t aiv_index, int64_t aiv_count)
{
    // Invalid AIV index and empty input skip the split and modulo logic.
    if (route_count <= 0 || aiv_count <= 0 || aiv_index < 0 || aiv_index >= aiv_count) {
        return 0;
    }

    int64_t route_begin = 0;
    int64_t route_end = 0;
    DispatchSplitWeightRange(route_count, aiv_count, aiv_index, route_begin, route_end);
    // When there are fewer route groups than AIVs, some AIVs have no work.
    if (route_begin >= route_end) {
        return 0;
    }

    const uint32_t route_count_u32 = static_cast<uint32_t>(route_count);
    const uint32_t route_count_is_power_of_two =
        (route_count_u32 & (route_count_u32 - 1U)) == 0U ? 1U : 0U;
    uint32_t route_shift = 0;
    // Precompute the shift for power-of-two route_count, avoiding recomputation per SIMT thread.
    if (route_count_is_power_of_two != 0U) {
        for (uint32_t value = route_count_u32; value > 1U; value >>= 1U) {
            ++route_shift;
        }
    }

    __gm__ float *weights = reinterpret_cast<__gm__ float *>(weights_gm);
    __gm__ int32_t *encoded_destinations = reinterpret_cast<__gm__ int32_t *>(dst_gm);
    __gm__ float *weight_window = reinterpret_cast<__gm__ float *>(weights_buf_gm);
    const int32_t current_rank = aclshmem_my_pe();
    const uint32_t route_begin_u32 = static_cast<uint32_t>(route_begin);
    const uint32_t local_route_count = static_cast<uint32_t>(route_end - route_begin);
    const uint32_t local_group_count =
        (local_route_count + kWeightsPerGmConflictGroup - 1U) / kWeightsPerGmConflictGroup;
    const uint32_t first_group =
        (static_cast<uint32_t>(current_rank) + static_cast<uint32_t>(aiv_index)) % local_group_count;

    asc_vf_call<DispatchScatterWeightsSimtVf>(
        dim3(kWeightSimtThreadCount), weights, encoded_destinations, weight_window,
        route_begin_u32, local_route_count, route_count_u32, route_count_is_power_of_two,
        route_shift, first_group);

    // The scalar side waits for VF completion and ensures SIMT GM stores are visible to the later cross-PE barrier.
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    // The non-profile kernel emits no byte-counter code.
    if constexpr (Profile) {
        return static_cast<uint64_t>(local_route_count) * sizeof(float);
    }
    return 0;
}

// Phase 3: expand duplicate hidden per plan directly on this rank's symmetric hidden output.
// Primary rows are bulk-loaded into the double-buffered hidden_stage and written back to
// GM once per duplicate, replacing the per-tile CopyLocalBytes round trip.
__aicore__ inline void DispatchEpilogue(GM_ADDR hidden_buf_gm, GM_ADDR dup_groups_gm,
    GM_ADDR dup_loffs_gm, GM_ADDR dup_counts_gm, int64_t route_count, int64_t hidden_dim,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &mte_buffer,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &local_copy_buffer,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &slot_select_buffer)
{
    const int64_t hidden_row_bytes = hidden_dim * sizeof(DataType);
    const int64_t aiv_index = AscendC::GetBlockIdx();
    const int64_t aiv_count = AscendC::GetBlockNum();
    __gm__ int32_t *duplicate_groups = reinterpret_cast<__gm__ int32_t *>(dup_groups_gm);
    __gm__ int32_t *duplicate_local_offsets = reinterpret_cast<__gm__ int32_t *>(dup_loffs_gm);
    __gm__ int32_t *duplicate_counts = reinterpret_cast<__gm__ int32_t *>(dup_counts_gm);
    int64_t number_of_groups = duplicate_counts[0];
    int64_t number_of_duplicate_offsets = duplicate_counts[1];
    if (number_of_groups <= 0 || number_of_duplicate_offsets <= 0) {
        return;
    }
    if (number_of_groups > route_count) {
        number_of_groups = route_count;
    }
    if (number_of_duplicate_offsets > route_count) {
        number_of_duplicate_offsets = route_count;
    }

    int64_t group_begin = 0;
    int64_t group_end = 0;
    DispatchSplitRange(number_of_groups, aiv_count, aiv_index, group_begin, group_end);
    if (group_begin >= group_end) {
        return;
    }

    AscendC::LocalTensor<int32_t> group_stage = mte_buffer.Get<int32_t>();
    AscendC::LocalTensor<int32_t> loff_stage = local_copy_buffer.Get<int32_t>();
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    uint32_t hidden_stage_index = 0;

    for (int64_t tile_begin = group_begin; tile_begin < group_end;) {
        int64_t tile_count = group_end - tile_begin;
        if (tile_count > static_cast<int64_t>(kDuplicateGroupsPerTile)) {
            tile_count = kDuplicateGroupsPerTile;
        }
        CopyGmInt32ToUb(group_stage,
            duplicate_groups + tile_begin * kDuplicateGroupFields,
            static_cast<uint32_t>(tile_count * kDuplicateGroupFields));

        int64_t loff_span_begin = number_of_duplicate_offsets;
        int64_t loff_span_end = 0;
        for (int64_t local_group_index = 0; local_group_index < tile_count; ++local_group_index) {
            const uint32_t group_offset = static_cast<uint32_t>(local_group_index * kDuplicateGroupFields);
            const int32_t primary_local_offset = group_stage.GetValue(group_offset);
            const int32_t duplicate_list_begin = group_stage.GetValue(group_offset + 1);
            int64_t duplicate_count = group_stage.GetValue(group_offset + 2);
            if (primary_local_offset < 0 || duplicate_list_begin < 0 || duplicate_count <= 0 ||
                primary_local_offset >= route_count || duplicate_list_begin >= number_of_duplicate_offsets) {
                continue;
            }
            const int64_t available_duplicates = number_of_duplicate_offsets - duplicate_list_begin;
            if (duplicate_count > available_duplicates) {
                duplicate_count = available_duplicates;
            }
            if (duplicate_list_begin < loff_span_begin) {
                loff_span_begin = duplicate_list_begin;
            }
            const int64_t duplicate_list_end = duplicate_list_begin + duplicate_count;
            if (duplicate_list_end > loff_span_end) {
                loff_span_end = duplicate_list_end;
            }
        }

        const int64_t loff_span = loff_span_end > loff_span_begin ?
            loff_span_end - loff_span_begin : 0;
        const bool stage_duplicate_offsets = loff_span > 0 &&
            loff_span <= static_cast<int64_t>(kDuplicateLoffStagingBytes / sizeof(int32_t));
        if (stage_duplicate_offsets) {
            CopyGmInt32ToUb(loff_stage, duplicate_local_offsets + loff_span_begin,
                static_cast<uint32_t>(loff_span));
        }

        for (int64_t local_group_index = 0; local_group_index < tile_count; ++local_group_index) {
            const uint32_t group_offset = static_cast<uint32_t>(local_group_index * kDuplicateGroupFields);
            const int32_t primary_local_offset = group_stage.GetValue(group_offset);
            const int32_t duplicate_list_begin = group_stage.GetValue(group_offset + 1);
            int64_t duplicate_count = group_stage.GetValue(group_offset + 2);
            if (primary_local_offset < 0 || duplicate_list_begin < 0 || duplicate_count <= 0 ||
                primary_local_offset >= route_count || duplicate_list_begin >= number_of_duplicate_offsets) {
                continue;
            }
            const int64_t available_duplicates = number_of_duplicate_offsets - duplicate_list_begin;
            if (duplicate_count > available_duplicates) {
                duplicate_count = available_duplicates;
            }

            GM_ADDR primary_hidden_row =
                hidden_buf_gm + static_cast<int64_t>(primary_local_offset) * hidden_row_bytes;
            for (int64_t hidden_offset_bytes = 0; hidden_offset_bytes < hidden_row_bytes;
                hidden_offset_bytes += kDuplicateHiddenStageBytes) {
                int64_t current_hidden_bytes = hidden_row_bytes - hidden_offset_bytes;
                if (current_hidden_bytes > static_cast<int64_t>(kDuplicateHiddenStageBytes)) {
                    current_hidden_bytes = kDuplicateHiddenStageBytes;
                }
                const AscendC::TEventID event_id = hidden_stage_index == 0 ? EVENT_ID0 : EVENT_ID1;
                AscendC::LocalTensor<uint8_t> hidden_stage = slot_select_buffer.GetWithOffset<uint8_t>(
                    kDuplicateHiddenStageBytes, hidden_stage_index * kDuplicateHiddenStageBytes);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
                CopyGmBytesToUb(hidden_stage, primary_hidden_row + hidden_offset_bytes,
                    static_cast<uint32_t>(current_hidden_bytes), event_id);

                for (int64_t duplicate_index = 0; duplicate_index < duplicate_count; ++duplicate_index) {
                    const int64_t loff_index = duplicate_list_begin + duplicate_index;
                    const int32_t duplicate_local_offset = stage_duplicate_offsets ?
                        loff_stage.GetValue(static_cast<uint32_t>(loff_index - loff_span_begin)) :
                        duplicate_local_offsets[loff_index];
                    if (duplicate_local_offset < 0 || duplicate_local_offset >= route_count) {
                        continue;
                    }
                    GM_ADDR duplicate_hidden_chunk = hidden_buf_gm +
                        static_cast<int64_t>(duplicate_local_offset) * hidden_row_bytes + hidden_offset_bytes;
                    IssueUbBytesToGm(duplicate_hidden_chunk, hidden_stage,
                        static_cast<uint32_t>(current_hidden_bytes));
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
                hidden_stage_index ^= 1U;
            }
        }

        tile_begin += tile_count;
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    AscendC::PipeBarrier<PIPE_ALL>();
}

} // namespace DispatchMoonep

// ============================================================
// Dispatch kernel body — the bool Profile template parameter controls timing
// (cf. MoonEPDispatchKernelImpl, the UBMEM version on ascend-moonep-timing).
// One build produces both extern entries:
//   moonep_dispatch_udma_kernel         (Profile=false)
//   moonep_dispatch_udma_profile_kernel (Profile=true)
// ============================================================

template <bool Profile>
__forceinline__ __aicore__ void MoonEPDispatchUdmaKernelImpl(
    GM_ADDR hidden_gm,
    GM_ADDR weights_gm,
    GM_ADDR dst_gm,
    GM_ADDR dup_groups_gm,
    GM_ADDR dup_loffs_gm,
    GM_ADDR dup_counts_gm,
    GM_ADDR hidden_buf_gm,
    GM_ADDR weights_buf_gm,
    GM_ADDR sync_buffer_gm,
    GM_ADDR profile_cycles_gm,   // int64 [kMoonEPAivNum, kMoonEPDispatchUdmaProfileFields], used only by the profile kernel
    ascend_moonep::MoonEPDispatchTilingData tiling,
    uint64_t magic)
{
    if constexpr (g_coreType == AscendC::AIV) {
        if (hidden_gm == nullptr || weights_gm == nullptr || dst_gm == nullptr ||
            dup_groups_gm == nullptr || dup_loffs_gm == nullptr || dup_counts_gm == nullptr ||
            hidden_buf_gm == nullptr || weights_buf_gm == nullptr ||
            sync_buffer_gm == nullptr) {
            return;
        }

        const int64_t num_tokens = static_cast<int64_t>(tiling.S);
        const int64_t hidden_dim = static_cast<int64_t>(tiling.H);
        const int64_t num_topk = static_cast<int64_t>(tiling.K);
        const uint32_t with_weights = tiling.with_weights;

        // route_count = num_tokens * num_topk is used in later divide/modulo; require it positive up front to avoid divide-by-zero.
        if (num_tokens <= 0 || hidden_dim <= 0 || num_topk <= 0) {
            return;
        }

        __gm__ aclshmemi_aiv_udma_info_t* udma_info = aclshmemi_udma_qp_info_fetch();
        const uint32_t qp_num = udma_info->qp_num;
        const uint32_t active_qp_count =
            (qp_num >= 1U && qp_num < DispatchMoonep::kMaxActiveUdmaQpCount) ? qp_num :
            (qp_num == DispatchMoonep::kMaxCreatedUdmaQpCount) ? DispatchMoonep::kMaxActiveUdmaQpCount : 0U;

        const int32_t world_size = aclshmem_n_pes();
        const int64_t aiv_count = AscendC::GetBlockNum();
        const int64_t aiv_index = AscendC::GetBlockIdx();
        // Below 16P all ranks use one group; larger sizes split into 16-rank groups.
        if (!DispatchMoonep::CanUseGroupFlowControl(world_size, aiv_count) ||
            active_qp_count == 0U || active_qp_count > DispatchMoonep::kMaxActiveUdmaQpCount) {
            return;
        }
        const int64_t token_aiv_count = DispatchMoonep::GetDispatchGroupWidth(world_size);
        const bool token_worker = aiv_index < token_aiv_count;
        const bool weight_worker = !token_worker;
        const int64_t weight_aiv_index = aiv_index - token_aiv_count;
        const int64_t weight_aiv_count = aiv_count - token_aiv_count;

        // With Profile=true, per-phase cycles and byte counts go to the GM profile buffer;
        // with Profile=false UdmaProfileStart<Profile>() compiles to constant 0, zero cost.
        const uint64_t operator_start_cycle = DispatchMoonep::UdmaProfileStart<Profile>();
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECOUT> udma_wqe_buffer;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> mte_buffer;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> local_copy_buffer;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> slot_select_buffer;
        pipe.InitBuffer(udma_wqe_buffer, DispatchMoonep::kUdmaWqeScratchBytes);
        pipe.InitBuffer(mte_buffer, DispatchMoonep::kMteUbBytes);
        pipe.InitBuffer(local_copy_buffer, DispatchMoonep::kLocalCopyTileBytes);
        pipe.InitBuffer(slot_select_buffer, DispatchMoonep::kSlotSelectUbBytes);
        __ubuf__ uint8_t *udma_wqe =
            reinterpret_cast<__ubuf__ uint8_t *>(udma_wqe_buffer.Get<uint8_t>().GetPhyAddr());

        const int64_t route_count = num_tokens * num_topk;

        // Wait until all ranks have prepared normal GM inputs and local symmetric windows.
        aclshmemx_barrier_all_vec();

        // Scatter step 1: the first 16 AIVs batch-issue Token; the last 16 non-Token AIVs are reserved for Weight.
        uint64_t token_wqe_submit_cycles = 0;
        uint64_t token_wqe_submit_packets = 0;
        uint64_t token_sync_cycles = 0;
        const uint64_t token_scatter_start_cycle = DispatchMoonep::UdmaProfileStart<Profile>();
        uint64_t token_scatter_bytes = 0;
        if (token_worker) {
            token_scatter_bytes = DispatchMoonep::DispatchScatterToken<Profile>(
                hidden_gm, dst_gm, hidden_buf_gm, sync_buffer_gm,
                num_tokens, hidden_dim, num_topk, magic, active_qp_count,
                token_wqe_submit_cycles, token_wqe_submit_packets, token_sync_cycles,
                udma_wqe, local_copy_buffer, slot_select_buffer);
        }
        const uint64_t token_scatter_end_cycle = token_worker ?
            DispatchMoonep::UdmaProfileStart<Profile>() : token_scatter_start_cycle;

        // Serial mode joins Token first; the default parallel mode lets the last 16 AIVs enter Weight directly.
        if constexpr (!DispatchMoonep::kEnableParallelTokenWeight) {
            aclshmemx_barrier_all_vec();
        }

        // Scatter step 2: only non-Token AIVs run SIMT Weight; at 128P these are always the last 16 AIVs.
        const uint64_t weight_scatter_start_cycle = DispatchMoonep::UdmaProfileStart<Profile>();
        uint64_t weight_scatter_bytes = 0;
        if (weight_worker && with_weights != 0) {
            weight_scatter_bytes = DispatchMoonep::DispatchScatterWeightsSimt<Profile>(
                weights_gm, dst_gm, weights_buf_gm,
                route_count, weight_aiv_index, weight_aiv_count);
        }
        const uint64_t weight_scatter_end_cycle = weight_worker ?
            DispatchMoonep::UdmaProfileStart<Profile>() : weight_scatter_start_cycle;

        // Parallel mode joins Token/Weight here; serial mode joins Weight with the idle Token AIVs.
        aclshmemx_barrier_all_vec();

        // Scatter step 3: once all remote output is visible, expand duplicates inside this rank's Hidden window.
        const uint64_t duplicate_start_cycle = DispatchMoonep::UdmaProfileStart<Profile>();
        DispatchMoonep::DispatchEpilogue(
            hidden_buf_gm, dup_groups_gm, dup_loffs_gm, dup_counts_gm,
            route_count, hidden_dim,
            mte_buffer, local_copy_buffer, slot_select_buffer);
        const uint64_t duplicate_end_cycle = DispatchMoonep::UdmaProfileStart<Profile>();

        // All ranks exit together after normal GM outputs complete; the host does no phase synchronization.
        aclshmemx_barrier_all_vec();
        const uint64_t operator_end_cycle = DispatchMoonep::UdmaProfileStart<Profile>();

        if constexpr (Profile) {
            // Per-phase cycles and bytes are first written to UB, then pushed to GM via
            // MTE3 DataCopy, with a final PipeBarrier so the host sees them. Direct
            // __gm__ scalar writes bypass the pipe; PipeBarrier/dcci cannot guarantee
            // they land, and the host would read back the -1 sentinel.
            AscendC::LocalTensor<int64_t> profile_ub = local_copy_buffer.Get<int64_t>();
            profile_ub.SetValue(0, static_cast<int64_t>(operator_end_cycle - operator_start_cycle));
            profile_ub.SetValue(1, static_cast<int64_t>(token_scatter_end_cycle - token_scatter_start_cycle));
            profile_ub.SetValue(2, static_cast<int64_t>(token_wqe_submit_cycles));
            profile_ub.SetValue(3, static_cast<int64_t>(token_sync_cycles));
            profile_ub.SetValue(4, static_cast<int64_t>(token_scatter_bytes));
            profile_ub.SetValue(5, static_cast<int64_t>(token_wqe_submit_packets));
            profile_ub.SetValue(
                6, static_cast<int64_t>(weight_scatter_end_cycle - weight_scatter_start_cycle));
            profile_ub.SetValue(7, static_cast<int64_t>(weight_scatter_bytes));
            profile_ub.SetValue(8, static_cast<int64_t>(duplicate_end_cycle - duplicate_start_cycle));

            // One row is 9 int64 = 72 bytes; plain DataCopy's 32B granularity would
            // drop the last field. Reuse IssueUbBytesToGm's byte-granular DataCopyPad,
            // then PipeBarrier to wait for MTE3, so the host reads all 9 fields.
            DispatchMoonep::IssueUbBytesToGm(
                profile_cycles_gm +
                    static_cast<uint64_t>(AscendC::GetBlockIdx()) *
                        ascend_moonep::kMoonEPDispatchUdmaProfileFields * sizeof(int64_t),
                profile_ub.ReinterpretCast<uint8_t>(),
                static_cast<uint32_t>(
                    ascend_moonep::kMoonEPDispatchUdmaProfileFields * sizeof(int64_t)));
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
}

// ============================================================
// AscendC kernel entry points — the two template<Profile> instantiations
// ============================================================

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void moonep_dispatch_udma_kernel(
    GM_ADDR hidden_gm,
    GM_ADDR weights_gm,
    GM_ADDR dst_gm,
    GM_ADDR dup_groups_gm,
    GM_ADDR dup_loffs_gm,
    GM_ADDR dup_counts_gm,
    GM_ADDR hidden_buf_gm,
    GM_ADDR weights_buf_gm,
    GM_ADDR sync_buffer_gm,
    ascend_moonep::MoonEPDispatchTilingData tiling,
    uint64_t magic)
{
    MoonEPDispatchUdmaKernelImpl<false>(
        hidden_gm, weights_gm, dst_gm,
        dup_groups_gm, dup_loffs_gm, dup_counts_gm,
        hidden_buf_gm, weights_buf_gm, sync_buffer_gm,
        hidden_buf_gm,   // dummy profile_cycles_gm: never dereferenced when Profile=false
        tiling, magic);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void moonep_dispatch_udma_profile_kernel(
    GM_ADDR hidden_gm,
    GM_ADDR weights_gm,
    GM_ADDR dst_gm,
    GM_ADDR dup_groups_gm,
    GM_ADDR dup_loffs_gm,
    GM_ADDR dup_counts_gm,
    GM_ADDR hidden_buf_gm,
    GM_ADDR weights_buf_gm,
    GM_ADDR sync_buffer_gm,
    GM_ADDR profile_cycles_gm,   // int64 [kMoonEPAivNum, kMoonEPDispatchUdmaProfileFields]
    ascend_moonep::MoonEPDispatchTilingData tiling,
    uint64_t magic)
{
    MoonEPDispatchUdmaKernelImpl<true>(
        hidden_gm, weights_gm, dst_gm,
        dup_groups_gm, dup_loffs_gm, dup_counts_gm,
        hidden_buf_gm, weights_buf_gm, sync_buffer_gm,
        profile_cycles_gm,
        tiling, magic);
}

// ============================================================
// Host-side launcher (C-linkage, matches launch.h ABI)
// ============================================================

extern "C" __attribute__((visibility("default")))
void moonep_dispatch_udma_kernel_do(
    void* stream,
    uint8_t* hidden_sh,
    uint8_t* route_weights,
    uint8_t* dst,
    uint8_t* dup_groups,
    uint8_t* dup_loffs,
    uint8_t* dup_counts,
    uint8_t* hidden_buf,
    uint8_t* weights_buf,
    uint8_t* sync_buffer,
    uint8_t* tiling_host,
    uint64_t magic)
{
    auto* td = reinterpret_cast<ascend_moonep::MoonEPDispatchTilingData*>(tiling_host);

    // Always launch 32 AIVs: the first group_width run Token, the rest run Weight independently.
    const uint32_t block_dim = ascend_moonep::kMoonEPAivNum;
    moonep_dispatch_udma_kernel<<<block_dim, nullptr, stream>>>(
        hidden_sh,
        route_weights,
        dst,
        dup_groups,
        dup_loffs,
        dup_counts,
        hidden_buf,
        weights_buf,
        sync_buffer,
        *td,
        magic);
}

// Profile kernel launcher — differs from the regular version only by the extra
// profile_cycles output buffer (int64 [kMoonEPAivNum, kMoonEPDispatchUdmaProfileFields]),
// from which the host reads per-AIV phase cycles and byte counts.
// Cf. moonep_dispatch_profile_kernel_do on the timing branch.
extern "C" __attribute__((visibility("default")))
void moonep_dispatch_udma_profile_kernel_do(
    void* stream,
    uint8_t* hidden_sh,
    uint8_t* route_weights,
    uint8_t* dst,
    uint8_t* dup_groups,
    uint8_t* dup_loffs,
    uint8_t* dup_counts,
    uint8_t* hidden_buf,
    uint8_t* weights_buf,
    uint8_t* sync_buffer,
    uint8_t* profile_cycles,
    uint8_t* tiling_host,
    uint64_t magic)
{
    auto* td = reinterpret_cast<ascend_moonep::MoonEPDispatchTilingData*>(tiling_host);

    const uint32_t block_dim = ascend_moonep::kMoonEPAivNum;
    moonep_dispatch_udma_profile_kernel<<<block_dim, nullptr, stream>>>(
        hidden_sh,
        route_weights,
        dst,
        dup_groups,
        dup_loffs,
        dup_counts,
        hidden_buf,
        weights_buf,
        sync_buffer,
        profile_cycles,
        *td,
        magic);
}
