#pragma once

#include <cstdint>

namespace ascend_moonep {

// Shared host/device launch width. Lives here so unity builds of multiple
// kernels do not redefine a file-local kMoonEPAivNum.
inline constexpr uint32_t kMoonEPAivNum = 32;
inline constexpr uint32_t kMoonEPUdmaCombineQpCount = 2;
inline constexpr uint32_t kMoonEPUdmaCombineBandwidthScratchAlign = 64;
inline constexpr uint32_t kMoonEPUdmaCombineBandwidthStatsStride = 8;
inline constexpr uint32_t kMoonEPUdmaCombineBandwidthStatsElems =
    kMoonEPAivNum * kMoonEPUdmaCombineBandwidthStatsStride;
inline constexpr uint32_t kMoonEPUdmaCombineBandwidthScratchBytes =
    kMoonEPUdmaCombineBandwidthStatsElems * sizeof(int64_t);

struct alignas(8) MoonEPCombineTilingData {
    uint32_t rank;              // current rank
    uint32_t world_size;        // EP size
    uint32_t num_tokens;        // S
    uint32_t hidden;            // H
    uint32_t num_topk;          // K
    uint32_t block_dim;         // fixed 32 AIV (host always writes 32)
    uint32_t with_weight;       // 1: gather weights → [S,K]; 0: skip
    uint32_t reserved;
};

struct alignas(8) MoonEPUdmaCombineTilingData {
    uint32_t S;
    uint32_t H;
    uint32_t K;
    uint32_t R;
    uint32_t rank;
    uint32_t with_weights;
    uint32_t reserved;
};

struct alignas(8) MoonEPDispatchTilingData {
    uint32_t S;
    uint32_t H;
    uint32_t K;
    uint32_t R;
    uint32_t rank;
    uint32_t with_weights;
    uint32_t reserved;
};

// Udma dispatch profile ABI (mirrors the UBMEM template-<bool Profile> pattern
// from the timing branch): moonep_dispatch_udma_profile_kernel writes each AIV's
// metrics as an int64 row into a [kMoonEPAivNum, kMoonEPDispatchUdmaProfileFields]
// GM buffer (profile_cycles_gm); the host reads the buffer back instead of
// kernel printf. Plain (non-profile) kernel passes a dummy pointer that is
// never dereferenced when Profile=false.
inline constexpr uint32_t kMoonEPDispatchUdmaProfileFields = 9;

enum MoonEPDispatchUdmaProfileField : uint32_t {
    kDispatchUdmaProfileTotal = 0,      // full operator wall cycles (incl. barriers)
    kDispatchUdmaProfileTokenScatter,   // token scatter stage cycles
    kDispatchUdmaProfileTokenWqeSubmit, // accumulated WQE fill/submit cycles
    kDispatchUdmaProfileTokenSync,      // token scatter completion sync cycles
    kDispatchUdmaProfileTokenBytes,     // bytes actually handled by this AIV (token)
    kDispatchUdmaProfileWqePackets,     // WQE packets issued by this AIV
    kDispatchUdmaProfileWeightScatter,  // weight scatter stage cycles
    kDispatchUdmaProfileWeightBytes,    // weight bytes scattered by this AIV
    kDispatchUdmaProfileDuplicate,      // duplicate expansion stage cycles
};

struct alignas(8) PlanningTilingData {
    uint32_t rank;
    uint32_t world_size;
    uint32_t num_tokens;
    uint32_t num_topk;
    uint32_t num_experts;
    uint32_t num_max_tokens_per_rank;
    uint32_t num_prefetch_slots;
    uint32_t block_dim;
    uint32_t symmetric_tpe_offset;
    uint32_t symmetric_dup_rank_counts_offset;
    uint32_t source_dup_tokens_per_tile;
    uint32_t owner_reduce_tile_elems;
    uint32_t reserved;
};

static_assert(sizeof(PlanningTilingData) == 56,
              "PlanningTilingData ABI must remain 56 bytes");

}  // namespace ascend_moonep
