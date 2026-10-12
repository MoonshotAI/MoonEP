#include <cstdint>

#include "kernel_operator.h"
#include "shmem.h"

#include "ascend_moonep/kernels/types.h"
#include "opdev/bfloat16.h"

using namespace AscendC;

#if defined(__GNUC__)
#define ASCEND_MOONEP_EXPORT __attribute__((visibility("default")))
#else
#define ASCEND_MOONEP_EXPORT
#endif

// in MoonEP
// E：896
// H：7168
// Hp：3584
// B：(E + R - 1) / R

namespace {

// cann-shmem reserves [188 KiB, 190 KiB) for internal collective state.
// Keep the caller-supplied GM-to-GM staging buffer strictly below it.
constexpr int32_t kCannShmemInternalUbStartBytes = 188 * 1024;
constexpr int64_t kPrefetchUbScratchOffset = 64;
constexpr int32_t kUbExpertTabOffset = kPrefetchUbScratchOffset;
constexpr int32_t kUbAlignBytes = 32;

__aicore__ inline int32_t AlignUp(int32_t value, int32_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

__aicore__ inline int32_t AivWorkerIndex()
{
    const int32_t sub_blocks = static_cast<int32_t>(GetSubBlockNum());
    return sub_blocks > 0 ?
        static_cast<int32_t>(GetBlockIdx()) / sub_blocks : 0;
}

}  // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void prefetch_weight_kernel(
    GM_ADDR remote_expert_addr,
    GM_ADDR prefetch_buffers_addr,
    GM_ADDR experts_to_copy_addr,
    int64_t E,
    int64_t H,
    int64_t Hp,
    int64_t B)
{
    if ASCEND_IS_AIC {
        return;
    }
    __gm__ bfloat16_t* remote_expert_ptr =
        reinterpret_cast<__gm__ bfloat16_t*>(remote_expert_addr);
    __gm__ bfloat16_t* prefetch_ptr =
        reinterpret_cast<__gm__ bfloat16_t*>(prefetch_buffers_addr);
    __gm__ int32_t* plan =
        reinterpret_cast<__gm__ int32_t*>(experts_to_copy_addr);

    const int32_t ranks = aclshmem_n_pes();
    const int64_t epn = (E + ranks - 1) / ranks;
    const int32_t block = AivWorkerIndex();
    const int32_t block_count = GetBlockNum();
    const int32_t expert_tab_bytes = static_cast<int32_t>(
        2 * B * static_cast<int64_t>(sizeof(int32_t)));
    const int32_t data_offset = AlignUp(
        kUbExpertTabOffset + expert_tab_bytes, kUbAlignBytes);
    const int32_t ub_bytes =
        kCannShmemInternalUbStartBytes - data_offset;
    if (ub_bytes <= 0 || ranks <= 0 || epn <= 0) {
        return;
    }

    __ubuf__ int32_t* expert_tab =
        reinterpret_cast<__ubuf__ int32_t*>(kUbExpertTabOffset);
    __ubuf__ int32_t* slot_tab = reinterpret_cast<__ubuf__ int32_t*>(
        kUbExpertTabOffset + B * static_cast<int32_t>(sizeof(int32_t)));
    // Compact the sparse plan in UB. All vector lanes execute the same scan,
    // so the resulting table is deterministic without a cross-lane reduction.
    int32_t active_experts = 0;
    for (int32_t slot = 0; slot < B; ++slot) {
        const int32_t expert = plan[slot];
        if (expert >= 0 && expert < E) {
            expert_tab[active_experts] = expert;
            slot_tab[active_experts] = slot;
            ++active_experts;
        }
    }

    // Use one remote get for as many complete H rows as fit in UB.
    // The resulting task space is compact and load-balanced across AIVs.
    const int64_t elements_per_row = Hp;
    int buffernum = 2;
    int bufferid = 0;
    const int32_t buffer_bytes = ub_bytes / buffernum / kUbAlignBytes * kUbAlignBytes;
    const int64_t rows_per_chunk = buffer_bytes /
        static_cast<int32_t>(sizeof(bfloat16_t)) / elements_per_row;
    if (rows_per_chunk <= 0) {
        return;
    }
    const int64_t chunks_per_expert =
        (H + rows_per_chunk - 1) / rows_per_chunk;
    const int64_t total_chunks = static_cast<int64_t>(active_experts) *
        chunks_per_expert;

    for (int i = 0; i < buffernum; ++i) {
        SetFlag<HardEvent::MTE3_MTE2>(i == 0 ? EVENT_ID0 : EVENT_ID1);
    }
    for (int64_t task = block; task < total_chunks; task += block_count) {
        const TEventID event_id = bufferid == 0 ? EVENT_ID0 : EVENT_ID1;
        __ubuf__ bfloat16_t* ub = reinterpret_cast<__ubuf__ bfloat16_t*>(
            data_offset + bufferid * buffer_bytes);
        WaitFlag<HardEvent::MTE3_MTE2>(event_id);
        const int64_t compact_slot = task / chunks_per_expert;
        const int64_t chunk = task % chunks_per_expert;
        const int32_t expert = expert_tab[compact_slot];
        const int32_t slot = slot_tab[compact_slot];

        const int64_t row = chunk * rows_per_chunk;
        const int64_t rows = (row + rows_per_chunk <= H) ?
            rows_per_chunk : H - row;
        const int32_t owner = expert / epn;
        const int32_t local_expert = expert % epn;
        // Each PE owns an [ceil(E/R) + B, H, Hp] allocation.  The SHMEM MTE
        // get resolves the peer base from the local symmetric source and owner.
        const int64_t source =
            (static_cast<int64_t>(local_expert) * H + row) * Hp;
        const int64_t destination =
            (static_cast<int64_t>(slot) * H + row) * Hp;
        aclshmemx_mte_get_nbi(
            prefetch_ptr + destination,
            remote_expert_ptr + source,
            ub,
            static_cast<uint32_t>(buffer_bytes),
             static_cast<uint32_t>(rows * elements_per_row),
            owner,
            event_id);
        SetFlag<HardEvent::MTE3_MTE2>(event_id);
        bufferid = (bufferid + 1) % buffernum;
    }
    for (int i = 0; i < buffernum; ++i) {
        WaitFlag<HardEvent::MTE3_MTE2>(
            i == 0 ? EVENT_ID0 : EVENT_ID1);
    }
    aclshmem_quiet();
}

extern "C" ASCEND_MOONEP_EXPORT void prefetch_weight_kernel_do(
    void* stream,
    uint8_t* remote_expert,
    uint8_t* prefetch_buffers,
    uint8_t* experts_to_copy,
    int64_t num_experts,
    int64_t hidden,
    int64_t hidden_prime,
    int64_t num_slots,
    int64_t block_dim) {

    prefetch_weight_kernel<<<block_dim, nullptr, stream>>>(
        remote_expert,
        prefetch_buffers,
        experts_to_copy,
        num_experts,
        hidden,
        hidden_prime,
        num_slots);
}
