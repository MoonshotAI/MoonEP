// SPDX-License-Identifier: Apache-2.0
// MTE implementation of prepared expert-pointer weight PUSH.
#include <cstdint>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "shmem.h"
#include "ascend_moonep/kernels/launch.h"
#include "opdev/bfloat16.h"
using namespace AscendC;
namespace {
constexpr int32_t kPrefetchUbBytes = 188 * 1024;
constexpr uint64_t kPrefetchUbOffset = 64;
__aicore__ inline int32_t PrefetchWorkerIndex()
{
    const int32_t sub_blocks = static_cast<int32_t>(GetSubBlockNum());
    return sub_blocks > 0 ? static_cast<int32_t>(GetBlockIdx()) / sub_blocks : 0;
}

__aicore__ inline int64_t PrefetchCeilDiv(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

} // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void prefetch_pointer_push_mte(
    GM_ADDR source_ptrs_addr,
    GM_ADDR weight_slots_addr,
    GM_ADDR experts_to_copy_addr,
    int64_t row_elements,
    int32_t local_experts,
    int32_t ranks,
    int32_t slots,
    int32_t rank)
{
    if ASCEND_IS_AIC {
        return;
    }
    auto* source_ptrs = reinterpret_cast<__gm__ uint64_t*>(source_ptrs_addr);
    auto* weight_slots = reinterpret_cast<__gm__ bfloat16_t*>(weight_slots_addr);
    auto* plan = reinterpret_cast<__gm__ int32_t*>(experts_to_copy_addr);
    const int32_t block = PrefetchWorkerIndex();
    const int32_t block_count = GetBlockNum();
    const int32_t tile_elements = (kPrefetchUbBytes - static_cast<int32_t>(kPrefetchUbOffset)) /
        static_cast<int32_t>(sizeof(bfloat16_t)) / 2;
    const int64_t chunks = PrefetchCeilDiv(row_elements, tile_elements);
    const int64_t entries = static_cast<int64_t>(ranks) * slots;
    const int32_t local_begin = rank * local_experts;
    auto* ping = reinterpret_cast<__ubuf__ bfloat16_t*>(kPrefetchUbOffset);
    auto* pong = ping + tile_elements;
    int32_t buffer = 0;
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int64_t task = block; task < entries * chunks; task += block_count) {
        const int64_t entry = task / chunks;
        const int64_t chunk = task % chunks;
        const int32_t expert = plan[entry];
        if (expert < local_begin || expert >= local_begin + local_experts) {
            continue;
        }
        const int32_t local_expert = expert - local_begin;
        const int32_t destination_pe = static_cast<int32_t>(entry / slots);
        const int32_t destination_slot = static_cast<int32_t>(entry % slots);
        const int64_t begin = chunk * tile_elements;
        const int32_t elements = static_cast<int32_t>(
            row_elements - begin < tile_elements ? row_elements - begin : tile_elements);
        const TEventID event = buffer == 0 ? EVENT_ID0 : EVENT_ID1;
        auto* ub = buffer == 0 ? ping : pong;
        WaitFlag<HardEvent::MTE3_MTE2>(event);
        auto* src = reinterpret_cast<__gm__ bfloat16_t*>(source_ptrs[local_expert]) + begin;
        auto* dst = weight_slots + static_cast<int64_t>(destination_slot) * row_elements + begin;
        aclshmemx_mte_put_nbi(
            dst, src, ub,
            static_cast<uint32_t>(tile_elements * sizeof(bfloat16_t)),
            static_cast<uint32_t>(elements), destination_pe, event);
        SetFlag<HardEvent::MTE3_MTE2>(event);
        buffer = 1 - buffer;
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    aclshmem_quiet();
}

extern "C" ASCEND_MOONEP_EXPORT int moonep_prefetch_pointer_push_mte(
    void* stream, const uint64_t* source_ptrs, void* weight_slots,
    const int32_t* experts_to_copy, int64_t row_elements, int32_t local_experts,
    int32_t ranks, int32_t slots, int32_t rank, int32_t blocks)
{
    if (stream == nullptr || source_ptrs == nullptr || weight_slots == nullptr ||
        experts_to_copy == nullptr || row_elements <= 0 || local_experts <= 0 ||
        ranks <= 0 || slots <= 0 || rank < 0 || rank >= ranks || blocks <= 0 ||
        ranks != aclshmem_n_pes()) {
        return ACL_ERROR_INVALID_PARAM;
    }
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    prefetch_pointer_push_mte<<<blocks, nullptr, stream>>>(
        reinterpret_cast<uint8_t*>(const_cast<uint64_t*>(source_ptrs)),
        reinterpret_cast<uint8_t*>(weight_slots),
        reinterpret_cast<uint8_t*>(const_cast<int32_t*>(experts_to_copy)),
        row_elements, local_experts, ranks, slots, rank);
    return aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
}
