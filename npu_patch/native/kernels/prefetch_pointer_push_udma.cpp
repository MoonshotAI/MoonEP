// SPDX-License-Identifier: Apache-2.0
#include "ascend_moonep/kernels/launch.h"
#include "acl/acl.h"
#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510
#include <cstdint>
#include "kernel_operator.h"
#include "shmem.h"
#include "opdev/bfloat16.h"

namespace moonep_urma_push {
using namespace AscendC;
constexpr int32_t kUdmaCannShmemInternalUbStartBytes = 188 * 1024;
constexpr int64_t kUdmaPrefetchUbScratchOffset = 64;
constexpr int32_t kUdmaUbExpertTabOffset = kUdmaPrefetchUbScratchOffset;
constexpr int32_t kUdmaUbAlignBytes = 32;
constexpr int64_t kUdmaMaxTransferBytes = 256LL * 1024 * 1024;
constexpr uint32_t kUdmaLocalCopyMaxBytes = 65504;
constexpr uint32_t kUdmaWqeElements = 256;

__aicore__ inline int32_t UdmaAlignUp(int32_t value, int32_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

__aicore__ inline int32_t UdmaAivWorkerIndex()
{
    const int32_t sub_blocks = static_cast<int32_t>(GetSubBlockNum());
    return sub_blocks > 0 ?
        static_cast<int32_t>(GetBlockIdx()) / sub_blocks : 0;
}

__aicore__ inline void UdmaCopyLocalExpertChunk(
    __gm__ bfloat16_t* destination,
    __gm__ bfloat16_t* source,
    uint32_t elements,
    __ubuf__ bfloat16_t* ub)
{
    GlobalTensor<bfloat16_t> local_src;
    GlobalTensor<bfloat16_t> local_dst;
    local_src.SetGlobalBuffer(source, elements);
    local_dst.SetGlobalBuffer(destination, elements);

    LocalTensor<bfloat16_t> local_ub;
    local_ub.address_.logicPos = static_cast<uint8_t>(TPosition::VECIN);
    local_ub.address_.bufferAddr = reinterpret_cast<uint64_t>(ub);

    for (uint32_t copied = 0; copied < elements;) {
        uint32_t copy_elems = elements - copied;
        const uint32_t max_elems = kUdmaLocalCopyMaxBytes / sizeof(bfloat16_t);
        if (copy_elems > max_elems) {
            copy_elems = max_elems;
        }
        const DataCopyExtParams copy_params{
            1, static_cast<uint32_t>(copy_elems * sizeof(bfloat16_t)), 0, 0, 0};
        const DataCopyPadExtParams<bfloat16_t> pad_params{false, 0, 0, 0};
        DataCopyPad(local_ub, local_src[copied], copy_params, pad_params);
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID1);
        DataCopyPad(local_dst[copied], local_ub, copy_params);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        copied += copy_elems;
    }
}


__aicore__ inline void Run(
    GM_ADDR source_addr, GM_ADDR slots_addr, GM_ADDR plan_addr,
    int64_t experts, int64_t row_elements, int64_t slots)
{
    if ASCEND_IS_AIC { return; }
    if (UdmaAivWorkerIndex() != 0) { return; }
    auto* pointers = reinterpret_cast<__gm__ uint64_t*>(source_addr);
    auto* destination = reinterpret_cast<__gm__ bfloat16_t*>(slots_addr);
    auto* plan = reinterpret_cast<__gm__ int32_t*>(plan_addr);
    const int32_t ranks = aclshmem_n_pes();
    const int32_t rank = aclshmem_my_pe();
    const int64_t epn = (experts + ranks - 1) / ranks;
    const int32_t data_offset = UdmaAlignUp(
        kUdmaUbExpertTabOffset + UdmaAlignUp(
            static_cast<int32_t>(2 * slots * sizeof(int32_t)), 32), 32);
    TPipe pipe;
    TBuf<TPosition::VECOUT> tab_buf, wqe_buf, ub_buf;
    pipe.InitBuffer(tab_buf, data_offset);
    pipe.InitBuffer(wqe_buf, kUdmaWqeElements * sizeof(bfloat16_t));
    pipe.InitBuffer(ub_buf, kUdmaCannShmemInternalUbStartBytes -
        data_offset - kUdmaWqeElements * sizeof(bfloat16_t));
    auto tab = tab_buf.Get<int32_t>();
    auto wqe = wqe_buf.Get<bfloat16_t>();
    auto local = ub_buf.Get<bfloat16_t>();
    auto* expert_tab = reinterpret_cast<__ubuf__ int32_t*>(tab.GetPhyAddr());
    auto* slot_tab = expert_tab + slots;
    auto* scratch = reinterpret_cast<__ubuf__ bfloat16_t*>(wqe.GetPhyAddr());
    auto* ub = reinterpret_cast<__ubuf__ bfloat16_t*>(local.GetPhyAddr());
    Duplicate(wqe, static_cast<bfloat16_t>(0), kUdmaWqeElements);
    // SDK WQE construction uses scalar stores through the same raw UB pointer.
    // Order the vector clear before those stores; Tensor auto-sync cannot
    // infer this pointer handoff. SDK and local-copy events use IDs 0 and 1.
    SetFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
    // Validate the complete plan before this owner submits any payload.
    for (int64_t i = 0; i < ranks * slots; ++i) {
        if (plan[i] < -1 || plan[i] >= experts) {
            AscendC::printf("URMA_PUSH_INVALID_EXPERT rank=%d entry=%ld expert=%d\n",
                           rank, i, plan[i]);
            trap();
        }
    }
    const int64_t max_elements = kUdmaMaxTransferBytes / sizeof(bfloat16_t);
    for (int32_t peer = 0; peer < ranks; ++peer) {
        int64_t active = 0;
        for (int64_t slot = 0; slot < slots; ++slot) {
            const int32_t expert = plan[peer * slots + slot];
            if (expert >= 0 && expert / epn == rank) {
                expert_tab[active] = expert % epn;
                slot_tab[active++] = slot;
            }
        }
        for (int64_t i = 0; i < active; ++i) {
            __gm__ bfloat16_t* source;
            source = reinterpret_cast<__gm__ bfloat16_t*>(pointers[expert_tab[i]]);
            if (source == nullptr) {
                AscendC::printf("URMA_PUSH_NULL_SOURCE rank=%d expert=%d\n",
                               rank, expert_tab[i]);
                trap();
            }
            for (int64_t offset = 0; offset < row_elements; offset += max_elements) {
                const uint32_t count = static_cast<uint32_t>(
                    row_elements - offset < max_elements ? row_elements - offset : max_elements);
                auto* dst = destination + slot_tab[i] * row_elements + offset;
                if (peer == rank) {
                    UdmaCopyLocalExpertChunk(dst, source + offset, count, ub);
                } else {
                    aclshmemx_udma_put_nbi(dst, source + offset, scratch,
                        count, peer, EVENT_ID0);
                }
            }
        }
    }
    // One AIV issues all peers before waiting for their payloads. The SDK has
    // finished publishing each UB-staged WQE before put_nbi returns; the WQE
    // refers to the retained GM source, not this reusable UB scratch. Queue
    // credit and any overflow drain remain owned by the default SDK path.
    // No peer-visible publication is permitted until every required quiet has
    // returned. The caller's post barrier and consumer events are unchanged.
    for (int32_t peer = 0; peer < ranks; ++peer) {
        if (peer != rank) {
            aclshmemx_udma_quiet(peer);
        }
    }
    aclshmem_quiet();
}
} // namespace moonep_urma_push

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void prefetch_pointer_push_udma(GM_ADDR pointers, GM_ADDR slots, GM_ADDR plan,
    int64_t experts, int64_t row_elements, int64_t num_slots) {
    moonep_urma_push::Run(pointers,slots,plan,experts,row_elements,num_slots);
}
extern "C" ASCEND_MOONEP_EXPORT int prefetch_pointer_push_udma_available() { return 1; }
extern "C" ASCEND_MOONEP_EXPORT int moonep_prefetch_pointer_push_udma(
    void* stream, uint8_t* pointers, uint8_t* slots, uint8_t* plan,
    int64_t experts, int64_t row_elements, int64_t num_slots) {
    const auto status = aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
    if (status != ACL_SUCCESS) {
        return status;
    }
    prefetch_pointer_push_udma<<<1,nullptr,stream>>>(pointers,slots,plan,experts,row_elements,num_slots);
    return aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
}
#else
extern "C" ASCEND_MOONEP_EXPORT int prefetch_pointer_push_udma_available() { return 0; }
extern "C" ASCEND_MOONEP_EXPORT int moonep_prefetch_pointer_push_udma(
    void*,uint8_t*,uint8_t*,uint8_t*,int64_t,int64_t,int64_t) { return -1; }
#endif
