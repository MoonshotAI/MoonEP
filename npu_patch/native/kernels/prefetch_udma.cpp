#include <cstdint>

#include "kernel_operator.h"
#include "shmem.h"

#include "ascend_moonep/kernels/launch.h"
#include "opdev/bfloat16.h"

using namespace AscendC;

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)

namespace {

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

}  // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void prefetch_weight_kernel_udma(
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
    // UDMA is shared by the whole UB and must only be driven by one AIV.
    const int32_t block = UdmaAivWorkerIndex();
    if (block != 0) {
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
    const int32_t my_rank = aclshmem_my_pe();
    const int32_t expert_tab_bytes = UdmaAlignUp(
        static_cast<int32_t>(2 * B * static_cast<int64_t>(sizeof(int32_t))),
        kUdmaUbAlignBytes);
    const int32_t data_offset = UdmaAlignUp(
        kUdmaUbExpertTabOffset + expert_tab_bytes, kUdmaUbAlignBytes);
    const int32_t wqe_size = kUdmaWqeElements * sizeof(bfloat16_t);

    TPipe pipe;
    TBuf<TPosition::VECOUT> tab_buf, wqe_buf, ub_buf;
    pipe.InitBuffer(tab_buf, data_offset);
    pipe.InitBuffer(wqe_buf, wqe_size);
    pipe.InitBuffer(
        ub_buf,
        kUdmaCannShmemInternalUbStartBytes - wqe_size - data_offset);
    LocalTensor<int32_t> expert_tab_local = tab_buf.Get<int32_t>();
    LocalTensor<int32_t> slot_tab_local = expert_tab_local[B];
    LocalTensor<bfloat16_t> wqe_local = wqe_buf.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> ub_local = ub_buf.Get<bfloat16_t>();

    __ubuf__ int32_t* expert_tab =
        reinterpret_cast<__ubuf__ int32_t*>(expert_tab_local.GetPhyAddr());
    __ubuf__ int32_t* slot_tab =
        reinterpret_cast<__ubuf__ int32_t*>(slot_tab_local.GetPhyAddr());
    __ubuf__ bfloat16_t* wqe_scratch =
        reinterpret_cast<__ubuf__ bfloat16_t*>(wqe_local.GetPhyAddr());
    __ubuf__ bfloat16_t* ub =
        reinterpret_cast<__ubuf__ bfloat16_t*>(ub_local.GetPhyAddr());
    Duplicate(wqe_local, static_cast<bfloat16_t>(0), kUdmaWqeElements);

    int64_t active_experts = 0;
    for (int64_t slot = 0; slot < B; ++slot) {
        const int32_t expert = plan[slot];
        if (expert >= 0 && expert < E) {
            expert_tab[active_experts] = expert;
            slot_tab[active_experts] = slot;
            ++active_experts;
        }
    }

    const int64_t single_expert_elements = static_cast<int64_t>(H) * Hp;
    const int64_t max_udma_elements =
        kUdmaMaxTransferBytes / sizeof(bfloat16_t);
    const int64_t tasks_per_expert =
        (single_expert_elements + max_udma_elements - 1) / max_udma_elements;
    const int64_t total_tasks = active_experts * tasks_per_expert;

    for (int64_t task = block; task < total_tasks; ++task) {
        const int64_t compact_slot = task / tasks_per_expert;
        const int64_t chunk = task % tasks_per_expert;
        const int64_t expert = expert_tab[compact_slot];
        const int64_t slot = slot_tab[compact_slot];
        const int64_t owner = expert / epn;
        const int64_t local_expert = expert % epn;
        const int64_t inner_offset = chunk * max_udma_elements;
        const int64_t source =
            local_expert * single_expert_elements + inner_offset;
        const int64_t destination =
            slot * single_expert_elements + inner_offset;
        const int64_t current_elements =
            (inner_offset + max_udma_elements <= single_expert_elements)
            ? max_udma_elements
            : single_expert_elements - inner_offset;

        if (owner == my_rank) {
            // UDMA does not support self operations.
            UdmaCopyLocalExpertChunk(
                prefetch_ptr + destination,
                remote_expert_ptr + source,
                static_cast<uint32_t>(current_elements),
                ub);
        } else {
            aclshmemx_udma_get_nbi(
                prefetch_ptr + destination,
                remote_expert_ptr + source,
                wqe_scratch,
                static_cast<uint32_t>(current_elements),
                owner,
                EVENT_ID0);
            aclshmemx_udma_quiet(owner);
        }
    }
    aclshmem_quiet();
}

extern "C" ASCEND_MOONEP_EXPORT void prefetch_weight_kernel_udma_do(
    void* stream,
    uint8_t* remote_expert,
    uint8_t* prefetch_buffers,
    uint8_t* experts_to_copy,
    int64_t num_experts,
    int64_t hidden,
    int64_t hidden_prime,
    int64_t num_slots,
    int64_t block_dim)
{
    prefetch_weight_kernel_udma<<<block_dim, nullptr, stream>>>(
        remote_expert,
        prefetch_buffers,
        experts_to_copy,
        num_experts,
        hidden,
        hidden_prime,
        num_slots);
}

#else

extern "C" ASCEND_MOONEP_EXPORT void prefetch_weight_kernel_udma_do(
    void* stream,
    uint8_t* remote_expert,
    uint8_t* prefetch_buffers,
    uint8_t* experts_to_copy,
    int64_t num_experts,
    int64_t hidden,
    int64_t hidden_prime,
    int64_t num_slots,
    int64_t block_dim)
{
    // The Python policy never selects UDMA on non-Ascend950 devices. Keep a
    // defensive MTE fallback so the shared host extension has one stable ABI.
    prefetch_weight_kernel_do(
        stream,
        remote_expert,
        prefetch_buffers,
        experts_to_copy,
        num_experts,
        hidden,
        hidden_prime,
        num_slots,
        block_dim);
}

#endif
