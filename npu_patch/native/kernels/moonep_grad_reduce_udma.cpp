// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "acl/acl.h"
#include "kernel_operator.h"
#include "shmem.h"

#include "ascend_moonep/kernels/launch.h"

using namespace AscendC;

namespace moonEP {
namespace gradReduce {

constexpr int64_t INNER_LOOP_SIZE = 30 * 1024;
constexpr int64_t OUTER_LOOP_SIZE = 512 * 1024;

__aicore__ inline void ComputeSingle(
    __gm__ float* grads,
    __gm__ float* workspace,
    LocalTensor<float>& ubLocal,
    int64_t matrix)
{
    GlobalTensor<float> srcGm_, dstGm_;
    srcGm_.SetGlobalBuffer(workspace);
    dstGm_.SetGlobalBuffer(grads);

    int32_t pingpongFlag = 0;
    SetFlag<HardEvent::MTE3_MTE2>(0);
    SetFlag<HardEvent::MTE3_MTE2>(1);
    for (int64_t offset = 0; offset < matrix; offset += INNER_LOOP_SIZE) {
        LocalTensor<float> srcLocal = ubLocal[pingpongFlag * INNER_LOOP_SIZE];
        int64_t loopSize = min(matrix - offset, INNER_LOOP_SIZE);
        uint32_t loopByte = loopSize * sizeof(float);
        WaitFlag<HardEvent::MTE3_MTE2>(pingpongFlag);
        DataCopyPad(srcLocal, srcGm_[offset], {1, loopByte, 0, 0, 0}, {});
        SetFlag<HardEvent::MTE2_MTE3>(pingpongFlag);
        WaitFlag<HardEvent::MTE2_MTE3>(pingpongFlag);
        SetAtomicAdd<float>();
        DataCopyPad(dstGm_[offset], srcLocal, {1, loopByte, 0, 0, 0});
        SetAtomicNone();
        SetFlag<HardEvent::MTE3_MTE2>(pingpongFlag);

        pingpongFlag = 1 - pingpongFlag;
    }
    WaitFlag<HardEvent::MTE3_MTE2>(0);
    WaitFlag<HardEvent::MTE3_MTE2>(1);
}

}  // namespace gradReduce
}  // namespace moonEP

using namespace moonEP::gradReduce;

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void MoonEPShmemGradReduceUdmaFused(
    uint64_t ffts_addr,
    GM_ADDR expert_grads_addr,
    GM_ADDR reduce_buffers_addr,
    GM_ADDR workspace_addr,
    GM_ADDR experts_to_copy_addr,
    int32_t epn,
    int32_t hidden,
    int32_t projection,
    int32_t ranks,
    int32_t slots,
    int32_t rank)
{
    util_set_ffts_config(ffts_addr);

    const int32_t coreIdx = GetBlockIdx();
    const int32_t coreNum = GetBlockNum();
    const int64_t matrix = static_cast<int64_t>(hidden) * projection;

    auto* grads = reinterpret_cast<__gm__ float*>(expert_grads_addr);
    auto* reduce = reinterpret_cast<__gm__ float*>(reduce_buffers_addr);
    auto* plan = reinterpret_cast<__gm__ int32_t*>(experts_to_copy_addr);
    auto* workspace = reinterpret_cast<__gm__ float*>(workspace_addr) + coreIdx * OUTER_LOOP_SIZE;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> wqeBuf, ubBuf;
    pipe.InitBuffer(ubBuf, 2 * INNER_LOOP_SIZE * sizeof(float));
    pipe.InitBuffer(wqeBuf, 256);
    AscendC::LocalTensor<float> ubLocal = ubBuf.Get<float>();
    AscendC::LocalTensor<uint8_t> wqeLocal = wqeBuf.Get<uint8_t>();

    __ubuf__ uint8_t* wqe_scratch = (__ubuf__ uint8_t*)wqeLocal.GetPhyAddr();
    Duplicate(wqeLocal, (uint8_t)0, 256);

    for (int32_t srcRankIdx = coreIdx; srcRankIdx < ranks; srcRankIdx+=coreNum) {
        for (int32_t srcSlotIdx = 0; srcSlotIdx < slots; srcSlotIdx++) {
            int32_t srcIdx = srcRankIdx * slots + srcSlotIdx;
            int32_t dstIdx = plan[srcIdx];
            if (dstIdx == -1) {
                continue;
            }

            int32_t dstRankIdx = dstIdx / epn;
            int32_t dstExpertIdx = dstIdx % epn;
            if (dstRankIdx != rank) {
                continue;
            }

            for (int64_t slotOffset = 0; slotOffset < matrix; slotOffset += OUTER_LOOP_SIZE)
            {
                int64_t loopSize = min(matrix - slotOffset, OUTER_LOOP_SIZE);
                if (srcRankIdx != dstRankIdx) {
                    aclshmemx_udma_get_nbi((__gm__ uint8_t*)(workspace), (__gm__ uint8_t*)(reduce + srcSlotIdx * matrix + slotOffset), wqe_scratch, loopSize * sizeof(float), srcRankIdx, 0);
                    aclshmemx_udma_quiet(srcRankIdx);
                    ComputeSingle(grads + dstExpertIdx * matrix + slotOffset, workspace, ubLocal, loopSize);
                } else {
                    ComputeSingle(grads + dstExpertIdx * matrix + slotOffset, reduce + srcSlotIdx * matrix + slotOffset, ubLocal, loopSize);
                }
            }
        }
    }

    aclshmem_quiet();
    aclshmemx_barrier_all_vec();

    if (coreIdx == 0) {
        GlobalTensor<float> reduceGm_;
        reduceGm_.SetGlobalBuffer(reduce);
        for (int32_t localIdx = 0; localIdx < slots; localIdx++) {
            if (plan[rank * slots + localIdx] >= 0) {
                InitOutput(reduceGm_[localIdx * matrix], matrix, 0.0f);
            }
        }
    }
}

extern "C" ASCEND_MOONEP_EXPORT int moonep_shmem_grad_reduce_udma(
    void* stream,
    void* expert_grads,
    void* reduce_buffers,
    void* workspace,
    const int32_t* experts_to_copy,
    int32_t epn,
    int32_t hidden,
    int32_t projection,
    int32_t ranks,
    int32_t slots,
    int32_t rank,
    int32_t blocks)
{
    if (stream == nullptr || expert_grads == nullptr ||
        reduce_buffers == nullptr || experts_to_copy == nullptr ||
        workspace == nullptr || epn <= 0 || hidden <= 0 ||
        projection <= 0 || ranks <= 0 || slots < 0 || rank < 0 ||
        rank >= ranks || blocks <= 0) {
        return ACL_ERROR_INVALID_PARAM;
    }

    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    const int32_t shmem_ranks = aclshmem_n_pes();
    if (shmem_ranks <= 0 || ranks != shmem_ranks) {
        return ACL_ERROR_INVALID_PARAM;
    }

    MoonEPShmemGradReduceUdmaFused<<<blocks, nullptr, stream>>>(
        shmemx_get_ffts_config(),
        reinterpret_cast<std::uint8_t*>(expert_grads),
        reinterpret_cast<std::uint8_t*>(reduce_buffers),
        reinterpret_cast<std::uint8_t*>(workspace),
        reinterpret_cast<std::uint8_t*>(const_cast<int32_t*>(experts_to_copy)),
        epn,
        hidden,
        projection,
        ranks,
        slots,
        rank);
    return aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
}
