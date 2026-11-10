// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "acl/acl.h"
#include "kernel_operator.h"
#include "shmem.h"

#include "ascend_moonep/kernels/launch.h"

using namespace AscendC;

namespace moonEP {
namespace gradReduce {

constexpr int32_t kUbTotalBytes = 188 * 1024;
constexpr uint64_t kUbScratchOffset = 64;

constexpr int32_t kTileElements = 12 * 1024;
constexpr int32_t kTileBytes = kTileElements * sizeof(float);

constexpr int64_t kFp32PerCacheline = 32 / sizeof(float);
constexpr int64_t kClearTileCachelines = kTileElements / kFp32PerCacheline;

static_assert(
    kUbScratchOffset + 3 * kTileBytes <= kUbTotalBytes,
    "UB tiles overlap cann-shmem reserved region");

__aicore__ inline int32_t GetWorkerIndex()
{
    const int32_t sub_blocks = static_cast<int32_t>(GetSubBlockNum());
    return sub_blocks > 0 ?
        static_cast<int32_t>(GetBlockIdx()) / sub_blocks : 0;
}

__aicore__ inline int64_t CeilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

__aicore__ inline LocalTensor<float> MakeFp32Local(
    __ubuf__ float* address,
    int32_t elements,
    TPosition position)
{
    LocalTensor<float> tensor;
    tensor.address_.logicPos = static_cast<uint8_t>(position);
    tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(address);
    tensor.address_.dataLen =
        static_cast<uint32_t>((elements * sizeof(float) + 31) & ~31);
    return tensor;
}

__aicore__ inline void LoadFp32(
    LocalTensor<float> dst,
    __gm__ float* src,
    int32_t elements)
{
    GlobalTensor<float> src_tensor;
    src_tensor.SetGlobalBuffer(src);
    DataCopyExtParams params(
        1, static_cast<uint32_t>(elements * sizeof(float)), 0, 0, 0);
    DataCopyPadExtParams<float> pad{false, 0, 0, 0};
    DataCopyPad(dst, src_tensor, params, pad);
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void StoreFp32(
    __gm__ float* dst,
    LocalTensor<float> src,
    int32_t elements)
{
    GlobalTensor<float> dst_tensor;
    dst_tensor.SetGlobalBuffer(dst);
    DataCopyExtParams params(
        1, static_cast<uint32_t>(elements * sizeof(float)), 0, 0, 0);
    PipeBarrier<PIPE_ALL>();
    DataCopyPad(dst_tensor, src, params);
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void StartAsyncLoad(
    __ubuf__ float* dst,
    __gm__ float* symmetric_src,
    int32_t elements,
    int32_t pe,
    int32_t rank,
    uint32_t event_id)
{
    if (pe == rank) {
        auto dst_local = MakeFp32Local(dst, elements, TPosition::VECIN);
        GlobalTensor<float> src_tensor;
        src_tensor.SetGlobalBuffer(symmetric_src);
        DataCopyExtParams params(
            1, static_cast<uint32_t>(elements * sizeof(float)), 0, 0, 0);
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(dst_local, src_tensor, params, pad);
    } else {
        aclshmemx_mte_get_nbi(dst, symmetric_src, elements, pe, event_id);
    }
    SetFlag<HardEvent::MTE2_V>(event_id);
}

__aicore__ inline void AccumulatePlanTile(
    __gm__ float* grads,
    __gm__ float* reduce,
    const __gm__ int32_t* plan,
    __ubuf__ float* accum_ub,
    __ubuf__ float* reduce_ping_ub,
    __ubuf__ float* reduce_pong_ub,
    int32_t local_expert,
    int32_t global_expert,
    int64_t matrix,
    int64_t tile_begin,
    int32_t elements,
    int32_t rank,
    int32_t slots,
    int64_t reduce_entries)
{
    // Scan the plan in rb order so accumulation order matches MoonEP.
    int64_t current_rb = 0;
    while (current_rb < reduce_entries && plan[current_rb] != global_expert) {
        ++current_rb;
    }
    if (current_rb == reduce_entries) {
        return;
    }

    auto accum_in = MakeFp32Local(accum_ub, elements, TPosition::VECIN);
    auto accum_out = MakeFp32Local(accum_ub, elements, TPosition::VECOUT);
    LoadFp32(accum_in, grads + local_expert * matrix + tile_begin, elements);

    int32_t current_buffer = 0;  // 0=ping, 1=pong

    const int32_t current_source_pe =
        static_cast<int32_t>(current_rb / slots);
    const int32_t current_source_slot =
        static_cast<int32_t>(current_rb % slots);
    StartAsyncLoad(
        reduce_ping_ub,
        reduce + static_cast<int64_t>(current_source_slot) * matrix + tile_begin,
        elements,
        current_source_pe,
        rank,
        EVENT_ID0);

    bool ping_pending = false;
    bool pong_pending = false;

    while (current_rb < reduce_entries) {
        const uint32_t current_event =
            current_buffer == 0 ? EVENT_ID0 : EVENT_ID1;
        auto* current_reduce_ub =
            current_buffer == 0 ? reduce_ping_ub : reduce_pong_ub;

        WaitFlag<HardEvent::MTE2_V>(current_event);

        // Continue scanning the plan directly for the next contribution.
        int64_t next_rb = current_rb + 1;
        while (next_rb < reduce_entries && plan[next_rb] != global_expert) {
            ++next_rb;
        }

        if (next_rb < reduce_entries) {
            const int32_t next_buffer = 1 - current_buffer;
            const uint32_t next_event =
                next_buffer == 0 ? EVENT_ID0 : EVENT_ID1;
            auto* next_reduce_ub =
                next_buffer == 0 ? reduce_ping_ub : reduce_pong_ub;

            if (next_buffer == 0 && ping_pending) {
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                ping_pending = false;
            } else if (next_buffer == 1 && pong_pending) {
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
                pong_pending = false;
            }

            const int32_t next_source_pe =
                static_cast<int32_t>(next_rb / slots);
            const int32_t next_source_slot =
                static_cast<int32_t>(next_rb % slots);
            StartAsyncLoad(
                next_reduce_ub,
                reduce + static_cast<int64_t>(next_source_slot) * matrix + tile_begin,
                elements,
                next_source_pe,
                rank,
                next_event);
        }

        auto reduce_local = MakeFp32Local(
            current_reduce_ub, elements, TPosition::VECIN);
        Add(accum_out, accum_in, reduce_local, elements);

        SetFlag<HardEvent::V_MTE2>(current_event);
        if (current_buffer == 0) {
            ping_pending = true;
        } else {
            pong_pending = true;
        }

        current_rb = next_rb;
        current_buffer = 1 - current_buffer;
    }

    if (ping_pending) {
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }
    if (pong_pending) {
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    }

    StoreFp32(grads + local_expert * matrix + tile_begin, accum_out, elements);
}

__aicore__ inline void InitZeroTile(__ubuf__ float* zero_ub)
{
    auto zero_local = MakeFp32Local(
        zero_ub, kTileElements, TPosition::VECOUT);
    Duplicate(zero_local, 0.0F, kTileElements);
    PipeBarrier<PIPE_ALL>();
}

// Partition local storage by cacheline so adjacent slots are never written
// to the same cacheline by different AIV workers.
__aicore__ inline void ClearLocalSlots(
    __gm__ float* reduce,
    const __gm__ int32_t* plan,
    __ubuf__ float* zero_ub,
    int64_t matrix,
    int32_t slots,
    int32_t rank,
    int32_t block,
    int32_t block_count)
{
    if (slots <= 0) {
        return;
    }
    constexpr int64_t local_begin = 0;
    const int64_t local_end = static_cast<int64_t>(slots) * matrix;
    const int64_t first_line = local_begin / kFp32PerCacheline;
    const int64_t line_end = CeilDiv(local_end, kFp32PerCacheline);
    const int64_t total_chunks =
        CeilDiv(line_end - first_line, kClearTileCachelines);
    if (block >= total_chunks) {
        return;
    }
    InitZeroTile(zero_ub);
    auto zero_local = MakeFp32Local(
        zero_ub, kTileElements, TPosition::VECOUT);
    for (int64_t chunk = block; chunk < total_chunks; chunk += block_count) {
        const int64_t chunk_first_line =
            first_line + chunk * kClearTileCachelines;
        const int64_t chunk_line_end =
            chunk_first_line + kClearTileCachelines < line_end ?
                chunk_first_line + kClearTileCachelines : line_end;
        const int64_t chunk_begin =
            chunk_first_line * kFp32PerCacheline > local_begin ?
                chunk_first_line * kFp32PerCacheline : local_begin;
        const int64_t chunk_end =
            chunk_line_end * kFp32PerCacheline < local_end ?
                chunk_line_end * kFp32PerCacheline : local_end;
        const int32_t first_slot = static_cast<int32_t>(
            (chunk_begin - local_begin) / matrix);
        const int32_t last_slot = static_cast<int32_t>(
            (chunk_end - 1 - local_begin) / matrix);
        for (int32_t slot = first_slot; slot <= last_slot; ++slot) {
            if (plan[static_cast<int64_t>(rank) * slots + slot] < 0) {
                continue;
            }
            const int64_t slot_begin = static_cast<int64_t>(slot) * matrix;
            const int64_t slot_end = slot_begin + matrix;
            const int64_t begin =
                chunk_begin > slot_begin ? chunk_begin : slot_begin;
            const int64_t end =
                chunk_end < slot_end ? chunk_end : slot_end;
            StoreFp32(
                reduce + begin,
                zero_local,
                static_cast<int32_t>(end - begin));
        }
    }
}

}  // namespace gradReduce
}  // namespace moonEP

using namespace moonEP::gradReduce;

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void MoonEPShmemGradReduceFused(
    uint64_t ffts_addr,
    GM_ADDR expert_grads_addr,
    GM_ADDR reduce_buffers_addr,
    GM_ADDR experts_to_copy_addr,
    int32_t epn,
    int32_t hidden,
    int32_t projection,
    int32_t ranks,
    int32_t slots,
    int32_t rank)
{
    util_set_ffts_config(ffts_addr);
    if ASCEND_IS_AIC {
        return;
    }

    auto* grads = reinterpret_cast<__gm__ float*>(expert_grads_addr);
    auto* reduce = reinterpret_cast<__gm__ float*>(reduce_buffers_addr);
    auto* plan = reinterpret_cast<__gm__ int32_t*>(experts_to_copy_addr);
    const int32_t local_begin = rank * epn;
    const int64_t matrix = static_cast<int64_t>(hidden) * projection;
    const int64_t reduce_entries = static_cast<int64_t>(ranks) * slots;
    const int64_t tiles_per_expert = CeilDiv(matrix, kTileElements);
    const int32_t block = GetWorkerIndex();
    const int32_t block_count = GetBlockNum();

    auto* accum_ub = reinterpret_cast<__ubuf__ float*>(kUbScratchOffset);
    auto* reduce_ping_ub = reinterpret_cast<__ubuf__ float*>(
        kUbScratchOffset + kTileBytes);
    auto* reduce_pong_ub = reinterpret_cast<__ubuf__ float*>(
        kUbScratchOffset + 2 * kTileBytes);

    for (int32_t local_expert = 0; local_expert < epn; ++local_expert) {
        const int32_t global_expert = local_begin + local_expert;
        for (int64_t tile = block; tile < tiles_per_expert; tile += block_count) {
            const int64_t tile_begin = tile * kTileElements;
            const int32_t elements = static_cast<int32_t>(
                matrix - tile_begin < kTileElements ?
                    matrix - tile_begin : kTileElements);
            AccumulatePlanTile(
                grads,
                reduce,
                plan,
                accum_ub,
                reduce_ping_ub,
                reduce_pong_ub,
                local_expert,
                global_expert,
                matrix,
                tile_begin,
                elements,
                rank,
                slots,
                reduce_entries);
        }
    }

    // Every AIV on every rank must reach this collective. Do not add an AIV
    // early return above it, including for empty plans or blocks with no work.
    aclshmem_quiet();
    // This kernel is [[core_ratio(0, 1)]] and launches AIV workers only.
    // The installed CANN-SHMEM generic barrier also includes non-vector
    // participants and deadlocks for this launch shape; use its vector-only
    // device collective so the participant set matches the kernel grid.
    aclshmemx_barrier_all_vec();

    ClearLocalSlots(
        reduce,
        plan,
        accum_ub,
        matrix,
        slots,
        rank,
        block,
        block_count);
}

extern "C" ASCEND_MOONEP_EXPORT int moonep_shmem_grad_reduce(
    void* stream,
    void* expert_grads,
    void* reduce_buffers,
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
        epn <= 0 || hidden <= 0 ||
        projection <= 0 || ranks <= 0 || slots < 0 || rank < 0 ||
        rank >= ranks || blocks <= 0) {
        return ACL_ERROR_INVALID_PARAM;
    }

    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    const int32_t shmem_ranks = aclshmem_n_pes();
    if (shmem_ranks <= 0 || ranks != shmem_ranks) {
        return ACL_ERROR_INVALID_PARAM;
    }

    MoonEPShmemGradReduceFused<<<blocks, nullptr, stream>>>(
        shmemx_get_ffts_config(),
        reinterpret_cast<std::uint8_t*>(expert_grads),
        reinterpret_cast<std::uint8_t*>(reduce_buffers),
        reinterpret_cast<std::uint8_t*>(const_cast<int32_t*>(experts_to_copy)),
        epn,
        hidden,
        projection,
        ranks,
        slots,
        rank);
    return aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
}
