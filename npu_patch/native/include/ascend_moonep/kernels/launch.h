#pragma once

#include <cstdint>

#if defined(__GNUC__)
#define ASCEND_MOONEP_EXPORT __attribute__((visibility("default")))
#else
#define ASCEND_MOONEP_EXPORT
#endif

extern "C" {

ASCEND_MOONEP_EXPORT void device_barrier_bench_kernel_do(
    void* stream,
    int32_t iterations,
    int32_t block_dim);

ASCEND_MOONEP_EXPORT void moonep_combine_kernel_do(
    void* stream,
    uint8_t* hidden_buf,
    uint8_t* weight_buf,
    uint8_t* dst,
    uint8_t* dup_groups,
    uint8_t* dup_loffs,
    uint8_t* dup_counts,
    uint8_t* hidden_sk_out,
    uint8_t* route_weights_sk_out,
    uint8_t* tiling_host);

ASCEND_MOONEP_EXPORT void moonep_udma_combine_kernel_do(
    void* stream,
    uint8_t* hidden_nvsh,
    uint8_t* route_weights_nvs,
    uint8_t* sync_buffer,
    uint8_t* dst,
    uint8_t* dup_groups,
    uint8_t* dup_loffs,
    uint8_t* dup_counts,
    uint8_t* output_sh,
    uint8_t* output_sk,
    uint8_t* local_workspace,
    uint8_t* bandwidth_scratch,
    uint8_t* tiling_host,
    uint64_t invocation_magic);

ASCEND_MOONEP_EXPORT void moonep_dispatch_kernel_do(
    void* stream,
    uint8_t* hidden_sh,
    uint8_t* route_weights,
    uint8_t* dst,
    uint8_t* dup_groups,
    uint8_t* dup_loffs,
    uint8_t* dup_counts,
    uint8_t* hidden_buf,
    uint8_t* route_weights_buf,
    uint8_t* tiling_host);

ASCEND_MOONEP_EXPORT void moonep_dispatch_udma_kernel_do(
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
    uint64_t magic);

ASCEND_MOONEP_EXPORT void moonep_dispatch_udma_profile_kernel_do(
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
    uint64_t magic);

ASCEND_MOONEP_EXPORT void planning_kernel_do(
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
    uint64_t ffts_addr);

ASCEND_MOONEP_EXPORT int moonep_shmem_grad_reduce(
    void* stream,
    void* expert_grads,
    void* reduce_buffers,
    const int32_t* experts_to_copy,
    int32_t experts,
    int32_t hidden,
    int32_t projection,
    int32_t ranks,
    int32_t slots,
    int32_t rank,
    int32_t blocks);

ASCEND_MOONEP_EXPORT int moonep_shmem_grad_reduce_udma(
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
    int32_t blocks);

ASCEND_MOONEP_EXPORT void prefetch_weight_kernel_do(
    void* stream,
    uint8_t* remote_expert,
    uint8_t* prefetch_buffers,
    uint8_t* experts_to_copy,
    int64_t num_experts,
    int64_t hidden,
    int64_t hidden_prime,
    int64_t num_slots,
    int64_t block_dim);

ASCEND_MOONEP_EXPORT void prefetch_weight_kernel_udma_do(
    void* stream,
    uint8_t* remote_expert,
    uint8_t* prefetch_buffers,
    uint8_t* experts_to_copy,
    int64_t num_experts,
    int64_t hidden,
    int64_t hidden_prime,
    int64_t num_slots,
    int64_t block_dim);

}  // extern "C"
