#include "launcher.h"

#include <acl/acl.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "ascend_moonep/kernels/launch.h"
#include "csrc/common/check.h"

namespace {

aclrtStream GetCurrentAclStream() {
    return c10_npu::getCurrentNPUStream().stream();
}

uint8_t* BytePtr(void* ptr) {
    return static_cast<uint8_t*>(ptr);
}

}  // namespace

void LaunchDeviceBarrierBench(int32_t iterations, int32_t block_dim) {
    TORCH_CHECK(iterations > 0, "iterations must be positive, got ", iterations);
    TORCH_CHECK(block_dim > 0, "block_dim must be positive, got ", block_dim);
    aclrtStream stream = GetCurrentAclStream();
    device_barrier_bench_kernel_do(stream, iterations, block_dim);
}

void SynchronizeCurrentNpuStream() {
    ASCEND_MOONEP_CHECK_ACL(aclrtSynchronizeStream(GetCurrentAclStream()));
}

void LaunchMoonEPCombineKernel(
    torch::Tensor hidden_buf,
    torch::Tensor weight_buf,
    torch::Tensor dst,
    torch::Tensor dup_groups,
    torch::Tensor dup_loffs,
    torch::Tensor dup_counts,
    torch::Tensor hidden_sk_out,
    torch::Tensor route_weights_sk_out,
    const ascend_moonep::MoonEPCombineTilingData& tiling) {
    aclrtStream stream = GetCurrentAclStream();
    moonep_combine_kernel_do(
        stream,
        BytePtr(hidden_buf.data_ptr()),
        BytePtr(weight_buf.data_ptr()),
        BytePtr(dst.data_ptr()),
        BytePtr(dup_groups.data_ptr()),
        BytePtr(dup_loffs.data_ptr()),
        BytePtr(dup_counts.data_ptr()),
        BytePtr(hidden_sk_out.data_ptr()),
        BytePtr(route_weights_sk_out.data_ptr()),
        reinterpret_cast<uint8_t*>(
            const_cast<ascend_moonep::MoonEPCombineTilingData*>(&tiling)));
}

void LaunchMoonEPDispatchKernel(
    torch::Tensor hidden_sh,
    torch::Tensor route_weights,
    torch::Tensor dst,
    torch::Tensor dup_groups,
    torch::Tensor dup_loffs,
    torch::Tensor dup_counts,
    torch::Tensor hidden_buf,
    torch::Tensor route_weights_buf,
    const ascend_moonep::MoonEPDispatchTilingData& tiling) {
    aclrtStream stream = GetCurrentAclStream();
    moonep_dispatch_kernel_do(
        stream,
        BytePtr(hidden_sh.data_ptr()),
        BytePtr(route_weights.data_ptr()),
        BytePtr(dst.data_ptr()),
        BytePtr(dup_groups.data_ptr()),
        BytePtr(dup_loffs.data_ptr()),
        BytePtr(dup_counts.data_ptr()),
        BytePtr(hidden_buf.data_ptr()),
        BytePtr(route_weights_buf.data_ptr()),
        reinterpret_cast<uint8_t*>(
            const_cast<ascend_moonep::MoonEPDispatchTilingData*>(&tiling)));
}

void LaunchMoonEPDispatchUdmaKernel(
    torch::Tensor hidden_sh,
    torch::Tensor route_weights,
    torch::Tensor dst,
    torch::Tensor dup_groups,
    torch::Tensor dup_loffs,
    torch::Tensor dup_counts,
    torch::Tensor hidden_buf,
    torch::Tensor route_weights_buf,
    torch::Tensor sync_buffer,
    const ascend_moonep::MoonEPDispatchTilingData& tiling,
    uint64_t magic) {
#if defined(ASCEND_MOONEP_UDMA)
    aclrtStream stream = GetCurrentAclStream();
    moonep_dispatch_udma_kernel_do(
        stream,
        BytePtr(hidden_sh.data_ptr()),
        BytePtr(route_weights.data_ptr()),
        BytePtr(dst.data_ptr()),
        BytePtr(dup_groups.data_ptr()),
        BytePtr(dup_loffs.data_ptr()),
        BytePtr(dup_counts.data_ptr()),
        BytePtr(hidden_buf.data_ptr()),
        BytePtr(route_weights_buf.data_ptr()),
        BytePtr(sync_buffer.data_ptr()),
        reinterpret_cast<uint8_t*>(
            const_cast<ascend_moonep::MoonEPDispatchTilingData*>(&tiling)),
        magic);
#else
    (void)hidden_sh; (void)route_weights; (void)dst; (void)dup_groups;
    (void)dup_loffs; (void)dup_counts; (void)hidden_buf;
    (void)route_weights_buf; (void)sync_buffer; (void)tiling; (void)magic;
    TORCH_CHECK(false,
        "dispatch UDMA kernel support is not enabled in this build; "
        "A2/A3 do not provide the UDMA dispatch kernel");
#endif
}

void LaunchMoonEPDispatchUdmaProfileKernel(
    torch::Tensor hidden_sh,
    torch::Tensor route_weights,
    torch::Tensor dst,
    torch::Tensor dup_groups,
    torch::Tensor dup_loffs,
    torch::Tensor dup_counts,
    torch::Tensor hidden_buf,
    torch::Tensor route_weights_buf,
    torch::Tensor sync_buffer,
    torch::Tensor profile_cycles,
    const ascend_moonep::MoonEPDispatchTilingData& tiling,
    uint64_t magic) {
#if defined(ASCEND_MOONEP_UDMA)
    aclrtStream stream = GetCurrentAclStream();
    moonep_dispatch_udma_profile_kernel_do(
        stream,
        BytePtr(hidden_sh.data_ptr()),
        BytePtr(route_weights.data_ptr()),
        BytePtr(dst.data_ptr()),
        BytePtr(dup_groups.data_ptr()),
        BytePtr(dup_loffs.data_ptr()),
        BytePtr(dup_counts.data_ptr()),
        BytePtr(hidden_buf.data_ptr()),
        BytePtr(route_weights_buf.data_ptr()),
        BytePtr(sync_buffer.data_ptr()),
        BytePtr(profile_cycles.data_ptr()),
        reinterpret_cast<uint8_t*>(
            const_cast<ascend_moonep::MoonEPDispatchTilingData*>(&tiling)),
        magic);
#else
    (void)hidden_sh; (void)route_weights; (void)dst; (void)dup_groups;
    (void)dup_loffs; (void)dup_counts; (void)hidden_buf;
    (void)route_weights_buf; (void)sync_buffer; (void)profile_cycles;
    (void)tiling; (void)magic;
    TORCH_CHECK(false,
        "dispatch profile UDMA kernel support is not enabled in this build; "
        "A2/A3 do not provide the UDMA dispatch kernel");
#endif
}

void LaunchPlanningKernel(
    torch::Tensor topk_idx,
    torch::Tensor dst,
    torch::Tensor cu_seqlens,
    torch::Tensor experts_to_copy,
    torch::Tensor dup_groups,
    torch::Tensor dup_loffs,
    torch::Tensor dup_counts,
    torch::Tensor tpe_all,
    torch::Tensor tpe_prefix,
    torch::Tensor balance,
    torch::Tensor z_tensor,
    torch::Tensor alloc_tensor,
    torch::Tensor alloc_cumsum,
    torch::Tensor expert_offsets,
    torch::Tensor core_expert_counts,
    torch::Tensor dup_workspace,
    torch::Tensor dup_key_workspace,
    const ascend_moonep::PlanningTilingData& tiling,
    void* symmetric_buffer,
    uint64_t ffts_addr) {
    TORCH_CHECK(symmetric_buffer != nullptr,
                "planning symmetric_buffer must not be null");
    aclrtStream stream = GetCurrentAclStream();
    planning_kernel_do(
        stream,
        BytePtr(topk_idx.data_ptr()),
        BytePtr(symmetric_buffer),
        BytePtr(dst.data_ptr()),
        BytePtr(cu_seqlens.data_ptr()),
        BytePtr(experts_to_copy.data_ptr()),
        BytePtr(dup_groups.data_ptr()),
        BytePtr(dup_loffs.data_ptr()),
        BytePtr(dup_counts.data_ptr()),
        BytePtr(tpe_all.data_ptr()),
        BytePtr(tpe_prefix.data_ptr()),
        BytePtr(balance.data_ptr()),
        BytePtr(z_tensor.data_ptr()),
        BytePtr(alloc_tensor.data_ptr()),
        BytePtr(alloc_cumsum.data_ptr()),
        BytePtr(expert_offsets.data_ptr()),
        BytePtr(core_expert_counts.data_ptr()),
        BytePtr(dup_workspace.data_ptr()),
        BytePtr(dup_key_workspace.data_ptr()),
        reinterpret_cast<uint8_t*>(
            const_cast<ascend_moonep::PlanningTilingData*>(&tiling)),
        ffts_addr);
}

void LaunchMoonEPUdmaCombineKernel(
    torch::Tensor hidden_nvsh,
    torch::Tensor route_weights_nvs,
    torch::Tensor sync_buffer,
    torch::Tensor dst,
    torch::Tensor dup_groups,
    torch::Tensor dup_loffs,
    torch::Tensor dup_counts,
    torch::Tensor output_sh,
    torch::Tensor output_sk,
    void* bandwidth_scratch,
    const ascend_moonep::MoonEPUdmaCombineTilingData& tiling,
    void* local_workspace,
    uint64_t invocation_magic) {
#if defined(ASCEND_MOONEP_UDMA)
    TORCH_CHECK(local_workspace != nullptr, "local_workspace must not be null");
    TORCH_CHECK(bandwidth_scratch != nullptr,
                "bandwidth_scratch must not be null");

    aclrtStream stream = GetCurrentAclStream();
    moonep_udma_combine_kernel_do(
        stream,
        BytePtr(hidden_nvsh.data_ptr()),
        BytePtr(route_weights_nvs.data_ptr()),
        BytePtr(sync_buffer.data_ptr()),
        BytePtr(dst.data_ptr()),
        BytePtr(dup_groups.data_ptr()),
        BytePtr(dup_loffs.data_ptr()),
        BytePtr(dup_counts.data_ptr()),
        BytePtr(output_sh.data_ptr()),
        BytePtr(output_sk.data_ptr()),
        BytePtr(local_workspace),
        BytePtr(bandwidth_scratch),
        reinterpret_cast<uint8_t*>(
            const_cast<ascend_moonep::MoonEPUdmaCombineTilingData*>(&tiling)),
        invocation_magic);
#else
    (void)hidden_nvsh; (void)route_weights_nvs; (void)sync_buffer; (void)dst;
    (void)dup_groups; (void)dup_loffs; (void)dup_counts; (void)output_sh;
    (void)output_sk; (void)bandwidth_scratch; (void)tiling;
    (void)local_workspace; (void)invocation_magic;
    TORCH_CHECK(false,
        "combine UDMA kernel support is not enabled in this build; "
        "A2/A3 do not provide the UDMA combine kernel");
#endif
}
