#include "csrc/ops/prefetch.h"

#include <acl/acl.h>
#include <algorithm>
#include <cstdint>

#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "shmem.h"
#include "ascend_moonep/kernels/launch.h"
#include "csrc/common/check.h"
#include "csrc/shmem_instance.h"

namespace {

void CheckNpuContiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(
        tensor.device().type() == c10::DeviceType::PrivateUse1,
        name,
        " must be an NPU tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void CheckNpu(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(
        tensor.device().type() == c10::DeviceType::PrivateUse1,
        name,
        " must be an NPU tensor");
}

}  // namespace

namespace ascend_moonep::ops {

torch::Tensor LaunchPrefetch(
    torch::Tensor remote_expert,
    torch::Tensor prefetch_buffers,
    torch::Tensor experts_to_copy,
    int64_t num_sms,
    int64_t instance_id,
    bool use_udma) {
    // The current SHMEM runtime is a single global instance.
    (void)instance_id;
    (void)num_sms;
    // remote_expert is a logical cross-PE view. Its data pointer is the local
    // SHMEM chunk base; the device kernel resolves peer bases with
    // aclshmem_ptr and therefore does not require a globally contiguous view.
    CheckNpu(remote_expert, "remote_expert");
    CheckNpuContiguous(prefetch_buffers, "prefetch_buffers");
    CheckNpuContiguous(experts_to_copy, "experts_to_copy");
    TORCH_CHECK(
        remote_expert.device() == prefetch_buffers.device() &&
            remote_expert.device() == experts_to_copy.device(),
        "remote_expert, prefetch_buffers and experts_to_copy must be on the same device");
    TORCH_CHECK(
        remote_expert.scalar_type() == torch::kBFloat16,
        "remote_expert must be bfloat16");
    TORCH_CHECK(
        prefetch_buffers.scalar_type() == torch::kBFloat16,
        "prefetch_buffers must be bfloat16");
    TORCH_CHECK(
        experts_to_copy.scalar_type() == torch::kInt32,
        "experts_to_copy must be int32");
    TORCH_CHECK(
        remote_expert.dim() == 3,
        "remote_expert must have shape [E, H, H']");
    TORCH_CHECK(
        prefetch_buffers.dim() == 3,
        "prefetch_buffers must have shape [B, H, H']");
    TORCH_CHECK(
        experts_to_copy.dim() == 1,
        "experts_to_copy must have shape [B]");
    TORCH_CHECK(
        remote_expert.size(1) == prefetch_buffers.size(1) &&
            remote_expert.size(2) == prefetch_buffers.size(2),
        "remote_expert and prefetch_buffers must have the same H and H'");
    TORCH_CHECK(
        experts_to_copy.numel() == prefetch_buffers.size(0),
        "experts_to_copy length must equal prefetch_buffers.shape[0]");

    const int64_t E = remote_expert.size(0);
    const int64_t H = remote_expert.size(1);
    const int64_t Hp = remote_expert.size(2);
    const int64_t B = prefetch_buffers.size(0);
    TORCH_CHECK(E > 0 && H > 0 && Hp > 0 && B > 0,
                "prefetch dimensions must be positive");

    int32_t device_id = 0;
    int64_t aicore_count = 0;
    ASCEND_MOONEP_CHECK_ACL(aclrtGetDevice(&device_id));
    ASCEND_MOONEP_CHECK_ACL(aclrtGetDeviceInfo(
        device_id,
        ACL_DEV_ATTR_AICORE_CORE_NUM,
        &aicore_count));
    TORCH_CHECK(aicore_count > 0, "current NPU has no available AICore");
    const int64_t block_dim = aicore_count;

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    aclshmemx_barrier_all_on_stream(stream);
    auto kernel = use_udma
        ? prefetch_weight_kernel_udma_do
        : prefetch_weight_kernel_do;
    kernel(
        stream,
        static_cast<uint8_t*>(remote_expert.data_ptr()),
        static_cast<uint8_t*>(prefetch_buffers.data_ptr()),
        static_cast<uint8_t*>(experts_to_copy.data_ptr()),
        E,
        H,
        Hp,
        B,
        block_dim);
    aclshmemx_barrier_all_on_stream(stream);
    return prefetch_buffers;
}

}  // namespace ascend_moonep::ops
