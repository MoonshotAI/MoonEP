#include "csrc/ops/prefetch.h"

#include <acl/acl.h>
#include <algorithm>
#include <cstdint>
#include <limits>

#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "ascend_moonep/kernels/launch.h"
#include "csrc/common/check.h"
#include "csrc/barrier.h"
#include "csrc/shmem_instance.h"
#include "shmem.h"

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
#if defined(ASCEND_MOONEP_UDMA)
    auto kernel = use_udma
        ? prefetch_weight_kernel_udma_do
        : prefetch_weight_kernel_do;
#else
    TORCH_CHECK(!use_udma,
        "prefetch UDMA kernel support is not enabled in this build; "
        "A2/A3 do not provide the UDMA prefetch kernel");
    auto kernel = prefetch_weight_kernel_do;
#endif
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
    return prefetch_buffers;
}


namespace {
int32_t PrefetchMteBlocks() {
    int32_t device_id = 0;
    int64_t cores = 0;
    ASCEND_MOONEP_CHECK_ACL(aclrtGetDevice(&device_id));
    ASCEND_MOONEP_CHECK_ACL(
        aclrtGetDeviceInfo(device_id, ACL_DEV_ATTR_AICORE_CORE_NUM, &cores));
    TORCH_CHECK(cores > 0 && cores <= std::numeric_limits<int32_t>::max(),
                "current NPU reports an invalid AICore count");
    return static_cast<int32_t>(cores);
}

int64_t CheckPushProjection(const torch::Tensor& source, const torch::Tensor& slots,
    const torch::Tensor& plan, int64_t local_experts, bool use_udma) {
    CheckNpuContiguous(source, "source");
    CheckNpuContiguous(slots, "slots");
    CheckNpuContiguous(plan, "plan");
    TORCH_CHECK(source.device() == slots.device() && slots.device() == plan.device(),
                "source, slots and plan must share one NPU");
    TORCH_CHECK(slots.scalar_type() == torch::kBFloat16 &&
                plan.scalar_type() == torch::kInt32 && slots.dim() == 3 && plan.dim() == 2,
                "expected BF16 slots[B,M,N] and INT32 plan[EP,B]");
    const int64_t ep = plan.size(0);
    TORCH_CHECK(ep > 0 && ep == aclshmem_n_pes() && local_experts > 0 &&
                local_experts <= INT32_MAX / ep,
                "local expert count or SHMEM topology differs");
    TORCH_CHECK(plan.size(1) == slots.size(0) && slots.size(0) > 0 &&
                (!use_udma || slots.size(0) <= 15736) &&
                slots.size(1) > 0 && slots.size(2) > 0 &&
                slots.size(1) <= std::numeric_limits<int64_t>::max() / slots.size(2),
                "invalid projection, slots or backend UB extent");
    return slots.size(1) * slots.size(2);
}
}

torch::Tensor LaunchPrefetchPushPointers(torch::Tensor pointers, torch::Tensor slots,
    torch::Tensor plan, int64_t local_experts, bool use_udma) {
    if (use_udma) {
        TORCH_CHECK(prefetch_pointer_push_udma_available() == 1,
                    "URMA pointer PUSH unavailable; no MTE fallback");
    }
    const int64_t elements = CheckPushProjection(pointers, slots, plan, local_experts, use_udma);
    TORCH_CHECK(pointers.scalar_type() == torch::kInt64 && pointers.dim() == 1 &&
                pointers.numel() == local_experts, "expected INT64 pointer table[L]");
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    BarrierAllOnStream(stream);
    if (use_udma) {
        ASCEND_MOONEP_CHECK_ACL(moonep_prefetch_pointer_push_udma(stream,
            reinterpret_cast<uint8_t*>(pointers.data_ptr()),
            reinterpret_cast<uint8_t*>(slots.data_ptr()),
            reinterpret_cast<uint8_t*>(plan.data_ptr()),
            local_experts * plan.size(0), elements, slots.size(0)));
    } else {
        ASCEND_MOONEP_CHECK_ACL(moonep_prefetch_pointer_push_mte(stream,
            reinterpret_cast<const uint64_t*>(pointers.data_ptr<int64_t>()),
            slots.data_ptr(), plan.data_ptr<int32_t>(), elements,
            static_cast<int32_t>(local_experts), static_cast<int32_t>(plan.size(0)),
            static_cast<int32_t>(slots.size(0)), static_cast<int32_t>(aclshmem_my_pe()),
            PrefetchMteBlocks()));
    }
    BarrierAllOnStream(stream);
    return slots;
}

}  // namespace ascend_moonep::ops
