#include "csrc/ops/grad_reduce.h"

#include <limits>

#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "ascend_moonep/kernels/launch.h"

namespace ascend_moonep::ops {

namespace {

void CheckNpuContiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(
        tensor.device().type() == c10::DeviceType::PrivateUse1,
        name,
        " must be an NPU tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void CheckInt32(int64_t value, const char* name) {
    TORCH_CHECK(
        value >= std::numeric_limits<int32_t>::min() &&
            value <= std::numeric_limits<int32_t>::max(),
        name,
        " is outside int32 range");
}

void CheckCommon(
    const torch::Tensor& reduce_buffers,
    const torch::Tensor& experts_to_copy,
    int64_t hidden,
    int64_t projection,
    int64_t ranks,
    int64_t slots,
    int64_t rank,
    int64_t blocks) {
    CheckNpuContiguous(reduce_buffers, "reduce_buffers");
    CheckNpuContiguous(experts_to_copy, "experts_to_copy");
    TORCH_CHECK(
        reduce_buffers.scalar_type() == torch::kFloat32,
        "reduce_buffers must be float32");
    TORCH_CHECK(
        experts_to_copy.scalar_type() == torch::kInt32,
        "experts_to_copy must be int32");
    TORCH_CHECK(
        reduce_buffers.device() == experts_to_copy.device(),
        "reduce_buffers and experts_to_copy must be on the same device");
    TORCH_CHECK(
        hidden > 0 && projection > 0 && ranks > 0 && slots >= 0 && rank >= 0 &&
            rank < ranks && blocks > 0,
        "invalid grad_reduce dimensions/rank/blocks");
    TORCH_CHECK(
        reduce_buffers.dim() == 3 && reduce_buffers.size(0) == slots &&
            reduce_buffers.size(1) == hidden &&
            reduce_buffers.size(2) == projection,
        "reduce_buffers must have shape [slots, hidden, projection]");
    TORCH_CHECK(
        experts_to_copy.dim() == 2 && experts_to_copy.size(0) == ranks &&
            experts_to_copy.size(1) == slots,
        "experts_to_copy must have shape [ranks, slots]");
    CheckInt32(hidden, "hidden");
    CheckInt32(projection, "projection");
    CheckInt32(ranks, "ranks");
    CheckInt32(slots, "slots");
    CheckInt32(rank, "rank");
    CheckInt32(blocks, "blocks");
}

}  // namespace

void GradReduce(
    torch::Tensor expert_grads,
    torch::Tensor reduce_buffers,
    torch::Tensor experts_to_copy,
    int64_t epn,
    int64_t hidden,
    int64_t projection,
    int64_t ranks,
    int64_t slots,
    int64_t rank,
    int64_t blocks,
    bool use_udma) {
    // The current grad-reduce implementation has no separate UDMA kernel.
    // Keep the unified ABI and deliberately use the existing SHMEM path.
    (void)use_udma;
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

    CheckNpuContiguous(expert_grads, "expert_grads");
    TORCH_CHECK(
        expert_grads.scalar_type() == torch::kFloat32,
        "expert_grads must be float32");
    TORCH_CHECK(
        expert_grads.device() == reduce_buffers.device(),
        "expert_grads and reduce_buffers must be on the same device");
    TORCH_CHECK(
        expert_grads.dim() == 3 && expert_grads.size(0) == epn &&
            expert_grads.size(1) == hidden &&
            expert_grads.size(2) == projection,
        "expert_grads must have shape [epn, hidden, projection]");
    CheckCommon(
        reduce_buffers,
        experts_to_copy,
        hidden,
        projection,
        ranks,
        slots,
        rank,
        blocks);
    TORCH_CHECK(
        epn > 0,
        "epn must be positive");
    CheckInt32(epn, "epn");
    if (use_udma) {
        torch::Tensor workspace = torch::empty(512 * 1024 * (blocks * 2), expert_grads.options());
        const int ret = moonep_shmem_grad_reduce_udma(
            stream,
            expert_grads.data_ptr(),
            reduce_buffers.data_ptr(),
            workspace.data_ptr(),
            experts_to_copy.data_ptr<int32_t>(),
            static_cast<int32_t>(epn),
            static_cast<int32_t>(hidden),
            static_cast<int32_t>(projection),
            static_cast<int32_t>(ranks),
            static_cast<int32_t>(slots),
            static_cast<int32_t>(rank),
            static_cast<int32_t>(blocks));
        TORCH_CHECK(ret == 0, "grad_reduce kernel failed with ACL error: ", ret);
    } else {
        const int ret = moonep_shmem_grad_reduce(
            stream,
            expert_grads.data_ptr(),
            reduce_buffers.data_ptr(),
            experts_to_copy.data_ptr<int32_t>(),
            static_cast<int32_t>(epn),
            static_cast<int32_t>(hidden),
            static_cast<int32_t>(projection),
            static_cast<int32_t>(ranks),
            static_cast<int32_t>(slots),
            static_cast<int32_t>(rank),
            static_cast<int32_t>(blocks));
        TORCH_CHECK(ret == 0, "grad_reduce kernel failed with ACL error: ", ret);
    }
}

}  // namespace ascend_moonep::ops
