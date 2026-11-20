#pragma once

#include <cstdint>

#include <torch/extension.h>

namespace ascend_moonep::ops {

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
    bool use_udma);

// Extension for canonical targets; the dense entry and its backend contract
// remain unchanged. use_udma selects MTE or UDMA; UDMA requires caller-owned
// local workspace. Both use pre/post barriers.
void GradReduceTargets(
    torch::Tensor target_ptrs,
    torch::Tensor reduce_buffers,
    torch::Tensor experts_to_copy,
    int64_t epn,
    int64_t hidden,
    int64_t projection,
    int64_t ranks,
    int64_t slots,
    int64_t rank,
    int64_t blocks,
    bool use_udma = false,
    torch::Tensor udma_workspace = {});

}  // namespace ascend_moonep::ops
