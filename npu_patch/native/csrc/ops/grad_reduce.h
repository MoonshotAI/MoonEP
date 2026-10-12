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

}  // namespace ascend_moonep::ops
