#pragma once

#include <torch/extension.h>

namespace ascend_moonep::ops {

// Caller-owned-output overload selected by ASCEND_MOONEP_COMBINE_USE_UDMA.
// local_workspace is
// ordinary rank-local NPU memory; only the inputs are symmetric.
void MoonEPUdmaCombine(
    torch::Tensor hidden_nvsh,
    c10::optional<torch::Tensor> route_weights_nvs,
    torch::Tensor sync_buffer,
    torch::Tensor dst,
    torch::Tensor dup_groups,
    torch::Tensor dup_loffs,
    torch::Tensor dup_counts,
    torch::Tensor output_sh,
    c10::optional<torch::Tensor> output_sk,
    int64_t S,
    int64_t H,
    int64_t K,
    int64_t R,
    int64_t rank,
    torch::Tensor local_workspace);

}  // namespace ascend_moonep::ops
