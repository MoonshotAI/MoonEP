#pragma once

#include <torch/extension.h>

namespace ascend_moonep::ops {

// Plan-driven fused MoonEP dispatch + local duplicate expansion.
// Writes the local result directly into SHMEM-backed output tensors.
void
MoonEPDispatch(
    torch::Tensor hidden_sh,
    c10::optional<torch::Tensor> route_weights,
    torch::Tensor dst,
    torch::Tensor dup_groups,
    torch::Tensor dup_loffs,
    torch::Tensor dup_counts,
    torch::Tensor hidden_buf,
    c10::optional<torch::Tensor> route_weights_buf,
    torch::Tensor sync_buffer,
    int64_t S,
    int64_t H,
    int64_t K,
    int64_t R,
    int64_t rank,
    bool use_udma);

void
MoonEPDispatchProfile(
    torch::Tensor hidden_sh,
    c10::optional<torch::Tensor> route_weights,
    torch::Tensor dst,
    torch::Tensor dup_groups,
    torch::Tensor dup_loffs,
    torch::Tensor dup_counts,
    torch::Tensor hidden_buf,
    c10::optional<torch::Tensor> route_weights_buf,
    torch::Tensor sync_buffer,
    torch::Tensor profile_cycles,
    int64_t S,
    int64_t H,
    int64_t K,
    int64_t R,
    int64_t rank,
    bool use_udma);

}  // namespace ascend_moonep::ops
