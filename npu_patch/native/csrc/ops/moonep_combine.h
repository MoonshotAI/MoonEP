#pragma once

#include <torch/extension.h>

namespace ascend_moonep::ops {

// Plan-driven moonEP_combine on already-symmetric SHMEM shards.
// Outputs are caller-allocated. Host ABI mirrors MoonEPDispatch.
// Provide weight_buf and route_weights_sk_out together, or omit both.
// Launch width is fixed at 32 AIV.
//
// route_weights_sk_out must be fp32 [S, AlignUp(K, 8)] when present
// (kernel DataCopy width).
void MoonEPCombine(
    torch::Tensor hidden_buf,
    c10::optional<torch::Tensor> weight_buf,
    torch::Tensor dst,
    torch::Tensor dup_groups,
    torch::Tensor dup_loffs,
    torch::Tensor dup_counts,
    torch::Tensor hidden_sk_out,
    c10::optional<torch::Tensor> route_weights_sk_out,
    int64_t S,
    int64_t H,
    int64_t K,
    int64_t R,
    int64_t rank,
    bool use_udma);

}  // namespace ascend_moonep::ops
