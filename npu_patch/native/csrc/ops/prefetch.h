#pragma once

#include <torch/extension.h>

namespace ascend_moonep::ops {

torch::Tensor LaunchPrefetch(
    torch::Tensor remote_expert,
    torch::Tensor prefetch_buffers,
    torch::Tensor experts_to_copy,
    int64_t num_sms,
    int64_t instance_id,
    bool use_udma);

}  // namespace ascend_moonep::ops
