#include "csrc/ops/moonep_combine.h"

#include <cstdint>
#include <limits>

#include "ascend_moonep/kernels/types.h"
#include "csrc/launcher.h"

namespace {

void CheckNpuContiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
                name, " must be an NPU tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void CheckSameDevice(
    const torch::Tensor& tensor,
    const torch::Tensor& reference,
    const char* name) {
    TORCH_CHECK(tensor.device() == reference.device(),
                name, " must be on the same NPU device as hidden_buf");
}

int64_t AlignUp8(int64_t n) {
    return ((n + 7) / 8) * 8;
}

}  // namespace

namespace ascend_moonep::ops {

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
    bool use_udma) {
    (void)use_udma;
    CheckNpuContiguous(hidden_buf, "hidden_buf");
    CheckNpuContiguous(dst, "dst");
    CheckNpuContiguous(dup_groups, "dup_groups");
    CheckNpuContiguous(dup_loffs, "dup_loffs");
    CheckNpuContiguous(dup_counts, "dup_counts");
    CheckNpuContiguous(hidden_sk_out, "hidden_sk_out");

    TORCH_CHECK(S > 0 && H > 0 && K > 0 && R > 0,
                "S/H/K/R must be positive");
    TORCH_CHECK(rank >= 0 && rank < R, "rank must be in [0, R)");
    TORCH_CHECK(K <= 16, "K must be <= 16");
    TORCH_CHECK(static_cast<uint64_t>(S) <= std::numeric_limits<uint32_t>::max() &&
                static_cast<uint64_t>(H) <= std::numeric_limits<uint32_t>::max() &&
                static_cast<uint64_t>(K) <= std::numeric_limits<uint32_t>::max() &&
                static_cast<uint64_t>(R) <= std::numeric_limits<uint32_t>::max(),
                "S/H/K/R must fit in uint32 tiling fields");
    const uint64_t num_slots_u64 =
        static_cast<uint64_t>(S) * static_cast<uint64_t>(K);
    TORCH_CHECK(num_slots_u64 <= std::numeric_limits<uint32_t>::max(),
                "S*K must fit in uint32");
    TORCH_CHECK(num_slots_u64 <= std::numeric_limits<uint32_t>::max() / 3,
                "S*K is too large for dup_groups indexing");
    const int64_t N = static_cast<int64_t>(num_slots_u64);
    const int64_t sk_stride = AlignUp8(K);

    TORCH_CHECK(hidden_buf.scalar_type() == torch::kBFloat16,
                "hidden_buf must be bf16");
    // Allow oversized symmetric capacity (suite may pin max-large then reuse).
    TORCH_CHECK(hidden_buf.dim() == 2 &&
                hidden_buf.size(0) >= N && hidden_buf.size(1) == H,
                "hidden_buf must be bf16 [>=S*K, H]");
    CheckSameDevice(dst, hidden_buf, "dst");
    CheckSameDevice(dup_groups, hidden_buf, "dup_groups");
    CheckSameDevice(dup_loffs, hidden_buf, "dup_loffs");
    CheckSameDevice(dup_counts, hidden_buf, "dup_counts");
    CheckSameDevice(hidden_sk_out, hidden_buf, "hidden_sk_out");

    const bool with_weight = weight_buf.has_value();
    TORCH_CHECK(route_weights_sk_out.has_value() == with_weight,
                "route_weights_sk_out must be provided exactly when weight_buf is provided");
    if (with_weight) {
        CheckNpuContiguous(*weight_buf, "weight_buf");
        CheckSameDevice(*weight_buf, hidden_buf, "weight_buf");
        TORCH_CHECK(weight_buf->scalar_type() == torch::kFloat32 &&
                    weight_buf->dim() == 1 && weight_buf->size(0) >= N,
                    "weight_buf must be fp32 [>=S*K]");
        CheckNpuContiguous(*route_weights_sk_out, "route_weights_sk_out");
        CheckSameDevice(*route_weights_sk_out, hidden_buf, "route_weights_sk_out");
        TORCH_CHECK(route_weights_sk_out->scalar_type() == torch::kFloat32 &&
                    route_weights_sk_out->dim() == 2 &&
                    route_weights_sk_out->size(0) == S &&
                    route_weights_sk_out->size(1) == sk_stride,
                    "route_weights_sk_out must be fp32 [S, AlignUp(K, 8)]");
    }

    TORCH_CHECK(hidden_sk_out.scalar_type() == torch::kBFloat16,
                "hidden_sk_out must be bf16");
    TORCH_CHECK(hidden_sk_out.dim() == 2 &&
                hidden_sk_out.size(0) == S && hidden_sk_out.size(1) == H,
                "hidden_sk_out must be bf16 [S, H]");

    TORCH_CHECK(dst.scalar_type() == torch::kInt32, "dst must be int32");
    TORCH_CHECK(dst.dim() == 1 && dst.size(0) == N,
                "dst must be int32 [S*K]");
    TORCH_CHECK(dup_groups.scalar_type() == torch::kInt32 &&
                dup_groups.dim() == 2 &&
                dup_groups.size(0) == N && dup_groups.size(1) == 3,
                "dup_groups must be int32 [S*K, 3]");
    TORCH_CHECK(dup_loffs.scalar_type() == torch::kInt32 &&
                dup_loffs.dim() == 1 && dup_loffs.size(0) == N,
                "dup_loffs must be int32 [S*K]");
    TORCH_CHECK(dup_counts.scalar_type() == torch::kInt32 &&
                dup_counts.dim() == 1 && dup_counts.size(0) == 2,
                "dup_counts must be int32 [2]");

    TORCH_CHECK(H % 16 == 0, "H must be multiple of 16");
    TORCH_CHECK(static_cast<uint64_t>(H) * 10 + 256 <= 192ull * 1024ull,
                "H too large for single-pass combine UB (max ~19680)");

    // Placeholders when weights are omitted (launcher ignores them via tiling).
    const torch::Tensor& weight_arg =
        with_weight ? *weight_buf : hidden_buf;
    const torch::Tensor& route_weights_sk_arg =
        with_weight ? *route_weights_sk_out : hidden_sk_out;

    MoonEPCombineTilingData tiling{};
    tiling.rank = static_cast<uint32_t>(rank);
    tiling.world_size = static_cast<uint32_t>(R);
    tiling.num_tokens = static_cast<uint32_t>(S);
    tiling.hidden = static_cast<uint32_t>(H);
    tiling.num_topk = static_cast<uint32_t>(K);
    tiling.block_dim = 32u;
    tiling.with_weight = with_weight ? 1u : 0u;
    tiling.reserved = 0u;

    LaunchMoonEPCombineKernel(
        hidden_buf,
        weight_arg,
        dst,
        dup_groups,
        dup_loffs,
        dup_counts,
        hidden_sk_out,
        route_weights_sk_arg,
        tiling);
}

}  // namespace ascend_moonep::ops
