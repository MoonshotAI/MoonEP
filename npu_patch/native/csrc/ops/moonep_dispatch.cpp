#include "csrc/ops/moonep_dispatch.h"

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
                name, " must be on the same NPU device as hidden_sh");
}

}  // namespace

namespace ascend_moonep::ops {

namespace {

struct CheckedDispatchArgs {
    torch::Tensor route_weights;
    torch::Tensor route_weights_buf;
    MoonEPDispatchTilingData tiling;
};

CheckedDispatchArgs CheckDispatchArgs(
    torch::Tensor hidden_sh,
    c10::optional<torch::Tensor> route_weights,
    torch::Tensor dst,
    torch::Tensor dup_groups,
    torch::Tensor dup_loffs,
    torch::Tensor dup_counts,
    torch::Tensor hidden_buf,
    c10::optional<torch::Tensor> route_weights_buf,
    int64_t S,
    int64_t H,
    int64_t K,
    int64_t R,
    int64_t rank) {

    CheckNpuContiguous(hidden_sh, "hidden_sh");
    CheckNpuContiguous(dst, "dst");
    CheckNpuContiguous(dup_groups, "dup_groups");
    CheckNpuContiguous(dup_loffs, "dup_loffs");
    CheckNpuContiguous(dup_counts, "dup_counts");
    CheckNpuContiguous(hidden_buf, "hidden_buf");

    TORCH_CHECK(S > 0 && H > 0 && K > 0 && R > 0,
                "S/H/K/R must be positive");
    TORCH_CHECK(rank >= 0 && rank < R, "rank must be in [0, R)");
    TORCH_CHECK(static_cast<uint64_t>(S) <= std::numeric_limits<uint32_t>::max() &&
                static_cast<uint64_t>(H) <= std::numeric_limits<uint32_t>::max() &&
                static_cast<uint64_t>(K) <= std::numeric_limits<uint32_t>::max() &&
                static_cast<uint64_t>(R) <= std::numeric_limits<uint32_t>::max(),
                "S/H/K/R must fit in uint32 tiling fields");
    const uint64_t num_output_slots_u64 =
        static_cast<uint64_t>(S) * static_cast<uint64_t>(K);
    TORCH_CHECK(num_output_slots_u64 <= std::numeric_limits<uint32_t>::max(),
                "S*K must fit in uint32");
    TORCH_CHECK(num_output_slots_u64 <= std::numeric_limits<uint32_t>::max() / 3,
                "S*K is too large for dup_groups indexing");
    TORCH_CHECK(static_cast<uint64_t>(R) * num_output_slots_u64 < (1ull << 31),
                "MoonEP dispatch requires R*NvS to fit in positive int32 encoding");
    const int64_t NvS = static_cast<int64_t>(num_output_slots_u64);

    TORCH_CHECK(hidden_sh.dim() == 2, "hidden_sh must be [S, H]");
    TORCH_CHECK(hidden_sh.scalar_type() == torch::kBFloat16,
                "hidden_sh must be bf16");
    TORCH_CHECK(hidden_sh.size(0) == S && hidden_sh.size(1) == H,
                "hidden_sh shape mismatch");
    CheckSameDevice(dst, hidden_sh, "dst");
    CheckSameDevice(dup_groups, hidden_sh, "dup_groups");
    CheckSameDevice(dup_loffs, hidden_sh, "dup_loffs");
    CheckSameDevice(dup_counts, hidden_sh, "dup_counts");
    const bool with_weights = route_weights.has_value();
    if (with_weights) {
        CheckNpuContiguous(*route_weights, "route_weights");
        CheckSameDevice(*route_weights, hidden_sh, "route_weights");
        TORCH_CHECK(route_weights->scalar_type() == torch::kFloat32,
                    "route_weights must be fp32 when with_weights is true");
        TORCH_CHECK(route_weights->dim() == 2 &&
                    route_weights->size(0) == S && route_weights->size(1) == K,
                    "route_weights shape mismatch");
    }
    TORCH_CHECK(hidden_buf.scalar_type() == torch::kBFloat16,
                "hidden_buf must be bf16");
    // Allow oversized symmetric capacity (suite may pin max-large then reuse).
    TORCH_CHECK(hidden_buf.dim() == 2 &&
                hidden_buf.size(0) >= NvS && hidden_buf.size(1) == H,
                "hidden_buf must be bf16 [>=S*K, H]");
    CheckSameDevice(hidden_buf, hidden_sh, "hidden_buf");
    TORCH_CHECK(route_weights_buf.has_value() == with_weights,
                "route_weights_buf must be provided exactly when route_weights is provided");
    if (with_weights) {
        CheckNpuContiguous(*route_weights_buf, "route_weights_buf");
        CheckSameDevice(*route_weights_buf, hidden_sh, "route_weights_buf");
        TORCH_CHECK(route_weights_buf->scalar_type() == torch::kFloat32 &&
                    route_weights_buf->dim() == 1 &&
                    route_weights_buf->size(0) >= NvS,
                    "route_weights_buf must be fp32 [>=S*K]");
    }
    TORCH_CHECK(dst.scalar_type() == torch::kInt32,
                "dst must be int32");
    TORCH_CHECK(dst.dim() == 1 && dst.size(0) == NvS,
                "dst must be int32 [S*K]");
    TORCH_CHECK(dup_groups.scalar_type() == torch::kInt32 &&
                dup_groups.dim() == 2 &&
                dup_groups.size(0) == NvS && dup_groups.size(1) == 3,
                "dup_groups must be int32 [NvS, 3]");
    TORCH_CHECK(dup_loffs.scalar_type() == torch::kInt32 &&
                dup_loffs.dim() == 1 && dup_loffs.size(0) == NvS,
                "dup_loffs must be int32 [NvS]");
    TORCH_CHECK(dup_counts.scalar_type() == torch::kInt32 &&
                dup_counts.dim() == 1 && dup_counts.size(0) == 2,
                "dup_counts must be int32 [2]");
    TORCH_CHECK(H % 8 == 0, "H must be multiple of 8");
    TORCH_CHECK(static_cast<uint64_t>(H) * 2 <= 128ull * 1024ull,
                "H row bytes exceed MoonEP dispatch UB row capacity");
    const torch::Tensor& route_weights_arg =
        with_weights ? *route_weights : hidden_sh;
    const torch::Tensor& route_weights_buf_arg =
        with_weights ? *route_weights_buf : hidden_buf;

    MoonEPDispatchTilingData tiling{};
    tiling.S = static_cast<uint32_t>(S);
    tiling.H = static_cast<uint32_t>(H);
    tiling.K = static_cast<uint32_t>(K);
    tiling.R = static_cast<uint32_t>(R);
    tiling.rank = static_cast<uint32_t>(rank);
    tiling.with_weights = with_weights ? 1u : 0u;
    tiling.reserved = 0;

    return {route_weights_arg, route_weights_buf_arg, tiling};
}

uint64_t NextDispatchMagic() {
    // Maintain magic counter, shared by the normal and profile pipelines so
    // interleaved dispatches keep disjoint token completion magics.
    static uint64_t magic = 2048;
    ++magic;
    return magic;
}

}  // namespace

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
    bool use_udma) {

    auto args = CheckDispatchArgs(
        hidden_sh, route_weights, dst, dup_groups, dup_loffs, dup_counts,
        hidden_buf, route_weights_buf, S, H, K, R, rank);

    if (use_udma) {
        LaunchMoonEPDispatchUdmaKernel(
            hidden_sh, args.route_weights, dst,
            dup_groups, dup_loffs, dup_counts,
            hidden_buf, args.route_weights_buf, sync_buffer,
            args.tiling, NextDispatchMagic());
    } else {
        LaunchMoonEPDispatchKernel(
            hidden_sh, args.route_weights, dst,
            dup_groups, dup_loffs, dup_counts,
            hidden_buf, args.route_weights_buf,
            args.tiling);
    }
}

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
    bool use_udma) {

    auto args = CheckDispatchArgs(
        hidden_sh, route_weights, dst, dup_groups, dup_loffs, dup_counts,
        hidden_buf, route_weights_buf, S, H, K, R, rank);
    CheckNpuContiguous(profile_cycles, "profile_cycles");
    CheckSameDevice(profile_cycles, hidden_sh, "profile_cycles");

    if (use_udma) {
        TORCH_CHECK(profile_cycles.scalar_type() == torch::kInt64 &&
                    profile_cycles.dim() == 2 &&
                    profile_cycles.size(0) == kMoonEPAivNum &&
                    profile_cycles.size(1) == kMoonEPDispatchUdmaProfileFields,
                    "profile_cycles must be int64 [32, ",
                    kMoonEPDispatchUdmaProfileFields, "]");

        LaunchMoonEPDispatchUdmaProfileKernel(
            hidden_sh, args.route_weights, dst,
            dup_groups, dup_loffs, dup_counts,
            hidden_buf, args.route_weights_buf, sync_buffer, profile_cycles,
            args.tiling, NextDispatchMagic());
    }
}

}  // namespace ascend_moonep::ops
