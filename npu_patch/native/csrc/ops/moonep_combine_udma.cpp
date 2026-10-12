#include "csrc/ops/moonep_combine_udma.h"

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
                name, " must be on the same NPU device as hidden_nvsh");
}

uint64_t NextCombineMagic() {
    // Every PE executes collectives in the same order, so the process-local
    // counters stay aligned while preventing stale grant/done consumption.
    static uint64_t magic = 4096;
    return ++magic;
}

}  // namespace

namespace ascend_moonep::ops {

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
    torch::Tensor local_workspace) {

    CheckNpuContiguous(hidden_nvsh, "hidden_nvsh");
    CheckNpuContiguous(sync_buffer, "sync_buffer");
    CheckNpuContiguous(dst, "dst");
    CheckNpuContiguous(dup_groups, "dup_groups");
    CheckNpuContiguous(dup_loffs, "dup_loffs");
    CheckNpuContiguous(dup_counts, "dup_counts");

    TORCH_CHECK(S > 0 && H > 0 && K > 0 && R > 0,
                "S/H/K/R must be positive");
    TORCH_CHECK(rank >= 0 && rank < R, "rank must be in [0, R)");
    TORCH_CHECK(K <= 16, "K must be <= 16");
    constexpr int64_t kMaxGroupWidth = 16;
    TORCH_CHECK(
        R == 1 ||
            (R % 2 == 0 &&
             (R <= kMaxGroupWidth || R % kMaxGroupWidth == 0)),
        "grouped MoonEP UDMA combine requires R=1, an even R<=16, "
        "or an R divisible by 16");
    constexpr int64_t kHiddenUbBytes = 128 * 1024;
    constexpr int64_t kFullRowDbBytesPerElement = 16;
    TORCH_CHECK(
        H <= kHiddenUbBytes / kFullRowDbBytesPerElement,
        "MoonEP UDMA combine requires H<=8192 so its full-row "
        "double buffer fits in 128 KiB of UB");
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
                "MoonEP combine requires R*NvS to fit in positive int32 encoding");
    const int64_t NvS = static_cast<int64_t>(num_output_slots_u64);

    TORCH_CHECK(hidden_nvsh.dim() == 2, "hidden_nvsh must be [NvS, H]");
    TORCH_CHECK(hidden_nvsh.scalar_type() == torch::kBFloat16,
                "hidden_nvsh must be bf16");
    TORCH_CHECK(hidden_nvsh.size(0) == NvS && hidden_nvsh.size(1) == H,
                "hidden_nvsh shape mismatch");
    CheckSameDevice(dst, hidden_nvsh, "dst");
    CheckSameDevice(sync_buffer, hidden_nvsh, "sync_buffer");
    TORCH_CHECK(sync_buffer.scalar_type() == torch::kUInt8,
                "sync_buffer must be uint8");
    constexpr uint64_t kGroupSyncBytesPerRank = 16ULL * 2ULL * 512ULL;
    TORCH_CHECK(static_cast<uint64_t>(sync_buffer.numel()) >=
                    static_cast<uint64_t>(R) * kGroupSyncBytesPerRank,
                "sync_buffer is too small for grouped combine");
    CheckSameDevice(dup_groups, hidden_nvsh, "dup_groups");
    CheckSameDevice(dup_loffs, hidden_nvsh, "dup_loffs");
    CheckSameDevice(dup_counts, hidden_nvsh, "dup_counts");
    CheckNpuContiguous(output_sh, "output_sh");
    CheckSameDevice(output_sh, hidden_nvsh, "output_sh");
    TORCH_CHECK(output_sh.scalar_type() == torch::kBFloat16 &&
                output_sh.dim() == 2 &&
                output_sh.size(0) == S && output_sh.size(1) == H,
                "output_sh must be bf16 [S, H]");

    const bool with_weights = route_weights_nvs.has_value();
    TORCH_CHECK(output_sk.has_value() == with_weights,
                "output_sk must be provided exactly when route_weights_nvs is provided");
    if (with_weights) {
        CheckNpuContiguous(*route_weights_nvs, "route_weights_nvs");
        CheckSameDevice(*route_weights_nvs, hidden_nvsh, "route_weights_nvs");
        TORCH_CHECK(route_weights_nvs->scalar_type() == torch::kFloat32,
                    "route_weights_nvs must be fp32 when with_weights is true");
        TORCH_CHECK(route_weights_nvs->dim() == 1 && route_weights_nvs->size(0) == NvS,
                    "route_weights_nvs shape mismatch");
        CheckNpuContiguous(*output_sk, "output_sk");
        CheckSameDevice(*output_sk, hidden_nvsh, "output_sk");
        TORCH_CHECK(output_sk->scalar_type() == torch::kFloat32 &&
                    output_sk->dim() == 2 &&
                    output_sk->size(0) == S && output_sk->size(1) == K,
                    "output_sk must be fp32 [S, K]");
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
    CheckNpuContiguous(local_workspace, "local_workspace");
    CheckSameDevice(local_workspace, hidden_nvsh, "local_workspace");
    TORCH_CHECK(local_workspace.scalar_type() == torch::kUInt8,
                "local_workspace must be uint8");
    TORCH_CHECK(
        num_output_slots_u64 <=
            std::numeric_limits<uint64_t>::max() /
                static_cast<uint64_t>(H) / sizeof(c10::BFloat16),
        "MoonEP UDMA combine hidden workspace size overflows uint64_t");
    const uint64_t hidden_bytes =
        num_output_slots_u64 * static_cast<uint64_t>(H) * sizeof(c10::BFloat16);
    const uint64_t weight_regions = static_cast<uint64_t>(R);
    TORCH_CHECK(
        num_output_slots_u64 <=
            std::numeric_limits<uint64_t>::max() /
                weight_regions / sizeof(float),
        "MoonEP UDMA combine gathered weight workspace size overflows uint64_t");
    const uint64_t weight_bytes =
        weight_regions * num_output_slots_u64 * sizeof(float);
    TORCH_CHECK(
        hidden_bytes <=
            std::numeric_limits<uint64_t>::max() - weight_bytes,
        "MoonEP UDMA combine workspace size overflows uint64_t");
    const uint64_t bandwidth_alignment =
        ascend_moonep::kMoonEPUdmaCombineBandwidthScratchAlign;
    TORCH_CHECK(
        hidden_bytes + weight_bytes <=
            std::numeric_limits<uint64_t>::max() - (bandwidth_alignment - 1),
        "MoonEP UDMA combine workspace alignment overflows uint64_t");
    const uint64_t bandwidth_offset =
        (hidden_bytes + weight_bytes + bandwidth_alignment - 1) &
        ~(bandwidth_alignment - 1);
    TORCH_CHECK(
        bandwidth_offset <= std::numeric_limits<uint64_t>::max() -
            ascend_moonep::kMoonEPUdmaCombineBandwidthScratchBytes,
        "MoonEP UDMA combine bandwidth scratch size overflows uint64_t");
    const uint64_t required_workspace_bytes = bandwidth_offset +
        ascend_moonep::kMoonEPUdmaCombineBandwidthScratchBytes;
    const uint64_t local_workspace_bytes =
        static_cast<uint64_t>(local_workspace.numel()) *
        static_cast<uint64_t>(local_workspace.element_size());
    TORCH_CHECK(required_workspace_bytes <= local_workspace_bytes,
                "local workspace is too small: need ",
                required_workspace_bytes, " bytes, got ",
                local_workspace_bytes, " bytes");

    const torch::Tensor& route_weights_arg =
        with_weights ? *route_weights_nvs : hidden_nvsh;
    const torch::Tensor& output_sk_arg =
        with_weights ? *output_sk : output_sh;

    MoonEPUdmaCombineTilingData tiling{};
    tiling.S = static_cast<uint32_t>(S);
    tiling.H = static_cast<uint32_t>(H);
    tiling.K = static_cast<uint32_t>(K);
    tiling.R = static_cast<uint32_t>(R);
    tiling.rank = static_cast<uint32_t>(rank);
    tiling.with_weights = with_weights ? 1u : 0u;
    tiling.reserved = 0;

    auto* workspace_ptr = static_cast<uint8_t*>(local_workspace.data_ptr());
    void* bandwidth_scratch = workspace_ptr + bandwidth_offset;

    LaunchMoonEPUdmaCombineKernel(
        hidden_nvsh, route_weights_arg, sync_buffer, dst,
        dup_groups, dup_loffs, dup_counts,
        output_sh, output_sk_arg,
        bandwidth_scratch,
        tiling, local_workspace.data_ptr(), NextCombineMagic());
}

}  // namespace ascend_moonep::ops
