#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <climits>
#include <cstdint>
#include "nvl_shared_buffer.cuh"

namespace {

__global__ void materialize_source_rows_kernel(
    const uint4* __restrict__ mapped_source,
    const int32_t* __restrict__ mapped_source_weights,
    uint4* __restrict__ local_output,
    int32_t* __restrict__ local_output_weights,
    const int32_t* __restrict__ slot_to_source,
    const int32_t* __restrict__ slot_to_primary,
    const int32_t* __restrict__ slot_to_next,
    int32_t NvS,
    int32_t K,
    int32_t source_rows_padded,
    int32_t source_topk_padded,
    bool with_weights,
    int32_t hidden_vecs) {
    for (int32_t loff = static_cast<int32_t>(blockIdx.x);
         loff < NvS;
         loff += static_cast<int32_t>(gridDim.x)) {
        const int32_t encoded = slot_to_source[loff];
        if (encoded >= 0) {
            const int32_t source_rank = encoded / NvS;
            const int32_t source_topk_offset = encoded - source_rank * NvS;
            const int32_t source_token = source_topk_offset / K;
            if (with_weights && threadIdx.x == 0) {
                local_output_weights[loff] = mapped_source_weights[
                    static_cast<int64_t>(source_rank) * source_topk_padded
                    + source_topk_offset];
            }
            if (slot_to_primary[loff] != loff) {
                continue;
            }
            const int64_t source_row =
                static_cast<int64_t>(source_rank) * source_rows_padded
                + source_token;
            const uint4* source =
                mapped_source + source_row * hidden_vecs;
            for (int32_t vec = static_cast<int32_t>(threadIdx.x);
                 vec < hidden_vecs;
                 vec += static_cast<int32_t>(blockDim.x)) {
                const uint4 value = source[vec];
                int32_t target = loff;
                for (int32_t fanout = 0;
                     target >= 0 && fanout < K;
                     ++fanout) {
                    local_output[
                        static_cast<int64_t>(target) * hidden_vecs + vec
                    ] = value;
                    target = slot_to_next[target];
                }
            }
        }
    }
}

__global__ void zero_source_padding_kernel(
    uint4* __restrict__ local_output,
    int32_t* __restrict__ local_output_weights,
    const int32_t* __restrict__ zero_fill_ranges,
    int32_t zero_groups,
    bool with_weights,
    int32_t hidden_vecs) {
    for (int32_t group = static_cast<int32_t>(blockIdx.x);
         group < zero_groups;
         group += static_cast<int32_t>(gridDim.x)) {
        const int32_t pad_start = zero_fill_ranges[group * 2];
        const int32_t pad_count = zero_fill_ranges[group * 2 + 1];
        for (int32_t row = 0; row < pad_count; ++row) {
            const int32_t loff = pad_start + row;
            uint4* output =
                local_output + static_cast<int64_t>(loff) * hidden_vecs;
            for (int32_t vec = static_cast<int32_t>(threadIdx.x);
                 vec < hidden_vecs;
                 vec += static_cast<int32_t>(blockDim.x)) {
                output[vec] = make_uint4(0, 0, 0, 0);
            }
            if (with_weights && threadIdx.x == 0) {
                local_output_weights[loff] = 0;
            }
        }
    }
}

void materialize_source_rows(
    torch::Tensor mapped_source,
    torch::Tensor mapped_source_weights,
    torch::Tensor local_output,
    torch::Tensor local_output_weights,
    torch::Tensor slot_to_source,
    torch::Tensor slot_to_primary,
    torch::Tensor slot_to_next,
    torch::Tensor zero_fill_ranges,
    int64_t NvS,
    int64_t K,
    int64_t source_rows_padded,
    int64_t source_topk_padded,
    bool with_weights,
    int64_t num_blocks,
    int64_t num_threads) {
    TORCH_CHECK(mapped_source.is_cuda(), "mapped_source must be CUDA");
    TORCH_CHECK(
        mapped_source_weights.is_cuda(),
        "mapped_source_weights must be CUDA");
    TORCH_CHECK(local_output.is_cuda(), "local_output must be CUDA");
    TORCH_CHECK(
        local_output_weights.is_cuda(),
        "local_output_weights must be CUDA");
    TORCH_CHECK(slot_to_source.is_cuda(), "slot_to_source must be CUDA");
    TORCH_CHECK(
        slot_to_primary.is_cuda(),
        "slot_to_primary must be CUDA");
    TORCH_CHECK(slot_to_next.is_cuda(), "slot_to_next must be CUDA");
    TORCH_CHECK(zero_fill_ranges.is_cuda(), "zero_fill_ranges must be CUDA");
    TORCH_CHECK(
        mapped_source.scalar_type() == torch::kBFloat16,
        "mapped_source must be bf16");
    TORCH_CHECK(
        local_output.scalar_type() == torch::kBFloat16,
        "local_output must be bf16");
    TORCH_CHECK(
        mapped_source_weights.scalar_type() == torch::kFloat32,
        "mapped_source_weights must be fp32");
    TORCH_CHECK(
        local_output_weights.scalar_type() == torch::kInt32,
        "local_output_weights must be the int32 view of fp32");
    TORCH_CHECK(
        slot_to_source.scalar_type() == torch::kInt32,
        "slot_to_source must be int32");
    TORCH_CHECK(
        slot_to_primary.scalar_type() == torch::kInt32,
        "slot_to_primary must be int32");
    TORCH_CHECK(
        slot_to_next.scalar_type() == torch::kInt32,
        "slot_to_next must be int32");
    TORCH_CHECK(
        zero_fill_ranges.scalar_type() == torch::kInt32,
        "zero_fill_ranges must be int32");
    TORCH_CHECK(mapped_source.is_contiguous(), "mapped_source must be contiguous");
    TORCH_CHECK(
        mapped_source_weights.is_contiguous(),
        "mapped_source_weights must be contiguous");
    TORCH_CHECK(local_output.is_contiguous(), "local_output must be contiguous");
    TORCH_CHECK(
        local_output_weights.is_contiguous(),
        "local_output_weights must be contiguous");
    TORCH_CHECK(slot_to_source.is_contiguous(), "slot_to_source must be contiguous");
    TORCH_CHECK(
        slot_to_primary.is_contiguous(),
        "slot_to_primary must be contiguous");
    TORCH_CHECK(slot_to_next.is_contiguous(), "slot_to_next must be contiguous");
    TORCH_CHECK(
        zero_fill_ranges.is_contiguous(),
        "zero_fill_ranges must be contiguous");
    TORCH_CHECK(mapped_source.dim() == 2, "mapped_source must be 2-D");
    TORCH_CHECK(local_output.dim() == 2, "local_output must be 2-D");
    TORCH_CHECK(slot_to_source.dim() == 1, "slot_to_source must be 1-D");
    TORCH_CHECK(
        mapped_source_weights.dim() == 1,
        "mapped_source_weights must be 1-D");
    TORCH_CHECK(
        local_output_weights.dim() == 1,
        "local_output_weights must be 1-D");
    TORCH_CHECK(
        slot_to_primary.dim() == 1,
        "slot_to_primary must be 1-D");
    TORCH_CHECK(slot_to_next.dim() == 1, "slot_to_next must be 1-D");
    TORCH_CHECK(
        zero_fill_ranges.dim() == 2 && zero_fill_ranges.size(1) == 2,
        "zero_fill_ranges must be [groups, 2]");
    TORCH_CHECK(NvS > 0 && NvS <= INT32_MAX, "NvS must fit positive int32");
    TORCH_CHECK(K > 0 && K <= INT32_MAX, "K must fit positive int32");
    TORCH_CHECK(
        source_rows_padded > 0 && source_rows_padded <= INT32_MAX,
        "source_rows_padded must fit positive int32");
    TORCH_CHECK(
        source_topk_padded > 0 && source_topk_padded <= INT32_MAX,
        "source_topk_padded must fit positive int32");
    TORCH_CHECK(
        num_blocks >= 0 && num_blocks <= INT32_MAX,
        "num_blocks must be zero or fit positive int32");
    TORCH_CHECK(
        num_threads == 64 || num_threads == 128 || num_threads == 256
            || num_threads == 512,
        "num_threads must be one of 64, 128, 256, or 512");
    TORCH_CHECK(
        slot_to_source.numel() == NvS,
        "slot_to_source length must equal NvS");
    TORCH_CHECK(
        slot_to_primary.numel() == NvS,
        "slot_to_primary length must equal NvS");
    TORCH_CHECK(
        slot_to_next.numel() == NvS,
        "slot_to_next length must equal NvS");
    TORCH_CHECK(
        local_output.size(0) == NvS,
        "local_output first dimension must equal NvS");
    TORCH_CHECK(
        local_output_weights.numel() == NvS,
        "local_output_weights length must equal NvS");
    TORCH_CHECK(
        mapped_source.size(1) == local_output.size(1),
        "mapped_source and local_output hidden sizes must match");
    TORCH_CHECK(
        mapped_source.size(0) % source_rows_padded == 0,
        "mapped_source rows must be a multiple of source_rows_padded");
    const int64_t source_ranks =
        mapped_source.size(0) / source_rows_padded;
    TORCH_CHECK(source_ranks > 0, "mapped_source must contain a source rank");
    TORCH_CHECK(
        mapped_source_weights.numel()
            == source_ranks * source_topk_padded,
        "mapped_source_weights length must match the mapped source ranks");
    TORCH_CHECK(
        source_topk_padded >= K,
        "source_topk_padded must be at least K");
    TORCH_CHECK(
        mapped_source.size(1) % 8 == 0,
        "hidden size must be divisible by 8 for uint4 vectorization");
    TORCH_CHECK(
        mapped_source.size(1) / 8 <= INT32_MAX,
        "vectorized hidden size must fit int32");
    TORCH_CHECK(
        zero_fill_ranges.size(0) <= INT32_MAX,
        "zero-fill group count must fit int32");
    TORCH_CHECK(
        reinterpret_cast<uintptr_t>(mapped_source.data_ptr()) % 16 == 0
            && reinterpret_cast<uintptr_t>(local_output.data_ptr()) % 16 == 0,
        "mapped_source and local_output must be 16-byte aligned");
    TORCH_CHECK(
        mapped_source.device() == local_output.device()
            && mapped_source.device() == mapped_source_weights.device()
            && mapped_source.device() == local_output_weights.device()
            && mapped_source.device() == slot_to_source.device()
            && mapped_source.device() == slot_to_primary.device()
            && mapped_source.device() == slot_to_next.device()
            && mapped_source.device() == zero_fill_ranges.device(),
        "all tensors must use the current CUDA device mapping");

    const int32_t hidden_vecs =
        static_cast<int32_t>(mapped_source.size(1) / 8);
    const int threads = static_cast<int>(num_threads);
    const int blocks = static_cast<int>(
        num_blocks == 0
            ? std::min<int64_t>(NvS, 49152)
            : std::min<int64_t>(NvS, num_blocks));
    const auto stream = at::cuda::getCurrentCUDAStream();
    materialize_source_rows_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint4*>(mapped_source.data_ptr()),
        reinterpret_cast<const int32_t*>(
            mapped_source_weights.data_ptr()),
        reinterpret_cast<uint4*>(local_output.data_ptr()),
        local_output_weights.data_ptr<int32_t>(),
        slot_to_source.data_ptr<int32_t>(),
        slot_to_primary.data_ptr<int32_t>(),
        slot_to_next.data_ptr<int32_t>(),
        static_cast<int32_t>(NvS),
        static_cast<int32_t>(K),
        static_cast<int32_t>(source_rows_padded),
        static_cast<int32_t>(source_topk_padded),
        with_weights,
        hidden_vecs);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int64_t zero_groups = zero_fill_ranges.size(0);
    const int zero_blocks =
        static_cast<int>(std::min<int64_t>(zero_groups, 4096));
    zero_source_padding_kernel<<<zero_blocks, threads, 0, stream>>>(
        reinterpret_cast<uint4*>(local_output.data_ptr()),
        local_output_weights.data_ptr<int32_t>(),
        zero_fill_ranges.data_ptr<int32_t>(),
        static_cast<int32_t>(zero_groups),
        with_weights,
        hidden_vecs);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

/**
 * Return the VMM allocation granularity in bytes for the current device.
 */
int64_t get_vmm_granularity() {
    int device_id;
    CUDACHECK(cudaGetDevice(&device_id));

    CUdevice cu_device;
    CUCHECK(cuDeviceGet(&cu_device, device_id));

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_id;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

    size_t granularity;
    CUCHECK(cuMemGetAllocationGranularity(
        &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    return static_cast<int64_t>(granularity);
}

PYBIND11_MODULE(_C, m) {
    m.def("nvl_dist_alloc", &nvl_dist_alloc,
          pybind11::arg("shape"), pybind11::arg("dtype"),
          "Allocate a local chunk and export it as a POSIX fd for IPC sharing.");
    m.def("nvl_dist_map", &nvl_dist_map,
          pybind11::arg("chunk_shape"), pybind11::arg("dtype"),
          pybind11::arg("fds"), pybind11::arg("local_rank"),
          pybind11::arg("world_size"),
          "Map all chunk fds into a contiguous VA region (all RW).");
    m.def("get_vmm_granularity", &get_vmm_granularity,
          "Return VMM allocation granularity in bytes.");
    m.def("get_multicast_granularity", &get_multicast_granularity,
          pybind11::arg("num_devices"),
          "Return multicast recommended allocation granularity in bytes.");
    m.def("nvl_multicast_supported", &nvl_multicast_supported,
          "Whether the current device supports multicast (NVSwitch SHARP).");
    m.def("nvl_multicast_create", &nvl_multicast_create,
          pybind11::arg("size_bytes"), pybind11::arg("num_devices"),
          "Root-only: create a multicast object and export it as a POSIX fd.");
    m.def("nvl_multicast_import", &nvl_multicast_import,
          pybind11::arg("fd"),
          "Non-root: import a multicast object from the root's POSIX fd.");
    m.def("nvl_multicast_add_device", &nvl_multicast_add_device,
          pybind11::arg("mc_handle"),
          "Add the current device to a multicast object (before bind).");
    m.def("nvl_multicast_bind_map", &nvl_multicast_bind_map,
          pybind11::arg("mc_handle"), pybind11::arg("owned_mem_handle"),
          pybind11::arg("size_bytes"), pybind11::arg("num_devices"),
          "Bind the local chunk's owned physical memory (nvl_dist_alloc handle) "
          "to the multicast object and map a multicast VA; returns a keepalive "
          "int32 tensor whose data_ptr is the multimem address.");
    m.def("nvl_release_mem_handle", &nvl_release_mem_handle,
          pybind11::arg("mem_handle"),
          "Release an owned mem handle returned by nvl_dist_alloc.");
    m.def(
        "materialize_source_rows",
        &materialize_source_rows,
        pybind11::arg("mapped_source"),
        pybind11::arg("mapped_source_weights"),
        pybind11::arg("local_output"),
        pybind11::arg("local_output_weights"),
        pybind11::arg("slot_to_source"),
        pybind11::arg("slot_to_primary"),
        pybind11::arg("slot_to_next"),
        pybind11::arg("zero_fill_ranges"),
        pybind11::arg("NvS"),
        pybind11::arg("K"),
        pybind11::arg("source_rows_padded"),
        pybind11::arg("source_topk_padded"),
        pybind11::arg("with_weights"),
        pybind11::arg("num_blocks") = 0,
        pybind11::arg("num_threads") = 64,
        "Materialize a peer-indexed owner-sharded source view into the local "
        "contiguous expert layout.");
}
