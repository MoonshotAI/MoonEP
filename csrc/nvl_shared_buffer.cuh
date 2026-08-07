#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <tuple>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#ifndef CUCHECK
#define CUCHECK(cmd) do {                                      \
    CUresult err = cmd;                                            \
    if (err != CUDA_SUCCESS) {                                     \
        const char *errStr;                                        \
        cuGetErrorString(err, &errStr);                            \
        fprintf(stderr, "Failed: CUDA error %s:%d '%s'\n",        \
            __FILE__, __LINE__, errStr);                           \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
} while(0)
#endif

#ifndef CUDACHECK
#define CUDACHECK(cmd) do {                                    \
    cudaError_t err = cmd;                                         \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "Failed: CUDA error %s:%d '%s'\n",        \
            __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
} while(0)
#endif

static inline bool nvl_fabric_supported() {
    int device_id;
    CUDACHECK(cudaGetDevice(&device_id));
    CUdevice cu_device;
    CUCHECK(cuDeviceGet(&cu_device, device_id));
    int supported = 0;
    CUresult err = cuDeviceGetAttribute(&supported,
        CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, cu_device);
    // Older drivers do not know the attribute at all.
    if (err != CUDA_SUCCESS || !supported) return false;

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_id;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    size_t size;
    if (cuMemGetAllocationGranularity(
            &size, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) != CUDA_SUCCESS)
        return false;

    CUmemGenericAllocationHandle handle;
    if (cuMemCreate(&handle, size, &prop, 0) != CUDA_SUCCESS) return false;
    CUCHECK(cuMemRelease(handle));
    return true;
}

static inline size_t nvl_granularity_for(int device_id,
                                         CUmemAllocationHandleType ht) {
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_id;
    prop.requestedHandleTypes = ht;

    size_t granularity;
    CUCHECK(cuMemGetAllocationGranularity(
        &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    return granularity;
}

static inline size_t nvl_granularity_max(int device_id) {
    size_t gran = nvl_granularity_for(
        device_id, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    if (nvl_fabric_supported()) {
        gran = std::max(
            gran, nvl_granularity_for(device_id, CU_MEM_HANDLE_TYPE_FABRIC));
    }
    return gran;
}

static inline CUmemAllocationHandleType nvl_handle_type(bool use_fabric) {
    return use_fabric ? CU_MEM_HANDLE_TYPE_FABRIC
                      : CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
}

static inline std::tuple<size_t, size_t, int> nvl_prepare(
    const std::vector<int64_t> &shape,
    at::ScalarType dtype,
    bool use_fabric
) {
    TORCH_CHECK(!shape.empty(), "Shape must be non-empty");

    int device_id;
    CUDACHECK(cudaGetDevice(&device_id));

    size_t size = c10::elementSize(dtype);
    for (auto dim : shape) {
        TORCH_CHECK(dim > 0, "Size dimensions must be positive");
        size *= static_cast<size_t>(dim);
    }

    size_t granularity = nvl_granularity_for(
        device_id, nvl_handle_type(use_fabric));
    size_t allocated_size = (size + granularity - 1) / granularity * granularity;

    return {size, allocated_size, device_id};
}

static inline at::Tensor make_vmm_tensor(
    void *ptr,
    size_t nbytes,
    size_t allocated_size,
    int device_id,
    const std::vector<int64_t> &shape,
    at::ScalarType dtype
) {
    auto deleter = [device_id, ptr, allocated_size](void *p) {
        if (!p) return;
        c10::cuda::CUDAGuard guard(device_id);
        cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream().stream());
        cuMemUnmap(reinterpret_cast<CUdeviceptr>(ptr), allocated_size);
        cuMemAddressFree(reinterpret_cast<CUdeviceptr>(ptr), allocated_size);
    };

    auto storage = c10::Storage(
        c10::Storage::use_byte_size_t(),
        static_cast<int64_t>(nbytes),
        at::InefficientStdFunctionContext::makeDataPtr(
            ptr, std::move(deleter), c10::Device(c10::kCUDA, device_id)),
        /*allocator=*/nullptr,
        /*resizable=*/false);

    auto impl = c10::make_intrusive<at::TensorImpl>(
        std::move(storage),
        c10::DispatchKeySet(c10::DispatchKey::CUDA),
        c10::scalarTypeToTypeMeta(dtype));
    impl->set_sizes_contiguous(shape);
    return at::Tensor(std::move(impl));
}

static constexpr int64_t kFabricHandleBytes =
    static_cast<int64_t>(sizeof(CUmemFabricHandle::data));

// Export an allocation (or multicast object) as a CPU tensor matching
// `use_fabric`: uint8[64] holding a CUmemFabricHandle, or int64[] (a scalar
// tensor) holding a POSIX fd.
static inline at::Tensor nvl_export_shareable(
    CUmemGenericAllocationHandle handle, bool use_fabric
) {
    if (use_fabric) {
        CUmemFabricHandle fabric = {};
        CUCHECK(cuMemExportToShareableHandle(
            &fabric, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
        auto out = at::empty({kFabricHandleBytes},
                             at::TensorOptions().dtype(at::kByte));
        std::memcpy(out.data_ptr(), fabric.data, sizeof(fabric.data));
        return out;
    }
    int fd = -1;
    CUCHECK(cuMemExportToShareableHandle(
        &fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
    TORCH_CHECK(fd >= 0, "cuMemExportToShareableHandle returned invalid fd");
    return at::scalar_tensor(fd, at::TensorOptions().dtype(at::kLong));
}

static inline void nvl_check_handles(
    const at::Tensor &handles, int64_t world_size, bool use_fabric
) {
    TORCH_CHECK(handles.device().is_cpu() && handles.is_contiguous(),
        "shareable handles must be a contiguous CPU tensor, got device=",
        handles.device(), " contiguous=", handles.is_contiguous());
    if (use_fabric) {
        TORCH_CHECK(handles.scalar_type() == at::kByte && handles.dim() == 2
                    && handles.size(0) == world_size
                    && handles.size(1) == kFabricHandleBytes,
            "fabric handles must be uint8[", world_size, ", ",
            kFabricHandleBytes, "], got ", handles.scalar_type(), handles.sizes());
    } else {
        TORCH_CHECK(handles.scalar_type() == at::kLong
                    && handles.numel() == world_size,
            "fd handles must be int64 with ", world_size, " elements, got ",
            handles.scalar_type(), handles.sizes());
    }
}

// Import row `index` of a handle tensor already validated by nvl_check_handles.
static inline CUmemGenericAllocationHandle nvl_import_shareable(
    const at::Tensor &handles, int64_t index, bool use_fabric
) {
    CUmemGenericAllocationHandle handle;
    if (use_fabric) {
        CUmemFabricHandle fabric = {};
        std::memcpy(fabric.data,
                    handles.const_data_ptr<uint8_t>() + index * kFabricHandleBytes,
                    sizeof(fabric.data));
        CUCHECK(cuMemImportFromShareableHandle(
            &handle, &fabric, CU_MEM_HANDLE_TYPE_FABRIC));
    } else {
        int64_t fd = handles.const_data_ptr<int64_t>()[index];
        TORCH_CHECK(fd >= 0, "invalid fd ", fd, " at index ", index);
        CUCHECK(cuMemImportFromShareableHandle(
            &handle, reinterpret_cast<void *>(static_cast<intptr_t>(fd)),
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
    }
    return handle;
}

// Returns (keepalive VA tensor, shareable handle, owned mem handle as int64).
// The owned mem handle is not released here (kept for multicast cuMulticastBindMem);
// the caller must call nvl_release_mem_handle when done (buffers that do not need
// multicast release it immediately, matching the old behavior; buffers that need
// multicast release it after bind).
// The shareable handle is what peers import: an int64 scalar tensor holding an
// fd the caller must close once all peers have imported it, or a uint8[64]
// fabric handle that needs no cleanup.
inline std::tuple<at::Tensor, at::Tensor, int64_t> nvl_dist_alloc(
    const std::vector<int64_t> &chunk_shape,
    at::ScalarType dtype,
    bool use_fabric
) {
    auto [nbytes, allocated_size, device_id] =
        nvl_prepare(chunk_shape, dtype, use_fabric);

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_id;
    prop.requestedHandleTypes = nvl_handle_type(use_fabric);

    CUmemGenericAllocationHandle mem_handle;
    CUCHECK(cuMemCreate(&mem_handle, allocated_size, &prop, 0));

    CUdeviceptr dptr;
    CUCHECK(cuMemAddressReserve(&dptr, allocated_size, 0, 0, 0));
    CUCHECK(cuMemMap(dptr, allocated_size, 0, mem_handle, 0));

    CUmemAccessDesc access_desc = {};
    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id = device_id;
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUCHECK(cuMemSetAccess(dptr, allocated_size, &access_desc, 1));

    // Export for other processes to import. Do not release mem_handle: keep the
    // owned generic handle for multicast BindMem. The caller is responsible for
    // nvl_release_mem_handle. The physical memory is referenced by the unicast
    // map (keepalive) and (optionally) multicast; it is only truly freed after
    // all unmaps once the handle is released.
    auto shareable = nvl_export_shareable(mem_handle, use_fabric);

    auto keepalive = make_vmm_tensor(
        reinterpret_cast<void *>(dptr), nbytes, allocated_size,
        device_id, chunk_shape, dtype);

    return {std::move(keepalive), std::move(shareable),
            static_cast<int64_t>(mem_handle)};
}

// Release the owned mem handle returned by nvl_dist_alloc (the physical memory
// stays alive via the existing maps).
static inline void nvl_release_mem_handle(int64_t mem_handle_u64) {
    CUCHECK(cuMemRelease(static_cast<CUmemGenericAllocationHandle>(mem_handle_u64)));
}

/**
 * Map all chunk handles into a contiguous VA region.
 * Key difference from reference: ALL chunks get RW access (not just local_rank),
 * because dispatch needs to write to remote ranks' regions.
 *
 * `shareables` are the handles exported by each rank via nvl_dist_alloc and
 * routed to this process by the caller: POSIX fds (already valid here, the
 * caller may close them once imported) or 64-byte fabric blobs.
 */
inline at::Tensor nvl_dist_map(
    const std::vector<int64_t> &chunk_shape,
    at::ScalarType dtype,
    const at::Tensor &shareables,
    int64_t local_rank,
    int64_t world_size,
    bool use_fabric
) {
    nvl_check_handles(shareables, world_size, use_fabric);

    auto [chunk_nbytes, chunk_allocated_size, device_id] =
        nvl_prepare(chunk_shape, dtype, use_fabric);

    TORCH_CHECK(chunk_allocated_size == chunk_nbytes,
        "Chunk byte size (", chunk_nbytes, ") must be aligned to VMM granularity (",
        chunk_allocated_size, "). Adjust chunk dimensions so that the total bytes "
        "are a multiple of the allocation granularity.");

    size_t total_allocated_size = chunk_allocated_size * world_size;
    size_t total_nbytes = chunk_nbytes * world_size;

    CUdeviceptr dptr;
    CUCHECK(cuMemAddressReserve(&dptr, total_allocated_size, 0, 0, 0));

    for (int64_t i = 0; i < world_size; i++) {
        CUmemGenericAllocationHandle mem_handle =
            nvl_import_shareable(shareables, i, use_fabric);

        CUdeviceptr chunk_va = dptr + i * chunk_allocated_size;
        CUCHECK(cuMemMap(chunk_va, chunk_allocated_size, 0, mem_handle, 0));

        // All chunks get RW access for dispatch writes
        CUmemAccessDesc access = {};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = device_id;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CUCHECK(cuMemSetAccess(chunk_va, chunk_allocated_size, &access, 1));

        CUCHECK(cuMemRelease(mem_handle));
    }

    std::vector<int64_t> full_shape = chunk_shape;
    full_shape[0] *= world_size;

    int ws = world_size;
    size_t cas = chunk_allocated_size;
    size_t tas = total_allocated_size;
    auto deleter = [device_id, dptr, cas, ws, tas](void *p) {
        if (!p) return;
        c10::cuda::CUDAGuard guard(device_id);
        cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream().stream());
        for (int i = 0; i < ws; i++) {
            cuMemUnmap(dptr + i * cas, cas);
        }
        cuMemAddressFree(dptr, tas);
    };

    auto storage = c10::Storage(
        c10::Storage::use_byte_size_t(),
        static_cast<int64_t>(total_nbytes),
        at::InefficientStdFunctionContext::makeDataPtr(
            reinterpret_cast<void *>(dptr), std::move(deleter),
            c10::Device(c10::kCUDA, device_id)),
        /*allocator=*/nullptr,
        /*resizable=*/false);

    auto impl = c10::make_intrusive<at::TensorImpl>(
        std::move(storage),
        c10::DispatchKeySet(c10::DispatchKey::CUDA),
        c10::scalarTypeToTypeMeta(dtype));
    impl->set_sizes_contiguous(full_shape);
    return at::Tensor(std::move(impl));
}

// ============================================================
// Multicast (NVSwitch SHARP) — register once, hold persistently
// ============================================================
//
// Adds a multicast mapping on top of an already-allocated NVLink chunk (each
// rank's own cuMemCreate physical memory): bind every rank's chunk physical
// memory to the same multicast object, then map a multicast VA. A single
// multimem.st write to the multicast VA is fanned out by NVSwitch hardware to
// the corresponding offset of every rank's local physical memory.
//
// Uses no extra device memory — it only adds one more VA mapping over the
// existing chunk physical memory. Lifetime: the caller (MoonEP Buffer) holds
// the returned mc VA tensor, same lifetime as meta_buf; the mc VA tensor's
// deleter releases the multicast handle after unmap/free of the VA.
//
// Registration must follow the Python-side coordination order (separated by
// dist.barrier, otherwise bind fails with CUDA_ERROR_ILLEGAL_STATE or hangs):
//   root: nvl_multicast_create -> send the exported POSIX fd to other ranks
//   non-root: nvl_multicast_import
//   all ranks: nvl_multicast_add_device
//   --- dist.barrier ---
//   all ranks: nvl_multicast_bind_map
//   --- dist.barrier ---

// Whether the device supports multicast.
static inline bool nvl_multicast_supported() {
    int device_id;
    CUDACHECK(cudaGetDevice(&device_id));
    CUdevice cu_device;
    CUCHECK(cuDeviceGet(&cu_device, device_id));
    int supported = 0;
    CUCHECK(cuDeviceGetAttribute(&supported,
        CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, cu_device));
    return supported != 0;
}

static inline size_t nvl_multicast_granularity_for(
    int num_devices, CUmemAllocationHandleType ht
) {
    CUmulticastObjectProp prop = {};
    prop.numDevices = static_cast<unsigned int>(num_devices);
    prop.size = 0;
    prop.handleTypes = ht;
    prop.flags = 0;
    size_t gran = 0;
    CUCHECK(cuMulticastGetGranularity(
        &gran, &prop, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    return gran;
}

// Recommended multicast alignment granularity (bytes). Both the addr and size
// of a bind must be aligned to it. Like nvl_granularity_max, this is the max
// over every handle type that may be used, so a buffer padded to it stays
// valid whichever type the process group picks.
static inline size_t nvl_multicast_granularity(int num_devices) {
    size_t gran = nvl_multicast_granularity_for(
        num_devices, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    if (nvl_fabric_supported()) {
        gran = std::max(gran, nvl_multicast_granularity_for(
            num_devices, CU_MEM_HANDLE_TYPE_FABRIC));
    }
    return gran;
}

static inline int64_t get_multicast_granularity(int64_t num_devices) {
    return static_cast<int64_t>(
        nvl_multicast_granularity(static_cast<int>(num_devices)));
}

// root only: create the multicast object and export it for the other ranks.
// Returns (mc_handle as uint64, shareable handle).
// mc_handle is a handle value valid within this process; Python holds it and
// passes it back unchanged to bind_map. The caller passes the shareable handle
// to the other ranks (closing the fd, if it is one, after all have imported).
inline std::tuple<int64_t, at::Tensor> nvl_multicast_create(
    int64_t size_bytes, int64_t num_devices, bool use_fabric
) {
    TORCH_CHECK(nvl_multicast_supported(),
        "Multicast not supported on this device");
    const size_t gran = nvl_multicast_granularity(static_cast<int>(num_devices));
    const size_t aligned =
        (static_cast<size_t>(size_bytes) + gran - 1) / gran * gran;

    CUmulticastObjectProp prop = {};
    prop.numDevices = static_cast<unsigned int>(num_devices);
    prop.size = aligned;
    prop.handleTypes = nvl_handle_type(use_fabric);
    prop.flags = 0;

    CUmemGenericAllocationHandle mc_handle;
    CUCHECK(cuMulticastCreate(&mc_handle, &prop));

    auto shareable = nvl_export_shareable(mc_handle, use_fabric);
    return {static_cast<int64_t>(mc_handle), std::move(shareable)};
}

// non-root: import the multicast object from the handle sent by root. If it is
// an fd, the caller may close it once imported.
inline int64_t nvl_multicast_import(
    const at::Tensor &shareable, bool use_fabric
) {
    auto one = use_fabric ? shareable.view({1, kFabricHandleBytes})
                          : shareable.view({1});
    nvl_check_handles(one, 1, use_fabric);
    return static_cast<int64_t>(nvl_import_shareable(one, 0, use_fabric));
}

// all ranks: add this rank's device to the multicast group. Must happen before
// bind, and bind may only happen after all ranks have added (dist.barrier).
inline void nvl_multicast_add_device(int64_t mc_handle_u64) {
    int device_id;
    CUDACHECK(cudaGetDevice(&device_id));
    CUdevice cu_device;
    CUCHECK(cuDeviceGet(&cu_device, device_id));
    CUCHECK(cuMulticastAddDevice(
        static_cast<CUmemGenericAllocationHandle>(mc_handle_u64), cu_device));
}

// all ranks: bind this rank's local chunk owned physical memory (the mem
// handle kept by nvl_dist_alloc) to the multicast object, then reserve a VA
// for the multicast handle + map + setAccess. Returns an int32 keepalive
// tensor holding the multicast VA (.data_ptr() is the multimem address). Uses
// BindMem (owned handle) instead of BindAddr: BindAddr returns invalid
// argument for IPC-imported VAs on this platform.
inline at::Tensor nvl_multicast_bind_map(
    int64_t mc_handle_u64, int64_t owned_mem_handle_u64,
    int64_t size_bytes, int64_t num_devices
) {
    int device_id;
    CUDACHECK(cudaGetDevice(&device_id));
    CUmemGenericAllocationHandle mc_handle =
        static_cast<CUmemGenericAllocationHandle>(mc_handle_u64);
    CUmemGenericAllocationHandle mem_handle =
        static_cast<CUmemGenericAllocationHandle>(owned_mem_handle_u64);

    const size_t gran = nvl_multicast_granularity(static_cast<int>(num_devices));
    TORCH_CHECK(static_cast<size_t>(size_bytes) % gran == 0,
        "multicast bind size (", size_bytes, ") must be a multiple of multicast "
        "granularity (", gran, "); pad meta_buf chunk accordingly");
    const size_t size = static_cast<size_t>(size_bytes);

    // Bind this rank's owned physical allocation to offset 0 of the multicast object.
    CUCHECK(cuMulticastBindMem(
        mc_handle, /*mcOffset=*/0, mem_handle, /*memOffset=*/0, size, 0));

    // Reserve a separate VA for the multicast handle and map it. Pass 0 for
    // alignment (the driver aligns to granularity by default, same as
    // nvl_dist_alloc/nvl_dist_map; a non-zero alignment is rejected by
    // cuMemAddressReserve as invalid argument).
    CUdeviceptr mc_va;
    CUCHECK(cuMemAddressReserve(&mc_va, size, 0, 0, 0));
    CUCHECK(cuMemMap(mc_va, size, 0, mc_handle, 0));

    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device_id;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUCHECK(cuMemSetAccess(mc_va, size, &access, 1));

    // keepalive: the deleter unmaps + frees the VA, then releases mc_handle.
    // data_ptr() = multimem address.
    auto deleter = [device_id, mc_va, size, mc_handle](void* p) {
        if (!p) return;
        c10::cuda::CUDAGuard guard(device_id);
        cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream().stream());
        cuMemUnmap(mc_va, size);
        cuMemAddressFree(mc_va, size);
        cuMemRelease(mc_handle);
    };

    std::vector<int64_t> shape = {static_cast<int64_t>(size / sizeof(int32_t))};
    auto storage = c10::Storage(
        c10::Storage::use_byte_size_t(),
        static_cast<int64_t>(size),
        at::InefficientStdFunctionContext::makeDataPtr(
            reinterpret_cast<void*>(mc_va), std::move(deleter),
            c10::Device(c10::kCUDA, device_id)),
        /*allocator=*/nullptr,
        /*resizable=*/false);

    auto impl = c10::make_intrusive<at::TensorImpl>(
        std::move(storage),
        c10::DispatchKeySet(c10::DispatchKey::CUDA),
        c10::scalarTypeToTypeMeta(at::kInt));
    impl->set_sizes_contiguous(shape);
    return at::Tensor(std::move(impl));
}
