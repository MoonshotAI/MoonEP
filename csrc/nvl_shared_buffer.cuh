#pragma once

#include <cstdint>
#include <cstring>
#include <tuple>
#include <vector>

#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/ScopeExit.h>

#ifndef CUCHECK
#define CUCHECK(cmd) do {                                                   \
    const CUresult err = (cmd);                                             \
    if (err != CUDA_SUCCESS) {                                              \
        const char *err_name = nullptr;                                     \
        const char *err_string = nullptr;                                   \
        cuGetErrorName(err, &err_name);                                     \
        cuGetErrorString(err, &err_string);                                 \
        TORCH_CHECK(false, "CUDA driver call `", #cmd, "` failed with ",   \
            err_name ? err_name : "unknown error", " (code ",              \
            static_cast<int>(err), "): ",                                  \
            err_string ? err_string : "unknown error");                    \
    }                                                                       \
} while(0)
#endif

#ifndef CUDACHECK
#define CUDACHECK(cmd) do {                                                \
    const cudaError_t err = (cmd);                                         \
    TORCH_CHECK(err == cudaSuccess, "CUDA runtime call `", #cmd,          \
        "` failed with ", cudaGetErrorName(err), " (code ",               \
        static_cast<int>(err), "): ", cudaGetErrorString(err));            \
} while(0)
#endif

static inline std::tuple<size_t, size_t, int> nvl_prepare(
    const std::vector<int64_t> &shape,
    at::ScalarType dtype
) {
    TORCH_CHECK(!shape.empty(), "Shape must be non-empty");

    int device_id;
    CUDACHECK(cudaGetDevice(&device_id));

    size_t size = c10::elementSize(dtype);
    for (auto dim : shape) {
        TORCH_CHECK(dim > 0, "Size dimensions must be positive");
        size *= static_cast<size_t>(dim);
    }

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_id;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

    size_t granularity;
    CUCHECK(cuMemGetAllocationGranularity(
        &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    size_t allocated_size = (size + granularity - 1) / granularity * granularity;

    return {size, allocated_size, device_id};
}

static inline auto make_vmm_deleter(
    int device_id,
    CUdeviceptr dptr,
    size_t allocated_size
) {
    return [device_id, dptr, allocated_size](void *p) {
        if (!p) return;
        c10::cuda::CUDAGuard guard(device_id);
        cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream().stream());
        cuMemUnmap(dptr, allocated_size);
        cuMemAddressFree(dptr, allocated_size);
    };
}

// Returns (keepalive VA tensor, exported POSIX fd, owned mem handle as int64).
// The owned mem handle is not released here (kept for multicast cuMulticastBindMem);
// the caller must call nvl_release_mem_handle when done (buffers that do not need
// multicast release it immediately, matching the old behavior; buffers that need
// multicast release it after bind).
// The fd is used for cross-process IPC sharing (POSIX file descriptor); the caller
// must close it after all peers have imported it.
inline std::tuple<at::Tensor, int64_t, int64_t> nvl_dist_alloc(
    const std::vector<int64_t> &chunk_shape,
    at::ScalarType dtype
) {
    auto [nbytes, allocated_size, device_id] = nvl_prepare(chunk_shape, dtype);

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_id;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

    CUmemGenericAllocationHandle mem_handle = {};
    bool mem_handle_owned = false;
    CUdeviceptr dptr = 0;
    bool va_reserved = false;
    bool va_mapped = false;
    int fd = -1;
    auto cleanup = c10::make_scope_exit([&]() {
        if (fd >= 0) {
            close(fd);
        }
        if (va_mapped) {
            cuMemUnmap(dptr, allocated_size);
        }
        if (va_reserved) {
            cuMemAddressFree(dptr, allocated_size);
        }
        if (mem_handle_owned) {
            cuMemRelease(mem_handle);
        }
    });

    CUCHECK(cuMemCreate(&mem_handle, allocated_size, &prop, 0));
    mem_handle_owned = true;

    CUCHECK(cuMemAddressReserve(&dptr, allocated_size, 0, 0, 0));
    va_reserved = true;
    CUCHECK(cuMemMap(dptr, allocated_size, 0, mem_handle, 0));
    va_mapped = true;

    CUmemAccessDesc access_desc = {};
    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id = device_id;
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUCHECK(cuMemSetAccess(dptr, allocated_size, &access_desc, 1));

    // Export the POSIX fd for other processes to import. Do not release
    // mem_handle: keep the owned generic handle for multicast BindMem. The
    // caller is responsible for nvl_release_mem_handle. The physical memory is
    // referenced by the unicast map (keepalive) and (optionally) multicast;
    // it is only truly freed after all unmaps once the handle is released.
    CUCHECK(cuMemExportToShareableHandle(
        &fd, mem_handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
    TORCH_CHECK(fd >= 0, "cuMemExportToShareableHandle returned invalid fd");

    auto data_ptr = at::InefficientStdFunctionContext::makeDataPtr(
        reinterpret_cast<void *>(dptr),
        make_vmm_deleter(device_id, dptr, allocated_size),
        c10::Device(c10::kCUDA, device_id));

    // data_ptr now owns the mapping and VA.
    va_mapped = false;
    va_reserved = false;
    auto storage = c10::Storage(
        c10::Storage::use_byte_size_t(),
        static_cast<int64_t>(nbytes),
        std::move(data_ptr),
        /*allocator=*/nullptr,
        /*resizable=*/false);
    auto impl = c10::make_intrusive<at::TensorImpl>(
        std::move(storage),
        c10::DispatchKeySet(c10::DispatchKey::CUDA),
        c10::scalarTypeToTypeMeta(dtype));
    impl->set_sizes_contiguous(chunk_shape);
    auto keepalive = at::Tensor(std::move(impl));

    // Construct the complete return value before transferring the fd and
    // allocation handle to the caller.
    auto result = std::make_tuple(
        std::move(keepalive), static_cast<int64_t>(fd),
        static_cast<int64_t>(mem_handle));
    fd = -1;
    mem_handle_owned = false;
    return result;
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
 * `fds` are the POSIX fds exported by each rank via nvl_dist_alloc (passed
 * between processes by the caller, already valid in this process). The caller
 * may close the fds once they are imported.
 */
inline at::Tensor nvl_dist_map(
    const std::vector<int64_t> &chunk_shape,
    at::ScalarType dtype,
    const std::vector<int64_t> &fds,
    int64_t local_rank,
    int64_t world_size
) {
    TORCH_CHECK((int64_t)fds.size() == world_size,
        "fds.size()=", fds.size(), " != world_size=", world_size);

    auto [chunk_nbytes, chunk_allocated_size, device_id] = nvl_prepare(chunk_shape, dtype);

    TORCH_CHECK(chunk_allocated_size == chunk_nbytes,
        "Chunk byte size (", chunk_nbytes, ") must be aligned to VMM granularity (",
        chunk_allocated_size, "). Adjust chunk dimensions so that the total bytes "
        "are a multiple of the allocation granularity.");

    size_t total_allocated_size = chunk_allocated_size * world_size;
    size_t total_nbytes = chunk_nbytes * world_size;

    CUdeviceptr dptr = 0;
    bool va_reserved = false;
    size_t mapped_chunks = 0;
    auto cleanup = c10::make_scope_exit([&]() {
        for (size_t i = 0; i < mapped_chunks; i++) {
            cuMemUnmap(dptr + i * chunk_allocated_size, chunk_allocated_size);
        }
        if (va_reserved) {
            cuMemAddressFree(dptr, total_allocated_size);
        }
    });

    CUCHECK(cuMemAddressReserve(&dptr, total_allocated_size, 0, 0, 0));
    va_reserved = true;

    for (int64_t i = 0; i < world_size; i++) {
        int fd = static_cast<int>(fds[i]);
        CUmemGenericAllocationHandle mem_handle = {};
        bool mem_handle_owned = false;
        auto release_handle = c10::make_scope_exit([&]() {
            if (mem_handle_owned) {
                cuMemRelease(mem_handle);
            }
        });
        CUCHECK(cuMemImportFromShareableHandle(
            &mem_handle,
            reinterpret_cast<void *>(static_cast<intptr_t>(fd)),
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
        mem_handle_owned = true;

        CUdeviceptr chunk_va = dptr + i * chunk_allocated_size;
        CUCHECK(cuMemMap(chunk_va, chunk_allocated_size, 0, mem_handle, 0));
        mapped_chunks++;

        // All chunks get RW access for dispatch writes
        CUmemAccessDesc access = {};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = device_id;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CUCHECK(cuMemSetAccess(chunk_va, chunk_allocated_size, &access, 1));

        // Release exactly once. A failed driver call may report an earlier
        // asynchronous error, so retrying this opaque handle is not safe.
        mem_handle_owned = false;
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

    auto data_ptr = at::InefficientStdFunctionContext::makeDataPtr(
        reinterpret_cast<void *>(dptr), std::move(deleter),
        c10::Device(c10::kCUDA, device_id));
    cleanup.release();
    auto storage = c10::Storage(
        c10::Storage::use_byte_size_t(),
        static_cast<int64_t>(total_nbytes),
        std::move(data_ptr),
        /*allocator=*/nullptr,
        /*resizable=*/false);

    auto impl = c10::make_intrusive<at::TensorImpl>(
        std::move(storage),
        c10::DispatchKeySet(c10::DispatchKey::CUDA),
        c10::scalarTypeToTypeMeta(dtype));
    impl->set_sizes_contiguous(full_shape);
    auto result = at::Tensor(std::move(impl));
    return result;
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

// Recommended multicast alignment granularity (bytes). Both the addr and size
// of a bind must be aligned to it.
static inline size_t nvl_multicast_granularity(int num_devices) {
    CUmulticastObjectProp prop = {};
    prop.numDevices = static_cast<unsigned int>(num_devices);
    prop.size = 0;
    prop.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    prop.flags = 0;
    size_t gran = 0;
    CUCHECK(cuMulticastGetGranularity(
        &gran, &prop, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    return gran;
}

static inline int64_t get_multicast_granularity(int64_t num_devices) {
    return static_cast<int64_t>(
        nvl_multicast_granularity(static_cast<int>(num_devices)));
}

// root only: create the multicast object and export a POSIX fd.
// Returns (mc_handle as uint64, exported fd).
// mc_handle is a handle value valid within this process; Python holds it and
// passes it back unchanged to bind_map. The caller passes the fd to the other
// ranks and closes it after all peers have imported it.
inline std::tuple<int64_t, int64_t> nvl_multicast_create(
    int64_t size_bytes, int64_t num_devices
) {
    TORCH_CHECK(nvl_multicast_supported(),
        "Multicast not supported on this device");
    const size_t gran = nvl_multicast_granularity(static_cast<int>(num_devices));
    const size_t aligned =
        (static_cast<size_t>(size_bytes) + gran - 1) / gran * gran;

    CUmulticastObjectProp prop = {};
    prop.numDevices = static_cast<unsigned int>(num_devices);
    prop.size = aligned;
    prop.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    prop.flags = 0;

    CUmemGenericAllocationHandle mc_handle = {};
    bool mc_handle_owned = false;
    int fd = -1;
    auto cleanup = c10::make_scope_exit([&]() {
        if (fd >= 0) {
            close(fd);
        }
        if (mc_handle_owned) {
            cuMemRelease(mc_handle);
        }
    });

    CUCHECK(cuMulticastCreate(&mc_handle, &prop));
    mc_handle_owned = true;

    CUCHECK(cuMemExportToShareableHandle(
        &fd, mc_handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
    TORCH_CHECK(fd >= 0, "cuMemExportToShareableHandle returned invalid fd");

    auto result = std::make_tuple(
        static_cast<int64_t>(mc_handle), static_cast<int64_t>(fd));
    mc_handle_owned = false;
    fd = -1;
    return result;
}

// non-root: import the multicast object from the POSIX fd sent by root. The
// caller may close the fd once it is imported.
inline int64_t nvl_multicast_import(int64_t fd) {
    CUmemGenericAllocationHandle mc_handle;
    CUCHECK(cuMemImportFromShareableHandle(
        &mc_handle, reinterpret_cast<void*>(static_cast<intptr_t>(fd)),
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
    return static_cast<int64_t>(mc_handle);
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
    // Ownership transfers here: _create_nvl_multicast_view has no outer
    // release for the created/imported multicast handle. On success the
    // returned tensor's deleter releases it; this guard covers failure.
    CUmemGenericAllocationHandle mc_handle =
        static_cast<CUmemGenericAllocationHandle>(mc_handle_u64);
    // Borrowed: create_nvl_dist_multicast_tensor separately releases the
    // physical allocation handle in its outer finally block, including when
    // this function throws.
    CUmemGenericAllocationHandle mem_handle =
        static_cast<CUmemGenericAllocationHandle>(owned_mem_handle_u64);
    bool mc_handle_owned = true;
    CUdeviceptr mc_va = 0;
    bool va_reserved = false;
    bool va_mapped = false;
    auto cleanup = c10::make_scope_exit([&]() {
        if (va_mapped) {
            cuMemUnmap(mc_va, static_cast<size_t>(size_bytes));
        }
        if (va_reserved) {
            cuMemAddressFree(mc_va, static_cast<size_t>(size_bytes));
        }
        if (mc_handle_owned) {
            cuMemRelease(mc_handle);
        }
    });

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
    CUCHECK(cuMemAddressReserve(&mc_va, size, 0, 0, 0));
    va_reserved = true;
    CUCHECK(cuMemMap(mc_va, size, 0, mc_handle, 0));
    va_mapped = true;

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
    auto data_ptr = at::InefficientStdFunctionContext::makeDataPtr(
        reinterpret_cast<void*>(mc_va), std::move(deleter),
        c10::Device(c10::kCUDA, device_id));
    va_mapped = false;
    va_reserved = false;
    mc_handle_owned = false;
    auto storage = c10::Storage(
        c10::Storage::use_byte_size_t(),
        static_cast<int64_t>(size),
        std::move(data_ptr),
        /*allocator=*/nullptr,
        /*resizable=*/false);

    auto impl = c10::make_intrusive<at::TensorImpl>(
        std::move(storage),
        c10::DispatchKeySet(c10::DispatchKey::CUDA),
        c10::scalarTypeToTypeMeta(at::kInt));
    impl->set_sizes_contiguous(shape);
    auto result = at::Tensor(std::move(impl));
    return result;
}
