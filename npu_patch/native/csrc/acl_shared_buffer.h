#pragma once

#include <ATen/ATen.h>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace ascend_moonep {

// Init/finalize (DEFAULT bootstrap; all ranks pass the same ip_port, no MPI).
// Idempotent: returns 0 directly if already initialized.
int acl_shmem_init(int my_pe, int n_pes, int64_t local_mem_size,
                   const std::string& ip_port, int device_id);
int acl_shmem_finalize();

// UID bootstrap: rank 0 generates the UID (returned as raw bytes); other ranks
// call acl_shmem_init_uid after receiving the broadcast. The caller broadcasts
// the UID in Python with dist within the group, which naturally scopes the
// symmetric space (no manual ip_port needed).
std::string acl_shmem_get_unique_id();
int acl_shmem_init_uid(int my_pe, int n_pes, int64_t local_mem_size,
                       const std::string& uid_bytes, int device_id);

// Symmetric allocation of the local chunk. Returns (keepalive_tensor, -1, -1):
//   keepalive: tensor over the local chunk (data_ptr = chunk address), aclshmem_free on destruction;
//   fd / owned_handle: shmem has no cross-process fd or mem handle; fixed placeholder -1.
std::tuple<at::Tensor, int64_t, int64_t> acl_dist_alloc(
    const std::vector<int64_t>& shape, at::ScalarType dtype);

// Build a contiguous cross-rank view. local_ptr is the local chunk address
// (keepalive.data_ptr()) and local_rank must equal shmem my_pe. Returns a
// tensor of shape [world_size] + chunk_shape with stride[0] =
// shmem_rank_stride() / elem_size, so rank s's chunk starts at data_ptr +
// s * shmem_rank_stride(). The view owns no memory.
at::Tensor acl_dist_map(const std::vector<int64_t>& chunk_shape,
                        at::ScalarType dtype, int64_t local_ptr,
                        int64_t local_rank, int64_t world_size);

// Host side: translate a local symmetric address to pe's symmetric address
// (equivalent to shmem's aclshmem_ptr).
int64_t acl_shmem_ptr(int64_t local_ptr, int pe);

// Host-side RMA (wrappers of the shmem on_stream variants, stream-async):
//   put: dst_local is a local symmetric address (shmem translates it for pe), src is a local device address;
//   get: src_local is a local symmetric address (translated for pe), dst is a local device address;
//   stream: aclrtStream handle as an integer; 0 selects the default stream.
void acl_shmem_putmem(int64_t dst_local, int64_t src, int64_t nbytes, int pe,
                      int64_t stream);
void acl_shmem_getmem(int64_t dst, int64_t src_local, int64_t nbytes, int pe,
                      int64_t stream);

// shmem barrier: wait for all ranks and ensure pending RMA completes (required after on_stream).
void acl_shmem_barrier_all();

// ================================================================
// CANN VMM (aclrt* physical allocation + shareable VA, for buffer-backed shmem init)
// ================================================================
// Low-level primitives: handles are passed as int64 (internal reinterpret_cast<void*>); device_id is the NPU id.
int64_t acl_vmm_granularity(int64_t device_id);
int64_t acl_vmm_malloc_physical(int64_t size, int64_t device_id);
void acl_vmm_free_physical(int64_t handle);
int64_t acl_vmm_reserve(int64_t size);
void acl_vmm_release(int64_t va);
void acl_vmm_map(int64_t va, int64_t size, int64_t handle, int64_t offset);
void acl_vmm_unmap(int64_t va);
void acl_vmm_set_access(int64_t va, int64_t size, int64_t device_id);
// Export a Fabric V2 shareable handle (A3/A5: caller-supplied; shmem does not export).
// Returns the raw bytes of aclrtMemFabricHandle.
std::string acl_vmm_export_fabric(int64_t handle);
// High-level transaction: malloc -> reserve -> map -> set_access -> export
// (any failure rolls back unmap -> release -> free).
// Returns (source_va, mem_handle, size, fabric_bytes):
//   - A3/A5: fabric_bytes is the Fabric V2 handle (shmem init_with_buffers uses
//     it directly, no re-export);
//   - A2: fabric_bytes is empty; shmem performs the single V2 DEFAULT export on
//     mem_handle.
// The two id spaces are NOT equal under the ASCEND_RT_VISIBLE_DEVICES mask
// (measured on Ascend950):
//   - device_id       : process-visible device index 0..N-1 (granularity/malloc/map);
//   - access_device_id: desc.location.id for aclrtMemSetAccess, the PHYSICAL card
//                       id (= mask entries, e.g. 4/5 in "4,5"); -1 = same as device_id.
// A wrong space yields 107001 (allocation side) or 507899 (set_access side).
std::tuple<int64_t, int64_t, int64_t, std::string> acl_vmm_alloc_source(
    int64_t size, int64_t device_id, int64_t access_device_id = -1);
void acl_vmm_free_source(int64_t va, int64_t handle);

// ================================================================
// shmem buffer-heap (PR-653: aclshmemx_init_attr_with_buffers)
// ================================================================
// mem_type: 0=HOST_SIDE, 1=DEVICE_SIDE (default DEVICE_SIDE).
int64_t acl_shmem_get_heap_base(int64_t mem_type);
// Translate a local VA registered in the buffer-backed symmetric space to its
// local symmetric address; 0 = unregistered.
int64_t acl_shmem_get_buffer_ptr(int64_t local_va);
// descriptors = [(addr, size, mem_handle, fabric_bytes), ...], matching the new
// aclshmemx_buffer_desc_t (64B): when both mem/fabric handles are empty/0,
// optional_attr is null and shmem reserves/allocates the handle from addr;
// otherwise the caller's physical/fabric handles are passed through (non-empty
// fabric_bytes makes shmem use it directly, no export).
// Two bootstraps:
//   acl_shmem_init_with_buffers        -- UID bootstrap (single node; rank 0
//     generates the UID, broadcast via Python dist; ACLSHMEMX_INIT_WITH_UNIQUEID);
//   acl_shmem_init_with_buffers_ip_port -- ip_port/DEFAULT bootstrap (multi-node;
//     all ranks pass the same "tcp://<ip>:<port>"; ACLSHMEMX_INIT_WITH_DEFAULT).
int acl_shmem_init_with_buffers(
    int my_pe, int n_pes, int64_t tail_bytes, int data_op_engine_type,
    const std::string& uid_bytes,
    const std::vector<std::tuple<int64_t, int64_t, int64_t, std::string>>& descriptors,
    int device_id, bool enable_udma = false, uint32_t qp_count = 0);
int acl_shmem_init_with_buffers_ip_port(
    int my_pe, int n_pes, int64_t tail_bytes, int data_op_engine_type,
    const std::string& ip_port,
    const std::vector<std::tuple<int64_t, int64_t, int64_t, std::string>>& descriptors,
    int device_id, bool enable_udma = false, uint32_t qp_count = 0);
void acl_shmem_barrier_all_on_stream(int64_t stream);
}
