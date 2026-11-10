#include <torch/extension.h>

#include "acl_shared_buffer.h"
#include "launcher.h"
#include "shmem.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "MoonEP NPU symmetric-memory extension (shmem C++ backend)";
    m.attr("SHMEM_BUFFER_DESC_ABI_BYTES") =
        py::int_(sizeof(aclshmemx_buffer_desc_t));
    m.attr("SHMEM_BUFFER_OPTIONAL_ATTR_ABI_BYTES") =
        py::int_(sizeof(aclshmemx_buffer_optional_attr_t));

    using py::arg;

    m.def("launch_device_barrier_bench", &LaunchDeviceBarrierBench,
          arg("iterations"), arg("block_dim") = 32,
          "Launch a device kernel containing only repeated "
          "aclshmemx_barrier_all_vec calls");

    // shmem init/finalize (new; absent in the CUDA build).
    m.def("acl_shmem_init", &ascend_moonep::acl_shmem_init,
          arg("my_pe"), arg("n_pes"), arg("local_mem_size"), arg("ip_port"),
          arg("device_id") = -1,
          "Initialize ACLSHMEM with default bootstrap (ip_port discovery)");
    m.def("acl_shmem_finalize", &ascend_moonep::acl_shmem_finalize,
          "Finalize ACLSHMEM");

    // UID bootstrap: rank 0 generates the UID (bytes); Python broadcasts it and
    // other ranks call init_uid. Must return py::bytes: the UID is binary, and
    // a std::string -> str conversion would fail UTF-8 decoding.
    m.def("acl_shmem_get_unique_id",
          []() { return py::bytes(ascend_moonep::acl_shmem_get_unique_id()); },
          "Generate a UID on rank 0 (raw bytes for dist broadcast)");
    m.def("acl_shmem_init_uid", &ascend_moonep::acl_shmem_init_uid,
          arg("my_pe"), arg("n_pes"), arg("local_mem_size"), arg("uid_bytes"),
          arg("device_id") = -1,
          "Initialize ACLSHMEM with UID bootstrap (follows the dist group)");

    // Symmetric allocation / contiguous view.
    m.def("acl_dist_alloc", &ascend_moonep::acl_dist_alloc,
          arg("shape"), arg("dtype"),
          "Allocate a symmetric chunk; returns (keepalive_tensor, -1, -1)");
    m.def("acl_dist_map", &ascend_moonep::acl_dist_map,
          arg("chunk_shape"), arg("dtype"), arg("local_ptr"),
          arg("local_rank"), arg("world_size"),
          "Build a contiguous cross-rank view over symmetric chunks");

    // Dispatch / Combine / Planning / GradReduce are registered in
    // csrc/torch_classes.cpp as torch.classes.AscendMoonEP.*.

    // Host side: translate a local symmetric address to pe's address (like create_sym_peer_addr).
    m.def("acl_shmem_ptr", &ascend_moonep::acl_shmem_ptr,
          arg("local_ptr"), arg("pe"),
          "Translate a local symmetric address to PE pe's address");

    // Host-side RMA (on_stream async): local -> remote / remote -> local.
    m.def("acl_shmem_putmem", &ascend_moonep::acl_shmem_putmem,
          arg("dst_local"), arg("src"), arg("nbytes"), arg("pe"), arg("stream"),
          "Copy local src to PE pe's symmetric region at dst_local (on stream)");
    m.def("acl_shmem_getmem", &ascend_moonep::acl_shmem_getmem,
          arg("dst"), arg("src_local"), arg("nbytes"), arg("pe"), arg("stream"),
          "Copy PE pe's symmetric region at src_local into local dst (on stream)");

    // shmem barrier (v1.3 aclshmem_barrier_all): wait for all ranks; ensure pending RMA completes.
    m.def("acl_shmem_barrier_all", &ascend_moonep::acl_shmem_barrier_all,
          "Global shmem barrier (waits for all ranks + pending RMA)");

    // ---- CANN VMM (aclrt* physical allocation + shareable VA) ----
    m.def("acl_vmm_granularity", &ascend_moonep::acl_vmm_granularity,
          arg("device_id"), "VMM minimum allocation granularity (bytes)");
    m.def("acl_vmm_malloc_physical", &ascend_moonep::acl_vmm_malloc_physical,
          arg("size"), arg("device_id"),
          "Allocate a physical memory handle via aclrtMallocPhysical");
    m.def("acl_vmm_free_physical", &ascend_moonep::acl_vmm_free_physical,
          arg("handle"), "Free a physical memory handle");
    m.def("acl_vmm_reserve", &ascend_moonep::acl_vmm_reserve,
          arg("size"), "Reserve a contiguous virtual address range");
    m.def("acl_vmm_release", &ascend_moonep::acl_vmm_release,
          arg("va"), "Release a reserved virtual address range");
    m.def("acl_vmm_map", &ascend_moonep::acl_vmm_map,
          arg("va"), arg("size"), arg("handle"), arg("offset"),
          "Map a physical handle (or subrange) into a reserved VA");
    m.def("acl_vmm_unmap", &ascend_moonep::acl_vmm_unmap,
          arg("va"), "Unmap the backing of a VA range");
    m.def("acl_vmm_set_access", &ascend_moonep::acl_vmm_set_access,
          arg("va"), arg("size"), arg("device_id"),
          "Set read/write access on a mapped VA range");
    m.def("acl_vmm_export_fabric",
          [](int64_t handle) {
              return py::bytes(ascend_moonep::acl_vmm_export_fabric(handle));
          },
          arg("handle"),
          "Export a physical handle to a Fabric V2 shareable handle (raw bytes)");
    // High-level transaction: malloc -> reserve -> map -> set_access -> export.
    // Returns (source_va, mem_handle, size, fabric_bytes).
    m.def("acl_vmm_alloc_source",
          [](int64_t size, int64_t device_id, int64_t access_device_id) {
              auto [va, handle, sz, fabric] =
                  ascend_moonep::acl_vmm_alloc_source(
                      size, device_id, access_device_id);
              return py::make_tuple(va, handle, sz, py::bytes(fabric));
          },
          arg("size"), arg("device_id"), arg("access_device_id") = -1,
          "Allocate a VMM source (source_va + mem handle + optional V2 Fabric "
          "bytes); device_id is the process-visible index, access_device_id is "
          "the physical card id for aclrtMemSetAccess (-1 = same as device_id)");
    m.def("acl_vmm_free_source", &ascend_moonep::acl_vmm_free_source,
          arg("va"), arg("handle"),
          "Free a VMM source (unmap -> release -> free_physical)");

    // ---- shmem buffer-heap (PR-653) ----
    m.def("acl_shmem_get_heap_base", &ascend_moonep::acl_shmem_get_heap_base,
          arg("mem_type"),
          "Return the shmem symmetric heap base for a memory type (0=host, 1=device)");
    m.def("acl_shmem_get_buffer_ptr", &ascend_moonep::acl_shmem_get_buffer_ptr,
          arg("local_va"),
          "Translate a registered local VA to its local symmetric address (0 if unregistered)");
    m.def("acl_shmem_init_with_buffers",
          &ascend_moonep::acl_shmem_init_with_buffers,
          arg("my_pe"), arg("n_pes"), arg("tail_bytes"),
          arg("data_op_engine_type"), arg("uid_bytes"), arg("descriptors"),
          arg("device_id") = -1,
          arg("enable_udma") = false, arg("qp_count") = 0,
          "Initialize ACLSHMEM with caller-mapped VMM buffers "
          "(PR-653 buffer-backed heap, UID bootstrap)");
    m.def("acl_shmem_init_with_buffers_ip_port",
          &ascend_moonep::acl_shmem_init_with_buffers_ip_port,
          arg("my_pe"), arg("n_pes"), arg("tail_bytes"),
          arg("data_op_engine_type"), arg("ip_port"), arg("descriptors"),
          arg("device_id") = -1,
          arg("enable_udma") = false, arg("qp_count") = 0,
          "Initialize ACLSHMEM with caller-mapped VMM buffers via ip_port/DEFAULT "
          "bootstrap (multi-node)");
    m.def("acl_shmem_barrier_all_on_stream",
          &ascend_moonep::acl_shmem_barrier_all_on_stream,
          arg("stream"),
          "Global shmem barrier issued on the given aclrtStream");
}
