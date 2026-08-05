#include <torch/extension.h>
#include "nvl_shared_buffer.cuh"

/**
 * Return the VMM allocation granularity in bytes for the current device.
 */
int64_t get_vmm_granularity() {
    int device_id;
    CUDACHECK(cudaGetDevice(&device_id));
    return static_cast<int64_t>(nvl_granularity_max(device_id));
}

PYBIND11_MODULE(_C, m) {
    m.attr("FABRIC_HANDLE_BYTES") = kFabricHandleBytes;
    m.def("nvl_dist_alloc", &nvl_dist_alloc,
          pybind11::arg("shape"), pybind11::arg("dtype"),
          pybind11::arg("use_fabric") = false,
          "Allocate a local chunk and export it for sharing, as a POSIX fd "
          "(same node) or a 64-byte fabric handle (same NVLink domain).");
    m.def("nvl_dist_map", &nvl_dist_map,
          pybind11::arg("chunk_shape"), pybind11::arg("dtype"),
          pybind11::arg("shareables"), pybind11::arg("local_rank"),
          pybind11::arg("world_size"), pybind11::arg("use_fabric") = false,
          "Map all ranks' chunks into a contiguous VA region (all RW).");
    m.def("nvl_fabric_supported", &nvl_fabric_supported,
          "Whether the current device can export/import fabric handles.");
    m.def("get_vmm_granularity", &get_vmm_granularity,
          "Return VMM allocation granularity in bytes.");
    m.def("get_multicast_granularity", &get_multicast_granularity,
          pybind11::arg("num_devices"),
          "Return multicast recommended allocation granularity in bytes.");
    m.def("nvl_multicast_supported", &nvl_multicast_supported,
          "Whether the current device supports multicast (NVSwitch SHARP).");
    m.def("nvl_multicast_create", &nvl_multicast_create,
          pybind11::arg("size_bytes"), pybind11::arg("num_devices"),
          pybind11::arg("use_fabric") = false,
          "Root-only: create a multicast object and export it for sharing.");
    m.def("nvl_multicast_import", &nvl_multicast_import,
          pybind11::arg("shareable"), pybind11::arg("use_fabric") = false,
          "Non-root: import a multicast object from the root's handle.");
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
}
