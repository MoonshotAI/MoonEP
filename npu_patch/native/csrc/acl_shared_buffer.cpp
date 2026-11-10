#include "acl_shared_buffer.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <torch/extension.h>
#include <torch_npu/csrc/aten/common/from_blob.h>

#include "acl/acl.h"
#include "shmem.h"

namespace ascend_moonep {

namespace {

// Records which shmem init path initialized this process. The five entry
// points (acl_shmem_init / acl_shmem_init_uid / acl_shmem_init_with_buffers /
// acl_shmem_init_with_buffers_ip_port) share one aclshmemx_init_status()
// idempotency check that cannot tell paths apart and would hide mixing
// (buffer-backed heap vs legacy UID/ip_port heap, or the two buffer
// bootstraps). An explicit path enum enforces:
//   same-path repeat init -> idempotent return 0;
//   different-path init    -> error (paths must not mix);
//   already initialized externally (no path recorded here) -> error instead
//   of silently swallowing a second init.
enum class InitPath : int {
    NONE = 0,
    LEGACY_IPP,
    LEGACY_UID,
    BUFFERED,      // buffer-backed, UID bootstrap (single node)
    BUFFERED_IPP   // buffer-backed, ip_port/DEFAULT bootstrap (multi-node)
};
InitPath g_init_path = InitPath::NONE;
int g_finalize_failure_rc = 0;

static_assert(sizeof(aclshmemx_buffer_optional_attr_t) == 64,
              "unexpected aclshmemx_buffer_optional_attr_t ABI size");
static_assert(alignof(aclshmemx_buffer_optional_attr_t) == alignof(uint64_t),
              "unexpected aclshmemx_buffer_optional_attr_t ABI alignment");
static_assert(sizeof(aclshmemx_buffer_desc_t) == 64,
              "unexpected aclshmemx_buffer_desc_t ABI size");
static_assert(alignof(aclshmemx_buffer_desc_t) == alignof(uint64_t),
              "unexpected aclshmemx_buffer_desc_t ABI alignment");
static_assert(std::is_same_v<
                  decltype(aclshmemx_buffer_optional_attr_t::mem_handle),
                  aclshmemx_drv_mem_handle_t>,
              "unexpected aclshmemx buffer mem_handle field type");
static_assert(std::is_same_v<
                  decltype(aclshmemx_buffer_optional_attr_t::fabric_handle),
                  aclshmemx_mem_fabric_handle_t>,
              "unexpected aclshmemx buffer fabric_handle field type");

// Returns true if the caller should proceed with init; false means same-path
// idempotency (return 0). Throws if already initialized on another/unknown path.
bool acquire_init_path(InitPath requested) {
    TORCH_CHECK(g_finalize_failure_rc == 0,
        "shmem process is poisoned by a prior aclshmem_finalize failure (rc=",
        g_finalize_failure_rc,
        "); do not reinitialize or release caller-owned VMM backing");
    if (aclshmemx_init_status() != ACLSHMEM_STATUS_IS_INITIALIZED) {
        return true;  // not initialized yet, proceed
    }
    if (g_init_path == InitPath::NONE) {
        TORCH_CHECK(false,
            "shmem is already initialized before any ascend_moonep._C init "
            "(e.g. by MindSpeed libmoonep_symmetric_allocator or the shmem "
            "Python package); two shmem init paths must not mix — collect all "
            "descriptors into a single init_with_buffer instead");
    }
    TORCH_CHECK(g_init_path == requested,
        "shmem already initialized via a different path (requested path ",
        static_cast<int>(requested), ", current path ",
        static_cast<int>(g_init_path),
        "); two shmem init paths must not mix");
    return false;  // same-path idempotent
}

int64_t elem_size(at::ScalarType dtype) {
    return static_cast<int64_t>(c10::elementSize(dtype));
}

// Rank stride of the symmetric heap, ALIGNED = ALIGN_UP(heap_size, 1GB).
// Derived via aclshmem_ptr as |this heap base - the neighboring PE's base|.
// For the last rank, (pe+1)%npes wraps to an earlier rank (lower address); the
// difference is negative but its absolute value is still ALIGNED.
int64_t shmem_rank_stride() {
    const int npes = aclshmem_n_pes();
    const int pe = aclshmem_my_pe();
    if (npes <= 1) {
        return static_cast<int64_t>(1) << 30;  // no remote peers; fall back to 1GB
    }
    void* hb = aclshmemx_get_heap_base();
    TORCH_CHECK(hb != nullptr, "shmem not initialized or heap base is null");
    void* hb_next = aclshmem_ptr(hb, (pe + 1) % npes);
    TORCH_CHECK(hb_next != nullptr, "aclshmem_ptr(heap_base, next_pe) failed");
    const int64_t stride = std::llabs(
        static_cast<int64_t>(reinterpret_cast<uintptr_t>(hb_next) -
                             reinterpret_cast<uintptr_t>(hb)));
    TORCH_CHECK(stride > 0, "invalid shmem rank stride: ", stride);
    return stride;
}

// Build sizes/strides of the cross-rank view:
//   sizes[0]   = world_size
//   strides[0] = stride_elem (rank step, in elements)
//   sizes[i+1] = chunk_shape[i]; strides[i+1] = contiguous inner-chunk stride
void build_view_meta(const std::vector<int64_t>& chunk_shape,
                     int64_t world_size, int64_t stride_elem,
                     std::vector<int64_t>& sizes,
                     std::vector<int64_t>& strides) {
    const size_t ndim = chunk_shape.size();
    sizes.resize(ndim + 1);
    strides.resize(ndim + 1);
    sizes[0] = world_size;
    strides[0] = stride_elem;
    int64_t acc = 1;
    for (size_t i = ndim; i-- > 0;) {
        sizes[i + 1] = chunk_shape[i];
        strides[i + 1] = acc;
        acc *= chunk_shape[i];
    }
}

}  // namespace

// Ensure ACL is ready and verify the host-selected device.
// If the host (e.g. torch_npu) already initialized ACL, aclInit returns
// non-success (duplicate init); continuing is fine as long as the current
// device is queryable.
// device_id is the process-visible logical device index chosen by the host
// (torch.npu view; visible cards renumbered 0..N-1), with no mapping to my_pe
// in the EP subgroup. Python passes torch.npu.current_device(), and under the
// ASCEND_RT_VISIBLE_DEVICES mask aclrtGetDevice returns the same renumbered
// logical id, so the two agree. VMM physical allocation also requires it to be
// the current device (aclrtMallocPhysical returns 107001 for other visible ids).
// Never call set_device here: torchrun/the host must select the device before
// creating any tensor/VMM, otherwise 8-rank EP2 subgroup PE 0/1 would switch
// onto NPU 0/1.
void ensure_acl_ready(int device_id) {
    aclError acl_err = aclInit(nullptr);
    int32_t current_device = -1;
    aclError get_err = aclrtGetDevice(&current_device);
    TORCH_CHECK(get_err == ACL_SUCCESS && current_device >= 0,
                "aclrtGetDevice failed with ", get_err,
                " after aclInit returned ", acl_err,
                "; the host must select a physical NPU before shmem init");
    TORCH_CHECK(device_id < 0 || current_device == device_id,
                "current physical NPU ", current_device,
                " does not match device_id ", device_id,
                "; EP my_pe must not be used as a device id");
}

int acl_shmem_init(int my_pe, int n_pes, int64_t local_mem_size,
                   const std::string& ip_port, int device_id) {
    if (!acquire_init_path(InitPath::LEGACY_IPP)) {
        return 0;  // same-path idempotent
    }
    TORCH_CHECK(n_pes > 0 && my_pe >= 0 && my_pe < n_pes,
                "invalid shmem init args: my_pe=", my_pe, " n_pes=", n_pes);
    TORCH_CHECK(!ip_port.empty(),
                "ip_port must be provided for shmem default bootstrap");
    TORCH_CHECK(ip_port.size() + 1 <= ACLSHMEM_MAX_IP_PORT_LEN,
                "ip_port too long: ", ip_port);

    ensure_acl_ready(device_id);

    aclshmemx_init_attr_t attr;  // option_attr/comm_args keep their defaults
    attr.my_pe = my_pe;
    attr.n_pes = n_pes;
    attr.local_mem_size = static_cast<uint64_t>(local_mem_size);
    std::strncpy(attr.ip_port, ip_port.c_str(), ACLSHMEM_MAX_IP_PORT_LEN - 1);
    attr.ip_port[ACLSHMEM_MAX_IP_PORT_LEN - 1] = '\0';
    // Pin the data op engine to MTE, matching option_attr.data_op_engine_type
    // = OpEngineType.MTE on the legacy shmem Python ip_port path; do not rely
    // on the aclshmemx_init_attr_t default member initializer, whose value
    // would change silently if the default ever changes.
    attr.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_MTE;

    const int rc = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
    TORCH_CHECK(rc == 0,
                "aclshmemx_init_attr failed with ", rc,
                " (my_pe=", my_pe, ", n_pes=", n_pes,
                ", local_mem_size=", local_mem_size, ", ip_port=", ip_port, ")");
    g_init_path = InitPath::LEGACY_IPP;
    return rc;
}

int acl_shmem_finalize() {
    if (aclshmemx_init_status() != ACLSHMEM_STATUS_IS_INITIALIZED) {
        return g_finalize_failure_rc;
    }
    const int rc = aclshmem_finalize();
    if (rc == 0) {
        g_init_path = InitPath::NONE;
        g_finalize_failure_rc = 0;
    } else {
        g_finalize_failure_rc = rc;
    }
    return rc;
}

// Rank 0 only: generate a cross-process shared UID (raw bytes); the Python
// layer broadcasts it to the other ranks via dist.broadcast_object_list.
std::string acl_shmem_get_unique_id() {
    aclshmemx_uniqueid_t uid = ACLSHMEM_UNIQUEID_INITIALIZER;
    const int rc = aclshmemx_get_uniqueid(&uid);
    TORCH_CHECK(rc == 0, "aclshmemx_get_uniqueid failed with ", rc);
    return std::string(reinterpret_cast<char*>(&uid), sizeof(uid));
}

// UID bootstrap: uid_bytes is the single UID generated on rank 0 and broadcast
// over the group. The symmetric-space scope follows the Python broadcast group,
// so no ip_port is needed.
int acl_shmem_init_uid(int my_pe, int n_pes, int64_t local_mem_size,
                       const std::string& uid_bytes, int device_id) {
    if (!acquire_init_path(InitPath::LEGACY_UID)) {
        return 0;  // same-path idempotent
    }
    TORCH_CHECK(n_pes > 0 && my_pe >= 0 && my_pe < n_pes,
                "invalid shmem init args: my_pe=", my_pe, " n_pes=", n_pes);
    TORCH_CHECK(uid_bytes.size() == sizeof(aclshmemx_uniqueid_t),
                "uid_bytes size (", uid_bytes.size(),
                ") != sizeof(aclshmemx_uniqueid_t) (",
                sizeof(aclshmemx_uniqueid_t), ")");
    ensure_acl_ready(device_id);

    aclshmemx_uniqueid_t uid;
    std::memcpy(&uid, uid_bytes.data(), sizeof(uid));
    aclshmemx_init_attr_t attr;
    const int rc = aclshmemx_set_attr_uniqueid_args(
        my_pe, n_pes, static_cast<int64_t>(local_mem_size), &uid, &attr);
    TORCH_CHECK(rc == 0, "aclshmemx_set_attr_uniqueid_args failed with ", rc);

    const int init_rc = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, &attr);
    TORCH_CHECK(init_rc == 0,
                "aclshmemx_init_attr(UNIQUEID) failed with ", init_rc,
                " (my_pe=", my_pe, ", n_pes=", n_pes,
                ", local_mem_size=", local_mem_size, ")");
    g_init_path = InitPath::LEGACY_UID;
    return init_rc;
}

std::tuple<at::Tensor, int64_t, int64_t> acl_dist_alloc(
    const std::vector<int64_t>& shape, at::ScalarType dtype) {
    TORCH_CHECK(!shape.empty(), "shape must be non-empty");
    int64_t numel = 1;
    for (int64_t d : shape) {
        TORCH_CHECK(d > 0, "shape dims must be positive, got ", d);
        numel *= d;
    }
    const int64_t nbytes = numel * elem_size(dtype);

    void* ptr = aclshmem_malloc(static_cast<size_t>(nbytes));
    TORCH_CHECK(ptr != nullptr,
                "aclshmem_malloc(", nbytes, " bytes) failed; check shmem "
                "local_mem_size covers the symmetric allocations");

    // RAII: aclshmem_free on tensor destruction (matches CUDA make_vmm_tensor).
    auto deleter = [](void* p) {
        if (p != nullptr) {
            aclshmem_free(p);
        }
    };
    at::Tensor keepalive = at_npu::native::from_blob(
        ptr, shape, deleter, at::TensorOptions().dtype(dtype));
    // shmem has no cross-process fd or mem handle; placeholder -1.
    return {std::move(keepalive), -1, -1};
}

at::Tensor acl_dist_map(const std::vector<int64_t>& chunk_shape,
                        at::ScalarType dtype, int64_t local_ptr,
                        int64_t local_rank, int64_t world_size) {
    TORCH_CHECK(world_size > 0 && local_rank >= 0 && local_rank < world_size,
                "invalid rank args: local_rank=", local_rank,
                " world_size=", world_size);
    TORCH_CHECK(local_rank == aclshmem_my_pe(),
                "local_rank (", local_rank, ") != shmem my_pe (",
                aclshmem_my_pe(), "); shmem init must use the same rank ids");

    const int64_t stride_bytes = shmem_rank_stride();
    const int64_t elem = elem_size(dtype);
    TORCH_CHECK(stride_bytes % elem == 0,
                "shmem rank stride (", stride_bytes,
                ") not divisible by element size (", elem, ")");

    // View base = rank 0's chunk address. The symmetric heap is contiguous,
    // so rank s's chunk = ptr0 + s * stride_bytes.
    const int64_t ptr0 = local_ptr - local_rank * stride_bytes;

    std::vector<int64_t> sizes, strides;
    build_view_meta(chunk_shape, world_size, stride_bytes / elem, sizes,
                    strides);

    // The view owns no memory (pure reference), so the deleter is a no-op; each
    // rank's keepalive holds the physical memory and the caller binds lifetime
    // via `view._keepalive = keepalive`.
    at::Tensor view = at_npu::native::from_blob(
        reinterpret_cast<void*>(ptr0), sizes, strides, [](void*) {},
        at::TensorOptions().dtype(dtype));
    return view;
}

int64_t acl_shmem_ptr(int64_t local_ptr, int pe) {
    void* remote = aclshmem_ptr(reinterpret_cast<void*>(local_ptr), pe);
    TORCH_CHECK(remote != nullptr,
                "aclshmem_ptr(", local_ptr, ", pe=", pe, ") failed");
    return static_cast<int64_t>(reinterpret_cast<uintptr_t>(remote));
}

void acl_shmem_putmem(int64_t dst_local, int64_t src, int64_t nbytes, int pe,
                      int64_t stream) {
    aclshmemx_putmem_on_stream(
        reinterpret_cast<void*>(dst_local), reinterpret_cast<void*>(src),
        static_cast<size_t>(nbytes), pe,
        reinterpret_cast<aclrtStream>(static_cast<uintptr_t>(stream)));
}

void acl_shmem_getmem(int64_t dst, int64_t src_local, int64_t nbytes, int pe,
                      int64_t stream) {
    aclshmemx_getmem_on_stream(
        reinterpret_cast<void*>(dst), reinterpret_cast<void*>(src_local),
        static_cast<size_t>(nbytes), pe,
        reinterpret_cast<aclrtStream>(static_cast<uintptr_t>(stream)));
}

void acl_shmem_barrier_all() {
    aclshmem_barrier_all();
}

// ================================================================
// CANN VMM (aclrt* physical allocation + shareable VA)
// ================================================================

// Build ACL physical-mem properties: HBM huge page, DEVICE memory on device_id.
aclrtPhysicalMemProp vmm_mem_prop(int64_t device_id) {
    aclrtPhysicalMemProp prop{};
    prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    prop.memAttr = ACL_HBM_MEM_HUGE;
    prop.location = aclrtMemLocation{
        static_cast<uint32_t>(device_id), ACL_MEM_LOCATION_TYPE_DEVICE};
    prop.reserve = 0;
    return prop;
}

int64_t acl_vmm_granularity(int64_t device_id) {
    aclrtPhysicalMemProp prop = vmm_mem_prop(device_id);
    size_t gran = 0;
    const aclError rc = aclrtMemGetAllocationGranularity(
        &prop, ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM, &gran);
    TORCH_CHECK(rc == ACL_SUCCESS && gran > 0,
                "aclrtMemGetAllocationGranularity failed with ", rc,
                " (device_id=", device_id, ")");
    return static_cast<int64_t>(gran);
}

int64_t acl_vmm_malloc_physical(int64_t size, int64_t device_id) {
    TORCH_CHECK(size > 0, "vmm physical size must be positive, got ", size);
    aclrtDrvMemHandle handle = nullptr;
    aclrtPhysicalMemProp prop = vmm_mem_prop(device_id);
    const aclError rc = aclrtMallocPhysical(
        &handle, static_cast<size_t>(size), &prop, 0);
    TORCH_CHECK(rc == ACL_SUCCESS && handle != nullptr,
                "aclrtMallocPhysical(", size, ", device_id=", device_id,
                ") failed with ", rc);
    return reinterpret_cast<int64_t>(handle);
}

void acl_vmm_free_physical(int64_t handle) {
    const aclError rc =
        aclrtFreePhysical(reinterpret_cast<aclrtDrvMemHandle>(handle));
    TORCH_CHECK(rc == ACL_SUCCESS,
                "aclrtFreePhysical failed with ", rc);
}

int64_t acl_vmm_reserve(int64_t size) {
    TORCH_CHECK(size > 0, "vmm reserve size must be positive, got ", size);
    void* va = nullptr;
    const aclError rc = aclrtReserveMemAddress(&va, static_cast<size_t>(size),
                                               0, nullptr, 0);
    TORCH_CHECK(rc == ACL_SUCCESS && va != nullptr,
                "aclrtReserveMemAddress(", size, ") failed with ", rc);
    return reinterpret_cast<int64_t>(va);
}

void acl_vmm_release(int64_t va) {
    const aclError rc =
        aclrtReleaseMemAddress(reinterpret_cast<void*>(va));
    TORCH_CHECK(rc == ACL_SUCCESS,
                "aclrtReleaseMemAddress failed with ", rc);
}

void acl_vmm_map(int64_t va, int64_t size, int64_t handle, int64_t offset) {
    const aclError rc = aclrtMapMem(
        reinterpret_cast<void*>(va), static_cast<size_t>(size),
        static_cast<size_t>(offset),
        reinterpret_cast<aclrtDrvMemHandle>(handle), 0);
    TORCH_CHECK(rc == ACL_SUCCESS,
                "aclrtMapMem(va=", va, ", size=", size, ", offset=", offset,
                ") failed with ", rc);
}

void acl_vmm_unmap(int64_t va) {
    const aclError rc = aclrtUnmapMem(reinterpret_cast<void*>(va));
    TORCH_CHECK(rc == ACL_SUCCESS,
                "aclrtUnmapMem(va=", va, ") failed with ", rc);
}

void acl_vmm_set_access(int64_t va, int64_t size, int64_t device_id) {
    aclrtMemAccessDesc desc{};
    desc.flags = ACL_RT_MEM_ACCESS_FLAGS_READWRITE;
    desc.location = aclrtMemLocation{
        static_cast<uint32_t>(device_id), ACL_MEM_LOCATION_TYPE_DEVICE};
    const aclError rc = aclrtMemSetAccess(
        reinterpret_cast<void*>(va), static_cast<size_t>(size), &desc, 1);
    TORCH_CHECK(rc == ACL_SUCCESS,
                "aclrtMemSetAccess(va=", va, ", size=", size,
                ") failed with ", rc);
}

std::string acl_vmm_export_fabric(int64_t handle) {
    aclrtMemFabricHandle fabric{};
    const aclError rc = aclrtMemExportToShareableHandleV2(
        reinterpret_cast<aclrtDrvMemHandle>(handle),
        ACL_RT_VMM_EXPORT_FLAG_DISABLE_PID_VALIDATION,
        ACL_MEM_SHARE_HANDLE_TYPE_FABRIC, &fabric);
    TORCH_CHECK(rc == ACL_SUCCESS,
                "aclrtMemExportToShareableHandleV2(FABRIC) failed with ", rc);
    return std::string(reinterpret_cast<const char*>(&fabric), sizeof(fabric));
}

std::tuple<int64_t, int64_t, int64_t, std::string> acl_vmm_alloc_source(
    int64_t size, int64_t device_id, int64_t access_device_id) {
    const int64_t gran = acl_vmm_granularity(device_id);
    TORCH_CHECK(size > 0 && size % gran == 0,
                "vmm source size must be a positive multiple of granularity ",
                gran, ", got ", size);
    // A physical handle can be exported only once. A3/A5 export a Fabric V2
    // handle here and hand it to init_with_buffers, which consumes it directly
    // without re-exporting (see shmem ExportUserBufferHeap: has_fabric_handle
    // uses the passed value). A2 does not pre-export; SHMEM performs the single
    // V2 DEFAULT export on mem_handle. Architecture is a compile-time macro,
    // not a runtime SoC-name probe.
    int64_t handle = 0;
    int64_t va = 0;
    bool mapped = false;
    try {
        handle = acl_vmm_malloc_physical(size, device_id);
        va = acl_vmm_reserve(size);
        acl_vmm_map(va, size, handle, 0);
        mapped = true;
        // set_access's desc.location.id is the physical card id (not the
        // visible index used for allocation): under mask "4,5", visible=0
        // allocates fine but set_access(0) fails with 507899; set_access(4)
        // passes.
        acl_vmm_set_access(
            va, size, access_device_id < 0 ? device_id : access_device_id);
#if defined(ASCEND_MOONEP_ARCH_A3) || defined(ASCEND_MOONEP_ARCH_A5)
        std::string fabric = acl_vmm_export_fabric(handle);
        return {va, handle, size, std::move(fabric)};
#elif defined(ASCEND_MOONEP_ARCH_A2)
        return {va, handle, size, std::string()};
#else
#error "Unsupported ASCEND_MOONEP architecture"
#endif
    } catch (...) {
        if (mapped) {
            try { acl_vmm_unmap(va); } catch (...) {}
        }
        if (va != 0) {
            try { acl_vmm_release(va); } catch (...) {}
        }
        if (handle != 0) {
            try { acl_vmm_free_physical(handle); } catch (...) {}
        }
        throw;
    }
}

void acl_vmm_free_source(int64_t va, int64_t handle) {
    // Order follows MindSpeed AscendPhysicalAllocation.destroy:
    // unmap -> release -> free_physical.
    // Error isolation: each step gets its own try and a failure does not stop
    // the rest; throw only after all three finish if any failed. Otherwise a
    // throw at unmap would skip release/free and leak reserved VA and the
    // physical handle.
    std::string error;
    try {
        acl_vmm_unmap(va);
    } catch (const std::exception& e) {
        error += "unmap: ";
        error += e.what();
        error += "; ";
    }
    try {
        acl_vmm_release(va);
    } catch (const std::exception& e) {
        error += "release: ";
        error += e.what();
        error += "; ";
    }
    try {
        acl_vmm_free_physical(handle);
    } catch (const std::exception& e) {
        error += "free_physical: ";
        error += e.what();
        error += "; ";
    }
    if (!error.empty()) {
        TORCH_CHECK(false, "acl_vmm_free_source partially failed: ", error);
    }
}

// ================================================================
// shmem buffer-heap (PR-653)
// ================================================================

int64_t acl_shmem_get_heap_base(int64_t mem_type) {
    const aclshmem_mem_type_t t =
        (mem_type == 0) ? HOST_SIDE : DEVICE_SIDE;
    void* hb = aclshmemx_get_heap_base(t);
    TORCH_CHECK(hb != nullptr,
                "aclshmemx_get_heap_base(mem_type=", mem_type, ") failed");
    return reinterpret_cast<int64_t>(hb);
}

int64_t acl_shmem_get_buffer_ptr(int64_t local_va) {
    void* sym = aclshmemx_get_buffer_ptr(reinterpret_cast<void*>(local_va));
    // Unregistered returns 0 (the caller reports it as unregistered);
    // registered returns the local symmetric address.
    if (sym == nullptr) {
        return 0;
    }
    return reinterpret_cast<int64_t>(sym);
}

// Shared buffer-backed init. Both bootstraps (UID single-node / ip_port
// multi-node) share descriptor construction and the
// aclshmemx_init_attr_with_buffers call; only bootstrap_flags/init attrs differ:
//   ACLSHMEMX_INIT_WITH_UNIQUEID: uid_bytes is the rank-0-generated UID
//     (broadcast via Python dist); comm_args are filled with
//     aclshmemx_set_attr_uniqueid_args.
//   ACLSHMEMX_INIT_WITH_DEFAULT: ip_port is the config store address shared by
//     all ranks ("tcp://<ip>:<port>" or "tcp6://[ipv6]:<port>"); rank 0 runs
//     the server and the rest connect as clients (multi-node; same as the
//     user_buffer_mte example); comm_args stay null.
static int init_with_buffers_impl(
    InitPath path,
    aclshmemx_bootstrap_t bootstrap_flags,
    int my_pe,
    int n_pes,
    int64_t tail_bytes,
    int data_op_engine_type,
    const std::string& ip_port,
    const std::string& uid_bytes,
    const std::vector<std::tuple<int64_t, int64_t, int64_t, std::string>>& descriptors,
    int device_id,
    bool enable_udma,
    uint32_t qp_count) {
    if (!acquire_init_path(path)) {
        return 0;  // same-path idempotent
    }
    TORCH_CHECK(n_pes > 0 && my_pe >= 0 && my_pe < n_pes,
                "invalid shmem init args: my_pe=", my_pe, " n_pes=", n_pes);
    TORCH_CHECK(!descriptors.empty(),
                "buffer-backed shmem init requires at least one descriptor");
    if (bootstrap_flags == ACLSHMEMX_INIT_WITH_DEFAULT) {
        TORCH_CHECK(!ip_port.empty(),
                    "ip_port must be provided for IP-port buffer-backed shmem init");
        TORCH_CHECK(ip_port.size() + 1 <= ACLSHMEM_MAX_IP_PORT_LEN,
                    "ip_port too long: ", ip_port);
    } else {
        TORCH_CHECK(uid_bytes.size() == sizeof(aclshmemx_uniqueid_t),
                    "uid_bytes size (", uid_bytes.size(),
                    ") != sizeof(aclshmemx_uniqueid_t) (",
                    sizeof(aclshmemx_uniqueid_t), ")");
    }
    ensure_acl_ready(device_id);

    aclshmemx_init_attr_t attr;  // option_attr keeps default member initializers (incl. version)
    // set_attr_uniqueid_args stores &uid in attr.comm_args without copying it.
    // Keep the UID alive through the native initialization below.
    aclshmemx_uniqueid_t uid{};
    aclshmemx_uniqueid_t default_uid{};
    if (bootstrap_flags == ACLSHMEMX_INIT_WITH_UNIQUEID) {
        std::memcpy(&uid, uid_bytes.data(), sizeof(uid));
        int rc = aclshmemx_set_attr_uniqueid_args(
            my_pe, n_pes, tail_bytes, &uid, &attr);
        TORCH_CHECK(rc == 0, "aclshmemx_set_attr_uniqueid_args failed with ", rc);
    } else {
        // DEFAULT/IP-port: fill attr directly per the user_buffer_mte example;
        // comm_args stay null (the config store is parsed from attr.ip_port,
        // no UID involved).
        attr.my_pe = my_pe;
        attr.n_pes = n_pes;
        attr.local_mem_size = static_cast<uint64_t>(tail_bytes);
        std::copy_n(ip_port.data(), ip_port.size(), attr.ip_port);
        attr.ip_port[ip_port.size()] = '\0';
        attr.comm_args = nullptr;
    }
    attr.option_attr.data_op_engine_type =
        static_cast<data_op_engine_type_t>(data_op_engine_type);

    // UDMA: set QP count before init and ensure comm_args is non-null.
    if (enable_udma && qp_count > 0) {
        int qp_rc = aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, qp_count);
        TORCH_CHECK(qp_rc == 0,
                    "aclshmemx_set_qp_num failed with ", qp_rc,
                    " (qp_count=", qp_count, ")");
        // UDMA requires comm_args to be a valid pointer (even for DEFAULT bootstrap).
        if (attr.comm_args == nullptr) {
            attr.comm_args = &default_uid;
        }
    }

    // Build optional attrs + native buffer descs (matching
    // aclshmemx_buffer_desc_t).
    std::vector<aclshmemx_buffer_optional_attr_t> optional_attrs(descriptors.size());
    std::vector<aclshmemx_buffer_desc_t> native_buffers(descriptors.size());
    // Caller-supplied Fabric V2 handles (64B) must stay alive during init:
    // shmem deep-copies them, but optional_attr.fabric_handle points at these
    // objects, so the vector keeps them alive.
    std::vector<aclrtMemFabricHandle> fabric_handles(descriptors.size());
    for (size_t i = 0; i < descriptors.size(); ++i) {
        const int64_t addr = std::get<0>(descriptors[i]);
        const int64_t size = std::get<1>(descriptors[i]);
        const int64_t mem_handle = std::get<2>(descriptors[i]);
        const std::string& fabric_bytes = std::get<3>(descriptors[i]);
        TORCH_CHECK(addr > 0 && size > 0,
                    "descriptor[", i, "] requires a non-null addr and positive size");
        if (!fabric_bytes.empty()) {
            TORCH_CHECK(fabric_bytes.size() == sizeof(aclrtMemFabricHandle),
                        "descriptor[", i, "] fabric handle bytes size (",
                        fabric_bytes.size(), ") != sizeof(aclrtMemFabricHandle) (",
                        sizeof(aclrtMemFabricHandle), ")");
            std::memcpy(&fabric_handles[i], fabric_bytes.data(),
                        sizeof(aclrtMemFabricHandle));
        }

        auto& nb = native_buffers[i];
        nb.addr = reinterpret_cast<void*>(addr);
        nb.size = static_cast<uint64_t>(size);
        if (mem_handle != 0 || !fabric_bytes.empty()) {
            auto& optional = optional_attrs[i];
            optional.mem_handle = reinterpret_cast<aclshmemx_drv_mem_handle_t>(
                static_cast<uintptr_t>(mem_handle));
            if (!fabric_bytes.empty()) {
                // Passing fabric_handle makes shmem use it directly without
                // re-export (see shmem ExportUserBufferHeap: the
                // has_fabric_handle branch uses the passed value).
                optional.fabric_handle = &fabric_handles[i];
            }
            nb.optional_attr = &optional;
        } else {
            // VA-only: shmem reserves/allocates the handle from addr.
            nb.optional_attr = nullptr;
        }
    }

    const int init_rc = aclshmemx_init_attr_with_buffers(
        bootstrap_flags, &attr, native_buffers.data(), native_buffers.size());
    TORCH_CHECK(init_rc == 0,
                "aclshmemx_init_attr_with_buffers failed with ", init_rc,
                " (my_pe=", my_pe, ", n_pes=", n_pes, ", buffers=",
                native_buffers.size(), ")");
    g_init_path = path;
    return init_rc;
}

// UID bootstrap (single node): rank 0 generates the UID and Python broadcasts
// it; other ranks initialize with the same UID. Symmetric-space scope follows
// the broadcast group.
int acl_shmem_init_with_buffers(
    int my_pe, int n_pes, int64_t tail_bytes, int data_op_engine_type,
    const std::string& uid_bytes,
    const std::vector<std::tuple<int64_t, int64_t, int64_t, std::string>>& descriptors,
    int device_id, bool enable_udma, uint32_t qp_count) {
    return init_with_buffers_impl(
        InitPath::BUFFERED, ACLSHMEMX_INIT_WITH_UNIQUEID, my_pe, n_pes,
        tail_bytes, data_op_engine_type, /*ip_port=*/"", uid_bytes,
        descriptors, device_id, enable_udma, qp_count);
}

// IP-port/DEFAULT bootstrap (multi-node): all ranks pass the same ip_port
// ("tcp://<rank0_ip>:<port>"); rank 0 runs the config store server, the rest
// connect as clients.
int acl_shmem_init_with_buffers_ip_port(
    int my_pe, int n_pes, int64_t tail_bytes, int data_op_engine_type,
    const std::string& ip_port,
    const std::vector<std::tuple<int64_t, int64_t, int64_t, std::string>>& descriptors,
    int device_id, bool enable_udma, uint32_t qp_count) {
    return init_with_buffers_impl(
        InitPath::BUFFERED_IPP, ACLSHMEMX_INIT_WITH_DEFAULT, my_pe, n_pes,
        tail_bytes, data_op_engine_type, ip_port, /*uid_bytes=*/"",
        descriptors, device_id, enable_udma, qp_count);
}

void acl_shmem_barrier_all_on_stream(int64_t stream) {
    aclshmemx_barrier_all_on_stream(
        reinterpret_cast<aclrtStream>(static_cast<uintptr_t>(stream)));
}

}
