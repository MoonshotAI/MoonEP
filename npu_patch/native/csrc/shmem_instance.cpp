#include "shmem_instance.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "shmem.h"
#include "csrc/common/check.h"

namespace {

constexpr uint64_t kDeviceLargePageBytes = 2ULL * 1024ULL * 1024ULL;

std::mutex g_runtime_mutex;
int g_runtime_refcount = 0;
bool g_runtime_initialized = false;
int g_runtime_rank = -1;
int g_runtime_world_size = -1;
uint64_t g_runtime_local_mem_size = 0;
std::string g_runtime_ip_port;
aclshmemx_uniqueid_t g_default_flag_uid{};

void FillInitAttributes(
    int rank,
    int world_size,
    uint64_t local_mem_size,
    const std::string& ip_port,
    aclshmemx_init_attr_t* attributes) {
    if (attributes == nullptr) {
        throw std::invalid_argument("attributes must not be null");
    }
    std::memset(attributes, 0, sizeof(*attributes));

    const std::size_t ip_len = std::min(
        ip_port.size(),
        static_cast<std::size_t>(ACLSHMEM_MAX_IP_PORT_LEN - 1));
    std::copy_n(ip_port.data(), ip_len, attributes->ip_port);
    attributes->ip_port[ip_len] = '\0';
    if (attributes->ip_port[0] == '\0') {
        throw std::invalid_argument("SHMEM ip_port must not be empty");
    }

    // Match the library default (option_attr initializer in shmem_host_def.h):
    // (1 << 16) + sizeof(aclshmem_init_optional_attr_t).
    const int attr_version = (1 << 16) + static_cast<int>(sizeof(aclshmem_init_optional_attr_t));

    attributes->my_pe = rank;
    attributes->n_pes = world_size;
    attributes->local_mem_size = local_mem_size;
    attributes->option_attr = {
        attr_version,
        ACLSHMEM_DATA_OP_MTE,
        DEFAULT_TIMEOUT,
        DEFAULT_TIMEOUT,
        DEFAULT_TIMEOUT};
    attributes->comm_args = reinterpret_cast<void*>(&g_default_flag_uid);
}

}  // namespace

ShmemInstance::ShmemInstance(
    int64_t rank,
    int64_t world_size,
    std::string ip_port)
    : rank_(static_cast<int>(rank)),
      world_size_(static_cast<int>(world_size)),
      local_mem_size_(0),
      ip_port_(std::move(ip_port)),
      initialized_(false),
      acquired_global_runtime_(false),
      ffts_addr_(0) {
    if (rank < 0 || rank > std::numeric_limits<int>::max() ||
        world_size <= 0 || world_size > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("rank/world_size is out of range");
    }
    if (rank_ < 0 || rank_ >= world_size_) {
        throw std::invalid_argument("invalid rank/world_size");
    }
}

ShmemInstance::~ShmemInstance() {
    DestroyNoThrow();
}

void ShmemInstance::Init() {
    if (initialized_) {
        return;
    }
    if (local_mem_size_ == 0) {
        throw std::runtime_error("ShmemInstance local_mem_size is not configured; call init(local_mem_size)");
    }

    InitRuntime();

    try {
        ffts_addr_ = shmemx_get_ffts_config();
        const int actual_rank = aclshmem_my_pe();
        const int actual_world_size = aclshmem_n_pes();
        if (actual_rank != rank_ || actual_world_size != world_size_) {
            throw std::runtime_error("CANN SHMEM PE information does not match the process group");
        }

        aclshmem_barrier_all();
        initialized_ = true;
    } catch (...) {
        DestroyNoThrow();
        throw;
    }
}

void ShmemInstance::Init(int64_t local_mem_size) {
    if (local_mem_size <= 0 ||
        static_cast<uint64_t>(local_mem_size) > std::numeric_limits<uint64_t>::max()) {
        throw std::invalid_argument("local_mem_size must be greater than zero");
    }
    const auto requested_local_mem_size = static_cast<uint64_t>(local_mem_size);
    if ((requested_local_mem_size % kDeviceLargePageBytes) != 0) {
        throw std::invalid_argument("local_mem_size must be aligned to DEVICE_LARGE_PAGE_SIZE (2MB)");
    }
    if (local_mem_size_ != 0 && local_mem_size_ != requested_local_mem_size) {
        throw std::runtime_error("ShmemInstance local_mem_size has already been configured differently");
    }
    local_mem_size_ = requested_local_mem_size;
    Init();
}

void ShmemInstance::InitRuntime() {
    if (acquired_global_runtime_) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_runtime_mutex);
    if (!g_runtime_initialized) {
        // Buffer-backed era only: if the process already initialized via
        // aclshmemx_init_attr_with_buffers (ShmemRuntime.init_with_buffer), the
        // legacy ACLSHMEMX_INIT_WITH_DEFAULT path must not run a second init
        // (the two shmem init paths must not mix).
        if (aclshmemx_init_status() == ACLSHMEM_STATUS_IS_INITIALIZED) {
            throw std::runtime_error(
                "CANN SHMEM is already initialized in this process (e.g. by "
                "ShmemRuntime.init_with_buffer / "
                "aclshmemx_init_attr_with_buffers); ShmemInstance's legacy "
                "ACLSHMEMX_INIT_WITH_DEFAULT init must not run in the same "
                "process (two shmem init paths must not mix)");
        }
        aclshmemx_init_attr_t attributes{};
        FillInitAttributes(
            rank_,
            world_size_,
            local_mem_size_,
            ip_port_,
            &attributes);
        ASCEND_MOONEP_CHECK_SHMEM(
            aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes));

        g_runtime_initialized = true;
        g_runtime_rank = rank_;
        g_runtime_world_size = world_size_;
        g_runtime_local_mem_size = local_mem_size_;
        g_runtime_ip_port = ip_port_;
    } else if (
        g_runtime_rank != rank_ ||
        g_runtime_world_size != world_size_ ||
        g_runtime_local_mem_size != local_mem_size_ ||
        g_runtime_ip_port != ip_port_) {
        throw std::runtime_error(
            "CANN SHMEM has already been initialized with incompatible rank/world/local_mem_size/ip_port");
    }
    ++g_runtime_refcount;
    acquired_global_runtime_ = true;
}

void ShmemInstance::Barrier() {
    if (!initialized_) {
        throw std::runtime_error("ShmemInstance is not initialized");
    }
    aclshmem_barrier_all();
}

void ShmemInstance::WaitHandle(aclrtStream stream) const {
    if (!initialized_) {
        throw std::runtime_error("ShmemInstance is not initialized");
    }
    aclshmem_handle_t handle{};
    handle.team_id = ACLSHMEM_TEAM_WORLD;
    aclshmemx_handle_wait(handle, stream);
}

void ShmemInstance::Synchronize(aclrtStream stream) const {
    WaitHandle(stream);
    ASCEND_MOONEP_CHECK_ACL(aclrtSynchronizeStream(stream));
}

void ShmemInstance::Destroy() {
    if (!initialized_ && !acquired_global_runtime_) {
        return;
    }

    if (initialized_) {
        aclshmem_barrier_all();
    }

    initialized_ = false;
    ffts_addr_ = 0;

    if (acquired_global_runtime_) {
        std::lock_guard<std::mutex> lock(g_runtime_mutex);
        --g_runtime_refcount;
        acquired_global_runtime_ = false;
        if (g_runtime_refcount == 0 && g_runtime_initialized) {
            aclshmem_finalize();
            g_runtime_initialized = false;
            g_runtime_rank = -1;
            g_runtime_world_size = -1;
            g_runtime_local_mem_size = 0;
            g_runtime_ip_port.clear();
            std::memset(&g_default_flag_uid, 0, sizeof(g_default_flag_uid));
        }
    }
}

void ShmemInstance::DestroyNoThrow() noexcept {
    try {
        Destroy();
    } catch (...) {
    }
}

uint64_t ShmemInstance::FftsAddr() const {
    return ffts_addr_;
}

bool ShmemInstance::Initialized() const {
    return initialized_;
}

bool ShmemInstance::IsAvailable() const {
    return initialized_;
}

int64_t ShmemInstance::Rank() const {
    return rank_;
}

int64_t ShmemInstance::WorldSize() const {
    return world_size_;
}
