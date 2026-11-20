#include "barrier.h"

#include <cstdlib>
#include <string>
#include <torch/extension.h>
#include "shmem.h"
#include "host_device/shmem_common_types.h"
#include "ascend_moonep/kernels/launch.h"

// Same active-instance state used by the pinned SHMEM host barrier. Use its
// actual default stream, including after an instance context switch.
extern aclshmem_host_state_t g_state_host;

namespace ascend_moonep {
void BarrierAllOnStream(aclrtStream stream) {
    const char* setting = std::getenv("ASCEND_MOONEP_BARRIER_MODE");
    const std::string mode = setting == nullptr ? "mixed" : setting;
    TORCH_CHECK(mode == "mixed" || mode == "aiv",
                "ASCEND_MOONEP_BARRIER_MODE must be mixed or aiv");
    if (mode == "mixed") {
        aclshmemx_barrier_all_on_stream(stream);
        return;
    }
    TORCH_CHECK(aclshmemx_init_status() == ACLSHMEM_STATUS_IS_INITIALIZED,
                "shmem is not initialized, please initialize first");
    if (stream == nullptr) stream = g_state_host.default_stream;
    TORCH_CHECK(aclshmemx_barrier_all_on_stream_aiv(stream) == ACL_SUCCESS,
                "single AIV barrier launch failed");
}
}
