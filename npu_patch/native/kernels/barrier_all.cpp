// SPDX-License-Identifier: Apache-2.0
#include "ascend_moonep/kernels/launch.h"
#include "acl/acl.h"
#include "kernel_operator.h"
#include "shmem.h"

// Match the SDK mixed barrier algorithm, using one AIV and no AIC.
// This is a barrier (quiet + team sync), not the vector sync-only wrapper.
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void aclshmemi_barrier_all_on_stream_aiv_kernel(uint64_t ffts_config) {
    util_set_ffts_config(ffts_config);
    aclshmemi_barrier_algo<true>(ACLSHMEM_TEAM_WORLD);
}

extern "C" ASCEND_MOONEP_EXPORT int aclshmemx_barrier_all_on_stream_aiv(void* stream) {
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    aclshmemi_barrier_all_on_stream_aiv_kernel<<<1, nullptr, stream>>>(shmemx_get_ffts_config());
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}
