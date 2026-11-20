#include "kernel_operator.h"
#include "shmem.h"

#include "ascend_moonep/kernels/launch.h"

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
device_barrier_bench_kernel(int32_t iterations) {
    for (int32_t i = 0; i < iterations; ++i) {
        aclshmemx_barrier_all_vec();
    }
}

extern "C" ASCEND_MOONEP_EXPORT void device_barrier_bench_kernel_do(
    void* stream,
    int32_t iterations,
    int32_t block_dim) {
    device_barrier_bench_kernel<<<block_dim, nullptr, stream>>>(iterations);
}
