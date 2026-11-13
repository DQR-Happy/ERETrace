// Independent synthetic workloads. Every launch updates each output once;
// profiling must not turn one invocation into a counting pass plus a replay.
#include <stdint.h>
extern "C" __global__ void core_memory(uint32_t *a, int loops) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t v = a[i];
    for (int j = 0; j < loops; ++j) v = v * 1664525u + 1013904223u;
    a[i] = v;
}
extern "C" __global__ void core_pressure(uint32_t *a, int loops) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t x[24];
    #pragma unroll
    for (int k = 0; k < 24; ++k) x[k] = a[i] + k;
    for (int j = 0; j < loops; ++j) {
        #pragma unroll
        for (int k = 0; k < 24; ++k)
            asm volatile("mad.lo.u32 %0, %0, 1664525, %1;" : "+r"(x[k]) : "r"(x[(k+1)%24]));
    }
    uint32_t v = 0;
    #pragma unroll
    for (int k = 0; k < 24; ++k) v ^= x[k];
    a[i] = v;
}
extern "C" __global__ void core_reduce(uint32_t *a, int loops) {
    __shared__ uint32_t tmp[1024];
    unsigned t = threadIdx.x, i = blockIdx.x * blockDim.x + t;
    uint32_t v = a[i];
    for (int j = 0; j < loops; ++j) v = v * 1664525u + 1013904223u;
    tmp[t] = v;
    __syncthreads();
    for (unsigned n = blockDim.x/2; n; n >>= 1) {
        if (t < n) tmp[t] += tmp[t+n];
        __syncthreads();
    }
    a[i] = tmp[0] + t;
}
