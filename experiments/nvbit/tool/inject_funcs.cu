#include <stdint.h>
struct WarpRecord { unsigned long long start; unsigned int elapsed; unsigned int smid; };
extern "C" __device__ __noinline__ void trace_scope(int pred, uint64_t ptr, int is_entry) {
    if(!pred)return;
    const uint64_t linear_tid = threadIdx.x + (uint64_t)blockDim.x * (threadIdx.y + (uint64_t)blockDim.y * threadIdx.z);
    if ((linear_tid & 31) != 0) return;
    const uint64_t threads_per_cta = (uint64_t)blockDim.x * blockDim.y * blockDim.z;
    const uint64_t warps_per_cta = (threads_per_cta + 31) / 32;
    const uint64_t linear_cta = blockIdx.x + (uint64_t)gridDim.x * (blockIdx.y + (uint64_t)gridDim.y * blockIdx.z);
    const uint64_t index = linear_cta * warps_per_cta + linear_tid / 32;
    uint64_t t; uint32_t sm;
    asm volatile("mov.u64 %0, %clock64;" : "=l"(t));
    asm volatile("mov.u32 %0, %smid;" : "=r"(sm));
    WarpRecord *r = (WarpRecord*)ptr;
    if (is_entry) { r[index].start=t; }
    else { r[index].elapsed=(unsigned int)(t-r[index].start); r[index].smid=sm; }
}
