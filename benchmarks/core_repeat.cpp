#include <cuda.h>
// C submission loop avoids Python/ctypes gaps between short kernels. Event
// intervals still include driver submission starvation; they are NOT CUPTI
// kernel Activity durations.
extern "C" int core_repeat(CUfunction f, int grid, int block, int loops,
                          CUdeviceptr data, CUdeviceptr trace, int has_trace,
                          int count, float *ms) {
    (void)has_trace;
    void *args[] = {&data, &loops, &trace};
    CUevent start = nullptr, end = nullptr;
    CUresult r = cuEventCreate(&start, 0);
    if (r) return r;
    r = cuEventCreate(&end, 0);
    if (!r) r = cuEventRecord(start, nullptr);
    for (int i = 0; !r && i < count; ++i)
        r = cuLaunchKernel(f, grid, 1, 1, block, 1, 1, 0, nullptr, args, nullptr);
    if (!r) r = cuEventRecord(end, nullptr);
    if (!r) r = cuEventSynchronize(end);
    if (!r) r = cuEventElapsedTime(ms, start, end);
    if (start) cuEventDestroy(start);
    if (end) cuEventDestroy(end);
    return r;
}
