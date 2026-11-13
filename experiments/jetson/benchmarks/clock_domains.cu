// Paired clock64/globaltimer samples from the same thread. Summaries only.
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <set>
#include <vector>
struct Sample{uint64_t clock,timer,end_clock,end_timer;unsigned sm,end_sm;};
#define CHECK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));return 1;}}while(0)
__global__ void clock_samples(Sample*out){
    if(threadIdx.x)return;
    Sample s;asm volatile("mov.u64 %0, %%clock64; mov.u64 %1, %%globaltimer; mov.u32 %2, %%smid;":"=l"(s.clock),"=l"(s.timer),"=r"(s.sm));
    unsigned value=blockIdx.x;
    for(int i=0;i<4096;i++)asm volatile("add.u32 %0, %0, 1;":"+r"(value));
    asm volatile("mov.u64 %0, %%clock64; mov.u64 %1, %%globaltimer; mov.u32 %2, %%smid;":"=l"(s.end_clock),"=l"(s.end_timer),"=r"(s.end_sm));
    out[blockIdx.x]=s;
}
int main(){
    Sample*d;CHECK(cudaMalloc(&d,4096*sizeof(Sample)));std::vector<Sample>h(4096);cudaEvent_t a,b;
    CHECK(cudaEventCreate(&a));CHECK(cudaEventCreate(&b));
    clock_samples<<<4096,32>>>(d);CHECK(cudaDeviceSynchronize());
    puts("{\"rounds\":[");
    for(int r=0;r<7;r++){
        CHECK(cudaEventRecord(a));clock_samples<<<4096,32>>>(d);CHECK(cudaEventRecord(b));CHECK(cudaEventSynchronize(b));
        float ms;CHECK(cudaEventElapsedTime(&ms,a,b));CHECK(cudaMemcpy(h.data(),d,h.size()*sizeof(Sample),cudaMemcpyDeviceToHost));
        uint64_t cmin=UINT64_MAX,cmax=0,tmin=UINT64_MAX,tmax=0;unsigned migrated=0;std::set<unsigned>sms;
        uint64_t low[256],high[256];for(int s=0;s<256;s++){low[s]=UINT64_MAX;high[s]=0;}
        for(auto s:h){if(s.sm>=256)return 2;sms.insert(s.sm);if(s.sm!=s.end_sm)migrated++;
            if(s.clock<cmin)cmin=s.clock;if(s.clock>cmax)cmax=s.clock;if(s.timer<tmin)tmin=s.timer;if(s.timer>tmax)tmax=s.timer;
            if(s.clock<low[s.sm])low[s.sm]=s.clock;if(s.clock>high[s.sm])high[s.sm]=s.clock;}
        uint64_t per_sm=0;for(int s=0;s<256;s++)if(low[s]!=UINT64_MAX && high[s]-low[s]>per_sm)per_sm=high[s]-low[s];
        printf("%s{\"samples\":4096,\"sm_count\":%zu,\"sm_changed\":%u,\"event_ms\":%.6f,\"clock64_start_span_cycles\":%llu,\"globaltimer_start_span_ns\":%llu,\"max_within_sm_clock64_span_cycles\":%llu}",r?",\n":"",sms.size(),migrated,ms,(unsigned long long)(cmax-cmin),(unsigned long long)(tmax-tmin),(unsigned long long)per_sm);
    }
    puts("\n],\"raw_retained\":false}");CHECK(cudaEventDestroy(a));CHECK(cudaEventDestroy(b));CHECK(cudaFree(d));return 0;
}
