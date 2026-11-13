/* Transfer-only mechanism test: same completed device buffer, no model/file.
 * Compile: cc -O3 -I/usr/local/cuda/include d2h_micro.c -ldl -o d2h_micro
 * API time includes two monotonic reads; allocation and verification are separate.
 */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#include <dlfcn.h>
#define API(n) static __typeof__(&n) fn_##n
API(cuInit); API(cuDeviceGet); API(cuCtxCreate); API(cuCtxDestroy);
API(cuMemAlloc); API(cuMemFree); API(cuMemsetD32); API(cuCtxSynchronize);
API(cuMemcpyDtoH); API(cuMemAllocHost); API(cuMemFreeHost);
#define STR1(n) #n
#define STR(n) STR1(n)
#define LOAD(n) do {fn_##n=dlsym(driver,STR(n)); if(!fn_##n){fprintf(stderr,"Missing %s\n",#n);exit(1);}} while(0)
#define OK(x) do{ CUresult r=(x); if(r!=CUDA_SUCCESS){fprintf(stderr,"CUDA %d line %d\n",r,__LINE__);exit(1);}}while(0)
static uint64_t now(void) {struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (uint64_t)t.tv_sec*1000000000+t.tv_nsec;}
static void touch(void *p,size_t n) {memset(p,0,n); __asm__ __volatile__(""::"r"(p):"memory");}
int main(int argc,char **argv) {
    int loops=argc>1?atoi(argv[1]):64,rounds=argc>2?atoi(argv[2]):7;
    if(loops<1 || rounds<1) return 1;
    void *driver=dlopen("/usr/lib/x86_64-linux-gnu/libcuda.so.580.105.08",RTLD_NOW);
    if(!driver) {fprintf(stderr,"%s\n",dlerror());return 1;}
    LOAD(cuInit);LOAD(cuDeviceGet);LOAD(cuCtxCreate);LOAD(cuCtxDestroy);
    LOAD(cuMemAlloc);LOAD(cuMemFree);LOAD(cuMemsetD32);LOAD(cuCtxSynchronize);
    LOAD(cuMemcpyDtoH);LOAD(cuMemAllocHost);LOAD(cuMemFreeHost);
    OK(fn_cuInit(0));CUdevice dev;OK(fn_cuDeviceGet(&dev,0));CUcontext ctx;OK(fn_cuCtxCreate(&ctx,0,dev));
    size_t sizes[]={512,1024,4096,24576,65536,131072,1048576,9723904};
    const char *modes[]={"fresh_pageable","reused_pageable","fresh_pretouched_pageable","reused_pinned"};
    printf("{\"loops\":%d,\"rounds\":%d,\"results\":[",loops,rounds);int comma=0;
    for(int round=0;round<rounds;round++) for(size_t k=0;k<sizeof(sizes)/sizeof(sizes[0]);k++) {
        size_t n=sizes[k];CUdeviceptr device;OK(fn_cuMemAlloc(&device,n));
        OK(fn_cuMemsetD32(device,0x1234abcd,n/4));OK(fn_cuCtxSynchronize());
        for(int mi=0;mi<4;mi++) {
            int mode=(mi+round)%4;void *host=NULL;
            uint64_t setup=now();
            if(mode==3) OK(fn_cuMemAllocHost(&host,n));
            else if(mode==1) host=malloc(n);
            if(mode==1 || mode==3) {if(!host) return 1;OK(fn_cuMemcpyDtoH(host,device,n));}
            setup=now()-setup;
            struct rusage a,b;getrusage(RUSAGE_SELF,&a);uint64_t wall=now(),api_ns=0,allocation_ns=0,pretouch_ns=0,free_ns=0;
            for(int j=0;j<loops;j++) {
                if(mode==0 || mode==2) {
                    uint64_t t=now();host=malloc(n);allocation_ns+=now()-t;if(!host) return 1;
                    if(mode==2) {t=now();touch(host,n);pretouch_ns+=now()-t;}
                }
                uint64_t t=now();OK(fn_cuMemcpyDtoH(host,device,n));api_ns+=now()-t;
                uint32_t *u=host;if(u[0]!=0x1234abcd || u[n/4-1]!=0x1234abcd) return 2;
                if(mode==0 || mode==2) {t=now();free(host);free_ns+=now()-t;host=NULL;}
            }
            wall=now()-wall;getrusage(RUSAGE_SELF,&b);
            /* Independent full-buffer validation outside measured loop. */
            void *check=malloc(n);OK(fn_cuMemcpyDtoH(check,device,n));
            for(size_t j=0;j<n/4;j++) if(((uint32_t*)check)[j]!=0x1234abcd) return 3;
            free(check);
            if(mode==3) OK(fn_cuMemFreeHost(host));else if(mode==1) free(host);
            printf("%s{\"round\":%d,\"bytes\":%zu,\"mode\":\"%s\",\"api_ns\":%llu,\"wall_ns\":%llu,\"setup_ns\":%llu,\"allocation_ns\":%llu,\"pretouch_ns\":%llu,\"free_ns\":%llu,\"minor_faults\":%ld,\"major_faults\":%ld,\"user_s\":%.9f,\"system_s\":%.9f}",comma?",":"",round+1,n,modes[mode],(unsigned long long)api_ns,(unsigned long long)wall,(unsigned long long)setup,(unsigned long long)allocation_ns,(unsigned long long)pretouch_ns,(unsigned long long)free_ns,b.ru_minflt-a.ru_minflt,b.ru_majflt-a.ru_majflt,(b.ru_utime.tv_sec-a.ru_utime.tv_sec)+(b.ru_utime.tv_usec-a.ru_utime.tv_usec)/1e6,(b.ru_stime.tv_sec-a.ru_stime.tv_sec)+(b.ru_stime.tv_usec-a.ru_stime.tv_usec)/1e6);comma=1;
        }
        OK(fn_cuMemFree(device));
    }
    printf("],\"full_validation\":true}\n");OK(fn_cuCtxDestroy(ctx));return 0;
}
