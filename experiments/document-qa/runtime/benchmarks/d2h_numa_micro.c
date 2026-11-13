/* Completed-buffer transfer test: first-touch NUMA, registration, async+wait.
 * Alloc/registration/first-touch costs are separate; no profiler in main runs.
 */
#define _GNU_SOURCE
#include <cuda.h>
#include <sched.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <dlfcn.h>
#include <errno.h>
#define API(n) static __typeof__(&n) fn_##n
API(cuInit);API(cuDeviceGet);API(cuCtxCreate);API(cuCtxDestroy);API(cuMemAlloc);API(cuMemFree);
API(cuMemsetD32);API(cuCtxSynchronize);API(cuMemcpyDtoH);API(cuMemAllocHost);API(cuMemFreeHost);
API(cuMemHostRegister);API(cuMemHostUnregister);API(cuMemcpyDtoHAsync);API(cuStreamCreate);API(cuStreamDestroy);
API(cuEventCreate);API(cuEventRecord);API(cuEventSynchronize);API(cuEventDestroy);
#define STR1(x) #x
#define STR(x) STR1(x)
#define LOAD(n) do {fn_##n=dlsym(driver,STR(n));if(!fn_##n)return 1;}while(0)
#define OK(x) do {CUresult z=(x);if(z){fprintf(stderr,"CUDA %d line %d\n",z,__LINE__);return 2;}}while(0)
static uint64_t now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return(uint64_t)t.tv_sec*1000000000+t.tv_nsec;}
static void placement(void *p,char *output,size_t capacity){
    FILE*f=fopen("/proc/self/maps","r");char line[2048];unsigned long a,b,base=0;
    while(fgets(line,sizeof(line),f))if(sscanf(line,"%lx-%lx",&a,&b)==2 && (unsigned long)p>=a && (unsigned long)p<b){base=a;break;}fclose(f);
    f=fopen("/proc/self/numa_maps","r");output[0]=0;
    while(fgets(line,sizeof(line),f))if(sscanf(line,"%lx",&a)==1 && a==base){line[strcspn(line,"\n")]=0;snprintf(output,capacity,"%s",line);break;}fclose(f);
}
static void quoted(const char*s){putchar('"');for(;*s;s++){if(*s=='"'||*s=='\\')putchar('\\');if((unsigned char)*s>=32)putchar(*s);}putchar('"');}
int main(int argc,char**argv){
    int rounds=argc>1?atoi(argv[1]):7;if(rounds<1)return 1;
    void *driver=dlopen("/usr/lib/x86_64-linux-gnu/libcuda.so.580.105.08",RTLD_NOW);
    LOAD(cuInit);LOAD(cuDeviceGet);LOAD(cuCtxCreate);LOAD(cuCtxDestroy);LOAD(cuMemAlloc);LOAD(cuMemFree);
    LOAD(cuMemsetD32);LOAD(cuCtxSynchronize);LOAD(cuMemcpyDtoH);LOAD(cuMemAllocHost);LOAD(cuMemFreeHost);
    LOAD(cuMemHostRegister);LOAD(cuMemHostUnregister);LOAD(cuMemcpyDtoHAsync);LOAD(cuStreamCreate);LOAD(cuStreamDestroy);
    LOAD(cuEventCreate);LOAD(cuEventRecord);LOAD(cuEventSynchronize);LOAD(cuEventDestroy);
    cpu_set_t original; if(sched_getaffinity(0,sizeof(original),&original))return 1;
    OK(fn_cuInit(0));CUdevice dev;OK(fn_cuDeviceGet(&dev,0));CUcontext ctx;OK(fn_cuCtxCreate(&ctx,0,dev));
    CUstream stream;OK(fn_cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING));CUevent event;OK(fn_cuEventCreate(&event,CU_EVENT_DISABLE_TIMING));
    size_t sizes[]={512,4096,65536,1048576,9723904};const char*modes[]={"mmap_pageable_sync","cuda_pinned_sync","mmap_registered_sync","cuda_pinned_async_wait"};
    printf("{\"rounds\":%d,\"cpu_nodes\":[0,24],\"results\":[",rounds);int comma=0;
    for(int round=0;round<rounds;round++)for(int ni=0;ni<2;ni++){
        int cpu=((ni+round)%2)?24:0;cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(cpu,&mask);if(sched_setaffinity(0,sizeof(mask),&mask))return 1;
        for(unsigned k=0;k<5;k++){
            size_t n=sizes[k];int loops=n==9723904?32:(32*1024*1024)/n;
            CUdeviceptr device;OK(fn_cuMemAlloc(&device,n));OK(fn_cuMemsetD32(device,0x1234abcd,n/4));OK(fn_cuCtxSynchronize());
            for(int mi=0;mi<4;mi++){
                int mode=(mi+round)%4;void*host=NULL;uint64_t alloc=now(),touch=0,reg=0;
                if(mode==1 || mode==3)OK(fn_cuMemAllocHost(&host,n));
                else{host=mmap(NULL,n,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);if(host==MAP_FAILED)return 1;}
                alloc=now()-alloc;
                uint64_t t=now();memset(host,0,n);__asm__ __volatile__(""::"r"(host):"memory");touch=now()-t;
                if(mode==2){t=now();OK(fn_cuMemHostRegister(host,n,0));reg=now()-t;}
                OK(fn_cuMemcpyDtoH(host,device,n));char map[2048];placement(host,map,sizeof(map));
                struct rusage a,b;getrusage(RUSAGE_SELF,&a);uint64_t wall=now(),api=0,wait=0;
                for(int j=0;j<loops;j++){
                    t=now();if(mode==3){OK(fn_cuMemcpyDtoHAsync(host,device,n,stream));api+=now()-t;t=now();OK(fn_cuEventRecord(event,stream));OK(fn_cuEventSynchronize(event));wait+=now()-t;}
                    else{OK(fn_cuMemcpyDtoH(host,device,n));api+=now()-t;}
                    uint32_t*u=host;if(u[0]!=0x1234abcd || u[n/4-1]!=0x1234abcd)return 3;
                }
                wall=now()-wall;getrusage(RUSAGE_SELF,&b);
                for(size_t j=0;j<n/4;j++)if(((uint32_t*)host)[j]!=0x1234abcd)return 3;
                t=now();if(mode==2)OK(fn_cuMemHostUnregister(host));if(mode==1||mode==3)OK(fn_cuMemFreeHost(host));else if(munmap(host,n))return 1;uint64_t release=now()-t;
                printf("%s{\"round\":%d,\"cpu\":%d,\"bytes\":%zu,\"loops\":%d,\"mode\":",comma?",":"",round+1,cpu,n,loops);quoted(modes[mode]);
                printf(",\"wall_ns\":%llu,\"enqueue_or_sync_api_ns\":%llu,\"event_wait_ns\":%llu,\"allocation_ns\":%llu,\"first_touch_ns\":%llu,\"registration_ns\":%llu,\"release_ns\":%llu,\"minor_faults\":%ld,\"major_faults\":%ld,\"user_s\":%.9f,\"system_s\":%.9f,\"numa_mapping\":",(unsigned long long)wall,(unsigned long long)api,(unsigned long long)wait,(unsigned long long)alloc,(unsigned long long)touch,(unsigned long long)reg,(unsigned long long)release,b.ru_minflt-a.ru_minflt,b.ru_majflt-a.ru_majflt,(b.ru_utime.tv_sec-a.ru_utime.tv_sec)+(b.ru_utime.tv_usec-a.ru_utime.tv_usec)/1e6,(b.ru_stime.tv_sec-a.ru_stime.tv_sec)+(b.ru_stime.tv_usec-a.ru_stime.tv_usec)/1e6);quoted(map);putchar('}');comma=1;
            }OK(fn_cuMemFree(device));
        }
    }
    puts("],\"full_validation\":true,\"gpu_numa_node\":0}");
    OK(fn_cuEventDestroy(event));OK(fn_cuStreamDestroy(stream));OK(fn_cuCtxDestroy(ctx));sched_setaffinity(0,sizeof(original),&original);return 0;
}
