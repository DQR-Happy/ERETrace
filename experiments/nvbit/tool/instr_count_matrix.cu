#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <deque>
#include <dlfcn.h>
#include <string>
#include <unordered_set>
#include <vector>
#include "nvbit_tool.h"
#include "nvbit.h"
#include "utils/utils.h"
#include <cuda_runtime.h>

struct alignas(8) WarpRecord { uint64_t start; uint32_t elapsed; uint32_t smid; };
struct PackDesc { uint64_t source,dest,n; uint32_t first,blocks; };
struct NvFrame { uint64_t src,bytes; };
struct Slot {
    WarpRecord *d_raw=nullptr,*h_raw=nullptr;
    unsigned char *d_enc=nullptr,*h_enc=nullptr;
    PackDesc *d_desc=nullptr,desc={};
    unsigned *d_indices=nullptr;
    cudaEvent_t produced=nullptr,ready=nullptr;
    const uint64_t *nv_meta=nullptr;
    size_t enc_cap=0,nominal=0,raw_bytes=0;
    unsigned records=0,ordinal=0,grid[3]={},block[3]={},warps_per_cta=0;
    std::string function;
    bool used=false,attempted=false;
};

static std::vector<Slot> slots;
static unsigned max_warps=0,slot_count=4;
static std::string backend,trace_dir;
static unsigned local_bits=12,elapsed_bytes=2;
static bool verify_oracle=false,profiling=false,capture_all=false,active_from_start=false;
static bool stream_output=false,quiet_output=false,selected_launch=false;
static unsigned seen_target=0;
static int active_slot=-1;
static CUmodule encoder_module=nullptr;
static CUfunction encoder_init=nullptr,encoder_run=nullptr;
static void *nv_library=nullptr,*nv_state=nullptr;
static void* (*nv_create)(size_t,unsigned,unsigned,unsigned)=nullptr;
static size_t (*nv_bound)(void*,size_t,size_t)=nullptr;
static const uint64_t* (*nv_submit)(void*,unsigned,const NvFrame*,unsigned,size_t,uint64_t,void*)=nullptr;
static cudaStream_t copy_stream=nullptr;
static pthread_mutex_t nvbit_mutex=PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t queue_mutex=PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t queue_condition=PTHREAD_COND_INITIALIZER;
static pthread_t writer_thread;
static bool writer_started=false,writer_stop=false;
static std::deque<unsigned> ready_queue;
static unsigned pending=0;
static FILE *stream_file=nullptr;
static uint64_t measured_d2h=0,measured_file=0,measured_encoded=0,measured_raw=0,measured_fallback=0;
static std::unordered_set<CUfunction> instrumented;
static thread_local bool tool_cuda=false;

static void open_stream() {
    if(stream_file)return;
    assert(stream_output);
    std::string path=trace_dir+"/trace-stream.nvb";
    stream_file=fopen(path.c_str(),"wb");assert(stream_file);
    assert(fwrite("NVB1\n",1,5,stream_file)==5);
    measured_file+=5;
}

static void *writer_main(void *opaque) {
    CUcontext ctx=(CUcontext)opaque;
    nvbit_set_tool_pthread(pthread_self());
    tool_cuda=true;
    assert(cuCtxSetCurrent(ctx)==CUDA_SUCCESS);
    for(;;) {
        pthread_mutex_lock(&queue_mutex);
        while(ready_queue.empty()&&!writer_stop)
            pthread_cond_wait(&queue_condition,&queue_mutex);
        if(ready_queue.empty()&&writer_stop){pthread_mutex_unlock(&queue_mutex);break;}
        unsigned id=ready_queue.front();ready_queue.pop_front();
        pthread_mutex_unlock(&queue_mutex);
        Slot &s=slots[id];
        CUDA_SAFECALL(cudaEventSynchronize(s.ready));
        unsigned flags=0;
        bool encoded=false;
        size_t payload=s.raw_bytes;
        const void *data=s.h_raw;
        if(s.attempted && backend=="ours") {
            memcpy(&flags,s.h_enc,sizeof(flags));
            if(flags==0){encoded=true;payload=s.nominal;data=s.h_enc;}
        } else if(s.attempted && backend=="nvcomp") {
            if(s.nv_meta[0] && s.nv_meta[0]!=UINT64_MAX &&
               s.nv_meta[0]==s.nv_meta[1] && s.nv_meta[0]<=s.enc_cap) {
                encoded=true;payload=s.nv_meta[0];
                CUDA_SAFECALL(cudaMemcpyAsync(s.h_enc,s.d_enc,payload,cudaMemcpyDeviceToHost,copy_stream));
                CUDA_SAFECALL(cudaStreamSynchronize(copy_stream));
                measured_d2h+=payload;
                data=s.h_enc;
            } else flags=32;
        }
        if(!encoded && s.attempted)measured_fallback++;
        if(!encoded && s.attempted || (!s.attempted && backend!="raw")) {
            CUDA_SAFECALL(cudaMemcpyAsync(s.h_raw,s.d_raw,s.raw_bytes,cudaMemcpyDeviceToHost,copy_stream));
            CUDA_SAFECALL(cudaStreamSynchronize(copy_stream));
            measured_d2h+=s.raw_bytes;
        }
        if(verify_oracle) {
            if(encoded){
                CUDA_SAFECALL(cudaMemcpyAsync(s.h_raw,s.d_raw,s.raw_bytes,cudaMemcpyDeviceToHost,copy_stream));
                CUDA_SAFECALL(cudaStreamSynchronize(copy_stream));
                char path[4096];snprintf(path,sizeof(path),"%s/nvbit-%u.oracle.raw",trace_dir.c_str(),s.ordinal);
                FILE *oracle=fopen(path,"wb");assert(oracle);
                assert(fwrite(s.h_raw,1,s.raw_bytes,oracle)==s.raw_bytes);
                assert(fclose(oracle)==0);
            }
            for(unsigned i=0;i<s.records;i++)
                assert(s.h_raw[i].start>0 && s.h_raw[i].elapsed>0 && s.h_raw[i].smid<82);
        }
        open_stream();
        const char *fmt=encoded?(backend=="nvcomp"?"nvc":"enc"):"raw";
        int head=fprintf(stream_file,
            "FRAME launch=%u records=%u bytes=%zu format=%s flags=%u grid=%u,%u,%u block=%u,%u,%u function=%s\n",
            s.ordinal,s.records,payload,fmt,flags,s.grid[0],s.grid[1],s.grid[2],
            s.block[0],s.block[1],s.block[2],s.function.c_str());
        assert(head>0);
        assert(fwrite(data,1,payload,stream_file)==payload);
        measured_file+=head+payload;
        if(encoded)measured_encoded++;else measured_raw++;
        pthread_mutex_lock(&queue_mutex);
        s.used=false;pending--;
        pthread_cond_broadcast(&queue_condition);
        pthread_mutex_unlock(&queue_mutex);
    }
    return nullptr;
}

static void allocate_once(CUcontext ctx) {
    if(!slots.empty())return;
    CUDA_SAFECALL(cudaStreamCreateWithFlags(&copy_stream,cudaStreamNonBlocking));
    if(backend=="ours") {
        const char *path=getenv("NVBIT_ENCODER_CUBIN");assert(path);
        assert(cuModuleLoad(&encoder_module,path)==CUDA_SUCCESS);
        assert(cuModuleGetFunction(&encoder_init,encoder_module,
            local_bits==12?"neutrino_capture_compact_init":"neutrino_capture_reference_init")==CUDA_SUCCESS);
        assert(cuModuleGetFunction(&encoder_run,encoder_module,
            local_bits==12?"neutrino_capture_compact_batch":"neutrino_capture_reference_batch")==CUDA_SUCCESS);
    }
    if(backend=="nvcomp") {
        const char *path=getenv("NVBIT_NVCOMP_LIB");assert(path);
        nv_library=dlopen(path,RTLD_NOW|RTLD_LOCAL);assert(nv_library);
        nv_create=(decltype(nv_create))dlsym(nv_library,"nvpipe_create");
        nv_bound=(decltype(nv_bound))dlsym(nv_library,"nvpipe_bound");
        nv_submit=(decltype(nv_submit))dlsym(nv_library,"nvpipe_submit");
        assert(nv_create&&nv_bound&&nv_submit);
        nv_state=nv_create(size_t(max_warps)*16,1,65536,2);assert(nv_state);
    }
    slots.resize(slot_count);
    for(Slot &s:slots) {
        CUDA_SAFECALL(cudaMalloc((void**)&s.d_raw,size_t(max_warps)*16));
        CUDA_SAFECALL(cudaHostAlloc((void**)&s.h_raw,size_t(max_warps)*16,cudaHostAllocDefault));
        if(backend=="ours") {
            s.enc_cap=size_t(max_warps)*16+2064;
            CUDA_SAFECALL(cudaMalloc((void**)&s.d_desc,sizeof(PackDesc)));
            CUDA_SAFECALL(cudaMalloc((void**)&s.d_indices,256*sizeof(unsigned)));
        } else if(backend=="nvcomp") s.enc_cap=nv_bound(nv_state,size_t(max_warps)*16,16384);
        if(s.enc_cap){
            CUDA_SAFECALL(cudaMalloc((void**)&s.d_enc,s.enc_cap));
            CUDA_SAFECALL(cudaHostAlloc((void**)&s.h_enc,s.enc_cap,cudaHostAllocDefault));
        }
        CUDA_SAFECALL(cudaEventCreateWithFlags(&s.produced,cudaEventDisableTiming));
        CUDA_SAFECALL(cudaEventCreateWithFlags(&s.ready,cudaEventDisableTiming));
    }
    assert(pthread_create(&writer_thread,nullptr,writer_main,ctx)==0);
    writer_started=true;
    printf("NVBIT_ASYNC_ALLOCATION slots=%u raw_capacity_each=%zu encoded_capacity_each=%zu\n",
           slot_count,size_t(max_warps)*16,slots[0].enc_cap);
}

void nvbit_at_init() {
    const char *b=getenv("NVBIT_BACKEND");backend=b?b:"raw";
    assert(backend=="raw"||backend=="ours"||backend=="nvcomp");
    const char *d=getenv("SCOPE_TRACE_DIR");assert(d);trace_dir=d;
    const char *m=getenv("NVBIT_MAX_WARPS");max_warps=m?strtoul(m,nullptr,10):1048576;
    const char *n=getenv("NVBIT_ASYNC_SLOTS");slot_count=n?strtoul(n,nullptr,10):4;
    assert(max_warps>0&&max_warps<=1048576&&slot_count>=2&&slot_count<=32);
    if(getenv("NVBIT_LOCAL_BITS"))local_bits=strtoul(getenv("NVBIT_LOCAL_BITS"),nullptr,10);
    if(getenv("NVBIT_ELAPSED_BYTES"))elapsed_bytes=strtoul(getenv("NVBIT_ELAPSED_BYTES"),nullptr,10);
    assert((local_bits==12||local_bits==16)&&(elapsed_bytes==2||elapsed_bytes==3));
    verify_oracle=getenv("NVBIT_VERIFY")&&atoi(getenv("NVBIT_VERIFY"))!=0;
    capture_all=getenv("NVBIT_CAPTURE_ALL")&&atoi(getenv("NVBIT_CAPTURE_ALL"))!=0;
    active_from_start=getenv("ACTIVE_FROM_START")&&atoi(getenv("ACTIVE_FROM_START"))!=0;
    stream_output=getenv("NVBIT_STREAM")&&atoi(getenv("NVBIT_STREAM"))!=0;
    quiet_output=getenv("NVBIT_QUIET")&&atoi(getenv("NVBIT_QUIET"))!=0;
    assert(stream_output);
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC","1",1);
    printf("NVBIT_ASYNC_CONFIG backend=%s slots=%u local_bits=%u elapsed_bytes=%u\n",
           backend.c_str(),slot_count,local_bits,elapsed_bytes);
}

void nvbit_at_term() {
    if(writer_started){
        pthread_mutex_lock(&queue_mutex);writer_stop=true;pthread_cond_broadcast(&queue_condition);
        pthread_mutex_unlock(&queue_mutex);pthread_join(writer_thread,nullptr);
    }
}

static void instrument_scope(CUcontext ctx,CUfunction func) {
    if(!instrumented.insert(func).second)return;
    allocate_once(ctx);
    const std::vector<Instr*> &ins=nvbit_get_instrs(ctx,func);
    assert(!ins.empty());
    Instr *entry=ins.front();
    nvbit_insert_call(entry,"trace_scope",IPOINT_BEFORE);
    nvbit_add_call_arg_guard_pred_val(entry);
    nvbit_add_call_arg_launch_val64(entry,0);
    nvbit_add_call_arg_const_val32(entry,1);
    for(Instr *i:ins){
        const char *op=i->getOpcode();
        if(op&&(!strcmp(op,"EXIT")||!strcmp(op,"RET"))){
            nvbit_insert_call(i,"trace_scope",IPOINT_BEFORE);
            nvbit_add_call_arg_guard_pred_val(i);
            nvbit_add_call_arg_launch_val64(i,0);
            nvbit_add_call_arg_const_val32(i,0);
        }
    }
}

static CUstream launch_stream(nvbit_api_cuda_t cbid,void *params) {
    if(cbid==API_CUDA_cuLaunchKernelEx||cbid==API_CUDA_cuLaunchKernelEx_ptsz)
        return ((cuLaunchKernelEx_params*)params)->config->hStream;
    if(cbid==API_CUDA_cuLaunchKernel||cbid==API_CUDA_cuLaunchKernel_ptsz)
        return ((cuLaunchKernel_params*)params)->hStream;
    return nullptr;
}

void nvbit_at_cuda_event(CUcontext ctx,int is_exit,nvbit_api_cuda_t cbid,
                         const char *name,void *params,CUresult *status) {
    if(tool_cuda)return;
    if(cbid==API_CUDA_cuProfilerStart&&is_exit){
        assert(pending==0&&!stream_file);
        profiling=true;seen_target=0;
        measured_d2h=measured_file=measured_encoded=measured_raw=measured_fallback=0;
        return;
    }
    if(cbid==API_CUDA_cuProfilerStop&&is_exit){
        profiling=false;
        pthread_mutex_lock(&queue_mutex);
        while(pending)pthread_cond_wait(&queue_condition,&queue_mutex);
        pthread_mutex_unlock(&queue_mutex);
        if(stream_file){assert(fclose(stream_file)==0);stream_file=nullptr;}
        printf("NVBIT_ASYNC_STATS backend=%s launches=%u encoded=%llu raw=%llu fallback=%llu d2h_bytes=%llu file_bytes=%llu\n",
            backend.c_str(),seen_target,(unsigned long long)measured_encoded,
            (unsigned long long)measured_raw,(unsigned long long)measured_fallback,
            (unsigned long long)measured_d2h,(unsigned long long)measured_file);
        return;
    }
    if(!(cbid==API_CUDA_cuLaunchKernel||cbid==API_CUDA_cuLaunchKernel_ptsz||
         cbid==API_CUDA_cuLaunchKernelEx||cbid==API_CUDA_cuLaunchKernelEx_ptsz))return;
    CUfunction func=(cbid==API_CUDA_cuLaunchKernelEx||cbid==API_CUDA_cuLaunchKernelEx_ptsz)
        ?((cuLaunchKernelEx_params*)params)->f:((cuLaunchKernel_params*)params)->f;
    if(!is_exit){
        pthread_mutex_lock(&nvbit_mutex);
        bool armed=active_from_start||profiling;
        selected_launch=armed&&capture_all;
        if(selected_launch){
            tool_cuda=true;
            instrument_scope(ctx,func);
            pthread_mutex_lock(&queue_mutex);
            unsigned id=slot_count;
            while(id==slot_count){
                for(unsigned i=0;i<slot_count;i++)if(!slots[i].used){id=i;break;}
                if(id==slot_count)pthread_cond_wait(&queue_condition,&queue_mutex);
            }
            Slot &s=slots[id];s.used=true;pending++;
            pthread_mutex_unlock(&queue_mutex);
            active_slot=int(id);
            CUstream stream=launch_stream(cbid,params);
            if(cbid==API_CUDA_cuLaunchKernelEx||cbid==API_CUDA_cuLaunchKernelEx_ptsz){
                auto *p=(cuLaunchKernelEx_params*)params;
                s.grid[0]=p->config->gridDimX;s.grid[1]=p->config->gridDimY;s.grid[2]=p->config->gridDimZ;
                s.block[0]=p->config->blockDimX;s.block[1]=p->config->blockDimY;s.block[2]=p->config->blockDimZ;
            }else{
                auto *p=(cuLaunchKernel_params*)params;
                s.grid[0]=p->gridDimX;s.grid[1]=p->gridDimY;s.grid[2]=p->gridDimZ;
                s.block[0]=p->blockDimX;s.block[1]=p->blockDimY;s.block[2]=p->blockDimZ;
            }
            s.warps_per_cta=(s.block[0]*s.block[1]*s.block[2]+31)/32;
            s.records=s.grid[0]*s.grid[1]*s.grid[2]*s.warps_per_cta;
            assert(s.records>0&&s.records<=max_warps);
            s.raw_bytes=size_t(s.records)*16;s.ordinal=seen_target;
            s.function=nvbit_get_func_name(ctx,func,true);
            CUDA_SAFECALL(cudaMemsetAsync(s.d_raw,0,s.raw_bytes,(cudaStream_t)stream));
            nvbit_set_at_launch(ctx,func,(uint64_t)s.d_raw,stream);
            tool_cuda=false;
        }
        nvbit_enable_instrumented(ctx,func,selected_launch);
    }else{
        if(selected_launch){
            tool_cuda=true;
            Slot &s=slots[active_slot];
            CUstream stream=launch_stream(cbid,params);
            CUDA_SAFECALL(cudaEventRecord(s.produced,(cudaStream_t)stream));
            CUDA_SAFECALL(cudaStreamWaitEvent(copy_stream,s.produced,0));
            s.attempted=s.raw_bytes>=16384&&backend!="raw";
            s.nominal=0;s.nv_meta=nullptr;
            if(backend=="raw"){
                CUDA_SAFECALL(cudaMemcpyAsync(s.h_raw,s.d_raw,s.raw_bytes,cudaMemcpyDeviceToHost,copy_stream));
                measured_d2h+=s.raw_bytes;
            }else if(s.attempted&&backend=="ours"){
                unsigned groups=(s.records+s.warps_per_cta-1)/s.warps_per_cta;
                size_t locals=local_bits==12?((s.records-groups)*12+7)/8:size_t(s.records)*2;
                s.nominal=2064+groups*4+locals+size_t(s.records)*elapsed_bytes;
                assert(s.nominal<=s.enc_cap);
                s.desc={(uint64_t)s.d_raw,(uint64_t)s.d_enc,s.records,0,s.warps_per_cta};
                CUDA_SAFECALL(cudaMemcpyAsync(s.d_desc,&s.desc,sizeof(s.desc),cudaMemcpyHostToDevice,copy_stream));
                unsigned a=elapsed_bytes,b=local_bits,c=1;
                void *ia[]={&s.d_desc,&s.d_indices,&a,&b};
                assert(cuLaunchKernel(encoder_init,1,1,1,256,1,1,0,(CUstream)copy_stream,ia,nullptr)==CUDA_SUCCESS);
                void *ea[]={&s.d_desc,&c,&s.d_indices,&a,&b};
                unsigned grid=(s.records+1023)/1024;
                assert(cuLaunchKernel(encoder_run,grid,1,1,256,1,1,0,(CUstream)copy_stream,ea,nullptr)==CUDA_SUCCESS);
                CUDA_SAFECALL(cudaMemcpyAsync(s.h_enc,s.d_enc,s.nominal,cudaMemcpyDeviceToHost,copy_stream));
                measured_d2h+=s.nominal;
            }else if(s.attempted&&backend=="nvcomp"){
                NvFrame frame={(uint64_t)s.d_raw,s.raw_bytes};
                s.nv_meta=nv_submit(nv_state,unsigned(active_slot),&frame,1,16384,(uint64_t)s.d_enc,copy_stream);
                assert(s.nv_meta);
            }
            CUDA_SAFECALL(cudaEventRecord(s.ready,copy_stream));
            pthread_mutex_lock(&queue_mutex);ready_queue.push_back(unsigned(active_slot));
            pthread_cond_broadcast(&queue_condition);pthread_mutex_unlock(&queue_mutex);
            seen_target++;active_slot=-1;tool_cuda=false;
        }
        selected_launch=false;
        pthread_mutex_unlock(&nvbit_mutex);
    }
}
