#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <stdint.h>
#include <time.h>
#include <vector>
#include <cstring>
struct Rec {unsigned long long v[4];};
struct Binding {uint64_t index,offset,bytes,value;};
struct Engine {CUfunction expected_fn=nullptr;unsigned grid=0,block=0;std::vector<Binding> bindings;unsigned n;size_t slots,tmp_bytes;int compact;unsigned long long *sizes,*offsets,*caps,*src_offsets,*meta;Rec *out;void *tmp;void *host;unsigned long long *host_counts,*host_meta;CUevent events[4];};
struct Result {uint64_t total_ns,kernel_wait_ns,compact_ns,metadata_ns,copy_ns,bytes,events;float kernel_ms,scan_ms,gather_ms;uint64_t overflow;};
static uint64_t now(){timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (uint64_t)t.tv_sec*1000000000+t.tv_nsec;}
__global__ void clamp_counts(const unsigned long long*c,const unsigned long long*cap,unsigned long long*s,unsigned long long*meta,unsigned n){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){s[i]=min(c[i],cap[i]);if(c[i]>cap[i])atomicExch(meta+1,1ULL);}if(i==n)s[n]=0;}
__global__ void gather_records(const Rec*in,Rec*out,const unsigned long long*src,const unsigned long long*dst,const unsigned long long*s,unsigned long long*meta,unsigned n){unsigned t=(blockIdx.x*blockDim.x+threadIdx.x)/32,l=threadIdx.x%32;if(t<n){for(unsigned long long j=l;j<s[t]*4;j+=32)((unsigned long long*)out)[dst[t]*4+j]=((const unsigned long long*)in)[src[t]*4+j];}if(blockIdx.x==0&&threadIdx.x==0)meta[0]=dst[n];}
#define CK(x) do{int rc=(int)(x);if(rc)return rc;}while(0)
extern "C" int cert_create(unsigned n,size_t slots,int compact,const unsigned long long*offsets,const unsigned long long*caps,void**ret){
 Engine*e=new Engine{};e->n=n;e->slots=slots;e->compact=compact;
 if(compact){
 CK(cudaMalloc(&e->sizes,(n+1)*8));CK(cudaMalloc(&e->offsets,(n+1)*8));CK(cudaMalloc(&e->src_offsets,n*8));CK(cudaMalloc(&e->caps,n*8));CK(cudaMalloc(&e->meta,16));CK(cudaMalloc(&e->out,slots*32+128));
 CK(cudaMemcpy(e->src_offsets,offsets,n*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(e->caps,caps,n*8,cudaMemcpyHostToDevice));
 CK(cub::DeviceScan::ExclusiveSum(nullptr,e->tmp_bytes,e->sizes,e->offsets,n+1));CK(cudaMalloc(&e->tmp,e->tmp_bytes));}
 CK(cudaHostAlloc(&e->host,slots*32+128,cudaHostAllocDefault));CK(cudaHostAlloc(&e->host_counts,n*8,cudaHostAllocDefault));CK(cudaHostAlloc(&e->host_meta,16,cudaHostAllocDefault));
 for(auto &ev:e->events)CK(cuEventCreate(&ev,0));*ret=e;return 0;
}
extern "C" int cert_bind(void*p,CUfunction f,unsigned grid,unsigned block,const Binding*b,unsigned n){for(unsigned i=0;i<n;i++)if(b[i].bytes!=1&&b[i].bytes!=2&&b[i].bytes!=4&&b[i].bytes!=8)return -12;Engine*e=(Engine*)p;e->expected_fn=f;e->grid=grid;e->block=block;e->bindings.assign(b,b+n);return 0;}
extern "C" int cert_run(void*ptr,CUfunction f,unsigned grid,unsigned block,void**args,CUdeviceptr records,CUdeviceptr counts,unsigned shared,int mode,int diagnostic,Result*r){
 Engine*e=(Engine*)ptr;*r={};uint64_t start=now();if(e->expected_fn){if(f!=e->expected_fn||grid!=e->grid||block!=e->block)return -10;for(const auto&b:e->bindings){uint64_t v=0;memcpy(&v,(char*)args[b.index]+b.offset,b.bytes);if(v!=b.value)return -11;}}if(diagnostic)CK(cuEventRecord(e->events[0],0));
 CK(cuLaunchKernel(f,grid,1,1,block,1,1,shared,0,args,0));if(diagnostic)CK(cuEventRecord(e->events[1],0));
 CK(cuCtxSynchronize());uint64_t after_model=now();r->kernel_wait_ns=after_model-start;
 if(mode==2){
  CK(cudaMemset(e->meta,0,16));clamp_counts<<<(e->n+256)/256,256>>>((unsigned long long*)counts,e->caps,e->sizes,e->meta,e->n);CK(cudaGetLastError());
  CK(cub::DeviceScan::ExclusiveSum(e->tmp,e->tmp_bytes,e->sizes,e->offsets,e->n+1));if(diagnostic)CK(cuEventRecord(e->events[2],0));
  gather_records<<<(e->n+7)/8,256>>>((Rec*)records,e->out,e->src_offsets,e->offsets,e->sizes,e->meta,e->n);CK(cudaGetLastError());if(diagnostic)CK(cuEventRecord(e->events[3],0));
  uint64_t before_meta=now();r->compact_ns=before_meta-after_model;CK(cudaMemcpy(e->host_meta,e->meta,16,cudaMemcpyDeviceToHost));r->metadata_ns=now()-before_meta;
  r->events=e->host_meta[0];r->overflow=e->host_meta[1];if(r->events>e->slots)return -9;
  uint64_t cp=now();CK(cudaMemcpy(e->host,e->out,r->events*32,cudaMemcpyDeviceToHost));CK(cuMemcpyDtoH(e->host_counts,counts,e->n*8));r->copy_ns=now()-cp;r->bytes=r->events*32+e->n*8+16;
 }else if(mode==1){uint64_t cp=now();CK(cuMemcpyDtoH(e->host,records,e->slots*32));CK(cuMemcpyDtoH(e->host_counts,counts,e->n*8));r->copy_ns=now()-cp;r->bytes=e->slots*32+e->n*8;}
 r->total_ns=now()-start;
 if(diagnostic){CK(cuEventElapsedTime(&r->kernel_ms,e->events[0],e->events[1]));if(mode==2){CK(cuEventElapsedTime(&r->scan_ms,e->events[1],e->events[2]));CK(cuEventElapsedTime(&r->gather_ms,e->events[2],e->events[3]));}}
 return 0;
}
extern "C" void*cert_host(void*p){return ((Engine*)p)->host;}
extern "C" void*cert_counts(void*p){return ((Engine*)p)->host_counts;}
extern "C" size_t cert_space(void*p){Engine*e=(Engine*)p;return e->compact?((e->n+1)*16+e->n*16+16+e->slots*32+128+e->tmp_bytes):0;}
extern "C" void cert_destroy(void*p){Engine*e=(Engine*)p;cudaFree(e->sizes);cudaFree(e->offsets);cudaFree(e->caps);cudaFree(e->src_offsets);cudaFree(e->meta);cudaFree(e->out);cudaFree(e->tmp);cudaFreeHost(e->host);cudaFreeHost(e->host_counts);cudaFreeHost(e->host_meta);for(auto ev:e->events)cuEventDestroy(ev);delete e;}

extern "C" size_t cert_workspace(void*p){return ((Engine*)p)->tmp_bytes;}
// The graph is only a microbenchmark harness. Complete one-shot measurements above remain Graph off.
extern "C" int cert_gpu_repeat(CUfunction f,unsigned grid,unsigned block,unsigned shared,void**args,unsigned iterations,float*us){
 cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t exec;cudaEvent_t first,last;
 CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));CK(cudaEventCreate(&first));CK(cudaEventCreate(&last));
 CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));CK(cudaEventRecordWithFlags(first,stream,cudaEventRecordExternal));
 for(unsigned i=0;i<iterations;++i)CK(cuLaunchKernel(f,grid,1,1,block,1,1,shared,(CUstream)stream,args,nullptr));
 CK(cudaEventRecordWithFlags(last,stream,cudaEventRecordExternal));CK(cudaStreamEndCapture(stream,&graph));CK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
 for(unsigned i=0;i<3;++i){CK(cudaGraphLaunch(exec,stream));CK(cudaStreamSynchronize(stream));}
 CK(cudaGraphLaunch(exec,stream));CK(cudaStreamSynchronize(stream));float ms;CK(cudaEventElapsedTime(&ms,first,last));*us=ms*1000/iterations;
 CK(cudaGraphExecDestroy(exec));CK(cudaGraphDestroy(graph));CK(cudaEventDestroy(first));CK(cudaEventDestroy(last));CK(cudaStreamDestroy(stream));return 0;
}
