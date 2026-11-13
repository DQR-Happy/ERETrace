// Optional nvCOMP Cascaded adapter. All GPU work uses the supplied copy stream.
// Input records are immutable; variable lengths return to the writer, never the
// producer. The writer owns the segment until payload D2H and file output finish.
#include <nvcomp/cascaded.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#define CK(x) do{auto rc=(x);if(rc){fprintf(stderr,"nvpipe CUDA %d line %d\n",int(rc),__LINE__);abort();}}while(0)
#define NK(x) do{auto rc=(x);if(rc){fprintf(stderr,"nvpipe nvCOMP %d line %d\n",int(rc),__LINE__);abort();}}while(0)
struct Frame {uint64_t src,bytes;};
struct Piece {uint64_t source,bytes;unsigned frame,index,ordinal,chunks;int encoded;};
struct SoaFrame {uint64_t src,dest,n;unsigned first,blocks;};
struct Slot {Piece*pieces=nullptr;SoaFrame*frames=nullptr;const void**inputs=nullptr;size_t*lengths=nullptr;uint64_t*meta=nullptr;};
struct Context {unsigned maxpieces,maxframes,chunk,mode;size_t maxout,device_bytes=0,host_bytes=0;unsigned slots=0;
 unsigned char*scratch,*soa;void**outputs;const void**inputs;size_t*lengths,*actual;nvcompStatus_t*status;Piece*pieces;SoaFrame*frames;uint64_t*positions,*meta;Slot slot[512];
 nvcompBatchedCascadedCompressOpts_t opts={4096,NVCOMP_TYPE_UINT,2,1,1,{0}};};
template<class T>static void device_alloc(Context*c,T**p,size_t n){CK(cudaMalloc((void**)p,n));c->device_bytes+=n;}
template<class T>static void host_alloc(Context*c,T**p,size_t n){CK(cudaMallocHost((void**)p,n));c->host_bytes+=n;}
__global__ void to_soa(const SoaFrame*frames,unsigned count){unsigned lo=0,hi=count;while(lo+1<hi){unsigned m=(lo+hi)/2;if(frames[m].first<=blockIdx.x)lo=m;else hi=m;}SoaFrame f=frames[lo];uint64_t i=uint64_t(blockIdx.x-f.first)*256+threadIdx.x;if(i>=f.n)return;auto src=(const uint32_t*)f.src;auto dst=(uint32_t*)f.dest;for(unsigned j=0;j<4;j++)dst[j*f.n+i]=src[4*i+j];}
__global__ void make_layout(const Piece*p,unsigned np,const size_t*sizes,const nvcompStatus_t*status,uint64_t*positions,uint64_t*meta,unsigned char*out){
 if(threadIdx.x||blockIdx.x)return;uint64_t pos=0;unsigned last=~0U;uint64_t start=0;
 for(unsigned i=0;i<np;i++){Piece q=p[i];if(q.frame!=last){if(last!=~0U)meta[last+1]=pos-start;last=q.frame;start=pos;
   if(q.encoded){uint32_t*head=(uint32_t*)(out+pos);head[0]=q.chunks;head[1]=0x4e564331;head[2]=q.bytes;head[3]=0;pos+=16+8*q.chunks;}}
  size_t len=q.encoded?sizes[q.index]:q.bytes;if(q.encoded&&status[q.index]!=nvcompSuccess){meta[0]=~uint64_t(0);return;}
  positions[i]=pos;if(q.encoded)((uint64_t*)(out+start+16))[q.ordinal]=len;pos+=q.encoded?((len+7)&~size_t(7)):len;
 }
 if(last!=~0U)meta[last+1]=pos-start;meta[0]=pos;
}
__global__ void gather_pieces(const Piece*p,const uint64_t*positions,const size_t*sizes,const unsigned char*scratch,size_t cap,unsigned char*out,const uint64_t*meta){if(meta[0]==~uint64_t(0))return;Piece q=p[blockIdx.x];size_t n=q.encoded?sizes[q.index]:q.bytes;const unsigned char*src=q.encoded?scratch+q.index*cap:(const unsigned char*)q.source;unsigned char*dst=out+positions[blockIdx.x];for(size_t i=threadIdx.x;i<n/4;i+=blockDim.x)((uint32_t*)dst)[i]=((const uint32_t*)src)[i];for(size_t i=n/4*4+threadIdx.x;i<(q.encoded?((n+7)&~size_t(7)):n);i+=blockDim.x)dst[i]=i<n?src[i]:0;}
extern "C" void* nvpipe_create(size_t raw_capacity,unsigned maxframes,unsigned chunk,unsigned mode){auto*c=new Context;c->maxframes=maxframes;c->chunk=chunk;c->mode=mode;c->maxpieces=raw_capacity/chunk+maxframes+4;
 NK(nvcompBatchedCascadedCompressGetMaxOutputChunkSize(chunk,c->opts,&c->maxout));c->maxout=(c->maxout+255)&~size_t(255);
 device_alloc(c,&c->scratch,c->maxpieces*c->maxout);device_alloc(c,&c->soa,mode==2?raw_capacity:4);
 device_alloc(c,&c->inputs,c->maxpieces*sizeof(void*));device_alloc(c,&c->outputs,c->maxpieces*sizeof(void*));device_alloc(c,&c->lengths,c->maxpieces*sizeof(size_t));device_alloc(c,&c->actual,c->maxpieces*sizeof(size_t));device_alloc(c,&c->status,c->maxpieces*sizeof(nvcompStatus_t));device_alloc(c,&c->pieces,c->maxpieces*sizeof(Piece));device_alloc(c,&c->frames,maxframes*sizeof(SoaFrame));device_alloc(c,&c->positions,c->maxpieces*8);device_alloc(c,&c->meta,(maxframes+1)*8);
 std::vector<void*>ptr(c->maxpieces);for(unsigned i=0;i<c->maxpieces;i++)ptr[i]=c->scratch+i*c->maxout;CK(cudaMemcpy(c->outputs,ptr.data(),ptr.size()*sizeof(void*),cudaMemcpyHostToDevice));return c;}
extern "C" size_t nvpipe_bound(void*state,size_t bytes,size_t threshold){auto*c=(Context*)state;if(bytes<threshold)return bytes;size_t full=bytes/c->chunk,tail=bytes%c->chunk,last=0;if(tail)NK(nvcompBatchedCascadedCompressGetMaxOutputChunkSize(tail,c->opts,&last));return 16+(full+(tail!=0))*8+full*c->maxout+((last+7)&~size_t(7));}
extern "C" const uint64_t* nvpipe_submit(void*state,unsigned slot,const Frame*f,unsigned count,size_t threshold,uint64_t dest,void*stream){auto*c=(Context*)state;auto&s=c->slot[slot];cudaStream_t st=(cudaStream_t)stream;
 if(!s.pieces){host_alloc(c,&s.pieces,c->maxpieces*sizeof(Piece));host_alloc(c,&s.frames,c->maxframes*sizeof(SoaFrame));host_alloc(c,&s.inputs,c->maxpieces*sizeof(void*));host_alloc(c,&s.lengths,c->maxpieces*sizeof(size_t));host_alloc(c,&s.meta,(c->maxframes+1)*8);c->slots++;}
 unsigned np=0,ne=0,nb=0,nf=0;size_t soaoff=0;
 for(unsigned j=0;j<count;j++){bool enc=f[j].bytes>=threshold;uint64_t src=f[j].src;if(enc&&c->mode==2){unsigned b=(f[j].bytes/16+255)/256;s.frames[nf++]={src,(uint64_t)(c->soa+soaoff),f[j].bytes/16,nb,b};nb+=b;src=(uint64_t)(c->soa+soaoff);}unsigned chunks=(f[j].bytes+c->chunk-1)/c->chunk;
  for(unsigned k=0;k<chunks;k++){if(np>=c->maxpieces)abort();size_t n=std::min(size_t(c->chunk),size_t(f[j].bytes)-size_t(k)*c->chunk);uint64_t p=src+size_t(k)*c->chunk;
   s.pieces[np++]={p,n,j,ne,k,chunks,enc};if(enc){s.inputs[ne]=(void*)p;s.lengths[ne++]=n;}}
  soaoff+=f[j].bytes;
 }
 if(nf){CK(cudaMemcpyAsync(c->frames,s.frames,nf*sizeof(SoaFrame),cudaMemcpyHostToDevice,st));to_soa<<<nb,256,0,st>>>(c->frames,nf);}
 CK(cudaMemcpyAsync(c->pieces,s.pieces,np*sizeof(Piece),cudaMemcpyHostToDevice,st));
 if(ne){CK(cudaMemcpyAsync(c->inputs,s.inputs,ne*sizeof(void*),cudaMemcpyHostToDevice,st));CK(cudaMemcpyAsync(c->lengths,s.lengths,ne*sizeof(size_t),cudaMemcpyHostToDevice,st));NK(nvcompBatchedCascadedCompressAsync(c->inputs,c->lengths,c->chunk,ne,nullptr,0,c->outputs,c->actual,c->opts,c->status,st));}
 make_layout<<<1,1,0,st>>>(c->pieces,np,c->actual,c->status,c->positions,c->meta,(unsigned char*)dest);gather_pieces<<<np,256,0,st>>>(c->pieces,c->positions,c->actual,c->scratch,c->maxout,(unsigned char*)dest,c->meta);CK(cudaGetLastError());CK(cudaMemcpyAsync(s.meta,c->meta,(count+1)*8,cudaMemcpyDeviceToHost,st));return s.meta;}
extern "C" void nvpipe_space(void*state,uint64_t*out){auto*c=(Context*)state;out[0]=c->device_bytes;out[1]=c->host_bytes;out[2]=c->slots;out[3]=c->maxout;out[4]=c->maxpieces;}
// Independent decoder used only after completion, outside performance windows.

struct DecodeWorkspace {
 unsigned char *input=nullptr,*output=nullptr;void **di=nullptr,**doo=nullptr;
 size_t *ds=nullptr,*dl=nullptr,*da=nullptr;nvcompStatus_t *status=nullptr;
 size_t input_cap=0,output_cap=0;unsigned count_cap=0;std::vector<char> host;
};
static thread_local DecodeWorkspace decoder_workspace;
extern "C" int nvpipe_decode(const void*data,size_t bytes,void*output,size_t rawbytes,unsigned mode,unsigned chunk){
 if(bytes<16)return 1;auto*h=(const uint32_t*)data;unsigned n=h[0];chunk=h[2];
 if(h[1]!=0x4e564331||!n||!chunk||bytes<16+8*n||n!=(rawbytes+chunk-1)/chunk)return 2;
 auto&w=decoder_workspace;
 if(w.input_cap<bytes){if(w.input)CK(cudaFree(w.input));CK(cudaMalloc(&w.input,bytes));w.input_cap=bytes;}
 if(w.output_cap<rawbytes){if(w.output)CK(cudaFree(w.output));CK(cudaMalloc(&w.output,rawbytes));w.output_cap=rawbytes;}
 if(w.count_cap<n){
  for(void*p:{(void*)w.di,(void*)w.doo,(void*)w.ds,(void*)w.dl,(void*)w.da,(void*)w.status})if(p)CK(cudaFree(p));
  CK(cudaMalloc(&w.di,n*sizeof(void*)));CK(cudaMalloc(&w.doo,n*sizeof(void*)));
  CK(cudaMalloc(&w.ds,n*sizeof(size_t)));CK(cudaMalloc(&w.dl,n*sizeof(size_t)));CK(cudaMalloc(&w.da,n*sizeof(size_t)));CK(cudaMalloc(&w.status,n*sizeof(nvcompStatus_t)));w.count_cap=n;
 }
 std::vector<size_t>sizes(n),lengths(n),got(n);std::vector<void*>input(n),out(n);
 memcpy(sizes.data(),(char*)data+16,n*8);size_t offset=16+n*8;
 for(unsigned i=0;i<n;i++){
  lengths[i]=std::min(size_t(chunk),rawbytes-size_t(i)*chunk);
  if(!sizes[i]||offset+sizes[i]>bytes)return 3;
  input[i]=w.input+offset;out[i]=w.output+size_t(i)*chunk;offset+=(sizes[i]+7)&~size_t(7);
 }
 if(offset!=bytes)return 4;
 CK(cudaMemcpy(w.input,data,bytes,cudaMemcpyHostToDevice));CK(cudaMemcpy(w.di,input.data(),n*sizeof(void*),cudaMemcpyHostToDevice));
 CK(cudaMemcpy(w.doo,out.data(),n*sizeof(void*),cudaMemcpyHostToDevice));CK(cudaMemcpy(w.ds,sizes.data(),n*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(w.dl,lengths.data(),n*8,cudaMemcpyHostToDevice));
 NK(nvcompBatchedCascadedDecompressAsync((const void*const*)w.di,w.ds,w.dl,w.da,n,nullptr,0,w.doo,nvcompBatchedCascadedDecompressDefaultOpts,w.status,0));
 CK(cudaDeviceSynchronize());std::vector<nvcompStatus_t>sts(n);CK(cudaMemcpy(sts.data(),w.status,n*sizeof(nvcompStatus_t),cudaMemcpyDeviceToHost));CK(cudaMemcpy(got.data(),w.da,n*8,cudaMemcpyDeviceToHost));
 for(unsigned i=0;i<n;i++)if(sts[i]!=nvcompSuccess||got[i]!=lengths[i])return 5;
 w.host.resize(rawbytes);CK(cudaMemcpy(w.host.data(),w.output,rawbytes,cudaMemcpyDeviceToHost));
 if(mode==2){auto*s=(uint32_t*)w.host.data(),*d=(uint32_t*)output;size_t records=rawbytes/16;
  for(size_t i=0;i<records;i++)for(unsigned j=0;j<4;j++)d[4*i+j]=s[j*records+i];
 }else memcpy(output,w.host.data(),rawbytes);
 return 0;
}
extern "C" void nvpipe_destroy(void*state){auto*c=(Context*)state;for(auto&s:c->slot)if(s.pieces){CK(cudaFreeHost(s.pieces));CK(cudaFreeHost(s.frames));CK(cudaFreeHost(s.inputs));CK(cudaFreeHost(s.lengths));CK(cudaFreeHost(s.meta));}for(void*p:{(void*)c->scratch,(void*)c->soa,(void*)c->inputs,(void*)c->outputs,(void*)c->lengths,(void*)c->actual,(void*)c->status,(void*)c->pieces,(void*)c->frames,(void*)c->positions,(void*)c->meta})CK(cudaFree(p));delete c;}
