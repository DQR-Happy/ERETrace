// Same recorded input, exact validation, full fresh tmpfs file per iteration.
// This is a completed-data service microbenchmark, not an inference benchmark.
#include "capture_pack.cu"
#include <nvcomp/cascaded.h>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <random>
#include <sys/resource.h>
#include <unistd.h>
#include <sys/mman.h>
#define CK(x) do{auto z=(x);if(z){fprintf(stderr,"CUDA %d line %d\n",int(z),__LINE__);exit(2);}}while(0)
#define NK(x) do{auto z=(x);if(z){fprintf(stderr,"NVCOMP %d line %d\n",int(z),__LINE__);exit(3);}}while(0)
static uint64_t ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static uint64_t us(timeval t){return t.tv_sec*1000000ULL+t.tv_usec;}
__global__ void soa32(const Record*src,uint32_t*dst,uint64_t n){uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n){Record r=src[i];dst[i]=uint32_t(r.start);dst[n+i]=uint32_t(r.start>>32);dst[2*n+i]=r.elapsed;dst[3*n+i]=r.sm;}}
struct Input{unsigned wp;std::vector<Record> rows;Record*dev;uint32_t*soa;};
struct Mode {int codec,soa;size_t chunk;unsigned dw,ew;const char*name;int direct=0;};
int main(int argc,char**argv){if(argc!=4)return 1;std::ifstream f(argv[1],std::ios::binary);uint32_t count;f.read((char*)&count,4);std::vector<Input> ins(count);size_t raw_total=0;
 for(auto&in:ins){uint64_t n;f.read((char*)&n,8);f.read((char*)&in.wp,4);if(!in.wp||n%in.wp)return 15;in.rows.resize(n);f.read((char*)in.rows.data(),n*16);raw_total+=n*16;CK(cudaMalloc(&in.dev,n*16));CK(cudaMalloc(&in.soa,n*16));CK(cudaMemcpy(in.dev,in.rows.data(),n*16,cudaMemcpyHostToDevice));}if(!f)return 4;
 std::vector<Mode> modes={{0,0,0,0,0,"raw16"},{1,0,0,4,4,"gpu9"},{2,0,0,3,2,"flex32"},{4,0,0,3,2,"hier2"},{4,0,0,3,2,"hier2_direct",1},{3,1,65536,0,0,"nv_soa64k"},{3,1,262144,0,0,"nv_soa256k"}};
 printf("{\"raw_bytes\":%zu,\"frames\":%u,\"rows\":[",raw_total,count);bool comma=false;
 // Each mode has independent setup. Seven repetitions plus one warmup; order
 // of modes is shuffled. Repetition-level order rotation handled by caller runs.
 std::mt19937 rng(21042+atoi(argv[3]));std::shuffle(modes.begin(),modes.end(),rng);
 for(auto mode:modes){uint64_t setup=ns();std::vector<void*> sources,outputs,recovered;std::vector<size_t> lengths,caps;std::vector<PackDesc> desc;uint32_t blocks=0;
  nvcompBatchedCascadedCompressOpts_t opts={4096,NVCOMP_TYPE_UINT,2,1,1,{0}};
  for(auto&in:ins){size_t n=in.rows.size();if(mode.codec==3){size_t maxout;NK(nvcompBatchedCascadedCompressGetMaxOutputChunkSize(mode.chunk,opts,&maxout));maxout=(maxout+255)&~size_t(255);
    for(size_t off=0;off<n*16;off+=mode.chunk){sources.push_back((char*)(mode.soa?(void*)in.soa:(void*)in.dev)+off);lengths.push_back(std::min(mode.chunk,n*16-off));caps.push_back(maxout);}}
   else{sources.push_back(in.dev);lengths.push_back(n*16);caps.push_back(mode.codec==4&&in.wp>2?2064+4*(n/in.wp)+n*(2+mode.ew):mode.codec?2064+n*(mode.dw+mode.ew+1):n*16);}}
  size_t pieces=sources.size(),capacity=0;for(size_t n:caps){void*p;CK(cudaMalloc(&p,n));outputs.push_back(p);capacity+=n;}for(size_t n:lengths){void*p;CK(cudaMalloc(&p,n));recovered.push_back(p);}
  void**ds,**dout,**drec;size_t*dl,*dc,*actual;nvcompStatus_t*statuses;unsigned*indices;PackDesc*dd;
  CK(cudaMalloc(&ds,pieces*sizeof(void*)));CK(cudaMalloc(&dout,pieces*sizeof(void*)));CK(cudaMalloc(&drec,pieces*sizeof(void*)));CK(cudaMalloc(&dl,pieces*8));CK(cudaMalloc(&dc,pieces*8));CK(cudaMalloc(&actual,pieces*8));CK(cudaMalloc(&statuses,pieces*sizeof(nvcompStatus_t)));CK(cudaMalloc(&indices,count*1024));CK(cudaMalloc(&dd,count*sizeof(PackDesc)));
  CK(cudaMemcpy(ds,sources.data(),pieces*sizeof(void*),cudaMemcpyHostToDevice));CK(cudaMemcpy(dout,outputs.data(),pieces*sizeof(void*),cudaMemcpyHostToDevice));CK(cudaMemcpy(drec,recovered.data(),pieces*sizeof(void*),cudaMemcpyHostToDevice));CK(cudaMemcpy(dl,lengths.data(),pieces*8,cudaMemcpyHostToDevice));
  if(mode.codec==1||mode.codec==2||mode.codec==4){for(unsigned i=0;i<count;i++){uint64_t n=ins[i].rows.size();unsigned nb=(n+255)/256;desc.push_back({(uint64_t)ins[i].dev,(uint64_t)outputs[i],n,blocks,mode.codec==4?ins[i].wp:nb});blocks+=nb;}CK(cudaMemcpy(dd,desc.data(),count*sizeof(PackDesc),cudaMemcpyHostToDevice));}
  char*host;CK(cudaMallocHost(&host,capacity));std::vector<size_t> sizes= caps;std::vector<nvcompStatus_t> status(pieces);cudaEvent_t a,b,c,d;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));CK(cudaEventCreate(&c));CK(cudaEventCreate(&d));CK(cudaDeviceSynchronize());setup=ns()-setup;
  for(int round=-1;round<7;round++){rusage rb,re;getrusage(RUSAGE_SELF,&rb);uint64_t begin=ns();CK(cudaEventRecord(a));
   if(mode.codec==3){if(mode.soa)for(auto&in:ins)soa32<<<(in.rows.size()+255)/256,256>>>(in.dev,in.soa,in.rows.size());NK(nvcompBatchedCascadedCompressAsync((const void*const*)ds,dl,mode.chunk,pieces,nullptr,0,dout,dc,opts,statuses,0));}
   else if(mode.codec==1){neutrino_capture_delta_init<<<count,256>>>(dd,indices);neutrino_capture_delta_batch<<<blocks,256>>>(dd,count,indices);}
   else if(mode.codec==2){neutrino_capture_flex_init<<<count,256>>>(dd,indices,mode.dw,mode.ew);neutrino_capture_flex_batch<<<blocks,256>>>(dd,count,indices,mode.dw,mode.ew);}
   else if(mode.codec==4){neutrino_capture_hier_init<<<count,256>>>(dd,indices,mode.dw,mode.ew);neutrino_capture_hier_batch<<<blocks,256>>>(dd,count,indices,mode.dw,mode.ew);}
   CK(cudaEventRecord(b));uint64_t size_begin=ns();
   if(mode.codec==3){CK(cudaMemcpy(sizes.data(),dc,pieces*8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(status.data(),statuses,pieces*sizeof(nvcompStatus_t),cudaMemcpyDeviceToHost));for(auto x:status)if(x!=nvcompSuccess)return 5;}
   uint64_t size_ns=ns()-size_begin;size_t expected=0;for(size_t v:sizes)expected+=v;size_t meta_bytes=16+pieces*16;
   std::string path=std::string(argv[2])+"/codec-service.bin";FILE*out=nullptr;void*map=nullptr;char*receive=host;uint64_t receive_setup=ns();
   if(mode.direct){out=fopen(path.c_str(),"w+b");if(!out||ftruncate(fileno(out),meta_bytes+expected))return 11;
    map=mmap(nullptr,meta_bytes+expected,PROT_READ|PROT_WRITE,MAP_SHARED,fileno(out),0);if(map==MAP_FAILED)return 12;
    CK(cudaHostRegister(map,meta_bytes+expected,cudaHostRegisterDefault));receive=(char*)map+meta_bytes;
    uint64_t meta[]={pieces,expected};memcpy(map,meta,16);memcpy((char*)map+16,lengths.data(),pieces*8);memcpy((char*)map+16+pieces*8,sizes.data(),pieces*8);}
   receive_setup=ns()-receive_setup;CK(cudaEventRecord(c));size_t total=0;
   for(size_t i=0;i<pieces;i++){if(sizes[i]>caps[i])return 6;CK(cudaMemcpyAsync(receive+total,mode.codec?outputs[i]:sources[i],sizes[i],cudaMemcpyDeviceToHost));total+=sizes[i];}CK(cudaEventRecord(d));CK(cudaEventSynchronize(d));uint64_t copy_end=ns();
   uint64_t bytes=total+meta_bytes;
   if(mode.direct){CK(cudaHostUnregister(map));if(munmap(map,meta_bytes+total)||fclose(out))return 13;}
   else{out=fopen(path.c_str(),"wb");if(!out)return 7;uint64_t meta[]={pieces,total};if(fwrite(meta,8,2,out)!=2||fwrite(lengths.data(),8,pieces,out)!=pieces||fwrite(sizes.data(),8,pieces,out)!=pieces||fwrite(host,1,total,out)!=total||fclose(out))return 8;}
   uint64_t end=ns();getrusage(RUSAGE_SELF,&re);float comp,copy;CK(cudaEventElapsedTime(&comp,a,b));CK(cudaEventElapsedTime(&copy,c,d));
   if(mode.direct){FILE*r=fopen(path.c_str(),"rb");if(!r||fseek(r,meta_bytes,SEEK_SET)||fread(host,1,total,r)!=total||fclose(r))return 14;}
   bool escaped=false;if(mode.codec==1||mode.codec==2||mode.codec==4){size_t off=0;for(size_t i=0;i<pieces;i++){unsigned flag;memcpy(&flag,host+off,4);escaped|=flag!=0;off+=sizes[i];}}
   // Decompression/oracle is separate from capture timing, reported separately.
   uint64_t decode_begin=ns();bool exact=!escaped;
   if(mode.codec==3){NK(nvcompBatchedCascadedDecompressAsync((const void*const*)dout,dc,dl,actual,pieces,nullptr,0,drec,nvcompBatchedCascadedDecompressDefaultOpts,statuses,0));CK(cudaDeviceSynchronize());CK(cudaMemcpy(status.data(),statuses,pieces*sizeof(nvcompStatus_t),cudaMemcpyDeviceToHost));std::vector<size_t> got(pieces);CK(cudaMemcpy(got.data(),actual,pieces*8,cudaMemcpyDeviceToHost));for(size_t i=0;i<pieces;i++){std::vector<char>x(lengths[i]),y(lengths[i]);CK(cudaMemcpy(x.data(),recovered[i],lengths[i],cudaMemcpyDeviceToHost));CK(cudaMemcpy(y.data(),sources[i],lengths[i],cudaMemcpyDeviceToHost));exact &= status[i]==nvcompSuccess&&got[i]==lengths[i]&&x==y;}}
   else if(mode.codec){size_t off=0;for(size_t j=0;j<pieces;j++){size_t n=ins[j].rows.size();const char*p=host+off;unsigned dw=mode.dw,ew=mode.ew;for(size_t i=0;i<n&&!escaped;i++){Record r{};if(mode.codec==4&&ins[j].wp>2){size_t c=i/ins[j].wp,ctas=n/ins[j].wp;const unsigned char*q=(const unsigned char*)p+2064+4*c;uint32_t bits=q[0]|uint32_t(q[1])<<8|uint32_t(q[2])<<16;int32_t d=int32_t(bits<<8)>>8;r.sm=q[3];uint64_t base;memcpy(&base,p+16+8*r.sm,8);int16_t local;memcpy(&local,p+2064+4*ctas+2*i,2);r.start=base+uint64_t(int64_t(d))+uint64_t(int64_t(local));for(unsigned k=0;k<ew;k++)r.elapsed|=uint32_t((unsigned char)p[2064+4*ctas+2*n+i*ew+k])<<(8*k);exact &= memcmp(&r,&ins[j].rows[i],16)==0;continue;}uint32_t v=0;for(unsigned k=0;k<dw;k++)v|=uint32_t((unsigned char)p[2064+i*dw+k])<<(8*k);int32_t delta=int32_t(v<<(32-8*dw))>>(32-8*dw);r.sm=(unsigned char)p[2064+n*(dw+ew)+i];uint64_t base;memcpy(&base,p+16+8*r.sm,8);r.start=base+uint64_t(int64_t(delta));for(unsigned k=0;k<ew;k++)r.elapsed|=uint32_t((unsigned char)p[2064+n*dw+i*ew+k])<<(8*k);exact &= memcmp(&r,&ins[j].rows[i],16)==0;}off+=sizes[j];}}
   else{size_t off=0;for(size_t i=0;i<count;i++){exact &= memcmp(host+off,ins[i].rows.data(),lengths[i])==0;off+=lengths[i];}}
   uint64_t decode_ns=ns()-decode_begin;FILE*check=fopen(path.c_str(),"rb");if(!check)return 9;if(fseek(check,16+pieces*16,SEEK_SET))return 9;std::vector<char>readback(total);exact &= fread(readback.data(),1,total,check)==total&&memcmp(readback.data(),host,total)==0;fclose(check);unlink(path.c_str());
   if(!exact&&!escaped){fprintf(stderr,"wrong reconstruction %s\n",mode.name);return 10;}
   if(round>=0){printf("%s{\"mode\":\"%s\",\"round\":%d,\"pieces\":%zu,\"bytes\":%llu,\"capacity_bytes\":%zu,\"setup_ns\":%llu,\"compress_gpu_ms\":%.6f,\"copy_gpu_ms\":%.6f,\"receive_setup_ns\":%llu,\"size_status_ns\":%llu,\"copy_complete_ns\":%llu,\"file_ns\":%llu,\"wall_ns\":%llu,\"user_us\":%llu,\"system_us\":%llu,\"decode_oracle_ns\":%llu,\"escaped\":%s,\"exact\":%s}",comma?",":"",mode.name,round,pieces,(unsigned long long)bytes,capacity,(unsigned long long)setup,comp,copy,(unsigned long long)receive_setup,(unsigned long long)size_ns,(unsigned long long)(copy_end-begin),(unsigned long long)(end-copy_end),(unsigned long long)(end-begin),(unsigned long long)(us(re.ru_utime)-us(rb.ru_utime)),(unsigned long long)(us(re.ru_stime)-us(rb.ru_stime)),(unsigned long long)decode_ns,escaped?"true":"false",exact?"true":"false");comma=true;fflush(stdout);}
  }
  CK(cudaFreeHost(host));for(void*p:outputs)CK(cudaFree(p));for(void*p:recovered)CK(cudaFree(p));CK(cudaFree(ds));CK(cudaFree(dout));CK(cudaFree(drec));CK(cudaFree(dl));CK(cudaFree(dc));CK(cudaFree(actual));CK(cudaFree(statuses));CK(cudaFree(indices));CK(cudaFree(dd));CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));CK(cudaEventDestroy(c));CK(cudaEventDestroy(d));
 }printf("]}\n");for(auto&in:ins){CK(cudaFree(in.dev));CK(cudaFree(in.soa));}return 0;}
