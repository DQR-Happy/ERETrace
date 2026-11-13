// XTrace full-record collector. Same producer/order/tails in both arms.
// GPU checked CTA/SM references, explicit raw recovery, asynchronous D2H/writer.
#include <cuda_runtime.h>
#include <openssl/sha.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <string>
#define CK(x) do { auto e=(x); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while(0)
constexpr uint64_t REF_UNIT=32; // Exact difference unit; nonaligned values escape.
constexpr unsigned MAXN=(16*1024*1024)/24, TABLE=1u<<18, SMS=256;
constexpr size_t OUTCAP=32ul<<20;
struct Record { uint64_t t,grid; uint32_t warp; uint16_t sm; uint8_t type,kid; };
static_assert(sizeof(Record)==24,"raw schema");
struct Node { unsigned state,index,cta,sm; uint64_t grid,stamp; };
struct Group { uint64_t grid,stamp; unsigned cta,sm; };
struct Header { uint64_t magic; uint32_t n,k,recover,format; uint64_t bytes,first; uint64_t local_ok,anchor_ok,reserved; };
static_assert(sizeof(Header)==64,"frame header");
constexpr uint64_t MAGIC=0x3147505552455858ull;
struct Dev { Node* table; Group* groups; unsigned *count,*ids,*width; uint64_t *refs; };
__device__ unsigned hashkey(uint64_t grid,unsigned cta) { uint64_t x=grid^(uint64_t(cta)*0x9e3779b97f4a7c15ull); x^=x>>33; x*=0xff51afd7ed558ccdull; x^=x>>33; return unsigned(x)&(TABLE-1); }
__global__ void init_table(Node* table) {
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=TABLE)return;
 table[i]={0,0,0,0xffffffffu,0,~uint64_t(0)};
}
__device__ Record read_record(const Record* p) {
 Record r;
 asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(r.t) : "l"(p));
 asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(r.grid) : "l"((const char*)p+8));
 asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(r.warp) : "l"((const char*)p+16));
 unsigned tail;asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(tail) : "l"((const char*)p+20));
 r.sm=tail&65535u;r.type=(tail>>16)&255u;r.kid=tail>>24;return r;
}
__global__ void group_records(const Record* raw,unsigned n,Dev d) {
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 Record r=read_record(raw+i);unsigned w=d.width[r.kid];
 if(!w||r.sm>=SMS||!r.t||!r.grid||r.grid>0xffffffffull||(r.type!=1&&r.type!=2)){atomicOr(d.count+2,1u);return;}
 unsigned cta=r.warp/w,slot=hashkey(r.grid,cta);uint64_t key=(r.grid<<32)|cta;
 for(unsigned tries=0;tries<64;tries++,slot=(slot+1)&(TABLE-1)) {
  Node*p=d.table+slot;uint64_t old=atomicCAS((unsigned long long*)&p->grid,0ull,(unsigned long long)key);
  if(old==0||old==key){unsigned sm=atomicCAS(&p->sm,0xffffffffu,r.sm);if(sm!=0xffffffffu&&sm!=r.sm)atomicOr(d.count+2,2u);
   atomicMin((unsigned long long*)&p->stamp,(unsigned long long)r.t);d.ids[i]=slot;return;}
 }
 atomicOr(d.count+2,4u);
}
__global__ void collect_groups(Dev d) {
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=TABLE)return;
 Node*p=d.table+i;if(!p->grid)return;unsigned g=atomicAdd(d.count,1u);p->index=g;
 d.groups[g]={p->grid>>32,p->stamp,(unsigned)p->grid,p->sm};
 atomicMin((unsigned long long*)(d.refs+p->sm),(unsigned long long)p->stamp);
}
__device__ void store_bytes(unsigned char* p,uint64_t x,unsigned bytes){for(unsigned j=0;j<bytes;j++)p[j]=(unsigned char)(x>>(j*8));}
__device__ void recover_stamp(unsigned char* out,Dev d,unsigned index,unsigned kind,uint64_t stamp,unsigned n,unsigned k) {
 unsigned q=atomicAdd(d.count+1,1u);size_t off=64+SMS*8+size_t(k)*18+size_t(n)*18+size_t(q)*16;
 if(off+16<=OUTCAP){store_bytes(out+off,index,4);store_bytes(out+off+4,kind,4);store_bytes(out+off+8,stamp,8);}
}
__global__ void pack_groups(unsigned char* out,Dev d,unsigned n) {
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;unsigned k=*d.count;if(d.count[2])return;
 if(i<SMS)store_bytes(out+64+i*8,d.refs[i],8);
 if(i>=k)return;Group g=d.groups[i];uint64_t delta=g.stamp-d.refs[g.sm];bool fits=delta<=uint64_t(0xffffffu)*REF_UNIT && delta%REF_UNIT==0;
 size_t a=64+SMS*8;
 store_bytes(out+a+i*8,g.grid,8);store_bytes(out+a+k*8+i*4,g.cta,4);
 store_bytes(out+a+k*12+i*2,g.sm,2);out[a+k*14+i]=!fits;
 store_bytes(out+a+k*15+i*3,delta/REF_UNIT,3);
 if(!fits)recover_stamp(out,d,i,1,g.stamp,n,k);else atomicAdd(d.count+4,1u);
}
__global__ void pack_records(const Record* raw,unsigned char* out,Dev d,unsigned n) {
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||d.count[2])return;
 Record r=read_record(raw+i);Node p=d.table[d.ids[i]];unsigned k=*d.count;
 uint64_t delta=r.t-p.stamp;bool fits=delta<=uint64_t(65534u)*REF_UNIT && delta%REF_UNIT==0;
 size_t a=64+SMS*8+size_t(k)*18;
 // All 16 non-timestamp bytes are copied verbatim, in original append order.
 store_bytes(out+a+size_t(i)*16,r.grid,8);store_bytes(out+a+size_t(i)*16+8,r.warp,4);
 store_bytes(out+a+size_t(i)*16+12,r.sm,2);out[a+size_t(i)*16+14]=r.type;out[a+size_t(i)*16+15]=r.kid;
 store_bytes(out+a+size_t(n)*16+size_t(i)*2,fits?delta/REF_UNIT:65535u,2);
 if(!fits)recover_stamp(out,d,i,0,r.t,n,k);else atomicAdd(d.count+3,1u);
}
__global__ void finish_frame(unsigned char* out,Dev d,unsigned n,uint64_t first) {
 if(threadIdx.x)return;Header h{};h.magic=MAGIC;h.n=n;h.k=d.count[0];h.recover=d.count[1];h.first=first;
 h.bytes=64+SMS*8+uint64_t(h.k)*18+uint64_t(n)*18+uint64_t(h.recover)*16;
 h.format=(d.count[2]||h.bytes>=uint64_t(n)*24+64||h.bytes>OUTCAP)?0:1;
 h.reserved=REF_UNIT;h.local_ok=d.count[3];h.anchor_ok=d.count[4];*((Header*)out)=h;
}
static uint64_t load(const unsigned char* p,unsigned n){uint64_t x=0;for(unsigned j=0;j<n;j++)x|=uint64_t(p[j])<<(8*j);return x;}
struct Key {uint64_t grid;unsigned cta;bool operator==(const Key&o)const{return grid==o.grid&&cta==o.cta;}};
struct Hash {size_t operator()(const Key&k)const{return size_t(k.grid^(uint64_t(k.cta)*0x9e3779b97f4a7c15ull));}};
static bool decode(const Header&h,const unsigned char*p,const unsigned*width,std::vector<unsigned char>&raw) {
 raw.resize(size_t(h.n)*24);
 if(!h.format){memcpy(raw.data(),p,raw.size());return h.bytes==raw.size()+64;}
 if(h.bytes!=64+SMS*8+uint64_t(h.k)*18+uint64_t(h.n)*18+uint64_t(h.recover)*16)return false;
 uint64_t unit=h.reserved?h.reserved:1;if(unit>1000000)return false;
 const unsigned char*a=p+SMS*8,*tail=a+size_t(h.k)*18,*delta=tail+size_t(h.n)*16,*rec=delta+size_t(h.n)*2;
 std::vector<uint64_t> anchors(h.k),escaped(h.n,0);std::unordered_map<Key,unsigned,Hash> map;map.reserve(h.k*2);
 for(unsigned i=0;i<h.k;i++) {
  uint64_t grid=load(a+i*8,8);unsigned cta=load(a+h.k*8+i*4,4),sm=load(a+h.k*12+i*2,2);
  if(sm>=SMS||!map.emplace(Key{grid,cta},i).second)return false;
  anchors[i]=load(p+sm*8,8)+load(a+h.k*15+i*3,3)*unit;
 }
 for(unsigned j=0;j<h.recover;j++){unsigned i=load(rec+j*16,4),kind=load(rec+j*16+4,4);uint64_t stamp=load(rec+j*16+8,8);
  if(kind==1){if(i>=h.k||!a[h.k*14+i])return false;anchors[i]=stamp;}else {if(kind||i>=h.n||escaped[i])return false;escaped[i]=stamp;}}
 for(unsigned i=0;i<h.n;i++) {
  const unsigned char*t=tail+size_t(i)*16;unsigned kid=t[15],warp=load(t+8,4);if(!width[kid])return false;
  auto it=map.find(Key{load(t,8),warp/width[kid]});if(it==map.end())return false;
  unsigned d=load(delta+i*2,2);uint64_t stamp=d==65535?escaped[i]:anchors[it->second]+uint64_t(d)*unit;if(!stamp)return false;
  unsigned char*o=raw.data()+size_t(i)*24;memcpy(o,&stamp,8);memcpy(o+8,t,16);
 }return true;
}
struct Slot {Record*snapshot;unsigned char*d,*h,*oracle;cudaEvent_t ready;bool busy=false;Header head{};};
struct Collector {
 bool ere,verify;Dev dev{};cudaStream_t enc,copy,count_stream;Slot s[2];
 unsigned width[256]{};FILE*fp=nullptr;std::thread writer;std::mutex mu;std::condition_variable cv;
 std::deque<unsigned> q;bool stop=false;std::string error;uint64_t submitted=0,records=0,bytes=0,frames=0,encoded=0,raw_frames=0,recover=0,groups=0;
 SHA256_CTX raw_hash,decoded_hash;unsigned char raw_digest[32]{},decoded_digest[32]{};uint64_t scratch=0;
 Collector(bool e,bool v,const unsigned*w):ere(e),verify(v){
  memcpy(width,w,sizeof(width));CK(cudaStreamCreateWithFlags(&enc,cudaStreamNonBlocking));CK(cudaStreamCreateWithFlags(&copy,cudaStreamNonBlocking));CK(cudaStreamCreateWithFlags(&count_stream,cudaStreamNonBlocking));
  // Equal 64 MiB output and 128 MiB pinned-host arenas in both arms.
  for(auto&x:s){CK(cudaMalloc((void**)&x.snapshot,MAXN*24ul));CK(cudaMalloc((void**)&x.d,OUTCAP));CK(cudaHostAlloc((void**)&x.h,OUTCAP*2,cudaHostAllocDefault));x.oracle=x.h+OUTCAP;CK(cudaEventCreateWithFlags(&x.ready,cudaEventDisableTiming));}
  if(ere){CK(cudaMalloc((void**)&dev.table,TABLE*sizeof(Node)));CK(cudaMalloc((void**)&dev.groups,MAXN*sizeof(Group)));CK(cudaMalloc((void**)&dev.ids,MAXN*4));CK(cudaMalloc((void**)&dev.refs,SMS*8));CK(cudaMalloc((void**)&dev.count,5*4));CK(cudaMalloc((void**)&dev.width,256*4));CK(cudaMemcpy(dev.width,width,256*4,cudaMemcpyHostToDevice));
   scratch=TABLE*sizeof(Node)+MAXN*(sizeof(Group)+4)+SMS*8+5*4+256*4;
  }
  writer=std::thread([this]{loop();});
 }
 void loop(){try{for(;;){unsigned i;{std::unique_lock<std::mutex>lk(mu);cv.wait(lk,[&]{return stop||!q.empty();});if(q.empty()&&stop)return;i=q.front();q.pop_front();}
   Slot&x=s[i];CK(cudaEventSynchronize(x.ready));
   if(verify){std::vector<unsigned char> rebuilt;if(!decode(x.head,x.h,width,rebuilt)||memcmp(rebuilt.data(),x.oracle,rebuilt.size()))throw std::runtime_error("same-execution bytewise recovery failed");SHA256_Update(&raw_hash,x.oracle,rebuilt.size());SHA256_Update(&decoded_hash,rebuilt.data(),rebuilt.size());}
   if(fwrite(&x.head,1,64,fp)!=64||fwrite(x.h,1,x.head.bytes-64,fp)!=x.head.bytes-64)throw std::runtime_error("trace write failed");
   records+=x.head.n;bytes+=x.head.bytes;frames++;encoded+=x.head.format;raw_frames+=!x.head.format;recover+=x.head.recover;groups+=x.head.k;
   {std::lock_guard<std::mutex>lk(mu);x.busy=false;}cv.notify_all();
  }}catch(const std::exception&e){{std::lock_guard<std::mutex>lk(mu);error=e.what();for(auto&x:s)x.busy=false;}cv.notify_all();}}
 void drain(){std::unique_lock<std::mutex>lk(mu);cv.wait(lk,[&]{return !error.empty()||(!s[0].busy&&!s[1].busy);});if(!error.empty())throw std::runtime_error(error);}
 void begin(const char*path){drain();submitted=records=bytes=frames=encoded=raw_frames=recover=groups=0;fp=fopen(path,"wb");if(!fp)throw std::runtime_error("trace open failed");setvbuf(fp,nullptr,_IOFBF,1<<20);SHA256_Init(&raw_hash);SHA256_Init(&decoded_hash);}
 void submit(const Record*raw,uint64_t end){
  while(submitted<end){unsigned i=0;
   {std::unique_lock<std::mutex>lk(mu);cv.wait(lk,[&]{return !error.empty()||!s[0].busy||!s[1].busy;});if(!error.empty())throw std::runtime_error(error);i=!s[0].busy?0:1;s[i].busy=true;}
   Slot&x=s[i];unsigned n=(unsigned)std::min<uint64_t>(MAXN,end-submitted);const Record*p=raw+submitted;
   CK(cudaMemcpyAsync(x.snapshot,p,n*24ul,cudaMemcpyDeviceToDevice,enc));p=x.snapshot;
   if(ere){init_table<<<(TABLE+255)/256,256,0,enc>>>(dev.table);CK(cudaMemsetAsync(dev.count,0,20,enc));CK(cudaMemsetAsync(dev.refs,255,SMS*8,enc));
    group_records<<<(n+255)/256,256,0,enc>>>(p,n,dev);collect_groups<<<(TABLE+255)/256,256,0,enc>>>(dev);
    pack_groups<<<(n+255)/256,256,0,enc>>>(x.d,dev,n);pack_records<<<(n+255)/256,256,0,enc>>>(p,x.d,dev,n);finish_frame<<<1,1,0,enc>>>(x.d,dev,n,submitted);CK(cudaGetLastError());
    CK(cudaMemcpyAsync(&x.head,x.d,64,cudaMemcpyDeviceToHost,enc));CK(cudaStreamSynchronize(enc));
   }else{CK(cudaStreamSynchronize(enc));x.head={};x.head.magic=MAGIC;x.head.n=n;x.head.first=submitted;}
   if(!x.head.format){x.head.bytes=64+uint64_t(n)*24;CK(cudaMemcpyAsync(x.h,p,n*24ul,cudaMemcpyDeviceToHost,copy));}
   else CK(cudaMemcpyAsync(x.h,x.d+64,x.head.bytes-64,cudaMemcpyDeviceToHost,copy));
   if(verify)CK(cudaMemcpyAsync(x.oracle,p,n*24ul,cudaMemcpyDeviceToHost,copy));
   CK(cudaEventRecord(x.ready,copy));submitted+=n;
   {std::lock_guard<std::mutex>lk(mu);q.push_back(i);}cv.notify_all();
  }
 }
 void finish(){drain();if(!fp||fflush(fp)||fclose(fp))throw std::runtime_error("trace close failed");fp=nullptr;if(verify){SHA256_Final(raw_digest,&raw_hash);SHA256_Final(decoded_digest,&decoded_hash);}}
 ~Collector(){try{try{drain();}catch(...){}{std::lock_guard<std::mutex>lk(mu);stop=true;}cv.notify_all();writer.join();if(fp)fclose(fp);for(auto&x:s){cudaFree(x.snapshot);cudaFree(x.d);cudaFreeHost(x.h);cudaEventDestroy(x.ready);}if(ere){cudaFree(dev.table);cudaFree(dev.groups);cudaFree(dev.ids);cudaFree(dev.refs);cudaFree(dev.count);cudaFree(dev.width);}cudaStreamDestroy(enc);cudaStreamDestroy(copy);cudaStreamDestroy(count_stream);}catch(...){}}
};
static thread_local std::string last_error;
#define API_TRY try
#define API_CATCH catch(const std::exception&e){last_error=e.what();return -1;}
extern "C" const char*xg_error(){return last_error.c_str();}
extern "C" void*xg_create(int ere,int verify,const unsigned*w){try{return new Collector(ere,verify,w);}catch(const std::exception&e){last_error=e.what();return nullptr;}}
extern "C" int xg_begin(void*c,const char*p){API_TRY{((Collector*)c)->begin(p);return 0;}API_CATCH}
extern "C" int xg_submit(void*c,uint64_t raw,uint64_t n){API_TRY{((Collector*)c)->submit((const Record*)raw,n);return 0;}API_CATCH}
extern "C" int xg_count(void*c,uint64_t ptr,uint64_t*out){API_TRY{auto*p=(Collector*)c;CK(cudaMemcpyAsync(out,(void*)ptr,8,cudaMemcpyDeviceToHost,p->count_stream));CK(cudaStreamSynchronize(p->count_stream));return 0;}API_CATCH}
extern "C" int xg_finish(void*c){API_TRY{((Collector*)c)->finish();return 0;}API_CATCH}
extern "C" void xg_stats(void*c,uint64_t*out){auto*p=(Collector*)c;uint64_t v[]={p->records,p->bytes,p->frames,p->encoded,p->raw_frames,p->recover,p->groups,p->scratch,64ul<<20,128ul<<20};memcpy(out,v,sizeof(v));}
extern "C" void xg_hashes(void*c,unsigned char*out){auto*p=(Collector*)c;memcpy(out,p->raw_digest,32);memcpy(out+32,p->decoded_digest,32);}
extern "C" void xg_destroy(void*c){delete (Collector*)c;}
extern "C" int xg_validate_file(const char*path,const unsigned*width,unsigned char*hash,uint64_t*stats){
 API_TRY{FILE*f=fopen(path,"rb");if(!f)throw std::runtime_error("validation open failed");SHA256_CTX ctx;SHA256_Init(&ctx);uint64_t n=0,frames=0,bytes=0;Header h;
 while(fread(&h,1,64,f)==64){if(h.magic!=MAGIC||h.first!=n||h.bytes<64||h.bytes>OUTCAP+64||h.n>MAXN)throw std::runtime_error("bad frame");std::vector<unsigned char>p(h.bytes-64),raw;if(fread(p.data(),1,p.size(),f)!=p.size()||!decode(h,p.data(),width,raw))throw std::runtime_error("bad payload");SHA256_Update(&ctx,raw.data(),raw.size());n+=h.n;frames++;bytes+=h.bytes;}
 if(ferror(f)||uint64_t(ftell(f))!=bytes)throw std::runtime_error("trailing/truncated frame");fclose(f);SHA256_Final(hash,&ctx);stats[0]=n;stats[1]=frames;stats[2]=bytes;return 0;}API_CATCH
}

extern "C" uint64_t xg_snapshot_bytes(){return 2*MAXN*24ul;}
extern "C" int xg_source_hash(uint64_t ptr,uint64_t n,unsigned char*out){
 API_TRY{std::vector<unsigned char>b(MAXN*24ul);SHA256_CTX ctx;SHA256_Init(&ctx);
 for(uint64_t i=0;i<n;i+=MAXN){size_t z=std::min<uint64_t>(MAXN,n-i)*24;CK(cudaMemcpy(b.data(),(const unsigned char*)ptr+i*24,z,cudaMemcpyDeviceToHost));SHA256_Update(&ctx,b.data(),z);}
 SHA256_Final(out,&ctx);return 0;}API_CATCH
}

extern "C" int xg_compare_source(uint64_t ptr,const char*path,const unsigned*width){
 API_TRY{FILE*f=fopen(path,"rb");if(!f)throw std::runtime_error("comparison open failed");Header h;
 while(fread(&h,1,64,f)==64){std::vector<unsigned char>p(h.bytes-64),decoded,original(h.n*24ul);if(fread(p.data(),1,p.size(),f)!=p.size()||!decode(h,p.data(),width,decoded))throw std::runtime_error("bad comparison frame");
 CK(cudaMemcpy(original.data(),(const unsigned char*)ptr+h.first*24,original.size(),cudaMemcpyDeviceToHost));
 if(memcmp(original.data(),decoded.data(),original.size())){size_t i=0;while(original[i]==decoded[i])i++;Record a{},b{};memcpy(&a,original.data()+(i/24)*24,24);memcpy(&b,decoded.data()+(i/24)*24,24);
 FILE*z=fopen("/dev/shm/xtrace-paper-gpu-unit32/final-source-failure.raw","wb");if(z){fwrite(original.data(),1,original.size(),z);fclose(z);}z=fopen("/dev/shm/xtrace-paper-gpu-unit32/final-source-failure.frame","wb");if(z){fwrite(&h,1,64,z);fwrite(p.data(),1,p.size(),z);fclose(z);}fclose(f);
 throw std::runtime_error("source difference: record="+std::to_string(h.first+i/24)+" byte="+std::to_string(i%24)+" source_t="+std::to_string(a.t)+" trace_t="+std::to_string(b.t)+" source_grid="+std::to_string(a.grid)+" trace_grid="+std::to_string(b.grid)+" source_warp="+std::to_string(a.warp)+" trace_warp="+std::to_string(b.warp)+" source_sm="+std::to_string(a.sm)+" trace_sm="+std::to_string(b.sm)+" source_type="+std::to_string(a.type)+" trace_type="+std::to_string(b.type)+" source_kid="+std::to_string(a.kid)+" trace_kid="+std::to_string(b.kid));}}
 fclose(f);return 0;}API_CATCH
}

extern "C" uint64_t xg_ref_unit(){return REF_UNIT;}
