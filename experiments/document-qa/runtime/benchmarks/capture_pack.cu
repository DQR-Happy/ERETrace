// One pass, exact clock64/elapsed32 and checked SM8; field arrays stay aligned.
// Original model kernel executes once. This kernel only reads completed Maps.
#include <stdint.h>
struct Record { uint64_t start;uint32_t elapsed,sm; };
extern "C" __global__ void neutrino_capture_pack(const Record*src,unsigned char*out,
                                                unsigned long long n,unsigned*error){
    unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    Record r=src[i];if(r.sm>255)atomicOr(error,1U);
    reinterpret_cast<uint64_t*>(out)[i]=r.start;
    reinterpret_cast<uint32_t*>(out+n*8)[i]=r.elapsed;
    out[n*12+i]=(unsigned char)r.sm;
}

struct PackDesc { uint64_t source,dest,n;uint32_t first,blocks; };
// Flatten work in CTA units, rather than padding every Map to the largest Map.
extern "C" __global__ void neutrino_capture_pack_batch(const PackDesc* desc,unsigned count,unsigned*error){
    unsigned lo=0,hi=count;
    while(lo+1<hi){unsigned mid=(lo+hi)/2;if(desc[mid].first<=blockIdx.x)lo=mid;else hi=mid;}
    PackDesc d=desc[lo];uint64_t i=(uint64_t)(blockIdx.x-d.first)*blockDim.x+threadIdx.x;
    if(i>=d.n)return;
    Record r=((const Record*)d.source)[i];unsigned char*out=(unsigned char*)d.dest;
    if(r.sm>255)atomicOr(error,1U);
    ((uint64_t*)out)[i]=r.start;((uint32_t*)(out+d.n*8))[i]=r.elapsed;out[d.n*12+i]=(unsigned char)r.sm;
}

#include <cuda/atomic>
// Fixed metadata size avoids a host length round-trip: flag/padding 16 + 256 bases*8.
// The first arriving record on each SM supplies a base through a stable raw index.
extern "C" __global__ void neutrino_capture_delta_init(const PackDesc*desc,unsigned*indices){
    unsigned f=blockIdx.x,sm=threadIdx.x;unsigned char*out=(unsigned char*)desc[f].dest;
    indices[f*256+sm]=0xffffffffU;((uint64_t*)(out+16))[sm]=0;
    if(sm<4)((unsigned*)out)[sm]=0;
}
extern "C" __global__ void neutrino_capture_delta_batch(const PackDesc*desc,unsigned count,unsigned*indices){
    unsigned lo=0,hi=count;while(lo+1<hi){unsigned mid=(lo+hi)/2;if(desc[mid].first<=blockIdx.x)lo=mid;else hi=mid;}
    PackDesc d=desc[lo];uint64_t i=(uint64_t)(blockIdx.x-d.first)*blockDim.x+threadIdx.x;if(i>=d.n)return;
    const Record*src=(const Record*)d.source;Record r=src[i];unsigned char*out=(unsigned char*)d.dest;
    if(r.sm>255){atomicOr((unsigned*)out,1U);return;}
    cuda::atomic_ref<unsigned,cuda::thread_scope_device> index(indices[lo*256+r.sm]);
    unsigned chosen=index.load(cuda::memory_order_relaxed);
    if(chosen==0xffffffffU){unsigned expected=0xffffffffU;
        if(index.compare_exchange_strong(expected,(unsigned)i,cuda::memory_order_relaxed)){chosen=(unsigned)i;((uint64_t*)(out+16))[r.sm]=r.start;}
        else chosen=expected;
    }
    uint64_t base=src[chosen].start,difference=r.start-base;
    int32_t delta=(int32_t)(uint32_t)difference;
    if(base+(uint64_t)(int64_t)delta!=r.start)atomicOr((unsigned*)out,2U);
    ((int32_t*)(out+2064))[i]=delta;((uint32_t*)(out+2064+d.n*4))[i]=r.elapsed;out[2064+d.n*8+i]=(unsigned char)r.sm;
}

// Experimental fixed-width family. Widths are part of every frame, never inferred
// by the decoder. Range escapes retain the original Map and use the raw fallback.
extern "C" __global__ void neutrino_capture_flex_init(const PackDesc*desc,unsigned*indices,unsigned dw,unsigned ew){
    unsigned f=blockIdx.x,sm=threadIdx.x;unsigned char*out=(unsigned char*)desc[f].dest;
    indices[f*256+sm]=0xffffffffU;((uint64_t*)(out+16))[sm]=0;
    if(sm==0){((unsigned*)out)[0]=0;((unsigned*)out)[1]=dw;((unsigned*)out)[2]=ew;((unsigned*)out)[3]=1;}
}
extern "C" __global__ void neutrino_capture_flex_batch(const PackDesc*desc,unsigned count,unsigned*indices,unsigned dw,unsigned ew){
    unsigned lo=0,hi=count;while(lo+1<hi){unsigned mid=(lo+hi)/2;if(desc[mid].first<=blockIdx.x)lo=mid;else hi=mid;}
    PackDesc d=desc[lo];uint64_t i=(uint64_t)(blockIdx.x-d.first)*blockDim.x+threadIdx.x;if(i>=d.n)return;
    const Record*src=(const Record*)d.source;Record r=src[i];unsigned char*out=(unsigned char*)d.dest;
    if(r.sm>255){atomicOr((unsigned*)out,1U);return;}
    cuda::atomic_ref<unsigned,cuda::thread_scope_device> index(indices[lo*256+r.sm]);
    unsigned chosen=index.load(cuda::memory_order_relaxed);
    if(chosen==0xffffffffU){unsigned expected=0xffffffffU;
        if(index.compare_exchange_strong(expected,(unsigned)i,cuda::memory_order_relaxed)){chosen=(unsigned)i;((uint64_t*)(out+16))[r.sm]=r.start;}else chosen=expected;}
    uint64_t base=src[chosen].start,difference=r.start-base;
    unsigned bits=dw*8,shift=32-bits;int32_t delta=(int32_t)((uint32_t)difference<<shift)>>shift;
    if(base+(uint64_t)(int64_t)delta!=r.start)atomicOr((unsigned*)out,2U);
    if(ew<4 && (r.elapsed>>(ew*8)))atomicOr((unsigned*)out,4U);
    for(unsigned b=0;b<dw;b++)out[2064+i*dw+b]=(unsigned char)((uint32_t)delta>>(8*b));
    for(unsigned b=0;b<ew;b++)out[2064+d.n*dw+i*ew+b]=(unsigned char)(r.elapsed>>(8*b));
    out[2064+d.n*(dw+ew)+i]=(unsigned char)r.sm;
}

// Hierarchical reference domains: SM -> CTA -> warp. PackDesc.blocks carries
// warps/CTA here; first remains the flattened helper CTA offset. Small CTA
// shapes use the flat format when its exact size is no larger.
extern "C" __global__ void neutrino_capture_hier_init(const PackDesc*desc,unsigned*indices,unsigned dw,unsigned ew){
    unsigned f=blockIdx.x,sm=threadIdx.x;PackDesc d=desc[f];unsigned char*out=(unsigned char*)d.dest;
    indices[f*256+sm]=0xffffffffU;((uint64_t*)(out+16))[sm]=0;
    bool hier=d.blocks>2; // 4-byte CTA anchor vs two saved bytes per warp.
    if(sm==0){((unsigned*)out)[0]=0;((unsigned*)out)[1]=hier?d.blocks:3;((unsigned*)out)[2]=ew;((unsigned*)out)[3]=hier?2:1;}
}
extern "C" __global__ void neutrino_capture_hier_batch(const PackDesc*desc,unsigned count,unsigned*indices,unsigned dw,unsigned ew){
    unsigned lo=0,hi=count;while(lo+1<hi){unsigned mid=(lo+hi)/2;if(desc[mid].first<=blockIdx.x)lo=mid;else hi=mid;}
    PackDesc d=desc[lo];uint64_t i=(uint64_t)(blockIdx.x-d.first)*blockDim.x+threadIdx.x;if(i>=d.n)return;
    const Record*src=(const Record*)d.source;Record r=src[i];unsigned char*out=(unsigned char*)d.dest;
    unsigned wp=d.blocks;bool hier=wp>2;uint64_t anchor=hier?(i/wp)*wp:i;Record a=src[anchor];
    if(r.sm>255||a.sm>255){atomicOr((unsigned*)out,1U);return;}
    if(hier&&r.sm!=a.sm)atomicOr((unsigned*)out,8U);
    cuda::atomic_ref<unsigned,cuda::thread_scope_device> index(indices[lo*256+a.sm]);unsigned chosen=index.load(cuda::memory_order_relaxed);
    if(chosen==0xffffffffU){unsigned expected=0xffffffffU;if(index.compare_exchange_strong(expected,(unsigned)anchor,cuda::memory_order_relaxed)){chosen=(unsigned)anchor;((uint64_t*)(out+16))[a.sm]=a.start;}else chosen=expected;}
    uint64_t base=src[chosen].start;int32_t delta=int32_t(uint32_t(a.start-base)<<8)>>8;
    if(base+(uint64_t)(int64_t)delta!=a.start)atomicOr((unsigned*)out,2U);
    if(ew<4&&(r.elapsed>>(ew*8)))atomicOr((unsigned*)out,4U);
    if(hier){uint64_t ctas=d.n/wp;int32_t local=int32_t(uint32_t(r.start-a.start)<<16)>>16;
        if(a.start+(uint64_t)(int64_t)local!=r.start)atomicOr((unsigned*)out,16U);
        if(i==anchor){for(unsigned b=0;b<3;b++)out[2064+(i/wp)*4+b]=(unsigned char)((uint32_t)delta>>(8*b));out[2064+(i/wp)*4+3]=(unsigned char)a.sm;}
        out[2064+ctas*4+i*2]=(unsigned char)local;out[2064+ctas*4+i*2+1]=(unsigned char)((uint32_t)local>>8);
        for(unsigned b=0;b<ew;b++)out[2064+ctas*4+d.n*2+i*ew+b]=(unsigned char)(r.elapsed>>(8*b));
    }else{for(unsigned b=0;b<3;b++)out[2064+i*3+b]=(unsigned char)((uint32_t)delta>>(8*b));for(unsigned b=0;b<ew;b++)out[2064+d.n*3+i*ew+b]=(unsigned char)(r.elapsed>>(8*b));out[2064+d.n*(3+ew)+i]=(unsigned char)r.sm;}
}

// Matched reference-domain controls: blocks is a chosen group size, which may
// come from the CTA shape or a fixed tile without observing that shape.
// Each thread owns four adjacent records, so packed12 never races on a byte.
extern "C" __global__ void neutrino_capture_reference_init(const PackDesc*desc,unsigned*indices,unsigned ew,unsigned local_bits){
    unsigned f=blockIdx.x,sm=threadIdx.x;PackDesc d=desc[f];unsigned char*out=(unsigned char*)d.dest;
    indices[f*256+sm]=0xffffffffU;((uint64_t*)(out+16))[sm]=0;
    if(!sm){((unsigned*)out)[0]=0;((unsigned*)out)[1]=d.blocks>2?d.blocks:3;((unsigned*)out)[2]=ew;((unsigned*)out)[3]=d.blocks>2?(local_bits==12?4:3):1;}
}
extern "C" __global__ void neutrino_capture_reference_batch(const PackDesc*desc,unsigned count,unsigned*indices,unsigned ew,unsigned local_bits){
    unsigned lo=0,hi=count;while(lo+1<hi){unsigned mid=(lo+hi)/2;if(desc[mid].first<=blockIdx.x)lo=mid;else hi=mid;}
    PackDesc d=desc[lo];uint64_t first=((uint64_t)(blockIdx.x-d.first)*blockDim.x+threadIdx.x)*4;if(first>=d.n)return;
    const Record*src=(const Record*)d.source;unsigned char*out=(unsigned char*)d.dest;unsigned k=d.blocks;bool grouped=k>2;
    uint64_t groups=(d.n+k-1)/k,local_bytes=(d.n*local_bits+7)/8;unsigned valid=(unsigned)(d.n-first<4?d.n-first:4);uint64_t packed=0;
    for(unsigned j=0;j<valid;j++){uint64_t i=first+j,anchor=grouped?(i/k)*k:i;Record r=src[i],a=src[anchor];
        if(r.sm>255||a.sm>255){atomicOr((unsigned*)out,1U);continue;}
        if(grouped&&r.sm!=a.sm)atomicOr((unsigned*)out,8U);
        cuda::atomic_ref<unsigned,cuda::thread_scope_device> idx(indices[lo*256+a.sm]);unsigned chosen=idx.load(cuda::memory_order_relaxed);
        if(chosen==0xffffffffU){unsigned expected=0xffffffffU;if(idx.compare_exchange_strong(expected,(unsigned)anchor,cuda::memory_order_relaxed)){chosen=(unsigned)anchor;((uint64_t*)(out+16))[a.sm]=a.start;}else chosen=expected;}
        uint64_t base=src[chosen].start;int32_t delta=int32_t(uint32_t(a.start-base)<<8)>>8;
        if(base+(uint64_t)(int64_t)delta!=a.start)atomicOr((unsigned*)out,2U);
        if(ew<4&&(r.elapsed>>(8*ew)))atomicOr((unsigned*)out,4U);
        if(grouped){unsigned shift=32-local_bits;int32_t local=int32_t(uint32_t(r.start-a.start)<<shift)>>shift;
            if(a.start+(uint64_t)(int64_t)local!=r.start)atomicOr((unsigned*)out,16U);
            if(i==anchor){for(unsigned b=0;b<3;b++)out[2064+(i/k)*4+b]=(unsigned char)(uint32_t(delta)>>(8*b));out[2064+(i/k)*4+3]=(unsigned char)a.sm;}
            if(local_bits==12)packed|=(uint64_t(uint32_t(local)&4095U)<<(12*j));
            else{out[2064+4*groups+2*i]=(unsigned char)local;out[2064+4*groups+2*i+1]=(unsigned char)(uint32_t(local)>>8);}
            for(unsigned b=0;b<ew;b++)out[2064+4*groups+local_bytes+i*ew+b]=(unsigned char)(r.elapsed>>(8*b));
        }else{for(unsigned b=0;b<3;b++)out[2064+i*3+b]=(unsigned char)(uint32_t(delta)>>(8*b));for(unsigned b=0;b<ew;b++)out[2064+d.n*3+i*ew+b]=(unsigned char)(r.elapsed>>(8*b));out[2064+d.n*(3+ew)+i]=(unsigned char)r.sm;}
    }
    if(grouped&&local_bits==12){unsigned bytes=(valid*12+7)/8;for(unsigned b=0;b<bytes;b++)out[2064+4*groups+first*12/8+b]=(unsigned char)(packed>>(8*b));}
}

#include "capture_compact.cuh"
