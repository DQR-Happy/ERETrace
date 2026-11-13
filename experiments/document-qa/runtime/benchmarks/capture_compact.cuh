// Elide the identically-zero local delta at each reference-group anchor.
// Four COMPACT locals per thread own exactly six bytes; no overlapping stores.
extern "C" __global__ void neutrino_capture_compact_init(const PackDesc*desc,unsigned*indices,unsigned ew,unsigned bits){
    unsigned f=blockIdx.x,sm=threadIdx.x;PackDesc d=desc[f];unsigned char*out=(unsigned char*)d.dest;
    indices[f*256+sm]=0xffffffffU;((uint64_t*)(out+16))[sm]=0;
    if(!sm){((unsigned*)out)[0]=0;((unsigned*)out)[1]=d.blocks>2?d.blocks:3;((unsigned*)out)[2]=ew;((unsigned*)out)[3]=d.blocks>2?5:1;}
}
__device__ uint64_t compact_base(const Record*src,uint64_t anchor,unsigned sm,unsigned*idx,unsigned char*out){
    cuda::atomic_ref<unsigned,cuda::thread_scope_device> ref(*idx);unsigned chosen=ref.load(cuda::memory_order_relaxed);
    if(chosen==0xffffffffU){unsigned expected=0xffffffffU;if(ref.compare_exchange_strong(expected,(unsigned)anchor,cuda::memory_order_relaxed)){chosen=anchor;((uint64_t*)(out+16))[sm]=src[anchor].start;}else chosen=expected;}
    return src[chosen].start;
}
template<unsigned FIXED_K>
__device__ __forceinline__ void compact_body(const PackDesc*desc,unsigned lo,unsigned*indices,unsigned ew,unsigned bits){
    PackDesc d=desc[lo];uint64_t first=((uint64_t)(blockIdx.x-d.first)*blockDim.x+threadIdx.x)*4;if(first>=d.n)return;
    const Record*src=(const Record*)d.source;unsigned char*out=(unsigned char*)d.dest;unsigned k=FIXED_K?FIXED_K:d.blocks;
    if(k<=2){for(unsigned j=0;j<4&&first+j<d.n;j++){uint64_t i=first+j;Record r=src[i];if(r.sm>255){atomicOr((unsigned*)out,1U);continue;}
        uint64_t base=compact_base(src,i,r.sm,indices+lo*256+r.sm,out);int32_t delta=int32_t(uint32_t(r.start-base)<<8)>>8;
        if(base+(uint64_t)(int64_t)delta!=r.start)atomicOr((unsigned*)out,2U);if(ew<4&&(r.elapsed>>(8*ew)))atomicOr((unsigned*)out,4U);
        for(unsigned b=0;b<3;b++)out[2064+i*3+b]=uint32_t(delta)>>(8*b);for(unsigned b=0;b<ew;b++)out[2064+d.n*3+i*ew+b]=r.elapsed>>(8*b);out[2064+d.n*(3+ew)+i]=r.sm;}return;}
    uint64_t groups=(d.n+k-1)/k,m=d.n-groups,bytes=(m*12+7)/8,packed=0;unsigned valid=first<m?(unsigned)(m-first<4?m-first:4):0;
    for(unsigned j=0;j<4;j++){uint64_t x=first+j;
        if(x<groups){uint64_t anchor=x*k;Record a=src[anchor];if(a.sm>255)atomicOr((unsigned*)out,1U);else{
            uint64_t base=compact_base(src,anchor,a.sm,indices+lo*256+a.sm,out);int32_t delta=int32_t(uint32_t(a.start-base)<<8)>>8;
            if(base+(uint64_t)(int64_t)delta!=a.start)atomicOr((unsigned*)out,2U);
            for(unsigned b=0;b<3;b++)out[2064+4*x+b]=uint32_t(delta)>>(8*b);out[2064+4*x+3]=a.sm;}}
        if(x<d.n){uint32_t elapsed=src[x].elapsed;if(ew<4&&(elapsed>>(8*ew)))atomicOr((unsigned*)out,4U);for(unsigned b=0;b<ew;b++)out[2064+groups*4+bytes+x*ew+b]=elapsed>>(8*b);}
        if(x<m){uint64_t i=x+1+x/(k-1),anchor=(i/k)*k;Record r=src[i],a=src[anchor];
            if(r.sm>255||a.sm>255){atomicOr((unsigned*)out,1U);continue;}if(r.sm!=a.sm)atomicOr((unsigned*)out,8U);
            int32_t local=int32_t(uint32_t(r.start-a.start)<<20)>>20;if(a.start+(uint64_t)(int64_t)local!=r.start)atomicOr((unsigned*)out,16U);
            packed|=uint64_t(uint32_t(local)&4095U)<<(12*j);}
    }
    for(unsigned b=0;b<(valid*12+7)/8;b++)out[2064+groups*4+first*12/8+b]=packed>>(8*b);
}
extern "C" __global__ void neutrino_capture_compact_batch(const PackDesc*desc,unsigned count,unsigned*indices,unsigned ew,unsigned bits){
    unsigned lo=0,hi=count;while(lo+1<hi){unsigned m=(lo+hi)/2;if(desc[m].first<=blockIdx.x)lo=m;else hi=m;}
#if NEUTRINO_COMPACT_SPECIALIZE
    // Constant divisors preserve the index map and avoid generic 64-bit division.
    switch(desc[lo].blocks){
        case 4: compact_body<4>(desc,lo,indices,ew,bits);break;
        case 8: compact_body<8>(desc,lo,indices,ew,bits);break;
        default: compact_body<0>(desc,lo,indices,ew,bits);break;
    }
#else
    compact_body<0>(desc,lo,indices,ew,bits);
#endif
}
