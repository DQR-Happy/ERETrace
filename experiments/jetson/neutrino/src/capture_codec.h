/* Lossless clock64 records: independent SM epochs, exact elapsed, zero sentinel.
 * Return 0 to retain the raw frame when range/ID/size prevents useful packing.
 * Destination capacity must be at least raw_bytes. No GPU or allocation calls.
 */
#ifndef NEUTRINO_CAPTURE_CODEC_H
#define NEUTRINO_CAPTURE_CODEC_H
#include <stdint.h>
#include <stddef.h>
#include <string.h>
static size_t nt_pack_records(void*dst,const void*src,size_t bytes,uint64_t*zeros,uint64_t*wide){
    if(!bytes || bytes%16)return 0;
    uint64_t low[255],high[255],global_low=UINT64_MAX,global_high=0;unsigned char seen[255]={0};
    unsigned active=0;int possible=1;
    for(int s=0;s<255;s++){low[s]=UINT64_MAX;high[s]=0;}
    for(size_t i=0;i<bytes;i+=16){uint64_t start;uint32_t elapsed,sm;
        memcpy(&start,(const char*)src+i,8);memcpy(&elapsed,(const char*)src+i+8,4);memcpy(&sm,(const char*)src+i+12,4);
        if(!start&&!elapsed&&!sm){if(zeros)(*zeros)++;continue;}
        if(sm>=255){possible=0;break;}seen[sm]=1;
        if(start<low[sm])low[sm]=start;if(start>high[sm])high[sm]=start;
        if(start<global_low)global_low=start;if(start>global_high)global_high=start;
    }
    if(wide && global_high>=global_low && global_high-global_low>UINT32_MAX)(*wide)++;
    for(int s=0;s<255;s++)if(seen[s]){active++;if(high[s]-low[s]>UINT32_MAX)possible=0;}
    size_t wanted=2+active*9+bytes/16*9;
    if(!possible || wanted>=bytes)return 0;
    unsigned char*out=dst;uint16_t count=active;memcpy(out,&count,2);size_t pos=2;
    for(int s=0;s<255;s++)if(seen[s]){out[pos++]=s;memcpy(out+pos,&low[s],8);pos+=8;}
    for(size_t i=0;i<bytes;i+=16,pos+=9){uint64_t start;uint32_t elapsed,sm;
        memcpy(&start,(const char*)src+i,8);memcpy(&elapsed,(const char*)src+i+8,4);memcpy(&sm,(const char*)src+i+12,4);
        uint32_t delta;if(!start&&!elapsed&&!sm){delta=0;sm=255;}else delta=(uint32_t)(start-low[sm]);
        memcpy(out+pos,&delta,4);memcpy(out+pos+4,&elapsed,4);out[pos+8]=(unsigned char)sm;
    }return wanted;
}
static int nt_unpack_records(void*dst,size_t bytes,const void*src,size_t packed){
    if(bytes%16 || packed<2)return -1;
    uint64_t bases[255];unsigned char seen[255]={0};uint16_t count;const unsigned char*in=src;
    memcpy(&count,in,2);size_t pos=2;if(count>255 || packed!=2+count*9+bytes/16*9)return -1;
    for(unsigned s=0;s<count;s++){unsigned id=in[pos++];if(id>=255 || seen[id])return -1;
        seen[id]=1;memcpy(&bases[id],in+pos,8);pos+=8;}
    for(size_t i=0;i<bytes;i+=16,pos+=9){uint32_t delta,elapsed,sm=in[pos+8];
        memcpy(&delta,in+pos,4);memcpy(&elapsed,in+pos+4,4);uint64_t start;
        if(sm==255){if(delta||elapsed)return -1;start=0;elapsed=0;sm=0;}
        else{if(!seen[sm] || UINT64_MAX-bases[sm]<delta)return -1;start=bases[sm]+delta;}
        memcpy((char*)dst+i,&start,8);memcpy((char*)dst+i+8,&elapsed,4);memcpy((char*)dst+i+12,&sm,4);
    }return 0;
}
#endif
