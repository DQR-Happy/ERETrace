// Exact CPU decoder for the existing CTA-reference representation. No summaries.
#include <cstdint>
#include <cstddef>
#include <cstring>
static uint32_t small(const unsigned char*p,unsigned width){uint32_t v=0;for(unsigned k=0;k<width;k++)v|=uint32_t(p[k])<<(8*k);return v;}
extern "C" int case_reference_decode(const void*input,size_t bytes,void*output,size_t n){
 if(bytes<2064)return 1;auto*p=(const unsigned char*)input;uint32_t h[4];memcpy(h,p,16);
 unsigned group=h[1],ew=h[2],version=h[3];if(h[0]||group<=2||ew<2||ew>4||version<3||version>5)return 2;
 size_t groups=(n+group-1)/group,m=n-(version==5?groups:0),width=version==3?16:12;
 size_t locals=(m*width+7)/8;if(bytes!=2064+groups*4+locals+n*ew)return 3;
 auto*anchor=p+2064;auto*local=anchor+groups*4;auto*elapsed=local+locals;auto*out=(unsigned char*)output;
 for(size_t i=0;i<n;i++){
  size_t g=i/group;const auto*a=anchor+4*g;uint32_t bits=small(a,3);int64_t dt=int64_t((bits^0x800000u))-0x800000;
  uint64_t base;memcpy(&base,p+16+8*a[3],8);int64_t delta=0;
  if(version!=5 || i%group){
   size_t j=version==5?i-g-1:i;
   if(width==16){int16_t x;memcpy(&x,local+2*j,2);delta=x;}
   else {size_t bit=12*j;uint32_t v=small(local+bit/8,2)>>(bit%8);delta=int64_t((v&4095)^2048)-2048;}
  }
  uint64_t start=base+uint64_t(dt)+uint64_t(delta);uint32_t e=small(elapsed+i*ew,ew),sm=a[3];
  memcpy(out+16*i,&start,8);memcpy(out+16*i+8,&e,4);memcpy(out+16*i+12,&sm,4);
 }
 return 0;
}
