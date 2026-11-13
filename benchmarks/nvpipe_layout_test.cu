// Comparator-adapter validation, independent CPU offsets and error oracle.
#include "nvcomp_pipeline.cu"
int main(){
 unsigned cases=0;uint64_t tested=0;
 for(unsigned nf: {1U,2U,63U,64U,255U,256U,257U,513U}){
  std::vector<Piece> pieces;std::vector<size_t> lengths;std::vector<nvcompStatus_t> status;
  std::vector<uint64_t> expected_pos,expected_meta(nf+1);std::vector<unsigned char> expected_headers;
  uint64_t total=0;unsigned ne=0;
  for(unsigned f=0;f<nf;f++){
   bool enc=f%3!=0;unsigned chunks=1+f%7;uint64_t start=total;
   if(enc)total+=16+8*chunks;
   for(unsigned k=0;k<chunks;k++){
    size_t rawbytes=16*(1+(f+k)%256),len=enc?(1+(f*71+k*31)%499):rawbytes;
    pieces.push_back({0,rawbytes,f,ne,k,chunks,int(enc)});expected_pos.push_back(total);
    if(enc){lengths.push_back(len);status.push_back(nvcompSuccess);ne++;}
    total+=enc?((len+7)&~uint64_t(7)):len;
   }expected_meta[f+1]=total-start;
  }expected_meta[0]=total;
  Piece*dp;size_t*dl;nvcompStatus_t*ds;uint64_t*pos,*meta,*starts;unsigned char*out;
  CK(cudaMalloc(&dp,pieces.size()*sizeof(Piece)));CK(cudaMalloc(&dl,std::max(size_t(1),lengths.size())*8));
  CK(cudaMalloc(&ds,std::max(size_t(1),status.size())*sizeof(nvcompStatus_t)));
  CK(cudaMalloc(&pos,pieces.size()*8));CK(cudaMalloc(&meta,(nf+1)*8));CK(cudaMalloc(&starts,nf*8));CK(cudaMalloc(&out,total));
  CK(cudaMemcpy(dp,pieces.data(),pieces.size()*sizeof(Piece),cudaMemcpyHostToDevice));
  if(ne){CK(cudaMemcpy(dl,lengths.data(),ne*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(ds,status.data(),ne*sizeof(nvcompStatus_t),cudaMemcpyHostToDevice));}
  for(unsigned fail=0;fail<(ne?2U:1U);fail++){
   if(fail){status[ne/2]=(nvcompStatus_t)1;CK(cudaMemcpy(ds,status.data(),ne*sizeof(nvcompStatus_t),cudaMemcpyHostToDevice));}
   make_layout_parallel<<<1,256>>>(dp,pieces.size(),nf,dl,ds,pos,meta,starts,out);CK(cudaDeviceSynchronize());
   std::vector<uint64_t> got(nf+1),offsets(pieces.size());CK(cudaMemcpy(got.data(),meta,(nf+1)*8,cudaMemcpyDeviceToHost));
   if(fail){if(got[0]!=~uint64_t(0))return 1;cases++;continue;}
   CK(cudaMemcpy(offsets.data(),pos,offsets.size()*8,cudaMemcpyDeviceToHost));
   if(got!=expected_meta||offsets!=expected_pos)return 2;
   std::vector<unsigned char> host(total);CK(cudaMemcpy(host.data(),out,total,cudaMemcpyDeviceToHost));
   uint64_t start=0;unsigned i=0;
   for(unsigned f=0;f<nf;f++){
    auto q=pieces[i];if(q.encoded){uint32_t h[4];memcpy(h,host.data()+start,16);
     if(h[0]!=q.chunks||h[1]!=0x4e564331||h[2]!=q.bytes||h[3])return 3;
     for(unsigned k=0;k<q.chunks;k++){uint64_t len;memcpy(&len,host.data()+start+16+k*8,8);if(len!=lengths[pieces[i+k].index])return 4;}}
    i+=q.chunks;start+=got[f+1];
   }cases++;tested+=pieces.size();
  }
  for(void*p:{(void*)dp,(void*)dl,(void*)ds,(void*)pos,(void*)meta,(void*)starts,(void*)out})CK(cudaFree(p));
 }
 printf("{\"passed\":true,\"cases\":%u,\"pieces\":%llu}\n",cases,(unsigned long long)tested);
}
