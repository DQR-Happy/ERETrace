"""Use a reusable contiguous decoder workspace for the Cascaded baseline."""
from pathlib import Path
r=Path(__file__).resolve().parents[1];p=r/'source/benchmarks/nvcomp_pipeline.cu'
s=p.read_text();start=s.index('extern "C" int nvpipe_decode(');end=s.index('extern "C" void nvpipe_destroy',start)
decoder=r'''
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
'''
p.write_text(s[:start]+decoder+s[end:]);print(p)
