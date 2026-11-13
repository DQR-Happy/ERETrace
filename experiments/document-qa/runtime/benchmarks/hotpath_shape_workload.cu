#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <dlfcn.h>
#include <chrono>
#define CK(x) do { cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s at %d\n",cudaGetErrorString(e),__LINE__);return 2;} } while(0)
extern "C" __global__ void neutrino_reference_shape(int*y){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;y[i]+=1;}
int main(int argc,char**argv){if(argc!=3)return 1;unsigned threads=atoi(argv[1]),grid=4096;size_t n=size_t(threads)*grid;int*y;CK(cudaMalloc(&y,n*4));CK(cudaMemset(y,0,n*4));CK(cudaDeviceSynchronize());
 void*hook=nullptr;int(*boundary)()=nullptr;int(*summary)(const char*)=nullptr;if(const char*p=getenv("NEUTRINO_HOOK_DRIVER")){hook=dlopen(p,RTLD_NOW);if(!hook)return 3;boundary=(int(*)())dlsym(hook,"neutrino_ablation_boundary");summary=(int(*)(const char*))dlsym(hook,"neutrino_capture_summary");if(!boundary||!summary)return 4;}
 std::vector<long long> times;for(int r=0;r<8;r++){auto t=std::chrono::steady_clock::now();neutrino_reference_shape<<<grid,threads,(r/2)%2?64:0>>>(y);CK(cudaGetLastError());CK(cudaDeviceSynchronize());if(boundary&&boundary())return 5;times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-t).count());}
 std::vector<int> h(n);CK(cudaMemcpy(h.data(),y,n*4,cudaMemcpyDeviceToHost));for(int x:h)if(x!=8)return 6;CK(cudaFree(y));
 std::string dir=argv[2];if(summary&&summary((dir+"/capture-summary.json").c_str()))return 7;
 FILE*f=fopen((dir+"/workload.json").c_str(),"w");if(!f)return 8;fprintf(f,"{\"passed\":true,\"threads\":%u,\"grid\":%u,\"launches\":8,\"warmup\":2,\"elements_checked\":%zu,\"expected_each\":8,\"measured_ns\":[",threads,grid,n);for(int i=2;i<8;i++)fprintf(f,"%s%lld",i==2?"":",",times[i]);fprintf(f,"]}\n");fclose(f);return 0;}
