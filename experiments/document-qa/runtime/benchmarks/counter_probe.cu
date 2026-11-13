#include <cuda_runtime.h>
#include <stdio.h>
__global__ void counter_permission_probe(int *data){data[threadIdx.x]=(int)threadIdx.x;}
int main(){int *p,host[256];if(cudaMalloc(&p,sizeof(host))!=cudaSuccess)return 1;
 counter_permission_probe<<<1,256>>>(p);if(cudaDeviceSynchronize()!=cudaSuccess)return 2;
 if(cudaMemcpy(host,p,sizeof(host),cudaMemcpyDeviceToHost)!=cudaSuccess)return 3;
 for(int i=0;i<256;i++)if(host[i]!=i)return 4;cudaFree(p);puts("CUDA counter probe correct");return 0;}
