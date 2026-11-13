#include <cuda.h>
extern "C" int capacity_repeat(CUfunction f,int grid,int block,CUdeviceptr data,
 CUdeviceptr records,CUdeviceptr counts,CUdeviceptr plan,int table,int repetitions,float *ms) {
 void *args[]={&data,&records,&counts,&counts};
 if(table){args[2]=&plan;args[3]=&counts;}
 CUevent a=nullptr,b=nullptr;CUresult r=cuEventCreate(&a,0);
 if(!r)r=cuEventCreate(&b,0);if(!r)r=cuEventRecord(a,nullptr);
 for(int i=0;!r&&i<repetitions;i++)r=cuLaunchKernel(f,grid,1,1,block,1,1,0,nullptr,args,nullptr);
 if(!r)r=cuEventRecord(b,nullptr);if(!r)r=cuEventSynchronize(b);
 if(!r)r=cuEventElapsedTime(ms,a,b);if(a)cuEventDestroy(a);if(b)cuEventDestroy(b);return r;
}
