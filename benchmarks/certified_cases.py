"""Launch specs for two unmodified real llama.cpp PTX entries and a density control."""
import ctypes as C,math
from pathlib import Path
from neutrino.probe.core_bounds import Program

def density_ptx():return '''.version 8.4
.target sm_86
.address_size 64
.visible .entry density(.param .u64 data,.param .u32 mask)
{
.reg .u32 %t,%b,%n,%g,%v,%j,%lim;
.reg .u64 %p,%off;
.reg .pred %again;
ld.param.u64 %p, [data];
ld.param.u32 %n, [mask];
mov.u32 %t, %tid.x;
mov.u32 %b, %ctaid.x;
mov.u32 %g, %ntid.x;
mad.lo.u32 %t, %b, %g, %t;
mul.wide.u32 %off, %t, 8;
add.u64 %p, %p, %off;
ld.global.u32 %v, [%p];
and.b32 %lim, %v, %n;
add.u32 %lim, %lim, 1;
mov.u32 %j, 0;
LOOP:
ld.global.u32 %v, [%p+4];
add.u32 %v, %v, 1;
st.global.u32 [%p+4], %v;
add.u32 %j, %j, 1;
setp.lt.u32 %again, %j, %lim;
@%again bra LOOP;
ret;
}'''

def cases(inputs):
 out=[]
 for n in [1025,1536,4097]:out.append(dict(tag='rms-'+str(n),kind='rms',n=n,grid=82,block=1024,shared=128,ptx=(inputs/'rms.ptx').read_text()))
 for n in [1025,16385]:out.append(dict(tag='silu-'+str(n),kind='silu',n=n,grid=(n+255)//256,block=256,shared=0,ptx=(inputs/'silu.ptx').read_text()))
 for grid in [3,82]:
  for density in [0,3,15,31]:out.append(dict(tag=f'density-{grid}-{density}',kind='density',n=grid*256,grid=grid,block=256,shared=0,mask=31,density=density,ptx=density_ptx()))
 return out

def scalar_params(case):
 names=Program(case['ptx']).params;n=case['n'];v={}
 if case['kind']=='rms':
  width=1<<(n-1).bit_length()
  for i,x in {2:n,3:n,4:case['grid']*n,5:case['grid']*n,8:0,9:0,10:0}.items():v[names[i]]=x
  for i in [11,12,13,14]:
   arr=[0,int(math.log2(width)),width] if i==11 else [0,0,1]
   for j,x in enumerate(arr):v[names[i]+('+'+str(j*4) if j else '')]=x
 elif case['kind']=='silu':
  for i in [3,4,5,6]:v[names[i]]=n
 else:v['mask']=case['mask']
 return v

class UInt3(C.Structure):_fields_=[('x',C.c_uint32),('y',C.c_uint32),('z',C.c_uint32)]
def arguments(case,ptrs):
 from core_driver import P
 n=case['n']
 if case['kind']=='rms':
  width=1<<(n-1).bit_length();a=[ptrs[0],ptrs[2],C.c_int(n),C.c_int64(n),C.c_int64(n*case['grid']),C.c_int64(n*case['grid']),C.c_float(1e-5),ptrs[1],C.c_int64(0),C.c_int64(0),C.c_int64(0),UInt3(0,int(math.log2(width)),width),UInt3(0,0,1),UInt3(0,0,1),UInt3(0,0,1),P(0),C.c_int64(0),C.c_int64(0),C.c_int64(0),UInt3(),UInt3(),UInt3(),UInt3(),C.c_float(1.)]
 elif case['kind']=='silu':a=[*ptrs[:3],*[C.c_int64(n) for _ in range(4)]]
 else:a=[ptrs[0],C.c_uint32(case['mask'])]
 return a
