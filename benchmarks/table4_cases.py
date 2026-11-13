"""Controlled PTX capacities plus two unmodified llama.cpp entries."""
import numpy as np
from certified_cases import cases as original_cases, scalar_params, arguments, density_ptx

def irregular_ptx():
    text=density_ptx()
    text=text.replace('.reg .u32 %t,%b,%n,%g,%v,%j,%lim;', '.reg .u32 %t,%b,%n,%g,%v,%j,%lim,%local,%class;\n.reg .pred %class1,%class2,%class3;')
    text=text.replace('mov.u32 %t, %tid.x;', 'mov.u32 %t, %tid.x;\nmov.u32 %local, %tid.x;')
    text=text.replace('and.b32 %lim, %v, %n;\nadd.u32 %lim, %lim, 1;', '''div.u32 %class, %local, 64;
setp.eq.u32 %class1, %class, 1;
setp.eq.u32 %class2, %class, 2;
setp.eq.u32 %class3, %class, 3;
mov.u32 %lim, 1;
selp.u32 %lim, 4, %lim, %class1;
selp.u32 %lim, 8, %lim, %class2;
selp.u32 %lim, 32, %lim, %class3;''')
    return text

def cases(inputs):
    old=original_cases(inputs)
    rms=next(c for c in old if c['kind']=='rms')
    silu=next(c for c in old if c['kind']=='silu')
    result=[dict(tag=f'irregular-{g}-256',kind='irregular',n=g*256,grid=g,block=256,shared=0,mask=31,density=0,ptx=irregular_ptx()) for g in [82,656]]
    result += [dict(rms,tag=f'rms-{n}',n=n) for n in [1024,1025]]
    result += [dict(silu,tag='silu-1025',n=1025,grid=5)]
    result += [dict(tag=f'density-656-{d}',kind='density',n=656*256,grid=656,block=256,shared=0,mask=31,density=d,ptx=density_ptx()) for d in [0,3,15,31]]
    return result

def event_counts(case):
    threads=case['grid']*case['block'];ids=np.arange(threads)
    if case['kind']=='rms':return 4*np.maximum(0,(case['n']-1-ids%case['block'])//case['block']+1)
    if case['kind']=='silu':return np.where(ids<case['n'],3,0)
    if case['kind']=='irregular':return np.asarray([3,9,17,65],dtype='int64')[(ids%256)//64]
    return np.full(threads,1+2*(case['density']+1),dtype='int64')
