"""Use a reproduced launch-bound certificate to emit private event addressing."""
import math
from .core_bounds import check_certificate,Unsupported
from .core_events import instrument_memory_events

def compile_layout(ptx,certificate,params,grid,block,mode='classes'):
    check_certificate(ptx,certificate,params,grid,block)
    caps=certificate['capacities'];maximum=max(caps)
    if maximum==0:raise Unsupported('launch has no events')
    if mode not in ['uniform','table','classes']:raise Unsupported('layout mode')
    code,sites=instrument_memory_events(ptx,maximum)
    if mode=='uniform':
        anchor=f'mul.wide.u32 %obs_off, %buf1, {maximum*32};'
        if code.count(anchor)!=1:raise Unsupported('uniform address template')
        code=code.replace('.param .u64 obs_counts','.param .u64 obs_plan,\n.param .u64 obs_counts',1)
        entry='''.reg .u64 %obs_capacity, %obs_slot, %obs_plan_ptr, %obs_tid64;
ld.param.u64 %obs_plan_ptr, [obs_plan];
ld.global.v2.u64 {%obs_slot, %obs_capacity}, [%obs_plan_ptr];
cvt.u64.u32 %obs_tid64, %buf1;
mul.lo.u64 %obs_off, %obs_tid64, %obs_capacity;
add.u64 %obs_off, %obs_off, %obs_slot;
shl.b64 %obs_off, %obs_off, 5;'''
        code=code.replace(anchor,entry).replace(f'setp.lt.u64 %obs_space, %obs_index, {maximum};','setp.lt.u64 %obs_space, %obs_index, %obs_capacity;')
        return code,dict(plan=[(0,maximum)],offsets=[i*maximum for i in range(len(caps))],caps=[maximum]*len(caps),slots=len(caps)*maximum,patterns=[])
    offsets=[];total=0
    for c in caps:offsets.append(total);total+=c
    if mode=='table':plan=list(zip(offsets,caps));patterns=[]
    else:
        size=math.prod(block);patterns=[];plan=[]
        for i in range(0,len(caps),size):
            row=caps[i:i+size]
            if row not in patterns:patterns.append(row)
            plan.append((offsets[i],patterns.index(row)))
    anchor=f'mul.wide.u32 %obs_off, %buf1, {maximum*32};'
    if code.count(anchor)!=1:raise Unsupported('address template mismatch')
    code=code.replace('.param .u64 obs_counts','.param .u64 obs_plan,\n.param .u64 obs_counts',1)
    entry='''.reg .u64 %obs_capacity, %obs_slot, %obs_plan_ptr, %obs_plan_off;
ld.param.u64 %obs_plan_ptr, [obs_plan];
'''
    if mode=='table':entry+='''mul.wide.u32 %obs_plan_off, %buf1, 16;
add.u64 %obs_plan_ptr, %obs_plan_ptr, %obs_plan_off;
ld.global.v2.u64 {%obs_slot, %obs_capacity}, [%obs_plan_ptr];
'''
    else:
        entry+='''.reg .u64 %obs_pid, %obs_local_slot, %obs_tmp, %obs_rel;
.reg .pred %obs_match, %obs_lo, %obs_hi, %obs_class;
mul.wide.u32 %obs_plan_off, %buf17, 16;
add.u64 %obs_plan_ptr, %obs_plan_ptr, %obs_plan_off;
ld.global.v2.u64 {%obs_slot, %obs_pid}, [%obs_plan_ptr];
mov.u64 %obs_local_slot, 0;
mov.u64 %obs_capacity, 0;
'''
        for pi,row in enumerate(patterns):
            start=prefix=0
            while start<len(row):
                end=start+1
                while end<len(row) and row[end]==row[start]:end+=1
                c=row[start]
                entry+=f'''setp.eq.u64 %obs_class, %obs_pid, {pi};
setp.ge.u32 %obs_lo, %buf15, {start};
setp.lt.u32 %obs_hi, %buf15, {end};
and.pred %obs_match, %obs_lo, %obs_hi;
and.pred %obs_match, %obs_match, %obs_class;
cvt.u64.u32 %obs_rel, %buf15;
sub.u64 %obs_rel, %obs_rel, {start};
mul.lo.u64 %obs_tmp, %obs_rel, {c};
add.u64 %obs_tmp, %obs_tmp, {prefix};
selp.u64 %obs_local_slot, %obs_tmp, %obs_local_slot, %obs_match;
selp.u64 %obs_capacity, {c}, %obs_capacity, %obs_match;
'''
                prefix+=(end-start)*c;start=end
        entry+='add.u64 %obs_slot, %obs_slot, %obs_local_slot;\n'
    entry+='shl.b64 %obs_off, %obs_slot, 5;'
    code=code.replace(anchor,entry).replace(f'setp.lt.u64 %obs_space, %obs_index, {maximum};','setp.lt.u64 %obs_space, %obs_index, %obs_capacity;')
    return code,dict(plan=plan,offsets=offsets,caps=caps,slots=total,patterns=patterns)
