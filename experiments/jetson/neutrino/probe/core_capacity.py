"""Bounded event budgets and exact private partitions for periodic thread classes.

This is NOT a static analyzer for arbitrary PTX. Budgets are supplied by a
restricted event description. Runtime attempted counts remain authoritative:
an incorrect bound produces an explicitly incomplete capture, not an overwrite.
"""
import math
from .core_events import instrument_memory_events

def event_bound(node,residue):
    tag=node[0]
    if tag=='events':
        n=node[1]
        if not isinstance(n,int) or n<0:raise ValueError('negative/noninteger event count')
        return n
    if tag=='seq':return sum(event_bound(n,residue) for n in node[1:])
    if tag=='repeat':
        if not isinstance(node[1],int) or node[1]<0:raise ValueError('unbounded loop')
        return node[1]*event_bound(node[2],residue)
    if tag=='residue':
        mod,values,yes,no=node[1:]
        if not isinstance(mod,int) or mod<1 or mod>32 or mod&(mod-1):raise ValueError('power-of-two residue modulus required')
        if any(not isinstance(v,int) or v<0 or v>=mod for v in values):raise ValueError('invalid residue')
        return event_bound(yes if residue%mod in values else no,residue)
    if tag=='unknown':return max(event_bound(n,residue) for n in node[1:])
    raise ValueError('unsupported event-bound construct')

def infer_period(node):
    if node[0]=='events':return 1
    if node[0]=='repeat':return infer_period(node[2])
    if node[0]=='residue':return max(node[1],infer_period(node[3]),infer_period(node[4]))
    if node[0] in ('seq','unknown'):return max([1]+[infer_period(n) for n in node[1:]])
    raise ValueError('unsupported event-bound construct')

def capacity_classes(node):
    p=infer_period(node)
    caps=[event_bound(node,r) for r in range(p)]
    validate_caps(caps)
    return caps

def validate_caps(caps):
    p=len(caps)
    if p<1 or p>32 or p&(p-1):raise ValueError('1..32 power-of-two classes required')
    if any(not isinstance(c,int) or not 0<=c<=65536 for c in caps) or max(caps)==0:
        raise ValueError('invalid capacities')

def private_layout(caps,block_threads,blocks):
    validate_caps(caps)
    if block_threads<1 or block_threads>1024 or blocks<1 or blocks*block_threads>=2**31:
        raise ValueError('launch index outside supported range')
    prefix=[0]
    for c in caps:prefix.append(prefix[-1]+c)
    p=len(caps);stride=(block_threads//p)*prefix[-1]+prefix[block_threads%p]
    rows=[(b*stride+(t//p)*prefix[-1]+prefix[t%p],caps[t%p])
          for b in range(blocks) for t in range(block_threads)]
    return rows,stride*blocks

def instrument_stratified_events(ptx,caps,layout='periodic'):
    validate_caps(caps)
    if layout not in ('periodic','table'):raise ValueError(layout)
    code,sites=instrument_memory_events(ptx,max(caps))
    anchor=f'mul.wide.u32 %obs_off, %buf1, {max(caps)*32};'
    if code.count(anchor)!=1:raise ValueError('unexpected private-address template')
    extra='.reg .u64 %obs_capacity, %obs_slot;\n'
    if layout=='table':
        code=code.replace('.param .u64 obs_counts','.param .u64 obs_plan,\n.param .u64 obs_counts',1)
        extra+='''.reg .u64 %obs_plan_ptr, %obs_plan_off;
ld.param.u64 %obs_plan_ptr, [obs_plan];
mul.wide.u32 %obs_plan_off, %buf1, 16;
add.u64 %obs_plan_ptr, %obs_plan_ptr, %obs_plan_off;
ld.global.v2.u64 {%obs_slot, %obs_capacity}, [%obs_plan_ptr];
'''
    else:
        p=len(caps);shift=int(math.log2(p));prefix=[0]
        for c in caps:prefix.append(prefix[-1]+c)
        global_decl=(f'.const .align 8 .u64 obs_class_caps[{p}] = {{{",".join(map(str,caps))}}};\n'
                     f'.const .align 8 .u64 obs_class_prefix[{p+1}] = {{{",".join(map(str,prefix))}}};\n')
        at=code.find('.visible .entry')
        if at<0:at=code.find('.entry')
        code=code[:at]+global_decl+code[at:]
        # buf15=local thread index; buf17=CTA index; buf19=CTA threads.
        extra+=f'''.reg .u32 %obs_r, %obs_q, %obs_tail, %obs_groups;
.reg .u64 %obs_lookup, %obs_prefix, %obs_stride, %obs_tmp, %obs_const;
and.b32 %obs_r, %buf15, {p-1};
shr.u32 %obs_q, %buf15, {shift};
and.b32 %obs_tail, %buf19, {p-1};
shr.u32 %obs_groups, %buf19, {shift};
mul.wide.u32 %obs_lookup, %obs_r, 8;
mov.u64 %obs_const, obs_class_caps;
add.u64 %obs_const, %obs_const, %obs_lookup;
ld.const.u64 %obs_capacity, [%obs_const];
mov.u64 %obs_const, obs_class_prefix;
add.u64 %obs_const, %obs_const, %obs_lookup;
ld.const.u64 %obs_prefix, [%obs_const];
mul.wide.u32 %obs_lookup, %obs_tail, 8;
mov.u64 %obs_const, obs_class_prefix;
add.u64 %obs_const, %obs_const, %obs_lookup;
ld.const.u64 %obs_stride, [%obs_const];
mul.wide.u32 %obs_tmp, %obs_groups, {prefix[-1]};
add.u64 %obs_stride, %obs_stride, %obs_tmp;
cvt.u64.u32 %obs_tmp, %buf17;
mul.lo.u64 %obs_slot, %obs_tmp, %obs_stride;
mul.wide.u32 %obs_tmp, %obs_q, {prefix[-1]};
add.u64 %obs_slot, %obs_slot, %obs_tmp;
add.u64 %obs_slot, %obs_slot, %obs_prefix;
'''
    extra+='shl.b64 %obs_off, %obs_slot, 5;'
    code=code.replace(anchor,extra)
    code=code.replace(f'setp.lt.u64 %obs_space, %obs_index, {max(caps)};',
                      'setp.lt.u64 %obs_space, %obs_index, %obs_capacity;')
    return code,sites
