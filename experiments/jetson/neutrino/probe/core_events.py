"""Experimental single-execution, bounded per-thread memory-event capture.

Supported input is one already-pruned PTX entry, with one instruction per line,
simple register +/- constant addresses, and ld.global/st.global. This is an
explicitly different observation mode from lane-0 block_sched. It records every
participating thread, has a finite capacity, and reports attempted event counts.
An overflow is a failed *complete* capture, never a silently complete trace.
"""
import re
from neutrino.probe.cuda import COMMON_BUFFER_CALC


def instrument_memory_events(ptx: str, capacity: int):
    if not 1 <= capacity <= 65536:
        raise ValueError("capacity must be in [1, 65536]")
    if len(re.findall(r"\.entry\s", ptx)) != 1:
        raise ValueError("provide one pruned entry")
    code_only=re.sub(r'/\*.*?\*/|//[^\n]*','',ptx,flags=re.S)
    if re.search(r'(?<![\w$])(?:%?obs_[\w$]*|OBS_SKIP_[\w$]*|%buf\d+)\b',code_only):
        raise ValueError('input uses reserved event-instrumentation names')
    start=ptx.index('(',ptx.index('.entry'));end=ptx.index(')',start)
    params=ptx[start+1:end].strip().rstrip(',')
    ptx=ptx[:start+1]+'\n'+params+(',' if params else '')+'\n.param .u64 obs_records,\n.param .u64 obs_counts\n'+ptx[end:]
    body=ptx.index('{',ptx.index(')',start))
    entry=COMMON_BUFFER_CALC+f'''
.reg .u64 %obs_base, %obs_count_base, %obs_count, %obs_off;
ld.param.u64 %obs_base, [obs_records];
ld.param.u64 %obs_count_base, [obs_counts];
mul.wide.u32 %obs_off, %buf1, {capacity*32};
add.u64 %obs_base, %obs_base, %obs_off;
mul.wide.u32 %obs_off, %buf1, 8;
add.u64 %obs_count_base, %obs_count_base, %obs_off;
mov.u64 %obs_count, 0;
'''
    lines=ptx[body+1:].splitlines();out=[];sites=[]
    pattern=re.compile(r'^\s*(?P<p>@!?%[\w$]+\s+)?(?P<op>(?:ld|st)\.global[.\w]*)\s+(?P<args>[^;]+);')
    for line in lines:
        code=line.split('//',1)[0]
        terminal=re.match(r'^\s*(@!?%[\w$]+\s+)?(?:ret|exit)(?:\.uni)?;',code)
        if re.search(r'\b(?:ret|exit)(?:\.uni)?\s*;',code) and not terminal:
            raise ValueError('terminal instruction must occupy its own line')
        if terminal:
            out.append((terminal[1] or '')+'st.global.u64 [%obs_count_base], %obs_count;')
        m=pattern.match(code)
        if not m:
            if re.search(r'\b(?:ld|st)\.global\b',code):
                raise ValueError('memory instruction must occupy its own line')
            out.append(line);continue
        if code[m.end():].strip():raise ValueError('multiple instructions on memory-event line')
        address=re.search(r'\[\s*(%[\w$]+)(?:\s*([+-])\s*(-?(?:0x[0-9a-fA-F]+|\d+)))?\s*\]',m['args'])
        bits=re.search(r'\.([busf])(8|16|32|64)$',m['op'])
        if not address or not bits:
            raise ValueError(f'unsupported memory operand: {line}')
        vector=re.search(r'\.v([24])\.',m['op']);width=int(bits[2])//8*(int(vector[1]) if vector else 1)
        site=len(sites);sites.append(dict(site=site,opcode=m['op'],width=width,address=address[0]))
        pred=(m['p'] or '').strip().lstrip('@')
        enabled=('not.pred %obs_enabled, '+pred[1:]+';' if pred.startswith('!') else 'mov.pred %obs_enabled, '+pred+';') if pred else 'setp.eq.u32 %obs_enabled, 0, 0;'
        addr_op=(('add' if address[2]=='+' else 'sub')+f'.u64 %obs_addr, {address[1]}, {address[3]};') if address[2] else f'mov.u64 %obs_addr, {address[1]};'
        out.append(f'''{{
.reg .pred %obs_enabled, %obs_space;
.reg .u32 %obs_active, %obs_mask, %obs_site, %obs_width;
.reg .u64 %obs_addr, %obs_time, %obs_index, %obs_dest;
{enabled}
activemask.b32 %obs_active;
vote.ballot.sync.b32 %obs_mask, %obs_enabled, %obs_active;
@!%obs_enabled bra OBS_SKIP_{site};
mov.u64 %obs_time, %clock64;
{addr_op}
mov.u64 %obs_index, %obs_count;
add.u64 %obs_count, %obs_count, 1;
setp.lt.u64 %obs_space, %obs_index, {capacity};
@!%obs_space bra OBS_SKIP_{site};
shl.b64 %obs_index, %obs_index, 5;
add.u64 %obs_dest, %obs_base, %obs_index;
mov.u32 %obs_site, {site};
mov.u32 %obs_width, {width};
st.global.v2.u64 [%obs_dest], {{%obs_time, %obs_addr}};
st.global.v4.u32 [%obs_dest+16], {{%obs_site, %buf1, %obs_mask, %obs_width}};
OBS_SKIP_{site}:
}}''')
        out.append(line)
    if not sites:
        raise ValueError('no supported memory sites')
    return ptx[:body+1]+'\n'+entry+'\n'+'\n'.join(out), sites
