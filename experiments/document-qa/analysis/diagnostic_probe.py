"""Base warp trace everywhere, supplemental exact memory/branch events."""
from neutrino import Map,probe
import neutrino.language as nl

@Map(level='warp',type='array',size=16,cap=1)
class block_sched:
    start:nl.u64
    elapsed:nl.u32
    cuid:nl.u32

@Map(level='thread',type='array',size=16,cap='dynamic')
class DMAT:
    clock:nl.u64
    addr:nl.u64

warp_start:nl.u64=0
thread_start:nl.u64=0
elapsed:nl.u64=0
mem_clock:nl.u64=0
branch_tag:nl.u64=0
saved_address:nl.u64=0

@probe(pos='kernel',level='warp',before=True)
def start_warp():
    warp_start=nl.clock()

@probe(pos='kernel',level='thread',before=True)
def start_thread():
    thread_start=nl.clock()
    branch_tag=thread_start-thread_start

@probe(pos='kernel',level='warp')
def end_warp():
    elapsed=nl.clock()-warp_start
    block_sched.save(warp_start,elapsed,nl.cuid())

@probe(pos='ld.global:st.global:cp.async.cg:cp.async.ca',level='thread')
def memory_access():
    mem_clock=nl.clock()-thread_start
    DMAT.save(mem_clock,saved_address)

@probe(pos='ld.global:st.global:cp.async.cg:cp.async.ca',level='thread',before=True)
def preserve_address():
    saved_address=nl.addr

@probe(pos='bra',level='thread',before=True)
def branch_progress():
    mem_clock=nl.clock()-thread_start
    DMAT.save(mem_clock,branch_tag)
