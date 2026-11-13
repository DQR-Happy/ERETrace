"""Frozen collector, batch16 entry extension, paired controls and diagnostic scans."""
import copy, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT=Path('/root/autodl-tmp/neutrino-b16-attribution-20261007')
PREV=Path('/root/autodl-tmp/neutrino-length-batch-fair-20261007')
OLD=PREV/'collector-old'
BASE=Path('/root/autodl-tmp/neutrino-paper-eval-20261005')
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert not (ROOT/'started.json').exists(), 'Do not overwrite previous runs'
    save(ROOT/'started.json',{'time_ns':time.time_ns()})
    shutil.copytree(OLD,ROOT/'collector-old',ignore=shutil.ignore_patterns('__pycache__','.git','._*'))
    frozen=ROOT/'collector-old'
    expected=json.loads((PREV/'build-provenance.json').read_text())['expected_identical_files']
    assert all(digest(frozen/k)==v for k,v in expected.items())
    cpp=BASE/'runs/3090-unblocked-20261006/llama_adapter_extended.cpp'
    text=cpp.read_text();assert text.count('batch_size > 8')==3
    assert text.count('"NEUTRINO_BENCH_SEQUENCES", 1, 1, 8')==1
    text=text.replace('batch_size > 8','batch_size > 16').replace('"NEUTRINO_BENCH_SEQUENCES", 1, 1, 8','"NEUTRINO_BENCH_SEQUENCES", 1, 1, 16')
    newcpp=ROOT/'source/llama_adapter_batch16.cpp';newcpp.write_text(text)
    (ROOT/'build').mkdir()
    headers=BASE/'source/llama.cpp-master'
    backend=BASE/'llama-build/bin'
    build=['g++','-O2','-shared','-fPIC','-std=c++17',str(newcpp),'-I'+str(headers/'include'),'-I'+str(headers/'ggml/include'),'-L'+str(backend),'-Wl,-rpath,'+str(backend),'-lllama','-lggml','-o',str(ROOT/'build/libqwen_batch16.so')]
    save(ROOT/'adapter-build-command.json',build)
    subprocess.run(build,check=True)
    wl=(PREV/'source/exact_arena_workload.py').read_text()
    assert wl.count('a.batch_size > 8')==1
    wl=wl.replace('a.batch_size > 8','a.batch_size > 16').replace(str(OLD/'benchmarks'),str(frozen/'benchmarks'))
    # Trace distribution analysis runs only after fclose and outside latency.
    marker="    checks = [trace_check(path, False) for path in paths]"
    assert marker in wl
    wl=wl.replace(marker,marker+"\n    if os.environ.get('B16_TRACE_DISTRIBUTION') == '1':\n        from trace_distribution import collect\n        for path in paths: collect(path, out / 'trace-distribution.jsonl', label)")
    wl=wl.replace('def run_request():\n    t0', "def run_request():\n    if ablation and mark and os.environ.get('B16_TRACE_DISTRIBUTION') == '1':\n        assert mark(0) == 0\n    t0",1)
    workload=ROOT/'source/workload.py';workload.write_text(wl)
    shutil.copy2(PREV/'source/run_arena_matrix.py',ROOT/'source/run_arena_matrix.py')
    cfg=json.loads((PREV/'reference-main-config.json').read_text())
    cmd=cfg['command'][:];cmd[1]=str(workload)
    old_adapter=cmd[cmd.index('--adapter')+1]
    cmd[cmd.index('--adapter')+1]=str(ROOT/'build/libqwen_batch16.so')
    for flag,value in [('--input-tokens',256),('--output-tokens',64)]:cmd[cmd.index(flag)+1]=str(value)
    specs=copy.deepcopy(json.loads((PREV/'batch-b8-i256-o64-specs.json').read_text()))
    for spec in specs.values():
        spec['root']=str(frozen)
        if spec['mode']:spec['env']['NEUTRINO_CAPTURE_ENCODER']=str(frozen/'benchmarks/capture_pack.cubin')
    for name,changes in [('E24',{'NEUTRINO_ARENA_ELAPSED_WIDTH':'3'}),('C64',{'NEUTRINO_ARENA_CODEC_MIN_KIB':'64'}),('C128',{'NEUTRINO_ARENA_CODEC_MIN_KIB':'128'})]:
        specs[name]=copy.deepcopy(specs['R']);specs[name]['env'].update(changes)
    specs['NO']=copy.deepcopy(specs['N'])
    save(ROOT/'provenance.json',{'collector_expected_hashes':expected,'copied_collector':str(frozen),'collector_modified':False,'tiny_branch_absent':'TINY_SCOPE' not in (frozen/'neutrino/src/cuda.c').read_text(),'old_adapter_sha256':digest(Path(old_adapter)),'old_adapter_source_sha256':digest(cpp),'new_adapter_sha256':digest(ROOT/'build/libqwen_batch16.so'),'new_adapter_source_sha256':digest(newcpp),'adapter_changes':'Only three batch_size upper guards and env sequence upper guard 8 -> 16; same n_batch/n_ubatch=512, context logic, math and backend','assets':json.loads((PREV/'assets.json').read_text())})
    env={k:v for k,v in os.environ.items() if not k.startswith('NEUTRINO_')}
    env['PATH']='/root/miniconda3/bin:/usr/local/cuda/bin:'+env.get('PATH','')
    env['PYTHONPATH']=str(ROOT/'source')+':'+str(frozen/'benchmarks')+':'+str(frozen)
    states=[]
    def run(name,batch,conditions,rounds,diagnostic=False,requests=3,context=8192):
        sp=copy.deepcopy({c:specs[c] for c in conditions})
        for s in sp.values():
            s['env'].update(NEUTRINO_BENCH_CONTEXT=str(context),NEUTRINO_BENCH_SEQUENCES=str(batch))
            s['diagnostic']=diagnostic
            if diagnostic:s['env'].update(NEUTRINO_ARENA_DIAG='1',NEUTRINO_ABLATION_STATS='1',NEUTRINO_TIMING='1',B16_TRACE_DISTRIBUTION='1')
        command=cmd[:];command[command.index('--batch-size')+1]=str(batch);command[command.index('--requests')+1]=str(requests)
        if 'NO' in conditions:
            # Each method can point to its own workload command via an adapter selector.
            sp['NO']['env']['B16_OLD_ADAPTER']=old_adapter
        save(ROOT/f'{name}-specs.json',sp)
        state=dict(id=name,batch=batch,context=context,conditions=conditions,rounds=rounds,requests=requests,diagnostic=diagnostic,status='running',started_ns=time.time_ns());states.append(state)
        save(ROOT/'status.json',dict(matrices=states,complete=False))
        argv=[sys.executable,str(ROOT/'source/run_arena_matrix.py'),'--output',str(ROOT/'matrices'/name),'--trace-root','/dev/shm/neutrino-b16-attribution-20261007','--spec',str(ROOT/f'{name}-specs.json'),'--conditions',*conditions,'--rounds',str(rounds),'--seed','21042','--timeout','1200','--',*command]
        save(ROOT/f'{name}-launch.json',argv)
        print('MATRIX',name,'START',flush=True)
        try:subprocess.run(argv,env=env,check=True)
        except BaseException as exc:
            state.update(status='failed',error=str(exc));save(ROOT/'status.json',dict(matrices=states,complete=False));raise
        state.update(status='complete',ended_ns=time.time_ns());save(ROOT/'status.json',dict(matrices=states,complete=False))
        print('MATRIX',name,'COMPLETE',flush=True)
    # Keep the old backend and measure the guard-only rebuild on an already supported batch.
    wl=workload.read_text().replace("lib = C.CDLL(a.adapter)","lib = C.CDLL(os.environ.get('B16_OLD_ADAPTER', a.adapter))")
    workload.write_text(wl)
    (ROOT/'matrices').mkdir()
    run('adapter-b8-control',8,['NO','N'],1)
    # Candidate screen is separate from independent confirmation.
    run('b16-screen',16,['N','NC','R','E24','C64','C128'],2)
    rows=json.loads((ROOT/'matrices/b16-screen/runs.json').read_text())
    import statistics
    means={c:statistics.mean(x['wall_ns']/len(x['requests']) for x in rows if x['condition']==c) for c in ['N','NC','R','E24','C64','C128']}
    best=min(['E24','C64','C128'],key=means.get)
    save(ROOT/'screen-selection.json',{'means_ns':means,'candidate':best,'rule':'Lowest screen mean among three prespecified candidates; fresh confirmation follows'})
    run('b16-confirm',16,['N','NC','R',best],7)
    # Common context/adapter controls permit batch trend analysis, without cross-date pairing.
    run('b4-control',4,['N','NC','R'],3)
    run('b8-control',8,['N','NC','R'],3)
    for batch in [4,8,16]:run(f'b{batch}-diagnostic',batch,['NC','R',best],2,True,requests=1)
    assert all(digest(frozen/k)==v for k,v in expected.items())
    tr=Path('/dev/shm/neutrino-b16-attribution-20261007');assert not list(tr.iterdir());tr.rmdir()
    save(ROOT/'cleanup.json',{'task_tmpfs_absent':not tr.exists(),'raw_trace_exported':False})
    save(ROOT/'status.json',dict(matrices=states,complete=True,ended_ns=time.time_ns()))
    print('ALL MATRICES COMPLETE',flush=True)
if __name__=='__main__':main()
