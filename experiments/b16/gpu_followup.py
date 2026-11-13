"""Separate GPU Activity diagnosis after all unprofiled performance windows."""
import copy,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/root/autodl-tmp/neutrino-b16-attribution-20261007')
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
while True:
    try:s=json.loads((ROOT/'status.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):time.sleep(10);continue
    if any(m['status']=='failed' for m in s['matrices']):raise RuntimeError('Base study failed')
    if s['complete']:break
    time.sleep(10)
best=json.loads((ROOT/'screen-selection.json').read_text())['candidate']
env={k:v for k,v in os.environ.items() if not k.startswith('NEUTRINO_')}
env['PATH']='/root/miniconda3/bin:/usr/local/cuda/bin:'+env.get('PATH','')
env['PYTHONPATH']=str(ROOT/'source')+':'+str(ROOT/'collector-old/benchmarks')+':'+str(ROOT/'collector-old')
# Existing integer oracle: independently compare each encoded Map against the
# same completed raw Map. Extra copies and CPU decoding never enter performance.
screen=json.loads((ROOT/'b16-screen-specs.json').read_text())
vs={c:copy.deepcopy(screen[c]) for c in dict.fromkeys(['E24',best])}
for v in vs.values():v['env']['NEUTRINO_ARENA_VERIFY']='1'
save(ROOT/'codec-validation-specs.json',vs)
vc=json.loads((ROOT/'matrices/b16-confirm/config.json').read_text())['command']
vc[vc.index('--requests')+1]='1'
va=[sys.executable,str(ROOT/'source/run_arena_matrix.py'),'--output',str(ROOT/'codec-validation'),'--trace-root','/dev/shm/neutrino-b16-verify-20261007','--spec',str(ROOT/'codec-validation-specs.json'),'--conditions',*vs,'--rounds','1','--timeout','1200','--',*vc]
print('CODEC ORACLE START',flush=True);subprocess.run(va,env=env,check=True)
vt=Path('/dev/shm/neutrino-b16-verify-20261007');assert not list(vt.iterdir());vt.rmdir()
save(ROOT/'codec-validation-complete.json',dict(passed=True,conditions=list(vs),tmpfs_absent=True))
print('CODEC ORACLE COMPLETE',flush=True)
wl=(ROOT/'source/workload.py').read_text()
wl=wl.replace('handle = lib.qwen_create', '''deep=C.CDLL(os.environ['DEEP_CUPTI_LIBRARY'])
deep.deep_stop.argtypes=[C.c_char_p]
assert deep.deep_init()==0
handle = lib.qwen_create''',1)
wl=wl.replace('    usage0 = resource.getrusage','    assert deep.deep_start()==0\n    usage0 = resource.getrusage',1)
wl=wl.replace('    usage1 = resource.getrusage(resource.RUSAGE_SELF)','    usage1 = resource.getrusage(resource.RUSAGE_SELF)\n    assert deep.deep_stop(str(out/"cupti.json").encode())==0',1)
(ROOT/'source/gpu_workload.py').write_text(wl)
specs=json.loads((ROOT/'b16-confirm-specs.json').read_text())
env={k:v for k,v in os.environ.items() if not k.startswith('NEUTRINO_')}
env['PATH']='/root/miniconda3/bin:/usr/local/cuda/bin:'+env.get('PATH','')
env['PYTHONPATH']=str(ROOT/'source')+':'+str(ROOT/'collector-old/benchmarks')+':'+str(ROOT/'collector-old')
states=[]
for batch,rounds in [(4,1),(8,1),(16,2)]:
    name=f'b{batch}-gpu';sp=copy.deepcopy(specs)
    for c in sp:
        sp[c]['diagnostic']=False
        sp[c]['env'].update(NEUTRINO_BENCH_CONTEXT='8192',NEUTRINO_BENCH_SEQUENCES=str(batch),
          DEEP_CUPTI='1',DEEP_CUPTI_ACTIVITY='1',DEEP_CUPTI_CALLBACKS='0',NEUTRINO_PROFILE_COMPAT='1',
          DEEP_CUPTI_LIBRARY='/root/autodl-tmp/neutrino-paper-eval-20261005/regression-recheck-20261005/deep_gpuonly.so')
    save(ROOT/f'{name}-specs.json',sp)
    cmd=json.loads((ROOT/'matrices/b16-confirm/config.json').read_text())['command']
    cmd[1]=str(ROOT/'source/gpu_workload.py')
    cmd[cmd.index('--batch-size')+1]=str(batch);cmd[cmd.index('--requests')+1]='1'
    argv=[sys.executable,str(ROOT/'source/run_arena_matrix.py'),'--output',str(ROOT/'gpu'/name),'--trace-root','/dev/shm/neutrino-b16-gpu-20261007','--spec',str(ROOT/f'{name}-specs.json'),'--conditions',*sp,'--rounds',str(rounds),'--seed','21042','--timeout','1200','--',*cmd]
    (ROOT/'gpu').mkdir(exist_ok=True)
    state=dict(id=name,batch=batch,conditions=list(sp),rounds=rounds,status='running');states.append(state)
    save(ROOT/'gpu-status.json',dict(matrices=states,complete=False));print(name,'START',flush=True)
    try:subprocess.run(argv,env=env,check=True)
    except BaseException as exc:
        state.update(status='failed',error=str(exc));save(ROOT/'gpu-status.json',dict(matrices=states,complete=False));raise
    state['status']='complete';save(ROOT/'gpu-status.json',dict(matrices=states,complete=False));print(name,'COMPLETE',flush=True)
tr=Path('/dev/shm/neutrino-b16-gpu-20261007');assert not list(tr.iterdir());tr.rmdir()
save(ROOT/'gpu-status.json',dict(matrices=states,complete=True,tmpfs_absent=True))
print('GPU DIAGNOSIS COMPLETE',flush=True)
