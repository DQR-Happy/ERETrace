"""Run one preregistered capture process, archive every formal trace exactly."""
from pathlib import Path
import os,sys,json,subprocess,time,hashlib,shutil,argparse,threading
import zstandard as z

p=argparse.ArgumentParser();p.add_argument('condition',choices=['serial_neutrino','async_raw','nvcomp_cascaded','ispa_ere']);p.add_argument('round',type=int);p.add_argument('--validate',action='store_true');a=p.parse_args()
r=Path(__file__).resolve().parents[1];src=r/'source';name=f'r{a.round}-{a.condition}'+('-validate' if a.validate else '')
out=r/'capture'/name;out.mkdir(parents=True,exist_ok=False)
trace=Path('/dev/shm/neutrino-case-capture')/name;trace.mkdir(parents=True,exist_ok=False)
env=dict(os.environ,PATH='/usr/local/cuda/bin:/root/miniconda3/bin:'+os.environ.get('PATH',''),
 LD_LIBRARY_PATH='/root/autodl-tmp/ispa-3090new-cpu-build/bin:/root/autodl-tmp/neutrino-review-sm-only-20261009/llama-jetson-build/bin',
 LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libstdc++.so.6',CASE_BACKEND=str(r/'backend'),CASE_BATCH='512',
 CASE_ANALYZE='1',CASE_DISCARD_WARMUP='1',CASE_PROMPT_INDEX=str((a.round-1)%3),NEUTRINO_BENCH_THREADS='8',
 NEUTRINO_CASE_SIDECAR=str(out/'launches.tsv'),NEUTRINO_CAPTURE_CLEAR='1',NEUTRINO_CORE_MODE='shift',
 NEUTRINO_ABLATION_STATS='1',NEUTRINO_NVPIPE_LIBRARY=str(src/'benchmarks/nvcomp_pipeline.so'),
 NEUTRINO_COMPILER_CACHE_ROOT=str(r/'compiler-cache'))
if a.condition=='serial_neutrino':env.update(NEUTRINO_ABLATION_WRITER='1',NEUTRINO_ABLATION_FILE_BUFFER='1048576',NEUTRINO_CAPTURE_CLEAR='0')
else:
 env.update(NEUTRINO_ARENA='1',NEUTRINO_CAPTURE_ENCODER=str(src/'benchmarks/capture_pack.cubin'),
  NEUTRINO_ARENA_HOST_MIB='128',NEUTRINO_ARENA_DEVICE_MIB='64',NEUTRINO_ARENA_CHUNK_MIB='64',
  NEUTRINO_ARENA_INCREMENTAL='1',NEUTRINO_ARENA_BATCH_ENCODER='1',NEUTRINO_ARENA_DECOUPLE='0',
  NEUTRINO_ARENA_TOKEN_FLUSH='1',NEUTRINO_ARENA_WRITER='1',NEUTRINO_ARENA_HOTPATH='3',NEUTRINO_ARENA_CODEC_MIN_KIB='16')
 if a.condition=='async_raw':env['NEUTRINO_CASE_RAW']='1'
 if a.condition=='nvcomp_cascaded':env['NEUTRINO_ARENA_NVCOMP']='2';env['NEUTRINO_NVCOMP_CHUNK_KIB']='64'
 if a.condition=='ispa_ere':env.update(NEUTRINO_ARENA_DELTA='1',NEUTRINO_ARENA_DELTA_WIDTH='3',NEUTRINO_ARENA_ELAPSED_WIDTH='2',NEUTRINO_ARENA_HIER='1',NEUTRINO_ARENA_REFERENCE='1',NEUTRINO_ARENA_LOCAL_BITS='12',NEUTRINO_ARENA_OMIT_ANCHOR='1')
 if a.validate:env.update(NEUTRINO_ARENA_VERIFY='1',NEUTRINO_CAPTURE_CLEAR='2',NEUTRINO_ARENA_DIAG='1')
cmd=[sys.executable,'-m','neutrino.cli','-p','block_sched','--callback','/dev/null','--python',str(r/'cache-python'),'--ablation-mode','files','--tracedir',str(trace),sys.executable,str(r/'code/case_workload.py'),'--root',str(r),'--out',str(out),'--prompts',str(r/'code/prompts.json'),'--warmup','0' if a.validate else '2','--requests','1','--set','heldout']
(out/'command.json').write_text(json.dumps({'argv':cmd,'env':{k:v for k,v in env.items() if k.startswith(('CASE_','NEUTRINO_','GGML_','CUDA_')) or k in ['PATH','LD_LIBRARY_PATH','LD_PRELOAD','PYTHONPATH']}},indent=2)+'\n')
stop=threading.Event();samples=[]
def sample():
 while not stop.is_set():
  try:
   gpu=subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem,power.draw','--format=csv,noheader,nounits'],text=True).strip()
   samples.append({'monotonic_ns':time.monotonic_ns(),'gpu':gpu,'tmpfs_used':shutil.disk_usage('/dev/shm').used})
  except Exception as e:samples.append({'error':str(e)})
  stop.wait(.2)
t=threading.Thread(target=sample);t.start();started=time.perf_counter_ns()
try:
 with (out/'log.txt').open('wb') as f:subprocess.run(cmd,cwd=src,env=env,stdout=f,stderr=f,check=True,timeout=900)
finally:
 stop.set();t.join();(out/'resource-samples.json').write_text(json.dumps(samples)+'\n')
work=json.loads((out/'workload.json').read_text());archives=[]
for req in work['requests']:
 for i,path in enumerate(map(Path,req['trace_paths'])):
  archive_dir=out
  # Retention happens after both measured endpoints. Balance the two disks
  # instead of filling the root disk while the data disk still has headroom.
  if shutil.disk_usage(out).free < (10<<30) and shutil.disk_usage('/root').free > shutil.disk_usage(out).free:
   archive_dir=Path('/root/neutrino-case-archives-20261013')/name;archive_dir.mkdir(parents=True,exist_ok=True)
  dest=archive_dir/f'trace-{req["request"]}-{i}{path.suffix}.zst';h=hashlib.sha256();tick=time.perf_counter_ns()
  with path.open('rb') as f,dest.open('wb') as g:
   with z.ZstdCompressor(level=5,threads=4).stream_writer(g) as w:
    for b in iter(lambda:f.read(8<<20),b''):h.update(b);w.write(b)
  check=hashlib.sha256()
  with dest.open('rb') as f,z.ZstdDecompressor().stream_reader(f) as reader:
   for b in iter(lambda:reader.read(8<<20),b''):check.update(b)
  assert h.hexdigest()==check.hexdigest()
  archives.append({'request':req['request'],'original_path':str(path),'original_bytes':path.stat().st_size,
   'original_sha256':h.hexdigest(),'archive':str(dest),'archive_bytes':dest.stat().st_size,'archive_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'roundtrip_exact':True,'archive_ns':time.perf_counter_ns()-tick})
  path.unlink()
(out/'archives.json').write_text(json.dumps(archives,indent=2)+'\n')
# Preserve logs and manifests for generated artifacts; compiler cache retains
# unique original/probed PTX and binaries. Remove only redundant transient copies.
generated=[]
for path in trace.rglob('*'):
 if path.is_file():
  generated.append({'relative':str(path.relative_to(trace)),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
  if path.suffix=='.log' or path.name.endswith('.json') or path.name=='kernel.info':
   dest=out/'runtime-metadata'/path.relative_to(trace);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
(out/'runtime-artifacts.json').write_text(json.dumps(generated,indent=2)+'\n')
shutil.rmtree(trace)
(out/'complete.json').write_text(json.dumps({'condition':a.condition,'round':a.round,'validation_only':a.validate,'process_wall_ns':time.perf_counter_ns()-started,'archives':archives},indent=2)+'\n')
print(json.dumps({'out':str(out),'capture_ms':work['requests'][0]['capture_ns']/1e6,'answer_ms':work['requests'][0]['answer_ns']/1e6,'trace_bytes':work['requests'][0]['trace_bytes'],'archive_bytes':sum(x['archive_bytes'] for x in archives)}),flush=True)
