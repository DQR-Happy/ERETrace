"""Same-file CPU replay: exact reconstructed raw vs ERE, outside formal data."""
from pathlib import Path
import json,struct,time,hashlib,cProfile,pstats,shutil
import zstandard as z
from case_analyze import analyze,frames
r=Path(__file__).resolve().parents[1];out=r/'consumer-profile';out.mkdir(exist_ok=True)
validation=r/'capture/r0-ispa_ere-validate';w=json.loads((validation/'workload.json').read_text())['requests'][0]
archive=next(validation.glob('trace-*.zst'));tmp=Path('/dev/shm/neutrino-case-consumer-profile');tmp.mkdir(exist_ok=True)
encoded=tmp/'encoded.nts';raw=tmp/'raw.bin'
with archive.open('rb') as f,z.ZstdDecompressor().stream_reader(f) as stream,encoded.open('wb') as g:shutil.copyfileobj(stream,g,8<<20)
rawsha=hashlib.sha256()
with raw.open('wb') as f:
 for _,dims,data in frames(encoded):
  header=struct.pack('<10IQ',*dims);f.write(header);f.write(data);rawsha.update(header);rawsha.update(data)
result=[];outputs=[]
for label,path in [('ere',encoded),('same_map_raw',raw)]:
 prof=cProfile.Profile();t=time.perf_counter_ns();prof.enable();answer=analyze([path],validation/'launches.tsv',w['request']);prof.disable();elapsed=time.perf_counter_ns()-t
 digest=hashlib.sha256(json.dumps(answer,sort_keys=True,separators=(',',':')).encode()).hexdigest();outputs.append(digest)
 prof.dump_stats(out/f'{label}.prof');stats=pstats.Stats(prof);rows=[]
 for (file,line,func),(cc,nc,tt,ct,callers) in stats.stats.items():rows.append({'file':file,'line':line,'function':func,'primitive_calls':cc,'calls':nc,'self_seconds':tt,'cumulative_seconds':ct})
 rows.sort(key=lambda x:x['cumulative_seconds'],reverse=True)
 result.append({'condition':label,'profiled_wall_ns':elapsed,'bytes':path.stat().st_size,'frames':answer['frames'],'records':answer['records'],'decoded_sha256':answer['decoded_sha256'],'answer_sha256':digest,'functions':rows})
 assert answer['decoded_sha256']==rawsha.hexdigest()
assert outputs[0]==outputs[1]
summary={'same_map_analysis_identical':True,'complete_raw_map_sha256':rawsha.hexdigest(),'source':str(archive),'replays':result,'purpose':'Consumer attribution and exact full-analysis equality, separate from formal timing; cProfile overhead included; no GPU work'}
(out/'profile.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'same_map_analysis_identical':True,'timings':[{k:x[k]for k in ['condition','profiled_wall_ns']}for x in result]}));shutil.rmtree(tmp)
