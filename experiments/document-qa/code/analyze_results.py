"""Paired full-workflow estimates, no profile-time subtraction."""
from pathlib import Path
import csv,json,hashlib,sys
import numpy as np
r=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[1]
blocks=json.loads((r/'allocation.json').read_text())['blocks']
methods=['serial_neutrino','async_raw','nvcomp_cascaded','ispa_ere']
rows=[];checks=[]
for b in blocks:
 n=b['block'];native=[];outputs=[];scopes=[];records=[]
 for arm in ['native_before','native_after']:
  req=json.loads((r/'capture'/f'r{n}-{arm}'/'workload.json').read_text())['requests'][0]
  native.append(req['inference_ns']);outputs.append((req['prompt_id'],req['input_ids'],req['output_ids']))
 inference=sum(native)/2
 for method in methods:
  out=r/'capture'/f'r{n}-{method}'
  assert (out/'complete.json').exists(),out
  req=json.loads((out/'workload.json').read_text())['requests'][0]
  analysis=json.loads((out/'analysis-0.json').read_text())
  outputs.append((req['prompt_id'],req['input_ids'],req['output_ids']))
  scope=[(x['token'],x['name'],x['dims'],x['records']) for x in analysis['launches']]
  scopes.append(hashlib.sha256(json.dumps(scope,separators=(',',':')).encode()).hexdigest())
  records.append(analysis['records'])
  assert req['answer_ns']==req['capture_ns']+req['analysis_ns']
  assert 'decoded_sha256' not in analysis
  vals=dict(block=n,method=method,prompt=req['prompt_id'],inference_ms=inference/1e6,
            capture_increment_ms=(req['capture_ns']-inference)/1e6,
            analysis_ms=req['analysis_ns']/1e6,total_ms=req['answer_ns']/1e6,
            capture_ms=req['capture_ns']/1e6,instrumented_execution_ms=req['inference_ns']/1e6,
            capture_tail_ms=req['drain_ns']/1e6,trace_bytes=req['trace_bytes'],frames=analysis['frames'],records=analysis['records'],
            native_before_ms=native[0]/1e6,native_after_ms=native[1]/1e6,
            process_max_rss_bytes=json.loads((out/'workload.json').read_text())['max_rss_bytes'],
            sampled_gpu_peak_mib=max(float(x['gpu'].split(',')[1].strip()) for x in json.loads((out/'resource-samples.json').read_text()) if 'gpu' in x))
  rows.append(vals)
 assert all(x==outputs[0] for x in outputs),('Answer token mismatch',n)
 assert len(set(scopes))==1,('Scope mismatch',n)
 assert len(set(records))==1
 checks.append({'block':n,'identical_input_and_output_tokens':True,'identical_full_scope':True,'scope_sha256':scopes[0],'records':records[0],'frames':analysis['frames']})
means={m:{k:float(np.mean([x[k] for x in rows if x['method']==m])) for k in ['inference_ms','capture_increment_ms','analysis_ms','total_ms','capture_ms','instrumented_execution_ms','capture_tail_ms','trace_bytes','process_max_rss_bytes','sampled_gpu_peak_mib']} for m in methods}
rng=np.random.default_rng(2026101318);indices=rng.integers(0,len(blocks),size=(50000,len(blocks)))
effects={}
ere=np.array([x['total_ms'] for x in rows if x['method']=='ispa_ere'])
for m in methods[:-1]:
 base=np.array([x['total_ms'] for x in rows if x['method']==m]);diff=base-ere
 ci=np.percentile(diff[indices].mean(axis=1),[2.5,97.5]);speed=base[indices].mean(axis=1)/ere[indices].mean(axis=1)
 effects[m]={'baseline_minus_ere_mean_ms':float(diff.mean()),'paired_differences_ms':diff.tolist(),'ere_faster_blocks':int(np.count_nonzero(diff>0)),'blocks':len(diff),'descriptive_bootstrap_ci95_ms':ci.tolist(),'mean_speedup':float(base.mean()/ere.mean()),'descriptive_bootstrap_speedup_ci95':np.percentile(speed,[2.5,97.5]).tolist()}
validation=[]
for m in methods:
 validation.append({'method':m,**json.loads((r/'capture'/f'r1-{m}'/'validation.json').read_text())})
summary={'n_paired_blocks':len(blocks),'primary_metric':'direct measured complete inference+capture+analysis, excluding added verification; no profile subtraction','decomposition':'native matched inference + incremental capture wall time + real analysis; overlaps retained','means':means,'ere_vs_baselines':effects,'block_scope_and_answers':checks,'independent_validation':validation,'winner_by_measured_mean':min(means,key=lambda m:means[m]['total_ms']),'small_n_note':'4 blocks describe this exact task/configuration; bootstrap intervals exploratory, not cross-workload significance'}
(r/'statistics.json').write_text(json.dumps(summary,indent=2)+'\n')
with (r/'measurements.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
np.save(r/'bootstrap-indices.npy',indices)
print(json.dumps({'means':means,'effects':effects,'winner':summary['winner_by_measured_mean']},indent=2))
