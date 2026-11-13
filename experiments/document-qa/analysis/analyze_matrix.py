"""Paired process-level estimates and full-scope / retention audit."""
from pathlib import Path
import json,csv,hashlib,collections
import numpy as np
r=Path(__file__).resolve().parents[1];capture=r/'capture';conds=['serial_neutrino','async_raw','nvcomp_cascaded','ispa_ere']
native={x['prompt_id']:x for x in json.loads((r/'native-heldout/workload.json').read_text())['requests']}
records=[];checks=[];identities={}
for block in range(1,8):
 for condition in conds:
  d=capture/f'r{block}-{condition}'
  if not (d/'complete.json').exists() or not (d/'retention.json').exists():continue
  w=json.loads((d/'workload.json').read_text())['requests'][0]
  analysis=json.loads((d/'analysis-0.json').read_text());ab=json.loads((d/'ablation.json').read_text());summary=json.loads((d/'capture-summary.json').read_text());ret=json.loads((d/'retention.json').read_text())
  seq=hashlib.sha256()
  for x in analysis['launches']:seq.update(json.dumps([x['token'],x['name'],x['dims'],x['records']],separators=(',',':')).encode())
  identity=seq.hexdigest();identities.setdefault(block,set()).add(identity)
  e=summary.get('arena_engines',[{}])[0]
  metadata_files=['launches.tsv','analysis-0.json','capture-summary.json','ablation.json','workload.json','resource-samples.json']
  metadata_bytes=sum((d/x).stat().st_size for x in metadata_files)
  resources=json.loads((d/'resource-samples.json').read_text());gpu=[float(x['gpu'].split(',')[1])for x in resources if 'gpu' in x]
  row={'round':block,'condition':condition,'prompt_id':w['prompt_id'],'input_tokens':w['input_tokens'],'output_tokens':w['output_tokens'],'frames':analysis['frames'],'records':analysis['records'],**{k:w[k]/1e6 for k in ['prefill_ns','decode_ns','capture_ns','drain_ns','analysis_ns','answer_ns']},'trace_bytes':w['trace_bytes'],'metadata_bytes':metadata_bytes,'trace_plus_metadata_bytes':w['trace_bytes']+metadata_bytes,'max_gpu_mib':max(gpu),'max_process_rss_bytes':json.loads((d/'workload.json').read_text())['max_rss_bytes'],'collector_host_capacity_bytes':e.get('host_capacity_bytes',0),'collector_device_capacity_bytes':e.get('device_capacity_bytes',0),'encoder_device_workspace_bytes':e.get('nvcomp_device_workspace',0)+e.get('delta_scratch_bytes',0),'encoder_host_workspace_bytes':e.get('nvcomp_host_workspace',0),'transfer_bytes':e.get('transfer_bytes',sum(g['d2h_bytes']for g in ab['groups'])),'encoded_frames':e.get('encoder_frames',0),'fallback_frames':e.get('delta_fallbacks',0),'copies':e.get('copies',sum(g['calls']for g in ab['groups'])),'encoder_calls':e.get('encoders',0)}
  for k in ['prefill_ns','decode_ns','capture_ns','drain_ns','analysis_ns','answer_ns']:row[k.replace('_ns','_ms')]=row.pop(k)
  records.append(row)
  check={'round':block,'condition':condition,'input_equal_native':w['input_ids']==native[w['prompt_id']]['input_ids'],'output_equal_native':w['output_ids']==native[w['prompt_id']]['output_ids'],'scope_digest':identity,'capture_errors':sum(g['errors']for g in ab['groups']),'original_or_pruned_or_excluded':sum(g['original']+g['pruned']+g['excluded']for g in ab['groups']),'probed_calls':sum(g['probed']for g in ab['groups']),'frames':analysis['frames'],'all_archives_roundtrip_exact':all(x['roundtrip_exact']for x in ret),'all_archives_exist_local_or_remote':all(Path(x['retained_path']).exists()if x['retained_host']=='local' else True for x in ret)}
  assert check['input_equal_native'] and check['output_equal_native'] and check['capture_errors']==0 and check['original_or_pruned_or_excluded']==0 and check['frames']==check['probed_calls'] and check['all_archives_roundtrip_exact'] and check['all_archives_exist_local_or_remote'],check
  checks.append(check)
assert all(len(x)==1 for x in identities.values()),'scope differs within paired block'
with (r/'measurements.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
complete=len(records)==28
audit={'complete':complete,'formal_processes':len(records),'scope_equal_within_all_blocks':True,'checks':checks}
(r/'correctness-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
if not complete:
 print(json.dumps({'complete':False,'formal_processes':len(records)}));raise SystemExit(0)
metrics=['capture_ms','analysis_ms','answer_ms','trace_bytes','trace_plus_metadata_bytes','max_gpu_mib','max_process_rss_bytes']
means={c:{m:float(np.mean([x[m]for x in records if x['condition']==c]))for m in metrics}for c in conds}
rng=np.random.default_rng(20261014);indices=rng.integers(0,7,size=(10000,7));by={(x['round'],x['condition']):x for x in records};comparisons=[]
for baseline in conds[:-1]:
 for metric in ['capture_ms','analysis_ms','answer_ms','trace_bytes','trace_plus_metadata_bytes']:
  b=np.array([by[k,baseline][metric]for k in range(1,8)]);m=np.array([by[k,'ispa_ere'][metric]for k in range(1,8)])
  absolute=b-m;relative=100*(b-m)/b;bootstrap=relative[indices].mean(axis=1)
  comparisons.append({'baseline':baseline,'method':'ispa_ere','metric':metric,'n_paired_processes':7,'mean_baseline':b.mean(),'mean_method':m.mean(),'paired_mean_absolute_reduction':absolute.mean(),'paired_mean_percent_reduction':relative.mean(),'ci95_percent_reduction':np.quantile(bootstrap,[.025,.975]).tolist(),'paired_percent_reductions':relative.tolist(),'practical_5_percent_gate':bool(metric=='answer_ms' and relative.mean()>=5 and np.quantile(bootstrap,.025)>0)})
out={'complete':True,'unit':'independent process block; same prompt paired within block','n_blocks':7,'bootstrap_draws':10000,'bootstrap_seed':20261014,'mean_results':means,'comparisons':comparisons,'resource_ranges':{c:{m:[min(x[m]for x in records if x['condition']==c),max(x[m]for x in records if x['condition']==c)]for m in ['max_gpu_mib','max_process_rss_bytes','encoder_device_workspace_bytes','collector_host_capacity_bytes','collector_device_capacity_bytes']}for c in conds},'notes':['No warmup/pilot/cold-JIT diagnostic results in formal estimates','Archive zstd bytes are retention cost only, never method compression gain','Confidence intervals describe seven process blocks on one GPU and three related documentation questions; not application-wide generalization','No post-hoc deletion of slow valid samples']}
(r/'statistics.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'complete':True,'mean_results':means,'comparisons':[x for x in comparisons if x['metric']in ['answer_ms','capture_ms','trace_bytes']]},indent=2))
