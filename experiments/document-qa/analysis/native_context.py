"""Attribute native GPU activity only; do not substitute probe cycles for time."""
from pathlib import Path
import json, collections, csv
r=Path(__file__).resolve().parents[1]
a=json.loads((r/'native-cupti/cupti.json').read_text())
p=[x for x in json.loads((r/'native-cupti/cupti-phase.json').read_text()) if x['request']==0]
decode_start=next(x['stamp'] for x in p if x['token']==1)
def category(name):
 if 'flash_attn' in name:return 'attention_aux' if 'fixup' in name or 'combine' in name else 'attention'
 if 'mul_mat' in name:return 'matmul_matvec'
 if 'norm' in name:return 'norm'
 if 'copy' in name or 'cpy' in name:return 'copy'
 return 'other'
totals=collections.defaultdict(lambda:dict(launches=0,gpu_ns=0))
for x in a['raw_gpu_records']:
 if x['kind']!=10:continue
 g=a['kernels'][x['group']]
 key=('prefill' if x['start']<decode_start else 'decode')+'/'+category(g['name'])
 totals[key]['launches']+=1;totals[key]['gpu_ns']+=x['end']-x['start']
total=sum(x['gpu_ns'] for x in totals.values())
for x in totals.values():x['fraction_of_all_kernel_time']=x['gpu_ns']/total
groups=sorted(a['kernels'],key=lambda x:x['gpu_ns'],reverse=True)
out={'total_gpu_kernel_ns':total,'launches':sum(x['launches'] for x in totals.values()),'dropped':a['dropped'],'overflow':a['overflow'],'phase_operator_totals':totals,'top_kernel_groups':groups[:15], 'attention_groups':[x for x in groups if 'flash_attn' in x['name']], 'attention_resources':[x for x in a['function_resources'] if 'flash_attn_ext' in x['name']], 'method':'Native CUPTI activity duration; phase markers use CUPTI timestamps; diagnostic single full request, not formal capture timing'}
(r/'native-context.json').write_text(json.dumps(out,indent=2)+'\n')
with (r/'native-phase-operators.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['phase','operator','launches','gpu_ms','fraction'])
 for k,v in totals.items():w.writerow([*k.split('/'),v['launches'],v['gpu_ns']/1e6,v['fraction_of_all_kernel_time']])
print(json.dumps({'total_ms':total/1e6,'phase_operator_totals':totals},indent=2))
