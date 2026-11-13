"""Recompute the exact K/V address diagnosis from every original DMAT slot."""
from pathlib import Path
import json,hashlib
import numpy as np
r=Path(__file__).resolve().parents[1];calls=[];sets={'K':[],'V':[]};sources=[]
for token in [0,1,63]:
 p=r/f'diagnostic-v7/extracted/dmat-token{token}.npz'
 a=np.load(p);d=a['dmat'];pointers=a['pointers']
 addr=d['addr'][(d['clock']>0)&(d['addr']>0)];regions={}
 for name,i in [('K',1),('V',2)]:
  values=addr[(addr>=pointers[i])&(addr<pointers[i]+(16<<20))]
  unique,counts=np.unique(values,return_counts=True);sets[name].append(unique)
  regions[name]={'events':len(values),'unique_instruction_addresses':len(unique),'min_repetitions':int(counts.min()),'max_repetitions':int(counts.max()),'observed_issue_bytes_16B':16*len(values)}
 calls.append({'token':token,'regions':regions})
 sources.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
result={'calls':calls,'comparisons':{f'unique_{name}_addresses_equal_all_calls':bool(all(np.array_equal(values[0],x)for x in values[1:]))for name,values in sets.items()},'interpretation':'Instruction issue address repetitions, not cache misses or DRAM bytes'}
assert result==json.loads((r/'address-diagnosis.json').read_text())
(r/'address-diagnosis-verification.json').write_text(json.dumps({'passed':True,'source_files':sources,'all_original_slots_read':True,'branch_and_padding_not_memory_events':True,'notes':'16B issue-byte field follows the selected kernel K/V copy instruction width; not measured DRAM traffic.'},indent=2)+'\n')
print(json.dumps({'passed':True,'exact_address_diagnosis_recomputed':True}))
