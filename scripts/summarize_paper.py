#!/usr/bin/env python3
"""Recreate paper-facing summary tables from retained scalar evidence (no GPU)."""
import argparse,csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'outputs/paper-summary');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    data=json.loads((ROOT/'results/paper/main-results.json').read_text())
    rows=data['primary'];tables={}
    for metric,label in [('mean_request_ms','End-to-end request latency (ms)'),('trace_file_GB_per_request_mean','Closed trace file size (decimal GB/request)')]:
        out=[]
        for model in dict.fromkeys(r['model'] for r in rows):
            entries={r['method_id']:float(r[metric]) for r in rows if r['model']==model}
            out.append({'model':model,**entries})
        tables[label]=out
        with (a.output/(metric+'.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=out[0].keys());w.writeheader();w.writerows(out)
    text=['# Retained primary results','', '256 input tokens / 64 output tokens, B=1, context=4096. Historical RTX 3090 measurements; no new GPU execution.','']
    for label,out in tables.items():
        text+=['## '+label,'','| Model | Native | Serial | +Async | +nvCOMP | Ours |','|---|---:|---:|---:|---:|---:|']
        text+=['| '+r['model']+' | '+' | '.join(f'{r[k]:.3f}' for k in ['N','U','A','NC','R'])+' |' for r in out]
        text+=['']
    (a.output/'report.md').write_text('\n'.join(text)+'\n');print(a.output/'report.md')
if __name__=='__main__':main()
