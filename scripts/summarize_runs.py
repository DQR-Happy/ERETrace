#!/usr/bin/env python3
"""Summarize complete paired runs; resample process rounds, never individual requests."""
import argparse,csv,json,random,statistics
from pathlib import Path

def summarize(rows):
    by={}
    for r in rows:
        key=r['condition'];round_=int(r['round'])
        if round_ in by.setdefault(key,{}):raise ValueError('Duplicate condition/round')
        if not r['requests']:raise ValueError('Empty request list')
        by[key][round_]=r['wall_ns']/len(r['requests'])/1e6
    if not by:raise ValueError('No runs')
    rounds=next(iter(by.values())).keys()
    if any(x.keys()!=rounds for x in by.values()):raise ValueError('Incomplete paired rounds')
    output=[]
    for key,values in by.items():
        out=dict(condition=key,rounds=len(values),mean_ms=statistics.mean(values.values()),min_ms=min(values.values()),max_ms=max(values.values()))
        if 'native' in by:
            diff=[values[r]-by['native'][r] for r in sorted(rounds)]
            out['delta_native_ms']=statistics.mean(diff)
            out['relative_native']=out['mean_ms']/statistics.mean(by['native'].values())
            rng=random.Random(21042);boot=sorted(statistics.mean(rng.choices(diff,k=len(diff))) for _ in range(10000))
            out['paired_mean_ci95_low_ms']=boot[249];out['paired_mean_ci95_high_ms']=boot[9749]
        output.append(out)
    return output

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
    rows=json.loads((a.directory/'runs.json').read_text())
    if not (a.directory/'complete.json').exists():p.error('Missing completion marker')
    digests={r.get('aggregate_output_digest',r['output_digest']) for r in rows}
    if len(digests)!=1:p.error('Output digests differ')
    summary=summarize(rows)
    with (a.directory/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    (a.directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
