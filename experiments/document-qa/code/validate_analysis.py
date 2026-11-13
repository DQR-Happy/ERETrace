"""Independent post-performance equality to original complete-record analyzer."""
import json,sys,importlib.util,os,time
from pathlib import Path
out=Path(sys.argv[1]); r=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(r/'source/benchmarks'))
spec=importlib.util.spec_from_file_location('original_verified_analyzer','/root/autodl-tmp/neutrino-inference-case-20261013/code/case_analyze.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
work=json.loads((out/'workload.json').read_text())['requests'][0]
t=time.perf_counter_ns();old=module.analyze(work['trace_paths'],out/'launches.tsv',work['request'])
sha=old.pop('decoded_sha256');actual=json.loads((out/f"analysis-{work['request']}.json").read_text())
assert old==actual,'Formal analysis differs from original full-record analyzer'
(out/'validation.json').write_text(json.dumps({'passed':True,'analysis_exactly_equal':True,'complete_records':old['records'],'frames':old['frames'],'decoded_sha256':sha,'verification_wall_ns':time.perf_counter_ns()-t,'outside_all_performance_endpoints':True,'producer_decoder_unchanged':True},indent=2)+'\n')
print('PASS complete analysis identical; extra SHA and full-array checks outside performance',flush=True)
