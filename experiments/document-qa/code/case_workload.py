"""Real-text complete prefill + fixed 64-token greedy decode, exact trace kept."""
import argparse, ctypes as C, hashlib, json, os, resource, time
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--prompts',type=Path,required=True)
p.add_argument('--warmup',type=int,default=2)
p.add_argument('--requests',type=int,default=3)
p.add_argument('--tokens',type=int,default=64)
p.add_argument('--set',default='heldout')
a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
cupti=None; phase_stamps=[]
if os.environ.get('CASE_CUPTI_LIBRARY'):
 cupti=C.CDLL(os.environ['CASE_CUPTI_LIBRARY']);cupti.deep_stop.argtypes=[C.c_char_p]
 cupti.case_stamp.restype=C.c_uint64
 assert cupti.deep_init()==0
backend=Path(os.environ.get('CASE_BACKEND','/root/autodl-tmp/neutrino-review-sm-only-20261009/llama-jetson-build/bin'))
os.environ['GGML_CUDA_DISABLE_GRAPHS']='1'
os.environ['NEUTRINO_BENCH_THREADS']='8'
lib=C.CDLL(str(a.root/'source/benchmarks/libcase.so'))
for name,types,restype in [
 ('qwen_create',[C.c_char_p,C.c_char_p],C.c_void_p),
 ('qwen_prefill',[C.c_void_p,C.c_char_p],C.c_int),
 ('qwen_argmax',[C.c_void_p],C.c_int),
 ('qwen_step',[C.c_void_p,C.c_int,C.c_int],C.c_int),
 ('qwen_copy_logits',[C.c_void_p,C.c_void_p,C.c_int],C.c_int),
 ('qwen_vocab_size',[C.c_void_p],C.c_int),
 ('qwen_tokenize',[C.c_void_p,C.c_char_p,C.c_void_p,C.c_int],C.c_int),
 ('qwen_detokenize',[C.c_void_p,C.c_void_p,C.c_int,C.c_char_p,C.c_int],C.c_int),
 ('qwen_destroy',[C.c_void_p],None)]:
 f=getattr(lib,name);f.argtypes=types;f.restype=restype
model=os.environ['CASE_MODEL']
load_start=time.perf_counter_ns(); h=lib.qwen_create(model.encode(),str(backend).encode())
assert h,'model GPU load failed'
load_ns=time.perf_counter_ns()-load_start
hook=None
if os.environ.get('NEUTRINO_HOOK_DRIVER'):
 hook=C.CDLL(os.environ['NEUTRINO_HOOK_DRIVER'])
 hook.neutrino_case_mark.argtypes=[C.c_int,C.c_int]
 hook.neutrino_capture_summary.argtypes=[C.c_char_p]
 hook.neutrino_ablation_flush.argtypes=[C.c_char_p]
 hook.neutrino_ablation_boundary()

prompts=json.loads(a.prompts.read_text())[a.set]
if os.environ.get('CASE_PROMPT_INDEX'):prompts=[prompts[int(os.environ['CASE_PROMPT_INDEX'])]]
teacher=json.loads(Path(os.environ['CASE_TEACHER_IDS']).read_text()) if os.environ.get('CASE_TEACHER_IDS') else None
vocab=lib.qwen_vocab_size(h)
logits=np.empty(vocab,dtype=np.float32)
all_logits=[]; rows=[]
def mark(request,token):
 if cupti: phase_stamps.append({'request':request,'token':token,'stamp':int(cupti.case_stamp())})
 if hook:
  assert hook.neutrino_case_mark(request,token)==0
  assert hook.neutrino_capture_mark(token)==0

def request(prompt,number,keep):
 text=prompt['text'].encode()
 ids=np.empty(8192,dtype=np.int32)
 n=lib.qwen_tokenize(h,text,ids.ctypes.data,len(ids));assert n>0
 before_files=set(Path(os.environ['NEUTRINO_TRACEDIR']).glob('*/result/*')) if hook else set()
 begin=time.perf_counter_ns();mark(number,0)
 count=lib.qwen_prefill(h,text); assert count==n,(count,n)
 first=time.perf_counter_ns()
 generated=[]; step=[]; ll=[]
 for i in range(a.tokens):
  assert lib.qwen_copy_logits(h,logits.ctypes.data,vocab)==vocab
  if keep and os.environ.get('CASE_SAVE_LOGITS')=='1': ll.append(logits.copy())
  generated.append(lib.qwen_argmax(h))
  if i<a.tokens-1:
   mark(number,i+1);t=time.perf_counter_ns()
   feed=teacher[prompt['id']][i] if teacher else generated[-1]
   assert lib.qwen_step(h,feed,count+i)==0
   step.append(time.perf_counter_ns()-t)
 inferred=time.perf_counter_ns()
 if hook: assert hook.neutrino_ablation_boundary()==0
 captured=time.perf_counter_ns()
 # Analysis is deliberately performed on the actual complete closed trace.
 after_files=set(Path(os.environ['NEUTRINO_TRACEDIR']).glob('*/result/*')) if hook else set()
 paths=sorted(str(x) for x in after_files-before_files if x.suffix in ('.nts','.bin'))
 analysis=None
 if paths and os.environ.get('CASE_ANALYZE')=='1':
  hook.neutrino_case_pause(1)
  from case_analyze import analyze
  analysis=analyze(paths,Path(os.environ['NEUTRINO_CASE_SIDECAR']),number)
  if keep: (a.out/f'analysis-{number}.json').write_text(json.dumps(analysis,indent=2)+'\n')
  else: (a.out/f'warmup-analysis-{number}.json').write_text(json.dumps({k:analysis[k] for k in ['request','frames','records','totals']},indent=2)+'\n')
  hook.neutrino_case_pause(0)
 answered=time.perf_counter_ns()
 arr=np.array(generated,dtype=np.int32);buf=C.create_string_buffer(32768)
 nt=lib.qwen_detokenize(h,arr.ctypes.data,len(arr),buf,len(buf));assert 0<=nt<len(buf)
 result={'request':number,'prompt_id':prompt['id'],'input_tokens':count,'output_tokens':len(arr),
  'input_ids':ids[:n].tolist(),'output_ids':arr.tolist(),'answer':buf.raw[:nt].decode(errors='replace'),
  'prefill_ns':first-begin,'decode_ns':inferred-first,'inference_ns':inferred-begin,
  'capture_ns':captured-begin,'drain_ns':captured-inferred,'analysis_ns':answered-captured,
  'answer_ns':answered-begin,'decode_steps_ns':step,'trace_paths':paths,
  'trace_bytes':sum(Path(x).stat().st_size for x in paths)}
 if ll: all_logits.append(np.array(ll))
 if keep:
  rows.append(result);(a.out/'workload.json').write_text(json.dumps({'load_ns':load_ns,'requests':rows,'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024},indent=2)+'\n')
 elif os.environ.get('CASE_DISCARD_WARMUP')=='1':
  for path in paths: Path(path).unlink()
 return result

try:
 for i in range(a.warmup): request(prompts[i%len(prompts)],-i-2,False)
 if cupti: assert cupti.deep_start()==0
 if hook and os.environ.get('NEUTRINO_ABLATION_MODE'):
  hook.neutrino_ablation_reset()
 for i in range(a.requests): request(prompts[i%len(prompts)],i,True)
 if hook and os.environ.get('NEUTRINO_ABLATION_MODE'):
  hook.neutrino_capture_summary(str(a.out/'capture-summary.json').encode())
  hook.neutrino_ablation_flush(str(a.out/'ablation.json').encode())
 if all_logits: np.save(a.out/'logits.npy',np.array(all_logits))
 if cupti:
  assert cupti.deep_stop(str(a.out/'cupti.json').encode())==0
  (a.out/'cupti-phase.json').write_text(json.dumps(phase_stamps,indent=2)+'\n')
finally:
 lib.qwen_destroy(h)
print(json.dumps({'requests':len(rows),'input_tokens':[x['input_tokens'] for x in rows],'capture_ms':[x['capture_ns']/1e6 for x in rows]}),flush=True)
