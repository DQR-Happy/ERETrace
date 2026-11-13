"""Create an isolated full-trace snapshot; never edit authoritative evidence."""
from pathlib import Path
import shutil, json, hashlib

run = Path(__file__).resolve().parents[1]
project = run.parents[2]
origin = project / 'reports/neutrino-review-sm-only-20261009/source'
dest = run / 'source'
dest.mkdir(exist_ok=True)
for name in ('neutrino', 'benchmarks'):
    shutil.copytree(origin/name, dest/name, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__','*.so','*.cubin','build'))
for name in ('README.md','LICENSE','setup.py'):
    if (origin/name).exists(): shutil.copyfile(origin/name,dest/name)

# Explicit request/phase attribution for every captured launch. No GPU probe
# fields, precision, work, or capture filter is changed.
p = dest/'neutrino/src/cuda.c'
s = p.read_text()
needle = '    timing.name = kernel_name;\n    if (ab_mode && succeed'
assert needle in s
definition = '''
static FILE *case_sidecar;
static unsigned long long case_ordinal;
static int case_request=-1, case_token=-1;
static int case_paused;
int neutrino_case_mark(int request,int token){case_request=request;case_token=token;return 0;}
int neutrino_case_pause(int paused){case_paused=paused;return 0;}
static void case_launch(const char *name,unsigned *dims,CUstream stream,int captured){
    const char *p=getenv("NEUTRINO_CASE_SIDECAR");if(!p)return;
    if(!case_sidecar){case_sidecar=fopen(p,"w");if(!case_sidecar){perror(p);abort();}}
    fprintf(case_sidecar,"%llu\\t%d\\t%d\\t%d\\t%p\\t%u,%u,%u,%u,%u,%u,%u\\t%s\\n",
        case_ordinal++,case_request,case_token,captured,(void*)stream,
        dims[0],dims[1],dims[2],dims[3],dims[4],dims[5],dims[6],name?name:"UNKNOWN");
    fflush(case_sidecar);
}
'''
pos = s.index('CUresult cuLaunchKernel(')
s = s[:pos]+definition+s[pos:]
s=s.replace(needle,'    case_launch(kernel_name,ab_dims,hStream,succeed);\n'+needle,1)
s=s.replace('if (nt_passthrough || ab_excluded)', 'if(case_paused)return real_cuLaunchKernel(f,gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ,sharedMemBytes,hStream,kernelParams,extra);\n    if (nt_passthrough || ab_excluded)',1)
p.write_text(s)
p=dest/'neutrino/src/arena_capture.h'
s=p.read_text();assert 'int codec=n>=ar_codec_min;' in s
s=s.replace('int codec=n>=ar_codec_min;','int codec=!getenv("NEUTRINO_CASE_RAW") && n>=ar_codec_min;',1)
p.write_text(s)

# Adapter additions are host control / real input / output access only.
p = dest/'benchmarks/llama_adapter_dynamic.cpp'
s=p.read_text().replace('cp.n_ctx = 4096;','cp.n_ctx = 8192;\n    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;')
s=s.replace('cp.n_batch = 512;','cp.n_batch = std::getenv("CASE_BATCH") ? std::atoi(std::getenv("CASE_BATCH")) : 512;')
s=s.replace('cp.n_ubatch = 512;','cp.n_ubatch = cp.n_batch;')
s=s.replace('for (int offset = 0; offset < n; offset += 512) {','const int chunk = std::getenv("CASE_BATCH") ? std::atoi(std::getenv("CASE_BATCH")) : 512;\n    for (int offset = 0; offset < n; offset += chunk) {')
s=s.replace('std::min(512, n-offset)','std::min(chunk, n-offset)')
s=s.replace('int qwen_vocab_size(void * handle) {','''int qwen_tokenize(void * handle,const char *text,int *out,int cap){
    auto &s=*static_cast<Session*>(handle);
    return llama_tokenize(s.vocab,text,std::strlen(text),out,cap,true,true);
}
int qwen_vocab_size(void * handle) {''')
p.write_text(s)
manifest={str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(dest.rglob('*')) if p.is_file()}
(run/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(dest)
