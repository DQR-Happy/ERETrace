"""Isolated Neutrino diagnostic: full base trace + three natural attention launches."""
from pathlib import Path
import shutil,json
run=Path(__file__).resolve().parents[1];src=run/'source';dst=run/'diagnostic-source'
shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__','build','*.so','*.cubin'),dirs_exist_ok=True)
p=dst/'neutrino/probe/cuda.py';s=p.read_text()
s=s.replace('[a-z][\\w.]*', '[a-z][\\w.:]*') # L2:: cache modifiers are part of cp.async opcodes.
# Per-kernel probing contract: base-only for other operators. For attention,
# retain a base-only variant so every unselected launch still saves every warp.
needle='        # convert probes from Python Dict to data structure'
addition='''        if 'flash_attn_ext' in kernel_name:
            from neutrino.language.compiler import compile as compile_dsl
            base_dict = compile_dsl('CUDA', open(os.environ['CASE_BASE_PROBE']).read())
            base_probes, base_maps, base_regs = load(base_dict)
            base_ptx = global_section + '\\n' + func_section + '\\n' + probing(entry_section,base_probes,base_maps,base_regs)
            base_path = os.path.join(workdir,'base.ptx')
            open(base_path,'w').write(base_ptx)
            subprocess.run(['ptxas','-arch='+get_arch(),base_path,'-o',os.path.join(workdir,'base.bin')],check=True)
        else:
            from neutrino.language.compiler import compile as compile_dsl
            probes = compile_dsl('CUDA', open(os.environ['CASE_BASE_PROBE']).read())
            dynamic = False

'''
assert needle in s;s=s.replace(needle,addition+needle,1);p.write_text(s)

p=dst/'neutrino/src/cuda.c';s=p.read_text()
s=s.replace('    if (DYNAMIC) {\n                NT_STAGE(MODULE_COUNT);','    if (DYNAMIC && n_probe>1) {\n                NT_STAGE(MODULE_COUNT);',1)
# Other occurrences govern count replay and sizing; derive from active Maps.
s=s.replace('if (DYNAMIC)', 'if (DYNAMIC && n_probe>1)')
needle='    if (ab_mode && succeed && (DYNAMIC || n_probe != 1))'
replacement='''    if (succeed && n_probe>1) {
        static CUfunction full_func[128], base_func[128];static int nbase;
        CUfunction full=probed,base=NULL;int i;
        for(i=0;i<nbase;i++)if(full_func[i]==full){base=base_func[i];break;}
        if(!base){
            const char *found=NULL; /* locate retained base variant by name */
            DIR *dp=opendir(KERNEL_DIR);struct dirent *ent;char path[4096];
            while((ent=readdir(dp))){if(ent->d_name[0]=='.')continue;
                snprintf(path,sizeof(path),"%s/%s/kernel.info",KERNEL_DIR,ent->d_name);
                FILE *q=fopen(path,"r");if(!q)continue;char nm[2048];fgets(nm,sizeof(nm),q);fclose(q);
                nm[strcspn(nm,"\\n")]=0;if(strcmp(nm,kernel_name))continue;
                snprintf(path,sizeof(path),"%s/%s/base.bin",KERNEL_DIR,ent->d_name);found=path;break;
            }
            closedir(dp);if(!found){fprintf(stderr,"base variant missing\\n");abort();}
            CUmodule m;CUDA_CHECK(real_cuModuleLoad(&m,found));CUDA_CHECK(real_cuModuleGetFunction(&base,m,kernel_name));
            if(nbase>=128)abort();full_func[nbase]=full;base_func[nbase++]=base;
        }
        static int previous_token=-999,attention_index;
        if(previous_token!=case_token){previous_token=case_token;attention_index=0;}
        int index=attention_index++;
        const char *wanted=getenv("CASE_DETAIL_PREFILL_INDEX");int pi=wanted?atoi(wanted):180;
        int selected=case_request==0 && ((case_token==0&&index==pi)||((case_token==1||case_token==63)&&index==0));
        if(selected){char logpath[4096];snprintf(logpath,sizeof(logpath),"%s.params.tsv",getenv("NEUTRINO_CASE_SIDECAR"));
            FILE *q=fopen(logpath,"a");if(!q)abort();fprintf(q,"%llu\\t%d\\t%d",case_ordinal-1,case_token,index);
            for(int k=0;k<8;k++){uint64_t pointer;memcpy(&pointer,kernelParams[k],8);fprintf(q,"\\t%llx",(unsigned long long)pointer);}fputc('\\n',q);fclose(q);}
        if(!selected){static int one_size=16,one_type=PROBE_TYPE_WARP;
            probed=base;n_probe=1;probe_sizes=&one_size;probe_types=&one_type;}
    }
    if (ab_mode && succeed && (DYNAMIC || n_probe != 1))'''
assert needle in s;s=s.replace(needle,replacement,1)
s=s.replace('#include <cuda.h>','#include <cuda.h>\n#include <dirent.h>',1)
needle='        NT_STAGE(COUNT_D2H);'
assert needle in s
s=s.replace(needle,'        CUDA_CHECK(result);\n        CUDA_CHECK(real_cuStreamSynchronize(hStream));\n'+needle,1)
p.write_text(s)
print(dst)
