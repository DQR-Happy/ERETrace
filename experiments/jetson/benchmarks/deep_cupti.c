/* Diagnostic only: CUPTI driver/API correlation, thread CPU and GPU activity.
 * Raw records live in bounded process memory; only aggregates are written.
 * Do not run simultaneously with another CUPTI subscriber (Nsight).
 */
#define _GNU_SOURCE
#include <cuda.h>
#include <cupti.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <dlfcn.h>
#include <limits.h>

#define CAP (1u<<20)
#define GCAP 131072
#define NAMES 1024
#define GROUPS 512
#define CHECK(x) do {CUptiResult r=(x);if(r!=CUPTI_SUCCESS){const char*s;cuptiGetResultString(r,&s);fprintf(stderr,"CUPTI %s: %s\n",#x,s);return -1;}}while(0)
typedef struct {uint64_t start,end,cpu_start,cpu_ns,api_start,api_end,copy_bytes;CUcontext context;CUstream stream;unsigned cbid,cid,tid,depth,done,copy_dir,copy_async,stream_id;} api_record;
typedef struct {uint64_t start,end,bytes;unsigned cid,kind,src,dst,context,stream,group;} gpu_record;
typedef struct {char name[512];int dims[7];unsigned regs,local,context,stream;uint64_t calls,ns,min,max;} kernel_group;
typedef struct {CUfunction f;int block,shared;char name[512];} function_resource;
static api_record *api;
static gpu_record *gpu;
static unsigned *cidmap;
static kernel_group groups[GROUPS];
static function_resource resources[GROUPS];
static char names[NAMES][100];
static unsigned ng,ngr,nf;
static atomic_uint na;
static atomic_int active;
static uint64_t begin_ns,end_ns,dropped,overflow,window_cpu_start,window_cpu_end;
static uint64_t seen[128],inactive_records,outside_records;
static uint64_t requested_buffers,completed_buffers,completed_bytes;
static unsigned parse_status;
static pthread_mutex_t lock=PTHREAD_MUTEX_INITIALIZER;
static CUpti_SubscriberHandle subscriber;
static __thread unsigned depth;
static int activity_enabled=1;
static uint64_t cpu_now(void){struct timespec t;clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t);return (uint64_t)t.tv_sec*1000000000+t.tv_nsec;}
static uint64_t stamp(void){uint64_t t=0;cuptiGetTimestamp(&t);return t;}
static void json_string(FILE *fp,const char*s){fputc('"',fp);for(;*s;s++){if(*s=='"'||*s=='\\')fputc('\\',fp);if((unsigned char)*s>=32)fputc(*s,fp);}fputc('"',fp);}

static void callback(void *userdata,CUpti_CallbackDomain domain,CUpti_CallbackId id,const void *data){
    (void)userdata;
    if(domain!=CUPTI_CB_DOMAIN_DRIVER_API || !atomic_load(&active))return;
    const CUpti_CallbackData*d=data;
    if(d->callbackSite==CUPTI_API_ENTER){
        unsigned i=atomic_fetch_add(&na,1);
        *d->correlationData=(uint64_t)i+1;
        if(i>=CAP){__atomic_fetch_add(&overflow,1,__ATOMIC_RELAXED);return;}
        api_record*r=&api[i];r->cbid=id;r->cid=d->correlationId;r->tid=syscall(SYS_gettid);r->depth=depth++;
        r->context=d->context;r->stream_id=UINT_MAX;
        if(id==CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2){const cuMemcpyDtoH_v2_params*p=d->functionParams;r->copy_bytes=p->ByteCount;r->copy_dir=2;}
        if(id==CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2){const cuMemcpyDtoHAsync_v2_params*p=d->functionParams;r->copy_bytes=p->ByteCount;r->copy_dir=2;r->copy_async=1;r->stream=p->hStream;}
        if(id==CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2){const cuMemcpyHtoD_v2_params*p=d->functionParams;r->copy_bytes=p->ByteCount;r->copy_dir=1;}
        if(id==CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2){const cuMemcpyHtoDAsync_v2_params*p=d->functionParams;r->copy_bytes=p->ByteCount;r->copy_dir=1;r->copy_async=1;r->stream=p->hStream;}
        unsigned h=r->cid&(CAP-1);
        for(;;){unsigned old=__atomic_load_n(&cidmap[h],__ATOMIC_ACQUIRE);
            if(!old){unsigned expected=0;if(__atomic_compare_exchange_n(&cidmap[h],&expected,i+1,0,__ATOMIC_RELEASE,__ATOMIC_RELAXED))break;}
            else if(api[old-1].cid==r->cid)break;
            h=(h+1)&(CAP-1);}
        if(id<NAMES && !names[id][0]){pthread_mutex_lock(&lock);snprintf(names[id],sizeof(names[id]),"%s",d->functionName);pthread_mutex_unlock(&lock);}
        if(id==CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel && d->symbolName){
            const cuLaunchKernel_params*p=d->functionParams;
            pthread_mutex_lock(&lock);unsigned j;
            for(j=0;j<nf;j++)if(resources[j].f==p->f && resources[j].block==(int)(p->blockDimX*p->blockDimY*p->blockDimZ) && resources[j].shared==(int)p->sharedMemBytes)break;
            if(j==nf && nf<GROUPS){resources[j].f=p->f;resources[j].block=p->blockDimX*p->blockDimY*p->blockDimZ;resources[j].shared=p->sharedMemBytes;snprintf(resources[j].name,512,"%s",d->symbolName);nf++;}
            pthread_mutex_unlock(&lock);
        }
        r->start=stamp();r->cpu_start=cpu_now();
    }else{
        uint64_t t=cpu_now(),s=stamp(),v=*d->correlationData;
        if(v && v<=CAP){api_record*r=&api[v-1];r->end=s;r->cpu_ns=t-r->cpu_start;r->done=1;if(depth)depth--;}
    }
}
static void CUPTIAPI buffer_requested(uint8_t **buffer,size_t *size,size_t *max){
    __atomic_fetch_add(&requested_buffers,1,__ATOMIC_RELAXED);
    *size=16*1024*1024;*max=0;if(posix_memalign((void**)buffer,8,*size))abort();
}
static unsigned group_kernel(CUpti_ActivityKernel9*k){
    int dims[7]={k->gridX,k->gridY,k->gridZ,k->blockX,k->blockY,k->blockZ,k->dynamicSharedMemory};
    unsigned j;for(j=0;j<ng;j++)if(!strcmp(groups[j].name,k->name) && !memcmp(groups[j].dims,dims,sizeof(dims)) && groups[j].context==k->contextId && groups[j].stream==k->streamId)break;
    if(j==ng){if(ng==GROUPS){overflow++;return UINT_MAX;}ng++;snprintf(groups[j].name,512,"%s",k->name);memcpy(groups[j].dims,dims,sizeof(dims));groups[j].regs=k->registersPerThread;groups[j].local=k->localMemoryPerThread;groups[j].context=k->contextId;groups[j].stream=k->streamId;groups[j].min=UINT64_MAX;}
    kernel_group*g=&groups[j];uint64_t n=k->end-k->start;g->calls++;g->ns+=n;if(n<g->min)g->min=n;if(n>g->max)g->max=n;return j;
}
static void CUPTIAPI buffer_completed(CUcontext ctx,uint32_t stream,uint8_t *buffer,size_t size,size_t valid){
    (void)size;CUpti_Activity*r=NULL;
    pthread_mutex_lock(&lock);
    completed_buffers++;completed_bytes+=valid;
    CUptiResult next;
    while((next=cuptiActivityGetNextRecord(buffer,valid,&r))==CUPTI_SUCCESS){
        if((unsigned)r->kind<128)seen[r->kind]++;
        if(!atomic_load(&active)){inactive_records++;continue;}
        if(r->kind==CUPTI_ACTIVITY_KIND_DRIVER){
            CUpti_ActivityAPI*a=(void*)r;
            if(a->start<begin_ns || a->start>end_ns){outside_records++;continue;}
            unsigned h=a->correlationId&(CAP-1);
            while(cidmap[h] && api[cidmap[h]-1].cid!=a->correlationId)h=(h+1)&(CAP-1);
            if(cidmap[h]){api_record*t=&api[cidmap[h]-1];if(t->cbid==a->cbid){t->api_start=a->start;t->api_end=a->end;}}
            continue;
        }
        gpu_record g={0};g.kind=r->kind;
        if(r->kind==CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || r->kind==CUPTI_ACTIVITY_KIND_KERNEL){
            CUpti_ActivityKernel9*k=(void*)r;if(k->start<begin_ns || k->start>end_ns){outside_records++;continue;}
            g.start=k->start;g.end=k->end;g.cid=k->correlationId;g.context=k->contextId;g.stream=k->streamId;g.group=group_kernel(k);
        }else if(r->kind==CUPTI_ACTIVITY_KIND_MEMCPY){
            CUpti_ActivityMemcpy5*m=(void*)r;if(m->start<begin_ns || m->start>end_ns){outside_records++;continue;}
            g.start=m->start;g.end=m->end;g.bytes=m->bytes;g.cid=m->correlationId;g.src=m->srcKind;g.dst=m->dstKind;g.context=m->contextId;g.stream=m->streamId;g.group=m->copyKind;
        }else if(r->kind==CUPTI_ACTIVITY_KIND_MEMSET){
            CUpti_ActivityMemset4*m=(void*)r;if(m->start<begin_ns || m->start>end_ns){outside_records++;continue;}
            g.start=m->start;g.end=m->end;g.bytes=m->bytes;g.cid=m->correlationId;g.context=m->contextId;g.stream=m->streamId;
        }else continue;
        if(ngr<GCAP)gpu[ngr++]=g;else overflow++;
    }
    parse_status=next;
    size_t count=0;cuptiActivityGetNumDroppedRecords(ctx,stream,&count);dropped+=count;
    pthread_mutex_unlock(&lock);free(buffer);
}
int deep_init(void){
    api=calloc(CAP,sizeof(*api));gpu=calloc(GCAP,sizeof(*gpu));cidmap=calloc(CAP,sizeof(*cidmap));
    if(!api||!gpu||!cidmap)return -1;
    const char *v=getenv("DEEP_CUPTI_ACTIVITY");if(v && !atoi(v))activity_enabled=0;
    CHECK(cuptiSubscribe(&subscriber,callback,NULL));CHECK(cuptiEnableDomain(1,subscriber,CUPTI_CB_DOMAIN_DRIVER_API));
    if(activity_enabled){CHECK(cuptiActivityRegisterCallbacks(buffer_requested,buffer_completed));
        CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
        CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));}return 0;
}
int deep_start(void){
    if(activity_enabled){CHECK(cuptiActivityFlushAll(0));}atomic_store(&na,0);ng=ngr=nf=0;dropped=overflow=0;
    memset(seen,0,sizeof(seen));inactive_records=outside_records=0;
    memset(api,0,CAP*sizeof(*api));memset(cidmap,0,CAP*sizeof(*cidmap));
    window_cpu_start=cpu_now();begin_ns=stamp();end_ns=UINT64_MAX;atomic_store(&active,1);return 0;
}
static int gpu_compare(const void*a,const void*b){const gpu_record*x=a,*y=b;return(x->start>y->start)-(x->start<y->start);}
static uint64_t gpu_union(void){qsort(gpu,ngr,sizeof(*gpu),gpu_compare);uint64_t total=0,a=0,b=0;for(unsigned i=0;i<ngr;i++){if(gpu[i].start>b){total+=b-a;a=gpu[i].start;b=gpu[i].end;}else if(gpu[i].end>b)b=gpu[i].end;}return total+b-a;}
static unsigned gpu_class(gpu_record*g){
    if(g->kind==CUPTI_ACTIVITY_KIND_MEMCPY)return 4;
    if(g->kind==CUPTI_ACTIVITY_KIND_MEMSET)return 8;
    if(g->group<ng&&!strcmp(groups[g->group].name,"neutrino_capture_pack"))return 2;
    return 1;
}
/* gpu_union above has sorted records by device timestamp. */
static uint64_t gpu_mask_union(unsigned mask,unsigned ctx,unsigned stream,int filter){
    uint64_t total=0,a=0,b=0;
    for(unsigned i=0;i<ngr;i++){gpu_record*g=&gpu[i];if(!(gpu_class(g)&mask)||(filter&&(g->context!=ctx||g->stream!=stream)))continue;
        if(g->start>b){total+=b-a;a=g->start;b=g->end;}else if(g->end>b)b=g->end;
    }return total+b-a;
}
int deep_stop(const char *path){
    end_ns=stamp();window_cpu_end=cpu_now();if(activity_enabled){CHECK(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));}atomic_store(&active,0);
    unsigned n=atomic_load(&na);if(n>CAP)n=CAP;
    typedef struct {uint64_t count,wall,cpu,activity,matched;uint64_t min,max;} agg;
    agg apis[NAMES]={0};uint64_t topwall=0,topcpu=0;
    for(unsigned i=0;i<n;i++){api_record*r=&api[i];if(!r->done || r->cbid>=NAMES)continue;agg*a=&apis[r->cbid];uint64_t wall=r->end-r->start;a->count++;a->wall+=wall;a->cpu+=r->cpu_ns;if(!r->depth){topwall+=wall;topcpu+=r->cpu_ns;}if(r->api_end>r->api_start){a->activity+=r->api_end-r->api_start;a->matched++;}}
    FILE*fp=fopen(path,"w");if(!fp)return -1;
    fprintf(fp,"{\"window_ns\":%llu,\"window_thread_cpu_ns\":%llu,\"dropped\":%llu,\"overflow\":%llu,\"driver_callbacks\":%u,\"top_driver_wall_ns\":%llu,\"top_driver_cpu_ns\":%llu,\"gpu_activity_union_ns\":%llu,\"apis\":[",(unsigned long long)(end_ns-begin_ns),(unsigned long long)(window_cpu_end-window_cpu_start),(unsigned long long)dropped,(unsigned long long)overflow,n,(unsigned long long)topwall,(unsigned long long)topcpu,(unsigned long long)gpu_union());
    int comma=0;for(unsigned i=0;i<NAMES;i++)if(apis[i].count){agg*a=&apis[i];fprintf(fp,"%s{\"name\":",comma?",":"");json_string(fp,names[i]);fprintf(fp,",\"cbid\":%u,\"count\":%llu,\"callback_wall_ns\":%llu,\"thread_cpu_ns\":%llu,\"driver_activity_count\":%llu,\"driver_activity_ns\":%llu}",i,(unsigned long long)a->count,(unsigned long long)a->wall,(unsigned long long)a->cpu,(unsigned long long)a->matched,(unsigned long long)a->activity);comma=1;}
    fputs("],\"api_threads\":[",fp);
    typedef struct {unsigned tid,cbid;uint64_t count,wall,cpu;} thread_api;
    thread_api ta[1024]={0};unsigned nta=0,ta_overflow=0;
    for(unsigned i=0;i<n;i++){api_record*r=&api[i];if(!r->done)continue;unsigned j;
        for(j=0;j<nta;j++)if(ta[j].tid==r->tid&&ta[j].cbid==r->cbid)break;
        if(j==nta){if(nta==1024){ta_overflow++;continue;}ta[j].tid=r->tid;ta[j].cbid=r->cbid;nta++;}
        ta[j].count++;ta[j].wall+=r->end-r->start;ta[j].cpu+=r->cpu_ns;
    }
    for(unsigned i=0;i<nta;i++){thread_api*t=&ta[i];fprintf(fp,"%s{\"tid\":%u,\"api\":",i?",":"",t->tid);json_string(fp,t->cbid<NAMES?names[t->cbid]:"unknown");fprintf(fp,",\"count\":%llu,\"wall_ns\":%llu,\"thread_cpu_ns\":%llu}",(unsigned long long)t->count,(unsigned long long)t->wall,(unsigned long long)t->cpu);}
    uint64_t mu=gpu_mask_union(1,0,0,0),pu=gpu_mask_union(2,0,0,0),cu=gpu_mask_union(4,0,0,0),zu=gpu_mask_union(8,0,0,0);
    uint64_t max_end=0;for(unsigned i=0;i<ngr;i++)if(gpu[i].end>max_end)max_end=gpu[i].end;
    fprintf(fp,"],\"api_thread_group_overflow\":%u,\"gpu_timeline\":{\"model_union_ns\":%llu,\"encoder_union_ns\":%llu,\"copy_union_ns\":%llu,\"memset_union_ns\":%llu,\"model_copy_intersection_ns\":%llu,\"activity_span_ns\":%llu,\"streams\":[",ta_overflow,(unsigned long long)mu,(unsigned long long)pu,(unsigned long long)cu,(unsigned long long)zu,(unsigned long long)(mu+cu-gpu_mask_union(5,0,0,0)),(unsigned long long)(ngr?max_end-gpu[0].start:0));
    unsigned stream_ctx[32],stream_id[32],nst=0;
    for(unsigned i=0;i<ngr;i++){unsigned j;for(j=0;j<nst;j++)if(stream_ctx[j]==gpu[i].context&&stream_id[j]==gpu[i].stream)break;if(j==nst&&nst<32){stream_ctx[nst]=gpu[i].context;stream_id[nst++]=gpu[i].stream;}}
    for(unsigned i=0;i<nst;i++)fprintf(fp,"%s{\"context\":%u,\"stream\":%u,\"activity_union_ns\":%llu}",i?",":"",stream_ctx[i],stream_id[i],(unsigned long long)gpu_mask_union(15,stream_ctx[i],stream_id[i],1));
    fputs("]},\"kernels\":[",fp);
    for(unsigned i=0;i<ng;i++){kernel_group*g=&groups[i];fprintf(fp,"%s{\"name\":",i?",":"");json_string(fp,g->name);fprintf(fp,",\"shape\":[%d,%d,%d,%d,%d,%d,%d],\"context\":%u,\"stream\":%u,\"registers\":%u,\"local_bytes_per_thread\":%u,\"count\":%llu,\"gpu_ns\":%llu,\"min_ns\":%llu,\"max_ns\":%llu}",g->dims[0],g->dims[1],g->dims[2],g->dims[3],g->dims[4],g->dims[5],g->dims[6],g->context,g->stream,g->regs,g->local,(unsigned long long)g->calls,(unsigned long long)g->ns,(unsigned long long)g->min,(unsigned long long)g->max);}
    /* One API may generate several GPU pieces: aggregate per correlation before size grouping. */
    typedef struct {uint64_t bytes,ns,start,end;unsigned pieces,src,dst,stream,context;} copy_join;
    copy_join *joins=calloc(n,sizeof(*joins));unsigned *copy_heads=calloc(CAP,sizeof(unsigned)),*copy_next=calloc(n,sizeof(unsigned));
    unsigned unmatched=0,ambiguous=0,copy_count=0,memset_count=0;uint64_t copy_bytes=0,memset_ns=0;
    for(unsigned i=0;i<n;i++)if(api[i].copy_dir){
        cuptiGetStreamId(api[i].context,api[i].stream,&api[i].stream_id);
        unsigned h=api[i].cid&(CAP-1);while(copy_heads[h] && api[copy_heads[h]-1].cid!=api[i].cid)h=(h+1)&(CAP-1);
        copy_next[i]=copy_heads[h];copy_heads[h]=i+1;
    }
    for(unsigned i=0;i<ngr;i++){
        gpu_record*g=&gpu[i];if(g->kind==CUPTI_ACTIVITY_KIND_MEMSET){memset_count++;memset_ns+=g->end-g->start;continue;}
        if(g->kind!=CUPTI_ACTIVITY_KIND_MEMCPY)continue;copy_count++;copy_bytes+=g->bytes;
        unsigned h=g->cid&(CAP-1);while(copy_heads[h] && api[copy_heads[h]-1].cid!=g->cid)h=(h+1)&(CAP-1);
        unsigned candidate=UINT_MAX,matches=0;
        for(unsigned j=copy_heads[h];j;j=copy_next[j-1]){
            api_record*a=&api[j-1];
            if(!a->done || a->copy_dir!=g->group || a->copy_bytes<g->bytes || a->stream_id!=g->stream)continue;
            if(g->start+1000<a->start || (!a->copy_async && g->end>a->end+1000))continue;
            matches++;candidate=j-1;
        }
        if(matches!=1){if(matches)ambiguous++;else unmatched++;continue;}
        copy_join*j=&joins[candidate];j->bytes+=g->bytes;j->ns+=g->end-g->start;
        if(!j->pieces || g->start<j->start)j->start=g->start;if(g->end>j->end)j->end=g->end;
        j->pieces++;j->src=g->src;j->dst=g->dst;j->stream=g->stream;j->context=g->context;
    }
    fputs("],\"copy_groups\":[",fp);comma=0;
    for(unsigned i=0;i<n;i++){copy_join*j=&joins[i];if(!j->pieces)continue;
        uint64_t calls=0,pieces=0,bytes=0,ns=0,wall=0,cpu=0,activity=0;int64_t prefix=0,suffix=0,callback_prefix=0,callback_suffix=0;
        for(unsigned k=i;k<n;k++){copy_join*q=&joins[k];if(q->pieces && q->bytes==j->bytes && q->src==j->src && q->dst==j->dst && api[k].cbid==api[i].cbid && q->stream==j->stream && q->context==j->context){calls++;pieces+=q->pieces;bytes+=q->bytes;ns+=q->ns;wall+=api[k].end-api[k].start;cpu+=api[k].cpu_ns;callback_prefix+=(int64_t)q->start-(int64_t)api[k].start;callback_suffix+=(int64_t)api[k].end-(int64_t)q->end;if(api[k].api_end>api[k].api_start){activity+=api[k].api_end-api[k].api_start;prefix+=(int64_t)q->start-(int64_t)api[k].api_start;suffix+=(int64_t)api[k].api_end-(int64_t)q->end;}if(k!=i)q->pieces=0;}}
        fprintf(fp,"%s{\"api\":",comma?",":"");json_string(fp,names[api[i].cbid]);fprintf(fp,",\"bytes_each\":%llu,\"src_kind\":%u,\"dst_kind\":%u,\"context\":%u,\"stream\":%u,\"calls\":%llu,\"gpu_pieces\":%llu,\"bytes\":%llu,\"gpu_ns\":%llu,\"callback_wall_ns\":%llu,\"thread_cpu_ns\":%llu,\"driver_activity_ns\":%llu,\"prefix_ns\":%lld,\"suffix_ns\":%lld,\"callback_prefix_ns\":%lld,\"callback_suffix_ns\":%lld}",(unsigned long long)j->bytes,j->src,j->dst,j->context,j->stream,(unsigned long long)calls,(unsigned long long)pieces,(unsigned long long)bytes,(unsigned long long)ns,(unsigned long long)wall,(unsigned long long)cpu,(unsigned long long)activity,(long long)prefix,(long long)suffix,(long long)callback_prefix,(long long)callback_suffix);comma=1;
    }
    fprintf(fp,"],\"correlation_schema\":2,\"copy_activity_count\":%u,\"copy_activity_bytes\":%llu,\"unmatched_copies\":%u,\"ambiguous_copies\":%u,\"memset_count\":%u,\"memset_gpu_ns\":%llu,\"function_resources\":[",copy_count,(unsigned long long)copy_bytes,unmatched,ambiguous,memset_count,(unsigned long long)memset_ns);
    void*driver=dlopen("/usr/lib/x86_64-linux-gnu/libcuda.so.580.105.08",RTLD_NOW);
    __typeof__(&cuFuncGetAttribute) fa=dlsym(driver,"cuFuncGetAttribute");
    __typeof__(&cuOccupancyMaxActiveBlocksPerMultiprocessor) occ=dlsym(driver,"cuOccupancyMaxActiveBlocksPerMultiprocessor");
    for(unsigned i=0;i<nf;i++){function_resource*f=&resources[i];int regs=-1,local=-1,smem=-1,blocks=-1;CUresult rc=fa(&regs,CU_FUNC_ATTRIBUTE_NUM_REGS,f->f);fa(&local,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,f->f);fa(&smem,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,f->f);CUresult orc=occ(&blocks,f->f,f->block,f->shared);fprintf(fp,"%s{\"name\":",i?",":"");json_string(fp,f->name);fprintf(fp,",\"block_threads\":%d,\"dynamic_shared\":%d,\"registers\":%d,\"local_bytes\":%d,\"static_shared\":%d,\"max_active_blocks_per_sm\":%d,\"attribute_result\":%d,\"occupancy_result\":%d}",f->block,f->shared,regs,local,smem,blocks,rc,orc);}
    fprintf(fp,"],\"requested_buffers\":%llu,\"completed_buffers\":%llu,\"completed_bytes\":%llu,\"parse_status\":%u,\"inactive_records\":%llu,\"outside_records\":%llu,\"seen_kinds\":{",(unsigned long long)requested_buffers,(unsigned long long)completed_buffers,(unsigned long long)completed_bytes,parse_status,(unsigned long long)inactive_records,(unsigned long long)outside_records);comma=0;for(unsigned i=0;i<128;i++)if(seen[i]){fprintf(fp,"%s\"%u\":%llu",comma?",":"",i,(unsigned long long)seen[i]);comma=1;}fputs("}}\n",fp);free(joins);free(copy_heads);free(copy_next);int bad=ferror(fp);if(fclose(fp))bad=1;return bad?-1:0;
}
