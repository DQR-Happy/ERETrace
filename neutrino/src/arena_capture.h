/* Experimental byte-bounded mixed-Map collection. Single producer/context.
 * Device circular arenas retire after copy completion (or writer completion in
 * the coupled control). Host arena retires after FIFO output. No extra thread.
 * Small Maps are written directly into the output arena; large Maps use the
 * unchanged GPU13 encoder. A segment has one producer->copy dependency.
 */
#ifndef NEUTRINO_ARENA_CAPTURE_H
#define NEUTRINO_ARENA_CAPTURE_H
#include <sys/uio.h>
#include <errno.h>
#include <sys/mman.h>
static int ar_decouple=1,ar_frames=64,ar_frame_copies,ar_token_flush,ar_diag,ar_verify,ar_template;
static int ar_trim,ar_incremental,ar_batch_encoder,ar_writer_mode,ar_delta,ar_force_fallback;
static int ar_check_first,ar_release_checked,ar_hier,ar_reference,ar_tile,ar_omit;
static size_t ar_codec_min=65536;
static unsigned ar_local_bits=16;
static int ar_hot;static int ar_nv;static unsigned ar_nv_chunk=65536;static __thread int ar_external_depth;
typedef struct {uint64_t source,bytes;} ar_nv_frame;
static void*(*nv_create)(size_t,unsigned,unsigned,unsigned);
static size_t(*nv_bound)(void*,size_t,size_t);
static const uint64_t*(*nv_submit)(void*,unsigned,const ar_nv_frame*,unsigned,size_t,uint64_t,void*);
static void(*nv_space)(void*,uint64_t*);static void(*nv_destroy)(void*);
static int(*nv_decode)(const void*,size_t,void*,size_t,unsigned,unsigned);
static void ar_nv_load(void){if(nv_create)return;const char*p=getenv("NEUTRINO_NVPIPE_LIBRARY");if(!p)async_fatal("nvpipe library path required");void*h=dlopen(p,RTLD_NOW|RTLD_LOCAL);if(!h){fprintf(stderr,"%s\n",dlerror());async_fatal("nvpipe load");}
#define NV_LOAD(name) *(void**)(&nv_##name)=dlsym(h,"nvpipe_" #name);if(!nv_##name)async_fatal("nvpipe symbol")
 NV_LOAD(create);NV_LOAD(bound);NV_LOAD(submit);NV_LOAD(space);NV_LOAD(destroy);NV_LOAD(decode);
#undef NV_LOAD
}
static unsigned ar_delta_width=4,ar_elapsed_width=4;static int ar_flex;
static size_t ar_stdio_bytes=1048576,ar_soft_bytes;
static uint64_t ar_age_ns;
static size_t ar_chunk=16ULL<<20,ar_host_bytes=256ULL<<20,ar_device_bytes=128ULL<<20;
enum { AR_QUEUE=512, AR_FRAMES=1024 };
static __thread int ar_token;
typedef struct { uint64_t source,dest,n;uint32_t first,blocks; } ar_pack_desc;
typedef struct { uint64_t tail,head;size_t capacity; } ar_ring;
typedef struct { unsigned dims[7];size_t raw_offset,out_offset,bytes,copy_bytes;uint64_t launch;int token,codec;void*write_payload,*write_allocation;size_t write_bytes;uint64_t write_tag; } ar_frame;
typedef struct { CUfunction function;CUstream stream;unsigned dims[7];size_t bytes,raw,out;unsigned count; } ar_plan;
typedef struct {size_t incoming,head_max;int head_state,kind;uint64_t waits,ns;} ar_stall;
typedef struct {
    CUevent ready,done;CUstream stream;int state,written,retired,checked;
    uint64_t raw_end,out_end,host_end;size_t raw_base,out_base,host_base;
    size_t raw_limit,out_limit,raw_used,out_used;unsigned count,target,plan_start;
    ar_frame *frames;unsigned frame_capacity;
    ar_pack_desc *descriptors;CUdeviceptr device_descriptors;uint64_t born;const uint64_t*nv_meta;
} ar_segment;
typedef struct ar_engine {
    CUcontext context;CUstream copy;CUdeviceptr raw,out,error,delta_indices;void *host;
    struct {CUfunction f;unsigned shared;} attributes[128];unsigned attribute_count;
    void*nv;CUstream receive;CUmodule module;CUfunction encoder,batch_encoder,delta_init,delta_encoder;pthread_t writer;pthread_mutex_t mutex;pthread_cond_t changed;
    ar_ring raw_ring,out_ring,host_ring;ar_segment segment[AR_QUEUE];
    uint64_t published,consumed,retired,launch,request;ar_segment *pending;
    uint64_t file_position;
    int stop;FILE *fp,*ids;char*stdio_buffer,*directory;
    ar_plan *plan;unsigned plan_capacity,plan_count,plan_pos;int plan_valid,plan_using,plan_learning;
    size_t incoming_bytes;ar_stall stalls[128];unsigned nstalls;
    uint64_t calls,payload,copy_payload,transfer_bytes,storage_bytes,batches,encoders,copies;
    uint64_t ready_records,done_records,stream_waits,queries,device_syncs,worker_syncs;
    uint64_t host_waits,device_waits,descriptor_waits,host_wait_ns,device_wait_ns;
    uint64_t prepare_ns,model_ns,clear_ns,encode_enqueue_ns,copy_enqueue_ns,flush_ns;
    uint64_t worker_wait_ns,worker_write_ns,worker_cpu_ns,drain_ns,verified_frames;
    uint64_t peak_host,peak_raw,peak_out,peak_segments,peak_frames,inflight_frames;
    uint64_t flush_full,flush_count,flush_stream,flush_token,flush_boundary;
    uint64_t template_hits,template_invalidations,template_builds,reserved_raw_bytes,reserved_out_bytes;
    uint64_t trimmed_raw_bytes,trimmed_out_bytes,incremental_extensions,flush_pressure;
    uint64_t argument_ns,attribute_calls,attribute_skips,argument_heap_calls;
    uint64_t nv_metadata_bytes,nv_payload_ns,nv_validation_ns;
    uint64_t delta_initializers,delta_fallbacks,fallback_copy_bytes,debt_bytes,debt_segments,debt_requests;
    uint64_t encoder_frames,descriptor_bytes,writer_syscalls,writer_prepare_ns,writer_output_ns,flush_bytes,flush_age;
    uint64_t checked_notifications,check_first_ns,fallback_host_peak;
    uint64_t escape_frames[5],escape_raw_bytes[5],forced_fallbacks;
    uint64_t mmap_setup_ns,mmap_copy_ns,mmap_unmap_ns,mmap_bytes,mmap_calls;
    uint64_t write_bucket_ns[3],write_bucket_frames[3],write_bucket_bytes[3];
    struct ar_engine *next;
} ar_engine;
static __thread ar_engine *ar_engines;
static uint64_t ar_now(void){return ar_diag?async_clock(CLOCK_MONOTONIC):0;}
// Validation-only oracle over the SAME original Map. Independent file consumer
// recomputes these integer queries after decoding. Never enabled in timings.
static void ar_query_oracle(ar_engine*e,ar_frame*f,const void*raw){
    if(!getenv("NEUTRINO_QUERY_ORACLE"))return;
    uint64_t count[256]={0},sum[256]={0},hist[33]={0},total=0;uint32_t maximum=0;
    const unsigned char*p=raw;size_t n=f->bytes/16;
    for(size_t i=0;i<n;i++){uint32_t elapsed,sm;memcpy(&elapsed,p+i*16+8,4);memcpy(&sm,p+i*16+12,4);
        if(sm>=256)async_fatal("query oracle SM exceeds validated domain");
        count[sm]++;sum[sm]+=elapsed;total+=elapsed;if(elapsed>maximum)maximum=elapsed;
        hist[elapsed?32-__builtin_clz(elapsed):0]++;
    }
    char*path=malloc(strlen(e->directory)+80);if(!path)async_fatal("query oracle path");
    sprintf(path,"%s/arena-%llu.rawq",e->directory,(unsigned long long)e->request);
    FILE*q=fopen(path,"ab");free(path);if(!q)async_fatal("query oracle open");
    fprintf(q,"%llu,%d,%zu,%llu,%u;",(unsigned long long)f->launch,f->token,n,(unsigned long long)total,maximum);
    for(unsigned b=0;b<33;b++)fprintf(q,"%s%llu",b?",":"",(unsigned long long)hist[b]);
    fputc(';',q);for(unsigned sm=0;sm<256;sm++)if(count[sm])fprintf(q,"%u:%llu:%llu,",sm,(unsigned long long)count[sm],(unsigned long long)sum[sm]);
    fputc('\n',q);if(fclose(q))async_fatal("query oracle close");
}
static size_t ar_align(size_t n){return(n+15)&~(size_t)15;}
static int ar_fit(ar_ring*r,size_t n,uint64_t*begin,uint64_t*end){
    if(n>r->capacity)async_fatal("arena allocation exceeds byte budget");
    uint64_t b=r->tail;size_t p=b%r->capacity;n=ar_align(n);
    if(n && n>r->capacity-p){b+=r->capacity-p;if(r->head==r->tail)r->head=r->tail=b;}
    *begin=b;*end=b+n;return *end-r->head<=r->capacity;
}
static void ar_attribute(ar_engine*e,CUfunction f,unsigned shared){
    if(ar_hot&2){unsigned i;for(i=0;i<e->attribute_count;i++)if(e->attributes[i].f==f)break;
        if(i<e->attribute_count&&e->attributes[i].shared==shared){e->attribute_skips++;return;}
        CUDA_CHECK(real_cuFuncSetAttribute(f,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,shared));e->attribute_calls++;
        if(i<128){e->attributes[i].f=f;e->attributes[i].shared=shared;if(i==e->attribute_count)e->attribute_count++;}return;
    }
    CUDA_CHECK(real_cuFuncSetAttribute(f,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,shared));e->attribute_calls++;
}
static void ar_peak(uint64_t *p,uint64_t n){if(n>*p)*p=n;}
static void ar_put(ar_engine*e,const void*p,size_t n){if(fwrite(p,1,n,e->fp)!=n)async_fatal("arena file write");e->storage_bytes+=n;e->file_position+=n;}
// GPU9 retains raw ownership until the writer has decided whether fallback is needed.
static void*ar_file_payload(ar_engine*e,ar_segment*s,ar_frame*f,size_t*bytes,uint64_t*packed,void**allocation){
    void*p=(char*)e->host+s->host_base+f->out_offset;*bytes=f->copy_bytes;*packed=f->codec?f->copy_bytes:0;*allocation=NULL;
    if(ar_nv){*packed=f->codec?(UINT64_C(3)<<62)|f->copy_bytes:0;return p;}
    if(ar_delta&&f->codec){unsigned flags;memcpy(&flags,p,4);if(ar_force_fallback&&f->launch%101==0){flags|=4;e->forced_fallbacks++;}
        if(flags)for(unsigned b=0;b<5;b++)if(flags&(1U<<b)){e->escape_frames[b]++;e->escape_raw_bytes[b]+=f->bytes;}
        if(flags){*bytes=f->bytes;*packed=(UINT64_C(1)<<62)|f->bytes;*allocation=malloc(f->bytes);if(!*allocation)async_fatal("GPU9 fallback allocation");
            CUDA_CHECK(real_cuMemcpyDtoH_v2(*allocation,e->raw+s->raw_base+f->raw_offset,f->bytes));e->delta_fallbacks++;e->fallback_copy_bytes+=f->bytes;p=*allocation;
        }else *packed=(ar_flex?(UINT64_C(3)<<62):(UINT64_C(1)<<63))|f->copy_bytes;
    }return p;
}
static void ar_write(ar_engine*e,ar_segment*s){
    if(!e->fp){char*path=malloc(strlen(e->directory)+80);if(!path)async_fatal("arena file path");
        sprintf(path,"%s/arena-%llu.nts",e->directory,(unsigned long long)e->request);e->fp=fopen(path,ar_writer_mode==2?"w+b":"wb");
        if(ar_diag){sprintf(path,"%s/arena-%llu.ids",e->directory,(unsigned long long)e->request);e->ids=fopen(path,"wb");if(!e->ids)async_fatal("arena IDs open");}
        free(path);e->file_position=0;e->stdio_buffer=malloc(ar_stdio_bytes);
        if(!e->fp||!e->stdio_buffer||setvbuf(e->fp,e->stdio_buffer,ar_writer_mode?_IONBF:_IOFBF,ar_stdio_bytes))async_fatal("arena output open");char magic[9]="NTGPUV2\n";if(ar_flex)snprintf(magic,sizeof(magic),ar_hier?"NTHR%u%u!\n":"NTFX%u%u!\n",ar_delta_width,ar_elapsed_width);if(ar_reference){unsigned lg=0;for(unsigned q=ar_tile;q>1;q>>=1)lg++;snprintf(magic,sizeof(magic),"NTRF%u%u%u\n",lg,ar_omit?2:ar_local_bits==12,ar_elapsed_width);}if(ar_nv)memcpy(magic,ar_nv==2?"NTNVCS1\n":"NTNVCA1\n",8);ar_put(e,(ar_flex||ar_nv)?magic:ar_delta?"NTGPUV2\n":"NTGPU13\n",8);
    }
    if(ar_writer_mode==2){
        // A fresh file extent is mapped for each consumed segment. All page
        // establishment and teardown are inside the same request completion.
        void*allocations[AR_FRAMES],*payloads[AR_FRAMES];size_t lengths[AR_FRAMES];uint64_t tags[AR_FRAMES];
        size_t total=0;uint64_t tick=ar_now();
        for(unsigned i=0;i<s->count;i++){payloads[i]=ar_file_payload(e,s,&s->frames[i],&lengths[i],&tags[i],&allocations[i]);total+=56+lengths[i];}
        uint64_t start=e->file_position;size_t offset=start&4095,span=offset+total;
        if(ftruncate(fileno(e->fp),start+total))async_fatal("arena mmap extend");
        void*map=mmap(NULL,span,PROT_READ|PROT_WRITE,MAP_SHARED,fileno(e->fp),start-offset);if(map==MAP_FAILED)async_fatal("arena mmap");
        if(ar_diag)e->mmap_setup_ns+=ar_now()-tick;tick=ar_now();char*dst=(char*)map+offset;
        for(unsigned i=0;i<s->count;i++){ar_frame*f=&s->frames[i];trace_header_t h={f->dims[0],f->dims[1],f->dims[2],f->dims[3],f->dims[4],f->dims[5],f->dims[6],1};trace_section_t sec={16,32,48};
            if(e->ids)fprintf(e->ids,"%llu,%d,%llu,%zu,%zu\n",(unsigned long long)f->launch,f->token,(unsigned long long)e->file_position,f->bytes,lengths[i]);
            memcpy(dst,&h,32);memcpy(dst+32,&sec,16);memcpy(dst+48,&tags[i],8);memcpy(dst+56,payloads[i],lengths[i]);dst+=56+lengths[i];e->file_position+=56+lengths[i];free(allocations[i]);}
        if(ar_diag)e->mmap_copy_ns+=ar_now()-tick;tick=ar_now();if(munmap(map,span))async_fatal("arena munmap");
        if(ar_diag)e->mmap_unmap_ns+=ar_now()-tick;e->storage_bytes+=total;e->mmap_bytes+=total;e->mmap_calls++;return;
    }
    if(ar_writer_mode==1){
        if(ar_check_first){uint64_t tick=ar_now();size_t allocated=0;
            for(unsigned i=0;i<s->count;i++){ar_frame*f=&s->frames[i];f->write_payload=ar_file_payload(e,s,f,&f->write_bytes,&f->write_tag,&f->write_allocation);if(f->write_allocation)allocated+=f->write_bytes;}
            ar_peak(&e->fallback_host_peak,allocated);if(ar_diag)e->check_first_ns+=ar_now()-tick;
            if(ar_release_checked){pthread_mutex_lock(&e->mutex);s->checked=1;e->checked_notifications++;pthread_cond_broadcast(&e->changed);pthread_mutex_unlock(&e->mutex);}
        }
        // At most 512 iovecs per writev. Handle partial writes and EINTR.
        unsigned char headers[256][56];struct iovec vectors[512];void*allocations[256];
        for(unsigned base=0;base<s->count;base+=256){uint64_t start=ar_now();unsigned count=s->count-base;if(count>256)count=256;
            for(unsigned j=0;j<count;j++){ar_frame*f=&s->frames[base+j];
                trace_header_t h={f->dims[0],f->dims[1],f->dims[2],f->dims[3],f->dims[4],f->dims[5],f->dims[6],1};trace_section_t sec={16,32,48};uint64_t packed;size_t bytes;void*payload;if(ar_check_first){payload=f->write_payload;bytes=f->write_bytes;packed=f->write_tag;allocations[j]=f->write_allocation;}else payload=ar_file_payload(e,s,f,&bytes,&packed,&allocations[j]);
                memcpy(headers[j],&h,32);memcpy(headers[j]+32,&sec,16);memcpy(headers[j]+48,&packed,8);
                if(e->ids)fprintf(e->ids,"%llu,%d,%llu,%zu,%zu\n",(unsigned long long)f->launch,f->token,(unsigned long long)e->file_position,f->bytes,bytes);
                e->file_position+=56+bytes;
                vectors[2*j]=(struct iovec){headers[j],56};vectors[2*j+1]=(struct iovec){payload,bytes};
            }
            if(ar_diag)e->writer_prepare_ns+=ar_now()-start;start=ar_now();unsigned pos=0;
            while(pos<2*count){ssize_t n=writev(fileno(e->fp),vectors+pos,2*count-pos);e->writer_syscalls++;
                if(n<0&&errno==EINTR)continue;if(n<=0)async_fatal("arena writev");e->storage_bytes+=n;
                while(pos<2*count&&(size_t)n>=vectors[pos].iov_len){n-=vectors[pos].iov_len;pos++;}
                if(n){vectors[pos].iov_base=(char*)vectors[pos].iov_base+n;vectors[pos].iov_len-=n;}
            }for(unsigned j=0;j<count;j++)free(allocations[j]);if(ar_diag)e->writer_output_ns+=ar_now()-start;
        }return;
    }
    for(unsigned i=0;i<s->count;i++){
        uint64_t tick=ar_now();
        ar_frame*f=&s->frames[i];trace_header_t h={f->dims[0],f->dims[1],f->dims[2],f->dims[3],f->dims[4],f->dims[5],f->dims[6],1};
        trace_section_t sec={16,32,48};uint64_t packed;size_t bytes;void*allocation;void*payload=ar_file_payload(e,s,f,&bytes,&packed,&allocation);
        if(e->ids)fprintf(e->ids,"%llu,%d,%llu,%zu,%zu\n",(unsigned long long)f->launch,f->token,(unsigned long long)e->file_position,f->bytes,bytes);
        ar_put(e,&h,sizeof(h));ar_put(e,&sec,sizeof(sec));ar_put(e,&packed,8);
        ar_put(e,payload,bytes);free(allocation);
        if(ar_diag){int b=f->bytes<=4096?0:f->bytes>=9000000?2:1;e->write_bucket_ns[b]+=ar_now()-tick;e->write_bucket_frames[b]++;e->write_bucket_bytes[b]+=f->copy_bytes;}
    }
}
static void *ar_writer(void*arg){
    ar_engine*e=arg;CUDA_CHECK(real_cuCtxSetCurrent(e->context));
    for(;;){pthread_mutex_lock(&e->mutex);
        while(e->consumed==e->published&&!e->stop)pthread_cond_wait(&e->changed,&e->mutex);
        if(e->consumed==e->published&&e->stop){pthread_mutex_unlock(&e->mutex);break;}
        ar_segment*s=&e->segment[e->consumed%AR_QUEUE];if(s->state!=2)async_fatal("arena writer sequence");s->state=3;
        pthread_mutex_unlock(&e->mutex);uint64_t cpu=ar_diag?async_clock(CLOCK_THREAD_CPUTIME_ID):0,t=ar_now();
        CUDA_CHECK(real_cuEventSynchronize(s->done));e->worker_syncs++;if(ar_diag)e->worker_wait_ns+=ar_now()-t;
        if(ar_nv){uint64_t tick=ar_now();size_t total=s->nv_meta[0];if(total>s->out_used)async_fatal("nvpipe bound or compression failure");size_t off=0;
            for(unsigned i=0;i<s->count;i++){ar_frame*f=&s->frames[i];f->out_offset=off;f->copy_bytes=s->nv_meta[i+1];off+=f->copy_bytes;}if(off!=total)async_fatal("nvpipe metadata sum");
            CUDA_CHECK(real_cuMemcpyDtoHAsync_v2((char*)e->host+s->host_base,e->out+s->out_base,total,e->receive));CUDA_CHECK(real_cuStreamSynchronize(e->receive));e->copies++;e->transfer_bytes+=total;e->copy_payload+=total;if(ar_diag)e->nv_payload_ns+=ar_now()-tick;
            if(ar_verify){tick=ar_now();for(unsigned i=0;i<s->count;i++){ar_frame*f=&s->frames[i];void*raw=malloc(f->bytes),*decoded=malloc(f->bytes);if(!raw||!decoded)async_fatal("nvpipe oracle allocation");CUDA_CHECK(real_cuMemcpyDtoH_v2(raw,e->raw+s->raw_base+f->raw_offset,f->bytes));void*payload=(char*)e->host+s->host_base+f->out_offset;
                if(f->codec){ar_external_depth++;int rc=nv_decode(payload,f->copy_bytes,decoded,f->bytes,ar_nv,ar_nv_chunk);ar_external_depth--;if(rc)async_fatal("nvpipe decoding");}else memcpy(decoded,payload,f->bytes);
                if(memcmp(raw,decoded,f->bytes))async_fatal("nvpipe same Map mismatch");ar_query_oracle(e,f,raw);free(raw);free(decoded);e->verified_frames++;}if(ar_diag)e->nv_validation_ns+=ar_now()-tick;}
        }
        t=ar_now();ar_write(e,s);if(ar_diag){e->worker_write_ns+=ar_now()-t;e->worker_cpu_ns+=async_clock(CLOCK_THREAD_CPUTIME_ID)-cpu;}
        pthread_mutex_lock(&e->mutex);e->host_ring.head=s->host_end;s->written=1;e->inflight_frames-=s->count;
        if(s->retired)s->state=0;e->consumed++;pthread_cond_broadcast(&e->changed);pthread_mutex_unlock(&e->mutex);
    }CUDA_CHECK(real_cuCtxSetCurrent(NULL));return NULL;
}
static ar_engine *ar_get(void){
    CUcontext ctx;CUDA_CHECK(real_cuCtxGetCurrent(&ctx));for(ar_engine*e=ar_engines;e;e=e->next)if(e->context==ctx)return e;
    ar_engine*e=calloc(1,sizeof(*e));if(!e)async_fatal("arena engine");e->context=ctx;e->directory=strdup(RESULT_DIR);
    e->raw_ring.capacity=e->out_ring.capacity=ar_device_bytes;e->host_ring.capacity=ar_host_bytes;
    CUDA_CHECK(real_cuMemAlloc_v2(&e->raw,ar_device_bytes));CUDA_CHECK(real_cuMemAlloc_v2(&e->out,ar_device_bytes));
    CUDA_CHECK(real_cuMemAllocHost_v2(&e->host,ar_host_bytes));CUDA_CHECK(real_cuMemAlloc_v2(&e->error,4));CUDA_CHECK(real_cuMemsetD32_v2(e->error,0,1));
    CUDA_CHECK(real_cuStreamCreate(&e->copy,CU_STREAM_NON_BLOCKING));const char*path=getenv("NEUTRINO_CAPTURE_ENCODER");if(!path)async_fatal("arena encoder missing");
    CUDA_CHECK(real_cuModuleLoad(&e->module,path));CUDA_CHECK(real_cuModuleGetFunction(&e->encoder,e->module,"neutrino_capture_pack"));
    if(ar_batch_encoder)CUDA_CHECK(real_cuModuleGetFunction(&e->batch_encoder,e->module,"neutrino_capture_pack_batch"));
    if(ar_delta){CUDA_CHECK(real_cuModuleGetFunction(&e->delta_init,e->module,ar_omit?"neutrino_capture_compact_init":ar_reference?"neutrino_capture_reference_init":ar_hier?"neutrino_capture_hier_init":ar_flex?"neutrino_capture_flex_init":"neutrino_capture_delta_init"));CUDA_CHECK(real_cuModuleGetFunction(&e->delta_encoder,e->module,ar_omit?"neutrino_capture_compact_batch":ar_reference?"neutrino_capture_reference_batch":ar_hier?"neutrino_capture_hier_batch":ar_flex?"neutrino_capture_flex_batch":"neutrino_capture_delta_batch"));CUDA_CHECK(real_cuMemAlloc_v2(&e->delta_indices,ar_frames*256*sizeof(unsigned)));}
    if(ar_nv){ar_nv_load();ar_external_depth++;e->nv=nv_create(ar_device_bytes,ar_frames,ar_nv_chunk,ar_nv);ar_external_depth--;CUDA_CHECK(real_cuStreamCreate(&e->receive,CU_STREAM_NON_BLOCKING));}
    pthread_mutex_init(&e->mutex,NULL);pthread_cond_init(&e->changed,NULL);
    for(int i=0;i<AR_QUEUE;i++){CUDA_CHECK(real_cuEventCreate(&e->segment[i].ready,CU_EVENT_DISABLE_TIMING));CUDA_CHECK(real_cuEventCreate(&e->segment[i].done,CU_EVENT_DISABLE_TIMING));}
    e->next=ar_engines;ar_engines=e;if(pthread_create(&e->writer,NULL,ar_writer,e))async_fatal("arena writer create");return e;
}
/* Producer alone retires device spans. Descriptor/event remains alive until
 * both retirement and FIFO host consumption, preventing event-record reuse. */
static void ar_retire(ar_engine*e,int wait){
    while(e->retired<e->published){ar_segment*s=&e->segment[e->retired%AR_QUEUE];
        pthread_mutex_lock(&e->mutex);int written=s->written,checked=s->checked;pthread_mutex_unlock(&e->mutex);
        if(!ar_decouple&&!written&&!(ar_release_checked&&checked))break;
        if(ar_decouple&&!written){CUresult rc;
            if(wait){uint64_t t=ar_now();CUDA_CHECK(real_cuEventSynchronize(s->done));e->device_syncs++;if(ar_diag)e->device_wait_ns+=ar_now()-t;rc=CUDA_SUCCESS;wait=0;}
            else {rc=real_cuEventQuery(s->done);e->queries++;}
            if(rc==CUDA_ERROR_NOT_READY)break;if(rc!=CUDA_SUCCESS)async_fatal("arena completion query");
        }
        pthread_mutex_lock(&e->mutex);if(e->raw_ring.head<s->raw_end)e->raw_ring.head=s->raw_end;if(e->out_ring.head<s->out_end)e->out_ring.head=s->out_end;s->retired=1;if(s->written)s->state=0;e->retired++;pthread_mutex_unlock(&e->mutex);
    }
}
static ar_segment *ar_reserve(ar_engine*e,size_t raw,size_t out,CUstream stream){
    ar_segment*s=&e->segment[e->published%AR_QUEUE];uint64_t rb,re,ob,oe,hb,he;
    for(;;){ar_retire(e,0);pthread_mutex_lock(&e->mutex);
        int hf=ar_fit(&e->host_ring,out,&hb,&he),rf=ar_fit(&e->raw_ring,raw,&rb,&re),of=ar_fit(&e->out_ring,out,&ob,&oe);
        if(!s->state&&hf&&rf&&of)break;
        if(ar_decouple&&(!rf||!of)&&e->retired<e->published){e->device_waits++;pthread_mutex_unlock(&e->mutex);ar_retire(e,1);continue;}
        if(!hf)e->host_waits++;if(!rf||!of)e->device_waits++;if(s->state)e->descriptor_waits++;
        uint64_t t=ar_now();ar_stall entry={.incoming=e->incoming_bytes,.kind=(!hf?1:0)|(!rf||!of?2:0)|(s->state?4:0)};
        if(ar_diag&&e->consumed<e->published){ar_segment*h=&e->segment[e->consumed%AR_QUEUE];entry.head_state=h->state;for(unsigned i=0;i<h->count;i++)if(h->frames[i].bytes>entry.head_max)entry.head_max=h->frames[i].bytes;}
        pthread_cond_wait(&e->changed,&e->mutex);if(ar_diag){uint64_t ns=ar_now()-t;e->host_wait_ns+=ns;unsigned i;for(i=0;i<e->nstalls;i++){ar_stall*x=&e->stalls[i];if(x->incoming==entry.incoming&&x->head_max==entry.head_max&&x->head_state==entry.head_state&&x->kind==entry.kind)break;}if(i==128)async_fatal("arena stall diagnostic bound");if(i==e->nstalls)e->stalls[e->nstalls++]=entry;e->stalls[i].waits++;e->stalls[i].ns+=ns;}pthread_mutex_unlock(&e->mutex);
    }
    e->host_ring.tail=he;e->raw_ring.tail=re;e->out_ring.tail=oe;e->reserved_raw_bytes+=raw;e->reserved_out_bytes+=out;
    s->raw_end=re;s->out_end=oe;s->host_end=he;s->raw_base=rb%e->raw_ring.capacity;s->out_base=ob%e->out_ring.capacity;s->host_base=hb%e->host_ring.capacity;
    s->raw_limit=raw;s->out_limit=out;s->raw_used=s->out_used=s->count=0;s->born=ar_age_ns?async_clock(CLOCK_MONOTONIC):0;s->target=ar_frames;s->plan_start=e->plan_pos;s->state=1;s->written=s->retired=s->checked=0;s->stream=stream;
    ar_peak(&e->peak_host,he-e->host_ring.head);ar_peak(&e->peak_raw,re-e->raw_ring.head);ar_peak(&e->peak_out,oe-e->out_ring.head);
    pthread_mutex_unlock(&e->mutex);if(s->frame_capacity<(unsigned)ar_frames){void*p=realloc(s->frames,ar_frames*sizeof(ar_frame));if(!p)async_fatal("arena frame metadata");s->frames=p;s->frame_capacity=ar_frames;}
    if(ar_batch_encoder&&!s->descriptors){CUDA_CHECK(real_cuMemAllocHost_v2((void**)&s->descriptors,ar_frames*sizeof(ar_pack_desc)));CUDA_CHECK(real_cuMemAlloc_v2(&s->device_descriptors,ar_frames*sizeof(ar_pack_desc)));}return s;
}
static void ar_verify_segment(ar_engine*e,ar_segment*s){
    CUDA_CHECK(real_cuEventSynchronize(s->done));
    for(unsigned j=0;j<s->count;j++){ar_frame*f=&s->frames[j];if(!f->codec){ar_query_oracle(e,f,(char*)e->host+s->host_base+f->out_offset);continue;}
        char*raw=malloc(f->bytes);if(!raw)async_fatal("arena validation allocation");CUDA_CHECK(real_cuMemcpyDtoH_v2(raw,e->raw+s->raw_base+f->raw_offset,f->bytes));
        ar_query_oracle(e,f,raw);
        char*packed=(char*)e->host+s->host_base+f->out_offset;size_t n=f->bytes/16;
        unsigned flags=0;if(ar_delta)memcpy(&flags,packed,4);
        for(size_t i=0;i<n;i++){uint64_t a,b;uint32_t c,d,sm,id;memcpy(&a,raw+i*16,8);memcpy(&c,raw+i*16+8,4);memcpy(&sm,raw+i*16+12,4);
            if(ar_hier && ((unsigned*)packed)[3]>=2){if(flags)continue;unsigned wp=((unsigned*)packed)[1],ew=ar_elapsed_width,version=((unsigned*)packed)[3];size_t group=i/wp,groups=(n+wp-1)/wp;uint32_t bits=0;for(unsigned k=0;k<3;k++)bits|=(uint32_t)(unsigned char)packed[2064+group*4+k]<<(8*k);int32_t dt=(int32_t)(bits<<8)>>8;id=(unsigned char)packed[2064+group*4+3];uint64_t base;memcpy(&base,packed+16+id*8,8);int32_t loc;size_t locals;
                if(version==4||version==5){locals=((n-(version==5?groups:0))*12+7)/8;loc=0;if(version!=5||i%wp){size_t q=version==5?i-i/wp-1:i,bit=q*12,byte=bit/8;unsigned shift=bit%8;uint32_t v=(unsigned char)packed[2064+groups*4+byte]|((uint32_t)(unsigned char)packed[2064+groups*4+byte+1]<<8);loc=(int32_t)(((v>>shift)&4095)^2048)-2048;}}
                else{int16_t v;memcpy(&v,packed+2064+groups*4+i*2,2);loc=v;locals=n*2;}
                b=base+(uint64_t)(int64_t)dt+(uint64_t)(int64_t)loc;d=0;for(unsigned k=0;k<ew;k++)d|=(uint32_t)(unsigned char)packed[2064+groups*4+locals+i*ew+k]<<(8*k);}
            else if(ar_delta){if(flags)continue;uint64_t base;uint32_t bits=0;d=0;unsigned dw=ar_delta_width,ew=ar_elapsed_width;id=(unsigned char)packed[2064+n*(dw+ew)+i];memcpy(&base,packed+16+id*8,8);
                for(unsigned k=0;k<dw;k++)bits|=(uint32_t)(unsigned char)packed[2064+i*dw+k]<<(k*8);int32_t delta=(int32_t)(bits<<(32-dw*8))>>(32-dw*8);b=base+(uint64_t)(int64_t)delta;
                for(unsigned k=0;k<ew;k++)d|=(uint32_t)(unsigned char)packed[2064+n*dw+i*ew+k]<<(k*8);}
            else {memcpy(&b,packed+i*8,8);memcpy(&d,packed+n*8+i*4,4);id=(unsigned char)packed[n*12+i];}
            if(a!=b||c!=d||sm!=id)async_fatal("arena GPU codec mismatch");}
        free(raw);e->verified_frames++;
    }
}
static void ar_flush(ar_engine*e){
    ar_segment*s=e->pending;if(!s)return;uint64_t begin=ar_now(),t;
    if(ar_trim&&!ar_incremental){
        pthread_mutex_lock(&e->mutex);size_t raw=s->raw_limit-s->raw_used,out=s->out_limit-s->out_used;
        if(e->raw_ring.tail!=s->raw_end||e->out_ring.tail!=s->out_end||e->host_ring.tail!=s->host_end)async_fatal("arena trim ownership");
        s->raw_end-=raw;s->out_end-=out;s->host_end-=out;e->raw_ring.tail=s->raw_end;e->out_ring.tail=s->out_end;e->host_ring.tail=s->host_end;e->trimmed_raw_bytes+=raw;e->trimmed_out_bytes+=out;pthread_mutex_unlock(&e->mutex);
    }
    if(e->plan_learning){ar_plan*p=&e->plan[s->plan_start];p->raw=s->raw_used;p->out=s->out_used;p->count=s->count;}
    CUDA_CHECK(real_cuEventRecord(s->ready,s->stream));e->ready_records++;CUDA_CHECK(real_cuStreamWaitEvent(e->copy,s->ready,0));e->stream_waits++;
    t=ar_now();if(ar_nv){ar_nv_frame frames[AR_FRAMES];for(unsigned i=0;i<s->count;i++){ar_frame*f=&s->frames[i];frames[i]=(ar_nv_frame){e->raw+s->raw_base+f->raw_offset,f->bytes};e->encoder_frames+=f->codec;}
        ar_external_depth++;s->nv_meta=nv_submit(e->nv,(unsigned)(s-e->segment),frames,s->count,ar_codec_min,e->out+s->out_base,e->copy);ar_external_depth--;e->nv_metadata_bytes+=(s->count+1)*8;e->encoders++;if(ar_diag)e->encode_enqueue_ns+=ar_now()-t;goto ar_finish_copy;}
    unsigned encoded=0,blocks=0;
    for(unsigned i=0;i<s->count;i++){ar_frame*f=&s->frames[i];if(!f->codec)continue;e->encoder_frames++;
        unsigned long long n=f->bytes/16;unsigned nb=(unsigned)((n+(ar_reference?1023:255))/(ar_reference?1024:256));
        if(ar_batch_encoder)s->descriptors[encoded]=(ar_pack_desc){e->raw+s->raw_base+f->raw_offset,e->out+s->out_base+f->out_offset,n,blocks,ar_hier?(ar_tile?(unsigned)ar_tile:(f->dims[3]*f->dims[4]*f->dims[5]+31)/32):nb};encoded++;blocks+=nb;
    }
    if(ar_batch_encoder&&encoded){size_t bytes=encoded*sizeof(ar_pack_desc);CUDA_CHECK(real_cuMemcpyHtoDAsync_v2(s->device_descriptors,s->descriptors,bytes,e->copy));e->descriptor_bytes+=bytes;}
    if(ar_delta&&encoded){void*init[]={&s->device_descriptors,&e->delta_indices,ar_reference?&ar_elapsed_width:&ar_delta_width,ar_reference?&ar_local_bits:&ar_elapsed_width};CUDA_CHECK(real_cuLaunchKernel(e->delta_init,encoded,1,1,256,1,1,0,e->copy,init,NULL));e->delta_initializers++;
        void*args[]={&s->device_descriptors,&encoded,&e->delta_indices,ar_reference?&ar_elapsed_width:&ar_delta_width,ar_reference?&ar_local_bits:&ar_elapsed_width};CUDA_CHECK(real_cuLaunchKernel(e->delta_encoder,blocks,1,1,256,1,1,0,e->copy,args,NULL));e->encoders++;
    }else if(ar_batch_encoder==1&&encoded){void*args[]={&s->device_descriptors,&encoded,&e->error};CUDA_CHECK(real_cuLaunchKernel(e->batch_encoder,blocks,1,1,256,1,1,0,e->copy,args,NULL));e->encoders++;}
    else for(unsigned i=0;i<s->count;i++){ar_frame*f=&s->frames[i];if(!f->codec)continue;
        CUdeviceptr source=e->raw+s->raw_base+f->raw_offset,dest=e->out+s->out_base+f->out_offset;unsigned long long n=f->bytes/16;void*args[]={&source,&dest,&n,&e->error};
        CUDA_CHECK(real_cuLaunchKernel(e->encoder,(unsigned)((n+255)/256),1,1,256,1,1,0,e->copy,args,NULL));e->encoders++;
    }if(ar_diag)e->encode_enqueue_ns+=ar_now()-t;
    t=ar_now();if(ar_frame_copies){for(unsigned i=0;i<s->count;i++){ar_frame*f=&s->frames[i];CUDA_CHECK(real_cuMemcpyDtoHAsync_v2((char*)e->host+s->host_base+f->out_offset,e->out+s->out_base+f->out_offset,f->copy_bytes,e->copy));e->copies++;e->transfer_bytes+=f->copy_bytes;}}
    else {CUDA_CHECK(real_cuMemcpyDtoHAsync_v2((char*)e->host+s->host_base,e->out+s->out_base,s->out_used,e->copy));e->copies++;e->transfer_bytes+=s->out_used;}
    if(ar_diag)e->copy_enqueue_ns+=ar_now()-t;
ar_finish_copy:
    CUDA_CHECK(real_cuEventRecord(s->done,e->copy));e->done_records++;
    if(ar_verify&&!ar_nv)ar_verify_segment(e,s);
    pthread_mutex_lock(&e->mutex);s->state=2;e->published++;e->batches++;e->inflight_frames+=s->count;ar_peak(&e->peak_segments,e->published-e->consumed);ar_peak(&e->peak_frames,e->inflight_frames);pthread_cond_broadcast(&e->changed);pthread_mutex_unlock(&e->mutex);
    e->pending=NULL;if(ar_diag)e->flush_ns+=ar_now()-begin;
}
static CUdeviceptr ar_prepare(ar_engine*e,size_t n,CUstream stream,unsigned*dims,CUfunction function){
    int codec=n>=ar_codec_min;size_t wp=ar_tile?(size_t)ar_tile:(dims[3]*dims[4]*dims[5]+31)/32;size_t raw=codec?n:0,out=codec?(ar_delta?n/16*(ar_delta_width+ar_elapsed_width+1)+2064:n/16*13):n;if(codec&&ar_hier&&wp>2)out=2064+((n/16+wp-1)/wp)*4+((n/16-(ar_omit?(n/16+wp-1)/wp:0))*(ar_reference?ar_local_bits:16)+7)/8+n/16*ar_elapsed_width;ar_segment*s=e->pending;
    if(ar_nv){raw=n;out=nv_bound(e->nv,n,ar_codec_min);}
    e->incoming_bytes=n;
    if(e->plan_using){ar_plan*p=e->plan_pos<e->plan_count?&e->plan[e->plan_pos]:NULL;
        if(!p||p->bytes!=n||p->function!=function||p->stream!=stream||memcmp(p->dims,dims,7*sizeof(unsigned))){
            ar_flush(e);s=NULL;e->plan_using=e->plan_valid=0;e->template_invalidations++;
        }else e->template_hits++;
    }
    if(n>ar_chunk)async_fatal("Map exceeds configured segment bound");
    if(s&&ar_soft_bytes&&s->out_used+ar_align(out)>ar_soft_bytes){e->flush_bytes++;ar_flush(e);s=NULL;}
    if(s&&ar_age_ns&&async_clock(CLOCK_MONOTONIC)-s->born>=ar_age_ns){e->flush_age++;ar_flush(e);s=NULL;}
    if(s&&(s->stream!=stream||s->count>=(unsigned)ar_frames||raw>s->raw_limit-s->raw_used||ar_align(out)>s->out_limit-s->out_used)){
        if(s->stream!=stream)e->flush_stream++;else if(s->count>=(unsigned)ar_frames)e->flush_count++;else e->flush_full++;ar_flush(e);s=NULL;
    }
    if(s&&ar_incremental){size_t additional=ar_align(out);pthread_mutex_lock(&e->mutex);
        int fit=s->raw_base+s->raw_used+raw<=e->raw_ring.capacity&&s->out_base+s->out_used+additional<=e->out_ring.capacity&&s->host_base+s->out_used+additional<=e->host_ring.capacity&&
            s->raw_end+raw-e->raw_ring.head<=e->raw_ring.capacity&&s->out_end+additional-e->out_ring.head<=e->out_ring.capacity&&s->host_end+additional-e->host_ring.head<=e->host_ring.capacity;
        if(fit){s->raw_end+=raw;s->out_end+=additional;s->host_end+=additional;e->raw_ring.tail=s->raw_end;e->out_ring.tail=s->out_end;e->host_ring.tail=s->host_end;e->incremental_extensions++;
            e->reserved_raw_bytes+=raw;e->reserved_out_bytes+=additional;ar_peak(&e->peak_raw,s->raw_end-e->raw_ring.head);ar_peak(&e->peak_out,s->out_end-e->out_ring.head);ar_peak(&e->peak_host,s->host_end-e->host_ring.head);}
        pthread_mutex_unlock(&e->mutex);if(!fit){e->flush_pressure++;ar_flush(e);s=NULL;}
    }
    if(!s){size_t r=(ar_frames==1||ar_incremental)?raw:ar_chunk,o=(ar_frames==1||ar_incremental)?ar_align(out):ar_chunk;unsigned target=ar_frames;
        if(e->plan_using){ar_plan*p=&e->plan[e->plan_pos];if(!p->count||p->raw<raw||p->out<ar_align(out))async_fatal("invalid learned segment");r=p->raw;o=p->out;target=p->count;}
        s=ar_reserve(e,r,o,stream);s->target=target;if(ar_incremental)s->raw_limit=s->out_limit=ar_chunk;e->pending=s;}
    if(e->plan_learning){if(e->plan_pos==e->plan_capacity){unsigned cap=e->plan_capacity?e->plan_capacity*2:1024;if(cap>65536)async_fatal("template bound exceeded");void*p=realloc(e->plan,cap*sizeof(ar_plan));if(!p)async_fatal("template allocation");e->plan=p;e->plan_capacity=cap;}
        ar_plan*p=&e->plan[e->plan_pos];memset(p,0,sizeof(*p));p->function=function;p->stream=stream;p->bytes=n;memcpy(p->dims,dims,7*sizeof(unsigned));}
    ar_frame*f=&s->frames[s->count++];memcpy(f->dims,dims,7*sizeof(unsigned));f->bytes=n;f->copy_bytes=out;f->codec=codec;f->raw_offset=s->raw_used;f->out_offset=s->out_used;f->launch=e->launch++;f->token=ar_token;
    s->raw_used+=raw;s->out_used+=ar_align(out);e->calls++;e->payload+=n;if(!ar_nv)e->copy_payload+=out;if(ar_template)e->plan_pos++;
    return (codec||ar_nv)?e->raw+s->raw_base+f->raw_offset:e->out+s->out_base+f->out_offset;
}
static void ar_commit(ar_engine*e){if(e->pending&&e->pending->count>=e->pending->target){e->flush_count++;ar_flush(e);}}
static void ar_mark(int token){
    for(ar_engine*e=ar_engines;e;e=e->next){
        if(ar_token_flush||ar_template){if(e->pending)e->flush_token++;ar_flush(e);}
        if(ar_template){
            if(ar_token>0&&e->plan_learning&&e->plan_pos){e->plan_count=e->plan_pos;e->plan_valid=1;e->template_builds++;}
            if(e->plan_using&&e->plan_pos!=e->plan_count){e->plan_valid=0;e->template_invalidations++;}
            e->plan_pos=0;e->plan_using=token>0&&e->plan_valid;e->plan_learning=token>0&&!e->plan_valid;
        }
    }ar_token=token;
}
static int ar_boundary(void){for(ar_engine*e=ar_engines;e;e=e->next){uint64_t t=ar_now();
    pthread_mutex_lock(&e->mutex);e->debt_bytes+=e->host_ring.tail-e->host_ring.head;e->debt_segments+=e->published-e->consumed+(e->pending!=NULL);e->debt_requests++;pthread_mutex_unlock(&e->mutex);if(e->pending)e->flush_boundary++;ar_flush(e);
    pthread_mutex_lock(&e->mutex);while(e->consumed!=e->published)pthread_cond_wait(&e->changed,&e->mutex);pthread_mutex_unlock(&e->mutex);ar_retire(e,0);
    if(e->retired!=e->published)async_fatal("arena boundary retirement");
    if(e->fp){int bad=ferror(e->fp);if(fclose(e->fp))bad=1;e->fp=NULL;free(e->stdio_buffer);e->stdio_buffer=NULL;if(bad)return -1;}
    if(e->ids){if(fclose(e->ids))return -1;e->ids=NULL;}
    unsigned error=0;CUDA_CHECK(real_cuMemcpyDtoH_v2(&error,e->error,4));if(error)async_fatal("arena SM truncation");e->request++;if(ar_diag)e->drain_ns+=ar_now()-t;
    if(e->raw_ring.head!=e->raw_ring.tail||e->out_ring.head!=e->out_ring.tail||e->host_ring.head!=e->host_ring.tail)async_fatal("arena byte leak");
}return 0;}
static void ar_reset(void){for(ar_engine*e=ar_engines;e;e=e->next){
    /* Counters form one contiguous zeroable range, all engines drained. */
    memset(&e->calls,0,(char*)&e->next-(char*)&e->calls);e->launch=0;e->nstalls=0;memset(e->stalls,0,sizeof(e->stalls));
}}
static void ar_json(FILE*fp){fputs(",\"arena_engines\":[",fp);int sep=0;for(ar_engine*e=ar_engines;e;e=e->next){
    fprintf(fp,"%s{\"host_capacity_bytes\":%zu,\"device_capacity_bytes\":%zu,\"segment_bytes\":%zu,\"max_frames\":%d,\"decoupled\":%d,\"diagnostic\":%d",sep?",":"",ar_host_bytes,ar_device_bytes*2+4,ar_chunk,ar_frames,ar_decouple,ar_diag);
#define AR_JSON(x) fprintf(fp,",\"" #x "\":%llu",(unsigned long long)e->x)
    fprintf(fp,",\"omit_anchor_locals\":%d",ar_omit);fprintf(fp,",\"hotpath_mode\":%d",ar_hot);AR_JSON(argument_ns);AR_JSON(attribute_calls);AR_JSON(attribute_skips);AR_JSON(argument_heap_calls);AR_JSON(nv_metadata_bytes);AR_JSON(nv_payload_ns);AR_JSON(nv_validation_ns);AR_JSON(forced_fallbacks);AR_JSON(checked_notifications);AR_JSON(check_first_ns);AR_JSON(fallback_host_peak);
    AR_JSON(mmap_setup_ns);AR_JSON(mmap_copy_ns);AR_JSON(mmap_unmap_ns);AR_JSON(mmap_calls);AR_JSON(mmap_bytes);
    AR_JSON(debt_bytes);AR_JSON(debt_segments);AR_JSON(debt_requests);
    AR_JSON(delta_initializers);AR_JSON(delta_fallbacks);AR_JSON(fallback_copy_bytes);
    AR_JSON(encoder_frames);AR_JSON(descriptor_bytes);AR_JSON(writer_syscalls);AR_JSON(writer_prepare_ns);AR_JSON(writer_output_ns);AR_JSON(flush_bytes);AR_JSON(flush_age);
    AR_JSON(calls);AR_JSON(payload);AR_JSON(copy_payload);AR_JSON(transfer_bytes);AR_JSON(storage_bytes);AR_JSON(batches);AR_JSON(encoders);AR_JSON(copies);AR_JSON(ready_records);AR_JSON(done_records);AR_JSON(stream_waits);AR_JSON(queries);AR_JSON(device_syncs);AR_JSON(worker_syncs);AR_JSON(host_waits);AR_JSON(device_waits);AR_JSON(descriptor_waits);AR_JSON(host_wait_ns);AR_JSON(device_wait_ns);AR_JSON(prepare_ns);AR_JSON(model_ns);AR_JSON(clear_ns);AR_JSON(encode_enqueue_ns);AR_JSON(copy_enqueue_ns);AR_JSON(flush_ns);AR_JSON(worker_wait_ns);AR_JSON(worker_write_ns);AR_JSON(worker_cpu_ns);AR_JSON(drain_ns);AR_JSON(verified_frames);AR_JSON(peak_host);AR_JSON(peak_raw);AR_JSON(peak_out);AR_JSON(peak_segments);AR_JSON(peak_frames);AR_JSON(flush_full);AR_JSON(flush_count);AR_JSON(flush_stream);AR_JSON(flush_token);AR_JSON(flush_boundary);AR_JSON(template_hits);AR_JSON(template_invalidations);AR_JSON(template_builds);AR_JSON(reserved_raw_bytes);AR_JSON(reserved_out_bytes);AR_JSON(trimmed_raw_bytes);AR_JSON(trimmed_out_bytes);AR_JSON(incremental_extensions);AR_JSON(flush_pressure);
#undef AR_JSON
    fputs(",\"write_size_buckets\":[",fp);for(int i=0;i<3;i++)fprintf(fp,"%s{\"frames\":%llu,\"bytes\":%llu,\"ns\":%llu}",i?",":"",(unsigned long long)e->write_bucket_frames[i],(unsigned long long)e->write_bucket_bytes[i],(unsigned long long)e->write_bucket_ns[i]);
    fputs("],\"condition_wait_owners\":[",fp);for(unsigned i=0;i<e->nstalls;i++){ar_stall*x=&e->stalls[i];fprintf(fp,"%s{\"incoming_bytes\":%zu,\"head_max_bytes\":%zu,\"head_state\":%d,\"reason_bits\":%d,\"waits\":%llu,\"ns\":%llu}",i?",":"",x->incoming,x->head_max,x->head_state,x->kind,(unsigned long long)x->waits,(unsigned long long)x->ns);}fputc(']',fp);
    fprintf(fp,",\"delta_codec\":%d,\"delta_scratch_bytes\":%zu",ar_delta,ar_delta?ar_frames*256*sizeof(unsigned):0);
    fprintf(fp,",\"delta_width\":%u,\"elapsed_width\":%u,\"flex_codec\":%d",ar_delta_width,ar_elapsed_width,ar_flex);
    fprintf(fp,",\"reference_codec\":%d,\"tile_records\":%d,\"local_bits\":%u,\"codec_min_bytes\":%zu",ar_reference,ar_tile,ar_local_bits,ar_codec_min);
    fputs(",\"escape_reasons\":[",fp);for(unsigned b=0;b<5;b++)fprintf(fp,"%s{\"flag\":%u,\"frames\":%llu,\"raw_bytes\":%llu}",b?",":"",1U<<b,(unsigned long long)e->escape_frames[b],(unsigned long long)e->escape_raw_bytes[b]);fputc(']',fp);
    uint64_t nvbytes[5]={0};if(ar_nv)nv_space(e->nv,nvbytes);fprintf(fp,",\"nvcomp_mode\":%d,\"nvcomp_chunk\":%u,\"nvcomp_device_workspace\":%llu,\"nvcomp_host_workspace\":%llu,\"nvcomp_slots\":%llu",ar_nv,ar_nv_chunk,(unsigned long long)nvbytes[0],(unsigned long long)nvbytes[1],(unsigned long long)nvbytes[2]);
    size_t descriptor_capacity=0;for(int i=0;i<AR_QUEUE;i++)if(e->segment[i].descriptors)descriptor_capacity+=ar_frames*sizeof(ar_pack_desc);
    fprintf(fp,",\"descriptor_host_bytes\":%zu,\"descriptor_device_bytes\":%zu,\"batch_encoder\":%d,\"writer_mode\":%d,\"stdio_bytes\":%zu",descriptor_capacity,descriptor_capacity,ar_batch_encoder,ar_writer_mode,ar_stdio_bytes);
    size_t metadata=sizeof(*e)+e->plan_capacity*sizeof(ar_plan);for(int i=0;i<AR_QUEUE;i++)metadata+=e->segment[i].frame_capacity*sizeof(ar_frame);
    fprintf(fp,",\"metadata_capacity_bytes\":%zu,\"published_total\":%llu,\"consumed_total\":%llu,\"retired_total\":%llu,\"outstanding_bytes\":%llu}",metadata,(unsigned long long)e->published,(unsigned long long)e->consumed,(unsigned long long)e->retired,(unsigned long long)(e->host_ring.tail-e->host_ring.head+e->raw_ring.tail-e->raw_ring.head+e->out_ring.tail-e->out_ring.head));sep=1;}fputc(']',fp);}
static void ar_release(void){CUcontext saved;CUDA_CHECK(real_cuCtxGetCurrent(&saved));while(ar_engines){ar_engine*e=ar_engines;
    pthread_mutex_lock(&e->mutex);e->stop=1;pthread_cond_broadcast(&e->changed);pthread_mutex_unlock(&e->mutex);pthread_join(e->writer,NULL);CUDA_CHECK(real_cuCtxSetCurrent(e->context));
    for(int i=0;i<AR_QUEUE;i++){CUDA_CHECK(real_cuEventDestroy_v2(e->segment[i].ready));CUDA_CHECK(real_cuEventDestroy_v2(e->segment[i].done));free(e->segment[i].frames);if(e->segment[i].descriptors){CUDA_CHECK(real_cuMemFreeHost(e->segment[i].descriptors));CUDA_CHECK(real_cuMemFree_v2(e->segment[i].device_descriptors));}}
    if(ar_nv){ar_external_depth++;nv_destroy(e->nv);ar_external_depth--;CUDA_CHECK(real_cuStreamDestroy_v2(e->receive));}
    if(e->delta_indices)CUDA_CHECK(real_cuMemFree_v2(e->delta_indices));
    CUDA_CHECK(real_cuMemFreeHost(e->host));CUDA_CHECK(real_cuMemFree_v2(e->raw));CUDA_CHECK(real_cuMemFree_v2(e->out));CUDA_CHECK(real_cuMemFree_v2(e->error));CUDA_CHECK(real_cuModuleUnload(e->module));CUDA_CHECK(real_cuStreamDestroy_v2(e->copy));
    pthread_mutex_destroy(&e->mutex);pthread_cond_destroy(&e->changed);ar_engines=e->next;free(e->directory);free(e->plan);free(e);
}CUDA_CHECK(real_cuCtxSetCurrent(saved));}
#endif
