/* Experimental fixed block_sched backend. Each frame is retained verbatim.
 * Consecutive small Maps share an arena and one copy/completion dependency.
 * Optional completion thread releases device arenas before CPU output.
 * This is explicit opt-in, one submitting thread per engine, no CUDA Graphs.
 */
#ifndef NEUTRINO_BULK_CAPTURE_H
#define NEUTRINO_BULK_CAPTURE_H
#include "capture_codec.h"
static int ab_bulk,bulk_batch_frames=1,bulk_step_flush,bulk_decouple;
static int bulk_sink,bulk_compress,bulk_verify_codec;
static int bulk_hot_cache;
static int bulk_clear_policy; /* 0 zero; 1 proven full-write probe only; 2 sentinel diagnostic. */
static int bulk_small_hosts=16,bulk_large_hosts=16,bulk_small_devices=16,bulk_large_devices=16;
static size_t bulk_small_bytes=131072,bulk_large_bytes=9723904,bulk_map_cutoff=4096;
static __thread int bulk_token;
enum { BULK_MAX_FRAMES=256, BULK_MAX_TOKENS=4096 };
typedef struct { unsigned dims[7];size_t offset,bytes; } bulk_frame;
typedef struct { CUdeviceptr ptr;int busy; } bulk_device;
typedef struct {
    void *host;CUevent kernel_done,done;int state,kind,device_index;
    size_t bytes;unsigned frames;CUstream producer;
    bulk_frame frame[BULK_MAX_FRAMES];
} bulk_host;
typedef struct { uint64_t calls,bytes,signature; } bulk_phase;
typedef struct bulk_engine {
    CUcontext context;CUstream copy;pthread_t writer,completer;pthread_mutex_t mutex;
    pthread_cond_t changed;bulk_host *host[2];bulk_device *device[2];int nh[2],nd[2];
    bulk_host **queue,*pending;unsigned queue_cap;uint64_t published,completed,copy_completed;
    int stop;FILE *fp;char *stdio_buffer,*directory;uint64_t file_seq;
    void *ram;size_t ram_capacity,ram_used,ram_last_size;void *encoded,*decoded;size_t codec_capacity;
    void *zlib;size_t(*zbound)(size_t);size_t(*zcompress)(void*,size_t,const void*,size_t,int);
    size_t(*zdecompress)(void*,size_t,const void*,size_t);unsigned(*zerror)(size_t);
    uint64_t calls,payload,logical_trace,storage_bytes,batches,peak_queue;
    uint64_t host_wait_ns,device_wait_ns,copy_submit_ns,drain_ns,event_wait_ns,write_ns,encode_ns,verify_ns,worker_cpu_ns;
    uint64_t early_releases,codec_verified_frames,codec_raw_frames,codec_zero_records,codec_global_wide_frames;
    void **args;size_t args_capacity;CUfunction attr_func[64];unsigned attr_value[64];int attr_count;
    uint64_t attr_calls,attr_hits;
    bulk_phase phase[BULK_MAX_TOKENS];struct bulk_engine *next;
} bulk_engine;
static __thread bulk_engine *bulk_engines;
static size_t bulk_capacity(int k){return k?bulk_large_bytes:bulk_small_bytes;}
static void bulk_put(bulk_engine*e,const void*p,size_t n){
    if(bulk_sink){
        if(n>e->ram_capacity-e->ram_used)async_fatal("complete RAM receiver capacity exceeded");
        memcpy((char*)e->ram+e->ram_used,p,n);e->ram_used+=n;
    }else if(fwrite(p,1,n,e->fp)!=n)async_fatal("bulk output write");
    e->storage_bytes+=n;
}
static void bulk_output_open(bulk_engine*e){
    if(bulk_sink){
        if(!e->ram){e->ram_capacity=768ULL*1024*1024;e->ram=malloc(e->ram_capacity);if(!e->ram)async_fatal("RAM allocation");}
        return;
    }
    if(e->fp)return;
    char *path=malloc(strlen(e->directory)+64);if(!path)async_fatal("bulk path");
    sprintf(path,"%s/bulk-%llu.%s",e->directory,(unsigned long long)e->file_seq++,bulk_compress==2?"ntp":bulk_compress?"ntz":"bin");
    e->fp=fopen(path,"wb");free(path);if(!e->fp)async_fatal("bulk file open");
    e->stdio_buffer=malloc(1048576);if(!e->stdio_buffer || setvbuf(e->fp,e->stdio_buffer,_IOFBF,1048576))async_fatal("bulk stdio");
    if(bulk_compress)bulk_put(e,bulk_compress==2?"NTPACK2\n":"NTZSTD1\n",8);
}
static void bulk_output_frame(bulk_engine*e,bulk_host*h,bulk_frame*f){
    trace_header_t th={f->dims[0],f->dims[1],f->dims[2],f->dims[3],f->dims[4],f->dims[5],f->dims[6],1};
    trace_section_t ts={16,32,sizeof(th)+sizeof(ts)};
    void *raw=(char*)h->host+f->offset;size_t packed=0;uint64_t tick=0;
    if(bulk_compress){
        tick=ab_stats?async_clock(CLOCK_MONOTONIC):0;
        size_t cap=bulk_compress==2?f->bytes:e->zbound(f->bytes);
        if(cap>e->codec_capacity){void*p=realloc(e->encoded,cap);if(!p)async_fatal("codec allocation");e->encoded=p;e->codec_capacity=cap;}
        if(bulk_compress==2){
            packed=nt_pack_records(e->encoded,raw,f->bytes,&e->codec_zero_records,&e->codec_global_wide_frames);
        }else{packed=e->zcompress(e->encoded,e->codec_capacity,raw,f->bytes,1);if(e->zerror(packed))async_fatal("ZSTD compress");}
        if(!packed || packed>=f->bytes){packed=0;e->codec_raw_frames++;}
        if(ab_stats)e->encode_ns+=async_clock(CLOCK_MONOTONIC)-tick;
        if(bulk_verify_codec && packed){
            tick=ab_stats?async_clock(CLOCK_MONOTONIC):0;
            void*p=realloc(e->decoded,f->bytes);if(!p)async_fatal("codec verification allocation");e->decoded=p;
            size_t n=f->bytes;
            if(bulk_compress==2){
                if(nt_unpack_records(p,n,e->encoded,packed))async_fatal("packed decode");
            }else{n=e->zdecompress(p,f->bytes,e->encoded,packed);if(e->zerror(n))async_fatal("ZSTD decode");}
            if(n!=f->bytes||memcmp(p,raw,n))async_fatal("lossless codec mismatch");
            e->codec_verified_frames++;if(ab_stats)e->verify_ns+=async_clock(CLOCK_MONOTONIC)-tick;
        }
    }
    tick=ab_stats?async_clock(CLOCK_MONOTONIC):0;
    bulk_output_open(e);bulk_put(e,&th,sizeof(th));bulk_put(e,&ts,sizeof(ts));
    if(bulk_compress){uint64_t n=packed;bulk_put(e,&n,8);}
    bulk_put(e,packed?e->encoded:raw,packed?packed:f->bytes);
    if(ab_stats)e->write_ns+=async_clock(CLOCK_MONOTONIC)-tick;
    e->calls++;e->payload+=f->bytes;e->logical_trace+=f->bytes+48;
}
static void *bulk_complete_worker(void*arg){
    bulk_engine*e=arg;CUDA_CHECK(real_cuCtxSetCurrent(e->context));
    for(;;){
        pthread_mutex_lock(&e->mutex);
        while(e->copy_completed==e->published && !e->stop)pthread_cond_wait(&e->changed,&e->mutex);
        if(e->stop && e->copy_completed==e->published){pthread_mutex_unlock(&e->mutex);break;}
        bulk_host*h=e->queue[e->copy_completed%e->queue_cap];pthread_mutex_unlock(&e->mutex);
        uint64_t tick=ab_stats?async_clock(CLOCK_MONOTONIC):0;
        CUDA_CHECK(real_cuEventSynchronize(h->done));
        uint64_t wait=ab_stats?async_clock(CLOCK_MONOTONIC)-tick:0;
        pthread_mutex_lock(&e->mutex);e->event_wait_ns+=wait;
        e->device[h->kind][h->device_index].busy=0;e->early_releases++;h->state=3;e->copy_completed++;
        pthread_cond_broadcast(&e->changed);pthread_mutex_unlock(&e->mutex);
    }
    CUDA_CHECK(real_cuCtxSetCurrent(NULL));return NULL;
}
static void *bulk_write_worker(void*arg){
    bulk_engine*e=arg;CUDA_CHECK(real_cuCtxSetCurrent(e->context));
    for(;;){
        pthread_mutex_lock(&e->mutex);
        while(e->completed==(bulk_decouple?e->copy_completed:e->published) && !e->stop)pthread_cond_wait(&e->changed,&e->mutex);
        if(e->stop && e->completed==e->published){pthread_mutex_unlock(&e->mutex);break;}
        bulk_host*h=e->queue[e->completed%e->queue_cap];pthread_mutex_unlock(&e->mutex);
        uint64_t cpu=ab_stats?async_clock(CLOCK_THREAD_CPUTIME_ID):0;
        if(!bulk_decouple){
            uint64_t tick=ab_stats?async_clock(CLOCK_MONOTONIC):0;CUDA_CHECK(real_cuEventSynchronize(h->done));
            if(ab_stats)e->event_wait_ns+=async_clock(CLOCK_MONOTONIC)-tick;
        }
        for(unsigned i=0;i<h->frames;i++)bulk_output_frame(e,h,&h->frame[i]);
        uint64_t elapsed=ab_stats?async_clock(CLOCK_THREAD_CPUTIME_ID)-cpu:0;
        pthread_mutex_lock(&e->mutex);e->worker_cpu_ns+=elapsed;
        if(!bulk_decouple)e->device[h->kind][h->device_index].busy=0;
        h->state=0;e->completed++;pthread_cond_broadcast(&e->changed);pthread_mutex_unlock(&e->mutex);
    }
    CUDA_CHECK(real_cuCtxSetCurrent(NULL));return NULL;
}
static bulk_engine *bulk_get(void){
    CUcontext ctx;CUDA_CHECK(real_cuCtxGetCurrent(&ctx));
    for(bulk_engine*e=bulk_engines;e;e=e->next)if(e->context==ctx)return e;
    bulk_engine*e=calloc(1,sizeof(*e));if(!e)async_fatal("bulk engine");
    e->context=ctx;e->directory=strdup(RESULT_DIR);e->nh[0]=bulk_small_hosts;e->nh[1]=bulk_large_hosts;
    e->nd[0]=bulk_small_devices;e->nd[1]=bulk_large_devices;e->queue_cap=e->nh[0]+e->nh[1];
    e->queue=calloc(e->queue_cap,sizeof(*e->queue));pthread_mutex_init(&e->mutex,NULL);pthread_cond_init(&e->changed,NULL);
    if(!e->directory||!e->queue)async_fatal("bulk metadata allocation");
    CUDA_CHECK(real_cuStreamCreate(&e->copy,CU_STREAM_NON_BLOCKING));
    for(int k=0;k<2;k++){
        e->host[k]=calloc(e->nh[k],sizeof(bulk_host));e->device[k]=calloc(e->nd[k],sizeof(bulk_device));
        if(!e->host[k]||!e->device[k])async_fatal("bulk pool allocation");
        for(int i=0;i<e->nh[k];i++){
            bulk_host*h=&e->host[k][i];h->kind=k;CUDA_CHECK(real_cuMemAllocHost_v2(&h->host,bulk_capacity(k)));
            CUDA_CHECK(real_cuEventCreate(&h->done,CU_EVENT_DISABLE_TIMING));CUDA_CHECK(real_cuEventCreate(&h->kernel_done,CU_EVENT_DISABLE_TIMING));
        }
        for(int i=0;i<e->nd[k];i++)CUDA_CHECK(real_cuMemAlloc_v2(&e->device[k][i].ptr,bulk_capacity(k)));
    }
    if(bulk_compress==1){
        e->zlib=dlopen("libzstd.so.1",RTLD_NOW|RTLD_LOCAL);if(!e->zlib)async_fatal("libzstd unavailable");
        e->zbound=dlsym(e->zlib,"ZSTD_compressBound");e->zcompress=dlsym(e->zlib,"ZSTD_compress");
        e->zdecompress=dlsym(e->zlib,"ZSTD_decompress");e->zerror=dlsym(e->zlib,"ZSTD_isError");
        if(!e->zbound||!e->zcompress||!e->zdecompress||!e->zerror)async_fatal("ZSTD ABI");
    }
    e->next=bulk_engines;bulk_engines=e;
    if(bulk_decouple && pthread_create(&e->completer,NULL,bulk_complete_worker,e))async_fatal("completion thread");
    if(pthread_create(&e->writer,NULL,bulk_write_worker,e))async_fatal("bulk writer thread");return e;
}
static void bulk_flush_pending(bulk_engine*e){
    bulk_host*h=e->pending;if(!h)return;
    uint64_t tick=ab_stats?async_clock(CLOCK_MONOTONIC):0;
    CUDA_CHECK(real_cuEventRecord(h->kernel_done,h->producer));CUDA_CHECK(real_cuStreamWaitEvent(e->copy,h->kernel_done,0));
    CUDA_CHECK(real_cuMemcpyDtoHAsync_v2(h->host,e->device[h->kind][h->device_index].ptr,h->bytes,e->copy));
    CUDA_CHECK(real_cuEventRecord(h->done,e->copy));
    pthread_mutex_lock(&e->mutex);
    if(e->published-e->completed>=e->queue_cap)async_fatal("bulk queue overflow");
    e->queue[e->published%e->queue_cap]=h;h->state=2;e->published++;e->batches++;
    if(e->published-e->completed>e->peak_queue)e->peak_queue=e->published-e->completed;
    pthread_cond_broadcast(&e->changed);pthread_mutex_unlock(&e->mutex);
    e->pending=NULL;if(ab_stats)e->copy_submit_ns+=async_clock(CLOCK_MONOTONIC)-tick;
}
static bulk_host *bulk_reserve(bulk_engine*e,int k){
    pthread_mutex_lock(&e->mutex);int hi=-1,di=-1;uint64_t tick=ab_stats?async_clock(CLOCK_MONOTONIC):0;
    for(;;){for(int i=0;i<e->nh[k];i++)if(!e->host[k][i].state){hi=i;break;}if(hi>=0)break;pthread_cond_wait(&e->changed,&e->mutex);}
    if(ab_stats)e->host_wait_ns+=async_clock(CLOCK_MONOTONIC)-tick;
    tick=ab_stats?async_clock(CLOCK_MONOTONIC):0;
    for(;;){for(int i=0;i<e->nd[k];i++)if(!e->device[k][i].busy){di=i;break;}if(di>=0)break;pthread_cond_wait(&e->changed,&e->mutex);}
    if(ab_stats)e->device_wait_ns+=async_clock(CLOCK_MONOTONIC)-tick;
    bulk_host*h=&e->host[k][hi];h->state=1;h->device_index=di;e->device[k][di].busy=1;
    h->frames=0;h->bytes=0;pthread_mutex_unlock(&e->mutex);return h;
}
static CUdeviceptr bulk_prepare(bulk_engine*e,size_t n,CUstream stream,unsigned*dims,const char*name){
    int k=n>bulk_small_bytes;if(n>bulk_capacity(k))async_fatal("Map exceeds configured arena capacity");
    bulk_host*h=e->pending;
    if(h && (h->kind!=k || h->producer!=stream || h->frames>=(unsigned)bulk_batch_frames ||
             n>bulk_capacity(k)-h->bytes || n>bulk_map_cutoff || h->frame[0].bytes>bulk_map_cutoff)){
        bulk_flush_pending(e);h=NULL;
    }
    if(!h){h=bulk_reserve(e,k);h->producer=stream;e->pending=h;}
    bulk_frame*f=&h->frame[h->frames++];memcpy(f->dims,dims,7*sizeof(unsigned));f->offset=h->bytes;f->bytes=n;h->bytes+=n;
    if(ab_stats){
        if(bulk_token<0||bulk_token>=BULK_MAX_TOKENS)async_fatal("token metadata limit");
        bulk_phase*p=&e->phase[bulk_token];p->calls++;p->bytes+=n;
        uint64_t hash=p->signature?p->signature:1469598103934665603ULL;
        for(const unsigned char*s=(const unsigned char*)name;*s;s++){hash^=*s;hash*=1099511628211ULL;}
        for(int j=0;j<7;j++){hash^=dims[j];hash*=1099511628211ULL;}p->signature=hash;
    }
    return e->device[k][h->device_index].ptr+f->offset;
}
static void **bulk_args(bulk_engine*e,size_t count){
    if(!bulk_hot_cache){void**p=malloc(count*sizeof(void*));if(!p)async_fatal("bulk args");return p;}
    if(e->args_capacity<count){void**p=realloc(e->args,count*sizeof(void*));if(!p)async_fatal("cached args");e->args=p;e->args_capacity=count;}
    return e->args;
}
static void bulk_attribute(bulk_engine*e,CUfunction f,unsigned value){
    if(bulk_hot_cache){
        int i;for(i=0;i<e->attr_count;i++)if(e->attr_func[i]==f)break;
        if(i<e->attr_count && e->attr_value[i]==value){e->attr_hits++;return;}
        if(i==64)async_fatal("attribute cache capacity");
        e->attr_func[i]=f;e->attr_value[i]=value;if(i==e->attr_count)e->attr_count++;
    }
    CUDA_CHECK(real_cuFuncSetAttribute(f,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,value));e->attr_calls++;
}
static void bulk_commit(bulk_engine*e){
    bulk_host*h=e->pending;
    if(h && (h->frames>=(unsigned)bulk_batch_frames || h->frame[0].bytes>bulk_map_cutoff))bulk_flush_pending(e);
}
int neutrino_capture_mark(int token){
    if(ab_arena)ar_mark(token);
    cd_token=token;
    if(!ab_bulk)return 0;
    for(bulk_engine*e=bulk_engines;e;e=e->next)if(bulk_step_flush)bulk_flush_pending(e);
    bulk_token=token;return 0;
}
static int bulk_boundary(void){
    for(bulk_engine*e=bulk_engines;e;e=e->next){
        uint64_t tick=ab_stats?async_clock(CLOCK_MONOTONIC):0;bulk_flush_pending(e);
        pthread_mutex_lock(&e->mutex);while(e->completed!=e->published)pthread_cond_wait(&e->changed,&e->mutex);
        int bad=0;if(e->fp){bad=ferror(e->fp);if(fclose(e->fp))bad=1;e->fp=NULL;free(e->stdio_buffer);e->stdio_buffer=NULL;}
        if(bulk_sink && e->ram_used){e->ram_last_size=e->ram_used;e->ram_used=0;}
        pthread_mutex_unlock(&e->mutex);if(ab_stats)e->drain_ns+=async_clock(CLOCK_MONOTONIC)-tick;if(bad)return -1;
    }return 0;
}
static void bulk_reset(void){
    for(bulk_engine*e=bulk_engines;e;e=e->next){
        e->calls=e->payload=e->logical_trace=e->storage_bytes=e->batches=e->peak_queue=0;
        e->host_wait_ns=e->device_wait_ns=e->copy_submit_ns=e->drain_ns=e->event_wait_ns=e->write_ns=e->encode_ns=e->verify_ns=e->worker_cpu_ns=0;
        e->early_releases=e->codec_verified_frames=e->codec_raw_frames=0;memset(e->phase,0,sizeof(e->phase));
        e->attr_calls=e->attr_hits=e->codec_zero_records=e->codec_global_wide_frames=0;
    }
}
static void bulk_json(FILE*fp){
    fputs(",\"bulk_engines\":[",fp);int comma=0;
    for(bulk_engine*e=bulk_engines;e;e=e->next){
        fprintf(fp,"%s{\"host_capacity_bytes\":%zu,\"device_capacity_bytes\":%zu,\"frames\":%llu,\"batches\":%llu,\"payload_bytes\":%llu,\"logical_trace_bytes\":%llu,\"storage_bytes\":%llu,\"peak_queue\":%llu,\"host_reserve_ns\":%llu,\"device_reserve_ns\":%llu,\"copy_submit_ns\":%llu,\"request_drain_ns\":%llu,\"worker_event_wall_ns\":%llu,\"worker_write_wall_ns\":%llu,\"worker_encode_wall_ns\":%llu,\"codec_verify_ns\":%llu,\"worker_cpu_ns\":%llu,\"early_releases\":%llu,\"codec_verified_frames\":%llu,\"codec_raw_frames\":%llu,\"phases\":[",comma?",":"",
            e->nh[0]*bulk_small_bytes+e->nh[1]*bulk_large_bytes,e->nd[0]*bulk_small_bytes+e->nd[1]*bulk_large_bytes,
            (unsigned long long)e->calls,(unsigned long long)e->batches,(unsigned long long)e->payload,(unsigned long long)e->logical_trace,(unsigned long long)e->storage_bytes,
            (unsigned long long)e->peak_queue,(unsigned long long)e->host_wait_ns,(unsigned long long)e->device_wait_ns,(unsigned long long)e->copy_submit_ns,
            (unsigned long long)e->drain_ns,(unsigned long long)e->event_wait_ns,(unsigned long long)e->write_ns,(unsigned long long)e->encode_ns,(unsigned long long)e->verify_ns,
            (unsigned long long)e->worker_cpu_ns,(unsigned long long)e->early_releases,(unsigned long long)e->codec_verified_frames,(unsigned long long)e->codec_raw_frames);
        int sep=0;for(int t=0;t<BULK_MAX_TOKENS;t++)if(e->phase[t].calls){bulk_phase*p=&e->phase[t];
            fprintf(fp,"%s{\"token\":%d,\"calls\":%llu,\"bytes\":%llu,\"signature\":\"%016llx\"}",sep?",":"",t,(unsigned long long)p->calls,(unsigned long long)p->bytes,(unsigned long long)p->signature);sep=1;}
        fprintf(fp,"],\"attribute_calls\":%llu,\"attribute_cache_hits\":%llu,\"codec_zero_records\":%llu,\"codec_global_wide_frames\":%llu}",(unsigned long long)e->attr_calls,(unsigned long long)e->attr_hits,(unsigned long long)e->codec_zero_records,(unsigned long long)e->codec_global_wide_frames);comma=1;
    }fputc(']',fp);
}
int neutrino_capture_summary(const char*path){
    FILE*fp=fopen(path,"w");if(!fp)return -1;
    fputs("{\"schema_version\":1",fp);if(ab_bulk)bulk_json(fp);if(ab_async)async_json(fp);if(ab_arena)ar_json(fp);fputs("}\n",fp);
    int bad=ferror(fp);if(fclose(fp))bad=1;return bad?-1:0;
}
static void bulk_release(void){
    CUcontext saved;CUDA_CHECK(real_cuCtxGetCurrent(&saved));
    while(bulk_engines){bulk_engine*e=bulk_engines;pthread_mutex_lock(&e->mutex);e->stop=1;pthread_cond_broadcast(&e->changed);pthread_mutex_unlock(&e->mutex);
        if(bulk_decouple)pthread_join(e->completer,NULL);pthread_join(e->writer,NULL);CUDA_CHECK(real_cuCtxSetCurrent(e->context));
        if(bulk_sink && e->ram_last_size){char*path=malloc(strlen(e->directory)+32);sprintf(path,"%s/ram-last.bin",e->directory);FILE*fp=fopen(path,"wb");free(path);
            if(!fp || fwrite(e->ram,1,e->ram_last_size,fp)!=e->ram_last_size || fclose(fp))async_fatal("RAM validation dump");}
        for(int k=0;k<2;k++){for(int i=0;i<e->nh[k];i++){bulk_host*h=&e->host[k][i];CUDA_CHECK(real_cuMemFreeHost(h->host));CUDA_CHECK(real_cuEventDestroy_v2(h->done));CUDA_CHECK(real_cuEventDestroy_v2(h->kernel_done));}
            for(int i=0;i<e->nd[k];i++)CUDA_CHECK(real_cuMemFree_v2(e->device[k][i].ptr));free(e->host[k]);free(e->device[k]);}
        CUDA_CHECK(real_cuStreamDestroy_v2(e->copy));if(e->zlib)dlclose(e->zlib);
        free(e->args);free(e->encoded);free(e->decoded);free(e->ram);free(e->queue);free(e->directory);pthread_mutex_destroy(&e->mutex);pthread_cond_destroy(&e->changed);
        bulk_engines=e->next;free(e);
    }CUDA_CHECK(real_cuCtxSetCurrent(saved));
}
#endif
