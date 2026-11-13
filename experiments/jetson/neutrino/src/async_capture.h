/* Explicit experimental fixed block_sched backend: bounded pinned/device ring,
 * stream-ordered clear/kernel/D2H, completion events, FIFO writer, request drain.
 * Every slot remains owned until its GPU copy AND CPU write have completed.
 */
#ifndef NEUTRINO_ASYNC_CAPTURE_H
#define NEUTRINO_ASYNC_CAPTURE_H
#include "capture_detail.h"
static int ab_async,ab_async_slots=8,ab_async_blocking,ab_async_copy_stream;
static int legacy_output_mode,legacy_verify_encoder;
typedef struct {
    void *host;CUdeviceptr device;size_t capacity,bytes;
    CUdeviceptr encoded;size_t encoded_capacity,copy_bytes;void*verify_host;size_t verify_capacity;int codec;
    CUevent done,kernel_done;unsigned dims[7];int state,detail_group;
    uint64_t reserved_at,published_at;
} async_slot;
typedef struct async_engine {
    CUcontext context;CUstream copy_stream;pthread_t worker;pthread_mutex_t mutex;
    pthread_cond_t ready,free_slot;
    async_slot *slots;unsigned capacity;
    uint64_t reserved,published,consumed,file_seq;
    int shutdown;FILE *fp;char *stdio_buffer;char *directory;
    uint64_t submitted,completed,payload,trace,peak_queue,backpressure_ns,drain_ns;
    uint64_t worker_cpu_ns,worker_event_wall_ns,worker_write_wall_ns;
    void *output_state;cd_state *detail;
    struct async_engine *next;
} async_engine;
static void legacy_write(async_engine*,async_slot*);
static void legacy_copy(async_engine*,async_slot*,CUstream);
static int legacy_boundary(async_engine*);
static void legacy_reset(async_engine*);
static void legacy_json(FILE*);
static void legacy_release(async_engine*);
static __thread async_engine *async_engines;
static uint64_t async_clock(clockid_t clock){struct timespec t;clock_gettime(clock,&t);return(uint64_t)t.tv_sec*1000000000+t.tv_nsec;}
static void async_fatal(const char *message){fprintf(stderr,"async capture: %s\n",message);exit(EXIT_FAILURE);}
static void *async_writer(void *arg){
    async_engine *e=arg;CUDA_CHECK(real_cuCtxSetCurrent(e->context));
    if(e->detail)e->detail->worker_tid=syscall(SYS_gettid);
    for(;;){
        cd_bind(e->detail,-1,1);
        CD_RUN(DEQUEUE_LOCK,pthread_mutex_lock(&e->mutex));
        while(e->consumed==e->published && !e->shutdown){CD_RUN(READY_WAIT,pthread_cond_wait(&e->ready,&e->mutex));}
        if(e->shutdown && e->consumed==e->published){pthread_mutex_unlock(&e->mutex);break;}
        async_slot*s=&e->slots[e->consumed%e->capacity];if(s->state!=2)async_fatal("FIFO state");s->state=3;
        cd_bind(e->detail,s->detail_group,1);
        if(cd_active()){
            uint64_t now=cd_clock(CLOCK_MONOTONIC);cd_transition(e->detail,now);e->detail->pending--;e->detail->writer++;
            cd_interval(e->detail,s->detail_group,CD_QUEUE_RESIDENCE,s->published_at,now);
        }
        pthread_mutex_unlock(&e->mutex);
        uint64_t cpu=ab_stats?async_clock(CLOCK_THREAD_CPUTIME_ID):0;
        uint64_t start=ab_stats?async_clock(CLOCK_MONOTONIC):0;
        CD_RUN(EVENT_WAIT,CUDA_CHECK(real_cuEventSynchronize(s->done)));
        uint64_t wait=ab_stats?async_clock(CLOCK_MONOTONIC)-start:0;
        start=ab_stats?async_clock(CLOCK_MONOTONIC):0;
        if(legacy_output_mode)legacy_write(e,s);
        else {
        if(!e->fp){
            char *path=malloc(strlen(e->directory)+48);if(!path)async_fatal("path allocation");
            sprintf(path,"%s/async-%llu.bin",e->directory,(unsigned long long)e->file_seq++);
            e->fp=fopen(path,"wb");free(path);if(!e->fp)async_fatal("file open");
            e->stdio_buffer=malloc(1048576);if(!e->stdio_buffer || setvbuf(e->fp,e->stdio_buffer,_IOFBF,1048576))async_fatal("stdio buffer");
        }
        trace_header_t h={s->dims[0],s->dims[1],s->dims[2],s->dims[3],s->dims[4],s->dims[5],s->dims[6],1};
        trace_section_t section={16,32,sizeof(h)+sizeof(trace_section_t)};
        if(fwrite(&h,sizeof(h),1,e->fp)!=1 || fwrite(&section,sizeof(section),1,e->fp)!=1 || fwrite(s->host,1,s->bytes,e->fp)!=s->bytes)async_fatal("file write");
        }
        uint64_t write=ab_stats?async_clock(CLOCK_MONOTONIC)-start:0;
        uint64_t elapsed_cpu=ab_stats?async_clock(CLOCK_THREAD_CPUTIME_ID)-cpu:0;
        CD_RUN(RELEASE_LOCK,pthread_mutex_lock(&e->mutex));
        cd_tick release_tick=cd_begin();
        e->completed++;e->payload+=s->bytes;e->trace+=s->bytes+sizeof(trace_header_t)+sizeof(trace_section_t);
        e->worker_cpu_ns+=elapsed_cpu;e->worker_event_wall_ns+=wait;e->worker_write_wall_ns+=write;
        if(cd_active()){
            uint64_t now=cd_clock(CLOCK_MONOTONIC);cd_transition(e->detail,now);
            e->detail->busy--;e->detail->writer--;e->detail->busy_bytes-=s->bytes;
            e->detail->groups[s->detail_group].worker_calls++;
            cd_interval(e->detail,s->detail_group,CD_SLOT_RESIDENCE,s->reserved_at,now);
        }
        s->state=0;e->consumed++;pthread_cond_broadcast(&e->free_slot);
        cd_record(CD_RELEASE_STATE,release_tick);pthread_mutex_unlock(&e->mutex);
    }
    CUDA_CHECK(real_cuCtxSetCurrent(NULL));return NULL;
}
static async_engine *async_get(void){
    CUcontext ctx;CUDA_CHECK(real_cuCtxGetCurrent(&ctx));
    for(async_engine*e=async_engines;e;e=e->next)if(e->context==ctx)return e;
    async_engine*e=calloc(1,sizeof(*e));if(!e)async_fatal("engine allocation");
    e->context=ctx;e->capacity=ab_async_slots;e->slots=calloc(e->capacity,sizeof(*e->slots));e->directory=strdup(RESULT_DIR);
    if(cd_enabled){e->detail=calloc(1,sizeof(cd_state));if(!e->detail)async_fatal("detail allocation");}
    if(ab_async_copy_stream)CUDA_CHECK(real_cuStreamCreate(&e->copy_stream,CU_STREAM_NON_BLOCKING));
    if(!e->slots || !e->directory)async_fatal("slot allocation");
    pthread_mutex_init(&e->mutex,NULL);pthread_cond_init(&e->ready,NULL);pthread_cond_init(&e->free_slot,NULL);
    e->next=async_engines;async_engines=e;
    if(pthread_create(&e->worker,NULL,async_writer,e))async_fatal("writer thread");return e;
}
static async_slot *async_reserve(async_engine*e,size_t n){
    CD_RUN(RESERVE_LOCK,pthread_mutex_lock(&e->mutex));async_slot*s=&e->slots[e->reserved%e->capacity];
    uint64_t wait=0;if(s->state && ab_stats)wait=async_clock(CLOCK_MONOTONIC);
    while(s->state){
        cd_tick stall_tick=cd_begin();cd_stall entry={0};
        if(stall_tick.wall){async_slot*head=&e->slots[e->consumed%e->capacity];
            entry=(cd_stall){.incoming=cd_group_index,.slot=s->detail_group,.head=head->detail_group,
                .slot_state=s->state,.head_state=head->state,.requested_bytes=n,.slot_bytes=s->bytes,.head_bytes=head->bytes};}
        pthread_cond_wait(&e->free_slot,&e->mutex);
        if(stall_tick.wall)cd_stall_add(e->detail,&entry,cd_clock(CLOCK_MONOTONIC)-stall_tick.wall);
        cd_record(CD_RESERVE_WAIT,stall_tick);
    }
    if(wait)e->backpressure_ns+=async_clock(CLOCK_MONOTONIC)-wait;
    cd_tick state_tick=cd_begin();
    s->state=1;e->reserved++;s->bytes=n;s->detail_group=cd_group_index;
    if(cd_active()){
        uint64_t now=cd_clock(CLOCK_MONOTONIC);cd_transition(e->detail,now);
        e->detail->busy++;e->detail->busy_bytes+=n;cd_peak(e->detail);s->reserved_at=now;
        cd_group*g=&e->detail->groups[s->detail_group];g->producer_calls++;g->logical_bytes+=n;
    }
    cd_record(CD_RESERVE_STATE,state_tick);pthread_mutex_unlock(&e->mutex);
    if(s->capacity<n){
        cd_tick grow_tick=cd_begin();
        if(s->host)CUDA_CHECK(real_cuMemFreeHost(s->host));
        if(s->device)CUDA_CHECK(real_cuMemFree_v2(s->device));
        CUDA_CHECK(real_cuMemAllocHost_v2(&s->host,n));CUDA_CHECK(real_cuMemAlloc_v2(&s->device,n));s->capacity=n;
        cd_record(CD_GROW,grow_tick);
    }
    if(!s->done)CUDA_CHECK(real_cuEventCreate(&s->done,CU_EVENT_DISABLE_TIMING|(ab_async_blocking?CU_EVENT_BLOCKING_SYNC:0)));
    if(ab_async_copy_stream && !s->kernel_done)CUDA_CHECK(real_cuEventCreate(&s->kernel_done,CU_EVENT_DISABLE_TIMING));
    s->bytes=n;return s;
}
static void async_publish(async_engine*e,async_slot*s,unsigned*dims){
    memcpy(s->dims,dims,7*sizeof(unsigned));CD_RUN(PUBLISH_LOCK,pthread_mutex_lock(&e->mutex));
    cd_tick publish_tick=cd_begin();
    s->state=2;e->published++;e->submitted++;
    uint64_t queued=e->published-e->consumed;if(queued>e->peak_queue)e->peak_queue=queued;
    if(cd_active()){
        uint64_t now=cd_clock(CLOCK_MONOTONIC);cd_transition(e->detail,now);e->detail->pending++;cd_peak(e->detail);s->published_at=now;
        e->detail->groups[s->detail_group].copy_bytes+=cd_no_d2h?0:s->copy_bytes;
    }
    pthread_cond_signal(&e->ready);cd_record(CD_PUBLISH_STATE,publish_tick);pthread_mutex_unlock(&e->mutex);
}
static int async_boundary(void){
    for(async_engine*e=async_engines;e;e=e->next){
        cd_bind(e->detail,-1,0);
        uint64_t start=ab_stats?async_clock(CLOCK_MONOTONIC):0;
        CD_RUN(DRAIN_LOCK,pthread_mutex_lock(&e->mutex));
        while(e->consumed!=e->published){CD_RUN(DRAIN_WAIT,pthread_cond_wait(&e->free_slot,&e->mutex));}
        int bad=0;if(e->fp){bad=ferror(e->fp);if(fclose(e->fp))bad=1;e->fp=NULL;free(e->stdio_buffer);e->stdio_buffer=NULL;}
        if(legacy_output_mode && legacy_boundary(e))bad=1;
        if(ab_stats)e->drain_ns+=async_clock(CLOCK_MONOTONIC)-start;
        pthread_mutex_unlock(&e->mutex);if(bad)return -1;
    }return 0;
}
static void async_reset(void){
    for(async_engine*e=async_engines;e;e=e->next){pthread_mutex_lock(&e->mutex);
        e->submitted=e->completed=e->payload=e->trace=e->peak_queue=e->backpressure_ns=e->drain_ns=0;
        e->worker_cpu_ns=e->worker_event_wall_ns=e->worker_write_wall_ns=0;pthread_mutex_unlock(&e->mutex);}
    if(legacy_output_mode)for(async_engine*e=async_engines;e;e=e->next)legacy_reset(e);
}
static void async_json(FILE*fp){
    fputs(",\"async_engines\":[",fp);int comma=0;
    for(async_engine*e=async_engines;e;e=e->next){pthread_mutex_lock(&e->mutex);size_t capacity=0;for(unsigned i=0;i<e->capacity;i++)capacity+=e->slots[i].capacity;
        fprintf(fp,"%s{\"slots\":%u,\"host_capacity_bytes\":%zu,\"device_capacity_bytes\":%zu,\"submitted\":%llu,\"completed\":%llu,\"payload_bytes\":%llu,\"trace_bytes\":%llu,\"peak_queue\":%llu,\"producer_backpressure_ns\":%llu,\"request_drain_ns\":%llu,\"worker_cpu_ns\":%llu,\"worker_event_wall_ns\":%llu,\"worker_write_wall_ns\":%llu,\"slot_capacity_bytes\":[",comma?",":"",e->capacity,capacity,capacity,(unsigned long long)e->submitted,(unsigned long long)e->completed,(unsigned long long)e->payload,(unsigned long long)e->trace,(unsigned long long)e->peak_queue,(unsigned long long)e->backpressure_ns,(unsigned long long)e->drain_ns,(unsigned long long)e->worker_cpu_ns,(unsigned long long)e->worker_event_wall_ns,(unsigned long long)e->worker_write_wall_ns);
        for(unsigned i=0;i<e->capacity;i++)fprintf(fp,"%s%zu",i?",":"",e->slots[i].capacity);fputs("]}",fp);comma=1;pthread_mutex_unlock(&e->mutex);}
    fputc(']',fp);
    if(legacy_output_mode)legacy_json(fp);
}
static void async_release(void){
    CUcontext saved;CUDA_CHECK(real_cuCtxGetCurrent(&saved));
    while(async_engines){async_engine*e=async_engines;
        pthread_mutex_lock(&e->mutex);e->shutdown=1;pthread_cond_signal(&e->ready);pthread_mutex_unlock(&e->mutex);pthread_join(e->worker,NULL);
        CUDA_CHECK(real_cuCtxSetCurrent(e->context));
        if(legacy_output_mode)legacy_release(e);
        for(unsigned i=0;i<e->capacity;i++){async_slot*s=&e->slots[i];if(s->host)CUDA_CHECK(real_cuMemFreeHost(s->host));if(s->device)CUDA_CHECK(real_cuMemFree_v2(s->device));if(s->verify_host)CUDA_CHECK(real_cuMemFreeHost(s->verify_host));if(s->done)CUDA_CHECK(real_cuEventDestroy_v2(s->done));if(s->kernel_done)CUDA_CHECK(real_cuEventDestroy_v2(s->kernel_done));}
        if(e->copy_stream)CUDA_CHECK(real_cuStreamDestroy_v2(e->copy_stream));
        async_engines=e->next;free(e->slots);free(e->directory);free(e->detail);pthread_mutex_destroy(&e->mutex);pthread_cond_destroy(&e->ready);pthread_cond_destroy(&e->free_slot);free(e);
    }cd_bind(NULL,-1,0);CUDA_CHECK(real_cuCtxSetCurrent(saved));
}
#endif
