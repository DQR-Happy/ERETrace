/* Opt-in diagnostics for the fixed async ring. No CUDA synchronization is added.
 * Producer and consumer own disjoint metric arrays. Read/reset only after drain.
 * Occupancy transitions run under the existing ring mutex. Histograms are
 * powers-of-two nanosecond buckets, NOT exact quantiles.
 */
#ifndef NEUTRINO_CAPTURE_DETAIL_H
#define NEUTRINO_CAPTURE_DETAIL_H
#include <stdint.h>
#include <inttypes.h>
#include <stdatomic.h>
#include <sys/syscall.h>
#include <unistd.h>

#define CD_METRICS(X) \
 X(ENGINE_GET,"producer_context_and_engine") \
 X(RESERVE_LOCK,"producer_reserve_mutex") \
 X(RESERVE_WAIT,"producer_slot_condition_wait") \
 X(RESERVE_STATE,"producer_reserve_state") \
 X(GROW,"slot_buffer_and_event_setup") \
 X(CLEAR,"producer_map_clear_enqueue") \
 X(ARGS,"producer_args_alloc_copy") \
 X(ATTRIBUTE,"producer_attribute_api") \
 X(MODEL,"producer_model_launch_api") \
 X(KERNEL_EVENT,"producer_kernel_done_record") \
 X(STREAM_WAIT,"copy_stream_wait_enqueue") \
 X(ENCODER_PREP,"encoder_lookup_and_scratch") \
 X(ENCODER,"encoder_launch_api") \
 X(COPY,"trace_d2h_enqueue_api") \
 X(DONE_EVENT,"copy_done_record") \
 X(PUBLISH_LOCK,"producer_publish_mutex") \
 X(PUBLISH_STATE,"producer_publish_state_signal") \
 X(ARGS_FREE,"producer_args_free") \
 X(DEQUEUE_LOCK,"writer_dequeue_mutex") \
 X(READY_WAIT,"writer_ready_condition_wait_completed") \
 X(EVENT_WAIT,"writer_copy_done_wait") \
 X(OPEN,"writer_path_open_buffer_magic") \
 X(HEADER,"writer_header_section_length") \
 X(PAYLOAD,"writer_payload_fwrite_or_memcpy") \
 X(RELEASE_LOCK,"writer_release_mutex") \
 X(RELEASE_STATE,"writer_release_state_broadcast") \
 X(DRAIN_LOCK,"request_boundary_mutex") \
 X(DRAIN_WAIT,"request_boundary_condition_wait") \
 X(CLOSE,"request_fclose_and_buffer_free") \
 X(ERROR_COPY,"request_encoder_error_d2h") \
 X(QUEUE_RESIDENCE,"interval_publish_to_writer") \
 X(SLOT_RESIDENCE,"interval_reserve_to_release")
#define CD_ENUM(id,name) CD_##id,
enum { CD_METRICS(CD_ENUM) CD_COUNT };
#undef CD_ENUM
#define CD_NAME(id,name) name,
static const char *cd_names[]={CD_METRICS(CD_NAME)};
#undef CD_NAME
enum { CD_GROUPS=128, CD_HIST=40 };
typedef struct {uint64_t count,ns,cpu_ns,min,max,hist[CD_HIST];} cd_metric;
typedef struct {
    char name[512];unsigned dims[7];int phase;
    uint64_t producer_calls,worker_calls,logical_bytes,copy_bytes,storage_bytes;
    cd_metric producer[CD_COUNT],worker[CD_COUNT];
} cd_group;
typedef struct {
    int incoming,slot,head,slot_state,head_state;
    uint64_t requested_bytes,slot_bytes,head_bytes,count,wall_ns,max_ns;
} cd_stall;
typedef struct {
    atomic_int active;unsigned ng;
    cd_group groups[CD_GROUPS];
    cd_metric general_producer[CD_COUNT],general_worker[CD_COUNT];
    uint64_t begin,end,last_transition,busy,pending,writer,busy_bytes;
    uint64_t peak_busy,peak_pending,peak_busy_bytes,occupancy_ns[65];
    uint64_t busy_byte_ns,transitions,producer_tid,worker_tid;
    cd_stall stalls[256];unsigned nstalls;
} cd_state;
static int cd_enabled,cd_cpu,cd_consume,cd_no_d2h,cd_no_pack,cd_kernel=2;
static __thread cd_state *cd_current;
static __thread int cd_group_index=-1,cd_worker;
static __thread int cd_token;
typedef struct {uint64_t wall,cpu;} cd_tick;
static uint64_t cd_clock(clockid_t which){struct timespec t;clock_gettime(which,&t);return(uint64_t)t.tv_sec*1000000000+t.tv_nsec;}
static int cd_active(void){return cd_enabled && cd_current && atomic_load_explicit(&cd_current->active,memory_order_relaxed);}
static cd_tick cd_begin(void){cd_tick t={0,0};if(cd_active()){t.wall=cd_clock(CLOCK_MONOTONIC);if(cd_cpu)t.cpu=cd_clock(CLOCK_THREAD_CPUTIME_ID);}return t;}
static void cd_add(cd_metric*m,uint64_t ns,uint64_t cpu){
    if(!m->count||ns<m->min)m->min=ns;if(ns>m->max)m->max=ns;
    m->count++;m->ns+=ns;m->cpu_ns+=cpu;
    unsigned b=ns?63-__builtin_clzll(ns):0;if(b>=CD_HIST)b=CD_HIST-1;m->hist[b]++;
}
static void cd_record(int id,cd_tick t){
    if(!t.wall||!cd_active())return;
    uint64_t cpu=cd_cpu?cd_clock(CLOCK_THREAD_CPUTIME_ID)-t.cpu:0;
    uint64_t wall=cd_clock(CLOCK_MONOTONIC)-t.wall;
    cd_metric*m=cd_group_index<0?(cd_worker?cd_current->general_worker:cd_current->general_producer):
        (cd_worker?cd_current->groups[cd_group_index].worker:cd_current->groups[cd_group_index].producer);
    cd_add(&m[id],wall,cpu);
}
#define CD_RUN(id,statement) do {cd_tick cd_t_=cd_begin();statement;cd_record(CD_##id,cd_t_);} while(0)
static void cd_interval(cd_state*d,int group,int id,uint64_t start,uint64_t end){
    if(!cd_enabled||!atomic_load_explicit(&d->active,memory_order_relaxed)||!start)return;
    if(end<start){fprintf(stderr,"capture detail reversed interval\n");abort();}
    cd_add(&d->groups[group].worker[id],end-start,0);
}
static void cd_bind(cd_state*d,int group,int worker){cd_current=d;cd_group_index=group;cd_worker=worker;}
static int cd_find(cd_state*d,const char*name,unsigned*dims){
    int phase=cd_token?1:0;
    for(unsigned i=0;i<d->ng;i++)if(d->groups[i].phase==phase&&!strcmp(d->groups[i].name,name)&&!memcmp(d->groups[i].dims,dims,7*sizeof(unsigned)))return i;
    if(d->ng==CD_GROUPS){fprintf(stderr,"capture detail group overflow\n");abort();}
    unsigned i=d->ng++;cd_group*g=&d->groups[i];snprintf(g->name,sizeof(g->name),"%s",name);memcpy(g->dims,dims,7*sizeof(unsigned));g->phase=phase;return i;
}
/* Before changing occupancy, charge the old state. Caller holds ring mutex. */
static void cd_transition(cd_state*d,uint64_t now){
    if(!cd_enabled||!d||!atomic_load_explicit(&d->active,memory_order_relaxed))return;
    if(d->busy>64||now<d->last_transition){fprintf(stderr,"capture detail occupancy invariant\n");abort();}
    uint64_t dt=now-d->last_transition;d->occupancy_ns[d->busy]+=dt;d->busy_byte_ns+=d->busy_bytes*dt;
    d->last_transition=now;d->transitions++;
}
static void cd_peak(cd_state*d){
    if(d->busy>d->peak_busy)d->peak_busy=d->busy;
    if(d->pending>d->peak_pending)d->peak_pending=d->pending;
    if(d->busy_bytes>d->peak_busy_bytes)d->peak_busy_bytes=d->busy_bytes;
}
/* Snapshot under ring mutex before cond_wait. It labels the owner at entry,
 * not a claim that this owner accounts for every nanosecond until wakeup. */
static void cd_stall_add(cd_state*d,cd_stall*s,uint64_t ns){
    if(!d||!atomic_load(&d->active))return;
    unsigned i;for(i=0;i<d->nstalls;i++){
        cd_stall*t=&d->stalls[i];
        if(t->incoming==s->incoming&&t->slot==s->slot&&t->head==s->head&&
           t->slot_state==s->slot_state&&t->head_state==s->head_state&&
           t->requested_bytes==s->requested_bytes&&t->slot_bytes==s->slot_bytes&&t->head_bytes==s->head_bytes)break;
    }
    if(i==d->nstalls){if(i==256){fprintf(stderr,"capture stall group overflow\n");abort();}d->stalls[i]=*s;d->nstalls++;}
    cd_stall*t=&d->stalls[i];t->count++;t->wall_ns+=ns;if(ns>t->max_ns)t->max_ns=ns;
}
static void cd_start(cd_state*d){
    for(unsigned i=0;i<d->ng;i++){cd_group*g=&d->groups[i];g->producer_calls=g->worker_calls=g->logical_bytes=g->copy_bytes=g->storage_bytes=0;memset(g->producer,0,sizeof(g->producer));memset(g->worker,0,sizeof(g->worker));}
    memset(d->general_producer,0,sizeof(d->general_producer));memset(d->general_worker,0,sizeof(d->general_worker));
    memset(d->occupancy_ns,0,sizeof(d->occupancy_ns));
    memset(d->stalls,0,sizeof(d->stalls));d->nstalls=0;
    d->busy=d->pending=d->writer=d->busy_bytes=d->peak_busy=d->peak_pending=d->peak_busy_bytes=d->busy_byte_ns=d->transitions=0;
    d->begin=d->last_transition=cd_clock(CLOCK_MONOTONIC);d->end=0;d->producer_tid=syscall(SYS_gettid);atomic_store(&d->active,1);
}
static void cd_json_metrics(FILE*fp,cd_metric*m){
    fputc('{',fp);int sep=0;for(unsigned i=0;i<CD_COUNT;i++)if(m[i].count){cd_metric*v=&m[i];
        fprintf(fp,"%s\"%s\":{\"count\":%"PRIu64",\"wall_ns\":%"PRIu64",\"thread_cpu_ns\":%"PRIu64",\"min_ns\":%"PRIu64",\"max_ns\":%"PRIu64",\"log2_ns_hist\":[",sep?",":"",cd_names[i],v->count,v->ns,v->cpu_ns,v->min,v->max);
        for(unsigned b=0;b<CD_HIST;b++)fprintf(fp,"%s%"PRIu64,b?",":"",v->hist[b]);fputs("]}",fp);sep=1;
    }fputc('}',fp);
}
static void cd_json(FILE*fp,cd_state*d){
    fprintf(fp,"{\"schema_version\":1,\"clock\":\"CLOCK_MONOTONIC\",\"cpu_clock_enabled\":%s,\"window_ns\":%"PRIu64",\"producer_tid\":%"PRIu64",\"worker_tid\":%"PRIu64",\"instrument_state_bytes\":%zu,\"peak_busy_slots\":%"PRIu64",\"peak_pending_slots\":%"PRIu64",\"peak_live_payload_bytes\":%"PRIu64",\"live_payload_byte_ns\":%"PRIu64",\"transitions\":%"PRIu64",\"final_busy\":%"PRIu64",\"final_pending\":%"PRIu64",\"final_live_bytes\":%"PRIu64",\"occupancy_ns\":[",cd_cpu?"true":"false",d->end-d->begin,d->producer_tid,d->worker_tid,sizeof(*d),d->peak_busy,d->peak_pending,d->peak_busy_bytes,d->busy_byte_ns,d->transitions,d->busy,d->pending,d->busy_bytes);
    for(unsigned i=0;i<65;i++)fprintf(fp,"%s%"PRIu64,i?",":"",d->occupancy_ns[i]);
    fputs("],\"general_producer\":",fp);cd_json_metrics(fp,d->general_producer);fputs(",\"general_worker\":",fp);cd_json_metrics(fp,d->general_worker);fputs(",\"groups\":[",fp);
    int sep=0;for(unsigned i=0;i<d->ng;i++){cd_group*g=&d->groups[i];if(!g->producer_calls&&!g->worker_calls)continue;
        fprintf(fp,"%s{\"group_id\":%u,\"kernel\":",sep?",":"",i);nt_json_string(fp,g->name);fprintf(fp,",\"phase\":\"%s\",\"shape\":[",g->phase?"decode":"prefill");
        for(unsigned j=0;j<7;j++)fprintf(fp,"%s%u",j?",":"",g->dims[j]);
        fprintf(fp,"],\"producer_calls\":%"PRIu64",\"worker_calls\":%"PRIu64",\"logical_bytes\":%"PRIu64",\"copy_bytes\":%"PRIu64",\"storage_bytes\":%"PRIu64",\"producer\":",g->producer_calls,g->worker_calls,g->logical_bytes,g->copy_bytes,g->storage_bytes);
        cd_json_metrics(fp,g->producer);fputs(",\"worker\":",fp);cd_json_metrics(fp,g->worker);fputc('}',fp);sep=1;
    }fputs("],\"condition_wait_entry_owners\":[",fp);
    for(unsigned i=0;i<d->nstalls;i++){cd_stall*s=&d->stalls[i];
        fprintf(fp,"%s{\"incoming_group\":%d,\"reuse_slot_group\":%d,\"fifo_head_group\":%d,\"reuse_slot_state\":%d,\"fifo_head_state\":%d,\"requested_bytes\":%"PRIu64",\"reuse_slot_bytes\":%"PRIu64",\"fifo_head_bytes\":%"PRIu64",\"waits\":%"PRIu64",\"wall_ns\":%"PRIu64",\"max_ns\":%"PRIu64"}",i?",":"",s->incoming,s->slot,s->head,s->slot_state,s->head_state,s->requested_bytes,s->slot_bytes,s->head_bytes,s->count,s->wall_ns,s->max_ns);
    }fputs("]}",fp);
}
#endif
