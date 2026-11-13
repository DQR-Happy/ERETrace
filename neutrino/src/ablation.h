/* CUDA-only experimental controls. Disabled unless explicitly selected.
 * Pools are thread-local, keyed by context and stream. All reuse occurs after
 * the existing synchronous completion boundary; this is not async capture.
 */
#ifndef NEUTRINO_ABLATION_H
#define NEUTRINO_ABLATION_H
#include <inttypes.h>
enum { AB_OFF, AB_PASS, AB_FRAMEWORK, AB_BUFFERS, AB_PRUNED, AB_PROBE,
       AB_D2H, AB_FILES, AB_LOGGED };
static int ab_mode, ab_quiet, ab_stats, ab_host, ab_device_pool, ab_event_pool;
static int ab_blocking, ab_writer, ab_log_null;
static int ab_no_events, ab_one_wait;
static size_t ab_file_buffer;
static pid_t ab_owner;
static const char *ab_exclude;
static FILE *ab_original_log;
#define AB_LOG(...) do { if (!ab_quiet) fprintf(event_log, __VA_ARGS__); } while (0)
#define AB_FLUSH() do { if (!ab_quiet) fflush(event_log); } while (0)
typedef struct ab_slot {
    CUcontext context; CUstream stream;
    void *host; CUdeviceptr device; size_t host_cap, device_cap;
    CUevent start, end; struct ab_slot *next;
} ab_slot;
static __thread ab_slot *ab_slots;
static __thread FILE *ab_output;
static __thread char *ab_stdio_buffer;
static __thread void *ab_sink;
static __thread size_t ab_sink_cap;
static __thread uint64_t ab_file_seq;
enum { AB_GROUPS = 256 };
typedef struct {
    char name[512]; unsigned dims[7];
    uint64_t calls, original, pruned, probed, excluded, fallback, errors;
    uint64_t planned, d2h, trace;
} ab_group;
static ab_group ab_groups[AB_GROUPS];
static pthread_mutex_t ab_lock = PTHREAD_MUTEX_INITIALIZER;
static int ab_group_count;
static uint64_t ab_cold;
static int ab_arena;
static void ar_mark(int);
static void ar_json(FILE*);
#include "async_capture.h"
#include "bulk_capture.h"
#include "legacy_capture.h"
#include "arena_capture.h"

static int ab_value(const char *key) {
    const char *v = getenv(key); return v ? atoi(v) : 0;
}
static void ab_init(void) {
    ab_mode = ab_value("NEUTRINO_ABLATION_MODE");
    if (!ab_mode) return;
    if (ab_mode < AB_PASS || ab_mode > AB_LOGGED || DYNAMIC ||
        NEUTRINO_BENCHMARK || NEUTRINO_MEMUSAGE ||
        (NEUTRINO_CALLBACK && strcmp(NEUTRINO_CALLBACK, "/dev/null"))) {
        fprintf(stderr, "Ablation requires fixed Map, no callback/benchmark/memusage\n");
        exit(EXIT_FAILURE);
    }
    ab_owner = getpid();
    ab_quiet = ab_mode != AB_LOGGED;
    ab_stats = ab_value("NEUTRINO_ABLATION_STATS");
    ab_host = ab_value("NEUTRINO_ABLATION_HOST"); /* 0 fresh, 1 pool, 2 pinned pool */
    ab_device_pool = ab_value("NEUTRINO_ABLATION_DEVICE_POOL");
    ab_event_pool = ab_value("NEUTRINO_ABLATION_EVENT_POOL");
    ab_blocking = ab_value("NEUTRINO_ABLATION_BLOCKING");
    ab_writer = ab_value("NEUTRINO_ABLATION_WRITER"); /* 0 files, 1 request file, 2 memory sink */
    ab_file_buffer = (size_t)ab_value("NEUTRINO_ABLATION_FILE_BUFFER");
    ab_exclude = getenv("NEUTRINO_ABLATION_EXCLUDE");
    ab_log_null = ab_value("NEUTRINO_ABLATION_LOG_NULL");
    ab_no_events = ab_value("NEUTRINO_ABLATION_NO_EVENTS");
    ab_one_wait = ab_value("NEUTRINO_ABLATION_ONE_WAIT");
    ab_async = ab_value("NEUTRINO_ABLATION_ASYNC");
    ab_arena = ab_value("NEUTRINO_ARENA");
    if(ab_value("NEUTRINO_ABLATION_ASYNC_SLOTS"))ab_async_slots=ab_value("NEUTRINO_ABLATION_ASYNC_SLOTS");
    ab_async_blocking=ab_value("NEUTRINO_ABLATION_ASYNC_BLOCKING");
    ab_async_copy_stream=ab_value("NEUTRINO_ABLATION_ASYNC_COPY_STREAM");
    cd_enabled=ab_value("NEUTRINO_CAPTURE_DETAIL");cd_cpu=ab_value("NEUTRINO_CAPTURE_DETAIL_CPU");
    cd_consume=ab_value("NEUTRINO_CAPTURE_CONSUME");cd_no_d2h=ab_value("NEUTRINO_CAPTURE_NO_D2H");
    cd_no_pack=ab_value("NEUTRINO_CAPTURE_NO_PACK");
    if(getenv("NEUTRINO_CAPTURE_KERNEL"))cd_kernel=ab_value("NEUTRINO_CAPTURE_KERNEL");
    bulk_clear_policy=ab_value("NEUTRINO_CAPTURE_CLEAR");
    if(bulk_clear_policy<0 || bulk_clear_policy>2 || (bulk_clear_policy && !(ab_value("NEUTRINO_BULK") || ab_async || ab_arena))){
        fprintf(stderr,"Clear policy is experimental fixed block_sched only\n");exit(EXIT_FAILURE);
    }
    ab_bulk=ab_value("NEUTRINO_BULK");
    legacy_output_mode=ab_value("NEUTRINO_CAPTURE_OUTPUT");legacy_verify_encoder=ab_value("NEUTRINO_CAPTURE_VERIFY_ENCODER");
    if(legacy_output_mode){
        if(legacy_output_mode<1 || legacy_output_mode>3 || !ab_async || ab_bulk || ab_mode!=AB_FILES || (legacy_output_mode==3&&!ab_async_copy_stream)){
            fprintf(stderr,"Legacy output experiments require the fixed async ring\n");exit(EXIT_FAILURE);
        }
        bulk_sink=legacy_output_mode==1;bulk_compress=legacy_output_mode==2?2:0;
    }
    if((cd_enabled||cd_cpu||cd_consume||cd_no_d2h||cd_no_pack||cd_kernel!=2) &&
       (!ab_async||legacy_output_mode!=3||ab_bulk||cd_consume<0||cd_consume>2||cd_kernel<0||cd_kernel>2||
        (cd_no_d2h&&cd_consume!=2)||(cd_no_pack&&!cd_no_d2h)||legacy_verify_encoder)){
        fprintf(stderr,"Detailed LG64 controls require GPU13 async; no-D2H requires discard; validation separate\n");exit(EXIT_FAILURE);
    }
    if(cd_consume==1)bulk_sink=1;
    if(ab_arena){
        if(ab_mode!=AB_FILES||ab_async||ab_bulk||legacy_output_mode||cd_enabled||cd_cpu||cd_consume||cd_no_d2h||cd_no_pack||cd_kernel!=2||ab_host||ab_device_pool||ab_event_pool||ab_writer)async_fatal("incompatible arena controls");
        if(getenv("NEUTRINO_ARENA_DECOUPLE"))ar_decouple=ab_value("NEUTRINO_ARENA_DECOUPLE");
        if(getenv("NEUTRINO_ARENA_FRAMES"))ar_frames=ab_value("NEUTRINO_ARENA_FRAMES");
        if(getenv("NEUTRINO_ARENA_CHUNK_MIB"))ar_chunk=(size_t)ab_value("NEUTRINO_ARENA_CHUNK_MIB")<<20;
        if(getenv("NEUTRINO_ARENA_HOST_MIB"))ar_host_bytes=(size_t)ab_value("NEUTRINO_ARENA_HOST_MIB")<<20;
        if(getenv("NEUTRINO_ARENA_DEVICE_MIB"))ar_device_bytes=(size_t)ab_value("NEUTRINO_ARENA_DEVICE_MIB")<<20;
        ar_frame_copies=ab_value("NEUTRINO_ARENA_FRAME_COPIES");ar_token_flush=ab_value("NEUTRINO_ARENA_TOKEN_FLUSH");ar_diag=ab_value("NEUTRINO_ARENA_DIAG");ar_verify=ab_value("NEUTRINO_ARENA_VERIFY");ar_template=ab_value("NEUTRINO_ARENA_TEMPLATE");
        ar_delta=ab_value("NEUTRINO_ARENA_DELTA");ar_force_fallback=ab_value("NEUTRINO_ARENA_FORCE_FALLBACK");if(ar_force_fallback&&!ar_delta)async_fatal("forced fallback requires delta");
        ar_flex=getenv("NEUTRINO_ARENA_DELTA_WIDTH")||getenv("NEUTRINO_ARENA_ELAPSED_WIDTH");
        if(getenv("NEUTRINO_ARENA_DELTA_WIDTH"))ar_delta_width=ab_value("NEUTRINO_ARENA_DELTA_WIDTH");
        if(getenv("NEUTRINO_ARENA_ELAPSED_WIDTH"))ar_elapsed_width=ab_value("NEUTRINO_ARENA_ELAPSED_WIDTH");
        if(ar_flex&&(!ar_delta||ar_delta_width<2||ar_delta_width>4||ar_elapsed_width<2||ar_elapsed_width>4))async_fatal("invalid exact field widths");
        ar_hier=ab_value("NEUTRINO_ARENA_HIER");if(ar_hier&&(!ar_flex||ar_delta_width!=3))async_fatal("hierarchical codec requires explicit 24-bit SM reference");
        ar_omit=ab_value("NEUTRINO_ARENA_OMIT_ANCHOR");ar_reference=ab_value("NEUTRINO_ARENA_REFERENCE");ar_tile=ab_value("NEUTRINO_ARENA_TILE");
        if(getenv("NEUTRINO_ARENA_LOCAL_BITS"))ar_local_bits=ab_value("NEUTRINO_ARENA_LOCAL_BITS");
        if((ar_reference||ar_tile||ar_local_bits!=16)&&(!ar_reference||!ar_hier||ar_tile<0||ar_tile>32||(ar_tile&&(ar_tile<4||(ar_tile&(ar_tile-1))))||(ar_local_bits!=12&&ar_local_bits!=16)))async_fatal("invalid reference-domain controls");
        if(getenv("NEUTRINO_ARENA_CODEC_MIN_KIB")){int kib=ab_value("NEUTRINO_ARENA_CODEC_MIN_KIB");if(kib<4||kib>1024)async_fatal("invalid codec cutoff");ar_codec_min=(size_t)kib*1024;}
        if(ar_omit&&(!ar_reference||ar_local_bits!=12))async_fatal("anchor omission requires reference12");
        ar_batch_encoder=ab_value("NEUTRINO_ARENA_BATCH_ENCODER");ar_writer_mode=ab_value("NEUTRINO_ARENA_WRITER");
        if(getenv("NEUTRINO_ARENA_STDIO_KIB"))ar_stdio_bytes=(size_t)ab_value("NEUTRINO_ARENA_STDIO_KIB")<<10;
        ar_soft_bytes=(size_t)ab_value("NEUTRINO_ARENA_SOFT_KIB")<<10;ar_age_ns=(uint64_t)ab_value("NEUTRINO_ARENA_AGE_US")*1000;
        if(ar_batch_encoder<0||ar_batch_encoder>2||ar_writer_mode<0||ar_writer_mode>2||ar_stdio_bytes<4096||ar_stdio_bytes>(16ULL<<20))async_fatal("invalid arena experiment mode");
        ar_check_first=ab_value("NEUTRINO_ARENA_CHECK_FIRST");ar_release_checked=ab_value("NEUTRINO_ARENA_RELEASE_CHECKED");
        if((ar_check_first||ar_release_checked)&&(!ar_delta||ar_writer_mode!=1||ar_decouple||!ar_check_first))async_fatal("checked retirement requires coupled delta writev with preflight");
        if(ar_delta&&(ar_delta!=1||ar_decouple||ar_batch_encoder!=1))async_fatal("GPU9 requires batched encoding and coupled raw lifetime");
        ar_trim=ab_value("NEUTRINO_ARENA_TRIM");ar_incremental=ab_value("NEUTRINO_ARENA_INCREMENTAL");if(ar_incremental&&(ar_template||ar_trim))async_fatal("incremental allocation is a separate generic control");
        ar_hot=ab_value("NEUTRINO_ARENA_HOTPATH");if(ar_hot<0||ar_hot>3)async_fatal("invalid hotpath control");
        ar_nv=ab_value("NEUTRINO_ARENA_NVCOMP");if(getenv("NEUTRINO_NVCOMP_CHUNK_KIB"))ar_nv_chunk=ab_value("NEUTRINO_NVCOMP_CHUNK_KIB")*1024;
        if(ar_nv&&(ar_nv<1||ar_nv>2||ar_decouple||ar_delta||ar_omit||ar_template||ar_writer_mode!=1||ar_batch_encoder!=1||(ar_nv_chunk!=65536&&ar_nv_chunk!=262144)))async_fatal("invalid nvcomp controls");
        if(ar_frames<1||ar_frames>AR_FRAMES||ar_chunk>ar_device_bytes||ar_chunk>ar_host_bytes||!ar_chunk||ar_decouple<0||ar_decouple>1)async_fatal("invalid arena bounds");
    }
    if(ab_bulk){
        if(ab_value("NEUTRINO_BULK_BATCH"))bulk_batch_frames=ab_value("NEUTRINO_BULK_BATCH");
        bulk_step_flush=ab_value("NEUTRINO_BULK_STEP_FLUSH");bulk_decouple=ab_value("NEUTRINO_BULK_DECOUPLE");
        bulk_sink=ab_value("NEUTRINO_BULK_RAM");bulk_compress=ab_value("NEUTRINO_BULK_ZSTD");bulk_verify_codec=ab_value("NEUTRINO_BULK_VERIFY_CODEC");
        if(ab_value("NEUTRINO_BULK_PACK"))bulk_compress=2;
        bulk_hot_cache=ab_value("NEUTRINO_BULK_HOT_CACHE");
        if(ab_value("NEUTRINO_BULK_SMALL_HOSTS"))bulk_small_hosts=ab_value("NEUTRINO_BULK_SMALL_HOSTS");
        if(ab_value("NEUTRINO_BULK_SMALL_DEVICES"))bulk_small_devices=ab_value("NEUTRINO_BULK_SMALL_DEVICES");
        if(ab_value("NEUTRINO_BULK_LARGE_DEVICES"))bulk_large_devices=ab_value("NEUTRINO_BULK_LARGE_DEVICES");
        if(ab_mode!=AB_FILES || ab_async || ab_host || ab_writer || ab_device_pool || ab_event_pool ||
           bulk_batch_frames<1 || bulk_batch_frames>BULK_MAX_FRAMES || bulk_small_hosts<1 || bulk_small_hosts>64 || bulk_small_devices<1 || bulk_small_devices>64 ||
           bulk_large_devices<1 || bulk_large_devices>64 || (bulk_sink&&bulk_compress)){
            fprintf(stderr,"Invalid bulk experimental variant\n");exit(EXIT_FAILURE);
        }
    }
    if (ab_host < 0 || ab_host > 2 || ab_writer < 0 || ab_writer > 2 ||
        ((ab_host || ab_device_pool) && ab_mode < AB_BUFFERS) ||
        (ab_writer && ab_mode < AB_FILES) ||
        ((ab_no_events || ab_one_wait) && ab_mode != AB_FRAMEWORK) ||
        (ab_no_events && ab_one_wait) || (nt_passthrough && ab_mode != AB_PASS) ||
        (ab_async && (ab_mode!=AB_FILES || ab_host || ab_device_pool || ab_event_pool ||
                     ab_writer || ab_no_events || ab_one_wait || ab_async_slots<1 || ab_async_slots>64))) {
        fprintf(stderr, "Invalid ablation variant\n"); exit(EXIT_FAILURE);
    }
    if (ab_log_null) {
        ab_original_log = event_log;
        event_log = fopen("/dev/null", "w");
        if (!event_log) { perror("ablation log"); exit(EXIT_FAILURE); }
    }
    if (ab_mode == AB_PASS) nt_passthrough = 1;
}
static int ab_is_excluded(const char *name) {
    if (!ab_mode || !ab_exclude || !name) return 0;
    const char *p = ab_exclude;
    while (*p) {
        const char *end = strchr(p, ':'); size_t n = end ? (size_t)(end-p) : strlen(p);
        if (strlen(name) == n && !strncmp(p, name, n)) return 1;
        if (!end) break; p = end + 1;
    }
    return 0;
}
static ab_slot *ab_slot_get(CUstream stream) {
    CUcontext ctx; CUDA_CHECK(real_cuCtxGetCurrent(&ctx));
    ab_slot *s = ab_slots;
    for (; s; s=s->next) if (s->context==ctx && s->stream==stream) return s;
    s=calloc(1,sizeof(*s)); if (!s) { perror("ablation pool"); exit(EXIT_FAILURE); }
    s->context=ctx; s->stream=stream; s->next=ab_slots; ab_slots=s; return s;
}
static void *ab_host_get(ab_slot *s, size_t n) {
    if (s->host_cap<n) {
        if (s->host) { if (ab_host==2) CUDA_CHECK(real_cuMemFreeHost(s->host)); else free(s->host); }
        if (ab_host==2) CUDA_CHECK(real_cuMemAllocHost_v2(&s->host,n));
        else { s->host=malloc(n); if (!s->host) { perror("ablation host"); exit(EXIT_FAILURE); } }
        s->host_cap=n;
    }
    return s->host;
}
static CUdeviceptr ab_device_get(ab_slot *s, size_t n) {
    if (s->device_cap<n) {
        if (s->device) CUDA_CHECK(real_cuMemFree_v2(s->device));
        CUDA_CHECK(real_cuMemAlloc_v2(&s->device,n)); s->device_cap=n;
    }
    return s->device;
}
static void ab_record(const char *name, unsigned *dims, int kind, int excluded,
                      int fallback, int error, int cold, uint64_t planned,
                      uint64_t d2h, uint64_t trace) {
    if (!ab_stats || getpid()!=ab_owner) return;
    pthread_mutex_lock(&ab_lock);
    int i;
    for(i=0;i<ab_group_count;i++)
        if(!strcmp(ab_groups[i].name,name?name:"<unknown>") &&
           !memcmp(ab_groups[i].dims,dims,7*sizeof(unsigned))) break;
    if(i==AB_GROUPS) { fprintf(stderr,"ablation group overflow\n"); exit(EXIT_FAILURE); }
    ab_group *g=&ab_groups[i];
    if(i==ab_group_count) {
        ab_group_count++;
        snprintf(g->name,sizeof(g->name),"%s",name?name:"<unknown>");
        memcpy(g->dims,dims,7*sizeof(unsigned));
    }
    g->calls++; g->original+=kind==0; g->pruned+=kind==1; g->probed+=kind==2;
    g->excluded+=excluded; g->fallback+=fallback; g->errors+=error;
    g->planned+=planned; g->d2h+=d2h; g->trace+=trace; ab_cold+=cold;
    pthread_mutex_unlock(&ab_lock);
}
/* Boundary closes request output and is included in workload request timing. */
int neutrino_ablation_boundary(void) {
    if (!ab_mode || getpid()!=ab_owner) return 0;
    if(ab_async && async_boundary())return -1;
    if(ab_bulk && bulk_boundary())return -1;
    if(ab_arena && ar_boundary())return -1;
    if (ab_output) {
        int bad=ferror(ab_output); if(fclose(ab_output)) bad=1;
        ab_output=NULL; free(ab_stdio_buffer); ab_stdio_buffer=NULL;
        if(bad) return -1;
    }
    return 0;
}
int neutrino_ablation_reset(void) {
    if (!ab_mode || getpid()!=ab_owner) return -1;
    if(neutrino_ablation_boundary()) return -1;
    if(ab_async)async_reset();
    if(ab_bulk)bulk_reset();
    if(ab_arena)ar_reset();
    pthread_mutex_lock(&ab_lock);
    memset(ab_groups,0,sizeof(ab_groups)); ab_group_count=0; ab_cold=0;
    pthread_mutex_unlock(&ab_lock); return 0;
}
int neutrino_capture_detail_start(void){
    if(!cd_enabled)return 0;
    if(neutrino_ablation_boundary())return -1;
    for(async_engine*e=async_engines;e;e=e->next){pthread_mutex_lock(&e->mutex);cd_start(e->detail);pthread_mutex_unlock(&e->mutex);}return 0;
}
int neutrino_capture_detail_stop(const char*path){
    if(!cd_enabled)return 0;
    uint64_t end=cd_clock(CLOCK_MONOTONIC);
    for(async_engine*e=async_engines;e;e=e->next){pthread_mutex_lock(&e->mutex);
        if(!atomic_load(&e->detail->active)){pthread_mutex_unlock(&e->mutex);continue;}
        cd_state*d=e->detail;if(e->consumed!=e->published||d->busy||d->pending||d->busy_bytes){pthread_mutex_unlock(&e->mutex);return -1;}
        cd_transition(d,end);d->end=end;atomic_store(&d->active,0);pthread_mutex_unlock(&e->mutex);
    }
    if(!path)return 0;
    FILE*fp=fopen(path,"w");if(!fp)return -1;fputs("{\"engines\":[",fp);int sep=0;
    for(async_engine*e=async_engines;e;e=e->next){if(sep)fputc(',',fp);cd_json(fp,e->detail);sep=1;}
    fputs("]}\n",fp);int bad=ferror(fp);if(fclose(fp))bad=1;return bad?-1:0;
}
int neutrino_ablation_flush(const char *path) {
    if (!ab_mode || getpid()!=ab_owner || !ab_stats) return -1;
    FILE *fp=fopen(path,"w"); if(!fp) return -1;
    pthread_mutex_lock(&ab_lock);
    fprintf(fp,"{\"mode\":%d,\"cold\":%" PRIu64 ",\"groups\":[",ab_mode,ab_cold);
    for(int i=0;i<ab_group_count;i++) {
        ab_group *g=&ab_groups[i]; if(i) fputc(',',fp);
        fputs("{\"kernel\":",fp); nt_json_string(fp,g->name);
        fputs(",\"shape\":[",fp);
        for(int j=0;j<7;j++) fprintf(fp,"%s%u",j?",":"",g->dims[j]);
        fprintf(fp,"],\"calls\":%" PRIu64 ",\"original\":%" PRIu64
            ",\"pruned\":%" PRIu64 ",\"probed\":%" PRIu64
            ",\"excluded\":%" PRIu64 ",\"fallback\":%" PRIu64
            ",\"errors\":%" PRIu64 ",\"planned_bytes\":%" PRIu64
            ",\"d2h_bytes\":%" PRIu64 ",\"trace_bytes\":%" PRIu64 "}",
            g->calls,g->original,g->pruned,g->probed,g->excluded,g->fallback,
            g->errors,g->planned,g->d2h,g->trace);
    }
    fputc(']',fp);if(ab_async)async_json(fp);if(ab_bulk)bulk_json(fp);if(ab_arena)ar_json(fp);fputs("}\n",fp); pthread_mutex_unlock(&ab_lock);
    int bad=ferror(fp); if(fclose(fp)) bad=1; return bad?-1:0;
}
int neutrino_ablation_release(void) {
    if (!ab_mode || getpid()!=ab_owner) return 0;
    if(neutrino_ablation_boundary()) return -1;
    if(ab_async)async_release();
    if(ab_bulk)bulk_release();
    if(ab_arena)ar_release();
    /* Called before backend context teardown, on the benchmark submitting thread. */
    CUcontext saved; CUDA_CHECK(real_cuCtxGetCurrent(&saved));
    while(ab_slots) {
        ab_slot *s=ab_slots; CUDA_CHECK(real_cuCtxSetCurrent(s->context));
        if(s->host) { if(ab_host==2) CUDA_CHECK(real_cuMemFreeHost(s->host)); else free(s->host); }
        if(s->device) CUDA_CHECK(real_cuMemFree_v2(s->device));
        if(s->start) { CUDA_CHECK(real_cuEventDestroy_v2(s->start)); CUDA_CHECK(real_cuEventDestroy_v2(s->end)); }
        ab_slots=s->next; free(s);
    }
    CUDA_CHECK(real_cuCtxSetCurrent(saved)); free(ab_sink); ab_sink=NULL; ab_sink_cap=0;
    return 0;
}
#endif
