/* Optional CUDA host timing. No CUDA calls, per-stage allocation or per-launch I/O.
 * Stages partition a sampled hook invocation; GPU observations overlap that wall
 * time and must never be added to it. See docs/timing.md for the measurement contract.
 */
#ifndef NEUTRINO_TIMING_H
#define NEUTRINO_TIMING_H
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <inttypes.h>
#include <errno.h>

#define NT_STAGES(X) \
 X(LOOKUP, "cache_lookup") \
 X(EVENT_SETUP, "event_setup") \
 X(CACHE_PREPARE, "cold_directory") \
 X(BINARY_EXPORT, "cold_binary_export") \
 X(PROBE_PROCESS, "cold_python_process") \
 X(METADATA_READ, "cold_metadata_read") \
 X(BINARY_READ, "cold_binary_read") \
 X(MODULE_PROBED, "cold_module_load_probed") \
 X(MODULE_PRUNED, "cold_module_load_pruned") \
 X(MODULE_COUNT, "cold_module_load_count") \
 X(CACHE_STORE, "cold_cache_store") \
 X(LOG, "event_log") \
 X(COUNT_ALLOC, "count_alloc_and_args") \
 X(COUNT_ZERO, "count_zero") \
 X(COUNT_SUBMIT, "count_submit") \
 X(COUNT_D2H, "count_d2h_and_wait") \
 X(COUNT_REDUCE, "count_cpu_reduce") \
 X(COUNT_FREE, "count_free") \
 X(BUFFER_PLAN, "buffer_size_plan") \
 X(HOST_ALLOC, "host_buffer_alloc") \
 X(DEVICE_ALLOC, "device_buffer_alloc") \
 X(DEVICE_ZERO, "device_buffer_zero") \
 X(ARGS, "probe_args_and_attributes") \
 X(PROLOGUE_SYNC, "prologue_event_wait") \
 X(PROBE_SUBMIT, "probe_submit") \
 X(PROBE_WAIT, "probe_event_wait") \
 X(D2H, "trace_d2h") \
 X(TRACE_OPEN, "trace_open") \
 X(TRACE_HEADER, "trace_header_write") \
 X(TRACE_WRITE, "trace_payload_write") \
 X(TRACE_CLOSE, "trace_close") \
 X(HOST_FREE, "host_buffer_free") \
 X(DEVICE_FREE, "device_buffer_free") \
 X(EPILOGUE_SYNC, "epilogue_event_wait") \
 X(BENCHMARK, "legacy_benchmark_replay") \
 X(CALLBACK, "callback_process") \
 X(CLEANUP, "cleanup") \
 X(EVENT_DESTROY, "event_destroy") \
 X(ORIGINAL_SUBMIT, "original_submit")

#define NT_ENUM(id, name) NT_##id,
enum nt_stage { NT_STAGES(NT_ENUM) NT_STAGE_COUNT };
#undef NT_ENUM
#define NT_NAME(id, name) name,
static const char *nt_stage_names[] = { NT_STAGES(NT_NAME) };
#undef NT_NAME
enum nt_outcome { NT_CAPTURED, NT_FALLBACK, NT_PASSTHROUGH, NT_BYPASS };
static const char *nt_outcome_names[] = {"captured", "fallback", "passthrough", "launch_ex_bypass"};
enum { NT_GROUPS = 1024 };
typedef struct { uint64_t count, total, min, max; } nt_stat;
typedef struct {
    int enabled, sampled, stage, cold;
    uintptr_t key;
    const char *name;
    uint64_t begin, last, entered, ns[NT_STAGE_COUNT];
    uint64_t trace_bytes, d2h_bytes;
    double gpu_ms;
    int gpu_valid;
} nt_span;
typedef struct {
    int used, cold, outcome;
    uintptr_t key;
    char name[256];
    uint64_t calls, sampled, errors, trace_bytes, d2h_bytes;
    nt_stat wall, stages[NT_STAGE_COUNT], gpu;
} nt_group;
static nt_group *nt_groups;
static pthread_mutex_t nt_lock = PTHREAD_MUTEX_INITIALIZER;
static int nt_enabled, nt_passthrough;
static pid_t nt_owner;
static uint64_t nt_sample_every = 1, nt_flush_every, nt_calls, nt_sampled;
static uint64_t nt_init_ns, nt_start_ns, nt_accounting_ns, nt_snapshot_ns, nt_snapshots;
static uint64_t nt_clock_pair_mean_ns;
static char nt_path[4096];
static __thread uint64_t nt_sequence;

static inline uint64_t nt_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + ts.tv_nsec;
}
static void nt_stat_add(nt_stat *s, uint64_t ns) {
    if (!s->count || ns < s->min) s->min = ns;
    if (ns > s->max) s->max = ns;
    s->count++; s->total += ns;
}
static void nt_json_string(FILE *f, const char *s) {
    fputc('"', f);
    for (const unsigned char *p = (const unsigned char *)s; *p; ++p) {
        if (*p == '"' || *p == '\\') { fputc('\\', f); fputc(*p, f); }
        else if (*p < 32 || *p >= 127) fprintf(f, "\\u%04x", *p);
        else fputc(*p, f);
    }
    fputc('"', f);
}
static void nt_json_stat(FILE *f, const nt_stat *s) {
    fprintf(f, "{\"samples\":%" PRIu64 ",\"total_ns\":%" PRIu64
            ",\"min_ns\":%" PRIu64 ",\"max_ns\":%" PRIu64 "}",
            s->count, s->total, s->min, s->max);
}
/* Caller holds nt_lock. Atomic replace: readers never see a partial JSON file.
 * Snapshot cost is reported in the NEXT snapshot (including the final one).
 * No signal handlers: arbitrary application SIGTERM semantics are preserved. */
static int nt_snapshot_locked(const char *kind) {
    uint64_t begin = nt_now();
    char tmp[4128];
    snprintf(tmp, sizeof(tmp), "%s.tmp", nt_path);
    FILE *f = fopen(tmp, "w");
    if (!f) { perror("Neutrino timing snapshot"); return -1; }
    fprintf(f, "{\"schema_version\":1,\"pid\":%ld,\"clock\":\"CLOCK_MONOTONIC\",\"snapshot_kind\":",
            (long)nt_owner);
    nt_json_string(f, kind);
    fprintf(f, ",\"sample_every\":%" PRIu64 ",\"calls_completed\":%" PRIu64
            ",\"calls_sampled\":%" PRIu64 ",\"init_ns\":%" PRIu64
            ",\"window_elapsed_ns\":%" PRIu64 ",\"accounting_ns\":%" PRIu64
            ",\"previous_snapshot_ns\":%" PRIu64 ",\"previous_snapshots\":%" PRIu64
            ",\"clock_pair_mean_ns\":%" PRIu64 ",\"groups\":[",
            nt_sample_every, nt_calls, nt_sampled, nt_init_ns, begin - nt_start_ns,
            nt_accounting_ns, nt_snapshot_ns, nt_snapshots, nt_clock_pair_mean_ns);
    int comma = 0;
    for (int i = 0; i < NT_GROUPS; ++i) {
        nt_group *g = &nt_groups[i];
        if (!g->used) continue;
        fprintf(f, "%s{\"function\":\"0x%" PRIxPTR "\",\"kernel\":", comma++ ? "," : "", g->key);
        nt_json_string(f, g->name);
        fprintf(f, ",\"cold\":%s,\"outcome\":\"%s\",\"calls\":%" PRIu64
                ",\"sampled\":%" PRIu64 ",\"errors\":%" PRIu64
                ",\"sampled_trace_bytes\":%" PRIu64 ",\"sampled_d2h_bytes\":%" PRIu64 ",\"host_wall\":",
                g->cold ? "true" : "false", i == NT_GROUPS - 1 ? "overflow" : nt_outcome_names[g->outcome],
                g->calls, g->sampled, g->errors, g->trace_bytes, g->d2h_bytes);
        nt_json_stat(f, &g->wall);
        fputs(",\"gpu_probe_interval\":", f); nt_json_stat(f, &g->gpu);
        fputs(",\"stages\":{", f);
        int sc = 0;
        for (int j = 0; j < NT_STAGE_COUNT; ++j) {
            if (!g->stages[j].count) continue;
            fprintf(f, "%s\"%s\":", sc++ ? "," : "", nt_stage_names[j]);
            nt_json_stat(f, &g->stages[j]);
        }
        fputs("}}", f);
    }
    fputs("]}\n", f);
    int failed = ferror(f);
    if (fclose(f) != 0) failed = 1;
    if (failed || rename(tmp, nt_path) != 0) { unlink(tmp); return -1; }
    nt_snapshot_ns += nt_now() - begin; nt_snapshots++;
    return 0;
}
/* Public control API: invoke at a quiescent workload boundary, never in a signal handler. */
int neutrino_timing_flush(void) {
    if (!nt_enabled || getpid() != nt_owner) return -1;
    pthread_mutex_lock(&nt_lock);
    int rc = nt_snapshot_locked("explicit");
    pthread_mutex_unlock(&nt_lock);
    return rc;
}
int neutrino_timing_reset(void) {
    if (!nt_enabled || getpid() != nt_owner) return -1;
    pthread_mutex_lock(&nt_lock);
    memset(nt_groups, 0, NT_GROUPS * sizeof(*nt_groups));
    nt_calls = nt_sampled = nt_accounting_ns = nt_snapshot_ns = nt_snapshots = 0;
    nt_start_ns = nt_now();
    pthread_mutex_unlock(&nt_lock);
    return 0;
}
static void nt_at_exit(void) {
    if (!nt_enabled || getpid() != nt_owner) return; /* fork/exec failure child */
    pthread_mutex_lock(&nt_lock);
    nt_snapshot_locked("normal_exit");
    pthread_mutex_unlock(&nt_lock);
}
static void nt_init(const char *result_dir) {
    const char *v = getenv("NEUTRINO_CAPTURE_MODE");
    nt_passthrough = v && strcmp(v, "passthrough") == 0;
    v = getenv("NEUTRINO_TIMING");
    if (!v || strcmp(v, "1") != 0) return;
    nt_groups = calloc(NT_GROUPS, sizeof(*nt_groups));
    if (!nt_groups) return;
    nt_owner = getpid(); nt_start_ns = nt_now();
    v = getenv("NEUTRINO_TIMING_SAMPLE_EVERY");
    if (v && strtoull(v, NULL, 10)) nt_sample_every = strtoull(v, NULL, 10);
    v = getenv("NEUTRINO_TIMING_FLUSH_EVERY");
    if (v) nt_flush_every = strtoull(v, NULL, 10);
    v = getenv("NEUTRINO_TIMING_DIR");
    if (v && *v) snprintf(nt_path, sizeof(nt_path), "%s/timing-%ld.json", v, (long)nt_owner);
    else snprintf(nt_path, sizeof(nt_path), "%s/../timing-%ld.json", result_dir, (long)nt_owner);
    for (int i = 0; i < 1000; ++i) { uint64_t a = nt_now(); nt_clock_pair_mean_ns += nt_now() - a; }
    nt_clock_pair_mean_ns /= 1000;
    nt_enabled = 1;
    atexit(nt_at_exit);
}
static inline nt_span nt_begin(void *function) {
    nt_span s = {0};
    if (!nt_enabled || getpid() != nt_owner) return s;
    s.enabled = 1; s.key = (uintptr_t)function; s.name = "<unknown>";
    s.sampled = nt_sequence++ % nt_sample_every == 0;
    if (s.sampled) {
        s.begin = s.last = nt_now(); s.stage = NT_LOOKUP;
        s.entered = UINT64_C(1) << NT_LOOKUP;
    }
    return s;
}
static inline void nt_stage(nt_span *s, enum nt_stage next) {
    if (!s->sampled) return;
    uint64_t now = nt_now();
    s->ns[s->stage] += now - s->last;
    s->last = now; s->stage = next; s->entered |= UINT64_C(1) << next;
}
static void nt_finish(nt_span *s, enum nt_outcome outcome, int error) {
    if (!s->enabled) return;
    uint64_t done = nt_now();
    if (s->sampled) s->ns[s->stage] += done - s->last;
    pthread_mutex_lock(&nt_lock);
    size_t h = ((s->key >> 4) ^ ((unsigned)outcome * 131u) ^ (s->cold * 61u)) % (NT_GROUPS - 1);
    nt_group *g = NULL;
    for (int i = 0; i < NT_GROUPS - 1; ++i) {
        nt_group *candidate = &nt_groups[(h + i) % (NT_GROUPS - 1)];
        if (!candidate->used || (candidate->key == s->key && candidate->cold == s->cold && candidate->outcome == (int)outcome)) { g = candidate; break; }
    }
    if (!g) g = &nt_groups[NT_GROUPS - 1];
    if (!g->used) {
        g->used = 1; g->key = s->key; g->cold = s->cold; g->outcome = outcome;
        snprintf(g->name, sizeof(g->name), "%s", s->name ? s->name : "<unknown>");
    }
    g->calls++; nt_calls++; if (error) g->errors++;
    if (s->sampled) {
        g->sampled++; nt_sampled++;
        nt_stat_add(&g->wall, done - s->begin);
        for (int i = 0; i < NT_STAGE_COUNT; ++i)
            if (s->entered & (UINT64_C(1) << i)) nt_stat_add(&g->stages[i], s->ns[i]);
        if (s->gpu_valid) nt_stat_add(&g->gpu, (uint64_t)(s->gpu_ms * 1e6));
        g->trace_bytes += s->trace_bytes; g->d2h_bytes += s->d2h_bytes;
    }
    nt_accounting_ns += nt_now() - done;
    if (nt_flush_every && nt_calls % nt_flush_every == 0) nt_snapshot_locked("periodic");
    pthread_mutex_unlock(&nt_lock);
}
#define NT_STAGE(id) nt_stage(&timing, NT_##id)
#endif
