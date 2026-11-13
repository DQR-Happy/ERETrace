#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct Session {
    llama_model * model{};
    llama_context * ctx{};
    const llama_vocab * vocab{};
    int max_sequences{1};
};

static int env_int(const char * name, int fallback, int minimum, int maximum) {
    const char * value = std::getenv(name);
    if (!value || !*value) return fallback;
    char * end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (!end || *end || parsed < minimum || parsed > maximum) return -1;
    return static_cast<int>(parsed);
}

extern "C" {
void * qwen_create(const char * model_path, const char * backend_dir) {
    ggml_backend_load_all_from_path(backend_dir);
    llama_backend_init();
    bool gpu = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto type = ggml_backend_dev_type(ggml_backend_dev_get(i));
        gpu |= type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
    }
    if (!gpu) return nullptr;
    const int n_ctx = env_int("NEUTRINO_BENCH_CONTEXT", 4096, 128, 131072);
    const int n_seq = env_int("NEUTRINO_BENCH_SEQUENCES", 1, 1, 8);
    const int threads = env_int("NEUTRINO_BENCH_THREADS", 48, 1, 1024);
    if (n_ctx < 0 || n_seq < 0 || threads < 0) return nullptr;
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    auto * model = llama_model_load_from_file(model_path, mp);
    if (!model) return nullptr;
    auto cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_seq_max = n_seq;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) { llama_model_free(model); return nullptr; }
    return new Session{model, ctx, llama_model_get_vocab(model), n_seq};
}

int qwen_vocab_size(void * handle) {
    return llama_vocab_n_tokens(static_cast<Session *>(handle)->vocab);
}

// Produces a deterministic valid token-ID sequence of exactly n tokens.
// This is a synthetic fixed-token workload used only for controlled length/batch tests.
int qwen_fill_token_ids(void * handle, int n, int * output) {
    if (!handle || n <= 0 || !output) return -1;
    auto & s = *static_cast<Session *>(handle);
    const char * candidates[] = {" x", " a", " the", " 0", " .", " GPU", " token"};
    llama_token token = LLAMA_TOKEN_NULL;
    for (const char * candidate : candidates) {
        llama_token tmp[8];
        int count = llama_tokenize(s.vocab, candidate, std::strlen(candidate), tmp, 8, false, false);
        if (count == 1) { token = tmp[0]; break; }
    }
    if (token == LLAMA_TOKEN_NULL) return -2;
    std::fill(output, output + n, static_cast<int>(token));
    return n;
}

int qwen_prefill_ids(void * handle, const int * token_ids, int n) {
    if (!handle || !token_ids || n <= 0 || n >= static_cast<int>(llama_n_ctx(static_cast<Session *>(handle)->ctx))) return -1;
    auto & s = *static_cast<Session *>(handle);
    llama_memory_clear(llama_get_memory(s.ctx), true);
    std::vector<llama_token> tokens(token_ids, token_ids + n);
    for (int offset = 0; offset < n; offset += 512) {
        auto batch = llama_batch_get_one(tokens.data() + offset, std::min(512, n - offset));
        int rc = llama_decode(s.ctx, batch);
        if (rc != 0) { llama_synchronize(s.ctx); return -200 + rc; }
    }
    llama_synchronize(s.ctx);
    return n;
}

int qwen_argmax(void * handle) {
    auto & s = *static_cast<Session *>(handle);
    const auto * logits = llama_get_logits_ith(s.ctx, -1);
    int n = llama_vocab_n_tokens(s.vocab);
    return static_cast<int>(std::max_element(logits, logits + n) - logits);
}

int qwen_step(void * handle, int token_id, int position) {
    auto & s = *static_cast<Session *>(handle);
    if (!llama_memory_seq_rm(llama_get_memory(s.ctx), 0, position, -1)) return -300;
    llama_token token = token_id;
    auto batch = llama_batch_get_one(&token, 1);
    int rc = llama_decode(s.ctx, batch);
    llama_synchronize(s.ctx);
    return rc;
}

// Same llama_context, independent sequence IDs and one producer submission per decode step.
int qwen_batch_prefill(void * handle, const int * token_ids, int batch_size, int input_tokens, int * last_rows) {
    if (!handle || !token_ids || !last_rows || batch_size < 1 || batch_size > 8) return -1;
    auto & s = *static_cast<Session *>(handle);
    if (batch_size > s.max_sequences || input_tokens <= 0) return -2;
    llama_memory_clear(llama_get_memory(s.ctx), true);
    auto batch = llama_batch_init(512, 0, batch_size);
    if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) return -3;
    std::vector<int> last(batch_size, -1);
    const int tokens_per_chunk = std::max(1, 512 / batch_size);
    int rc = 0;
    for (int base = 0; base < input_tokens; base += tokens_per_chunk) {
        batch.n_tokens = 0;
        for (int seq = 0; seq < batch_size; ++seq) {
            const int end = std::min(input_tokens, base + tokens_per_chunk);
            for (int pos = base; pos < end; ++pos) {
                const int i = batch.n_tokens++;
                batch.token[i] = token_ids[seq * input_tokens + pos];
                batch.pos[i] = pos;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = seq;
                batch.logits[i] = (pos == input_tokens - 1) ? 1 : 0;
                if (pos == input_tokens - 1) last[seq] = i;
            }
        }
        rc = llama_decode(s.ctx, batch);
        if (rc != 0) break;
    }
    llama_batch_free(batch);
    if (rc != 0) { llama_synchronize(s.ctx); return -200 + rc; }
    llama_synchronize(s.ctx);
    for (int seq = 0; seq < batch_size; ++seq) {
        if (last[seq] < 0) return -4;
        last_rows[seq] = last[seq];
    }
    return 0;
}

int qwen_batch_argmax(void * handle, const int * rows, int batch_size, int * output) {
    if (!handle || !rows || !output || batch_size < 1 || batch_size > 8) return -1;
    auto & s = *static_cast<Session *>(handle);
    const int n_vocab = llama_vocab_n_tokens(s.vocab);
    for (int i = 0; i < batch_size; ++i) {
        const float * logits = llama_get_logits_ith(s.ctx, rows[i]);
        output[i] = static_cast<int>(std::max_element(logits, logits + n_vocab) - logits);
    }
    return batch_size;
}

int qwen_batch_step(void * handle, const int * input_ids, int batch_size, int position, int * output_ids) {
    if (!handle || !input_ids || !output_ids || batch_size < 1 || batch_size > 8) return -1;
    auto & s = *static_cast<Session *>(handle);
    if (batch_size > s.max_sequences) return -2;
    auto batch = llama_batch_init(batch_size, 0, batch_size);
    if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) return -3;
    batch.n_tokens = batch_size;
    for (int seq = 0; seq < batch_size; ++seq) {
        batch.token[seq] = input_ids[seq];
        batch.pos[seq] = position;
        batch.n_seq_id[seq] = 1;
        batch.seq_id[seq][0] = seq;
        batch.logits[seq] = 1;
    }
    int rc = llama_decode(s.ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) { llama_synchronize(s.ctx); return -200 + rc; }
    llama_synchronize(s.ctx);
    const int n_vocab = llama_vocab_n_tokens(s.vocab);
    for (int seq = 0; seq < batch_size; ++seq) {
        const float * logits = llama_get_logits_ith(s.ctx, seq);
        output_ids[seq] = static_cast<int>(std::max_element(logits, logits + n_vocab) - logits);
    }
    return batch_size;
}

void qwen_destroy(void * handle) {
    auto * s = static_cast<Session *>(handle);
    if (!s) return;
    llama_free(s->ctx);
    llama_model_free(s->model);
    delete s;
}
}
