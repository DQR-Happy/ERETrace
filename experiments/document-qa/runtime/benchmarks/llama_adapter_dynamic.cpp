#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <vector>

struct Session {
    llama_model * model;
    llama_context * ctx;
    const llama_vocab * vocab;
};

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
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    auto * model = llama_model_load_from_file(model_path, mp);
    if (!model) return nullptr;
    auto cp = llama_context_default_params();
    cp.n_ctx = 8192;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_batch = std::getenv("CASE_BATCH") ? std::atoi(std::getenv("CASE_BATCH")) : 512;
    cp.n_ubatch = cp.n_batch;
    int threads = 48;
    if (const char * value = std::getenv("NEUTRINO_BENCH_THREADS")) {
        char * end = nullptr;
        long parsed = std::strtol(value, &end, 10);
        if (!end || *end || parsed < 1 || parsed > 1024) {
            llama_model_free(model);
            return nullptr;
        }
        threads = static_cast<int>(parsed);
    }
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) { llama_model_free(model); return nullptr; }
    return new Session{model, ctx, llama_model_get_vocab(model)};
}

int qwen_prefill(void * handle, const char * prompt) {
    auto & s = *static_cast<Session *>(handle);
    llama_memory_clear(llama_get_memory(s.ctx), true);
    const int needed = -llama_tokenize(s.vocab, prompt, std::strlen(prompt), nullptr, 0, true, true);
    if (needed <= 0 || needed >= (int)llama_n_ctx(s.ctx)) return -100;
    std::vector<llama_token> tokens(needed);
    int n = llama_tokenize(s.vocab, prompt, std::strlen(prompt), tokens.data(), tokens.size(), true, true);
    if (n <= 0) return -100;
    // Preserve the 512-token physical batch and use ordered prefill chunks.
    // llama_batch_get_one continues the existing sequence position.
    const int chunk = std::getenv("CASE_BATCH") ? std::atoi(std::getenv("CASE_BATCH")) : 512;
    for (int offset = 0; offset < n; offset += chunk) {
        auto batch = llama_batch_get_one(tokens.data() + offset, std::min(chunk, n-offset));
        int rc = llama_decode(s.ctx, batch);
        if (rc != 0) { llama_synchronize(s.ctx); return -200 + rc; }
    }
    llama_synchronize(s.ctx);
    return n;
}

int qwen_tokenize(void * handle,const char *text,int *out,int cap){
    auto &s=*static_cast<Session*>(handle);
    return llama_tokenize(s.vocab,text,std::strlen(text),out,cap,true,true);
}
int qwen_vocab_size(void * handle) {
    return llama_vocab_n_tokens(static_cast<Session *>(handle)->vocab);
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

int qwen_copy_logits(void * handle, float * dst, int count) {
    auto & s = *static_cast<Session *>(handle);
    int n = llama_vocab_n_tokens(s.vocab);
    if (count != n) return -1;
    std::copy_n(llama_get_logits_ith(s.ctx, -1), n, dst);
    return n;
}

int qwen_argmax(void * handle) {
    auto & s = *static_cast<Session *>(handle);
    const auto * logits = llama_get_logits_ith(s.ctx, -1);
    int n = llama_vocab_n_tokens(s.vocab);
    return static_cast<int>(std::max_element(logits, logits + n) - logits);
}

int qwen_generate(void * handle, int count, int * output) {
    auto & s = *static_cast<Session *>(handle);
    int n_vocab = llama_vocab_n_tokens(s.vocab);
    for (int i = 0; i < count; ++i) {
        const auto * logits = llama_get_logits_ith(s.ctx, -1);
        llama_token token = std::max_element(logits, logits + n_vocab) - logits;
        output[i] = token;
        if (llama_vocab_is_eog(s.vocab, token)) return i + 1;
        auto batch = llama_batch_get_one(&token, 1);
        int rc = llama_decode(s.ctx, batch);
        if (rc != 0) return -200 + rc;
    }
    llama_synchronize(s.ctx);
    return count;
}

int qwen_detokenize(void * handle, const int * tokens, int count, char * text, int capacity) {
    auto & s = *static_cast<Session *>(handle);
    return llama_detokenize(s.vocab, tokens, count, text, capacity, false, true);
}

void qwen_destroy(void * handle) {
    auto * s = static_cast<Session *>(handle);
    llama_free(s->ctx);
    llama_model_free(s->model);
    delete s;
}
}
