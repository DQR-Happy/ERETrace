#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cstring>
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
        gpu |= ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU;
    }
    if (!gpu) return nullptr;
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    auto * model = llama_model_load_from_file(model_path, mp);
    if (!model) return nullptr;
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_threads = 48;
    cp.n_threads_batch = 48;
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) { llama_model_free(model); return nullptr; }
    return new Session{model, ctx, llama_model_get_vocab(model)};
}

int qwen_prefill(void * handle, const char * prompt) {
    auto & s = *static_cast<Session *>(handle);
    llama_memory_clear(llama_get_memory(s.ctx), true);
    std::vector<llama_token> tokens(512);
    int n = llama_tokenize(s.vocab, prompt, std::strlen(prompt), tokens.data(), tokens.size(), true, true);
    if (n <= 0) return -100;
    auto batch = llama_batch_get_one(tokens.data(), n);
    int rc = llama_decode(s.ctx, batch);
    llama_synchronize(s.ctx);
    return rc == 0 ? n : -200 + rc;
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
