// Loads a GGUF and generates text — the end-to-end check on a converted model.
//
// Bit-identical tensors prove the weights survived conversion. This proves the
// result is actually a working model: that the metadata llama.cpp needs is
// present and correct, that the vocabulary tokenizes and detokenizes, and that
// the RoPE permutation produces coherent text rather than fluent-looking noise.
//
// Greedy decoding by default so two runs are directly comparable.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "llama.h"

namespace {

void quiet_log(ggml_log_level level, const char * text, void * /*user_data*/) {
    if (level >= GGML_LOG_LEVEL_ERROR && text != nullptr) {
        std::fputs(text, stderr);
    }
}

std::string piece_for(const llama_vocab * vocab, llama_token token) {
    char buf[256];
    const int32_t n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, false);
    return n <= 0 ? std::string() : std::string(buf, static_cast<size_t>(n));
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <model.gguf> [prompt] [n_tokens]\n", argv[0]);
        return 2;
    }

    const std::string model_path = argv[1];
    const std::string prompt = argc > 2 ? argv[2] : "The capital of France is";
    const int n_predict = argc > 3 ? std::atoi(argv[3]) : 48;

    llama_log_set(quiet_log, nullptr);
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load %s\n", model_path.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    cparams.n_batch = 512;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    std::vector<llama_token> tokens(prompt.size() + 16);
    int32_t n_prompt = llama_tokenize(
        vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
        tokens.data(), static_cast<int32_t>(tokens.size()), true, true);
    if (n_prompt < 0) {
        tokens.resize(static_cast<size_t>(-n_prompt));
        n_prompt = llama_tokenize(
            vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
            tokens.data(), static_cast<int32_t>(tokens.size()), true, true);
    }
    tokens.resize(static_cast<size_t>(n_prompt));

    std::printf("model:  %s\n", model_path.c_str());
    std::printf("prompt: %s\n", prompt.c_str());
    std::printf("tokens: [");
    for (int32_t i = 0; i < n_prompt; ++i) {
        std::printf("%s%d", i ? ", " : "", tokens[i]);
    }
    std::printf("]\n\noutput: %s", prompt.c_str());
    std::fflush(stdout);

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    llama_batch batch = llama_batch_get_one(tokens.data(), n_prompt);
    if (llama_decode(ctx, batch) != 0) {
        std::fprintf(stderr, "\nprefill failed\n");
        return 1;
    }

    for (int i = 0; i < n_predict; ++i) {
        const llama_token id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) {
            std::printf(" [end-of-generation]");
            break;
        }
        llama_sampler_accept(smpl, id);

        const std::string piece = piece_for(vocab, id);
        std::fwrite(piece.data(), 1, piece.size(), stdout);
        std::fflush(stdout);

        llama_token next = id;
        llama_batch step = llama_batch_get_one(&next, 1);
        if (llama_decode(ctx, step) != 0) {
            std::fprintf(stderr, "\ndecode failed\n");
            break;
        }
    }
    std::printf("\n");

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
