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
#include "gguf_summary.h"

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

/// Formats a single user turn with the model's own chat template.
///
/// Verifies the other half of what the converter wrote: bit-identical tensors
/// prove the weights are right, but an instruct model also needs its
/// tokenizer.chat_template to survive conversion and be applicable. Without it,
/// the model is prompted as a base model and answers markedly worse.
bool apply_chat_template(
    const llama_model * model, const std::string & user_text, std::string * out) {

    const char * tmpl = llama_model_chat_template(model, nullptr);
    if (tmpl == nullptr) {
        return false;
    }

    const llama_chat_message messages[] = {
        {"user", user_text.c_str()},
    };

    std::vector<char> buf(user_text.size() * 4 + 2048);
    int32_t written = llama_chat_apply_template(
        tmpl, messages, 1, /*add_ass=*/true, buf.data(), static_cast<int32_t>(buf.size()));
    if (written > static_cast<int32_t>(buf.size())) {
        buf.resize(static_cast<size_t>(written) + 1);
        written = llama_chat_apply_template(
            tmpl, messages, 1, true, buf.data(), static_cast<int32_t>(buf.size()));
    }
    if (written < 0) {
        return false;
    }

    out->assign(buf.data(), static_cast<size_t>(written));
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <model.gguf> [prompt] [n_tokens] [--chat]\n"
                     "       %s <model.gguf> --info\n"
                     "\n"
                     "  --chat  wrap the prompt in the model's chat template\n"
                     "  --info  print the header summary the app's model detail "
                     "screen shows\n",
                     argv[0], argv[0]);
        return 2;
    }

    const std::string model_path = argv[1];

    // Same code path the app uses, so what the detail screen will show can be
    // checked here rather than only by installing on a phone.
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--info") == 0) {
            const std::string summary = sruti::gguf_summary(model_path.c_str());
            if (summary.empty()) {
                std::fprintf(stderr, "not a readable GGUF: %s\n", model_path.c_str());
                return 1;
            }
            std::fputs(summary.c_str(), stdout);
            return 0;
        }
    }

    std::string prompt = argc > 2 ? argv[2] : "The capital of France is";
    const int n_predict = argc > 3 ? std::atoi(argv[3]) : 48;

    bool chat_mode = false;
    std::string grammar_path;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--chat") == 0) {
            chat_mode = true;
        } else if (std::strcmp(argv[i], "--grammar") == 0 && i + 1 < argc) {
            grammar_path = argv[++i];
        }
    }

    // Reading the grammar before the model loads means a syntax error is reported
    // in a second rather than after a multi-second load.
    std::string grammar_text;
    if (!grammar_path.empty()) {
        FILE * gf = std::fopen(grammar_path.c_str(), "rb");
        if (gf == nullptr) {
            std::fprintf(stderr, "cannot open grammar %s\n", grammar_path.c_str());
            return 1;
        }
        char buf[4096];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof(buf), gf)) > 0) {
            grammar_text.append(buf, n);
        }
        std::fclose(gf);
    }

    llama_log_set(quiet_log, nullptr);
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load %s\n", model_path.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (chat_mode) {
        std::string formatted;
        if (!apply_chat_template(model, prompt, &formatted)) {
            std::fprintf(stderr,
                         "error: model declares no usable chat template\n");
            llama_model_free(model);
            return 1;
        }
        std::printf("chat template applied:\n---\n%s\n---\n\n", formatted.c_str());
        prompt = formatted;
    }

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

    if (!grammar_text.empty()) {
        // Must precede the selection sampler: the grammar masks tokens that would
        // break the structure, and the sampler then chooses among what is left.
        llama_sampler * grammar =
            llama_sampler_init_grammar(vocab, grammar_text.c_str(), "root");
        if (grammar == nullptr) {
            std::fprintf(stderr, "error: grammar failed to parse\n");
            llama_sampler_free(smpl);
            return 1;
        }
        llama_sampler_chain_add(smpl, grammar);
        std::printf("grammar: %s (parsed)\n", grammar_path.c_str());
    }

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
        // No explicit accept: llama_sampler_sample already accepted this token
        // into the chain. Accepting again advances stateful samplers twice --
        // which crashes a grammar and silently double-counts repetitions.

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
