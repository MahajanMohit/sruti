#include "gguf_summary.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "ggml.h"
#include "gguf.h"

namespace sruti {

std::string gguf_summary(const char * path) {

    gguf_init_params params = { /*no_alloc=*/true, /*ctx=*/nullptr };
    gguf_context * gguf = gguf_init_from_file(path, params);
    if (gguf == nullptr) {
        return {};
    }

    std::string out;
    auto row = [&out](const std::string & label, const std::string & value) {
        if (value.empty()) {
            return;
        }
        out += label;
        out += '\t';
        out += value;
        out += '\n';
    };

    // Values are looked up by key rather than by index because the interesting
    // ones are namespaced under the architecture, which is itself a value.
    auto find = [gguf](const std::string & key) -> int64_t {
        return gguf_find_key(gguf, key.c_str());
    };

    auto str_of = [gguf, &find](const std::string & key) -> std::string {
        const int64_t id = find(key);
        if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_STRING) {
            return {};
        }
        return gguf_get_val_str(gguf, id);
    };

    // Counts are written as any of the integer widths depending on the writer,
    // so read whichever one is actually there.
    auto int_of = [gguf, &find](const std::string & key) -> std::string {
        const int64_t id = find(key);
        if (id < 0) {
            return {};
        }
        switch (gguf_get_kv_type(gguf, id)) {
            case GGUF_TYPE_UINT8:  return std::to_string(gguf_get_val_u8(gguf, id));
            case GGUF_TYPE_INT8:   return std::to_string(gguf_get_val_i8(gguf, id));
            case GGUF_TYPE_UINT16: return std::to_string(gguf_get_val_u16(gguf, id));
            case GGUF_TYPE_INT16:  return std::to_string(gguf_get_val_i16(gguf, id));
            case GGUF_TYPE_UINT32: return std::to_string(gguf_get_val_u32(gguf, id));
            case GGUF_TYPE_INT32:  return std::to_string(gguf_get_val_i32(gguf, id));
            case GGUF_TYPE_UINT64: return std::to_string(gguf_get_val_u64(gguf, id));
            case GGUF_TYPE_INT64:  return std::to_string(gguf_get_val_i64(gguf, id));
            case GGUF_TYPE_FLOAT32: {
                // RoPE base is stored as a float but is always a round number,
                // and %g renders it as 1e+06, which reads as a parse failure.
                const float v = gguf_get_val_f32(gguf, id);
                char buf[32];
                if (v == static_cast<float>(static_cast<int64_t>(v))) {
                    snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(v));
                } else {
                    snprintf(buf, sizeof(buf), "%g", v);
                }
                return buf;
            }
            default: return {};
        }
    };

    const std::string arch = str_of("general.architecture");

    row("Architecture", arch);
    row("Name", str_of("general.name"));
    row("GGUF version", std::to_string(gguf_get_version(gguf)));
    row("Base model", str_of("general.basename"));

    if (!arch.empty()) {
        row("Layers",           int_of(arch + ".block_count"));
        row("Embedding size",   int_of(arch + ".embedding_length"));
        row("Feed-forward size", int_of(arch + ".feed_forward_length"));
        row("Attention heads",  int_of(arch + ".attention.head_count"));
        row("Key/value heads",  int_of(arch + ".attention.head_count_kv"));
        row("Trained context",  int_of(arch + ".context_length"));
        row("RoPE base",        int_of(arch + ".rope.freq_base"));
    }

    row("Tokenizer",     str_of("tokenizer.ggml.model"));
    row("Pre-tokenizer", str_of("tokenizer.ggml.pre"));

    const int64_t tokens_id = find("tokenizer.ggml.tokens");
    if (tokens_id >= 0 && gguf_get_kv_type(gguf, tokens_id) == GGUF_TYPE_ARRAY) {
        row("Vocabulary", std::to_string(gguf_get_arr_n(gguf, tokens_id)));
    }
    row("BOS token id", int_of("tokenizer.ggml.bos_token_id"));
    row("EOS token id", int_of("tokenizer.ggml.eos_token_id"));

    // Whether a template exists decides whether the model is prompted as an
    // instruct model or as a base model, which is the difference between
    // coherent replies and continued gibberish.
    row("Chat template",
        find("tokenizer.chat_template") >= 0 ? "present" : "none — prompted as a base model");

    // Weight types, most-used first. A "Q4_K_M" file is really a mixture, and
    // the mixture is what actually determines size and quality.
    const int64_t n_tensors = gguf_get_n_tensors(gguf);
    std::vector<std::pair<ggml_type, int64_t>> histogram;
    int64_t total_elements = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const ggml_type type = gguf_get_tensor_type(gguf, i);
        const int64_t * ne = gguf_get_tensor_ne(gguf, i);
        int64_t elements = 1;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            elements *= ne[d];
        }
        total_elements += elements;

        auto it = std::find_if(histogram.begin(), histogram.end(),
                               [type](const auto & e) { return e.first == type; });
        if (it == histogram.end()) {
            histogram.emplace_back(type, 1);
        } else {
            it->second += 1;
        }
    }
    std::sort(histogram.begin(), histogram.end(),
              [](const auto & a, const auto & b) { return a.second > b.second; });

    std::string weights;
    for (const auto & [type, count] : histogram) {
        if (!weights.empty()) {
            weights += ", ";
        }
        weights += std::string(ggml_type_name(type)) + " x" + std::to_string(count);
    }
    row("Weight types", weights);
    row("Tensors", std::to_string(n_tensors));

    // Counted from the tensor shapes rather than read from a key, because the
    // key is optional and several writers omit it.
    char param_text[32];
    if (total_elements >= 1000000000) {
        snprintf(param_text, sizeof(param_text), "%.2f B", total_elements / 1e9);
    } else {
        snprintf(param_text, sizeof(param_text), "%.0f M", total_elements / 1e6);
    }
    row("Parameters", param_text);
    row("Metadata keys", std::to_string(gguf_get_n_kv(gguf)));

    gguf_free(gguf);
    return out;
}

}  // namespace sruti
