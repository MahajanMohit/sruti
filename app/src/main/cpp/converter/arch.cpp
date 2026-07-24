#include "arch.h"

#include <algorithm>
#include <cstdio>
#include <initializer_list>
#include <unordered_set>

#include <nlohmann/json.hpp>

namespace sruti {

namespace {

using json = nlohmann::json;

std::string block_name(const char * tmpl, int64_t bid) {
    // Templates are of the form "blk.%lld.attn_q.weight".
    char buf[128];
    std::snprintf(buf, sizeof(buf), tmpl, static_cast<long long>(bid));
    return buf;
}

std::string layer_prefix(int64_t bid) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "model.layers.%lld.", static_cast<long long>(bid));
    return buf;
}

/// Reads the first present key from `names`, or returns `fallback`.
template <typename T>
T get_or(const json & j, std::initializer_list<const char *> names, T fallback) {
    for (const char * name : names) {
        const auto it = j.find(name);
        if (it != j.end() && !it->is_null()) {
            try {
                return it->get<T>();
            } catch (const std::exception &) {
                return fallback;
            }
        }
    }
    return fallback;
}

}  // namespace

// --- KvPair -----------------------------------------------------------------

KvPair KvPair::of_u32(std::string key, uint32_t v) {
    KvPair kv; kv.key = std::move(key); kv.type = Type::U32; kv.u32 = v; return kv;
}
KvPair KvPair::of_i32(std::string key, int32_t v) {
    KvPair kv; kv.key = std::move(key); kv.type = Type::I32; kv.i32 = v; return kv;
}
KvPair KvPair::of_f32(std::string key, float v) {
    KvPair kv; kv.key = std::move(key); kv.type = Type::F32; kv.f32 = v; return kv;
}
KvPair KvPair::of_bool(std::string key, bool v) {
    KvPair kv; kv.key = std::move(key); kv.type = Type::Bool; kv.b = v; return kv;
}
KvPair KvPair::of_str(std::string key, std::string v) {
    KvPair kv; kv.key = std::move(key); kv.type = Type::Str; kv.str = std::move(v); return kv;
}

// --- architecture identity --------------------------------------------------

Arch arch_from_hf_name(const std::string & hf_arch) {
    // Llama-family checkpoints that share the interleaved Q/K layout.
    if (hf_arch == "LlamaForCausalLM" || hf_arch == "LLaMAForCausalLM" ||
        hf_arch == "MistralForCausalLM") {
        return Arch::Llama;
    }
    if (hf_arch == "Qwen2ForCausalLM") return Arch::Qwen2;
    if (hf_arch == "Qwen3ForCausalLM") return Arch::Qwen3;
    if (hf_arch == "Gemma2ForCausalLM") return Arch::Gemma2;
    if (hf_arch == "Gemma3ForCausalLM") return Arch::Gemma3;
    if (hf_arch == "Phi3ForCausalLM") return Arch::Phi3;
    return Arch::Unknown;
}

const char * gguf_arch_name(Arch arch) {
    switch (arch) {
        case Arch::Llama:  return "llama";
        case Arch::Qwen2:  return "qwen2";
        case Arch::Qwen3:  return "qwen3";
        case Arch::Gemma2: return "gemma2";
        case Arch::Gemma3: return "gemma3";
        case Arch::Phi3:   return "phi3";
        default:           return "unknown";
    }
}

const char * arch_display_name(Arch arch) {
    switch (arch) {
        case Arch::Llama:  return "Llama";
        case Arch::Qwen2:  return "Qwen2";
        case Arch::Qwen3:  return "Qwen3";
        case Arch::Gemma2: return "Gemma 2";
        case Arch::Gemma3: return "Gemma 3";
        case Arch::Phi3:   return "Phi-3";
        default:           return "unknown";
    }
}

// --- config.json ------------------------------------------------------------

bool parse_hparams(const std::string & config_json, HParams * out, std::string * err) {
    json cfg;
    try {
        cfg = json::parse(config_json);
    } catch (const std::exception & e) {
        if (err) *err = std::string("config.json is not valid JSON: ") + e.what();
        return false;
    }
    if (!cfg.is_object()) {
        if (err) *err = "config.json is not a JSON object";
        return false;
    }

    // Multimodal checkpoints nest the language model under text_config; the text
    // half is all this converter handles.
    const auto text_cfg_it = cfg.find("text_config");
    if (text_cfg_it != cfg.end() && text_cfg_it->is_object()) {
        for (const auto & [k, v] : text_cfg_it->items()) {
            cfg[k] = v;
        }
    }

    HParams hp;

    const auto archs_it = cfg.find("architectures");
    if (archs_it == cfg.end() || !archs_it->is_array() || archs_it->empty()) {
        if (err) *err = "config.json has no architectures[] entry";
        return false;
    }
    hp.hf_arch = (*archs_it)[0].get<std::string>();
    hp.arch = arch_from_hf_name(hp.hf_arch);

    if (hp.arch == Arch::Unknown) {
        if (err) {
            *err = "unsupported architecture '" + hp.hf_arch +
                   "'. Supported: LlamaForCausalLM, MistralForCausalLM, "
                   "Qwen2ForCausalLM, Qwen3ForCausalLM, Gemma2ForCausalLM, "
                   "Gemma3ForCausalLM, Phi3ForCausalLM";
        }
        return false;
    }

    hp.vocab_size        = get_or<int64_t>(cfg, {"vocab_size"}, 0);
    hp.hidden_size       = get_or<int64_t>(cfg, {"hidden_size", "n_embd"}, 0);
    hp.intermediate_size = get_or<int64_t>(cfg, {"intermediate_size", "n_inner"}, 0);
    hp.block_count       = get_or<int64_t>(cfg, {"num_hidden_layers", "n_layer"}, 0);
    hp.head_count        = get_or<int64_t>(cfg, {"num_attention_heads", "n_head"}, 0);
    hp.context_length    = get_or<int64_t>(cfg, {"max_position_embeddings", "n_positions"}, 0);

    // Absent num_key_value_heads means multi-head attention, i.e. kv == q.
    hp.head_count_kv = get_or<int64_t>(cfg, {"num_key_value_heads"}, hp.head_count);

    hp.rms_norm_eps = get_or<float>(cfg, {"rms_norm_eps", "layer_norm_eps"}, 1e-5f);
    hp.rope_freq_base = get_or<float>(cfg, {"rope_theta"}, 10000.0f);
    hp.tie_word_embeddings = get_or<bool>(cfg, {"tie_word_embeddings"}, false);

    // Qwen3 and Gemma3 state head_dim explicitly and it is NOT hidden/heads for
    // them; deriving it would silently produce a wrong rope dimension count.
    hp.head_dim = get_or<int64_t>(cfg, {"head_dim"}, 0);
    hp.head_dim_explicit = hp.head_dim > 0;
    if (hp.head_dim == 0) {
        if (hp.head_count == 0) {
            if (err) *err = "config.json has neither head_dim nor num_attention_heads";
            return false;
        }
        hp.head_dim = hp.hidden_size / hp.head_count;
    }

    hp.sliding_window = get_or<int64_t>(cfg, {"sliding_window"}, 0);
    // Absent use_sliding_window means the window applies if declared; Qwen2.5 sets
    // it to false explicitly while still declaring a window.
    hp.use_sliding_window = get_or<bool>(cfg, {"use_sliding_window"}, true);
    hp.attn_logit_softcapping = get_or<float>(cfg, {"attn_logit_softcapping"}, 0.0f);
    hp.final_logit_softcapping = get_or<float>(cfg, {"final_logit_softcapping"}, 0.0f);
    hp.query_pre_attn_scalar = get_or<float>(cfg, {"query_pre_attn_scalar"}, 0.0f);

    const auto rope_it = cfg.find("rope_scaling");
    if (rope_it != cfg.end() && rope_it->is_object()) {
        hp.rope_scaling_type = get_or<std::string>(*rope_it, {"rope_type", "type"}, "");
        hp.rope_scaling_factor = get_or<float>(*rope_it, {"factor"}, 0.0f);
    }

    // Validate the fields that every downstream step depends on.
    struct { const char * name; int64_t value; } required[] = {
        {"hidden_size", hp.hidden_size},
        {"intermediate_size", hp.intermediate_size},
        {"num_hidden_layers", hp.block_count},
        {"num_attention_heads", hp.head_count},
    };
    for (const auto & field : required) {
        if (field.value <= 0) {
            if (err) *err = std::string("config.json is missing or has invalid ") + field.name;
            return false;
        }
    }

    *out = std::move(hp);
    return true;
}

// --- inspection -------------------------------------------------------------

int64_t estimate_parameter_count(const HParams & hp) {
    const int64_t q_dim = hp.head_count * hp.head_dim;
    const int64_t kv_dim = hp.head_count_kv * hp.head_dim;

    // Per block: Q, K, V and the output projection, then gate, up and down.
    const int64_t attn = hp.hidden_size * (q_dim + 2 * kv_dim + q_dim);
    const int64_t ffn = 3 * hp.hidden_size * hp.intermediate_size;
    const int64_t per_block = attn + ffn;

    // The embedding table counts twice unless the output head is tied to it.
    const int64_t embeddings =
        hp.vocab_size * hp.hidden_size * (hp.tie_word_embeddings ? 1 : 2);

    return embeddings + hp.block_count * per_block;
}

CheckpointInfo inspect_config(const std::string & config_json) {
    CheckpointInfo info;

    HParams hp;
    std::string err;
    if (!parse_hparams(config_json, &hp, &err)) {
        info.supported = false;
        info.error = err;

        // Even when unsupported, report the architecture so the UI can say which
        // one was rejected rather than just refusing.
        try {
            const json cfg = json::parse(config_json);
            const auto archs = cfg.find("architectures");
            if (archs != cfg.end() && archs->is_array() && !archs->empty()) {
                info.hf_arch = (*archs)[0].get<std::string>();
            }
        } catch (const std::exception &) {
            // Malformed JSON; the parse error already explains it.
        }
        return info;
    }

    info.supported = true;
    info.hf_arch = hp.hf_arch;
    info.arch = gguf_arch_name(hp.arch);
    info.display_name = arch_display_name(hp.arch);
    info.block_count = hp.block_count;
    info.hidden_size = hp.hidden_size;
    info.head_count = hp.head_count;
    info.head_count_kv = hp.head_count_kv;
    info.context_length = hp.context_length;
    info.vocab_size = hp.vocab_size;
    info.parameter_count = estimate_parameter_count(hp);
    return info;
}

// --- tensor plan ------------------------------------------------------------

bool build_tensor_plan(
    const HParams & hp,
    const std::vector<std::string> & available,
    std::vector<TensorPlan> * out,
    std::string * err) {

    const std::unordered_set<std::string> have(available.begin(), available.end());
    std::vector<TensorPlan> plan;

    const auto has = [&](const std::string & name) { return have.count(name) != 0; };

    const auto require = [&](const std::string & source,
                             const std::string & gguf,
                             WeightTransform tf) -> bool {
        if (!has(source)) {
            if (err) *err = "checkpoint is missing required tensor '" + source + "'";
            return false;
        }
        plan.push_back({gguf, source, tf});
        return true;
    };

    const auto optional = [&](const std::string & source,
                              const std::string & gguf,
                              WeightTransform tf) {
        if (has(source)) {
            plan.push_back({gguf, source, tf});
        }
    };

    // --- embeddings and output ---------------------------------------------
    if (!require("model.embed_tokens.weight", "token_embd.weight", WeightTransform::None)) {
        return false;
    }

    const WeightTransform norm_tf =
        hp.is_gemma() ? WeightTransform::AddOne : WeightTransform::None;

    if (!require("model.norm.weight", "output_norm.weight", norm_tf)) {
        return false;
    }

    // Gemma ties output to the embedding table and ships an lm_head that must be
    // ignored; including it produces a file llama.cpp rejects.
    if (!hp.is_gemma() && !hp.tie_word_embeddings) {
        optional("lm_head.weight", "output.weight", WeightTransform::None);
    }

    // --- blocks -------------------------------------------------------------
    for (int64_t bid = 0; bid < hp.block_count; ++bid) {
        const std::string p = layer_prefix(bid);

        if (!require(p + "input_layernorm.weight",
                     block_name("blk.%lld.attn_norm.weight", bid), norm_tf)) {
            return false;
        }

        // Attention projections. Phi-3 ships them fused; llama.cpp splits attn_qkv
        // internally, so no slicing is needed here.
        if (hp.arch == Arch::Phi3) {
            if (!require(p + "self_attn.qkv_proj.weight",
                         block_name("blk.%lld.attn_qkv.weight", bid),
                         WeightTransform::None)) {
                return false;
            }
        } else {
            if (!require(p + "self_attn.q_proj.weight",
                         block_name("blk.%lld.attn_q.weight", bid),
                         hp.undo_permute() ? WeightTransform::PermuteQuery
                                           : WeightTransform::None)) {
                return false;
            }
            if (!require(p + "self_attn.k_proj.weight",
                         block_name("blk.%lld.attn_k.weight", bid),
                         hp.undo_permute() ? WeightTransform::PermuteKey
                                           : WeightTransform::None)) {
                return false;
            }
            if (!require(p + "self_attn.v_proj.weight",
                         block_name("blk.%lld.attn_v.weight", bid),
                         WeightTransform::None)) {
                return false;
            }

            // Qwen2 carries QKV biases; nothing else in the supported set does.
            // Biases are not permuted: the interleaving applies to weight rows.
            optional(p + "self_attn.q_proj.bias",
                     block_name("blk.%lld.attn_q.bias", bid), WeightTransform::None);
            optional(p + "self_attn.k_proj.bias",
                     block_name("blk.%lld.attn_k.bias", bid), WeightTransform::None);
            optional(p + "self_attn.v_proj.bias",
                     block_name("blk.%lld.attn_v.bias", bid), WeightTransform::None);
        }

        if (!require(p + "self_attn.o_proj.weight",
                     block_name("blk.%lld.attn_output.weight", bid),
                     WeightTransform::None)) {
            return false;
        }

        // Qwen3 and Gemma3 normalise Q and K per head.
        optional(p + "self_attn.q_norm.weight",
                 block_name("blk.%lld.attn_q_norm.weight", bid), norm_tf);
        optional(p + "self_attn.k_norm.weight",
                 block_name("blk.%lld.attn_k_norm.weight", bid), norm_tf);

        // The trap: the same HF name means different things per architecture.
        // Gemma has four norms per block; Llama and Qwen have two.
        if (hp.is_gemma()) {
            if (!require(p + "post_attention_layernorm.weight",
                         block_name("blk.%lld.post_attention_norm.weight", bid), norm_tf)) {
                return false;
            }
            if (!require(p + "pre_feedforward_layernorm.weight",
                         block_name("blk.%lld.ffn_norm.weight", bid), norm_tf)) {
                return false;
            }
            if (!require(p + "post_feedforward_layernorm.weight",
                         block_name("blk.%lld.post_ffw_norm.weight", bid), norm_tf)) {
                return false;
            }
        } else {
            if (!require(p + "post_attention_layernorm.weight",
                         block_name("blk.%lld.ffn_norm.weight", bid), norm_tf)) {
                return false;
            }
        }

        // Feed-forward. Phi-3 fuses gate and up into gate_up_proj, which llama.cpp
        // maps onto ffn_up and splits internally.
        if (hp.arch == Arch::Phi3) {
            if (!require(p + "mlp.gate_up_proj.weight",
                         block_name("blk.%lld.ffn_up.weight", bid), WeightTransform::None)) {
                return false;
            }
        } else {
            if (!require(p + "mlp.gate_proj.weight",
                         block_name("blk.%lld.ffn_gate.weight", bid), WeightTransform::None)) {
                return false;
            }
            if (!require(p + "mlp.up_proj.weight",
                         block_name("blk.%lld.ffn_up.weight", bid), WeightTransform::None)) {
                return false;
            }
        }

        if (!require(p + "mlp.down_proj.weight",
                     block_name("blk.%lld.ffn_down.weight", bid), WeightTransform::None)) {
            return false;
        }
    }

    *out = std::move(plan);
    return true;
}

// --- metadata ---------------------------------------------------------------

std::vector<KvPair> build_metadata(const HParams & hp, const std::string & model_name) {
    const std::string arch = gguf_arch_name(hp.arch);
    const auto key = [&arch](const char * suffix) { return arch + "." + suffix; };

    std::vector<KvPair> kv;

    kv.push_back(KvPair::of_str("general.architecture", arch));
    kv.push_back(KvPair::of_str("general.type", "model"));
    if (!model_name.empty()) {
        kv.push_back(KvPair::of_str("general.name", model_name));
    }

    kv.push_back(KvPair::of_u32(key("context_length"),
                                static_cast<uint32_t>(hp.context_length)));
    kv.push_back(KvPair::of_u32(key("embedding_length"),
                                static_cast<uint32_t>(hp.hidden_size)));
    kv.push_back(KvPair::of_u32(key("block_count"),
                                static_cast<uint32_t>(hp.block_count)));
    kv.push_back(KvPair::of_u32(key("feed_forward_length"),
                                static_cast<uint32_t>(hp.intermediate_size)));

    kv.push_back(KvPair::of_u32(key("attention.head_count"),
                                static_cast<uint32_t>(hp.head_count)));
    kv.push_back(KvPair::of_u32(key("attention.head_count_kv"),
                                static_cast<uint32_t>(hp.head_count_kv)));
    kv.push_back(KvPair::of_f32(key("attention.layer_norm_rms_epsilon"), hp.rms_norm_eps));

    kv.push_back(KvPair::of_u32(key("rope.dimension_count"),
                                static_cast<uint32_t>(hp.head_dim)));
    kv.push_back(KvPair::of_f32(key("rope.freq_base"), hp.rope_freq_base));

    if (hp.vocab_size > 0) {
        kv.push_back(KvPair::of_u32(key("vocab_size"), static_cast<uint32_t>(hp.vocab_size)));
    }

    // Stated when config.json declares head_dim, and for Llama where the reference
    // emits them unconditionally. Where they are omitted, llama.cpp derives the
    // same value from embedding length and head count.
    if (hp.head_dim_explicit || hp.arch == Arch::Llama) {
        kv.push_back(KvPair::of_u32(key("attention.key_length"),
                                    static_cast<uint32_t>(hp.head_dim)));
        kv.push_back(KvPair::of_u32(key("attention.value_length"),
                                    static_cast<uint32_t>(hp.head_dim)));
    }

    // Only when the model actually uses it. Qwen2.5 declares a 32k window with
    // use_sliding_window false; emitting the key anyway would switch on
    // sliding-window attention that the model was never trained with.
    if (hp.sliding_window > 0 && hp.use_sliding_window) {
        kv.push_back(KvPair::of_u32(key("attention.sliding_window"),
                                    static_cast<uint32_t>(hp.sliding_window)));
    }
    if (hp.attn_logit_softcapping > 0.0f) {
        kv.push_back(KvPair::of_f32(key("attn_logit_softcapping"), hp.attn_logit_softcapping));
    }
    if (hp.final_logit_softcapping > 0.0f) {
        kv.push_back(KvPair::of_f32(key("final_logit_softcapping"),
                                    hp.final_logit_softcapping));
    }
    if (!hp.rope_scaling_type.empty() && hp.rope_scaling_factor > 0.0f) {
        kv.push_back(KvPair::of_str(key("rope.scaling.type"), hp.rope_scaling_type));
        kv.push_back(KvPair::of_f32(key("rope.scaling.factor"), hp.rope_scaling_factor));
    }

    return kv;
}

}  // namespace sruti
