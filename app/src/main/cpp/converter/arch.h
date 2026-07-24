#pragma once

// Architecture detection, hyperparameter parsing, and HF -> GGUF tensor mapping.
//
// This file deliberately has no ggml dependency so it can be tested on the host.
// Metadata is produced as a list of typed key/value pairs which the GGUF writer
// then applies.
//
// Two traps live here, both of which produce a model that loads cleanly and
// generates garbage rather than failing:
//
//  1. Llama-family checkpoints store Q and K projections in an interleaved
//     layout that must be un-permuted for GGUF. Qwen and Gemma do not.
//  2. `post_attention_layernorm` maps to `ffn_norm` for Llama and Qwen, but to
//     `post_attention_norm` for Gemma, which carries four norms per block rather
//     than two.

#include <cstdint>
#include <string>
#include <vector>

namespace sruti {

enum class Arch {
    Llama,
    Qwen2,
    Qwen3,
    Gemma2,
    Gemma3,
    Phi3,
    Unknown,
};

/// Per-tensor transform applied between reading and quantizing.
enum class WeightTransform {
    None,
    /// Llama-family RoPE de-interleaving, grouped by head count.
    PermuteQuery,
    /// Same, grouped by KV head count (differs under GQA).
    PermuteKey,
    /// Gemma stores RMSNorm weights as w but implements (1 + w).
    AddOne,
};

/// One output tensor and where it comes from.
struct TensorPlan {
    std::string gguf_name;
    std::string source_name;
    WeightTransform transform = WeightTransform::None;
};

/// A typed GGUF metadata entry.
struct KvPair {
    enum class Type { U32, I32, F32, Bool, Str };

    std::string key;
    Type type = Type::U32;

    uint32_t u32 = 0;
    int32_t i32 = 0;
    float f32 = 0.0f;
    bool b = false;
    std::string str;

    static KvPair of_u32(std::string key, uint32_t v);
    static KvPair of_i32(std::string key, int32_t v);
    static KvPair of_f32(std::string key, float v);
    static KvPair of_bool(std::string key, bool v);
    static KvPair of_str(std::string key, std::string v);
};

/// Hyperparameters read from config.json, normalised across architectures.
struct HParams {
    std::string hf_arch;
    Arch arch = Arch::Unknown;

    int64_t vocab_size = 0;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t block_count = 0;
    int64_t head_count = 0;
    int64_t head_count_kv = 0;
    /// Per-head dimension. Explicit in config for Qwen3 and Gemma3; derived otherwise.
    int64_t head_dim = 0;
    /// Whether head_dim came from config.json rather than being derived.
    bool head_dim_explicit = false;
    int64_t context_length = 0;

    float rms_norm_eps = 1e-5f;
    float rope_freq_base = 10000.0f;

    bool tie_word_embeddings = false;

    // Optional, architecture-specific. Zero means "absent".
    int64_t sliding_window = 0;
    /// Qwen2.5 declares a sliding_window it does not actually use. Emitting the
    /// key regardless would make llama.cpp apply sliding-window attention the
    /// model was never trained with.
    bool use_sliding_window = false;
    float rope_scaling_factor = 0.0f;
    std::string rope_scaling_type;
    float attn_logit_softcapping = 0.0f;
    float final_logit_softcapping = 0.0f;
    float query_pre_attn_scalar = 0.0f;

    /// True when Q/K need de-interleaving on the way to GGUF.
    bool undo_permute() const {
        return arch == Arch::Llama;
    }

    /// Gemma's four-norms-per-block layout, and the (1 + w) norm convention.
    bool is_gemma() const {
        return arch == Arch::Gemma2 || arch == Arch::Gemma3;
    }
};

/// What can be learned about a checkpoint from config.json alone.
///
/// config.json is a few kilobytes while the weights are gigabytes, so fetching it
/// first turns "download 2.5 GB, then discover the architecture is unsupported"
/// into an instant answer.
struct CheckpointInfo {
    bool supported = false;
    /// Why it is unsupported. Empty when `supported`.
    std::string error;

    std::string hf_arch;
    /// GGUF architecture name, e.g. "llama".
    std::string arch;
    /// Human-readable, e.g. "Qwen3".
    std::string display_name;

    int64_t block_count = 0;
    int64_t hidden_size = 0;
    int64_t head_count = 0;
    int64_t head_count_kv = 0;
    int64_t context_length = 0;
    int64_t vocab_size = 0;

    /// Estimated parameter count, derived from the shapes config.json implies.
    int64_t parameter_count = 0;
};

/// Estimates a model's parameter count from its hyperparameters.
///
/// Counts the tensors that dominate — embeddings, attention projections and the
/// feed-forward block — and ignores norms and biases, which together are well
/// under a tenth of a percent.
int64_t estimate_parameter_count(const HParams & hp);

/// Reads config.json and reports whether this checkpoint can be converted.
CheckpointInfo inspect_config(const std::string & config_json);

/// Maps an HF `architectures[0]` string to a supported architecture.
Arch arch_from_hf_name(const std::string & hf_arch);

/// The `general.architecture` value llama.cpp expects.
const char * gguf_arch_name(Arch arch);

/// Human-readable name for diagnostics.
const char * arch_display_name(Arch arch);

/// Parses config.json. Returns false and sets `err` when unsupported or malformed.
bool parse_hparams(const std::string & config_json, HParams * out, std::string * err);

/// Builds the full HF -> GGUF tensor plan.
///
/// `available` is every tensor name present in the checkpoint; entries that the
/// architecture does not expect are ignored, and required entries that are absent
/// produce an error. Returns false and sets `err` on failure.
bool build_tensor_plan(
    const HParams & hp,
    const std::vector<std::string> & available,
    std::vector<TensorPlan> * out,
    std::string * err);

/// Builds the architecture metadata block. Tokenizer and file-type keys are added
/// separately by the writer.
std::vector<KvPair> build_metadata(const HParams & hp, const std::string & model_name);

}  // namespace sruti
