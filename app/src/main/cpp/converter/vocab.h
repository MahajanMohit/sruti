#pragma once

// tokenizer.json -> GGUF vocabulary.
//
// The fiddliest part of conversion, and the part whose failures are quietest: a
// vocabulary that is subtly wrong still loads, still generates text, and simply
// tokenizes differently from the reference implementation.
//
// Two details carry most of the risk:
//
//  1. `tokenizer.ggml.pre` selects which pre-tokenizer regex set llama.cpp uses.
//     It is identified by matching the actual regex out of tokenizer.json, not by
//     guessing from the model name. Llama 3 and Qwen2 differ only by `\p{N}{1,3}`
//     versus `\p{N}` — a single quantifier that changes how digits are split.
//
//  2. Token id space must be dense. `vocab_size` from config.json can exceed the
//     number of tokens the tokenizer declares, and llama.cpp indexes the token
//     array directly, so holes must be filled rather than skipped.

#include <cstdint>
#include <string>
#include <vector>

namespace sruti {

/// GGUF token type codes, matching llama.cpp's `llama_token_type`.
enum class TokenType : int32_t {
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
    Unused = 5,
    Byte = 6,
};

struct Vocab {
    /// "gpt2" for byte-level BPE. SentencePiece checkpoints are rejected.
    std::string model = "gpt2";
    /// Pre-tokenizer identifier, e.g. "llama-bpe", "qwen2", "gpt-2", "default".
    std::string pre = "default";

    std::vector<std::string> tokens;
    std::vector<int32_t> token_types;
    std::vector<std::string> merges;

    int32_t bos_id = -1;
    int32_t eos_id = -1;
    int32_t unk_id = -1;
    int32_t pad_id = -1;
    int32_t eot_id = -1;

    bool add_bos = false;
    bool add_eos = false;
    bool has_add_bos = false;
    bool has_add_eos = false;

    /// Whether a leading space is prepended before tokenizing.
    bool add_space_prefix = false;
    bool has_add_space_prefix = false;

    std::string chat_template;

    // Sampler defaults from generation_config.json. Negative means unset.
    float sampling_temp = -1.0f;
    float sampling_top_p = -1.0f;
    int32_t sampling_top_k = -1;

    /// Non-fatal problems worth surfacing to the user, e.g. an unrecognised
    /// pre-tokenizer that fell back to "default".
    std::vector<std::string> warnings;

    size_t size() const { return tokens.size(); }
};

/// Identifies the pre-tokenizer from a regex found in tokenizer.json.
/// Returns an empty string when unrecognised.
std::string pre_tokenizer_from_regex(const std::string & regex);

/// Whether a token should be treated as a control token despite not being flagged
/// `special` in tokenizer.json.
///
/// Several families mark tokens that are plainly control tokens as non-special —
/// Qwen's `<|fim_prefix|>` and friends among them. Trusting the flag alone lets
/// those tokens be rendered as visible text in generated output.
bool token_looks_special(const std::string & token);

/// Identifies the pre-tokenizer from its structural signature, for families whose
/// tokenizer.json carries no regex at all.
///
/// `signature` is a "+"-joined list of component types with their distinguishing
/// parameters, e.g. "Digits(individual)+ByteLevel". Returns an empty string when
/// unrecognised.
std::string pre_tokenizer_from_signature(const std::string & signature);

/// Builds the GGUF vocabulary.
///
/// `tokenizer_config_json` and `generation_config_json` may be empty. `vocab_size`
/// comes from config.json and is used to pad the token array; pass 0 to size the
/// vocabulary purely from the tokenizer.
///
/// Returns false and sets `err` on failure.
bool build_vocab(
    const std::string & tokenizer_json,
    const std::string & tokenizer_config_json,
    const std::string & generation_config_json,
    int64_t vocab_size,
    Vocab * out,
    std::string * err);

}  // namespace sruti
