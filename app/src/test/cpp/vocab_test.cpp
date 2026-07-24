// Host-side tests for tokenizer.json -> GGUF vocabulary conversion.
//
// Most of these cover failures that produce a working-but-wrong model: a vocab
// with holes, a mis-identified pre-tokenizer, or an EOS token taken from the
// wrong file so generation never stops.

#include "../../main/cpp/converter/vocab.h"

#include <string>
#include <vector>

#include "test_util.h"

using namespace sruti;

namespace {

// The genuine Llama 3 pre-tokenizer regex, as JSON-escaped inside tokenizer.json.
const char * kLlama3RegexJson =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+)";

// Qwen2's, which differs only in `\p{N}` versus `\p{N}{1,3}`.
const char * kQwen2RegexJson =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+)";

std::string tokenizer_json_with(
    const std::string & regex_json,
    const std::string & vocab_body = R"("a":0,"b":1,"c":2)",
    const std::string & added_tokens = "[]",
    const std::string & merges = R"(["a b"])") {

    return std::string(R"({"pre_tokenizer":{"type":"Sequence","pretokenizers":[)") +
           R"({"type":"Split","pattern":{"Regex":")" + regex_json + R"("},"behavior":"Isolated"},)" +
           R"({"type":"ByteLevel","add_prefix_space":false}]},)" +
           R"("added_tokens":)" + added_tokens + "," +
           R"("model":{"type":"BPE","vocab":{)" + vocab_body + R"(},"merges":)" + merges + "}}";
}

// --- pre-tokenizer identification -------------------------------------------

void test_identifies_llama3_pre_tokenizer() {
    Vocab v;
    std::string err;
    ASSERT_TRUE(build_vocab(tokenizer_json_with(kLlama3RegexJson), "", "", 0, &v, &err),
                std::string("built vocab: ") + err);
    ASSERT_EQ(v.pre, std::string("llama-bpe"), "Llama 3 pre-tokenizer identified");
}

void test_identifies_qwen2_pre_tokenizer() {
    Vocab v;
    std::string err;
    ASSERT_TRUE(build_vocab(tokenizer_json_with(kQwen2RegexJson), "", "", 0, &v, &err),
                std::string("built vocab: ") + err);
    ASSERT_EQ(v.pre, std::string("qwen2"), "Qwen2 pre-tokenizer identified");
}

void test_llama3_and_qwen2_are_not_confused() {
    // These two regexes differ by a single quantifier and select different digit
    // grouping. Conflating them changes tokenization of every number.
    ASSERT_TRUE(pre_tokenizer_from_regex(
                    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)") ==
                    "llama-bpe",
                "the {1,3} form is llama-bpe");
    ASSERT_TRUE(pre_tokenizer_from_regex(
                    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)") ==
                    "qwen2",
                "the bare \\p{N} form is qwen2");
}

void test_unknown_pre_tokenizer_falls_back_with_a_warning() {
    Vocab v;
    std::string err;
    ASSERT_TRUE(build_vocab(tokenizer_json_with("some-unknown-regex"), "", "", 0, &v, &err),
                std::string("built vocab: ") + err);
    ASSERT_EQ(v.pre, std::string("default"), "unknown pre-tokenizer falls back");

    bool warned = false;
    for (const auto & w : v.warnings) {
        if (w.find("could not identify the pre-tokenizer") != std::string::npos) warned = true;
    }
    ASSERT_TRUE(warned, "fallback is surfaced rather than silent");
}

// --- vocabulary density ------------------------------------------------------

void test_builds_dense_token_array() {
    Vocab v;
    std::string err;
    ASSERT_TRUE(build_vocab(tokenizer_json_with(kLlama3RegexJson), "", "", 0, &v, &err),
                std::string("built vocab: ") + err);
    ASSERT_EQ(v.size(), size_t(3), "one slot per token");
    ASSERT_EQ(v.tokens[0], std::string("a"), "token 0");
    ASSERT_EQ(v.tokens[2], std::string("c"), "token 2");
    ASSERT_EQ(v.token_types.size(), v.size(), "a type for every token");
}

void test_pads_to_config_vocab_size() {
    // Llama 3.2 declares vocab_size 128256 while the tokenizer lists fewer tokens.
    // llama.cpp indexes the array directly, so the tail must exist.
    Vocab v;
    std::string err;
    ASSERT_TRUE(build_vocab(tokenizer_json_with(kLlama3RegexJson), "", "",
                            /*vocab_size=*/8, &v, &err),
                std::string("built vocab: ") + err);
    ASSERT_EQ(v.size(), size_t(8), "padded up to the configured vocab size");
    ASSERT_TRUE(!v.tokens[7].empty(), "padding slots carry placeholder text");
    ASSERT_EQ(v.token_types[7], int32_t(TokenType::Unused), "padding marked unused");
}

void test_fills_holes_in_the_id_space() {
    // Ids 0 and 5 present, nothing between: every intermediate slot must still be
    // a usable entry.
    Vocab v;
    std::string err;
    ASSERT_TRUE(build_vocab(tokenizer_json_with(kLlama3RegexJson, R"("a":0,"z":5)"),
                            "", "", 0, &v, &err),
                std::string("built vocab: ") + err);
    ASSERT_EQ(v.size(), size_t(6), "array sized from the maximum id");

    bool all_filled = true;
    for (const auto & t : v.tokens) {
        if (t.empty()) all_filled = false;
    }
    ASSERT_TRUE(all_filled, "no empty token slots remain");
    ASSERT_EQ(v.token_types[3], int32_t(TokenType::Unused), "hole marked unused");
    ASSERT_EQ(v.token_types[5], int32_t(TokenType::Normal), "real token still normal");
}

void test_does_not_shrink_below_max_id() {
    // A vocab_size smaller than the largest token id must not truncate the array.
    Vocab v;
    std::string err;
    ASSERT_TRUE(build_vocab(tokenizer_json_with(kLlama3RegexJson, R"("a":0,"z":9)"),
                            "", "", /*vocab_size=*/4, &v, &err),
                std::string("built vocab: ") + err);
    ASSERT_EQ(v.size(), size_t(10), "array covers the highest declared id");
}

// --- token types -------------------------------------------------------------

void test_special_added_tokens_become_control_tokens() {
    const std::string added =
        R"([{"id":3,"content":"<|eot_id|>","special":true},)"
        R"({"id":4,"content":"<custom>","special":false}])";

    Vocab v;
    std::string err;
    ASSERT_TRUE(build_vocab(tokenizer_json_with(kLlama3RegexJson, R"("a":0,"b":1,"c":2)", added),
                            "", "", 0, &v, &err),
                std::string("built vocab: ") + err);

    ASSERT_EQ(v.tokens[3], std::string("<|eot_id|>"), "added token text stored");
    ASSERT_EQ(v.token_types[3], int32_t(TokenType::Control),
              "special added token is a control token");
    ASSERT_EQ(v.token_types[4], int32_t(TokenType::UserDefined),
              "non-special added token is user defined");
    ASSERT_EQ(v.token_types[0], int32_t(TokenType::Normal), "ordinary tokens stay normal");
}

// --- merges ------------------------------------------------------------------

void test_accepts_both_merge_encodings() {
    Vocab legacy, current;
    std::string err;

    ASSERT_TRUE(build_vocab(tokenizer_json_with(kLlama3RegexJson, R"("a":0,"b":1)", "[]",
                                                R"(["a b","b a"])"),
                            "", "", 0, &legacy, &err),
                std::string("legacy merges: ") + err);
    ASSERT_EQ(legacy.merges.size(), size_t(2), "legacy string merges read");
    ASSERT_EQ(legacy.merges[0], std::string("a b"), "legacy merge text");

    ASSERT_TRUE(build_vocab(tokenizer_json_with(kLlama3RegexJson, R"("a":0,"b":1)", "[]",
                                                R"([["a","b"],["b","a"]])"),
                            "", "", 0, &current, &err),
                std::string("pair merges: ") + err);
    ASSERT_EQ(current.merges.size(), size_t(2), "pair merges read");
    ASSERT_EQ(current.merges[0], std::string("a b"), "pair merge joined with a space");
}

void test_warns_when_merges_are_absent() {
    Vocab v;
    std::string err;
    build_vocab(tokenizer_json_with(kLlama3RegexJson, R"("a":0)", "[]", "[]"), "", "", 0, &v, &err);

    bool warned = false;
    for (const auto & w : v.warnings) {
        if (w.find("no BPE merges") != std::string::npos) warned = true;
    }
    ASSERT_TRUE(warned, "missing merges surfaced");
}

// --- special token ids -------------------------------------------------------

void test_reads_special_tokens_from_tokenizer_config() {
    const std::string added =
        R"([{"id":1,"content":"<s>","special":true},)"
        R"({"id":2,"content":"</s>","special":true}])";
    const std::string cfg =
        R"({"bos_token":"<s>","eos_token":"</s>","add_bos_token":true,"add_eos_token":false})";

    Vocab v;
    std::string err;
    ASSERT_TRUE(build_vocab(tokenizer_json_with(kLlama3RegexJson, R"("a":0)", added),
                            cfg, "", 0, &v, &err),
                std::string("built vocab: ") + err);

    ASSERT_EQ(v.bos_id, int32_t(1), "BOS id resolved");
    ASSERT_EQ(v.eos_id, int32_t(2), "EOS id resolved");
    ASSERT_TRUE(v.has_add_bos && v.add_bos, "add_bos parsed");
    ASSERT_TRUE(v.has_add_eos && !v.add_eos, "add_eos parsed");
}

void test_reads_special_tokens_given_as_objects() {
    const std::string added = R"([{"id":7,"content":"<|end|>","special":true}])";
    const std::string cfg = R"({"eos_token":{"content":"<|end|>","lstrip":false}})";

    Vocab v;
    std::string err;
    ASSERT_TRUE(build_vocab(tokenizer_json_with(kLlama3RegexJson, R"("a":0)", added),
                            cfg, "", 0, &v, &err),
                std::string("built vocab: ") + err);
    ASSERT_EQ(v.eos_id, int32_t(7), "object-form special token resolved");
}

void test_generation_config_overrides_eos() {
    // Instruction-tuned models routinely stop on a turn-end token that differs
    // from the base EOS in tokenizer_config.json. Taking the wrong one means
    // generation runs to the token limit on every turn.
    const std::string added =
        R"([{"id":2,"content":"</s>","special":true},)"
        R"({"id":9,"content":"<|eot_id|>","special":true}])";
    const std::string cfg = R"({"eos_token":"</s>"})";
    const std::string gen = R"({"eos_token_id":[9,2],"bos_token_id":1})";

    Vocab v;
    std::string err;
    ASSERT_TRUE(build_vocab(tokenizer_json_with(kLlama3RegexJson, R"("a":0)", added),
                            cfg, gen, 0, &v, &err),
                std::string("built vocab: ") + err);

    ASSERT_EQ(v.eos_id, int32_t(9), "generation_config EOS wins");
    ASSERT_EQ(v.eot_id, int32_t(2), "second stop token recorded as EOT");
    ASSERT_EQ(v.bos_id, int32_t(1), "generation_config BOS applied");
}

void test_scalar_generation_config_eos() {
    const std::string gen = R"({"eos_token_id":5})";
    Vocab v;
    std::string err;
    ASSERT_TRUE(build_vocab(tokenizer_json_with(kLlama3RegexJson, R"("a":0,"z":9)"),
                            "", gen, 0, &v, &err),
                std::string("built vocab: ") + err);
    ASSERT_EQ(v.eos_id, int32_t(5), "scalar EOS applied");
}

void test_warns_when_no_eos_is_found() {
    Vocab v;
    std::string err;
    build_vocab(tokenizer_json_with(kLlama3RegexJson), "", "", 0, &v, &err);

    bool warned = false;
    for (const auto & w : v.warnings) {
        if (w.find("no EOS token") != std::string::npos) warned = true;
    }
    ASSERT_TRUE(warned, "missing EOS surfaced");
}

// --- chat template -----------------------------------------------------------

void test_reads_chat_template() {
    const std::string cfg = R"({"chat_template":"{{ messages }}"})";
    Vocab v;
    std::string err;
    build_vocab(tokenizer_json_with(kLlama3RegexJson), cfg, "", 0, &v, &err);
    ASSERT_EQ(v.chat_template, std::string("{{ messages }}"), "chat template captured");
}

void test_reads_named_chat_template_list() {
    const std::string cfg =
        R"({"chat_template":[{"name":"tool_use","template":"A"},)"
        R"({"name":"default","template":"B"}]})";
    Vocab v;
    std::string err;
    build_vocab(tokenizer_json_with(kLlama3RegexJson), cfg, "", 0, &v, &err);
    ASSERT_EQ(v.chat_template, std::string("B"), "default template selected from the list");
}

// --- structural pre-tokenizer identification ---------------------------------

void test_identifies_smollm_without_any_regex() {
    // SmolLM2's tokenizer.json carries no Split regex at all — only Digits and
    // ByteLevel — so regex matching cannot see it. Falling back to "default" here
    // changes how every number is tokenized.
    const std::string tok =
        R"({"pre_tokenizer":{"type":"Sequence","pretokenizers":[)"
        R"({"type":"Digits","individual_digits":true},)"
        R"({"type":"ByteLevel","add_prefix_space":false}]},)"
        R"("model":{"type":"BPE","vocab":{"a":0},"merges":["a b"]}})";

    Vocab v;
    std::string err;
    ASSERT_TRUE(build_vocab(tok, "", "", 0, &v, &err), std::string("built vocab: ") + err);
    ASSERT_EQ(v.pre, std::string("smollm"), "SmolLM identified structurally");
}

void test_identifies_plain_bytelevel_as_gpt2() {
    const std::string tok =
        R"({"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false},)"
        R"("model":{"type":"BPE","vocab":{"a":0},"merges":["a b"]}})";

    Vocab v;
    std::string err;
    ASSERT_TRUE(build_vocab(tok, "", "", 0, &v, &err), std::string("built vocab: ") + err);
    ASSERT_EQ(v.pre, std::string("gpt-2"), "bare ByteLevel identified as gpt-2");
}

void test_signature_lookup_is_exact() {
    ASSERT_EQ(pre_tokenizer_from_signature("Digits(individual)+ByteLevel"),
              std::string("smollm"), "known signature resolves");
    ASSERT_TRUE(pre_tokenizer_from_signature("Digits+ByteLevel").empty(),
                "non-individual digits is a different signature");
    ASSERT_TRUE(pre_tokenizer_from_signature("Whitespace").empty(),
                "unknown signature returns empty");
}

// --- control tokens that are not flagged special -----------------------------

void test_recognises_control_tokens_by_shape() {
    // Qwen marks <|fim_prefix|> and friends as special=false. Trusting that flag
    // lets those tokens be rendered as visible text in generated output.
    ASSERT_TRUE(token_looks_special("<|fim_prefix|>"), "<|...|> is a control token");
    ASSERT_TRUE(token_looks_special("<|endoftext|>"), "<|endoftext|> is a control token");
    ASSERT_TRUE(token_looks_special("<unused42>"), "Gemma reserved slots");
    ASSERT_TRUE(token_looks_special("<pad>"), "known literal");
    ASSERT_TRUE(token_looks_special("<mask>"), "known literal");

    ASSERT_TRUE(!token_looks_special("<tool_call>"), "plain angle brackets are not control");
    ASSERT_TRUE(!token_looks_special("hello"), "ordinary text is not control");
    ASSERT_TRUE(!token_looks_special("<|incomplete"), "unterminated marker is not control");
    ASSERT_TRUE(!token_looks_special(""), "empty token is not control");
}

void test_unflagged_control_tokens_become_control() {
    const std::string added =
        R"([{"id":1,"content":"<|fim_prefix|>","special":false},)"
        R"({"id":2,"content":"<tool_call>","special":false}])";

    Vocab v;
    std::string err;
    ASSERT_TRUE(build_vocab(tokenizer_json_with(kQwen2RegexJson, R"("a":0)", added),
                            "", "", 0, &v, &err),
                std::string("built vocab: ") + err);

    ASSERT_EQ(v.token_types[1], int32_t(TokenType::Control),
              "<|fim_prefix|> promoted to control despite special=false");
    ASSERT_EQ(v.token_types[2], int32_t(TokenType::UserDefined),
              "<tool_call> stays user defined");
}

// --- add_bos derivation ------------------------------------------------------

void test_derives_add_bos_from_post_processor() {
    // Llama 3 prepends BOS via a TemplateProcessing post-processor.
    const std::string with_template =
        R"({"post_processor":{"type":"TemplateProcessing",)"
        R"("single":[{"SpecialToken":{"id":"<|begin_of_text|>","type_id":0}},)"
        R"({"Sequence":{"id":"A","type_id":0}}]},)"
        R"("model":{"type":"BPE","vocab":{"a":0},"merges":[]}})";

    Vocab v;
    std::string err;
    ASSERT_TRUE(build_vocab(with_template, "", "", 0, &v, &err),
                std::string("built vocab: ") + err);
    ASSERT_TRUE(v.has_add_bos && v.add_bos, "TemplateProcessing implies add_bos");
}

void test_no_post_processor_means_no_bos() {
    // SmolLM2 has post_processor null; Qwen has a ByteLevel one. Neither adds BOS,
    // and assuming otherwise inserts a spurious token into every prompt.
    const std::string none =
        R"({"post_processor":null,"model":{"type":"BPE","vocab":{"a":0},"merges":[]}})";
    const std::string byte_level =
        R"({"post_processor":{"type":"ByteLevel","add_prefix_space":false},)"
        R"("model":{"type":"BPE","vocab":{"a":0},"merges":[]}})";

    Vocab v;
    std::string err;
    build_vocab(none, "", "", 0, &v, &err);
    ASSERT_TRUE(v.has_add_bos && !v.add_bos, "absent post-processor means no BOS");

    build_vocab(byte_level, "", "", 0, &v, &err);
    ASSERT_TRUE(v.has_add_bos && !v.add_bos, "ByteLevel post-processor means no BOS");
}

void test_explicit_config_overrides_derived_add_bos() {
    const std::string tok =
        R"({"post_processor":null,"model":{"type":"BPE","vocab":{"a":0},"merges":[]}})";
    Vocab v;
    std::string err;
    build_vocab(tok, R"({"add_bos_token":true})", "", 0, &v, &err);
    ASSERT_TRUE(v.add_bos, "explicit tokenizer_config value wins over derivation");
}

void test_reads_add_space_prefix_from_bytelevel() {
    const std::string tok =
        R"({"pre_tokenizer":{"type":"Sequence","pretokenizers":[)"
        R"({"type":"ByteLevel","add_prefix_space":true}]},)"
        R"("model":{"type":"BPE","vocab":{"a":0},"merges":[]}})";
    Vocab v;
    std::string err;
    build_vocab(tok, "", "", 0, &v, &err);
    ASSERT_TRUE(v.has_add_space_prefix && v.add_space_prefix, "add_prefix_space read");
}

// --- sampler defaults --------------------------------------------------------

void test_reads_sampler_defaults_from_generation_config() {
    const std::string gen = R"({"temperature":0.7,"top_p":0.8,"top_k":20})";
    Vocab v;
    std::string err;
    build_vocab(tokenizer_json_with(kQwen2RegexJson), "", gen, 0, &v, &err);

    ASSERT_TRUE(v.sampling_temp > 0.69f && v.sampling_temp < 0.71f, "temperature read");
    ASSERT_TRUE(v.sampling_top_p > 0.79f && v.sampling_top_p < 0.81f, "top_p read");
    ASSERT_EQ(v.sampling_top_k, int32_t(20), "top_k read");
}

void test_sampler_defaults_absent_when_not_declared() {
    Vocab v;
    std::string err;
    build_vocab(tokenizer_json_with(kQwen2RegexJson), "", "", 0, &v, &err);
    ASSERT_TRUE(v.sampling_temp < 0.0f, "temperature unset");
    ASSERT_TRUE(v.sampling_top_k < 0, "top_k unset");
}

// --- rejections --------------------------------------------------------------

void test_rejects_sentencepiece_with_an_actionable_message() {
    const std::string spm =
        R"({"model":{"type":"Unigram","vocab":[["a",0.0]]}})";
    Vocab v;
    std::string err;
    ASSERT_TRUE(!build_vocab(spm, "", "", 0, &v, &err), "SentencePiece rejected");
    ASSERT_TRUE(err.find("SentencePiece") != std::string::npos,
                "error explains which format is unsupported");
}

void test_rejects_malformed_tokenizer_json() {
    Vocab v;
    std::string err;
    ASSERT_TRUE(!build_vocab("{not json", "", "", 0, &v, &err), "malformed JSON rejected");
    ASSERT_TRUE(!build_vocab("{}", "", "", 0, &v, &err), "missing model object rejected");
}

void test_rejects_empty_vocabulary() {
    Vocab v;
    std::string err;
    const std::string empty = R"({"model":{"type":"BPE","vocab":{},"merges":[]}})";
    ASSERT_TRUE(!build_vocab(empty, "", "", 0, &v, &err), "empty vocabulary rejected");
}

void test_tolerates_absent_optional_config_files() {
    // tokenizer_config.json and generation_config.json are both optional.
    Vocab v;
    std::string err;
    ASSERT_TRUE(build_vocab(tokenizer_json_with(kLlama3RegexJson), "", "", 0, &v, &err),
                "conversion succeeds without optional config files");
    ASSERT_TRUE(build_vocab(tokenizer_json_with(kLlama3RegexJson), "not json", "also not", 0,
                            &v, &err),
                "malformed optional config files are tolerated");
}

}  // namespace

int main() {
    test_identifies_llama3_pre_tokenizer();
    test_identifies_qwen2_pre_tokenizer();
    test_llama3_and_qwen2_are_not_confused();
    test_unknown_pre_tokenizer_falls_back_with_a_warning();

    test_builds_dense_token_array();
    test_pads_to_config_vocab_size();
    test_fills_holes_in_the_id_space();
    test_does_not_shrink_below_max_id();

    test_special_added_tokens_become_control_tokens();

    test_accepts_both_merge_encodings();
    test_warns_when_merges_are_absent();

    test_reads_special_tokens_from_tokenizer_config();
    test_reads_special_tokens_given_as_objects();
    test_generation_config_overrides_eos();
    test_scalar_generation_config_eos();
    test_warns_when_no_eos_is_found();

    test_reads_chat_template();
    test_reads_named_chat_template_list();

    test_identifies_smollm_without_any_regex();
    test_identifies_plain_bytelevel_as_gpt2();
    test_signature_lookup_is_exact();

    test_recognises_control_tokens_by_shape();
    test_unflagged_control_tokens_become_control();

    test_derives_add_bos_from_post_processor();
    test_no_post_processor_means_no_bos();
    test_explicit_config_overrides_derived_add_bos();
    test_reads_add_space_prefix_from_bytelevel();

    test_reads_sampler_defaults_from_generation_config();
    test_sampler_defaults_absent_when_not_declared();

    test_rejects_sentencepiece_with_an_actionable_message();
    test_rejects_malformed_tokenizer_json();
    test_rejects_empty_vocabulary();
    test_tolerates_absent_optional_config_files();

    return test_summary("vocab");
}
