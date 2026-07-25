// Host-side tests for architecture detection and HF -> GGUF tensor mapping.
//
// The cases that matter most are the ones where a wrong answer still produces a
// loadable file: the Llama Q/K permutation, and Gemma's four-norms-per-block
// layout where `post_attention_layernorm` means something different than it does
// for Llama.

#include "../../main/cpp/converter/arch.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "test_util.h"

using namespace sruti;

namespace {

std::string config_for(
    const std::string & hf_arch,
    const std::string & extra = {}) {

    std::string cfg =
        R"({"architectures":[")" + hf_arch + R"("],)"
        R"("vocab_size":32000,"hidden_size":64,"intermediate_size":128,)"
        R"("num_hidden_layers":2,"num_attention_heads":8,"num_key_value_heads":4,)"
        R"("max_position_embeddings":4096,"rms_norm_eps":1e-6,"rope_theta":10000.0)";
    if (!extra.empty()) {
        cfg += "," + extra;
    }
    return cfg + "}";
}

/// Every tensor a two-block model of the given flavour would contain.
std::vector<std::string> checkpoint_tensors(
    bool gemma_norms, bool qkv_bias, bool qk_norm, bool has_lm_head) {

    std::vector<std::string> names = {
        "model.embed_tokens.weight",
        "model.norm.weight",
    };
    if (has_lm_head) names.push_back("lm_head.weight");

    for (int b = 0; b < 2; ++b) {
        const std::string p = "model.layers." + std::to_string(b) + ".";
        names.push_back(p + "input_layernorm.weight");
        names.push_back(p + "self_attn.q_proj.weight");
        names.push_back(p + "self_attn.k_proj.weight");
        names.push_back(p + "self_attn.v_proj.weight");
        names.push_back(p + "self_attn.o_proj.weight");
        names.push_back(p + "post_attention_layernorm.weight");
        names.push_back(p + "mlp.gate_proj.weight");
        names.push_back(p + "mlp.up_proj.weight");
        names.push_back(p + "mlp.down_proj.weight");

        if (qkv_bias) {
            names.push_back(p + "self_attn.q_proj.bias");
            names.push_back(p + "self_attn.k_proj.bias");
            names.push_back(p + "self_attn.v_proj.bias");
        }
        if (qk_norm) {
            names.push_back(p + "self_attn.q_norm.weight");
            names.push_back(p + "self_attn.k_norm.weight");
        }
        if (gemma_norms) {
            names.push_back(p + "pre_feedforward_layernorm.weight");
            names.push_back(p + "post_feedforward_layernorm.weight");
        }
    }
    return names;
}

std::map<std::string, TensorPlan> by_gguf_name(const std::vector<TensorPlan> & plan) {
    std::map<std::string, TensorPlan> out;
    for (const auto & p : plan) out[p.gguf_name] = p;
    return out;
}

const KvPair * find_kv(const std::vector<KvPair> & kv, const std::string & key) {
    for (const auto & p : kv) {
        if (p.key == key) return &p;
    }
    return nullptr;
}

// --- architecture identity --------------------------------------------------

void test_recognises_supported_architectures() {
    ASSERT_TRUE(arch_from_hf_name("LlamaForCausalLM") == Arch::Llama, "Llama recognised");
    ASSERT_TRUE(arch_from_hf_name("MistralForCausalLM") == Arch::Llama,
                "Mistral treated as Llama family");
    ASSERT_TRUE(arch_from_hf_name("Qwen2ForCausalLM") == Arch::Qwen2, "Qwen2 recognised");
    ASSERT_TRUE(arch_from_hf_name("Qwen3ForCausalLM") == Arch::Qwen3, "Qwen3 recognised");
    ASSERT_TRUE(arch_from_hf_name("Gemma3ForCausalLM") == Arch::Gemma3, "Gemma3 recognised");
    ASSERT_TRUE(arch_from_hf_name("Phi3ForCausalLM") == Arch::Phi3, "Phi3 recognised");
    ASSERT_TRUE(arch_from_hf_name("MambaForCausalLM") == Arch::Unknown,
                "unknown architecture reported as unknown");
}

void test_rejects_unsupported_architecture_with_a_useful_message() {
    HParams hp;
    std::string err;
    const bool ok = parse_hparams(config_for("MambaForCausalLM"), &hp, &err);
    ASSERT_TRUE(!ok, "unsupported architecture rejected");
    ASSERT_TRUE(err.find("MambaForCausalLM") != std::string::npos,
                "error names the offending architecture");
    ASSERT_TRUE(err.find("Supported:") != std::string::npos,
                "error lists what is supported");
}

void test_rejects_malformed_config() {
    HParams hp;
    std::string err;
    ASSERT_TRUE(!parse_hparams("{not json", &hp, &err), "malformed JSON rejected");
    ASSERT_TRUE(!parse_hparams("{}", &hp, &err), "config without architectures rejected");
}

// --- hyperparameters --------------------------------------------------------

void test_parses_core_hyperparameters() {
    HParams hp;
    std::string err;
    ASSERT_TRUE(parse_hparams(config_for("LlamaForCausalLM"), &hp, &err),
                std::string("parsed config: ") + err);

    ASSERT_EQ(hp.hidden_size, int64_t(64), "hidden_size");
    ASSERT_EQ(hp.intermediate_size, int64_t(128), "intermediate_size");
    ASSERT_EQ(hp.block_count, int64_t(2), "block_count");
    ASSERT_EQ(hp.head_count, int64_t(8), "head_count");
    ASSERT_EQ(hp.head_count_kv, int64_t(4), "head_count_kv");
    ASSERT_EQ(hp.head_dim, int64_t(8), "head_dim derived from hidden/heads");
    ASSERT_EQ(hp.context_length, int64_t(4096), "context_length");
}

void test_defaults_kv_heads_to_head_count_when_absent() {
    // Absent num_key_value_heads means plain multi-head attention. Defaulting to
    // anything else would silently reshape K and V.
    const std::string cfg =
        R"({"architectures":["LlamaForCausalLM"],"vocab_size":32,"hidden_size":64,)"
        R"("intermediate_size":128,"num_hidden_layers":1,"num_attention_heads":8})";
    HParams hp;
    std::string err;
    ASSERT_TRUE(parse_hparams(cfg, &hp, &err), std::string("parsed: ") + err);
    ASSERT_EQ(hp.head_count_kv, int64_t(8), "kv heads default to head count");
}

void test_prefers_explicit_head_dim() {
    // Qwen3 states head_dim explicitly and it is not hidden_size / head_count.
    // Deriving it would produce a wrong rope.dimension_count.
    HParams hp;
    std::string err;
    ASSERT_TRUE(parse_hparams(config_for("Qwen3ForCausalLM", R"("head_dim":128)"), &hp, &err),
                std::string("parsed: ") + err);
    ASSERT_EQ(hp.head_dim, int64_t(128), "explicit head_dim wins over derivation");
}

void test_flattens_text_config_for_multimodal_checkpoints() {
    const std::string cfg =
        R"({"architectures":["Gemma3ForCausalLM"],"text_config":{"hidden_size":128,)"
        R"("intermediate_size":256,"num_hidden_layers":4,"num_attention_heads":8,)"
        R"("head_dim":32,"vocab_size":1000}})";
    HParams hp;
    std::string err;
    ASSERT_TRUE(parse_hparams(cfg, &hp, &err), std::string("parsed: ") + err);
    ASSERT_EQ(hp.hidden_size, int64_t(128), "hidden_size read from text_config");
    ASSERT_EQ(hp.block_count, int64_t(4), "block_count read from text_config");
}

// --- the permutation trap ---------------------------------------------------

void test_llama_permutes_q_and_k_but_nothing_else() {
    HParams hp;
    std::string err;
    parse_hparams(config_for("LlamaForCausalLM"), &hp, &err);
    ASSERT_TRUE(hp.undo_permute(), "Llama flagged for Q/K de-interleaving");

    std::vector<TensorPlan> plan;
    ASSERT_TRUE(build_tensor_plan(hp, checkpoint_tensors(false, false, false, true),
                                  &plan, &err),
                std::string("built plan: ") + err);

    const auto map = by_gguf_name(plan);
    ASSERT_TRUE(map.at("blk.0.attn_q.weight").transform == WeightTransform::PermuteQuery,
                "Q permuted with head count");
    ASSERT_TRUE(map.at("blk.0.attn_k.weight").transform == WeightTransform::PermuteKey,
                "K permuted with KV head count");
    ASSERT_TRUE(map.at("blk.0.attn_v.weight").transform == WeightTransform::None,
                "V is not permuted");
    ASSERT_TRUE(map.at("blk.0.attn_output.weight").transform == WeightTransform::None,
                "output projection is not permuted");
}

void test_qwen_does_not_permute() {
    // Qwen stores Q/K in the layout GGUF already expects. Permuting would produce
    // a file that loads and generates garbage.
    for (const char * hf : {"Qwen2ForCausalLM", "Qwen3ForCausalLM"}) {
        HParams hp;
        std::string err;
        parse_hparams(config_for(hf), &hp, &err);
        ASSERT_TRUE(!hp.undo_permute(), std::string(hf) + " is not permuted");

        std::vector<TensorPlan> plan;
        ASSERT_TRUE(build_tensor_plan(hp, checkpoint_tensors(false, true, true, true),
                                      &plan, &err),
                    std::string("built plan: ") + err);
        const auto map = by_gguf_name(plan);
        ASSERT_TRUE(map.at("blk.0.attn_q.weight").transform == WeightTransform::None,
                    std::string(hf) + " Q untouched");
        ASSERT_TRUE(map.at("blk.0.attn_k.weight").transform == WeightTransform::None,
                    std::string(hf) + " K untouched");
    }
}

void test_gemma_does_not_permute() {
    HParams hp;
    std::string err;
    parse_hparams(config_for("Gemma3ForCausalLM"), &hp, &err);
    ASSERT_TRUE(!hp.undo_permute(), "Gemma is not permuted");
}

// --- the Gemma norm trap ----------------------------------------------------

void test_llama_maps_post_attention_layernorm_to_ffn_norm() {
    HParams hp;
    std::string err;
    parse_hparams(config_for("LlamaForCausalLM"), &hp, &err);

    std::vector<TensorPlan> plan;
    ASSERT_TRUE(build_tensor_plan(hp, checkpoint_tensors(false, false, false, true),
                                  &plan, &err),
                std::string("built plan: ") + err);

    const auto map = by_gguf_name(plan);
    ASSERT_EQ(map.at("blk.0.ffn_norm.weight").source_name,
              std::string("model.layers.0.post_attention_layernorm.weight"),
              "Llama ffn_norm comes from post_attention_layernorm");
    ASSERT_TRUE(map.count("blk.0.post_attention_norm.weight") == 0,
                "Llama has no separate post-attention norm");
}

void test_gemma_maps_the_same_name_somewhere_else() {
    HParams hp;
    std::string err;
    parse_hparams(config_for("Gemma3ForCausalLM"), &hp, &err);

    std::vector<TensorPlan> plan;
    ASSERT_TRUE(build_tensor_plan(hp, checkpoint_tensors(true, false, true, false),
                                  &plan, &err),
                std::string("built plan: ") + err);

    const auto map = by_gguf_name(plan);
    // The whole trap in one assertion: identical source name, different target.
    ASSERT_EQ(map.at("blk.0.post_attention_norm.weight").source_name,
              std::string("model.layers.0.post_attention_layernorm.weight"),
              "Gemma post_attention_layernorm becomes post_attention_norm");
    ASSERT_EQ(map.at("blk.0.ffn_norm.weight").source_name,
              std::string("model.layers.0.pre_feedforward_layernorm.weight"),
              "Gemma ffn_norm comes from pre_feedforward_layernorm instead");
    ASSERT_TRUE(map.count("blk.0.post_ffw_norm.weight") == 1,
                "Gemma post-feedforward norm mapped");
}

void test_gemma_adds_one_to_norm_weights() {
    HParams hp;
    std::string err;
    parse_hparams(config_for("Gemma3ForCausalLM"), &hp, &err);

    std::vector<TensorPlan> plan;
    build_tensor_plan(hp, checkpoint_tensors(true, false, true, false), &plan, &err);
    const auto map = by_gguf_name(plan);

    // Gemma's RMSNorm computes (1 + w), so the stored weight needs shifting.
    ASSERT_TRUE(map.at("blk.0.attn_norm.weight").transform == WeightTransform::AddOne,
                "attn_norm shifted by one");
    ASSERT_TRUE(map.at("output_norm.weight").transform == WeightTransform::AddOne,
                "output_norm shifted by one");
    // Non-norm tensors must not be shifted.
    ASSERT_TRUE(map.at("blk.0.attn_q.weight").transform == WeightTransform::None,
                "projections are not shifted");
}

void test_llama_does_not_add_one() {
    HParams hp;
    std::string err;
    parse_hparams(config_for("LlamaForCausalLM"), &hp, &err);
    std::vector<TensorPlan> plan;
    build_tensor_plan(hp, checkpoint_tensors(false, false, false, true), &plan, &err);
    const auto map = by_gguf_name(plan);
    ASSERT_TRUE(map.at("blk.0.attn_norm.weight").transform == WeightTransform::None,
                "Llama norms are used as stored");
}

// --- output head ------------------------------------------------------------

void test_gemma_skips_lm_head() {
    HParams hp;
    std::string err;
    parse_hparams(config_for("Gemma3ForCausalLM"), &hp, &err);

    // Even when the checkpoint ships lm_head, Gemma ties output to embeddings and
    // llama.cpp rejects a file that carries both.
    auto names = checkpoint_tensors(true, false, true, /*has_lm_head=*/true);
    std::vector<TensorPlan> plan;
    ASSERT_TRUE(build_tensor_plan(hp, names, &plan, &err), std::string("built plan: ") + err);

    const auto map = by_gguf_name(plan);
    ASSERT_TRUE(map.count("output.weight") == 0, "Gemma lm_head skipped");
}

void test_tied_embeddings_skip_output_tensor() {
    HParams hp;
    std::string err;
    parse_hparams(config_for("Qwen3ForCausalLM", R"("tie_word_embeddings":true)"), &hp, &err);
    ASSERT_TRUE(hp.tie_word_embeddings, "tie_word_embeddings parsed");

    std::vector<TensorPlan> plan;
    build_tensor_plan(hp, checkpoint_tensors(false, false, true, true), &plan, &err);
    ASSERT_TRUE(by_gguf_name(plan).count("output.weight") == 0,
                "tied models omit a separate output tensor");
}

void test_untied_model_keeps_output_tensor() {
    HParams hp;
    std::string err;
    parse_hparams(config_for("LlamaForCausalLM"), &hp, &err);
    std::vector<TensorPlan> plan;
    build_tensor_plan(hp, checkpoint_tensors(false, false, false, true), &plan, &err);
    ASSERT_TRUE(by_gguf_name(plan).count("output.weight") == 1,
                "untied model keeps lm_head as output.weight");
}

// --- optional tensors -------------------------------------------------------

void test_qwen2_biases_are_mapped_and_not_permuted() {
    HParams hp;
    std::string err;
    parse_hparams(config_for("Qwen2ForCausalLM"), &hp, &err);

    std::vector<TensorPlan> plan;
    ASSERT_TRUE(build_tensor_plan(hp, checkpoint_tensors(false, true, false, true),
                                  &plan, &err),
                std::string("built plan: ") + err);

    const auto map = by_gguf_name(plan);
    ASSERT_TRUE(map.count("blk.0.attn_q.bias") == 1, "Q bias mapped");
    ASSERT_TRUE(map.at("blk.0.attn_q.bias").transform == WeightTransform::None,
                "biases are never permuted");
}

void test_absent_optional_tensors_are_simply_omitted() {
    // A Llama checkpoint has no q_norm; asking for one must not fail the plan.
    HParams hp;
    std::string err;
    parse_hparams(config_for("LlamaForCausalLM"), &hp, &err);

    std::vector<TensorPlan> plan;
    ASSERT_TRUE(build_tensor_plan(hp, checkpoint_tensors(false, false, false, true),
                                  &plan, &err),
                "plan succeeds without optional tensors");
    ASSERT_TRUE(by_gguf_name(plan).count("blk.0.attn_q_norm.weight") == 0,
                "absent optional tensor omitted");
}

void test_missing_required_tensor_fails_loudly() {
    HParams hp;
    std::string err;
    parse_hparams(config_for("LlamaForCausalLM"), &hp, &err);

    auto names = checkpoint_tensors(false, false, false, true);
    names.erase(std::remove(names.begin(), names.end(),
                            "model.layers.1.mlp.down_proj.weight"),
                names.end());

    std::vector<TensorPlan> plan;
    ASSERT_TRUE(!build_tensor_plan(hp, names, &plan, &err), "missing tensor rejected");
    ASSERT_TRUE(err.find("down_proj") != std::string::npos, "error names the tensor");
}

void test_plan_covers_every_block() {
    HParams hp;
    std::string err;
    parse_hparams(config_for("LlamaForCausalLM"), &hp, &err);

    std::vector<TensorPlan> plan;
    build_tensor_plan(hp, checkpoint_tensors(false, false, false, true), &plan, &err);

    const auto map = by_gguf_name(plan);
    // 2 blocks x 9 tensors + token_embd + output_norm + output
    ASSERT_EQ(plan.size(), size_t(2 * 9 + 3), "every block and global tensor planned");
    ASSERT_TRUE(map.count("blk.1.ffn_down.weight") == 1, "second block present");
}

void test_phi3_uses_fused_projections() {
    HParams hp;
    std::string err;
    parse_hparams(config_for("Phi3ForCausalLM"), &hp, &err);

    std::vector<std::string> names = {
        "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
    };
    for (int b = 0; b < 2; ++b) {
        const std::string p = "model.layers." + std::to_string(b) + ".";
        names.push_back(p + "input_layernorm.weight");
        names.push_back(p + "self_attn.qkv_proj.weight");
        names.push_back(p + "self_attn.o_proj.weight");
        names.push_back(p + "post_attention_layernorm.weight");
        names.push_back(p + "mlp.gate_up_proj.weight");
        names.push_back(p + "mlp.down_proj.weight");
    }

    std::vector<TensorPlan> plan;
    ASSERT_TRUE(build_tensor_plan(hp, names, &plan, &err), std::string("built plan: ") + err);

    const auto map = by_gguf_name(plan);
    ASSERT_TRUE(map.count("blk.0.attn_qkv.weight") == 1, "fused QKV mapped");
    ASSERT_TRUE(map.count("blk.0.attn_q.weight") == 0, "no split Q expected");
    ASSERT_EQ(map.at("blk.0.ffn_up.weight").source_name,
              std::string("model.layers.0.mlp.gate_up_proj.weight"),
              "fused gate_up becomes ffn_up");
}

// --- inspection --------------------------------------------------------------

void test_estimates_parameter_count_for_real_models() {
    // Hyperparameters taken from the published config.json of each model. The
    // estimate drives the "this is a 1.2 B model, you need 3.4 GB free" preflight,
    // so being wrong by a large factor would either block valid models or let the
    // user start a download that cannot finish.
    struct Case {
        const char * name;
        const char * config;
        int64_t expected;   // published parameter count
        double tolerance;   // fractional
    };

    const Case cases[] = {
        {"SmolLM2-135M",
         R"({"architectures":["LlamaForCausalLM"],"hidden_size":576,)"
         R"("intermediate_size":1536,"num_hidden_layers":30,"num_attention_heads":9,)"
         R"("num_key_value_heads":3,"vocab_size":49152,"tie_word_embeddings":true})",
         135'000'000, 0.05},
        {"Qwen2.5-0.5B",
         R"({"architectures":["Qwen2ForCausalLM"],"hidden_size":896,)"
         R"("intermediate_size":4864,"num_hidden_layers":24,"num_attention_heads":14,)"
         R"("num_key_value_heads":2,"vocab_size":151936,"tie_word_embeddings":true})",
         494'000'000, 0.05},
        {"Llama-3.2-1B",
         R"({"architectures":["LlamaForCausalLM"],"hidden_size":2048,)"
         R"("intermediate_size":8192,"num_hidden_layers":16,"num_attention_heads":32,)"
         R"("num_key_value_heads":8,"vocab_size":128256,"tie_word_embeddings":true})",
         1'236'000'000, 0.05},
    };

    for (const Case & c : cases) {
        const CheckpointInfo info = inspect_config(c.config);
        ASSERT_TRUE(info.supported, std::string(c.name) + " is supported");

        const double ratio =
            static_cast<double>(info.parameter_count) / static_cast<double>(c.expected);
        ASSERT_TRUE(ratio > 1.0 - c.tolerance && ratio < 1.0 + c.tolerance,
                    std::string(c.name) + " parameter estimate within tolerance");
    }
}

void test_untied_embeddings_counted_twice() {
    const std::string tied =
        R"({"architectures":["LlamaForCausalLM"],"hidden_size":64,"intermediate_size":128,)"
        R"("num_hidden_layers":1,"num_attention_heads":8,"vocab_size":10000,)"
        R"("tie_word_embeddings":true})";
    const std::string untied =
        R"({"architectures":["LlamaForCausalLM"],"hidden_size":64,"intermediate_size":128,)"
        R"("num_hidden_layers":1,"num_attention_heads":8,"vocab_size":10000,)"
        R"("tie_word_embeddings":false})";

    const int64_t a = inspect_config(tied).parameter_count;
    const int64_t b = inspect_config(untied).parameter_count;
    ASSERT_EQ(b - a, int64_t(10000 * 64), "untied output head adds one embedding table");
}

void test_inspect_reports_unsupported_architecture_with_its_name() {
    const CheckpointInfo info = inspect_config(config_for("MambaForCausalLM"));
    ASSERT_TRUE(!info.supported, "unsupported architecture reported");
    ASSERT_EQ(info.hf_arch, std::string("MambaForCausalLM"),
              "the rejected architecture is still named");
    ASSERT_TRUE(!info.error.empty(), "an explanation is provided");
}

void test_inspect_survives_malformed_json() {
    const CheckpointInfo info = inspect_config("{not json");
    ASSERT_TRUE(!info.supported, "malformed config rejected");
    ASSERT_TRUE(!info.error.empty(), "an explanation is provided");
}

void test_inspect_reports_shape_for_supported_models() {
    const CheckpointInfo info = inspect_config(config_for("Qwen3ForCausalLM"));
    ASSERT_TRUE(info.supported, "supported");
    ASSERT_EQ(info.arch, std::string("qwen3"), "gguf architecture name");
    ASSERT_EQ(info.display_name, std::string("Qwen3"), "display name");
    ASSERT_EQ(info.block_count, int64_t(2), "block count");
    ASSERT_EQ(info.context_length, int64_t(4096), "context length");
}

// --- metadata ---------------------------------------------------------------

void test_metadata_carries_required_keys() {
    HParams hp;
    std::string err;
    parse_hparams(config_for("LlamaForCausalLM"), &hp, &err);

    const auto kv = build_metadata(hp, "test-model");

    ASSERT_EQ(find_kv(kv, "general.architecture")->str, std::string("llama"),
              "architecture key");
    ASSERT_EQ(find_kv(kv, "llama.block_count")->u32, uint32_t(2), "block count");
    ASSERT_EQ(find_kv(kv, "llama.attention.head_count")->u32, uint32_t(8), "head count");
    ASSERT_EQ(find_kv(kv, "llama.attention.head_count_kv")->u32, uint32_t(4), "kv head count");
    ASSERT_EQ(find_kv(kv, "llama.rope.dimension_count")->u32, uint32_t(8), "rope dims");
    ASSERT_TRUE(find_kv(kv, "llama.context_length") != nullptr, "context length present");
    ASSERT_TRUE(find_kv(kv, "llama.feed_forward_length") != nullptr, "ffn length present");
    ASSERT_TRUE(find_kv(kv, "llama.attention.layer_norm_rms_epsilon") != nullptr,
                "rms eps present");
}

void test_metadata_is_namespaced_per_architecture() {
    HParams hp;
    std::string err;
    parse_hparams(config_for("Qwen3ForCausalLM"), &hp, &err);
    const auto kv = build_metadata(hp, "q");
    ASSERT_TRUE(find_kv(kv, "qwen3.block_count") != nullptr, "keys use the arch prefix");
    ASSERT_TRUE(find_kv(kv, "llama.block_count") == nullptr, "no cross-arch leakage");
}

void test_metadata_states_key_length_when_head_dim_is_explicit() {
    // Qwen3 declares head_dim, and it is not hidden_size / head_count, so
    // llama.cpp cannot infer key/value length and must be told.
    HParams hp;
    std::string err;
    parse_hparams(config_for("Qwen3ForCausalLM", R"("head_dim":128)"), &hp, &err);
    ASSERT_TRUE(hp.head_dim_explicit, "explicit head_dim flagged");

    const auto kv = build_metadata(hp, "q");
    ASSERT_TRUE(find_kv(kv, "qwen3.attention.key_length") != nullptr,
                "key_length stated when head_dim is explicit");
    ASSERT_EQ(find_kv(kv, "qwen3.attention.key_length")->u32, uint32_t(128), "key_length value");
}

void test_metadata_omits_key_length_for_qwen_without_explicit_head_dim() {
    // Verified against the reference converter: it omits these for Qwen2 and
    // llama.cpp derives the same value from embedding length and head count.
    HParams hp;
    std::string err;
    parse_hparams(config_for("Qwen2ForCausalLM"), &hp, &err);
    ASSERT_TRUE(!hp.head_dim_explicit, "derived head_dim not flagged explicit");

    const auto kv = build_metadata(hp, "q");
    ASSERT_TRUE(find_kv(kv, "qwen2.attention.key_length") == nullptr,
                "key_length omitted when derivable");
}

void test_metadata_states_key_length_for_llama() {
    // The reference emits these unconditionally for Llama, so matching it keeps
    // the two outputs byte-comparable.
    HParams hp;
    std::string err;
    parse_hparams(config_for("LlamaForCausalLM"), &hp, &err);
    const auto kv = build_metadata(hp, "l");
    ASSERT_TRUE(find_kv(kv, "llama.attention.key_length") != nullptr,
                "key_length always stated for Llama");
}

/// transformers 5.x moved rope_theta into a nested "rope_parameters" object.
///
/// The old layout still dominates the Hub, so both have to work. Reading only
/// the old one is invisible in every check that matters -- the file converts,
/// loads and generates -- and shows up only as answers that fall apart as the
/// context grows.
void test_reads_rope_theta_from_either_config_layout() {
    std::string err;

    HParams legacy;
    ASSERT_TRUE(parse_hparams(config_for("Qwen2ForCausalLM"), &legacy, &err), err);
    ASSERT_EQ(10000, static_cast<int64_t>(legacy.rope_freq_base),
                  "top-level rope_theta is read");

    // config_for already sets a top-level rope_theta, so this also pins that the
    // nested object wins rather than merely being consulted as a fallback.
    HParams nested;
    ASSERT_TRUE(
        parse_hparams(
            config_for("Qwen2ForCausalLM",
                       R"("rope_parameters":{"rope_theta":1000000.0,"rope_type":"default"})"),
            &nested, &err),
        err);
    ASSERT_EQ(1000000, static_cast<int64_t>(nested.rope_freq_base),
                  "nested rope_parameters.rope_theta is read");

    const auto kv = build_metadata(nested, "q");
    const auto * emitted = find_kv(kv, "qwen2.rope.freq_base");
    ASSERT_TRUE(emitted != nullptr, "rope.freq_base is emitted");
}

/// A rope_type of "default" means no scaling at all.
void test_default_rope_type_is_not_treated_as_scaling() {
    std::string err;
    HParams hp;
    ASSERT_TRUE(
        parse_hparams(
            config_for("Qwen2ForCausalLM",
                       R"("rope_parameters":{"rope_theta":1000000.0,"rope_type":"default"})"),
            &hp, &err),
        err);
    ASSERT_TRUE(hp.rope_scaling_type.empty(), "\"default\" is not a scaling type");

    const auto kv = build_metadata(hp, "q");
    ASSERT_TRUE(find_kv(kv, "qwen2.rope.scaling.type") == nullptr,
                "no scaling key is emitted for an unscaled model");
}

/// Real scaling still comes through, from either layout.
void test_real_rope_scaling_survives_both_layouts() {
    std::string err;

    HParams old_layout;
    ASSERT_TRUE(
        parse_hparams(
            config_for("LlamaForCausalLM",
                       R"("rope_scaling":{"rope_type":"linear","factor":4.0})"),
            &old_layout, &err),
        err);
    ASSERT_TRUE(old_layout.rope_scaling_type == "linear", "old-layout scaling type");

    HParams new_layout;
    ASSERT_TRUE(
        parse_hparams(
            config_for("LlamaForCausalLM",
                       R"("rope_parameters":{"rope_theta":500000.0,)"
                       R"("rope_type":"linear","factor":4.0})"),
            &new_layout, &err),
        err);
    ASSERT_TRUE(new_layout.rope_scaling_type == "linear", "new-layout scaling type");
    ASSERT_EQ(500000, static_cast<int64_t>(new_layout.rope_freq_base),
                  "new-layout theta alongside scaling");
}

void test_sliding_window_is_gated_on_use_sliding_window() {
    // Qwen2.5 declares a 32k window with use_sliding_window false. Emitting the
    // key regardless switches on attention the model was never trained with.
    HParams disabled;
    std::string err;
    parse_hparams(
        config_for("Qwen2ForCausalLM", R"("sliding_window":32768,"use_sliding_window":false)"),
        &disabled, &err);
    // Bind the vector: find_kv returns a pointer into it, so passing a temporary
    // would leave the result dangling.
    const auto disabled_kv = build_metadata(disabled, "q");
    ASSERT_TRUE(find_kv(disabled_kv, "qwen2.attention.sliding_window") == nullptr,
                "declared-but-unused sliding window is not emitted");

    HParams enabled;
    parse_hparams(
        config_for("Qwen2ForCausalLM", R"("sliding_window":4096,"use_sliding_window":true)"),
        &enabled, &err);
    const auto enabled_kv = build_metadata(enabled, "q");
    const auto * kv = find_kv(enabled_kv, "qwen2.attention.sliding_window");
    ASSERT_TRUE(kv != nullptr, "an actually-used sliding window is emitted");
    ASSERT_EQ(kv->u32, uint32_t(4096), "sliding window value");
}

}  // namespace

int main() {
    test_recognises_supported_architectures();
    test_rejects_unsupported_architecture_with_a_useful_message();
    test_rejects_malformed_config();

    test_parses_core_hyperparameters();
    test_defaults_kv_heads_to_head_count_when_absent();
    test_prefers_explicit_head_dim();
    test_flattens_text_config_for_multimodal_checkpoints();

    test_llama_permutes_q_and_k_but_nothing_else();
    test_qwen_does_not_permute();
    test_gemma_does_not_permute();

    test_llama_maps_post_attention_layernorm_to_ffn_norm();
    test_gemma_maps_the_same_name_somewhere_else();
    test_gemma_adds_one_to_norm_weights();
    test_llama_does_not_add_one();

    test_gemma_skips_lm_head();
    test_tied_embeddings_skip_output_tensor();
    test_untied_model_keeps_output_tensor();

    test_qwen2_biases_are_mapped_and_not_permuted();
    test_absent_optional_tensors_are_simply_omitted();
    test_missing_required_tensor_fails_loudly();
    test_plan_covers_every_block();
    test_phi3_uses_fused_projections();

    test_estimates_parameter_count_for_real_models();
    test_untied_embeddings_counted_twice();
    test_inspect_reports_unsupported_architecture_with_its_name();
    test_inspect_survives_malformed_json();
    test_inspect_reports_shape_for_supported_models();

    test_metadata_carries_required_keys();
    test_metadata_is_namespaced_per_architecture();
    test_metadata_states_key_length_when_head_dim_is_explicit();
    test_metadata_omits_key_length_for_qwen_without_explicit_head_dim();
    test_metadata_states_key_length_for_llama();
    test_sliding_window_is_gated_on_use_sliding_window();
    test_reads_rope_theta_from_either_config_layout();
    test_default_rope_type_is_not_treated_as_scaling();
    test_real_rope_scaling_survives_both_layouts();

    return test_summary("arch");
}
