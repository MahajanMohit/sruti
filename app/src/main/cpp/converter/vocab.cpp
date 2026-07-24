#include "vocab.h"

#include <algorithm>
#include <cstring>
#include <unordered_map>

#include <nlohmann/json.hpp>

namespace sruti {

namespace {

using json = nlohmann::json;

/// Pre-tokenizer regexes, taken verbatim from the tokenizer.json of each family.
/// llama.cpp keys its regex sets off these identifiers, so an exact match here is
/// what makes on-device tokenization agree with the reference implementation.
struct PreTokenizerSignature {
    const char * regex;
    const char * name;
};

const PreTokenizerSignature kPreTokenizers[] = {
    // Llama 3.x. Note `\p{N}{1,3}`: digits group in runs of up to three.
    {R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)",
     "llama-bpe"},
    // Same expression with a case-sensitive alternation, as some exports emit.
    {R"((?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)",
     "llama-bpe"},
    // Qwen2 and Qwen3. Differs from Llama 3 only by `\p{N}` — digits split one at
    // a time. A single quantifier, and the only thing distinguishing the two.
    {R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)",
     "qwen2"},
    // GPT-2, which Phi-3's GPT2Tokenizer variant uses.
    {R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)",
     "gpt-2"},
};

/// Structural signatures for families whose tokenizer.json carries no regex.
///
/// The reference converter identifies these by encoding a probe string and
/// hashing the resulting token ids, which needs a full BPE implementation. The
/// component structure distinguishes the supported families just as well and
/// costs nothing to compute.
struct StructuralSignature {
    const char * signature;
    const char * name;
};

const StructuralSignature kStructuralPreTokenizers[] = {
    // SmolLM and SmolLM2: digits split individually, then byte-level.
    {"Digits(individual)+ByteLevel", "smollm"},
    // Plain GPT-2 style.
    {"ByteLevel", "gpt-2"},
};

/// Walks a (possibly nested) pre_tokenizer node, collecting Split regexes and a
/// structural signature of the component types in order.
void walk_pre_tokenizer(
    const json & node,
    std::vector<std::string> * regexes,
    std::vector<std::string> * components) {

    if (node.is_null()) {
        return;
    }
    if (node.is_array()) {
        for (const auto & child : node) {
            walk_pre_tokenizer(child, regexes, components);
        }
        return;
    }
    if (!node.is_object()) {
        return;
    }

    const std::string type = node.value("type", std::string());

    const auto pattern_it = node.find("pattern");
    if (pattern_it != node.end() && pattern_it->is_object()) {
        const auto regex_it = pattern_it->find("Regex");
        if (regex_it != pattern_it->end() && regex_it->is_string()) {
            regexes->push_back(regex_it->get<std::string>());
        }
    }

    if (type == "Digits") {
        components->push_back(node.value("individual_digits", false)
                                  ? "Digits(individual)"
                                  : "Digits");
    } else if (!type.empty() && type != "Sequence") {
        components->push_back(type);
    }

    // "Sequence" nests its children under one of these keys depending on version.
    for (const char * key : {"pretokenizers", "processors", "normalizers"}) {
        const auto it = node.find(key);
        if (it != node.end()) {
            walk_pre_tokenizer(*it, regexes, components);
        }
    }
}

std::string join(const std::vector<std::string> & parts, const char * sep) {
    std::string out;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) out += sep;
        out += parts[i];
    }
    return out;
}

/// Whether the tokenizer's post-processor prepends a special token.
///
/// HF encodes this as a TemplateProcessing whose `single` sequence begins with a
/// SpecialToken — that is precisely what "add BOS" means. Llama 3 has one; Qwen
/// and SmolLM have a ByteLevel post-processor or none at all, and add no BOS.
bool post_processor_adds_bos(const json & tok) {
    const auto pp_it = tok.find("post_processor");
    if (pp_it == tok.end() || !pp_it->is_object()) {
        return false;
    }

    // A Sequence of processors: any TemplateProcessing inside counts.
    const auto type = pp_it->value("type", std::string());
    if (type == "Sequence") {
        const auto list_it = pp_it->find("processors");
        if (list_it != pp_it->end() && list_it->is_array()) {
            for (const auto & child : *list_it) {
                if (!child.is_object()) continue;
                if (child.value("type", std::string()) != "TemplateProcessing") continue;
                const auto single_it = child.find("single");
                if (single_it != child.end() && single_it->is_array() &&
                    !single_it->empty() && (*single_it)[0].contains("SpecialToken")) {
                    return true;
                }
            }
        }
        return false;
    }

    if (type != "TemplateProcessing") {
        return false;
    }
    const auto single_it = pp_it->find("single");
    return single_it != pp_it->end() && single_it->is_array() && !single_it->empty() &&
           (*single_it)[0].contains("SpecialToken");
}

/// Reads ByteLevel's add_prefix_space out of the pre_tokenizer tree.
bool find_add_prefix_space(const json & node, bool * value) {
    if (node.is_array()) {
        for (const auto & child : node) {
            if (find_add_prefix_space(child, value)) return true;
        }
        return false;
    }
    if (!node.is_object()) {
        return false;
    }
    if (node.value("type", std::string()) == "ByteLevel") {
        const auto it = node.find("add_prefix_space");
        if (it != node.end() && it->is_boolean()) {
            *value = it->get<bool>();
            return true;
        }
    }
    for (const char * key : {"pretokenizers", "processors"}) {
        const auto it = node.find(key);
        if (it != node.end() && find_add_prefix_space(*it, value)) return true;
    }
    return false;
}

/// Reads a special-token spec that may be a bare string or an object with
/// "content", as tokenizer_config.json uses both forms.
std::string special_token_text(const json & node) {
    if (node.is_string()) {
        return node.get<std::string>();
    }
    if (node.is_object()) {
        const auto it = node.find("content");
        if (it != node.end() && it->is_string()) {
            return it->get<std::string>();
        }
    }
    return {};
}

json parse_or_empty(const std::string & text) {
    if (text.empty()) {
        return json::object();
    }
    try {
        json parsed = json::parse(text);
        return parsed.is_object() ? parsed : json::object();
    } catch (const std::exception &) {
        return json::object();
    }
}

}  // namespace

std::string pre_tokenizer_from_regex(const std::string & regex) {
    for (const auto & sig : kPreTokenizers) {
        if (regex == sig.regex) {
            return sig.name;
        }
    }
    return {};
}

bool token_looks_special(const std::string & token) {
    // Tokens some families mark non-special even though they are control tokens.
    for (const char * literal : {"<pad>", "<mask>", "<2mass>", "[@BOS@]"}) {
        if (token == literal) return true;
    }

    const auto wrapped = [&token](const char * open, const char * close) {
        const size_t o = std::strlen(open);
        const size_t c = std::strlen(close);
        return token.size() >= o + c &&
               token.compare(0, o, open) == 0 &&
               token.compare(token.size() - c, c, close) == 0;
    };

    // The common convention: <|...|>, and deepseek's fullwidth variant.
    if (wrapped("<|", "|>")) return true;
    if (wrapped("<\xEF\xBD\x9C", "\xEF\xBD\x9C>")) return true;
    // Gemma's reserved slots.
    if (wrapped("<unused", ">")) return true;

    return false;
}

std::string pre_tokenizer_from_signature(const std::string & signature) {
    for (const auto & sig : kStructuralPreTokenizers) {
        if (signature == sig.signature) {
            return sig.name;
        }
    }
    return {};
}

bool build_vocab(
    const std::string & tokenizer_json,
    const std::string & tokenizer_config_json,
    const std::string & generation_config_json,
    int64_t vocab_size,
    Vocab * out,
    std::string * err) {

    json tok;
    try {
        tok = json::parse(tokenizer_json);
    } catch (const std::exception & e) {
        if (err) *err = std::string("tokenizer.json is not valid JSON: ") + e.what();
        return false;
    }
    if (!tok.is_object()) {
        if (err) *err = "tokenizer.json is not a JSON object";
        return false;
    }

    const auto model_it = tok.find("model");
    if (model_it == tok.end() || !model_it->is_object()) {
        if (err) *err = "tokenizer.json has no model object";
        return false;
    }

    const std::string model_type =
        model_it->value("type", std::string());
    if (model_type != "BPE") {
        if (err) {
            *err = "unsupported tokenizer type '" + model_type +
                   "'. Only byte-level BPE (tokenizer.json) is supported. "
                   "SentencePiece checkpoints, which ship a tokenizer.model "
                   "protobuf, cannot be converted yet.";
        }
        return false;
    }

    Vocab vocab;

    // --- vocabulary ---------------------------------------------------------
    const auto vocab_it = model_it->find("vocab");
    if (vocab_it == model_it->end() || !vocab_it->is_object()) {
        if (err) *err = "tokenizer.json model has no vocab object";
        return false;
    }

    // Build id -> token. The JSON object is keyed by token text, so the ids are
    // scattered and the array has to be sized from the maximum.
    std::unordered_map<std::string, int32_t> token_to_id;
    int32_t max_id = -1;
    for (const auto & [text, id_node] : vocab_it->items()) {
        const int32_t id = id_node.get<int32_t>();
        token_to_id.emplace(text, id);
        max_id = std::max(max_id, id);
    }

    const auto added_it = tok.find("added_tokens");
    if (added_it != tok.end() && added_it->is_array()) {
        for (const auto & entry : *added_it) {
            if (!entry.is_object()) continue;
            const auto id_it = entry.find("id");
            const auto content_it = entry.find("content");
            if (id_it == entry.end() || content_it == entry.end()) continue;
            const int32_t id = id_it->get<int32_t>();
            token_to_id.emplace(content_it->get<std::string>(), id);
            max_id = std::max(max_id, id);
        }
    }

    if (max_id < 0) {
        if (err) *err = "tokenizer.json declares an empty vocabulary";
        return false;
    }

    // llama.cpp indexes the token array directly, so it must be dense and at
    // least as long as config.json's vocab_size.
    const size_t count = std::max<size_t>(static_cast<size_t>(max_id) + 1,
                                          vocab_size > 0 ? static_cast<size_t>(vocab_size) : 0);

    vocab.tokens.assign(count, std::string());
    vocab.token_types.assign(count, static_cast<int32_t>(TokenType::Unused));

    for (const auto & [text, id] : token_to_id) {
        vocab.tokens[static_cast<size_t>(id)] = text;
        vocab.token_types[static_cast<size_t>(id)] = static_cast<int32_t>(TokenType::Normal);
    }

    // Mark added tokens by kind: special ones are control tokens and must not be
    // produced by ordinary sampling.
    if (added_it != tok.end() && added_it->is_array()) {
        for (const auto & entry : *added_it) {
            if (!entry.is_object()) continue;
            const auto id_it = entry.find("id");
            if (id_it == entry.end()) continue;
            const auto id = static_cast<size_t>(id_it->get<int32_t>());
            if (id >= vocab.token_types.size()) continue;

            const bool special = entry.value("special", false) ||
                                 token_looks_special(entry.value("content", std::string()));
            vocab.token_types[id] = static_cast<int32_t>(
                special ? TokenType::Control : TokenType::UserDefined);
        }
    }

    // Fill holes with placeholders. A gap left empty would make llama.cpp treat a
    // valid id as an empty string rather than as unusable.
    size_t holes = 0;
    for (size_t i = 0; i < vocab.tokens.size(); ++i) {
        if (vocab.tokens[i].empty() &&
            vocab.token_types[i] == static_cast<int32_t>(TokenType::Unused)) {
            vocab.tokens[i] = "[PAD" + std::to_string(i) + "]";
            ++holes;
        }
    }
    if (holes > 0) {
        vocab.warnings.push_back(
            "padded " + std::to_string(holes) +
            " unused token slots to reach a dense vocabulary of " +
            std::to_string(count));
    }

    // --- merges -------------------------------------------------------------
    const auto merges_it = model_it->find("merges");
    if (merges_it != model_it->end() && merges_it->is_array()) {
        vocab.merges.reserve(merges_it->size());
        for (const auto & entry : *merges_it) {
            if (entry.is_string()) {
                // Legacy form: a single "a b" string.
                vocab.merges.push_back(entry.get<std::string>());
            } else if (entry.is_array() && entry.size() == 2) {
                // Current form: a ["a", "b"] pair.
                vocab.merges.push_back(entry[0].get<std::string>() + " " +
                                       entry[1].get<std::string>());
            }
        }
    }
    if (vocab.merges.empty()) {
        vocab.warnings.push_back(
            "tokenizer.json declares no BPE merges; tokenization will fall back to "
            "single-character splits");
    }

    // --- pre-tokenizer ------------------------------------------------------
    std::vector<std::string> regexes;
    std::vector<std::string> components;
    const auto pre_it = tok.find("pre_tokenizer");
    if (pre_it != tok.end()) {
        walk_pre_tokenizer(*pre_it, &regexes, &components);
    }

    // A Split regex is the most specific signal, so try it first.
    for (const std::string & regex : regexes) {
        const std::string name = pre_tokenizer_from_regex(regex);
        if (!name.empty()) {
            vocab.pre = name;
            break;
        }
    }
    // Families like SmolLM carry no regex at all and are identified structurally.
    if (vocab.pre == "default") {
        const std::string name = pre_tokenizer_from_signature(join(components, "+"));
        if (!name.empty()) {
            vocab.pre = name;
        }
    }
    if (vocab.pre == "default") {
        vocab.warnings.push_back(
            "could not identify the pre-tokenizer from tokenizer.json (components: " +
            join(components, "+") +
            "); falling back to \"default\". Tokenization may differ slightly from "
            "the reference implementation.");
    }

    // --- prefix space -------------------------------------------------------
    bool add_prefix_space = false;
    if (pre_it != tok.end() && find_add_prefix_space(*pre_it, &add_prefix_space)) {
        vocab.add_space_prefix = add_prefix_space;
        vocab.has_add_space_prefix = true;
    }

    // --- special tokens -----------------------------------------------------
    const json cfg = parse_or_empty(tokenizer_config_json);
    const json gen = parse_or_empty(generation_config_json);

    const auto lookup = [&](const std::string & text) -> int32_t {
        if (text.empty()) return -1;
        const auto it = token_to_id.find(text);
        return it == token_to_id.end() ? -1 : it->second;
    };

    const auto special_from_config = [&](const char * key) -> int32_t {
        const auto it = cfg.find(key);
        if (it == cfg.end()) return -1;
        return lookup(special_token_text(*it));
    };

    vocab.bos_id = special_from_config("bos_token");
    vocab.eos_id = special_from_config("eos_token");
    vocab.unk_id = special_from_config("unk_token");
    vocab.pad_id = special_from_config("pad_token");

    // generation_config.json is authoritative for stopping behaviour and often
    // names a different EOS than tokenizer_config.json — instruction-tuned models
    // routinely stop on a chat turn-end token rather than the base EOS.
    const auto gen_eos = gen.find("eos_token_id");
    if (gen_eos != gen.end()) {
        if (gen_eos->is_number_integer()) {
            vocab.eos_id = gen_eos->get<int32_t>();
        } else if (gen_eos->is_array() && !gen_eos->empty()) {
            // A list means several tokens end generation; the first is EOS and a
            // second, when present, is the end-of-turn token.
            vocab.eos_id = (*gen_eos)[0].get<int32_t>();
            if (gen_eos->size() > 1) {
                vocab.eot_id = (*gen_eos)[1].get<int32_t>();
            }
        }
    }
    const auto gen_bos = gen.find("bos_token_id");
    if (gen_bos != gen.end() && gen_bos->is_number_integer()) {
        vocab.bos_id = gen_bos->get<int32_t>();
    }

    if (vocab.eos_id < 0) {
        vocab.warnings.push_back(
            "no EOS token identified; generation will only stop at the token limit");
    }

    // --- flags and chat template -------------------------------------------
    // tokenizer_config.json states this explicitly when it cares. Otherwise derive
    // it from the post-processor, which is where HF actually encodes "prepend BOS":
    // a TemplateProcessing whose single-sequence template starts with a special
    // token. Assuming true by default would insert a spurious BOS into every
    // prompt for models like Qwen and SmolLM that do not want one.
    const auto add_bos_it = cfg.find("add_bos_token");
    if (add_bos_it != cfg.end() && add_bos_it->is_boolean()) {
        vocab.add_bos = add_bos_it->get<bool>();
        vocab.has_add_bos = true;
    } else {
        vocab.add_bos = post_processor_adds_bos(tok);
        vocab.has_add_bos = true;
    }
    const auto add_eos_it = cfg.find("add_eos_token");
    if (add_eos_it != cfg.end() && add_eos_it->is_boolean()) {
        vocab.add_eos = add_eos_it->get<bool>();
        vocab.has_add_eos = true;
    }

    // Sampler defaults the model publishes for itself. llama.cpp surfaces these as
    // recommended settings rather than applying them automatically.
    const auto temp_it = gen.find("temperature");
    if (temp_it != gen.end() && temp_it->is_number()) {
        vocab.sampling_temp = temp_it->get<float>();
    }
    const auto top_p_it = gen.find("top_p");
    if (top_p_it != gen.end() && top_p_it->is_number()) {
        vocab.sampling_top_p = top_p_it->get<float>();
    }
    const auto top_k_it = gen.find("top_k");
    if (top_k_it != gen.end() && top_k_it->is_number_integer()) {
        vocab.sampling_top_k = top_k_it->get<int32_t>();
    }

    const auto template_it = cfg.find("chat_template");
    if (template_it != cfg.end()) {
        if (template_it->is_string()) {
            vocab.chat_template = template_it->get<std::string>();
        } else if (template_it->is_array() && !template_it->empty()) {
            // Some configs ship several named templates; the default one is what
            // llama.cpp's chat handling expects.
            for (const auto & entry : *template_it) {
                if (!entry.is_object()) continue;
                if (entry.value("name", std::string()) == "default") {
                    vocab.chat_template = entry.value("template", std::string());
                    break;
                }
            }
            if (vocab.chat_template.empty() && (*template_it)[0].is_object()) {
                vocab.chat_template = (*template_it)[0].value("template", std::string());
            }
        }
    }

    *out = std::move(vocab);
    return true;
}

}  // namespace sruti
