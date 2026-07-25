#include "converter.h"

#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "arch.h"
#include "safetensors.h"
#include "tensor_ops.h"
#include "vocab.h"

#include "ggml.h"
#include "gguf.h"
#include "llama.h"

namespace sruti {

namespace {

/// Cap on the f32 scratch used while streaming a tensor.
///
/// token_embd is the reason this exists: a 128k x 2048 embedding table is 262M
/// elements, which is a full gigabyte as f32. Streaming it in row blocks keeps
/// peak memory bounded regardless of vocabulary size.
constexpr size_t kMaxScratchFloats = 8u * 1024 * 1024;  // 32 MiB of f32

std::string join_path(const std::string & dir, const std::string & name) {
    if (dir.empty()) return name;
    if (dir.back() == '/') return dir + name;
    return dir + "/" + name;
}

std::string basename_of(const std::string & path) {
    const size_t slash = path.find_last_of('/');
    std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
    if (!name.empty() && name.back() == '/') name.pop_back();
    return name;
}

bool read_text_file(const std::string & path, std::string * out) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return false;
    }
    out->clear();
    char buf[8192];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) {
        out->append(buf, n);
    }
    std::fclose(f);
    return true;
}

uint64_t free_disk_bytes(const std::string & path) {
    struct statvfs st{};
    if (::statvfs(path.c_str(), &st) != 0) {
        return 0;
    }
    return static_cast<uint64_t>(st.f_bavail) * static_cast<uint64_t>(st.f_frsize);
}

llama_ftype ftype_from_name(const std::string & name) {
    if (name == "Q4_0")   return LLAMA_FTYPE_MOSTLY_Q4_0;
    if (name == "Q8_0")   return LLAMA_FTYPE_MOSTLY_Q8_0;
    if (name == "Q4_K_S") return LLAMA_FTYPE_MOSTLY_Q4_K_S;
    if (name == "Q4_K_M") return LLAMA_FTYPE_MOSTLY_Q4_K_M;
    if (name == "Q5_K_M") return LLAMA_FTYPE_MOSTLY_Q5_K_M;
    if (name == "Q6_K")   return LLAMA_FTYPE_MOSTLY_Q6_K;
    return LLAMA_FTYPE_MOSTLY_F16;
}

/// Approximate bits per weight for disk preflighting.
double bits_per_weight(const std::string & quant_type) {
    if (quant_type == "Q4_0")   return 4.5;
    if (quant_type == "Q4_K_S") return 4.6;
    if (quant_type == "Q4_K_M") return 4.9;
    if (quant_type == "Q5_K_M") return 5.7;
    if (quant_type == "Q6_K")   return 6.6;
    if (quant_type == "Q8_0")   return 8.5;
    return 16.0;
}

/// Appends `n` zero bytes so the next tensor starts on an alignment boundary.
bool pad_to_alignment(FILE * f, size_t alignment) {
    const long pos = std::ftell(f);
    if (pos < 0) return false;
    const size_t rem = static_cast<size_t>(pos) % alignment;
    if (rem == 0) return true;

    static const uint8_t zeros[GGUF_DEFAULT_ALIGNMENT] = {0};
    const size_t pad = alignment - rem;
    return std::fwrite(zeros, 1, pad, f) == pad;
}

/// Writes one tensor's data, streaming when no whole-tensor transform is needed.
///
/// `out_type` is GGML_TYPE_F32 for 1-D tensors (llama.cpp expects norms and biases
/// in full precision) and GGML_TYPE_F16 otherwise.
bool write_tensor_data(
    FILE * f,
    const TensorView & view,
    const TensorPlan & plan,
    const HParams & hp,
    ggml_type out_type,
    std::string * err) {

    const int64_t numel = view.numel();
    const int64_t rows = view.shape.empty() ? 0 : view.shape[0];
    const int64_t cols = numel == 0 || rows == 0 ? 0 : numel / rows;

    const bool needs_whole_tensor =
        plan.transform == WeightTransform::PermuteQuery ||
        plan.transform == WeightTransform::PermuteKey;

    std::vector<float> f32;
    std::vector<uint16_t> f16;

    if (needs_whole_tensor) {
        // The permutation moves rows across the whole tensor, so it cannot be
        // streamed. Q and K are at most hidden x hidden, which is tens of MiB.
        f32.resize(static_cast<size_t>(numel));
        if (!tensor_to_f32(view, f32.data(), err)) {
            return false;
        }

        const int64_t n_head = plan.transform == WeightTransform::PermuteQuery
                                   ? hp.head_count
                                   : hp.head_count_kv;
        if (!permute_rope(f32.data(), rows, cols, n_head)) {
            if (err) {
                *err = "cannot permute '" + view.name + "': " + std::to_string(rows) +
                       " rows is not divisible by 2 x " + std::to_string(n_head) + " heads";
            }
            return false;
        }

        if (out_type == GGML_TYPE_F16) {
            f16.resize(f32.size());
            f32_to_f16_row(f32.data(), f16.data(), f32.size());
            return std::fwrite(f16.data(), sizeof(uint16_t), f16.size(), f) == f16.size();
        }
        return std::fwrite(f32.data(), sizeof(float), f32.size(), f) == f32.size();
    }

    // Streaming path. Row-aligned blocks so an element never straddles a chunk.
    const size_t row_floats = static_cast<size_t>(cols > 0 ? cols : numel);
    const size_t rows_per_chunk =
        std::max<size_t>(1, kMaxScratchFloats / std::max<size_t>(1, row_floats));
    const int64_t total_rows = rows > 0 ? rows : 1;

    f32.resize(rows_per_chunk * row_floats);
    if (out_type == GGML_TYPE_F16) {
        f16.resize(f32.size());
    }

    const size_t elem_size = st_dtype_size(view.dtype);

    for (int64_t start = 0; start < total_rows; start += static_cast<int64_t>(rows_per_chunk)) {
        const int64_t chunk_rows =
            std::min<int64_t>(static_cast<int64_t>(rows_per_chunk), total_rows - start);
        const size_t chunk_floats = static_cast<size_t>(chunk_rows) * row_floats;

        TensorView slice = view;
        slice.shape = {chunk_rows, static_cast<int64_t>(row_floats)};
        slice.data = view.data + static_cast<size_t>(start) * row_floats * elem_size;
        slice.nbytes = chunk_floats * elem_size;

        if (!tensor_to_f32(slice, f32.data(), err)) {
            return false;
        }

        if (plan.transform == WeightTransform::AddOne) {
            add_scalar(f32.data(), chunk_floats, 1.0f);
        }

        if (out_type == GGML_TYPE_F16) {
            f32_to_f16_row(f32.data(), f16.data(), chunk_floats);
            if (std::fwrite(f16.data(), sizeof(uint16_t), chunk_floats, f) != chunk_floats) {
                if (err) *err = "short write for tensor '" + view.name + "'";
                return false;
            }
        } else {
            if (std::fwrite(f32.data(), sizeof(float), chunk_floats, f) != chunk_floats) {
                if (err) *err = "short write for tensor '" + view.name + "'";
                return false;
            }
        }
    }

    return true;
}

void apply_kv(gguf_context * ctx, const KvPair & kv) {
    switch (kv.type) {
        case KvPair::Type::U32:  gguf_set_val_u32(ctx, kv.key.c_str(), kv.u32); break;
        case KvPair::Type::I32:  gguf_set_val_i32(ctx, kv.key.c_str(), kv.i32); break;
        case KvPair::Type::F32:  gguf_set_val_f32(ctx, kv.key.c_str(), kv.f32); break;
        case KvPair::Type::Bool: gguf_set_val_bool(ctx, kv.key.c_str(), kv.b); break;
        case KvPair::Type::Str:  gguf_set_val_str(ctx, kv.key.c_str(), kv.str.c_str()); break;
    }
}

void apply_vocab(gguf_context * ctx, const Vocab & vocab) {
    gguf_set_val_str(ctx, "tokenizer.ggml.model", vocab.model.c_str());
    gguf_set_val_str(ctx, "tokenizer.ggml.pre", vocab.pre.c_str());

    std::vector<const char *> token_ptrs;
    token_ptrs.reserve(vocab.tokens.size());
    for (const auto & t : vocab.tokens) {
        token_ptrs.push_back(t.c_str());
    }
    gguf_set_arr_str(ctx, "tokenizer.ggml.tokens", token_ptrs.data(), token_ptrs.size());

    gguf_set_arr_data(ctx, "tokenizer.ggml.token_type", GGUF_TYPE_INT32,
                      vocab.token_types.data(), vocab.token_types.size());

    if (!vocab.merges.empty()) {
        std::vector<const char *> merge_ptrs;
        merge_ptrs.reserve(vocab.merges.size());
        for (const auto & m : vocab.merges) {
            merge_ptrs.push_back(m.c_str());
        }
        gguf_set_arr_str(ctx, "tokenizer.ggml.merges", merge_ptrs.data(), merge_ptrs.size());
    }

    if (vocab.bos_id >= 0) gguf_set_val_u32(ctx, "tokenizer.ggml.bos_token_id", vocab.bos_id);
    if (vocab.eos_id >= 0) gguf_set_val_u32(ctx, "tokenizer.ggml.eos_token_id", vocab.eos_id);
    if (vocab.eot_id >= 0) gguf_set_val_u32(ctx, "tokenizer.ggml.eot_token_id", vocab.eot_id);
    if (vocab.unk_id >= 0) gguf_set_val_u32(ctx, "tokenizer.ggml.unknown_token_id", vocab.unk_id);
    if (vocab.pad_id >= 0) gguf_set_val_u32(ctx, "tokenizer.ggml.padding_token_id", vocab.pad_id);

    if (vocab.has_add_bos) gguf_set_val_bool(ctx, "tokenizer.ggml.add_bos_token", vocab.add_bos);
    if (vocab.has_add_eos) gguf_set_val_bool(ctx, "tokenizer.ggml.add_eos_token", vocab.add_eos);
    if (vocab.has_add_space_prefix) {
        gguf_set_val_bool(ctx, "tokenizer.ggml.add_space_prefix", vocab.add_space_prefix);
    }

    if (!vocab.chat_template.empty()) {
        gguf_set_val_str(ctx, "tokenizer.chat_template", vocab.chat_template.c_str());
    }

    if (vocab.sampling_temp >= 0.0f) {
        gguf_set_val_f32(ctx, "general.sampling.temp", vocab.sampling_temp);
    }
    if (vocab.sampling_top_p >= 0.0f) {
        gguf_set_val_f32(ctx, "general.sampling.top_p", vocab.sampling_top_p);
    }
    if (vocab.sampling_top_k >= 0) {
        gguf_set_val_u32(ctx, "general.sampling.top_k",
                         static_cast<uint32_t>(vocab.sampling_top_k));
    }
}

struct GgufDeleter {
    void operator()(gguf_context * c) const { if (c) gguf_free(c); }
};
struct GgmlDeleter {
    void operator()(ggml_context * c) const { if (c) ggml_free(c); }
};

/// Stage 1: safetensors -> F16 GGUF.
bool write_f16_gguf(
    const ConvertOptions & options,
    const SafetensorsModel & model,
    const HParams & hp,
    const std::vector<TensorPlan> & plan,
    const Vocab & vocab,
    const std::string & out_path,
    const ProgressFn & progress,
    std::string * err) {

    std::unique_ptr<gguf_context, GgufDeleter> gguf(gguf_init_empty());

    for (const KvPair & kv : build_metadata(hp, options.model_name)) {
        apply_kv(gguf.get(), kv);
    }
    apply_vocab(gguf.get(), vocab);
    gguf_set_val_u32(gguf.get(), "general.file_type",
                     static_cast<uint32_t>(LLAMA_FTYPE_MOSTLY_F16));
    gguf_set_val_u32(gguf.get(), "general.quantization_version", 2);

    // Metadata-only ggml context: tensors carry shape and type but no storage.
    ggml_init_params init{};
    init.mem_size = ggml_tensor_overhead() * (plan.size() + 1);
    init.no_alloc = true;
    std::unique_ptr<ggml_context, GgmlDeleter> meta(ggml_init(init));
    if (meta == nullptr) {
        if (err) *err = "could not allocate tensor metadata context";
        return false;
    }

    std::vector<ggml_type> out_types;
    out_types.reserve(plan.size());
    uint64_t total_bytes = 0;

    for (const TensorPlan & tp : plan) {
        const TensorView * view = model.find(tp.source_name);
        if (view == nullptr) {
            if (err) *err = "tensor disappeared between planning and writing: " + tp.source_name;
            return false;
        }
        if (view->shape.empty() || view->shape.size() > 2) {
            if (err) {
                *err = "tensor '" + tp.source_name + "' has " +
                       std::to_string(view->shape.size()) +
                       " dimensions; only 1-D and 2-D are supported";
            }
            return false;
        }

        // Norms and biases stay F32: llama.cpp expects full precision there, and
        // quantizing them costs quality for almost no space.
        const ggml_type type =
            view->shape.size() == 1 ? GGML_TYPE_F32 : GGML_TYPE_F16;
        out_types.push_back(type);

        // GGUF stores dimensions in ggml order, which is the reverse of HF's.
        // An HF [out_features, in_features] weight becomes ne = {in, out}.
        ggml_tensor * t;
        if (view->shape.size() == 1) {
            t = ggml_new_tensor_1d(meta.get(), type, view->shape[0]);
        } else {
            t = ggml_new_tensor_2d(meta.get(), type, view->shape[1], view->shape[0]);
        }
        if (t == nullptr) {
            if (err) *err = "could not create metadata for tensor " + tp.gguf_name;
            return false;
        }
        ggml_set_name(t, tp.gguf_name.c_str());
        gguf_add_tensor(gguf.get(), t);

        total_bytes += ggml_nbytes(t);
    }

    ConvertProgress p;
    p.stage = ConvertStage::WritingMetadata;
    p.tensors_total = static_cast<int64_t>(plan.size());
    p.bytes_total = total_bytes;
    if (!progress(p)) {
        if (err) *err = "cancelled";
        return false;
    }

    FILE * f = std::fopen(out_path.c_str(), "wb");
    if (f == nullptr) {
        if (err) *err = "cannot open " + out_path + " for writing: " + std::strerror(errno);
        return false;
    }

    // Writes header, KV block and tensor info, then pads to the data alignment.
    if (!gguf_write_to_file_ptr(gguf.get(), f, /*only_meta=*/true)) {
        std::fclose(f);
        if (err) *err = "failed to write GGUF metadata";
        return false;
    }

    p.stage = ConvertStage::WritingTensors;

    for (size_t i = 0; i < plan.size(); ++i) {
        const TensorPlan & tp = plan[i];
        const TensorView * view = model.find(tp.source_name);

        if (!write_tensor_data(f, *view, tp, hp, out_types[i], err)) {
            std::fclose(f);
            return false;
        }
        // gguf_add_tensor computed each offset assuming this padding.
        if (!pad_to_alignment(f, GGUF_DEFAULT_ALIGNMENT)) {
            std::fclose(f);
            if (err) *err = "failed to pad tensor data";
            return false;
        }

        p.tensors_done = static_cast<int64_t>(i + 1);
        p.bytes_done += view->nbytes;
        p.detail = tp.gguf_name;
        if (!progress(p)) {
            std::fclose(f);
            if (err) *err = "cancelled";
            return false;
        }
    }

    if (std::fclose(f) != 0) {
        if (err) *err = "failed to flush " + out_path;
        return false;
    }
    return true;
}

}  // namespace

const char * convert_stage_name(ConvertStage stage) {
    switch (stage) {
        case ConvertStage::Reading:         return "Reading checkpoint";
        case ConvertStage::WritingMetadata: return "Writing metadata";
        case ConvertStage::WritingTensors:  return "Converting tensors";
        case ConvertStage::Quantizing:      return "Quantizing";
        case ConvertStage::Finalising:      return "Finalising";
    }
    return "Working";
}

uint64_t estimated_working_bytes(uint64_t checkpoint_bytes, const std::string & quant_type) {
    // The checkpoint is bf16 or f16, so parameters are roughly bytes / 2.
    const uint64_t params = checkpoint_bytes / 2;
    const uint64_t f16_bytes = params * 2;
    const auto quant_bytes =
        static_cast<uint64_t>(static_cast<double>(params) * bits_per_weight(quant_type) / 8.0);
    // Both exist on disk at once, plus a margin for metadata and slack.
    return f16_bytes + quant_bytes + (64ull << 20);
}

ConvertResult convert_model(const ConvertOptions & options, const ProgressFn & progress) {
    ConvertResult result;

    const auto report = [&](ConvertStage stage, const std::string & detail) -> bool {
        ConvertProgress p;
        p.stage = stage;
        p.detail = detail;
        return progress(p);
    };

    if (!report(ConvertStage::Reading, "reading configuration")) {
        result.error = "cancelled";
        return result;
    }

    // --- inputs -------------------------------------------------------------
    std::string config_json;
    if (!read_text_file(join_path(options.model_dir, "config.json"), &config_json)) {
        result.error = "no config.json in " + options.model_dir;
        return result;
    }

    HParams hp;
    if (!parse_hparams(config_json, &hp, &result.error)) {
        return result;
    }
    result.arch = gguf_arch_name(hp.arch);

    std::string tokenizer_json;
    if (!read_text_file(join_path(options.model_dir, "tokenizer.json"), &tokenizer_json)) {
        result.error =
            "no tokenizer.json in " + options.model_dir +
            ". SentencePiece-only checkpoints, which ship tokenizer.model instead, "
            "cannot be converted yet.";
        return result;
    }

    std::string tokenizer_config_json;
    std::string generation_config_json;
    read_text_file(join_path(options.model_dir, "tokenizer_config.json"), &tokenizer_config_json);
    read_text_file(join_path(options.model_dir, "generation_config.json"),
                   &generation_config_json);

    // Newer exports put the chat template in its own file rather than inside
    // tokenizer_config.json. Missing it does not fail conversion -- it produces a
    // model that loads and is then prompted as a base model, which for an
    // instruct tune is a large and silent quality loss.
    std::string chat_template;
    read_text_file(join_path(options.model_dir, "chat_template.jinja"), &chat_template);

    Vocab vocab;
    if (!build_vocab(tokenizer_json, tokenizer_config_json, generation_config_json,
                     hp.vocab_size, &vocab, &result.error)) {
        return result;
    }
    if (vocab.chat_template.empty() && !chat_template.empty()) {
        vocab.chat_template = chat_template;
    }
    result.warnings = vocab.warnings;

    auto model = SafetensorsModel::open(options.model_dir, &result.error);
    if (model == nullptr) {
        return result;
    }

    std::vector<TensorPlan> plan;
    if (!build_tensor_plan(hp, model->names(), &plan, &result.error)) {
        return result;
    }

    // --- preflight ----------------------------------------------------------
    ConvertOptions opts = options;
    if (opts.model_name.empty()) {
        opts.model_name = basename_of(options.model_dir);
    }
    const std::string work_path =
        opts.work_path.empty() ? opts.out_path + ".f16" : opts.work_path;

    const uint64_t checkpoint_bytes = model->total_tensor_bytes();
    const uint64_t needed = estimated_working_bytes(checkpoint_bytes, opts.quant_type);

    // statvfs the output's directory; the file itself does not exist yet.
    const size_t slash = opts.out_path.find_last_of('/');
    const std::string out_dir =
        slash == std::string::npos ? std::string(".") : opts.out_path.substr(0, slash);
    const uint64_t available = free_disk_bytes(out_dir);

    if (available > 0 && available < needed) {
        result.error =
            "not enough free space: conversion needs about " +
            std::to_string(needed >> 20) + " MiB but only " +
            std::to_string(available >> 20) + " MiB is available";
        return result;
    }

    // --- stage 1 ------------------------------------------------------------
    if (!write_f16_gguf(opts, *model, hp, plan, vocab, work_path, progress, &result.error)) {
        ::unlink(work_path.c_str());
        return result;
    }

    // Release the checkpoint mapping before quantizing: holding several GB of
    // mapped pages while llama.cpp reads the intermediate invites the low-memory
    // killer on a phone.
    model.reset();

    // --- stage 2 ------------------------------------------------------------
    const bool quantize = opts.quant_type != "F16";

    if (quantize) {
        if (!report(ConvertStage::Quantizing, opts.quant_type)) {
            ::unlink(work_path.c_str());
            result.error = "cancelled";
            return result;
        }

        llama_model_quantize_params qparams = llama_model_quantize_default_params();
        qparams.ftype = ftype_from_name(opts.quant_type);
        qparams.nthread = opts.n_threads > 0
                              ? opts.n_threads
                              : static_cast<int32_t>(std::thread::hardware_concurrency());

        const uint32_t rc =
            llama_model_quantize(work_path.c_str(), opts.out_path.c_str(), &qparams);
        if (rc != 0) {
            ::unlink(work_path.c_str());
            ::unlink(opts.out_path.c_str());
            result.error = "quantization failed with code " + std::to_string(rc);
            return result;
        }
        ::unlink(work_path.c_str());
    } else {
        // F16 requested: the intermediate is the deliverable.
        if (std::rename(work_path.c_str(), opts.out_path.c_str()) != 0) {
            result.error = std::string("could not move the converted file into place: ") +
                           std::strerror(errno);
            ::unlink(work_path.c_str());
            return result;
        }
    }

    if (!report(ConvertStage::Finalising, "")) {
        result.error = "cancelled";
        return result;
    }

    struct stat st{};
    if (::stat(opts.out_path.c_str(), &st) == 0) {
        result.output_bytes = static_cast<uint64_t>(st.st_size);
    }

    // Checkpoints are bf16 or f16, so two bytes per parameter.
    result.param_count = static_cast<int64_t>(checkpoint_bytes / 2);

    result.ok = true;
    return result;
}

}  // namespace sruti
