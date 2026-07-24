#pragma once

// Orchestration: a Hugging Face safetensors checkpoint directory becomes a
// quantized GGUF file, entirely on the device.
//
// Two stages:
//
//   1. safetensors -> F16 GGUF. Ours. This is where architecture mapping, the
//      RoPE permutation, and vocabulary conversion happen.
//   2. F16 GGUF -> Q4_K_M (or similar) via llama.cpp's own `llama_model_quantize`.
//      Reusing it means per-tensor type selection matches desktop `llama-quantize`
//      exactly, rather than being an approximation of it.
//
// The intermediate costs disk transiently — roughly 2 bytes per parameter — and is
// deleted once stage 2 succeeds.

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace sruti {

struct ConvertOptions {
    /// Directory holding config.json, tokenizer.json and the safetensors shards.
    std::string model_dir;
    /// Destination .gguf path.
    std::string out_path;
    /// One of "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q4_0", "Q8_0", "F16".
    std::string quant_type = "Q4_K_M";
    /// Name recorded in general.name; defaults to the directory name.
    std::string model_name;
    /// Zero selects the hardware concurrency.
    int n_threads = 0;
    /// Where the F16 intermediate is written. Defaults to out_path + ".f16".
    std::string work_path;
};

enum class ConvertStage {
    Reading,
    WritingMetadata,
    WritingTensors,
    Quantizing,
    Finalising,
};

struct ConvertProgress {
    ConvertStage stage = ConvertStage::Reading;
    /// Free-text detail, e.g. the tensor currently being written.
    std::string detail;
    int64_t tensors_done = 0;
    int64_t tensors_total = 0;
    uint64_t bytes_done = 0;
    uint64_t bytes_total = 0;
};

/// Return false to cancel; the converter then aborts and removes partial output.
using ProgressFn = std::function<bool(const ConvertProgress &)>;

struct ConvertResult {
    bool ok = false;
    std::string error;
    /// Non-fatal issues, chiefly from vocabulary conversion.
    std::vector<std::string> warnings;

    std::string arch;
    int64_t param_count = 0;
    uint64_t output_bytes = 0;
};

/// Human-readable label for a stage, for UI.
const char * convert_stage_name(ConvertStage stage);

/// Bytes of free disk the conversion needs, given the checkpoint size.
///
/// The F16 intermediate and the quantized output are on disk simultaneously, so
/// this is larger than the final file. Preflight before starting: running out
/// part-way wastes however long has already been spent.
uint64_t estimated_working_bytes(uint64_t checkpoint_bytes, const std::string & quant_type);

/// Runs the full conversion. Never throws; failures land in `ConvertResult::error`.
ConvertResult convert_model(const ConvertOptions & options, const ProgressFn & progress);

}  // namespace sruti
