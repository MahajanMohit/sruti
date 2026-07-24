// Host-side driver for the on-device converter.
//
// Runs exactly the same code the Android app runs, so a checkpoint that converts
// here converts on the phone. Its purpose is validation: convert a real model with
// this, convert the same model with llama.cpp's convert_hf_to_gguf.py, and compare
// the two GGUF files tensor by tensor. That comparison is the only trustworthy
// check that the RoPE permutation and vocabulary conversion are right.

#include <cstdio>
#include <cstring>
#include <string>

#include "converter.h"

namespace {

void usage(const char * argv0) {
    std::fprintf(stderr,
                 "usage: %s <model-dir> <out.gguf> [quant-type] [threads]\n"
                 "\n"
                 "  model-dir   directory with config.json, tokenizer.json and\n"
                 "              the .safetensors shards\n"
                 "  out.gguf    destination file\n"
                 "  quant-type  Q4_K_M (default), Q4_K_S, Q5_K_M, Q4_0, Q8_0, Q6_K, F16\n"
                 "  threads     quantization threads (default: hardware concurrency)\n",
                 argv0);
}

const char * human_bytes(uint64_t bytes, char * buf, size_t buf_size) {
    const char * units[] = {"B", "KiB", "MiB", "GiB"};
    double value = static_cast<double>(bytes);
    size_t unit = 0;
    while (value >= 1024.0 && unit + 1 < 4) {
        value /= 1024.0;
        ++unit;
    }
    std::snprintf(buf, buf_size, "%.1f %s", value, units[unit]);
    return buf;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        usage(argv[0]);
        return 2;
    }

    sruti::ConvertOptions options;
    options.model_dir = argv[1];
    options.out_path = argv[2];
    options.quant_type = argc > 3 ? argv[3] : "Q4_K_M";
    options.n_threads = argc > 4 ? std::atoi(argv[4]) : 0;

    sruti::ConvertStage last_stage = sruti::ConvertStage::Finalising;
    int64_t last_percent = -1;

    const auto progress = [&](const sruti::ConvertProgress & p) -> bool {
        if (p.stage != last_stage) {
            last_stage = p.stage;
            last_percent = -1;
            std::fprintf(stderr, "\n%s", sruti::convert_stage_name(p.stage));
            if (!p.detail.empty() && p.stage == sruti::ConvertStage::Quantizing) {
                std::fprintf(stderr, " to %s", p.detail.c_str());
            }
            std::fprintf(stderr, "\n");
        }

        if (p.tensors_total > 0) {
            const int64_t percent = p.tensors_done * 100 / p.tensors_total;
            if (percent != last_percent) {
                last_percent = percent;
                std::fprintf(stderr, "\r  %3lld%%  %lld/%lld tensors  %-40.40s",
                             static_cast<long long>(percent),
                             static_cast<long long>(p.tensors_done),
                             static_cast<long long>(p.tensors_total),
                             p.detail.c_str());
                std::fflush(stderr);
            }
        }
        return true;
    };

    const sruti::ConvertResult result = sruti::convert_model(options, progress);
    std::fprintf(stderr, "\n");

    for (const std::string & warning : result.warnings) {
        std::fprintf(stderr, "warning: %s\n", warning.c_str());
    }

    if (!result.ok) {
        std::fprintf(stderr, "error: %s\n", result.error.c_str());
        return 1;
    }

    char buf[64];
    std::fprintf(stderr,
                 "\nConverted %s\n  architecture: %s\n  parameters:   %.2f B\n  output:       %s\n",
                 options.out_path.c_str(),
                 result.arch.c_str(),
                 static_cast<double>(result.param_count) / 1e9,
                 human_bytes(result.output_bytes, buf, sizeof(buf)));
    return 0;
}
