#pragma once

// Dtype conversion and weight transforms.
//
// No ggml dependency, so the RoPE permutation — the transform whose failure mode
// is a model that loads and generates garbage — can be tested against a reference
// implementation on the host.
//
// Every read from checkpoint memory goes through memcpy. Safetensors data begins
// immediately after a variable-length JSON header, so tensor data is routinely
// misaligned; a `const float *` dereference is undefined behaviour and faults on
// some ARM configurations.

#include <cstddef>
#include <cstdint>
#include <string>

#include "safetensors.h"

namespace sruti {

// --- scalar conversions -----------------------------------------------------

/// bfloat16 is simply the top 16 bits of an IEEE-754 float.
float bf16_to_f32(uint16_t v);

/// IEEE-754 half precision, including subnormals, infinities and NaN.
float f16_to_f32(uint16_t v);

/// Round-to-nearest-even, saturating to infinity on overflow.
uint16_t f32_to_f16(float v);

// --- tensor conversion ------------------------------------------------------

/// Converts a whole tensor to f32.
///
/// `out` must have room for `view.numel()` floats. Integer dtypes are rejected:
/// nothing in a supported checkpoint stores weights as integers, and silently
/// coercing them would hide a real problem.
bool tensor_to_f32(const TensorView & view, float * out, std::string * err);

/// Converts a row of f32 to f16 in place-compatible form.
void f32_to_f16_row(const float * src, uint16_t * dst, size_t n);

// --- weight transforms ------------------------------------------------------

/// Un-permutes a Llama-family Q or K projection.
///
/// Llama checkpoints interleave RoPE pairs as two contiguous halves per head,
/// while GGUF expects them adjacent. This is the numpy expression
///
///     w.reshape(n_head, 2, rows // n_head // 2, cols).swapaxes(1, 2).reshape(rows, cols)
///
/// done in place via a scratch buffer. `n_head` is the query head count for Q and
/// the key/value head count for K — under GQA they differ, and using the wrong one
/// corrupts K silently.
///
/// Returns false when `rows` is not divisible by `n_head * 2`.
bool permute_rope(float * data, int64_t rows, int64_t cols, int64_t n_head);

/// Adds `value` to every element. Gemma stores RMSNorm weights as w but
/// implements (1 + w).
void add_scalar(float * data, size_t n, float value);

}  // namespace sruti
