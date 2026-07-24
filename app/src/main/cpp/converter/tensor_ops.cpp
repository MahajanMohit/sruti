#include "tensor_ops.h"

#include <cstring>
#include <vector>

namespace sruti {

namespace {

/// Alignment-safe load of a 16-bit word from checkpoint memory.
inline uint16_t load_u16(const uint8_t * p) {
    uint16_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

/// Alignment-safe load of a 32-bit float from checkpoint memory.
inline float load_f32(const uint8_t * p) {
    float v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline double load_f64(const uint8_t * p) {
    double v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

}  // namespace

// --- scalar conversions -----------------------------------------------------

float bf16_to_f32(uint16_t v) {
    const uint32_t bits = static_cast<uint32_t>(v) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

float f16_to_f32(uint16_t v) {
    const uint32_t sign     = static_cast<uint32_t>(v & 0x8000u) << 16;
    const uint32_t exponent = (v >> 10) & 0x1Fu;
    const uint32_t mantissa = v & 0x3FFu;

    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;  // signed zero
        } else {
            // Subnormal: renormalise into the float exponent range.
            uint32_t m = mantissa;
            int32_t e = -1;
            do {
                m <<= 1;
                ++e;
            } while ((m & 0x400u) == 0);
            m &= 0x3FFu;
            bits = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exponent == 0x1Fu) {
        // Infinity or NaN; the mantissa payload carries across unchanged.
        bits = sign | 0x7F800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }

    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

uint16_t f32_to_f16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFFu;

    if (((bits >> 23) & 0xFFu) == 0xFFu) {
        // Preserve infinity, and keep NaN as NaN rather than letting the mantissa
        // truncate to zero and turn it into an infinity.
        if (mantissa != 0) {
            return static_cast<uint16_t>(sign | 0x7E00u);
        }
        return static_cast<uint16_t>(sign | 0x7C00u);
    }

    if (exponent >= 0x1F) {
        return static_cast<uint16_t>(sign | 0x7C00u);  // overflow to infinity
    }

    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<uint16_t>(sign);  // underflows to signed zero
        }
        // Subnormal half: shift the implicit leading 1 into the mantissa.
        mantissa |= 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exponent);
        const uint32_t half = mantissa >> shift;
        // Round to nearest, ties to even.
        const uint32_t remainder = mantissa & ((1u << shift) - 1);
        const uint32_t halfway = 1u << (shift - 1);
        uint32_t rounded = half;
        if (remainder > halfway || (remainder == halfway && (half & 1u) != 0)) {
            ++rounded;
        }
        return static_cast<uint16_t>(sign | rounded);
    }

    uint32_t result = (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13);
    const uint32_t remainder = mantissa & 0x1FFFu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (result & 1u) != 0)) {
        ++result;  // may carry into the exponent, which is the correct behaviour
    }
    return static_cast<uint16_t>(sign | result);
}

// --- tensor conversion ------------------------------------------------------

bool tensor_to_f32(const TensorView & view, float * out, std::string * err) {
    const int64_t n = view.numel();

    switch (view.dtype) {
        case StDType::F32:
            for (int64_t i = 0; i < n; ++i) {
                out[i] = load_f32(view.data + i * 4);
            }
            return true;

        case StDType::F16:
            for (int64_t i = 0; i < n; ++i) {
                out[i] = f16_to_f32(load_u16(view.data + i * 2));
            }
            return true;

        case StDType::BF16:
            for (int64_t i = 0; i < n; ++i) {
                out[i] = bf16_to_f32(load_u16(view.data + i * 2));
            }
            return true;

        case StDType::F64:
            for (int64_t i = 0; i < n; ++i) {
                out[i] = static_cast<float>(load_f64(view.data + i * 8));
            }
            return true;

        default:
            if (err) {
                *err = "tensor '" + view.name + "' has dtype " +
                       st_dtype_name(view.dtype) +
                       ", which is not a float type; refusing to coerce it";
            }
            return false;
    }
}

void f32_to_f16_row(const float * src, uint16_t * dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = f32_to_f16(src[i]);
    }
}

// --- weight transforms ------------------------------------------------------

bool permute_rope(float * data, int64_t rows, int64_t cols, int64_t n_head) {
    if (n_head <= 0 || rows <= 0 || cols <= 0) {
        return false;
    }
    if (rows % (n_head * 2) != 0) {
        return false;
    }

    const int64_t head_dim = rows / n_head;
    const int64_t half = head_dim / 2;

    // The permutation moves whole rows, so a row-sized scratch is not enough;
    // build the result then copy back.
    std::vector<float> scratch(static_cast<size_t>(rows) * static_cast<size_t>(cols));

    for (int64_t h = 0; h < n_head; ++h) {
        for (int64_t a = 0; a < 2; ++a) {
            for (int64_t b = 0; b < half; ++b) {
                // Source layout: (n_head, 2, half, cols)
                const int64_t src_row = h * head_dim + a * half + b;
                // After swapaxes(1, 2): (n_head, half, 2, cols)
                const int64_t dst_row = h * head_dim + b * 2 + a;

                std::memcpy(&scratch[static_cast<size_t>(dst_row) * cols],
                            &data[static_cast<size_t>(src_row) * cols],
                            static_cast<size_t>(cols) * sizeof(float));
            }
        }
    }

    std::memcpy(data, scratch.data(), scratch.size() * sizeof(float));
    return true;
}

void add_scalar(float * data, size_t n, float value) {
    for (size_t i = 0; i < n; ++i) {
        data[i] += value;
    }
}

}  // namespace sruti
