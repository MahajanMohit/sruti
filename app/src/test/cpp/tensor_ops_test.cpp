// Host-side tests for dtype conversion and weight transforms.
//
// The RoPE permutation is checked against an independently written reference that
// walks explicit 4-D strides, rather than against the row-copy loop the
// implementation uses. Two implementations of the same expression agreeing is
// worth far more than one implementation agreeing with itself.

#include "../../main/cpp/converter/tensor_ops.h"

#include <cmath>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "test_util.h"

using namespace sruti;

namespace {

/// numpy equivalent of
///   w.reshape(n_head, 2, half, cols).swapaxes(1, 2).reshape(rows, cols)
/// written with explicit strides so it shares no structure with the implementation.
std::vector<float> reference_permute(
    const std::vector<float> & in, int64_t rows, int64_t cols, int64_t n_head) {

    const int64_t half = rows / n_head / 2;
    std::vector<float> out(in.size());

    for (int64_t h = 0; h < n_head; ++h) {
        for (int64_t b = 0; b < half; ++b) {
            for (int64_t a = 0; a < 2; ++a) {
                for (int64_t c = 0; c < cols; ++c) {
                    // destination indexed as (n_head, half, 2, cols)
                    const int64_t d = ((h * half + b) * 2 + a) * cols + c;
                    // source indexed as (n_head, 2, half, cols)
                    const int64_t s = ((h * 2 + a) * half + b) * cols + c;
                    out[d] = in[s];
                }
            }
        }
    }
    return out;
}

std::vector<float> ramp(size_t n) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = static_cast<float>(i);
    }
    return v;
}

bool vectors_equal(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

/// Builds a TensorView over a deliberately misaligned buffer.
///
/// `storage` is sized to hold `offset` padding bytes plus the payload; the view
/// points at `storage.data() + offset`, reproducing what a safetensors header of
/// odd length does to real checkpoints.
TensorView misaligned_view(
    std::vector<uint8_t> & storage,
    const void * payload,
    size_t payload_bytes,
    StDType dtype,
    std::vector<int64_t> shape,
    size_t offset) {

    storage.assign(offset + payload_bytes, 0);
    std::memcpy(storage.data() + offset, payload, payload_bytes);

    TensorView view;
    view.name = "test";
    view.dtype = dtype;
    view.shape = std::move(shape);
    view.data = storage.data() + offset;
    view.nbytes = payload_bytes;
    return view;
}

// --- bf16 -------------------------------------------------------------------

void test_bf16_is_the_top_half_of_a_float() {
    // 1.0f is 0x3F800000, so its bf16 form is 0x3F80.
    ASSERT_EQ(bf16_to_f32(0x3F80), 1.0f, "bf16 1.0");
    ASSERT_EQ(bf16_to_f32(0x4000), 2.0f, "bf16 2.0");
    ASSERT_EQ(bf16_to_f32(0xBF80), -1.0f, "bf16 -1.0");
    ASSERT_EQ(bf16_to_f32(0x0000), 0.0f, "bf16 zero");
}

// --- f16 --------------------------------------------------------------------

void test_f16_normal_values() {
    ASSERT_EQ(f16_to_f32(0x3C00), 1.0f, "f16 1.0");
    ASSERT_EQ(f16_to_f32(0x4000), 2.0f, "f16 2.0");
    ASSERT_EQ(f16_to_f32(0xC000), -2.0f, "f16 -2.0");
    ASSERT_EQ(f16_to_f32(0x3800), 0.5f, "f16 0.5");
    ASSERT_EQ(f16_to_f32(0x0000), 0.0f, "f16 +0");
    ASSERT_EQ(f16_to_f32(0x8000), -0.0f, "f16 -0");
}

void test_f16_subnormals() {
    // Smallest positive subnormal half is 2^-24.
    const float smallest = f16_to_f32(0x0001);
    ASSERT_TRUE(std::fabs(smallest - 5.9604645e-8f) < 1e-14f, "f16 smallest subnormal");
    // Largest subnormal, just below the smallest normal.
    const float largest_sub = f16_to_f32(0x03FF);
    ASSERT_TRUE(largest_sub > 0.0f && largest_sub < 6.104e-5f, "f16 largest subnormal");
}

void test_f16_infinities_and_nan() {
    ASSERT_TRUE(std::isinf(f16_to_f32(0x7C00)) && f16_to_f32(0x7C00) > 0,
                "f16 +infinity");
    ASSERT_TRUE(std::isinf(f16_to_f32(0xFC00)) && f16_to_f32(0xFC00) < 0,
                "f16 -infinity");
    ASSERT_TRUE(std::isnan(f16_to_f32(0x7E00)), "f16 NaN");
}

void test_f32_to_f16_round_trips_representable_values() {
    const float values[] = {0.0f, -0.0f, 1.0f, -1.0f, 0.5f, 2.0f, 1024.0f, -0.25f};
    bool all_ok = true;
    for (float v : values) {
        if (f16_to_f32(f32_to_f16(v)) != v) all_ok = false;
    }
    ASSERT_TRUE(all_ok, "exactly representable values survive f32->f16->f32");
}

void test_f32_to_f16_saturates_and_preserves_nan() {
    ASSERT_EQ(f32_to_f16(1e30f), uint16_t(0x7C00), "overflow becomes +infinity");
    ASSERT_EQ(f32_to_f16(-1e30f), uint16_t(0xFC00), "negative overflow becomes -infinity");
    ASSERT_TRUE(std::isnan(f16_to_f32(f32_to_f16(std::nanf("")))),
                "NaN stays NaN rather than collapsing to infinity");
    ASSERT_EQ(f32_to_f16(1e-30f), uint16_t(0x0000), "underflow becomes zero");
}

void test_f32_to_f16_rounds_to_nearest_even() {
    // 2049 is exactly halfway between two representable halves at this magnitude.
    const float rounded = f16_to_f32(f32_to_f16(2049.0f));
    ASSERT_EQ(rounded, 2048.0f, "halfway case rounds to even");
}

// --- tensor conversion ------------------------------------------------------

void test_converts_f32_from_a_misaligned_pointer() {
    // The case UBSan caught: checkpoint data starting on an odd address.
    const float payload[] = {1.5f, -2.5f, 3.25f};
    std::vector<uint8_t> storage;
    const TensorView view = misaligned_view(
        storage, payload, sizeof(payload), StDType::F32, {3}, /*offset=*/1);

    std::vector<float> out(3);
    std::string err;
    ASSERT_TRUE(tensor_to_f32(view, out.data(), &err), std::string("converted: ") + err);
    ASSERT_EQ(out[0], 1.5f, "misaligned f32 element 0");
    ASSERT_EQ(out[1], -2.5f, "misaligned f32 element 1");
    ASSERT_EQ(out[2], 3.25f, "misaligned f32 element 2");
}

void test_converts_bf16_from_a_misaligned_pointer() {
    const uint16_t payload[] = {0x3F80, 0x4000, 0xBF80};  // 1.0, 2.0, -1.0
    std::vector<uint8_t> storage;
    const TensorView view = misaligned_view(
        storage, payload, sizeof(payload), StDType::BF16, {3}, /*offset=*/3);

    std::vector<float> out(3);
    std::string err;
    ASSERT_TRUE(tensor_to_f32(view, out.data(), &err), std::string("converted: ") + err);
    ASSERT_EQ(out[0], 1.0f, "misaligned bf16 element 0");
    ASSERT_EQ(out[2], -1.0f, "misaligned bf16 element 2");
}

void test_converts_f16_tensor() {
    const uint16_t payload[] = {0x3C00, 0x4000};  // 1.0, 2.0
    std::vector<uint8_t> storage;
    const TensorView view = misaligned_view(
        storage, payload, sizeof(payload), StDType::F16, {2}, /*offset=*/1);

    std::vector<float> out(2);
    std::string err;
    ASSERT_TRUE(tensor_to_f32(view, out.data(), &err), std::string("converted: ") + err);
    ASSERT_EQ(out[0], 1.0f, "f16 tensor element 0");
    ASSERT_EQ(out[1], 2.0f, "f16 tensor element 1");
}

void test_refuses_integer_dtypes() {
    const int32_t payload[] = {1, 2};
    std::vector<uint8_t> storage;
    const TensorView view = misaligned_view(
        storage, payload, sizeof(payload), StDType::I32, {2}, /*offset=*/0);

    std::vector<float> out(2);
    std::string err;
    ASSERT_TRUE(!tensor_to_f32(view, out.data(), &err), "integer dtype rejected");
    ASSERT_TRUE(err.find("not a float type") != std::string::npos,
                "error explains the refusal");
}

// --- the RoPE permutation ---------------------------------------------------

void test_permute_matches_the_reference_implementation() {
    struct Case { int64_t rows, cols, n_head; };
    // Shapes drawn from real models: Llama 3.2 1B has 32 q heads and 8 kv heads
    // with head_dim 64, so Q is 2048 rows and K is 512.
    const Case cases[] = {
        {8, 1, 2}, {8, 3, 2}, {16, 4, 4}, {64, 8, 8}, {2048, 16, 32}, {512, 16, 8},
    };

    bool all_match = true;
    for (const Case & c : cases) {
        std::vector<float> data = ramp(static_cast<size_t>(c.rows * c.cols));
        const std::vector<float> expected = reference_permute(data, c.rows, c.cols, c.n_head);

        if (!permute_rope(data.data(), c.rows, c.cols, c.n_head)) {
            all_match = false;
            continue;
        }
        if (!vectors_equal(data, expected)) all_match = false;
    }
    ASSERT_TRUE(all_match, "permutation matches the reference across realistic shapes");
}

void test_permute_is_a_bijection() {
    // Every input value must appear exactly once. A permutation that duplicates or
    // drops rows would still produce a loadable file.
    const int64_t rows = 64, cols = 4, n_head = 8;
    std::vector<float> data = ramp(static_cast<size_t>(rows * cols));
    ASSERT_TRUE(permute_rope(data.data(), rows, cols, n_head), "permutation applied");

    std::set<float> seen(data.begin(), data.end());
    ASSERT_EQ(seen.size(), size_t(rows * cols), "no element duplicated or lost");
}

void test_permute_known_small_case() {
    // 4 rows, 1 col, 1 head: head_dim 4, half 2.
    // Source rows (a, b): (0,0)=0 (0,1)=1 (1,0)=2 (1,1)=3
    // Destination row = b*2 + a, so: row0=(a0,b0)=0, row1=(a1,b0)=2,
    //                                row2=(a0,b1)=1, row3=(a1,b1)=3
    std::vector<float> data = {0.0f, 1.0f, 2.0f, 3.0f};
    ASSERT_TRUE(permute_rope(data.data(), 4, 1, 1), "permutation applied");

    const std::vector<float> expected = {0.0f, 2.0f, 1.0f, 3.0f};
    ASSERT_TRUE(vectors_equal(data, expected), "interleaved halves become adjacent pairs");
}

void test_permute_preserves_column_contents() {
    // Rows move as units; values within a row must never be reordered.
    const int64_t rows = 8, cols = 3, n_head = 2;
    std::vector<float> data = ramp(static_cast<size_t>(rows * cols));
    permute_rope(data.data(), rows, cols, n_head);

    bool rows_intact = true;
    for (int64_t r = 0; r < rows; ++r) {
        const float first = data[static_cast<size_t>(r * cols)];
        for (int64_t c = 0; c < cols; ++c) {
            if (data[static_cast<size_t>(r * cols + c)] != first + static_cast<float>(c)) {
                rows_intact = false;
            }
        }
    }
    ASSERT_TRUE(rows_intact, "row contents move together");
}

void test_permute_differs_between_q_and_k_under_gqa() {
    // The GQA trap: Q uses head_count, K uses head_count_kv. Using the same value
    // for both corrupts K in a way nothing downstream detects.
    const int64_t rows = 32, cols = 2;
    std::vector<float> as_q = ramp(static_cast<size_t>(rows * cols));
    std::vector<float> as_k = as_q;

    ASSERT_TRUE(permute_rope(as_q.data(), rows, cols, /*n_head=*/8), "Q permuted");
    ASSERT_TRUE(permute_rope(as_k.data(), rows, cols, /*n_head=*/2), "K permuted");
    ASSERT_TRUE(!vectors_equal(as_q, as_k),
                "different head counts genuinely produce different layouts");
}

void test_permute_rejects_indivisible_shapes() {
    std::vector<float> data = ramp(12);
    // 12 rows cannot be split into 5 heads of an even head_dim.
    ASSERT_TRUE(!permute_rope(data.data(), 12, 1, 5), "indivisible row count rejected");
    ASSERT_TRUE(!permute_rope(data.data(), 12, 1, 0), "zero heads rejected");
    ASSERT_TRUE(!permute_rope(data.data(), 0, 1, 1), "empty tensor rejected");
}

// --- add_scalar -------------------------------------------------------------

void test_add_scalar_shifts_every_element() {
    std::vector<float> data = {0.0f, -1.0f, 2.5f};
    add_scalar(data.data(), data.size(), 1.0f);
    ASSERT_EQ(data[0], 1.0f, "zero shifted");
    ASSERT_EQ(data[1], 0.0f, "negative shifted");
    ASSERT_EQ(data[2], 3.5f, "fractional shifted");
}

}  // namespace

int main() {
    test_bf16_is_the_top_half_of_a_float();

    test_f16_normal_values();
    test_f16_subnormals();
    test_f16_infinities_and_nan();
    test_f32_to_f16_round_trips_representable_values();
    test_f32_to_f16_saturates_and_preserves_nan();
    test_f32_to_f16_rounds_to_nearest_even();

    test_converts_f32_from_a_misaligned_pointer();
    test_converts_bf16_from_a_misaligned_pointer();
    test_converts_f16_tensor();
    test_refuses_integer_dtypes();

    test_permute_matches_the_reference_implementation();
    test_permute_is_a_bijection();
    test_permute_known_small_case();
    test_permute_preserves_column_contents();
    test_permute_differs_between_q_and_k_under_gqa();
    test_permute_rejects_indivisible_shapes();

    test_add_scalar_shifts_every_element();

    return test_summary("tensor_ops");
}
