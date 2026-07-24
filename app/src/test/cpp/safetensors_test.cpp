// Host-side tests for the safetensors reader.
//
// Synthesises checkpoint files on disk rather than using fixtures, so the tests
// stay self-contained and can cover malformed input that no real model produces.

#include "../../main/cpp/converter/safetensors.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "test_util.h"

using namespace sruti;

namespace {

// Builds a safetensors file: 8-byte header length, JSON header, then data.
void write_safetensors(
    const std::string & path,
    const std::string & header_json,
    const std::vector<uint8_t> & data) {

    FILE * f = std::fopen(path.c_str(), "wb");
    ASSERT_TRUE(f != nullptr, "opened output file");

    const uint64_t header_len = header_json.size();
    std::fwrite(&header_len, sizeof(header_len), 1, f);
    std::fwrite(header_json.data(), 1, header_json.size(), f);
    if (!data.empty()) {
        std::fwrite(data.data(), 1, data.size(), f);
    }
    std::fclose(f);
}

std::vector<uint8_t> float_bytes(const std::vector<float> & values) {
    std::vector<uint8_t> out(values.size() * sizeof(float));
    std::memcpy(out.data(), values.data(), out.size());
    return out;
}

/// Reads element `i` from a tensor's raw bytes.
///
/// Via memcpy deliberately: the data region starts after a variable-length JSON
/// header, so tensor data is frequently misaligned and a `const float *`
/// dereference would be undefined behaviour.
float read_f32(const TensorView * t, size_t i) {
    float value;
    std::memcpy(&value, t->data + i * sizeof(float), sizeof(float));
    return value;
}

void test_reads_single_tensor() {
    TempDir dir;
    const std::string path = dir.file("model.safetensors");

    const std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    write_safetensors(
        path,
        R"({"weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}})",
        float_bytes(values));

    std::string err;
    auto file = SafetensorsFile::open(path, &err);
    ASSERT_TRUE(file != nullptr, std::string("opened file: ") + err);
    ASSERT_EQ(file->tensors().size(), size_t(1), "one tensor present");

    const TensorView * w = file->find("weight");
    ASSERT_TRUE(w != nullptr, "found tensor by name");
    ASSERT_TRUE(w->dtype == StDType::F32, "dtype is F32");
    ASSERT_EQ(w->shape.size(), size_t(2), "shape has two dims");
    ASSERT_EQ(w->shape[0], int64_t(2), "dim 0");
    ASSERT_EQ(w->shape[1], int64_t(3), "dim 1");
    ASSERT_EQ(w->numel(), int64_t(6), "element count");
    ASSERT_EQ(w->nbytes, size_t(24), "byte count");

    bool contents_match = true;
    for (size_t i = 0; i < values.size(); ++i) {
        if (read_f32(w, i) != values[i]) contents_match = false;
    }
    ASSERT_TRUE(contents_match, "tensor data readable through the mapping");
}

void test_reads_multiple_tensors_with_offsets() {
    TempDir dir;
    const std::string path = dir.file("model.safetensors");

    std::vector<uint8_t> data = float_bytes({1.0f, 2.0f});
    const std::vector<uint8_t> second = float_bytes({7.0f, 8.0f, 9.0f});
    data.insert(data.end(), second.begin(), second.end());

    write_safetensors(
        path,
        R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},)"
        R"("b":{"dtype":"F32","shape":[3],"data_offsets":[8,20]}})",
        data);

    std::string err;
    auto file = SafetensorsFile::open(path, &err);
    ASSERT_TRUE(file != nullptr, std::string("opened file: ") + err);

    const TensorView * b = file->find("b");
    ASSERT_TRUE(b != nullptr, "found second tensor");
    ASSERT_EQ(read_f32(b, 0), 7.0f, "second tensor starts at its declared offset");
    ASSERT_EQ(read_f32(b, 2), 9.0f, "second tensor reads to its end");
}

void test_parses_metadata() {
    TempDir dir;
    const std::string path = dir.file("model.safetensors");
    write_safetensors(
        path,
        R"({"__metadata__":{"format":"pt"},)"
        R"("w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})",
        float_bytes({1.0f}));

    std::string err;
    auto file = SafetensorsFile::open(path, &err);
    ASSERT_TRUE(file != nullptr, std::string("opened file: ") + err);
    ASSERT_EQ(file->tensors().size(), size_t(1), "__metadata__ is not counted as a tensor");
    ASSERT_TRUE(file->metadata_json().find("\"pt\"") != std::string::npos,
                "metadata captured");
}

void test_supports_all_float_dtypes() {
    TempDir dir;
    const std::string path = dir.file("model.safetensors");
    // 2 bf16 + 2 f16 + 1 f32 = 4 + 4 + 4 = 12 bytes
    std::vector<uint8_t> data(12, 0);
    write_safetensors(
        path,
        R"({"bf":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},)"
        R"("h":{"dtype":"F16","shape":[2],"data_offsets":[4,8]},)"
        R"("f":{"dtype":"F32","shape":[1],"data_offsets":[8,12]}})",
        data);

    std::string err;
    auto file = SafetensorsFile::open(path, &err);
    ASSERT_TRUE(file != nullptr, std::string("opened file: ") + err);
    ASSERT_TRUE(file->find("bf")->dtype == StDType::BF16, "BF16 recognised");
    ASSERT_TRUE(file->find("h")->dtype == StDType::F16, "F16 recognised");
    ASSERT_TRUE(file->find("f")->dtype == StDType::F32, "F32 recognised");
}

void test_rejects_offsets_past_end_of_data() {
    TempDir dir;
    const std::string path = dir.file("model.safetensors");
    // Declares 40 bytes of data but only 8 follow — a truncated download.
    write_safetensors(
        path,
        R"({"w":{"dtype":"F32","shape":[10],"data_offsets":[0,40]}})",
        float_bytes({1.0f, 2.0f}));

    std::string err;
    auto file = SafetensorsFile::open(path, &err);
    ASSERT_TRUE(file == nullptr, "truncated file rejected rather than mapped");
    ASSERT_TRUE(err.find("exceed the data region") != std::string::npos,
                "error names the cause");
}

void test_rejects_shape_dtype_mismatch() {
    TempDir dir;
    const std::string path = dir.file("model.safetensors");
    // shape [2] of F32 is 8 bytes, but the range spans 12.
    write_safetensors(
        path,
        R"({"w":{"dtype":"F32","shape":[2],"data_offsets":[0,12]}})",
        std::vector<uint8_t>(12, 0));

    std::string err;
    auto file = SafetensorsFile::open(path, &err);
    ASSERT_TRUE(file == nullptr, "inconsistent shape/dtype/size rejected");
}

void test_rejects_unknown_dtype() {
    TempDir dir;
    const std::string path = dir.file("model.safetensors");
    write_safetensors(
        path,
        R"({"w":{"dtype":"FP8_E4M3","shape":[4],"data_offsets":[0,4]}})",
        std::vector<uint8_t>(4, 0));

    std::string err;
    auto file = SafetensorsFile::open(path, &err);
    ASSERT_TRUE(file == nullptr, "unsupported dtype rejected");
    ASSERT_TRUE(err.find("unsupported dtype") != std::string::npos, "error names the dtype");
}

void test_rejects_implausible_header_length() {
    TempDir dir;
    const std::string path = dir.file("model.safetensors");

    FILE * f = std::fopen(path.c_str(), "wb");
    const uint64_t bogus = 0xFFFFFFFFFFFFFFFFull;
    std::fwrite(&bogus, sizeof(bogus), 1, f);
    std::fwrite("garbage", 1, 7, f);
    std::fclose(f);

    std::string err;
    auto file = SafetensorsFile::open(path, &err);
    ASSERT_TRUE(file == nullptr, "absurd header length rejected");
    ASSERT_TRUE(err.find("implausible header length") != std::string::npos,
                "error names the cause");
}

void test_rejects_missing_file() {
    std::string err;
    auto file = SafetensorsFile::open("/nonexistent/model.safetensors", &err);
    ASSERT_TRUE(file == nullptr, "missing file rejected");
    ASSERT_TRUE(!err.empty(), "error message provided");
}

void test_opens_single_file_model_directory() {
    TempDir dir;
    write_safetensors(
        dir.file("model.safetensors"),
        R"({"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})",
        float_bytes({3.0f, 4.0f}));

    std::string err;
    auto model = SafetensorsModel::open(dir.path(), &err);
    ASSERT_TRUE(model != nullptr, std::string("opened directory: ") + err);
    ASSERT_EQ(model->shard_count(), size_t(1), "single shard");
    ASSERT_TRUE(model->find("w") != nullptr, "tensor reachable through the model");
    ASSERT_EQ(model->total_tensor_bytes(), uint64_t(8), "total bytes");
}

void test_opens_sharded_model_via_index() {
    TempDir dir;

    write_safetensors(
        dir.file("model-00001-of-00002.safetensors"),
        R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})",
        float_bytes({1.0f, 2.0f}));
    write_safetensors(
        dir.file("model-00002-of-00002.safetensors"),
        R"({"b":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})",
        float_bytes({3.0f, 4.0f}));

    write_text_file(
        dir.file("model.safetensors.index.json"),
        R"({"weight_map":{"a":"model-00001-of-00002.safetensors",)"
        R"("b":"model-00002-of-00002.safetensors"}})");

    std::string err;
    auto model = SafetensorsModel::open(dir.path(), &err);
    ASSERT_TRUE(model != nullptr, std::string("opened sharded model: ") + err);
    ASSERT_EQ(model->shard_count(), size_t(2), "two shards opened");
    ASSERT_EQ(model->names().size(), size_t(2), "tensors from both shards indexed");

    const TensorView * b = model->find("b");
    ASSERT_TRUE(b != nullptr, "tensor from the second shard is reachable");
    // Pointers must survive the shards being moved into the model's own storage.
    ASSERT_EQ(read_f32(b, 1), 4.0f,
              "shard data still readable after ownership transfer");
}

void test_rejects_directory_without_checkpoint() {
    TempDir dir;
    std::string err;
    auto model = SafetensorsModel::open(dir.path(), &err);
    ASSERT_TRUE(model == nullptr, "empty directory rejected");
}

}  // namespace

int main() {
    test_reads_single_tensor();
    test_reads_multiple_tensors_with_offsets();
    test_parses_metadata();
    test_supports_all_float_dtypes();
    test_rejects_offsets_past_end_of_data();
    test_rejects_shape_dtype_mismatch();
    test_rejects_unknown_dtype();
    test_rejects_implausible_header_length();
    test_rejects_missing_file();
    test_opens_single_file_model_directory();
    test_opens_sharded_model_via_index();
    test_rejects_directory_without_checkpoint();

    return test_summary("safetensors");
}
