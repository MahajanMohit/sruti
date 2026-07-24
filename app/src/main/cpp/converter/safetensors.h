#pragma once

// Reader for the safetensors container format.
//
// Layout: an 8-byte little-endian u64 header length, a JSON header describing
// every tensor's dtype/shape/byte-range, then the raw tensor data. Offsets in the
// header are relative to the start of the data region.
//
// The file is mmapped rather than read: a 1-2B model in bf16 is several GB, and
// the whole point is to stream it through quantization a tensor at a time without
// ever holding it all in memory. The kernel pages in what is touched and evicts
// the rest under pressure.

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace sruti {

enum class StDType {
    F64,
    F32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U8,
    BOOL,
    Unknown,
};

const char * st_dtype_name(StDType dtype);
size_t st_dtype_size(StDType dtype);

/// A tensor's metadata plus a pointer into the mapped region. Non-owning.
struct TensorView {
    std::string name;
    StDType dtype = StDType::Unknown;
    /// Row-major, in the order the checkpoint declares (HF convention).
    std::vector<int64_t> shape;

    /// Raw bytes inside the mapping.
    ///
    /// IMPORTANT: this is NOT guaranteed to be naturally aligned. The data region
    /// begins immediately after a variable-length JSON header, so a tensor of
    /// floats can start on an odd address. Casting to `const float *` and
    /// dereferencing is undefined behaviour and can fault on ARM — read through
    /// `load_f32` / `std::memcpy` instead.
    const uint8_t * data = nullptr;
    size_t nbytes = 0;

    int64_t numel() const;
    bool empty() const { return data == nullptr; }
};

/// A single .safetensors file, mmapped for the object's lifetime.
class SafetensorsFile {
public:
    ~SafetensorsFile();

    SafetensorsFile(const SafetensorsFile &) = delete;
    SafetensorsFile & operator=(const SafetensorsFile &) = delete;

    /// Returns nullptr and sets `err` on failure.
    static std::unique_ptr<SafetensorsFile> open(const std::string & path, std::string * err);

    const std::vector<TensorView> & tensors() const { return tensors_; }

    /// Returns nullptr when absent.
    const TensorView * find(const std::string & name) const;

    /// Raw contents of the header's optional "__metadata__" object.
    const std::string & metadata_json() const { return metadata_json_; }

    const std::string & path() const { return path_; }

private:
    SafetensorsFile() = default;

    std::string path_;
    int fd_ = -1;
    void * map_ = nullptr;
    size_t map_size_ = 0;

    std::vector<TensorView> tensors_;
    std::unordered_map<std::string, size_t> index_;
    std::string metadata_json_;
};

/// A checkpoint directory, which may hold a single file or many shards.
///
/// Sharded checkpoints ship a `model.safetensors.index.json` mapping tensor names
/// to shard filenames. When that is absent this falls back to the single-file
/// `model.safetensors`, then to any *.safetensors present.
class SafetensorsModel {
public:
    /// Returns nullptr and sets `err` on failure.
    static std::unique_ptr<SafetensorsModel> open(const std::string & dir, std::string * err);

    /// Returns nullptr when absent.
    const TensorView * find(const std::string & name) const;

    /// Every tensor name across all shards, in checkpoint order.
    const std::vector<std::string> & names() const { return names_; }

    size_t shard_count() const { return files_.size(); }

    /// Sum of every tensor's byte size. Not the file size — headers are excluded.
    uint64_t total_tensor_bytes() const;

private:
    std::vector<std::unique_ptr<SafetensorsFile>> files_;
    std::unordered_map<std::string, const TensorView *> index_;
    std::vector<std::string> names_;
};

}  // namespace sruti
