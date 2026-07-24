#include "safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>

#include <nlohmann/json.hpp>

namespace sruti {

namespace {

using json = nlohmann::json;

// The header must stay small enough that a corrupt length cannot make us try to
// parse gigabytes of tensor data as JSON.
constexpr uint64_t kMaxHeaderBytes = 256ull * 1024 * 1024;

StDType parse_dtype(const std::string & s) {
    if (s == "F64")  return StDType::F64;
    if (s == "F32")  return StDType::F32;
    if (s == "F16")  return StDType::F16;
    if (s == "BF16") return StDType::BF16;
    if (s == "I64")  return StDType::I64;
    if (s == "I32")  return StDType::I32;
    if (s == "I16")  return StDType::I16;
    if (s == "I8")   return StDType::I8;
    if (s == "U8")   return StDType::U8;
    if (s == "BOOL") return StDType::BOOL;
    return StDType::Unknown;
}

bool ends_with(const std::string & s, const std::string & suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string join_path(const std::string & dir, const std::string & name) {
    if (dir.empty()) return name;
    if (dir.back() == '/') return dir + name;
    return dir + "/" + name;
}

bool file_exists(const std::string & path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

std::string read_text_file(const std::string & path, std::string * err) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        if (err) *err = "cannot open " + path + ": " + std::strerror(errno);
        return {};
    }
    std::string out;
    char buf[8192];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) {
        out.append(buf, n);
    }
    std::fclose(f);
    return out;
}

}  // namespace

const char * st_dtype_name(StDType dtype) {
    switch (dtype) {
        case StDType::F64:  return "F64";
        case StDType::F32:  return "F32";
        case StDType::F16:  return "F16";
        case StDType::BF16: return "BF16";
        case StDType::I64:  return "I64";
        case StDType::I32:  return "I32";
        case StDType::I16:  return "I16";
        case StDType::I8:   return "I8";
        case StDType::U8:   return "U8";
        case StDType::BOOL: return "BOOL";
        default:            return "UNKNOWN";
    }
}

size_t st_dtype_size(StDType dtype) {
    switch (dtype) {
        case StDType::F64:
        case StDType::I64:  return 8;
        case StDType::F32:
        case StDType::I32:  return 4;
        case StDType::F16:
        case StDType::BF16:
        case StDType::I16:  return 2;
        case StDType::I8:
        case StDType::U8:
        case StDType::BOOL: return 1;
        default:            return 0;
    }
}

int64_t TensorView::numel() const {
    int64_t n = 1;
    for (int64_t d : shape) {
        n *= d;
    }
    return shape.empty() ? 0 : n;
}

// --- SafetensorsFile --------------------------------------------------------

SafetensorsFile::~SafetensorsFile() {
    if (map_ != nullptr && map_ != MAP_FAILED) {
        ::munmap(map_, map_size_);
    }
    if (fd_ >= 0) {
        ::close(fd_);
    }
}

std::unique_ptr<SafetensorsFile> SafetensorsFile::open(
    const std::string & path, std::string * err) {

    std::unique_ptr<SafetensorsFile> self(new SafetensorsFile());
    self->path_ = path;

    self->fd_ = ::open(path.c_str(), O_RDONLY);
    if (self->fd_ < 0) {
        if (err) *err = "cannot open " + path + ": " + std::strerror(errno);
        return nullptr;
    }

    struct stat st{};
    if (::fstat(self->fd_, &st) != 0) {
        if (err) *err = "cannot stat " + path + ": " + std::strerror(errno);
        return nullptr;
    }
    self->map_size_ = static_cast<size_t>(st.st_size);

    if (self->map_size_ < 8) {
        if (err) *err = path + " is too small to be a safetensors file";
        return nullptr;
    }

    self->map_ = ::mmap(nullptr, self->map_size_, PROT_READ, MAP_PRIVATE, self->fd_, 0);
    if (self->map_ == MAP_FAILED) {
        if (err) *err = "cannot mmap " + path + ": " + std::strerror(errno);
        self->map_ = nullptr;
        return nullptr;
    }

    const auto * base = static_cast<const uint8_t *>(self->map_);

    uint64_t header_len = 0;
    std::memcpy(&header_len, base, sizeof(header_len));  // little-endian on all targets

    if (header_len == 0 || header_len > kMaxHeaderBytes ||
        header_len + 8 > self->map_size_) {
        if (err) {
            *err = path + ": implausible header length " + std::to_string(header_len);
        }
        return nullptr;
    }

    const uint8_t * data_base = base + 8 + header_len;
    const size_t data_size = self->map_size_ - 8 - header_len;

    json header;
    try {
        header = json::parse(base + 8, base + 8 + header_len);
    } catch (const std::exception & e) {
        if (err) *err = path + ": malformed header JSON: " + e.what();
        return nullptr;
    }
    if (!header.is_object()) {
        if (err) *err = path + ": header is not a JSON object";
        return nullptr;
    }

    for (const auto & [name, entry] : header.items()) {
        if (name == "__metadata__") {
            self->metadata_json_ = entry.dump();
            continue;
        }
        if (!entry.is_object()) {
            if (err) *err = path + ": entry '" + name + "' is not an object";
            return nullptr;
        }

        TensorView view;
        view.name = name;

        const auto dtype_it = entry.find("dtype");
        const auto shape_it = entry.find("shape");
        const auto offs_it  = entry.find("data_offsets");
        if (dtype_it == entry.end() || shape_it == entry.end() || offs_it == entry.end()) {
            if (err) *err = path + ": entry '" + name + "' is missing required fields";
            return nullptr;
        }

        view.dtype = parse_dtype(dtype_it->get<std::string>());
        if (view.dtype == StDType::Unknown) {
            if (err) {
                *err = path + ": tensor '" + name + "' has unsupported dtype " +
                       dtype_it->get<std::string>();
            }
            return nullptr;
        }

        for (const auto & dim : *shape_it) {
            view.shape.push_back(dim.get<int64_t>());
        }

        if (!offs_it->is_array() || offs_it->size() != 2) {
            if (err) *err = path + ": tensor '" + name + "' has malformed data_offsets";
            return nullptr;
        }
        const uint64_t begin = (*offs_it)[0].get<uint64_t>();
        const uint64_t end   = (*offs_it)[1].get<uint64_t>();

        // Reject out-of-range ranges rather than handing out pointers past the
        // mapping; a truncated download must fail loudly, not segfault later.
        if (end < begin || end > data_size) {
            if (err) {
                *err = path + ": tensor '" + name + "' offsets [" +
                       std::to_string(begin) + ", " + std::to_string(end) +
                       ") exceed the data region of " + std::to_string(data_size) + " bytes";
            }
            return nullptr;
        }

        view.nbytes = static_cast<size_t>(end - begin);
        view.data = data_base + begin;

        const size_t expected = static_cast<size_t>(view.numel()) * st_dtype_size(view.dtype);
        if (expected != view.nbytes) {
            if (err) {
                *err = path + ": tensor '" + name + "' declares " +
                       std::to_string(view.nbytes) + " bytes but its shape and dtype imply " +
                       std::to_string(expected);
            }
            return nullptr;
        }

        self->index_.emplace(name, self->tensors_.size());
        self->tensors_.push_back(std::move(view));
    }

    return self;
}

const TensorView * SafetensorsFile::find(const std::string & name) const {
    const auto it = index_.find(name);
    return it == index_.end() ? nullptr : &tensors_[it->second];
}

// --- SafetensorsModel -------------------------------------------------------

std::unique_ptr<SafetensorsModel> SafetensorsModel::open(
    const std::string & dir, std::string * err) {

    std::vector<std::string> shard_files;

    const std::string index_path = join_path(dir, "model.safetensors.index.json");
    if (file_exists(index_path)) {
        std::string read_err;
        const std::string text = read_text_file(index_path, &read_err);
        if (text.empty()) {
            if (err) *err = read_err.empty() ? (index_path + " is empty") : read_err;
            return nullptr;
        }

        json index;
        try {
            index = json::parse(text);
        } catch (const std::exception & e) {
            if (err) *err = index_path + ": malformed JSON: " + e.what();
            return nullptr;
        }

        const auto map_it = index.find("weight_map");
        if (map_it == index.end() || !map_it->is_object()) {
            if (err) *err = index_path + " has no weight_map object";
            return nullptr;
        }

        // Deduplicate while preserving first-seen order, so shards load in the
        // order the index introduces them.
        for (const auto & [_, shard] : map_it->items()) {
            const std::string file = shard.get<std::string>();
            if (std::find(shard_files.begin(), shard_files.end(), file) == shard_files.end()) {
                shard_files.push_back(file);
            }
        }
    } else if (file_exists(join_path(dir, "model.safetensors"))) {
        shard_files.push_back("model.safetensors");
    } else {
        if (err) {
            *err = dir + " contains neither model.safetensors nor "
                         "model.safetensors.index.json";
        }
        return nullptr;
    }

    std::unique_ptr<SafetensorsModel> self(new SafetensorsModel());

    for (const std::string & file : shard_files) {
        if (!ends_with(file, ".safetensors")) {
            if (err) *err = "index references a non-safetensors file: " + file;
            return nullptr;
        }

        auto shard = SafetensorsFile::open(join_path(dir, file), err);
        if (shard == nullptr) {
            return nullptr;
        }

        for (const TensorView & view : shard->tensors()) {
            if (self->index_.count(view.name) != 0) {
                if (err) *err = "tensor '" + view.name + "' appears in more than one shard";
                return nullptr;
            }
            self->index_.emplace(view.name, &view);
            self->names_.push_back(view.name);
        }

        self->files_.push_back(std::move(shard));
    }

    return self;
}

const TensorView * SafetensorsModel::find(const std::string & name) const {
    const auto it = index_.find(name);
    return it == index_.end() ? nullptr : it->second;
}

uint64_t SafetensorsModel::total_tensor_bytes() const {
    uint64_t total = 0;
    for (const auto & file : files_) {
        for (const TensorView & view : file->tensors()) {
            total += view.nbytes;
        }
    }
    return total;
}

}  // namespace sruti
