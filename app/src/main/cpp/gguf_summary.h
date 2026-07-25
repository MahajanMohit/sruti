#pragma once

#include <string>

namespace sruti {

// Reads a GGUF's metadata without loading its weights.
//
// Deliberately not llama_model_load_from_file: that maps a gigabyte of tensors to
// answer questions the header already contains, and this needs to be cheap enough
// to run while the user is scrolling a list.
//
// Returns tab-separated `label\tvalue` rows, one per line, in display order, or an
// empty string when the file cannot be read as GGUF. Text rather than a struct
// because every value is only ever rendered, and a struct would need a Kotlin
// twin kept in sync by hand.
std::string gguf_summary(const char * path);

}  // namespace sruti
