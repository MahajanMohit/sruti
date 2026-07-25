cmake_minimum_required(VERSION 3.22.1)

project(sruti_llm CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Repo root is four levels up from app/src/main/cpp.
get_filename_component(SRUTI_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
set(LLAMA_DIR "${SRUTI_ROOT}/native/llama.cpp")

if(NOT EXISTS "${LLAMA_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "llama.cpp submodule missing at ${LLAMA_DIR}. "
        "Run: git submodule update --init --recursive")
endif()

# CPU-only for now. GPU and NPU backends come later, behind flags, once there are
# baseline numbers to compare them against.
#
# GGML_CPU_ALL_VARIANTS builds ggml's per-feature-set CPU backends for Android
# (armv8.0 through armv9.2) and picks one at runtime from the features the device
# reports. That is what lets a single APK run correctly on a budget part and still
# use dot-product and int8 matmul instructions on a flagship.
set(GGML_OPENMP     OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE  OFF CACHE BOOL "" FORCE)

# Runtime CPU dispatch. Requires the backends to be separate shared objects.
set(GGML_CPU_ALL_VARIANTS ON  CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL       ON  CACHE BOOL "" FORCE)

# GPU offload is not built yet -- see docs/gpu.md.
#
# Not an oversight and not a stub: llama.cpp's Vulkan backend needs a glslc newer
# than the one the NDK ships (its shaders use GL_KHR_cooperative_matrix, which the
# bundled compiler does not parse), so enabling it means vendoring shaderc and
# making that work in CI too. OpenCL is the better fit for Android -- its kernels
# are compiled by the driver at runtime, so no shader toolchain is needed at build
# time -- and is the intended route.
#
# The runtime side is already in place: the GPU-layer count is a user setting that
# defaults to zero, so a backend can be switched on without touching this file's
# callers.

add_subdirectory("${LLAMA_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/llama.cpp" EXCLUDE_FROM_ALL)

file(GLOB CONVERTER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/converter/*.cpp")

add_library(sruti_llm SHARED
    sruti_llm.cpp
    sruti_chat.cpp
    sruti_converter.cpp
    gguf_summary.cpp
    ${CONVERTER_SOURCES}
)

target_include_directories(sruti_llm PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/converter"
    "${LLAMA_DIR}/include"
    "${LLAMA_DIR}/ggml/include"
    "${LLAMA_DIR}/vendor"
)

# Baseline arm64 only, deliberately.
#
# Naming CPU features here would apply them to every source in this target, and
# the compiler is then free to emit those instructions anywhere — including on
# devices that lack them. i8mm in particular is optional in ARMv8.2 and mandatory
# only from ARMv8.6, so requiring it would turn any mid-range part into a SIGILL
# on first inference. Feature selection belongs to ggml, which dispatches on what
# the CPU actually reports at runtime.
#
# -ffast-math is likewise absent on purpose. The converter must produce
# bit-identical output to the reference implementation, and reassociation and
# flush-to-zero are precisely what would break that. Inference performance lives
# in ggml's kernels, not in this glue.
target_compile_options(sruti_llm PRIVATE -O3)

find_library(log-lib log)

target_link_libraries(sruti_llm
    llama
    ggml
    ${log-lib}
    android
)
