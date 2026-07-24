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

# Phase 0 is CPU-only. Adreno OpenCL and Hexagon come later, behind flags, once
# there are baseline numbers to compare them against.
set(GGML_OPENMP     OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE  OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL      OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)

add_subdirectory("${LLAMA_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/llama.cpp" EXCLUDE_FROM_ALL)

add_library(sruti_llm SHARED
    sruti_llm.cpp
)

target_include_directories(sruti_llm PRIVATE
    "${LLAMA_DIR}/include"
    "${LLAMA_DIR}/ggml/include"
)

# i8mm and dotprod are present on every Snapdragon 8-class part we target and are
# a large win for Q4 matmul. minSdk 31 guarantees armv8.2-a.
target_compile_options(sruti_llm PRIVATE
    -O3
    -ffast-math
    -march=armv8.2-a+dotprod+i8mm
)

find_library(log-lib log)

target_link_libraries(sruti_llm
    llama
    ggml
    ${log-lib}
    android
)
