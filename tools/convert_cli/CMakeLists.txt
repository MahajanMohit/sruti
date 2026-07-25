cmake_minimum_required(VERSION 3.22)

project(sruti_convert_cli CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(SRUTI_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(LLAMA_DIR "${SRUTI_ROOT}/native/llama.cpp")
set(CONVERTER_DIR "${SRUTI_ROOT}/app/src/main/cpp/converter")

if(NOT EXISTS "${LLAMA_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "llama.cpp submodule missing. Run: git submodule update --init --recursive")
endif()

set(LLAMA_CURL          OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)

add_subdirectory("${LLAMA_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/llama.cpp" EXCLUDE_FROM_ALL)

file(GLOB CONVERTER_SOURCES "${CONVERTER_DIR}/*.cpp")

add_executable(sruti-convert
    main.cpp
    ${CONVERTER_SOURCES}
)

target_include_directories(sruti-convert PRIVATE
    "${CONVERTER_DIR}"
    "${LLAMA_DIR}/include"
    "${LLAMA_DIR}/ggml/include"
    "${LLAMA_DIR}/vendor"
)

target_link_libraries(sruti-convert PRIVATE llama ggml)

# Loads a converted GGUF and generates text, so a conversion can be checked
# end to end rather than only by comparing bytes.
add_executable(sruti-run
    run.cpp
    "${SRUTI_ROOT}/app/src/main/cpp/gguf_summary.cpp"
)

target_include_directories(sruti-run PRIVATE
    "${SRUTI_ROOT}/app/src/main/cpp"
    "${LLAMA_DIR}/include"
    "${LLAMA_DIR}/ggml/include"
)

target_link_libraries(sruti-run PRIVATE llama ggml)
