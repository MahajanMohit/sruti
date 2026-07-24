#!/usr/bin/env bash
# Compiles and runs the host-side native tests.
#
# These cover the parts of the native layer that have no JNI or ggml dependency —
# safetensors parsing, tensor mapping, dtype conversion, tokenizer conversion — so
# they run on the build machine in seconds. No device or emulator needed.
#
# Everything is built with ASan and UBSan: this code does pointer arithmetic over
# mmapped multi-gigabyte files, where an off-by-one silently reads a neighbouring
# tensor instead of crashing.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/build/native-tests"
LLAMA="${ROOT}/native/llama.cpp"
CONVERTER="${ROOT}/app/src/main/cpp/converter"

if [[ ! -f "${LLAMA}/vendor/nlohmann/json.hpp" ]]; then
    echo "error: llama.cpp submodule missing. Run: git submodule update --init --recursive" >&2
    exit 1
fi

mkdir -p "${OUT}"

CXX="${CXX:-g++}"

# converter.cpp is excluded: it links ggml and llama, which would turn a
# two-second unit-test run into a full library build. Its orchestration is covered
# end to end by tools/convert_cli, which converts a real checkpoint.
SOURCES=()
for f in "${CONVERTER}"/*.cpp; do
    [[ "$(basename "${f}")" == "converter.cpp" ]] && continue
    SOURCES+=("${f}")
done

INCLUDES=(
    -I"${LLAMA}/vendor"
    -I"${LLAMA}/include"
    -I"${LLAMA}/ggml/include"
)

status=0

for src in "${ROOT}"/app/src/test/cpp/*_test.cpp; do
    name="$(basename "${src}" .cpp)"
    echo "==> ${name}"

    "${CXX}" -std=c++17 -O1 -g -Wall -Wextra -Wno-unused-parameter \
        -fsanitize=address,undefined -fno-omit-frame-pointer \
        "${INCLUDES[@]}" \
        -o "${OUT}/${name}" "${src}" "${SOURCES[@]}"

    "${OUT}/${name}" || status=1
    echo
done

if [[ "${status}" -eq 0 ]]; then
    echo "All native test suites passed."
fi
exit "${status}"
