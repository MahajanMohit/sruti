#!/usr/bin/env bash
# Compiles and runs the host-side native tests.
#
# These cover pure C++ logic with no JNI or llama.cpp dependency, so they run on
# the build machine in about a second — no device or emulator needed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/build/native-tests"
mkdir -p "${OUT}"

CXX="${CXX:-g++}"
status=0

for src in "${ROOT}"/app/src/test/cpp/*_test.cpp; do
    name="$(basename "${src}" .cpp)"
    echo "==> ${name}"
    "${CXX}" -std=c++17 -O1 -Wall -Wextra -fsanitize=address,undefined \
        -o "${OUT}/${name}" "${src}"
    "${OUT}/${name}" || status=1
    echo
done

exit "${status}"
