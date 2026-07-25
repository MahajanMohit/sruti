#!/usr/bin/env bash
# Verifies that generated GBNF grammars parse and actually constrain a real model.
#
# The Kotlin unit tests check the grammars' structure, which is necessary but not
# sufficient: a grammar can look right and still be rejected by llama.cpp's parser,
# or be accepted and constrain nothing. This feeds them to an actual model.
#
# Prerequisites:
#   ./gradlew :app:testDebugUnitTest                    # writes app/build/grammars/
#   cmake -S tools/convert_cli -B build/convert-cli -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/convert-cli -j"$(nproc)"
#
# Usage: tools/verify_grammars.sh <model.gguf>
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-}"
RUN="${ROOT}/build/convert-cli/sruti-run"
GRAMMARS="${ROOT}/app/build/grammars"

if [[ -z "${MODEL}" || ! -f "${MODEL}" ]]; then
    echo "usage: $0 <model.gguf>" >&2
    exit 2
fi
if [[ ! -x "${RUN}" ]]; then
    echo "error: ${RUN} not built" >&2
    exit 1
fi
if [[ ! -d "${GRAMMARS}" ]]; then
    echo "error: no grammars in ${GRAMMARS}; run the Kotlin unit tests first" >&2
    exit 1
fi

echo "model: ${MODEL}"
echo

echo "== tool choice =="
echo "Structure is guaranteed by the grammar. Whether the choice is *correct* is a"
echo "property of the model, and small models get it wrong."
echo
"${RUN}" "${MODEL}" \
    "Tools:
read_file(path) - read a text file from disk
set_mode(mode) - change the operating mode
get_time() - get the current time

Task: read the file at /tmp/notes.txt
Which tool? Answer with the tool name only." \
    10 --grammar "${GRAMMARS}/choice.gbnf" 2>/dev/null | tail -2
echo

echo "== argument extraction =="
echo "The narrow question: the tool is already chosen, only the slots need filling."
echo
"${RUN}" "${MODEL}" \
    'Tool: read_file(path, max_lines?) - read a text file
Task: read the file at /tmp/notes.txt
Output the arguments as JSON:' \
    40 --grammar "${GRAMMARS}/read_file.gbnf" 2>/dev/null | tail -2
echo

echo "== enum constraint =="
echo "Values outside the declared set are unreachable, not merely improbable."
echo
"${RUN}" "${MODEL}" \
    'Tool: set_mode(mode) where mode is one of fast, balanced, quality.
Task: switch to the highest quality setting.
Output the arguments as JSON:' \
    24 --grammar "${GRAMMARS}/set_mode.gbnf" 2>/dev/null | tail -2
