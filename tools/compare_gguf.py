#!/usr/bin/env python3
"""Compares two GGUF files tensor by tensor.

The point of this script is to answer one question: does the on-device converter
produce the same file as llama.cpp's own convert_hf_to_gguf.py? That comparison is
the only trustworthy check on the RoPE permutation and the vocabulary, both of
which fail silently — a model with either one wrong still loads and still emits
fluent-looking text.

Usage:
    python3 tools/compare_gguf.py <ours.gguf> <reference.gguf>

Exits non-zero when the files differ in any way that would change inference.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "native" / "llama.cpp" / "gguf-py"))

import numpy as np  # noqa: E402
from gguf import GGUFReader  # noqa: E402


# Keys that legitimately differ: provenance and naming, not inference behaviour.
IGNORED_KEYS = {
    "general.name",
    "general.basename",
    "general.size_label",
    "general.finetune",
    "general.license",
    "general.languages",
    "general.tags",
    "general.organization",
    "general.repo_url",
    "general.type",
    "general.file_type",
    "general.quantization_version",
    "general.version",
    "general.description",
    "general.url",
    "general.author",
    "general.source.url",
    "general.base_model.count",
    # Synthetic field from GGUFReader. It counts every key including the ignored
    # provenance ones, so it necessarily differs and says nothing about inference.
    "GGUF.kv_count",
}


def load(path: str):
    reader = GGUFReader(path, "r")
    tensors = {t.name: t for t in reader.tensors}
    fields = {}
    for key, field in reader.fields.items():
        try:
            fields[key] = field.contents()
        except Exception:  # noqa: BLE001 - some field types are not readable
            fields[key] = "<unreadable>"
    return fields, tensors


def compare_metadata(ours: dict, theirs: dict) -> list[str]:
    problems: list[str] = []

    our_keys = set(ours) - IGNORED_KEYS
    their_keys = set(theirs) - IGNORED_KEYS

    for key in sorted(their_keys - our_keys):
        problems.append(f"MISSING KEY   {key} = {theirs[key]!r}")
    for key in sorted(our_keys - their_keys):
        problems.append(f"EXTRA KEY     {key} = {ours[key]!r}")

    for key in sorted(our_keys & their_keys):
        a, b = ours[key], theirs[key]
        if isinstance(a, float) and isinstance(b, float):
            if abs(a - b) > 1e-9 * max(1.0, abs(b)):
                problems.append(f"VALUE DIFFERS {key}: ours={a!r} ref={b!r}")
        elif a != b:
            # Long arrays (the token list) get a summarised diff.
            if isinstance(a, list) and isinstance(b, list):
                if len(a) != len(b):
                    problems.append(
                        f"LENGTH DIFFERS {key}: ours={len(a)} ref={len(b)}"
                    )
                else:
                    diffs = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
                    if diffs:
                        head = diffs[:5]
                        detail = ", ".join(
                            f"[{i}] ours={a[i]!r} ref={b[i]!r}" for i in head
                        )
                        problems.append(
                            f"ARRAY DIFFERS {key}: {len(diffs)} of {len(a)} entries; {detail}"
                        )
            else:
                problems.append(f"VALUE DIFFERS {key}: ours={a!r} ref={b!r}")

    return problems


def compare_tensors(ours: dict, theirs: dict) -> tuple[list[str], list[str]]:
    problems: list[str] = []
    notes: list[str] = []

    for name in sorted(set(theirs) - set(ours)):
        problems.append(f"MISSING TENSOR {name}")
    for name in sorted(set(ours) - set(theirs)):
        problems.append(f"EXTRA TENSOR   {name}")

    worst_name, worst_diff = None, 0.0
    exact = 0

    for name in sorted(set(ours) & set(theirs)):
        a, b = ours[name], theirs[name]

        if list(a.shape) != list(b.shape):
            problems.append(
                f"SHAPE DIFFERS {name}: ours={list(a.shape)} ref={list(b.shape)}"
            )
            continue
        if a.tensor_type != b.tensor_type:
            problems.append(
                f"TYPE DIFFERS  {name}: ours={a.tensor_type} ref={b.tensor_type}"
            )
            continue

        x = np.asarray(a.data).ravel()
        y = np.asarray(b.data).ravel()

        if x.dtype == y.dtype and np.array_equal(x, y):
            exact += 1
            continue

        xf = x.astype(np.float32)
        yf = y.astype(np.float32)
        diff = float(np.max(np.abs(xf - yf))) if xf.size else 0.0

        if diff > worst_diff:
            worst_diff, worst_name = diff, name

        # A permutation error shows up as a large difference, not a rounding one.
        if diff > 1e-3:
            problems.append(f"DATA DIFFERS  {name}: max|delta| = {diff:g}")

    notes.append(f"{exact} of {len(set(ours) & set(theirs))} tensors bit-identical")
    if worst_name is not None:
        notes.append(f"largest non-exact difference: {worst_name} = {worst_diff:g}")
    return problems, notes


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2

    ours_path, ref_path = sys.argv[1], sys.argv[2]
    print(f"ours:      {ours_path}")
    print(f"reference: {ref_path}\n")

    our_fields, our_tensors = load(ours_path)
    ref_fields, ref_tensors = load(ref_path)

    meta_problems = compare_metadata(our_fields, ref_fields)
    tensor_problems, notes = compare_tensors(our_tensors, ref_tensors)

    print(f"tensors: ours={len(our_tensors)} reference={len(ref_tensors)}")
    for note in notes:
        print(f"  {note}")
    print()

    if meta_problems:
        print(f"METADATA DIFFERENCES ({len(meta_problems)}):")
        for p in meta_problems:
            print(f"  {p}")
        print()

    if tensor_problems:
        print(f"TENSOR DIFFERENCES ({len(tensor_problems)}):")
        for p in tensor_problems[:40]:
            print(f"  {p}")
        if len(tensor_problems) > 40:
            print(f"  ... and {len(tensor_problems) - 40} more")
        print()

    if not meta_problems and not tensor_problems:
        print("IDENTICAL: no differences that would change inference.")
        return 0

    print(
        f"DIFFERENCES FOUND: {len(meta_problems)} metadata, "
        f"{len(tensor_problems)} tensor"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
