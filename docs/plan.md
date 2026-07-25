# Sruti — On-Device LLM Runtime + Agent for Android

## Context

`mahajanmohit/sruti` is an empty repo. The goal is a native Android app for a
modern arm64 Android device that:

1. Pulls models **from GitHub / Hugging Face in safetensors form** and runs them locally
   — not just pre-baked `.task`/`.litertlm` bundles the way Google AI Edge Gallery does.
2. Targets the **1–2B parameter class**, which genuinely fits in phone RAM and thermals.
3. Goes beyond a chat box — an **agent that actually does things on the device**, in the
   spirit of Claude Code, driven by the local model.
4. Is **visually and haptically refined** to an Apple-tier standard.

### Decisions taken

| Decision | Choice |
|---|---|
| Distribution | **Sideload / GitHub Releases** — not Play Store |
| Inference | **Strictly 100% on-device.** No LLM ever reached over the network |
| v1 scope | **On-device safetensors → GGUF converter ships in v1** |
| Elevated capability | **Termux shell only.** No Shizuku, no AccessibilityService, no root |

---

## Verdict: yes, with two corrections to the premise

**Buildable.** Nothing here needs a research breakthrough. But two parts of the idea, as
stated, need reshaping before they become an engineering plan.

### ✅ What is straightforwardly true

- A **1–2B model at Q4 is comfortable** on a modern arm64 phone. Weights land around
  0.7–1.3 GB plus KV cache; on 12–16 GB devices RAM is not the binding constraint.
- Published figures put **Llama 3.2 3B at ~10 tok/s** and **Llama 3.1 8B at ~5 tok/s on
  the Hexagon NPU** on Snapdragon 8 Elite. A 1–2B model at Q4 should land well above that
  — expect **20–40 tok/s decode** CPU-side, better on GPU/NPU. Every one of these is a
  hypothesis to benchmark on *your* device in Phase 0, not a promise.
- **Fetching models from GitHub/HF at runtime is fine.** Resumable downloads, integrity
  checks, a real model manager. No platform obstacle.
- The **refinement bar is reachable** with Compose + Material 3 Expressive, spring-physics
  motion, custom haptics, and predictive back (mandatory at API 36 anyway).
- **Sideload distribution removes the policy ceiling** — no Play review to design around.

### ⚠️ Correction 1 — "run safetensors" needs a mechanism

Safetensors is a **weight container, not an executable graph**. It stores named tensors
and nothing about how to run them. No mainstream mobile runtime ingests it directly:
llama.cpp still requires GGUF conversion as of 2026, MLC compiles per-target, ExecuTorch
needs `.pte`, MediaPipe needs `.task`, Cactus uses a proprietary `.cact`. So "runs
safetensors" must mean one of two things:

- **(A) Convert on-device** — read the safetensors header, map tensor names for a
  whitelist of architectures, quantize, emit GGUF. The user points the app at any HF repo
  and it becomes runnable *on the phone*, with no desktop step anywhere in the loop.
- **(B) Write a safetensors-native executor** — implement the transformer forward pass
  against mmapped safetensors. Far more work, far slower (bf16 weights, no quantization),
  worse on every axis.

**(A) is the right call and it is exactly the differentiator you want.** AI Edge Gallery
cannot open an arbitrary HF repo; Sruti can. It is also bounded work, because llama.cpp
already exposes the two hard primitives as C APIs — `ggml_quantize_chunk` for quantization
and the `gguf_*` writer for output. What we build is the metadata bridge: `config.json` →
GGUF KV pairs, HF tensor names → GGUF names, the Llama-family Q/K RoPE permutation, and
`tokenizer.json` → GGUF vocab. Scoped to Llama / Qwen / Gemma / Phi this is roughly
~1.5k lines of C++, not the 7k-line Python script that covers 100+ architectures.

> Do **not** attempt `convert_hf_to_gguf.py` under Chaquopy. It imports torch and
> transformers; there is no usable Android torch wheel and the APK cost is prohibitive.

### ⚠️ Correction 2 — a 1–2B model cannot be an autonomous agent, but the *harness* can

This is the load-bearing risk of the project, and the all-local decision makes it the
central engineering problem rather than something a cloud call can paper over.

The evidence is consistent: sub-7B models **emit malformed tool calls once tasks go past a
single step**, and are dependable only for short, constrained outputs. Some 1–2B models
score well on single-turn function-calling benchmarks — LFM2.5-1.2B is reported above
0.900 agent score — but *single-turn* is the operative word. Claude Code's autonomy comes
from a frontier model holding a long plan in working memory. A 1.5B model will not do
that, and no prompt changes it.

Since there is no cloud fallback, **all multi-step competence must live in code**:

- **Grammar-constrained decoding.** llama.cpp GBNF grammars make tool calls syntactically
  valid *by construction*. Malformed-JSON stops being a failure mode — it becomes
  structurally impossible. This is the single highest-leverage decision in the project.
- **The harness owns the loop, not the model.** A typed state machine drives
  plan → confirm → execute → observe. The model is asked one narrow question per step
  ("which tool?", "extract these arguments") — never "plan and execute this task."
- **Retrieval-narrowed tool surface.** The model sees 3–5 candidate tools per step,
  selected by embedding retrieval, never the full registry. Accuracy collapses with tool
  count at this model size.
- **Self-check pass.** Before executing, the same model re-reads the proposed call against
  the tool schema in a second constrained turn. Cheap locally, and catches
  argument-level nonsense that grammar alone cannot.
- **The shell is the force multiplier.** With Termux, one correct tool call can accomplish
  what would otherwise take a ten-step plan the model cannot hold. Prefer composing a
  single shell command over long tool chains — it plays directly to the model's strengths
  and away from its weakness.
- **Templated skills.** Common tasks ship as parameterized recipes where the model only
  fills slots. Deterministic where it can be, generative only where it must be.

Framed this way the target is real and honest: **a local-first agent that reliably does
bounded, useful things**, and degrades visibly rather than hallucinating actions.

### The capability surface

With Termux as the only elevated tier, the agent gets:

| Tier | Mechanism | Grants |
|---|---|---|
| 0 | In-app | Sandbox files, SAF-picked documents, HTTP, RAG over local docs, calendar, contacts, clipboard, notifications |
| 1 | **Termux** `RUN_COMMAND` intent | A real POSIX shell — git, python, coreutils, text processing |

Explicitly **out of scope**: Shizuku, AccessibilityService, root. Cross-app control is not
a feature of this app. Termux alone covers the Claude Code-shaped work — file
manipulation, git, scripting, data wrangling — which is the part that actually matters.

### Other real constraints, named up front

- **Thermals are the ceiling, not FLOPs.** Sustained decode pulls 5–8 W and the phone
  throttles within minutes. Needs a governor reading
  `PowerManager.getCurrentThermalStatus()` that throttles token rate *before* the SoC
  does, with honest UI when it engages.
- **Android 16 tightens JobScheduler quotas even with a foreground service running.** Long
  generations need a correctly typed foreground service and must survive backgrounding.
- **Streaming tokens into Compose naively will destroy frame rate.** 30 tok/s of
  individual state writes causes recomposition storms. Buffer off the main thread, flush
  on a ~16 ms cadence into a stable `AnnotatedString`.
- **Conversion is disk-hungry.** A 2B bf16 download is ~4 GB, GGUF output another ~1.2 GB.
  Stream tensor-by-tensor, never materialize the whole model, preflight free space.

---

## Recommended stack

- **Kotlin + Jetpack Compose**, native. Not Flutter/RN — the motion and haptic ceiling you
  described needs direct platform access, and the heavy lifting is JNI regardless.
- **llama.cpp** via a thin JNI layer, vendored as a submodule. Chosen over MLC (per-target
  compilation is hostile to "download any model"), ExecuTorch (needs `.pte`) and Cactus
  (closed format, defeats the premise). It is also the only option that hands us
  GGUF-writing and quantization primitives for the converter, plus GBNF grammars for the
  agent.
- Backends in priority order: **CPU (ARM NEON/i8mm) → OpenCL (Adreno) → Hexagon NPU**.
  Ship CPU first; it is the only one guaranteed to work everywhere.
- Room for persistence, DataStore for prefs, Hilt for DI, Kotlin Flow for token streaming.

---

## Build plan

### Phase 0 — Prove the risky parts (before any UI)

Everything downstream is dead if this fails, so it comes first.

1. Bare app + llama.cpp submodule + CMake NDK build, `arm64-v8a` only.
2. JNI bridge: load GGUF → tokenize → decode loop → stream tokens to Kotlin.
3. Sideload a known-good Q4_K_M 1–2B GGUF and **measure on the actual target device**: prefill
   tok/s, decode tok/s, peak RSS, sustained rate and temperature over 10 minutes, battery
   drain per 1k tokens.
4. Publish results in `docs/benchmarks.md`.

**Gate:** if sustained decode is under ~10 tok/s, revisit model size or backend before
building anything on top.

### Phase 1 — Model acquisition + on-device conversion

The core differentiator. Ships in v1.

- HF/GitHub repo browser and resolver; parse `config.json` to detect architecture and
  **reject unsupported ones before downloading gigabytes**.
- Resumable, checksummed, chunked downloader on a foreground service.
- **Native converter** (`native/converter/`):
  - safetensors header parse (JSON) + mmap tensor access
  - arch-specific tensor-name mapping — Llama, Qwen2/3, Gemma, Phi to start
  - Llama-family Q/K permutation for RoPE (output is silently wrong if missed)
  - bf16/f16 → f32 → `ggml_quantize_chunk` → Q4_K_M / Q4_0 / Q8_0
  - `tokenizer.json` → GGUF vocab, merges, pre-tokenizer identification
  - `gguf_*` streaming writer with progress callbacks
- Q4_0 selectable as an output, since it has the best Hexagon NPU path.
- Model manager UI: storage accounting, delete, re-quantize, verify.

**Risk:** tokenizer conversion is the fiddliest piece. Mitigate by validating converted
output against a desktop-converted reference GGUF, tensor-by-tensor, in CI.

### Phase 2 — Chat, done properly

- Streaming chat using the buffered-flush renderer described above.
- Conversation persistence, KV-cache reuse across turns, context-window management with
  visible token accounting.
- Sampling controls, system prompts, per-model chat templates parsed from GGUF metadata.
- Thermal governor and battery-aware throttling, surfaced honestly in the UI.

### Phase 3 — Agent harness

- Tool registry with typed schemas and per-tool permission gating.
- **GBNF grammar-constrained tool-call decoding.**
- Embedding-based tool retrieval so the model sees 3–5 candidates, never the full set.
- Deterministic step executor: plan → confirm → execute → observe, with a hard step
  ceiling and a fully visible trace.
- Constrained self-check pass over each proposed call before execution.
- Tier 0 tools: files, SAF documents, HTTP fetch, RAG over local docs, calendar, contacts,
  clipboard, notifications.
- **Termux integration** behind an explicit opt-in that states exactly what it grants.
  Every command is shown verbatim and confirmed before it runs, until the user disables
  confirmation per-tool.
- Templated skills for common multi-step tasks.

### Phase 4 — The refinement pass

Budgeted as real work, not polish-at-the-end.

- Motion system: one spring spec table, no ad-hoc durations. Shared-element transitions
  between model list and model detail. Predictive back throughout.
- Custom haptics via `VibratorManager` waveforms — distinct signatures for token-start,
  tool-call, completion, error.
- 120 Hz LTPO-aware rendering; jank budget enforced with Macrobenchmark in CI.
- Material 3 Expressive theming with real dynamic color; dark theme designed, not inverted.
- Baseline Profiles for startup; Compose stability audit (`@Immutable`, no lambda
  allocation in hot paths).

### Phase 5 — Release

- Reproducible release build, signed APK, GitHub Releases with per-ABI artifacts.
- In-app update check against the releases endpoint.

---

## Verification

- **Phase 0 gate:** measured tok/s, RSS and thermal curve on the target device, written to
  `docs/benchmarks.md`. Everything downstream depends on these being real numbers.
- **Converter correctness:** convert a model on-device, and separately with desktop
  `convert_hf_to_gguf.py`; assert tensor-level equivalence and compare perplexity on a
  fixed sample. The only trustworthy check that the RoPE permutation and tokenizer are
  right.
- **Agent reliability:** a fixed suite of ~50 scripted tasks tracking tool-call validity
  rate and task completion rate. Grammar constraints should put validity at ~100%;
  completion rate is the honest measure of how far a 1–2B model actually gets, and it is
  the number that decides whether Phase 3 needs another iteration.
- **UI performance:** Macrobenchmark frame timing during active token streaming — the worst
  case — with a p99 frame-time budget enforced in CI.
- **End-to-end smoke test:** install → download a safetensors model from HF → convert
  on-device → chat → run one Termux-backed agent task.

---

## What this will not be

Stated plainly so it is not discovered late:

- **Not Play Store distributable** — by choice.
- **Not autonomous the way Claude Code is autonomous.** With no cloud escalation, the
  ceiling is reliable bounded tasks and shell-composed one-shots, not open-ended
  multi-step planning. The harness maximizes what a 1–2B model can do; it cannot exceed it.
- **No cross-app control** — no Shizuku, no AccessibilityService, no root.
- **Not able to run models above ~4B usefully**, and not able to sustain full-rate decode
  indefinitely. Physics, not engineering.
- **Not able to ingest arbitrary HF architectures on day one** — the converter covers a
  whitelist that grows over time.
