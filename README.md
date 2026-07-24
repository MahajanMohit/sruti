# Sruti

An on-device LLM runtime and agent for Android.

Sruti runs 1–2B parameter models locally on the phone. Unlike apps that consume
pre-baked bundles, it is being built to take **arbitrary safetensors models from
Hugging Face or GitHub and convert them on the device itself** — no desktop step
anywhere in the pipeline.

Inference is **strictly local**. No prompt, no completion, and no telemetry is ever
sent to a remote model.

> **Status: Phase 0.** The inference path works end to end — GGUF loads, tokens
> stream, benchmarks run. The safetensors converter and the agent harness are not
> built yet. See [the plan](#roadmap).

---

## Why this exists

Modern phones have the memory bandwidth and thermal budget to run small models
usefully, but the tooling assumes a desktop is nearby. Existing on-device apps ship
a fixed catalogue in a proprietary container. Sruti aims at the thing that is
actually missing: point it at a model repository and have it work.

The design decisions, the feasibility analysis, and the honest limits are documented
in [`docs/plan.md`](docs/plan.md). Read that before contributing — in particular the
section on why a 1–2B model cannot be an autonomous agent and what the harness does
about it.

---

## Requirements

- A device on **Android 12 (API 31) or newer**, arm64. Primary target is a OnePlus
  on Android 16, Snapdragon 8-class.
- **JDK 17+** and the Android SDK with **NDK 27.2.12479018** and CMake 3.22.1.
- Roughly 4 GB of free disk for the build (llama.cpp objects are not small).

## Building

```bash
git clone --recurse-submodules https://github.com/mahajanmohit/sruti.git
cd sruti

# If you already cloned without --recurse-submodules:
git submodule update --init --recursive

echo "sdk.dir=/path/to/android-sdk" > local.properties

./gradlew :app:assembleDebug
./gradlew :app:installDebug
```

The first build compiles llama.cpp from source and takes several minutes.
Subsequent builds are incremental.

Only `arm64-v8a` is built. Every target device has it, and dropping the other ABIs
keeps both the APK and the native build time down.

## Running the Phase 0 gate

Phase 0 exists to answer one question with measurements instead of estimates: **is
a 1–2B model at Q4 fast enough on this specific device to build a product on?**

1. Put a Q4_K_M GGUF of a 1–2B model on the device.
2. Launch Sruti, tap **Choose a GGUF file**, and pick it.
3. **Smoke test** runs one short generation — this verifies the whole JNI path.
4. **Full benchmark** measures prefill scaling, decode rate, and a ten-minute
   sustained run sampling thermal status, battery draw and RSS.

The benchmark writes `benchmarks.md` to the app's external files directory:

```bash
adb pull /sdcard/Android/data/dev.sruti/files/benchmarks.md docs/benchmarks.md
```

**The gate is sustained decode at or above 10 tok/s.** Below that, model size or
backend needs revisiting before anything gets built on top. The generated report
states PASS or FAIL explicitly.

## Testing

```bash
./gradlew :app:testDebugUnitTest   # Kotlin unit tests
./tools/run_native_tests.sh        # host-side C++ tests, under ASan + UBSan
```

The native tests cover UTF-8 reassembly across token boundaries. This matters more
than it sounds: a llama.cpp token piece is a byte string, and multi-byte characters
routinely straddle two tokens — emoji and CJK almost always do. Getting it wrong
corrupts output silently rather than crashing.

---

## Architecture

```
app/src/main/cpp/
  sruti_llm.cpp          JNI bridge: load, tokenize, prefill + decode loop
  utf8_assembler.h       UTF-8 reassembly across token boundaries (header-only, tested)
  CMakeLists.txt         links llama.cpp statically into libsruti_llm.so

app/src/main/java/dev/sruti/
  llm/LlamaBridge.kt         raw external fun declarations — unsafe, internal
  llm/LlamaEngine.kt         safe lifecycle wrapper, generation as a Flow
  llm/DeviceCapabilities.kt  performance-core detection for thread count
  bench/BenchmarkRunner.kt   the Phase 0 gate
  bench/ThermalMonitor.kt    thermal status, battery draw
  bench/ProcessMemory.kt     RSS and peak RSS from /proc/self/status
  ui/StreamCoalescing.kt     batches token updates to the frame cadence
  ui/theme/Motion.kt         the single source of animation specs

native/llama.cpp           git submodule
```

Three decisions worth knowing about:

**Threads are capped at the performance-core count, not all cores.** Decode is a
synchronous fork-join across threads, so the slowest core sets the pace. Scheduling
work onto efficiency cores makes the whole step slower, not faster.

**Token updates are coalesced to the frame cadence.** A model at 30 tok/s writing
Compose state 30 times a second re-measures the entire paragraph on every write.
Batching makes UI cost independent of token rate. Nothing is dropped — emissions
are merged, not discarded.

**Model weights are mmapped, never copied to the Java heap.** The kernel pages them
in on demand, which is why RSS rather than heap size is the number the benchmark
tracks: RSS is what makes the low-memory killer take an interest.

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | JNI bridge, decode loop, benchmark harness | code complete, **awaiting on-device numbers** |
| 1 | Model acquisition + on-device safetensors → GGUF conversion | not started |
| 2 | Chat: KV-cache reuse, context management, thermal governor | not started |
| 3 | Agent harness: grammar-constrained tool calls, Termux shell | not started |
| 4 | Refinement: motion, haptics, 120 Hz, jank budget in CI | not started |
| 5 | Signed release via GitHub Releases | not started |

### Deliberate non-goals

- **Not distributed via Play Store.** The elevated capability tier could not survive
  its policy on automation.
- **No cross-app control.** No Shizuku, no AccessibilityService, no root.
- **No cloud escalation.** This bounds what the agent can do, and that is accepted.
- **Not models above ~4B.** Physics, not engineering.

## Licence

Sruti's own code is MIT. Bundled [llama.cpp](https://github.com/ggml-org/llama.cpp)
is MIT.
