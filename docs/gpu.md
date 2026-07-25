# GPU offload

**Status: not built. CPU only.**

This is a deliberate stopping point rather than an oversight, and the reasoning is
worth recording because "add GPU support" sounds like a smaller job than it is on
Android.

## What was attempted

llama.cpp's **Vulkan** backend, chosen over OpenCL because it is the
device-agnostic option — OpenCL's fast path on Android is Adreno-specific, while
Vulkan covers Adreno, Mali and Xclipse from one build.

It got four blockers deep before being abandoned:

1. **`find_package(SPIRV-Headers CONFIG REQUIRED)`** — not in the NDK. Solved by
   vendoring the headers and staging an install.
2. **`CMAKE_PREFIX_PATH` ignored** — the NDK toolchain sets
   `CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY`, confining `find_package` to the
   sysroot. Solved by setting `SPIRV-Headers_DIR` directly.
3. **`Vulkan_GLSLC_EXECUTABLE-NOTFOUND`** — the same restriction hides the glslc
   the NDK does ship. Solved by pointing at `$ANDROID_NDK/shader-tools/*/glslc`.
4. **`'coopmat' : undeclared identifier`** — and this one does not have a cheap
   fix. llama.cpp's compute shaders now use `GL_KHR_cooperative_matrix`, and the
   glslc bundled with NDK 27 predates it.

Clearing (4) means vendoring **shaderc** and building it for the host, then making
that work inside a GitHub Actions run that is already ~90 minutes. That is a
meaningful amount of build infrastructure for a benefit nobody has measured.

## Why stopping here is the right call

**The benefit is unmeasured, and plausibly negative.** A phone GPU shares memory
bandwidth with the CPU. For a 1–2B model at Q4, decode is bandwidth-bound rather
than compute-bound, and GPU offload frequently loses on mobile parts. Prefill is
the case where it more often wins. Which of those dominates depends entirely on
the device and the workload — exactly the question the Phase 0 benchmark exists to
answer, and it has not been run on hardware yet.

Shipping an installable APK is worth more right now than a backend that might be
slower.

## What is already in place

The runtime side is done, so adding a backend later touches the build and nothing
else:

- `SettingsStore.gpuLayers` — how many transformer layers to offload, default 0.
- `LlamaEngine.load(nGpuLayers = ...)` — threaded through and read from settings.
- No UI toggle, on purpose. Exposing a control that cannot do anything would be
  worse than not having one.

## The intended route

**OpenCL, not Vulkan.** Its kernels are compiled by the GPU driver at runtime,
which sidesteps the shader-toolchain problem entirely — no glslc, no shaderc, no
host-side shader generation in CI. It needs OpenCL headers and an ICD loader built
for arm64, both of which are small and self-contained. Qualcomm's Adreno work in
llama.cpp targets this path, and Mali drivers support it too.

Sequence, in order:

1. Run the Phase 0 benchmark on real hardware and record CPU numbers.
2. Add the OpenCL backend behind a build flag.
3. Extend the benchmark to compare CPU against GPU offload at several layer counts.
4. Expose the setting in the UI **only if** the measurement justifies it, with the
   default set by what the numbers say.
