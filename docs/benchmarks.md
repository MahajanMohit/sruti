# Phase 0 Benchmarks

> **No measurements recorded yet.**
>
> This file is a placeholder. Run the benchmark on the target device and replace
> it with the generated report:
>
> ```bash
> adb pull /sdcard/Android/data/dev.sruti/files/benchmarks.md docs/benchmarks.md
> ```

## Why this file gates everything else

Phase 0 exists to replace estimates with measurements. Published figures for
nearby models put Llama 3.2 3B at roughly 10 tok/s and Llama 3.1 8B at about
5 tok/s on the Hexagon NPU of a Snapdragon 8 Elite, which suggests a 1–2B model at
Q4 should land somewhere around 20–40 tok/s on CPU.

**That is a hypothesis, not a result.** It is drawn from different silicon, a
different chassis, a different quantisation and a different thermal envelope. The
only number that decides whether this project is viable is the one measured on the
actual device.

## The gate

**Sustained decode must hold at or above 10 tok/s across a ten-minute run.**

Sustained is the operative word. A phone will produce an impressive first-minute
figure and then throttle: continuous decode draws several watts, and the SoC sheds
clock to stay within its thermal budget. A benchmark that stops at 60 seconds
measures the best case and tells you nothing about what a user experiences in a
real conversation.

If the gate fails, the response is to revisit model size or backend — not to build
Phase 1 on top of a foundation that is too slow.

## What gets recorded

| Section | Contents |
|---|---|
| Device | Model, SoC, Android build, core topology, threads used |
| Model | Description, parameter count, on-disk size, context length, load time |
| Prefill | tok/s at 64, 256 and 1024 prompt tokens — characterises scaling |
| Decode | Cold decode rate after warmup |
| Sustained | 10 min continuous generation: tok/s, thermal status, battery current, RSS |
| Summary | First/last/min/mean rate, throttling loss, peak RSS, PASS or FAIL |

Warmup is excluded from all figures. The first decode pays for page faults on the
mmapped weights and for the scheduler settling threads onto cores; including it
would understate steady-state throughput.

## Reading the results

- **Throttling loss** is the drop from first to last sample. A large figure means
  the thermal governor planned for Phase 2 is load-bearing rather than a nicety.
- **Peak RSS** is what determines low-memory-killer risk. Java heap metrics are the
  wrong instrument — weights are mmapped by native code and never appear there.
- **Prefill scaling** across the three prompt lengths shows whether long contexts
  are viable or whether prompt caching needs to move up the roadmap.
