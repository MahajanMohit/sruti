package dev.sruti.bench

import android.content.Context
import android.os.Build
import dev.sruti.llm.DeviceCapabilities
import dev.sruti.llm.GenerationEvent
import dev.sruti.llm.LlamaEngine
import dev.sruti.llm.SamplingParams
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.last
import java.io.File
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.TimeZone

/**
 * The Phase 0 gate.
 *
 * Answers one question with measurements rather than estimates: is a 1-2B model at
 * Q4 fast enough on this specific device to build a product on? Published figures
 * for nearby models are a starting hypothesis, not an answer — thermal behaviour
 * especially is device- and chassis-specific.
 *
 * Passing means sustained decode holds above roughly 10 tok/s over ten minutes.
 */
class BenchmarkRunner(
    private val context: Context,
    private val modelFile: File,
) {

    data class Config(
        /** Prompt lengths (in approximate tokens) to characterise prefill scaling. */
        val prefillProbeLengths: List<Int> = listOf(64, 256, 1024),
        val decodeTokens: Int = 256,
        val sustainedDurationMillis: Long = 10 * 60 * 1000L,
        val sustainedSampleIntervalMillis: Long = 15_000L,
        val contextLength: Int = 4096,
    )

    sealed interface Progress {
        data class Stage(val label: String) : Progress
        data class Line(val text: String) : Progress
        data class Done(val report: String, val reportFile: File) : Progress
    }

    fun run(config: Config = Config()): Flow<Progress> = flow {
        val threads = DeviceCapabilities.recommendedThreadCount()

        emit(Progress.Stage("Loading model"))
        val loadStart = System.nanoTime()
        val engine = LlamaEngine.load(
            modelFile = modelFile,
            dispatcher = Dispatchers.Default,
            nCtx = config.contextLength,
            nThreads = threads,
        )
        val loadMillis = (System.nanoTime() - loadStart) / 1_000_000

        try {
            val afterLoad = ProcessMemory.read()
            emit(Progress.Line("Loaded ${engine.info.description} in ${loadMillis}ms"))
            emit(Progress.Line("RSS after load: ${afterLoad.residentBytes.asMib()} MiB"))

            // Warmup. The first decode pays for page faults on mmapped weights and
            // for the scheduler settling threads onto cores; including it would
            // understate steady-state throughput.
            emit(Progress.Stage("Warmup"))
            engine.reset()
            engine.generate(
                prompt = "Hello.",
                params = SamplingParams(maxTokens = 16, temperature = 0f),
            ).last()

            // --- prefill ----------------------------------------------------
            emit(Progress.Stage("Measuring prefill"))
            val prefillResults = mutableListOf<PrefillResult>()
            for (target in config.prefillProbeLengths) {
                engine.reset()
                val prompt = syntheticPrompt(target)
                val actualTokens = engine.tokenize(prompt).size
                val completed = engine.generate(
                    prompt = prompt,
                    // One token is the minimum that still forces a full prefill.
                    params = SamplingParams(maxTokens = 1, temperature = 0f),
                ).last() as GenerationEvent.Completed

                val result = PrefillResult(
                    requestedTokens = target,
                    actualTokens = actualTokens,
                    tokensPerSecond = completed.metrics.prefillTokensPerSecond,
                )
                prefillResults += result
                emit(
                    Progress.Line(
                        "prefill ${result.actualTokens} tok: " +
                            "${"%.1f".format(result.tokensPerSecond)} tok/s",
                    ),
                )
            }

            // --- decode -----------------------------------------------------
            emit(Progress.Stage("Measuring decode"))
            engine.reset()
            val decodeCompleted = engine.generate(
                prompt = DECODE_PROMPT,
                params = SamplingParams(maxTokens = config.decodeTokens, temperature = 0.7f),
            ).last() as GenerationEvent.Completed
            val coldDecodeRate = decodeCompleted.metrics.decodeTokensPerSecond
            emit(Progress.Line("decode: ${"%.1f".format(coldDecodeRate)} tok/s"))

            // --- sustained --------------------------------------------------
            emit(Progress.Stage("Sustained run (${config.sustainedDurationMillis / 60_000} min)"))
            val thermal = ThermalMonitor(context)
            val samples = mutableListOf<SustainedSample>()
            val runStart = System.currentTimeMillis()
            var lastSampleAt = 0L
            var round = 0

            while (System.currentTimeMillis() - runStart < config.sustainedDurationMillis) {
                engine.reset()
                val completed = engine.generate(
                    prompt = DECODE_PROMPT,
                    params = SamplingParams(maxTokens = 128, temperature = 0.7f),
                ).last() as GenerationEvent.Completed

                val elapsed = System.currentTimeMillis() - runStart
                round++

                if (elapsed - lastSampleAt >= config.sustainedSampleIntervalMillis) {
                    lastSampleAt = elapsed
                    val sample = SustainedSample(
                        thermal = thermal.sample(elapsed),
                        memory = ProcessMemory.read(),
                        decodeTokensPerSecond = completed.metrics.decodeTokensPerSecond,
                    )
                    samples += sample
                    emit(
                        Progress.Line(
                            "t=${elapsed / 1000}s " +
                                "${"%.1f".format(sample.decodeTokensPerSecond)} tok/s " +
                                "thermal=${sample.thermal.thermalStatusName} " +
                                "battery=${sample.thermal.batteryPercent}%",
                        ),
                    )
                }
            }

            val report = buildReport(
                engine = engine,
                threads = threads,
                loadMillis = loadMillis,
                config = config,
                prefill = prefillResults,
                coldDecodeRate = coldDecodeRate,
                samples = samples,
                rounds = round,
            )

            val reportFile = File(context.getExternalFilesDir(null), "benchmarks.md")
            reportFile.writeText(report)
            emit(Progress.Done(report, reportFile))
        } finally {
            engine.close()
        }
    }

    private data class PrefillResult(
        val requestedTokens: Int,
        val actualTokens: Int,
        val tokensPerSecond: Double,
    )

    private data class SustainedSample(
        val thermal: ThermalMonitor.Sample,
        val memory: ProcessMemory.Snapshot,
        val decodeTokensPerSecond: Double,
    )

    private fun buildReport(
        engine: LlamaEngine,
        threads: Int,
        loadMillis: Long,
        config: Config,
        prefill: List<PrefillResult>,
        coldDecodeRate: Double,
        samples: List<SustainedSample>,
        rounds: Int,
    ): String = buildString {
        val timestamp = SimpleDateFormat("yyyy-MM-dd HH:mm:ss 'UTC'", Locale.US)
            .apply { timeZone = TimeZone.getTimeZone("UTC") }
            .format(System.currentTimeMillis())

        appendLine("# Phase 0 Benchmarks")
        appendLine()
        appendLine("Generated $timestamp")
        appendLine()

        appendLine("## Device")
        appendLine()
        appendLine("| Field | Value |")
        appendLine("|---|---|")
        appendLine("| Device | ${Build.MANUFACTURER} ${Build.MODEL} |")
        appendLine("| SoC | ${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL} |")
        appendLine("| Android | ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT}) |")
        appendLine("| Build | ${Build.DISPLAY} |")
        appendLine("| Total cores | ${Runtime.getRuntime().availableProcessors()} |")
        appendLine("| Performance cores | ${DeviceCapabilities.performanceCoreCount()} |")
        appendLine("| Inference threads | $threads |")
        appendLine()

        appendLine("## Model")
        appendLine()
        appendLine("| Field | Value |")
        appendLine("|---|---|")
        appendLine("| File | ${modelFile.name} |")
        appendLine("| Description | ${engine.info.description} |")
        appendLine("| Parameters | ${"%.2f".format(engine.info.paramCount / 1e9)} B |")
        appendLine("| On-disk size | ${engine.info.sizeBytes.asMib()} MiB |")
        appendLine("| Trained context | ${engine.info.trainedContextLength} |")
        appendLine("| Context used | ${config.contextLength} |")
        appendLine("| Load time | ${loadMillis} ms |")
        appendLine()

        appendLine("## Prefill")
        appendLine()
        appendLine("| Prompt tokens | tok/s |")
        appendLine("|---|---|")
        prefill.forEach {
            appendLine("| ${it.actualTokens} | ${"%.1f".format(it.tokensPerSecond)} |")
        }
        appendLine()

        appendLine("## Decode")
        appendLine()
        appendLine("Cold (first full generation after warmup): " +
            "**${"%.1f".format(coldDecodeRate)} tok/s**")
        appendLine()

        appendLine("## Sustained load")
        appendLine()
        appendLine(
            "${config.sustainedDurationMillis / 60_000} minutes of continuous " +
                "generation, $rounds rounds of 128 tokens.",
        )
        appendLine()
        appendLine("| t (s) | tok/s | Thermal | Battery % | Current (mA) | RSS (MiB) |")
        appendLine("|---|---|---|---|---|---|")
        samples.forEach { s ->
            appendLine(
                "| ${s.thermal.elapsedMillis / 1000} " +
                    "| ${"%.1f".format(s.decodeTokensPerSecond)} " +
                    "| ${s.thermal.thermalStatusName} " +
                    "| ${s.thermal.batteryPercent} " +
                    "| ${s.thermal.currentMicroAmps / 1000} " +
                    "| ${s.memory.residentBytes.asMib()} |",
            )
        }
        appendLine()

        if (samples.isNotEmpty()) {
            val rates = samples.map { it.decodeTokensPerSecond }
            val first = rates.first()
            val last = rates.last()
            val peakRss = samples.maxOf { it.memory.peakResidentBytes }

            appendLine("### Summary")
            appendLine()
            appendLine("| Metric | Value |")
            appendLine("|---|---|")
            appendLine("| First sample | ${"%.1f".format(first)} tok/s |")
            appendLine("| Last sample | ${"%.1f".format(last)} tok/s |")
            appendLine("| Min | ${"%.1f".format(rates.min())} tok/s |")
            appendLine("| Mean | ${"%.1f".format(rates.average())} tok/s |")
            appendLine(
                "| Throttling loss | ${"%.1f".format((1 - last / first) * 100)}% |",
            )
            appendLine("| Peak RSS | ${peakRss.asMib()} MiB |")
            appendLine(
                "| Max thermal status | ${
                    ThermalMonitor.thermalStatusName(samples.maxOf { it.thermal.thermalStatus })
                } |",
            )
            appendLine()

            val verdict = if (rates.min() >= GATE_TOKENS_PER_SECOND) "PASS" else "FAIL"
            appendLine(
                "**Phase 0 gate: $verdict** " +
                    "(sustained decode must hold at or above " +
                    "$GATE_TOKENS_PER_SECOND tok/s; observed minimum " +
                    "${"%.1f".format(rates.min())} tok/s)",
            )
        }
    }

    /** Builds a prompt of roughly [targetTokens] tokens using ordinary prose. */
    private fun syntheticPrompt(targetTokens: Int): String {
        val sentence = "The quick brown fox jumps over the lazy dog near the river bank. "
        // Prose averages a little under two tokens per word for most BPE vocabularies.
        val approxTokensPerSentence = 15
        val repeats = (targetTokens / approxTokensPerSentence).coerceAtLeast(1)
        return "Summarise the following text.\n\n" + sentence.repeat(repeats)
    }

    private companion object {
        const val GATE_TOKENS_PER_SECOND = 10.0

        const val DECODE_PROMPT =
            "Write a detailed explanation of how a bicycle transmission works, " +
                "covering the chain, derailleur, and gear ratios."
    }
}

private fun Long.asMib(): String = "%.1f".format(this / (1024.0 * 1024.0))
