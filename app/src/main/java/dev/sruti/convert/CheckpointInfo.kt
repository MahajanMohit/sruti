package dev.sruti.convert

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * What can be learned about a checkpoint from its config.json.
 *
 * config.json is a few kilobytes; the weights are gigabytes. Fetching it first
 * turns "download 2.5 GB, then discover the architecture is unsupported" into an
 * instant answer, which is the difference between a usable model browser and an
 * infuriating one.
 */
@Serializable
data class CheckpointInfo(
    val supported: Boolean = false,
    /** Why it is unsupported. Empty when [supported]. */
    val error: String = "",

    @SerialName("hfArch") val hfArch: String = "",
    /** GGUF architecture name, e.g. "llama". */
    val arch: String = "",
    /** Human-readable, e.g. "Qwen3". */
    val displayName: String = "",

    val blockCount: Long = 0,
    val hiddenSize: Long = 0,
    val headCount: Long = 0,
    val headCountKv: Long = 0,
    val contextLength: Long = 0,
    val vocabSize: Long = 0,
    val parameterCount: Long = 0,
) {
    val parametersInBillions: Double get() = parameterCount / 1e9

    /** Approximate download size, checkpoints being bf16 or f16. */
    val estimatedCheckpointBytes: Long get() = parameterCount * 2

    /** Approximate size of the converted file at [quantType]. */
    fun estimatedOutputBytes(quantType: QuantType): Long =
        (parameterCount * quantType.bitsPerWeight / 8.0).toLong()

    /**
     * Whether this is in the size class Sruti targets.
     *
     * Not a hard limit — the converter will happily process a larger checkpoint —
     * but above roughly 4B, phone memory bandwidth and thermals make the result
     * too slow to be worth the wait.
     */
    val isComfortableSize: Boolean get() = parameterCount in 1 until 4_500_000_000L

    companion object {
        private val json = Json { ignoreUnknownKeys = true }

        /** Parses a checkpoint's config.json. Never throws. */
        fun inspect(configJson: String): CheckpointInfo =
            runCatching {
                json.decodeFromString<CheckpointInfo>(
                    ConverterBridge.nativeInspectConfig(configJson),
                )
            }.getOrElse {
                CheckpointInfo(
                    supported = false,
                    error = it.message ?: "could not read config.json",
                )
            }
    }
}
