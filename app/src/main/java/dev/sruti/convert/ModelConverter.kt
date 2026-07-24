package dev.sruti.convert

import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.isActive
import java.io.File

/** Quantization formats the converter can produce. */
enum class QuantType(val id: String, val label: String, val bitsPerWeight: Double) {
    /** The default. Best quality-per-byte for phone-class models. */
    Q4_K_M("Q4_K_M", "Q4_K_M · balanced", 4.9),
    Q4_K_S("Q4_K_S", "Q4_K_S · smaller", 4.6),
    Q5_K_M("Q5_K_M", "Q5_K_M · higher quality", 5.7),

    /** Best Hexagon NPU support, so worth having when the NPU path lands. */
    Q4_0("Q4_0", "Q4_0 · NPU friendly", 4.5),
    Q8_0("Q8_0", "Q8_0 · near-lossless", 8.5),
    F16("F16", "F16 · unquantized", 16.0),
    ;

    companion object {
        fun fromId(id: String): QuantType = entries.firstOrNull { it.id == id } ?: Q4_K_M
    }
}

/** Conversion stages, matching the native `ConvertStage` ordinals. */
enum class ConvertStage(val label: String) {
    Reading("Reading checkpoint"),
    WritingMetadata("Writing metadata"),
    WritingTensors("Converting tensors"),
    Quantizing("Quantizing"),
    Finalising("Finalising"),
    ;

    internal companion object {
        fun fromNative(value: Int): ConvertStage = entries.getOrElse(value) { Reading }
    }
}

sealed interface ConvertEvent {
    data class Progress(
        val stage: ConvertStage,
        val detail: String,
        val tensorsDone: Long,
        val tensorsTotal: Long,
    ) : ConvertEvent {
        /** Fraction complete within the tensor-writing stage, or null when unknown. */
        val fraction: Float?
            get() = if (tensorsTotal > 0) tensorsDone.toFloat() / tensorsTotal else null
    }

    @JvmInline
    value class Warning(val message: String) : ConvertEvent

    data class Completed(val outputFile: File) : ConvertEvent
}

/**
 * Converts a Hugging Face safetensors checkpoint into a quantized GGUF, on device.
 *
 * This is what separates Sruti from apps that consume a fixed catalogue: point it
 * at any supported checkpoint directory and it produces a runnable model without a
 * desktop anywhere in the loop.
 *
 * Conversion is CPU- and disk-heavy and takes minutes for a 1-2B model. Run it
 * from a foreground service so it survives the screen turning off.
 */
class ModelConverter(private val dispatcher: CoroutineDispatcher) {

    /**
     * Free bytes needed to convert a checkpoint of [checkpointBytes].
     *
     * Larger than the final file: the F16 intermediate and the quantized output
     * exist on disk simultaneously. Preflight with this — running out part-way
     * wastes everything already spent.
     */
    fun estimatedWorkingBytes(checkpointBytes: Long, quantType: QuantType): Long =
        ConverterBridge.nativeEstimatedWorkingBytes(checkpointBytes, quantType.id)

    /**
     * Runs a conversion, emitting progress until [ConvertEvent.Completed].
     *
     * Cancelling the collecting coroutine stops the conversion and removes partial
     * output. Failure throws [ConversionException].
     */
    fun convert(
        modelDir: File,
        outputFile: File,
        quantType: QuantType = QuantType.Q4_K_M,
        modelName: String = modelDir.name,
        nThreads: Int = 0,
    ): Flow<ConvertEvent> = channelFlow {
        require(modelDir.isDirectory) { "not a directory: ${modelDir.absolutePath}" }
        outputFile.parentFile?.mkdirs()

        val callback = object : ConverterBridge.ConvertCallback {
            override fun onProgress(
                stage: Int,
                detail: String,
                tensorsDone: Long,
                tensorsTotal: Long,
            ): Boolean {
                trySend(
                    ConvertEvent.Progress(
                        stage = ConvertStage.fromNative(stage),
                        detail = detail,
                        tensorsDone = tensorsDone,
                        tensorsTotal = tensorsTotal,
                    ),
                )
                return isActive
            }

            override fun onWarning(message: String) {
                trySend(ConvertEvent.Warning(message))
            }
        }

        val error = ConverterBridge.nativeConvert(
            modelDir = modelDir.absolutePath,
            outPath = outputFile.absolutePath,
            quantType = quantType.id,
            modelName = modelName,
            nThreads = nThreads,
            callback = callback,
        )

        when {
            error == null -> send(ConvertEvent.Completed(outputFile))
            // Cancellation is the collector's own doing, not a failure to report.
            error == "cancelled" -> return@channelFlow
            else -> throw ConversionException(error)
        }
    }.buffer(Channel.UNLIMITED).flowOn(dispatcher)
}

class ConversionException(message: String) : Exception(message)
