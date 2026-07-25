package dev.sruti.llm

import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

/** Sampling configuration for a single generation. */
data class SamplingParams(
    val maxTokens: Int = 256,
    /** Zero or below selects greedy decoding. */
    val temperature: Float = 0.7f,
    val topK: Int = 40,
    val topP: Float = 0.95f,
    val minP: Float = 0.05f,
    val seed: Int = -1,
)

/** Static facts about a loaded model. */
data class ModelInfo(
    val description: String,
    val paramCount: Long,
    val sizeBytes: Long,
    val trainedContextLength: Int,
)

/** What generation produced, and how fast. */
data class GenerationMetrics(
    val promptTokens: Int,
    val generatedTokens: Int,
    val prefillMicros: Long,
    val decodeMicros: Long,
    val stopReason: StopReason,
) {
    val prefillTokensPerSecond: Double
        get() = if (prefillMicros <= 0) 0.0 else promptTokens * 1_000_000.0 / prefillMicros

    val decodeTokensPerSecond: Double
        get() = if (decodeMicros <= 0) 0.0 else generatedTokens * 1_000_000.0 / decodeMicros
}

enum class StopReason {
    EndOfGeneration,
    MaxTokens,
    Cancelled,
    ContextFull,
    ;

    internal companion object {
        fun fromNative(value: Long): StopReason = when (value) {
            0L -> EndOfGeneration
            1L -> MaxTokens
            2L -> Cancelled
            3L -> ContextFull
            else -> EndOfGeneration
        }
    }
}

/** One streamed generation event. */
sealed interface GenerationEvent {
    @JvmInline
    value class Token(val piece: String) : GenerationEvent

    @JvmInline
    value class Completed(val metrics: GenerationMetrics) : GenerationEvent
}

/**
 * Owns a llama.cpp model and context.
 *
 * Not thread-safe: a llama context holds a KV cache that cannot service concurrent
 * decodes, so callers must serialise generations. [close] is idempotent.
 */
class LlamaEngine private constructor(
    private val modelHandle: Long,
    private val contextHandle: Long,
    private val dispatcher: CoroutineDispatcher,
    private val configuredContextLength: Int,
) : AutoCloseable {

    private val closed = AtomicBoolean(false)

    val info: ModelInfo = ModelInfo(
        description = LlamaBridge.nativeModelDesc(modelHandle),
        paramCount = LlamaBridge.nativeModelParamCount(modelHandle),
        sizeBytes = LlamaBridge.nativeModelSizeBytes(modelHandle),
        trainedContextLength = LlamaBridge.nativeModelCtxTrain(modelHandle),
    )

    fun tokenize(text: String, addSpecial: Boolean = true): IntArray {
        check(!closed.get()) { "engine is closed" }
        return LlamaBridge.nativeTokenize(modelHandle, text, addSpecial)
    }

    /** Size of the context window this engine was created with. */
    val contextLength: Int get() = configuredContextLength

    /**
     * Opens a stateful conversation over this model.
     *
     * The session holds the KV cache between turns, so it must be closed before
     * the engine is. Only one session at a time is meaningful: they would share a
     * single context and overwrite each other's cache.
     */
    fun newChatSession(governor: ThermalGovernor? = null): ChatSession {
        check(!closed.get()) { "engine is closed" }
        return ChatSession(modelHandle, contextHandle, dispatcher, governor)
    }

    /** Drops the KV cache so the next generation starts from a clean context. */
    suspend fun reset() = withContext(dispatcher) {
        check(!closed.get()) { "engine is closed" }
        LlamaBridge.nativeResetContext(contextHandle)
    }

    /**
     * Streams [GenerationEvent.Token] as text is decoded, then a single
     * [GenerationEvent.Completed]. Cancelling the collecting coroutine stops the
     * native decode loop at the next token boundary.
     */
    fun generate(prompt: String, params: SamplingParams = SamplingParams()): Flow<GenerationEvent> =
        channelFlow {
            check(!closed.get()) { "engine is closed" }

            val metrics = LlamaBridge.nativeGenerate(
                modelHandle = modelHandle,
                ctxHandle = contextHandle,
                prompt = prompt,
                maxTokens = params.maxTokens,
                temperature = params.temperature,
                topK = params.topK,
                topP = params.topP,
                minP = params.minP,
                seed = if (params.seed >= 0) params.seed else System.nanoTime().toInt(),
            ) { piece ->
                // trySend must not block the native decode thread. The .buffer()
                // below fuses into this channelFlow and makes the channel unbounded,
                // so a slow collector cannot stall inference or drop tokens.
                trySend(GenerationEvent.Token(piece))
                isActive
            }

            send(
                GenerationEvent.Completed(
                    GenerationMetrics(
                        promptTokens = metrics[0].toInt(),
                        generatedTokens = metrics[1].toInt(),
                        prefillMicros = metrics[2],
                        decodeMicros = metrics[3],
                        stopReason = StopReason.fromNative(metrics[4]),
                    ),
                ),
            )
        }.buffer(Channel.UNLIMITED).flowOn(dispatcher)

    override fun close() {
        if (closed.compareAndSet(false, true)) {
            LlamaBridge.nativeFreeContext(contextHandle)
            LlamaBridge.nativeFreeModel(modelHandle)
        }
    }

    companion object {
        /**
         * Loads [modelFile] and builds a context.
         *
         * [nThreads] should be the number of performance cores, not all cores —
         * scheduling decode work onto little cores costs more than it adds.
         */
        suspend fun load(
            modelFile: File,
            dispatcher: CoroutineDispatcher,
            nCtx: Int = 4096,
            nThreads: Int = DeviceCapabilities.recommendedThreadCount(),
            nBatch: Int = 512,
            nGpuLayers: Int = 0,
        ): LlamaEngine = withContext(dispatcher) {
            require(modelFile.isFile) { "model not found: ${modelFile.absolutePath}" }

            LlamaBridge.nativeBackendInit()

            val model = LlamaBridge.nativeLoadModel(modelFile.absolutePath, nGpuLayers)
            check(model != 0L) { "failed to load model: ${modelFile.absolutePath}" }

            val ctx = try {
                LlamaBridge.nativeNewContext(model, nCtx, nThreads, nBatch).also {
                    check(it != 0L) { "failed to create context" }
                }
            } catch (t: Throwable) {
                // The model is already resident at this point; leaking it would
                // strand hundreds of MB until the process dies.
                LlamaBridge.nativeFreeModel(model)
                throw t
            }

            LlamaEngine(model, ctx, dispatcher, nCtx)
        }
    }
}
