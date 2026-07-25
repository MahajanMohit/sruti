package dev.sruti.llm

import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean

/** One turn in a conversation. */
data class ChatMessage(
    val role: Role,
    val content: String,
) {
    enum class Role(val id: String) {
        System("system"),
        User("user"),
        Assistant("assistant"),
        ;

        companion object {
            fun fromId(id: String): Role =
                entries.firstOrNull { it.id == id } ?: User
        }
    }
}

data class ChatParams(
    val maxTokens: Int = 512,
    val temperature: Float = 0.7f,
    val topK: Int = 40,
    val topP: Float = 0.95f,
    val minP: Float = 0.05f,
    /** 1.0 disables. */
    val repeatPenalty: Float = 1.1f,
    val repeatLastN: Int = 64,
    val seed: Int = -1,
)

data class ChatMetrics(
    val promptTokens: Int,
    /** Prompt tokens served from the existing KV cache rather than recomputed. */
    val reusedTokens: Int,
    val generatedTokens: Int,
    val prefillMicros: Long,
    val decodeMicros: Long,
    val stopReason: StopReason,
    /** Tokens now held in the context window. */
    val cachedTokens: Int,
) {
    val prefillTokensPerSecond: Double
        get() {
            val computed = promptTokens - reusedTokens
            return if (prefillMicros <= 0 || computed <= 0) 0.0
            else computed * 1_000_000.0 / prefillMicros
        }

    val decodeTokensPerSecond: Double
        get() = if (decodeMicros <= 0) 0.0 else generatedTokens * 1_000_000.0 / decodeMicros

    /** Fraction of the prompt that did not need recomputing. */
    val cacheHitRate: Double
        get() = if (promptTokens <= 0) 0.0 else reusedTokens.toDouble() / promptTokens
}

sealed interface ChatEvent {
    @JvmInline
    value class Token(val piece: String) : ChatEvent

    @JvmInline
    value class Completed(val metrics: ChatMetrics) : ChatEvent
}

/**
 * A stateful conversation over a loaded model.
 *
 * Holds the KV cache between turns. That is the difference between a chat app
 * that stays responsive and one that gets slower with every message: without
 * reuse, turn ten re-processes the entire conversation before producing its first
 * token.
 *
 * Not thread-safe — a llama context cannot service concurrent decodes. Callers
 * must serialise turns.
 */
class ChatSession internal constructor(
    private val modelHandle: Long,
    private val contextHandle: Long,
    private val dispatcher: CoroutineDispatcher,
    private val governor: ThermalGovernor?,
) : AutoCloseable {

    private val sessionHandle: Long = ChatBridge.nativeNewSession()
    private val closed = AtomicBoolean(false)

    /** The model's own chat template, or null when it declares none. */
    val chatTemplate: String? = ChatBridge.nativeChatTemplate(modelHandle)

    val hasChatTemplate: Boolean get() = chatTemplate != null

    /** Tokens currently held in the context window. */
    val cachedTokens: Int get() = ChatBridge.nativeCachedTokens(sessionHandle)

    /**
     * Formats a conversation using the model's template.
     *
     * Falls back to a plain transcript when the model declares no template. That
     * fallback is a compromise, not an equivalent: a base model without a template
     * genuinely has no turn structure, and pretending otherwise would be worse.
     */
    fun formatPrompt(messages: List<ChatMessage>, addAssistantPrefix: Boolean = true): String {
        check(!closed.get()) { "session is closed" }

        val formatted = ChatBridge.nativeApplyTemplate(
            modelHandle = modelHandle,
            roles = messages.map { it.role.id }.toTypedArray(),
            contents = messages.map { it.content }.toTypedArray(),
            addAssistant = addAssistantPrefix,
        )
        if (formatted != null) return formatted

        return buildString {
            messages.forEach { message ->
                append(message.role.id).append(": ").append(message.content).append('\n')
            }
            if (addAssistantPrefix) append("assistant: ")
        }
    }

    /** Drops the cache so the next turn starts from an empty context. */
    suspend fun reset() = withContext(dispatcher) {
        check(!closed.get()) { "session is closed" }
        ChatBridge.nativeResetSession(contextHandle, sessionHandle)
    }

    /**
     * Generates a reply to [messages].
     *
     * The full conversation is passed every turn; the shared prefix with the
     * cache is found natively and only the remainder is prefilled.
     *
     * [systemPromptTokens] is preserved when the conversation outgrows the context
     * window — without it, the system prompt is the first thing evicted and the
     * model's instructions silently disappear mid-conversation.
     */
    fun generate(
        messages: List<ChatMessage>,
        params: ChatParams = ChatParams(),
        systemPromptTokens: Int = 0,
        /**
         * GBNF source constraining what may be emitted, or null for free text.
         *
         * With one supplied, output that would break the structure is not merely
         * unlikely — it is unreachable, because the offending tokens are masked
         * during sampling.
         */
        grammar: String? = null,
    ): Flow<ChatEvent> = channelFlow {
        check(!closed.get()) { "session is closed" }

        val prompt = formatPrompt(messages)
        val tokens = LlamaBridge.nativeTokenize(modelHandle, prompt, /*addSpecial=*/true)

        val callback = object : ChatBridge.ChatCallback {
            override fun onToken(piece: String): Boolean {
                trySend(ChatEvent.Token(piece))
                return isActive
            }

            override fun pacingMicros(): Long = governor?.pacingMicros() ?: 0L
        }

        val raw = ChatBridge.nativeGenerate(
            modelHandle = modelHandle,
            ctxHandle = contextHandle,
            sessionHandle = sessionHandle,
            tokens = tokens,
            nKeep = systemPromptTokens,
            maxTokens = params.maxTokens,
            temperature = params.temperature,
            topK = params.topK,
            topP = params.topP,
            minP = params.minP,
            repeatPenalty = params.repeatPenalty,
            repeatLastN = params.repeatLastN,
            seed = if (params.seed >= 0) params.seed else System.nanoTime().toInt(),
            grammar = grammar.orEmpty(),
            callback = callback,
        )

        send(
            ChatEvent.Completed(
                ChatMetrics(
                    promptTokens = raw[0].toInt(),
                    reusedTokens = raw[1].toInt(),
                    generatedTokens = raw[2].toInt(),
                    prefillMicros = raw[3],
                    decodeMicros = raw[4],
                    stopReason = StopReason.fromNative(raw[5]),
                    cachedTokens = raw[6].toInt(),
                ),
            ),
        )
    }.buffer(Channel.UNLIMITED).flowOn(dispatcher)

    /** Token count for a piece of text, for context accounting. */
    fun countTokens(text: String): Int =
        LlamaBridge.nativeTokenize(modelHandle, text, /*addSpecial=*/false).size

    override fun close() {
        if (closed.compareAndSet(false, true)) {
            ChatBridge.nativeFreeSession(sessionHandle)
        }
    }
}
