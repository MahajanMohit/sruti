package dev.sruti.llm

/**
 * Raw JNI surface for multi-turn chat. Use [ChatSession].
 */
internal object ChatBridge {

    interface ChatCallback {
        /** Return false to stop generation. */
        fun onToken(piece: String): Boolean

        /**
         * Microseconds to pause before decoding the next token; 0 for full speed.
         *
         * Called every token so the thermal governor can throttle live, rather
         * than the rate being fixed when generation started.
         */
        fun pacingMicros(): Long
    }

    init {
        System.loadLibrary("sruti_llm")
    }

    external fun nativeNewSession(): Long
    external fun nativeFreeSession(handle: Long)
    external fun nativeResetSession(ctxHandle: Long, sessionHandle: Long)
    external fun nativeCachedTokens(sessionHandle: Long): Int

    /** The model's chat template, or null when it declares none. */
    external fun nativeChatTemplate(modelHandle: Long): String?

    /** Formats a conversation, or null when the template cannot be applied. */
    external fun nativeApplyTemplate(
        modelHandle: Long,
        roles: Array<String>,
        contents: Array<String>,
        addAssistant: Boolean,
    ): String?

    /**
     * Returns `[promptTokens, reusedTokens, generatedTokens, prefillMicros,
     * decodeMicros, stopReason, cachedTokens]`.
     */
    external fun nativeGenerate(
        modelHandle: Long,
        ctxHandle: Long,
        sessionHandle: Long,
        tokens: IntArray,
        nKeep: Int,
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        minP: Float,
        repeatPenalty: Float,
        repeatLastN: Int,
        seed: Int,
        /** GBNF source, or empty for unconstrained decoding. */
        grammar: String,
        callback: ChatCallback,
    ): LongArray
}
