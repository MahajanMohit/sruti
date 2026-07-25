package dev.sruti.llm

/**
 * Raw JNI surface over llama.cpp. Every method here is unsafe by design: handles are
 * native pointers and there is no lifecycle enforcement. Use [LlamaEngine] instead.
 */
internal object LlamaBridge {

    /** Receives decoded text as it is produced. Return false to stop generation. */
    fun interface TokenCallback {
        fun onToken(piece: String): Boolean
    }

    init {
        System.loadLibrary("sruti_llm")
    }

    /** [nativeLibDir] must be the app's nativeLibraryDir; see [NativeBackends]. */
    external fun nativeBackendInit(nativeLibDir: String)
    external fun nativeBackendCount(): Int
    external fun nativeBackendFree()

    external fun nativeLoadModel(path: String, nGpuLayers: Int): Long
    external fun nativeFreeModel(handle: Long)

    external fun nativeModelDesc(handle: Long): String
    external fun nativeModelParamCount(handle: Long): Long
    external fun nativeModelSizeBytes(handle: Long): Long
    external fun nativeModelCtxTrain(handle: Long): Int

    external fun nativeNewContext(modelHandle: Long, nCtx: Int, nThreads: Int, nBatch: Int): Long
    external fun nativeFreeContext(handle: Long)
    external fun nativeResetContext(handle: Long)

    external fun nativeTokenize(modelHandle: Long, text: String, addSpecial: Boolean): IntArray

    /** Returns `[promptTokens, generatedTokens, prefillMicros, decodeMicros, stopReason]`. */
    external fun nativeGenerate(
        modelHandle: Long,
        ctxHandle: Long,
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        minP: Float,
        seed: Int,
        callback: TokenCallback,
    ): LongArray
}
