package dev.sruti.convert

/**
 * Raw JNI surface over the safetensors -> GGUF converter. Use [ModelConverter].
 */
internal object ConverterBridge {

    /** Callbacks fire on the calling thread while conversion runs. */
    interface ConvertCallback {
        /** Return false to cancel. */
        fun onProgress(stage: Int, detail: String, tensorsDone: Long, tensorsTotal: Long): Boolean
        fun onWarning(message: String)
    }

    init {
        System.loadLibrary("sruti_llm")
    }

    /** Returns a JSON summary; see [CheckpointInfo]. */
    external fun nativeInspectConfig(configJson: String): String

    external fun nativeEstimatedWorkingBytes(checkpointBytes: Long, quantType: String): Long

    /** Returns null on success, or an error message. */
    external fun nativeConvert(
        modelDir: String,
        outPath: String,
        quantType: String,
        modelName: String,
        nThreads: Int,
        callback: ConvertCallback,
    ): String?
}
