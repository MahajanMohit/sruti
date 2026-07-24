package dev.sruti.work

import dev.sruti.convert.QuantType
import java.io.File

/** What the acquisition pipeline is doing right now. */
sealed interface ModelJobState {
    data object Idle : ModelJobState

    data class Running(
        val repoId: String,
        val quantType: QuantType,
        val phase: Phase,
        val detail: String = "",
        /** Null when the phase cannot report a meaningful fraction. */
        val fraction: Float? = null,
        val bytesDone: Long = 0,
        val bytesTotal: Long = 0,
    ) : ModelJobState {
        enum class Phase(val label: String) {
            Resolving("Checking model"),
            Downloading("Downloading"),
            Converting("Converting"),
            Cleaning("Cleaning up"),
        }
    }

    data class Failed(val repoId: String, val message: String) : ModelJobState

    data class Succeeded(
        val repoId: String,
        val outputFile: File,
        val warnings: List<String> = emptyList(),
    ) : ModelJobState
}
