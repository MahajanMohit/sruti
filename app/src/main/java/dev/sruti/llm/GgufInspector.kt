package dev.sruti.llm

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/** One labelled fact about a model file. */
data class GgufFact(val label: String, val value: String)

/**
 * Reads a converted model's metadata back out of its own file.
 *
 * The sidecar written at conversion time records what the app intended; this
 * records what actually ended up in the file. They should agree, and when they
 * do not the file is the truth — which is the whole reason for reading it rather
 * than displaying the sidecar twice.
 */
object GgufInspector {

    // No backend registration needed: reading the header is pure file parsing,
    // and loading the library is enough.
    suspend fun inspect(file: File): List<GgufFact> =
        withContext(Dispatchers.IO) {
            runCatching { LlamaBridge.nativeGgufSummary(file.absolutePath) }
                .getOrDefault("")
                .lineSequence()
                .mapNotNull { line ->
                    val tab = line.indexOf('\t')
                    if (tab <= 0) null else GgufFact(line.take(tab), line.substring(tab + 1))
                }
                .toList()
        }
}
