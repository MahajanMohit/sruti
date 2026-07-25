package dev.sruti.hub

import android.content.Context
import androidx.compose.runtime.Immutable
import dev.sruti.convert.QuantType
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File

/**
 * A converted model that is ready to run.
 *
 * Persisted as a JSON sidecar next to the .gguf rather than in a database: the
 * filesystem is the real source of truth here, and a table would only drift from
 * it the first time a file is deleted or a conversion is interrupted.
 */
@Serializable
data class LocalModel(
    val fileName: String,
    val displayName: String,
    val sourceRepo: String = "",
    val architecture: String = "",
    val quantType: String = QuantType.Q4_K_M.id,
    val parameterCount: Long = 0,
    val convertedAtMillis: Long = 0,
) {
    val quant: QuantType get() = QuantType.fromId(quantType)
    val parametersInBillions: Double get() = parameterCount / 1e9
}

/** A model on disk, with its actual file. */
// java.io.File is a path holder, never mutated here, but Compose cannot know
// that. Without this every list of models is unstable, and every screen holding
// one recomposes on any unrelated state change.
@Immutable
data class InstalledModel(
    val metadata: LocalModel,
    val file: File,
) {
    val sizeBytes: Long get() = file.length()
}

/** A checkpoint directory awaiting conversion. */
@Immutable
data class StagedCheckpoint(
    val directory: File,
    val repoId: String,
) {
    val sizeBytes: Long
        get() = directory.walkTopDown().filter { it.isFile }.sumOf { it.length() }
}

/**
 * Owns the app's model directories.
 *
 * Two kinds of thing live here and they have very different lifetimes: downloaded
 * safetensors checkpoints, which are large and disposable once converted, and the
 * converted GGUFs, which are what the user actually keeps.
 */
class ModelStore(
    context: Context,
    private val dispatcher: CoroutineDispatcher,
) {

    private val json = Json { ignoreUnknownKeys = true; prettyPrint = true }

    /** Downloaded safetensors, one directory per repository. */
    val checkpointsDir: File = File(context.filesDir, "checkpoints").apply { mkdirs() }

    /** Converted, runnable models. */
    val modelsDir: File = File(context.filesDir, "models").apply { mkdirs() }

    fun checkpointDir(repoId: String): File =
        File(checkpointsDir, repoId.replace('/', '~'))

    fun repoIdOf(dir: File): String = dir.name.replace('~', '/')

    fun outputFileFor(repoId: String, quantType: QuantType): File =
        File(modelsDir, "${repoId.substringAfterLast('/')}-${quantType.id}.gguf")

    suspend fun installedModels(): List<InstalledModel> = withContext(dispatcher) {
        modelsDir.listFiles { f -> f.isFile && f.extension == "gguf" }
            ?.sortedByDescending { it.lastModified() }
            ?.map { file -> InstalledModel(readMetadata(file), file) }
            ?: emptyList()
    }

    suspend fun stagedCheckpoints(): List<StagedCheckpoint> = withContext(dispatcher) {
        checkpointsDir.listFiles { f -> f.isDirectory }
            ?.sortedByDescending { it.lastModified() }
            ?.map { dir -> StagedCheckpoint(dir, repoIdOf(dir)) }
            ?: emptyList()
    }

    suspend fun writeMetadata(file: File, metadata: LocalModel) = withContext(dispatcher) {
        runCatching { sidecarFor(file).writeText(json.encodeToString(metadata)) }
        Unit
    }

    suspend fun delete(model: InstalledModel) = withContext(dispatcher) {
        model.file.delete()
        sidecarFor(model.file).delete()
        Unit
    }

    suspend fun delete(checkpoint: StagedCheckpoint) = withContext(dispatcher) {
        checkpoint.directory.deleteRecursively()
        Unit
    }

    /** Total bytes held by checkpoints and converted models. */
    suspend fun usedBytes(): Long = withContext(dispatcher) {
        listOf(checkpointsDir, modelsDir).sumOf { root ->
            root.walkTopDown().filter { it.isFile }.sumOf { it.length() }
        }
    }

    private fun sidecarFor(file: File) = File(file.absolutePath + ".json")

    private fun readMetadata(file: File): LocalModel {
        val sidecar = sidecarFor(file)
        if (sidecar.isFile) {
            runCatching { return json.decodeFromString<LocalModel>(sidecar.readText()) }
        }
        // A model sideloaded by hand has no sidecar; it is still perfectly runnable.
        return LocalModel(
            fileName = file.name,
            displayName = file.nameWithoutExtension,
            convertedAtMillis = file.lastModified(),
        )
    }
}
