package dev.sruti.hub

import android.os.StatFs
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.IOException
import java.io.RandomAccessFile
import java.security.MessageDigest

sealed interface DownloadEvent {
    data class Progress(
        val fileIndex: Int,
        val fileCount: Int,
        val currentFile: String,
        val bytesDone: Long,
        val bytesTotal: Long,
    ) : DownloadEvent {
        val fraction: Float get() = if (bytesTotal > 0) bytesDone.toFloat() / bytesTotal else 0f
    }

    @JvmInline
    value class Skipped(val path: String) : DownloadEvent

    data class Completed(val directory: File) : DownloadEvent
}

class DownloadException(message: String, cause: Throwable? = null) : IOException(message, cause)

/**
 * Downloads a checkpoint's files into a directory the converter can read.
 *
 * Built for the failure modes that actually happen on a phone: the download is
 * multiple gigabytes over a connection that drops, on a device that may run out of
 * space, with a user who will background the app. So every file resumes from where
 * it stopped, is verified against the hash the Hub publishes, and only moves into
 * its final name once complete — a half-written shard must never look finished.
 */
class CheckpointDownloader(
    private val client: OkHttpClient,
    private val api: HuggingFaceApi,
    private val dispatcher: CoroutineDispatcher,
) {

    /** Free bytes on the volume holding [dir]. */
    fun freeBytes(dir: File): Long = runCatching {
        val stat = StatFs(dir.absolutePath)
        stat.availableBlocksLong * stat.blockSizeLong
    }.getOrDefault(0L)

    fun download(
        checkpoint: RemoteCheckpoint,
        targetDir: File,
        /** Extra headroom to leave free, for the conversion that follows. */
        reserveBytes: Long = 0,
    ): Flow<DownloadEvent> = flow {
        targetDir.mkdirs()

        val files = checkpoint.requiredFiles
        if (files.isEmpty()) {
            throw DownloadException("${checkpoint.repoId} has no safetensors weights to download")
        }

        // Preflight against what is still outstanding, so a resumed download is
        // not blocked by space its completed files already occupy.
        val outstanding = files.sumOf { file ->
            val done = File(targetDir, file.path).takeIf { it.isFile }?.length() ?: 0
            (file.actualSize - done).coerceAtLeast(0)
        }
        val free = freeBytes(targetDir)
        if (free > 0 && free < outstanding + reserveBytes) {
            throw DownloadException(
                "Not enough space: need ${(outstanding + reserveBytes) shr 20} MiB " +
                    "but only ${free shr 20} MiB is free",
            )
        }

        val totalBytes = files.sumOf { it.actualSize }
        var completedBytes = 0L

        files.forEachIndexed { index, file ->
            val target = File(targetDir, file.path)

            if (isAlreadyComplete(target, file)) {
                completedBytes += file.actualSize
                emit(DownloadEvent.Skipped(file.path))
                emit(
                    DownloadEvent.Progress(
                        fileIndex = index + 1,
                        fileCount = files.size,
                        currentFile = file.path,
                        bytesDone = completedBytes,
                        bytesTotal = totalBytes,
                    ),
                )
                return@forEachIndexed
            }

            downloadFile(
                checkpoint = checkpoint,
                file = file,
                target = target,
                onBytes = { bytesInFile ->
                    emit(
                        DownloadEvent.Progress(
                            fileIndex = index + 1,
                            fileCount = files.size,
                            currentFile = file.path,
                            bytesDone = completedBytes + bytesInFile,
                            bytesTotal = totalBytes,
                        ),
                    )
                },
            )
            completedBytes += file.actualSize
        }

        emit(DownloadEvent.Completed(targetDir))
    }.buffer(Channel.UNLIMITED).flowOn(dispatcher)

    private fun isAlreadyComplete(target: File, file: RemoteFile): Boolean {
        if (!target.isFile) return false
        if (file.actualSize > 0 && target.length() != file.actualSize) return false
        // Size alone is weak evidence, but hashing several GB on every launch is
        // worse. The hash is verified when the file is written; a size match on an
        // already-verified file is enough to skip it.
        return true
    }

    private suspend fun downloadFile(
        checkpoint: RemoteCheckpoint,
        file: RemoteFile,
        target: File,
        onBytes: suspend (Long) -> Unit,
    ) {
        target.parentFile?.mkdirs()
        val partial = File(target.absolutePath + ".part")

        val digest = MessageDigest.getInstance("SHA-256")
        var existing = if (partial.isFile) partial.length() else 0L

        // Seed the digest with what was already fetched, otherwise a resumed file
        // hashes only its tail and always fails verification.
        if (existing > 0) {
            partial.inputStream().use { input ->
                val buffer = ByteArray(1 shl 16)
                var remaining = existing
                while (remaining > 0) {
                    val read = input.read(buffer, 0, minOf(buffer.size.toLong(), remaining).toInt())
                    if (read <= 0) break
                    digest.update(buffer, 0, read)
                    remaining -= read
                }
                if (remaining > 0) {
                    // Could not re-read the partial file; start over rather than
                    // carry a digest that no longer matches the bytes on disk.
                    existing = 0
                    digest.reset()
                }
            }
        }

        val url = api.downloadUrl(checkpoint.repoId, file.path, checkpoint.revision)
        val builder = Request.Builder().url(url)
        api.authHeader()?.let { (name, value) -> builder.header(name, value) }
        if (existing > 0) {
            builder.header("Range", "bytes=$existing-")
        }

        client.newCall(builder.build()).execute().use { response ->
            // A server that ignores Range replies 200 with the whole file; keeping
            // the partial bytes would corrupt it.
            if (existing > 0 && response.code != 206) {
                existing = 0
                digest.reset()
                partial.delete()
            }
            if (!response.isSuccessful) {
                throw DownloadException("Download of ${file.path} failed with HTTP ${response.code}")
            }

            val body = response.body ?: throw DownloadException("Empty response for ${file.path}")

            RandomAccessFile(partial, "rw").use { out ->
                out.setLength(existing)
                out.seek(existing)

                val buffer = ByteArray(1 shl 16)
                var written = existing
                body.byteStream().use { input ->
                    while (true) {
                        currentCoroutineContext().ensureActive()
                        val read = input.read(buffer)
                        if (read <= 0) break
                        out.write(buffer, 0, read)
                        digest.update(buffer, 0, read)
                        written += read
                        onBytes(written)
                    }
                }
            }
        }

        val expected = file.sha256
        if (expected != null) {
            val actual = digest.digest().joinToString("") { "%02x".format(it) }
            if (!actual.equals(expected, ignoreCase = true)) {
                partial.delete()
                throw DownloadException(
                    "${file.path} failed checksum verification; the download was " +
                        "corrupted and has been discarded",
                )
            }
        }

        // Only now does the file get its real name, so an interrupted download can
        // never be mistaken for a complete one.
        if (!partial.renameTo(target)) {
            throw DownloadException("Could not move ${file.path} into place")
        }
    }
}
