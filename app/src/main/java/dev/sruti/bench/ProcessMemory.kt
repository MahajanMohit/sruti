package dev.sruti.bench

import java.io.File

/**
 * Resident-set readings from `/proc/self/status`.
 *
 * Java heap metrics are the wrong instrument here: model weights are mmapped by
 * native code and never appear on the Dalvik heap. RSS and its high-water mark are
 * what determine whether the low-memory killer takes an interest in the process.
 */
object ProcessMemory {

    data class Snapshot(
        /** Current resident set size in bytes. */
        val residentBytes: Long,
        /** Peak resident set size in bytes since process start. */
        val peakResidentBytes: Long,
    )

    fun read(): Snapshot {
        val fields = runCatching {
            File("/proc/self/status").readLines()
                .mapNotNull { line ->
                    val parts = line.split(":", limit = 2)
                    if (parts.size != 2) return@mapNotNull null
                    val kb = parts[1].trim().removeSuffix(" kB").trim().toLongOrNull()
                        ?: return@mapNotNull null
                    parts[0].trim() to kb * 1024
                }
                .toMap()
        }.getOrDefault(emptyMap())

        return Snapshot(
            residentBytes = fields["VmRSS"] ?: 0L,
            peakResidentBytes = fields["VmHWM"] ?: 0L,
        )
    }
}
