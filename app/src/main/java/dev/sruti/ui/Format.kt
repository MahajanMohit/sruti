package dev.sruti.ui

import java.util.Locale

/** Human-readable byte size. Binary units, because that is what storage UIs use. */
fun formatBytes(bytes: Long): String {
    if (bytes <= 0) return "0 B"
    val units = listOf("B", "KiB", "MiB", "GiB", "TiB")
    var value = bytes.toDouble()
    var unit = 0
    while (value >= 1024.0 && unit < units.lastIndex) {
        value /= 1024.0
        unit++
    }
    // Whole numbers for bytes and kibibytes; a decimal only where it carries
    // information.
    return if (unit <= 1) {
        String.format(Locale.US, "%.0f %s", value, units[unit])
    } else {
        String.format(Locale.US, "%.1f %s", value, units[unit])
    }
}

/** Parameter count as a short label, e.g. "1.24 B". */
fun formatParameters(count: Long): String = when {
    count <= 0 -> "unknown size"
    count >= 1_000_000_000 -> String.format(Locale.US, "%.2f B", count / 1e9)
    else -> String.format(Locale.US, "%.0f M", count / 1e6)
}
