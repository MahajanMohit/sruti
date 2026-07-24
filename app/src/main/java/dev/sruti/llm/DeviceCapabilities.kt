package dev.sruti.llm

import java.io.File

/**
 * CPU topology probing for thread-count selection.
 *
 * Snapdragon 8-class parts are heterogeneous (prime + performance + efficiency
 * clusters). Handing llama.cpp every core is actively harmful: decode is a
 * synchronous fork-join over all threads, so the slowest core sets the pace and
 * little cores drag the whole step down. Counting only the fast cores is the
 * cheap, reliable heuristic.
 */
object DeviceCapabilities {

    /** Max frequency in kHz for each CPU, or null where it could not be read. */
    private fun coreMaxFrequencies(): List<Long?> {
        val cpuCount = Runtime.getRuntime().availableProcessors()
        return (0 until cpuCount).map { cpu ->
            runCatching {
                File("/sys/devices/system/cpu/cpu$cpu/cpufreq/cpuinfo_max_freq")
                    .readText()
                    .trim()
                    .toLong()
            }.getOrNull()
        }
    }

    /**
     * Number of cores in the fastest frequency tier.
     *
     * Cores within 15% of the top frequency count as one tier, which groups the
     * prime core together with the performance cluster on typical big.LITTLE
     * layouts while still excluding the efficiency cores.
     */
    fun performanceCoreCount(): Int {
        val freqs = coreMaxFrequencies().filterNotNull()
        if (freqs.isEmpty()) {
            // sysfs unreadable (some OEM kernels restrict it) — assume half the
            // cores are fast, which is right for most current layouts.
            return (Runtime.getRuntime().availableProcessors() / 2).coerceAtLeast(1)
        }
        val max = freqs.max()
        val threshold = (max * 0.85).toLong()
        return freqs.count { it >= threshold }.coerceAtLeast(1)
    }

    /**
     * Thread count for inference. Capped at 6: beyond that, memory bandwidth
     * rather than compute is the limit on phone-class parts, and extra threads
     * only add scheduling jitter and heat.
     */
    fun recommendedThreadCount(): Int = performanceCoreCount().coerceIn(2, 6)
}
