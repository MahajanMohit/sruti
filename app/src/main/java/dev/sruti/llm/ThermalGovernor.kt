package dev.sruti.llm

import android.content.Context
import android.os.PowerManager

/**
 * Decides how fast generation is allowed to run.
 *
 * Sustained decode draws several watts and a phone will throttle within minutes.
 * Left alone, the SoC sheds clock abruptly part-way through a reply — the user
 * sees text that starts fast and then crawls, with no explanation.
 *
 * Backing off deliberately, a little earlier, keeps the part more even and the
 * device below the point where the kernel intervenes. The cost is real and is not
 * hidden: [status] is surfaced in the UI so a slow reply is explained rather than
 * mysterious.
 */
class ThermalGovernor(context: Context) {

    private val powerManager =
        context.getSystemService(Context.POWER_SERVICE) as PowerManager

    enum class Level(val label: String, val microsPerToken: Long) {
        /** No intervention. */
        Unrestricted("", 0),

        /** Gentle: costs little, buys headroom before throttling begins. */
        Easing("Easing off to manage heat", 15_000),

        /** The device is already hot; a real slowdown beats a hard throttle. */
        Throttled("Slowed down — the device is warm", 60_000),

        /** Close to intervention. Generation still completes, slowly. */
        Severe("Heavily throttled — let the device cool", 200_000),
    }

    /** Current thermal state, sampled fresh. */
    fun level(): Level = when (powerManager.currentThermalStatus) {
        PowerManager.THERMAL_STATUS_NONE -> Level.Unrestricted
        PowerManager.THERMAL_STATUS_LIGHT -> Level.Easing
        PowerManager.THERMAL_STATUS_MODERATE -> Level.Throttled
        else -> Level.Severe
    }

    /**
     * Microseconds to pause before the next token.
     *
     * Sampled per call rather than cached: thermal status changes during a long
     * generation, which is exactly when the governor needs to react.
     */
    fun pacingMicros(): Long = level().microsPerToken
}
