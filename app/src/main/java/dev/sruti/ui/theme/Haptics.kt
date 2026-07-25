package dev.sruti.ui.theme

import android.content.Context
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import androidx.compose.runtime.Composable
import androidx.compose.runtime.Immutable
import androidx.compose.runtime.ProvidableCompositionLocal
import androidx.compose.runtime.compositionLocalOf
import androidx.compose.runtime.remember
import androidx.compose.ui.platform.LocalContext

/**
 * What just happened, in the vocabulary the hand understands.
 *
 * Named for meaning rather than for waveform: the point of a haptic system is
 * that the same event always feels the same everywhere in the app, and that two
 * different events never feel alike. Callers say what occurred; this decides how
 * it feels.
 */
enum class Haptic {
    /** A message was sent — the user's own action, acknowledged. */
    Send,

    /** The model produced its first token. The wait is over. */
    FirstToken,

    /** A reply finished cleanly. */
    Complete,

    /** The agent is about to run a tool. Deliberately the most distinct of these. */
    ToolCall,

    /** Something failed. */
    Error,

    /** A model or conversation was deleted. */
    Destructive,

    /** A selection changed: a filter chip, a model in the menu. */
    Select,
}

/**
 * Plays haptic signatures.
 *
 * Composition primitives rather than raw durations wherever the device supports
 * them: a `CLICK` is tuned by the manufacturer for that specific actuator, and a
 * hand-written 20 ms buzz feels like a cheap phone on hardware that can do
 * better. Where primitives are unsupported the fallback is a plain waveform,
 * which is worse but never silent.
 *
 * Every call is best-effort. A device with no vibrator, a user who turned haptics
 * off, and a manufacturer ROM that rejects an effect all end up in the same
 * place: nothing happens, and nothing throws.
 */
@Immutable
class Haptics private constructor(private val vibrator: Vibrator?) {

    private val supportsPrimitives: Boolean = vibrator?.areAllPrimitivesSupported(
        VibrationEffect.Composition.PRIMITIVE_CLICK,
        VibrationEffect.Composition.PRIMITIVE_TICK,
    ) == true

    fun play(haptic: Haptic) {
        val vibrator = vibrator ?: return
        if (!vibrator.hasVibrator()) return

        runCatching {
            vibrator.vibrate(effectFor(haptic))
        }
    }

    private fun effectFor(haptic: Haptic): VibrationEffect {
        if (!supportsPrimitives) return fallbackFor(haptic)

        return when (haptic) {
            // A single soft tick. Sending happens constantly and anything
            // stronger becomes irritating within a session.
            Haptic.Send -> VibrationEffect.startComposition()
                .addPrimitive(VibrationEffect.Composition.PRIMITIVE_TICK, 0.4f)
                .compose()

            // Lighter still. This fires while the user is reading, and a
            // noticeable jolt at that moment reads as an error.
            Haptic.FirstToken -> VibrationEffect.startComposition()
                .addPrimitive(VibrationEffect.Composition.PRIMITIVE_TICK, 0.25f)
                .compose()

            // Two rising ticks: a small sense of arrival without a thump.
            Haptic.Complete -> VibrationEffect.startComposition()
                .addPrimitive(VibrationEffect.Composition.PRIMITIVE_TICK, 0.35f)
                .addPrimitive(VibrationEffect.Composition.PRIMITIVE_CLICK, 0.5f, 60)
                .compose()

            // The one signature that must be unmistakable through a pocket: the
            // agent is about to do something to the device.
            Haptic.ToolCall -> VibrationEffect.startComposition()
                .addPrimitive(VibrationEffect.Composition.PRIMITIVE_CLICK, 0.7f)
                .addPrimitive(VibrationEffect.Composition.PRIMITIVE_CLICK, 0.7f, 90)
                .compose()

            // Falling rather than rising — the inverse of Complete, which is what
            // makes the two distinguishable without looking.
            Haptic.Error -> VibrationEffect.startComposition()
                .addPrimitive(VibrationEffect.Composition.PRIMITIVE_CLICK, 0.8f)
                .addPrimitive(VibrationEffect.Composition.PRIMITIVE_TICK, 0.4f, 80)
                .compose()

            Haptic.Destructive -> VibrationEffect.startComposition()
                .addPrimitive(VibrationEffect.Composition.PRIMITIVE_CLICK, 0.6f)
                .compose()

            Haptic.Select -> VibrationEffect.startComposition()
                .addPrimitive(VibrationEffect.Composition.PRIMITIVE_TICK, 0.3f)
                .compose()
        }
    }

    /** Durations chosen to keep the same relative weights the primitives have. */
    private fun fallbackFor(haptic: Haptic): VibrationEffect = when (haptic) {
        Haptic.Send -> VibrationEffect.createOneShot(12, 60)
        Haptic.FirstToken -> VibrationEffect.createOneShot(8, 40)
        Haptic.Complete -> VibrationEffect.createWaveform(longArrayOf(0, 10, 50, 16), -1)
        Haptic.ToolCall -> VibrationEffect.createWaveform(longArrayOf(0, 20, 70, 20), -1)
        Haptic.Error -> VibrationEffect.createWaveform(longArrayOf(0, 24, 60, 10), -1)
        Haptic.Destructive -> VibrationEffect.createOneShot(18, 120)
        Haptic.Select -> VibrationEffect.createOneShot(8, 50)
    }

    companion object {
        fun create(context: Context): Haptics {
            val manager = context.getSystemService(VibratorManager::class.java)
            return Haptics(manager?.defaultVibrator)
        }

        /** Does nothing; for previews and tests. */
        val None = Haptics(null)
    }
}

/**
 * The app's haptics, available anywhere without threading it through parameters.
 *
 * Defaults to silence rather than throwing, so a composable used outside the app
 * theme — a preview, a test — still renders.
 */
val LocalHaptics: ProvidableCompositionLocal<Haptics> = compositionLocalOf { Haptics.None }

@Composable
fun rememberHaptics(): Haptics {
    val context = LocalContext.current
    return remember(context) { Haptics.create(context) }
}
