package dev.sruti.ui.theme

import androidx.compose.animation.core.FiniteAnimationSpec
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.spring
import androidx.compose.ui.unit.IntSize

/**
 * The single source of motion in the app.
 *
 * Every animation pulls its spec from here. Ad-hoc durations scattered across
 * composables are what makes an interface feel assembled rather than designed —
 * elements that should move as one end up drifting apart by a few frames, and the
 * eye reads that as cheapness even when it cannot name the cause.
 *
 * Springs rather than durations, because spring motion stays continuous when a
 * gesture interrupts it mid-flight. Predictive back depends on exactly that.
 */
object Motion {

    /** Small, frequent state changes: toggles, selection, icon swaps. */
    fun <T> quick(): FiniteAnimationSpec<T> = spring(
        dampingRatio = Spring.DampingRatioNoBouncy,
        stiffness = Spring.StiffnessMediumLow,
    )

    /** Content entering or leaving: sheets, expanding cards, list items. */
    fun <T> standard(): FiniteAnimationSpec<T> = spring(
        dampingRatio = 0.85f,
        stiffness = 380f,
    )

    /** Large surfaces travelling a long distance: full-screen transitions. */
    fun <T> expressive(): FiniteAnimationSpec<T> = spring(
        dampingRatio = 0.75f,
        stiffness = 240f,
    )

    /**
     * Size changes need a visibility threshold in the size domain, otherwise the
     * spring settles a fraction of a pixel at a time and layout keeps thrashing.
     */
    fun size(): FiniteAnimationSpec<IntSize> = spring(
        dampingRatio = 0.85f,
        stiffness = 380f,
        visibilityThreshold = IntSize(1, 1),
    )
}
