package dev.sruti.ui.theme

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext

// Dark is designed, not derived. Inverting a light palette produces muddy surfaces
// and blown-out accents; these are picked for an OLED panel where true black is
// genuinely black and the eye is far more sensitive to accent luminance.
private val DarkColors = darkColorScheme(
    primary = Color(0xFFB9C6FF),
    onPrimary = Color(0xFF01218A),
    primaryContainer = Color(0xFF1B36A8),
    onPrimaryContainer = Color(0xFFDDE1FF),
    secondary = Color(0xFFC2C5DD),
    onSecondary = Color(0xFF2C2F42),
    background = Color(0xFF0B0B0F),
    onBackground = Color(0xFFE4E1E9),
    surface = Color(0xFF0B0B0F),
    onSurface = Color(0xFFE4E1E9),
    surfaceVariant = Color(0xFF45464F),
    onSurfaceVariant = Color(0xFFC6C5D0),
    error = Color(0xFFFFB4AB),
    onError = Color(0xFF690005),
)

private val LightColors = lightColorScheme(
    primary = Color(0xFF3A4DBE),
    onPrimary = Color(0xFFFFFFFF),
    primaryContainer = Color(0xFFDDE1FF),
    onPrimaryContainer = Color(0xFF001159),
    secondary = Color(0xFF5A5D72),
    onSecondary = Color(0xFFFFFFFF),
    background = Color(0xFFFBF8FF),
    onBackground = Color(0xFF1B1B21),
    surface = Color(0xFFFBF8FF),
    onSurface = Color(0xFF1B1B21),
    surfaceVariant = Color(0xFFE2E1EC),
    onSurfaceVariant = Color(0xFF45464F),
    error = Color(0xFFBA1A1A),
    onError = Color(0xFFFFFFFF),
)

@Composable
fun SrutiTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    /** Material You wallpaper extraction, available from Android 12. */
    dynamicColor: Boolean = true,
    content: @Composable () -> Unit,
) {
    val context = LocalContext.current
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }
        darkTheme -> DarkColors
        else -> LightColors
    }

    // Provided alongside the colour scheme because haptics are part of the same
    // design system: both describe how the app presents itself, and a composable
    // that needs one usually needs the other.
    CompositionLocalProvider(LocalHaptics provides rememberHaptics()) {
        MaterialTheme(
            colorScheme = colorScheme,
            typography = SrutiTypography,
            content = content,
        )
    }
}
