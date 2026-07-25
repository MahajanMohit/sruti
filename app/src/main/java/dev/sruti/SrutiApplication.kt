package dev.sruti

import android.app.Application
import dagger.hilt.android.HiltAndroidApp
import dev.sruti.llm.NativeBackends

@HiltAndroidApp
class SrutiApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        // Registers ggml's CPU backend variants from the APK's library directory.
        // Everything that touches a model depends on this having happened.
        NativeBackends.ensureInitialized(this)
    }
}
