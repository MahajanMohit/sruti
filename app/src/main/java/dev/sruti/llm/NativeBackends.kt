package dev.sruti.llm

import android.content.Context
import java.util.concurrent.atomic.AtomicBoolean

/**
 * One-time initialisation of ggml's backend registry.
 *
 * ggml ships a CPU backend per ARM feature set (armv8.0 through armv9.2) and picks
 * the best one the device can actually execute. That is what lets a single APK run
 * on a budget part without crashing and still use dot-product and int8 matmul
 * instructions on a flagship — rather than compiling for one class of chip and
 * failing with SIGILL everywhere else.
 *
 * Those backends are separate shared objects, discovered by scanning a directory.
 * ggml's own defaults search the executable's directory and the working directory,
 * which on Android are /system/bin and / — never where an APK's libraries live. So
 * the path has to be supplied, and getting this wrong means no backend registers
 * and nothing runs.
 */
object NativeBackends {

    private val initialized = AtomicBoolean(false)
    private var libraryDir: String = ""

    /** Idempotent; safe to call from anywhere. */
    fun ensureInitialized(context: Context) {
        if (initialized.compareAndSet(false, true)) {
            libraryDir = context.applicationInfo.nativeLibraryDir
            LlamaBridge.nativeBackendInit(libraryDir)
        }
    }

    /** Registered backend count. Zero means inference cannot work at all. */
    fun backendCount(): Int = LlamaBridge.nativeBackendCount()

    /** Whether a usable compute backend was found. */
    fun isUsable(): Boolean = backendCount() > 0

    /**
     * What was found and where, for when nothing loaded.
     *
     * "No compute backend is available" on its own gives the user nothing to act
     * on and gives a bug report nothing to work from. This names the directory and
     * lists what is in it.
     */
    fun diagnostics(): String =
        runCatching { LlamaBridge.nativeBackendDiagnostics(libraryDir) }
            .getOrElse { "could not read backend diagnostics: ${it.message}" }
}
