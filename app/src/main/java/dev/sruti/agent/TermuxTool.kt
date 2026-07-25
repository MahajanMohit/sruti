package dev.sruti.agent

import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import java.io.File

/**
 * A real POSIX shell, via Termux's RUN_COMMAND service.
 *
 * This is the single largest capability jump available without root or an
 * accessibility service, and the reason it is worth the integration cost: one
 * correct shell command accomplishes what would otherwise be a ten-step plan the
 * model cannot hold. Playing to that — one command rather than a long chain — is
 * the design, not a workaround.
 *
 * It is also unbounded. A shell can do anything the shell can do, so this is off
 * until the user turns it on, and every command is shown verbatim before it runs.
 * The confirmation is not a formality; it is the only meaningful control here.
 */
class TermuxTool(
    private val context: Context,
    private val dispatcher: CoroutineDispatcher,
) {

    /** Whether Termux is installed at all. */
    fun isTermuxInstalled(): Boolean = runCatching {
        context.packageManager.getPackageInfo(TERMUX_PACKAGE, 0)
        true
    }.getOrDefault(false)

    /**
     * Whether the RUN_COMMAND permission has been granted.
     *
     * Termux also requires `allow-external-apps=true` in its own
     * `~/.termux/termux.properties`, which cannot be detected from here — so a
     * granted permission is necessary but not sufficient, and a failure to run
     * says so rather than blaming the command.
     */
    fun hasPermission(): Boolean =
        ContextCompat.checkSelfPermission(context, RUN_COMMAND_PERMISSION) ==
            PackageManager.PERMISSION_GRANTED

    fun isAvailable(): Boolean = isTermuxInstalled() && hasPermission()

    fun tool(): Tool = Tool(
        name = "run_shell",
        description = "Run a shell command in Termux and return its output",
        tier = ToolTier.Shell,
        requiresConfirmation = true,
        keywords = listOf(
            "shell", "command", "terminal", "bash", "git", "python", "script", "run",
        ),
        parameters = listOf(
            ToolParameter("command", ToolParameter.Type.String, "The command line to run"),
        ),
        execute = { args -> run(args["command"].orEmpty()) },
    )

    private suspend fun run(command: String): ToolResult = withContext(dispatcher) {
        if (command.isBlank()) {
            return@withContext ToolResult.Failure("empty command")
        }
        if (!isTermuxInstalled()) {
            return@withContext ToolResult.Failure(
                "Termux is not installed. Install it from F-Droid to enable shell access.",
            )
        }
        if (!hasPermission()) {
            return@withContext ToolResult.Failure(
                "Termux permission has not been granted. Enable shell access in settings.",
            )
        }

        // Output comes back through a file rather than the result intent: command
        // output routinely exceeds what an intent extra will carry.
        val outputDir = File(context.filesDir, "shell").apply { mkdirs() }
        val stdout = File(outputDir, "stdout-${System.currentTimeMillis()}.txt")

        val wrapped = "{ $command ; } > ${shellQuote(stdout.absolutePath)} 2>&1"

        val intent = Intent().apply {
            setClassName(TERMUX_PACKAGE, RUN_COMMAND_SERVICE)
            action = ACTION_RUN_COMMAND
            putExtra(EXTRA_COMMAND_PATH, "/data/data/$TERMUX_PACKAGE/files/usr/bin/bash")
            putExtra(EXTRA_ARGUMENTS, arrayOf("-c", wrapped))
            putExtra(EXTRA_BACKGROUND, true)
            putExtra(EXTRA_SESSION_ACTION, "0")
        }

        val started = runCatching { context.startService(intent) }.isSuccess
        if (!started) {
            return@withContext ToolResult.Failure(
                "Termux refused the command. Set allow-external-apps=true in " +
                    "~/.termux/termux.properties and restart Termux.",
            )
        }

        // Termux's background result callback is awkward to receive reliably, so
        // completion is detected by watching the output file settle.
        val output = awaitOutput(stdout)
        stdout.delete()

        when {
            output == null -> ToolResult.Failure(
                "Command timed out after ${TIMEOUT_MILLIS / 1000}s, or Termux did not run it. " +
                    "Check that allow-external-apps=true is set in termux.properties.",
            )
            output.isBlank() -> ToolResult.Success("(no output)")
            else -> ToolResult.Success(output.take(MAX_OUTPUT))
        }
    }

    /** Waits for the output file to appear and stop growing. */
    private suspend fun awaitOutput(file: File): String? = withTimeoutOrNull(TIMEOUT_MILLIS) {
        val done = CompletableDeferred<String>()
        var lastSize = -1L
        var stableFor = 0

        while (!done.isCompleted) {
            kotlinx.coroutines.delay(POLL_MILLIS)
            if (!file.exists()) continue

            val size = file.length()
            if (size == lastSize) {
                stableFor += POLL_MILLIS.toInt()
                // A command that writes nothing still creates the file, so
                // "unchanged for a while" is the only available completion signal.
                if (stableFor >= STABLE_MILLIS) {
                    done.complete(file.readText())
                }
            } else {
                lastSize = size
                stableFor = 0
            }
        }
        done.await()
    }

    /** Single-quotes a path for the shell. */
    private fun shellQuote(text: String): String = "'" + text.replace("'", "'\\''") + "'"

    companion object {
        const val TERMUX_PACKAGE = "com.termux"
        const val RUN_COMMAND_PERMISSION = "com.termux.permission.RUN_COMMAND"

        private const val RUN_COMMAND_SERVICE = "com.termux.app.RunCommandService"
        private const val ACTION_RUN_COMMAND = "com.termux.RUN_COMMAND"
        private const val EXTRA_COMMAND_PATH = "com.termux.RUN_COMMAND_PATH"
        private const val EXTRA_ARGUMENTS = "com.termux.RUN_COMMAND_ARGUMENTS"
        private const val EXTRA_BACKGROUND = "com.termux.RUN_COMMAND_BACKGROUND"
        private const val EXTRA_SESSION_ACTION = "com.termux.RUN_COMMAND_SESSION_ACTION"

        private const val TIMEOUT_MILLIS = 60_000L
        private const val POLL_MILLIS = 250L
        private const val STABLE_MILLIS = 750
        private const val MAX_OUTPUT = 4000
    }
}
