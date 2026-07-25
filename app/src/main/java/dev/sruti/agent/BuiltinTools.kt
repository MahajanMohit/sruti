package dev.sruti.agent

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File

/**
 * The tools available without any elevated permission.
 *
 * File access is confined to a single directory inside the app's own storage.
 * That is a real boundary, not a naming convention: paths are resolved and then
 * checked to be inside it, so `../` cannot walk out.
 */
class BuiltinTools(
    private val context: Context,
    private val client: OkHttpClient,
    private val dispatcher: CoroutineDispatcher,
) {

    /** Everything the agent may read or write lives here. */
    val workspace: File = File(context.filesDir, "workspace").apply { mkdirs() }

    fun all(): List<Tool> = listOf(
        readFile(),
        writeFile(),
        listFiles(),
        fetchUrl(),
        readClipboard(),
        writeClipboard(),
    )

    private fun resolve(path: String): File? = resolveInWorkspace(workspace, path)

    private fun readFile() = Tool(
        name = "read_file",
        description = "Read a text file from the workspace",
        keywords = listOf("open", "load", "contents", "cat", "show"),
        parameters = listOf(
            ToolParameter("path", ToolParameter.Type.String, "Path relative to the workspace"),
            ToolParameter(
                name = "max_lines",
                type = ToolParameter.Type.Integer,
                description = "Maximum lines to return",
                required = false,
            ),
        ),
        execute = { args ->
            withContext(dispatcher) {
                val file = resolve(args["path"].orEmpty())
                    ?: return@withContext ToolResult.Failure("path is outside the workspace")
                when {
                    !file.isFile -> ToolResult.Failure("no such file: ${args["path"]}")
                    else -> {
                        val limit = args["max_lines"]?.toIntOrNull() ?: DEFAULT_LINES
                        val lines = file.useLines { it.take(limit).toList() }
                        // The result goes back into a small context window, so it
                        // is truncated deliberately rather than by accident.
                        ToolResult.Success(lines.joinToString("\n").take(MAX_OUTPUT))
                    }
                }
            }
        },
    )

    private fun writeFile() = Tool(
        name = "write_file",
        description = "Write text to a file in the workspace, replacing it",
        keywords = listOf("save", "create", "store", "put"),
        // Destructive: it replaces whatever was there.
        requiresConfirmation = true,
        parameters = listOf(
            ToolParameter("path", ToolParameter.Type.String, "Path relative to the workspace"),
            ToolParameter("content", ToolParameter.Type.String, "Text to write"),
        ),
        execute = { args ->
            withContext(dispatcher) {
                val file = resolve(args["path"].orEmpty())
                    ?: return@withContext ToolResult.Failure("path is outside the workspace")
                runCatching {
                    file.parentFile?.mkdirs()
                    file.writeText(args["content"].orEmpty())
                    ToolResult.Success("wrote ${file.length()} bytes to ${args["path"]}")
                }.getOrElse { ToolResult.Failure(it.message ?: "write failed") }
            }
        },
    )

    private fun listFiles() = Tool(
        name = "list_files",
        description = "List files in the workspace",
        keywords = listOf("ls", "directory", "folder", "what", "files"),
        parameters = listOf(
            ToolParameter(
                name = "path",
                type = ToolParameter.Type.String,
                description = "Directory relative to the workspace",
                required = false,
            ),
        ),
        execute = { args ->
            withContext(dispatcher) {
                val dir = resolve(args["path"].orEmpty().ifBlank { "." })
                    ?: return@withContext ToolResult.Failure("path is outside the workspace")
                when {
                    !dir.isDirectory -> ToolResult.Failure("not a directory")
                    else -> {
                        val entries = dir.listFiles()?.sortedBy { it.name }.orEmpty()
                        if (entries.isEmpty()) {
                            ToolResult.Success("(empty)")
                        } else {
                            ToolResult.Success(
                                entries.joinToString("\n") { entry ->
                                    if (entry.isDirectory) "${entry.name}/" else "${entry.name} (${entry.length()} bytes)"
                                }.take(MAX_OUTPUT),
                            )
                        }
                    }
                }
            }
        },
    )

    private fun fetchUrl() = Tool(
        name = "fetch_url",
        description = "Fetch the text content of a web page",
        keywords = listOf("http", "web", "download", "get", "page", "site"),
        // Reaches outside the device, so the user sees the address first.
        requiresConfirmation = true,
        parameters = listOf(
            ToolParameter("url", ToolParameter.Type.String, "Full URL including https://"),
        ),
        execute = { args ->
            withContext(dispatcher) {
                val url = args["url"].orEmpty()
                if (!url.startsWith("http://") && !url.startsWith("https://")) {
                    return@withContext ToolResult.Failure("url must start with http:// or https://")
                }
                runCatching {
                    client.newCall(Request.Builder().url(url).build()).execute().use { response ->
                        if (!response.isSuccessful) {
                            return@runCatching ToolResult.Failure("HTTP ${response.code}")
                        }
                        val body = response.body?.string().orEmpty()
                        ToolResult.Success(stripMarkup(body).take(MAX_OUTPUT))
                    }
                }.getOrElse { ToolResult.Failure(it.message ?: "request failed") }
            }
        },
    )

    private fun readClipboard() = Tool(
        name = "read_clipboard",
        description = "Read the current clipboard text",
        keywords = listOf("paste", "copied", "clipboard"),
        execute = {
            val manager = context.getSystemService(ClipboardManager::class.java)
            val text = manager?.primaryClip?.getItemAt(0)?.coerceToText(context)?.toString()
            if (text.isNullOrEmpty()) {
                ToolResult.Failure("clipboard is empty")
            } else {
                ToolResult.Success(text.take(MAX_OUTPUT))
            }
        },
    )

    private fun writeClipboard() = Tool(
        name = "write_clipboard",
        description = "Copy text to the clipboard",
        keywords = listOf("copy", "clipboard"),
        requiresConfirmation = true,
        parameters = listOf(
            ToolParameter("text", ToolParameter.Type.String, "Text to copy"),
        ),
        execute = { args ->
            val manager = context.getSystemService(ClipboardManager::class.java)
            if (manager == null) {
                ToolResult.Failure("clipboard unavailable")
            } else {
                manager.setPrimaryClip(ClipData.newPlainText("Sruti", args["text"].orEmpty()))
                ToolResult.Success("copied to clipboard")
            }
        },
    )

    /** Crude tag stripping. Enough to stop HTML swamping a small context window. */
    private fun stripMarkup(html: String): String = html
        .replace(Regex("(?is)<(script|style)[^>]*>.*?</\\1>"), " ")
        .replace(Regex("<[^>]+>"), " ")
        .replace(Regex("&nbsp;?"), " ")
        .replace(Regex("[ \\t]+"), " ")
        .replace(Regex("\\n\\s*\\n+"), "\n")
        .trim()

    private companion object {
        const val DEFAULT_LINES = 200
        const val MAX_OUTPUT = 4000
    }
}

/**
 * Resolves a model-supplied path inside [workspace], or null when it escapes.
 *
 * Canonical paths are compared rather than the raw strings, so `notes/../../x` is
 * judged on what it resolves to rather than on how it is spelled. Free of Android
 * dependencies so the boundary can actually be tested — this is the check that
 * stops a hallucinated path from reaching the rest of the filesystem.
 */
internal fun resolveInWorkspace(workspace: File, path: String): File? {
    val root = runCatching { workspace.canonicalFile }.getOrNull() ?: return null
    val resolved = runCatching { File(root, path).canonicalFile }.getOrNull() ?: return null
    return if (resolved == root || resolved.path.startsWith(root.path + File.separator)) {
        resolved
    } else {
        null
    }
}
