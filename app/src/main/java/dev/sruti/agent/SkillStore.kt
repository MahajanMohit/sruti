package dev.sruti.agent

import android.content.Context
import android.net.Uri
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.IOException

/**
 * Skills on disk, plus the ones that ship with the app.
 *
 * Plain files in a directory rather than rows in a database: a skill is a
 * document, and keeping it as one means it can be exported, shared, edited in
 * any text editor and dropped back in — which is most of the point of having
 * user-authored skills at all.
 */
class SkillStore(
    private val context: Context,
    private val client: OkHttpClient,
    private val dispatcher: CoroutineDispatcher,
) {

    val directory: File = File(context.filesDir, "skills").apply { mkdirs() }

    /** Built-ins first, then user skills, each alphabetical. */
    suspend fun all(): List<Skill> = withContext(dispatcher) {
        BUILT_IN.mapNotNull { text ->
            runCatching { SkillParser.parse(text, builtIn = true) }.getOrNull()
        } + userSkills()
    }

    suspend fun userSkills(): List<Skill> = withContext(dispatcher) {
        directory.listFiles { f -> f.isFile && f.extension == "md" }
            ?.sortedBy { it.name }
            // A file that fails to parse is skipped rather than fatal: one bad
            // import must not make every other skill unreachable.
            ?.mapNotNull { file -> runCatching { SkillParser.parse(file.readText()) }.getOrNull() }
            ?: emptyList()
    }

    /** Raw text of a user skill, for editing. */
    suspend fun rawText(name: String): String? = withContext(dispatcher) {
        fileFor(name).takeIf { it.isFile }?.readText()
    }

    /**
     * Validates and writes a skill.
     *
     * Parsed before it is written, so a file on disk is always one that loads.
     * Throws [SkillFormatException] with a message meant to be shown to whoever
     * is editing.
     */
    suspend fun save(text: String): Skill = withContext(dispatcher) {
        val skill = SkillParser.parse(text)
        fileFor(skill.name).writeText(text)
        skill
    }

    suspend fun delete(name: String) = withContext(dispatcher) {
        fileFor(name).delete()
        Unit
    }

    /** Imports from a picked document. */
    suspend fun importFrom(uri: Uri): Skill = withContext(dispatcher) {
        val text = context.contentResolver.openInputStream(uri)?.use {
            it.bufferedReader().readText()
        } ?: throw SkillFormatException("could not read that file")
        save(text)
    }

    /** Imports from a URL — a gist, a repository raw link. */
    suspend fun importFrom(url: String): Skill = withContext(dispatcher) {
        val request = Request.Builder().url(url).build()
        val text = client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw IOException("download failed with HTTP ${response.code}")
            }
            response.body?.string().orEmpty()
        }
        save(text)
    }

    private fun fileFor(name: String): File {
        // A skill name reaches the filesystem, so it is reduced to something that
        // cannot escape the directory or collide with a path separator.
        val safe = name.map { if (it.isLetterOrDigit() || it == '-' || it == '_') it else '-' }
            .joinToString("")
            .trim('-')
            .ifBlank { "skill" }
        return File(directory, "$safe.md")
    }

    private companion object {
        /**
         * Skills that ship with the app.
         *
         * Chosen to be the shapes that actually work at this model size: one or
         * two fixed tool calls with a small number of slots. They double as
         * worked examples of the format, which is why they are written out in
         * full rather than constructed in code.
         */
        val BUILT_IN = listOf(
            """
            ---
            name: summarise-file
            description: Read a file from the workspace and write a summary beside it
            keywords: summarise, summary, digest, condense, shorten
            parameters:
              - path: The file to summarise
            steps:
              - read_file(path={{path}})
              - write_file(path={{path}}.summary.md, content={{step1}})
            ---

            Summarise the file's contents in a few sentences. Keep facts and
            numbers; drop repetition.
            """.trimIndent(),

            """
            ---
            name: save-page
            description: Fetch a web page and save its text into the workspace
            keywords: fetch, download, save, url, page, article
            parameters:
              - url: The address to fetch
              - path: Where to save it
            steps:
              - fetch_url(url={{url}})
              - write_file(path={{path}}, content={{step1}})
            ---
            """.trimIndent(),

            """
            ---
            name: note-clipboard
            description: Save whatever is on the clipboard into a file
            keywords: clipboard, paste, note, capture, save
            parameters:
              - path: Where to save the note
            steps:
              - read_clipboard()
              - write_file(path={{path}}, content={{step1}})
            ---
            """.trimIndent(),

            """
            ---
            name: git-status
            description: Show the git status of a repository using the shell
            keywords: git, status, repository, repo, changes
            parameters:
              - directory: The repository directory
            steps:
              - run_shell(command=cd {{directory}} && git status --short --branch)
            ---

            Report which files changed and which branch is checked out.
            """.trimIndent(),
        )
    }
}
