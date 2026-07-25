package dev.sruti.hub

import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.withContext
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.IOException

/** A model as it appears in search results. */
@Serializable
data class RemoteModelSummary(
    @SerialName("id") val repoId: String,
    val downloads: Long = 0,
    val likes: Long = 0,
    /**
     * Raw because the Hub is inconsistent here: this comes back as `false`, as
     * `null`, or as the strings `"auto"` or `"manual"` depending on the endpoint.
     * Decoding it as a Boolean fails on real responses.
     */
    val gated: kotlinx.serialization.json.JsonElement? = null,
    val tags: List<String> = emptyList(),
    @SerialName("lastModified") val lastModified: String? = null,
) {
    val owner: String get() = repoId.substringBefore('/', "")
    val name: String get() = repoId.substringAfter('/')

    /** True unless the field is absent, null, or literally false. */
    val isGated: Boolean
        get() {
            val raw = gated?.toString()?.trim('"') ?: return false
            return raw != "false" && raw != "null"
        }

    /** Whether the repository advertises safetensors weights. */
    val hasSafetensors: Boolean get() = "safetensors" in tags

    /** Whether the repository already publishes a runnable GGUF. */
    val hasGguf: Boolean get() = "gguf" in tags

    /**
     * Parameter count in billions, read out of the repository name.
     *
     * Search results carry no size field — the Hub only reports one on the model
     * page — so the name is the only signal available before downloading
     * anything. Naming it is near-universal (`Qwen2.5-0.5B-Instruct`,
     * `Llama-3.2-1B`, `SmolLM2-135M`), and where it is absent the model is simply
     * unfiltered rather than wrongly excluded.
     */
    val billionsFromName: Double?
        get() {
            val match = SIZE_IN_NAME.find(name) ?: return null
            val value = match.groupValues[1].toDoubleOrNull() ?: return null
            return if (match.groupValues[2].lowercase() == "m") value / 1000.0 else value
        }

    private companion object {
        val SIZE_IN_NAME = Regex("""(?:^|[-_. ])(\d+(?:\.\d+)?)([bBmM])(?:$|[-_. ])""")
    }
}

/** Size bands, chosen around what a phone can actually run. */
enum class SizeBand(val label: String, val range: ClosedFloatingPointRange<Double>) {
    UNDER_1B("Under 1B", 0.0..1.0),
    ONE_TO_TWO("1–2B", 1.0..2.0),
    TWO_TO_FOUR("2–4B", 2.0..4.0),
    ;

    fun contains(billions: Double) = billions > range.start - 1e-9 && billions <= range.endInclusive
}

/** What a repository has to offer, which decides how long it takes to install. */
enum class FormatFilter(val label: String) {
    ANY("Any format"),

    /** Already quantized: downloads smaller and skips conversion entirely. */
    GGUF("Ready to run"),

    /** Raw weights, which this app converts on device. */
    SAFETENSORS("Convert on device"),
}

/**
 * Narrowing applied to a search.
 *
 * Split deliberately between what the Hub can filter and what it cannot. Library
 * tags and the architecture term go to the server, where they narrow the result
 * set before it is paged. Size and gating are applied here, because the Hub
 * exposes neither in a search response — which is also why the request asks for
 * more rows than it shows whenever a local filter is active.
 */
data class SearchFilters(
    val size: SizeBand? = null,
    val format: FormatFilter = FormatFilter.ANY,
    val family: String? = null,
    val hideGated: Boolean = false,
) {
    val isActive: Boolean
        get() = size != null || format != FormatFilter.ANY || family != null || hideGated

    /** True when narrowing happens after the response, so more rows are needed. */
    internal val narrowsLocally: Boolean get() = size != null || hideGated

    internal fun keeps(model: RemoteModelSummary): Boolean {
        if (hideGated && model.isGated) return false
        if (size != null) {
            val billions = model.billionsFromName ?: return false
            if (!size.contains(billions)) return false
        }
        return true
    }

    internal companion object {
        /** Architecture families the converter supports, as search terms. */
        val FAMILIES = listOf("Llama", "Qwen", "Gemma", "Phi", "Mistral", "SmolLM")
    }
}

/** One file in a repository. */
@Serializable
data class RemoteFile(
    val path: String,
    val size: Long = 0,
    val type: String = "file",
    val lfs: LfsInfo? = null,
) {
    /**
     * SHA-256 of the content, when the file is stored in LFS.
     *
     * Weight shards always are, which is exactly where verification matters — a
     * silently truncated multi-gigabyte download otherwise surfaces as a
     * confusing conversion error much later.
     */
    val sha256: String? get() = lfs?.oid?.removePrefix("sha256:")

    /** LFS records the true size; the outer field is the pointer's size. */
    val actualSize: Long get() = lfs?.size ?: size
}

@Serializable
data class LfsInfo(
    val oid: String = "",
    val size: Long = 0,
)

/** Everything needed to fetch a checkpoint. */
data class RemoteCheckpoint(
    val repoId: String,
    val revision: String,
    val files: List<RemoteFile>,
) {
    /**
     * Ready-made GGUF files in the repository, largest last.
     *
     * Worth checking before anything else: many repositories publish quantized
     * GGUFs alongside the safetensors, and downloading one is both several times
     * smaller and skips conversion entirely.
     */
    val prebuiltGguf: List<RemoteFile>
        get() = files
            .filter { it.path.endsWith(".gguf", ignoreCase = true) && it.actualSize > 0 }
            .sortedBy { it.actualSize }

    /**
     * The published GGUF that best matches [quantId], or the smallest one.
     *
     * A repository often publishes several quantizations of the same model, and
     * taking whichever sorted first would quietly ignore the setting the user
     * chose. Falling back to the smallest is the right default when none matches:
     * it is the one most likely to run on a phone.
     */
    fun pickGguf(quantId: String): RemoteFile? {
        val candidates = prebuiltGguf
        return candidates.firstOrNull {
            it.path.substringAfterLast('/').contains(quantId, ignoreCase = true)
        } ?: candidates.firstOrNull()
    }

    /** Files a conversion actually needs; skips READMEs, images and ONNX exports. */
    val requiredFiles: List<RemoteFile>
        get() = files.filter { file ->
            val name = file.path.substringAfterLast('/')
            // Nested directories hold alternative exports (onnx/, gguf/) that would
            // multiply the download for no benefit.
            if (file.path.contains('/')) return@filter false
            name.endsWith(".safetensors") ||
                name in setOf(
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "generation_config.json",
                    "model.safetensors.index.json",
                    "special_tokens_map.json",
                    // Newer exports keep the chat template here rather than
                    // inside tokenizer_config.json. Skipping it converts fine and
                    // then prompts an instruct model as a base model.
                    "chat_template.jinja",
                )
        }

    val totalBytes: Long get() = requiredFiles.sumOf { it.actualSize }

    val hasSafetensors: Boolean
        get() = requiredFiles.any { it.path.endsWith(".safetensors") }
}

/** A request failed in a way the user can act on. */
class HubException(message: String, val code: Int = 0) : IOException(message)

/**
 * Minimal Hugging Face Hub client.
 *
 * Only the three things the model browser needs: search, list a repository's
 * files, and fetch a small text file. Gated repositories — Llama 3.x among them —
 * return 401 or 403 without a token, so [token] is threaded through everything and
 * the failure is reported in terms the user can act on.
 */
class HuggingFaceApi(
    private val client: OkHttpClient,
    private val dispatcher: CoroutineDispatcher,
    private val tokenProvider: suspend () -> String? = { null },
) {

    // coerceInputValues matters as much as ignoreUnknownKeys here: the Hub sends
    // explicit nulls for fields it declares as numbers, which would otherwise
    // throw rather than fall back to the declared default.
    private val json = Json {
        ignoreUnknownKeys = true
        coerceInputValues = true
    }

    /**
     * Searches the Hub, and resolves an exact `owner/name` directly.
     *
     * The direct lookup matters: search only returns models tagged with the
     * text-generation pipeline, and a freshly uploaded or personal fine-tune
     * usually carries no pipeline tag at all — so it is invisible to search while
     * being perfectly convertible. Typing its full id finds it.
     */
    suspend fun search(
        query: String,
        filters: SearchFilters = SearchFilters(),
        limit: Int = 25,
    ): List<RemoteModelSummary> {
        val trimmed = query.trim().removePrefix("https://huggingface.co/").trim('/')

        // An exact repo id is a lookup, not a search. Filters do not apply — the
        // user named one specific model, and hiding it would only be confusing.
        if (trimmed.count { it == '/' } == 1 && !trimmed.contains(' ')) {
            runCatching { modelInfo(trimmed) }
                .onSuccess { return listOf(it) }
            // Fall through: it may be a partial phrase that happens to have a
            // slash, and a failed lookup should not swallow the query.
        }

        // Post-filtering discards rows, so ask for enough that a full page can
        // still be shown afterwards.
        val fetch = if (filters.narrowsLocally) limit * 4 else limit
        val terms = listOfNotNull(trimmed.takeIf { it.isNotBlank() }, filters.family)
            .joinToString(" ")

        val tagged = rawSearch(terms, filters, fetch, pipelineFiltered = true)
        val kept = tagged.filter(filters::keeps)
        if (kept.isNotEmpty()) return kept.take(limit)

        // Nothing tagged matched. Retry unfiltered rather than reporting no
        // results for a model that exists but is not labelled.
        return rawSearch(terms, filters, fetch, pipelineFiltered = false)
            .filter(filters::keeps)
            .take(limit)
    }

    private suspend fun rawSearch(
        query: String,
        filters: SearchFilters,
        limit: Int,
        pipelineFiltered: Boolean,
    ): List<RemoteModelSummary> {
        val url = buildString {
            append("$BASE/api/models?limit=$limit&sort=downloads&direction=-1")
            if (pipelineFiltered) append("&filter=text-generation")
            when (filters.format) {
                FormatFilter.GGUF -> append("&filter=gguf")
                FormatFilter.SAFETENSORS -> append("&filter=safetensors")
                FormatFilter.ANY -> Unit
            }
            if (query.isNotBlank()) {
                append("&search=").append(query.urlEncoded())
            }
        }
        return runCatching { json.decodeFromString<List<RemoteModelSummary>>(get(url)) }
            .getOrDefault(emptyList())
    }

    /** Web page for a repository, for accepting a licence. */
    fun modelPageUrl(repoId: String): String = "$BASE/${repoId.pathEncoded()}"

    suspend fun modelInfo(repoId: String): RemoteModelSummary =
        json.decodeFromString(get("$BASE/api/models/${repoId.pathEncoded()}"))

    /**
     * Lists a repository's files with real sizes.
     *
     * The tree endpoint is used rather than the model endpoint's `siblings`,
     * because siblings carries filenames only — and knowing the download size
     * before starting is the whole point of the preflight.
     */
    suspend fun listFiles(repoId: String, revision: String = "main"): RemoteCheckpoint {
        val url = "$BASE/api/models/${repoId.pathEncoded()}/tree/$revision?recursive=true"
        val files: List<RemoteFile> = json.decodeFromString(get(url))
        return RemoteCheckpoint(repoId, revision, files.filter { it.type == "file" })
    }

    /** Fetches a small text file, e.g. config.json. */
    suspend fun fetchText(repoId: String, path: String, revision: String = "main"): String =
        get(downloadUrl(repoId, path, revision))

    fun downloadUrl(repoId: String, path: String, revision: String = "main"): String =
        "$BASE/${repoId.pathEncoded()}/resolve/$revision/$path"

    /** Authorization header for the downloader, or null when no token is set. */
    suspend fun authHeader(): Pair<String, String>? =
        tokenProvider()?.takeIf { it.isNotBlank() }?.let { "Authorization" to "Bearer $it" }

    private suspend fun get(url: String): String = withContext(dispatcher) {
        val builder = Request.Builder().url(url)
        authHeader()?.let { (name, value) -> builder.header(name, value) }

        client.newCall(builder.build()).execute().use { response ->
            if (!response.isSuccessful) {
                throw HubException(describeFailure(response.code, url), response.code)
            }
            response.body?.string() ?: throw HubException("empty response from $url")
        }
    }

    private suspend fun describeFailure(code: Int, url: String): String {
        val hasToken = tokenProvider()?.isNotBlank() == true
        return when {
            code == 401 || code == 403 -> if (hasToken) {
                "Access denied. This model is gated and the saved token does not " +
                    "grant access — accept its licence on huggingface.co first."
            } else {
                "This model is gated. Add a Hugging Face access token in settings, " +
                    "and accept the model's licence on huggingface.co."
            }
            code == 404 -> "Not found: $url"
            code == 429 -> "Rate limited by Hugging Face. Adding an access token raises the limit."
            else -> "Request failed with HTTP $code: $url"
        }
    }

    private companion object {
        const val BASE = "https://huggingface.co"
    }
}

// Repository ids contain a slash that must survive as a path separator, so only
// the segments are encoded.
private fun String.pathEncoded(): String = split('/').joinToString("/") { it.urlEncoded() }

private fun String.urlEncoded(): String =
    java.net.URLEncoder.encode(this, Charsets.UTF_8.name()).replace("+", "%20")
