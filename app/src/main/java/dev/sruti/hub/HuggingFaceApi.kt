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

    suspend fun search(query: String, limit: Int = 25): List<RemoteModelSummary> {
        val url = buildString {
            append("$BASE/api/models?limit=$limit&sort=downloads&direction=-1")
            append("&filter=text-generation")
            if (query.isNotBlank()) {
                append("&search=").append(query.urlEncoded())
            }
        }
        return json.decodeFromString(get(url))
    }

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
