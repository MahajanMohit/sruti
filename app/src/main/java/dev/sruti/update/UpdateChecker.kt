package dev.sruti.update

import dev.sruti.BuildConfig
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.withContext
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.IOException

/** What the check found. */
sealed interface UpdateStatus {
    data object UpToDate : UpdateStatus

    data class Available(
        val version: String,
        val notes: String,
        val pageUrl: String,
        val downloadUrl: String?,
        val sizeBytes: Long,
    ) : UpdateStatus

    /** Checking failed; the reason is worth showing rather than swallowing. */
    data class Failed(val message: String) : UpdateStatus
}

@Serializable
private data class GithubRelease(
    @SerialName("tag_name") val tagName: String = "",
    val name: String = "",
    val body: String = "",
    val draft: Boolean = false,
    val prerelease: Boolean = false,
    @SerialName("html_url") val htmlUrl: String = "",
    val assets: List<GithubAsset> = emptyList(),
)

@Serializable
private data class GithubAsset(
    val name: String = "",
    val size: Long = 0,
    @SerialName("browser_download_url") val downloadUrl: String = "",
)

/**
 * Asks GitHub whether a newer build exists.
 *
 * This app is sideloaded, so nothing tells the user an update shipped — there is
 * no store to notice it. Checking is therefore a real feature rather than a
 * convenience, and it is manual: a background poller would be a network call the
 * user did not ask for, in an app whose whole premise is that it does not talk
 * to servers unless told to.
 *
 * It reports rather than installs. Downloading and launching an APK install is a
 * meaningfully different level of trust, and a link to the release page keeps
 * the decision where it belongs.
 */
class UpdateChecker(
    private val client: OkHttpClient,
    private val dispatcher: CoroutineDispatcher,
    private val currentVersion: String = BuildConfig.VERSION_NAME,
    private val repo: String = BuildConfig.RELEASES_REPO,
) {

    private val json = Json { ignoreUnknownKeys = true; coerceInputValues = true }

    suspend fun check(): UpdateStatus = withContext(dispatcher) {
        runCatching { fetchLatest() }
            .fold(
                onSuccess = { release -> compare(release) },
                onFailure = { t -> UpdateStatus.Failed(describe(t)) },
            )
    }

    private fun fetchLatest(): GithubRelease {
        val request = Request.Builder()
            .url("https://api.github.com/repos/$repo/releases/latest")
            .header("Accept", "application/vnd.github+json")
            .build()

        client.newCall(request).execute().use { response ->
            if (response.code == 404) {
                throw IOException("no releases have been published yet")
            }
            if (!response.isSuccessful) {
                throw IOException("GitHub returned HTTP ${response.code}")
            }
            val body = response.body?.string().orEmpty()
            return json.decodeFromString(body)
        }
    }

    private fun compare(release: GithubRelease): UpdateStatus {
        if (release.draft || release.prerelease) return UpdateStatus.UpToDate

        val latest = release.tagName.removePrefix("v")
        if (latest.isBlank()) return UpdateStatus.UpToDate
        if (compareVersions(latest, currentVersion) <= 0) return UpdateStatus.UpToDate

        // The arm64 APK is the only artifact a phone can use; ignore checksums
        // and source archives attached alongside it.
        val apk = release.assets.firstOrNull { it.name.endsWith(".apk", ignoreCase = true) }

        return UpdateStatus.Available(
            version = latest,
            notes = release.body.trim().take(NOTES_LIMIT),
            pageUrl = release.htmlUrl.ifBlank { "https://github.com/$repo/releases/latest" },
            downloadUrl = apk?.downloadUrl,
            sizeBytes = apk?.size ?: 0,
        )
    }

    private fun describe(t: Throwable): String = when {
        t is IOException && t.message != null -> t.message!!
        else -> t.message ?: "could not reach GitHub"
    }

    private companion object {
        const val NOTES_LIMIT = 2000
    }
}

/**
 * Compares two dotted version strings.
 *
 * Numeric per component, not lexicographic, because "0.10.0" is newer than
 * "0.9.0" and a string comparison says the opposite — which would leave a user
 * permanently told they are up to date.
 *
 * A pre-release suffix (`0.2.0-rc1`) sorts before the release it precedes.
 */
internal fun compareVersions(a: String, b: String): Int {
    fun parts(v: String): List<Int> =
        v.substringBefore('-').split('.').map { it.trim().toIntOrNull() ?: 0 }

    val left = parts(a)
    val right = parts(b)

    repeat(maxOf(left.size, right.size)) { i ->
        val l = left.getOrElse(i) { 0 }
        val r = right.getOrElse(i) { 0 }
        if (l != r) return l.compareTo(r)
    }

    // Same numbers: a suffixed version is a pre-release of the plain one.
    val leftPre = a.contains('-')
    val rightPre = b.contains('-')
    return when {
        leftPre == rightPre -> 0
        leftPre -> -1
        else -> 1
    }
}
