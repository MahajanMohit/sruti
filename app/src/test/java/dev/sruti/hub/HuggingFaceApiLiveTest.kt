package dev.sruti.hub

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import okhttp3.OkHttpClient
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test

/**
 * Runs the real client against the real Hub.
 *
 * Skipped when there is no network rather than failed, so it never breaks a
 * build on an offline machine. It exists because the client reports every
 * failure as an empty result set, which makes a broken request indistinguishable
 * from a search that genuinely matched nothing.
 */
class HuggingFaceApiLiveTest {

    private val api = HuggingFaceApi(OkHttpClient(), Dispatchers.IO)

    private fun online(): Boolean = runCatching {
        runBlocking { api.modelInfo("Qwen/Qwen3-0.6B") }
        true
    }.getOrDefault(false)

    @Test
    fun `a plain search returns results`() = runBlocking {
        assumeTrue("no network", online())

        val results = api.search("Qwen")
        assertTrue("plain search returned nothing", results.isNotEmpty())
    }

    @Test
    fun `every filter combination still returns results`() = runBlocking {
        assumeTrue("no network", online())

        val combinations = listOf(
            SearchFilters(format = FormatFilter.GGUF),
            SearchFilters(format = FormatFilter.SAFETENSORS),
            SearchFilters(size = SizeBand.UNDER_1B),
            SearchFilters(family = "Qwen"),
            SearchFilters(hideGated = true),
        )
        combinations.forEach { filters ->
            val results = api.search("Qwen", filters)
            assertTrue("no results for $filters", results.isNotEmpty())
        }
    }
}
