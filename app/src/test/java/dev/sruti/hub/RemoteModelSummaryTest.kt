package dev.sruti.hub

import kotlinx.serialization.json.Json
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Decoding against a captured Hub response.
 *
 * The client swallows decode failures and reports "no models matched", which
 * makes a serialization break look exactly like an empty result set. A real
 * payload in a test is the only thing that tells those two apart.
 */
class RemoteModelSummaryTest {

    private val json = Json {
        ignoreUnknownKeys = true
        coerceInputValues = true
    }

    private fun payload(): String =
        checkNotNull(javaClass.classLoader?.getResourceAsStream("hf_search_qwen.json"))
            .bufferedReader()
            .readText()

    @Test
    fun `decodes a real search response`() {
        val models = json.decodeFromString<List<RemoteModelSummary>>(payload())

        assertTrue("response should not decode to an empty list", models.isNotEmpty())
        assertEquals("Qwen/Qwen3-0.6B", models.first().repoId)
        assertTrue(models.first().downloads > 0)
        assertTrue(models.first().hasSafetensors)
    }

    @Test
    fun `default filters keep every row of a real response`() {
        val models = json.decodeFromString<List<RemoteModelSummary>>(payload())
        val kept = models.filter(SearchFilters()::keeps)
        assertEquals(models.size, kept.size)
    }
}
