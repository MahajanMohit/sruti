package dev.sruti.hub

import kotlinx.serialization.json.JsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The size filter reads a parameter count out of the repository name, because
 * search responses carry no size field. These are the naming conventions that
 * actually appear on the Hub.
 */
class SearchFiltersTest {

    private fun summary(repoId: String, gated: String? = null) = RemoteModelSummary(
        repoId = repoId,
        gated = gated?.let { JsonPrimitive(it) },
    )

    @Test
    fun `reads billions from common naming conventions`() {
        assertEquals(0.5, summary("Qwen/Qwen2.5-0.5B-Instruct").billionsFromName!!, 1e-9)
        assertEquals(1.0, summary("meta-llama/Llama-3.2-1B").billionsFromName!!, 1e-9)
        assertEquals(1.5, summary("MohitM2/sruti-1.5b").billionsFromName!!, 1e-9)
        assertEquals(7.0, summary("mistralai/Mistral-7B-v0.3").billionsFromName!!, 1e-9)
    }

    @Test
    fun `reads millions as a fraction of a billion`() {
        assertEquals(0.135, summary("HuggingFaceTB/SmolLM2-135M").billionsFromName!!, 1e-9)
    }

    @Test
    fun `version numbers are not sizes`() {
        // The 2.5 in Qwen2.5 and the 3.2 in Llama-3.2 are followed by a
        // separator, not by a B or an M, so neither is read as a size.
        assertEquals(0.5, summary("Qwen/Qwen2.5-0.5B").billionsFromName!!, 1e-9)
        assertEquals(3.0, summary("meta-llama/Llama-3.2-3B-Instruct").billionsFromName!!, 1e-9)
    }

    @Test
    fun `an unstated size is null rather than a guess`() {
        assertNull(summary("openai-community/gpt2").billionsFromName)
        assertNull(summary("someone/my-finetune").billionsFromName)
    }

    @Test
    fun `size band excludes its lower bound and includes its upper`() {
        assertTrue(SizeBand.UNDER_1B.contains(0.5))
        assertTrue(SizeBand.UNDER_1B.contains(1.0))
        assertFalse(SizeBand.UNDER_1B.contains(1.5))
        assertTrue(SizeBand.ONE_TO_TWO.contains(1.5))
        assertFalse(SizeBand.ONE_TO_TWO.contains(2.5))
        assertTrue(SizeBand.TWO_TO_FOUR.contains(3.0))
    }

    @Test
    fun `a model that does not state its size is filtered out by a size band`() {
        // Deliberate: including it would defeat the point of the filter. The
        // empty-results message says so, so the behaviour is not a mystery.
        val filters = SearchFilters(size = SizeBand.ONE_TO_TWO)
        assertFalse(filters.keeps(summary("someone/my-finetune")))
        assertTrue(filters.keeps(summary("someone/my-finetune-1.5B")))
    }

    @Test
    fun `hide gated respects the several shapes the Hub sends`() {
        val filters = SearchFilters(hideGated = true)
        assertTrue(filters.keeps(summary("a/b")))
        assertTrue(filters.keeps(summary("a/b", gated = "false")))
        assertFalse(filters.keeps(summary("a/b", gated = "auto")))
        assertFalse(filters.keeps(summary("a/b", gated = "manual")))
    }

    @Test
    fun `no filters is not active`() {
        assertFalse(SearchFilters().isActive)
        assertTrue(SearchFilters(family = "Qwen").isActive)
        assertTrue(SearchFilters(format = FormatFilter.GGUF).isActive)
    }
}
