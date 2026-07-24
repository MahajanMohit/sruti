package dev.sruti.ui

import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The property that matters for [coalesceToFrames] is losslessness: it may merge
 * emissions freely, but concatenating what comes out must always reproduce the
 * input exactly. A dropped token is a corrupted response.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class StreamCoalescingTest {

    @Test
    fun `preserves all text when emissions are merged`() = runTest {
        val pieces = List(200) { "tok$it " }

        val result = pieces.asFlow().coalesceToFrames(frameMillis = 16).toList()

        assertEquals(pieces.joinToString(""), result.joinToString(""))
    }

    @Test
    fun `merges bursts into fewer emissions than inputs`() = runTest {
        // 500 pieces arriving with no delay should collapse substantially rather
        // than passing through one-for-one.
        val pieces = List(500) { "x" }

        val result = pieces.asFlow().coalesceToFrames(frameMillis = 16).toList()

        assertEquals(500, result.joinToString("").length)
        assertTrue(
            "expected bursts to be merged, got ${result.size} emissions",
            result.size < pieces.size,
        )
    }

    @Test
    fun `emits nothing for an empty upstream`() = runTest {
        val result = flowOf<String>().coalesceToFrames().toList()

        assertEquals(emptyList<String>(), result)
    }

    @Test
    fun `passes through a single emission`() = runTest {
        val result = flowOf("only").coalesceToFrames().toList()

        assertEquals("only", result.joinToString(""))
    }

    @Test
    fun `does not drop text emitted just before upstream completes`() = runTest {
        // The final piece lands in the window between the last drain and upstream
        // finishing — the case a naive implementation loses.
        val result = flow {
            emit("a")
            delay(50)
            emit("b")
            emit("c")
        }.coalesceToFrames(frameMillis = 16).toList()

        assertEquals("abc", result.joinToString(""))
    }

    @Test
    fun `preserves slow trickle where each piece lands in its own frame`() = runTest {
        val result = flow {
            repeat(10) {
                emit("$it,")
                delay(40)
            }
        }.coalesceToFrames(frameMillis = 16).toList()

        assertEquals("0,1,2,3,4,5,6,7,8,9,", result.joinToString(""))
    }

    private fun <T> List<T>.asFlow() = flow { forEach { emit(it) } }
}
