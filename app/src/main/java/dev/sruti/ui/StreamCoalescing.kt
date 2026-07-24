package dev.sruti.ui

import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/**
 * Coalesces a high-frequency string stream into at most one emission per frame.
 *
 * A model decoding at 30 tok/s writes Compose state 30 times a second, and each
 * write invalidates the text node, re-measures, and re-lays-out. That alone is
 * enough to miss frames, and it worsens as the message grows because the whole
 * paragraph is re-measured on every update.
 *
 * Batching to the frame cadence makes UI cost independent of token rate: the text
 * updates ~60 times a second whether the model produces 10 tokens per second or
 * 200. Nothing is dropped — emissions are concatenated, not discarded. Each
 * emission is the text accumulated since the previous one, so consumers append.
 */
fun Flow<String>.coalesceToFrames(frameMillis: Long = 16L): Flow<String> = channelFlow {
    val pending = StringBuilder()
    var upstreamDone = false

    val producer = launch {
        try {
            collect { piece ->
                synchronized(pending) { pending.append(piece) }
            }
        } finally {
            upstreamDone = true
        }
    }

    fun drain(): String? = synchronized(pending) {
        if (pending.isEmpty()) {
            null
        } else {
            pending.toString().also { pending.setLength(0) }
        }
    }

    while (isActive) {
        val chunk = drain()
        if (chunk != null) {
            send(chunk)
        } else if (upstreamDone) {
            break
        }
        delay(frameMillis)
    }

    producer.join()

    // Anything appended between the last drain and upstream completing.
    drain()?.let { send(it) }
}
