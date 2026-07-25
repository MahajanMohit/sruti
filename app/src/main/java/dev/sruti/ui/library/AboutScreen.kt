package dev.sruti.ui.library

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.ArrowBack
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

/**
 * What the app actually does, in enough detail to be useful.
 *
 * Written because "converts models on device" is not a claim a user can evaluate.
 * Which architectures work and why, what conversion involves, and where it can go
 * wrong are the things that make the behaviour predictable rather than magical.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AboutScreen(
    backendCount: Int,
    backendDetail: String,
    threadCount: Int,
    onBack: () -> Unit,
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("About Sruti", style = MaterialTheme.typography.titleMedium) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Outlined.ArrowBack, contentDescription = "Back")
                    }
                },
            )
        },
    ) { insets ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(insets)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 20.dp)
                .padding(bottom = 40.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Section("What this does")
            Body(
                "Sruti runs language models entirely on this device. Nothing you type " +
                    "and nothing the model produces is sent anywhere — there is no server " +
                    "involved in inference at all.",
            )
            Body(
                "It also converts models here. Point it at a Hugging Face repository and " +
                    "it downloads the raw weights and turns them into a runnable file on " +
                    "the phone, with no desktop step anywhere in the process.",
            )

            Divider()
            Section("Supported architectures")
            Body(
                "Llama and Mistral, Qwen2 and Qwen3, Gemma 2 and 3, and Phi-3.",
            )
            Body(
                "The list is short on purpose. Each architecture lays its weights out " +
                    "differently, and a mapping that is close but wrong produces a model " +
                    "that loads happily and generates fluent nonsense. Supporting one " +
                    "means having verified it rather than assuming it.",
            )
            Body(
                "Tokenizers must be byte-level BPE, which means a tokenizer.json in the " +
                    "repository. SentencePiece checkpoints are refused rather than " +
                    "converted incorrectly.",
            )

            Divider()
            Section("What conversion actually does")
            Body(
                "Raw weights ship as safetensors, which is a container: it stores named " +
                    "tensors and nothing about how to run them. Making one runnable means " +
                    "rewriting it into GGUF, and that is not a copy.",
            )
            Body(
                "Llama-family checkpoints interleave the query and key projections in a " +
                    "layout GGUF does not use, so those are rearranged. Gemma stores its " +
                    "normalisation weights offset by one and they are shifted back. " +
                    "Dimensions are reversed, because the two formats disagree on order. " +
                    "The vocabulary is rebuilt, including which tokens are control tokens " +
                    "— several models mark those wrongly, and left alone they show up as " +
                    "visible text in replies.",
            )
            Body(
                "Then it is quantized, which trades a little quality for roughly a " +
                    "quarter of the size. Q4_K_M is the default because it is the best of " +
                    "that trade for a phone.",
            )
            Body(
                "The intermediate file is about two bytes per parameter and exists " +
                    "alongside the finished one, so conversion needs noticeably more free " +
                    "space than the result. It is checked before starting.",
            )
            Body(
                "Where a repository already publishes a GGUF, that is downloaded instead " +
                    "and conversion is skipped entirely — it is smaller and there is " +
                    "nothing to gain by rebuilding it.",
            )

            Divider()
            Section("How it runs")
            Body(
                "Every processor feature set gets its own compute backend, and the best " +
                    "one this device can actually execute is chosen when the app starts. " +
                    "That is why one build runs on a budget phone without crashing and " +
                    "still uses the faster instructions on a flagship.",
            )
            Body("Backends available on this device: $backendCount")
            Body("Inference threads: $threadCount, matched to the performance cores")
            Body(
                "Generation is deliberately slowed when the device gets warm. Sustained " +
                    "decoding draws several watts, and easing off early keeps the pace " +
                    "even instead of letting the chip cut speed abruptly mid-reply.",
            )

            Divider()
            Section("Agent mode")
            Body(
                "Turned on from the chat screen, a message becomes a task rather " +
                    "than a question. The model can read and write files in its own " +
                    "workspace, fetch a URL, use the clipboard, and — if you enable " +
                    "it — run shell commands through Termux.",
            )
            Body(
                "The model is never asked to plan. It is asked one narrow question at " +
                    "a time: which of these few tools, then fill these slots. The loop " +
                    "between those questions is ordinary code with a hard step limit. " +
                    "A model this size cannot hold a multi-step plan, but it can answer " +
                    "a narrow question, and the harness is what remembers.",
            )
            Body(
                "Answers are constrained by a grammar while they are being generated, " +
                    "so a malformed tool call is impossible rather than merely " +
                    "unlikely. That does not make a call correct — measured on a 0.5B " +
                    "model, argument extraction was reliable and tool selection was " +
                    "not — which is why candidates are narrowed before the model sees " +
                    "them, arguments are checked against the schema afterwards, and " +
                    "every step is shown as it happens.",
            )
            Body(
                "Anything that changes something asks first, and shows the exact " +
                    "arguments it would use. Shell access is off until you turn it on.",
            )
            Body(
                "Skills go further. A skill writes the steps down — which tools, in " +
                    "what order, and where each result goes — so the model is asked only " +
                    "to read the values out of your request, which is the part it does " +
                    "well. Four ship with the app, and you can write or import your own.",
            )

            Divider()
            Section("Limits worth knowing")
            Body(
                "Models above roughly 4 billion parameters will run but not pleasantly. " +
                    "Decoding is limited by memory bandwidth rather than raw speed, and a " +
                    "phone shares that bandwidth with everything else.",
            )
            Body(
                "The agent is built around a small model's actual abilities. It is asked " +
                    "one narrow question at a time and the sequencing is ordinary code, " +
                    "because a model this size cannot hold a multi-step plan. Tool calls " +
                    "are constrained so a malformed one is impossible rather than merely " +
                    "unlikely — but a well-formed call can still be the wrong one.",
            )
            Body("Inference is CPU-only in this build. GPU offload is not enabled yet.")

            Divider()
            Section("Diagnostics")
            Mono(backendDetail)
        }
    }
}

@Composable
private fun Section(text: String) {
    Text(
        text = text,
        style = MaterialTheme.typography.titleMedium,
        modifier = Modifier.padding(top = 8.dp),
    )
}

@Composable
private fun Body(text: String) {
    Text(
        text = text,
        style = MaterialTheme.typography.bodyMedium,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
    )
}

@Composable
private fun Mono(text: String) {
    Text(
        text = text,
        style = MaterialTheme.typography.labelSmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
    )
}

@Composable
private fun Divider() {
    HorizontalDivider(modifier = Modifier.padding(vertical = 6.dp))
}
