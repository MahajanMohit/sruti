package dev.sruti.ui.phase0

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dev.sruti.ui.theme.Motion

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun Phase0Screen(viewModel: Phase0ViewModel = hiltViewModel()) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    val context = LocalContext.current

    val picker = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument(),
    ) { uri ->
        if (uri != null) {
            val name = context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                val index = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                if (index >= 0 && cursor.moveToFirst()) cursor.getString(index) else null
            } ?: "model.gguf"
            viewModel.importModel(uri, name)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(title = { Text("Sruti · Phase 0", style = MaterialTheme.typography.titleMedium) })
        },
    ) { insets ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(insets)
                .padding(horizontal = 20.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            ModelCard(
                label = state.modelLabel,
                threadCount = state.threadCount,
                isImporting = state.isImporting,
                onPick = { picker.launch(arrayOf("*/*")) },
            )

            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Button(
                    onClick = { viewModel.runSmokeTest(SMOKE_PROMPT) },
                    enabled = state.canRun,
                ) { Text("Smoke test") }

                OutlinedButton(
                    onClick = { viewModel.runBenchmark() },
                    enabled = state.canRun,
                ) { Text("Full benchmark") }

                AnimatedVisibility(visible = state.isBusy, enter = fadeIn(), exit = fadeOut()) {
                    OutlinedButton(onClick = viewModel::cancel) { Text("Cancel") }
                }
            }

            AnimatedVisibility(
                visible = state.stage != null,
                enter = fadeIn(Motion.quick()) + expandVertically(Motion.size()),
                exit = fadeOut(Motion.quick()) + shrinkVertically(Motion.size()),
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    CircularProgressIndicator(modifier = Modifier.padding(2.dp))
                    Text(
                        text = state.stage.orEmpty(),
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }
            }

            state.error?.let { message ->
                Text(
                    text = message,
                    color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodyMedium,
                )
            }

            AnimatedVisibility(visible = state.streamedText.isNotEmpty()) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(16.dp))
                        .background(MaterialTheme.colorScheme.surfaceVariant)
                        .padding(16.dp)
                        .animateContentSize(Motion.size()),
                ) {
                    Text(
                        text = state.streamedText,
                        style = MaterialTheme.typography.bodyLarge,
                    )
                }
            }

            LogList(lines = state.log, modifier = Modifier.fillMaxWidth())
        }
    }
}

@Composable
private fun ModelCard(
    label: String?,
    threadCount: Int,
    isImporting: Boolean,
    onPick: () -> Unit,
) {
    val containerColor by animateColorAsState(
        targetValue = if (label == null) {
            MaterialTheme.colorScheme.surfaceVariant
        } else {
            MaterialTheme.colorScheme.primaryContainer
        },
        animationSpec = Motion.standard(),
        label = "modelCardContainer",
    )

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(20.dp))
            .background(containerColor)
            .padding(20.dp)
            .animateContentSize(Motion.size()),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text(
            text = label ?: "No model loaded",
            style = MaterialTheme.typography.titleMedium,
            maxLines = 1,
            overflow = TextOverflow.MiddleEllipsis,
        )
        Text(
            text = "$threadCount inference threads",
            style = MaterialTheme.typography.bodyMedium,
        )
        OutlinedButton(onClick = onPick, enabled = !isImporting) {
            Text(if (label == null) "Choose a GGUF file" else "Replace model")
        }
    }
}

@Composable
private fun LogList(lines: List<String>, modifier: Modifier = Modifier) {
    val listState = rememberLazyListState()

    // Follow the tail as new lines arrive, which is what makes a long benchmark
    // watchable without touching the screen.
    LaunchedEffect(lines.size) {
        if (lines.isNotEmpty()) {
            listState.animateScrollToItem(lines.lastIndex)
        }
    }

    LazyColumn(
        modifier = modifier,
        state = listState,
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        items(lines) { line ->
            Text(
                text = line,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

private const val SMOKE_PROMPT =
    "In two sentences, explain why running a language model on a phone is " +
        "limited more by memory bandwidth than by raw compute."
