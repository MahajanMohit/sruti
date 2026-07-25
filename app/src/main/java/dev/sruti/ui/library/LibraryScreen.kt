package dev.sruti.ui.library

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Add
import androidx.compose.material.icons.outlined.Delete
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.material.icons.outlined.Storage
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import dev.sruti.hub.InstalledModel
import dev.sruti.hub.StagedCheckpoint
import dev.sruti.ui.formatBytes
import dev.sruti.ui.formatParameters
import dev.sruti.ui.theme.Haptic
import dev.sruti.ui.theme.LocalHaptics
import dev.sruti.ui.theme.Motion
import dev.sruti.work.ModelJobState

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LibraryScreen(
    state: LibraryUiState,
    onBrowse: () -> Unit,
    onDeleteModel: (InstalledModel) -> Unit,
    onOpenModel: (InstalledModel) -> Unit,
    onDeleteCheckpoint: (StagedCheckpoint) -> Unit,
    onCancelJob: () -> Unit,
    onDismissJob: () -> Unit,
    onOpenSettings: () -> Unit,
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Models", style = MaterialTheme.typography.headlineMedium) },
                actions = {
                    IconButton(onClick = onOpenSettings) {
                        Icon(Icons.Outlined.Settings, contentDescription = "Settings")
                    }
                },
            )
        },
        floatingActionButton = {
            ExtendedFloatingActionButton(
                onClick = onBrowse,
                icon = { Icon(Icons.Outlined.Add, contentDescription = null) },
                text = { Text("Add model") },
            )
        },
    ) { insets ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(insets),
            contentPadding = androidx.compose.foundation.layout.PaddingValues(
                start = 20.dp, end = 20.dp, bottom = 96.dp,
            ),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            item {
                JobCard(job = state.job, onCancel = onCancelJob, onDismiss = onDismissJob)
            }

            item {
                StorageRow(usedBytes = state.usedBytes)
            }

            if (state.installed.isEmpty() && !state.isLoading) {
                item { EmptyState() }
            }

            items(state.installed, key = { it.file.absolutePath }) { model ->
                InstalledModelCard(
                    model = model,
                    onDelete = { onDeleteModel(model) },
                    onOpen = { onOpenModel(model) },
                )
            }

            if (state.staged.isNotEmpty()) {
                item {
                    Text(
                        text = "Unconverted checkpoints",
                        style = MaterialTheme.typography.titleMedium,
                        modifier = Modifier.padding(top = 12.dp),
                    )
                }
                items(state.staged, key = { it.directory.absolutePath }) { checkpoint ->
                    StagedCheckpointCard(
                        checkpoint = checkpoint,
                        onDelete = { onDeleteCheckpoint(checkpoint) },
                    )
                }
            }
        }
    }
}

@Composable
private fun JobCard(job: ModelJobState, onCancel: () -> Unit, onDismiss: () -> Unit) {
    AnimatedVisibility(
        visible = job !is ModelJobState.Idle,
        enter = fadeIn(Motion.standard()) + expandVertically(Motion.size()),
        exit = fadeOut(Motion.standard()) + shrinkVertically(Motion.size()),
    ) {
        val container = when (job) {
            is ModelJobState.Failed -> MaterialTheme.colorScheme.errorContainer
            is ModelJobState.Succeeded -> MaterialTheme.colorScheme.primaryContainer
            else -> MaterialTheme.colorScheme.surfaceVariant
        }

        Column(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(20.dp))
                .background(container)
                .padding(20.dp)
                .animateContentSize(Motion.size()),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            when (job) {
                is ModelJobState.Running -> {
                    Text(job.phase.label, style = MaterialTheme.typography.titleMedium)
                    Text(
                        text = job.repoId,
                        style = MaterialTheme.typography.bodyMedium,
                        maxLines = 1,
                        overflow = TextOverflow.MiddleEllipsis,
                    )

                    // Animating the fraction keeps the bar from stuttering between
                    // the coarse updates a multi-gigabyte download produces.
                    val fraction by animateFloatAsState(
                        targetValue = job.fraction ?: 0f,
                        animationSpec = Motion.standard(),
                        label = "jobProgress",
                    )
                    if (job.fraction != null) {
                        LinearProgressIndicator(
                            progress = { fraction },
                            modifier = Modifier.fillMaxWidth(),
                        )
                    } else {
                        LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                    }

                    if (job.bytesTotal > 0) {
                        Text(
                            text = "${formatBytes(job.bytesDone)} of ${formatBytes(job.bytesTotal)}" +
                                if (job.detail.isNotBlank()) " · ${job.detail}" else "",
                            style = MaterialTheme.typography.labelSmall,
                        )
                    } else if (job.detail.isNotBlank()) {
                        Text(job.detail, style = MaterialTheme.typography.labelSmall)
                    }

                    TextButton(onClick = onCancel) { Text("Cancel") }
                }

                is ModelJobState.Failed -> {
                    Text("Could not prepare model", style = MaterialTheme.typography.titleMedium)
                    Text(job.message, style = MaterialTheme.typography.bodyMedium)
                    TextButton(onClick = onDismiss) { Text("Dismiss") }
                }

                is ModelJobState.Succeeded -> {
                    Text("Model ready", style = MaterialTheme.typography.titleMedium)
                    Text(
                        text = job.outputFile.nameWithoutExtension,
                        style = MaterialTheme.typography.bodyMedium,
                    )
                    job.warnings.forEach { warning ->
                        Text(warning, style = MaterialTheme.typography.labelSmall)
                    }
                    TextButton(onClick = onDismiss) { Text("Dismiss") }
                }

                ModelJobState.Idle -> Unit
            }
        }
    }
}

@Composable
private fun StorageRow(usedBytes: Long) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        modifier = Modifier.padding(vertical = 4.dp),
    ) {
        Icon(
            Icons.Outlined.Storage,
            contentDescription = null,
            modifier = Modifier.size(18.dp),
            tint = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            text = "${formatBytes(usedBytes)} used by models",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun InstalledModelCard(
    model: InstalledModel,
    onDelete: () -> Unit,
    onOpen: () -> Unit,
) {
    val haptics = LocalHaptics.current
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(18.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant)
            .clickable(onClick = onOpen)
            .padding(start = 18.dp, top = 14.dp, bottom = 14.dp, end = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = model.metadata.displayName,
                style = MaterialTheme.typography.titleMedium,
                maxLines = 1,
                overflow = TextOverflow.MiddleEllipsis,
            )
            Spacer(Modifier.height(4.dp))
            Text(
                text = buildList {
                    add(formatBytes(model.sizeBytes))
                    add(model.metadata.quant.id)
                    if (model.metadata.parameterCount > 0) {
                        add(formatParameters(model.metadata.parameterCount))
                    }
                    if (model.metadata.architecture.isNotBlank()) {
                        add(model.metadata.architecture)
                    }
                }.joinToString(" · "),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        IconButton(
            onClick = {
                haptics.play(Haptic.Destructive)
                onDelete()
            },
        ) {
            Icon(Icons.Outlined.Delete, contentDescription = "Delete model")
        }
    }
}

@Composable
private fun StagedCheckpointCard(checkpoint: StagedCheckpoint, onDelete: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(18.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant)
            .padding(start = 18.dp, top = 14.dp, bottom = 14.dp, end = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = checkpoint.repoId,
                style = MaterialTheme.typography.bodyLarge,
                maxLines = 1,
                overflow = TextOverflow.MiddleEllipsis,
            )
            Spacer(Modifier.height(4.dp))
            Text(
                // Checkpoints are several times the size of what they convert to,
                // so it is worth being direct about the cost of keeping them.
                text = "${formatBytes(checkpoint.sizeBytes)} · left over from an " +
                    "interrupted conversion",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        IconButton(onClick = onDelete) {
            Icon(Icons.Outlined.Delete, contentDescription = "Delete checkpoint")
        }
    }
}

@Composable
private fun EmptyState() {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 48.dp),
        contentAlignment = Alignment.Center,
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text("No models yet", style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(8.dp))
            Text(
                text = "Add one from Hugging Face and Sruti will convert it on this device.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}
