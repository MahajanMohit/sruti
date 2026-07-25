package dev.sruti.ui.library

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.ArrowBack
import androidx.compose.material.icons.outlined.Lock
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import dev.sruti.convert.CheckpointInfo
import dev.sruti.convert.QuantType
import dev.sruti.hub.FormatFilter
import dev.sruti.hub.RemoteModelSummary
import dev.sruti.hub.SearchFilters
import dev.sruti.hub.SizeBand
import dev.sruti.ui.formatBytes
import dev.sruti.ui.formatParameters
import dev.sruti.ui.theme.Haptic
import dev.sruti.ui.theme.LocalHaptics
import dev.sruti.ui.theme.Motion

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BrowseScreen(
    state: BrowseUiState,
    defaultQuant: QuantType,
    tokenSet: Boolean,
    busy: Boolean,
    onQueryChanged: (String) -> Unit,
    onFiltersChanged: (SearchFilters) -> Unit,
    onInspect: (String) -> Unit,
    onAcquire: (String) -> Unit,
    onOpenModelPage: (String) -> Unit,
    onBack: () -> Unit,
    onOpenSettings: () -> Unit,
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Add model", style = MaterialTheme.typography.titleMedium) },
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
                .padding(horizontal = 20.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            OutlinedTextField(
                value = state.query,
                onValueChange = onQueryChanged,
                label = { Text("Search Hugging Face") },
                supportingText = {
                    Text(
                        // Personal fine-tunes usually carry no pipeline tag and
                        // never appear in search; their full id always resolves.
                        "Paste an exact owner/model id to find models that search misses",
                        style = MaterialTheme.typography.labelSmall,
                    )
                },
                placeholder = { Text("qwen3, or a full id like owner/model") },
                singleLine = true,
                modifier = Modifier.fillMaxWidth(),
            )

            FilterBar(filters = state.filters, onFiltersChanged = onFiltersChanged)

            AnimatedVisibility(
                visible = !tokenSet,
                enter = fadeIn(Motion.quick()) + expandVertically(Motion.size()),
                exit = fadeOut(Motion.quick()) + shrinkVertically(Motion.size()),
            ) {
                // Worth saying before the user hits a 401 on Llama 3.2, which is
                // both gated and one of the most likely things they will try.
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(14.dp))
                        .background(MaterialTheme.colorScheme.surfaceVariant)
                        .clickable(onClick = onOpenSettings)
                        .padding(14.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    Icon(
                        Icons.Outlined.Lock,
                        contentDescription = null,
                        modifier = Modifier.size(18.dp),
                    )
                    Text(
                        text = "Gated models such as Llama 3.2 need an access token. Tap to add one.",
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

            AnimatedVisibility(visible = state.isSearching) {
                CircularProgressIndicator(modifier = Modifier.size(22.dp))
            }

            LazyColumn(
                contentPadding = PaddingValues(bottom = 32.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                if (state.results.isEmpty() && !state.isSearching && state.error == null) {
                    item {
                        Text(
                            text = if (state.filters.isActive) {
                                "Nothing matched. Filters narrow hard — sizes come from " +
                                    "the model's name, so a model that does not state " +
                                    "its size is filtered out."
                            } else if (state.query.isBlank()) {
                                "Search for a model, or paste its full id."
                            } else {
                                "No models matched that search."
                            },
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }

                items(state.results, key = { it.repoId }) { model ->
                    RepoCard(
                        model = model,
                        info = state.inspected[model.repoId],
                        inspecting = state.inspecting == model.repoId,
                        defaultQuant = defaultQuant,
                        busy = busy,
                        onInspect = { onInspect(model.repoId) },
                        onAcquire = { onAcquire(model.repoId) },
                        onOpenModelPage = { onOpenModelPage(model.repoId) },
                    )
                }
            }
        }
    }
}

/**
 * Filter chips over the search.
 *
 * A single scrolling row rather than a dialog, because every one of these is a
 * one-tap toggle and hiding them behind a sheet would cost more taps than the
 * filtering saves. Tapping a selected chip clears it.
 */
@Composable
private fun FilterBar(
    filters: SearchFilters,
    onFiltersChanged: (SearchFilters) -> Unit,
) {
    val haptics = LocalHaptics.current
    // Wraps every chip: a filter that narrows the list below deserves the same
    // confirmation a physical switch gives, and repeating the call at four sites
    // is how they end up drifting apart.
    val apply: (SearchFilters) -> Unit = {
        haptics.play(Haptic.Select)
        onFiltersChanged(it)
    }
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .horizontalScroll(rememberScrollState()),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        FormatFilter.entries.filter { it != FormatFilter.ANY }.forEach { format ->
            val selected = filters.format == format
            FilterChip(
                selected = selected,
                onClick = {
                    apply(filters.copy(format = if (selected) FormatFilter.ANY else format))
                },
                label = { Text(format.label) },
            )
        }

        SizeBand.entries.forEach { band ->
            val selected = filters.size == band
            FilterChip(
                selected = selected,
                onClick = { apply(filters.copy(size = if (selected) null else band)) },
                label = { Text(band.label) },
            )
        }

        FilterChip(
            selected = filters.hideGated,
            onClick = { apply(filters.copy(hideGated = !filters.hideGated)) },
            label = { Text("Open access") },
        )

        SearchFilters.FAMILIES.forEach { family ->
            val selected = filters.family == family
            FilterChip(
                selected = selected,
                onClick = { apply(filters.copy(family = if (selected) null else family)) },
                label = { Text(family) },
            )
        }
    }
}

@Composable
private fun RepoCard(
    model: RemoteModelSummary,
    info: CheckpointInfo?,
    inspecting: Boolean,
    defaultQuant: QuantType,
    busy: Boolean,
    onInspect: () -> Unit,
    onAcquire: () -> Unit,
    onOpenModelPage: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(18.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant)
            .clickable(enabled = info == null && !inspecting, onClick = onInspect)
            .padding(18.dp)
            .animateContentSize(Motion.size()),
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                text = model.repoId,
                style = MaterialTheme.typography.titleMedium,
                maxLines = 1,
                overflow = TextOverflow.MiddleEllipsis,
                modifier = Modifier.weight(1f),
            )
            if (model.isGated) {
                Icon(
                    Icons.Outlined.Lock,
                    contentDescription = "Gated",
                    modifier = Modifier.size(16.dp),
                    tint = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }

        Text(
            text = buildList {
                add("${model.downloads} downloads")
                add("${model.likes} likes")
                // Worth calling out: it means no conversion step and a much
                // smaller download.
                if (model.hasGguf) add("GGUF ready")
                if (!model.hasSafetensors && !model.hasGguf) add("no safetensors")
            }.joinToString(" · "),
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        when {
            inspecting -> {
                Spacer(Modifier.height(4.dp))
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    CircularProgressIndicator(modifier = Modifier.size(16.dp))
                    Text("Checking compatibility…", style = MaterialTheme.typography.labelSmall)
                }
            }

            info == null -> {
                Text(
                    text = "Tap to check whether this model can be converted",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            !info.supported -> {
                Spacer(Modifier.height(4.dp))
                Text(
                    text = info.error,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.error,
                )
            }

            else -> {
                Spacer(Modifier.height(4.dp))
                Text(
                    text = "${info.displayName} · ${formatParameters(info.parameterCount)} " +
                        "parameters · ${info.contextLength} context",
                    style = MaterialTheme.typography.bodyMedium,
                )
                Text(
                    // Stated up front because it is the number that decides whether
                    // the conversion can finish at all.
                    text = "Download ~${formatBytes(info.estimatedCheckpointBytes)}, " +
                        "converts to ~${formatBytes(info.estimatedOutputBytes(defaultQuant))} " +
                        "at ${defaultQuant.id}",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )

                if (!info.isComfortableSize) {
                    Text(
                        text = "Larger than this device will run comfortably — expect it to " +
                            "be slow and to throttle.",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.error,
                    )
                }

                if (model.isGated) {
                    // A gated repo returns 401 until the licence is accepted on
                    // the website, and there is no way to do that from here. A
                    // link is the only useful thing to offer.
                    Spacer(Modifier.height(4.dp))
                    Text(
                        text = "Gated — accept the licence on huggingface.co first, " +
                            "then add a token in settings.",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    TextButton(onClick = onOpenModelPage) { Text("Open model page") }
                }

                Spacer(Modifier.height(6.dp))
                Button(onClick = onAcquire, enabled = !busy) {
                    Text(if (busy) "Another model is being prepared" else "Download and convert")
                }
            }
        }
    }
}
