package dev.sruti.ui.chat

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.Send
import androidx.compose.material.icons.outlined.Add
import androidx.compose.material.icons.outlined.Delete
import androidx.compose.material.icons.outlined.History
import androidx.compose.material.icons.outlined.Stop
import androidx.compose.material.icons.outlined.Tune
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import dev.sruti.llm.ChatMessage
import dev.sruti.ui.theme.Motion
import java.text.DateFormat
import java.util.Date
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(
    state: ChatUiState,
    onSend: (String) -> Unit,
    onStop: () -> Unit,
    onNewConversation: () -> Unit,
    onOpenConversation: (Long) -> Unit,
    onDeleteConversation: (Long) -> Unit,
    onSelectModel: (dev.sruti.hub.InstalledModel) -> Unit,
    onOpenModels: () -> Unit,
) {
    var draft by remember { mutableStateOf("") }
    var modelMenuOpen by remember { mutableStateOf(false) }
    var historyOpen by remember { mutableStateOf(false) }

    if (historyOpen) {
        HistorySheet(
            conversations = state.conversations,
            currentId = state.conversationId,
            onOpen = { id ->
                historyOpen = false
                onOpenConversation(id)
            },
            onDelete = onDeleteConversation,
            onDismiss = { historyOpen = false },
        )
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text(
                            text = state.selectedModel?.metadata?.displayName ?: "No model",
                            style = MaterialTheme.typography.titleMedium,
                            maxLines = 1,
                            overflow = TextOverflow.MiddleEllipsis,
                        )
                        ContextLine(state)
                    }
                },
                actions = {
                    // Sits before "new" on purpose: it is what makes starting a
                    // new conversation safe, by showing where the old one went.
                    IconButton(
                        onClick = { historyOpen = true },
                        enabled = state.conversations.isNotEmpty(),
                    ) {
                        Icon(
                            Icons.Outlined.History,
                            contentDescription = "Past conversations",
                        )
                    }
                    IconButton(onClick = onNewConversation, enabled = state.modelReady) {
                        Icon(Icons.Outlined.Add, contentDescription = "New conversation")
                    }
                    Box {
                        IconButton(onClick = { modelMenuOpen = true }) {
                            Icon(Icons.Outlined.Tune, contentDescription = "Choose model")
                        }
                        DropdownMenu(
                            expanded = modelMenuOpen,
                            onDismissRequest = { modelMenuOpen = false },
                        ) {
                            state.models.forEach { model ->
                                DropdownMenuItem(
                                    text = { Text(model.metadata.displayName) },
                                    onClick = {
                                        modelMenuOpen = false
                                        onSelectModel(model)
                                    },
                                )
                            }
                            if (state.models.isEmpty()) {
                                DropdownMenuItem(
                                    text = { Text("Add a model…") },
                                    onClick = {
                                        modelMenuOpen = false
                                        onOpenModels()
                                    },
                                )
                            }
                        }
                    }
                },
            )
        },
    ) { insets ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(insets)
                .imePadding(),
        ) {
            ContextMeter(state)

            AnimatedVisibility(
                visible = state.thermalNotice.isNotEmpty(),
                enter = fadeIn(Motion.quick()) + expandVertically(Motion.size()),
                exit = fadeOut(Motion.quick()) + shrinkVertically(Motion.size()),
            ) {
                // Naming the slowdown is the whole point: an unexplained crawl
                // reads as a broken app, a labelled one reads as a hot phone.
                Text(
                    text = state.thermalNotice,
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(horizontal = 20.dp, vertical = 4.dp),
                )
            }

            state.error?.let { message ->
                // Selectable and bounded: a load failure carries diagnostic detail
                // that is only useful if it can be read and copied, and an
                // unbounded error block would push the composer off screen.
                SelectionContainer {
                    Text(
                        text = message,
                        color = MaterialTheme.colorScheme.error,
                        style = MaterialTheme.typography.labelSmall,
                        modifier = Modifier
                            .heightIn(max = 220.dp)
                            .verticalScroll(rememberScrollState())
                            .padding(horizontal = 20.dp, vertical = 6.dp),
                    )
                }
            }

            Transcript(state = state, modifier = Modifier.weight(1f))

            Composer(
                draft = draft,
                onDraftChange = { draft = it },
                enabled = state.canSend,
                isGenerating = state.isGenerating,
                onSend = {
                    onSend(draft)
                    draft = ""
                },
                onStop = onStop,
            )
        }
    }
}

@Composable
private fun ContextLine(state: ChatUiState) {
    val text = when {
        state.isLoadingModel -> "Loading model…"
        !state.modelReady -> "Choose a model to begin"
        else -> buildString {
            append("${state.contextUsed} / ${state.contextTotal} tokens")
            state.lastMetrics?.let { m ->
                if (m.decodeTokensPerSecond > 0) {
                    append(" · ")
                    append(String.format(Locale.US, "%.1f tok/s", m.decodeTokensPerSecond))
                }
                // Worth surfacing: it is the difference between a snappy follow-up
                // and one that reprocesses the whole conversation.
                if (m.reusedTokens > 0) {
                    append(" · ")
                    append("${(m.cacheHitRate * 100).toInt()}% cached")
                }
            }
        }
    }
    Text(
        text = text,
        style = MaterialTheme.typography.labelSmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
    )
}

@Composable
private fun ContextMeter(state: ChatUiState) {
    if (!state.modelReady) return

    val fraction by animateFloatAsState(
        targetValue = state.contextFraction,
        animationSpec = Motion.standard(),
        label = "contextFill",
    )
    LinearProgressIndicator(
        progress = { fraction },
        modifier = Modifier
            .fillMaxWidth()
            .height(2.dp),
        // Turns a warning colour as the window fills, which is when the oldest
        // turns are about to be evicted.
        color = if (state.contextFraction > 0.85f) {
            MaterialTheme.colorScheme.error
        } else {
            MaterialTheme.colorScheme.primary
        },
    )
}

@Composable
private fun Transcript(state: ChatUiState, modifier: Modifier = Modifier) {
    val listState = rememberLazyListState()

    // Follow the tail as the reply grows. Keyed on length rather than a token
    // count so it also tracks the streaming message.
    LaunchedEffect(state.messages.size, state.streamingText.length) {
        val lastIndex = state.messages.size + if (state.streamingText.isNotEmpty()) 1 else 0
        if (lastIndex > 0) {
            listState.animateScrollToItem(lastIndex - 1)
        }
    }

    LazyColumn(
        modifier = modifier.fillMaxWidth(),
        state = listState,
        contentPadding = androidx.compose.foundation.layout.PaddingValues(
            start = 16.dp, end = 16.dp, top = 12.dp, bottom = 12.dp,
        ),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        items(state.messages, key = { it.id }) { message ->
            MessageBubble(
                content = message.content,
                isUser = message.role == ChatMessage.Role.User,
                footnote = if (message.tokensPerSecond > 0) {
                    String.format(
                        Locale.US,
                        "%d tokens · %.1f tok/s",
                        message.tokenCount,
                        message.tokensPerSecond,
                    )
                } else {
                    null
                },
            )
        }

        if (state.streamingText.isNotEmpty()) {
            item(key = "streaming") {
                MessageBubble(content = state.streamingText, isUser = false, footnote = null)
            }
        }

        if (state.isGenerating && state.streamingText.isEmpty()) {
            item(key = "thinking") {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                    modifier = Modifier.padding(vertical = 8.dp),
                ) {
                    CircularProgressIndicator(modifier = Modifier.size(14.dp))
                    Text(
                        // Prefill is the slow part of a long conversation, so say
                        // so instead of showing an unexplained pause.
                        text = "Reading the conversation…",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }
}

@Composable
private fun MessageBubble(content: String, isUser: Boolean, footnote: String?) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start,
    ) {
        Column(
            modifier = Modifier
                .widthIn(max = 320.dp)
                .clip(
                    RoundedCornerShape(
                        topStart = 18.dp,
                        topEnd = 18.dp,
                        bottomStart = if (isUser) 18.dp else 4.dp,
                        bottomEnd = if (isUser) 4.dp else 18.dp,
                    ),
                )
                .background(
                    if (isUser) {
                        MaterialTheme.colorScheme.primaryContainer
                    } else {
                        MaterialTheme.colorScheme.surfaceVariant
                    },
                )
                .padding(horizontal = 14.dp, vertical = 10.dp)
                .animateContentSize(Motion.size()),
        ) {
            Text(text = content, style = MaterialTheme.typography.bodyLarge)
            footnote?.let {
                Spacer(Modifier.height(4.dp))
                Text(
                    text = it,
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
private fun Composer(
    draft: String,
    onDraftChange: (String) -> Unit,
    enabled: Boolean,
    isGenerating: Boolean,
    onSend: () -> Unit,
    onStop: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 10.dp),
        verticalAlignment = Alignment.Bottom,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        OutlinedTextField(
            value = draft,
            onValueChange = onDraftChange,
            placeholder = { Text("Message") },
            modifier = Modifier.weight(1f),
            maxLines = 5,
        )

        if (isGenerating) {
            FilledIconButton(onClick = onStop) {
                Icon(Icons.Outlined.Stop, contentDescription = "Stop")
            }
        } else {
            FilledIconButton(onClick = onSend, enabled = enabled && draft.isNotBlank()) {
                Icon(Icons.AutoMirrored.Outlined.Send, contentDescription = "Send")
            }
        }
    }
}

/** Shown when no models are installed at all. */
@Composable
fun ChatEmptyState(onOpenModels: () -> Unit) {
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text("No models installed", style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(8.dp))
            TextButton(onClick = onOpenModels) { Text("Add a model") }
        }
    }
}

/**
 * Past conversations, newest first.
 *
 * A bottom sheet rather than a screen: the list is the only thing on it, and
 * reaching it should not cost a navigation the user then has to back out of.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun HistorySheet(
    conversations: List<ConversationSummary>,
    currentId: Long,
    onOpen: (Long) -> Unit,
    onDelete: (Long) -> Unit,
    onDismiss: () -> Unit,
) {
    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true),
    ) {
        Text(
            text = "Conversations",
            style = MaterialTheme.typography.titleMedium,
            modifier = Modifier.padding(start = 24.dp, end = 24.dp, bottom = 8.dp),
        )
        HorizontalDivider()

        LazyColumn(
            modifier = Modifier.fillMaxWidth(),
            contentPadding = androidx.compose.foundation.layout.PaddingValues(bottom = 32.dp),
        ) {
            items(conversations, key = { it.id }) { conversation ->
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { onOpen(conversation.id) }
                        .padding(start = 24.dp, end = 8.dp, top = 12.dp, bottom = 12.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = conversation.title,
                            style = MaterialTheme.typography.bodyLarge,
                            color = if (conversation.id == currentId) {
                                MaterialTheme.colorScheme.primary
                            } else {
                                MaterialTheme.colorScheme.onSurface
                            },
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                        )
                        Text(
                            text = relativeTime(conversation.updatedAtMillis),
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    IconButton(onClick = { onDelete(conversation.id) }) {
                        Icon(
                            Icons.Outlined.Delete,
                            contentDescription = "Delete conversation",
                            modifier = Modifier.size(18.dp),
                        )
                    }
                }
            }
        }
    }
}

/// Coarse on purpose: within a chat list the exact minute is never the question.
private fun relativeTime(millis: Long): String {
    val elapsed = System.currentTimeMillis() - millis
    val minutes = elapsed / 60_000
    return when {
        minutes < 1 -> "just now"
        minutes < 60 -> "$minutes min ago"
        minutes < 60 * 24 -> "${minutes / 60} h ago"
        minutes < 60 * 24 * 7 -> "${minutes / (60 * 24)} d ago"
        else -> DateFormat.getDateInstance(DateFormat.MEDIUM).format(Date(millis))
    }
}
