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
import androidx.compose.material.icons.outlined.Build
import androidx.compose.material.icons.outlined.Delete
import androidx.compose.material.icons.outlined.History
import androidx.compose.material.icons.outlined.Stop
import androidx.compose.material.icons.outlined.Tune
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LocalContentColor
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
import dev.sruti.llm.ChatMetrics
import dev.sruti.ui.theme.Haptic
import dev.sruti.ui.theme.LocalHaptics
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
    onSetAgentMode: (Boolean) -> Unit,
    onResolveConfirmation: (Boolean) -> Unit,
    onOpenSettings: () -> Unit,
) {
    var draft by remember { mutableStateOf("") }
    var modelMenuOpen by remember { mutableStateOf(false) }
    var historyOpen by remember { mutableStateOf(false) }
    val haptics = LocalHaptics.current

    // The model's half of the conversation, felt rather than watched. Keyed on
    // the transition itself so each fires exactly once: the first token says the
    // wait is over, and completion says the reply is whole.
    LaunchedEffect(state.streamingText.isNotEmpty()) {
        if (state.streamingText.isNotEmpty()) haptics.play(Haptic.FirstToken)
    }
    // Only on the true -> false edge. Keying on the flag alone would fire once on
    // entering the screen, and again every time a stored conversation is opened.
    var wasGenerating by remember { mutableStateOf(false) }
    LaunchedEffect(state.isGenerating) {
        if (wasGenerating && !state.isGenerating) haptics.play(Haptic.Complete)
        wasGenerating = state.isGenerating
    }
    LaunchedEffect(state.error) {
        if (state.error != null) haptics.play(Haptic.Error)
    }

    state.pendingConfirmation?.let { pending ->
        LaunchedEffect(pending) { haptics.play(Haptic.ToolCall) }
        ConfirmationDialog(
            pending = pending,
            onAllow = { onResolveConfirmation(true) },
            onDecline = { onResolveConfirmation(false) },
        )
    }

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
                        ContextLine(
                            isLoadingModel = state.isLoadingModel,
                            modelReady = state.modelReady,
                            contextUsed = state.contextUsed,
                            contextTotal = state.contextTotal,
                            metrics = state.lastMetrics,
                        )
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
                    IconButton(
                        onClick = {
                            haptics.play(Haptic.Select)
                            onSetAgentMode(!state.agentMode)
                        },
                        enabled = state.modelReady,
                    ) {
                        Icon(
                            Icons.Outlined.Build,
                            contentDescription = if (state.agentMode) {
                                "Agent mode on"
                            } else {
                                "Agent mode off"
                            },
                            tint = if (state.agentMode) {
                                MaterialTheme.colorScheme.primary
                            } else {
                                LocalContentColor.current
                            },
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
                                        haptics.play(Haptic.Select)
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
            ContextMeter(modelReady = state.modelReady, contextFraction = state.contextFraction)

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

            AgentBanner(
                agentMode = state.agentMode,
                shellEnabled = state.shellEnabled,
                termuxAvailable = state.termuxAvailable,
                onOpenSettings = onOpenSettings,
            )

            AgentTrace(trace = state.trace)

            // Passed field by field rather than as the whole state: streamingText
            // changes on every flushed frame, and a composable taking the state
            // object recomposes then even when nothing it draws has changed.
            Transcript(
                messages = state.messages,
                streamingText = state.streamingText,
                isGenerating = state.isGenerating,
                modifier = Modifier.weight(1f),
            )

            Composer(
                draft = draft,
                onDraftChange = { draft = it },
                enabled = state.canSend,
                isGenerating = state.isGenerating,
                onSend = {
                    haptics.play(Haptic.Send)
                    onSend(draft)
                    draft = ""
                },
                onStop = {
                    haptics.play(Haptic.Select)
                    onStop()
                },
            )
        }
    }
}

@Composable
private fun ContextLine(
    isLoadingModel: Boolean,
    modelReady: Boolean,
    contextUsed: Int,
    contextTotal: Int,
    metrics: ChatMetrics?,
) {
    val text = when {
        isLoadingModel -> "Loading model…"
        !modelReady -> "Choose a model to begin"
        else -> buildString {
            append("$contextUsed / $contextTotal tokens")
            metrics?.let { m ->
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
private fun ContextMeter(modelReady: Boolean, contextFraction: Float) {
    if (!modelReady) return

    val fraction by animateFloatAsState(
        targetValue = contextFraction,
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
        color = if (contextFraction > 0.85f) {
            MaterialTheme.colorScheme.error
        } else {
            MaterialTheme.colorScheme.primary
        },
    )
}

@Composable
private fun Transcript(
    messages: List<DisplayMessage>,
    streamingText: String,
    isGenerating: Boolean,
    modifier: Modifier = Modifier,
) {
    val listState = rememberLazyListState()

    // Follow the tail as the reply grows. Keyed on length rather than a token
    // count so it also tracks the streaming message.
    LaunchedEffect(messages.size, streamingText.length) {
        val lastIndex = messages.size + if (streamingText.isNotEmpty()) 1 else 0
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
        items(messages, key = { it.id }) { message ->
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

        if (streamingText.isNotEmpty()) {
            item(key = "streaming") {
                MessageBubble(content = streamingText, isUser = false, footnote = null)
            }
        }

        if (isGenerating && streamingText.isEmpty()) {
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
    val haptics = LocalHaptics.current
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
                    IconButton(
                        onClick = {
                            haptics.play(Haptic.Destructive)
                            onDelete(conversation.id)
                        },
                    ) {
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

/**
 * States what agent mode can and cannot reach right now.
 *
 * Shown rather than hidden because the difference between "can read files" and
 * "can run shell commands" is the difference the user most needs to know before
 * typing a request — and because a tool that silently is not there looks like a
 * model that cannot follow instructions.
 */
@Composable
private fun AgentBanner(
    agentMode: Boolean,
    shellEnabled: Boolean,
    termuxAvailable: Boolean,
    onOpenSettings: () -> Unit,
) {
    AnimatedVisibility(
        visible = agentMode,
        enter = fadeIn(Motion.quick()) + expandVertically(Motion.size()),
        exit = fadeOut(Motion.quick()) + shrinkVertically(Motion.size()),
    ) {
        val text = when {
            shellEnabled && termuxAvailable ->
                "Agent mode · files, network, clipboard and the Termux shell"
            shellEnabled ->
                "Agent mode · shell is on but Termux is not reachable. Tap to fix."
            else ->
                "Agent mode · files, network and clipboard. Tap to enable the shell."
        }
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .clickable(onClick = onOpenSettings)
                .background(MaterialTheme.colorScheme.surfaceVariant)
                .padding(horizontal = 20.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Icon(Icons.Outlined.Build, contentDescription = null, modifier = Modifier.size(15.dp))
            Text(text, style = MaterialTheme.typography.labelSmall)
        }
    }
}

/** The agent's steps, in the order they happened. */
@Composable
private fun AgentTrace(trace: List<TraceLine>) {
    if (trace.isEmpty()) return

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(max = 200.dp)
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 6.dp),
        verticalArrangement = Arrangement.spacedBy(3.dp),
    ) {
        trace.forEach { line ->
            Text(
                text = line.text,
                style = MaterialTheme.typography.labelSmall,
                color = when (line.kind) {
                    TraceLine.Kind.Failure -> MaterialTheme.colorScheme.error
                    TraceLine.Kind.Tool -> MaterialTheme.colorScheme.primary
                    else -> MaterialTheme.colorScheme.onSurfaceVariant
                },
            )
        }
    }
}

/**
 * Asks before a tool that changes something runs.
 *
 * The arguments are shown verbatim, not summarised. A shell command the user
 * cannot read in full is a shell command they cannot meaningfully approve.
 */
@Composable
private fun ConfirmationDialog(
    pending: PendingConfirmation,
    onAllow: () -> Unit,
    onDecline: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = onDecline,
        title = { Text("Run ${pending.toolName}?") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(pending.description, style = MaterialTheme.typography.bodyMedium)
                SelectionContainer {
                    Column(
                        modifier = Modifier
                            .heightIn(max = 240.dp)
                            .verticalScroll(rememberScrollState()),
                        verticalArrangement = Arrangement.spacedBy(4.dp),
                    ) {
                        pending.arguments.forEach { (name, value) ->
                            Text(
                                text = "$name = $value",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                }
            }
        },
        confirmButton = { TextButton(onClick = onAllow) { Text("Run") } },
        dismissButton = { TextButton(onClick = onDecline) { Text("Decline") } },
    )
}
