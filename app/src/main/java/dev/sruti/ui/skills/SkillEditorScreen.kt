package dev.sruti.ui.skills

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.ArrowBack
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import dev.sruti.ui.theme.Haptic
import dev.sruti.ui.theme.LocalHaptics

/**
 * Writing a skill.
 *
 * A single monospaced text field over the file itself, rather than a form. The
 * format is small enough to hold in the head, a form would hide it, and a file
 * the user can read is a file they can share — which is the point.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SkillEditorScreen(
    state: SkillEditorState,
    onTextChanged: (String) -> Unit,
    onSave: () -> Unit,
    onBack: () -> Unit,
) {
    val haptics = LocalHaptics.current

    // Saving is the end of the task, so leaving is the expected next step.
    LaunchedEffect(state.saved) {
        if (state.saved) {
            haptics.play(Haptic.Complete)
            onBack()
        }
    }
    LaunchedEffect(state.error) {
        if (state.error != null) haptics.play(Haptic.Error)
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        text = state.originalName ?: "New skill",
                        style = MaterialTheme.typography.titleMedium,
                    )
                },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Outlined.ArrowBack, contentDescription = "Back")
                    }
                },
                actions = {
                    TextButton(onClick = onSave) { Text("Save") }
                },
            )
        },
    ) { insets ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(insets)
                .imePadding()
                .padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            state.error?.let { message ->
                Text(
                    // The parser's own message, unaltered: it names the line that
                    // is wrong, which is the only thing that helps here.
                    text = message,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.error,
                )
            }

            OutlinedTextField(
                value = state.text,
                onValueChange = onTextChanged,
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
                textStyle = TextStyle(
                    // Monospaced because alignment carries meaning here: the
                    // frontmatter delimiters and list indentation are the syntax.
                    fontFamily = FontFamily.Monospace,
                    fontSize = 13.sp,
                ),
            )

            Text(
                text = "Steps run in order. Use {{name}} for a parameter and {{step1}} for " +
                    "the first step's output. A skill with no steps is guidance the model " +
                    "reads before choosing tools itself.",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(bottom = 12.dp),
            )
        }
    }
}
