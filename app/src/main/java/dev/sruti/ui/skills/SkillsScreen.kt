package dev.sruti.ui.skills

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateContentSize
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.ArrowBack
import androidx.compose.material.icons.outlined.Add
import androidx.compose.material.icons.outlined.Delete
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.unit.dp
import dev.sruti.agent.Skill
import dev.sruti.ui.theme.Haptic
import dev.sruti.ui.theme.LocalHaptics
import dev.sruti.ui.theme.Motion

/**
 * The skills the agent can use.
 *
 * Presented as documents rather than settings, because that is what they are —
 * each one is a file the user can write, edit, share and import.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SkillsScreen(
    state: SkillsUiState,
    missingTools: (Skill) -> List<String>,
    onNew: () -> Unit,
    onEdit: (Skill) -> Unit,
    onDelete: (Skill) -> Unit,
    onImportFile: () -> Unit,
    onImportUrl: (String) -> Unit,
    onDismissError: () -> Unit,
    onBack: () -> Unit,
) {
    val haptics = LocalHaptics.current
    var urlDialogOpen by remember { mutableStateOf(false) }

    if (urlDialogOpen) {
        ImportUrlDialog(
            onImport = {
                urlDialogOpen = false
                onImportUrl(it)
            },
            onDismiss = { urlDialogOpen = false },
        )
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Skills", style = MaterialTheme.typography.titleMedium) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Outlined.ArrowBack, contentDescription = "Back")
                    }
                },
            )
        },
        floatingActionButton = {
            ExtendedFloatingActionButton(
                onClick = onNew,
                icon = { Icon(Icons.Outlined.Add, contentDescription = null) },
                text = { Text("New skill") },
            )
        },
    ) { insets ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(insets),
            contentPadding = PaddingValues(start = 20.dp, end = 20.dp, bottom = 96.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            item {
                Text(
                    text = "A skill writes down the steps for a task, so the model only has " +
                        "to read the values out of your request rather than plan. That is " +
                        "the difference between something a small model does reliably and " +
                        "something it does sometimes.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(bottom = 6.dp),
                )
            }

            item {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    TextButton(onClick = onImportFile) { Text("Import file") }
                    TextButton(onClick = { urlDialogOpen = true }) { Text("Import from URL") }
                }
            }

            state.error?.let { message ->
                item {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text(
                            text = message,
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.error,
                            modifier = Modifier.weight(1f),
                        )
                        TextButton(onClick = onDismissError) { Text("Dismiss") }
                    }
                }
            }

            item {
                AnimatedVisibility(visible = state.importing) {
                    CircularProgressIndicator(modifier = Modifier.size(20.dp))
                }
            }

            items(state.skills, key = { it.name }) { skill ->
                SkillCard(
                    skill = skill,
                    missing = missingTools(skill),
                    onEdit = { onEdit(skill) },
                    onDelete = {
                        haptics.play(Haptic.Destructive)
                        onDelete(skill)
                    },
                )
            }
        }
    }
}

@Composable
private fun SkillCard(
    skill: Skill,
    missing: List<String>,
    onEdit: () -> Unit,
    onDelete: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(18.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant)
            // Built-ins are read-only, so making them tappable would promise an
            // editor that then refuses to open.
            .clickable(enabled = !skill.builtIn, onClick = onEdit)
            .padding(18.dp)
            .animateContentSize(Motion.size()),
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                text = skill.name,
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier.weight(1f),
            )
            if (!skill.builtIn) {
                IconButton(onClick = onDelete) {
                    Icon(
                        Icons.Outlined.Delete,
                        contentDescription = "Delete skill",
                        modifier = Modifier.size(18.dp),
                    )
                }
            }
        }

        Text(
            text = skill.description,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        Text(
            text = buildList {
                add(if (skill.builtIn) "built in" else "yours")
                add(
                    if (skill.isTemplated) {
                        "${skill.steps.size} fixed step" + if (skill.steps.size == 1) "" else "s"
                    } else {
                        "guidance only"
                    },
                )
                if (skill.parameters.isNotEmpty()) {
                    add(skill.parameters.joinToString(", ") { it.name })
                }
            }.joinToString(" · "),
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        if (missing.isNotEmpty()) {
            Text(
                // Stated on the card rather than discovered at run time, when the
                // user has already asked for something that cannot happen.
                text = "Needs tools this device does not have: " + missing.joinToString(", "),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.error,
            )
        }
    }
}

@Composable
private fun ImportUrlDialog(onImport: (String) -> Unit, onDismiss: () -> Unit) {
    var url by remember { mutableStateOf("") }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Import a skill") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(
                    text = "A direct link to the skill file itself — a gist raw URL, or a " +
                        "raw file from a repository.",
                    style = MaterialTheme.typography.bodyMedium,
                )
                OutlinedTextField(
                    value = url,
                    onValueChange = { url = it },
                    label = { Text("URL") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        },
        confirmButton = {
            TextButton(onClick = { onImport(url) }, enabled = url.isNotBlank()) {
                Text("Import")
            }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}
