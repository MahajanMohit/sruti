package dev.sruti.ui.library

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.ArrowBack
import androidx.compose.material.icons.outlined.Check
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
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
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import dev.sruti.convert.QuantType

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    tokenSet: Boolean,
    defaultQuant: QuantType,
    onSetToken: (String?) -> Unit,
    onSetQuant: (QuantType) -> Unit,
    keepCheckpoints: Boolean,
    onSetKeepCheckpoints: (Boolean) -> Unit,
    shellEnabled: Boolean,
    termuxInstalled: Boolean,
    termuxPermitted: Boolean,
    onSetShellEnabled: (Boolean) -> Unit,
    onRequestTermuxPermission: () -> Unit,
    onRunBenchmark: () -> Unit,
    onOpenAbout: () -> Unit,
    onBack: () -> Unit,
) {
    var token by remember { mutableStateOf("") }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Settings", style = MaterialTheme.typography.titleMedium) },
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
                // Without this the quantization options fall off the bottom once
                // the token field and its explanation are present.
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 20.dp)
                .padding(bottom = 32.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Text("Hugging Face access token", style = MaterialTheme.typography.titleMedium)
            Text(
                text = if (tokenSet) {
                    "A token is saved. Gated models such as Llama 3.2 will download, " +
                        "provided you have accepted their licence on huggingface.co."
                } else {
                    "Needed for gated models. Create a read token at " +
                        "huggingface.co/settings/tokens."
                },
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            OutlinedTextField(
                value = token,
                onValueChange = { token = it },
                label = { Text(if (tokenSet) "Replace token" else "Token") },
                singleLine = true,
                // The token grants access to the account; it should not sit
                // legible on screen while the user is in a public place.
                visualTransformation = PasswordVisualTransformation(),
                modifier = Modifier.fillMaxWidth(),
            )

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                TextButton(
                    onClick = {
                        onSetToken(token)
                        token = ""
                    },
                    enabled = token.isNotBlank(),
                ) { Text("Save") }

                if (tokenSet) {
                    TextButton(onClick = { onSetToken(null) }) { Text("Remove") }
                }
            }

            Text(
                text = "Default quantization",
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier.padding(top = 8.dp),
            )

            QuantType.entries.forEach { quant ->
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(14.dp))
                        .background(
                            if (quant == defaultQuant) {
                                MaterialTheme.colorScheme.primaryContainer
                            } else {
                                MaterialTheme.colorScheme.surfaceVariant
                            },
                        )
                        .clickable { onSetQuant(quant) }
                        .padding(16.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        text = quant.label,
                        style = MaterialTheme.typography.bodyLarge,
                        modifier = Modifier.weight(1f),
                    )
                    if (quant == defaultQuant) {
                        Icon(Icons.Outlined.Check, contentDescription = "Selected")
                    }
                }
            }

            Text(
                text = "Storage",
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier.padding(top = 8.dp),
            )
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(14.dp))
                    .background(MaterialTheme.colorScheme.surfaceVariant)
                    .clickable { onSetKeepCheckpoints(!keepCheckpoints) }
                    .padding(16.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text("Keep downloaded weights", style = MaterialTheme.typography.bodyLarge)
                    Text(
                        // The trade is worth stating plainly: several GB of disk
                        // against not re-downloading them to try another setting.
                        text = "Uses several GB more, but re-converting at a different " +
                            "quantization then costs no download",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                Switch(checked = keepCheckpoints, onCheckedChange = onSetKeepCheckpoints)
            }

            Text(
                text = "Agent",
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier.padding(top = 8.dp),
            )
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(14.dp))
                    .background(MaterialTheme.colorScheme.surfaceVariant)
                    .clickable { onSetShellEnabled(!shellEnabled) }
                    .padding(16.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text("Allow shell commands", style = MaterialTheme.typography.bodyLarge)
                    Text(
                        // Said plainly rather than softened. This grants a language
                        // model a real POSIX shell, and the user should decide that
                        // knowing exactly what it means.
                        text = "Lets the agent run commands through Termux — git, python, " +
                            "coreutils. Every command is shown and confirmed before it runs.",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                Switch(checked = shellEnabled, onCheckedChange = onSetShellEnabled)
            }

            // Only once the user has asked for it: naming a prerequisite for
            // something they have not enabled is noise.
            if (shellEnabled) {
                when {
                    !termuxInstalled -> Text(
                        text = "Termux is not installed. Install it from F-Droid or GitHub — " +
                            "the Play Store build cannot accept commands from other apps.",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.error,
                    )

                    !termuxPermitted -> {
                        Text(
                            text = "Termux is installed but has not granted permission yet.",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.error,
                        )
                        TextButton(onClick = onRequestTermuxPermission) {
                            Text("Grant Termux permission")
                        }
                    }

                    else -> Text(
                        // Not detectable from here, so it has to be stated rather
                        // than checked: a command failing for this reason otherwise
                        // looks like the model getting it wrong.
                        text = "Termux is reachable. It must also have " +
                            "allow-external-apps=true in ~/.termux/termux.properties.",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }

            Text(
                text = "About",
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier.padding(top = 8.dp),
            )
            Text(
                text = "What this app does, which architectures it supports and why, and " +
                    "what conversion actually involves.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            TextButton(onClick = onOpenAbout) { Text("How Sruti works") }

            Text(
                text = "Diagnostics",
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier.padding(top = 8.dp),
            )
            Text(
                text = "Measures prefill and decode speed, then holds a sustained load for " +
                    "ten minutes to see how far this device throttles.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            TextButton(onClick = onRunBenchmark) { Text("Run benchmark") }
        }
    }
}
