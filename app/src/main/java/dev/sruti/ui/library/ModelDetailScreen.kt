package dev.sruti.ui.library

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.ArrowBack
import androidx.compose.material3.CircularProgressIndicator
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
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import dev.sruti.hub.InstalledModel
import dev.sruti.llm.GgufFact
import dev.sruti.ui.formatBytes
import java.text.DateFormat
import java.util.Date

/**
 * Everything known about one installed model.
 *
 * Two sources, kept visibly separate. The top half is what the app recorded when
 * it built the file; the bottom half is read back out of the file's own header,
 * so it describes what is actually there rather than what was intended. When a
 * model behaves oddly this is the screen that says why — a missing chat template
 * and a mixed quantization are both things you can only see here.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelDetailScreen(
    model: InstalledModel,
    facts: List<GgufFact>,
    loading: Boolean,
    onBack: () -> Unit,
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        text = model.metadata.displayName,
                        style = MaterialTheme.typography.titleMedium,
                        maxLines = 1,
                        overflow = TextOverflow.MiddleEllipsis,
                    )
                },
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
            verticalArrangement = Arrangement.spacedBy(2.dp),
        ) {
            Heading("Where it came from")
            Fact("Source", model.metadata.sourceRepo.ifBlank { "sideloaded" })
            Fact("File", model.file.name)
            Fact("On disk", formatBytes(model.sizeBytes))
            Fact("Quantization", model.metadata.quant.id)
            if (model.metadata.convertedAtMillis > 0) {
                Fact(
                    "Converted",
                    DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT)
                        .format(Date(model.metadata.convertedAtMillis)),
                )
            }

            HorizontalDivider(modifier = Modifier.padding(vertical = 14.dp))
            Heading("Read from the file")

            when {
                loading -> Row(
                    modifier = Modifier.fillMaxWidth().padding(vertical = 12.dp),
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    CircularProgressIndicator(modifier = Modifier.size(16.dp))
                    Text("Reading header…", style = MaterialTheme.typography.bodyMedium)
                }

                facts.isEmpty() -> Text(
                    text = "The file's header could not be read. It may be truncated — " +
                        "delete it and convert again.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.error,
                    modifier = Modifier.padding(vertical = 8.dp),
                )

                else -> facts.forEach { fact -> Fact(fact.label, fact.value) }
            }
        }
    }
}

@Composable
private fun Heading(text: String) {
    Text(
        text = text,
        style = MaterialTheme.typography.titleMedium,
        modifier = Modifier.padding(top = 8.dp, bottom = 6.dp),
    )
}

@Composable
private fun Fact(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 5.dp),
        horizontalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.weight(1f),
        )
        Text(
            // Values run long — a weight-type mixture lists every quantization in
            // the file — so this side takes the extra room and wraps.
            text = value,
            style = MaterialTheme.typography.bodyMedium,
            textAlign = TextAlign.End,
            modifier = Modifier.weight(1.4f),
        )
    }
}
