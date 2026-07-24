package dev.sruti.ui.phase0

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import dev.sruti.bench.BenchmarkRunner
import dev.sruti.llm.DeviceCapabilities
import dev.sruti.llm.GenerationEvent
import dev.sruti.llm.GenerationMetrics
import dev.sruti.llm.LlamaEngine
import dev.sruti.llm.SamplingParams
import dev.sruti.ui.coalesceToFrames
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.mapNotNull
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import javax.inject.Inject

data class Phase0UiState(
    val modelFile: File? = null,
    val modelLabel: String? = null,
    val isImporting: Boolean = false,
    val isBusy: Boolean = false,
    val stage: String? = null,
    val log: List<String> = emptyList(),
    val streamedText: String = "",
    val reportPath: String? = null,
    val error: String? = null,
) {
    val threadCount: Int get() = DeviceCapabilities.recommendedThreadCount()
    val canRun: Boolean get() = modelFile != null && !isBusy && !isImporting
}

@HiltViewModel
class Phase0ViewModel @Inject constructor(
    application: Application,
) : AndroidViewModel(application) {

    private val _uiState = MutableStateFlow(Phase0UiState())
    val uiState: StateFlow<Phase0UiState> = _uiState.asStateFlow()

    private var runJob: Job? = null

    init {
        // Survive process recreation without forcing a re-import.
        modelsDir().listFiles { f -> f.extension == "gguf" }
            ?.maxByOrNull { it.lastModified() }
            ?.let { existing ->
                _uiState.update {
                    it.copy(modelFile = existing, modelLabel = existing.name)
                }
            }
    }

    /**
     * Copies a picked GGUF into app storage.
     *
     * llama.cpp mmaps by path, so a SAF content URI will not do. Copying also puts
     * the file somewhere the app can still reach after a permission grant lapses.
     */
    fun importModel(uri: Uri, displayName: String) {
        if (_uiState.value.isImporting) return

        viewModelScope.launch {
            _uiState.update { it.copy(isImporting = true, error = null) }
            runCatching {
                withContext(Dispatchers.IO) {
                    val target = File(modelsDir(), displayName.ifBlank { "model.gguf" })
                    getApplication<Application>().contentResolver
                        .openInputStream(uri)
                        ?.use { input -> target.outputStream().use(input::copyTo) }
                        ?: error("could not open $uri")
                    target
                }
            }.onSuccess { file ->
                _uiState.update {
                    it.copy(
                        isImporting = false,
                        modelFile = file,
                        modelLabel = file.name,
                        log = it.log + "Imported ${file.name} (${file.length() / (1024 * 1024)} MiB)",
                    )
                }
            }.onFailure { t ->
                _uiState.update {
                    it.copy(isImporting = false, error = t.message ?: "import failed")
                }
            }
        }
    }

    /** Loads the model and streams one short generation — the end-to-end smoke test. */
    fun runSmokeTest(prompt: String) {
        val model = _uiState.value.modelFile ?: return
        if (_uiState.value.isBusy) return

        runJob = viewModelScope.launch {
            _uiState.update {
                it.copy(isBusy = true, stage = "Loading model", streamedText = "", error = null)
            }
            runCatching {
                val engine = LlamaEngine.load(
                    modelFile = model,
                    dispatcher = Dispatchers.Default,
                    nCtx = 2048,
                )
                try {
                    _uiState.update { it.copy(stage = "Generating") }

                    var metrics: GenerationMetrics? = null

                    engine.generate(prompt, SamplingParams(maxTokens = 192))
                        .mapNotNull { event ->
                            when (event) {
                                is GenerationEvent.Token -> event.piece
                                // Terminal event carries no text; capture and drop it
                                // so the downstream stays a pure String flow.
                                is GenerationEvent.Completed -> {
                                    metrics = event.metrics
                                    null
                                }
                            }
                        }
                        .coalesceToFrames()
                        .collect { chunk ->
                            _uiState.update { it.copy(streamedText = it.streamedText + chunk) }
                        }

                    metrics?.let { m ->
                        _uiState.update {
                            it.copy(
                                log = it.log + "prefill %.1f tok/s · decode %.1f tok/s · %d tokens"
                                    .format(
                                        m.prefillTokensPerSecond,
                                        m.decodeTokensPerSecond,
                                        m.generatedTokens,
                                    ),
                            )
                        }
                    }
                } finally {
                    engine.close()
                }
            }.onFailure { t ->
                _uiState.update { it.copy(error = t.message ?: "generation failed") }
            }
            _uiState.update { it.copy(isBusy = false, stage = null) }
        }
    }

    fun runBenchmark() {
        val model = _uiState.value.modelFile ?: return
        if (_uiState.value.isBusy) return

        runJob = viewModelScope.launch {
            _uiState.update { it.copy(isBusy = true, log = emptyList(), error = null) }
            runCatching {
                BenchmarkRunner(getApplication(), model)
                    .run()
                    .collect { progress ->
                        when (progress) {
                            is BenchmarkRunner.Progress.Stage ->
                                _uiState.update { it.copy(stage = progress.label) }

                            is BenchmarkRunner.Progress.Line ->
                                _uiState.update { it.copy(log = it.log + progress.text) }

                            is BenchmarkRunner.Progress.Done ->
                                _uiState.update {
                                    it.copy(
                                        stage = null,
                                        reportPath = progress.reportFile.absolutePath,
                                        log = it.log + "Report written to ${progress.reportFile.name}",
                                    )
                                }
                        }
                    }
            }.onFailure { t ->
                _uiState.update { it.copy(error = t.message ?: "benchmark failed") }
            }
            _uiState.update { it.copy(isBusy = false, stage = null) }
        }
    }

    fun cancel() {
        runJob?.cancel()
        runJob = null
        _uiState.update { it.copy(isBusy = false, stage = null) }
    }

    private fun modelsDir(): File =
        File(getApplication<Application>().filesDir, "models").apply { mkdirs() }
}
