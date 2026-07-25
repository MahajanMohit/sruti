package dev.sruti.ui.library

import android.app.Application
import androidx.compose.runtime.Immutable
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import dagger.hilt.android.lifecycle.HiltViewModel
import dev.sruti.convert.CheckpointInfo
import dev.sruti.convert.QuantType
import dev.sruti.hub.HubException
import dev.sruti.hub.HuggingFaceApi
import dev.sruti.hub.InstalledModel
import dev.sruti.hub.ModelStore
import dev.sruti.hub.RemoteModelSummary
import dev.sruti.hub.SearchFilters
import dev.sruti.hub.StagedCheckpoint
import dev.sruti.llm.GgufFact
import dev.sruti.llm.GgufInspector
import dev.sruti.settings.SettingsStore
import dev.sruti.work.ModelJobState
import dev.sruti.work.ModelWorkService
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

@Immutable
data class LibraryUiState(
    val installed: List<InstalledModel> = emptyList(),
    val staged: List<StagedCheckpoint> = emptyList(),
    val usedBytes: Long = 0,
    val defaultQuant: QuantType = QuantType.Q4_K_M,
    val job: ModelJobState = ModelJobState.Idle,
    val tokenSet: Boolean = false,
    val keepCheckpoints: Boolean = false,
    val shellEnabled: Boolean = false,
    val termuxInstalled: Boolean = false,
    val termuxPermitted: Boolean = false,
    val isLoading: Boolean = true,
)

@Immutable
data class BrowseUiState(
    val query: String = "",
    val filters: SearchFilters = SearchFilters(),
    val results: List<RemoteModelSummary> = emptyList(),
    val isSearching: Boolean = false,
    val error: String? = null,
    /** Inspection of the currently selected repository, keyed by repo id. */
    val inspecting: String? = null,
    val inspected: Map<String, CheckpointInfo> = emptyMap(),
)

@Immutable
data class ModelDetailUiState(
    val model: InstalledModel? = null,
    val facts: List<GgufFact> = emptyList(),
    val loading: Boolean = false,
)

@HiltViewModel
class LibraryViewModel @Inject constructor(
    application: Application,
    private val store: ModelStore,
    private val api: HuggingFaceApi,
    private val settings: SettingsStore,
    private val termux: dev.sruti.agent.TermuxTool,
) : AndroidViewModel(application) {

    private val _library = MutableStateFlow(LibraryUiState())
    val library: StateFlow<LibraryUiState> = _library.asStateFlow()

    private val _browse = MutableStateFlow(BrowseUiState())
    val browse: StateFlow<BrowseUiState> = _browse.asStateFlow()

    private val _detail = MutableStateFlow(ModelDetailUiState())
    val detail: StateFlow<ModelDetailUiState> = _detail.asStateFlow()

    private var searchJob: Job? = null

    init {
        viewModelScope.launch {
            ModelWorkService.state.collect { state ->
                _library.update { it.copy(job = state) }
                // A finished job changes what is on disk, so re-read it.
                if (state is ModelJobState.Succeeded) refresh()
            }
        }
        viewModelScope.launch {
            settings.defaultQuantType.collect { quant ->
                _library.update { it.copy(defaultQuant = quant) }
            }
        }
        viewModelScope.launch {
            settings.huggingFaceToken.collect { token ->
                _library.update { it.copy(tokenSet = !token.isNullOrBlank()) }
            }
        }
        viewModelScope.launch {
            settings.keepCheckpoints.collect { keep ->
                _library.update { it.copy(keepCheckpoints = keep) }
            }
        }
        viewModelScope.launch {
            settings.shellEnabled.collect { enabled ->
                _library.update { it.copy(shellEnabled = enabled) }
                refreshTermuxStatus()
            }
        }
        refresh()
    }

    fun refresh() {
        viewModelScope.launch {
            _library.update { it.copy(isLoading = true) }
            val installed = store.installedModels()
            val staged = store.stagedCheckpoints()
            val used = store.usedBytes()
            _library.update {
                it.copy(installed = installed, staged = staged, usedBytes = used, isLoading = false)
            }
        }
    }

    // --- browsing -----------------------------------------------------------

    fun onQueryChanged(query: String) {
        _browse.update { it.copy(query = query) }

        searchJob?.cancel()
        searchJob = viewModelScope.launch {
            // Debounce: a search per keystroke would rate-limit against the Hub
            // and make the list flicker through irrelevant intermediate results.
            delay(350)
            search(query)
        }
    }

    /**
     * Applies a filter change.
     *
     * Re-searches immediately rather than debouncing: a tap is a deliberate act,
     * unlike the keystrokes [onQueryChanged] has to absorb.
     */
    fun onFiltersChanged(filters: SearchFilters) {
        _browse.update { it.copy(filters = filters) }
        search()
    }

    fun search(query: String = _browse.value.query) {
        searchJob?.cancel()
        searchJob = viewModelScope.launch {
            _browse.update { it.copy(isSearching = true, error = null) }
            runCatching { api.search(query, _browse.value.filters) }
                .onSuccess { results ->
                    _browse.update { it.copy(results = results, isSearching = false) }
                }
                .onFailure { t ->
                    _browse.update {
                        it.copy(
                            isSearching = false,
                            error = (t as? HubException)?.message ?: t.message ?: "search failed",
                        )
                    }
                }
        }
    }

    /**
     * Checks whether a repository can actually be converted.
     *
     * Fetches config.json only — a few kilobytes against gigabytes of weights — so
     * the user learns an architecture is unsupported before starting a download
     * rather than after it.
     */
    fun inspect(repoId: String) {
        if (_browse.value.inspected.containsKey(repoId)) return

        viewModelScope.launch {
            _browse.update { it.copy(inspecting = repoId) }
            val info = runCatching { api.fetchText(repoId, "config.json") }
                .fold(
                    onSuccess = { CheckpointInfo.inspect(it) },
                    onFailure = { t ->
                        CheckpointInfo(
                            supported = false,
                            error = (t as? HubException)?.message
                                ?: t.message
                                ?: "could not read config.json",
                        )
                    },
                )
            _browse.update {
                it.copy(inspecting = null, inspected = it.inspected + (repoId to info))
            }
        }
    }

    // --- model detail -------------------------------------------------------

    /**
     * Loads one model's details.
     *
     * Resolved from the store rather than from [library], because this view model
     * is scoped to the detail route and its own list has not been read yet when
     * the screen first composes. Keyed by absolute path rather than by position,
     * which would point at a different model once another finishes converting.
     */
    fun openDetail(path: String) {
        _detail.value = ModelDetailUiState(loading = true)
        viewModelScope.launch {
            val model = store.installedModels().firstOrNull { it.file.absolutePath == path }
            if (model == null) {
                _detail.value = ModelDetailUiState(loading = false)
                return@launch
            }
            _detail.value = ModelDetailUiState(model = model, loading = true)
            val facts = GgufInspector.inspect(model.file)
            _detail.update { it.copy(facts = facts, loading = false) }
        }
    }

    // --- actions ------------------------------------------------------------

    fun startAcquire(repoId: String, quantType: QuantType = _library.value.defaultQuant) {
        ModelWorkService.start(getApplication(), repoId, quantType)
    }

    fun cancelJob() {
        ModelWorkService.cancel(getApplication())
    }

    fun acknowledgeJob() {
        ModelWorkService.acknowledge()
    }

    fun delete(model: InstalledModel) {
        viewModelScope.launch {
            store.delete(model)
            refresh()
        }
    }

    fun delete(checkpoint: StagedCheckpoint) {
        viewModelScope.launch {
            store.delete(checkpoint)
            refresh()
        }
    }

    fun setDefaultQuant(quantType: QuantType) {
        viewModelScope.launch { settings.setDefaultQuantType(quantType) }
    }

    /**
     * Re-reads whether Termux is installed and has granted permission.
     *
     * Both can change while the app is running — the user leaves to install
     * Termux, or answers the permission prompt — so this is called on entering
     * the screen rather than cached at construction.
     */
    fun refreshTermuxStatus() {
        _library.update {
            it.copy(
                termuxInstalled = termux.isTermuxInstalled(),
                termuxPermitted = termux.hasPermission(),
            )
        }
    }

    fun setShellEnabled(enabled: Boolean) {
        viewModelScope.launch { settings.setShellEnabled(enabled) }
    }

    fun setKeepCheckpoints(keep: Boolean) {
        viewModelScope.launch { settings.setKeepCheckpoints(keep) }
    }

    /** Web page for a repository, so a gated licence can be accepted. */
    fun modelPageUrl(repoId: String): String = api.modelPageUrl(repoId)

    fun setToken(token: String?) {
        viewModelScope.launch { settings.setHuggingFaceToken(token) }
    }
}
