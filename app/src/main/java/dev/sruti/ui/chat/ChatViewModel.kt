package dev.sruti.ui.chat

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import dagger.hilt.android.lifecycle.HiltViewModel
import dev.sruti.data.ChatDao
import dev.sruti.data.ConversationEntity
import dev.sruti.data.MessageEntity
import dev.sruti.hub.InstalledModel
import dev.sruti.hub.ModelStore
import dev.sruti.llm.ChatEvent
import dev.sruti.llm.ChatMessage
import dev.sruti.llm.ChatMetrics
import dev.sruti.llm.ChatParams
import dev.sruti.llm.ChatSession
import dev.sruti.llm.LlamaEngine
import dev.sruti.llm.ThermalGovernor
import dev.sruti.ui.coalesceToFrames
import dev.sruti.work.ModelWorkService
import dev.sruti.work.ModelJobState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.mapNotNull
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

/** A message as shown in the transcript. */
data class DisplayMessage(
    val id: Long,
    val role: ChatMessage.Role,
    val content: String,
    val tokensPerSecond: Double = 0.0,
    val tokenCount: Int = 0,
)

/** A past conversation, as listed in the history sheet. */
data class ConversationSummary(
    val id: Long,
    val title: String,
    val updatedAtMillis: Long,
    val modelFileName: String,
)

data class ChatUiState(
    val models: List<InstalledModel> = emptyList(),
    val selectedModel: InstalledModel? = null,
    val isLoadingModel: Boolean = false,
    val modelReady: Boolean = false,

    val conversations: List<ConversationSummary> = emptyList(),
    /** Zero until the current conversation has been written, i.e. has a message. */
    val conversationId: Long = 0,

    val messages: List<DisplayMessage> = emptyList(),
    /** Text of the reply currently being generated. */
    val streamingText: String = "",
    val isGenerating: Boolean = false,

    val contextUsed: Int = 0,
    val contextTotal: Int = 0,
    val lastMetrics: ChatMetrics? = null,
    /** Non-empty when the thermal governor is actively slowing generation. */
    val thermalNotice: String = "",

    val systemPrompt: String = "",
    val params: ChatParams = ChatParams(),
    val error: String? = null,
) {
    val contextFraction: Float
        get() = if (contextTotal > 0) contextUsed.toFloat() / contextTotal else 0f

    val canSend: Boolean get() = modelReady && !isGenerating && !isLoadingModel
}

@HiltViewModel
class ChatViewModel @Inject constructor(
    application: Application,
    private val store: ModelStore,
    private val dao: ChatDao,
    private val settings: dev.sruti.settings.SettingsStore,
) : AndroidViewModel(application) {

    private val _uiState = MutableStateFlow(ChatUiState())
    val uiState: StateFlow<ChatUiState> = _uiState.asStateFlow()

    private val governor = ThermalGovernor(application)

    private var engine: LlamaEngine? = null
    private var session: ChatSession? = null
    private var conversationId: Long = 0
    private var generationJob: Job? = null

    init {
        refreshModels()

        // A conversion finishing changes what is installed, and this screen is
        // usually not the one in front of the user when it does. Without this the
        // new model only appears after the app is relaunched.
        viewModelScope.launch {
            ModelWorkService.state.collect { state ->
                if (state is ModelJobState.Succeeded) refreshModels()
            }
        }

        viewModelScope.launch {
            dao.deleteEmptyConversations()
            dao.observeConversations().collect { rows ->
                _uiState.update {
                    it.copy(
                        conversations = rows.map { row ->
                            ConversationSummary(
                                id = row.id,
                                title = row.title,
                                updatedAtMillis = row.updatedAtMillis,
                                modelFileName = row.modelFileName,
                            )
                        },
                    )
                }
            }
        }
    }

    /**
     * Re-reads installed models, selecting one if none is loaded yet.
     *
     * Called on entering the screen as well, since a model can also arrive by
     * being sideloaded while the app is running.
     */
    fun refreshModels() {
        viewModelScope.launch {
            val models = store.installedModels()
            _uiState.update { it.copy(models = models) }

            val current = _uiState.value.selectedModel
            val stillPresent = models.any { it.file == current?.file }
            if (!stillPresent || current == null) {
                models.firstOrNull()?.let { selectModel(it) }
            }
        }
    }

    fun selectModel(model: InstalledModel) {
        if (_uiState.value.selectedModel?.file == model.file && _uiState.value.modelReady) return

        generationJob?.cancel()
        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    selectedModel = model,
                    isLoadingModel = true,
                    modelReady = false,
                    error = null,
                )
            }

            releaseEngine()

            runCatching {
                val loaded = LlamaEngine.load(
                    modelFile = model.file,
                    dispatcher = Dispatchers.Default,
                    nCtx = DEFAULT_CONTEXT,
                    // Zero unless the user opted in. Whether offload helps is a
                    // property of the device, so it is a setting, not a default.
                    nGpuLayers = settings.currentGpuLayers(),
                )
                engine = loaded
                session = loaded.newChatSession(governor)
                loaded
            }.onSuccess { loaded ->
                _uiState.update {
                    it.copy(
                        isLoadingModel = false,
                        modelReady = true,
                        contextTotal = loaded.contextLength,
                        contextUsed = 0,
                    )
                }
                startNewConversation()
            }.onFailure { t ->
                _uiState.update {
                    it.copy(
                        isLoadingModel = false,
                        modelReady = false,
                        error = t.message ?: "could not load model",
                    )
                }
            }
        }
    }

    /**
     * Clears the transcript to start a fresh conversation.
     *
     * No row is written here. A conversation only exists once it has a message —
     * otherwise every model load, every app launch and every tap of "new" left an
     * identical empty entry in the history, which is worse than no history at all.
     * The previous conversation is already saved and reachable from the list.
     */
    fun startNewConversation() {
        if (_uiState.value.selectedModel == null) return
        generationJob?.cancel()

        viewModelScope.launch {
            session?.reset()
            conversationId = 0
            _uiState.update {
                it.copy(
                    conversationId = 0,
                    messages = emptyList(),
                    streamingText = "",
                    contextUsed = 0,
                    lastMetrics = null,
                    error = null,
                )
            }
        }
    }

    /** Reopens a stored conversation, replaying it into the transcript. */
    fun openConversation(id: Long) {
        generationJob?.cancel()

        viewModelScope.launch {
            val stored = dao.conversation(id) ?: return@launch
            val messages = dao.messages(id)

            // The cache holds the tokens of whatever was on screen before, and
            // none of them belong to this conversation.
            session?.reset()
            conversationId = id

            _uiState.update {
                it.copy(
                    conversationId = id,
                    systemPrompt = stored.systemPrompt,
                    messages = messages.map { row ->
                        DisplayMessage(
                            id = row.id,
                            role = ChatMessage.Role.fromId(row.role),
                            content = row.content,
                            tokensPerSecond = row.tokensPerSecond,
                            tokenCount = row.tokenCount,
                        )
                    },
                    streamingText = "",
                    contextUsed = 0,
                    lastMetrics = null,
                    error = null,
                )
            }
        }
    }

    fun deleteConversation(id: Long) {
        viewModelScope.launch {
            dao.deleteConversation(id)
            if (conversationId == id) startNewConversation()
        }
    }

    fun setSystemPrompt(prompt: String) {
        _uiState.update { it.copy(systemPrompt = prompt) }
        // The system prompt sits at the head of every prompt, so changing it
        // invalidates the whole cache; a reset is honest about that.
        viewModelScope.launch { session?.reset() }
    }

    fun setParams(params: ChatParams) {
        _uiState.update { it.copy(params = params) }
    }

    fun send(text: String) {
        val trimmed = text.trim()
        if (trimmed.isEmpty()) return

        val active = session ?: return
        if (_uiState.value.isGenerating) return

        generationJob = viewModelScope.launch {
            // The conversation row is created here rather than when the transcript
            // was cleared, so an untouched screen never becomes a history entry.
            // Titled from this first message, which is what makes the list
            // navigable without opening every entry.
            if (conversationId == 0L) {
                val model = _uiState.value.selectedModel ?: return@launch
                conversationId = dao.insert(
                    ConversationEntity(
                        title = trimmed.take(60),
                        modelFileName = model.file.name,
                        systemPrompt = _uiState.value.systemPrompt,
                    ),
                )
                _uiState.update { it.copy(conversationId = conversationId) }
            }

            val userMessageId = dao.insert(
                MessageEntity(
                    conversationId = conversationId,
                    role = ChatMessage.Role.User.id,
                    content = trimmed,
                ),
            )

            _uiState.update {
                it.copy(
                    messages = it.messages + DisplayMessage(
                        id = userMessageId,
                        role = ChatMessage.Role.User,
                        content = trimmed,
                    ),
                    streamingText = "",
                    isGenerating = true,
                    error = null,
                )
            }

            val history = buildHistory()
            val systemTokens = _uiState.value.systemPrompt
                .takeIf { it.isNotBlank() }
                ?.let { active.countTokens(it) + SYSTEM_PROMPT_TOKEN_MARGIN }
                ?: 0

            var metrics: ChatMetrics? = null

            runCatching {
                active.generate(
                    messages = history,
                    params = _uiState.value.params,
                    systemPromptTokens = systemTokens,
                )
                    .mapNotNull { event ->
                        when (event) {
                            is ChatEvent.Token -> event.piece
                            is ChatEvent.Completed -> {
                                metrics = event.metrics
                                null
                            }
                        }
                    }
                    .coalesceToFrames()
                    .collect { chunk ->
                        _uiState.update {
                            it.copy(
                                streamingText = it.streamingText + chunk,
                                // Sampled per chunk rather than per token: the
                                // status changes on the order of seconds.
                                thermalNotice = governor.level().label,
                            )
                        }
                    }
            }.onFailure { t ->
                if (t !is kotlinx.coroutines.CancellationException) {
                    _uiState.update { it.copy(error = t.message ?: "generation failed") }
                }
            }

            finishTurn(metrics)
        }
    }

    fun stop() {
        generationJob?.cancel()
        generationJob = null
    }

    fun clearError() {
        _uiState.update { it.copy(error = null) }
    }

    /** Persists whatever was generated, including a reply cut short by [stop]. */
    private suspend fun finishTurn(metrics: ChatMetrics?) {
        val reply = _uiState.value.streamingText
        if (reply.isNotBlank()) {
            val id = dao.insert(
                MessageEntity(
                    conversationId = conversationId,
                    role = ChatMessage.Role.Assistant.id,
                    content = reply,
                    tokensPerSecond = metrics?.decodeTokensPerSecond ?: 0.0,
                    tokenCount = metrics?.generatedTokens ?: 0,
                ),
            )
            _uiState.update {
                it.copy(
                    messages = it.messages + DisplayMessage(
                        id = id,
                        role = ChatMessage.Role.Assistant,
                        content = reply,
                        tokensPerSecond = metrics?.decodeTokensPerSecond ?: 0.0,
                        tokenCount = metrics?.generatedTokens ?: 0,
                    ),
                )
            }
            dao.touch(conversationId)
        }

        _uiState.update {
            it.copy(
                streamingText = "",
                isGenerating = false,
                lastMetrics = metrics ?: it.lastMetrics,
                contextUsed = metrics?.cachedTokens ?: session?.cachedTokens ?: it.contextUsed,
                thermalNotice = "",
            )
        }
    }

    private fun buildHistory(): List<ChatMessage> = buildList {
        _uiState.value.systemPrompt.takeIf { it.isNotBlank() }?.let {
            add(ChatMessage(ChatMessage.Role.System, it))
        }
        _uiState.value.messages.forEach { message ->
            add(ChatMessage(message.role, message.content))
        }
    }

    private fun releaseEngine() {
        session?.close()
        session = null
        engine?.close()
        engine = null
    }

    override fun onCleared() {
        generationJob?.cancel()
        releaseEngine()
        super.onCleared()
    }

    private companion object {
        const val DEFAULT_CONTEXT = 4096

        /**
         * Slack added to the measured system-prompt length when telling the native
         * side how much to preserve during eviction. The formatted prompt wraps
         * the system text in template markup, so the raw token count understates
         * it — and under-reserving would evict part of the system prompt.
         */
        const val SYSTEM_PROMPT_TOKEN_MARGIN = 16
    }
}
