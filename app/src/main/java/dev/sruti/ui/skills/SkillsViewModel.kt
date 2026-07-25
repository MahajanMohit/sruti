package dev.sruti.ui.skills

import android.net.Uri
import androidx.compose.runtime.Immutable
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dagger.hilt.android.lifecycle.HiltViewModel
import dev.sruti.agent.Skill
import dev.sruti.agent.SkillStore
import dev.sruti.agent.ToolRegistry
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

@Immutable
data class SkillsUiState(
    val skills: List<Skill> = emptyList(),
    val isLoading: Boolean = true,
    val error: String? = null,
    val importing: Boolean = false,
)

@Immutable
data class SkillEditorState(
    val text: String = TEMPLATE,
    val originalName: String? = null,
    val error: String? = null,
    val saved: Boolean = false,
) {
    companion object {
        /**
         * What a new skill starts as.
         *
         * A filled-in example rather than an empty field: the format is
         * unfamiliar, and editing something that already works is a much lower
         * bar than writing one from a blank page on a phone.
         */
        const val TEMPLATE = """---
name: my-skill
description: What this does, in one line
keywords: words, that, appear, in, requests
parameters:
  - path: What this value is
steps:
  - read_file(path={{path}})
---

Any prose here is given to the model as extra guidance.
"""
    }
}

@HiltViewModel
class SkillsViewModel @Inject constructor(
    private val store: SkillStore,
    private val registry: ToolRegistry,
) : ViewModel() {

    private val _state = MutableStateFlow(SkillsUiState())
    val state: StateFlow<SkillsUiState> = _state.asStateFlow()

    private val _editor = MutableStateFlow(SkillEditorState())
    val editor: StateFlow<SkillEditorState> = _editor.asStateFlow()

    init {
        refresh()
    }

    fun refresh() {
        viewModelScope.launch {
            _state.update { it.copy(isLoading = true) }
            _state.update { it.copy(skills = store.all(), isLoading = false) }
        }
    }

    /** Tools a skill names that this device does not have — the shell, usually. */
    fun missingTools(skill: Skill): List<String> = skill.missingTools(registry)

    // --- editing --------------------------------------------------------------

    fun startNew() {
        _editor.value = SkillEditorState()
    }

    fun startEditing(skill: Skill) {
        viewModelScope.launch {
            val text = store.rawText(skill.name) ?: return@launch
            _editor.value = SkillEditorState(text = text, originalName = skill.name)
        }
    }

    fun onEditorTextChanged(text: String) {
        // Clearing the error as soon as the text changes keeps a stale complaint
        // from sitting under a line the user has already fixed.
        _editor.update { it.copy(text = text, error = null, saved = false) }
    }

    fun save() {
        viewModelScope.launch {
            val current = _editor.value
            runCatching { store.save(current.text) }
                .onSuccess { saved ->
                    // Renaming writes a new file; the old one would otherwise
                    // remain as a duplicate under its previous name.
                    val previous = current.originalName
                    if (previous != null && previous != saved.name) store.delete(previous)

                    _editor.update { it.copy(saved = true, error = null, originalName = saved.name) }
                    refresh()
                }
                .onFailure { t ->
                    _editor.update { it.copy(error = t.message ?: "that skill could not be read") }
                }
        }
    }

    fun delete(skill: Skill) {
        viewModelScope.launch {
            store.delete(skill.name)
            refresh()
        }
    }

    // --- importing ------------------------------------------------------------

    fun importFrom(uri: Uri) = importWith { store.importFrom(uri) }

    fun importFrom(url: String) = importWith { store.importFrom(url.trim()) }

    private fun importWith(block: suspend () -> Skill) {
        viewModelScope.launch {
            _state.update { it.copy(importing = true, error = null) }
            runCatching { block() }
                .onSuccess {
                    _state.update { it.copy(importing = false) }
                    refresh()
                }
                .onFailure { t ->
                    _state.update {
                        it.copy(importing = false, error = t.message ?: "import failed")
                    }
                }
        }
    }

    fun clearError() {
        _state.update { it.copy(error = null) }
    }
}
