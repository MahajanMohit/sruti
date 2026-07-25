package dev.sruti.settings

import android.content.Context
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import dev.sruti.convert.QuantType
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map

private val Context.dataStore by preferencesDataStore(name = "sruti_settings")

/**
 * User settings.
 *
 * The Hugging Face token is the one that matters: gated repositories — Llama 3.x
 * among them — return 401 without it, and that is the most likely first thing a
 * user hits when browsing for a model to run.
 */
class SettingsStore(private val context: Context) {

    val huggingFaceToken: Flow<String?> =
        context.dataStore.data.map { it[TOKEN_KEY]?.takeIf(String::isNotBlank) }

    val defaultQuantType: Flow<QuantType> =
        context.dataStore.data.map { QuantType.fromId(it[QUANT_KEY] ?: QuantType.Q4_K_M.id) }

    /**
     * Transformer layers to offload to the GPU. Zero keeps everything on the CPU.
     *
     * Defaults to zero deliberately. A mobile GPU shares memory bandwidth with
     * the CPU, and for a 1–2B model it frequently loses — so this is exposed as a
     * measurable choice rather than switched on as an assumption. The benchmark
     * is the thing that should decide it.
     */
    val gpuLayers: Flow<Int> = context.dataStore.data.map { it[GPU_LAYERS_KEY] ?: 0 }

    /**
     * Whether to keep the downloaded safetensors after converting.
     *
     * Off by default because a checkpoint is several times the size of what it
     * converts to. On, re-converting at a different quantization costs only the
     * conversion — the gigabytes are already on disk.
     */
    /**
     * Whether the agent may run shell commands through Termux.
     *
     * Off until the user turns it on, and separate from whether Termux is
     * installed: having the app available is not consent to let a language model
     * drive it.
     */
    val shellEnabled: Flow<Boolean> =
        context.dataStore.data.map { it[SHELL_ENABLED_KEY] ?: false }

    val keepCheckpoints: Flow<Boolean> =
        context.dataStore.data.map { it[KEEP_CHECKPOINTS_KEY] ?: false }

    suspend fun currentToken(): String? = huggingFaceToken.first()

    suspend fun setHuggingFaceToken(token: String?) {
        context.dataStore.edit { prefs ->
            if (token.isNullOrBlank()) prefs.remove(TOKEN_KEY) else prefs[TOKEN_KEY] = token.trim()
        }
    }

    suspend fun setDefaultQuantType(quantType: QuantType) {
        context.dataStore.edit { it[QUANT_KEY] = quantType.id }
    }

    suspend fun setGpuLayers(layers: Int) {
        context.dataStore.edit { it[GPU_LAYERS_KEY] = layers.coerceAtLeast(0) }
    }

    suspend fun currentGpuLayers(): Int = gpuLayers.first()

    suspend fun setShellEnabled(enabled: Boolean) {
        context.dataStore.edit { it[SHELL_ENABLED_KEY] = enabled }
    }

    suspend fun setKeepCheckpoints(keep: Boolean) {
        context.dataStore.edit { it[KEEP_CHECKPOINTS_KEY] = keep }
    }

    suspend fun currentKeepCheckpoints(): Boolean = keepCheckpoints.first()

    suspend fun currentShellEnabled(): Boolean = shellEnabled.first()

    private companion object {
        val TOKEN_KEY = stringPreferencesKey("hf_token")
        val QUANT_KEY = stringPreferencesKey("default_quant")
        val GPU_LAYERS_KEY = intPreferencesKey("gpu_layers")
        val KEEP_CHECKPOINTS_KEY = booleanPreferencesKey("keep_checkpoints")
        val SHELL_ENABLED_KEY = booleanPreferencesKey("shell_enabled")
    }
}
