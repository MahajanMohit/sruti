package dev.sruti.settings

import android.content.Context
import androidx.datastore.preferences.core.edit
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

    suspend fun currentToken(): String? = huggingFaceToken.first()

    suspend fun setHuggingFaceToken(token: String?) {
        context.dataStore.edit { prefs ->
            if (token.isNullOrBlank()) prefs.remove(TOKEN_KEY) else prefs[TOKEN_KEY] = token.trim()
        }
    }

    suspend fun setDefaultQuantType(quantType: QuantType) {
        context.dataStore.edit { it[QUANT_KEY] = quantType.id }
    }

    private companion object {
        val TOKEN_KEY = stringPreferencesKey("hf_token")
        val QUANT_KEY = stringPreferencesKey("default_quant")
    }
}
