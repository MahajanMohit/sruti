// JNI bridge for multi-turn chat.
//
// Distinct from sruti_llm.cpp's one-shot generate because chat needs state that
// survives between calls: the KV cache and the token sequence that produced it.
//
// The native side owns the token history rather than taking a position from
// Kotlin. That matters because the cache can be evicted mid-conversation when the
// context fills — after which any position Kotlin believed in is wrong. Passing
// the full sequence every turn and diffing it here keeps the two honest, and is
// what makes "reuse the cache" safe rather than merely fast.

#include <jni.h>
#include <android/log.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "llama.h"
#include "utf8_assembler.h"

#define LOG_TAG "sruti-chat"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

namespace {

using clock_type = std::chrono::steady_clock;
using sruti::Utf8Assembler;

int64_t micros_since(clock_type::time_point start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               clock_type::now() - start)
        .count();
}

/// Tokens already present in the context, in order.
struct ChatSession {
    std::vector<llama_token> cached;
};

std::string jstring_to_utf8(JNIEnv * env, jstring s) {
    if (s == nullptr) {
        return {};
    }
    const char * chars = env->GetStringUTFChars(s, nullptr);
    std::string out = chars == nullptr ? std::string() : std::string(chars);
    if (chars != nullptr) {
        env->ReleaseStringUTFChars(s, chars);
    }
    return out;
}

void throw_illegal_state(JNIEnv * env, const char * msg) {
    jclass cls = env->FindClass("java/lang/IllegalStateException");
    if (cls != nullptr) {
        env->ThrowNew(cls, msg);
    }
}

std::string piece_for_token(const llama_vocab * vocab, llama_token token) {
    char buf[256];
    int32_t n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, /*special=*/false);
    if (n >= 0) {
        return std::string(buf, static_cast<size_t>(n));
    }
    std::vector<char> big(static_cast<size_t>(-n));
    n = llama_token_to_piece(vocab, token, big.data(),
                             static_cast<int32_t>(big.size()), 0, /*special=*/false);
    return n < 0 ? std::string() : std::string(big.data(), static_cast<size_t>(n));
}

/// Length of the shared prefix between the cache and the new sequence.
size_t common_prefix(const std::vector<llama_token> & a, const std::vector<llama_token> & b) {
    const size_t limit = std::min(a.size(), b.size());
    size_t n = 0;
    while (n < limit && a[n] == b[n]) {
        ++n;
    }
    return n;
}

}  // namespace

extern "C" {

// --- session lifecycle ------------------------------------------------------

JNIEXPORT jlong JNICALL
Java_dev_sruti_llm_ChatBridge_nativeNewSession(JNIEnv *, jobject) {
    return reinterpret_cast<jlong>(new ChatSession());
}

JNIEXPORT void JNICALL
Java_dev_sruti_llm_ChatBridge_nativeFreeSession(JNIEnv *, jobject, jlong handle) {
    delete reinterpret_cast<ChatSession *>(handle);
}

JNIEXPORT void JNICALL
Java_dev_sruti_llm_ChatBridge_nativeResetSession(
    JNIEnv *, jobject, jlong ctx_handle, jlong session_handle) {

    if (session_handle != 0) {
        reinterpret_cast<ChatSession *>(session_handle)->cached.clear();
    }
    if (ctx_handle != 0) {
        auto * ctx = reinterpret_cast<llama_context *>(ctx_handle);
        llama_memory_clear(llama_get_memory(ctx), /*data=*/true);
    }
}

JNIEXPORT jint JNICALL
Java_dev_sruti_llm_ChatBridge_nativeCachedTokens(JNIEnv *, jobject, jlong session_handle) {
    if (session_handle == 0) {
        return 0;
    }
    return static_cast<jint>(reinterpret_cast<ChatSession *>(session_handle)->cached.size());
}

// --- chat template ----------------------------------------------------------

/// The model's own chat template, or null when it declares none.
JNIEXPORT jstring JNICALL
Java_dev_sruti_llm_ChatBridge_nativeChatTemplate(JNIEnv * env, jobject, jlong model_handle) {
    if (model_handle == 0) {
        return nullptr;
    }
    auto * model = reinterpret_cast<llama_model *>(model_handle);
    const char * tmpl = llama_model_chat_template(model, /*name=*/nullptr);
    return tmpl == nullptr ? nullptr : env->NewStringUTF(tmpl);
}

/// Formats a conversation with the model's template.
///
/// Instruct models are trained with a specific turn structure; feeding them raw
/// concatenated text produces markedly worse output than the same model prompted
/// correctly. Returns null when the template cannot be applied, so the caller can
/// fall back rather than silently mis-prompting.
JNIEXPORT jstring JNICALL
Java_dev_sruti_llm_ChatBridge_nativeApplyTemplate(
    JNIEnv * env, jobject, jlong model_handle,
    jobjectArray roles, jobjectArray contents, jboolean add_assistant) {

    if (model_handle == 0 || roles == nullptr || contents == nullptr) {
        return nullptr;
    }

    auto * model = reinterpret_cast<llama_model *>(model_handle);
    const char * tmpl = llama_model_chat_template(model, nullptr);
    if (tmpl == nullptr) {
        return nullptr;
    }

    const jsize count = env->GetArrayLength(roles);
    if (env->GetArrayLength(contents) != count) {
        throw_illegal_state(env, "roles and contents must have the same length");
        return nullptr;
    }

    // Backing strings must outlive the llama_chat_message array that points at them.
    std::vector<std::string> role_storage;
    std::vector<std::string> content_storage;
    role_storage.reserve(static_cast<size_t>(count));
    content_storage.reserve(static_cast<size_t>(count));

    for (jsize i = 0; i < count; ++i) {
        auto role = reinterpret_cast<jstring>(env->GetObjectArrayElement(roles, i));
        auto content = reinterpret_cast<jstring>(env->GetObjectArrayElement(contents, i));
        role_storage.push_back(jstring_to_utf8(env, role));
        content_storage.push_back(jstring_to_utf8(env, content));
        env->DeleteLocalRef(role);
        env->DeleteLocalRef(content);
    }

    std::vector<llama_chat_message> messages(static_cast<size_t>(count));
    size_t total_chars = 0;
    for (jsize i = 0; i < count; ++i) {
        messages[i].role = role_storage[i].c_str();
        messages[i].content = content_storage[i].c_str();
        total_chars += role_storage[i].size() + content_storage[i].size();
    }

    // The documented starting size; a template that expands more than this
    // reports the size it needs and we retry once.
    std::vector<char> buf(total_chars * 2 + 1024);
    int32_t written = llama_chat_apply_template(
        tmpl, messages.data(), messages.size(), add_assistant == JNI_TRUE,
        buf.data(), static_cast<int32_t>(buf.size()));

    if (written > static_cast<int32_t>(buf.size())) {
        buf.resize(static_cast<size_t>(written) + 1);
        written = llama_chat_apply_template(
            tmpl, messages.data(), messages.size(), add_assistant == JNI_TRUE,
            buf.data(), static_cast<int32_t>(buf.size()));
    }
    if (written < 0) {
        return nullptr;
    }

    return env->NewStringUTF(std::string(buf.data(), static_cast<size_t>(written)).c_str());
}

// --- generation -------------------------------------------------------------

/// Generates a reply, reusing whatever of the KV cache still applies.
///
/// `tokens` is the entire prompt for this turn, not a delta. The shared prefix
/// with the cache is computed here and only the remainder is prefilled, which is
/// what makes a long conversation's second turn fast.
///
/// Returns [promptTokens, reusedTokens, generatedTokens, prefillUs, decodeUs,
/// stopReason, cachedTokens]. stopReason: 0=eog, 1=maxTokens, 2=cancelled,
/// 3=contextFull.
JNIEXPORT jlongArray JNICALL
Java_dev_sruti_llm_ChatBridge_nativeGenerate(
    JNIEnv * env, jobject,
    jlong model_handle, jlong ctx_handle, jlong session_handle,
    jintArray tokens, jint n_keep, jint max_tokens,
    jfloat temperature, jint top_k, jfloat top_p, jfloat min_p,
    jfloat repeat_penalty, jint repeat_last_n, jint seed,
    jobject callback) {

    if (model_handle == 0 || ctx_handle == 0 || session_handle == 0) {
        throw_illegal_state(env, "null model, context or session handle");
        return nullptr;
    }

    auto * model = reinterpret_cast<llama_model *>(model_handle);
    auto * ctx = reinterpret_cast<llama_context *>(ctx_handle);
    auto * session = reinterpret_cast<ChatSession *>(session_handle);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_memory_t mem = llama_get_memory(ctx);

    jclass cb_cls = env->FindClass("dev/sruti/llm/ChatBridge$ChatCallback");
    if (cb_cls == nullptr) {
        return nullptr;
    }
    const jmethodID on_token = env->GetMethodID(cb_cls, "onToken", "(Ljava/lang/String;)Z");
    const jmethodID pacing = env->GetMethodID(cb_cls, "pacingMicros", "()J");
    env->DeleteLocalRef(cb_cls);
    if (on_token == nullptr || pacing == nullptr) {
        throw_illegal_state(env, "callback is missing onToken or pacingMicros");
        return nullptr;
    }

    // --- copy the prompt out of the JVM ------------------------------------
    const jsize n_input = env->GetArrayLength(tokens);
    std::vector<llama_token> prompt(static_cast<size_t>(n_input));
    if (n_input > 0) {
        static_assert(sizeof(llama_token) == sizeof(jint), "llama_token must be 32-bit");
        env->GetIntArrayRegion(tokens, 0, n_input, reinterpret_cast<jint *>(prompt.data()));
    }
    if (prompt.empty()) {
        throw_illegal_state(env, "empty prompt");
        return nullptr;
    }

    const auto n_ctx = static_cast<int32_t>(llama_n_ctx(ctx));

    // --- make room ----------------------------------------------------------
    // A conversation that outgrows the window has to lose something. Dropping the
    // oldest turns while keeping `n_keep` (the system prompt) is the least-bad
    // option: the alternative is refusing to continue.
    if (static_cast<int32_t>(prompt.size()) + max_tokens > n_ctx) {
        const int32_t keep = std::max(0, std::min(n_keep, static_cast<int32_t>(prompt.size())));
        const int32_t budget = n_ctx - max_tokens;
        if (budget <= keep) {
            throw_illegal_state(env, "context window is too small for this request");
            return nullptr;
        }

        // Keep the head and the most recent tail; the middle is what goes.
        const int32_t tail = budget - keep;
        std::vector<llama_token> trimmed;
        trimmed.reserve(static_cast<size_t>(budget));
        trimmed.insert(trimmed.end(), prompt.begin(), prompt.begin() + keep);
        trimmed.insert(trimmed.end(), prompt.end() - tail, prompt.end());
        prompt.swap(trimmed);
    }

    // --- reuse what still matches ------------------------------------------
    const size_t reused = common_prefix(session->cached, prompt);

    if (reused < session->cached.size()) {
        // Evict everything after the divergence; those positions describe tokens
        // that are no longer in the prompt.
        llama_memory_seq_rm(mem, 0, static_cast<llama_pos>(reused), -1);
        session->cached.resize(reused);
    }

    const size_t n_new = prompt.size() - reused;
    int64_t prefill_us = 0;

    if (n_new > 0) {
        const auto t0 = clock_type::now();

        // Submit in chunks so a long first turn does not exceed n_batch.
        constexpr int32_t kChunk = 512;
        for (size_t offset = 0; offset < n_new; offset += kChunk) {
            const auto count =
                static_cast<int32_t>(std::min<size_t>(kChunk, n_new - offset));
            llama_batch batch =
                llama_batch_get_one(prompt.data() + reused + offset, count);
            if (llama_decode(ctx, batch) != 0) {
                throw_illegal_state(env, "llama_decode failed during prefill");
                return nullptr;
            }
        }
        prefill_us = micros_since(t0);

        session->cached.insert(session->cached.end(), prompt.begin() + reused, prompt.end());
    }

    // --- sampler ------------------------------------------------------------
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    if (repeat_penalty > 1.0f && repeat_last_n > 0) {
        llama_sampler_chain_add(
            smpl, llama_sampler_init_penalties(repeat_last_n, repeat_penalty, 0.0f, 0.0f));
    }
    if (temperature <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        if (top_k > 0) {
            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
        }
        if (top_p > 0.0f && top_p < 1.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
        }
        if (min_p > 0.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_min_p(min_p, 1));
        }
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(static_cast<uint32_t>(seed)));
    }

    // --- decode -------------------------------------------------------------
    Utf8Assembler assembler;
    int32_t n_generated = 0;
    int64_t stop_reason = 0;
    const auto t_decode = clock_type::now();

    for (int32_t i = 0; i < max_tokens; ++i) {
        const llama_token id = llama_sampler_sample(smpl, ctx, -1);

        if (llama_vocab_is_eog(vocab, id)) {
            stop_reason = 0;
            break;
        }

        llama_sampler_accept(smpl, id);
        ++n_generated;
        session->cached.push_back(id);

        const std::string piece = piece_for_token(vocab, id);
        const std::string ready = assembler.push(piece.data(), piece.size());
        if (!ready.empty()) {
            jstring jpiece = env->NewStringUTF(ready.c_str());
            const jboolean keep_going = env->CallBooleanMethod(callback, on_token, jpiece);
            env->DeleteLocalRef(jpiece);

            if (env->ExceptionCheck()) {
                llama_sampler_free(smpl);
                return nullptr;
            }
            if (keep_going == JNI_FALSE) {
                stop_reason = 2;
                break;
            }
        }

        if (i == max_tokens - 1) {
            stop_reason = 1;
            break;
        }
        if (static_cast<int32_t>(session->cached.size()) + 1 >= n_ctx) {
            stop_reason = 3;
            break;
        }

        // The thermal governor throttles here rather than after the fact. Slowing
        // generation deliberately keeps the SoC below the point where it sheds
        // clock abruptly, which is both faster on average and less jarring.
        const jlong sleep_us = env->CallLongMethod(callback, pacing);
        if (env->ExceptionCheck()) {
            llama_sampler_free(smpl);
            return nullptr;
        }
        if (sleep_us > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
        }

        llama_token next = id;
        llama_batch batch = llama_batch_get_one(&next, 1);
        if (llama_decode(ctx, batch) != 0) {
            llama_sampler_free(smpl);
            throw_illegal_state(env, "llama_decode failed during generation");
            return nullptr;
        }
    }

    const int64_t decode_us = micros_since(t_decode);

    const std::string tail = assembler.flush();
    if (!tail.empty()) {
        jstring jpiece = env->NewStringUTF(tail.c_str());
        env->CallBooleanMethod(callback, on_token, jpiece);
        env->DeleteLocalRef(jpiece);
        env->ExceptionClear();
    }

    llama_sampler_free(smpl);

    jlongArray out = env->NewLongArray(7);
    if (out == nullptr) {
        return nullptr;
    }
    const jlong metrics[7] = {
        static_cast<jlong>(prompt.size()),
        static_cast<jlong>(reused),
        static_cast<jlong>(n_generated),
        static_cast<jlong>(prefill_us),
        static_cast<jlong>(decode_us),
        static_cast<jlong>(stop_reason),
        static_cast<jlong>(session->cached.size()),
    };
    env->SetLongArrayRegion(out, 0, 7, metrics);
    return out;
}

}  // extern "C"
