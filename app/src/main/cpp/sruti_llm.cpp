// JNI bridge over llama.cpp.
//
// Scope is deliberately narrow for Phase 0: load a GGUF, build a context, run a
// prefill + decode loop, and stream pieces back to Kotlin with enough timing
// detail to answer the "is this fast enough" question honestly.

#include <jni.h>
#include <android/log.h>
#include <dirent.h>

#include <chrono>
#include <cstring>
#include <string>
#include <vector>

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf_summary.h"
#include "utf8_assembler.h"

#define LOG_TAG "sruti-native"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

using clock_type = std::chrono::steady_clock;

int64_t micros_since(clock_type::time_point start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               clock_type::now() - start)
        .count();
}

// llama.cpp logs are verbose and land on stderr by default, which Android drops.
void forward_log(ggml_log_level level, const char * text, void * /*user_data*/) {
    if (text == nullptr) {
        return;
    }
    int prio = ANDROID_LOG_DEBUG;
    switch (level) {
        case GGML_LOG_LEVEL_ERROR: prio = ANDROID_LOG_ERROR; break;
        case GGML_LOG_LEVEL_WARN:  prio = ANDROID_LOG_WARN;  break;
        case GGML_LOG_LEVEL_INFO:  prio = ANDROID_LOG_INFO;  break;
        default: break;
    }
    __android_log_write(prio, LOG_TAG, text);
}

using sruti::Utf8Assembler;

std::string piece_for_token(const llama_vocab * vocab, llama_token token) {
    char buf[256];
    int32_t n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, /*special=*/false);
    if (n >= 0) {
        return std::string(buf, static_cast<size_t>(n));
    }
    // Negative means the buffer was too small; -n is the required size.
    std::vector<char> big(static_cast<size_t>(-n));
    n = llama_token_to_piece(vocab, token, big.data(),
                             static_cast<int32_t>(big.size()), 0, /*special=*/false);
    if (n < 0) {
        return {};
    }
    return std::string(big.data(), static_cast<size_t>(n));
}

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

}  // namespace

extern "C" {

/// Loads ggml's CPU backend variants and initialises llama.
///
/// `native_lib_dir` must be the application's nativeLibraryDir. ggml's own
/// discovery searches the executable's directory and the working directory, which
/// on Android are /system/bin and / — never where an APK's libraries live. Without
/// an explicit path no backend registers at all and model loading fails outright,
/// so this is not an optimisation.
JNIEXPORT void JNICALL
Java_dev_sruti_llm_LlamaBridge_nativeBackendInit(
    JNIEnv * env, jobject, jstring native_lib_dir) {

    llama_log_set(forward_log, nullptr);

    const std::string dir = jstring_to_utf8(env, native_lib_dir);
    if (!dir.empty()) {
        ggml_backend_load_all_from_path(dir.c_str());
    } else {
        ggml_backend_load_all();
    }

    // Fallback: load each variant by bare filename.
    //
    // The directory scan above only works when the libraries exist on disk. If an
    // APK is ever packaged with extractNativeLibs=false they stay inside it, the
    // directory is empty, and nothing registers -- which surfaces to the user as
    // "no compute backend is available on this device" with no further clue.
    // Android's linker resolves a bare soname out of the APK regardless, so
    // asking for them by name works either way.
    if (ggml_backend_reg_count() == 0) {
        LOGE("no backends found by scanning %s; falling back to loading by name",
             dir.c_str());

        // Best first: ggml keeps the highest-scoring one the CPU can execute, and
        // a variant the device cannot run simply fails to load.
        static const char * kVariants[] = {
            "libggml-cpu-android_armv9.2_2.so",
            "libggml-cpu-android_armv9.2_1.so",
            "libggml-cpu-android_armv9.0_1.so",
            "libggml-cpu-android_armv8.6_1.so",
            "libggml-cpu-android_armv8.2_2.so",
            "libggml-cpu-android_armv8.2_1.so",
            "libggml-cpu-android_armv8.0_1.so",
        };
        for (const char * name : kVariants) {
            if (ggml_backend_load(name) != nullptr) {
                LOGI("loaded backend %s by name", name);
            }
        }
    }

    llama_backend_init();

    LOGI("llama backend initialised, %zu ggml backend(s) registered from %s",
         static_cast<size_t>(ggml_backend_reg_count()),
         dir.empty() ? "(default paths)" : dir.c_str());
}

/// Diagnostic detail for when no backend loads.
///
/// Without this the failure is a bare sentence with nothing to act on; with it,
/// the library directory and its contents are visible, which is the difference
/// between guessing and knowing.
JNIEXPORT jstring JNICALL
Java_dev_sruti_llm_LlamaBridge_nativeBackendDiagnostics(
    JNIEnv * env, jobject, jstring native_lib_dir) {

    const std::string dir = jstring_to_utf8(env, native_lib_dir);
    std::string out = "backends registered: " +
                      std::to_string(ggml_backend_reg_count()) + "\n";
    out += "library dir: " + dir + "\n";

    DIR * d = ::opendir(dir.c_str());
    if (d == nullptr) {
        out += "  (cannot open directory)\n";
    } else {
        int count = 0;
        while (dirent * entry = ::readdir(d)) {
            const std::string name = entry->d_name;
            if (name == "." || name == "..") continue;
            out += "  " + name + "\n";
            ++count;
        }
        ::closedir(d);
        if (count == 0) {
            out += "  (empty — native libraries were not extracted from the APK)\n";
        }
    }
    return env->NewStringUTF(out.c_str());
}

/// Number of registered ggml backends. Zero means inference cannot work.
JNIEXPORT jint JNICALL
Java_dev_sruti_llm_LlamaBridge_nativeBackendCount(JNIEnv *, jobject) {
    return static_cast<jint>(ggml_backend_reg_count());
}

JNIEXPORT void JNICALL
Java_dev_sruti_llm_LlamaBridge_nativeBackendFree(JNIEnv *, jobject) {
    llama_backend_free();
}

JNIEXPORT jlong JNICALL
Java_dev_sruti_llm_LlamaBridge_nativeLoadModel(
    JNIEnv * env, jobject, jstring path, jint n_gpu_layers) {

    const std::string model_path = jstring_to_utf8(env, path);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    // mmap is what keeps a 1-2B model off the Java heap and lets the kernel page
    // weights in on demand, so it stays on (the default).

    const auto t0 = clock_type::now();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (model == nullptr) {
        throw_illegal_state(env, ("failed to load model: " + model_path).c_str());
        return 0;
    }
    LOGI("model loaded in %lld ms", static_cast<long long>(micros_since(t0) / 1000));

    return reinterpret_cast<jlong>(model);
}

JNIEXPORT void JNICALL
Java_dev_sruti_llm_LlamaBridge_nativeFreeModel(JNIEnv *, jobject, jlong handle) {
    if (handle != 0) {
        llama_model_free(reinterpret_cast<llama_model *>(handle));
    }
}

JNIEXPORT jstring JNICALL
Java_dev_sruti_llm_LlamaBridge_nativeModelDesc(JNIEnv * env, jobject, jlong handle) {
    if (handle == 0) {
        return env->NewStringUTF("");
    }
    auto * model = reinterpret_cast<llama_model *>(handle);
    char buf[256];
    llama_model_desc(model, buf, sizeof(buf));
    return env->NewStringUTF(buf);
}

JNIEXPORT jlong JNICALL
Java_dev_sruti_llm_LlamaBridge_nativeModelParamCount(JNIEnv *, jobject, jlong handle) {
    if (handle == 0) {
        return 0;
    }
    return static_cast<jlong>(
        llama_model_n_params(reinterpret_cast<llama_model *>(handle)));
}

JNIEXPORT jlong JNICALL
Java_dev_sruti_llm_LlamaBridge_nativeModelSizeBytes(JNIEnv *, jobject, jlong handle) {
    if (handle == 0) {
        return 0;
    }
    return static_cast<jlong>(
        llama_model_size(reinterpret_cast<llama_model *>(handle)));
}

JNIEXPORT jint JNICALL
Java_dev_sruti_llm_LlamaBridge_nativeModelCtxTrain(JNIEnv *, jobject, jlong handle) {
    if (handle == 0) {
        return 0;
    }
    return llama_model_n_ctx_train(reinterpret_cast<llama_model *>(handle));
}

JNIEXPORT jlong JNICALL
Java_dev_sruti_llm_LlamaBridge_nativeNewContext(
    JNIEnv * env, jobject, jlong model_handle,
    jint n_ctx, jint n_threads, jint n_batch) {

    if (model_handle == 0) {
        throw_illegal_state(env, "null model handle");
        return 0;
    }
    auto * model = reinterpret_cast<llama_model *>(model_handle);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = static_cast<uint32_t>(n_ctx);
    cparams.n_batch         = static_cast<uint32_t>(n_batch);
    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.no_perf         = false;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        throw_illegal_state(env, "failed to create llama context");
        return 0;
    }
    return reinterpret_cast<jlong>(ctx);
}

JNIEXPORT void JNICALL
Java_dev_sruti_llm_LlamaBridge_nativeFreeContext(JNIEnv *, jobject, jlong handle) {
    if (handle != 0) {
        llama_free(reinterpret_cast<llama_context *>(handle));
    }
}

JNIEXPORT void JNICALL
Java_dev_sruti_llm_LlamaBridge_nativeResetContext(JNIEnv *, jobject, jlong handle) {
    if (handle != 0) {
        auto * ctx = reinterpret_cast<llama_context *>(handle);
        llama_memory_clear(llama_get_memory(ctx), /*data=*/true);
    }
}

JNIEXPORT jstring JNICALL
Java_dev_sruti_llm_LlamaBridge_nativeGgufSummary(JNIEnv * env, jobject, jstring path) {
    const char * c_path = env->GetStringUTFChars(path, nullptr);
    const std::string summary = sruti::gguf_summary(c_path);
    env->ReleaseStringUTFChars(path, c_path);
    return env->NewStringUTF(summary.c_str());
}

JNIEXPORT jintArray JNICALL
Java_dev_sruti_llm_LlamaBridge_nativeTokenize(
    JNIEnv * env, jobject, jlong model_handle, jstring text, jboolean add_special) {

    if (model_handle == 0) {
        throw_illegal_state(env, "null model handle");
        return nullptr;
    }
    auto * model = reinterpret_cast<llama_model *>(model_handle);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const std::string input = jstring_to_utf8(env, text);

    // A negative return is the required capacity; size generously, then retry.
    std::vector<llama_token> tokens(input.size() + 16);
    int32_t n = llama_tokenize(vocab, input.c_str(), static_cast<int32_t>(input.size()),
                               tokens.data(), static_cast<int32_t>(tokens.size()),
                               add_special == JNI_TRUE, /*parse_special=*/true);
    if (n < 0) {
        tokens.resize(static_cast<size_t>(-n));
        n = llama_tokenize(vocab, input.c_str(), static_cast<int32_t>(input.size()),
                           tokens.data(), static_cast<int32_t>(tokens.size()),
                           add_special == JNI_TRUE, /*parse_special=*/true);
        if (n < 0) {
            throw_illegal_state(env, "tokenization failed");
            return nullptr;
        }
    }

    jintArray out = env->NewIntArray(n);
    if (out == nullptr) {
        return nullptr;
    }
    static_assert(sizeof(llama_token) == sizeof(jint), "llama_token must be 32-bit");
    env->SetIntArrayRegion(out, 0, n, reinterpret_cast<const jint *>(tokens.data()));
    return out;
}

// Runs prefill then a decode loop, streaming decoded text to `callback`.
//
// Returns [promptTokens, generatedTokens, prefillMicros, decodeMicros, stopReason]
// where stopReason is 0=eog, 1=hit maxTokens, 2=cancelled by callback, 3=context full.
JNIEXPORT jlongArray JNICALL
Java_dev_sruti_llm_LlamaBridge_nativeGenerate(
    JNIEnv * env, jobject, jlong model_handle, jlong ctx_handle,
    jstring prompt, jint max_tokens,
    jfloat temperature, jint top_k, jfloat top_p, jfloat min_p, jint seed,
    jobject callback) {

    if (model_handle == 0 || ctx_handle == 0) {
        throw_illegal_state(env, "null model or context handle");
        return nullptr;
    }

    auto * model = reinterpret_cast<llama_model *>(model_handle);
    auto * ctx   = reinterpret_cast<llama_context *>(ctx_handle);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Resolve against the interface rather than the object's own class. The
    // callback is a Kotlin lambda, so its runtime class is a synthetic that R8 is
    // free to rename; the interface is pinned by a keep rule in proguard-rules.pro.
    jclass cb_cls = env->FindClass("dev/sruti/llm/LlamaBridge$TokenCallback");
    if (cb_cls == nullptr) {
        return nullptr;  // NoClassDefFoundError already pending
    }
    jmethodID on_token = env->GetMethodID(cb_cls, "onToken", "(Ljava/lang/String;)Z");
    env->DeleteLocalRef(cb_cls);
    if (on_token == nullptr) {
        throw_illegal_state(env, "callback is missing onToken(String):boolean");
        return nullptr;
    }

    // --- tokenize -----------------------------------------------------------
    const std::string prompt_str = jstring_to_utf8(env, prompt);
    std::vector<llama_token> tokens(prompt_str.size() + 16);
    int32_t n_prompt = llama_tokenize(
        vocab, prompt_str.c_str(), static_cast<int32_t>(prompt_str.size()),
        tokens.data(), static_cast<int32_t>(tokens.size()),
        /*add_special=*/true, /*parse_special=*/true);
    if (n_prompt < 0) {
        tokens.resize(static_cast<size_t>(-n_prompt));
        n_prompt = llama_tokenize(
            vocab, prompt_str.c_str(), static_cast<int32_t>(prompt_str.size()),
            tokens.data(), static_cast<int32_t>(tokens.size()),
            /*add_special=*/true, /*parse_special=*/true);
    }
    if (n_prompt <= 0) {
        throw_illegal_state(env, "prompt tokenization failed");
        return nullptr;
    }
    tokens.resize(static_cast<size_t>(n_prompt));

    const auto n_ctx = static_cast<int32_t>(llama_n_ctx(ctx));
    if (n_prompt >= n_ctx) {
        throw_illegal_state(env, "prompt is longer than the context window");
        return nullptr;
    }

    // --- sampler chain ------------------------------------------------------
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

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

    int64_t prefill_us = 0;
    int64_t decode_us  = 0;
    int32_t n_generated = 0;
    int64_t stop_reason = 0;

    // --- prefill ------------------------------------------------------------
    {
        const auto t0 = clock_type::now();
        llama_batch batch = llama_batch_get_one(tokens.data(), n_prompt);
        if (llama_decode(ctx, batch) != 0) {
            llama_sampler_free(smpl);
            throw_illegal_state(env, "llama_decode failed during prefill");
            return nullptr;
        }
        prefill_us = micros_since(t0);
    }

    // --- decode loop --------------------------------------------------------
    Utf8Assembler assembler;
    int32_t n_past = n_prompt;
    const auto t_decode = clock_type::now();

    for (int32_t i = 0; i < max_tokens; ++i) {
        const llama_token id = llama_sampler_sample(smpl, ctx, -1);

        if (llama_vocab_is_eog(vocab, id)) {
            stop_reason = 0;
            break;
        }

        // llama_sampler_sample has already accepted this token into the chain.
        // A second accept advances stateful samplers twice over.
        ++n_generated;

        const std::string piece = piece_for_token(vocab, id);
        const std::string ready = assembler.push(piece.data(), piece.size());
        if (!ready.empty()) {
            jstring jpiece = env->NewStringUTF(ready.c_str());
            const jboolean keep_going = env->CallBooleanMethod(callback, on_token, jpiece);
            env->DeleteLocalRef(jpiece);

            if (env->ExceptionCheck()) {
                // Let the Kotlin-side exception propagate rather than masking it.
                llama_sampler_free(smpl);
                return nullptr;
            }
            if (keep_going == JNI_FALSE) {
                stop_reason = 2;
                break;
            }
        }

        if (i == max_tokens - 1) {
            // Nothing will be sampled after this, so skip the forward pass that
            // would only produce logits no one reads. Running it would also
            // inflate the measured decode time by a full step.
            stop_reason = 1;
            break;
        }

        if (n_past + 1 >= n_ctx) {
            stop_reason = 3;
            break;
        }

        llama_token next = id;
        llama_batch batch = llama_batch_get_one(&next, 1);
        if (llama_decode(ctx, batch) != 0) {
            llama_sampler_free(smpl);
            throw_illegal_state(env, "llama_decode failed during generation");
            return nullptr;
        }
        ++n_past;
    }

    decode_us = micros_since(t_decode);

    // Emit any bytes still held back by the UTF-8 assembler.
    const std::string tail = assembler.flush();
    if (!tail.empty()) {
        jstring jpiece = env->NewStringUTF(tail.c_str());
        env->CallBooleanMethod(callback, on_token, jpiece);
        env->DeleteLocalRef(jpiece);
        env->ExceptionClear();
    }

    llama_sampler_free(smpl);

    jlongArray out = env->NewLongArray(5);
    if (out == nullptr) {
        return nullptr;
    }
    const jlong metrics[5] = {
        static_cast<jlong>(n_prompt),
        static_cast<jlong>(n_generated),
        static_cast<jlong>(prefill_us),
        static_cast<jlong>(decode_us),
        static_cast<jlong>(stop_reason),
    };
    env->SetLongArrayRegion(out, 0, 5, metrics);
    return out;
}

}  // extern "C"
