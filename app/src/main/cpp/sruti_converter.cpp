// JNI bridge over the safetensors -> GGUF converter.
//
// Conversion is long-running and reports progress, so the callback is invoked on
// the same thread that called in. Kotlin runs the whole thing on a background
// dispatcher inside a foreground service.

#include <jni.h>
#include <android/log.h>

#include <string>
#include <vector>

#include "converter.h"

#define LOG_TAG "sruti-convert"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

namespace {

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

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_dev_sruti_convert_ConverterBridge_nativeEstimatedWorkingBytes(
    JNIEnv * env, jobject, jlong checkpoint_bytes, jstring quant_type) {

    return static_cast<jlong>(sruti::estimated_working_bytes(
        static_cast<uint64_t>(checkpoint_bytes), jstring_to_utf8(env, quant_type)));
}

/// Returns null on success, or an error message.
///
/// Progress and warnings are delivered through `callback`; returning false from
/// onProgress cancels the conversion.
JNIEXPORT jstring JNICALL
Java_dev_sruti_convert_ConverterBridge_nativeConvert(
    JNIEnv * env, jobject,
    jstring model_dir, jstring out_path, jstring quant_type,
    jstring model_name, jint n_threads, jobject callback) {

    sruti::ConvertOptions options;
    options.model_dir = jstring_to_utf8(env, model_dir);
    options.out_path = jstring_to_utf8(env, out_path);
    options.quant_type = jstring_to_utf8(env, quant_type);
    options.model_name = jstring_to_utf8(env, model_name);
    options.n_threads = n_threads;

    // Resolve against the interface: the callback is a Kotlin object whose runtime
    // class R8 may rename, while the interface is pinned by a keep rule.
    jclass cb_cls = env->FindClass("dev/sruti/convert/ConverterBridge$ConvertCallback");
    if (cb_cls == nullptr) {
        return nullptr;  // NoClassDefFoundError already pending
    }
    const jmethodID on_progress =
        env->GetMethodID(cb_cls, "onProgress", "(ILjava/lang/String;JJ)Z");
    const jmethodID on_warning =
        env->GetMethodID(cb_cls, "onWarning", "(Ljava/lang/String;)V");
    env->DeleteLocalRef(cb_cls);

    if (on_progress == nullptr || on_warning == nullptr) {
        jclass cls = env->FindClass("java/lang/IllegalStateException");
        if (cls != nullptr) {
            env->ThrowNew(cls, "callback is missing onProgress or onWarning");
        }
        return nullptr;
    }

    bool cancelled = false;

    const auto progress = [&](const sruti::ConvertProgress & p) -> bool {
        jstring detail = env->NewStringUTF(p.detail.c_str());
        const jboolean keep_going = env->CallBooleanMethod(
            callback, on_progress,
            static_cast<jint>(p.stage), detail,
            static_cast<jlong>(p.tensors_done), static_cast<jlong>(p.tensors_total));
        env->DeleteLocalRef(detail);

        if (env->ExceptionCheck()) {
            cancelled = true;
            return false;
        }
        if (keep_going == JNI_FALSE) {
            cancelled = true;
            return false;
        }
        return true;
    };

    const sruti::ConvertResult result = sruti::convert_model(options, progress);

    // A pending Kotlin exception must propagate rather than be masked by a
    // returned error string.
    if (env->ExceptionCheck()) {
        return nullptr;
    }

    for (const std::string & warning : result.warnings) {
        jstring text = env->NewStringUTF(warning.c_str());
        env->CallVoidMethod(callback, on_warning, text);
        env->DeleteLocalRef(text);
        if (env->ExceptionCheck()) {
            return nullptr;
        }
    }

    if (result.ok) {
        LOGI("converted %s (%s, %.2f B params, %llu bytes)",
             options.out_path.c_str(), result.arch.c_str(),
             static_cast<double>(result.param_count) / 1e9,
             static_cast<unsigned long long>(result.output_bytes));
        return nullptr;
    }

    if (cancelled && result.error == "cancelled") {
        return env->NewStringUTF("cancelled");
    }
    return env->NewStringUTF(result.error.c_str());
}

}  // extern "C"
