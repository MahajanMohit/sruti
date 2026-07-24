# JNI entry points are resolved by name from native code, so R8 cannot see the
# references and would otherwise strip or rename them.
-keepclasseswithmembernames,includedescriptorclasses class dev.sruti.llm.LlamaBridge {
    native <methods>;
}
-keep class dev.sruti.llm.LlamaBridge { *; }

# Called from C++ via GetMethodID.
-keep interface dev.sruti.llm.LlamaBridge$TokenCallback { *; }
