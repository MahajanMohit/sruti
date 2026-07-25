# JNI entry points are resolved by name from native code, so R8 cannot see the
# references and would otherwise strip or rename them.
-keepclasseswithmembernames,includedescriptorclasses class dev.sruti.llm.LlamaBridge {
    native <methods>;
}
-keep class dev.sruti.llm.LlamaBridge { *; }

# Called from C++ via GetMethodID.
-keep interface dev.sruti.llm.LlamaBridge$TokenCallback { *; }

-keepclasseswithmembernames,includedescriptorclasses class dev.sruti.convert.ConverterBridge {
    native <methods>;
}
-keep class dev.sruti.convert.ConverterBridge { *; }
-keep interface dev.sruti.convert.ConverterBridge$ConvertCallback { *; }

-keepclasseswithmembernames,includedescriptorclasses class dev.sruti.llm.ChatBridge {
    native <methods>;
}
-keep class dev.sruti.llm.ChatBridge { *; }
-keep interface dev.sruti.llm.ChatBridge$ChatCallback { *; }
