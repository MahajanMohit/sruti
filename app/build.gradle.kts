import java.net.URI

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.serialization)
    alias(libs.plugins.ksp)
    alias(libs.plugins.hilt)
}

/**
 * The version being built.
 *
 * Taken from the tag at release time (`-Psruti.version=v0.2.0`) so the tag, the
 * APK and the in-app update check can never disagree — a version bumped by hand
 * in this file is one that eventually gets forgotten.
 */
data class ReleaseVersion(val name: String, val code: Int)

val releaseVersion: ReleaseVersion = run {
    val raw = (project.findProperty("sruti.version") as String?)
        ?.trim()
        ?.removePrefix("v")
        ?.takeIf { it.isNotBlank() }
        ?: "0.1.0"

    // Monotonic across semver: 1.2.3 -> 10203. Two digits per component is
    // plenty and keeps the number readable when a device reports it.
    val parts = raw.substringBefore('-').split('.').mapNotNull { it.toIntOrNull() }
    val code = when (parts.size) {
        3 -> parts[0] * 10_000 + parts[1] * 100 + parts[2]
        2 -> parts[0] * 10_000 + parts[1] * 100
        1 -> parts[0] * 10_000
        else -> 1
    }.coerceAtLeast(1)

    ReleaseVersion(raw, code)
}

/** Which repository the in-app update check asks. Overridable for forks. */
val releasesRepo: String =
    (project.findProperty("sruti.releasesRepo") as String?)?.takeIf { it.isNotBlank() }
        ?: "MahajanMohit/sruti"

android {
    namespace = "dev.sruti"
    compileSdk = 36
    ndkVersion = "27.2.12479018"

    defaultConfig {
        applicationId = "dev.sruti"
        minSdk = 31
        targetSdk = 36
        versionCode = releaseVersion.code
        versionName = releaseVersion.name

        // The update check needs to know which repository to ask, and a fork
        // should check its own releases rather than this one.
        buildConfigField("String", "RELEASES_REPO", "\"$releasesRepo\"")

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // arm64 covers every Android device this can usefully run on. CPU feature
        // selection within arm64 is ggml's job at runtime, not a build-time
        // assumption — see the note in src/main/cpp/CMakeLists.txt.
        //
        // Override for emulator testing:
        //   ./gradlew :app:assembleDebug -Psruti.abis=arm64-v8a,x86_64
        ndk {
            val requested = (project.findProperty("sruti.abis") as String?)
                ?.split(',')
                ?.map(String::trim)
                ?.filter(String::isNotEmpty)
                ?: listOf("arm64-v8a")
            abiFilters += requested
        }

        externalNativeBuild {
            cmake {
                // GGML_LLAMAFILE pulls in sgemm kernels that are a net loss on ARM.
                arguments += listOf(
                    "-DANDROID_STL=c++_shared",
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DGGML_LLAMAFILE=OFF",
                    "-DGGML_OPENMP=OFF",
                    "-DLLAMA_CURL=OFF",
                    "-DLLAMA_BUILD_TOOLS=OFF",
                    "-DLLAMA_BUILD_EXAMPLES=OFF",
                    "-DLLAMA_BUILD_TESTS=OFF",
                    "-DLLAMA_BUILD_SERVER=OFF",
                )
                cppFlags += listOf("-O3", "-fexceptions", "-frtti")
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    // Signing details come from the environment, never from a file in the tree.
    // Absent, the release build still succeeds and produces an unsigned APK —
    // which keeps forks and fresh clones building rather than failing on a secret
    // they were never going to have.
    val keystoreFile = System.getenv("SRUTI_KEYSTORE_FILE")?.takeIf { it.isNotBlank() }
    val hasSigningConfig = keystoreFile != null && file(keystoreFile).exists()

    signingConfigs {
        if (hasSigningConfig) {
            create("release") {
                storeFile = file(keystoreFile!!)
                storePassword = System.getenv("SRUTI_KEYSTORE_PASSWORD")
                keyAlias = System.getenv("SRUTI_KEY_ALIAS")
                keyPassword = System.getenv("SRUTI_KEY_PASSWORD")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
            if (hasSigningConfig) {
                signingConfig = signingConfigs.getByName("release")
            }
        }
        debug {
            isJniDebuggable = true
            // So a debug and a release build can sit on the same device at once,
            // which matters when comparing behaviour between them.
            applicationIdSuffix = ".debug"
            versionNameSuffix = "-debug"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlin {
        compilerOptions {
            jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
        }
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
        jniLibs {
            // Extract the .so files to disk on install.
            //
            // Not the modern default, and deliberately so: ggml discovers its CPU
            // backend variants by listing a directory, and with extractNativeLibs
            // false the libraries stay inside the APK and that directory is empty.
            // Nothing registers, and every model load fails with "no compute
            // backend is available on this device".
            //
            // Costs some install size. Buys an app that runs.
            useLegacyPackaging = true
        }
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.activity.compose)

    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.material.icons.extended)
    implementation(libs.androidx.navigation.compose)

    implementation(libs.hilt.android)
    ksp(libs.hilt.compiler)
    implementation(libs.androidx.hilt.navigation.compose)

    implementation(libs.androidx.room.runtime)
    implementation(libs.androidx.room.ktx)
    ksp(libs.androidx.room.compiler)

    implementation(libs.androidx.datastore.preferences)
    implementation(libs.okhttp)
    implementation(libs.kotlinx.serialization.json)
    implementation(libs.kotlinx.coroutines.android)

    debugImplementation(libs.androidx.compose.ui.tooling)

    testImplementation(libs.junit)
    testImplementation(libs.kotlinx.coroutines.test)
    androidTestImplementation(libs.androidx.test.ext.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom))
}

// Compose compiler stability and recomposition reports.
//
// Off by default because they slow the build; enable with -Psruti.composeReports
// when changing anything on the token-streaming path. Unstable parameters there
// are the difference between a smooth 120 Hz stream and a recomposition storm,
// and the reports are the only way to see them rather than guess.
composeCompiler {
    if (project.hasProperty("sruti.composeReports")) {
        val dir = layout.buildDirectory.dir("compose-reports")
        reportsDestination.set(dir)
        metricsDestination.set(dir)
    }
}

// Unit tests that reach the network need the same proxy the rest of the build
// uses; without it OkHttp bypasses it and the live Hub tests self-skip.
tasks.withType<Test>().configureEach {
    listOf("http", "https").forEach { scheme ->
        System.getenv("${scheme.uppercase()}_PROXY")?.let { raw ->
            val uri = URI(raw)
            systemProperty("$scheme.proxyHost", uri.host)
            systemProperty("$scheme.proxyPort", uri.port.toString())
        }
    }
    System.getProperty("javax.net.ssl.trustStore")?.let {
        systemProperty("javax.net.ssl.trustStore", it)
        systemProperty("javax.net.ssl.trustStorePassword",
            System.getProperty("javax.net.ssl.trustStorePassword") ?: "changeit")
        systemProperty("javax.net.ssl.trustStoreType",
            System.getProperty("javax.net.ssl.trustStoreType") ?: "PKCS12")
    }
}
