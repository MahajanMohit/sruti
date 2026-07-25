# Building and releasing

## Local

```bash
git clone --recurse-submodules https://github.com/mahajanmohit/sruti.git
cd sruti
echo "sdk.dir=$ANDROID_HOME" > local.properties

./gradlew assembleDebug
./gradlew installDebug
```

Requires JDK 17+, NDK **27.2.12479018** and CMake 3.22.1. The first build compiles
llama.cpp and seven ggml CPU backends from source — several minutes. Later builds
are incremental.

`local.properties` is gitignored and must never be committed; the SDK path is
per-machine.

## What comes out

| | Debug | Release |
|---|---|---|
| Application id | `dev.sruti.debug` | `dev.sruti` |
| Size | ~71 MB | ~14 MB |
| R8 + resource shrinking | no | yes |
| Installable | yes | only if signed |

Both carry the same ~7.5 MB of native libraries; the difference is dex and
resources. The debug id is suffixed so both can sit on one device at once, which
matters when comparing behaviour between them.

## CI

`.github/workflows/build.yml` runs three jobs:

| Job | When | Artifact | Kept |
|---|---|---|---|
| `test` | every push and PR | test reports | 14 days |
| `build-debug` | every push and PR | `sruti-debug-<run>` | 30 days |
| `build-release` | `main`/`master`, or manual | `sruti-release-{signed,unsigned}-<run>` | 90 days |

Download an APK from the run's **Artifacts** section, unzip, and install.

Three things in the workflow are load-bearing and easy to get wrong:

- **`submodules: recursive` on checkout.** Without llama.cpp, CMake fails at
  configure with an error that does not mention submodules.
- **`app/.cxx` is cached, not just Gradle's cache.** The native build dominates
  and Gradle's caching does not cover CMake output.
- **The packaged-libraries check.** If the ggml CPU variants are missing, the APK
  still builds and then fails at runtime with "no backends are loaded". CI should
  catch that, not the user.

## Signed releases

Unsigned is the default and needs no configuration — forks and fresh clones build
without secrets. Signing switches on when `KEYSTORE_BASE64` is present.

```bash
keytool -genkey -v -keystore sruti-release.jks \
  -keyalg RSA -keysize 2048 -validity 9125 -alias sruti-release

# Single line, no newlines — a wrapped value decodes to a corrupt keystore.
base64 -w0 sruti-release.jks > keystore-base64.txt
```

Add four repository secrets under **Settings → Secrets and variables → Actions**:
`KEYSTORE_BASE64`, `KEYSTORE_PASSWORD`, `KEY_ALIAS`, `KEY_PASSWORD`.

**Back the keystore up somewhere you will still have it in five years.** An app
already published under one signing key cannot be updated with a different one;
losing it means a new package name and no upgrade path for existing installs. The
`.jks` must never be committed — it is gitignored, and the workflow decodes it to
`$RUNNER_TEMP` so a stray `git add -A` cannot pick it up.

## Version bumping

```kotlin
versionCode = 2        // must strictly increase
versionName = "0.2.0"
```

## When a release build fails but debug succeeds

Almost always R8 removing something reached by reflection or JNI. Three keep-rule
sets already exist, one per JNI bridge — `LlamaBridge`, `ChatBridge`,
`ConverterBridge` — and each pins its native methods **and** its callback
interface, because C++ resolves those by name.

If a new bridge or reflective class is added, add a matching rule. Do not disable
minification to make the error go away: R8 is the difference between a 14 MB APK
and a 71 MB one, and turning it off hides the real problem rather than fixing it.

To confirm a class survived:

```bash
grep "dev.sruti.llm.ChatBridge" app/build/outputs/mapping/release/mapping.txt
# an entry mapping to itself means it was kept unrenamed
```

## Emulator builds

```bash
./gradlew assembleDebug -Psruti.abis=arm64-v8a,x86_64
```

Roughly doubles native build time. Only worth it for an x86 emulator; a physical
arm64 device needs the default.
