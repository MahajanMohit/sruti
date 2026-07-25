# Releasing

Sruti is sideloaded, not published to the Play Store. A release is a signed APK
attached to a GitHub Release.

## One-time setup: the signing key

Android identifies an app by its signing key. **An update signed with a
different key cannot install over an existing one** — the user has to uninstall
first, losing their models and conversations. So this key is generated once and
then never regenerated. Back it up somewhere you will still have in two years.

Generate it:

```bash
keytool -genkeypair -v \
  -keystore sruti-release.jks \
  -alias sruti \
  -keyalg RSA -keysize 4096 \
  -validity 10000 \
  -storetype pkcs12
```

Encode it for GitHub — `-w 0` matters, because a wrapped base64 string decodes
to a corrupt file and the workflow will tell you so:

```bash
base64 -w 0 sruti-release.jks > sruti-release.jks.base64
```

Add four repository secrets under **Settings → Secrets and variables → Actions**:

| Secret | Value |
|---|---|
| `KEYSTORE_BASE64` | contents of `sruti-release.jks.base64` |
| `KEYSTORE_PASSWORD` | the store password you chose |
| `KEY_ALIAS` | `sruti` |
| `KEY_PASSWORD` | the key password you chose |

Keep `sruti-release.jks` out of the repository. It is not in `.gitignore` by
name because it should never be in the working tree at all.

## Cutting a release

The tag is the version. Nothing is edited by hand:

```bash
git tag v0.2.0
git push origin v0.2.0
```

That runs `.github/workflows/release.yml`, which:

1. Runs the native and Kotlin test suites, and stops if either fails.
2. Fails immediately, with the reason, if no signing key is configured — an
   unsigned APK cannot be installed, so publishing one would be worse than
   publishing nothing.
3. Builds `assembleRelease` with `-Psruti.version=0.2.0`, which sets both
   `versionName` and a monotonic `versionCode` (`0.2.0` → `200`).
4. Checks the APK actually contains every ggml CPU variant. Missing ones install
   fine and then fail at runtime with "no compute backend is available".
5. Verifies the signature with `apksigner`.
6. Publishes a GitHub Release with the APK, its SHA-256, and the commit log
   since the previous tag.

A tag with a suffix — `v0.3.0-rc1` — is published as a pre-release, and the
in-app update check ignores pre-releases.

To rehearse without publishing, run the workflow manually from the Actions tab
with **publish** unchecked. It builds and uploads a workflow artifact instead.

## Versioning

`versionCode` is derived as `major * 10000 + minor * 100 + patch`, so it rises
monotonically across semver and Android will accept each release as an upgrade
of the last. Never lower a version component in a tag.

## Why one ABI

Only `arm64-v8a` is built. Every Android device that can usefully run a 1–2B
model at a tolerable speed is 64-bit ARM, and shipping a 32-bit build would
mostly serve to let someone install it on hardware where it will not perform.

Choosing *within* arm64 is a runtime decision, not a build-time one: ggml ships
a CPU backend per feature level (ARMv8.0 through ARMv9.2) and selects the best
one the device can actually execute when the app starts. That is why a single
APK runs on a budget part without SIGILL and still uses dot-product and int8
matmul instructions on a flagship.

## Updates

The app checks `https://api.github.com/repos/<owner>/<repo>/releases/latest`
when the user taps **Check for updates** in settings. Manual, not background:
this app promises not to make network calls the user did not ask for, and a
poller would break that for a convenience.

It reports and links; it does not download and install. That is a different
level of trust, and the decision belongs with the user.

Forks should build with `-Psruti.releasesRepo=owner/repo` so the check asks
their own releases.
