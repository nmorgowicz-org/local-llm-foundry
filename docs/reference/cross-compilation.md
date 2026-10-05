# Cross-Compilation Guide

This project builds four targets from a single self-hosted GitHub Actions runner (a privileged Proxmox LXC container, Ubuntu 24.04, x86_64).

| Target | Binary | Toolchain |
|--------|--------|-----------|
| `x86_64-unknown-linux-gnu` | `llama-monitor-linux-x86_64` | Native GCC |
| `aarch64-unknown-linux-gnu` | `llama-monitor-linux-aarch64` | `aarch64-linux-gnu-gcc` cross-compiler + multiarch apt packages |
| `x86_64-pc-windows-gnu` | `llama-monitor-windows-x86_64.exe` | MinGW (`gcc-mingw-w64`) |
| `aarch64-apple-darwin` | `llama-monitor-macos-aarch64` | osxcross at `/opt/osxcross/` |

---

## Runner requirements

The runner is a **privileged** Proxmox LXC container with `features: nesting=1` in the Proxmox LXC config. Privileged mode is required for binfmt_misc to be accessible inside the container.

QEMU user-static must be installed on the **Proxmox host** (not inside the container):

```bash
# On the Proxmox host
apt install qemu-user-static
```

Once installed on the host, the binfmt handlers (including `qemu-aarch64`) are visible inside all containers via the shared `/proc/sys/fs/binfmt_misc/`. This is what allows `apt install :arm64` post-install scripts to run correctly inside the container.

---

## Adding a new Linux arm64 dependency

When a Rust crate requires a system library (gtk, xdo, etc.) you must install **both** packages on the runner — the runtime and the dev package — for **both** architectures:

```bash
# On the runner as root
apt install -y libfoo-dev          # native x86_64 (for the x86_64-unknown-linux-gnu build)
apt install -y libfoo-dev:arm64    # arm64 (for the aarch64-unknown-linux-gnu build)
```

**Why both?**
- The runtime package (e.g. `libfoo3:arm64`) installs the versioned `.so.X` file but no unversioned `.so` symlink.
- The dev package (e.g. `libfoo-dev:arm64`) installs the unversioned `libfoo.so` symlink that the linker needs when resolving `-lfoo`.
- Missing only the dev package causes a link error: `cannot find -lfoo: No such file or directory`.

**Packages currently installed for arm64 cross-compilation:**

```bash
libgtk-3-dev:arm64
libayatana-appindicator3-dev:arm64
libxdo-dev        # amd64 only — provides /usr/include/xdo.h and x86_64 .so symlink
libwebkit2gtk-4.1-dev:arm64  # tray-popover feature — required by wry on Linux
libwebkit2gtk-4.1-dev        # amd64 — same for native x86_64 build
```

> **libxdo conflict:** `libxdo-dev` and `libxdo-dev:arm64` declare a dpkg conflict over the shared header `/usr/include/xdo.h` and cannot both be installed. Install only `libxdo-dev` (amd64), then manually create the arm64 symlink:
> ```bash
> ln -sf libxdo.so.3 /usr/lib/aarch64-linux-gnu/libxdo.so
> ```
> This gives the arm64 cross-linker the `.so` symlink it needs without conflicting packages. The header from the amd64 dev package is arch-neutral and works for both targets.

### pkg-config for arm64

The workflow sets these env vars when building `aarch64-unknown-linux-gnu`:

```
PKG_CONFIG_ALLOW_CROSS=1
PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/share/pkgconfig
```

`PKG_CONFIG_ALLOW_CROSS=1` — tells pkg-config it's allowed to return results for a foreign arch.
`PKG_CONFIG_PATH` — points at the arm64 `.pc` files installed by the `:arm64` dev packages.

If you add a new dep whose crate build script uses pkg-config, confirm its `.pc` file lands in `/usr/lib/aarch64-linux-gnu/pkgconfig/` after installing the `:arm64` dev package:

```bash
ls /usr/lib/aarch64-linux-gnu/pkgconfig/libfoo*.pc
```

### Transitive shared library symbols

The arm64 linker flag `-Wl,--allow-shlib-undefined` is set in `.cargo/config.toml` (written by the workflow's "Setup osxcross" step). This suppresses undefined symbol errors for transitive deps listed as `Requires.private` in `.pc` files (e.g. `epoxy` via `gtk+-3.0.pc`). Those symbols are present at runtime on a real arm64 system — this flag just tells the cross-linker not to fail because it can't verify them.

---

## Adding a new macOS arm64 dependency

macOS cross-compilation uses osxcross at `/opt/osxcross/` with the `MacOSX26.1.sdk`.

The key environment variables set during the macOS arm64 build:

```
SDKROOT=/opt/osxcross/target/SDK/MacOSX26.1.sdk
CC_aarch64_apple_darwin=/opt/osxcross/target/bin/aarch64-apple-darwin25.1-clang
AR_aarch64_apple_darwin=/opt/osxcross/target/bin/aarch64-apple-darwin25.1-ar
```

`SDKROOT` — prevents rustc from trying to invoke `xcrun --sdk macosx --show-sdk-path` (which doesn't exist on Linux).
`CC_aarch64_apple_darwin` — tells cc-rs and the cmake crate to use osxcross clang instead of the system GCC, which doesn't understand Apple-specific flags like `-arch arm64` and `-mmacosx-version-min`.

The cargo linker config (written by the workflow's "Setup osxcross" step) is:

```toml
[target.aarch64-apple-darwin]
linker = "/opt/osxcross/target/bin/aarch64-apple-darwin25.1-clang"
ar     = "/opt/osxcross/target/bin/aarch64-apple-darwin25.1-ar"
rustflags = [
  "-C", "link-arg=-fuse-ld=/opt/osxcross/target/bin/aarch64-apple-darwin25.1-ld",
  "-C", "link-arg=-isysroot",
  "-C", "link-arg=/opt/osxcross/target/SDK/MacOSX26.1.sdk",
]
```

`-fuse-ld` — forces clang to use the osxcross ld64 instead of falling back to the system GNU ld (which doesn't understand Mach-O objects).
`-isysroot` as two separate args — must be split; passing `-isysroot /path` as a single `link-arg` causes a leading space in the path which the linker rejects.

macOS system libraries (CoreFoundation, Security, etc.) are bundled in the SDK and require no additional installation.

---

## Windows deps

Windows cross-compilation uses MinGW. System libraries (WMI, Win32 API, etc.) are provided by the MinGW sysroot and require no additional installation.

Windows-only Rust crates should be gated in `Cargo.toml`:

```toml
[target.'cfg(windows)'.dependencies]
wmi = { version = "0.18.4", features = ["default"] }
```

This prevents Windows-only crates (which may pull in COM/Win32 bindings) from compiling on Linux and macOS builds.

---

## Checking for unused dependencies

```bash
cargo install cargo-machete
cargo machete
```

`cargo-machete` uses text search and is fast but can miss deps used only via macros or re-exports. If it flags something that looks wrong, verify manually before removing.

For more accurate analysis (requires nightly):

```bash
cargo install cargo-udeps
cargo +nightly udeps
```

---

## macOS signing and notarization

The `aarch64-apple-darwin` binary is code-signed with a **Developer ID
Application** certificate and notarized by Apple as part of the release
workflow — entirely on the Linux runner, using
[`rcodesign`](https://github.com/indygreg/apple-platform-rs) (pure Rust, no
Xcode or macOS runner required). The pipeline mirrors the verified
persona-forge setup (see that repo's
`docs/archive/macos-code-signing/20260918-macos_code_signing.md`).

### Secrets

All five are organization secrets, available to the release workflow:

| Secret | Content |
| --- | --- |
| `MACOS_KEY_PEM` | Developer ID Application private key (PEM) |
| `MACOS_CERT_PEM` | Developer ID Application certificate chain (PEM; include the Apple intermediate + root) |
| `APPLE_ISSUER_ID` | App Store Connect API key issuer UUID |
| `APPLE_KEY_ID` | App Store Connect API key identifier |
| `APPLE_PRIVATE_KEY` | App Store Connect API private key (`.p8` PEM) |

### Flow

1. `rcodesign sign --pem-file key.pem --pem-file cert.pem --for-notarization`
   — `--for-notarization` enables the hardened runtime and forces a secure
   timestamp in one flag. (Signing with only
   `--code-signature-flags runtime` omits the timestamp and the notary
   service rejects the binary.)
2. Zip the binary (`rcodesign` cannot notarize a bare Mach-O) and
   `rcodesign notary-submit --api-key-file … --wait`. Typical wait is 1–5
   minutes.
3. `rcodesign verify`.

No stapling step: Apple supports stapling only for `.app`/`.pkg`/`.dmg`,
not bare executables. Gatekeeper performs its normal online ticket check on
first launch instead, which removes the manual
`xattr -dr com.apple.quarantine` workaround.

---

## Common errors and fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `cannot find -lfoo` (arm64 build) | `libfoo-dev:arm64` not installed | `apt install libfoo-dev:arm64` |
| `cannot find -lfoo` (x86_64 build) | `libfoo-dev` not installed | `apt install libfoo-dev` |
| `cc: error: unrecognized command-line option '-arch'` | cc-rs using GCC for macOS target | Ensure `CC_aarch64_apple_darwin` is set to osxcross clang |
| `unrecognised emulation mode: llvm` | GNU ld used for macOS link step | Ensure `-fuse-ld=.../aarch64-apple-darwin25.1-ld` is in rustflags |
| `no such sysroot directory: ' /opt/...'` | `-isysroot /path` passed as one arg | Split into two separate `link-arg` entries |
| `xcrun --sdk macosx --show-sdk-path failed` | rustc looking for Xcode | Set `SDKROOT` env var |
| `pkg-config: error: ...` on arm64 | pkg-config not configured for cross | Set `PKG_CONFIG_ALLOW_CROSS=1` and `PKG_CONFIG_PATH` |
| Runner service `CHDIR` failure after LXC conversion | UID mapping shift (unprivileged→privileged) | `find /home/github-runner -uid 101000 -exec chown github-runner:github-runner {} +` |
| binfmt_misc `Permission denied` writing to register | LXC can't write to host's binfmt_misc | Install `qemu-user-static` on the Proxmox **host**, not in the container |
