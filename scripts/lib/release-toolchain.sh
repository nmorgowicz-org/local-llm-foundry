#!/usr/bin/env bash
# Shared toolchain helpers for the release build scripts. Source this file; do
# not execute it. Callers are expected to run with `set -euo pipefail`.
#
# Cross-compilation settings are exported as per-target CARGO_TARGET_<TRIPLE>_*
# environment variables instead of writing ~/.cargo/config.toml, so a developer's
# (or runner's) existing Cargo configuration is never overwritten.

OSXCROSS_ROOT="${OSXCROSS_ROOT:-/opt/osxcross/target}"

# detect_osxcross: sets DARWIN_VERSION (e.g. darwin25.1), MACOS_SDK (e.g.
# MacOSX26.1.sdk) and LLD_BIN (path to ld64.lld). Returns 1 (without exiting the
# caller) when the toolchain is incomplete, so callers can print a friendly FAIL.
detect_osxcross() {
  DARWIN_VERSION=""
  MACOS_SDK=""
  LLD_BIN=""

  local clang
  # Pipelines below can legitimately fail when osxcross is absent; `|| true`
  # keeps `set -e -o pipefail` from aborting before the caller reports it.
  clang=$(compgen -G "${OSXCROSS_ROOT}/bin/aarch64-apple-darwin*-clang" 2>/dev/null | sort -V | tail -1 || true)
  if [[ -n "$clang" ]]; then
    DARWIN_VERSION=$(basename "$clang" | sed -n 's/^aarch64-apple-\(darwin[0-9.]*\)-clang$/\1/p' || true)
  fi
  local sdk
  sdk=$(compgen -G "${OSXCROSS_ROOT}/SDK/MacOSX*.sdk" 2>/dev/null | sort -V | tail -1 || true)
  if [[ -n "$sdk" ]]; then
    MACOS_SDK=$(basename "$sdk")
  fi
  LLD_BIN=$(command -v ld64.lld 2>/dev/null || true)

  [[ -n "$DARWIN_VERSION" && -n "$MACOS_SDK" ]]
}

# print_osxcross_diagnostics: explain what was (not) found under OSXCROSS_ROOT.
print_osxcross_diagnostics() {
  echo "FAIL: could not detect osxcross toolchain in ${OSXCROSS_ROOT}/" >&2
  echo "  clang binaries: $(compgen -G "${OSXCROSS_ROOT}/bin/*-clang" 2>/dev/null | tr '\n' ' ' || true)" >&2
  echo "  SDKs: $(compgen -G "${OSXCROSS_ROOT}/SDK/*.sdk" 2>/dev/null | tr '\n' ' ' || true)" >&2
}

# export_target_env <triple>: export the linker/rustflags/cc settings needed to
# build one target. Environment variables take precedence over config files.
export_target_env() {
  case "$1" in
    aarch64-apple-darwin)
      detect_osxcross || { print_osxcross_diagnostics; return 1; }
      [[ -n "$LLD_BIN" ]] || { echo "FAIL: ld64.lld not found on PATH" >&2; return 1; }
      local prefix="${OSXCROSS_ROOT}/bin/aarch64-apple-${DARWIN_VERSION}"
      local sdk="${OSXCROSS_ROOT}/SDK/${MACOS_SDK}"
      echo "osxcross: ${DARWIN_VERSION}, ${MACOS_SDK}, linker ${LLD_BIN}" >&2
      export CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER="${prefix}-clang"
      export CARGO_TARGET_AARCH64_APPLE_DARWIN_RUSTFLAGS="-C link-arg=-fuse-ld=${LLD_BIN} -C link-arg=-isysroot -C link-arg=${sdk}"
      export SDKROOT="$sdk"
      export CC_aarch64_apple_darwin="${prefix}-clang"
      export AR_aarch64_apple_darwin="${prefix}-ar"
      export AR="${prefix}-ar"
      export RANLIB="${prefix}-ranlib"
      ;;
    aarch64-unknown-linux-gnu)
      export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER="aarch64-linux-gnu-gcc"
      export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C link-arg=-Wl,--allow-shlib-undefined"
      export PKG_CONFIG_ALLOW_CROSS=1
      export PKG_CONFIG_LIBDIR="/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/share/pkgconfig"
      export CC_aarch64_unknown_linux_gnu="aarch64-linux-gnu-gcc"
      export CXX_aarch64_unknown_linux_gnu="aarch64-linux-gnu-g++"
      export AR_aarch64_unknown_linux_gnu="aarch64-linux-gnu-ar"
      ;;
    x86_64-pc-windows-gnu)
      export CROSS_REMOTE=1
      export CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUSTFLAGS="-C target-feature=+crt-static"
      export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER="cc"
      ;;
    x86_64-unknown-linux-gnu) ;;
    *)
      echo "ERROR: unknown target '$1'" >&2
      return 1
      ;;
  esac
}
