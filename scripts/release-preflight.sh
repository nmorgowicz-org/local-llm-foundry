#!/usr/bin/env bash
# Release toolchain preflight.
#
# Usage:
#   release-preflight.sh [<target>]   check the build toolchain (all targets when omitted)
#   release-preflight.sh --publish    check the tools the release/publish job needs
#
# Targets: x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu
#          x86_64-pc-windows-gnu aarch64-apple-darwin
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/release-toolchain.sh
source "$SCRIPT_DIR/lib/release-toolchain.sh"

MODE="${1:-all}"

need() {
  command -v "$1" >/dev/null 2>&1 || { echo "FAIL: $1 not found${2:+ ($2)}"; exit 1; }
}

echo "Running release preflight checks (${MODE})..."

if [[ "$MODE" == "--publish" ]]; then
  need jq "builds checksums.json"
  need sha256sum "checksums"
  need unzip "validate-release-contract.mjs inspects the Windows zip"
  need tar "validate-release-contract.mjs inspects the macOS tarball"
  need node "validate-release-contract.mjs"
  echo "All preflight checks passed."
  exit 0
fi

case "$MODE" in
  all|x86_64-unknown-linux-gnu|aarch64-unknown-linux-gnu|x86_64-pc-windows-gnu|aarch64-apple-darwin) ;;
  *) echo "FAIL: unknown preflight mode '$MODE'"; exit 2 ;;
esac

wants() { [[ "$MODE" == "all" || "$MODE" == "$1" ]]; }

need cargo
need rustup
need pkg-config

if wants x86_64-pc-windows-gnu; then
  need cross "Windows build"
  need docker "cross runs inside a container"
  docker info >/dev/null 2>&1 || { echo "FAIL: Docker daemon not reachable"; exit 1; }
  docker buildx version >/dev/null 2>&1 || { echo "FAIL: docker buildx plugin not available"; exit 1; }
  need zip "Windows bundle is zipped"
  # The workflow publishes the sensor bridge for the Windows target only. The
  # `all` mode (CI smoke builds) does not publish, so dotnet is required only
  # when the Windows target is requested explicitly.
  if [[ "$MODE" == "x86_64-pc-windows-gnu" ]]; then
    need dotnet "sensor_bridge needs the .NET 10 SDK"
    dotnet --list-sdks | grep -q '^10\.' || { echo "FAIL: .NET 10 SDK is required for sensor_bridge"; exit 1; }
  fi
fi

if wants aarch64-unknown-linux-gnu; then
  need aarch64-linux-gnu-gcc
  need aarch64-linux-gnu-g++
fi

if wants aarch64-apple-darwin; then
  if ! detect_osxcross; then
    print_osxcross_diagnostics
    exit 1
  fi
  echo "Checking osxcross toolchain (${DARWIN_VERSION}, ${MACOS_SDK})..."
  for tool in clang ar ranlib; do
    bin="${OSXCROSS_ROOT}/bin/aarch64-apple-${DARWIN_VERSION}-${tool}"
    test -x "$bin" || { echo "FAIL: missing osxcross tool: $bin"; exit 1; }
  done
  # build-single-target.sh links with ld64.lld, not the osxcross ld.
  [[ -n "$LLD_BIN" ]] || { echo "FAIL: ld64.lld not found on PATH (needed to link aarch64-apple-darwin)"; exit 1; }
  test -d "${OSXCROSS_ROOT}/SDK/${MACOS_SDK}" \
    || { echo "FAIL: missing SDK dir: ${OSXCROSS_ROOT}/SDK/${MACOS_SDK}"; exit 1; }
fi

# Capture first: `rustup ... | grep -q` can SIGPIPE rustup and trip pipefail.
installed_targets="$(rustup target list --installed || true)"
for target in \
  x86_64-unknown-linux-gnu \
  aarch64-unknown-linux-gnu \
  x86_64-pc-windows-gnu \
  aarch64-apple-darwin; do
  wants "$target" || continue
  grep -qx "$target" <<<"$installed_targets" \
    || { echo "FAIL: rustup target ${target} not installed"; exit 1; }
done

for lib in webkit2gtk-4.1 javascriptcoregtk-4.1 libsoup-3.0; do
  if wants x86_64-unknown-linux-gnu; then
    pkg-config --libs "$lib" >/dev/null 2>&1 || { echo "FAIL: pkg-config cannot find ${lib} (x86_64)"; exit 1; }
  fi
  if wants aarch64-unknown-linux-gnu; then
    PKG_CONFIG_ALLOW_CROSS=1 \
      PKG_CONFIG_LIBDIR=/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/share/pkgconfig \
      pkg-config --libs "$lib" >/dev/null 2>&1 \
      || { echo "FAIL: pkg-config cannot find ${lib} (aarch64 cross)"; exit 1; }
  fi
done

echo "All preflight checks passed."
