#!/usr/bin/env bash
# Build one release target. Called by the matrix release workflow.
# Usage: build-single-target.sh <target>
#
# Toolchain settings are passed through per-target CARGO_TARGET_<TRIPLE>_*
# environment variables (see scripts/lib/release-toolchain.sh); ~/.cargo/config.toml
# is never modified. CARGO_TARGET_DIR is honoured if set by the caller.
set -euo pipefail

TARGET="${1:?Usage: build-single-target.sh <target>}"

# shellcheck source=lib/release-toolchain.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib/release-toolchain.sh"

export_target_env "$TARGET" || exit 1

case "$TARGET" in
  x86_64-unknown-linux-gnu|aarch64-unknown-linux-gnu|aarch64-apple-darwin)
    cargo build --release --target "$TARGET"
    ;;
  x86_64-pc-windows-gnu)
    cross build --release --target x86_64-pc-windows-gnu \
      --no-default-features --features native-tray,webview-popover
    ;;
esac
