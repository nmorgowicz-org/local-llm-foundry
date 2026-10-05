#!/usr/bin/env bash
# Smoke-build every release target in parallel by delegating to
# build-single-target.sh, so features and toolchain detection stay identical to
# the release workflow. Each target gets its own CARGO_TARGET_DIR
# (target/smoke-<target>) so the parallel builds do not contend for the lock.
# Usage: build-release-targets.sh [target ...]   (default: all four targets)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

if [[ $# -gt 0 ]]; then
  TARGETS=("$@")
else
  TARGETS=(
    x86_64-unknown-linux-gnu
    aarch64-unknown-linux-gnu
    x86_64-pc-windows-gnu
    aarch64-apple-darwin
  )
fi

LOG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/build-release-targets.XXXXXX")"
trap 'rm -rf "$LOG_DIR"' EXIT

echo "Building release targets (logs: $LOG_DIR)..."

PIDS=()
for target in "${TARGETS[@]}"; do
  CARGO_TARGET_DIR="target/smoke-${target}" \
    bash "$SCRIPT_DIR/build-single-target.sh" "$target" \
    > "$LOG_DIR/${target}.log" 2>&1 &
  PIDS+=("$!")
done

result=0
for i in "${!TARGETS[@]}"; do
  target="${TARGETS[$i]}"
  if ! wait "${PIDS[$i]}"; then
    echo "FAILED: $target"
    cat "$LOG_DIR/${target}.log"
    result=1
  fi
done

[[ "$result" -eq 0 ]] && echo "All release targets built successfully."
exit "$result"
