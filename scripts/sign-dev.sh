#!/usr/bin/env bash
# Sign the release binary with the stable Developer ID identity.
#
# macOS Local Network (TCC) permission is keyed to the binary's code-signing
# identity. An unsigned/ad-hoc binary gets a new identity on every `cargo
# build`, which silently resets Local Network permission — remote (LAN)
# attach endpoints then fail with "No route to host" while localhost keeps
# working. Signing with the Developer ID makes the attribution stable, so
# the permission you grant once survives rebuilds.
set -euo pipefail

BIN="${1:-target/release/local-llm-foundry}"
codesign --force --sign "Developer ID Application: Nicholas Morgowicz (AX88UTBYB7)" "$BIN"
codesign -v "$BIN"
echo "Signed $BIN — Local Network permission now persists across rebuilds."
