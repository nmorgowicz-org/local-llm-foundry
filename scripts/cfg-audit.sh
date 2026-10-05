#!/usr/bin/env bash
# Audits #[cfg(target_os = "macos")] module guards for consistency: modules
# declared behind that gate must not be imported by `use` statements that are not
# themselves gated.
#
# A gate may sit on the line directly above the item or, for multi-line / stacked
# attributes, up to two lines above it, so the two preceding lines are inspected.
# Usage: scripts/cfg-audit.sh

set -euo pipefail

cd "$(dirname "$0")/.."

MACOS_GATE='cfg\((all\()?[^)]*target_os *= *"macos"'

# has_macos_gate FILE LINE: succeeds when one of the 2 lines before LINE (or the
# line itself, for single-line forms) carries a macOS cfg gate.
has_macos_gate() {
    local file="$1" line="$2" start
    start=$((line - 2))
    [ "$start" -lt 1 ] && start=1
    sed -n "${start},${line}p" "$file" | grep -Eq "$MACOS_GATE"
}

echo "=== cfg(target_os = \"macos\") audit ==="

errors=0
gated_names=()
gated_files=()   # "<dir>/<name>.rs" of each gated module
gated_dirs=()    # "<dir>/<name>/"   of each gated module

# Pass 1: collect macOS-gated module declarations.
while IFS=: read -r decl_file decl_line decl_text; do
    mod_name=$(printf '%s\n' "$decl_text" | sed -n 's/.*mod \([A-Za-z0-9_]*\) *;.*/\1/p')
    [ -z "$mod_name" ] && continue
    has_macos_gate "$decl_file" "$decl_line" || continue
    decl_dir=$(dirname "$decl_file")
    gated_names+=("$mod_name:$decl_file:$decl_line")
    gated_files+=("${decl_dir}/${mod_name}.rs")
    gated_dirs+=("${decl_dir}/${mod_name}/")
done < <(grep -rnE '^[[:space:]]*(pub(\([a-z]+\))? )?mod [A-Za-z0-9_]+ *;' src/ --include='*.rs' || true)

gated_modules=${#gated_names[@]}

# is_gated_file FILE: the importing file is itself part of a macOS-gated module.
is_gated_file() {
    local f="$1" i
    for i in "${!gated_files[@]}"; do
        [ "$f" = "${gated_files[$i]}" ] && return 0
        case "$f" in "${gated_dirs[$i]}"*) return 0 ;; esac
    done
    return 1
}

# Pass 2: look for ungated imports of each gated module.
for entry in ${gated_names[@]+"${gated_names[@]}"}; do
    IFS=: read -r mod_name decl_file decl_line <<<"$entry"
    while IFS=: read -r use_file use_line _; do
        # Entirely macOS-only files (#![cfg(...)] or gated modules) may import freely.
        if grep -Eq "^#!\[${MACOS_GATE}" "$use_file" || is_gated_file "$use_file"; then
            continue
        fi
        if ! has_macos_gate "$use_file" "$use_line"; then
            echo "⚠ Unconditional import of macOS-gated module '${mod_name}' in ${use_file}:${use_line}"
            errors=$((errors + 1))
        fi
    done < <(grep -rnE "^[[:space:]]*(pub(\([a-z]+\))? )?use .*(::|\{|[[:space:]])${mod_name}(::|[[:space:]]*[;,}])" src/ --include='*.rs' 2>/dev/null \
                | grep -v "^${decl_file}:${decl_line}:" || true)
done

if [ "$gated_modules" -eq 0 ]; then
    echo "FAILED: found no cfg(target_os = \"macos\")-gated modules; the audit pattern is stale"
    exit 1
fi

if [ "$errors" -gt 0 ]; then
    echo "FAILED: Found $errors cfg guard violations (${gated_modules} gated modules checked)"
    exit 1
fi

echo "✓ All cfg(target_os = \"macos\") guards are consistent (${gated_modules} gated modules checked)"
exit 0
