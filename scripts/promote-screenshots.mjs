#!/usr/bin/env node
// Refreshes docs/screenshots/*.png (the curated, doc-referenced screenshots)
// from the freshly captured docs/screenshots/artifacts/** tree produced by
// `node tests/ui/capture/index.mjs --scenario <name>`.
//
// Promotion is filename-based: a root screenshot is refreshed from the
// artifact with the same basename, provided:
//   - it isn't in docs/screenshots/stale-scenario-allowlist.json — those
//     filenames have no live scenario producing them anymore, so the only
//     "match" under artifacts/ is stale leftover data. Overwriting from it
//     would silently revert a manually re-verified image. Refresh those by
//     hand, deliberately, then re-verify with check-stale-screenshot-scenarios.
//   - artifacts/windows/** is never treated as a source — it's a separate
//     platform capture, not a refresh of the native (macOS) promoted image.
//   - when a filename exists under more than one category directory (a
//     leftover from a scenario's prior category before a reorg, e.g. an
//     old wizard-llamacpp copy of a now wizard-rapidmlx artifact), the
//     most recently modified copy wins — that's the one a live scenario
//     actually just produced.
//
// A basename with zero matching artifacts is left untouched and reported —
// most commonly because the capturing scenario has a known flake on this
// host (see e.g. tls.mjs's db-admin-section/mode-acme-full states) or the
// image was hand-composed rather than captured (e.g. the README hero shot).
//
// Usage: node scripts/promote-screenshots.mjs [--dry-run]

import { readFileSync, readdirSync, statSync, copyFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const rootScreenshotsDir = path.join(repoRoot, 'docs/screenshots');
const artifactsDir = path.join(rootScreenshotsDir, 'artifacts');
const dryRun = process.argv.includes('--dry-run');

let allowlist = new Set();
try {
    allowlist = new Set(JSON.parse(readFileSync(path.join(rootScreenshotsDir, 'stale-scenario-allowlist.json'), 'utf8')));
} catch {
    allowlist = new Set();
}

function findArtifacts(dir, basename, out = []) {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
        if (entry.name === '__capture-receipt-test__') continue;
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) {
            findArtifacts(full, basename, out);
        } else if (entry.name === basename) {
            out.push(full);
        }
    }
    return out;
}

const rootScreenshots = readdirSync(rootScreenshotsDir).filter(f => f.endsWith('.png'));

let refreshed = 0;
let unchanged = 0;
let skippedAllowlist = 0;
let noSource = 0;

for (const basename of rootScreenshots) {
    const target = path.join(rootScreenshotsDir, basename);

    if (allowlist.has(basename)) {
        console.log(`SKIP (allowlisted, refresh by hand): ${basename}`);
        skippedAllowlist += 1;
        continue;
    }

    const candidates = findArtifacts(artifactsDir, basename)
        .filter(p => !path.relative(artifactsDir, p).split(path.sep).includes('windows'));

    if (candidates.length === 0) {
        console.log(`NO SOURCE (nothing currently produces this filename): ${basename}`);
        noSource += 1;
        continue;
    }

    candidates.sort((a, b) => statSync(b).mtimeMs - statSync(a).mtimeMs);
    const source = candidates[0];

    if (candidates.length > 1) {
        console.log(`  (${candidates.length} candidates for ${basename}, using newest: ${path.relative(repoRoot, source)})`);
    }

    const sourceBytes = readFileSync(source);
    const targetBytes = readFileSync(target);
    if (sourceBytes.equals(targetBytes)) {
        unchanged += 1;
        continue;
    }

    console.log(`REFRESH: ${basename}`);
    if (!dryRun) copyFileSync(source, target);
    refreshed += 1;
}

console.log(
    `\n${refreshed} refreshed, ${unchanged} already current, ${skippedAllowlist} allowlisted (untouched), ${noSource} with no current source` +
    (dryRun ? ' [dry run — no files written]' : '') + '.',
);
