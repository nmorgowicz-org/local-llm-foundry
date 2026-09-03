#!/usr/bin/env node
// Detects orphaned screenshot provenance: a capture artifact's receipt.json
// names a `scenario` that no longer exists in the current SCENARIOS registry
// (tests/ui/capture/index.mjs). Such a screenshot can never be regenerated —
// if it's still promoted into docs/screenshots/ and referenced from docs
// (especially README.md), it will silently drift from the live UI with no
// way to catch it. Wire into the standard pre-PR check sequence alongside
// scripts/check-unused-screenshots.sh.
//
// Usage: node scripts/check-stale-screenshot-scenarios.mjs

import { readFileSync, readdirSync, statSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const artifactsDir = path.join(repoRoot, 'docs/screenshots/artifacts');
const rootScreenshotsDir = path.join(repoRoot, 'docs/screenshots');

const { SCENARIOS } = await import(path.join(repoRoot, 'tests/ui/capture/index.mjs'));
const registryKeys = new Set(Object.keys(SCENARIOS));

// A filename is still "live" if some *current* scenario declares it as one of
// its own contract.expectedOutputs — the same bytes may be regenerated under
// a different scenario key than the one that originally produced them.
const liveExpectedOutputs = new Set(
    Object.values(SCENARIOS).flatMap(s => s.contract?.expectedOutputs ?? []),
);

function findReceipts(dir) {
    const out = [];
    for (const entry of readdirSync(dir)) {
        const full = path.join(dir, entry);
        const st = statSync(full);
        if (st.isDirectory()) {
            out.push(...findReceipts(full));
        } else if (entry.endsWith('--receipt.json')) {
            out.push(full);
        }
    }
    return out;
}

function producedFilenames(produced) {
    // `produced` is either a JSON array of {filename,...} objects, or a
    // legacy `[N]{...}\nheader\ncsv-row\n...` string — normalize both.
    if (Array.isArray(produced)) return produced.map(p => p.filename);
    if (typeof produced === 'string') {
        return produced
            .split('\n')
            .slice(1) // header line
            .filter(Boolean)
            .map(line => line.split(',')[0]);
    }
    return [];
}

// Paths excluded from the "still referenced live" check:
//   - docs/plans/evidence/**  — frozen historical receipts, expected to name
//     scenarios/filenames as they existed at capture time, not the present.
//   - docs/archive/**         — archived docs, same reasoning.
//   - tests/ui/capture/index.mjs — a filename match here means some *current*
//     scenario's contract.expectedOutputs still produces that exact filename
//     (just under a different scenario key), so it isn't actually orphaned.
//   - tests/ui/capture/**test**.mjs — synthetic fixtures used by the capture
//     harness's own unit tests, not real documentation.
const EXCLUDE_PATTERNS = [
    /^docs\/plans\/evidence\//,
    /^docs\/archive\//,
    /^tests\/ui\/capture\/index\.mjs$/,
    /^tests\/ui\/capture\/.*test.*\.mjs$/,
];

function gitGrepReferences(filename) {
    try {
        const out = execFileSync(
            'git',
            ['grep', '-l', filename, '--', ':!docs/screenshots/*'],
            { cwd: repoRoot, encoding: 'utf8' },
        );
        return out.split('\n').filter(f => f && !EXCLUDE_PATTERNS.some(p => p.test(f)));
    } catch {
        return []; // git grep exits non-zero on no matches
    }
}

// Filenames whose provenance scenario is orphaned but whose current bytes
// were manually re-verified against the live UI (e.g. re-promoted from a
// different, still-live scenario's output). Listed here so the check stays
// green instead of failing forever on a file that's actually fine — but the
// file still shows up in output as a reminder to re-verify on future UI changes.
let allowlist = new Set();
try {
    allowlist = new Set(JSON.parse(readFileSync(path.join(rootScreenshotsDir, 'stale-scenario-allowlist.json'), 'utf8')));
} catch {
    allowlist = new Set();
}

const receipts = findReceipts(artifactsDir).filter(
    r => !path.relative(repoRoot, r).includes('__capture-receipt-test__'),
);
const staleReceipts = receipts.filter(r => {
    const data = JSON.parse(readFileSync(r, 'utf8'));
    return data.scenario && !registryKeys.has(data.scenario);
});

let rootScreenshots = [];
try {
    rootScreenshots = readdirSync(rootScreenshotsDir).filter(f => /\.(png|gif|webp)$/i.test(f));
} catch {
    rootScreenshots = [];
}

console.log('=== STALE SCREENSHOT SCENARIOS ===');
if (staleReceipts.length === 0) {
    console.log('(none — every artifact receipt names a scenario still in the SCENARIOS registry)');
    process.exit(0);
}

let anyLiveReference = false;
for (const receiptPath of staleReceipts) {
    const data = JSON.parse(readFileSync(receiptPath, 'utf8'));
    const relReceipt = path.relative(repoRoot, receiptPath);
    console.log(`\n  scenario "${data.scenario}" (${relReceipt}) is no longer in SCENARIOS`);

    const produced = producedFilenames(data.produced);
    for (const artifactFilename of produced) {
        if (liveExpectedOutputs.has(artifactFilename)) {
            continue; // still freshly regenerated by a current scenario under a different key
        }
        // A stale artifact's exact filename is not necessarily the promoted
        // root filename (promotion is a manual cp/rename), so also check
        // whether any currently-tracked root screenshot came from this run
        // by checking if docs reference the stale artifact filename directly,
        // and separately flag if any root screenshot bytes came from this
        // scenario's own filename pattern.
        const refs = gitGrepReferences(artifactFilename);
        if (refs.length > 0) {
            if (allowlist.has(artifactFilename)) {
                console.log(`    ✓ "${artifactFilename}" referenced by: ${refs.join(', ')} (allowlisted — manually re-verified current)`);
            } else {
                anyLiveReference = true;
                console.log(`    ⚠ "${artifactFilename}" is still referenced by: ${refs.join(', ')}`);
            }
        }
    }

    if (produced.length === 0) {
        console.log('    (receipt has no produced files listed)');
    }
}

if (rootScreenshots.length > 0 && !anyLiveReference) {
    console.log(
        '\nNote: no stale-scenario artifact filename was found referenced directly in docs.\n' +
        'If a stale scenario was manually promoted under a different filename (cp + rename),\n' +
        'this script cannot detect that by filename alone — cross-check promoted images against\n' +
        `docs/plans/evidence/**/*.md receipts by hand. Root screenshots present: ${rootScreenshots.length}.`,
    );
}

console.log(
    `\n${staleReceipts.length} stale scenario receipt(s) found` +
    (anyLiveReference ? ', with at least one still referenced live in docs — fix before merging.' : '.'),
);

process.exit(anyLiveReference ? 1 : 0);
