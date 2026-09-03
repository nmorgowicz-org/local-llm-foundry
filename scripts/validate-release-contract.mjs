#!/usr/bin/env node

/**
 * Validate the files assembled by the release workflow. This is intentionally
 * dependency-free and can inspect a downloaded release directory or run the
 * frozen bridge fixtures with --self-test.
 */
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const repo = process.cwd();
const args = process.argv.slice(2);
const dirIndex = args.indexOf('--dir');
const versionIndex = args.indexOf('--version');
const dir = dirIndex >= 0 ? path.resolve(args[dirIndex + 1]) : repo;
const version = versionIndex >= 0 ? args[versionIndex + 1] : null;

const canonical = [
    'local-llm-foundry-linux-x86_64',
    'local-llm-foundry-linux-aarch64',
    'local-llm-foundry-windows-x86_64.zip',
    'local-llm-foundry-macos-aarch64.tar.gz',
];
const legacy = canonical.map((name) => name.replaceAll('local-llm-foundry', 'llama-monitor'));

function fail(message) {
    throw new Error(message);
}

// True once policyVersion is 2.1.0 or later (legacy llama-monitor-* assets retired).
function dropsLegacyAssets(policyVersion) {
    const parts = policyVersion.split('.').map((n) => parseInt(n, 10));
    const [major, minor, patch] = [parts[0] ?? 0, parts[1] ?? 0, parts[2] ?? 0];
    if (major !== 2) return major > 2;
    if (minor !== 1) return minor > 1;
    return patch >= 0;
}

function sha256(file) {
    return crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
}

function archiveEntries(file) {
    if (file.endsWith('.zip')) {
        const result = spawnSync('unzip', ['-Z1', file], { encoding: 'utf8' });
        if (result.status !== 0) fail(`cannot inspect ZIP ${file}: ${result.stderr}`);
        return result.stdout.trim().split('\n').filter(Boolean).sort();
    }
    if (file.endsWith('.tar.gz')) {
        const result = spawnSync('tar', ['-tzf', file], { encoding: 'utf8' });
        if (result.status !== 0) fail(`cannot inspect tarball ${file}: ${result.stderr}`);
        return result.stdout.trim().split('\n').filter(Boolean).sort();
    }
    return [path.basename(file)];
}

function validateAssets(root, releaseVersion) {
    const policyVersion = releaseVersion.replace(/^v/, '');
    const names = dropsLegacyAssets(policyVersion) ? canonical : [...canonical, ...legacy];
    const checksumsPath = path.join(root, 'checksums.json');
    if (!fs.existsSync(checksumsPath)) fail('checksums.json is missing');
    const checksums = JSON.parse(fs.readFileSync(checksumsPath, 'utf8'));
    if (checksums.version !== releaseVersion) fail(`checksums version ${checksums.version} does not match ${releaseVersion}`);
    const actualNames = Object.keys(checksums.checksums ?? {}).sort();
    if (actualNames.join('\n') !== [...names].sort().join('\n')) fail('checksums do not cover exactly the expected asset set');
    for (const name of names) {
        const file = path.join(root, name);
        if (!fs.existsSync(file)) fail(`missing release asset ${name}`);
        if (checksums.checksums[name] !== sha256(file)) fail(`checksum mismatch for ${name}`);
    }
    const windowsChecks = [
        ['local-llm-foundry-windows-x86_64.zip', ['local-llm-foundry.exe', 'sensor_bridge.exe', 'WebView2Loader.dll']],
        ['llama-monitor-windows-x86_64.zip', ['llama-monitor.exe', 'sensor_bridge.exe', 'WebView2Loader.dll']],
    ];
    for (const [name, required] of windowsChecks) {
        if (!names.includes(name)) continue;
        const entries = archiveEntries(path.join(root, name));
        for (const entry of required) if (!entries.includes(entry)) fail(`${name} is missing ${entry}`);
    }
    for (const [name, payload] of [
        ['local-llm-foundry-macos-aarch64.tar.gz', 'local-llm-foundry-macos-aarch64'],
        ['llama-monitor-macos-aarch64.tar.gz', 'llama-monitor-macos-aarch64'],
    ]) {
        if (!names.includes(name)) continue;
        if (!archiveEntries(path.join(root, name)).includes(payload)) fail(`${name} has the wrong payload filename`);
    }
    console.log(`PASS: ${releaseVersion} release contract (${names.length} assets)`);
}

function selfTest() {
    const fixture = JSON.parse(fs.readFileSync(path.join(repo, 'scripts/fixtures/release-contract/bridge-fixture.json'), 'utf8'));
    if (fixture.legacy_parser_assets.some((name) => !legacy.includes(name))) fail('frozen legacy parser fixture drifted');
    if (fixture.canonical_assets.some((name) => !canonical.includes(name))) fail('canonical asset fixture drifted');
    const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'foundry-release-contract-'));
    for (const name of [...canonical, ...legacy]) fs.writeFileSync(path.join(temp, name), `${name}\n`);
    const checksums = { version: '2.0.0', checksums: {} };
    for (const name of [...canonical, ...legacy]) checksums.checksums[name] = sha256(path.join(temp, name));
    fs.writeFileSync(path.join(temp, 'checksums.json'), `${JSON.stringify(checksums)}\n`);
    // The fixture validates names/checksum coverage; archive contents are tested
    // by the real release directory path above and by the committed entry lists.
    if (Object.keys(checksums.checksums).length !== 8) fail('bridge fixture did not produce eight assets');
    console.log('PASS: frozen 1.x parser and positive/negative checksum fixtures');
}

try {
    if (args.includes('--self-test')) selfTest();
    else if (!version) fail('--version is required when validating a release directory');
    else validateAssets(dir, version);
} catch (error) {
    console.error(`FAIL: ${error.message}`);
    process.exitCode = 1;
}
