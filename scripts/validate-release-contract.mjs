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

function fail(message) {
    throw new Error(message);
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
    const names = canonical;
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
    ];
    for (const [name, required] of windowsChecks) {
        if (!names.includes(name)) continue;
        const entries = archiveEntries(path.join(root, name));
        for (const entry of required) if (!entries.includes(entry)) fail(`${name} is missing ${entry}`);
    }
    for (const [name, payload] of [
        ['local-llm-foundry-macos-aarch64.tar.gz', 'local-llm-foundry-macos-aarch64'],
    ]) {
        if (!names.includes(name)) continue;
        if (!archiveEntries(path.join(root, name)).includes(payload)) fail(`${name} has the wrong payload filename`);
    }
    console.log(`PASS: ${releaseVersion} release contract (${names.length} assets)`);
}

function selfTest() {
    const fixture = JSON.parse(fs.readFileSync(path.join(repo, 'scripts/fixtures/release-contract/bridge-fixture.json'), 'utf8'));
    if (fixture.canonical_assets.some((name) => !canonical.includes(name))) fail('canonical asset fixture drifted');
    const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'foundry-release-contract-'));
    try {
        const payload = path.join(temp, 'payload');
        fs.mkdirSync(payload);
        for (const name of ['local-llm-foundry.exe', 'sensor_bridge.exe', 'WebView2Loader.dll', 'local-llm-foundry-macos-aarch64']) {
            fs.writeFileSync(path.join(payload, name), `${name}\n`);
        }
        const zip = spawnSync('zip', ['-j', path.join(temp, canonical[2]), ...['local-llm-foundry.exe', 'sensor_bridge.exe', 'WebView2Loader.dll'].map((name) => path.join(payload, name))]);
        const tar = spawnSync('tar', ['-czf', path.join(temp, canonical[3]), '-C', payload, 'local-llm-foundry-macos-aarch64']);
        if (zip.status !== 0 || tar.status !== 0) fail('could not assemble release fixture archives');
        for (const name of canonical.slice(0, 2)) fs.writeFileSync(path.join(temp, name), `${name}\n`);
        const checksums = { version: '2.1.2', checksums: {} };
        for (const name of canonical) checksums.checksums[name] = sha256(path.join(temp, name));
        const save = () => fs.writeFileSync(path.join(temp, 'checksums.json'), `${JSON.stringify(checksums)}\n`);
        save();
        validateAssets(temp, checksums.version);
        checksums.checksums['llama-monitor-linux-x86_64'] = 'retired';
        save();
        let rejected = false;
        try { validateAssets(temp, checksums.version); } catch { rejected = true; }
        if (!rejected) fail('legacy asset unexpectedly accepted');
        delete checksums.checksums['llama-monitor-linux-x86_64'];
        checksums.checksums[canonical[0]] = 'bad checksum';
        save();
        rejected = false;
        try { validateAssets(temp, checksums.version); } catch { rejected = true; }
        if (!rejected) fail('invalid checksum unexpectedly accepted');
        console.log('PASS: canonical archive/checksum fixtures and legacy rejection');
    } finally {
        fs.rmSync(temp, { recursive: true, force: true });
    }
}

try {
    if (args.includes('--self-test')) selfTest();
    else if (!version) fail('--version is required when validating a release directory');
    else validateAssets(dir, version);
} catch (error) {
    console.error(`FAIL: ${error.message}`);
    process.exitCode = 1;
}
