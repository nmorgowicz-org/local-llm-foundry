#!/usr/bin/env node
// run-server.mjs — cross-platform replacement for run-server.sh.
// Creates a fresh temp config dir, starts the local-llm-foundry binary, and
// cleans up on exit. Works on Linux, macOS, and Windows.
//
// IMPORTANT (live dev instance):
// - Default port is 7778. When running tests locally while a live
//   llama-monitor (or AI coding session) is using 7778, ALWAYS set:
//     LLAMA_MONITOR_TEST_PORT=17778 (or another free port)
//   before running Playwright tests.
// - Never let tests kill, restart, or connect to the active 7778
//   instance; doing so will drop any running model session without
//   warning.

import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, '..', '..');

const testConfigDir = mkdtempSync(join(tmpdir(), 'llama-monitor-test-'));
const configPathFile = join(tmpdir(), 'llama-monitor-test-config-path');

console.log(`[run-server] Fresh config dir: ${testConfigDir}`);

try {
    writeFileSync(configPathFile, testConfigDir, 'utf8');
} catch {
    // Non-critical — used only by manual debugging scripts
}

function cleanup() {
    try { rmSync(testConfigDir, { recursive: true, force: true }); } catch { /* ignore */ }
    try { rmSync(configPathFile, { force: true }); } catch { /* ignore */ }
}

const extraArgs = (process.env.LLAMA_MONITOR_TEST_ARGS || '').split(/\s+/).filter(Boolean);

// LLAMA_MONITOR_USE_RELEASE=1 skips `cargo run` and uses the pre-built release binary.
// Useful for local test runs to avoid long build times.
const useRelease = process.env.LLAMA_MONITOR_USE_RELEASE === '1';

// LLAMA_MONITOR_TEST_PORT:
// - Override the test server port when a live instance is already using 7778.
// - CRITICAL: Do NOT leave this at 7778 while a coding session is running there.
const testPort = process.env.LLAMA_MONITOR_TEST_PORT || '7778';

const testEnv = { ...process.env };

// Force Rapid-MLX to report as locally available for UI tests on Linux/CI runners.
// This allows the spawn wizard to show the Rapid-MLX engine card so tests can
// exercise that flow without requiring a real macOS/Apple Silicon runner.
testEnv.LLAMA_MONITOR_FAKE_RAPID_MLX_LOCAL_AVAILABLE = '1';

let child;
if (useRelease) {
    const binaryPath = join(repoRoot, 'target', 'release', process.platform === 'win32' ? 'local-llm-foundry.exe' : 'local-llm-foundry');
    child = spawn(binaryPath, [
        '--headless',
        '--port', testPort,
        '--config-dir', testConfigDir,
        ...extraArgs,
    ], {
        cwd: repoRoot,
        stdio: 'inherit',
        env: testEnv,
    });
} else {
    const cargoArgs = [
        'run',
        '--bin', 'local-llm-foundry',
        '--',
        '--headless',
        '--port', testPort,
        '--config-dir', testConfigDir,
        ...extraArgs,
    ];
    child = spawn('cargo', cargoArgs, {
        cwd: repoRoot,
        stdio: 'inherit',
        env: testEnv,
        // On Windows, spawn needs shell:false but cargo must be on PATH — which it is
        // after a standard rustup install. No shell:true needed.
    });
}

child.on('error', err => {
    console.error(`[run-server] Failed to start cargo: ${err.message}`);
    cleanup();
    process.exit(1);
});

child.on('exit', (code, signal) => {
    cleanup();
    if (signal) {
        process.kill(process.pid, signal);
    } else {
        process.exit(code ?? 0);
    }
});

for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP']) {
    process.on(sig, () => {
        child.kill(sig);
    });
}

// Windows: handle CTRL_C_EVENT
if (process.platform === 'win32') {
    process.on('SIGBREAK', () => child.kill('SIGTERM'));
}
