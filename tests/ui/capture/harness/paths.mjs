// Shared paths, ports, and constants for the capture harness.
// Extracted from the original tests/ui/capture.mjs (Phase A1).
import fs from 'fs';
import os from 'os';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

export const __filename = fileURLToPath(import.meta.url);
export const __dirname = dirname(__filename);
export const UI_DIR = join(__dirname, '../..');
export const ROOT_DIR = join(UI_DIR, '../..');
export const ARTIFACTS_DIR = join(ROOT_DIR, 'docs/screenshots/artifacts');
export const SCREENSHOTS_DIR = join(ROOT_DIR, 'docs/screenshots');
export const FRAME_DIR = join(UI_DIR, 'frames');
function populatedDirectory(path) {
    try { return fs.statSync(path).isDirectory() && fs.readdirSync(path).length > 0; } catch { return false; }
}

const configuredCaptureRoot = process.env.LLAMA_MONITOR_CAPTURE_CONFIG_DIR;
const configParent = process.platform === 'win32'
    ? (process.env.APPDATA || join(os.homedir(), 'AppData', 'Roaming'))
    : join(process.env.HOME || os.homedir(), '.config');
const canonicalCaptureRoot = join(configParent, 'local-llm-foundry');
const legacyCaptureRoot = join(configParent, 'llama-monitor');
// Follow the same cutover rule as AppPaths: an explicit override wins, then a
// populated canonical root, then the populated legacy root during migration.
export const REAL_APP_CONFIG_DIR = configuredCaptureRoot
    || (populatedDirectory(canonicalCaptureRoot) ? canonicalCaptureRoot : legacyCaptureRoot);
export const TEMP_HOME = fs.mkdtempSync(join(os.tmpdir(), 'llama-monitor-capture-'));
export const TEMP_CONFIG_HOME = join(TEMP_HOME, '.config');
export const TEMP_APP_CONFIG_DIR = join(TEMP_CONFIG_HOME, 'llama-monitor');
export const TEMP_WINDOWS_APPDATA = join(TEMP_HOME, 'AppData', 'Roaming');
export const TEMP_WINDOWS_LOCALAPPDATA = join(TEMP_HOME, 'AppData', 'Local');
export const SCREENSHOT_TAB_PREFIX = '[screenshot]';
export const DEFAULT_VIEWPORT = { width: 1440, height: 900, deviceScaleFactor: 1 };
export const DEFAULT_PORT = parseInt(process.env.SCREENSHOT_PORT || '8892', 10);
// Set RUNNING_PORT to connect to an already-running llama-monitor (e.g. your production instance
// with a remote agent connected). When set, no binary is spawned and no temp config is seeded.
// Example: RUNNING_PORT=8080 node tests/ui/capture.mjs --scenario dashboard
export const RUNNING_PORT = process.env.RUNNING_PORT ? parseInt(process.env.RUNNING_PORT, 10) : null;
export const REMOTE_SERVER = process.env.REMOTE_SERVER || 'http://192.168.2.16:8001';
const configuredAttachTimeout = Number.parseInt(process.env.CAPTURE_ATTACH_TIMEOUT_MS || '120000', 10);
export const CAPTURE_ATTACH_TIMEOUT_MS = Number.isFinite(configuredAttachTimeout) && configuredAttachTimeout > 0
    ? configuredAttachTimeout
    : 120000;
export const BINARY_PATH = join(ROOT_DIR, 'target/release/local-llm-foundry');
export const CAPTURE_FORM_AUTH = process.env.SCREENSHOT_FORM_AUTH || 'admin:secret123';
export const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

fs.mkdirSync(ARTIFACTS_DIR, { recursive: true });
fs.mkdirSync(SCREENSHOTS_DIR, { recursive: true });
fs.mkdirSync(TEMP_APP_CONFIG_DIR, { recursive: true });
fs.mkdirSync(TEMP_WINDOWS_APPDATA, { recursive: true });
fs.mkdirSync(TEMP_WINDOWS_LOCALAPPDATA, { recursive: true });

// Screenshots are grouped into category subdirectories (mirroring
// tests/ui/capture/scenarios/<category>/) instead of one flat 150+ file
// directory. index.mjs sets this before running each scenario; shot.mjs
// reads it to pick the output directory.
let activeCategory = null;
export function setArtifactCategory(category) {
    activeCategory = category || null;
}
export function currentArtifactsDir() {
    const dir = activeCategory ? join(ARTIFACTS_DIR, activeCategory) : ARTIFACTS_DIR;
    fs.mkdirSync(dir, { recursive: true });
    return dir;
}

// Runtime tag (which backend the captured UI is showing — or `neutral` for
// backend-independent chrome) gets woven into every filename so it's visible
// without opening the image: `<scenario>--<runtime>--<description>.ext`.
let activeScenarioKey = null;
let activeRuntimeTag = null;
export function setArtifactRuntime(scenarioKey, runtimeTag) {
    activeScenarioKey = scenarioKey || null;
    activeRuntimeTag = runtimeTag || null;
}

// Individual shots within a mixed-runtime scenario (e.g. spawn-wizard-engines
// showing both llama.cpp and Rapid-MLX cards) can override the scenario's
// default tag by passing { runtimeTag: '...' } into captureShot/etc.
export function tagFilename(filename, overrideRuntimeTag) {
    const runtimeTag = overrideRuntimeTag || activeRuntimeTag;
    if (!runtimeTag) return filename;
    const scenarioKey = activeScenarioKey;
    if (scenarioKey && filename.startsWith(`${scenarioKey}-`)) {
        const rest = filename.slice(scenarioKey.length + 1);
        return `${scenarioKey}--${runtimeTag}--${rest}`;
    }
    if (scenarioKey && filename === scenarioKey) {
        return `${scenarioKey}--${runtimeTag}`;
    }
    return `${runtimeTag}--${filename}`;
}
