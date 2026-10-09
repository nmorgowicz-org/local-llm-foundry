import assert from 'node:assert/strict';
import test from 'node:test';
import { SCENARIOS, capturePlatformSkipReason } from './index.mjs';

test('Rapid capture scenarios require Apple Silicon macOS', () => {
    const rapid = SCENARIOS['rapid-preset'];
    assert.match(
        capturePlatformSkipReason(rapid, { platform: 'win32', arch: 'x64' }),
        /requires Apple Silicon macOS/,
    );
    assert.match(
        capturePlatformSkipReason(rapid, { platform: 'linux', arch: 'x64' }),
        /requires Apple Silicon macOS/,
    );
    assert.equal(capturePlatformSkipReason(rapid, { platform: 'darwin', arch: 'arm64' }), null);
});

test('non-Rapid captures remain cross-platform', () => {
    assert.equal(
        capturePlatformSkipReason(SCENARIOS.welcome, { platform: 'win32', arch: 'x64' }),
        null,
    );
    assert.equal(
        capturePlatformSkipReason(SCENARIOS.dashboard, { platform: 'win32', arch: 'x64' }),
        null,
    );
});
