// ── Config ────────────────────────────────────────────────────────────────────
// Config modal, GPU environment, and config save.

import { showToast } from './toast.js';
import { openDeferredFileBrowser } from './file-browser-launcher.js';
import { collectSettings, closeSettingsModal } from './settings.js';
import { settingsState } from '../core/app-state.js';

// ── Config Modal ──────────────────────────────────────────────────────────────

export function openConfigModal() {
    closeSettingsModal();
    document.getElementById('config-modal').classList.add('open');
}

export function closeConfigModal() {
    document.getElementById('config-modal').classList.remove('open');
}

// ── Save Config ───────────────────────────────────────────────────────────────

function saveConfig() {
    clearTimeout(settingsState.saveTimer);
    fetch('/api/settings', {
        method: 'PUT',
        headers: window.authHeaders
            ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
            : { 'Content-Type': 'application/json' },
        body: JSON.stringify(collectSettings()),
    }).catch(() => {});

    const env = {
        arch: document.getElementById('gpu-env-arch').value,
        devices: document.getElementById('gpu-env-devices').value.trim(),
        rocm_path: document.getElementById('gpu-env-rocm-path').value.trim() || '/opt/rocm',
        extra_env: [],
    };
    fetch('/api/gpu-env', {
        method: 'PUT',
        headers: window.authHeaders
            ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
            : { 'Content-Type': 'application/json' },
        body: JSON.stringify(env),
    }).catch(() => {});

    closeConfigModal();
    showToast('Configuration saved', 'success');
}

function useDefaultServerBinary() {
    const input = document.getElementById('set-server-path');
    if (input) input.value = '';
    showToast('Using Foundry default binary location', 'info');
}

// ── Public API ────────────────────────────────────────────────────────────────

export function initConfig() {
    const configModal = document.getElementById('config-modal');
    if (configModal) {
        configModal.addEventListener('click', e => {
            if (e.target === e.currentTarget) closeConfigModal();
        });
    }

    // Bind config modal buttons
    const configClose = document.getElementById('config-modal-close');
    if (configClose) configClose.addEventListener('click', closeConfigModal);

    const configCancel = document.getElementById('config-modal-cancel');
    if (configCancel) configCancel.addEventListener('click', closeConfigModal);

    const configSave = document.getElementById('config-modal-save');
    if (configSave) configSave.addEventListener('click', saveConfig);

    // Bind Browse buttons in config modal
    const browseServerPath = document.getElementById('config-browse-server-path');
    if (browseServerPath) browseServerPath.addEventListener('click', () => openDeferredFileBrowser('set-server-path', 'executable'));

    const usePathBtn = document.getElementById('config-use-path-btn');
    if (usePathBtn) usePathBtn.addEventListener('click', useDefaultServerBinary);

    const browseCwd = document.getElementById('config-browse-cwd');
    if (browseCwd) browseCwd.addEventListener('click', () => openDeferredFileBrowser('set-server-cwd', 'dir'));

    // Bind "Open Runtime Configuration" in settings modal (Loaders tab)
    const openConfigBtn = document.getElementById('settings-advanced-open-config-btn');
    if (openConfigBtn) openConfigBtn.addEventListener('click', openConfigModal);

}
