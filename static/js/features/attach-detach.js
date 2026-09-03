// ── Attach / Detach / Start / Stop ─────────────────────────────────────────────
// LLM lifecycle: start, stop, attach, detach, kill.

import { sessionState, setupViewState } from '../core/app-state.js';
import { updateActiveSessionInfo } from './sessions.js';
import { showToast, showToastWithActions } from './toast.js';
import { saveSettings } from './settings.js';
import { hideConnectingState, saveLastSessionData, showConnectingState, switchView, restorePreviousPosition, savePreviousPosition } from './setup-view.js';
import Router from './router.js';
import { _showConfirm } from './presets.js';
import { setTuneConfig, showTunePanel, hideTunePanel } from './tune-panel.js';
import { hideDisconnectedBanner } from './chat-transport.js';
import { monitorState } from '../core/app-state.js';
import { waitForSpawnReadiness } from './spawn-readiness.js';

// ── Config ─────────────────────────────────────────────────────────────────────

export function getConfig() {
    const id = document.getElementById('preset-select').value;
    return { preset_id: id };
}

function hasModelSource(config) {
    if (config.backend === 'rapid_mlx') {
        const rapidMlx = config.rapid_mlx;
        return !!rapidMlx?.model_source_view;
    }
    return !!(config.model_path || config.hf_repo);
}

// ── Start / Stop ───────────────────────────────────────────────────────────────

// Fetch the db-admin-token needed for V2 spawn endpoints.
async function _fetchDbAdminToken() {
    const tokenResp = await fetch('/api/db/admin-token', {
        headers: window.authHeaders ? window.authHeaders() : {},
    });
    const tokenData = tokenResp.ok ? await tokenResp.json().catch(() => ({})) : {};
    return tokenData.token || null;
}

export async function doRestoreSession(sessionId, apiKey = null) {
    const adminToken = await _fetchDbAdminToken();
    if (!adminToken) throw new Error('Authentication required');
    const body = { session_id: sessionId };
    if (apiKey) body.api_key = apiKey;
    const response = await fetch('/api/sessions/spawn', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${adminToken}`,
        },
        body: JSON.stringify(body),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.ok) throw new Error(data.error || 'Restore failed');
    await waitForSpawnReadiness(data.port);
    return data;
}

// Disable start buttons for N seconds, showing countdown.
function applyCooldown(seconds, button) {
    if (!seconds || seconds <= 0 || !button) return;
    button.disabled = true;
    const remaining = seconds;
    const label = button.textContent || button.value || '';
    const orig = label;
    const interval = setInterval(() => {
        if (seconds <= 0) {
            clearInterval(interval);
            button.disabled = false;
            button.textContent = orig;
            return;
        }
        button.textContent = `Wait ${seconds}s`;
        seconds--;
    }, 1000);
}

export async function doStart(cooldownBtn, options = {}) {
    const buttonArg = cooldownBtn instanceof Event ? null : cooldownBtn;
    const config = getConfig();
    const preset = sessionState.presets.find(item => item.id === config.preset_id);
    const rapidMlx = preset?.rapid_mlx;
    const hasSource = preset?.backend === 'rapid_mlx'
        ? !!rapidMlx?.model_source_view
        : !!(preset?.model_path || preset?.hf_repo);
    if (!hasSource) {
        showToast('No model source set. Edit the preset to select a local model or HuggingFace repo.', 'error');
        return;
    }
    return doStartWithConfig(config, options, buttonArg);
}

export async function doStartWithConfig(config, options = {}, buttonArg = null) {
    const { skipRunningConfirm = false } = options;
    if (!config.preset_id && !hasModelSource(config)) {
        showToast('No model source set.', 'error');
        return;
    }

    const btnStart = document.getElementById('btn-start');
    if (btnStart) btnStart.disabled = true;

    try {
        if (!skipRunningConfirm) {
            const activeResp = await fetch('/api/sessions/active', {
                headers: window.authHeaders ? window.authHeaders() : {},
            }).catch(() => null);
            const active = activeResp?.ok ? await activeResp.json().catch(() => ({})) : {};
            const activeStatus = String(active.status || '').toLowerCase();
            const activeMode = String(active.mode || '').toLowerCase();
            const activePresetId = active.preset_id || '';
            // Only prompt if the API confirms a *different* preset is actively running.
            // Guard against stale state: mode 'off' or 'sleep' means the server stopped.
            const actuallyRunning = activeStatus === 'running'
                && activeMode !== 'off'
                && activeMode !== 'sleep';
            if (actuallyRunning && activePresetId && activePresetId !== config.preset_id) {
                const ok = await _showConfirm(
                    'Switch preset',
                    'A different preset is already running. Stop it and start the selected preset?'
                );
                if (!ok) {
                    if (btnStart) btnStart.disabled = false;
                    return;
                }
            }
        }

        await doKillLlamaInternal();

        const adminToken = await _fetchDbAdminToken();
        if (!adminToken) {
            showToast('Failed: authentication required', 'error');
            hideConnectingState();
            return;
        }

        // Phase 8a: for a bundle the launch is resolve-and-launch, so we ship only
        // the preset id + expected_revision (never the selection). Flat presets omit
        // expected_revision entirely to keep the legacy payload byte-for-byte.
        const spawnBody = { ...config };
        if (options.expectedRevision != null) spawnBody.expected_revision = options.expectedRevision;
        const resp = await fetch('/api/sessions/spawn', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${adminToken}`,
            },
            body: JSON.stringify(spawnBody),
        });

        if (!resp.ok) {
            if (resp.status === 429) {
                try {
                    const data = await resp.json().catch(() => null);
                    const wait = data?.seconds_remaining || data?.error || '';
                    if (typeof wait === 'number') {
                        showToast('Start failed: Please wait ' + wait + 's', 'warning');
                        applyCooldown(wait, buttonArg || btnStart);
                    } else {
                        showToast('Start failed: too soon; please wait', 'warning');
                    }
                } catch {
                    showToast('Start failed: too soon; please wait', 'warning');
                }
                hideConnectingState();
                return;
            }

            // Phase 8a resolve-and-launch refusals. These mean the saved selection or
            // the bundle revision moved under us — one-shot re-render + re-ask, never
            // a silent retry with a stale revision (architecture invariant: 409 is
            // terminal, the user re-asks explicitly).
            if (resp.status === 409 || resp.status === 412 || resp.status === 400) {
                let serverError = '';
                try { serverError = (await resp.json().catch(() => ({}))).error || ''; } catch { /* non-JSON */ }
                let message;
                if (resp.status === 409 || serverError === 'revision_conflict') {
                    message = 'This preset changed since it was last loaded. Reloaded the current selection — review and start again.';
                } else if (resp.status === 412 || serverError === 'preview_stale') {
                    message = 'The selection preview is stale. Reloaded the latest settings — start again.';
                } else {
                    message = 'This selection can no longer launch with the current system settings. Start again to review the current configuration.';
                }
                showToast(message, 'warning', serverError, { duration: 8000 });
                hideConnectingState();
                import('./presets.js')
                    .then(({ loadPresets }) => loadPresets())
                    .catch(() => {});
                return;
            }

            const text = await resp.text().catch(() => 'Request failed');
            showToast('Start failed: ' + text, 'error');
            hideConnectingState();
            return;
        }

        const data = await resp.json().catch(() => ({}));

        if (!data.ok) {
            showToast('Start failed: ' + (data.error || 'server responded with an error'), 'error');
            hideConnectingState();
            return;
        }

        const backendLabel = data.backend === 'rapid_mlx' ? 'Rapid-MLX' : 'llama-server';
        const launchPort = data.port ?? config.port;
        showToast(`Starting ${backendLabel}…`, 'info', 'Loading model on port ' + launchPort, { duration: 12000 });
        await waitForSpawnReadiness(launchPort);

        showToast(`${backendLabel} is running`, 'success', '', { duration: 6000 });
        const resolvedPreset = config.preset_id
            ? sessionState.presets.find(item => item.id === config.preset_id)
            : null;
        setTuneConfig(resolvedPreset || config);
        showTunePanel();
        setHeaderMode('Spawn:' + launchPort);
        Router.navigate('/server');
        hideConnectingState();
        saveSettings();
        setTimeout(() => restorePreviousPosition(), 600);
    } catch (e) {
        const msg = (e.message || 'network or server error').split('\n')[0].trim();
        showToast('Start failed: ' + msg, 'error');
        hideConnectingState();
    } finally {
        if (btnStart) btnStart.disabled = false;
    }
}

export async function doStop() {
    const btnStop = document.getElementById('btn-stop');
    if (btnStop) btnStop.disabled = true;

    // Capture the preset that was running before killing so we can offer restart
    const stoppedPresetId = sessionState.activeSessionPresetId || '';
    const stoppedPreset = sessionState.presets?.find(p => p.id === stoppedPresetId);
    const stoppedName = stoppedPreset?.name || 'server';

    await doKillLlamaInternal();
    sessionState.activeSessionPresetId = '';
    hideTunePanel();
    setHeaderMode(null);
    window.__presetUserSelected = false;

    if (btnStop) btnStop.disabled = false;

    const actions = [];

    if (stoppedPresetId) {
        actions.push({
            id: 'restart',
            label: 'Restart',
            primary: true,
            handler: async () => {
                const { syncSelectedPresetSelection } = await import('./presets.js');
                syncSelectedPresetSelection(stoppedPresetId, { userIntent: true, persist: true });
                doStart(null, { skipRunningConfirm: true });
            },
        });
    }

    actions.push({
        id: 'home',
        label: '↩ Home',
        primary: false,
        handler: () => Router.navigate('/'),
    });

    showToastWithActions(
        stoppedName + ' stopped',
        'info',
        stoppedPresetId ? 'Restart with the same preset, or return home.' : 'Return home to start a new session.',
        actions,
        { duration: 10000 },
    );
}

// ── Kill ───────────────────────────────────────────────────────────────────────

export async function doKillLlama() {
    if (!await _showConfirm('Stop server', 'Kill all running inference server processes?')) return;

    const btnKill = document.getElementById('btn-kill');
    if (btnKill) btnKill.disabled = true;

    try {
        const tokenResp = await fetch('/api/db/admin-token', {
            headers: window.authHeaders ? window.authHeaders() : {},
        });
        const tokenData = await tokenResp.json();
        const token = tokenData.token;
        if (!token) {
            showToast('Kill failed: admin token not available', 'error');
            return;
        }

        const resp = await fetch('/api/kill-server', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`,
            },
            body: JSON.stringify({ confirm: 'kill' }),
        });

        const data = await resp.json();

        if (!data.ok) {
            if (resp.status === 429) {
                const wait = data?.seconds_remaining
                    ? `Too soon; please wait ${data.seconds_remaining}s`
                    : 'Too soon; please wait';
                showToast('Kill failed: ' + wait, 'warning');
                if (data?.seconds_remaining) {
                    applyCooldown(data.seconds_remaining, btnKill);
                }
            } else {
                showToast('Kill failed: ' + (data.error || 'unknown'), 'error');
            }
        } else {
            showToast('Inference server killed', 'success');
            hideTunePanel();
        }
    } catch (e) {
        showToast('Kill failed: ' + e.message, 'error');
    } finally {
        if (btnKill) btnKill.disabled = false;
    }
}

export async function doKillLlamaInternal() {
    try {
        const tokenResp = await fetch('/api/db/admin-token', {
            headers: window.authHeaders ? window.authHeaders() : {},
        });
        const tokenData = await tokenResp.json();
        const token = tokenData.token;
        if (!token) return;

        await fetch('/api/kill-server', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`,
            },
            body: JSON.stringify({ confirm: 'kill' }),
        });
    } catch(e) {
        // Ignore errors from kill, just try to continue
    }
}

// ── Attach / Detach ────────────────────────────────────────────────────────────

export async function doAttach(options = {}) {
    const endpointInput = document.getElementById('server-endpoint');
    const endpoint = endpointInput.value.trim();

    if (!endpoint) {
        showToast('Please enter a server endpoint', 'error');
        return;
    }

    // Read API key from welcome screen or monitor view input
    const apiKeyInput = document.getElementById('setup-endpoint-api-key');
    const apiKey = apiKeyInput ? apiKeyInput.value.trim() : '';

    const backend = options.backend || 'llama_cpp';
    const modelIdentity = options.modelIdentity?.trim() || '';
    const resp = await fetch('/api/attach', {
        method: 'POST',
        headers: (typeof window.authHeaders === 'function')
            ? window.authHeaders({ 'Content-Type': 'application/json' })
            : { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            endpoint,
            api_key: apiKey || undefined,
            backend,
            model_identity: modelIdentity || undefined,
        }),
    });
    const data = await resp.json();

    if (!data.ok) {
        showToast('Attach failed: ' + (data.error || 'unknown'), 'error');
        hideConnectingState();
    } else {
        showToast('Attached to server', 'success');
        hideConnectingState();

        if (data.warning) {
            showToast(data.warning, 'warning');
        }

        setHeaderMode('Attach:' + (document.getElementById('server-endpoint')?.value?.trim() || ''));

        monitorState.speedMax = { prompt: 0, generation: 0 };
        hideDisconnectedBanner();
        Router.navigate('/server');
        showTunePanel();
        setTimeout(() => restorePreviousPosition(), 600);
    }

    updateActiveSessionInfo();
}

export async function doDetach() {
    try {
        const headers = window.authHeaders
            ? window.authHeaders({ 'Content-Type': 'application/json' })
            : { 'Content-Type': 'application/json' };
        const resp = await fetch('/api/detach', { method: 'POST', headers });
        if (resp.status === 401) {
            showToast('Detach failed: authentication required', 'error');
            return;
        }
        const data = await resp.json();

        if (!data.ok) {
            showToast('Detach failed: ' + (data.error || 'unknown'), 'error');
        } else {
            showToast('Detached from server', 'success');

            saveLastSessionData({
                promptRate: monitorState.speedMax.prompt > 0 ? monitorState.speedMax.prompt + ' t/s' : '—',
                genRate: monitorState.speedMax.generation > 0 ? monitorState.speedMax.generation + ' t/s' : '—',
                sessionName: sessionState.activeSessionId || '—',
                endpoint: document.getElementById('server-endpoint')?.value?.trim() || '',
                telemetryGrade: window.__telemetryGrade || '',
                telemetryLabel: document.getElementById('telemetry-grade-chip')?.textContent?.trim() || '',
            });

            const btnAttach = document.getElementById('btn-attach');
            const btnDetach = document.getElementById('btn-detach');
            const btnDetachTop = document.getElementById('btn-detach-top');

            if (btnAttach && btnDetach) {
                btnAttach.style.display = 'inline-block';
                btnDetach.style.display = 'none';
            }
            if (btnDetachTop) btnDetachTop.style.display = 'none';

            const serverHeader = document.getElementById('server-header');
            if (serverHeader) serverHeader.style.display = '';

            const historicBadge = document.getElementById('inference-historic-badge');
            if (historicBadge) historicBadge.style.display = 'inline-block';

            monitorState.speedMax = { prompt: 0, generation: 0 };
            hideTunePanel();
            Router.navigate('/');
        }

        updateActiveSessionInfo();
    } catch (err) {
        showToast('Detach failed: ' + err.message, 'error');
    }
}

// ── Setup page helpers ─────────────────────────────────────────────────────────

export async function doAttachFromSetup() {
    const input = document.getElementById('setup-endpoint-url');
    const url = input ? input.value.trim() : '';
    if (!url) {
        input?.focus();
        return;
    }
    const serverEndpoint = document.getElementById('server-endpoint');
    if (serverEndpoint) serverEndpoint.value = url;
    localStorage.setItem('llama-monitor-last-endpoint', url);

    const btn = document.getElementById('setup-attach-btn');
    if (btn) {
        btn.disabled = true;
        btn.textContent = 'Connecting...';
    }

    showConnectingState();
    const backend = document.getElementById('setup-endpoint-backend')?.value || 'llama_cpp';
    const modelIdentity = document.getElementById('setup-endpoint-model')?.value || '';
    await doAttach({ backend, modelIdentity });

    if (btn) {
        btn.disabled = false;
        btn.textContent = 'Connect';
    }
}

export function doStartFromSetup(options = {}) {
    const select = document.getElementById('setup-preset-select');
    if (select) {
        const presetSelect = document.getElementById('preset-select');
        if (presetSelect) presetSelect.value = select.value;
    }
    showConnectingState();
    doStart(document.getElementById('setup-start-btn'), options);
}

// ── Button init ────────────────────────────────────────────────────────────────

// Update the server header visibility based on session mode.
// Call this whenever session mode changes (spawn, attach, detach, init).
export function setHeaderMode(mode) {
    const serverHeader = document.getElementById('server-header');
    const btnAttach = document.getElementById('btn-attach');
    const btnDetach = document.getElementById('btn-detach');
    if (!serverHeader) return;
    if (mode && mode.startsWith('Spawn:')) {
        serverHeader.style.display = 'none';
    } else if (mode && mode.startsWith('Attach:')) {
        serverHeader.style.display = '';
        if (btnAttach) btnAttach.style.display = 'none';
        if (btnDetach) btnDetach.style.display = 'inline-block';
    } else {
        serverHeader.style.display = '';
        if (btnAttach) btnAttach.style.display = 'inline-block';
        if (btnDetach) btnDetach.style.display = 'none';
    }
}

export async function initAttachDetachButtons() {
    try {
        const headers = window.authHeaders ? window.authHeaders() : {};
        const resp = await fetch('/api/sessions/active', { headers });
        if (resp.status === 401) {
            console.error('initAttachDetachButtons: authentication required');
            return;
        }
        const data = await resp.json();
        setHeaderMode(data?.mode ?? null);

        // Reconcile the restored URL with the actual session state. This runs
        // before Router.init() dispatches, so adjusting location here with
        // replaceState steers the initial dispatch (no view flash).
        const path = location.pathname || '/';
        // Routes inside the monitor view that require a live session to be useful:
        // the dashboard/logs surface live server telemetry, and chat can't talk to
        // anything without an attached or spawned server. With no session these are
        // dead ends, so they fall back to the welcome screen.
        const needsSessionRoute =
            path === '/server' || path === '/logs' ||
            path === '/chat' || path.startsWith('/chat/');

        if (data?.status === 'Running') {
            // A session is live, so restore the monitor view instead of leaving
            // the user stranded on the welcome screen after a hard refresh. Only
            // auto-restore when the user landed on '/' (the welcome route) — an
            // explicit deep link like /chat or /logs must be respected.
            if (setupViewState.view === 'setup' &&
                (location.pathname === '/' || location.pathname === '')) {
                switchView('monitor');
                try { history.replaceState({ path: '/server' }, '', '/server'); } catch {}
            }
            showTunePanel();
        } else if (needsSessionRoute) {
            // No active session (e.g. a hard refresh that restored a dashboard
            // URL like /server with nothing attached or spawned). The monitor
            // view has nothing to show, so fall back to the welcome screen the
            // same way a fresh session always did before SPA routing existed.
            if (setupViewState.view === 'monitor') switchView('setup');
            try { history.replaceState({ path: '/' }, '', '/'); } catch {}
        }
    } catch (err) {
        console.error('Failed to initialize attach/detach buttons:', err);
    }
}

// ── Init ───────────────────────────────────────────────────────────────────────

export function initAttachDetach() {
    // Bind top detach button
    const detachTop = document.getElementById('btn-detach-top');
    if (detachTop) detachTop.addEventListener('click', doDetach);

    // Bind setup page buttons
    const setupAttach = document.getElementById('setup-attach-btn');
    if (setupAttach) setupAttach.addEventListener('click', doAttachFromSetup);
    const setupBackend = document.getElementById('setup-endpoint-backend');
    const setupModel = document.getElementById('setup-endpoint-model');
    const syncAttachBackendFields = () => {
        if (setupModel) setupModel.hidden = setupBackend?.value !== 'rapid_mlx';
    };
    if (setupBackend) setupBackend.addEventListener('change', syncAttachBackendFields);
    syncAttachBackendFields();

    // Setup wizard button — opens the spawn wizard overlay from the welcome screen
    const setupWizardBtn = document.getElementById('setup-spawn-wizard-btn');
    if (setupWizardBtn) setupWizardBtn.addEventListener('click', () => {
        Router.navigate('/spawn');
    });

    const setupStart = document.getElementById('setup-start-btn');
    if (setupStart) setupStart.addEventListener('click', doStartFromSetup);

    // Bind monitor view buttons
    const btnAttach = document.getElementById('btn-attach');
    if (btnAttach) btnAttach.addEventListener('click', doAttach);

    const btnDetach = document.getElementById('btn-detach');
    if (btnDetach) btnDetach.addEventListener('click', doDetach);

    const btnStart = document.getElementById('btn-start');
    if (btnStart) btnStart.addEventListener('click', doStart);

    const btnStop = document.getElementById('btn-stop');
    if (btnStop) btnStop.addEventListener('click', doStop);

    // Bind Switch Model button
    const btnSwitchModel = document.getElementById('btn-switch-model');
    if (btnSwitchModel) btnSwitchModel.addEventListener('click', () => {
        import('./models.js').then(({ openModelsModal }) => openModelsModal());
    });

    // Bind control bar spawn button — opens wizard from monitor view
    const btnControlSpawn = document.getElementById('btn-control-spawn');
    if (btnControlSpawn) btnControlSpawn.addEventListener('click', () => {
        Router.navigate('/spawn');
    });

    // Bind logs empty state button — opens wizard
    const btnSpawnFromLogs = document.getElementById('btn-spawn-server');
    if (btnSpawnFromLogs) btnSpawnFromLogs.addEventListener('click', () => {
        Router.navigate('/spawn');
    });

    // Note: initAttachDetachButtons() is awaited by bootstrap just before
    // Router.init() so the initial view/URL is reconciled with the live session
    // state before the router dispatches the restored path.
}
