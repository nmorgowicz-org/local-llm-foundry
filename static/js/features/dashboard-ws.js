// ── Dashboard WebSocket Transport ──────────────────────────────────────────────
// WebSocket creation, onmessage dispatch, and dashboard update logic.
// Imports state from app-state.js and render functions from dashboard-render.js.
//
// POWER OPTIMIZATION: Page Visibility API throttling.
// When the tab is hidden, we skip the dashboard update entirely — no need to
// parse JSON and write DOM if the user isn't looking at it. The WebSocket
// still stays connected and receives data, but we discard the messages until
// the tab becomes visible again. This saves ~100+ DOM writes per tick.

import { formatMetricAge, formatMetricNumber, escapeHtml } from '../core/format.js';
import { setHtml } from '../core/set-html.js';
import { deriveTelemetryGrade, gradeLabel, gradeStatusClass, gradeActionCopy } from '../features/telemetry-grade.js';
import {
  sessionState,
  prevValues,
  metricSeries,
  setWsData,
  setLastServerState,
  setLastLlamaMetrics,
  setLastRapidMlxMetrics,
  getLastRapidMlxMetrics,
  setContextCapacityTokens,
  setLastSystemMetrics,
  setLastGpuMetrics,
  setLastCapabilities,
  lastLlamaMetrics,
  lastSystemMetrics,
  wsData,
  setupViewState,
} from '../core/app-state.js';
import {
  updateLiveOutputEstimate,
  updateRequestActivity,
  renderRecentTask,
  renderActivityRail,
  renderSlotGrid,
  getPrimarySlot,
  renderSlotUtilization,
  renderBatchEfficiency,
  renderRequestStats,
  renderDecodingConfig,
  renderCapabilityPopover,
  setMetricSectionVisibility,
} from './dashboard-render.js';
import { refreshChatTelemetry } from './chat-params.js';
import { updateContextCard, updateContextCardFromChatTabs } from './context-card.js';
import { refreshTopCockpit } from './nav.js';
import './chat-state.js';
import { setRemoteAgentStatus } from './remote-agent.js';
import { hideConnectingState, switchView } from './setup-view.js';
import Router from './router.js';
import { showToast, showToastWithActions } from './toast.js';
import { loadPresets, syncSelectedPresetSelection } from './presets.js';
import { updateStateCard, updateMetricCards } from './metric-cards.js';
import { renderLlamaCppDetails, isLlamaTelemetryForTarget } from './llama-cpp-details.js';

// ── Cached DOM elements (populated at init time to avoid repeated queries) ──
let cachedElements = null;
let dashboardSocket = null;
let _lastPresetRefreshId = null;
let overlayStateObserver = null;
const LOG_FONT_SIZE_KEY = 'llama-monitor-log-font-size';
const LOG_FONT_SIZE_DEFAULT = 13;
const LOG_FONT_SIZE_MIN = 8;
const LOG_FONT_SIZE_MAX = 18;
const LOG_TAIL_ENABLED_KEY = 'llama-monitor-log-tail-enabled';
const LOG_TAIL_LINES_KEY = 'llama-monitor-log-tail-lines';
const LOG_TAIL_LINES_DEFAULT = 2;
const LOG_TAIL_LINES_MIN = 1;
const LOG_TAIL_LINES_MAX = 6;

// ── Badge change detection — skip DOM writes when badge content is unchanged ──
let prevBadgeState = { server: null, chat: null, logs: null };

// ── Power optimization: Page Visibility API throttling ─────────────────────────
// When the tab is hidden, skip all dashboard updates (no DOM writes needed) and
// pause all infinite CSS animations (70+) via the power-saver body class.
// The GPU poller interval (in main.rs) automatically tracks the configured
// ws_push_interval_ms setting, so when the user selects "Battery Saver" the
// GPU hardware reads slow down proportionally.
// When the tab becomes visible, do one full update to refresh stale state.
let isTabVisible = true;

function isElementActuallyVisible(el) {
    if (!el) return false;
    const style = getComputedStyle(el);
    return style.display !== 'none' && style.visibility !== 'hidden';
}

function hasBlockingOverlayOpen() {
    const candidates = [
        ...document.querySelectorAll(
            '.modal-overlay.open, .modal-overlay.active, .modal-overlay[style*="display: block"], .modal-overlay[style*="display:block"], .keyboard-shortcut-overlay.open, .keyboard-shortcut-overlay.active, #release-notes-panel.open'
        ),
        document.getElementById('command-palette-overlay'),
    ].filter(isElementActuallyVisible);

    return candidates.length > 0;
}

function syncBackgroundPowerState() {
    document.body.classList.toggle('power-saver', !isTabVisible);
    document.body.classList.toggle('background-paused', hasBlockingOverlayOpen());
}

function ensureOverlayStateObserver() {
    if (overlayStateObserver || !document.body) return;
    let scheduled = false;
    const scheduleSync = () => {
        if (scheduled) return;
        scheduled = true;
        requestAnimationFrame(() => {
            scheduled = false;
            syncBackgroundPowerState();
        });
    };

    overlayStateObserver = new MutationObserver(scheduleSync);
    overlayStateObserver.observe(document.body, {
        subtree: true,
        attributes: true,
        attributeFilter: ['class', 'style', 'aria-hidden', 'inert'],
    });
    scheduleSync();
}

function computeClientMode() {
    // T-053: active / idle / sleep modes for backend sleep orchestration
    if (!isTabVisible) return 'sleep';
    if (hasBlockingOverlayOpen()) return 'idle';
    const tab = currentMonitorTab();
    if (tab === 'chat') return 'active';
    if (isMonitorViewActive()) return 'active';
    return 'idle';
}

function sendWsClientState() {
    if (!dashboardSocket || dashboardSocket.readyState !== WebSocket.OPEN) return;
    try {
        const mode = computeClientMode();
        dashboardSocket.send(JSON.stringify({
            type: 'client-visibility',
            visible: isTabVisible,
            mode: mode,
        }));
    } catch {
        // Ignore transient send failures; the next state change will retry.
    }
}

function currentMonitorTab() {
    const activePage = document.querySelector('.page.active');
    if (!activePage || !activePage.id) return 'server';
    return activePage.id.replace(/^page-/, '');
}

function isMonitorViewActive() {
    return setupViewState.view === 'monitor';
}

document.addEventListener('visibilitychange', () => {
    const wasVisible = isTabVisible;
    isTabVisible = !document.hidden;

    // Toggle background power classes to pause/resume animation work
    syncBackgroundPowerState();
    sendWsClientState();

    if (!wasVisible && isTabVisible) {
        // Tab just became visible — schedule a refresh so stale data gets updated
        requestAnimationFrame(() => {
            if (wsData) {
                updateDashboard(wsData);
            }
        });
    }
});

function ensureCachedElements() {
    if (cachedElements) return;
    cachedElements = {
        // Inference metrics
        mPrompt: document.getElementById('m-prompt'),
        mGen: document.getElementById('m-gen'),
        mPromptDelta: document.getElementById('m-prompt-delta'),
        mGenDelta: document.getElementById('m-gen-delta'),
        mPromptMax: document.getElementById('m-prompt-max'),
        mGenMax: document.getElementById('m-gen-max'),
        mPromptBar: document.getElementById('m-prompt-bar'),
        mGenBar: document.getElementById('m-gen-bar'),
        mThroughputState: document.getElementById('m-throughput-state'),
        mThroughputAge: document.getElementById('m-throughput-age'),
        mThroughputEmpty: document.getElementById('m-throughput-empty'),
        mGenState: document.getElementById('m-generation-state'),
        mGenMain: document.getElementById('m-generation-main'),
        mGenSub: document.getElementById('m-generation-sub'),
        mGenDetails: document.getElementById('m-generation-details'),
        mGenRing: document.getElementById('m-generation-ring'),
        mLiveVelocity: document.getElementById('m-live-velocity'),
        mStagePrompt: document.getElementById('m-stage-prompt'),
        mStageOutput: document.getElementById('m-stage-output'),
        mRatioBar: document.getElementById('m-throughput-ratio-bar'),
        mRatioValue: document.getElementById('m-throughput-ratio'),
        // Cards
        throughputCard: document.querySelector('.widget-speed'),
        generationCard: document.querySelector('.widget-generation'),
        contextCard: document.querySelector('.widget-context'),
        mContextState: document.getElementById('m-context-state'),
        mContextEmpty: document.getElementById('m-context-empty'),
        // Badges
        badgeServer: document.getElementById('badge-server'),
        badgeLogs: document.getElementById('badge-logs'),
        // Endpoint
        endpointMode: document.getElementById('endpoint-mode'),
        endpointUrl: document.getElementById('endpoint-url'),
        endpointStatus: document.getElementById('endpoint-status'),
        // Agent
        agentStatus: document.getElementById('agent-status'),
        agentLatency: document.getElementById('agent-latency'),
        // Status
        statusText: document.getElementById('status-text'),
        // Server state
        serverHeader: document.getElementById('server-header'),
        btnAttach: document.getElementById('btn-attach'),
        btnDetach: document.getElementById('btn-detach'),
        btnDetachTop: document.getElementById('btn-detach-top'),
        historicBadge: document.getElementById('inference-historic-badge'),
        statusDot: document.getElementById('status-dot'),
        btnStart: document.getElementById('btn-start'),
        btnStop: document.getElementById('btn-stop'),
        mGenEmpty: document.getElementById('m-generation-empty'),
        mSlotsState: document.getElementById('m-slots-state'),
        mActivityState: document.getElementById('m-activity-state'),
        // Logs
        logPanel: document.getElementById('log-panel'),
        // Live log tail (Inference Metrics)
        logTailGroup: document.getElementById('inference-log-tail-group'),
        logTailBadge: document.getElementById('inference-log-tail-badge'),
        logTailEl: document.getElementById('inference-log-tail'),
        logTailMinus: document.getElementById('inference-log-tail-minus'),
        logTailPlus: document.getElementById('inference-log-tail-plus'),
    };
}

/** Live log tail feature state */
let logTailActive = false;
let logTailLines = LOG_TAIL_LINES_DEFAULT;
let logTailAllowed = false;
let logTailLastUpdateMs = 0;
const LOG_TAIL_UPDATE_INTERVAL_MS = 600;

function _readLogTailConfig() {
    const cfg = { enabled: false, lines: LOG_TAIL_LINES_DEFAULT };
    try {
        const en = localStorage.getItem(LOG_TAIL_ENABLED_KEY);
        if (en === '1' || en === 'true') cfg.enabled = true;
        const n = Number.parseInt(localStorage.getItem(LOG_TAIL_LINES_KEY), 10);
        if (Number.isFinite(n)) {
            cfg.lines = Math.min(LOG_TAIL_LINES_MAX, Math.max(LOG_TAIL_LINES_MIN, n));
        }
    } catch {
        // ignore
    }
    return cfg;
}

function _saveLogTailConfig(enabled, lines) {
    try {
        localStorage.setItem(LOG_TAIL_ENABLED_KEY, enabled ? '1' : '0');
        localStorage.setItem(LOG_TAIL_LINES_KEY, String(lines));
    } catch {
        // ignore
    }
}

function _applyLogTailLines(lines) {
    logTailLines = Math.min(LOG_TAIL_LINES_MAX, Math.max(LOG_TAIL_LINES_MIN, lines));
    ensureCachedElements();
    const tail = cachedElements.logTailEl;
    if (tail) {
        tail.style.setProperty('--log-tail-lines', logTailLines);
    }
    _updateLogTail(wsData || null);
}

// ── WebSocket setup ───────────────────────────────────────────────────────────

export function initWebSocket() {
    ensureOverlayStateObserver();
    _initLogFontControls();
    _initLogTailFeature();
    const ws = new WebSocket(
        (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws'
    );
    dashboardSocket = ws;

ws.onmessage = e => {
    let d;
    try {
        d = JSON.parse(e.data);
    } catch {
        return; // malformed frame — ignore rather than kill the socket loop
    }
    // Keep wsData current even when tab is hidden (needed for refresh on show)
    setWsData(d);
    if (!isTabVisible) return; // skip DOM writes while tab is hidden
    updateDashboard(d);
};

    // One-time restore-hint on first connect (T-061)
    let _restoreHintDone = false;
    ws.onopen = () => {
        sendWsClientState();

        if (_restoreHintDone) return;
        _restoreHintDone = true;

        fetch('/api/sessions/restore-hint')
            .then(r => r.json())
            .then(data => {
                try {
                    if (data.suggested_action === 'resume_active' && typeof switchView === 'function') {
                        switchView('monitor');
                    } else if (data.suggested_action === 'suggest_recent_attach') {
                        // T-061: show a non-intrusive banner prompting reconnect to the running server
                        showToastWithActions(
                            'Running server detected',
                            'info',
                            'A server is running from your last session. Reconnect from "Recent servers".',
                            [{
                                id: 'reconnect',
                                label: 'Reconnect',
                                primary: true,
                                handler: () => {
                                    const recent = document.querySelector('.recent-endpoint-item');
                                    if (recent) recent.click();
                                },
                            }]
                        );
                    }
                } catch (_) {
                    // best-effort; never block startup
                }
            })
            .catch(() => {
                // ignore — no restore hint available
            });
    };

// Poll context card after chat tabs load
window.onChatTabsLoaded = () => {
    setTimeout(() => {
        updateContextCardFromChatTabs();
    }, 500);
};

    ws.onerror = e => console.error('WebSocket error:', e);

    ws.onclose = () => {
        if (dashboardSocket === ws) dashboardSocket = null;
        ensureCachedElements();
        if (cachedElements.statusText) cachedElements.statusText.textContent = 'Disconnected';
        sessionState.prevLogLen = 0;
        sessionState.prevLogs = [];
    };

    return ws;
}

// ── Main dashboard update (replaces ws.onmessage in app.js) ──────────────────

function updateDashboard(d) {
    // Ensure DOM elements are cached (avoids repeated queries on every WS message)
    ensureCachedElements();

    // Store for use by status alert and other components
    setWsData(d);

    sessionState.activeSessionPresetId =
        d.session_mode === 'spawn' && d.active_session_preset_id ? d.active_session_preset_id : '';

    // Sync preset selector to match the running session's preset (spawn mode only).
    // Don't override when the user has explicitly changed it in the dropdown.
    if (d.session_mode === 'spawn' && d.active_session_preset_id && !window.__presetUserSelected) {
        const sel = document.getElementById('preset-select');
        if (sel && sel.value !== d.active_session_preset_id) {
            const opt = sel.querySelector(`option[value="${CSS.escape(d.active_session_preset_id)}"]`);
            if (opt) {
                syncSelectedPresetSelection(d.active_session_preset_id);
            } else if (_lastPresetRefreshId !== d.active_session_preset_id) {
                _lastPresetRefreshId = d.active_session_preset_id;
                loadPresets(d.active_session_preset_id);
            }
        }
    }
    // Clear user-selection flag once the backend and dropdown agree again.
    if (d.session_mode !== 'spawn' || !d.active_session_preset_id) {
        window.__presetUserSelected = false;
    }

    // Derive and store telemetry grade for consumption by all dashboard components
    const grade = deriveTelemetryGrade(d);
    window.__telemetryGrade = grade;
    const inMonitorView = isMonitorViewActive();
    const activeTab = currentMonitorTab();

    // Attach/Detach buttons and server header
    updateAttachDetach(d);

    // Keep session/runtime state current even when the setup flow is on-screen.
    // The setup path and blocking modal flows still need state sync, but they
    // do not need the hidden background surfaces to keep re-rendering 2x/sec.
    updateServerState(d);

    if (!inMonitorView || hasBlockingOverlayOpen()) {
        return;
    }

    // Endpoint health strip
    updateEndpointStrip(d);

    // Agent status
    updateAgentStatus(d);

    // T-055: 3-way sleep_mode handling
    // mode: "off" | "logs-only" | "sleep"
    const mode = d.mode ?? (d.sleep_mode ? 'sleep' : 'off');
    const isSleeping = mode === 'sleep';

    // Inference metrics (lightweight; always update for basic status)
    updateInferenceMetrics(d);

    // Log tail: enabled in off and logs-only (KEY feature of logs-only mode)
    if (!isSleeping) _updateLogTail(d);
    if (activeTab === 'chat') {
        refreshChatTelemetry();
    }
    refreshTopCockpit();

    // Logs tab: enabled in off and logs-only
    if (activeTab === 'logs' && !isSleeping) updateLogs(d);

    // Badges
    updateBadges(d);
}

// ── Endpoint health strip ────────────────────────────────────────────────────

function updateEndpointStrip(d) {
    const ce = cachedElements;
    const endpointModeEl = ce.endpointMode;
    const endpointUrlEl = ce.endpointUrl;
    const endpointStatusEl = ce.endpointStatus;

    if (d.capabilities && d.endpoint_kind) {
        let modeClass = 'unknown';
        let modeText = 'Unknown';
        let statusClass = 'ok';
        let statusText = 'Full telemetry';

        if (d.endpoint_kind === 'Local') {
            modeClass = 'local';
            modeText = 'Local';
        } else if (d.endpoint_kind === 'Remote') {
            modeClass = 'remote';
            modeText = 'Remote';
        }

        const grade = window.__telemetryGrade || 'local_full';
        statusClass = gradeStatusClass(grade);
        statusText = gradeLabel(grade);

        if (endpointModeEl) {
            endpointModeEl.textContent = modeText;
            endpointModeEl.className = 'endpoint-mode ' + modeClass;
        }
        if (endpointUrlEl) {
            endpointUrlEl.textContent = d.active_session_endpoint || d.active_session_id || 'No session';
        }
        if (endpointStatusEl) {

            setHtml(endpointStatusEl, '<span class="status-dot ' + statusClass + '"></span>' + statusText);
        }
    }
}

// ── Agent status ─────────────────────────────────────────────────────────────

function updateAgentStatus(d) {
    const ce = cachedElements;
    const agentStatusEl = ce.agentStatus;
    const agentLatencyEl = ce.agentLatency;

    if (!agentStatusEl) return;

    const showAgent = d.session_mode === 'attach' && d.endpoint_kind === 'Remote';
    agentStatusEl.style.display = showAgent ? '' : 'none';

    const grade = window.__telemetryGrade || 'local_full';

    // Badge class
    let badgeClass = 'agent-status';
    if (grade === 'remote_agent_connected') badgeClass += ' connected';
    else if (grade === 'remote_agent_connecting') badgeClass += ' update-available';
    else if (grade === 'remote_agent_update_available') badgeClass += ' update-available';
    else if (grade === 'remote_agent_firewall_blocked') badgeClass += ' firewall-blocked';
    else if (grade === 'remote_error') badgeClass += ' disconnected';
    else badgeClass += ' disconnected';

    // Text
    let agentText = 'Remote Agent';
    if (grade === 'remote_agent_connecting') agentText = 'Connecting...';
    else if (grade === 'remote_agent_update_available') agentText = 'Update Available';
    else if (grade === 'remote_agent_firewall_blocked') agentText = 'Firewall blocked';
    else if (grade === 'remote_agent_degraded') agentText = 'Degraded';
    else if (grade === 'remote_partial_sensors') agentText = 'Partial sensors';
    else if (grade === 'remote_error') agentText = 'Agent error';
    else if (grade === 'remote_inference_only') agentText = 'No Remote Agent';

    // Tooltip
    let tooltipText = 'Connected';
    let tooltipClass = 'connected';
    if (grade === 'remote_agent_connecting') { tooltipText = 'Connecting'; tooltipClass = 'warning'; }
    else if (grade === 'remote_agent_update_available') { tooltipText = 'Update available'; tooltipClass = 'warning'; }
    else if (grade === 'remote_agent_firewall_blocked') { tooltipText = 'Firewall blocked'; tooltipClass = 'warning'; }
    else if (grade === 'remote_agent_degraded') { tooltipText = 'Degraded compatibility'; tooltipClass = 'warning'; }
    else if (grade === 'remote_partial_sensors') { tooltipText = 'Partial sensor coverage'; tooltipClass = 'warning'; }
    else if (grade === 'remote_inference_only') { tooltipText = 'Inference only'; tooltipClass = 'disconnected'; }
    else if (grade === 'remote_error') { tooltipText = 'Connection failed'; tooltipClass = 'disconnected'; }

    // Fix button — show for any non-fully-connected state
    const needsFix = !['remote_agent_connected', 'local_full'].includes(grade);
    // Fix button copy
    let fixCopy = '\u26a1 Fix';
    let fixTitle = 'Set up remote agent';
    if (grade === 'remote_agent_connecting') { fixCopy = '\u23F3 Wait'; fixTitle = 'Remote agent activity is already in progress'; }
    else if (grade === 'remote_agent_update_available') { fixCopy = '\u26a1 Upgrade'; fixTitle = 'Upgrade remote agent to latest version'; }
    else if (grade === 'remote_agent_firewall_blocked') { fixTitle = 'Repair remote agent connectivity'; }
    else if (grade === 'remote_agent_degraded') { fixCopy = '\u26a1 Upgrade'; fixTitle = 'Upgrade agent for full compatibility'; }

    // Tooltip details — include action copy
    let details = '';
    if (d.remote_agent_version) details = 'Running v' + d.remote_agent_version;
    if (d.remote_agent_url) details += (details ? ' | ' : '') + d.remote_agent_url;
    const actionCopy = gradeActionCopy(grade);
    if (actionCopy) details += (details ? '\n' : '') + actionCopy;

    agentStatusEl.className = badgeClass;

    const textEl = agentStatusEl.querySelector('.agent-text');
    const fixBtn = agentStatusEl.querySelector('.btn-agent-fix');
    if (textEl) {
        textEl.textContent = agentText;
    }
    if (fixBtn) {
        fixBtn.style.display = needsFix ? '' : 'none';
        fixBtn.textContent = fixCopy;
        fixBtn.title = fixTitle;
    }

    // Update tooltip
    const tooltipStatus = document.getElementById('agent-tooltip-status');
    const tooltipDetails = document.getElementById('agent-tooltip-details');
    if (tooltipStatus) {
        tooltipStatus.textContent = tooltipText;
        tooltipStatus.className = 'agent-tooltip-status ' + tooltipClass;
    }
    if (tooltipDetails) {
        tooltipDetails.textContent = details;
    }

    // Telemetry grade chip
    const gradeChip = document.getElementById('telemetry-grade-chip');
    if (gradeChip) {
        const chipGrade = window.__telemetryGrade || 'local_full';
        gradeChip.textContent = gradeLabel(chipGrade);
        gradeChip.className = 'telemetry-grade-chip grade-' + gradeStatusClass(chipGrade);
        // Hide chip when agent is fully healthy — the endpoint-status pill already
        // shows "Full telemetry" and the badge label alone is sufficient.
        gradeChip.style.display = (d.endpoint_kind === 'Remote' && chipGrade !== 'remote_agent_connected') ? '' : 'none';
    }

    if (grade === 'remote_agent_firewall_blocked') {
        setRemoteAgentStatus('Agent connected but HTTP is not reachable (firewall blocked)', 'warning');
    }

    if (agentLatencyEl) {
        agentLatencyEl.textContent = '';
    }
}

// ── Attach/Detach buttons ────────────────────────────────────────────────────

function _updateSetupRunningStrip(isRunning, endpoint) {
    const strip = document.getElementById('setup-spawn-running-strip');
    if (!strip) return;
    if (!isRunning) {
        strip.style.display = 'none';
        return;
    }
    const nameEl = document.getElementById('setup-spawn-running-name');
    const endpointEl = document.getElementById('setup-spawn-running-endpoint');
    const btn = document.getElementById('setup-spawn-running-btn');

    if (nameEl) {
        const presetName = document.getElementById('preset-select')?.selectedOptions[0]?.text || 'Server';
        nameEl.textContent = presetName;
    }
    if (endpointEl && endpoint) {
        const port = endpoint.split(':').pop();
        endpointEl.textContent = ':' + port;
    }
    if (btn && !btn._bound) {
        btn._bound = true;
        btn.addEventListener('click', () => Router.navigate('/server'));
    }
    strip.style.display = '';
}

function updateAttachDetach(d) {
    const ce = cachedElements;
    const serverHeader = ce.serverHeader;
    const btnAttach = ce.btnAttach;
    const btnDetach = ce.btnDetach;
    const btnDetachTop = ce.btnDetachTop;
    const historicBadge = ce.historicBadge;

    const isSpawn = d.session_mode === 'spawn';
    const isAttach = d.session_mode === 'attach' && d.active_session_endpoint;
    const endpointLabel = document.getElementById('server-endpoint-label');
    if (endpointLabel) endpointLabel.style.display = isSpawn ? 'none' : '';

    // Welcome screen running-server banner
    _updateSetupRunningStrip(isSpawn && d.server_running, d.active_session_endpoint);

    if (isSpawn) {
        // Spawn endpoints have no attach controls, but llama.cpp identity/facts remain visible.
        const hasLlamaIdentity = d.backend === 'llama_cpp' && d.active_session_id
            && (d.active_session_model_identity
                || (isLlamaTelemetryForTarget(d.llama, d.active_session_id,
                    d.active_session_endpoint_tag ?? d.active_session_endpoint) && d.llama?.runtime_facts));
        if (serverHeader) serverHeader.style.display = hasLlamaIdentity ? '' : 'none';
        btnAttach.style.display = 'none';
        btnDetach.style.display = 'none';
        if (btnDetachTop) btnDetachTop.style.display = 'none';
    } else if (isAttach) {
        // Attach mode: show header so user can see the connected endpoint
        if (serverHeader) serverHeader.style.display = '';
        btnAttach.style.display = 'none';
        btnDetach.style.display = 'inline-block';
        if (btnDetachTop) btnDetachTop.style.display = 'inline-block';
        if (typeof setupViewState !== 'undefined' && setupViewState.view === 'setup') {
            hideConnectingState();
            switchView('monitor');
        }
    } else {
        // No active session: show header with Attach button
        if (serverHeader) serverHeader.style.display = '';
        btnAttach.style.display = 'inline-block';
        btnDetach.style.display = 'none';
        if (btnDetachTop) btnDetachTop.style.display = 'none';
    }

   if (historicBadge) {
        historicBadge.style.display = isAttach ? 'none' : 'inline-block';
    }

  // Live log tail pill group: show benchmark pill when server connected.
    // Log tail badge only allowed when spawn mode and logs exist.
    const serverRunning = d.server_running ?? false;
    const hasLogs = isSpawn && d.logs && d.logs.length > 0;
    logTailAllowed = hasLogs;

    if (!serverRunning) {
        // Server not running: hide UI, but preserve user's open/closed preference
        // so that after restart (e.g., during updates) the tail doesn't silently close.
        if (ce.logTailGroup) ce.logTailGroup.style.display = 'none';
        if (ce.logTailBadge) ce.logTailBadge.classList.remove('is-active');
        if (ce.logTailEl) {
            ce.logTailEl.style.display = 'none';
            setHtml(ce.logTailEl, '');
        }
    } else if (ce.logTailGroup) {
        // Server running: show group (benchmark pill visible in all modes)
        _syncLogTailVisibility();
    }
}

// Last session error tracking: used to show a prominent warning when the
// server crashes (e.g., OOM) so the user can see/understand/take action.
let lastSessionErrorShown = '';

// ── Server state ─────────────────────────────────────────────────────────────

function updateServerState(d) {
    sessionState.serverRunning = d.server_running;
    const ce = cachedElements;

    const dot = ce.statusDot;
    const txt = ce.statusText;
    const btnStart = ce.btnStart;
    const btnStop = ce.btnStop;

    // Handle active session error status (OOM, crash, etc.) prominently.
    const sessionStatus = d.active_session_status || '';
    const sessionError = d.active_session_error || '';
    const isError = sessionStatus === 'error' && sessionError;

    if (isError && sessionError !== lastSessionErrorShown) {
        lastSessionErrorShown = sessionError;
        showToast(
            'Server error',
            'error',
            sessionError,
            { duration: 12000 }
        );

        // Show diagnostics in the main control bar if present
        showServerErrorDetails(d);

        // Show diagnostics on the welcome / Local Server panel if present
        showLocalServerErrorBar(sessionError, d);
    }

    if (isError) {
        dot.className = 'status-dot stopped';
        txt.textContent = 'Error';
        // Make sure Start button is enabled so user can try again immediately
        if (btnStart) btnStart.disabled = false;
        if (btnStop) btnStop.disabled = true;
    } else {
        dot.className = 'status-dot ' + (sessionState.serverRunning ? 'running' : 'stopped');
        txt.textContent = sessionState.serverRunning ? 'Running' : 'Stopped';
        hideServerErrorDetails();
        hideLocalServerErrorBar();
    }

    if (!isError) {
        const localRunning = d.local_server_running || false;
        if (btnStart) btnStart.disabled = localRunning;
        if (btnStop) btnStop.disabled = !localRunning;
    }

    const btnSwitchModel = document.getElementById('btn-switch-model');
    if (btnSwitchModel) btnSwitchModel.style.display = (d.local_server_running || false) ? '' : 'none';

    setLastServerState(d.server_running);
    const llamaMetrics = d.backend === 'llama_cpp' && d.active_session_id
        && isLlamaTelemetryForTarget(d.llama, d.active_session_id,
            d.active_session_endpoint_tag ?? d.active_session_endpoint)
        ? (d.llama ?? null) : null;
    setLastLlamaMetrics(llamaMetrics);
    // Run before monitor/overlay early returns so switching clears hidden stale facts too.
    renderLlamaCppDetails({
        backend: d.backend,
        attached: !!d.active_session_id,
        sessionId: d.active_session_id,
        endpoint: d.active_session_endpoint,
        endpointTag: d.active_session_endpoint_tag,
        attachedModel: d.active_session_model_identity,
        metrics: llamaMetrics,
    });
    setLastRapidMlxMetrics(d.backend === 'rapid_mlx' ? (d.inference ?? null) : null);
    // Normalize context capacity to the actual loaded limit.
    // KV-only reports can be stale; prefer reported capacity, then KV max, then a
    // safe default so context-pressure math is consistent across telemetry and chat.
    const l = llamaMetrics;
    let capacity = l?.context_capacity_tokens || l?.kv_cache_max || 0;
    if (capacity <= 0) capacity = l?.context_size || 0;
    capacity = Math.max(0, Math.min(capacity, 2_097_152));
    if (capacity > 0) {
        setContextCapacityTokens(capacity);
    }
    const prevSystemMetrics = lastSystemMetrics;
    setLastSystemMetrics(d.system || null);
    setLastCapabilities(d.capabilities || null);
    setLastGpuMetrics(d.gpu || {});

    // If system metrics just became available or changed (e.g., pCores known),
    // notify consumers that need to update hints (spawn wizard, preset editor, etc.)
    if (d.system && (!prevSystemMetrics || prevSystemMetrics.p_cores == null)) {
      window.__refreshSpawnWizardHints?.();
      window.__refreshPresetEditorHints?.();
    }

    // Sync session port/endpoint from WebSocket data (replaces removed HTTP poll)
    const ep = d.active_session_endpoint;
    if (ep) {
        try {
            const url = new URL(ep);
            const port = parseInt(url.port) || 8080;
            if (sessionState.activeSessionPort !== port) {
                sessionState.activeSessionPort = port;
            }
            // Keep the endpoint input in sync
            const endpointInput = document.getElementById('server-endpoint');
            if (endpointInput && endpointInput.value !== ep) {
                endpointInput.value = ep;
            }
        } catch {
            // ignore parse errors
        }
    }
}

// ── Inference metrics ────────────────────────────────────────────────────────

function updateInferenceMetrics(d) {
    const ce = cachedElements;
    const l = lastLlamaMetrics;
    const rm = getLastRapidMlxMetrics();
    const hasActiveEndpoint = !!d.active_session_id;
    const backend = d.backend || (rm ? 'rapid_mlx' : (l ? 'llama_cpp' : 'unknown'));

    // Normalized snapshot: llama.cpp and Rapid-MLX (and remote agents) feed
    // the same card registry, so the dashboard no longer swaps card sets.
    const promptRate = l?.prompt_tokens_per_sec || 0;
    const genRate = l?.generation_tokens_per_sec || 0;
    const promptDisplay = promptRate > 0 ? promptRate : (l?.last_prompt_tokens_per_sec || 0)
        || (Number.isFinite(rm?.prompt_tokens_per_second) ? rm.prompt_tokens_per_second : null);
    const genDisplay = genRate > 0 ? genRate : (l?.last_generation_tokens_per_sec || 0)
        || (Number.isFinite(rm?.generation_tokens_per_second) ? rm.generation_tokens_per_second : null);

    const slotsProcessing = (l?.slots_processing || 0);
    const rapidRunning = Number(rm?.running_requests) || 0;
    // Classify by whether any token has been decoded this request — rate
    // fields lag one poll and can hold stale values while a new prompt is
    // still prefilling, which made the state flip Reading ↔ Generating.
    const slotGenTokens = l?.slot_generation_tokens || 0;
    const promptOnly = slotsProcessing > 0 && slotGenTokens === 0;
    const generationActive = (genRate > 0 || slotsProcessing > 0 || rapidRunning > 0) && !promptOnly;
    const reading = !generationActive && (promptRate > 0 || promptOnly);
    const queued = Number(l?.waiting_requests) || Number(rm?.waiting_requests) || 0;
    const sessionError = d.active_session_status === 'error';

    // Progress: generating → tokens vs budget (llama slots) or backend
    // progress (Rapid-MLX); reading has no observable fraction → hidden bar.
    let stateProgress = null;
    let stateTone = null;
    let stateLabel = 'Idle';
    let stateDetail = '';
    if (sessionError) {
        stateLabel = 'Error';
        stateTone = 'danger';
    } else if (generationActive) {
        stateLabel = 'Generating';
        const generated = l?.slot_generation_tokens || 0;
        const remaining = l?.slot_generation_remaining || 0;
        const slotLimit = getPrimarySlot(l)?.output_limit || 0;
        const total = l?.slot_generation_limit || slotLimit || (generated + remaining);
        if (total > 0) {
            stateProgress = generated / total;
            stateDetail = `${formatMetricNumber(generated)} / ${formatMetricNumber(total)} tokens`;
        } else {
            const ratio = normalizedProgressRatio(rm?.backend_details?.progress);
            stateProgress = ratio;
            stateDetail = ratio != null ? `${Math.round(ratio * 100)}%` : '';
        }
        if (genDisplay) stateDetail += `${stateDetail ? ' · ' : ''}${fmtTps(genDisplay)} t/s`;
        stateTone = 'generating';
    } else if (reading) {
        stateLabel = 'Reading';
        stateTone = 'reading';
        const parts = [];
        if (l?.slot_prompt_total > 0 && l?.slot_prompt_processed > 0) {
            parts.push(`${formatMetricNumber(l.slot_prompt_processed)} / ${formatMetricNumber(l.slot_prompt_total)} prompt tokens`);
            stateProgress = Math.min(1, l.slot_prompt_processed / l.slot_prompt_total);
        }
        if (promptDisplay) parts.push(`${fmtTps(promptDisplay)} t/s prefill`);
        stateDetail = parts.join(' · ');
    } else if (!hasActiveEndpoint) {
        stateLabel = 'Waiting for a request';
    } else if (queued > 0) {
        stateLabel = 'Queued';
        stateDetail = `${queued} waiting`;
    }

    const view = {
        backend,
        attached: hasActiveEndpoint,
        sessionId: d.active_session_id ?? null,
        endpointTag: d.active_session_endpoint_tag ?? d.active_session_endpoint ?? null,
        state: sessionError ? 'error' : generationActive ? 'generating' : reading ? 'reading' : 'idle',
        queued,
        running: slotsProcessing || rapidRunning,
        waiting: queued,
        decodeTps: genDisplay || null,
        prefillTps: promptDisplay || null,
        stateLabel,
        stateDetail,
        stateProgress,
        stateTone,
        gpu: buildGpuView(d),
        sys: buildSysView(d),
    };

    updateStateCard(view);
    updateMetricCards(view);

    // Loader-specific detail panels below the strip (unchanged behavior).
    if (backend !== 'rapid_mlx') {
        renderSlotGrid(l, hasActiveEndpoint);
        renderSlotUtilization(l);
        renderBatchEfficiency(l);
    } else {
        if (ce.mSlotsState) ce.mSlotsState.textContent = '';
        if (ce.mActivityState) ce.mActivityState.textContent = '';
        renderSlotGrid(null, false);
        renderSlotUtilization(null);
        renderBatchEfficiency(null);
    }
    updateRequestActivity(generationActive ? l?.active_task_id : l?.last_task_id, generationActive, l?.slot_generation_tokens || 0, Date.now());
    renderActivityRail(generationActive);
    renderRecentTask();
    renderRequestStats();
    renderDecodingConfig(l, hasActiveEndpoint, generationActive);

    updateContextMetrics(d, l, hasActiveEndpoint);
    renderCapabilityPopover(d, l, !!l?.slot_generation_available, !!(l?.context_live_tokens_available || l?.kv_cache_tokens_available || (l?.context_capacity_tokens || 0) > 0));

    const hostMetricsVisible = d.host_metrics_available === true;
    setMetricSectionVisibility('gpu-card', hostMetricsVisible && !!d.capabilities?.gpu, 'gpu-section');
    setMetricSectionVisibility('system-card', hostMetricsVisible && !!d.capabilities?.system, 'system-section');
}

function normalizedProgressRatio(value) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0 || value > 1) return null;
    return value;
}

function fmtTps(v) {
    return v >= 100 ? Math.round(v).toLocaleString() : v.toFixed(1);
}

function buildGpuView(d) {
    const entries = Object.entries(d.gpu || {});
    if (!entries.length) return null;
    const [name, m] = entries[0];
    // GpuMetrics reports MiB; the dashboard cards work in bytes.
    const toBytes = (x) => (Number(x) || 0) * 1024 * 1024;
    // On Apple Silicon the meaningful ceiling is the Metal wired-limit cap;
    // vram_used/vram_total describe the unified memory pool.
    const metalCap = toBytes(m.metal_gpu_limit_mb);
    const unifiedTotal = toBytes(m.vram_total);
    return {
        name,
        load: Number(m.load) || 0,
        vramUsed: toBytes(m.vram_used),
        vramTotal: metalCap > 0 ? metalCap : unifiedTotal,
        unifiedTotal,
        metalUnified: metalCap > 0,
        temp: Number(m.temp) || 0,
        power: Number(m.power_consumption) || 0,
        powerLimit: Number(m.power_limit) || 0,
    };
}

function buildSysView(d) {
    const sys = lastSystemMetrics;
    if (!sys || d.host_metrics_available !== true || !d.capabilities?.system) return null;
    return {
        cpu: Number(sys.cpu_load) || 0,
        cpuName: sys.cpu_name || '',
        ramUsed: Number(sys.ram_used) || 0,
        ramTotal: Number(sys.ram_total) || 0,
    };
}

// ── Context metrics ──────────────────────────────────────────────────────────

function updateContextMetrics(d, l, hasActiveEndpoint) {
    updateContextCard(d, l, hasActiveEndpoint);
}

// ── Logs ─────────────────────────────────────────────────────────────────────

let logAutoScroll = true; // true = follow tail; false = user has scrolled up

function _parseLogLevel(line) {
    // llama.cpp v2 format: "INFO  [component] message" or "WARN [...]" etc.
    const levelMatch = line.match(/^(INFO|WARN|ERROR|DEBUG|VERB|FATAL)\b/i);
    if (levelMatch) return levelMatch[1].toUpperCase();
    // Legacy format: starts with timestamp + level, e.g. "2024-01-01T... INFO ..."
    const tsFmt = line.match(/^\d{4}-\d{2}-\d{2}T\S+\s+(INFO|WARN|ERROR|DEBUG)\b/i);
    if (tsFmt) return tsFmt[1].toUpperCase();
    // llama.cpp single-letter level after short timestamp, e.g. "1515.01.190.648 W slot ..."
    const singleLetter = line.match(/^\d[\d\.]+\s+([IiWwEeDd])\b/);
    if (singleLetter) {
        const c = singleLetter[1].toUpperCase();
        if (c === 'W') return 'WARN';
        if (c === 'E') return 'ERROR';
        if (c === 'I') return 'INFO';
        if (c === 'D') return 'DEBUG';
    }
    return 'OTHER';
}

function _levelClass(level) {
    if (level === 'WARN')  return 'log-warn';
    if (level === 'ERROR' || level === 'FATAL') return 'log-error';
    if (level === 'DEBUG' || level === 'VERB')  return 'log-debug';
    return 'log-info';
}

function _colorizeLogLine(line) {
    if (!line) return escapeHtml(line || '');

    let s = escapeHtml(line);

    // 1) Full-level words: INFO/WARN/ERROR/DEBUG/VERB/FATAL
    s = s.replace(
        /\b((?:INFO|WARN(ING)?|ERROR|DEBUG|VERB|FATAL))\b/g,
        '<span class="log-lev">$1</span>'
    );

    // 2) Single-letter level after timestamp/IP-like prefix: e.g. "1392.56.898.909 W slot ..."
    // Match: leading digits/letters/dots, then space, then one of I/W/E/D near a word boundary.
    s = s.replace(
        /^([\w\.]+)\s+([IiWwEeDd])\b/g,
        '$1 <span class="log-lev">$2</span>'
    );

    // 3) Key components / actions (llama.cpp style: "slot", "srv", "llm_load_tensors", etc.)
    s = s.replace(
        /\b((?:slot|srv|begin|create_check|update_slots|release_slot|launch_slot|print_timing|statistics|reasoning-budget|ngram-mod|draft-acceptance|draft-mtp|llm_load_tensors|llama_perf_context|ggml_backend_\w+_log_allocated_size|prompt_cache|prompt_save|prompt_load))\b/g,
        '<span class="log-comp">$1</span>'
    );

    // 4) Metrics: e.g. "335.46 ms", "635 tokens", "124.16 tokens per second"
    // - "X.Y ms" or integer ms
    s = s.replace(
        /(\d+\.?\d*)\s*ms\b/g,
        '<span class="log-metric">$1 ms</span>'
    );
    // - tokens per second
    s = s.replace(
        /(\d+\.?\d*)\s*tokens?\s+per\s+second\b/g,
        '<span class="log-metric">$1 tokens per second</span>'
    );
    // - "N tokens" (only when preceded by space, to avoid hitting random IDs too harshly)
    s = s.replace(
        /(?<=\s)(\d+\.?\d*)\s*tokens?\b/g,
        ' <span class="log-metric">$1 tokens</span>'
    );

    // 5) n_tokens values (prompt processing / final token counts)
    s = s.replace(
        /\bn_tokens\s*=\s*(\d+)/g,
        'n_tokens = <span class="log-important">$1</span>'
    );

    // 6) n_decoded (generation progress)
    s = s.replace(
        /\bn_decoded\s*=\s*(\d+)/g,
        'n_decoded = <span class="log-important">$1</span>'
    );

    // 7) progress (prompt processing progress, e.g. "progress = 0.84")
    s = s.replace(
        /\bprogress\s*=\s*([\d.]+)/g,
        'progress = <span class="log-important">$1</span>'
    );

    // 8) draft acceptance rate and related stats
    s = s.replace(
        /\bdraft acceptance\s*=\s*([\d.]+)/g,
        'draft acceptance = <span class="log-metric">$1</span>'
    );
    // - "N accepted" / "N generated" in speculative lines
    s = s.replace(
        /(\d+)\s+(?:accepted|generated)\b/g,
        '<span class="log-metric">$1</span> $2'
    );

    // 9) Cache and checkpoint metrics
    // - "cache reuse" / "cache state"
    s = s.replace(
        /cache (?:reuse|state|save|rm|load)\b/gi,
        '<span class="log-metric">cache $1</span>'
    );
    // - "checkpoints: N"
    s = s.replace(
        /\bcheckpoints:\s*(\d+)/gi,
        'checkpoints: <span class="log-metric">$1</span>'
    );

    // 10) graphs reused
    s = s.replace(
        /graphs reused = (\d+)/g,
        'graphs reused = <span class="log-metric">$1</span>'
    );

    // 11) prompt eval time / eval time / total time in print_timing output
    s = s.replace(
        /(prompt eval time|eval time|total time)\s*=/g,
        '<span class="log-metric">$1</span> ='
    );

    // 12) Memory / cache sizes (MiB/GiB)
    s = s.replace(
        /(\d+\.?\d*)\s*(MiB|GiB)/gi,
        '<span class="log-size">$1 $2</span>'
    );

    // 13) Similarity scores (f_keep, sim) important for caching
    s = s.replace(
        /\b(f_keep|sim)\s*=\s*([\d.]+)/g,
        '$1 = <span class="log-metric">$2</span>'
    );

    return s;
}

function _renderLogLine(line) {
    const div = document.createElement('div');
    div.className = 'log-line ' + _levelClass(_parseLogLevel(line));

    setHtml(div, _colorizeLogLine(line));
    return div;
}

function _initLogScrollTracking(el) {
    if (el._logScrollBound) return;
    el._logScrollBound = true;
    el.addEventListener('scroll', () => {
        const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
        if (!atBottom && logAutoScroll) {
            logAutoScroll = false;
            _syncLogScrollLockBtn();
        } else if (atBottom && !logAutoScroll) {
            logAutoScroll = true;
            _syncLogScrollLockBtn();
        }
    }, { passive: true });
}

function _syncLogScrollLockBtn() {
    const btn = document.getElementById('log-scroll-lock-btn');
    if (!btn) return;
    btn.setAttribute('aria-pressed', String(logAutoScroll));
    btn.classList.toggle('log-tool-btn--locked', !logAutoScroll);
}

function _readLogFontSize() {
    try {
        const stored = Number.parseInt(localStorage.getItem(LOG_FONT_SIZE_KEY), 10);
        if (Number.isFinite(stored)) {
            return Math.min(LOG_FONT_SIZE_MAX, Math.max(LOG_FONT_SIZE_MIN, stored));
        }
    } catch {
        // Storage can be unavailable in hardened browser contexts.
    }
    return LOG_FONT_SIZE_DEFAULT;
}

function _applyLogFontSize(size) {
    const normalized = Math.min(LOG_FONT_SIZE_MAX, Math.max(LOG_FONT_SIZE_MIN, size));
    const panel = document.getElementById('log-panel');
    const valueBtn = document.getElementById('log-font-size-btn');
    const decreaseBtn = document.getElementById('log-font-decrease-btn');
    const increaseBtn = document.getElementById('log-font-increase-btn');

    if (panel) panel.style.setProperty('--log-font-size', `${normalized}px`);
    if (valueBtn) valueBtn.textContent = `${normalized}px`;
    if (decreaseBtn) decreaseBtn.disabled = normalized <= LOG_FONT_SIZE_MIN;
    if (increaseBtn) increaseBtn.disabled = normalized >= LOG_FONT_SIZE_MAX;

    try {
        localStorage.setItem(LOG_FONT_SIZE_KEY, String(normalized));
    } catch {
        // Keep the current-page setting even when persistence is unavailable.
    }
    return normalized;
}

function _initLogFontControls() {
    const decreaseBtn = document.getElementById('log-font-decrease-btn');
    const increaseBtn = document.getElementById('log-font-increase-btn');
    const valueBtn = document.getElementById('log-font-size-btn');
    if (!decreaseBtn || !increaseBtn || !valueBtn || decreaseBtn._logFontBound) return;

    decreaseBtn._logFontBound = true;
    let size = _applyLogFontSize(_readLogFontSize());
    decreaseBtn.addEventListener('click', () => {
        size = _applyLogFontSize(size - 1);
    });
    increaseBtn.addEventListener('click', () => {
        size = _applyLogFontSize(size + 1);
    });
    valueBtn.addEventListener('click', () => {
        size = _applyLogFontSize(LOG_FONT_SIZE_DEFAULT);
    });
}

function _initLogTailFeature() {
    ensureCachedElements();
    const group = cachedElements.logTailGroup;
    const badge = cachedElements.logTailBadge;
    const tail = cachedElements.logTailEl;
    const minusBtn = cachedElements.logTailMinus;
    const plusBtn = cachedElements.logTailPlus;

    if (!group || !badge || !tail || minusBtn === undefined || plusBtn === undefined || badge._logTailBound) return;
    badge._logTailBound = true;

    const cfg = _readLogTailConfig();
    logTailActive = cfg.enabled;
    _applyLogTailLines(cfg.lines);

    // Pill toggle
    badge.addEventListener('click', () => {
        logTailActive = !logTailActive;
        badge.classList.toggle('is-active', logTailActive);
        _syncLogTailVisibility();
        _saveLogTailConfig(logTailActive, logTailLines);
        if (logTailActive) {
            _updateLogTail(wsData || null);
        } else {
            setHtml(tail, '');
        }
    });

    // + / - buttons
    minusBtn.addEventListener('click', () => {
        _applyLogTailLines(logTailLines - 1);
        _saveLogTailConfig(logTailActive, logTailLines);
    });

    plusBtn.addEventListener('click', () => {
        _applyLogTailLines(logTailLines + 1);
        _saveLogTailConfig(logTailActive, logTailLines);
    });

    // Don't show anything until updateAttachDetach confirms it is allowed.
}

function _syncLogTailVisibility() {
    if (!logTailAllowed) return;

    const group = cachedElements.logTailGroup;
    const tail = cachedElements.logTailEl;
    const minusBtn = cachedElements.logTailMinus;
    const plusBtn = cachedElements.logTailPlus;
    const badge = cachedElements.logTailBadge;

    if (group) group.style.display = 'inline-flex';
    if (badge) badge.classList.toggle('is-active', logTailActive);
    if (tail) tail.style.display = logTailActive ? '' : 'none';
    if (minusBtn) minusBtn.classList.toggle('show', logTailActive);
    if (plusBtn) plusBtn.classList.toggle('show', logTailActive);
}

function _updateLogTail(d) {
    if (!logTailActive || !logTailAllowed) return;
    const now = Date.now();
    if (now - logTailLastUpdateMs < LOG_TAIL_UPDATE_INTERVAL_MS) return;
    logTailLastUpdateMs = now;

    ensureCachedElements();
    const tail = cachedElements.logTailEl;
    if (!tail) return;

    const logs = d.logs;
    if (!Array.isArray(logs) || logs.length === 0) {
        setHtml(tail, '');
        return;
    }

    const lastN = logs.slice(-logTailLines);

    // Ensure tail only contains the lines we want to show
    while (tail.children.length > lastN.length) {
        tail.removeChild(tail.lastChild);
    }

    for (let i = 0; i < lastN.length; i++) {
        const line = lastN[i];
        if (!line) continue;

        let el = tail.children[i];
        if (!el) {
            el = document.createElement('div');
            tail.appendChild(el);
        }

        el.className = 'log-line ' + _levelClass(_parseLogLevel(line));

        setHtml(el, _colorizeLogLine(line));
    }
}

function _initLogToolbar(el) {
    const copyBtn = document.getElementById('log-copy-btn');
    if (copyBtn && !copyBtn._logBound) {
        copyBtn._logBound = true;
        copyBtn.addEventListener('click', () => {
            const text = (copyBtn._currentLogs || []).join('\n');
            navigator.clipboard?.writeText(text).then(() => {
                const span = copyBtn.querySelector('span');
                const orig = span?.textContent;
                if (span) span.textContent = 'Copied!';
                setTimeout(() => { if (span) span.textContent = orig; }, 1500);
            });
        });
    }
    const scrollBtn = document.getElementById('log-scroll-lock-btn');
    if (scrollBtn && !scrollBtn._logBound) {
        scrollBtn._logBound = true;
        scrollBtn.addEventListener('click', () => {
            logAutoScroll = !logAutoScroll;
            _syncLogScrollLockBtn();
            if (logAutoScroll && el) el.scrollTop = el.scrollHeight;
        });
    }
    const cmdBtn = document.getElementById('log-cmd-btn');
    const cmdPanel = document.getElementById('log-cmd-panel');
    if (cmdBtn && cmdPanel && !cmdBtn._logBound) {
        cmdBtn._logBound = true;
        cmdBtn.addEventListener('click', () => {
            const open = cmdPanel.getAttribute('aria-hidden') === 'false';
            cmdPanel.setAttribute('aria-hidden', String(open));
            cmdPanel.classList.toggle('open', !open);
            cmdBtn.classList.toggle('log-tool-btn--active', !open);
        });
    }
    const cmdCopyBtn = document.getElementById('log-cmd-copy-btn');
    if (cmdCopyBtn && !cmdCopyBtn._logBound) {
        cmdCopyBtn._logBound = true;
        cmdCopyBtn.addEventListener('click', () => {
            const pre = document.getElementById('log-cmd-pre');
            const text = pre?.textContent || '';
            if (!text) return;
            navigator.clipboard?.writeText(text).then(() => {
                const orig = cmdCopyBtn.textContent;
                cmdCopyBtn.textContent = 'Copied!';
                setTimeout(() => { cmdCopyBtn.textContent = orig; }, 1500);
            });
        });
    }
}

function _findLogOverlap(previous, current) {
    const maxOverlap = Math.min(previous.length, current.length);
    for (let overlap = maxOverlap; overlap > 0; overlap--) {
        const previousStart = previous.length - overlap;
        let matches = true;
        for (let i = 0; i < overlap; i++) {
            if (previous[previousStart + i] !== current[i]) {
                matches = false;
                break;
            }
        }
        if (matches) return overlap;
    }
    return 0;
}

export function updateLogs(d) {
    ensureCachedElements();
    const logs = d.logs || [];
    const emptyState = document.getElementById('logs-empty-state');
    const logsPage = document.getElementById('page-logs');
    const el = cachedElements.logPanel;

    // Show/hide empty state
    if (emptyState) {
        if (logs.length === 0) {
            emptyState.classList.add('visible');
            logsPage?.classList.add('logs-empty-mode');
        } else {
            emptyState.classList.remove('visible');
            logsPage?.classList.remove('logs-empty-mode');
        }
    }

    if (!el) return;

    _initLogScrollTracking(el);
    _initLogToolbar(el);

    // Update copy button's closure with current logs
    const copyBtn = document.getElementById('log-copy-btn');
    if (copyBtn) copyBtn._currentLogs = logs;

    const previousLogs = sessionState.prevLogs || [];
    const overlap = _findLogOverlap(previousLogs, logs);
    const expiredCount = previousLogs.length - overlap;
    const newLines = logs.slice(overlap);

    if (previousLogs.length > 0 && overlap === 0) {
        // The buffer was cleared or replaced with unrelated output.
        setHtml(el, '');
    } else {
        for (let i = 0; i < expiredCount; i++) {
            el.firstElementChild?.remove();
        }
    }

    if (newLines.length > 0) {
        const frag = document.createDocumentFragment();
        for (const line of newLines) {
            frag.appendChild(_renderLogLine(line));
        }
        el.appendChild(frag);

        if (logAutoScroll) el.scrollTop = el.scrollHeight;
    }
    sessionState.prevLogLen = logs.length;
    sessionState.prevLogs = logs.slice();

    // Update count badge
    const badge = document.getElementById('log-count-badge');
    if (badge) badge.textContent = logs.length === 0 ? '' : `${logs.length} line${logs.length === 1 ? '' : 's'}`;

    // Update spawn command display if present in payload
    if (d.last_spawn_cmd !== undefined) {
        const pre = document.getElementById('log-cmd-pre');
        const cmdBtn = document.getElementById('log-cmd-btn');
        if (pre) pre.textContent = d.last_spawn_cmd || '(no spawn command recorded — attach mode or server not yet started)';
        if (cmdBtn) cmdBtn.style.display = d.last_spawn_cmd ? '' : 'none';
    }
}

// ── Badges ───────────────────────────────────────────────────────────────────

function updateBadges(d) {
    const isAttached = d.session_mode === 'attach' && d.active_session_endpoint;
    const ce = cachedElements;

    // Server badge \u2014 GPU temp changes infrequently; skip write when unchanged
    const gpuEntries = Object.entries(d.gpu || {});
    const serverText = (isAttached && gpuEntries.length > 0)
        ? ' GPU ' + Math.max(...gpuEntries.map(([,m]) => m.temp)).toFixed(0) + 'C'
        : '';
    if (serverText !== prevBadgeState.server) {
        prevBadgeState.server = serverText;
        const badgeServer = ce.badgeServer;
        if (badgeServer) {
            badgeServer.textContent = serverText;
            if (!serverText) {
                badgeServer.style.display = 'none';
            }
        }
    }

    // Logs badge \u2014 count only changes when new logs arrive
    const logs = d.logs || [];
    if (logs.length !== prevBadgeState.logs) {
        prevBadgeState.logs = logs.length;
        const badgeLogs = ce.badgeLogs;
        if (badgeLogs) {
            if (logs.length > 0) {
                badgeLogs.textContent = ' ' + logs.length;
                badgeLogs.style.display = '';
            } else {
                badgeLogs.textContent = '';
                badgeLogs.style.display = 'none';
            }
        }
    }
}

// ── Server error details panel ─────────────────────────────────────────

let _lastServerErrorData = null;

function showServerErrorDetails(d) {
    _lastServerErrorData = d;
    const wrapper = document.getElementById('server-error-details-wrapper');
    const btn = document.getElementById('btn-server-error-details');
    if (!wrapper || !btn) return;
    wrapper.style.display = 'flex';
    btn.style.display = '';
}

function hideServerErrorDetails() {
    const wrapper = document.getElementById('server-error-details-wrapper');
    const btn = document.getElementById('btn-server-error-details');
    if (wrapper) wrapper.style.display = 'none';
    if (btn) btn.style.display = 'none';
    _lastServerErrorData = null;
}

function populateServerErrorDetails(data) {
    const wrapper = document.getElementById('server-error-details-wrapper');
    const panel = document.getElementById('server-error-details-panel');
    const body = document.getElementById('server-error-details-body');
    if (!wrapper || !panel || !body || !data) return;

    const err = data.active_session_error || '';
    const cmd = data.last_spawn_cmd || '';
    const logs = data.logs || [];

    const summary = err.length > 200
        ? err.substring(0, 200).trim() + '...'
        : err;

    const serverLogs = logs
        .filter(l => !l.startsWith('[monitor]'))
        .slice(-20);

    let html = '';

    if (summary) {
        html += `<div class="error-summary">${escapeHtml(summary)}</div>`;
    }

    if (cmd) {
        html += `<code class="error-cmd">${escapeHtml(cmd)}</code>`;
    }

    if (serverLogs.length > 0) {
        html += `<code class="error-logs">${serverLogs.map(escapeHtml).join('\n')}</code>`;
    }

    if (!summary && !cmd && serverLogs.length === 0) {
        html += `<div class="error-summary">
No additional context captured. Check the full Logs tab for more details.
</div>`;
    }

    // Doctor findings placeholder (filled async)
    html += `<div class="doctor-findings" id="server-doctor-findings">
        <div style="font-size:9px;color:var(--color-text-muted);">Loading diagnostics…</div>
    </div>`;

    html += `<div style="margin-top:4px;">
    <a href="#" id="error-open-logs-link" style="color:var(--color-primary);font-size:10px;text-decoration:underline;">
        Open Logs tab
    </a>
</div>`;


    setHtml(body, html);

    const link = body.querySelector('#error-open-logs-link');
    if (link) {
        link.addEventListener('click', e => {
            e.preventDefault();
            const logsBtn = document.querySelector('.sidebar-btn[data-tab="logs"]');
            if (logsBtn) logsBtn.click();
        });
    }

    wrapper.style.display = 'flex';
    panel.style.display = '';

    // Load doctor findings async
    loadDoctorFindings('server-doctor-findings');
}

// Bind buttons once (after DOM ready).
document.addEventListener('DOMContentLoaded', () => {
    const detailsBtn = document.getElementById('btn-server-error-details');
    const closeBtn = document.getElementById('server-error-details-close');

    if (detailsBtn) {
        detailsBtn.addEventListener('click', () => {
            if (!_lastServerErrorData) return;
            populateServerErrorDetails(_lastServerErrorData);
            detailsBtn.blur();
        });
    }

    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            const panel = document.getElementById('server-error-details-panel');
            if (panel) panel.style.display = 'none';
        });
    }

    // Welcome screen "Show details"
    const localDetailsBtn = document.getElementById('local-server-error-details-btn');
    const localCloseBtn = document.getElementById('local-server-error-details-close');

    if (localDetailsBtn) {
        localDetailsBtn.addEventListener('click', () => {
            if (!_lastLocalServerErrorData) return;
            populateLocalServerErrorDetails(_lastLocalServerErrorData);
            localDetailsBtn.blur();
        });
    }

    if (localCloseBtn) {
        localCloseBtn.addEventListener('click', () => {
            const panel = document.getElementById('local-server-error-details');
            if (panel) panel.style.display = 'none';
        });
    }
});

// ── Local Server (welcome screen) spawn error bar ──────────────────────

let _lastLocalServerErrorData = null;

function showLocalServerErrorBar(sessionError, d) {
    const bar = document.getElementById('local-server-error-bar');
    const textEl = document.getElementById('local-server-error-bar-text');
    if (!bar || !textEl) return;

    _lastLocalServerErrorData = d;

    const short = (sessionError || '').length > 140
        ? (sessionError || '').substring(0, 140).trim() + '...'
        : (sessionError || '');

    textEl.textContent = short || 'Launch failed';
    bar.style.display = 'flex';
}

function hideLocalServerErrorBar() {
    const bar = document.getElementById('local-server-error-bar');
    const details = document.getElementById('local-server-error-details');
    if (bar) bar.style.display = 'none';
    if (details) details.style.display = 'none';
    _lastLocalServerErrorData = null;
}

function populateLocalServerErrorDetails(data) {
    const details = document.getElementById('local-server-error-details');
    const body = document.getElementById('local-server-error-details-body');
    if (!details || !body || !data) return;

    const err = data.active_session_error || '';
    const cmd = data.last_spawn_cmd || '';
    const logs = data.logs || [];

    const summary = (err || '').length > 220
        ? (err || '').substring(0, 220).trim() + '...'
        : (err || '');

    const serverLogs = logs
        .filter(l => !l.startsWith('[monitor]'))
        .slice(-20);

    let html = '';

    if (summary) {
        html += `<div class="error-summary">${escapeHtml(summary)}</div>`;
    }

    if (cmd) {
        html += `<code class="error-cmd">${escapeHtml(cmd)}</code>`;
    }

    if (serverLogs.length > 0) {
        html += `<code class="error-logs">${serverLogs.map(escapeHtml).join('\n')}</code>`;
    }

    if (!summary && !cmd && serverLogs.length === 0) {
        html += `<div class="error-summary">
No additional context captured. Open the Logs tab or run llama-monitor from a terminal to see full output.
</div>`;
    }

    html += `<div class="doctor-findings" id="local-doctor-findings">
        <div style="font-size:9px;color:var(--color-text-muted);">Loading diagnostics…</div>
    </div>`;

    html += `<div style="margin-top:4px;">
    <a href="#" id="local-error-open-logs-link" style="color:var(--color-primary);font-size:10px;text-decoration:underline;">
        Open Logs tab
    </a>
</div>`;


    setHtml(body, html);

    const link = body.querySelector('#local-error-open-logs-link');
    if (link) {
        link.addEventListener('click', e => {
            e.preventDefault();
            const logsBtn = document.querySelector('.sidebar-btn[data-tab="logs"]');
            if (logsBtn) logsBtn.click();
        });
    }

    details.style.display = 'block';

    loadDoctorFindings('local-doctor-findings');
}

// ── Doctor findings loader ──────────────────────────────────────

async function loadDoctorFindings(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const allFindings = [];

    // Fetch Rapid-MLX doctor findings (platform-guarded)
    try {
        const rmResp = await fetch('/api/rapid-mlx/doctor', {
            headers: window.authHeaders ? window.authHeaders() : {},
        });
        if (rmResp.ok) {
            const rmData = await rmResp.json();
            if (rmData.ok && Array.isArray(rmData.findings)) {
                allFindings.push(...rmData.findings);
            }
        }
    } catch (_err) {
        // Rapid-MLX doctor not available (non-Apple Silicon or not installed)
    }

    // Fetch llama.cpp diagnostics
    try {
        const lcResp = await fetch('/api/sessions/llama-cpp-diagnostics', {
            headers: window.authHeaders ? window.authHeaders() : {},
        });
        if (lcResp.ok) {
            const lcData = await lcResp.json();
            if (lcData.ok && Array.isArray(lcData.findings)) {
                allFindings.push(...lcData.findings);
            }
        }
    } catch (_err) {
        // llama.cpp diagnostics unavailable
    }

    // Fetch Rapid-MLX preset-flag advisor findings (fixable — carries `fix`)
    try {
        const faResp = await fetch('/api/rapid-mlx/flag-advisor', {
            headers: window.authHeaders ? window.authHeaders() : {},
        });
        if (faResp.ok) {
            const faData = await faResp.json();
            if (faData.ok && Array.isArray(faData.findings)) {
                allFindings.push(...faData.findings);
            }
        }
    } catch (_err) {
        // Flag advisor unavailable (no active Rapid-MLX session/preset, etc.)
    }

    if (allFindings.length === 0) {
        setHtml(container, `<div style="font-size:9px;color:var(--color-text-muted);">
Diagnostics endpoints unavailable on this system.
</div>`);
        return;
    }

    // Group findings by section
    const bySection = {};
    for (const f of allFindings) {
        const section = f.section || 'general';
        if (!bySection[section]) bySection[section] = [];
        bySection[section].push(f);
    }

    let html = '';
    for (const [section, findings] of Object.entries(bySection)) {
        html += `<div class="doctor-section">`;
        html += `<div class="doctor-section-title">${escapeHtml(section)}</div>`;
        for (const finding of findings) {
            const iconChar = finding.severity === 'ok' ? '✓' : finding.severity === 'warning' ? '⚠' : '✗';
            html += `<div class="doctor-finding" data-severity="${escapeHtml(finding.severity)}">`;
            html += `<span class="doctor-finding-icon">${iconChar}</span>`;
            html += `<span class="doctor-finding-message">${escapeHtml(finding.message)}</span>`;
            if (finding.fix) {
                html += `<button class="doctor-fix-btn" data-fix="${escapeHtml(finding.fix)}">Apply fix</button>`;
            }
            html += `</div>`;
        }
        html += `</div>`;
    }


    setHtml(container, html);

    // Wire up fix buttons
    container.querySelectorAll('.doctor-fix-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            const fixAction = btn.getAttribute('data-fix');
            if (!fixAction) return;
            btn.textContent = 'Applying…';
            btn.disabled = true;
            try {
                const resp = await fetch('/api/diagnostics/apply-fix', {
                    method: 'POST',
                    headers: window.authHeaders
                        ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
                        : { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ fix: fixAction }),
                });
                const result = await resp.json();
                if (result.ok) {
                    btn.textContent = 'Applied';
                    btn.style.color = '#4ade80';
                    btn.style.borderColor = 'rgba(74, 222, 128, 0.4)';
                } else {
                    btn.textContent = result.error || 'Failed';
                    btn.style.color = '#ef4444';
                    btn.style.borderColor = 'rgba(239, 68, 68, 0.4)';
                }
            } catch (err) {
                btn.textContent = 'Error';
                btn.style.color = '#ef4444';
                console.error('Apply fix failed:', err);
            }
        });
    });
}
