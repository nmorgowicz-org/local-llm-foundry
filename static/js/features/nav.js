// ── Navigation ────────────────────────────────────────────────────────────────
// Tab switching and sidebar collapse.

import { chat, contextCapacityTokens, lastLlamaMetrics, lastSystemMetrics, sessionState, setWsData, wsData } from '../core/app-state.js';
import { chatScroll } from './chat-render.js';
import { showSessionPanel, hideSessionPanel } from './chat-sessions-sidebar.js';
import { isFocusModeActive, exitFocusMode } from './chat-focus-mode.js';
import { renderCapabilityPopover } from './dashboard-render.js';
import { showToast } from './toast.js';
import Router from './router.js';
import { isLlamaTelemetryForTarget } from './llama-cpp-details.js';
import { estimateChatContextPct } from './chat-state.js';
import { rapidModelInfo } from './rapid-model-download.js';

export function switchTab(name) {
    if (name !== 'chat' && isFocusModeActive()) exitFocusMode();

    const page = document.getElementById('page-' + name);

    // Handle modal tabs (no corresponding page div)
    if (!page) {
        document.querySelectorAll('.sidebar-btn').forEach(b => b.classList.remove('active'));
        const sidebarButton = document.querySelector(`.sidebar-btn[data-tab="${name}"]`);
        if (sidebarButton) sidebarButton.classList.add('active');
        if (name === 'settings') {
            Router.navigate('/settings');
        }
        return;
    }

    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.sidebar-btn').forEach(b => b.classList.remove('active'));

    page.classList.add('active');

    const sidebarButton = document.querySelector(`.sidebar-btn[data-tab="${name}"]`);
    if (sidebarButton) sidebarButton.classList.add('active');

    if (name === 'chat') {
        showSessionPanel();
    } else {
        hideSessionPanel();
    }

    // Scroll chat to bottom when entering chat page (no tab switch = no re-render)
    if (name === 'chat') {
        setTimeout(() => chatScroll(true), 50);
    }
}

function toggleSidebarCollapse() {
    const sidebar = document.getElementById('sidebar-nav');
    const icon = document.querySelector('.sidebar-collapse-icon');

    sidebar.classList.toggle('collapsed');
    document.body.classList.toggle('sidebar-collapsed');

    const isCollapsed = sidebar.classList.contains('collapsed');
    localStorage.setItem('sidebarCollapsed', isCollapsed.toString());

    if (icon) {
        icon.textContent = isCollapsed ? '▶' : '◀';
    }
}

function restoreSidebarState() {
    const sidebar = document.getElementById('sidebar-nav');
    const icon = document.querySelector('.sidebar-collapse-icon');
    const isCollapsed = localStorage.getItem('sidebarCollapsed') === 'true';

    if (isCollapsed) {
        sidebar.classList.add('collapsed');
        document.body.classList.add('sidebar-collapsed');
        if (icon) icon.textContent = '▶';
    }
}

// ── Endpoint status popover ──────────────────────────────────────────────────

function initEndpointStatus() {
    const endpointStatus = document.getElementById('endpoint-status');
    const endpointStatusWrap = endpointStatus?.closest('.endpoint-status-wrap');
    const popover = document.getElementById('capability-popover');
    if (!endpointStatus || !endpointStatusWrap || !popover) return;

    function positionPopover() {
        const rect = endpointStatusWrap.getBoundingClientRect();
        popover.style.top = (rect.bottom + 8) + 'px';
        popover.style.left = Math.min(rect.left, window.innerWidth - 370) + 'px';
    }

    endpointStatus.addEventListener('click', event => {
        event.stopPropagation();
        const open = endpointStatusWrap.classList.toggle('open');
        endpointStatus.setAttribute('aria-expanded', open ? 'true' : 'false');
        if (open) {
            popover.classList.add('open');
            renderCapabilityPopover(wsData, wsData?.llama);
            positionPopover();
        } else {
            popover.classList.remove('open');
        }
    });

    document.addEventListener('click', event => {
        if (!event.target.closest('.endpoint-status-wrap')) {
            endpointStatusWrap.classList.remove('open');
            endpointStatus.setAttribute('aria-expanded', 'false');
            popover.classList.remove('open');
        }
    });
}

function deriveTabCtxPct(tab, capacity) {
    return estimateChatContextPct(tab, capacity) || 0;
}


function refreshMonitoringChip(mode, isManual, hasActiveEndpoint) {
    const chip = document.getElementById('nav-monitoring-chip');
    if (!chip) return;

    const effectiveMode = mode ?? (wsData?.sleep_mode ? 'sleep' : 'off');
    const isSleeping = effectiveMode === 'sleep';
    const isLogsOnly = effectiveMode === 'logs-only';

    chip.style.display = hasActiveEndpoint ? 'inline-flex' : 'none';
    if (!hasActiveEndpoint) return;

    const dot = document.getElementById('nav-monitoring-dot');
    const label = document.getElementById('nav-monitoring-label');

    const unavailable = chip.getAttribute('data-unavailable') === 'true';
    chip.classList.toggle('is-paused', isSleeping || isLogsOnly);
    chip.setAttribute('aria-pressed', (isSleeping || isLogsOnly) ? 'true' : 'false');

    if (dot) {
        if (isSleeping) {
            dot.className = 'status-dot warning';
        } else if (isLogsOnly) {
            dot.className = 'status-dot info';
        } else {
            dot.className = 'status-dot ok';
        }
    }
    if (label) {
        if (isSleeping) {
            label.textContent = 'Paused';
        } else if (isLogsOnly) {
            label.textContent = 'Logs only';
        } else {
            label.textContent = 'Monitoring';
        }
    }

    if (unavailable) {
        chip.setAttribute('title', 'Monitoring control is not available on this server.');
    } else if (isLogsOnly) {
        chip.setAttribute('title', 'Logs-only mode — only live logs active. Click to change mode.');
    } else if (isSleeping && isManual) {
        chip.setAttribute('title', 'Monitoring paused (manual) — inference server keeps running. Click to change mode.');
    } else if (isSleeping) {
        chip.setAttribute('title', 'Monitoring paused (idle timeout) — inference server keeps running. Click to resume.');
    } else {
        chip.setAttribute('title', 'Dashboard monitoring active — click to cycle modes.');
    }
}

function refreshMemoryPressureChip() {
    const wrap = document.getElementById('nav-memory-pressure-wrap');
    const chip = document.getElementById('nav-memory-pressure-chip');
    if (!chip) return;
    const sys = lastSystemMetrics || {};
    const level = sys.memory_pressure_level || '';
    const visible = level === 'warning' || level === 'critical';
    if (wrap) wrap.style.display = visible ? 'inline-flex' : 'none';
    else chip.style.display = visible ? 'inline-flex' : 'none';
    if (!visible) return;

    const dot = document.getElementById('nav-memory-pressure-dot');
    const label = document.getElementById('nav-memory-pressure-label');
    if (dot) dot.className = 'status-dot ' + (level === 'critical' ? 'error' : 'warning');
    if (label) label.textContent = level === 'critical' ? 'Memory critical' : 'Memory pressure';

    const free = Number(sys.memory_free_gb || 0).toFixed(1);
    const wired = Number(sys.memory_wired_gb || 0);
    const compressed = Number(sys.memory_compressor_gb || 0).toFixed(1);
    const isCritical = level === 'critical';

    const hcTitle = document.getElementById('nav-memory-pressure-hovercard-title');
    const hcStats = document.getElementById('nav-memory-pressure-hovercard-stats');
    const hcBody = document.getElementById('nav-memory-pressure-hovercard-body');
    if (hcTitle) hcTitle.textContent = isCritical ? 'Memory Critical' : 'Memory Pressure';

    if (hcStats) {
        const purgeableGb = Number(sys.memory_purgeable_gb || 0);
        const inactiveGb = Number(sys.memory_inactive_gb || 0);
        const entries = [
            ['Free', `${free} GB`],
            ['Wired', wired > 0 ? `${wired.toFixed(1)} GB` : '—'],
            ['Compressed', Number(compressed) > 0 ? `${compressed} GB` : '—'],
            ['Purgeable', purgeableGb > 0 ? `${purgeableGb.toFixed(1)} GB` : '—'],
            ['Inactive', inactiveGb > 0 ? `${inactiveGb.toFixed(1)} GB` : '—'],
        ];
        hcStats.textContent = '';
        entries.forEach(([k, v]) => {
            const row = document.createElement('div');
            row.className = 'mem-hc-row';
            const key = document.createElement('span');
            key.className = 'mem-hc-key';
            key.textContent = k;
            const val = document.createElement('span');
            val.className = 'mem-hc-val';
            val.textContent = v;
            row.appendChild(key);
            row.appendChild(val);
            hcStats.appendChild(row);
        });
    }

    if (hcBody) {
        const advice = sys.memory_pressure_advice || (isCritical
            ? 'Disable mlock in your preset or reduce context to free wired memory. Use "Free Memory" to reclaim inactive pages.'
            : 'Reduce context, pause downloads, or disable mlock in your preset to relieve pressure.');
        hcBody.textContent = advice;
    }

    // Wire purge button once
    const navPurgeBtn = document.getElementById('nav-pressure-purge-btn');
    if (navPurgeBtn && !navPurgeBtn._wired) {
        navPurgeBtn._wired = true;
        navPurgeBtn.addEventListener('click', async (e) => {
            e.stopPropagation();
            if (navPurgeBtn._purging) return;
            navPurgeBtn._purging = true;
            const statusEl = document.getElementById('nav-pressure-purge-status');
            navPurgeBtn.textContent = 'Requesting…';
            if (statusEl) { statusEl.style.display = ''; statusEl.textContent = 'Waiting for macOS admin dialog…'; }
            try {
                const adminToken = await fetchDbAdminTokenForSystemAction();
                if (!adminToken) throw new Error('Authentication required.');
                const res = await fetch('/system/purge', {
                    method: 'POST',
                    headers: {
                        Authorization: `Bearer ${adminToken}`,
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ confirm: 'purge-memory' }),
                });
                const data = await res.json();
                if (statusEl) {
                    statusEl.textContent = data.message || (data.ok ? 'Done.' : 'Failed.');
                    statusEl.className = 'mem-pressure-hovercard-purge-status' + (data.ok ? ' purge-ok' : ' purge-err');
                }
            } catch {
                if (statusEl) { statusEl.textContent = 'Request failed.'; statusEl.className = 'mem-pressure-hovercard-purge-status purge-err'; }
            } finally {
                navPurgeBtn._purging = false;
                navPurgeBtn.textContent = 'Free Memory';
                setTimeout(() => { if (statusEl) statusEl.style.display = 'none'; }, 6000);
            }
        });
    }
}

async function fetchDbAdminTokenForSystemAction() {
    const tokenResp = await fetch('/api/db/admin-token', {
        headers: window.authHeaders ? window.authHeaders() : {},
    });
    const tokenData = tokenResp.ok ? await tokenResp.json().catch(() => ({})) : {};
    return tokenData.token || null;
}

// Loaded identity is distinct from the OpenAI served alias. Never infer a
// quantization, context limit, or runtime lane from the model's name.
const modelText = value => typeof value === 'string' && value.trim().length <= 512 ? value.trim() : '';
const modelBasename = value => modelText(value).split(/[\\/]/).filter(Boolean).pop() || '';
const modelRepo = value => {
    const text = modelText(value).split('@')[0];
    return /^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/.test(text) ? text : '';
};
function physicalModel(value) {
    const text = modelText(value);
    const snapshot = text.match(/(?:^|[/\\])models--([^/\\]+)--([^/\\]+)(?:[/\\]|$)/);
    if (snapshot) return { repo: modelRepo(`${snapshot[1]}/${snapshot[2]}`), model: modelRepo(`${snapshot[1]}/${snapshot[2]}`) };
    const repo = modelRepo(text);
    if (repo) return { repo, model: repo };
    if (/^(?:[/\\.~]|[A-Za-z]:[\\/])/.test(text)) return { repo: '', model: modelBasename(text) };
    return { repo: '', model: '' };
}

export function resolveRapidLoadedModel({ sample, config, identity, modelInfo } = {}) {
    const runtime = sample?.backend_details?.runtime_facts || {};
    const launch = sample?.backend_details?.launch_facts || {};
    const source = config?.model_source;
    const view = config?.model_source_view;
    const served = modelText(launch.served_model_name) || modelText(config?.served_model_name);
    const statusModel = modelText(sample?.model);
    const sourceAlias = modelText(launch.source_alias)
        || (source?.kind === 'alias' ? modelText(source.value) : '')
        || (view?.kind === 'alias' ? modelText(view.canonical_identity) : '');
    const candidates = [
        runtime.repo_id, runtime.model_path, launch.repo_id, launch.model_path, view?.repo_id,
        source?.kind === 'hugging_face_repo' ? source.repo_id : '',
        source?.kind === 'mlx_directory' ? source.path : '', view?.local_path,
        // /v1/status.model is a served name, even when it looks like owner/repo.
        // A distinct session source or a concrete local snapshot can carry identity.
        identity !== served && (identity !== statusModel || /^(?:[/\\.~]|[A-Za-z]:[\\/])/.test(modelText(identity))) ? identity : '',
    ];
    const physical = candidates.map(physicalModel).find(value => value.model) || { model: '', repo: '' };
    const sessionSource = identity !== served ? modelText(identity) : '';
    const unresolved = sourceAlias || (!physical.model ? sessionSource || statusModel : '');
    // model-status passes owner/repo straight through; that is not alias resolution
    // evidence. Only bare catalog aliases may promote an unknown served name.
    const lookupSource = physical.repo || (unresolved && !/[/\\]/.test(unresolved) ? unresolved : '');
    const resolvedRepo = modelInfo?.ok ? modelRepo(modelInfo.repo_id) : '';
    const repo = physical.repo || resolvedRepo;
    const model = repo || physical.model;
    const alias = served || (statusModel && statusModel !== model ? (physicalModel(statusModel).model ? statusModel : modelBasename(statusModel)) : '')
        || (!model ? modelText(identity) || sourceAlias : '');
    const rows = [['Engine', 'Rapid-MLX'], ['Model', model || 'Physical model not reported']];
    if (repo) rows.push(['Repository', repo]);
    if (alias && alias !== model) rows.push(['Alias', alias]);
    const revision = modelText(runtime.revision) || modelText(launch.revision)
        || modelText(view?.revision) || (source?.kind === 'hugging_face_repo' ? modelText(source.revision) : '');
    if (revision) rows.push(['Revision', revision]);
    const version = modelText(runtime.version) || modelText(launch.version);
    if (version) rows.push(['Runtime version', version]);
    // Runtime-reported quantization is authoritative; the catalog value describes
    // the catalog entry for this repo, not what the process was observed to load.
    const runtimeQuant = modelText(runtime.quantization);
    const catalogQuant = resolvedRepo === repo ? modelText(modelInfo?.quant) : '';
    if (runtimeQuant) rows.push(['Quantization', runtimeQuant]);
    else if (catalogQuant) rows.push(['Quantization (catalog)', catalogQuant]);
    const positive = value => typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : null;
    const context = positive(runtime.context_length);
    const configuredContext = positive(launch.context_length) ?? positive(config?.context_length);
    if (context) rows.push(['Context', context.toLocaleString()]);
    else if (configuredContext) rows.push(['Context (configured)', configuredContext.toLocaleString()]);
    const lane = [modelText(runtime.model_type), modelText(runtime.engine_type)].filter(value => value && value !== 'unknown');
    if (lane.length) rows.push(['Runtime lane', lane.join(' · ')]);
    const spec = runtime.speculative_config || launch.speculative_config || config?.speculative_config;
    if (spec && typeof spec === 'object') {
        const parts = [modelText(spec.method), positive(spec.num_speculative_tokens) ? `${spec.num_speculative_tokens} tokens` : '',
            spec.model ? physicalModel(spec.model).model || modelBasename(spec.model) : ''].filter(Boolean);
        if (parts.length) rows.push([runtime.speculative_config ? 'Speculative decoding' : 'Speculative decoding (configured)', parts.join(' · ')]);
    }
    return { model, alias, lookupSource, rows, display: model || (alias ? `Alias: ${alias}` : 'Model not reported') };
}

let rapidLoadedLookup = { key: '', info: null };

// Also called by the server-state path before monitor/overlay early returns.
export function refreshLoadedModel() {
    const backend = wsData?.backend;
    const running = !!wsData?.active_session_id && ['running', 'disconnected'].includes(wsData?.active_session_status);
    const indicator = document.getElementById('engine-indicator');
    const labelEl = indicator?.querySelector('.engine-indicator-label');
    const dotEl = indicator?.querySelector('.engine-indicator-dot');
    const popover = document.getElementById('engine-model-popover');
    const facts = document.getElementById('engine-model-facts');
    const header = document.getElementById('server-model-identity');
    if (!running || !backend || wsData?.endpoint_kind !== 'Local') {
        rapidLoadedLookup = { key: '', info: null };
        if (indicator) indicator.style.display = 'none';
        facts?.replaceChildren();
        if (popover?.matches(':popover-open')) popover.hidePopover();
        if (backend === 'rapid_mlx' && header) {
            header.textContent = '';
            header.hidden = true;
            header.style.display = 'none';
        }
        return;
    }
    let rows, display, alias;
    let active;
    if (backend === 'rapid_mlx') {
        // The server filters inference by active session before forwarding it.
        // Do not consult retained lastRapid/llama snapshots after a source switch.
        const sample = wsData?.inference?.backend === 'rapid_mlx' ? wsData.inference : null;
        const config = wsData?.session_mode === 'spawn'
            ? sessionState.presets.find(preset => preset.id === wsData.active_session_preset_id && preset.backend === 'rapid_mlx')?.rapid_mlx
            : null;
        const input = { sample, config, identity: wsData?.active_session_model_identity };
        let resolved = resolveRapidLoadedModel(input);
        const key = JSON.stringify([wsData.active_session_id, wsData.active_session_endpoint_tag ?? wsData.active_session_endpoint,
            input.identity, sample?.model, resolved.lookupSource]);
        if (rapidLoadedLookup.key !== key) {
            rapidLoadedLookup = { key, info: null };
            // One attempt per identity/target, including failure. The shared helper
            // memoizes successes; never spawn a catalog query on every WS tick.
            if (resolved.lookupSource) {
                rapidModelInfo(resolved.lookupSource).then(info => {
                    if (rapidLoadedLookup.key !== key || wsData?.backend !== 'rapid_mlx') return;
                    rapidLoadedLookup.info = info;
                    refreshLoadedModel();
                });
            }
        }
        resolved = resolveRapidLoadedModel({ ...input, modelInfo: rapidLoadedLookup.info });
        ({ rows, display, alias } = resolved);
        active = (sample?.running_requests || 0) > 0 || (sample?.generation_tokens_per_second || 0) > 0;
        if (header) {
            header.textContent = `Rapid-MLX · ${display}`;
            header.hidden = false;
            header.style.display = '';
        }
    } else {
        rapidLoadedLookup = { key: '', info: null };
        const runtime = isLlamaTelemetryForTarget(lastLlamaMetrics, wsData.active_session_id,
            wsData.active_session_endpoint_tag ?? wsData.active_session_endpoint) ? lastLlamaMetrics?.runtime_facts : null;
        const model = modelBasename(runtime?.model_name);
        alias = modelBasename(runtime?.model_alias);
        const identity = modelBasename(wsData.active_session_model_identity);
        display = model || identity || 'Model not reported';
        rows = [['Engine', 'llama.cpp'], ['Model', model || 'Model file not reported by server']];
        if (alias) rows.push(['Alias', alias]);
        else if (identity) rows.push(['Model identity', identity]);
        active = !!lastLlamaMetrics?.slot_generation_active || (lastLlamaMetrics?.slots_processing || 0) > 0
            || (lastLlamaMetrics?.generation_tokens_per_sec || 0) > 0;
    }
    if (!indicator || !labelEl || !dotEl) return;
    const engine = backend === 'rapid_mlx' ? 'Rapid-MLX' : 'llama.cpp';
    const friendly = display.startsWith('Alias: ') ? display : modelBasename(display).replace(/\.(gguf|safetensors)$/i, '');
    labelEl.textContent = `${engine} · ${friendly.length > 24 ? friendly.slice(0, 22) + '…' : friendly}`;
    dotEl.className = `engine-indicator-dot ${active ? 'live' : 'idle-active'}`;
    indicator.setAttribute('data-tooltip', `${engine}\n${display}${alias && display !== `Alias: ${alias}` ? '\nAlias: ' + alias : ''}\n${active ? 'Generating' : 'Idle'} · Click for model details`);
    facts?.replaceChildren(...rows.flatMap(([label, value]) => {
        const term = document.createElement('dt');
        const definition = document.createElement('dd');
        term.textContent = label;
        definition.textContent = value;
        return [term, definition];
    }));
    indicator.style.display = 'inline-flex';
}

export function refreshTopCockpit() {
    refreshLoadedModel();
    const cockpit = document.getElementById('nav-cockpit');
    if (!cockpit) return;

    const stateEl = document.getElementById('nav-cockpit-state');
    const throughputEl = document.getElementById('nav-cockpit-throughput');
    const specEl = document.getElementById('nav-cockpit-spec');
    const contextEl = document.getElementById('nav-cockpit-context');

    const hasActiveEndpoint = !!wsData?.active_session_id;
    const l = hasActiveEndpoint ? lastLlamaMetrics : null;
    const promptRate = l?.prompt_tokens_per_sec || 0;
    const genRate = l?.generation_tokens_per_sec || 0;
    const promptDisplayRate = promptRate > 0 ? promptRate : (l?.last_prompt_tokens_per_sec || 0);
    const genDisplayRate = genRate > 0 ? genRate : (l?.last_generation_tokens_per_sec || 0);
    const generationActive = !!l?.slot_generation_active || (l?.slots_processing || 0) > 0 || genRate > 0;


    const wsMode = wsData?.mode ?? (wsData?.sleep_mode ? 'sleep' : 'off');
    const isSleeping = wsMode === 'sleep';
    const isLogsOnly = wsMode === 'logs-only';
    let label = 'idle';
    let stateClass = 'idle';

    if (isSleeping) {
        label = 'paused';
        stateClass = 'sleep';
    } else if (isLogsOnly) {
        label = 'logs';
        stateClass = 'logs-only';
    } else if (!hasActiveEndpoint) {
        label = 'attach';
    } else if (promptRate > 0 && genRate <= 0) {
        label = 'prompting';
        stateClass = 'live';
    } else if (generationActive) {
        label = 'generating';
        stateClass = 'live';
    }

    if (stateEl) {
        stateEl.textContent = label;
        stateEl.className = 'metric-live-chip nav-cockpit-state ' + stateClass;
    }
    cockpit.classList.toggle('is-live', stateClass === 'live');
    cockpit.classList.toggle('is-idle', stateClass !== 'live' && stateClass !== 'sleep' && stateClass !== 'logs-only');
    cockpit.classList.toggle('has-session', hasActiveEndpoint);

    // Update monitoring chip in nav-right
    refreshMonitoringChip(wsMode, wsData?.sleep_mode_manual, hasActiveEndpoint);
    refreshMemoryPressureChip();

    if (throughputEl) {
        // G first, then P — matching the Speed card's TG-over-PP order.
        throughputEl.textContent = 'G ' + (genDisplayRate > 0 ? genDisplayRate.toFixed(0) : '—') + ' · P ' + (promptDisplayRate > 0 ? promptDisplayRate.toFixed(0) : '—');
    }

    if (specEl) {
        const tpd = l?.tokens_per_decode ?? 0;
        if (tpd > 1.05) {
            specEl.textContent = tpd.toFixed(2) + '× S';
            specEl.classList.remove('hidden');
        } else {
            specEl.classList.add('hidden');
        }
    }

    const capacity = hasActiveEndpoint ? (contextCapacityTokens || l?.context_capacity_tokens || l?.kv_cache_max || 0) : 0;
    let worstCtx = 0;
    if (capacity > 0) {
        worstCtx = (chat.tabs || []).reduce((max, tab) => Math.max(max, deriveTabCtxPct(tab, capacity)), 0);
    }
    if (contextEl) {
        contextEl.textContent = 'Ctx ' + (worstCtx > 0 ? Math.round(worstCtx) + '%' : '—');
        contextEl.title = worstCtx > 0 ? 'Highest chat context pressure across tabs' : 'No live context pressure available';
    }

}

// ── Sidebar drag-resize ───────────────────────────────────────────────────────

const SIDEBAR_RESIZE_KEY = 'appNavWidth';
const SIDEBAR_MIN = 140;
const SIDEBAR_MAX = 320;

function setSidebarWidth(px) {
    const clamped = Math.min(SIDEBAR_MAX, Math.max(SIDEBAR_MIN, px));
    document.documentElement.style.setProperty('--sidebar-width-expanded', clamped + 'px');
    localStorage.setItem(SIDEBAR_RESIZE_KEY, clamped);
}

function initSidebarResize() {
    const handle = document.getElementById('sidebar-resize-handle');
    const sidebar = document.getElementById('sidebar-nav');
    if (!handle || !sidebar) return;

    const saved = Number(localStorage.getItem(SIDEBAR_RESIZE_KEY));
    if (saved >= SIDEBAR_MIN && saved <= SIDEBAR_MAX) {
        setSidebarWidth(saved);
    }

    let startX = 0;
    let startWidth = 0;

    handle.addEventListener('mousedown', e => {
        if (e.button !== 0) return;
        e.preventDefault();
        startX = e.clientX;
        startWidth = sidebar.getBoundingClientRect().width;
        sidebar.classList.add('is-resizing');
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
    });

    document.addEventListener('mousemove', e => {
        if (!sidebar.classList.contains('is-resizing')) return;
        setSidebarWidth(startWidth + (e.clientX - startX));
    });

    document.addEventListener('mouseup', () => {
        if (!sidebar.classList.contains('is-resizing')) return;
        sidebar.classList.remove('is-resizing');
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
    });
}

// ── Public API ────────────────────────────────────────────────────────────────

export function initNav() {
    // Bind sidebar tab switching with Router so URLs stay in sync
    document.querySelectorAll('.sidebar-btn[data-tab]').forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            switch (tab) {
                case 'server':
                    Router.navigate('/server');
                    break;
                case 'chat':
                    Router.navigate('/chat');
                    break;
                case 'logs':
                    Router.navigate('/logs');
                    break;
                case 'settings':
                    Router.navigate('/settings');
                    break;
                default:
                    // For any non-mapped tab, fall back to existing switchTab
                    switchTab(tab);
                    break;
            }
        });
    });

    // Bind sidebar collapse
    const collapseBtn = document.getElementById('sidebar-collapse-btn');
    if (collapseBtn) {
        collapseBtn.addEventListener('click', toggleSidebarCollapse);
    }

    // Memory pressure chip: hover to preview, click to pin open
    const memChip = document.getElementById('nav-memory-pressure-chip');
    const memHovercard = document.getElementById('nav-memory-pressure-hovercard');
    if (memChip && memHovercard) {
        let _pinned = false;

        // Move hovercard into body so its fixed positioning is truly viewport-relative
        // (prevents being clipped by nav strip overflow / containing-block issues)
        document.body.appendChild(memHovercard);

        function _positionHovercard() {
            const rect = memChip.getBoundingClientRect();
            memHovercard.style.top = (rect.bottom + 8) + 'px';
            const rightEdge = window.innerWidth - rect.right;
            memHovercard.style.right = Math.max(8, rightEdge) + 'px';
        }

        function _openHovercard() {
            _positionHovercard();
            memHovercard.classList.add('mem-pressure-hovercard--open');
        }
        function _closeHovercard() { if (!_pinned) memHovercard.classList.remove('mem-pressure-hovercard--open'); }

        memChip.addEventListener('mouseenter', _openHovercard);
        memChip.addEventListener('mouseleave', () => {
            if (!_pinned) setTimeout(() => {
                if (!memHovercard.matches(':hover')) _closeHovercard();
            }, 80);
        });

        memHovercard.addEventListener('mouseenter', _openHovercard);
        memHovercard.addEventListener('mouseleave', () => { if (!_pinned) _closeHovercard(); });

        memChip.addEventListener('click', (e) => {
            e.stopPropagation();
            _pinned = !_pinned;
            if (_pinned) _openHovercard();
            else _closeHovercard();
        });
        document.addEventListener('click', (e) => {
            if (!memChip.contains(e.target) && !memHovercard.contains(e.target)) {
                _pinned = false;
                _closeHovercard();
            }
        });
    }

    // Bind nav logo — returns to home (setup) view when in monitor view
    const navLogo = document.getElementById('nav-logo');
    if (navLogo) {
        navLogo.addEventListener('click', event => {
            event.preventDefault();
            if (!document.body.classList.contains('setup-active')) {
                Router.navigate('/');
            }
        });
    }

    const navHomeBtn = document.getElementById('nav-home-btn');
    if (navHomeBtn) {
        navHomeBtn.addEventListener('click', () => {
            Router.navigate('/');
        });
        // Hide on welcome screen, show on dashboard
        const observer = new MutationObserver(() => {
            navHomeBtn.style.display = document.body.classList.contains('setup-active') ? 'none' : '';
        });
        observer.observe(document.body, { attributes: true, attributeFilter: ['class'] });
        // Set initial state
        navHomeBtn.style.display = document.body.classList.contains('setup-active') ? 'none' : '';
    }

    const cockpit = document.getElementById('nav-cockpit');
    if (cockpit) {
        cockpit.addEventListener('click', () => {
            Router.navigate('/server');
        });
        cockpit.addEventListener('keydown', (e) => {
            if (e.key !== 'Enter' && e.key !== ' ') return;
            e.preventDefault();
            Router.navigate('/server');
        });
    }

    const monitoringChip = document.getElementById('nav-monitoring-chip');
    if (monitoringChip) {
        monitoringChip.addEventListener('click', async (e) => {
            e.stopPropagation();

            if (monitoringChip.getAttribute('data-unavailable') === 'true') return;
            if (monitoringChip.getAttribute('data-disabled') === 'true') return;
            monitoringChip.setAttribute('data-disabled', 'true');

            try {
                const auth = window.authHeaders ? window.authHeaders() : {};
                const res = await fetch('/api/sleep-mode/toggle', {
                    method: 'POST',
                    headers: { ...auth, 'Content-Type': 'application/json' },
                });

                if (!res.ok) {
                    if (res.status === 404) {
                        showToast('Monitoring control is not available on this server.', 'info');
                        monitoringChip.setAttribute('data-unavailable', 'true');
                        monitoringChip.setAttribute('title', 'Monitoring control is not available on this server.');
                    } else {
                        showToast('Failed to toggle monitoring.', 'error');
                    }
                    return;
                }

                let nextMode = 'off';
                let nextSleepModeManual = false;
                try {
                    const data = await res.json();
                    nextMode = data.mode || (data.sleep_mode ? 'sleep' : 'off');
                    nextSleepModeManual = data.sleep_mode_manual ?? !!data.enabled;
                } catch (_) {
                    nextMode = 'sleep';
                }

                setWsData({
                    ...(wsData || {}),
                    mode: nextMode,
                    sleep_mode: nextMode !== 'off',
                    sleep_mode_manual: nextSleepModeManual,
                });
                refreshTopCockpit();

                const messages = {
                    'off': 'Monitoring resumed.',
                    'logs-only': 'Logs-only mode — only live logs active.',
                    'sleep': 'Monitoring paused — inference server keeps running.',
                };
                showToast(messages[nextMode] || ('Mode: ' + nextMode), 'success');
            } catch (_err) {
                showToast('Monitoring toggle failed (network error).', 'error');
            } finally {
                if (monitoringChip.getAttribute('data-unavailable') !== 'true') {
                    setTimeout(() => monitoringChip.removeAttribute('data-disabled'), 600);
                }
            }
        });
    }

    restoreSidebarState();
    initSidebarResize();
    initEndpointStatus();
    refreshTopCockpit();
}
