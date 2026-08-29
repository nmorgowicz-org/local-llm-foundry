// ── Navigation ────────────────────────────────────────────────────────────────
// Tab switching and sidebar collapse.

import { chat, contextCapacityTokens, lastLlamaMetrics, lastSystemMetrics, metricSeries, setWsData, wsData } from '../core/app-state.js';
import { chatScroll } from './chat-render.js';
import { showSessionPanel, hideSessionPanel } from './chat-sessions-sidebar.js';
import { isFocusModeActive, exitFocusMode } from './chat-focus-mode.js';
import { renderCapabilityPopover } from './dashboard-render.js';
import { showToast } from './toast.js';
import Router from './router.js';

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
    if (!tab || !capacity) return 0;
    const asst = (tab.messages || []).filter(m => m.role === 'assistant' && !m.compaction_marker);
    if (!asst.length) return tab.last_ctx_pct || 0;
    // Use tab-level cumulative totals (most accurate); fall back to summing message fields.
    const totalInput = tab.total_input_tokens
        || asst.reduce((sum, m) => sum + (m.input_tokens || 0), 0);
    const totalOutput = tab.total_output_tokens
        || asst.reduce((sum, m) => sum + (m.output_tokens || 0), 0);
    return Math.min(200, (totalInput + totalOutput) / capacity * 100);
}

function buildCockpitSparkline(points) {
    if (!points || points.length < 2) return '';
    const width = 120;
    const height = 28;
    const max = Math.max(...points, 1);
    const step = width / (points.length - 1);
    const currentValue = points[points.length - 1];
    const currentX = width;
    const currentY = height - ((currentValue / max) * (height - 6)) - 3;
    const path = points.map((value, index) => {
        const x = index * step;
        const y = height - ((value / max) * (height - 6)) - 3;
        return (index === 0 ? 'M' : 'L') + x.toFixed(2) + ' ' + y.toFixed(2);
    }).join(' ');
    return [
        '<path class="sparkline-fill live-output" d="' + path + ' L 120 28 L 0 28 Z" fill="currentColor"></path>',
        '<path class="sparkline-line live-output" d="' + path + '"></path>',
        '<line class="sparkline-current-trace live-output" x1="' + Math.max(currentX - 8, 0).toFixed(2) + '" y1="' + currentY.toFixed(2) + '" x2="' + currentX.toFixed(2) + '" y2="' + currentY.toFixed(2) + '"></line>',
        '<circle class="sparkline-current-halo live-output" cx="' + currentX.toFixed(2) + '" cy="' + currentY.toFixed(2) + '" r="7.4"></circle>',
        '<circle class="sparkline-current live-output" cx="' + currentX.toFixed(2) + '" cy="' + currentY.toFixed(2) + '" r="3.6"></circle>',
        '<circle class="sparkline-current-core live-output" cx="' + currentX.toFixed(2) + '" cy="' + currentY.toFixed(2) + '" r="1.2"></circle>',
    ].join('');
}

function buildIdleCockpitPoints() {
    return [0.18, 0.42, 0.3, 0.54, 0.36, 0.62, 0.4, 0.5, 0.34, 0.46];
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

export function refreshTopCockpit() {
    const cockpit = document.getElementById('nav-cockpit');
    if (!cockpit) return;

    const stateEl = document.getElementById('nav-cockpit-state');
    const throughputEl = document.getElementById('nav-cockpit-throughput');
    const specEl = document.getElementById('nav-cockpit-spec');
    const contextEl = document.getElementById('nav-cockpit-context');
    const gpuEl = document.getElementById('nav-cockpit-gpu');
    const sparkEl = document.getElementById('nav-cockpit-spark');

    const hasActiveEndpoint = !!wsData?.active_session_id;
    const l = hasActiveEndpoint ? lastLlamaMetrics : null;
    const promptRate = l?.prompt_tokens_per_sec || 0;
    const genRate = l?.generation_tokens_per_sec || 0;
    const promptDisplayRate = promptRate > 0 ? promptRate : (l?.last_prompt_tokens_per_sec || 0);
    const genDisplayRate = genRate > 0 ? genRate : (l?.last_generation_tokens_per_sec || 0);
    const generationActive = !!l?.slot_generation_active || (l?.slots_processing || 0) > 0 || genRate > 0;

    // ── Engine · Model indicator ────────────────────────────────────────
    {
      const indicator = document.getElementById('engine-indicator');
      const labelEl = indicator?.querySelector('.engine-indicator-label');
      const dotEl = indicator?.querySelector('.engine-indicator-dot');
      if (indicator && labelEl && dotEl) {
        const backend = wsData?.backend || null;
        const endpointKind = wsData?.endpoint_kind || null;
        const modelIdentity = wsData?.active_session_model_identity || null;
        const sessionStatus = wsData?.active_session_status || null;
        const isRunning = sessionStatus === 'running' || sessionStatus === 'disconnected';

        if (backend && endpointKind === 'Local' && isRunning && (modelIdentity || sessionStatus)) {
          const engineName = backend === 'rapid_mlx' ? 'Rapid-MLX' : 'llama.cpp';
          const modelName = modelIdentity
            ? modelIdentity.length > 24
              ? modelIdentity.slice(0, 22) + '…'
              : modelIdentity
            : 'Active';
          labelEl.textContent = `${engineName} · ${modelName}`;

          if (generationActive) {
            dotEl.className = 'engine-indicator-dot live';
            indicator.setAttribute('title', `${engineName} · ${modelIdentity || modelName} (generating)`);
          } else {
            dotEl.className = 'engine-indicator-dot idle-active';
            indicator.setAttribute('title', `${engineName} · ${modelIdentity || modelName} (idle)`);
          }

          indicator.style.display = 'inline-flex';
        } else {
          indicator.style.display = 'none';
        }
      }
    }

    const wsMode = wsData?.mode ?? (wsData?.sleep_mode ? 'sleep' : 'off');
    const isSleeping = wsMode === 'sleep';
    const isManualSleep = isSleeping && wsData?.sleep_mode_manual === true;
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
        throughputEl.textContent = 'P ' + (promptDisplayRate > 0 ? promptDisplayRate.toFixed(0) : '—') + ' · G ' + (genDisplayRate > 0 ? genDisplayRate.toFixed(0) : '—');
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

    const gpuEntries = Object.values(wsData?.gpu || {});
    const hottestGpu = gpuEntries.length > 0
        ? Math.max(...gpuEntries.map(m => Number(m?.temp) || 0))
        : 0;
    if (gpuEl) {
        gpuEl.textContent = 'GPU ' + (hottestGpu > 0 ? hottestGpu.toFixed(0) + 'C' : '—');
    }

    if (sparkEl) {
        const promptPoints = hasActiveEndpoint ? (metricSeries.prompt || []) : [];
        const genPoints = hasActiveEndpoint ? (metricSeries.generation || []) : [];
        const livePoints = hasActiveEndpoint ? (metricSeries.liveOutput || []) : [];
        const maxLen = Math.max(promptPoints.length, genPoints.length);
        const points = [];
        for (let i = 0; i < maxLen; i += 1) {
            const p = promptPoints[promptPoints.length - maxLen + i] || 0;
            const g = genPoints[genPoints.length - maxLen + i] || 0;
            points.push(Math.max(p, g));
        }
        const fallbackPoints = livePoints.filter(value => value > 0);
        let displayPoints = points.some(value => value > 0) ? points : fallbackPoints;
        if (displayPoints.length < 2) {
            displayPoints = buildIdleCockpitPoints();
        }
        // eslint-disable-next-line no-unsanitized/property -- SVG markup is generated internally from numeric series values only
        sparkEl.innerHTML = displayPoints.length >= 2 ? buildCockpitSparkline(displayPoints) : '';
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
