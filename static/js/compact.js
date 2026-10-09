import { applyProductIdentity, LEGACY_STORAGE_KEYS } from './core/identity.js';

applyProductIdentity();

// ── Theme + palette ──────────────────────────────────────────────────────────
// The popover is served from the same origin as the main app, so it shares
// localStorage. Mirror the main UI's theme/palette onto <html> so the tray uses
// the exact same tokens.css variables (dark/light + accent palette).
function normalizePaletteId(palette) {
    return palette === 'carbon-mint' || palette === 'carbon_mint' ? '' : (palette || '');
}

function readSavedPreferences() {
    try {
        return JSON.parse(localStorage.getItem(LEGACY_STORAGE_KEYS.preferences) || '{}') || {};
    } catch (_) {
        return {};
    }
}

function applyAppearancePreferences() {
    const prefs = readSavedPreferences();
    const theme = prefs.theme || 'dark';
    const effectiveTheme = theme === 'auto'
        ? (window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark')
        : theme;
    document.documentElement.dataset.theme = effectiveTheme;

    const palette = normalizePaletteId(prefs.palette);
    if (palette) {
        document.documentElement.dataset.palette = palette;
    } else {
        delete document.documentElement.dataset.palette;
    }
}

applyAppearancePreferences();
// Re-apply if the user changes theme/palette in the main window while the
// popover is open, or if the OS appearance changes under "auto".
window.addEventListener('storage', (e) => {
    if (e.key === LEGACY_STORAGE_KEYS.preferences) applyAppearancePreferences();
});
window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', applyAppearancePreferences);

document.getElementById('compact-close').addEventListener('click', () => {
    if (window.ipc?.postMessage) {
        window.ipc.postMessage(JSON.stringify({ action: 'close' }));
        return;
    }
    window.close();
});

// Borderless Windows popovers do not have a native title bar. Let the
// branded header drag the window while keeping the close control clickable.
const compactHeader = document.querySelector('.header');
compactHeader?.addEventListener('pointerdown', (event) => {
    if (event.target.closest('#compact-close')) return;
    event.preventDefault();
    if (window.ipc?.postMessage) {
        // Native drag owns the pointer sequence on Windows. Do not also send
        // synthetic move deltas: that races the OS drag loop and can leave a
        // second drag effectively attached to later mouse movement.
        window.ipc.postMessage(JSON.stringify({ action: 'drag' }));
    }
});

const PORT = window.__COMPACT_PORT__;
let ws = null;
let reconnectTimer = null;

function connectWebSocket() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        return;
    }
    try {
        ws = new WebSocket('ws://127.0.0.1:' + PORT + '/ws');
        ws.onopen = function() {
            console.log('WebSocket connected');
            document.getElementById('status-bar').innerHTML = '<span class="running">● Connected</span>';
        };
        ws.onerror = function(e) {
            console.error('WebSocket error:', e);
        };
        ws.onclose = function() {
            console.log('WebSocket closed, reconnecting in 2s...');
            document.getElementById('status-bar').innerHTML = '<span class="stopped">● Disconnected</span>';
            if (reconnectTimer) clearTimeout(reconnectTimer);
            reconnectTimer = setTimeout(connectWebSocket, 2000);
        };
    } catch(e) {
        console.error('Failed to create WebSocket:', e);
        if (reconnectTimer) clearTimeout(reconnectTimer);
        reconnectTimer = setTimeout(connectWebSocket, 2000);
    }
}

connectWebSocket();

function setBar(id, pct, maxVal) {
    const bar = document.getElementById(id);
    if (!bar) return;
    const p = Math.min(100, Math.max(0, (pct / maxVal) * 100));
    // Width-based fill (matches the GPU bars and animates via the CSS width
    // transition). Avoids scaleX, which scaled from the element centre because
    // .metric-bar has no transform-origin, making bars render incorrectly.
    bar.style.width = p + '%';
}

function updateStatus(running) {
    const el = document.getElementById('status-bar');
    if (running) {
        el.innerHTML = '<span class="running">● Running</span>';
    } else {
        el.innerHTML = '<span class="stopped">● Stopped</span>';
    }
}

function renderGPUs(gpuEntries) {
    const container = document.getElementById('gpu-sections');
    if (!container) return;
    container.style.display = '';
    container.replaceChildren();
    for (const [name, m] of gpuEntries) {
        const tempPct = m.temp / 120 * 100;
        // False marks an unavailable measurement; older collectors omit the flag.
        const loadKnown = m.load_available !== false && Number.isFinite(m.load);
        const loadPct = loadKnown ? m.load : 0;
        const vramPct = m.vram_total > 0 ? (m.vram_used / m.vram_total * 100) : 0;
        const vramGB = (m.vram_used / 1024).toFixed(1);
        const vramTotalGB = (m.vram_total / 1024).toFixed(1);

        const loadSeverity = loadKnown ? getSeverity(m.load, 100) : '';
        const tempSeverity = getSeverity(m.temp, 120);
        const vramSeverity = getSeverity(m.vram_used, m.vram_total);

        const div = document.createElement('div');
        div.className = 'gpu-section';
        const gpuName = document.createElement('div');
        gpuName.className = 'gpu-name';
        gpuName.textContent = name;
        div.appendChild(gpuName);
        const rows = [
            ['Load', 'gpu-load', loadPct, loadSeverity, loadKnown ? m.load + '%' : '—'],
            ['Temp', 'gpu-temp', tempPct, tempSeverity, Math.round(m.temp) + '°C'],
            ['VRAM', 'gpu-vram', vramPct, vramSeverity, vramGB + ' / ' + vramTotalGB + ' GB'],
        ];
        for (const [label, type, pct, severity, value] of rows) {
            const row = document.createElement('div');
            row.className = 'metric-row';
            const labelEl = document.createElement('span');
            labelEl.className = 'metric-label';
            labelEl.textContent = label;
            const wrap = document.createElement('div');
            wrap.className = 'metric-bar-wrap';
            const bar = document.createElement('div');
            bar.className = 'metric-bar ' + type + (severity ? ' severity-' + severity : '');
            bar.style.width = pct + '%';
            wrap.appendChild(bar);
            const valueEl = document.createElement('span');
            valueEl.className = 'metric-value';
            valueEl.textContent = value;
            row.append(labelEl, wrap, valueEl);
            div.appendChild(row);
        }
        container.appendChild(div);
    }
}

// Expose function to get content dimensions for native resize.
// Span from the first visible child's top to the last visible child's bottom so
// inter-section margins are counted (summing rect heights would miss them and
// clip the popover), then add the body's vertical padding.
window.getContentDimensions = function() {
    const visible = Array.from(document.body.children).filter((child) => {
        if (child.tagName === 'SCRIPT') return false;
        return window.getComputedStyle(child).display !== 'none';
    });
    const bodyStyle = window.getComputedStyle(document.body);
    const pad = parseFloat(bodyStyle.paddingTop) + parseFloat(bodyStyle.paddingBottom);
    let height = pad;
    if (visible.length > 0) {
        const top = visible[0].getBoundingClientRect().top;
        const bottom = visible[visible.length - 1].getBoundingClientRect().bottom;
        height = (bottom - top) + pad;
    }
    const width = Math.ceil(Math.max(document.body.getBoundingClientRect().width, document.body.scrollWidth));
    height = Math.ceil(height);
    return { width, height };
};

function reportPopoverSize() {
    if (!window.ipc || !window.ipc.postMessage) return;
    requestAnimationFrame(function() {
        window.ipc.postMessage(JSON.stringify({
            action: 'resize',
            ...window.getContentDimensions(),
        }));
    });
}

// Signal ready when content is loaded
window.addEventListener('load', function() {
    window.__popoverReady = true;
    reportPopoverSize();
});

function clearMetrics() {
    document.getElementById('cpu-section').style.display = '';
    document.getElementById('ram-section').style.display = '';
    document.getElementById('capability-note').style.display = 'none';
    setBar('cpu-load-bar', 0, 100);
    document.getElementById('cpu-load').textContent = '—';
    setBar('cpu-temp-bar', 0, 120);
    document.getElementById('cpu-temp').textContent = '—';
    setBar('ram-bar', 0, 100);
    document.getElementById('ram').textContent = '—';
    document.getElementById('gpu-sections').innerHTML = '';
    document.getElementById('inf-prompt').innerHTML = '<span class="dash-line">—</span> tok/s';
    document.getElementById('inf-generate').innerHTML = '<span class="dash-line">—</span> tok/s';
    document.getElementById('inf-context').innerHTML = '<span class="dash-line">—</span> %';
    reportPopoverSize();
}

function clearLocalMetrics(reasons) {
    document.getElementById('cpu-section').style.display = 'none';
    document.getElementById('ram-section').style.display = 'none';
    document.getElementById('gpu-sections').innerHTML = '';
    document.getElementById('gpu-sections').style.display = 'none';
    
    const reasonEl = document.getElementById('capability-note');
    if (reasons && (reasons.system || reasons.gpu || reasons.cpu_temp)) {
        reasonEl.style.color = 'var(--color-error)';
        reasonEl.style.textAlign = 'center';
        reasonEl.style.fontStyle = 'italic';
        const reasonList = [];
        if (reasons.system) reasonList.push(getEmptyStateMessage(reasons.system));
        if (reasons.gpu) reasonList.push(getEmptyStateMessage(reasons.gpu));
        if (reasons.cpu_temp) reasonList.push(getEmptyStateMessage(reasons.cpu_temp));
        reasonEl.textContent = Array.from(new Set(reasonList)).join(', ');
        reasonEl.style.display = reasonEl.textContent ? '' : 'none';
    } else {
        reasonEl.style.display = 'none';
    }
    
    reportPopoverSize();
}

function getSeverity(value, max) {
    if (!value || value <= 0) return '';
    const pct = (value / max) * 100;
    if (pct >= 90) return 'critical';
    if (pct >= 70) return 'warning';
    return 'normal';
}

function getEmptyStateMessage(reason, fallback) {
    if (reason === 'RemoteEndpoint') {
        return 'Unavailable (remote)';
    }
    if (reason === 'NoDisplay') {
        return 'Unavailable (no display)';
    }
    if (reason === 'TrayUnavailable') {
        return 'Unavailable (tray)';
    }
    if (reason === 'SensorUnavailable') {
        return 'Temp sensor unavailable';
    }
    if (reason === 'BackendUnavailable') {
        return 'GPU unavailable';
    }
    if (reason === 'CommandMissing') {
        return 'Command missing';
    }
    if (reason === 'PermissionDenied') {
        return 'Permission denied';
    }
    if (reason === 'MetricsUnreachable') {
        return 'Metrics unreachable';
    }
    if (reason === 'NotApplicable') {
        return 'Not applicable';
    }
    if (reason === 'Unavailable') {
        return 'Unavailable';
    }
    return fallback || '—';
}

ws.onmessage = function(e) {
    let d;
    try {
        d = JSON.parse(e.data);
    } catch(_err) { return; }

    updateStatus(d.server_running);

    if (!d.server_running) {
        clearMetrics();
        return;
    }

    const gpuAvailable = d.capabilities?.gpu ?? true;
    const systemAvailable = d.capabilities?.system ?? true;
    
    if (!systemAvailable && !gpuAvailable) {
        const reasons = d.availability || {};
        clearLocalMetrics(reasons);
    } else {
        document.getElementById('capability-note').style.display = 'none';
        if (systemAvailable) {
            document.getElementById('cpu-section').style.display = '';
            document.getElementById('ram-section').style.display = '';
        } else {
            document.getElementById('cpu-section').style.display = 'none';
            document.getElementById('ram-section').style.display = 'none';
        }

    const sys = d.system || {};
    // cpu_load_available=false: first sampling interval, 0 is a placeholder (absent = real value).
    const cpuLoadKnown = sys.cpu_load_available !== false && Number.isFinite(sys.cpu_load);
    const cpuLoad = cpuLoadKnown ? sys.cpu_load : 0;
    const cpuLoadSeverity = getSeverity(cpuLoad, 100);
    const cpuLoadBar = document.getElementById('cpu-load-bar');
    cpuLoadBar.className = 'metric-bar cpu' + (cpuLoadSeverity ? ' severity-' + cpuLoadSeverity : '');
    setBar('cpu-load-bar', cpuLoad, 100);
    document.getElementById('cpu-load').textContent = cpuLoadKnown ? cpuLoad + '%' : '—';

    if (sys.cpu_temp_available && sys.cpu_temp > 0) {
        const severity = getSeverity(sys.cpu_temp, 120);
        const tempBar = document.getElementById('cpu-temp-bar');
        tempBar.className = 'metric-bar cpu' + (severity ? ' severity-' + severity : '');
        setBar('cpu-temp-bar', sys.cpu_temp, 120);
        document.getElementById('cpu-temp').textContent = Math.round(sys.cpu_temp) + '°C';
    } else {
        setBar('cpu-temp-bar', 0, 120);
        document.getElementById('cpu-temp').textContent = '—';
        const tempBar = document.getElementById('cpu-temp-bar');
        tempBar.className = 'metric-bar cpu';
    }

    if (sys.ram_total_gb > 0) {
        const ramPct = (sys.ram_used_gb / sys.ram_total_gb) * 100;
        const ramSeverity = getSeverity(ramPct, 100);
        const ramBar = document.getElementById('ram-bar');
        ramBar.className = 'metric-bar ram' + (ramSeverity ? ' severity-' + ramSeverity : '');
        setBar('ram-bar', ramPct, 100);
        document.getElementById('ram').textContent = sys.ram_used_gb.toFixed(1) + ' / ' + sys.ram_total_gb.toFixed(1) + ' GB';
    } else {
        setBar('ram-bar', 0, 100);
        document.getElementById('ram').textContent = '—';
        const ramBar = document.getElementById('ram-bar');
        ramBar.className = 'metric-bar ram';
    }

    const gpuEntries = Object.entries(d.gpu || {});
    
    if (gpuAvailable) {
        renderGPUs(gpuEntries);
    } else {
        document.getElementById('gpu-sections').innerHTML = '';
        document.getElementById('gpu-sections').style.display = 'none';
    }
    }

    const l = d.inference_poll_failed ? {} : (d.llama || {});
    const promptRate = l.prompt_tokens_per_sec > 0 ? l.prompt_tokens_per_sec : (l.last_prompt_tokens_per_sec || 0);
    const genRate = l.generation_tokens_per_sec > 0 ? l.generation_tokens_per_sec : (l.last_generation_tokens_per_sec || 0);
    const contextCapacity = l.context_capacity_tokens || l.kv_cache_max || 0;
    const contextLive = l.context_live_tokens || l.kv_cache_tokens || 0;
    const contextPeak = l.context_high_water_tokens || l.kv_cache_high_water || 0;
    const contextLiveAvailable = l.context_live_tokens_available || l.kv_cache_tokens_available;

    if (promptRate > 0) {
        document.getElementById('inf-prompt').textContent = promptRate.toFixed(1) + ' tok/s';
    } else {
        document.getElementById('inf-prompt').innerHTML = '<span class="dash-line">—</span> tok/s';
    }

    if (genRate > 0) {
        document.getElementById('inf-generate').textContent = genRate.toFixed(1) + ' tok/s';
    } else {
        document.getElementById('inf-generate').innerHTML = '<span class="dash-line">—</span> tok/s';
    }

    if (contextCapacity > 0 && contextLiveAvailable) {
        const ctxPct = ((contextLive / contextCapacity) * 100).toFixed(0);
        document.getElementById('inf-context').textContent = contextLive + ' / ' + contextCapacity + ' (' + ctxPct + '%)';
    } else if (contextCapacity > 0) {
        document.getElementById('inf-context').textContent = 'cap ' + contextCapacity + (contextPeak > 0 ? ' · peak ' + contextPeak : '');
    } else {
        document.getElementById('inf-context').innerHTML = '<span class="dash-line">—</span> %';
    }

    reportPopoverSize();
};
