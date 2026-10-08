// Optional Rapid telemetry stays optional: no inferred request outcomes or caps.
import { escapeHtml } from '../core/format.js';
import { setHtml } from '../core/set-html.js';

export const REQUEST_ROW_LIMIT = 32;
export const metricNumber = value => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null;
const ratio = value => {
    const number = metricNumber(value);
    return number != null && number <= 1 ? number : null;
};
const count = value => metricNumber(value) == null ? 'Unavailable' : value.toLocaleString(undefined, { maximumFractionDigits: 0 });
const bytes = value => metricNumber(value) == null ? 'Unavailable' : `${(value / 1024 ** 3).toLocaleString(undefined, { maximumFractionDigits: 2 })} GiB`;
const pct = value => ratio(value) == null ? 'Unavailable' : `${(value * 100).toFixed(1)}%`;
const decimal = value => metricNumber(value) == null ? 'Unavailable' : value.toLocaleString(undefined, { maximumFractionDigits: 2 });
const boundedText = value => typeof value === 'string' ? value.slice(0, 128) : '';

export function requestPhase(request) {
    const phase = boundedText(request?.phase || request?.status).trim().toLowerCase().replace(/[ -]/g, '_');
    if (['reading', 'prefill', 'prefilling', 'prompt', 'prompt_processing', 'processing_prompt'].includes(phase)) return 'reading';
    if (['generating', 'generation', 'decode', 'decoding'].includes(phase)) return 'generating';
    if (['waiting', 'queued', 'pending'].includes(phase)) return 'queued';
    return 'unknown';
}

export function progressRatio(value) {
    const direct = ratio(value);
    if (direct != null) return direct;
    const current = metricNumber(value?.current);
    const total = metricNumber(value?.total);
    return current != null && total > 0 && current <= total ? current / total : null;
}

export function normalizeRapidDashboard(sample) {
    const requests = Array.isArray(sample?.active_requests)
        ? sample.active_requests.slice(0, 64).filter(request => request && typeof request === 'object' && !Array.isArray(request)) : [];
    const phases = requests.map(requestPhase);
    const readingCount = phases.filter(phase => phase === 'reading').length;
    const generatingCount = phases.filter(phase => phase === 'generating').length;
    const running = metricNumber(sample?.running_requests);
    const waiting = metricNumber(sample?.waiting_requests);
    // Running is not proof of decode. Unknown/missing phase remains explicitly busy.
    const busy = running > 0 || phases.includes('unknown');
    const state = sample?.telemetry_unavailable ? 'unavailable'
        : generatingCount ? 'generating' : readingCount ? 'reading' : busy ? 'busy'
            : waiting > 0 || phases.includes('queued') ? 'queued'
                : running === 0 && waiting === 0 ? 'idle' : 'unavailable';
    const decodeTps = metricNumber(sample?.generation_tokens_per_second);
    const prefillTps = metricNumber(sample?.prompt_tokens_per_second);
    return {
        requests, readingCount, generatingCount, running, waiting, state,
        decodeTps, prefillTps,
        decodeCurrent: state !== 'unavailable' && generatingCount > 0 && decodeTps != null,
        prefillCurrent: state !== 'unavailable' && readingCount > 0 && prefillTps != null,
        decodeScope: decodeTps == null ? 'unavailable' : generatingCount || state === 'busy' || state === 'unavailable' ? 'aggregate reported' : 'aggregate last measured',
        prefillScope: prefillTps == null ? 'unavailable' : readingCount || state === 'busy' || state === 'unavailable' ? 'aggregate reported' : 'aggregate last measured',
        progress: progressRatio(sample?.backend_details?.progress),
    };
}

function text(id, value) {
    const el = document.getElementById(id);
    if (el && el.textContent !== value) el.textContent = value;
}

function facts(id, entries) {
    const el = document.getElementById(id);
    if (!el) return;
    const signature = JSON.stringify(entries);
    if (el.dataset.signature === signature) return;
    el.dataset.signature = signature;
    setHtml(el, entries.map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`).join(''));
}

export function renderRapidDashboard(sample, { attached, backend, hostSystem, gpu } = {}) {
    const panel = document.getElementById('rapid-dashboard-details');
    const visible = attached && backend === 'rapid_mlx';
    if (panel) panel.hidden = !visible;
    if (!visible) sample = null;
    const normalized = normalizeRapidDashboard(sample);
    text('rapid-telemetry-status', !sample || sample.telemetry_unavailable ? 'Telemetry unavailable' : 'Reported snapshot · rates are aggregate, not per-request');
    const cache = sample?.cache_metrics;
    const hits = metricNumber(cache?.hits);
    const misses = metricNumber(cache?.misses);
    const lookups = hits != null && misses != null ? hits + misses : null;
    const hitRate = ratio(cache?.hit_rate) ?? ratio(sample?.global_cache_hit_rate);
    const cacheKinds = Array.isArray(cache?.multimodal_cache_kinds) ? cache.multimodal_cache_kinds.slice(0, 8).map(boundedText).filter(Boolean).join(', ') : '';
    facts('rapid-cache-facts', [
        ['Hit rate', lookups === 0 ? 'No lookups yet' : pct(hitRate)],
        ['Hits / misses', `${count(hits)} / ${count(misses)}`],
        ['Entries', count(metricNumber(cache?.entry_count) ?? metricNumber(sample?.global_cache_entries))],
        ['Prefix cache memory', bytes(cache?.current_memory_bytes)],
        ['Multimodal cache kinds', cacheKinds || 'Unavailable'],
    ]);
    const telemetry = sample?.backend_details?.telemetry;
    facts('rapid-outcome-facts', [
        ['MTP / speculative acceptance', pct(ratio(telemetry?.speculative_acceptance_rate) ?? ratio(sample?.speculative_acceptance_rate))],
        ['Succeeded requests', count(telemetry?.succeeded_requests_total)],
        ['Failed requests', count(telemetry?.failed_requests_total)],
        ['Cancelled requests', count(telemetry?.cancelled_requests_total)],
        ['Completed requests (backend-reported)', count(sample?.completed_requests_total)],
    ]);
    facts('rapid-memory-facts', [
        ['Runtime active allocations', bytes(sample?.active_memory_bytes)],
        ['Runtime peak (not a cap)', bytes(sample?.peak_memory_bytes)],
        ['Runtime allocator cache', bytes(sample?.cache_memory_bytes)],
        ['Runtime allocation cap', metricNumber(sample?.backend_details?.memory_limit_bytes) > 0 ? bytes(sample.backend_details.memory_limit_bytes) : 'Unknown · not inferred from hardware'],
        ['Hardware Metal wired cap', gpu?.metalUnified ? bytes(gpu.vramTotal) : 'Unavailable'],
    ]);
    renderRequestTable(normalized.requests, sample?.active_requests);
    renderMemoryHealth(visible ? sample : null, hostSystem, gpu);
}

function renderRequestTable(requests, rawRequests) {
    const body = document.getElementById('rapid-request-rows');
    if (!body) return;
    const rows = requests.slice(0, REQUEST_ROW_LIMIT);
    const cells = rows.map(request => [
        boundedText(request.id || request.request_id) || 'Unknown',
        boundedText(request.phase || request.status) || 'Unknown',
        count(request.prompt_tokens),
        `${count(request.completion_tokens)} / ${count(request.max_tokens)}`,
        count(request.cached_tokens),
        decimal(request.tokens_per_second),
        decimal(request.ttft_s),
        decimal(request.elapsed_s),
        boundedText(request.cache_hit_type) || 'Unavailable',
    ]);
    const signature = JSON.stringify(cells);
    if (body.dataset.signature !== signature) {
        body.dataset.signature = signature;
        setHtml(body, cells.length ? cells.map(row => `<tr>${row.map(value => `<td>${escapeHtml(value)}</td>`).join('')}</tr>`).join('')
            : '<tr><td colspan="9">No active request details reported</td></tr>');
    }
    const total = Array.isArray(rawRequests) ? rawRequests.length : 0;
    text('rapid-request-summary', total > rows.length ? `Showing ${rows.length} of ${total} reported requests (bounded view)` : `${rows.length} reported requests · missing fields are unavailable`);
}

export function memoryHealthMessages(sample, system, gpu) {
    const messages = [];
    let danger = false;
    const level = boundedText(system?.memory_pressure_level).toLowerCase();
    if (level === 'warning' || level === 'critical') {
        danger = level === 'critical';
        messages.push(`${level === 'critical' ? 'Critical' : 'Elevated'} host memory pressure${system.memory_pressure_source ? ` (${boundedText(system.memory_pressure_source)})` : ''}.`);
        if (system.memory_pressure_advice) messages.push(boundedText(system.memory_pressure_advice));
    }
    const swapIn = metricNumber(system?.swapins_delta);
    const swapOut = metricNumber(system?.swapouts_delta);
    if (swapIn > 0 || swapOut > 0) messages.push(`Active swapping: ${count(swapIn)} in / ${count(swapOut)} out since last host sample. This can stall inference.`);
    const active = metricNumber(sample?.active_memory_bytes);
    const cache = metricNumber(sample?.cache_memory_bytes);
    const cap = metricNumber(sample?.backend_details?.memory_limit_bytes);
    // Cached allocator blocks are reclaimable; don't add them to active allocations.
    if (active != null && cap > 0 && active / cap >= 0.9) messages.push('Runtime active allocations are near the explicitly reported allocation cap.');
    if (active != null && gpu?.metalUnified && gpu.vramTotal > 0 && active / gpu.vramTotal >= 0.9) messages.push('Runtime active allocations are near the hardware Metal wired cap; this is not a runtime allocation limit.');
    if (messages.length && active != null) messages.push(`Runtime active ${bytes(active)}${cache != null ? `; reclaimable allocator cache ${bytes(cache)}` : ''}.`);
    return { messages, danger };
}

function renderMemoryHealth(sample, system, gpu) {
    const warning = document.getElementById('dashboard-memory-warning');
    const { messages, danger } = memoryHealthMessages(sample, system, gpu);
    if (warning) {
        warning.hidden = messages.length === 0;
        warning.dataset.tone = danger ? 'danger' : 'warning';
        warning.textContent = messages.join(' ');
    }
    const pressure = boundedText(system?.memory_pressure_level);
    const swap = metricNumber(system?.swap_used_gb);
    text('dashboard-memory-summary', system
        ? `Host pressure: ${pressure || 'unavailable'} · Swap in use: ${swap == null ? 'unavailable' : `${swap.toFixed(2)} GiB`} (swap occupancy alone is not pressure)`
        : 'Host memory pressure and swap telemetry unavailable');
}
