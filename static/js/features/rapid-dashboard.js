// Optional Rapid telemetry stays optional: no inferred request outcomes or caps.

const REQUEST_ROW_LIMIT = 32;
export const metricNumber = value => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null;
const ratio = value => {
    const number = metricNumber(value);
    return number != null && number <= 1 ? number : null;
};
const count = value => metricNumber(value) == null ? 'Unavailable' : value.toLocaleString(undefined, { maximumFractionDigits: 0 });
const decimal = value => metricNumber(value) == null ? 'Unavailable' : value.toLocaleString(undefined, { maximumFractionDigits: 2 });

// Runtime allocations are reported in bytes and shown in binary units (GiB/MiB).
// Hardware pools elsewhere in the dashboard keep the app-wide "GB" convention.
const MIB = 1024 ** 2;
const GIB = 1024 ** 3;
const fixed = (value, digits) => value.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
// Zero stays "0.0 GiB" (a reported zero is data); anything non-zero under 1 GiB
// is shown in MiB so tiny reported values never collapse to "0.0 GiB".
export function formatRuntimeBytes(value) {
    const number = metricNumber(value);
    if (number == null) return null;
    if (number === 0) return { value: '0.0', unit: 'GiB' };
    if (number < GIB) {
        const mib = number / MIB;
        return { value: mib < 0.1 ? '<0.1' : fixed(mib, mib < 10 ? 1 : 0), unit: 'MiB' };
    }
    const gib = number / GIB;
    return { value: fixed(gib, gib >= 100 ? 0 : 1), unit: 'GiB' };
}
const bytes = value => {
    const parts = formatRuntimeBytes(value);
    return parts ? `${parts.value} ${parts.unit}` : 'Unavailable';
};
// One bound for free text, including request IDs (display, row key, and the
// accessible name must all agree on the same truncated ID).
const TEXT_LIMIT = 128;
const boundedText = value => typeof value === 'string' ? value.slice(0, TEXT_LIMIT) : '';
const requestIdOf = request => boundedText([request?.id, request?.request_id].find(id => typeof id === 'string' && id));

const totalFor = (requests, field) => {
    if (!requests.length || requests.some(request => metricNumber(request[field]) == null)) return null;
    const total = requests.reduce((sum, request) => sum + request[field], 0);
    return metricNumber(total);
};

function optionalFacts(sample) {
    const cache = sample?.cache_metrics;
    const kinds = Array.isArray(cache?.multimodal_cache_kinds)
        ? cache.multimodal_cache_kinds.slice(0, 8).map(boundedText).filter(value => value.trim()) : [];
    const cacheFacts = {
        hitRate: ratio(cache?.hit_rate) ?? ratio(sample?.global_cache_hit_rate),
        hits: metricNumber(cache?.hits), misses: metricNumber(cache?.misses),
        entries: metricNumber(cache?.entry_count) ?? metricNumber(sample?.global_cache_entries),
        memoryBytes: metricNumber(cache?.current_memory_bytes), kinds,
    };
    cacheFacts.available = kinds.length > 0 || Object.values(cacheFacts).some(value => typeof value === 'number');
    const telemetry = sample?.backend_details?.telemetry;
    return {
        cache: cacheFacts,
        acceptance: ratio(telemetry?.speculative_acceptance_rate) ?? ratio(sample?.speculative_acceptance_rate),
        outcomes: {
            succeeded: metricNumber(telemetry?.succeeded_requests_total),
            failed: metricNumber(telemetry?.failed_requests_total),
            cancelled: metricNumber(telemetry?.cancelled_requests_total),
            completed: metricNumber(sample?.completed_requests_total),
        },
        memory: {
            active: metricNumber(sample?.active_memory_bytes), peak: metricNumber(sample?.peak_memory_bytes),
            cache: metricNumber(sample?.cache_memory_bytes), limit: metricNumber(sample?.backend_details?.memory_limit_bytes),
        },
    };
}

function requestPhase(request) {
    const phase = boundedText(request?.phase || request?.status).trim().toLowerCase().replace(/[ -]/g, '_');
    if (['reading', 'prefill', 'prefilling', 'prompt', 'prompt_processing', 'processing_prompt'].includes(phase)) return 'reading';
    if (['generating', 'generation', 'decode', 'decoding'].includes(phase)) return 'generating';
    if (['waiting', 'queued', 'pending'].includes(phase)) return 'queued';
    return 'unknown';
}

function progressRatio(value) {
    const direct = ratio(value);
    if (direct != null) return direct;
    const current = metricNumber(value?.current);
    const total = metricNumber(value?.total);
    return current != null && total > 0 && current <= total ? current / total : null;
}

export function normalizeRapidDashboard(sample) {
    // Unavailable telemetry can carry a retained snapshot. None of its facts,
    // request phases or rates describe current activity.
    if (sample?.telemetry_unavailable) sample = null;
    const requests = Array.isArray(sample?.active_requests)
        ? sample.active_requests.slice(0, 64).filter(request => request && typeof request === 'object' && !Array.isArray(request)) : [];
    const phases = requests.map(requestPhase);
    const readingCount = phases.filter(phase => phase === 'reading').length;
    const generatingCount = phases.filter(phase => phase === 'generating').length;
    const running = metricNumber(sample?.running_requests);
    const waiting = metricNumber(sample?.waiting_requests);
    // Running is not proof of decode. Unknown/missing phase remains explicitly busy.
    const busy = running > 0 || phases.includes('unknown');
    const state = generatingCount ? 'generating'
        : readingCount ? 'reading' : busy ? 'busy'
            : waiting > 0 || phases.includes('queued') ? 'queued'
                : running === 0 && waiting === 0 ? 'idle' : 'unavailable';
    const decodeTps = metricNumber(sample?.generation_tokens_per_second);
    const prefillTps = metricNumber(sample?.prompt_tokens_per_second);
    const generating = requests.filter(request => requestPhase(request) === 'generating');
    const reading = requests.filter(request => requestPhase(request) === 'reading');
    const outputTokens = totalFor(generating, 'completion_tokens');
    const outputLimit = totalFor(generating, 'max_tokens');
    return {
        requests, readingCount, generatingCount, running, waiting, state,
        decodeTps, prefillTps,
        decodeCurrent: state !== 'unavailable' && generatingCount > 0 && decodeTps != null,
        prefillCurrent: state !== 'unavailable' && readingCount > 0 && prefillTps != null,
        decodeScope: decodeTps == null ? 'unavailable' : generatingCount || state === 'busy' || state === 'unavailable' ? 'aggregate reported' : 'aggregate last measured',
        prefillScope: prefillTps == null ? 'unavailable' : readingCount || state === 'busy' || state === 'unavailable' ? 'aggregate reported' : 'aggregate last measured',
        progress: progressRatio(sample?.backend_details?.progress),
        outputTokens, outputLimit,
        outputBudget: outputTokens != null && outputLimit > 0
            && generating.every(request => request.max_tokens > 0) ? Math.min(1, outputTokens / outputLimit) : null,
        promptTokens: totalFor(reading, 'prompt_tokens'),
        elapsed: reading.length === 1 ? metricNumber(reading[0].elapsed_s) : null,
        ...optionalFacts(sample),
    };
}

function text(id, value) {
    const el = document.getElementById(id);
    if (el && el.textContent !== value) el.textContent = value;
}


export function renderRapidDashboard(sample, { attached, backend, hostSystem, gpu, sessionId, endpointTag } = {}) {
    const panel = document.getElementById('rapid-dashboard-details');
    const visible = attached && backend === 'rapid_mlx';
    if (panel) panel.hidden = !visible;
    if (!visible || sample?.telemetry_unavailable) sample = null;
    const normalized = normalizeRapidDashboard(sample);
    renderRequestTable(normalized.requests, sample?.active_requests,
        // Same owner identity the metric-card history uses (dashboard-ws view).
        { backend, attached, sessionId: sessionId ?? null, endpointTag: endpointTag ?? null });
    renderMemoryHealth(visible ? sample : null, hostSystem, gpu);
}

// Rapid-MLX reports only elapsed_s, not a start time. Estimate it as
// "when we received the snapshot minus elapsed_s", and keep the first estimate
// per request so the displayed time does not jitter with each poll.
//
// Request IDs are only unique within one inference target, so estimates are
// keyed `${sessionId or endpoint tag}|${requestId}` and dropped wholesale when
// the owning target (backend/session/endpoint/attachment) changes.
// Snapshot receipt time and elapsed_s are both rounded/jittered by polling, so
// an estimate is only replaced when a new one disagrees by more than this.
const START_ESTIMATE_TOLERANCE_MS = 5000;
const startEstimates = new Map();
let startOwner = null;
let startOwnerId = '';
const startKey = request => `${startOwnerId}|${requestIdOf(request)}`;

function selectStartOwner({ backend, attached, sessionId, endpointTag }) {
    const owner = JSON.stringify([backend ?? null, sessionId ?? null, endpointTag ?? null, attached ?? null]);
    if (owner === startOwner) return;
    startOwner = owner;
    startOwnerId = String(sessionId ?? endpointTag ?? '');
    startEstimates.clear();
}

function estimateStarts(rows, owner, now = Date.now()) {
    selectStartOwner(owner);
    const seen = new Set();
    rows.forEach(request => {
        if (!requestIdOf(request)) return;
        const key = startKey(request);
        // A row that merely lacks elapsed_s is still present: keep its estimate.
        seen.add(key);
        const elapsed = metricNumber(request.elapsed_s);
        if (elapsed == null) return;
        const estimate = now - elapsed * 1000;
        const known = startEstimates.get(key);
        if (known == null || Math.abs(known - estimate) > START_ESTIMATE_TOLERANCE_MS) startEstimates.set(key, estimate);
    });
    for (const key of startEstimates.keys()) if (!seen.has(key)) startEstimates.delete(key);
}
const startOf = request => requestIdOf(request) ? startEstimates.get(startKey(request)) ?? null : null;
const clockTime = ms => ms == null ? 'Unavailable'
    : new Date(ms).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' });

const REQUEST_COLUMNS = [
    { key: 'id', label: 'Request', always: true, value: requestIdOf },
    // Only a request that reports neither phase nor status is "Unknown".
    { key: 'phase', label: 'Phase / status', always: true, empty: 'Unknown', value: request => boundedText(request.phase || request.status) },
    { key: 'started', label: 'Started (est.)', value: startOf, format: clockTime,
        title: ms => `Estimated: ${new Date(ms).toLocaleString()} (snapshot time minus reported elapsed)` },
    { key: 'prompt', label: 'Prompt tokens', value: request => metricNumber(request.prompt_tokens), format: count },
    { key: 'output', label: 'Output / limit', value: request => metricNumber(request.completion_tokens) ?? metricNumber(request.max_tokens),
        format: (_, request) => `${count(request.completion_tokens)} / ${count(request.max_tokens)}` },
    { key: 'cached', label: 'Cached tokens', value: request => metricNumber(request.cached_tokens), format: count },
    { key: 'rate', label: 't/s', value: request => metricNumber(request.tokens_per_second), format: decimal },
    { key: 'ttft', label: 'TTFT (s)', value: request => metricNumber(request.ttft_s), format: decimal },
    { key: 'elapsed', label: 'Elapsed (s)', value: request => metricNumber(request.elapsed_s), format: decimal },
    { key: 'cache', label: 'Cache hit', value: request => boundedText(request.cache_hit_type) },
];

const EMPTY_ROW_KEY = '\u0000empty';

// Rows are keyed by request ID (repeats and ID-less rows get an ordinal) so a
// poll that only changes volatile cells patches text in place. Rebuilding the
// DOM every tick destroyed text selection/copy of a request ID.
function rowKeys(rows) {
    const seen = new Map();
    return rows.map((request, index) => {
        const id = requestIdOf(request);
        const base = id || `\u0000row-${index}`;
        const n = seen.get(base) ?? 0;
        seen.set(base, n + 1);
        return n ? `${base}\u0000${n}` : base;
    });
}

function buildIdCell(td, value) {
    const id = document.createElement('span');
    id.dataset.requestId = value;
    id.title = value;
    const short = document.createElement('span');
    short.className = 'rapid-request-id-short';
    short.setAttribute('aria-hidden', 'true');
    short.textContent = value.length > 20 ? `${value.slice(0, 10)}…${value.slice(-6)}` : value;
    // Real text for assistive tech: the title attribute alone is mouse-only.
    const full = document.createElement('span');
    full.className = 'rapid-request-id-full';
    full.textContent = `Request ID ${value}`;
    id.append(short, full);
    td.replaceChildren(id);
}

function patchCell(td, column, request) {
    const value = column.value(request);
    if (column.key === 'id') {
        if (td.className !== 'rapid-request-id-cell') td.className = 'rapid-request-id-cell';
        if (value) {
            if (td.firstElementChild?.dataset.requestId !== value) buildIdCell(td, value);
        } else if (td.textContent !== 'Unavailable' || td.firstElementChild) {
            td.textContent = 'Unavailable';
        }
        return;
    }
    const next = column.format ? column.format(value, request) : value || column.empty || 'Unavailable';
    if (td.textContent !== next) td.textContent = next;
    const title = column.title && value != null ? column.title(value) : null;
    if (title == null) {
        if (td.hasAttribute('title')) td.removeAttribute('title');
    } else if (td.title !== title) {
        td.title = title;
    }
}

function renderRequestTable(requests, rawRequests, owner) {
    const body = document.getElementById('rapid-request-rows');
    const head = document.getElementById('rapid-request-columns');
    if (!body || !head) return;
    const rows = requests.slice(0, REQUEST_ROW_LIMIT);
    estimateStarts(rows, owner);
    const columns = REQUEST_COLUMNS.filter(column => column.always
        || rows.some(request => {
            const value = column.value(request);
            return value !== null && value !== '';
        }));
    const columnSignature = columns.map(column => column.key).join();
    if (body.dataset.columnSignature !== columnSignature) {
        // A changed column set is a structural change: rebuild head and body.
        body.dataset.columnSignature = columnSignature;
        head.replaceChildren(...columns.map(column => {
            // DOM construction preserves table context; sanitizing bare <tr> strings
            // in a document/body context discards row/cell tags before tbody insertion.
            const th = document.createElement('th');
            th.scope = 'col';
            th.textContent = column.label;
            return th;
        }));
        body.replaceChildren();
        body.closest('table').dataset.columns = String(columns.length);
    }
    const existing = new Map([...body.children].map(tr => [tr.dataset.rowKey, tr]));
    const keys = rows.length ? rowKeys(rows) : [EMPTY_ROW_KEY];
    const wanted = keys.map((key, index) => {
        let tr = existing.get(key);
        if (!tr) {
            tr = document.createElement('tr');
            tr.dataset.rowKey = key;
        }
        if (!rows.length) {
            let td = tr.firstElementChild;
            if (!td) {
                td = document.createElement('td');
                td.textContent = 'No active request details reported';
                tr.append(td);
            }
            if (td.colSpan !== columns.length) td.colSpan = columns.length;
            return tr;
        }
        while (tr.children.length < columns.length) tr.append(document.createElement('td'));
        columns.forEach((column, i) => patchCell(tr.children[i], column, rows[index]));
        return tr;
    });
    const keep = new Set(wanted);
    [...body.children].forEach(tr => { if (!keep.has(tr)) tr.remove(); });
    wanted.forEach((tr, index) => {
        if (body.children[index] !== tr) body.insertBefore(tr, body.children[index] ?? null);
    });
    const total = Array.isArray(rawRequests) ? rawRequests.length : 0;
    text('rapid-request-summary', !Array.isArray(rawRequests) ? 'Active request details unavailable'
        : total > rows.length ? `Showing ${rows.length} of ${total} reported requests (bounded view)`
            : `${rows.length} reported requests · unreported columns hidden; missing fields are unavailable`);
}

function memoryHealthMessages(sample, system, gpu) {
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
        // role=status announces every text mutation: write only on change, and
        // empty the region when it hides so stale copy is never re-announced.
        const next = messages.join(' ');
        const tone = danger ? 'danger' : 'warning';
        if (warning.hidden !== (messages.length === 0)) warning.hidden = messages.length === 0;
        if (warning.dataset.tone !== tone) warning.dataset.tone = tone;
        if (warning.textContent !== next) warning.textContent = next;
    }
    const pressure = boundedText(system?.memory_pressure_level);
    const swap = metricNumber(system?.swap_used_gb);
    text('dashboard-memory-summary', system
        ? `Host pressure: ${pressure || 'unavailable'} · Swap in use: ${swap == null ? 'unavailable' : `${swap.toFixed(2)} GiB`} (swap occupancy alone is not pressure)`
        : 'Host memory pressure and swap telemetry unavailable');
}
