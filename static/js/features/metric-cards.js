// ── Unified metric cards + model state strip ───────────────────────────────────
// Strata-style dashboard presentation: a state card (pills + progress) and a
// registry-driven grid of compact metric cards, each with a single inline-SVG
// sparkline (line + 12% area fill, tone via data-tone).
//
// The registry decouples presentation from the runtime loader: every card
// declares a `pick(view)` reader over the NORMALIZED snapshot built by
// dashboard-ws.js, so llama.cpp, Rapid-MLX, and remote agents feed the same
// cards. Generic cards render dimmed ("n/a") when unavailable; optional Rapid
// cards hide immediately and clear their contents when their source is absent.

import { setHtml } from '../core/set-html.js';
import { formatRuntimeBytes, metricNumber } from './rapid-dashboard.js';

// Every key accepts at most one sample per HISTORY_SAMPLE_MS (time-based, not
// per-update: dashboard pushes arrive up to ~4/s), so HISTORY_LIMIT samples
// cover at most ~5 minutes of observed updates. Samples are appended only when
// an update arrives: while the tab is hidden or pushes slow down, the series
// simply gains fewer points, so its wall-clock span can exceed 5 minutes.
const HISTORY_LIMIT = 300;
const HISTORY_SAMPLE_MS = 1000;
const history = new Map();
const historyLastPush = new Map();
let historyOwner = null;

function selectHistoryOwner(view) {
    // Counters belong to an inference target, not the lifetime of this module.
    // Detach and backend changes also replace the owner, clearing every plot
    // and every per-key throttle before the first sample from the new target.
    const owner = JSON.stringify([view.backend ?? null, view.sessionId ?? null, view.endpointTag ?? null, view.attached ?? null]);
    if (owner === historyOwner) return;
    historyOwner = owner;
    history.clear();
    historyLastPush.clear();
}

function pushHistory(key, value, now = Date.now()) {
    if (!Number.isFinite(value)) return;
    const last = historyLastPush.get(key);
    if (last != null && now - last < HISTORY_SAMPLE_MS && now >= last) return;
    historyLastPush.set(key, now);
    let series = history.get(key);
    if (!series) {
        series = [];
        history.set(key, series);
    }
    series.push(value);
    if (series.length > HISTORY_LIMIT) series.shift();
}

// ── Sparkline: one implementation for every card ───────────────────────────────
// 100x32 viewBox, line + area, tone via data-tone on the <svg>.

function sparklineMarkup(tone) {
    return `<svg class="mcard__spark" viewBox="0 0 100 32" preserveAspectRatio="none" aria-hidden="true"${tone ? ` data-tone="${tone}"` : ''}><path class="mcard__spark-area" fill="currentColor" opacity=".12"/><path class="mcard__spark-line" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/></svg>`;
}

// ── Registry ────────────────────────────────────────────────────────────────────
// fmt helpers kept dependency-light on purpose.

const fmtNum = (v, digits = 0) =>
    Number(v).toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });

const fmtGb = (bytes) => {
    const gb = bytes / (1024 * 1024 * 1024);
    return gb >= 100 ? fmtNum(gb, 0) : fmtNum(gb, 1);
};

const fmtSpeed = (v) => (v >= 100 ? fmtNum(v, 0) : fmtNum(v, 1));

// Optional fact cards only expose backend-reported fields. In particular, do
// not sum overlapping outcome totals or infer a runtime cap from hardware/peak.
const reportedRatio = (value) => {
    const number = metricNumber(value);
    return number != null && number <= 1 ? number : null;
};
const reportedCount = (value) => value == null ? 'Unavailable' : fmtNum(value);
// Runtime bytes are binary: GiB, or MiB for non-zero values under 1 GiB.
const reportedBytes = (value) => {
    const parts = formatRuntimeBytes(value);
    return parts ? `${parts.value} ${parts.unit}` : 'Unavailable';
};
const reportedPercent = (value) => value == null ? 'Unavailable' : `${fmtNum(value * 100, 1)}%`;

function factCard(key, label, icon, read, opts = {}) {
    return {
        key, label, icon, optional: true, wide: Boolean(opts.wide),
        build() {
            return `<div class="mcard__values"><span class="mcard__value" id="mv-${key}">–</span><span class="mcard__unit" id="mu-${key}"></span></div>` +
                `<div class="mcard__facts" id="ms-${key}"></div>` +
                `<div class="mcard__note" id="mn-${key}"></div>`;
        },
        update(view, card) {
            const data = read(view.rapid);
            card.hidden = !data.present;
            // Explanatory copy is visible compact text (a title tooltip is
            // unreachable by touch and keyboard); it stays one muted line.
            if (card.hasAttribute('title')) card.removeAttribute('title');
            card.classList.toggle('mcard--na', !data.present);
            // Always replace every field, even when hidden: a previous reported
            // zero is data, but must never survive a missing snapshot/source.
            const hasValue = data.present && data.value != null;
            setValue(`mv-${key}`, hasValue ? data.value : null, String);
            // A unit with no headline value ("% hit rate" next to "–") is noise.
            setText(`mu-${key}`, hasValue ? data.unit || '' : '');
            setText(`mn-${key}`, data.present ? data.note || '' : '');
            const facts = document.getElementById(`ms-${key}`);
            const rows = data.present ? data.rows : [];
            const signature = JSON.stringify(rows);
            if (facts && facts.dataset.signature !== signature) {
                facts.dataset.signature = signature;
                facts.replaceChildren(...rows.map(text => {
                    const row = document.createElement('div');
                    row.textContent = text;
                    return row;
                }));
            }
        },
    };
}

const RAPID_DEFS = [
    factCard('mtp', 'MTP acceptance', 'gauge', (sample) => {
        const rate = reportedRatio(sample?.acceptance);
        return { present: rate != null, value: rate == null ? null : fmtNum(rate * 100, 1),
            unit: '%', rows: ['Backend-reported speculative acceptance', 'Not a speedup measurement'] };
    }),
    factCard('runtime-memory', 'Runtime memory', 'ram', (sample) => {
        const active = metricNumber(sample?.memory?.active);
        const peak = metricNumber(sample?.memory?.peak);
        const cache = metricNumber(sample?.memory?.cache);
        const cap = metricNumber(sample?.memory?.limit);
        const activeParts = formatRuntimeBytes(active);
        return {
            present: [active, peak, cache, cap].some(value => value != null),
            value: activeParts?.value ?? null, unit: activeParts ? `${activeParts.unit} active` : '',
            // A cap row appears only when the backend actually reports one.
            rows: [`Peak (not a cap): ${reportedBytes(peak)}`,
                `Reclaimable allocator cache: ${reportedBytes(cache)}`,
                ...(cap == null ? [] : [`Explicit runtime cap: ${cap === 0 ? '0.0 GiB reported; usable cap unavailable' : reportedBytes(cap)}`])],
            note: 'Runtime allocations are separate from hardware residency and wired limits.',
        };
    }, { wide: true }),
    factCard('cache', 'Prefix / multimodal cache', 'layers', (sample) => {
        const cache = sample?.cache;
        const hits = metricNumber(cache?.hits);
        const misses = metricNumber(cache?.misses);
        const rate = reportedRatio(cache?.hitRate);
        const entries = metricNumber(cache?.entries);
        const memory = metricNumber(cache?.memoryBytes);
        const kinds = Array.isArray(cache?.kinds)
            ? cache.kinds.slice(0, 8)
                .filter(value => typeof value === 'string')
                .map(value => value.slice(0, 128).trim()).filter(Boolean) : [];
        return {
            present: [hits, misses, rate, entries, memory].some(value => value != null) || kinds.length > 0,
            value: rate == null ? null : fmtNum(rate * 100, 1), unit: '% hit rate',
            rows: [`Hit rate: ${reportedPercent(rate)}`, `Hits: ${reportedCount(hits)}`,
                `Misses: ${reportedCount(misses)}`, `Entries: ${reportedCount(entries)}`,
                `Prefix cache memory: ${reportedBytes(memory)}`,
                `Multimodal kinds: ${kinds.join(', ') || 'Unavailable'}`],
        };
    }),
];

const METRIC_DEFS = [
    {
        key: 'speed',
        label: 'Speed',
        icon: 'gauge',
        // The one special card: decode + prefill values and two overlaid
        // sparklines (prefill nested, dimmer — Strata's layered bar look).
        build() {
            return `<div class="mcard__values"><div><span class="mcard__value" id="mv-decode">–</span><span class="mcard__unit">t/s</span></div>` +
                `<div class="mcard__alt"><span class="mcard__value" id="mv-prefill">–</span><span class="mcard__unit">t/s</span></div></div>` +
                `<div class="mcard__sub"><span id="ms-decode">Decode</span><span id="ms-prefill" class="mcard__sub-alt">Prefill</span></div>` +
                // One strip, two overlaid lines — Strata-style: the active
                // phase draws bright in front, the other phase dimmed behind.
                `<svg class="mcard__spark" viewBox="0 0 100 32" preserveAspectRatio="none" aria-hidden="true">` +
                `<path class="mcard__spark-area" fill="currentColor" opacity=".12"/><path class="mcard__spark-line" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>` +
                `<g class="mcard__spark-alt"><path class="mcard__spark-area" fill="currentColor" opacity=".08"/><path class="mcard__spark-line" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/></g></svg>`;
        },
        update(view) {
            const decode = view.decodeTps;
            const prefill = view.prefillTps;
            setPairValue('decode', decode, 't/s');
            setValue('mv-prefill', prefill, fmtSpeed);
            const decodeSub = view.decodeScope ? `Decode ${view.decodeScope}`
                : decode != null ? 'Decode last measured' : 'Decode unavailable';
            const prefillSub = view.prefillScope ? `Prefill ${view.prefillScope}`
                : prefill != null ? 'Prefill last measured' : 'Prefill unavailable';
            setText('ms-decode', decodeSub);
            setText('ms-prefill', prefillSub);
            const svg = document.getElementById('mc-card-speed');
            if (!svg) return;
            // Bright line = the active phase; dim line = the other one
            // (held at its last reported value while inactive).
            const decoding = view.state === 'generating';
            const main = decoding ? history.get('decode') : history.get('prefill');
            const alt = decoding ? history.get('prefill') : history.get('decode');
            drawPaths(svg.querySelector('.mcard__spark-line'), svg.querySelector('.mcard__spark-area'), main);
            const altLine = svg.querySelector('.mcard__spark-alt .mcard__spark-line');
            const altArea = svg.querySelector('.mcard__spark-alt .mcard__spark-area');
            drawPaths(altLine, altArea, alt);
        },
    },
    {
        key: 'queue',
        label: 'Queue',
        icon: 'layers',
        pick: (v) => v.waiting,
        unit: '',
        max: null,
        sub: (v) => (v.running == null ? 'Running count unavailable' : v.running ? `${fmtNum(v.running)} running` : 'waiting requests'),
    },
    {
        key: 'gpu',
        label: 'GPU load',
        icon: 'gpu',
        pick: (v) => (v.gpu ? v.gpu.load : null),
        unit: '%',
        max: 100,
        sub: (v) => (v.gpu ? v.gpu.loadNote || v.gpu.name : ''),
    },
    {
        key: 'vram',
        label: 'VRAM',
        icon: 'layers',
        pick: (v) => (v.gpu?.vramUsed != null && v.gpu.unifiedTotal > 0 ? v.gpu.vramUsed / v.gpu.unifiedTotal : null),
        unit: '',
        max: 1,
        value: (v) => (v.gpu ? `${fmtGb(v.gpu.vramUsed)}` : null),
        // Hardware residency and wired limits are not runtime allocations/caps.
        // Keep the physical pool denominator; runtime memory is shown separately.
        unitText: (v) => (v.gpu?.metalUnified ? `of ${fmtGb(v.gpu.unifiedTotal)} GB` : (v.gpu && v.gpu.vramTotal ? `/ ${fmtGb(v.gpu.vramTotal)} GB` : 'GB')),
        sub: (v) => (v.gpu?.metalUnified
            ? `Hardware Metal wired cap ${fmtGb(v.gpu.vramTotal)} GB`
            : (v.gpu ? v.gpu.name : '')),
    },
    {
        key: 'temp',
        label: 'GPU temp',
        icon: 'thermometer',
        pick: (v) => (v.gpu ? v.gpu.temp : null),
        unit: '°C',
        max: 100,
        tone: 'warn',
        sub: () => 'of 100 °C',
    },
    {
        key: 'power',
        label: 'GPU power',
        icon: 'bolt',
        pick: (v) => (v.gpu ? v.gpu.power : null),
        unit: 'W',
        sub: (v) => (v.gpu && v.gpu.powerLimit ? `of ${fmtNum(v.gpu.powerLimit)} W limit` : 'GPU only; not CPU / SoC power'),
    },
    {
        key: 'cpu',
        label: 'CPU',
        icon: 'cpu',
        pick: (v) => (v.sys ? v.sys.cpu : null),
        unit: '%',
        max: 100,
        sub: (v) => (v.sys ? v.sys.cpuName : ''),
    },
    ...RAPID_DEFS,
];

// ── Shell rendering ─────────────────────────────────────────────────────────────

const ICONS = {
    gauge: '<path d="M12 4a10 10 0 0 1 10 10 10 10 0 0 1-1.4 5.1M12 4A10 10 0 0 0 2 14a10 10 0 0 0 1.4 5.1"/><path d="M12 8v4l3 3"/>',
    gpu: '<rect x="2" y="6" width="20" height="12" rx="2"/><circle cx="9" cy="12" r="2.5"/><path d="M14 10h5M14 14h5"/>',
    layers: '<path d="M12 2 2 7l10 5 10-5-10-5z"/><path d="m2 12 10 5 10-5"/><path d="m2 17 10 5 10-5"/>',
    thermometer: '<path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z"/>',
    bolt: '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    cpu: '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M1 9h3M1 15h3M20 9h3M20 15h3"/>',
    ram: '<rect x="3" y="7" width="18" height="10" rx="2"/><path d="M7 17v3M12 17v3M17 17v3"/>',
};

function iconSvg(name) {
    const path = ICONS[name] || ICONS.gauge;
    return `<svg class="mcard__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${path}</svg>`;
}

export function initMetricCards() {
    const grid = document.getElementById('metrics-grid');
    if (!grid || grid.dataset.built === '1') return;
    grid.dataset.built = '1';
    // Preserve the static llama.cpp runtime card while building registry cards.
    const runtimeCard = document.getElementById('llama-runtime-card');
    // Markup is built entirely from the static registry below — no user data.
    setHtml(grid, METRIC_DEFS.map((m) => {
        if (m.build) {
            return `<div class="mcard${m.wide ? ' mcard--wide' : ''}" id="mc-card-${m.key}"${m.optional ? ' hidden' : ''}><div class="mcard__label">${iconSvg(m.icon)}${m.label}</div>${m.build()}</div>`;
        }
        return `<div class="mcard" id="mc-card-${m.key}"><div class="mcard__label">${iconSvg(m.icon)}${m.label}</div>` +
            `<div class="mcard__values"><span class="mcard__value" id="mv-${m.key}">–</span><span class="mcard__unit" id="mu-${m.key}">${m.unit || ''}</span></div>` +
            `<div class="mcard__sub" id="ms-${m.key}"></div>` +
            sparklineMarkup(m.tone) + `</div>`;
    }).join(''));
    if (runtimeCard) grid.appendChild(runtimeCard);
}

// ── Update ──────────────────────────────────────────────────────────────────────

function setText(id, text) {
    const el = document.getElementById(id);
    if (el && el.textContent !== text) el.textContent = text;
}

function setValue(id, value, fmt) {
    const el = document.getElementById(id);
    if (!el) return;
    const next = value == null ? '–' : fmt(value);
    if (el.textContent !== next) el.textContent = next;
}

function setPairValue(key, value) {
    const el = document.getElementById(`mv-${key}`);
    if (!el) return;
    const next = value == null ? '–' : `${fmtSpeed(value)}`;
    if (el.textContent !== next) el.textContent = next;
}

function drawPaths(line, area, values, max) {
    if (!line || !area) return;
    // Strata semantics: nulls plot as 0, so a series visibly drops to the
    // baseline when its phase ends and rises when it starts again.
    const v = (values || []).map((x) => (Number.isFinite(x) ? x : 0));
    if (v.length < 2) {
        line.setAttribute('d', '');
        area.setAttribute('d', '');
        return;
    }
    const top = Math.max(max || 0, ...v, 1e-9);
    const pts = v.map((x, i) => [(i / (v.length - 1)) * 100, 30 - (x / top) * 26]);
    const d = pts.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join('');
    line.setAttribute('d', d);
    area.setAttribute('d', `${d}L100,32L0,32Z`);
}

export function updateMetricCards(view) {
    initMetricCards();
    selectHistoryOwner(view);
    // Optional facts belong only to the attached Rapid target. Gating here
    // also clears stale normalized facts if a caller switches the backend.
    const cardView = { ...view, rapid: view.backend === 'rapid_mlx' && view.attached ? view.rapid : null };
    for (const m of METRIC_DEFS) {
        const card = document.getElementById(`mc-card-${m.key}`);
        if (!card) continue;
        if (m.key === 'speed') {
            // Strata plot semantics: the active phase plots its rate, the
            // inactive phase plots 0 (the line drops to the baseline the
            // moment the phase ends), and Idle appends nothing — the whole
            // card freezes once the request completes.
            // pushHistory throttles each series to one sample per second.
            if (view.state === 'generating' || view.state === 'reading') {
                // An inactive phase may plot zero; an active phase without a
                // current rate is missing data and must not fabricate a zero.
                const decodeActive = view.decodeActive ?? view.state === 'generating';
                const prefillActive = view.prefillActive ?? view.state === 'reading';
                if (!decodeActive) pushHistory('decode', 0);
                else if (view.decodeCurrent) pushHistory('decode', view.decodeTps);
                if (!prefillActive) pushHistory('prefill', 0);
                else if (view.prefillCurrent) pushHistory('prefill', view.prefillTps);
            }
            m.update(cardView, card);
            continue;
        }
        // Custom registry cards are not restricted to the dual-rate speed card.
        if (m.update) {
            m.update(cardView, card);
            continue;
        }
        const raw = m.pick(view);
        pushHistory(m.key, raw);
        const na = raw == null;
        card.classList.toggle('mcard--na', na);
        if (na) {
            setValue(`mv-${m.key}`, null, String);
            setText(`ms-${m.key}`, m.key === 'gpu' && view.gpu?.loadNote
                ? view.gpu.loadNote : 'unavailable');
            setText(`mu-${m.key}`, m.unit || '');
            const svg = card.querySelector('.mcard__spark');
            if (svg) {
                drawPaths(svg.querySelector('.mcard__spark-line'), svg.querySelector('.mcard__spark-area'), []);
            }
            continue;
        }
        // Value: custom formatter, else scaled number with unit.
        let display;
        if (m.value) display = m.value(view);
        else if (m.max === 1) display = `${Math.round(raw * 100)}`;
        else display = fmtNum(raw, raw < 10 && raw % 1 !== 0 ? 1 : 0);
        setValue(`mv-${m.key}`, display, String);
        setText(`mu-${m.key}`, m.unitText ? m.unitText(view) : (m.unit || ''));
        setText(`ms-${m.key}`, m.sub ? m.sub(view) : '');
        const svg = card.querySelector('.mcard__spark');
        if (svg) {
            const values = history.get(m.key) || [];
            drawPaths(svg.querySelector('.mcard__spark-line'), svg.querySelector('.mcard__spark-area'), values, m.max);
        }
        // RAM danger tone at >92%.
        if (m.toneDangerAt != null) {
            if (raw >= m.toneDangerAt) svg?.setAttribute('data-tone', 'danger');
            else svg?.removeAttribute('data-tone');
        }
    }
}

// ── Model state card ────────────────────────────────────────────────────────────

// Every view.state maps to exactly one pill. Unknown states fall back to
// "unavailable" rather than leaving the strip with no active pill.
const STATE_PILLS = ['idle', 'reading', 'generating', 'queued', 'busy', 'unavailable', 'error'];

function activePillFor(view) {
    // Nothing attached: the strip reads as idle, not as a telemetry fault.
    if (view.attached === false) return 'idle';
    return STATE_PILLS.includes(view.state) ? view.state : 'unavailable';
}

export function updateStateCard(view) {
    const card = document.getElementById('state-card');
    if (!card) return;
    const activePill = activePillFor(view);
    for (const s of STATE_PILLS) {
        const pill = card.querySelector(`.state-pill[data-s="${s}"]`);
        if (!pill) continue;
        pill.classList.toggle('active', activePill === s || (s === 'queued' && view.queued > 0));
    }
    const queuedPill = card.querySelector('.state-pill[data-s="queued"]');
    if (queuedPill) {
        queuedPill.classList.toggle('counted', !!view.queued);
        const queuedText = view.queued > 0 ? `Queued · ${view.queued}` : 'Queued';
        if (queuedPill.textContent !== queuedText) queuedPill.textContent = queuedText;
    }
    // #state-label is aria-live=polite and only changes on a state transition;
    // #state-detail changes every tick and is deliberately not live.
    setText('state-label', view.stateLabel || 'Waiting for a request');
    setText('state-detail', view.stateDetail || '');
    const bar = document.getElementById('state-bar');
    const prog = document.getElementById('state-progress');
    if (bar && prog) {
        const determinate = typeof view.stateProgress === 'number' && Number.isFinite(view.stateProgress);
        const indeterminate = view.state === 'reading' && !determinate;
        const pct = determinate ? Math.min(100, Math.max(0, view.stateProgress * 100)) : 0;
        const hidden = !determinate && !indeterminate;
        // Indeterminate width belongs to CSS (animated segment, or the static
        // full-width bar under reduced motion); clear any inline determinate width.
        const width = indeterminate ? '' : `${pct}%`;
        if (bar.style.width !== width) bar.style.width = width;
        if (view.stateTone) prog.dataset.tone = view.stateTone;
        else delete prog.dataset.tone;
        if (indeterminate) prog.dataset.indeterminate = 'true';
        else delete prog.dataset.indeterminate;
        prog.classList.toggle('state-progress--hidden', hidden);
        // role/valuemin/valuemax are static in index.html; only dynamic ARIA here.
        prog.setAttribute('aria-hidden', String(hidden));
        prog.setAttribute('aria-label', view.state === 'reading' ? 'Prompt tokens processed' : 'Output budget used');
        if (determinate) {
            prog.setAttribute('aria-valuenow', String(Math.round(pct)));
            prog.setAttribute('aria-valuetext', `${pct.toFixed(1)}% ${view.state === 'reading' ? 'of prompt tokens processed' : 'of output budget used; not predicted completion'}`);
        } else {
            prog.removeAttribute('aria-valuenow');
            if (indeterminate) prog.setAttribute('aria-valuetext', 'Reading activity; processed prompt tokens unavailable');
            else prog.removeAttribute('aria-valuetext');
        }
    }
}
