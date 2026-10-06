// ── Unified metric cards + model state strip ───────────────────────────────────
// Strata-style dashboard presentation: a state card (pills + progress) and a
// registry-driven grid of compact metric cards, each with a single inline-SVG
// sparkline (line + 12% area fill, tone via data-tone).
//
// The registry decouples presentation from the runtime loader: every card
// declares a `pick(view)` reader over the NORMALIZED snapshot built by
// dashboard-ws.js, so llama.cpp, Rapid-MLX, and remote agents feed the same
// cards. A card whose source has no data for the active loader renders dimmed
// dimmed ("n/a") instead of disappearing.

import { setHtml } from '../core/set-html.js';

// 300 samples at the ~1s dashboard push ≈ a 5-minute window per sparkline.
// Pushes slow down when the tab is hidden or in low-power mode, so the wall
// clock span can stretch — the sample budget stays fixed.
const HISTORY_LIMIT = 300;
const history = new Map();

function pushHistory(key, value) {
    if (!Number.isFinite(value)) return;
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
            const decodeSub = view.state === 'generating' ? 'Decode now'
                : decode != null ? 'Decode last request' : 'Decode';
            const prefillSub = view.state === 'reading' ? 'Prefill now'
                : view.state === 'generating' ? 'Prefill this request'
                    : prefill != null ? 'Prefill last request' : 'Prefill';
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
        sub: (v) => (v.running ? `${fmtNum(v.running)} running` : 'waiting requests'),
    },
    {
        key: 'gpu',
        label: 'GPU load',
        icon: 'gpu',
        pick: (v) => (v.gpu ? v.gpu.load : null),
        unit: '%',
        max: 100,
        sub: (v) => (v.gpu ? v.gpu.name : ''),
    },
    {
        key: 'vram',
        label: 'VRAM',
        icon: 'layers',
        pick: (v) => (v.gpu && v.gpu.unifiedTotal ? v.gpu.vramUsed / v.gpu.unifiedTotal : null),
        unit: '',
        max: 1,
        value: (v) => (v.gpu ? `${fmtGb(v.gpu.vramUsed)}` : null),
        // Apple Silicon memory is drawn from the whole unified pool, so the
        // denominator is the pool size; the Metal wired-limit cap (the real
        // ceiling for GPU-resident weights) goes in the sub-label.
        unitText: (v) => (v.gpu?.metalUnified ? `of ${fmtGb(v.gpu.unifiedTotal)} GB` : (v.gpu && v.gpu.vramTotal ? `/ ${fmtGb(v.gpu.vramTotal)} GB` : 'GB')),
        sub: (v) => (v.gpu?.metalUnified
            ? `Metal cap ${fmtGb(v.gpu.vramTotal)} GB`
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
        label: 'Power',
        icon: 'bolt',
        pick: (v) => (v.gpu ? v.gpu.power : null),
        unit: 'W',
        sub: (v) => (v.gpu && v.gpu.powerLimit ? `of ${fmtNum(v.gpu.powerLimit)} W limit` : ''),
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
    // Markup is built entirely from the static registry below — no user data.
    setHtml(grid, METRIC_DEFS.map((m) => {
        if (m.build) {
            return `<div class="mcard" id="mc-card-${m.key}"><div class="mcard__label">${iconSvg(m.icon)}${m.label}</div>${m.build()}</div>`;
        }
        return `<div class="mcard" id="mc-card-${m.key}"><div class="mcard__label">${iconSvg(m.icon)}${m.label}</div>` +
            `<div class="mcard__values"><span class="mcard__value" id="mv-${m.key}">–</span><span class="mcard__unit" id="mu-${m.key}">${m.unit || ''}</span></div>` +
            `<div class="mcard__sub" id="ms-${m.key}"></div>` +
            sparklineMarkup(m.tone) + `</div>`;
    }).join(''));
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
    for (const m of METRIC_DEFS) {
        const card = document.getElementById(`mc-card-${m.key}`);
        if (!card) continue;
        if (m.key === 'speed') {
            // Push each series only while its phase is active; the other
            // holds its last reported value (pushHistory skips non-finite).
            if (view.state === 'generating') pushHistory('decode', view.decodeTps);
            else if (view.state === 'reading') pushHistory('prefill', view.prefillTps);
            m.update(view);
            continue;
        }
        const raw = m.pick(view);
        pushHistory(m.key, m.max === 1 ? raw : raw);
        const na = raw == null;
        card.classList.toggle('mcard--na', na);
        if (na) {
            setValue(`mv-${m.key}`, null, String);
            setText(`ms-${m.key}`, 'not available for this backend');
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

const STATE_PILLS = ['idle', 'reading', 'generating', 'queued', 'error'];

export function updateStateCard(view) {
    const card = document.getElementById('state-card');
    if (!card) return;
    for (const s of STATE_PILLS) {
        const pill = card.querySelector(`.state-pill[data-s="${s}"]`);
        if (!pill) continue;
        pill.classList.toggle('active', view.state === s || (s === 'queued' && view.queued > 0));
    }
    const queuedPill = card.querySelector('.state-pill[data-s="queued"]');
    if (queuedPill) {
        queuedPill.classList.toggle('counted', !!view.queued);
        queuedPill.textContent = view.queued > 0 ? `Queued · ${view.queued}` : 'Queued';
    }
    setText('state-label', view.stateLabel || 'Waiting for a request');
    setText('state-detail', view.stateDetail || '');
    const bar = document.getElementById('state-bar');
    const prog = document.getElementById('state-progress');
    if (bar && prog) {
        const pct = view.stateProgress != null ? Math.min(100, Math.max(0, view.stateProgress * 100)) : 0;
        bar.style.width = `${pct}%`;
        if (view.stateTone) prog.dataset.tone = view.stateTone;
        else delete prog.dataset.tone;
        prog.classList.toggle('state-progress--hidden', view.stateProgress == null);
    }
}
