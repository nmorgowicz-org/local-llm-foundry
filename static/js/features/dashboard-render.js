import { escapeHtml, formatMetricNumber, formatDuration, formatClockReadout } from '../core/format.js';
import { gradeActionCopy } from './telemetry-grade.js';
import {
    chat,
    metricSeries,
    liveOutputTracker,
    requestActivity,
    recentTasks,
    lastGpuData,
    lastSystemMetrics,
    lastCapabilities,
    wsData,
    setLastGpuData,
} from '../core/app-state.js';

function setChipState(el, label, state) {
    if (!el) return;
    el.textContent = label;
    el.className = 'metric-live-chip ' + (state || '');
}

/** Read a CSS custom property from :root, resolved at call time so palette changes are reflected immediately. */
function getCSSVar(name, fallback = '') {
    const val = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return val || fallback;
}

function lerpColor(a, b, t) {
    return [
        Math.round(a[0] + (b[0] - a[0]) * t),
        Math.round(a[1] + (b[1] - a[1]) * t),
        Math.round(a[2] + (b[2] - a[2]) * t)
    ];
}

let sparklineGradientSeq = 0;

function nextSparklineGradientId(prefix) {
    sparklineGradientSeq += 1;
    return prefix + '-spark-fill-' + sparklineGradientSeq;
}

function hexToRgb(color) {
    if (typeof color !== 'string') return null;
    const match = color.trim().match(/^#([0-9a-f]{6})$/i);
    if (!match) return null;
    const raw = match[1];
    return [
        parseInt(raw.slice(0, 2), 16),
        parseInt(raw.slice(2, 4), 16),
        parseInt(raw.slice(4, 6), 16)
    ];
}

function getSparklineFillColor(ratio) {
    const clamped = Math.max(0, Math.min(1, Number.isFinite(ratio) ? ratio : 0));
    if (clamped < 0.6) {
        return lerpColor([80, 200, 120], [235, 203, 139], clamped / 0.6);
    }
    return lerpColor([235, 203, 139], [200, 80, 80], (clamped - 0.6) / 0.4);
}

function getThemedSparklineFillColor(color, ratio = 0.5) {
    return hexToRgb(color) || getSparklineFillColor(ratio);
}

function buildSparklineFillDefs(fillId, fillColor, topOpacity = 0.62, midOpacity = 0.2, bottomOpacity = 0.04) {
    const fillRgb = 'rgb(' + fillColor.join(',') + ')';
    return (
        '<defs>' +
          '<linearGradient id="' + fillId + '" x1="0" y1="0" x2="0" y2="1">' +
            '<stop offset="0%" stop-color="' + fillRgb + '" stop-opacity="' + topOpacity.toFixed(2) + '"></stop>' +
            '<stop offset="62%" stop-color="' + fillRgb + '" stop-opacity="' + midOpacity.toFixed(2) + '"></stop>' +
            '<stop offset="100%" stop-color="' + fillRgb + '" stop-opacity="' + bottomOpacity.toFixed(2) + '"></stop>' +
          '</linearGradient>' +
        '</defs>'
    );
}

function getInferenceSparklineColor(className) {
    const accent = getCSSVar('--color-accent', '#2dd4bf');
    const accentLight = getCSSVar('--color-accent-light', '#5eead4');
    const clock = getCSSVar('--metric-clock', '#60a5fa');
    if (className === 'prompt') return clock;
    if (className === 'generation') return accentLight;
    if (className === 'live-output') return accent;
    return accentLight;
}

function setCardState(card, state) {
    if (!card) return;
    card.classList.remove('is-live', 'is-idle', 'is-unavailable', 'is-dormant');
    if (state) card.classList.add('is-' + state);
}

function pushSparklinePoint(name, value) {
    metricSeries[name].push(Number.isFinite(value) ? value : 0);
    const limit = name === 'liveOutput' ? 90 : 40;
    if (metricSeries[name].length > limit) {
        metricSeries[name].shift();
    }
}

/** Last sparkline data snapshot for change detection */
var lastSparklineSnapshots = {};

function renderSparkline(id, points, className, isBlocked) {
    const svg = document.getElementById(id);
    if (!svg || !points || points.length < 2) return;

    // POWER OPT: skip rebuild if data hasn't changed
    const key = id + ':' + className;
    const lastSnapshot = lastSparklineSnapshots[key];
    if (lastSnapshot && lastSnapshot.length === points.length && lastSnapshot[lastSnapshot.length - 1] === points[points.length - 1]) {
        return;
    }
    lastSparklineSnapshots[key] = [...points];

    svg.style.color = getInferenceSparklineColor(className);
    const width = 120;
    const height = 28;
    const max = Math.max(...points, 1);
    const step = width / (points.length - 1);
    const currentValue = points[points.length - 1];
    const currentX = width;
    const currentY = height - ((currentValue / max) * (height - 4)) - 2;
    const path = points.map((value, index) => {
        const x = index * step;
        const y = height - ((value / max) * (height - 4)) - 2;
        return (index === 0 ? 'M' : 'L') + x.toFixed(2) + ' ' + y.toFixed(2);
    }).join(' ');
    // Level-based fill color: green (low) → yellow (mid) → red (high)
    const ratio = max > 0 ? currentValue / max : 0;
    const fillColor = getThemedSparklineFillColor(getInferenceSparklineColor(className), ratio);
    const fillId = nextSparklineGradientId(id);
    const wallLine = isBlocked ? '<line x1="120" y1="0" x2="120" y2="28" stroke="' + getCSSVar('--color-warning', '#f59e0b') + '" stroke-width="1" stroke-dasharray="3 3" opacity="0.5"/>' : '';
    // eslint-disable-next-line no-unsanitized/property -- SVG path data from numeric array values; className is a hardcoded CSS class
    svg.innerHTML =
        buildSparklineFillDefs(fillId, fillColor, 0.88, 0.18, 0.02) +
        '<path class="sparkline-fill ' + className + '" d="' + path + ' L 120 28 L 0 28 Z" fill="url(#' + fillId + ')"></path>' +
        '<line class="sparkline-grid" x1="0" y1="7" x2="120" y2="7"/><line class="sparkline-grid" x1="0" y1="14" x2="120" y2="14"/><line class="sparkline-grid" x1="0" y1="21" x2="120" y2="21"/>' +
        '<path class="sparkline-line ' + className + '" d="' + path + '"></path>' +
        '<line class="sparkline-current-trace ' + className + '" x1="' + Math.max(currentX - 8, 0).toFixed(2) + '" y1="' + currentY.toFixed(2) + '" x2="' + currentX.toFixed(2) + '" y2="' + currentY.toFixed(2) + '"></line>' +
        '<circle class="sparkline-current-halo ' + className + '" cx="' + currentX.toFixed(2) + '" cy="' + currentY.toFixed(2) + '" r="7.4"></circle>' +
        '<circle class="sparkline-current ' + className + '" cx="' + currentX.toFixed(2) + '" cy="' + currentY.toFixed(2) + '" r="3.6"></circle>' +
        '<circle class="sparkline-current-core ' + className + '" cx="' + currentX.toFixed(2) + '" cy="' + currentY.toFixed(2) + '" r="1.2"></circle>' +
        wallLine;
}

function renderLiveSparkline(id, points) {
    const svg = document.getElementById(id);
    if (!svg) return;
    if (!points || points.length < 2) {
        svg.innerHTML = '';
        return;
    }

    // POWER OPT: skip rebuild if data hasn't changed
    const key = id + ':live';
    const lastSnap = lastSparklineSnapshots[key];
    if (lastSnap && lastSnap.length === points.length && lastSnap[lastSnap.length - 1] === points[points.length - 1]) {
        return;
    }
    lastSparklineSnapshots[key] = [...points];
    svg.style.color = getInferenceSparklineColor('live-output');
    const width = 120;
    const height = 28;
    const max = Math.max(...points, 1);
    const step = width / (points.length - 1);
    let peak = { value: -1, x: 0, y: height - 2 };
    const path = points.map((value, index) => {
        const x = index * step;
        const y = height - ((value / max) * (height - 6)) - 3;
        if (value > peak.value) peak = { value, x, y };
        return (index === 0 ? 'M' : 'L') + x.toFixed(2) + ' ' + y.toFixed(2);
    }).join(' ');
    const currentValue = points[points.length - 1];
    const currentX = width;
    const currentY = height - ((currentValue / max) * (height - 6)) - 3;
    const ratio = max > 0 ? currentValue / max : 0;
    const fillColor = getThemedSparklineFillColor(getInferenceSparklineColor('live-output'), ratio);
    const fillId = nextSparklineGradientId(id);
    // eslint-disable-next-line no-unsanitized/property -- SVG path data built from numeric array values only
    svg.innerHTML = [
        buildSparklineFillDefs(fillId, fillColor, 0.90, 0.20, 0.02),
        '<path class="sparkline-fill live-output" d="' + path + ' L 120 28 L 0 28 Z" fill="url(#' + fillId + ')"></path>',
        '<line class="sparkline-grid" x1="0" y1="7" x2="120" y2="7"/><line class="sparkline-grid" x1="0" y1="14" x2="120" y2="14"/><line class="sparkline-grid" x1="0" y1="21" x2="120" y2="21"/>',
        '<path class="sparkline-line live-output" d="' + path + '"></path>',
        '<line class="sparkline-current-trace live-output" x1="' + Math.max(currentX - 8, 0).toFixed(2) + '" y1="' + currentY.toFixed(2) + '" x2="' + currentX.toFixed(2) + '" y2="' + currentY.toFixed(2) + '"></line>',
        '<circle class="sparkline-peak live-output" cx="' + peak.x.toFixed(2) + '" cy="' + peak.y.toFixed(2) + '" r="2.6"></circle>',
        '<circle class="sparkline-current-halo live-output" cx="' + currentX.toFixed(2) + '" cy="' + currentY.toFixed(2) + '" r="5"></circle>',
        '<circle class="sparkline-current live-output" cx="' + currentX.toFixed(2) + '" cy="' + currentY.toFixed(2) + '" r="2.8"></circle>',
        '<circle class="sparkline-current-core live-output" cx="' + currentX.toFixed(2) + '" cy="' + currentY.toFixed(2) + '" r="1"></circle>'
    ].join('');
}

function getTaskKey(taskId, active) {
    if (taskId !== null && taskId !== undefined) return String(taskId);
    return active ? 'active-unknown' : null;
}

function updateLiveOutputEstimate(taskId, decoded, active, nowMs) {
    const tracker = liveOutputTracker;
    const taskChanged = tracker.taskId !== taskId;
    if (taskChanged) {
        tracker.taskId = taskId;
        tracker.previousDecoded = Number.isFinite(decoded) ? decoded : null;
        tracker.previousMs = nowMs;
        tracker.latestRate = 0;
        tracker.rates = [];
        metricSeries.liveOutput = [];
        pushSparklinePoint('liveOutput', 0);
        return 0;
    }

    let rate = 0;
    if (active && Number.isFinite(decoded) && Number.isFinite(tracker.previousDecoded) && tracker.previousMs) {
        const deltaTokens = Math.max(0, decoded - tracker.previousDecoded);
        const deltaSeconds = (nowMs - tracker.previousMs) / 1000;
        rate = deltaSeconds > 0 ? deltaTokens / deltaSeconds : 0;
    }

    if (active && rate > 0) {
        tracker.rates.push(rate);
        if (tracker.rates.length > 6) tracker.rates.shift();
    } else if (!active) {
        tracker.rates = [];
    }

    const smoothedRate = tracker.rates.length
        ? tracker.rates.reduce((sum, value) => sum + value, 0) / tracker.rates.length
        : 0;

    tracker.previousDecoded = Number.isFinite(decoded) ? decoded : tracker.previousDecoded;
    tracker.previousMs = nowMs;
    tracker.latestRate = smoothedRate;
    pushSparklinePoint('liveOutput', smoothedRate);
    return smoothedRate;
}

function updateRequestActivity(taskId, active, outputTokens, nowMs) {
    const taskKey = getTaskKey(taskId, active);
    let openSegment = requestActivity.find(segment => !segment.endedAtMs);

    if (active && taskKey) {
        if (!openSegment || openSegment.taskKey !== taskKey) {
            if (openSegment) {
                openSegment.endedAtMs = nowMs;
                openSegment.outputTokens = outputTokens || openSegment.outputTokens || 0;
                recentTasks.unshift(openSegment);
            }
            requestActivity.push({
                taskKey,
                taskId,
                startedAtMs: nowMs,
                firstOutputAtMs: outputTokens > 0 ? nowMs : null,
                endedAtMs: null,
                state: 'active',
                outputTokens: outputTokens || 0
            });
        } else {
            if (!openSegment.firstOutputAtMs && outputTokens > 0) {
                openSegment.firstOutputAtMs = nowMs;
            }
            openSegment.outputTokens = outputTokens || openSegment.outputTokens || 0;
        }
    } else if (openSegment) {
        openSegment.endedAtMs = nowMs;
        openSegment.state = 'complete';
        if (!openSegment.firstOutputAtMs && outputTokens > 0) {
            openSegment.firstOutputAtMs = nowMs;
        }
        openSegment.outputTokens = outputTokens || openSegment.outputTokens || 0;
        recentTasks.unshift(openSegment);
    }

    const cutoff = nowMs - (10 * 60 * 1000);
    const kept = requestActivity
        .filter(segment => !segment.endedAtMs || segment.endedAtMs >= cutoff)
        .slice(-100);
    requestActivity.splice(0, requestActivity.length, ...kept);
    recentTasks.splice(8);
}

function renderRecentTask() {
    const el = document.getElementById('m-recent-task');
    if (!el) return;
    const task = recentTasks[0];
    if (!task || !task.endedAtMs) {
        el.style.display = 'none';
        el.textContent = '';
        return;
    }
    const durationMs = task.endedAtMs - task.startedAtMs;
    const rate = durationMs > 0 ? (task.outputTokens / (durationMs / 1000)) : 0;
    const id = task.taskId !== null && task.taskId !== undefined ? task.taskId : 'unknown';
    el.style.display = '';
    el.textContent = 'Last task ' + id + ' · ' + formatMetricNumber(task.outputTokens || 0) + ' output tokens · ' + formatDuration(durationMs) + ' · ~' + rate.toFixed(1) + ' t/s estimated';
}

function renderActivityRail(active) {
    const rail = document.getElementById('m-activity-rail');
    if (!rail) return;
    const now = Date.now();
    const windowMs = 5 * 60 * 1000;
    const segments = requestActivity.slice(-28);
    if (!segments.length) {
        rail.innerHTML = '<span class="activity-empty">No recent tasks</span>';
        return;
    }
    // all values are numeric (toFixed) or internal hardcoded enums ('active'/'complete')
    // eslint-disable-next-line no-unsanitized/property -- DOMPurify sanitizes HTML
    rail.innerHTML = window.DOMPurify.sanitize(segments.map(segment => {
        let start = Math.max(0, Math.min(100, ((segment.startedAtMs - (now - windowMs)) / windowMs) * 100));
        const endMs = segment.endedAtMs || now;
        const minWidth = segment.endedAtMs ? 3 : 8;
        let width = Math.max(minWidth, Math.min(100 - start, ((endMs - segment.startedAtMs) / windowMs) * 100));
        if (start + width > 100) {
            start = Math.max(0, 100 - width);
        }
        const firstOutputAtMs = segment.firstOutputAtMs || (segment.outputTokens > 0 ? endMs : null);
        const phaseTotalMs = Math.max(1, endMs - segment.startedAtMs);
        const promptPct = firstOutputAtMs
            ? Math.max(12, Math.min(72, ((firstOutputAtMs - segment.startedAtMs) / phaseTotalMs) * 100))
            : 100;
        const generationPct = firstOutputAtMs ? Math.max(18, 100 - promptPct) : 0;
        const duration = formatDuration(endMs - segment.startedAtMs);
        const task = segment.taskId !== null && segment.taskId !== undefined ? segment.taskId : 'unknown';
        const cls = segment.endedAtMs ? 'complete' : active ? 'active' : 'complete';
        const title = 'task ' + task + ' · ' + duration + ' · ' + formatMetricNumber(segment.outputTokens || 0) + ' output tokens';
        const phases = [
            '<span class="activity-phase prompt" style="width:' + promptPct.toFixed(2) + '%"></span>',
            generationPct > 0 ? '<span class="activity-phase generation" style="width:' + generationPct.toFixed(2) + '%"></span>' : '',
            segment.endedAtMs ? '<span class="activity-marker" aria-hidden="true"></span>' : ''
        ].join('');
        return '<span class="activity-segment ' + cls + '" style="left:' + start.toFixed(2) + '%;width:' + width.toFixed(2) + '%" tabindex="0" title="' + title + '">' + phases + '</span>';
    }).join(''));
}

function renderSamplerParamsInline(slot) {
    const el = document.getElementById('m-sampler-params-inline');
    if (!el || !slot || !slot.sampler_config) {
        el.innerHTML = '';
        return;
    }
    const samplerItems = slot.sampler_config || [];
    const priorityKeys = ['temp', 'top_k', 'top_p', 'min_p', 'dry', 'xtc'];
    const priorityItems = samplerItems
        .filter(item => priorityKeys.includes(item.label))
        .sort((a, b) => priorityKeys.indexOf(a.label) - priorityKeys.indexOf(b.label));
    if (priorityItems.length === 0) {
        el.innerHTML = '';
        return;
    }
    // eslint-disable-next-line no-unsanitized/property -- all interpolated values wrapped in escapeHtml()
    el.innerHTML = priorityItems.slice(0, 5).map(item => {
        const displayValue = formatConfigValue(item.value);
        return '<span class="config-kv"><span>' + escapeHtml(item.label) + '</span><strong>' + escapeHtml(displayValue) + '</strong></span>';
    }).join('');
}

function formatConfigValue(value) {
    const num = parseFloat(value);
    if (!Number.isNaN(num) && Number.isFinite(num)) {
        const rounded = Math.round(num * 100) / 100;
        return String(rounded);
    }
    return String(value);
}

function renderConfigItems(id, items, emptyText) {
    const el = document.getElementById(id);
    if (!el) return;
    if (!items || !items.length) {
        // eslint-disable-next-line no-unsanitized/property -- emptyText is always a hardcoded string literal at every call site
        el.innerHTML = '<span class="config-empty">' + emptyText + '</span>';
        return;
    }
    // eslint-disable-next-line no-unsanitized/property -- all interpolated values wrapped in escapeHtml()
    el.innerHTML = items.map(item => {
        const displayValue = formatConfigValue(item.value);
        return '<span class="config-kv"><span>' + escapeHtml(item.label) + '</span><strong>' + escapeHtml(displayValue) + '</strong></span>';
    }).join('');
}


function renderSlotGrid(l, hasActiveEndpoint) {
    const grid = document.getElementById('m-slot-grid');
    if (!grid) return;
    if (!hasActiveEndpoint || !l) {
        grid.innerHTML = '<div class="slot-tile idle"><div class="slot-tile-top"><span>slots</span><strong>waiting</strong></div><div class="slot-tile-task">attach endpoint</div><div class="slot-tile-meta"><span>0 output</span><span>ctx unknown</span></div></div>';
        return;
    }
    const slotSnapshots = Array.isArray(l.slots) ? l.slots : [];
    if (slotSnapshots.length > 0) {
        // server strings (slot.id, task id) are wrapped in escapeHtml(); busy/idle are hardcoded
        // eslint-disable-next-line no-unsanitized/property -- DOMPurify sanitizes HTML
        grid.innerHTML = window.DOMPurify.sanitize(slotSnapshots.map(slot => {
            const busy = !!slot.is_processing;
            const task = busy && slot.id_task !== null && slot.id_task !== undefined ? 'task ' + slot.id_task : 'idle';
            const output = slot.output_available ? formatMetricNumber(slot.output_tokens || 0) + ' output' : 'output unknown';
            const ctx = slot.n_ctx > 0 ? formatMetricNumber(slot.n_ctx) + ' ctx' : 'ctx unknown';
            return '<div class="slot-tile ' + (busy ? 'busy' : 'idle') + '">' +
                '<div class="slot-tile-top"><span>slot ' + escapeHtml(slot.id ?? '?') + '</span><strong>' + (busy ? 'active' : 'idle') + '</strong></div>' +
                '<div class="slot-tile-task">' + escapeHtml(task) + '</div>' +
                '<div class="slot-tile-meta"><span>' + output + '</span><span>' + ctx + '</span></div>' +
            '</div>';
        }).join(''));
        return;
    }
    const processing = l?.slots_processing || 0;
    const idle = l?.slots_idle || 0;
    const total = Math.max(1, processing + idle);
    const activeTask = l?.active_task_id;
    const capacity = l?.context_capacity_tokens || l?.kv_cache_max || 0;
    const generated = l?.slot_generation_tokens || 0;
    const tiles = [];

    for (let index = 0; index < total; index += 1) {
        const busy = index < processing;
        const task = busy && activeTask !== null && activeTask !== undefined ? 'task ' + activeTask : 'idle';
        const output = busy || total === 1 ? formatMetricNumber(generated) + ' output' : '0 output';
        const slotCapacity = capacity > 0 ? Math.round(capacity / total) : 0;
        const ctx = slotCapacity > 0 ? formatMetricNumber(slotCapacity) + ' ctx' : 'ctx unknown';
        tiles.push(
            '<div class="slot-tile ' + (busy ? 'busy' : 'idle') + '">' +
                '<div class="slot-tile-top"><span>slot ' + index + '</span><strong>' + (busy ? 'active' : 'idle') + '</strong></div>' +
                '<div class="slot-tile-task">' + task + '</div>' +
                '<div class="slot-tile-meta"><span>' + output + '</span><span>' + ctx + '</span></div>' +
            '</div>'
        );
    }

    // tiles built from numeric index, numeric formatMetricNumber output, and hardcoded 'busy'/'idle'/'active' strings
    // eslint-disable-next-line no-unsanitized/property -- DOMPurify sanitizes HTML
    grid.innerHTML = window.DOMPurify.sanitize(tiles.join(''));
}

function getPrimarySlot(l) {
    const slots = Array.isArray(l?.slots) ? l.slots : [];
    return slots.find(slot => slot.is_processing) || slots[0] || null;
}

function renderSlotUtilization(l) {
    const utilBar = document.getElementById('m-slot-util-bar');
    const utilValue = document.getElementById('m-slot-util');
    if (!l || l.slots_processing === undefined || l.slots_idle === undefined) {
        if (utilBar) utilBar.style.transform = 'scaleX(0)';
        if (utilValue) utilValue.textContent = '\u2014';
        return;
    }
    const processing = l.slots_processing || 0;
    const idle = l.slots_idle || 0;
    const total = processing + idle;
    if (total === 0) {
        if (utilBar) utilBar.style.transform = 'scaleX(0)';
        if (utilValue) utilValue.textContent = '\u2014';
        return;
    }
    const utilPct = Math.round((processing / total) * 100);
    if (utilBar) utilBar.style.transform = 'scaleX(' + (utilPct / 100) + ')';
    if (utilValue) utilValue.textContent = utilPct + '%';
}

function renderBatchEfficiency(l) {
    const container = document.getElementById('m-slot-batch-efficiency');
    const valueEl = document.getElementById('m-busy-slots-per-decode');
    const specContainer = document.getElementById('m-spec-efficiency');
    const specValueEl = document.getElementById('m-tokens-per-decode');

    if (container && valueEl) {
        const total = (l?.slots_processing || 0) + (l?.slots_idle || 0);
        const value = l?.n_busy_slots_per_decode;
        // Only meaningful with multi-slot servers (>1 slot configured)
        if (!value || value <= 0 || total <= 1) {
            container.style.display = 'none';
        } else {
            container.style.display = '';
            valueEl.textContent = value.toFixed(2);
        }
    }

    if (specContainer && specValueEl) {
        const tpd = l?.tokens_per_decode ?? 0;
        if (tpd > 1.05) {
            specContainer.style.display = '';
            specValueEl.textContent = tpd.toFixed(2) + '×';
        } else {
            specContainer.style.display = 'none';
        }
    }
}

function renderRequestStats() {
    const reqCount = document.getElementById('m-req-count');
    const reqAvg = document.getElementById('m-req-avg');
    const now = Date.now();
    const windowMs = 10 * 60 * 1000;
    const segments = requestActivity.filter(s => (s.endedAtMs || now) >= now - windowMs);
    const completed = segments.filter(s => s.endedAtMs);
    if (reqCount) reqCount.textContent = formatMetricNumber(completed.length);
    if (reqAvg && completed.length > 0) {
        const totalDuration = completed.reduce((sum, s) => sum + (s.endedAtMs - s.startedAtMs), 0);
        const avgDuration = totalDuration / completed.length;
        reqAvg.textContent = formatDuration(avgDuration);
    } else if (reqAvg) {
        reqAvg.textContent = '\u2014';
    }
}

function renderGenerationDetailItems(el, parts) {
    if (!el) return;
    // eslint-disable-next-line no-unsanitized/property -- all parts wrapped in escapeHtml()
    el.innerHTML = parts
        .filter(Boolean)
        .map(part => '<span class="generation-detail-chip">' + escapeHtml(part) + '</span>')
        .join('');
}

function primarySpeculativeType(specType) {
    if (!specType) return '';
    const parts = String(specType)
        .split(',')
        .map(part => part.trim())
        .filter(part => part && part !== 'none');
    return parts[0] || '';
}

function renderDecodingConfig(l, hasActiveEndpoint, isGenerating) {
    const slot = getPrimarySlot(l);
    const specChip = document.getElementById('m-speculative-chip');
    const decodingState = document.getElementById('m-decoding-state');
    const modelInfoRow = document.getElementById('model-info-row');
    const hasConfig = !!slot && ((slot.sampler_stack || []).length > 0 || (slot.speculative_config || []).length > 0);

    setChipState(decodingState, hasConfig ? 'config' : 'waiting', hasConfig ? 'live' : 'idle');

    // Model info row
    if (modelInfoRow) {
        const modelName = l?.model_name || '';
        const modelParams = l?.model_params || null;
        const tpd = l?.tokens_per_decode ?? 0;
        if (modelName) {
            const parts = [escapeHtml(modelName)];
            if (modelParams) {
                parts.push(escapeHtml(formatParamCount(modelParams)));
            }
            const stateClass = isGenerating ? 'generating' : 'idle';
            const decodePill = tpd > 1.05
                ? '<span class="model-info-pill">' + escapeHtml(tpd.toFixed(2) + '× tok/decode') + '</span>'
                : '';
            // eslint-disable-next-line no-unsanitized/property -- stateClass is hardcoded enum; dynamic text is wrapped in escapeHtml()
            modelInfoRow.innerHTML =
                '<span class="model-info-text ' + stateClass + '">' + parts.join(' · ') + '</span>' + decodePill;
        } else {
            modelInfoRow.innerHTML = '';
        }
    }

    if (!hasActiveEndpoint || !slot) {
        if (specChip) specChip.textContent = 'Attach an endpoint for decoding config';
        renderConfigItems('m-speculative-config', [], 'Speculative config unavailable');
        renderSamplerParamsInline(null);
        return;
    }

    if (specChip) {
        const specType = primarySpeculativeType(slot.speculative_type || '');
        const nMax = (slot.speculative_config || []).find(item => item.label === 'n_max');
        if (slot.speculative_enabled || (slot.speculative_config || []).length > 0) {
            const parts = ['Speculative'];
            if (specType) parts.push(specType);
            if (nMax) parts.push('n_max ' + nMax.value);
            const acceptance = l?.speculative_acceptance_rate;
            if (typeof acceptance === 'number' && Number.isFinite(acceptance)) {
                parts.push((acceptance * 100).toFixed(1) + '% accepted');
            }
            specChip.textContent = parts.join(' · ');
            specChip.classList.add('enabled');
        } else {
            specChip.textContent = 'Speculative decoding not enabled';
            specChip.classList.remove('enabled');
        }
    }

    renderConfigItems('m-speculative-config', slot.speculative_config || [], 'Configuration only appears when exposed');

    renderSamplerParamsInline(slot);
}

function formatParamCount(params) {
    if (!params || params === 0) return '';
    if (params >= 1_000_000_000_000) {
        return (params / 1_000_000_000_000).toFixed(0) + 'T params';
    }
    if (params >= 1_000_000_000) {
        return (params / 1_000_000_000).toFixed(0) + 'B params';
    }
    if (params >= 1_000_000) {
        return (params / 1_000_000).toFixed(0) + 'M params';
    }
    return params + ' params';
}

function chatDerivedContextAvailable() {
    const tabs = chat.tabs || [];
    return tabs.length > 0 && tabs.some(t => t.messageCount > 0);
}

function renderCapabilityPopover(d, l, generationAvailable, contextLiveAvailable) {
    const popover = document.getElementById('capability-popover');
    if (!popover) return;

    // No-op when called with no explicit data and no live wsData — preserves existing content
    if (d === undefined && l === undefined && !wsData) return;
    const data = d || wsData || {};
    const lama = l || (wsData ? wsData.llama : null);
    const genAvail = generationAvailable !== undefined ? generationAvailable : (lama ? !!lama.slots_processing : false);
    const ctxLive = contextLiveAvailable !== undefined ? contextLiveAvailable : (lama ? !!lama.context_live_tokens_available : false);

    const hasInference = !!data.capabilities?.inference;
    const slotsAvailable = !!lama && ((lama.slots_processing || 0) + (lama.slots_idle || 0) > 0);
    const metricsAvailable = !!lama && (
        (lama.prompt_tokens_total || 0) > 0 ||
        (lama.generation_tokens_total || 0) > 0 ||
        (lama.context_high_water_tokens || 0) > 0 ||
        (lama.last_generation_throughput_unix_ms || 0) > 0 ||
        (lama.last_prompt_throughput_unix_ms || 0) > 0
    );
    const rows = [
        ['Inference', hasInference ? 'live' : 'unavailable', hasInference],
        ['Slots', slotsAvailable ? 'live' : 'waiting', slotsAvailable],
        ['Metrics', metricsAvailable ? 'live' : 'waiting', metricsAvailable],
        ['Generation progress', genAvail ? 'live' : 'not exposed', genAvail],
        ['Throughput', metricsAvailable ? 'retained avg + live estimate' : 'waiting', metricsAvailable],
        ['Context capacity', (lama?.context_capacity_tokens || 0) > 0 ? 'live' : 'waiting', (lama?.context_capacity_tokens || 0) > 0],
        ['Context usage', ctxLive ? 'live' : chatDerivedContextAvailable() ? 'derived from chat' : 'not exposed', ctxLive || chatDerivedContextAvailable()],
        ['Host metrics', data.host_metrics_available ? 'live' : 'unavailable', !!data.host_metrics_available],
        ['Remote agent', data.remote_agent_connected ? 'connected' : 'disconnected', !!data.remote_agent_connected]
    ];

    if (!data.active_session_id && !hasInference && !slotsAvailable) {
        popover.innerHTML = '<div class="capability-empty">No endpoint attached</div>';
        return;
    }

    // eslint-disable-next-line no-unsanitized/property -- label and value are all hardcoded string literals from the rows array above; ok is boolean
    popover.innerHTML = '<div class="capability-title">Telemetry sources</div>' + rows.map(([label, value, ok]) => {
        const tone = ok ? 'live' : (value === 'waiting' ? 'waiting' : 'absent');
        return '<span class="capability-row"><span class="capability-led ' + (ok ? 'ok' : 'muted') + '"></span><span class="capability-label">' + label + '</span><strong class="capability-val" data-tone="' + tone + '">' + value + '</strong></span>';
    }).join('');
}

function updateMetricDelta(el, previous, current, decimals = 1) {
    if (!el || !Number.isFinite(previous) || previous <= 0 || !Number.isFinite(current)) return;
    const delta = current - previous;
    if (Math.abs(delta) < 0.05) return;
    el.textContent = (delta > 0 ? '+' : '') + delta.toFixed(decimals);
    el.className = 'metric-delta ' + (delta > 0 ? 'positive' : 'negative') + ' show';
    window.clearTimeout(el._hideTimer);
    el._hideTimer = window.setTimeout(() => {
        el.classList.remove('show');
    }, 900);
}

function setEmptyState(el, show) {
    if (!el) return;
    el.classList.toggle('visible', !!show);
}

function hardwareEmptyStateCopy(kind, grade) {
    const action = gradeActionCopy(grade);
    if (grade === 'remote_agent_connecting') {
        return `${kind} telemetry is coming online. ${action}`;
    }
    if (grade === 'remote_inference_only') {
        return `${kind} telemetry requires the remote agent. ${action}`;
    }
    if (grade === 'remote_agent_firewall_blocked') {
        return `${kind} telemetry is blocked even though the agent started. ${action}`;
    }
    if (grade === 'remote_partial_sensors') {
        return `${kind} telemetry is only partially available. ${action}`;
    }
    if (grade === 'remote_agent_degraded' || grade === 'remote_agent_update_available') {
        return `${kind} telemetry is limited by agent compatibility. ${action}`;
    }
    if (grade === 'remote_error') {
        return `${kind} telemetry is unavailable because the remote agent failed. ${action}`;
    }
    return `${kind} metrics appear after attach`;
}

function getMetricTone(kind) {
    const load   = getCSSVar('--metric-load',   '#34d399');
    const power  = getCSSVar('--metric-power',  '#2dd4bf');
    const mem    = getCSSVar('--metric-memory', '#22d3ee');
    const clock  = getCSSVar('--metric-clock',  '#60a5fa');
    const accent = getCSSVar('--color-accent',  '#2dd4bf');
    const light  = getCSSVar('--color-accent-light', '#5eead4');
    switch (kind) {
    case 'load':   return { start: load,  end: accent, line: load };
    case 'power':  return { start: power, end: mem,    line: power };
    case 'memory': return { start: mem,   end: light,  line: mem };
    case 'clock':  return { start: clock, end: light,  line: clock };
    default:       return { start: load,  end: light,  line: accent };
    }
}

function getClockTone(kind) {
    const clock  = getCSSVar('--metric-clock',  '#60a5fa');
    const accent = getCSSVar('--color-accent',  '#2dd4bf');
    const light  = getCSSVar('--color-accent-light', '#5eead4');
    switch (kind) {
    case 'memory': return { start: clock,  end: light, line: clock };
    default:       return { start: accent, end: light, line: accent };
    }
}

function getTempSeverityColor(temp) {
    if (temp >= 90) return getCSSVar('--color-error',   '#f43f5e');
    if (temp >= 75) return getCSSVar('--color-warning', '#f59e0b');
    return getCSSVar('--color-success', '#10b981');
}

function setVizContent(container, html) {
    if (!container) return;
    // html always built internally from numeric values, hardcoded CSS class names, and getSeverityColor() hex strings
    // eslint-disable-next-line no-unsanitized/property -- DOMPurify sanitizes HTML
    container.innerHTML = window.DOMPurify.sanitize(html);
}

function renderHwBar(container, pct, tone, isAlert) {
    if (!container) return;
    const bgCls = isAlert ? 'hw-bar-bg is-hot' : 'hw-bar-bg';
    const scale = (pct / 100).toFixed(4);
    setVizContent(container,
        '<div class="' + bgCls + '" style="--pct:' + pct.toFixed(1) + '%;--bar-start:' + tone.start + ';--bar-end:' + tone.end + ';">' +
          '<div class="hw-bar-fill" style="transform:scaleX(' + scale + ');--bar-start:' + tone.start + ';--bar-end:' + tone.end + '"></div>' +
          '<div class="hw-bar-cap"></div>' +
        '</div>');
}

export function buildRingMarkup(pct, colorHex, isAlert) {
    const cls = isAlert ? 'hw-ring-viz is-warming' : 'hw-ring-viz';
    return '<div class="' + cls + '" style="--pct:' + pct.toFixed(1) + ';--gauge-color:' + colorHex + '"></div>';
}

function renderHwRing(container, pct, tone, isAlert) {
    if (!container) return;
    setVizContent(container, buildRingMarkup(pct, tone.line, isAlert));
}

function renderHwSparkline(container, history, color) {
    if (!container || !history || history.length < 2) {
        setVizContent(container, '');
        return;
    }
    const svg = buildSparklineSVG(history, 'hw-sparkline', color || '#8fbcbb');
    setVizContent(container, svg);
}

/** Last hardware sparkline data snapshot for change detection */
var lastHwSparklineSnapshots = {};

function renderHwMetricSparkline(svgId, history, color, show) {
    const svg = document.getElementById(svgId);
    if (!svg) return;
    if (!show || !history || history.length < 2) {
        svg.style.visibility = (show && history && history.length >= 2) ? '' : 'hidden';
        return;
    }

    // POWER OPT: skip rebuild if data hasn't changed
    const lastHwSnap = lastHwSparklineSnapshots[svgId];
    if (lastHwSnap && lastHwSnap.length === history.length && lastHwSnap[lastHwSnap.length - 1] === history[history.length - 1]) {
        return;
    }
    lastHwSparklineSnapshots[svgId] = [...history];

    svg.style.visibility = '';
    svg.style.color = color;
    const width = 120;
    const height = 28;
    const plotLeft = 5;
    const plotRight = width - 5;
    const plotTop = 5;
    const plotBottom = height - 5;
    const fillBottom = height - 2;
    const max = Math.max(...history, 1);
    const min = Math.min(...history, 0);
    const range = Math.max(max - min, 1);
    const step = (plotRight - plotLeft) / (history.length - 1);
    const currentValue = history[history.length - 1];
    const currentX = plotRight;
    const currentY = plotBottom - (((currentValue - min) / range) * (plotBottom - plotTop));
    const path = history.map((value, index) => {
        const x = plotLeft + (index * step);
        const y = plotBottom - (((value - min) / range) * (plotBottom - plotTop));
        return (index === 0 ? 'M' : 'L') + x.toFixed(2) + ' ' + y.toFixed(2);
    }).join(' ');
    var ratio = range > 0 ? (currentValue - min) / range : 0;
    var fillColor = getThemedSparklineFillColor(color, ratio);
    var fillId = nextSparklineGradientId(svgId);
    // eslint-disable-next-line no-unsanitized/property -- SVG path from numeric values; svgId/color from getSeverityColor()
    svg.innerHTML =
        buildSparklineFillDefs(fillId, fillColor, 0.82, 0.14, 0.02) +
        '<path class="sparkline-fill" d="' + path + ' L ' + plotRight + ' ' + fillBottom + ' L ' + plotLeft + ' ' + fillBottom + ' Z" fill="url(#' + fillId + ')"></path>' +
        '<line class="sparkline-grid" x1="' + plotLeft + '" y1="7" x2="' + plotRight + '" y2="7"/><line class="sparkline-grid" x1="' + plotLeft + '" y1="14" x2="' + plotRight + '" y2="14"/><line class="sparkline-grid" x1="' + plotLeft + '" y1="21" x2="' + plotRight + '" y2="21"/>' +
        '<path class="sparkline-line" d="' + path + '" stroke="' + color + '" fill="none" stroke-width="2.3" vector-effect="non-scaling-stroke" stroke-linecap="round" stroke-linejoin="round" filter="drop-shadow(0 0 2.5px ' + color + ')"></path>' +
        '<circle class="sparkline-current-halo" cx="' + currentX.toFixed(2) + '" cy="' + currentY.toFixed(2) + '" r="4.2"></circle>' +
        '<circle class="sparkline-current" cx="' + currentX.toFixed(2) + '" cy="' + currentY.toFixed(2) + '" r="2.1"></circle>' +
        '<circle class="sparkline-current-core" cx="' + currentX.toFixed(2) + '" cy="' + currentY.toFixed(2) + '" r="0.9"></circle>';
}

function renderHwStacked(container, pct, tone, isAlert) {
    if (!container) return;
    const bgCls = isAlert ? 'hw-stacked-bg is-hot' : 'hw-stacked-bg';
    const scale = (pct / 100).toFixed(4);
    setVizContent(container,
        '<div class="' + bgCls + '" style="--pct:' + pct.toFixed(1) + '%;--bar-start:' + tone.start + ';--bar-end:' + tone.end + ';">' +
          '<div class="hw-stacked-fill" style="transform:scaleX(' + scale + ');--bar-start:' + tone.start + ';--bar-end:' + tone.end + '"></div>' +
          '<div class="hw-stacked-free"></div>' +
          '<div class="hw-bar-cap"></div>' +
        '</div>');
}

function renderHwChips(container, chips) {
    if (!container) return;
    setVizContent(container, '<div class="hw-chips">' + chips.map(function(c) { return '<span class="hw-chip">' + c + '</span>'; }).join('') + '</div>');
}

function computeClockBand(history, current) {
    var points = (history || []).filter(Number.isFinite);
    if (Number.isFinite(current) && current > 0) points.push(current);
    if (points.length === 0) {
        return { min: 0, max: 0, pct: 0, peakPct: 0, lowPct: 0 };
    }
    var min = Math.min.apply(null, points);
    var max = Math.max.apply(null, points);
    var span = Math.max(max - min, 1);
    var normalized = function(value) {
        if (!Number.isFinite(value)) return 0;
        return Math.max(0, Math.min(100, ((value - min) / span) * 100));
    };
    var pct = span <= 1 ? 100 : normalized(current);
    return {
        min: min,
        max: max,
        pct: pct,
        peakPct: normalized(max),
        lowPct: normalized(min)
    };
}

function renderHwDualRing(container, sclk, mclk) {
    if (!container) return;
    var sclkBand = computeClockBand(gpuHistory.sclk, sclk);
    var mclkBand = computeClockBand(gpuHistory.mclk, mclk);
    var sclkColor = getClockTone('core').line;
    var mclkColor = getClockTone('memory').line;
    var sclkPulse = (3.4 - Math.min(sclkBand.pct, 100) * 0.014).toFixed(2) + 's';
    var mclkPulse = (3.8 - Math.min(mclkBand.pct, 100) * 0.016).toFixed(2) + 's';
    setVizContent(container,
        '<div class="hw-clock-gpu-layout">' +
          '<div class="hw-clock-cluster hw-clock-gpu" style="--dot-radius:-53px;">' +
            '<div class="hw-clock-orbit outer" style="--pct:' + mclkBand.pct.toFixed(1) + ';--peak-pct:' + mclkBand.peakPct.toFixed(1) + ';--low-pct:' + mclkBand.lowPct.toFixed(1) + ';--orbit-color:' + mclkColor + ';--dot-radius:-53px;--pulse-duration:' + mclkPulse + ';">' +
              '<div class="hw-clock-orbit-track"></div>' +
              '<div class="hw-clock-orbit-fill"></div>' +
              '<div class="hw-clock-orbit-peak"></div>' +
              '<div class="hw-clock-orbit-low"></div>' +
              '<div class="hw-clock-orbit-dot"></div>' +
            '</div>' +
            '<div class="hw-clock-orbit inner" style="--pct:' + sclkBand.pct.toFixed(1) + ';--peak-pct:' + sclkBand.peakPct.toFixed(1) + ';--low-pct:' + sclkBand.lowPct.toFixed(1) + ';--orbit-color:' + sclkColor + ';--dot-radius:-33px;--pulse-duration:' + sclkPulse + ';">' +
              '<div class="hw-clock-orbit-track"></div>' +
              '<div class="hw-clock-orbit-fill"></div>' +
              '<div class="hw-clock-orbit-peak"></div>' +
              '<div class="hw-clock-orbit-low"></div>' +
              '<div class="hw-clock-orbit-dot"></div>' +
            '</div>' +
          '</div>' +
          '<div class="hw-clock-gpu-legend">' +
            '<div class="hw-clock-legend-item">' +
              '<span class="hw-clock-legend-swatch sclk"></span>' +
              '<span class="hw-clock-legend-text">Core clock</span>' +
            '</div>' +
            '<div class="hw-clock-legend-item">' +
              '<span class="hw-clock-legend-swatch mclk"></span>' +
              '<span class="hw-clock-legend-text">Memory clock</span>' +
            '</div>' +
          '</div>' +
          '<div class="hw-clock-gpu-readout">' +
            '<div class="hw-clock-meter" style="--meter-color:' + sclkColor + ';">' +
              '<div class="hw-clock-meter-label">SCLK</div>' +
              '<div class="hw-clock-meter-bar" style="--pct:' + sclkBand.pct.toFixed(1) + ';--peak-pct:' + sclkBand.peakPct.toFixed(1) + ';--low-pct:' + sclkBand.lowPct.toFixed(1) + ';">' +
                '<div class="hw-clock-meter-fill" style="--pct:' + sclkBand.pct.toFixed(1) + ';--meter-color:' + sclkColor + ';--pulse-duration:' + sclkPulse + ';"></div>' +
                '<div class="hw-clock-meter-marker"></div>' +
                '<div class="hw-clock-meter-marker-low"></div>' +
              '</div>' +
              '<div class="hw-clock-meter-value">' + formatClockReadout(sclk).value + ' ' + formatClockReadout(sclk).unit + '</div>' +
              '<div class="hw-clock-meter-band">' + formatClockReadout(sclkBand.min).value + '-' + formatClockReadout(sclkBand.max).value + ' ' + formatClockReadout(sclkBand.max).unit + '</div>' +
              (gpuHistory.sclk.length > 1 ? '<div class="hw-clock-footer-spark">' + buildSparklineSVG(gpuHistory.sclk, 'hw-clock-footer-spark', sclkColor) + '</div>' : '') +
            '</div>' +
            '<div class="hw-clock-meter" style="--meter-color:' + mclkColor + ';">' +
              '<div class="hw-clock-meter-label">MCLK</div>' +
              '<div class="hw-clock-meter-bar" style="--pct:' + mclkBand.pct.toFixed(1) + ';--peak-pct:' + mclkBand.peakPct.toFixed(1) + ';--low-pct:' + mclkBand.lowPct.toFixed(1) + ';">' +
                '<div class="hw-clock-meter-fill" style="--pct:' + mclkBand.pct.toFixed(1) + ';--meter-color:' + mclkColor + ';--pulse-duration:' + mclkPulse + ';"></div>' +
                '<div class="hw-clock-meter-marker"></div>' +
                '<div class="hw-clock-meter-marker-low"></div>' +
              '</div>' +
              '<div class="hw-clock-meter-value">' + formatClockReadout(mclk).value + ' ' + formatClockReadout(mclk).unit + '</div>' +
              '<div class="hw-clock-meter-band">' + formatClockReadout(mclkBand.min).value + '-' + formatClockReadout(mclkBand.max).value + ' ' + formatClockReadout(mclkBand.max).unit + '</div>' +
              (gpuHistory.mclk.length > 1 ? '<div class="hw-clock-footer-spark">' + buildSparklineSVG(gpuHistory.mclk, 'hw-clock-footer-spark', mclkColor) + '</div>' : '') +
            '</div>' +
          '</div>' +
        '</div>');
}

function renderHwClockRing(container, clock) {
    if (!container) return;
    var band = computeClockBand(sysHistory.cpuClock, clock);
    var display = formatClockReadout(clock);
    var color = getClockTone('core').line;
    var pulse = (3.6 - Math.min(band.pct, 100) * 0.016).toFixed(2) + 's';
    var footerSpark = sysHistory.cpuClock.length > 1
        ? '<div class="hw-clock-footer sparkline-only"><div class="hw-clock-footer-spark">' + buildSparklineSVG(sysHistory.cpuClock, 'hw-clock-footer-spark', color) + '</div></div>'
        : '';
    setVizContent(container,
        '<div class="hw-clock-system-layout">' +
          '<div class="hw-clock-cluster hw-clock-system">' +
            '<div class="hw-clock-orbit outer" style="--pct:' + band.pct.toFixed(1) + ';--peak-pct:' + band.peakPct.toFixed(1) + ';--low-pct:' + band.lowPct.toFixed(1) + ';--orbit-color:' + color + ';--dot-radius:-61px;--pulse-duration:' + pulse + ';">' +
              '<div class="hw-clock-orbit-track"></div>' +
              '<div class="hw-clock-orbit-fill"></div>' +
              '<div class="hw-clock-orbit-peak"></div>' +
              '<div class="hw-clock-orbit-low"></div>' +
              '<div class="hw-clock-orbit-dot"></div>' +
            '</div>' +
            '<div class="hw-clock-core hw-clock-system-core">' +
              '<div class="hw-clock-row-value">' + display.value + '</div>' +
              '<div class="hw-clock-unit">' + display.unit + '</div>' +
            '</div>' +
          '</div>' +
          '<div class="hw-clock-system-readout">' +
            '<div class="hw-clock-meter">' +
              '<div class="hw-clock-meter-label">CPU CLOCK</div>' +
              '<div class="hw-clock-meter-bar" style="--pct:' + band.pct.toFixed(1) + ';--peak-pct:' + band.peakPct.toFixed(1) + ';--low-pct:' + band.lowPct.toFixed(1) + ';">' +
                '<div class="hw-clock-meter-fill" style="--pct:' + band.pct.toFixed(1) + ';--meter-color:' + color + ';--pulse-duration:' + pulse + ';"></div>' +
                '<div class="hw-clock-meter-marker"></div>' +
                '<div class="hw-clock-meter-marker-low"></div>' +
              '</div>' +
            '</div>' +
            '<div class="hw-clock-system-meta">' +
              '<div class="hw-clock-meter-value">' + formatClockReadout(clock).value + ' ' + formatClockReadout(clock).unit + '</div>' +
              '<div class="hw-clock-meter-band">Range ' + formatClockReadout(band.min).value + '-' + formatClockReadout(band.max).value + ' ' + formatClockReadout(band.max).unit + '</div>' +
            '</div>' +
            (sysHistory.cpuClock.length > 1 ? '<div class="hw-clock-footer-spark hw-clock-system-spark">' + buildSparklineSVG(sysHistory.cpuClock, 'hw-clock-footer-spark', color) + '</div>' : '') +
          '</div>' +
        '</div>');
}

export function buildSparklineSVG(points, cssClass, color) {
    var len = points.length;
    if (len < 2) return '';
    var w = 120, h = 24, pad = 5;
    var plotLeft = 5, plotRight = w - 5;
    var plotTop = pad, plotBottom = h - pad;
    var max = Math.max.apply(null, points);
    var min = Math.min.apply(null, points);
    var range = max - min || 1;
    var step = (plotRight - plotLeft) / (len - 1);
    var pts = points.map(function(v, i) {
        return (plotLeft + i * step) + ',' + (plotBottom - ((v - min) / range) * (plotBottom - plotTop));
    });
    var linePath = 'M' + pts.join(' L');
    var fillPath = linePath + ' L' + plotRight + ',' + (h - 2) + ' L' + plotLeft + ',' + (h - 2) + ' Z';
    var currentVal = points[len - 1];
    var currentX = plotRight;
    var currentY = plotBottom - ((currentVal - min) / range) * (plotBottom - plotTop);
    var ratio = range > 0 ? (currentVal - min) / range : 0;
    var fillColor = getThemedSparklineFillColor(color, ratio);
    var fillId = nextSparklineGradientId(cssClass);
    return '<svg class="metric-sparkline ' + cssClass + '" viewBox="0 0 ' + w + ' ' + h + '" preserveAspectRatio="none" aria-hidden="true" style="color:' + color + ';">' +
        buildSparklineFillDefs(fillId, fillColor, 0.80, 0.12, 0.02) +
        '<path class="sparkline-fill" d="' + fillPath + '" fill="url(#' + fillId + ')"/>' +
        '<line class="sparkline-grid" x1="' + plotLeft + '" y1="6" x2="' + plotRight + '" y2="6"/><line class="sparkline-grid" x1="' + plotLeft + '" y1="12" x2="' + plotRight + '" y2="12"/><line class="sparkline-grid" x1="' + plotLeft + '" y1="18" x2="' + plotRight + '" y2="18"/>' +
        '<path class="sparkline-line" d="' + linePath + '" fill="none" stroke="' + color + '" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>' +
        '<line class="sparkline-current-trace" x1="' + Math.max(currentX - 10, plotLeft).toFixed(1) + '" y1="' + currentY.toFixed(1) + '" x2="' + currentX.toFixed(1) + '" y2="' + currentY.toFixed(1) + '" stroke="' + color + '"></line>' +
        '<circle class="sparkline-current-halo" cx="' + currentX.toFixed(1) + '" cy="' + currentY.toFixed(1) + '" r="3.4"></circle>' +
        '<circle class="sparkline-current" cx="' + currentX.toFixed(1) + '" cy="' + currentY.toFixed(1) + '" r="2.0"></circle>' +
        '<circle class="sparkline-current-core" cx="' + currentX.toFixed(1) + '" cy="' + currentY.toFixed(1) + '" r="0.9"></circle>' +
        '</svg>';
}

/** Last GPU metric snapshot for change detection — skip full card rebuild when metrics are stable */
var lastGpuSnapshot = null;

var gpuHistory = { load: [], power: [], vramPct: [], sclk: [], mclk: [] };
function pushGpuHistory(key, value) {
    if (!Number.isFinite(value)) return;
    gpuHistory[key].push(value);
    var limit = key === 'load' || key === 'power' || key === 'vramPct' ? 60 : 30;
    if (gpuHistory[key].length > limit) gpuHistory[key].shift();
}

var sysHistory = { cpuLoad: [], ramPct: [], memoryPressure: [], cpuClock: [], power: [], pCluster: [], sCluster: [], eCluster: [] };
function pushSysHistory(key, value) {
    if (!Number.isFinite(value)) return;
    sysHistory[key].push(value);
    var limit = 60;
    if (sysHistory[key].length > limit) sysHistory[key].shift();
}

var vizPrefs = {
    gpu: { load: 'bar', power: 'bar', vram: 'bar', clocks: 'ring' },
    system: { load: 'bar', ram: 'bar', clock: 'ring', power: 'bar' }
};


function loadVizPrefs() {
    try {
        var gpuStr = localStorage.getItem('llama-monitor-gpu-viz');
        if (gpuStr) vizPrefs.gpu = JSON.parse(gpuStr);
        var sysStr = localStorage.getItem('llama-monitor-system-viz');
        if (sysStr) vizPrefs.system = JSON.parse(sysStr);
    } catch(e) {}
}

function saveVizPrefs(card) {
    try {
        var key = card === 'gpu' ? 'llama-monitor-gpu-viz' : 'llama-monitor-system-viz';
        localStorage.setItem(key, JSON.stringify(vizPrefs[card]));
    } catch(e) {}
}

function toggleVizSwitcher(card) {
    var prefix = card === 'gpu' ? 'gpu' : 'sys';
    var sw = document.getElementById(prefix + '-viz-switcher');
    if (!sw) return;
    var isOpen = sw.style.display !== 'none';
    // Close all switchers
    document.querySelectorAll('.viz-switcher').forEach(function(el) { el.style.display = 'none'; });
    if (!isOpen) {
        sw.style.display = 'flex';
        // Position relative to card
        var cardEl = document.getElementById(card === 'gpu' ? 'gpu-card' : 'system-card');
        if (cardEl) {
            var rect = cardEl.getBoundingClientRect();
            var parentRect = cardEl.parentElement.getBoundingClientRect();
            sw.style.top = (rect.height + 8) + 'px';
            sw.style.right = '0';
        }
    }
}

function selectVizStyle(card, metric, style) {
    vizPrefs[card][metric] = style;
    saveVizPrefs(card);
    // Update active state in switcher
    var prefix = card === 'gpu' ? 'gpu' : 'sys';
    var sw = document.getElementById(prefix + '-viz-switcher');
    if (sw) {
        sw.querySelectorAll('.viz-option').forEach(function(btn) {
            btn.classList.toggle('active', btn.getAttribute('data-style') === style && btn.closest('.viz-switcher-options').getAttribute('data-metric') === metric);
        });
    }
    // Fade out viz containers, then re-render
    var cardEl = document.getElementById(card === 'gpu' ? 'gpu-card' : 'system-card');
    if (cardEl) {
        cardEl.querySelectorAll('.hw-metric-viz').forEach(function(el) { el.classList.add('viz-fade-out'); });
        setTimeout(function() {
            if (card === 'gpu') renderGpuCard(lastGpuData || {}, !!lastGpuData && Object.keys(lastGpuData).length > 0, window.__telemetryGrade);
            else renderSystemCard(lastSystemMetrics, !!lastSystemMetrics, window.__telemetryGrade);
            cardEl.querySelectorAll('.hw-metric-viz').forEach(function(el) {
                el.classList.remove('viz-fade-out');
                el.classList.add('viz-fade-in');
                setTimeout(function() { el.classList.remove('viz-fade-in'); }, 160);
            });
        }, 120);
    }
}

function resetVizPrefs(card) {
    vizPrefs[card] = card === 'gpu'
        ? { load: 'bar', power: 'bar', vram: 'bar', clocks: 'chips' }
        : { load: 'bar', ram: 'bar', clock: 'chip' };
    saveVizPrefs(card);
    var prefix = card === 'gpu' ? 'gpu' : 'sys';
    var sw = document.getElementById(prefix + '-viz-switcher');
    if (sw) {
        sw.querySelectorAll('.viz-option').forEach(function(btn) {
            btn.classList.remove('active');
        });
        sw.querySelectorAll('.viz-option[data-style="bar"], .viz-option[data-style="chips"], .viz-option[data-style="chip"]').forEach(function(btn) {
            btn.classList.add('active');
        });
    }
    // Fade out viz containers, then re-render
    var cardEl = document.getElementById(card === 'gpu' ? 'gpu-card' : 'system-card');
    if (cardEl) {
        cardEl.querySelectorAll('.hw-metric-viz').forEach(function(el) { el.classList.add('viz-fade-out'); });
        setTimeout(function() {
            if (card === 'gpu') renderGpuCard(lastGpuData || {}, !!lastGpuData && Object.keys(lastGpuData).length > 0, window.__telemetryGrade);
            else renderSystemCard(lastSystemMetrics, !!lastSystemMetrics, window.__telemetryGrade);
            cardEl.querySelectorAll('.hw-metric-viz').forEach(function(el) {
                el.classList.remove('viz-fade-out');
                el.classList.add('viz-fade-in');
                setTimeout(function() { el.classList.remove('viz-fade-in'); }, 160);
            });
        }, 120);
    }
}

function renderGpuCard(gpuMap, visible, grade) {
    var card = document.getElementById('gpu-card');
    var emptyEl = document.getElementById('gpu-empty');
    var deviceName = document.getElementById('gpu-device-name');
    var tempGauge = document.getElementById('gpu-temp-gauge');
    var tempValue = document.getElementById('gpu-temp-value');
    var stateChip = document.getElementById('gpu-state');

    if (!card || !visible) {
        if (card) setCardState(card, 'dormant');
        return;
    }

    var entries = Object.entries(gpuMap);
    if (entries.length === 0) {
        setCardState(card, 'unavailable');
        if (emptyEl) emptyEl.textContent = hardwareEmptyStateCopy('GPU', grade);
        setChipState(stateChip, grade === 'remote_agent_connecting' ? 'connecting' : 'unavailable', 'warning');
        setEmptyState(emptyEl, true);
        return;
    }

    setLastGpuData(gpuMap);
    setEmptyState(emptyEl, false);

    // Use first GPU (most common case)
    var _loop = entries[0];
    // POWER OPT: skip full rebuild when GPU values AND viz prefs haven't changed
    var gpuKey = _loop[0];
    var gpuSnap = JSON.stringify([gpuKey, _loop[1].load, _loop[1].power_consumption, _loop[1].vram_used, _loop[1].temp, _loop[1].sclk_mhz, _loop[1].mclk_mhz, _loop[1].metal_gpu_limit_mb, vizPrefs.gpu.load, vizPrefs.gpu.power, vizPrefs.gpu.vram, vizPrefs.gpu.clocks]);
    if (gpuSnap === lastGpuSnapshot) return;
    lastGpuSnapshot = gpuSnap;
    var name = _loop[0];
    var m = _loop[1];

    setCardState(card, 'live');
    setChipState(stateChip, 'live', 'live');

    // Device name
    if (deviceName) deviceName.textContent = name;

    // Temperature gauge
    var temp = Math.round(m.temp);
    var tempPct = Math.min(100, (temp / 100) * 100);
    var tempColor = getTempSeverityColor(temp);
    var isTempHot = temp >= 75;
    if (tempGauge) {
        tempGauge.style.setProperty('--pct', tempPct.toFixed(1));
        tempGauge.style.setProperty('--gauge-color', tempColor);
        tempGauge.classList.toggle('is-warming', isTempHot);
    }
    if (tempValue) tempValue.textContent = temp + '\u00B0';

    var hasMetalCap = Number(m.metal_gpu_limit_mb || 0) > 0;
    var effectiveVramTotalMb = hasMetalCap ? Number(m.metal_gpu_limit_mb) : m.vram_total;
    var totalUnifiedGb = m.vram_total > 0 ? (m.vram_total / 1024).toFixed(0) : '0';

    // Push history
    pushGpuHistory('load', m.load);
    pushGpuHistory('power', m.power_consumption);
    var vramPct = effectiveVramTotalMb > 0 ? (m.vram_used / effectiveVramTotalMb) * 100 : 0;
    pushGpuHistory('vramPct', vramPct);
    pushGpuHistory('sclk', m.sclk_mhz);
    pushGpuHistory('mclk', m.mclk_mhz);

    // Load
    var loadViz = document.getElementById('gpu-load-viz');
    var loadVal = document.getElementById('gpu-load-value');
    var loadStyle = vizPrefs.gpu.load;
    var loadTone = getMetricTone('load');
    var loadColor = loadTone.line;
    if (loadStyle === 'ring') renderHwRing(loadViz, m.load, loadTone, false);
    else if (loadStyle === 'sparkline') renderHwSparkline(loadViz, gpuHistory.load, loadColor);
    else renderHwBar(loadViz, m.load, loadTone, false);
    renderHwMetricSparkline('gpu-load-spark', gpuHistory.load, loadColor, loadStyle !== 'sparkline');
    if (loadVal) loadVal.textContent = m.load + '%';

    // Power
    var powerViz = document.getElementById('gpu-power-viz');
    var powerVal = document.getElementById('gpu-power-value');
    var powerBlock = document.getElementById('gpu-power-block');
    var powerLabelEl = powerBlock ? powerBlock.querySelector('.hw-metric-label') : null;
    var isAppleUnified = hasMetalCap;
    var powerPct = m.power_limit > 0 ? (m.power_consumption / m.power_limit) * 100 : Math.min(100, (m.power_consumption / 150) * 100);
    var isCapped = m.power_consumption >= m.power_limit && m.power_limit > 0;
    var powerStyle = vizPrefs.gpu.power;
    var powerTone = getMetricTone('power');
    var powerColor = isCapped ? '#f43f5e' : powerTone.line;
    if (powerBlock) powerBlock.classList.toggle('hw-power-capped', isCapped);
    if (powerStyle === 'ring') renderHwRing(powerViz, powerPct, isCapped ? { line: '#f43f5e' } : powerTone, isCapped);
    else if (powerStyle === 'sparkline') renderHwSparkline(powerViz, gpuHistory.power, powerColor);
    else renderHwBar(powerViz, powerPct, isCapped ? { start: '#fb7185', end: '#f43f5e' } : powerTone, isCapped);
    renderHwMetricSparkline('gpu-power-spark', gpuHistory.power, powerColor, powerStyle !== 'sparkline');
    if (powerVal) {
        if (isAppleUnified) {
            powerVal.textContent = m.power_consumption.toFixed(1) + 'W';
            if (powerLabelEl) powerLabelEl.textContent = 'GPU Power';
        } else if (m.power_limit > 0) {
            powerVal.textContent = m.power_consumption.toFixed(1) + 'W' + (isCapped ? '!' : '') + ' / ' + m.power_limit + 'W';
            if (powerLabelEl) powerLabelEl.textContent = 'Power';
        } else {
            powerVal.textContent = m.power_consumption.toFixed(1) + 'W';
            if (powerLabelEl) powerLabelEl.textContent = 'Power';
        }
    }

    // VRAM / Memory
    var vramViz = document.getElementById('gpu-vram-viz');
    var vramVal = document.getElementById('gpu-vram-value');
    var vramBlock = document.getElementById('gpu-vram-block');
    var vramLabelEl = vramBlock ? vramBlock.querySelector('.hw-metric-label') : null;
    var vramStyle = vizPrefs.gpu.vram;
    var vramGb = effectiveVramTotalMb > 0 ? (m.vram_used / 1024).toFixed(1) : '0';
    var vramTotalGb = effectiveVramTotalMb > 0 ? (effectiveVramTotalMb / 1024).toFixed(0) : '0';
    var vramTone = getMetricTone('memory');
    var vramColor = vramTone.line;
    if (vramStyle === 'ring') renderHwRing(vramViz, vramPct, vramTone, false);
    else if (vramStyle === 'sparkline') renderHwSparkline(vramViz, gpuHistory.vramPct, vramColor);
    else if (vramStyle === 'stacked') renderHwStacked(vramViz, vramPct, vramTone, false);
    else renderHwBar(vramViz, vramPct, vramTone, false);
    renderHwMetricSparkline('gpu-vram-spark', gpuHistory.vramPct, vramColor, vramStyle !== 'sparkline');
    if (vramVal) {
        if (isAppleUnified) {
            vramVal.textContent = vramGb + ' / ' + totalUnifiedGb + ' GB';
            vramVal.title = hasMetalCap
                ? 'Metal GPU memory cap ' + vramTotalGb + ' GB (of ' + totalUnifiedGb + ' GB unified memory total)'
                : '';
            if (vramLabelEl) vramLabelEl.textContent = 'Memory';
        } else {
            vramVal.textContent = vramGb + ' / ' + vramTotalGb + ' GB';
            vramVal.title = '';
            if (vramLabelEl) vramLabelEl.textContent = 'VRAM';
        }
    }

    // Clocks
    var clocksViz = document.getElementById('gpu-clocks-viz');
    var clocksVal = document.getElementById('gpu-clocks-value');
    var clocksStyle = vizPrefs.gpu.clocks;
    if (clocksStyle === 'ring') {
        renderHwDualRing(clocksViz, m.sclk_mhz, m.mclk_mhz);
        if (clocksVal) clocksVal.textContent = '';
    } else if (clocksStyle === 'chips') {
        renderHwChips(clocksViz, ['SCLK ' + m.sclk_mhz + 'MHz', 'MCLK ' + m.mclk_mhz + 'MHz']);
        if (clocksVal) clocksVal.textContent = '';
    } else {
        if (clocksViz) clocksViz.innerHTML = '';
        if (clocksVal) clocksVal.textContent = m.sclk_mhz + ' / ' + m.mclk_mhz + ' MHz';
    }
}

function renderSystemCard(sys, visible, grade) {
    var card = document.getElementById('system-card');
    var emptyEl = document.getElementById('sys-empty');
    var deviceName = document.getElementById('sys-device-name');
    var tempGauge = document.getElementById('sys-temp-gauge');
    var tempValue = document.getElementById('sys-temp-value');
    var stateChip = document.getElementById('sys-state');

    if (!card || !visible) {
        if (card) setCardState(card, 'dormant');
        return;
    }

    if (!sys) {
        setCardState(card, 'unavailable');
        if (emptyEl) emptyEl.textContent = hardwareEmptyStateCopy('System', grade);
        setChipState(stateChip, grade === 'remote_agent_connecting' ? 'connecting' : 'unavailable', 'warning');
        setEmptyState(emptyEl, true);
        return;
    }

    setEmptyState(emptyEl, false);
    setCardState(card, 'live');
    setChipState(stateChip, 'live', 'live');

    // Device name: CPU model + motherboard
    var parts = [];
    if (sys.cpu_name) parts.push(sys.cpu_name);
    if (sys.motherboard && sys.motherboard !== 'Unknown Motherboard') parts.push(sys.motherboard);
    if (deviceName) deviceName.textContent = parts.join(' / ') || 'System';

    // Temperature
    var hasTemp = sys.cpu_temp_available && sys.cpu_temp > 0;
    var sysTemp = hasTemp ? Math.round(sys.cpu_temp) : 0;
    var tempPct = Math.min(100, (sysTemp / 100) * 100);
    var tempColor = getTempSeverityColor(sysTemp);
    var isTempHot = sysTemp >= 75;
    if (tempGauge) {
        tempGauge.style.setProperty('--pct', hasTemp ? tempPct.toFixed(1) : '0');
        tempGauge.style.setProperty('--gauge-color', tempColor);
        tempGauge.classList.toggle('is-warming', isTempHot);
    }
    if (tempValue) tempValue.textContent = hasTemp ? sysTemp + '\u00B0' : '\u2014';

    // Show temp unavailable badge when connected to remote agent without temp data
    var tempBadge = document.getElementById('sys-temp-unavailable-badge');
    if (tempBadge) {
        var isRemoteAgent = wsData && wsData.endpoint_kind === 'Remote';
        if (!hasTemp && isRemoteAgent) {
            var reason = sys.cpu_temp_available
                ? 'Sensor returned no data'
                : 'sensor_bridge not installed';
            tempBadge.style.display = 'inline-flex';
            tempBadge.title = 'CPU temperature unavailable: ' + reason;
            tempBadge.querySelector('.hw-temp-badge-text').textContent = 'No temp data';
        } else {
            tempBadge.style.display = 'none';
        }
    }

    // Show sensor_bridge setup callout on Windows when temp is unavailable
    var sbSetup = document.getElementById('sensor-bridge-setup-callout');
    var setupAvailable = lastCapabilities && lastCapabilities.sensor_bridge_setup_available;
    if (sbSetup) {
        sbSetup.style.display = (setupAvailable && !hasTemp) ? '' : 'none';
    }

    // Push history
    if (sys.cpu_load > 0) pushSysHistory('cpuLoad', sys.cpu_load);
    var ramPct = sys.ram_total_gb > 0 ? (sys.ram_used_gb / sys.ram_total_gb) * 100 : 0;
    if (sys.ram_total_gb > 0) pushSysHistory('ramPct', ramPct);

    // CPU Load
    var loadViz = document.getElementById('sys-load-viz');
    var loadVal = document.getElementById('sys-load-value');
    var loadStyle = vizPrefs.system.load;
    var cpuLoad = sys.cpu_load || 0;
    var loadTone = getMetricTone('load');
    var loadColor = loadTone.line;
    if (loadStyle === 'ring') renderHwRing(loadViz, cpuLoad, loadTone, false);
    else if (loadStyle === 'sparkline') renderHwSparkline(loadViz, sysHistory.cpuLoad, loadColor);
    else renderHwBar(loadViz, cpuLoad, loadTone, false);
    renderHwMetricSparkline('sys-load-spark', sysHistory.cpuLoad, loadColor, loadStyle !== 'sparkline');
    if (loadVal) loadVal.textContent = cpuLoad > 0 ? cpuLoad + '%' : '\u2014';

    // RAM
    var ramViz = document.getElementById('sys-ram-viz');
    var ramVal = document.getElementById('sys-ram-value');
    var ramStyle = vizPrefs.system.ram;
    var ramTone = getMetricTone('memory');
    var ramColor = ramTone.line;
    if (ramStyle === 'ring') renderHwRing(ramViz, ramPct, ramTone, false);
    else if (ramStyle === 'sparkline') renderHwSparkline(ramViz, sysHistory.ramPct, ramColor);
    else if (ramStyle === 'stacked') renderHwStacked(ramViz, ramPct, ramTone, false);
    else renderHwBar(ramViz, ramPct, ramTone, false);
    renderHwMetricSparkline('sys-ram-spark', sysHistory.ramPct, ramColor, ramStyle !== 'sparkline');
    if (ramVal) ramVal.textContent = sys.ram_total_gb > 0 ? sys.ram_used_gb.toFixed(1) + ' / ' + sys.ram_total_gb.toFixed(0) + ' GB' : '\u2014';

    // Memory pressure (macOS vm_stat; hidden on platforms without pressure fields)
    var pressureBlock = document.getElementById('sys-pressure-block');
    var pressureViz = document.getElementById('sys-pressure-viz');
    var pressureVal = document.getElementById('sys-pressure-value');
    var pressureLevel = sys.memory_pressure_level || '';
    var hasPressure = pressureLevel || sys.memory_compressor_gb > 0 || sys.memory_free_gb > 0;
    if (pressureBlock) pressureBlock.style.display = hasPressure ? '' : 'none';
    if (hasPressure) {
        var compressorRatio = sys.ram_total_gb > 0 ? (sys.memory_compressor_gb / sys.ram_total_gb) * 100 : 0;
        var pressurePct = pressureLevel === 'critical' ? 96 : pressureLevel === 'warning' ? 72 : Math.min(55, Math.max(5, compressorRatio * 2));
        pushSysHistory('memoryPressure', pressurePct);
        var pressureTone = pressureLevel === 'critical'
            ? { line: '#f87171', fill: 'rgba(248,113,113,0.18)', glow: 'rgba(248,113,113,0.35)' }
            : pressureLevel === 'warning'
                ? { line: '#f59e0b', fill: 'rgba(245,158,11,0.18)', glow: 'rgba(245,158,11,0.35)' }
                : getMetricTone('memory');
        renderHwMetricSparkline('sys-pressure-spark', sysHistory.memoryPressure, pressureTone.line, true);
        // Structured breakdown rows
        var bkFree = document.getElementById('sys-pressure-free');
        var bkWired = document.getElementById('sys-pressure-wired');
        var bkComp = document.getElementById('sys-pressure-compressed');
        var bkPurg = document.getElementById('sys-pressure-purgeable');
        var bkInact = document.getElementById('sys-pressure-inactive');
        var wiredGb = sys.memory_wired_gb || 0;
        var purgeableGb = sys.memory_purgeable_gb || 0;
        var inactiveGb = sys.memory_inactive_gb || 0;
        if (bkFree) bkFree.textContent = sys.memory_free_gb > 0 ? sys.memory_free_gb.toFixed(1) + ' GB' : '—';
        if (bkWired) {
            bkWired.textContent = wiredGb > 0 ? wiredGb.toFixed(1) + ' GB' : '—';
            bkWired.className = 'mem-pressure-val' + (wiredGb > 8 ? ' mem-pressure-val--warn' : '');
        }
        if (bkComp) bkComp.textContent = sys.memory_compressor_gb > 0
            ? sys.memory_compressor_gb.toFixed(1) + ' GB (' + (sys.memory_compressed_gb || 0).toFixed(1) + ' logical)'
            : '—';
        if (bkPurg) bkPurg.textContent = purgeableGb > 0 || sys.memory_reclaimable_gb > 0
            ? (sys.memory_reclaimable_gb || purgeableGb).toFixed(1) + ' GB'
            : '—';
        if (bkInact) bkInact.textContent = inactiveGb > 0
            ? inactiveGb.toFixed(1) + ' GB'
            : '—';

        // Keep the old pressureVal for tooltip accessibility but hide the element
        if (pressureVal) pressureVal.style.display = 'none';

        // Wire the purge button once
        var purgeBtn = document.getElementById('sys-pressure-purge-btn');
        if (purgeBtn && !purgeBtn._wired) {
            purgeBtn._wired = true;
            purgeBtn.addEventListener('click', () => _triggerMemoryPurge('sys-pressure-purge-btn', 'sys-pressure-purge-status', 'sys-pressure-purge-label'));
        }
    }

    // Clock
    var clockViz = document.getElementById('sys-clock-viz');
    var clockVal = document.getElementById('sys-clock-value');
    var clockStyle = vizPrefs.system.clock;
    var clockMhz = sys.cpu_clock_mhz || 0;
    if (clockMhz > 0) pushSysHistory('cpuClock', clockMhz);
    if (clockStyle === 'ring') {
        renderHwClockRing(clockViz, clockMhz);
        if (clockVal) clockVal.textContent = '';
    } else if (clockStyle === 'chip') {
        var display = formatClockReadout(clockMhz);
        renderHwChips(clockViz, [clockMhz > 0 ? display.value + ' ' + display.unit : '\u2014']);
        if (clockVal) clockVal.textContent = '';
    } else {
        if (clockViz) clockViz.innerHTML = '';
        if (clockVal) clockVal.textContent = clockMhz > 0 ? clockMhz + ' MHz' : '\u2014';
    }

    // Power (Apple Silicon only)
    var powerBlock = document.getElementById('sys-power-block');
    var powerTotal = sys.power_total_w || 0;
    if (powerBlock && powerTotal > 0) {
        powerBlock.style.display = '';
        var powerVizEl = document.getElementById('sys-power-viz');
        var powerValEl = document.getElementById('sys-power-value');
        var powerPctSys = Math.min(100, (powerTotal / 150) * 100);
        pushSysHistory('power', powerTotal);
        var powerStyle = vizPrefs.system.power || 'bar';
        var powerTone = getMetricTone('power');
        if (powerStyle === 'ring') renderHwRing(powerVizEl, powerPctSys, powerTone, false);
        else if (powerStyle === 'sparkline') renderHwSparkline(powerVizEl, sysHistory.power, powerTone.line);
        else renderHwBar(powerVizEl, powerPctSys, powerTone, false);
        renderHwMetricSparkline('sys-power-spark', sysHistory.power, powerTone.line, powerStyle !== 'sparkline');
        if (powerValEl) powerValEl.textContent = powerTotal.toFixed(1) + 'W';
    } else if (powerBlock) {
        powerBlock.style.display = 'none';
    }

    // CPU Clusters (Apple Silicon only)
    var clustersBlock = document.getElementById('sys-clusters-block');
    var pActive = sys.p_cluster_active || 0;
    var sActive = sys.s_cluster_active || 0;
    var eActive = sys.e_cluster_active || 0;
    var hasClusters = (pActive > 0 || sActive > 0 || eActive > 0);

    if (clustersBlock && hasClusters) {
        clustersBlock.style.display = '';

        var pCores = sys.p_cores || 0;
        var eCores = sys.e_cores || 0;

        // Push history for each cluster
        if (pActive > 0) pushSysHistory('pCluster', pActive);
        if (sActive > 0) pushSysHistory('sCluster', sActive);
        if (eActive > 0) pushSysHistory('eCluster', eActive);

        function clusterVal(pct, freqMhz) {
            var s = Math.round(pct) + '%';
            if (freqMhz > 0) s += ' · ' + (freqMhz >= 1000 ? (freqMhz / 1000).toFixed(2) + ' GHz' : freqMhz + ' MHz');
            return s;
        }

        // Build cluster labels from sysctl names e.g. "Super (6)", "Performance-A (6)"
        var pClusterName = sys.p_cluster_name || '';
        var secClusterName = sys.secondary_cluster_name || '';
        var sCores = sys.s_cores || 0;

        function clusterLabel(sysctlName, coreCount, suffix) {
            var base = sysctlName ? sysctlName : suffix;
            return coreCount > 0 ? base + ' (' + coreCount + ')' : base;
        }

        // P-cores row
        var pRow = document.getElementById('sys-cluster-p-row');
        var pBar = document.getElementById('sys-cluster-p-bar');
        var pVal = document.getElementById('sys-cluster-p-value');
        var pName = document.getElementById('sys-cluster-p-name');
        if (pRow && pBar && pVal) {
            pRow.style.display = '';
            var pTone = getMetricTone('load');
            renderHwBar(pBar, Math.min(100, pActive), pTone, false);
            pVal.textContent = clusterVal(pActive, sys.p_cluster_freq_mhz || 0);
            if (pName) pName.textContent = clusterLabel(pClusterName, pCores, 'P-Cores');
            renderHwMetricSparkline('sys-cluster-p-spark', sysHistory.pCluster, pTone.line, true);
        }

        // S-cores row (show only if present)
        var sRow = document.getElementById('sys-cluster-s-row');
        var sBar = document.getElementById('sys-cluster-s-bar');
        var sVal = document.getElementById('sys-cluster-s-value');
        var sName = document.getElementById('sys-cluster-s-name');
        if (sActive > 0 && sRow && sBar && sVal) {
            sRow.style.display = '';
            var sTone = getMetricTone('load');
            renderHwBar(sBar, Math.min(100, sActive), sTone, false);
            sVal.textContent = clusterVal(sActive, sys.s_cluster_freq_mhz || 0);
            var sLabelName = (secClusterName && eActualCores > 0)
                ? secClusterName + '-A'
                : (secClusterName || '');
            if (sName) sName.textContent = clusterLabel(sLabelName, sCores, 'S-Cores');
            renderHwMetricSparkline('sys-cluster-s-spark', sysHistory.sCluster, sTone.line, true);
        } else if (sRow) {
            sRow.style.display = 'none';
        }

        // E-cores row
        var eRow = document.getElementById('sys-cluster-e-row');
        var eBar = document.getElementById('sys-cluster-e-bar');
        var eVal = document.getElementById('sys-cluster-e-value');
        var eName = document.getElementById('sys-cluster-e-name');
        var eActualCores = sCores > 0 ? eCores : eCores;
        if (eActualCores > 0 && eRow && eBar && eVal) {
            eRow.style.display = '';
            var eTone = getMetricTone('load');
            renderHwBar(eBar, Math.min(100, eActive), eTone, false);
            eVal.textContent = clusterVal(eActive, sys.e_cluster_freq_mhz || 0);
            if (eName) eName.textContent = clusterLabel(secClusterName ? secClusterName + '-B' : '', eActualCores, 'E-Cores');
            renderHwMetricSparkline('sys-cluster-e-spark', sysHistory.eCluster, eTone.line, true);
        }
    } else if (clustersBlock) {
        clustersBlock.style.display = 'none';
    }
}

function setMetricSectionVisibility(cardId, visible, sectionId) {
    const card = document.getElementById(cardId);
    if (!card) return;
    const section = sectionId ? document.getElementById(sectionId) : card.closest('.metric-section');
    if (section) section.style.display = visible ? '' : 'none';
}

// ── Memory purge + top-process helpers ───────────────────────────────────────

// Shared purge handler used by both the dashboard card and nav hovercard buttons.
// btnId: element to show spinner on, statusId: status message element, labelId: button label element.
async function _triggerMemoryPurge(btnId, statusId, labelId) {
    const btn = document.getElementById(btnId);
    const statusEl = document.getElementById(statusId);
    const labelEl = document.getElementById(labelId);
    if (!btn || btn._purging) return;
    btn._purging = true;
    if (labelEl) labelEl.textContent = 'Requesting…';
    if (statusEl) { statusEl.style.display = ''; statusEl.textContent = 'Waiting for macOS admin dialog…'; statusEl.className = 'mem-pressure-purge-status'; }

    try {
        const adminToken = await _fetchDbAdminTokenForSystemAction();
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
        const msg = data.message || data.error || (data.ok ? 'Done.' : `Request failed (${res.status}).`);
        if (statusEl) {
            statusEl.textContent = msg;
            statusEl.className = 'mem-pressure-purge-status' + (data.ok ? ' mem-pressure-purge-status--ok' : ' mem-pressure-purge-status--err');
        }
    } catch {
        if (statusEl) {
            statusEl.textContent = 'Request failed — check server connection.';
            statusEl.className = 'mem-pressure-purge-status mem-pressure-purge-status--err';
        }
    } finally {
        btn._purging = false;
        if (labelEl) labelEl.textContent = 'Free Memory';
        // Auto-hide status after 6 s
        if (statusEl) setTimeout(() => { statusEl.style.display = 'none'; }, 6000);
    }
}

async function _fetchDbAdminTokenForSystemAction() {
    const tokenResp = await fetch('/api/db/admin-token', {
        headers: window.authHeaders ? window.authHeaders() : {},
    });
    const tokenData = tokenResp.ok ? await tokenResp.json().catch(() => ({})) : {};
    return tokenData.token || null;
}

// Fetch top memory processes and render them into a container element.
async function _renderTopProcesses(containerId) {
    const el = document.getElementById(containerId);
    if (!el) return;
    el.style.display = '';
    el.textContent = 'Loading…';
    el.className = 'mem-pressure-proc-loading';
    try {
        const res = await fetch('/system/top-processes', { headers: window.authHeaders ? window.authHeaders() : {} });
        const procs = await res.json();
        el.textContent = '';
        el.className = '';
        if (!Array.isArray(procs) || procs.length === 0) {
            el.textContent = 'No process data available.';
            el.className = 'mem-pressure-proc-empty';
            return;
        }
        const title = document.createElement('div');
        title.className = 'mem-pressure-proc-title';
        title.textContent = 'Top processes by RAM';
        el.appendChild(title);
        procs.slice(0, 10).forEach(p => {
            const row = document.createElement('div');
            row.className = 'mem-pressure-proc-row';
            const name = document.createElement('span');
            name.className = 'mem-pressure-proc-name';
            name.title = p.command;
            name.textContent = p.name;
            const mb = document.createElement('span');
            mb.className = 'mem-pressure-proc-mb';
            mb.textContent = Math.round(p.rss_mb) + ' MB';
            row.appendChild(name);
            row.appendChild(mb);
            el.appendChild(row);
        });
    } catch {
        el.textContent = 'Could not load processes.';
        el.className = 'mem-pressure-proc-empty';
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

export function initDashboardRender() {
    // Bind viz gear buttons
    document.getElementById('viz-gear-gpu')?.addEventListener('click', () => toggleVizSwitcher('gpu'));
    document.getElementById('viz-gear-system')?.addEventListener('click', () => toggleVizSwitcher('system'));

    // Bind viz reset buttons
    document.getElementById('viz-reset-gpu')?.addEventListener('click', () => resetVizPrefs('gpu'));
    document.getElementById('viz-reset-system')?.addEventListener('click', () => resetVizPrefs('system'));

    // Event delegation for viz style options
    document.querySelectorAll('.viz-switcher-options').forEach(container => {
        container.addEventListener('click', (e) => {
            const opt = e.target.closest('.viz-option');
            if (!opt) return;
            const card = container.dataset.card;
            const metric = container.dataset.metric;
            const style = opt.dataset.style;
            selectVizStyle(card, metric, style);
        });
    });
}

// Export render functions for dashboard-ws.js (replaces window.* bridges)
export {
    setChipState,
    setCardState,
    pushSparklinePoint,
    renderSparkline,
    renderLiveSparkline,
    updateLiveOutputEstimate,
    updateRequestActivity,
    renderRecentTask,
    renderActivityRail,
    renderSlotGrid,
    getPrimarySlot,
    renderSlotUtilization,
    renderBatchEfficiency,
    renderRequestStats,
    renderGenerationDetailItems,
    renderDecodingConfig,
    formatParamCount,
    renderCapabilityPopover,
    updateMetricDelta,
    setEmptyState,
    renderGpuCard,
    renderSystemCard,
    setMetricSectionVisibility,
};
