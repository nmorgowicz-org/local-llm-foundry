// llama.cpp Runtime panel: prerequisites, source build, live log, disk, rollback.
//
// The server owns every decision. This module draws what `/api/llama-build/*`
// and `/api/llama-binary/*` report and sends the user's choices back. All
// server text is inserted with textContent, never as HTML.

import { showToast } from './toast.js';
import { attachModalFocusTrap, detachModalFocusTrap } from './updater-shared.js';
import {
    PHASES, buildDisabledReason, cancelState, currentRuntimeLine, diagnosticsText, formatBytes,
    isSourceBuild, jobIsActive, phaseProgressText, rollbackLabel, runtimeLastModel,
} from './llama-runtime-state.js';

const OVERLAY_ID = 'llama-runtime-panel';
const FAST_POLL_MS = 1500;
const IDLE_POLL_MS = 6000;

const view = {
    accelerator: null,
    status: null,
    prereqs: null,
    options: null,
    job: null,
    cache: null,
    runtime: null,
    version: null,
    logLines: [],
    logNext: 0,
    follow: true,
    form: { ref: 'latest_release', cudaArch: 'auto', jobs: '', headroom: false, wipe: false },
    timer: null,
    open: false,
    busy: new Set(),
    onChange: null,
};

function headers(json = false) {
    const base = window.authHeaders ? window.authHeaders() : {};
    return json ? { ...base, 'Content-Type': 'application/json' } : base;
}

async function api(path, { method = 'GET', body } = {}) {
    const resp = await fetch(path, {
        method,
        headers: headers(body !== undefined),
        body: body === undefined ? undefined : JSON.stringify(body),
    });
    let data = {};
    try { data = await resp.json(); } catch (_) { /* non-JSON error body */ }
    return { status: resp.status, ok: resp.ok && data.ok !== false, data };
}

function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
        if (value === undefined || value === null || value === false) continue;
        if (key === 'class') node.className = value;
        else if (key === 'text') node.textContent = value;
        else if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2), value);
        else node.setAttribute(key, value === true ? '' : String(value));
    }
    for (const child of children.flat()) {
        if (child === undefined || child === null || child === false) continue;
        node.append(child.nodeType ? child : document.createTextNode(String(child)));
    }
    return node;
}

function overlay() {
    return document.getElementById(OVERLAY_ID);
}

const STATUS_ICON = { pass: '✔', warn: '⚠', fail: '✖', skipped: '–' };

// ── Data loading ────────────────────────────────────────────────────────────

async function loadStatus() {
    const r = await api('/api/llama-build/status');
    if (!r.ok) return;
    view.status = r.data;
    if (!view.accelerator) view.accelerator = r.data.default_accelerator;
    const id = r.data.active_job_id || r.data.last_job?.id;
    if (id) await loadJob(id);
    else view.job = null;
}

async function loadJob(id) {
    const r = await api(`/api/llama-build/jobs/${encodeURIComponent(id)}`);
    if (!r.ok) return;
    const switched = view.job?.id !== r.data.id;
    view.job = r.data;
    if (switched) {
        view.logLines = [];
        view.logNext = 0;
    }
    await loadLog();
}

async function loadLog() {
    if (!view.job) return;
    const r = await api(`/api/llama-build/jobs/${encodeURIComponent(view.job.id)}/log?after=${view.logNext}`);
    if (!r.ok) return;
    view.logLines.push(...(r.data.lines || []));
    view.logNext = r.data.next ?? view.logNext;
}

async function loadPrereqs() {
    if (!view.accelerator) return;
    const r = await api(`/api/llama-build/prereqs?accelerator=${encodeURIComponent(view.accelerator)}`);
    if (r.ok) view.prereqs = r.data;
}

async function loadOptions() {
    if (!view.accelerator) return;
    const r = await api(`/api/llama-build/options?accelerator=${encodeURIComponent(view.accelerator)}`);
    if (r.ok) view.options = r.data;
}

async function loadSide() {
    const [cache, runtime, version] = await Promise.all([
        api('/api/llama-build/cache'),
        api('/api/llama-binary/runtime-status'),
        api('/api/llama-binary/version'),
    ]);
    if (cache.ok) view.cache = cache.data;
    if (runtime.ok) view.runtime = runtime.data;
    if (version.ok) view.version = version.data;
}

async function refreshAll() {
    await loadStatus();
    await Promise.all([loadPrereqs(), loadOptions(), loadSide()]);
    render();
    notifyChange();
}

function notifyChange() {
    if (typeof view.onChange === 'function') view.onChange({ job: view.job, runtime: view.runtime, version: view.version });
}

function schedule() {
    clearTimeout(view.timer);
    if (!view.open) return;
    const fast = jobIsActive(view.job) || view.prereqs?.validating;
    view.timer = setTimeout(async () => {
        await loadStatus();
        if (view.prereqs?.validating || jobIsActive(view.job)) await loadPrereqs();
        else await Promise.all([loadPrereqs(), loadSide()]);
        render();
        notifyChange();
        schedule();
    }, fast ? FAST_POLL_MS : IDLE_POLL_MS);
}

// ── Actions ─────────────────────────────────────────────────────────────────

async function act(key, work) {
    if (view.busy.has(key)) return;
    view.busy.add(key);
    render();
    try { await work(); } finally { view.busy.delete(key); }
    await refreshAll();
    schedule();
}

function errorToast(title, r) {
    const owner = r.data?.owner ? ` (${r.data.owner})` : '';
    showToast(title, 'error', `${r.data?.error || 'Request failed'}${owner}`);
}

function validate() {
    return act('validate', async () => {
        const r = await api('/api/llama-build/prereqs/validate', { method: 'POST', body: { accelerator: view.accelerator } });
        if (!r.ok) errorToast('Could not validate', r);
    });
}

function brewInstall(packages) {
    return act('fix', async () => {
        const r = await api('/api/llama-build/prereqs/fix', { method: 'POST', body: { action: 'brew_install', packages } });
        if (!r.ok) errorToast('Install failed', r);
        else showToast('Installed', 'success', packages.join(', '));
    });
}

function startBuild(overrides = {}) {
    return act('build', async () => {
        const form = { ...view.form, ...overrides };
        const body = {
            accelerator: view.accelerator,
            ref: form.ref,
            headroom: !!form.headroom,
            wipe: !!form.wipe,
        };
        if (view.options?.show_cuda && view.accelerator === 'cuda' && form.cudaArch && form.cudaArch !== 'auto') body.cuda_arch = form.cudaArch;
        if (String(form.jobs).trim()) body.jobs = Number(form.jobs);
        const r = await api('/api/llama-build/start', { method: 'POST', body });
        if (!r.ok) errorToast('Could not start build', r);
        else {
            view.logLines = [];
            view.logNext = 0;
            view.follow = true;
        }
    });
}

function cancelBuild() {
    if (!view.job) return;
    if (!window.confirm('Cancel the build? Nothing will be installed.')) return;
    return act('cancel', async () => {
        const r = await api(`/api/llama-build/jobs/${encodeURIComponent(view.job.id)}/cancel`, { method: 'POST' });
        if (!r.ok) errorToast('Cannot cancel', r);
    });
}

function rollback() {
    if (!window.confirm('Swap back to the previous llama.cpp runtime? A running model will be restarted.')) return;
    return act('rollback', async () => {
        const r = await api('/api/llama-binary/rollback', { method: 'POST' });
        if (!r.ok) errorToast('Rollback failed', r);
        else if (r.data.outcome === 'installed_restart_failed') showToast('Rolled back; model restart failed', 'error', r.data.restart_error || 'Restart the model manually.');
        else showToast('Rolled back', 'success', r.data.label || 'Previous runtime restored.');
    });
}

function cleanCache(includeSource) {
    const what = includeSource ? 'build trees, ccache and the source checkout' : 'inactive build trees and ccache';
    if (!window.confirm(`Delete ${what}? The next build will be a full rebuild.`)) return;
    return act('clean', async () => {
        const r = await api(`/api/llama-build/cache?include_source=${includeSource ? 'true' : 'false'}`, { method: 'DELETE' });
        if (!r.ok) errorToast('Could not clean', r);
        else showToast('Build cache cleaned', 'success', `Freed ${formatBytes(r.data.freed_bytes)}`);
    });
}

async function copyText(text, label) {
    try {
        await navigator.clipboard.writeText(text);
        showToast('Copied', 'success', label);
    } catch (_) {
        showToast('Copy failed', 'error', 'Clipboard is not available here.');
    }
}

export function dismissKey(key) {
    const set = dismissedSet();
    set.add(key);
    sessionStorage.setItem('llama-build-dismissed', JSON.stringify([...set]));
    notifyChange();
    render();
}

export function dismissCurrent() {
    if (view.job) dismissKey(view.job.id);
}

export function dismissedSet() {
    try { return new Set(JSON.parse(sessionStorage.getItem('llama-build-dismissed') || '[]')); } catch (_) { return new Set(); }
}

// ── Rendering ───────────────────────────────────────────────────────────────

function buildBlockReason() {
    return buildDisabledReason({
        report: view.prereqs?.report,
        owner: view.status?.owner || view.status?.recovery_required,
        activeJobId: view.status?.active_job_id,
        validating: view.prereqs?.validating,
    });
}

function repairProcesses() {
    if (!window.confirm('Clear the leftover build process record? Only do this after stopping any leftover cmake, ninja or compiler processes. Nothing is killed for you.')) return;
    return act('repair', async () => {
        const r = await api('/api/llama-build/repair-processes', { method: 'POST', body: { confirm: true } });
        if (r.ok) showToast('Process record cleared', 'success', `${r.data.cleared ?? 0} record(s)`);
        else {
            const live = (r.data?.processes || []).map(p => `${p.pid}: ${p.command}`).join('\n');
            showToast('Cannot repair yet', 'error', `${r.data?.error || 'Request failed'}${live ? `\n${live}` : ''}`);
        }
    });
}

function recoveryBanner() {
    const reason = view.status?.recovery_required;
    if (!reason) return null;
    return el('section', { class: 'lrp-banner lrp-err', 'data-lrp': 'recovery-banner', role: 'alert' },
        el('div', { class: 'lrp-banner-head' },
            el('strong', { text: '✖ A previous build left process records' }),
            el('button', { type: 'button', class: 'lrp-btn', 'data-lrp': 'repair-processes', disabled: view.busy.has('repair'), text: 'Clear record…', onclick: repairProcesses })),
        el('div', { class: 'lrp-banner-summary', text: reason }));
}

function rolledBackBanner() {
    const last = runtimeLastModel(view.runtime?.last);
    if (!last || last.outcome !== 'rolled_back' || dismissedSet().has(`tx:${last.id}`)) return null;
    return el('section', { class: 'lrp-banner lrp-warn', 'data-lrp': 'rolled-back-banner', role: 'status' },
        el('div', { class: 'lrp-banner-head' },
            el('strong', { text: `⚠ Update rolled back${last.label ? `: ${last.label}` : ''}` }),
            el('button', { type: 'button', class: 'lrp-btn', 'data-lrp': 'dismiss-rolled-back', text: 'Dismiss', onclick: () => dismissKey(`tx:${last.id}`) })),
        last.reason && el('div', { class: 'lrp-banner-summary', text: last.reason }),
        el('div', { class: 'lrp-muted', text: 'The previous runtime was restored.' }));
}

function errorBanner() {
    const job = view.job;
    if (!job || !['failed', 'interrupted', 'succeeded_restart_failed'].includes(job.status)) return null;
    if (dismissedSet().has(job.id)) return null;
    const restart = job.status === 'succeeded_restart_failed';
    const failure = job.failure;
    const where = failure?.phase || job.phase;
    const title = restart
        ? 'Runtime installed; the model restart failed'
        : `Last build failed at ${String(where).toUpperCase()}${failure?.exit_code != null ? `  (exit ${failure.exit_code})` : ''}`;
    const children = [
        el('div', { class: 'lrp-banner-head' },
            el('strong', { text: `${restart ? '⚠' : '✖'} ${title}` }),
            el('span', { class: 'lrp-banner-actions' },
                el('button', { type: 'button', class: 'lrp-btn', text: 'Copy diagnostics', onclick: () => copyText(diagnosticsText({ job, report: view.prereqs?.report, provenance: view.version?.provenance, status: view.status }), 'Diagnostics copied') }),
                el('button', { type: 'button', class: 'lrp-btn', 'data-lrp': 'dismiss', text: 'Dismiss', onclick: dismissCurrent }))),
    ];
    if (job.outcome) children.push(el('div', { class: 'lrp-banner-summary', text: job.outcome }));
    if (failure?.first_error) children.push(el('pre', { class: 'lrp-first-error', text: failure.first_error }));
    if (failure?.hint) {
        const hint = failure.hint;
        const row = el('div', { class: 'lrp-hint' }, el('span', { text: `Hint: ${hint.message || hint.text || ''} ` }));
        if (hint.fix?.kind === 'action') {
            const actions = {
                wipe_build_dir: () => startBuild({ wipe: true }),
                leave_headroom: () => startBuild({ headroom: true }),
                clean_build_cache: () => cleanCache(false),
                revalidate: () => validate(),
            };
            const run = actions[hint.fix.id];
            const startsBuild = ['wipe_build_dir', 'leave_headroom'].includes(hint.fix.id);
            const blocked = startsBuild ? buildBlockReason() : '';
            if (run) row.append(el('button', { type: 'button', class: 'lrp-btn lrp-primary', 'data-lrp': `hint-${hint.fix.id}`, disabled: !!blocked, title: blocked || undefined, text: hint.fix.label || hint.fix.id, onclick: run }));
        } else if (hint.fix?.kind === 'copy') {
            row.append(el('button', { type: 'button', class: 'lrp-btn', text: 'Copy fix', onclick: () => copyText(hint.fix.command, 'Fix copied') }));
        } else if (hint.fix?.kind === 'link') {
            row.append(el('a', { href: hint.fix.url, target: '_blank', rel: 'noopener noreferrer', text: hint.fix.label || 'Open' }));
        }
        children.push(row);
    }
    if (failure?.log_tail?.length) {
        children.push(el('details', { class: 'lrp-tail' }, el('summary', { text: `Last ${failure.log_tail.length} log lines` }), el('pre', { text: failure.log_tail.join('\n') })));
    }
    return el('section', { class: `lrp-banner ${restart ? 'lrp-warn' : 'lrp-err'}`, 'data-lrp': restart ? 'restart-banner' : 'error-banner', role: 'alert' }, children);
}

function currentSection() {
    const provenance = view.version?.provenance;
    const label = rollbackLabel(view.runtime);
    const owner = view.runtime?.busy_owner;
    const mismatch = provenance?.hashes_match === false;
    return el('section', { class: 'lrp-section', 'data-lrp': 'current' },
        el('div', { class: 'lrp-row' },
            el('span', { class: 'lrp-k', text: 'Current' }),
            el('span', { class: 'lrp-v', 'data-lrp': 'current-line', text: currentRuntimeLine(provenance, view.version?.build) })),
        provenance && provenance.hashes_match != null && el('div', { class: mismatch ? 'lrp-warn-text' : 'lrp-muted', 'data-lrp': 'provenance-hashes', text: mismatch ? '⚠ Installed files do not match the recorded provenance hashes.' : 'Provenance hashes match.' }),
        provenance?.commit && el('div', { class: 'lrp-muted', text: `commit ${provenance.commit.slice(0, 12)}` }),
        label && el('button', { type: 'button', class: 'lrp-btn', 'data-lrp': 'rollback', disabled: !!owner || view.busy.has('rollback'), title: owner ? `Busy: ${owner}` : undefined, text: label, onclick: rollback }));
}

function checkRow(check) {
    const row = el('li', { class: `lrp-check lrp-${check.status}`, 'data-check': check.id },
        el('span', { class: 'lrp-icon', text: STATUS_ICON[check.status] || '?' }),
        el('span', { class: 'lrp-check-text' }, el('strong', { text: check.title }), ' ', el('span', { class: 'lrp-muted', text: check.detail })));
    const fix = check.fix;
    if (fix?.kind === 'copy') row.append(el('button', { type: 'button', class: 'lrp-btn', text: 'Copy fix', onclick: () => copyText(fix.command, 'Fix copied') }));
    if (fix?.kind === 'link') row.append(el('a', { class: 'lrp-btn', href: fix.url, target: '_blank', rel: 'noopener noreferrer', text: fix.label || 'Open' }));
    if (fix?.kind === 'action' && fix.id?.startsWith('brew_install:')) {
        const packages = fix.id.slice('brew_install:'.length).split(',').filter(Boolean);
        row.append(el('button', { type: 'button', class: 'lrp-btn lrp-primary', 'data-lrp': 'brew-install', disabled: view.busy.has('fix') || !!view.status?.owner, text: fix.label || `brew install ${packages.join(' ')}`, onclick: () => brewInstall(packages) }));
    }
    if (check.evidence?.length) {
        row.append(el('details', { class: 'lrp-details' }, el('summary', { text: 'Details' }), el('pre', { text: check.evidence.join('\n') })));
    }
    return row;
}

function prereqSection() {
    const report = view.prereqs?.report;
    const validating = view.prereqs?.validating || view.busy.has('validate');
    const owner = view.status?.owner;
    const when = report?.validated_at_unix ? new Date(report.validated_at_unix * 1000).toLocaleString() : 'never';
    const list = el('ul', { class: 'lrp-checks' }, (report?.checks || []).map(checkRow));
    return el('section', { class: 'lrp-section', 'data-lrp': 'prereqs' },
        el('div', { class: 'lrp-row' },
            el('h3', { class: 'lrp-h', text: 'Prerequisites' }),
            el('span', { class: 'lrp-muted', text: `validated ${when}${view.prereqs?.stale ? ' (stale)' : ''}` }),
            el('button', { type: 'button', class: 'lrp-btn', 'data-lrp': 'validate', disabled: validating || !!owner, title: owner ? `Busy: ${owner}` : 'Run every check, including a real compile smoke test', text: validating ? 'Validating…' : 'Validate', onclick: validate })),
        report ? list : el('div', { class: 'lrp-muted', text: 'No validation result yet.' }));
}

function refRadio(ref) {
    const checked = view.form.ref === ref.id;
    const label = ref.commit ? `${ref.label} (${ref.reference || ''} ${ref.commit.slice(0, 7)})` : ref.label;
    return el('label', { class: 'lrp-radio' },
        el('input', { type: 'radio', name: 'lrp-ref', value: ref.id, checked, onchange: () => { view.form.ref = ref.id; } }),
        ' ', label);
}

function buildSection() {
    const options = view.options;
    const job = view.job;
    const active = jobIsActive(job);
    const reason = buildBlockReason();
    const cudaShown = !!options?.show_cuda && view.accelerator === 'cuda';
    const refs = options?.refs?.length ? options.refs : [{ id: 'latest_release', label: 'Latest release' }, { id: 'master', label: 'master HEAD' }];
    const form = el('div', { class: 'lrp-form' },
        el('div', { class: 'lrp-row' }, el('span', { class: 'lrp-k', text: 'Source' }), refs.map(refRadio)),
        options?.refs_error && el('div', { class: 'lrp-muted', text: `Could not look up upstream: ${options.refs_error}` }),
        cudaShown && el('div', { class: 'lrp-row', 'data-lrp': 'cuda-controls' },
            el('span', { class: 'lrp-k', text: 'FA K/V kernels' }), el('span', { text: 'all pairs (fixed)' })),
        cudaShown && el('div', { class: 'lrp-row' },
            el('label', { for: 'lrp-arch', text: 'CUDA arch' }),
            el('select', { id: 'lrp-arch', onchange: e => { view.form.cudaArch = e.target.value; } },
                el('option', { value: 'auto', selected: view.form.cudaArch === 'auto', text: `auto${options.cuda_arch_auto ? `: ${options.cuda_arch_auto}` : ''}` }),
                (options.cuda_arch_choices || []).map(a => el('option', { value: a, selected: view.form.cudaArch === a, text: a })))),
        el('div', { class: 'lrp-row' },
            el('label', { for: 'lrp-jobs', text: 'Jobs' }),
            el('input', { id: 'lrp-jobs', type: 'number', min: 1, placeholder: options?.jobs_auto ? `auto (${options.jobs_auto})` : 'auto', value: view.form.jobs, oninput: e => { view.form.jobs = e.target.value; } }),
            el('label', { class: 'lrp-check-label' }, el('input', { type: 'checkbox', id: 'lrp-headroom', checked: view.form.headroom, onchange: e => { view.form.headroom = e.target.checked; } }), ' leave headroom')),
        el('label', { class: 'lrp-check-label' }, el('input', { type: 'checkbox', id: 'lrp-wipe', checked: view.form.wipe, onchange: e => { view.form.wipe = e.target.checked; } }), ' Wipe build dir (full rebuild)'));
    const cancel = cancelState(job);
    const actions = el('div', { class: 'lrp-row' },
        el('button', { type: 'button', class: 'lrp-btn lrp-primary', 'data-lrp': 'build', disabled: !!reason || view.busy.has('build'), title: reason || 'Build llama.cpp from source and install it', text: 'Build & install', onclick: startBuild }),
        reason && el('span', { class: 'lrp-muted', 'data-lrp': 'build-reason', text: reason }),
        cancel.visible && el('button', { type: 'button', class: 'lrp-btn', 'data-lrp': 'cancel', disabled: !cancel.enabled, title: cancel.reason || 'Stop the build; nothing is installed', text: 'Cancel', onclick: cancelBuild }),
        cancel.visible && !cancel.enabled && el('span', { class: 'lrp-muted', 'data-lrp': 'cancel-reason', text: cancel.reason }),
        options?.last_build_seconds ? el('span', { class: 'lrp-muted', text: `last build took ${Math.round(options.last_build_seconds / 60)}m` }) : null);
    return el('section', { class: 'lrp-section', 'data-lrp': 'build-section' }, el('h3', { class: 'lrp-h', text: 'Build' }), active ? el('div', { class: 'lrp-muted', text: 'A build is running.' }) : form, actions);
}

function logSection() {
    const job = view.job;
    const stepper = el('div', { class: 'lrp-steps', 'data-lrp': 'phases' }, PHASES.map(p => {
        const idx = PHASES.indexOf(p);
        const cur = PHASES.indexOf(job?.phase);
        const failed = job && ['failed', 'interrupted'].includes(job.status) && job.phase === p;
        const done = job && (idx < cur || (job.status === 'succeeded' || job.status === 'succeeded_restart_failed'));
        const mark = failed ? '✖' : done ? '✔' : (job?.phase === p && jobIsActive(job)) ? '…' : '';
        const text = job?.phase === p && jobIsActive(job) ? `${p} ${mark} ${phaseProgressText(job).replace(p, '').trim()}` : `${p} ${mark}`;
        return el('span', { class: `lrp-step${job?.phase === p ? ' lrp-step-cur' : ''}`, text: text.trim() });
    }));
    const pre = el('pre', { class: 'lrp-log', id: 'lrp-log', 'data-lrp': 'log', text: view.logLines.join('\n') });
    queueMicrotask(() => { if (view.follow) pre.scrollTop = pre.scrollHeight; });
    return el('section', { class: 'lrp-section', 'data-lrp': 'log-section' },
        el('div', { class: 'lrp-row' },
            el('h3', { class: 'lrp-h', text: 'Log' }),
            el('label', { class: 'lrp-check-label' }, el('input', { type: 'checkbox', checked: view.follow, onchange: e => { view.follow = e.target.checked; } }), ' Follow'),
            el('button', { type: 'button', class: 'lrp-btn', text: 'Copy', onclick: () => copyText(view.logLines.join('\n'), 'Log copied') }),
            job?.failure?.log_path && el('span', { class: 'lrp-muted', text: job.failure.log_path })),
        job ? stepper : el('div', { class: 'lrp-muted', text: 'No build has run yet.' }),
        job && pre);
}

function diskSection() {
    const c = view.cache;
    if (!c) return el('section', { class: 'lrp-section', 'data-lrp': 'disk' });
    const trees = (c.build_trees || []).map(t => `build-${t.accelerator} ${formatBytes(t.bytes)}`);
    const parts = [`Source ${formatBytes(c.source_bytes)}`, ...trees, `ccache ${formatBytes(c.ccache_bytes)} / ${formatBytes(c.ccache_cap_bytes)}`];
    if (c.previous_bytes) parts.push(`previous set ${formatBytes(c.previous_bytes)}`);
    const owner = c.owner;
    return el('section', { class: 'lrp-section', 'data-lrp': 'disk' },
        el('h3', { class: 'lrp-h', text: 'Disk' }),
        el('div', { class: 'lrp-muted', 'data-lrp': 'disk-line', text: parts.join(' · ') }),
        el('div', { class: 'lrp-row' },
            el('button', { type: 'button', class: 'lrp-btn', 'data-lrp': 'clean', disabled: !!owner || view.busy.has('clean'), title: owner ? `Busy: ${owner}` : 'Next build will be a full rebuild', text: 'Clean build cache', onclick: () => cleanCache(false) }),
            el('button', { type: 'button', class: 'lrp-btn', 'data-lrp': 'clean-src', disabled: !!owner || view.busy.has('clean'), text: 'Also remove source', onclick: () => cleanCache(true) })));
}

function render() {
    const root = overlay();
    if (!root) return;
    const body = root.querySelector('.lrp-body');
    if (!body) return;
    const scroll = body.scrollTop;
    const nodes = [recoveryBanner(), rolledBackBanner(), errorBanner(), currentSection(), prereqSection(), buildSection(), logSection(), diskSection()].filter(Boolean);
    body.replaceChildren(...nodes);
    body.scrollTop = scroll;
}

// ── Open / close ────────────────────────────────────────────────────────────

function buildShell() {
    const closeBtn = el('button', { type: 'button', class: 'lrp-close', id: 'llama-runtime-panel-close', 'aria-label': 'Close', text: '×', onclick: closePanel });
    const shell = el('div', { class: 'modal-overlay lrp-overlay', id: OVERLAY_ID, role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': 'lrp-title', style: 'display:none;' },
        el('div', { class: 'lrp-shell' },
            el('div', { class: 'lrp-header' }, el('h2', { id: 'lrp-title', text: 'llama.cpp Runtime' }), closeBtn),
            el('div', { class: 'lrp-body' })));
    shell.addEventListener('click', e => { if (e.target === shell) closePanel(); });
    document.body.append(shell);
    return shell;
}

export async function openRuntimePanel() {
    const root = overlay() || buildShell();
    root.style.display = 'flex';
    root.classList.add('open');
    document.body.style.overflow = 'hidden';
    view.open = true;
    attachModalFocusTrap(root, closePanel);
    root.querySelector('#llama-runtime-panel-close')?.focus();
    await refreshAll();
    schedule();
}

export function closePanel() {
    const root = overlay();
    if (!root) return;
    root.style.display = 'none';
    root.classList.remove('open');
    document.body.style.overflow = '';
    detachModalFocusTrap(root);
    view.open = false;
    clearTimeout(view.timer);
}

export function setRuntimePanelObserver(fn) {
    view.onChange = fn;
}

export function isSourceRuntime() {
    return isSourceBuild(view.version?.provenance);
}
