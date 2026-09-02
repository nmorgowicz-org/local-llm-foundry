// Phase 8b — the Configure drawer for a v6 preset bundle.
//
// Draft-state-until-explicit-action: everything the user touches here is a
// draft. Three exit actions are the only way out — Start without saving
// (launches the normalized draft, never persists), Save (PATCH + canonical
// loadPresets refresh), and Save & Start (PATCH first, then launch the exact
// returned revision). sessionState.presets is the saved source of truth and is
// never mutated by draft edits.
//
// Copies the accessibility lifecycle from evidence-drawer.js (dialog semantics,
// backdrop, Escape, focus trap, focus restoration, narrow bottom sheet, light
// theme, reduced motion) — not its data-rendering code. Every untrusted value
// from the backend is inserted via textContent, never innerHTML.

import { sessionState } from '../core/app-state.js';
import { loadPresets, syncSelectedPresetSelection, openPresetModal } from './presets.js';
import { showToastWithActions } from './toast.js';
import { openEvidenceDrawer, evidenceFromLaunchObservation } from './evidence-drawer.js';

// ── The four known workload policies (wire ids), in display order ────────────
const WORKLOAD_POLICIES = [
    { value: 'agentic_tools', label: 'Agentic / tool use' },
    { value: 'general_chat', label: 'General chat' },
    { value: 'roleplay_creative', label: 'Roleplay / creative' },
    { value: 'custom_unknown', label: 'Custom / unknown' },
];

// Intent row: the three curated fit intents. `Custom` is derived, not one of
// these.
const INTENTS = [
    { value: 'quality_first', label: 'Quality first' },
    { value: 'balanced', label: 'Balanced' },
    { value: 'low_vram', label: 'Low VRAM' },
];

const KV_LABELS = {
    f16_f16: 'f16 / f16',
    q8_0_q8_0: 'Quality first · q8_0 / q8_0',
    q4_0_q4_0: 'Lower KV memory · q4_0 / q4_0',
    q8_0_q4_0: 'Mixed · q8_0 / q4_0',
};

function gb(bytes) {
    if (bytes == null || Number.isNaN(Number(bytes))) return '—';
    return `${(Number(bytes) / (1024 ** 3)).toFixed(1)}`;
}

function miBToGb(mib) {
    if (mib == null || Number.isNaN(Number(mib))) return '—';
    return `${(Number(mib) / 1024).toFixed(1)}`;
}

function ctxLabel(v) {
    const k = Math.round(v / 1024);
    return k >= 1000 ? `${(k / 1024).toFixed(1)}M` : `${k}k`;
}

function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
}

// Build one drawer instance. Called once per page; subsequent opens re-use it.
function buildDrawer() {
    const root = el('div', 'bundle-drawer');
    root.id = 'bundle-drawer';
    root.hidden = true;

    const backdrop = el('button', 'bundle-drawer-backdrop');
    backdrop.type = 'button';
    backdrop.setAttribute('data-bundle-close', '');
    backdrop.setAttribute('aria-label', 'Close configuration');
    backdrop.tabIndex = -1;
    root.appendChild(backdrop);

    const panel = el('aside', 'bundle-drawer-panel');
    panel.setAttribute('role', 'dialog');
    panel.setAttribute('aria-modal', 'true');
    panel.setAttribute('aria-labelledby', 'bundle-drawer-title');
    panel.tabIndex = -1;
    root.appendChild(panel);

    // Header
    const header = el('header', 'bundle-drawer-header');
    const titleWrap = el('div', 'bundle-drawer-title-wrap');
    const title = el('h2', 'bundle-drawer-title');
    title.id = 'bundle-drawer-title';
    titleWrap.appendChild(title);
    header.appendChild(titleWrap);
    const close = el('button', 'bundle-drawer-close');
    close.type = 'button';
    close.setAttribute('data-bundle-close', '');
    close.setAttribute('aria-label', 'Close');
    close.textContent = '×';
    header.appendChild(close);
    panel.appendChild(header);

    const body = el('div', 'bundle-drawer-body');
    panel.appendChild(body);

    // Body rows (populated on each open; cleared here)
    body.appendChild((() => {
        const intent = el('section', 'bundle-row bundle-row-intent');
        intent.appendChild(el('h3', 'bundle-row-label', 'What matters most?'));
        const controls = el('div', 'bundle-controls');
        INTENTS.forEach(int => {
            const btn = el('button', 'bundle-intent', int.label);
            btn.type = 'button';
            btn.dataset.intent = int.value;
            controls.appendChild(btn);
        });
        const custom = el('span', 'bundle-deriv-custom', 'Custom');
        custom.dataset.slot = 'intent';
        controls.appendChild(custom);
        intent.appendChild(controls);
        return intent;
    })());

    body.appendChild((() => {
        const row = el('section', 'bundle-row bundle-row-quant');
        row.appendChild(el('h3', 'bundle-row-label', 'Model quantization'));
        const wrap = el('div', 'bundle-controls bundle-radio-group');
        wrap.dataset.slot = 'quant';
        row.appendChild(wrap);
        return row;
    })());

    body.appendChild((() => {
        const row = el('section', 'bundle-row bundle-row-context');
        row.appendChild(el('h3', 'bundle-row-label', 'Context'));
        const controls = el('div', 'bundle-controls');
        const custom = el('span', 'bundle-deriv-custom', 'Custom');
        custom.dataset.slot = 'context';
        controls.appendChild(custom);
        row.appendChild(controls);
        return row;
    })());

    body.appendChild((() => {
        const row = el('section', 'bundle-row bundle-row-kv');
        row.appendChild(el('h3', 'bundle-row-label', 'KV quality'));
        const wrap = el('div', 'bundle-controls bundle-radio-group');
        wrap.dataset.slot = 'kv';
        row.appendChild(wrap);
        return row;
    })());

    body.appendChild((() => {
        const row = el('section', 'bundle-row bundle-row-perf');
        row.appendChild(el('h3', 'bundle-row-label', 'Performance'));
        const controls = el('div', 'bundle-controls');
        const custom = el('span', 'bundle-deriv-custom', 'Custom');
        custom.dataset.slot = 'perf';
        controls.appendChild(custom);
        row.appendChild(controls);
        const current = el('div', 'bundle-perf-current');
        row.appendChild(current);
        return row;
    })());

    body.appendChild((() => {
        const row = el('section', 'bundle-row bundle-row-moe');
        row.appendChild(el('h3', 'bundle-row-label', 'Expert placement · MoE model'));
        const controls = el('div', 'bundle-controls');
        const allGpu = el('button', 'bundle-moe', 'All GPU');
        allGpu.type = 'button';
        allGpu.dataset.moe = '0';
        const fit = el('button', 'bundle-moe', 'Fit automatically');
        fit.type = 'button';
        fit.dataset.moe = 'fit';
        controls.appendChild(allGpu);
        controls.appendChild(fit);
        const customWrap = el('label', 'bundle-moe-custom');
        const customLabel = el('span', 'bundle-moe-custom-label', 'Custom:');
        const customInput = el('input', 'bundle-moe-custom-input');
        customInput.type = 'number';
        customInput.min = '0';
        customWrap.appendChild(customLabel);
        customWrap.appendChild(customInput);
        const total = el('span', 'bundle-moe-total');
        controls.appendChild(customWrap);
        controls.appendChild(total);
        const headroomWrap = el('label', 'bundle-moe-headroom-wrap');
        const headroomLabel = el('span', 'bundle-moe-headroom-label', 'VRAM buffer (GB):');
        const headroomInput = el('input', 'bundle-moe-headroom');
        headroomInput.type = 'number';
        headroomInput.min = '0';
        headroomInput.step = '0.5';
        headroomInput.placeholder = '0';
        headroomWrap.appendChild(headroomLabel);
        headroomWrap.appendChild(headroomInput);
        controls.appendChild(headroomWrap);
        row.appendChild(controls);
        const probe = el('div', 'bundle-moe-probe');
        row.appendChild(probe);
        return row;
    })());

    body.appendChild((() => {
        const row = el('section', 'bundle-row bundle-row-workload');
        row.appendChild(el('h3', 'bundle-row-label', 'Workload'));
        const select = el('select', 'bundle-workload');
        WORKLOAD_POLICIES.forEach(wp => {
            const opt = el('option', null, wp.label);
            opt.value = wp.value;
            select.appendChild(opt);
        });
        row.appendChild(select);
        return row;
    })());

    body.appendChild((() => {
        const row = el('section', 'bundle-row bundle-row-result');
        row.appendChild(el('h3', 'bundle-row-label', 'Predicted result'));
        const result = el('div', 'bundle-result');
        const evidence = el('div', 'bundle-evidence');
        evidence.hidden = true;
        const line = el('div', 'bundle-result-line');
        const verdict = el('div', 'bundle-result-verdict');
        const moeWarn = el('div', 'bundle-result-moe');
        moeWarn.hidden = true;
        result.appendChild(evidence);
        result.appendChild(line);
        result.appendChild(verdict);
        result.appendChild(moeWarn);
        row.appendChild(result);
        return row;
    })());

    body.appendChild((() => {
        const row = el('section', 'bundle-row bundle-row-diff');
        row.appendChild(el('h3', 'bundle-row-label', 'Changed from saved:'));
        const list = el('ul', 'bundle-diff-list');
        const cause = el('div', 'bundle-diff-cause');
        cause.hidden = true;
        row.appendChild(list);
        row.appendChild(cause);
        return row;
    })());

    // Footer
    const footer = el('footer', 'bundle-drawer-footer');
    const editFull = el('button', 'bundle-edit-full', 'Edit full preset…');
    editFull.type = 'button';
    editFull.dataset.slot = 'edit-full';
    const reset = el('button', 'bundle-reset', 'Reset');
    reset.type = 'button';
    reset.dataset.slot = 'reset';
    const startDraft = el('button', 'bundle-start-draft', 'Start without saving');
    startDraft.type = 'button';
    startDraft.dataset.slot = 'start-draft';
    const save = el('button', 'bundle-save', 'Save');
    save.type = 'button';
    save.dataset.slot = 'save';
    const saveStart = el('button', 'bundle-save-start', 'Save & Start');
    saveStart.type = 'button';
    saveStart.dataset.slot = 'save-start';
    footer.appendChild(editFull);
    footer.appendChild(reset);
    footer.appendChild(startDraft);
    footer.appendChild(save);
    footer.appendChild(saveStart);
    panel.appendChild(footer);

    document.body.appendChild(root);

    return { root, panel, body, title, close };
}

let _drawer = null;
function drawer() {
    if (!_drawer) _drawer = buildDrawer();
    return _drawer;
}

// The module-local state contract (architecture 3.2). A single drawer instance
// owns one open-at-a-time draft, so a module singleton is sufficient.
const state = {
    presetId: '',
    bundleId: '',
    serverRevision: 0,
    savedSelection: null,
    draftSelection: null,
    normalizedPreview: null,
    dirty: false,
    previewRequestGeneration: 0,
    previewAbortController: null,
    opener: null,
    activeIntent: null,
    probeStatus: 'idle',
};

function isMoeArtifact(artifact) {
    const kind = artifact?.metadata?.model_kind;
    return kind === 'moe' || kind === 'hybrid_moe';
}

function selectedArtifact(bundle, selection) {
    return (bundle.artifacts || []).find(a => a.id === selection?.artifact_id)
        || (bundle.artifacts || []).find(a => a.role === 'weights')
        || (bundle.artifacts || [])[0]
        || null;
}

function moeLayerCount(bundle, selection) {
    return Number(selectedArtifact(bundle, selection)?.metadata?.moe_layer_count) || 0;
}

function cloneSelection(sel) {
    return sel ? JSON.parse(JSON.stringify(sel)) : {};
}

function selectionsEqual(a, b) {
    return JSON.stringify(a || {}) === JSON.stringify(b || {});
}

function apiHeaders(json = false) {
    const extra = json ? { 'Content-Type': 'application/json' } : {};
    return window.authHeaders ? window.authHeaders(extra) : extra;
}

// A reason lookup against the backend's capability_reasons. Entries are
// { field, value, reason }; a match means the option is unavailable and must be
// rendered disabled (never hidden) with the reason wired via aria-describedby.
function reasonFor(bundleId, capabilityReasons, field, value) {
    const match = (capabilityReasons || []).find(r => r && r.field === field && (value == null || r.value === value));
    return match ? match.reason || 'Unavailable' : null;
}

// ── Preview (resolve) ────────────────────────────────────────────────────────
async function requestResolve({ selection, fitAutomatically = false, fitTargetMib } = {}) {
    state.previewRequestGeneration += 1;
    const generation = state.previewRequestGeneration;
    if (state.previewAbortController) {
        try { state.previewAbortController.abort(); } catch { /* ignore */ }
    }
    const controller = new AbortController();
    state.previewAbortController = controller;

    const typedSelection = selection ? { ...selection } : selection;
    const body = { selection: typedSelection };
    if (selection?.workload_policy) {
        body.workload_policy = selection.workload_policy;
        delete body.selection.workload_policy;
    }
    if (fitAutomatically) body.fit_automatically = true;
    if (fitTargetMib != null) body.fit_target_mib = fitTargetMib;
    try {
        const avail = await fetchMemoryAvailability(controller.signal);
        if (avail != null) body.available_vram_bytes = avail;
    } catch { /* availability is estimate-class only; proceed without it */ }

    if (generation !== state.previewRequestGeneration) return;
    if (controller.signal.aborted) return;

    let data = null;
    try {
        const resp = await fetch(`/api/presets/${encodeURIComponent(state.presetId)}/resolve`, {
            method: 'POST',
            headers: apiHeaders(true),
            body: JSON.stringify(body),
            signal: controller.signal,
        });
        data = await resp.json().catch(() => null);
    } catch {
        // Network failure or abort: surface the backend reason only if we have
        // one; otherwise leave the last preview intact (no stale/inferred guess).
        if (generation === state.previewRequestGeneration) renderResolveUnavailable({ status: 'unavailable', code: 'resolve_failed', message: 'Could not reach the resolver.' });
        return;
    }

    // Only the newest generation may touch preview state.
    if (generation !== state.previewRequestGeneration) return;
    applyResolve(data);
}

async function fetchMemoryAvailability(signal) {
    const resp = await fetch('/api/memory-availability', { headers: apiHeaders(), signal });
    if (!resp.ok) return null;
    const json = await resp.json().catch(() => null);
    return json?.snapshot?.current_safe_availability_bytes ?? null;
}

function applyResolve(data) {
    if (!data || !data.ok) {
        renderResolveUnavailable(data || { status: 'unavailable', code: 'resolve_failed', message: 'Resolve failed.' });
        return;
    }
    state.probeStatus = 'idle';
    if (data.selection) {
        // The resolver's normalized selection is the draft going forward — not
        // just a transient preview payload. This is what lets an applied intent
        // or a "Fit automatically" proposal actually stick as the editable draft.
        const { intent_source, ...rest } = data.selection;
        state.draftSelection = { ...state.draftSelection, ...rest };
    }
    state.normalizedPreview = {
        selection: data.selection || cloneSelection(state.draftSelection),
        changes: data.changes || [],
        estimate: data.estimate || null,
        capability_reasons: data.capability_reasons || [],
        evidence: data.evidence || null,
        selection_hash: data.selection_hash || '',
        resolved_config_hash: data.resolved_config_hash || '',
        revision: data.revision ?? state.serverRevision,
    };
    if (!selectionsEqual(state.draftSelection, state.savedSelection)) state.dirty = true;
    renderAll();
}

function renderResolveUnavailable(est) {
    state.normalizedPreview = {
        ...(state.normalizedPreview || {}),
        selection: state.normalizedPreview?.selection || cloneSelection(state.draftSelection),
        changes: state.normalizedPreview?.changes || [],
        estimate: est || null,
        capability_reasons: state.normalizedPreview?.capability_reasons || [],
    };
    renderFromPreview();
}

// ── Rendering ────────────────────────────────────────────────────────────────
function currentBundle() {
    return (sessionState.presets || []).find(p => p.id === state.presetId)?.bundle || null;
}

function renderFromPreview() {
    const d = drawer();
    const bundle = currentBundle();
    const preview = state.normalizedPreview || {};
    const reasons = preview.capability_reasons || [];

    renderQuant(d, bundle, reasons);
    renderContext(d, bundle, reasons);
    renderKv(d, bundle, reasons);
    renderPerf(d, bundle);
    renderMoe(d, bundle, reasons, preview);
    renderWorkload(d, bundle);
    renderResult(d, bundle);
    renderDiff(d, bundle);
    renderDirty(d);
}

function setActiveIndicator(control, active) {
    control.classList.toggle('is-active', active);
    control.setAttribute('aria-pressed', active ? 'true' : 'false');
}

function renderIntentRow(d) {
    const controls = d.body.querySelector('.bundle-row-intent .bundle-controls');
    controls.querySelectorAll('button.bundle-intent').forEach(btn => {
        setActiveIndicator(btn, btn.dataset.intent === state.activeIntent);
    });
    const custom = controls.querySelector('.bundle-deriv-custom[data-slot="intent"]');
    showDerivedCustom(custom, state.activeIntent == null);
}

function showDerivedCustom(node, show) {
    node.classList.toggle('is-active', show);
    node.setAttribute('aria-hidden', show ? 'false' : 'true');
    node.hidden = !show;
}

function renderQuant(d, bundle, reasons) {
    const wrap = d.body.querySelector('.bundle-row-quant .bundle-radio-group');
    wrap.textContent = '';
    (bundle?.artifacts || []).filter(a => a.role === 'weights').forEach(artifact => {
        const id = `bundle-quant-${artifact.id}`;
        const label = el('label', 'bundle-radio-label');
        const input = el('input', 'bundle-quant');
        input.type = 'radio';
        input.name = 'bundle-quant';
        input.value = artifact.id;
        input.id = id;
        input.checked = state.draftSelection?.artifact_id === artifact.id;
        const reason = reasonFor(state.bundleId, reasons, 'quant', artifact.id);
        if (reason) {
            input.disabled = true;
            const reasonSpan = el('span', 'bundle-option-reason', reason);
            reasonSpan.id = `${id}-reason`;
            input.setAttribute('aria-describedby', reasonSpan.id);
            label.appendChild(reasonSpan);
        }
        const name = el('span', 'bundle-radio-name', (artifact.quantization?.value || artifact.display_name || '').toUpperCase());
        const size = el('span', 'bundle-radio-size', artifact.size_bytes != null ? `${gb(artifact.size_bytes)} GB` : '');
        label.appendChild(input);
        label.appendChild(name);
        if (size.textContent) label.appendChild(size);
        input.addEventListener('change', () => {
            if (input.disabled) return;
            state.draftSelection.artifact_id = artifact.id;
            clearIntent();
            markDirty();
            requestResolve({ selection: state.draftSelection });
        });
        wrap.appendChild(label);
    });
}

function renderContext(d, bundle, reasons) {
    const controls = d.body.querySelector('.bundle-row-context .bundle-controls');
    controls.textContent = '';
    (bundle?.context_options || []).forEach(value => {
        const btn = el('button', 'bundle-ctx', ctxLabel(value));
        btn.type = 'button';
        btn.dataset.context = String(value);
        const reason = reasonFor(state.bundleId, reasons, 'context', String(value));
        if (reason) {
            btn.disabled = true;
            btn.id = `bundle-ctx-${value}`;
            const reasonSpan = el('span', 'bundle-option-reason', reason);
            reasonSpan.id = `${btn.id}-reason`;
            btn.setAttribute('aria-describedby', reasonSpan.id);
            controls.appendChild(reasonSpan);
        }
        setActiveIndicator(btn, Number(state.draftSelection?.context_size) === value);
        btn.addEventListener('click', () => {
            if (btn.disabled) return;
            state.draftSelection.context_size = value;
            clearIntent();
            markDirty();
            requestResolve({ selection: state.draftSelection });
        });
        controls.appendChild(btn);
    });
    // Custom context: a real control (distinct from the derived Custom intent
    // indicator).
    const custom = el('label', 'bundle-ctx-custom');
    const customLabel = el('span', null, 'Custom:');
    const customInput = el('input', 'bundle-ctx-custom-input');
    customInput.type = 'number';
    customInput.min = '1';
    customInput.placeholder = 'tokens';
    const isCustom = !(bundle?.context_options || []).includes(Number(state.draftSelection?.context_size));
    customInput.value = isCustom ? String(state.draftSelection?.context_size ?? '') : '';
    customInput.addEventListener('input', () => {
        const v = Number(customInput.value);
        if (Number.isInteger(v) && v > 0) {
            state.draftSelection.context_size = v;
            clearIntent();
            markDirty();
            requestResolve({ selection: state.draftSelection });
        }
    });
    custom.appendChild(customLabel);
    custom.appendChild(customInput);
    controls.appendChild(custom);
}

function renderKv(d, bundle, reasons) {
    const wrap = d.body.querySelector('.bundle-row-kv .bundle-radio-group');
    wrap.textContent = '';
    (bundle?.kv_policy_options || []).forEach(policy => {
        const id = `bundle-kv-${policy}`;
        const label = el('label', 'bundle-radio-label');
        const input = el('input', 'bundle-kv');
        input.type = 'radio';
        input.name = 'bundle-kv';
        input.value = policy;
        input.id = id;
        input.checked = state.draftSelection?.kv_policy === policy;
        const reason = reasonFor(state.bundleId, reasons, 'kv_policy', policy);
        if (reason) {
            input.disabled = true;
            const reasonSpan = el('span', 'bundle-option-reason', reason);
            reasonSpan.id = `${id}-reason`;
            input.setAttribute('aria-describedby', reasonSpan.id);
            label.appendChild(reasonSpan);
        }
        label.appendChild(input);
        label.appendChild(el('span', 'bundle-radio-name', KV_LABELS[policy] || policy.replace(/_/g, ' / ')));
        input.addEventListener('change', () => {
            if (input.disabled) return;
            state.draftSelection.kv_policy = policy;
            clearIntent();
            markDirty();
            requestResolve({ selection: state.draftSelection });
        });
        wrap.appendChild(label);
    });
}

function renderPerf(d, bundle) {
    const controls = d.body.querySelector('.bundle-row-perf .bundle-controls');
    controls.textContent = '';
    (bundle?.performance_options || []).forEach(option => {
        const btn = el('button', 'bundle-perf', option.label || `${option.batch_size} / ${option.ubatch_size}`);
        btn.type = 'button';
        btn.dataset.perf = option.id;
        setActiveIndicator(btn, state.draftSelection?.performance_id === option.id);
        btn.addEventListener('click', () => {
            state.draftSelection.performance_id = option.id;
            const perf = (bundle?.performance_options || []).find(o => o.id === option.id);
            if (perf) {
                state.draftSelection.batch_size = perf.batch_size;
                state.draftSelection.ubatch_size = perf.ubatch_size;
            }
            clearIntent();
            markDirty();
            requestResolve({ selection: state.draftSelection });
        });
        controls.appendChild(btn);
    });
    const custom = el('span', 'bundle-deriv-custom', 'Custom');
    custom.dataset.slot = 'perf';
    controls.appendChild(custom);
    const isCustom = !(bundle?.performance_options || []).some(o => o.id === state.draftSelection?.performance_id);
    showDerivedCustom(custom, isCustom);
    const current = d.body.querySelector('.bundle-perf-current');
    current.textContent = `Current: batch ${state.draftSelection?.batch_size ?? '—'} · ubatch ${state.draftSelection?.ubatch_size ?? '—'}`;
}

function renderMoe(d, bundle, reasons, preview) {
    const row = d.body.querySelector('.bundle-row-moe');
    const sel = state.draftSelection || {};
    const artifact = selectedArtifact(bundle, sel);
    const moe = isMoeArtifact(artifact);
    row.hidden = !moe;
    if (!moe) return;

    const total = moeLayerCount(bundle, sel);
    row.querySelector('.bundle-moe-total').textContent = ` / ${total}`;
    const input = row.querySelector('.bundle-moe-custom-input');
    input.max = String(total);
    row.querySelector('.bundle-moe-custom-label').textContent = 'Custom:';

    row.querySelectorAll('button.bundle-moe').forEach(btn => {
        const isFit = btn.dataset.moe === 'fit';
        const reason = isFit ? reasonFor(state.bundleId, reasons, 'fit') : null;
        if (isFit && reason) {
            btn.disabled = true;
            btn.id = btn.id || 'bundle-moe-fit';
            const reasonSpan = el('span', 'bundle-option-reason', reason);
            reasonSpan.id = `${btn.id}-reason`;
            btn.setAttribute('aria-describedby', reasonSpan.id);
            if (!row.querySelector(`#${reasonSpan.id}`)) row.querySelector('.bundle-moe-probe').appendChild(reasonSpan);
        } else {
            btn.disabled = false;
        }
        setActiveIndicator(btn, isFit
            ? false
            : Number(sel.n_cpu_moe ?? 0) === 0 && btn.dataset.moe === '0');
        if (!btn.dataset.wired) {
            btn.dataset.wired = '1';
            btn.addEventListener('click', () => {
                if (btn.disabled) return;
                if (btn.dataset.moe === '0') {
                    state.draftSelection.n_cpu_moe = 0;
                    state.probeStatus = 'idle';
                    clearIntent();
                    markDirty();
                    requestResolve({ selection: state.draftSelection });
                } else {
                    // Fit automatically: a bounded two-sided probe search that
                    // returns a draft proposal. Never auto-applies or persists.
                    state.probeStatus = 'searching';
                    renderMoeProbe();
                    requestResolve({ selection: state.draftSelection, fitAutomatically: true, fitTargetMib: state.fitTargetMib ?? null });
                }
            });
        }
    });

    if (input.dataset.wired !== '1') {
        input.dataset.wired = '1';
        input.addEventListener('input', () => {
            let v = Number(input.value);
            if (!Number.isFinite(v)) return;
            v = Math.max(0, Math.min(total, Math.round(v)));
            state.draftSelection.n_cpu_moe = v;
            clearIntent();
            markDirty();
            // Manual n_cpu_moe: a single-point probe at the chosen value, not a
            // search.
            state.probeStatus = 'measuring';
            requestResolve({ selection: state.draftSelection });
        });
    }
    const customValue = Number(sel.n_cpu_moe ?? 0);
    input.value = customValue > 0 ? String(customValue) : '';

    const headroom = row.querySelector('.bundle-moe-headroom');
    if (headroom && document.activeElement !== headroom) {
        headroom.value = state.fitTargetMib != null ? String(miBToGb(state.fitTargetMib)) : '';
    }
    renderMoeProbe();
}

function renderMoeProbe() {
    const row = drawer().body.querySelector('.bundle-row-moe');
    if (!row || row.hidden) return;
    const probe = row.querySelector('.bundle-moe-probe');
    const preview = state.normalizedPreview || {};
    const est = estimateBody(preview.estimate);

    probe.textContent = '';
    if (state.probeStatus === 'searching') {
        probe.appendChild(el('span', 'bundle-moe-probe-status', 'Searching for the best fit…'));
        return;
    }
    if (est && Number(est.n_cpu_moe ?? state.draftSelection?.n_cpu_moe ?? 0) > 0 || est) {
        const bits = [];
        if (est.probe_device_total_mib != null) bits.push(`Device VRAM ${miBToGb(est.probe_device_total_mib)} GB`);
        if (est.ram_bytes != null && Number(est.ram_bytes) > 0) bits.push(`experts in system RAM ${gb(est.ram_bytes)} GB`);
        if (est.ram_headroom_bytes != null) bits.push(`RAM headroom ${gb(Math.abs(est.ram_headroom_bytes))} GB${est.ram_headroom_bytes < 0 ? ' over budget' : ''}`);
        if (est.headroom_bytes != null) bits.push(`device headroom ${gb(est.headroom_bytes)} GB`);
        if (bits.length) probe.appendChild(el('span', 'bundle-moe-probe-status', bits.join(' · ')));
    }
}

function estimateBody(estimate) {
    if (!estimate) return null;
    if (estimate.status === 'available') return estimate.estimate || null;
    return null;
}

function renderWorkload(d, bundle) {
    const select = d.body.querySelector('.bundle-workload');
    select.value = state.draftSelection?.workload_policy || bundle?.workload_policy || '';
    if (select.dataset.wired !== '1') {
        select.dataset.wired = '1';
        select.addEventListener('change', () => {
            // Workload changes aggressive-KV eligibility only through the
            // resolver; it is part of the draft and shows up in the diff.
            state.draftSelection.workload_policy = select.value;
            clearIntent();
            markDirty();
            requestResolve({ selection: state.draftSelection });
        });
    }
}

// Evidence class is the resolver's own EvidenceMatch.class (architecture 12:
// exact/compatible/related/stale). Render it verbatim — never upgrade a
// compatible/related/stale match to exact.
const EVIDENCE_LABELS = {
    exact: 'Measured on this machine',
    compatible: 'Compatible model evidence',
    related: 'Related model evidence',
    stale: 'Stale evidence — reverify before trusting',
};

function renderEvidence(target, evidence) {
    target.className = 'bundle-evidence';
    target.replaceChildren();
    if (!evidence || !evidence.class) {
        target.hidden = true;
        return;
    }
    target.hidden = false;
    target.classList.add(`bundle-evidence--${evidence.class}`);
    const label = EVIDENCE_LABELS[evidence.class] || evidence.class;
    const text = document.createElement('span');
    text.className = 'bundle-evidence-text';
    text.textContent = evidence.summary ? `${label} — ${evidence.summary}` : label;
    target.appendChild(text);
    if (evidence.detail) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'bundle-evidence-details';
        btn.dataset.bundleEvidenceDetails = '1';
        btn.textContent = 'Details';
        target.appendChild(btn);
    }
}

function renderResult(d, bundle) {
    const preview = state.normalizedPreview || {};
    const est = estimateBody(preview.estimate);
    const evidenceEl = d.body.querySelector('.bundle-evidence');
    const line = d.body.querySelector('.bundle-result-line');
    const verdict = d.body.querySelector('.bundle-result-verdict');
    const moeWarn = d.body.querySelector('.bundle-result-moe');
    renderEvidence(evidenceEl, preview.evidence);
    line.textContent = '';
    verdict.textContent = '';
    moeWarn.textContent = '';
    moeWarn.hidden = true;

    if (!est) {
        if (preview.estimate && preview.estimate.status === 'unavailable') {
            verdict.textContent = preview.estimate.message || 'Estimate unavailable.';
        } else if (preview.estimate && preview.estimate.status === 'not_applicable') {
            verdict.textContent = 'Estimate not applicable.';
        } else {
            verdict.textContent = 'No preview yet.';
        }
    } else {
        const weights = est.weights_bytes ?? 0;
        const kv = est.kv_cache_bytes ?? 0;
        const total = est.total_bytes ?? (weights + kv);
        const compute = Math.max(0, total - weights - kv);
        line.textContent = `Weights ${gb(weights)} · KV ${gb(kv)} · Compute ${gb(compute)} = ${gb(total)} GB`;
        verdict.textContent = est.headroom_bytes >= 0
            ? `Fits with ${gb(est.headroom_bytes)} GB to spare`
            : `Needs ${gb(-est.headroom_bytes)} GB more — free cache or lower context`;

        const sel = state.draftSelection || {};
        const n = Number(sel.n_cpu_moe ?? 0);
        if (isMoeArtifact(selectedArtifact(bundle, sel)) && n > 0) {
            // Qualitative only: the magnitude varies with model, RAM, GPU
            // bandwidth and load, and there is no honest number to show.
            moeWarn.textContent = `⚠ ${n} expert layers on CPU — slower generation`;
            moeWarn.hidden = false;
        }
    }
}

function diffFromSaved() {
    const bundle = currentBundle();
    const saved = state.savedSelection || {};
    const draft = state.draftSelection || {};
    const rows = [];
    if (saved.artifact_id !== draft.artifact_id) {
        const bundle = currentBundle();
        const a = (bundle?.artifacts || []).find(x => x.id === saved.artifact_id)?.quantization?.value || saved.artifact_id;
        const b = (bundle?.artifacts || []).find(x => x.id === draft.artifact_id)?.quantization?.value || draft.artifact_id;
        rows.push(`${String(a).toUpperCase()} → ${String(b).toUpperCase()}`);
    }
    if (saved.context_size !== draft.context_size) rows.push(`${ctxLabel(saved.context_size)} → ${ctxLabel(draft.context_size)} context`);
    if (saved.kv_policy !== draft.kv_policy) rows.push(`KV ${String(saved.kv_policy).replace(/_/g, ' / ')} → ${String(draft.kv_policy).replace(/_/g, ' / ')}`);
    if (saved.performance_id !== draft.performance_id) {
        const bundle = currentBundle();
        const a = (bundle?.performance_options || []).find(o => o.id === saved.performance_id)?.label || saved.performance_id;
        const b = (bundle?.performance_options || []).find(o => o.id === draft.performance_id)?.label || draft.performance_id;
        rows.push(`perf ${a} → ${b}`);
    }
    if ((saved.n_cpu_moe ?? 0) !== (draft.n_cpu_moe ?? 0)) rows.push(`experts on CPU ${saved.n_cpu_moe ?? 0} → ${draft.n_cpu_moe ?? 0}`);
    const savedWorkload = saved.workload_policy || bundle?.workload_policy || '';
    const draftWorkload = draft.workload_policy || savedWorkload;
    if (savedWorkload !== draftWorkload) {
        rows.push(`workload ${(WORKLOAD_POLICIES.find(w => w.value === savedWorkload) || {}).label || savedWorkload} → ${(WORKLOAD_POLICIES.find(w => w.value === draftWorkload) || {}).label || draftWorkload}`);
    }
    return rows;
}

function renderDiff(d, bundle) {
    const list = d.body.querySelector('.bundle-diff-list');
    const cause = d.body.querySelector('.bundle-diff-cause');
    list.textContent = '';
    cause.textContent = '';
    const changes = (state.normalizedPreview?.changes) || [];
    const structural = diffFromSaved();
    // The resolver's own changes are the authority for what changed; fall back
    // to a structural diff so manual edits are reviewable too. Workload policy
    // is bundle-level rather than a launch field, so it is intentionally not
    // present in resolver changes and must remain visible beside them.
    const items = changes.length
        ? [
            ...changes.map(c => `${c.field}: ${c.before ?? '—'} → ${c.after}`),
            ...structural.filter(item => item.startsWith('workload ')),
        ]
        : structural;
    if (!items.length) {
        d.body.querySelector('.bundle-row-diff').hidden = true;
        cause.hidden = true;
        return;
    }
    d.body.querySelector('.bundle-row-diff').hidden = false;
    items.forEach(text => list.appendChild(el('li', null, text)));
    const intent = INTENTS.find(i => i.value === state.activeIntent);
    if (intent) {
        cause.textContent = `(applied by “${intent.label}”)`;
        cause.hidden = false;
    } else {
        cause.hidden = true;
    }
}

function renderDirty(d) {
    const dirty = state.dirty;
    d.root.classList.toggle('is-dirty', dirty);
    d.panel.querySelectorAll('.bundle-edit-full,.bundle-reset').forEach(b => b.disabled = false);
    d.panel.querySelector('.bundle-reset').disabled = !dirty;
}

// ── Dirty / close lifecycle ──────────────────────────────────────────────────
function clearIntent() { state.activeIntent = null; }
function markDirty() { state.dirty = true; renderDirty(drawer()); }

function resetDraft() {
    state.draftSelection = cloneSelection(state.savedSelection);
    state.activeIntent = null;
    state.probeStatus = 'idle';
    state.fitTargetMib = null;
    state.dirty = false;
    renderAll();
    requestResolve({ selection: state.draftSelection });
}

function confirmDiscard(message) {
    // Only explicit Reset discards without prompting; every other exit that
    // would abandon a dirty draft must confirm first.
    if (!state.dirty) return true;
    if (typeof window.confirm !== 'function') return true;
    return window.confirm(message);
}

async function reloadFromServer() {
    await loadPresets(state.presetId);
    const updated = (sessionState.presets || []).find(p => p.id === state.presetId);
    if (!updated?.bundle) return;
    state.serverRevision = updated.revision ?? state.serverRevision;
    state.savedSelection = cloneSelection(updated.bundle.default_selection || {});
    state.draftSelection = cloneSelection(state.savedSelection);
    state.normalizedPreview = null;
    state.activeIntent = state.savedSelection.intent_source || null;
    state.probeStatus = 'idle';
    state.fitTargetMib = null;
    state.dirty = false;
    renderAll();
    requestResolve({ selection: state.draftSelection });
}

function showRevisionConflict(message) {
    showToastWithActions(
        'Preset changed elsewhere',
        'warning',
        message,
        [{ id: 'reload', label: 'Reload', primary: true, handler: () => { void reloadFromServer(); } }],
    );
}

function requestClose() {
    if (!confirmDiscard('Discard unsaved changes?')) return false;
    closeDrawer();
    return true;
}

function closeDrawer(opener) {
    const d = drawer();
    // Discard the draft: closing never persists.
    state.previewRequestGeneration += 1;
    const focusTarget = opener !== undefined ? opener : state.opener;
    if (state.previewAbortController) {
        try { state.previewAbortController.abort(); } catch { /* ignore */ }
        state.previewAbortController = null;
    }
    d.root.classList.remove('open');
    const finish = () => {
        d.root.hidden = true;
        state.dirty = false;
        state.draftSelection = null;
        state.savedSelection = null;
        state.normalizedPreview = null;
        if (focusTarget && typeof focusTarget.focus === 'function') focusTarget.focus();
        state.opener = null;
        document.removeEventListener('keydown', onKeydown, true);
        _keydownBound = false;
    };
    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        finish();
    } else {
        setTimeout(finish, 180);
    }
}

let _keydownBound = false;
function onKeydown(event) {
    if (event.key === 'Escape') {
        event.preventDefault();
        event.stopPropagation();
        requestClose();
        return;
    }
    if (event.key === 'Tab') {
        const d = drawer();
        const focusables = Array.from(
            d.panel.querySelectorAll('button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])'),
        ).filter(node => node.offsetParent !== null);
        if (!focusables.length) return;
        const first = focusables[0];
        const last = focusables[focusables.length - 1];
        if (event.shiftKey && (document.activeElement === first || document.activeElement === d.panel)) {
            event.preventDefault();
            last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
            event.preventDefault();
            first.focus();
        }
    }
}

// ── Exit actions ─────────────────────────────────────────────────────────────
async function spawnDraft(selection) {
    const payload = {
        preset_id: state.presetId,
        selection: { ...selection },
        expected_revision: state.serverRevision,
    };
    if (state.normalizedPreview?.resolved_config_hash) {
        payload.expected_resolved_config_hash = state.normalizedPreview.resolved_config_hash;
    }
    const resp = await fetch('/api/sessions/spawn', {
        method: 'POST',
        headers: apiHeaders(true),
        body: JSON.stringify(payload),
    });
    const data = await resp.json().catch(() => null);
    return { status: resp.status, data };
}

async function saveSelection() {
    const selection = { ...state.draftSelection };
    const workloadPolicy = selection.workload_policy;
    delete selection.workload_policy;
    const resp = await fetch(`/api/presets/${encodeURIComponent(state.presetId)}/selection`, {
        method: 'PATCH',
        headers: apiHeaders(true),
        body: JSON.stringify({
            expected_revision: state.serverRevision,
            selection,
            ...(workloadPolicy ? { workload_policy: workloadPolicy } : {}),
        }),
    });
    const data = await resp.json().catch(() => null);
    return { status: resp.status, data };
}

async function handleStartDraft() {
    // Start without saving: pass the normalized draft to the authenticated
    // spawn route but never mutate the preset.
    const selection = state.normalizedPreview?.selection || cloneSelection(state.draftSelection);
    const { status, data } = await spawnDraft(selection);
    if (status === 409) {
        showRevisionConflict('Reload the current selection before starting this draft.');
        return;
    }
    if (status === 200) {
        closeDrawer();
        return;
    }
    showToast(data?.error || 'Start failed.', 'error');
}

async function handleSave() {
    const { status, data } = await saveSelection();
    if (status === 409) {
        showRevisionConflict('Reload the latest settings before saving again.');
        return;
    }
    if (status !== 200) {
        showToast(data?.error || 'Save failed.', 'error');
        return;
    }
    // Save merges only the server-returned preset/revision via the canonical
    // reload path. loadPresets re-fetches and re-renders the launch grid.
    await loadPresets(state.presetId);
    const updated = (sessionState.presets || []).find(p => p.id === state.presetId);
    if (updated) {
        state.serverRevision = updated.revision ?? state.serverRevision;
        state.savedSelection = cloneSelection(updated.bundle?.default_selection || state.savedSelection);
        state.draftSelection = cloneSelection(state.savedSelection);
        state.dirty = false;
        state.activeIntent = null;
        renderAll();
    }
}

async function handleSaveAndStart() {
    const { status, data } = await saveSelection();
    if (status === 409) {
        showRevisionConflict('Reload the latest settings before saving and starting again.');
        return;
    }
    if (status !== 200) {
        // A failed save must not launch a different or stale selection.
        showToast(data?.error || 'Save failed — nothing was started.', 'error');
        return;
    }
    // Persisted successfully: launch the exact revision the PATCH returned.
    const savedRevision = data?.revision ?? state.serverRevision;
    const selection = cloneSelection(state.draftSelection);
    state.serverRevision = savedRevision;
    const spawn = await spawnDraft(selection);
    if (spawn.status === 409) {
        showRevisionConflict('Reload the latest settings before starting again.');
        return;
    }
    if (spawn.status === 200) {
        closeDrawer();
        return;
    }
    showToast(spawn.data?.error || 'Save succeeded but the launch failed.', 'error');
}

function showToast(message, kind = 'info') {
    // Reuse the app's toast if present; otherwise fall back to console so the
    // drawer never blocks on a missing host.
    try {
        if (typeof window.showToast === 'function') {
            window.showToast(message, kind);
            return;
        }
    } catch { /* ignore */ }
    console.warn(`[bundle-drawer] ${message}`);
}

// ── Public entry point ───────────────────────────────────────────────────────
function openBundleDrawer(preset, opener) {
    if (!preset || !preset.bundle) return;
    const d = drawer();

    state.presetId = preset.id;
    state.bundleId = preset.bundle.identity?.bundle_id || preset.id;
    state.serverRevision = preset.revision ?? 0;
    state.savedSelection = cloneSelection(preset.bundle.default_selection || {});
    if (typeof preset.bundle.workload_policy === 'string') {
        state.savedSelection.workload_policy = preset.bundle.workload_policy;
    }
    state.draftSelection = cloneSelection(state.savedSelection);
    state.normalizedPreview = null;
    state.dirty = false;
    state.activeIntent = state.savedSelection.intent_source || null;
    state.probeStatus = 'idle';
    state.fitTargetMib = null;
    state.previewRequestGeneration = 0;
    state.opener = opener || null;

    d.title.textContent = `Configure ${preset.bundle.identity?.display_name || preset.name}`;
    d.root.hidden = false;
    renderAll();
    d.root.classList.add('open');
    d.close.focus();

    if (!_keydownBound) {
        document.addEventListener('keydown', onKeydown, true);
        _keydownBound = true;
    }
    bindStaticHandlers();
    requestResolve({ selection: state.draftSelection });
}

function bindStaticHandlers() {
    const d = drawer();
    if (d.root.dataset.bound === '1') return;
    d.root.dataset.bound = '1';

    d.root.addEventListener('click', event => {
        if (event.target.closest('[data-bundle-close]')) {
            requestClose();
        }
        const detailsBtn = event.target.closest('[data-bundle-evidence-details]');
        if (detailsBtn) {
            const evidence = state.normalizedPreview?.evidence;
            if (evidence) openEvidenceDrawer(evidenceFromLaunchObservation(evidence), detailsBtn);
        }
    });

    d.body.querySelector('.bundle-row-intent .bundle-controls').addEventListener('click', event => {
        const btn = event.target.closest('button.bundle-intent');
        if (!btn || btn.disabled) return;
        // Applying an intent issues a resolve that returns the normalized
        // selection; the drawer adopts it and keeps the intent attributed.
        state.activeIntent = btn.dataset.intent;
        state.probeStatus = 'idle';
        markDirty();
        requestResolve({
            selection: { ...state.draftSelection, intent_source: btn.dataset.intent },
        });
        renderIntentRow(d);
    });

    d.panel.querySelector('.bundle-reset').addEventListener('click', () => {
        // Explicit Reset is the only zero-friction path back to the saved
        // selection.
        resetDraft();
    });

    d.panel.querySelector('.bundle-start-draft').addEventListener('click', () => handleStartDraft());
    d.panel.querySelector('.bundle-save').addEventListener('click', () => handleSave());
    d.panel.querySelector('.bundle-save-start').addEventListener('click', () => handleSaveAndStart());

    d.panel.querySelector('.bundle-edit-full').addEventListener('click', () => {
        // Leaving for the full editor still abandons the draft, so it must
        // confirm on a dirty draft like any other non-Reset exit.
        if (!confirmDiscard('Discard unsaved changes and edit the full preset?')) return;
        syncSelectedPresetSelection(state.presetId, { userIntent: true, persist: true });
        openPresetModal('edit');
        closeDrawer(null);
    });

    // Headroom / VRAM buffer: re-runs the two-sided probe search against a
    // larger reserve. Wired lazily if a host adds the control.
    const headroom = d.body.querySelector('.bundle-moe-headroom');
    if (headroom && headroom.dataset.wired !== '1') {
        headroom.dataset.wired = '1';
        headroom.addEventListener('input', () => {
            const v = Number(headroom.value);
            state.fitTargetMib = Number.isFinite(v) && v > 0 ? Math.round(v * 1024) : null;
            markDirty();
            requestResolve({ selection: state.draftSelection, fitAutomatically: true, fitTargetMib: state.fitTargetMib });
        });
    }
}

function renderAll() {
    renderIntentRow(drawer());
    renderFromPreview();
}

export { openBundleDrawer };
export { openBundleDrawer as default };
