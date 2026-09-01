// ── Presets ────────────────────────────────────────────────────────────────────
/* global DOMPurify */
// Preset CRUD: load, save, copy, delete, reset. Modal management.

import { sessionState, lastSystemMetrics } from '../core/app-state.js';
import { getPlatformInfo } from '../core/platform-info.js';
import { escapeHtml } from '../core/format.js';
import { buildArchitectureLabel, isMoEEligible } from './setup-view.js';
import { openModelFileBrowser, openChatTemplateLibraryBrowser, uploadChatTemplateFromBrowser } from './file-browser-launcher.js';
import { configureMlxPresetEditor } from './preset-editor-mlx.js';
import { applySettings, saveSettings } from './settings.js';
import { showToast, showToastWithActions, showConfirmDialog } from './toast.js';
import { renderSuggestionCards, suggestionPatch, requestNcpuMoeTune } from './tuning-cards.js';
import {
    buildCommunityTemplateInstallRequest,
    communityFamilyFromGgufArchitecture,
    communityTemplateFamilyFor,
    getDefaultTemplateForFamily,
} from './chat-template-registry.js';
import { buildEstimateBody, rapidEstimatePolicyFromConfig } from './vram-estimate.js';
import {
    RAPID_MLX_DEFAULT_SPECULATIVE_TOKENS,
    rapidMlxPrefillStepSizeDefault,
    rapidMlxProfileHasVision,
} from './rapid-mlx-prefill.js';
import {
    findRapidMlxSidecarForTrunk,
    rapidMlxSidecarProvenance,
} from '../core/rapid-mlx-sidecars.js';
import { openEstimateEvidenceDrawer } from './evidence-drawer.js';
import { initCalibrationUi } from './calibration.js';
import { FIELD_CATALOG } from './spawn-wizard-groups.js';
import {
    chatTemplateStatusText,
    openChatTemplateManageModal,
    bindChatTemplateManageModalChrome,
    fetchReleases,
} from './chat-template-panel.js';


let newPresetSeed = null;
let _presetRapidMlxPrefillExplicit = false;
let _presetEditorNavInitialized = false;
let _llamaCapabilitiesPromise = null;
let _presetBundleDraft = null;

// ── Helpers ────────────────────────────────────────────────────────────────────

function setVal(id, v) { document.getElementById(id).value = v ?? ''; }
function setChk(id, v) { document.getElementById(id).checked = !!v; }
function setOpt(id, v) { document.getElementById(id).value = v || ''; }
function formatNumberForInput(v) {
    if (v == null || v === '') return '';
    const n = typeof v === 'number' ? v : Number(v);
    if (!Number.isFinite(n)) return String(v);
    if (Number.isInteger(n)) return String(n);
    // toPrecision(6) removes float32 noise (e.g. 0.949999988079 → 0.95)
    return String(parseFloat(n.toPrecision(6)));
}
function numOrEmpty(id, v) { document.getElementById(id).value = formatNumberForInput(v); }
function intOrNull(id) { const v = document.getElementById(id).value; return v !== '' ? parseInt(v) : null; }
function floatOrNull(id) { const v = document.getElementById(id).value; return v !== '' ? parseFloat(v) : null; }
function strVal(id) { return document.getElementById(id).value.trim(); }
function valOrNull(id) { const v = strVal(id); return v === '' ? null : v; }
function nullableBoolOpt(id) {
    const v = document.getElementById(id).value;
    if (v === 'true') return true;
    if (v === 'false') return false;
    return null;
}

function capabilityAvailable(state) {
    return state === 'Available' || (state && Object.prototype.hasOwnProperty.call(state, 'Available'));
}

function capabilityReason(state, fallback) {
    if (typeof state === 'string' && state !== 'Available') return state;
    if (state?.Unavailable) return state.Unavailable;
    return fallback;
}

async function loadLlamaCapabilities() {
    if (_llamaCapabilitiesPromise) return _llamaCapabilitiesPromise;
    _llamaCapabilitiesPromise = (async () => {
        try {
            const headers = window.authHeaders ? window.authHeaders() : {};
            const response = await fetch('/api/llama-binary/capabilities', { headers });
            if (!response.ok) return null;
            const data = await response.json();
            return data?.ok ? data.snapshot : null;
        } catch {
            return null;
        }
    })();
    return _llamaCapabilitiesPromise;
}

function replaceOptions(select, options, selected) {
    if (!select) return;
    select.replaceChildren(...options.map(option => {
        const el = document.createElement('option');
        if (option.separator) {
            el.disabled = true;
            el.textContent = '────────';
        } else {
            el.value = option.value;
            el.textContent = option.label || option.value;
            el.disabled = !!option.disabled;
            if (option.title) el.title = option.title;
        }
        return el;
    }));
    if (selected != null && [...select.options].some(option => option.value === String(selected))) {
        select.value = String(selected);
    }
}

function configureCapabilityFields(snapshot, preset) {
    const typed = snapshot?.typed || {};
    const cacheValues = snapshot?.cache?.kv_type_values || [];
    const editorId = key => FIELD_CATALOG.find(field => field.presetKey === key)?.editorId;
    const storedCtk = preset?.ctk || document.getElementById(editorId('ctk') || 'modal-ctk')?.value || 'q8_0';
    const storedCtv = preset?.ctv || document.getElementById(editorId('ctv') || 'modal-ctv')?.value || 'f16';
    const advertised = new Set(cacheValues);
    const common = ['f16', 'q8_0', 'q4_0'];
    const buildKvOptions = (storedValue) => {
        const kvValues = [...new Set([...common, ...cacheValues, storedValue])];
        const options = kvValues.map(value => ({
        value,
        disabled: advertised.size > 0 && !advertised.has(value),
        title: advertised.size > 0 && !advertised.has(value) ? 'Not advertised by the selected llama.cpp binary' : '',
        }));
        options.splice(common.length, 0, { separator: true });
        return options;
    };
    replaceOptions(document.getElementById(editorId('ctk') || 'modal-ctk'), buildKvOptions(storedCtk), storedCtk);
    replaceOptions(document.getElementById(editorId('ctv') || 'modal-ctv'), buildKvOptions(storedCtv), storedCtv);

    const effort = typed.reasoning_effort || {};
    const effortValues = ['minimal', 'low', 'medium', 'high', 'xhigh', 'max'];
    const effortSelected = preset?.llama_reasoning_effort || 'default';
    replaceOptions(document.getElementById('modal-llama-reasoning-effort'), [
        { value: 'default', label: 'Default' },
        ...effortValues.map(value => ({
            value,
            disabled: !capabilityAvailable(effort.supported) || (effort.accepted_values?.length > 0 && !effort.accepted_values.includes(value)),
            title: !capabilityAvailable(effort.supported) ? capabilityReason(effort.supported, 'Not advertised by this binary') : '',
        })),
        ...(String(effortSelected) !== 'default' && !effortValues.includes(String(effortSelected)) ? [{ value: effortSelected, label: `${effortSelected} (stored; unsupported)`, disabled: true }] : []),
    ], effortSelected);
    const effortHint = document.getElementById('modal-llama-reasoning-effort-hint');
    if (effortHint) effortHint.textContent = capabilityAvailable(effort.supported) ? '' : capabilityReason(effort.supported, 'This binary does not advertise --reasoning-effort.');

    const format = typed.reasoning_format || {};
    const formatValues = ['none', 'deepseek', 'deepseek-legacy'];
    const formatSelected = preset?.llama_reasoning_format || '';
    replaceOptions(document.getElementById('modal-llama-reasoning-format'), [
        { value: '', label: 'Default (auto)' },
        ...formatValues.map(value => ({
            value,
            disabled: !capabilityAvailable(format.supported) || (format.accepted_values?.length > 0 && !format.accepted_values.includes(value)),
            title: !capabilityAvailable(format.supported) ? capabilityReason(format.supported, 'Not advertised by this binary') : '',
        })),
        ...(formatSelected && !formatValues.includes(String(formatSelected)) ? [{ value: formatSelected, label: `${formatSelected} (stored; unsupported)`, disabled: true }] : []),
    ], formatSelected);
    const formatHint = document.getElementById('modal-llama-reasoning-format-hint');
    if (formatHint) formatHint.textContent = capabilityAvailable(format.supported) ? '' : capabilityReason(format.supported, 'This binary does not advertise --reasoning-format.');

    const preserve = typed.reasoning_preserve || {};
    const preserveSelect = document.getElementById('modal-llama-reasoning-preserve');
    if (preserveSelect) {
        for (const option of preserveSelect.options) {
            if (option.value === 'true') option.disabled = !capabilityAvailable(preserve.positive);
            if (option.value === 'false') option.disabled = !capabilityAvailable(preserve.negative);
        }
    }
    const preserveHint = document.getElementById('modal-llama-reasoning-preserve-hint');
    if (preserveHint && !capabilityAvailable(preserve.positive) && !capabilityAvailable(preserve.negative)) {
        preserveHint.textContent = capabilityReason(preserve.positive, 'This binary does not advertise native reasoning preservation.');
    } else if (preserveHint) preserveHint.textContent = '';

    const mmproj = typed.mmproj_offload || {};
    const mmprojSelect = document.getElementById('modal-mmproj-offload');
    if (mmprojSelect) {
        for (const option of mmprojSelect.options) {
            if (option.value === 'true') option.disabled = !capabilityAvailable(mmproj.positive);
            if (option.value === 'false') option.disabled = !capabilityAvailable(mmproj.negative);
        }
    }
    const mmprojHint = document.getElementById('modal-mmproj-offload-hint');
    if (mmprojHint && !capabilityAvailable(mmproj.positive) && !capabilityAvailable(mmproj.negative)) {
        mmprojHint.textContent = capabilityReason(mmproj.positive, 'This binary does not advertise projector offload controls.');
    } else if (mmprojHint) mmprojHint.textContent = '';
}

function getStructuredOutputMode() {
    return document.getElementById('modal-structured-output-mode')?.value || '';
}

function setStructuredOutputMode(mode) {
    const normalized = mode === 'grammar' || mode === 'json_schema' ? mode : '';
    const modeEl = document.getElementById('modal-structured-output-mode');
    const grammarWrap = document.getElementById('modal-grammar-wrap');
    const schemaWrap = document.getElementById('modal-json-schema-wrap');
    if (modeEl) modeEl.value = normalized;
    if (grammarWrap) grammarWrap.style.display = normalized === 'grammar' ? '' : 'none';
    if (schemaWrap) schemaWrap.style.display = normalized === 'json_schema' ? '' : 'none';
}

function isRunningStatus(status) {
    return String(status || '').toLowerCase() === 'running';
}

export function presetModelSource(preset) {
    const rapidMlx = preset?.rapid_mlx;
    if (preset?.backend === 'rapid_mlx') {
        const ms = rapidMlx?.model_source;
        if (ms?.kind === 'hugging_face_repo') {
            return ms.repo_id || '';
        }
        if (ms?.kind === 'mlx_directory') {
            return ms.path || '';
        }
        if (ms?.kind === 'alias') {
            return ms.value || '';
        }
        return rapidMlx?.model_source_view?.canonical_identity
            || rapidMlx?.model_source_view?.display_name || '';
    }
    return preset?.model_path || preset?.hf_repo || '';
}

// Same lookup `savePreset`/`_buildFormPreset` use to find the preset currently
// loaded in the modal — needed so VRAM estimates route to the right backend.
function _currentModalPreset() {
    const id = document.getElementById('modal-preset-id')?.value;
    return id ? (sessionState.presets.find(p => p.id === id) || {}) : (newPresetSeed || {});
}

function bundleClone(value) {
    return value ? structuredClone(value) : null;
}

function bundleArtifactLabel(artifact) {
    const quant = artifact.quantization?.value ? ` · ${artifact.quantization.value}` : '';
    const role = artifact.role || 'weights';
    return `${artifact.display_name || artifact.local_path || artifact.id || 'Artifact'}${quant} · ${role}`;
}

function bundleMetadataFromResponse(data) {
    const modelKind = data.expert_count || data.expert_used_count ? 'moe' : 'dense';
    return {
        gguf_architecture: data.architecture || null,
        model_kind: modelKind,
        block_count: data.block_count || null,
        moe_layer_count: data.expert_count ? (data.block_count || null) : null,
        native_context_limit: data.context_length || null,
        metadata_digest: null,
    };
}

function quantizationHint(path) {
    const match = path.match(/(?:^|[-_.])(q(?:2|3|4|5|6|8)(?:[_-](?:k[_-])?(?:s|m|l|xl)|[_-]0|[_-]1)?)(?:[-_.]|$)/i);
    return match ? match[1].toLowerCase().replaceAll('-', '_') : '';
}

function selectedBundleArtifact() {
    if (!_presetBundleDraft) return null;
    return (_presetBundleDraft.artifacts || []).find(a => a.id === _presetBundleDraft.default_selection?.artifact_id) || null;
}

function updateBundleSelectionFromEditor() {
    if (!_presetBundleDraft) return;
    const selected = _presetBundleDraft.default_selection || {};
    const value = id => document.getElementById(id)?.value || '';
    const number = id => {
        const n = Number(value(id));
        return Number.isFinite(n) ? n : null;
    };
    selected.artifact_id = value('modal-bundle-artifact');
    const context = number('modal-bundle-context');
    if (context != null) selected.context_size = context;
    selected.kv_policy = value('modal-bundle-kv') || selected.kv_policy;
    selected.performance_id = value('modal-bundle-performance') || selected.performance_id;
    const moe = number('modal-bundle-cpu-moe');
    selected.n_cpu_moe = moe == null || moe === 0 ? (moe === 0 ? 0 : null) : moe;
    _presetBundleDraft.default_selection = selected;
    if (selected.context_size) setVal('modal-context-size', selected.context_size);
    const kvPair = { f16_f16: ['f16', 'f16'], q8_0_q8_0: ['q8_0', 'q8_0'], q4_0_q4_0: ['q4_0', 'q4_0'], q8_0_q4_0: ['q8_0', 'q4_0'] }[selected.kv_policy];
    if (kvPair) {
        setOpt('modal-ctk', kvPair[0]);
        setOpt('modal-ctv', kvPair[1]);
    }
    const performance = (_presetBundleDraft.performance_options || []).find(option => option.id === selected.performance_id);
    if (performance) {
        setVal('modal-batch-size', performance.batch_size);
        setVal('modal-ubatch-size', performance.ubatch_size);
    }
    setVal('modal-n-cpu-moe', selected.n_cpu_moe ?? '');
    const artifact = selectedBundleArtifact();
    if (artifact?.local_path) setVal('modal-model-path', artifact.local_path);
    if (artifact?.mmproj_artifact_id) {
        const mmproj = _presetBundleDraft.artifacts.find(item => item.id === artifact.mmproj_artifact_id);
        if (mmproj?.local_path) setVal('modal-mmproj', mmproj.local_path);
    }
    if (artifact?.draft_artifact_id) {
        const draft = _presetBundleDraft.artifacts.find(item => item.id === artifact.draft_artifact_id);
        if (draft?.local_path) setVal('modal-draft-model', draft.local_path);
    }
    if (artifact?.metadata?.model_kind) {
        const hint = document.getElementById('modal-bundle-artifact-meta');
        if (hint) hint.textContent = `${artifact.metadata.gguf_architecture || 'Unknown architecture'} · ${artifact.metadata.model_kind}${artifact.metadata.block_count ? ` · ${artifact.metadata.block_count} layers` : ''}`;
    }
}

function renderPresetBundleEditor() {
    const section = document.querySelector('.preset-editor-section[data-section="variants"]');
    if (!section) return;
    const empty = section.querySelector('#preset-bundle-empty');
    const body = section.querySelector('#preset-bundle-editor');
    if (!_presetBundleDraft) {
        if (empty) empty.hidden = false;
        if (body) body.hidden = true;
        return;
    }
    if (empty) empty.hidden = true;
    if (body) body.hidden = false;
    const artifacts = _presetBundleDraft.artifacts || [];
    const artifactList = section.querySelector('#preset-bundle-artifacts');
    if (artifactList) {
        artifactList.replaceChildren(...artifacts.map(artifact => {
            const row = document.createElement('div');
            row.className = 'preset-bundle-artifact';
            row.dataset.artifactId = artifact.id;
            const info = document.createElement('div');
            info.className = 'preset-bundle-artifact-info';
            const title = document.createElement('strong');
            title.textContent = bundleArtifactLabel(artifact);
            const path = document.createElement('span');
            path.textContent = artifact.local_path || 'Not adopted locally';
            info.append(title, path);
            const remove = document.createElement('button');
            remove.type = 'button';
            remove.className = 'btn-sm btn-preset preset-bundle-remove';
            remove.dataset.artifactId = artifact.id;
            remove.textContent = 'Remove';
            remove.disabled = artifact.id === _presetBundleDraft.default_selection?.artifact_id;
            remove.title = remove.disabled ? 'Select a replacement artifact before removing the active one' : 'Remove this artifact';
            row.append(info, remove);
            return row;
        }));
    }
    const weights = artifacts.filter(a => a.role === 'weights');
    const artifactSelect = section.querySelector('#modal-bundle-artifact');
    if (artifactSelect) {
        replaceOptions(artifactSelect, weights.map(a => ({ value: a.id, label: bundleArtifactLabel(a) })), _presetBundleDraft.default_selection?.artifact_id);
    }
    const contextSelect = section.querySelector('#modal-bundle-context');
    if (contextSelect) replaceOptions(contextSelect, (_presetBundleDraft.context_options || []).map(v => ({ value: v, label: `${v.toLocaleString()} tokens` })), _presetBundleDraft.default_selection?.context_size);
    const kvSelect = section.querySelector('#modal-bundle-kv');
    if (kvSelect) replaceOptions(kvSelect, (_presetBundleDraft.kv_policy_options || []).map(v => ({ value: v, label: v.replaceAll('_', ' / ') })), _presetBundleDraft.default_selection?.kv_policy);
    const performanceSelect = section.querySelector('#modal-bundle-performance');
    if (performanceSelect) replaceOptions(performanceSelect, (_presetBundleDraft.performance_options || []).map(option => ({ value: option.id, label: option.label || `${option.batch_size} / ${option.ubatch_size}` })), _presetBundleDraft.default_selection?.performance_id);
    const selected = selectedBundleArtifact();
    const moeWrap = section.querySelector('#preset-bundle-moe-wrap');
    const moeSelect = section.querySelector('#modal-bundle-cpu-moe');
    const isMoe = selected?.metadata?.model_kind === 'moe' || selected?.metadata?.model_kind === 'hybrid_moe';
    if (moeWrap) {
        moeWrap.hidden = !isMoe;
        if (isMoe) moeWrap.style.removeProperty('display');
        else moeWrap.style.setProperty('display', 'none', 'important');
    }
    if (moeSelect && isMoe) replaceOptions(moeSelect, (_presetBundleDraft.cpu_moe_options || []).map(v => ({ value: v, label: v === 0 ? 'All experts on GPU (0)' : `${v} expert layers on CPU` })), _presetBundleDraft.default_selection?.n_cpu_moe ?? 0);
    const meta = section.querySelector('#modal-bundle-artifact-meta');
    if (meta && selected) meta.textContent = `${selected.metadata?.gguf_architecture || 'Unknown architecture'} · ${selected.metadata?.model_kind || 'unknown'}${selected.metadata?.block_count ? ` · ${selected.metadata.block_count} layers` : ''}`;
    updateBundleSelectionFromEditor();
}

async function freshPresetCatalogEtag() {
    const response = await fetch('/api/preset-cards', { headers: window.authHeaders ? window.authHeaders() : {} });
    if (!response.ok) throw new Error(`Catalog unavailable (HTTP ${response.status})`);
    const data = await response.json();
    return data.catalog_etag || null;
}

async function convertCurrentPresetToBundle() {
    const id = document.getElementById('modal-preset-id')?.value;
    const current = sessionState.presets.find(preset => preset.id === id);
    if (!id || !current || current.bundle) return;
    const confirmed = await showConfirmDialog(
        'Convert to managed bundle',
        'This creates one explicitly managed artifact from the current preset. Files with similar names are not grouped automatically.',
        'Convert'
    );
    if (!confirmed) return;
    try {
        const response = await fetch(`/api/presets/${encodeURIComponent(id)}/convert-to-bundle`, {
            method: 'POST',
            headers: { ...(window.authHeaders ? window.authHeaders() : {}), 'Content-Type': 'application/json' },
            body: JSON.stringify({ expected_revision: current.revision, conversion: {} }),
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
        await loadPresets(id);
        _presetBundleDraft = bundleClone(data.preset?.bundle || sessionState.presets.find(p => p.id === id)?.bundle);
        renderPresetBundleEditor();
        showToast('Preset converted to a managed bundle', 'success');
    } catch (error) {
        showToast(`Bundle conversion failed: ${error.message || error}`, 'error');
    }
}

async function addBundleArtifact() {
    if (!_presetBundleDraft) return;
    const pathEl = document.getElementById('modal-bundle-artifact-path');
    const role = document.getElementById('modal-bundle-artifact-role')?.value || 'weights';
    const path = pathEl?.value.trim() || '';
    const warning = document.getElementById('modal-bundle-artifact-warning');
    if (!path) {
        if (warning) warning.textContent = 'Choose a local GGUF artifact first.';
        return;
    }
    if ((_presetBundleDraft.artifacts || []).some(artifact => artifact.local_path === path)) {
        if (warning) warning.textContent = 'That local artifact is already in this bundle.';
        return;
    }
    const headers = { ...(window.authHeaders ? window.authHeaders() : {}), 'Content-Type': 'application/json' };
    let metadata;
    try {
        const response = await fetch('/api/models/gguf-meta', { method: 'POST', headers, body: JSON.stringify({ model_path: path }) });
        const data = await response.json().catch(() => ({}));
        if (!response.ok || !data.ok) throw new Error(data.error || `HTTP ${response.status}`);
        metadata = bundleMetadataFromResponse(data);
    } catch (error) {
        if (warning) warning.textContent = `GGUF metadata could not be verified: ${error.message || error}`;
        return;
    }
    const existingWeights = (_presetBundleDraft.artifacts || []).find(artifact => artifact.role === 'weights');
    const mismatch = existingWeights && role === 'weights' && (
        (existingWeights.metadata?.gguf_architecture && metadata.gguf_architecture && existingWeights.metadata.gguf_architecture !== metadata.gguf_architecture)
        || (existingWeights.metadata?.block_count && metadata.block_count && existingWeights.metadata.block_count !== metadata.block_count)
    );
    if (mismatch) {
        const message = `GGUF metadata does not match the existing tune (${existingWeights.metadata.gguf_architecture || 'unknown'} / ${existingWeights.metadata.block_count || '?'} layers versus ${metadata.gguf_architecture || 'unknown'} / ${metadata.block_count || '?'} layers). Add it only if you have confirmed this is the same exact tune.`;
        if (warning) warning.textContent = message;
        if (!await showConfirmDialog('Metadata mismatch', message, 'Add artifact anyway')) return;
    }
    const id = `artifact_${crypto.randomUUID()}`;
    const filename = path.split(/[\\/]/).pop() || 'Artifact';
    const artifact = {
        id,
        role,
        display_name: filename,
        local_path: path,
        hf_origin: null,
        size_bytes: null,
        digest: null,
        quantization: { value: quantizationHint(filename), provenance: 'filename_hint' },
        metadata,
        mmproj_artifact_id: null,
        draft_artifact_id: null,
        extensions: {},
    };
    _presetBundleDraft.artifacts = [...(_presetBundleDraft.artifacts || []), artifact];
    if (role === 'mmproj' || role === 'draft') {
        const weights = _presetBundleDraft.artifacts.find(item => item.role === 'weights');
        if (weights) weights[role === 'mmproj' ? 'mmproj_artifact_id' : 'draft_artifact_id'] = id;
    }
    if (role === 'weights' && !_presetBundleDraft.default_selection?.artifact_id) {
        _presetBundleDraft.default_selection.artifact_id = id;
    }
    if (warning) warning.textContent = 'Artifact metadata verified. Review the bundle and save to persist it.';
    if (pathEl) pathEl.value = '';
    renderPresetBundleEditor();
}

function removeBundleArtifact(artifactId) {
    if (!_presetBundleDraft) return;
    const artifact = _presetBundleDraft.artifacts.find(item => item.id === artifactId);
    if (!artifact) return;
    if (artifactId === _presetBundleDraft.default_selection?.artifact_id) {
        const replacement = _presetBundleDraft.artifacts.find(item => item.role === 'weights' && item.id !== artifactId);
        if (!replacement) {
            showToast('Select a replacement weights artifact before removing the active one.', 'warn');
            return;
        }
        _presetBundleDraft.default_selection.artifact_id = replacement.id;
    }
    _presetBundleDraft.artifacts = _presetBundleDraft.artifacts.filter(item => item.id !== artifactId);
    for (const item of _presetBundleDraft.artifacts) {
        if (item.mmproj_artifact_id === artifactId) item.mmproj_artifact_id = null;
        if (item.draft_artifact_id === artifactId) item.draft_artifact_id = null;
    }
    renderPresetBundleEditor();
}

export function syncSelectedPresetSelection(presetId, options = {}) {
    const id = presetId || '';
    const {
        userIntent = false,
        syncSetup = true,
        syncDisplay = true,
        persist = false,
    } = options;

    const mainSelect = document.getElementById('preset-select');
    if (mainSelect && id) {
        const opt = mainSelect.querySelector(`option[value="${CSS.escape(id)}"]`);
        if (opt) mainSelect.value = id;
    } else if (mainSelect && !id) {
        mainSelect.value = '';
    }

    const selectedId = mainSelect?.value || id;
    sessionState.selectedPresetId = selectedId;

    if (syncSetup) {
        const setupSelect = document.getElementById('setup-preset-select');
        if (setupSelect && selectedId) {
            const opt = setupSelect.querySelector(`option[value="${CSS.escape(selectedId)}"]`);
            if (opt) setupSelect.value = selectedId;
        }
    }

    if (syncDisplay && mainSelect) {
        syncPresetDisplay(mainSelect);
    }

    if (userIntent) {
        window.__presetUserSelected = true;
    }

    if (persist) {
        saveSettings();
    }

    import('./setup-view.js').then(m => {
        m.updateRunningCardHighlight?.();
    }).catch(() => {});

    return selectedId;
}

function isWindowsAbsolutePath(value) {
    return /^[A-Za-z]:[\\/]/.test(value);
}

function looksLikeLocalModelSource(value) {
    const v = (value || '').trim();
    if (!v) return false;
    const lower = v.toLowerCase();
    return v.startsWith('/') ||
        v.startsWith('./') ||
        v.startsWith('../') ||
        v.startsWith('~') ||
        v.includes('\\') ||
        isWindowsAbsolutePath(v) ||
        lower.endsWith('.gguf');
}

function normalizeModelSourceInput(value) {
    const input = (value || '').trim();
    if (!input) {
        return { model_path: '', hf_repo: null };
    }
    if (looksLikeLocalModelSource(input)) {
        return { model_path: input, hf_repo: null };
    }
    return { model_path: '', hf_repo: input };
}

function _presetChatTemplateName(path) {
    if (!path) return '';
    return path.split(/[\\/]/).pop().replace(/\.jinja$/, '');
}

// Refreshes the novice-view status line + primary button label from the
// current value of the chat-template-file field. Async: looks up
// installed-at from the release index when a template is set.
async function updatePresetChatTemplateStatusLine() {
    const statusEl = document.getElementById('preset-chat-template-status');
    const primaryBtn = document.getElementById('preset-recommended-chat-template-btn');
    const path = strVal('modal-chat-template-file');
    if (!statusEl) return;

    if (!path) {
        statusEl.textContent = chatTemplateStatusText({ mode: 'builtin' });
        if (primaryBtn) primaryBtn.textContent = 'Use recommended template';
        return;
    }

    const name = _presetChatTemplateName(path);
    let installedAt = null;
    try {
        const releases = await fetchReleases(name);
        if (releases.ok) {
            const active = (releases.releases || []).find(r => r.sha256 === releases.active_sha256) || releases.releases?.[0];
            installedAt = active?.installed_at || null;
        }
    } catch { /* non-fatal — status line still shows the custom state */ }

    statusEl.textContent = chatTemplateStatusText({ mode: 'custom', tplDisplay: name, installedAt });
    if (primaryBtn) primaryBtn.textContent = 'Revert to built-in';
}

// Resolves the community-template family for a preset using real model
// metadata only — never filename matching. Prefers the backend-derived
// `preset.family` (set from GGUF `general.architecture` by ensure_gguf_metadata
// on the server); falls back to a live GGUF-metadata read for local models
// whose preset hasn't been backfilled yet.
async function communityTemplateFamilyForPreset(preset) {
    const fromPreset = communityTemplateFamilyFor(preset?.family);
    if (fromPreset) return fromPreset;

    const modelPath = preset?.model_path || strVal('modal-model-path');
    if (!modelPath || !looksLikeLocalModelSource(modelPath)) return null;

    try {
        const headers = window.authHeaders
            ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
            : { 'Content-Type': 'application/json' };
        const resp = await fetch('/api/models/gguf-meta', {
            method: 'POST',
            headers,
            body: JSON.stringify({ model_path: modelPath }),
        });
        if (!resp.ok) return null;
        const meta = await resp.json().catch(() => ({}));
        if (!meta.ok || !meta.architecture) return null;
        return communityFamilyFromGgufArchitecture(meta.architecture);
    } catch {
        return null;
    }
}

async function installRecommendedChatTemplateForPreset() {
    // The primary button toggles: "Use recommended template" installs one,
    // "Revert to built-in" (shown once a custom template is active) clears it.
    const currentPath = strVal('modal-chat-template-file');
    if (currentPath) {
        setVal('modal-chat-template-file', '');
        await updatePresetChatTemplateStatusLine();
        showToast('Reverted to built-in template', 'success');
        return;
    }

    const family = await communityTemplateFamilyForPreset(_currentModalPreset());
    const template = getDefaultTemplateForFamily(family);
    if (!template) {
        showToast('No community template recommendation for this model', 'warn');
        return;
    }

    const button = document.getElementById('preset-recommended-chat-template-btn');
    if (button) button.disabled = true;
    try {
        const headers = window.authHeaders
            ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
            : { 'Content-Type': 'application/json' };
        const install = buildCommunityTemplateInstallRequest(template);
        const resp = await fetch(install.endpoint, {
            method: 'POST',
            headers,
            body: JSON.stringify(install.body),
        });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok || !data.ok || !data.path) {
            throw new Error(data.error || `HTTP ${resp.status}`);
        }
        const templatePath = (template.transformed && data.transformed_path) ? data.transformed_path : data.path;
        setVal('modal-chat-template-file', templatePath);
        await updatePresetChatTemplateStatusLine();
        showToast(
            data.already_existed ? 'Recommended template selected' : 'Recommended template installed',
            'success',
            template.display,
        );
    } catch (err) {
        showToast('Template install failed: ' + (err.message || String(err)), 'error');
    } finally {
        if (button) button.disabled = false;
    }
}

function bindRecommendedChatTemplateButton() {
    const modal = document.getElementById('preset-modal');
    if (!modal || modal.dataset.recommendedTemplateBound === 'true') return;
    // Delegate from the stable modal shell. The form contents can be refreshed
    // while model metadata is loading, so binding the transient button node can
    // lose the handler between modal open and the user's click.
    modal.addEventListener('click', event => {
        const button = event.target.closest?.('#preset-recommended-chat-template-btn');
        if (!button || button.disabled) return;
        event.preventDefault();
        void installRecommendedChatTemplateForPreset();
    });
    modal.dataset.recommendedTemplateBound = 'true';
}

function clearFieldErrors() {
    // Clear field-error class
    document.querySelectorAll('#preset-form .field-error').forEach(el => el.classList.remove('field-error'));
    // Remove any inline error messages we added
    document.querySelectorAll('#preset-form .field-error-msg').forEach(el => el.remove());
}

function markFieldError(fieldId, message) {
    const el = document.getElementById(fieldId);
    if (!el) return;
    el.classList.add('field-error');
    // Insert an inline error message if not already present
    const existing = el.parentElement.querySelector('.field-error-msg');
    if (existing) {
        existing.textContent = message;
    } else {
        const msg = document.createElement('div');
        msg.className = 'field-error-msg';
        msg.textContent = message;
        el.after(msg);
    }
}

function scrollToFirstError() {
    const first = document.querySelector('#preset-form .field-error');
    if (!first) return;
    first.scrollIntoView({ behavior: 'smooth', block: 'center' });
    if (first.focus) first.focus();
}

// ── Load ───────────────────────────────────────────────────────────────────────

export async function loadPresets(selectId) {
    const auth = window.authHeaders ? window.authHeaders() : {};

    const [presetsResp, settingsResp, activeResp, collectionsResp] = await Promise.all([
        fetch('/api/presets', { headers: auth }),
        selectId === undefined ? fetch('/api/settings', { headers: auth }) : Promise.resolve(null),
        selectId === undefined ? fetch('/api/sessions/active', { headers: auth }) : Promise.resolve(null),
        selectId === undefined ? fetch('/api/collections', { headers: auth }) : Promise.resolve(null),
    ]);

    if (presetsResp.status === 401) {
        showToast('Unauthorized: API token missing or invalid', 'error');
        return;
    }

    sessionState.presets = await presetsResp.json();
    if (collectionsResp && collectionsResp.ok) {
        try {
            const collectionsData = await collectionsResp.json();
            sessionState.collections = collectionsData.collections || [];
        } catch {
            sessionState.collections = [];
        }
    } else {
        sessionState.collections = [];
    }
    let saved = null;
    if (settingsResp) {
        if (settingsResp.status === 401) {
            console.warn('[presets] /api/settings returned 401');
        } else {
            saved = await settingsResp.json();
        }
    }

    const sel = document.getElementById('preset-select');
    sel.innerHTML = '';
    sessionState.presets.forEach(p => {
        // Skip built-in/example presets that have no model (they are templates, not usable)
        if (!presetModelSource(p)) return;
        const opt = document.createElement('option');
        opt.value = p.id;
        opt.textContent = p.name;
        sel.appendChild(opt);
    });

    // Determine which preset to select:
    // 1) explicit selectId (used by spawn-wizard / CRUD)
    // 2) running session's preset_id (if available and session is Running)
    // 3) saved UiSettings.preset_id
    // 4) first preset as fallback
    let targetId = selectId ?? null;
    if (targetId === null && activeResp) {
        const activeData = await activeResp.json().catch(() => null);
        if (activeData && activeData.preset_id && isRunningStatus(activeData.status)) {
            sessionState.activeSessionPresetId = activeData.preset_id;
            targetId = activeData.preset_id;
        } else {
            sessionState.activeSessionPresetId = '';
        }
    }
    if (targetId === null) {
        targetId = saved?.preset_id || null;
    }

    if (targetId && sessionState.presets.find(p => p.id === targetId)) {
        syncSelectedPresetSelection(targetId, { syncSetup: false, syncDisplay: false });
    } else if (sel.options.length > 0) {
        syncSelectedPresetSelection(sel.options[0].value, { syncSetup: false, syncDisplay: false });
    }

    if (selectId === undefined && saved) {
        applySettings(saved);
    }
    if (selectId === undefined) {
        saveSettings();
    }

    // Sync the visual preset display (short name + chips)
    if (sel && sel.value) syncSelectedPresetSelection(sel.value, { syncSetup: false });

    // Keep the setup view preset dropdown and launch grid in sync
    import('./setup-view.js').then(m => m.syncSetupPresetSelect?.()).catch(() => {});

    // Main dashboard: handle user changing the preset while on the dashboard
    wirePresetSelectChangeHandler();
}

// Handle user switching presets in the main dashboard dropdown.
// If a model is already loaded (different preset), prompt to stop it and load the new one.
function wirePresetSelectChangeHandler() {
    const sel = document.getElementById('preset-select');
    if (!sel) return;

    // Avoid duplicate wiring (populatePresetSelect can run more than once)
    if (sel.__presetChangeWired) return;
    sel.__presetChangeWired = true;

    sel.addEventListener('change', async () => {
        const chosenId = sel.value;
        if (!chosenId) return;

        // Mark as user-initiated so WS sync won't force it back
        syncSelectedPresetSelection(chosenId, { userIntent: true, persist: true });

        // Fetch current active session to see if something is running
        try {
            const resp = await fetch('/api/sessions/active', {
                headers: window.authHeaders ? window.authHeaders() : {},
            });
            if (!resp.ok) return;
            const active = await resp.json().catch(() => ({}));

            // If nothing is running, or it's the same preset, nothing special to do
            if (!active || !isRunningStatus(active.status) || active.preset_id === chosenId) {
                showToast('Preset selected', 'info');
                return;
            }

            // Different preset is running: show a non-blocking toast to confirm switch
            const chosenPreset = sessionState.presets.find(p => p.id === chosenId);
            const runningPreset = sessionState.presets.find(p => p.id === active.preset_id);
            const chosenName = chosenPreset?.name || 'selected preset';
            const runningName = runningPreset?.name || 'current preset';

            const revert = () => {
                syncSelectedPresetSelection(active.preset_id, { persist: true });
                window.__presetUserSelected = false;
            };

            showToastWithActions(
                `Switch to "${chosenName}"?`,
                'info',
                `Currently running: ${runningName}`,
                [
                    {
                        id: 'restart',
                        label: 'Restart Now',
                        primary: true,
                        handler: async () => {
                            showToast('Switching preset…', 'info');
                            const { doKillLlamaInternal, doStart } = await import('./attach-detach.js');
                            await doKillLlamaInternal();
                            await new Promise(r => setTimeout(r, 400));
                            await doStart(null, { skipRunningConfirm: true });
                        },
                    },
                    {
                        id: 'cancel',
                        label: 'Cancel',
                        primary: false,
                        handler: revert,
                    },
                ],
                { duration: 12000, onDismiss: revert },
            );

        } catch (e) {
            console.warn('[presets] preset-select change error:', e);
        } finally {
            syncSelectedPresetSelection(sel.value);
        }
    });
}

// ── Preset display (short name + chips) ────────────────────────────────────────

function syncPresetDisplay(sel) {
    const labelEl = document.getElementById('preset-display-label');
    const chipsEl = document.getElementById('preset-display-chips');
    if (!labelEl || !chipsEl || !sel || !sel.value) return;

    const preset = (sessionState.presets || []).find(p => p.id === sel.value);
    if (!preset) return;

    const fullName = preset.name || presetModelSource(preset).split('/').pop() || '';
    const displayName = buildShortPresetName(preset, fullName);

    labelEl.textContent = displayName;
    labelEl.title = fullName;

    chipsEl.innerHTML = '';
    const chips = buildPresetChips(preset);
    for (const chip of chips) {
        const span = document.createElement('span');
        span.className = 'preset-display-chip';
        span.textContent = chip.label;
        if (chip.title) span.title = chip.title;
        chipsEl.appendChild(span);
    }
}

function buildShortPresetName(p, fullName) {
    const base = fullName || p.name || presetModelSource(p).split('/').pop() || '';
    if (!base) return '';
    // Normalize underscores to hyphens; CSS text-overflow handles truncation.
    return base.replace(/_/g, '-').replace(/-{2,}/g, '-').trim();
}

function buildPresetChips(p) {
    const chips = [];
    const name = p.name || presetModelSource(p).split('/').pop() || '';

    // Quant chip
    const qMatch = name.match(/(Q\d+[_-]?[A-Z0-9]+)/i);
    if (qMatch) {
        chips.push({ label: qMatch[1].toUpperCase() });
    }

    // Context chip
    const ctx = p.context_size;
    if (ctx != null && ctx > 0) {
        let ctxLabel;
        if (ctx <= 1000) {
            ctxLabel = String(ctx);
        } else if (ctx < 1_000_000) {
            ctxLabel = Math.round(ctx / 1024) + 'k';
        } else {
            ctxLabel = 'large';
        }
        chips.push({ label: ctxLabel, title: `${ctx.toLocaleString()} tokens` });
    }

    // Draft / speculative chip:
    // Only show "DRAFT" when there's an actual draft model or MTP-style spec decoding.
    // Basic ngram or simple lookahead do NOT warrant a "DRAFT" pill.
    const spec = (p.spec_type || '').toLowerCase();
    const draftModel = (p.draft_model_path || p.draft_model || '').trim();
    const hasMtp = spec.includes('mtp');
    const isNgram = spec.includes('ngram');
    const isSimple = spec === 'simple';
    const isEmpty = !spec || spec === 'none';
    if (draftModel || hasMtp) {
        chips.push({ label: 'Draft', title: `Speculative decoding: ${p.spec_type}` });
    } else if (!isNgram && !isSimple && !isEmpty) {
        // Unknown/advanced spec type without ngram → still show Draft
        chips.push({ label: 'Draft', title: `Speculative decoding: ${p.spec_type}` });
    }

    return chips.slice(0, 3);
}

// Click the preset display wrapper to open the underlying <select>
document.addEventListener('DOMContentLoaded', () => {
    const wrapper = document.querySelector('.preset-display-wrapper');
    const sel = document.getElementById('preset-select');
    if (!wrapper || !sel) return;

    wrapper.addEventListener('click', (e) => {
        // Don't interfere with child button clicks
        if (e.target.closest('.preset-inline-actions')) return;
        e.stopPropagation();
        // showPicker is the preferred, least intrusive way to open the native menu
        if (typeof sel.showPicker === 'function') {
            sel.showPicker();
        } else {
            // Fallback: click the select directly so the browser opens its options
            sel.focus();
            sel.click();
        }
    });

});

// ── Modal ──────────────────────────────────────────────────────────────────────

// ── Performance advisor (config-time hints) ──────────────────────────────────
 let _presetAdvisorTimer = null;
let _presetAdvisorSeq = 0;
let _presetIsUnified = null; // cached platform check
let _presetRamUsedBytes = 0;
let _presetVramBytes = 0;
let _presetRamBytes = 0;    // cached RAM total (bytes)
let _presetMetalLimitMb = 0; // cached iogpu.wired_limit_mb (0 = use heuristic)
let _presetSnapshot = null; // MemoryAvailabilitySnapshot — refreshed periodically
let _presetSnapshotAge = 0; // timestamp when snapshot was fetched

// ── VRAM live estimate ────────────────────────────────────────────────────────
let _presetVramTimer = null;
let _presetVramSeq = 0;

/// Phase 5b Part C: Fetch MemoryAvailabilitySnapshot for accurate availability.
/// Cached briefly (30s) to avoid excessive API calls while remaining responsive.
async function _presetRefreshSnapshot() {
    const now = Date.now();
    // Refresh if we have no snapshot or it's older than 30 seconds.
    if (_presetSnapshot && (now - _presetSnapshotAge) < 30000) {
        return _presetSnapshot;
    }
    try {
        const headers = window.authHeaders ? window.authHeaders() : {};
        const resp = await fetch('/api/memory-availability', { headers });
        if (!resp.ok) return null;
        const data = await resp.json();
        if (!data.ok || !data.snapshot) return null;
        _presetSnapshot = data.snapshot;
        _presetSnapshotAge = now;
        return _presetSnapshot;
    } catch {
        return null;
    }
}

function _presetAvailBytes() {
    if (!_presetIsUnified) return _presetVramBytes;
    // Phase 5b Part C: use current_safe_availability_bytes from the snapshot,
    // NOT a stale fraction of total RAM.
    if (_presetSnapshot && _presetSnapshot.current_safe_availability_bytes > 0) {
        return _presetSnapshot.current_safe_availability_bytes;
    }
    // Fallback: stale heuristic (should not normally be used after snapshot is fetched).
    if (_presetRamBytes === 0) return 0;
    const limitBytes = _presetMetalLimitMb > 0 ? _presetMetalLimitMb * 1024 * 1024 : null;
    const fraction = _presetRamBytes <= 36 * 1024 ** 3 ? 2 / 3 : 3 / 4;
    const cap = limitBytes ?? Math.floor(_presetRamBytes * fraction);
    return Math.max(0, Math.min(cap, _presetRamBytes) - 512 * 1024 * 1024);
}

function _presetAvailableRamBytes() {
    return _presetIsUnified ? 0 : Math.max(0, _presetRamBytes - _presetRamUsedBytes);
}

async function _ensureUnifiedFlag() {
    if (_presetIsUnified !== null) return _presetIsUnified;
    try {
        const headers = window.authHeaders ? window.authHeaders() : {};
        const [platform, sys, gpu] = await Promise.all([
            getPlatformInfo().catch(() => null),
            fetch('/metrics/system', { headers }).then(r => r.ok ? r.json() : null).catch(() => null),
            fetch('/metrics/gpu', { headers }).then(r => r.ok ? r.json() : null).catch(() => null),
        ]);
        _presetIsUnified = platform?.auto_backend === 'metal';
        _presetRamBytes = (sys?.ram_total_gb || lastSystemMetrics?.ram_total_gb || 0) * 1024 ** 3;
        _presetRamUsedBytes = (sys?.ram_used_gb || lastSystemMetrics?.ram_used_gb || 0) * 1024 ** 3;
        if (!_presetIsUnified && gpu) {
            const gpus = Array.isArray(gpu) ? gpu : (gpu.gpus ? gpu.gpus : Object.values(gpu));
            _presetVramBytes = gpus.reduce((sum, g) => {
                const totalMb = g.vram_total_mb || g.total_mb || g.total_memory_mb || g.vram_total || 0;
                const usedMb = g.vram_used_mb || g.used_mb || g.vram_used || 0;
                return sum + Math.max(0, totalMb - usedMb) * 1024 * 1024;
            }, 0);
        }
        if (_presetIsUnified) {
            const lim = await fetch('/api/system/metal-gpu-limit', { headers }).then(r => r.ok ? r.json() : null).catch(() => null);
            if (lim?.ok && lim.limit_mb > 0) _presetMetalLimitMb = lim.limit_mb;
        }
    } catch { _presetIsUnified = _presetIsUnified ?? false; }
    return _presetIsUnified;
}

function _parseParamB(name) {
    const m = /(\d+(?:\.\d+)?)\s*b\b/i.exec(name || '');
    return m ? parseFloat(m[1]) : 0;
}

function looksLikeQwenName(name) {
    const n = (name || '').toLowerCase();
    return n.includes('qwen');
}

function qwenVLImageTokens(p) {
    // Return recommended image token budgets for multimodal (mmproj) models.
    // Qwen3.6 vision: 1024 / 4096
    // Gemma4: 280 / 1120 (valid: 70, 140, 280, 560, 1120)
    const name = p.model_path || '';
    const repo = p.hf_repo || '';
    const hasMmproj = !!p.mmproj;
    if (!hasMmproj) return { min_tokens: null, max_tokens: null };

    if (looksLikeQwenName(name) || looksLikeQwenName(repo)) {
        return { min_tokens: 1024, max_tokens: 4096 };
    }
    if ((name || '').toLowerCase().includes('gemma') || (repo || '').toLowerCase().includes('gemma')) {
        return { min_tokens: 280, max_tokens: 1120 };
    }
    return { min_tokens: null, max_tokens: null };
}

function applyPresetSuggestion(suggestion) {
    const patch = suggestionPatch(suggestion);
    const map = { ctk: 'modal-ctk', ctv: 'modal-ctv', context_size: 'modal-context-size', spec_type: 'modal-spec-type' };
    Object.entries(patch).forEach(([k, v]) => {
        const id = map[k];
        const el = id && document.getElementById(id);
        if (!el) return; // spec_draft_n_max has no direct preset field; spec-type drives MTP
        el.value = String(v);
        el.dispatchEvent(new Event('change', { bubbles: true }));
    });
    updatePresetAdvisor();
    showToast('Applied', 'success', suggestion.label);
}

export function updatePresetAdvisor() {
    const box = document.getElementById('preset-advisor');
    const cards = document.getElementById('preset-advisor-cards');
    if (!box || !cards) return;
    clearTimeout(_presetAdvisorTimer);
    _presetAdvisorTimer = setTimeout(async () => {
        const isUnified = await _ensureUnifiedFlag();
        const modelVal = document.getElementById('modal-model-path')?.value.trim() || '';
        const name = modelVal.split(/[/\\]/).pop() || '';
        if (!name) { box.style.display = 'none'; return; }
        const ctk = document.getElementById('modal-ctk')?.value || 'q8_0';
        const ctv = document.getElementById('modal-ctv')?.value || 'q8_0';
        const ctx = parseInt(document.getElementById('modal-context-size')?.value) || 8192;
        const specType = document.getElementById('modal-spec-type')?.value || '';
        const body = {
            name,
            param_b: _parseParamB(name),
            context_size: ctx,
            ctk, ctv,
            is_unified: isUnified,
            spec_type: specType || null,
  // Draft/head artifacts often lack introspectable MTP metadata. Keep the
  // filename signal as a provisional hint, never as confirmed model state.
  has_mtp: false,
  mtp_inferred: /(?:mtp|draft(?:[-_]model)?)/i.test(name),
        };
        const seq = ++_presetAdvisorSeq;
        try {
            const headers = window.authHeaders
                ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
                : { 'Content-Type': 'application/json' };
            const r = await fetch('/api/advise', { method: 'POST', headers, body: JSON.stringify(body) });
            if (seq !== _presetAdvisorSeq) return;
            const data = await r.json();
            const suggestions = (data && data.suggestions) || [];
            const cfgView = { ctk, ctv, context_size: ctx, spec_type: specType };
            renderSuggestionCards(cards, suggestions, { onApply: applyPresetSuggestion, config: cfgView });
            box.style.display = cards.childElementCount ? '' : 'none';
        } catch { box.style.display = 'none'; }
    }, 250);
}

// Render a structured hint (icon + headline + body + optional inline action) into one of
// the `.preset-memory-warning` slots. `severity` picks the colour: info (blue), caution
// (amber, the base), danger (orange-red). Always sets the base class so the box styling
// applies (a prior bug overwrote className to an unstyled class).
function _renderPresetHint(el, { severity = 'caution', icon = '', head = '', body = '', action = null }) {
    if (!el) return;
    el.className =
        severity === 'danger'
            ? 'preset-memory-warning preset-mlock-warning--suggest-off'
            : severity === 'info'
              ? 'preset-memory-warning preset-memory-warning--info'
              : 'preset-memory-warning';
    // Build with DOM APIs (textContent) rather than innerHTML — injection-safe and lint-clean.
    el.replaceChildren();
    const wrap = document.createElement('div');
    wrap.className = 'pe-hint';
    const ic = document.createElement('span');
    ic.className = 'pe-hint-icon';
    ic.textContent = icon;
    const bodyEl = document.createElement('span');
    bodyEl.className = 'pe-hint-body';
    if (head) {
        const h = document.createElement('span');
        h.className = 'pe-hint-head';
        h.textContent = head;
        bodyEl.append(h, document.createTextNode(' '));
    }
    bodyEl.append(document.createTextNode(body));
    if (action) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'pe-hint-action';
        btn.textContent = action.label;
        btn.addEventListener('click', action.onClick);
        const btnWrap = document.createElement('div');
        btnWrap.append(btn);
        bodyEl.append(btnWrap);
    }
    wrap.append(ic, bodyEl);
    el.append(wrap);
    el.style.display = '';
}

function updatePresetMlockWarning(estimate = null) {
    const el = document.getElementById('preset-mlock-warning');
    if (!el) return;
    const checked = document.getElementById('modal-mlock')?.checked;
    if (!checked) {
        el.style.display = 'none';
        el.innerHTML = '';
        return;
    }

    const rec = estimate?.recommendation || '';
    const total = estimate?.total_bytes || 0;
    const avail = estimate?.available_vram_bytes || 0;
    const ratio = avail > 0 ? total / avail : 0;
    const pressure = rec === 'risk' || rec === 'tight' || ratio >= 0.82;
    const sys = lastSystemMetrics;
    const wiredGb = sys?.memory_wired_gb || 0;
    const modelGib = total / (1024 ** 3);
    const wiredAfter = wiredGb + modelGib;

    // Would mlock push total system RAM above 90%? Past that, macOS can't compress model
    // pages and gets starved of headroom.
    const totalRamGb = sys?.ram_total_gb || 0;
    const usedRamGb = sys?.ram_used_gb || 0;
    const projectedPct = totalRamGb > 0 ? Math.round(((usedRamGb + modelGib) / totalRamGb) * 100) : 0;
    const wiredOverload = _presetIsUnified && totalRamGb > 0 && projectedPct >= 90;
    const wiredNote =
        _presetIsUnified && wiredGb > 0 && modelGib > 0
            ? ` System wired memory: ${wiredGb.toFixed(1)} GB now → ~${wiredAfter.toFixed(1)} GB after loading (wired = non-compressible).`
            : '';

    const turnOff = {
        label: 'Turn off mlock',
        onClick: () => {
            const c = document.getElementById('modal-mlock');
            if (c) {
                c.checked = false;
                // Dispatch change so form listeners (estimate, hints, dirty-tracking) all fire.
                c.dispatchEvent(new Event('change', { bubbles: true }));
            }
        },
    };

    if (wiredOverload) {
        _renderPresetHint(el, {
            severity: 'danger',
            icon: '⚠️',
            head: 'mlock isn’t recommended here.',
            body: `Loading this model pins ~${projectedPct}% of system RAM as non-compressible — macOS can’t relieve pressure and the desktop may stall. On Apple Silicon, Metal already keeps model memory resident while the server runs, so mlock adds risk with no benefit.${wiredNote}`,
            action: turnOff,
        });
    } else if (pressure) {
        _renderPresetHint(el, {
            severity: 'caution',
            icon: '⚠️',
            head: 'Tight fit with mlock.',
            body: `This estimate is already tight — pinned memory can push macOS into compression or swap and make the desktop unresponsive.${wiredNote}`,
            action: _presetIsUnified ? turnOff : null,
        });
    } else {
        _renderPresetHint(el, {
            severity: 'caution',
            icon: '📌',
            head: 'mlock pins model memory.',
            body: `It stops the OS reclaiming model pages. Leave enough free RAM for macOS, browsers, and background tasks.${wiredNote}`,
        });
    }
}

// Apple Silicon recommendation: keep mmap ON (no-mmap OFF). Measured identical throughput
// on M-series, and mmap is zero-copy into Metal — disabling it only slows loads and commits
// the whole model to RAM up front. Shown only on unified memory when no-mmap is enabled.
function updatePresetMmapHint() {
    const el = document.getElementById('preset-mmap-hint');
    if (!el) return;
    const noMmap = document.getElementById('modal-no-mmap')?.checked;
    if (!_presetIsUnified || !noMmap) {
        el.style.display = 'none';
        el.innerHTML = '';
        return;
    }
    _renderPresetHint(el, {
        severity: 'info',
        icon: '🍎',
        head: 'no-mmap isn’t recommended on Apple Silicon.',
        body: 'On unified memory, mmap is zero-copy into Metal — it doesn’t change throughput (measured identical tok/s on M-series) but loads faster and avoids committing the whole model to RAM up front. Leave it off unless the model lives on slow network storage.',
        action: {
            label: 'Turn off no-mmap',
            onClick: () => {
                const c = document.getElementById('modal-no-mmap');
                if (c) {
                    c.checked = false;
                    // Dispatch change so form listeners (estimate, hints, dirty-tracking) all fire.
                    c.dispatchEvent(new Event('change', { bubbles: true }));
                }
            },
        },
    });
}

// ── VRAM live estimate for preset editor ─────────────────────────────────────

async function autoSizePreset() {
    const btn = document.getElementById('preset-vram-auto-size');
    const modelVal = document.getElementById('modal-model-path')?.value.trim() || '';
    if (!modelVal) {
        showToast('Auto-size requires a model', 'warn');
        return;
    }
    if (!btn) return;
    const origText = btn.textContent;
    btn.disabled = true;
    btn.textContent = 'Sizing...';

    try {
        const headers = window.authHeaders
            ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
            : { 'Content-Type': 'application/json' };
        const body = {
            model_path: modelVal,
            native_context_limit: _presetNativeContextLimit || null,
            n_ctx: parseInt(document.getElementById('modal-context-size')?.value) || 128000,
            ctk: document.getElementById('modal-ctk')?.value || 'q8_0',
            ctv: document.getElementById('modal-ctv')?.value || 'f16',
            parallel_slots: parseInt(document.getElementById('modal-parallel-slots')?.value) || 1,
            ubatch_size: parseInt(document.getElementById('modal-ubatch-size')?.value) || 512,
            n_cpu_moe: parseInt(document.getElementById('modal-n-cpu-moe')?.value) || 0,
            gpu_layers: Number.isFinite(parseInt(document.getElementById('modal-gpu-layers')?.value))
                ? parseInt(document.getElementById('modal-gpu-layers')?.value)
                : -1,
            available_vram_bytes: _presetAvailBytes(),
            available_ram_bytes: _presetAvailableRamBytes(),
            is_unified_memory: !!_presetIsUnified,
            backend: _currentModalPreset()?.backend === 'rapid_mlx' ? 'rapid_mlx' : 'llama_cpp',
        };

        const resp = await fetch('/api/vram/auto-size', { method: 'POST', headers, body: JSON.stringify(body) });
        if (!resp.ok) {
            showToast('Auto-size failed', 'error');
            return;
        }
        const data = await resp.json();
        if (!data.ok || !data.result) {
            showToast('Auto-size: no result', 'warning');
            return;
        }

        const r = data.result;
        setVal('modal-context-size', r.context_size);
        setVal('modal-ctk', r.kv_quant_k);
        setVal('modal-ctv', r.kv_quant_v);
        setVal('modal-ubatch-size', r.ubatch_size);

        // Trigger UI updates
        updatePresetVram();
        updatePresetAdvisor();
        showToast('Auto-sized', 'success', `Optimized to ${r.context_size} tokens`);

    } catch (err) {
        showToast('Auto-size error', 'error', err.message);
    } finally {
        btn.disabled = false;
        btn.textContent = origText;
    }
}

export function updatePresetVram() {
    const box = document.getElementById('preset-vram-display');
    const strip = document.getElementById('preset-vram-strip');
    if (!box) return;
    const setStripVisible = (visible) => {
        if (!strip) return;
        strip.classList.toggle('is-empty', !visible);
        strip.setAttribute('aria-hidden', String(!visible));
    };
    updatePresetMlockWarning();
    updatePresetMmapHint();
    const modelVal = document.getElementById('modal-model-path')?.value.trim() || '';
    if (!modelVal) { setStripVisible(false); return; }
    setStripVisible(true);
    box.innerHTML = '<div class="preset-vram-loading">Estimating VRAM…</div>';
    clearTimeout(_presetVramTimer);
    _presetVramTimer = setTimeout(async () => {
        const isUnified = await _ensureUnifiedFlag();
        // Phase 5b Part C: refresh memory state from the snapshot (no infinite cache).
        if (isUnified) {
            await _presetRefreshSnapshot();
        }
        // Platform flag is now resolved — refresh the Apple Silicon mmap hint.
        updatePresetMmapHint();
        const nCtx = parseInt(document.getElementById('modal-context-size')?.value) || 131072;
        const ctk = document.getElementById('modal-ctk')?.value || 'q8_0';
        const ctv = document.getElementById('modal-ctv')?.value || 'f16';
        const parallelSlots = parseInt(document.getElementById('modal-parallel-slots')?.value) || 1;
        const ubatch = parseInt(document.getElementById('modal-ubatch-size')?.value) || 512;
        const nCpuMoe = parseInt(document.getElementById('modal-n-cpu-moe')?.value) || 0;
        const gpuLayers = parseInt(document.getElementById('modal-gpu-layers')?.value);
        const mmprojPath = document.getElementById('modal-mmproj')?.value?.trim() || '';
        const available_vram_bytes = _presetAvailBytes();
        const currentPreset = _currentModalPreset();
        const isRapidMlx = currentPreset?.backend === 'rapid_mlx';
        const backend = isRapidMlx ? 'rapid_mlx' : 'llama_cpp';
        const rapidPolicy = isRapidMlx ? _rapidEstimatePolicyFromForm(currentPreset?.rapid_mlx) : {};

        // Builder item 6: use canonical body builder for cross-surface equality.
        const body = buildEstimateBody({
            backend,
            model_path: modelVal,
            n_ctx: nCtx,
            parallel_slots: parallelSlots,
            ubatch_size: ubatch,
            ctk: backend === 'llama_cpp' ? ctk : undefined,
            ctv: backend === 'llama_cpp' ? ctv : undefined,
            n_cpu_moe: nCpuMoe,
            gpu_layers: Number.isFinite(gpuLayers) ? gpuLayers : -1,
            available_vram_bytes,
            available_ram_bytes: _presetAvailableRamBytes(),
            is_unified_memory: !!isUnified,
            mmproj_path: mmprojPath || null,
            ...rapidPolicy,
        });
        const seq = ++_presetVramSeq;
        try {
            const headers = window.authHeaders
                ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
                : { 'Content-Type': 'application/json' };
            const r = await fetch('/api/vram-estimate', { method: 'POST', headers, body: JSON.stringify(body) });
            if (seq !== _presetVramSeq) return;
            const hideStrip = () => setStripVisible(false);
            if (!r.ok) { hideStrip(); return; }
            const data = await r.json();
            if (data.error) { hideStrip(); return; }
            _renderPresetVram(box, data);
            updatePresetMlockWarning(data);
        } catch { if (seq === _presetVramSeq) setStripVisible(false); }
    }, 350);
}

function _rapidEstimatePolicyFromForm(fallback = {}) {
    const prefixEnabled = document.getElementById('modal-rapid-prefix-cache-enabled')?.checked !== false;
    const speculativeEnabled = !!document.getElementById('modal-rapid-speculative-enabled')?.checked;
    const speculativeSource = document.getElementById('modal-rapid-speculative-source')?.value || 'embedded';
    const speculativeModel = document.getElementById('modal-rapid-speculative-model')?.value?.trim() || '';
    const speculativeReady = speculativeEnabled && (speculativeSource !== 'external' || speculativeModel);
    const config = {
        ...fallback,
        kv_cache_dtype: document.getElementById('modal-rapid-kv-cache-dtype')?.value || null,
        turboquant_mode: document.getElementById('modal-rapid-turboquant-mode')?.value || null,
        prefix_cache_enabled: prefixEnabled,
        retained_cache_mib: prefixEnabled
            ? Number(document.getElementById('modal-rapid-cache-memory-mib')?.value || 8192)
            : 0,
        prefill_step_size: Number(document.getElementById('modal-rapid-prefill-step-size')?.value || 512),
        speculative_config: speculativeReady ? {
            method: 'mtp',
            model: speculativeSource === 'external' ? speculativeModel : null,
            num_speculative_tokens: Number(document.getElementById('modal-rapid-speculative-tokens')?.value || RAPID_MLX_DEFAULT_SPECULATIVE_TOKENS),
            disable_auto_k: !!document.getElementById('modal-rapid-speculative-disable-auto-k')?.checked,
        } : null,
    };
    return rapidEstimatePolicyFromConfig(config);
}

function _renderPresetVram(el, data) {
    const fmt = b => {
        const gib = b / (1024 ** 3);
        return gib >= 1 ? gib.toFixed(1) + ' GiB' : (b / (1024 ** 2)).toFixed(0) + ' MiB';
    };
    const avail   = data.available_bytes || 0;  // budget we sent
    const used    = data.total_bytes || 0;       // total model + context size
    const weights = data.weights_bytes || 0;

    // Builder item 6: Rapid-MLX active/retained KV split — distinct totals.
    // For Rapid-MLX with workload_scenario: show active + retained separately.
    // For llama.cpp or legacy calls: unified kv_cache_bytes.
    const isRapidSplit = (data.active_kv_bytes || 0) > 0 && (data.retained_kv_bytes || 0) > 0;
    const activeKV = data.active_kv_bytes || 0;
    const retainedKV = data.retained_kv_bytes || 0;
    const kv = isRapidSplit ? 0 : (data.kv_cache_bytes || 0); // unified KV when no split
    const mmproj  = data.mmproj_bytes || 0;
    const mtp     = data.mtp_bytes || 0;
    const overhead = data.overhead_bytes || 0;
    const linearState = data.linear_attn_state_bytes || 0;
    const tqTransient = data.turboquant_transient_peak_bytes || 0;
    // Phase 6 Part B: prefix cache budget display (informational, not consumed until active).
    const prefixCacheBudget = data.mlx_prefix_cache_bytes || 0;

    // Bar 100% = budget so free headroom is visible; fall back to used if no budget
    const barTotal = avail > 0 ? avail : used;
    const free = avail > 0 ? Math.max(0, avail - used) : 0;
    const pct = v => barTotal > 0 ? Math.max(0, Math.min(100, (v / barTotal) * 100)).toFixed(1) + '%' : '0%';

    const rec = data.recommendation || 'fit';
    const recLabel = rec === 'fit' ? 'Fits' : rec === 'tight' ? 'Tight' : 'Risk';
    const recClass = rec === 'fit' ? 'fit' : rec === 'tight' ? 'tight' : 'risk';
    const ramBytes = data.ram_bytes || 0;
    const ramAvail = data.available_ram_bytes || _presetAvailableRamBytes();
    const ramPct = ramAvail > 0
        ? Math.max(0, Math.min(100, (ramBytes / ramAvail) * 100)).toFixed(1) + '%'
        : '0%';
    const ramLabel = ramAvail > 0 ? `${fmt(ramBytes)} / ${fmt(ramAvail)}` : fmt(ramBytes);
    const ramBar = !_presetIsUnified && ramBytes > 0
        ? `<div class="preset-vram-row preset-vram-row--ram">
            <span class="preset-memory-kind">RAM</span>
            <div class="vram-bar${ramBytes > ramAvail && ramAvail > 0 ? ' over-budget' : ''}">
                <div class="vram-segment seg-ram-moe" style="width:${ramPct}" title="CPU model weights"></div>
            </div>
            <span class="launch-card-vram-total">${ramLabel}</span>
        </div>`
        : '';

    const parts = [];
    if (weights > 0) parts.push(`Weights ${fmt(weights)}`);
    if (isRapidSplit) {
        if (activeKV > 0) parts.push(`Active KV ${fmt(activeKV)}`);
        if (retainedKV > 0) parts.push(`Retained KV ${fmt(retainedKV)}`);
    } else if (kv > 0) {
        parts.push(`KV ${fmt(kv)}`);
    }
    if (linearState > 0) parts.push(`Linear attn ${fmt(linearState)}`);
    if (mmproj > 0) parts.push(`mmproj ${fmt(mmproj)}`);
    if (mtp > 0) parts.push(`MTP ${fmt(mtp)}`);
    if (tqTransient > 0) parts.push(`TQ transient ${fmt(tqTransient)}`);
    if (overhead > 0) parts.push(`overhead ${fmt(overhead)}`);
    if (avail > 0 && free > 0) parts.push(`${fmt(free)} budget headroom`);
    // Phase 6 Part B: show prefix cache budget as informational (not consumed until active).
    if (prefixCacheBudget > 0) parts.push(`Rapid retained cache ${fmt(prefixCacheBudget)}`);
    const nativeContextLimit = Number(data.native_context_limit || 0);
    const selectedContext = Number(document.getElementById('modal-context-size')?.value || 0);
    if (_presetNativeContextLimit !== nativeContextLimit) {
        _presetNativeContextLimit = nativeContextLimit;
        _renderContextPills();
    }
    const nativeHint = document.getElementById('preset-context-native-hint');
    if (nativeHint) {
        if (nativeContextLimit > 0) {
            nativeHint.textContent = selectedContext > nativeContextLimit
                ? `Native model maximum: ${Math.round(nativeContextLimit / 1024)}k. This selection needs Advanced Context extension controls and is not benchmark-qualified.`
                : `Native model maximum: ${Math.round(nativeContextLimit / 1024)}k. Higher context requires separately qualified Advanced Context extension controls.`;
            nativeHint.style.display = '';
        } else {
            nativeHint.style.display = 'none';
        }
    }
    if (nativeContextLimit > 0) {
        const nativeLabel = `${Math.round(nativeContextLimit / 1024)}k native max`;
        parts.push(selectedContext > nativeContextLimit
            ? `${nativeLabel} · Advanced Context required`
            : nativeLabel);
    }

    // Show post-load system RAM projection when we have live metrics
    const sys = lastSystemMetrics;
    let systemLine = '';
    if (sys && sys.ram_total_gb > 0 && sys.ram_used_gb > 0 && used > 0 && _presetIsUnified) {
        const usedGib = used / (1024 ** 3);
        const sysGib = sys.ram_used_gb;
        const totalGib = sys.ram_total_gb;
        const afterGib = sysGib + usedGib;
        const pctAfter = Math.round((afterGib / totalGib) * 100);
        const isTight = pctAfter >= 90;
        const wiredGib = sys.memory_wired_gb || 0;
        const mlockOn = document.getElementById('modal-mlock')?.checked;
        const wiredAfter = wiredGib + (mlockOn ? usedGib : 0);
        const wiredNote = mlockOn && wiredAfter > 0
            ? ` · ${wiredAfter.toFixed(1)} GiB wired (mlock)`
            : '';
        // When mlock is on and the projected load would exceed 90% of RAM, append a
        // direct suggestion to disable it so the user sees it without expanding the warning.
        const mlockHint = mlockOn && isTight && _presetIsUnified
            ? ' — disable mlock to avoid wiring all model memory'
            : '';
        systemLine = `<div class="preset-vram-sysram${isTight ? ' preset-vram-sysram--warn' : ''}">` +
            `System RAM: ${sysGib.toFixed(1)} GiB now → ~${afterGib.toFixed(1)} GiB after loading (${pctAfter}% of ${totalGib.toFixed(0)} GiB${wiredNote})${mlockHint}` +
            `</div>`;
    } else if (!_presetIsUnified && (data.ram_bytes || 0) > 0) {
        const ramNeeded = data.ram_bytes || 0;
        const ramAvail = data.available_ram_bytes || _presetAvailableRamBytes();
        const ramOver = ramAvail > 0 && ramNeeded > ramAvail;
        const ramCapacity = ramAvail > 0 ? ` / ${fmt(ramAvail)} system RAM available` : '';
        systemLine = `<div class="preset-vram-sysram${ramOver ? ' preset-vram-sysram--warn' : ''}">` +
            `CPU weights: ${fmt(ramNeeded)}${ramCapacity}` +
            `</div>`;
    }

    // Builder item 6: distinct active/retained segments for Rapid-MLX when applicable.
    const kvSegments = isRapidSplit
        ? `<div class="vram-segment seg-active-kv" style="width:${pct(activeKV)}" title="Active KV Cache"></div>
           <div class="vram-segment seg-retained-kv" style="width:${pct(retainedKV)}" title="Retained KV Cache"></div>`
        : `<div class="vram-segment seg-kv" style="width:${pct(kv)}" title="KV Cache"></div>`;
    const linearSeg = linearState > 0
        ? `<div class="vram-segment seg-overhead" style="width:${pct(linearState)}" title="Linear Attention State"></div>`
        : '';
    const tqSeg = tqTransient > 0
        ? `<div class="vram-segment seg-overhead" style="width:${pct(tqTransient)}" title="TurboQuant Transient"></div>`
        : '';

    // eslint-disable-next-line no-unsanitized/property -- DOMPurify sanitizes the VRAM bar HTML
    el.innerHTML = window.DOMPurify.sanitize(`
        <div class="preset-vram-row">
            <span class="preset-memory-kind">${_presetIsUnified ? 'MEM' : 'VRAM'}</span>
            <div class="vram-bar">
                <div class="vram-segment seg-weights" style="width:${pct(weights)}" title="Weights"></div>
                ${kvSegments}
                <div class="vram-segment seg-mmproj" style="width:${pct(mmproj)}" title="Vision Projector"></div>
                <div class="vram-segment seg-mtp" style="width:${pct(mtp)}" title="MTP Heads"></div>
                ${linearSeg}
                ${tqSeg}
                <div class="vram-segment seg-overhead" style="width:${pct(overhead)}" title="Overhead"></div>
                <div class="vram-segment seg-free" style="width:${pct(free)}" title="Budget Headroom"></div>
            </div>
            <span class="launch-card-vram-total">~${fmt(used)}</span>
            <span class="preset-vram-badge preset-vram-badge--${recClass}">${recLabel}</span>
        </div>
        ${ramBar}
        ${parts.length ? `<div class="preset-vram-breakdown">${parts.join(' · ')}</div>` : ''}
        ${systemLine}
    `);
    el.style.display = '';
    const explain = document.getElementById('preset-vram-explain');
    if (explain) explain.onclick = () => openEstimateEvidenceDrawer(data, 'Preset memory estimate', explain);
}

// Empirically auto-tune n_cpu_moe for the preset's model via llama-bench.
async function autoTunePreset() {
    const statusEl = document.getElementById('preset-moe-autotune-status');
    const modelVal = document.getElementById('modal-model-path')?.value.trim() || '';
    if (!modelVal.toLowerCase().endsWith('.gguf')) {
        // The sweep launches llama-bench on a local file, not an HF repo id.
        showToast('Sweep needs a local .gguf model file', 'warn');
        return;
    }
    const body = {
        name: modelVal.split(/[/\\]/).pop() || '',
        param_b: _parseParamB(modelVal),
        model_path: modelVal,
        ngl: -1, // -1 → -ngl all
        ctk: document.getElementById('modal-ctk')?.value || 'q8_0',
        ctv: document.getElementById('modal-ctv')?.value || 'q8_0',
        flash_attn: true,
        is_unified_memory: !!(await _ensureUnifiedFlag()),
        verify: true,
    };
    if (statusEl) statusEl.innerHTML = '<span class="moe-autotune-spinner"></span>Running sweep… this can take a few minutes';
    try {
        const data = await requestNcpuMoeTune(body);
        if (data.error) { if (statusEl) statusEl.textContent = data.error; return; }
        const rec = data.recommended_n_cpu_moe;
        const input = document.getElementById('modal-n-cpu-moe');
        if (input) { input.value = String(rec); input.dispatchEvent(new Event('change', { bubbles: true })); }
        if (statusEl) statusEl.textContent = `Best: ${rec} (measured)`;
    } catch {
        if (statusEl) statusEl.textContent = 'Auto-tune failed';
    }
}

export function openPresetModal(mode, section, seedPreset = null) {
    _speculativeSidecarAutoSelected = false;
    const modal = document.getElementById('preset-modal');
    const title = document.getElementById('modal-title');
    const subtitle = document.getElementById('preset-editor-subtitle');
    const formatPill = document.getElementById('preset-editor-format');
    const form = document.getElementById('preset-form');
    // Tests and other modal managers may close overlays by setting inert and
    // aria-hidden directly. Always restore the modal's interactive state when
    // opening it again; otherwise the overlay can paint while swallowing every
    // click and select action inside it.
    modal.removeAttribute('aria-hidden');
    modal.inert = false;
    modal.classList.remove('closing');
    initPresetEditorNav();
    bindRecommendedChatTemplateButton();
    form.reset();
    updatePresetChatTemplateStatusLine();
    if (formatPill) formatPill.hidden = true;
    clearFieldErrors();
    newPresetSeed = mode === 'new' && seedPreset ? structuredClone(seedPreset) : null;
    _presetRapidMlxProfile = null;
    _presetRapidMlxPrefillExplicit = false;

    if (mode === 'edit') {
        const id = document.getElementById('preset-select').value;
        const p = sessionState.presets.find(pr => pr.id === id);
        if (!p) { showToast('No preset selected', 'warn'); return; }
        _presetBundleDraft = bundleClone(p.bundle);
        title.textContent = 'Edit Preset';
        if (subtitle) {
            subtitle.textContent = 'Model profile';
            subtitle.title = p.name;
        }
        if (formatPill) {
            const source = presetModelSource(p).toLowerCase();
            const format = p.backend === 'rapid_mlx' ? 'MLX' : source.endsWith('.gguf') ? 'GGUF' : '';
            formatPill.textContent = format;
            formatPill.hidden = !format;
            formatPill.title = format ? `${format} preset` : '';
        }
        setVal('modal-preset-id', p.id);
        // Model & Memory
        setVal('modal-name', p.name);
        // Prefill model field:
        // - If model_path present, treat as local file.
        // - Else if hf_repo present, treat as HF repo.
        const modelValue = presetModelSource(p);
        setVal('modal-model-path', modelValue);
        document.getElementById('modal-model-path').title = modelValue;
        _renderPresetArchInfo(p);
        setVal('modal-alias', p.alias || '');
        // Fetch live Rapid-MLX profile when editing a Rapid-MLX preset
        _presetRapidMlxPrefillExplicit = p.rapid_mlx?.prefill_step_size != null;
        _schedulePresetRapidMlxProfile();
        numOrEmpty('modal-gpu-layers', p.gpu_layers);
  setChk('modal-no-mmap', p.no_mmap);
  setOpt('modal-load-mode', p.load_mode || (p.no_mmap ? 'none' : 'mmap'));
        setChk('modal-mlock', p.mlock);
        // Context & KV
        setVal('modal-context-size', p.context_size || 128000);
        setVal('modal-ctk', p.ctk || 'q8_0');
        const pillsContainer = document.getElementById('preset-context-pills'); if (pillsContainer) pillsContainer.style.display = 'flex';
        _renderContextPills(mode, section);
        setVal('modal-ctv', p.ctv || 'f16');
        setOpt('modal-flash-attn', p.flash_attn);
        // Batching
        setVal('modal-batch-size', p.batch_size || 2048);
        setVal('modal-ubatch-size', p.ubatch_size || p.batch_size || 2048);
        setVal('modal-parallel-slots', p.parallel_slots || 1);
        const cacheIdleHint = document.getElementById('cache-idle-slots-hint');
        if (cacheIdleHint) cacheIdleHint.style.display = (p.parallel_slots || 1) > 1 ? '' : 'none';
        setOpt('modal-prio', p.prio != null ? String(p.prio) : '');
 setOpt('modal-prio-batch', p.prio_batch != null ? String(p.prio_batch) : '');
        numOrEmpty('modal-verbosity', p.verbosity ?? 4);
        setChk('modal-no-cont-batching', p.no_cont_batching);
        setChk('modal-swa-full', p.swa_full);
        numOrEmpty('modal-ctx-checkpoints', p.ctx_checkpoints);
        numOrEmpty('modal-checkpoint-min-step', p.checkpoint_min_step);
        numOrEmpty('modal-cache-reuse', p.cache_reuse);
        setOpt('modal-cache-idle-slots', p.cache_idle_slots == null ? '' : p.cache_idle_slots ? 'true' : 'false');
        numOrEmpty('modal-threads', p.threads);
        numOrEmpty('modal-threads-batch', p.threads_batch);
        // Generation
        numOrEmpty('modal-temperature', p.temperature);
        numOrEmpty('modal-top-p', p.top_p);
        numOrEmpty('modal-top-k', p.top_k);
        numOrEmpty('modal-min-p', p.min_p);
        numOrEmpty('modal-repeat-penalty', p.repeat_penalty);
        numOrEmpty('modal-repeat-last-n', p.repeat_last_n);
        numOrEmpty('modal-presence-penalty', p.presence_penalty);
        setOpt('modal-enable-thinking', p.enable_thinking == null ? '' : String(!!p.enable_thinking));
        setOpt('modal-preserve-thinking', p.preserve_thinking == null ? '' : String(!!p.preserve_thinking));
        setOpt('modal-tool-call-format', p.tool_call_format || '');
 setOpt('modal-reasoning', p.reasoning || '');
 setOpt('modal-llama-reasoning-effort', p.llama_reasoning_effort || 'default');
 setOpt('modal-llama-reasoning-format', p.llama_reasoning_format || '');
 setOpt('modal-llama-reasoning-preserve', p.llama_reasoning_preserve == null ? '' : String(p.llama_reasoning_preserve));
        numOrEmpty('modal-reasoning-budget', p.reasoning_budget);
        setVal('modal-reasoning-budget-message', (p.reasoning_budget_message || '').replace(/\n/g, '\\n'));
        // GPU
        setVal('modal-tensor-split', p.tensor_split);
        setOpt('modal-split-mode', p.split_mode);
        numOrEmpty('modal-main-gpu', p.main_gpu);
        // Threading
        numOrEmpty('modal-threads', p.threads);
        numOrEmpty('modal-threads-batch', p.threads_batch);
        // n_cpu_moe: only for MoE / hybrid-moE with experts
        const moeRow = document.getElementById('modal-n-cpu-moe')?.closest('.pe-field') ||
                       document.getElementById('modal-n-cpu-moe')?.parentElement;
        const moeAutotuneBtn = document.getElementById('preset-moe-autotune-verify');
        if (moeRow) {
            moeRow.style.display = isMoEEligible(p) ? '' : 'none';
        }
        if (moeAutotuneBtn) {
            moeAutotuneBtn.style.display = isMoEEligible(p) ? '' : 'none';
        }
        const moeLayersHint = document.getElementById('modal-n-cpu-moe-layers');
        if (isMoEEligible(p)) {
            numOrEmpty('modal-n-cpu-moe', p.n_cpu_moe);
            const nMoeEl = document.getElementById('modal-n-cpu-moe');
            // Bound the input to the model's layer count (from backend GGUF metadata).
            if (nMoeEl && p.block_count != null) nMoeEl.max = p.block_count;
            else if (nMoeEl) nMoeEl.removeAttribute('max');
            if (moeLayersHint) {
                if (p.block_count != null) {
                    // Real measured routed-expert bytes per MoE layer (VRAM freed per offload).
                    const freed = p.expert_bytes_per_layer
                        ? ` Each offloaded layer frees ~${_formatLayerBytes(p.expert_bytes_per_layer)} of VRAM.`
                        : '';
                    moeLayersHint.textContent = `This model has ${p.block_count} expert layers — values are clamped to 0–${p.block_count}.${freed}`;
                    moeLayersHint.style.display = '';
                } else {
                    moeLayersHint.style.display = 'none';
                }
            }
        } else {
            const el = document.getElementById('modal-n-cpu-moe');
            if (el) { el.value = ''; el.removeAttribute('max'); }
            if (moeLayersHint) moeLayersHint.style.display = 'none';
        }
        // Bound --gpu-layers (-ngl) to the layer count for all models (the primary
        // GPU-offload knob for dense models, where there are no experts to offload).
        const nglEl = document.getElementById('modal-gpu-layers');
        if (nglEl) {
            if (p.block_count != null) nglEl.max = p.block_count;
            else nglEl.removeAttribute('max');
        }
        const nglHint = document.getElementById('modal-gpu-layers-layers');
        if (nglHint) {
            if (p.block_count != null) {
                const off = Math.max(0, p.block_count - 4);
                // Real measured per-layer weight bytes (VRAM each GPU layer occupies).
                const perLayer = p.bytes_per_layer
                    ? ` (~${_formatLayerBytes(p.bytes_per_layer)} of VRAM each)`
                    : '';
                nglHint.textContent = `This model has ${p.block_count} layers${perLayer}. Enter 0–${p.block_count}: layers above your value stay on CPU/RAM (e.g. ${off} keeps 4 layers off the GPU).`;
                nglHint.style.display = '';
            } else {
                nglHint.style.display = 'none';
            }
        }
        // Rope
        setOpt('modal-rope-scaling', p.rope_scaling);
        numOrEmpty('modal-rope-freq-base', p.rope_freq_base);
        numOrEmpty('modal-rope-freq-scale', p.rope_freq_scale);
        // Spec decoding — use spec_type; fallback: ngram_spec bool → ngram-mod
        const specType = p.spec_type || (p.ngram_spec ? 'ngram-mod' : '');
        setOpt('modal-spec-type', specType);
        numOrEmpty('modal-spec-ngram-size', p.spec_ngram_size);
        numOrEmpty('modal-draft-min', p.draft_min);
        numOrEmpty('modal-draft-max', p.draft_max);
        numOrEmpty('modal-spec-draft-n-max', p.spec_draft_n_max);
        numOrEmpty('modal-spec-draft-n-min', p.spec_draft_n_min);
        numOrEmpty('modal-spec-draft-p-min', p.spec_draft_p_min);
        setVal('modal-spec-draft-type-k', p.spec_draft_type_k || '');
        setVal('modal-spec-draft-type-v', p.spec_draft_type_v || '');
        setVal('modal-draft-model', p.draft_model);
        numOrEmpty('modal-spec-draft-ngl', p.spec_draft_ngl);
        setVal('modal-spec-draft-device', p.spec_draft_device ?? '');
        numOrEmpty('modal-spec-draft-n-cpu-moe', p.spec_draft_n_cpu_moe);
        const specDefaultEl = document.getElementById('modal-spec-default');
        if (specDefaultEl) specDefaultEl.checked = !!p.spec_default;
        _toggleSpecFields(specType);
        // Context extras
        setOpt('modal-kv-unified', p.kv_unified == null ? '' : String(p.kv_unified));
        setOpt('modal-cache-mode', p.cache_mode || 'custom');
        numOrEmpty('modal-cache-ram-mib', p.cache_ram_mib);
        _toggleCacheRamField(p.cache_mode || 'custom');
        // Model extras
        setVal('modal-mmproj', p.mmproj || '');
        setOpt('modal-mmproj-offload', p.mmproj_offload == null ? '' : String(p.mmproj_offload));
        _toggleVisionTokens(!!p.mmproj);
        setVal('modal-chat-template-file', p.chat_template_file || '');
        updatePresetChatTemplateStatusLine();
        // Advanced
        setOpt('modal-bind-host', p.bind_host || '');
        numOrEmpty('modal-port', p.backend === 'rapid_mlx' ? p.rapid_mlx?.port : p.port);
        setOpt('modal-rapid-enable-thinking', p.rapid_mlx?.enable_thinking == null ? '' : String(!!p.rapid_mlx.enable_thinking));
        // reasoning_effort removed: config field exists but argv builder does not emit --reasoning-effort.
        // Phase 6: 8 GiB retained cache is the qualified interactive default.
        const prefixCacheEnabled = p.rapid_mlx?.prefix_cache_enabled ?? true;
        if (document.getElementById('modal-rapid-prefix-cache-enabled')) {
            document.getElementById('modal-rapid-prefix-cache-enabled').checked = prefixCacheEnabled;
        }
        setOpt('modal-rapid-cache-memory-mib', String(p.rapid_mlx?.retained_cache_mib ?? (prefixCacheEnabled ? 8192 : 0)));
        setOpt('modal-rapid-hybrid-cache-entries', String(p.rapid_mlx?.hybrid_cache_entries ?? 0));
        setOpt('modal-rapid-cache-mode', p.rapid_mlx?.cache_mode || 'custom');
        _toggleRapidCacheFields(p.rapid_mlx?.cache_mode || 'custom');
        // Phase 7: Rapid-MLX advanced controls (D6 catalog IDs).
        setOpt('modal-rapid-kv-cache-dtype', p.rapid_mlx?.kv_cache_dtype || '');
        setOpt('modal-rapid-prefill-step-size', String(p.rapid_mlx?.prefill_step_size || 512));
        const speculative = p.rapid_mlx?.speculative_config || null;
        const speculativeEnabled = !!speculative;
        const speculativeSource = speculative?.model ? 'external' : 'embedded';
        const speculativeEnabledEl = document.getElementById('modal-rapid-speculative-enabled');
        if (speculativeEnabledEl) speculativeEnabledEl.checked = speculativeEnabled;
        setOpt('modal-rapid-speculative-source', speculativeSource);
        setVal('modal-rapid-speculative-model', speculative?.model || '');
        setOpt('modal-rapid-speculative-tokens', String(speculative?.num_speculative_tokens || RAPID_MLX_DEFAULT_SPECULATIVE_TOKENS));
        const disableAutoKEl = document.getElementById('modal-rapid-speculative-disable-auto-k');
        if (disableAutoKEl) disableAutoKEl.checked = !!speculative?.disable_auto_k;
        /* Restore trust_remote_code_consent for MTP companion if previously granted */
        const trustWrap = document.getElementById('modal-rapid-speculative-trust-wrap');
        const trustCheck = document.getElementById('modal-rapid-speculative-trust-consent');
        const trustWarning = document.getElementById('modal-rapid-speculative-trust-warning');
        const storedConsent = p.rapid_mlx?.trust_remote_code_consent || null;
        if (trustWrap && trustCheck && trustWarning && storedConsent) {
            _speculativeTrustState.repoId = storedConsent.split('@')[0] || '';
            _speculativeTrustState.revision = storedConsent.split('@').slice(1).join('@') || '';
            _speculativeTrustState.trustRequired = true;
            trustWarning.textContent =
                'This companion model requires trust_remote_code (custom Python code execution).';
            trustCheck.checked = true;
            trustWrap.style.display = '';
        }
        const autoToolChoiceEl = document.getElementById('modal-rapid-auto-tool-choice');
        if (autoToolChoiceEl) autoToolChoiceEl.checked = !!p.rapid_mlx?.auto_tool_choice;
        _syncRapidSpeculativeEditor();
        setOpt('modal-rapid-turboquant-mode', p.rapid_mlx?.turboquant_mode || 'none');
        setOpt('modal-rapid-gpu-memory-utilization', p.rapid_mlx?.gpu_memory_utilization == null ? '' : String(p.rapid_mlx.gpu_memory_utilization));
        setOpt('modal-rapid-max-num-seqs', p.rapid_mlx?.max_num_seqs == null ? '' : String(p.rapid_mlx.max_num_seqs));
        setOpt('modal-rapid-max-concurrent-requests', p.rapid_mlx?.max_concurrent_requests == null ? '' : String(p.rapid_mlx.max_concurrent_requests));
        setOpt('modal-rapid-prefill-batch-size', p.rapid_mlx?.prefill_batch_size == null ? '' : String(p.rapid_mlx.prefill_batch_size));
        setOpt('modal-rapid-completion-batch-size', p.rapid_mlx?.completion_batch_size == null ? '' : String(p.rapid_mlx.completion_batch_size));
        setOpt('modal-rapid-pflash-policy', p.rapid_mlx?.pflash_policy || 'off');
        setOpt('modal-rapid-tool-call-parser', p.rapid_mlx?.tool_call_parser || '');
        setOpt('modal-rapid-reasoning-parser', p.rapid_mlx?.reasoning_parser || '');
        setOpt('modal-rapid-sampling-mode', p.rapid_mlx?.sampling_mode || 'auto');
        const reasoningModeChecked = !p.rapid_mlx?.no_thinking && p.rapid_mlx?.reasoning_mode !== 'off';
        if (document.getElementById('modal-rapid-reasoning-mode')) {
            document.getElementById('modal-rapid-reasoning-mode').checked = reasoningModeChecked;
        }
        // Load Rapid-MLX sampling defaults into shared form fields
        if (p.rapid_mlx?.default_temperature != null) numOrEmpty('modal-temperature', p.rapid_mlx.default_temperature);
        if (p.rapid_mlx?.default_top_p != null) numOrEmpty('modal-top-p', p.rapid_mlx.default_top_p);
        if (p.rapid_mlx?.default_top_k != null) numOrEmpty('modal-top-k', p.rapid_mlx.default_top_k);
        if (p.rapid_mlx?.default_min_p != null) numOrEmpty('modal-min-p', p.rapid_mlx.default_min_p);
        if (p.rapid_mlx?.default_repetition_penalty != null) numOrEmpty('modal-repeat-penalty', p.rapid_mlx.default_repetition_penalty);
        if (p.rapid_mlx?.default_presence_penalty != null) numOrEmpty('modal-presence-penalty', p.rapid_mlx.default_presence_penalty);
        if (p.rapid_mlx?.default_frequency_penalty != null) numOrEmpty('modal-rapid-frequency-penalty', p.rapid_mlx.default_frequency_penalty);
        if (p.rapid_mlx?.max_tokens != null) numOrEmpty('modal-max-tokens', p.rapid_mlx.max_tokens);
        setVal('modal-api-key', p.api_key || '');
        numOrEmpty('modal-max-tokens', p.max_tokens);
        numOrEmpty('modal-seed', p.seed);
        setOpt('modal-fit-enabled', p.fit_enabled == null ? '' : String(p.fit_enabled));
        setVal('modal-fit-target', p.fit_target || '');
        _toggleFitTarget(p.fit_enabled === true);
        setVal('modal-system-prompt-file', p.system_prompt_file);
        setStructuredOutputMode(p.json_schema ? 'json_schema' : p.grammar ? 'grammar' : '');
        setVal('modal-grammar', p.grammar || '');
        setVal('modal-json-schema', p.json_schema || '');
        setVal('modal-extra-args', p.extra_args);
        numOrEmpty('modal-spec-draft-p-split', p.spec_draft_p_split);
        numOrEmpty('modal-image-min-tokens', qwenVLImageTokens(p).min_tokens);
        numOrEmpty('modal-image-max-tokens', p.image_max_tokens);
        _configureBackendPresetEditor(p);
        const workloadWrap = document.getElementById('modal-workload-policy-wrap');
        if (workloadWrap) workloadWrap.style.display = p.bundle ? '' : 'none';
        setOpt('modal-workload-policy', p.bundle?.workload_policy || 'general_chat');
    } else {
        _presetBundleDraft = bundleClone(newPresetSeed?.bundle);
        title.textContent = 'New Preset';
        if (subtitle) {
            subtitle.textContent = newPresetSeed?.backend === 'rapid_mlx'
                ? 'Rapid-MLX model profile'
                : 'New model profile';
            subtitle.title = '';
        }
        if (formatPill && newPresetSeed?.backend === 'rapid_mlx') {
            formatPill.textContent = 'MLX';
            formatPill.title = 'MLX preset';
            formatPill.hidden = false;
        }
        setVal('modal-preset-id', '');
        setVal('modal-name', newPresetSeed?.name || '');
        setVal('modal-model-path', presetModelSource(newPresetSeed));
        document.getElementById('modal-model-path').title = presetModelSource(newPresetSeed);
        setVal('modal-context-size', 128000);
        setVal('modal-ctk', 'q8_0');
        setVal('modal-ctv', 'f16');
        setVal('modal-batch-size', 2048);
        setVal('modal-ubatch-size', 2048);
        setVal('modal-parallel-slots', 1);
        setOpt('modal-mmproj-offload', '');
        setOpt('modal-llama-reasoning-effort', 'default');
        setOpt('modal-llama-reasoning-format', '');
        setOpt('modal-llama-reasoning-preserve', '');
        const workloadWrap = document.getElementById('modal-workload-policy-wrap');
        if (workloadWrap) workloadWrap.style.display = newPresetSeed?.bundle ? '' : 'none';
        setOpt('modal-workload-policy', newPresetSeed?.bundle?.workload_policy || 'general_chat');
        numOrEmpty('modal-port', newPresetSeed?.backend === 'rapid_mlx'
            ? newPresetSeed.rapid_mlx?.port
            : newPresetSeed?.port);
        _toggleFitTarget(false);
        _toggleSpecFields('');
        setStructuredOutputMode('');
        _configureBackendPresetEditor(newPresetSeed);
        _presetRapidMlxPrefillExplicit = newPresetSeed?.rapid_mlx?.prefill_step_size != null;
    }

    const editorPreset = _currentModalPreset();
    if (editorPreset?.backend !== 'rapid_mlx') {
        void loadLlamaCapabilities().then(snapshot => configureCapabilityFields(snapshot, editorPreset));
    }

    const presetModel = document.getElementById('modal-model-path')?.value.trim();
    // New preset: fill empty sampling fields + show preset pills.
    // Edit preset: only show preset pills (don't overwrite the user's saved values).
    if (presetModel) _suggestGenerationDefaults(presetModel, mode !== 'edit');
    else _renderGenerationPresetPills([]);

    // Reset change-summary state
    _hideSummary();

    // Show "Delete preset" button only when editing
    const deleteBtn = document.getElementById('preset-modal-delete');
    if (mode === 'edit') {
        if (deleteBtn) deleteBtn.style.display = '';
        const calibrateBtn = document.getElementById('preset-modal-calibrate');
        const editingPreset = sessionState.presets.find(preset => preset.id === document.getElementById('modal-preset-id')?.value);
        if (calibrateBtn) calibrateBtn.style.display = editingPreset?.backend === 'rapid_mlx' ? 'none' : '';
    } else {
        if (deleteBtn) deleteBtn.style.display = 'none';
        const calibrateBtn = document.getElementById('preset-modal-calibrate');
        if (calibrateBtn) calibrateBtn.style.display = 'none';
    }

    const variantsNav = modal.querySelector('.preset-nav-item[data-section="variants"]');
    const variantsSection = modal.querySelector('.preset-editor-section[data-section="variants"]');
    const supportsBundles = newPresetSeed?.backend !== 'rapid_mlx'
        && _currentModalPreset()?.backend !== 'rapid_mlx';
    if (variantsNav) variantsNav.hidden = !supportsBundles;
    if (variantsSection) variantsSection.hidden = !supportsBundles;
    const convertButton = modal.querySelector('#preset-convert-bundle');
    if (convertButton) convertButton.hidden = !supportsBundles || !document.getElementById('modal-preset-id')?.value;
    renderPresetBundleEditor();
    modal.classList.add('open');
    // Navigate to specified section, or reset to first section
    const targetSection = section || 'model';
    document.querySelector(`.preset-nav-item[data-section="${targetSection}"]`)?.click();
    const body = modal.querySelector('.modal-body');
    if (body) body.scrollTop = 0;

    // Apple Silicon-aware hints for Threads fields
    _refreshPresetThreadsHints();
    if (!lastSystemMetrics) _fetchSystemInfoAndRefreshPresetHints();

    // Config-time performance advisor and VRAM estimate
    updatePresetAdvisor();
    updatePresetVram();

    // Focus first interactive element in the modal
    const firstFocusable = modal.querySelector('.preset-nav-item, button, input, select, textarea');
    if (firstFocusable) firstFocusable.focus();

    // Escape key handler to close modal
    function setupEscapeHandler() {
        window.addEventListener('keydown', function escHandler(e) {
            if (e.key === 'Escape') {
                window.removeEventListener('keydown', escHandler);
                closePresetModal();
            }
        });
    }
    setupEscapeHandler();
}

export function closePresetModal() {
    document.getElementById('preset-modal').classList.remove('open');
    newPresetSeed = null;
    _presetBundleDraft = null;
}

// ── Presets Panel ──────────────────────────────────────────────────────────────

export function openPresetsPanel() {
    const overlay = document.getElementById('presets-panel-overlay');
    if (!overlay) return;
    overlay.style.display = '';
    overlay.classList.add('open');
    _renderPresetsPanel();
    document.getElementById('presets-panel-wizard-btn')?.addEventListener('click', () => {
        closePresetsPanel();
        // Route through the Router so the URL, Back/Forward, and wizard step state
        // stay in sync instead of opening the wizard out-of-band.
        import('./router.js').then(({ default: Router }) => Router.navigate('/spawn'));
    }, { once: true });
}

export function closePresetsPanel() {
    const overlay = document.getElementById('presets-panel-overlay');
    if (!overlay) return;
    overlay.classList.remove('open');
    overlay.style.display = 'none';
}

function _renderPresetsPanel() {
    const body = document.getElementById('presets-panel-body');
    if (!body) return;
    body.innerHTML = '';

    const presets = (sessionState.presets || []).filter(presetModelSource);
    if (!presets.length) {
        const empty = document.createElement('div');
        empty.className = 'presets-panel-empty';
        empty.textContent = 'No presets yet. Use the Setup wizard to create one.';
        body.appendChild(empty);
        return;
    }

    presets.forEach(preset => {
        const card = document.createElement('div');
        card.className = 'preset-panel-card';

        const icon = document.createElement('div');
        icon.className = 'preset-panel-card-icon';
        icon.textContent = '🧠';
        card.appendChild(icon);

        const info = document.createElement('div');
        info.className = 'preset-panel-card-info';

        const name = document.createElement('div');
        name.className = 'preset-panel-card-name';
        name.textContent = preset.name || 'Unnamed preset';
        info.appendChild(name);

        const metaParts = [];
        const rapidMlx = preset.rapid_mlx;
        if (preset.backend === 'rapid_mlx') {
            const modelIdentity = rapidMlx?.model_source_view?.canonical_identity
                || rapidMlx?.model_source_view?.display_name;
            if (modelIdentity) {
                metaParts.push(modelIdentity.split(/[/\\]/).pop() || modelIdentity);
                metaParts.push('Rapid-MLX');
            }
        } else if (preset.model_path) metaParts.push(preset.model_path.split(/[/\\]/).pop() || preset.model_path);
        else if (preset.hf_repo) metaParts.push(preset.hf_repo);
        if (preset.bind_host === '0.0.0.0') metaParts.push('LAN');
        if (preset.context_size) metaParts.push(`${Math.round(preset.context_size / 1024)}k context`);
        const ctk = preset.ctk || 'q8_0';
        const ctv = preset.ctv || 'q8_0';
        const kvText = `KV cache: ${ctk}/${ctv}`;
        if (ctk || ctv) metaParts.push(kvText);

        const meta = document.createElement('div');
        meta.className = 'preset-panel-card-meta';
        meta.textContent = metaParts.join(' · ') || 'No details';
        meta.title = metaParts.join(' · ') +
          (ctk || ctv ? ' · KV cache precision (how accurately the model stores past tokens). q8_0 is recommended for most users.' : '');
        info.appendChild(meta);
        card.appendChild(info);

        const actions = document.createElement('div');
        actions.className = 'preset-panel-card-actions';

        const startBtn = document.createElement('button');
        startBtn.type = 'button';
        startBtn.className = 'btn-preset-quick-start';
        startBtn.textContent = '▶ Quick Start';
        startBtn.title = 'Spawn this server configuration now';
        startBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            syncSelectedPresetSelection(preset.id, { userIntent: true, persist: true });
            closePresetsPanel();
            import('./attach-detach.js').then(({ doStartFromSetup }) => {
                doStartFromSetup();
            });
        });
        actions.appendChild(startBtn);

        const delBtn = document.createElement('button');
        delBtn.type = 'button';
        delBtn.className = 'btn-preset-delete';
        delBtn.title = 'Delete preset';
        delBtn.textContent = '✕';
        delBtn.addEventListener('click', async (e) => {
            e.stopPropagation();
            showToastWithActions(
                'Delete preset',
                'warning',
                `Delete "${preset.name}"? This cannot be undone.`,
                [
                    { id: 'cancel', label: 'Cancel', primary: false },
                    {
                        id: 'delete',
                        label: 'Delete',
                        primary: true,
                        handler: async () => {
                            try {
                                const headers = window.authHeaders ? { ...window.authHeaders() } : {};
                                const resp = await fetch(`/api/presets/${preset.id}`, { method: 'DELETE', headers });
                                if (resp.ok) {
                                    await loadPresets();
                                    _renderPresetsPanel();
                                }
                            } catch (err) {
                                console.error('Delete preset failed:', err);
                            }
                        }
                    }
                ]
            );
        });
        actions.appendChild(delBtn);

        card.appendChild(actions);

        // Top-right trash icon (subtle)
        const trashBtn = document.createElement('button');
        trashBtn.type = 'button';
        trashBtn.className = 'preset-panel-card-trash';
        trashBtn.title = 'Delete preset';
        trashBtn.innerHTML =
            '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
            '<path d="M3 6h18"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>' +
            '<path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>' +
            '<line x1="10" y1="11" x2="10" y2="17"/><line x1="14" y1="11" x2="14" y2="17"/>' +
            '</svg>';
        trashBtn.addEventListener('click', async (e) => {
            e.stopPropagation();
            showToastWithActions(
                'Delete preset',
                'warning',
                `Delete "${preset.name}"? This cannot be undone.`,
                [
                    { id: 'cancel', label: 'Cancel', primary: false },
                    {
                        id: 'delete',
                        label: 'Delete',
                        primary: true,
                        handler: async () => {
                            try {
                                const headers = window.authHeaders ? { ...window.authHeaders() } : {};
                                const resp = await fetch(`/api/presets/${preset.id}`, { method: 'DELETE', headers });
                                if (resp.ok) {
                                    await loadPresets();
                                    _renderPresetsPanel();
                                }
                            } catch (err) {
                                console.error('Delete preset failed:', err);
                            }
                        }
                    }
                ]
            );
        });
        card.appendChild(trashBtn);

        body.appendChild(card);
    });
}

// ── Change summary ────────────────────────────────────────────────────────────

function _toggleFitTarget(enabled) {
    const wrap = document.getElementById('modal-fit-target-wrap');
    if (wrap) wrap.style.display = enabled ? '' : 'none';
}

function _toggleVisionTokens(enabled) {
    const wrap = document.getElementById('vision-tokens-wrap');
    if (wrap) wrap.style.display = enabled ? '' : 'none';
}

function _toggleCacheRamField(cacheMode) {
    const wrap = document.getElementById('modal-cache-ram-mib-wrap');
    if (wrap) wrap.style.display = cacheMode === 'custom' ? '' : 'none';
}

function _toggleRapidCacheFields(cacheMode) {
    const custom = cacheMode === 'custom';
    ['modal-rapid-prefix-cache-enabled', 'modal-rapid-cache-memory-mib', 'modal-rapid-hybrid-cache-entries']
        .forEach(id => {
            const el = document.getElementById(id);
            if (el) el.disabled = !custom;
        });
}

function _ensureUbatchForImageTokens(imageMaxTokens) {
    // Gemma4-specific: non-causal vision attention requires all image tokens in a single ubatch.
    // If image-max-tokens > ubatch, we auto-raise ubatch to avoid crashes.
    // This constraint is Gemma4-only; other models are unaffected.
    if (!imageMaxTokens || imageMaxTokens <= 0) return;

    const modelVal = (document.getElementById('modal-model-path')?.value || '').toLowerCase();
    const repoVal = (document.getElementById('modal-hf-repo')?.value || '').toLowerCase();
    const isGemma4 = modelVal.includes('gemma') || repoVal.includes('gemma');
    if (!isGemma4) return;

    const ubInput = document.getElementById('modal-ubatch-size');
    const hintEl = document.getElementById('vision-ubatch-hint');
    if (!ubInput) return;
    const currentUbatch = Math.max(1, Number(ubInput.value || 0));
    if (imageMaxTokens <= currentUbatch) {
        if (hintEl) { hintEl.style.display = 'none'; hintEl.textContent = ''; }
        return;
    }
    const prev = currentUbatch;
    ubInput.value = imageMaxTokens;
    if (hintEl) {
        hintEl.textContent = `Micro-batch increased from ${prev} to ${imageMaxTokens} (required for Gemma4: all image tokens must fit in one batch).`;
        hintEl.style.display = '';
    }
}

export function openPresetMtpRepairForModel(modelPath, modelName = 'MLX model') {
    const target = (modelPath || '').trim();
    if (!target.startsWith('/')) {
        showToast('MTP sidecar repair requires a local MLX model directory.', 'warn');
        return;
    }

    openPresetModal('new', 'advanced', {
        backend: 'rapid_mlx',
        name: `${modelName} · MTP sidecar repair`,
        model_path: '',
        rapid_mlx: {
            model_source: { kind: 'mlx_directory', path: target },
        },
    });
    _configureBackendPresetEditor(newPresetSeed);
    setVal('modal-port', '');

    const enabled = document.getElementById('modal-rapid-speculative-enabled');
    const source = document.getElementById('modal-rapid-speculative-source');
    if (enabled) enabled.checked = true;
    if (source) source.value = 'external';
    enabled?.dispatchEvent(new Event('change', { bubbles: true }));
    source?.dispatchEvent(new Event('change', { bubbles: true }));
    _syncRapidSpeculativeEditor();
    document.getElementById('modal-rapid-speculative-sidecars-wrap')?.style.setProperty('display', 'block', 'important');

    const repairToggle = document.getElementById('modal-rapid-speculative-repair-toggle');
    const repairForm = document.getElementById('modal-rapid-speculative-repair-form');
    repairToggle?.closest('details')?.setAttribute('open', '');
    if (repairToggle && repairForm?.hidden) repairToggle.click();
}

function _toggleSpecFields(specType) {
    const hasNgram = specType.includes('ngram');
    const hasMtp   = specType.includes('draft-mtp');
    const hasDraft = specType === 'draft-model';
    const hasAny   = !!specType;
    const ngWrap     = document.getElementById('spec-ngram-params-wrap');
    const mtpWrap    = document.getElementById('spec-mtp-wrap');
    const dmWrap     = document.getElementById('spec-draft-model-wrap');
    const hwWrap     = document.getElementById('spec-draft-hw-wrap');
    const defWrap    = document.getElementById('spec-default-wrap');
    const hint       = document.getElementById('spec-type-hint');
    if (ngWrap)  ngWrap.style.display  = hasNgram ? '' : 'none';
    if (mtpWrap) mtpWrap.style.display = hasMtp   ? '' : 'none';
    // Show draft-model input for both draft-model and MTP with external assistant.
    if (dmWrap)  dmWrap.style.display  = (hasDraft || hasMtp) ? '' : 'none';
    // Draft hardware (ngl, device, cpu-moe) only relevant for an external draft file.
    if (hwWrap)  hwWrap.style.display  = hasDraft ? '' : 'none';
    // spec-default checkbox appears whenever any spec type is active.
    if (defWrap) defWrap.style.display = hasAny ? '' : 'none';

    // Auto-populate draft model path from modal-draft-model if available and empty
    const draftInput = document.getElementById('modal-draft-model');
    if (dmWrap && (hasDraft || hasMtp)) {
        if (draftInput && !draftInput.value.trim()) {
            // Try to get from session state preset if available
            const currentPreset = sessionState?.presets?.find(p =>
                document.getElementById('modal-preset-id')?.value === p.id
            );
            if (currentPreset && currentPreset.draft_model) {
                draftInput.value = currentPreset.draft_model;
            }
        }
    }

    const hints = {
        'ngram-mod': 'Best for server deployments with multiple slots. Uses a shared hash pool — requires no extra files or VRAM.',
        'ngram-simple': 'Lightest-weight option. Scans recent history for matching n-grams. Good for single-slot use.',
        'ngram-map-k': 'Hash-map based pattern matching. Works well for repetitive content like code or structured data.',
        'ngram-map-k4v': 'Experimental. Tracks up to 4 candidate tokens per n-gram key. May outperform ngram-map-k on long repetitive content.',
        'draft-mtp,ngram-mod': 'MTP with n-gram fallback. If your model requires an external assistant (e.g. Gemma4-style), set the Draft Model path below. MTP + ngram-mod forces --parallel 1.',
        'draft-mtp': 'Pure MTP with no n-gram fallback. If your model requires an external assistant (e.g. Gemma4-style), set the Draft Model path below. Forces --parallel 1.',
    };
    if (hint) {
        const text = hints[specType] || '';
        hint.textContent = text;
        hint.style.display = text ? '' : 'none';
    }
}

function _hideSummary() {
    const summary = document.getElementById('preset-change-summary');
    const back = document.getElementById('preset-modal-back');
    const cancel = document.getElementById('preset-modal-cancel');
    const saveBtn = document.getElementById('btn-modal-save');
    if (summary) summary.style.display = 'none';
    if (back) back.style.display = 'none';
    if (cancel) cancel.style.display = '';
    if (saveBtn) { saveBtn.textContent = 'Save'; saveBtn.dataset.confirmed = ''; }
}

function _configureBackendPresetEditor(preset) {
    const modal = document.getElementById('preset-modal');
    const isRapid = preset?.backend === 'rapid_mlx';
    modal?.classList.toggle('preset-editor--rapid-mlx', isRapid);

    // Phase 7: Toggle Rapid-MLX advanced rows based on backend (inline styles override CSS).
    // The webui rows are gone from this list because they are gone from index.html: they were
    // gated on llama.cpp's --ui/--path, which rapid-mlx does not have.
    const rapidRows = ['pe-row-rapid-advanced', 'pe-row-rapid-workload', 'pe-row-rapid-reasoning', 'pe-row-rapid-reasoning-mode', 'pe-row-rapid-parser-overrides', 'pe-row-rapid-architecture-overrides', 'pe-row-rapid-speculative', 'pe-row-rapid-throughput', 'pe-row-rapid-batch-sizes', 'pe-row-rapid-cache-mode', 'pe-row-rapid-prefix-cache', 'pe-row-rapid-cache-memory', 'pe-row-rapid-hybrid-cache-entries'];
    rapidRows.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.style.display = isRapid ? '' : 'none';
    });

    // MTP/concurrency teaching panel removed (Phase 7B2).

    const modelLabel = document.querySelector('label[for="modal-model-path"]');
    const modelInput = document.getElementById('modal-model-path');
    const portLabel = document.querySelector('label[for="modal-port"]');
    const modelSection = modal?.querySelector('.preset-editor-section[data-section="model"]');
    const modelTitle = modelSection?.querySelector('.pe-section-title');
    const modelDescription = modelSection?.querySelector('.pe-section-desc');
    const advancedSection = modal?.querySelector('.preset-editor-section[data-section="advanced"]');
    const advancedNavLabel = modal?.querySelector('.preset-nav-item[data-section="advanced"] .pni-label');
    const advancedTitle = advancedSection?.querySelector('.pe-section-title');
    const advancedDescription = advancedSection?.querySelector('.pe-section-desc');
    if (modelTitle) modelTitle.textContent = isRapid ? 'Rapid-MLX Model' : 'Model & Memory';
    if (modelDescription) {
        modelDescription.textContent = isRapid
            ? 'MLX model directory or Hugging Face repository'
            : 'Model file path, GPU offloading, memory locking';
    }
    if (advancedTitle) advancedTitle.textContent = isRapid ? 'Rapid-MLX Server' : 'Advanced';
    if (advancedNavLabel) advancedNavLabel.textContent = isRapid ? 'Server' : 'Advanced';
    if (advancedDescription) {
        advancedDescription.textContent = isRapid
            ? 'Local server port'
            : 'Server access, fit-to-VRAM, seed, and extra CLI flags';
    }
    if (modelLabel) {
        modelLabel.firstChild.textContent = isRapid ? 'MLX Model Path ' : 'Model Path ';
        modelLabel.title = isRapid
            ? 'Local MLX model directory or Hugging Face MLX repository id'
            : 'Absolute path to the .gguf model file on disk';
    }
    if (modelInput) {
        modelInput.placeholder = isRapid ? 'mlx-community/model-name or /path/to/model' : '';
    }
    if (portLabel) {
        portLabel.title = isRapid
            ? 'TCP port Rapid-MLX listens on.'
            : 'TCP port llama-server listens on. Default 8001. Change if you run multiple servers simultaneously.';
    }
    configureMlxPresetEditor(modal, isRapid);
}

let _speculativeSidecarAutoSelected = false;
const PRESET_MTP_REPAIR_MAX_DURATION_MS = 2 * 60 * 60 * 1000;

async function _fetchSidecarsForPreset() {
    const listEl = document.getElementById('modal-rapid-speculative-sidecars-list');
    if (!listEl) return;

    listEl.innerHTML = '<span style="color:var(--text-muted,#888);">Loading…</span>';

    try {
        const resp = await fetch('/api/hf/mtp-sidecars', { headers: window.authHeaders ? window.authHeaders() : {} });
        const data = await resp.json();

        if (!data.ok || !data.sidecars || data.sidecars.length === 0) {
            listEl.innerHTML = '<span style="color:var(--text-muted,#888);">No local sidecars found. Build one with scripts/build-mtp-head.py</span>';
            return;
        }

        const modelInput = document.getElementById('modal-rapid-speculative-model');
        const trunkInput = document.getElementById('modal-model-path');
        const selectedTrunk = (trunkInput?.value || '').trim();
        const matchingSidecar = findRapidMlxSidecarForTrunk(data.sidecars, selectedTrunk);

        // A duplicated/saved preset keeps an explicitly persisted sidecar. When
        // the external field is empty, select the validated sidecar registered
        // for the preset's local MLX trunk and keep that path editable.
        if (matchingSidecar && (!modelInput?.value.trim() || _speculativeSidecarAutoSelected)) {
            if (modelInput) modelInput.value = matchingSidecar.path;
            _speculativeSidecarAutoSelected = true;
            updatePresetVram();
        } else if (!matchingSidecar && _speculativeSidecarAutoSelected) {
            if (modelInput) modelInput.value = '';
            _speculativeSidecarAutoSelected = false;
            updatePresetVram();
        }

        // Build sidecar list
        let html = '';
        if (matchingSidecar && _speculativeSidecarAutoSelected) {
            html += '<div class="pe-field-hint" style="color:var(--success,#5ce68a); margin-bottom:5px;">Auto-selected validated sidecar for this trunk. The path remains editable below.</div>';
        } else if (!matchingSidecar) {
            const reason = selectedTrunk.startsWith('/')
                ? (modelInput?.value.trim()
                    ? 'No managed sidecar matches this trunk; the existing manual path is preserved—verify the pairing before launch.'
                    : 'No validated sidecar is registered for this trunk; speculation will stay off until one is selected.')
                : (modelInput?.value.trim()
                    ? 'Managed auto-selection is unavailable for this model reference; the explicit sidecar path is preserved.'
                    : 'Select a local MLX trunk or enter an explicit local sidecar.');
            html += '<div class="pe-field-hint" style="color:var(--warn,#e6a41c); margin-bottom:5px;">' + reason + '</div>';
        }
        data.sidecars.forEach((s, i) => {
            const p = rapidMlxSidecarProvenance(s);
            const vram = p.estimatedMemoryBytes != null
                ? '~' + (p.estimatedMemoryBytes >= 1073741824
                    ? (p.estimatedMemoryBytes / 1073741824).toFixed(1) + ' GB'
                    : Math.round(p.estimatedMemoryBytes / 1048576) + ' MB')
                : '? VRAM';

            const trunkShort = p.trunk ? p.trunk.split('/').pop() : '?';

            html += '<button type="button" class="pe-action-btn" data-sidecar-index="' + i + '" style="display:block; width:100%; text-align:left; padding:6px 8px; margin-bottom:4px; font-size:11px; background:var(--color-surface,#1a1d24); border:1px solid var(--color-border,#2a2d34); border-radius:4px; cursor:pointer;">';
            html += '<strong>' + DOMPurify.sanitize(s.slug) + '</strong> ';
            html += '<span style="color:var(--text-muted,#888);">' + vram + '</span>';
            if (p.trunk) html += ' <span style="color:var(--text-muted,#888);">for ' + DOMPurify.sanitize(trunkShort) + '</span>';
            if (p.repairMode === 'recipe_reconstruction') html += ' <span style="color:var(--accent,#8cc8ff);">Recipe reconstructed</span>';
            else if (p.repairMode === 'direct_parent') html += ' <span style="color:var(--text-muted,#888);">Direct parent</span>';
            if (p.requalificationStatus === 'qualified') html += ' <span style="color:var(--success,#5ce68a);">Qualified</span>';
            else if (p.requalificationStatus === 'screened') html += ' <span style="color:var(--success,#5ce68a);">Screened</span>';
            else if (p.requalificationStatus === 'still-blocked') html += ' <span style="color:var(--warn,#e6a41c);">StillBlocked</span>';
            else if (p.requalificationStatus === 'uninterpretable') html += ' <span style="color:var(--err,#e65c5c);">Uninterpretable</span>';
            if (p.builtAt) html += ' <span style="color:var(--text-muted,#888);">(' + (function(dt) { if (!dt) return ''; const d = new Date(dt); const diff = (Date.now() - d.getTime()) / 1000; if (diff < 60) return 'just now'; if (diff < 3600) return Math.floor(diff / 60) + 'm ago'; if (diff < 86400) return Math.floor(diff / 3600) + 'h ago'; return Math.floor(diff / 86400) + 'd ago'; })(p.builtAt) + ')</span>';
            if (!p.normCheckPassed) html += ' <span style="color:var(--err,#e65c5c);">⚠ norm check failed</span>';
            html += '</button>';
        });

        // eslint-disable-next-line no-unsanitized/property -- sidecar list built from our own API, all server strings are safe
        listEl.innerHTML = html;

        // Wire click handlers
        listEl.querySelectorAll('[data-sidecar-index]').forEach(btn => {
            btn.addEventListener('click', async () => {
                const idx = parseInt(btn.getAttribute('data-sidecar-index'));
                const sidecar = data.sidecars[idx];
                const p = rapidMlxSidecarProvenance(sidecar);

                // Set the companion model path
                if (modelInput) {
                    modelInput.value = sidecar.path;
                    _speculativeSidecarAutoSelected = false;
                    updatePresetVram();
                }

                // Update trust state for pin status display
                _speculativeTrustState.repoId = sidecar.slug;
                _speculativeTrustState.revision = p.sha256 ? p.sha256.substring(0, 12) : '';
                _speculativeTrustState.trustRequired = false;
                _speculativeTrustState.estimatedMemoryBytes = p.estimatedMemoryBytes;
                _speculativeTrustState.resolvedAt = p.builtAt;
                _speculativeTrustState.stale = false;
                _renderSpeculativePinStatus();
            });
        });
    } catch (err) {
        // eslint-disable-next-line no-unsanitized/property -- error message sanitized via DOMPurify
        listEl.innerHTML = '<span style="color:var(--err,#e65c5c);">Failed to load sidecars: ' + DOMPurify.sanitize(err.message) + '</span>';
    }
}

function _syncRapidSpeculativeEditor() {
    const enabled = !!document.getElementById('modal-rapid-speculative-enabled')?.checked;
    const external = document.getElementById('modal-rapid-speculative-source')?.value === 'external';
    const modelWrap = document.getElementById('modal-rapid-speculative-model-wrap');
    if (modelWrap) modelWrap.style.display = enabled && external ? '' : 'none';
    // If toggling off or switching away from external, also hide trust section and pin status
    const trustWrap = document.getElementById('modal-rapid-speculative-trust-wrap');
    if (trustWrap && (!enabled || !external)) {
        trustWrap.style.display = 'none';
    }
    const pinWrap = document.getElementById('modal-rapid-speculative-pin-status-wrap');
    if (pinWrap && (!enabled || !external)) {
        pinWrap.style.display = 'none';
    }
    const sidecarsWrap = document.getElementById('modal-rapid-speculative-sidecars-wrap');
    if (sidecarsWrap) {
        if (enabled && external) {
            sidecarsWrap.style.display = '';
            _fetchSidecarsForPreset();
        } else {
            sidecarsWrap.style.display = 'none';
        }
    }
}

function _syncPresetMtpRepairKind() {
    const kind = document.getElementById('modal-rapid-speculative-repair-kind')?.value || 'source';
    const revisionLabel = document.getElementById('modal-rapid-speculative-repair-revision-label');
    const revisionInput = document.getElementById('modal-rapid-speculative-repair-revision');
    const isBf16 = kind === 'bf16';
    if (revisionLabel) revisionLabel.style.display = isBf16 ? '' : 'none';
    if (revisionInput) revisionInput.style.display = isBf16 ? '' : 'none';
}

function _setPresetMtpRepairStatus(message, tone = 'muted') {
    const status = document.getElementById('modal-rapid-speculative-repair-status');
    if (!status) return;
    status.textContent = message;
    status.style.display = message ? '' : 'none';
    status.style.color = tone === 'error'
        ? 'var(--err,#e65c5c)'
        : tone === 'success' ? 'var(--success,#5ce68a)'
            : tone === 'warn' ? 'var(--warn,#e6a41c)' : 'var(--text-muted,#888)';
}

let _presetMtpRepairJobId = null;
let _presetMtpRepairOperation = null;
let _presetMtpRepairStartedAt = 0;

function _formatPresetMtpRepairDuration(seconds) {
    const total = Math.max(0, Math.floor(seconds));
    const minutes = Math.floor(total / 60);
    if (minutes < 60) return `${minutes}m elapsed`;
    return `${Math.floor(minutes / 60)}h ${minutes % 60}m elapsed`;
}

function _renderPresetRequalificationOutcome(result) {
    const outcome = result?.outcome;
    const reason = result?.reason || 'No explanation was recorded.';
    if (outcome === 'qualified') {
        _setPresetMtpRepairStatus('Qualified — sampled and tool-grammar gates passed.', 'success');
    } else if (outcome === 'screened') {
        _setPresetMtpRepairStatus('Screened — live sampled probe passed. Run Full qualification before relying on runtime promotion.', 'success');
    } else if (outcome === 'still-blocked') {
        _setPresetMtpRepairStatus('StillBlocked — ' + reason, 'warn');
    } else if (outcome === 'uninterpretable') {
        _setPresetMtpRepairStatus('Uninterpretable — ' + reason, 'error');
    } else {
        _setPresetMtpRepairStatus('Uninterpretable — the requalification result was missing or invalid.', 'error');
    }
}

function _setPresetMtpRepairButtonsDisabled(disabled) {
    const startButton = document.getElementById('modal-rapid-speculative-repair-start');
    const validateButton = document.getElementById('modal-rapid-speculative-repair-validate');
    const requalifyButton = document.getElementById('modal-rapid-speculative-repair-requalify');
    if (startButton) startButton.disabled = disabled;
    if (validateButton) validateButton.disabled = disabled;
    if (requalifyButton) requalifyButton.disabled = disabled;
}

function _finishPresetMtpRepair(jobId) {
    if (_presetMtpRepairJobId !== jobId) return;
    _presetMtpRepairJobId = null;
    _presetMtpRepairOperation = null;
    _presetMtpRepairStartedAt = 0;
    _setPresetMtpRepairButtonsDisabled(false);
}

async function _pollPresetMtpRepair(jobId) {
    if (Date.now() - _presetMtpRepairStartedAt > PRESET_MTP_REPAIR_MAX_DURATION_MS) {
        _setPresetMtpRepairStatus('Sidecar job timed out. Check the server logs before retrying.', 'error');
        _finishPresetMtpRepair(jobId);
        return;
    }
    try {
        const response = await fetch('/api/rapid-mlx/mtp-repair/' + encodeURIComponent(jobId), {
            headers: window.authHeaders ? window.authHeaders() : {},
        });
        const data = await response.json().catch(() => ({}));
        const job = data.job;
        if (!response.ok || !job) throw new Error(data.error || 'repair job status unavailable');
        if (_presetMtpRepairOperation === 'requalify' && job.status === 'running') {
            const mode = document.getElementById('modal-rapid-speculative-requalification-mode')?.value || 'screen';
            const estimate = mode === 'full-diagnostic'
                ? 'ETA about 60–90 min'
                : mode === 'full' ? 'ETA about 40–60 min' : 'ETA about 2–5 min';
            const steps = job.totalSteps ? ` ${job.completedSteps || 0}/${job.totalSteps} stages` : '';
            _setPresetMtpRepairStatus(`${job.message || 'Running live sidecar validation…'} —${steps} ${_formatPresetMtpRepairDuration((Date.now() - _presetMtpRepairStartedAt) / 1000)}; ${estimate}`);
        } else {
            _setPresetMtpRepairStatus(job.message || `${job.phase || 'working'}…`);
        }
        if (job.status === 'completed') {
            if (_presetMtpRepairOperation === 'requalify') {
                _renderPresetRequalificationOutcome(job.result);
            } else {
                _setPresetMtpRepairStatus('Sidecar candidate registered. Served requalification is still required.', 'success');
            }
            await _fetchSidecarsForPreset();
            _syncRapidSpeculativeEditor();
            updatePresetVram();
            _finishPresetMtpRepair(jobId);
            return;
        }
        if (job.status === 'failed' || job.status === 'cancelled') {
            _setPresetMtpRepairStatus(job.error || job.message || 'Sidecar repair did not complete.', 'error');
            _finishPresetMtpRepair(jobId);
            return;
        }
        window.setTimeout(() => _pollPresetMtpRepair(jobId), 1000);
    } catch (error) {
        _setPresetMtpRepairStatus('Repair status failed: ' + error.message, 'error');
        _finishPresetMtpRepair(jobId);
    }
}

async function _startPresetMtpRepair(operation = 'repair') {
    if (_presetMtpRepairJobId) {
        _setPresetMtpRepairStatus('A sidecar job is already running. Wait for it to finish.', 'error');
        return;
    }
    const target = document.getElementById('modal-model-path')?.value.trim() || '';
    if (!target.startsWith('/')) {
        _setPresetMtpRepairStatus('Choose a local MLX trunk before starting a managed sidecar job.', 'error');
        return;
    }
    const payload = { target, operation };
    if (operation === 'requalify') {
        payload.numSpeculativeTokens = Number(document.getElementById('modal-rapid-speculative-tokens')?.value || 3);
        payload.disableAutoK = !!document.getElementById('modal-rapid-speculative-disable-auto-k')?.checked;
        payload.requalificationMode = document.getElementById('modal-rapid-speculative-requalification-mode')?.value || 'screen';
    } else if (operation === 'repair') {
        const kind = document.getElementById('modal-rapid-speculative-repair-kind')?.value || 'source';
        const source = document.getElementById('modal-rapid-speculative-repair-source')?.value.trim() || '';
        if (!source) {
            _setPresetMtpRepairStatus('Enter a source directory, recipe path, or BF16 repository.', 'error');
            return;
        }
        if (kind === 'recipe') {
            payload.recipe = source;
        } else if (kind === 'bf16') {
            payload.bf16Source = source;
            payload.bf16Revision = document.getElementById('modal-rapid-speculative-repair-revision')?.value.trim() || '';
        } else {
            payload.source = source;
            payload.sourceFormat = 'mlx';
        }
    }

    _setPresetMtpRepairButtonsDisabled(true);
    _setPresetMtpRepairStatus(operation === 'validate' ? 'Starting sidecar validation…' : 'Starting sidecar repair…');
    try {
        const endpoint = operation === 'requalify'
            ? '/api/rapid-mlx/mtp-requalification'
            : '/api/rapid-mlx/mtp-repair';
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                ...(window.authHeaders ? window.authHeaders() : {}),
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok || !data.job?.jobId) throw new Error(data.error || 'could not start repair');
        _presetMtpRepairJobId = data.job.jobId;
        _presetMtpRepairOperation = operation;
        _presetMtpRepairStartedAt = Date.now();
        _pollPresetMtpRepair(data.job.jobId);
    } catch (error) {
        _setPresetMtpRepairStatus('Could not start sidecar job: ' + error.message, 'error');
        _setPresetMtpRepairButtonsDisabled(false);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const toggle = document.getElementById('modal-rapid-speculative-repair-toggle');
    const form = document.getElementById('modal-rapid-speculative-repair-form');
    toggle?.addEventListener('click', () => {
        if (!form) return;
        const open = !form.hidden;
        form.hidden = open;
        form.style.display = open ? 'none' : '';
        toggle.setAttribute('aria-expanded', String(!open));
        toggle.textContent = open ? 'Build / repair sidecar' : 'Hide sidecar builder';
        if (!open) _syncPresetMtpRepairKind();
    });
    document.getElementById('modal-rapid-speculative-repair-kind')
        ?.addEventListener('change', _syncPresetMtpRepairKind);
    document.getElementById('modal-rapid-speculative-repair-start')
        ?.addEventListener('click', () => _startPresetMtpRepair('repair'));
    document.getElementById('modal-rapid-speculative-repair-validate')
        ?.addEventListener('click', () => _startPresetMtpRepair('validate'));
    document.getElementById('modal-rapid-speculative-repair-requalify')
        ?.addEventListener('click', () => _startPresetMtpRepair('requalify'));
    _syncPresetMtpRepairKind();
});

document.addEventListener('change', (event) => {
    if (event.target?.id === 'modal-rapid-speculative-enabled' || event.target?.id === 'modal-rapid-speculative-source') {
        _syncRapidSpeculativeEditor();
    }
});

/* ── Trust remote code: MTP companion preflight ─────────────────────────── */

let _speculativeTrustState = {
    repoId: '',
    revision: '',
    trustRequired: false,
    loading: false,
    timeout: null,
    resolvedAt: '',
    lastRecheckAt: '',
    upstreamUnchanged: null,
    stale: false,
    estimatedMemoryBytes: null,
    mtpSidecar: null,
    mtpDepthMax: null,
};

function _timeAgo(dt) {
    if (!dt) return '';
    const d = new Date(dt);
    const diff = (Date.now() - d.getTime()) / 1000;
    if (diff < 60) return 'just now';
    if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
    if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
    return Math.floor(diff / 86400) + 'd ago';
}

function _renderSpeculativePinStatus() {
    const wrap = document.getElementById('modal-rapid-speculative-pin-status-wrap');
    const el = document.getElementById('modal-rapid-speculative-pin-status');
    if (!wrap || !el) return;

    if (!_speculativeTrustState.repoId) {
        wrap.style.display = 'none';
        return;
    }

    const rev = _speculativeTrustState.revision.substring(0, 12);
    const stale = _speculativeTrustState.stale;
    const trust = _speculativeTrustState.trustRequired;
    const mem = _speculativeTrustState.estimatedMemoryBytes;
    const sidecar = _speculativeTrustState.mtpSidecar;
    const depth = _speculativeTrustState.mtpDepthMax;

    // Determine if this is a local sidecar (no revision sha or revision is a hash)
    const isLocalSidecar = !_speculativeTrustState.revision || _speculativeTrustState.revision.length > 12;

    let parts = [];
    /* Status indicator */
    parts.push('<span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:' + (stale ? 'var(--warn,#e6a41c)' : 'var(--success,#5ce68a)') + '"></span>');

    if (isLocalSidecar) {
        /* Local sidecar: show slug + label */
        parts.push('<span>' + _speculativeTrustState.repoId + '</span>');
        parts.push('<span style="color:var(--text-muted,#888);">(local sidecar)</span>');
    } else {
        /* HF repo pin: show repo@sha */
        parts.push('<span>' + _speculativeTrustState.repoId + '@' + rev + '</span>');
    }

    /* Trust flag */
    if (trust) {
        parts.push('<span style="color:var(--err,#e65c5c);">(trust_remote_code)</span>');
    }
    /* Memory estimate */
    if (mem != null) {
        let memStr;
        if (mem >= 1073741824) {
            memStr = '~' + (mem / 1073741824).toFixed(1) + ' GB';
        } else {
            memStr = '~' + Math.round(mem / 1048576) + ' MB';
        }
        parts.push('<span style="color:var(--text-muted,#888);">~' + memStr + ' VRAM</span>');
    }
    /* Quantization info (from mtplx_runtime.json) */
    if (sidecar) {
        parts.push('<span style="color:var(--text-muted,#888);">sidecar:' + sidecar + (depth != null ? ' d' + depth : '') + '</span>');
    }
    /* Resolved time */
    if (_speculativeTrustState.resolvedAt) {
        parts.push('<span style="color:var(--text-muted,#888);">resolved ' + _timeAgo(_speculativeTrustState.resolvedAt) + '</span>');
    }

    /* Re-check button only for HF repo pins */
    if (!isLocalSidecar) {
        parts.push('<button type="button" class="pe-action-btn" id="modal-rapid-speculative-pin-recheck" style="font-size:11px; padding:2px 8px; margin-left:4px;">Re-check</button>');
    }

    // eslint-disable-next-line no-unsanitized/property -- DOMPurify sanitizes HTML
    el.innerHTML = DOMPurify.sanitize(parts.join(' '));
    wrap.style.display = '';

    /* Wire re-check button */
    const btn = document.getElementById('modal-rapid-speculative-pin-recheck');
    if (btn) {
        btn.addEventListener('click', async () => {
            if (!_speculativeTrustState.repoId) return;
            btn.disabled = true;
            btn.textContent = '…';
            try {
                const resp = await fetch(
                    '/api/hf/mtp-preflight/recheck?repo=' + encodeURIComponent(_speculativeTrustState.repoId),
                    { method: 'POST', headers: window.authHeaders ? window.authHeaders() : {} }
                );
                const data = await resp.json();
                if (!data || data.ok !== true) {
                    // eslint-disable-next-line no-unsanitized/property -- DOMPurify sanitizes HTML
                    el.innerHTML = DOMPurify.sanitize('<span style="color:var(--err,#e65c5c);">Re-check failed: ' + (data?.error || 'unknown') + '</span>');
                    setTimeout(_renderSpeculativePinStatus, 3000);
                    return;
                }
                _speculativeTrustState.revision = data.revision;
                _speculativeTrustState.trustRequired = !!data.trustRemoteCodeRequired;
                _speculativeTrustState.lastRecheckAt = data.lastRecheckAt || '';
                _speculativeTrustState.upstreamUnchanged = data.upstreamUnchanged ?? null;
                _speculativeTrustState.resolvedAt = data.resolvedAt || '';
                _speculativeTrustState.stale = false;
                _renderSpeculativePinStatus();
            } catch (e) {
                // eslint-disable-next-line no-unsanitized/property -- DOMPurify sanitizes HTML
                el.innerHTML = DOMPurify.sanitize('<span style="color:var(--err,#e65c5c);">Re-check failed: ' + e.message + '</span>');
                setTimeout(_renderSpeculativePinStatus, 3000);
            } finally {
                btn.disabled = false;
            }
        });
    }
}

async function _checkSpeculativeModelTrust(repoId) {
    const trustWrap = document.getElementById('modal-rapid-speculative-trust-wrap');
    const warningEl = document.getElementById('modal-rapid-speculative-trust-warning');
    const consentCheck = document.getElementById('modal-rapid-speculative-trust-consent');
    const pinWrap = document.getElementById('modal-rapid-speculative-pin-status-wrap');
    if (!trustWrap || !warningEl || !consentCheck) return;

    /* Reset */
    _speculativeTrustState = { repoId: '', revision: '', trustRequired: false, loading: false, timeout: null, resolvedAt: '', lastRecheckAt: '', upstreamUnchanged: null, stale: false, estimatedMemoryBytes: null, mtpSidecar: null, mtpDepthMax: null };
    trustWrap.style.display = 'none';
    consentCheck.checked = false;
    warningEl.textContent = '';
    if (pinWrap) pinWrap.style.display = 'none';

    if (!repoId || !repoId.includes('/')) return;
    if (!/^[\w._-]+\/[\w._-]+$/.test(repoId)) return;

    _speculativeTrustState.loading = true;
    try {
        const resp = await fetch(
            '/api/hf/mtp-preflight?repo=' + encodeURIComponent(repoId),
            { headers: window.authHeaders ? window.authHeaders() : {} }
        );
        const data = await resp.json();
        if (!data || data.ok !== true) {
            _speculativeTrustState.loading = false;
            return;
        }
        _speculativeTrustState.repoId = data.repoId;
        _speculativeTrustState.revision = data.revision;
        _speculativeTrustState.trustRequired = !!data.trustRemoteCodeRequired;
        _speculativeTrustState.resolvedAt = data.resolvedAt || '';
        _speculativeTrustState.lastRecheckAt = data.lastRecheckAt || '';
        _speculativeTrustState.upstreamUnchanged = data.upstreamUnchanged ?? null;
        _speculativeTrustState.stale = !!data.stale;
        _speculativeTrustState.estimatedMemoryBytes = data.estimatedMemoryBytes ?? null;
        _speculativeTrustState.mtpSidecar = data.mtpSidecar ?? null;
        _speculativeTrustState.mtpDepthMax = data.mtpDepthMax ?? null;
        _speculativeTrustState.loading = false;

        /* Pin status UI — always show when pinned */
        _renderSpeculativePinStatus();

        if (_speculativeTrustState.trustRequired) {
            let msg = 'This companion model requires trust_remote_code (custom Python code execution).';
            msg += '\nPinned to ' + data.repoId + '@' + data.revision.substring(0, 12);
            if (_speculativeTrustState.stale) {
                msg += ' (stale — consider re-checking)';
            }
            if (data.upstreamUnchanged === false) {
                msg += ' (upstream changed)';
            }
            warningEl.textContent = msg;
            consentCheck.checked = false;
            trustWrap.style.display = '';
        } else {
            warningEl.textContent = 'Pinned to ' + data.repoId + '@' + data.revision.substring(0, 12) + ' (no trust_remote_code needed)';
            trustWrap.style.display = '';
        }
    } catch {
        _speculativeTrustState.loading = false;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const modelInput = document.getElementById('modal-rapid-speculative-model');
    if (!modelInput) return;

    modelInput.addEventListener('input', () => {
        _speculativeSidecarAutoSelected = false;
        const value = (modelInput.value || '').trim();
        if (_speculativeTrustState.timeout) clearTimeout(_speculativeTrustState.timeout);
        if (!value || _speculativeTrustState.loading) return;
        _speculativeTrustState.timeout = setTimeout(() => _checkSpeculativeModelTrust(value), 600);
    });

    /* Re-check pin button */
    const recheckBtn = document.getElementById('modal-rapid-speculative-recheck');
    const recheckStatus = document.getElementById('modal-rapid-speculative-recheck-status');
    if (recheckBtn) {
        recheckBtn.addEventListener('click', async () => {
            if (!_speculativeTrustState.repoId) return;
            recheckBtn.disabled = true;
            recheckBtn.textContent = 'Re-checking...';
            try {
                const resp = await fetch(
                    '/api/hf/mtp-preflight/recheck?repo=' + encodeURIComponent(_speculativeTrustState.repoId),
                    { method: 'POST', headers: window.authHeaders ? window.authHeaders() : {} }
                );
                const data = await resp.json();
                if (!data || data.ok !== true) {
                    recheckStatus.textContent = 'Re-check failed: ' + (data?.error || 'unknown error');
                    recheckStatus.style.display = '';
                    recheckStatus.style.color = 'var(--err,#e65c5c)';
                    return;
                }
                /* Update state */
                _speculativeTrustState.revision = data.revision;
                _speculativeTrustState.trustRequired = !!data.trustRemoteCodeRequired;
                _speculativeTrustState.lastRecheckAt = data.lastRecheckAt || '';
                _speculativeTrustState.upstreamUnchanged = data.upstreamUnchanged ?? null;
                _speculativeTrustState.resolvedAt = data.resolvedAt || '';
                _speculativeTrustState.stale = false;

                /* Update warning text */
                const warningEl = document.getElementById('modal-rapid-speculative-trust-warning');
                if (_speculativeTrustState.trustRequired) {
                    let msg = 'This companion model requires trust_remote_code (custom Python code execution).';
                    msg += '\nPinned to ' + data.repoId + '@' + data.revision.substring(0, 12);
                    if (data.upstreamUnchanged === false) {
                        msg += ' (upstream changed)';
                    }
                    warningEl.textContent = msg;
                } else {
                    warningEl.textContent = 'Pinned to ' + data.repoId + '@' + data.revision.substring(0, 12) + ' (no trust_remote_code needed)';
                }

                recheckStatus.textContent = 'Pin verified at ' + new Date(data.lastRecheckAt).toLocaleTimeString();
                recheckStatus.style.display = '';
                recheckStatus.style.color = 'var(--success,#5ce68a)';
            } catch (e) {
                recheckStatus.textContent = 'Re-check failed: ' + e.message;
                recheckStatus.style.display = '';
                recheckStatus.style.color = 'var(--err,#e65c5c)';
            } finally {
                recheckBtn.disabled = false;
                recheckBtn.textContent = 'Re-check pin';
            }
        });
    }
});

function _buildFormPreset(existing) {
    updateBundleSelectionFromEditor();
    if (existing.backend === 'rapid_mlx') {
        const rapidPort = intOrNull('modal-port');
        return {
            ...existing,
            name: strVal('modal-name'),
            port: rapidPort,
            rapid_mlx: existing.rapid_mlx ? {
                ...existing.rapid_mlx,
                // model_source preserved via spread above; input is display-only for Rapid-MLX.
                // Writing model_path from display_name creates stale noise (DoD item 22, gap 3).
                port: rapidPort,
                ...(function() {
                    const et = nullableBoolOpt('modal-rapid-enable-thinking');
                    const re = strVal('modal-rapid-reasoning-effort');
                    const out = {};
                    out.enable_thinking = et;
                    // reasoning_effort removed: config field exists but argv builder does not emit --reasoning-effort.
                    // Phase 6 Part B: prefix cache enabled toggle.
                    out.cache_mode = strVal('modal-rapid-cache-mode') || 'custom';
                    const pceInput = document.getElementById('modal-rapid-prefix-cache-enabled');
                    const cacheMib = Number(document.getElementById('modal-rapid-cache-memory-mib')?.value || 0);
                    const retainedCacheEnabled = !!pceInput?.checked && cacheMib > 0;
                    out.prefix_cache_enabled = retainedCacheEnabled;
                    out.retained_cache_mib = retainedCacheEnabled ? cacheMib : null;
                    out.disk_checkpoint_interval = 0;
                    // Entry count is meaningless with the cache off, so it follows the toggle
                    // rather than persisting a number that would never reach the runtime --
                    // the argv builder emits --hybrid-cache-entries on the field alone, without
                    // consulting the toggle, so a stale value would still reach the server.
                    const hybridEntries = Number(strVal('modal-rapid-hybrid-cache-entries') || 0);
                    out.hybrid_cache_entries = retainedCacheEnabled && hybridEntries > 0 ? hybridEntries : null;
                    // Phase 7: Rapid-MLX advanced controls (D6 catalog IDs).
                    // Every one of these writes unconditionally, including when the choice is
                    // "unset"/"auto". `out` is spread over the stored rapid_mlx below, so a key
                    // the save path omits keeps its previous value -- which meant choosing
                    // "(unset)" or "Auto" on a preset that already had a value silently did
                    // nothing and the old value kept reaching argv. The load path fills all of
                    // these controls, so re-reading them preserves untouched values; serde
                    // reads null into the same None an absent key would have produced.
                    const kvDtype = strVal('modal-rapid-kv-cache-dtype');
                    const tqMode = strVal('modal-rapid-turboquant-mode');
                    const toolParser = strVal('modal-rapid-tool-call-parser');
                    const reasoningParser = strVal('modal-rapid-reasoning-parser');
                    const samplingMode = strVal('modal-rapid-sampling-mode');
                    const rmInput = document.getElementById('modal-rapid-reasoning-mode');
                    out.kv_cache_dtype = kvDtype || null;
                    out.turboquant_mode = tqMode && tqMode !== 'auto' ? tqMode : null;
                    out.tool_call_parser = toolParser || null;
                    out.reasoning_parser = reasoningParser || null;
                    out.sampling_mode = samplingMode && samplingMode !== 'auto' ? samplingMode : null;
                    if (rmInput) out.reasoning_mode = rmInput.checked ? 'on' : 'off';
                    out.no_thinking = rmInput ? !rmInput.checked : false;
                    out.auto_tool_choice = !!document.getElementById('modal-rapid-auto-tool-choice')?.checked;
                    const specEnabled = !!document.getElementById('modal-rapid-speculative-enabled')?.checked;
                    const specSource = strVal('modal-rapid-speculative-source') || 'embedded';
                    const specModel = strVal('modal-rapid-speculative-model').trim();
                    out.speculative_config = specEnabled ? {
                        method: 'mtp',
                        ...(specSource === 'external' ? { model: specModel } : {}),
                        num_speculative_tokens: Number(strVal('modal-rapid-speculative-tokens') || RAPID_MLX_DEFAULT_SPECULATIVE_TOKENS),
                        disable_auto_k: !!document.getElementById('modal-rapid-speculative-disable-auto-k')?.checked,
                    } : null;
                    // Trust remote code consent for MTP companion: only set when required
                    // and explicitly consented. If consent checkbox is off but trust is
                    // required, launch will fail-closed — that's correct and safe.
                    const tcCheck = document.getElementById('modal-rapid-speculative-trust-consent');
                    if (_speculativeTrustState.trustRequired && tcCheck?.checked && _speculativeTrustState.repoId && _speculativeTrustState.revision) {
                        out.trust_remote_code_consent = _speculativeTrustState.repoId + '@' + _speculativeTrustState.revision;
                    } else {
                        out.trust_remote_code_consent = null;
                    }
                    // prefill_step_size is a plain u32 backend-side, not an Option, so it takes
                    // the control's value directly rather than a null.
                    out.prefill_step_size = Number(strVal('modal-rapid-prefill-step-size') || 512);
                    // Throughput / memory. Auto ('') means "omit the flag and take the runtime
                    // default", which is not the same as an explicit value -- so Auto writes
                    // null, not a placeholder number. It has to write *something*: `out` is
                    // spread over the stored rapid_mlx, so an omitted key leaves the old value
                    // in place and selecting Auto would never stick. The load path fills every
                    // one of these controls, so re-reading them preserves untouched values.
                    const numOrNull = (id) => { const v = strVal(id); return v ? Number(v) : null; };
                    out.gpu_memory_utilization = numOrNull('modal-rapid-gpu-memory-utilization');
                    out.max_num_seqs = numOrNull('modal-rapid-max-num-seqs');
                    out.max_concurrent_requests = numOrNull('modal-rapid-max-concurrent-requests');
                    out.prefill_batch_size = numOrNull('modal-rapid-prefill-batch-size');
                    out.completion_batch_size = numOrNull('modal-rapid-completion-batch-size');
                    const pflash = strVal('modal-rapid-pflash-policy');
                    if (pflash) out.pflash_policy = pflash;
                    // Sampling defaults (--default-* flags for Rapid-MLX)
                    const temp = floatOrNull('modal-temperature');
                    const topP = floatOrNull('modal-top-p');
                    const topK = intOrNull('modal-top-k');
                    const minP = floatOrNull('modal-min-p');
                    const repeatPen = floatOrNull('modal-repeat-penalty');
                    const presencePen = floatOrNull('modal-presence-penalty');
                    const frequencyPen = floatOrNull('modal-rapid-frequency-penalty');
                    const maxTok = intOrNull('modal-max-tokens');
                    // Same rule as above: clearing one of these inputs has to write null, or
                    // the spread would keep the old default and it could never be cleared.
                    out.default_temperature = temp;
                    out.default_top_p = topP;
                    out.default_top_k = topK;
                    out.default_min_p = minP;
                    out.default_repetition_penalty = repeatPen;
                    out.default_presence_penalty = presencePen;
                    out.default_frequency_penalty = frequencyPen;
                    out.max_tokens = maxTok;
                    return out;
                })(),
            } : null,
        };
    }
    const modelSource = normalizeModelSourceInput(strVal('modal-model-path'));
    const fitEnabled = nullableBoolOpt('modal-fit-enabled');
    return {
        // Spread ALL existing fields first — preserves wizard-set values not shown in the editor
        ...existing,
        // Override only what the editor manages
        name: strVal('modal-name'),
        model_path: modelSource.model_path,
        hf_repo: modelSource.hf_repo,
        alias: strVal('modal-alias') || null,
        mmproj: strVal('modal-mmproj') || null,
        mmproj_offload: nullableBoolOpt('modal-mmproj-offload'),
        chat_template_file: strVal('modal-chat-template-file') || null,
        gpu_layers: intOrNull('modal-gpu-layers'),
    no_mmap: document.getElementById('modal-no-mmap').checked,
    load_mode: strVal('modal-load-mode') || null,
        mlock: document.getElementById('modal-mlock').checked,
        context_size: parseInt(document.getElementById('modal-context-size').value) || 128000,
        ctk: strVal('modal-ctk') || 'q8_0',
        ctv: strVal('modal-ctv') || 'f16',
        flash_attn: strVal('modal-flash-attn'),
        kv_unified: nullableBoolOpt('modal-kv-unified'),
        cache_mode: strVal('modal-cache-mode') || 'custom',
        cache_ram_mib: intOrNull('modal-cache-ram-mib'),
        batch_size: parseInt(document.getElementById('modal-batch-size').value) || 2048,
        ubatch_size: parseInt(document.getElementById('modal-ubatch-size').value) || 2048,
        parallel_slots: parseInt(document.getElementById('modal-parallel-slots').value) || 1,
        prio: intOrNull('modal-prio'),
        prio_batch: intOrNull('modal-prio-batch'),
        verbosity: intOrNull('modal-verbosity') ?? 4,
        no_cont_batching: document.getElementById('modal-no-cont-batching').checked,
        swa_full: document.getElementById('modal-swa-full').checked,
        ctx_checkpoints: intOrNull('modal-ctx-checkpoints'),
        checkpoint_min_step: intOrNull('modal-checkpoint-min-step'),
        cache_reuse: intOrNull('modal-cache-reuse'),
        cache_idle_slots: nullableBoolOpt('modal-cache-idle-slots'),
        threads: intOrNull('modal-threads'),
        threads_batch: intOrNull('modal-threads-batch'),
        temperature: floatOrNull('modal-temperature'),
        top_p: floatOrNull('modal-top-p'),
        top_k: intOrNull('modal-top-k'),
        min_p: floatOrNull('modal-min-p'),
        repeat_penalty: floatOrNull('modal-repeat-penalty'),
        repeat_last_n: intOrNull('modal-repeat-last-n'),
        presence_penalty: floatOrNull('modal-presence-penalty'),
        enable_thinking: nullableBoolOpt('modal-enable-thinking'),
        preserve_thinking: nullableBoolOpt('modal-preserve-thinking'),
        tool_call_format: strVal('modal-tool-call-format') || null,
        reasoning: strVal('modal-reasoning') || null,
        llama_reasoning_effort: strVal('modal-llama-reasoning-effort') || 'default',
        llama_reasoning_format: valOrNull('modal-llama-reasoning-format'),
        llama_reasoning_preserve: nullableBoolOpt('modal-llama-reasoning-preserve'),
        reasoning_budget: intOrNull('modal-reasoning-budget'),
        reasoning_budget_message: (document.getElementById('modal-reasoning-budget-message').value || '').replace(/\\n/g, '\n') || null,
        tensor_split: strVal('modal-tensor-split'),
        split_mode: strVal('modal-split-mode'),
        main_gpu: intOrNull('modal-main-gpu'),
        n_cpu_moe: intOrNull('modal-n-cpu-moe'),
        rope_scaling: strVal('modal-rope-scaling'),
        rope_freq_base: floatOrNull('modal-rope-freq-base'),
        rope_freq_scale: floatOrNull('modal-rope-freq-scale'),
        spec_type: strVal('modal-spec-type') || null,
        spec_default: document.getElementById('modal-spec-default')?.checked || false,
        ngram_spec: false,
        spec_ngram_size: intOrNull('modal-spec-ngram-size'),
        draft_min: intOrNull('modal-draft-min'),
        draft_max: intOrNull('modal-draft-max'),
        spec_draft_n_max: intOrNull('modal-spec-draft-n-max'),
        spec_draft_n_min: intOrNull('modal-spec-draft-n-min'),
        spec_draft_p_min: floatOrNull('modal-spec-draft-p-min'),
        spec_draft_type_k: valOrNull('modal-spec-draft-type-k'),
        spec_draft_type_v: valOrNull('modal-spec-draft-type-v'),
        draft_model: strVal('modal-draft-model'),
        spec_draft_ngl: intOrNull('modal-spec-draft-ngl'),
        spec_draft_device: valOrNull('modal-spec-draft-device'),
        spec_draft_n_cpu_moe: intOrNull('modal-spec-draft-n-cpu-moe'),
        spec_draft_cpu_moe: (intOrNull('modal-spec-draft-n-cpu-moe') ?? 0) > 0,
        bind_host: strVal('modal-bind-host') || null,
        port: intOrNull('modal-port'),
        api_key: strVal('modal-api-key') || null,
        max_tokens: intOrNull('modal-max-tokens'),
        seed: intOrNull('modal-seed'),
        fit_enabled: fitEnabled,
        fit_target: fitEnabled === true ? (strVal('modal-fit-target') || null) : null,
        system_prompt_file: strVal('modal-system-prompt-file'),
        grammar: getStructuredOutputMode() === 'grammar' ? (document.getElementById('modal-grammar').value.trim() || null) : null,
        json_schema: getStructuredOutputMode() === 'json_schema' ? (document.getElementById('modal-json-schema').value.trim() || null) : null,
        extra_args: strVal('modal-extra-args'),
        spec_draft_p_split: floatOrNull('modal-spec-draft-p-split'),
        image_min_tokens: intOrNull('modal-image-min-tokens'),
        image_max_tokens: intOrNull('modal-image-max-tokens'),
        ...(_presetBundleDraft ? {
            bundle: {
                ..._presetBundleDraft,
                workload_policy: strVal('modal-workload-policy') || _presetBundleDraft.workload_policy || 'general_chat',
            },
        } : {}),
    };
}
const CHANGE_LABELS = {
    name: 'Name', model_path: 'Model (local path or HF repo)', hf_repo: 'HuggingFace Repo',
    alias: 'Server Alias', mmproj: 'Multimodal Projector', chat_template_file: 'Chat Template File',
    mmproj_offload: 'Projector Offload',
    image_min_tokens: 'Vision Min Tokens', image_max_tokens: 'Vision Max Tokens',
    gpu_layers: 'GPU Layers', no_mmap: 'no-mmap', mlock: 'mlock',
    context_size: 'Context Size', ctk: 'KV Key Type', ctv: 'KV Value Type',
    flash_attn: 'Flash Attn', kv_unified: 'KV Unified', cache_mode: 'Prompt Cache Mode', cache_ram_mib: 'Prefix Cache RAM',
    fit_enabled: 'Fit to VRAM', fit_target: 'Fit Target',
    batch_size: 'Batch Size', ubatch_size: 'Micro-batch', parallel_slots: 'Parallel Slots',
    prio: 'Thread Priority', prio_batch: 'Batch Priority', verbosity: 'Server Log Verbosity', cache_idle_slots: 'Cache Idle Slots',
    threads: 'Threads (-t)', threads_batch: 'Batch Threads (-tb)',
    temperature: 'Temperature', top_p: 'Top-P', top_k: 'Top-K',
 min_p: 'Min-P', repeat_penalty: 'Repeat Penalty', repeat_last_n: 'Repeat Last N', presence_penalty: 'Presence Penalty',
    enable_thinking: 'Thinking Mode', preserve_thinking: 'Preserve Thinking',
    tool_call_format: 'Tool Call Format',
    reasoning: 'Reasoning', reasoning_budget: 'Reasoning Budget',
    reasoning_budget_message: 'Reasoning Budget Message',
    llama_reasoning_effort: 'llama.cpp Reasoning Effort',
    llama_reasoning_format: 'llama.cpp Reasoning Format',
    llama_reasoning_preserve: 'llama.cpp Preserve Reasoning',
    tensor_split: 'Tensor Split', split_mode: 'Split Mode', main_gpu: 'Main GPU',
    n_cpu_moe: 'CPU MoE Threads',
    rope_scaling: 'RoPE Scaling', rope_freq_base: 'RoPE Freq Base', rope_freq_scale: 'RoPE Freq Scale',
    spec_type: 'Speculative Mode', spec_default: 'Spec Defaults',
    spec_ngram_size: 'N-gram Size',
    draft_min: 'Draft Min', draft_max: 'Draft Max', spec_draft_n_max: 'MTP Depth',
    spec_draft_n_min: 'MTP Draft N Min', spec_draft_p_min: 'MTP Draft P Min',
    spec_draft_ngl: 'Draft GPU Layers', spec_draft_device: 'Draft Device',
    spec_draft_n_cpu_moe: 'Draft CPU MoE', draft_model: 'Draft Model',
    bind_host: 'Bind Host', port: 'Port', api_key: 'API Key', max_tokens: 'Max Tokens',
    seed: 'Seed',
    system_prompt_file: 'System Prompt File', grammar: 'Grammar', json_schema: 'JSON Schema', extra_args: 'Extra Args',
};

// Every Rapid-MLX setting lives under preset.rapid_mlx.*, which _buildChangeSummary used to
// ignore entirely -- so the "confirm these changes" dialog was blind to the whole backend and
// a user could change reusable prompt storage or speculative decoding and be shown nothing.
const RAPID_CHANGE_LABELS = {
    port: 'Port', model_source: 'Model', enable_thinking: 'Thinking Mode',
    reasoning_mode: 'Reasoning Mode', reasoning_parser: 'Reasoning Parser',
    tool_call_parser: 'Tool-call Parser', sampling_mode: 'Sampling Mode',
    kv_cache_dtype: 'KV Cache Type', turboquant_mode: 'Reusable Prompt Storage',
    cache_mode: 'Prompt Cache Mode', prefix_cache_enabled: 'Prefix Cache', retained_cache_mib: 'Retained Cache (MiB)',
    prefill_step_size: 'Prefill Step Size', hybrid_mode: 'Hybrid Architecture',
    gpu_memory_utilization: 'GPU Memory Utilization', max_num_seqs: 'Max Batched Sequences',
    max_concurrent_requests: 'Max Concurrent Requests', pflash_policy: 'PFlash',
    hybrid_cache_entries: 'Retained Prefix Entries',
    prefill_batch_size: 'Prefill Batch Size', completion_batch_size: 'Completion Batch Size',
    default_temperature: 'Temperature', default_top_p: 'Top-P', default_top_k: 'Top-K',
    default_min_p: 'Min-P', default_repetition_penalty: 'Repeat Penalty',
    default_presence_penalty: 'Presence Penalty', default_frequency_penalty: 'Frequency Penalty',
    max_tokens: 'Max Tokens',
};

function _buildChangeSummary(existing, incoming) {
    const changes = [];
    const fmt = v => {
        if (v == null || v === '') return '(none)';
        if (typeof v === 'number') return formatNumberForInput(v);
        return String(v);
    };
    for (const key of Object.keys(CHANGE_LABELS)) {
        const prev = existing[key] ?? null;
        const next = incoming[key] ?? null;
        // Compare formatted representations so float32 noise (0.949999... vs 0.95) doesn't
        // produce a false-positive change that shows as "0.95 → 0.95" in the summary.
        const fPrev = fmt(prev);
        const fNext = fmt(next);
        if (fPrev !== fNext) {
            changes.push({ label: CHANGE_LABELS[key], from: fPrev, to: fNext });
        }
    }
    const prevRapid = existing?.rapid_mlx;
    const nextRapid = incoming?.rapid_mlx;
    if (prevRapid || nextRapid) {
        for (const key of Object.keys(RAPID_CHANGE_LABELS)) {
            const fPrev = fmt(prevRapid?.[key] ?? null);
            const fNext = fmt(nextRapid?.[key] ?? null);
            if (fPrev !== fNext) {
                changes.push({ label: RAPID_CHANGE_LABELS[key], from: fPrev, to: fNext });
            }
        }
    }
    const prevBundle = existing?.bundle ? JSON.stringify(existing.bundle) : '';
    const nextBundle = incoming?.bundle ? JSON.stringify(incoming.bundle) : '';
    if (prevBundle !== nextBundle) {
        changes.push({
            label: 'Model variants and launch choices',
            from: existing?.bundle ? `${existing.bundle.artifacts?.length || 0} artifacts` : 'single artifact',
            to: incoming?.bundle ? `${incoming.bundle.artifacts?.length || 0} artifacts` : 'single artifact',
        });
    }
    return changes;
}

// ── CRUD ───────────────────────────────────────────────────────────────────────

export async function savePreset(event) {
    event.preventDefault();
    clearFieldErrors();

    const id = document.getElementById('modal-preset-id').value;
    const saveBtn = document.getElementById('btn-modal-save');
    const existing = id
        ? (sessionState.presets.find(p => p.id === id) || {})
        : (newPresetSeed || {});
    const preset = _buildFormPreset(existing);

    // Inline validation
    let valid = true;
    if (!preset.name) {
        markFieldError('modal-name', 'Preset name is required.');
        valid = false;
    }
    const rapidMlx = preset.rapid_mlx;
    const hasModelSource = preset.backend === 'rapid_mlx'
        ? !!(rapidMlx?.model_source || rapidMlx?.model_source_view)
        : !!(preset.model_path || preset.hf_repo);
    if (!hasModelSource) {
        markFieldError('modal-model-path', 'Model path or HuggingFace repo is required.');
        valid = false;
    }
    if (preset.backend === 'rapid_mlx' && !preset.rapid_mlx?.port) {
        markFieldError('modal-port', 'Rapid-MLX requires a valid server port.');
        valid = false;
    }
    const gpuLayers = parseInt(document.getElementById('modal-gpu-layers').value, 10);
    if (!isNaN(gpuLayers) && gpuLayers < -1) {
        markFieldError('modal-gpu-layers', 'GPU layers must be -1, 0, or a positive number.');
        valid = false;
    }
    const ctxSize = parseInt(document.getElementById('modal-context-size').value, 10);
    if (!isNaN(ctxSize) && ctxSize <= 0) {
        markFieldError('modal-context-size', 'Context size must be a positive number.');
        valid = false;
    }
    const threads = parseInt(document.getElementById('modal-threads').value, 10);
    if (!isNaN(threads) && threads !== -1 && threads < 1) {
        markFieldError('modal-threads', 'Threads must be -1 (auto) or 1 or higher.');
        valid = false;
    }
    const threadsBatch = parseInt(document.getElementById('modal-threads-batch').value, 10);
    if (!isNaN(threadsBatch) && threadsBatch !== -1 && threadsBatch < 1) {
        markFieldError('modal-threads-batch', 'Batch threads must be -1 (auto) or 1 or higher.');
        valid = false;
    }
    if (!valid) {
        scrollToFirstError();
        showToast('Please fix the highlighted error(s)', 'error');
        return;
    }

    // For edits: show change summary and require confirmation
    if (id && saveBtn.dataset.confirmed !== 'yes') {
        const changes = _buildChangeSummary(existing, preset);
        if (changes.length > 0) {
            const summary = document.getElementById('preset-change-summary');
            const list = document.getElementById('preset-change-summary-list');
            const back = document.getElementById('preset-modal-back');
            const cancel = document.getElementById('preset-modal-cancel');
            if (summary && list) {
                list.innerHTML = '';
                changes.forEach(({ label, from, to }) => {
                    const li = document.createElement('li');
                    li.className = 'preset-change-item';
                    li.innerHTML = `<span class="preset-change-field">${escapeHtml(label)}</span> <span class="preset-change-from">${escapeHtml(from)}</span><span class="preset-change-arrow">→</span><span class="preset-change-to">${escapeHtml(to)}</span>`;
                    list.appendChild(li);
                });
                summary.style.display = '';
                if (back) back.style.display = '';
                if (cancel) cancel.style.display = 'none';
                saveBtn.textContent = 'Confirm Save';
                saveBtn.dataset.confirmed = 'yes';

                // If user edits any field after seeing the summary, reset so the
                // next Save click rebuilds the summary with all accumulated changes.
                const form = document.getElementById('preset-form');
                if (form) {
                    const resetOnEdit = () => { _hideSummary(); form.removeEventListener('input', resetOnEdit); form.removeEventListener('change', resetOnEdit); };
                    form.addEventListener('input', resetOnEdit);
                    form.addEventListener('change', resetOnEdit);
                }
            }
            return;
        }
    }

    saveBtn.classList.add('saving');
    saveBtn.textContent = 'Saving...';
    _hideSummary();

    try {
        let resp;
        let savedId;
        if (id) {
            resp = await fetch('/api/presets/' + encodeURIComponent(id), {
                method: 'PUT',
                headers: window.authHeaders
                    ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
                    : { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ...preset, expected_revision: existing.revision ?? 1 }),
            });
            if (!resp.ok) {
                const err = await resp.text().catch(() => 'Unknown error');
                showToast('Save failed: ' + err, 'error');
                return;
            }
            savedId = id;
        } else {
            resp = await fetch('/api/presets', {
                method: 'POST',
                headers: window.authHeaders
                    ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
                    : { 'Content-Type': 'application/json' },
                body: JSON.stringify(preset),
            });
            if (!resp.ok) {
                const err = await resp.text().catch(() => 'Unknown error');
                showToast('Save failed: ' + err, 'error');
                return;
            }
            const data = await resp.json();
            savedId = data.id || null;
        }
        closePresetModal();
        await loadPresets(savedId);
        if (savedId) syncSelectedPresetSelection(savedId, { userIntent: true, persist: true });
        showToast('Preset saved', 'success');

        // If this is the active preset and the server is running, offer a reload.
        const activePresetId = sessionState.activeSessionPresetId || '';
        if (savedId && activePresetId === savedId && sessionState.serverRunning) {
            _offerRestartAfterPresetSave(savedId);
        }
    } catch (err) {
        showToast('Save failed: ' + err.message, 'error');
    } finally {
        saveBtn.classList.remove('saving');
        saveBtn.textContent = 'Save';
    }
}

export async function copyPreset() {
    const id = document.getElementById('preset-select').value;
    return duplicatePresetById(id, { reopenEditor: false });
}

function buildDuplicatePresetName(baseName) {
    const base = baseName || 'Preset';
    let copyName = base + ' (copy)';
    let suffixNum = 2;
    while (sessionState.presets.some(pr => pr.name === copyName)) {
        copyName = base + ' (copy ' + suffixNum + ')';
        suffixNum++;
    }
    return copyName;
}

async function duplicatePresetById(id, options = {}) {
    const { reopenEditor = false } = options;
    const p = sessionState.presets.find(pr => pr.id === id);
    if (!p) { showToast('No preset selected', 'warn'); return; }

    try {
        const resp = await fetch('/api/presets/' + encodeURIComponent(id) + '/copy', {
            method: 'POST',
            headers: window.authHeaders
                ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
                : { 'Content-Type': 'application/json' },
            body: JSON.stringify({ expected_revision: p.revision ?? 1, new_name: buildDuplicatePresetName(p.name) }),
        });
        if (!resp.ok) {
            const err = await resp.text().catch(() => 'Unknown error');
            showToast('Copy failed: ' + err, 'error');
            return;
        }
        const data = await resp.json();
        const newId = data.preset?.id || data.id || null;
        await loadPresets(newId);
        if (newId) {
            syncSelectedPresetSelection(newId, { userIntent: true, persist: true });
            if (reopenEditor) openPresetModal('edit');
        }
        showToast(reopenEditor && newId ? 'Preset duplicated - editing copy' : 'Preset copied', 'success');
    } catch (err) {
        showToast('Copy failed: ' + err.message, 'error');
    }
}

export async function deletePreset() {
    const id = document.getElementById('preset-select').value;
    const p = sessionState.presets.find(pr => pr.id === id);
    if (!p) { showToast('No preset selected', 'warn'); return; }

    let catalogEtag;
    try {
        catalogEtag = await freshPresetCatalogEtag();
    } catch (error) {
        showToast(`Delete cancelled: ${error.message || error}`, 'error');
        return;
    }
    const confirmed = await _showConfirm('Delete preset', 'Delete preset "' + escapeHtml(p.name) + '"? This cannot be undone.');
    if (!confirmed) return;

    try {
        const resp = await fetch('/api/presets/' + encodeURIComponent(id), {
            method: 'DELETE',
            headers: { ...(window.authHeaders ? window.authHeaders() : {}), 'Content-Type': 'application/json' },
            body: JSON.stringify({ expected_revision: p.revision ?? 1, expected_catalog_etag: catalogEtag, confirmation: 'DELETE PRESET' }),
        });
        if (!resp.ok) {
            const err = await resp.text().catch(() => 'Unknown error');
            showToast('Delete failed: ' + err, 'error');
            return;
        }
        await loadPresets(null);
        showToast('Preset deleted', 'success');
    } catch (err) {
        showToast('Delete failed: ' + err.message, 'error');
    }
}

export async function resetPresets() {
    let catalogEtag;
    try {
        catalogEtag = await freshPresetCatalogEtag();
    } catch (error) {
        showToast(`Reset cancelled: ${error.message || error}`, 'error');
        return;
    }
    const ok = await showConfirmDialog(
        'Reset presets',
        'Reset all presets to built-in defaults? Custom presets will be removed.',
        'Reset all'
    );
    if (!ok) return;
    try {
        const resp = await fetch('/api/presets/reset', {
            method: 'POST',
            headers: { ...(window.authHeaders ? window.authHeaders() : {}), 'Content-Type': 'application/json' },
            body: JSON.stringify({ expected_catalog_etag: catalogEtag, confirmation: 'RESET PRESETS' }),
        });
        if (!resp.ok) {
            const err = await resp.text().catch(() => 'Unknown error');
            showToast('Reset failed: ' + err, 'error');
            return;
        }
        await loadPresets();
        showToast('Presets reset to defaults', 'success');
    } catch (err) {
        showToast('Reset failed: ' + err.message, 'error');
    }
}

// ── Preset Editor Nav ─────────────────────────────────────────────────────────

function initPresetEditorNav() {
    const modal = document.getElementById('preset-modal');
    if (!modal || _presetEditorNavInitialized) return;

    // Delegate from the stable modal shell. The editor moves nav items between
    // groups for Rapid-MLX, so one-time listeners on the original NodeList are
    // fragile and can miss a click after a reconfiguration.
    modal.addEventListener('click', event => {
        const btn = event.target.closest?.('.preset-nav-item');
        if (!btn || !modal.contains(btn)) return;
        const target = btn.dataset.section;
        modal.querySelectorAll('.preset-nav-item').forEach(item => {
            const active = item === btn;
            item.classList.toggle('active', active);
            item.setAttribute('aria-pressed', active ? 'true' : 'false');
        });
        modal.querySelectorAll('.preset-editor-section').forEach(panel => {
            panel.classList.toggle('active', panel.dataset.section === target);
        });
        modal.querySelector('.modal-body')?.scrollTo({ top: 0, behavior: 'instant' });
    });
    _presetEditorNavInitialized = true;
}

// ── Model-family generation defaults ─────────────────────────────────────────

// fillEmpty=true: fill blank sampling fields with model defaults (for new presets).
// fillEmpty=false: only render the preset pill switchers, don't overwrite existing values.
async function _suggestGenerationDefaults(modelPath, fillEmpty = true) {
    const modelName = modelPath.split(/[/\\]/).pop() || modelPath;
    const backend = _currentModalPreset()?.backend || 'llama_cpp';
    try {
        const headers = window.authHeaders
            ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
            : { 'Content-Type': 'application/json' };

        // Try a quick GGUF metadata read to get arch for finetunes with non-canonical names
        let ggufArch = '';
        if (modelPath.startsWith('/')) {
            try {
                const ir = await fetch('/api/models/gguf-meta', {
                    method: 'POST', headers,
                    body: JSON.stringify({ model_path: modelPath }),
                });
                if (ir.ok) {
                    const id = await ir.json();
                    if (id.ok && id.architecture) ggufArch = id.architecture;
                }
            } catch (_) { /* non-fatal */ }
        }

        const resp = await fetch('/api/model-defaults', {
            method: 'POST',
            headers,
            body: JSON.stringify({ model_name_or_repo: modelName, size_bytes: 0, tags: [], gguf_arch: ggufArch, backend }),
        });
        if (!resp.ok) return;
        const d = await resp.json();
        if (d.error) return;
        const defaults = d.defaults || d;
        const coverage = backend === 'rapid_mlx'
            ? (d.modes?.[0]?.rapid_mlx_coverage || {})
            : (d.modes?.[0]?.llama_cpp_coverage || {});
        const canApplyDefaults = Object.values(coverage).some(Boolean);

        if (fillEmpty && canApplyDefaults) {
            // Only fill fields the user hasn't already set
            const fill = (id, val) => {
                const el = document.getElementById(id);
                if (el && el.value === '') numOrEmpty(id, val);
            };
            fill('modal-temperature', defaults.temperature ?? null);
            fill('modal-top-p', defaults.top_p ?? null);
            fill('modal-top-k', defaults.top_k ?? null);
            fill('modal-min-p', defaults.min_p ?? null);
            fill('modal-repeat-penalty', defaults.repeat_penalty ?? null);
            fill('modal-presence-penalty', defaults.presence_penalty ?? null);
            fill('modal-max-tokens', defaults.max_tokens ?? null);
            _fillSelectIfEmpty('modal-enable-thinking', defaults.enable_thinking);
            _fillSelectIfEmpty('modal-preserve-thinking', defaults.preserve_thinking);
            // tool_call_format is intentionally never auto-filled from model-family
            // defaults — it's a template-level opt-in, left blank unless the user
            // explicitly selects "json".
            _fillSelectIfEmpty('modal-reasoning', defaults.reasoning ? 'on' : 'off');
            fill('modal-reasoning-budget', defaults.reasoning_budget ?? null);
            const msgEl = document.getElementById('modal-reasoning-budget-message');
            if (msgEl && msgEl.value === '' && defaults.reasoning_budget_message != null) {
                msgEl.value = defaults.reasoning_budget_message.replace(/\n/g, '\\n');
            }
        }
        _renderGenerationPresetPills(d.modes || []);
    } catch (_) {
        // Silent — best-effort only
    }
}

function _fillSelectIfEmpty(id, value) {
    const el = document.getElementById(id);
    if (!el || el.value !== '' || value == null) return;
    el.value = typeof value === 'boolean' ? String(value) : String(value);
}

function _renderGenerationPresetPills(presets) {
    const container = document.getElementById('modal-generation-presets');
    if (!container) return;
    if (!presets || presets.length <= 1) {
        container.style.display = 'none';
        container.innerHTML = '';
        return;
    }

    container.style.display = 'flex';
    container.style.cssText = 'display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:12px;';
    container.innerHTML = '';

    const label = document.createElement('span');
    label.style.cssText = 'font-size:11px;color:var(--color-text-muted);flex-shrink:0;';
    label.textContent = 'Mode:';
    container.appendChild(label);

    presets.forEach((preset, index) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'sampling-preset-pill' + (index === 0 ? ' active' : '');
        btn.textContent = preset.name;
        const provenance = preset.provenance?.unsloth?.url || preset.provenance?.model_author?.source;
        const badges = (preset.workload_badges || []).join(', ');
        btn.title = [preset.description, badges && `Best for: ${badges}`, provenance && `Source: ${provenance}`]
            .filter(Boolean).join('\n');
        btn.addEventListener('click', () => {
            container.querySelectorAll('.sampling-preset-pill').forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            _applyGenerationPreset(preset);
        });
        container.appendChild(btn);
    });
}

function _applyGenerationPreset(preset) {
    if (preset.id === 'model_default') {
        ['modal-temperature', 'modal-top-p', 'modal-top-k', 'modal-min-p', 'modal-repeat-penalty', 'modal-presence-penalty', 'modal-rapid-frequency-penalty', 'modal-max-tokens', 'modal-reasoning-budget']
            .forEach(id => { const el = document.getElementById(id); if (el) el.value = ''; });
        ['modal-enable-thinking', 'modal-preserve-thinking', 'modal-tool-call-format', 'modal-reasoning']
            .forEach(id => setOpt(id, ''));
        setVal('modal-reasoning-budget-message', '');
        return;
    }
    if (preset.id === 'custom') return;
    numOrEmpty('modal-temperature', preset.temperature);
    numOrEmpty('modal-top-p', preset.top_p);
    numOrEmpty('modal-top-k', preset.top_k);
    numOrEmpty('modal-min-p', preset.min_p);
    numOrEmpty('modal-repeat-penalty', preset.repeat_penalty);
    numOrEmpty('modal-presence-penalty', preset.presence_penalty);
    numOrEmpty('modal-max-tokens', preset.max_tokens);
    setOpt('modal-enable-thinking', preset.enable_thinking == null ? '' : String(!!preset.enable_thinking));
    setOpt('modal-preserve-thinking', preset.preserve_thinking == null ? '' : String(!!preset.preserve_thinking));
    setOpt('modal-tool-call-format', preset.tool_call_format || '');
    setOpt('modal-reasoning', preset.reasoning ? 'on' : 'off');
    numOrEmpty('modal-reasoning-budget', preset.reasoning_budget);
    setVal('modal-reasoning-budget-message', (preset.reasoning_budget_message || '').replace(/\n/g, '\\n'));
    const samplingMode = document.getElementById('modal-rapid-sampling-mode');
    if (samplingMode && preset.id) samplingMode.value = preset.id;
}

// ── Restart after preset save ──────────────────────────────────────────────────

function _offerRestartAfterPresetSave(presetId) {
    if (!presetId) return;

    showToastWithActions(
        'Apply changes?',
        'info',
        'Restart the local model server to load the updated preset.',
        [
            {
                id: 'restart',
                label: 'Restart Now',
                primary: true,
                handler: async () => {
                    showToast('Restarting local model server…', 'info');
                    try {
                        await _restartServerWithPreset(presetId);
                    } catch (e) {
                        showToast('Restart failed: ' + (e.message || String(e)), 'error');
                    }
                },
            },
            {
                id: 'later',
                label: 'Not now',
                primary: false,
                handler: () => {},
            },
        ],
        { duration: 12000 },
    );
}

async function _restartServerWithPreset(presetId) {
    const p = sessionState.presets.find(pr => pr.id === presetId);
    if (!p) throw new Error('Preset not found');

    // Kill current server
    try {
        const tokenResp = await fetch('/api/db/admin-token', {
            headers: window.authHeaders ? window.authHeaders() : {},
        });
        const tokenData = tokenResp.ok ? await tokenResp.json().catch(() => ({})) : {};
        const token = tokenData.token;
        if (!token) throw new Error('Authentication required');

        await fetch('/api/kill-server', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + token,
            },
            body: JSON.stringify({ confirm: 'kill' }),
        });
    } catch (e) {
        throw new Error('Failed to stop server: ' + (e.message || e));
    }

    // Rust owns backend selection, native config, and the resolved launch port.
    const config = { preset_id: presetId };

    // Spawn new server
    const adminToken = await (async () => {
        const tokenResp = await fetch('/api/db/admin-token', {
            headers: window.authHeaders ? window.authHeaders() : {},
        });
        const tokenData = tokenResp.ok ? await tokenResp.json().catch(() => ({})) : {};
        return tokenData.token || null;
    })();

    if (!adminToken) throw new Error('Authentication required');

    const resp = await fetch('/api/sessions/spawn', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + adminToken,
        },
        body: JSON.stringify(config),
    });

    if (!resp.ok) {
        const text = await resp.text().catch(() => 'Request failed');
        throw new Error('Spawn failed: ' + text);
    }

    const data = await resp.json().catch(() => ({}));
    if (!data.ok) {
        throw new Error(data.error || 'Spawn responded with an error');
    }

    // Wait for server to be ready
    const backendLabel = data.backend === 'rapid_mlx' ? 'Rapid-MLX' : 'llama-server';
    const launchPort = data.port;
    showToast(`Starting ${backendLabel}…`, 'info', 'Loading model on port ' + launchPort, { duration: 12000 });
    try {
        await (await import('./spawn-readiness.js')).waitForSpawnReadiness(launchPort);
    } catch (e) {
        throw new Error('Server did not become ready: ' + (e.message || e));
    }

    showToast(`${backendLabel} restarted`, 'success', '', { duration: 6000 });
}

// ── Model architecture info (preset editor) ───────────────────────────────────

// Format a byte count as GiB/MiB for per-layer VRAM hints.
function _formatLayerBytes(bytes) {
    const gib = bytes / (1024 ** 3);
    if (gib >= 1) return gib.toFixed(gib >= 10 ? 0 : 2) + ' GiB';
    return Math.round(bytes / (1024 ** 2)) + ' MiB';
}

function _renderPresetArchInfo(preset) {
    const container = document.getElementById('pe-arch-info');
    if (!container) return;
    container.innerHTML = '';

    const arch = buildArchitectureLabel(preset, null);
    if (!arch) return;

    // Main line: "Architecture: MoE • 35B (3B active)"
    const main = document.createElement('div');
    main.className = 'pe-arch-main';
    main.textContent = 'Architecture: ' + arch.display;
    main.title = arch.tooltip;

    container.appendChild(main);

    // Expert sub-line (if present)
    if (preset.expert_count != null || preset.expert_used_count != null) {
        const sub = document.createElement('div');
        sub.className = 'pe-arch-sub';
        const parts = [];
        if (preset.expert_count != null) {
            parts.push(preset.expert_count + ' experts');
        }
        if (preset.expert_used_count != null) {
            parts.push(preset.expert_used_count + ' active per token');
        }
        sub.textContent = parts.join(', ');
        container.appendChild(sub);
    }

    // Layer-count sub-line — the value users need to bound the GPU-offload knobs.
    // For MoE/hybrid: --n-cpu-moe offloads expert layers to CPU/RAM.
    // For dense: --gpu-layers (-ngl) is the primary offload knob (no experts).
    if (preset.block_count != null) {
        const layers = document.createElement('div');
        layers.className = 'pe-arch-sub';
        layers.textContent = isMoEEligible(preset)
            ? preset.block_count + ' layers — set --n-cpu-moe between 0 and ' +
                preset.block_count + ' to offload expert layers to CPU/RAM'
            : preset.block_count + ' layers — set --gpu-layers (-ngl) between 0 and ' +
                preset.block_count + ' to offload layers to the GPU';
        container.appendChild(layers);
    }
}

// Clear architecture info + layer hints when model path changes so they don't show
// stale data (block_count is refreshed by the backend only after the preset is saved).
document.getElementById('modal-model-path')?.addEventListener('input', () => {
    const container = document.getElementById('pe-arch-info');
    if (container) container.innerHTML = '';
    ['modal-n-cpu-moe-layers', 'modal-gpu-layers-layers'].forEach(id => {
        const el = document.getElementById(id);
        if (el) { el.textContent = ''; el.style.display = 'none'; }
    });
    ['modal-gpu-layers', 'modal-n-cpu-moe'].forEach(id => {
        document.getElementById(id)?.removeAttribute('max');
    });
    // Fetch live Rapid-MLX profile for this model when backend is rapid_mlx
    document.getElementById('modal-rapid-prefill-step-size')?.addEventListener('change', () => {
        _presetRapidMlxPrefillExplicit = true;
    });
    _schedulePresetRapidMlxProfile();
    if (document.getElementById('modal-rapid-speculative-enabled')?.checked
        && document.getElementById('modal-rapid-speculative-source')?.value === 'external') {
        _fetchSidecarsForPreset();
    }
});

// ── Rapid-MLX live model profile for preset editor ────────────────────────────

let _presetRapidMlxProfileTimer = null;
let _presetRapidMlxProfile = null;
let _presetUnifiedProfile = null;
let _presetNativeContextLimit = 0;

function _schedulePresetRapidMlxProfile() {
    clearTimeout(_presetRapidMlxProfileTimer);
    _presetRapidMlxProfileTimer = setTimeout(async () => {
        const preset = _currentModalPreset();
        if (!preset || preset.backend !== 'rapid_mlx') {
            _presetRapidMlxProfile = null;
            _presetUnifiedProfile = null;
            return;
        }
        const rapidMlx = preset.rapid_mlx;
        const modelId = rapidMlx?.model_source_view?.canonical_identity
            || rapidMlx?.model_source_view?.display_name
            || presetModelSource(preset) || '';
        if (!modelId || modelId.trim().length < 2) {
            _presetRapidMlxProfile = null;
            _presetUnifiedProfile = null;
            return;
        }
        try {
            const headers = window.authHeaders ? window.authHeaders() : {};

            // Fetch both profiles in parallel
            const [profileRes, unifiedRes] = await Promise.allSettled([
                fetch(`/api/rapid-mlx/models/${encodeURIComponent(modelId)}/profile`, { headers }),
                fetch(`/api/rapid-mlx/models/${encodeURIComponent(modelId)}/unified-profile`, { headers })
            ]);

            if (profileRes.status === 'fulfilled' && profileRes.value.ok) {
                const pdata = await profileRes.value.json().catch(() => ({}));
                _presetRapidMlxProfile = pdata.profile || null;
            } else {
                _presetRapidMlxProfile = null;
            }

            if (unifiedRes.status === 'fulfilled' && unifiedRes.value.ok) {
                const udata = await unifiedRes.value.json().catch(() => ({}));
                _presetUnifiedProfile = udata || null;
            } else {
                _presetUnifiedProfile = null;
            }

            if (rapidMlxProfileHasVision(_presetRapidMlxProfile) && !_presetRapidMlxPrefillExplicit) {
                setOpt('modal-rapid-prefill-step-size', String(rapidMlxPrefillStepSizeDefault(_presetRapidMlxProfile)));
                updatePresetVram();
            }

            // Apply unified profile recommendations as hints
            if (_presetUnifiedProfile) {
                _applyPresetUnifiedProfileHints(_presetUnifiedProfile);
            }
        } catch {
            _presetRapidMlxProfile = null;
            _presetUnifiedProfile = null;
        }
    }, 350);
}

function _applyPresetUnifiedProfileHints(up) {
    const rec = up.recommended;
    if (!rec) return;

    // Hybrid mode recommendation hint
    if (rec.hybrid_mode && rec.hybrid_mode !== 'auto') {
        const hybridSel = document.getElementById('modal-rapid-hybrid-mode');
        if (hybridSel) {
            hybridSel.title = `Recommended: ${rec.hybrid_mode} (${up.sources?.hybrid_mode || 'unknown'} source)`;
        }
    }

    // Tool-call parser recommendation hint
    if (rec.tool_format) {
        const toolSel = document.getElementById('modal-rapid-tool-call-parser');
        if (toolSel) {
            toolSel.title = `Recommended: ${rec.tool_format} (${up.sources?.tool_format || 'unknown'} source)`;
        }
    }

    // Reasoning parser recommendation hint
    if (rec.reasoning_parser) {
        const reasonSel = document.getElementById('modal-rapid-reasoning-parser');
        if (reasonSel) {
            reasonSel.title = `Recommended: ${rec.reasoning_parser} (${up.sources?.reasoning_parser || 'unknown'} source)`;
        }
    }
}

export function getPresetRapidMlxProfile() {
    return _presetRapidMlxProfile;
}

export function getPresetUnifiedProfile() {
    return _presetUnifiedProfile;
}

// ── Init ───────────────────────────────────────────────────────────────────────

export function initPresets() {
    // Init preset editor nav
    initPresetEditorNav();
    initCalibrationUi();
    window.addEventListener('presets:reload', () => { loadPresets(); });

    // Bind preset action buttons (toolbar — minimal)
    document.getElementById('preset-edit-btn')?.addEventListener('click', () => openPresetModal('edit'));
    document.getElementById('preset-new-btn')?.addEventListener('click', () => openPresetModal('new'));

    // Refresh the performance advisor as the preset form changes
    const presetForm = document.getElementById('preset-form');
    if (presetForm) {
        presetForm.addEventListener('input', () => { updatePresetAdvisor(); updatePresetVram(); });
        presetForm.addEventListener('change', () => { updatePresetAdvisor(); updatePresetVram(); });
    }

    // MoE offload auto-tuner (empirical sweep)
    document.getElementById('preset-moe-autotune-verify')?.addEventListener('click', autoTunePreset);

    // Bind preset modal buttons
    document.getElementById('preset-modal-close')?.addEventListener('click', closePresetModal);
    document.getElementById('preset-modal-cancel')?.addEventListener('click', closePresetModal);
    document.getElementById('preset-modal-back')?.addEventListener('click', _hideSummary);
    document.getElementById('preset-vram-auto-size')?.addEventListener('click', autoSizePreset);
    document.getElementById('preset-convert-bundle')?.addEventListener('click', convertCurrentPresetToBundle);
    document.getElementById('preset-bundle-add')?.addEventListener('click', addBundleArtifact);
    document.getElementById('preset-bundle-browse')?.addEventListener('click', () => openModelFileBrowser('modal-bundle-artifact-path', 'gguf', null, 'model'));
    document.querySelector('.preset-editor-section[data-section="variants"]')?.addEventListener('change', event => {
        if (event.target.matches('#modal-bundle-artifact, #modal-bundle-context, #modal-bundle-kv, #modal-bundle-performance, #modal-bundle-cpu-moe')) {
            updateBundleSelectionFromEditor();
            renderPresetBundleEditor();
        }
    });
    document.querySelector('.preset-editor-section[data-section="variants"]')?.addEventListener('click', event => {
        const remove = event.target.closest('.preset-bundle-remove');
        if (remove) removeBundleArtifact(remove.dataset.artifactId);
    });

    // Chat Template Manage modal — shared with the Spawn Wizard(s) via chat-template-panel.js.
    bindChatTemplateManageModalChrome();
    document.getElementById('lifecycle-library-btn')?.addEventListener('click', async () => {
        try {
            await openChatTemplateLibraryBrowser('modal-chat-template-file');
            await updatePresetChatTemplateStatusLine();
        } catch (err) {
            showToast('Template library unavailable: ' + (err.message || String(err)), 'error');
        }
    });
    document.getElementById('lifecycle-upload-btn')?.addEventListener('click', async () => {
        try {
            const uploaded = await uploadChatTemplateFromBrowser();
            if (!uploaded?.path) return;
            setVal('modal-chat-template-file', uploaded.path);
            await updatePresetChatTemplateStatusLine();
            showToast('Template uploaded', 'success', uploaded.filename || 'Saved to template library');
        } catch {
            // uploadChatTemplateFromBrowser already surfaced the error
        }
    });

    // Duplicate preset from within the modal
    document.getElementById('preset-modal-duplicate')?.addEventListener('click', async () => {
        const id = document.getElementById('modal-preset-id').value;
        await duplicatePresetById(id, { reopenEditor: true });
    });

    // Delete preset from within the modal (only visible in edit mode)
    document.getElementById('preset-modal-delete')?.addEventListener('click', async () => {
        const id = document.getElementById('modal-preset-id').value;
        const p = sessionState.presets.find(pr => pr.id === id);
        if (!p) { showToast('No preset selected', 'warn'); return; }
        let catalogEtag;
        try {
            catalogEtag = await freshPresetCatalogEtag();
        } catch (error) {
            showToast(`Delete cancelled: ${error.message || error}`, 'error');
            return;
        }
        const ok = await showConfirmDialog(
            'Delete preset',
            `Delete preset "${p.name}"? This cannot be undone.`,
            'Delete'
        );
        if (!ok) return;
        try {
            const resp = await fetch('/api/presets/' + encodeURIComponent(id), {
                method: 'DELETE',
                headers: { ...(window.authHeaders ? window.authHeaders() : {}), 'Content-Type': 'application/json' },
                body: JSON.stringify({ expected_revision: p.revision ?? 1, expected_catalog_etag: catalogEtag, confirmation: 'DELETE PRESET' }),
            });
            if (!resp.ok) {
                const err = await resp.text().catch(() => 'Unknown error');
                showToast('Delete failed: ' + err, 'error');
                return;
            }
            closePresetModal();
            await loadPresets();
            showToast('Preset deleted', 'success');
        } catch (err) {
            showToast('Delete failed: ' + err.message, 'error');
        }
    });
    document.getElementById('preset-browse-model-btn')?.addEventListener('click', () => openModelFileBrowser('modal-model-path', 'gguf', null, 'model'));
    document.getElementById('preset-browse-mmproj-btn')?.addEventListener('click', () => openModelFileBrowser('modal-mmproj', 'gguf', null, 'mmproj'));
    document.getElementById('modal-mmproj')?.addEventListener('input', (e) => {
        _toggleVisionTokens(!!(e.target.value || '').trim());
    });
    document.getElementById('modal-image-max-tokens')?.addEventListener('input', (e) => {
        _ensureUbatchForImageTokens(Number(e.target.value || 0));
    });
    document.getElementById('modal-cache-mode')?.addEventListener('change', (e) => {
        _toggleCacheRamField(e.target.value);
    });
    document.getElementById('modal-rapid-cache-mode')?.addEventListener('change', (e) => {
        _toggleRapidCacheFields(e.target.value);
    });
    bindRecommendedChatTemplateButton();
    document.getElementById('preset-clear-chat-template-btn')?.addEventListener('click', async () => {
        setVal('modal-chat-template-file', '');
        await updatePresetChatTemplateStatusLine();
    });
    document.getElementById('preset-chat-template-manage-btn')?.addEventListener('click', async () => {
        const path = strVal('modal-chat-template-file');
        if (!path) {
            showToast('No template selected yet', 'warn');
            return;
        }
        const tplName = _presetChatTemplateName(path);
        await openChatTemplateManageModal({
            tplName,
            currentPath: path,
            onActivated: updatePresetChatTemplateStatusLine,
            origin: 'preset-editor',
        });
    });
    document.getElementById('modal-chat-template-file')?.addEventListener('change', updatePresetChatTemplateStatusLine);
    document.getElementById('preset-browse-draft-model-btn')?.addEventListener('click', () => openModelFileBrowser('modal-draft-model', 'gguf', null, 'draft-model'));

    // Fit-to-VRAM toggle shows/hides fit target
    document.getElementById('modal-fit-enabled')?.addEventListener('change', function() {
        _toggleFitTarget(this.value === 'true');
    });

    // Spec type dropdown shows/hides relevant fields
    document.getElementById('modal-spec-type')?.addEventListener('change', function() {
        _toggleSpecFields(this.value);
    });

    // Show cache-idle-slots hint when parallel slots > 1
    document.getElementById('modal-parallel-slots')?.addEventListener('input', function() {
        const hint = document.getElementById('cache-idle-slots-hint');
        if (hint) hint.style.display = parseInt(this.value) > 1 ? '' : 'none';
    });

    document.getElementById('modal-structured-output-mode')?.addEventListener('change', function() {
        setStructuredOutputMode(this.value);
    });

    // Bind preset form submit
    if (presetForm) presetForm.addEventListener('submit', savePreset);

    // Bind setup view link
    document.getElementById('setup-manage-presets-link')?.addEventListener('click', (e) => {
        e.preventDefault();
        openPresetModal('new');
    });

    // Modal overlay click
    const modal = document.getElementById('preset-modal');
    if (modal) {
        modal.addEventListener('click', e => {
            if (e.target === e.currentTarget) closePresetModal();
        });
    }

    window.closePresetsPanel = closePresetsPanel;

    // Clear field errors on input
    ['modal-name', 'modal-model-path'].forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('input', function() {
                this.classList.remove('field-error');
            });
        }
    });

    // When model path changes, suggest model-family generation defaults (only fills empty fields)
    let _modelDefaultsTimer = null;
    document.getElementById('modal-model-path')?.addEventListener('input', function() {
        clearTimeout(_modelDefaultsTimer);
        const path = this.value.trim();
        if (!path) return;
        _modelDefaultsTimer = setTimeout(() => _suggestGenerationDefaults(path), 600);
    });
    // Initial load
    loadPresets();
}

// ── Apple Silicon-aware Threads hints in preset editor ─────────────────────────

function _refreshPresetThreadsHints() {
  const modal = document.getElementById('preset-modal');
  if (!modal || !modal.classList.contains('open')) return;

  const metrics = lastSystemMetrics;
  const pCores = metrics?.p_cores || 0;
  const metricsReady = metrics != null;

  const threadsInput = document.getElementById('modal-threads');
  const batchThreadsInput = document.getElementById('modal-threads-batch');
  if (!threadsInput && !batchThreadsInput) return;

  const hintEl = document.getElementById('preset-threads-hint');
  if (pCores > 0 && metricsReady) {
    if (threadsInput && !threadsInput.value) {
      threadsInput.placeholder = '1 recommended';
    }
    if (batchThreadsInput && !batchThreadsInput.value) {
      batchThreadsInput.placeholder = `${pCores} recommended`;
    }
    if (hintEl && _presetIsUnified) {
      hintEl.textContent = `Apple Silicon: threads = 1 (Metal GPU handles inference), threads-batch = ${pCores} (P-cores for faster prefill).`;
      hintEl.style.display = '';
    }
  } else {
    if (threadsInput && !threadsInput.value) {
      threadsInput.placeholder = 'auto';
    }
    if (batchThreadsInput && !batchThreadsInput.value) {
      batchThreadsInput.placeholder = 'auto';
    }
    if (hintEl) hintEl.style.display = 'none';
  }
}
window.__refreshPresetEditorHints = _refreshPresetThreadsHints;

async function _fetchSystemInfoAndRefreshPresetHints() {
  try {
    const headers = window.authHeaders ? window.authHeaders() : {};
    const [res] = await Promise.all([
      fetch('/api/system/info', { headers }),
      _ensureUnifiedFlag(), // populate _presetIsUnified before hint renders
    ]);
    if (!res.ok) return;
    const data = await res.json();
    if (data.ok && data.p_cores > 0) {
      const { setLastSystemMetrics } = await import('../core/app-state.js');
      setLastSystemMetrics({ p_cores: data.p_cores, e_cores: data.e_cores, cpu_name: data.cpu_name });
      _refreshPresetThreadsHints();
    }
  } catch (err) { console.warn('Failed to fetch system info for preset hints:', err); }
}

export async function _showConfirm(title, message) {
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay';
    overlay.style.zIndex = '2000';
    overlay.style.display = 'grid';

    const dialog = document.createElement('div');
    dialog.className = 'modal';
    dialog.style.width = '420px';
    dialog.style.padding = '14px 16px';

    const titleEl = document.createElement('div');
    titleEl.style.fontSize = '15px';
    titleEl.style.fontWeight = '600';
    titleEl.style.marginBottom = '8px';
    titleEl.textContent = title;

    const msg = document.createElement('div');
    msg.style.fontSize = '13px';
    msg.style.color = 'var(--color-text-muted)';
    msg.style.marginBottom = '12px';
    msg.textContent = message;

    const actions = document.createElement('div');
    actions.style.display = 'flex';
    actions.style.justifyContent = 'flex-end';
    actions.style.gap = '8px';

    const cancelBtn = document.createElement('button');
    cancelBtn.type = 'button';
    cancelBtn.className = 'btn btn-modal-cancel';
    cancelBtn.textContent = 'Cancel';

    const confirmBtn = document.createElement('button');
    confirmBtn.type = 'button';
    confirmBtn.className = 'btn btn-modal-save';
    confirmBtn.textContent = 'Confirm';

    return new Promise(resolve => {
        let decided = false;

        function cleanup() {
            if (overlay.parentElement) overlay.remove();
        }

        cancelBtn.addEventListener('click', () => {
            if (decided) return;
            decided = true;
            cleanup();
            resolve(false);
        });

        confirmBtn.addEventListener('click', () => {
            if (decided) return;
            decided = true;
            cleanup();
            resolve(true);
        });

        overlay.addEventListener('click', (e) => {
            if (e.target === overlay && !decided) {
                decided = true;
                cleanup();
                resolve(false);
            }
        });

        actions.appendChild(cancelBtn);
        actions.appendChild(confirmBtn);
        dialog.appendChild(titleEl);
        dialog.appendChild(msg);
        dialog.appendChild(actions);
        overlay.appendChild(dialog);
        document.body.appendChild(overlay);
        cancelBtn.focus();
    });
}

function _renderContextPills(mode, section) {
    const pillsContainer = document.getElementById('preset-context-pills');
    if (!pillsContainer) return;
    const pills = [
        { label: '32k', value: 32768 },
        { label: '65k', value: 65536 },
        { label: '131k', value: 131072 },
        { label: '160k', value: 163840 },
        { label: '200k', value: 200000 },
        { label: '262k', value: 262144 },
    ];
    pillsContainer.innerHTML = '';
    pills.forEach(pill => {
        const pillEl = document.createElement('button');
        pillEl.type = 'button';
        pillEl.className = 'preset-context-pill';
        pillEl.textContent = pill.label;
        const advancedOnly = _presetNativeContextLimit > 0 && pill.value > _presetNativeContextLimit;
        pillEl.disabled = advancedOnly;
        pillEl.classList.toggle('preset-context-pill--advanced', advancedOnly);
        if (advancedOnly) {
            pillEl.title = `${pill.label} exceeds the native ${Math.round(_presetNativeContextLimit / 1024)}k model limit. Advanced Context extension is not configured.`;
        }
        pillEl.onclick = (e) => {
            e.preventDefault();
            if (advancedOnly) return;
            const input = document.getElementById('modal-context-size');
            if (input) {
                input.value = pill.value;
                input.dispatchEvent(new Event('input', { bubbles: true }));
                input.dispatchEvent(new Event('change', { bubbles: true }));
            }
        };
        pillsContainer.appendChild(pillEl);
    });
}
