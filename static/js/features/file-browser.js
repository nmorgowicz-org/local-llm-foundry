// ── File Browser ───────────────────────────────────────────────────────────────
// File/directory browser modal. Used by preset modal, session modal, settings.

import { escapeHtml } from '../core/format.js';

let fbTargetId = '';
let fbFilter = '';
let fbCurrentPath = '';
let fbContext = { kind: '', engine: '' };
let fbModelLocations = {};
let fbModelInventory = null;
let fbModelInventoryPromise = null;
let fbEntries = [];
let fbCompanionNames = { mmproj: [], draft: [] };
let fbInitialPath = '';
let fbInitialFallbackTried = false;
let initialized = false;

function normalizeContext(context) {
    if (typeof context === 'string') {
        return { kind: context, engine: context === 'rapid_mlx' ? 'rapid_mlx' : '' };
    }
    return context && typeof context === 'object' ? { ...context } : { kind: '', engine: '' };
}

function isModelBrowser() {
    return fbContext.kind === 'model';
}

function normalizePath(path) {
    return String(path || '').replace(/\\/g, '/').replace(/\/+$/, '').toLowerCase();
}

function modelForPath(path) {
    const wanted = normalizePath(path);
    return fbModelInventory?.find(model => normalizePath(model.path) === wanted) || null;
}

function hfCacheInfo(path) {
    const normalized = String(path || '').replace(/\\/g, '/');
    const match = normalized.match(/\/models--([^/]+)--([^/]+)(?:\/snapshots\/([^/]+))?/);
    if (!match) return null;
    return {
        repoId: `${match[1]}/${match[2]}`,
        revision: match[3] || '',
    };
}

async function loadModelBrowserData() {
    if (!isModelBrowser()) return;
    if (!fbModelInventoryPromise) {
        const headers = window.authHeaders ? window.authHeaders() : {};
        fbModelInventoryPromise = Promise.all([
            fetch('/api/models', { headers }).then(response => response.ok ? response.json() : []),
            fetch('/api/hf/download-dir', { headers }).then(response => response.ok ? response.json() : {}),
        ]).then(([inventory, downloadInfo]) => {
            fbModelInventory = Array.isArray(inventory) ? inventory : [];
            fbModelLocations = downloadInfo.locations || {};
        }).catch(() => {
            fbModelInventory = [];
            fbModelLocations = {};
        });
    }
    await fbModelInventoryPromise;
}

// Rapid-MLX downloads land in the app HF cache, not mlx/native. When the picker opens
// on an empty starting folder, show the app cache instead so a model the user just
// downloaded is visible without hunting through the location tabs.
async function maybeFallbackToManagedHfCache(path) {
    const target = fbModelLocations.managed_hf_cache;
    if (!isModelBrowser()
        || fbContext.engine !== 'rapid_mlx'
        || fbInitialFallbackTried
        || !target
        || normalizePath(path) !== normalizePath(fbInitialPath)
        || normalizePath(path) === normalizePath(target)) {
        return false;
    }
    fbInitialFallbackTried = true;
    try {
        const response = await fetch('/api/browse?' + new URLSearchParams({ path: target }), {
            headers: window.authHeaders ? window.authHeaders() : {},
        });
        if (!response.ok) return false;
        const data = await response.json();
        if (!Array.isArray(data.entries) || !data.entries.some(entry => entry.is_dir)) return false;
        await fileBrowserGo(target);
        return true;
    } catch (_) {
        return false;
    }
}

async function maybeFallbackToLegacyRoot(path) {
    if (!isModelBrowser()
        || fbContext.engine !== 'llama_cpp'
        || fbInitialFallbackTried
        || !fbModelLocations.library_root
        || normalizePath(path) !== normalizePath(fbInitialPath)
        || normalizePath(path) === normalizePath(fbModelLocations.library_root)) {
        return false;
    }

    fbInitialFallbackTried = true;
    try {
        const params = new URLSearchParams({
            path: fbModelLocations.library_root,
            filter: 'gguf',
        });
        const response = await fetch('/api/browse?' + params, {
            headers: window.authHeaders ? window.authHeaders() : {},
        });
        if (!response.ok) return false;
        const data = await response.json();
        const hasLegacyGguf = Array.isArray(data.entries)
            && data.entries.some(entry => !entry.is_dir && entry.name.toLowerCase().endsWith('.gguf'));
        if (!hasLegacyGguf) return false;
        await fileBrowserGo(fbModelLocations.library_root);
        return true;
    } catch (_) {
        return false;
    }
}

function setModelBrowserChrome() {
    const modal = document.getElementById('file-browser-modal');
    const toolbar = document.getElementById('fb-model-toolbar');
    const modelMode = isModelBrowser();
    modal?.classList.toggle('file-browser-modal--model', modelMode);
    if (toolbar) toolbar.hidden = !modelMode;
    const intent = document.getElementById('fb-model-intent');
    if (intent && modelMode) {
        const rapid = fbContext.engine === 'rapid_mlx';
        intent.textContent = rapid
            ? 'Rapid-MLX model picker · choose a native MLX directory or HF snapshot'
            : 'llama.cpp model picker · choose a GGUF model file';
    }
    renderModelLocations();
    refreshModelBrowserFilters();
}

function refreshModelBrowserFilters() {
    const familySelect = document.getElementById('fb-model-family');
    if (!familySelect || !isModelBrowser()) return;
    const families = [...new Set((fbModelInventory || [])
        .map(model => model.classification?.family)
        .filter(Boolean))].sort((a, b) => a.localeCompare(b));
    const current = familySelect.value;
    familySelect.replaceChildren(new Option('All families', 'all'));
    families.forEach(family => familySelect.appendChild(new Option(family, family)));
    familySelect.value = families.includes(current) ? current : 'all';
}

function renderModelLocations() {
    const host = document.getElementById('fb-model-locations');
    if (!host || !isModelBrowser()) return;
    const rapid = fbContext.engine === 'rapid_mlx';
    const locations = rapid
        ? [
            ['mlx_native', 'MLX native'],
            ['external_hf_cache', 'External HF cache'],
            ['managed_hf_cache', 'App HF cache'],
            ['library_root', 'Library root'],
        ]
        : [
            ['gguf', 'GGUF library'],
            ['external_hf_cache', 'External HF cache'],
            ['managed_hf_cache', 'App HF cache'],
            ['library_root', 'Library root'],
        ];
    host.replaceChildren(...locations.filter(([key]) => fbModelLocations[key]).map(([key, label]) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'fb-location-btn';
        button.textContent = label;
        button.title = fbModelLocations[key];
        button.addEventListener('click', () => fileBrowserGo(fbModelLocations[key]));
        return button;
    }));
}

// Strip quant suffix to get model stem for mmproj matching.
function _fbModelStem(filename) {
    return filename
        .replace(/\.gguf$/i, '')
        .replace(/-?(Q\d[^.]*|IQ\d[^.]*|F16|BF16)$/i, '');
}

// Returns true if mmprojName looks like a companion for modelName.
function _mmprojMatchesModel(mmprojName, modelName) {
    const stem = _fbModelStem(modelName).toLowerCase();
    const mp   = mmprojName.toLowerCase();
    // Stem-in-mmproj: "Qwen2.5-VL-7B-Q4…" → stem "Qwen2.5-VL-7B" ∈ "qwen2.5-vl-7b-mmproj-f16"
    if (stem.length > 4 && mp.includes(stem)) return true;
    return false;
}

// Returns true if draftName looks like an MTP draft companion for modelName.
function _draftMatchesModel(draftName, modelName) {
    const stem = _fbModelStem(modelName).toLowerCase();
    // Strip quant suffix then strip MTP keyword + tail to get draft's base prefix
    const dr = draftName.toLowerCase()
        .replace(/\.gguf$/i, '')
        .replace(/[-_](?:mtp[-_]draft|draft[-_]model|mtp_small|mtp[-_]heads|mtp[-_]\d+|assistant)\b.*$/i, '');
    if (stem.length <= 4) return false;
    // QAT parity: a QAT-trained draft only works with QAT bases and vice versa
    const drQat = /-qat[-_]/.test(dr) || dr.endsWith('-qat');
    const stemQat = /-qat[-_]/.test(stem) || stem.endsWith('-qat');
    if (drQat !== stemQat) return false;
    // Direct containment covers the generic assistant case
    if (stem.includes(dr) || dr.includes(stem)) return true;
    // QAT/variant drafts diverge from the base name after a shared prefix
    // (e.g. "...-qat-q4_0-unquantized" vs "...-qat-UD"). Match when >= 5
    // leading dash-segments are identical so "gemma-4-31b-it-qat" qualifies
    // but a plain "gemma-4-31b-it" draft doesn't over-match a different family.
    const drSegs = dr.split('-');
    const stemSegs = stem.split('-');
    let shared = 0;
    for (let i = 0; i < Math.min(drSegs.length, stemSegs.length); i++) {
        if (drSegs[i] === stemSegs[i]) shared++;
        else break;
    }
    return shared >= 5;
}

export function openFileBrowser(targetId, filter, defaultPath, context) {
    fbTargetId = targetId;
    fbFilter = filter === 'dir' ? '' : (filter || '');
    fbContext = normalizeContext(context);
    fbEntries = [];
    fbCompanionNames = { mmproj: [], draft: [] };
    fbInitialPath = '';
    fbInitialFallbackTried = false;
    const modal = document.getElementById('file-browser-modal');
    const title = document.getElementById('fb-title');
    const selectBtn = modal.querySelector('.btn-modal-save');

    // Set title based on filter
    if (isModelBrowser() && fbContext.engine === 'rapid_mlx') {
        title.textContent = 'Choose a Rapid-MLX Model';
    } else if (filter === 'gguf') {
        title.textContent = 'Browse Model Files';
    } else if (filter === 'executable') {
        title.textContent = 'Browse Executable';
    } else if (filter === 'dir') {
        title.textContent = 'Browse Directory';
    } else {
        title.textContent = 'Browse Files';
    }

    // Show/hide Select button based on mode
    if (filter === 'dir') {
        selectBtn.style.display = '';
        selectBtn.textContent = 'Select This Folder';
    } else {
        selectBtn.style.display = '';
        selectBtn.textContent = 'Select';
    }

    // Determine starting path: current input value → defaultPath → ''
    const current = document.getElementById(targetId).value;
    let startPath = defaultPath || '';
    if (current) {
        // Try to extract parent directory from current path
        const sep = current.includes('\\') ? '\\' : '/';
        const parts = current.split(sep);
        parts.pop();
        startPath = parts.join(sep) || (current.includes('\\') ? 'C:\\' : '/');
    }

    modal.classList.add('open');
    setModelBrowserChrome();
    if (isModelBrowser()) {
        fbInitialPath = startPath;
        void loadModelBrowserData().then(() => {
            renderModelLocations();
            refreshModelBrowserFilters();
            return fileBrowserGo(startPath);
        });
    } else {
        fileBrowserGo(startPath);
    }
}

export function closeFileBrowser() {
    document.getElementById('file-browser-modal').classList.remove('open');
}

export async function fileBrowserGo(path) {
    const entriesEl = document.getElementById('fb-entries');
    entriesEl.innerHTML = '<div class="fb-empty">Loading...</div>';

    if (isModelBrowser()) await loadModelBrowserData();

    const params = new URLSearchParams();
    if (path) params.set('path', path);
    if (fbFilter) params.set('filter', fbFilter);

    try {
        const resp = await fetch('/api/browse?' + params, {
            headers: window.authHeaders ? window.authHeaders() : {},
        });
        const data = await resp.json();
        if (data.error) {
            if (isModelBrowser()
                && data.error === 'Path not found'
                && !fbInitialFallbackTried
                && normalizePath(path) === normalizePath(fbInitialPath)
                && fbModelLocations.library_root) {
                fbInitialFallbackTried = true;
                return fileBrowserGo(fbModelLocations.library_root);
            }
            // Handle "Path not allowed" with a clear message
            if (data.error === 'Path not allowed') {
                entriesEl.innerHTML =
                    '<div class="fb-empty">' +
                    'This path is not allowed. Only certain directories are accessible for security.' +
                    '</div>';
            } else {
                entriesEl.innerHTML = '<div class="fb-empty">' + escapeHtml(data.error) + '</div>';
            }
            return;
        }

        fbCurrentPath = data.path;
        document.getElementById('fb-path-input').value = data.path;
        if (data.entries.length === 0) {
            if (await maybeFallbackToLegacyRoot(path)) return;
            if (await maybeFallbackToManagedHfCache(path)) return;
            entriesEl.innerHTML = '<div class="fb-empty">Empty directory</div>';
            return;
        }

        // In model-selection context: hide mmproj and draft assistant files;
        // badge base models that have a companion.
        fbEntries = data.entries;
        let entries = fbEntries;
        let mmprojNames = [];
        let draftNames = [];
        if (isModelBrowser()) {
            // Mirror src/models/mod.rs is_draft_assistant_filename: unambiguous keywords
            // are safe at any size; broad keywords require size <= 3 GB to avoid
            // mis-tagging instruct-tuned main models.
            const isAssist = (e) => {
                const n = e.name.toLowerCase();
                const sz = e.size || 0;
                const isUnambiguous = n.includes('mtp-draft')
                    || n.includes('mtp_small')
                    || n.includes('mtp-heads')
                    || (n.startsWith('mtp-') && sz <= 3_000_000_000);
                if (isUnambiguous) return true;
                const isBroad = n.includes('assistant') || n.includes('draft-model');
                return isBroad && sz > 0 && sz <= 3_000_000_000;
            };
            mmprojNames = entries
                .filter(e => !e.is_dir && e.name.toLowerCase().includes('mmproj'))
                .map(e => e.name);
            draftNames = entries
                .filter(e => !e.is_dir && isAssist(e))
                .map(e => e.name);
            fbCompanionNames = { mmproj: mmprojNames, draft: draftNames };
            entries = entries.filter(e => {
                if (e.is_dir) return true;
                const n = e.name.toLowerCase();
                if (n.includes('mmproj')) return false;
                if (isAssist(e)) return false;
                return true;
            });
        }

        renderEntries(entries);
    } catch (err) {
        entriesEl.innerHTML = '<div class="fb-empty">Error: ' + escapeHtml(err.message) + '</div>';
    }
}

function renderEntries(sourceEntries) {
    const entriesEl = document.getElementById('fb-entries');
    if (!entriesEl) return;
    let entries = [...sourceEntries];
    if (isModelBrowser()) {
        if (fbContext.engine === 'rapid_mlx') {
            entries = entries.filter(entry => entry.is_dir);
        }
        const search = document.getElementById('fb-model-search')?.value.trim().toLowerCase() || '';
        const family = document.getElementById('fb-model-family')?.value || 'all';
        const format = document.getElementById('fb-model-format')?.value || 'all';
        const sort = document.getElementById('fb-model-sort')?.value || 'name';
        entries = entries.filter(entry => {
            const model = modelForPath(entry.path);
            const cache = hfCacheInfo(entry.path);
            const haystack = [entry.name, model?.model_name, model?.filename, model?.quant_type, cache?.repoId]
                .filter(Boolean).join(' ').toLowerCase();
            if (search && !haystack.includes(search)) return false;
            if (family !== 'all' && (model?.classification?.family || 'unknown') !== family) return false;
            if (format !== 'all' && (model?.format || 'unknown') !== format) return false;
            return true;
        });
        entries.sort((a, b) => {
            const aModel = modelForPath(a.path);
            const bModel = modelForPath(b.path);
            if (a.is_dir !== b.is_dir) return a.is_dir ? -1 : 1;
            if (sort === 'size') return (bModel?.size_bytes || b.size || 0) - (aModel?.size_bytes || a.size || 0);
            if (sort === 'status') return (aModel?.lifecycle || 'unknown').localeCompare(bModel?.lifecycle || 'unknown');
            return a.name.toLowerCase().localeCompare(b.name.toLowerCase());
        });
        const summary = document.getElementById('fb-model-summary');
        if (summary) summary.textContent = `${entries.length} item${entries.length === 1 ? '' : 's'} in this location`;
    }
    if (entries.length === 0) {
        entriesEl.innerHTML = '<div class="fb-empty">No matching models in this location</div>';
        return;
    }
    // eslint-disable-next-line no-unsanitized/property -- server strings are escaped before interpolation
    entriesEl.innerHTML = entries.map(e => {
            const name = escapeHtml(e.name);
            const size = escapeHtml(e.size_display || '');
            const model = isModelBrowser() ? modelForPath(e.path) : null;
            const cache = isModelBrowser() ? hfCacheInfo(e.path) : null;
            const badges = modelBrowserBadges(model, cache);
            const selectButton = modelBrowserSelectButton(e, model);
            if (e.is_dir) {
                return `<div class="fb-entry fb-entry-dir" role="button" tabindex="0" data-path="${escapeHtml(e.path)}" title="${name}">` +
                    '<span class="fb-entry-icon">\u{1F4C1}</span>' +
                    '<span class="fb-entry-name">' + name + '</span>' + badges + selectButton + '</div>';
            } else {
                let badge = '';
                if (isModelBrowser()) {
                    if (fbCompanionNames.mmproj.length > 0) {
                        const matched = fbCompanionNames.mmproj.some(mp => _mmprojMatchesModel(mp, e.name));
                        if (matched) {
                            badge += '<span class="fb-mmproj-badge fb-mmproj-matched" title="A companion mmproj file was found in this folder">mmproj ✓</span>';
                        } else {
                            // Dir has unmatched mmproj files — softer hint
                            badge += '<span class="fb-mmproj-badge" title="mmproj file(s) present in this folder">mmproj</span>';
                        }
                    }
                    if (fbCompanionNames.draft.length > 0) {
                        const matched = fbCompanionNames.draft.some(d => _draftMatchesModel(d, e.name));
                        if (matched) {
                            badge += '<span class="fb-mtp-badge fb-mtp-matched" title="A companion MTP draft file was found in this folder">mtp ✓</span>';
                        } else {
                            badge += '<span class="fb-mtp-badge" title="MTP draft file(s) present in this folder">mtp</span>';
                        }
                    }
                }
                return `<div class="fb-entry fb-entry-file fb-match" role="button" tabindex="0" data-path="${escapeHtml(e.path)}" title="${name}">` +
                    '<span class="fb-entry-icon">\u{1F4C4}</span>' +
                    '<span class="fb-entry-name">' + name + '</span>' +
                    badges +
                    badge +
                    '<span class="fb-entry-size">' + (escapeHtml(model?.size_display || size)) + '</span>' + selectButton + '</div>';
            }
        }).join('');
}

function modelBrowserBadges(model, cache) {
    if (!isModelBrowser()) return '';
    const labels = [];
    if (model) {
        const format = model.format === 'mlx' ? 'MLX' : model.format === 'gguf' ? 'GGUF' : model.format;
        if (format) labels.push([format, 'format']);
        const sourceLabels = {
            local: 'Local',
            hugging_face: 'HF library',
            legacy: 'Legacy root',
            official_conversion: 'Converted',
            recovered_gguf: 'Recovered',
            requantized_mlx: 'Re-quantized',
        };
        if (sourceLabels[model.source]) labels.push([sourceLabels[model.source], 'source']);
        if (model.lifecycle && model.lifecycle !== 'ready') labels.push([model.lifecycle, model.lifecycle]);
        if (model.classification?.family) labels.push([model.classification.family, 'family']);
        if (model.supported_backends?.length) labels.push([model.supported_backends.join(' · '), 'backend']);
    } else if (cache) {
        labels.push([cache.repoId, 'hf']);
        if (cache.revision) labels.push([cache.revision.slice(0, 10), 'revision']);
    }
    return labels.map(([label, kind]) => `<span class="fb-model-badge fb-model-badge--${escapeHtml(kind)}">${escapeHtml(label)}</span>`).join('');
}

function modelBrowserSelectButton(entry, model) {
    if (!isModelBrowser()) return '';
    const rapid = fbContext.engine === 'rapid_mlx';
    const valid = model && model.lifecycle === 'ready' && model.supported_backends?.includes(rapid ? 'rapid_mlx' : 'llama_cpp');
    const ggufFile = !entry.is_dir && entry.name.toLowerCase().endsWith('.gguf');
    if (!valid && !(ggufFile && !rapid)) return '';
    const label = entry.is_dir ? 'Select model' : 'Select';
    return `<button type="button" class="fb-entry-select" data-select-path="${escapeHtml(entry.path)}">${label}</button>`;
}

export function fileBrowserUp() {
    if (fbCurrentPath && fbCurrentPath !== '/') {
        const parts = fbCurrentPath.split('/');
        parts.pop();
        fileBrowserGo(parts.join('/') || '/');
    }
}

export function fileBrowserSelect(path) {
    const value = path || fbCurrentPath;
    const el = document.getElementById(fbTargetId);
    if (!el) { closeFileBrowser(); return; }

    if (el.tagName === 'SELECT') {
        // For <select> targets: add or update an option for this value
        let opt = el.querySelector(`option[value="${value
            .replace(/\\/g, '\\\\')
            .replace(/"/g, '\\"')}"]`);
        if (!opt) {
            const file = value.split(/[\\/]/).pop() || value;
            opt = document.createElement('option');
            opt.value = value;
            opt.textContent = file;
            el.appendChild(opt);
        }
        el.value = value;
        el.dispatchEvent(new Event('change', { bubbles: true }));
    } else {
        el.value = value;
        el.dispatchEvent(new Event('input', { bubbles: true }));
    }
    closeFileBrowser();
}

// ── Init ───────────────────────────────────────────────────────────────────────

export function initFileBrowser() {
    if (initialized) return;
    initialized = true;

    // Bind file browser buttons
    const fbClose = document.getElementById('filebrowser-close');
    if (fbClose) fbClose.addEventListener('click', closeFileBrowser);

    const fbCancel = document.getElementById('filebrowser-cancel');
    if (fbCancel) fbCancel.addEventListener('click', closeFileBrowser);

    const fbSelect = document.getElementById('filebrowser-select');
    if (fbSelect) fbSelect.addEventListener('click', () => fileBrowserSelect());

    const fbUp = document.getElementById('filebrowser-up');
    if (fbUp) fbUp.addEventListener('click', fileBrowserUp);

    // Bind path input Enter key
    const fbPathInput = document.getElementById('fb-path-input');
    if (fbPathInput) {
        fbPathInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') fileBrowserGo(fbPathInput.value);
        });
    }

    // Event delegation for dynamically generated entries
    const entriesEl = document.getElementById('fb-entries');
    if (entriesEl) {
        entriesEl.addEventListener('click', (e) => {
            const selectButton = e.target.closest('[data-select-path]');
            if (selectButton) {
                e.stopPropagation();
                fileBrowserSelect(selectButton.dataset.selectPath);
                return;
            }
            const entry = e.target.closest('.fb-entry');
            if (!entry) return;
            const path = entry.dataset.path;
            if (entry.classList.contains('fb-entry-dir')) {
                fileBrowserGo(path);
            } else {
                fileBrowserSelect(path);
            }
        });
        entriesEl.addEventListener('keydown', e => {
            if (e.key !== 'Enter' && e.key !== ' ') return;
            const entry = e.target.closest('.fb-entry');
            if (!entry) return;
            e.preventDefault();
            const select = entry.querySelector('[data-select-path]');
            if (select && e.key === 'Enter') {
                fileBrowserSelect(select.dataset.selectPath);
            } else if (entry.classList.contains('fb-entry-dir')) {
                fileBrowserGo(entry.dataset.path);
            } else {
                fileBrowserSelect(entry.dataset.path);
            }
        });
    }

    ['fb-model-search', 'fb-model-sort', 'fb-model-format', 'fb-model-family'].forEach(id => {
        document.getElementById(id)?.addEventListener('input', () => renderEntries(fbEntries));
        document.getElementById(id)?.addEventListener('change', () => renderEntries(fbEntries));
    });

    // Modal overlay click
    const modal = document.getElementById('file-browser-modal');
    if (modal) {
        modal.addEventListener('click', e => {
            if (e.target === e.currentTarget) closeFileBrowser();
        });
    }

    // Escape key
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape' && modal && modal.classList.contains('open')) {
            closeFileBrowser();
            e.stopImmediatePropagation();
        }
    }, true);
}
