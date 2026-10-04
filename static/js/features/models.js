// ── Models ────────────────────────────────────────────────────────────────────
// Models modal: open, close, load, refresh, delete, and HF download tab.

import { sessionState } from '../core/app-state.js';
import { escapeHtml } from '../core/format.js';
import { getPlatformInfo } from '../core/platform-info.js';
import { resolveNotification, showToast, showToastWithActions } from './toast.js';
import Router from './router.js';
import { _showConfirm, openPresetMtpRepairForModel } from './presets.js';
import { openCardPanel, openSpawnWizard } from './spawn-wizard.js';
import { buildEstimateBody } from './vram-estimate.js';
import {
    hfSearch,
    hfListFiles,
    hfStartCompanionDownload,
    hfStartDownload,
    hfPollDownload,
    hfCancelDownload,
    hfShowDownloadPanel,
    hfHideDownloadPanel,
    hfRenderDiscoverPills,
    hfLoadQuickPicks,
    getRecommendedMmproj,
    hfCreateScopeSelector,
    hfCreateSortSelector,
    ensureCommunitySourceCatalog,
    _resetCommunitySourceCatalog,
    resolveAuthorRole,
    HF_SCOPE,
    HF_SORT,
} from './hf-browse.js';

const PREFS_KEY = 'llama-monitor-models-prefs';

const ICON_LIST_VIEW = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="11" height="11"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>';
const ICON_CARDS_VIEW = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="11" height="11"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>';

const KNOWN_TAGS = ['coding', 'roleplay', 'general', 'art', 'fast', 'default'];

const INVENTORY_BADGES = {
    format: {
        gguf: ['GGUF', 'Format: GGUF'],
        mlx: ['MLX', 'Format: MLX'],
        transformers: ['Transformers', 'Format: Transformers / safetensors'],
        unknown: ['Unknown format', 'Format: Unknown'],
    },
    source: {
        local: ['Local', 'Source: Local library'],
        hugging_face: ['Hugging Face', 'Source: Hugging Face'],
        official_conversion: ['Official conversion', 'Source: Official MLX conversion'],
        recovered_gguf: ['Recovered from GGUF', 'Source: Experimental GGUF recovery'],
        requantized_mlx: ['Re-quantized MLX', 'Source: Experimental MLX re-quantization'],
        legacy: ['Legacy location', 'Source: Legacy model-library location'],
        unknown: ['Unknown source', 'Source: Unknown'],
    },
    lifecycle: {
        ready: ['Ready', 'Lifecycle: Ready'],
        incomplete: ['Incomplete', 'Lifecycle: Incomplete'],
        converting: ['Converting', 'Lifecycle: Converting'],
        invalid: ['Invalid', 'Lifecycle: Invalid'],
        unknown: ['Status unknown', 'Lifecycle: Unknown'],
    },
    compatibility: {
        verified: ['Verified', 'Compatibility: Verified'],
        experimental: ['Experimental', 'Compatibility: Experimental and not launchable'],
        provisional: ['Provisional', 'Compatibility: Provisional'],
        unsupported: ['Unsupported', 'Compatibility: Unsupported'],
        unknown: ['Unknown compatibility', 'Compatibility: Unknown'],
    },
};

const BACKEND_LABELS = {
    llama_cpp: 'llama.cpp',
    rapid_mlx: 'Rapid-MLX',
};

// Which directory-shaped models the Library tab offers to delete in place. Both sets must
// match for a Delete button to appear: `hugging_face` is deliberately absent from the sources
// because a cache snapshot is only links into a shared blob store, so deleting the snapshot
// frees nothing — those are reclaimed per repo on the Disk tab instead.
const MANAGED_DIRECTORY_FORMATS = new Set(['mlx', 'transformers']);
const MANAGED_DIRECTORY_SOURCES = new Set([
    'local',
    'official_conversion',
    'recovered_gguf',
    'requantized_mlx',
]);

let modelCardSequence = 0;
let rapidMlxLocalAvailable = false;
let rapidMlxLocalRequirement = 'Rapid-MLX local execution requires macOS on Apple Silicon';

let initialized = false;
let inventoryCache = null;
let modelLibraryRoot = '';
let startupInventoryChecked = false;
let startupInventoryPromise = null;

const INCOMPLETE_DOWNLOAD_NOTIFICATION_ID = 'models-incomplete-downloads';

let communitySourcesState = {
    initialized: false,
    catalog: null,
    roles: [],
    editingIndex: null,
};

// State for the HF download tab
let hfState = {
    selectedRepoId: null,
    selectedFile: null,
    currentDownloadIds: new Set(),
    initialized: false,
    // Wizard-like state
    paramB: 0,
    modelBytes: 0,
    nCtxTrain: 0,
    mmprojFiles: [],
    mmprojPath: '',
    mmprojRepoId: '',
    mmprojBytes: 0,
    // Active filters (mirrors Quick Start behavior)
    activeAuthor: null,          // e.g. "bartowski"
    activeDiscoverQuery: null,   // e.g. "qwen3" from a discover pill
    // Discovery scope + sort (Phase 8B1) — additive toggles: MLX and GGUF can both be active
    discoveryScopeMlx: false,
    discoveryScopeGguf: true, // default: GGUF always active; macOS will also activate MLX below
    // Matches the ui-settings.json default; the persisted choice overrides it once loaded.
    discoverySort: HF_SORT.LAST_UPDATED,
    discoveryQuantsOnly: false, // filter to show only quantized variants
    previewCtx: 65536, // default context for VRAM calculation
};

// Cached hardware
let cachedVram = 0;
// True when the GPU pool is unified memory (Apple Silicon / Metal) — selects the Metal
// overhead model in the backend estimator instead of the discrete-GPU one.
let cachedUnified = false;
let cachedRamTotal = 0;

// Library preferences
let prefs = loadPrefs();

function loadPrefs() {
    const def = {
        viewMode: 'cards',
        search: '',
        sort: 'name-asc',
        showMmproj: true,
        showMain: true,
        showSplit: true,
        showDraftModels: false,
        sizeMaxGb: 0,
        quantFilters: {},
        tagFilter: '',
        familyFilter: '',
        sizeClassFilter: '',
    };
    try {
        const raw = localStorage.getItem(PREFS_KEY);
        if (raw) {
            const saved = JSON.parse(raw);
            return { ...def, ...saved };
        }
    } catch {
        // ignore
    }
    return def;
}

function savePrefs() {
    try {
        localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
    } catch {
        // ignore
    }
}

export function openModelsModal() {
    document.getElementById('models-modal')?.classList.add('open');
    // Startup inventory notification already fetched a snapshot. Reuse it for
    // the first open to avoid a duplicate request; invalidation paths clear the
    // cache before operations that require a fresh scan.
    void (async () => {
        if (startupInventoryPromise) await startupInventoryPromise;
        loadModels({ refresh: !inventoryCache });
    })();
}

function closeModelsModal() {
    document.getElementById('models-modal')?.classList.remove('open');
}

export { closeModelsModal };

export function invalidateModelInventory() {
    inventoryCache = null;
}

async function loadModels({ refresh = false } = {}) {
    const grid = document.getElementById('models-list');
    const summary = document.getElementById('models-summary');
    const tabCount = document.getElementById('models-tab-count');
    const dirLabel = document.getElementById('models-dir-label');
    if (!grid) return;

    const needsFetch = refresh || !inventoryCache;
    if (needsFetch) {
        if (summary) summary.textContent = 'Loading...';
        grid.innerHTML = '<div class="mm-loading">Scanning...</div>';
    }

    try {
        if (needsFetch) {
            const dirResp = await fetch('/api/hf/download-dir', {
                headers: window.authHeaders ? window.authHeaders() : {},
            });
            if (dirResp.ok && dirLabel) {
                const dirInfo = await dirResp.json();
                if (dirInfo.dir) {
                    modelLibraryRoot = dirInfo.dir;
                    if (dirInfo.configured) {
                        dirLabel.textContent = dirInfo.dir;
                    } else {
                        dirLabel.textContent = 'Using default: ' + dirInfo.dir;
                    }
                } else {
                    dirLabel.textContent = 'No models directory configured';
                }
            }
            const platform = await getPlatformInfo();
            rapidMlxLocalAvailable = platform.rapid_mlx_local_available === true;
            rapidMlxLocalRequirement = platform.rapid_mlx_local_requirement
                || rapidMlxLocalRequirement;
            const resp = await fetch('/api/models', {
                headers: window.authHeaders ? window.authHeaders() : {},
            });
            if (!resp.ok) throw new Error(`Model inventory failed (${resp.status})`);
            inventoryCache = await resp.json();
            if (!inventoryCache.some(model => model.lifecycle === 'incomplete' || model.lifecycle === 'converting')) {
                resolveNotification(INCOMPLETE_DOWNLOAD_NOTIFICATION_ID, 'No incomplete model downloads remain.');
            }
        }
        const models = inventoryCache || [];
        renderLegacyMigrationNotice(models);

        const count = models.length;
        if (tabCount) tabCount.textContent = count ? String(count) : '';

        // Build or refresh toolbar filter chips based on models list
        rebuildLibraryFilters(models);

        // Apply client-side filter/sort/search
        const filtered = applyFilters(models);
        const sorted = applySort(filtered);
        const result = applySearch(sorted);

        if (summary) {
            if (result.length === count) {
                summary.textContent = count
                    ? count + ' model' + (count === 1 ? '' : 's') + ' found'
                    : 'No models found';
            } else {
                summary.textContent = result.length + ' of ' + count + ' models shown';
            }
        }

        if (!count) {
            grid.innerHTML = '<div class="mm-empty">No models found in this directory. You can download one from the Download tab.</div>';
            grid.className = 'mm-model-grid';
            return;
        }

        if (!result.length) {
            grid.innerHTML = '<div class="mm-empty">No models match the current filters or search.</div>';
            grid.className = 'mm-model-grid';
            return;
        }

        grid.className = prefs.viewMode === 'list'
            ? 'mm-model-grid mm-model-grid--list'
            : 'mm-model-grid';

        // The lineage row asks the community source catalog for the uploader's role. Awaiting
        // here rather than inside the card keeps it to one fetch for the whole grid, and
        // without it every card would silently fall back to the repo-name heuristic on the
        // first render. It resolves even when the fetch fails, so a catalog outage delays the
        // grid by one request rather than blocking it.
        await ensureCommunitySourceCatalog();

        grid.innerHTML = '';
        result.forEach(m => {
            grid.appendChild(buildModelCard(m));
        });
    } catch (err) {
        if (summary) summary.textContent = 'Failed to load models';
        const errDiv = document.createElement('div');
        errDiv.className = 'mm-empty';
        errDiv.textContent = 'Error: ' + err.message;
        grid.innerHTML = '';
        grid.appendChild(errDiv);
    }
}

function isMmproj(m) {
    if (m.companion_kind != null) return m.companion_kind === 'mmproj';
    const f = (m.filename || '').toLowerCase();
    return f.includes('mmproj') || f.includes('.mmproj.') || f.includes('-mmproj-');
}

function isDraftModel(m) {
    if (m.companion_kind != null) return m.companion_kind === 'draft';
    // Trust the backend's is_draft_assistant (size-guarded).
    // Fallback: quick client-side check using same logic as backend.
    if (m.is_draft_assistant !== undefined) return m.is_draft_assistant;
    const f = (m.filename || '').toLowerCase();
    const hasKeyword =
        f.includes('mtp-draft') ||
        f.includes('mtp_small') ||
        f.includes('mtp-heads') ||
        f.startsWith('mtp-') ||
        f.endsWith('-mtp.gguf') ||
        f.includes('mtp') ||
        f.includes('draft') ||
        f.includes('assistant') ||
        f.includes('draft-model');
    const size = m.size_bytes || 0;
    // Same tiered thresholds as backend:
    const isUnambiguous =
        f.includes('mtp-draft') ||
        f.includes('mtp_small') ||
        f.includes('mtp-heads') ||
        (f.startsWith('mtp-') && size <= 5_000_000_000);
    if (isUnambiguous && size <= 5_000_000_000) return true;
    if (hasKeyword && size > 0 && size <= 3_000_000_000) return true;
    return false;
}

function applyFilters(models) {
    return models.filter(m => {
        // mmproj vs main
        const mmproj = isMmproj(m);
        if (mmproj && !prefs.showMmproj) return false;
        if (!mmproj && !prefs.showMain) return false;

        // draft model vs main
        const draft = isDraftModel(m);

        if (draft && !prefs.showDraftModels) return false;
        // split
        if (m.is_split && !prefs.showSplit) return false;

        // size filter
        if (prefs.sizeMaxGb > 0) {
            if ((m.size_bytes || 0) > prefs.sizeMaxGb * 1024 ** 3) return false;
        }

        // quant filter
        const qt = (m.quant_type || '').toUpperCase();
        if (qt && Object.keys(prefs.quantFilters).length > 0) {
            if (!prefs.quantFilters[qt]) return false;
        }

        // tag filter
        if (prefs.tagFilter) {
            const tags = Array.isArray(m.tags) ? m.tags : [];
            if (!tags.includes(prefs.tagFilter)) return false;
        }

        // family filter (from backend classification)
        if (prefs.familyFilter) {
            const c = m.classification || {};
            if ((c.family || '') !== prefs.familyFilter) return false;
        }

        // size-class filter (from backend classification)
        if (prefs.sizeClassFilter) {
            const c = m.classification || {};
            if ((c.size_class || '') !== prefs.sizeClassFilter) return false;
        }

        return true;
    });
}

function applySort(models) {
    const mode = prefs.sort || 'name-asc';
    return [...models].sort((a, b) => {
        const attention = model => ({ incomplete: 0, converting: 1, invalid: 2 }[model.lifecycle] ?? 3);
        const attentionDiff = attention(a) - attention(b);
        if (attentionDiff !== 0) return attentionDiff;
        switch (mode) {
            case 'name-asc':
                return (a.model_name || a.filename || '').localeCompare(b.model_name || b.filename || '');
            case 'name-desc':
                return (b.model_name || b.filename || '').localeCompare(a.model_name || a.filename || '');
            case 'size-asc':
                return (a.size_bytes || 0) - (b.size_bytes || 0);
            case 'size-desc':
                return (b.size_bytes || 0) - (a.size_bytes || 0);
            case 'vram-asc':
                return (a.vram_est_gb || 0) - (b.vram_est_gb || 0);
            case 'vram-desc':
                return (b.vram_est_gb || 0) - (a.vram_est_gb || 0);
            case 'date-asc':
                return (a.last_modified || 0) - (b.last_modified || 0);
            case 'date-desc':
                return (b.last_modified || 0) - (a.last_modified || 0);
            default:
                return (a.model_name || a.filename || '').localeCompare(b.model_name || a.filename || '');
        }
    });
}

function applySearch(models) {
    const q = (prefs.search || '').trim().toLowerCase();
    if (!q) return models;
    return models.filter(m => {
        const haystack = [
            m.model_name,
            m.filename,
            m.path,
            m.quant_type,
        ]
            .filter(Boolean)
            .join(' ')
            .toLowerCase();
        return haystack.includes(q);
    });
}

function renderLegacyMigrationNotice(models) {
    const notice = document.getElementById('mm-legacy-migration-notice');
    if (!notice) return;
    const legacy = models.filter(model => model.legacy_location);
    notice.hidden = legacy.length === 0;
    if (!legacy.length || notice.dataset.wired === '1') return;
    notice.dataset.wired = '1';
    const count = notice.querySelector('[data-legacy-count]');
    if (count) count.textContent = String(legacy.length);
    notice.querySelector('[data-migrate-legacy]')?.addEventListener('click', migrateLegacyLibrary);
}

async function migrateLegacyLibrary() {
    try {
        const headers = window.authHeaders
            ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
            : { 'Content-Type': 'application/json' };
        const preview = await fetch('/api/models/library/migration/preview', {
            method: 'POST', headers, body: '{}',
        });
        const plan = await preview.json().catch(() => ({}));
        if (!preview.ok) throw new Error(plan.error || 'Could not preview legacy migration');
        if (!plan.moves?.length) {
            invalidateModelInventory();
            await loadModels({ refresh: true });
            showToast('The library is already organized.', 'info');
            return;
        }
        const confirmed = await _showConfirm(
            'Organize legacy model files',
            `${plan.moves.length} file(s) will move into gguf/ and .staging/downloads/. Existing files are never overwritten. Continue?`,
        );
        if (!confirmed) return;
        const tokenResponse = await fetch('/api/db/admin-token', {
            headers: window.authHeaders ? window.authHeaders() : {},
        });
        const tokenData = await tokenResponse.json().catch(() => ({}));
        if (!tokenResponse.ok || !tokenData.token) throw new Error('DB admin authorization is unavailable');
        const execute = await fetch('/api/models/library/migration/execute', {
            method: 'POST',
            headers: { Authorization: `Bearer ${tokenData.token}`, 'Content-Type': 'application/json' },
            body: JSON.stringify({ plan_id: plan.plan_id, confirmation: 'MIGRATE_MODEL_LIBRARY' }),
        });
        const result = await execute.json().catch(() => ({}));
        if (!execute.ok || !result.ok) throw new Error(result.error || 'Legacy migration failed');
        showToast(`Organized ${plan.moves.length} legacy model file(s)`, 'success');
        invalidateModelInventory();
        await loadModels({ refresh: true });
    } catch (error) {
        showToast(`Legacy migration failed: ${error.message || error}`, 'error');
    }
}

async function notifyIncompleteDownloadsAtStartup() {
    if (startupInventoryChecked) return;
    startupInventoryChecked = true;
    try {
        // Seed the same platform capability state used by the Library renderer
        // before caching the startup inventory. Otherwise the first Library
        // open reuses the snapshot without knowing Rapid-MLX is available and
        // incorrectly downgrades provisional MLX entries to unsupported.
        const platform = await getPlatformInfo();
        rapidMlxLocalAvailable = platform.rapid_mlx_local_available === true;
        rapidMlxLocalRequirement = platform.rapid_mlx_local_requirement
            || rapidMlxLocalRequirement;
        const response = await fetch('/api/models', {
            headers: window.authHeaders ? window.authHeaders() : {},
        });
        if (!response.ok) return;
        const models = await response.json();
        inventoryCache = models;
        const incomplete = models.filter(model => model.lifecycle === 'incomplete' || model.lifecycle === 'converting');
        if (incomplete.length) {
            showToastWithActions(
                `${incomplete.length} incomplete model download${incomplete.length === 1 ? '' : 's'} need attention`,
                'warning',
                'Resume or remove them from the Model Library.',
                [{
                    id: 'open-model-library',
                    label: 'Open Model Library',
                    primary: true,
                    handler: () => openModelsModal(),
                }],
                { notificationId: INCOMPLETE_DOWNLOAD_NOTIFICATION_ID },
            );
        } else {
            resolveNotification(INCOMPLETE_DOWNLOAD_NOTIFICATION_ID, 'No incomplete model downloads remain.');
        }
    } catch {
        // Startup notification is advisory; opening the Library performs the authoritative scan.
    }
}

// Where a model came from, for the card's lineage row.
//
// This row shipped in Phase 8B2 reading `hf_repo_id || originRepo || repo_id`. None of those
// three ever existed on an inventory entry, so it had never rendered. Both sources below are
// real fields the backend populates: `download_provenance` is written when the app downloads
// a file, and `model_source` carries the repo for snapshot directories imported from the
// shared Hugging Face cache.
function resolveLineage(m) {
    const dp = m.download_provenance;
    if (dp?.repoId) {
        return {
            repoId: dp.repoId,
            revision: dp.revision || '',
            pinned: !!dp.pinned,
            sourceUrl: dp.sourceUrl || '',
        };
    }
    // A snapshot directory records its repo and the snapshot's own commit, so it is always
    // pinned -- the commit is the directory name, not a branch we happened to read.
    if (m.model_source?.kind === 'hugging_face_repo' && m.model_source.repo_id) {
        const revision = m.model_source.revision || '';
        return {
            repoId: m.model_source.repo_id,
            revision,
            pinned: !!revision && revision !== 'main',
            sourceUrl: `https://huggingface.co/${m.model_source.repo_id}`,
        };
    }
    return null;
}

function buildLineageRow(m) {
    const lineage = resolveLineage(m);
    if (!lineage) return null;

    const lineageEl = document.createElement('div');
    lineageEl.className = 'mm-card-lineage';

    const repoLink = document.createElement('a');
    repoLink.className = 'mm-lineage-repo';
    repoLink.textContent = lineage.repoId;
    repoLink.href = lineage.sourceUrl;
    repoLink.target = '_blank';
    repoLink.rel = 'noopener noreferrer';
    lineageEl.appendChild(repoLink);

    // A commit is shown only when there is one. An unpinned download came from whatever the
    // branch pointed at, and rendering "main" as though it were a revision would present a
    // model as reproducible when it is not.
    if (lineage.pinned && lineage.revision) {
        lineageEl.appendChild(document.createTextNode(' · '));
        const revSpan = document.createElement('span');
        revSpan.className = 'mm-lineage-rev';
        const short = lineage.revision.slice(0, 7);
        revSpan.textContent = short === lineage.revision ? short : short + '…';
        revSpan.title = `Pinned to commit ${lineage.revision}`;
        lineageEl.appendChild(revSpan);
    }

    // The uploader's role comes from the community source catalog, the same lookup the HF
    // browse results use. The format stands in for the repo tags the browse path has: a
    // locally downloaded GGUF is the evidence a `gguf` tag would have been.
    const roleInfo = resolveAuthorRole(lineage.repoId, m.format ? [m.format] : []);
    if (roleInfo) {
        lineageEl.appendChild(document.createTextNode(' · '));
        const roleSpan = document.createElement('span');
        roleSpan.className = `mm-lineage-role mm-lineage-role--${roleInfo.role}`;
        roleSpan.textContent = roleInfo.label;
        if (roleInfo.description) roleSpan.title = roleInfo.description;
        lineageEl.appendChild(roleSpan);
    }

    return lineageEl;
}

async function openMlxIntrospectionEvidence(model, opener) {
    const { openEvidenceDrawer } = await import('./evidence-drawer.js');
    let payload = {};
    try {
        const response = await fetch('/api/models/mlx-introspect', {
            method: 'POST',
            headers: { ...(window.authHeaders ? window.authHeaders() : {}), 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_path: model.path || '' }),
        });
        payload = await response.json().catch(() => ({}));
        if (!response.ok || payload.ok === false) throw new Error(payload.error || `Introspection failed (${response.status})`);
    } catch (error) {
        payload = { error: error?.message || String(error) };
    }
    const data = payload.data || {};
    const config = data.config || {};
    const errors = [payload.error, ...(data.errors || [])].filter(Boolean);
    openEvidenceDrawer({
        title: `${model.model_name || model.filename || 'MLX model'} evidence`,
        status: errors.length || !Object.keys(config).length ? 'caution' : 'good',
        summary: Object.keys(config).length
            ? 'Local MLX architecture metadata was read from the configured model directory.'
            : 'Local MLX architecture metadata could not be fully resolved.',
        consequence: 'This introspection is used to ground memory and compatibility decisions in local artifacts.',
        remediation: errors.length ? 'Verify the model directory contains a readable config.json and weight index.' : '',
        evidence: [
            config.model_type ? `Model type: ${config.model_type}` : '',
            config.num_layers ? `Layers: ${config.num_layers}` : '',
            config.max_position_embeddings ? `Native context: ${Number(config.max_position_embeddings).toLocaleString()} tokens` : '',
            config.quantization ? `Quantization: ${JSON.stringify(config.quantization)}` : '',
            config.vision_confidence ? `Vision evidence: ${config.vision_confidence} (${config.vision_source || 'source unavailable'})` : '',
        ].filter(Boolean),
        warnings: errors,
        provenance: [payload.model_path ? `Local path: ${payload.model_path}` : ''].filter(Boolean),
    }, opener);
}

function buildModelCard(m) {
    const name = m.model_name || m.filename || 'Unnamed model';
    const quant = m.quant_type || 'unknown';
    const size = m.size_display || '';
    const vramEst = m.vram_est_gb != null
        ? (typeof m.vram_estimate_display === 'string'
            ? m.vram_estimate_display
            : (m.vram_est_gb % 1 === 0 ? m.vram_est_gb + ' GB' : m.vram_est_gb.toFixed(1) + ' GB'))
        : '';
    const vramPct = m.vram_percent != null ? Math.min(100, m.vram_percent) : null;
    const isSplit = m.is_split;
    const inventory = normalizeInventory(m);
    const mmproj = inventory.companionKind === 'mmproj';
    const companion = inventory.companionKind !== null;
    const tags = Array.isArray(m.tags) ? m.tags : [];
    const relatedPresets = mmproj ? [] : findPresetsForModel(m);
    const isPartialFile = inventory.lifecycle === 'incomplete'
        && (m.path || '').toLowerCase().endsWith('.part');

    const card = document.createElement('article');
    card.className = 'mm-model-card';
    card.dataset.format = inventory.format;
    card.dataset.source = inventory.source;
    card.dataset.lifecycle = inventory.lifecycle;
    card.dataset.compatibility = inventory.compatibility;
    if (mmproj) card.classList.add('mm-model-card--mmproj');
    if (companion) card.classList.add('mm-model-card--companion');

    // Top row: name + quant badge
    const top = document.createElement('div');
    top.className = 'mm-card-top';

    const nameEl = document.createElement('div');
    nameEl.className = 'mm-card-name';
    modelCardSequence += 1;
    nameEl.id = `mm-model-name-${modelCardSequence}`;
    card.setAttribute('aria-labelledby', nameEl.id);
    nameEl.title = m.path || '';
    nameEl.textContent = name;
    top.appendChild(nameEl);

    // Only show quant badge when it's meaningful — skip for mmproj files with no known quant
    if (quant !== 'unknown') {
        const badge = document.createElement('span');
        badge.className = 'mm-quant-badge';
        badge.textContent = quant;
        top.appendChild(badge);
    }

    if (isSplit) {
        const splitBadge = document.createElement('span');
        splitBadge.className = 'mm-quant-badge mm-split-badge';
        splitBadge.textContent = 'split';
        top.appendChild(splitBadge);
    }

    // MTP / draft model badge
    const isDraft = m.is_draft_assistant || (m.classification && m.classification.has_mtp);
    if (isDraft && !mmproj) {
        const mtpBadge = document.createElement('span');
        mtpBadge.className = 'mm-quant-badge mm-mtp-badge';
        mtpBadge.textContent = 'MTP';
        mtpBadge.title = 'Multi-token prediction / draft model';
        top.appendChild(mtpBadge);
    }

    card.appendChild(top);

    card.appendChild(buildInventoryBadgeRail(inventory));

    // Meta row: filename (ellipsis + full tooltip)
    const meta = document.createElement('div');
    meta.className = 'mm-card-meta';
    const metaText = m.filename || '';
    meta.textContent = metaText;
    if (metaText) meta.title = metaText;
    card.appendChild(meta);

    // Stats row: size, tag pills (always include if any)
    if (size || tags.length > 0) {
        const stats = document.createElement('div');
        stats.className = 'mm-card-stats';
        if (size) {
            const sizeEl = document.createElement('span');
            sizeEl.className = 'mm-stat';
            sizeEl.textContent = size;
            stats.appendChild(sizeEl);
        }
        tags.forEach(tag => {
            const pill = document.createElement('span');
            pill.className = 'mm-tag-pill';
            pill.textContent = tag;
            pill.title = 'Click to remove tag';
            pill.addEventListener('click', e => {
                e.stopPropagation();
                removeModelTag(m.path, tag);
            });
            stats.appendChild(pill);
        });
        card.appendChild(stats);
    }

    // VRAM estimate: always show when available, regardless of presets.
    if (vramEst) {
        const vramEl = document.createElement('div');
        vramEl.className = 'mm-card-stats';
        const vramSpan = document.createElement('span');
        vramSpan.className = 'mm-stat mm-stat-vram';
        vramSpan.textContent = 'VRAM ~' + vramEst;
        vramEl.appendChild(vramSpan);
        card.appendChild(vramEl);
    }

    // VRAM bar: always show when available, regardless of presets.
    if (vramPct !== null) {
        const barWrap = document.createElement('div');
        barWrap.className = 'mm-vram-bar';
        const fill = document.createElement('div');
        fill.className = 'mm-vram-fill';
        fill.style.width = vramPct + '%';
        if (vramPct > 90) fill.classList.add('mm-vram-fill--warn');
        barWrap.appendChild(fill);
        card.appendChild(barWrap);
    }

    const lineageEl = buildLineageRow(m);
    if (lineageEl) card.appendChild(lineageEl);

    if (relatedPresets.length) {
        const presetMeta = document.createElement('div');
        presetMeta.className = 'mm-card-meta';
        presetMeta.textContent = buildPresetSummary(relatedPresets);
        presetMeta.title = relatedPresets.map(formatPresetSummaryLine).join('\n');
        card.appendChild(presetMeta);
    }

    // Actions row
    const actions = document.createElement('div');
    actions.className = 'mm-card-actions';
    if (inventory.lifecycle === 'incomplete' && (m.path || '').toLowerCase().endsWith('.part')) {
        actions.classList.add('mm-card-actions--incomplete');
    }

    if (!companion && inventory.launchable) {
        const serverRunning = isLocalServerRunning();

        if (serverRunning) {
            // ── Server is running: primary action is Switch or Quick Load ──────

            if (relatedPresets.length === 1) {
                // One preset → simple Switch button
                const switchBtn = document.createElement('button');
                switchBtn.type = 'button';
                switchBtn.className = 'mm-action-btn mm-action-btn--switch';
                switchBtn.title = `Switch to preset: ${relatedPresets[0].name}`;
                switchBtn.innerHTML = '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M7 16V4m0 0L3 8m4-4l4 4"/><path d="M17 8v12m0 0l4-4m-4 4l-4-4"/></svg> Switch';
                switchBtn.addEventListener('click', () => doSwitchToPreset(relatedPresets[0].id));
                actions.appendChild(switchBtn);

            } else if (relatedPresets.length > 1) {
                // Multiple presets → inline select + Switch button
                const switchWrap = document.createElement('div');
                switchWrap.className = 'mm-switch-wrap';

                const switchSelect = document.createElement('select');
                switchSelect.className = 'mm-switch-select';
                switchSelect.title = 'Pick which preset to load';
                relatedPresets.forEach(p => {
                    const opt = document.createElement('option');
                    opt.value = p.id;
                    const ctxLabel = p.context_size ? ' · ' + Math.round(p.context_size / 1024) + 'k' : '';
                    opt.textContent = p.name + ctxLabel;
                    switchSelect.appendChild(opt);
                });

                const switchBtn = document.createElement('button');
                switchBtn.type = 'button';
                switchBtn.className = 'mm-action-btn mm-action-btn--switch';
                switchBtn.title = 'Switch to selected preset';
                switchBtn.innerHTML = '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M7 16V4m0 0L3 8m4-4l4 4"/><path d="M17 8v12m0 0l4-4m-4 4l-4-4"/></svg> Switch';
                switchBtn.addEventListener('click', () => doSwitchToPreset(switchSelect.value));

                switchWrap.appendChild(switchSelect);
                switchWrap.appendChild(switchBtn);
                actions.appendChild(switchWrap);

            } else if (inventory.supportedBackends.includes('llama_cpp')) {
                // No preset → Quick Load (inherits current server settings)
                const loadBtn = document.createElement('button');
                loadBtn.type = 'button';
                loadBtn.className = 'mm-action-btn mm-action-btn--switch';
                loadBtn.title = 'Load this model using current server settings (port, GPU layers, etc.)';
                loadBtn.innerHTML = '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M7 16V4m0 0L3 8m4-4l4 4"/><path d="M17 8v12m0 0l4-4m-4 4l-4-4"/></svg> Quick Load';
                loadBtn.addEventListener('click', () => doQuickLoad(m));
                actions.appendChild(loadBtn);
            } else {
                appendRapidMlxConfigureButton(actions, m);
            }

            // Edit preset(s) — secondary action when running
            if (relatedPresets.length > 0) {
                const editBtn = document.createElement('button');
                editBtn.type = 'button';
                editBtn.className = 'mm-action-btn';
                editBtn.title = relatedPresets.length === 1
                    ? 'Edit the preset for this model'
                    : `Edit one of ${relatedPresets.length} presets`;
                editBtn.textContent = relatedPresets.length === 1 ? 'Edit' : 'Edit…';
                editBtn.addEventListener('click', () => _openEditPreset(relatedPresets));
                actions.appendChild(editBtn);
            }

        } else {
            // ── Server is not running: wizard / new-preset ───────────────────

            if (inventory.supportedBackends.includes('llama_cpp')) {
                const useBtn = document.createElement('button');
                useBtn.type = 'button';
                useBtn.className = 'mm-action-btn';
                useBtn.title = relatedPresets.length ? 'Build a new preset from this model' : 'Open this model in the spawn wizard';
                useBtn.textContent = relatedPresets.length ? 'New Preset' : 'Use in Wizard';
                useBtn.addEventListener('click', () => {
                    closeModelsModal();
                    window.__spawnWizardOpts = {
                        localPath: m.path || '',
                        localModel: m,
                    };
                    Router.navigate('/spawn');
                });
                actions.appendChild(useBtn);
            } else if (!relatedPresets.length) {
                appendRapidMlxConfigureButton(actions, m);
            }

            if (relatedPresets.length) {
                const editBtn = document.createElement('button');
                editBtn.type = 'button';
                editBtn.className = 'mm-action-btn';
                editBtn.title = relatedPresets.length === 1
                    ? 'Edit the saved preset that uses this model'
                    : `Edit one of the ${relatedPresets.length} presets using this model`;
                editBtn.textContent = relatedPresets.length === 1 ? 'Edit Preset' : 'Edit Presets…';
                editBtn.addEventListener('click', () => _openEditPreset(relatedPresets));
                actions.appendChild(editBtn);
            }
        }
    } else {
        const unavailable = document.createElement('div');
        unavailable.className = `mm-card-action-note${isPartialFile ? ' mm-card-action-note--warning' : ''}`;
        unavailable.setAttribute('role', 'status');
        unavailable.textContent = isPartialFile
            ? 'Resume unavailable: this partial file has no download source metadata.'
            : inventoryActionNote(inventory);
        actions.appendChild(unavailable);
    }

    const copyBtn = document.createElement('button');
    copyBtn.type = 'button';
    copyBtn.className = 'mm-action-btn mm-action-copy';
    copyBtn.title = 'Copy path';
    copyBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="13" height="13"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy Path';
    // Typed HF sources retain a stable repo id while `path` points at a resolved
    // cache snapshot. Copy the source users can reuse in configuration.
    const pathToCopy = inventoryModelSourceValue(m) || m.path || '';
    copyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(pathToCopy).then(() => {
            showToast('Path copied', 'success');
        }).catch(() => {
            showToast('Copy failed', 'error');
        });
    });
    if (pathToCopy) actions.appendChild(copyBtn);

    if (inventory.format === 'mlx' && m.path) {
        const explainBtn = document.createElement('button');
        explainBtn.type = 'button';
        explainBtn.className = 'mm-action-btn';
        explainBtn.textContent = 'Explain';
        explainBtn.title = 'Inspect local MLX architecture evidence';
        explainBtn.addEventListener('click', async () => {
            explainBtn.disabled = true;
            await openMlxIntrospectionEvidence(m, explainBtn);
            explainBtn.disabled = false;
        });
        actions.appendChild(explainBtn);
    }

    if (!companion
        && inventory.format === 'mlx'
        && inventory.lifecycle === 'ready'
        && inventory.supportedBackends.includes('rapid_mlx')
        && (m.path || '').startsWith('/')) {
        const repairBtn = document.createElement('button');
        repairBtn.type = 'button';
        repairBtn.className = 'mm-action-btn mm-action-btn--switch';
        repairBtn.textContent = 'Repair MTP sidecar';
        repairBtn.title = 'Build, reassemble, or validate a managed MTP sidecar for this MLX trunk';
        repairBtn.addEventListener('click', () => {
            closeModelsModal();
            openPresetMtpRepairForModel(m.path, name);
        });
        actions.appendChild(repairBtn);
    }

    if (m.resume_download) {
        const resumeBtn = document.createElement('button');
        resumeBtn.type = 'button';
        resumeBtn.className = 'mm-action-btn mm-action-btn--switch';
        resumeBtn.textContent = 'Resume download';
        resumeBtn.title = `Resume ${m.resume_download.repo_id}/${m.resume_download.file_path}`;
        resumeBtn.addEventListener('click', () => resumeModelDownload(m.path));
        actions.appendChild(resumeBtn);
    } else if (isPartialFile) {
        const findSourceBtn = document.createElement('button');
        findSourceBtn.type = 'button';
        findSourceBtn.className = 'mm-action-btn mm-action-btn--switch';
        findSourceBtn.textContent = 'Find source';
        findSourceBtn.title = 'Search Hugging Face for a matching repository and GGUF file';
        findSourceBtn.addEventListener('click', () => openResumeSourcePicker(m));
        actions.appendChild(findSourceBtn);
    }

    // Two delete paths, because a single-file model and a directory-shaped one have
    // different endpoints and different safety arguments. A GGUF goes through the
    // suffix-validated file delete; MLX and Transformers directories go through the
    // library's managed-directory delete, which only accepts an immediate child of a
    // managed root. Anything else — notably HF cache snapshots, whose bytes live in a
    // shared blob store — deletes the owning app-managed repo, not an individual snapshot.
    const isGgufFile = inventory.format === 'gguf' && (m.path || '').toLowerCase().endsWith('.gguf');
    const isManagedDirectory = MANAGED_DIRECTORY_FORMATS.has(inventory.format)
        && MANAGED_DIRECTORY_SOURCES.has(inventory.source);
    const isManagedCache = !!m.managed_cache_repo;
    if (isGgufFile || isManagedDirectory || isManagedCache || isPartialFile) {
        const deleteBtn = document.createElement('button');
        deleteBtn.type = 'button';
        deleteBtn.className = 'mm-action-btn mm-action-delete';
        deleteBtn.title = isPartialFile
            ? 'Delete incomplete download'
            : isManagedCache
            ? `Delete cached Hugging Face repository ${m.managed_cache_repo}`
            : isGgufFile
                ? 'Delete this model from library'
                : 'Delete this model directory from library';
        deleteBtn.setAttribute('aria-label', deleteBtn.title);
        deleteBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="13" height="13"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/></svg>';
        deleteBtn.addEventListener('click', () => (isPartialFile
            ? deletePartialModel(m.path, m.filename || name)
            : isManagedCache
            ? deleteManagedCache(m.managed_cache_repo, name, m.size_display)
            : isGgufFile
                ? deleteModel(m.path, m.filename || name)
                : deleteModelDirectory(m.path, m.filename || name, m.size_display)));
        actions.appendChild(deleteBtn);
    }

    const tagBtn = document.createElement('button');
    tagBtn.type = 'button';
    tagBtn.className = 'mm-action-btn mm-action-tags';
    tagBtn.title = 'Add tag';
    tagBtn.setAttribute('aria-label', 'Add tag');
    tagBtn.textContent = '+';
    tagBtn.addEventListener('click', () => {
        openTagPicker(m.path, tags);
    });
    actions.appendChild(tagBtn);

    card.appendChild(actions);
    return card;
}

function normalizeInventory(model) {
    const format = Object.hasOwn(INVENTORY_BADGES.format, model.format) ? model.format : 'unknown';
    const source = Object.hasOwn(INVENTORY_BADGES.source, model.source) ? model.source : 'unknown';
    const lifecycle = Object.hasOwn(INVENTORY_BADGES.lifecycle, model.lifecycle) ? model.lifecycle : 'unknown';
    let compatibility = Object.hasOwn(INVENTORY_BADGES.compatibility, model.compatibility)
        ? model.compatibility
        : 'unknown';
    const advertisedBackends = Array.isArray(model.supported_backends)
        ? [...new Set(model.supported_backends.filter(backend => Object.hasOwn(BACKEND_LABELS, backend)))]
        : [];
    const rapidMlxPlatformBlocked = advertisedBackends.includes('rapid_mlx')
        && !rapidMlxLocalAvailable;
    const supportedBackends = rapidMlxPlatformBlocked
        ? advertisedBackends.filter(backend => backend !== 'rapid_mlx')
        : advertisedBackends;
    if (rapidMlxPlatformBlocked && supportedBackends.length === 0) compatibility = 'unsupported';
    const companionKind = ['mmproj', 'draft'].includes(model.companion_kind)
        ? model.companion_kind
        : null;
    return {
        format,
        source,
        lifecycle,
        compatibility,
        supportedBackends,
        rapidMlxPlatformBlocked,
        companionKind,
        launchable: lifecycle === 'ready' && compatibility !== 'unsupported' && supportedBackends.length > 0,
    };
}

function createInventoryBadge(category, value, descriptor) {
    const badge = document.createElement('span');
    badge.className = `mm-inventory-badge mm-inventory-badge--${category} mm-inventory-badge--${value}`;
    badge.textContent = descriptor[0];
    badge.title = descriptor[1];
    badge.setAttribute('aria-label', descriptor[1]);
    return badge;
}

function buildInventoryBadgeRail(inventory) {
    const rail = document.createElement('div');
    rail.className = 'mm-inventory-badges';
    rail.setAttribute('aria-label', 'Model inventory metadata');
    rail.appendChild(createInventoryBadge('format', inventory.format, INVENTORY_BADGES.format[inventory.format]));
    rail.appendChild(createInventoryBadge('source', inventory.source, INVENTORY_BADGES.source[inventory.source]));
    rail.appendChild(createInventoryBadge('lifecycle', inventory.lifecycle, INVENTORY_BADGES.lifecycle[inventory.lifecycle]));
    rail.appendChild(createInventoryBadge(
        'compatibility',
        inventory.compatibility,
        INVENTORY_BADGES.compatibility[inventory.compatibility],
    ));

    if (inventory.supportedBackends.length) {
        inventory.supportedBackends.forEach(backend => {
            rail.appendChild(createInventoryBadge(
                'backend',
                backend,
                [BACKEND_LABELS[backend], `Supported backend: ${BACKEND_LABELS[backend]}`],
            ));
        });
    } else if (inventory.rapidMlxPlatformBlocked) {
        rail.appendChild(createInventoryBadge(
            'backend',
            'platform-unavailable',
            ['Apple Silicon required', rapidMlxLocalRequirement],
        ));
    } else {
        rail.appendChild(createInventoryBadge('backend', 'none', ['No backend', 'Supported backend: None']));
    }

    if (inventory.companionKind) {
        const descriptor = inventory.companionKind === 'mmproj'
            ? ['Vision companion', 'Companion type: Multimodal projector']
            : ['Draft companion', 'Companion type: Draft / MTP model'];
        rail.appendChild(createInventoryBadge('companion', inventory.companionKind, descriptor));
    }
    return rail;
}

function inventoryActionNote(inventory) {
    if (inventory.companionKind === 'mmproj') return 'Vision companion — select it with a compatible primary model.';
    if (inventory.companionKind === 'draft') return 'Draft companion — select it from speculative decoding settings.';
    if (inventory.lifecycle === 'converting') return 'Conversion in progress. Launch will be available after validation.';
    if (inventory.lifecycle === 'incomplete') return 'Incomplete model. Finish the download or conversion before launch.';
    if (inventory.lifecycle === 'invalid') return 'Invalid model. Review its files or provenance before launch.';
    if (inventory.rapidMlxPlatformBlocked) return `${rapidMlxLocalRequirement}. You can still manage or copy this model.`;
    if (inventory.compatibility === 'unsupported') return 'No installed backend supports this model.';
    if (inventory.source === 'recovered_gguf') return 'Experimental GGUF recovery; launch is disabled pending profile promotion.';
    if (inventory.source === 'requantized_mlx') return 'Experimental MLX re-quantization; launch is disabled pending profile promotion.';
    return 'Backend compatibility has not been verified.';
}

function appendRapidMlxConfigureButton(actions, model) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'mm-action-btn mm-action-btn--switch';
    button.title = 'Create a Rapid-MLX preset for this model';
    button.textContent = 'Configure Rapid-MLX';
    button.addEventListener('click', async () => {
        const modelPath = inventoryModelSourceValue(model) || model.path || '';
        const modelSource = model.model_source || {
            kind: 'mlx_directory',
            path: model.path || modelPath,
        };
        const seed = {
            backend: 'rapid_mlx',
            name: `${model.model_name || model.filename || 'MLX model'} · Rapid-MLX`,
            model_path: '',
            port: 8000,
            rapid_mlx: {
                model_path: modelPath,
                model_source: modelSource,
                host: '127.0.0.1',
                port: 8000,
                log_level: 'INFO',
            },
        };
        closeModelsModal();
        const { openPresetModal } = await import('./presets.js');
        openPresetModal('new', 'model', seed);
    });
    actions.appendChild(button);
}

function inventoryModelSourceValue(model) {
    const source = model.model_source;
    if (!source || typeof source !== 'object') return '';
    if (source.kind === 'mlx_directory' || source.kind === 'gguf_file') return source.path || '';
    if (source.kind === 'hugging_face_repo') return source.repo_id || '';
    if (source.kind === 'alias') return source.value || '';
    if (source.kind === 'authoritative_safetensors') {
        return source.source?.path || source.source?.repo_id || '';
    }
    return '';
}

function isLocalServerRunning() {
    return !document.getElementById('btn-stop')?.disabled;
}

function findPresetsForModel(model) {
    const paths = new Set([model.path, inventoryModelSourceValue(model)].filter(Boolean));
    if (!paths.size) return [];
    return (sessionState.presets || []).filter(preset => {
        if (paths.has(preset.model_path)) return true;
        const rapidMlx = preset.rapid_mlx;
        if (!rapidMlx) return false;
        const candidate = rapidMlx.model_source_view?.canonical_identity
            || rapidMlx.model_source_view?.display_name;
        return candidate ? paths.has(candidate) : false;
    });
}

async function doSwitchToPreset(presetId) {
    closeModelsModal();
    const { syncSelectedPresetSelection } = await import('./presets.js');
    syncSelectedPresetSelection(presetId, { userIntent: true, persist: true });
    const { doStart } = await import('./attach-detach.js');
    await doStart(null, { skipRunningConfirm: true });
}

async function doQuickLoad(model) {
    closeModelsModal();

    // Derive base config from the currently running preset
    const runningPreset = sessionState.presets?.find(p => p.id === sessionState.activeSessionPresetId) || {};

    const config = {
        preset_id: '',
        model_path: model.path || '',
        hf_repo: null,
        context_size: runningPreset.context_size || 32768,
        ctk: runningPreset.ctk || 'q8_0',
        ctv: runningPreset.ctv || 'f16',
        port: runningPreset.port || 8001,
        bind_host: runningPreset.bind_host || '127.0.0.1',
        api_key: runningPreset.api_key || null,
        gpu_layers: runningPreset.gpu_layers ?? null,
        threads: runningPreset.threads ?? null,
        threads_batch: runningPreset.threads_batch ?? null,
        batch_size: runningPreset.batch_size || 2048,
        ubatch_size: runningPreset.ubatch_size || runningPreset.batch_size || 2048,
        flash_attn: runningPreset.flash_attn || '',
        no_mmap: !!runningPreset.no_mmap,
        mlock: !!runningPreset.mlock,
        parallel_slots: runningPreset.parallel_slots || 1,
        tensor_split: runningPreset.tensor_split || '',
        split_mode: runningPreset.split_mode || '',
        main_gpu: runningPreset.main_gpu ?? null,
        kv_unified: runningPreset.kv_unified ?? null,
        cache_ram_mib: runningPreset.cache_ram_mib ?? null,
        // Clear model-specific fields — let llama-server auto-detect from GGUF
        mmproj: null,
        chat_template_file: null,
        reasoning: null,
        enable_thinking: null,
        preserve_thinking: null,
        reasoning_budget: null,
        reasoning_budget_message: null,
        draft_model: '',
        draft_min: null,
        draft_max: null,
        spec_type: null,
        spec_ngram_size: null,
        spec_draft_n_max: null,
        ngram_spec: false,
        seed: runningPreset.seed ?? null,
        alias: null,
        max_tokens: runningPreset.max_tokens ?? null,
        fit_enabled: null,
        fit_target: null,
        system_prompt_file: '',
        extra_args: runningPreset.extra_args || '',
        // Generation defaults — leave unset so llama-server uses its own defaults
        temperature: runningPreset.temperature,
        top_p: runningPreset.top_p,
        top_k: runningPreset.top_k,
        min_p: runningPreset.min_p,
        repeat_penalty: runningPreset.repeat_penalty,
        presence_penalty: runningPreset.presence_penalty ?? null,
        n_cpu_moe: runningPreset.n_cpu_moe,
        rope_scaling: runningPreset.rope_scaling || '',
        rope_freq_base: runningPreset.rope_freq_base ?? null,
        rope_freq_scale: runningPreset.rope_freq_scale ?? null,
    };

    // Cap context to the model's training limit if known
    if (model.n_ctx_train > 0 && config.context_size > model.n_ctx_train) {
        config.context_size = model.n_ctx_train;
    }

    const modelName = model.model_name || model.filename || 'model';
    showToast('Loading ' + modelName + '…', 'info', 'Stopping current server', { duration: 14000 });

    const { doStartWithConfig } = await import('./attach-detach.js');
    await doStartWithConfig(config, { skipRunningConfirm: true });
}

function formatPresetSummaryLine(preset) {
    const parts = [preset.name || 'Unnamed preset'];
    if (preset.context_size) parts.push(`${Math.round(preset.context_size / 1024)}k context`);
    const ctk = preset.ctk || '';
    const ctv = preset.ctv || '';
    if (ctk || ctv) parts.push(`KV ${ctk || 'default'}/${ctv || 'default'}`);
    if (preset.reasoning) parts.push(`reasoning ${preset.reasoning}`);
    else if (preset.enable_thinking != null) parts.push(preset.enable_thinking ? 'thinking on' : 'thinking off');
    if (preset.mmproj) parts.push('vision');
    if (preset.bind_host === '0.0.0.0') parts.push('LAN');
    if (preset.api_key) parts.push('API key');
    return parts.join(' · ');
}

function _openEditPreset(presets) {
    if (presets.length === 1) {
        const select = document.getElementById('preset-select');
        if (select) select.value = presets[0].id;
        closeModelsModal();
        import('./presets.js').then(({ openPresetModal }) => openPresetModal('edit'));
        return;
    }

    // Multiple presets: show an inline chooser via toast instead of prompt()
    showToastWithActions(
        'Which preset to edit?',
        'info',
        presets.map(p => p.name).join(' · '),
        presets.slice(0, 3).map((p, i) => ({
            id: 'p' + i,
            label: p.name,
            primary: i === 0,
            handler: () => {
                const select = document.getElementById('preset-select');
                if (select) select.value = p.id;
                closeModelsModal();
                import('./presets.js').then(({ openPresetModal }) => openPresetModal('edit'));
            },
        })),
        { duration: 12000 },
    );
}

function buildPresetSummary(presets) {
    if (!presets.length) return '';
    const first = presets[0];
    const summary = formatPresetSummaryLine(first);
    if (presets.length === 1) {
        return `Saved preset: ${summary}`;
    }
    return `Saved presets (${presets.length}): ${summary} +${presets.length - 1} more`;
}

function resumeSourceFilename(model) {
    const filename = model.filename || model.path?.split(/[\\/]/).pop() || '';
    const withoutPart = filename.replace(/\.part$/i, '');
    return withoutPart.toLowerCase().endsWith('.gguf') ? withoutPart : `${withoutPart}.gguf`;
}

function createResumeSourcePicker(model) {
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay';
    overlay.style.zIndex = '2000';
    overlay.style.display = 'grid';
    const dialog = document.createElement('div');
    dialog.className = 'modal';
    dialog.style.width = 'min(620px, calc(100vw - 32px))';
    dialog.style.maxHeight = 'min(760px, calc(100vh - 32px))';
    dialog.style.overflow = 'auto';
    dialog.setAttribute('role', 'dialog');
    dialog.setAttribute('aria-modal', 'true');
    const title = document.createElement('h3');
    title.textContent = 'Find download source';
    const message = document.createElement('p');
    message.className = 'mm-card-meta';
    message.textContent = `Search for the Hugging Face repository and GGUF file matching ${model.filename || 'this partial download'}. Confirm the file before resuming.`;
    const searchRow = document.createElement('div');
    searchRow.style.display = 'flex';
    searchRow.style.gap = '8px';
    const searchInput = document.createElement('input');
    searchInput.className = 'mm-lib-search-input';
    searchInput.style.flex = '1';
    searchInput.placeholder = 'model name or owner/repo';
    searchInput.value = resumeSourceFilename(model).replace(/\.gguf$/i, '');
    searchInput.setAttribute('aria-label', 'Hugging Face model or repository search');
    const searchButton = document.createElement('button');
    searchButton.type = 'button';
    searchButton.className = 'btn-modal-save';
    searchButton.textContent = 'Search';
    searchRow.append(searchInput, searchButton);
    const status = document.createElement('div');
    status.className = 'mm-card-action-note';
    status.style.marginTop = '10px';
    const results = document.createElement('div');
    results.style.display = 'grid';
    results.style.gap = '8px';
    results.style.marginTop = '10px';
    const actions = document.createElement('div');
    actions.style.display = 'flex';
    actions.style.justifyContent = 'flex-end';
    actions.style.marginTop = '14px';
    const cancelButton = document.createElement('button');
    cancelButton.type = 'button';
    cancelButton.className = 'btn-modal-cancel';
    cancelButton.textContent = 'Cancel';
    cancelButton.addEventListener('click', () => overlay.remove());
    actions.appendChild(cancelButton);
    dialog.append(title, message, searchRow, status, results, actions);
    overlay.appendChild(dialog);
    overlay.addEventListener('click', event => {
        if (event.target === overlay) overlay.remove();
    });
    return { overlay, searchInput, searchButton, status, results };
}

async function openResumeSourcePicker(model) {
    const picker = createResumeSourcePicker(model);
    document.body.appendChild(picker.overlay);
    picker.searchInput.focus();
    const authHeaders = () => ({
        ...(window.authHeaders ? window.authHeaders() : {}),
        'Content-Type': 'application/json',
    });
    const expected = resumeSourceFilename(model).toLowerCase();
    const loadFiles = async repoId => {
        picker.status.textContent = `Loading GGUF files from ${repoId}…`;
        picker.results.replaceChildren();
        try {
            const response = await fetch('/api/hf/files', {
                method: 'POST',
                headers: authHeaders(),
                body: JSON.stringify({ repo_id: repoId, format: 'gguf' }),
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok || !data.ok) throw new Error(data.error || 'Could not list repository files');
            const files = (data.files || [])
                .filter(file => file.path && !file.is_mmproj && !file.is_draft_assistant)
                .sort((a, b) => {
                    const aName = a.path.split('/').pop().toLowerCase();
                    const bName = b.path.split('/').pop().toLowerCase();
                    return Number(bName === expected) - Number(aName === expected)
                        || Number(bName.endsWith(expected)) - Number(aName.endsWith(expected))
                        || aName.localeCompare(bName);
                });
            picker.status.textContent = files.length
                ? `Choose the file to resume from ${repoId}.`
                : 'No GGUF files were found in that repository.';
            files.forEach(file => {
                const row = document.createElement('div');
                row.className = 'mm-card-action-note';
                row.style.display = 'flex';
                row.style.alignItems = 'center';
                row.style.justifyContent = 'space-between';
                row.style.gap = '10px';
                const label = document.createElement('span');
                label.textContent = `${file.path}${file.size ? ` · ${formatBytes(file.size)}` : ''}`;
                label.title = file.path;
                const resumeButton = document.createElement('button');
                resumeButton.type = 'button';
                resumeButton.className = 'mm-action-btn mm-action-btn--switch';
                resumeButton.textContent = 'Resume';
                resumeButton.addEventListener('click', async () => {
                    resumeButton.disabled = true;
                    await resumeModelDownload(model.path, {
                        repo_id: repoId,
                        file_path: file.path,
                        save_as: file.path.split('/').pop(),
                    }, picker.overlay);
                    resumeButton.disabled = false;
                });
                row.append(label, resumeButton);
                picker.results.appendChild(row);
            });
        } catch (error) {
            picker.status.textContent = `Could not load repository files: ${error.message || error}`;
        }
    };
    const search = async () => {
        const query = picker.searchInput.value.trim();
        if (!query) return;
        picker.searchButton.disabled = true;
        picker.status.textContent = 'Searching Hugging Face…';
        picker.results.replaceChildren();
        try {
            let candidates;
            if (/^[^/\s]+\/[^/\s]+$/.test(query)) {
                candidates = [{ repoId: query, confidence: 1, reason: 'Repository entered directly' }];
            } else {
                const response = await fetch('/api/hf/resolve-origin', {
                    method: 'POST',
                    headers: authHeaders(),
                    body: JSON.stringify({ filename: query.endsWith('.gguf') ? query : `${query}.gguf`, size_bytes: 0 }),
                });
                const data = await response.json().catch(() => ({}));
                if (!response.ok || !data.candidates?.length) throw new Error(data.error || 'No matching repositories found');
                candidates = data.candidates;
            }
            picker.status.textContent = 'Choose a repository to inspect its GGUF files.';
            candidates.slice(0, 8).forEach(candidate => {
                const button = document.createElement('button');
                button.type = 'button';
                button.className = 'mm-card-action-note';
                button.style.textAlign = 'left';
                button.textContent = `${candidate.repoId} · ${candidate.reason || 'candidate'}${candidate.confidence != null ? ` · ${Math.round(candidate.confidence * 100)}% match` : ''}`;
                button.addEventListener('click', () => loadFiles(candidate.repoId));
                picker.results.appendChild(button);
            });
        } catch (error) {
            picker.status.textContent = `Could not find a source: ${error.message || error}`;
        } finally {
            picker.searchButton.disabled = false;
        }
    };
    picker.searchButton.addEventListener('click', search);
    picker.searchInput.addEventListener('keydown', event => {
        if (event.key === 'Enter') { event.preventDefault(); search(); }
    });
    search();
}

async function resumeModelDownload(path, source = null, overlay = null) {
    try {
        const resp = await fetch('/api/models/download/resume', {
            method: 'POST',
            headers: window.authHeaders
                ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
                : { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path, ...(source || {}) }),
        });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok || !data.ok) throw new Error(data.error || 'Resume failed');
        showToast('Download resumed', 'success');
        overlay?.remove();
        invalidateModelInventory();
        await loadModels({ refresh: true });
    } catch (error) {
        showToast(`Could not resume download: ${error.message || error}`, 'error');
    }
}

async function deleteManagedCache(repoId, name, sizeDisplay) {
    const confirmed = await _showConfirm(
        'Delete cached Hugging Face repository',
        `${repoId}\n${name}${sizeDisplay ? ` · ${sizeDisplay}` : ''}\n\nThis removes the app-managed repository, snapshots, and shared blobs from the library cache. This action cannot be undone.`,
    );
    if (!confirmed) return;
    try {
        const resp = await fetch('/api/models/library/cache', {
            method: 'DELETE',
            headers: window.authHeaders
                ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
                : { 'Content-Type': 'application/json' },
            body: JSON.stringify({ repo_id: repoId }),
        });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok || !data.ok) throw new Error(data.error || 'Cache deletion failed');
        showToast(`Deleted ${repoId} from the model library cache`, 'success');
        invalidateModelInventory();
        await loadModels({ refresh: true });
    } catch (error) {
        showToast(`Cache deletion failed: ${error.message || error}`, 'error');
    }
}

async function deleteModel(path, filename) {
    const presets = findPresetsForModel({ path });
    const extra = presets.length
        ? `\nThis will break presets that use this model:\n- ${presets.map(p => p.name || 'Unnamed preset').join('\n- ')}\n`
        : '';
    const confirmed = await _showConfirm(
        'Delete model file',
        `"${escapeHtml(filename)}"\nPath: ${escapeHtml(path)}\n\nThis will permanently remove the file from disk.${extra}\nThis action cannot be undone.`
    );
    if (!confirmed) return;

    try {
        const resp = await fetch('/api/models/file', {
            method: 'DELETE',
            headers: window.authHeaders
                ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
                : { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path }),
        });
        // A refusal comes back as 200 with ok:false — the endpoint reports a rejected path
        // as an answer, not as a server fault — so resp.ok alone would call it a success.
        const data = await resp.json().catch(() => ({ ok: false, error: 'Malformed response' }));
        if (!resp.ok || !data.ok) {
            showToast('Delete failed: ' + (data.error || resp.statusText || 'unknown'), 'error');
            return;
        }
        showToast('Model deleted', 'success');
        invalidateModelInventory();
        await loadModels({ refresh: true });
    } catch (err) {
        showToast('Delete failed: ' + err.message, 'error');
    }
}

async function deletePartialModel(path, filename) {
    const confirmed = await _showConfirm(
        'Delete incomplete download',
        `"${escapeHtml(filename)}"\nPath: ${escapeHtml(path)}\n\nThis removes the partial file and its resume metadata. This action cannot be undone.`,
    );
    if (!confirmed) return;
    try {
        const resp = await fetch('/api/models/file', {
            method: 'DELETE',
            headers: window.authHeaders
                ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
                : { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path }),
        });
        const data = await resp.json().catch(() => ({ ok: false, error: 'Malformed response' }));
        if (!resp.ok || !data.ok) throw new Error(data.error || resp.statusText || 'unknown');
        showToast('Incomplete download deleted', 'success');
        invalidateModelInventory();
        await loadModels({ refresh: true });
    } catch (error) {
        showToast(`Delete failed: ${error.message || error}`, 'error');
    }
}

// A path + optional-name prompt. `_showConfirm` is confirm-only and there is no shared
// prompt helper, so this mirrors its structure to stay visually consistent. Resolves to
// `{ path, name }` or null if dismissed.
async function _promptForLocalModelPath() {
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay';
    overlay.style.zIndex = '2000';
    overlay.style.display = 'grid';

    const dialog = document.createElement('div');
    dialog.className = 'modal';
    dialog.style.width = '480px';
    dialog.style.padding = '14px 16px';

    const titleEl = document.createElement('div');
    titleEl.style.fontSize = '15px';
    titleEl.style.fontWeight = '600';
    titleEl.style.marginBottom = '8px';
    titleEl.textContent = 'Add local model';

    const msg = document.createElement('div');
    msg.style.fontSize = '13px';
    msg.style.color = 'var(--color-text-muted)';
    msg.style.marginBottom = '10px';
    msg.textContent = 'Path to an MLX or Transformers model directory — one holding config.json, '
        + 'a tokenizer, and .safetensors weights. It is copied into the library, or hard-linked '
        + 'if it is already on the same volume, so nothing is moved or removed.';

    const pathInput = document.createElement('input');
    pathInput.type = 'text';
    pathInput.className = 'mm-lib-search-input';
    pathInput.style.width = '100%';
    pathInput.style.marginBottom = '8px';
    pathInput.placeholder = '~/Downloads/some-model-4bit';
    pathInput.setAttribute('aria-label', 'Model directory path');
    pathInput.setAttribute('autocomplete', 'off');

    const nameInput = document.createElement('input');
    nameInput.type = 'text';
    nameInput.className = 'mm-lib-search-input';
    nameInput.style.width = '100%';
    nameInput.style.marginBottom = '12px';
    nameInput.placeholder = 'Library name (optional — defaults to the directory name)';
    nameInput.setAttribute('aria-label', 'Library name');
    nameInput.setAttribute('autocomplete', 'off');

    const actions = document.createElement('div');
    actions.style.display = 'flex';
    actions.style.justifyContent = 'flex-end';
    actions.style.gap = '8px';

    const cancelBtn = document.createElement('button');
    cancelBtn.type = 'button';
    cancelBtn.className = 'btn btn-modal-cancel';
    cancelBtn.textContent = 'Cancel';

    const nextBtn = document.createElement('button');
    nextBtn.type = 'button';
    nextBtn.className = 'btn btn-modal-save';
    nextBtn.textContent = 'Check';

    actions.appendChild(cancelBtn);
    actions.appendChild(nextBtn);
    dialog.appendChild(titleEl);
    dialog.appendChild(msg);
    dialog.appendChild(pathInput);
    dialog.appendChild(nameInput);
    dialog.appendChild(actions);
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);
    pathInput.focus();

    return new Promise(resolve => {
        let decided = false;
        const finish = (value) => {
            if (decided) return;
            decided = true;
            overlay.remove();
            resolve(value);
        };
        const submit = () => {
            const path = pathInput.value.trim();
            if (!path) {
                pathInput.focus();
                return;
            }
            finish({ path, name: nameInput.value.trim() || null });
        };
        cancelBtn.addEventListener('click', () => finish(null));
        nextBtn.addEventListener('click', submit);
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) finish(null);
        });
        [pathInput, nameInput].forEach(el => el.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') submit();
            if (e.key === 'Escape') finish(null);
        }));
    });
}

// Preview first, then execute. The preview is what makes this decidable: whether the bytes
// will be copied or hard-linked changes both how long it takes and whether deleting the
// import later frees any disk, and neither is guessable from the path alone.
async function importLocalModelDirectory() {
    const asked = await _promptForLocalModelPath();
    if (!asked) return;

    const headers = window.authHeaders
        ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
        : { 'Content-Type': 'application/json' };

    let plan;
    try {
        const resp = await fetch('/api/models/library/adopt/preview', {
            method: 'POST',
            headers,
            body: JSON.stringify({ path: asked.path, name: asked.name }),
        });
        const data = await resp.json().catch(() => ({ ok: false, error: 'Malformed response' }));
        if (!resp.ok) {
            showToast('Cannot import: ' + (data.error || resp.statusText || 'unknown'), 'error');
            return;
        }
        plan = data;
    } catch (err) {
        showToast('Cannot import: ' + err.message, 'error');
        return;
    }

    const method = plan.method === 'hardlink'
        ? 'Hard-linked (instant; the bytes are shared with the source, so deleting either one '
          + 'will not free disk until both are gone)'
        : 'Copied (this can take several minutes for a large model)';
    const symlinks = plan.resolved_symlinks
        ? `\n${plan.resolved_symlinks} symlink(s) will be resolved to their real files.`
        : '';
    const warnings = (plan.warnings || []).length
        ? '\n\n' + plan.warnings.map(w => '• ' + w).join('\n')
        : '';
    const confirmed = await _showConfirm(
        'Import model into library',
        `${escapeHtml(plan.source)}\n→ ${escapeHtml(plan.destination)}\n\n`
        + `${plan.file_count} file(s), ${formatBytes(plan.bytes)}\n${method}${symlinks}${warnings}`
    );
    if (!confirmed) return;

    showToast('Importing ' + plan.slug + '…', 'info');
    try {
        const resp = await fetch('/api/models/library/adopt', {
            method: 'POST',
            headers,
            body: JSON.stringify({ path: asked.path, name: asked.name }),
        });
        const data = await resp.json().catch(() => ({ ok: false, error: 'Malformed response' }));
        if (!resp.ok || !data.ok) {
            showToast('Import failed: ' + (data.error || resp.statusText || 'unknown'), 'error');
            return;
        }
        const model = data.model || {};
        showToast(
            `Imported ${model.slug || plan.slug} — ${formatBytes(model.bytes || plan.bytes)}`,
            'success'
        );
        invalidateModelInventory();
        await loadModels({ refresh: true });
    } catch (err) {
        showToast('Import failed: ' + err.message, 'error');
    }
}

// Directory-shaped models (MLX, Transformers) go through their own endpoint rather than a
// widened /api/models/file, because that one's containment rule allows anything under $HOME.
// The server only lets a request select from the managed directories it found on disk.
async function deleteModelDirectory(path, name, sizeDisplay) {
    const presets = findPresetsForModel({ path });
    const extra = presets.length
        ? `\nThis will break presets that use this model:\n- ${presets.map(p => p.name || 'Unnamed preset').join('\n- ')}\n`
        : '';
    const size = sizeDisplay ? ` (${escapeHtml(sizeDisplay)})` : '';
    const confirmed = await _showConfirm(
        'Delete model directory',
        `"${escapeHtml(name)}"${size}\nPath: ${escapeHtml(path)}\n\nThis will permanently remove the directory and everything in it.${extra}\nThis action cannot be undone.`
    );
    if (!confirmed) return;

    try {
        const resp = await fetch('/api/models/library/directory', {
            method: 'DELETE',
            headers: window.authHeaders
                ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
                : { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path }),
        });
        const data = await resp.json().catch(() => ({ ok: false, error: 'Malformed response' }));
        if (!resp.ok || !data.ok) {
            showToast('Delete failed: ' + (data.error || resp.statusText || 'unknown'), 'error');
            return;
        }
        const removed = data.removed || {};
        // Two things the user cannot infer from having clicked Delete: an experimental cache
        // entry deletes the wrapper it lives in, not the path on the card; and hard-linked
        // bytes are shared with an import source, so the disk will not actually shrink.
        const notes = [];
        if (removed.path && removed.path !== path) notes.push(`removed ${removed.path}`);
        if (removed.shared_links) notes.push('some files were hard-linked, so disk usage may not drop');
        showToast(
            'Model directory deleted' + (notes.length ? ' — ' + notes.join('; ') : ''),
            'success'
        );
        invalidateModelInventory();
        await loadModels({ refresh: true });
    } catch (err) {
        showToast('Delete failed: ' + err.message, 'error');
    }
}

async function refreshModels() {
    const summary = document.getElementById('models-summary');
    if (summary) summary.textContent = 'Refreshing...';
    const btn = document.getElementById('models-refresh-btn');
    if (btn) btn.classList.add('spinning');
    try {
        const resp = await fetch('/api/models/refresh', {
            method: 'POST',
            headers: window.authHeaders ? window.authHeaders() : {},
        });
        const data = await resp.json();
        if (!data.ok) showToast('Model refresh failed: ' + (data.error || 'unknown'), 'error');
    } catch (err) {
        showToast('Model refresh failed: ' + err.message, 'error');
    } finally {
        if (btn) btn.classList.remove('spinning');
    }
    invalidateModelInventory();
    await loadModels({ refresh: true });
}

// ── Model tags ────────────────────────────────────────────────────────────────

async function updateModelTags(modelPath, tags) {
    try {
        const resp = await fetch('/api/models/tags', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_path: modelPath, tags }),
        });
        const data = await resp.json();
        if (!data.ok) {
            showToast('Tag update failed: ' + (data.error || 'unknown'), 'error');
            return false;
        }
        return true;
    } catch (err) {
        showToast('Tag update failed: ' + err.message, 'error');
        return false;
    }
}

async function removeModelTag(modelPath, tag) {
    const resp = await fetch('/api/models/tags', {
        headers: window.authHeaders ? window.authHeaders() : {},
    });
    if (!resp.ok) return;
    const data = await resp.json();
    const currentTags = (data.tags[modelPath] || []).filter(t => t !== tag);
    await updateModelTags(modelPath, currentTags);
    invalidateModelInventory();
    await loadModels({ refresh: true });
}

function openTagPicker(modelPath, currentTags) {
    const existing = document.getElementById('mm-tag-picker');
    if (existing) existing.remove();

    const picker = document.createElement('div');
    picker.id = 'mm-tag-picker';
    picker.className = 'mm-tag-picker';

    const title = document.createElement('div');
    title.className = 'mm-tag-picker-title';
    title.textContent = 'Tags';
    picker.appendChild(title);

    const pillsWrap = document.createElement('div');
    pillsWrap.className = 'mm-tag-picker-pills';

    const allTags = new Set([...KNOWN_TAGS, ...currentTags]);
    allTags.forEach(tag => {
        const pill = document.createElement('span');
        const has = currentTags.includes(tag);
        pill.className = 'mm-tag-pill' + (has ? ' mm-tag-pill--active' : '');
        pill.textContent = tag;
        pill.addEventListener('click', () => {
            const newTags = has
                ? currentTags.filter(t => t !== tag)
                : [...currentTags, tag];
            updateModelTags(modelPath, newTags).then(ok => {
                if (ok) {
                    invalidateModelInventory();
                    loadModels({ refresh: true });
                }
            });
        });
        pillsWrap.appendChild(pill);
    });

    picker.appendChild(pillsWrap);
    document.body.appendChild(picker);

    const close = (e) => {
        if (!picker.contains(e.target)) {
            picker.remove();
            document.removeEventListener('mousedown', close);
        }
    };
    setTimeout(() => document.addEventListener('mousedown', close), 0);
}

// ── Library toolbar ───────────────────────────────────────────────────────────

// ── Library toolbar (stable container to avoid focus loss) ────────────────────

let _toolbarInitialized = false;

function ensureLibraryToolbar() {
    if (_toolbarInitialized) return;
    _toolbarInitialized = true;
    const container = document.getElementById('mm-library-toolbar');
    if (!container) return;
    container.innerHTML = '';

    // Search input
    const wrap = document.createElement('div');
    wrap.className = 'mm-lib-search-wrap';

    const searchIcon = document.createElement('span');
    searchIcon.className = 'mm-lib-search-icon';
    searchIcon.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="11" height="11"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>';

    const input = document.createElement('input');
    input.type = 'text';
    input.id = 'mm-lib-search-input';
    input.className = 'mm-lib-search-input';
    input.placeholder = 'Search models...';
    input.value = prefs.search || '';
    input.setAttribute('autocomplete', 'off');
    input.setAttribute('aria-label', 'Search models');

    wrap.appendChild(searchIcon);
    wrap.appendChild(input);
    container.appendChild(wrap);

    let lastSearch = null;
    input.addEventListener('input', () => {
        clearTimeout(lastSearch);
        lastSearch = setTimeout(() => {
            prefs.search = input.value;
            savePrefs();
            loadModels();
        }, 250);
    });

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            clearTimeout(lastSearch);
            prefs.search = input.value;
            savePrefs();
            loadModels();
        }
    });

    // Right-side controls
    const right = document.createElement('div');
    right.className = 'mm-lib-controls';

    // Filters button
    const filtersWrap = document.createElement('div');
    filtersWrap.className = 'mm-lib-filters';

    const filtersBtn = document.createElement('button');
    filtersBtn.type = 'button';
    filtersBtn.className = 'mm-lib-btn mm-lib-btn--labeled';
    filtersBtn.id = 'mm-lib-filters-toggle';
    filtersBtn.title = 'Filter models by type, quantization, or tag';
    filtersBtn.innerHTML =
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="11" height="11"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>'
        + '<span>Filter</span>';

    const filtersPanel = document.createElement('div');
    filtersPanel.className = 'mm-lib-filters-panel';
    filtersPanel.id = 'mm-lib-filters-panel';

    filtersWrap.appendChild(filtersBtn);
    filtersWrap.appendChild(filtersPanel);
    right.appendChild(filtersWrap);

    // Toggle filters panel
    filtersBtn.addEventListener('click', () => {
        filtersPanel.classList.toggle('open');
    });

    // Sort select
    const sortWrap = document.createElement('div');
    sortWrap.className = 'mm-lib-sort-wrap';

    const sortSelect = document.createElement('select');
    sortSelect.className = 'mm-lib-sort-select';
    sortSelect.id = 'mm-lib-sort-select';
    const sortOptions = [
        { value: 'name-asc', label: 'Name A–Z' },
        { value: 'name-desc', label: 'Name Z–A' },
        { value: 'size-desc', label: 'Size (largest)' },
        { value: 'size-asc', label: 'Size (smallest)' },
        { value: 'vram-desc', label: 'VRAM (highest)' },
        { value: 'vram-asc', label: 'VRAM (lowest)' },
        { value: 'date-desc', label: 'Date (newest)' },
        { value: 'date-asc', label: 'Date (oldest)' },
    ];
    sortOptions.forEach(o => {
        const opt = document.createElement('option');
        opt.value = o.value;
        opt.textContent = o.label;
        sortSelect.appendChild(opt);
    });
    sortSelect.value = prefs.sort || 'name-asc';

    sortSelect.addEventListener('change', () => {
        prefs.sort = sortSelect.value;
        savePrefs();
        loadModels();
    });

    const sortLabel = document.createElement('span');
    sortLabel.className = 'mm-lib-sort-label';
    sortLabel.textContent = 'Sort:';
    sortWrap.appendChild(sortLabel);
    sortWrap.appendChild(sortSelect);
    right.appendChild(sortWrap);

    // View mode toggle
    const viewBtn = document.createElement('button');
    viewBtn.type = 'button';
    viewBtn.className = 'mm-lib-btn mm-lib-btn--labeled';
    viewBtn.id = 'mm-lib-view-toggle';
    viewBtn.title = prefs.viewMode === 'cards' ? 'Switch to list view' : 'Switch to cards view';
    // eslint-disable-next-line no-unsanitized/property -- static SVG, no user data
    viewBtn.innerHTML = prefs.viewMode === 'cards'
        ? ICON_LIST_VIEW + '<span>List</span>'
        : ICON_CARDS_VIEW + '<span>Cards</span>';

    viewBtn.addEventListener('click', () => {
        prefs.viewMode = prefs.viewMode === 'cards' ? 'list' : 'cards';
        viewBtn.title = prefs.viewMode === 'cards' ? 'Switch to list view' : 'Switch to cards view';
        // eslint-disable-next-line no-unsanitized/property -- static SVG, no user data
        viewBtn.innerHTML = prefs.viewMode === 'cards'
            ? ICON_LIST_VIEW + '<span>List</span>'
            : ICON_CARDS_VIEW + '<span>Cards</span>';
        savePrefs();
        loadModels();
    });

    right.appendChild(viewBtn);

    // Add local model. Sits next to the view toggle rather than on the Disk tab: the Disk tab
    // reclaims the shared Hugging Face cache, whereas this brings in a directory from anywhere
    // on the machine, which is how MLX models tend to arrive during development.
    const addBtn = document.createElement('button');
    addBtn.type = 'button';
    addBtn.className = 'mm-lib-btn mm-lib-btn--labeled';
    addBtn.id = 'mm-lib-add-local';
    addBtn.title = 'Import an MLX or Transformers model directory from elsewhere on this machine';
    addBtn.innerHTML =
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="11" height="11"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/><line x1="12" y1="11" x2="12" y2="17"/><line x1="9" y1="14" x2="15" y2="14"/></svg>'
        + '<span>Add local</span>';
    addBtn.addEventListener('click', () => { importLocalModelDirectory(); });
    right.appendChild(addBtn);

    container.appendChild(right);
}

// Called each time models list is loaded; only rebuilds dynamic filter chips.
function rebuildLibraryFilters(models) {
    const container = document.getElementById('mm-library-toolbar');
    if (!container) return;
    ensureLibraryToolbar();

    const filtersPanel = document.getElementById('mm-lib-filters-panel');
    const filtersBtn = document.getElementById('mm-lib-filters-toggle');
    if (!filtersPanel || !filtersBtn) return;

    // Update filter button active indicator
    const hasActiveFilters = !prefs.showMmproj || !prefs.showMain || !prefs.showSplit ||
        !prefs.showDraftModels || prefs.sizeMaxGb > 0 ||
        Object.values(prefs.quantFilters).some(v => v === false) ||
        !!prefs.tagFilter || !!prefs.familyFilter || !!prefs.sizeClassFilter;
    filtersBtn.classList.toggle('mm-lib-btn--active', hasActiveFilters);
    const spanEl = filtersBtn.querySelector('span');
    if (spanEl) spanEl.textContent = 'Filter' + (hasActiveFilters ? ' •' : '');

    // Clear previous filter rows
    filtersPanel.innerHTML = '';

    // Type filters
    const typeRow = document.createElement('div');
    typeRow.className = 'mm-lib-filter-row';
    const typeLabel = document.createElement('span');
    typeLabel.className = 'mm-lib-filter-label';
    typeLabel.textContent = 'Type';

    const mmprojChip = createChip('mmproj', prefs.showMmproj);
    const mainChip = createChip('Main', prefs.showMain);
    const splitChip = createChip('Split', prefs.showSplit);
    const draftChip = createChip('Draft / MTP', prefs.showDraftModels);

    mmprojChip.addEventListener('click', () => {
        prefs.showMmproj = !prefs.showMmproj;
        mmprojChip.classList.toggle('active', prefs.showMmproj);
        savePrefs(); loadModels();
    });
    mainChip.addEventListener('click', () => {
        prefs.showMain = !prefs.showMain;
        mainChip.classList.toggle('active', prefs.showMain);
        savePrefs(); loadModels();
    });
    splitChip.addEventListener('click', () => {
        prefs.showSplit = !prefs.showSplit;
        splitChip.classList.toggle('active', prefs.showSplit);
        savePrefs(); loadModels();
    });
    draftChip.addEventListener('click', () => {
        prefs.showDraftModels = !prefs.showDraftModels;
        draftChip.classList.toggle('active', prefs.showDraftModels);
        savePrefs(); loadModels();
    });

    typeRow.appendChild(typeLabel);
    typeRow.appendChild(mmprojChip);
    typeRow.appendChild(mainChip);
    typeRow.appendChild(splitChip);
    typeRow.appendChild(draftChip);
    filtersPanel.appendChild(typeRow);

    // Size filter row
    const sizeRow = document.createElement('div');
    sizeRow.className = 'mm-lib-filter-row';
    const sizeLabel = document.createElement('span');
    sizeLabel.className = 'mm-lib-filter-label';
    sizeLabel.textContent = 'Size';
    const sizeAnyChip = createChip('Any', prefs.sizeMaxGb === 0);
    const size3Chip = createChip('< 3 GB', prefs.sizeMaxGb === 3);
    const size8Chip = createChip('< 8 GB', prefs.sizeMaxGb === 8);
    const setSizeFilter = (gb) => {
        prefs.sizeMaxGb = gb;
        sizeAnyChip.classList.toggle('active', gb === 0);
        size3Chip.classList.toggle('active', gb === 3);
        size8Chip.classList.toggle('active', gb === 8);
        savePrefs(); loadModels();
    };
    sizeAnyChip.addEventListener('click', () => setSizeFilter(0));
    size3Chip.addEventListener('click', () => setSizeFilter(prefs.sizeMaxGb === 3 ? 0 : 3));
    size8Chip.addEventListener('click', () => setSizeFilter(prefs.sizeMaxGb === 8 ? 0 : 8));
    sizeRow.appendChild(sizeLabel);
    sizeRow.appendChild(sizeAnyChip);
    sizeRow.appendChild(size3Chip);
    sizeRow.appendChild(size8Chip);
    filtersPanel.appendChild(sizeRow);

    // Family filter (from classification)
    const familySet = new Set();
    models.forEach(m => {
        const f = (m.classification || {}).family || '';
        if (f && f !== 'other') familySet.add(f);
    });

    if (familySet.size > 0) {
        const familyRow = document.createElement('div');
        familyRow.className = 'mm-lib-filter-row';
        const familyLabel = document.createElement('span');
        familyLabel.className = 'mm-lib-filter-label';
        familyLabel.textContent = 'Family';

        const noneChip = createChip('All', !prefs.familyFilter);
        noneChip.addEventListener('click', () => {
            prefs.familyFilter = ''; savePrefs(); loadModels();
        });
        familyRow.appendChild(familyLabel);
        familyRow.appendChild(noneChip);

        const familyLabels = {
            qwen36: 'Qwen3.6', qwen35: 'Qwen3.5', qwen3: 'Qwen3',
            llama3: 'Llama 3.x', gemma4: 'Gemma4', gemma: 'Gemma',
            mistral: 'Mistral', exaone: 'EXAONE', heretic: 'Heretic',
        };

        familySet.forEach(fam => {
            const chip = createChip(familyLabels[fam] || fam, prefs.familyFilter === fam);
            chip.classList.toggle('active', prefs.familyFilter === fam);
            chip.addEventListener('click', () => {
                prefs.familyFilter = prefs.familyFilter === fam ? '' : fam;
                savePrefs(); loadModels();
            });
            familyRow.appendChild(chip);
        });

        filtersPanel.appendChild(familyRow);
    }

    // Size-class filter (from classification)
    const sizeClassSet = new Set();
    models.forEach(m => {
        const s = (m.classification || {}).size_class || '';
        if (s && s !== 'unknown') sizeClassSet.add(s);
    });

    if (sizeClassSet.size > 0) {
        const sizeClassRow = document.createElement('div');
        sizeClassRow.className = 'mm-lib-filter-row';
        const sizeClassLabel = document.createElement('span');
        sizeClassLabel.className = 'mm-lib-filter-label';
        sizeClassLabel.textContent = 'Size class';

        const noneChip = createChip('All', !prefs.sizeClassFilter);
        noneChip.addEventListener('click', () => {
            prefs.sizeClassFilter = ''; savePrefs(); loadModels();
        });
        sizeClassRow.appendChild(sizeClassLabel);
        sizeClassRow.appendChild(noneChip);

        ['tiny', 'small', 'medium', 'large', 'huge'].forEach(sc => {
            if (!sizeClassSet.has(sc)) return;
            const chip = createChip(sc.charAt(0).toUpperCase() + sc.slice(1), prefs.sizeClassFilter === sc);
            chip.classList.toggle('active', prefs.sizeClassFilter === sc);
            chip.addEventListener('click', () => {
                prefs.sizeClassFilter = prefs.sizeClassFilter === sc ? '' : sc;
                savePrefs(); loadModels();
            });
            sizeClassRow.appendChild(chip);
        });

        filtersPanel.appendChild(sizeClassRow);
    }

    // Quant filters (dynamic from models list)
    const quantRow = document.createElement('div');
    quantRow.className = 'mm-lib-filter-row';
    const quantLabel = document.createElement('span');
    quantLabel.className = 'mm-lib-filter-label';
    quantLabel.textContent = 'Quant';

    const quantSet = new Set();
    models.forEach(m => {
        const qt = (m.quant_type || '').toUpperCase();
        if (qt && qt !== 'UNKNOWN') quantSet.add(qt);
    });

    if (quantSet.size > 0 && quantSet.size <= 30) {
        quantSet.forEach(qt => {
            const chip = createChip(qt, prefs.quantFilters[qt] !== false);
            chip.addEventListener('click', () => {
                const active = !prefs.quantFilters[qt];
                prefs.quantFilters[qt] = active;
                chip.classList.toggle('active', active);
                savePrefs(); loadModels();
            });
            quantRow.appendChild(chip);
        });
    }

    filtersPanel.appendChild(quantRow);

    // Tag filter
    const tagRow = document.createElement('div');
    tagRow.className = 'mm-lib-filter-row';
    const tagLabel = document.createElement('span');
    tagLabel.className = 'mm-lib-filter-label';
    tagLabel.textContent = 'Tag';

    const allTags = new Set(KNOWN_TAGS);
    models.forEach(m => {
        (Array.isArray(m.tags) ? m.tags : []).forEach(t => allTags.add(t));
    });

    const allTagArr = Array.from(allTags);
    if (allTagArr.length > 0) {
        const noneChip = createChip('All', !prefs.tagFilter);
        noneChip.addEventListener('click', () => {
            prefs.tagFilter = ''; savePrefs(); loadModels();
        });
        tagRow.appendChild(noneChip);

        allTagArr.forEach(tag => {
            const chip = createChip(tag, prefs.tagFilter === tag);
            chip.classList.toggle('active', prefs.tagFilter === tag);
            chip.addEventListener('click', () => {
                prefs.tagFilter = prefs.tagFilter === tag ? '' : tag;
                savePrefs(); loadModels();
            });
            tagRow.appendChild(chip);
        });
    }

    tagRow.appendChild(tagLabel);
    filtersPanel.appendChild(tagRow);
}

function createChip(label, active) {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'mm-lib-chip' + (active ? ' active' : '');
    chip.textContent = label;
    return chip;
}

// ── Experimental Import Lab ─────────────────────────────────────────────────

let importLabInitialized = false;
let importLabAvailability = null;
let importLabPollTimer = null;
let importLabCompletedJobs = new Set();

function apiHeaders(json = false) {
    const headers = window.authHeaders ? { ...window.authHeaders() } : {};
    if (json) headers['Content-Type'] = 'application/json';
    return headers;
}

async function importLabJson(url, options = {}) {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
    return data;
}

function formatImportBytes(bytes) {
    const value = Number(bytes || 0);
    if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(1)} GiB`;
    if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(0)} MiB`;
    return `${Math.round(value / 1024)} KiB`;
}

function exactImportProfile(path) {
    const name = String(path || '').split('/').pop();
    return [
        'SmolLM2-135M-Instruct-F16.gguf',
        'SmolLM2-135M-Instruct-Q8_0.gguf',
        'SmolLM2-135M-Instruct-Q6_K.gguf',
        'SmolLM2-135M-Instruct-Q4_K_M.gguf',
    ].includes(name);
}

async function initImportLab() {
    if (!importLabInitialized) {
        importLabInitialized = true;
        document.getElementById('mm-import-analyze')?.addEventListener('click', analyzeImportSource);
        document.getElementById('mm-import-source')?.addEventListener('keydown', event => {
            if (event.key === 'Enter') analyzeImportSource();
        });
        document.getElementById('mm-import-jobs-refresh')?.addEventListener('click', loadImportJobs);
        document.getElementById('mm-import-library-use')?.addEventListener('click', () => {
            const select = document.getElementById('mm-import-library-select');
            const input = document.getElementById('mm-import-source');
            if (select?.value && input) input.value = select.value;
        });
    }
    await populateImportLibraryChoices();
    try {
        importLabAvailability = await importLabJson('/api/models/import-lab/availability', {
            headers: apiHeaders(),
        });
        const status = document.getElementById('mm-import-platform');
        if (status) {
            status.classList.toggle('unavailable', !importLabAvailability.local_execution_available);
            status.textContent = importLabAvailability.local_execution_available
                ? 'Apple Silicon ready · Local recovery available'
                : 'Local recovery unavailable · Manage models or use llama.cpp';
        }
    } catch (error) {
        showToast(`Import Lab availability failed: ${error.message}`, 'error');
    }
    await loadImportJobs();
}

async function populateImportLibraryChoices() {
    if (!inventoryCache) {
        try {
            const response = await fetch('/api/models', { headers: window.authHeaders ? window.authHeaders() : {} });
            if (response.ok) inventoryCache = await response.json();
        } catch {
            return;
        }
    }
    const select = document.getElementById('mm-import-library-select');
    if (!select) return;
    const dir = modelLibraryRoot;
    const entries = inventoryCache.filter(model => model.format === 'gguf' && model.lifecycle === 'ready');
    select.replaceChildren(new Option('Choose a GGUF from your library…', ''));
    entries.forEach(model => {
        const absolute = model.path || '';
        const relative = dir && absolute.startsWith(dir + '/') ? absolute.slice(dir.length + 1) : '';
        if (relative) {
            select.appendChild(new Option(`${model.model_name || model.filename} · ${relative}`, relative));
        }
    });
}

async function analyzeImportSource() {
    const input = document.getElementById('mm-import-source');
    const reportEl = document.getElementById('mm-import-report');
    const engineNote = document.getElementById('mm-import-engine-note');
    const path = input?.value.trim();
    if (!path || !reportEl) return;
    reportEl.replaceChildren(Object.assign(document.createElement('div'), {
        className: 'mm-import-empty-state', textContent: 'Reading bounded GGUF metadata…',
    }));
    try {
        const body = JSON.stringify({ path });
        const [report, resource] = await Promise.all([
            importLabJson('/api/models/gguf/import/compatibility/preview', {
                method: 'POST', headers: apiHeaders(true), body,
            }),
            importLabJson('/api/models/import-lab/resource-estimate', {
                method: 'POST', headers: apiHeaders(true), body,
            }),
        ]);
        renderImportReport(reportEl, report, resource, path);
        if (engineNote) {
            const recoverable = report.compatibility === 'experimental' && exactImportProfile(path);
            engineNote.textContent = recoverable
                ? 'Exact experimental profile found. llama.cpp remains available; recovery creates a separate non-launchable MLX copy.'
                : 'Recommended engine: llama.cpp. No safe MLX recovery profile is available for this model.';
        }
    } catch (error) {
        reportEl.replaceChildren(Object.assign(document.createElement('div'), {
            className: 'mm-import-empty-state', textContent: error.message,
        }));
    }
}

function appendImportFact(container, label, value) {
    const fact = document.createElement('div');
    fact.className = 'mm-import-fact';
    const labelEl = document.createElement('span');
    labelEl.textContent = label;
    const valueEl = document.createElement('strong');
    valueEl.textContent = value;
    fact.append(labelEl, valueEl);
    container.appendChild(fact);
}

function renderImportReport(container, report, resource, path) {
    container.replaceChildren();
    const verdict = document.createElement('div');
    verdict.className = 'mm-import-verdict';
    const title = document.createElement('strong');
    title.textContent = report.architecture
        ? `${report.architecture} · ${report.tensor_count || 0} tensors`
        : 'GGUF compatibility report';
    const badge = document.createElement('span');
    const compatibility = report.compatibility || 'unsupported';
    badge.className = `mm-import-verdict-badge ${compatibility}`;
    badge.textContent = compatibility;
    verdict.append(title, badge);
    container.appendChild(verdict);

    const facts = document.createElement('div');
    facts.className = 'mm-import-facts';
    appendImportFact(facts, 'Source', formatImportBytes(resource.source_bytes));
    appendImportFact(facts, 'Recovered FP16', formatImportBytes(resource.estimated_fp16_bytes));
    appendImportFact(facts, 'Disk required', formatImportBytes(resource.required_disk_bytes));
    appendImportFact(facts, 'Disk headroom', resource.disk_sufficient === true
        ? 'Available'
        : (resource.disk_sufficient === false ? 'Insufficient' : 'Unknown'));
    appendImportFact(facts, 'Memory', resource.ram_guidance?.replaceAll('_', ' ') || 'Unknown');
    appendImportFact(facts, 'Engine fallback', 'llama.cpp');
    container.appendChild(facts);

    const reasons = [
        ...(report.unsupported_reasons || []),
        ...(report.missing_profile_fields || []).map(value => `Missing profile: ${value}`),
        ...(report.missing_assets || []).map(value => `Missing asset: ${value}`),
        ...(report.warnings || []),
    ].slice(0, 8);
    if (reasons.length) {
        const list = document.createElement('ul');
        list.className = 'mm-import-reasons';
        reasons.forEach(reason => {
            const item = document.createElement('li');
            item.textContent = reason;
            list.appendChild(item);
        });
        container.appendChild(list);
    }

    const actions = document.createElement('div');
    actions.className = 'mm-import-actions';
    const start = document.createElement('button');
    start.type = 'button';
    start.className = 'mm-action-btn mm-action-btn--switch';
    start.textContent = 'Recover experimental FP16';
    const allowed = report.compatibility === 'experimental'
        && exactImportProfile(path)
        && importLabAvailability?.local_execution_available
        && resource.disk_sufficient === true;
    start.disabled = !allowed;
    start.title = allowed
        ? 'Create a separate non-launchable MLX recovery cache'
        : 'Recovery requires Apple Silicon, disk headroom, and an exact supported profile';
    start.addEventListener('click', () => startImportJob(path));
    actions.appendChild(start);
    container.appendChild(actions);
}

async function startImportJob(path) {
    try {
        await importLabJson('/api/models/import-lab/jobs', {
            method: 'POST', headers: apiHeaders(true), body: JSON.stringify({ source_path: path }),
        });
        showToast('Experimental recovery queued', 'success');
        await loadImportJobs();
    } catch (error) {
        showToast(`Recovery could not start: ${error.message}`, 'error');
    }
}

async function cancelImportJob(id) {
    try {
        await importLabJson(`/api/models/import-lab/jobs/${encodeURIComponent(id)}/cancel`, {
            method: 'POST', headers: apiHeaders(),
        });
        await loadImportJobs();
    } catch (error) {
        showToast(`Cancellation failed: ${error.message}`, 'error');
    }
}

async function forgetImportJob(id) {
    try {
        await importLabJson(`/api/models/import-lab/jobs/${encodeURIComponent(id)}`, {
            method: 'DELETE', headers: apiHeaders(),
        });
        await loadImportJobs();
    } catch (error) {
        showToast(`Cleanup failed: ${error.message}`, 'error');
    }
}

async function loadImportJobs() {
    const container = document.getElementById('mm-import-jobs-list');
    if (!container) return;
    try {
        const jobs = await importLabJson('/api/models/import-lab/jobs', { headers: apiHeaders() });
        renderImportJobs(container, jobs);
        const active = jobs.some(job => job.can_cancel);
        clearTimeout(importLabPollTimer);
        if (active) importLabPollTimer = setTimeout(loadImportJobs, 800);
        const newlyComplete = jobs.filter(job => job.state === 'complete' && !importLabCompletedJobs.has(job.id));
        if (newlyComplete.length) {
            newlyComplete.forEach(job => importLabCompletedJobs.add(job.id));
            invalidateModelInventory();
        }
    } catch (error) {
        container.replaceChildren(Object.assign(document.createElement('div'), {
            className: 'mm-import-empty-state', textContent: error.message,
        }));
    }
}

function renderImportJobs(container, jobs) {
    container.replaceChildren();
    if (!jobs.length) {
        container.appendChild(Object.assign(document.createElement('div'), {
            className: 'mm-import-empty-state', textContent: 'No recovery jobs yet.',
        }));
        return;
    }
    jobs.slice().reverse().forEach(job => {
        const card = document.createElement('article');
        card.className = 'mm-import-job';
        const top = document.createElement('div');
        top.className = 'mm-import-job-top';
        const message = document.createElement('strong');
        message.textContent = job.message;
        const state = document.createElement('span');
        state.className = 'mm-import-job-state';
        state.textContent = job.state;
        top.append(message, state);
        const progress = document.createElement('div');
        progress.className = 'mm-import-progress';
        const progressPercent = Math.max(0, Math.min(100, job.progress_percent || 0));
        progress.setAttribute('role', 'progressbar');
        progress.setAttribute('aria-label', `Recovery progress: ${job.phase || job.state}`);
        progress.setAttribute('aria-valuemin', '0');
        progress.setAttribute('aria-valuemax', '100');
        progress.setAttribute('aria-valuenow', String(progressPercent));
        const fill = document.createElement('span');
        fill.style.width = `${progressPercent}%`;
        progress.appendChild(fill);
        card.append(top, progress);
        if (Array.isArray(job.diagnostics) && job.diagnostics.length) {
            const diagnostics = document.createElement('div');
            diagnostics.className = 'mm-import-diagnostics';
            diagnostics.textContent = job.diagnostics.join('\n');
            card.appendChild(diagnostics);
        }
        const actions = document.createElement('div');
        actions.className = 'mm-import-actions';
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'mm-action-btn';
        if (job.can_cancel) {
            button.textContent = 'Cancel and clean staging';
            button.addEventListener('click', () => cancelImportJob(job.id));
        } else {
            button.textContent = 'Clear job';
            button.addEventListener('click', () => forgetImportJob(job.id));
        }
        actions.appendChild(button);
        card.appendChild(actions);
        container.appendChild(card);
    });
}

// ── Public API ────────────────────────────────────────────────────────────────

function sourceStatus(message, tone = '') {
    const status = document.getElementById('mm-sources-status');
    if (!status) return;
    status.textContent = message;
    if (tone) status.dataset.tone = tone;
    else delete status.dataset.tone;
}

async function communitySourcesRequest(url, options = {}) {
    const response = await fetch(url, {
        ...options,
        headers: {
            ...(window.authHeaders ? window.authHeaders() : {}),
            ...(options.body ? { 'Content-Type': 'application/json' } : {}),
            ...(options.headers || {}),
        },
    });
    let data;
    try {
        data = await response.json();
    } catch {
        throw new Error(`Community sources returned an invalid response (${response.status})`);
    }
    if (!response.ok || data?.ok !== true) {
        throw new Error(data?.error || `Community sources request failed (${response.status})`);
    }
    return data;
}

function roleMeta(roleId) {
    return communitySourcesState.roles.find(role => role.id === roleId) || null;
}

function appendSourceBadge(parent, text, bundled = false) {
    const badge = document.createElement('span');
    badge.className = bundled ? 'mm-source-badge mm-source-badge--bundled' : 'mm-source-badge';
    badge.textContent = text;
    parent.appendChild(badge);
}

function renderCommunitySources() {
    const list = document.getElementById('mm-sources-list');
    const entries = communitySourcesState.catalog?.entries || [];
    if (!list) return;
    list.replaceChildren();

    if (!entries.length) {
        const empty = document.createElement('div');
        empty.className = 'mm-empty';
        empty.textContent = 'No community sources yet. Add one or restore the bundled catalog.';
        list.appendChild(empty);
        return;
    }

    entries.forEach((entry, index) => {
        const card = document.createElement('article');
        card.className = 'mm-source-card';
        card.dataset.username = entry.username;

        const head = document.createElement('div');
        head.className = 'mm-source-card-head';
        const name = document.createElement('div');
        name.className = 'mm-source-name';
        name.textContent = entry.displayName || entry.username;
        head.appendChild(name);
        if (entry.bundled) appendSourceBadge(head, 'Bundled', true);

        const username = document.createElement('div');
        username.className = 'mm-source-username';
        username.textContent = `@${entry.username}`;

        const badges = document.createElement('div');
        badges.className = 'mm-source-categories';
        appendSourceBadge(badges, roleMeta(entry.role)?.label || entry.role.replaceAll('_', ' '));
        for (const secondaryRole of entry.alsoKnownFor || []) {
            appendSourceBadge(badges, roleMeta(secondaryRole)?.label || secondaryRole.replaceAll('_', ' '));
        }

        const description = document.createElement('div');
        description.className = 'mm-source-description';
        description.textContent = entry.description;

        card.append(head, username, badges, description);

        if (entry.categories?.length) {
            const categories = document.createElement('div');
            categories.className = 'mm-source-categories';
            for (const category of entry.categories) {
                const chip = document.createElement('span');
                chip.className = 'mm-source-category';
                chip.textContent = category;
                categories.appendChild(chip);
            }
            card.appendChild(categories);
        }
        if (entry.note) {
            const note = document.createElement('div');
            note.className = 'mm-source-note';
            note.textContent = entry.note;
            card.appendChild(note);
        }

        const actions = document.createElement('div');
        actions.className = 'mm-source-card-actions';
        const edit = document.createElement('button');
        edit.type = 'button';
        edit.textContent = 'Edit';
        edit.setAttribute('aria-label', `Edit ${entry.displayName || entry.username}`);
        edit.addEventListener('click', () => openCommunitySourceEditor(index));
        actions.appendChild(edit);
        if (!entry.bundled) {
            const remove = document.createElement('button');
            remove.type = 'button';
            remove.textContent = 'Remove';
            remove.setAttribute('aria-label', `Remove ${entry.displayName || entry.username}`);
            remove.addEventListener('click', () => removeCommunitySource(index));
            actions.appendChild(remove);
        }
        card.appendChild(actions);
        list.appendChild(card);
    });
}

function buildCommunityRoleControls() {
    const select = document.getElementById('mm-source-role');
    const checks = document.getElementById('mm-source-role-checks');
    if (!select || !checks) return;
    select.replaceChildren();
    checks.replaceChildren();
    for (const role of communitySourcesState.roles) {
        const option = document.createElement('option');
        option.value = role.id;
        option.textContent = role.label;
        select.appendChild(option);

        const label = document.createElement('label');
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.name = 'alsoKnownFor';
        checkbox.value = role.id;
        label.append(checkbox, document.createTextNode(role.label));
        checks.appendChild(label);
    }
    select.addEventListener('change', updateCommunityRoleHelp);
}

function updateCommunityRoleHelp() {
    const roleId = document.getElementById('mm-source-role')?.value;
    const help = document.getElementById('mm-source-role-help');
    if (help) help.textContent = roleMeta(roleId)?.description || '';
    document.querySelectorAll('#mm-source-role-checks input').forEach(input => {
        input.disabled = input.value === roleId;
        if (input.disabled) input.checked = false;
    });
}

function closeCommunitySourceEditor() {
    const editor = document.getElementById('mm-sources-editor');
    if (editor) editor.hidden = true;
    communitySourcesState.editingIndex = null;
}

function openCommunitySourceEditor(index = null) {
    const editor = document.getElementById('mm-sources-editor');
    const entry = index == null ? null : communitySourcesState.catalog?.entries?.[index];
    if (!editor || (index != null && !entry)) return;
    communitySourcesState.editingIndex = index;
    editor.reset();
    document.getElementById('mm-sources-editor-title').textContent = entry ? 'Edit source' : 'Add source';
    document.getElementById('mm-source-username').value = entry?.username || '';
    document.getElementById('mm-source-display-name').value = entry?.displayName || '';
    document.getElementById('mm-source-role').value = entry?.role || communitySourcesState.roles[0]?.id || '';
    document.getElementById('mm-source-description').value = entry?.description || '';
    document.getElementById('mm-source-categories').value = (entry?.categories || []).join(', ');
    document.getElementById('mm-source-note').value = entry?.note || '';
    const heldRoles = new Set(entry?.alsoKnownFor || []);
    document.querySelectorAll('#mm-source-role-checks input').forEach(input => {
        input.checked = heldRoles.has(input.value);
    });
    updateCommunityRoleHelp();
    editor.hidden = false;
    document.getElementById('mm-source-username').focus();
}

function communitySourceFromEditor() {
    const categories = document.getElementById('mm-source-categories').value
        .split(',')
        .map(value => value.trim())
        .filter(Boolean);
    const role = document.getElementById('mm-source-role').value;
    const alsoKnownFor = [...document.querySelectorAll('#mm-source-role-checks input:checked')]
        .map(input => input.value)
        .filter(value => value !== role);
    const note = document.getElementById('mm-source-note').value.trim();
    return {
        username: document.getElementById('mm-source-username').value.trim(),
        displayName: document.getElementById('mm-source-display-name').value.trim(),
        description: document.getElementById('mm-source-description').value.trim(),
        role,
        alsoKnownFor,
        categories: [...new Set(categories)],
        ...(note ? { note } : {}),
        bundled: false,
    };
}

function sourceRoleCombinationIsValid(candidate, editingIndex) {
    const converterRoles = new Set(['mlx_converter', 'gguf_quantizer']);
    return !(communitySourcesState.catalog?.entries || []).some((entry, index) => {
        if (index === editingIndex || entry.username.toLowerCase() !== candidate.username.toLowerCase()) return false;
        return (entry.role === 'original_author' && converterRoles.has(candidate.role))
            || (candidate.role === 'original_author' && converterRoles.has(entry.role));
    });
}

async function saveCommunitySource(event) {
    event.preventDefault();
    const editor = event.currentTarget;
    if (!editor.reportValidity()) return;
    const entry = communitySourceFromEditor();
    const index = communitySourcesState.editingIndex;
    if (!sourceRoleCombinationIsValid(entry, index)) {
        sourceStatus('An original author cannot also be registered as a converter or quantizer for the same username.', 'error');
        return;
    }

    const save = document.getElementById('mm-source-save');
    save.disabled = true;
    sourceStatus(index == null ? 'Adding source…' : 'Saving source…');
    try {
        if (index == null) {
            await communitySourcesRequest('/api/hf/community-sources/entry', {
                method: 'POST',
                body: JSON.stringify(entry),
            });
        } else {
            const existing = communitySourcesState.catalog.entries[index];
            const catalog = structuredClone(communitySourcesState.catalog);
            catalog.entries[index] = { ...entry, bundled: existing.bundled === true };
            await communitySourcesRequest('/api/hf/community-sources', {
                method: 'PUT',
                body: JSON.stringify(catalog),
            });
        }
        closeCommunitySourceEditor();
        await loadCommunitySources();
        showToast(index == null ? 'Community source added' : 'Community source updated', 'success');
    } catch (error) {
        sourceStatus(error.message, 'error');
    } finally {
        save.disabled = false;
    }
}

async function removeCommunitySource(index) {
    const entry = communitySourcesState.catalog?.entries?.[index];
    if (!entry || entry.bundled) return;
    const confirmed = await _showConfirm(
        'Remove community source?',
        `Remove ${entry.displayName || entry.username} from discovery provenance?`,
    );
    if (!confirmed) return;
    sourceStatus('Removing source…');
    try {
        const params = new URLSearchParams({ username: entry.username, role: entry.role });
        const data = await communitySourcesRequest(`/api/hf/community-sources/entry?${params}`, { method: 'DELETE' });
        if (data.removed !== true) throw new Error('The source was not removed');
        await loadCommunitySources();
        showToast('Community source removed', 'success');
    } catch (error) {
        sourceStatus(error.message, 'error');
    }
}

async function resetCommunitySources() {
    const confirmed = await _showConfirm(
        'Restore bundled sources?',
        'This replaces every catalog entry with the bundled defaults. Discovery preferences are preserved, but user-added sources and source edits are removed.',
    );
    if (!confirmed) return;
    sourceStatus('Restoring bundled sources…');
    try {
        await communitySourcesRequest('/api/hf/community-sources/reset', { method: 'POST' });
        closeCommunitySourceEditor();
        await loadCommunitySources();
        showToast('Bundled community sources restored', 'success');
    } catch (error) {
        sourceStatus(error.message, 'error');
    }
}

async function loadCommunitySources() {
    sourceStatus('Loading sources…');
    try {
        const data = await communitySourcesRequest('/api/hf/community-sources');
        communitySourcesState.catalog = data.catalog;
        communitySourcesState.roles = data.roles || [];
        buildCommunityRoleControls();
        renderCommunitySources();
        _resetCommunitySourceCatalog();
        sourceStatus(`${data.catalog?.entries?.length || 0} sources · roles and descriptions supplied by the server`);
    } catch (error) {
        sourceStatus(error.message, 'error');
        const list = document.getElementById('mm-sources-list');
        if (list) list.replaceChildren();
    }
}

function initCommunitySourcesTab() {
    if (!communitySourcesState.initialized) {
        communitySourcesState.initialized = true;
        document.getElementById('mm-sources-add')?.addEventListener('click', () => openCommunitySourceEditor());
        document.getElementById('mm-sources-reset')?.addEventListener('click', resetCommunitySources);
        document.getElementById('mm-sources-editor-close')?.addEventListener('click', closeCommunitySourceEditor);
        document.getElementById('mm-source-cancel')?.addEventListener('click', closeCommunitySourceEditor);
        document.getElementById('mm-sources-editor')?.addEventListener('submit', saveCommunitySource);
    }
    loadCommunitySources();
}

export function initModels() {
    if (initialized) return;
    initialized = true;
    startupInventoryPromise = notifyIncompleteDownloadsAtStartup();

    // Initialize toolbar structure once (search, sort, view controls)
    ensureLibraryToolbar();

    // Bind modal close buttons
    const closeBtn = document.getElementById('models-modal-close');
    if (closeBtn) closeBtn.addEventListener('click', closeModelsModal);

    const cancelBtn = document.getElementById('models-modal-cancel');
    if (cancelBtn) cancelBtn.addEventListener('click', closeModelsModal);

    const refreshBtn = document.getElementById('models-refresh-btn');
    if (refreshBtn) refreshBtn.addEventListener('click', refreshModels);

    // Tab switching
    document.querySelectorAll('#models-modal .mm-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.tab;
            const summary = document.getElementById('models-summary');
            if (summary) summary.style.visibility = target === 'library' ? '' : 'hidden';
            document.querySelectorAll('#models-modal .mm-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('#models-modal .mm-tab-panel').forEach(p => p.classList.remove('active'));
            tab.classList.add('active');
            const panel = document.querySelector('#models-modal .mm-tab-panel[data-tab="' + target + '"]');
            if (panel) panel.classList.add('active');
            if (target === 'download') {
                initHfDownloadTab();
            } else if (target === 'import-lab') {
                initImportLab();
            } else if (target === 'disk') {
                initDiskTab();
            } else if (target === 'sources') {
                initCommunitySourcesTab();
            } else if (target === 'library' && !inventoryCache) {
                loadModels();
            }
        });
    });

    // Configure settings link
    const settingsLink = document.getElementById('models-open-settings-link');
    if (settingsLink) {
        settingsLink.addEventListener('click', () => {
            closeModelsModal();
            Router.navigate('/settings#models');
        });
    }

    // Modal overlay click to close
    const modal = document.getElementById('models-modal');
    if (modal) {
        modal.addEventListener('click', e => {
            if (e.target === modal) closeModelsModal();
        });
    }
}

// ── HF Download tab (inside models modal) ─────────────────────────────────────

async function initHfDownloadTab() {
    if (hfState.initialized) return;
    hfState.initialized = true;

    const searchInput = document.getElementById('mm-hf-search-input');
    const discoverPills = document.getElementById('mm-hf-discover-pills');
    const quickpicks = document.getElementById('mm-hf-quickpicks');
    const resultsContainer = document.getElementById('mm-hf-search-results');
    const filelistContainer = document.getElementById('mm-hf-file-list');
    const downloadPanel = document.getElementById('mm-hf-download-panel');

    if (!searchInput || !resultsContainer || !filelistContainer || !downloadPanel) return;

    // Fetch hardware info
    await fetchGpuVram();
    await fetchSystemRam();

    // Helpers to build search params while preserving active filters
    const buildSearchParams = () => {
        const typedQuery = (searchInput.value || '').trim();
        const query = typedQuery || hfState.activeDiscoverQuery || '';
        const author = (typedQuery ? hfState.activeAuthor : (hfState.activeAuthor || null));
        return { query: query || undefined, author: author || undefined };
    };

    // Render discover pills
    hfRenderDiscoverPills({
        container: discoverPills,
        quickpicksContainer: quickpicks,
        onPillClick: (cat) => {
            // Track active discover query so sort changes still work
            hfState.activeDiscoverQuery = cat.params.query || null;
            hfState.activeAuthor = null;
            hfSearch({
                query: cat.params.query,
                mlxActive: hfState.discoveryScopeMlx,
                ggufActive: hfState.discoveryScopeGguf,
                hfSort: hfState.discoverySort,
                limit: cat.params.limit || 20,
                container: resultsContainer,
                filelistContainer,
                quickpicksContainer: quickpicks,
                discoverPillsContainerId: 'mm-hf-discover-pills',
                onOpenCardPanel: openCardPanel,
                onSelectModel: (m) => onHfModelSelected(m, filelistContainer, downloadPanel),
                quantsOnly: hfState.discoveryQuantsOnly,
                vramGb: cachedVram > 0 ? cachedVram / (1024 ** 3) : 0,
            });
        },
    });

    // Load quick-picks
    hfLoadQuickPicks({
        container: quickpicks,
        discoverPillsContainerId: 'mm-hf-discover-pills',
        onAuthorClick: (author) => {
            hfState.activeAuthor = author;
            hfState.activeDiscoverQuery = null;
            hfSearch({
                query: '',
                author,
                mlxActive: hfState.discoveryScopeMlx,
                ggufActive: hfState.discoveryScopeGguf,
                hfSort: hfState.discoverySort,
                limit: 20,
                container: resultsContainer,
                filelistContainer,
                quickpicksContainer: quickpicks,
                discoverPillsContainerId: 'mm-hf-discover-pills',
                onOpenCardPanel: openCardPanel,
                onSelectModel: (m) => onHfModelSelected(m, filelistContainer, downloadPanel),
                quantsOnly: hfState.discoveryQuantsOnly,
                vramGb: cachedVram > 0 ? cachedVram / (1024 ** 3) : 0,
            });
        },
    });

    // Search on input (debounced)
    let searchTimer = null;
    const doSearch = () => {
        const { query, author, sort } = buildSearchParams();
        console.log('[HF-SEARCH] mlxActive:', hfState.discoveryScopeMlx, 'ggufActive:', hfState.discoveryScopeGguf, 'query:', query);
        hfSearch({
            query,
            author,
            sort,
            mlxActive: hfState.discoveryScopeMlx,
            ggufActive: hfState.discoveryScopeGguf,
            hfSort: hfState.discoverySort,
            limit: 20,
            container: resultsContainer,
            filelistContainer,
            quickpicksContainer: quickpicks,
            discoverPillsContainerId: 'mm-hf-discover-pills',
            onOpenCardPanel: openCardPanel,
            onSelectModel: (m) => onHfModelSelected(m, filelistContainer, downloadPanel),
            quantsOnly: hfState.discoveryQuantsOnly,
            vramGb: cachedVram > 0 ? cachedVram / (1024 ** 3) : 0,
        });
    };

    searchInput.addEventListener('input', () => {
        clearTimeout(searchTimer);
        searchTimer = setTimeout(doSearch, 3000);
    });

    searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            clearTimeout(searchTimer);
            doSearch();
        }
    });

    // Phase 8B1: create discovery scope selector and sort selector
    // Platform-smart defaults: macOS → MLX+GGUF; Win/Linux → GGUF
    const isMac = typeof navigator !== 'undefined' && navigator.platform?.includes('Mac');
    const defaultMlx = isMac;
    const defaultGguf = true;

    hfState.discoveryScopeMlx = defaultMlx;
    hfState.discoveryScopeGguf = defaultGguf;

    const scopeContainer = document.getElementById('mm-hf-scope-container');
    const sortContainer = document.getElementById('mm-hf-sort-container');

    if (scopeContainer) {
        // Set initial state via dataset for additive toggles
        scopeContainer.dataset.hfScopeMlx = defaultMlx ? '1' : '';
        scopeContainer.dataset.hfScopeGguf = '1';
        scopeContainer.dataset.hfScopeAll = '';

        hfCreateScopeSelector({
            container: scopeContainer,
            onChange: (mlxActive, ggufActive, allActive) => {
                hfState.discoveryScopeMlx = mlxActive || allActive;
                hfState.discoveryScopeGguf = ggufActive || allActive;
                // A prior selection's panels (mmproj, download, quant advisor, VRAM)
                // can be stale once the scope no longer includes that model's format —
                // re-searching doesn't re-select anything, so nothing else clears them.
                // Also clear the selection itself so any in-flight async lookup for the
                // old repo (e.g. detectMmprojCompanion's fetch) invalidates on resolve
                // instead of re-showing UI for a model that's no longer selected.
                hfState.selectedRepoId = '';
                hfState.modelFormat = 'unknown';
                hfHideDownloadPanel(downloadPanel);
                hideQuantAdvisor();
                hideMmprojSection();
                hideVramPanel();
                hideCtxTrainWarning();
                hideHardwareInfoCard();
                clearTimeout(searchTimer);
                searchTimer = setTimeout(doSearch, 200);
            },
        });
    }

    if (sortContainer) {
        hfCreateSortSelector({
            container: sortContainer,
            defaultSort: HF_SORT.LAST_UPDATED,
            onChange: (sort) => {
                hfState.discoverySort = sort;
                clearTimeout(searchTimer);
                searchTimer = setTimeout(doSearch, 200);
            },
        });
    }

    const quantsOnlyCheckbox = document.getElementById('mm-hf-quants-only');
    if (quantsOnlyCheckbox) {
        quantsOnlyCheckbox.addEventListener('change', () => {
            hfState.discoveryQuantsOnly = quantsOnlyCheckbox.checked;
            clearTimeout(searchTimer);
            searchTimer = setTimeout(doSearch, 200);
        });
    }

    // Context size pills
    const ctxPills = document.getElementById('mm-vram-ctx-pills');
    if (ctxPills) {
        ctxPills.addEventListener('click', (e) => {
            const pill = e.target.closest('.vram-ctx-pill');
            if (!pill) return;
            ctxPills.querySelectorAll('.vram-ctx-pill').forEach(p => p.classList.remove('active'));
            pill.classList.add('active');
            const newCtx = parseInt(pill.dataset.ctx, 10);
            hfState.previewCtx = newCtx;
            // Item 14: a context change invalidates the quant comparison too — the
            // advisor persists after a file is picked (it's a comparison tool),
            // so it must not go stale either way.
            if (hfState.paramB > 0) triggerQuantAdvisor();
            // Update VRAM for both GGUF files and MLX models
            const isMlx = hfState.modelFormat === 'mlx';
            if (hfState.selectedFile) {
                scheduleVramUpdate(hfState.selectedFile);
            } else if (isMlx && hfState.modelBytes > 0) {
                // MLX models: recalculate VRAM with new context size
                // Show brief loading state since MLX needs to fetch config.json from HF
                const panel = document.getElementById('mm-vram-panel');
                if (panel) panel.classList.add('vram-panel-loading');
                scheduleVramUpdate({ size: hfState.modelBytes });
            } else if (isMlx && hfState.paramB > 0) {
                // MLX models without size: estimate from param count
                const bpw = 4.85; // default 4-bit
                const estBytes = Math.round(hfState.paramB * 1e9 * bpw / 8);
                scheduleVramUpdate({ size: estBytes });
            }
        });
    }

    // Settings link from warning
    const settingsBtn = document.getElementById('mm-hf-dlp-open-settings');
    if (settingsBtn) {
        settingsBtn.addEventListener('click', () => {
            Router.navigate('/settings');
        });
    }
}

async function onHfModelSelected(model, filelistContainer, downloadPanel) {
    // Handle both raw model objects and selection payloads
    const repoId = model.repoId || model.id || '';
    const paramB = model.param_b || model._raw?.param_b || 0;
    const modelFormat = model.format || model._raw?.format || 'unknown';

    hfState.selectedRepoId = repoId;
    hfState.paramB = paramB;
    hfState.modelFormat = modelFormat;
    hfState.mmprojFiles = [];
    hfState.mmprojPath = '';
    hfState.mmprojRepoId = '';
    hfState.mmprojBytes = 0;
    hfState.availableFiles = modelFormat === 'mlx' && Array.isArray(model.siblings)
        ? model.siblings.filter(s => s.size > 0).map(s => ({ label: s.label, size: s.size }))
        : [];
    hfState.mlxSiblings = modelFormat === 'mlx' && Array.isArray(model.siblings) ? model.siblings : [];
    hfHideDownloadPanel(downloadPanel);
    hideQuantAdvisor();
    hideMmprojSection();
    hideVramPanel();
    hideCtxTrainWarning();
    hideHardwareInfoCard();

    // Hide file list container (GGUF only)
    filelistContainer.innerHTML = '';
    filelistContainer.classList.remove('visible');

    // Show selected model info
    showSelectedModel(repoId, model);

    // Quant advisor stays useful even after a specific quant is picked — it's a
    // comparison tool, not just a pre-download prompt — so trigger it here,
    // before any format/file-specific branching below early-returns.
    if (paramB > 0) triggerQuantAdvisor();

    // If clicked from inline file list, _file is set
    if (model._file) {
        await onHfFileSelected(model._file, repoId, downloadPanel);
        return;
    }

    // MLX repos: treat repo itself as the model — use model_size_bytes if available
    if (modelFormat === 'mlx') {
        let modelBytes = model.model_size_bytes || 0;
        // Update selected model display
        const nameEl = document.getElementById('mm-selected-model-name');
        const metaEl = document.getElementById('mm-selected-model-meta');
        if (nameEl) nameEl.textContent = repoId;
        if (metaEl) {
            const parts = [];
            if (model.param_b > 0) parts.push(formatParams(model.param_b));
            if (modelBytes > 0) parts.push(formatBytes(modelBytes));
            if (model.quant_label) parts.push(model.quant_label);
            metaEl.textContent = parts.join(' · ');
        }
        // If no model_size_bytes from search, fetch from tree API
        if (modelBytes === 0) {
            try {
                const headers = window.authHeaders ? window.authHeaders() : {};
                const resp = await fetch('/api/hf/files', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', ...headers },
                    body: JSON.stringify({ repo_id: repoId, format: 'mlx' }),
                });
                if (resp.ok) {
                    const data = await resp.json();
                    if (data.files && data.files.length > 0) {
                        // Sum file sizes from tree API response
                        modelBytes = data.files.reduce((sum, f) => sum + (f.size || f.bytes || 0), 0);
                        hfState.modelBytes = modelBytes;
                        hfState.mlxModelBytes = modelBytes;
                        if (metaEl) {
                            const parts = [];
                            if (model.param_b > 0) parts.push(formatParams(model.param_b));
                            parts.push(formatBytes(modelBytes));
                            if (model.quant_label) parts.push(model.quant_label);
                            metaEl.textContent = parts.join(' · ');
                        }
                    }
                }
            } catch { /* non-fatal */ }
        }
        // Show download panel with MLX-specific info
         if (modelBytes > 0) {
            hfState.modelBytes = modelBytes;
            hfState.mlxModelBytes = modelBytes;
            await hfShowDownloadPanel(downloadPanel, repoId);

            // Update file name display for MLX
            const fileEl = document.getElementById('mm-hf-dlp-file-name');
            if (fileEl) fileEl.textContent = repoId;

            // Enable download button (downloads the MLX repo snapshot into the
            // models cache via the Rapid-MLX resolver, then offers the wizard)
            const btn = document.getElementById('mm-hf-dlp-download-btn');
            if (btn) {
                btn.disabled = false;
                if (!btn.dataset.mlxHandler) {
                    btn.dataset.mlxHandler = '1';
                    // Replace button to remove existing listeners
                    const newBtn = btn.cloneNode(true);
                    btn.parentNode.replaceChild(newBtn, btn);
                    newBtn.addEventListener('click', async () => {
                        if (newBtn.dataset.dlBusy === '1') return;
                        if (newBtn.dataset.dlDone === '1') {
                            // Weights are local now; the wizard configures and launches.
                            if (typeof openSpawnWizard === 'function') {
                                openSpawnWizard({
                                    templatePreset: {
                                        backend: 'rapid_mlx',
                                        rapid_mlx: { model_source: { kind: 'hugging_face_repo', repo_id: repoId } }
                                    }
                                });
                            }
                            return;
                        }
                        newBtn.dataset.dlBusy = '1';
                        newBtn.disabled = true;
                        newBtn.textContent = 'Starting download\u2026';
                        const fmtGb = (b) => (b / (1024 ** 3)).toFixed(1) + ' GB';
                        try {
                            const headers = window.authHeaders
                                ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
                                : { 'Content-Type': 'application/json' };
                            const resp = await fetch('/api/models/downloads', {
                                method: 'POST',
                                headers,
                                body: JSON.stringify({ repo_id: repoId, engine: 'rapid-mlx' }),
                            });
                            const data = await resp.json().catch(() => ({}));
                            if (!resp.ok || !data.ok) {
                                showToast(data.error || 'Download failed to start.', 'error');
                                newBtn.dataset.dlBusy = '';
                                newBtn.disabled = false;
                                newBtn.textContent = 'Download to models folder';
                                return;
                            }
                            showToast('Download started: ' + repoId, 'info');
                            newBtn.textContent = 'Downloading\u2026';
                            const jobId = data.job_id;
                            let stalledToasted = false;
                            const poll = async () => {
                                try {
                                    const r = await fetch(`/api/models/downloads/${encodeURIComponent(jobId)}`, { headers: window.authHeaders ? window.authHeaders() : {} });
                                    const d = await r.json().catch(() => ({}));
                                    const job = d.job || {};
                                    if (job.bytes_total > 0 && job.state === 'running') {
                                        const pct = Math.min(100, Math.round((job.bytes_done / job.bytes_total) * 100));
                                        newBtn.textContent = `Downloading ${pct}% (${fmtGb(job.bytes_done)} / ${fmtGb(job.bytes_total)})`;
                                        if (job.stalled && !stalledToasted) {
                                            stalledToasted = true;
                                            showToast(`Download of ${repoId} stalled \u2014 resuming automatically (finished files are kept).`, 'warning');
                                        }
                                        if (!job.stalled) stalledToasted = false;
                                    } else if (job.state === 'running') {
                                        newBtn.textContent = 'Downloading\u2026';
                                    }
                                    if (job.state === 'complete') {
                                        showToast('Downloaded: ' + repoId, 'success');
                                        invalidateModelInventory();
                                        loadModels({ refresh: true });
                                        newBtn.dataset.dlBusy = '';
                                        newBtn.disabled = false;
                                        newBtn.textContent = 'Open in Spawn Wizard';
                                        newBtn.dataset.dlDone = '1';
                                        return;
                                    }
                                    if (job.state === 'failed') {
                                        showToast('Download failed: ' + (job.error || 'unknown error'), 'error');
                                        newBtn.dataset.dlBusy = '';
                                        newBtn.disabled = false;
                                        newBtn.textContent = 'Retry download';
                                        return;
                                    }
                                    setTimeout(poll, 2000);
                                } catch {
                                    setTimeout(poll, 4000);
                                }
                            };
                            poll();
                        } catch (err) {
                            showToast('Download request failed: ' + (err.message || err), 'error');
                            newBtn.dataset.dlBusy = '';
                            newBtn.disabled = false;
                            newBtn.textContent = 'Download to models folder';
                        }
                    });
                }
            }
        }
        // Show VRAM estimate
        if (paramB > 0 || modelBytes > 0) scheduleVramUpdate({ size: modelBytes });
        return;
    }

    // GGUF repos: list files (quant advisor already triggered above).
    await hfListFiles({
        repoId,
        container: filelistContainer,
        vramGb: cachedVram > 0 ? cachedVram / (1024 * 1024 * 1024) : 0,
        onSelectFile: (file, repoId) => onHfFileSelected(file, repoId, downloadPanel),
        onFilesLoaded: (files) => { hfState.availableFiles = files; },
    });
}

async function onHfFileSelected(file, repoId, downloadPanel) {
    hfState.selectedFile = file;
    hfState.modelBytes = Number(file.size) || 0;

    // Update selected model display with file info
    const nameEl = document.getElementById('mm-selected-model-name');
    const metaEl = document.getElementById('mm-selected-model-meta');
    if (nameEl) nameEl.textContent = (file.path || file.name || '').split('/').pop() || '';
    if (metaEl) {
        const parts = [];
        if (repoId) parts.push(repoId);
        if (file.size) parts.push(formatBytes(file.size));
        if (file.label) parts.push(file.label);
        metaEl.textContent = parts.join(' · ');
    }

    // Quant advisor stays visible after a file is picked — it remains a useful
    // comparison against the other available quants, not just a pre-selection aid.

    // Show download panel
    await hfShowDownloadPanel(downloadPanel, file.path || file.name || '');

    // Update VRAM panel with live estimate
    if (hfState.paramB > 0 || hfState.modelBytes > 0) {
        scheduleVramUpdate(file);
    }

    // Check for mmproj companion in this repo
    await detectMmprojCompanion(repoId);

    // Wire download button with double-click guard
    const newBtn = document.getElementById('mm-hf-dlp-download-btn');
    if (newBtn) {
        newBtn.onclick = null;
    }
    if (newBtn) {
        newBtn.disabled = false;
        let clickLocked = false;
        newBtn.onclick = async () => {
            if (clickLocked) return;
            clickLocked = true;
            const modelFilePath = file.path || file.name;
            const started = await hfStartDownload({
                repoId,
                filePath: modelFilePath,
                panelEl: downloadPanel,
                onComplete: (downloadId, localPath) => {
                    hfState.currentDownloadIds.add(downloadId);
                    // Refresh library tab
                    invalidateModelInventory();
                    loadModels({ refresh: true });
                },
                onValidationError: (msg) => {
                    // hfStartDownload now shows its own toast for most cases.
                    // This handler is a fallback for non-standard messages.
                    const e = (msg || '').toLowerCase();
                    if (!e.includes('already downloading') &&
                        !e.includes('already exists') &&
                        !e.includes('too many downloads') &&
                        !e.includes('download started')) {
                        showToast(msg || 'Download failed', 'error');
                    }
                },
            });
            if (!started || !hfState.mmprojPath) {
                clickLocked = false;
                return;
            }

            const companion = await hfStartCompanionDownload({
                repoId: hfState.mmprojRepoId || repoId,
                filePath: hfState.mmprojPath,
                saveAs: deriveMmprojSaveName(modelFilePath, hfState.mmprojPath),
            });
            if (!companion) {
                showToast('Model download started, but the mmproj download could not start.', 'error');
            }
            // Unlock only after progress is actively showing (polling started).
            // hfStartDownload sets state to 'progress' internally; keep button
            // disabled visually via the progress panel, so clickLocked is only
            // to prevent a second click before that transition.
            clickLocked = false;
        };
    }

    // Wire cancel button
    const newCancel = document.getElementById('mm-hf-dlp-cancel-btn');
    if (newCancel) {
        newCancel.onclick = null;
    }
    if (newCancel) {
        newCancel.onclick = () => {
            if (hfState.currentDownloadIds.size > 0) {
                const downloadId = [...hfState.currentDownloadIds][0];
                hfCancelDownload({
                    downloadId,
                    panelEl: downloadPanel,
                });
                hfState.currentDownloadIds.delete(downloadId);
            }
        };
    }
}

// ── Selected model display (wizard-style) ────────────────────────────────────

function showSelectedModel(repoId, model) {
    const el = document.getElementById('mm-selected-model');
    const nameEl = document.getElementById('mm-selected-model-name');
    const metaEl = document.getElementById('mm-selected-model-meta');
    if (!el) return;

    el.style.display = '';
    if (nameEl) nameEl.textContent = repoId;
    if (metaEl) {
        const parts = [];
        if (model.param_b > 0) parts.push(formatParams(model.param_b));
        if (model.downloads > 0) parts.push(model.downloads + ' downloads');
        metaEl.textContent = parts.join(' · ');
    }
}

// ── Quant advisor (wizard-style) ─────────────────────────────────────────────

let quantAdvisorDebounce = null;

function triggerQuantAdvisor() {
    if (quantAdvisorDebounce) clearTimeout(quantAdvisorDebounce);
    quantAdvisorDebounce = setTimeout(loadQuantAdvisor, 200);
}

function hideQuantAdvisor() {
    const el = document.getElementById('mm-quant-advisor');
    if (el) el.style.display = 'none';
}

function hideHardwareInfoCard() {
    const el = document.getElementById('mm-hw-info');
    if (el) el.style.display = 'none';
}

async function loadQuantAdvisor() {
    const paramB = hfState.paramB;
    if (!paramB || paramB <= 0) return;

    // GGUF repos get the advisor from the parameter count (or the repo's file list). An MLX
    // repo ships one fixed quantization, so the comparison is only meaningful across the
    // sibling conversions of the same model; with fewer than two there is nothing to rank.
    const isMlx = hfState.modelFormat === 'mlx';
    const mlxFiles = isMlx
        ? (hfState.mlxSiblings || [])
            .filter(s => s.size > 0)
            .map(s => ({ name: s.label, size_bytes: s.size }))
        : [];
    if (isMlx && mlxFiles.length < 2) {
        hideQuantAdvisor();
        return;
    }

    // Unified memory handling: use current_safe_availability_bytes instead of total
    let availVram = cachedVram || 0;
    if (cachedUnified) {
        const snapshot = await fetchMemoryAvailability();
        if (snapshot && snapshot.current_safe_availability_bytes > 0) {
            availVram = snapshot.current_safe_availability_bytes;
        }
    }
    if (!availVram) return;

    try {
        const headers = window.authHeaders
            ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
            : { 'Content-Type': 'application/json' };

        // Item 9/14: pass backend/unified-memory/concurrency so the comparison
        // reflects what will actually launch, not a generic llama.cpp/8k guess.
        const body = {
            param_b: paramB,
            model_name: hfState.selectedRepoId || '',
            available_vram_bytes: availVram,
            backend: isMlx ? 'rapid_mlx' : 'llama_cpp',
            is_unified_memory: isMlx || cachedUnified,
            use_case: 'general',
            parallel_slots: 1,
        };
        // MLX: rank the sibling conversions by their real repo sizes.
        if (isMlx) body.available_files = mlxFiles;

        const resp = await fetch('/api/vram/quant-compare', { method: 'POST', headers, body: JSON.stringify(body) });
        if (!resp.ok) return;
        const data = await resp.json();
        if (!data.ok || !data.quants) return;

        renderQuantAdvisor(data.quants, availVram);
    } catch {
        // ignore
    }
}

function renderQuantAdvisor(quants, availVram) {
    const isMlxTable = hfState.modelFormat === 'mlx';
    const panel = document.getElementById('mm-quant-advisor');
    const tableEl = document.getElementById('mm-quant-advisor-table');
    const subtitleEl = document.getElementById('mm-quant-advisor-subtitle');
    if (!panel || !tableEl) return;
    if (!quants || quants.length === 0) { panel.style.display = 'none'; return; }

    // Filter to only quants that exist in the repo's files
    const fileLabels = new Set();
    if (hfState.availableFiles && hfState.availableFiles.length > 0) {
        hfState.availableFiles.forEach(f => {
            const label = f.label || '';
            if (label) fileLabels.add(label);
        });
    }
    const availableQuants = fileLabels.size > 0
        ? quants.filter(q => fileLabels.has(q.label))
        : quants;

    if (!availableQuants || availableQuants.length === 0) { panel.style.display = 'none'; return; }

    const availGb = Math.round(availVram / (1024 ** 3));
    // Item 14: recompute against the same context target the VRAM panel uses,
    // instead of a hard-coded 16k/q8 assumption baked into the table copy.
    const desiredCtx = hfState.previewCtx || 0;
    const annotateCtx = desiredCtx > 8192;
    const qualityRecQuant = availableQuants.find(q => q.recommended && q.fits_vram);
    const qualityRecFitsCtx = !annotateCtx || (qualityRecQuant && qualityRecQuant.max_ctx_q8 >= desiredCtx);
    const ctxFitQuant = annotateCtx
        ? availableQuants.find(q => q.fits_vram && q.max_ctx_q8 >= desiredCtx)
        : null;

    let subtitle = `Estimated VRAM available: ${availGb} GB`;
    if (annotateCtx) subtitle += ` \u00b7 Context target: ${formatCtx(desiredCtx)}`;
    if (subtitleEl) subtitleEl.textContent = subtitle;

    // MLX: the row for the variant the user picked, so picking another visibly moves the focus.
    const selectedMlxLabel = isMlxTable
        ? ((hfState.mlxSiblings || []).find(s => s.repoId === hfState.selectedRepoId)?.label || '')
        : '';
    // A variant whose weights alone exceed free memory has no context to report; say so rather
    // than leaving a bare dash that reads like missing data.
    const setNoCtx = (td, q) => {
        td.classList.add('qa-ctx-na');
        if (isMlxTable && !q.fits_vram) {
            td.textContent = 'won\u2019t fit';
            td.title = `${q.model_size_gb.toFixed(1)} GB of weights plus overhead exceeds the ${availGb} GB currently free`;
        } else {
            td.textContent = '\u2014';
        }
    };

    const table = document.createElement('table');
    table.className = 'qa-table';

    const thead = table.createTHead();
    const hrow = thead.insertRow();
    const headers = isMlxTable
        ? ['', 'Quant', 'Size', 'Max ctx (bf16 KV)', 'Max ctx (int8 KV)', 'Quality']
        : ['', 'Quant', 'Size', 'Max ctx (q8_0 KV)', 'Max ctx (q4_0 KV)', 'Quality'];
    headers.forEach(h => {
        const th = document.createElement('th');
        th.textContent = h;
        hrow.appendChild(th);
    });

    const tbody = table.createTBody();
    for (const q of availableQuants) {
        const tr = tbody.insertRow();
        if (q.recommended) tr.className = 'qa-row-rec';
        if (!q.fits_vram) tr.className = (tr.className + ' qa-row-nofit').trim();

        // Fit dot
        const dotTd = tr.insertCell();
        const dot = document.createElement('span');
        dot.className = 'qa-fit-dot ' + (q.fits_vram ? 'fits' : 'nofit');
        dotTd.appendChild(dot);

        // Quant name + rec badge
        const nameTd = tr.insertCell();
        const nameSpan = document.createElement('span');
        nameSpan.style.fontWeight = '600';
        nameSpan.textContent = q.label;
        nameTd.appendChild(nameSpan);
        if (q.recommended) {
            const badge = document.createElement('span');
            badge.className = 'qa-badge-rec';
            badge.textContent = (annotateCtx && !qualityRecFitsCtx) ? '\u2605 Quality' : '\u2605 Rec';
            badge.style.marginLeft = '6px';
            nameTd.appendChild(badge);
        }
        if (annotateCtx && ctxFitQuant && q.label === ctxFitQuant.label && !qualityRecFitsCtx) {
            const ctxBadge = document.createElement('span');
            ctxBadge.className = 'qa-badge-ctx';
            ctxBadge.textContent = `\u2713 fits ${formatCtx(desiredCtx)}`;
            ctxBadge.style.marginLeft = '6px';
            nameTd.appendChild(ctxBadge);
        }
        if (q.is_imatrix) {
            const im = document.createElement('span');
            im.style.cssText = 'margin-left:4px; font-size:10px; color:var(--color-text-muted);';
            im.textContent = 'imatrix';
            nameTd.appendChild(im);
        }

        if (selectedMlxLabel && q.label === selectedMlxLabel) tr.classList.add('qa-row-selected');

        // Size
        const sizeTd = tr.insertCell();
        sizeTd.textContent = q.model_size_gb.toFixed(1) + ' GB';
        sizeTd.style.color = 'var(--color-text-muted)';

        // Max ctx at unquantized bf16 KV (MLX only): the baseline int8 is compared against.
        if (isMlxTable) {
            const ctxF16Td = tr.insertCell();
            ctxF16Td.className = 'qa-ctx';
            if (q.max_ctx_f16 > 0) {
                ctxF16Td.textContent = formatCtx(q.max_ctx_f16);
                const underF16 = annotateCtx && q.max_ctx_f16 < desiredCtx;
                ctxF16Td.classList.add(underF16 ? 'qa-ctx-under' : 'qa-ctx-q8');
                if (underF16) ctxF16Td.title = `Max ${formatCtx(q.max_ctx_f16)} \u2014 below your ${formatCtx(desiredCtx)} target`;
            } else {
                setNoCtx(ctxF16Td, q);
            }
        }

        // Max ctx q8_0 \u2014 warn if below context target
        const ctxQ8Td = tr.insertCell();
        ctxQ8Td.className = 'qa-ctx';
        if (q.max_ctx_q8 > 0) {
            ctxQ8Td.textContent = formatCtx(q.max_ctx_q8);
            const underTarget = annotateCtx && q.max_ctx_q8 < desiredCtx;
            ctxQ8Td.classList.add(underTarget ? 'qa-ctx-under' : 'qa-ctx-q8');
            if (underTarget) ctxQ8Td.title = `Max ${formatCtx(q.max_ctx_q8)} \u2014 below your ${formatCtx(desiredCtx)} target`;
        } else {
            setNoCtx(ctxQ8Td, q);
        }

        // Max ctx q4_0 (llama.cpp only: Rapid-MLX thinking mode pins the KV cache to int8)
        if (!isMlxTable) {
            const ctxQ4Td = tr.insertCell();
            ctxQ4Td.className = 'qa-ctx';
            if (q.max_ctx_q4 > 0) {
                ctxQ4Td.textContent = formatCtx(q.max_ctx_q4);
                ctxQ4Td.classList.add('qa-ctx-q4');
            } else {
                ctxQ4Td.textContent = '\u2014'; ctxQ4Td.classList.add('qa-ctx-na');
            }
        }

        // Quality badge
        const qualTd = tr.insertCell();
        const qualBadge = document.createElement('span');
        const qClass = 'qa-quality-' + (q.quality || '').toLowerCase();
        qualBadge.className = `qa-quality-badge ${qClass}`;
        qualBadge.textContent = q.quality_label || q.quality;
        qualTd.appendChild(qualBadge);
    }

    tableEl.innerHTML = '';
    tableEl.appendChild(table);
    panel.style.display = '';
}

// ── VRAM estimation (wizard-style) ───────────────────────────────────────────

let vramDebounce = null;

function scheduleVramUpdate(file) {
    if (vramDebounce) clearTimeout(vramDebounce);
    vramDebounce = setTimeout(() => updateVramDisplay(file), 250);
}

function hideVramPanel() {
    const el = document.getElementById('mm-vram-panel');
    if (el) el.style.display = 'none';
}

async function fetchGpuVram(retriesLeft = 30, background = false) {
    try {
        const headers = window.authHeaders ? window.authHeaders() : {};
        const resp = await fetch('/metrics/gpu', { headers });
        if (!resp.ok) return;
        const data = await resp.json();
        let totalVram = 0;
        const gpus = Array.isArray(data) ? data : (data.gpus ? data.gpus : Object.values(data));
        // The GPU metrics poller idles for power savings and is woken by this very
        // request (see server-side "wake-on-activity"); the first call after a period
        // of dormancy can race the poller's first tick and get an empty snapshot back.
        // Retry a few times, but only the FIRST attempt is awaited by the caller —
        // retries run in the background so a slow-to-wake poller doesn't hold up
        // the rest of the Download tab's init (e.g. wiring the search box).
        if (gpus.length === 0 && retriesLeft > 0) {
            if (!background) {
                fetchGpuVram(retriesLeft, true);
                return;
            }
            await new Promise(r => setTimeout(r, 1200));
            return fetchGpuVram(retriesLeft - 1, true);
        }
        for (const g of gpus) {
            const t = g.vram_total_mb || g.total_mb || g.total_memory_mb || g.vram_total || 0;
            totalVram += t * 1024 * 1024;
            const id = `${g.name || ''} ${g.vendor || ''} ${g.backend || ''}`;
            if (/apple|metal/i.test(id) || g.metal_gpu_limit_mb != null) cachedUnified = true;
        }
        if (totalVram > 0 && !cachedUnified) cachedVram = totalVram;
        // A background retry can resolve real GPU data after loadQuantAdvisor()
        // already ran once and bailed on availVram === 0; give it a second chance.
        if (background && gpus.length > 0 && hfState.paramB > 0) {
            triggerQuantAdvisor();
        }
    } catch {
        /* ignore */
    }
}

async function fetchSystemRam() {
    try {
        const headers = window.authHeaders ? window.authHeaders() : {};
        const resp = await fetch('/metrics/system', { headers });
        if (!resp.ok) return;
        const d = await resp.json();
        cachedRamTotal = (d.ram_total_gb || 0) * 1024 * 1024 * 1024;
    } catch {
        // ignore
    }
}

/// Fetch MemoryAvailabilitySnapshot from the single backend source of truth.
/// Uses current_safe_availability_bytes for fit determination (never total unified memory).
async function fetchMemoryAvailability() {
    try {
        const headers = window.authHeaders ? window.authHeaders() : {};
        const resp = await fetch('/api/memory-availability', { headers });
        if (!resp.ok) return null;
        const data = await resp.json();
        if (!data.ok || !data.snapshot) return null;
        return data.snapshot;
    } catch {
        return null;
    }
}

async function updateVramDisplay(file) {
    const panel = document.getElementById('mm-vram-panel');
    if (!panel) return;

    // Phase 5b Part C: On unified memory (Apple Silicon), use current_safe_availability_bytes
    // from the MemoryAvailabilitySnapshot, NOT total unified memory.
    let availVram = cachedVram || 0;
    let metalLimit = 0;
    if (cachedUnified) {
        const snapshot = await fetchMemoryAvailability();
        if (snapshot && snapshot.current_safe_availability_bytes > 0) {
            availVram = snapshot.current_safe_availability_bytes;
        }
        if (snapshot && snapshot.configured_ceiling_bytes > 0) {
            metalLimit = snapshot.configured_ceiling_bytes;
        }
    }
    if (!availVram) {
        panel.style.display = 'none';
        return;
    }

    // Use model bytes from file or estimate from paramB
    let modelBytes = hfState.modelBytes;
    if (!modelBytes && hfState.paramB > 0) {
        const fname = (file?.path || file?.name || '').toLowerCase();
        const quant = guessQuantFromName(fname);
        const BPW = { q8_0: 8.5, q6_k: 6.5625, q5_k_m: 5.69, q4_k_m: 4.85, iq4_xs: 4.25, q3_k_m: 3.875, q2_k: 2.625, iq2_xxs: 2.0625, f16: 16, bf16: 16 };
        const bpw = BPW[quant] ?? 4.85;
        modelBytes = Math.round(hfState.paramB * 1e9 * bpw / 8);
    }
    if (!modelBytes) {
        panel.style.display = 'none';
        return;
    }

    // Estimate via the backend so the preview uses the SAME math as the spawn wizard /
    // preset editor. Pre-download, the backend range-fetches the GGUF header from HuggingFace
    // Use user-selected context size (default 64K) for VRAM calculation.
    const previewCtx = hfState.previewCtx || 65536;
    const mmprojBytes = hfState.mmprojBytes || 0;
    let data;
    try {
        // Builder item 6: canonical body builder for cross-surface equality.
        // MLX models use mlx backend (hf_repo_id alone, no file_path needed).
        const isMlx = hfState.modelFormat === 'mlx';
        const body = buildEstimateBody({
            backend: isMlx ? 'mlx' : 'llama_cpp',
            hf_repo_id: hfState.selectedRepoId || null,
            hf_file_path: file?.path || file?.name || null,
            model_size_bytes: modelBytes,
            n_ctx: previewCtx,
            parallel_slots: 1,
            ubatch_size: 512,
            ctk: 'q8_0',
            ctv: 'q8_0',
            available_vram_bytes: availVram,
            is_unified_memory: isMlx || cachedUnified,
            mmproj_bytes: mmprojBytes,
            // No workload_scenario. This read was `sessionState.workloadProfile?.id`, and
            // that property is declared nowhere and written nowhere, so the fallback was
            // taken on every call and the Library quietly estimated every model as a coding
            // agent. Workload is a spawn-wizard input derived from the page-1 use-case
            // cards; the Library has no such selection to start from, so it takes the
            // estimator's own default rather than inventing one.
        });
        const headers = window.authHeaders
            ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
            : { 'Content-Type': 'application/json' };
        const resp = await fetch('/api/vram-estimate', { method: 'POST', headers, body: JSON.stringify(body) });
        if (resp.ok) data = await resp.json();
    } catch {
        // network error — fall through to hide
    }
    if (!data || !data.ok) {
        panel.style.display = 'none';
        return;
    }

    // Fold MTP + linear-attention state into the displayed "overhead" segment.
    const weightsBytes = data.weights_bytes || 0;
    const kvEstimate = data.kv_cache_bytes || 0;
    const overhead = (data.overhead_bytes || 0) + (data.mtp_bytes || 0) + (data.linear_attn_state_bytes || 0);
    const total = data.total_bytes || (weightsBytes + kvEstimate + overhead + mmprojBytes);
    const free = availVram - total;

    // Show panel (remove loading state)
    panel.style.display = '';
    panel.classList.remove('vram-panel-loading');

    // Update header — show needed vs available, plus max possible (metal limit)
    const labelEl = document.getElementById('mm-vram-panel-label');
    const totalEl = document.getElementById('mm-vram-panel-total');
    if (labelEl) {
        labelEl.textContent = '';
        const neededSpan = document.createElement('span');
        neededSpan.textContent = formatGB(total) + ' needed';
        labelEl.appendChild(neededSpan);
        const sep = document.createElement('span');
        sep.textContent = ' · ';
        labelEl.appendChild(sep);
        const availSpan = document.createElement('span');
        availSpan.textContent = formatGB(availVram) + ' available';
        labelEl.appendChild(availSpan);
        if (cachedUnified && metalLimit > 0) {
            const maxSep = document.createElement('span');
            maxSep.textContent = ' · ';
            labelEl.appendChild(maxSep);
            const maxSpan = document.createElement('span');
            maxSpan.textContent = formatGB(metalLimit) + ' max';
            labelEl.appendChild(maxSpan);
        }
    }
    if (totalEl) {
        if (metalLimit > 0 && total > metalLimit) {
            totalEl.textContent = 'Exceeds metal limit · need ' + formatGB(metalLimit - total) + ' less';
            totalEl.style.color = 'var(--color-error, #f44336)';
        } else if (free < 0) {
            totalEl.textContent = 'Won\'t fit · need ' + formatGB(Math.abs(free)) + ' more';
            totalEl.style.color = 'var(--color-error, #f44336)';
        } else {
            totalEl.textContent = 'Fits · ' + formatGB(free) + ' remaining';
            totalEl.style.color = 'var(--color-success, #4ade80)';
        }
    }

    // Update bar — show breakdown of what's needed as parts of total
    const denom = total > 0 ? total : 1;
    const weightsPct = weightsBytes / denom;
    const kvPct = kvEstimate / denom;
    const mmprojPct = mmprojBytes / denom;
    const overheadPct = overhead / denom;

    setSegWidth(document.getElementById('mm-vseg-weights'), weightsPct);
    setSegWidth(document.getElementById('mm-vseg-kv'), kvPct);
    setSegWidth(document.getElementById('mm-vseg-mmproj'), mmprojPct);
    setSegWidth(document.getElementById('mm-vseg-overhead'), overheadPct);
    setSegWidth(document.getElementById('mm-vseg-free'), 0);

    const barEl = document.getElementById('mm-vram-bar');
    if (barEl) {
        const ratio = availVram > 0 ? total / availVram : 0;
        barEl.classList.toggle('tight', ratio >= 0.88 && ratio < 1.0);
        barEl.classList.toggle('over', ratio >= 1.0);
    }

    // Update legend
    const weightsLabel = document.getElementById('mm-vleg-weights-label');
    const kvLabel = document.getElementById('mm-vleg-kv-label');
    const mmprojItem = document.getElementById('mm-vleg-mmproj');
    const mmprojLabel = document.getElementById('mm-vleg-mmproj-label');
    const overheadLabel = document.getElementById('mm-vleg-overhead-label');
    const freeLabel = document.getElementById('mm-vleg-free-label');
    const freeDot = document.querySelector('#mm-vleg-free .vram-legend-dot-free');

    if (weightsLabel) weightsLabel.textContent = 'Weights ' + formatGB(weightsBytes);
    if (kvLabel) kvLabel.textContent = 'KV ' + formatGB(kvEstimate);

    if (mmprojBytes > 0) {
        if (mmprojItem) mmprojItem.style.display = '';
        if (mmprojLabel) mmprojLabel.textContent = 'mmproj ' + formatGB(mmprojBytes);
    } else {
        if (mmprojItem) mmprojItem.style.display = 'none';
    }

    if (overheadLabel) overheadLabel.textContent = 'Overhead ' + formatGB(overhead);

    if (freeLabel) {
        // Hide free label in legend — fit status is in header
        freeLabel.textContent = '';
    }
    if (freeDot) freeDot.style.background = free >= 0 ? '' : 'var(--color-error)';

    // Update ctx train warning (if introspection revealed n_ctx_train)
    updateCtxTrainWarning();
}

// ── Context train warning (wizard-style) ─────────────────────────────────────

function hideCtxTrainWarning() {
    const el = document.getElementById('mm-ctx-train-warning');
    if (el) el.style.display = 'none';
}

function updateCtxTrainWarning() {
    const el = document.getElementById('mm-ctx-train-warning');
    if (!el) return;
    const nCtxTrain = hfState.nCtxTrain;
    if (!nCtxTrain) {
        el.style.display = 'none';
        return;
    }
    // Show warning if we're looking at a model with a known training limit
    // and the user might set context beyond it
    const fmtK = n => n >= 1024 ? Math.round(n / 1024) + 'k' : String(n);
    el.textContent = '';
    const strong = document.createElement('strong');
    strong.textContent = 'Training context: ' + fmtK(nCtxTrain) + ' tokens';
    el.appendChild(strong);
    el.appendChild(document.createTextNode(
        ' — setting context beyond this may degrade quality. Use --rope-scaling yarn if needed.'
    ));
    el.className = 'ctx-fit-warning';
    el.style.display = '';
}

// ── mmproj companion detection (wizard-style) ────────────────────────────────

function deriveMmprojSaveName(modelPath, mmprojPath) {
    const modelBase = (modelPath.split('/').pop() || modelPath).replace(/\.gguf$/i, '');
    const stem = modelBase.replace(/-?(Q\d[\w.]*|IQ\d[\w.]*|BF16|F16)$/i, '');
    const mmprojBase = mmprojPath.split('/').pop() || mmprojPath;
    return `${stem}-${mmprojBase}`;
}

async function detectMmprojCompanion(repoId) {
    const section = document.getElementById('mm-mmproj-section');
    const content = document.getElementById('mm-mmproj-content');
    if (!section || !content) return;

    // mmproj is a llama.cpp/GGUF concept. The selected repo's own files can carry an
    // is_mmproj-flagged file even when the user is viewing it in MLX scope (some repos
    // host both formats) — showing the toggle there would offer a control that does
    // nothing for the MLX download path. See spawn-wizard-mmproj.js for the equivalent
    // spawn-wizard gate.
    if (!repoId || hfState.modelFormat === 'mlx') {
        section.style.display = 'none';
        return;
    }

    try {
        const headers = window.authHeaders ? window.authHeaders() : {};
        const resp = await fetch('/api/hf/files', {
            method: 'POST',
            headers: { ...headers, 'Content-Type': 'application/json' },
            body: JSON.stringify({ repo_id: repoId }),
        });
        if (!resp.ok) { section.style.display = 'none'; return; }
        const data = await resp.json();

        // The scope toggle or a newer selection may have fired while this fetch was in
        // flight — re-check before showing anything so a stale response can't reveal the
        // section for a repo the user is no longer viewing in GGUF scope.
        if (hfState.modelFormat === 'mlx' || hfState.selectedRepoId !== repoId) {
            section.style.display = 'none';
            return;
        }

        const files = data.files || [];
        const mmprojFiles = files.filter(f => f.is_mmproj);

        if (!mmprojFiles.length) {
            section.style.display = 'none';
            hfState.mmprojFiles = [];
            hfState.mmprojBytes = 0;
            return;
        }

        hfState.mmprojFiles = mmprojFiles;
        section.style.display = '';
        const recommendedMmproj = getRecommendedMmproj(mmprojFiles);

        // Render mmproj options
        content.innerHTML = '';

        // Checkbox to enable mmproj
        const checkLabel = document.createElement('label');
        checkLabel.className = 'hw-toggle-label';
        const check = document.createElement('input');
        check.type = 'checkbox';
        check.checked = !!hfState.mmprojPath;
        checkLabel.appendChild(check);
        const span = document.createElement('span');
        span.textContent = 'Include vision projector (mmproj)';
        checkLabel.appendChild(span);
        content.appendChild(checkLabel);

        // Select for mmproj file
        const select = document.createElement('select');
        select.className = 'hw-mmproj-select';
        select.style.marginLeft = '22px';
        select.style.display = check.checked ? '' : 'none';

        const noneOpt = document.createElement('option');
        noneOpt.value = '';
        noneOpt.textContent = '(none — text-only)';
        select.appendChild(noneOpt);

        mmprojFiles.forEach(f => {
            const fpath = f.path || f.name || '';
            const fname = fpath.split('/').pop();
            const opt = document.createElement('option');
            opt.value = fpath;
            opt.dataset.repoId = f.repo_id || repoId;
            const sizeStr = f.size ? ' · ' + formatBytes(f.size) : '';
            opt.textContent = fname + sizeStr + (f.is_recommended_mmproj ? ' · Recommended' : '');
            if (f.is_recommended_mmproj) {
                opt.title = f.mmproj_recommendation || 'Preferred projector format for this model family';
            }
            if (fpath === hfState.mmprojPath) opt.selected = true;
            select.appendChild(opt);
        });

        select.addEventListener('change', () => {
            const fpath = select.value;
            hfState.mmprojPath = fpath;
            const f = mmprojFiles.find(x => (x.path || x.name) === fpath);
            hfState.mmprojRepoId = f?.repo_id || repoId;
            hfState.mmprojBytes = f?.size ? Number(f.size) : 0;
            scheduleVramUpdate(hfState.selectedFile);
        });

        check.addEventListener('change', () => {
            select.style.display = check.checked ? '' : 'none';
            if (!check.checked) {
                hfState.mmprojPath = '';
                hfState.mmprojRepoId = '';
                hfState.mmprojBytes = 0;
            } else if (!select.value && mmprojFiles.length) {
                const preferred = recommendedMmproj || mmprojFiles[0];
                select.value = preferred.path || preferred.name || '';
                hfState.mmprojPath = select.value;
                hfState.mmprojRepoId = preferred.repo_id || repoId;
                hfState.mmprojBytes = preferred.size ? Number(preferred.size) : 0;
            }
            scheduleVramUpdate(hfState.selectedFile);
        });

        content.appendChild(select);

        // Hint
        const hint = document.createElement('div');
        hint.className = 'mm-hint';
        const sizeFile = recommendedMmproj || mmprojFiles[0];
        const recommendation = recommendedMmproj?.mmproj_recommendation
            ? recommendedMmproj.mmproj_recommendation + '. '
            : '';
        hint.textContent = recommendation + 'Adds ~' + (sizeFile?.size ? formatBytes(sizeFile.size) : '0.5–1.5 GB') + ' VRAM. Required for multimodal inference.';
        content.appendChild(hint);

    } catch {
        section.style.display = 'none';
    }
}

function hideMmprojSection() {
    const el = document.getElementById('mm-mmproj-section');
    if (el) el.style.display = 'none';
}

// ── Disk tab: the Hugging Face cache the app does not manage ─────────────────
//
// The library tab reports what can be launched. This one reports what is merely
// taking up space: repos downloaded by hand, by `mlx_lm.convert`, or by a script,
// all of which land in ~/.cache/huggingface and are invisible to every other tab.
//
// Never scans on open. A cold walk of a few hundred gigabytes is not something to
// start because someone clicked a tab, so the first render is an empty state with a
// button, and every later render reuses what that button fetched.

let diskAudit = null;
const diskSelection = new Set();

const DISK_KIND_LABELS = {
    text: 'Text',
    vision: 'Vision',
    audio: 'Audio',
    embedding: 'Embedding',
    unknown: 'Unknown',
};

function initDiskTab() {
    const scanBtn = document.getElementById('mm-disk-scan');
    if (!scanBtn || scanBtn.dataset.wired === '1') return;
    scanBtn.dataset.wired = '1';
    scanBtn.addEventListener('click', () => scanDiskCache());
    document.getElementById('mm-disk-delete')?.addEventListener('click', deleteSelectedDiskRepos);
    document.getElementById('mm-disk-import')?.addEventListener('click', importSelectedDiskRepos);
    document.getElementById('mm-disk-filters')?.addEventListener('change', renderDiskRows);
    document.getElementById('mm-disk-rows')?.addEventListener('change', (event) => {
        const box = event.target.closest('input[data-repo]');
        if (!box) return;
        if (box.checked) diskSelection.add(box.dataset.repo);
        else diskSelection.delete(box.dataset.repo);
        renderDiskSelection();
    });
}

async function scanDiskCache() {
    const rows = document.getElementById('mm-disk-rows');
    const scanBtn = document.getElementById('mm-disk-scan');
    if (rows) rows.innerHTML = '<div class="mm-loading">Walking the shared cache…</div>';
    if (scanBtn) scanBtn.disabled = true;
    try {
        const resp = await fetch('/api/models/external-cache', {
            headers: window.authHeaders ? window.authHeaders() : {},
        });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(data.error || 'Scan failed');
        if (!data.present) {
            diskAudit = null;
            if (rows) {
                rows.innerHTML = '<div class="mm-import-empty-state">No shared Hugging Face cache on this machine. Everything the app can see is already in the library.</div>';
            }
            setDiskTotals('Nothing outside the library.');
            return;
        }
        diskAudit = data.audit;
        diskSelection.clear();
        renderDiskTotals();
        renderDiskRows();
    } catch (error) {
        if (rows) rows.innerHTML = `<div class="mm-import-empty-state">${escapeHtml(error.message || String(error))}</div>`;
    } finally {
        if (scanBtn) scanBtn.disabled = false;
    }
}

function setDiskTotals(text) {
    const totals = document.getElementById('mm-disk-totals');
    if (totals) totals.textContent = text;
}

function renderDiskTotals() {
    if (!diskAudit) return;
    const byKind = new Map();
    let duplicateBytes = 0;
    for (const repo of diskAudit.repos) {
        byKind.set(repo.kind, (byKind.get(repo.kind) || 0) + repo.bytes);
        if (repo.in_library) duplicateBytes += repo.bytes;
    }
    const parts = [...byKind.entries()]
        .sort((a, b) => b[1] - a[1])
        .map(([kind, bytes]) => `${DISK_KIND_LABELS[kind] || kind} ${formatBytes(bytes)}`);
    const totals = document.getElementById('mm-disk-totals');
    if (totals) {
        // The duplicate line is the actionable one: those bytes exist twice on this
        // machine, so reclaiming them costs nothing.
        // eslint-disable-next-line no-unsanitized/property -- only formatBytes numerics and escapeHtml'd text
        totals.innerHTML = [
            `<strong>${formatBytes(diskAudit.total_bytes)}</strong> in ${diskAudit.repos.length} repos`,
            parts.length ? `<span class="mm-disk-breakdown">${escapeHtml(parts.join(' · '))}</span>` : '',
            duplicateBytes ? `<span class="mm-disk-dupe">${formatBytes(duplicateBytes)} already in your library</span>` : '',
            diskAudit.unaccounted_bytes ? `<span class="mm-disk-note">${formatBytes(diskAudit.unaccounted_bytes)} not in a model repo, not listed below</span>` : '',
            diskAudit.truncated ? '<span class="mm-disk-note">Listing was truncated; some repos are not shown.</span>' : '',
        ].filter(Boolean).join('<br>');
    }
    const count = document.getElementById('mm-disk-tab-count');
    if (count) count.textContent = String(diskAudit.repos.length);
    document.getElementById('mm-disk-filters')?.removeAttribute('hidden');
}

function diskKindFilter() {
    const boxes = document.querySelectorAll('#mm-disk-filters input[data-kind]');
    const allowed = new Set();
    boxes.forEach(box => { if (box.checked) allowed.add(box.dataset.kind); });
    return allowed;
}

function renderDiskRows() {
    const rows = document.getElementById('mm-disk-rows');
    if (!rows || !diskAudit) return;
    const allowed = diskKindFilter();
    const visible = diskAudit.repos.filter(repo => allowed.has(repo.kind));
    if (!visible.length) {
        rows.innerHTML = '<div class="mm-import-empty-state">No repos match the selected kinds.</div>';
        renderDiskSelection();
        return;
    }
    // eslint-disable-next-line no-unsanitized/property -- every interpolated repo field goes through escapeHtml
    rows.innerHTML = visible.map(repo => {
        const badges = [];
        badges.push(`<span class="mm-disk-badge mm-disk-badge--${escapeHtml(repo.kind)}">${escapeHtml(DISK_KIND_LABELS[repo.kind] || repo.kind)}</span>`);
        // Where the verdict came from, always shown: "the config says so" and "the
        // folder name says so" are different levels of confidence and the next button
        // on this row is Delete.
        if (repo.kind_source === 'repo-name') badges.push('<span class="mm-disk-badge mm-disk-badge--weak" title="No config.json was readable; this is a guess from the repo name">name only</span>');
        if (repo.kind_source === 'none') badges.push('<span class="mm-disk-badge mm-disk-badge--weak" title="Nothing readable said what this is">unidentified</span>');
        if (repo.has_vision) badges.push('<span class="mm-disk-badge mm-disk-badge--cap" title="Config declares a vision tower">+vision</span>');
        if (repo.has_audio) badges.push('<span class="mm-disk-badge mm-disk-badge--cap" title="Config declares an audio tower">+audio</span>');
        if (repo.name_hint) badges.push(`<span class="mm-disk-badge mm-disk-badge--weak" title="The repo name suggests ${escapeHtml(DISK_KIND_LABELS[repo.name_hint] || repo.name_hint)}, the config does not">name says ${escapeHtml(DISK_KIND_LABELS[repo.name_hint] || repo.name_hint)}</span>`);
        if (repo.in_library) badges.push('<span class="mm-disk-badge mm-disk-badge--dupe" title="Your library already has this repo cached; this copy is redundant">duplicate</span>');
        const meta = [`${repo.file_count} files`];
        if (repo.revisions.length > 1) meta.push(`${repo.revisions.length} revisions`);
        if (repo.last_modified_unix) meta.push(`touched ${new Date(repo.last_modified_unix * 1000).toLocaleDateString()}`);
        const warn = repo.partial_reason
            ? `<div class="mm-disk-warn">${escapeHtml(repo.partial_reason)}</div>`
            : '';
        return `<label class="mm-disk-row">
            <input type="checkbox" data-repo="${escapeHtml(repo.repo_id)}"${diskSelection.has(repo.repo_id) ? ' checked' : ''}>
            <span class="mm-disk-size">${formatBytes(repo.bytes)}</span>
            <span class="mm-disk-id">${escapeHtml(repo.repo_id)}</span>
            <span class="mm-disk-badges">${badges.join('')}</span>
            <span class="mm-disk-meta">${escapeHtml(meta.join(' · '))}</span>
            ${warn}
        </label>`;
    }).join('');
    renderDiskSelection();
}

function renderDiskSelection() {
    const label = document.getElementById('mm-disk-selection');
    const selected = diskAudit
        ? diskAudit.repos.filter(repo => diskSelection.has(repo.repo_id))
        : [];
    const bytes = selected.reduce((sum, repo) => sum + repo.bytes, 0);
    if (label) {
        label.textContent = selected.length
            ? `${selected.length} selected · ${formatBytes(bytes)}`
            : '';
    }
    const disabled = selected.length === 0;
    const deleteBtn = document.getElementById('mm-disk-delete');
    const importBtn = document.getElementById('mm-disk-import');
    if (deleteBtn) deleteBtn.disabled = disabled;
    if (importBtn) importBtn.disabled = disabled;
    return selected;
}

async function deleteSelectedDiskRepos() {
    const selected = renderDiskSelection();
    if (!selected.length) return;
    const bytes = selected.reduce((sum, repo) => sum + repo.bytes, 0);
    // Spelled out per repo rather than summarised. A wrong bulk delete here is
    // unrecoverable and the sizes involved are large enough to be worth re-reading.
    const lines = selected.map(repo => `${formatBytes(repo.bytes)}  ${repo.repo_id}${repo.in_library ? '  (duplicate of a library copy)' : ''}`);
    const confirmed = await _showConfirm(
        'Delete from the shared cache',
        `${selected.length} repo(s), ${formatBytes(bytes)}:\n\n${lines.join('\n')}\n\nThese are deleted from disk, not from the library. Anything not marked duplicate would have to be downloaded again.\nThis cannot be undone.`
    );
    if (!confirmed) return;
    try {
        const resp = await fetch('/api/models/external-cache/delete', {
            method: 'POST',
            headers: window.authHeaders
                ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
                : { 'Content-Type': 'application/json' },
            body: JSON.stringify({ repo_ids: selected.map(repo => repo.repo_id) }),
        });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(data.error || 'Delete failed');
        const failures = (data.results || []).filter(row => !row.ok);
        if (failures.length) {
            showToast(`Freed ${formatBytes(data.freed_bytes)}; ${failures.length} failed: ${failures[0].error}`, 'warning');
        } else {
            showToast(`Freed ${formatBytes(data.freed_bytes)}`, 'success');
        }
    } catch (error) {
        showToast(error.message || String(error), 'error');
        return;
    }
    // Re-scan rather than patching the rows: sizes and duplicate flags are now stale,
    // and this screen is the wrong place to be showing a guess.
    diskSelection.clear();
    await scanDiskCache();
}

/// Repos per import. Mirrors the backend's own cap; checked here so an oversized
/// selection is a message instead of a whole-batch rejection after the round trip.
const DISK_IMPORT_LIMIT = 32;

async function importSelectedDiskRepos() {
    const selected = renderDiskSelection();
    if (!selected.length) return;

    // The migration planner refuses the entire batch on the first problem, so anything
    // it would reject is filtered out here and named, rather than taking the rest down
    // with it. A duplicate would collide with the library copy that already exists; a
    // repo id that is not exactly `owner/name` is a directory whose name did not decode.
    const blocked = [];
    const importable = [];
    for (const repo of selected) {
        if (repo.in_library) blocked.push(`${repo.repo_id} — your library already has it`);
        else if (!/^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/.test(repo.repo_id)) blocked.push(`${repo.repo_id} — not a recognisable repo id`);
        else importable.push(repo);
    }
    if (!importable.length) {
        showToast(`Nothing to import: ${blocked.join('; ')}`, 'warning');
        return;
    }
    if (importable.length > DISK_IMPORT_LIMIT) {
        showToast(`Import is limited to ${DISK_IMPORT_LIMIT} repos at a time (${importable.length} selected)`, 'warning');
        return;
    }

    const repoIds = importable.map(repo => repo.repo_id);
    try {
        const previewResp = await fetch('/api/models/library/migration/preview', {
            method: 'POST',
            headers: window.authHeaders
                ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
                : { 'Content-Type': 'application/json' },
            body: JSON.stringify({ hf_repos: repoIds }),
        });
        const plan = await previewResp.json().catch(() => ({}));
        if (!previewResp.ok) throw new Error(plan.error || 'Import preview failed');

        // The planner returns a full library migration, not just these repos: anything
        // already in the library that is not in its canonical folder gets moved too, and
        // presets and sessions are rewritten to follow. That is a bigger action than
        // "import these", so it is stated rather than buried in a move count.
        const moves = Array.isArray(plan.moves) ? plan.moves.length : 0;
        const rewrites = Array.isArray(plan.persistence_rewrites) ? plan.persistence_rewrites.length : 0;
        const extra = moves > repoIds.length
            ? `\n\nThis plan also reorganises ${moves - repoIds.length} other path(s) already inside your library into their canonical folders${rewrites ? `, and rewrites ${rewrites} saved file(s) (presets, sessions, tags) to match` : ''}.`
            : '';
        const skipped = blocked.length ? `\n\nSkipped: ${blocked.join('; ')}` : '';
        const confirmed = await _showConfirm(
            'Import into the library',
            `${repoIds.length} repo(s) will be moved out of the shared cache and into your managed library:\n\n${repoIds.join('\n')}\n\nThis is a move, not a copy, so the shared cache shrinks by the same amount.${extra}${skipped}`
        );
        if (!confirmed) return;

        // Execution is admin-gated because it rewrites paths that presets and sessions
        // point at, so it needs the stronger token the migration endpoint requires.
        const tokenResp = await fetch('/api/db/admin-token', {
            headers: window.authHeaders ? window.authHeaders() : {},
        });
        const tokenData = tokenResp.ok ? await tokenResp.json().catch(() => ({})) : {};
        if (!tokenData.token) throw new Error('Authentication required to modify the library');

        const execResp = await fetch('/api/models/library/migration/execute', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + tokenData.token,
            },
            body: JSON.stringify({
                plan_id: plan.plan_id,
                confirmation: 'MIGRATE_MODEL_LIBRARY',
                hf_repos: repoIds,
            }),
        });
        const result = await execResp.json().catch(() => ({}));
        if (!execResp.ok) throw new Error(result.error || 'Import failed');
        showToast(`Imported ${repoIds.length} repo(s) into the library`, 'success');
    } catch (error) {
        showToast(error.message || String(error), 'error');
        return;
    }
    diskSelection.clear();
    inventoryCache = null;
    await scanDiskCache();
}

// ── Utilities ────────────────────────────────────────────────────────────────

function formatBytes(bytes) {
    if (!bytes) return '';
    const b = Number(bytes);
    if (!isFinite(b)) return '';
    if (b >= 1024 ** 3) return (b / (1024 ** 3)).toFixed(1) + ' GiB';
    if (b >= 1024 ** 2) return (b / (1024 ** 2)).toFixed(1) + ' MiB';
    if (b >= 1024) return (b / 1024).toFixed(0) + ' KiB';
    return b + ' B';
}

function formatParams(paramB) {
    if (paramB >= 1000) return (paramB / 1000).toFixed(1) + 'T';
    return paramB + 'B';
}

function formatCtx(n) {
    if (n >= 1024) return Math.round(n / 1024) + 'k';
    return String(n);
}

function formatVramTotal(bytes) {
    const gb = bytes / (1024 ** 3);
    if (gb >= 100) return Math.round(gb) + ' GB';
    return gb.toFixed(1) + ' GB';
}

function formatGB(bytes) {
    if (!bytes || !isFinite(bytes)) return '0 MB';
    const gb = bytes / (1024 ** 3);
    if (gb >= 1) return gb.toFixed(1) + ' GB';
    return Math.round(bytes / (1024 ** 2)) + ' MB';
}

function setSegWidth(el, ratio) {
    if (!el) return;
    const pct = Math.max(0, Math.min(1, ratio)) * 100;
    el.style.width = pct + '%';
}

function guessQuantFromName(name) {
    const lower = name.toLowerCase();
    if (lower.includes('q8_0')) return 'q8_0';
    if (lower.includes('q6_k')) return 'q6_k';
    if (lower.includes('q5_k_m')) return 'q5_k_m';
    if (lower.includes('q5_k_s')) return 'q5_k_s';
    if (lower.includes('q4_k_m')) return 'q4_k_m';
    if (lower.includes('q4_k_s')) return 'q4_k_s';
    if (lower.includes('iq4_xs')) return 'iq4_xs';
    if (lower.includes('q4_0')) return 'q4_0';
    if (lower.includes('q3_k_m')) return 'q3_k_m';
    if (lower.includes('q2_k')) return 'q2_k';
    if (lower.includes('iq2_xxs')) return 'iq2_xxs';
    if (lower.includes('f16') || lower.includes('bf16')) return 'f16';
    return 'q4_k_m';
}
