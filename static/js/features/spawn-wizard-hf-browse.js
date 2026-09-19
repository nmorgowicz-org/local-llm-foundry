// HF discover/search/quant-advisor/community-picks/quantizer-editor widgets for the spawn wizard.
import { showToast } from './toast.js';
import { openCardPanel } from './spawn-wizard-model-card.js';
import { autoInstallChatTemplate } from './spawn-wizard-chat-template.js';
import { scheduleRapidMlxProfileFetch } from './spawn-wizard-rapid-mlx.js';
import { formatCtx, formatGB } from './spawn-wizard-format.js';
import { scheduleEstimate } from './vram-estimate.js';
import {
  HF_DISCOVER_CATEGORIES,
  hfSearch,
  hfListFiles,
  hfRenderDiscoverPills,
  hfLoadQuickPicks,
  hfCreateScopeSelector,
  hfCreateSortSelector,
  HF_SORT,
  openHfEvidence,
} from './hf-browse.js';
import {
  dom,
  wizardState,
  showValidationError,
  clearValidationError,
  refreshEngineRecommendation,
  refreshStepGuardrails,
  selectWizardEngine,
  updateSelectedModelDisplay,
  introspectHfFileMetadata,
  effectiveAvailBytes,
  isUnifiedMemory,
  ensureGpuVramFetched,
  scheduleVramUpdate,
  setOnMemoryAvailabilityReady,
} from './spawn-wizard.js';

export const hfBrowseState = {
  mlxActive: false,
  ggufActive: true,
  allActive: false,
  sort: HF_SORT.LAST_UPDATED,
  quantsOnly: false,
};

// Handle from hfCreateScopeSelector(), kept module-level so the engine-driven default (below)
// can update the live selector without a teardown/rebuild.
let _scopeSelectorHandle = null;
// Set the moment the user clicks a scope button directly. Once true, engine switches stop
// overriding their choice (plan §1.5: "engine-driven default is a pure narrowing", not something
// that should fight a deliberate manual pick).
let _userTouchedScope = false;

// Narrows the HF scope toggle to the engine actually selected, replacing the old
// platform-based (isMac) default. Additive resolveScope() means 'gguf'-only for llama.cpp and
// 'mlx'-only for rapid_mlx is a pure narrowing of the previous always-both-on-Mac default, not a
// semantic change (plan §1.5). No-op once the user has touched the selector directly.
export function _applyScopeDefaultForEngine(engine) {
  if (_userTouchedScope || !_scopeSelectorHandle) return;
  const mlxActive = engine === 'rapid_mlx';
  const ggufActive = engine !== 'rapid_mlx';
  hfBrowseState.mlxActive = mlxActive;
  hfBrowseState.ggufActive = ggufActive;
  hfBrowseState.allActive = false;
  _scopeSelectorHandle.setScope(mlxActive, ggufActive, false);
}


export function initHfBrowseWidgets() {
     // HF discover pills
    const discoverPillsEl = document.getElementById('hf-discover-pills');
    const quickpicksEl = dom.hfQuickpicks;

    hfRenderDiscoverPills({
      container: discoverPillsEl,
      quickpicksContainer: quickpicksEl,
      onPillClick: (cat, pillEl) => {
        wizardState.hfBrowseAuthor = null;
        if (dom.hfRepoInput) dom.hfRepoInput.value = '';
        const sort = cat.params.query ? hfBrowseState.sort : (cat.params.sort || hfBrowseState.sort);
        hfSearchForWizard({ ...cat.params, sort });
      },
    });

    // HF quick-picks
    hfLoadQuickPicks({
      container: quickpicksEl,
      discoverPillsContainerId: 'hf-discover-pills',
      onAuthorClick: (author) => {
        browseHfAuthor(author);
      },
    });

    // Engine-driven default (plan §1.5): mlx-only for rapid_mlx, gguf-only for llama.cpp,
    // replacing the old platform-based (isMac) default. _applyScopeDefaultForEngine below
    // needs the handle, so seed dataset attrs from the current engine before creating it.
    const initialEngine = wizardState.engine.selected;
    const initialMlx = initialEngine === 'rapid_mlx';
    const initialGguf = initialEngine !== 'rapid_mlx';
    hfBrowseState.mlxActive = initialMlx;
    hfBrowseState.ggufActive = initialGguf;
    const scopeContainer = document.getElementById('spawn-hf-scope-container');
    scopeContainer?.setAttribute('data-hf-scope-mlx', initialMlx ? '1' : '');
    scopeContainer?.setAttribute('data-hf-scope-gguf', initialGguf ? '1' : '');
    scopeContainer?.setAttribute('data-hf-scope-all', '');
    _scopeSelectorHandle = hfCreateScopeSelector({
      container: scopeContainer,
      onChange: (mlxActive, ggufActive, allActive) => {
        _userTouchedScope = true;
        hfBrowseState.mlxActive = mlxActive || allActive;
        hfBrowseState.ggufActive = ggufActive || allActive;
        hfBrowseState.allActive = allActive;
        refireHfSearch();
      },
    });
    hfCreateSortSelector({
      container: document.getElementById('spawn-hf-sort-container'),
      defaultSort: hfBrowseState.sort,
      onChange: (sort) => {
        hfBrowseState.sort = sort;
        refireHfSearch();
      },
    });
    document.getElementById('spawn-hf-quants-only')?.addEventListener('change', (event) => {
      hfBrowseState.quantsOnly = event.target.checked;
      refireHfSearch();
    });

    // Context-size target pills (sidebar) — set BEFORE the Hardware step so the
    // quant advisor's recommendation can be filtered to quants that actually
    // reach the desired context, instead of just the highest-quality quant.
    if (dom.sidebarCtxPills) {
      dom.sidebarCtxPills.addEventListener('click', (e) => {
        const pill = e.target.closest('.vram-ctx-pill');
        if (!pill) return;
        dom.sidebarCtxPills.querySelectorAll('.vram-ctx-pill').forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        wizardState.hardware.contextSize = parseInt(pill.dataset.ctx, 10);
        if (wizardState.model.paramB > 0) {
          if (wizardState.engine.selected === 'rapid_mlx') renderMlxSidebarVramBar();
          else triggerQuantAdvisor();
        }
      });
    }

}

// Wrapper for hfSearch used by wizard (wires callbacks)
function hfSearchForWizard({ query, author, sort, limit }) {
  if (!wizardState.model.hfTokenSet) {
    showValidationError('HuggingFace token not set. Set it in the top-right panel to search for models.');
    return;
  }
  const minParamB = parseFloat(dom.hfMinSize?.value || '0') || 0;
  // When a size filter is active we need a large batch so enough results survive
  // the client-side filter. Without a filter, a smaller page is better UX.
  const effectiveLimit = limit ?? (minParamB > 0 ? 100 : 25);
  hfSearch({
    query,
    author,
    sort,
    vramGb: effectiveAvailBytes() / (1024 ** 3),
    mlxActive: hfBrowseState.mlxActive,
    ggufActive: hfBrowseState.ggufActive,
    allActive: hfBrowseState.allActive,
    hfSort: hfBrowseState.sort,
    quantsOnly: hfBrowseState.quantsOnly,
    limit: effectiveLimit,
    minParamB,
    container: dom.hfSearchResults,
    filelistContainer: dom.hfFileList,
    quickpicksContainer: dom.hfQuickpicks,
    discoverPillsContainerId: 'hf-discover-pills',
    onOpenCardPanel: (repoId) => openCardPanel(repoId),
    onSelectModel: (m) => {
      // Clear stale data from previous model selection
      wizardState.model.paramB = 0;
      wizardState.model.modelBytes = 0;
      wizardState.model.quantFiles = [];
      wizardState.model.originRepo = '';
      wizardState.model.originFile = '';
      wizardState.model.path = '';
      wizardState.model.hfFile = '';
      wizardState.model.hfRepo = m.id;
      wizardState.model.rapidMlxSource = m.format === 'mlx'
        ? { kind: 'hugging_face_repo', repo_id: m.id, revision: 'main' }
        : null;
      if (dom.hfRepoInput) dom.hfRepoInput.value = m.id;
      if (m.param_b > 0) wizardState.model.paramB = m.param_b;
      if (dom.hfSearchResults) dom.hfSearchResults.style.display = 'none';
      dom.hfQuickpicks?.querySelectorAll('.hf-qp-btn').forEach(b => b.classList.remove('active'));
      if (m.format === 'mlx') {
        wizardState.model.modelBytes = Number(m.model_size_bytes) || 0;
        selectWizardEngine('rapid_mlx', true);
        updateSelectedModelDisplay();
        autoInstallChatTemplate();
      } else {
        fetchHfFiles(m.id);
      }
      refreshEngineRecommendation();
      if (m.param_b > 0) {
        if (m.format === 'mlx') triggerMlxSidebarBody(m.id);
        else triggerQuantAdvisor();
      }
      clearValidationError();
      refreshStepGuardrails();
      scheduleRapidMlxProfileFetch(m.id);
    },
  });
}

// ── Mode toggle ───────────────────────────────────────────────────────────────

// ── Step management ───────────────────────────────────────────────────────────


let quantAdvisorDebounce = null;

export function triggerQuantAdvisor() {
  if (quantAdvisorDebounce) clearTimeout(quantAdvisorDebounce);
  quantAdvisorDebounce = setTimeout(loadQuantAdvisor, 600);
}

// Re-run once memory/VRAM data arrives (it's fetched async on step entry, which
// can resolve after the user has already picked a model and hit the paramB gate).
// Deferred to a microtask: this module is pulled in while spawn-wizard.js's own
// imports are still being resolved (circular import), so calling this at top
// level here would hit spawn-wizard.js's `onMemoryAvailabilityReady` binding
// before that module's body has run.
queueMicrotask(() => setOnMemoryAvailabilityReady(triggerQuantAdvisor));

async function loadQuantAdvisor() {
  const paramB = wizardState.model.paramB;
  if (!paramB || paramB <= 0) return;

  const availVram = effectiveAvailBytes();
  if (!availVram) return; // need VRAM to give useful numbers

  // Show loading hint so user knows we're working
  if (dom.quantAdvisorSubtitle) {
    dom.quantAdvisorSubtitle.textContent = 'Analyzing model…';
  }
  if (dom.quantAdvisor) {
    dom.quantAdvisor.style.display = '';
  }

  try {
    const headers = window.authHeaders
      ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
      : { 'Content-Type': 'application/json' };

    // Pass arch info if we have it
    const body = {
      param_b: paramB,
      model_name: wizardState.model.path || wizardState.model.hfRepo || '',
      available_vram_bytes: availVram,
      is_unified_memory: isUnifiedMemory(),
      use_case: wizardState.useCase,
      parallel_slots: wizardState.hardware.parallelSlots,
      n_layers: wizardState.arch.nLayers || undefined,
      n_kv_heads: wizardState.arch.nKvHeads || undefined,
      head_dim: wizardState.arch.headDim || undefined,
      global_head_dim: wizardState.arch.globalHeadDim || undefined,
      n_experts: wizardState.arch.nExperts || undefined,
      mtp_depth: wizardState.arch.mtpDepth || undefined,
    };

    // If the repo's real GGUF file listing is already known (populated when the
    // user picked/confirmed an HF repo), use it so the advisor shows the repo's
    // actual quant names/sizes instead of a synthetic standard-quant estimate —
    // this matters for repos with non-standard naming (imatrix IQ variants,
    // custom mixed-precision schemes like APEX).
    const quantFiles = wizardState.model.quantFiles;
    if (Array.isArray(quantFiles) && quantFiles.length > 0) {
      body.available_files = quantFiles
        .filter(f => f.size > 0)
        .map(f => ({ name: f.label || f.name, size_bytes: f.size }));
    }

    const resp = await fetch('/api/vram/quant-compare', { method: 'POST', headers, body: JSON.stringify(body) });
    if (!resp.ok) {
      if (dom.quantAdvisorSubtitle) dom.quantAdvisorSubtitle.textContent = 'Failed to analyze model.';
      return;
    }
    const data = await resp.json();
    if (!data.ok || !data.quants) {
      if (dom.quantAdvisorSubtitle) dom.quantAdvisorSubtitle.textContent = 'Failed to analyze model.';
      return;
    }

    renderQuantAdvisor(data.quants, availVram);
  } catch {
    if (dom.quantAdvisorSubtitle) dom.quantAdvisorSubtitle.textContent = 'Failed to analyze model.';
  }
}

function renderQuantAdvisor(quants, availVram) {
  if (!dom.quantAdvisor || !dom.quantAdvisorTable) return;
  if (!quants || quants.length === 0) {
    dom.quantAdvisor.style.display = 'none';
    if (dom.sidebarVram) dom.sidebarVram.style.display = 'none';
    if (dom.sidebarQaHint) dom.sidebarQaHint.style.display = 'none';
    return;
  }

  const availGb = Math.round(availVram / (1024 ** 3));
  const budgetLabel = isUnifiedMemory() ? 'Unified memory available' : 'VRAM available';

  // Context-aware pass: find the best quant that fits the user's context target.
  // Only annotate when the user has set a non-trivial context (> 8k default).
  const desiredCtx = wizardState.hardware?.contextSize || 0;
  const annotateCtx = desiredCtx > 8192;
  // First quant (highest quality) whose q8_0 KV max context meets the target.
  const ctxFitQuant = annotateCtx
    ? quants.find(q => q.fits_vram && q.max_ctx_q8 >= desiredCtx)
    : null;
  const qualityRecQuant = quants.find(q => q.recommended && q.fits_vram);
  // Does the quality recommendation also satisfy the context target?
  const qualityRecFitsCtx = !annotateCtx || (qualityRecQuant && qualityRecQuant.max_ctx_q8 >= desiredCtx);

  let subtitle = `${budgetLabel}: ${availGb} GB`;
  if (annotateCtx) subtitle += ` · Context target: ${formatCtx(desiredCtx)}`;
  if (dom.quantAdvisorSubtitle) dom.quantAdvisorSubtitle.textContent = subtitle;

  const table = document.createElement('table');
  table.className = 'qa-table';

  const thead = table.createTHead();
  const hrow = thead.insertRow();
  ['', 'Quant', 'Size', 'Max ctx (q8_0 KV)', 'Max ctx (q4_0 KV)', 'Quality'].forEach(h => {
    const th = document.createElement('th');
    th.textContent = h;
    hrow.appendChild(th);
  });

  const tbody = table.createTBody();
  for (const q of quants) {
    const tr = tbody.insertRow();
    if (q.recommended) tr.className = 'qa-row-rec';
    if (!q.fits_vram) tr.className = (tr.className + ' qa-row-nofit').trim();

    // Fit dot
    const dotTd = tr.insertCell();
    const dot = document.createElement('span');
    dot.className = 'qa-fit-dot ' + (q.fits_vram ? 'fits' : 'nofit');
    dotTd.appendChild(dot);

    // Quant name + badges
    const nameTd = tr.insertCell();
    const nameSpan = document.createElement('span');
    nameSpan.style.fontWeight = '600';
    nameSpan.textContent = q.label;
    nameTd.appendChild(nameSpan);

    if (q.recommended) {
      const badge = document.createElement('span');
      badge.className = 'qa-badge-rec';
      // If context target is active and this rec won't meet it, clarify it's quality-only
      badge.textContent = (annotateCtx && !qualityRecFitsCtx) ? '★ Quality' : '★ Rec';
      badge.style.marginLeft = '6px';
      nameTd.appendChild(badge);
    }
    // Context recommendation badge — shown when it differs from the quality pick
    if (annotateCtx && ctxFitQuant && q.label === ctxFitQuant.label && !qualityRecFitsCtx) {
      const ctxBadge = document.createElement('span');
      ctxBadge.className = 'qa-badge-ctx';
      ctxBadge.textContent = `✓ fits ${formatCtx(desiredCtx)}`;
      ctxBadge.style.marginLeft = '6px';
      nameTd.appendChild(ctxBadge);
    }
    if (q.is_imatrix) {
      const im = document.createElement('span');
      im.style.cssText = 'margin-left:4px; font-size:10px; color:var(--color-text-muted);';
      im.textContent = 'imatrix';
      nameTd.appendChild(im);
    }

    // Size
    const sizeTd = tr.insertCell();
    sizeTd.textContent = q.model_size_gb.toFixed(1) + ' GB';
    sizeTd.style.color = 'var(--color-text-muted)';

    // Max ctx q8_0 — warn if below context target
    const ctxQ8Td = tr.insertCell();
    ctxQ8Td.className = 'qa-ctx';
    if (q.max_ctx_q8 > 0) {
      ctxQ8Td.textContent = formatCtx(q.max_ctx_q8);
      const underTarget = annotateCtx && q.max_ctx_q8 < desiredCtx;
      ctxQ8Td.classList.add(underTarget ? 'qa-ctx-under' : 'qa-ctx-q8');
      if (underTarget) ctxQ8Td.title = `Max ${formatCtx(q.max_ctx_q8)} — below your ${formatCtx(desiredCtx)} target`;
    } else {
      ctxQ8Td.textContent = '—'; ctxQ8Td.classList.add('qa-ctx-na');
    }

    // Max ctx q4_0
    const ctxQ4Td = tr.insertCell();
    ctxQ4Td.className = 'qa-ctx';
    if (q.max_ctx_q4 > 0) {
      ctxQ4Td.textContent = formatCtx(q.max_ctx_q4);
      ctxQ4Td.classList.add('qa-ctx-q4');
    } else {
      ctxQ4Td.textContent = '—'; ctxQ4Td.classList.add('qa-ctx-na');
    }

    // Quality badge
    const qualTd = tr.insertCell();
    const qualBadge = document.createElement('span');
    const qClass = 'qa-quality-' + (q.quality || '').toLowerCase();
    qualBadge.className = `qa-quality-badge ${qClass}`;
    qualBadge.textContent = q.quality_label || q.quality;
    qualTd.appendChild(qualBadge);
  }

  dom.quantAdvisorTable.innerHTML = '';
  dom.quantAdvisorTable.appendChild(table);
  dom.quantAdvisor.style.display = '';

  updateSidebarVramSummary({ quants, availGb, qualityRecQuant, budgetLabel, annotateCtx, ctxFitQuant, qualityRecFitsCtx, desiredCtx });
}

// Compact VRAM/quant summary shown in the right-column sidebar, so the fit
// picture is visible as soon as a model is selected — without needing to
// scroll down to the full quant advisor table in the main column.
function updateSidebarVramSummary({ quants, availGb, qualityRecQuant, budgetLabel, annotateCtx, ctxFitQuant, qualityRecFitsCtx, desiredCtx }) {
  // Sync pill active state with the current context target (default pill = 8K
  // matches wizardState.hardware's own default so the two never disagree).
  if (dom.sidebarCtxPills) {
    const activeCtx = wizardState.hardware?.contextSize || 8192;
    dom.sidebarCtxPills.querySelectorAll('.vram-ctx-pill').forEach(p => {
      p.classList.toggle('active', parseInt(p.dataset.ctx, 10) === activeCtx);
    });
  }

  const pick = (annotateCtx && ctxFitQuant) ? ctxFitQuant : qualityRecQuant;

  if (dom.sidebarVram && dom.sidebarVramValue) {
    dom.sidebarVram.style.display = '';
    if (dom.sidebarVramLabel) dom.sidebarVramLabel.textContent = budgetLabel;
    if (pick) {
      dom.sidebarVramValue.textContent = `${availGb} GB available`;
      dom.sidebarVramValue.className = 'wizard-sidebar-vram-value fits';
      dom.sidebarVramHint.textContent = `${pick.label} recommended · ${pick.model_size_gb.toFixed(1)} GB`;
    } else {
      dom.sidebarVramValue.textContent = `${availGb} GB available`;
      dom.sidebarVramValue.className = 'wizard-sidebar-vram-value nofit';
      dom.sidebarVramHint.textContent = 'No quant fits comfortably — consider a smaller model or CPU offload.';
    }
  }

  if (dom.sidebarQaHint) {
    if (!quants || quants.length === 0) {
      dom.sidebarQaHint.style.display = 'none';
    } else {
      dom.sidebarQaHint.style.display = '';
      dom.sidebarQaHint.textContent = annotateCtx && !qualityRecFitsCtx
        ? `Quality pick (${qualityRecQuant?.label || '—'}) doesn't reach your ${formatCtx(desiredCtx)} context target — see table below for a better-fitting quant.`
        : 'See the quant advisor table below for the full size/context/quality comparison.';
    }
  }

  renderSidebarVramBar(pick);
}

// Weight/KV/overhead breakdown bar for the sidebar's recommended (or best
// context-fitting) quant — mirrors the Models page's vram-bar but scoped to
// the advisor's own pick rather than a separately-chosen file, since at this
// step the user hasn't necessarily downloaded/selected a specific file yet.
function renderSidebarVramBar(pick) {
  if (!dom.sidebarVramBar) return;
  if (!pick || !pick.model_size_gb) {
    dom.sidebarVramBar.style.display = 'none';
    if (dom.sidebarVramLegend) dom.sidebarVramLegend.style.display = 'none';
    return;
  }

  const availVram = effectiveAvailBytes();
  if (!availVram) {
    dom.sidebarVramBar.style.display = 'none';
    if (dom.sidebarVramLegend) dom.sidebarVramLegend.style.display = 'none';
    return;
  }

  const savedState = {
    ...wizardState,
    hardware: { ...wizardState.hardware },
    model: { ...wizardState.model, modelBytes: Math.round(pick.model_size_gb * 1e9) },
    vram: { available: availVram, availableRam: 0, isUnifiedMemory: isUnifiedMemory() },
  };

  scheduleEstimate(savedState, (est) => {
    if (!est || !est.total_bytes) {
      dom.sidebarVramBar.style.display = 'none';
      if (dom.sidebarVramLegend) dom.sidebarVramLegend.style.display = 'none';
      return;
    }

    const weights = est.weights_bytes || 0;
    const kv = est.kv_cache_bytes || 0;
    const overhead = (est.overhead_bytes || 0) + (est.mtp_bytes || 0) + (est.linear_attn_state_bytes || 0);
    const total = est.total_bytes || (weights + kv + overhead);
    const free = Math.max(0, availVram - total);
    const denom = availVram > 0 ? availVram : total;
    if (denom <= 0) return;

    const setW = (el, frac) => {
      if (!el) return;
      const pct = Math.max(0, Math.min(1, frac)) * 100;
      el.style.width = pct.toFixed(2) + '%';
      el.style.display = pct < 0.3 ? 'none' : '';
    };
    setW(dom.sidebarVsegWeights, weights / denom);
    setW(dom.sidebarVsegKv, kv / denom);
    setW(dom.sidebarVsegOverhead, overhead / denom);
    setW(dom.sidebarVsegFree, free / denom);

    if (dom.sidebarVlegWeights) dom.sidebarVlegWeights.textContent = `Weights ${formatGB(weights)}`;
    if (dom.sidebarVlegKv) dom.sidebarVlegKv.textContent = `KV ${formatGB(kv)}`;
    if (dom.sidebarVlegOverhead) dom.sidebarVlegOverhead.textContent = `Overhead ${formatGB(overhead)}`;

    dom.sidebarVramBar.style.display = '';
    dom.sidebarVramBar.classList.toggle('tight', total / availVram >= 0.88 && total / availVram < 1.0);
    dom.sidebarVramBar.classList.toggle('over', total / availVram >= 1.0);
    if (dom.sidebarVramLegend) dom.sidebarVramLegend.style.display = '';
  }, { force: true });
}


// ── MLX sidebar body (plan §2.7) ────────────────────────────────────────────
// An MLX repo already *is* a specific quant — there's no ladder to advise on.
// This replaces the quant advisor with: what the repo already is (bits/group +
// size), ctx pills seeded from the repo's own native context, the existing
// vram-bar fed by the real MLX estimate, and real sibling variants (not a
// synthetic ladder).

let mlxSidebarDebounce = null;

function triggerMlxSidebarBody(repoId) {
  if (mlxSidebarDebounce) clearTimeout(mlxSidebarDebounce);
  mlxSidebarDebounce = setTimeout(() => loadMlxSidebarBody(repoId), 300);
}

async function loadMlxSidebarBody(repoId) {
  const body = document.getElementById('mlx-sidebar-body');
  const infoEl = document.getElementById('mlx-sidebar-repo-info');
  const variantsEl = document.getElementById('mlx-sidebar-variants');
  if (!body || !infoEl || !variantsEl) return;
  if (dom.quantAdvisor) dom.quantAdvisor.style.display = 'none';
  body.style.display = '';
  infoEl.innerHTML = '<div class="mlx-info-label">What this repo is</div>Loading…';
  variantsEl.innerHTML = '';

  const headers = window.authHeaders
    ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
    : { 'Content-Type': 'application/json' };
  const revision = wizardState.model.hfRevision || 'main';

  const [introspectResp, derivativesResp, qualifyResp] = await Promise.allSettled([
    fetch('/api/models/mlx-introspect', {
      method: 'POST', headers, body: JSON.stringify({ repo_id: repoId, revision }),
    }).then(r => r.ok ? r.json() : null),
    fetch('/api/hf/mlx-derivatives', {
      method: 'POST', headers, body: JSON.stringify({ repoId, revision }),
    }).then(r => r.ok ? r.json() : null),
    fetch('/api/hf/qualify', {
      method: 'POST', headers, body: JSON.stringify({ repoId, revision, backend: 'rapid_mlx' }),
    }).then(r => r.ok ? r.json() : null),
  ]);

  // "What this repo already is": bits/group size + resolved size.
  const introspect = introspectResp.status === 'fulfilled' ? introspectResp.value : null;
  const quant = introspect?.data?.config?.quantization;
  const sizeBytes = introspect?.data?.recursive_size_bytes || wizardState.model.modelBytes || 0;
  const quantLabel = quant?.bits ? `${quant.bits}-bit${quant.group_size ? `, group ${quant.group_size}` : ''}` : 'precision not reported';
  // quantLabel/sizeBytes are numeric fields from the qualify/introspect API responses
  // (bits, group_size, byte counts), never raw text.
  // eslint-disable-next-line no-unsanitized/property
  infoEl.innerHTML = `<div class="mlx-info-label">What this repo is</div>${quantLabel} · ${sizeBytes ? formatGB(sizeBytes) : 'size unknown'}`;

  // Ctx pills seeded from the repo's own native context rather than left at the static list.
  const qualify = qualifyResp.status === 'fulfilled' ? qualifyResp.value : null;
  const nativeCtx = qualify?.config?.contextLength;
  if (nativeCtx && dom.sidebarCtxPills) {
    dom.sidebarCtxPills.querySelectorAll('.vram-ctx-pill').forEach(p => {
      const pillCtx = parseInt(p.dataset.ctx, 10);
      p.style.display = pillCtx <= nativeCtx ? '' : 'none';
    });
  }

  // Real sibling variants (this MLX's honest analogue of the quant ladder).
  const derivatives = derivativesResp.status === 'fulfilled' ? derivativesResp.value : null;
  const siblings = (derivatives?.native_mlx_derivatives || []).filter(d => d.repo_id !== repoId);
  if (siblings.length === 0) {
    variantsEl.innerHTML = '';
  } else {
    variantsEl.innerHTML = '<div class="mlx-sidebar-variants-title">Also published as</div>';
    for (const sib of siblings) {
      const row = document.createElement('div');
      row.className = 'mlx-variant-row';
      const bits = sib.quant?.bits ? `${sib.quant.bits}-bit` : sib.format;
      const name = document.createElement('span');
      name.className = 'mlx-variant-name';
      name.textContent = `${bits} · ${sib.converter}`;
      name.title = sib.repo_id;
      const size = document.createElement('span');
      size.className = 'mlx-variant-size';
      size.textContent = sib.size ? formatGB(sib.size) : '?';
      const evidenceBtn = document.createElement('button');
      evidenceBtn.type = 'button';
      evidenceBtn.className = 'mlx-variant-evidence-btn';
      evidenceBtn.textContent = '?';
      evidenceBtn.title = `Evidence for ${sib.repo_id}`;
      evidenceBtn.addEventListener('click', () => openHfEvidence(sib.repo_id, sib.revision, 'rapid_mlx', evidenceBtn));
      row.appendChild(name);
      row.appendChild(size);
      row.appendChild(evidenceBtn);
      variantsEl.appendChild(row);
    }
  }

  renderMlxSidebarVramBar();
}

// Memory estimate bar (§2.7 item 3): reuses the same #wizard-sidebar-vram-bar
// markup as the llama.cpp quant advisor, fed by a real MLX estimate for the
// currently-selected repo instead of a synthetic quant pick.
function renderMlxSidebarVramBar() {
  const availVram = effectiveAvailBytes();
  if (!availVram || !wizardState.model.modelBytes) {
    if (dom.sidebarVram) dom.sidebarVram.style.display = 'none';
    if (dom.sidebarVramBar) dom.sidebarVramBar.style.display = 'none';
    if (dom.sidebarVramLegend) dom.sidebarVramLegend.style.display = 'none';
    return;
  }
  if (dom.sidebarVram) {
    dom.sidebarVram.style.display = '';
    if (dom.sidebarVramLabel) dom.sidebarVramLabel.textContent = 'Memory estimate';
  }

  scheduleEstimate(wizardState, (est) => {
    if (!est || !est.total_bytes) {
      if (dom.sidebarVramBar) dom.sidebarVramBar.style.display = 'none';
      if (dom.sidebarVramLegend) dom.sidebarVramLegend.style.display = 'none';
      return;
    }
    const weights = est.weights_bytes || 0;
    const kv = est.kv_cache_bytes || 0;
    const overhead = (est.overhead_bytes || 0) + (est.mtp_bytes || 0) + (est.linear_attn_state_bytes || 0);
    const total = est.total_bytes || (weights + kv + overhead);
    const free = Math.max(0, availVram - total);
    const denom = availVram > 0 ? availVram : total;
    if (denom <= 0) return;

    if (dom.sidebarVramValue) {
      dom.sidebarVramValue.textContent = `${(availVram / 1e9).toFixed(1)} GB available`;
      dom.sidebarVramValue.className = `wizard-sidebar-vram-value ${total <= availVram ? 'fits' : 'nofit'}`;
    }
    if (dom.sidebarVramHint) {
      dom.sidebarVramHint.textContent = total <= availVram
        ? `Fits · ${formatGB(total)} of ${formatGB(availVram)}`
        : `Over budget · ${formatGB(total)} needed, ${formatGB(availVram)} available`;
    }

    const setW = (el, frac) => {
      if (!el) return;
      const pct = Math.max(0, Math.min(1, frac)) * 100;
      el.style.width = pct.toFixed(2) + '%';
      el.style.display = pct < 0.3 ? 'none' : '';
    };
    setW(dom.sidebarVsegWeights, weights / denom);
    setW(dom.sidebarVsegKv, kv / denom);
    setW(dom.sidebarVsegOverhead, overhead / denom);
    setW(dom.sidebarVsegFree, free / denom);

    if (dom.sidebarVlegWeights) dom.sidebarVlegWeights.textContent = `Weights ${formatGB(weights)}`;
    if (dom.sidebarVlegKv) dom.sidebarVlegKv.textContent = `KV ${formatGB(kv)}`;
    if (dom.sidebarVlegOverhead) dom.sidebarVlegOverhead.textContent = `Overhead ${formatGB(overhead)}`;

    if (dom.sidebarVramBar) {
      dom.sidebarVramBar.style.display = '';
      dom.sidebarVramBar.classList.toggle('tight', total / availVram >= 0.88 && total / availVram < 1.0);
      dom.sidebarVramBar.classList.toggle('over', total / availVram >= 1.0);
    }
    if (dom.sidebarVramLegend) dom.sidebarVramLegend.style.display = '';
  }, { force: true });
}

// ── HF discover categories: imported from hf-browse.js ────────────────────────

// ── Community picks ───────────────────────────────────────────────────────────

let communityPicksData = null;
let communityPicksActiveCat = 0;

export async function loadCommunityPicks() {
  // Attach the accordion toggle once, regardless of whether data loads.
  {
    const toggle = document.getElementById('hf-cp-toggle');
    const body = document.getElementById('hf-cp-body');
    if (toggle && body) {
      toggle.addEventListener('click', () => {
        const open = toggle.getAttribute('aria-expanded') === 'true';
        toggle.setAttribute('aria-expanded', String(!open));
        body.style.display = open ? 'none' : '';
      });
    }
  }

  try {
    const headers = window.authHeaders ? window.authHeaders() : {};
    const panel = document.getElementById('hf-community-picks');
    const list = document.getElementById('hf-cp-list');

    // Show panel + skeleton while loading
    if (panel) {
      panel.style.display = '';
      if (list) {
        list.innerHTML = '<div class="hf-cp-skeleton"><span class="hf-cp-skeleton-item"></span><span class="hf-cp-skeleton-item"></span><span class="hf-cp-skeleton-item"></span></div>';
      }
    }

    const resp = await fetch('/api/hf/community-picks', { headers });
    if (!resp.ok) {
      if (panel) panel.style.display = 'none';
      return;
    }
    const json = await resp.json();
    if (!json.ok || !json.data) {
      // No data available (missing or invalid community-picks.json)
      if (panel) panel.style.display = 'none';
      return;
    }

    communityPicksData = json.data;
    if (!panel) return;

    const cats = communityPicksData.categories || [];
    const totalModels = cats.reduce((s, c) => s + (c.models?.length || 0), 0);
    const meta = document.getElementById('hf-cp-toggle-meta');
    if (meta) {
      const gen = communityPicksData.generated_at
        ? new Date(communityPicksData.generated_at).toLocaleDateString()
        : '';
      meta.textContent = `${totalModels} models${gen ? ' · ' + gen : ''}`;
    }

    panel.style.display = '';
    renderCommunityPicksTabs(cats);
    renderCommunityPicksList(cats[0]);

    // Auto-expand on first load if the repo field is still empty — otherwise
    // the panel sits collapsed below the (empty) search results/file list and
    // is easy to miss entirely.
    const repoInput = document.getElementById('spawn-hf-repo');
    if (totalModels > 0 && (!repoInput || !repoInput.value.trim())) {
      const toggle = document.getElementById('hf-cp-toggle');
      const body = document.getElementById('hf-cp-body');
      if (toggle && body && toggle.getAttribute('aria-expanded') !== 'true') {
        toggle.setAttribute('aria-expanded', 'true');
        body.style.display = '';
      }
    }
  } catch {
    // Hide panel on unexpected errors
    const panel = document.getElementById('hf-community-picks');
    if (panel) panel.style.display = 'none';
  }
}

function renderCommunityPicksTabs(cats) {
  const tabs = document.getElementById('hf-cp-tabs');
  if (!tabs) return;
  tabs.innerHTML = '';
  cats.forEach((cat, i) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'hf-cp-tab' + (i === communityPicksActiveCat ? ' active' : '');
    btn.textContent = cat.label;
    btn.title = cat.description || '';
    btn.addEventListener('click', () => {
      communityPicksActiveCat = i;
      tabs.querySelectorAll('.hf-cp-tab').forEach((t, j) =>
        t.classList.toggle('active', j === i)
      );
      renderCommunityPicksList(cat);
    });
    tabs.appendChild(btn);
  });
}

function renderCommunityPicksList(cat) {
  const list = document.getElementById('hf-cp-list');
  if (!list) return;
  const models = cat?.models || [];

  if (!models.length) {
    const emptyDiv = document.createElement('div');
    emptyDiv.className = 'hf-cp-empty';
    const strong = document.createElement('strong');
    strong.textContent = 'No picks in this category yet';
    emptyDiv.appendChild(strong);
    emptyDiv.appendChild(
      document.createTextNode(
        'Community picks are populated by an external curation process. See docs/reference/community-picks.md for details.'
      )
    );
    list.appendChild(emptyDiv);
    return;
  }

  list.innerHTML = '';
  for (const m of models) {
    const item = document.createElement('div');
    item.className = 'hf-cp-item';
    item.tabIndex = 0;
    item.setAttribute('role', 'button');

    const sizeLabel = m.param_b > 0
      ? (m.param_b >= 1000 ? (m.param_b / 1000).toFixed(1) + 'T' : m.param_b + 'B')
      : '';

    const mainDiv = document.createElement('div');
    mainDiv.className = 'hf-cp-item-main';
    const nameDiv = document.createElement('div');
    nameDiv.className = 'hf-cp-name';
    nameDiv.textContent = m.display_name || m.hf_repo;
    mainDiv.appendChild(nameDiv);
    if (m.why) {
      const whyDiv = document.createElement('div');
      whyDiv.className = 'hf-cp-why';
      whyDiv.textContent = m.why;
      mainDiv.appendChild(whyDiv);
    }
    item.appendChild(mainDiv);

    const metaDiv = document.createElement('div');
    metaDiv.className = 'hf-cp-meta';
    const mkBadge = (cls, text) => {
      const s = document.createElement('span');
      s.className = `hf-cp-badge ${cls}`;
      s.textContent = text;
      return s;
    };
    if (sizeLabel)       metaDiv.appendChild(mkBadge('hf-cp-badge-size', sizeLabel));
    if (m.quant_rec)     metaDiv.appendChild(mkBadge('hf-cp-badge-quant', m.quant_rec));
    if (m.is_moe)        metaDiv.appendChild(mkBadge('hf-cp-badge-moe', 'MoE'));
    if (m.mention_count > 0) {
      const mentions = document.createElement('span');
      mentions.className = 'hf-cp-mentions';
      mentions.textContent = `${m.mention_count} mentions`;
      metaDiv.appendChild(mentions);
    }
    item.appendChild(metaDiv);

    const loadPick = () => {
      // Deactivate discover/quantizer pills
      document.getElementById('hf-discover-pills')
        ?.querySelectorAll('.hf-discover-pill').forEach(p => p.classList.remove('active'));
      dom.hfQuickpicks?.querySelectorAll('.hf-qp-btn').forEach(b => b.classList.remove('active'));
      // Pre-fill repo input and load files
      if (dom.hfRepoInput) dom.hfRepoInput.value = m.hf_repo;
      wizardState.model.hfRepo = m.hf_repo;
      if (m.param_b > 0) wizardState.model.paramB = m.param_b;
      if (dom.hfSearchResults) dom.hfSearchResults.style.display = 'none';
      fetchHfFiles(m.hf_repo);
      if (m.param_b > 0) triggerQuantAdvisor();
      clearValidationError();
    };
    item.addEventListener('click', loadPick);
    item.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); loadPick(); } });

    list.appendChild(item);
  }
}

// ── HF quick-picks: imported from hf-browse.js ────────────────────────────────

// ── Quantizer editor ──────────────────────────────────────────────────────────

let quantizerEditorList = []; // live copy while editor is open

export function bindQuantizerEditor() {
  const editBtn = document.getElementById('hf-qp-edit-btn');
  const editor  = document.getElementById('hf-qp-editor');
  if (!editBtn || !editor) return;

  editBtn.addEventListener('click', () => {
    const open = editBtn.getAttribute('aria-expanded') === 'true';
    editBtn.setAttribute('aria-expanded', String(!open));
    editor.style.display = open ? 'none' : '';
    if (!open) openQuantizerEditor();
  });

  document.getElementById('hf-qp-editor-add-btn')?.addEventListener('click', () => {
    const usernameEl = document.getElementById('hf-qp-editor-username');
    const displayEl  = document.getElementById('hf-qp-editor-displayname');
    const username = usernameEl?.value.trim();
    if (!username) return;
    const display_name = displayEl?.value.trim() || username;
    quantizerEditorList.push({ username, display_name, description: '', quant_style: 'standard', note: null });
    renderEditorList();
    if (usernameEl) usernameEl.value = '';
    if (displayEl)  displayEl.value = '';
  });

  document.getElementById('hf-qp-editor-save-btn')?.addEventListener('click', saveQuantizerEdits);
  document.getElementById('hf-qp-editor-reset-btn')?.addEventListener('click', resetQuantizersToDefaults);
}

async function openQuantizerEditor() {
  try {
    const headers = window.authHeaders ? window.authHeaders() : {};
    const resp = await fetch('/api/hf/quantizers', { headers });
    const data = await resp.json();
    if (data.ok && data.quantizers) {
      quantizerEditorList = data.quantizers.map(q => ({ ...q }));
      renderEditorList();
    }
  } catch {}
}

function renderEditorList() {
  const container = document.getElementById('hf-qp-editor-list');
  if (!container) return;
  container.innerHTML = '';
  for (let i = 0; i < quantizerEditorList.length; i++) {
    const q = quantizerEditorList[i];
    const row = document.createElement('div');
    row.className = 'hf-qp-editor-row';

    const styleClass = q.quant_style === 'imatrix' ? 'hf-qp-imatrix'
                     : q.quant_style === 'ud'       ? 'hf-qp-ud' : '';
    const label = document.createElement('span');
    label.className = `hf-qp-editor-name ${styleClass}`;
    label.textContent = q.display_name || q.username;
    label.title = q.username + (q.description ? '\n' + q.description : '');

    const removeBtn = document.createElement('button');
    removeBtn.type = 'button';
    removeBtn.className = 'hf-qp-editor-remove';
    removeBtn.textContent = '×';
    removeBtn.title = `Remove ${q.username}`;
    removeBtn.addEventListener('click', () => {
      quantizerEditorList.splice(i, 1);
      renderEditorList();
    });

    row.appendChild(label);
    row.appendChild(removeBtn);
    container.appendChild(row);
  }
}

async function saveQuantizerEdits() {
  const headers = window.authHeaders
    ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
    : { 'Content-Type': 'application/json' };
  try {
    const resp = await fetch('/api/hf/quantizers', {
      method: 'PUT',
      headers,
      body: JSON.stringify(quantizerEditorList),
    });
    const data = await resp.json();
    if (data.ok) {
      // Close editor and reload quick-picks
      document.getElementById('hf-qp-edit-btn')?.setAttribute('aria-expanded', 'false');
      document.getElementById('hf-qp-editor')?.style && (document.getElementById('hf-qp-editor').style.display = 'none');
      _reloadHfQuickPicks();
    }
  } catch {}
}

async function resetQuantizersToDefaults() {
  const headers = window.authHeaders
    ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
    : { 'Content-Type': 'application/json' };
  try {
    await fetch('/api/hf/quantizers', { method: 'PUT', headers, body: '[]' });
    // Reload defaults into editor
    await openQuantizerEditor();
    _reloadHfQuickPicks();
  } catch {}
}


function _reloadHfQuickPicks() {
  hfLoadQuickPicks({
    container: dom.hfQuickpicks,
    discoverPillsContainerId: 'hf-discover-pills',
    onAuthorClick: (author) => {
      browseHfAuthor(author);
    },
  });
}

async function browseHfAuthor(author) {
  wizardState.hfBrowseAuthor = author;
  hfSearchForWizard({ author, sort: hfBrowseState.sort });
}

// ── HF file listing ───────────────────────────────────────────────────────────

export function triggerHfFileFetch() {
  const input = dom.hfRepoInput?.value.trim();
  if (!input) return;

  const isRepoId = input.includes('/') && !input.includes(' ');

  if (isRepoId) {
    wizardState.model.hfRepo = input;
    if (dom.hfSearchResults) dom.hfSearchResults.style.display = 'none';
    dom.hfQuickpicks?.querySelectorAll('.hf-qp-btn').forEach(b => b.classList.remove('active'));
    // No file is selected yet at this point, so there's nothing to introspect —
    // paramB stays unset until a specific GGUF file is chosen below (Phase 10e:
    // never guess from the typed repo-id text).
    fetchHfFiles(input);
  } else {
    const sort = hfBrowseState.sort;
    hfSearchForWizard({ query: input, sort });
  }
}

// Shared Models-modal search controls plus the wizard's minimum-size filter.
function refireHfSearch() {
  const author = wizardState.hfBrowseAuthor;
  const query = dom.hfRepoInput?.value.trim() || '';
  if (author) {
    browseHfAuthor(author);
  } else if (query && !query.includes('/')) {
    hfSearchForWizard({ query, sort: hfBrowseState.sort, limit: 20 });
  } else {
    const activePill = document.querySelector('#hf-discover-pills .hf-discover-pill.active');
    if (activePill) {
      const cat = HF_DISCOVER_CATEGORIES.find(c => c.id === activePill.dataset.catId);
      if (cat) hfSearchForWizard({ ...cat.params, sort: hfBrowseState.sort });
    }
  }
}

export function bindHfSearchControls() {
  dom.hfMinSize?.addEventListener('change', refireHfSearch);
}

async function fetchHfFiles(repo) {
  if (!dom.hfFileList) return;

  // Also fetch VRAM so quant advisor has numbers
  await ensureGpuVramFetched();

  const vramGb = effectiveAvailBytes() / (1024 ** 3);

  hfListFiles({
    repoId: repo,
    container: dom.hfFileList,
    vramGb,
    onOpenCardPanel: (repoId) => openCardPanel(repoId),
    onSelectFile: (file, repoId) => {
      const fname = file.path || file.name || '';
      if (!fname) return;

      if (file.is_mmproj) {
        wizardState.model.mmprojPath = fname;
        wizardState.model.mmprojHfFile = fname;
        wizardState.model.mmprojHfRepo = repoId;
        if (file.size) wizardState.arch.mmprojBytes = Number(file.size);
        showToast('mmproj selected', 'success', fname.split('/').pop());
        dom.hfFileList.querySelectorAll('.hf-file-item.selected[data-mmproj]').forEach(el => el.classList.remove('selected'));
        const itemEl = dom.hfFileList.querySelector(`.hf-file-item[data-filename="${fname}"]`);
        if (itemEl) { itemEl.classList.add('selected'); itemEl.dataset.mmproj = '1'; }
        scheduleVramUpdate();
        return;
      }

      if (file.is_draft_assistant) {
        showToast('Draft file', 'info', 'Select a base model first — this file will be offered as the MTP draft model.');
        return;
      }

      dom.hfFileList.querySelectorAll('.hf-file-item.selected:not([data-mmproj])').forEach(el => el.classList.remove('selected'));
      const itemEl = dom.hfFileList.querySelector(`.hf-file-item[data-filename="${fname}"]`);
      if (itemEl) itemEl.classList.add('selected');

      wizardState.model.hfFile = fname;
      wizardState.model.delivery = 'stream_hf';
      wizardState.model.originRepo = repoId;
      wizardState.model.originFile = fname;
      wizardState.model.localMeta = null;
      wizardState.model.path = '';
      if (file.size) wizardState.model.modelBytes = Number(file.size);

      // Real GGUF-header introspection (Phase 10e) — no filename guessing. Fires a
      // range-fetch against the real header so paramB/MoE/MTP/gguf_arch reflect the
      // actual file, same as local doIntrospect() does for downloaded models.
      const introspectionDone = introspectHfFileMetadata(repoId, fname, file.size ? Number(file.size) : 0);

      // Store file lists so hardware step can offer quant swap + mmproj
      wizardState.model.quantFiles = [];
      wizardState.model.mmprojFiles = [];
      dom.hfFileList.querySelectorAll('.hf-file-item').forEach(el => {
        const f = {
          path: el.dataset.filename,
          name: el.dataset.filename,
          size: el.dataset.size ? Number(el.dataset.size) : 0,
          label: el.dataset.label || '',
          is_mmproj: el.dataset.mmproj === '1',
           is_draft_assistant: el.dataset.isDraftModel === '1',
          repo_id: el.dataset.repoId || repoId,
          is_recommended_mmproj: el.dataset.recommendedMmproj === '1',
          mmproj_recommendation: el.dataset.mmprojRecommendation || '',
        };
        if (f.is_mmproj) wizardState.model.mmprojFiles.push(f);
        else if (f.is_draft_assistant) {
          // Exclude mmproj files and full-size models (>3 GB) from draft head candidates.
          const fn = (f.path || '').toLowerCase();
          if (!fn.includes('mmproj') && (f.size <= 0 || f.size <= 3_000_000_000)) {
            wizardState.model.draftCandidates.push(f);
          }
        } else wizardState.model.quantFiles.push(f);
      });

      updateSelectedModelDisplay();
      clearValidationError();
      if (wizardState.model.paramB > 0) triggerQuantAdvisor();
      scheduleVramUpdate();
      autoInstallChatTemplate();
      refreshEngineRecommendation();
      refreshStepGuardrails();

      // After introspection completes, trigger quant advisor if paramB was set
      void introspectionDone.then((ok) => {
        if (ok && wizardState.model.paramB > 0) {
          triggerQuantAdvisor();
        }
      });
    },
  });
}


