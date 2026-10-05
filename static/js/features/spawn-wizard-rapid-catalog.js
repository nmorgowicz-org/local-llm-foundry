// Curated Rapid-MLX catalog picker.
//
// Rapid-MLX only behaves predictably for models upstream hand-validates (the
// `rapid-mlx models` listing): parser pairing, MTP sidecars and measured sizes
// come from their curation. Raw HF discovery for the Rapid-MLX engine produced
// exactly the failures curation exists to prevent (text-lane auto-downgrades,
// ArraysCache KV-quant refusals, unknown tool parsers), so when the Rapid-MLX
// engine is selected this picker replaces the HF browse UI entirely.
//
// Upstream's per-machine tier recommendations (`rapid-mlx recipe`) are pinned
// on top; dedicated audio/image/video pipelines are excluded from this LLM picker.

import { wizardState } from './spawn-wizard.js';

let _catalog = null;          // { models: [...], recommendations: [...] }
let _catalogGeneration = 0;
let _filterTimer = null;

function isChatModel(entry) {
  // Upstream also uses the parser/template columns for pipeline labels such
  // as audio:tts, image:gen and video:gen. Those are not chat pairings.
  const pairings = [entry.parser, entry.template]
    .filter(value => typeof value === 'string')
    .map(value => value.trim().toLowerCase());
  if (pairings.some(value => /^(audio|image|video)(:|$)/.test(value))) return false;
  return pairings.some(value => value && !['—', '-', 'none', '(none)', 'n/a'].includes(value));
}

export function filterChatModels(models) {
  return (models || []).filter(isChatModel);
}

export async function loadRapidCatalog(force = false) {
  if (_catalog && !force) return _catalog;
  const generation = ++_catalogGeneration;
  try {
    const headers = window.authHeaders ? window.authHeaders() : {};
    const resp = await fetch('/api/rapid-mlx/catalog', { headers });
    if (!resp.ok) throw new Error(`catalog unavailable (${resp.status})`);
    const data = await resp.json();
    if (generation !== _catalogGeneration) return _catalog;
    _catalog = {
      models: filterChatModels(data.models || []),
      recommendations: data.recommendations || [],
    };
  } catch (err) {
    if (generation !== _catalogGeneration) return _catalog;
    _catalog = { models: [], recommendations: [], error: err.message };
  }
  return _catalog;
}

function catalogRow(entry) {
  const row = document.createElement('button');
  row.type = 'button';
  row.className = 'rapid-catalog-row';
  row.dataset.alias = entry.name;

  const name = document.createElement('span');
  name.className = 'rapid-catalog-name';
  name.textContent = entry.name;

  const meta = document.createElement('span');
  meta.className = 'rapid-catalog-meta';
  const bits = [];
  if (entry.size_bytes) bits.push(`${(entry.size_bytes / 1024 ** 3).toFixed(1)} GB`);
  if (entry.hybrid) bits.push('hybrid');
  if (entry.parser) bits.push(entry.parser);
  meta.textContent = bits.join(' · ');

  row.append(name, meta);

  if (entry.mtp) {
    const mtp = document.createElement('span');
    mtp.className = 'rapid-catalog-badge';
    mtp.textContent = 'MTP';
    mtp.title = entry.mtp_sidecar
      ? `Pairs with speculative sidecar ${entry.mtp_sidecar}`
      : 'Supports speculative decoding';
    row.appendChild(mtp);
  }
  return row;
}

function recommendationRow(rec, modelsByName) {
  const row = catalogRow(modelsByName.get(rec.name) || { name: rec.name });
  row.classList.add('rapid-catalog-row--recommended');
  const badge = document.createElement('span');
  badge.className = 'rapid-catalog-tier';
  badge.textContent = rec.label;
  badge.title = rec.specs || `Upstream ${rec.label} pick for this Mac`;
  row.appendChild(badge);
  if (rec.cached) {
    const cached = document.createElement('span');
    cached.className = 'rapid-catalog-badge';
    cached.textContent = 'cached';
    row.appendChild(cached);
  }
  return row;
}

export function renderRapidCatalogPicker(container, { query = '' } = {}) {
  if (!container) return;
  container.replaceChildren();
  if (!_catalog) return;

  if (_catalog.error) {
    const err = document.createElement('div');
    err.className = 'rapid-catalog-empty';
    err.textContent = `Rapid-MLX catalog unavailable: ${_catalog.error}`;
    container.appendChild(err);
    return;
  }

  const modelsByName = new Map(_catalog.models.map(m => [m.name, m]));
  const q = query.trim().toLowerCase();
  const matches = (e) =>
    !q || e.name.toLowerCase().includes(q) || (e.parser || '').toLowerCase().includes(q);

  const pinned = _catalog.recommendations.filter(r => modelsByName.has(r.name) && matches(modelsByName.get(r.name)));
  const rest = _catalog.models.filter(m => !pinned.some(r => r.name === m.name) && matches(m));

  if (pinned.length) {
    const head = document.createElement('div');
    head.className = 'rapid-catalog-heading';
    head.textContent = 'Recommended for your Mac';
    container.appendChild(head);
    pinned.forEach(rec => container.appendChild(recommendationRow(rec, modelsByName)));
  }

  if (rest.length) {
    if (pinned.length) {
      const head = document.createElement('div');
      head.className = 'rapid-catalog-heading';
      head.textContent = 'All models';
      container.appendChild(head);
    }
    rest.forEach(m => container.appendChild(catalogRow(m)));
  }

  if (!pinned.length && !rest.length) {
    const empty = document.createElement('div');
    empty.className = 'rapid-catalog-empty';
    empty.textContent = 'No curated models match your search.';
    container.appendChild(empty);
  }
}

// Apply a catalog selection exactly the way an MLX HF-repo selection lands:
// the alias becomes the model identity, the rapid source is the alias kind,
// and the profile fetch + template install run from the same entry points.
function applyCatalogSelection(entry) {
  resetHfModelSelectionStateIfAvailable();
  wizardState.model.modelBytes = 0;
  wizardState.model.quantFiles = [];
  wizardState.model.originRepo = '';
  wizardState.model.originFile = '';
  wizardState.model.path = '';
  wizardState.model.hfRepo = entry.name;
  wizardState.model.rapidMlxSource = { kind: 'alias', name: entry.name };
  wizardState.model.modelBytes = entry.size_bytes || 0;
  wizardState.model.rapidCatalogEntry = entry;
  if (entry.mtp && entry.mtp_sidecar) {
    wizardState.model.rapidMlxSpeculative = { repo_id: entry.mtp_sidecar.split('@')[0] };
  }
  updateSelectedModelDisplayIfAvailable();
  autoInstallChatTemplateIfAvailable();
  clearValidationErrorIfAvailable();
  refreshAfterSelection(entry);
  document.dispatchEvent(new CustomEvent('rapid-catalog-selected', { detail: { alias: entry.name } }));
}

function resetHfModelSelectionStateIfAvailable() {
  import('./spawn-wizard-hf-browse.js').then(m => m.resetHfModelSelectionState?.()).catch(() => {});
}
function updateSelectedModelDisplayIfAvailable() {
  import('./spawn-wizard.js').then(m => m.updateSelectedModelDisplay?.()).catch(() => {});
}
function autoInstallChatTemplateIfAvailable() {
  import('./spawn-wizard-chat-template.js').then(m => m.autoInstallChatTemplate?.()).catch(() => {});
}
function clearValidationErrorIfAvailable() {
  import('./spawn-wizard.js').then(m => m.clearValidationError?.()).catch(() => {});
}
function refreshAfterSelection(entry) {
  import('./spawn-wizard-hf-browse.js').then(m => m.triggerMlxSidebarBody?.(entry.name)).catch(() => {});
  import('./spawn-wizard.js').then(m => {
    m.refreshEngineRecommendation?.();
    m.refreshStepGuardrails?.();
  }).catch(() => {});
  import('./spawn-wizard-rapid-mlx.js').then(m => m.scheduleRapidMlxProfileFetch?.(entry.name)).catch(() => {});
}

export async function mountRapidCatalogPicker(panel, { visible } = {}) {
  if (!panel) return;
  if (!visible) {
    panel.style.display = 'none';
    return;
  }
  panel.style.display = '';
  if (panel.dataset.mounted !== '1') {
    panel.dataset.mounted = '1';
    panel.innerHTML = `
      <div class="rapid-catalog-toolbar">
        <input type="search" class="rapid-catalog-search" placeholder="Search curated models…" aria-label="Search curated Rapid-MLX models">
        <button type="button" class="rapid-catalog-refresh" title="Re-query the Rapid-MLX catalog">Refresh</button>
      </div>
      <div class="rapid-catalog-list" role="listbox" aria-label="Curated Rapid-MLX models"></div>`;
    const search = panel.querySelector('.rapid-catalog-search');
    const list = panel.querySelector('.rapid-catalog-list');
    search?.addEventListener('input', () => {
      clearTimeout(_filterTimer);
      _filterTimer = setTimeout(() => renderRapidCatalogPicker(list, { query: search.value }), 120);
    });
    panel.querySelector('.rapid-catalog-refresh')?.addEventListener('click', async () => {
      await loadRapidCatalog(true);
      renderRapidCatalogPicker(list, { query: search?.value || '' });
    });
    list.addEventListener('click', (ev) => {
      const row = ev.target.closest('.rapid-catalog-row');
      if (!row) return;
      const entry = _catalog?.models.find(m => m.name === row.dataset.alias)
        || _catalog?.recommendations.filter(r => r.name === row.dataset.alias)
          .map(r => _catalog.models.find(m => m.name === r.name) || { name: r.name })[0];
      if (entry) applyCatalogSelection(entry);
    });
  }
  const list = panel.querySelector('.rapid-catalog-list');
  const loading = document.createElement('div');
  loading.className = 'rapid-catalog-empty';
  loading.textContent = 'Loading curated models…';
  list.replaceChildren(loading);
  await loadRapidCatalog(true);
  renderRapidCatalogPicker(list, { query: panel.querySelector('.rapid-catalog-search')?.value || '' });
}

export function catalogEntryFor(alias) {
  return _catalog?.models.find(m => m.name === alias) || null;
}
