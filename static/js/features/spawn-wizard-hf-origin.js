// HF-origin auto-resolve / widget / confirm cluster for the spawn wizard.
// Detects and confirms which HuggingFace repo a local model file came from.
import { wizardState, getAuthHeaders } from './spawn-wizard.js';
import { _refreshHwTagsRow, resetTagsRowOrigin } from './spawn-wizard-hf-tags.js';
import { renderHardwareModelHeader, _fetchAndShowQuantOptions } from './spawn-wizard-hardware-model.js';

const HUGGINGFACE_ORIGIN = 'https://huggingface.co';
const HF_REPO_ID_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,95}\/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$/;

function encodeRepoId(repoId) {
  if (!HF_REPO_ID_RE.test(repoId || '')) return '';
  return repoId.split('/').map((part) => encodeURIComponent(part)).join('/');
}

function safeHuggingFaceUrl(repoId) {
  const fallback = `${HUGGINGFACE_ORIGIN}/${encodeRepoId(repoId)}`;
  return fallback;
}

export let _originResolverPromise = null; // in-flight origin resolver to prevent double-fire
export let _hfOriginWidgetData = null;    // last resolve-origin response, cached for widget reuse

// Cross-module mutators: several shell call sites (bindEvents, doIntrospect,
// third-party import) reset/kick off origin resolution without owning these bindings.
export function resetOriginState() {
  _originResolverPromise = null;
  _hfOriginWidgetData = null;
}
export function startOriginResolve() {
  if (!wizardState.model.originRepo) {
    _originResolverPromise = _autoResolveHfOrigin();
  }
  return _originResolverPromise;
}
export function setOriginResolverPromise(p) {
  _originResolverPromise = p;
}
export function awaitOriginResolve(timeoutMs) {
  const resolveTimeout = new Promise(r => setTimeout(r, timeoutMs));
  return Promise.race([(_originResolverPromise || Promise.resolve()), resolveTimeout]);
}

// Attach HF origin + family tags for a local model (used by auto-resolve and
// the suggestion picker).  Replaces any stale origin/family tags.
export async function _attachOriginTags(localPath, repoId, family) {
  if (!localPath || !repoId) return;
  try {
    const headers = { ...getAuthHeaders(), 'Content-Type': 'application/json' };
    const getResp = await fetch('/api/models/tags', { headers });
    const existing = getResp.ok
      ? ((await getResp.json().catch(() => ({}))).tags?.[localPath] || [])
      : [];
    const merged = [
      ...existing.filter(t => !t.startsWith('hf_origin:') && !t.startsWith('family:')),
      `hf_origin:${repoId}`,
    ];
    if (family) merged.push(`family:${family}`);
    await fetch('/api/models/tags', {
      method: 'PUT',
      headers,
      body: JSON.stringify({ model_path: localPath, tags: merged }),
    });
  } catch { /* non-fatal */ }
}

// Auto-resolve the HF origin of a local model from its filename.
// Drives the #hf-origin-section widget with live detecting/confirmed/ambiguous/notfound states.
// Returns a promise so callers can await the resolution.
export async function _autoResolveHfOrigin() {
  const { source, path, modelBytes } = wizardState.model;
  if (source !== 'local' && source !== 'import') return;
  if (wizardState.model.originRepo) { _refreshHfOriginSection(); return; }
  // HF cache paths already encode provenance. Resolve both repository containers
  // and snapshots locally instead of sending their directory names to GGUF search.
  const cacheComponent = (path || '').split(/[\\/]/).find(part => part.startsWith('models--'));
  const cacheRepo = cacheComponent?.slice('models--'.length).replace('--', '/');
  if (cacheRepo && HF_REPO_ID_RE.test(cacheRepo)) {
    await _confirmHfOrigin(cacheRepo, '', '', path);
    _refreshHfOriginSection();
    return;
  }
  const filename = (path || '').split(/[\\/]/).pop() || '';
  if (!filename || filename.length < 8) return;

  _refreshHfOriginSection(); // show detecting state immediately

  try {
    const headers = { ...getAuthHeaders(), 'Content-Type': 'application/json' };
    const res = await fetch('/api/hf/resolve-origin', {
      method: 'POST',
      headers,
      body: JSON.stringify({ filename, size_bytes: modelBytes || 0 }),
    });
    _hfOriginWidgetData = res.ok ? await res.json() : { ok: false, candidates: [] };
    const data = _hfOriginWidgetData;

    if (data.ok && data.confident && data.candidates?.length) {
      await _confirmHfOrigin(data.candidates[0].repoId, data.candidates[0].family || '',
        data.candidates[0].cardUrl || '', path);
    }
    _refreshHfOriginSection();
  } catch {
    _hfOriginWidgetData = { ok: false, candidates: [] };
    _refreshHfOriginSection();
  }
}

// Decide which state to render for the HF origin widget and call _renderHfOriginWidget.
export function _refreshHfOriginSection() {
  const el = document.getElementById('hf-origin-section');
  if (!el) return;
  const { source, path, originRepo, cardUrl } = wizardState.model;
  const isLocal = source === 'local' || source === 'import';
  if (!isLocal || !path) { el.style.display = 'none'; return; }
  el.style.display = '';
  if (originRepo) {
    _renderHfOriginWidget(el, 'confirmed', { repoId: originRepo, cardUrl });
    return;
  }
  if (!_hfOriginWidgetData) { _renderHfOriginWidget(el, 'detecting'); return; }
  if (_hfOriginWidgetData.candidates?.length) {
    _renderHfOriginWidget(el, 'ambiguous', _hfOriginWidgetData);
  } else {
    _renderHfOriginWidget(el, 'notfound');
  }
}

// Render the HF origin widget into `el` for the given state.
function _renderHfOriginWidget(el, state, data = {}) {
  el.innerHTML = '';
  const widget = document.createElement('div');
  widget.className = 'hf-origin-widget';

  const label = document.createElement('span');
  label.className = 'hf-origin-label';
  label.textContent = 'HF Source';
  widget.appendChild(label);

  if (state === 'detecting') {
    const status = document.createElement('span');
    status.className = 'hf-origin-status';
    status.textContent = 'Detecting…';
    widget.appendChild(status);
    el.appendChild(widget);
    return;
  }

  if (state === 'confirmed') {
    const { repoId } = data;
    const slashIdx = (repoId || '').indexOf('/');
    const repo = document.createElement('span');
    repo.className = 'hf-origin-repo';
    if (slashIdx > 0) {
      const au = document.createElement('span');
      au.className = 'hf-origin-author';
      au.textContent = repoId.slice(0, slashIdx + 1);
      const nm = document.createElement('strong');
      nm.textContent = repoId.slice(slashIdx + 1);
      repo.appendChild(au);
      repo.appendChild(nm);
    } else {
      repo.textContent = repoId || '';
    }
    widget.appendChild(repo);

    const link = document.createElement('a');
    link.className = 'hf-origin-card-link';
    link.href = safeHuggingFaceUrl(repoId);
    link.target = '_blank';
    link.rel = 'noopener';
    link.textContent = '↗ Card';
    widget.appendChild(link);

    const editBtn = document.createElement('button');
    editBtn.type = 'button';
    editBtn.className = 'hf-origin-edit-btn';
    editBtn.title = 'Search for a different HuggingFace repo';
    editBtn.textContent = '✎';
    editBtn.addEventListener('click', () => _renderHfOriginWidget(el, 'search', { prefill: repoId }));
    widget.appendChild(editBtn);

    const quantDiv = document.createElement('div');
    quantDiv.id = 'hf-origin-quants-container';
    quantDiv.className = 'hf-origin-quants-container';
    widget.appendChild(quantDiv);
    el.appendChild(widget);
    // Fetch quant files async so the widget appears immediately
    _fetchAndShowQuantOptions(repoId);
    return;
  }

  if (state === 'ambiguous') {
    const { candidates } = data;
    const hint = document.createElement('span');
    hint.className = 'hf-origin-hint';
    hint.textContent = 'Multiple matches found — select the right repo:';
    widget.appendChild(hint);

    const row = document.createElement('div');
    row.className = 'hf-origin-search-row';

    const select = document.createElement('select');
    select.className = 'hf-origin-select';
    (candidates || []).forEach((c, i) => {
      const opt = document.createElement('option');
      opt.value = String(i);
      const pct = c.confidence ? ` (${Math.round(c.confidence * 100)}%)` : '';
      opt.textContent = `${c.repoId}${pct}`;
      select.appendChild(opt);
    });
    const manualOpt = document.createElement('option');
    manualOpt.value = '__search';
    manualOpt.textContent = 'Not listed — search or enter manually…';
    select.appendChild(manualOpt);
    row.appendChild(select);

    const confirmBtn = document.createElement('button');
    confirmBtn.type = 'button';
    confirmBtn.className = 'btn-wizard-secondary';
    confirmBtn.style.cssText = 'font-size:11px;padding:3px 10px;flex-shrink:0;';
    confirmBtn.textContent = 'Confirm';
    confirmBtn.addEventListener('click', async () => {
      const val = select.value;
      if (val === '__search') { _renderHfOriginWidget(el, 'search', {}); return; }
      const c = (candidates || [])[parseInt(val, 10)];
      if (!c) return;
      await _confirmHfOrigin(c.repoId, c.family || '', c.cardUrl || '', wizardState.model.path);
      _renderHfOriginWidget(el, 'confirmed', { repoId: c.repoId, cardUrl: c.cardUrl });
    });
    row.appendChild(confirmBtn);
    widget.appendChild(row);

    // Instantly switch to search if user selects the manual option
    select.addEventListener('change', () => {
      if (select.value === '__search') _renderHfOriginWidget(el, 'search', {});
    });

    const quantDiv = document.createElement('div');
    quantDiv.id = 'hf-origin-quants-container';
    quantDiv.className = 'hf-origin-quants-container';
    widget.appendChild(quantDiv);
    el.appendChild(widget);
    return;
  }

  // notfound or search — show inline search input
  const hint = document.createElement('span');
  hint.className = 'hf-origin-hint';
  hint.textContent = state === 'notfound'
    ? 'Not found automatically — enter owner/repo or search by name:'
    : 'Search for a different repo:';
  widget.appendChild(hint);

  const searchRow = document.createElement('div');
  searchRow.className = 'hf-origin-search-row';

  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'hf-origin-search-input';
  input.placeholder = 'owner/repo or search terms';
  if (data.prefill) input.value = data.prefill;
  searchRow.appendChild(input);

  const searchBtn = document.createElement('button');
  searchBtn.type = 'button';
  searchBtn.className = 'btn-wizard-secondary';
  searchBtn.style.cssText = 'font-size:11px;padding:3px 10px;flex-shrink:0;';
  searchBtn.textContent = 'Search';
  searchRow.appendChild(searchBtn);
  widget.appendChild(searchRow);

  const resultsDiv = document.createElement('div');
  resultsDiv.className = 'hf-origin-search-results';
  widget.appendChild(resultsDiv);

  const quantDiv = document.createElement('div');
  quantDiv.id = 'hf-origin-quants-container';
  quantDiv.className = 'hf-origin-quants-container';
  widget.appendChild(quantDiv);

  const doSearch = async () => {
    const q = input.value.trim();
    if (!q) return;
    // Direct owner/repo format — confirm immediately without searching
    if (/^[^/\s]+\/[^/\s]+$/.test(q)) {
      resultsDiv.innerHTML = '';
      await _confirmHfOrigin(q, '', '', wizardState.model.path);
      _renderHfOriginWidget(el, 'confirmed', { repoId: q, cardUrl: safeHuggingFaceUrl(q) });
      return;
    }
    resultsDiv.innerHTML = '';
    const spinner = document.createElement('span');
    spinner.className = 'hf-origin-status';
    spinner.style.padding = '4px 0';
    spinner.textContent = 'Searching…';
    resultsDiv.appendChild(spinner);
    try {
      const headers = { ...getAuthHeaders(), 'Content-Type': 'application/json' };
      const res = await fetch('/api/hf/resolve-origin', {
        method: 'POST',
        headers,
        body: JSON.stringify({ filename: q.endsWith('.gguf') ? q : `${q}.gguf`, size_bytes: 0 }),
      });
      const d = await res.json();
      resultsDiv.innerHTML = '';
      if (!d.ok || !d.candidates?.length) {
        const msg = document.createElement('span');
        msg.className = 'hf-origin-status';
        msg.style.padding = '4px 0';
        msg.textContent = 'No results — try different terms or type owner/repo directly.';
        resultsDiv.appendChild(msg);
        return;
      }
      d.candidates.slice(0, 8).forEach(c => {
        const resultRow = document.createElement('div');
        resultRow.className = 'hf-origin-result-row';
        const name = document.createElement('span');
        name.className = 'hf-origin-result-name';
        name.textContent = c.repoId;
        const pct = document.createElement('span');
        pct.className = 'hf-origin-result-pct';
        if (c.confidence) pct.textContent = `${Math.round(c.confidence * 100)}%`;
        resultRow.appendChild(name);
        resultRow.appendChild(pct);
        resultRow.addEventListener('click', async () => {
          await _confirmHfOrigin(c.repoId, c.family || '', c.cardUrl || '', wizardState.model.path);
          _renderHfOriginWidget(el, 'confirmed', { repoId: c.repoId, cardUrl: c.cardUrl });
        });
        resultsDiv.appendChild(resultRow);
      });
    } catch {
      resultsDiv.innerHTML = '';
      const msg = document.createElement('span');
      msg.className = 'hf-origin-status';
      msg.style.padding = '4px 0';
      msg.textContent = 'Search failed — check your connection.';
      resultsDiv.appendChild(msg);
    }
  };

  searchBtn.addEventListener('click', doSearch);
  input.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); doSearch(); } });
  el.appendChild(widget);
}

// Persist HF origin to wizardState + tags, then refresh downstream UI.
export async function _confirmHfOrigin(repoId, family, _cardUrl, path) {
  wizardState.model.originRepo = repoId;
  wizardState.model.family = family || '';
  wizardState.model.cardUrl = safeHuggingFaceUrl(repoId);
  await _attachOriginTags(path, repoId, family);
  resetTagsRowOrigin();
  _refreshHwTagsRow();
  if (wizardState.currentStep === 2) renderHardwareModelHeader();
}
