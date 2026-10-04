// Hardware-step mmproj (multimodal projector) handling: name-matching helpers
// to auto-select the best local mmproj file for a model, the mmproj dropdown
// section UI, and the "no local mmproj found" HuggingFace auto-search/manual
// fetch/download flow.
import { wizardState, scheduleVramUpdate, _modelStemForSearch, _hfFilesPost, _deriveMmprojSaveName } from './spawn-wizard.js';
import { formatBytes } from './spawn-wizard-format.js';
import { showToast } from './toast.js';

// ── Hardware step: mmproj name-matching helper ────────────────────────────────

function _mmprojQuantLabel(file) {
  if (file.label) return String(file.label).toUpperCase();
  const name = (file.path || file.name || '').toUpperCase();
  if (name.includes('BF16')) return 'BF16';
  if (/(?:^|[-_.])F16(?:[-_.]|$)/.test(name)) return 'F16';
  if (name.includes('Q8_0')) return 'Q8_0';
  if (name.includes('F32')) return 'F32';
  return '';
}

function _preferredMmprojQuant(_modelFilename = '') {
  // Family is authoritative only when supplied by model metadata/profile. Do
  // not infer a runtime capability or preferred projector from a filename.
  const family = wizardState.model.family || '';
  if (family === 'qwen3.5' || family === 'qwen3.6') return 'F16';
  if (family === 'gemma4') return 'F16';
  return '';
}

function _isRecommendedMmproj(file, modelFilename = '') {
  if (file.is_recommended_mmproj) return true;
  const preferred = _preferredMmprojQuant(modelFilename);
  return !!preferred && _mmprojQuantLabel(file) === preferred;
}

function _mmprojPracticalRank(file) {
  return { F16: 0, BF16: 1, Q8_0: 2, F32: 3 }[_mmprojQuantLabel(file)] ?? 4;
}

// Return the mmproj file whose name shares the longest common prefix (after
// stripping quant suffix and normalising to alphanumeric) with the model stem.
// Returns null if the best match is shorter than 5 normalised characters —
// that threshold prevents grabbing a completely unrelated model's mmproj.
export function _bestMmprojForModel(modelFilename, files) {
  if (!files.length) return null;
  const stem = _modelStemForSearch(modelFilename)
    .toLowerCase().replace(/[^a-z0-9]/g, '');
  if (!stem) return files.length === 1 ? files[0] : null;
  let best = null, bestScore = -1, bestRecommended = false, bestQuantRank = Infinity;
  for (const f of files) {
    const base = (f.path || f.name || '').split(/[\\/]/).pop() || '';
    const fstem = _modelStemForSearch(base)
      .toLowerCase().replace(/[^a-z0-9]/g, '');
    let score = 0;
    for (let i = 0; i < Math.min(stem.length, fstem.length); i++) {
      if (stem[i] === fstem[i]) score++;
      else break;
    }
    const recommended = _isRecommendedMmproj(f, modelFilename);
    const quantRank = _mmprojPracticalRank(f);
    if (score > bestScore
      || (score === bestScore && recommended && !bestRecommended)
      || (score === bestScore && recommended === bestRecommended && quantRank < bestQuantRank)) {
      bestScore = score;
      bestRecommended = recommended;
      bestQuantRank = quantRank;
      best = f;
    }
  }
  if (bestScore >= 5) return best;
  const recommended = files.filter(f => _isRecommendedMmproj(f, modelFilename));
  return recommended.length === 1 ? recommended[0] : null;
}

// ── Hardware step: mmproj section ────────────────────────────────────────────

export function renderMmprojSection() {
  const row = document.getElementById('hw-mmproj-row');
  if (!row) return;

  // mmproj is a llama.cpp/GGUF concept — Rapid-MLX vision uses its own MLX-VLM
  // component set and keeps the vision tower inside the native model. Showing
  // this companion-projector control under Rapid-MLX would offer a knob that
  // does nothing for that backend.
  if (wizardState.engine.selected === 'rapid_mlx') {
    row.style.display = 'none';
    return;
  }

  const files = wizardState.model.mmprojFiles || [];

  // When no local mmproj files exist, show a "download from HuggingFace" option
  // so users can fetch a companion mmproj for any model (especially ones that
  // were already downloaded without the mmproj).
  if (!files.length) {
    _renderMmprojDownloadFromHf(row);
    return;
  }
  row.style.display = '';

  // Clear any "download from HF" panel that was previously shown
  const hfPanel = row.querySelector('.hw-mmproj-hf-panel');
  if (hfPanel) hfPanel.remove();

  const select = document.getElementById('hw-mmproj-select');
  if (!select) return;
  const modelFilename = (wizardState.model.path || wizardState.model.hfFile || '')
    .split(/[\\/]/).pop() || '';
  const populationKey = `${files.length}:${_preferredMmprojQuant(modelFilename)}`;

  // Re-populate if the file list changed (e.g. after a companion download)
  if (select.dataset.populated !== populationKey) {
    select.dataset.populated = populationKey;
    select.innerHTML = '';
    const noneOpt = document.createElement('option');
    noneOpt.value = ''; noneOpt.textContent = '(none — text-only)';
    select.appendChild(noneOpt);
    files.forEach(f => {
      const fpath = f.path || f.name || '';
      const fname = fpath.split('/').pop();
      const opt = document.createElement('option');
      opt.value = fpath;
      const sizeStr = f.size ? ` · ${formatBytes(f.size)}` : '';
      const recommended = _isRecommendedMmproj(f, modelFilename);
      opt.textContent = fname + sizeStr + (recommended ? ' · Recommended' : '');
      if (recommended) {
        opt.title = f.mmproj_recommendation || `${_mmprojQuantLabel(f)} is preferred for this model family`;
      }
      if (fpath === (wizardState.model.mmprojPath || wizardState.model.mmprojHfFile)) opt.selected = true;
      select.appendChild(opt);
    });
    select.addEventListener('change', () => {
      const fpath = select.value;
      wizardState.model.mmprojHfFile = fpath;
      wizardState.model.mmprojPath = fpath;
      const f = files.find(x => (x.path || x.name) === fpath);
      wizardState.model.mmprojHfRepo =
        f?.repo_id || wizardState.model.originRepo || wizardState.model.hfRepo || '';
      wizardState.arch.mmprojBytes = f?.size ? Number(f.size) : 0;
      scheduleVramUpdate();
    });
  }

  // Sync selection state
  const active = wizardState.model.mmprojPath || wizardState.model.mmprojHfFile;
  if (active) select.value = active;

  // Auto-select using name proximity to the model file rather than
  // alphabetical order — avoids grabbing a different model's mmproj.
  if (!select.value && files.length) {
    const best = _bestMmprojForModel(modelFilename, files);
    if (best) {
      const bestPath = best.path || best.name || '';
      select.value = bestPath;
      wizardState.model.mmprojPath = bestPath;
      wizardState.model.mmprojHfFile = bestPath;
      wizardState.model.mmprojHfRepo =
        best.repo_id || wizardState.model.originRepo || wizardState.model.hfRepo || '';
      wizardState.arch.mmprojBytes = best.size ? Number(best.size) : 0;
      scheduleVramUpdate();
    }
  }
}

// Search HF for the first GGUF repo that contains an mmproj file matching this model.
// Returns {repoId, mmprojFiles} or null.
async function _autoFindMmprojRepo(modelFilename) {
  const stem = _modelStemForSearch(modelFilename);
  if (!stem) return null;
  const headers = window.authHeaders ? window.authHeaders() : {};

  // Try progressively broader queries: exact stem, stem + GGUF keyword,
  // then a shorter version without minor version/variant suffixes.
  const shorter = stem
    .replace(/-v\d+(?:\.\d+)?(?:-[A-Za-z]+)*$/i, '') // strip -v2-MTP etc.
    .replace(/-MTP$/i, '');
  const queries = [...new Set([stem, stem + ' GGUF', shorter])].filter(Boolean);

  for (const query of queries) {
    try {
      const searchRes = await fetch('/api/hf/search', {
        method: 'POST',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, sort: 'downloads', limit: 10 }),
      });
      if (!searchRes.ok) continue;
      const searchData = await searchRes.json();
      if (!searchData.ok || !searchData.models?.length) continue;

      for (const model of searchData.models) {
        const filesData = await _hfFilesPost(model.id);
        if (!filesData?.ok) continue;
        const mmprojFiles = (filesData.files || []).filter(f => f.is_mmproj);
        if (mmprojFiles.length > 0) {
          return { repoId: mmprojFiles[0].repo_id || model.id, mmprojFiles };
        }
      }
    } catch { continue; }
  }
  return null;
}

function _renderMmprojDownloadFromHf(row) {
  // Show the row with a "download mmproj from HuggingFace" mini-panel
  row.style.display = '';
  const select = document.getElementById('hw-mmproj-select');
  if (select) select.style.display = 'none';

  // Remove stale panel so auto-search re-runs if user navigates back and forward
  row.querySelector('.hw-mmproj-hf-panel')?.remove();

  const panel = document.createElement('div');
  panel.className = 'hw-mmproj-hf-panel';
  row.appendChild(panel);

  const originRepo = wizardState.model.originRepo || '';
  const modelFilename = (wizardState.model.path || wizardState.model.hfFile || '')
    .split(/[\\/]/).pop() || '';

  if (originRepo) {
    // Already know the repo — go straight to the fetch form, which auto-fetches
    _showMmprojHfFetchForm(row, panel);
    return;
  }

  // No originRepo — try to auto-find from model filename
  panel.innerHTML = `
    <span class="hw-quant-label" style="color:var(--color-text-muted);font-size:10px;">
      No mmproj found. Searching HuggingFace…
    </span>
  `;

  _autoFindMmprojRepo(modelFilename).then(result => {
    panel.innerHTML = '';

    if (result) {
      // Auto-found a repo with mmproj — show it for one-click download
      const statusEl = document.createElement('div');
      statusEl.style.cssText = 'font-size:10px;color:var(--color-text-muted);margin-top:4px;';

      const repoLabel = document.createElement('span');
      repoLabel.className = 'hw-quant-label';
      repoLabel.style.cssText = 'font-size:10px;color:var(--color-text-muted);';
      repoLabel.textContent = `Found: ${result.repoId}`;
      panel.appendChild(repoLabel);

      const listEl = document.createElement('div');
      listEl.style.cssText = 'display:flex;flex-direction:column;gap:4px;margin-top:6px;';
      result.mmprojFiles.forEach(f => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'btn-wizard-secondary';
        btn.style.cssText = 'min-height:26px;padding:4px 10px;font-size:10px;text-align:left;';
        const fname = (f.rfilename || f.path || '').split('/').pop();
        const sizeStr = f.size ? ` · ${formatBytes(f.size)}` : '';
        const recommended = _isRecommendedMmproj(f, modelFilename);
        btn.textContent = `⬇ ${fname}${sizeStr}${recommended ? ' · Recommended' : ''}`;
        if (recommended) btn.title = f.mmproj_recommendation || 'Preferred projector format for this model family';
        btn.addEventListener('click', () => _downloadMmprojFromHf(result.repoId, f, wizardState.model.path || wizardState.model.hfFile, statusEl));
        listEl.appendChild(btn);
      });
      panel.appendChild(listEl);
      panel.appendChild(statusEl);

      const manualLink = document.createElement('a');
      manualLink.href = '#';
      manualLink.style.cssText = 'font-size:10px;color:var(--color-text-muted);margin-top:6px;display:inline-block;';
      manualLink.textContent = 'Wrong repo? Search manually…';
      manualLink.addEventListener('click', e => {
        e.preventDefault();
        panel.innerHTML = '';
        _showMmprojHfFetchForm(row, panel);
      });
      panel.appendChild(manualLink);

      const browseLocalLink = document.createElement('a');
      browseLocalLink.href = '#';
      browseLocalLink.style.cssText = 'font-size:10px;color:var(--color-text-muted);margin-top:2px;display:inline-block;';
      browseLocalLink.textContent = 'Or browse local files…';
      browseLocalLink.addEventListener('click', e => {
        e.preventDefault();
        document.getElementById('hw-mmproj-browse-btn')?.click();
      });
      panel.appendChild(browseLocalLink);
    } else {
      // Auto-find failed — show manual form with the stem pre-filled and a note
      const stem = _modelStemForSearch(modelFilename);
      _showMmprojHfFetchForm(row, panel, stem);
    }
  });
}

function _showMmprojHfFetchForm(row, panel, prefill = '') {
  const originRepo = wizardState.model.originRepo || '';
  const initialValue = originRepo || prefill;
  const showNotFound = !originRepo && prefill;

  // eslint-disable-next-line no-unsanitized/property -- static HTML, no user data
  panel.innerHTML = `
    ${showNotFound ? `<div style="font-size:10px;color:var(--color-text-muted);margin-bottom:6px;">Couldn't auto-find it — enter the HuggingFace repo that contains the mmproj:</div>` : ''}
    <div style="display:flex;gap:6px;align-items:center;width:100%;flex-wrap:wrap;">
      <input type="text" class="hw-mmproj-repo-input" placeholder="owner/repo (e.g. unsloth/Qwen3-VL-7B-GGUF)"
        style="flex:1;min-width:120px;padding:6px 10px;border-radius:6px;border:1px solid rgba(255,255,255,0.1);
          background:rgba(28,34,42,0.9);color:var(--color-text-primary);font-size:11px;">
      <button type="button" class="btn-wizard-secondary hw-mmproj-repo-go" style="min-height:28px;padding:5px 10px;font-size:11px;">Fetch</button>
      <button type="button" class="btn-wizard-tertiary hw-mmproj-cancel" style="font-size:10px;">Cancel</button>
    </div>
    <div class="hw-mmproj-repo-status" style="font-size:10px;color:var(--color-text-muted);margin-top:4px;"></div>
    <div class="hw-mmproj-repo-list" style="display:none;flex-direction:column;gap:4px;margin-top:6px;max-height:120px;overflow-y:auto;"></div>
  `;

  const input = panel.querySelector('.hw-mmproj-repo-input');
  if (initialValue) input.value = initialValue;

  panel.querySelector('.hw-mmproj-cancel').addEventListener('click', () => {
    panel.remove();
    if (row.querySelector('#hw-mmproj-select')) {
      row.querySelector('#hw-mmproj-select').style.display = '';
    }
    row.style.display = 'none';
  });

  const goBtn = panel.querySelector('.hw-mmproj-repo-go');
  const statusEl = panel.querySelector('.hw-mmproj-repo-status');
  const listEl = panel.querySelector('.hw-mmproj-repo-list');

  async function doFetch() {
    const repoId = input.value.trim();
    if (!repoId) return;
    goBtn.disabled = true; statusEl.textContent = 'Fetching files…';
    const data = await _hfFilesPost(repoId);
    goBtn.disabled = false;
    if (!data?.ok || !data.files?.length) {
      statusEl.textContent = 'No GGUF files found. Check the repo ID.'; return;
    }
    const mmprojFiles = data.files.filter(f => f.is_mmproj);
    if (!mmprojFiles.length) {
      statusEl.textContent = 'No mmproj file found in this repo.'; return;
    }
    statusEl.textContent = '';
    listEl.style.display = 'flex';
    listEl.innerHTML = '';
    mmprojFiles.forEach(f => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'btn-wizard-secondary';
      btn.style.cssText = 'min-height:26px;padding:4px 10px;font-size:10px;text-align:left;';
      const fname = (f.rfilename || f.path || '').split('/').pop();
      const sizeStr = f.size ? ` · ${formatBytes(f.size)}` : '';
      const modelFilename = (wizardState.model.path || wizardState.model.hfFile || '')
        .split(/[\\/]/).pop() || '';
      const recommended = _isRecommendedMmproj(f, modelFilename);
      btn.textContent = `⬇ ${fname}${sizeStr}${recommended ? ' · Recommended' : ''}`;
      if (recommended) btn.title = f.mmproj_recommendation || 'Preferred projector format for this model family';
      btn.addEventListener('click', () => _downloadMmprojFromHf(
        f.repo_id || repoId,
        f,
        wizardState.model.path || wizardState.model.hfFile,
        statusEl
      ));
      listEl.appendChild(btn);
    });
  }

  goBtn.addEventListener('click', doFetch);
  input.addEventListener('keydown', e => { if (e.key === 'Enter') doFetch(); });
  // Auto-fetch when we already know the exact repo (originRepo), not for search hints
  if (originRepo) doFetch();
}

async function _downloadMmprojFromHf(repoId, file, modelPath, statusEl) {
  const mmprojHfPath = file.rfilename || file.path || '';
  const modelFilename = (modelPath || '').split(/[\\/]/).pop() || '';
  const saveAs = modelFilename ? _deriveMmprojSaveName(modelFilename, mmprojHfPath) : mmprojHfPath.split('/').pop();
  if (statusEl) statusEl.textContent = `Downloading ${saveAs}…`;
  try {
    const headers = window.authHeaders
      ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
      : { 'Content-Type': 'application/json' };
    const res = await fetch('/api/hf/download', {
      method: 'POST',
      headers,
      body: JSON.stringify({ repo_id: repoId, file_path: mmprojHfPath, save_as: saveAs, companion: true, resume: true }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || !data.ok) {
      if (statusEl) statusEl.textContent = data.error || 'Download failed to start.';
      return;
    }
    // Poll until complete, then update mmproj state
    _pollMmprojDownload(data.download_id, data.local_path, file.size || 0, statusEl);
  } catch (e) {
    if (statusEl) statusEl.textContent = `Error: ${e.message}`;
  }
}

function _pollMmprojDownload(downloadId, localPath, expectedSize, statusEl) {
  const headers = window.authHeaders ? window.authHeaders() : {};
  async function poll() {
    try {
      const res = await fetch(`/api/models/download/${downloadId}/status`, { headers });
      if (!res.ok) { setTimeout(poll, 1000); return; }
      const data = await res.json();
      const s = data.status;
      if (!s) { setTimeout(poll, 1000); return; }
      const { status, bytes_downloaded = 0, total_bytes = 0 } = s;
      const pct = total_bytes > 0 ? Math.round(bytes_downloaded / total_bytes * 100) : 0;
      if (status === 'running') {
        if (statusEl) statusEl.textContent = `Downloading mmproj… ${pct}%`;
        setTimeout(poll, 1000);
        return;
      }
      if (status === 'completed') {
        const mmprojName = localPath.split(/[\\/]/).pop() || localPath;
        wizardState.model.mmprojPath = localPath;
        wizardState.model.mmprojHfFile = localPath;
        wizardState.model.mmprojFiles = [{
          path: localPath, name: mmprojName,
          size: expectedSize || 0, is_mmproj: true,
        }];
        wizardState.arch.mmprojBytes = expectedSize || 0;
        if (statusEl) statusEl.textContent = '';
        // Re-render mmproj section to show the new file in the dropdown
        renderMmprojSection();
        scheduleVramUpdate();
        showToast('mmproj downloaded', 'success', mmprojName);
      } else if (status === 'failed') {
        if (statusEl) statusEl.textContent = s.message || 'Download failed.';
      }
    } catch { setTimeout(poll, 1000); }
  }
  setTimeout(poll, 800);
}
