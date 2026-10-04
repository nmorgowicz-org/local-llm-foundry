// Hardware-step model header (repo/quant display, tag row trigger) and the
// local-model quant-swap discovery flow (auto-search HF for sibling quants,
// candidate-list/manual-input fallbacks, and the resulting download/stream
// action row).
import { wizardState, effectiveAvailBytes, refreshEngineRecommendation, scheduleVramUpdate, _hfFilesPost, _extractQuantLabel } from './spawn-wizard.js';
import { formatBytes } from './spawn-wizard-format.js';
import { _openHwRepoEditor, _refreshHwTagsRow } from './spawn-wizard-hf-tags.js';
import { _attachOriginTags } from './spawn-wizard-hf-origin.js';
import { showToast } from './toast.js';
import { hfStartDownload, hfShowDownloadPanel } from './hf-browse.js';

function getRecommendedQuant(vramGb) {
  if (vramGb < 8)  return 'Q4_K_M';
  if (vramGb <= 16) return 'Q5_K_M';
  if (vramGb <= 24) return 'Q5_K_M';
  return 'Q8_0';
}


// Fetch quant file list from HF for the confirmed repo and show an expandable list.
// Also populates wizardState.model.quantFiles so step 2 shows the dropdown directly.
export async function _fetchAndShowQuantOptions(repoId) {
  const container = document.getElementById('hf-origin-quants-container');
  if (!container || !repoId) return;
  const data = await _hfFilesPost(repoId);
  if (!data?.ok) return;

  const currentFilename = (wizardState.model.path || '').split(/[\\/]/).pop().toLowerCase();
  const ggufFiles = (data.files || []).filter(f =>
    !f.is_mmproj && (f.rfilename || f.path || '').toLowerCase().endsWith('.gguf')
  );
  if (!ggufFiles.length) return;

  // Populate quantFiles so step 2 shows the select dropdown without re-searching
  wizardState.model.quantFiles = ggufFiles.map(f => ({
    path: f.rfilename || f.path || '',
    name: f.rfilename || f.path || '',
    size: f.size || 0,
    label: _extractQuantLabel(f.rfilename || f.path || ''),
  }));
  wizardState.model._quantSwapRepo = repoId;

  const others = ggufFiles.filter(f =>
    (f.rfilename || f.path || '').split('/').pop().toLowerCase() !== currentFilename
  );

  if (!others.length) {
    const msg = document.createElement('span');
    msg.className = 'hf-origin-quants-toggle';
    msg.style.cursor = 'default';
    msg.textContent = 'Only this quantization available in this repo';
    container.appendChild(msg);
    return;
  }

  const toggleBtn = document.createElement('button');
  toggleBtn.type = 'button';
  toggleBtn.className = 'hf-origin-quants-toggle';
  toggleBtn.textContent = `▸ ${others.length} other quant${others.length !== 1 ? 's' : ''} available`;

  const list = document.createElement('div');
  list.className = 'hf-origin-quants-list';

  others.forEach(f => {
    const fname = (f.rfilename || f.path || '').split('/').pop();
    const quantLabel = _extractQuantLabel(fname);
    const sizeGb = f.size ? ` · ${(f.size / 1073741824).toFixed(1)} GB` : '';
    const item = document.createElement('div');
    item.className = 'hf-origin-quant-item';
    item.textContent = `${quantLabel}${sizeGb}`;
    list.appendChild(item);
  });

  toggleBtn.addEventListener('click', () => {
    const open = list.style.display !== 'none';
    list.style.display = open ? 'none' : 'flex';
    toggleBtn.textContent = `${open ? '▸' : '▾'} ${others.length} other quant${others.length !== 1 ? 's' : ''} available`;
  });

  container.appendChild(toggleBtn);
  container.appendChild(list);
}

const USE_CASE_LABELS = { agentic: 'Agentic / tool-calling', general: 'General chat', roleplay: 'Roleplay / creative' };

// Sticky context bar (plan §2): compact loader/context/KV-precision/use-case
// summary rendered into the sticky #hw-model-header so it stays visible while
// the rest of step 2 scrolls. Reads the same DOM the fields themselves write
// to rather than duplicating state.
export function renderContextChipRow() {
  const row = document.getElementById('hw-context-chip-row');
  if (!row) return;
  const rapid = wizardState.engine.selected === 'rapid_mlx';
  const loaderLabel = rapid ? 'Rapid-MLX' : 'llama.cpp';
  const ctxVal = document.getElementById('spawn-context-size')?.value || '';
  const kvLabel = rapid
    ? 'int8 (pinned)'
    : (document.getElementById('spawn-cache-type-k')?.value || 'f16');
  const useCaseLabel = USE_CASE_LABELS[wizardState.useCase] || wizardState.useCase;

  const chips = [
    { label: 'Loader', value: loaderLabel },
    { label: 'Use case', value: useCaseLabel },
  ];
  if (ctxVal) chips.push({ label: 'Context', value: ctxVal });
  chips.push({ label: 'KV', value: kvLabel });

  row.innerHTML = '';
  chips.forEach(({ label, value }) => {
    const chip = document.createElement('span');
    chip.className = 'hw-context-chip';
    chip.append(`${label}: `);
    const strong = document.createElement('strong');
    strong.textContent = value;
    chip.appendChild(strong);
    row.appendChild(chip);
  });
}

export function renderHardwareModelHeader() {
  const header = document.getElementById('hw-model-header');
  if (!header) return;
  const { source, path, hfRepo, hfFile, quantFiles } = wizardState.model;
  if (!hfRepo && !path) { header.style.display = 'none'; return; }
  header.style.display = '';

  const repoEl = document.getElementById('hw-model-repo');
  if (repoEl) {
    // Remove any previous inline editor
    repoEl.classList.remove('hw-model-repo-editing');
    repoEl.innerHTML = '';
    repoEl.style.cursor = 'default';

    const fullRepo = hfRepo || (wizardState.model.originRepo || '');
    const displayName = fullRepo || path.split(/[/\\]/).pop() || path;
    const slashIdx = fullRepo ? fullRepo.indexOf('/') : -1;

    if (fullRepo && slashIdx > 0) {
      const author = fullRepo.slice(0, slashIdx + 1);
      const modelName = fullRepo.slice(slashIdx + 1);
      const authorSpan = document.createElement('span');
      authorSpan.className = 'hw-model-author';
      authorSpan.textContent = author;
      const nameSpan = document.createElement('span');
      nameSpan.className = 'hw-model-name';
      nameSpan.textContent = modelName;
      repoEl.appendChild(authorSpan);
      repoEl.appendChild(nameSpan);
    } else {
      repoEl.textContent = displayName;
    }

    // Add a subtle change button so the user can alter the HF repo.
    const changeBtn = document.createElement('span');
    changeBtn.className = 'hw-model-repo-change';
    changeBtn.textContent = '✎';
    changeBtn.title = 'Change HuggingFace repo';
    changeBtn.style.cssText =
      'margin-left:6px;font-size:10px;cursor:pointer;opacity:0.35;';
    changeBtn.addEventListener('mouseenter', () => { changeBtn.style.opacity = '1'; });
    changeBtn.addEventListener('mouseleave', () => { changeBtn.style.opacity = '0.35'; });
    changeBtn.addEventListener('click', () => {
      _openHwRepoEditor(repoEl, fullRepo || '');
    });
    repoEl.appendChild(changeBtn);
  }

  const quantRow = document.getElementById('hw-quant-row');
  const quantSelect = document.getElementById('hw-quant-select');
  const vramGb = effectiveAvailBytes() / (1024 ** 3);

  if (quantSelect && quantFiles && quantFiles.length > 1) {
    quantSelect.innerHTML = '';
    const loadedBasename = (wizardState.model.path || '').split(/[\\/]/).pop().toLowerCase();
    let matched = false;
    let recOpt = null;

    quantFiles.forEach(qf => {
      const fpath = qf.path || qf.name || '';
      const fname = fpath.split('/').pop();
      if (!fname) return;
      const opt = document.createElement('option');
      opt.value = fpath;
      const dispLabel = qf.label || fname;
      const sizeStr = qf.size ? ` · ${formatBytes(qf.size)}` : '';
      const isRec = qf.label && vramGb > 0 && qf.label === getRecommendedQuant(vramGb);
      opt.textContent = dispLabel + sizeStr + (isRec ? ' ★' : '');

      // Primary: match exact HF path
      if (fpath === hfFile) { opt.selected = true; matched = true; }
      // Secondary: local file basename matches
      else if (!matched && loadedBasename && fname.toLowerCase() === loadedBasename) {
        opt.selected = true; matched = true;
      }

      if (!recOpt && isRec) recOpt = opt;
      quantSelect.appendChild(opt);
    });

    // Tertiary: if still nothing selected, pick the VRAM-appropriate recommended quant
    if (!matched && recOpt) { recOpt.selected = true; }

    // For local/import models, only show the quant selector when the user explicitly
    // picked a swap repo (via "Find other quantizations…"). For HF models, always show
    // it so the user can pick which quant to download/stream.
    const isLocalSource = source === 'local' || source === 'import';
    const hasSwapRepo = !!wizardState.model._quantSwapRepo;
    if (quantRow) quantRow.style.display = (isLocalSource && !hasSwapRepo) ? 'none' : '';
  } else {
    if (quantRow) quantRow.style.display = 'none';
    const fileEl = document.getElementById('hw-model-file');
    if (fileEl) fileEl.textContent = hfFile ? hfFile.split('/').pop() : (path.split(/[/\\]/).pop() || '');
  }

  // Library tags row — refresh whenever origin is known (non-blocking async).
  _refreshHwTagsRow();

  // "Find other quantizations…" row: show for local models that haven't selected a swap
  // repo yet. Once the user picks a repo, hide this row and show the quant dropdown instead.
  const localRow = document.getElementById('hw-quant-local-row');
  if (localRow) {
    const isLocal = source === 'local' || source === 'import';
    const hasSwapRepo = !!wizardState.model._quantSwapRepo;
    localRow.style.display = (isLocal && !hasSwapRepo) ? '' : 'none';
  }

  // Always clear the download/stream actions bar when re-rendering the header.
  // The hw-quant-select onChange handler will re-populate it if the user picks a
  // different quant — it must not persist from a previous wizard session.
  const actionsRow = document.getElementById('hw-quant-swap-actions');
  if (actionsRow) actionsRow.style.display = 'none';

  renderContextChipRow();
}

let _lastQuantSearchFile = ''; // prevent redundant searches
let _quantSwapSearching = false;

export function resetQuantSwapSearchState() {
  _lastQuantSearchFile = '';
  _quantSwapSearching = false;
}

// userTriggered = true: user clicked "Find other quantizations…"; show dropdown on success.
// userTriggered = false (default): background auto-discover; only populate files + status hint,
//   never auto-show the dropdown (sets quantFiles without setting _quantSwapRepo).
export async function _autoDiscoverLocalModelQuants(userTriggered = false) {
  if (_quantSwapSearching) return;
  const { source, path, originRepo } = wizardState.model;
  if (source !== 'local' && source !== 'import') return;

  const filename = (path || '').split(/[\\/]/).pop() || '';
  if (!filename || filename === _lastQuantSearchFile) return;

  // Quants already loaded: if user-triggered, open the dropdown now.
  // If background auto-trigger, just refresh the header without changing visibility.
  if (wizardState.model.quantFiles?.length > 0) {
    _lastQuantSearchFile = filename;
    if (userTriggered && !wizardState.model._quantSwapRepo) {
      wizardState.model._quantSwapRepo = wizardState.model.originRepo || '_local_quants_';
    }
    renderHardwareModelHeader();
    return;
  }

  _lastQuantSearchFile = filename;
  _quantSwapSearching = true;

  const statusEl = document.getElementById('hw-quant-local-status');
  const btn = document.getElementById('hw-quant-local-btn');
  if (statusEl) statusEl.textContent = 'Searching HuggingFace…';
  if (btn) btn.disabled = true;

  try {
        let repoId = originRepo || '';
        let rawFiles = [];
        let showCandidateList = false;

        // 1) If originRepo is already known (e.g., from pencil editor), use it.
        //    Even a single GGUF is acceptable since this is the confirmed source.
        if (repoId) {
          const data = await _hfFilesPost(repoId);
          if (data?.ok) {
            rawFiles = (data.files || []).filter(f =>
              !f.is_mmproj && (f.rfilename || f.path || '').toLowerCase().endsWith('.gguf'));
          }
        }

        // 2) If no originRepo or it has no GGUFs, use resolve-origin to search.
        if (!rawFiles.length && !repoId) {
          const headers = window.authHeaders ? { ...window.authHeaders(), 'Content-Type': 'application/json' } : { 'Content-Type': 'application/json' };
          const modelBytes = wizardState.model.modelBytes || 0;
          const res = await fetch('/api/hf/resolve-origin', {
            method: 'POST',
            headers,
            body: JSON.stringify({ filename, size_bytes: modelBytes }),
          });

          if (res.ok) {
            const data = await res.json();
            const candidates = (data.candidates || [])
              .slice(0, 5)
              .map(c => ({ repoId: c.repoId, confidence: c.confidence }));

            if (data.confident && candidates.length > 0) {
              // Confident match: use top candidate
              repoId = candidates[0].repoId;
              const fd = await _hfFilesPost(repoId);
              if (fd?.ok) {
                rawFiles = (fd.files || []).filter(f =>
                  !f.is_mmproj && (f.rfilename || f.path || '').toLowerCase().endsWith('.gguf'));
              }
            } else if (candidates.length > 1) {
              // Multiple plausible repos: show list of those with ≥1 GGUF
              const withFiles = [];
              for (const c of candidates) {
                const fd = await _hfFilesPost(c.repoId);
                if (!fd?.ok) continue;
                const gguf = (fd.files || []).filter(f =>
                  !f.is_mmproj && (f.rfilename || f.path || '').toLowerCase().endsWith('.gguf'));
                if (gguf.length >= 1) {
                  withFiles.push({ repoId: c.repoId, ggufFiles: gguf });
                }
              }
              if (withFiles.length > 0) {
                if (statusEl) statusEl.textContent = 'Multiple possible sources found';
                _showQuantSwapCandidateList(withFiles);
                showCandidateList = true;
              }
            } else if (candidates.length === 1) {
              // Single candidate with ≥1 GGUF
              const c = candidates[0];
              const fd = await _hfFilesPost(c.repoId);
              if (fd?.ok) {
                const gguf = (fd.files || []).filter(f =>
                  !f.is_mmproj && (f.rfilename || f.path || '').toLowerCase().endsWith('.gguf'));
                if (gguf.length >= 1) {
                  repoId = c.repoId;
                  rawFiles = gguf;
                }
              }
            }
          }
        }

      if (showCandidateList) {
        return;
      }

      // 3) If we have any GGUFs from the chosen repo, use it.
      if (rawFiles.length > 0 && repoId) {
        wizardState.model.quantFiles = rawFiles.map(f => ({
          path: f.rfilename || f.path || '',
          name: f.rfilename || f.path || '',
          size: f.size || 0,
          label: _extractQuantLabel(f.rfilename || f.path || ''),
        }));

        // Pre-select the entry matching the currently loaded local file.
        const currentLower = filename.toLowerCase();
        const match = rawFiles.find(f =>
          (f.rfilename || f.path || '').split('/').pop().toLowerCase() === currentLower);
        if (match) wizardState.model.hfFile = match.rfilename || match.path || '';
        refreshEngineRecommendation();

        if (userTriggered) {
          // User explicitly asked: open the dropdown immediately.
          wizardState.model._quantSwapRepo = repoId;
          if (statusEl) statusEl.textContent = '';
        } else {
          // Background auto-discover: don't show the dropdown yet.
          // Record the repo on originRepo so the click handler can use it without re-searching.
          if (!wizardState.model.originRepo) wizardState.model.originRepo = repoId;
          const n = rawFiles.length;
          if (statusEl) statusEl.textContent = n === 1
            ? 'No other quants available'
            : `${n} quants available — click to switch`;
          // Don't clear the status — user needs to see it to know they can click
        }
        renderHardwareModelHeader();
      } else {
        // 4) Fallback: let user type a repo manually.
        if (statusEl) statusEl.textContent = 'Not found — type or paste the repo:';
        _showQuantSwapManualInput();
      }
   } catch {
     if (statusEl) statusEl.textContent = 'Search failed';
   } finally {
     _quantSwapSearching = false;
     if (btn) btn.disabled = false;
   }
}

// Show a compact list of candidate repos when multiple are found.
// Includes a "Not this one?" option to let user type manually.
// No auto-select: user must choose explicitly.
function _showQuantSwapCandidateList(candidates) {
  const row = document.getElementById('hw-quant-local-row');
  if (!row) return;
  const btn = row.querySelector('#hw-quant-local-btn');
  if (!btn) return;

  // Clear existing controls in this row and replace with list.
  row.innerHTML = '';
  row.style.display = '';

  const listWrap = document.createElement('div');
  listWrap.style.cssText =
    'display:flex;flex-direction:column;gap:4px;margin-top:4px;min-width:0;';

  const selectCandidate = (candidate) => {
    wizardState.model.quantFiles = candidate.ggufFiles.map(f => ({
      path: f.rfilename || f.path || '',
      name: f.rfilename || f.path || '',
      size: f.size || 0,
      label: _extractQuantLabel(f.rfilename || f.path || ''),
    }));
    wizardState.model._quantSwapRepo = candidate.repoId;

    // Pre-select the entry matching the currently loaded local file.
    const filename = (wizardState.model.path || '').split(/[\\/]/).pop() || '';
    const currentLower = filename.toLowerCase();
    const match = candidate.ggufFiles.find(f =>
      (f.rfilename || f.path || '').split('/').pop().toLowerCase() === currentLower);
    if (match) wizardState.model.hfFile = match.rfilename || match.path || '';
    refreshEngineRecommendation();

    const statusEl = document.getElementById('hw-quant-local-status');
    if (statusEl) statusEl.textContent = `${candidate.ggufFiles.length} quants selected`;
    setTimeout(() => { if (statusEl) statusEl.textContent = ''; }, 3000);
    renderHardwareModelHeader();
  };

  // Render candidate options (no auto-select).
  candidates.forEach((candidate, _index) => {
    const item = document.createElement('div');
    item.style.cssText =
      'display:flex;justify-content:space-between;align-items:center;padding:4px 6px;' +
      'border-radius:4px;border:1px solid rgba(255,255,255,0.1);cursor:pointer;' +
      'background:rgba(15,23,42,0.6);font-size:10px;';

    const repoText = document.createElement('span');
    repoText.style.cssText = 'flex:1;min-width:0;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;';
    repoText.textContent = candidate.repoId;

    const meta = document.createElement('span');
    meta.style.cssText =
      'margin-left:6px;font-size:9px;color:var(--color-text-muted);flex-shrink:0;';
    meta.textContent = `${candidate.ggufFiles.length} GGUFs`;

    item.appendChild(repoText);
    item.appendChild(meta);

    item.addEventListener('click', () => {
      selectCandidate(candidate);
    });

    listWrap.appendChild(item);
  });

  // "Not this one?" option → manual input.
  const notThisOne = document.createElement('div');
  notThisOne.style.cssText =
    'margin-top:2px;font-size:9px;color:var(--color-text-muted);cursor:pointer;' +
    'text-decoration:underline;text-underline-offset:2px;';
  notThisOne.textContent = 'Not the right repository? Enter it manually…';
  notThisOne.addEventListener('click', () => {
    listWrap.innerHTML = '';
    _showQuantSwapManualInput();
  });
  listWrap.appendChild(notThisOne);

  row.appendChild(listWrap);
}

function _showQuantSwapManualInput() {
  const row = document.getElementById('hw-quant-local-row');
  if (!row) return;
  const btn = row.querySelector('#hw-quant-local-btn');
  if (!btn) return;

  const wrap = document.createElement('span');
  wrap.style.cssText = 'display:flex;gap:5px;align-items:center;flex:1;';
  const input = document.createElement('input');
  input.type = 'text';
  input.placeholder = 'owner/repo-GGUF';
  input.style.cssText = 'flex:1;padding:4px 8px;border-radius:5px;border:1px solid rgba(255,255,255,0.1);background:rgba(28,34,42,0.9);color:var(--color-text-primary);font-size:10px;min-width:0;';
  const goBtn = document.createElement('button');
  goBtn.type = 'button';
  goBtn.className = 'btn-wizard-tertiary';
  goBtn.style.cssText = 'font-size:10px;min-height:22px;padding:2px 8px;flex-shrink:0;';
  goBtn.textContent = 'Load';
  wrap.appendChild(input);
  wrap.appendChild(goBtn);
  btn.replaceWith(wrap);

  const statusEl = document.getElementById('hw-quant-local-status');

  const doFetch = async () => {
    const repoId = input.value.trim();
    if (!repoId) return;
    goBtn.disabled = true;
    if (statusEl) statusEl.textContent = 'Loading…';
    const data = await _hfFilesPost(repoId);
    goBtn.disabled = false;
    if (!data?.ok) { if (statusEl) statusEl.textContent = 'Repo not found'; return; }
    const rawFiles = (data.files || []).filter(f =>
      !f.is_mmproj && (f.rfilename || f.path || '').toLowerCase().endsWith('.gguf'));
    if (!rawFiles.length) { if (statusEl) statusEl.textContent = 'No GGUFs found'; return; }
    wizardState.model.quantFiles = rawFiles.map(f => ({
      path: f.rfilename || f.path || '',
      name: f.rfilename || f.path || '',
      size: f.size || 0,
      label: _extractQuantLabel(f.rfilename || f.path || ''),
    }));
    wizardState.model._quantSwapRepo = repoId;
    if (statusEl) statusEl.textContent = '';
    renderHardwareModelHeader();
  };
  goBtn.addEventListener('click', doFetch);
  input.addEventListener('keydown', e => { if (e.key === 'Enter') doFetch(); });
}

export function _renderQuantSwapActions(quantPath, repoId) {
  const actionsRow = document.getElementById('hw-quant-swap-actions');
  if (!actionsRow) return;
  const quantName = quantPath.split('/').pop() || quantPath;

  actionsRow.innerHTML = '';
  actionsRow.style.display = '';

  const dlBtn = document.createElement('button');
  dlBtn.type = 'button';
  dlBtn.className = 'btn-wizard-secondary';
  dlBtn.style.cssText = 'font-size:10px;min-height:24px;padding:3px 10px;';
  dlBtn.textContent = `⬇ Download ${quantName}`;

  const streamBtn = document.createElement('button');
  streamBtn.type = 'button';
  streamBtn.className = 'btn-wizard-tertiary';
  streamBtn.style.cssText = 'font-size:10px;min-height:24px;padding:3px 10px;';
  streamBtn.textContent = '▶ Stream from HF';

  const statusEl = document.createElement('span');
  statusEl.style.cssText = 'font-size:10px;color:var(--color-text-muted);margin-left:4px;';

  dlBtn.addEventListener('click', () => {
    dlBtn.disabled = true; streamBtn.disabled = true;
    statusEl.textContent = 'Starting download…';
    const dlPanel = document.getElementById('hf-download-panel');
    if (dlPanel) {
      // Temporarily set hfRepo so hfStartDownload resolves the URL correctly.
      const prevRepo = wizardState.model.hfRepo;
      wizardState.model.hfRepo = repoId;
      hfShowDownloadPanel(dlPanel, quantName);
      hfStartDownload({
        repoId,
        filePath: quantPath,
        panelEl: dlPanel,
        onComplete: (_id, localPath) => {
          wizardState.model.source = 'local';
          wizardState.model.delivery = 'downloaded_hf';
          wizardState.model.path = localPath;
          wizardState.model.hfRepo = '';
          wizardState.model.hfFile = '';
          wizardState.model.originRepo = repoId;
          wizardState.model.originFile = quantPath;
          _attachOriginTags(localPath, repoId);
          actionsRow.style.display = 'none';
          statusEl.textContent = '✓ Downloaded and selected';
          showToast('Quant downloaded', 'success', quantName);
        },
        onValidationError: msg => { statusEl.textContent = msg; dlBtn.disabled = false; streamBtn.disabled = false; },
        onClearValidationError: () => {},
      });
      if (!prevRepo) wizardState.model.hfRepo = '';
    }
  });

  streamBtn.addEventListener('click', () => {
    wizardState.model.source = 'hf';
    wizardState.model.hfRepo = repoId;
    wizardState.model.hfFile = quantPath;
    wizardState.model.delivery = 'stream_hf';
    wizardState.model.path = '';
    actionsRow.style.display = 'none';
    showToast('Switched to HF stream', 'success', quantName);
    scheduleVramUpdate();
    refreshEngineRecommendation();
  });

  actionsRow.appendChild(dlBtn);
  actionsRow.appendChild(streamBtn);
  actionsRow.appendChild(statusEl);
}
