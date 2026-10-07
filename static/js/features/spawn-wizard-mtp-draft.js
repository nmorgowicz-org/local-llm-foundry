// Hardware-step MTP (Multi-Token Prediction) draft-model handling: this is
// llama.cpp GGUF speculative decoding — matching/checking/rendering candidate
// draft models and the MTP section UI. Explicitly distinct from the Rapid-MLX
// sidecar flow in spawn-wizard-rapid-mlx.js and the future MTPLX runtime; per
// the MTPLX audit these are three unrelated speculative-decoding mechanisms.
import {
  dom, wizardState, scheduleVramUpdate, guessQuantFromName,
  _modelStemForSearch,
} from './spawn-wizard.js';
import { formatBytes } from './spawn-wizard-format.js';
import { showToast } from './toast.js';
import { openModelFileBrowser } from './file-browser-launcher.js';
import { refreshGuidedCards } from './spawn-wizard-guided.js';

// Generic draft-model matching: score candidates by shared token overlap.
export function _bestDraftForModel(modelFilename, candidates) {
  if (!candidates.length) return null;
  const stem = _modelStemForSearch(modelFilename)
    .toLowerCase()
    .replace(/[^a-z0-9-]/g, ' ')
    .trim()
    .split(/\s+/)
    .filter(Boolean);

  const scoreCandidate = (f) => {
    const base = (f.path || f.name || '').split(/[\\/]/).pop() || '';
    const fstem = _modelStemForSearch(base)
      .toLowerCase()
      .replace(/[^a-z0-9-]/g, ' ')
      .trim()
      .split(/\s+/)
      .filter(Boolean);

    // Count shared tokens between model stem and candidate stem.
    const cset = new Set(fstem);
    let shared = 0;
    for (const t of stem) {
      if (cset.has(t)) shared++;
    }

    // Prefer smaller candidates (more likely real draft/model).
    const size = f.size || 0;
    const sizeBonus = size > 0 && size < 1_500_000_000 ? 1 : 0;

    return { shared, sizeBonus };
  };

  let best = null, bestShared = -1, bestSizeBonus = 0;
  for (const f of candidates) {
     const { shared, sizeBonus } = scoreCandidate(f);
     if (shared >= 3 &&
         (shared > bestShared || (shared === bestShared && sizeBonus > bestSizeBonus))) {
       bestShared = shared;
       bestSizeBonus = sizeBonus;
       best = f;
     }
   }

   // Allow a single candidate even with a weak score.
   if (best) return best;
   if (candidates.length === 1) return candidates[0];
   return null;
}

// ── Gemma4 MTP draft check ───────────────────────────────────────────────────

export async function _checkGemma4MtpDraft(modelPath) {
  if (!dom.mtpDownloadSection) return;

  const modelFilename = (modelPath || '').split(/[\\/]/).pop() || '';
  const lower = modelFilename.toLowerCase();
  const isGemma4 = lower.includes('gemma-4') || lower.includes('gemma4');

  if (!isGemma4) {
    dom.mtpDownloadSection.style.display = 'none';
    return;
  }

  // Only show download offer if no local draft candidates were found
  const candidates = wizardState.model.draftCandidates || [];
  if (candidates.length > 0) {
    dom.mtpDownloadSection.style.display = 'none';
    return;
  }

  try {
    const headers = window.authHeaders ? window.authHeaders() : {};
    const quantLabel = guessQuantFromName(modelFilename) || 'Q8_0';
    const repoId = (wizardState.model.repoId || '').trim();

    const resp = await fetch('/api/spawn-wizard/mtp-draft-check', {
      method: 'POST',
      headers: { ...headers, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_name: modelFilename,
        repo_id: repoId,
        quant_label: quantLabel,
      }),
    });

    if (!resp.ok) return;

    const data = await resp.json();
    if (!data.ok) return;

    if (data.draft_available && data.draft_path) {
      // Local draft found — add to candidates and auto-select
      wizardState.model.draftCandidates = wizardState.model.draftCandidates || [];
      wizardState.model.draftCandidates.push({
        path: data.draft_path,
        name: data.draft_path.split(/[\\/]/).pop(),
        size: 0,
        is_draft: true,
      });
      wizardState.model.selectedDraftPath = data.draft_path;
      dom.mtpDownloadSection.style.display = 'none';

      // Re-render MTP section to show draft selector
      if (wizardState.currentStep === 2) renderMtpSection();
      return;
    }

    // No local draft — show download button if HF URL available
    if (data.hf_download_url) {
      dom.mtpDownloadInfo.textContent = `Available: ${data.tier} · ${quantLabel} · ~42 MB estimated`;
      dom.mtpDownloadSection.style.display = '';

      // Wire the download button (only bind once)
      if (!dom.mtpDownloadBtn.dataset.bound) {
        dom.mtpDownloadBtn.dataset.bound = '1';
        dom.mtpDownloadBtn.addEventListener('click', async () => {
          dom.mtpDownloadBtn.disabled = true;
          dom.mtpDownloadBtn.textContent = 'Downloading…';

          try {
            // Use the existing HF download endpoint
            const downloadResp = await fetch('/api/hf/download', {
              method: 'POST',
              headers: { ...headers, 'Content-Type': 'application/json' },
              body: JSON.stringify({
                repo_id: data.hf_repo_id,
                filename: data.hf_filename,
                local_filename: data.local_filename || data.hf_filename,
                path: 'MTP',
              }),
            });

            const downloadData = await downloadResp.json();
            if (downloadData.ok) {
              showToast('MTP Draft', 'success', 'MTP draft model download started. Check status in the sidebar.');
            } else {
              showToast('MTP Draft', 'warning', downloadData.error || 'Download failed');
            }
          } catch {
            showToast('MTP Draft', 'warning', 'Download request failed');
          } finally {
            dom.mtpDownloadBtn.disabled = false;
            dom.mtpDownloadBtn.textContent = 'Download MTP Draft Model';
          }
        });
      }
    }
  } catch {
    // Non-fatal — silently fail
  }
}

// ── Draft candidate pill buttons ─────────────────────────────────────────────

function _renderDraftCandidatePills() {
  const container = document.getElementById('spawn-draft-candidates');
  if (!container) return;

  const candidates = wizardState.model.draftCandidates || [];
  container.innerHTML = '';

  if (candidates.length === 0) {
    const emptySpan = document.createElement('span');
    emptySpan.style.cssText = 'color:var(--color-text-secondary);font-size:12px;';
    emptySpan.textContent = '(no draft model candidates detected)';
    container.appendChild(emptySpan);
    return;
  }

  candidates.forEach((candidate, index) => {
    const pill = document.createElement('button');
    pill.type = 'button';
    pill.className = 'btn-wizard-tertiary';
    pill.style.cssText =
      'font-size:11px;min-height:24px;padding:0 8px;cursor:pointer;border-radius:12px;' +
      'border:1px solid rgba(148,163,253,0.25);' +
      'background:rgba(80,120,200,0.15);color:var(--color-text-primary);white-space:nowrap;';
    
    const fname = (candidate.path || candidate.name || '').split(/[\\/]/).pop();
    const sizeStr = candidate.size ? ` · ${formatBytes(candidate.size)}` : '';
    pill.textContent = fname + sizeStr;
    pill.title = candidate.path || '';
    pill.dataset.index = index;

    pill.addEventListener('click', () => {
      wizardState.model.selectedDraftPath = candidate.path;
      if (dom.draftModelInput) dom.draftModelInput.value = candidate.path;
      scheduleVramUpdate();
    });

    container.appendChild(pill);
  });
}

// _mtpUserConfigured tracks whether the user explicitly toggled MTP via the
// checkbox, so auto-selection logic (e.g. picking a draft candidate) doesn't
// silently override an explicit user choice. Read by spawn-wizard-binary-prereq.js.
export let _mtpUserConfigured = false;

// ── Hardware step: MTP section ───────────────────────────────────────────────

export function renderMtpSection() {
  const section = document.getElementById('hw-mtp-section');
  if (!section) return;

  const hasBuiltInMtp = wizardState.arch.mtpDepth > 0;
  const hasAssistantSelected = (wizardState.model.selectedDraftPath || '').trim().length > 0;
  const showMtp = hasBuiltInMtp || hasAssistantSelected;

  if (!showMtp) {
    section.style.display = 'none';
    if (dom.mtpAssistantSection) dom.mtpAssistantSection.style.display = 'none';
    return;
  }

  // If a draft is selected but MTP is not explicitly disabled by the user,
  // we consider the MTP section "active" so tuning controls and draft-mtp mode are exposed.
  if (hasAssistantSelected && !hasBuiltInMtp && !_mtpUserConfigured) {
    wizardState.hardware.mtpEnabled = true;
  }

  section.style.display = '';

  const infoNote = document.getElementById('hw-mtp-info-note');
  if (infoNote && hasBuiltInMtp) { infoNote.style.display = ''; }

  // Render companion draft selector: always show for MTP models even
  // if no candidates were auto-detected, so user can still browse.
  if (dom.mtpAssistantSection && dom.mtpAssistantSelect) {
    const candidates = wizardState.model.draftCandidates || [];

    // Bind the change listener once; always repopulate options so new
    // candidates discovered after first render (or from a different model)
    // are not silently lost.
    if (!dom.mtpAssistantSelect.dataset.bound) {
      dom.mtpAssistantSelect.dataset.bound = '1';
       dom.mtpAssistantSelect.addEventListener('change', async () => {
         const selected = dom.mtpAssistantSelect.value || '';
         // Browse sentinel: open file browser for companion assistant
         if (selected === '__browse__') {
           await openModelFileBrowser(
             'hw-mtp-draft-select',
             'gguf',
             null,
             'draft-model',
           );
           return;
         }
         wizardState.model.selectedDraftPath = selected;

         // If draft model selected and no explicit conflicting choice, default to draft-mtp.
         if (selected && dom.specTypeSelect && !dom.specTypeSelect.value.includes('draft-mtp')) {
           dom.specTypeSelect.value = 'draft-mtp,ngram-mod';
           dom.specTypeSelect.dispatchEvent(new Event('change'));
         }

         // Ensure MTP is enabled for draft-model-based drafts (unless user explicitly disabled).
         if (selected && !_mtpUserConfigured) {
           wizardState.hardware.mtpEnabled = true;
         }

         // Reset n-max to null when draft model changes so renderMtpSection
         // re-applies the default (2 — universal starting point).
         if (!_mtpUserConfigured) {
           wizardState.hardware.mtpDraftNMax = null;
         }

         // Refresh MTP section to expose tuning controls and update visibility.
         renderMtpSection();
         scheduleVramUpdate();
       });
    }

    dom.mtpAssistantSection.style.display = '';

    dom.mtpAssistantSelect.innerHTML = '';
    const noneOpt = document.createElement('option');
    noneOpt.value = '';
    noneOpt.textContent = '(none — use built-in MTP only)';
    dom.mtpAssistantSelect.appendChild(noneOpt);

    candidates.forEach(f => {
      const fpath = f.path || f.name || '';
      const fname = fpath.split(/[\\/]/).pop();
      const sizeStr = f.size ? ` · ${formatBytes(f.size)}` : '';
      const opt = document.createElement('option');
      opt.value = fpath;
      opt.textContent = fname + sizeStr;
      dom.mtpAssistantSelect.appendChild(opt);
    });

    // If no candidates were auto-detected, add a sentinel option that
    // triggers a file browser when chosen by the user.
    if (candidates.length === 0) {
      const browseOpt = document.createElement('option');
      browseOpt.value = '__browse__';
      browseOpt.textContent = '(browse for a companion draft model GGUF…)';
      dom.mtpAssistantSelect.appendChild(browseOpt);
    }

    // Sync selection
    const current = wizardState.model.selectedDraftPath || '';
    if (current) dom.mtpAssistantSelect.value = current;
  }

  // Render draft candidate pills (quick selection buttons)
  _renderDraftCandidatePills();

  const checkbox = document.getElementById('hw-use-mtp');
  // The user-facing control is spec-draft-n-max (draft tokens per step), not "depth"
  // arch.mtpDepth = number of MTP heads built into the model (VRAM estimation only)
  const draftNMaxInput = document.getElementById('hw-mtp-depth');

  if (draftNMaxInput) {
    if (!draftNMaxInput.dataset.bound) {
      draftNMaxInput.dataset.bound = '1';
      draftNMaxInput.addEventListener('input', () => {
        const v = parseInt(draftNMaxInput.value, 10);
        if (v >= 0 && v <= 8) {
          wizardState.hardware.mtpDraftNMax = v;
        }
      });
    }
    // Show family-appropriate default: 4 for external draft model, 2 for built-in MTP.
    const draftNMaxDisplay = wizardState.hardware.mtpDraftNMax ?? 2;
    draftNMaxInput.value = draftNMaxDisplay;
  }

  // Derive whether MTP tuning controls should be visible:
  // - If user explicitly enabled via checkbox, OR
  // - If an assistant is selected (implies draft-mtp mode), OR
  // - If the speculative decoding dropdown is on draft-mtp.
  const specVal = dom.specTypeSelect?.value || '';
  const specUsesMtp = specVal && (specVal.includes('draft-mtp') || specVal.includes('draft-model'));
  const mtpEffectivelyEnabled = wizardState.hardware.mtpEnabled || hasAssistantSelected || specUsesMtp;

  if (checkbox) {
    if (!checkbox.dataset.bound) {
      checkbox.dataset.bound = '1';
      checkbox.addEventListener('change', () => {
        _mtpUserConfigured = true;
        wizardState.hardware.mtpEnabled = checkbox.checked;
        if (checkbox.checked) wizardState.hardware.parallelSlots = 1;

        // Sync spec dropdown to draft-mtp when enabling
        if (checkbox.checked && dom.specTypeSelect && !dom.specTypeSelect.value.includes('draft-mtp')) {
          dom.specTypeSelect.value = 'draft-mtp,ngram-mod';
          dom.specTypeSelect.dispatchEvent(new Event('change'));
        }
        if (!checkbox.checked && dom.specTypeSelect
          && /draft-(mtp|model)/.test(dom.specTypeSelect.value)) {
          dom.specTypeSelect.value = '';
          dom.specTypeSelect.dispatchEvent(new Event('change'));
        }

        renderMtpSection();
        scheduleVramUpdate();
      });
    }
    checkbox.checked = mtpEffectivelyEnabled;
  }

  // Ensure hw-mtp-depth-row is visible when MTP is in use.
  const depthRow = document.getElementById('hw-mtp-depth-row');
  if (depthRow) {
    depthRow.style.display = mtpEffectivelyEnabled ? '' : 'none';
  }

  // MTP requires parallel=1 (buildSpawnPayload forces this at spawn time,
  // spawn-wizard-spawn.js:446) — mirror it visibly here so the field never
  // silently diverges from what's actually launched (plan §6.1).
  const slotsInput = document.getElementById('spawn-parallel-slots');
  const slotsHint = document.getElementById('spawn-parallel-slots-mtp-hint');
  if (slotsInput) {
    if (mtpEffectivelyEnabled) {
      if (!slotsInput.dataset.mtpForcedPrev) {
        slotsInput.dataset.mtpForcedPrev = slotsInput.value || '1';
      }
      slotsInput.value = '1';
      slotsInput.disabled = true;
    } else if (slotsInput.dataset.mtpForcedPrev) {
      slotsInput.value = slotsInput.dataset.mtpForcedPrev;
      slotsInput.disabled = false;
      delete slotsInput.dataset.mtpForcedPrev;
    }
  }
  if (slotsHint) slotsHint.style.display = mtpEffectivelyEnabled ? '' : 'none';
  refreshGuidedCards();
}
