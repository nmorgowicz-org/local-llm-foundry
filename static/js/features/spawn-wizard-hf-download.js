// HF model download panel: start/cancel/complete flow, plus companion mmproj
// download that piggybacks on the main GGUF download (bypasses the normal
// per-file download cooldown so both files fetch concurrently).
import { showToast } from './toast.js';
import Router from './router.js';
import {
  hfStartDownload,
  hfCancelDownload,
  hfShowDownloadPanel,
  hfHideDownloadPanel,
} from './hf-browse.js';
import { formatBytes } from './spawn-wizard-format.js';
import { _attachOriginTags } from './spawn-wizard-hf-origin.js';
import {
  dom,
  wizardState,
  STEP_LABELS,
  showValidationError,
  clearValidationError,
  showStep,
  guessQuantFromName,
  updateModelInputVisibility,
  updateSelectedModelDisplay,
  renderLocalModelHint,
  refreshStepGuardrails,
  refreshHfTokenState,
  getAuthHeaders,
  _deriveMmprojSaveName,
} from './spawn-wizard.js';
import { renderMmprojSection } from './spawn-wizard-mmproj.js';

let _dlCurrentId = null;
let _mmprojCompanionId = null;
let _mmprojCompanionLocalPath = null;

export function bindHfDownloadPanel() {
  const dlPanel = document.getElementById('hf-download-panel');

  document.getElementById('hf-dlp-download-btn')?.addEventListener('click', () => {
    const { hfRepo, hfFile, mmprojHfFile, mmprojHfRepo } = wizardState.model;
    if (!hfRepo || !hfFile) return;

    // Companion mmproj download (bypasses cooldown)
    if (mmprojHfFile) {
      _startCompanionMmprojDownload(mmprojHfRepo || hfRepo, mmprojHfFile, hfFile);
    }

    hfStartDownload({
      repoId: hfRepo,
      filePath: hfFile,
      panelEl: dlPanel,
      companionId: _mmprojCompanionId || null,
      onComplete: (downloadId, localPath) => {
        _dlCurrentId = null;
        onHfDownloadComplete(downloadId, localPath);
      },
      onValidationError: (msg) => {
        _dlCurrentId = null;
        showValidationError(msg);
        // If main failed but companion already exists, associate it so retry is not blocked.
        if (_mmprojCompanionLocalPath) {
          _maybeAssociateCompanionOnMainFailure(_mmprojCompanionLocalPath);
        }
      },
      onClearValidationError: () => {
        clearValidationError();
      },
    }).then((data) => {
      _dlCurrentId = data?.download_id || null;
    });
  });

  document.getElementById('hf-dlp-use-hf-btn')?.addEventListener('click', () => {
    hfHideDownloadPanel(dlPanel);
  });

  document.getElementById('hf-dlp-cancel-btn')?.addEventListener('click', () => {
    hfCancelDownload({
      downloadId: _dlCurrentId,
      panelEl: dlPanel,
    });
    _dlCurrentId = null;
    // Also cancel companion mmproj download if running.
    if (_mmprojCompanionId) {
      hfCancelDownload({
        downloadId: _mmprojCompanionId,
        panelEl: dlPanel,
      });
      _mmprojCompanionId = null;
    }
  });

  document.getElementById('hf-dlp-open-settings')?.addEventListener('click', () => {
    Router.navigate('/settings#models');
    // Focus is secondary; openSettingsModal can handle tab, but we may still want to focus.
    setTimeout(() => document.getElementById('settings-hf-token')?.focus(), 80);
  });

  // Refresh download destination when settings change (e.g., models dir updated)
  window.addEventListener('settings-applied', () => {
    refreshHfTokenState();
    if (dlPanel && dlPanel.style.display !== 'none') {
      const fname = (wizardState.model?.hfFile || '').split('/').pop();
      if (fname) hfShowDownloadPanel(dlPanel, fname);
    }
  });
}

// Called by hfStartDownload onComplete when download finishes.
function onHfDownloadComplete(_downloadId, localPath) {
  const effectivePath = localPath;
  if (!effectivePath) return;

  const downloadedFile = wizardState.model.hfFile || '';
  const downloadedRepo = wizardState.model.hfRepo || '';

  wizardState.model.source = 'local';
  wizardState.model.delivery = 'downloaded_hf';
  wizardState.model.path = effectivePath;
  wizardState.model.originRepo = downloadedRepo;
  wizardState.model.originFile = downloadedFile;
  // Persist origin so future wizard sessions skip the HF search entirely.
  _attachOriginTags(effectivePath, downloadedRepo);
  wizardState.model.localMeta = {
    path: effectivePath,
    filename: effectivePath.split(/[\\/]/).pop() || effectivePath,
    size_display: wizardState.model.modelBytes ? formatBytes(wizardState.model.modelBytes) : '',
    quant_type: guessQuantFromName(downloadedFile || effectivePath),
    param_b: wizardState.model.paramB || null,
  };

  // Companion mmproj
  if (_mmprojCompanionLocalPath) {
    const mmprojLocalPath = _mmprojCompanionLocalPath;
    const mmprojName = mmprojLocalPath.split(/[\\/]/).pop() || mmprojLocalPath;
    wizardState.model.mmprojPath = mmprojLocalPath;
    wizardState.model.mmprojFiles = [{
      path: mmprojLocalPath, name: mmprojName,
      size: wizardState.arch.mmprojBytes || 0, is_mmproj: true,
    }];
    _mmprojCompanionId = null;
    _mmprojCompanionLocalPath = null;
  }

  wizardState.model.hfRepo = '';
  wizardState.model.hfFile = '';
  if (dom.modelPathInput) dom.modelPathInput.value = effectivePath;
  dom.modelSourceCards?.forEach(c => {
    c.classList.toggle('selected', c.dataset.source === 'local');
  });
  updateModelInputVisibility();
  updateSelectedModelDisplay();
  renderLocalModelHint();
  refreshStepGuardrails();

  // Toast notification + auto-advance to next step
  const filename = effectivePath.split(/[\\/]/).pop() || 'Model';
  showToast('Download complete', 'success', filename);
  const next = Math.min(wizardState.currentStep + 1, STEP_LABELS.length - 1);
  if (next > wizardState.currentStep) showStep(next);
}

async function _startCompanionMmprojDownload(repo, mmprojHfPath, modelHfPath) {
  const saveAs = _deriveMmprojSaveName(modelHfPath, mmprojHfPath);
  try {
    const headers = { ...getAuthHeaders(), 'Content-Type': 'application/json' };
    const res = await fetch('/api/hf/download', {
      method: 'POST',
      headers,
      body: JSON.stringify({ repo_id: repo, file_path: mmprojHfPath, save_as: saveAs, companion: true, resume: true }),
    });
    const data = await res.json().catch(() => ({}));
    if (res.ok && data.ok) {
      _mmprojCompanionId = data.download_id;
      _mmprojCompanionLocalPath = data.local_path;
    }
  } catch { /* companion download failure is non-fatal */ }
}

// If the main GGUF download failed but the companion mmproj already exists on disk,
// associate it into wizardState so that retrying the main download is not blocked.
async function _maybeAssociateCompanionOnMainFailure(mmprojLocalPath) {
  if (!mmprojLocalPath) return;

  // Check if companion is completed via the backend
  try {
    const headers = window.authHeaders ? window.authHeaders() : {};
    const res = await fetch(`/api/models/download/${_mmprojCompanionId || ''}/status`, { headers });
    if (!res.ok) return;
    const data = await res.json().catch(() => ({}));
    const s = data?.status?.status || data?.status;
    if (s?.status !== 'completed') return;
  } catch {
    return;
  }

  // Associate into wizardState.
  if (!wizardState.model.mmprojPath) {
    const mmprojName = mmprojLocalPath.split(/[\\/]/).pop() || mmprojLocalPath;
    wizardState.model.mmprojPath = mmprojLocalPath;
    wizardState.model.mmprojFiles = [{
      path: mmprojLocalPath, name: mmprojName,
      size: wizardState.arch.mmprojBytes || 0, is_mmproj: true,
    }];
    renderMmprojSection?.();
  }
}
