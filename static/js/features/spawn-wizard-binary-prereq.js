// llama.cpp binary prerequisite check & download for the spawn wizard.
import { getPlatformInfo } from '../core/platform-info.js';
import { showToast } from './toast.js';
import { dom, wizardState } from './spawn-wizard.js';
import { _mtpUserConfigured } from './spawn-wizard-mtp-draft.js';
import { refreshGuidedCards } from './spawn-wizard-guided.js';

export let _binaryReady  = false;
export let _platformInfo = null;   // cached result of /api/llama-binary/platform-info
let _selectedBackend = null;

// Cross-module mutator: refreshEngineAvailability() in the shell needs to
// refresh this cached platform info without owning the binding.
export function setWizardPlatformInfo(v) {
  _platformInfo = v || _platformInfo;
}

export async function _checkBinaryPrereq() {
  if (!dom.binaryPrereq) return;
  if (wizardState.engine.selected === 'rapid_mlx') {
    dom.binaryPrereq.style.display = 'none';
    _updateSpawnBtnForPrereq();
    return;
  }
  try {
    const headers = window.authHeaders ? window.authHeaders() : {};

    // Fetch platform info and current version in parallel
    const [vResp, platformInfo] = await Promise.all([
      fetch('/api/llama-binary/version', { headers }),
      getPlatformInfo().catch(() => null),
    ]);

    const vData = vResp.ok ? await vResp.json() : {};
    _platformInfo = platformInfo;

    // The recommendation can switch engines while the llama.cpp prerequisite
    // requests are in flight. Never let that stale response restore the
    // llama.cpp banner over a Rapid-MLX flow.
    if (wizardState.engine.selected === 'rapid_mlx') {
      dom.binaryPrereq.style.display = 'none';
      _updateSpawnBtnForPrereq();
      return;
    }

    if (_selectedBackend === null && _platformInfo) {
      _selectedBackend = _platformInfo.auto_backend;
    }

    // MTP is now enabled by default on all platforms including Metal.
    // Users can disable if quality issues arise on their hardware.
    if (!_mtpUserConfigured) {
      wizardState.hardware.mtpEnabled = Number(wizardState.arch?.mtpDepth || 0) > 0;
      const mtpCheck = document.getElementById('hw-use-mtp');
      const mtpDepthRow = document.getElementById('hw-mtp-depth-row');
      if (mtpCheck) mtpCheck.checked = wizardState.hardware.mtpEnabled;
      if (mtpDepthRow) {
        mtpDepthRow.style.display = wizardState.hardware.mtpEnabled ? '' : 'none';
      }
    }
    refreshGuidedCards();

    // Unified-memory Auto disables the extra host prompt-state cache. Preserve explicit
    // values, including -1 (unlimited), which remains an Advanced user choice.
    if (_platformInfo?.auto_backend === 'metal' && wizardState.hardware.cacheRam == null) {
      wizardState.hardware.cacheRam = 0;
      if (dom.cacheRamInput) dom.cacheRamInput.value = '0';
    }
    // Show unified memory note about -cram.
    if (_platformInfo?.auto_backend === 'metal') {
      const hint = document.getElementById('unified-cram-hint');
      if (hint) hint.style.display = '';
    }

    if (vData.build) {
      _binaryReady = true;
      if (dom.binaryPrereq.style.display !== 'none') {
        _showPrereqState('success');
        if (dom.prereqSuccessText) {
          const label = _platformInfo ? _platformInfo.label : 'llama.cpp';
          dom.prereqSuccessText.textContent = `${label} b${vData.build} installed and ready.`;
        }
        setTimeout(() => { if (dom.binaryPrereq) dom.binaryPrereq.style.display = 'none'; }, 3000);
      }
      _updateSpawnBtnForPrereq();
    } else {
      _binaryReady = false;
      _showPrereqState('idle');
      dom.binaryPrereq.style.display = '';
      _renderPrereqIdle(vData, _platformInfo);
      _updateSpawnBtnForPrereq();
    }
  } catch {
    // Network error — don't block the wizard
  }
}

function _renderPrereqIdle(vData, platform) {
  // Update download button label with platform detail
  if (dom.prereqDownloadBtn && platform) {
    const label = platform.label || 'llama.cpp';
    dom.prereqDownloadBtn.textContent = `Download ${label}`;
  }

  // Show configured path if present but binary missing
  if (dom.prereqPath && dom.prereqPathRow) {
    const path = vData.path || '';
    dom.prereqPath.textContent = path || '(not configured — will use app default)';
    dom.prereqPathRow.style.display = '';
  }

  // For multi-backend platforms (Windows, Linux), inject a backend selector
  const existingPicker = document.getElementById('wizard-prereq-backend-picker');
  if (existingPicker) existingPicker.remove();

  if (platform && platform.multi_backend && platform.backends && platform.backends.length > 1) {
    const picker = document.createElement('div');
    picker.id = 'wizard-prereq-backend-picker';
    picker.className = 'wizard-prereq-backend-picker';

    const pickerLabel = document.createElement('label');
    pickerLabel.className = 'wizard-prereq-backend-label';
    pickerLabel.htmlFor = 'wizard-prereq-backend-select';
    pickerLabel.textContent = 'Select your GPU / backend:';
    picker.appendChild(pickerLabel);

    const select = document.createElement('select');
    select.id = 'wizard-prereq-backend-select';
    select.className = 'wizard-prereq-backend-select';
    platform.backends.forEach(b => {
      const opt = document.createElement('option');
      opt.value = b.id;
      opt.textContent = b.label;
      if (b.id === _selectedBackend) opt.selected = true;
      select.appendChild(opt);
    });
    select.addEventListener('change', () => {
      _selectedBackend = select.value;
      // Update the note shown below the selector
      const selected = platform.backends.find(b => b.id === _selectedBackend);
      if (noteEl) noteEl.textContent = selected ? selected.note : '';
      // Update download button label
      if (dom.prereqDownloadBtn) {
        dom.prereqDownloadBtn.textContent = `Download llama.cpp (${select.options[select.selectedIndex].text.split(' —')[0]})`;
      }
    });
    picker.appendChild(select);

    // Note line below selector
    const noteEl = document.createElement('div');
    noteEl.className = 'wizard-prereq-backend-note';
    const currentBackend = platform.backends.find(b => b.id === _selectedBackend);
    noteEl.textContent = currentBackend ? currentBackend.note : '';
    picker.appendChild(noteEl);

    // Insert before the actions div
    const actions = dom.prereqIdle.querySelector('.wizard-prereq-actions');
    if (actions) dom.prereqIdle.insertBefore(picker, actions);

    // Update initial download button label for Windows
    if (dom.prereqDownloadBtn && currentBackend) {
      dom.prereqDownloadBtn.textContent = `Download llama.cpp (${currentBackend.label.split(' —')[0]})`;
    }
  }
}

function _showPrereqState(state) {
  if (dom.prereqIdle)     dom.prereqIdle.style.display     = state === 'idle'     ? '' : 'none';
  if (dom.prereqProgress) dom.prereqProgress.style.display = state === 'progress' ? '' : 'none';
  if (dom.prereqSuccess)  dom.prereqSuccess.style.display  = state === 'success'  ? '' : 'none';
}

export function _updateSpawnBtnForPrereq() {
  if (!dom.spawnServerBtn) return;
  if (wizardState.engine.selected === 'rapid_mlx') {
    const ready = wizardState.engine.rapidMlxLocalAvailable && wizardState.engine.rapidMlxRuntimeCompatible;
    dom.spawnServerBtn.disabled = !ready;
    dom.spawnServerBtn.title = ready ? '' : 'Rapid-MLX requires Apple Silicon and a compatible managed or external runtime';
  } else if (!_binaryReady) {
    dom.spawnServerBtn.disabled = true;
    dom.spawnServerBtn.title = 'llama.cpp binary required — download it above first';
  } else {
    dom.spawnServerBtn.disabled = false;
    dom.spawnServerBtn.title = '';
  }
}

export async function _downloadBinaryForWizard() {
  if (!dom.binaryPrereq || !dom.prereqDownloadBtn) return;
  _showPrereqState('progress');
  if (dom.prereqDownloadBtn) dom.prereqDownloadBtn.disabled = true;

  const backend = _selectedBackend || (_platformInfo && _platformInfo.auto_backend) || null;
  const platformLabel = _platformInfo ? _platformInfo.label : 'llama.cpp';

  // Update progress description with what we're downloading
  const descEl = dom.prereqProgress?.querySelector('.wizard-prereq-desc');
  if (descEl) {
    descEl.textContent = backend && _platformInfo && _platformInfo.multi_backend
      ? `Downloading llama.cpp ${backend.toUpperCase()} build — this may take a minute…`
      : `Downloading ${platformLabel} build — this may take a minute…`;
  }

  const startTime = Date.now();
  let elapsedTimer = setInterval(() => {
    const s = Math.floor((Date.now() - startTime) / 1000);
    if (dom.prereqElapsed) dom.prereqElapsed.textContent = `${s}s elapsed…`;
    if (dom.prereqBar) {
      const pct = Math.min(90, 5 + Math.floor(s * 1.2));
      dom.prereqBar.style.width = pct + '%';
    }
  }, 1000);

  try {
    const headers = window.authHeaders
      ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
      : { 'Content-Type': 'application/json' };
    const body = backend ? { backend } : {};
    const resp = await fetch('/api/llama-binary/update', {
      method: 'POST', headers, body: JSON.stringify(body)
    });
    clearInterval(elapsedTimer);
    if (dom.prereqBar) dom.prereqBar.style.width = '100%';

    if (!resp.ok) {
      const txt = await resp.text().catch(() => `HTTP ${resp.status}`);
      throw new Error(txt);
    }
    const data = await resp.json();
    if (data.ok === false) throw new Error(data.error || 'Download failed');

    _binaryReady = true;
    _showPrereqState('success');
    if (dom.prereqSuccessText) {
      // data.version is the tag like "b5678"; data.backend is what was installed
      const ver  = data.version || 'installed';
      const back = data.backend ? ` · ${data.backend.toUpperCase()}` : '';
      dom.prereqSuccessText.textContent = `llama.cpp ${ver}${back} downloaded and ready.`;
    }
    _updateSpawnBtnForPrereq();
    setTimeout(() => { if (dom.binaryPrereq) dom.binaryPrereq.style.display = 'none'; }, 4000);
  } catch (err) {
    clearInterval(elapsedTimer);
    _showPrereqState('idle');
    _renderPrereqIdle({}, _platformInfo);
    if (dom.prereqDownloadBtn) dom.prereqDownloadBtn.disabled = false;
    showToast('Binary download failed', 'error', (err.message || 'Unknown error').split('\n')[0]);
  }
}
