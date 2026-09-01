// Final wizard steps: command-preview card, spawn-server submission, and the
// canonical payload builders. buildSpawnPayload/launchPortForPayload/
// supportsTunePanelForPayload are the Playwright test contract -- re-exported
// from the shell (tests dynamically import them from
// '/js/features/spawn-wizard.js').
import Router from './router.js';
import {
  dom, wizardState, closeSpawnWizard, getEffectiveArch, isUnifiedMemory,
} from './spawn-wizard.js';
import { buildPresetPayload } from './spawn-wizard-review-step.js';
import { _binaryReady } from './spawn-wizard-binary-prereq.js';
import { buildRapidMlxConfig } from './spawn-wizard-rapid-mlx.js';
import { openEvidenceDrawer, evidenceFromCommandPreview } from './evidence-drawer.js';
import { setTuneConfig, showTunePanel } from './tune-panel.js';
import { setHeaderMode } from './attach-detach.js';
import { showToast } from './toast.js';
import { applyChatTemplateDegradeFromReasons } from './spawn-wizard-chat-template.js';

// Reasons from the most recent command-preview fetch (step 6). The spawn endpoint itself does
// not currently return launch warnings, so this is the freshest signal available at click time
// for surfacing the chat-template-degraded toast (plan §2.2, step-5 surface).
let _lastPreviewReasons = [];

// ── Spawn config preview card (step 6) ────────────────────────────────────────

/**
 * Fetches and renders the exact `rapid-mlx serve` argv for the current wizard state.
 *
 * Rendered into the step-6 config card. `/api/rapid-mlx/command-preview` resolves the
 * binary and the model the same way the launcher does, so a flag missing here is a flag
 * that will be missing at launch — including ones the installed runtime does not support,
 * which the endpoint drops rather than passes on.
 */
async function _renderCommandPreview(host) {
  const mk = (tag, cls, text) => {
    const el = document.createElement(tag);
    if (cls) el.className = cls;
    if (text !== undefined) el.textContent = text;
    return el;
  };

  host.innerHTML = '';
  host.appendChild(mk('div', 'spawn-command-preview-status', 'Building launch command…'));

  let data;
  try {
    const payload = buildSpawnPayload();
    // The key is a secret and the preview does not need it; the same strip is done for
    // presets at buildPresetPayload().
    const { api_key: _apiKey, ...config } = payload?.rapid_mlx || {};
    const headers = Object.assign(
      { 'Content-Type': 'application/json' },
      window.authHeaders ? window.authHeaders() : {},
    );
    const resp = await fetch('/api/rapid-mlx/command-preview', {
      method: 'POST',
      headers,
      body: JSON.stringify(config),
    });
    data = await resp.json();
    if (!resp.ok || data?.ok === false) {
      throw new Error(data?.error || `Preview failed (${resp.status})`);
    }
  } catch (err) {
    host.innerHTML = '';
    // A failed preview must not read as a failed launch. It is a preview of the command,
    // and the spawn button stays enabled.
    const warn = mk('div', 'spawn-command-preview-error');
    warn.appendChild(mk('div', 'spawn-command-preview-error-title', 'Could not preview the launch command'));
    warn.appendChild(mk('div', 'spawn-command-preview-error-detail', String(err.message || err)));
    host.appendChild(warn);
    return;
  }

  host.innerHTML = '';

  _lastPreviewReasons = Array.isArray(data.reasons) ? data.reasons : [];
  applyChatTemplateDegradeFromReasons(_lastPreviewReasons);
  const aliasReason = _lastPreviewReasons.find(r => typeof r === 'string' && r.includes('[alias_source]'));
  if (aliasReason) {
    const warnBlock = mk('div', 'wizard-review-warning');
    warnBlock.appendChild(mk(
      'div',
      'wizard-review-warning-text',
      'Chat template: not applied (alias source). The model\'s built-in template will be used.',
    ));
    host.appendChild(warnBlock);
  }

  const hdr = mk('div', 'spawn-command-preview-header');
  hdr.appendChild(mk('span', 'spawn-command-preview-title', 'Launch command'));
  const copyBtn = mk('button', 'spawn-command-preview-copy', 'Copy');
  copyBtn.type = 'button';
  const argvText = ['rapid-mlx', ...(data.argv || [])].join(' ');
  copyBtn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(argvText);
      copyBtn.textContent = 'Copied';
      setTimeout(() => { copyBtn.textContent = 'Copy'; }, 1500);
    } catch {
      copyBtn.textContent = 'Copy failed';
      setTimeout(() => { copyBtn.textContent = 'Copy'; }, 1500);
    }
  });
  hdr.appendChild(copyBtn);
  const explainBtn = mk('button', 'evidence-trigger', 'Explain');
  explainBtn.type = 'button';
  explainBtn.addEventListener('click', () => openEvidenceDrawer(evidenceFromCommandPreview(data), explainBtn));
  hdr.appendChild(explainBtn);
  host.appendChild(hdr);

  host.appendChild(mk('pre', 'spawn-command-preview-argv', argvText));

  if (data.redacted) {
    host.appendChild(mk(
      'div',
      'spawn-command-preview-note',
      'Secrets are redacted from this preview; the launched process receives the real values.',
    ));
  }

  // Flags the runtime declined, or changed. Empty on a clean build, which is the normal case.
  const diff = data.requested_vs_effective && typeof data.requested_vs_effective === 'object'
    ? Object.entries(data.requested_vs_effective)
    : [];
  if (diff.length > 0) {
    const box = mk('div', 'spawn-command-preview-diff');
    box.appendChild(mk('div', 'spawn-command-preview-diff-title', 'Adjusted by the runtime'));
    diff.forEach(([key, value]) => {
      const row = mk('div', 'spawn-command-preview-diff-row');
      row.appendChild(mk('span', 'spawn-command-preview-diff-key', key));
      row.appendChild(mk(
        'span',
        'spawn-command-preview-diff-value',
        typeof value === 'string' ? value : JSON.stringify(value),
      ));
      box.appendChild(row);
    });
    host.appendChild(box);
  }

  const otherReasons = _lastPreviewReasons.filter(r => !(typeof r === 'string' && r.includes('[alias_source]')));
  if (otherReasons.length > 0) {
    const list = mk('ul', 'spawn-command-preview-reasons');
    otherReasons.forEach((reason) => list.appendChild(mk('li', null, reason)));
    host.appendChild(list);
  }
}


export async function _renderSpawnConfigCard() {
  const card = document.getElementById('spawn-config-card');
  const sidebar = document.getElementById('spawn-sidebar-config');
  const m = wizardState.model, hw = wizardState.hardware, acc = wizardState.access;
  const rapid = wizardState.engine.selected === 'rapid_mlx';

  const modelName = m.source === 'hf'
    ? (m.hfFile ? m.hfFile.split('/').pop() : (m.hfRepo || '—'))
    : (m.path ? (m.path.split(/[\\/]/).pop() || m.path) : '—');
  const port     = acc.port || 8001;
  const ctx      = hw.contextSize ? hw.contextSize.toLocaleString() + ' tok' : '—';
  const gpu      = hw.gpuLayers === 'manual'
    ? `${hw.gpuLayersManual ?? '?'} layers`
    : (hw.gpuLayers === 'all' ? 'All GPU' : 'Auto');
  const bind     = acc.bindHost === '0.0.0.0' ? '0.0.0.0 (LAN)' : '127.0.0.1';
  const kvStr    = `${(hw.cacheTypeK || 'q8_0').toUpperCase()} / ${(hw.cacheTypeV || 'q8_0').toUpperCase()}`;
  const alias    = hw.alias || modelName.replace(/\.gguf$/i, '').replace(/[^A-Za-z0-9._-]/g, '-');

  // Fetch tags for this model
  let modelTags = [];
  try {
    const modelPath = m.path || (m.localPath && m.localPath.trim());
    if (modelPath) {
      const headers = window.authHeaders ? window.authHeaders() : {};
      const resp = await fetch('/api/models/tags', { headers });
      if (resp.ok) {
        const data = await resp.json();
        modelTags = (data.tags && data.tags[modelPath]) || [];
      }
    }
  } catch { /* ignore */ }

  if (card) {
    card.style.display = '';
    const mk = (tag, cls, text) => {
      const el = document.createElement(tag);
      if (cls) el.className = cls;
      if (text !== undefined) el.textContent = text;
      return el;
    };
    card.innerHTML = '';

    const hdr = mk('div', 'spawn-config-card-header');
    hdr.appendChild(mk('span', 'spawn-config-card-title', 'Model'));
    hdr.appendChild(mk('span', 'spawn-config-card-model', modelName));
    card.appendChild(hdr);

    // Add tag pills if model has tags
    if (modelTags.length > 0) {
      const tagsRow = mk('div', 'spawn-config-card-tags');
      modelTags.forEach(tag => {
        const pill = mk('span', 'mm-tag-pill', tag);
        tagsRow.appendChild(pill);
      });
      card.appendChild(tagsRow);
    }

    const grid = mk('div', 'spawn-config-grid');
    const items = rapid ? [
      ['Engine',  'Rapid-MLX'],
      ['Port',    String(port)],
      ['Host',    bind],
      ['Served as', alias],
    ] : [
      ['Engine',  'llama.cpp'],
      ['Port',    String(port)],
      ['Host',    bind],
      ['Context', ctx],
      ['GPU',     gpu],
      ['KV quant', kvStr],
      ['Alias',   alias],
    ];
    items.forEach(([label, value]) => {
      const item = mk('div', 'spawn-config-item');
      item.appendChild(mk('div', 'spawn-config-item-label', label));
      item.appendChild(mk('div', 'spawn-config-item-value', value));
      grid.appendChild(item);
    });
    card.appendChild(grid);

    // Rapid-MLX takes most of its behaviour from serve flags rather than from the four
    // fields above, so show the argv the launcher will actually run. This is also the only
    // surface that consumes effective_policy / requested_vs_effective, which is where the
    // runtime reports flags it silently declined to pass through.
    if (rapid) {
      const host = mk('div', 'spawn-command-preview');
      host.id = 'spawn-command-preview';
      card.appendChild(host);
      _renderCommandPreview(host);
    }
  }

  if (sidebar) {
    sidebar.innerHTML = '';
    [
      ['Model', modelName],
      ['Port',  String(port)],
      ['Host',  bind],
    ].forEach(([label, value]) => {
      const stat = document.createElement('div');
      stat.className = 'spawn-sidebar-stat';
      const lbl = document.createElement('span');
      lbl.className = 'spawn-sidebar-stat-label';
      lbl.textContent = label;
      const val = document.createElement('span');
      val.className = 'spawn-sidebar-stat-value';
      val.textContent = value;
      stat.appendChild(lbl);
      stat.appendChild(val);
      sidebar.appendChild(stat);
    });
  }
}

// ── Spawn server ──────────────────────────────────────────────────────────────

export async function spawnServer() {
  if (wizardState.spawn.inFlight) return;
  if (wizardState.engine.selected === 'llama_cpp' && !_binaryReady) {
    showErrorText('llama.cpp binary not found. Download it using the banner above.');
    return;
  }
  wizardState.spawn.inFlight = true; wizardState.spawn.error = '';
  if (!dom.spawnServerBtn) return;
  dom.spawnServerBtn.disabled = true;
  setStatusText('Preparing configuration…'); setProgress(10); clearStatusMessages();
  try {
    const payload = buildSpawnPayload();
    setStatusText(wizardState.engine.selected === 'rapid_mlx' ? 'Starting Rapid-MLX…' : 'Starting llama-server…'); setProgress(30);
    // /api/sessions/spawn requires the db-admin-token, not the llama-server API token.
    const tokenResp = await fetch('/api/db/admin-token', {
      headers: window.authHeaders ? window.authHeaders() : {},
    });
    const tokenData = tokenResp.ok ? await tokenResp.json().catch(() => ({})) : {};
    const adminToken = tokenData.token;
    if (!adminToken) { throw new Error('Authentication required. Could not retrieve admin token.'); }
    const headers = { 'Content-Type': 'application/json', 'Authorization': `Bearer ${adminToken}` };
    const resp = await fetch('/api/sessions/spawn', { method: 'POST', headers, body: JSON.stringify(payload) });
    setProgress(60);
    if (!resp.ok) {
      if (resp.status === 429) {
        // Cooldown: parse seconds_remaining and disable button
        const d = await resp.json().catch(() => null);
        const seconds = d?.seconds_remaining || 15;
        showErrorText(`Spawn request too soon. Please wait ${seconds} seconds.`);
        setStatusText('Cooldown active.');
        setProgress(0);
        // Disable button with countdown
        if (dom.spawnServerBtn) {
          dom.spawnServerBtn.disabled = true;
          const origText = dom.spawnServerBtn.textContent || '';
          let left = seconds;
          const iv = setInterval(() => {
            if (left <= 0) {
              clearInterval(iv);
              if (dom.spawnServerBtn) {
                dom.spawnServerBtn.disabled = false;
                dom.spawnServerBtn.textContent = origText;
              }
            } else {
              if (dom.spawnServerBtn) dom.spawnServerBtn.textContent = `Wait ${left}s`;
              left--;
            }
          }, 1000);
        }
        wizardState.spawn.inFlight = false;
        return;
      }
      const t = await resp.text().catch(()=>'Unknown error');
      throw new Error(t || `HTTP ${resp.status}`);
    }
    const data = await resp.json().catch(() => null);
    if (!data?.ok) throw new Error(data?.error || 'Spawn request failed.');
    setStatusText('Server process started. Waiting for endpoint…');
    setProgress(75);
    const launchPort = launchPortForPayload(payload);
    await waitForSpawnReadiness(launchPort);
    setProgress(100); setStatusText('Server started.');
    showSuccessText('Server is running.');
    showToast('Server started', 'success', '', { duration: 8000 });
    if (_lastPreviewReasons.some(r => typeof r === 'string' && r.includes('[alias_source]'))) {
      showToast(
        'Chat template not applied',
        'warning',
        "This model was selected by alias — the server is using the model's built-in template.",
        { duration: 8000 },
      );
    }
    if (supportsTunePanelForPayload(payload)) setTuneConfig(payload);
    setTimeout(() => {
      closeSpawnWizard();
      setHeaderMode('Spawn:' + launchPort);
      if (document.body.classList.contains('setup-active')) {
        Router.navigate('/server');
      }
      if (supportsTunePanelForPayload(payload)) showTunePanel();
      // Select the preset that was saved during this wizard run (if any)
      if (wizardState.savedPresetId) {
        import('./presets.js').then(({ loadPresets }) => {
          loadPresets(wizardState.savedPresetId);
        });
      }
      setTimeout(() => window.restorePreviousPosition?.(), 600);
    }, 1200);
  } catch (err) {
    const msg = (err.message || String(err)).split('\n')[0].trim();
    showErrorText(msg || 'Failed to start server.'); setStatusText('Spawn failed.'); wizardState.spawn.error = msg;
    showToast('Spawn failed', 'error', msg || 'Check logs.');
  } finally {
    wizardState.spawn.inFlight = false;
    if (dom.spawnServerBtn) dom.spawnServerBtn.disabled = false;
  }
}

export function launchPortForPayload(payload) {
  return (payload?.backend === 'rapid_mlx' ? payload.rapid_mlx?.port : payload?.port) || 8001;
}

export function supportsTunePanelForPayload(payload) {
  return payload?.backend === 'llama_cpp' || payload?.backend === 'rapid_mlx';
}

async function waitForSpawnReadiness(port, timeoutMs = 30000) {
  const started = Date.now();
  const headers = window.authHeaders ? window.authHeaders() : {};

  while (Date.now() - started < timeoutMs) {
    try {
      const resp = await fetch('/api/sessions/active/readiness', {
        method: 'GET',
        headers,
        cache: 'no-store',
      });
      const data = await resp.json().catch(() => null);
      if (resp.ok && data?.ok && data.ready) return;
    } catch {
      // Keep polling until timeout; the backend route may lag while the process boots.
    }
    const elapsed = Date.now() - started;
    setStatusText(`Waiting for endpoint on port ${port}…`);
    setProgress(Math.min(95, 75 + Math.floor((elapsed / timeoutMs) * 20)));
    await new Promise(r => setTimeout(r, 800));
  }

  const engineName = wizardState.engine.selected === 'rapid_mlx' ? 'Rapid-MLX' : 'llama-server';
  throw new Error(`${engineName} started but did not become reachable on port ${port} in time.`);
}

// Clamp a value that must be a u32 (non-negative integer). Returns null if absent or negative.
function _u32(v, min = 0) {
  if (v == null || v < min) return null;
  return Math.floor(v);
}

// Handle -t / -tb: allow -1 (llama-server auto) or any positive integer; null if blank/invalid.
function _threadsValue(v) {
  if (v == null || (typeof v !== 'number' && v === '')) return null;
  const n = Number(v);
  if (Number.isNaN(n) || !Number.isFinite(n)) return null;
  const i = Math.floor(n);
  // -1 = llama-server default (auto); only positive integers allowed otherwise
  if (i === -1 || i >= 1) return i;
  return null;
}

export function buildSpawnPayload() {
  const h = wizardState.hardware, m = wizardState.model;
  if (wizardState.engine.selected === 'rapid_mlx') {
    return {
      backend: 'rapid_mlx',
      rapid_mlx: buildRapidMlxConfig(h, m),
    };
  }
  const arch = getEffectiveArch();
  const gpuLayers = h.gpuLayers === 'manual' ? (h.gpuLayersManual ?? -1) : (h.gpuLayers === 'all' ? -1 : null);

 // MTP: when enabled, use draft-mtp spec type and force parallel=1
  const mtpActive = arch.mtpDepth > 0 && h.mtpEnabled;
  const specTypeUser = dom.specTypeSelect?.value || '';
  // Respect an explicit 'draft-mtp' choice from the advanced dropdown; only
  // default to 'draft-mtp,ngram-mod' when the user hasn't already chosen a
  // draft-mtp variant (covers the common case where the dropdown is empty).
  const draftPath = m.selectedDraftPath || (dom.draftModelInput?.value || '').trim() || '';
  const hasDraft = draftPath.length > 0;

  // When the user has a draft model configured and has not
  // explicitly chosen a conflicting mode, default to draft-mtp.
  let specType = specTypeUser;
  if (mtpActive || hasDraft) {
    if (!specType) {
      specType = 'draft-mtp,ngram-mod';
    }
  }

  // MTP requires parallel=1 when active.
  const parallelSlots = (mtpActive || (hasDraft && specType.includes('draft-mtp'))) ? 1 : h.parallelSlots;
  const selectedLoadMode = h.loadMode || (isUnifiedMemory() ? 'mmap' : 'none');
  const loadMode = h.mlock && selectedLoadMode === 'mmap'
    ? 'mmap+mlock'
    : h.mlock && selectedLoadMode === 'none'
      ? 'mlock'
      : selectedLoadMode;

  // Resolve mmproj local path: prefer mmprojPath (local file), fall back to
  // mmprojHfFile only if it looks like an absolute path (i.e. was set from
  // a directory scan, not from an HF file list).
  const mmprojLocal = m.mmprojPath && m.mmprojPath.startsWith('/') ? m.mmprojPath : null;

  // MTP n-max: 2 is the safe universal starting point (community consensus). Users can
  // increase to 3–4 for Gemma4-style external draft on high-bandwidth hardware (Apple Silicon).
  // Use != null so an explicit 0 from the user is not treated as "unset".
  const usesMtpSpec = mtpActive || (hasDraft && (specType.includes('draft-mtp') || specType.includes('draft-model')));
  const mtpNMaxDefault = 2;
  const mtpNMax = usesMtpSpec
    ? (h.mtpDraftNMax != null ? h.mtpDraftNMax : mtpNMaxDefault)
    : undefined;

  return {
    backend: 'llama_cpp',
    model_path: m.source !== 'hf' ? (m.path || '') : '',
    hf_repo: m.source === 'hf' ? (m.hfRepo || null) : null,
    hf_file: m.source === 'hf' ? (m.hfFile || null) : null,
    mmproj: mmprojLocal,
    port: wizardState.access.port || 8001,
    bind_host: wizardState.access.bindHost || '127.0.0.1',
    gpu_layers: gpuLayers,
    context_size: h.contextSize,
    batch_size: h.batchSize,
    ubatch_size: h.ubatchSize,
    parallel_slots: parallelSlots,
    ctk: h.cacheTypeK || '',
    ctv: h.cacheTypeV || '',
    spec_draft_type_k: h.specDraftTypeK || '',
    spec_draft_type_v: h.specDraftTypeV || '',
    n_cpu_moe: h.nCpuMoe || null,
    tensor_split: h.tensorSplit || '',
    // mmap default is platform-aware: on Apple Silicon (unified memory) keep mmap ON
    // (no_mmap=false) — it's zero-copy into Metal, doesn't change throughput (measured
    // identical tok/s on M5 Max), and avoids slow loads + committing RAM. On discrete GPUs
    // --no-mmap sidesteps slow Windows mmap paths, so default it on there.
    no_mmap: !isUnifiedMemory(),
    load_mode: loadMode,
    ngram_spec: false,
    spec_type: specType,
    spec_draft_n_max: _u32(mtpNMax, 1),
    spec_draft_n_min: usesMtpSpec ? _u32(h.mtpDraftNMin) : undefined,
    spec_draft_p_min: usesMtpSpec && h.mtpDraftPMin != null ? h.mtpDraftPMin : undefined,
    draft_model: draftPath || '',
    kv_unified: h.kvUnified,
    flash_attn: h.flashAttn || '',
    mlock: h.mlock || false,
    prio: h.prio != null ? h.prio : null,
    verbosity: h.verbosity != null ? h.verbosity : 4,
    ctx_checkpoints: h.ctxCheckpoints != null ? h.ctxCheckpoints : 32,
    checkpoint_min_step: h.checkpointMinStep != null ? h.checkpointMinStep : 8192,
    cache_reuse: h.cacheReuse != null ? h.cacheReuse : null,
    cache_idle_slots: h.cacheIdleSlots,
    no_cont_batching: !!h.noContBatching,
    swa_full: !!h.swaFull,
    mmproj_offload: h.mmprojOffload,
    llama_reasoning_effort: h.llamaReasoningEffort || 'default',
    llama_reasoning_format: h.llamaReasoningFormat || null,
    llama_reasoning_preserve: h.llamaReasoningPreserve,
    threads: _threadsValue(h.threads),
    threads_batch: _threadsValue(h.threadsBatch),
    fit_enabled: h.fitEnabled,
    fit_target: h.fitEnabled === true ? (h.fitTarget || null) : null,
    cache_mode: h.cacheMode || 'custom',
    cache_ram_mib: (h.cacheRam !== null && h.cacheRam !== undefined) ? h.cacheRam : null,
    // Sampling defaults (null = use llama-server built-in defaults)
    temperature: h.temperature != null ? h.temperature : null,
    top_p: h.topP != null ? h.topP : null,
    top_k: h.topK != null ? h.topK : null,
    min_p: h.minP != null ? h.minP : null,
    repeat_penalty: h.repeatPenalty != null ? h.repeatPenalty : null,
    repeat_last_n: h.repeatLastN != null ? h.repeatLastN : null,
    presence_penalty: h.presencePenalty != null && h.presencePenalty > 0 ? h.presencePenalty : null,
    max_tokens: h.maxTokens != null ? h.maxTokens : null,
    seed: h.seed != null ? h.seed : null,
     // Thinking / reasoning
     enable_thinking: h.enableThinking,
     preserve_thinking: h.preserveThinking,
     tool_call_format: h.toolCallFormat || null,
     reasoning_budget: h.reasoningBudget,
    reasoning: h.reasoningMode || null,
    reasoning_budget_message: h.reasoningBudgetMessage || null,
    grammar: h.grammar.trim() ? h.grammar.trim() : null,
    json_schema: h.jsonSchema.trim() ? h.jsonSchema.trim() : null,
    // Image token budget — only passed when mmproj is active.
    // Values are derived from model family; user can override via extra_args.
    image_min_tokens: mmprojLocal ? _imageMinTokens(m) : null,
    image_max_tokens: mmprojLocal ? _imageMaxTokens(m) : null,
    api_key: wizardState.access.apiKey || null,
    alias: h.alias || null,
    extra_args: h.extraArgs || '',
    chat_template_file: wizardState.model.chatTemplatePath || null,
    profile: wizardState.profile,
    use_case: wizardState.useCase,
    preset_id: wizardState.savedPresetId || null,
  };
}

function _modelNameLower(m) {
  return ((m.hfFile || '').split('/').pop() || m.path?.split(/[\\/]/).pop() || m.hfRepo || '').toLowerCase();
}

function _imageMinTokens(m) {
  const name = _modelNameLower(m);
  if (name.includes('gemma')) return 280;
  return 1024; // Qwen3.6 / default for other vision models
}

function _imageMaxTokens(m) {
  const name = _modelNameLower(m);
  if (name.includes('gemma')) return 560;
  return 4096; // Qwen3.6 / default for other vision models
}

// ── Status helpers ────────────────────────────────────────────────────────────

function setStatusText(t) { if (dom.statusText) dom.statusText.textContent = t; }
function setProgress(p) { if (dom.progressFill) dom.progressFill.style.width = Math.min(100, Math.max(0, p)) + '%'; }
function showErrorText(t) { if (dom.errorText) dom.errorText.textContent = t || ''; }
function showSuccessText(t) { if (dom.successText) dom.successText.textContent = t || ''; }
function clearStatusMessages() { if (dom.errorText) dom.errorText.textContent = ''; if (dom.successText) dom.successText.textContent = ''; }
export function resetSpawnStatus() { wizardState.spawn = { inFlight:false, error:'' }; setStatusText('Ready to spawn.'); setProgress(0); clearStatusMessages(); }
