// Review-step (wizard step 4/5) rendering: summary card, sampling-field
// sync, preset save/load, and the structured params-review table.
import {
  dom, wizardState, showStep,
  effectiveAvailBytes, getModelBytes, getSizingArch, isUnifiedMemory,
} from './spawn-wizard.js';
import { buildSpawnPayload } from './spawn-wizard-spawn.js';
import {
  kvBpe, formatCtx, formatGB,
} from './spawn-wizard-format.js';
import { buildEstimateBody, rapidEstimatePolicyFromWizardHardware } from './vram-estimate.js';
import { syncRapidSpeculativeFields } from './spawn-wizard-rapid-mlx.js';
import { showToast } from './toast.js';

// ── Model-specific sampling defaults (from /api/model-defaults) ──────────────

function _applyPresetToHardware(preset) {
  const h = wizardState.hardware;
  if (preset.id === 'model_default') {
    h.temperature = null;
    h.topP = null;
    h.topK = null;
    h.minP = null;
    h.repeatPenalty = null;
    h.presencePenalty = null;
    h.maxTokens = null;
    h.enableThinking = null;
    h.preserveThinking = null;
    h.toolCallFormat = null;
    h.reasoningBudget = null;
    h.reasoningMode = null;
    h.reasoningBudgetMessage = null;
    _syncThinkingFields();
    return;
  }
  if (preset.id === 'custom') return;
  if (preset.temperature != null) h.temperature = preset.temperature;
  if (preset.top_p != null) h.topP = preset.top_p;
  if (preset.top_k != null) h.topK = preset.top_k;
  if (preset.min_p != null) h.minP = preset.min_p;
  if (preset.repeat_penalty != null) h.repeatPenalty = preset.repeat_penalty;
  h.presencePenalty = preset.presence_penalty ?? null;
  h.maxTokens = preset.max_tokens != null ? preset.max_tokens : null;
  h.enableThinking   = preset.enable_thinking   ?? null;
   h.preserveThinking = preset.preserve_thinking ?? null;
   h.toolCallFormat = preset.tool_call_format ?? null;
   h.reasoningBudget  = preset.reasoning_budget  ?? null;
  h.reasoningMode = typeof preset.reasoning === 'boolean'
    ? (preset.reasoning ? 'on' : 'off')
    : (preset.reasoning || null);
  h.reasoningBudgetMessage = preset.reasoning_budget_message ?? null;
  _syncThinkingFields();
}

function _renderSamplingPresetPills(presets) {
  const container = document.getElementById('spawn-sampling-presets');
  if (!container) return;

  if (!presets || presets.length <= 1) {
    container.style.display = 'none';
    container.innerHTML = '';
    return;
  }

  container.style.display = 'flex';
  container.style.cssText = 'display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:10px;';
  container.innerHTML = '';

  const label = document.createElement('span');
  label.style.cssText = 'font-size:11px;color:var(--color-text-muted);flex-shrink:0;';
  label.textContent = 'Mode:';
  container.appendChild(label);

  presets.forEach((preset, i) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'sampling-preset-pill' + (i === 0 ? ' active' : '');
    btn.textContent = preset.name;
    const provenance = preset.provenance?.unsloth?.url || preset.provenance?.model_author?.source;
    const badges = (preset.workload_badges || []).join(', ');
    const rapidCoverage = preset.rapid_mlx_coverage || {};
    const rapidUnqualified = wizardState.engine.selected === 'rapid_mlx'
      && !Object.values(rapidCoverage).some(Boolean);
    btn.title = [preset.description, badges && `Best for: ${badges}`, provenance && `Source: ${provenance}`,
      rapidUnqualified && 'Rapid-MLX sampler defaults are pending runtime qualification; this mode is informational.']
      .filter(Boolean).join('\n');
    btn.disabled = rapidUnqualified && preset.id !== 'model_default' && preset.id !== 'custom';
    btn.dataset.presetIndex = String(i);
    btn.addEventListener('click', () => {
      container.querySelectorAll('.sampling-preset-pill').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      _applyPresetToHardware(preset);
      wizardState.hardware.samplingMode = preset.id || 'custom';
      if (dom.samplingModeSelect) dom.samplingModeSelect.value = wizardState.hardware.samplingMode;
      _syncSamplingFields();
    });
    container.appendChild(btn);
  });
}

export async function _fetchAndApplyModelSamplingDefaults() {
  const m = wizardState.model;
  const name = m.hfFile
    ? (m.hfFile.split('/').pop() || m.hfFile)
    : (m.path ? m.path.split(/[\\/]/).pop() : '') || m.hfRepo || '';
  if (!name) return;

  try {
    const headers = window.authHeaders
      ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
      : { 'Content-Type': 'application/json' };
    const res = await fetch('/api/model-defaults', {
      method: 'POST',
      headers,
      body: JSON.stringify({
        model_name_or_repo: name,
        size_bytes: m.modelBytes || 0,
        tags: [],
        gguf_arch: wizardState.arch.ggufArch || '',
        arch_family: wizardState.model.family || '',
        backend: wizardState.engine.selected || 'llama_cpp',
        // Streamed-but-not-yet-downloaded HF models never run local introspection
        // (their model.path is intentionally empty). Send the HF coordinates so the
        // server can fetch the real GGUF header via a range request instead of ever
        // falling back to a filename/repo-name guess.
        hf_repo_id: !wizardState.arch.ggufArch ? (m.hfRepo || '') : '',
        hf_file_path: !wizardState.arch.ggufArch ? (m.hfFile || '') : '',
      }),
    });
    if (!res.ok) return;
    const data = await res.json();
    const defaults = data.defaults || data;
    // Server-side introspection succeeded for a not-yet-downloaded HF model — persist
    // the real architecture so later calls (and other UI reading wizardState.arch) don't
    // need to re-fetch it, and never fall through to a filename heuristic.
    if (data.introspected) {
      const meta = data.introspected;
      if (meta.n_ctx_train) wizardState.model.nCtxTrain = meta.n_ctx_train;
      if (meta.n_layers) wizardState.arch.nLayers = meta.n_layers;
      if (meta.n_kv_heads) wizardState.arch.nKvHeads = meta.n_kv_heads;
      if (meta.head_dim) wizardState.arch.headDim = meta.head_dim;
      if (meta.n_experts) wizardState.arch.nExperts = meta.n_experts;
      if (meta.n_experts_used) wizardState.arch.nExpertsUsed = meta.n_experts_used;
      if (meta.mtp_depth) wizardState.arch.mtpDepth = meta.mtp_depth;
      if (meta.n_attn_layers) wizardState.arch.nAttnLayers = meta.n_attn_layers;
      if (meta.linear_attn_state_bytes) wizardState.arch.linearAttnStateBytes = meta.linear_attn_state_bytes;
      if (meta.n_global_attn_layers) wizardState.arch.nGlobalAttnLayers = meta.n_global_attn_layers;
      if (meta.local_kv_heads) wizardState.arch.localKvHeads = meta.local_kv_heads;
      if (meta.global_head_dim) wizardState.arch.globalHeadDim = meta.global_head_dim;
      if (meta.local_head_dim) wizardState.arch.localHeadDim = meta.local_head_dim;
      if (meta.sliding_window) {
        wizardState.arch.slidingWindow = meta.sliding_window;
        wizardState.arch.localAttnWindow = meta.sliding_window;
      }
      if (meta.mmproj_required != null) wizardState.arch.mmprojRequired = !!meta.mmproj_required;
      if (meta.gguf_arch && !wizardState.arch.ggufArch) {
        wizardState.arch.ggufArch = meta.gguf_arch;
        if (!wizardState.model.family) {
          wizardState.model.family = String(meta.gguf_arch).toLowerCase().replace(/_/g, '.');
        }
      }
      if (meta.gguf_arch || meta.n_layers || meta.n_ctx_train) {
        wizardState.arch.metadataStatus = 'resolved';
        wizardState.arch.metadataReason = 'Progressive GGUF header';
      }
    }
    if (data.provenance) {
      wizardState.arch.metadataStatus = data.provenance === 'unknown/degraded' ? 'degraded' : 'resolved';
      wizardState.arch.metadataReason = data.provenance_evidence || data.degraded_reason || '';
    }
    const h = wizardState.hardware;
    const effectiveCoverage = wizardState.engine.selected === 'rapid_mlx'
      ? (data.modes?.[0]?.rapid_mlx_coverage || {})
      : (data.modes?.[0]?.llama_cpp_coverage || {});
    const canApplyDefaults = Object.values(effectiveCoverage).some(Boolean);

    // Apply first preset values (only overwrite fields not yet explicitly set)
    if (canApplyDefaults) {
      if (h.temperature == null && defaults.temperature != null) h.temperature = defaults.temperature;
      if (h.topP == null && defaults.top_p != null) h.topP = defaults.top_p;
      if (h.topK == null && defaults.top_k != null) h.topK = defaults.top_k;
      if (h.minP == null && defaults.min_p != null) h.minP = defaults.min_p;
      if (h.repeatPenalty == null && defaults.repeat_penalty != null) h.repeatPenalty = defaults.repeat_penalty;
      if (h.presencePenalty == null && defaults.presence_penalty != null) {
        h.presencePenalty = defaults.presence_penalty;
      }
      if (h.maxTokens == null && defaults.max_tokens != null) h.maxTokens = defaults.max_tokens;
      if (h.enableThinking == null && defaults.enable_thinking != null) h.enableThinking = defaults.enable_thinking;
      if (h.preserveThinking == null && defaults.preserve_thinking != null) h.preserveThinking = defaults.preserve_thinking;
      if (h.reasoningMode == null && defaults.reasoning != null) {
        h.reasoningMode = defaults.reasoning ? 'on' : 'off';
      }
      if (h.reasoningBudget == null && defaults.reasoning_budget != null) h.reasoningBudget = defaults.reasoning_budget;
      if (h.reasoningBudgetMessage == null && defaults.reasoning_budget_message != null) {
        h.reasoningBudgetMessage = defaults.reasoning_budget_message;
      }
    }
    _syncThinkingFields();

    // Rust owns the complete catalog, including stable IDs/provenance/coverage.
    _renderSamplingPresetPills(data.modes || []);
  } catch { /* non-fatal */ }
}

// ── Summary (Step 4) ──────────────────────────────────────────────────────────

export async function renderSummary() {
  if (!dom.summaryList) return;
  dom.summaryList.innerHTML = '';

  // Sync sampling fields in the review step form
  _syncSamplingFields();

  // Pre-fill preset name input if empty
   if (dom.presetNameInput && !dom.presetNameInput.value.trim()) {
     const m = wizardState.model;
     const ctx = wizardState.hardware.contextSize || 0;
     const modelFile = (m.path || m.hfRepo || '').split(/[/\\]/).pop() || '';
     const base = (modelFile || '').replace(/\.gguf$/i, '').trim();
     const name = base && ctx
       ? base + '-' + formatCtx(ctx).toLowerCase()
       : base || 'My Preset';
     dom.presetNameInput.value = name;
   }

  const m = wizardState.model, hw = wizardState.hardware;
  const rapid = wizardState.engine.selected === 'rapid_mlx';
  const arch = getSizingArch();
  const availVram = effectiveAvailBytes();
  const modelBytes = getModelBytes();

  const modelDisplay = m.source === 'hf'
    ? (m.hfFile ? `${m.hfRepo} / ${m.hfFile.split('/').pop()}` : m.hfRepo || '(none)')
    : (m.path ? m.path.split(/[\\/]/).pop() || m.path : '(none)');

  let acquisition = 'Local file';
  if (m.delivery === 'stream_hf') {
    // originRepo is set for normal HF models; hfRepo is set when streaming via quant swap
    const repo = m.originRepo || m.hfRepo || '';
    const file = m.originFile || m.hfFile || '';
    if (repo) acquisition = `Stream from HuggingFace · ${repo}${file ? ` / ${file.split('/').pop()}` : ''}`;
  } else if (m.delivery === 'downloaded_hf' && m.originRepo) {
    acquisition = `Downloaded from HuggingFace · ${m.originRepo}${m.originFile ? ` / ${m.originFile.split('/').pop()}` : ''}`;
  } else if (m.delivery === 'imported_local') {
    acquisition = 'Imported local file';
  }

  const ctxK = hw.cacheTypeK || 'q8_0', ctxV = hw.cacheTypeV || 'q8_0';
   const kvSize = await (async () => {
     if (rapid) return 0;
     if (!modelBytes) return 0;
     try {
       // Builder item 6: canonical body builder for cross-surface equality.
       const body = buildEstimateBody({
         backend: 'llama_cpp',
         model_path: m.path || m.localPath || '',
         n_ctx: hw.contextSize || 4096,
         parallel_slots: hw.parallelSlots || 1,
         ubatch_size: hw.ubatchSize || 2048,
         ctk: ctxK,
         ctv: ctxV,
         n_cpu_moe: hw.nCpuMoe || 0,
         available_vram_bytes: availVram,
         is_unified_memory: isUnifiedMemory(),
         mmproj_path: m.mmprojPath || null,
         mmproj_bytes: m.mmprojBytes || 0,
         ...(wizardState.engine.selected === 'rapid_mlx' ? rapidEstimatePolicyFromWizardHardware(hw) : {}),
       });
       const headers = (window.authHeaders ? window.authHeaders() : {});
       const res = await fetch('/api/vram-estimate', {
         method: 'POST',
         headers: { ...headers, 'Content-Type': 'application/json' },
         body: JSON.stringify(body),
       });
       if (!res.ok) return 0;
       const d = await res.json();
       return (d.ok && d.kv_cache_bytes != null) ? d.kv_cache_bytes : 0;
     } catch {
       return 0;
     }
   })();

  const sharedRows = [
    { label: 'Engine',        value: rapid ? 'Rapid-MLX' : 'llama.cpp' },
    { label: 'Use case',      value: { agentic: 'Agentic / RAG', general: 'General chat', roleplay: 'Roleplay / creative' }[wizardState.useCase] || wizardState.useCase },
    { label: 'Profile',       value: wizardState.profile },
    { label: 'Acquisition',   value: acquisition },
    { label: 'Port',          value: String(wizardState.access.port || 8001) },
    { label: 'Model',         value: modelDisplay },
    { label: 'Bind host',     value: wizardState.access.bindHost === '0.0.0.0' ? '0.0.0.0 (LAN visible)' : '127.0.0.1 only' },
  ];
  const rows = rapid ? sharedRows : [...sharedRows,
    { label: 'Context size',  value: `${hw.contextSize.toLocaleString()} tokens` },
    { label: 'GPU layers',    value: hw.gpuLayers === 'manual' ? String(hw.gpuLayersManual ?? '—') : hw.gpuLayers },
    { label: 'KV quant (K/V)', value: `${ctxK.toUpperCase()} / ${ctxV.toUpperCase()}` },
    { label: 'KV cache',      value: kvSize > 0 ? formatGB(kvSize) : '—' },
    { label: 'Batch / ubatch', value: `${hw.batchSize} / ${hw.ubatchSize}` },
    ...(hw.fitTarget ? [{ label: '--fit-target', value: String(hw.fitTarget) }] : []),
  ];
  if (!rapid && hw.flashAttn && hw.flashAttn !== 'auto') rows.push({ label: 'Flash Attn', value: hw.flashAttn });
  if (!rapid && hw.kvUnified != null) rows.push({ label: 'KV unified', value: hw.kvUnified ? 'On' : 'Off' });
  if (!rapid && hw.fitEnabled != null) rows.push({ label: 'Fit', value: hw.fitEnabled ? 'On' : 'Off' });
  if (!rapid && hw.mlock) rows.push({ label: 'mlock', value: 'Yes' });
 if (!rapid && hw.prio != null) rows.push({ label: 'Priority', value: ['Normal', 'Medium', 'High', 'Realtime'][hw.prio] ?? String(hw.prio) }); if (!rapid && hw.verbosity != null) rows.push({ label: 'Log verbosity', value: String(hw.verbosity) });
 if (!rapid && hw.loadMode) rows.push({ label: 'Load mode', value: hw.loadMode });
 if (!rapid && hw.ctxCheckpoints != null) rows.push({ label: 'Context checkpoints', value: String(hw.ctxCheckpoints) });
 if (!rapid && hw.checkpointMinStep != null) rows.push({ label: 'Checkpoint minimum step', value: String(hw.checkpointMinStep) });
  if (!rapid && hw.cacheReuse != null) rows.push({ label: 'Cache reuse threshold', value: String(hw.cacheReuse) });
  if (!rapid && hw.cacheIdleSlots != null) rows.push({ label: 'Idle slot cache', value: hw.cacheIdleSlots ? 'On' : 'Off' });
  if (!rapid && hw.noContBatching) rows.push({ label: 'Continuous batching', value: 'Disabled' });
  if (!rapid && hw.swaFull) rows.push({ label: 'Full SWA cache', value: 'Enabled' });
  if (!rapid && hw.mmprojOffload != null) rows.push({ label: 'Projector offload', value: hw.mmprojOffload ? 'On' : 'Off' });
  if (!rapid && hw.llamaReasoningEffort && hw.llamaReasoningEffort !== 'default') rows.push({ label: 'llama.cpp reasoning effort', value: hw.llamaReasoningEffort });
  if (!rapid && hw.llamaReasoningFormat) rows.push({ label: 'llama.cpp reasoning format', value: hw.llamaReasoningFormat });
  if (!rapid && hw.llamaReasoningPreserve != null) rows.push({ label: 'llama.cpp preserve reasoning', value: hw.llamaReasoningPreserve ? 'On' : 'Off' });
  if (!rapid && hw.nCpuMoe > 0 && arch.nExperts > 0) rows.push({ label: 'MoE CPU offload', value: `${hw.nCpuMoe} of ${arch.nLayers} layers` });
  if (!rapid && hw.tensorSplit) rows.push({ label: 'Tensor split', value: hw.tensorSplit });
  if (!rapid && arch.mmprojBytes > 0) rows.push({ label: 'mmproj', value: formatGB(arch.mmprojBytes) });
  const hasExternalDraft = !!(m.selectedDraftPath || '').trim();
  if (!rapid && (arch.mtpDepth > 0 || hasExternalDraft)) {
    const mtpActive = hw.mtpEnabled || hasExternalDraft;
    const nMaxDisplay = hw.mtpDraftNMax ?? (hasExternalDraft ? 4 : 2);
    rows.push({ label: 'MTP', value: mtpActive ? `enabled · draft ${nMaxDisplay} tokens/step · --parallel 1` : 'disabled' });
  }
  if ((m.delivery === 'stream_hf' || m.delivery === 'downloaded_hf' || m.originRepo) && m.hfTokenSet != null) {
    rows.push({ label: 'HF token', value: m.hfTokenSet ? 'Saved in app settings' : 'Not saved' });
  }
  const tplPath = wizardState.model.chatTemplatePath;
  const tplFamily = m.family || null;
  if (!rapid && tplPath) {
    const tplName = tplPath.split(/[/\\]/).pop() || tplPath;
    rows.push({ label: 'Chat template', value: tplName });
  } else if (!rapid && tplFamily) {
    rows.push({ label: 'Chat template', value: 'Embedded (from model file)' });
  }

  const specType = dom.specTypeSelect?.value || '';
    if (!rapid && specType) {
      let sv = { 'ngram-mod': 'N-gram (fast)', 'draft-model': 'Draft model' }[specType] || specType;
      // Show draft model filename for draft-model or MTP modes with external draft model
      if (dom.draftModelInput?.value) {
        const fileName = dom.draftModelInput.value.split(/[\\/]/).pop();
        if (specType === 'draft-model') sv += ` (${fileName})`;
        else if (specType.includes('draft-mtp')) sv += ` + ${fileName}`;
      }
      rows.push({ label: 'Speculative', value: sv });
    }
  if (!rapid && hw.fitTarget) rows.push({ label: '--fit-target', value: `${hw.fitTarget} MB` });
  if (!rapid && hw.cacheRam !== null && hw.cacheRam !== undefined) {
    const cramDisplay = hw.cacheRam < 0 ? 'no limit' : hw.cacheRam === 0 ? 'disabled' : `${hw.cacheRam} MiB`;
    rows.push({ label: '-cram', value: cramDisplay });
  }
  if (wizardState.access.apiKey) rows.push({ label: 'Server API key', value: `${wizardState.access.apiKey.slice(0, 4)}…${wizardState.access.apiKey.slice(-4)}` });

  // Summary list header
  const listHeader = document.createElement('div');
  listHeader.className = 'summary-list-header';
  listHeader.textContent = 'Configuration summary';
  dom.summaryList.appendChild(listHeader);

  rows.forEach(r => {
    const row = document.createElement('div');
    row.className = 'summary-row';
    const lbl = document.createElement('span'); lbl.className = 'summary-label'; lbl.textContent = r.label;
    const val = document.createElement('span'); val.className = 'summary-value'; val.textContent = r.value;
    row.appendChild(lbl); row.appendChild(val);
    dom.summaryList.appendChild(row);
  });

  // Warnings — stronger and more explicit for risky configs
  if (dom.summaryWarnings) {
    const warns = [];
    if (!modelDisplay || modelDisplay === '(none)') warns.push('No model selected.');
    const ratio = availVram > 0 && modelBytes > 0 ? (modelBytes + kvSize) / availVram : 0;

    if (ratio > 1.5) {
      warns.push("CRITICAL: Your configuration heavily exceeds available VRAM. The server will likely crash or run extremely slowly. Reduce context size, increase KV quant, or choose a smaller model.");
    } else if (ratio > 1.2) {
      warns.push("HIGH RISK: Configuration likely exceeds VRAM. The server may crash. Reduce context size or use a stronger KV quant (e.g., q4_K_M).");
    } else if (ratio > 1.0) {
      warns.push("RISKY: VRAM is exceeded or barely covered. Expect instability. Consider reducing context or using KV quantization.");
    } else if (ratio > 0.88) {
      warns.push("VRAM is tight. Minor increases in context or requests can trigger OOM errors. Watch GPU memory.");
    }

    // High context size: warn if user sets a very large context that strains VRAM
    if (hw.contextSize >= 32768) {
      const ctxRisk = ratio > 0.9;
      warns.push(
        ctxRisk
          ? "Very large context size selected. This puts significant pressure on VRAM and may slow generation."
          : "Large context size selected. This improves long tasks but uses more VRAM and may slow generation."
      );
    }

    // Agentic use case KV recommendation
    if (wizardState.useCase === 'agentic' && kvBpe(ctxK) < 1.0) {
      warns.push("q4_0 KV not recommended for agentic workflows — reduces tool-call coherence. Prefer q8_0 or q6_K when VRAM allows.");
    }

    // Binding/host visibility warnings
    if (wizardState.access.bindHost === '0.0.0.0' && !wizardState.access.apiKey) {
      warns.push('LAN-visible endpoint without a server API key. Set one unless you intentionally want an open local-network server.');
    } else if (wizardState.access.bindHost === '0.0.0.0') {
      warns.push('LAN-visible endpoint enabled. Make sure clients know the API key you set.');
    }

    if (warns.length) {
      dom.summaryWarnings.style.display = '';
      dom.summaryWarnings.innerHTML = '';
      const textWrap = document.createElement('div');
      textWrap.className = 'summary-warnings-text';
      warns.forEach(w => {
        const p = document.createElement('div');
        p.textContent = w;
        textWrap.appendChild(p);
      });
      dom.summaryWarnings.appendChild(textWrap);
    } else {
      dom.summaryWarnings.style.display = 'none';
    }
  }
  // (health check button removed — it checked the currently-running server, not the new config)

  // Add step shortcuts for last-minute changes
  const editRow = document.createElement('div');
  editRow.className = 'summary-edit-row';
  editRow.style.display = 'flex';
  editRow.style.gap = '8px';
  editRow.style.flexWrap = 'wrap';
  editRow.style.marginTop = '10px';

  const shortcuts = [
    { label: 'Edit model', step: 0 },
    { label: 'Edit hardware', step: 1 },
    { label: 'Edit sampling', step: 1, focusId: 'spawn-temperature' },
  ];
  shortcuts.forEach(({ label, step, focusId }) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'btn-wizard-tertiary';
    btn.textContent = label;
    btn.addEventListener('click', () => {
      showStep(step);
      if (focusId) {
        setTimeout(() => document.getElementById(focusId)?.focus(), 50);
      }
    });
    editRow.appendChild(btn);
  });
  dom.summaryList.appendChild(editRow);
}

// ── Sampling field sync (Review step) ────────────────────────────────────────

function _syncSamplingFields() {
  const h = wizardState.hardware;
  // Round f32 API values to avoid 0.699999988079071 style display artifacts
  const fmt = v => v == null ? '' : String(parseFloat(Number(v).toFixed(4)));
  const setVal = (id, val) => {
    const el = document.getElementById(id);
    if (el && val != null) el.value = fmt(val);
    else if (el) el.value = '';
  };
  setVal('spawn-temperature', h.temperature);
  setVal('spawn-seed', h.seed);
  setVal('spawn-top-p', h.topP);
  setVal('spawn-top-k', h.topK);
  setVal('spawn-min-p', h.minP);
  setVal('spawn-repeat-penalty', h.repeatPenalty);
  setVal('spawn-repeat-last-n', h.repeatLastN);
  setVal('spawn-presence-penalty', h.presencePenalty);
  setVal('spawn-max-tokens', h.maxTokens);
  if (dom.bindHostSelect) dom.bindHostSelect.value = wizardState.access.bindHost || '127.0.0.1';
  if (dom.portInput) dom.portInput.value = String(wizardState.access.port || 8001);
  if (dom.apiKeyInput) dom.apiKeyInput.value = wizardState.access.apiKey || '';
  const aliasEl = document.getElementById('spawn-alias');
  if (aliasEl) aliasEl.value = h.alias || '';
  const extraArgsEl = document.getElementById('spawn-extra-args');
  if (extraArgsEl) extraArgsEl.value = h.extraArgs || '';
  _syncStructuredOutputFields();
}

function _structuredOutputMode() {
  const h = wizardState.hardware;
  if (h.outputMode) return h.outputMode;
  if (h.jsonSchema) return 'json_schema';
  if (h.grammar) return 'grammar';
  return '';
}

function _syncStructuredOutputFields() {
  const mode = _structuredOutputMode();
  if (dom.outputModeSelect) dom.outputModeSelect.value = mode;
  if (dom.grammarWrap) dom.grammarWrap.style.display = mode === 'grammar' ? '' : 'none';
  if (dom.jsonSchemaWrap) dom.jsonSchemaWrap.style.display = mode === 'json_schema' ? '' : 'none';
  if (dom.grammarInput) dom.grammarInput.value = wizardState.hardware.grammar || '';
  if (dom.jsonSchemaInput) dom.jsonSchemaInput.value = wizardState.hardware.jsonSchema || '';
}

function _syncThinkingFields() {
  const h = wizardState.hardware;
  syncRapidSpeculativeFields();
  const section = document.getElementById('spawn-thinking-section');
  const hasThinking =
    h.enableThinking != null ||
    h.preserveThinking != null ||
    h.reasoningMode != null ||
    h.reasoningBudget != null ||
    h.reasoningBudgetMessage != null;

  if (section) {
    section.style.display = hasThinking ? '' : 'none';
    const preserveRow = document.getElementById('spawn-preserve-thinking-row');
    if (preserveRow) preserveRow.style.display = h.preserveThinking != null ? '' : 'none';
  }

  const chk = id => {
    const el = document.getElementById(id);
    if (el) el.checked = !!wizardState.hardware[id === 'spawn-enable-thinking' ? 'enableThinking' : 'preserveThinking'];
  };
  chk('spawn-enable-thinking');
  chk('spawn-preserve-thinking');

  // Sync Rapid-MLX checkbox from rapidReasoningMode
  if (dom.reasoningModeCheck) {
    dom.reasoningModeCheck.checked = h.rapidReasoningMode === 'on';
  }
  const budgetEl = document.getElementById('spawn-reasoning-budget');
  if (budgetEl) budgetEl.value = h.reasoningBudget != null ? String(h.reasoningBudget) : '';
   const msgEl = document.getElementById('spawn-reasoning-budget-message');
   if (msgEl) msgEl.value = (h.reasoningBudgetMessage || '').replace(/\n/g, '\\n');
   const tcfEl = document.getElementById('spawn-tool-call-format');
   if (tcfEl) tcfEl.value = h.toolCallFormat || '';
}

function _bindThinkingFields() {
  const bindChk = (id, key) => {
    const el = document.getElementById(id);
    if (!el || el.dataset.bound) return;
    el.dataset.bound = '1';
    el.addEventListener('change', () => { wizardState.hardware[key] = el.checked; });
  };
  const bindInput = (id, key, isInt = false) => {
    const el = document.getElementById(id);
    if (!el || el.dataset.bound) return;
    el.dataset.bound = '1';
    el.addEventListener('input', () => {
      const raw = el.value.trim();
      if (raw === '') { wizardState.hardware[key] = null; return; }
      wizardState.hardware[key] = isInt ? parseInt(raw, 10) : raw;
    });
  };
  const bindSel = (id, key) => {
    const el = document.getElementById(id);
    if (!el || el.dataset.bound) return;
    el.dataset.bound = '1';
    el.addEventListener('change', () => { wizardState.hardware[key] = el.value || null; });
  };
  bindChk('spawn-enable-thinking', 'enableThinking');
  bindChk('spawn-preserve-thinking', 'preserveThinking');
  // Reasoning mode: auto-fill budget + message defaults when user selects "on"
  const reasoningModeEl = document.getElementById('spawn-reasoning-mode');
  if (reasoningModeEl && !reasoningModeEl.dataset.bound) {
    reasoningModeEl.dataset.bound = '1';
    reasoningModeEl.addEventListener('change', () => {
      wizardState.hardware.reasoningMode = reasoningModeEl.value || null;
      if (reasoningModeEl.value === 'on') {
        const budgetEl  = document.getElementById('spawn-reasoning-budget');
        const msgEl     = document.getElementById('spawn-reasoning-budget-message');
        if (budgetEl && !budgetEl.value) {
          budgetEl.value = '8192';
          wizardState.hardware.reasoningBudget = 8192;
        }
        if (msgEl && !msgEl.value) {
          msgEl.value = '\\nFinal Answer:';
          wizardState.hardware.reasoningBudgetMessage = '\nFinal Answer:';
        }
      }
    });
  }
  bindInput('spawn-reasoning-budget', 'reasoningBudget', true);
  const bmEl = document.getElementById('spawn-reasoning-budget-message');
  if (bmEl && !bmEl.dataset.bound) {
    bmEl.dataset.bound = '1';
    bmEl.addEventListener('input', () => {
      const raw = bmEl.value.trim();
      wizardState.hardware.reasoningBudgetMessage = raw === '' ? null : raw.replace(/\\n/g, '\n');
    });
  }
  bindSel('spawn-tool-call-format', 'toolCallFormat');
}

export function _bindSamplingFields() {
  const bind = (id, key, isInt = false) => {
    const el = document.getElementById(id);
    if (!el || el.dataset.bound) return;
    el.dataset.bound = '1';
    el.addEventListener('input', () => {
      const raw = el.value.trim();
      if (raw === '') { wizardState.hardware[key] = null; return; }
      const v = isInt ? parseInt(raw, 10) : parseFloat(raw);
      if (!isNaN(v)) wizardState.hardware[key] = v;
    });
  };
  bind('spawn-temperature', 'temperature');
  bind('spawn-seed', 'seed', true);
  bind('spawn-top-p', 'topP');
  bind('spawn-top-k', 'topK', true);
  bind('spawn-min-p', 'minP');
  bind('spawn-repeat-penalty', 'repeatPenalty');
  bind('spawn-repeat-last-n', 'repeatLastN', true);
  bind('spawn-presence-penalty', 'presencePenalty');
  bind('spawn-max-tokens', 'maxTokens', true);
  _bindThinkingFields();

  if (dom.outputModeSelect && !dom.outputModeSelect.dataset.bound) {
    dom.outputModeSelect.dataset.bound = '1';
    dom.outputModeSelect.addEventListener('change', () => {
      const mode = dom.outputModeSelect.value || '';
      wizardState.hardware.outputMode = mode;
      if (!mode) {
        wizardState.hardware.grammar = '';
        wizardState.hardware.jsonSchema = '';
      }
      if (mode === 'grammar') wizardState.hardware.jsonSchema = '';
      if (mode === 'json_schema') wizardState.hardware.grammar = '';
      _syncStructuredOutputFields();
    });
  }
  if (dom.grammarInput && !dom.grammarInput.dataset.bound) {
    dom.grammarInput.dataset.bound = '1';
    dom.grammarInput.addEventListener('input', () => {
      wizardState.hardware.outputMode = 'grammar';
      wizardState.hardware.grammar = dom.grammarInput.value || '';
    });
  }
  if (dom.jsonSchemaInput && !dom.jsonSchemaInput.dataset.bound) {
    dom.jsonSchemaInput.dataset.bound = '1';
    dom.jsonSchemaInput.addEventListener('input', () => {
      wizardState.hardware.outputMode = 'json_schema';
      wizardState.hardware.jsonSchema = dom.jsonSchemaInput.value || '';
    });
  }

  // Alias and extra args — string fields, no parsing
  const bindStr = (id, key) => {
    const el = document.getElementById(id);
    if (!el || el.dataset.bound) return;
    el.dataset.bound = '1';
    el.addEventListener('input', () => { wizardState.hardware[key] = el.value; });
  };
  bindStr('spawn-alias', 'alias');
  bindStr('spawn-extra-args', 'extraArgs');
}

// ── Save as preset ────────────────────────────────────────────────────────────

export async function saveAsPreset() {
  const nameInput = dom.presetNameInput;
  const name = nameInput ? nameInput.value.trim() : '';
  if (!name) {
    if (nameInput) {
      nameInput.focus();
      nameInput.classList.add('field-error');
      setTimeout(() => nameInput.classList.remove('field-error'), 1500);
    }
    showToast('Enter a preset name first', 'warn');
    return;
  }

  const payload = buildPresetPayload();
  payload.name = name;

  // Ensure the payload captures the current Wizard state correctly
  // buildPresetPayload calls buildSpawnPayload which pulls from wizardState.

  const btn = dom.savePresetBtn;

  if (btn) { btn.disabled = true; btn.textContent = 'Saving…'; }

  try {
    const headers = window.authHeaders
      ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
      : { 'Content-Type': 'application/json' };

    let isUpdate = Boolean(wizardState.savedPresetId);

    // Verify the saved preset ID still exists before attempting a PUT.
    if (isUpdate) {
      try {
        const r = await fetch(`/api/presets/${wizardState.savedPresetId}`, { headers });
        const data = r.ok ? await r.json().catch(() => ({})) : {};
        if (!r.ok || data.ok === false) {
          // Preset was deleted or ID is stale — create a new one instead.
          wizardState.savedPresetId = null;
          isUpdate = false;
        }
      } catch {
        // Network error — fall back to create to be safe.
        wizardState.savedPresetId = null;
        isUpdate = false;
      }
    }

    let resp;
    if (isUpdate) {
      // Update existing preset
      resp = await fetch(`/api/presets/${wizardState.savedPresetId}`, {
        method: 'PUT',
        headers,
        body: JSON.stringify(payload),
      });
    } else {
      // Create new preset
      resp = await fetch('/api/presets', {
        method: 'POST',
        headers,
        body: JSON.stringify(payload),
      });
    }

    if (!resp.ok) {
      showToast('Save preset failed: ' + await resp.text().catch(() => ''), 'error');
      return;
    }

    // If this was the first save, store the preset id so next saves update the same preset.
    if (!wizardState.savedPresetId) {
      try {
        const data = await resp.json().catch(() => ({}));
        const savedId = data.preset?.id || data.id || null;
        if (savedId) wizardState.savedPresetId = savedId;
      } catch {
        // non-fatal
      }
    }

    // Refresh the setup view preset dropdown
    import('./presets.js').then(({ loadPresets }) => loadPresets(wizardState.savedPresetId).then(() => {
      import('./setup-view.js').then(({ syncSetupPresetSelect }) => syncSetupPresetSelect());
    }));

    const verb = isUpdate ? 'updated' : 'saved';
    showToast(`Preset "${name}" ${verb}`, 'success');
    window.refreshWizardCalibrationOffer?.();
    if (dom.savedPresetName) {
      dom.savedPresetName.textContent = `✓ ${isUpdate ? 'Updated' : 'Saved'} as "${name}"`;
      dom.savedPresetName.style.display = '';
    }
  } catch (err) {
    showToast('Save preset failed: ' + (err.message || String(err)), 'error');
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = 'Save as Preset'; }
  }
}

export function buildPresetPayload() {
  const spawnPayload = buildSpawnPayload();
  if (spawnPayload.backend === 'rapid_mlx') {
    const { api_key: protectedApiKey, ...rapidMlx } = spawnPayload.rapid_mlx;
    spawnPayload.rapid_mlx = rapidMlx;
    if (protectedApiKey) spawnPayload.api_key = protectedApiKey;

    // Sampling is shared preset state even though Rapid-MLX sends its
    // server-facing defaults through the backend branch. Keep the canonical
    // top-level fields for preset round-trips and editor parity.
    const h = wizardState.hardware;
    Object.assign(spawnPayload, {
      temperature: h.temperature != null ? h.temperature : null,
      top_p: h.topP != null ? h.topP : null,
      top_k: h.topK != null ? h.topK : null,
      min_p: h.minP != null ? h.minP : null,
      repeat_penalty: h.repeatPenalty != null ? h.repeatPenalty : null,
      presence_penalty: h.presencePenalty != null ? h.presencePenalty : null,
      max_tokens: h.maxTokens != null ? h.maxTokens : null,
      seed: h.seed != null ? h.seed : null,
    });
  }
  return {
    name: 'Setup wizard preset',
    ...spawnPayload,
  };
}

// ── Health check ──────────────────────────────────────────────────────────────

async function runHealthCheck() {
  if (!dom.healthCheckBtn) return;
  const btn = dom.healthCheckBtn, orig = btn.textContent;
  btn.disabled = true; btn.textContent = 'Running…';
  try {
    const headers = window.authHeaders ? { ...window.authHeaders(), 'Content-Type': 'application/json' } : { 'Content-Type': 'application/json' };
    const resp = await fetch('/api/benchmark', { method: 'POST', headers });
    if (!resp.ok) { showToast('Health check failed: ' + await resp.text().catch(()=>''), 'error'); return; }
    const data = await resp.json();
    const verdict = (data.verdict || '').toLowerCase();
    const details = [
      data.prompt_tokens_per_second ? `Prompt: ${data.prompt_tokens_per_second.toFixed(1)} t/s` : '',
      data.gen_tokens_per_second ? `Gen: ${data.gen_tokens_per_second.toFixed(1)} t/s` : '',
      data.time_to_first_token_ms ? `TTFT: ${data.time_to_first_token_ms.toFixed(0)} ms` : '',
    ].filter(Boolean).join(' · ');
    showToast(`Health: ${verdict || 'complete'}`, verdict === 'good' ? 'success' : verdict === 'poor' ? 'error' : 'warning', details || (data.hints?.[0] ?? ''));
  } catch (err) { showToast('Health check error', 'error', err.message || String(err)); }
  finally { btn.disabled = false; btn.textContent = orig; }
}

// ── Preset parameters review (step 5) ─────────────────────────────────────────

export function _renderPresetParamsStep() {
  const container = dom.presetParamsTable;
  if (!container) return;

 // Pre-fill preset name from model filename if empty
   if (dom.presetNameInput && !dom.presetNameInput.value.trim()) {
     const m = wizardState.model;
     const ctx = wizardState.hardware.contextSize || 0;
     const modelFile = (m.path || m.hfRepo || '').split(/[/\\]/).pop() || '';
     const base = (modelFile || '').replace(/\.gguf$/i, '').trim();
     const name = base && ctx
       ? base + '-' + formatCtx(ctx).toLowerCase()
       : base || 'My Preset';
     dom.presetNameInput.value = name;
   }
  if (dom.savedPresetName) dom.savedPresetName.style.display = 'none';

  const h = wizardState.hardware, m = wizardState.model;
  const rapid = wizardState.engine.selected === 'rapid_mlx';
  const arch = getSizingArch();

  const modelDisplay = m.source === 'hf'
    ? (m.hfFile ? m.hfFile.split('/').pop() : (m.hfRepo || '—'))
    : (m.path ? m.path.split(/[\\/]/).pop() || m.path : '—');

  const gpuDisplay = h.gpuLayers === 'manual'
    ? String(h.gpuLayersManual ?? '—')
    : h.gpuLayers;

  const kvK = (h.cacheTypeK || 'q8_0').toUpperCase();
  const kvV = (h.cacheTypeV || 'q8_0').toUpperCase();
  const fmtSampling = v => v != null ? String(parseFloat(Number(v).toFixed(4))) : '— (server default)';

  const sections = [
    {
      label: 'Model',
      rows: [
        { label: 'Engine', value: rapid ? 'Rapid-MLX' : 'llama.cpp' },
        { label: rapid ? 'Source' : 'File', value: modelDisplay },
        ...(m.source === 'hf' ? [{ label: 'HF repo', value: m.hfRepo || '—' }] : []),
        ...(m.mmprojPath ? [{ label: 'mmproj', value: m.mmprojPath.split(/[\\/]/).pop() || m.mmprojPath }] : []),
        ...(wizardState.model.chatTemplatePath ? [{ label: 'Chat template', value: wizardState.model.chatTemplatePath.split(/[\\/]/).pop() || wizardState.model.chatTemplatePath }] : []),
      ],
    },
    ...(!rapid ? [{
      label: 'Hardware',
      rows: [
        { label: 'GPU layers', value: gpuDisplay },
        { label: 'Context size', value: `${h.contextSize.toLocaleString()} tokens` },
        { label: 'Batch / uBatch', value: `${h.batchSize} / ${h.ubatchSize}` },
        { label: 'Parallel slots', value: String(h.parallelSlots) },
        { label: 'KV cache K', value: kvK },
        { label: 'KV cache V', value: kvV },
        ...(h.flashAttn && h.flashAttn !== 'auto' ? [{ label: 'Flash Attn', value: h.flashAttn }] : []),
        ...(h.kvUnified != null ? [{ label: 'KV unified', value: h.kvUnified ? 'On' : 'Off' }] : []),
        ...(h.fitEnabled != null ? [{ label: 'Fit', value: h.fitEnabled ? 'On' : 'Off' }] : []),
        ...(h.mlock ? [{ label: 'mlock', value: 'Yes' }] : []),
      ...(h.prio != null ? [{ label: 'Priority', value: ['Normal', 'Medium', 'High', 'Realtime'][h.prio] ?? String(h.prio) }] : []),
      ...(h.loadMode ? [{ label: 'Load mode', value: h.loadMode }] : []),
      ...(h.verbosity != null ? [{ label: 'Log verbosity', value: String(h.verbosity) }] : []),
      ...(h.ctxCheckpoints != null ? [{ label: 'Context checkpoints', value: String(h.ctxCheckpoints) }] : []),
      ...(h.checkpointMinStep != null ? [{ label: 'Checkpoint minimum step', value: String(h.checkpointMinStep) }] : []),
      ...(h.cacheReuse != null ? [{ label: 'Cache reuse threshold', value: String(h.cacheReuse) }] : []),
      ...(h.noContBatching ? [{ label: 'Continuous batching', value: 'Disabled' }] : []),
      ...(h.swaFull ? [{ label: 'Full SWA cache', value: 'Enabled' }] : []),
        ...(h.nCpuMoe > 0 && arch.nExperts > 0 ? [{ label: 'MoE CPU offload', value: `${h.nCpuMoe} of ${arch.nLayers} layers` }] : []),
        ...(h.tensorSplit ? [{ label: 'Tensor split', value: h.tensorSplit }] : []),
        ...(h.fitTarget ? [{ label: '--fit-target', value: `${h.fitTarget} MB` }] : []),
        ...(h.cacheRam != null ? [{ label: '--cache-ram', value: h.cacheRam < 0 ? 'no limit' : h.cacheRam === 0 ? 'disabled' : `${h.cacheRam} MiB` }] : []),
      ],
    }] : []),
    {
      label: 'Sampling',
      rows: [
        { label: 'Temperature', value: fmtSampling(h.temperature) },
        { label: 'Top-P', value: fmtSampling(h.topP) },
        { label: 'Top-K', value: fmtSampling(h.topK) },
        { label: 'Min-P', value: fmtSampling(h.minP) },
        { label: 'Repeat penalty', value: fmtSampling(h.repeatPenalty) },
        { label: 'Presence penalty', value: fmtSampling(h.presencePenalty) },
        { label: 'Max tokens', value: h.maxTokens != null ? String(h.maxTokens) : '— (server default)' },
        { label: 'Seed', value: h.seed != null ? String(h.seed) : '— (random)' },
      ],
    },
  ];

  // Thinking section only when something is set
  const hasThinking = h.enableThinking != null || h.preserveThinking != null ||
                      h.reasoningMode != null || h.reasoningBudget != null;
  if (hasThinking) {
    const rows = [];
    if (h.enableThinking != null) rows.push({ label: 'Enable thinking', value: h.enableThinking ? 'Yes' : 'No' });
    if (h.preserveThinking != null) rows.push({ label: 'Preserve thinking', value: h.preserveThinking ? 'Yes' : 'No' });
    if (h.reasoningMode) rows.push({ label: 'Reasoning mode', value: h.reasoningMode });
    if (h.reasoningBudget != null) rows.push({ label: 'Reasoning budget', value: `${h.reasoningBudget} tokens` });
    if (h.reasoningBudgetMessage) rows.push({ label: 'Budget message', value: h.reasoningBudgetMessage });
    if (rows.length) sections.push({ label: 'Thinking & Reasoning', rows });
  }

  const outputMode = _structuredOutputMode();
  if (outputMode) {
    sections.push({
      label: 'Response Shaping',
      rows: [{
        label: outputMode === 'grammar' ? 'Grammar' : 'JSON schema',
        value: outputMode === 'grammar'
          ? ((h.grammar || '').split('\n')[0] || 'configured')
          : ((h.jsonSchema || '').split('\n')[0] || 'configured'),
      }],
    });
  }

  sections.push({
    label: 'Network & Identity',
    rows: [
      { label: 'Port', value: String(wizardState.access.port || 8001) },
      { label: 'Bind host', value: wizardState.access.bindHost === '0.0.0.0' ? '0.0.0.0 (LAN visible)' : '127.0.0.1 only' },
      { label: 'Alias', value: h.alias || '(derived from filename)' },
      { label: 'API key', value: wizardState.access.apiKey ? `${wizardState.access.apiKey.slice(0, 4)}…${wizardState.access.apiKey.slice(-4)}` : 'Not set' },
    ],
  });

  const specType = dom.specTypeSelect?.value || '';
  if (!rapid && specType) {
      const rows = [{ label: 'Type', value: specType }];
      // Show draft model info for draft-model or MTP modes with external draft model
      if (dom.draftModelInput?.value) {
        const fileName = dom.draftModelInput.value.split(/[\\/]/).pop() || dom.draftModelInput.value;
        if (specType === 'draft-model') rows.push({ label: 'Draft model', value: fileName });
        else if (specType.includes('draft-mtp')) rows.push({ label: 'Draft model', value: fileName });
      }
      sections.push({ label: 'Speculative Decoding', rows });
    }

    // Rapid-MLX Phase 7 advanced settings summary
    if (rapid) {
      const rapidRows = [];
       // Show requested vs effective when reasoning overrides the selection
       const requestedKv = h.kvCacheDtype || 'int4';
       if (requestedKv !== 'int8') {
        rapidRows.push({ label: 'KV cache dtype', value: requestedKv.toUpperCase() + ' → INT8 (reasoning profile)' });
      } else {
        rapidRows.push({ label: 'KV cache dtype', value: requestedKv.toUpperCase() });
      }
     rapidRows.push({ label: 'Prompt storage', value: h.turboquantMode === 'auto' ? 'Auto — runtime default' : h.turboquantMode === 'none' ? 'Standard (int4)' : h.turboquantMode === 'k8v4' ? 'TurboQuant K8V4' : 'TurboQuant V-only' });
     // Vocabulary here must match USE_CASE_TO_PROFILE, which is what the page-1 use-case
     // cards actually set. This row was written against the canonical estimator keys
     // (coding_agent, batch_eval, ...) that the wizard never produces, so every lookup
     // missed and the review step printed the raw snake_case key — for the default
     // selection too, since the guard compared against a value nothing sets.
     if (h.workloadScenario && h.workloadScenario !== 'interactive_coding_agent') {
       rapidRows.push({ label: 'Workload scenario', value: { interactive_coding_agent: 'Interactive Coding Agent', general_chat: 'General Chat', roleplay_storytelling: 'Roleplay / Storytelling' }[h.workloadScenario] || h.workloadScenario });
     }
     if (h.samplingMode && h.samplingMode !== 'auto') {
       rapidRows.push({ label: 'Sampling mode', value: { general: 'General', coding: 'Coding/Agentic', precise: 'Precise/Deterministic', creative: 'Creative/Roleplay', custom: 'Custom' }[h.samplingMode] || h.samplingMode });
     }
      if (h.rapidReasoningMode) {
        rapidRows.push({ label: 'Reasoning quality profile', value: 'On' });
        rapidRows.push({ label: 'Thinking output', value: h.rapidReasoningMode === 'off' ? 'Disabled' : 'Allowed' });
      }
     if (rapidRows.length > 0) {
       sections.push({ label: 'Rapid-MLX advanced', rows: rapidRows });
     }
   }

  if (!rapid && h.extraArgs) {
    sections.push({ label: 'Extra', rows: [{ label: 'Extra command-line arguments', value: h.extraArgs }] });
  }

  const stateLegend = document.getElementById('spawn-full-config-state');
  if (stateLegend) {
    stateLegend.textContent = rapid
      ? 'Requested · runtime-effective in command preview'
      : 'Requested · estimator-effective in VRAM budget';
  }

  container.innerHTML = '';

  for (const section of sections) {
    if (!section.rows.length) continue;
    const block = document.createElement('div');
    block.className = 'summary-list';
    block.style.marginTop = '8px';

    const hdr = document.createElement('div');
    hdr.className = 'summary-list-header';
    const sectionLabel = document.createElement('span');
    sectionLabel.textContent = section.label;
    const sectionState = document.createElement('span');
    sectionState.className = 'summary-state';
    sectionState.textContent = rapid && section.label === 'Rapid-MLX advanced'
      ? 'requested / runtime-effective'
      : section.label === 'Hardware' && !rapid
        ? 'requested / estimator-effective'
        : 'requested';
    hdr.append(sectionLabel, sectionState);
    block.appendChild(hdr);

    for (const r of section.rows) {
      const row = document.createElement('div');
      row.className = 'summary-row';
      const lbl = document.createElement('span');
      lbl.className = 'summary-label';
      lbl.textContent = r.label;
      const val = document.createElement('span');
      val.className = 'summary-value';
      val.textContent = r.value;
      row.appendChild(lbl);
      row.appendChild(val);
      block.appendChild(row);
    }
    container.appendChild(block);
  }

  // Edit shortcuts
  const editRow = document.createElement('div');
  editRow.className = 'summary-edit-row';
  editRow.style.cssText = 'display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;padding:0;';
  [
    { label: 'Edit model', step: 0 },
    { label: 'Edit hardware', step: 1 },
    { label: 'Edit sampling', step: 1, focusId: 'spawn-temperature' },
  ].forEach(({ label, step, focusId }) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'btn-wizard-tertiary';
    btn.textContent = label;
    btn.addEventListener('click', () => {
      showStep(step);
      if (focusId) setTimeout(() => document.getElementById(focusId)?.focus(), 50);
    });
    editRow.appendChild(btn);
  });
  container.appendChild(editRow);
}
