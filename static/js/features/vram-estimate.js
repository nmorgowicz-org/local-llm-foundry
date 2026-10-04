// Centralized VRAM estimation using the Rust backend as the single source of truth.
// No local VRAM formulas — all math comes from /api/vram-estimate.
//
// Phase 5a Part 5: Cross-surface equality — every UI surface must call this
// canonical builder with matching parameters so all surfaces display identical
// MemoryBreakdown values for the same model/context/backend.

import './toast.js';
import {
  RAPID_MLX_DEFAULT_SPECULATIVE_TOKENS,
  RAPID_MLX_TEXT_PREFILL_STEP_SIZE,
  RAPID_MLX_VISION_PREFILL_STEP_SIZE,
} from './rapid-mlx-prefill.js';

let debounce = null;
let currentRequestId = 0;

const slotRequestIds = new Map();
// Extract the persisted Rapid settings which materially change an estimate.
// Keeping this at the canonical request boundary prevents the wizard, preset
// editor, and welcome cards from quietly using different defaults.
export function rapidEstimatePolicyFromConfig(rapidMlx = {}) {
  return {
    kv_cache_dtype: rapidMlx.kv_cache_dtype || null,
    turboquant_mode: rapidMlx.turboquant_mode || null,
    // llama-monitor always emits Rapid's qualified --reasoning quality/KV profile.
    // A persisted "off" now means a separate thinking-output opt-out.
    reasoning_mode: true,
    // No workload_scenario: a persisted RapidMlxConfig has no such field, so reading one here
    // only ever produced null. Scenario is a wizard-side input, derived from the page-1
    // use-case cards; every other surface takes the estimator default.
    retained_cache_mib: rapidMlx.prefix_cache_enabled === false ? 0 : (rapidMlx.retained_cache_mib ?? 8192),
        prefill_step_size: Number(rapidMlx.prefill_step_size ?? (
            rapidMlx.mllm_vision === 'on'
                ? RAPID_MLX_VISION_PREFILL_STEP_SIZE
                : RAPID_MLX_TEXT_PREFILL_STEP_SIZE
        )),
    speculative_config: rapidMlx.speculative_config || null,
  };
}

export function rapidEstimatePolicyFromWizardHardware(hardware = {}) {
  const speculativeConfig = hardware.speculativeEnabled ? {
    method: 'mtp',
    model: hardware.speculativeSource === 'external' ? (hardware.speculativeModel || null) : null,
    num_speculative_tokens: Number(hardware.speculativeTokens || RAPID_MLX_DEFAULT_SPECULATIVE_TOKENS),
    disable_auto_k: !!hardware.speculativeDisableAutoK,
  } : null;
  return {
    kv_cache_dtype: hardware.kvCacheDtype || null,
    turboquant_mode: hardware.turboquantMode || null,
    reasoning_mode: true,
    workload_scenario: hardware.workloadScenario || null,
    speculative_config: speculativeConfig,
    retained_cache_mib: Number(hardware.retainedCacheMib ?? 8192),
    prefill_step_size: Number(hardware.prefillStepSize ?? 512),
  };
}

// ── Canonical body builder ───────────────────────────────────────────────────
//
// Builder item 6: single source of truth for /api/vram-estimate parameters.
// Every JS surface (wizard, presets, welcome cards, Model Library, HF preview)
// must use this to construct its request body. Identical inputs → identical
// API result → identical displayed breakdown.
//
// @param {Object} params — canonical estimate parameters
// @param {string} params.backend — 'llama_cpp' or 'rapid_mlx'
// @param {string} params.model_path — local path or HF-repo-style alias (Rapid)
// @param {string|null} params.hf_repo_id — explicit HF repo (optional when model_path is alias)
// @param {string|null} params.hf_file_path — HF file path for GGUF introspection
// @param {number|null} params.model_size_bytes — explicit size override
// @param {number} params.n_ctx — context size (tokens)
// @param {number} params.parallel_slots — parallel generation slots
// @param {number} params.ubatch_size — llama.cpp ubatch size
// @param {number} params.prefill_step_size — Rapid prefill working width
// @param {string} params.ctk — KV key quant (llama.cpp only; e.g. 'q8_0')
// @param {string} params.ctv — KV value quant (llama.cpp only; e.g. 'q8_0')
// @param {string|null} params.kv_cache_dtype — Rapid KV dtype: 'bf16'|'int8'|'int4'
// @param {boolean} params.reasoning_mode — Rapid reasoning mode (pins KV to int8)
// @param {string|null} params.turboquant_mode — Rapid TurboQuant: 'v4'|'k8v4'|'none'
// @param {number} params.n_cpu_moe — MoE layers on CPU
// @param {number|null} params.gpu_layers — GPU layer count (-1 = auto/all)
// @param {number} params.available_vram_bytes — available VRAM/RAM budget
// @param {number} params.available_ram_bytes — available system RAM
// @param {boolean} params.is_unified_memory — unified memory (Apple Silicon)
// @param {string|null} params.mmproj_path — vision projector path
// @param {number} params.mmproj_bytes — vision projector size override
// @param {string|null} params.workload_scenario — scenario key (e.g. 'coding_agent')
// @param {number|null} params.rapid_planning_context_tokens — Rapid planning context tokens
// @param {number|null} params.rapid_retained_cache_tokens — Rapid retained cache tokens
// @param {string|null} params.client_type — 'app' or 'external_client'
// @param {Object|null} params.speculative_config — typed Rapid speculative configuration
// @param {string|null} params.hf_repo_revision — pinned revision (Phase 8B2)
// @param {string|null} params.hf_quant_label — quant label/variant name (Phase 8B2)
// @returns {Object} request body ready for JSON.stringify
export function buildEstimateBody(params) {
  const body = {
    backend: params.backend || 'llama_cpp',
    model_path: params.model_path || '',
    n_ctx: params.n_ctx || 4096,
    parallel_slots: params.parallel_slots || 1,
    n_cpu_moe: params.n_cpu_moe || 0,
    gpu_layers: Number.isFinite(params.gpu_layers) ? params.gpu_layers : -1,
    available_vram_bytes: params.available_vram_bytes || 0,
    available_ram_bytes: params.available_ram_bytes || 0,
    is_unified_memory: !!params.is_unified_memory,
    mmproj_path: params.mmproj_path || '',
    mmproj_bytes: params.mmproj_bytes || 0,
  };

  const isMlxBackend = params.backend === 'mlx' || params.backend === 'rapid_mlx';

  if (isMlxBackend) {
    body.prefill_step_size = params.prefill_step_size || 512;
  } else {
    body.ubatch_size = params.ubatch_size || 2048;
  }

  // llama.cpp KV quant (ignored by Rapid path; kept for backward compatibility).
  if (!isMlxBackend) {
    body.ctk = params.ctk || 'q8_0';
    body.ctv = params.ctv || 'q8_0';
  }

  // HF coordinates (pre-download introspection / Rapid alias resolution).
  // llama.cpp requires both repo_id and file_path.
  // MLX backend only needs repo_id (repo IS the model, no specific file_path).
  if (params.hf_repo_id && params.hf_file_path) {
    body.hf_repo_id = params.hf_repo_id;
    body.hf_file_path = params.hf_file_path;
    body.model_size_bytes = params.model_size_bytes || 0;
  } else if (isMlxBackend && params.hf_repo_id) {
    body.hf_repo_id = params.hf_repo_id;
    body.model_size_bytes = params.model_size_bytes || 0;
  }

  // Phase 8B2: revision pin and quant label preservation.
  if (params.hf_repo_revision) body.hf_repo_revision = params.hf_repo_revision;
  if (params.hf_quant_label) body.hf_quant_label = params.hf_quant_label;

  // Rapid-MLX execution policy (Phase 5a Part 5: requested/effective distinction).
  if (params.backend === 'rapid_mlx') {
    if (params.kv_cache_dtype) body.kv_cache_dtype = params.kv_cache_dtype;
    if (params.reasoning_mode) body.reasoning_mode = params.reasoning_mode;
    if (params.turboquant_mode) body.turboquant_mode = params.turboquant_mode;
    if (Number.isFinite(Number(params.retained_cache_mib))) {
      body.retained_cache_mib = Number(params.retained_cache_mib);
    }
  }

    // Workload scenario (Phase 5a Part 4: scenario-aware estimates).
    if (params.workload_scenario) body.workload_scenario = params.workload_scenario;
    if (params.rapid_planning_context_tokens != null)
        body.rapid_planning_context_tokens = params.rapid_planning_context_tokens;
    if (params.rapid_retained_cache_tokens != null)
        body.rapid_retained_cache_tokens = params.rapid_retained_cache_tokens;
    if (params.client_type) body.client_type = params.client_type;
    if (params.speculative_config) body.speculative_config = params.speculative_config;
    // Phase 5b Part C: max_cache_blocks from preset for prelaunch estimates.

    return body;
}

// ── Fetch estimate (canonical API call) ──────────────────────────────────────
//
// All surfaces use this to ensure identical API consumption.
async function fetchEstimate(body) {
  const isMlxBackend = body.backend === 'mlx' || body.backend === 'rapid_mlx';
  const hasMlxInput = isMlxBackend && body.hf_repo_id;
  const hasLlamaInput = body.hf_repo_id && body.hf_file_path;
  if (!body.model_path && !hasLlamaInput && !hasMlxInput) return null;
  try {
    const headers = (window.authHeaders ? window.authHeaders() : {}) ;
    const res = await fetch('/api/vram-estimate', {
      method: 'POST',
      headers: { ...headers, 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) return null;
    const data = await res.json();
    if (!data.ok || !data.total_bytes) return null;
    return data;
  } catch {
    return null;
  }
}

// ── Legacy wrapper for wizardState ────────────────────────────────────────────
//
// Builds the request body for /api/vram-estimate from the given wizardState.
// Provided for backward compatibility with existing wizard wiring.

function buildEstimateBodyFromWizardState(state) {
  const hw = state.hardware;
  const m = state.model;
  const gpuLayers = hw.gpuLayers === 'manual'
    ? (Number.isFinite(Number(hw.gpuLayersManual)) ? Number(hw.gpuLayersManual) : -1)
    : -1;

  const modelPath = m.path || m.localPath || '';
  const backend = state.engine?.selected === 'rapid_mlx' ? 'rapid_mlx' : 'llama_cpp';

  return buildEstimateBody({
    backend,
    model_path: modelPath,
    n_ctx: hw.contextSize || 4096,
    parallel_slots: hw.parallelSlots || 1,
    ubatch_size: hw.ubatchSize || 2048,
    ctk: hw.cacheTypeK || 'q8_0',
    ctv: hw.cacheTypeV || 'q8_0',
    n_cpu_moe: hw.nCpuMoe || 0,
    gpu_layers: gpuLayers,
    available_vram_bytes: state.vram?.available || 0,
    available_ram_bytes: state.vram?.availableRam || 0,
    is_unified_memory: state.vram?.isUnifiedMemory || false,
    mmproj_path: m.mmprojPath || '',
    mmproj_bytes: m.mmprojBytes || 0,
    // A typed or handed-off Rapid-MLX repo has no originRepo (that is set from a local
    // model's provenance), so fall back to the repo the user actually chose. Without it
    // fetchEstimate bails before calling the server and the memory estimate stays blank.
    hf_repo_id: m.originRepo
      || (backend === 'rapid_mlx' && !modelPath ? (m.hfRepo || null) : null),
    hf_file_path: m.hfFile || null,
    model_size_bytes: m.modelBytes || null,
    // Rapid execution policy fields (when wizard populates them).
    ...rapidEstimatePolicyFromWizardHardware(hw),
    // Explicit planning fields are intentionally separate from the profile ID.
    rapid_planning_context_tokens: hw.rapidPlanningContextTokens || null,
    rapid_retained_cache_tokens: hw.rapidRetainedCacheTokens || null,
    client_type: hw.clientType || null,
  });
}

// Public: schedule an estimate (debounced). Callback receives estimate or null.
export function scheduleEstimate(
  wizardState,
  onEstimate,
  { debounceMs = 120, force = false, slot = '' } = {},
) {
  if (debounce && !force) {
    clearTimeout(debounce);
  }

  // Staleness is judged per slot. With one global counter, any other surface asking for an
  // estimate a moment later (the main VRAM panel, for instance) made this caller's response
  // look stale and silently dropped it, leaving that surface blank. Callers that render into
  // their own element pass a slot; callers without one keep sharing the default slot.
  const reqId = slot ? (slotRequestIds.get(slot) || 0) + 1 : ++currentRequestId;
  if (slot) slotRequestIds.set(slot, reqId);

  const doWork = async () => {
    const body = buildEstimateBodyFromWizardState(wizardState);
    const est = await fetchEstimate(body);
    const latest = slot ? slotRequestIds.get(slot) : currentRequestId;
    if (reqId === latest) {
      onEstimate(est);
    }
  };

  if (force) {
    doWork();
    debounce = null;
  } else {
    debounce = setTimeout(doWork, debounceMs);
  }
}

// Public: cancel pending debounce (used during cleanup).
export function cancelEstimate() {
  if (debounce) {
    clearTimeout(debounce);
    debounce = null;
  }
}
