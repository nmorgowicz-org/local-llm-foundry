console.log('[WIZARD] MODULE LOADED');
/* global DOMPurify */
// Set by spawn-wizard-hf-browse.js so this module can re-trigger the quant
// advisor once memory/VRAM data arrives, without a circular import. Declared
// before any imports since spawn-wizard-hf-browse.js calls
// setOnMemoryAvailabilityReady() at its own module top-level, and ES module
// `let` bindings are in the temporal dead zone until their declaration runs.
export let onMemoryAvailabilityReady = null;
export function setOnMemoryAvailabilityReady(fn) {
  onMemoryAvailabilityReady = fn;
}

import { buildArchitectureLabel, isMoEEligible } from './setup-view.js';
import { getPlatformInfo } from '../core/platform-info.js';
import { readLastStatus } from './template-autoupdater.js';
import { RAPID_MLX_DEFAULT_SPECULATIVE_TOKENS } from './rapid-mlx-prefill.js';
import {
  bindRapidMlxAdvancedControls,
  syncRapidSpeculativeFields,
  applyReasoningModeLock,
  applyRapidMlxDefaults,
  renderRapidExclusionWarnings,
  scheduleRapidMlxProfileFetch,
  refreshRapidMlxSidecars,
} from './spawn-wizard-rapid-mlx.js';
import {
  kvBpe,
  formatCtx,
  formatParams,
  formatGB,
  formatVramTotal,
  formatBytes,
  formatSpeed,
} from './spawn-wizard-format.js';
import { openCardPanel, _closeCardPanel } from './spawn-wizard-model-card.js';
export { openCardPanel };
import { configureMlxWizardIA, applyMlxTierVisibility } from './spawn-wizard-mlx-ia.js';
import { configureLlamaWizardIA, applyLlamaTierVisibility } from './spawn-wizard-llama-ia.js';
import { controlsForLoader, controlsForView, applyEffectiveLocks } from './spawn-wizard-groups.js';
import { createSettingStateRegistry } from './spawn-wizard-setting-state.js';
import { initGuidedCards, refreshGuidedCapabilityCards } from './spawn-wizard-guided.js';
import {
  _platformInfo,
  setWizardPlatformInfo,
  _checkBinaryPrereq,
  _updateSpawnBtnForPrereq,
  _downloadBinaryForWizard,
} from './spawn-wizard-binary-prereq.js';
import {
  applyWizardSuggestion,
  applyCalibrationPatch,
  autoTuneWizard,
  runBatchSweep,
  runDepthSweep,
} from './spawn-wizard-tuning.js';
import {
  resetOriginState,
  startOriginResolve,
  setOriginResolverPromise,
  awaitOriginResolve,
  _autoResolveHfOrigin,
  _refreshHfOriginSection,
  _attachOriginTags,
} from './spawn-wizard-hf-origin.js';
import {
  _applyCustomChatTemplate,
  autoInstallChatTemplate,
} from './spawn-wizard-chat-template.js';
import { loadThirdPartyModels } from './spawn-wizard-third-party-import.js';
import {
  initHfBrowseWidgets,
  bindHfSearchControls,
  bindQuantizerEditor,
  loadCommunityPicks,
  triggerQuantAdvisor,
  triggerHfFileFetch,
  _applyScopeDefaultForEngine,
} from './spawn-wizard-hf-browse.js';
import { bindHfDownloadPanel } from './spawn-wizard-hf-download.js';
import {
  _openHwTagPicker,
  resetTagsRowOrigin,
} from './spawn-wizard-hf-tags.js';
import {
  renderHardwareModelHeader,
  renderContextChipRow,
  _autoDiscoverLocalModelQuants,
  _renderQuantSwapActions,
  resetQuantSwapSearchState,
} from './spawn-wizard-hardware-model.js';
import { renderMmprojSection, _bestMmprojForModel } from './spawn-wizard-mmproj.js';
import {
  renderMtpSection,
  _bestDraftForModel,
  _checkGemma4MtpDraft,
} from './spawn-wizard-mtp-draft.js';
import {
  updateVramDisplay,
  updateMoeSliderVisuals,
} from './spawn-wizard-vram-display.js';

import {
  bindCtxQuickPicks,
  updateCtxQuickPickActive,
  updateCtxModelMaxHint,
  updateCtxTrainWarning,
} from './spawn-wizard-context-fit.js';
import { triggerAutoSize } from './spawn-wizard-autosize.js';
import {
  renderSummary,
  _renderPresetParamsStep,
  _bindSamplingFields,
  saveAsPreset,
  _fetchAndApplyModelSamplingDefaults,
} from './spawn-wizard-review-step.js';
// Re-exported for the Playwright test contract (tests dynamically import
// buildPresetPayload from '/js/features/spawn-wizard.js').
export { buildPresetPayload } from './spawn-wizard-review-step.js';
import { _renderSpawnConfigCard, spawnServer, resetSpawnStatus } from './spawn-wizard-spawn.js';
// Re-exported for the Playwright test contract (tests dynamically import
// these from '/js/features/spawn-wizard.js').
export { buildSpawnPayload, launchPortForPayload, supportsTunePanelForPayload } from './spawn-wizard-spawn.js';

// ── M4: All settings drawer ───────────────────────────────────────────────────
// Wraps decision-critical controls in always-open cards; everything else
// goes into a collapsible "All settings" drawer.

function _renderAllSettingsDrawer() {
  const drawer = document.getElementById('all-settings-drawer');
  const btn = document.getElementById('all-settings-btn');
  const countEl = document.getElementById('all-settings-count');
  const changedEl = document.getElementById('all-settings-changed');
  const body = document.getElementById('all-settings-body');
  const group = document.getElementById('all-settings-group');
  if (!drawer || !btn || !countEl || !changedEl || !body || !group) return;

  const loader = wizardState.engine.selected || 'llama_cpp';
  const settings = controlsForView(loader, 'guided').filter(control => control.view !== 'card');
  countEl.textContent = String(settings.length);
  settingStateRegistry.mount(dom.overlay || document, controlsForLoader(loader));

  // The IA engines own the canonical rows; the shared drawer only relocates
  // their wrapper, never clones individual inputs or creates proxy controls.
  const sourceId = loader === 'rapid_mlx' ? 'spawn-rapid-advanced-fields' : 'spawn-advanced-fields';
  const source = document.getElementById(sourceId);
  ['spawn-advanced-fields', 'spawn-rapid-advanced-fields'].forEach(id => {
    const candidate = document.getElementById(id);
    if (candidate && id !== sourceId) candidate.style.display = 'none';
  });
  if (source && source.parentElement !== group) {
    group.appendChild(source);
  }
  if (source) source.style.display = loader === 'rapid_mlx' ? 'block' : '';
  drawer.style.display = source ? '' : 'none';

  const refreshChangedCount = () => {
    const changed = settingStateRegistry.changedCount(settings);
    changedEl.textContent = `${changed} changed`;
    changedEl.dataset.count = String(changed);
  };
  refreshChangedCount();
  group._refreshAllSettingsChanged = refreshChangedCount;

  // Toggle
  if (btn.dataset.bound !== '1') {
    btn.dataset.bound = '1';
    btn.addEventListener('click', () => {
      const isOpen = body.style.display === 'block';
      body.style.display = isOpen ? 'none' : 'block';
      btn.setAttribute('aria-expanded', String(!isOpen));
    });
  }

  const overlay = dom.overlay || document;
  if (!overlay.dataset?.settingStateBound) {
    if (overlay.dataset) overlay.dataset.settingStateBound = '1';
    const sync = event => {
      const control = event.target?.closest?.('select, input, textarea') || event.target;
      const entry = settingStateRegistry.syncControl(control);
      if (!entry) return;
      updateVramDisplay();
      renderContextChipRow();
      refreshStepGuardrails();
      group._refreshAllSettingsChanged?.();
    };
    overlay.addEventListener('input', sync);
    overlay.addEventListener('change', sync);
  }

  // Initially hide
  if (!btn.getAttribute('aria-expanded')) btn.setAttribute('aria-expanded', 'false');
}

export { _renderAllSettingsDrawer };

export const settingStateRegistry = createSettingStateRegistry();

const PRO_CATEGORIES = [
  'Model & compatibility',
  'Memory & context',
  'Performance',
  'Generation & reasoning',
  'Tools & conversation formatting',
  'Network & observability',
  'Advanced',
];

const PRO_WRAPPER_IDS = ['spawn-advanced-fields', 'spawn-rapid-advanced-fields'];
const PRO_RELOCATED_SELECTORS = [
  '#hw-decision-ctx', '#hw-decision-kv', '#hw-decision-vision', '#hw-decision-speed',
  '#wizard-step-1 > .wizard-main > .sampling-params-section',
];
const PRO_RELOCATED_FIELD_IDS = ['hw-quant-select', 'spawn-kv-unified', 'hw-mtp-depth'];
const PRO_SHARED_ACCESS_FIELD_IDS = ['spawn-port', 'spawn-bind-host', 'spawn-alias', 'spawn-api-key'];
const PRO_DECISION_SURFACES = [
  { selector: '#hw-decision-ctx', category: 'Memory & context', controls: ['spawn-context-size'] },
  { selector: '#hw-decision-kv', category: 'Memory & context', controls: ['spawn-cache-type-k', 'spawn-cache-type-v'] },
  { selector: '#hw-decision-vision', category: 'Model & compatibility', controls: ['hw-mmproj-select'] },
  { selector: '#hw-decision-speed', category: 'Performance', controls: ['hw-use-mtp'] },
];
const proOriginalPositions = new Map();

function _rememberProPosition(node) {
  if (!node || proOriginalPositions.has(node)) return;
  proOriginalPositions.set(node, {
    parent: node.parentElement,
    next: node.nextSibling,
    display: node.style.display,
  });
}

function _moveProSurfaces(host, loader) {
  if (loader === 'llama_cpp') {
    document.querySelectorAll(PRO_RELOCATED_SELECTORS.join(',')).forEach(node => {
      _rememberProPosition(node);
      host.appendChild(node);
      node.style.display = '';
    });
  }
  const fieldIds = loader === 'llama_cpp'
    ? [...PRO_RELOCATED_FIELD_IDS, ...PRO_SHARED_ACCESS_FIELD_IDS]
    : PRO_SHARED_ACCESS_FIELD_IDS;
  fieldIds.forEach(id => {
    const control = document.getElementById(id);
    const field = control?.closest('.hardware-field, .hw-field, .sampling-field, .kv-inline-row, .hardware-row');
    if (field && !host.contains(field)) {
      _rememberProPosition(field);
      host.appendChild(field);
      field.style.display = '';
    }
  });
  if (loader !== 'llama_cpp') return;
  PRO_DECISION_SURFACES.forEach(surface => {
    const card = document.querySelector(surface.selector);
    if (!card) return;
    card.dataset.proCategory = surface.category;
    card.dataset.proControlIds = surface.controls.join(' ');
    card.dataset.proSearchText = `${surface.category} ${surface.controls.join(' ')}`;
  });
}

function _renderRapidProNotes(host, loader) {
  host.querySelectorAll('.pro-backend-note').forEach(note => note.remove());
  if (loader !== 'rapid_mlx') return;
  const notes = [
    {
      category: 'Model & compatibility',
      title: 'Rapid-MLX model compatibility',
      body: 'Rapid-MLX uses the resolved MLX artifact and introspected model profile. llama.cpp quantization and mmproj controls are not applicable here.',
    },
  ];
  notes.forEach(note => {
    const card = document.createElement('section');
    card.className = 'pro-backend-note';
    card.dataset.proCategory = note.category;
    card.dataset.proSearchText = `${note.category} ${note.title}`;
    const title = document.createElement('h3');
    title.textContent = note.title;
    const body = document.createElement('p');
    body.textContent = note.body;
    card.append(title, body);
    host.prepend(card);
  });
}

function _restoreProSurfaces() {
  for (const [node, position] of proOriginalPositions) {
    if (!position.parent) continue;
    if (position.next?.parentNode === position.parent) position.parent.insertBefore(node, position.next);
    else position.parent.appendChild(node);
    node.style.display = position.display;
  }
}

function _proFieldForControl(control) {
  return control?.closest('.hardware-field, .sampling-field, .kv-inline-row, .hardware-row, .hw-field') || control;
}

function _refreshProControls() {
  const host = document.getElementById('pro-controls-host');
  if (!host) return;
  const loader = wizardState.engine.selected || 'llama_cpp';
  const descriptors = controlsForView(loader, 'pro');
  const modifiedOnly = document.getElementById('pro-modified-only')?.checked;
  const query = (document.getElementById('pro-filter-input')?.value || '').trim().toLowerCase();
  const fields = [...host.querySelectorAll('[data-pro-category]')];
  fields.forEach(field => {
    const control = field.querySelector('input, select, textarea');
    const controlIds = (field.dataset.proControlIds || control?.id || '').split(/\s+/).filter(Boolean);
    const fieldDescriptors = descriptors.filter(item => controlIds.includes(item.id));
    const dirty = fieldDescriptors.some(item => settingStateRegistry.entries?.get?.(item.semanticId)?.dirty);
    const text = `${field.textContent || ''} ${field.dataset.proSearchText || ''} ${controlIds.join(' ')}`.toLowerCase();
    const visible = (!modifiedOnly || dirty) && (!query || text.includes(query));
    field.classList.toggle('pro-search-hidden', !visible);
    field.classList.toggle('pro-modified', dirty);
  });
}

function _initProShell() {
  const nav = document.getElementById('pro-rail-nav');
  if (!nav || nav.dataset.bound === '1') return;
  nav.dataset.bound = '1';
  nav.replaceChildren();
  PRO_CATEGORIES.forEach((category, index) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'pro-rail-item';
    button.dataset.category = category;
    button.textContent = category;
    button.addEventListener('click', () => {
      nav.querySelectorAll('.pro-rail-item').forEach(item => item.classList.toggle('active', item === button));
      const target = document.querySelector(`#pro-controls-host [data-pro-category="${CSS.escape(category)}"]`);
      const host = document.getElementById('pro-controls-host');
      if (target && host) {
        const behavior = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth';
        host.scrollTo({ top: Math.max(0, target.offsetTop - 8), behavior });
        target.scrollIntoView({ block: 'start', behavior });
      }
    });
    if (index === 0) button.classList.add('active');
    nav.appendChild(button);
  });
  document.getElementById('pro-filter-input')?.addEventListener('input', _refreshProControls);
  document.getElementById('pro-modified-only')?.addEventListener('change', _refreshProControls);
  if (!document.documentElement.dataset.proShortcutBound) {
    document.documentElement.dataset.proShortcutBound = '1';
    document.addEventListener('keydown', event => {
      const overlay = document.getElementById('spawn-wizard-overlay');
      if (!overlay?.classList.contains('open')) return;
      const target = event.target;
      const typing = target?.matches?.('input, textarea, select, [contenteditable="true"]');
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k' && !typing) {
        event.preventDefault();
        document.getElementById('pro-filter-input')?.focus();
        return;
      }
      if (event.key === 'Escape') {
        const filter = document.getElementById('pro-filter-input');
        if (filter?.value) {
          event.preventDefault();
          event.stopPropagation();
          event.stopImmediatePropagation();
          filter.value = '';
          _refreshProControls();
        }
      }
    }, true);
  }
  document.getElementById('pro-reset-all')?.addEventListener('click', () => {
    const loader = wizardState.engine.selected || 'llama_cpp';
    controlsForView(loader, 'pro').forEach(descriptor => settingStateRegistry.reset(descriptor.semanticId));
  document.querySelectorAll('input, select, textarea').forEach(control => {
      control.dispatchEvent(new Event('input', { bubbles: true }));
      control.dispatchEvent(new Event('change', { bubbles: true }));
    });
    _refreshProControls();
  });
}

function renderProLayout(mode = wizardState.viewMode) {
  const layout = document.getElementById('pro-layout');
  const host = document.getElementById('pro-controls-host');
  const drawerGroup = document.getElementById('all-settings-group');
  if (!layout || !host || !drawerGroup) return;
  _initProShell();
  const isPro = mode === 'pro';
  const loader = wizardState.engine.selected || 'llama_cpp';
  const guidedSurface = document.querySelector('.hw-guided-old-layout');
  const stickyBar = document.getElementById('hw-sticky-bar');
  // The Pro surface is a view over the same controls, not a section appended
  // below Guided's cards. Move the shell out of the legacy wrapper once and
  // keep the identity strip as the shared context header while Pro is active.
  if (isPro) {
    _restoreProSurfaces();
    if (guidedSurface?.contains(layout)) {
      (stickyBar || guidedSurface).after(layout);
    }
    if (guidedSurface) guidedSurface.style.display = 'none';
  } else if (guidedSurface) {
    guidedSurface.style.display = 'block';
    _restoreProSurfaces();
  }
  layout.style.display = isPro ? '' : 'none';
  document.querySelectorAll('#hw-decision-ctx, #hw-decision-kv, #hw-decision-vision, #hw-decision-speed')
    .forEach(card => { card.style.display = isPro ? 'none' : ''; });
  const drawer = document.getElementById('all-settings-drawer');
  if (drawer) drawer.style.display = isPro ? 'none' : '';

  PRO_WRAPPER_IDS.forEach(id => {
    const wrapper = document.getElementById(id);
    if (!wrapper) return;
    const activeWrapper = loader === 'rapid_mlx' ? 'spawn-rapid-advanced-fields' : 'spawn-advanced-fields';
    const activeDisplay = loader === 'rapid_mlx' ? 'block' : '';
    if (isPro) {
      wrapper.style.display = id === activeWrapper ? activeDisplay : 'none';
      if (id === activeWrapper) host.appendChild(wrapper);
    } else {
      wrapper.style.display = id === activeWrapper ? activeDisplay : 'none';
      if (id === activeWrapper) drawerGroup.appendChild(wrapper);
    }
  });

  if (isPro) {
    _moveProSurfaces(host, loader);
    if (loader === 'rapid_mlx') {
      document.querySelectorAll('#wizard-step-1 > .wizard-main > .sampling-params-section').forEach(section => {
        _rememberProPosition(section);
        section.style.display = 'none';
      });
    }
    _renderRapidProNotes(host, loader);
    host.querySelectorAll('.hw-decision-card').forEach(card => { card.style.display = ''; });
    // Pro is the explicit power-user view: every canonical category is
    // expanded so search/scroll-spy has a real target and no setting is
    // hidden behind Guided's profile disclosure state.
    host.querySelectorAll('details').forEach(details => { details.open = true; });
    controlsForView(loader, 'pro').forEach(descriptor => {
      const control = document.getElementById(descriptor.id);
      const field = _proFieldForControl(control);
      if (field) field.dataset.proCategory = descriptor.proCategory;
    });
    applyEffectiveLocks(host);
    _refreshProControls();
  }
}

// ── View mode toggle ────────────────────────────────────────────────────────

function _initViewMode() {
  const select = document.getElementById('view-mode-select');
  if (!select) return;

  // Restore from sessionStorage
  try {
    const saved = sessionStorage.getItem('wizard_view_mode');
    if (saved === 'pro' && !select.querySelector('option[value="pro"]')?.disabled) {
      wizardState.viewMode = 'pro';
      select.value = 'pro';
    } else if (saved === 'pro') {
      sessionStorage.removeItem('wizard_view_mode');
    }
  } catch {}

  select.addEventListener('change', () => {
    const newMode = select.value;
    try { sessionStorage.setItem('wizard_view_mode', newMode); } catch {}
    wizardState.viewMode = newMode;
    renderProLayout(newMode);
    const status = document.getElementById('view-mode-status');
    if (status) status.textContent = newMode === 'pro'
      ? 'Power-user controls'
      : 'Guided defaults + All settings drawer';
  });
}

// ── Spawn Wizard Module ───────────────────────────────────────────────────────
// Spawn Llama-Server V2 — complete guided wizard.
//
// Key features:
//  - Use-case selector (agentic / general / roleplay)
//  - Pre-download quant advisor with size + max-context table
//  - Architecture-aware VRAM breakdown (live animated bar)
//  - Context fit modes that translate KV cache precision into plain-language outcomes
//  - MoE expert offload slider with live feedback
//  - Auto-size button pulls backend recommendation
//  - Step validation before advancing

import { openDeferredFileBrowser, openModelFileBrowser } from './file-browser-launcher.js';
import { showToast, showToastWithActions, resolveNotification } from './toast.js';
import Router, { routeForCurrentView } from './router.js';
import { scheduleEstimate, cancelEstimate, buildEstimateBody, rapidEstimatePolicyFromWizardHardware } from './vram-estimate.js';
import { openEvidenceDrawer, openEstimateEvidenceDrawer, evidenceFromCommandPreview } from './evidence-drawer.js';
import { setTuneConfig, showTunePanel } from './tune-panel.js';
import { renderSuggestionCards } from './tuning-cards.js';
import { setHeaderMode } from './attach-detach.js';
import { lastCapabilities, lastSystemMetrics } from '../core/app-state.js';
let llamaBinaryCapabilitiesPromise = null;
let llamaBinaryCapabilities = null;
import {
  hfStartDownload,
  hfShowDownloadPanel,
  hfHideDownloadPanel,
} from './hf-browse.js';

// ── VRAM math (client-side, for instant slider feedback) ──────────────────────

// VRAM math is now centralized in the Rust backend (vram_estimator).
// The spawn wizard uses scheduleEstimate from vram-estimate.js to call
// /api/vram-estimate as the single source of truth. No local VRAM formulas.

// ── Workload profiles ────────────────────────────────────────────────────────
// UI removed (Phase 7B2: dedicated step-3 picker + confirmation gate was redundant
// with page-1 use-case selection). Backend integration (workload_scenario → VRAM
// estimation) remains active. wizardState.hardware.workloadScenario defaults to
// 'interactive_coding_agent' and is serialized as workload_scenario on spawn.
// Page-1 "what are you running this for?" cards map to workload_scenario strings
// consumed by the backend VRAM estimator.
const USE_CASE_TO_PROFILE = {
  agentic: 'interactive_coding_agent',
  general: 'general_chat',
};

// The one concrete thing a use-case should change on llama.cpp: how hard the KV cache is
// quantized. Tool-calling degrades badly below q8_0 -- q4_0 KV sends agentic runs into
// repeat loops -- so anything that has to call tools holds the q8_0 floor. Roleplay has no
// tool grammar to corrupt and would rather spend the saved VRAM on a longer memory, so it
// drops to q4_0.
//
// Rapid-MLX is deliberately absent: its KV quantization is a different mechanism with a
// different floor, and `--kv-cache-dtype` does not quantize the live cache on the batch
// path at all, so copying llama.cpp's numbers across would be guesswork dressed as a
// recommendation.
const USE_CASE_TO_KV_DTYPE = {
  agentic: 'q8_0',
  general: 'q4_0',
};


// ── State ─────────────────────────────────────────────────────────────────────

export const STEP_LABELS = ['Choose model', 'Hardware & memory', 'Start server'];
// URL slugs parallel to STEP_LABELS so wizard steps are deep-linkable and browser
// Back/Forward traverses them. Step 0 has no slug ('/spawn'); the rest are
// '/spawn/<slug>'. Keep in sync with STEP_LABELS.
const STEP_SLUGS = ['', 'hardware', 'start'];

// Set while openSpawnWizard runs its own initial showStep() so that internal step
// change doesn't push a history entry; showSpawnRoute reconciles the URL afterward.
let _suppressStepUrl = false;

function stepRoute(index) {
  const slug = STEP_SLUGS[index];
  return slug ? '/spawn/' + slug : '/spawn';
}

// Map a /spawn or /spawn/<slug> path to a step index, or 0 if unrecognized.
function stepFromRoute(path) {
  const m = /^\/spawn(?:\/([^/]+))?$/.exec(path || '');
  if (!m) return 0;
  const idx = STEP_SLUGS.indexOf(m[1] || '');
  return idx >= 0 ? idx : 0;
}

// Reflect the wizard's current step in the URL without adding history (used after
// an open, where openSpawnWizard may have jumped to a step, e.g. model for a
// dropped local file).
function syncStepUrlToCurrent() {
  if (!dom.overlay || !dom.overlay.classList.contains('open')) return;
  if (!(location.pathname === '/spawn' || location.pathname.startsWith('/spawn/'))) return;
  const target = stepRoute(wizardState.currentStep);
  if (location.pathname !== target) {
    try { history.replaceState({ path: target }, '', target); } catch {}
  }
}

// Router entry point: open the wizard if needed (consuming queued options) and
// show the step encoded in the path. Idempotent — re-dispatching a /spawn route
// for an already-open wizard just changes the step, never resets it.
export function showSpawnRoute(path) {
  const wasOpen = !!dom.overlay && dom.overlay.classList.contains('open');
  const explicitStep = path !== '/spawn';
  if (wasOpen) {
    showStep(stepFromRoute(path));
    return;
  }
  const opts = window.__spawnWizardOpts || {};
  window.__spawnWizardOpts = null;
  _suppressStepUrl = true;
  try {
    openSpawnWizard(opts);
  } finally {
    _suppressStepUrl = false;
  }
  // A bare /spawn honors whatever initial step openSpawnWizard chose (it may jump
  // to the model step for a queued local model); an explicit /spawn/<slug> wins.
  if (explicitStep) showStep(stepFromRoute(path));
  else syncStepUrlToCurrent();
}

// Exposed for testing/screenshot scripts; internal state is mutable.
export const wizardState = {
  currentStep: 0,
  engine: {
    selected: 'llama_cpp',
    explicit: false,
    recommendation: null,
    rapidMlxLocalAvailable: false,
    rapidMlxRuntimeCompatible: false,
  },
  profile: 'balanced',
  useCase: 'agentic',    // 'agentic' | 'general' | 'roleplay'
  mode: 'guided',
  viewMode: 'guided',    // 'guided' | 'pro'
  model: {
    source: 'local',     // 'local' | 'hf' | 'import'
    path: '',
    hfRepo: '',
    hfFile: '',
    hfTokenSet: false,
    rapidMlxSource: null,
    rapidMlxProfile: null,          // live profile from rapid-mlx info <model>
    rapidMlxUnifiedProfile: null,   // qualified profile with effective recommendations
    rapidMlxMllm: true,              // --mllm toggle (vision enabled by default when available)
    rapidMlxEmbeddingModel: null,    // --embedding-model repo ID/alias
    delivery: 'local_file', // 'local_file' | 'imported_local' | 'stream_hf' | 'downloaded_hf'
    originRepo: '',
    originFile: '',
    localMeta: null,
    paramB: 0,           // estimated parameter count (from HF metadata if available)
    modelBytes: 0,       // file size in bytes once known
    nCtxTrain: 0,        // training context length from GGUF metadata (0 = unknown)
    quantFiles: [],      // GGUF files from HF repo for hardware-step quant swap
    mmprojFiles: [],     // mmproj files found in HF repo
    mmprojHfRepo: '',    // HF repo owning the selected projector
    draftCandidates: [], // draft files detected near this model
    selectedDraftPath: '', // path of chosen draft for MTP
    chatTemplatePath: null,  // local path to installed .jinja template (null = use embedded)
    chatTemplateMode: 'auto', // 'auto' | 'custom' | 'embedded'
  },
  // Architecture from introspection (or heuristic)
  arch: {
    nLayers: 0, nKvHeads: 0, headDim: 0,
    nGlobalAttnLayers: 0, localAttnWindow: 0, localKvHeads: 1,
    globalHeadDim: 0, localHeadDim: 0, slidingWindow: 0,
    nAttnLayers: 0, linearAttnStateBytes: 0,
    nExperts: 0, nExpertsUsed: 0, expertFraction: 0.65,
    mtpDepth: 0, mmprojBytes: 0, mmprojRequired: false, paramB: 0,
    metadataStatus: 'unknown',
    metadataReason: '',
  },
  hardware: {
    gpuLayers: 'auto', gpuLayersManual: null,
    contextSize: 8192,
    batchSize: 2048, ubatchSize: 2048,
    parallelSlots: 1,
    cacheTypeK: 'q8_0', cacheTypeV: 'q8_0',
    // True once the user expresses a KV preference; stops the use-case cards seeding it.
    kvDtypeUserSet: false,
    nCpuMoe: 0,
    tensorSplit: '',
    fitEnabled: null,
    fitTarget: '',
    cacheRam: null,
    cacheMode: 'custom',
    kvUnified: null,
    flashAttn: 'on',
  mlock: false,
  loadMode: 'mmap',
  prio: null,
  verbosity: 4,
  ctxCheckpoints: 32,
  checkpointMinStep: 8192,
  cacheReuse: null,
  cacheIdleSlots: null,
  noContBatching: false,
  swaFull: false,
  mmprojOffload: null,
  llamaReasoningEffort: 'default',
  llamaReasoningFormat: null,
  llamaReasoningPreserve: null,
    threads: null,
    threadsBatch: null,
    // MTP
    mtpEnabled: true,
    mtpDraftNMax: null,   // null = let spawn compute default (2 — universal starting point)
    mtpDraftNMin: null,
    mtpDraftPMin: null,
    // Sampling (null = use llama-server default)
    temperature: null,
    topP: null,
    topK: null,
    minP: null,
    repeatPenalty: null,
    presencePenalty: null,
    maxTokens: null,
    seed: null,
    outputMode: '',
    enableThinking: null,
    preserveThinking: null,
    toolCallFormat: null,
    reasoningBudget: null,
    reasoningBudgetMessage: null,
    kvCacheDtype: 'int4',
    turboquantMode: 'none',
    // Phase 7 Rapid-MLX fields. '' means "omit the flag and take the runtime default",
    // which is not the same as an explicit value, so these stay strings and are only
    // written into the payload when non-empty.
    gpuMemoryUtilization: '',
    maxNumSeqs: '',
    maxConcurrentRequests: '',
    pflashPolicy: 'off',
    // Measured general agent-workflow default. Zero remains the explicit omit value,
    // which takes Rapid-MLX's runtime default (0, hybrid reuse disabled).
    hybridCacheEntries: 16,
    prefillBatchSize: '',
    completionBatchSize: '',
      retainedCacheMib: 8192,
    cacheMode: 'custom',
    workloadScenario: 'interactive_coding_agent',
    reasoningMode: null,         // llama.cpp thinking/reasoning select
    rapidReasoningMode: 'on',    // Rapid-MLX checkbox (defaults to on)
    toolCallParser: '',
    reasoningParser: '',
    hybridMode: 'auto',
  prefillStepSize: 512,
  prefillStepSizeUserSet: false,
    speculativeEnabled: false,
    speculativeSource: 'embedded',
    speculativeModel: '',
    speculativeModelAutoSelected: false,
    speculativeTokens: RAPID_MLX_DEFAULT_SPECULATIVE_TOKENS,
    speculativeDisableAutoK: false,
    speculativeTrustRequired: false,
    speculativeTrustConsent: false,
    speculativeTrustRepoId: '',
    speculativeTrustRevision: '',
    speculativeTrustEstimatedMemoryBytes: null,
    speculativeTrustSidecar: null,
    speculativeTrustDepth: null,
    autoToolChoice: false,
    // Phase 7: Web UI (D26/A44)
    // Phase 7: Sampling mode (D27)
    samplingMode: 'auto',
    grammar: '',
    jsonSchema: '',
    alias: '',
    extraArgs: '',
    // Rapid-MLX escape-hatch flags (keyed by flag name)
    escapeHatchFlags: {},
  },
  access: {
    port: 8001,
    bindHost: '127.0.0.1',
    apiKey: '',
  },
  vram: { available: 0 },
  spawn: { inFlight: false, error: '' },
  savedPresetId: null, // ID of preset saved from this wizard run (to avoid duplicates)
  calibration: {
    matches: [],
    selectedMatch: null,
    jobId: null,
    polling: false,
  },
  // Snapshot for the "Switch to repo selection" recovery flow (plan §6.2). Set by
  // snapshotPendingRestore(), applied by _applyPendingRestore() on arrival at step 2, discarded
  // after apply, on wizard close, or after PENDING_RESTORE_TIMEOUT_MS idle.
  _pendingRestore: null,
};

const PENDING_RESTORE_TIMEOUT_MS = 5 * 60 * 1000;

// Snapshot step-2/3 configuration before re-entering step 1 to pick a different model, so a
// one-click recovery action (fixing a chat-template-degraded alias source) doesn't cost the
// user their whole configuration.
export function snapshotPendingRestore() {
  wizardState._pendingRestore = {
    hardware: JSON.parse(JSON.stringify(wizardState.hardware)),
    savedAt: Date.now(),
  };
}

export function discardPendingRestore() {
  wizardState._pendingRestore = null;
}

// Re-applies a snapshot taken by snapshotPendingRestore(), re-validating context-length-derived
// fields against the newly-selected model rather than blindly replaying raw values (plan §6.2).
function _applyPendingRestore() {
  const pending = wizardState._pendingRestore;
  if (!pending) return;
  wizardState._pendingRestore = null;
  if (Date.now() - pending.savedAt > PENDING_RESTORE_TIMEOUT_MS) return;

  const restored = pending.hardware;
  // The new model may have a smaller native context window than the restored value — clamp
  // rather than carry over a value the new model can't honor. nCtxTrain is 0 when unknown
  // (no GGUF/MLX metadata resolved yet), in which case there is nothing to validate against.
  const nCtxTrain = wizardState.model.nCtxTrain || 0;
  if (nCtxTrain > 0 && restored.contextSize > nCtxTrain) {
    restored.contextSize = nCtxTrain;
  }
  Object.assign(wizardState.hardware, restored);
}

// ── DOM refs ──────────────────────────────────────────────────────────────────

export let dom = {};
let pendingHardwareScrollReset = false;
let pendingHardwareScrollRestore = null;
let _escapeHatchDescriptors = null; // cached from /api/rapid-mlx/escape-hatch-flags

// ── Public API ────────────────────────────────────────────────────────────────

export function initSpawnWizard() {
  cacheDom();
  window.refreshWizardCalibrationOffer = refreshWizardCalibrationOffer;
  applyReducedMotion();
  bindEvents();
  bindHfSearchControls();
  bindQuantizerEditor();
  bindWizardHfToken();
  applyGuidedVisibility();
   refreshEngineAvailability();

  initHfBrowseWidgets();

  bindHfDownloadPanel();
  bindSidebarTipsToggle();
  initSidebarColumnResizers();

  loadCommunityPicks();
}

// Drops a draggable divider on the left edge of each right-column sidebar so
// the user can resize the column from the seam it shares with the left
// column, rather than hunting for a resize grip elsewhere.
function initSidebarColumnResizers() {
  const sidebars = document.querySelectorAll('.wizard-sidebar, .hw-vram-sidebar');
  sidebars.forEach((sidebar) => {
    if (sidebar.querySelector(':scope > .wizard-col-resizer')) return;
    const handle = document.createElement('div');
    handle.className = 'wizard-col-resizer';
    handle.setAttribute('role', 'separator');
    handle.setAttribute('aria-orientation', 'vertical');
    handle.setAttribute('aria-label', 'Resize panel');
    sidebar.prepend(handle);

    let startX = 0;
    let startWidth = 0;

    const onPointerMove = (e) => {
      const delta = startX - e.clientX;
      const min = parseFloat(getComputedStyle(sidebar).minWidth) || 260;
      const max = parseFloat(getComputedStyle(sidebar).maxWidth) || window.innerWidth * 0.7;
      const next = Math.min(max, Math.max(min, startWidth + delta));
      sidebar.style.width = `${next}px`;
    };

    const onPointerUp = () => {
      handle.classList.remove('is-dragging');
      document.removeEventListener('pointermove', onPointerMove);
      document.removeEventListener('pointerup', onPointerUp);
    };

    handle.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      startX = e.clientX;
      startWidth = sidebar.getBoundingClientRect().width;
      handle.classList.add('is-dragging');
      document.addEventListener('pointermove', onPointerMove);
      document.addEventListener('pointerup', onPointerUp);
    });
  });
}

function bindSidebarTipsToggle() {
  if (!dom.sidebarTipsToggle || !dom.sidebarTips) return;
  let collapsed = false;
  try { collapsed = localStorage.getItem('spawn_wizard_tips_collapsed') === '1'; } catch {}
  applySidebarTipsCollapsed(collapsed);
  dom.sidebarTipsToggle.addEventListener('click', () => {
    applySidebarTipsCollapsed(!dom.sidebarTips.classList.contains('collapsed'));
  });
}

function applySidebarTipsCollapsed(collapsed) {
  dom.sidebarTips.classList.toggle('collapsed', collapsed);
  dom.sidebarTipsToggle.textContent = collapsed ? 'Show tips' : 'Hide tips';
  dom.sidebarTipsToggle.setAttribute('aria-expanded', String(!collapsed));
  try { localStorage.setItem('spawn_wizard_tips_collapsed', collapsed ? '1' : '0'); } catch {}
}

function applyReducedMotion() {
  if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) {
    document.documentElement.classList.add('reduce-motion');
  }
}

export function openSpawnWizard(opts = {}) {
  if (!dom.overlay) return;
  window.wizardState = wizardState;
  resetWizardState();
  document.getElementById('models-modal')?.classList.remove('open');
  window.closePresetsPanel?.();
  dom.overlay.classList.add('open');
  refreshHfTokenState();

  // Check binary prereq every time wizard opens
  _checkBinaryPrereq();

  if (opts.templatePreset) {
    // "Use as template": pre-populate hardware/params from the example preset so
    // the user only needs to pick a model. Settings are applied but not locked.
    const t = opts.templatePreset;
    if (t.id) wizardState.savedPresetId = t.id;
    if (t.context_size)      wizardState.hardware.contextSize    = t.context_size;
    if (t.ctk || t.ctv)      wizardState.hardware.kvDtypeUserSet = true;
    if (t.ctk)               wizardState.hardware.cacheTypeK     = t.ctk;
    if (t.ctv)               wizardState.hardware.cacheTypeV     = t.ctv;
    if (t.batch_size)        wizardState.hardware.batchSize      = t.batch_size;
    if (t.ubatch_size)       wizardState.hardware.ubatchSize     = t.ubatch_size;
    if (t.parallel_slots)    wizardState.hardware.parallelSlots  = t.parallel_slots;
    if (t.gpu_layers != null) wizardState.hardware.gpuLayers     = String(t.gpu_layers);
    if (t.threads != null)   wizardState.hardware.threads        = t.threads;
 if (t.threads_batch != null) wizardState.hardware.threadsBatch = t.threads_batch;
    if (t.verbosity != null) wizardState.hardware.verbosity = t.verbosity;
    if (t.load_mode) wizardState.hardware.loadMode = t.load_mode;
 if (t.ctx_checkpoints != null) wizardState.hardware.ctxCheckpoints = t.ctx_checkpoints;
 if (t.checkpoint_min_step != null) wizardState.hardware.checkpointMinStep = t.checkpoint_min_step;
 if (t.cache_reuse != null) wizardState.hardware.cacheReuse = t.cache_reuse;
 if (t.cache_idle_slots != null) wizardState.hardware.cacheIdleSlots = !!t.cache_idle_slots;
 if (t.no_cont_batching != null) wizardState.hardware.noContBatching = !!t.no_cont_batching;
 if (t.swa_full != null) wizardState.hardware.swaFull = !!t.swa_full;
 if (t.mmproj_offload != null) wizardState.hardware.mmprojOffload = !!t.mmproj_offload;
 if (t.llama_reasoning_effort != null) wizardState.hardware.llamaReasoningEffort = t.llama_reasoning_effort;
 if (t.llama_reasoning_format != null) wizardState.hardware.llamaReasoningFormat = t.llama_reasoning_format;
 if (t.llama_reasoning_preserve != null) wizardState.hardware.llamaReasoningPreserve = !!t.llama_reasoning_preserve;
    if (t.temperature != null)   wizardState.hardware.temperature   = t.temperature;
    if (t.top_p != null)         wizardState.hardware.topP          = t.top_p;
    if (t.top_k != null)         wizardState.hardware.topK          = t.top_k;
    if (t.min_p != null)         wizardState.hardware.minP          = t.min_p;
 if (t.repeat_penalty != null) wizardState.hardware.repeatPenalty = t.repeat_penalty;
 if (t.repeat_last_n != null) wizardState.hardware.repeatLastN = t.repeat_last_n;
    if (t.presence_penalty != null) wizardState.hardware.presencePenalty = t.presence_penalty;
    if (t.max_tokens != null)    wizardState.hardware.maxTokens     = t.max_tokens;
    if (t.seed != null)          wizardState.hardware.seed          = t.seed;
    if (t.backend === 'rapid_mlx' && t.rapid_mlx) {
      const rapid = t.rapid_mlx;
      const source = rapid.model_source || null;
      wizardState.engine.selected = 'rapid_mlx';
      wizardState.engine.explicit = true;
      wizardState.model.rapidMlxSource = source;
      wizardState.access.port = rapid.port || t.port || 8001;
      wizardState.access.bindHost = rapid.host || t.bind_host || '127.0.0.1';
      wizardState.hardware.alias = rapid.served_model_name || '';
      wizardState.model.rapidMlxMllm = rapid.mllm_vision !== 'off';
      if (Array.isArray(rapid.escape_hatch_flags)) {
        wizardState.hardware.escapeHatchFlags = Object.fromEntries(rapid.escape_hatch_flags);
      }
      const authoritativeHf = source?.kind === 'authoritative_safetensors'
        && source.source?.kind === 'hugging_face_repo';
      if (source?.kind === 'hugging_face_repo' || authoritativeHf) {
        wizardState.model.source = 'hf';
        wizardState.model.hfRepo = source.repo_id || source.source?.repo_id || '';
        wizardState.model.hfFile = '';
      } else {
        wizardState.model.source = 'local';
        wizardState.model.path = source?.path || source?.source?.path || source?.value || rapid.model_path || '';
        wizardState.model.localMeta = source ? { source_kind: source.kind, model_source: source } : null;
      }
    }
  }

  if (opts.localPath) {
    // Pre-load a local model path and jump straight to the Model step.
    wizardState.model.source = 'local';
    wizardState.model.path = opts.localPath;
    wizardState.model.hfRepo = '';
    wizardState.model.hfFile = '';
    wizardState.model.delivery = 'local_file';
    wizardState.model.localMeta = opts.localModel || null;
    wizardState.model.modelBytes = Number(opts.localModel?.size_bytes) || 0;
    if (dom.modelPathInput) dom.modelPathInput.value = opts.localPath;
    if (dom.hfRepoInput) dom.hfRepoInput.value = '';
    // Select the "local" source card visually.
    dom.modelSourceCards?.forEach(c => {
      c.classList.toggle('selected', c.dataset.source === 'local');
    });
    updateModelInputVisibility();
    renderLocalModelHint();
    showStep(0); // step 0 = Model (0-indexed)
  } else if (opts.templatePreset?.backend !== 'rapid_mlx') {
    wizardState.model.source = 'local';
    updateModelInputVisibility();
    renderLocalModelHint();
    showStep(0);
  } else {
    if (dom.modelPathInput) dom.modelPathInput.value = wizardState.model.path;
    if (dom.hfRepoInput) dom.hfRepoInput.value = wizardState.model.hfRepo;
    dom.modelSourceCards?.forEach(c => {
      c.classList.toggle('selected', c.dataset.source === wizardState.model.source);
    });
    updateModelInputVisibility();
    renderLocalModelHint();
    showStep(0);
  }

  renderEngineSelection();
  refreshEngineRecommendation();

  _initViewMode();
  setupWizardEscape();
}

function setupWizardEscape() {
  window.addEventListener('keydown', function wizardEsc(e) {
    if (e.key === 'Escape') {
      window.removeEventListener('keydown', wizardEsc);
      // Full close (resets state + syncs URL back to the underlying view) so a
      // reload doesn't re-open the wizard, matching the X/close-button behavior.
      closeSpawnWizard();
    }
  });
}

// Clear all wizard state for a fresh start
function resetWizardState() {
  // Clear DOM inputs so they don't show stale values when the wizard re-opens
  if (dom.modelPathInput) dom.modelPathInput.value = '';
  if (dom.presetNameInput) dom.presetNameInput.value = '';
  if (dom.savedPresetName) {
    dom.savedPresetName.style.display = 'none';
    dom.savedPresetName.textContent = '';
  }

  // Reset model state
  wizardState.model.source = '';
  wizardState.model.path = '';
  wizardState.model.hfRepo = '';
  wizardState.model.hfFile = '';
  wizardState.model.mmprojPath = '';
  wizardState.model.mmprojHfFile = '';
  wizardState.model.mmprojHfRepo = '';
  wizardState.model.originRepo = '';
  wizardState.model.originFile = '';
  wizardState.model.delivery = '';
  wizardState.model.cardUrl = '';
  wizardState.model.family = '';
  wizardState.model.paramB = 0;
  wizardState.model.modelBytes = 0;
  wizardState.model.nCtxTrain = 0;
  wizardState.model.chatTemplatePath = '';
  wizardState.model.localMeta = null;
  wizardState.model.mmprojFiles = [];
  wizardState.model.quantFiles = [];
  wizardState.model.hfTokenSet = false;
  wizardState.model.rapidMlxSource = null;
  wizardState.model.rapidMlxProfile = null;
  wizardState.model.rapidMlxMllm = true;
  wizardState.model.rapidMlxEmbeddingModel = null;
  wizardState.model._quantSwapRepo = '';

  // Clear module-level quant-swap search state so next open starts fresh
  resetQuantSwapSearchState();

  // Reset hardware state
  wizardState.hardware.gpuLayers = '';
  wizardState.hardware.contextSize = 0;
  wizardState.hardware.batchSize = 0;
  wizardState.hardware.ubatchSize = 0;
  wizardState.hardware.parallelSlots = 1;
  wizardState.hardware.cacheTypeK = '';
  wizardState.hardware.cacheTypeV = '';
  wizardState.hardware.kvDtypeUserSet = false;
  wizardState.hardware.flashAttn = '';
  wizardState.hardware.kvUnified = null;
  wizardState.hardware.mlock = false;
  wizardState.hardware.loadMode = isUnifiedMemory() ? 'mmap' : 'none';
  wizardState.hardware.prio = null;
  wizardState.hardware.verbosity = 4;
  wizardState.hardware.ctxCheckpoints = 32;
  wizardState.hardware.checkpointMinStep = 8192;
  wizardState.hardware.cacheReuse = null;
  wizardState.hardware.cacheIdleSlots = null;
  wizardState.hardware.noContBatching = false;
  wizardState.hardware.swaFull = false;
  wizardState.hardware.mmprojOffload = null;
  wizardState.hardware.llamaReasoningEffort = 'default';
  wizardState.hardware.llamaReasoningFormat = null;
  wizardState.hardware.llamaReasoningPreserve = null;
  wizardState.hardware.nCpuMoe = 0;
  wizardState.hardware.tensorSplit = '';
  wizardState.hardware.fitEnabled = null;
  wizardState.hardware.fitTarget = '';
  wizardState.hardware.cacheRam = null;
  wizardState.hardware.cacheMode = 'custom';
  wizardState.hardware.temperature = null;
  wizardState.hardware.topP = null;
  wizardState.hardware.topK = null;
  wizardState.hardware.minP = null;
 wizardState.hardware.repeatPenalty = null;
 wizardState.hardware.repeatLastN = null;
  wizardState.hardware.presencePenalty = null;
  wizardState.hardware.maxTokens = null;
  wizardState.hardware.seed = null;
  wizardState.hardware.mtpEnabled = false;
  wizardState.hardware.mtpDraftNMax = null;
  wizardState.hardware.enableThinking = null;
  wizardState.hardware.preserveThinking = null;
  wizardState.hardware.toolCallFormat = null;
  wizardState.hardware.reasoningMode = null;
  wizardState.hardware.reasoningBudget = null;
  wizardState.hardware.reasoningBudgetMessage = null;
  wizardState.hardware.kvCacheDtype = '';
  wizardState.hardware.turboquantMode = 'none';
  wizardState.hardware.gpuMemoryUtilization = '';
  wizardState.hardware.maxNumSeqs = '';
  wizardState.hardware.maxConcurrentRequests = '';
  wizardState.hardware.pflashPolicy = 'off';
  wizardState.hardware.hybridCacheEntries = 16;
  wizardState.hardware.cacheMode = 'custom';
  wizardState.hardware.prefillBatchSize = '';
  wizardState.hardware.completionBatchSize = '';
    wizardState.hardware.prefillStepSize = 512;
    wizardState.hardware.prefillStepSizeUserSet = false;
  wizardState.hardware.rapidReasoningMode = 'on';
  wizardState.hardware.speculativeEnabled = false;
  wizardState.hardware.speculativeSource = 'embedded';
  wizardState.hardware.speculativeModel = '';
  wizardState.hardware.speculativeModelAutoSelected = false;
  wizardState.hardware.speculativeTokens = RAPID_MLX_DEFAULT_SPECULATIVE_TOKENS;
  wizardState.hardware.speculativeDisableAutoK = false;
  wizardState.hardware.autoToolChoice = false;
  wizardState.hardware.workloadScenario = 'interactive_coding_agent';
  wizardState.hardware.samplingMode = 'auto';
  if (dom.kvUnifiedSelect) dom.kvUnifiedSelect.value = '';
  if (dom.fitEnableSelect) dom.fitEnableSelect.value = '';
  if (dom.fitTargetWrap) dom.fitTargetWrap.style.display = 'none';
  if (dom.fitTargetInput) dom.fitTargetInput.value = '';
  wizardState.hardware.specType = '';
  wizardState.hardware.draftModelPath = '';
  wizardState.hardware.grammar = '';
  wizardState.hardware.jsonSchema = '';
  wizardState.hardware.alias = '';
  wizardState.hardware.extraArgs = '';
  wizardState.hardware.escapeHatchFlags = {};

  // Reset architecture
  wizardState.arch.nLayers = 0;
  wizardState.arch.nKvHeads = 0;
  wizardState.arch.headDim = 0;
  wizardState.arch.globalHeadDim = 0;
  wizardState.arch.nGlobalAttnLayers = 0;
  wizardState.arch.localAttnWindow = 0;
  wizardState.arch.localKvHeads = 1;
  wizardState.arch.nExperts = 0;
  wizardState.arch.nExpertsUsed = 0;
  wizardState.arch.expertFraction = 0.65;
  wizardState.arch.mtpDepth = 0;
  wizardState.arch.mmprojBytes = 0;
  wizardState.arch.linearAttnStateBytes = 0;
  wizardState.arch.isHybridAttn = false;
  wizardState.arch.ggufArch = '';

  // Reset UI state
  wizardState.currentStep = 0;
  wizardState.engine.selected = 'llama_cpp';
  wizardState.engine.explicit = false;
  wizardState.engine.recommendation = null;
  wizardState.useCase = 'agentic';
  wizardState.profile = 'balanced';
  wizardState.mode = 'guided';
  wizardState.model.chatTemplateMode = 'auto';
  wizardState.access.port = 8001;
  wizardState.access.bindHost = '127.0.0.1';
  wizardState.access.apiKey = '';
  wizardState.savedPresetId = null;
  wizardState.calibration.matches = [];
  wizardState.calibration.selectedMatch = null;
  wizardState.calibration.jobId = null;
  wizardState.calibration.polling = false;
}

function calibrationWorkload() {
  const kind = wizardState.useCase === 'agentic' ? 'agents' : 'interactive';
  return {
    kind,
    prompt_tokens: 512,
    generation_tokens: 256,
    parallel_requests: 1,
    minimum_context: Math.max(4096, Number(wizardState.hardware.contextSize) || 0),
    objective: 'balanced',
    fixture_id: 'calibration-v1-wizard',
  };
}

function calibrationHeaders(json = false) {
  const headers = window.authHeaders ? { ...window.authHeaders() } : {};
  if (json) headers['Content-Type'] = 'application/json';
  return headers;
}

function currentCalibrationCandidate(match) {
  const receipt = match?.receipt;
  const id = receipt?.selected_candidate;
  return receipt?.candidate_results?.find(result => result.candidate?.id === id)
    || receipt?.candidate_results?.[0]
    || null;
}

function renderCalibrationMatches(matches) {
  const container = dom.calibrationMatches;
  if (!container) return;
  container.replaceChildren();
  wizardState.calibration.matches = matches;
  const preferred = matches.find(match => match.match_kind === 'exact')
    || matches.find(match => match.match_kind === 'compatible')
    || matches.find(match => match.match_kind === 'related')
    || null;
  wizardState.calibration.selectedMatch = preferred;
  matches.forEach((match, index) => {
    const receipt = match.receipt || {};
    const article = document.createElement('article');
    article.className = 'spawn-calibration-match';
    article.dataset.matchIndex = String(index);
    article.classList.toggle('is-selected', match === preferred);
    const heading = document.createElement('div');
    heading.className = 'spawn-calibration-match-heading';
    const badge = document.createElement('span');
    badge.className = `spawn-calibration-badge spawn-calibration-badge--${match.match_kind || 'related'}`;
    badge.textContent = match.match_kind === 'exact'
      ? 'Measured on this model'
      : match.match_kind === 'compatible'
        ? 'Compatible model evidence'
        : 'Related model evidence';
    const source = document.createElement('span');
    source.className = 'spawn-calibration-match-source';
    source.textContent = receipt.fingerprint?.model?.library_relative_id || 'local GGUF';
    heading.append(badge, source);
    article.appendChild(heading);
    const details = document.createElement('div');
    details.className = 'spawn-calibration-match-details';
    details.textContent = `Candidate: ${receipt.selected_candidate || 'measured result'} · Job ${receipt.job_id || 'unknown'}`;
    article.appendChild(details);
    (match.warnings || []).forEach(warning => {
      const note = document.createElement('p');
      note.className = 'spawn-calibration-match-warning';
      note.textContent = warning;
      article.appendChild(note);
    });
    article.addEventListener('click', () => {
      wizardState.calibration.selectedMatch = match;
      container.querySelectorAll('.spawn-calibration-match').forEach(item => item.classList.remove('is-selected'));
      article.classList.add('is-selected');
      if (dom.calibrationApplyBtn) {
        dom.calibrationApplyBtn.hidden = !currentCalibrationCandidate(match);
        dom.calibrationApplyBtn.textContent = match.match_kind === 'related'
          ? 'Review related candidate'
          : 'Apply selected candidate';
      }
    });
    container.appendChild(article);
  });
  if (dom.calibrationApplyBtn) {
    dom.calibrationApplyBtn.hidden = !preferred || !currentCalibrationCandidate(preferred);
    dom.calibrationApplyBtn.textContent = preferred?.match_kind === 'related'
      ? 'Review related candidate'
      : 'Apply selected candidate';
  }
}

export async function refreshWizardCalibrationOffer() {
  const card = dom.calibrationCard;
  if (!card) return;
  const eligible = wizardState.engine.selected === 'llama_cpp'
    && wizardState.model.source !== 'hf'
    && String(wizardState.model.path || '').toLowerCase().endsWith('.gguf');
  card.hidden = !eligible || wizardState.viewMode !== 'pro';
  if (!eligible || wizardState.viewMode !== 'pro') return;
  if (!wizardState.savedPresetId) {
    if (dom.calibrationStatus) dom.calibrationStatus.textContent = 'Save this preset first to compare or calibrate it.';
    if (dom.calibrationCheckBtn) dom.calibrationCheckBtn.disabled = true;
    if (dom.calibrationStartBtn) dom.calibrationStartBtn.hidden = true;
    return;
  }
  if (dom.calibrationCheckBtn) dom.calibrationCheckBtn.disabled = false;
}

async function checkWizardCalibrationEvidence() {
  if (!wizardState.savedPresetId) return;
  if (dom.calibrationStatus) dom.calibrationStatus.textContent = 'Checking exact and compatible receipts…';
  if (dom.calibrationStartBtn) dom.calibrationStartBtn.hidden = true;
  try {
    const response = await fetch('/api/calibrations/match', {
      method: 'POST',
      headers: calibrationHeaders(true),
      body: JSON.stringify({ preset_id: wizardState.savedPresetId, workload: calibrationWorkload(), budget: 'balanced' }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.ok === false) throw new Error(data.error || 'Calibration evidence lookup failed');
    const matches = Array.isArray(data.matches) ? data.matches : [];
    renderCalibrationMatches(matches);
    if (dom.calibrationStatus) {
      dom.calibrationStatus.textContent = matches.length
        ? `${matches.length} measured evidence result${matches.length === 1 ? '' : 's'} found.`
        : 'No matching evidence found. You can opt in to a bounded calibration.';
    }
    const hasReusableEvidence = matches.some(match => match.match_kind === 'exact' || match.match_kind === 'compatible');
    if (dom.calibrationStartBtn) dom.calibrationStartBtn.hidden = hasReusableEvidence;
    if (matches.length && !hasReusableEvidence && dom.calibrationStatus) {
      dom.calibrationStatus.textContent = 'Related evidence found for review only. You can calibrate this exact model or explicitly review the related candidate.';
    }
  } catch (error) {
    if (dom.calibrationStatus) dom.calibrationStatus.textContent = error.message || 'Calibration evidence lookup failed.';
  }
}

async function startWizardCalibration() {
  if (!wizardState.savedPresetId) return;
  const workload = calibrationWorkload();
  if (dom.calibrationStartBtn) dom.calibrationStartBtn.disabled = true;
  if (dom.calibrationStatus) dom.calibrationStatus.textContent = 'Preparing bounded calibration…';
  try {
    const preflightResponse = await fetch('/api/calibrations/preflight', {
      method: 'POST',
      headers: calibrationHeaders(true),
      body: JSON.stringify({ preset_id: wizardState.savedPresetId, workload, budget: 'balanced' }),
    });
    const preflightData = await preflightResponse.json().catch(() => ({}));
    if (!preflightResponse.ok || preflightData.ok === false) throw new Error(preflightData.error || 'Calibration preflight failed');
    const startResponse = await fetch('/api/calibrations', {
      method: 'POST',
      headers: calibrationHeaders(true),
      body: JSON.stringify({
        preset_id: wizardState.savedPresetId,
        expected_preset_fingerprint: preflightData.preflight.preset_fingerprint,
        workload,
        budget: 'balanced',
    kv_quality_floor: 'q8_0',
    allow_stop_active_server: false,
    exact_confirmation: 'CALIBRATE',
    server_qualification: {
      tracks: ['latency_memory', 'tool_correctness'],
      parallel_requests: 1,
      allow_concurrency: false,
      prompt: 'Reply with one short sentence describing a calibration check.',
      generation_tokens: 256,
      timeout_ms: 30000,
      capability_evidence: [],
    },
      }),
    });
    const startData = await startResponse.json().catch(() => ({}));
    if (!startResponse.ok || startData.ok === false) throw new Error(startData.error || 'Calibration could not be queued');
    const jobId = startData.job?.id;
    wizardState.calibration.jobId = jobId || null;
    if (dom.calibrationStatus) dom.calibrationStatus.textContent = `Calibration queued${jobId ? ` (${jobId})` : ''}. You can continue setup.`;
    const notificationId = jobId ? `calibration-${jobId}` : 'calibration-wizard-active';
    showToastWithActions(
      'Calibration queued',
      'info',
      'The bounded llama.cpp calibration will continue while you finish setup.',
      [{ id: 'open-wizard', label: 'Open wizard', primary: true, handler: () => Router.navigate('/spawn/start') }],
      { notificationId },
    );
    if (jobId) pollWizardCalibration(jobId, notificationId);
  } catch (error) {
    if (dom.calibrationStatus) dom.calibrationStatus.textContent = error.message || 'Calibration could not be queued.';
    showToast('Calibration unavailable', 'warning', error.message || 'Calibration could not be queued.');
  } finally {
    if (dom.calibrationStartBtn) dom.calibrationStartBtn.disabled = false;
  }
}

async function pollWizardCalibration(jobId, notificationId) {
  try {
    const response = await fetch(`/api/calibrations/${encodeURIComponent(jobId)}`, { headers: calibrationHeaders() });
    const data = await response.json().catch(() => ({}));
    const job = data.job;
    if (!response.ok || !job) throw new Error(data.error || 'Calibration job disappeared');
    const terminal = ['complete', 'failed', 'cancelled'].includes(job.state);
    if (!terminal) {
      setTimeout(() => pollWizardCalibration(jobId, notificationId), 5000);
      return;
    }
    resolveNotification(notificationId, `Calibration ${job.state}.`);
    showToast('Calibration finished', job.state === 'complete' ? 'success' : 'warning', `Job ${jobId} is ${job.state}.`);
    if (dom.calibrationCard && !dom.calibrationCard.hidden) checkWizardCalibrationEvidence();
  } catch (error) {
    resolveNotification(notificationId, error.message || 'Calibration status unavailable.');
  }
}

export function applySelectedWizardCalibration() {
  const match = wizardState.calibration.selectedMatch;
  const result = currentCalibrationCandidate(match);
  if (!match || !result) return;
  if (match.match_kind === 'related' && !window.confirm('This evidence uses a different GGUF weight quantization. Review and apply anyway?')) return;
  applyCalibrationPatch(result.candidate?.typed_patch || {});
  if (dom.calibrationStatus) dom.calibrationStatus.textContent = match.match_kind === 'exact'
    ? 'Measured candidate applied to the wizard controls.'
    : match.match_kind === 'related'
      ? 'Related candidate applied after confirmation; review the quantization warning before saving or launching.'
      : 'Compatible candidate applied; review the warning before saving or launching.';
}

export function closeSpawnWizard() {
  if (!dom.overlay) return;
  // If dismissed directly while the URL still points at a /spawn route, return the
  // URL to the underlying view so a reload won't re-open the wizard. When closed as
  // part of a route change the path is already updated, so this is skipped.
  if (location.pathname === '/spawn' || location.pathname.startsWith('/spawn/')) {
    const base = routeForCurrentView();
    try { history.replaceState({ path: base }, '', base); } catch {}
  }
  dom.overlay.classList.remove('open');
  discardPendingRestore();
  resetSpawnStatus();
  resetWizardState();
}

// ── DOM caching ───────────────────────────────────────────────────────────────

function cacheDom() {
  dom.overlay  = document.getElementById('spawn-wizard-overlay');
  dom.closeBtn = document.getElementById('spawn-wizard-close');
  dom.stepLabel  = document.getElementById('wizard-step-label');
  dom.stepBadges = dom.overlay?.querySelectorAll('.step-badge[data-step]');
  dom.steps      = dom.overlay?.querySelectorAll('.wizard-step[id^="wizard-step-"]');
 dom.backBtn    = document.getElementById('wizard-back-btn');
   dom.nextBtn    = document.getElementById('wizard-next-btn');
   dom.closeWizardBtn = document.getElementById('wizard-close-btn');
  dom.footerHint = document.getElementById('wizard-footer-hint');

  // Step 1
  dom.usecaseCards  = dom.overlay?.querySelectorAll('.usecase-card[data-usecase]');

  // Step 2
  dom.engineCards = dom.overlay?.querySelectorAll('.wizard-engine-card[data-engine]');
  dom.engineReason = document.getElementById('wizard-engine-reason');
  dom.rapidHardwarePanel = document.getElementById('rapid-hardware-panel');
  dom.modelSourceCards = dom.overlay?.querySelectorAll('.model-source-card[data-source]');
  dom.modelInputLocal  = document.getElementById('model-input-local');
  dom.modelInputHf     = document.getElementById('model-input-hf');
  dom.modelInputImport = document.getElementById('model-input-import');
  dom.modelPathInput   = document.getElementById('spawn-model-path');
  dom.localModelHint   = document.getElementById('spawn-local-model-hint');
  dom.localModelHintTitle = document.getElementById('spawn-local-model-hint-title');
  dom.localModelHintMeta  = document.getElementById('spawn-local-model-hint-meta');
  dom.hfRepoInput       = document.getElementById('spawn-hf-repo');
  dom.hfMinSize         = document.getElementById('spawn-hf-min-size');
  dom.hfQuickpicks      = document.getElementById('hf-quickpicks');
  dom.hfSearchResults   = document.getElementById('hf-search-results');
  dom.importPathInput   = document.getElementById('spawn-import-path');
  dom.browseModelBtn   = document.getElementById('spawn-browse-model-btn');
  dom.importBrowseBtn  = document.getElementById('spawn-import-browse-btn');
  dom.selectedModel         = document.getElementById('spawn-selected-model');
  dom.selectedModelName     = document.getElementById('spawn-selected-model-name');
  dom.selectedModelMeta     = document.getElementById('spawn-selected-model-meta');
  dom.selectedModelArch     = document.getElementById('spawn-selected-model-arch');
  dom.hfFileList       = document.getElementById('spawn-hf-file-list');
  dom.quantAdvisor     = document.getElementById('quant-advisor');
  dom.quantAdvisorTable  = document.getElementById('quant-advisor-table');
  dom.quantAdvisorSubtitle = document.getElementById('quant-advisor-subtitle');
  dom.sidebarVram      = document.getElementById('wizard-sidebar-vram');
  dom.sidebarVramLabel = document.getElementById('wizard-sidebar-vram-label');
  dom.sidebarVramValue = document.getElementById('wizard-sidebar-vram-value');
  dom.sidebarVramHint  = document.getElementById('wizard-sidebar-vram-hint');
  dom.sidebarQaHint    = document.getElementById('wizard-sidebar-qa-hint');
  dom.sidebarCtxPills  = document.getElementById('wizard-sidebar-ctx-pills');
  dom.sidebarVramBar   = document.getElementById('wizard-sidebar-vram-bar');
  dom.sidebarVramLegend = document.getElementById('wizard-sidebar-vram-legend');
  dom.sidebarVsegWeights = document.getElementById('wizard-sidebar-vseg-weights');
  dom.sidebarVsegKv    = document.getElementById('wizard-sidebar-vseg-kv');
  dom.sidebarVsegOverhead = document.getElementById('wizard-sidebar-vseg-overhead');
  dom.sidebarVsegFree  = document.getElementById('wizard-sidebar-vseg-free');
  dom.sidebarVlegWeights = document.getElementById('wizard-sidebar-vleg-weights');
  dom.sidebarVlegKv    = document.getElementById('wizard-sidebar-vleg-kv');
  dom.sidebarVlegOverhead = document.getElementById('wizard-sidebar-vleg-overhead');
  dom.sidebarTipsToggle = document.getElementById('wizard-sidebar-tips-toggle');
  dom.sidebarTips      = document.getElementById('wizard-sidebar-tips');

  // Step 3
  dom.vramPanel       = document.getElementById('vram-panel');
  dom.vramPanelTotal  = document.getElementById('vram-panel-total');
  dom.vramBar         = document.getElementById('vram-bar');
  dom.vSegWeights  = document.getElementById('vseg-weights');
  dom.vSegKv       = document.getElementById('vseg-kv');
  dom.vSegMmproj   = document.getElementById('vseg-mmproj');
  dom.vSegMtp      = document.getElementById('vseg-mtp');
  dom.vSegOverhead = document.getElementById('vseg-overhead');
  dom.vSegFree     = document.getElementById('vseg-free');
  dom.vLegWeightsLabel  = document.getElementById('vleg-weights-label');
  dom.vLegKvLabel       = document.getElementById('vleg-kv-label');
  dom.vLegMmprojItem    = document.getElementById('vleg-mmproj');
  dom.vLegMmprojLabel   = document.getElementById('vleg-mmproj-label');
  dom.vLegMtpItem       = document.getElementById('vleg-mtp');
  dom.vLegMtpLabel      = document.getElementById('vleg-mtp-label');
  dom.vLegOverheadLabel = document.getElementById('vleg-overhead-label');
  dom.vLegFreeLabel     = document.getElementById('vleg-free-label');
  dom.vLegFreeDot       = document.getElementById('vleg-free-dot');
  dom.vLegPrefixCacheItem = document.getElementById('vleg-prefix-cache');
  dom.vLegPrefixCacheLabel = document.getElementById('vleg-prefix-cache-label');
  dom.vramScenarios   = document.getElementById('vram-scenarios');
  dom.ctxRailSummaryValue  = document.getElementById('ctx-rail-summary-value');
  dom.ctxRailSummaryStatus = document.getElementById('ctx-rail-summary-status');
  dom.ctxRailSummaryNote   = document.getElementById('ctx-rail-summary-note');
  dom.moeOffloadPanel   = document.getElementById('moe-offload-panel');
  dom.moeOffloadSlider  = document.getElementById('moe-offload-slider');
  dom.moeOffloadSubtitle= document.getElementById('moe-offload-subtitle');
  dom.moeOffloadHint    = document.getElementById('moe-offload-hint');
  dom.vramAutosizeBtn  = document.getElementById('vram-autosize-btn');
  dom.vramAutosizeNote = document.getElementById('vram-autosize-note');
  dom.vramPanelLabel   = document.getElementById('vram-panel-label');
  dom.metalLimitRow    = document.getElementById('metal-limit-row');
  dom.metalLimitText   = document.getElementById('metal-limit-text');
  dom.metalLimitBtn    = document.getElementById('metal-limit-btn');
  dom.ramPanel         = document.getElementById('ram-panel');
  dom.ramPanelTotal    = document.getElementById('ram-panel-total');
  dom.rSegUsed  = document.getElementById('rseg-used');
  dom.rSegMoe   = document.getElementById('rseg-moe');
  dom.rSegCram  = document.getElementById('rseg-cram');
  dom.rSegFree  = document.getElementById('rseg-free');
  dom.rLegUsed  = document.getElementById('rleg-used-label');
  dom.rLegMoeItem = document.getElementById('rleg-moe-item');
  dom.rLegMoe   = document.getElementById('rleg-moe-label');
  dom.rLegCram  = document.getElementById('rleg-cram-label');
  dom.rLegFree  = document.getElementById('rleg-free-label');


  dom.gpuLayersSelect      = document.getElementById('spawn-gpu-layers');
  dom.gpuLayersManualWrap  = document.getElementById('spawn-gpu-layers-manual-wrap');
  dom.gpuLayersManualInput = document.getElementById('spawn-gpu-layers-manual');
  dom.contextSizeInput   = document.getElementById('spawn-context-size');
  dom.batchSizeInput     = document.getElementById('spawn-batch-size');
  dom.advancedFields     = document.getElementById('spawn-advanced-fields');
  dom.ubatchSizeInput    = document.getElementById('spawn-ubatch-size');
  dom.parallelSlotsInput = document.getElementById('spawn-parallel-slots');
  dom.cacheTypeKSelect   = document.getElementById('spawn-cache-type-k');
  dom.cacheTypeVSelect   = document.getElementById('spawn-cache-type-v');
  dom.nCpuMoeInput       = document.getElementById('spawn-n-cpu-moe');
  dom.tensorSplitInput   = document.getElementById('spawn-tensor-split');
  dom.fitTargetInput     = document.getElementById('spawn-fit-target');

  // Legacy VRAM pill (kept for backward compat if HTML still has it)
  dom.vramEstimateText = document.getElementById('spawn-vram-estimate-text');
  dom.vramPill         = document.getElementById('spawn-vram-pill');
  dom.specTypeSelect     = document.getElementById('spawn-spec-type');
  dom.mtpDraftSection     = document.getElementById('hw-mtp-draft-section');
  dom.mtpDraftSelect      = document.getElementById('hw-mtp-draft-select');
  dom.mtpAssistantSection = dom.mtpDraftSection;
  dom.mtpAssistantSelect  = dom.mtpDraftSelect;
   dom.mtpDownloadSection  = document.getElementById('hw-mtp-download-section');
   dom.mtpDownloadInfo     = document.getElementById('hw-mtp-download-info');
   dom.mtpDownloadBtn      = document.getElementById('hw-mtp-download-btn');
   dom.draftModelWrap     = document.getElementById('spawn-draft-model-wrap');
  dom.draftModelInput    = document.getElementById('spawn-draft-model');
  dom.kvUnifiedSelect = document.getElementById('spawn-kv-unified');
  dom.flashAttnSelect    = document.getElementById('spawn-flash-attn');
  dom.mlockCheck         = document.getElementById('spawn-mlock');
  dom.mlockLabel         = dom.mlockCheck?.closest('label');
  dom.loadModeSelect     = document.getElementById('spawn-load-mode');
  dom.prioSelect         = document.getElementById('spawn-prio');
  dom.verbosityInput     = document.getElementById('spawn-verbosity');
  dom.ctxCheckpointsInput = document.getElementById('spawn-ctx-checkpoints');
  dom.checkpointMinStepInput = document.getElementById('spawn-checkpoint-min-step');
  dom.cacheReuseInput = document.getElementById('spawn-cache-reuse');
  dom.cacheIdleSlotsSelect = document.getElementById('spawn-cache-idle-slots');
  dom.noContBatchingCheck = document.getElementById('spawn-no-cont-batching');
  dom.swaFullCheck = document.getElementById('spawn-swa-full');
  dom.mmprojOffloadSelect = document.getElementById('spawn-mmproj-offload');
  dom.llamaReasoningEffortSelect = document.getElementById('spawn-reasoning-effort');
  dom.llamaReasoningFormatSelect = document.getElementById('spawn-reasoning-format');
  dom.llamaReasoningPreserveSelect = document.getElementById('spawn-reasoning-preserve');
  dom.threadsInput       = document.getElementById('spawn-threads');
  dom.threadsBatchInput  = document.getElementById('spawn-threads-batch');
  dom.specDraftNMinInput = document.getElementById('spawn-spec-draft-n-min');
  dom.specDraftPMinInput = document.getElementById('spawn-spec-draft-p-min');
  dom.specDraftTypeKSelect = document.getElementById('spawn-spec-draft-type-k');
  dom.specDraftTypeVSelect = document.getElementById('spawn-spec-draft-type-v');
  dom.draftKvRow           = document.getElementById('spawn-draft-kv-row');
  dom.fitEnableSelect = document.getElementById('spawn-fit-enable');
  dom.fitTargetWrap   = document.getElementById('spawn-fit-target-wrap');
  dom.cacheRamInput   = document.getElementById('spawn-cache-ram');
  dom.cacheModeSelect = document.getElementById('spawn-cache-mode');

   // Rapid-MLX advanced controls
   dom.rapidAdvancedFields  = document.getElementById('spawn-rapid-advanced-fields');
   dom.kvCacheDtypeSelect   = document.getElementById('spawn-kv-cache-dtype');
   dom.turboquantModeSelect = document.getElementById('spawn-turboquant-mode');
   dom.gpuMemoryUtilizationSelect = document.getElementById('spawn-rapid-gpu-memory-utilization');
   dom.maxNumSeqsSelect = document.getElementById('spawn-rapid-max-num-seqs');
   dom.maxConcurrentRequestsSelect = document.getElementById('spawn-rapid-max-concurrent-requests');
   dom.pflashPolicySelect = document.getElementById('spawn-rapid-pflash-policy');
   dom.hybridCacheEntriesSelect = document.getElementById('spawn-rapid-hybrid-cache-entries');
   dom.prefillBatchSizeSelect = document.getElementById('spawn-rapid-prefill-batch-size');
   dom.completionBatchSizeSelect = document.getElementById('spawn-rapid-completion-batch-size');
   dom.retainedCacheMibSelect = document.getElementById('spawn-retained-cache-mib');
   dom.rapidCacheModeSelect = document.getElementById('spawn-rapid-cache-mode');
   dom.workloadScenarioSelect = document.getElementById('spawn-workload-scenario'); // hidden select for compat
   dom.reasoningModeCheck   = document.getElementById('spawn-rapid-reasoning-mode');
   dom.toolCallParserSelect = document.getElementById('spawn-rapid-tool-call-parser');
   dom.reasoningParserSelect = document.getElementById('spawn-rapid-reasoning-parser');
   dom.hybridModeSelect = document.getElementById('spawn-rapid-hybrid-mode');
   dom.prefillStepSizeSelect = document.getElementById('spawn-rapid-prefill-step-size');
   dom.speculativeEnabledCheck = document.getElementById('spawn-rapid-speculative-enabled');
   dom.speculativeSourceSelect = document.getElementById('spawn-rapid-speculative-source');
   dom.speculativeModelInput = document.getElementById('spawn-rapid-speculative-model');
   dom.speculativeTokensSelect = document.getElementById('spawn-rapid-speculative-tokens');
    dom.speculativeDisableAutoKCheck = document.getElementById('spawn-rapid-speculative-disable-auto-k');
    dom.speculativeTrustWrap = document.getElementById('spawn-rapid-speculative-trust-wrap');
    dom.speculativeTrustWarning = document.getElementById('spawn-rapid-speculative-trust-warning');
    dom.speculativeTrustConsent = document.getElementById('spawn-rapid-speculative-trust-consent');
    dom.speculativeRecheckBtn = document.getElementById('spawn-rapid-speculative-recheck');
    dom.speculativeRecheckStatus = document.getElementById('spawn-rapid-speculative-recheck-status');
    dom.autoToolChoiceCheck = document.getElementById('spawn-rapid-auto-tool-choice');
   dom.samplingModeSelect   = document.getElementById('spawn-sampling-mode');

  // Step 4 (Summary)
  dom.summaryList      = document.getElementById('spawn-summary-list');
  dom.summaryWarnings  = document.getElementById('spawn-summary-warnings');
  dom.topKInput        = document.getElementById('spawn-top-k');
  dom.maxTokensInput   = document.getElementById('spawn-max-tokens');
  dom.outputModeSelect = document.getElementById('spawn-output-mode');
  dom.grammarWrap      = document.getElementById('spawn-grammar-wrap');
  dom.grammarInput     = document.getElementById('spawn-grammar');
  dom.jsonSchemaWrap   = document.getElementById('spawn-json-schema-wrap');
  dom.jsonSchemaInput  = document.getElementById('spawn-json-schema');
  // Step 5 (Preset Parameters)
  dom.presetParamsTable  = document.getElementById('preset-params-table');
  dom.savePresetBtn      = document.getElementById('spawn-save-preset-btn');
  dom.savedPresetName    = document.getElementById('spawn-saved-preset-name');
  dom.calibrationCard    = document.getElementById('spawn-calibration-card');
  dom.calibrationCheckBtn = document.getElementById('spawn-calibration-check-btn');
  dom.calibrationStartBtn = document.getElementById('spawn-calibration-start-btn');
  dom.calibrationApplyBtn = document.getElementById('spawn-calibration-apply-btn');
  dom.calibrationStatus   = document.getElementById('spawn-calibration-status');
  dom.calibrationMatches  = document.getElementById('spawn-calibration-matches');
  dom.presetNameInput    = document.getElementById('spawn-preset-name-input');
  dom.portInput        = document.getElementById('spawn-port');
  dom.bindHostSelect   = document.getElementById('spawn-bind-host');
  dom.apiKeyInput      = document.getElementById('spawn-api-key');

  // Step 5
  dom.spawnServerBtn = document.getElementById('spawn-server-btn');
  dom.statusText     = document.getElementById('spawn-status-text');
  dom.progressFill   = document.getElementById('spawn-progress-fill');
  dom.errorText      = document.getElementById('spawn-error-text');
  dom.successText    = document.getElementById('spawn-success-text');

  // Binary prereq banner
  dom.binaryPrereq        = document.getElementById('wizard-binary-prereq');
  dom.prereqIdle          = document.getElementById('wizard-prereq-idle');
  dom.prereqProgress      = document.getElementById('wizard-prereq-progress');
  dom.prereqSuccess       = document.getElementById('wizard-prereq-success');
  dom.prereqDownloadBtn   = document.getElementById('wizard-prereq-download-btn');
  dom.prereqSettingsBtn   = document.getElementById('wizard-prereq-settings-btn');
  dom.prereqPathRow       = document.getElementById('wizard-prereq-path-row');
  dom.prereqPath          = document.getElementById('wizard-prereq-path');
  dom.prereqBar           = document.getElementById('wizard-prereq-bar');
  dom.prereqElapsed       = document.getElementById('wizard-prereq-elapsed');
  dom.prereqSuccessText   = document.getElementById('wizard-prereq-success-text');

  // Model card panel
  dom.cardBackdrop        = document.getElementById('wizard-card-backdrop');
  dom.cardPanel           = document.getElementById('wizard-card-panel');
  dom.cardPanelTitle      = document.getElementById('wizard-card-panel-title');
  dom.cardPanelHfLink     = document.getElementById('wizard-card-panel-hf-link');
  dom.cardPanelClose      = document.getElementById('wizard-card-panel-close');
  dom.cardLoading         = document.getElementById('wizard-card-loading');
  dom.cardError           = document.getElementById('wizard-card-error');
   dom.cardFrontmatter     = document.getElementById('wizard-card-frontmatter');
   dom.cardFrontmatterPre  = document.getElementById('wizard-card-frontmatter-content');
   dom.cardContent         = document.getElementById('wizard-card-content');
   dom.rapidMlxAdvancedSection = document.getElementById('rapid-mlx-advanced-section');
   dom.rapidMlxAdvancedFlags = document.getElementById('rapid-mlx-advanced-flags');
}

// ── Events ────────────────────────────────────────────────────────────────────

function bindEvents() {
  dom.closeBtn?.addEventListener('click', closeSpawnWizard);
   bindCtxQuickPicks();
   bindSectionToggles();
   document.addEventListener('keydown', e => {
    if (!dom.overlay?.classList.contains('open')) return;
    if (e.key === 'Escape') {
      // Close any open browse dropdown first; only close wizard if none were open
      const anyOpen = ['spawn-browse-dropdown', 'spawn-import-browse-dropdown']
        .some(id => document.getElementById(id)?.style.display !== 'none');
      if (anyOpen) { _closeBrowseDropdowns(); return; }
      closeSpawnWizard();
    }
  });

  // Close browse dropdowns when clicking outside them
  document.addEventListener('click', e => {
    if (!e.target.closest('.browse-split')) _closeBrowseDropdowns();
  });

  dom.backBtn?.addEventListener('click', () => {
    if (wizardState.currentStep <= 0) return;
    // Each forward step pushed a history entry, so popping keeps the browser and
    // the wizard's own Back button in sync (no duplicate forward entries). Fall
    // back to a direct step change if we're somehow not on a /spawn route.
    if (location.pathname === '/spawn' || location.pathname.startsWith('/spawn/')) {
      history.back();
    } else {
      showStep(wizardState.currentStep - 1);
    }
  });
  dom.nextBtn?.addEventListener('click', () => {
    const next = wizardState.currentStep + 1;
    if (next < STEP_LABELS.length) {
      if (!validateStep(wizardState.currentStep)) return;
      showStep(next);
    }
  });
  dom.closeWizardBtn?.addEventListener('click', closeSpawnWizard);

   // Use-case cards map to workload_scenario strings for backend VRAM estimation.
   dom.usecaseCards?.forEach(card => {
      card.addEventListener('click', () => {
        wizardState.useCase = card.dataset.usecase;
        dom.usecaseCards.forEach(c => c.classList.remove('selected'));
        card.classList.add('selected');
        const profileId = USE_CASE_TO_PROFILE[card.dataset.usecase];
        if (profileId) wizardState.hardware.workloadScenario = profileId;
        applyUseCaseKvDtype(card.dataset.usecase);
        _updateKvProvenanceChips();
        updateVramDisplay();
        refreshStepGuardrails();
      });
    card.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); card.click(); } });
  });

  // Model source cards
  dom.engineCards?.forEach(card => {
    card.addEventListener('click', () => selectWizardEngine(card.dataset.engine, true));
  });

  dom.modelSourceCards?.forEach(card => {
    card.setAttribute('tabindex', '0'); card.setAttribute('role', 'button');
    card.addEventListener('click', () => {
      wizardState.model.source = card.dataset.source;
      wizardState.model.rapidMlxSource = null;
      if (card.dataset.source === 'local' && !wizardState.model.delivery) wizardState.model.delivery = 'local_file';
      if (card.dataset.source === 'import') wizardState.model.delivery = 'imported_local';
      if (card.dataset.source === 'hf') wizardState.model.delivery = 'stream_hf';
      dom.modelSourceCards.forEach(c => c.classList.remove('selected'));
      card.classList.add('selected');
      if (card.dataset.source !== 'hf') hfHideDownloadPanel(document.getElementById('hf-download-panel'));
      updateModelInputVisibility();
      renderLocalModelHint();
      clearValidationError();
      if (card.dataset.source === 'import') loadThirdPartyModels();
      refreshEngineRecommendation();
      refreshStepGuardrails();
    });
    card.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); card.click(); } });
  });

  dom.browseModelBtn?.addEventListener('click', async () => {
    const rapid = wizardState.engine.selected === 'rapid_mlx';
    let defaultPath = dom.modelPathInput?.value.trim() || '';
    if (!defaultPath) {
      // Fetch the effective models directory so Browse opens there by default.
      try {
        const headers = window.authHeaders ? window.authHeaders() : {};
        const r = await fetch('/api/hf/download-dir', { headers });
        if (r.ok) {
          const d = await r.json();
          defaultPath = d.dir || '';
          if (defaultPath) {
            const separator = defaultPath.includes('\\') ? '\\' : '/';
            const subdir = rapid ? ['mlx', 'native'] : ['gguf'];
            defaultPath = defaultPath.replace(/[\\/]+$/, '') + separator + subdir.join(separator);
          }
        }
      } catch { /* ignore — fall back to home */ }
    } else {
      // Strip the filename to get the parent directory.
      const sep = defaultPath.includes('\\') ? '\\' : '/';
      const parts = defaultPath.split(sep);
      parts.pop();
      defaultPath = parts.join(sep) || (defaultPath.includes('\\') ? 'C:\\' : '/');
    }
    openDeferredFileBrowser(
      'spawn-model-path',
      rapid ? 'dir' : 'gguf',
      defaultPath,
      { kind: 'model', engine: rapid ? 'rapid_mlx' : 'llama_cpp' },
    );
  });
  dom.importBrowseBtn?.addEventListener('click', () => openModelFileBrowser('spawn-import-path', 'gguf', null, 'model'));

  dom.modelPathInput?.addEventListener('input', () => {
    wizardState.model.path = dom.modelPathInput.value.trim();
    wizardState.model.source = 'local';
    wizardState.model.delivery = 'local_file';
    wizardState.model.rapidMlxSource = null;
    if (wizardState.model.localMeta?.path && wizardState.model.localMeta.path !== wizardState.model.path) {
      wizardState.model.localMeta = null;
    }
    wizardState.model.quantFiles = [];
    wizardState.model._quantSwapRepo = '';
    wizardState.model.originRepo = '';
    wizardState.model.originFile = '';
    wizardState.model.family = '';
    wizardState.model.cardUrl = '';
    resetQuantSwapSearchState();
    resetTagsRowOrigin();
    resetOriginState();
    onModelPathChanged();
    renderLocalModelHint();
    // Start origin resolver immediately so autoInstallChatTemplate can await it.
    // loadLocalModel also triggers one, but _autoResolveHfOrigin checks for
    // originRepo and is idempotent — the first call wins, second is a no-op.
    startOriginResolve();
  });

  dom.importPathInput?.addEventListener('input', () => {
    wizardState.model.path = dom.importPathInput.value.trim();
    wizardState.model.source = 'import';
    wizardState.model.delivery = 'imported_local';
    wizardState.model.localMeta = null;
    wizardState.model.originRepo = '';
    wizardState.model.originFile = '';
    wizardState.model.family = '';
    wizardState.model.cardUrl = '';
    resetOriginState();
    onModelPathChanged();
    renderLocalModelHint();
    // Start origin resolver immediately so autoInstallChatTemplate can await it.
    startOriginResolve();
  });

  dom.hfRepoInput?.addEventListener('input', () => {
    wizardState.model.hfRepo = dom.hfRepoInput.value.trim();
    if (!wizardState.model.hfRepo) wizardState.model.hfFile = '';
    wizardState.model.rapidMlxSource = null;
    refreshEngineRecommendation();
    refreshStepGuardrails();
    scheduleRapidMlxProfileFetch(wizardState.model.hfRepo);
  });
  dom.hfRepoInput?.addEventListener('blur', () => triggerHfFileFetch());
  dom.hfRepoInput?.addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.preventDefault(); triggerHfFileFetch(); }
  });

  document.getElementById('spawn-chat-template-path')?.addEventListener('input', (e) => {
    const value = e.target.value.trim();
    if (!value) return;
    _applyCustomChatTemplate(value);
  });

  // Hardware fields
  [
    dom.gpuLayersSelect, dom.gpuLayersManualInput, dom.contextSizeInput,
    dom.batchSizeInput, dom.ubatchSizeInput, dom.parallelSlotsInput,
    dom.cacheTypeKSelect, dom.cacheTypeVSelect, dom.nCpuMoeInput,
    dom.tensorSplitInput, dom.specTypeSelect, dom.draftModelInput,
    dom.kvUnifiedSelect, dom.flashAttnSelect, dom.mlockCheck, dom.prioSelect,
    dom.verbosityInput, dom.loadModeSelect,
    dom.ctxCheckpointsInput, dom.checkpointMinStepInput, dom.cacheReuseInput,
    dom.cacheIdleSlotsSelect, dom.noContBatchingCheck, dom.swaFullCheck,
    dom.mmprojOffloadSelect, dom.llamaReasoningEffortSelect,
    dom.llamaReasoningFormatSelect, dom.llamaReasoningPreserveSelect,
    dom.threadsInput, dom.threadsBatchInput,
    dom.fitEnableSelect, dom.fitTargetInput, dom.cacheRamInput, dom.cacheModeSelect,
    dom.specDraftNMinInput, dom.specDraftPMinInput,
  ].forEach(el => {
    el?.addEventListener('input', onHardwareChange);
    el?.addEventListener('change', onHardwareChange);
  });

  dom.cacheModeSelect?.addEventListener('change', () => {
    const wrap = document.getElementById('spawn-cache-ram-wrap');
    if (wrap) wrap.style.display = dom.cacheModeSelect.value === 'custom' ? '' : 'none';
  });

  // A change on either KV select means the user now has an opinion about KV quantization,
  // so the use-case cards stop seeding it. Advisor suggestions dispatch a real change event
  // through these same controls, and accepting one is just as much a choice as picking from
  // the dropdown, so it counts too.
  [dom.cacheTypeKSelect, dom.cacheTypeVSelect].forEach(el => {
    el?.addEventListener('change', () => { wizardState.hardware.kvDtypeUserSet = true; _updateKvProvenanceChips(); });
  });

  _updateKvProvenanceChips();

  // mmproj "Browse" button: open file browser for mmproj projectors
  const mmprojBrowseBtn = document.querySelector('#hw-mmproj-browse-btn');
  if (mmprojBrowseBtn) {
    mmprojBrowseBtn.addEventListener('click', async () => {
      const row = document.getElementById('hw-mmproj-row');
      const select = document.getElementById('hw-mmproj-select');
      const hfPanel = row?.querySelector('.hw-mmproj-hf-panel');
      // When HF download panel is shown, clear it and restore the select
      if (hfPanel) hfPanel.remove();
      if (select && select.style.display === 'none') select.style.display = '';
      await openModelFileBrowser('hw-mmproj-select', 'gguf', null, 'mmproj');
    });
  }

  // Ensure mmproj select always updates wizardState on change, even when
  // file was chosen via Browse (and no auto-detected local mmprojs existed).
  const mmprojSelect = document.getElementById('hw-mmproj-select');
  if (mmprojSelect && !mmprojSelect.dataset.boundGlobal) {
    mmprojSelect.dataset.boundGlobal = '1';
    mmprojSelect.addEventListener('change', () => {
      const fpath = mmprojSelect.value || '';
      wizardState.model.mmprojPath = fpath || '';
      wizardState.model.mmprojHfFile = fpath || '';
      wizardState.arch.mmprojBytes = 0;
      scheduleVramUpdate();
    });
  }

  // draft-model "Browse" button: open file browser for draft model
  const draftBrowseBtn = document.querySelector('#spawn-draft-browse-btn');
  if (draftBrowseBtn) {
    draftBrowseBtn.addEventListener('click', async () => {
      await openModelFileBrowser('spawn-draft-model', 'gguf', null, 'draft-model');
    });
  }

  bindHardwareToggleSwitch(dom.mlockLabel, dom.mlockCheck);

  dom.gpuLayersSelect?.addEventListener('change', () => {
    wizardState.hardware.gpuLayers = dom.gpuLayersSelect.value;
    if (dom.gpuLayersManualWrap) dom.gpuLayersManualWrap.style.display = dom.gpuLayersSelect.value === 'manual' ? '' : 'none';
    refreshStepGuardrails();
  });
  dom.specTypeSelect?.addEventListener('change', () => {
    const v = dom.specTypeSelect.value;
    const isNgram = v && (v.includes('ngram') || v === 'ngram');
    const isDraftMtp = v && v.includes('draft-mtp');
    const isDraftModel = v === 'draft-model';

    // draft-model: required external draft path.
    // draft-mtp variants: optional external model (overrides built-in heads); ngram-only: no draft model.
    const showDraftPath = isDraftModel || isDraftMtp;
    if (dom.draftModelWrap) {
      dom.draftModelWrap.style.display = showDraftPath ? '' : 'none';
      const lbl = dom.draftModelWrap.querySelector('label');
      if (lbl) lbl.textContent = isDraftMtp ? 'Draft model path (optional)' : 'Draft model path';
      let hint = dom.draftModelWrap.querySelector('.draft-model-hint');
      if (isDraftMtp) {
        if (!hint) {
          hint = document.createElement('div');
          hint.className = 'field-hint draft-model-hint';
          dom.draftModelWrap.appendChild(hint);
        }
        hint.textContent = 'Leave blank to use the built-in MTP heads. Enter a path to override with a separate draft model.';
      } else if (hint) {
        hint.remove();
      }
    }
    // Draft-MTP KV quant selects: only relevant when MTP is active.
    if (dom.draftKvRow) {
      dom.draftKvRow.style.display = isDraftMtp ? '' : 'none';
    }
    // Auto-expand the speculative decoding details when a mode is active;
    // update the collapsed summary badge so users can see the mode at a glance.
    const specDetails = document.getElementById('spawn-spec-details');
    if (specDetails && v) specDetails.open = true;
    const badge = document.getElementById('spawn-spec-summary-badge');
    if (badge) {
      badge.textContent = v ? `— ${v}` : '';
      badge.style.display = v ? '' : 'none';
    }

    // For draft-mtp: ensure MTP draft section is visible and auto-populate
    // from candidates if not already set.
    if (isDraftMtp) {
      renderMtpSection();
      // Also auto-populate the draft model input from candidates
      const candidates = wizardState.model.draftCandidates || [];
      const existing = (dom.draftModelInput?.value || '').trim();
      if (!existing && candidates.length > 0) {
        const best = _bestDraftForModel(
          (wizardState.model.path || wizardState.model.hfFile || '').split(/[\\/]/).pop() || '',
          candidates,
        );
        if (best && dom.draftModelInput) {
          dom.draftModelInput.value = best.path;
        }
      }
      // Sync from selectedDraftPath if set (e.g. from MTP draft dropdown)
      const selectedPath = wizardState.model.selectedDraftPath || '';
      if (selectedPath && dom.draftModelInput && !dom.draftModelInput.value) {
        dom.draftModelInput.value = selectedPath;
      }
    }

    // Auto-populate draft-model input when mode matches and candidates exist
    if (isDraftModel) {
      const candidates = wizardState.model.draftCandidates || [];
      const existing = (dom.draftModelInput?.value || '').trim();
      if (!existing && candidates.length > 0) {
        const best = _bestDraftForModel(
          (wizardState.model.path || wizardState.model.hfFile || '').split(/[\\/]/).pop() || '',
          candidates,
        );
        if (best && dom.draftModelInput) {
          dom.draftModelInput.value = best.path;
        }
      }
    }

    _updateSpecHint(v);
    refreshStepGuardrails();
  });

  // MoE slider
  dom.moeOffloadSlider?.addEventListener('input', () => {
    wizardState.hardware.nCpuMoe = Number(dom.moeOffloadSlider.value);
    if (dom.nCpuMoeInput) dom.nCpuMoeInput.value = wizardState.hardware.nCpuMoe;
    updateMoeSliderVisuals();
    scheduleVramUpdate();
  });

  // MoE offload auto-tuner
  document.getElementById('spawn-moe-autotune-btn')?.addEventListener('click', () => autoTuneWizard(false));
  document.getElementById('spawn-moe-autotune-verify')?.addEventListener('click', () => autoTuneWizard(true));

  // Batch/ubatch sweep
  document.getElementById('wizard-batch-sweep-btn')?.addEventListener('click', runBatchSweep);
  // Depth sweep
  document.getElementById('wizard-depth-sweep-btn')?.addEventListener('click', runDepthSweep);

  // Auto-size button
  dom.vramAutosizeBtn?.addEventListener('click', triggerAutoSize);

  dom.savePresetBtn?.addEventListener('click', saveAsPreset);
  dom.calibrationCheckBtn?.addEventListener('click', checkWizardCalibrationEvidence);
  dom.calibrationStartBtn?.addEventListener('click', startWizardCalibration);
  dom.calibrationApplyBtn?.addEventListener('click', applySelectedWizardCalibration);
  dom.spawnServerBtn?.addEventListener('click', spawnServer);



  // Sampling fields in review step
  _bindSamplingFields();
  dom.portInput?.addEventListener('input', () => {
    const parsed = parseInt(dom.portInput.value, 10);
    wizardState.access.port = Number.isFinite(parsed) && parsed > 0 ? parsed : 8001;
    if (wizardState.currentStep === 1) renderSummary();
    refreshStepGuardrails();
  });
  dom.bindHostSelect?.addEventListener('change', () => {
    wizardState.access.bindHost = dom.bindHostSelect.value || '127.0.0.1';
    if (wizardState.currentStep === 1) renderSummary();
    refreshStepGuardrails();
  });
  dom.apiKeyInput?.addEventListener('input', () => {
    wizardState.access.apiKey = (dom.apiKeyInput.value || '').trim();
    if (wizardState.currentStep === 1) renderSummary();
    refreshStepGuardrails();
  });

  // Hardware step quant swap (HF models and local-model quant discovery)
  document.getElementById('hw-quant-select')?.addEventListener('change', e => {
    const fpath = e.target.value;
    const qf = wizardState.model.quantFiles?.find(q => (q.path || q.name) === fpath);
    if (qf) {
      wizardState.model.hfFile = fpath;
      refreshEngineRecommendation();
      // Always reset modelBytes so getModelBytes() re-estimates from the new filename
      // if the file size is unknown — stale size from a different quant would corrupt the math.
      wizardState.model.modelBytes = Number(qf.size) || 0;
      if (wizardState.arch.mtpDepth > 0) {
        renderMtpSection();
      }
      scheduleVramUpdate();
      // For local models with quant-swap repo: show download / stream actions.
      const isLocalSwap = (wizardState.model.source === 'local' || wizardState.model.source === 'import')
        && wizardState.model._quantSwapRepo;
      if (isLocalSwap) {
        // Only show actions when the user picks a quant that differs from their current local file.
        const currentFile = (wizardState.model.path || '').split(/[\\/]/).pop() || '';
        const pickedFile = fpath.split('/').pop() || fpath;
        if (pickedFile.toLowerCase() !== currentFile.toLowerCase()) {
          _renderQuantSwapActions(fpath, wizardState.model._quantSwapRepo);
        } else {
          const actionsRow = document.getElementById('hw-quant-swap-actions');
          if (actionsRow) actionsRow.style.display = 'none';
        }
      } else {
        const panel = document.getElementById('hf-download-panel');
        if (panel && panel.style.display !== 'none') hfShowDownloadPanel(panel, fpath.split('/').pop());
      }
      refreshStepGuardrails();
    }
  });

  // Local-model quant swap trigger
  document.getElementById('hw-quant-local-btn')?.addEventListener('click', () => {
    // If quants were already found by the background auto-discover, just open the
    // dropdown immediately instead of re-searching.
    if (wizardState.model.quantFiles?.length > 1 && !wizardState.model._quantSwapRepo) {
      wizardState.model._quantSwapRepo = wizardState.model.originRepo || '_local_quants_';
      const statusEl = document.getElementById('hw-quant-local-status');
      if (statusEl) statusEl.textContent = '';
      renderHardwareModelHeader();
      return;
    }
    resetQuantSwapSearchState(); // reset cache so a re-click re-searches
    _autoDiscoverLocalModelQuants(true); // user-triggered: will show dropdown on success
  });

  // Library tag picker trigger
  document.getElementById('hw-tags-add-btn')?.addEventListener('click', e => {
    _openHwTagPicker(
      e.currentTarget,
      wizardState.model.path || '',
      wizardState.model.originRepo || '',
    );
  });

  // Model card panel
  dom.cardPanelClose?.addEventListener('click', _closeCardPanel);
  dom.cardBackdrop?.addEventListener('click', _closeCardPanel);

  // "Settings → Models" link inside the Import card description
  document.querySelector('.wizard-settings-link[data-open-settings="models"]')?.addEventListener('click', e => {
    e.preventDefault();
    Router.navigate('/settings#models');
  });

   // Binary prereq buttons
   dom.prereqDownloadBtn?.addEventListener('click', _downloadBinaryForWizard);
   dom.prereqSettingsBtn?.addEventListener('click', () => {
     Router.navigate('/settings#session');
     setTimeout(() => document.getElementById('set-server-path')?.focus(), 80);
   });

   // Rapid-MLX advanced controls
   bindRapidMlxAdvancedControls();
}

// ── Workload profile helpers ─────────────────────────────────────────────────
// Dedicated step-3 picker removed (Phase 7B2). workloadScenario is now derived
// from page-1 use-case selection only; no UI confirmation gate.

// Seed the KV dtype from the chosen use case, on llama.cpp only.
//
// This is guidance, not policy: it moves the starting point, and the moment the user
// expresses a KV preference of their own -- by changing the control, accepting an advisor
// suggestion, or loading a preset that carries one -- it stops touching the value. Same
// omission-only rule the VRAM estimator uses for its scenario-derived parameters: fill the
// gap, never overwrite an answer someone already gave.
function applyUseCaseKvDtype(useCase) {
  if (wizardState.engine.selected === 'rapid_mlx') return;
  if (wizardState.hardware.kvDtypeUserSet) return;

  const dtype = USE_CASE_TO_KV_DTYPE[useCase];
  if (!dtype) return;

  wizardState.hardware.cacheTypeK = dtype;
  wizardState.hardware.cacheTypeV = dtype;
  if (dom.cacheTypeKSelect) dom.cacheTypeKSelect.value = dtype;
  if (dom.cacheTypeVSelect) dom.cacheTypeVSelect.value = dtype;
}

// Provenance chips (M3-B): small pill next to KV labels showing where the value came from.
const USE_CASE_PROV_LABELS = { agentic: 'Agentic / RAG / tools', general: 'General chat / roleplay', roleplay: 'General chat / roleplay' };

function _updateKvProvenanceChips() {
  if (wizardState.engine.selected === 'rapid_mlx') return;
  const isAuto = !wizardState.hardware.kvDtypeUserSet;
  const [kField, vField] = [
    document.getElementById('spawn-cache-type-k')?.closest('.kv-inline-field'),
    document.getElementById('spawn-cache-type-v')?.closest('.kv-inline-field'),
  ];
  for (const field of [kField, vField]) {
    if (!field) continue;
    let chip = field.querySelector('.prov-chip');
    const label = field.querySelector('label');
    if (!chip && label) {
      chip = document.createElement('span');
      chip.className = 'prov-chip';
      label.appendChild(document.createTextNode(' '));
      label.appendChild(chip);
    }
    if (!chip) continue;
    chip.className = 'prov-chip';
    if (isAuto) {
      const source = USE_CASE_PROV_LABELS[wizardState.useCase] || 'Default';
      chip.className += ' prov-chip-auto';
      chip.textContent = `auto · ${source}`;
    } else {
      chip.className += ' prov-chip-you';
      chip.textContent = 'you';
    }
  }
}

function _ensureKvUserSet() {
  wizardState.hardware.kvDtypeUserSet = true;
  _updateKvProvenanceChips();
}

export async function refreshHfTokenState() {
  try {
    const headers = window.authHeaders ? window.authHeaders() : {};
    const res = await fetch('/api/hf/token', { headers });
    if (!res.ok) return;
    const data = await res.json();
    wizardState.model.hfTokenSet = !!data.set;
    _updateWizardHfTokenUI(!!data.set);
  } catch {}
  refreshStepGuardrails();
}

function _updateWizardHfTokenUI(isSet) {
  const badge      = document.getElementById('wizard-hf-token-badge');
  const inputRow   = document.getElementById('wizard-hf-token-input-row');
  const savedRow   = document.getElementById('wizard-hf-token-saved-row');
  if (badge) {
    badge.textContent = isSet ? '✓ Active' : 'Not set';
    badge.className = 'wizard-hf-token-badge ' + (isSet ? 'token-badge-ok' : 'token-badge-none');
    badge.style.display = '';
  }
  if (inputRow) inputRow.style.display = isSet ? 'none' : '';
  if (savedRow) savedRow.style.display = isSet ? ''     : 'none';
}

function bindWizardHfToken() {
  const saveBtn   = document.getElementById('wizard-hf-token-save');
  const removeBtn = document.getElementById('wizard-hf-token-remove');
  const input     = document.getElementById('wizard-hf-token-input');

  saveBtn?.addEventListener('click', async () => {
    const token = input?.value.trim() || '';
    if (!token) { input?.focus(); return; }
    const origText = saveBtn.textContent;
    saveBtn.disabled = true; saveBtn.textContent = 'Saving…';
    try {
      const headers = window.authHeaders
        ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
        : { 'Content-Type': 'application/json' };
      const res = await fetch('/api/hf/token', { method: 'PUT', headers, body: JSON.stringify({ token }) });
      const data = await res.json().catch(() => ({}));
      if (data.ok) {
        if (input) input.value = '';
        saveBtn.textContent = '✓ Saved';
        setTimeout(() => { saveBtn.textContent = origText; saveBtn.disabled = false; }, 1500);
        await refreshHfTokenState();
      } else {
        saveBtn.textContent = 'Failed'; setTimeout(() => { saveBtn.textContent = origText; saveBtn.disabled = false; }, 2000);
      }
    } catch {
      saveBtn.textContent = 'Error'; setTimeout(() => { saveBtn.textContent = origText; saveBtn.disabled = false; }, 2000);
    }
  });

  // Allow Enter key in token input to trigger save
  input?.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); saveBtn?.click(); } });

  removeBtn?.addEventListener('click', async () => {
    try {
      const headers = window.authHeaders ? window.authHeaders() : {};
      await fetch('/api/hf/token', { method: 'DELETE', headers });
      await refreshHfTokenState();
    } catch {}
  });
}

export function renderLocalModelHint() {
  if (!dom.localModelHint) return;
  const meta = wizardState.model.localMeta;
  const isLocalSource = wizardState.model.source === 'local' || wizardState.model.source === 'import';
  if (!isLocalSource || !meta) {
    dom.localModelHint.style.display = 'none';
    return;
  }
  dom.localModelHint.style.display = '';
  if (dom.localModelHintTitle) {
    dom.localModelHintTitle.textContent = meta.model_name || meta.name || meta.filename || (meta.path?.split(/[\\/]/).pop() || 'Selected model');
  }
  if (dom.localModelHintMeta) {
    const parts = [];
    if (meta.size_display) parts.push(meta.size_display);
    if (meta.quant_type) parts.push(meta.quant_type);
    if (meta.param_b != null) parts.push(formatParams(meta.param_b));
    if (meta.vram_est_gb != null) parts.push(`~${Number(meta.vram_est_gb).toFixed(0)} GB weights`);
    dom.localModelHintMeta.textContent = parts.join(' · ') || 'Opened from your local model library.';
  }
  _refreshHfOriginSection();
}

function getStepGuardState(step = wizardState.currentStep) {
  const info = (message) => ({ canProceed: true, tone: 'info', message, focusEl: null });
  const error = (message, focusEl = null) => ({ canProceed: false, tone: 'error', message, focusEl });
  const warning = (message) => ({ canProceed: true, tone: 'warning', message, focusEl: null });

  if (step === 0) {
    const { source, path, hfRepo, hfFile } = wizardState.model;
    const rapid = wizardState.engine.selected === 'rapid_mlx';
    const artifactKind = classifyWizardArtifact();
    if (rapid && !wizardState.engine.rapidMlxLocalAvailable) {
      return error('Local Rapid-MLX launch requires Apple Silicon macOS. Remote Rapid-MLX endpoints can still be attached from the welcome screen.');
    }
    if (rapid && wizardState.engine.recommendation?.state === 'checking') {
      return error('Checking the selected Rapid-MLX runtime compatibility…');
    }
    if (rapid && wizardState.engine.recommendation?.state === 'runtime_required') {
      return error('Install a compatible Rapid-MLX runtime in Settings before launching this engine.');
    }
    if (rapid && artifactKind === 'gguf') {
      return error('GGUF runs with llama.cpp. Switch engines or choose a validated MLX source.', dom.modelPathInput || dom.hfFileList);
    }
    if (!rapid && ['mlx_directory', 'authoritative_safetensors', 'rapid_mlx_alias', 'rapid_mlx_hf_repository'].includes(artifactKind)) {
      return error('This typed model source requires Rapid-MLX. Switch engines to continue.', dom.modelPathInput || dom.hfRepoInput);
    }
    if (source === 'local') {
      if (rapid && artifactKind === 'authoritative_safetensors'
          && !(wizardState.model.rapidMlxSource || wizardState.model.localMeta?.model_source)) {
        return error('Choose this safetensors model from the model library so its verified source revision and conversion recipe are preserved.', dom.modelPathInput);
      }
      return path
        ? info(rapid ? 'MLX model selected. Continue to review backend-specific launch settings.' : 'Local model selected. Continue to tune hardware and context settings.')
        : error(rapid ? 'Enter a validated local MLX model directory to continue.' : 'Choose a local GGUF file to continue.', dom.modelPathInput);
    }
    if (source === 'import') {
      return path
        ? info('Imported model selected. Continue to tune hardware and context settings.')
        : error('Pick an imported model or paste a GGUF path to continue.', dom.importPathInput);
    }
    if (!hfRepo) {
      return error('Enter a Hugging Face repo ID or pick a discover result to continue.', dom.hfRepoInput);
    }
    if (!hfFile && !rapid) {
      return error('Choose a GGUF file from the selected Hugging Face repo to continue.', dom.hfFileList || dom.hfRepoInput);
    }
    return info('Hugging Face model selected. Continue to review its hardware fit.');
  }

  if (step === 1) {
    // Workload/use-case is now chosen on page 1 and auto-applied with a sane
    // default (Phase 7B2's dedicated step-3 picker + confirmation gate was
    // redundant with it and has been removed).
    if (wizardState.engine.selected === 'rapid_mlx') {
      return info('Rapid-MLX backend controls remain isolated from llama.cpp memory and speculation flags.');
    }
    if (wizardState.hardware.gpuLayers === 'manual' && wizardState.hardware.gpuLayersManual == null) {
      return error('Enter a GPU layer count or switch GPU layers back to Auto.', dom.gpuLayersManualInput);
    }
    if (dom.fitEnableSelect?.value === 'true' && !dom.fitTargetInput?.value.trim()) {
      return error('Enter a fit target in MB or turn Auto-fit context to memory off.', dom.fitTargetInput);
    }
    // draft-model requires an external file; draft-mtp uses built-in prediction heads (no file needed).
    const specType = dom.specTypeSelect?.value || '';
    if (specType === 'draft-model' && !dom.draftModelInput?.value.trim()) {
      return error('Enter a draft model path for speculative decoding.', dom.draftModelInput);
    }
    // Network/access guard (formerly a separate Review step; its DOM is now
    // part of this same Hardware & memory step).
    if (wizardState.access.bindHost === '0.0.0.0' && !wizardState.access.apiKey) {
      return warning('This server will be LAN-visible without an API key. Add one unless you intentionally want an open endpoint.');
    }
    return info('Review the VRAM estimate and network exposure before continuing.');
  }

  if (step === 2) {
    return info('Saving a preset is optional. This starts the server with the configuration shown above.');
  }

  return info('');
}

export function refreshStepGuardrails() {
  const state = getStepGuardState();
  if (dom.nextBtn) {
    const isFinalStep = wizardState.currentStep >= STEP_LABELS.length - 1;
    dom.nextBtn.disabled = !isFinalStep && !state.canProceed;
    dom.nextBtn.title = !state.canProceed ? state.message : '';
    dom.nextBtn.setAttribute('aria-disabled', String(dom.nextBtn.disabled));
  }
  if (dom.footerHint) {
    dom.footerHint.textContent = state.message || '';
    dom.footerHint.classList.remove('is-warning', 'is-error');
    if (state.tone === 'warning') dom.footerHint.classList.add('is-warning');
    if (state.tone === 'error') dom.footerHint.classList.add('is-error');
  }
}

// ── Validation ────────────────────────────────────────────────────────────────

function validateStep(step) {
  const state = getStepGuardState(step);
  if (!state.canProceed) {
    showValidationError(state.message, state.focusEl);
    refreshStepGuardrails();
    return false;
  }
  clearValidationError();
  refreshStepGuardrails();
  return true;
}

export function showValidationError(msg, focusEl = null) {
  const stepEl = document.getElementById(`wizard-step-${wizardState.currentStep}`);
  if (!stepEl) return;
  let el = stepEl.querySelector('.wizard-validation-error');
  if (!el) {
    el = document.createElement('div');
    el.className = 'wizard-validation-error';
    el.setAttribute('role', 'alert');
    stepEl.querySelector('.wizard-main')?.prepend(el);
  }
  el.textContent = msg;
  el.style.display = '';
  // Scroll error visible, then if there's a focus target (e.g. confirmation checkbox),
  // focus it and scroll it into view so the user sees both the error and the control to fix it.
  el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  if (focusEl) {
    focusEl.focus?.();
    focusEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
}

export function clearValidationError() {
  dom.overlay?.querySelectorAll('.wizard-validation-error').forEach(el => { el.style.display = 'none'; });
  refreshStepGuardrails();
}

// ── HF download panel (wizard-specific wrappers) ─────────────────────────────


export function getAuthHeaders() {
  return window.authHeaders ? window.authHeaders() : {};
}


export function showStep(index) {
  wizardState.currentStep = index;

  // Plan §6.2: apply a preserve-and-restore snapshot taken before re-entering step 0 (Model)
  // from a chat-template-degraded warning's "Switch to repo selection" button. Applied on
  // arrival at step 1 (Hardware) — by then the new model selection is complete, so
  // context-length-dependent fields can be re-validated against it.
  if (index === 1) {
    _applyPendingRestore();
    _renderAllSettingsDrawer();
    renderProLayout(wizardState.viewMode);
    // Initialize Option A (Guided) decision cards on first step 2 visit
    if (!window._guidedCardsInitialized) {
      initGuidedCards();
      window._guidedCardsInitialized = true;
    }
  }

  // Keep the URL in sync so browser Back/Forward traverses wizard steps. Only
  // when the wizard is open and already on a /spawn route; pushState (no dispatch)
  // avoids recursing through the route handler. When the route handler drives the
  // step, target === current so this is a no-op.
  if (!_suppressStepUrl && dom.overlay && dom.overlay.classList.contains('open') &&
      (location.pathname === '/spawn' || location.pathname.startsWith('/spawn/'))) {
    const target = stepRoute(index);
    if (location.pathname !== target) {
      try { history.pushState({ path: target }, '', target); } catch {}
    }
  }

  clearValidationError();

  dom.steps?.forEach(s => s.classList.remove('active'));
  document.getElementById(`wizard-step-${index}`)?.classList.add('active');

  // wizard-body is always overflow:hidden flex — reset both column scroll
  // positions when switching steps so nothing is trapped off-screen.
  const newStep = document.getElementById('wizard-step-' + index);
  if (newStep) {
    const m = newStep.querySelector('.wizard-main');
    if (m) m.scrollTop = 0;
    const s = newStep.querySelector('.wizard-sidebar, .hw-vram-sidebar');
    if (s) s.scrollTop = 0;
  }

  dom.stepBadges?.forEach(b => {
    const s = Number(b.dataset.step);
    b.classList.remove('active', 'completed');
    if (s === index) b.classList.add('active');
    else if (s < index) b.classList.add('completed');
  });

  if (dom.stepLabel) dom.stepLabel.textContent = STEP_LABELS[index] || '';
  if (dom.backBtn) dom.backBtn.style.display = index === 0 ? 'none' : '';
  if (dom.nextBtn) dom.nextBtn.style.display = index === STEP_LABELS.length - 1 ? 'none' : '';

  if (index === 0) {
    _loadModelDirSwitcher();
    // The VRAM sidebar + quant advisor on this step need effectiveAvailBytes()
    // populated, but that data was previously only fetched on step 2 (Hardware).
    // A fresh page load never visits step 2 before the user picks a model, so
    // the sidebar silently stayed empty. Fetch it here too.
    if (!cachedMemorySnapshot && !cachedRamTotal) {
      Promise.all([fetchGpuVram(), fetchMetalGpuLimit(), fetchSystemRam(), fetchMemoryAvailability()]).then(() => {
        scheduleVramUpdate();
        onMemoryAvailabilityReady?.();
      });
    }
  }
  if (index === 1) {
    const rapid = wizardState.engine.selected === 'rapid_mlx';
    // A context-target pill picked on the Model step (step 1) only writes
    // wizardState.hardware.contextSize — it can't reach dom.contextSizeInput
    // because that field doesn't exist until this step's DOM is active. Sync
    // it here so the pill choice is actually reflected once the user arrives.
      if (dom.contextSizeInput && wizardState.hardware.contextSize > 0) {
        dom.contextSizeInput.value = wizardState.hardware.contextSize;
      }
      if (dom.loadModeSelect) dom.loadModeSelect.value = wizardState.hardware.loadMode || 'mmap';
      if (dom.verbosityInput) dom.verbosityInput.value = String(wizardState.hardware.verbosity ?? 4);
      if (dom.ctxCheckpointsInput) dom.ctxCheckpointsInput.value = wizardState.hardware.ctxCheckpoints ?? '';
 if (dom.checkpointMinStepInput) dom.checkpointMinStepInput.value = wizardState.hardware.checkpointMinStep ?? '';
 if (dom.cacheReuseInput) dom.cacheReuseInput.value = wizardState.hardware.cacheReuse ?? '';
 if (dom.cacheIdleSlotsSelect) dom.cacheIdleSlotsSelect.value = wizardState.hardware.cacheIdleSlots == null ? '' : String(wizardState.hardware.cacheIdleSlots);
 if (dom.noContBatchingCheck) dom.noContBatchingCheck.checked = !!wizardState.hardware.noContBatching;
 if (dom.swaFullCheck) dom.swaFullCheck.checked = !!wizardState.hardware.swaFull;
 if (dom.mmprojOffloadSelect) dom.mmprojOffloadSelect.value = wizardState.hardware.mmprojOffload == null ? '' : String(wizardState.hardware.mmprojOffload);
 if (dom.llamaReasoningEffortSelect) dom.llamaReasoningEffortSelect.value = wizardState.hardware.llamaReasoningEffort || 'default';
 if (dom.llamaReasoningFormatSelect) dom.llamaReasoningFormatSelect.value = wizardState.hardware.llamaReasoningFormat || '';
 if (dom.llamaReasoningPreserveSelect) dom.llamaReasoningPreserveSelect.value = wizardState.hardware.llamaReasoningPreserve == null ? '' : String(wizardState.hardware.llamaReasoningPreserve);
    if (!rapid) {
      updateCtxModelMaxHint();
      updateCtxQuickPickActive();
      updateCtxTrainWarning();
      // Also fetch the live MemoryAvailabilitySnapshot (D30/A58 single source of
      // truth) so effectiveAvailBytes() reflects current memory pressure instead
      // of only the theoretical Metal cap — this was previously Rapid-MLX-only,
      // which left llama.cpp's "available" figure ignoring what's actually free.
      Promise.all([fetchGpuVram(), fetchMetalGpuLimit(), fetchSystemRam(), fetchMemoryAvailability()]).then(() => {
        scheduleVramUpdate();
        renderHardwareModelHeader();
        _populateKvCacheOptions();
      });
    } else {
      // Rapid-MLX: fetch a fresh MemoryAvailabilitySnapshot — never stale llama/HF cache.
      // This is the single source of truth per D30/A58.
      fetchMemoryAvailability().then(() => {
        scheduleVramUpdate();
        renderHardwareModelHeader();
      });
    }
    if (!rapid) {
      // Auto-hint thread count from system P-core count (Apple Silicon)
      _refreshThreadsHint();
      // When no active session the WS doesn't broadcast system metrics — fetch directly.
      if (!lastSystemMetrics) {
        _fetchSystemInfoAndRefreshHints();
      }
    }
    if (!rapid) {
      renderMmprojSection();
      renderMtpSection();
      void _refreshLlamaBinaryCapabilities();
    }
    // Speculation is never enabled from a filename. A qualified GGUF/profile
    // capability may populate an explicit recommendation; otherwise leave the
    // control unset and let the server-owned safe default apply.
    if (!rapid) _updateSpecHint(dom.specTypeSelect?.value || '');
    // Trigger download panel now (moved from file-select to hardware step entry)
    const dlPanel = document.getElementById('hf-download-panel');
    if (wizardState.model.source === 'hf' && wizardState.model.hfFile) {
      hfShowDownloadPanel(dlPanel, wizardState.model.hfFile);
    } else {
      hfHideDownloadPanel(dlPanel);
    }
    // Fetch model-specific sampling defaults (temperature, presence_penalty, etc.)
    // so the review section pre-populates with the model's recommended settings.
    _fetchAndApplyModelSamplingDefaults();
    // Review/Summary content is now merged into this step's DOM (Option A
    // collapse); fetch VRAM and render the summary that lives further down
    // the same page instead of waiting for a separate step entry.
    if (rapid) {
      // Rapid-MLX: use fresh snapshot, not cached llama GPU values
      refreshHfTokenState().finally(() => {
        fetchMemoryAvailability().then(() => estimateVramFull().then(() => renderSummary()));
      });
    } else {
      refreshHfTokenState().finally(() => {
        Promise.all([fetchGpuVram(), fetchMetalGpuLimit()]).then(() => estimateVramFull().then(() => renderSummary()));
      });
    }
  }
  if (index === 2) {
    _renderPresetParamsStep();
    refreshWizardCalibrationOffer();
    _renderSpawnConfigCard();
  }

  // Focus management: make step container focusable and announce to screen readers
  const targetStep = document.getElementById('wizard-step-' + index);
  if (targetStep) {
    targetStep.setAttribute('tabindex', '-1');
    targetStep.setAttribute('aria-label', STEP_LABELS[index] || 'Wizard step');
    targetStep.focus();
  }

  refreshStepGuardrails();
}

// ── KV cache options from llama-server capabilities ──────────────────────────

function _populateKvCacheOptions() {
  const kSelect = dom.cacheTypeKSelect;
  const vSelect = dom.cacheTypeVSelect;
  if (!kSelect || !vSelect) return;

  // Use capabilities if available; otherwise fall back to safe defaults.
  const caps = lastCapabilities || {};
  const kvTypes = (caps.kv_cache_types || []).map(String);

  const baseOptions = [
    { value: 'q8_0', label: 'q8_0 — recommended' },
    { value: 'f16', label: 'f16 — lossless' },
    { value: 'bf16', label: 'bf16' },
    { value: 'q4_0', label: 'q4_0 — saves VRAM' },
    { value: 'q4_1', label: 'q4_1' },
    { value: 'iq4_nl', label: 'iq4_nl' },
    { value: 'q5_0', label: 'q5_0' },
    { value: 'q5_1', label: 'q5_1' },
    { value: 'f32', label: 'f32 — full precision' },
  ];

  // Build options: always include base, then add any extra from capabilities.
  const used = new Set(baseOptions.map(o => o.value));
  const allOptions = baseOptions.slice();
  for (const t of kvTypes) {
    if (!used.has(t)) {
      allOptions.push({ value: t, label: t });
      used.add(t);
    }
  }

  const fillSelect = (sel) => {
    const current = sel.value || 'q8_0';
    sel.innerHTML = '';
    for (const o of allOptions) {
      const opt = document.createElement('option');
      opt.value = o.value;
      opt.textContent = o.label;
      if (o.value === current) opt.selected = true;
      sel.appendChild(opt);
    }
    // Ensure a valid default if current is not in list
    if (!sel.querySelector('option:checked')) {
      const q8 = sel.querySelector('option[value="q8_0"]');
      if (q8) q8.selected = true;
    }
  };

  fillSelect(kSelect);
  fillSelect(vSelect);
}

function _capabilityAvailable(state) {
  return state === 'Available' || (state && Object.prototype.hasOwnProperty.call(state, 'Available'));
}

function _capabilityReason(state, fallback) {
  if (typeof state === 'string' && state !== 'Available') return state;
  if (state?.Unavailable) return state.Unavailable;
  return fallback;
}

function _setCapabilityHint(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text || '';
}

function _preserveUnknownSelectValue(select, value, label) {
  if (!select || value == null || value === '') return;
  const stringValue = String(value);
  if ([...select.options].some(option => option.value === stringValue)) return;
  const option = document.createElement('option');
  option.value = stringValue;
  option.textContent = label || `${stringValue} (stored; unsupported)`;
  option.disabled = true;
  select.appendChild(option);
  select.value = stringValue;
}

function _applyLlamaCapabilityLocks(snapshot) {
  const cache = snapshot?.cache || {};
  const typed = snapshot?.typed || {};
  const idleState = cache.idle_slot_cache;
  const idleSupported = _capabilityAvailable(idleState);
  if (dom.cacheIdleSlotsSelect) {
    dom.cacheIdleSlotsSelect.disabled = !idleSupported;
    dom.cacheIdleSlotsSelect.title = idleSupported ? '' : _capabilityReason(idleState, 'Capability evidence is unavailable for this binary.');
  }
  _setCapabilityHint('spawn-cache-idle-slots-hint', idleSupported
    ? ''
    : _capabilityReason(idleState, 'This binary does not advertise --cache-idle-slots.'));

  const mmproj = typed.mmproj_offload || {};
  const mmprojPositive = _capabilityAvailable(mmproj.positive);
  const mmprojNegative = _capabilityAvailable(mmproj.negative);
  if (dom.mmprojOffloadSelect) {
    for (const option of dom.mmprojOffloadSelect.options) {
      if (option.value === 'true') option.disabled = !mmprojPositive;
      if (option.value === 'false') option.disabled = !mmprojNegative;
    }
  }
  _setCapabilityHint('spawn-mmproj-offload-hint', mmprojPositive || mmprojNegative
    ? ''
    : _capabilityReason(mmproj.positive, 'This binary does not advertise projector offload controls.'));

  const effort = typed.reasoning_effort || {};
  const effortSupported = _capabilityAvailable(effort.supported);
  _preserveUnknownSelectValue(dom.llamaReasoningEffortSelect, wizardState.hardware.llamaReasoningEffort);
  if (dom.llamaReasoningEffortSelect) {
    for (const option of dom.llamaReasoningEffortSelect.options) {
      if (option.value === 'default') continue;
      option.disabled = !effortSupported || (effort.accepted_values?.length > 0 && !effort.accepted_values.includes(option.value));
      option.title = option.disabled ? _capabilityReason(effort.supported, 'Not advertised by this binary.') : '';
    }
  }
  _setCapabilityHint('spawn-reasoning-effort-hint', effortSupported
    ? ''
    : _capabilityReason(effort.supported, 'This binary does not advertise --reasoning-effort.'));

  const format = typed.reasoning_format || {};
  const formatSupported = _capabilityAvailable(format.supported);
  _preserveUnknownSelectValue(dom.llamaReasoningFormatSelect, wizardState.hardware.llamaReasoningFormat);
  if (dom.llamaReasoningFormatSelect) {
    for (const option of dom.llamaReasoningFormatSelect.options) {
      if (option.value === '') continue;
      option.disabled = !formatSupported || (format.accepted_values?.length > 0 && !format.accepted_values.includes(option.value));
      option.title = option.disabled ? _capabilityReason(format.supported, 'Not advertised by this binary.') : '';
    }
  }
  _setCapabilityHint('spawn-reasoning-format-hint', formatSupported
    ? ''
    : _capabilityReason(format.supported, 'This binary does not advertise --reasoning-format.'));

  const preserve = typed.reasoning_preserve || {};
  const templateState = typed.reasoning_preserve_template;
  const templateSupported = _capabilityAvailable(templateState);
  if (dom.llamaReasoningPreserveSelect) {
    for (const option of dom.llamaReasoningPreserveSelect.options) {
      if (option.value === 'true') option.disabled = !templateSupported || !preserve.positive || !_capabilityAvailable(preserve.positive);
      if (option.value === 'false') option.disabled = !templateSupported || !preserve.negative || !_capabilityAvailable(preserve.negative);
    }
  }
  _setCapabilityHint('spawn-reasoning-preserve-hint', !templateSupported
    ? _capabilityReason(templateState, 'Template compatibility for native reasoning preservation is unverified.')
    : _capabilityAvailable(preserve.positive) || _capabilityAvailable(preserve.negative)
      ? ''
      : _capabilityReason(preserve.positive, 'This binary does not advertise native reasoning preservation.'));
}

async function _refreshLlamaBinaryCapabilities() {
  if (llamaBinaryCapabilitiesPromise) return llamaBinaryCapabilitiesPromise;
  _applyLlamaCapabilityLocks(null);
  llamaBinaryCapabilitiesPromise = (async () => {
    try {
      const headers = window.authHeaders ? window.authHeaders() : {};
      const response = await fetch('/api/llama-binary/capabilities', { headers });
      if (!response.ok) return null;
      const data = await response.json();
      return data?.ok ? data.snapshot : null;
    } catch {
      return null;
    }
  })();
  llamaBinaryCapabilities = await llamaBinaryCapabilitiesPromise;
  _applyLlamaCapabilityLocks(llamaBinaryCapabilities);
  return llamaBinaryCapabilities;
}

// ── Guided disclosure & use-case ─────────────────────────────────────────────

// Guided is the only disclosure axis. Keep the legacy profile value in the
// payload for backward-compatible launch/preset contracts, but never let a
// stored Quick/Balanced/Advanced preference change controls or overwrite edits.
function applyGuidedVisibility() {
  wizardState.profile = 'balanced';
  applyLlamaTierVisibility(dom.overlay, 'balanced');
  controlsForLoader('llama_cpp').forEach(control => {
    if (!control.disableOnQuick) return;
    const el = document.getElementById(control.id);
    if (el) el.disabled = false;
  });
  if (wizardState.engine.selected === 'rapid_mlx') {
    applyMlxTierVisibility(dom.overlay, 'balanced');
  }
}

// ── Model source visibility ───────────────────────────────────────────────────

export function updateModelInputVisibility() {
  const src = wizardState.model.source;
  dom.modelInputLocal?.classList.toggle('visible', src === 'local');
  dom.modelInputHf?.classList.toggle('visible', src === 'hf');
  dom.modelInputImport?.classList.toggle('visible', src === 'import');
}

let engineRecommendationSequence = 0;

export function classifyWizardArtifact(model = wizardState.model) {
  const path = (model.path || '').trim();
  const file = (model.hfFile || '').trim();
  const typedSource = model.rapidMlxSource || model.localMeta?.model_source || null;
  const typedKind = typedSource?.kind || model.localMeta?.source_kind || model.localMeta?.format || '';
  const hasGgufInventory = (model.quantFiles || []).some(item => /\.gguf$/i.test(item.path || item.name || ''));
  if (/\.gguf$/i.test(path) || /\.gguf$/i.test(file) || hasGgufInventory) return 'gguf';
  if (typedKind === 'authoritative_safetensors') return 'authoritative_safetensors';
  if (typedKind === 'alias') return 'rapid_mlx_alias';
  if (typedKind === 'hugging_face_repo') return 'rapid_mlx_hf_repository';
  if (typedKind === 'mlx_directory' || /mlx/i.test(typedKind)) return 'mlx_directory';
  return 'unknown';
}

async function refreshEngineAvailability() {
  try {
    const headers = window.authHeaders ? window.authHeaders() : {};
    const [platform, statusResponse] = await Promise.all([
      getPlatformInfo().catch(() => null),
      fetch('/api/rapid-mlx/runtime/status', { headers }).catch(() => null),
    ]);
    setWizardPlatformInfo(platform);
    const status = statusResponse?.ok ? await statusResponse.json().catch(() => ({})) : {};
    wizardState.engine.rapidMlxLocalAvailable = !!platform?.rapid_mlx_local_available;
    // Managed status describes only app-owned environments. Compatibility for
    // custom, Brew, Pip, and Pipx installs comes from the recommendation probe.
    wizardState.engine.rapidMlxRuntimeCompatible = !!status.runtime?.active;
  } finally {
    renderEngineSelection();
    refreshEngineRecommendation();
  }
}

export async function refreshEngineRecommendation() {
  let artifactKind = classifyWizardArtifact();
  const explicitRapidRepo = wizardState.engine.explicit
    && wizardState.engine.selected === 'rapid_mlx'
    && wizardState.model.source === 'hf'
    && wizardState.model.hfRepo
    && !wizardState.model.hfFile;
  const explicitRapidDirectory = wizardState.engine.explicit
    && wizardState.engine.selected === 'rapid_mlx'
    && wizardState.model.source === 'local'
    && wizardState.model.path;
  if (artifactKind === 'unknown' && explicitRapidRepo) artifactKind = 'rapid_mlx_hf_repository';
  if (artifactKind === 'unknown' && explicitRapidDirectory) artifactKind = 'mlx_directory';
  const sequence = ++engineRecommendationSequence;
  if (artifactKind === 'unknown') {
    wizardState.engine.recommendation = {
      recommended_backend: null,
      state: 'manual_selection',
      reason: 'Choose an engine manually until the model source is specific enough to recommend one.',
    };
    renderEngineSelection();
    refreshStepGuardrails();
    return;
  }
  wizardState.engine.recommendation = {
    recommended_backend: null,
    state: 'checking',
    reason: 'Checking model and runtime compatibility…',
  };
  renderEngineSelection();
  refreshStepGuardrails();
  try {
    const headers = window.authHeaders
      ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
      : { 'Content-Type': 'application/json' };
    const response = await fetch('/api/rapid-mlx/recommend', {
      method: 'POST',
      headers,
      body: JSON.stringify({ artifact_kind: artifactKind }),
    });
    const recommendation = response.ok ? await response.json() : null;
    if (sequence !== engineRecommendationSequence || !recommendation) return;
    wizardState.engine.recommendation = recommendation;
    if (artifactKind !== 'gguf') {
      if (recommendation.state === 'ready') wizardState.engine.rapidMlxRuntimeCompatible = true;
      if (recommendation.state === 'runtime_required') wizardState.engine.rapidMlxRuntimeCompatible = false;
      if (recommendation.state === 'platform_unavailable') wizardState.engine.rapidMlxLocalAvailable = false;
    }
    if (!wizardState.engine.explicit && recommendation.recommended_backend) {
      wizardState.engine.selected = recommendation.recommended_backend;
      _checkBinaryPrereq();
    }
  } catch {
    if (sequence !== engineRecommendationSequence) return;
    wizardState.engine.recommendation = null;
  }
  renderEngineSelection();
  refreshStepGuardrails();
}

export function selectWizardEngine(engine, explicit) {
  if (!['llama_cpp', 'rapid_mlx'].includes(engine)) return;
  wizardState.engine.selected = engine;
  if (explicit) wizardState.engine.explicit = true;
  if (engine !== 'rapid_mlx') {
    wizardState.model.rapidMlxProfile = null;
  }
  if (engine === 'rapid_mlx' && wizardState.model.source === 'import') {
    wizardState.model.source = 'local';
    wizardState.model.path = '';
    wizardState.model.localMeta = null;
    wizardState.model.delivery = 'local_file';
    if (dom.modelPathInput) dom.modelPathInput.value = '';
    dom.modelSourceCards?.forEach(card => {
      card.classList.toggle('selected', card.dataset.source === 'local');
    });
    updateModelInputVisibility();
  }
  renderEngineSelection();
  _applyScopeDefaultForEngine(engine);
  clearValidationError();
  refreshStepGuardrails();
  _checkBinaryPrereq();
  refreshEngineRecommendation();
  // Fetch live profile when switching to Rapid-MLX with a model selected
  if (engine === 'rapid_mlx') {
    const modelId = wizardState.model.hfRepo || wizardState.model.path || '';
    scheduleRapidMlxProfileFetch(modelId);
  }
}

function renderEngineSelection() {
  const selected = wizardState.engine.selected;
  if (selected === 'rapid_mlx' && dom.binaryPrereq) {
    dom.binaryPrereq.style.display = 'none';
  }
  dom.engineCards?.forEach(card => {
    const active = card.dataset.engine === selected;
    card.classList.toggle('selected', active);
    card.setAttribute('aria-checked', String(active));
    if (card.dataset.engine === 'rapid_mlx') {
      card.classList.toggle('is-unavailable', !wizardState.engine.rapidMlxLocalAvailable);
    }
  });
  dom.overlay?.classList.toggle('engine-rapid-mlx', selected === 'rapid_mlx');
  if (dom.rapidHardwarePanel) dom.rapidHardwarePanel.hidden = selected !== 'rapid_mlx';
  if (dom.rapidAdvancedFields) {
    dom.rapidAdvancedFields.style.display = selected === 'rapid_mlx' ? 'block' : 'none';
    if (selected === 'rapid_mlx') applyRapidMlxDefaults();
  }
  configureMlxWizardIA(dom.overlay, selected === 'rapid_mlx', wizardState.profile);
  if (selected === 'rapid_mlx') applyEffectiveLocks(dom.overlay);
  renderContextChipRow();
  // llama.cpp's advanced fields exist in the DOM regardless of engine
  // selection (hidden via .engine-rapid-mlx CSS when MLX is active), so this
  // is always enabled — mirrors configureMlxWizardIA's build-once guard.
  configureLlamaWizardIA(dom.overlay, true, wizardState.profile);
  if (wizardState.currentStep === 1) {
    _renderAllSettingsDrawer();
    renderProLayout(wizardState.viewMode);
  }

  // Expose IA data on window for Pro renderer and other consumers
  window.spawnWizardLlamaIA = { GROUPS: window._llamaGroups || [], SUPERSECTIONS: window._llamaSupersections || [] };
  window.spawnWizardMlxIA = { GROUPS: window._mlxGroups || [], SUPERSECTIONS: window._mlxSupersections || [] };
  // §2.7: MLX repos are already a specific quant, so swap the llama.cpp quant
  // advisor for the MLX sidebar body (repo-is / sibling-variants) instead.
  const mlxSidebarBody = document.getElementById('mlx-sidebar-body');
  if (mlxSidebarBody) mlxSidebarBody.style.display = selected === 'rapid_mlx' && wizardState.model.paramB > 0 ? '' : 'none';
  if (selected === 'rapid_mlx' && dom.quantAdvisor) dom.quantAdvisor.style.display = 'none';
  if (dom.rapidMlxAdvancedSection) {
    dom.rapidMlxAdvancedSection.style.display = selected === 'rapid_mlx' ? '' : 'none';
    if (selected === 'rapid_mlx') ensureEscapeHatchRendered();
  }
  const rapidBadge = dom.overlay?.querySelector('[data-engine-badge="rapid_mlx"]');
  if (rapidBadge) {
    rapidBadge.textContent = !wizardState.engine.rapidMlxLocalAvailable
      ? 'Local launch · Apple Silicon only'
      : wizardState.engine.rapidMlxRuntimeCompatible
        ? 'Runtime ready'
        : 'Runtime setup required';
  }
  const recommendation = wizardState.engine.recommendation;
  if (dom.engineReason) {
    const override = wizardState.engine.explicit && recommendation?.recommended_backend
      && recommendation.recommended_backend !== selected;
    dom.engineReason.textContent = override
      ? `${recommendation.reason} Your manual ${selected === 'rapid_mlx' ? 'Rapid-MLX' : 'llama.cpp'} choice is preserved.`
      : (recommendation?.reason || 'Select a model and we’ll explain the best engine.');
  }
  document.querySelector('.model-source-card[data-source="import"]')?.toggleAttribute('hidden', selected === 'rapid_mlx');
  const modelDescription = document.querySelector('#wizard-step-1 .wizard-main > div:not(.wizard-engine-section) .wizard-section-desc');
  if (modelDescription) {
    modelDescription.textContent = selected === 'rapid_mlx'
      ? 'Choose a validated MLX directory or a Rapid-MLX Hugging Face repository.'
      : 'Choose where your GGUF model comes from.';
  }
  const localName = document.querySelector('.model-source-card[data-source="local"] .model-source-name');
  const localDescription = document.querySelector('.model-source-card[data-source="local"] .model-source-desc');
  const hfDescription = document.querySelector('.model-source-card[data-source="hf"] .model-source-desc');
  if (localName) localName.textContent = selected === 'rapid_mlx' ? 'Select local MLX model' : 'Select local model';
  if (localDescription) {
    localDescription.textContent = selected === 'rapid_mlx'
      ? 'Browse to a validated MLX model directory.'
      : 'Browse your filesystem for an existing GGUF file.';
  }
  if (hfDescription) {
    hfDescription.textContent = selected === 'rapid_mlx'
      ? 'Enter a Rapid-MLX-compatible Hugging Face repository ID.'
      : 'Enter a HF repo ID and we’ll list available GGUF files.';
  }
  if (dom.modelPathInput) {
    dom.modelPathInput.placeholder = selected === 'rapid_mlx'
      ? 'MLX model directory (e.g. /models/my-mlx-model)'
      : 'Model path (e.g. /models/my-model.gguf)';
  }
  _updateSpawnBtnForPrereq();
}

async function ensureEscapeHatchRendered() {
  if (!dom.rapidMlxAdvancedFlags) return;
  // Cache descriptors on first fetch
  if (!_escapeHatchDescriptors) {
    try {
      const headers = window.authHeaders ? window.authHeaders() : {};
      const resp = await fetch('/api/rapid-mlx/escape-hatch-flags', { headers });
      if (resp.ok) {
        _escapeHatchDescriptors = await resp.json();
      }
    } catch {
      return;
    }
  }
  if (!_escapeHatchDescriptors || dom.rapidMlxAdvancedFlags.dataset.rendered === '1') return;
  dom.rapidMlxAdvancedFlags.dataset.rendered = '1';
  dom.rapidMlxAdvancedFlags.innerHTML = '';
  for (const d of _escapeHatchDescriptors) {
    const row = document.createElement('div');
    row.className = 'esc-hatch-row';
    row.dataset.flagName = d.flag;
    const tooltipText = d.default ? `${d.tooltip || ''} (default: ${d.default})` : (d.tooltip || '');
    const label = document.createElement('div');
    label.className = 'esc-hatch-label';
    label.title = tooltipText;
    const nameSpan = document.createElement('span');
    nameSpan.className = 'esc-hatch-name';
    nameSpan.textContent = d.description || '';
    const typeSpan = document.createElement('span');
    typeSpan.className = 'esc-hatch-type';
    typeSpan.textContent = d.value_type || '';
    label.appendChild(nameSpan);
    label.appendChild(typeSpan);
    if (d.default) {
      const defaultSpan = document.createElement('span');
      defaultSpan.className = 'esc-hatch-default';
      defaultSpan.textContent = `default: ${d.default}`;
      label.appendChild(defaultSpan);
    }
    const controls = document.createElement('div');
    controls.className = 'esc-hatch-controls';
    const existing = wizardState.hardware.escapeHatchFlags?.[d.flag];
    if (d.value_type === 'bool') {
      const toggle = document.createElement('label');
      toggle.className = 'esc-hatch-toggle';
      toggle.title = tooltipText;
      const chk = document.createElement('input');
      chk.type = 'checkbox';
      chk.checked = !!existing;
      chk.addEventListener('change', () => {
        wizardState.hardware.escapeHatchFlags[d.flag] = chk.checked;
      });
      toggle.appendChild(chk);
      toggle.appendChild(document.createTextNode(d.description));
      controls.appendChild(toggle);
    } else if (d.value_type === 'enum') {
      const sel = document.createElement('select');
      sel.className = 'esc-hatch-select';
      sel.title = tooltipText;
      (d.enum_options || []).forEach(opt => {
        const o = document.createElement('option');
        o.value = opt;
        o.textContent = opt;
        if (existing === opt) o.selected = true;
        sel.appendChild(o);
      });
      sel.addEventListener('change', () => {
        wizardState.hardware.escapeHatchFlags[d.flag] = sel.value;
      });
      controls.appendChild(sel);
    } else {
      const inp = document.createElement('input');
      inp.type = d.value_type === 'float' ? 'number' : 'number';
      inp.className = 'esc-hatch-input';
      inp.title = tooltipText;
      inp.step = d.value_type === 'float' ? '0.01' : '1';
      if (d.default) {
        const numericDefault = parseFloat(d.default);
        if (Number.isFinite(numericDefault)) inp.placeholder = String(numericDefault);
      }
      if (existing !== undefined && existing !== null) inp.value = existing;
      inp.addEventListener('input', () => {
        const val = d.value_type === 'float' ? parseFloat(inp.value) : parseInt(inp.value, 10);
        wizardState.hardware.escapeHatchFlags[d.flag] = Number.isFinite(val) ? val : null;
      });
      controls.appendChild(inp);
    }
    row.appendChild(label);
    row.appendChild(controls);
    dom.rapidMlxAdvancedFlags.appendChild(row);
  }
}

export function updateSelectedModelDisplay() {
  const { source, path, hfRepo, hfFile } = wizardState.model;
  let name = '', meta = '';
  if (source === 'hf' && hfRepo) {
    name = hfFile ? hfFile.split('/').pop() : hfRepo;
    meta = hfFile ? `${hfRepo}  ·  HuggingFace` : 'HuggingFace repo';
  } else if (path) {
    name = path.split(/[\\/]/).pop() || path;
    meta = path;
  }
  if (!name) {
    if (dom.selectedModel?.classList) dom.selectedModel.classList.remove('visible');
    if (dom.selectedModelArch) dom.selectedModelArch.innerHTML = '';
    return;
  }
  dom.selectedModel?.classList.add('visible');
  if (dom.selectedModelName) dom.selectedModelName.textContent = name;
  if (dom.selectedModelMeta) dom.selectedModelMeta.textContent = meta;

  // Build architecture label if we have metadata (from introspection or presets)
  updateSelectedModelArchLabel();
}

function updateSelectedModelArchLabel() {
  const container = dom.selectedModelArch;
  if (!container) return;
  container.innerHTML = '';

  // Derive MoE state from real signals: an expert count (from introspection or the
  // filename guess) or the pending-MoE flag set off the "A<n>B" suffix. The legacy
  // `arch.isMoe` flag was never populated, so relying on it made every model fall
  // through to "dense".
  const isMoe =
      (wizardState.arch.nExperts || 0) > 0 ||
      wizardState.arch.isMoe === true ||
      wizardState.arch._isMoePending === true;
  // Hybrid attention is flagged by the GGUF arch name / filename (the previous
  // `arch.name` field was never set, so this never matched before).
  const archName = `${wizardState.arch.ggufArch || ''} ${wizardState.model.path || ''}`;
  const isHybrid = /hybrid|deltanet/i.test(archName);
  const totalB = wizardState.model.paramB || null;

  // Active-parameter budget: prefer the value parsed from the filename / introspection;
  // otherwise estimate from the expert ratio (active ≈ total / (1 + experts/used)),
  // mirroring the backend's simple_moe_active fallback.
  let activeB = wizardState.model.activeParamsB || null;
  if (isMoe && !activeB && totalB &&
      wizardState.arch.nExperts > 0 && wizardState.arch.nExpertsUsed > 0) {
    const ratio = wizardState.arch.nExperts / wizardState.arch.nExpertsUsed;
    activeB = totalB / (1 + ratio);
  }

  // Prefer the backend-derived architecture_kind (authoritative, from introspection);
  // fall back to the local derivation before introspection completes.
  const archKind = wizardState.arch.archKind ||
      (isMoe ? (isHybrid ? 'hybrid_moe' : 'moe')
             : (wizardState.arch.nLayers > 0 ? 'dense' : null));

  // Build a pseudo-preset from wizard state so we can reuse buildArchitectureLabel.
  const pseudoPreset = {
    architecture_kind: archKind,
    expert_count: wizardState.arch.nExperts || null,
    expert_used_count: wizardState.arch.nExpertsUsed || null,
    active_params_b: activeB,
    param_count: totalB ? totalB * 1e9 : null,
  };

  const arch = buildArchitectureLabel(pseudoPreset, { paramB: wizardState.model.paramB });
  if (!arch) return;

  const label = document.createElement('div');
  label.className = 'selected-model-arch-label';
  label.textContent = arch.display;
  label.title = arch.tooltip;
  container.appendChild(label);

  // Brief architecture note for educational clarity, including which GPU-offload knob
  // the model's layer count applies to (--n-cpu-moe for MoE, --gpu-layers/-ngl for dense).
  const note = document.createElement('div');
  note.className = 'selected-model-arch-hint';
  const nLayers = wizardState.arch.nLayers || 0;
  const layerStr = nLayers > 0 ? ` This model has ${nLayers} layers` : '';
  if (archKind === 'moe' || archKind === 'hybrid_moe') {
    // Real measured routed-expert bytes per MoE layer (the VRAM --n-cpu-moe frees).
    const perExpert = wizardState.arch.expertBytesPerLayer || 0;
    const perExpertStr = perExpert > 0 ? ` (~${formatBytes(perExpert)} freed per offloaded layer)` : '';
    note.textContent = 'MoE / Hybrid MoE: only a subset of parameters active per token; often more efficient.' +
        (layerStr ? layerStr + ' — set --n-cpu-moe between 0 and ' + nLayers + ' to offload expert layers to CPU/RAM' + perExpertStr + '.' : '');
  } else if (archKind === 'dense') {
    const perLayer = wizardState.arch.bytesPerLayer || 0;
    const perLayerStr = perLayer > 0 ? ` (~${formatBytes(perLayer)} of VRAM each)` : '';
    note.textContent = 'Dense: all parameters used each token.' +
        (layerStr ? layerStr + perLayerStr + ' — set --gpu-layers (-ngl) between 0 and ' + nLayers + ' to offload layers to the GPU.' : '');
  }
  container.appendChild(note);

  if (wizardState.arch.metadataStatus !== 'unknown') {
    const evidence = document.createElement('div');
    evidence.className = 'selected-model-arch-hint';
    const status = wizardState.arch.metadataStatus === 'resolved' ? 'Model-native' : 'Unknown / degraded';
    evidence.textContent = `Evidence: ${status} · ${wizardState.arch.metadataReason || 'safe defaults retained'}`;
    container.appendChild(evidence);
  }
}

// ── Model path changed ────────────────────────────────────────────────────────

export function onModelPathChanged() {
  updateSelectedModelDisplay();
  clearValidationError();
  refreshEngineRecommendation();

  const artifactKind = classifyWizardArtifact();
  const rapidLocalSource = wizardState.engine.selected === 'rapid_mlx'
    && (artifactKind === 'mlx_directory'
      || artifactKind === 'authoritative_safetensors'
      || (wizardState.engine.explicit && artifactKind === 'unknown'));
  if (rapidLocalSource) {
    // Rapid-MLX local/import sources skip the GGUF-oriented arch/quant pipeline
    // below, but the shared chat-template panel (#chat-template-section) still
    // applies — Rapid-MLX consumes chat_template_file via a generated overlay
    // model directory (see build_launch_argv / create_template_overlay in
    // src/inference/rapid_mlx/mod.rs), same field name as llama.cpp.
    autoInstallChatTemplate();
    refreshRapidMlxSidecars();
    refreshStepGuardrails();
    return;
  }

  const path = wizardState.model.path;
  if (path) {
    // Param count, MoE-ness, and MTP depth are never inferred from the filename (Phase
    // 10e: introspection-only for all model properties) — they stay unset until real GGUF
    // introspection below resolves them. UI reading these fields must treat them as
    // "pending" rather than substituting a guess.
    tryIntrospectModel(path);
  }

  // Update quant advisor if we have param count
  if (wizardState.model.paramB > 0) triggerQuantAdvisor();
  scheduleVramUpdate();
  autoInstallChatTemplate();
  refreshStepGuardrails();
}

// Real GGUF-header introspection for a not-yet-downloaded HF file (Phase 10e:
// introspection-only, never a filename/repo-name guess). Reuses /api/model-defaults'
// HF-aware branch, which range-fetches the real GGUF header server-side. Merges arch
// state the same way local doIntrospect() does; on failure (offline/gated/no range
// support) leaves fields unset rather than falling back to a guess.
export async function introspectHfFileMetadata(repoId, fname, sizeBytes) {
  if (!repoId || !fname) return false;
  try {
    const headers = window.authHeaders
      ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
      : { 'Content-Type': 'application/json' };
    const resp = await fetch('/api/model-defaults', {
      method: 'POST',
      headers,
      body: JSON.stringify({
        model_name_or_repo: repoId,
        size_bytes: sizeBytes || 0,
        tags: [],
        gguf_arch: '',
        arch_family: '',
        backend: wizardState.engine.selected || 'llama_cpp',
        hf_repo_id: repoId,
        hf_file_path: fname,
      }),
    });
    if (!resp.ok) {
      wizardState.arch.metadataStatus = 'degraded';
      wizardState.arch.metadataReason = `GGUF header request failed (${resp.status})`;
      return false;
    }
    const data = await resp.json();
    const m = data.introspected;
    if (!m) {
      wizardState.arch.metadataStatus = 'degraded';
      wizardState.arch.metadataReason = data.error || 'GGUF header metadata unavailable';
      return false;
    }

    if (m.n_ctx_train) wizardState.model.nCtxTrain = m.n_ctx_train;
    if (m.gguf_arch) {
      wizardState.arch.ggufArch = m.gguf_arch;
      if (!wizardState.model.family) {
        wizardState.model.family = String(m.gguf_arch).toLowerCase().replace(/_/g, '.');
      }
    }
    if (m.total_params_b != null && m.total_params_b > 0) wizardState.model.paramB = m.total_params_b;
    if (m.active_params_b != null) wizardState.model.activeParamsB = m.active_params_b;
    if (m.n_experts) wizardState.arch.nExperts = m.n_experts;
    if (m.n_experts_used) wizardState.arch.nExpertsUsed = m.n_experts_used;
    if (m.mtp_depth) wizardState.arch.mtpDepth = m.mtp_depth;
    if (m.n_attn_layers) wizardState.arch.nAttnLayers = m.n_attn_layers;
    if (m.linear_attn_state_bytes) wizardState.arch.linearAttnStateBytes = m.linear_attn_state_bytes;
    if (m.n_global_attn_layers) wizardState.arch.nGlobalAttnLayers = m.n_global_attn_layers;
    if (m.local_kv_heads) wizardState.arch.localKvHeads = m.local_kv_heads;
    if (m.global_head_dim) wizardState.arch.globalHeadDim = m.global_head_dim;
    if (m.local_head_dim) wizardState.arch.localHeadDim = m.local_head_dim;
    if (m.sliding_window) {
      wizardState.arch.slidingWindow = m.sliding_window;
      wizardState.arch.localAttnWindow = m.sliding_window;
    }
    if (m.mmproj_required != null) wizardState.arch.mmprojRequired = !!m.mmproj_required;
    wizardState.arch.metadataStatus = 'resolved';
    wizardState.arch.metadataReason = 'Progressive GGUF header';

    // The pill row / effective defaults were already rendered off the earlier,
    // arch-less call — refresh now that a real gguf_arch is known.
    if (m.gguf_arch) _fetchAndApplyModelSamplingDefaults();
    return true;
  } catch (error) {
    wizardState.arch.metadataStatus = 'degraded';
    wizardState.arch.metadataReason = error?.message || 'GGUF header request failed';
    return false;
  }
}

// ── Introspection ─────────────────────────────────────────────────────────────

let introspectDebounce = null;

function tryIntrospectModel(path) {
  // Allow both .gguf files and Ollama blobs (sha256-* content-addressed files)
  const lower = path.toLowerCase();
  if (!lower.endsWith('.gguf') && !lower.includes('/blobs/sha256-') && !lower.includes('\\blobs\\sha256-')) return;
  if (introspectDebounce) clearTimeout(introspectDebounce);
  introspectDebounce = setTimeout(() => doIntrospect(path), 1200);
}

export async function doIntrospect(path) {
  try {
    const headers = window.authHeaders ? { ...window.authHeaders(), 'Content-Type': 'application/json' } : { 'Content-Type': 'application/json' };
    const resp = await fetch('/api/model/introspect', { method: 'POST', headers, body: JSON.stringify({ model_path: path }) });
    if (!resp.ok) {
      wizardState.arch.metadataStatus = 'degraded';
      wizardState.arch.metadataReason = `GGUF metadata request failed (${resp.status})`;
      refreshGuidedCapabilityCards();
      return false;
    }
    const data = await resp.json();
    if (!data.ok || !data.metadata) {
      wizardState.arch.metadataStatus = 'degraded';
      wizardState.arch.metadataReason = data.error || 'GGUF metadata unavailable';
      refreshGuidedCapabilityCards();
      return false;
    }

    const m = data.metadata;

    // Use the actual file size from disk — exact, no estimation needed.
    if (data.file_size_bytes > 0) {
      wizardState.model.modelBytes = data.file_size_bytes;
    }

    // Merge arch state from GGUF metadata
    if (m.n_layers)       wizardState.arch.nLayers      = m.n_layers;
    if (m.n_ctx_train)    wizardState.model.nCtxTrain   = m.n_ctx_train;
    if (m.n_kv_heads)     wizardState.arch.nKvHeads     = m.n_kv_heads;
    if (m.head_dim)       wizardState.arch.headDim       = m.head_dim;
    if (m.n_experts)      wizardState.arch.nExperts      = m.n_experts;
    if (m.n_experts_used) wizardState.arch.nExpertsUsed = m.n_experts_used;
    if (m.mtp_depth)      wizardState.arch.mtpDepth      = m.mtp_depth;
    if (m.n_attn_layers)  wizardState.arch.nAttnLayers  = m.n_attn_layers;
    if (m.linear_attn_state_bytes) wizardState.arch.linearAttnStateBytes = m.linear_attn_state_bytes;
    if (m.n_global_attn_layers) wizardState.arch.nGlobalAttnLayers = m.n_global_attn_layers;
    if (m.local_kv_heads)  wizardState.arch.localKvHeads = m.local_kv_heads;
    if (m.global_head_dim) wizardState.arch.globalHeadDim = m.global_head_dim;
    if (m.local_head_dim)  wizardState.arch.localHeadDim = m.local_head_dim;
    if (m.sliding_window) {
      wizardState.arch.slidingWindow = m.sliding_window;
      wizardState.arch.localAttnWindow = m.sliding_window;
    }
    if (m.mmproj_required != null) wizardState.arch.mmprojRequired = !!m.mmproj_required;
    if (m.gguf_arch) {
      wizardState.arch.ggufArch = m.gguf_arch;
      if (!wizardState.model.family) {
        wizardState.model.family = String(m.gguf_arch).toLowerCase().replace(/_/g, '.');
      }
    }
    // Backend-derived architecture label + active-param estimate (authoritative —
    // same computation the preset editor uses, so the wizard label matches it).
    if (m.architecture_kind) wizardState.arch.archKind = m.architecture_kind;
    if (m.active_params_b != null) wizardState.model.activeParamsB = m.active_params_b;
    // Exact per-layer byte sizes measured from the GGUF tensor directory (real data).
    if (m.bytes_per_layer != null) wizardState.arch.bytesPerLayer = m.bytes_per_layer;
    if (m.expert_bytes_per_layer != null) wizardState.arch.expertBytesPerLayer = m.expert_bytes_per_layer;
    wizardState.arch.metadataStatus = 'resolved';
    wizardState.arch.metadataReason = 'GGUF header metadata';
    refreshGuidedCapabilityCards();
    if (wizardState.currentStep === 1) renderMtpSection();

    // Re-fetch sampling defaults now that gguf_arch is known — the earlier call
    // (on hardware step entry) ran before introspection completed and sent an empty
    // gguf_arch, which causes distills/finetunes to get generic fallback presets.
    if (m.gguf_arch) _fetchAndApplyModelSamplingDefaults();

    // Restore HF origin from persisted model tag so quant-swap can skip search.
    if (!wizardState.model.originRepo &&
        (wizardState.model.source === 'local' || wizardState.model.source === 'import')) {
      try {
        const tagsResp = await fetch('/api/models/tags', { headers });
        if (tagsResp.ok) {
          const td = await tagsResp.json().catch(() => ({}));
          const modelTags = td.tags?.[path] || [];
          const originTag = modelTags.find(t => t.startsWith('hf_origin:'));
          if (originTag) {
            wizardState.model.originRepo = originTag.slice('hf_origin:'.length);
            // Also restore family and card URL
            const familyTag = modelTags.find(t => t.startsWith('family:'));
            if (familyTag) wizardState.model.family = familyTag.slice('family:'.length);
            if (wizardState.model.originRepo) {
              wizardState.model.cardUrl = `https://huggingface.co/${wizardState.model.originRepo}`;
            }
          }
        }
      } catch { /* non-fatal */ }
    }

    // Auto-resolve HF origin for local models that lack a persisted origin tag.
    // The resolve endpoint already includes family detection (backend fetches
    // HF card tags in the same pass), so no separate _detectFamilyForOrigin needed.
    // Returns a promise so autoInstallChatTemplate can await the resolution.
    if (!wizardState.model.originRepo &&
        (wizardState.model.source === 'local' || wizardState.model.source === 'import')) {
      // Defer to avoid blocking the introspect flow; 200ms delay is imperceptible.
      setOriginResolverPromise((async () => {
        await new Promise(r => setTimeout(r, 200));
        return _autoResolveHfOrigin();
      })());
    }

    // Scan directory for companion mmproj file (local models only)
    if (wizardState.model.source === 'local' || wizardState.model.source === 'import') {
      const dir = path.replace(/[/\\][^/\\]+$/, '');
      try {
        const browseResp = await fetch(
          `/api/browse?path=${encodeURIComponent(dir)}&filter=gguf`,
          { headers }
        );
        if (browseResp.ok) {
          const bd = await browseResp.json();
          const found = (bd.entries || []).filter(
            e => !e.is_dir && e.name.toLowerCase().includes('mmproj')
          );
          if (found.length) {
            wizardState.model.mmprojFiles = found.map(e => ({
              path: e.path, name: e.name, size: e.size || 0, is_mmproj: true,
            }));
            // Pick best-matching mmproj by name proximity; avoids auto-selecting
            // a companion file from a different model in the same directory.
            const modelFilename = path.split(/[\\/]/).pop() || '';
            const bestMmproj = _bestMmprojForModel(modelFilename, wizardState.model.mmprojFiles);
            if (bestMmproj) {
              wizardState.model.mmprojPath = bestMmproj.path;
              wizardState.model.mmprojHfFile = bestMmproj.path;
              wizardState.arch.mmprojBytes = bestMmproj.size || 0;
            }
            // Re-render the mmproj section if hardware step is active
            if (wizardState.currentStep === 1) renderMmprojSection();
            scheduleVramUpdate();
          }
        }
      } catch { /* browse may be rate-limited; skip silently */ }
    }

    // Scan directory for MTP draft/draft model files (local models)
    if (wizardState.model.source === 'local' || wizardState.model.source === 'import') {
      const dir = path.replace(/[/\\][^/\\]+$/, '');
      try {
        const browseResp = await fetch(
          `/api/browse?path=${encodeURIComponent(dir)}&filter=gguf`,
          { headers }
        );
        if (browseResp.ok) {
          const bd = await browseResp.json();
          const MTP_HEAD_MAX_BYTES = 3_000_000_000; // >3 GB = full model, not an MTP head
          const drafts = (bd.entries || []).filter(e => {
            if (e.is_dir) return false;
            const n = e.name.toLowerCase();
            // mmproj files are image projection layers, never MTP heads.
            if (n.includes('mmproj')) return false;
            // Files larger than 3 GB are full models with "MTP" in their name — not heads.
            if (e.size > 0 && e.size > MTP_HEAD_MAX_BYTES) return false;
            const nameMatch = n.includes('assistant')
               || n.includes('mtp-draft')
               || n.includes('draft-model')
               || n.includes('mtp_small')
               || n.includes('mtp-heads')
               || n.startsWith('mtp-')
               || n.includes('-mtp.')
               || /[-_]mtp[-_]/.test(n);
            if (!nameMatch) return false;
            return true;
          });
          if (drafts.length) {
            wizardState.model.draftCandidates = drafts.map(e => ({
              path: e.path,
              name: e.name,
              size: e.size || 0,
              is_draft: true,
            }));
            const modelFilename = path.split(/[\\/]/).pop() || '';
            const bestDraft = _bestDraftForModel(modelFilename, wizardState.model.draftCandidates);
            if (bestDraft) {
              wizardState.model.selectedDraftPath = bestDraft.path;
            }
            // Re-render MTP section to include draft selector
            if (wizardState.currentStep === 1) renderMtpSection();
            scheduleVramUpdate();
          }
        }
      } catch { /* non-fatal */ }
    }

    // Check for MTP draft availability on Gemma4 models with no local draft
    await _checkGemma4MtpDraft(path);

    // Update MoE slider max — n_cpu_moe counts layers, so the max is the layer count
    if (wizardState.arch.nExperts > 0 && dom.moeOffloadSlider) {
      dom.moeOffloadSlider.max = wizardState.arch.nLayers || wizardState.arch.nExperts;
    }

    // Bound the manual --gpu-layers (-ngl) input to the layer count and show it, so the
    // user knows e.g. a 70B model has 69 layers and can put 65 on GPU, 4 on CPU/RAM.
    const nLayers = wizardState.arch.nLayers || 0;
    if (dom.gpuLayersManualInput && nLayers > 0) {
      dom.gpuLayersManualInput.max = nLayers;
      dom.gpuLayersManualInput.placeholder = `0–${nLayers}`;
    }
    const nglHint = document.getElementById('spawn-gpu-layers-manual-hint');
    if (nglHint) {
      if (nLayers > 0) {
        // Real per-layer VRAM from the GGUF tensor directory when available.
        const perLayer = wizardState.arch.bytesPerLayer || 0;
        const perLayerStr = perLayer > 0 ? ` (~${formatBytes(perLayer)} of VRAM each)` : '';
        nglHint.textContent = `This model has ${nLayers} layers${perLayerStr}. Enter 0–${nLayers}: layers above your value stay on CPU/RAM (e.g. ${Math.max(0, nLayers - 4)} keeps 4 layers off the GPU).`;
        nglHint.style.display = '';
      } else {
        nglHint.style.display = 'none';
      }
    }

    // GGUF reports n_ctx_train; local MLX reports config.max_position_embeddings.
    // Both are native supported ceilings, not an invitation to silently apply
    // RoPE/YaRN extension parameters.
    const nativeContextLimit = Number(m.n_ctx_train || m.config?.max_position_embeddings || 0);
    if (nativeContextLimit > 0) {
      wizardState.model.nCtxTrain = nativeContextLimit;
      updateCtxTrainWarning();
      updateCtxModelMaxHint();
      updateCtxQuickPickActive();
    }

    // Populate localMeta for the hint card — local browse never pre-populates it
    if (wizardState.model.source === 'local' && !wizardState.model.localMeta) {
      const fname = path.split(/[\\/]/).pop() || path;
      wizardState.model.localMeta = {
        path,
        filename: fname,
        size_display: wizardState.model.modelBytes ? formatBytes(wizardState.model.modelBytes) : '',
        quant_type: guessQuantFromName(fname),
        param_b: wizardState.model.paramB || null,
      };
      renderLocalModelHint(); // also calls _refreshHfOriginSection via its tail
    }

    scheduleVramUpdate();
    if (wizardState.model.paramB > 0) triggerQuantAdvisor();
    return true;
  } catch (error) {
    wizardState.arch.metadataStatus = 'degraded';
    wizardState.arch.metadataReason = error?.message || 'GGUF introspection failed';
    refreshGuidedCapabilityCards();
    return false;
  }
}

// ── GPU VRAM query ────────────────────────────────────────────────────────────

let cachedVram = 0;
export let cachedRamTotal = 0;
export let cachedRamUsed  = 0;
export let cachedMetalGpuLimitMb = 0; // 0 = system default; >0 = custom iogpu.wired_limit_mb

export async function ensureGpuVramFetched() {
  const tasks = [];
  if (!cachedVram) tasks.push(fetchGpuVram());
  // Live MemoryAvailabilitySnapshot (D30/A58 single source of truth) — needed as
  // early as the model-select quant advisor, not just the hardware step, so the
  // displayed "available" figure reflects current memory pressure rather than
  // only the theoretical Metal cap.
  if (!cachedMemorySnapshot) tasks.push(fetchMemoryAvailability());
  if (tasks.length) await Promise.all(tasks);
}

// ── Unified memory helpers (Apple Silicon) ────────────────────────────────────

// True when the platform backend is Metal (Apple Silicon unified memory).
// On unified memory, GPU and system RAM are the same pool; VRAM == RAM.
export function isUnifiedMemory() {
  return _platformInfo?.auto_backend === 'metal';
}

// On unified memory, the Metal cap IS the budget. macOS compresses other processes'
// pages to give Metal up to cap bytes — that's the purpose of the cap.
// The suggested cap (total − 8 GB) already reserves OS headroom, so we use the
// cap directly rather than constraining by current free RAM.
// On discrete GPU, the cached VRAM figure is the dedicated VRAM pool.
// macOS Metal GPU wired memory cap (default, without sysctl tweak):
//   ≤ 36 GB RAM → ~66% (2/3)  e.g. 24 GB → 16 GB
//   > 36 GB RAM → ~75% (3/4)  e.g. 64 GB → 48 GB, 128 GB → 96 GB
// If the user has applied iogpu.wired_limit_mb, that value overrides the default.
export function metalCap(ramTotal) {
  if (cachedMetalGpuLimitMb > 0) {
    return cachedMetalGpuLimitMb * 1024 * 1024; // MiB → bytes
  }
  const fraction = ramTotal <= 36 * 1024 ** 3 ? 2 / 3 : 3 / 4;
  return Math.floor(ramTotal * fraction);
}

// Suggested iogpu.wired_limit_mb value for the user's system.
// Leaves 8 GB for macOS (safe minimum per llama.cpp community docs).
// Returns 0 if the suggestion would not improve over the current/default cap.
export function suggestedMetalLimitMb(ramTotal) {
  const currentCap = metalCap(ramTotal);
  const suggested = Math.floor(ramTotal / (1024 * 1024)) - 8192; // total_MB - 8 GB
  return suggested > Math.floor(currentCap / (1024 * 1024)) ? suggested : 0;
}

// Metal driver initialization reserve (~512 MB).
// sysinfo::used_memory() on macOS already includes wire_count (kernel wired pages),
// so freeRam = total - used_memory already excludes wired kernel memory.
// The Metal cap handles the macro OS headroom (25–33% of RAM).
// This small reserve covers Metal driver startup allocations not yet reflected in
// the pre-launch snapshot (argument tables, shader cache, command buffer pools).
// Inference-time burst compute buffers are handled by computeHeadroom() separately.
const APPLE_OS_RESERVE_BYTES = 512 * 1024 * 1024;

// Discrete GPU headroom: 5% but capped at 1.5 GB — driver overhead is flat, not percentage-based
const DISCRETE_MAX_HEADROOM_BYTES = 1.5 * 1024 ** 3;

function computeHeadroom(availVram) {
  if (isUnifiedMemory()) {
    if (!availVram) return 0.10;
    // 10% base capped at 2 GB absolute — Metal burst compute buffers are flat, not percentage-based
    return Math.min(0.10, (2 * 1024 ** 3) / availVram);
  }
  if (!availVram) return 0.05;
  return Math.min(0.05, DISCRETE_MAX_HEADROOM_BYTES / availVram);
}

export function effectiveAvailBytes() {
  // Prefer the live MemoryAvailabilitySnapshot when available (Phase 5b Part A).
  // This ensures Rapid Wizard uses current_safe_availability from the single source
  // of truth, never stale llama/HF caches or theoretical caps.
  if (cachedMemorySnapshot && cachedMemorySnapshot.current_safe_availability_bytes > 0) {
    return cachedMemorySnapshot.current_safe_availability_bytes;
  }
  if (isUnifiedMemory() && cachedRamTotal > 0) {
    const cap = metalCap(cachedRamTotal);
    // Use the Metal cap as the budget. The cap was configured to leave OS headroom
    // (default: 75% of RAM; suggested sysctl: total − 8 GB). macOS compresses
    // other processes' pages as needed to give Metal up to cap bytes — the snapshot
    // free-RAM figure understates what's actually available on unified memory.
    return Math.max(0, Math.min(cap, cachedRamTotal) - APPLE_OS_RESERVE_BYTES);
  }
  return cachedVram || wizardState.vram.available;
}

async function fetchSystemRam() {
  try {
    const headers = window.authHeaders ? window.authHeaders() : {};
    const resp = await fetch('/metrics/system', { headers });
    if (!resp.ok) return;
    const d = await resp.json();
    cachedRamTotal = (d.ram_total_gb || 0) * 1024 * 1024 * 1024;
    cachedRamUsed  = (d.ram_used_gb  || 0) * 1024 * 1024 * 1024;
  } catch {}
}

export async function fetchGpuVram() {
  try {
    const headers = window.authHeaders ? window.authHeaders() : {};
    const resp = await fetch('/metrics/gpu', { headers });
    if (!resp.ok) return;
    const data = await resp.json();
    // /metrics/gpu returns BTreeMap<String, GpuMetrics> (object keyed by GPU name on Mac/Linux)
    // or an array or { gpus: [...] } depending on endpoint version
    let totalVram = 0;
    let usedVram = 0;
    const gpus = Array.isArray(data) ? data : (data.gpus ? data.gpus : Object.values(data));
    for (const g of gpus) {
      // Rust GpuMetrics struct uses `vram_total` field (value in MB); also check legacy names
      const t = g.vram_total_mb || g.total_mb || g.total_memory_mb || g.vram_total || 0;
      const u = g.vram_used_mb || g.used_mb || g.vram_used || 0;
      totalVram += t * 1024 * 1024;
      usedVram += u * 1024 * 1024;
      // Capture Metal GPU wired limit from Apple backend (0 = system default)
      if (g.metal_gpu_limit_mb !== undefined && g.metal_gpu_limit_mb !== null) {
        cachedMetalGpuLimitMb = g.metal_gpu_limit_mb;
      }
    }
    if (totalVram > 0) {
      cachedVram = isUnifiedMemory() ? totalVram : Math.max(0, totalVram - usedVram);
      wizardState.vram.available = cachedVram;
    }
  } catch {}
}

export async function fetchMetalGpuLimit() {
  if (!isUnifiedMemory()) return;
  try {
    const headers = window.authHeaders ? window.authHeaders() : {};
    const resp = await fetch('/api/system/metal-gpu-limit', { headers });
    if (!resp.ok) return;
    const data = await resp.json().catch(() => ({}));
    if (!data.ok) return;
    cachedMetalGpuLimitMb = Number(data.limit_mb || 0);
  } catch {}
}

// ── MemoryAvailabilitySnapshot fetch (Phase 5b Part A) ────────────────────────
//
// Single source of truth for memory availability. Never reuses llama/HF cached
// values. Used by Rapid Wizard and all fit surfaces per D30/A58.

let cachedMemorySnapshot = null;

async function fetchMemoryAvailability() {
  try {
    const headers = window.authHeaders ? window.authHeaders() : {};
    const resp = await fetch('/api/memory-availability', { headers });
    if (!resp.ok) return null;
    const data = await resp.json().catch(() => ({}));
    if (!data.ok || !data.snapshot) return null;
    cachedMemorySnapshot = data.snapshot;
    // Update cached values from the snapshot for downstream consumers
    if (cachedMemorySnapshot.total_unified_bytes > 0) {
      cachedRamTotal = cachedMemorySnapshot.total_unified_bytes;
      cachedRamUsed = cachedMemorySnapshot.total_unified_bytes
        - cachedMemorySnapshot.free_bytes
        - (cachedMemorySnapshot.speculative_bytes || 0);
    }
    if (cachedMemorySnapshot.metal_working_set_bytes > 0) {
      cachedVram = cachedMemorySnapshot.configured_ceiling_bytes;
      wizardState.vram.available = cachedMemorySnapshot.current_safe_availability_bytes;
      wizardState.vram.availableRam = cachedMemorySnapshot.current_safe_availability_bytes;
    }
    return cachedMemorySnapshot;
  } catch {
    return null;
  }
}

export function _deriveMmprojSaveName(modelHfPath, mmprojHfPath) {
  const modelBase = (modelHfPath.split('/').pop() || modelHfPath).replace(/\.gguf$/i, '');
  const stem = modelBase.replace(/-?(Q\d[\w.]*|IQ\d[\w.]*|BF16|F16)$/i, '');
  const mmprojBase = mmprojHfPath.split('/').pop() || mmprojHfPath;
  return `${stem}-${mmprojBase}`;
}

// ── Hardware change ───────────────────────────────────────────────────────────

let vramDebounce = null;

function onHardwareChange(e) {
  // Only save scroll position for toggle checkboxes that cause layout shifts
  // (fitTargetWrap appearing/disappearing). Regular number inputs don't shift
  // layout, so the deferred restore's nested rAF can conflict with the
  // browser's own Tab/focus scroll, making content appear to disappear.
  const isToggle = e && (
    e.target === dom.fitEnableSelect ||
    e.target === dom.mlockCheck
  );

  // If any non-toggle field fires after a toggle, cancel the pending scroll
  // restore so we don't undo the browser's natural scroll-into-view when the
  // user tabs away from their input.
  if (!isToggle && pendingHardwareScrollRestore) {
    pendingHardwareScrollRestore = null;
  }

  if (isToggle && wizardState.currentStep === 1 && !pendingHardwareScrollReset) {
    const main = document.querySelector('#wizard-step-1 .wizard-main');
    const sidebar = document.querySelector('#wizard-step-1 .hw-vram-sidebar');
    pendingHardwareScrollRestore = {
      main: main?.scrollTop ?? 0,
      sidebar: sidebar?.scrollTop ?? 0,
    };
  }
  readHardwareState();
  scheduleVramUpdate();
  refreshStepGuardrails();
  renderContextChipRow();
}

function readHardwareState() {
  const h = wizardState.hardware;
  h.gpuLayers = dom.gpuLayersSelect?.value ?? 'auto';
  if (dom.gpuLayersManualInput) {
    const v = dom.gpuLayersManualInput.value;
    h.gpuLayersManual = v !== '' ? Number(v) : null;
  }
  if (dom.contextSizeInput) { const v = Number(dom.contextSizeInput.value); h.contextSize = v > 0 ? v : 8192; }
  if (dom.batchSizeInput)   { const v = Number(dom.batchSizeInput.value);   h.batchSize   = v > 0 ? v : 2048; }
  if (dom.ubatchSizeInput)  { const v = Number(dom.ubatchSizeInput.value);  h.ubatchSize  = v > 0 ? v : 2048; }
  if (dom.parallelSlotsInput) { const v = Number(dom.parallelSlotsInput.value); h.parallelSlots = v > 0 ? v : 1; }
  if (dom.cacheTypeKSelect) h.cacheTypeK = dom.cacheTypeKSelect.value || 'q8_0';
  if (dom.cacheTypeVSelect) h.cacheTypeV = dom.cacheTypeVSelect.value || 'q8_0';
  if (dom.specDraftTypeKSelect) h.specDraftTypeK = dom.specDraftTypeKSelect.value || '';
  if (dom.specDraftTypeVSelect) h.specDraftTypeV = dom.specDraftTypeVSelect.value || '';
  if (dom.nCpuMoeInput) { const v = dom.nCpuMoeInput.value; h.nCpuMoe = v !== '' ? Number(v) : 0; }
  if (dom.tensorSplitInput) h.tensorSplit = dom.tensorSplitInput.value.trim() || '';
  if (dom.fitEnableSelect) {
    const enabled = dom.fitEnableSelect.value === 'true';
    h.fitEnabled = dom.fitEnableSelect.value === '' ? null : enabled;
    if (dom.fitTargetWrap) dom.fitTargetWrap.style.display = enabled ? '' : 'none';
    if (enabled && dom.fitTargetInput && !dom.fitTargetInput.value.trim()) {
      dom.fitTargetInput.value = '2048';
    }
    h.fitTarget = enabled && dom.fitTargetInput ? (dom.fitTargetInput.value.trim() || '') : '';
  } else if (dom.fitTargetInput) {
    h.fitTarget = dom.fitTargetInput.value.trim() || '';
  }
  if (dom.kvUnifiedSelect) {
    h.kvUnified = dom.kvUnifiedSelect.value === ''
      ? null
      : dom.kvUnifiedSelect.value === 'true';
  }
  if (dom.flashAttnSelect) h.flashAttn = dom.flashAttnSelect.value || '';
  if (dom.mlockCheck) h.mlock = dom.mlockCheck.checked;
 if (dom.prioSelect) { const v = dom.prioSelect.value; h.prio = v !== '' ? Number(v) : null; }
 if (dom.verbosityInput) { const v = dom.verbosityInput.value; h.verbosity = v !== '' ? Number(v) : 4; }
 if (dom.loadModeSelect) h.loadMode = dom.loadModeSelect.value || 'mmap';
 if (dom.ctxCheckpointsInput) { const v = dom.ctxCheckpointsInput.value; h.ctxCheckpoints = v !== '' ? Number(v) : null; }
 if (dom.checkpointMinStepInput) { const v = dom.checkpointMinStepInput.value; h.checkpointMinStep = v !== '' ? Number(v) : null; }
 if (dom.cacheReuseInput) { const v = dom.cacheReuseInput.value; h.cacheReuse = v !== '' ? Number(v) : null; }
 if (dom.cacheIdleSlotsSelect) h.cacheIdleSlots = dom.cacheIdleSlotsSelect.value === '' ? null : dom.cacheIdleSlotsSelect.value === 'true';
 if (dom.noContBatchingCheck) h.noContBatching = dom.noContBatchingCheck.checked;
 if (dom.swaFullCheck) h.swaFull = dom.swaFullCheck.checked;
 if (dom.mmprojOffloadSelect) h.mmprojOffload = dom.mmprojOffloadSelect.value === '' ? null : dom.mmprojOffloadSelect.value === 'true';
 if (dom.llamaReasoningEffortSelect) h.llamaReasoningEffort = dom.llamaReasoningEffortSelect.value || 'default';
 if (dom.llamaReasoningFormatSelect) h.llamaReasoningFormat = dom.llamaReasoningFormatSelect.value || null;
 if (dom.llamaReasoningPreserveSelect) h.llamaReasoningPreserve = dom.llamaReasoningPreserveSelect.value === '' ? null : dom.llamaReasoningPreserveSelect.value === 'true';
  if (dom.threadsInput) { const v = dom.threadsInput.value; h.threads = v !== '' ? Number(v) : null; }
  if (dom.threadsBatchInput) { const v = dom.threadsBatchInput.value; h.threadsBatch = v !== '' ? Number(v) : null; }
  if (dom.specDraftNMinInput) { const v = dom.specDraftNMinInput.value; h.mtpDraftNMin = v !== '' ? Number(v) : null; }
  if (dom.specDraftPMinInput) { const v = dom.specDraftPMinInput.value; h.mtpDraftPMin = v !== '' ? parseFloat(v) : null; }
  if (dom.cacheRamInput) {
    const v = dom.cacheRamInput.value.trim();
    h.cacheRam = v !== '' ? parseInt(v, 10) : null;
  }
  if (dom.cacheModeSelect) h.cacheMode = dom.cacheModeSelect.value || 'custom';
}

export function scheduleVramUpdate() {
  if (vramDebounce) clearTimeout(vramDebounce);
  vramDebounce = setTimeout(updateVramDisplay, 150);
}

export function maybeResetHardwareStepScroll() {
  if (!pendingHardwareScrollReset || wizardState.currentStep !== 1) return;
  pendingHardwareScrollReset = false;

  const main = document.querySelector('#wizard-step-1 .wizard-main');
  const sidebar = document.querySelector('#wizard-step-1 .hw-vram-sidebar');
  // When a toggle hides part of the hardware form, the column can shrink from
  // scrollable to non-scrollable in the same frame. Reset unconditionally so we
  // never leave the viewport stranded at a stale offset showing blank space.
  if (main) main.scrollTop = 0;
  if (sidebar) sidebar.scrollTop = 0;
}

export function maybeRestoreHardwareStepScroll() {
  if (!pendingHardwareScrollRestore || wizardState.currentStep !== 1) return;

  const snapshot = pendingHardwareScrollRestore;
  pendingHardwareScrollRestore = null;

  const restore = () => {
    if (wizardState.currentStep !== 1) return;

    const focused = document.activeElement;
    if (focused === dom.fitEnableSelect) {
      focused.blur?.();
    }

    const main = document.querySelector('#wizard-step-1 .wizard-main');
    const sidebar = document.querySelector('#wizard-step-1 .hw-vram-sidebar');
    if (main) {
      const maxScroll = Math.max(0, main.scrollHeight - main.clientHeight);
      main.scrollTop = Math.min(snapshot.main, maxScroll);
    }
    if (sidebar) {
      const maxScroll = Math.max(0, sidebar.scrollHeight - sidebar.clientHeight);
      sidebar.scrollTop = Math.min(snapshot.sidebar, maxScroll);
    }
  };

  requestAnimationFrame(() => {
    restore();
    requestAnimationFrame(restore);
  });
}

function bindHardwareToggleSwitch(labelEl, inputEl) {
  if (!labelEl || !inputEl) return;

  labelEl.addEventListener('pointerdown', e => {
    e.preventDefault();
  });

  labelEl.addEventListener('click', e => {
    if (e.target === inputEl) return;
    e.preventDefault();
    inputEl.checked = !inputEl.checked;
    inputEl.dispatchEvent(new Event('input', { bubbles: true }));
    inputEl.dispatchEvent(new Event('change', { bubbles: true }));
  });
}

// ── Animated VRAM display ─────────────────────────────────────────────────────

export function getEffectiveArch() {
  // Architecture and capability fields are authoritative only after local GGUF or
  // progressive HF-header introspection. Unknown/degraded models intentionally return
  // zero-valued fields so the estimator can report an unknown fit instead of guessing
  // from a filename or repository label.
  return { ...wizardState.arch };
}

export function getSizingArch() {
  const base = getEffectiveArch();
  const arch = { ...base };
  const hasSelectedMmproj = !!(wizardState.model.mmprojPath || wizardState.model.mmprojHfFile);

  // Size against the current wizard choices, not just what the model could support.
  arch.mtpDepth = wizardState.hardware.mtpEnabled ? (base.mtpDepth || 0) : 0;
  arch.mmprojBytes = hasSelectedMmproj ? (base.mmprojBytes || 0) : 0;

  return arch;
}

export function buildHeuristicArch(_name, _paramB) {
  // Retained as a compatibility export for integrations that imported the old helper.
  // Architecture properties must come from GGUF/MLX introspection; never infer them here.
  return {
    nLayers: 0, nKvHeads: 0, headDim: 0, nGlobalAttnLayers: 0,
    localAttnWindow: 0, localKvHeads: 1, nAttnLayers: 0,
    linearAttnStateBytes: 0, nExperts: 0, nExpertsUsed: 0,
    expertFraction: 0, mtpDepth: 0, mmprojBytes: 0, paramB: 0,
  };
}
/* Legacy filename/parameter heuristics removed from the active path. They remain below only
 * as historical context until the next generated-source cleanup pass. */
/*
  const lower = (name || '').toLowerCase();

  // ── Qwen3-Coder-Next: hybrid DeltaNet + MoE ──────────────────────────────
  if (lower.includes('coder-next') || lower.includes('qwen3-coder-next')) {
    // 48 layers (12 attn + 36 DeltaNet), 512 experts / 11 active, head_dim 256
    return {
      nLayers: 48, nKvHeads: 2, headDim: 256,
      nAttnLayers: 12, // only these 12 use KV cache
      linearAttnStateBytes: 36 * 32 * 128 * 128 * 2, // ~38 MB (negligible)
      nGlobalAttnLayers: 0, localAttnWindow: 0, localKvHeads: 1,
      nExperts: 512, nExpertsUsed: 11, expertFraction: 0.92,
      mtpDepth: wizardState.arch.mtpDepth || 0,
      mmprojBytes: wizardState.arch.mmprojBytes || 0,
    };
  }

// ── Qwen3.6 family: hybrid DeltaNet, 1/4 attn layers ─────────────────────
   // Covers: Qwen3.6-27B (dense), Qwen3.6-35B-A3B (MoE), davidau 40B expansion,
   // Qwopus3.6 derivatives, and all finetunes/distillations that mention Qwen3.6.
   if (lower.includes('qwen3.6') || lower.includes('qwen3-6') || lower.includes('qwopus3.6') || lower.includes('qwopus3-6') || lower.includes('qwopus36')) {
    const nLayers = paramB > 35 ? 96 : 64;
    const nAttnLayers = Math.floor(nLayers / 4); // exactly 1:3 attn:deltanet ratio
    const nDeltanet = nLayers - nAttnLayers;
    const linearState = nDeltanet * 48 * 128 * 128 * 2; // ~76 MB for 27B
    const isMoe = parseMoeSuffix(name) !== null || lower.includes('a3b');
    return {
      nLayers, nKvHeads: 4, headDim: 256,
      nAttnLayers, linearAttnStateBytes: linearState,
      nGlobalAttnLayers: 0, localAttnWindow: 0, localKvHeads: 1,
      nExperts: isMoe ? 64 : 0,
      nExpertsUsed: isMoe ? 3 : 0,
      expertFraction: isMoe ? 0.80 : 0.65,
      mtpDepth: wizardState.arch.mtpDepth || 0,
      mmprojBytes: wizardState.arch.mmprojBytes || 0,
      paramB,
    };
  }

  const isGemma4 = lower.includes('gemma-4') || lower.includes('gemma4');
  if (isGemma4) {
    const namedE2B = lower.includes('e2b');
    const namedE4B = lower.includes('e4b');
    const named12B = lower.includes('12b');
    const named26BA4B = lower.includes('26b-a4b') || lower.includes('26b_a4b') || lower.includes('a4b');
    const named31B = lower.includes('31b');
    const hasNamedSize = namedE2B || namedE4B || named12B || named26BA4B || named31B;
    const isE2B = namedE2B || (!hasNamedSize && paramB < 6);
    const isE4B = namedE4B || (!hasNamedSize && !isE2B && paramB < 10);
    const is12B = named12B || (!hasNamedSize && !isE2B && !isE4B && paramB < 20);
    let cfg;
    if (isE2B) cfg = [35, 7, 1, 1, 512, 0, 0];
    else if (isE4B) cfg = [42, 7, 2, 2, 512, 0, 0];
    else if (is12B) cfg = [48, 8, 1, 8, 1024, 0, 0];
    else if (named26BA4B || (!hasNamedSize && paramB < 30)) cfg = [30, 5, 2, 8, 1024, 128, 9];
    else cfg = [60, 10, 4, 16, 1024, 0, 0];
    return {
      nLayers: wizardState.arch.nLayers || cfg[0],
      nKvHeads: wizardState.arch.nKvHeads || cfg[2],
      headDim: wizardState.arch.headDim || 256,
      globalHeadDim: 512,
      nGlobalAttnLayers: cfg[1],
      localAttnWindow: cfg[4],
      localKvHeads: cfg[3],
      nExperts: wizardState.arch.nExperts || cfg[5],
      nExpertsUsed: wizardState.arch.nExpertsUsed || cfg[6],
      expertFraction: 0.65,
      mtpDepth: wizardState.arch.mtpDepth || 0,
      mmprojBytes: wizardState.arch.mmprojBytes || 0,
      paramB,
    };
  }

  const isGemma3 = lower.includes('gemma-3') || lower.includes('gemma3');
  if (isGemma3) {
    const n = paramB < 5 ? [34, 4, 256] : paramB < 14 ? [52, 8, 256] : [62, 16, 256];
    const globalL = Math.round(n[0] / 6);
    return {
      nLayers: n[0], nKvHeads: n[1], headDim: n[2],
      nGlobalAttnLayers: globalL, localAttnWindow: 512, localKvHeads: 1,
      // Inherit MoE state if already detected (e.g. Gemma-4-26B-A4B)
      nExperts: wizardState.arch.nExperts || 0,
      nExpertsUsed: wizardState.arch.nExpertsUsed || 0,
      expertFraction: 0.65,
      mtpDepth: wizardState.arch.mtpDepth || 0,
      mmprojBytes: wizardState.arch.mmprojBytes || 0,
    };
  }

  // Standard heuristic
  let nl, nkv, hd;
  if (paramB < 2)       { nl=22;  nkv=4;  hd=64;  }
  else if (paramB < 5)  { nl=28;  nkv=4;  hd=128; }
  else if (paramB < 10) { nl=32;  nkv=8;  hd=128; }
  else if (paramB < 18) { nl=40;  nkv=8;  hd=128; }
  else if (paramB < 35) { nl=40;  nkv=8;  hd=128; }
  else if (paramB < 55) { nl=60;  nkv=8;  hd=128; }
  else                  { nl=80;  nkv=8;  hd=128; }

  return {
    nLayers: nl, nKvHeads: nkv, headDim: hd,
    nGlobalAttnLayers: 0, localAttnWindow: 0, localKvHeads: 1,
    nExperts: wizardState.arch.nExperts || 0,
    nExpertsUsed: wizardState.arch.nExpertsUsed || 0,
    expertFraction: wizardState.arch.expertFraction || 0.65,
    mtpDepth: wizardState.arch.mtpDepth || 0,
    mmprojBytes: wizardState.arch.mmprojBytes || 0,
    paramB,
  };
}
*/

async function estimateVramFull() {
  // Called from JS math; no server round-trip needed for the breakdown
  updateVramDisplay();
}

export function getModelBytes() {
  if (wizardState.model.modelBytes > 0) return wizardState.model.modelBytes;
  const paramB = wizardState.model.paramB;
  if (!paramB) return 0;
  // Estimate from param count + quant for both local and HF models.
  // Use the selected HF file name first, then fall back to the local path.
  const fname = (wizardState.model.hfFile || wizardState.model.path || '').toLowerCase();
  const quant = guessQuantFromName(fname);
  const BPW = { f16:16, q8_0:8.5, q6_k:6.5625, q5_k_m:5.69, q5_k_s:5.52, q4_k_m:4.85, q4_k_s:4.58, q4_0:4.55, iq4_xs:4.25, q3_k_m:3.875, q2_k:2.625, iq2_xxs:2.0625, iq1_m:1.75 };
  const bpw = BPW[quant] ?? 4.85;
  return Math.round(paramB * 1e9 * bpw / 8);
}

export function guessQuantFromName(name) {
  const lower = name.toLowerCase();
  if (lower.includes('q8_0')) return 'q8_0';
  if (lower.includes('q6_k')) return 'q6_k';
  if (lower.includes('q5_k_m')) return 'q5_k_m';
  if (lower.includes('q5_k_s')) return 'q5_k_s';
  if (lower.includes('q4_k_m')) return 'q4_k_m';
  if (lower.includes('q4_k_s')) return 'q4_k_s';
  if (lower.includes('iq4_xs')) return 'iq4_xs';
  if (lower.includes('q4_0')) return 'q4_0';
  if (lower.includes('q3_k_m')) return 'q3_k_m';
  if (lower.includes('q2_k')) return 'q2_k';
  if (lower.includes('iq2_xxs')) return 'iq2_xxs';
  if (lower.includes('f16') || lower.includes('bf16')) return 'f16';
  return 'q4_k_m'; // reasonable default
}

// ── Performance advisor (config-time hints) ───────────────────────────────────
let _advisorTimer = null;
let _advisorSeq = 0;

export function updateAdvisor() {
  const box = document.getElementById('wizard-advisor');
  const cards = document.getElementById('wizard-advisor-cards');
  if (!box || !cards) return;

  clearTimeout(_advisorTimer);
  _advisorTimer = setTimeout(async () => {
    const arch = getSizingArch();
    const baseArch = getEffectiveArch(); // capability arch — mtpDepth not zeroed when MTP is off
    const hw = wizardState.hardware;
    const m = wizardState.model;

    // Batch/ubatch sweep and depth sweep are llama.cpp-only (require llama-bench + .gguf).
    const localGguf = !!(m.path && m.path.toLowerCase().endsWith('.gguf'));
    const isLlamaCpp = wizardState.engine.selected === 'llama_cpp';
    const sweepAvailable = localGguf && isLlamaCpp;
    const batchSweepBox = document.getElementById('wizard-batch-sweep');
    if (batchSweepBox) batchSweepBox.style.display = sweepAvailable ? '' : 'none';
    const sweepBox = document.getElementById('wizard-depth-sweep');
    if (sweepBox) sweepBox.style.display = sweepAvailable ? '' : 'none';

    // Confident MoE only: explicit isMoe flag, or expert counts from introspection.
    const moeConfident =
        wizardState.arch.isMoe === true ||
        (wizardState.arch.nExperts > 0 && wizardState.arch.nExpertsUsed > 0);

    const nCpuMoeInput = document.getElementById('spawn-n-cpu-moe');
    if (nCpuMoeInput) {
        const field = nCpuMoeInput.closest('.hardware-field') || nCpuMoeInput.parentElement;
        if (field) field.style.display = moeConfident ? '' : 'none';
        else nCpuMoeInput.style.display = moeConfident ? '' : 'none';
    }

    const moeBox = document.getElementById('spawn-moe-autotune');
    if (moeBox) moeBox.style.display = moeConfident ? '' : 'none';

    // On Apple Silicon / unified memory: warn that n_cpu_moe is for discrete GPUs
    const hintEl = document.getElementById('spawn-n-cpu-moe-hint');
    if (hintEl) {
        const isAppleUnified = isUnifiedMemory();
        if (moeConfident && isAppleUnified) {
            hintEl.style.display = '';
            hintEl.textContent = 'For discrete GPUs only. On Apple Silicon (unified memory), this usually slows performance—leave at Auto.';
        } else if (moeConfident) {
            hintEl.style.display = '';
            hintEl.textContent = 'Useful on discrete GPUs: offload some MoE expert layers to CPU RAM when VRAM is limited.';
        } else {
            hintEl.style.display = 'none';
        }
    }

    const name = (m.path || m.hfFile || m.hfRepo || m.originFile || '').split('/').pop() || '';
    const paramB = arch.paramB || m.paramB || 0;
    if (!name && !paramB) { box.style.display = 'none'; return; }

// MTP depth is authoritative only from introspected architecture/profile
// metadata. A model-browser draft/head classification can still provide a
// provisional candidate hint when that separate artifact exposes no head data.
const hasMtp = (baseArch.mtpDepth || 0) > 0;
const mtpInferred = !hasMtp && Boolean(m.isDraftModel || m.is_draft_assistant);
    const body = {
      name,
      param_b: paramB,
      context_size: hw.contextSize,
      ctk: hw.cacheTypeK,
      ctv: hw.cacheTypeV,
      is_unified: isUnifiedMemory(),
      spec_type: hw.mtpEnabled ? 'draft-mtp' : null,
      has_mtp: hasMtp,
      mtp_inferred: mtpInferred,
    };

    const seq = ++_advisorSeq;
    try {
      const headers = window.authHeaders
        ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
        : { 'Content-Type': 'application/json' };
      const r = await fetch('/api/advise', { method: 'POST', headers, body: JSON.stringify(body) });
      if (seq !== _advisorSeq) return; // a newer recompute superseded this one
      const data = await r.json();
      const suggestions = (data && data.suggestions) || [];
      const cfgView = {
        ctk: hw.cacheTypeK,
        ctv: hw.cacheTypeV,
        context_size: hw.contextSize,
        spec_type: hw.mtpEnabled ? 'draft-mtp' : '',
        spec_draft_n_max: hw.mtpDraftNMax,
      };
      renderSuggestionCards(cards, suggestions, { onApply: applyWizardSuggestion, config: cfgView });
      box.style.display = cards.childElementCount ? '' : 'none';
    } catch {
      box.style.display = 'none';
    }
  }, 250);
}

// ── Hardware step: model header + quant swap ─────────────────────────────────

function _refreshThreadsHint() {
  const hintEl = document.getElementById('spawn-threads-hint');
  const batchHintEl = document.getElementById('spawn-threads-batch-hint');
  if (!hintEl && !batchHintEl && !dom.threadsInput && !dom.threadsBatchInput) return;

  const pCores = lastSystemMetrics?.p_cores || 0;
  const metricsReady = lastSystemMetrics != null;

  if (pCores > 0 && metricsReady) {
    if (hintEl) {
      hintEl.textContent =
        `Apple Silicon: use 1. GPU (Metal) handles all inference — extra CPU threads add contention without benefit.`;
    }
    if (batchHintEl) {
      batchHintEl.textContent =
        `Apple Silicon: use ${pCores} (all P-cores) — CPU handles prefill/batch. More cores = faster prompt processing.`;
    }
    if (dom.threadsInput && !dom.threadsInput.value) {
      dom.threadsInput.placeholder = '1 recommended';
    }
    if (dom.threadsBatchInput && !dom.threadsBatchInput.value) {
      dom.threadsBatchInput.placeholder = `${pCores} recommended`;
    }
    return;
  }

 if (!metricsReady) {
    if (hintEl) {
      hintEl.textContent = 'Blank or -1 = server default (-t). Hardware-specific guidance loads automatically.';
    }
    if (batchHintEl) {
      batchHintEl.textContent = 'Prompt processing threads. Blank or -1 = inherit from -t.';
    }
    if (dom.threadsInput && !dom.threadsInput.value) {
      dom.threadsInput.placeholder = 'default';
    }
    if (dom.threadsBatchInput && !dom.threadsBatchInput.value) {
      dom.threadsBatchInput.placeholder = 'default';
    }
    return;
  }

  // Non-Apple Silicon / no P-cores: generic hint.
  if (hintEl) {
    hintEl.textContent = 'Blank or -1 = server default (-t). Sets CPU threads for inference. Do not exceed physical P-core count.';
  }
  if (batchHintEl) {
    batchHintEl.textContent = 'Prompt processing threads. Blank or -1 = inherit from -t.';
  }
  if (dom.threadsInput && !dom.threadsInput.value) {
    dom.threadsInput.placeholder = 'default';
  }
  if (dom.threadsBatchInput && !dom.threadsBatchInput.value) {
    dom.threadsBatchInput.placeholder = 'default';
  }
}
window.__refreshSpawnWizardHints = _refreshThreadsHint;

async function _fetchSystemInfoAndRefreshHints() {
  try {
    const headers = window.authHeaders ? window.authHeaders() : {};
    const res = await fetch('/api/system/info', { headers });
    if (!res.ok) return;
    const data = await res.json();
    if (data.ok && data.p_cores > 0) {
      // Populate lastSystemMetrics with at minimum the core counts so hints work.
      // Use setLastSystemMetrics from app-state so the live binding updates.
      const { setLastSystemMetrics } = await import('../core/app-state.js');
      setLastSystemMetrics({ p_cores: data.p_cores, e_cores: data.e_cores, cpu_name: data.cpu_name });
      _refreshThreadsHint();
    }
  } catch { /* non-fatal */ }
}


// e.g. "Qwopus3.6-27B-v2-MTP-Q8_0.gguf" → "Qwopus3.6-27B-v2-MTP"
export function _modelStemForSearch(filename) {
  return filename
    .replace(/\.gguf$/i, '')
    .replace(/-(?:(?:UD-)?(?:IQ|Q)[0-9][A-Z0-9_]*|BF16|F16|FP16|FP32)$/i, '');
}

// POST to /api/hf/files and return parsed JSON, or null on error.
export async function _hfFilesPost(repoId) {
  const headers = window.authHeaders ? window.authHeaders() : {};
  try {
    const res = await fetch('/api/hf/files', {
      method: 'POST',
      headers: { ...headers, 'Content-Type': 'application/json' },
      body: JSON.stringify({ repo_id: repoId }),
    });
    if (!res.ok) return null;
    return await res.json();
  } catch { return null; }
}

// ── Local-model quant-swap discovery ─────────────────────────────────────────

// Extract a short quant label from a GGUF filename, e.g. "Q4_K_M" or "IQ3_M".
export function _extractQuantLabel(filename) {
  const fname = (filename.split('/').pop() || filename).replace(/\.gguf$/i, '');
  const m = fname.match(/[-_]((?:UD-)?(?:IQ|Q)\d[\w.]*|BF16|F16|FP16|FP32)(?:[-_.]|$)/i);
  return m ? m[1].toUpperCase() : fname;
}

function _updateSpecHint(value) {
  const container = document.getElementById('spawn-spec-hint');
  if (!container) return;
  container.querySelectorAll('[data-spec]').forEach(el => {
    el.style.display = el.dataset.spec === value ? '' : 'none';
  });
}

// ── Collapsible hardware sections ─────────────────────────────────────────────

function bindSectionToggles() {
   // Response shaping is visible by default for review-step discoverability.
   const divider = document.getElementById('hw-section-response-shaping');
   const toggle  = document.getElementById('hw-toggle-response-shaping');
   const body    = document.getElementById('hw-body-response-shaping');
   if (!divider || !body) return;

   divider.style.cursor = 'pointer';
   divider.addEventListener('click', () => {
       const collapsed = body.style.display === 'none';
       body.style.display = collapsed ? '' : 'none';
       if (toggle) toggle.textContent = collapsed ? '▾' : '▸';
   });
}

// ── Model directory switcher ──────────────────────────────────────────────────

// ── Browse split-button dropdown ──────────────────────────────────────────────

function _closeBrowseDropdowns() {
  ['spawn-browse-dropdown', 'spawn-import-browse-dropdown'].forEach(id => {
    const dd = document.getElementById(id);
    if (dd) dd.style.display = 'none';
  });
  document.getElementById('spawn-browse-arrow-btn')?.setAttribute('aria-expanded', 'false');
  document.getElementById('spawn-import-browse-arrow-btn')?.setAttribute('aria-expanded', 'false');
}

function _buildBrowseDropdown(dropdownEl, targetInputId, allDirs) {
  dropdownEl.innerHTML = '';

  allDirs.forEach((dir, i) => {
    const parts = dir.replace(/\\/g, '/').split('/').filter(Boolean);
    const label = parts[parts.length - 1] || dir;
    const pathHint = parts.slice(0, -1).join('/');

    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'browse-dd-item';
    btn.title = dir;

    const labelEl = document.createElement('span');
    labelEl.className = 'dd-item-label';
    labelEl.textContent = label;
    btn.appendChild(labelEl);

    if (pathHint) {
      const pathEl = document.createElement('span');
      pathEl.className = 'dd-item-path';
      pathEl.textContent = '/' + pathHint;
      btn.appendChild(pathEl);
    }

    btn.addEventListener('click', () => {
      _closeBrowseDropdowns();
      const rapid = targetInputId === 'spawn-model-path' && wizardState.engine.selected === 'rapid_mlx';
      const separator = dir.includes('\\') ? '\\' : '/';
      const browseDir = rapid
        ? dir.replace(/[\\/]+$/, '') + separator + ['mlx', 'native'].join(separator)
        : dir.replace(/[\\/]+$/, '') + separator + 'gguf';
      const ctx = targetInputId === 'spawn-model-path'
        ? { kind: 'model', engine: rapid ? 'rapid_mlx' : 'llama_cpp' }
        : 'model';
      openDeferredFileBrowser(targetInputId, rapid ? 'dir' : 'gguf', browseDir, ctx);
    });
    dropdownEl.appendChild(btn);
  });

  const divider = document.createElement('div');
  divider.className = 'browse-dd-divider';
  dropdownEl.appendChild(divider);

  const manageBtn = document.createElement('button');
  manageBtn.type = 'button';
  manageBtn.className = 'browse-dd-item dd-manage';
  manageBtn.textContent = '⚙ Manage model locations…';
  manageBtn.addEventListener('click', () => {
    _closeBrowseDropdowns();
    Router.navigate('/settings#models');
  });
  dropdownEl.appendChild(manageBtn);
}

function _toggleBrowseDropdown(arrowBtnId, dropdownId, targetInputId) {
  const arrow = document.getElementById(arrowBtnId);
  const dd    = document.getElementById(dropdownId);
  if (!arrow || !dd) return;

  const isOpen = dd.style.display !== 'none';

  // Close all first, then open this one if it was closed
  _closeBrowseDropdowns();

  if (!isOpen) {
    dd.style.display = 'block';
    arrow.setAttribute('aria-expanded', 'true');
  }
}

async function _loadModelDirSwitcher() {
  try {
    const headers = window.authHeaders ? window.authHeaders() : {};
    const r = await fetch('/api/settings', { headers });
    if (!r.ok) return;
    const s = await r.json();

    const primary = s.models_dir || '';
    const extras = Array.isArray(s.extra_models_dirs) ? s.extra_models_dirs.filter(Boolean) : [];
    const allDirs = [primary, ...extras].filter(Boolean);

    const localDd  = document.getElementById('spawn-browse-dropdown');
    const importDd = document.getElementById('spawn-import-browse-dropdown');
    if (localDd)  _buildBrowseDropdown(localDd,  'spawn-model-path',  allDirs);
    if (importDd) _buildBrowseDropdown(importDd, 'spawn-import-path', allDirs);

    // Wire arrow buttons (idempotent — clone to remove old listeners)
    const wireArrow = (arrowId, dropdownId, targetInputId) => {
      const old = document.getElementById(arrowId);
      if (!old) return;
      const fresh = old.cloneNode(true);
      old.replaceWith(fresh);
      fresh.addEventListener('click', e => {
        e.stopPropagation();
        _toggleBrowseDropdown(arrowId, dropdownId, targetInputId);
      });
    };
    wireArrow('spawn-browse-arrow-btn',        'spawn-browse-dropdown',        'spawn-model-path');
    wireArrow('spawn-import-browse-arrow-btn', 'spawn-import-browse-dropdown', 'spawn-import-path');
  } catch { /* ignore */ }
}
