/* global DOMPurify */
// Rapid-MLX backend adapter for the shared Spawn Wizard.
//
// Deliberately named spawn-wizard-rapid-mlx.js, not spawn-wizard-mlx.js: per the
// 2026-07-29 Rapid-MLX/MTPLX runtime audit (docs/reference/apple-silicon-mtp-runtime-comparison.md),
// MTPLX's config surface, cache model, and MTP semantics are fundamentally different from Rapid's
// vLLM-style flags (kv_cache_dtype, turboquant_mode, pflash_policy, hybrid_cache_entries, sidecar
// pinning, etc. have no MTPLX equivalent). Everything below is a Rapid-MLX-specific adapter, not
// MLX-family-generic code -- do not treat this module as a template to copy for MTPLX; write a
// separate spawn-wizard-mtplx.js adapter against MTPLX's own capability surface instead.
//
// Extracted from spawn-wizard.js as a behavior-preserving move: same DOM ids,
// same event wiring, same wizardState field names. spawn-wizard.js remains the
// shared wizard shell and delegates to this module at its Rapid-MLX call sites.

import { wizardState, dom } from './spawn-wizard.js';
import { setHtml } from '../core/set-html.js';
import {
    RAPID_MLX_DEFAULT_SPECULATIVE_TOKENS,
    rapidMlxPrefillStepSizeDefault,
    rapidMlxProfileHasVision,
} from './rapid-mlx-prefill.js';
import {
    findRapidMlxSidecarForTrunk,
    rapidMlxSidecarProvenance,
} from '../core/rapid-mlx-sidecars.js';

/*
 * Mirrors the two Rapid-MLX mutual-exclusion rules that involve the Phase 7 throughput
 * fields, so an invalid pair is visible while it is being chosen rather than only at launch.
 *
 * These duplicate `mutual_exclusion_rules()` in src/inference/rapid_mlx/settings.rs. The
 * backend stays authoritative -- this is a nearer warning, not a replacement for it.
 */
export function renderRapidExclusionWarnings() {
  const h = wizardState.hardware;
  const conflicts = [];

  if (h.pflashPolicy === 'on' && (h.turboquantMode === 'v4' || h.turboquantMode === 'k8v4')) {
    conflicts.push('PFlash bypasses TurboQuant, so “On” cannot be combined with a TurboQuant reusable-prompt-storage mode.');
  }

  const host = dom.pflashPolicySelect?.closest('.hardware-grid');
  if (!host) return;
  let box = document.getElementById('spawn-rapid-exclusion-warning');
  if (conflicts.length === 0) {
    if (box) box.remove();
    return;
  }
  if (!box) {
    box = document.createElement('div');
    box.id = 'spawn-rapid-exclusion-warning';
    box.className = 'spawn-command-preview-error';
    box.style.gridColumn = '1 / -1';
    host.appendChild(box);
  }
  setHtml(box, '');
  const title = document.createElement('div');
  title.className = 'spawn-command-preview-error-title';
  title.textContent = conflicts.length === 1 ? 'Conflicting setting' : 'Conflicting settings';
  box.appendChild(title);
  conflicts.forEach((text) => {
    const detail = document.createElement('div');
    detail.className = 'spawn-command-preview-error-detail';
    detail.textContent = text;
    box.appendChild(detail);
  });
}

function _timeAgoSpawn(dt) {
  if (!dt) return '';
  const d = new Date(dt);
  const diff = (Date.now() - d.getTime()) / 1000;
  if (diff < 60) return 'just now';
  if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
  if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
  return Math.floor(diff / 86400) + 'd ago';
}

function _updateSpawnTrustUI(h, enabled) {
  if (!dom.speculativeTrustWrap) return;
  const show = enabled && h.speculativeSource === 'external' && h.speculativeTrustRequired;
  dom.speculativeTrustWrap.style.display = show ? '' : 'none';
  if (dom.speculativeTrustConsent) dom.speculativeTrustConsent.checked = !!h.speculativeTrustConsent;
}

function _hideSpawnTrust() {
  const h = wizardState.hardware;
  h.speculativeTrustRequired = false;
  h.speculativeTrustConsent = false;
  h.speculativeTrustRepoId = '';
  h.speculativeTrustRevision = '';
  _updateSpawnTrustUI(h, !!h.speculativeEnabled);
  if (dom.speculativePinStatusWrap) dom.speculativePinStatusWrap.style.display = 'none';
}

function _renderSpawnPinStatus() {
  const wrap = document.getElementById('spawn-rapid-speculative-pin-status-wrap');
  const el = document.getElementById('spawn-rapid-speculative-pin-status');
  if (!wrap || !el) return;

  const h = wizardState.hardware;
  if (!h.speculativeTrustRepoId) {
    wrap.style.display = 'none';
    return;
  }

  // Determine if this is a local sidecar (no revision sha, no trust required)
  // or an HF repo pin (has revision sha, may need trust)
  const isLocalSidecar = !h.speculativeTrustRevision || h.speculativeTrustRevision.length > 12;

  const rev = h.speculativeTrustRevision.substring(0, 12);
  const stale = !!h.speculativeTrustStale;
  const trust = h.speculativeTrustRequired;
  const mem = h.speculativeTrustEstimatedMemoryBytes;
  const sidecar = h.speculativeTrustSidecar;
  const depth = h.speculativeTrustDepth;

  let parts = [];
  parts.push('<span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:' + (stale ? 'var(--warn,#e6a41c)' : 'var(--success,#5ce68a)') + '"></span>');

  if (isLocalSidecar) {
    // Local sidecar: show slug + sidecar label
    parts.push('<span>' + h.speculativeTrustRepoId + '</span>');
    parts.push('<span style="color:var(--text-muted,#888);">(local sidecar)</span>');
  } else {
    // HF repo pin: show repo@sha
    parts.push('<span>' + h.speculativeTrustRepoId + '@' + rev + '</span>');
  }

  if (trust) {
    parts.push('<span style="color:var(--err,#e65c5c);">(trust_remote_code)</span>');
  }
  if (mem != null) {
    let memStr;
    if (mem >= 1073741824) {
      memStr = '~' + (mem / 1073741824).toFixed(1) + ' GB';
    } else {
      memStr = '~' + Math.round(mem / 1048576) + ' MB';
    }
    parts.push('<span style="color:var(--text-muted,#888);">~' + memStr + ' VRAM</span>');
  }
  if (sidecar) {
    parts.push('<span style="color:var(--text-muted,#888);">sidecar:' + sidecar + (depth != null ? ' d' + depth : '') + '</span>');
  }
  if (h.speculativeTrustResolvedAt) {
    parts.push('<span style="color:var(--text-muted,#888);">resolved ' + _timeAgoSpawn(h.speculativeTrustResolvedAt) + '</span>');
  }

  // Re-check button only for HF repo pins
  if (!isLocalSidecar) {
    parts.push('<button type="button" class="hw-action-btn" id="spawn-rapid-speculative-pin-recheck" style="font-size:11px; padding:2px 8px; margin-left:4px;">Re-check</button>');
  }


  setHtml(el, parts.join(' '));
  wrap.style.display = '';

  const btn = document.getElementById('spawn-rapid-speculative-pin-recheck');
  if (btn) {
    btn.addEventListener('click', async () => {
      if (!h.speculativeTrustRepoId) return;
      btn.disabled = true;
      btn.textContent = '…';
      try {
        const resp = await fetch(
          '/api/hf/mtp-preflight/recheck?repo=' + encodeURIComponent(h.speculativeTrustRepoId),
          { method: 'POST', headers: window.authHeaders ? window.authHeaders() : {} }
        );
        const data = await resp.json();
        if (!data || data.ok !== true) {

          setHtml(el, '<span style="color:var(--err,#e65c5c);">Re-check failed: ' + (data?.error || 'unknown') + '</span>');
          setTimeout(_renderSpawnPinStatus, 3000);
          return;
        }
        h.speculativeTrustRevision = data.revision;
        h.speculativeTrustRequired = !!data.trustRemoteCodeRequired;
        h.speculativeTrustLastRecheckAt = data.lastRecheckAt || '';
        h.speculativeTrustUpstreamUnchanged = data.upstreamUnchanged ?? null;
        h.speculativeTrustResolvedAt = data.resolvedAt || '';
        h.speculativeTrustStale = false;
        _renderSpawnPinStatus();
      } catch (e) {

        setHtml(el, '<span style="color:var(--err,#e65c5c);">Re-check failed: ' + e.message + '</span>');
        setTimeout(_renderSpawnPinStatus, 3000);
      } finally {
        btn.disabled = false;
      }
    });
  }
}

async function _spawnCheckTrust(repoId) {
  const h = wizardState.hardware;
  if (!repoId || !repoId.includes('/')) {
    _hideSpawnTrust();
    return;
  }
  if (!/^[\w._-]+\/[\w._-]+$/.test(repoId)) {
    _hideSpawnTrust();
    return;
  }
  try {
    const resp = await fetch(
      '/api/hf/mtp-preflight?repo=' + encodeURIComponent(repoId),
      { headers: window.authHeaders ? window.authHeaders() : {} }
    );
    const data = await resp.json();
    if (!data || data.ok !== true) {
      _hideSpawnTrust();
      return;
    }
    h.speculativeTrustRepoId = data.repoId;
    h.speculativeTrustRevision = data.revision;
    h.speculativeTrustRequired = !!data.trustRemoteCodeRequired;
    h.speculativeTrustResolvedAt = data.resolvedAt || '';
    h.speculativeTrustLastRecheckAt = data.lastRecheckAt || '';
    h.speculativeTrustUpstreamUnchanged = data.upstreamUnchanged ?? null;
    h.speculativeTrustStale = !!data.stale;
    h.speculativeTrustEstimatedMemoryBytes = data.estimatedMemoryBytes ?? null;
    h.speculativeTrustSidecar = data.mtpSidecar ?? null;
    h.speculativeTrustDepth = data.mtpDepthMax ?? null;
    _renderSpawnPinStatus();
    if (h.speculativeTrustRequired) {
      h.speculativeTrustConsent = false;
      if (dom.speculativeTrustWarning) {
        let msg = 'This companion model requires trust_remote_code (custom Python code execution).';
        msg += '\nPinned to ' + data.repoId + '@' + data.revision.substring(0, 12);
        if (h.speculativeTrustStale) {
          msg += ' (stale — consider re-checking)';
        }
        if (data.upstreamUnchanged === false) {
          msg += ' (upstream changed)';
        }
        dom.speculativeTrustWarning.textContent = msg;
      }
    } else {
      if (dom.speculativeTrustWarning) {
        dom.speculativeTrustWarning.textContent =
          'Pinned to ' + data.repoId + '@' + data.revision.substring(0, 12) + ' (no trust_remote_code needed)';
      }
    }
    if (dom.speculativeTrustWrap) {
      const enabled = !!h.speculativeEnabled && h.speculativeSource === 'external';
      _updateSpawnTrustUI(h, enabled);
    }
  } catch {
    _hideSpawnTrust();
  }
}

async function _spawnRecheckTrustPin() {
  const h = wizardState.hardware;
  if (!h.speculativeTrustRepoId) return;
  if (dom.speculativeRecheckBtn) {
    dom.speculativeRecheckBtn.disabled = true;
    dom.speculativeRecheckBtn.textContent = 'Re-checking...';
  }
  try {
    const resp = await fetch(
      '/api/hf/mtp-preflight/recheck?repo=' + encodeURIComponent(h.speculativeTrustRepoId),
      { method: 'POST', headers: window.authHeaders ? window.authHeaders() : {} }
    );
    const data = await resp.json();
    if (!data || data.ok !== true) {
      if (dom.speculativeRecheckStatus) {
        dom.speculativeRecheckStatus.textContent = 'Re-check failed: ' + (data?.error || 'unknown error');
        dom.speculativeRecheckStatus.style.display = '';
        dom.speculativeRecheckStatus.style.color = 'var(--err,#e65c5c)';
      }
      return;
    }
    h.speculativeTrustRevision = data.revision;
    h.speculativeTrustRequired = !!data.trustRemoteCodeRequired;
    h.speculativeTrustLastRecheckAt = data.lastRecheckAt || '';
    h.speculativeTrustUpstreamUnchanged = data.upstreamUnchanged ?? null;
    h.speculativeTrustResolvedAt = data.resolvedAt || '';
    h.speculativeTrustStale = false;
    _renderSpawnPinStatus();

    if (h.speculativeTrustRequired) {
      let msg = 'This companion model requires trust_remote_code (custom Python code execution).';
      msg += '\nPinned to ' + data.repoId + '@' + data.revision.substring(0, 12);
      if (data.upstreamUnchanged === false) {
        msg += ' (upstream changed)';
      }
      if (dom.speculativeTrustWarning) dom.speculativeTrustWarning.textContent = msg;
    } else {
      if (dom.speculativeTrustWarning) {
        dom.speculativeTrustWarning.textContent =
          'Pinned to ' + data.repoId + '@' + data.revision.substring(0, 12) + ' (no trust_remote_code needed)';
      }
    }
    _updateSpawnTrustUI(h, !!h.speculativeEnabled);

    if (dom.speculativeRecheckStatus) {
      dom.speculativeRecheckStatus.textContent = 'Pin verified at ' + new Date(data.lastRecheckAt).toLocaleTimeString();
      dom.speculativeRecheckStatus.style.display = '';
      dom.speculativeRecheckStatus.style.color = 'var(--success,#5ce68a)';
    }
  } catch (e) {
    if (dom.speculativeRecheckStatus) {
      dom.speculativeRecheckStatus.textContent = 'Re-check failed: ' + e.message;
      dom.speculativeRecheckStatus.style.display = '';
      dom.speculativeRecheckStatus.style.color = 'var(--err,#e65c5c)';
    }
  } finally {
    if (dom.speculativeRecheckBtn) {
      dom.speculativeRecheckBtn.disabled = false;
      dom.speculativeRecheckBtn.textContent = 'Re-check pin';
    }
  }
}

async function _fetchSpawnSidecars() {
  const listEl = document.getElementById('spawn-rapid-speculative-sidecars-list');
  if (!listEl) return;

  setHtml(listEl, '<span style="color:var(--text-muted,#888);">Loading…</span>');

  try {
    const resp = await fetch('/api/hf/mtp-sidecars', { headers: window.authHeaders ? window.authHeaders() : {} });
    const data = await resp.json();

    if (!data.ok || !data.sidecars || data.sidecars.length === 0) {
      // Even with no managed sidecar, a known tier may have an official upstream
      // draft worth preflighting before the user builds one from scratch.
      const emptyDraft = officialMtpDraftForTrunk((wizardState.model.path || '').trim());
      if (emptyDraft && !wizardState.hardware.speculativeModel) {
        setHtml(listEl, '<span style="color:var(--text-muted,#888);">No local sidecars found.</span> '
          + '<div class="field-hint" style="color:var(--accent,#8cc8ff); margin-top:5px;">'
          + 'Official upstream MTP draft exists for this tier: <code>' + emptyDraft.repo + '</code>'
          + (emptyDraft.note ? ' (' + emptyDraft.note + ')' : '')
          + ' <button type="button" class="hw-action-btn" id="spawn-rapid-speculative-official-preflight" style="font-size:11px; padding:2px 8px; margin-left:4px;">Preflight</button>'
          + '</div>');
        const emptyBtn = document.getElementById('spawn-rapid-speculative-official-preflight');
        if (emptyBtn) {
          emptyBtn.addEventListener('click', async () => {
            emptyBtn.disabled = true;
            emptyBtn.textContent = 'Preflighting…';
            try {
              await _spawnCheckTrust(emptyDraft.repo);
              emptyBtn.textContent = 'Preflighted';
            } finally {
              emptyBtn.disabled = false;
            }
          });
        }
        return;
      }
      setHtml(listEl, '<span style="color:var(--text-muted,#888);">No local sidecars found. Build one with scripts/build-mtp-head.py</span>');
      return;
    }

    const h = wizardState.hardware;
    const modelInput = document.getElementById('spawn-rapid-speculative-model');
    const selectedTrunk = (wizardState.model.path || '').trim();
    const matchingSidecar = findRapidMlxSidecarForTrunk(data.sidecars, selectedTrunk);

    // Auto-selection is deliberately one-way: only an empty field or a value we
    // previously selected automatically may be replaced. A typed path or a
    // sidecar chosen from the list is the user's override and must survive a
    // refresh or model-path change.
    if (matchingSidecar && (!h.speculativeModel || h.speculativeModelAutoSelected)) {
      h.speculativeModel = matchingSidecar.path;
      h.speculativeModelAutoSelected = true;
      if (modelInput) modelInput.value = matchingSidecar.path;
      window.scheduleVramUpdate?.();
    } else if (!matchingSidecar && h.speculativeModelAutoSelected) {
      h.speculativeModel = '';
      h.speculativeModelAutoSelected = false;
      if (modelInput) modelInput.value = '';
      _hideSpawnTrust();
      window.scheduleVramUpdate?.();
    }

    // Build sidecar list
    let html = '';
    if (matchingSidecar && h.speculativeModelAutoSelected) {
      html += '<div class="field-hint" style="color:var(--success,#5ce68a); margin-bottom:5px;">Auto-selected validated sidecar for this trunk. The path remains editable below.</div>';
    } else if (!matchingSidecar) {
      const reason = selectedTrunk.startsWith('/')
        ? (h.speculativeModel
          ? 'No managed sidecar matches this trunk; the existing manual path is preserved—verify the pairing before launch.'
          : 'No validated sidecar is registered for this trunk; speculation will stay off until one is selected.')
        : (h.speculativeModel
          ? 'Managed auto-selection is unavailable for this model reference; the explicit sidecar path is preserved.'
          : 'Select a local trunk or enter an explicit local sidecar; managed auto-selection is unavailable for this model reference.');
      html += '<div class="field-hint" style="color:var(--warn,#e6a41c); margin-bottom:5px;">' + reason + '</div>';
    }
    // No managed sidecar for this trunk? Offer the official upstream draft for the
    // trunk's tier — preflighting pins it and records provenance; launch still needs
    // an immutable local copy (build one from the preflighted source when ready).
    const officialDraft = officialMtpDraftForTrunk(selectedTrunk);
    if (!matchingSidecar && officialDraft && !h.speculativeModel) {
      html += '<div class="field-hint" style="color:var(--accent,#8cc8ff); margin-bottom:5px;">'
        + 'Official upstream MTP draft exists for this tier: <code>' + officialDraft.repo + '</code>'
        + (officialDraft.note ? ' (' + officialDraft.note + ')' : '')
        + ' <button type="button" class="hw-action-btn" id="spawn-rapid-speculative-official-preflight" style="font-size:11px; padding:2px 8px; margin-left:4px;">Preflight</button>'
        + '</div>';
    }

    data.sidecars.forEach((s, i) => {
      const p = rapidMlxSidecarProvenance(s);
      const vram = p.estimatedMemoryBytes != null
          ? '~' + (p.estimatedMemoryBytes >= 1073741824
            ? (p.estimatedMemoryBytes / 1073741824).toFixed(1) + ' GB'
            : Math.round(p.estimatedMemoryBytes / 1048576) + ' MB')
        : '? VRAM';

      const trunkShort = p.trunk ? p.trunk.split('/').pop() : '?';

      html += '<button type="button" class="hw-action-btn" data-sidecar-index="' + i + '" style="display:block; width:100%; text-align:left; padding:6px 8px; margin-bottom:4px; font-size:11px; background:var(--color-surface,#1a1d24); border:1px solid var(--color-border,#2a2d34); border-radius:4px; cursor:pointer;">';
      html += '<strong>' + DOMPurify.sanitize(s.slug) + '</strong> ';
      html += '<span style="color:var(--text-muted,#888);">' + vram + '</span>';
      if (p.trunk) html += ' <span style="color:var(--text-muted,#888);">for ' + DOMPurify.sanitize(trunkShort) + '</span>';
      if (p.repairMode === 'recipe_reconstruction') html += ' <span style="color:var(--accent,#8cc8ff);">' + (p.requalificationStatus === 'passed' ? 'Recipe reconstructed · qualified' : 'Recipe reconstructed · awaiting requalification') + '</span>';
      else if (p.repairMode === 'direct_parent') html += ' <span style="color:var(--text-muted,#888);">Direct parent</span>';
      if (p.builtAt) html += ' <span style="color:var(--text-muted,#888);">(' + _timeAgoSpawn(p.builtAt) + ')</span>';
      if (!p.normCheckPassed) html += ' <span style="color:var(--err,#e65c5c);">⚠ norm check failed</span>';
      html += '</button>';
    });


    setHtml(listEl, html);

    const officialBtn = document.getElementById('spawn-rapid-speculative-official-preflight');
    if (officialBtn) {
      officialBtn.addEventListener('click', async () => {
        officialBtn.disabled = true;
        officialBtn.textContent = 'Preflighting…';
        try {
          await _spawnCheckTrust(officialDraft.repo);
          officialBtn.textContent = 'Preflighted';
        } finally {
          officialBtn.disabled = false;
        }
      });
    }

    // Wire click handlers
    listEl.querySelectorAll('[data-sidecar-index]').forEach(btn => {
      btn.addEventListener('click', async () => {
        const idx = parseInt(btn.getAttribute('data-sidecar-index'));
        const sidecar = data.sidecars[idx];
        const p = rapidMlxSidecarProvenance(sidecar);

        // Set the companion model path
        if (modelInput) {
          modelInput.value = sidecar.path;
          h.speculativeModel = sidecar.path;
          h.speculativeModelAutoSelected = false;
          window.scheduleVramUpdate?.();
        }

        // Populate pin status from sidecar data
        h.speculativeTrustRepoId = sidecar.slug;
        h.speculativeTrustRevision = p.sha256 ? p.sha256.substring(0, 12) : '';
        h.speculativeTrustRequired = false; // local sidecars don't need trust
        h.speculativeTrustEstimatedMemoryBytes = p.estimatedMemoryBytes;
        h.speculativeTrustResolvedAt = p.builtAt;
        h.speculativeTrustStale = false;
        _renderSpawnPinStatus();
        _updateSpawnTrustUI(h, !!h.speculativeEnabled);
      });
    });
  } catch (err) {

    setHtml(listEl, '<span style="color:var(--err,#e65c5c);">Failed to load sidecars: ' + DOMPurify.sanitize(err.message) + '</span>');
  }
}

export function refreshRapidMlxSidecars() {
  const h = wizardState.hardware;
  if (h.speculativeEnabled && h.speculativeSource === 'external') {
    _fetchSpawnSidecars();
  }
}

export function syncRapidSpeculativeFields() {
  const h = wizardState.hardware;
  if (dom.speculativeEnabledCheck) dom.speculativeEnabledCheck.checked = !!h.speculativeEnabled;
  if (dom.speculativeSourceSelect) dom.speculativeSourceSelect.value = h.speculativeSource || 'embedded';
  if (dom.speculativeModelInput) dom.speculativeModelInput.value = h.speculativeModel || '';
    if (dom.speculativeTokensSelect) dom.speculativeTokensSelect.value = String(h.speculativeTokens || RAPID_MLX_DEFAULT_SPECULATIVE_TOKENS);
  if (dom.speculativeDisableAutoKCheck) dom.speculativeDisableAutoKCheck.checked = !!h.speculativeDisableAutoK;
  if (dom.autoToolChoiceCheck) dom.autoToolChoiceCheck.checked = !!h.autoToolChoice;
  const enabled = !!h.speculativeEnabled;
  ['spawn-rapid-speculative-mode-wrap', 'spawn-rapid-speculative-tokens-wrap', 'spawn-rapid-speculative-auto-k-wrap']
    .forEach(id => { const el = document.getElementById(id); if (el) el.style.display = enabled ? '' : 'none'; });
  const modelWrap = document.getElementById('spawn-rapid-speculative-model-wrap');
  if (modelWrap) modelWrap.style.display = enabled && h.speculativeSource === 'external' ? '' : 'none';
  const sidecarsWrap = document.getElementById('spawn-rapid-speculative-sidecars-wrap');
  if (sidecarsWrap) {
      if (enabled && h.speculativeSource === 'external') {
        sidecarsWrap.style.display = '';
        _fetchSpawnSidecars();
    } else {
      sidecarsWrap.style.display = 'none';
    }
  }
  _updateSpawnTrustUI(h, enabled);
}

export function applyReasoningModeLock() {
  // The Rapid reasoning quality profile is always enabled and pins active KV to int8.
  // The checkbox independently controls whether thinking output is allowed.
  if (!dom.kvCacheDtypeSelect) return;

  for (const opt of dom.kvCacheDtypeSelect.options) {
    if (opt.value === 'int4') {
      opt.disabled = true;
      opt.setAttribute('data-disabled-by', 'reasoning');
    }
  }
  if (dom.kvCacheDtypeSelect.value === 'int4') {
    dom.kvCacheDtypeSelect.value = 'int8';
    wizardState.hardware.kvCacheDtype = 'int8';
  }
  dom.kvCacheDtypeSelect.classList.add('kv-dtype-locked');
  dom.kvCacheDtypeSelect.title = 'The Rapid reasoning quality profile pins KV to int8';
}

export function applyRapidMlxDefaults() {
  const h = wizardState.hardware;

  if (!h.kvCacheDtype) {
    h.kvCacheDtype = 'int4';
    if (dom.kvCacheDtypeSelect) dom.kvCacheDtypeSelect.value = 'int4';
  }
  if (!h.turboquantMode) {
    h.turboquantMode = 'none';
    if (dom.turboquantModeSelect) dom.turboquantModeSelect.value = 'none';
  }
  // Default workloadScenario to Interactive Coding Agent if not set
    if (!h.workloadScenario) {
        h.workloadScenario = 'interactive_coding_agent';
    }
    if (!h.prefillStepSizeUserSet) {
        h.prefillStepSize = rapidMlxPrefillStepSizeDefault(wizardState.model.rapidMlxProfile);
        if (dom.prefillStepSizeSelect) {
            dom.prefillStepSizeSelect.value = String(h.prefillStepSize);
        }
    }
    applyReasoningModeLock();
  window.scheduleVramUpdate?.();
}

export function bindRapidMlxAdvancedControls() {
  const scheduleVramUpdate = window.scheduleVramUpdate || (() => {});

  const bindSel = (el, key) => {
    if (!el || el.dataset.bound) return;
    el.dataset.bound = '1';
    el.addEventListener('change', () => {
      wizardState.hardware[key] = el.value || '';
      scheduleVramUpdate();
    });
  };

   bindSel(dom.kvCacheDtypeSelect, 'kvCacheDtype');
  bindSel(dom.turboquantModeSelect, 'turboquantMode');
  bindSel(dom.gpuMemoryUtilizationSelect, 'gpuMemoryUtilization');
  bindSel(dom.maxNumSeqsSelect, 'maxNumSeqs');
  bindSel(dom.maxConcurrentRequestsSelect, 'maxConcurrentRequests');
  bindSel(dom.pflashPolicySelect, 'pflashPolicy');
  bindSel(dom.hybridCacheEntriesSelect, 'hybridCacheEntries');
  bindSel(dom.prefillBatchSizeSelect, 'prefillBatchSize');
  bindSel(dom.completionBatchSizeSelect, 'completionBatchSize');
  [
    dom.turboquantModeSelect,
    dom.pflashPolicySelect,
    dom.maxNumSeqsSelect,
  ].forEach((el) => el && el.addEventListener('change', renderRapidExclusionWarnings));
  bindSel(dom.retainedCacheMibSelect, 'retainedCacheMib');
  bindSel(dom.rapidCacheModeSelect, 'cacheMode');
  dom.rapidCacheModeSelect?.addEventListener('change', () => {
    const custom = dom.rapidCacheModeSelect.value === 'custom';
    const ramWrap = document.getElementById('spawn-retained-cache-mib-wrap');
    const entriesWrap = document.getElementById('spawn-rapid-hybrid-cache-entries-wrap');
    if (ramWrap) ramWrap.style.display = custom ? '' : 'none';
    if (entriesWrap) entriesWrap.style.display = custom ? '' : 'none';
  });
   // workloadScenario is derived from page-1 use-case selection
   bindSel(dom.samplingModeSelect, 'samplingMode');
  bindSel(dom.toolCallParserSelect, 'toolCallParser');
  bindSel(dom.reasoningParserSelect, 'reasoningParser');
  bindSel(dom.hybridModeSelect, 'hybridMode');
    bindSel(dom.prefillStepSizeSelect, 'prefillStepSize');
    dom.prefillStepSizeSelect?.addEventListener('change', () => {
        wizardState.hardware.prefillStepSizeUserSet = true;
    });
  bindSel(dom.speculativeSourceSelect, 'speculativeSource');
  bindSel(dom.speculativeTokensSelect, 'speculativeTokens');

   // Web UI config JSON and static path inputs
   const bindInput = (el, key) => {
     if (!el || el.dataset.bound) return;
     el.dataset.bound = '1';
     el.addEventListener('input', () => {
       wizardState.hardware[key] = el.value || '';
       scheduleVramUpdate();
     });
   };
bindInput(dom.speculativeModelInput, 'speculativeModel');
if (dom.speculativeModelInput && !dom.speculativeModelInput.dataset.sidecarOverrideBound) {
      dom.speculativeModelInput.dataset.sidecarOverrideBound = '1';
      dom.speculativeModelInput.addEventListener('input', () => {
        wizardState.hardware.speculativeModelAutoSelected = false;
      });
    }
    if (dom.speculativeModelInput && !dom.speculativeModelInput.dataset.trustBound) {
      dom.speculativeModelInput.dataset.trustBound = '1';
      (function() {
        let t = null;
        dom.speculativeModelInput.addEventListener('input', () => {
          const v = (dom.speculativeModelInput.value || '').trim();
          if (t) clearTimeout(t);
          if (!v) {
            _hideSpawnTrust();
            return;
          }
          t = setTimeout(() => _spawnCheckTrust(v), 600);
        });
      })();
    }

   const bindCheck = (el, key, onChange) => {
     if (!el || el.dataset.bound) return;
     el.dataset.bound = '1';
     el.addEventListener('change', () => {
       wizardState.hardware[key] = el.checked;
       onChange?.();
       scheduleVramUpdate();
     });
   };
   bindCheck(dom.speculativeEnabledCheck, 'speculativeEnabled', syncRapidSpeculativeFields);
   bindCheck(dom.speculativeDisableAutoKCheck, 'speculativeDisableAutoK');
   bindCheck(dom.autoToolChoiceCheck, 'autoToolChoice', () => {
     wizardState.hardware.autoToolChoiceTouched = true;
   });
   dom.speculativeSourceSelect?.addEventListener('change', syncRapidSpeculativeFields);

    if (dom.reasoningModeCheck && !dom.reasoningModeCheck.dataset.bound) {
     dom.reasoningModeCheck.dataset.bound = '1';
     dom.reasoningModeCheck.addEventListener('change', () => {
       wizardState.hardware.rapidReasoningMode = dom.reasoningModeCheck.checked ? 'on' : 'off';
       applyReasoningModeLock();
       scheduleVramUpdate();
     });
   }

  // Re-check pin button handler for spawn wizard
  dom.speculativeRecheckBtn?.addEventListener('click', _spawnRecheckTrustPin);
}

// ── Rapid-MLX live model profile (from rapid-mlx info <model>) ────────────────

let _rapidMlxProfileTimer = null;

// Render warnings from the unified profile (source conflicts, missing data).
function _renderUnifiedProfileWarnings(unified) {
  const hintsEl = document.getElementById('rapid-mlx-profile-hints');
  if (!hintsEl || !unified.warnings || unified.warnings.length === 0) return;

  for (const warning of unified.warnings) {
    const row = document.createElement('div');
    row.className = 'rapid-mlx-hint-row rapid-mlx-hint-row--warning';
    row.textContent = `⚠ ${warning}`;
    row.style.color = 'var(--accent-warning, #f59e0b)';
    row.style.fontSize = '0.82em';
    hintsEl.appendChild(row);
  }
}

// Apply recommendations from the unified profile to wizard controls.
// Only sets values if the user hasn't explicitly overridden them.
function _applyUnifiedProfileRecommendations() {
  const unified = wizardState.model.rapidMlxUnifiedProfile;
  if (!unified || !unified.recommended) return;

  const rec = unified.recommended;

  // Apply hybrid_mode recommendation if not already set by user
  if (rec.hybrid_mode && wizardState.hardware.hybridMode == null) {
    wizardState.hardware.hybridMode = rec.hybrid_mode;
    if (dom.hybridModeSelect) {
      dom.hybridModeSelect.value = rec.hybrid_mode;
    }
  }

  // Apply tool_format recommendation if not already set by user
  if (rec.tool_format && wizardState.hardware.toolCallParser == null) {
    wizardState.hardware.toolCallParser = rec.tool_format;
    if (dom.toolCallParserInput) {
      dom.toolCallParserInput.value = rec.tool_format;
    }
  }

  // Apply reasoning_parser recommendation if not already set by user
  if (rec.reasoning_parser && wizardState.hardware.reasoningParser == null) {
    wizardState.hardware.reasoningParser = rec.reasoning_parser;
    if (dom.reasoningParserInput) {
      dom.reasoningParserInput.value = rec.reasoning_parser;
    }
  }

  // Show warnings from unified profile
  _renderUnifiedProfileWarnings(unified);
}

// "Detected:" spans are tri-state, not just profile.field || 'unknown':
// - 'Detecting…' while a profile fetch is in flight
// - 'Not reported by this model' when the fetch completed but the field is absent
// - the literal API value otherwise (including a real 'unknown' if rapid-mlx ever sends one)
function _setDetectedText(elId, state, value) {
  const el = document.getElementById(elId);
  if (!el) return;
  if (state === 'detecting') {
    el.textContent = 'Detecting…';
  } else {
    el.textContent = value || 'Not reported by this model';
  }
}

export function markRapidMlxDetectionInFlight() {
  _setDetectedText('spawn-rapid-tool-parser-detected', 'detecting');
  _setDetectedText('spawn-rapid-reasoning-parser-detected', 'detecting');
}

// Render contextual hints from the live Rapid-MLX profile.
// Only renders when on the Rapid-MLX hardware panel and a profile is available.
function _renderRapidMlxProfileHints() {
  const profile = wizardState.model.rapidMlxProfile;
  _setDetectedText('spawn-rapid-tool-parser-detected', 'done', profile?.tool_format);
  _setDetectedText('spawn-rapid-reasoning-parser-detected', 'done', profile?.reasoning_parser);

  const hintsEl = document.getElementById('rapid-mlx-profile-hints');
  if (!hintsEl) return;
  if (!profile) {
    hintsEl.style.display = 'none';
    return;
  }
  hintsEl.style.display = '';
  setHtml(hintsEl, '');

    const hasVision = rapidMlxProfileHasVision(profile);
  const hasEmbeddings = profile.extras && profile.extras.embeddings;
  if (profile.reasoning_parser && wizardState.hardware.rapidReasoningMode == null) {
    wizardState.hardware.rapidReasoningMode = 'on';
    if (dom.reasoningModeCheck) dom.reasoningModeCheck.checked = true;
    applyReasoningModeLock();
    (window.scheduleVramUpdate || (() => {}))();
  }

  // A model that reports a tool-call parser is by definition compatible, so the
  // safe-by-default toggle can flip on automatically. Only when the user hasn't
  // expressed a choice yet (null sentinel from bindCheck's change tracking).
  if (profile.tool_format && !wizardState.hardware.autoToolChoice && !wizardState.hardware.autoToolChoiceTouched) {
    wizardState.hardware.autoToolChoice = true;
    if (dom.autoToolChoiceCheck) dom.autoToolChoiceCheck.checked = true;
  }

  // Tool format + reasoning parser row
  if (profile.tool_format || profile.reasoning_parser) {
    const row = document.createElement('div');
    row.className = 'rapid-mlx-hint-row';
    const parts = [];
    if (profile.tool_format) parts.push(`Tool: ${profile.tool_format}`);
    if (profile.reasoning_parser) parts.push(`Reasoning: ${profile.reasoning_parser}`);
    row.textContent = parts.join(' · ');
    hintsEl.appendChild(row);
  }

  // Spec-decode eligibility readout. Configuration remains hidden until the typed,
  // MTP-only vLLM-style schema and truthful admission UX are wired end to end.
  if (profile.spec_decode) {
    const row = document.createElement('div');
    row.className = 'rapid-mlx-hint-row rapid-mlx-hint-row--spec';
    const spec = profile.spec_decode;
    if (spec === 'supported') {
      row.textContent = 'Speculative decoding: eligible';
    } else if (spec === 'unsupported') {
      row.textContent = 'Speculative decoding: not eligible';
    } else {
      row.textContent = 'Speculative decoding: unknown';
    }
    hintsEl.appendChild(row);
  }

  // Answer "does this model have MTP, and embedded or sidecar?" straight from the
  // runtime profile, next to the toggle, instead of leaving the user to guess.
  const mtpEligibility = document.getElementById('spawn-rapid-mtp-eligibility');
  if (mtpEligibility) {
    const hasEmbeddedHeads = !!(profile.extras && (profile.extras.mtp || profile.extras.mtp_dflash))
      || profile.spec_decode === 'supported';
    if (profile.spec_decode === 'supported' || hasEmbeddedHeads) {
      mtpEligibility.textContent = 'This model reports embedded MTP prediction heads — “Embedded prediction heads” below needs no sidecar.';
      mtpEligibility.style.display = '';
    } else if (profile.spec_decode === 'unsupported') {
      mtpEligibility.textContent = 'No embedded MTP heads detected for this model. Speculation needs a matching local sidecar (build one from the model’s base checkpoint) or should stay off.';
      mtpEligibility.style.display = '';
    } else {
      mtpEligibility.style.display = 'none';
    }
  }

  // DFlash / DDTree eligibility
  const renderEligibility = (label, elig) => {
    if (!elig) return;
    const row = document.createElement('div');
    row.className = 'rapid-mlx-hint-row rapid-mlx-hint-row--eligibility';
    const status = elig.supported === true ? 'eligible' : elig.supported === false ? 'not eligible' : 'unknown';
    row.textContent = `${label}: ${status}`;
    if (elig.reasons && Object.keys(elig.reasons).length > 0) {
      const details = Object.entries(elig.reasons)
        .map(([k, v]) => `${k}: ${v || '—'}`)
        .join(' · ');
      row.title = details;
      const dot = document.createElement('span');
      dot.className = 'rapid-mlx-hint-dot';
      dot.textContent = ' ⓘ';
      row.appendChild(dot);
    }
    hintsEl.appendChild(row);
  };
  if (profile.dflash_eligibility) renderEligibility('DFlash', profile.dflash_eligibility);
  if (profile.ddtree_eligibility) renderEligibility('DDTree', profile.ddtree_eligibility);

  // MTP path
  if (profile.mtp_path && profile.mtp_path !== 'unknown') {
    const row = document.createElement('div');
    row.className = 'rapid-mlx-hint-row';
    row.textContent = `MTP path: ${profile.mtp_path}`;
    hintsEl.appendChild(row);
  }

    // Auto leaves Rapid's model-specific MLLM detection in control. The user
    // can only force the safe text lane here; force-on needs model evidence.
    if (hasVision) {
        if (!wizardState.hardware.prefillStepSizeUserSet) {
            wizardState.hardware.prefillStepSize = rapidMlxPrefillStepSizeDefault(profile);
            if (dom.prefillStepSizeSelect) {
                dom.prefillStepSizeSelect.value = String(wizardState.hardware.prefillStepSize);
            }
            window.scheduleVramUpdate?.();
        }
        const configRow = document.createElement('div');
    configRow.className = 'rapid-mlx-config-row';
    const configLabel = document.createElement('span');
    configLabel.className = 'rapid-mlx-config-label';
    configLabel.textContent = 'Vision mode';
    configRow.appendChild(configLabel);
    const toggleLabel = document.createElement('label');
    toggleLabel.className = 'rapid-mlx-config-toggle';
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = true;
    checkbox.addEventListener('change', () => {
      wizardState.model.rapidMlxMllm = checkbox.checked;
    });
    toggleLabel.appendChild(checkbox);
    const toggleText = document.createElement('span');
    toggleText.textContent = 'Use model vision when qualified';
    toggleLabel.appendChild(toggleText);
    configRow.appendChild(toggleLabel);
    hintsEl.appendChild(configRow);
  }

  // Embeddings value input (--embedding-model <repo>) — shown only when embeddings extra present
  if (hasEmbeddings) {
    const configRow = document.createElement('div');
    configRow.className = 'rapid-mlx-config-row';
    const configLabel = document.createElement('span');
    configLabel.className = 'rapid-mlx-config-label';
    configLabel.textContent = '--embedding-model';
    configRow.appendChild(configLabel);
    const input = document.createElement('input');
    input.className = 'rapid-mlx-config-input';
    input.type = 'text';
    input.placeholder = 'Repo ID or alias for embedding model';
    input.addEventListener('input', () => {
      wizardState.model.rapidMlxEmbeddingModel = input.value || null;
    });
    configRow.appendChild(input);
    hintsEl.appendChild(configRow);
  }

  // Extras tag row (always show extras summary)
  if (profile.extras) {
    const row = document.createElement('div');
    row.className = 'rapid-mlx-hint-row rapid-mlx-hint-row--extras';
    const tags = [];
    if (hasVision) tags.push('vision');
    if (hasEmbeddings) tags.push('embeddings');
    if (profile.extras.mtp_dflash) tags.push('mtp-dflash');
    if (tags.length > 0) {
      row.textContent = `Extras: ${tags.join(', ')}`;
      hintsEl.appendChild(row);
    }
  }
}

// Fetch the live model profile from rapid-mlx info and store it in wizardState.
// Only runs when Rapid-MLX engine is selected and a model identity is known.
// Non-blocking: failures degrade gracefully without stopping wizard flow.
async function _fetchRapidMlxModelProfile(modelId) {
  if (!modelId || modelId.trim().length < 2) {
    wizardState.model.rapidMlxProfile = null;
    wizardState.model.rapidMlxUnifiedProfile = null;
    wizardState.arch.metadataStatus = 'unknown';
    wizardState.arch.metadataReason = 'Rapid-MLX model identity is missing';
    return;
  }
  if (wizardState.engine.selected !== 'rapid_mlx') {
    wizardState.model.rapidMlxProfile = null;
    wizardState.model.rapidMlxUnifiedProfile = null;
    wizardState.arch.metadataStatus = 'unknown';
    wizardState.arch.metadataReason = 'Rapid-MLX is not selected';
    return;
  }
  try {
    const headers = window.authHeaders ? window.authHeaders() : {};
    const [profileUrl, unifiedUrl] = [
      `/api/rapid-mlx/models/${encodeURIComponent(modelId)}/profile`,
      `/api/rapid-mlx/models/${encodeURIComponent(modelId)}/unified-profile`,
    ];

    const [profileRes, unifiedRes] = await Promise.all([
      fetch(profileUrl, { headers }).catch(() => null),
      fetch(unifiedUrl, { headers }).catch(() => null),
    ]);

    if (profileRes && profileRes.ok) {
      const data = await profileRes.json().catch(() => ({}));
      wizardState.model.rapidMlxProfile = data.profile || null;
    } else {
      wizardState.model.rapidMlxProfile = null;
    }

    if (unifiedRes && unifiedRes.ok) {
      const data = await unifiedRes.json().catch(() => ({}));
      wizardState.model.rapidMlxUnifiedProfile = data.profile || null;
      _applyUnifiedProfileRecommendations();
    } else {
      wizardState.model.rapidMlxUnifiedProfile = null;
    }

    const hasEvidence = !!wizardState.model.rapidMlxProfile || !!wizardState.model.rapidMlxUnifiedProfile;
    wizardState.arch.metadataStatus = hasEvidence ? 'resolved' : 'degraded';
    wizardState.arch.metadataReason = hasEvidence
      ? 'Rapid-MLX model profile'
      : 'Rapid-MLX profile unavailable; safe defaults retained';

    _renderRapidMlxProfileHints();
  } catch {
    wizardState.model.rapidMlxProfile = null;
    wizardState.model.rapidMlxUnifiedProfile = null;
    wizardState.arch.metadataStatus = 'degraded';
    wizardState.arch.metadataReason = 'Rapid-MLX profile request failed; safe defaults retained';
    _renderRapidMlxProfileHints();
  }
}

// Official upstream MTP draft repos, keyed by trunk tier. Upstream publishes a
// standalone MTP drafter per Qwen tier (rapid_mlx/aliases.json), so a finetune of
// that tier can preflight the official draft instead of building one from scratch.
// Extract: tier regex, upstream draft repo id.
const OFFICIAL_MTP_DRAFTS = [
    { match: /qwen3\.8.*27b/i, repo: 'rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX', note: 'a bf16 variant (…-MTP-fp16-MLX) also exists' },
    { match: /qwen3\.6.*35b/i, repo: 'mlx-community/Qwen3.6-35B-A3B-MTP-4bit', note: '' },
    { match: /qwen3\.6.*27b/i, repo: 'mlx-community/Qwen3.6-27B-MTP-4bit', note: '' },
    { match: /qwen3\.5.*9b/i, repo: 'mlx-community/Qwen3.5-9B-MTP-4bit', note: '' },
    { match: /qwen3\.5.*4b/i, repo: 'mlx-community/Qwen3.5-4B-MTP-4bit', note: '' },
];

function officialMtpDraftForTrunk(trunkPath) {
    const name = String(trunkPath || '').toLowerCase();
    for (const entry of OFFICIAL_MTP_DRAFTS) {
        if (entry.match.test(name)) return entry;
    }
    return null;
}

// Debounced wrapper: schedule a profile fetch after model selection stabilizes.
export function scheduleRapidMlxProfileFetch(modelId) {
  if (modelId && modelId.trim().length >= 2 && wizardState.engine.selected === 'rapid_mlx') {
    markRapidMlxDetectionInFlight();
  }
  clearTimeout(_rapidMlxProfileTimer);
  _rapidMlxProfileTimer = setTimeout(() => {
    _fetchRapidMlxModelProfile(modelId);
  }, 350);
}

// Builds the `rapid_mlx` branch of the spawn payload. Extracted from
// buildSpawnPayload() in spawn-wizard-spawn.js as a behavior-preserving move:
// same field names, same shape. h/m are wizardState.hardware/wizardState.model.
export function buildRapidMlxConfig(h, m) {
  const preservedSource = m.rapidMlxSource || m.localMeta?.model_source || null;
  const modelSource = preservedSource || (m.source === 'hf'
    ? { kind: 'hugging_face_repo', repo_id: m.hfRepo || '', revision: 'main' }
    : { kind: 'mlx_directory', path: m.path || '' });
  const escapeHatchFlags = Object.entries(h.escapeHatchFlags || {})
    .filter(([_, v]) => v !== null && v !== undefined && v !== '' && !(typeof v === 'boolean' && !v))
    .map(([k, v]) => [k, v]);
  return {
    model_source: modelSource,
    served_model_name: h.alias || null,
    host: wizardState.access.bindHost || '127.0.0.1',
    port: wizardState.access.port || 8001,
    api_key: wizardState.access.apiKey || null,
    ...(h.enableThinking != null && { enable_thinking: h.enableThinking }),
    ...(h.toolCallParser && { tool_call_parser: h.toolCallParser }),
    ...(h.reasoningParser && { reasoning_parser: h.reasoningParser }),
    auto_tool_choice: !!h.autoToolChoice,
    no_thinking: h.rapidReasoningMode === 'off',
    hybrid_mode: h.hybridMode || 'auto',
        prefill_step_size: Number(h.prefillStepSize || rapidMlxPrefillStepSizeDefault(m.rapidMlxProfile)),
    ...(escapeHatchFlags.length > 0 && { escape_hatch_flags: escapeHatchFlags }),
    // Phase 7: KV/cache policy (D6 catalog IDs)
    ...(h.kvCacheDtype && h.kvCacheDtype !== 'int4' && { kv_cache_dtype: h.kvCacheDtype }),
    ...(h.turboquantMode && h.turboquantMode !== 'none' && h.turboquantMode !== 'auto' && { turboquant_mode: h.turboquantMode }),
    // '' means omit: an absent flag and an explicit runtime default are different states.
    ...(h.gpuMemoryUtilization && { gpu_memory_utilization: Number(h.gpuMemoryUtilization) }),
    ...(h.maxNumSeqs && { max_num_seqs: Number(h.maxNumSeqs) }),
    ...(h.maxConcurrentRequests && { max_concurrent_requests: Number(h.maxConcurrentRequests) }),
    ...(h.pflashPolicy && { pflash_policy: h.pflashPolicy }),
    ...(h.prefillBatchSize && { prefill_batch_size: Number(h.prefillBatchSize) }),
    ...(h.completionBatchSize && { completion_batch_size: Number(h.completionBatchSize) }),
    cache_mode: h.cacheMode || 'custom',
    prefix_cache_enabled: Number(h.retainedCacheMib ?? 8192) > 0,
    ...(Number(h.retainedCacheMib ?? 8192) > 0 && {
      retained_cache_mib: Number(h.retainedCacheMib ?? 8192),
      // Entry count only means something while the cache is on, so it rides the same gate.
      ...(Number(h.hybridCacheEntries || 0) > 0 && {
        hybrid_cache_entries: Number(h.hybridCacheEntries),
      }),
    }),
    disk_checkpoint_interval: 0,
    // Phase 9: chat template lifecycle — applied server-side via a generated
    // overlay model directory (create_template_overlay), not a direct CLI flag.
    chat_template_file: wizardState.model.chatTemplatePath || null,
    // No workload_scenario here: `RapidMlxConfig` has no such field, so serde dropped it
    // on arrival and the spawn path never saw it. The scenario reaches the server where
    // it is actually consumed — top level of the `/api/vram` body, sent by
    // vram-estimate.js — and it steers the wizard's own KV-dtype and context choices,
    // which do get sent. Left over from the workload-profile picker removed in 712c261.
    reasoning_mode: h.rapidReasoningMode || 'on',
    ...(h.speculativeEnabled && {
      speculative_config: {
        method: 'mtp',
        ...(h.speculativeSource === 'external' && { model: h.speculativeModel.trim() }),
                        num_speculative_tokens: Number(h.speculativeTokens || RAPID_MLX_DEFAULT_SPECULATIVE_TOKENS),
        disable_auto_k: !!h.speculativeDisableAutoK,
      },
      ...(h.speculativeTrustRequired && h.speculativeTrustConsent &&
        h.speculativeTrustRepoId && h.speculativeTrustRevision && {
          trust_remote_code_consent: h.speculativeTrustRepoId + '@' + h.speculativeTrustRevision
        }
      ),
    }),
    // Auto deliberately omits a flag; Off maps to Rapid's real
    // --no-mllm escape hatch for incomplete vision-tower checkpoints.
    ...(m.rapidMlxMllm === false && { mllm_vision: 'off' }),
    // Phase 7: Web UI (D26/A44)
    // Phase 7: Sampling mode (D27)
    ...(h.samplingMode && h.samplingMode !== 'auto' && { sampling_mode: h.samplingMode }),
    ...(h.temperature != null && { default_temperature: h.temperature }),
    ...(h.topP != null && { default_top_p: h.topP }),
    ...(h.topK != null && { default_top_k: h.topK }),
    ...(h.minP != null && { default_min_p: h.minP }),
    ...(h.repeatPenalty != null && { default_repetition_penalty: h.repeatPenalty }),
    ...(h.presencePenalty != null && { default_presence_penalty: h.presencePenalty }),
    ...(h.maxTokens != null && { max_tokens: h.maxTokens }),
    // Phase 7: Prompt storage (D31) — turboquant_mode already set above
  };
}
