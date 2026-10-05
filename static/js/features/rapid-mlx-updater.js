/* global DOMPurify */

// ── Rapid-MLX Runtime Manager ─────────────────────────────────────────────────
// Manages Rapid-MLX runtime lifecycle in Settings and nav pill.
// Uses existing API endpoints under /api/rapid-mlx/runtime/.

import { showToast } from './toast.js';
import { wireUninstallButton } from './runtime-uninstall.js';
import { setHtml } from '../core/set-html.js';
import { attachModalFocusTrap, detachModalFocusTrap, fetchReleaseList, buildReleaseBadges } from './updater-shared.js';

let _runtimeStatus = null;
let _releases = [];
let _developmentResolution = null;
let _mutationInflight = false;
let _changelogCache = Object.create(null);
const SHA_SHORT_LENGTH = 12;

function activeRuntimeLabel(active) {
  if (!active) return 'No Rapid-MLX runtime installed.';
  if (active.source_kind === 'git' && active.source_commit) {
    const repository = active.source_repository || 'development source';
    return `Rapid-MLX ${repository}@${active.source_commit.slice(0, SHA_SHORT_LENGTH)}`;
  }
  return `Rapid-MLX v${active.version}`;
}

export function initRapidMlxUpdater() {
  // Settings: Manage button
  const manageBtn = document.getElementById('rapid-mlx-manage-btn');
  if (manageBtn) {
    manageBtn.addEventListener('click', openRapidMlxModal);
  }

  // Nav pill (for when there is an active runtime)
  const pill = document.getElementById('rapid-mlx-pill');
  if (pill) {
    pill.addEventListener('click', openRapidMlxModal);
  }
  wireUninstallButton(document.getElementById('rapid-mlx-uninstall-btn'), 'rapid_mlx', {
    onDone: () => fetchRuntimeStatus(),
  });

  const actionRow = document.querySelector('#rapid-mlx-modal .rapid-mlx-modal-actions');
  const officialTab = document.getElementById('rapid-mlx-source-official');
  const developmentTab = document.getElementById('rapid-mlx-source-development');
  if (officialTab) officialTab.addEventListener('click', () => selectRuntimeSource('official'));
  if (developmentTab) developmentTab.addEventListener('click', () => selectRuntimeSource('development'));

  const resolveDevelopmentBtn = document.getElementById('rapid-mlx-development-resolve');
  if (resolveDevelopmentBtn) resolveDevelopmentBtn.addEventListener('click', resolveDevelopmentSource);
  const installDevelopmentBtn = document.getElementById('rapid-mlx-development-install');
  if (installDevelopmentBtn) installDevelopmentBtn.addEventListener('click', installDevelopmentSource);

  if (actionRow && !document.getElementById('rapid-mlx-settings-evidence-btn')) {
    const evidenceBtn = document.createElement('button');
    evidenceBtn.id = 'rapid-mlx-settings-evidence-btn';
    evidenceBtn.type = 'button';
    evidenceBtn.className = 'rapid-mlx-action-btn rapid-mlx-action-btn-soft';
    evidenceBtn.textContent = 'Runtime support';
    evidenceBtn.addEventListener('click', () => openRuntimeSettingsEvidence(evidenceBtn));
    actionRow.appendChild(evidenceBtn);
  }

  // Modal: Upgrade button
  const upgradeBtn = document.getElementById('rapid-mlx-upgrade-btn');
  if (upgradeBtn) {
    upgradeBtn.addEventListener('click', async () => {
      if (_mutationInflight) return;
      const latest = _releases[0];
      if (!latest) { showToast('No release info available.', 'info'); return; }
      const token = await getDbAdminToken();
      if (!token) { showToast('Upgrade requires authentication.', 'error'); return; }
      _mutationInflight = true;
      upgradeBtn.disabled = true;
      upgradeBtn.textContent = 'Upgrading…';
      try {
        const resp = await fetch('/api/rapid-mlx/runtime/upgrade', {
          method: 'POST',
          headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({
          version: latest.version,
          extras: selectedRuntimeExtras(),
          confirm: 'UPGRADE_RAPID_MLX_RUNTIME',
        }),
        });
        if (!resp.ok) {
          const body = await resp.json().catch(() => ({}));
          throw new Error(body.error || 'Upgrade failed');
        }
        const data = await resp.json();
        showToast(`Rapid-MLX upgrade started to v${latest.version}`, 'success');
        if (data.job_id) pollJob(data.job_id);
      } catch (err) {
        showToast(`Rapid-MLX upgrade failed: ${err.message}`, 'error');
        upgradeBtn.disabled = false;
        upgradeBtn.textContent = 'Upgrade';
        _mutationInflight = false;
      }
    });
  }

  // Modal: Repair button
  const repairBtn = document.getElementById('rapid-mlx-repair-btn');
  if (repairBtn) {
    repairBtn.addEventListener('click', async () => {
      if (_mutationInflight) return;
      const token = await getDbAdminToken();
      if (!token) { showToast('Repair requires authentication.', 'error'); return; }
      _mutationInflight = true;
      repairBtn.disabled = true;
      repairBtn.textContent = 'Repairing…';
      try {
        const resp = await fetch('/api/rapid-mlx/runtime/repair', {
          method: 'POST',
          headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
          body: JSON.stringify({ confirm: 'REPAIR_RAPID_MLX_RUNTIME' }),
        });
        if (!resp.ok) {
          const body = await resp.json().catch(() => ({}));
          throw new Error(body.error || 'Repair failed');
        }
        const data = await resp.json();
        showToast('Rapid-MLX repair started', 'success');
        if (data.job_id) pollJob(data.job_id);
      } catch (err) {
        showToast(`Rapid-MLX repair failed: ${err.message}`, 'error');
        repairBtn.disabled = false;
        repairBtn.textContent = 'Repair';
        _mutationInflight = false;
      }
    });
  }

  // Modal: Rollback button
  const rollbackBtn = document.getElementById('rapid-mlx-rollback-btn');
  if (rollbackBtn) {
    rollbackBtn.addEventListener('click', async () => {
      if (_mutationInflight) return;
      const token = await getDbAdminToken();
      if (!token) { showToast('Rollback requires authentication.', 'error'); return; }
      _mutationInflight = true;
      rollbackBtn.disabled = true;
      rollbackBtn.textContent = 'Rolling back…';
      try {
        const resp = await fetch('/api/rapid-mlx/runtime/rollback', {
          method: 'POST',
          headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
          body: JSON.stringify({ confirm: 'ROLLBACK_RAPID_MLX_RUNTIME' }),
        });
        if (!resp.ok) {
          const body = await resp.json().catch(() => ({}));
          throw new Error(body.error || 'Rollback failed');
        }
        const data = await resp.json();
        showToast('Rapid-MLX rollback started', 'success');
        if (data.job_id) pollJob(data.job_id);
      } catch (err) {
        showToast(`Rapid-MLX rollback failed: ${err.message}`, 'error');
        rollbackBtn.disabled = false;
        rollbackBtn.textContent = 'Rollback';
        _mutationInflight = false;
      }
    });
  }

  // Close button
  const closeBtn = document.getElementById('rapid-mlx-modal-close');
  if (closeBtn) closeBtn.addEventListener('click', closeRapidMlxModal);

   const overlay = document.getElementById('rapid-mlx-modal');
   if (overlay) {
     overlay.addEventListener('click', e => {
       if (e.target === overlay) closeRapidMlxModal();
     });
   }

   // Changelog expand/collapse toggle
   const changelogToggle = document.getElementById('rapid-mlx-changelog-toggle');
   if (changelogToggle) {
     changelogToggle.addEventListener('click', toggleChangelog);
   }

   // Initial status load
  setTimeout(() => {
    if (!document.hidden) {
      fetchRuntimeStatus();
      fetchReleases();
    }
  }, 2500);

  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
      fetchRuntimeStatus();
      fetchReleases();
    }
  });
}

async function openRuntimeSettingsEvidence(opener) {
  opener.disabled = true;
  try {
    const response = await fetch('/api/rapid-mlx/settings', {
      headers: window.authHeaders ? window.authHeaders() : {},
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.ok === false) throw new Error(data.error || `Settings catalog failed (${response.status})`);
    const settings = Array.isArray(data.settings) ? data.settings : [];
    const unsupported = settings.filter(setting => setting.unsupported_reason);
    const { openEvidenceDrawer } = await import('./evidence-drawer.js');
    openEvidenceDrawer({
      title: 'Rapid-MLX runtime support',
      status: data.snapshot_source === 'discovered' ? (unsupported.length ? 'caution' : 'good') : 'caution',
      summary: data.snapshot_source === 'discovered'
        ? `${settings.length - unsupported.length} of ${settings.length} catalog settings are supported by the active runtime snapshot.`
        : 'Runtime capabilities could not be discovered, so support is not confirmed.',
      consequence: 'Unsupported settings are omitted or downgraded rather than passed as invalid launch flags.',
      remediation: unsupported.length ? 'Review unsupported settings before launching or update the Rapid-MLX runtime.' : '',
      evidence: settings.filter(setting => !setting.unsupported_reason).map(setting => `${setting.id}: supported`),
      warnings: unsupported.map(setting => `${setting.id}: ${setting.unsupported_reason}`),
      provenance: [
        `Snapshot source: ${data.snapshot_source || 'none'}`,
        data.rapid_mlx_version ? `Rapid-MLX version: ${data.rapid_mlx_version}` : '',
      ].filter(Boolean),
    }, opener);
  } catch (error) {
    showToast(`Could not load runtime support: ${error.message}`, 'error');
  } finally {
    opener.disabled = false;
  }
}

// Dotted-numeric comparison: '0.15.10' > '0.15.9'. Non-numeric suffixes split
// as their own segments, so '0.16.0rc1' sorts after '0.16.0' pieces that match.
function compareVersions(a, b) {
  const pa = String(a).split('.');
  const pb = String(b).split('.');
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    const na = parseInt(pa[i], 10) || 0;
    const nb = parseInt(pb[i], 10) || 0;
    if (na !== nb) return na - nb;
    const sa = (pa[i] || '').replace(/^\d+/, '');
    const sb = (pb[i] || '').replace(/^\d+/, '');
    if (sa !== sb) return sa > sb ? 1 : -1;
  }
  return 0;
}

function updateRapidMlxPill() {
  const pill = document.getElementById('rapid-mlx-pill');
  const pillVer = document.getElementById('rapid-mlx-pill-version');
  const active = _runtimeStatus?.active ?? null;
  if (!pill) return;
  if (_runtimeStatus?.supported && active && active.source_kind !== 'git') {
    const latest = _releases[0]?.version || null;
    const installed = active.version || '';
    const updateAvailable = latest && installed && compareVersions(latest, installed) > 0;
    pill.classList.toggle('top-nav-pill-update', !!updateAvailable);
    if (updateAvailable) {
      if (pillVer) pillVer.textContent = `Rapid-MLX · ↑ v${latest}`;
      pill.title = `Update available: v${installed} → v${latest}. Click to manage.`;
    }
  } else {
    pill.classList.remove('top-nav-pill-update');
  }
}

export async function fetchRuntimeStatus() {
  try {
    const headers = window.authHeaders ? window.authHeaders() : {};
    const resp = await fetch('/api/rapid-mlx/runtime/status', { headers });
    if (!resp.ok) return;
    const data = await resp.json();
    _runtimeStatus = data.runtime || null;

    const supported = _runtimeStatus?.supported ?? false;
    const active = _runtimeStatus?.active ?? null;
    const mutating = _runtimeStatus?.mutation_in_progress ?? false;

    // Update nav pill (if present)
    const pill = document.getElementById('rapid-mlx-pill');
    const pillVer = document.getElementById('rapid-mlx-pill-version');
    if (pill && supported && active && !mutating) {
      pill.style.display = 'flex';
      if (active.source_kind !== 'git') {
        // Update state is applied by updateRapidMlxPill() once releases load;
        // render the plain version here so the pill never shows stale arrows.
        if (pillVer) pillVer.textContent = `Rapid-MLX · v${active.version}`;
        pill.title = `${activeRuntimeLabel(active)}. Click to manage.`;
      } else {
        if (pillVer) pillVer.textContent = `Rapid-MLX · ${active.source_commit?.slice(0, SHA_SHORT_LENGTH) || 'development'}`;
        pill.title = `${activeRuntimeLabel(active)}. Click to manage.`;
      }
    } else if (pill) {
      pill.style.display = 'none';
    }
    updateRapidMlxPill();

    // Update Settings card summary
    updateSettingsSummary();

    // If there's an active job, start polling
    // `jobs` sits beside `runtime` (newest first), not inside it. Resume polling the job that
    // is still going, e.g. after the page was reloaded mid-install.
    const runningJob = (data.jobs || []).find(job => job.state === 'queued' || job.state === 'running');
    if (mutating && runningJob?.id) pollJob(runningJob.id);
  } catch {
    // silent
  }
}

export async function fetchReleases() {
  try {
    const data = await fetchReleaseList('/api/rapid-mlx/runtime/releases');
    _releases = data.releases || [];
    updateRapidMlxPill();
  } catch {
    // silent
  }
}

function selectRuntimeSource(source) {
  const official = source === 'official';
  const officialTab = document.getElementById('rapid-mlx-source-official');
  const developmentTab = document.getElementById('rapid-mlx-source-development');
  const developmentPanel = document.getElementById('rapid-mlx-development-source');
  const releasesList = document.getElementById('rapid-mlx-releases-list');
  const releasesPane = releasesList?.closest('.rapid-mlx-pane-list');
  const notesPane = document.querySelector('#rapid-mlx-modal .rapid-mlx-pane-notes');

  officialTab?.classList.toggle('active', official);
  officialTab?.setAttribute('aria-selected', String(official));
  developmentTab?.classList.toggle('active', !official);
  developmentTab?.setAttribute('aria-selected', String(!official));
  if (developmentPanel) developmentPanel.hidden = official;
  if (releasesPane) releasesPane.style.display = official ? '' : 'none';
  if (notesPane) notesPane.style.display = official ? '' : 'none';
}

function developmentRequest(includeConfirmation = false) {
  const repository = document.getElementById('rapid-mlx-development-repository')?.value.trim() || '';
  const reference = document.getElementById('rapid-mlx-development-reference')?.value.trim() || '';
  return {
    repository,
    reference,
    resolved_commit: _developmentResolution?.resolved_commit || null,
    extras: selectedRuntimeExtras(),
    ...(includeConfirmation ? { confirm: 'INSTALL_RAPID_MLX_DEVELOPMENT' } : { confirm: '' }),
  };
}

function renderDevelopmentResolution(message, kind = '') {
  const statusEl = document.getElementById('rapid-mlx-development-resolution');
  const installBtn = document.getElementById('rapid-mlx-development-install');
  if (!statusEl) return;
  statusEl.className = `rapid-mlx-development-resolution${kind ? ` ${kind}` : ''}`;
  statusEl.textContent = message;
  if (installBtn) installBtn.disabled = !_developmentResolution || _mutationInflight;
}

async function resolveDevelopmentSource() {
  if (_mutationInflight) return;
  const resolveBtn = document.getElementById('rapid-mlx-development-resolve');
  if (resolveBtn) {
    resolveBtn.disabled = true;
    resolveBtn.textContent = 'Resolving…';
  }
  _developmentResolution = null;
  renderDevelopmentResolution('Resolving the reference to an immutable commit…');

  try {
    const response = await fetch('/api/rapid-mlx/runtime/development/resolve', {
      method: 'POST',
      headers: {
        ...(window.authHeaders ? window.authHeaders() : {}),
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(developmentRequest()),
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok || !body.source) throw new Error(body.error || 'Could not resolve development source');
    _developmentResolution = body.source;
    const source = body.source;
    const title = source.title ? ` · ${source.title}` : '';
    const base = source.base_commit ? ` · base ${source.base_commit.slice(0, SHA_SHORT_LENGTH)}` : '';
    renderDevelopmentResolution(
      `Resolved ${source.repository}@${source.requested_ref} → ${source.resolved_commit}${title}${base}`,
      'is-success'
    );
  } catch (error) {
    renderDevelopmentResolution(error.message, 'is-error');
  } finally {
    if (resolveBtn) {
      resolveBtn.disabled = false;
      resolveBtn.textContent = 'Resolve revision';
    }
  }
}

async function installDevelopmentSource() {
  if (_mutationInflight || !_developmentResolution) return;
  const installBtn = document.getElementById('rapid-mlx-development-install');
  _mutationInflight = true;
  if (installBtn) {
    installBtn.disabled = true;
    installBtn.textContent = 'Installing…';
  }

  const token = await getDbAdminToken();
  if (!token) {
    showToast('Rapid-MLX development install requires authentication.', 'error');
    _mutationInflight = false;
    if (installBtn) {
      installBtn.disabled = false;
      installBtn.textContent = 'Install and activate';
    }
    return;
  }

  try {
    const response = await fetch('/api/rapid-mlx/runtime/development/install', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(developmentRequest(true)),
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.error || 'Development install failed');
    showToast(`Rapid-MLX ${_developmentResolution.resolved_commit.slice(0, SHA_SHORT_LENGTH)} install started`, 'success');
    if (body.job_id) pollJob(body.job_id);
  } catch (error) {
    showToast(`Rapid-MLX development install failed: ${error.message}`, 'error');
    _mutationInflight = false;
    if (installBtn) {
      installBtn.disabled = false;
      installBtn.textContent = 'Install and activate';
    }
  }
}

function updateSettingsSummary() {
  const summaryEl = document.getElementById('rapid-mlx-runtime-summary');
  if (!summaryEl) return;

  if (!_runtimeStatus) {
    summaryEl.textContent = 'Status unavailable.';
    return;
  }

  const supported = _runtimeStatus.supported ?? false;
  const active = _runtimeStatus.active ?? null;
  const mutating = _runtimeStatus.mutation_in_progress ?? false;
  const rollbackAvail = _runtimeStatus.rollback_available ?? false;

  if (!supported) {
    summaryEl.textContent = 'Rapid-MLX is only available on Apple Silicon macOS.';
    return;
  }

  if (mutating) {
    summaryEl.textContent = 'Runtime operation in progress…';
    return;
  }

  if (active) {
    const note = rollbackAvail ? ' · Rollback available' : '';
    summaryEl.textContent = `${activeRuntimeLabel(active)} active${note}.`;
  } else {
    summaryEl.textContent = 'No Rapid-MLX runtime installed.';
  }
}

// ── Modal ────────────────────────────────────────────────────────────────────

export async function openRapidMlxModal() {
  if (_mutationInflight) return;

  await Promise.all([fetchRuntimeStatus(), fetchReleases()]);

  const modal = document.getElementById('rapid-mlx-modal');
  if (!modal) return;
  modal.classList.add('open');

  // Move focus to close button for accessibility.
  const closeBtn = document.getElementById('rapid-mlx-modal-close');
  if (closeBtn) closeBtn.focus();

  // Attach Escape key and tab-scope handlers.
  attachModalFocusTrap(modal, closeRapidMlxModal);

  renderRapidMlxModal();
}

function renderRapidMlxModal() {
  const supported = _runtimeStatus?.supported ?? false;
  const active = _runtimeStatus?.active ?? null;
  const mutating = _runtimeStatus?.mutation_in_progress ?? false;
  const rollbackAvail = _runtimeStatus?.rollback_available ?? false;
  const profileEl = document.getElementById('rapid-mlx-profile');
  if (profileEl) {
    profileEl.style.display = (supported && !mutating) ? '' : 'none';
    syncRuntimeProfile(active);
  }

  // Header: version info
  const statusEl = document.getElementById('rapid-mlx-status-text');
  if (statusEl) {
    if (!supported) {
      statusEl.textContent = 'Rapid-MLX is only available on Apple Silicon macOS.';
    } else if (mutating) {
      statusEl.textContent = 'Runtime operation in progress…';
    } else if (active) {
      statusEl.textContent = `${activeRuntimeLabel(active)} is active.`;
    } else {
      statusEl.textContent = 'No Rapid-MLX runtime installed.';
    }
  }

  // Action buttons
  const upgradeBtn = document.getElementById('rapid-mlx-upgrade-btn');
  const repairBtn = document.getElementById('rapid-mlx-repair-btn');
  const rollbackBtn = document.getElementById('rapid-mlx-rollback-btn');

  if (upgradeBtn) {
    upgradeBtn.style.display = (supported && active && active.source_kind !== 'git' && !mutating) ? '' : 'none';
  }
  if (repairBtn) {
    repairBtn.style.display = (supported && active && !mutating) ? '' : 'none';
  }
  if (rollbackBtn) {
    rollbackBtn.style.display = (supported && rollbackAvail && !mutating) ? '' : 'none';
  }

  selectRuntimeSource('official');
  _developmentResolution = null;
  renderDevelopmentResolution('No development revision resolved.');

  // Render releases list
  const listEl = document.getElementById('rapid-mlx-releases-list');
  if (listEl && supported && !mutating) {
    listEl.replaceChildren();
    const latestRelease = _releases[0] ?? null;

    _releases.slice(0, 10).forEach(release => {
      const isCurrent = active?.version === release.version;
      const isLatest = latestRelease?.version === release.version;
      const row = buildReleaseRow(release, isCurrent, isLatest);
      listEl.appendChild(row);
    });
  }
}

function closeRapidMlxModal() {
  const modal = document.getElementById('rapid-mlx-modal');
  if (modal) modal.classList.remove('open');
  detachModalFocusTrap(modal);
  resetChangelogSection();
}

function resetChangelogSection() {
  const changelogEl = document.getElementById('rapid-mlx-changelog');
  const bodyEl = document.getElementById('rapid-mlx-changelog-body');
  const toggleBtn = document.getElementById('rapid-mlx-changelog-toggle');
  if (changelogEl) changelogEl.style.display = 'none';
  if (bodyEl) {
    bodyEl.style.display = 'none';
    bodyEl.replaceChildren();
  }
  if (toggleBtn) toggleBtn.setAttribute('aria-expanded', 'false');
}

function selectReleaseRow(row) {
  document.querySelectorAll('.rapid-mlx-release-row--selected').forEach(el => el.classList.remove('rapid-mlx-release-row--selected'));
  row.classList.add('rapid-mlx-release-row--selected');
}

function selectedRuntimeExtras() {
  const inputs = ['guided', 'vision', 'embeddings', 'audio'].map(extra => document.getElementById(`rapid-mlx-extra-${extra}`));
  if (!inputs.some(Boolean)) {
    return Array.isArray(_runtimeStatus?.active?.extras)
      ? _runtimeStatus.active.extras
      : ['guided', 'vision'];
  }
  return ['guided', 'vision', 'embeddings', 'audio'].filter(extra => {
    const input = document.getElementById(`rapid-mlx-extra-${extra}`);
    return input?.checked === true;
  });
}

function syncRuntimeProfile(active) {
  const defaults = active ? (Array.isArray(active.extras) ? active.extras : []) : ['guided', 'vision'];
  ['guided', 'vision', 'embeddings', 'audio'].forEach(extra => {
    const input = document.getElementById(`rapid-mlx-extra-${extra}`);
    if (input) input.checked = defaults.includes(extra);
  });
}

function buildReleaseRow(release, isCurrent, isLatest) {
  const row = document.createElement('div');
  row.className = 'rapid-mlx-release-row';
  row.dataset.version = release.version;
  row.title = 'Click to view release details';
  row.style.cursor = 'pointer';
  row.addEventListener('click', e => {
    if (e.target.closest('.rapid-mlx-install-btn')) return;
    selectReleaseRow(row);
    showReleaseNotes(release);
  });

  const info = document.createElement('div');
  info.className = 'rapid-mlx-release-row-info';

  const ver = document.createElement('span');
  ver.className = 'rapid-mlx-release-row-ver';
  ver.textContent = `v${release.version}`;

  const badges = buildReleaseBadges({
    wrapperClass: 'rapid-mlx-release-row-badges',
    badgeClass: 'rapid-mlx-badge',
    isLatest,
    isCurrent,
  });

  const meta = document.createElement('span');
  meta.className = 'rapid-mlx-release-row-meta';
  if (release.channel && release.channel !== 'stable') {
    meta.textContent = release.channel;
  } else {
    const ago = timeAgo(release.published_at);
    meta.textContent = ago || '';
  }

  info.append(ver, badges);

  const right = document.createElement('div');
  right.className = 'rapid-mlx-release-row-right';
  right.appendChild(meta);

  if (!isCurrent) {
    const btn = document.createElement('button');
    btn.className = 'rapid-mlx-install-btn';
    btn.textContent = 'Install';
    btn.dataset.version = release.version;
    btn.addEventListener('click', () => installVersion(btn, release));
    right.appendChild(btn);
  }

  row.append(info, right);
  return row;
}

// ── Actions (use db-admin-token for mutations) ──────────────────────────────

async function getDbAdminToken() {
  try {
    const tokenResp = await fetch('/api/db/admin-token', {
      headers: window.authHeaders ? window.authHeaders() : {},
    });
    if (!tokenResp.ok) return null;
    const data = await tokenResp.json().catch(() => ({}));
    return data.token || null;
  } catch {
    // Callers use a null token to restore their mutation controls. Never let a
    // network/authentication failure strand the global mutation guard.
    return null;
  }
}

async function installVersion(btn, release) {
  if (_mutationInflight) return;
  _mutationInflight = true;

  btn.disabled = true;
  btn.textContent = 'Installing…';

  const token = await getDbAdminToken();
  if (!token) {
    showToast('Rapid-MLX install requires authentication.', 'error');
    btn.disabled = false;
    btn.textContent = 'Install';
    _mutationInflight = false;
    return;
  }

  const actionName = _runtimeStatus?.active ? 'upgrade' : 'install';

  try {
    const resp = await fetch(
      `/api/rapid-mlx/runtime/${actionName}`,
      {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          version: release.version,
          channel: release.channel || 'stable',
          extras: selectedRuntimeExtras(),
          confirm: `${actionName.toUpperCase()}_RAPID_MLX_RUNTIME`,
        }),
      }
    );

    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new Error(body.error || 'Install failed');
    }

    const data = await resp.json();
    const jobId = data.job_id;
    showToast(`Rapid-MLX install started for v${release.version}`, 'success');
    if (jobId) pollJob(jobId);
  } catch (err) {
    showToast(`Rapid-MLX install failed: ${err.message}`, 'error');
    btn.disabled = false;
    btn.textContent = 'Install';
    _mutationInflight = false;
  }
}

const POLL_JOB_INTERVAL_MS = 2000;
const POLL_JOB_MAX_CONSECUTIVE_ERRORS = 5;
const POLL_JOB_MAX_DURATION_MS = 10 * 60 * 1000; // 10 minutes

// Re-read status and releases, and redraw the manager if it is open, so a finished (or failed)
// operation is reflected immediately: the version pill, the header, and the release buttons.
async function refreshRuntimeUi() {
  await Promise.all([fetchRuntimeStatus(), fetchReleases()]);
  if (document.getElementById('rapid-mlx-modal')?.classList.contains('open')) {
    renderRapidMlxModal();
  }
}

function showRuntimeProgress(job, startedAt) {
  const el = document.getElementById('rapid-mlx-status-text');
  if (!el) return;
  const seconds = Math.round((Date.now() - startedAt) / 1000);
  el.textContent = `${job.message || 'Working'} (${seconds}s)`;
}

async function pollJob(jobId) {
  if (!jobId) return;
  const startedAt = Date.now();
  let consecutiveErrors = 0;

  const giveUp = (interval, message) => {
    clearInterval(interval);
    _mutationInflight = false;
    showToast('Rapid-MLX operation status unknown', 'error', message);
    refreshRuntimeUi();
  };

  const interval = setInterval(async () => {
    if (Date.now() - startedAt > POLL_JOB_MAX_DURATION_MS) {
      giveUp(interval, 'Timed out waiting for the operation to finish. Check server logs.');
      return;
    }

    try {
      const headers = window.authHeaders ? window.authHeaders() : {};
      const resp = await fetch(`/api/rapid-mlx/runtime/jobs/${jobId}`, { headers });
      if (!resp.ok) {
        consecutiveErrors += 1;
        if (consecutiveErrors >= POLL_JOB_MAX_CONSECUTIVE_ERRORS) {
          giveUp(interval, `Repeated failures checking job status (HTTP ${resp.status}).`);
        }
        return;
      }
      consecutiveErrors = 0;
      const data = await resp.json();
      // The route returns the job snapshot itself ({id, state, message, ...}); older code
      // looked for a wrapping `job` key that the server has never sent.
      const job = data.job || data;
      if (!job || !job.state) {
        clearInterval(interval);
        _mutationInflight = false;
        refreshRuntimeUi();
        return;
      }

      // The server's success state is `complete`.
      if (job.state === 'complete' || job.state === 'completed') {
        clearInterval(interval);
        _mutationInflight = false;
        showToast('Rapid-MLX runtime operation completed.', 'success');
        refreshRuntimeUi();
      } else if (job.state === 'failed' || job.state === 'cancelled') {
        clearInterval(interval);
        _mutationInflight = false;
        showToast(`Rapid-MLX operation ${job.state}: ${job.message || 'See logs'}`, 'error');
        refreshRuntimeUi();
      } else {
        showRuntimeProgress(job, startedAt);
      }
    } catch (err) {
      consecutiveErrors += 1;
      if (consecutiveErrors >= POLL_JOB_MAX_CONSECUTIVE_ERRORS) {
        giveUp(interval, `Repeated failures checking job status (${err.message || err}).`);
      }
    }
  }, POLL_JOB_INTERVAL_MS);
}

function showReleaseNotes(release) {
  const detailsEl = document.getElementById('rapid-mlx-release-notes');
  if (!detailsEl) return;

  if (release.release_notes) {
    setHtml(detailsEl, release.release_notes);
    detailsEl.style.display = '';
  } else {
    detailsEl.textContent = 'No release notes available.';
    detailsEl.style.display = '';
  }

  showChangelogSection(release);
}

function showChangelogSection(release) {
  const changelogEl = document.getElementById('rapid-mlx-changelog');
  if (!changelogEl) return;

  const active = _runtimeStatus?.active;
  const fromVersion = active?.version;

  if (!fromVersion || fromVersion === release.version) {
    changelogEl.style.display = 'none';
    return;
  }

  changelogEl.style.display = '';
  resetChangelogBody(fromVersion, release.version);
}

function resetChangelogBody(fromVersion, toVersion) {
  const bodyEl = document.getElementById('rapid-mlx-changelog-body');
  const toggleBtn = document.getElementById('rapid-mlx-changelog-toggle');
  if (!bodyEl || !toggleBtn) return;

   const cacheKey = `${fromVersion}...${toVersion}`;
  const cached = _changelogCache[cacheKey];

  toggleBtn.setAttribute('aria-expanded', 'false');
  bodyEl.style.display = 'none';

  if (cached) {
    setHtml(bodyEl, cached);
  } else {
    bodyEl.textContent = 'Loading changelog…';
    bodyEl.className = 'rapid-mlx-changelog-loading';
  }
}

function toggleChangelog() {
  const bodyEl = document.getElementById('rapid-mlx-changelog-body');
  const toggleBtn = document.getElementById('rapid-mlx-changelog-toggle');
  if (!bodyEl || !toggleBtn) return;

  const isExpanded = toggleBtn.getAttribute('aria-expanded') === 'true';

  if (isExpanded) {
    toggleBtn.setAttribute('aria-expanded', 'false');
    bodyEl.style.display = 'none';
    return;
  }

  toggleBtn.setAttribute('aria-expanded', 'true');
  bodyEl.style.display = '';

  const cachedContent = bodyEl.textContent.trim();
  if (!cachedContent.includes('Loading changelog') && !cachedContent.includes('Changelog unavailable')) {
    return;
  }

  fetchChangelog();
}

async function fetchChangelog() {
  const bodyEl = document.getElementById('rapid-mlx-changelog-body');
  if (!bodyEl) return;

  const active = _runtimeStatus?.active;
  const fromVersion = active?.version;
  const selectedRelease = findSelectedRelease();
  if (!fromVersion || !selectedRelease) {
    setHtml(bodyEl, '<div class="rapid-mlx-changelog-error">Changelog unavailable.</div>');
    return;
  }

  const toVersion = selectedRelease.version;
  const cacheKey = `${fromVersion}...${toVersion}`;

  try {
    const headers = window.authHeaders ? window.authHeaders() : {};
    const resp = await fetch(
      `/api/rapid-mlx/runtime/changelog?from=${encodeURIComponent(fromVersion)}&to=${encodeURIComponent(toVersion)}`,
      { headers }
    );

    if (!resp.ok) {
      handleChangelogError(resp, bodyEl);
      return;
    }

    const data = await resp.json();
    if (!data.ok) {
      handleChangelogError({ status: 500, json: () => Promise.resolve(data) }, bodyEl);
      return;
    }

    const html = renderChangelog(data.changelog);
    _changelogCache[cacheKey] = html;
    setHtml(bodyEl, html);
  } catch {
    setHtml(bodyEl, '<div class="rapid-mlx-changelog-error">Changelog unavailable (network error).</div>');
  }
}

function handleChangelogError(resp, bodyEl) {
  if (resp.status === 429 || resp.status === 503) {
    setHtml(bodyEl, '<div class="rapid-mlx-changelog-error">Changelog unavailable (rate-limited).</div>');
    return;
  }
  resp.json().then(data => {
    if (data.kind === 'rate_limited') {
      setHtml(bodyEl, '<div class="rapid-mlx-changelog-error">Changelog unavailable (rate-limited).</div>');
    } else if (data.kind === 'invalid_tag') {
      setHtml(bodyEl, '<div class="rapid-mlx-changelog-error">Changelog unavailable (invalid version).</div>');
    } else {
      setHtml(bodyEl, '<div class="rapid-mlx-changelog-error">Changelog unavailable.</div>');
    }
  }).catch(() => {
    setHtml(bodyEl, '<div class="rapid-mlx-changelog-error">Changelog unavailable.</div>');
  });
}

function findSelectedRelease() {
  const selectedRow = document.querySelector('.rapid-mlx-release-row--selected');
  if (!selectedRow) return null;

  const versionText = selectedRow.querySelector('.rapid-mlx-release-row-ver')?.textContent;
  if (!versionText) return null;

  const version = versionText.replace(/^v/, '');
  return _releases.find(r => r.version === version) || null;
}

function renderChangelog(changelog) {
  if (!changelog || !changelog.commits || changelog.commits.length === 0) {
    return '<div class="rapid-mlx-changelog-error">No commit details available.</div>';
  }

  let html = '';
  const commits = changelog.commits.slice(0, 50);

  html += `<div class="rapid-mlx-changelog-summary">
    ${commits.length} commit${commits.length === 1 ? '' : 's'} in this release
  </div>`;

  commits.forEach(commit => {
    const message = escapeHtml(commit.message);
    const author = escapeHtml(commit.author || 'unknown');
    html += `<div class="rapid-mlx-changelog-commit">
      <div class="rapid-mlx-changelog-commit-message">${message}</div>
      <div class="rapid-mlx-changelog-commit-meta">
        <span class="rapid-mlx-changelog-commit-sha">${commit.sha}</span>
        <span class="rapid-mlx-changelog-commit-author">${author}</span>
      </div>
    </div>`;
  });

  if (changelog.html_url) {
    html += `<div style="margin-top:6px;">
      <a class="rapid-mlx-changelog-link" href="${escapeHtml(changelog.html_url)}" target="_blank" rel="noopener noreferrer">View full diff on GitHub ↗</a>
    </div>`;
  }

  return html;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function timeAgo(iso) {
  if (!iso) return '';
  try {
    const diff = Date.now() - new Date(iso).getTime();
    const days = Math.max(0, Math.round(diff / 86_400_000));
    if (days === 0) return 'today';
    if (days < 7) return `${days}d ago`;
    if (days < 30) return `${Math.floor(days / 7)}w ago`;
    return `${Math.floor(days / 30)}mo ago`;
  } catch {
    return '';
  }
}
