/* global DOMPurify */

// ── llama-server Binary Updater ───────────────────────────────────────────────
// Shows current binary version in nav pill + setup toolbar button.
// Clicking opens a version picker modal with the last 8 releases;
// any build (including older ones) can be installed from there.

import { showToast } from './toast.js';
import { attachModalFocusTrap, detachModalFocusTrap, fetchReleaseList, buildReleaseBadges } from './updater-shared.js';

const LLAMA_CPP_REPO = 'https://github.com/ggml-org/llama.cpp';
const PR_LINK_RE = /#(\d+)/g;

function linkPrRefs(html) {
  if (!PR_LINK_RE.test(html)) return html;
  return html.replace(PR_LINK_RE, (_, id) =>
    `<a href="${LLAMA_CPP_REPO}/pull/${id}" target="_blank" rel="noopener noreferrer" class="llama-version-pr-link">#${id}</a>`,
  );
}

let _currentBuild = null;
let _latestBuild  = null;
let _updating     = false;
let _versionCheckTimer = null;

export function initLlamaUpdater() {
  const pill = document.getElementById('llama-pill');
  if (pill) pill.addEventListener('click', openVersionModal);

  const closeBtn = document.getElementById('llama-version-modal-close');
  if (closeBtn) closeBtn.addEventListener('click', closeVersionModal);

  const overlay = document.getElementById('llama-version-modal');
  if (overlay) {
    overlay.addEventListener('click', e => {
      if (e.target === overlay) closeVersionModal();
    });
  }

  // Delay the startup check slightly so it does not compete with first-paint
  // work, and stop the background timer while the tab is hidden.
  setTimeout(() => {
    if (!document.hidden) {
      checkVersion();
      ensureVersionPolling();
    }
  }, 2000);

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stopVersionPolling();
      return;
    }
    checkVersion();
    ensureVersionPolling();
  });
}

function ensureVersionPolling() {
  if (_versionCheckTimer) return;
  _versionCheckTimer = setInterval(() => {
    if (document.hidden) return;
    checkVersion();
  }, 30 * 60 * 1000);
}

function stopVersionPolling() {
  if (!_versionCheckTimer) return;
  clearInterval(_versionCheckTimer);
  _versionCheckTimer = null;
}

async function checkVersion() {
  const pill    = document.getElementById('llama-pill');
  const verSpan = document.getElementById('llama-pill-version');

  try {
    const headers = window.authHeaders ? window.authHeaders() : {};

    const vResp = await fetch('/api/llama-binary/version', { headers });
    if (!vResp.ok) return;
    const vData = await vResp.json();
    _currentBuild = vData.build ?? null;

    if (_currentBuild) {
      if (verSpan) verSpan.textContent = `llama.cpp · b${_currentBuild}`;
      if (pill) pill.style.display = 'flex';
    }

    const lResp = await fetch('/api/llama-binary/latest', { headers });
    if (!lResp.ok) return;
    const lData = await lResp.json();
    if (lData.error) return;
    _latestBuild = lData.build ?? null;

    if (_latestBuild && _currentBuild && _latestBuild > _currentBuild) {
      if (verSpan) verSpan.textContent = `llama.cpp · ↑ b${_latestBuild}`;
      if (pill) {
        pill.classList.remove('llama-pill-idle');
        pill.classList.add('llama-pill-update');
        pill.title = `Update available: b${_currentBuild} → b${_latestBuild}. Click to manage.`;
      }
    }
  } catch (_) {
    // silently ignore — background check
  }
}

// ── Modal ──────────────────────────────────────────────────────────────────────

async function openVersionModal() {
  if (_updating) return;

  const modal = document.getElementById('llama-version-modal');
  if (!modal) return;

  // Show current build in the header
  const buildEl = document.getElementById('llama-version-current-build');
  if (buildEl) {
    buildEl.textContent = _currentBuild ? `b${_currentBuild}` : '—';
    if (_latestBuild && _currentBuild && _latestBuild > _currentBuild) {
      buildEl.textContent += ` → b${_latestBuild} available`;
      buildEl.classList.add('llama-version-has-update');
    } else {
      buildEl.classList.remove('llama-version-has-update');
    }
  }

  modal.style.display = 'flex';
  modal.classList.add('open');
  document.body.style.overflow = 'hidden';

  // Move focus to close button for accessibility.
  const closeBtn = document.getElementById('llama-version-modal-close');
  if (closeBtn) closeBtn.focus();

  // Attach Escape key and tab-scope handlers.
  attachModalFocusTrap(modal, closeVersionModal);

  await loadReleaseList();
}

function closeVersionModal() {
  const modal = document.getElementById('llama-version-modal');
  if (!modal) return;
  modal.style.display = 'none';
  modal.classList.remove('open');
  document.body.style.overflow = '';
  detachModalFocusTrap(modal);
  // Reset notes pane for next open
  const empty = document.getElementById('llama-version-notes-empty');
  const content = document.getElementById('llama-version-notes-content');
  if (empty) empty.style.display = '';
  if (content) content.style.display = 'none';
}

async function loadReleaseList() {
  const listEl = document.getElementById('llama-version-list');
  if (!listEl) return;

  listEl.innerHTML = '<div class="llama-version-loading">Loading releases…</div>';

  try {
    const headers = window.authHeaders ? window.authHeaders() : {};
    const data = await fetchReleaseList('/api/llama-binary/releases');
    if (data.error) throw new Error(data.error);

    const releases = sortReleasesChronologically(data.releases ?? []);
    if (!releases.length) {
      listEl.innerHTML = '<div class="llama-version-loading">No releases found.</div>';
      return;
    }

    listEl.innerHTML = '';
    // The API may append the latest versioned (stable) release flagged
    // `stable: true` once it ages out of the newest-8 window; render it as a
    // pinned row below the window, after the installed-build pin.
    const pinnedStable = releases.find(r => r.stable === true);
    const windowReleases = releases.filter(r => r.stable !== true);
    const latestInstallableIndex = windowReleases.findIndex(r => r.installable !== false);
    windowReleases.forEach((r, i) => {
      const row = buildReleaseRow(r, i === latestInstallableIndex);
      listEl.appendChild(row);
    });

    // If the installed build is older than the 8-release window, pin it at the bottom
    if (_currentBuild && !windowReleases.some(r => r.build === _currentBuild)) {
      const sep = document.createElement('div');
      sep.className = 'llama-version-pinned-sep';
      sep.textContent = '— installed —';
      listEl.appendChild(sep);
      // Fetch release info for the pinned build (age + release notes)
      const pinnedRelease = await fetchReleaseInfo(_currentBuild, headers);
      const pinnedRow = buildReleaseRow(pinnedRelease, false);
      listEl.appendChild(pinnedRow);
    }

    // Pin the latest versioned (stable) release so its release notes remain
    // reachable after it leaves the newest-8 window.
    if (pinnedStable) {
      const sep = document.createElement('div');
      sep.className = 'llama-version-pinned-sep';
      sep.textContent = '— stable —';
      listEl.appendChild(sep);
      listEl.appendChild(buildReleaseRow(pinnedStable, false));
    }

    // Auto-select the latest release to show notes on open
    if (windowReleases.length > 0) showReleaseNotes(windowReleases[0]);
  } catch (err) {
    listEl.textContent = '';
    const errEl = document.createElement('div');
    errEl.className = 'llama-version-loading llama-version-error';
    errEl.textContent = `Failed to load releases: ${err.message}`;
    listEl.appendChild(errEl);
  }
}

function showReleaseNotes(release) {
  const empty = document.getElementById('llama-version-notes-empty');
  const content = document.getElementById('llama-version-notes-content');
  const tagEl = document.getElementById('llama-version-notes-tag');
  const bodyEl = document.getElementById('llama-version-notes-body');

  if (!content || !tagEl || !bodyEl) return;

  // Highlight selected row
  document.querySelectorAll('.llama-version-row').forEach(r => r.classList.remove('llama-version-row--selected'));
  const tag = release.tag ?? `b${release.build}`;
  document.querySelectorAll('.llama-version-row').forEach(r => {
    if (r.dataset.tag === tag) r.classList.add('llama-version-row--selected');
  });

  tagEl.textContent = tag;
  if (empty) empty.style.display = 'none';
  content.style.display = '';

  // Scroll notes pane to top
  const notesPane = document.getElementById('llama-version-notes-pane');
  if (notesPane) notesPane.scrollTop = 0;

  if (release.body && release.body.trim()) {
    let html;
    if (typeof marked !== 'undefined') {
      html = marked.parse(normalizeReleaseNotes(release.body));
    } else {
      html = normalizeReleaseNotes(release.body).replace(/\n/g, '<br>');
    }
    html = linkPrRefs(html);
    // eslint-disable-next-line no-unsanitized/property
    bodyEl.innerHTML = typeof DOMPurify !== 'undefined'
      ? DOMPurify.sanitize(html, { ADD_ATTR: ['target'] })
      : html;
  } else {
    bodyEl.innerHTML = '<p class="llama-version-notes-none">No release notes for this build.</p>';
  }
}

/**
 * GitHub's versioned llama.cpp releases publish changelog entries as bare
 * commit lines, unlike nightly releases which use Markdown list items. Add
 * list markers to those commit lines so the notes pane does not collapse the
 * entire changelog into one paragraph. Existing Markdown lists are left
 * untouched.
 */
function normalizeReleaseNotes(body) {
  return body.replace(
    /^(?!\s*[-*+]\s)([0-9a-f]{7,40}\s+.+)$/gim,
    '- $1',
  );
}

function buildReleaseRow(release, isLatest) {
  const isCurrent = _currentBuild !== null && release.build === _currentBuild;
  const isInstallable = release.installable !== false;
  const age = release.published_at ? relativeAge(release.published_at) : '';
  const tag = release.tag ?? `b${release.build}`;

  const row = document.createElement('div');
  row.className = 'llama-version-row' + (isCurrent ? ' llama-version-row--current' : '');
  row.dataset.tag = tag;
  row.title = 'Click to view release notes';
  row.style.cursor = 'pointer';
  row.addEventListener('click', (e) => {
    if (e.target.closest('.llama-version-install-btn')) return;
    showReleaseNotes(release);
  });

  const info = document.createElement('div');
  info.className = 'llama-version-row-info';

  const tagEl = document.createElement('span');
  tagEl.className = 'llama-version-row-tag';
  tagEl.textContent = tag;

  const badges = buildReleaseBadges({
    wrapperClass: 'llama-version-row-badges',
    badgeClass: 'llama-version-badge',
    isLatest: isLatest && isInstallable,
    isCurrent,
  });

  const meta = document.createElement('span');
  meta.className = 'llama-version-row-age';
  meta.textContent = age;

  info.append(tagEl, badges);

  const right = document.createElement('div');
  right.className = 'llama-version-row-right';
  right.appendChild(meta);

  if (!isCurrent && isInstallable) {
    const btn = document.createElement('button');
    btn.className = 'llama-version-install-btn';
    btn.textContent = 'Install';
    btn.dataset.tag = tag;
    btn.dataset.build = release.build ?? '';
    btn.addEventListener('click', () => installRelease(btn, release));
    right.appendChild(btn);
  } else if (!isInstallable) {
    const notesOnly = document.createElement('span');
    notesOnly.className = 'llama-version-badge llama-version-badge--notes';
    notesOnly.textContent = release.has_binaries ? 'No build for your platform' : 'Notes only';
    right.appendChild(notesOnly);
  }

  row.append(info, right);
  return row;
}

async function installRelease(btn, release) {
  if (_updating) return;
  _updating = true;

  const tag = release.tag ?? `b${release.build}`;
  btn.disabled = true;
  btn.textContent = 'Installing…';

  const startTime = Date.now();
  const timer = setInterval(() => {
    const s = Math.round((Date.now() - startTime) / 1000);
    btn.textContent = `${s}s…`;
  }, 1000);

  // Also update the nav pill to show progress
  const verSpan = document.getElementById('llama-pill-version');
  const pill    = document.getElementById('llama-pill');
  if (pill) { pill.classList.add('llama-pill-busy'); pill.disabled = true; }
  if (verSpan) verSpan.textContent = 'Installing…';

  try {
    const headers = window.authHeaders
      ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
      : { 'Content-Type': 'application/json' };

    const resp = await fetch('/api/llama-binary/update', {
      method: 'POST',
      headers,
      body: JSON.stringify({ tag }),
    });
    clearInterval(timer);
    const data = await resp.json();
    if (!data.ok) throw new Error(data.error || 'Install failed');

    _currentBuild = release.build ?? _latestBuild;
    const sha = data.sha256 ? `SHA256: ${data.sha256}` : '';
    // If the backend already restarted the server (server was running during update),
    // skip the explicit restart call to avoid a double-restart that kills the new process.
    if (data.server_restarted) {
      showToast(`Installed ${tag}`, 'success', 'Server restarted. Chat may reconnect shortly.');
    } else if (data.restart_applicable === false) {
      showToast(`Installed ${tag}`, 'success', sha || 'llama.cpp binary updated.');
    } else {
      showToast(`Installed ${tag}`, 'success', sha || 'Installed. Click to restart server.');

      // Attempt to restart the running server so it uses the new binary immediately.
      // This is safe: if no local server is running, the backend will return a clear error.
      try {
        const restartResp = await fetch('/api/llama/restart', {
          method: 'POST',
          headers: window.authHeaders
            ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
            : { 'Content-Type': 'application/json' },
        });
        const restartData = await restartResp.json();
        if (restartData.ok) {
          showToast(
            `Server restarted with ${tag}`,
            'success',
            'Chat may reconnect shortly.',
          );
        } else if (
          restartData.error &&
          (restartData.error.includes('No local') ||
           restartData.error.includes('no local'))
        ) {
          // Server not running — nothing special needed.
        } else {
          showToast(
            'Restart failed',
            'error',
            (restartData.error || 'Could not restart server automatically.'),
          );
        }
      } catch (_restartErr) {
        // Network error while restarting: warn but don't block install success.
        showToast(
          'Restart pending',
          'info',
          `Installed ${tag}. Restart llama-server manually or refresh your session to use the new binary.`,
        );
      }
    }

    // Refresh pill
    if (verSpan) verSpan.textContent = `llama.cpp · b${_currentBuild}`;
    if (pill) {
      pill.classList.remove('llama-pill-busy', 'llama-pill-update');
      pill.classList.add('llama-pill-idle');
      pill.disabled = false;
      pill.title = 'llama-server binary version';
    }
    // Refresh the list so badges update
    closeVersionModal();
    // Trigger immediate version check to refresh badge state
    checkVersion();
  } catch (err) {
    clearInterval(timer);
    showToast('Install failed', 'error', err.message || String(err));
    btn.disabled = false;
    btn.textContent = 'Install';
    if (pill) {
      pill.classList.remove('llama-pill-busy');
      if (_latestBuild && _currentBuild && _latestBuild > _currentBuild) {
        pill.classList.add('llama-pill-update');
      } else {
        pill.classList.add('llama-pill-idle');
      }
      pill.disabled = false;
    }
    if (verSpan) verSpan.textContent = `llama.cpp · b${_currentBuild ?? '?'}`;
  } finally {
    _updating = false;
  }
}

async function fetchReleaseInfo(build, headers) {
  try {
    const resp = await fetch(`/api/llama-binary/release?build=${build}`, { headers });
    if (!resp.ok) return { build, tag: `b${build}`, published_at: '', body: '' };
    const data = await resp.json();
    return data.error
      ? { build, tag: `b${build}`, published_at: '', body: '' }
      : data;
  } catch {
    return { build, tag: `b${build}`, published_at: '', body: '' };
  }
}

function relativeAge(iso) {
  if (!iso) return '';
  const diff = Date.now() - new Date(iso).getTime();
  const mins  = Math.floor(diff / 60_000);
  const hours = Math.floor(diff / 3_600_000);
  const days  = Math.floor(diff / 86_400_000);
  if (mins < 60)  return `${mins}m ago`;
  if (hours < 24) return `${hours}h ago`;
  if (days < 30)  return `${days}d ago`;
  const months = Math.floor(days / 30);
  return `${months}mo ago`;
}

function sortReleasesChronologically(releases) {
  return [...releases].sort((left, right) => {
    const leftTime = Date.parse(left.published_at || '') || 0;
    const rightTime = Date.parse(right.published_at || '') || 0;
    return rightTime - leftTime
      || String(right.tag || '').localeCompare(String(left.tag || ''));
  });
}
