// Chat template auto-install / model-family detection for the spawn wizard.
import { showToast } from './toast.js';
import { openChatTemplateLibraryBrowser, uploadChatTemplateFromBrowser } from './file-browser-launcher.js';
import {
  buildCommunityTemplateInstallRequest,
  communityFamilyFromGgufArchitecture,
  getDefaultTemplateForFamily,
  getTemplateFamilies,
  getTemplatesForFamily,
  provenanceSuffix,
} from './chat-template-registry.js';
import { wizardState, showStep, snapshotPendingRestore } from './spawn-wizard.js';
import { awaitOriginResolve } from './spawn-wizard-hf-origin.js';
import { CT_LABELS, openChatTemplateManageModal, repoFromSourceUrl } from './chat-template-panel.js';
import { applyTemplateUpdate, describeUpdate } from './chat-template-update.js';
import { readLastStatus } from './template-autoupdater.js';

// Cache of installed community templates keyed by template name.
// Avoids re-downloading the same template for each model of the same family.
const _installedTemplateCache = {};

// After an update (from the toast, the bell, Manage template, or the button on the status line),
// drop the session cache so the status line re-reads the refreshed install, then redraw it.
window.addEventListener('chatTemplateUpdated', (event) => {
  delete _installedTemplateCache[event.detail?.name];
  const section = document.getElementById('chat-template-section');
  if (wizardState.model.chatTemplateMode === 'auto' && section && section.style.display !== 'none') {
    void autoInstallChatTemplate();
  }
});

export function _chatTemplateDisplayName(path) {
  if (!path) return 'Embedded (from model file)';
  return path.split(/[\\/]/).pop() || path;
}

export function _applyCustomChatTemplate(path) {
  wizardState.model.chatTemplatePath = path || null;
  wizardState.model.chatTemplateMode = path ? 'custom' : 'embedded';
  const hiddenInput = document.getElementById('spawn-chat-template-path');
  if (hiddenInput) hiddenInput.value = path || '';
  const family = wizardState.model.family || null;
  const tpl = getDefaultTemplateForFamily(family);
  _renderChatTemplateStatus(path ? 'custom' : 'embedded', family, tpl, { path });
}

// Checks HF tags (own tags + any declared base_model tag) for architecture
// evidence — real repo metadata, never a filename/display-name string.
function _communityFamilyFromHfTags(tags) {
  const list = (tags || []).map(t => String(t).toLowerCase());
  const baseModelTag = list.find(t => t.startsWith('base_model:') && t.slice('base_model:'.length).includes('/'));
  const candidates = baseModelTag ? [...list, baseModelTag.slice('base_model:'.length)] : list;
  for (const c of candidates) {
    if (c.includes('qwen')) return 'qwen';
    if (c.includes('gemma-4') || c.includes('gemma4')) return 'gemma4';
  }
  return null;
}

// Async family detection — real model metadata only, never filename matching. Evidence
// ladder per plan §2.3c, ranked by trustworthiness rather than "first source that answers":
//   1) Persisted family tag from model-tags.json          -> confidence 'pinned'
//   2) Local/remote model config (architecture/model_type) -> confidence 'confirmed'
//   3) HF repo tags / declared base_model tag              -> confidence 'heuristic'
// Auto-install (autoInstallChatTemplate) only acts at 'confirmed' or better; 'heuristic'
// results are offered to the user rather than applied silently (see applyDetectedFamily).
// Returns { family, confidence, source } or null if nothing resolved.
export async function detectModelFamilyAsync(identityName, localPath, timeoutMs) {
  const timeout = timeoutMs || 5000;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  const headers = window.authHeaders ? window.authHeaders() : {};
  const isRapidMlx = wizardState.engine.selected === 'rapid_mlx';

  try {
    // 1) Check persisted family tag
    if (localPath) {
      try {
        const resp = await fetch('/api/models/tags', { headers, signal: controller.signal });
        if (resp.ok) {
          const td = await resp.json().catch(() => ({}));
          const tags = td.tags?.[localPath] || [];
          const familyTag = tags.find(t => t.startsWith('family:'));
          if (familyTag) {
            return { family: familyTag.slice('family:'.length), confidence: 'pinned', source: 'model-tags.json' };
          }
        }
      } catch (e) { if (e.name !== 'AbortError') { /* non-fatal */ } }
    }

    // 2) Read local model config — architecture field is authoritative
    if (localPath) {
      try {
        if (isRapidMlx) {
          const metaResp = await fetch('/api/models/mlx-introspect', {
            method: 'POST',
            headers: { ...headers, 'Content-Type': 'application/json' },
            signal: controller.signal,
            body: JSON.stringify({ model_path: localPath }),
          });
          if (metaResp.ok) {
            const meta = await metaResp.json().catch(() => ({}));
            // /api/models/mlx-introspect nests the parsed config under data.config,
            // not top-level — meta.model_type/meta.config were never populated.
            const modelType = meta.data?.config?.model_type || meta.model_type;
            if (modelType) {
              const family = communityFamilyFromGgufArchitecture(modelType);
              if (family) return { family, confidence: 'confirmed', source: 'config.json model_type' };
            }
          }
        } else {
          const metaResp = await fetch('/api/models/gguf-meta', {
            method: 'POST',
            headers: { ...headers, 'Content-Type': 'application/json' },
            signal: controller.signal,
            body: JSON.stringify({ model_path: localPath }),
          });
          if (metaResp.ok) {
            const meta = await metaResp.json().catch(() => ({}));
            if (meta.ok && meta.architecture) {
              const family = communityFamilyFromGgufArchitecture(meta.architecture);
              if (family) return { family, confidence: 'confirmed', source: 'GGUF general.architecture' };
            }
          }
        }
      } catch (e) { if (e.name !== 'AbortError') { /* non-fatal */ } }
    }

    // 2b) No local path (HF-selected, alias-sourced, etc.) — for MLX, use the remote
    // introspection mode of the same endpoint (§2.3b) so family detection isn't limited
    // to models that have already been downloaded.
    const repoId = identityName || wizardState.model.originRepo;
    const looksLikeHfRepo = repoId && repoId.includes('/') &&
      !repoId.includes('\\') &&
      !repoId.split('/')[0].includes('.');
    if (!localPath && looksLikeHfRepo && isRapidMlx) {
      try {
        const metaResp = await fetch('/api/models/mlx-introspect', {
          method: 'POST',
          headers: { ...headers, 'Content-Type': 'application/json' },
          signal: controller.signal,
          body: JSON.stringify({ repo_id: repoId, revision: wizardState.model.hfRevision || 'main' }),
        });
        if (metaResp.ok) {
          const meta = await metaResp.json().catch(() => ({}));
          const modelType = meta.data?.config?.model_type;
          if (modelType) {
            const family = communityFamilyFromGgufArchitecture(modelType);
            if (family) return { family, confidence: 'confirmed', source: 'remote config.json model_type' };
          }
        }
      } catch (e) { if (e.name !== 'AbortError') { /* non-fatal */ } }
    }

    // 3) Check HF repo tags / declared base_model tag (for locally resolved origins or HF repos)
    if (looksLikeHfRepo) {
      try {
        const metaResp = await fetch(`/api/hf/meta?repo=${encodeURIComponent(repoId)}`, { headers, signal: controller.signal });
        if (metaResp.ok) {
          const meta = await metaResp.json().catch(() => ({}));
          const detected = _communityFamilyFromHfTags(meta.tags);
          if (detected) return { family: detected, confidence: 'heuristic', source: 'HF tags' };
        }
      } catch (e) { if (e.name !== 'AbortError') { /* non-fatal */ } }
    }

    return null;
  } finally {
    clearTimeout(timer);
  }
}

export async function autoInstallChatTemplate(force = false) {
  const { source, path, hfRepo } = wizardState.model;
  const identityName = source === 'hf' ? hfRepo : path;

  // Fast path: family already known (resolved from real metadata earlier in the flow)
  let family = wizardState.model.family || null;
  const tpl = getDefaultTemplateForFamily(family);

  // If no family from fast path, we need to detect it.
  // For local/import models, await the origin resolver first (it fires from
  // the model path input handler and includes family detection in the same pass).
  if (!family && (source === 'local' || source === 'import')) {
    _renderChatTemplateStatus('detecting', null, null, null);
    // Await the origin resolver promise (created from model path input handler).
    // The resolver is idempotent — the one from loadLocalModel will be a no-op.
    // 1.5s timeout is generous: the HF search takes ~500ms.
    await awaitOriginResolve(1500);
    // After the resolver, check again (family may now be set by the resolver).
    family = wizardState.model.family || null;
  }
  // If still no family, query HF directly (for models not covered by origin resolver)
  if (!family) {
    try {
      const detected = await detectModelFamilyAsync(identityName, path, 8000);
      // Plan §2.3c: only act automatically at 'confirmed' or better. A 'heuristic' (HF
      // tags only) result is not applied here — surfacing it as a one-click offer instead
      // of auto-installing is deferred; for now it degrades to "no recommendation" rather
      // than silently acting on a guess, which is the safe half of that rule.
      if (detected && detected.confidence !== 'heuristic') {
        family = detected.family;
        wizardState.model.familyConfidence = detected.confidence;
        wizardState.model.familySource = detected.source;
      }
    } catch { /* non-fatal */ }
  }
  // Update wizard state for future use
  if (family) wizardState.model.family = family;

  // Use explicitly chosen candidate (from force-family dropdown) or fall back to default for family
  const candidates = getTemplatesForFamily(family);
  const tplForFamily = wizardState.model.chatTemplateCandidate
    ? candidates.find(c => c.name === wizardState.model.chatTemplateCandidate) || getDefaultTemplateForFamily(family)
    : getDefaultTemplateForFamily(family);

  if (wizardState.model.chatTemplateMode === 'custom' && wizardState.model.chatTemplatePath) {
    _renderChatTemplateStatus('custom', family, tplForFamily, { path: wizardState.model.chatTemplatePath });
    return;
  }

  if (wizardState.model.chatTemplateMode === 'embedded') {
    wizardState.model.chatTemplatePath = null;
    _renderChatTemplateStatus('embedded', family, tplForFamily, null);
    return;
  }

  if (!tplForFamily) {
    wizardState.model.chatTemplatePath = null;
    wizardState.model.chatTemplateMode = 'auto';
    _renderChatTemplateStatus('embedded', family, null, null);
    return;
  }

  // Cache hit: template already installed for this family (skip when forcing a re-fetch)
  const cached = !force && _installedTemplateCache[tplForFamily.name];
  if (cached) {
    wizardState.model.chatTemplatePath = cached.path;
    wizardState.model.chatTemplateMode = 'auto';
    _renderChatTemplateStatus('installed', family, tplForFamily, cached);
    return;
  }

  _renderChatTemplateStatus('installing', family, tplForFamily, null);

  try {
    const headers = window.authHeaders
      ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
      : { 'Content-Type': 'application/json' };
    const install = buildCommunityTemplateInstallRequest(tplForFamily, force);
    const resp = await fetch(install.endpoint, {
      method: 'POST', headers,
      body: JSON.stringify(install.body),
    });
    const data = resp.ok ? await resp.json() : { ok: false, error: `HTTP ${resp.status}` };
    if (data.ok && data.path) {
      wizardState.model.chatTemplatePath = data.path;
      wizardState.model.chatTemplateMode = 'auto';
      // Cache the template metadata for this family (avoids re-downloading for
      // other models of the same family in the same session)
      _installedTemplateCache[tplForFamily.name] = data;

      // Mark a force-refresh so the UI can show "Updated!" briefly
      const displayData = force
        ? { ...data, _forceRefresh: true }
        : data;

      _renderChatTemplateStatus('installed', family, tplForFamily, displayData);

       if (force) {
         showToast(
           _templateDisplayName(tplForFamily, family) + ' re-downloaded',
           'success',
           'Template has been refreshed from its upstream source',
           3000
         );
      }
    } else {
      _renderChatTemplateStatus('error', family, tplForFamily, data);
    }
  } catch (err) {
    _renderChatTemplateStatus('error', family, tplForFamily, { error: err.message || String(err) });
  }
}

// Display name for a template. A provenance suffix is added only when the family mixes
// provenance (see provenanceSuffix), so two community templates are told apart by name.
function _templateDisplayName(tpl, family) {
  if (!tpl) return '';
  return tpl.display + provenanceSuffix(tpl, family);
}

function _renderChatTemplateStatus(state, family, tpl, data) {
  const section = document.getElementById('chat-template-section');
  const statusEl = document.getElementById('ct-status');
  const bodyEl = document.getElementById('ct-body');
  const actionsEl = document.getElementById('ct-actions');
  if (!section) return;

  section.style.display = '';

  if (actionsEl) {
    actionsEl.innerHTML = '';

    if (tpl) {
      const recommendedBtn = document.createElement('button');
      recommendedBtn.type = 'button';
      recommendedBtn.className = 'btn-wizard-tertiary ct-action-btn';
       const isUsing = wizardState.model.chatTemplateMode === 'auto';
       const familyLabel = family ? ` (${family} family)` : '';
       const displayName = _templateDisplayName(tpl, family);
       recommendedBtn.textContent = isUsing
         ? `Re-fetch Recommended${familyLabel}`
         : `Use ${displayName}${familyLabel}`;
      recommendedBtn.title = isUsing
        ? 'Force re-download this template from source, even if already installed'
        : '';
      recommendedBtn.addEventListener('click', async () => {
        // If already on this template, treat the click as an explicit force
        // re-fetch (bypasses both the session cache and the on-disk shortcut) —
        // this is the only way to pick up upstream changes when "Check for
        // updates" isn't available or reports a false negative.
        const wasUsing = wizardState.model.chatTemplateMode === 'auto';
        wizardState.model.chatTemplateMode = 'auto';
        await autoInstallChatTemplate(wasUsing);
      });
      actionsEl.appendChild(recommendedBtn);
    }

    const embeddedBtn = document.createElement('button');
    embeddedBtn.type = 'button';
    embeddedBtn.className = 'btn-wizard-tertiary ct-action-btn';
    embeddedBtn.textContent = 'Use Embedded';
    embeddedBtn.disabled = wizardState.model.chatTemplateMode === 'embedded' && !wizardState.model.chatTemplatePath;
    embeddedBtn.addEventListener('click', () => {
      _applyCustomChatTemplate(null);
    });
    actionsEl.appendChild(embeddedBtn);

    const libraryBtn = document.createElement('button');
    libraryBtn.type = 'button';
    libraryBtn.className = 'btn-wizard-tertiary ct-action-btn';
    libraryBtn.textContent = 'Choose Existing';
    libraryBtn.addEventListener('click', async () => {
      try {
        await openChatTemplateLibraryBrowser('spawn-chat-template-path');
      } catch (err) {
        showToast('Template library unavailable: ' + (err.message || String(err)), 'error');
      }
    });
    actionsEl.appendChild(libraryBtn);

    const uploadBtn = document.createElement('button');
    uploadBtn.type = 'button';
    uploadBtn.className = 'btn-wizard-tertiary ct-action-btn';
    uploadBtn.textContent = 'Upload .jinja';
    uploadBtn.addEventListener('click', async () => {
      try {
        const uploaded = await uploadChatTemplateFromBrowser();
        if (!uploaded?.path) return;
        _applyCustomChatTemplate(uploaded.path);
        showToast('Template uploaded', 'success', uploaded.filename || 'Saved to template library');
      } catch {
        // uploadChatTemplateFromBrowser already surfaced the error
      }
                 });
    actionsEl.appendChild(uploadBtn);

    // Force family override — lets user manually pick a family when auto-detection fails
    const forceFamilyWrap = document.createElement('div');
    forceFamilyWrap.className = 'ct-force-family-wrap';
    forceFamilyWrap.style.display = 'flex';
    forceFamilyWrap.style.alignItems = 'center';
    forceFamilyWrap.style.gap = '6px';
    forceFamilyWrap.style.marginTop = '6px';

    const forceFamilyLabel = document.createElement('span');
    forceFamilyLabel.style.fontSize = '10px';
    forceFamilyLabel.style.fontWeight = '600';
    forceFamilyLabel.style.color = 'var(--color-text-muted)';
    forceFamilyLabel.style.textTransform = 'uppercase';
    forceFamilyLabel.style.letterSpacing = '0.06em';
    forceFamilyLabel.textContent = 'Force family';

    const forceFamilySelect = document.createElement('select');
    forceFamilySelect.className = 'ct-force-family-select';
    forceFamilySelect.style.fontSize = '11px';
    forceFamilySelect.style.padding = '3px 6px';
    forceFamilySelect.style.borderRadius = '4px';
    forceFamilySelect.style.border = '1px solid rgba(99,102,241,0.2)';
    forceFamilySelect.style.background = 'var(--color-surface-elevated)';
    forceFamilySelect.style.color = 'var(--color-text)';
    forceFamilySelect.title = 'Override the auto-detected model family to force a specific chat template';

    const currentFamily = wizardState.model.family || '';
    const currentCandidate = wizardState.model.chatTemplateCandidate || '';
    const families = getTemplateFamilies();

    const autoOpt = document.createElement('option');
    autoOpt.value = '';
    autoOpt.textContent = 'auto-detect';
    if (!currentFamily) autoOpt.selected = true;
    forceFamilySelect.appendChild(autoOpt);

    families.forEach(fam => {
      const candidates = getTemplatesForFamily(fam);
      if (candidates.length === 1) {
        const tpl = candidates[0];
        const opt = document.createElement('option');
        opt.value = fam;
        const provLabel = tpl.provenance ? ` (${tpl.provenance === 'official' ? 'Official' : 'Community'})` : '';
        opt.textContent = `${fam} — ${tpl.display}${provLabel}`;
        if (currentFamily === fam && !currentCandidate) opt.selected = true;
        forceFamilySelect.appendChild(opt);
      } else {
        const group = document.createElement('optgroup');
        const provLabels = {};
        const famName = fam.charAt(0).toUpperCase() + fam.slice(1);
        group.label = famName;
        candidates.forEach(tpl => {
          provLabels[tpl.name] = provenanceSuffix(tpl, fam);
          const opt = document.createElement('option');
          opt.value = `${fam}:${tpl.name}`;
          opt.textContent = `${tpl.display}${provLabels[tpl.name]}`;
          if (currentFamily === fam && currentCandidate === tpl.name) opt.selected = true;
          group.appendChild(opt);
        });
        forceFamilySelect.appendChild(group);
      }
    });

    forceFamilySelect.addEventListener('change', () => {
      const chosen = forceFamilySelect.value;
      if (!chosen) {
        wizardState.model.family = null;
        wizardState.model.chatTemplateCandidate = null;
        return;
      }
      const [fam, candidateName] = chosen.split(':');
      wizardState.model.family = fam;
      wizardState.model.chatTemplateCandidate = candidateName || null;
      wizardState.model.chatTemplateMode = 'auto';
      autoInstallChatTemplate();
    });

    forceFamilyWrap.appendChild(forceFamilyLabel);
    forceFamilyWrap.appendChild(forceFamilySelect);
    actionsEl.appendChild(forceFamilyWrap);
  }

  if (state === 'detecting') {
    const modelName = (wizardState.model.path || '').split(/[\\/]/).pop() || '';
    if (statusEl) { statusEl.textContent = 'Detecting…'; statusEl.className = 'ct-status ct-installing'; }
    if (bodyEl) {
      bodyEl.textContent = modelName
        ? `Detecting family for ${modelName}…`
        : 'Checking HuggingFace for model family and recommended template…';
    }
    return;
  }

  if (state === 'installing') {
    if (statusEl) { statusEl.textContent = 'Downloading…'; statusEl.className = 'ct-status ct-installing'; }
    if (bodyEl) {
      bodyEl.textContent = '';
      const nameEl = document.createElement('span');
      nameEl.className = 'ct-name';
       nameEl.textContent = _templateDisplayName(tpl, family);
       bodyEl.appendChild(nameEl);
       bodyEl.appendChild(document.createTextNode(' — downloading…'));
    }
    return;
  }

  if (state === 'embedded') {
    if (statusEl) { statusEl.textContent = 'Embedded'; statusEl.className = 'ct-status ct-neutral'; }
    if (bodyEl) {
      bodyEl.textContent = family && tpl
        ? 'Using the template embedded in the model file instead of the recommended community template.'
        : 'Using template embedded in model file. You can choose an existing Jinja or upload a new one here.';
    }
    return;
  }

  if (state === 'installed') {
    const isForceRefresh = !!data?._forceRefresh;
    const installedDate = data?.installed_at
      ? new Date(data.installed_at).toLocaleString()
      : null;

    if (statusEl) {
      if (isForceRefresh) {
        statusEl.textContent = 'Updated!';
        statusEl.className = 'ct-status ct-ok';
        // Briefly highlight the update, then fall back to standard text
        setTimeout(() => {
          if (!statusEl.textContent || statusEl.textContent === 'Updated!') {
            statusEl.textContent = '✓ Installed';
          }
        }, 2200);
      } else {
        statusEl.textContent = data?.already_existed ? '✓ Cached' : '✓ Installed';
        statusEl.className = 'ct-status ct-ok';
      }
    }
    if (bodyEl) {
      bodyEl.textContent = '';
      const nameEl = document.createElement('strong');
       nameEl.textContent = _templateDisplayName(tpl, family);
       const descEl = document.createElement('span');
      descEl.textContent = ` — ${tpl.description}`;
      const sourceUrl = data?.source_url || tpl?.sourceUrl;
      const link = sourceUrl
        ? (() => { const a = document.createElement('a'); a.href = sourceUrl; a.target = '_blank'; a.rel = 'noopener noreferrer'; a.textContent = ' ↗'; a.className = 'ct-hf-link'; return a; })()
        : null;
      bodyEl.appendChild(nameEl);
      bodyEl.appendChild(descEl);
      if (link) bodyEl.appendChild(link);

      // Staleness hint text (unified "Check for updates"/history/discussions/create-fix now
      // live in the shared "Manage template…" modal — see chat-template-panel.js).
      const hint = document.createElement('div');
      hint.style.fontSize = '10px';
      hint.style.color = 'var(--color-text-muted)';
      hint.style.marginTop = '4px';
      const hintSpan = document.createElement('span');
      hintSpan.textContent = installedDate ? `Installed ${installedDate}.` : 'Template installed.';
      hint.appendChild(hintSpan);

      // The raw install path: it carries the install meta.json sidecar that "Check for updates"
      // and Version history read.
      const tplPath = data?.path;
      const tplName = data?.name || tpl?.name;
      const manageBtn = document.createElement('button');
      manageBtn.type = 'button';
      manageBtn.className = 'btn-wizard-tertiary';
      manageBtn.style.fontSize = '11px';
      manageBtn.style.fontWeight = '700';
      manageBtn.style.marginLeft = '6px';
      manageBtn.style.padding = '2px 8px';
      manageBtn.style.color = 'var(--color-accent)';
      manageBtn.style.textDecoration = 'underline';
      manageBtn.textContent = CT_LABELS.manage;
      manageBtn.title = 'Current, updates, version history, and community fixes for this template';
      manageBtn.addEventListener('click', async () => {
        await openChatTemplateManageModal({
          tplName,
          tplRepo: repoFromSourceUrl(data?.source_url) || (tpl?.repo || ''),
          currentPath: tplPath,
          activePath: tplPath,
          onActivated: async () => {
            await autoInstallChatTemplate();
          },
        });
      });
      hint.appendChild(manageBtn);

      // If the background checker found a newer upstream version of this template, say so and
      // let the user act on it right here instead of hunting for it in Manage template.
      const pending = readLastStatus().templates_with_updates.find(item => item.name === tplName);
      if (pending) {
        const note = document.createElement('span');
        note.className = 'ct-update-note';
        note.style.marginLeft = '8px';
        note.style.color = 'var(--color-warning)';
        note.textContent = `Update available (${describeUpdate(pending)}).`;
        const updateBtn = document.createElement('button');
        updateBtn.type = 'button';
        updateBtn.className = 'btn-wizard-tertiary ct-update-btn';
        updateBtn.style.fontSize = '11px';
        updateBtn.style.fontWeight = '700';
        updateBtn.style.marginLeft = '6px';
        updateBtn.style.padding = '2px 8px';
        updateBtn.textContent = 'Update';
        updateBtn.title = 'Download the new version. The current one stays in Version history.';
        updateBtn.addEventListener('click', async () => {
          updateBtn.disabled = true;
          const applied = await applyTemplateUpdate({ name: tplName, sourceUrl: pending.source_url, info: pending });
          if (!applied.ok) updateBtn.disabled = false;
        });
        hint.append(note, updateBtn);
      }
      bodyEl.appendChild(hint);
    }
    return;
  }

  if (state === 'custom') {
    if (statusEl) {
      statusEl.textContent = 'Custom';
      statusEl.className = 'ct-status ct-ok';
    }
    if (bodyEl) {
      bodyEl.textContent = '';
      const customPath = data?.path || wizardState.model.chatTemplatePath;
      const nameEl = document.createElement('strong');
      nameEl.textContent = _chatTemplateDisplayName(customPath);
      const descEl = document.createElement('span');
      descEl.textContent = ' — using your selected template from the local template library.';
      bodyEl.appendChild(nameEl);
      bodyEl.appendChild(descEl);

      // A manually-picked template (via Choose Existing / Upload) can still be
      // one the community-install system knows about — e.g. an older release
      // kept in the library, or a file the user previously auto-installed and
      // then reselected. Give it the same "Manage template…" entry point as
      // the auto-installed path so upstream version browsing/rollback works
      // wherever we can resolve real metadata for it; the modal already
      // degrades gracefully to "Unknown upstream" when it can't.
      if (customPath) {
        const manageBtn = document.createElement('button');
        manageBtn.type = 'button';
        manageBtn.className = 'btn-wizard-tertiary';
        manageBtn.style.fontSize = '11px';
        manageBtn.style.fontWeight = '700';
        manageBtn.style.marginLeft = '6px';
        manageBtn.style.padding = '2px 8px';
        manageBtn.style.color = 'var(--color-accent)';
        manageBtn.style.textDecoration = 'underline';
        manageBtn.textContent = CT_LABELS.manage;
        manageBtn.title = 'Current, updates, version history, and community fixes for this template';
        manageBtn.addEventListener('click', async () => {
          let tplName = _chatTemplateDisplayName(customPath).replace(/\.jinja$/i, '');
          let tplRepo = '';
          try {
            const headers = window.authHeaders ? window.authHeaders() : {};
            const resp = await fetch('/api/chat-template/active', { headers });
            const result = resp.ok ? await resp.json() : { ok: false };
            const match = (result.templates || []).find(t => t.path === customPath);
            if (match) {
              tplName = match.name;
              tplRepo = repoFromSourceUrl(match.source_url) || '';
            }
          } catch { /* fall back to filename-derived name, unknown repo */ }
          await openChatTemplateManageModal({
            tplName,
            tplRepo,
            currentPath: customPath,
            activePath: customPath,
            onActivated: async () => {
              _applyCustomChatTemplate(customPath);
            },
          });
        });
        descEl.appendChild(manageBtn);
      }
    }
    return;
  }

  if (state === 'error') {
    if (statusEl) { statusEl.textContent = '⚠ Failed'; statusEl.className = 'ct-status ct-error'; }
    if (bodyEl) {
      bodyEl.textContent = '';
      const msg = document.createElement('span');
      msg.className = 'ct-error-msg';
      msg.textContent = `${data?.error || 'Download failed'} — server will use embedded template.`;
      const retryBtn = document.createElement('button');
      retryBtn.type = 'button'; retryBtn.className = 'ct-retry-btn btn-wizard-tertiary';
      retryBtn.textContent = 'Retry';
      retryBtn.addEventListener('click', autoInstallChatTemplate);
      bodyEl.appendChild(msg);
      bodyEl.appendChild(document.createTextNode(' '));
      bodyEl.appendChild(retryBtn);
    }
    return;
  }

  // Plan §2.2: the requested template resolved, but the launcher will not be able to apply it
  // (alias sources have no directory to overlay into). Distinct from 'error' — this is not a
  // download failure, it's a structural mismatch discovered from the launch/preview warning.
  if (state === 'degraded') {
    if (statusEl) { statusEl.textContent = '⚠ Not applied'; statusEl.className = 'ct-status ct-warning'; }
    if (bodyEl) {
      bodyEl.textContent = '';
      const nameEl = tpl ? _templateDisplayName(tpl, family) : (data?.templateName || 'the selected template');
      const msg = document.createElement('span');
      msg.className = 'ct-warning-msg';
      msg.textContent = `This model was selected by alias, so llama-monitor cannot install a chat template for it. `
        + `The server will use the template built into the model. To use ${nameEl}, select the same model by `
        + `HuggingFace repo or local folder instead.`;
      bodyEl.appendChild(msg);

      const switchBtn = document.createElement('button');
      switchBtn.type = 'button';
      switchBtn.className = 'ct-retry-btn btn-wizard-tertiary';
      switchBtn.textContent = 'Switch to repo selection';
      switchBtn.addEventListener('click', () => {
        snapshotPendingRestore();
        const hfCard = document.querySelector('.model-source-card[data-source="hf"]');
        if (hfCard) hfCard.click();
        const repoInput = document.getElementById('spawn-hf-repo');
        if (repoInput && !repoInput.value) {
          // Best-effort prefill: RapidMlxModelSource::Alias's value isn't guaranteed to be an
          // HF-repo-shaped string, so this seeds a search query rather than a resolved repo id.
          repoInput.value = wizardState.model.path || '';
        }
        showStep(0);
      });
      bodyEl.appendChild(document.createElement('br'));
      bodyEl.appendChild(switchBtn);
    }
    return;
  }
}

// Sets the 'degraded' status if `reasons` (from a launch or command-preview response) contains
// the ChatTemplateDegradeReason::AliasSource code. No-op otherwise — callers that already know
// the launch is clean should not clear a status set by autoInstallChatTemplate.
export function applyChatTemplateDegradeFromReasons(reasons) {
  const hit = (reasons || []).find(r => typeof r === 'string' && r.includes('[alias_source]'));
  if (!hit) return false;
  const family = wizardState.model.family || null;
  const tpl = getDefaultTemplateForFamily(family);
  _renderChatTemplateStatus('degraded', family, tpl, null);
  return true;
}
