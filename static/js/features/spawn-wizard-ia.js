// Generic tier-driven information-architecture (IA) relocation engine (plan
// §5 Phase 4 item 2): extracted from spawn-wizard-mlx-ia.js so the same
// group-relocation/tier-disclosure behavior can back both loaders. Each
// caller gets its own instance (own DOM state) via createWizardIA(), so
// llama.cpp and Rapid-MLX can be configured independently without sharing
// module-level singletons.
//
// Presentation only: existing DOM ids remain the serialization contract.
// This module only regroups a flat field dump into progressive, labelled
// <details> sections at runtime.

export function isOpenForProfile(critical, profile) {
  // critical true = auto-open everywhere; critical false = open only at Advanced
  if (critical) return true;
  return (profile ?? 'balanced') === 'advanced';
}

function rowForControl(id) {
  const el = document.getElementById(id);
  if (!el) return null;
  if (el.classList.contains('hardware-field') || el.classList.contains('kv-inline-row')) return el;
  return el.closest('.hardware-field, .kv-inline-row') || el;
}

// createWizardIA(config) returns an instance with its own relocation state.
//   config.groupClassName      — class applied to each built <details> group
//   config.rowClassName        — class applied to each relocated field row
//   config.originAnchorComment — comment text used for the "return home" anchor
export function createWizardIA(config) {
  const groupClassName = config.groupClassName || 'wiz-ia-group';
  const rowClassName = config.rowClassName || 'wiz-ia-row';
  const originAnchorComment = config.originAnchorComment || 'wiz-ia-origin';

  const originalPositions = new Map();
  let iaContainer = null;
  let hiddenSources = [];

  function rememberPosition(row) {
    if (originalPositions.has(row) || !row.parentNode) return;
    const anchor = document.createComment(originAnchorComment);
    row.parentNode.insertBefore(anchor, row);
    originalPositions.set(row, anchor);
  }

  function restorePositions() {
    originalPositions.forEach((anchor, row) => {
      if (anchor.parentNode) anchor.parentNode.insertBefore(row, anchor.nextSibling);
      row.classList.remove(rowClassName);
    });
    originalPositions.clear();
  }

  function hideSource(el) {
    if (!el || el.dataset.wizIaHidden) return;
    el.dataset.wizIaHidden = el.style.display || '(default)';
    el.style.display = 'none';
    hiddenSources.push(el);
  }

  function restoreSources() {
    hiddenSources.forEach(el => {
      const prev = el.dataset.wizIaHidden;
      el.style.display = prev === '(default)' ? '' : prev;
      delete el.dataset.wizIaHidden;
    });
    hiddenSources = [];
  }

  // A prebuilt group relocates an existing, already-structured <details>
  // element (e.g. llama.cpp's nested speculative-decoding block, which has
  // its own internal conditional-visibility wiring) instead of flattening
  // its fields into a freshly built container — avoids re-deriving that
  // wiring while still making the group tier-driven like every other one.
  function relocatePrebuiltGroup(group, profile) {
    const el = document.getElementById(group.prebuiltId);
    if (!el) return null;
    rememberPosition(el);
    el.classList.add('mlx-wiz-group', rowClassName);
    el.dataset.mlxWizGroup = group.id;
    el.dataset.mlxWizCritical = String(group.critical ?? true);
    el.open = isOpenForProfile(group.critical ?? true, profile);
    return el;
  }

  function buildGroup(group, profile) {
    if (group.prebuiltId) return relocatePrebuiltGroup(group, profile);
    const container = document.createElement('details');
    container.className = `${groupClassName} mlx-wiz-group`;
    container.dataset.mlxWizGroup = group.id;
    container.dataset.mlxWizCritical = String(group.critical ?? true);
    container.open = isOpenForProfile(group.critical ?? true, profile);
    const header = document.createElement('summary');
    header.className = 'mlx-native-group-header';
    const title = document.createElement('h4');
    title.className = 'mlx-native-group-title';
    title.textContent = group.title;
    const description = document.createElement('p');
    description.className = 'mlx-native-group-description';
    description.textContent = group.description;
    header.append(title, description);
    if (!group.critical) {
      const badge = document.createElement('span');
      badge.className = 'mlx-native-group-badge';
      badge.textContent = 'Advanced';
      header.appendChild(badge);
    }
    container.appendChild(header);
    let any = false;
    group.controls.forEach(id => {
      const row = rowForControl(id);
      if (!row) return;
      rememberPosition(row);
      row.classList.add(rowClassName);
      container.appendChild(row);
      any = true;
    });
    return any ? container : null;
  }

  // configure(root, enabled, profile, groups, supersections) builds (or tears
  // down) the grouped IA inside `advancedFieldsSelector`'s matched element.
  function configure(root, enabled, profile, groups, supersections, advancedFieldsSelector) {
    const advancedFields = (root || document).querySelector(advancedFieldsSelector);
    if (!advancedFields) return;

    if (!enabled) {
      if (iaContainer) {
        iaContainer.remove();
        iaContainer = null;
      }
      restorePositions();
      restoreSources();
      return;
    }

    if (iaContainer) {
      applyTierVisibility(root, profile);
      return; // already built for this activation
    }

    const introTitle = advancedFields.querySelector('.wizard-section-title');
    const introHint = advancedFields.querySelector('.field-hint');
    hideSource(introTitle);
    hideSource(introHint);
    advancedFields.querySelectorAll('.hardware-grid').forEach(hideSource);

    iaContainer = document.createElement('div');
    iaContainer.className = 'mlx-wiz-ia';

    supersections.forEach(super_ => {
      const groupsForSuper = groups.filter(g => g.supersection === super_.id);
      const builtGroups = groupsForSuper.map(g => buildGroup(g, profile)).filter(Boolean);
      if (!builtGroups.length) return;
      const section = document.createElement('section');
      section.className = 'mlx-wiz-supersection';
      section.dataset.mlxWizSuper = super_.id;
      const heading = document.createElement('h3');
      heading.className = 'mlx-wiz-supersection-title';
      heading.textContent = super_.title;
      const badge = document.createElement('span');
      badge.className = 'mlx-wiz-changed-badge';
      badge.hidden = true;
      heading.appendChild(badge);
      const desc = document.createElement('p');
      desc.className = 'mlx-wiz-supersection-desc';
      desc.textContent = super_.description;
      section.append(heading, desc, ...builtGroups);
      iaContainer.appendChild(section);
    });

    advancedFields.appendChild(iaContainer);
    // "All settings (N changed)" (plan §3): count fields inside this
    // supersection whose live value differs from its shipped default, so a
    // user can tell at a glance whether the collapsed group hides anything
    // non-default without opening it.
    captureDefaults(iaContainer);
    refreshChangedBadges(iaContainer);
    iaContainer.addEventListener('input', () => refreshChangedBadges(iaContainer));
    iaContainer.addEventListener('change', () => refreshChangedBadges(iaContainer));
  }

  function captureDefaults(container) {
    container.querySelectorAll('select, input').forEach(el => {
      if (el.dataset.wizDefaultCaptured) return;
      el.dataset.wizDefaultCaptured = '1';
      if (el.type === 'checkbox') el.dataset.wizDefault = el.defaultChecked ? '1' : '0';
      else el.dataset.wizDefault = el.tagName === 'SELECT'
        ? (Array.from(el.options).find(o => o.defaultSelected)?.value ?? el.options[0]?.value ?? '')
        : el.defaultValue;
    });
  }

  function isControlChanged(el) {
    if (el.type === 'checkbox') return (el.checked ? '1' : '0') !== el.dataset.wizDefault;
    return el.value !== el.dataset.wizDefault;
  }

  function refreshChangedBadges(container) {
    container.querySelectorAll('.mlx-wiz-supersection').forEach(section => {
      const n = section.querySelectorAll('select, input').length
        ? Array.from(section.querySelectorAll('select, input')).filter(isControlChanged).length
        : 0;
      const badge = section.querySelector('.mlx-wiz-changed-badge');
      if (!badge) return;
      badge.hidden = n === 0;
      badge.textContent = n === 0 ? '' : `${n} changed from default`;
    });
  }

  // Scoped to this instance's own iaContainer only — llama.cpp and MLX each
  // hold a separate createWizardIA() instance, and their groups share the
  // 'mlx-wiz-group' class, so a document-wide query here would cross-toggle
  // the other loader's groups.
  function applyTierVisibility(_root, profile) {
    if (!iaContainer) return;
    iaContainer.querySelectorAll('.mlx-wiz-group[data-mlx-wiz-critical]').forEach(el => {
      const critical = el.dataset.mlxWizCritical === 'true';
      el.open = isOpenForProfile(critical, profile);
    });
  }

  return { configure, applyTierVisibility };
}
