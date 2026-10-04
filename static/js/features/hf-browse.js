// ── HF Browse Module ──────────────────────────────────────────────────────────
// Shared HuggingFace browse, search, file-list, and download utilities.
// Consumed by spawn-wizard.js and (future) models-modal.

import { showToast } from './toast.js';

// ── Discovery scopes (Phase 8B1) ──────────────────────────────────────────────
// Additive toggles: MLX and GGUF can both be active. All = everything (including NVFP4/unsupported).
// Platform defaults: macOS → MLX+GGUF; Win/Linux → GGUF

export const HF_SCOPE = {
  GGUF: 'gguf',
  MLX: 'mlx',
  ALL: 'all',
};

// Guide tooltips for scope buttons
export const HF_SCOPE_TOOLTIPS = {
  [HF_SCOPE.MLX]: 'Rapid-MLX native format. Faster inference on Apple Silicon. macOS only.',
  [HF_SCOPE.GGUF]: 'GGUF format. Works everywhere via llama.cpp. Widely available.',
  [HF_SCOPE.ALL]: 'Show everything, including formats not yet supported (NVFP4, etc.).',
};

// ── Sorting modes (Phase 8B1) ─────────────────────────────────────────────────

export const HF_SORT = {
  RELEVANCE: 'relevance',
  NAME: 'name',
  SIZE: 'size',
  LAST_UPDATED: 'last_updated',
  DOWNLOADS: 'downloads',
};

// ── Category mapping from HF tags (Phase 8B1) ─────────────────────────────────

const HF_TAG_TO_CATEGORY = {
  'text-generation-inference': 'chat',
  'conversational': 'chat',
  'code-generation': 'coding',
  'code': 'coding',
  'instruct': 'chat',
  'chat': 'chat',
  'roleplay': 'roleplay',
  'creative-writing': 'roleplay',
  'story': 'roleplay',
  'role-playing': 'roleplay',
  'function-calling': 'tool-use',
  'tool-use': 'tool-use',
  'mcp': 'tool-use',
  'agentic': 'tool-use',
  'image-to-text': 'vision',
  'multimodal': 'vision',
  'document-question-answering': 'vision',
  'image-text-to-text': 'vision',
};

function resolveCategories(tags) {
  if (!Array.isArray(tags)) return [];
  const cats = new Set();
  for (const tag of tags) {
    const lower = tag.toLowerCase();
    const mapped = HF_TAG_TO_CATEGORY[lower];
    if (mapped) cats.add(mapped);
  }
  return [...cats];
}

// ── Author/converter role (Phase 8B1) ─────────────────────────────────────────

// Roles come from the CommunitySourceCatalog on the server, not from a list in this file.
// The list that used to live here classified `Qwen/` as a converter — Qwen is the original
// author of Qwen — and it had three roles against the catalog's seven, with no way for the
// user to correct either. The catalog is editable and evidence-bearing; this is just its view.
//
// `_catalogPromise` is the in-flight or settled load. A failed load caches `null` rather than
// retrying per card, so a server hiccup costs the badges, not the search.
let _catalogPromise = null;
let _sourcesByUsername = null;
let _roleMeta = null;

async function _fetchCommunitySources() {
  try {
    const res = await fetch('/api/hf/community-sources', { headers: getAuthHeaders() });
    if (!res.ok) return null;
    const data = await res.json();
    if (!data?.ok) return null;

    const byUser = new Map();
    for (const e of data.catalog?.entries || []) {
      const key = (e.username || '').toLowerCase();
      if (!key) continue;
      if (!byUser.has(key)) byUser.set(key, []);
      byUser.get(key).push(e);
    }
    const roles = new Map();
    for (const r of data.roles || []) roles.set(r.id, r);
    _sourcesByUsername = byUser;
    _roleMeta = roles;
    return byUser;
  } catch {
    return null;
  }
}

/** Load the catalog once per page. Safe to call repeatedly. */
export function ensureCommunitySourceCatalog() {
  if (!_catalogPromise) _catalogPromise = _fetchCommunitySources();
  return _catalogPromise;
}

/** Test seam: drop the cached catalog so the next call refetches. */
export function _resetCommunitySourceCatalog() {
  _catalogPromise = null;
  _sourcesByUsername = null;
  _roleMeta = null;
}

// Role ids are snake_case on the wire (the Rust enum's serde form); the badge CSS classes are
// kebab-case.
function _roleBadge(roleId) {
  const meta = _roleMeta?.get(roleId);
  // If the catalog could not be loaded the label is derived from the id. Sentence-case it
  // rather than keeping a second copy of the labels here: deriving the presentation is a
  // formatting rule, whereas a hardcoded label table would be the duplicated vocabulary this
  // change exists to remove, and it would drift from the enum the first time a role is added.
  const derived = roleId.replace(/_/g, ' ');
  return {
    role: roleId.replace(/_/g, '-'),
    label: meta?.label || derived.charAt(0).toUpperCase() + derived.slice(1),
    description: meta?.description || '',
  };
}

// Exported for the library cards in models.js, which resolve the same question about a
// downloaded model's repo. A second implementation there would be a second place for the
// Qwen-is-a-converter class of bug to reappear.
export function resolveAuthorRole(repoId, tags) {
  const lowerTags = (tags || []).map(t => t.toLowerCase());
  const hasMlxTag = lowerTags.some(t => t.includes('mlx'));
  const hasGgufTag = lowerTags.some(t => t.includes('gguf') || t.includes('gguf-file'));

  const parts = (repoId || '').split('/');
  const owner = parts[0] || '';
  const repoName = (parts[1] || '').toLowerCase();

  const entries = _sourcesByUsername?.get(owner.toLowerCase());
  if (entries?.length) {
    // Several catalog entries can share a username, and one entity can hold several roles
    // (Unsloth authors finetunes *and* quantizes). The repo's own format tags say which role
    // applies to *this* repo, so prefer a role the owner is actually known for and that the
    // tags corroborate; otherwise fall back to their primary entry.
    const held = new Set();
    for (const e of entries) {
      held.add(e.role);
      for (const r of e.also_known_for || e.alsoKnownFor || []) held.add(r);
    }
    if (hasMlxTag && held.has('mlx_converter')) return _roleBadge('mlx_converter');
    if (hasGgufTag && held.has('gguf_quantizer')) return _roleBadge('gguf_quantizer');
    return _roleBadge(entries[0].role);
  }

  // Not in the catalog. Fall back to the repo name, which is the only evidence available:
  // a `-gguf`/`-mlx`/`quant` suffix marks a derived repo, anything else reads as first-party.
  if (owner && repoName && !repoName.includes('-gguf') && !repoName.includes('-mlx') && !repoName.includes('quant')) {
    return _roleBadge('original_author');
  }

  return null;
}

// ── Discover categories ───────────────────────────────────────────────────────

export const HF_DISCOVER_CATEGORIES = [
  { id: 'trending',  label: 'Trending',      params: { query: '',           sort: 'trending',  limit: 30 } },
  { id: 'qwen3',     label: 'Qwen3',         params: { query: 'qwen3',      sort: 'downloads', limit: 30 } },
  { id: 'llama3',    label: 'Llama 3.x',     params: { query: 'llama-3',    sort: 'downloads', limit: 30 } },
  { id: 'mistral',   label: 'Mistral / MoE', params: { query: 'mistral',    sort: 'downloads', limit: 30 } },
  { id: 'gemma',     label: 'Gemma',         params: { query: 'gemma',      sort: 'downloads', limit: 30 } },
  { id: 'exaone',    label: 'EXAONE',        params: { query: 'exaone',     sort: 'downloads', limit: 30 } },
  { id: 'heretic',   label: 'Heretic',       params: { query: 'heretic',    sort: 'downloads', limit: 30 } },
];

// ── Small utilities ───────────────────────────────────────────────────────────

function escHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function hfRelativeAge(iso) {
  if (!iso) return '';
  const ms = Date.now() - new Date(iso).getTime();
  if (isNaN(ms) || ms < 0) return '';
  const mins  = Math.floor(ms / 60_000);
  const hours = Math.floor(ms / 3_600_000);
  const days  = Math.floor(ms / 86_400_000);
  const weeks = Math.floor(days / 7);
  const months = Math.floor(days / 30);
  const years  = Math.floor(days / 365);
  if (mins  <  60)  return `${mins}m ago`;
  if (hours < 24)   return `${hours}h ago`;
  if (days  <  7)   return `${days}d ago`;
  if (weeks <  5)   return `${weeks}w ago`;
  if (months < 12)  return `${months}mo ago`;
  return `${years}y ago`;
}

function formatBytes(bytes) {
  if (!bytes) return '';
  const b = Number(bytes);
  if (!isFinite(b)) return '';
  if (b >= 1024 ** 3) return (b / (1024 ** 3)).toFixed(1) + ' GiB';
  if (b >= 1024 ** 2) return (b / (1024 ** 2)).toFixed(1) + ' MiB';
  if (b >= 1024) return (b / 1024).toFixed(0) + ' KiB';
  return b + ' B';
}

function getRecommendedQuant(vramGb) {
  if (vramGb < 8)  return 'Q4_K_M';
  if (vramGb <= 16) return 'Q5_K_M';
  if (vramGb <= 24) return 'Q5_K_M';
  return 'Q8_0';
}

function getAuthHeaders() {
  return window.authHeaders ? window.authHeaders() : {};
}

// Trailing quant marker on an MLX repo name (after any "-mlx" suffix is removed).
const MLX_QUANT_SUFFIX_RE =
  /-(?:\d+(?:\.\d+)?-?bit|\d+bpw|mxfp\d+|nvfp\d+|fp\d+|bf16|qx\d+|q\d(?:_[a-z0-9_]+)?|dwq\d*|awq|gptq)(?:-hi)?$/i;

function isMlxModel(m) {
  return m.format === 'mlx' || (m.tags || []).includes('mlx') || /-mlx$/i.test(m.id || '');
}

// "nightmedia/Qwen3.8-27B-MindMeld-AREX-mxfp4-mlx" -> stem "Qwen3.8-27B-MindMeld-AREX", quant "mxfp4".
// A repo with no recognised quant marker (the bf16 base) has quant ''.
function splitMlxRepoName(repoId) {
  const name = (repoId || '').split('/')[1] || repoId || '';
  const noMlx = name.replace(/-mlx$/i, '');
  const match = noMlx.match(MLX_QUANT_SUFFIX_RE);
  if (!match) return { stem: noMlx, quant: '' };
  return { stem: noMlx.slice(0, match.index), quant: match[0].slice(1).toLowerCase() };
}

// What distinguishes one MLX repo from its siblings in a group: the quant tag the author
// wrote (mxfp4, qx64-hi ...), an edition tag for anything else in the name (AREX), and the
// measured bits per weight. `label` is unique within a group and feeds the quant advisor.
function mlxVariantInfo(m, siblings = []) {
  const { stem, quant } = splitMlxRepoName(m.id);
  const lowerStem = stem.toLowerCase();
  const base = siblings.reduce((shortest, s) => {
    const other = splitMlxRepoName(s.id).stem.toLowerCase();
    return other.length < shortest.length && lowerStem.startsWith(other) ? other : shortest;
  }, lowerStem);
  const edition = stem.slice(base.length).replace(/^-+/, '');
  const bpw = m.model_size_bytes > 0 && m.param_b > 0 ? (m.model_size_bytes * 8) / (m.param_b * 1e9) : 0;
  const tag = quant || (bpw >= 14 ? 'bf16' : 'base');
  return { edition, tag, bpw, label: edition ? `${edition} ${tag}` : tag };
}

// Same-author MLX conversions of one model belong together even when one repo carries an
// extra edition tag (…-AREX-mxfp4-mlx beside …-qx64-hi-mlx beside the bf16 base). Fold a repo
// into the shortest same-author stem that is a hyphen-bounded prefix of its own.
function mlxFamilyRoots(models) {
  const authorOf = m => ((m.id || '').split('/')[0] || '').toLowerCase();
  const stemsByAuthor = new Map();
  for (const m of models) {
    if (!isMlxModel(m)) continue;
    const author = authorOf(m);
    if (!stemsByAuthor.has(author)) stemsByAuthor.set(author, new Set());
    stemsByAuthor.get(author).add(splitMlxRepoName(m.id).stem.toLowerCase());
  }
  const roots = new Map();
  for (const m of models) {
    if (!isMlxModel(m)) continue;
    const author = authorOf(m);
    const stem = splitMlxRepoName(m.id).stem.toLowerCase();
    let root = stem;
    for (const other of stemsByAuthor.get(author)) {
      if (other.length < root.length && stem.startsWith(`${other}-`)) root = other;
    }
    roots.set(m, extractBaseModelName(`${author}/${root}`));
  }
  return roots;
}

// ── Phase 8B2: Base model name extraction ────────────────────────────────────
// Extract canonical base model name from repo_id for grouping.
// E.g., "bartowski/Qwen3-32B-GGUF" -> "Qwen3-32B"
// E.g., "mlx-community/Qwen3-30B-A3B-4bit" -> "Qwen3-30B-A3B"
// E.g., "unsloth/Qwen3.6-32B-heretic" -> "Qwen3.6-32B-heretic"

function extractBaseModelName(repoId) {
  const parts = (repoId || '').split('/');
  const name = (parts[1] || '').toLowerCase();
  if (!name) return repoId;

  // Remove GGUF suffix
  let base = name.replace(/-gguf$/, '');

  // Remove the trailing "-mlx" marker, then any quant suffix, so that the sibling
  // conversions of one model (…-mxfp4-mlx, …-mxfp8-mlx, …-4bit, …-q8-hi-mlx) share a
  // base name and group together as variants.
  base = base.replace(/-mlx$/, '');
  base = base.replace(MLX_QUANT_SUFFIX_RE, '');

  // Remove known converter prefixes from repo name
  base = base.replace(/^(?:mlx-|gguf-)/, '');

  // Capitalize sensibly: preserve known brand casing
  const lower = base;
  let result = '';
  let i = 0;
  while (i < lower.length) {
    const c = lower[i];
    if (c === '-' || c === '.') {
      result += c;
      i++;
    } else if (result.length === 0 || (result[result.length - 1] === '-' || result[result.length - 1] === '.')) {
      result += c.toUpperCase();
      i++;
    } else {
      result += c;
      i++;
    }
  }

  return result || repoId;
}

// ── Phase 8B2: MLX lineage discovery ──────────────────────────────────────────
// Fetch MLX derivatives and conversion recipes for a source repo.

async function fetchMlxLineage(repoId) {
  try {
    const headers = { ...getAuthHeaders(), 'Content-Type': 'application/json' };
    const resp = await fetch('/api/hf/mlx-derivatives', {
      method: 'POST',
      headers,
      body: JSON.stringify({ repoId }),
    });
    if (!resp.ok) return null;
    return await resp.json();
  } catch {
    return null;
  }
}

// ── Phase 8B2: Selection payload builder ──────────────────────────────────────
// Build structured selection payload with repoId/revision/variant/lineage.
// This survives through: selection → estimate → download → library → launch.

export function buildSelectionPayload(model, variantInfo) {
  return {
    repoId: model.id || '',
    revision: model.revision || variantInfo?.revision || null,
    variant: variantInfo?.variant || '',
    originalAuthor: model.original_author || variantInfo?.originalAuthor || null,
    converter: model.converter || variantInfo?.converter || null,
    format: variantInfo?.format || model.format || 'gguf',
    backendHint: variantInfo?.backendHint || null,
    hfFilePath: model.hf_file_path || variantInfo?.hfFilePath || null,
    modelSizeBytes: model.model_size_bytes || variantInfo?.modelSizeBytes || null,
    // Preserve raw model object for backward compatibility
    _raw: model,
  };
}

// Expensive qualification and identity resolution is intentionally on-demand. Search results
// are discovery hints; this drawer is the post-selection truth surface.
export async function openHfEvidence(repoId, revision, backend = 'rapid_mlx', opener = null) {
  const { openEvidenceDrawer } = await import('./evidence-drawer.js');
  const pinned = /^[0-9a-f]{40}$/i.test(revision || '');
  const headers = { ...getAuthHeaders(), 'Content-Type': 'application/json' };
  const request = async (url, body) => {
    try {
      const response = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body) });
      const data = await response.json().catch(() => ({}));
      return response.ok ? data : { error: data.error || `${response.status} ${response.statusText}` };
    } catch (error) {
      return { error: error?.message || String(error) };
    }
  };
  const [qualification, identity] = await Promise.all([
    request('/api/hf/qualify', { repoId, revision: revision || '', backend }),
    request('/api/hf/identity', { repoId, revision: revision || '' }),
  ]);
  const errors = [qualification.error, identity.error, ...(qualification.errors || []), ...(identity.errors || [])].filter(Boolean);
  const qualified = qualification.backendQualified === true;
  openEvidenceDrawer({
    title: `${repoId} evidence`,
    status: errors.length || !qualified || !pinned ? 'caution' : 'good',
    summary: qualified
      ? `The repository is provisionally qualified for ${qualification.backendHint || backend}.`
      : 'This repository is not yet qualified for the selected backend.',
    consequence: pinned
      ? 'The evidence is bound to an immutable Hugging Face commit.'
      : 'This result is not reproducible because the selected revision is mutable or unavailable.',
    remediation: pinned ? '' : 'Resolve and select an immutable commit before granting trust or recording qualification.',
    evidence: [
      qualification.qualificationReason,
      qualification.config?.architecture ? `Architecture: ${qualification.config.architecture}` : '',
      qualification.config?.contextLength ? `Native context: ${Number(qualification.config.contextLength).toLocaleString()} tokens` : '',
      ...(identity.roles || []).map(role => `${role.role}: ${role.username} (${role.confidence})`),
    ].filter(Boolean),
    warnings: errors,
    provenance: [
      `Repository: ${repoId}`,
      `Revision: ${revision || qualification.revision || identity.revision || 'mutable default'}`,
      qualification.qualifiedAt ? `Qualified at: ${new Date(qualification.qualifiedAt * 1000).toISOString()}` : '',
      identity.resolutionConfidence ? `Identity confidence: ${identity.resolutionConfidence}` : '',
    ].filter(Boolean),
  }, opener);
}

// ── Scope resolution (Phase 8B1) ──────────────────────────────────────────────
// Additive toggles: MLX and GGUF can both be active. All = everything (including NVFP4/unsupported).

async function resolveScope({ mlxActive, ggufActive, allActive }) {
  if (allActive) {
    return { format: 'all', includeUnsupported: true };
  }
  if (mlxActive && ggufActive) {
    return { format: 'both', includeUnsupported: false };
  }
  if (mlxActive) {
    return { format: 'mlx', includeUnsupported: false };
  }
  // Default to GGUF if neither explicitly active
  return { format: 'gguf', includeUnsupported: false };
}

let _cachedPlatformBackend = null;

async function detectPlatformBackend() {
  if (_cachedPlatformBackend !== null) return _cachedPlatformBackend;

  try {
    const headers = getAuthHeaders();
    const resp = await fetch('/api/llama-binary/platform-info', { headers });
    if (resp.ok) {
      const data = await resp.json();
      _cachedPlatformBackend = data.rapid_mlx_local_available ? 'rapid_mlx' : 'llama_cpp';
      return _cachedPlatformBackend;
    }
  } catch {
    // non-fatal; default to llama_cpp
  }

  _cachedPlatformBackend = 'llama_cpp';
  return 'llama_cpp';
}

// ── Sort resolution (Phase 8B1) ──────────────────────────────────────────────
// Two different jobs, and conflating them is what made this dropdown inert.
//
// resolveSortParam picks which *page* of results HF hands back. HF can only order by
// downloads, likes, creation, last-modified, trending, or relevance — it cannot order by
// name or by repo size. For those two we ask for the widest useful pool (downloads) and do
// the ordering ourselves, because `SimpleModelInfo` already carries `last_modified` and
// `model_size_bytes`.
//
// compareModels then decides the order the user actually sees. Previously that was hardcoded
// to downloads-descending for both variants and groups, so every one of the five options
// rendered identically no matter what HF had been asked for.

function resolveSortParam(legacySort, hfSort) {
  // If legacySort is set (e.g. discover pills), prefer it over hfSort
  if (legacySort) {
    switch (legacySort) {
      case 'trending':    return 'trendingScore';
      case 'newest':      return 'lastModified';
      case 'relevance':   return 'relevance';
      case 'downloads':   return 'downloads';
      case 'likes':       return 'likes';
      default:            return legacySort;
    }
  }
  switch (hfSort) {
    case HF_SORT.RELEVANCE:    return 'relevance';
    case HF_SORT.LAST_UPDATED: return 'lastModified';
    case HF_SORT.DOWNLOADS:    return 'downloads';
    // No server-side equivalent; ordered client-side below.
    case HF_SORT.NAME:
    case HF_SORT.SIZE:         return 'downloads';
    default:                   return legacySort || 'downloads';
  }
}

// HF only reports repo bytes for MLX repos; a GGUF search result has `model_size_bytes:
// null` for every row. Sorting those by raw bytes puts every GGUF at zero and leaves the
// list in server order while claiming to be sorted by size — the same silent no-op this fix
// exists to remove. Fall back to parameter count, which is populated for GGUF and is the
// size the user is actually choosing between. Scaled to bytes-ish so the two never interleave
// wrongly when a mixed MLX+GGUF scope returns both kinds.
function sizeRank(m) {
  if (m.model_size_bytes) return m.model_size_bytes;
  return (m.param_b || 0) * 1e9;
}

// Comparator for the chosen sort mode. Returns null when the server's own ordering should
// stand — relevance has no client-side equivalent, so re-sorting it would discard the only
// ranking that mode exists to show.
function modelComparator(hfSort) {
  switch (hfSort) {
    case HF_SORT.NAME:
      return (a, b) => (a.id || '').localeCompare(b.id || '', undefined, { sensitivity: 'base' });
    case HF_SORT.SIZE:
      return (a, b) => sizeRank(b) - sizeRank(a);
    case HF_SORT.LAST_UPDATED:
      return (a, b) => String(b.last_modified || '').localeCompare(String(a.last_modified || ''));
    case HF_SORT.DOWNLOADS:
      return (a, b) => (b.downloads || 0) - (a.downloads || 0);
    case HF_SORT.RELEVANCE:
      return null;
    default:
      return (a, b) => (b.downloads || 0) - (a.downloads || 0);
  }
}

// A group's rank is its best-ranked member under the same comparator, so groups and the
// variants inside them agree about what "first" means.
function groupComparator(hfSort) {
  const cmp = modelComparator(hfSort);
  if (!cmp) return null;
  return ([, a], [, b]) => cmp([...a].sort(cmp)[0], [...b].sort(cmp)[0]);
}

// ── Phase 8B2: Create a variant row within a group ───────────────────────────
function createGroupVariant(m, container, bodyEl, onOpenCardPanel, onSelectModel, vramGb = 0, siblings = []) {
  const repoIdLower = (m.id || '').toLowerCase();
  const isMlx = m.format === 'mlx' || (m.tags || []).includes('mlx') || repoIdLower.includes('.mlx') || repoIdLower.includes('/mlx/') || repoIdLower.includes('-mlx-') || repoIdLower.endsWith('-mlx') || repoIdLower.includes('.safetensors');
  const isGguf = m.format === 'gguf' || repoIdLower.includes('.gguf') || repoIdLower.includes('-gguf') || repoIdLower.includes('/gguf/');
  const format = m.format || (isGguf ? 'gguf' : isMlx ? 'mlx' : 'unknown');

  const variant = document.createElement('div');
  variant.className = 'hf-sg-variant';

  // MLX rows lead with the quantization as a pill and keep only the edition tag as text: the
  // full repo id is long, identical up to its last segment, and was being truncated exactly
  // where the variants differ. The id stays available as the tooltip and accessible name.
  const mlxInfo = format === 'mlx' ? mlxVariantInfo(m, siblings) : null;
  if (mlxInfo) {
    const quantPill = document.createElement('span');
    quantPill.className = 'hf-sg-quant-pill';
    quantPill.textContent = mlxInfo.tag;
    quantPill.title = mlxInfo.bpw > 0
      ? `${mlxInfo.tag} \u00b7 about ${mlxInfo.bpw.toFixed(1)} bits per weight (from the repo size)`
      : mlxInfo.tag;
    variant.appendChild(quantPill);
  }

  // Repo name
  const nameSpan = document.createElement('span');
  nameSpan.className = 'hf-sg-variant-name';
  nameSpan.textContent = mlxInfo ? mlxInfo.edition : m.id;
  nameSpan.title = m.id;
  nameSpan.setAttribute('aria-label', m.id);
  variant.appendChild(nameSpan);

  // Format badge
  const fmtBadge = document.createElement('span');
  fmtBadge.className = `hf-sg-format-badge hf-sg-format-badge--${format}`;
  fmtBadge.textContent = format.toUpperCase();
  variant.appendChild(fmtBadge);

  // Role badge
  const roleInfo = resolveAuthorRole(m.id, m.tags || []);
  if (roleInfo) {
    const roleBadge = document.createElement('span');
    roleBadge.className = `hf-sg-role-badge hf-sg-role-badge--${roleInfo.role}`;
    roleBadge.textContent = roleInfo.label;
    // The role's own explanation, straight from the catalog, so the badge says what it means.
    if (roleInfo.description) roleBadge.title = roleInfo.description;
    variant.appendChild(roleBadge);
  }

  // Meta (size/downloads)
  const metaSpan = document.createElement('span');
  metaSpan.className = 'hf-sg-variant-meta';
  const metaParts = [];
  if (format === 'mlx' && m.quant_label) metaParts.push(m.quant_label);
  if (m.param_b > 0) metaParts.push(`${m.param_b}B`);
  if (format === 'mlx' && m.model_size_bytes) metaParts.push(formatBytes(m.model_size_bytes));
  if (m.downloads > 0) {
    metaParts.push(m.downloads >= 1000 ? `${(m.downloads / 1000).toFixed(1)}k` : `${m.downloads}`);
  }
  metaSpan.textContent = metaParts.join(' · ');
  if (metaParts.length > 0) variant.appendChild(metaSpan);

  // Card link button
  const cardLink = document.createElement('button');
  cardLink.type = 'button';
  cardLink.className = 'hf-sg-card-link';
  cardLink.title = 'View model card';
  cardLink.innerHTML = '<svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>';
  cardLink.addEventListener('click', e => {
    e.stopPropagation();
    if (onOpenCardPanel) onOpenCardPanel(m.id);
    else window.open(`https://huggingface.co/${escHtml(m.id)}`, '_blank', 'noopener');
  });
  variant.appendChild(cardLink);

  const evidenceBtn = document.createElement('button');
  evidenceBtn.type = 'button';
  evidenceBtn.className = 'hf-sg-card-link hf-sg-evidence-link';
  evidenceBtn.title = 'Explain qualification and lineage';
  evidenceBtn.setAttribute('aria-label', `Explain qualification and lineage for ${m.id}`);
  evidenceBtn.textContent = '?';
  evidenceBtn.addEventListener('click', async e => {
    e.stopPropagation();
    evidenceBtn.disabled = true;
    evidenceBtn.textContent = '…';
    await openHfEvidence(m.id, m.revision || null, format === 'mlx' ? 'rapid_mlx' : 'llama_cpp', evidenceBtn);
    evidenceBtn.disabled = false;
    evidenceBtn.textContent = '?';
  });
  variant.appendChild(evidenceBtn);

  // Selection handler for MLX variants (direct select)
  if (format === 'mlx') {
    const selectVariant = () => {
      container.querySelectorAll('.hf-sg-variant.selected, .hf-sr-file-item.selected').forEach(el => el.classList.remove('selected'));
      variant.classList.add('selected');
      if (onSelectModel) {
        onSelectModel({
          repoId: m.id,
          id: m.id,
          format: 'mlx',
          param_b: m.param_b || 0,
          quant_label: m.quant_label || '',
          model_size_bytes: m.model_size_bytes || 0,
          // Sibling conversions of the same model, so the Models tab can rank them.
          siblings: siblings
            .filter(s => s.model_size_bytes > 0)
            .map(s => ({ repoId: s.id, label: mlxVariantInfo(s, siblings).label, size: s.model_size_bytes })),
        });
      }
    };
    variant.addEventListener('click', selectVariant);
    variant.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectVariant(); }
    });
  } else {
    // GGUF: expand file list inline within the group body
    const filesContainer = document.createElement('div');
    filesContainer.className = 'hf-sr-files';
    filesContainer.style.display = 'none';

    let filesLoaded = false;
    const selectFileItem = (fileItem, file) => {
      container.querySelectorAll('.hf-sr-file-item.selected, .hf-sg-variant.selected').forEach(el => el.classList.remove('selected'));
      fileItem.classList.add('selected');
      variant.classList.add('selected');
      if (onSelectModel) {
        onSelectModel({
          repoId: m.id,
          id: m.id,
          format: 'gguf',
          param_b: m.param_b || 0,
          _file: file,
        });
      }
    };

    // Loads the file list and, on first load (i.e. triggered by the initial row
    // click), immediately selects the VRAM-recommended quant so a single click
    // on a GGUF result yields a working selection. The list stays open so the
    // user can click a different quant to override it.
    const loadFiles = async (autoSelectFirstLoad) => {
      if (filesLoaded) return;
      filesLoaded = true;
      filesContainer.style.display = '';
      filesContainer.innerHTML = '<div class="hf-file-loading">Loading files…</div>';

      try {
        const headers = { ...getAuthHeaders(), 'Content-Type': 'application/json' };
        const resp = await fetch('/api/hf/files', {
          method: 'POST',
          headers,
          body: JSON.stringify({ repo_id: m.id, format: 'gguf' }),
        });
        if (!resp.ok) {
          filesContainer.innerHTML = '<div class="hf-file-empty">Failed to load files.</div>';
          return;
        }
        const data = await resp.json();
        const files = (data.files || []).filter(Boolean);
        filesContainer.innerHTML = '';
        if (!files.length) {
          filesContainer.innerHTML = '<div class="hf-file-empty">No files found in this repo.</div>';
          return;
        }

        const recommendedLabel = vramGb > 0 ? getRecommendedQuant(vramGb) : null;
        let autoSelectTarget = null;
        let firstSelectable = null;

        for (const file of files) {
          const fname = file.path || file.name || '';
          if (!fname) continue;

          const fileItem = document.createElement('div');
          fileItem.className = 'hf-file-item hf-sr-file-item';
          fileItem.setAttribute('tabindex', '0');
          fileItem.setAttribute('role', 'button');
          fileItem.dataset.filename = fname;
          fileItem.dataset.repoId = m.id;

          const fName = document.createElement('span');
          fName.className = 'hf-file-name';
          fName.textContent = fname.split('/').pop() || fname;
          fileItem.appendChild(fName);

          const fMeta = document.createElement('span');
          fMeta.className = 'hf-file-meta';
          const fileMeta = [];
          if (file.size) fileMeta.push(formatBytes(file.size));
          if (file.label) fileMeta.push(file.label);
          if (recommendedLabel && file.label === recommendedLabel) fileMeta.push('✓ Recommended');
          fMeta.textContent = fileMeta.join(' · ');
          fileItem.appendChild(fMeta);

          fileItem.addEventListener('click', e => {
            e.stopPropagation();
            selectFileItem(fileItem, file);
          });

          if (!firstSelectable) firstSelectable = { fileItem, file };
          if (!autoSelectTarget && recommendedLabel && file.label === recommendedLabel) {
            autoSelectTarget = { fileItem, file };
          }

          filesContainer.appendChild(fileItem);
        }

        if (autoSelectFirstLoad) {
          const pick = autoSelectTarget || firstSelectable;
          if (pick) selectFileItem(pick.fileItem, pick.file);
        }
      } catch {
        filesContainer.innerHTML = '<div class="hf-file-empty">Failed to load files.</div>';
      }
    };

    variant.addEventListener('click', () => {
      const wasOpen = filesContainer.style.display !== 'none';
      const firstOpen = !filesLoaded;
      bodyEl.querySelectorAll('.hf-sr-files').forEach(fc => {
        if (fc !== filesContainer) fc.style.display = 'none';
      });
      if (!wasOpen) {
        if (!filesContainer.parentNode) {
          variant.parentNode.insertBefore(filesContainer, variant.nextSibling);
        }
        loadFiles(firstOpen);
      } else {
        filesContainer.style.display = 'none';
      }
    });
  }

  return variant;
}

// ── hfSearch ──────────────────────────────────────────────────────────────────
// Search HuggingFace models and render results into container.
//
// params:
//   query, author, sort, limit          – search params
//   scope                               – discovery scope (HF_SCOPE.GGUF/MLX/ALL)
//   hfSort                              – sorting mode (HF_SORT.RELEVANCE/NAME/SIZE/LAST_UPDATED/DOWNLOADS)
//   minParamB                           – minimum parameter count filter
//   cursor                              – pagination cursor
//   append                              – append results instead of replacing
//   container                           – DOM element to render into
//   filelistContainer                   – optional element to hide when showing results
//   quickpicksContainer                 – optional element holding quick-pick buttons (for loading/active state)
//   discoverPillsContainerId            – optional id of discover-pills container (for loading/active state)
//   onOpenCardPanel                     – (repoId) => void
//   onSelectModel                       – (model) => void  (called when user clicks a result row)

export async function hfSearch({
  query,
  author,
  sort,
  limit,
  mlxActive = true,
  ggufActive = true,
  allActive = false,
  hfSort = HF_SORT.DOWNLOADS,
  minParamB = 0,
  cursor = null,
  append = false,
  _cascadeDepth = 0,
  container,
  filelistContainer,
  quickpicksContainer,
  discoverPillsContainerId,
  onOpenCardPanel,
  onSelectModel,
  quantsOnly = false,
  vramGb = 0,
}) {
  if (!container) return;

  // Re-run exactly this search (same cursor/append) from a failure's Retry button.
  const retrySearch = () => hfSearch({
    query, author, sort, limit, mlxActive, ggufActive, allActive, hfSort, minParamB, cursor, append,
    _cascadeDepth, container, filelistContainer, quickpicksContainer,
    discoverPillsContainerId, onOpenCardPanel, onSelectModel, quantsOnly, vramGb,
  });

  // Kick the catalog load off alongside the search rather than before it; it is awaited below,
  // just before the results are rendered, so the badges never render against a half-loaded
  // catalog and the first search is not serialised behind an extra round trip.
  const catalogReady = ensureCommunitySourceCatalog();

  if (!append) {
    container.innerHTML = '<div class="hf-search-loading">Searching HuggingFace…</div>';
    container.style.display = '';
    if (filelistContainer) {
      filelistContainer.innerHTML = '';
      filelistContainer.classList.remove('visible');
    }
  } else {
    // Remove existing load-more button so we can append fresh results + new button
    container.querySelector('.hf-load-more-btn')?.remove();
    const loadingEl = document.createElement('div');
    loadingEl.className = 'hf-search-loading hf-load-more-loading';
    loadingEl.textContent = 'Loading more…';
    container.appendChild(loadingEl);
  }

  const clearPillLoading = () => {
    quickpicksContainer?.querySelectorAll('.hf-qp-btn').forEach(b => b.classList.remove('loading'));
    document.getElementById(discoverPillsContainerId)
      ?.querySelectorAll('.hf-discover-pill').forEach(p => p.classList.remove('loading'));
  };

  // Find the nearest overflow-scrollable ancestor
  const scrollToResults = () => {
    let p = container.parentElement;
    while (p) {
      const s = getComputedStyle(p);
      if ((s.overflowY === 'auto' || s.overflowY === 'scroll') && p.scrollHeight > p.clientHeight) {
        const cRect = container.getBoundingClientRect();
        const pRect = p.getBoundingClientRect();
        p.scrollTo({ top: p.scrollTop + cRect.top - pRect.top - 12, behavior: 'smooth' });
        return;
      }
      p = p.parentElement;
    }
  };

  try {
    // Map discovery scope to backend format param.
    // MLX+GGUF both active = both formats; All = everything including NVFP4/unsupported.
    const resolvedScope = await resolveScope({ mlxActive, ggufActive, allActive });
    const body = {
      query: query || '',
      author: author || undefined,
      sort: resolveSortParam(sort, hfSort),
      limit: limit || 20,
      cursor: cursor || undefined,
      format: resolvedScope.format,
      includeUnsupported: resolvedScope.includeUnsupported,
      quantsOnly: quantsOnly,
    };

    const headers = { ...getAuthHeaders(), 'Content-Type': 'application/json' };
    const resp = await fetch('/api/hf/search', { method: 'POST', headers, body: JSON.stringify(body) });
    if (!resp.ok) {
      clearPillLoading();
      container.querySelector('.hf-load-more-loading')?.remove();
      // The server explains rate limits (429) with a wait time; surface it as text.
      const failure = await resp.json().catch(() => ({}));
      _renderSearchFailure(container, {
        message: failure.error || `Search failed (HTTP ${resp.status}).`,
        retryAfterSecs: failure.retry_after_secs || 0,
        append,
        onRetry: retrySearch,
      });
      return;
    }
    const data = await resp.json();
    const allModels = data.models || [];
    const nextCursor = data.next_cursor || null;
    const hasMore = !!nextCursor;

    // Role badges are rendered synchronously from here on, so the catalog has to be in hand.
    await catalogReady;

    clearPillLoading();

    // Remove "loading more" placeholder if appending
    if (append) container.querySelector('.hf-load-more-loading')?.remove();
    else container.innerHTML = '';

    // Apply client-side param_b filter
    const models = minParamB > 0 ? allModels.filter(m => (m.param_b || 0) >= minParamB) : allModels;

    if (!models.length && !append) {
      const msg = minParamB > 0
        ? `No models \u2265${minParamB}B found. Try a lower size filter or "Load more".`
        : 'No models found.';
      const empty = document.createElement('div');
      empty.className = 'hf-search-empty';
      empty.textContent = msg;
      container.replaceChildren(empty);
      if (hasMore) {
        const moreBtn = _makeLoadMoreBtn(() => hfSearch({
          query, author, sort, limit, mlxActive, ggufActive, allActive, hfSort, minParamB, cursor: nextCursor, append: true,
          container, filelistContainer, quickpicksContainer,
          discoverPillsContainerId, onOpenCardPanel, onSelectModel,
                }));
        container.appendChild(moreBtn);
      }
      if (!append) scrollToResults();
      return;
    }

    // Order by the sort the user picked. Relevance keeps the server's ranking untouched.
    const variantCmp = modelComparator(hfSort);
    const sortedModels = variantCmp ? [...models].sort(variantCmp) : models;

    // Phase 8B2: Group models by base model name for hierarchical display
    const mlxFamilies = mlxFamilyRoots(sortedModels);
    const groups = new Map();
    for (const m of sortedModels) {
      const baseName = mlxFamilies.get(m) || extractBaseModelName(m.id);
      if (!groups.has(baseName)) {
        groups.set(baseName, []);
      }
      groups.get(baseName).push(m);
    }

    // Rank groups by their best member under the same comparator. Insertion order already
    // follows the server's ranking, which is what relevance wants preserved.
    const groupCmp = groupComparator(hfSort);
    const sortedGroupEntries = groupCmp
      ? [...groups.entries()].sort(groupCmp)
      : [...groups.entries()];

    // Render grouped results
    for (const [baseName, groupModels] of sortedGroupEntries) {
      // ── Group wrapper ──
      const groupEl = document.createElement('div');
      groupEl.className = 'hf-search-group';

      // ── Group header ──
      const header = document.createElement('div');
      header.className = 'hf-sg-header';

      const nameRow = document.createElement('div');
      nameRow.className = 'hf-sg-header-name';

      const baseNameSpan = document.createElement('span');
      baseNameSpan.className = 'hf-sg-base-name';
      baseNameSpan.textContent = baseName;
      baseNameSpan.title = baseName;
      nameRow.appendChild(baseNameSpan);

      // Category badges (union of all model tags in group)
      const categorySet = new Set();
      for (const m of groupModels) {
        for (const cat of resolveCategories(m.tags || [])) {
          categorySet.add(cat);
        }
      }
      if (categorySet.size > 0) {
        const catContainer = document.createElement('span');
        catContainer.className = 'hf-sg-categories';
        for (const cat of [...categorySet].slice(0, 3)) {
          const badge = document.createElement('span');
          badge.className = 'hf-sr-pill hf-sr-pill--category';
          badge.textContent = cat;
          catContainer.appendChild(badge);
        }
        nameRow.appendChild(catContainer);
      }

      // Toggle "button" — visually a pill, but the whole header row is clickable
      // (see header click listener below); this stays a <span> so it doesn't
      // trap the click as a separate hit target the user has to aim for.
      const toggleBtn = document.createElement('span');
      toggleBtn.className = 'hf-sg-toggle';
      toggleBtn.textContent = `+${groupModels.length} variants`;
      nameRow.appendChild(toggleBtn);
      header.appendChild(nameRow);
      header.setAttribute('tabindex', '0');
      header.setAttribute('role', 'button');
      groupEl.appendChild(header);

      // ── Group body (collapsed by default) ──
      const bodyEl = document.createElement('div');
      bodyEl.className = 'hf-sg-body';
      bodyEl.style.display = 'none';

      // Sort group models by downloads within group
      groupModels.sort((a, b) => (b.downloads || 0) - (a.downloads || 0));

      for (const m of groupModels) {
        const variant = createGroupVariant(m, container, bodyEl, onOpenCardPanel, onSelectModel, vramGb, groupModels);
        bodyEl.appendChild(variant);
      }

      groupEl.appendChild(bodyEl);

      // Expand/collapse on a click anywhere in the header row (not just the
      // small pill) — the pill previously ate the click, so clicking the
      // model name itself did nothing. When a group has exactly one variant,
      // expanding it also fires that variant's own click so a single click
      // on the header takes the user straight to a working selection.
      let groupOpened = false;
      const toggleGroup = () => {
        const isExpanded = bodyEl.style.display !== 'none';
        // Close other groups
        container.querySelectorAll('.hf-search-group .hf-sg-body').forEach(b => {
          if (b !== bodyEl) b.style.display = 'none';
        });
        bodyEl.style.display = isExpanded ? 'none' : '';
        toggleBtn.textContent = isExpanded ? `+${groupModels.length} variants` : `−${groupModels.length} variants`;
        if (!isExpanded && !groupOpened) {
          groupOpened = true;
          if (groupModels.length === 1) {
            bodyEl.querySelector('.hf-sg-variant')?.click();
          }
        }
      };
      header.addEventListener('click', toggleGroup);
      header.addEventListener('keydown', e => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleGroup(); }
      });

      container.appendChild(groupEl);
    }

    // If appending and we hit the limit, add load-more button
    if (append && models.length < allModels.length) {
      const moreBtn = _makeLoadMoreBtn(() => hfSearch({
        query, author, sort, limit, mlxActive, ggufActive, allActive, hfSort, minParamB, cursor: nextCursor, append: true,
        _cascadeDepth: 0,
        container, filelistContainer, quickpicksContainer,
        discoverPillsContainerId, onOpenCardPanel, onSelectModel,
            }));
      container.appendChild(moreBtn);
    }

    if (!append) scrollToResults();

    // Auto-cascade: if fewer than 10 results are visible and there are more pages,
    // silently fetch the next page (capped at 3 auto-fetches to avoid runaway).
    const visibleCount = container.querySelectorAll('.hf-search-result').length;
    if (hasMore && visibleCount < 10 && _cascadeDepth < 3) {
      hfSearch({
        query, author, sort, limit, mlxActive, ggufActive, allActive, hfSort, minParamB, cursor: nextCursor, append: true,
        _cascadeDepth: _cascadeDepth + 1,
        container, filelistContainer, quickpicksContainer,
        discoverPillsContainerId, onOpenCardPanel, onSelectModel,
            });
    }
  } catch (err) {
    clearPillLoading();
    container.querySelector('.hf-load-more-loading')?.remove();
    _renderSearchFailure(container, {
      message: 'Search failed: ' + (err.message || String(err)),
      append,
      onRetry: retrySearch,
    });
  }
}

/**
 * Render a search failure with a Retry button. When the server reports a wait time
 * (HTTP 429), the button is disabled and counts down so the user is never left at a
 * dead end. `append` keeps already-rendered results (the "Load more" case).
 */
function _renderSearchFailure(container, { message, retryAfterSecs = 0, append = false, onRetry }) {
  const box = document.createElement('div');
  box.className = 'hf-search-empty hf-search-failure';
  box.setAttribute('role', 'alert');
  const text = document.createElement('span');
  text.className = 'hf-search-failure-text';
  text.textContent = message;
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'hf-load-more-btn hf-search-retry-btn';
  box.append(text, btn);
  if (append) container.appendChild(box);
  else container.replaceChildren(box);

  let remaining = Number.isFinite(retryAfterSecs) ? Math.max(0, Math.ceil(retryAfterSecs)) : 0;
  const syncLabel = () => {
    btn.disabled = remaining > 0;
    btn.textContent = remaining > 0 ? `Retry in ${remaining}s` : 'Retry';
  };
  syncLabel();
  if (remaining > 0) {
    const timer = setInterval(() => {
      if (!btn.isConnected) { clearInterval(timer); return; }
      remaining -= 1;
      syncLabel();
      if (remaining <= 0) clearInterval(timer);
    }, 1000);
  }
  btn.addEventListener('click', () => {
    box.remove();
    onRetry();
  });
}

function _makeLoadMoreBtn(onClick) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'hf-load-more-btn';
  btn.textContent = 'Load more';
  btn.addEventListener('click', onClick);
  return btn;
}

// ── hfListFiles ───────────────────────────────────────────────────────────────
// List GGUF files for a repo and render into container.
//
// params:
//   repoId                              – HF repo ID
//   container                           – DOM element to render into
//   vramGb                              – available VRAM in GiB (for recommendation badge)
//   onOpenCardPanel                     – (repoId) => void
//   onSelectFile                        – (file, repoId) => void

export function getRecommendedMmproj(files) {
  return (files || []).find(file => file.is_mmproj && file.is_recommended_mmproj) || null;
}

export async function hfListFiles({
  repoId,
  container,
  vramGb,
  onOpenCardPanel,
  onSelectFile,
  onFilesLoaded,
}) {
  if (!container) return;

  container.innerHTML = '<div class="hf-file-loading">Loading GGUF files…</div>';
  container.classList.add('visible');

  try {
    const headers = { ...getAuthHeaders(), 'Content-Type': 'application/json' };
    const resp = await fetch('/api/hf/files', {
      method: 'POST',
      headers,
      body: JSON.stringify({ repo_id: repoId }),
    });
    if (!resp.ok) {
      container.innerHTML = '<div class="hf-file-empty">Failed to load files. Check the repo ID.</div>';
      return;
    }
    const data = await resp.json();
    const files = (data.files || []).filter(Boolean);

    if (onFilesLoaded) onFilesLoaded(files);

    container.innerHTML = '';
    if (!files.length) {
      container.innerHTML = '<div class="hf-file-empty">No GGUF files found in this repo.</div>';
      return;
    }

    let autoSelectFn = null;
    let firstSelectFn = null;

    for (const file of files) {
      const fname = file.path || file.name || '';
      if (!fname) continue;

      const item = document.createElement('div');
      item.className = 'hf-file-item';
      item.setAttribute('tabindex', '0');
      item.setAttribute('role', 'button');
      item.dataset.filename = fname;
      item.dataset.repoId = file.repo_id || repoId;
      item.dataset.size = file.size || '';
      item.dataset.label = file.label || '';
      if (file.is_mmproj) item.dataset.mmproj = '1';
      if (file.is_draft_assistant) item.dataset.isDraftModel = '1';
      if (file.is_recommended_mmproj) item.dataset.recommendedMmproj = '1';
      if (file.mmproj_recommendation) {
        item.dataset.mmprojRecommendation = file.mmproj_recommendation;
      }

      const nameSpan = document.createElement('span');
      nameSpan.className = 'hf-file-name';
      const displayName = fname.split('/').pop() || fname;
      nameSpan.textContent = displayName;
      nameSpan.title = displayName;

      const metaSpan = document.createElement('span');
      metaSpan.className = 'hf-file-size';
      const parts = [];
      if (file.size) parts.push(formatBytes(file.size));
      if (file.label) {
        parts.push(file.label);
        if (file.is_recommended_mmproj) {
          parts.push('\u2713 Family recommended');
        } else if (!file.is_mmproj && vramGb > 0 && file.label === getRecommendedQuant(vramGb)) {
          parts.push('\u2713 Recommended');
        }
      }
      metaSpan.textContent = parts.join(' \u00b7 ');

      const qt = file.quant_type || '';
      if (qt === 'imatrix' || file.is_imatrix) {
        const b = document.createElement('span');
        b.className = 'hf-file-badge hf-file-badge-imatrix';
        b.textContent = 'imatrix';
        b.title = 'Importance-matrix calibrated — better quality at same bpw (mradermacher style)';
        nameSpan.appendChild(b);
      } else if (qt === 'unsloth_dynamic') {
        const b = document.createElement('span');
        b.className = 'hf-file-badge hf-file-badge-ud';
        b.textContent = 'UD';
        b.title = 'Unsloth Dynamic — mixed bits per layer, excellent quality/size tradeoff';
        nameSpan.appendChild(b);
      }
      if (file.is_mmproj) {
        const b = document.createElement('span');
        b.className = 'hf-file-badge hf-file-badge-mmproj';
        b.textContent = 'mmproj';
        b.title = 'Vision projector — load alongside the main model for multimodal inference';
        nameSpan.appendChild(b);
        if (file.is_recommended_mmproj) {
          const recommended = document.createElement('span');
          recommended.className = 'hf-file-badge hf-file-badge-recommended';
          recommended.textContent = 'recommended';
          recommended.title = file.mmproj_recommendation || 'Preferred projector format for this model family';
          nameSpan.appendChild(recommended);
        }
      }
      if (file.is_draft_assistant) {
        const b = document.createElement('span');
         b.className = 'hf-file-badge hf-file-badge-draft';
        b.textContent = 'Assistant';
         b.title = 'MTP draft model — use as --model-draft for speculative decoding';
        nameSpan.appendChild(b);
      }

      item.appendChild(nameSpan);
      item.appendChild(metaSpan);

      const selectFile = () => {
        if (onSelectFile) onSelectFile(file, file.repo_id || repoId);
      };
      item.addEventListener('click', selectFile);
      item.addEventListener('keydown', e => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectFile(); }
      });

      if (!file.is_mmproj && !file.is_draft_assistant) {
        if (!firstSelectFn) firstSelectFn = selectFile;
        if (!autoSelectFn && file.label && vramGb > 0 && file.label === getRecommendedQuant(vramGb)) {
          autoSelectFn = selectFile;
        }
      }

      container.appendChild(item);
    }

    // Auto-select the VRAM-recommended quant (falling back to the first file)
    // so landing on a repo's file list already yields a working selection.
    const autoPick = autoSelectFn || firstSelectFn;
    if (autoPick) autoPick();
  } catch {
    container.innerHTML = '<div class="hf-file-empty">Error loading files. Check the repo ID and your HF token.</div>';
  }
}

// ── hfStartDownload ───────────────────────────────────────────────────────────
// Start a download and show progress in panelEl.
//
// params:
//   repoId, filePath                    – HF repo and file to download
//   panelEl                             – DOM element for the download panel
//   onComplete                          – (downloadId, localPath) => void
//   onValidationError                   – (msg) => void
//   onClearValidationError              – () => void (optional)

export async function hfStartDownload({
  repoId,
  filePath,
  panelEl,
  onComplete,
  onValidationError,
  onClearValidationError,
  companionId,
}) {
  if (!repoId || !filePath) {
    if (onValidationError) onValidationError('Select a GGUF file first.');
    return;
  }
  if (!panelEl) return;

  const btn = panelEl.querySelector('#hf-dlp-download-btn, #mm-hf-dlp-download-btn');
  const shortName = filePath.split('/').pop() || filePath;
  if (btn) {
    btn.disabled = true;
    btn.textContent = 'Starting\u2026';
  }

  try {
    const headers = { ...getAuthHeaders(), 'Content-Type': 'application/json' };
    const res = await fetch('/api/hf/download', {
      method: 'POST',
      headers,
      body: JSON.stringify({ repo_id: repoId, file_path: filePath, resume: true }),
    });
    const data = await res.json().catch(() => ({}));

    if (!res.ok || !data.ok) {
      if (btn) {
        btn.disabled = false;
        btn.textContent = 'Download to models folder';
      }
      const err = data.error || 'Download failed to start.';
      _handleDownloadError(err, shortName, onValidationError);
      return null;
    }

    // Download started: toast + progress UI.
    showToast('Download started: ' + shortName, 'info');

    const downloadId = data.download_id;

    const fileEl = panelEl.querySelector('#hf-dlp-progress-file, #mm-hf-dlp-progress-file');
    if (fileEl) fileEl.textContent = shortName;
    _dlSetState(panelEl, 'progress');
    hfPollDownload(downloadId, panelEl, { onComplete, onValidationError, onClearValidationError, companionId });
    return data;
  } catch (err) {
    if (btn) {
      btn.disabled = false;
      btn.textContent = 'Download to models folder';
    }
    if (onValidationError) onValidationError('Download request failed: ' + (err.message || err));
    return null;
  }
}

/**
 * Centralized handler for download-start errors from the backend.
 * Shows user-friendly toasts and keeps onValidationError in sync.
 */
function _handleDownloadError(err, shortName, onValidationError) {
  const e = (err || '').toLowerCase();

  if (e.includes('already downloading')) {
    const msg = 'Already downloading: ' + shortName + '. Waiting for it to complete.';
    showToast(msg, 'warning');
    if (onValidationError) onValidationError(err || msg);
    return;
  }

  if (e.includes('already exists')) {
    const msg = 'File already exists. It may already be in your library.';
    showToast(msg, 'warning');
    if (onValidationError) onValidationError(err || msg);
    return;
  }

  if (e.includes('too many downloads')) {
    const msg = 'Too many downloads in progress. Please wait for one to finish.';
    showToast(msg, 'warning');
    if (onValidationError) onValidationError(err || msg);
    return;
  }

  // Generic backend error.
  const userMsg = 'Download failed: ' + (err || 'unknown error');
  showToast(userMsg, 'error');
  if (onValidationError) onValidationError(err || userMsg);
}

export async function hfStartCompanionDownload({ repoId, filePath, saveAs }) {
  if (!repoId || !filePath) return null;
  try {
    const headers = { ...getAuthHeaders(), 'Content-Type': 'application/json' };
    const res = await fetch('/api/hf/download', {
      method: 'POST',
      headers,
      body: JSON.stringify({
        repo_id: repoId,
        file_path: filePath,
        save_as: saveAs || undefined,
        companion: true,
        resume: true,
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (res.ok && data.ok) {
      const shortName = (saveAs || filePath.split('/').pop()) || filePath;
      showToast('Also downloading: ' + shortName, 'info');
      return data;
    }
    // Log error quietly for companion; primary handler will surface main message.
    if (data.error) {
      console.warn('[hf] companion download failed:', data.error);
    }
    return null;
  } catch {
    return null;
  }
}

// ── hfPollDownload ────────────────────────────────────────────────────────────
// Poll download status and update progress in panelEl.
// Caller owns state updates via onComplete/onValidationError.

export function hfPollDownload(downloadId, panelEl, { onComplete, onValidationError, onClearValidationError, companionId }) {
  if (!panelEl) return;
  _dlCancelPoll(panelEl);

  const headers = getAuthHeaders();
  let consecutiveErrors = 0;
  const maxConsecutiveErrors = 5;
  let companionResolved = !companionId;

  async function checkCompanion() {
    if (!companionId || companionResolved) return;
    try {
      const cRes = await fetch(`/api/models/download/${companionId}/status`, { headers });
      if (!cRes.ok) return;
      const cData = await cRes.json().catch(() => ({}));
      const cStatus = cData?.status?.status || cData?.status;
      if (!cStatus) return;
      const { status: cStatusVal, message: cMessage } = cStatus;
      companionResolved = true;

      if (cStatusVal === 'completed') {
        const cName = (cStatus.local_path || '').split('/').pop() || 'mmproj';
        showToast('Also downloaded: ' + cName, 'info');
      } else if (cStatusVal === 'failed' || cStatusVal === 'cancelled') {
        const reason = cMessage || 'Companion download failed.';
        showToast('mmproj download issue: ' + reason, 'error');
      }
    } catch {
      // non-fatal
    }
  }

  async function poll() {
    try {
      const res = await fetch(`/api/models/download/${downloadId}/status`, { headers });
      if (!res.ok) {
        consecutiveErrors++;
        if (consecutiveErrors >= maxConsecutiveErrors) {
          _dlCancelPoll(panelEl);
          _dlSetState(panelEl, 'idle');
          if (onValidationError) onValidationError('Unable to check download status. It may have failed or been cancelled.');
          return;
        }
        _dlSchedulePoll(panelEl, poll, 1000);
        return;
      }
      const data = await res.json();
      const s = data.status;
      if (!s) {
        consecutiveErrors++;
        if (consecutiveErrors >= maxConsecutiveErrors) {
          _dlCancelPoll(panelEl);
          _dlSetState(panelEl, 'idle');
          if (onValidationError) onValidationError('Unable to check download status. It may have failed or been cancelled.');
          return;
        }
        _dlSchedulePoll(panelEl, poll, 1000);
        return;
      }

      consecutiveErrors = 0;

      const { status, bytes_downloaded = 0, total_bytes = 0, speed = 0, eta = 0 } = s;
      const pct = total_bytes > 0 ? Math.round(bytes_downloaded / total_bytes * 100) : 0;

      const bar = panelEl.querySelector('#hf-dlp-bar, #mm-hf-dlp-bar');
      if (bar) bar.style.width = pct + '%';

      const pctEl = panelEl.querySelector('#hf-dlp-progress-pct, #mm-hf-dlp-progress-pct');
      if (pctEl) pctEl.textContent = total_bytes > 0 ? `${pct}%` : '';

      const statsEl = panelEl.querySelector('#hf-dlp-stats, #mm-hf-dlp-stats');
      if (statsEl) {
        const mb = (bytes_downloaded / 1_048_576).toFixed(1);
        const tot = total_bytes > 0 ? ` / ${(total_bytes / 1_048_576).toFixed(0)} MB` : '';
        const spd = speed > 0 ? ` \u00b7 ${(speed / 1_048_576).toFixed(1)} MB/s` : '';
        const etaStr = eta > 0
          ? ` \u00b7 ETA ${eta < 60 ? eta + 's' : Math.round(eta / 60) + 'm'}`
          : '';
        statsEl.textContent = `${mb} MB${tot}${spd}${etaStr}`;
      }

      // Also check companion status while main is running
      if (status === 'running' && companionId && !companionResolved) {
        checkCompanion();
      }

      if (status === 'completed') {
        _dlCancelPoll(panelEl);
        _dlSetState(panelEl, 'complete');
        if (onClearValidationError) onClearValidationError();
        const localPath = data.status?.local_path || data.local_path;
        if (onComplete) onComplete(downloadId, localPath);
        return;
      }
      if (status === 'failed') {
        _dlCancelPoll(panelEl);
        _dlSetState(panelEl, 'idle');
        const reason = s.message || 'Download failed.';
        const shortName = data.status?.local_path
            ? (data.status.local_path.split('/').pop() || reason)
            : null;
        const retryable = reason.toLowerCase().includes('connection') ||
            reason.toLowerCase().includes('timed out') ||
            reason.toLowerCase().includes('retry') ||
            reason.toLowerCase().includes('error');
        if (retryable) {
          showToast('Download failed: ' + (shortName ? shortName + ' — ' : '') + reason + ' You can retry.', 'error');
        }
        if (onValidationError) onValidationError(reason);
        return;
      }
      if (status === 'cancelled') {
        _dlCancelPoll(panelEl);
        _dlSetState(panelEl, 'idle');
        return;
      }
    } catch {
      consecutiveErrors++;
      if (consecutiveErrors >= maxConsecutiveErrors) {
        _dlCancelPoll(panelEl);
        _dlSetState(panelEl, 'idle');
        if (onValidationError) onValidationError('Lost connection while checking download status.');
        return;
      }
    }
    _dlSchedulePoll(panelEl, poll, 1000);
  }

  _dlSchedulePoll(panelEl, poll, 800);
}

// ── hfCancelDownload ──────────────────────────────────────────────────────────
// Cancel an active download.

export async function hfCancelDownload({ downloadId, panelEl }) {
  if (!downloadId || !panelEl) return;
  const headers = getAuthHeaders();
  await fetch(`/api/models/download/${downloadId}/cancel`, { method: 'POST', headers }).catch(() => {});
  _dlCancelPoll(panelEl);
  _dlSetState(panelEl, 'idle');
}

// ── hfShowDownloadPanel ───────────────────────────────────────────────────────
// Show the download panel and set the idle state + destination path.

export async function hfShowDownloadPanel(panelEl, fname) {
  if (!panelEl) return;
  _dlSetState(panelEl, 'idle');
  panelEl.style.display = '';

  try {
    const headers = getAuthHeaders();
    const res = await fetch('/api/hf/download-dir', { headers });
    const data = res.ok ? await res.json() : null;
    const dir = data?.dir || '~/.config/local-llm-foundry/models';
    const configured = data?.configured ?? false;
    const destPath = dir.replace(/\/$/, '') + '/' + (fname || '').split('/').pop();

    const destEl = panelEl.querySelector('.hf-dlp-dest-path');
    if (destEl) { destEl.textContent = destPath; destEl.title = destPath; }

    const warnEl = panelEl.querySelector('.hf-dlp-warn');
    if (warnEl) warnEl.style.display = configured ? 'none' : '';
  } catch { /* ignore */ }
}

export function hfHideDownloadPanel(panelEl) {
  if (!panelEl) return;
  panelEl.style.display = 'none';
  _dlCancelPoll(panelEl);
}

// ── hfRenderDiscoverPills ─────────────────────────────────────────────────────
// Render discover pills into container.
//
// params:
//   container                           – DOM element to render into
//   quickpicksContainer                 – optional element holding quick-pick buttons
//   onPillClick                         – (cat, pillEl) => void  (called when a pill is clicked)

export function hfRenderDiscoverPills({ container, quickpicksContainer, onPillClick }) {
  if (!container) return;
  container.innerHTML = '';

  for (const cat of HF_DISCOVER_CATEGORIES) {
    const pill = document.createElement('button');
    pill.type = 'button';
    pill.className = 'hf-discover-pill';
    pill.textContent = cat.label;
    pill.dataset.catId = cat.id;

    pill.addEventListener('click', () => {
      container.querySelectorAll('.hf-discover-pill').forEach(p => p.classList.remove('active', 'loading'));
      quickpicksContainer?.querySelectorAll('.hf-qp-btn').forEach(b => b.classList.remove('active', 'loading'));
      pill.classList.add('active', 'loading');
      if (onPillClick) onPillClick(cat, pill);
    });

    container.appendChild(pill);
  }
}

// ── hfLoadQuickPicks ──────────────────────────────────────────────────────────
// Load quantizer quick-picks and render into container.
//
// params:
//   container                           – DOM element to render into
//   discoverPillsContainerId            – optional id of discover-pills container
//   onAuthorClick                       – (author, btnEl) => void  (called when a quick-pick is clicked)

export async function hfLoadQuickPicks({ container, discoverPillsContainerId, onAuthorClick }) {
  if (!container) return;
  try {
    const headers = getAuthHeaders();
    const resp = await fetch('/api/hf/quantizers', { headers });
    if (!resp.ok) return;
    const data = await resp.json();
    if (!data.ok || !data.quantizers) return;

    container.innerHTML = '';
    for (const q of data.quantizers) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'hf-qp-btn';
      if (q.quant_style === 'imatrix') btn.classList.add('hf-qp-imatrix');
      if (q.quant_style === 'ud') btn.classList.add('hf-qp-ud');
      if (q.quant_style === 'mlx') btn.classList.add('hf-qp-mlx');
      btn.textContent = q.display_name;
      btn.title = q.description + (q.note ? `\n\n${q.note}` : '');
      btn.dataset.author = q.username;

      btn.addEventListener('click', () => {
        container.querySelectorAll('.hf-qp-btn').forEach(b => b.classList.remove('active', 'loading'));
        document.getElementById(discoverPillsContainerId)
          ?.querySelectorAll('.hf-discover-pill').forEach(p => p.classList.remove('active', 'loading'));
        btn.classList.add('active', 'loading');
        if (onAuthorClick) onAuthorClick(q.username, btn);
      });

      container.appendChild(btn);
    }
  } catch { /* non-fatal */ }
}

// ── Internal helpers (download panel state) ───────────────────────────────────

function _dlSetState(panelEl, state) {
  ['idle', 'progress', 'complete'].forEach(s => {
    const el = panelEl.querySelector(`#hf-dlp-${s}, #mm-hf-dlp-${s}`);
    if (el) el.style.display = s === state ? '' : 'none';
  });
  if (state === 'idle') {
    const btn = panelEl.querySelector('#hf-dlp-download-btn, #mm-hf-dlp-download-btn');
    if (btn) {
      btn.disabled = false;
      btn.textContent = 'Download to models folder';
    }
  }
}

function _dlSchedulePoll(panelEl, fn, ms) {
  const existing = panelEl._hfDlPollTimer;
  if (existing) clearTimeout(existing);
  panelEl._hfDlPollTimer = setTimeout(fn, ms);
}

function _dlCancelPoll(panelEl) {
  if (panelEl._hfDlPollTimer) {
    clearTimeout(panelEl._hfDlPollTimer);
    panelEl._hfDlPollTimer = null;
  }
}

// ── Discovery scope selector (Phase 8B1) ──────────────────────────────────────
// Additive toggles: MLX and GGUF can both be active. All = everything (including NVFP4/unsupported).
// Returns the selector element. Caller sets initial state via container.dataset.
//
// params:
//   container            – DOM element to append into. Set dataset.hfScopeMlx="1" and/or dataset.hfScopeGguf="1" for initial active scopes.
//   onChange             – (mlxActive, ggufActive, allActive) => void called when scope changes

export function hfCreateScopeSelector({ container, onChange }) {
  if (!container) return null;
  console.log('[SCOPE-SELECTOR] creating, existing wraps:', container.querySelectorAll('.hf-scope-selector').length);
  // Remove any existing scope selector (initHfDownloadTab is called per tab click)
  container.querySelectorAll('.hf-scope-selector').forEach(el => el.remove());

  const wrap = document.createElement('div');
  wrap.className = 'hf-scope-selector';
  wrap.style.cssText = 'display:flex;gap:0;border-radius:6px;overflow:hidden;border:1px solid rgba(255,255,255,0.08);';

  const scopes = [
    { key: 'mlx', label: 'MLX', tooltip: HF_SCOPE_TOOLTIPS[HF_SCOPE.MLX] },
    { key: 'gguf', label: 'GGUF', tooltip: HF_SCOPE_TOOLTIPS[HF_SCOPE.GGUF] },
    { key: 'all', label: 'All', tooltip: HF_SCOPE_TOOLTIPS[HF_SCOPE.ALL] },
  ];

  // Read initial state from container dataset
  const initialMlx = container.dataset.hfScopeMlx === '1';
  const initialGguf = container.dataset.hfScopeGguf === '1';
  const initialAll = container.dataset.hfScopeAll === '1';

  // Initialize wrap.dataset to match initial state (required for correct toggle behavior)
  wrap.dataset.hfScopeMlx = initialMlx ? '1' : '';
  wrap.dataset.hfScopeGguf = initialGguf ? '1' : '';
  wrap.dataset.hfScopeAll = initialAll ? '1' : '';

  // Applies a scope state to dataset + button visuals, optionally notifying onChange. Shared by
  // the click handler and the setScope() setter so a programmatic change can't drift from a
  // user click's visual/dataset update logic.
  function applyScope(mlxActive, ggufActive, allActive, { notify = true } = {}) {
    wrap.dataset.hfScopeMlx = mlxActive ? '1' : '';
    wrap.dataset.hfScopeGguf = ggufActive ? '1' : '';
    wrap.dataset.hfScopeAll = allActive ? '1' : '';
    container.dataset.hfScopeMlx = mlxActive ? '1' : '';
    container.dataset.hfScopeGguf = ggufActive ? '1' : '';
    container.dataset.hfScopeAll = allActive ? '1' : '';

    wrap.querySelectorAll('.hf-scope-btn').forEach(b => {
      const isActive = b.dataset.scopeKey === 'mlx' ? mlxActive
        : b.dataset.scopeKey === 'gguf' ? ggufActive
        : b.dataset.scopeKey === 'all' ? allActive
        : false;
      setScopeBtnState(b, isActive);
    });

    if (notify && onChange) onChange(mlxActive, ggufActive, allActive);
  }

  for (const s of scopes) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'hf-scope-btn';
    btn.dataset.scopeKey = s.key;
    btn.textContent = s.label;
    btn.title = s.tooltip;
    btn.style.cssText =
      'padding:2px 8px;font-size:10px;font-weight:600;border:none;cursor:pointer;' +
      'color:rgba(255,255,255,0.5);background:transparent;transition:all 0.15s ease;';

    btn.addEventListener('click', () => {
      // Read current state
      let mlxActive = wrap.dataset.hfScopeMlx === '1';
      let ggufActive = wrap.dataset.hfScopeGguf === '1';
      let allActive = wrap.dataset.hfScopeAll === '1';

      if (s.key === 'all') {
        // All: toggle everything on/off (exclusive with individual toggles)
        allActive = !allActive;
        mlxActive = allActive;
        ggufActive = allActive;
      } else if (s.key === 'mlx') {
        // MLX: toggle independently
        mlxActive = !mlxActive;
        allActive = false; // deselect All if user modifies individual toggles
      } else if (s.key === 'gguf') {
        // GGUF: toggle independently
        ggufActive = !ggufActive;
        allActive = false; // deselect All if user modifies individual toggles
      }

      applyScope(mlxActive, ggufActive, allActive);
    });

    // Set initial state
    const active = s.key === 'all' ? initialAll
      : s.key === 'mlx' ? initialMlx
      : s.key === 'gguf' ? initialGguf
      : false;
    setScopeBtnState(btn, active);
    wrap.appendChild(btn);
  }

  container.appendChild(wrap);

  return {
    el: wrap,
    // Sets scope programmatically (e.g. engine-driven default). Does not fire onChange by
    // default — callers that need a search re-fire opt in explicitly (plan §1.5/§5 Phase 1.2:
    // "suppress search re-fire on programmatic scope change").
    setScope(mlxActive, ggufActive, allActive, { notify = false } = {}) {
      applyScope(mlxActive, ggufActive, allActive, { notify });
    },
  };
}

function setScopeBtnState(btn, active) {
  btn.classList.toggle('hf-scope-btn--active', active);
  btn.style.color = active ? '#fff' : 'rgba(255,255,255,0.5)';
  btn.style.background = active ? 'rgba(99,102,241,0.85)' : 'transparent';
}

// ── Sort selector (Phase 8B1) ─────────────────────────────────────────────────
// Create a sort mode selector for discovery views.
// Returns the selector element. Caller should read container.dataset.hfSearchSort.
//
// params:
//   container            – DOM element to append the selector into
//   defaultSort          – HF_SORT value used until the persisted choice arrives
//   onChange             – (sort) => void  called when sort changes
//
// The chosen mode is persisted to ui-settings.json, because a sort you have to re-pick on
// every visit is a sort you fight rather than use. The persisted value arrives after first
// paint, so the selector renders with `defaultSort` and corrects itself once settings load.

export function hfCreateSortSelector({ container, defaultSort = HF_SORT.DOWNLOADS, onChange }) {
  if (!container) return null;

  const wrap = document.createElement('div');
  wrap.className = 'hf-sort-selector';
  wrap.style.cssText = 'display:flex;align-items:center;gap:4px;font-size:10px;color:rgba(255,255,255,0.5);';

  const label = document.createElement('span');
  label.textContent = 'Sort:';
  wrap.appendChild(label);

  const select = document.createElement('select');
  select.className = 'hf-sort-select';
  select.style.cssText =
    'font-size:10px;padding:2px 6px;border-radius:4px;border:1px solid rgba(255,255,255,0.1);' +
    'background:rgba(0,0,0,0.3);color:rgba(255,255,255,0.8);cursor:pointer;';

  const sorts = [
    { value: HF_SORT.DOWNLOADS, label: 'Most downloaded' },
    { value: HF_SORT.RELEVANCE, label: 'Relevance' },
    { value: HF_SORT.NAME, label: 'Name' },
    { value: HF_SORT.SIZE, label: 'Size' },
    { value: HF_SORT.LAST_UPDATED, label: 'Last updated' },
  ];

  for (const s of sorts) {
    const opt = document.createElement('option');
    opt.value = s.value;
    opt.textContent = s.label;
    select.appendChild(opt);
  }

  // A caller may restore a sort mode from an older UI version. Never leave the
  // native select visually blank when that value is no longer offered.
  select.value = sorts.some(s => s.value === defaultSort) ? defaultSort : HF_SORT.DOWNLOADS;
  select.addEventListener('change', () => {
    container.dataset.hfSearchSort = select.value;
    persistDiscoverySort(select.value);
    if (onChange) onChange(select.value);
  });

  container.dataset.hfSearchSort = defaultSort;
  wrap.appendChild(select);
  container.appendChild(wrap);

  // Adopt the persisted choice when it arrives. Only re-search if it actually differs, so
  // the common case of the persisted value matching the default costs nothing.
  loadDiscoverySort().then(saved => {
    if (!saved || saved === select.value) return;
    if (!sorts.some(s => s.value === saved)) return;
    select.value = saved;
    container.dataset.hfSearchSort = saved;
    if (onChange) onChange(saved);
  });

  return wrap;
}

// ── Sort persistence ──────────────────────────────────────────────────────────
// Stored in ui-settings.json alongside the other UI preferences, read-modify-written the
// same way the context card does it. Failures are swallowed: losing a sort preference is
// not worth interrupting a search over.

async function loadDiscoverySort() {
  try {
    const resp = await fetch('/api/settings', { headers: getAuthHeaders() });
    if (!resp.ok) return null;
    const settings = await resp.json();
    return settings?.hf_discovery_sort || null;
  } catch {
    return null;
  }
}

async function persistDiscoverySort(sort) {
  try {
    const headers = { ...getAuthHeaders(), 'Content-Type': 'application/json' };
    const resp = await fetch('/api/settings', { headers: getAuthHeaders() });
    if (!resp.ok) return;
    const settings = await resp.json();
    if (settings?.hf_discovery_sort === sort) return;
    await fetch('/api/settings', {
      method: 'PUT',
      headers,
      body: JSON.stringify({ ...settings, hf_discovery_sort: sort }),
    });
  } catch {
    // non-fatal
  }
}

// ── Format toggle chip (deprecated: use hfCreateScopeSelector) ────────────────
// Kept for backward compatibility with existing callers.

export function hfCreateFormatToggle({ container, defaultFormat = 'gguf', onChange }) {
  if (!container) return null;

  const wrap = document.createElement('div');
  wrap.className = 'hf-format-toggle';
  wrap.style.cssText = 'display:flex;gap:0;border-radius:6px;overflow:hidden;border:1px solid rgba(255,255,255,0.08);';

  const makeBtn = (fmt, label) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `hf-format-btn${fmt === defaultFormat ? ' hf-format-btn--active' : ''}`;
    btn.dataset.format = fmt;
    btn.textContent = label;
    btn.style.cssText =
      'padding:2px 8px;font-size:10px;font-weight:600;border:none;cursor:pointer;' +
      'color:rgba(255,255,255,0.5);background:transparent;transition:all 0.15s ease;';
    btn.addEventListener('click', () => {
      wrap.querySelectorAll('.hf-format-btn').forEach(b => {
        b.classList.remove('hf-format-btn--active');
        b.style.cssText =
          b.style.cssText.replace(/color:[^;]+;|background:[^;]+;/g, '') +
          'color:rgba(255,255,255,0.5);background:transparent;';
      });
      btn.classList.add('hf-format-btn--active');
      btn.style.cssText =
        btn.style.cssText.replace(/color:[^;]+;|background:[^;]+;/g, '') +
        'color:#fff;background:rgba(99,102,241,0.85);';
      container.dataset.hfSearchFormat = fmt;
      if (onChange) onChange(fmt);
    });
    if (fmt === defaultFormat) {
      btn.style.cssText =
        btn.style.cssText.replace(/color:[^;]+;|background:[^;]+;/g, '') +
        'color:#fff;background:rgba(99,102,241,0.85);';
    }
    return btn;
  };

  wrap.appendChild(makeBtn('gguf', 'GGUF'));
  wrap.appendChild(makeBtn('mlx', 'MLX'));
  container.dataset.hfSearchFormat = defaultFormat;
  container.appendChild(wrap);

  return wrap;
}
