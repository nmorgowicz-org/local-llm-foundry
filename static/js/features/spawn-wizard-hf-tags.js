// HF tag mapping/normalisation and the hardware-step tag-pill UI (add/remove
// library tags, inline repo editor for re-resolving origin, and the tag picker
// popup). Depends on the hardware-model header for re-render triggers.
import { showToast } from './toast.js';
import { openCardPanel } from './spawn-wizard-model-card.js';
import {
  wizardState,
  _hfFilesPost,
  _extractQuantLabel,
} from './spawn-wizard.js';
import { renderHardwareModelHeader } from './spawn-wizard-hardware-model.js';

// ── HF tag → local library tag mapping ───────────────────────────────────────

// Maps raw HF tag strings to a normalised, user-facing category key.
// Only tags that have meaningful categorisation value are included —
// infrastructure tags (gguf, transformers, llama.cpp, region:us, etc.) are ignored.
// ── HF tag normalisation: curated map + block-list + pass-through ─────────────
//
// Architecture: three-tier system so we never need to pre-map every domain tag.
//
//  Tier 1 — Curated map: well-known HF tags → our 9 normalised category keys.
//            Covers the most common cross-model categories (vision, agentic…).
//
//  Tier 2 — Block-list: infrastructure / noise tags that carry no useful signal
//            for a local model library.  Every tag that survives this AND is not
//            in Tier 1 passes through as-is to Tier 3.
//
//  Tier 3 — Pass-through: domain tags not in either list (medical, legal, biomed,
//            translation, cybersecurity, …) are shown verbatim after normalisation.
//            No code change needed when new HF domains appear.

const HF_TAG_MAP = {
  // Vision / multimodal
  vision: 'vision', multimodal: 'vision', 'image-text-to-text': 'vision',
  image: 'vision', vqa: 'vision', visual: 'vision', 'text-to-image': 'vision',
  'image-to-text': 'vision', 'video-text-to-text': 'vision',
  // Agentic / tool use
  agent: 'agentic', agentic: 'agentic', 'tool-use': 'agentic',
  'function-calling': 'agentic', tool_use: 'agentic', tool_calling: 'agentic',
  // Coding
  coder: 'coding', code: 'coding', coding: 'coding',
  'code-generation': 'coding', devops: 'coding', programming: 'coding',
  'code-llm': 'coding',
  // Reasoning / thinking
  reasoning: 'reasoning', 'chain-of-thought': 'reasoning',
  thinking: 'reasoning', cot: 'reasoning', 'step-by-step': 'reasoning',
  // Roleplay / creative writing
  roleplay: 'roleplay', creative: 'roleplay', storytelling: 'roleplay',
  'creative-writing': 'roleplay', fiction: 'roleplay', 'role-playing': 'roleplay',
  // NSFW / adult content — 'not-for-all-audiences' is HF boilerplate for the same thing
  nsfw: 'nsfw', 'not-for-all-audiences': 'nsfw', adult: 'nsfw', explicit: 'nsfw',
  // Uncensored / guardrails removed — technique varies but concept is the same
  uncensored: 'uncensored', decensored: 'uncensored', abliterated: 'uncensored',
  heretic: 'uncensored', jailbreak: 'uncensored', 'no-refusals': 'uncensored',
  unfiltered: 'uncensored',
  // Math / STEM
  math: 'math', mathematics: 'math', science: 'math', stem: 'math',
  arithmetic: 'math',
  // General chat / instruction following
  conversational: 'chat', chat: 'chat', instruct: 'chat',
  'instruction-following': 'chat',
};

// Two-letter ISO language codes — blocked outright (not mapped to any category).
const ISO2_LANGS = new Set([
  'zh','ja','ko','ru','es','fr','de','ar','pt','it','nl','pl','sv','tr',
  'hi','vi','uk','cs','ro','hu','da','fi','no','id','th','he','el','bg',
  'en',
]);

// Tags that carry no useful signal for a local model library.
// Anything matching these patterns is silently dropped before pass-through.
const HF_TAG_BLOCKLIST = new Set([
  // File / library format
  'gguf','safetensors','pytorch','tflite','onnx','mlx','coreml','openvino',
  'ggml','llamafile',
  // Serving infrastructure
  'transformers','llama.cpp','text-generation-inference','vllm',
  'unsloth','ctransformers','ggerganov','endpoints_compatible',
  'text-generation','text2text-generation','fill-mask','token-classification',
  // Frontend clients (not model capabilities)
  'sillytavern','openwebui','open-webui','koboldcpp','ollama-library',
  // Training / alignment methodology (not a use-case)
  'lora','qlora','sft','rlhf','dpo','ppo','orpo','grpo','kto',
  'generated_from_trainer','adapter','merge','mergekit','finetuned',
  // Quantisation method tags (already captured by wizard hardware step)
  'imatrix','awq','gptq','eetq','exl2','nvfp4','fp8','int4','int8',
  // Model-card boilerplate
  'autotrain_compatible','has_space',
]);

// Regex patterns for tags that are always noise regardless of exact value.
const HF_TAG_BLOCK_PATTERNS = [
  /^base_model:/,      // base_model:owner/repo — parsed separately for inheritance
  /^dataset:/,         // dataset:owner/dataset
  /^license:/,         // license:apache-2.0 etc.
  /^region:/,          // region:us
  /^doi:/,
  /^arxiv:/,
  /^\d/,               // tags starting with a digit (version numbers etc.)
  /^[a-z]{2,3}_[A-Z]{2}$/,   // locale codes: zh_CN, pt_BR
  /^[a-z]{2}_[a-z]{2}$/,     // locale codes: zh_cn
  // Model family identifiers: llama3, llama-3, mistral7b, qwen3_6, gemma2, phi3…
  // Match known family names immediately followed by a digit or version separator.
  /^(llama|mistral|qwen|gemma|phi|falcon|gpt|bloom|mpt|opt|yi|deepseek|starcoder|codellama|vicuna|alpaca|wizardlm|orca|openchat|solar|nous|hermes|dolphin|beluga|airoboros|guanaco|koala|zephyr|stablelm|openhermes|chatml|neural|magnum|euryale|midnight|psyfighter|noromaid|lumimaid)[-_]?\d/i,
];

const HF_CATEGORY_LABEL = {
  vision: 'Vision',
  agentic: 'Agentic',
  coding: 'Coding',
  reasoning: 'Reasoning',
  roleplay: 'Roleplay',
  nsfw: 'NSFW',
  uncensored: 'Uncensored',
  math: 'Math/STEM',
  chat: 'Chat',
};

// Standard vocabulary always offered in the picker, independent of HF.
const ALL_KNOWN_TAGS = [
  'coding', 'roleplay', 'nsfw', 'uncensored', 'general', 'art', 'fast', 'default',
  'vision', 'agentic', 'reasoning', 'math', 'chat',
];

// Normalise a raw HF tag string for use as a local library tag:
// lowercase, spaces/& stripped to hyphens, trailing hyphens removed.
function _normaliseTag(raw) {
  return raw
    .toLowerCase()
    .replace(/[&/\\]/g, '-')
    .replace(/\s+/g, '-')
    .replace(/-{2,}/g, '-')
    .replace(/^-|-$/g, '');
}

// Returns true if the tag should be silently dropped.
function _isBlockedHfTag(raw) {
  const lower = raw.toLowerCase();
  if (HF_TAG_BLOCKLIST.has(lower)) return true;
  if (HF_TAG_BLOCK_PATTERNS.some(re => re.test(raw))) return true;
  return false;
}

// Analyse raw HF tags and return:
//   { categories: Set<string>, passthrough: string[] }
// categories — matched curated keys (vision, agentic …)
// passthrough — normalised tags that are not blocked and not mapped to a category;
//               these are shown verbatim in the "From this model" picker section.
function _hfTagsToCategories(rawTags) {
  const categories = new Set();
  const passthroughSet = new Set();

  for (const raw of rawTags) {
    const lower = raw.toLowerCase();
    if (_isBlockedHfTag(raw)) continue;
    if (ISO2_LANGS.has(lower)) continue;

    const cat = HF_TAG_MAP[lower];
    if (cat) {
      categories.add(cat);
    } else {
      // Pass-through: normalise and keep if it's a non-trivial string
      const norm = _normaliseTag(raw);
      if (norm.length >= 2 && !ALL_KNOWN_TAGS.includes(norm)) {
        passthroughSet.add(norm);
      }
    }
  }

  // Remove passthrough entries that are already covered by a category label
  for (const cat of categories) {
    const label = _normaliseTag(HF_CATEGORY_LABEL[cat] || cat);
    passthroughSet.delete(label);
  }

  return { categories, passthrough: [...passthroughSet] };
}

// Fetch tags for a HF repo and, if it has a base_model: tag pointing to a
// non-quantized source, merge that source's tags too (one level only).
// Quantizers like bartowski often strip use-case tags, but the base model keeps them.
// Returns { categories: Set, passthrough: string[] }.
async function _fetchHfTagsWithBaseModel(repoId) {
  const headers = window.authHeaders ? window.authHeaders() : {};
  let rawTags = [];
  try {
    const r = await fetch(`/api/hf/meta?repo=${encodeURIComponent(repoId)}`, { headers });
    if (r.ok) {
      const d = await r.json().catch(() => ({}));
      if (d.ok && d.tags) rawTags = d.tags;
    }
  } catch { /* non-fatal */ }

  // Look for base_model:owner/repo (not base_model:quantized: or base_model:adapter:)
  const baseTag = rawTags.find(t => {
    if (!t.startsWith('base_model:')) return false;
    const rest = t.slice('base_model:'.length);
    return !rest.startsWith('quantized:') && !rest.startsWith('adapter:') && rest.includes('/');
  });
  if (baseTag) {
    const baseRepo = baseTag.slice('base_model:'.length);
    try {
      const r = await fetch(`/api/hf/meta?repo=${encodeURIComponent(baseRepo)}`, { headers });
      if (r.ok) {
        const d = await r.json().catch(() => ({}));
        if (d.ok && d.tags) rawTags = [...rawTags, ...d.tags];
      }
    } catch { /* non-fatal */ }
  }

  return _hfTagsToCategories(rawTags);
}

// Fetch HF model metadata and render the tags row for the hardware step.
// No-ops if originRepo is unknown or the row element is missing.
let _tagsRowOrigin = ''; // track which repo is currently loaded in the row
// Cross-module mutator: spawn-wizard-hf-origin.js needs to force a re-render of the
// tags row after confirming a new origin, without owning this binding. Will move to
// spawn-wizard-hf-tags.js in step 10.
export function resetTagsRowOrigin() { _tagsRowOrigin = ''; }

// Inline repo editor / picker: always shows candidates + custom input.
  //  - Searches by filename, builds a short candidate list.
  //  - If currentRepo is known, it is first and marked recommended.
   //  - User can pick from a dropdown or type a custom repo ID.
   export function _openHwRepoEditor(repoEl, currentRepo) {
     if (!repoEl) return;
     if (repoEl.classList.contains('hw-model-repo-editing')) return;

     repoEl.classList.add('hw-model-repo-editing');
     repoEl.innerHTML = '';

     const statusEl = document.createElement('span');
     statusEl.style.cssText =
       'font-size:10px;color:var(--color-text-muted);margin-left:6px;white-space:nowrap;';
     statusEl.textContent = 'Searching HuggingFace…';
     repoEl.appendChild(statusEl);

     const restore = () => {
       repoEl.classList.remove('hw-model-repo-editing');
       renderHardwareModelHeader();
     };

     const filename = (wizardState.model.path || '').split(/[\\/]/).pop() || '';
     const modelBytes = wizardState.model.modelBytes || 0;

     // Use resolve-origin to get ranked, verified candidates.
     const fetchCandidates = async () => {
       const headers = window.authHeaders ? { ...window.authHeaders(), 'Content-Type': 'application/json' } : { 'Content-Type': 'application/json' };
       try {
         const res = await fetch('/api/hf/resolve-origin', {
           method: 'POST',
           headers,
           body: JSON.stringify({ filename, size_bytes: modelBytes }),
         });
         if (!res.ok) return { confident: false, candidates: [] };
         return await res.json();
       } catch {
         return { confident: false, candidates: [] };
       }
     };

     // Apply a selected repo: fetch files, validate, and set wizardState.
     const applyRepo = async (repoId) => {
       if (!repoId) return;
       try {
         const data = await _hfFilesPost(repoId);

         if (!data?.ok) {
           showToast('Repo not found', 'error', 'Check the repo ID and try again.');
           restore();
           return;
         }

         const rawFiles = (data.files || []).filter(f =>
           !f.is_mmproj && (f.rfilename || f.path || '').toLowerCase().endsWith('.gguf'));
         if (!rawFiles.length) {
           showToast('No GGUFs', 'error', 'No GGUF files found in this repo.');
           restore();
           return;
         }

         wizardState.model.originRepo = repoId;
         wizardState.model.hfRepo = repoId;
         wizardState.model.quantFiles = rawFiles.map(f => ({
           path: f.rfilename || f.path || '',
           name: f.rfilename || f.path || '',
           size: f.size || 0,
           label: _extractQuantLabel(f.rfilename || f.path || ''),
         }));
         wizardState.model._quantSwapRepo = repoId;

         wizardState.model.mmprojFiles = (data.files || []).filter(f => f.is_mmproj).map(f => ({
           repo_id: f.repo_id || repoId,
           path: f.rfilename || f.path || '',
           name: f.rfilename || f.path || '',
           size: f.size || 0,
           is_mmproj: true,
           is_recommended_mmproj: f.is_recommended_mmproj || false,
           mmproj_recommendation: f.mmproj_recommendation || '',
         }));

          _tagsRowOrigin = '';
         showToast('Repo updated', 'success', `${rawFiles.length} quants loaded`);
         renderHardwareModelHeader();
       } catch {
         showToast('Error', 'error', 'Failed to load repo.');
         restore();
       }
     };

     // Render picker as dropdown with recommended + others + custom input.
     const renderPicker = (resolveData) => {
       repoEl.innerHTML = '';

       const wrap = document.createElement('span');
       wrap.style.cssText =
         'display:inline-flex;align-items:center;gap:4px;margin-left:4px;flex-wrap:wrap;';

       const candidates = (resolveData.candidates || []).slice(0, 5);
       const confident = !!resolveData.confident;
       const recommendedRepo = confident && candidates.length > 0 ? candidates[0].repoId : (currentRepo || (candidates.length > 0 ? candidates[0].repoId : ''));

       if (candidates.length > 0) {
         // Dropdown with full author/repo.
         const select = document.createElement('select');
         select.style.cssText =
           'font-size:9px;padding:1px 4px;border-radius:3px;border:1px solid rgba(148,163,253,0.4);' +
           'background:rgba(15,23,42,0.98);color:var(--color-text-primary);min-height:18px;';

         const defaultOption = document.createElement('option');
         defaultOption.value = '';
         defaultOption.textContent = 'Select origin…';
         select.appendChild(defaultOption);

         candidates.forEach((c, i) => {
           const repoId = c.repoId || '';
           const opt = document.createElement('option');
           opt.value = repoId;
           const isRecommended = !!(confident && i === 0);
           opt.textContent = (isRecommended ? '★ Recommended: ' : '') + repoId;
           if (repoId === recommendedRepo) {
             opt.selected = true;
           }
           select.appendChild(opt);
         });

         select.addEventListener('change', () => {
           const repoId = select.value || recommendedRepo;
           if (repoId && repoId !== wizardState.model.originRepo) {
             applyRepo(repoId);
           }
         });

         wrap.appendChild(select);
       }

       // Separator
       const sep = document.createElement('span');
       sep.style.cssText = 'font-size:8px;color:var(--color-text-muted);margin:0 1px;';
       sep.textContent = '|';
       wrap.appendChild(sep);

       // Custom input for typing any HF repo ID.
       const input = document.createElement('input');
       input.type = 'text';
       input.value = currentRepo || '';
       input.placeholder = 'owner/repo-GGUF';
       input.style.cssText =
         'width:160px;padding:2px 5px;border-radius:3px;border:1px solid rgba(148,163,253,0.4);' +
         'background:rgba(15,23,42,0.95);color:var(--color-text-primary);font-size:9px;';

       const loadBtn = document.createElement('button');
       loadBtn.type = 'button';
       loadBtn.className = 'btn-wizard-tertiary';
       loadBtn.style.cssText =
         'font-size:9px;min-height:18px;padding:1px 5px;flex-shrink:0;';
       loadBtn.textContent = 'Load';

       const cancelBtn = document.createElement('button');
       cancelBtn.type = 'button';
       cancelBtn.className = 'btn-wizard-tertiary';
       cancelBtn.style.cssText =
         'font-size:9px;min-height:18px;padding:1px 5px;flex-shrink:0;opacity:0.7;';
       cancelBtn.textContent = '✕';

       wrap.appendChild(input);
       wrap.appendChild(loadBtn);
       wrap.appendChild(cancelBtn);
       repoEl.appendChild(wrap);

       input.focus();
       input.select();

       const doLoad = async () => {
         const repoId = input.value.trim();
         if (!repoId) return;
         loadBtn.disabled = true;
         loadBtn.textContent = '⠋';

         try {
           await applyRepo(repoId);
         } catch {
           loadBtn.disabled = false;
           loadBtn.textContent = 'Load';
           showToast('Error', 'error', 'Failed to load repo.');
           restore();
         }
       };

       loadBtn.addEventListener('click', doLoad);
       cancelBtn.addEventListener('click', restore);
       input.addEventListener('keydown', e => {
         if (e.key === 'Enter') { e.preventDefault(); doLoad(); }
         if (e.key === 'Escape') { restore(); }
       });
     };

     (async () => {
       const resolveData = await fetchCandidates();
       renderPicker(resolveData);
     })();
   }

export async function _refreshHwTagsRow() {
  const row = document.getElementById('hw-tags-row');
  if (!row) return;
  const { originRepo, path, family } = wizardState.model;
  if (!originRepo) { row.style.display = 'none'; return; }
  row.style.display = '';
  if (_tagsRowOrigin === originRepo) return; // already populated for this repo
  _tagsRowOrigin = originRepo;

  // Load current library tags for this model path — exclude system tags.
  let currentTags = [];
  try {
    const headers = window.authHeaders ? window.authHeaders() : {};
    const r = await fetch('/api/models/tags', { headers });
    if (r.ok) {
      const d = await r.json().catch(() => ({}));
      currentTags = (d.tags?.[path] || []).filter(
        t => !t.startsWith('hf_origin:') && !t.startsWith('family:')
      );
    }
  } catch { /* non-fatal */ }

  // Fetch HF tags (including base model tags if the GGUF repo stripped them).
  let suggestedCats = new Set();
  let passthroughTags = [];
  try {
    ({ categories: suggestedCats, passthrough: passthroughTags } =
      await _fetchHfTagsWithBaseModel(originRepo));
  } catch { /* non-fatal */ }

  _renderHwTagPills(currentTags, suggestedCats, passthroughTags, path, originRepo);

  // Append family + card link pills
  const pillsWrap = document.getElementById('hw-tags-pills');
  if (pillsWrap) {
    if (family) {
      const famPill = document.createElement('span');
      famPill.className = 'mm-tag-pill';
      famPill.style.cssText = 'font-size:9px;padding:2px 7px;opacity:0.6;white-space:nowrap;';
      famPill.textContent = `family: ${family}`;
      famPill.title = `Detected model family (auto)`;
      pillsWrap.appendChild(famPill);
    }
    // Card pill: opens inline model card panel (uses existing wizard-card-panel)
    if (originRepo) {
      const cardPill = document.createElement('button');
      cardPill.type = 'button';
      cardPill.className = 'mm-tag-pill';
      cardPill.style.cssText = 'font-size:9px;padding:2px 7px;cursor:pointer;text-decoration:none;opacity:0.7;white-space:nowrap;border:none;background:none;font:inherit;';
      cardPill.textContent = '📄 Card';
      cardPill.title = 'View model card inline';
      cardPill.addEventListener('click', () => openCardPanel(originRepo));
      pillsWrap.appendChild(cardPill);
    }
  }
}

function _renderHwTagPills(currentTags, suggestedCats, passthroughTags, modelPath, originRepo) {
  const pillsWrap = document.getElementById('hw-tags-pills');
  if (!pillsWrap) return;
  pillsWrap.innerHTML = '';

  if (!currentTags.length) {
    // "No tags yet" hint
    const hint = document.createElement('span');
    hint.style.cssText = 'font-size:10px;color:var(--color-text-muted);margin-right:4px;';
    hint.textContent = 'No tags yet';
    pillsWrap.appendChild(hint);
  } else {
    // Render existing tag pills
    currentTags.forEach(tag => {
      const pill = document.createElement('span');
      pill.className = 'mm-tag-pill mm-tag-pill--active';
      pill.style.cssText = 'font-size:9px;padding:2px 7px;cursor:pointer;';
      pill.title = `Remove tag "${tag}"`;
      pill.textContent = tag + ' ×';
      pill.addEventListener('click', async () => {
        const newTags = currentTags.filter(t => t !== tag);
        await _saveHwModelTags(modelPath, newTags);
        _tagsRowOrigin = '';
        await _refreshHwTagsRow();
      });
      pillsWrap.appendChild(pill);
    });
  }

  // "Sync with HF" pill: shown whenever HF origin is known and HF offers tags.
  // This lets the user pull new tags OR refresh/remove tags the author changed.
  if (originRepo && (suggestedCats.size > 0 || (passthroughTags && passthroughTags.length > 0))) {
    const syncBtn = document.createElement('button');
    syncBtn.type = 'button';
    syncBtn.className = 'mm-tag-pill';
    syncBtn.style.cssText =
      'font-size:9px;padding:2px 7px;cursor:pointer;text-decoration:none;opacity:0.75;white-space:nowrap;border:none;background:none;font:inherit;display:inline-flex;align-items:center;gap:4px;';
    syncBtn.textContent = '⎇ Sync with HF';
    syncBtn.title = 'Sync library tags with this model\'s HuggingFace card';

    syncBtn.addEventListener('click', async () => {
      try {
        const hfResult = await _fetchHfTagsWithBaseModel(originRepo);
        const cats = Array.from(hfResult.categories || []);
        const pass = Array.from(hfResult.passthrough || []);
        const hfSet = new Set([...cats, ...pass]);

        // Re-read current tags to ensure we’re in sync.
        const headers = window.authHeaders ? window.authHeaders() : {};
        const r = await fetch('/api/models/tags', { headers });
        const td = r.ok ? await r.json().catch(() => ({})) : {};
        const allCurrent = (td.tags?.[modelPath] || []).filter(
          t => !t.startsWith('hf_origin:')
        );

        // Sync logic:
        // - Keep all core / user-friendly tags (ALL_KNOWN_TAGS).
        // - Keep any HF tags still present on the card.
        // - Drop any tag not in HF and not in ALL_KNOWN_TAGS
        //   so that removed / outdated tags are cleaned up.
        const newTags = allCurrent.filter(tag => {
          const inHf = hfSet.has(tag);
          const inCore = ALL_KNOWN_TAGS.includes(tag);
          return inHf || inCore;
        });

        // Add any HF tags not yet present.
        for (const t of hfSet) {
          if (!newTags.includes(t)) newTags.push(t);
        }

        if (newTags.length === 0) return;

        await _saveHwModelTags(modelPath, newTags);
        _tagsRowOrigin = '';
        await _refreshHwTagsRow();
      } catch {
        // Non-fatal: user can still use the manual tag picker.
      }
    });

    pillsWrap.appendChild(syncBtn);
  }
}

async function _saveHwModelTags(modelPath, tags) {
  try {
    const headers = window.authHeaders
      ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
      : { 'Content-Type': 'application/json' };
    const r = await fetch('/api/models/tags', {
      headers,
      method: 'GET',
    });
    const existing = r.ok ? ((await r.json().catch(() => ({}))).tags?.[modelPath] || []) : [];
    const originTags = existing.filter(t => t.startsWith('hf_origin:'));
    const merged = [...originTags, ...tags.filter(t => !t.startsWith('hf_origin:'))];
    await fetch('/api/models/tags', {
      method: 'PUT',
      headers,
      body: JSON.stringify({ model_path: modelPath, tags: merged }),
    });
  } catch { /* non-fatal */ }
}

export function _openHwTagPicker(btn, modelPath, originRepo) {
  // Remove any existing picker.
  document.getElementById('hw-tag-picker-popup')?.remove();

  const popup = document.createElement('div');
  popup.id = 'hw-tag-picker-popup';
  popup.className = 'hw-tag-picker-popup';

  // Fetch current state fresh so the picker reflects reality.
  const headers = window.authHeaders ? window.authHeaders() : {};
  Promise.all([
    fetch('/api/models/tags', { headers }).then(r => r.ok ? r.json().catch(() => ({})) : {}),
    originRepo ? _fetchHfTagsWithBaseModel(originRepo).catch(() => ({ categories: new Set(), passthrough: [] })) : Promise.resolve({ categories: new Set(), passthrough: [] }),
  ]).then(([tagsData, hfResult]) => {
    const rawCurrent = (tagsData.tags?.[modelPath] || [])
      .filter(t => !t.startsWith('hf_origin:'));
    const { categories: suggestedCats, passthrough: passthroughTags } = hfResult;

    popup.innerHTML = '';

    // ── Section: Curated HF suggestions ──────────────────────────────────────
    if (suggestedCats.size > 0) {
      const hdr = document.createElement('div');
      hdr.className = 'hw-tag-picker-section';
      hdr.textContent = `Suggested from ${originRepo.split('/')[0]}`;
      popup.appendChild(hdr);

      const sugRow = document.createElement('div');
      sugRow.className = 'hw-tag-picker-pills';
      [...suggestedCats].forEach(cat => {
        _appendTagPill(sugRow, HF_CATEGORY_LABEL[cat] || cat, cat, rawCurrent, modelPath, originRepo, popup);
      });
      popup.appendChild(sugRow);
    }

    // ── Section: Pass-through domain tags (medical, legal, biomed, …) ────────
    // These are HF tags that didn't map to a curated category but aren't noise.
    // Shown here so they can be applied without needing the custom input.
    if (passthroughTags.length > 0) {
      const hdr = document.createElement('div');
      hdr.className = 'hw-tag-picker-section';
      hdr.textContent = 'From this model';
      popup.appendChild(hdr);

      const ptRow = document.createElement('div');
      ptRow.className = 'hw-tag-picker-pills';
      passthroughTags.forEach(tag => {
        _appendTagPill(ptRow, tag, tag, rawCurrent, modelPath, originRepo, popup);
      });
      popup.appendChild(ptRow);
    }

    // ── Section: Standard vocabulary ─────────────────────────────────────────
    const hdr2 = document.createElement('div');
    hdr2.className = 'hw-tag-picker-section';
    hdr2.textContent = 'All tags';
    popup.appendChild(hdr2);

    const allRow = document.createElement('div');
    allRow.className = 'hw-tag-picker-pills';
    // Exclude tags already shown in the HF or pass-through sections to avoid duplication.
    const shownKeys = new Set([...suggestedCats, ...passthroughTags]);
    ALL_KNOWN_TAGS.filter(t => !shownKeys.has(t)).forEach(tag => {
      _appendTagPill(allRow, tag, tag, rawCurrent, modelPath, originRepo, popup);
    });
    popup.appendChild(allRow);

    // ── Custom tag input ──────────────────────────────────────────────────────
    const customRow = document.createElement('div');
    customRow.style.cssText = 'display:flex;gap:5px;margin-top:7px;';
    const customInput = document.createElement('input');
    customInput.type = 'text';
    customInput.placeholder = 'Custom tag…';
    customInput.style.cssText = 'flex:1;padding:4px 7px;border-radius:5px;border:1px solid rgba(255,255,255,0.1);background:rgba(28,34,42,0.9);color:var(--color-text-primary);font-size:10px;';
    const addBtn = document.createElement('button');
    addBtn.type = 'button';
    addBtn.className = 'btn-wizard-secondary';
    addBtn.style.cssText = 'font-size:10px;min-height:24px;padding:2px 8px;flex-shrink:0;';
    addBtn.textContent = 'Add';

    const doAdd = async () => {
      const val = customInput.value.trim().toLowerCase().replace(/\s+/g, '-');
      if (!val || rawCurrent.includes(val)) return;
      rawCurrent.push(val);
      await _saveHwModelTags(modelPath, rawCurrent);
      _tagsRowOrigin = ''; // force re-render of pills row
      await _refreshHwTagsRow();
      popup.remove();
      showToast(`Tagged "${val}"`, 'success');
    };
    addBtn.addEventListener('click', doAdd);
    customInput.addEventListener('keydown', e => { if (e.key === 'Enter') doAdd(); });
    customRow.appendChild(customInput);
    customRow.appendChild(addBtn);
    popup.appendChild(customRow);
  });

  // Position below the + button, or above if there's more room there.
  document.body.appendChild(popup);
  const rect = btn.getBoundingClientRect();
  const popupH = popup.offsetHeight;
  const spaceBelow = window.innerHeight - rect.bottom - 8;
  const spaceAbove = rect.top - 8;
  if (spaceBelow >= popupH || spaceBelow >= spaceAbove) {
    popup.style.top = `${rect.bottom + 4}px`;
    popup.style.maxHeight = `${Math.max(120, spaceBelow)}px`;
  } else {
    popup.style.maxHeight = `${Math.max(120, spaceAbove)}px`;
    popup.style.top = `${Math.max(8, rect.top - Math.min(popupH, spaceAbove) - 4)}px`;
  }
  popup.style.left = `${rect.left}px`;

  // Close on outside click.
  setTimeout(() => {
    const close = e => {
      if (!popup.contains(e.target) && e.target !== btn) {
        popup.remove();
        document.removeEventListener('mousedown', close);
      }
    };
    document.addEventListener('mousedown', close);
  }, 0);
}

function _appendTagPill(container, label, tagKey, currentTags, modelPath, _originRepo, _popup) {
  const has = currentTags.includes(tagKey);
  const pill = document.createElement('span');
  pill.className = 'mm-tag-pill' + (has ? ' mm-tag-pill--active' : '');
  pill.style.cssText = 'font-size:10px;padding:3px 9px;cursor:pointer;user-select:none;';
  pill.textContent = label;
  pill.title = has ? `Remove tag "${tagKey}"` : `Add tag "${tagKey}"`;
  pill.addEventListener('click', async () => {
    const newTags = has
      ? currentTags.filter(t => t !== tagKey)
      : [...currentTags, tagKey];
    // Mutate in place so other pills in the same picker stay in sync.
    currentTags.length = 0;
    newTags.forEach(t => currentTags.push(t));
    await _saveHwModelTags(modelPath, newTags);
    pill.className = 'mm-tag-pill' + (newTags.includes(tagKey) ? ' mm-tag-pill--active' : '');
    pill.title = newTags.includes(tagKey) ? `Remove tag "${tagKey}"` : `Add tag "${tagKey}"`;
    // Refresh the pills row in the header.
    _tagsRowOrigin = '';
    _refreshHwTagsRow();
  });
  container.appendChild(pill);
}
