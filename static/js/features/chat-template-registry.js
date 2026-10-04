// Community chat templates, by model family. The first entry in a family is its default.
//
// None of these pin a version or a commit. Upstream authors ship new template versions often,
// so each entry tracks the repo's moving `main`. The update checker compares what is installed
// against `main` and offers the update; nothing is ever installed silently. Every install
// records the exact commit it fetched, so Version history can roll back.
export const COMMUNITY_TEMPLATES = {
  qwen: [
    {
      name: 'qwen-froggeric-fixed',
      display: "froggeric's Fixed Template",
      installEndpoint: '/api/chat-template/install-hf',
      repo: 'froggeric/Qwen-Fixed-Chat-Templates',
      file: 'chat_template.jinja',
      description: 'Fixes Qwen tool calling, KV-cache invalidation and agentic-loop bugs',
      sourceUrl: 'https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/blob/main/chat_template.jinja',
      provenance: 'community',
    },
    {
      // A fork of froggeric's template that force-appends a system prompt tuned for knowledge
      // work and coding. It is rebased onto each froggeric release, so it is a drop-in swap.
      name: 'qwen-sharp',
      display: 'Sharp Template (peculiar-ragdoll)',
      installEndpoint: '/api/chat-template/install-hf',
      repo: 'peculiar-ragdoll/Qwen-Sharp-Chat-Templates',
      file: 'chat_template.jinja',
      description: "froggeric's template plus an appended system prompt for knowledge work and coding; answers in fewer tokens",
      sourceUrl: 'https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates/blob/main/chat_template.jinja',
      provenance: 'community',
    },
  ],
  // Google's official template is listed first (and is what getDefaultTemplateForFamily()
  // picks) because it's the priority recommendation. jscott3201's agentic fork is kept as
  // a fallback entry in case Google's template regresses tool-calling again in the future.
  gemma4: [
    {
      name: 'gemma4-google-official',
      display: "Google's Official Gemma 4 Template",
      installEndpoint: '/api/chat-template/install-hf',
      repo: 'google/gemma-4-31B-it',
      file: 'chat_template.jinja',
      description: 'Reference template shipped with Gemma 4 31B (from model repo)',
      sourceUrl: 'https://huggingface.co/google/gemma-4-31B-it/blob/main/chat_template.jinja',
      provenance: 'official',
    },
    {
      name: 'gemma4-jscott3201-agentic',
      display: "jscott3201's Gemma 4 Agentic Template",
      installEndpoint: '/api/chat-template/install-url',
      url: 'https://raw.githubusercontent.com/jscott3201/llm-tuning/main/gemma4/chat_templates/custom_pub_chat_template_gemma4.jinja',
      description: 'Improves thinking, tool calls, null arguments & multi-turn agentic workflows for Gemma 4',
      sourceUrl: 'https://github.com/jscott3201/llm-tuning/blob/main/gemma4/chat_templates/custom_pub_chat_template_gemma4.jinja',
      provenance: 'community',
    },
  ],
};

/** Returns array of templates for a family (normalizes single object → array). */
export function getTemplatesForFamily(family) {
  if (!family) return [];
  const entry = COMMUNITY_TEMPLATES[family];
  if (!entry) return [];
  return Array.isArray(entry) ? entry : [entry];
}

/** Returns the default template for a family (first candidate). */
export function getDefaultTemplateForFamily(family) {
  return getTemplatesForFamily(family)[0] || null;
}

/** Returns all family keys that have templates. */
export function getTemplateFamilies() {
  return Object.keys(COMMUNITY_TEMPLATES);
}

/** Looks a template up by its install name across every family. */
export function findTemplateByName(name) {
  if (!name) return null;
  for (const family of getTemplateFamilies()) {
    const hit = getTemplatesForFamily(family).find(tpl => tpl.name === name);
    if (hit) return hit;
  }
  return null;
}

/**
 * " (Official)" / " (Community)" suffix for a template's label, or "" when it adds nothing.
 * It only helps when a family mixes provenance (Gemma: Google's own beside a community fork).
 * A family of community templates is told apart by name, and labelling each one "Community"
 * is noise.
 */
export function provenanceSuffix(tpl, family) {
  if (!tpl?.provenance) return '';
  const candidates = getTemplatesForFamily(family);
  const mixed = new Set(candidates.map(c => c.provenance)).size > 1;
  if (!mixed) return '';
  return tpl.provenance === 'official' ? ' (Official)' : ' (Community)';
}

// Maps a backend-derived architecture family slug (e.g. preset.family /
// wizardState.model.family — sourced from GGUF `general.architecture` or an
// HF `base_model` tag, never a filename) to a community template group.
export function communityTemplateFamilyFor(family) {
  const f = (family || '').toLowerCase();
  if (!f) return null;
  if (f.startsWith('qwen') || f === 'qwopus') return 'qwen';
  if (f === 'gemma4') return 'gemma4';
  return null;
}

// Maps a raw GGUF `general.architecture` value (e.g. "qwen3_6", "qwen35moe")
// directly to a community template group. Used when no normalized family
// slug is available yet but live GGUF metadata was just read.
export function communityFamilyFromGgufArchitecture(arch) {
  const a = (arch || '').toLowerCase();
  if (a.includes('qwen')) return 'qwen';
  if (a.includes('gemma4') || a.includes('gemma_4')) return 'gemma4';
  return null;
}

export function buildCommunityTemplateInstallRequest(template, force = false) {
  const body = template.url
    ? { url: template.url, name: template.name }
    : {
      repo: template.repo,
      file: template.file,
      name: template.name,
      ...(template.revision ? { revision: template.revision } : {}),
    };
  if (force || template.revision) body.force = true;
  return { endpoint: template.installEndpoint, body };
}
