// MLX-family information architecture for the Spawn Wizard hardware step.
// Presentation-only: existing DOM ids remain the serialization contract. This
// module only regroups the flat Rapid-MLX field dump into progressive,
// labelled sections at runtime — mirroring preset-editor-mlx.js's pattern —
// so the wizard and the preset editor share the same MLX section vocabulary
// (Generation, Cache & Performance, Server & Safety).
//
// The relocation/tier-disclosure mechanics live in the loader-agnostic
// spawn-wizard-ia.js (plan §5 Phase 4 item 2); this module only supplies the
// MLX-specific group data and DOM anchor.

import { createWizardIA, isOpenForProfile } from './spawn-wizard-ia.js';

const SUPERSECTIONS = [
  {
    id: 'generation',
    title: 'Generation & Runtime',
    description: 'Thinking, sampling, and model protocol defaults.',
  },
  {
    id: 'cache-performance',
    title: 'Cache & Performance',
    description: 'Active KV precision, retained prompts, and scheduling.',
  },
  {
    id: 'server-safety',
    title: 'Server & Safety',
    description: 'Tool integration and qualified companions.',
  },
];

const GROUPS = [
  {
    supersection: 'generation', id: 'thinking', title: 'Thinking & reasoning',
    description: 'Control model reasoning behavior independently from sampling.',
    critical: true, view: 'card',
    controls: ['spawn-rapid-reasoning-mode'],
  },
  {
    supersection: 'generation', id: 'protocol', title: 'Model protocol',
    description: 'Keep automatic detection unless a modified finetune requires an override.',
    critical: true, view: 'both',
    controls: ['spawn-rapid-tool-call-parser', 'spawn-rapid-reasoning-parser', 'spawn-rapid-hybrid-mode'],
  },
  {
    supersection: 'generation', id: 'sampling', title: 'Sampling defaults',
    description: 'Pre-launch server defaults for this preset (temperature, top-p, …). An API client can override them per request; chats started in this app use these values.',
    critical: true, view: 'both',
    controls: ['spawn-sampling-mode'],
  },
  {
    supersection: 'cache-performance', id: 'active-memory', title: 'Active memory',
    description: 'Precision and prefill choices that affect live unified-memory pressure.',
    critical: false, view: 'both',
    controls: ['spawn-kv-cache-dtype', 'spawn-rapid-prefill-step-size', 'spawn-turboquant-mode'],
  },
  {
    supersection: 'cache-performance', id: 'retained-cache', title: 'Retained prompt cache',
    description: 'Bound reusable prompt snapshots by both memory and working-set size.',
    // Deliberate divergence from llama.cpp's -cram (Advanced): retained cache is
    // a §2.6 scenario axis the user picks alongside context, so it must be
    // reachable at Balanced.
    critical: true, view: 'both',
    controls: ['spawn-retained-cache-mib', 'spawn-rapid-hybrid-cache-entries'],
  },
  {
    supersection: 'cache-performance', id: 'scheduler', title: 'Scheduler & throughput',
    description: 'Advanced batching and admission limits; defaults suit one interactive user.',
    // spawn-rapid-max-num-seqs is also a §2.6 scenario axis (peer of
    // spawn-parallel-slots) — same Balanced reasoning as retained-cache above.
    critical: true, view: 'both',
    controls: [
      'spawn-rapid-gpu-memory-utilization', 'spawn-rapid-max-num-seqs',
      'spawn-rapid-max-concurrent-requests', 'spawn-rapid-pflash-policy',
      'spawn-rapid-prefill-batch-size', 'spawn-rapid-completion-batch-size',
    ],
  },
  {
    supersection: 'server-safety', id: 'tool-integration', title: 'Tool integration',
    description: 'Enable only for models with a compatible tool-call parser.',
    critical: true, view: 'both',
    controls: ['spawn-rapid-auto-tool-choice'],
  },
  {
    supersection: 'server-safety', id: 'companions', title: 'Companions & experimental acceleration',
    description: 'Only qualified local companions belong here; unsupported remote launches fail closed.',
    collapsible: true,
    critical: false, view: 'both',
    controls: [
      'spawn-rapid-speculative-enabled', 'spawn-rapid-speculative-mode-wrap',
      'spawn-rapid-speculative-sidecars-wrap', 'spawn-rapid-speculative-model-wrap',
      'spawn-rapid-speculative-pin-status-wrap', 'spawn-rapid-speculative-trust-wrap',
      'spawn-rapid-speculative-tokens-wrap', 'spawn-rapid-speculative-auto-k-wrap',
    ],
  },
];

const ia = createWizardIA({
  groupClassName: 'mlx-native-group',
  rowClassName: 'mlx-wiz-row',
  originAnchorComment: 'mlx-wiz-origin',
});

export function configureMlxWizardIA(root, enabled, profile = 'balanced') {
  ia.configure(root, enabled, profile, GROUPS, SUPERSECTIONS, '#spawn-rapid-advanced-fields');
  window._mlxGroups = GROUPS;
  window._mlxSupersections = SUPERSECTIONS;
}

// Re-applies each built group's tier-driven open/closed state without
// rebuilding the DOM — called on profile change (spawn-wizard.js's
// applyProfileVisibility) so switching Quick/Balanced/Advanced updates
// disclosure the same way #spawn-advanced-fields does for llama.cpp.
export function applyMlxTierVisibility(root, profile) {
  ia.applyTierVisibility(root, profile);
}

export { isOpenForProfile };
