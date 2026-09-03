// llama.cpp information architecture for the Spawn Wizard hardware step
// (plan §5 Phase 4 item 4). Retires the hand-written #spawn-advanced-fields /
// #spawn-spec-details <details> blocks: the flat field dump inside
// #spawn-advanced-fields is hidden and relocated into registry-generated,
// tier-driven groups by the shared engine in spawn-wizard-ia.js — the same
// mechanism spawn-wizard-mlx-ia.js uses for Rapid-MLX. Existing DOM ids
// remain the serialization contract; #spawn-advanced-fields itself is now a
// plain wrapper div (JS anchor), not a <details>.

import { createWizardIA } from './spawn-wizard-ia.js';

const SUPERSECTIONS = [
  {
    id: 'advanced-tuning',
    title: 'Advanced tuning',
    description: 'Safe defaults for most setups. Change only if you have a specific reason.',
  },
];

const GROUPS = [
  {
    supersection: 'advanced-tuning', id: 'batching-threads', title: 'Batching & threads',
    description: 'Prompt/micro-batch sizing, flash attention, and CPU thread allocation.',
    critical: true, view: 'both',
    controls: [
      'spawn-batch-size', 'spawn-parallel-slots', 'spawn-ubatch-size', 'spawn-flash-attn',
      'spawn-prio', 'spawn-threads', 'spawn-threads-batch',
      'spawn-no-cont-batching', 'spawn-swa-full', 'spawn-load-mode',
    ],
  },
  {
    supersection: 'advanced-tuning', id: 'diagnostics', title: 'Diagnostics',
    description: 'Server log detail used for troubleshooting and speculative-decoding telemetry.',
    critical: false, view: 'both',
    controls: ['spawn-verbosity'],
  },
  {
    supersection: 'advanced-tuning', id: 'moe-multigpu', title: 'MoE & multi-GPU',
    description: 'Mixture-of-Experts CPU offload and tensor-split placement across GPUs.',
    critical: false, view: 'both',
    controls: ['spawn-n-cpu-moe', 'spawn-tensor-split'],
  },
  {
    supersection: 'advanced-tuning', id: 'prompt-cache', title: 'Prompt cache',
    description: 'Persistent KV prefix-cache mode and size bound.',
    critical: false, view: 'both',
    controls: [
      'spawn-cache-mode', 'spawn-cache-ram', 'spawn-ctx-checkpoints',
      'spawn-checkpoint-min-step', 'spawn-cache-reuse', 'spawn-cache-idle-slots',
    ],
  },
  {
    supersection: 'advanced-tuning', id: 'model-compatibility', title: 'Model & compatibility',
    description: 'Native llama.cpp projector placement and model compatibility controls.',
    critical: false, view: 'both',
    controls: ['spawn-mmproj-offload'],
  },
  {
    supersection: 'advanced-tuning', id: 'reasoning-controls', title: 'Generation & reasoning',
    description: 'Native llama.cpp reasoning behavior, separate from chat-template thinking kwargs.',
    critical: false, view: 'both',
    controls: ['spawn-reasoning-effort', 'spawn-reasoning-format', 'spawn-reasoning-preserve'],
  },
  {
    supersection: 'advanced-tuning', id: 'fit-memory', title: 'Auto-fit & memory',
    description: 'Shrink context to fit a memory budget; pin the model in RAM.',
    critical: false, view: 'both',
    controls: ['spawn-fit-enable', 'spawn-fit-target', 'spawn-mlock'],
  },
  {
    // Relocates the existing #spawn-spec-details <details> as-is (plan §2.8:
    // Advanced, nested — the llama.cpp peer of MLX's 'companions' group) so
    // its internal conditional wiring (draft-KV rows, draft-model path) isn't
    // re-derived.
    supersection: 'advanced-tuning', id: 'speculative-decoding', title: 'Speculative decoding',
    prebuiltId: 'spawn-spec-details',
    critical: false, view: 'both',
  },
];

const ia = createWizardIA({
  groupClassName: 'mlx-native-group',
  rowClassName: 'llama-wiz-row',
  originAnchorComment: 'llama-wiz-origin',
});

export function configureLlamaWizardIA(root, enabled, profile = 'balanced') {
  ia.configure(root, enabled, profile, GROUPS, SUPERSECTIONS, '#spawn-advanced-fields');
  window._llamaGroups = GROUPS;
  window._llamaSupersections = SUPERSECTIONS;
}

export function applyLlamaTierVisibility(root, profile) {
  ia.applyTierVisibility(root, profile);
}
