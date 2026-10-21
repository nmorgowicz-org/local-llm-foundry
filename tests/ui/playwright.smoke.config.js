// playwright.smoke.config.js
//
// The "smoke" tier: a few minutes of broad, fast coverage instead of the full suite.
//
//   npm run test:smoke          (from tests/ui)
//
// It reuses every setting from playwright.config.js (one worker, same server wrapper, same
// retry policy) and only narrows WHICH spec files run, so a smoke pass is the same kind of
// run as a full pass, just shorter. It is a fast signal, not a substitute: the full suite is
// still the gate for merging.
//
// Selection rule: whole files, each short (about 30 s or less in CI), chosen so that
// together they cross every major surface once —
//   boot and DOM integrity ........ console-errors, html-integrity, performance
//   routing and navigation ........ spa-navigation
//   auth and TLS .................. security-auth, chat/auth-shell, tls-certificates
//   chat shell ..................... chat-connection, composer-draft
//   launch and wizard .............. launch-grid, spawn-wizard-served-name, kv-usecase,
//                                    phase7-presets (also the Rapid-MLX contract gate)
//   runtime and updates ............ llama-pill-interaction, inference-poll-failure
//   model/engine surfaces .......... engine-model-badge, evidence-drawer
// Add a file here only if it is fast and guards a surface no listed file already crosses.

import base from './playwright.config.js';

export default {
  ...base,
  testMatch: [
    'core/console-errors.spec.js',
    'core/html-integrity.spec.js',
    'core/performance.spec.js',
    'core/spa-navigation.spec.js',
    'core/security-auth.spec.js',
    'chat/auth-shell.spec.js',
    'core/tls-certificates.spec.js',
    'chat/chat-connection.spec.js',
    'chat/composer-draft.spec.js',
    'core/launch-grid.spec.js',
    'core/spawn-wizard-served-name.spec.js',
    'core/kv-usecase.spec.js',
    'core/phase7-presets.spec.js',
    'core/llama-pill-interaction.spec.js',
    'core/inference-poll-failure.spec.js',
    'core/engine-model-badge.spec.js',
    'core/evidence-drawer.spec.js',
  ],
};
