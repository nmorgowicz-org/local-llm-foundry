// Capture CLI entry point: arg parsing, scenario registry, run orchestration.
// Rebuilt from tests/ui/capture.mjs (Phase A1-A4 split); see docs/plans/
// 20260804-branch_audit_capture_split_chat_template_ux.md Phase A.
import {
    DEFAULT_VIEWPORT, RUNNING_PORT, setArtifactCategory, setArtifactRuntime,
} from './harness/paths.mjs';
import { seedConfig, findAvailablePort, spawnLlamaMonitor, cleanupServer, cleanupTempHome } from './harness/server.mjs';
import { seedRapidMlxCapturePreset, seedNestedMlxFixture, seedCanonicalMlxFixture, seedModelsDirFixture, seedCalibrationCapturePreset } from './harness/fixtures.mjs';
import { launchBrowser } from './harness/browser.mjs';
import { cleanupFrames } from './harness/shot.mjs';
import { beginCaptureReceipt, finishCaptureReceipt, setCaptureDiagnostics } from './harness/receipt.mjs';

import scenarioWelcome from './scenarios/core/welcome.mjs';
import scenarioFreeCache from './scenarios/core/free-cache.mjs';
import scenarioChat from './scenarios/core/chat.mjs';
import scenarioSidebar from './scenarios/core/sidebar.mjs';
import scenarioGuidedGen from './scenarios/core/guided-gen.mjs';
import scenarioNavbar from './scenarios/core/navbar.mjs';
import scenarioSmoke from './scenarios/core/smoke.mjs';
import scenarioPanels from './scenarios/core/panels.mjs';
import scenarioDashboard from './scenarios/core/dashboard.mjs';
import scenarioModels from './scenarios/models/models.mjs';
import scenarioModelsV2 from './scenarios/models/models-v2.mjs';
import scenarioModelDiscovery from './scenarios/models/model-discovery.mjs';
import scenarioFilebrowser from './scenarios/models/filebrowser.mjs';
import scenarioModelBrowser from './scenarios/models/model-browser.mjs';
import scenarioPresetEditor from './scenarios/presets/preset-editor.mjs';
import scenarioCalibration from './scenarios/presets/calibration.mjs';
import scenarioRapidPreset from './scenarios/presets/rapid-preset.mjs';
import scenarioEvidenceDrawer from './scenarios/presets/evidence-drawer.mjs';
import scenarioCommunitySources from './scenarios/presets/community-sources.mjs';
import scenarioDiscussions from './scenarios/presets/discussions.mjs';
import scenarioPresetBundleEditor from './scenarios/presets/preset-bundle-editor.mjs';
import scenarioPresetBundle from './scenarios/presets/preset-bundle.mjs';
import scenarioLaunchGrid from './scenarios/presets/launch-grid.mjs';
import scenarioWelcomeHero from './scenarios/presets/welcome-hero.mjs';
import scenarioSpawnWizard from './scenarios/wizard-llamacpp/spawn-wizard.mjs';
import scenarioSpawnWizardProBaseline from './scenarios/wizard-llamacpp/spawn-wizard-pro-baseline.mjs';
import scenarioSpawnWizardLaunchFullConfig from './scenarios/wizard-llamacpp/spawn-wizard-launch-full-config.mjs';
import scenarioSpawnWizardRapidGuidedBaseline from './scenarios/wizard-rapidmlx/spawn-wizard-rapid-guided-baseline.mjs';
import scenarioSpawnWizardRapidProBaseline from './scenarios/wizard-rapidmlx/spawn-wizard-rapid-pro-baseline.mjs';
import scenarioSpawnWizardRapidLaunchFullConfig from './scenarios/wizard-rapidmlx/spawn-wizard-rapid-launch-full-config.mjs';
import scenarioSpawnWizardGif from './scenarios/wizard-llamacpp/spawn-wizard-gif.mjs';
import scenarioSpawnWizardHfDownload from './scenarios/wizard-llamacpp/spawn-wizard-hf-download.mjs';
import scenarioSpawnWizardTierMatrix from './scenarios/wizard-llamacpp/spawn-wizard-tier-matrix.mjs';
import scenarioSpawnWizardMmprojSelection from './scenarios/wizard-llamacpp/spawn-wizard-mmproj-selection.mjs';
import scenarioSpawnWizardGuidedDrawer from './scenarios/wizard-llamacpp/spawn-wizard-guided-drawer.mjs';
import scenarioSpawnWizardCalibration from './scenarios/wizard-llamacpp/spawn-wizard-calibration.mjs';
import scenarioSpawnWizardRapidMlxGif from './scenarios/wizard-rapidmlx/spawn-wizard-rapid-mlx-gif.mjs';
import scenarioRapidMlxRuntime from './scenarios/features/rapid-mlx-runtime.mjs';
import scenarioRapidMlxLive from './scenarios/validation/rapid-mlx-live.mjs';
import scenarioDashboardRapidMlx from './scenarios/features/dashboard-rapid-mlx.mjs';
import scenarioSettings from './scenarios/config/settings.mjs';
import scenarioAppearancePalette from './scenarios/config/appearance-palette.mjs';
import scenarioTls from './scenarios/config/tls.mjs';
import scenarioTunePanel from './scenarios/features/tune-panel.mjs';
import scenarioBenchmarkResults from './scenarios/features/benchmark-results.mjs';
import scenarioLlamaUpdater from './scenarios/features/llama-updater.mjs';
import scenarioChatHistoryQA from './scenarios/features/chat-history-qa.mjs';
import scenarioSparkline from './scenarios/validation/sparkline.mjs';
import scenarioGifs from './scenarios/validation/gifs.mjs';

export function parseArgs(argv) {
    const options = {
        scenario: null,
        gpuOnly: false,
        inferenceOnly: false,
        listScenarios: false,
        noAttach: false,
        closeUp: false,
        source: null,
        viewport: { ...DEFAULT_VIEWPORT },
    };

    for (let i = 0; i < argv.length; i += 1) {
        const arg = argv[i];
        if (arg === '--scenario' && argv[i + 1]) {
            options.scenario = argv[i + 1];
            i += 1;
        } else if (arg === '--chat-only') {
            options.chatOnly = true;
        } else if (arg === '--gpu-only') {
            options.gpuOnly = true;
        } else if (arg === '--inference-only') {
            options.inferenceOnly = true;
        } else if (arg === '--list-scenarios') {
            options.listScenarios = true;
        } else if (arg === '--no-attach') {
            options.noAttach = true;
        } else if (arg === '--source' && argv[i + 1]) {
            options.source = argv[i + 1];
            i += 1;
        } else if (arg === '--close-up') {
            options.closeUp = true;
        } else if (arg === '--help' || arg === '-h') {
            options.help = true;
        }
    }

    if (options.gpuOnly && options.inferenceOnly) {
        throw new Error('Use only one of --gpu-only or --inference-only');
    }

    return options;
}

function printUsage() {
    console.log(`Usage:
  node tests/ui/capture.mjs --scenario <name> [options]

Scenarios:
  Core
    welcome          Welcome screen, auth shell, and setup wizard button (no attach required)
    free-cache       Native Free Cache confirmation on the welcome screen
    chat             Chat, telemetry, logs

  Chat Features
    guided-gen       Suggestions, quick guide, director, surprise, explicit mode
    sidebar          Sidebar, FTS search flyout, context menu, title filter
    chat-history-qa  History Q&A panel (ask questions about conversation)

  Models and Presets
    models-v2        Models modal: typed inventory, Import Lab, and HF download panel
    model-discovery  HF Download tab: scope selector, sort, category/qualification badges (real HF data)
    preset-editor    Preset editor: model/context, GPU, and advanced tabs
    calibration      Opt-in Calibration modal states using a real local GGUF for preflight
    rapid-preset     Rapid-MLX welcome cards and preset editor (legacy and typed sources)
    evidence-drawer  Shared decision evidence drawer: dark, expanded, light, and narrow reduced-motion
    community-sources Model Manager source catalog: list and editor in dark/light/narrow states

  Configuration
    settings         Settings modal, preferences, persona, models, shortcuts
    tls              TLS modes and ACME (Certificates tab, each TLS mode, custom certs, ACME config)
    filebrowser      File browser modal (Browse buttons in Config modal, modal open)
    model-browser    Intent-aware model picker from the Spawn Wizard (Rapid-MLX and llama.cpp)
    panels           Chat config panels (behavior, model, style, debug)
    dashboard        Server tab, GPU section
    dashboard-rapid-mlx  Deterministic Rapid-MLX telemetry cards (dark and light)

   Setup wizard
     spawn-wizard           3-step wizard: model & profiles, hardware & VRAM, launch summary/spawn
     spawn-wizard-engines      llama.cpp/Rapid-MLX engine cards and Rapid-MLX hardware handoff
     spawn-wizard-gif          Animated GIF walking through all setup wizard steps (llama.cpp path)
     spawn-wizard-rapid-mlx-gif  Animated GIF: Rapid-MLX engine, hardware controls, profile hints, spawn
     spawn-wizard-hf-download  HF download panel: idle options and simulated progress
   spawn-wizard-tier-matrix  Hardware step at Quick/Balanced/Advanced tiers (registry disclosure)
   spawn-wizard-mmproj-selection  Hardware vision/mmproj selector with typed recommendation evidence
     discussions             Chat template Discussions: Qwen (froggeric) and Gemma4 (google) feeds,
                            Create fix modal, and Lifecycle modal Discussions section

   Performance & Updates
    tune-panel          Performance benchmark dropdown (pill + panel) on server tab
    benchmark-results   Runs a real benchmark and captures the results view
    llama-updater       Llama-server binary update pill and version modal with release notes

  Validation
    sparkline        Sparkline validation screenshots
    gifs             Inference/GPU animated GIF capture
    smoke            Startup smoke test

  Rapid-MLX Runtime (developer only, NOT CI)
    rapid-mlx-live   Full runtime flow: spawn Qwen3-0.6B-4bit → telemetry → chat → stop

  Appearance
    appearance-palette Settings Appearance palette stills plus light-mode dashboard
    navbar           Nav bar close-ups: idle-dark, low-power active, idle-light (requires --close-up)

Options:
  --source <name> Select capture source (Phase 1 implements: remote)
  --gpu-only         For gifs scenario, capture only GPU/system animation
  --inference-only   For gifs scenario, capture only inference animation
  --no-attach        Skip remote attach for scenarios that do not require it
  --close-up         Also capture element-level close-ups (debugging only)
  --list-scenarios   Print available scenarios

Examples:
  node tests/ui/capture.mjs --scenario welcome
  SCREENSHOT_PORT=8892 node tests/ui/capture.mjs --scenario chat
  SCREENSHOT_PORT=9001 node tests/ui/capture.mjs --scenario guided-gen
  SCREENSHOT_PORT=8893 node tests/ui/capture.mjs --scenario sidebar
  SCREENSHOT_PORT=8895 node tests/ui/capture.mjs --scenario gifs --gpu-only
  SCREENSHOT_PORT=8894 node tests/ui/capture.mjs --scenario settings --close-up
  SCREENSHOT_PORT=8896 node tests/ui/capture.mjs --scenario spawn-wizard --no-attach
  SCREENSHOT_PORT=8898 node tests/ui/capture.mjs --scenario spawn-wizard-engines --no-attach
  SCREENSHOT_PORT=8897 node tests/ui/capture.mjs --scenario spawn-wizard-gif --no-attach
   SCREENSHOT_PORT=8997 node tests/ui/capture.mjs --scenario spawn-wizard-rapid-mlx-gif --no-attach
   SCREENSHOT_PORT=8910 node tests/ui/capture.mjs --scenario discussions --no-attach
   SCREENSHOT_PORT=8900 node tests/ui/capture.mjs --scenario tune-panel
   SCREENSHOT_PORT=8901 node tests/ui/capture.mjs --scenario llama-updater
  SCREENSHOT_PORT=8902 node tests/ui/capture.mjs --scenario chat-history-qa
   RUNNING_PORT=8080 node tests/ui/capture.mjs --scenario dashboard
   RUNNING_PORT=8080 node tests/ui/capture.mjs --scenario gifs --gpu-only
   SCREENSHOT_PORT=8910 node tests/ui/capture.mjs --scenario rapid-mlx-live

  Note: RUNNING_PORT connects to an already-running llama-monitor (e.g. your production instance
with a remote agent reporting GPU data). No binary is spawned; no temp config is seeded.

Calibration capture is opt-in and requires CALIBRATION_CAPTURE_MODEL=<local GGUF path>;
optionally set CALIBRATION_CAPTURE_MODELS_DIR, CALIBRATION_CAPTURE_LLAMA_SERVER, or
LLAMA_MONITOR_CAPTURE_CONFIG_DIR. It performs real preflight only; benchmark lifecycle
responses are intercepted for stable screenshots.
`);
}

// runtime: which backend the captured UI reflects — 'neutral' for
// backend-independent chrome, otherwise 'llamacpp-local' / 'llamacpp-remote' /
// 'rapidmlx-local' / 'mtplx-local'. Individual shots in a mixed-runtime
// scenario can override via captureShot(..., { runtimeTag: '...' }).
export const SCENARIOS = {
    'welcome': { run: scenarioWelcome, category: 'core', runtime: 'neutral' },
    'free-cache': { run: scenarioFreeCache, category: 'core', runtime: 'neutral' },
    'rapid-preset': { run: scenarioRapidPreset, setup: () => { seedRapidMlxCapturePreset(); }, category: 'presets', runtime: 'rapidmlx-local', requiresRapidMlx: true },
    'evidence-drawer': { run: scenarioEvidenceDrawer, category: 'presets', runtime: 'neutral' },
    'community-sources': { run: scenarioCommunitySources, category: 'presets', runtime: 'neutral' },
    'chat': { run: scenarioChat, source: 'remote', category: 'core', runtime: 'neutral' },
    'guided-gen': { run: scenarioGuidedGen, category: 'core', runtime: 'neutral' },
    'sidebar': { run: scenarioSidebar, category: 'core', runtime: 'neutral' },
    'models-v2': { run: scenarioModelsV2, setup: () => ({ extraArgs: seedModelsDirFixture() }), category: 'models', runtime: 'neutral' },
    'model-discovery': { run: scenarioModelDiscovery, category: 'models', runtime: 'neutral' },
    'preset-editor': { run: scenarioPresetEditor, category: 'presets', runtime: 'neutral' },
    'preset-bundle-editor': { run: scenarioPresetBundleEditor, category: 'presets', runtime: 'llamacpp-local' },
    'preset-bundle': {
        run: scenarioPresetBundle,
        category: 'presets',
        runtime: 'llamacpp-local',
        contract: {
            intent: 'Prove the delivered bundle launch card and Configure drawer visually match the architecture hierarchy.',
            expectedOutputs: [
                'llamacpp-local--preset-bundle-grid-dark.png',
                'llamacpp-local--preset-bundle-grid-light.png',
                'llamacpp-local--preset-bundle-drawer-default-dark.png',
                'llamacpp-local--preset-bundle-drawer-light.png',
                'llamacpp-local--preset-bundle-drawer-narrow.png',
                'llamacpp-local--preset-bundle-drawer-reduced-motion.png',
                'llamacpp-local--preset-bundle-drawer-low-vram-changes.png',
                'llamacpp-local--preset-bundle-drawer-no-fit.png',
                'llamacpp-local--preset-bundle-drawer-evidence-exact.png',
                'llamacpp-local--preset-bundle-drawer-evidence-related.png',
                'llamacpp-local--preset-bundle-drawer-revision-conflict.png',
            ],
        },
    },
    'launch-grid': {
        run: scenarioLaunchGrid,
        category: 'presets',
        runtime: 'llamacpp-local',
        contract: {
            intent: 'Prove the launch grid\'s sort order, collections filter, running badge, and per-card delete confirm all render correctly.',
            expectedOutputs: [
                'llamacpp-local--launch-grid-sort-name.png',
                'llamacpp-local--launch-grid-sort-size.png',
                'llamacpp-local--launch-grid-collections-filtered.png',
                'llamacpp-local--launch-grid-running-badge.png',
                'llamacpp-local--launch-grid-delete-confirm.png',
            ],
        },
    },
    'welcome-hero': {
        run: scenarioWelcomeHero,
        category: 'presets',
        runtime: 'welcome-hero',
        contract: {
            intent: 'Provide a single README-showcase hero image of the welcome screen with bundled presets: names, quant badges, context, KV policy, and a populated collections filter all visible at once.',
            expectedOutputs: ['welcome-hero--preset-bundle-hero.png'],
        },
    },
    'calibration': {
        run: scenarioCalibration,
        setup: () => ({ extraArgs: seedCalibrationCapturePreset() }),
        category: 'presets',
        runtime: 'llamacpp-calibration-local',
        requiresCalibrationModel: true,
    },
    'settings': { run: scenarioSettings, category: 'config', runtime: 'neutral' },
    'appearance-palette': { run: scenarioAppearancePalette, category: 'config', runtime: 'neutral' },
    'tls': { run: scenarioTls, category: 'config', runtime: 'neutral' },
    'filebrowser': { run: scenarioFilebrowser, category: 'models', runtime: 'neutral' },
    'model-browser': {
        run: scenarioModelBrowser,
        setup: () => {
            const extraArgs = seedModelsDirFixture();
            seedCanonicalMlxFixture();
            return { extraArgs };
        },
        category: 'models',
        runtime: 'neutral',
    },
    'panels': { run: scenarioPanels, setup: () => ({ extraArgs: seedModelsDirFixture() }), category: 'core', runtime: 'neutral' },
    'models': { run: scenarioModels, setup: () => ({ extraArgs: seedModelsDirFixture() }), category: 'models', runtime: 'neutral' },
    'dashboard': { run: scenarioDashboard, source: 'remote', category: 'core', runtime: 'neutral' },
    // Synthetic DOM-only telemetry cards are intentionally cross-platform; they
    // do not query platform-info or launch a Rapid executable.
    'dashboard-rapid-mlx': { run: scenarioDashboardRapidMlx, category: 'features', runtime: 'rapidmlx-local' },
    'spawn-wizard': {
        run: scenarioSpawnWizard, category: 'wizard-llamacpp', runtime: 'llamacpp-local',
        contract: {
            intent: 'Document the current llama.cpp wizard baseline with complete, realistic still evidence.',
            expectedOutputs: [
                'spawn-wizard--llamacpp-local--model-profiles.png', 'spawn-wizard--llamacpp-local--model-source-cards.png',
            ],
        },
    },
    'spawn-wizard-calibration': {
      run: scenarioSpawnWizardCalibration,
      category: 'wizard-llamacpp',
      runtime: 'llamacpp-local',
      contract: {
        intent: 'Capture exact and compatible calibration evidence reuse plus canonical candidate application in Pro review.',
        expectedOutputs: [
      'spawn-wizard-calibration--llamacpp-local--evidence.png',
      'spawn-wizard-calibration--llamacpp-local--applied.png',
      'spawn-wizard-calibration--llamacpp-local--related-review.png',
        ],
      },
    },
    'spawn-wizard-pro-baseline': {
        run: scenarioSpawnWizardProBaseline, category: 'wizard-llamacpp', runtime: 'llamacpp-local',
        contract: {
            intent: 'Capture the active Pro shell, category rail, search, modified/reset state, and responsive presentation.',
      expectedOutputs: [
        'llamacpp-local--spawn-wizard-pro-shell.png',
        'llamacpp-local--spawn-wizard-pro-rail-model.png',
        'llamacpp-local--spawn-wizard-pro-rail-memory.png',
        'llamacpp-local--spawn-wizard-pro-rail-performance.png',
        'llamacpp-local--spawn-wizard-pro-rail-generation.png',
        'llamacpp-local--spawn-wizard-pro-rail-tools.png',
        'llamacpp-local--spawn-wizard-pro-rail-network.png',
        'llamacpp-local--spawn-wizard-pro-rail-advanced.png',
        'llamacpp-local--spawn-wizard-pro-search-batch.png',
        'llamacpp-local--spawn-wizard-pro-modified-only.png',
        'llamacpp-local--spawn-wizard-pro-reset-light.png',
        'llamacpp-local--spawn-wizard-pro-agentic-q4-warning.png',
        'llamacpp-local--spawn-wizard-pro-roleplay-q4-baseline.png',
        'llamacpp-local--spawn-wizard-pro-narrow.png',
        'llamacpp-local--spawn-wizard-pro-model-compatibility.png',
        'llamacpp-local--spawn-wizard-pro-generation-reasoning.png',
        ],
        },
    },
  'spawn-wizard-rapid-guided-baseline': {
        run: scenarioSpawnWizardRapidGuidedBaseline, setup: () => { seedNestedMlxFixture(); }, category: 'wizard-rapidmlx', runtime: 'rapidmlx-local', requiresRapidMlx: true,
        contract: {
            intent: 'Document the current Rapid-MLX wizard baseline in its own Rapid artifact group.',
            expectedOutputs: [
                'spawn-wizard-rapid-guided-baseline--rapidmlx-local--dark.png', 'spawn-wizard-rapid-guided-baseline--rapidmlx-local--light.png',
                'spawn-wizard-rapid-guided-baseline--rapidmlx-local--reduced-narrow.png', 'rapidmlx-local--spawn-wizard-reasoning-mode-on.png',
                'rapidmlx-local--spawn-wizard-speculative-enabled-dark.png', 'rapidmlx-local--spawn-wizard-speculative-enabled-light.png',
                'rapidmlx-local--spawn-wizard-speculative-trust-consent-dark.png', 'rapidmlx-local--spawn-wizard-speculative-trust-consent-light.png',
                'rapidmlx-local--spawn-wizard-parser-detected.png', 'rapidmlx-local--spawn-wizard-rapid-mlx-advanced-controls.png',
                'rapidmlx-local--spawn-wizard-rapid-mlx-escape-hatch.png', 'rapidmlx-local--spawn-wizard-rapid-mlx-fit.png',
                'rapidmlx-local--spawn-wizard-rapid-mlx-review.png',
            ],
    },
  },
  'spawn-wizard-rapid-pro-baseline': {
    run: scenarioSpawnWizardRapidProBaseline,
    setup: () => { seedNestedMlxFixture(); },
    category: 'wizard-rapidmlx',
    runtime: 'rapidmlx-local',
    requiresRapidMlx: true,
    contract: {
      intent: 'Capture Rapid-MLX Pro parity, effective-state, access, filtering, companion-gating, and responsive evidence.',
      expectedOutputs: [
        'rapidmlx-local--spawn-wizard-rapid-pro-shell.png',
        'rapidmlx-local--spawn-wizard-rapid-pro-rail-model.png',
        'rapidmlx-local--spawn-wizard-rapid-pro-rail-memory.png',
        'rapidmlx-local--spawn-wizard-rapid-pro-rail-performance.png',
        'rapidmlx-local--spawn-wizard-rapid-pro-rail-generation.png',
        'rapidmlx-local--spawn-wizard-rapid-pro-rail-tools.png',
        'rapidmlx-local--spawn-wizard-rapid-pro-rail-network.png',
        'rapidmlx-local--spawn-wizard-rapid-pro-rail-advanced.png',
        'rapidmlx-local--spawn-wizard-rapid-pro-retained-cache-edit.png',
        'rapidmlx-local--spawn-wizard-rapid-pro-search-cache.png',
        'rapidmlx-local--spawn-wizard-rapid-pro-modified-only.png',
        'rapidmlx-local--spawn-wizard-rapid-pro-companion-gated.png',
        'rapidmlx-local--spawn-wizard-rapid-pro-narrow.png',
      ],
    },
  },
'spawn-wizard-launch-full-config': {
  run: scenarioSpawnWizardLaunchFullConfig,
  category: 'wizard-llamacpp',
  runtime: 'llamacpp-local',
  contract: {
    intent: 'Capture canonical llama.cpp launch Full config with requested and estimator-effective labels.',
    expectedOutputs: [
      'llamacpp-local--spawn-wizard-launch-full-config.png',
      'spawn-wizard-launch-full-config--llamacpp-local--details.png',
      'spawn-wizard-launch-full-config--llamacpp-local--narrow.png',
    ],
  },
},
'spawn-wizard-rapid-launch-full-config': {
  run: scenarioSpawnWizardRapidLaunchFullConfig,
  category: 'wizard-rapidmlx',
  runtime: 'rapidmlx-local',
  requiresRapidMlx: true,
  contract: {
    intent: 'Capture canonical Rapid-MLX launch Full config with requested and runtime-effective labels.',
    expectedOutputs: [
      'rapidmlx-local--spawn-wizard-launch-full-config.png',
      'rapidmlx-local--spawn-wizard-launch-full-config-details.png',
      'rapidmlx-local--spawn-wizard-launch-full-config-narrow.png',
    ],
  },
},
'spawn-wizard-gif': {
        run: scenarioSpawnWizardGif, category: 'wizard-llamacpp', runtime: 'llamacpp-local',
        contract: { intent: 'Capture the llama.cpp wizard flow as a diagnostic GIF.', expectedOutputs: ['llamacpp-local--spawn-wizard-flow.gif'] },
    },
    'spawn-wizard-hf-download': {
        run: scenarioSpawnWizardHfDownload, category: 'wizard-llamacpp', runtime: 'llamacpp-local',
        contract: { intent: 'Capture the llama.cpp HF download panel idle and progress states.', expectedOutputs: ['spawn-wizard-hf-download--llamacpp-local--idle.png', 'spawn-wizard-hf-download--llamacpp-local--progress.png'] },
    },
    'spawn-wizard-tier-matrix': {
        run: scenarioSpawnWizardTierMatrix, category: 'wizard-llamacpp', runtime: 'llamacpp-local',
        contract: { intent: 'Confirm legacy Quick/Balanced/Advanced disclosure is retired and capture the single Guided surface.', expectedOutputs: ['spawn-wizard-tier-matrix--llamacpp-local--guided.png'] },
    },
    'spawn-wizard-mmproj-selection': {
        run: scenarioSpawnWizardMmprojSelection, category: 'wizard-llamacpp', runtime: 'llamacpp-local',
        contract: {
            intent: 'Capture the llama.cpp hardware vision/mmproj selector with a typed family-backed recommendation.',
            expectedOutputs: ['spawn-wizard-mmproj-selection--llamacpp-local--vision.png'],
        },
    },
    'spawn-wizard-guided-drawer': {
        run: scenarioSpawnWizardGuidedDrawer, category: 'wizard-llamacpp', runtime: 'llamacpp-local', requiresRapidMlx: true,
        contract: {
            intent: 'Verify the single canonical Guided settings drawer across llama.cpp and Rapid-MLX, including engine switching and honest Pro availability.',
            expectedOutputs: [
                'spawn-wizard-guided-drawer--llamacpp-local--llama-closed.png',
                'spawn-wizard-guided-drawer--llamacpp-local--llama-open.png',
                'spawn-wizard-guided-drawer--rapidmlx-local--rapid-closed.png',
                'spawn-wizard-guided-drawer--rapidmlx-local--rapid-open.png',
                'spawn-wizard-guided-drawer--llamacpp-local--pro-available.png',
                'llamacpp-local--spawn-wizard-guided-cache-slots.png',
                'llamacpp-local--spawn-wizard-guided-mmproj-offload.png',
                'llamacpp-local--spawn-wizard-guided-reasoning.png',
            ],
        },
    },
    'spawn-wizard-rapid-mlx-gif': {
        run: scenarioSpawnWizardRapidMlxGif, category: 'wizard-rapidmlx', runtime: 'rapidmlx-local', requiresRapidMlx: true,
        contract: { intent: 'Capture the Rapid-MLX wizard flow as a diagnostic GIF.', expectedOutputs: ['rapidmlx-local--spawn-wizard-rapid-mlx-flow.gif'] },
    },
    'discussions': { run: scenarioDiscussions, setup: () => { seedRapidMlxCapturePreset(); }, category: 'presets', runtime: 'neutral', requiresRapidMlx: true },
    'tune-panel': { run: scenarioTunePanel, category: 'features', runtime: 'neutral' },
    'benchmark-results': { run: scenarioBenchmarkResults, category: 'features', runtime: 'neutral' },
    'llama-updater': { run: scenarioLlamaUpdater, category: 'features', runtime: 'llamacpp-local' },
    'chat-history-qa': { run: scenarioChatHistoryQA, category: 'features', runtime: 'neutral' },
    'rapid-mlx-runtime': { run: scenarioRapidMlxRuntime, category: 'features', runtime: 'rapidmlx-local', requiresRapidMlx: true },
    'rapid-mlx-live': { run: scenarioRapidMlxLive, category: 'validation', runtime: 'rapidmlx-local', requiresRapidMlx: true },
    'sparkline': { run: scenarioSparkline, category: 'validation', runtime: 'neutral' },
    'gifs': { run: scenarioGifs, category: 'validation', runtime: 'neutral' },
    'smoke': { run: scenarioSmoke, source: 'remote', category: 'core', runtime: 'neutral' },
    'navbar': { run: scenarioNavbar, category: 'core', runtime: 'neutral' },
};

/**
 * Return a deterministic skip reason before a scenario creates a temp home,
 * starts the application, or launches Chromium. Rapid-MLX local evidence is
 * only meaningful on Apple Silicon macOS; Windows/Linux should skip it rather
 * than exercising the platform gate and timing out in the browser.
 */
export function capturePlatformSkipReason(scenario, host = { platform: process.platform, arch: process.arch }) {
    if (scenario?.requiresRapidMlx && (host.platform !== 'darwin' || host.arch !== 'arm64')) {
        return `requires Apple Silicon macOS (host=${host.platform}/${host.arch})`;
    }
    if (scenario?.requiresCalibrationModel && !process.env.CALIBRATION_CAPTURE_MODEL) {
        return 'requires explicit CALIBRATION_CAPTURE_MODEL (opt-in; no benchmark is run by this scenario)';
    }
    return null;
}

export async function runCli({ scenario: forcedScenario = null, argv = process.argv.slice(2) } = {}) {
    const options = parseArgs(argv);
    if (options.help) {
        printUsage();
        return;
    }
    if (options.listScenarios) {
        Object.keys(SCENARIOS).forEach(name => console.log(name));
        return;
    }

    // Default to the welcome flow so an unqualified invocation still succeeds
    // without requiring a remote attach.
    const scenarioName = forcedScenario || options.scenario || 'welcome';
    const scenario = SCENARIOS[scenarioName];
    if (!scenario) {
        throw new Error(`Unknown scenario "${scenarioName}". Use --list-scenarios.`);
    }
    const skipReason = capturePlatformSkipReason(scenario);
    if (skipReason) {
        console.log(`[CAPTURE] SKIP "${scenarioName}": ${skipReason}.`);
        return { skipped: true, reason: skipReason };
    }
    setArtifactCategory(scenario.category);
    setArtifactRuntime(scenarioName, scenario.runtime);
    if (scenario.contract) beginCaptureReceipt({ scenario: scenarioName, category: scenario.category, runtime: scenario.runtime, ...scenario.contract });

    let server = null;
    let browser = null;
    let baseUrl;

    if (RUNNING_PORT) {
        baseUrl = `http://127.0.0.1:${RUNNING_PORT}`;
        console.log(`[CAPTURE] Using running llama-monitor at ${baseUrl} for scenario "${scenarioName}"...`);
    } else {
        seedConfig();
        const setupResult = scenario.setup ? (await scenario.setup()) || {} : {};
        const extraArgs = setupResult.extraArgs || [];

        const port = await findAvailablePort();
        console.log(`[CAPTURE] Spawning llama-monitor on port ${port} for scenario "${scenarioName}"...`);
        server = await spawnLlamaMonitor(port, extraArgs);
        baseUrl = server.url;
    }

    // Headless Chrome occasionally crashes its renderer mid-session on long,
    // real-network-bound scenarios (frame/context gets detached out from under
    // an in-flight page.evaluate). This is a Puppeteer/Chrome-level flake, not
    // an app bug, so retry once with a freshly launched browser before giving up.
    const maxAttempts = 2;
    let lastErr = null;
    for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
        cleanupFrames();
        browser = null;
        try {
            const launched = await launchBrowser(options.viewport);
            browser = launched.browser;
            const page = launched.page;
            await scenario.run({ page, baseUrl, browser }, { ...options, scenarioSource: scenario.source });
            if (scenario.contract) {
                setCaptureDiagnostics(page.__fontDiagnostics || null);
                finishCaptureReceipt();
            }
            console.log(`[CAPTURE] Scenario "${scenarioName}" complete.`);
            lastErr = null;
            break;
        } catch (err) {
            lastErr = err;
            const detached = /detached frame/i.test(err.message || '');
            console.error(err.stack || err.message);
            if (detached && attempt < maxAttempts) {
                console.log(`[CAPTURE] Detached-frame crash on attempt ${attempt}; retrying with a fresh browser...`);
            } else {
                break;
            }
        } finally {
            if (browser) await browser.close();
        }
    }
    if (lastErr) {
        process.exitCode = 1;
    }
    if (server) await cleanupServer(server);
    if (!RUNNING_PORT) cleanupTempHome();
}

if (import.meta.url === `file://${process.argv[1]}`) {
    await runCli();
}
