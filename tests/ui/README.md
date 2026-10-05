# UI Test Documentation

## Automated UI tests

Install dependencies from the repository root:

```bash
npm install
```

Run the Playwright suite:

```bash
npm test
npm test tests/ui/capability-rendering.spec.js
npm test -- --headless=false
npm test -- --debug
```

### Reading a local result

The suite runs **one worker, serially** — locally and in CI alike (`fullyParallel: false`,
`workers: 1`). Do not pass `--workers=N` to "speed things up" and then treat what comes back
as a verdict on the tests. Several specs share module-level wizard state and a single test
server; running them concurrently produces failures that say nothing about the code, and CI
will never reproduce them because CI never runs that way.

If you want parallelism deliberately, set `PLAYWRIGHT_PARALLEL=1`. Results from such a run
are for your own iteration speed only and must not be reported as a suite status.

Local retries are 0 where CI uses 2, on purpose: local is stricter, so a local pass implies
a CI pass. A local failure is worth investigating; it is not automatically a CI failure.

To exercise form-auth mode, pass extra server args through `LLAMA_MONITOR_TEST_ARGS`:

```bash
LLAMA_MONITOR_TEST_ARGS="--form-auth admin:secret123" npm test -- tests/ui/chat/auth-shell.spec.js
```

The UI suite covers:

- Top navigation, sidebar, and tab navigation
- Server/dashboard rendering
- Chat, logs, and chat-side controls
- Settings and configuration entry points
- Remote-agent flows
- Theme and responsive behavior
- Console/runtime regressions

## Screenshot and GIF capture harness

All repo-managed screenshots and GIFs go through `tests/ui/capture/index.mjs`.

The harness:

- launches `target/release/local-llm-foundry` on a temporary local port
- seeds a temporary config from local `ui-settings.json`, `presets.json`, and `gpu-env.json` when present
- attaches to `REMOTE_SERVER` unless the scenario is explicitly no-attach

If any `static/` file changed, rebuild first:

```bash
cargo build --release
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SCREENSHOT_PORT` | `8892` | Base port to try for the spawned dashboard |
| `REMOTE_SERVER` | `http://192.168.2.16:8001` | Remote llama.cpp server used by attach-based scenarios |
| `SCREENSHOT_FORM_AUTH` | `admin:secret123` | Credentials used for the auth-shell still captured by the `welcome` scenario |

### Listing scenarios

```bash
node tests/ui/capture/index.mjs --list-scenarios
```

### Wizard capture contract

Wizard still scenarios declare an intent and exact final output filenames. Before a run, the harness
removes only those filenames from that scenario's assigned artifact group; on success it writes a
`<scenario>--receipt.json` beside the images with the produced names and viewports. A missing,
unexpected, or unrealistic-viewport screenshot fails the scenario. Check static call-site metadata
without starting the app with:

```bash
node tests/ui/capture/cli-manifest.mjs --strict
```

Rapid wizard stills belong in `artifacts/wizard-rapidmlx`; llama.cpp wizard stills belong in
`artifacts/wizard-llamacpp`. Runtime/dashboard scenarios are grouped by their own feature area.

### Current scenarios

| Scenario | Purpose |
|----------|---------|
| `welcome` | Welcome/setup screen plus form-auth shell without remote attach |
| `free-cache` | Native Free Cache confirmation shown over the welcome/setup screen |
| `chat` | Core chat view, telemetry overlay, and logs |
| `guided-gen` | Context notes, suggestions, quick guide, director, surprise, explicit mode |
| `sidebar` | Chat sidebar, message-search flyout, context menu, title filter |
| `models-v2` | Typed model inventory, Import Lab, and Hugging Face download surfaces |
| `model-discovery` | HF Download tab search/discovery: scope selector, sort, category/qualification badges (real HF data, not the Library tab — `models-v2` covers Library) |
| `preset-editor` | llama.cpp preset model, GPU, and advanced sections |
| `rapid-preset` | Rapid-MLX welcome cards and preset editor with legacy and typed model-source fixtures |
| `settings` | Settings modal, performance tab, advanced tab, user preferences, persona, models, shortcuts |
| `panels` | Behavior, model, style, and prompt-debug surfaces |
| `dashboard` | Server tab and GPU section |
| `sparkline` | Sparkline validation stills and clipped metric captures |
| `gifs` | Animated inference and GPU/system captures |
| `smoke` | Startup smoke validation |
| `appearance-palette` | Settings Appearance palette stills and light-mode dashboard |
| `navbar` | Top nav bar close-ups: idle-dark, low-power active, idle-light; requires `--close-up` |

### Common commands

```bash
# Welcome screen only
node tests/ui/capture/index.mjs --scenario welcome

# Free Cache confirmation
node tests/ui/capture/index.mjs --scenario free-cache

# Core chat / logs / telemetry
SCREENSHOT_PORT=8892 node tests/ui/capture/index.mjs --scenario chat

# Guided-generation surfaces
SCREENSHOT_PORT=9001 node tests/ui/capture/index.mjs --scenario guided-gen

# Sidebar and search surfaces
SCREENSHOT_PORT=8893 node tests/ui/capture/index.mjs --scenario sidebar

# Rapid-MLX preset cards and editor source-contract coverage
SCREENSHOT_PORT=8902 node tests/ui/capture/index.mjs --scenario rapid-preset --no-attach

# Settings and modal surfaces
SCREENSHOT_PORT=8894 node tests/ui/capture/index.mjs --scenario settings

# Appearance palettes and light-mode dashboard
SCREENSHOT_PORT=8899 node tests/ui/capture/index.mjs --scenario appearance-palette --no-attach

# Chat configuration panels
SCREENSHOT_PORT=8896 node tests/ui/capture/index.mjs --scenario panels

# Server tab and GPU section
SCREENSHOT_PORT=8897 node tests/ui/capture/index.mjs --scenario dashboard

# Sparkline validation
SCREENSHOT_PORT=8898 node tests/ui/capture/index.mjs --scenario sparkline

# GIFs
SCREENSHOT_PORT=8895 node tests/ui/capture/index.mjs --scenario gifs
SCREENSHOT_PORT=8895 node tests/ui/capture/index.mjs --scenario gifs --gpu-only
SCREENSHOT_PORT=8895 node tests/ui/capture/index.mjs --scenario gifs --inference-only

# Smoke run
SCREENSHOT_PORT=8899 node tests/ui/capture/index.mjs --scenario smoke
```

### Useful options

| Option | Description |
|--------|-------------|
| `--gpu-only` | For `gifs`, capture only GPU/system animation |
| `--inference-only` | For `gifs`, capture only inference animation |
| `--no-attach` | Skip remote attach for scenarios that can run locally |
| `--close-up` | Capture extra element-level detail shots for debugging |
| `--list-scenarios` | Print the registered scenario names |

## Output locations

| Path | Purpose |
|------|---------|
| `docs/screenshots/` | Promoted hero assets used directly in `README.md` |
| `docs/screenshots/artifacts/` | Raw harness output for stills, GIFs, and validation captures |

The harness now writes its outputs to `docs/screenshots/artifacts/`. Promote selected stills or GIFs into `docs/screenshots/` only when you intentionally want a README-facing hero asset.

## Dashboard-related scenarios

When you change dashboard / metrics / server tab visuals, use this subset instead of the full suite:

- `dashboard` — primary Server tab + GPU section (produces `dashboard-performance-section.png`, `settings-server-tab.png`, `dashboard-gpu-section.png`)
- `gifs` — animated metric graphs
  - `--gpu-only` — only GPU/system metrics GIF
  - `--inference-only` — only inference metrics GIF
- `tune-panel` — tuning panel on Server tab
- `benchmark-results` — benchmark results view
- `llama-updater` — updater pill + version modal
- `appearance-palette` — includes light-mode dashboard screenshot

## Updating the harness

When adding or changing screenshot coverage:

1. Extend an existing scenario in `tests/ui/capture/index.mjs` when the surface already belongs to one.
2. Add a new scenario only when the coverage area is clearly distinct.
3. Register the scenario in the `SCENARIOS` map.
4. Update the usage text in `printUsage()`.
5. Update this README in the same change.

## Troubleshooting

- If captures do not reflect your latest UI edits, rebuild with `cargo build --release`.
- If attach-based scenarios fail, confirm `REMOTE_SERVER` is reachable and returns normal llama.cpp responses.
- If a port is busy, raise `SCREENSHOT_PORT`; the harness scans forward from that base.
- If a popover or panel appears missing, add geometry/state logging to the scenario before skipping the capture.
- If the scenario leaves extra test chats behind, use the shared screenshot-tab helpers in the harness instead of ad hoc tab mutations.
