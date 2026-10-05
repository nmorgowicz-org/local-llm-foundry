# Upgrade from Llama Monitor 1.x to Local LLM Foundry 2.0

Local LLM Foundry 2.0 is the product rebrand of Llama Monitor. It is a
compatibility-preserving upgrade, not a data reset. Legacy application roots,
tasks, processes, certificates, APIs, and browser-storage identifiers remain
discoverable through the 2.x line. Builds produce only `local-llm-foundry`.

## Before you upgrade

1. Stop active model servers if you are changing the application root.
2. Record the current config root, model root, custom binary paths, and any
   remote-agent targets.
3. Keep a backup of `chat.db`, `agent-tokens.json`, certificates, and custom
   presets. The app never deletes the legacy root or external model roots.
4. On Windows, ensure WebView2 is installed before validating the tray popover.

Releases publish four canonical `local-llm-foundry-*` assets. Checksums cover
every exact published filename. Legacy-named assets belonged to the 2.0.x
release bridge; current builds neither compile nor package `llama-monitor`.

## What changes

| Surface | 2.0 canonical | 1.x compatibility |
|---|---|---|
| Product | Local LLM Foundry | Llama Monitor is retained in historical and compatibility copy |
| GUI/CLI binary | `local-llm-foundry` / `.exe` | Update scripts and service commands to the canonical executable |
| Unix/macOS root | `~/.config/local-llm-foundry` | `~/.config/llama-monitor` remains discoverable |
| Windows root | `%APPDATA%\local-llm-foundry` | `%APPDATA%\llama-monitor` remains discoverable |
| Windows agent task | `LocalLLMFoundryAgent` | `llama-monitor-agent` is detected and retired only during explicit repair |
| Sensor task | `LocalLLMFoundrySensorBridge` | `LlamaMonitorSensorBridge` remains detectable |
| Release assets | `local-llm-foundry-*` | `llama-monitor-*` aliases only in historical 2.0.x releases |
| Rust library | `llama_monitor` | Stable internal namespace |

API routes, unbranded serialized fields, authentication, encryption identifiers,
and browser storage keys remain stable. Backend technology names such as
`llama.cpp`, `llama-server`, GGUF, MLX, Rapid-MLX, and LHM are not rebranded.

## First launch and migration choice

On first 2.0 launch, the app detects legacy state and shows a migration toast.
Choose **Keep legacy root** to continue using the existing files, or choose
**Migrate** to copy state into the canonical root. Migration is copy-first,
receipt-backed, resumable, and never silently merges two roots. The original
legacy root remains available until you explicitly remove it.

Migration explicitly does not copy the model tree: models are inventoried and
retained at the legacy root for an explicit follow-up decision. The destructive
operations are guarded against the only-copy scenario:

- **Rollback** refuses to run while the canonical root holds a populated
  `models` directory (a move would have left it as the only copy), and it
  removes the migration receipt so a later migration re-copies instead of
  early-returning an empty root.
- **Legacy-root cleanup** refuses to run while the legacy root still holds a
  populated `models` directory.
- **Model-root moves** are refused until the application-home migration has
  settled, because `models` inside the canonical root is itself a root-state
  marker and creating it early strands startup in `Conflict`.
- An interrupted migration resumes from its journal on the next start instead
  of blocking startup.

If the migration notice is already in notification history after a reload,
**Review migration** remains enabled and opens Settings → Migration.

Model libraries follow the same policy. An external/custom model root is never
moved automatically. Use the Model Library migration preview to select a
destination and retain or remove the source only after verification. Model
moves are a single rename with no copy fallback, so cross-filesystem moves are
rejected at planning time.

## Headless and remote upgrades

The canonical binary accepts every existing flag. Update scripts and service
commands to invoke `local-llm-foundry`; there is no legacy executable target:

```bash
./local-llm-foundry --headless --port 7778
```

For remote agents, run **Detect** before **Install**. Detection preserves an
existing custom install path and recognizes the legacy path if it is the only
installed copy. Windows scheduled tasks receive an explicit resolved
`--config-dir`, so SYSTEM does not accidentally use a different profile from
the SSH user. Existing CA/key material is reused; the stable `llama-monitor CA`
subject is intentionally not renamed.

## Rollback and cleanup

To roll back, stop the 2.0 process and relaunch the 1.x binary against the
legacy root. Do not delete the canonical root until you have verified that all
desired data was copied and that remote agents still authenticate. The app's
cleanup controls are explicit and receipt-backed; there is no automatic delete
of old roots, external model folders, certificates, or tokens.

## Support window

Legacy roots, tasks, process detection, and protocol identifiers remain
accepted through 2.x. This data/discovery compatibility does not require
building a second executable. Current releases ship only canonical assets,
and 2.0 clients continue to resolve those canonical assets.

## Troubleshooting

- **No models appear:** open Settings → Migration and inspect the root-state
  receipt; choose the legacy root or run the copy-first migration.
- **Remote agent is offline:** run Detect, verify the reported task command and
  path, then repair the managed task. Do not create a second SYSTEM task by
  hand.
- **Windows tray is missing:** install WebView2, restart Foundry, and validate
  the popover on native Windows. GNU cross-compilation cannot prove tray IPC.
- **Update says no compatible asset:** keep the 2.0.x release bridge in place
  and inspect `checksums.json`; all eight 2.0 asset names must be present.
