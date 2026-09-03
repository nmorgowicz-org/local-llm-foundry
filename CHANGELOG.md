# Llama Monitor Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.1](https://github.com/nmorgowicz-org/local-llm-foundry/compare/v2.1.0...v2.1.1) (2026-09-03)


### Bug Fixes

* **release:** align release workflow with legacy-asset drop policy ([#382](https://github.com/nmorgowicz-org/local-llm-foundry/issues/382)) ([509d66d](https://github.com/nmorgowicz-org/local-llm-foundry/commit/509d66d82b9819da719cd81c5a15549ebbb4ec95))

## [2.1.0](https://github.com/nmorgowicz-org/local-llm-foundry/compare/v2.0.13...v2.1.0) (2026-09-03)


### Features

* **calibration:** implement WddmTotalDeviceDelta nvidia-smi sampler ([44226a5](https://github.com/nmorgowicz-org/local-llm-foundry/commit/44226a50aab366a8506304363c23d2fe4711360a))
* **presets:** add bounded llama-fit-params probe ([44226a5](https://github.com/nmorgowicz-org/local-llm-foundry/commit/44226a50aab366a8506304363c23d2fe4711360a))
* **presets:** add evidence-details drawer action for bundle launch memory ([44226a5](https://github.com/nmorgowicz-org/local-llm-foundry/commit/44226a50aab366a8506304363c23d2fe4711360a))
* **presets:** add two-sided n_cpu_moe placement search ([44226a5](https://github.com/nmorgowicz-org/local-llm-foundry/commit/44226a50aab366a8506304363c23d2fe4711360a))
* **presets:** add typed launch bundles and reasoning flags ([44226a5](https://github.com/nmorgowicz-org/local-llm-foundry/commit/44226a50aab366a8506304363c23d2fe4711360a))
* **presets:** manage explicit model artifact bundles ([44226a5](https://github.com/nmorgowicz-org/local-llm-foundry/commit/44226a50aab366a8506304363c23d2fe4711360a))
* **presets:** render bundle presets as a single compact launch card ([44226a5](https://github.com/nmorgowicz-org/local-llm-foundry/commit/44226a50aab366a8506304363c23d2fe4711360a))
* **ui:** add preset bundle configure drawer ([44226a5](https://github.com/nmorgowicz-org/local-llm-foundry/commit/44226a50aab366a8506304363c23d2fe4711360a))
* **vram:** back launch estimates with fit probe ([44226a5](https://github.com/nmorgowicz-org/local-llm-foundry/commit/44226a50aab366a8506304363c23d2fe4711360a))
* **vram:** launch-evidence vocabulary, store, and resolve wiring ([44226a5](https://github.com/nmorgowicz-org/local-llm-foundry/commit/44226a50aab366a8506304363c23d2fe4711360a))
* **vram:** resolve explicit preset fit intents ([44226a5](https://github.com/nmorgowicz-org/local-llm-foundry/commit/44226a50aab366a8506304363c23d2fe4711360a))
* **wizard:** expose canonical llama runtime controls ([44226a5](https://github.com/nmorgowicz-org/local-llm-foundry/commit/44226a50aab366a8506304363c23d2fe4711360a))


### Bug Fixes

* **api:** don't commit preset mutations to memory when save fails ([44226a5](https://github.com/nmorgowicz-org/local-llm-foundry/commit/44226a50aab366a8506304363c23d2fe4711360a))
* **api:** run mixed K/V hard-gate check in Doctor independent of tool_enabled ([44226a5](https://github.com/nmorgowicz-org/local-llm-foundry/commit/44226a50aab366a8506304363c23d2fe4711360a))
* **api:** surface non-blocking K/V policy signal on vram-estimate ([44226a5](https://github.com/nmorgowicz-org/local-llm-foundry/commit/44226a50aab366a8506304363c23d2fe4711360a))
* **brand:** center the social-preview export correctly ([44226a5](https://github.com/nmorgowicz-org/local-llm-foundry/commit/44226a50aab366a8506304363c23d2fe4711360a))
* **calibration:** bound telemetry capture with a timeout ([44226a5](https://github.com/nmorgowicz-org/local-llm-foundry/commit/44226a50aab366a8506304363c23d2fe4711360a))
* **calibration:** enforce K/V launch policy gate at every calibration launch site ([44226a5](https://github.com/nmorgowicz-org/local-llm-foundry/commit/44226a50aab366a8506304363c23d2fe4711360a))
* **calibration:** fix flaky apply-validation fixture under parallel test load ([44226a5](https://github.com/nmorgowicz-org/local-llm-foundry/commit/44226a50aab366a8506304363c23d2fe4711360a))
* **calibration:** repeated-cycle agreement for metal_sampler ([44226a5](https://github.com/nmorgowicz-org/local-llm-foundry/commit/44226a50aab366a8506304363c23d2fe4711360a))
* **capabilities:** stop re-deriving help_hash from serve_flags at lookup ([44226a5](https://github.com/nmorgowicz-org/local-llm-foundry/commit/44226a50aab366a8506304363c23d2fe4711360a))
* **inference:** remove fail-closed gate on reasoning-preserve flag ([44226a5](https://github.com/nmorgowicz-org/local-llm-foundry/commit/44226a50aab366a8506304363c23d2fe4711360a))
* **presets:** close two Greptile-flagged bundle-launch races ([44226a5](https://github.com/nmorgowicz-org/local-llm-foundry/commit/44226a50aab366a8506304363c23d2fe4711360a))
* **presets:** enforce canonical llama launch validation ([44226a5](https://github.com/nmorgowicz-org/local-llm-foundry/commit/44226a50aab366a8506304363c23d2fe4711360a))
* **presets:** fix reachable resolve panic and extract shared cycle-agreement helper ([44226a5](https://github.com/nmorgowicz-org/local-llm-foundry/commit/44226a50aab366a8506304363c23d2fe4711360a))
* **presets:** remove dead fallback and close bundle.rs test gaps ([44226a5](https://github.com/nmorgowicz-org/local-llm-foundry/commit/44226a50aab366a8506304363c23d2fe4711360a))
* **vram:** enforce exact preset intent provenance ([44226a5](https://github.com/nmorgowicz-org/local-llm-foundry/commit/44226a50aab366a8506304363c23d2fe4711360a))
* **vram:** preserve context for low-vram intents ([44226a5](https://github.com/nmorgowicz-org/local-llm-foundry/commit/44226a50aab366a8506304363c23d2fe4711360a))
* **vram:** round layer-split fallback, handle nvidia-smi N/A, fix wired-limit tier gap ([44226a5](https://github.com/nmorgowicz-org/local-llm-foundry/commit/44226a50aab366a8506304363c23d2fe4711360a))

## [2.0.13](https://github.com/nmorgowicz-org/local-llm-foundry/compare/v2.0.12...v2.0.13) (2026-08-30)


### Bug Fixes

* **agent:** make remote agent upgrades idempotent ([#378](https://github.com/nmorgowicz-org/local-llm-foundry/issues/378)) ([ed46376](https://github.com/nmorgowicz-org/local-llm-foundry/commit/ed46376fbe501d1026a141ef4658b32e04a16813))

## [2.0.12](https://github.com/nmorgowicz-org/local-llm-foundry/compare/v2.0.11...v2.0.12) (2026-08-30)


### Bug Fixes

* **ui:** restore llama speculative telemetry ([#376](https://github.com/nmorgowicz-org/local-llm-foundry/issues/376)) ([737a517](https://github.com/nmorgowicz-org/local-llm-foundry/commit/737a51744193b650c3022abd96a559f74a19c96d))

## [2.0.11](https://github.com/nmorgowicz-org/local-llm-foundry/compare/v2.0.10...v2.0.11) (2026-08-29)


### Bug Fixes

* **agent:** repair Windows SSH token handoff ([#374](https://github.com/nmorgowicz-org/local-llm-foundry/issues/374)) ([ed91c55](https://github.com/nmorgowicz-org/local-llm-foundry/commit/ed91c559c94b7fcb30add3df0803cc0ad4ab01ac))

## [2.0.10](https://github.com/nmorgowicz-org/local-llm-foundry/compare/v2.0.9...v2.0.10) (2026-08-29)


### Bug Fixes

* **agent:** recover migrated remote metrics and release assets ([#372](https://github.com/nmorgowicz-org/local-llm-foundry/issues/372)) ([f8df927](https://github.com/nmorgowicz-org/local-llm-foundry/commit/f8df927c1836cc545d44d500e4111d15aad549fe))

## [2.0.9](https://github.com/nmorgowicz-org/local-llm-foundry/compare/v2.0.8...v2.0.9) (2026-08-29)


### Bug Fixes

* **agent:** recover remote metrics and scope engine indicator ([6cdc534](https://github.com/nmorgowicz-org/local-llm-foundry/commit/6cdc534a77a1ab128dce59b3978b436170fa441e))

## [2.0.8](https://github.com/nmorgowicz-org/local-llm-foundry/compare/v2.0.7...v2.0.8) (2026-08-29)


### Bug Fixes

* restore llama.cpp and remote-agent status ([5e92cf4](https://github.com/nmorgowicz-org/local-llm-foundry/commit/5e92cf43a8a4c2db8f24cb068018772a0dbf15c3))

## [2.0.7](https://github.com/nmorgowicz-org/local-llm-foundry/compare/v2.0.6...v2.0.7) (2026-08-28)


### Bug Fixes

* **ui:** reconcile remote agent state after repair ([#366](https://github.com/nmorgowicz-org/local-llm-foundry/issues/366)) ([f55a07b](https://github.com/nmorgowicz-org/local-llm-foundry/commit/f55a07b5cf0327a30c0514bf07381273d63c6222))

## [2.0.6](https://github.com/nmorgowicz-org/local-llm-foundry/compare/v2.0.5...v2.0.6) (2026-08-28)


### Bug Fixes

* **binary:** align managed paths and harden updates ([9e5c5ea](https://github.com/nmorgowicz-org/local-llm-foundry/commit/9e5c5ea0160e51c8ba8dd9205328b0f108f2fa42))
* **binary:** gate macOS rollback helper ([9e5c5ea](https://github.com/nmorgowicz-org/local-llm-foundry/commit/9e5c5ea0160e51c8ba8dd9205328b0f108f2fa42))

## [2.0.5](https://github.com/nmorgowicz-org/local-llm-foundry/compare/v2.0.4...v2.0.5) (2026-08-28)


### Bug Fixes

* **migration:** repair rebrand cutover and agent bootstrap ([51528a7](https://github.com/nmorgowicz-org/local-llm-foundry/commit/51528a7c8dd943b70050ca0b924283978c9da371))

## [2.0.4](https://github.com/nmorgowicz-org/local-llm-foundry/compare/v2.0.3...v2.0.4) (2026-08-22)


### Bug Fixes

* **binary:** remove dead Linux CUDA options and harden notifications ([b582a90](https://github.com/nmorgowicz-org/local-llm-foundry/commit/b582a90f4dcc62005c67c488caa78aff82092460))
* **ui:** harden notification menu modal layering ([b582a90](https://github.com/nmorgowicz-org/local-llm-foundry/commit/b582a90f4dcc62005c67c488caa78aff82092460))

## [2.0.3](https://github.com/nmorgowicz-org/local-llm-foundry/compare/v2.0.2...v2.0.3) (2026-08-21)


### Bug Fixes

* **rapid-mlx:** harden MTP sidecar repair and qualification ([ab3a3a6](https://github.com/nmorgowicz-org/local-llm-foundry/commit/ab3a3a6062d3d5f9d3b941da8390ef34bd11e862))
* **rapid-mlx:** manage pinned development runtimes ([ab3a3a6](https://github.com/nmorgowicz-org/local-llm-foundry/commit/ab3a3a6062d3d5f9d3b941da8390ef34bd11e862))

## [2.0.2](https://github.com/nmorgowicz-org/local-llm-foundry/compare/v2.0.1...v2.0.2) (2026-08-19)


### Bug Fixes

* **ci:** pin setup-node action correctly ([#355](https://github.com/nmorgowicz-org/local-llm-foundry/issues/355)) ([ff89d09](https://github.com/nmorgowicz-org/local-llm-foundry/commit/ff89d09913693150ff05c296b79b3841d501451f))

## [2.0.1](https://github.com/nmorgowicz-org/local-llm-foundry/compare/v2.0.0...v2.0.1) (2026-08-19)


### Bug Fixes

* **ci:** package WebView2 loader from Cargo registry ([#350](https://github.com/nmorgowicz-org/local-llm-foundry/issues/350)) ([6338b65](https://github.com/nmorgowicz-org/local-llm-foundry/commit/6338b65cc0ee9a600724ab52b11089b26647f778))

## [2.0.0](https://github.com/nmorgowicz-org/local-llm-foundry/compare/v1.8.1...v2.0.0) (2026-08-19)


### ⚠ BREAKING CHANGES

* launch Local LLM Foundry 2.0 identity and Token Ingot brand

### Features

* **agent:** preserve mixed-version updater, remote install, task, and process compatibility ([e0ea7b8](https://github.com/nmorgowicz-org/local-llm-foundry/commit/e0ea7b81ea03e0c9c8915eab11b5894a7ca1af78))
* **binary:** add canonical package and thin legacy CLI compatibility entrypoint ([e0ea7b8](https://github.com/nmorgowicz-org/local-llm-foundry/commit/e0ea7b81ea03e0c9c8915eab11b5894a7ca1af78))
* **hf:** add backend-aware Hugging Face discovery and model lineage ([e0ea7b8](https://github.com/nmorgowicz-org/local-llm-foundry/commit/e0ea7b81ea03e0c9c8915eab11b5894a7ca1af78))
* launch Local LLM Foundry 2.0 identity and Token Ingot brand ([e0ea7b8](https://github.com/nmorgowicz-org/local-llm-foundry/commit/e0ea7b81ea03e0c9c8915eab11b5894a7ca1af78))
* **migration:** add copy-first resumable application-home migration ([e0ea7b8](https://github.com/nmorgowicz-org/local-llm-foundry/commit/e0ea7b81ea03e0c9c8915eab11b5894a7ca1af78))
* **models:** add explicit receipt-backed managed model relocation ([e0ea7b8](https://github.com/nmorgowicz-org/local-llm-foundry/commit/e0ea7b81ea03e0c9c8915eab11b5894a7ca1af78))
* **rapid-mlx:** add first-class MLX runtime, capabilities, and profiles ([e0ea7b8](https://github.com/nmorgowicz-org/local-llm-foundry/commit/e0ea7b81ea03e0c9c8915eab11b5894a7ca1af78))
* **rapid-mlx:** make inference orchestration backend-neutral ([e0ea7b8](https://github.com/nmorgowicz-org/local-llm-foundry/commit/e0ea7b81ea03e0c9c8915eab11b5894a7ca1af78))
* **rapid-mlx:** unify presets, spawn wizard, chat, and launch contracts ([e0ea7b8](https://github.com/nmorgowicz-org/local-llm-foundry/commit/e0ea7b81ea03e0c9c8915eab11b5894a7ca1af78))
* **release:** publish canonical and legacy 2.0.x assets with exact checksums ([e0ea7b8](https://github.com/nmorgowicz-org/local-llm-foundry/commit/e0ea7b81ea03e0c9c8915eab11b5894a7ca1af78))
* **ui:** align wizard, preset editor, chat, dashboard, and model surfaces ([e0ea7b8](https://github.com/nmorgowicz-org/local-llm-foundry/commit/e0ea7b81ea03e0c9c8915eab11b5894a7ca1af78))
* **ui:** ship Token Ingot shell, manifests, themes, and migration UX ([e0ea7b8](https://github.com/nmorgowicz-org/local-llm-foundry/commit/e0ea7b81ea03e0c9c8915eab11b5894a7ca1af78))
* **vram:** extend estimator and metadata introspection across backends ([e0ea7b8](https://github.com/nmorgowicz-org/local-llm-foundry/commit/e0ea7b81ea03e0c9c8915eab11b5894a7ca1af78))


### Bug Fixes

* **deps:** update rust crate base64 to 0.23 ([#326](https://github.com/nmorgowicz-org/local-llm-foundry/issues/326)) ([efbe35e](https://github.com/nmorgowicz-org/local-llm-foundry/commit/efbe35e5e5e7aaf357b42598ea05a3d0570e84fa))
* **deps:** update rust crate hf-hub to v1 ([#307](https://github.com/nmorgowicz-org/local-llm-foundry/issues/307)) ([451ce28](https://github.com/nmorgowicz-org/local-llm-foundry/commit/451ce280040f0747684c15eda4fb79a815bdd489))
* **deps:** update rust crate wry to 0.56 ([#333](https://github.com/nmorgowicz-org/local-llm-foundry/issues/333)) ([57f8c09](https://github.com/nmorgowicz-org/local-llm-foundry/commit/57f8c093539ff55583d46e789458dba53b8997f8))
* **security:** preserve CA/encryption identity and harden temporary token entropy ([e0ea7b8](https://github.com/nmorgowicz-org/local-llm-foundry/commit/e0ea7b81ea03e0c9c8915eab11b5894a7ca1af78))

## [1.8.1](https://github.com/nmorgowicz-org/llama-monitor/compare/v1.8.0...v1.8.1) (2026-07-07)


### Bug Fixes

* **ui:** add 4-state card fit logic (fit / tight / conditional / no-fit) ([61d2f4c](https://github.com/nmorgowicz-org/llama-monitor/commit/61d2f4c208d3ae326c2626e1d52f856dda15a78c))
* **ui:** add macOS "Free cache" button on welcome screen ([61d2f4c](https://github.com/nmorgowicz-org/llama-monitor/commit/61d2f4c208d3ae326c2626e1d52f856dda15a78c))
* **ui:** clarify inference memory bar and add segmented memory UI ([61d2f4c](https://github.com/nmorgowicz-org/llama-monitor/commit/61d2f4c208d3ae326c2626e1d52f856dda15a78c))

## [1.8.0](https://github.com/nmorgowicz-org/llama-monitor/compare/v1.7.0...v1.8.0) (2026-07-07)


### Features

* **wizard:** add chat template auto-updater (Qwen/Gemma/future) ([3339657](https://github.com/nmorgowicz-org/llama-monitor/commit/3339657d65fb3998f1c8210de0e47e76411d4b5a))


### Bug Fixes

* **wizard:** add clear feedback for “Re-fetch Recommended” ([3339657](https://github.com/nmorgowicz-org/llama-monitor/commit/3339657d65fb3998f1c8210de0e47e76411d4b5a))
* **wizard:** clean up chat template “Check for updates” UX ([3339657](https://github.com/nmorgowicz-org/llama-monitor/commit/3339657d65fb3998f1c8210de0e47e76411d4b5a))
* **wizard:** hide community picks panel when data is missing ([3339657](https://github.com/nmorgowicz-org/llama-monitor/commit/3339657d65fb3998f1c8210de0e47e76411d4b5a))

## [1.7.0](https://github.com/nmorgowicz-org/llama-monitor/compare/v1.6.0...v1.7.0) (2026-07-06)


### Features

* **models:** add HF VRAM pre-download estimate, chat template lifecycle tracking, tool_call_format, and download fixes ([9b2408b](https://github.com/nmorgowicz-org/llama-monitor/commit/9b2408b2dfe1afc5c636e27dc9ed2c4845d45943))


### Bug Fixes

* **deps:** update rust crate aes-gcm to 0.11 ([#291](https://github.com/nmorgowicz-org/llama-monitor/issues/291)) ([25b1dba](https://github.com/nmorgowicz-org/llama-monitor/commit/25b1dba7bd37b78a6349828118f2d396aafd0159))
* **download:** fix stuck downloads, resume, cooldown, idle timeout, and lock safety ([9b2408b](https://github.com/nmorgowicz-org/llama-monitor/commit/9b2408b2dfe1afc5c636e27dc9ed2c4845d45943))

## [1.6.0](https://github.com/nmorgowicz-org/llama-monitor/compare/v1.5.3...v1.6.0) (2026-06-27)


### Features

* **agent:** diagnose failed Windows start; trim health-check retries ([b4ae87e](https://github.com/nmorgowicz-org/llama-monitor/commit/b4ae87e961eaf945927ed431382a19397eead4c5))
* **nav:** self-host vendor scripts via node_modules ([a199888](https://github.com/nmorgowicz-org/llama-monitor/commit/a1998884b682567d7baf3fb1f359093a58327ee0))
* **nav:** SPA routing with History API and deep-linking ([a199888](https://github.com/nmorgowicz-org/llama-monitor/commit/a1998884b682567d7baf3fb1f359093a58327ee0))


### Bug Fixes

* **agent:** ship WebView2Loader.dll on remote Windows agent install ([b4ae87e](https://github.com/nmorgowicz-org/llama-monitor/commit/b4ae87e961eaf945927ed431382a19397eead4c5))
* **deps:** update dependency marked to v18 ([#287](https://github.com/nmorgowicz-org/llama-monitor/issues/287)) ([4624937](https://github.com/nmorgowicz-org/llama-monitor/commit/4624937627ec867261f7d5b9ce5c2f32d3dfa925))
* **nav:** centralize history writes, harden CSP, remove CDN dependency ([a199888](https://github.com/nmorgowicz-org/llama-monitor/commit/a1998884b682567d7baf3fb1f359093a58327ee0))
* **nav:** return to welcome screen on fresh session after hard refresh ([b4ae87e](https://github.com/nmorgowicz-org/llama-monitor/commit/b4ae87e961eaf945927ed431382a19397eead4c5))
* **ui:** expose vendored highlight.js in browser ([b4ae87e](https://github.com/nmorgowicz-org/llama-monitor/commit/b4ae87e961eaf945927ed431382a19397eead4c5))
* **ui:** expose vendored highlight.js in browser ([d3d9d43](https://github.com/nmorgowicz-org/llama-monitor/commit/d3d9d4306cb9e6a52bf0c363143decf58a93f9e1))

## [1.5.3](https://github.com/nmorgowicz-org/llama-monitor/compare/v1.5.2...v1.5.3) (2026-06-25)


### Bug Fixes

* **spawn:** clear preset name input on wizard close ([44d48e5](https://github.com/nmorgowicz-org/llama-monitor/commit/44d48e568fc81c14e469ac4fd018799efa9fa1c5))
* **ui:** move preset card VRAM label below memory bar ([44d48e5](https://github.com/nmorgowicz-org/llama-monitor/commit/44d48e568fc81c14e469ac4fd018799efa9fa1c5))
* **ui:** show separate VRAM and RAM bars with available-budget awareness ([44d48e5](https://github.com/nmorgowicz-org/llama-monitor/commit/44d48e568fc81c14e469ac4fd018799efa9fa1c5))
* **vram:** correct Hybrid DeltaNet test comment and document active_params_b behavior ([44d48e5](https://github.com/nmorgowicz-org/llama-monitor/commit/44d48e568fc81c14e469ac4fd018799efa9fa1c5))
* **vram:** improve card VRAM bar math and Hybrid DeltaNet active-param estimates ([44d48e5](https://github.com/nmorgowicz-org/llama-monitor/commit/44d48e568fc81c14e469ac4fd018799efa9fa1c5))
* **vram:** sum VRAM totals across multiple GPUs ([44d48e5](https://github.com/nmorgowicz-org/llama-monitor/commit/44d48e568fc81c14e469ac4fd018799efa9fa1c5))
* **vram:** use exact GGUF tensor counts for active params, add partial GPU-layer and RAM-budget support ([44d48e5](https://github.com/nmorgowicz-org/llama-monitor/commit/44d48e568fc81c14e469ac4fd018799efa9fa1c5))

## [1.5.2](https://github.com/nmorgowicz-org/llama-monitor/compare/v1.5.1...v1.5.2) (2026-06-24)


### Bug Fixes

* **gguf:** improve architecture_kind, family, and preset backfill ([1d61931](https://github.com/nmorgowicz-org/llama-monitor/commit/1d619312eda9e387e5f01c0f306d63938398f256))
* **presets:** fix auto-backfill so Dense/MoE/Hybrid MoE labels appear on preset cards ([1d61931](https://github.com/nmorgowicz-org/llama-monitor/commit/1d619312eda9e387e5f01c0f306d63938398f256))
* **presets:** relax GGUF metadata check so arch labels auto-backfill ([1d61931](https://github.com/nmorgowicz-org/llama-monitor/commit/1d619312eda9e387e5f01c0f306d63938398f256))
* **ui:** improve architecture labels in setup view ([1d61931](https://github.com/nmorgowicz-org/llama-monitor/commit/1d619312eda9e387e5f01c0f306d63938398f256))

## [1.5.1](https://github.com/nmorgowicz-org/llama-monitor/compare/v1.5.0...v1.5.1) (2026-06-24)


### Bug Fixes

* **gguf:** restore all PR [#272](https://github.com/nmorgowicz-org/llama-monitor/issues/272) changes lost during revert/resquash ([156c222](https://github.com/nmorgowicz-org/llama-monitor/commit/156c222549c720a2532aac6b9401e31d9d0f88c3))
* **presets:** backfill GGUF metadata on load and add clear_gguf_metadata ([156c222](https://github.com/nmorgowicz-org/llama-monitor/commit/156c222549c720a2532aac6b9401e31d9d0f88c3))
* **presets:** include expert_bytes_per_layer in completeness guard, backfill comparison, and test assertions ([156c222](https://github.com/nmorgowicz-org/llama-monitor/commit/156c222549c720a2532aac6b9401e31d9d0f88c3))
* **ui:** wire GGUF arch and layer data into editor and spawn wizard ([156c222](https://github.com/nmorgowicz-org/llama-monitor/commit/156c222549c720a2532aac6b9401e31d9d0f88c3))
* **vram:** use measured expert bytes for --n-cpu-moe split ([156c222](https://github.com/nmorgowicz-org/llama-monitor/commit/156c222549c720a2532aac6b9401e31d9d0f88c3))

## [1.5.0](https://github.com/nmorgowicz-org/llama-monitor/compare/v1.4.0...v1.5.0) (2026-06-24)


### Features

* **gguf:** measure exact per-layer tensor sizes for -ngl/--n-cpu-moe tuning ([74485c9](https://github.com/nmorgowicz-org/llama-monitor/commit/74485c95b60042b83734e257a9b93a4c090e39a0))
* **gguf:** measure exact per-layer tensor sizes for -ngl/--n-cpu-moe tuning ([d464f81](https://github.com/nmorgowicz-org/llama-monitor/commit/d464f812da09b454120d40c9b8c193d94fe365af))
* **ui:** add Dense/MoE/Hybrid MoE architecture labels to launch cards, editor, and spawn wizard ([dba7cea](https://github.com/nmorgowicz-org/llama-monitor/commit/dba7ceae84e1b69cd23884bfcf7776cddbe1f879))
* **ui:** wire backend GGUF arch + layer data into editor and spawn wizard ([74485c9](https://github.com/nmorgowicz-org/llama-monitor/commit/74485c95b60042b83734e257a9b93a4c090e39a0))
* **ui:** wire backend GGUF arch + layer data into editor and spawn wizard ([d464f81](https://github.com/nmorgowicz-org/llama-monitor/commit/d464f812da09b454120d40c9b8c193d94fe365af))
* **vram:** use measured per-layer expert bytes for --n-cpu-moe split ([74485c9](https://github.com/nmorgowicz-org/llama-monitor/commit/74485c95b60042b83734e257a9b93a4c090e39a0))
* **vram:** use measured per-layer expert bytes for --n-cpu-moe split ([d464f81](https://github.com/nmorgowicz-org/llama-monitor/commit/d464f812da09b454120d40c9b8c193d94fe365af))


### Bug Fixes

* **deps:** update rust crate winreg to 0.56 ([#271](https://github.com/nmorgowicz-org/llama-monitor/issues/271)) ([9188c1e](https://github.com/nmorgowicz-org/llama-monitor/commit/9188c1e56d130a7035ee07139d817303b6a4fa67))
* **presets:** backfill GGUF architecture metadata on load; centralize reset ([74485c9](https://github.com/nmorgowicz-org/llama-monitor/commit/74485c95b60042b83734e257a9b93a4c090e39a0))
* **presets:** backfill GGUF architecture metadata on load; centralize reset ([d464f81](https://github.com/nmorgowicz-org/llama-monitor/commit/d464f812da09b454120d40c9b8c193d94fe365af))
* **ui:** fix label accessibility, prioritize prio guidance, align spawn wizard context options ([dba7cea](https://github.com/nmorgowicz-org/llama-monitor/commit/dba7ceae84e1b69cd23884bfcf7776cddbe1f879))
* **ui:** improve MoE active-params estimate, unify arch label logic, fix minor styling ([dba7cea](https://github.com/nmorgowicz-org/llama-monitor/commit/dba7ceae84e1b69cd23884bfcf7776cddbe1f879))
* **ui:** prevent context pills from submitting the preset form ([dba7cea](https://github.com/nmorgowicz-org/llama-monitor/commit/dba7ceae84e1b69cd23884bfcf7776cddbe1f879))
* **ui:** restrict n-cpu-moe to MoE models and add unified-memory hint ([dba7cea](https://github.com/nmorgowicz-org/llama-monitor/commit/dba7ceae84e1b69cd23884bfcf7776cddbe1f879))


### Reverts

* feat(gguf) ([#272](https://github.com/nmorgowicz-org/llama-monitor/issues/272)) ([#273](https://github.com/nmorgowicz-org/llama-monitor/issues/273)) ([93482ee](https://github.com/nmorgowicz-org/llama-monitor/commit/93482ee8509017346051edbc0ab3ed22cc1f36fe))

## [1.4.0](https://github.com/nmorgowicz-org/llama-monitor/compare/v1.3.1...v1.4.0) (2026-06-23)


### Features

* **gpu:** integrate PawnIO driver install and expose status in sensor bridge ([41b72e5](https://github.com/nmorgowicz-org/llama-monitor/commit/41b72e58c1c22e618c3c4082268e55d11daa8455))
* **tray:** add unified right-click menu, popover fix, and WebView2 auto-install ([41b72e5](https://github.com/nmorgowicz-org/llama-monitor/commit/41b72e58c1c22e618c3c4082268e55d11daa8455))
* **windows:** show console by default on Windows and log to file when headless ([41b72e5](https://github.com/nmorgowicz-org/llama-monitor/commit/41b72e58c1c22e618c3c4082268e55d11daa8455))
* **windows:** use APPDATA as default config dir and add legacy migration ([41b72e5](https://github.com/nmorgowicz-org/llama-monitor/commit/41b72e58c1c22e618c3c4082268e55d11daa8455))


### Bug Fixes

* **gpu:** apply no_window to GPU tools, LHM, and llama-server subprocesses ([41b72e5](https://github.com/nmorgowicz-org/llama-monitor/commit/41b72e58c1c22e618c3c4082268e55d11daa8455))
* **gpu:** correct WMI GPU VRAM wrap at 4 GiB using registry value ([41b72e5](https://github.com/nmorgowicz-org/llama-monitor/commit/41b72e58c1c22e618c3c4082268e55d11daa8455))
* **windows:** skip legacy migration when --config-dir is set and harden PawnIO check ([41b72e5](https://github.com/nmorgowicz-org/llama-monitor/commit/41b72e58c1c22e618c3c4082268e55d11daa8455))
* **windows:** update all release files in-place during self-update ([68c36b1](https://github.com/nmorgowicz-org/llama-monitor/commit/68c36b12ef652cb9253220e4988ee4866fa926e8))

## [1.3.1](https://github.com/nmorgowicz-org/llama-monitor/compare/v1.3.0...v1.3.1) (2026-06-22)


### Bug Fixes

* address CodeQL security findings ([#262](https://github.com/nmorgowicz-org/llama-monitor/issues/262)) ([51565fa](https://github.com/nmorgowicz-org/llama-monitor/commit/51565fa375b7d942ac8d6985a043c0e421b5fb58))

## [1.3.0](https://github.com/nmorgowicz-org/llama-monitor/compare/v1.2.0...v1.3.0) (2026-06-22)


### Features

* **system:** add cross-platform memory-pressure telemetry and purge hardening ([12a1398](https://github.com/nmorgowicz-org/llama-monitor/commit/12a1398169cddd53b2522734eada350e350ee72f))
* **system:** add cross-platform memory-pressure telemetry, kernel-anchored macOS detection, purge endpoint, and normalized 0-100 score bands ([4615343](https://github.com/nmorgowicz-org/llama-monitor/commit/4615343c6359cb9c58dc54737a6694f1abc4c28d))


### Bug Fixes

* **presets:** Apple Silicon mmap hint + clearer, actionable mlock warning ([12a1398](https://github.com/nmorgowicz-org/llama-monitor/commit/12a1398169cddd53b2522734eada350e350ee72f))
* **presets:** Apple Silicon mmap hint and clearer mlock warning ([4615343](https://github.com/nmorgowicz-org/llama-monitor/commit/4615343c6359cb9c58dc54737a6694f1abc4c28d))
* **self-update:** harden cross-platform restart, downloads, and cleanup ([4615343](https://github.com/nmorgowicz-org/llama-monitor/commit/4615343c6359cb9c58dc54737a6694f1abc4c28d))
* **self-update:** harden cross-platform restart, downloads, and cleanup ([0bd4d14](https://github.com/nmorgowicz-org/llama-monitor/commit/0bd4d145c0cff28dbb478f769d5bf5f5c75260d6))
* **system:** anchor macOS memory pressure on kernel verdict; fix Windows available-memory and swap ([4615343](https://github.com/nmorgowicz-org/llama-monitor/commit/4615343c6359cb9c58dc54737a6694f1abc4c28d))
* **wizard:** default mmap on (no_mmap=false) on Apple Silicon ([4615343](https://github.com/nmorgowicz-org/llama-monitor/commit/4615343c6359cb9c58dc54737a6694f1abc4c28d))
* **wizard:** default mmap on (no_mmap=false) on Apple Silicon ([12a1398](https://github.com/nmorgowicz-org/llama-monitor/commit/12a1398169cddd53b2522734eada350e350ee72f))

## [1.2.0](https://github.com/nmorgowicz-org/llama-monitor/compare/v1.1.0...v1.2.0) (2026-06-22)


### Features

* **api:** auto-populate preset GGUF metadata on create/update; refresh on model_path change ([c25d42a](https://github.com/nmorgowicz-org/llama-monitor/commit/c25d42abce15f3aa23d6b4ccd7c8ad4a80f68561))
* **api:** centralized ApiCtx, ApiError, auth helpers in common.rs ([bbb83ca](https://github.com/nmorgowicz-org/llama-monitor/commit/bbb83ca6bc523dc11c93ffccffebb0d6871ad2f7))
* **api:** ncpumoe tune endpoint and callers pass is_unified_memory for platform-correct MoE fit ([c25d42a](https://github.com/nmorgowicz-org/llama-monitor/commit/c25d42abce15f3aa23d6b4ccd7c8ad4a80f68561))
* **binary:** add release checksums.json and verify on self-update; Unix/macOS restart launcher ([c25d42a](https://github.com/nmorgowicz-org/llama-monitor/commit/c25d42abce15f3aa23d6b4ccd7c8ad4a80f68561))
* **chat:** map chat routes to new modular backend layout ([bbb83ca](https://github.com/nmorgowicz-org/llama-monitor/commit/bbb83ca6bc523dc11c93ffccffebb0d6871ad2f7))
* **gpu:** use Apple Silicon gpu_power instead of SoC total_power ([bbb83ca](https://github.com/nmorgowicz-org/llama-monitor/commit/bbb83ca6bc523dc11c93ffccffebb0d6871ad2f7))
* **models:** classified user-friendly download error messages ([bbb83ca](https://github.com/nmorgowicz-org/llama-monitor/commit/bbb83ca6bc523dc11c93ffccffebb0d6871ad2f7))
* **models:** concurrency limits and duplicate guard for model downloads ([bbb83ca](https://github.com/nmorgowicz-org/llama-monitor/commit/bbb83ca6bc523dc11c93ffccffebb0d6871ad2f7))
* **models:** derive model-card family and size from GGUF metadata; remove name-guessing ([c25d42a](https://github.com/nmorgowicz-org/llama-monitor/commit/c25d42abce15f3aa23d6b4ccd7c8ad4a80f68561))
* **models:** enhance GGUF filename parsing for IQ/UD-Q/F16/BF16/F32 ([bbb83ca](https://github.com/nmorgowicz-org/llama-monitor/commit/bbb83ca6bc523dc11c93ffccffebb0d6871ad2f7))
* **models:** stable size-based VRAM estimate in model scan ([bbb83ca](https://github.com/nmorgowicz-org/llama-monitor/commit/bbb83ca6bc523dc11c93ffccffebb0d6871ad2f7))
* **models:** tighter size-aware MTP/draft model classification ([bbb83ca](https://github.com/nmorgowicz-org/llama-monitor/commit/bbb83ca6bc523dc11c93ffccffebb0d6871ad2f7))
* **sessions:** simplify sessions.js; move flows to attach/spawn modules ([bbb83ca](https://github.com/nmorgowicz-org/llama-monitor/commit/bbb83ca6bc523dc11c93ffccffebb0d6871ad2f7))
* **sleep:** 3-level sleep/low-power mode and adaptive WebSocket polling ([bbb83ca](https://github.com/nmorgowicz-org/llama-monitor/commit/bbb83ca6bc523dc11c93ffccffebb0d6871ad2f7))
* **spawn-wizard:** recognize qwen35 arch and use embedding_length ([bbb83ca](https://github.com/nmorgowicz-org/llama-monitor/commit/bbb83ca6bc523dc11c93ffccffebb0d6871ad2f7))
* **spawn:** VRAM bar, scenario cards, and summary now use debounced /api/vram-estimate (single backend source) ([c25d42a](https://github.com/nmorgowicz-org/llama-monitor/commit/c25d42abce15f3aa23d6b4ccd7c8ad4a80f68561))
* **system:** improved CPU core topology and cluster naming ([bbb83ca](https://github.com/nmorgowicz-org/llama-monitor/commit/bbb83ca6bc523dc11c93ffccffebb0d6871ad2f7))
* **system:** macOS memory-pressure telemetry and purge support ([bbb83ca](https://github.com/nmorgowicz-org/llama-monitor/commit/bbb83ca6bc523dc11c93ffccffebb0d6871ad2f7))
* **system:** new top-processes and purge endpoints ([bbb83ca](https://github.com/nmorgowicz-org/llama-monitor/commit/bbb83ca6bc523dc11c93ffccffebb0d6871ad2f7))
* **ui:** deepened card gradients and refined surface design ([bbb83ca](https://github.com/nmorgowicz-org/llama-monitor/commit/bbb83ca6bc523dc11c93ffccffebb0d6871ad2f7))
* **ui:** in-app confirm/prompt dialogs across chat, sessions, DB admin, LHM, presets ([bbb83ca](https://github.com/nmorgowicz-org/llama-monitor/commit/bbb83ca6bc523dc11c93ffccffebb0d6871ad2f7))
* **ui:** memory-pressure card and nav chip with hovercard ([bbb83ca](https://github.com/nmorgowicz-org/llama-monitor/commit/bbb83ca6bc523dc11c93ffccffebb0d6871ad2f7))
* **ui:** mlock warnings in preset editor and spawn wizard when VRAM tight ([bbb83ca](https://github.com/nmorgowicz-org/llama-monitor/commit/bbb83ca6bc523dc11c93ffccffebb0d6871ad2f7))
* **ui:** model library toolbar and VRAM display improvements ([bbb83ca](https://github.com/nmorgowicz-org/llama-monitor/commit/bbb83ca6bc523dc11c93ffccffebb0d6871ad2f7))
* **ui:** new global tooltip system replacing native tooltips ([bbb83ca](https://github.com/nmorgowicz-org/llama-monitor/commit/bbb83ca6bc523dc11c93ffccffebb0d6871ad2f7))
* **update:** self-update restart uses config ports instead of hardcoded 7778/7779 ([c25d42a](https://github.com/nmorgowicz-org/llama-monitor/commit/c25d42abce15f3aa23d6b4ccd7c8ad4a80f68561))
* **vram:** include n_embd and GGUF embedding_length in VRAM estimator ([bbb83ca](https://github.com/nmorgowicz-org/llama-monitor/commit/bbb83ca6bc523dc11c93ffccffebb0d6871ad2f7))
* **vram:** new CUDA compute buffer for discrete GPUs; Metal exclusion ([bbb83ca](https://github.com/nmorgowicz-org/llama-monitor/commit/bbb83ca6bc523dc11c93ffccffebb0d6871ad2f7))
* **vram:** remove client-side VRAM formula; all bars use /api/vram-estimate with pre-download GGUF header range-fetch from HF ([c25d42a](https://github.com/nmorgowicz-org/llama-monitor/commit/c25d42abce15f3aa23d6b4ccd7c8ad4a80f68561))
* **vram:** replace flat Metal overhead with M5 Max-calibrated model (per-layer base + ~6.5% KV scaling) ([c25d42a](https://github.com/nmorgowicz-org/llama-monitor/commit/c25d42abce15f3aa23d6b4ccd7c8ad4a80f68561))
* **vram:** unify VRAM estimation to backend; new discrete-GPU overhead model calibrated on RTX 5090; GGUF-driven arch as single source of truth ([c25d42a](https://github.com/nmorgowicz-org/llama-monitor/commit/c25d42abce15f3aa23d6b4ccd7c8ad4a80f68561))


### Bug Fixes

* **api:** /api/vram/auto-size and /api/models/gguf-meta prefer GGUF-derived arch when model_path exists ([c25d42a](https://github.com/nmorgowicz-org/llama-monitor/commit/c25d42abce15f3aa23d6b4ccd7c8ad4a80f68561))
* **binary:** harden Unix/macOS restart launcher with crash-loop guard, port-check, and spawn logging ([c25d42a](https://github.com/nmorgowicz-org/llama-monitor/commit/c25d42abce15f3aa23d6b4ccd7c8ad4a80f68561))
* **db:** restrict PRAGMA allowlist to block dangerous queries ([bbb83ca](https://github.com/nmorgowicz-org/llama-monitor/commit/bbb83ca6bc523dc11c93ffccffebb0d6871ad2f7))
* **system:** correct s-core handling and cluster labels in system metrics ([bbb83ca](https://github.com/nmorgowicz-org/llama-monitor/commit/bbb83ca6bc523dc11c93ffccffebb0d6871ad2f7))
* **tray:** allow web fonts in compact popover CSP ([bbb83ca](https://github.com/nmorgowicz-org/llama-monitor/commit/bbb83ca6bc523dc11c93ffccffebb0d6871ad2f7))
* **tray:** theme macOS popover with shared dark/light + palette and fix metric bar rendering ([bbb83ca](https://github.com/nmorgowicz-org/llama-monitor/commit/bbb83ca6bc523dc11c93ffccffebb0d6871ad2f7))
* **ui:** reduce visual noise in sparklines and card hover states ([bbb83ca](https://github.com/nmorgowicz-org/llama-monitor/commit/bbb83ca6bc523dc11c93ffccffebb0d6871ad2f7))
* **vram:** correct Qwen3 Coder Next DeltaNet recurrent-state size from ~1.3 GB to ~38 MB; expose hybrid-attention and sliding-window fields from GGUF ([c25d42a](https://github.com/nmorgowicz-org/llama-monitor/commit/c25d42abce15f3aa23d6b4ccd7c8ad4a80f68561))
* **vram:** use discrete overhead model in auto-size MoE weight fit and quant advisor "fits" check for CUDA/ROCm ([c25d42a](https://github.com/nmorgowicz-org/llama-monitor/commit/c25d42abce15f3aa23d6b4ccd7c8ad4a80f68561))

## [1.1.0](https://github.com/nmorgowicz-org/llama-monitor/compare/v1.0.1...v1.1.0) (2026-06-19)


### Features

* **ui:** refine hardware metric sparklines ([09c9a05](https://github.com/nmorgowicz-org/llama-monitor/commit/09c9a0546931c59cba76b126d7588a49c96e45a3))


### Bug Fixes

* **binary:** refresh updater status after install ([09c9a05](https://github.com/nmorgowicz-org/llama-monitor/commit/09c9a0546931c59cba76b126d7588a49c96e45a3))
* **binary:** refresh updater status after install ([40e8eca](https://github.com/nmorgowicz-org/llama-monitor/commit/40e8eca4e17aa40b031009c1e514437d220a1526))
* **gpu:** separate Apple GPU and SoC power ([09c9a05](https://github.com/nmorgowicz-org/llama-monitor/commit/09c9a0546931c59cba76b126d7588a49c96e45a3))
* **gpu:** separate Apple GPU and SoC power ([40e8eca](https://github.com/nmorgowicz-org/llama-monitor/commit/40e8eca4e17aa40b031009c1e514437d220a1526))
* **ui:** inset hardware metric sparklines ([09c9a05](https://github.com/nmorgowicz-org/llama-monitor/commit/09c9a0546931c59cba76b126d7588a49c96e45a3))
* **ui:** inset hardware metric sparklines ([40e8eca](https://github.com/nmorgowicz-org/llama-monitor/commit/40e8eca4e17aa40b031009c1e514437d220a1526))

## [1.0.1](https://github.com/nmorgowicz-org/llama-monitor/compare/v1.0.0...v1.0.1) (2026-06-19)


### Bug Fixes

* **deps:** update rust crate zip to v8 ([#239](https://github.com/nmorgowicz-org/llama-monitor/issues/239)) ([07a82fc](https://github.com/nmorgowicz-org/llama-monitor/commit/07a82fc98418ba638107abd3936d65f5d7818bb2))

## [1.0.0](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.23.0...v1.0.0) (2026-06-19)


### Features

* **wizard:** five-step Spawn Wizard (Profile → Model → Hardware & Tuning → Summary → Spawn) with guided, VRAM-aware defaults and hardware presets
* **wizard:** Profile segmented control (Quick/Balanced/Advanced) and use-case selection persisted in localStorage
* **wizard:** integrated HuggingFace search/browse/download into the spawn flow—browse, pick, and launch from one wizard
* **wizard:** auto-resolve HF origin for local models with model card panel and markdown-rendered details
* **wizard:** quant swap step with VRAM-aware advisor, provider presets, and in-place quant changes without recreating presets
* **wizard:** Apple Silicon P/E core detection, unified memory budget model, and Metal GPU limit control
* **wizard:** animated VRAM breakdown panel with MoE expert offload slider and per-component tooltips
* **wizard:** speculative decoding, MTP/assistant model support, and draft-model auto-detection for Gemma4/Qwen3.6
* **wizard:** chat template auto-install, cache_ram_mib, and full preset field wiring to spawn
* **wizard:** thinking/reasoning controls, Qwen3.5/3.6 reasoning presets, and reasoning budget options
* **wizard:** context quick-picks up to 256k with free manual input and minimum-context warnings
* **wizard:** server alias and extra args in Summary; modernized footer and Close button
* **wizard:** auto-advance with toast after HF download completes; quick-edit chips open preset modal on context
* **vram:** architecture-aware VRAM estimator rewrite with direct GGUF metadata read, no subprocess needed
* **vram:** DeltaNet/hybrid-attention support for Qwen3.6, Qwen3.5, Qwen3-Coder-Next, davidau 40B
* **vram:** Gemma 3/4 sliding-window and Gemma 4 global head_dim modeling for accurate KV cache estimates
* **vram:** MoE support with VRAM/RAM split based on n_cpu_moe, expert_fraction, and MoE auto-tune suggestions
* **vram:** EXAONE 4.5 family support including hybrid attention, MTP depth, and vision encoder sizing
* **vram:** VRAM fit legend, inline context-size warnings, and "approx." labeling on launch cards
* **spawn:** unified V2 spawn architecture using -m or -hf model source with validation and clear errors
* **spawn:** pre-spawn health check (run llama-server --help) with captured stderr; fail fast on bad binary
* **spawn:** race-safe restart-after-update with server_stopping flag, MTP port-sweep cleanup, and improved timeouts
* **spawn:** always-on flags for long conversations: --no-context-shift, --ctx-checkpoints, --keep now default
* **spawn:** bind host (loopback vs 0.0.0.0) and per-session API key configuration for safer local use
* **spawn:** echo llama-server logs to terminal and attach last lines on spawn failure for clearer diagnostics
* **hf:** full HuggingFace integration with keyword search, sort (trending/downloads/likes/newest), and load-more
* **hf:** discover-pills for major families (Qwen3, Llama3, Mistral/MoE, Gemma, EXAONE, etc.) and per-author browsing
* **hf:** GGUF file listing with real sizes, mmproj/draft-model detection, and provider/quant-type labels
* **hf:** streaming downloads with resume, ETA, cancel, and partial-file rename safety
* **hf:** HF token integration (store/show/remove) for gated repos; token never echoed in responses
* **models:** model-library view with card/list modes, search, filters (quant, size, tag, mmproj, split, draft), and persisted prefs
* **models:** library tag system: attach/click-remove tags per model; filter by tags (coding, roleplay, etc.)
* **models:** per-model VRAM bar and estimate, plus related-presets linking on each model card
* **ui:** welcome screen redesign: preset launch grid, recent-launch timestamps, unified memory bar, and drop-zone
* **ui:** control bar redesign with two-zone layout and context-aware buttons (Running→Stop+Switch / Stopped→Start+Spawn)
* **ui:** post-stop choice modal: "Go to Welcome" or "Stay on Dashboard"
* **ui:** logs panel overhaul with token coloring, font size controls, incremental rendering, and live log tail viewer
* **ui:** teal accent retheme across hardware cards, inference widgets, logs, and navigation
* **ui:** full light theme pass with contrast-safe colors; reduced-motion queries disable transitions for accessibility
* **chat:** multi-select in conversations sidebar with bulk Delete/Archive and Undo toasts
* **chat:** streaming render throttled to ≤20 fps to prevent main-thread freezes on long responses
* **chat:** new chat preferences (date format, Enter-to-send, context view, persist-thinking-content, etc.)
* **chat:** raised new-tab max_tokens default from 8192 to 32768 for longer conversations
* **tuning:** new tuning panel with performance benchmark (A/B/C/D grades) and auto-tuning suggestions
* **tuning:** MTP n-max sweep with bar chart and one-click apply; MoE auto-tune and depth sweep via llama-bench
* **perf:** 80%+ battery-drain reduction via Page Visibility API, 70+ paused CSS animations, and transform-based animations
* **perf:** Low Power Mode with configurable auto-sleep, slower pollers, and reduced WebSocket broadcast in idle/battery-saver
* **api:** restore 27 spawn-wizard handler functions to prevent mass 404s on wizard/HF/tuning endpoints
* **api:** new endpoints for MTP-draft check, HF token, streaming download, VRAM breakdown, chat-templates, and session readiness
* **security:** path-traversal guards for HF downloads, model browse, and resolve-origin with unit tests
* **security:** multi-statement SQL rejection in DB query endpoint; tightened body-size and repo_id validations
* **security:** metrics and GPU/system endpoints now require api_token; Host header spoofing protection
* **security:** login brute-force resistance: cooldown increased from 2s to 10s


### Bug Fixes

* **wizard:** correct -ngl 99 to -ngl all; ensure wizard maps "All" to -1 safely across llama.cpp versions
* **spawn:** enable flash attention by default for eligible models; preset editor reflects new standard
* **spawn:** remove 30s cooldown from kill-llama; unify stop/kill for faster session cycling and safer restart loops
* **spawn:** prevent locally spawned servers from leaking into remote endpoints list
* **vram:** correct Gemma4 and Qwen DeltaNet heuristics (n_ctx_train integration, expert semantics) for accurate estimates
* **security:** fix token leak and header-spoofing issue in bootstrap-token and setup checks

## [0.23.0](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.22.5...v0.23.0) (2026-05-27)


### Features

* **api:** add optional tab_id scoping to FTS search endpoint ([8d635bc](https://github.com/nmorgowicz-org/llama-monitor/commit/8d635bc5ca3c6cc143c2bc5b3273149dbb90c5bf))
* **chat:** copy button, tab sync, stopped state, and Escape dismiss on History Q&A panel ([8d635bc](https://github.com/nmorgowicz-org/llama-monitor/commit/8d635bc5ca3c6cc143c2bc5b3273149dbb90c5bf))
* **chat:** dynamic context limits scaling from live context_capacity_tokens metrics ([8d635bc](https://github.com/nmorgowicz-org/llama-monitor/commit/8d635bc5ca3c6cc143c2bc5b3273149dbb90c5bf))
* **chat:** History Q&A panel with multi-turn Q&A, AI keyword search, context injection, and history insertion ([8d635bc](https://github.com/nmorgowicz-org/llama-monitor/commit/8d635bc5ca3c6cc143c2bc5b3273149dbb90c5bf))


### Bug Fixes

* **test:** eliminate temp-path collisions in concurrent cargo test runs ([8d635bc](https://github.com/nmorgowicz-org/llama-monitor/commit/8d635bc5ca3c6cc143c2bc5b3273149dbb90c5bf))

## [0.22.5](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.22.4...v0.22.5) (2026-05-27)


### Bug Fixes

* **deps:** update rust crate rusqlite to 0.40 ([#205](https://github.com/nmorgowicz-org/llama-monitor/issues/205)) ([7edf345](https://github.com/nmorgowicz-org/llama-monitor/commit/7edf34571c3e15d37fbf3e621b60cfef40abbeb2))

## [0.22.4](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.22.3...v0.22.4) (2026-05-26)


### Bug Fixes

* **chat:** restore density-class updates in focus mode ([c48922b](https://github.com/nmorgowicz-org/llama-monitor/commit/c48922bd9e69eb52b7472e58ffc849267952d766))

## [0.22.3](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.22.2...v0.22.3) (2026-05-26)


### Bug Fixes

* **agent:** add startup grace period to suppress health-check noise during install ([10f64b7](https://github.com/nmorgowicz-org/llama-monitor/commit/10f64b7266ab5fc6f86b776d6758c5d9d059397b))
* **chat:** sort sidebar tabs by last message timestamp instead of updated_at ([10f64b7](https://github.com/nmorgowicz-org/llama-monitor/commit/10f64b7266ab5fc6f86b776d6758c5d9d059397b))
* **chat:** stop backfillCtxPct from silently bumping updated_at on tab persist ([10f64b7](https://github.com/nmorgowicz-org/llama-monitor/commit/10f64b7266ab5fc6f86b776d6758c5d9d059397b))
* **ui:** add dashboard edge padding and prevent horizontal overflow ([10f64b7](https://github.com/nmorgowicz-org/llama-monitor/commit/10f64b7266ab5fc6f86b776d6758c5d9d059397b))
* **ui:** hide redundant grade chip when remote agent is fully connected ([10f64b7](https://github.com/nmorgowicz-org/llama-monitor/commit/10f64b7266ab5fc6f86b776d6758c5d9d059397b))
* **ui:** restore endpoint-status hover popover with bridge pseudo-element ([10f64b7](https://github.com/nmorgowicz-org/llama-monitor/commit/10f64b7266ab5fc6f86b776d6758c5d9d059397b))

## [0.22.2](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.22.1...v0.22.2) (2026-05-25)


### Bug Fixes

* **agent:** automatic SSH bootstrap enrollment — on mTLS failure, existing SSH credentials fetch the remote CA and api-token, enroll the device, and save the token with no user interaction required ([609b2d1](https://github.com/nmorgowicz-org/llama-monitor/commit/609b2d15793df658cda56e89ed405247e2301cbe))
* **agent:** multi-client mTLS — each device enrolls its own CA; cas/ entries hot-reload without restarting the agent ([609b2d1](https://github.com/nmorgowicz-org/llama-monitor/commit/609b2d15793df658cda56e89ed405247e2301cbe))

## [0.22.1](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.22.0...v0.22.1) (2026-05-25)


### Bug Fixes

* **agent:** fix mTLS role check to search raw DER bytes instead of base64 output ([e3aab16](https://github.com/nmorgowicz-org/llama-monitor/commit/e3aab1600a0b7e40a1e5b6fb5e68756f5c2533a6))
* **agent:** switch SCP to SFTP for Windows-compatible remote file transfer ([e3aab16](https://github.com/nmorgowicz-org/llama-monitor/commit/e3aab1600a0b7e40a1e5b6fb5e68756f5c2533a6))
* **agent:** use ~/.config path for certs on all platforms including macOS ([e3aab16](https://github.com/nmorgowicz-org/llama-monitor/commit/e3aab1600a0b7e40a1e5b6fb5e68756f5c2533a6))
* **agent:** use distinct CA CN to resolve TLS chain ambiguity on all clients ([e3aab16](https://github.com/nmorgowicz-org/llama-monitor/commit/e3aab1600a0b7e40a1e5b6fb5e68756f5c2533a6))

## [0.22.0](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.21.4...v0.22.0) (2026-05-24)


### Features

* **attach:** add optional API key support for remote endpoint authentication ([381f46b](https://github.com/nmorgowicz-org/llama-monitor/commit/381f46b73f221bfcc2dad02d5bdd5407d314c79b))


### Bug Fixes

* **attach:** fix broken detach button with proper headers and error handling ([381f46b](https://github.com/nmorgowicz-org/llama-monitor/commit/381f46b73f221bfcc2dad02d5bdd5407d314c79b))
* **test:** fix flaky command palette, tab cycling, pinned tabs, and profile menu tests ([381f46b](https://github.com/nmorgowicz-org/llama-monitor/commit/381f46b73f221bfcc2dad02d5bdd5407d314c79b))
* **ui:** set data-pinned attribute on pinned sidebar items ([381f46b](https://github.com/nmorgowicz-org/llama-monitor/commit/381f46b73f221bfcc2dad02d5bdd5407d314c79b))

## [0.21.4](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.21.3...v0.21.4) (2026-05-24)


### Bug Fixes

* **agent:** fix 4 bugs found in Windows remote install audit ([19c562c](https://github.com/nmorgowicz-org/llama-monitor/commit/19c562c0908c655c701066f2c692e244410e0240))
* **agent:** use resolved path for Windows cas/ mkdir and provision certs on Start ([19c562c](https://github.com/nmorgowicz-org/llama-monitor/commit/19c562c0908c655c701066f2c692e244410e0240))

## [0.21.3](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.21.2...v0.21.3) (2026-05-23)


### Bug Fixes

* **agent:** install ring CryptoProvider before rustls TLS setup to prevent agent panic on startup ([ca808d3](https://github.com/nmorgowicz-org/llama-monitor/commit/ca808d3f9b81626fbd5657920987d56202bf9cd8))
* **agent:** resolve %APPDATA% before SCP so CA cert and server cert are actually written to Windows ([ca808d3](https://github.com/nmorgowicz-org/llama-monitor/commit/ca808d3f9b81626fbd5657920987d56202bf9cd8))
* **chat:** sort sidebar chats by recency within each group ([abe8da2](https://github.com/nmorgowicz-org/llama-monitor/commit/abe8da28ecc46d823b8d5af70efeb10b144d563e))
* **chat:** use local-time day boundaries for Today/Yesterday/This Week grouping ([abe8da2](https://github.com/nmorgowicz-org/llama-monitor/commit/abe8da28ecc46d823b8d5af70efeb10b144d563e))
* **deps:** migrate to rand_core 0.10, rand 0.10, and argon2 0.6-rc ([#174](https://github.com/nmorgowicz-org/llama-monitor/issues/174)) ([f4c91fa](https://github.com/nmorgowicz-org/llama-monitor/commit/f4c91fa40d3f9370bbb8b00f1fe4243c87e311e8))
* **deps:** update rust crate sysinfo to 0.39 ([#175](https://github.com/nmorgowicz-org/llama-monitor/issues/175)) ([0b011e8](https://github.com/nmorgowicz-org/llama-monitor/commit/0b011e8d7a2124bbe5e400a8255f0ef8e125eb1d))
* **ui:** use local-time day boundaries in backup filenames and relative-time displays ([abe8da2](https://github.com/nmorgowicz-org/llama-monitor/commit/abe8da28ecc46d823b8d5af70efeb10b144d563e))

## [0.21.2](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.21.1...v0.21.2) (2026-05-22)


### Bug Fixes

* **agent:** agent cannot start after mTLS hardening — CA installed next to binary not found in certs_dir ([e7d9f7c](https://github.com/nmorgowicz-org/llama-monitor/commit/e7d9f7c42033a73c18deb10b04af39b9bb9cf1db))
* **agent:** health check loop terminates at 10/20 attempts due to per-request timeout and client rebuild overhead ([e7d9f7c](https://github.com/nmorgowicz-org/llama-monitor/commit/e7d9f7c42033a73c18deb10b04af39b9bb9cf1db))
* **agent:** LATEST version shows "Unavailable" when setup modal reopened within 30 seconds ([e7d9f7c](https://github.com/nmorgowicz-org/llama-monitor/commit/e7d9f7c42033a73c18deb10b04af39b9bb9cf1db))
* **agent:** wrong binary path (/tmp/llama-monitor) on Windows — OS detection ran before SSH connection was hydrated ([e7d9f7c](https://github.com/nmorgowicz-org/llama-monitor/commit/e7d9f7c42033a73c18deb10b04af39b9bb9cf1db))
* **ui:** shimmer animation artifact renders as rectangle left of model name in MODEL & DECODING card ([e7d9f7c](https://github.com/nmorgowicz-org/llama-monitor/commit/e7d9f7c42033a73c18deb10b04af39b9bb9cf1db))

## [0.21.1](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.21.0...v0.21.1) (2026-05-22)


### Bug Fixes

* **agent:** fix HTTPS client builder error preventing remote agent connections after reqwest 0.13 upgrade ([7203abf](https://github.com/nmorgowicz-org/llama-monitor/commit/7203abf6b0f8960e1729d99fa6dfb64a5d56d3cc))

## [0.21.0](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.20.0...v0.21.0) (2026-05-22)


### Features

* **chat:** add next reply plan summary showing active steering inputs ([72950e3](https://github.com/nmorgowicz-org/llama-monitor/commit/72950e3b25b3023854753b51b10390a9e86b6a78))
* **chat:** add workspace command palette (Ctrl+K omnibox) with conversation search and quick actions ([72950e3](https://github.com/nmorgowicz-org/llama-monitor/commit/72950e3b25b3023854753b51b10390a9e86b6a78))
* **chat:** persist composer drafts per-tab with save/restore/clear lifecycle ([72950e3](https://github.com/nmorgowicz-org/llama-monitor/commit/72950e3b25b3023854753b51b10390a9e86b6a78))
* **chat:** track persona version per tab with template_version_or_hash for safe drift detection ([72950e3](https://github.com/nmorgowicz-org/llama-monitor/commit/72950e3b25b3023854753b51b10390a9e86b6a78))
* **chat:** use backend message_count for inactive tabs in sidebar and command palette ([72950e3](https://github.com/nmorgowicz-org/llama-monitor/commit/72950e3b25b3023854753b51b10390a9e86b6a78))
* **sessions:** add recent-endpoints dashboard to setup screen with one-click reconnect ([72950e3](https://github.com/nmorgowicz-org/llama-monitor/commit/72950e3b25b3023854753b51b10390a9e86b6a78))
* **sessions:** extend Session model with connect metadata and GET /api/sessions/recent endpoint ([72950e3](https://github.com/nmorgowicz-org/llama-monitor/commit/72950e3b25b3023854753b51b10390a9e86b6a78))
* **ui:** add actionable empty-state copy for GPU and system cards with connecting state ([72950e3](https://github.com/nmorgowicz-org/llama-monitor/commit/72950e3b25b3023854753b51b10390a9e86b6a78))
* **ui:** add unified telemetry grade with 9-state derivation and grade chip on agent badge ([72950e3](https://github.com/nmorgowicz-org/llama-monitor/commit/72950e3b25b3023854753b51b10390a9e86b6a78))
* **ui:** expose protocol_too_old and protocol_version on WebSocket payload ([72950e3](https://github.com/nmorgowicz-org/llama-monitor/commit/72950e3b25b3023854753b51b10390a9e86b6a78))


### Bug Fixes

* **chat:** context notes badge now shows correct per-tab count instead of leaking between tabs ([72950e3](https://github.com/nmorgowicz-org/llama-monitor/commit/72950e3b25b3023854753b51b10390a9e86b6a78))
* **chat:** persist ai_gender in DB so gender pill survives reload ([72950e3](https://github.com/nmorgowicz-org/llama-monitor/commit/72950e3b25b3023854753b51b10390a9e86b6a78))
* **remote-agent:** restore managed agent upgrade and start flows after HTTPS mTLS hardening ([72950e3](https://github.com/nmorgowicz-org/llama-monitor/commit/72950e3b25b3023854753b51b10390a9e86b6a78))
* **settings:** centralize guided-generation settings in backend-backed settingsState ([72950e3](https://github.com/nmorgowicz-org/llama-monitor/commit/72950e3b25b3023854753b51b10390a9e86b6a78))
* **settings:** promote shared workflow prefs (enter_to_send, date_format, continuity) to shared storage ([72950e3](https://github.com/nmorgowicz-org/llama-monitor/commit/72950e3b25b3023854753b51b10390a9e86b6a78))
* **settings:** remove dead runtime controls from persistent settings panes ([72950e3](https://github.com/nmorgowicz-org/llama-monitor/commit/72950e3b25b3023854753b51b10390a9e86b6a78))
* **ui:** correct remote_agent_health_reachable always-equal-to-connected bug ([72950e3](https://github.com/nmorgowicz-org/llama-monitor/commit/72950e3b25b3023854753b51b10390a9e86b6a78))
* **ui:** correct telemetry metric grid layout to avoid forced full-width span ([72950e3](https://github.com/nmorgowicz-org/llama-monitor/commit/72950e3b25b3023854753b51b10390a9e86b6a78))

## [0.20.0](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.19.1...v0.20.0) (2026-05-20)


### Features

* **metrics:** add tokens_per_decode and per-request draft acceptance to dashboard and telemetry ([6303ea3](https://github.com/nmorgowicz-org/llama-monitor/commit/6303ea3a47ef4c1c1dc4c14044cdd2c0dccb2279))


### Bug Fixes

* **chat:** preserve scroll position when loading older messages ([6303ea3](https://github.com/nmorgowicz-org/llama-monitor/commit/6303ea3a47ef4c1c1dc4c14044cdd2c0dccb2279))
* **metrics:** fix speculative.types field name in metrics parsing ([6303ea3](https://github.com/nmorgowicz-org/llama-monitor/commit/6303ea3a47ef4c1c1dc4c14044cdd2c0dccb2279))
* **notes:** fix AI review panel scroll and spelling ([6303ea3](https://github.com/nmorgowicz-org/llama-monitor/commit/6303ea3a47ef4c1c1dc4c14044cdd2c0dccb2279))

## [0.19.1](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.19.0...v0.19.1) (2026-05-20)


### Bug Fixes

* **chat:** extend busy-wait timeout to 5 minutes for long inference tasks ([bb11a2e](https://github.com/nmorgowicz-org/llama-monitor/commit/bb11a2e6c4c0778b4f6b2f43ab033b8733c678e4))
* **chat:** restore hidden chat access via sidebar pill click ([bb11a2e](https://github.com/nmorgowicz-org/llama-monitor/commit/bb11a2e6c4c0778b4f6b2f43ab033b8733c678e4))
* **ui:** add active state visual feedback to management pills ([bb11a2e](https://github.com/nmorgowicz-org/llama-monitor/commit/bb11a2e6c4c0778b4f6b2f43ab033b8733c678e4))
* **ui:** correct config bar button collapse thresholds and enhance snug tier ([bb11a2e](https://github.com/nmorgowicz-org/llama-monitor/commit/bb11a2e6c4c0778b4f6b2f43ab033b8733c678e4))

## [0.19.0](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.18.0...v0.19.0) (2026-05-20)


### Features

* add graceful shutdown with WAL checkpoint and final session save ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* add hourly WAL checkpoint and auto backups for database maintenance ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **api:** add DB admin token auth and restricted query allowlist for /api/db/query ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **api:** secure DB backup/restore endpoints with token auth and path validation ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **auth:** add --form-auth and --clear-auth-config CLI flags for dashboard auth ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **auth:** add auth-config.json persistence and migration from startup flags ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **auth:** add Basic Auth / form login / both modes and per-route auth guard ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **auth:** add dashboard form login with HttpOnly session cookie and auth shell ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **auth:** add Security tab controls to enable/disable dashboard auth and change password ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **chat:** add chat archive and hidden visibility states with sidebar management pills ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **chat:** add Chat Sessions sidebar with recency grouping, pinning, avatars, and context menus ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **chat:** add full-text search (FTS) across all chat messages with sidebar UI and live results ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **chat:** add hidden surface with deliberate reveal flow for privacy ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **chat:** add Hide Now button in chat header for quick conversation hiding ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **chat:** add inline archive surface with restore/hide/delete actions and undo toasts ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **chat:** add lazy loading of tab messages and per-tab incremental persistence ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **chat:** scope message search to active tabs by default with Active/Archived filter chips ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **cli:** add TLS-related CLI flags (--tls, --tls-cert, --tls-key, --tls-self-signed) ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **gpu:** WMI-based GPU discovery for Intel and unknown GPUs on Windows ([497db06](https://github.com/nmorgowicz-org/llama-monitor/commit/497db06f71216acc399fed96e1e715b9497889c8))
* **security:** add ACME DNS-01 integration with auto-renewal and Certificates management in Settings ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **security:** add API token auth for sensitive endpoints (DB backup/restore, query) ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **security:** add api-token auth to previously unprotected endpoints (settings, presets, templates, LHM, sensor-bridge) ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **security:** add CSP nonce for inline scripts and DOMPurify sanitization to prevent XSS in chat ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **security:** add global Origin validation to mitigate CSRF on cookie-authenticated endpoints ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **security:** add in-place token rotation for agent, API, and DB admin tokens ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **security:** add mTLS for remote-agent with role-based trust ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **security:** add per-endpoint cooldowns for expensive operations (remote-agent, sessions, file browser, chat-search, DB, ACME) ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **security:** add remote-agent command validation to block dangerous commands ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **security:** add TLS support with self-signed, custom cert, and ACME (Let's Encrypt) modes ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **security:** add visibility-aware list and search endpoints with query param filtering ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **security:** enforce api-token auth on all chat and chat-search endpoints ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **security:** enforce api-token auth on all remote-agent endpoints ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **security:** enforce api-token auth on archive, hide, and restore endpoints ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **security:** enforce api-token auth on session CRUD and attach/detach endpoints ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **security:** enforce db-admin-token auth on remote-agent install and remove ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **security:** enforce db-admin-token auth on session delete and spawn ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **security:** mask hidden chat names from collapsed sidebar label ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **security:** mask sensitive tokens and SSH credentials in settings with show/hide toggle ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **tray:** Windows WebView popover — replaces static context menu with live metrics ([497db06](https://github.com/nmorgowicz-org/llama-monitor/commit/497db06f71216acc399fed96e1e715b9497889c8))
* **ui:** add Database Administration modal with maintenance, backups, indexes, repair, and SQL query ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **ui:** add guided-generation prompt templates and category management in Settings ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **ui:** add keyboard shortcuts panel and improved chat input buttons ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **ui:** polish Settings modal layout, sections, and controls for chat, appearance, and guided generation ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **ui:** redesign file browser modal with premium UX, parent folder button, and config browse button ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))


### Bug Fixes

* **chat:** serialize monitor inference requests and return explicit busy/offline errors when the active llama-server is occupied ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **chat:** stabilize AI response waiting to prevent "[stopped]" captures in guided-generation flows ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* resolve generic-array version conflict after dependabot updates ([50a1147](https://github.com/nmorgowicz-org/llama-monitor/commit/50a114786c22f04f7afd27744673e8f4d4eff639))
* **security:** restrict /api/browse to allowed root paths to prevent path traversal ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **ui:** resolve file browser hint clipping and stabilize related e2e tests ([db9000e](https://github.com/nmorgowicz-org/llama-monitor/commit/db9000eaa99673d009adb1a1e7d0cfb973007a08))
* **windows:** Windows ACL hardening for secret files via icacls ([497db06](https://github.com/nmorgowicz-org/llama-monitor/commit/497db06f71216acc399fed96e1e715b9497889c8))

## [0.18.0](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.17.1...v0.18.0) (2026-05-14)


### Features

* **chat:** add {{gender}} token support for dynamic system prompts ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **chat:** add advanced features (suggestion history, fix last response) ([3a78049](https://github.com/nmorgowicz-org/llama-monitor/commit/3a780495acc9a69eaceaca0964a92a0203a9dac0))
* **chat:** add adversarial prompt engineering to Coder explicit L2 policy ([3a78049](https://github.com/nmorgowicz-org/llama-monitor/commit/3a780495acc9a69eaceaca0964a92a0203a9dac0))
* **chat:** add compact confirmation modal with stats and editable summary preview ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **chat:** add context notes AI analysis endpoint and UI ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **chat:** add custom categories in suggestions dropdown ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **chat:** add custom role boundary override in behavior panel ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **chat:** add debug prompt inspector with system prompt breakdown and token counts ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **chat:** add Director Mode category and update build doc ([3a78049](https://github.com/nmorgowicz-org/llama-monitor/commit/3a780495acc9a69eaceaca0964a92a0203a9dac0))
* **chat:** add guided generation features (context notes, suggestions, quick guide) ([3a78049](https://github.com/nmorgowicz-org/llama-monitor/commit/3a780495acc9a69eaceaca0964a92a0203a9dac0))
* **chat:** add per-persona explicit policies with independent Level 1/Level 2 text ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **chat:** add persistent disconnected banner on connection loss ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **chat:** add pharmacology, harm reduction, and drug policy to explicit L2 ([3a78049](https://github.com/nmorgowicz-org/llama-monitor/commit/3a780495acc9a69eaceaca0964a92a0203a9dac0))
* **chat:** add reset to default button for built-in personas ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **chat:** add template list sections (Active/Custom/Built-in) with active badge ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **chat:** enhance explicit L2 policies with research findings ([3a78049](https://github.com/nmorgowicz-org/llama-monitor/commit/3a780495acc9a69eaceaca0964a92a0203a9dac0))
* **chat:** explicit mode v2 — persona-aware multi-level system ([3a78049](https://github.com/nmorgowicz-org/llama-monitor/commit/3a780495acc9a69eaceaca0964a92a0203a9dac0))
* **chat:** implement Pathweaver prompts and features ([3a78049](https://github.com/nmorgowicz-org/llama-monitor/commit/3a780495acc9a69eaceaca0964a92a0203a9dac0))
* **chat:** Phase 8 tag cloud, explicit v2 tests, and drug policy enhancements ([3a78049](https://github.com/nmorgowicz-org/llama-monitor/commit/3a780495acc9a69eaceaca0964a92a0203a9dac0))
* **chat:** rewrite send direction to inject suggestion directly as user message ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **docs:** add docs/README.md index, update README.md, chat.md, api.md, dashboard.md ([3a78049](https://github.com/nmorgowicz-org/llama-monitor/commit/3a780495acc9a69eaceaca0964a92a0203a9dac0))
* **test:** add quick-guide-revise.spec.js for Revise Last button tests ([3a78049](https://github.com/nmorgowicz-org/llama-monitor/commit/3a780495acc9a69eaceaca0964a92a0203a9dac0))
* **ui:** add focus keywords input with auto-generate button ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **ui:** add global Escape key handler for topmost modal ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **ui:** improve chat UI/UX — toasts, suggestions, quick guide, context notes ([3a78049](https://github.com/nmorgowicz-org/llama-monitor/commit/3a780495acc9a69eaceaca0964a92a0203a9dac0))


### Bug Fixes

* **api:** standardize on snake_case for token fields to avoid duplicate field errors ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **chat:** add alias for active_template_id to accept snake_case from frontend ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **chat:** add auto_compact_summarize and compact_mode fields to ChatTab ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **chat:** add periodic save every 30s to prevent data loss on force-kill ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **chat:** add settings controls for guided generation features ([3a78049](https://github.com/nmorgowicz-org/llama-monitor/commit/3a780495acc9a69eaceaca0964a92a0203a9dac0))
* **chat:** correct import paths and function names in guided generation modules ([3a78049](https://github.com/nmorgowicz-org/llama-monitor/commit/3a780495acc9a69eaceaca0964a92a0203a9dac0))
* **chat:** delay modal close until view transition completes ([3a78049](https://github.com/nmorgowicz-org/llama-monitor/commit/3a780495acc9a69eaceaca0964a92a0203a9dac0))
* **chat:** fix 5 clippy errors (collapsible_str_replace, invalid_regex, redundant_closure) ([3a78049](https://github.com/nmorgowicz-org/llama-monitor/commit/3a780495acc9a69eaceaca0964a92a0203a9dac0))
* **chat:** fix e2e tests for suggestions send mode, settings success class, tag cloud UI refresh ([3a78049](https://github.com/nmorgowicz-org/llama-monitor/commit/3a780495acc9a69eaceaca0964a92a0203a9dac0))
* **chat:** fix reset button visibility and always show per-persona explicit policies ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **chat:** handle streaming SSE response for auto-generate focus keywords ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **chat:** harden explicit mode toggle against null/undefined values ([3a78049](https://github.com/nmorgowicz-org/llama-monitor/commit/3a780495acc9a69eaceaca0964a92a0203a9dac0))
* **chat:** improve chat message coloring and load older messages UX ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **chat:** include explicit_policies in merged built-in templates ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **chat:** move reset button to right panel next to edit and apply ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **chat:** persist explicit mode level on tabs ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **chat:** re-render tab bar after explicit mode toggle ([3a78049](https://github.com/nmorgowicz-org/llama-monitor/commit/3a780495acc9a69eaceaca0964a92a0203a9dac0))
* **chat:** remove 'Format each as:...' from custom category prompt ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **chat:** remove recent suggestions section and fix send direction rewrite ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **chat:** restore guided generation buttons and fix clipped dropdowns ([3a78049](https://github.com/nmorgowicz-org/llama-monitor/commit/3a780495acc9a69eaceaca0964a92a0203a9dac0))
* **chat:** restore guided generation flow and unify screenshot capture ([3a78049](https://github.com/nmorgowicz-org/llama-monitor/commit/3a780495acc9a69eaceaca0964a92a0203a9dac0))
* **chat:** restore manage categories popup layout (built-in left, custom right) ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **chat:** set auto_compact_summarize to true by default for all chats ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **chat:** stabilize guided reply and rolling memory flows ([3a78049](https://github.com/nmorgowicz-org/llama-monitor/commit/3a780495acc9a69eaceaca0964a92a0203a9dac0))
* **chat:** standardize on camelCase for explicitLevel and activeTemplateId ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **chat:** update persona menu name on tab load and switch ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **chat:** use /api/chat/suggestions endpoint for auto-generate focus keywords ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **chat:** use correct thinking disable params for auto-generate focus call ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **dashboard:** cap slot grid height to prevent card blowout at high parallelism ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **ui:** add CSS variable aliases and remove duplicate selectors ([3a78049](https://github.com/nmorgowicz-org/llama-monitor/commit/3a780495acc9a69eaceaca0964a92a0203a9dac0))
* **ui:** fill card height and differentiate context window views ([3a78049](https://github.com/nmorgowicz-org/llama-monitor/commit/3a780495acc9a69eaceaca0964a92a0203a9dac0))
* **ui:** fix context notes sidebar, suggestions/quick guide buttons ([3a78049](https://github.com/nmorgowicz-org/llama-monitor/commit/3a780495acc9a69eaceaca0964a92a0203a9dac0))
* **ui:** fix context notes sidebar, tab badge, suggestions prompt ([3a78049](https://github.com/nmorgowicz-org/llama-monitor/commit/3a780495acc9a69eaceaca0964a92a0203a9dac0))
* **ui:** improve custom categories sizing and styling ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **ui:** make custom and built-in lists share height equally (50/50 split) ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **ui:** redesign context window card gauge and fleet views ([3a78049](https://github.com/nmorgowicz-org/llama-monitor/commit/3a780495acc9a69eaceaca0964a92a0203a9dac0))
* **ui:** reorganize manage categories modal with proper two-column layout ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))
* **ui:** wrap custom category buttons in chips container for proper layout ([3576cc3](https://github.com/nmorgowicz-org/llama-monitor/commit/3576cc35f60806e1d98d22cfd39b74cf700f691e))

## [0.17.1](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.17.0...v0.17.1) (2026-05-08)


### Bug Fixes

* **agent:** preserve quick upgrade button in update indicator ([47768f1](https://github.com/nmorgowicz-org/llama-monitor/commit/47768f1375eb93eb49bcb1e2507d83fff348be48))
* **chat:** add real-time token counter to thinking block header ([49c9898](https://github.com/nmorgowicz-org/llama-monitor/commit/49c98981a2d47e9601bead370d79eee55bbcb7a5))
* **chat:** add resend button to user messages for quick retry ([49c9898](https://github.com/nmorgowicz-org/llama-monitor/commit/49c98981a2d47e9601bead370d79eee55bbcb7a5))
* **chat:** allow inline edit save during AI generation ([9ad98c2](https://github.com/nmorgowicz-org/llama-monitor/commit/9ad98c2454d1c33f577feff3ed7aeaa480682e83))
* **chat:** allow regeneration from last variant in navigation ([9ad98c2](https://github.com/nmorgowicz-org/llama-monitor/commit/9ad98c2454d1c33f577feff3ed7aeaa480682e83))
* **chat:** disable auto-scroll when user scrolls up during generation ([49c9898](https://github.com/nmorgowicz-org/llama-monitor/commit/49c98981a2d47e9601bead370d79eee55bbcb7a5))
* **chat:** ensure auto-scroll resumes after sending message ([9ad98c2](https://github.com/nmorgowicz-org/llama-monitor/commit/9ad98c2454d1c33f577feff3ed7aeaa480682e83))
* **chat:** move timeout setting to main panel, improve toast duration ([49c9898](https://github.com/nmorgowicz-org/llama-monitor/commit/49c98981a2d47e9601bead370d79eee55bbcb7a5))
* **chat:** persist input resize to settings instead of localStorage ([9ad98c2](https://github.com/nmorgowicz-org/llama-monitor/commit/9ad98c2454d1c33f577feff3ed7aeaa480682e83))
* **chat:** prevent scroll force and DOM wipe during streaming ([9ad98c2](https://github.com/nmorgowicz-org/llama-monitor/commit/9ad98c2454d1c33f577feff3ed7aeaa480682e83))
* **chat:** reorder persona menu with active at top, add edit buttons ([49c9898](https://github.com/nmorgowicz-org/llama-monitor/commit/49c98981a2d47e9601bead370d79eee55bbcb7a5))
* **chat:** restore user position after reconnecting to server ([49c9898](https://github.com/nmorgowicz-org/llama-monitor/commit/49c98981a2d47e9601bead370d79eee55bbcb7a5))
* **chat:** scroll to thinking block when it appears during generation ([49c9898](https://github.com/nmorgowicz-org/llama-monitor/commit/49c98981a2d47e9601bead370d79eee55bbcb7a5))
* **chat:** show connection lost modal on all errors, not just regenerate ([49c9898](https://github.com/nmorgowicz-org/llama-monitor/commit/49c98981a2d47e9601bead370d79eee55bbcb7a5))
* **chat:** update erotic storyteller system prompt ([49c9898](https://github.com/nmorgowicz-org/llama-monitor/commit/49c98981a2d47e9601bead370d79eee55bbcb7a5))
* **ui:** fix connection lost modal spacing ([49c9898](https://github.com/nmorgowicz-org/llama-monitor/commit/49c98981a2d47e9601bead370d79eee55bbcb7a5))
* **ui:** remove infinite animations from settings modal fields ([47768f1](https://github.com/nmorgowicz-org/llama-monitor/commit/47768f1375eb93eb49bcb1e2507d83fff348be48))
* **ui:** restore min-height to allow content expansion ([9ad98c2](https://github.com/nmorgowicz-org/llama-monitor/commit/9ad98c2454d1c33f577feff3ed7aeaa480682e83))

## [0.17.0](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.16.0...v0.17.0) (2026-05-07)


### Features

* **agent:** add quick upgrade button to remote agent update indicator ([27618bf](https://github.com/nmorgowicz-org/llama-monitor/commit/27618bf3ad70bc0355446ca4efd9c7e25548b6db))
* **ui:** add breathing glow to chat telemetry trigger button ([27618bf](https://github.com/nmorgowicz-org/llama-monitor/commit/27618bf3ad70bc0355446ca4efd9c7e25548b6db))


### Bug Fixes

* **agent:** normalize version comparison to strip v prefix from GitHub tag ([27618bf](https://github.com/nmorgowicz-org/llama-monitor/commit/27618bf3ad70bc0355446ca4efd9c7e25548b6db))
* **ui:** pause infinite animations on hover instead of stopping them ([27618bf](https://github.com/nmorgowicz-org/llama-monitor/commit/27618bf3ad70bc0355446ca4efd9c7e25548b6db))

## [0.16.0](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.15.0...v0.16.0) (2026-05-07)


### Features

* add automatic network quality detection with polling adjustment ([b725c81](https://github.com/nmorgowicz-org/llama-monitor/commit/b725c81a1c85315fe068d90a09f9fff06377e972))
* add configurable dashboard WebSocket refresh rate (200ms-10s) ([b725c81](https://github.com/nmorgowicz-org/llama-monitor/commit/b725c81a1c85315fe068d90a09f9fff06377e972))
* **agent:** add remote agent upgrade flow with host key and OS detection ([b725c81](https://github.com/nmorgowicz-org/llama-monitor/commit/b725c81a1c85315fe068d90a09f9fff06377e972))
* **agent:** add remote agent version tracking and update detection ([b725c81](https://github.com/nmorgowicz-org/llama-monitor/commit/b725c81a1c85315fe068d90a09f9fff06377e972))
* **ui:** add chat telemetry popover with pin-to-inline toggle, throughput bars, context ring, and activity rail ([b725c81](https://github.com/nmorgowicz-org/llama-monitor/commit/b725c81a1c85315fe068d90a09f9fff06377e972))
* **ui:** add nav cockpit with live inference state, throughput, context pressure, GPU temp, and sparkline ([b725c81](https://github.com/nmorgowicz-org/llama-monitor/commit/b725c81a1c85315fe068d90a09f9fff06377e972))
* **ui:** add Performance settings tab with refresh rate and network indicator ([b725c81](https://github.com/nmorgowicz-org/llama-monitor/commit/b725c81a1c85315fe068d90a09f9fff06377e972))
* **ui:** elevate dashboard with ambient gradient orbs, typography hierarchy, and gap token standardization ([b725c81](https://github.com/nmorgowicz-org/llama-monitor/commit/b725c81a1c85315fe068d90a09f9fff06377e972))
* **ui:** refresh agent and settings modal styling with widget-card treatment ([b725c81](https://github.com/nmorgowicz-org/llama-monitor/commit/b725c81a1c85315fe068d90a09f9fff06377e972))


### Bug Fixes

* **agent:** log update available message only once on state transition ([b725c81](https://github.com/nmorgowicz-org/llama-monitor/commit/b725c81a1c85315fe068d90a09f9fff06377e972))
* **agent:** only check GitHub releases once per session to prevent rate limit errors ([b725c81](https://github.com/nmorgowicz-org/llama-monitor/commit/b725c81a1c85315fe068d90a09f9fff06377e972))
* **chat:** align chat tabs and accents with dashboard styling ([b725c81](https://github.com/nmorgowicz-org/llama-monitor/commit/b725c81a1c85315fe068d90a09f9fff06377e972))
* **ui:** add prefers-reduced-motion overrides for all dashboard animations ([b725c81](https://github.com/nmorgowicz-org/llama-monitor/commit/b725c81a1c85315fe068d90a09f9fff06377e972))
* **ui:** convert hardcoded Nord colors to CSS variables with light theme coverage ([b725c81](https://github.com/nmorgowicz-org/llama-monitor/commit/b725c81a1c85315fe068d90a09f9fff06377e972))
* **ui:** correct GPU core clock color mapping across clock visuals ([b725c81](https://github.com/nmorgowicz-org/llama-monitor/commit/b725c81a1c85315fe068d90a09f9fff06377e972))
* **ui:** fix spawn local server button on logs empty state ([b725c81](https://github.com/nmorgowicz-org/llama-monitor/commit/b725c81a1c85315fe068d90a09f9fff06377e972))
* **ui:** improve sparkline visibility and stabilize validation captures ([b725c81](https://github.com/nmorgowicz-org/llama-monitor/commit/b725c81a1c85315fe068d90a09f9fff06377e972))
* **ui:** reserve warning styling for health states instead of normal utilization spikes ([b725c81](https://github.com/nmorgowicz-org/llama-monitor/commit/b725c81a1c85315fe068d90a09f9fff06377e972))
* **ui:** standardize dashboard metric cards around a shared surface palette ([b725c81](https://github.com/nmorgowicz-org/llama-monitor/commit/b725c81a1c85315fe068d90a09f9fff06377e972))

## [0.15.0](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.14.0...v0.15.0) (2026-05-05)


### Features

* **agent:** auto-save remote agent token on install and start ([790d673](https://github.com/nmorgowicz-org/llama-monitor/commit/790d6739f6f2b28836cbacda948ce2e72e93a3e2))
* **chat:** add chat tab pinning, drag-to-reorder, persona/template menus, export/import flows, and edit/regenerate ([790d673](https://github.com/nmorgowicz-org/llama-monitor/commit/790d6739f6f2b28836cbacda948ce2e72e93a3e2))
* **chat:** add message timestamps with dates and SillyTavern integration link ([790d673](https://github.com/nmorgowicz-org/llama-monitor/commit/790d6739f6f2b28836cbacda948ce2e72e93a3e2))
* **chat:** add timeout adjustment actions and retry/dismiss recovery for chat failures ([790d673](https://github.com/nmorgowicz-org/llama-monitor/commit/790d6739f6f2b28836cbacda948ce2e72e93a3e2))
* **chat:** improve compaction with dynamic budgets, auto-compact post-response trigger, overflow guard, and structured summaries ([790d673](https://github.com/nmorgowicz-org/llama-monitor/commit/790d6739f6f2b28836cbacda948ce2e72e93a3e2))
* **ui:** add model metadata (param count, trained context) to decoding config card ([790d673](https://github.com/nmorgowicz-org/llama-monitor/commit/790d6739f6f2b28836cbacda948ce2e72e93a3e2))
* **ui:** add remote logs empty-state messaging and refresh README screenshots ([790d673](https://github.com/nmorgowicz-org/llama-monitor/commit/790d6739f6f2b28836cbacda948ce2e72e93a3e2))
* **ui:** redesign context window card with gauge/fleet views, chat-derived pressure, and most-recent-chat gauge ([790d673](https://github.com/nmorgowicz-org/llama-monitor/commit/790d6739f6f2b28836cbacda948ce2e72e93a3e2))
* **ui:** refresh dashboard cards, hardware metrics, sparkline indicators, navigation, and status treatments ([790d673](https://github.com/nmorgowicz-org/llama-monitor/commit/790d6739f6f2b28836cbacda948ce2e72e93a3e2))
* **ui:** replace agent dropdown menu with hover status tooltip ([790d673](https://github.com/nmorgowicz-org/llama-monitor/commit/790d6739f6f2b28836cbacda948ce2e72e93a3e2))


### Bug Fixes

* **agent:** write remote agent tokens to user home dirs and preserve unrelated settings on setup finish ([790d673](https://github.com/nmorgowicz-org/llama-monitor/commit/790d6739f6f2b28836cbacda948ce2e72e93a3e2))
* **api:** rename chat tab fields to camelCase to prevent duplicate-field panic ([790d673](https://github.com/nmorgowicz-org/llama-monitor/commit/790d6739f6f2b28836cbacda948ce2e72e93a3e2))
* **chat:** fix nav arrows visibility, settings panel opening, textarea auto-size, and unread message badge ([790d673](https://github.com/nmorgowicz-org/llama-monitor/commit/790d6739f6f2b28836cbacda948ce2e72e93a3e2))
* **chat:** preserve persona state, fix resend/edit flows, timeout rollback, and full-width edit layouts ([790d673](https://github.com/nmorgowicz-org/llama-monitor/commit/790d6739f6f2b28836cbacda948ce2e72e93a3e2))
* **security:** implement DOMPurify XSS sanitization and escape HTML in unsafe render paths ([790d673](https://github.com/nmorgowicz-org/llama-monitor/commit/790d6739f6f2b28836cbacda948ce2e72e93a3e2))
* **ui:** correct context usage calculations, persist pressure, and improve derived-context fallbacks ([790d673](https://github.com/nmorgowicz-org/llama-monitor/commit/790d6739f6f2b28836cbacda948ce2e72e93a3e2))
* **ui:** format GPU SCLK values, fix clock card spacing, and tighten sidebar layout ([790d673](https://github.com/nmorgowicz-org/llama-monitor/commit/790d6739f6f2b28836cbacda948ce2e72e93a3e2))
* **ui:** resolve modal navigation, export menu bugs, and visual regressions after ES module refactor ([790d673](https://github.com/nmorgowicz-org/llama-monitor/commit/790d6739f6f2b28836cbacda948ce2e72e93a3e2))
* **ui:** restore endpoint status, active button states, scroll button position, and hide empty sidebar badge ([790d673](https://github.com/nmorgowicz-org/llama-monitor/commit/790d6739f6f2b28836cbacda948ce2e72e93a3e2))

## [0.14.0](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.13.0...v0.14.0) (2026-05-02)


### Features

* **ui:** remove the legacy window facade and delete app.js startup wiring ([beee69e](https://github.com/nmorgowicz-org/llama-monitor/commit/beee69e3044296e1f78b82f577e95f6e63f9a4a3))


### Bug Fixes

* **ui:** improve modal interaction and accessibility for settings, models, and remote-agent flows ([beee69e](https://github.com/nmorgowicz-org/llama-monitor/commit/beee69e3044296e1f78b82f577e95f6e63f9a4a3))


### Performance Improvements

* **ui:** defer non-critical frontend modules to reduce startup work ([beee69e](https://github.com/nmorgowicz-org/llama-monitor/commit/beee69e3044296e1f78b82f577e95f6e63f9a4a3))
* **ui:** optimize frontend bootstrap and rendering hot paths ([beee69e](https://github.com/nmorgowicz-org/llama-monitor/commit/beee69e3044296e1f78b82f577e95f6e63f9a4a3))

## [0.13.0](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.12.0...v0.13.0) (2026-05-01)


### Features

* **ui:** refactor monolithic app.js into 22 ES modules ([005fe0f](https://github.com/nmorgowicz-org/llama-monitor/commit/005fe0f983bbcf4f9c588d53cae57ba889cbdf94))


### Bug Fixes

* **chat:** default auto-compaction on restored tabs ([005fe0f](https://github.com/nmorgowicz-org/llama-monitor/commit/005fe0f983bbcf4f9c588d53cae57ba889cbdf94))

## [0.12.0](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.11.0...v0.12.0) (2026-04-30)


### Features

* **chat:** context compaction with summarization, polish, and smart trigger ([#125](https://github.com/nmorgowicz-org/llama-monitor/issues/125)) ([ba018a7](https://github.com/nmorgowicz-org/llama-monitor/commit/ba018a7b4e27740417120109f0ff1b8675a53bdb))

## [0.11.0](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.10.2...v0.11.0) (2026-04-29)


### Features

* **chat:** UX overhaul — pagination, version display, and 18 enhancements ([#123](https://github.com/nmorgowicz-org/llama-monitor/issues/123)) ([a84bec0](https://github.com/nmorgowicz-org/llama-monitor/commit/a84bec047a082d11a302610fd7ea0c35fb12f490))

## [0.10.2](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.10.1...v0.10.2) (2026-04-29)


### Bug Fixes

* **agent:** wake poller immediately on token change, quiet expected 401s ([#121](https://github.com/nmorgowicz-org/llama-monitor/issues/121)) ([f571ae7](https://github.com/nmorgowicz-org/llama-monitor/commit/f571ae7a49e8e2df4372cc7dd68a0edffbb2a865))

## [0.10.1](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.10.0...v0.10.1) (2026-04-29)


### Bug Fixes

* **agent:** fix remote agent 401 loop and eliminate redundant SSH operations in update flow ([#119](https://github.com/nmorgowicz-org/llama-monitor/issues/119)) ([1c0af52](https://github.com/nmorgowicz-org/llama-monitor/commit/1c0af520b793b1eac4748fd4ff82afe6c676e524))

## [0.10.0](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.9.4...v0.10.0) (2026-04-29)


### Features

* **chat:** overhaul UX with labels, suggested prompts, safe defaults, and advanced toggle ([#117](https://github.com/nmorgowicz-org/llama-monitor/issues/117)) ([07a4d63](https://github.com/nmorgowicz-org/llama-monitor/commit/07a4d6340010b6fb69dfa905d283fb54844b05f1))

## [0.9.4](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.9.3...v0.9.4) (2026-04-28)


### Bug Fixes

* **docs:** enforce PR title convention for release-please compatibility ([#115](https://github.com/nmorgowicz-org/llama-monitor/issues/115)) ([d9e0e9e](https://github.com/nmorgowicz-org/llama-monitor/commit/d9e0e9eae1836bd58936327d90be2bb5cfdd9d87))

## [0.9.3](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.9.2...v0.9.3) (2026-04-28)


### Bug Fixes

* **security:** eliminate TOCTOU race via inline script execution ([#112](https://github.com/nmorgowicz-org/llama-monitor/issues/112)) ([f2113ec](https://github.com/nmorgowicz-org/llama-monitor/commit/f2113ecf4af38d9f124e9b6cce8b4d2d24de747a))

## [0.9.2](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.9.1...v0.9.2) (2026-04-28)


### Bug Fixes

* **security:** add HTTP security headers via warp-helmet ([#109](https://github.com/nmorgowicz-org/llama-monitor/issues/109)) ([b527c94](https://github.com/nmorgowicz-org/llama-monitor/commit/b527c943b1bc5681551491045b29e7e4dd12b126))

## [0.9.1](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.9.0...v0.9.1) (2026-04-27)


### Bug Fixes

* **security:** migrate extract_archive to tempfile crate ([#15](https://github.com/nmorgowicz-org/llama-monitor/issues/15)) ([#107](https://github.com/nmorgowicz-org/llama-monitor/issues/107)) ([20faa38](https://github.com/nmorgowicz-org/llama-monitor/commit/20faa382ec102018164cb5c4a5b9a9d91bf0c916))

## [0.9.0](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.8.5...v0.9.0) (2026-04-27)


### Features

* **security:** add mTLS infrastructure (cert generation, CA distribution) ([#104](https://github.com/nmorgowicz-org/llama-monitor/issues/104)) ([ffefb62](https://github.com/nmorgowicz-org/llama-monitor/commit/ffefb62ae4a282255a2e918f11c7c09cc63c9bac))

## [0.8.5](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.8.4...v0.8.5) (2026-04-27)


### Bug Fixes

* secure temp files and GPU UI improvements ([#102](https://github.com/nmorgowicz-org/llama-monitor/issues/102)) ([52a3a04](https://github.com/nmorgowicz-org/llama-monitor/commit/52a3a04c685ff4244d78f1a081998079c66c8dc9))

## [0.8.4](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.8.3...v0.8.4) (2026-04-27)


### Bug Fixes

* **ui:** hide Fix button by default and shrink GPU metrics ([#100](https://github.com/nmorgowicz-org/llama-monitor/issues/100)) ([7bc399f](https://github.com/nmorgowicz-org/llama-monitor/commit/7bc399fbf790d4ef65d7cfb893850b7fa61212c5))

## [0.8.3](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.8.2...v0.8.3) (2026-04-27)


### Bug Fixes

* **ssrf:** remove user-controlled port from chat endpoint ([#96](https://github.com/nmorgowicz-org/llama-monitor/issues/96)) ([ee5905c](https://github.com/nmorgowicz-org/llama-monitor/commit/ee5905c34a0b1f230bd34011b0a505d58b3ced82))

## [0.8.2](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.8.1...v0.8.2) (2026-04-27)


### Bug Fixes

* remote agent sensor_bridge install and UI improvements ([#94](https://github.com/nmorgowicz-org/llama-monitor/issues/94)) ([e6d531f](https://github.com/nmorgowicz-org/llama-monitor/commit/e6d531f4328cc24a56775fdfd44784df6662f102))

## [0.8.1](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.8.0...v0.8.1) (2026-04-26)


### Bug Fixes

* **agent:** use cmd.exe-compatible quoting for schtasks ([#92](https://github.com/nmorgowicz-org/llama-monitor/issues/92)) ([9c04aac](https://github.com/nmorgowicz-org/llama-monitor/commit/9c04aacda13b8fe93b6788c25b477b03b9305660))

## [0.8.0](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.7.10...v0.8.0) (2026-04-26)


### Features

* add --host flag and optional Basic Auth ([#90](https://github.com/nmorgowicz-org/llama-monitor/issues/90)) ([f2bb6f6](https://github.com/nmorgowicz-org/llama-monitor/commit/f2bb6f6a3cc440648a9f5d2aeaf93b9997d211b8))

## [0.7.10](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.7.9...v0.7.10) (2026-04-26)


### Bug Fixes

* **agent:** prevent command injection via install paths ([#88](https://github.com/nmorgowicz-org/llama-monitor/issues/88)) ([8eed884](https://github.com/nmorgowicz-org/llama-monitor/commit/8eed884f4aa758be9c322b2dc5a42abd9495b065))

## [0.7.9](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.7.8...v0.7.9) (2026-04-26)


### Bug Fixes

* **agent:** run Windows scheduled task as SYSTEM and cache system metrics ([#86](https://github.com/nmorgowicz-org/llama-monitor/issues/86)) ([dbfe6dd](https://github.com/nmorgowicz-org/llama-monitor/commit/dbfe6dd563458677314efd570e2d755c5090d1b3))

## [0.7.8](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.7.7...v0.7.8) (2026-04-26)


### Bug Fixes

* **agent:** attempt SSH autostart once per disconnect instead of retrying ([#84](https://github.com/nmorgowicz-org/llama-monitor/issues/84)) ([57b2d31](https://github.com/nmorgowicz-org/llama-monitor/commit/57b2d31365e0584b461dae376eaaddd1e60766db))

## [0.7.7](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.7.6...v0.7.7) (2026-04-26)


### Bug Fixes

* **agent:** suppress autostart during install and fix Windows file lock race ([#81](https://github.com/nmorgowicz-org/llama-monitor/issues/81)) ([f331e25](https://github.com/nmorgowicz-org/llama-monitor/commit/f331e255837da7cae642bed79a4ccb199006ab53))

## [0.7.6](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.7.5...v0.7.6) (2026-04-26)


### Bug Fixes

* **ui:** restore scrolling in settings modal ([#78](https://github.com/nmorgowicz-org/llama-monitor/issues/78)) ([6908b08](https://github.com/nmorgowicz-org/llama-monitor/commit/6908b08346b738d0988df4a1d30a94b19c5e7477))

## [0.7.5](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.7.4...v0.7.5) (2026-04-25)


### Bug Fixes

* **remote-agent:** repair Windows install follow-ups ([#75](https://github.com/nmorgowicz-org/llama-monitor/issues/75)) ([395e959](https://github.com/nmorgowicz-org/llama-monitor/commit/395e95987b0cb909a293933dc81665c5b9e21942))

## [0.7.4](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.7.3...v0.7.4) (2026-04-25)


### Bug Fixes

* **remote-agent:** repair autostart and clarify setup errors ([#72](https://github.com/nmorgowicz-org/llama-monitor/issues/72)) ([c5686e5](https://github.com/nmorgowicz-org/llama-monitor/commit/c5686e55365116fa902f2ffe61dd27110d955870))

## [0.7.3](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.7.2...v0.7.3) (2026-04-25)


### Bug Fixes

* **remote-agent:** stop Windows agent before repair install ([#70](https://github.com/nmorgowicz-org/llama-monitor/issues/70)) ([ee7fb1a](https://github.com/nmorgowicz-org/llama-monitor/commit/ee7fb1aeb4671dcf5e0756075ad56be75f71de59))

## [0.7.2](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.7.1...v0.7.2) (2026-04-25)


### Bug Fixes

* **remote-agent:** repair Windows remote install archive handling ([#68](https://github.com/nmorgowicz-org/llama-monitor/issues/68)) ([17d404c](https://github.com/nmorgowicz-org/llama-monitor/commit/17d404c5c31800ff7215ff7b7058b508630446e4))

## [0.7.1](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.7.0...v0.7.1) (2026-04-25)


### Bug Fixes

* **ci:** remove setup-dotnet action and add PR labeler ([#60](https://github.com/nmorgowicz-org/llama-monitor/issues/60)) ([c7ab22f](https://github.com/nmorgowicz-org/llama-monitor/commit/c7ab22f85f33a99c07ce01d639037b1be11e5f2f))

## [0.7.0](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.6.3...v0.7.0) (2026-04-24)


### Features

* add tool-call blocked state detection with throughput card integration ([#56](https://github.com/nmorgowicz-org/llama-monitor/issues/56)) ([36880b4](https://github.com/nmorgowicz-org/llama-monitor/commit/36880b4b93844ce0d746b6e6f20f14bacf97b11c))
* **agent:** resolve Windows %APPDATA% path for remote agent scheduler ([035b659](https://github.com/nmorgowicz-org/llama-monitor/commit/035b659579c896833f01ef8419eed116b99f1a54))


### Bug Fixes

* **agent:** resolve Windows %APPDATA% path for remote agent scheduler ([44ad9a9](https://github.com/nmorgowicz-org/llama-monitor/commit/44ad9a96ee51a47fde37834cdaa676c0ebee42da))
* **agent:** wire resolve_windows_appdata into all schtasks command paths ([0d96a13](https://github.com/nmorgowicz-org/llama-monitor/commit/0d96a13f9e45f160a1ca9f3c8c53aaee209eca03))
* **ui:** refine hardware metrics visuals and windows helper packaging ([#57](https://github.com/nmorgowicz-org/llama-monitor/issues/57)) ([44200fb](https://github.com/nmorgowicz-org/llama-monitor/commit/44200fbed630fb93991890c7fa3b19809b88f482))

## [0.6.3](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.6.2...v0.6.3) (2026-04-23)


### Bug Fixes

* **ci:** add explicit permissions to CI workflow ([791b826](https://github.com/nmorgowicz-org/llama-monitor/commit/791b82612747e27c307edfa53da54817d44a695c))
* **js:** properly escape backslashes in file browser paths ([8b89703](https://github.com/nmorgowicz-org/llama-monitor/commit/8b897038c8dcc93a4835a2f9a162e8c6597f7753))

## [0.6.2](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.6.1...v0.6.2) (2026-04-22)


### Bug Fixes

* **ci:** use shared cache key to eliminate cache duplication ([0bc4c96](https://github.com/nmorgowicz-org/llama-monitor/commit/0bc4c96a4cf37626041bdf2fb00231da77fb084c))
* **ci:** use shared cache key to eliminate cache duplication ([4ff264d](https://github.com/nmorgowicz-org/llama-monitor/commit/4ff264d9f0adf0d4bfb2301f8da2265ea33d16eb))

## [0.6.1](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.6.0...v0.6.1) (2026-04-22)


### Bug Fixes

* **release:** add AR and RANLIB env vars for macOS cross-compilation ([#41](https://github.com/nmorgowicz-org/llama-monitor/issues/41)) ([0fb73bf](https://github.com/nmorgowicz-org/llama-monitor/commit/0fb73bfa342a2700f5a20cfb656e850cb9e44298))

## [0.6.0](https://github.com/nmorgowicz-org/llama-monitor/compare/v0.5.1...v0.6.0) (2026-04-22)


### Features

* **ui:** comprehensive UI/UX modernization with remote agent, inference dashboard, and capability-aware rendering ([#38](https://github.com/nmorgowicz-org/llama-monitor/issues/38)) ([dfc0ed8](https://github.com/nmorgowicz-org/llama-monitor/commit/dfc0ed8870c55255a4205e7208ff2f5bc9833f13))
* **ui:** Phase 5 modern UI - dashboard grid, toast notifications, keyboard shortcuts ([dfc0ed8](https://github.com/nmorgowicz-org/llama-monitor/commit/dfc0ed8870c55255a4205e7208ff2f5bc9833f13))

## [Unreleased]

### Added

- Remote agent functionality
- `--agent` flag to run as lightweight remote metrics agent
- `--agent-host` and `--agent-port` CLI flags for agent configuration
- `--agent-token` for bearer token authentication
- `--remote-agent-url` and `--remote-agent-token` for dashboard polling configuration
- `--remote-agent-ssh-autostart` and SSH-related flags for remote agent autostart
- `/api/remote-agent/releases/latest` endpoint for release checking
- `/api/remote-agent/detect` endpoint for remote host detection
- `agent.rs` module with `run_agent_server`, `latest_release_info`, and `detect_remote_agent` functions
- Remote agent URL inference from attached llama-server endpoint
- SSH autostart support for unreachable remote agents
- Remote agent connection tracking in state
- Capability-aware UI with "Inference only" warning when host metrics unavailable
- Compact view support for remote agent scenarios
- GPU/System section visibility toggling based on capabilities
- Remote agent status display in web UI
- Session endpoint information in WebSocket updates
- `host_metrics_available()` and `remote_agent_connected()` state methods

### Changed

- Refactored `AppState` to track `remote_agent_connected` and `remote_agent_url`
- Added `current_session_kind()` and `current_endpoint_kind()` helper methods
- Updated tray to use `host_metrics_available()` instead of `active_session_uses_local_metrics()`
- Updated WebSocket updates to include remote agent connection state
- Updated metrics polling to skip when no local metrics needed
- GPU metrics types now implement `Deserialize` for remote agent compatibility
- System metrics types now implement `Deserialize` for remote agent compatibility
- Capability calculation now includes remote agent connected scenario
- UI settings now store remote agent configuration

### Fixed

- GPU detection logic now handles Windows `where` command
- System/GPU section visibility in compact view

## [0.2.0] - 2026-04-20

### Added

- Initial release with core functionality
- Session management (spawn/attach modes)
- GPU metrics monitoring (NVIDIA, AMD, Apple Silicon)
- Llama server integration
- System metrics collection
- Tray icon with capability-aware display
- Remote agent UI affordance
- Integration tests for capabilities
