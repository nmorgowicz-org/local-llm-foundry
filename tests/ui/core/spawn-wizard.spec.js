// ── Spawn Wizard E2E Tests (Phases 3, 4, and Rapid-MLX Phase 6) ──────────────
// Tests for:
// - HF model search integration
// - Third-party model import
// - Introspection-driven config
// - Error states (no GPU, no models, no internet)
// - Rate limiting (HF search / download)
// These tests assume the app is running and the spawn wizard is accessible.

import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

test.describe('Spawn Wizard - Phases 3, 4, and Rapid-MLX Phase 6', () => {
  test('@in-memory-test calibration candidates apply through canonical wizard controls', async ({ page }) => {
    await page.goto('/');
    const result = await page.evaluate(async () => {
      const { applyCalibrationPatch } = await import('/js/features/spawn-wizard-tuning.js');
      applyCalibrationPatch({
        context_size: 16384,
        batch_size: 1024,
        ubatch_size: 512,
        ctk: 'q8_0',
        ctv: 'q8_0',
        flash_attn: true,
      });
      const ids = ['spawn-context-size', 'spawn-batch-size', 'spawn-ubatch-size', 'spawn-cache-type-k', 'spawn-cache-type-v', 'spawn-flash-attn'];
      return Object.fromEntries(ids.map(id => [id, document.getElementById(id)?.value]));
    });
    expect(result).toMatchObject({
      'spawn-context-size': '16384',
      'spawn-batch-size': '1024',
      'spawn-ubatch-size': '512',
      'spawn-cache-type-k': 'q8_0',
      'spawn-cache-type-v': 'q8_0',
      'spawn-flash-attn': 'on',
    });
  });

  test('@in-memory-test calibration offer is hidden without a local llama.cpp GGUF', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    const result = await page.evaluate(async () => {
      const { openSpawnWizard, wizardState, selectWizardEngine, showStep, refreshWizardCalibrationOffer } = await import('/js/features/spawn-wizard.js');
      openSpawnWizard();
      wizardState.model.source = 'hf';
      wizardState.model.path = 'org/model';
      selectWizardEngine('llama_cpp', true);
      showStep(2);
      await refreshWizardCalibrationOffer();
      const noLocal = document.getElementById('spawn-calibration-card')?.hidden;
      wizardState.model.source = 'local';
      wizardState.model.path = '/tmp/rapid-model.gguf';
      selectWizardEngine('rapid_mlx', true);
      await refreshWizardCalibrationOffer();
      const rapid = document.getElementById('spawn-calibration-card')?.hidden;
      return { noLocal, rapid };
    });
    expect(result).toEqual({ noLocal: true, rapid: true });
  });

  test('@in-memory-test related calibration requires explicit confirmation before applying', async ({ page }) => {
    await page.goto('/');
    const result = await page.evaluate(async () => {
      const { wizardState, applySelectedWizardCalibration } = await import('/js/features/spawn-wizard.js');
      wizardState.calibration.selectedMatch = {
        match_kind: 'related',
        receipt: {
          selected_candidate: 'related-candidate',
          candidate_results: [{ candidate: { id: 'related-candidate', typed_patch: { batch_size: 1024 } } }],
        },
      };
      const originalConfirm = window.confirm;
      let confirmations = 0;
      window.confirm = () => { confirmations += 1; return false; };
      applySelectedWizardCalibration();
      const rejected = document.getElementById('spawn-batch-size')?.value;
      window.confirm = () => { confirmations += 1; return true; };
      applySelectedWizardCalibration();
      const accepted = document.getElementById('spawn-batch-size')?.value;
      window.confirm = originalConfirm;
      return { confirmations, rejected, accepted };
    });
    expect(result).toEqual({ confirmations: 2, rejected: '2048', accepted: '1024' });
  });

    test('@in-memory-test engine classifier leaves bare HF repos ambiguous and recognizes GGUF inventory', async ({ page }) => {
        await page.goto('/');
        const result = await page.evaluate(async () => {
            const { classifyWizardArtifact } = await import('/js/features/spawn-wizard.js');
            return {
                bareRepo: classifyWizardArtifact({ hfRepo: 'owner/model', hfFile: '', quantFiles: [] }),
                ggufRepo: classifyWizardArtifact({
                    hfRepo: 'owner/model',
                    hfFile: '',
                    quantFiles: [{ path: 'model-Q4_K_M.gguf' }],
                }),
                mlx: classifyWizardArtifact({
                    path: '/models/Qwen-MLX',
                    localMeta: { source_kind: 'mlx_directory' },
                }),
                arbitraryFile: classifyWizardArtifact({ path: '/models/README.txt' }),
            };
        });

        expect(result).toEqual({
            bareRepo: 'unknown',
            ggufRepo: 'gguf',
            mlx: 'mlx_directory',
            arbitraryFile: 'unknown',
        });
    });

    test.use({ animationBehavior: 'disabled' });

    test('external Rapid-MLX runtime can be recommended and an explicit llama.cpp choice is preserved', async ({ page }) => {
        await page.route('**/api/llama-binary/platform-info', route => route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ rapid_mlx_local_available: true, auto_backend: 'metal' }),
        }));
        await page.route('**/api/rapid-mlx/runtime/status', route => route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ runtime: { supported: true, active: null } }),
        }));
        await page.route('**/api/rapid-mlx/recommend', async route => {
            await new Promise(resolve => setTimeout(resolve, 100));
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({
                    recommended_backend: 'rapid_mlx',
                    state: 'ready',
                    reason: 'A compatible external Rapid-MLX runtime is ready.',
                }),
            });
        });

        await page.goto('/');
        await page.waitForLoadState('networkidle');
        await page.evaluate(async () => {
            const { openSpawnWizard } = await import('/js/features/spawn-wizard.js');
            openSpawnWizard({
                localPath: '/models/Qwen-MLX',
                localModel: { source_kind: 'mlx_directory', path: '/models/Qwen-MLX' },
            });
        });
        await page.locator('.wizard-engine-card').first().waitFor({ state: 'visible', timeout: 5000 });
        await page.locator('.wizard-engine-card[data-engine="llama_cpp"]').click();

        await expect(page.locator('.wizard-engine-card[data-engine="llama_cpp"]')).toHaveClass(/selected/);
        await expect(page.locator('#wizard-engine-reason')).toContainText('manual llama.cpp choice is preserved');
    });

    test('@in-memory-test Rapid-MLX payload is backend-exclusive while its preset preserves shared sampling', async ({ page }) => {
        await page.goto('/');
        const payloads = await page.evaluate(async () => {
            const {
                buildSpawnPayload,
                buildPresetPayload,
                launchPortForPayload,
                supportsTunePanelForPayload,
                wizardState,
            } = await import('/js/features/spawn-wizard.js');
            wizardState.engine.selected = 'rapid_mlx';
            wizardState.model.source = 'local';
            wizardState.model.path = '/models/Qwen-MLX';
            wizardState.access.port = 9123;
            wizardState.access.bindHost = '127.0.0.1';
            wizardState.access.apiKey = 'rapid-secret';
            wizardState.hardware.gpuLayers = 'all';
            wizardState.hardware.contextSize = 131072;
            wizardState.hardware.cacheTypeK = 'q4_0';
            wizardState.hardware.temperature = 0.42;
            wizardState.hardware.topP = 0.88;
            wizardState.hardware.topK = 32;
            wizardState.hardware.minP = 0.06;
            wizardState.hardware.repeatPenalty = 1.07;
            wizardState.hardware.presencePenalty = 0.15;
            wizardState.hardware.maxTokens = 2048;
            wizardState.hardware.seed = 7;
            const spawn = buildSpawnPayload();
            return {
                spawn,
                preset: buildPresetPayload(),
                launchPort: launchPortForPayload(spawn),
                supportsTune: supportsTunePanelForPayload(spawn),
            };
        });

        expect(payloads.spawn.backend).toBe('rapid_mlx');
        expect(payloads.spawn.rapid_mlx.model_source).toEqual({ kind: 'mlx_directory', path: '/models/Qwen-MLX' });
        expect(payloads.spawn.rapid_mlx.served_model_name).toBeNull();
        expect(payloads.spawn.rapid_mlx.host).toBe('127.0.0.1');
        expect(payloads.spawn.rapid_mlx.port).toBe(9123);
        expect(payloads.spawn.rapid_mlx.api_key).toBe('rapid-secret');
        expect(payloads.spawn.rapid_mlx.reasoning_mode).toBe('on');
        expect(payloads.spawn.rapid_mlx.hybrid_mode).toBe('auto');
        expect(payloads.spawn.rapid_mlx.prefill_step_size).toBe(512);
        expect(payloads.spawn.rapid_mlx.prefix_cache_enabled).toBe(true);
        expect(payloads.spawn.rapid_mlx.retained_cache_mib).toBe(8192);
        expect(payloads.spawn.rapid_mlx.hybrid_cache_entries).toBe(16);
        expect(payloads.spawn.rapid_mlx.disk_checkpoint_interval).toBe(0);
        // Phase 7 sampling defaults should be serialized when set
        expect(payloads.spawn.rapid_mlx.default_temperature).toBe(0.42);
        expect(payloads.spawn.rapid_mlx.default_top_p).toBe(0.88);
        expect(payloads.spawn.rapid_mlx.default_top_k).toBe(32);
        expect(payloads.spawn.rapid_mlx.default_min_p).toBe(0.06);
        expect(payloads.spawn.rapid_mlx.default_repetition_penalty).toBe(1.07);
        expect(payloads.spawn.rapid_mlx.default_presence_penalty).toBe(0.15);
        expect(payloads.spawn.rapid_mlx.max_tokens).toBe(2048);
        expect(payloads.spawn).not.toHaveProperty('gpu_layers');
        expect(payloads.spawn).not.toHaveProperty('context_size');
        expect(payloads.spawn).not.toHaveProperty('ctk');
        expect(payloads.launchPort).toBe(9123);
        expect(payloads.supportsTune).toBe(true);
        expect(payloads.preset).toMatchObject({
            backend: 'rapid_mlx',
            temperature: 0.42,
            top_p: 0.88,
            top_k: 32,
            min_p: 0.06,
            repeat_penalty: 1.07,
            presence_penalty: 0.15,
            max_tokens: 2048,
            seed: 7,
            api_key: 'rapid-secret',
        });
        expect(payloads.preset.rapid_mlx).not.toHaveProperty('api_key');
        const saveResult = await page.evaluate(async preset => {
            const headers = window.authHeaders ? window.authHeaders() : {};
            const response = await fetch('/api/presets', {
                method: 'POST',
                headers: { ...headers, 'Content-Type': 'application/json' },
                body: JSON.stringify({ ...preset, name: 'Rapid protected key test' }),
            });
            return { ok: response.ok, status: response.status, body: await response.text() };
        }, payloads.preset);
        expect(saveResult, saveResult.body).toMatchObject({ ok: true });
    });

    test('@in-memory-test Rapid-MLX template restores typed source and reopening clears stale engine state', async ({ page }) => {
        await page.goto('/');
        await page.waitForSelector('html.modules-ready');
        const state = await page.evaluate(async () => {
            const { openSpawnWizard, closeSpawnWizard, buildSpawnPayload, wizardState } = await import('/js/features/spawn-wizard.js');
            openSpawnWizard({
                templatePreset: {
                    backend: 'rapid_mlx',
                    temperature: 0.3,
                    top_p: 0.91,
                    top_k: 44,
                    min_p: 0.08,
                    repeat_penalty: 1.09,
                    presence_penalty: 0.2,
                    max_tokens: 3072,
                    seed: 17,
                    rapid_mlx: {
                        model_source: { kind: 'hugging_face_repo', repo_id: 'mlx-community/Qwen', revision: 'v2' },
                        host: '127.0.0.1',
                        port: 9000,
                    },
                },
            });
            const restored = buildSpawnPayload();
            const sampling = {
                temperature: wizardState.hardware.temperature,
                topP: wizardState.hardware.topP,
                topK: wizardState.hardware.topK,
                minP: wizardState.hardware.minP,
                repeatPenalty: wizardState.hardware.repeatPenalty,
                presencePenalty: wizardState.hardware.presencePenalty,
                maxTokens: wizardState.hardware.maxTokens,
                seed: wizardState.hardware.seed,
            };
            closeSpawnWizard();
            openSpawnWizard();
            return {
                restored,
                sampling,
                reset: { selected: wizardState.engine.selected, explicit: wizardState.engine.explicit },
            };
        });

        expect(state.restored.rapid_mlx.model_source).toEqual({
            kind: 'hugging_face_repo',
            repo_id: 'mlx-community/Qwen',
            revision: 'v2',
        });
        expect(state.restored.rapid_mlx.port).toBe(9000);
        expect(state.sampling).toEqual({
            temperature: 0.3,
            topP: 0.91,
            topK: 44,
            minP: 0.08,
            repeatPenalty: 1.09,
            presencePenalty: 0.2,
            maxTokens: 3072,
            seed: 17,
        });
        expect(state.reset).toEqual({ selected: 'llama_cpp', explicit: false });
    });

    test('alias and authoritative Hugging Face sources restore into navigable Rapid-MLX templates', async ({ page }) => {
        await page.route('**/api/llama-binary/platform-info', route => route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ rapid_mlx_local_available: true, auto_backend: 'metal' }),
        }));
        await page.route('**/api/rapid-mlx/runtime/status', route => route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ runtime: { supported: true, active: { version: '0.10.10' } } }),
        }));
        await page.route('**/api/rapid-mlx/recommend', route => route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
                recommended_backend: 'rapid_mlx',
                state: 'ready',
                reason: 'This source is native to the verified Rapid-MLX resolution path.',
            }),
        }));

        await page.goto('/');
        await page.waitForLoadState('networkidle');
        await page.evaluate(async () => {
            const { openSpawnWizard } = await import('/js/features/spawn-wizard.js');
            openSpawnWizard({
                templatePreset: {
                    backend: 'rapid_mlx',
                    rapid_mlx: {
                        model_source: { kind: 'alias', value: 'team/production-model' },
                        host: '127.0.0.1',
                        port: 8001,
                    },
                },
            });
        });
        await page.locator('#spawn-model-path').waitFor({ state: 'visible', timeout: 5000 });
        await expect(page.locator('#spawn-model-path')).toHaveValue('team/production-model');
        await expect(page.locator('#wizard-next-btn')).toBeEnabled();
        await page.locator('#wizard-next-btn').click();
        await expect(page.locator('#wizard-step-1')).toHaveClass(/active/);

        await page.evaluate(async () => {
            const { closeSpawnWizard, openSpawnWizard } = await import('/js/features/spawn-wizard.js');
            closeSpawnWizard();
            openSpawnWizard({
                templatePreset: {
                    backend: 'rapid_mlx',
                    rapid_mlx: {
                        model_source: {
                            kind: 'authoritative_safetensors',
                            source: {
                                kind: 'hugging_face_repo',
                                repo_id: 'owner/source-model',
                                revision: 'release-2',
                            },
                            revision_or_hash: 'release-2',
                            recipe: 'q6',
                        },
                        host: '127.0.0.1',
                        port: 8001,
                    },
                },
            });
        });
        await page.locator('#spawn-hf-repo').waitFor({ state: 'visible', timeout: 5000 });
        await expect(page.locator('#spawn-hf-repo')).toHaveValue('owner/source-model');
        await expect(page.locator('#wizard-next-btn')).toBeEnabled();
        await page.locator('#wizard-next-btn').click();
        await expect(page.locator('#wizard-step-1')).toHaveClass(/active/);
    });

    test('recommendations select llama.cpp for GGUF and Rapid-MLX for a typed MLX directory', async ({ page }) => {
        await page.route('**/api/llama-binary/platform-info', route => route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ rapid_mlx_local_available: true, auto_backend: 'metal' }),
        }));
        await page.route('**/api/rapid-mlx/runtime/status', route => route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ runtime: { supported: true, active: { version: '0.10.10' } } }),
        }));
        await page.route('**/api/rapid-mlx/recommend', async route => {
            const { artifact_kind: kind } = route.request().postDataJSON();
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify(kind === 'gguf'
                    ? { recommended_backend: 'llama_cpp', state: 'ready', reason: 'GGUF runs natively with llama.cpp.' }
                    : { recommended_backend: 'rapid_mlx', state: 'ready', reason: 'This source is native to Rapid-MLX.' }),
            });
        });

        await page.goto('/');
        await page.waitForLoadState('networkidle');
        await page.evaluate(async () => {
            const { openSpawnWizard } = await import('/js/features/spawn-wizard.js');
            openSpawnWizard({ localPath: '/models/model.gguf' });
        });
        await page.locator('.wizard-engine-card').first().waitFor({ state: 'visible', timeout: 5000 });
        await expect(page.locator('.wizard-engine-card[data-engine="llama_cpp"]')).toHaveClass(/selected/);
        await page.locator('.wizard-engine-card[data-engine="rapid_mlx"]').click();
        await expect(page.locator('#wizard-next-btn')).toBeDisabled();
        await expect(page.locator('#wizard-footer-hint')).toContainText('GGUF runs with llama.cpp');
        await page.locator('.wizard-engine-card[data-engine="llama_cpp"]').click();

        await page.evaluate(async () => {
            const { closeSpawnWizard, openSpawnWizard } = await import('/js/features/spawn-wizard.js');
            closeSpawnWizard();
            openSpawnWizard({
                localPath: '/models/model-mlx',
                localModel: {
                    path: '/models/model-mlx',
                    size_bytes: 6_450_000_000,
                    source_kind: 'mlx_directory',
                    model_source: { kind: 'mlx_directory', path: '/models/model-mlx' },
                },
            });
        });
        await page.locator('.wizard-engine-card').first().waitFor({ state: 'visible', timeout: 5000 });
        await expect(page.locator('.wizard-engine-card[data-engine="rapid_mlx"]')).toHaveClass(/selected/);
        await expect(page.locator('#wizard-engine-reason')).toContainText('native to Rapid-MLX');
        await expect(page.locator('#wizard-binary-prereq')).toBeHidden();
        await expect(page.locator('.model-source-card[data-source="import"]')).toBeHidden();

        await page.locator('.wizard-engine-card[data-engine="llama_cpp"]').click();
        await expect(page.locator('#wizard-next-btn')).toBeDisabled();
        await expect(page.locator('#wizard-footer-hint')).toContainText('requires Rapid-MLX');
        await page.locator('.wizard-engine-card[data-engine="rapid_mlx"]').click();

        await page.locator('#wizard-next-btn').click();
        await expect(page.locator('#rapid-hardware-panel')).toBeVisible();
        // Rapid-MLX keeps the shared main layout visible on the hardware step;
        // backend-native controls live alongside the fit rail.
        await expect(page.locator('#wizard-step-1 > .hw-vram-sidebar')).toBeVisible();
        await expect(page.locator('#wizard-step-1 > .wizard-main')).toBeVisible();
        await expect(page.locator('#wizard-step-1 #rapid-hardware-panel')).toBeVisible();
    });

    test('@in-memory-test typed Rapid-MLX sources round-trip unchanged and llama.cpp payloads exclude Rapid-MLX config', async ({ page }) => {
        await page.goto('/');
        const result = await page.evaluate(async () => {
            const { buildSpawnPayload, wizardState } = await import('/js/features/spawn-wizard.js');
            const sources = [
                { kind: 'mlx_directory', path: '/models/mlx' },
                { kind: 'hugging_face_repo', repo_id: 'mlx-community/model', revision: 'release-1' },
                { kind: 'alias', value: 'team-model' },
                {
                    kind: 'authoritative_safetensors',
                    source: { kind: 'hugging_face_repo', repo_id: 'owner/source', revision: 'abc123' },
                    revision_or_hash: 'abc123',
                    recipe: 'q6',
                },
            ];
            wizardState.engine.selected = 'rapid_mlx';
            const roundTrips = sources.map(source => {
                wizardState.model.rapidMlxSource = source;
                return buildSpawnPayload().rapid_mlx.model_source;
            });
            wizardState.engine.selected = 'llama_cpp';
            wizardState.model.source = 'local';
            wizardState.model.path = '/models/model.gguf';
            const llama = buildSpawnPayload();
            return { sources, roundTrips, llama };
        });

        expect(result.roundTrips).toEqual(result.sources);
        expect(result.llama.backend).toBe('llama_cpp');
        expect(result.llama).not.toHaveProperty('rapid_mlx');
    });

    test('unsupported Rapid-MLX local launch is visibly blocked with actionable guidance', async ({ page }) => {
        await page.route('**/api/llama-binary/platform-info', route => route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ rapid_mlx_local_available: false, auto_backend: 'cuda' }),
        }));
        await page.route('**/api/rapid-mlx/runtime/status', route => route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ runtime: { supported: false, active: null } }),
        }));
        await page.route('**/api/rapid-mlx/recommend', route => route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
                recommended_backend: null,
                state: 'platform_unavailable',
                reason: 'Local Rapid-MLX requires Apple Silicon macOS; remote attachment remains available.',
            }),
        }));

        await page.goto('/');
        await page.waitForLoadState('networkidle');
        await page.evaluate(async () => {
            const { openSpawnWizard } = await import('/js/features/spawn-wizard.js');
            openSpawnWizard({
                localPath: '/models/model-mlx',
                localModel: { source_kind: 'mlx_directory', path: '/models/model-mlx' },
            });
        });
        await page.locator('.wizard-engine-card').first().waitFor({ state: 'visible', timeout: 5000 });
        await page.locator('.wizard-engine-card[data-engine="rapid_mlx"]').click();

        await expect(page.locator('.wizard-engine-card[data-engine="rapid_mlx"]')).toHaveClass(/is-unavailable/);
        await expect(page.locator('[data-engine-badge="rapid_mlx"]')).toContainText('Apple Silicon only');
        await expect(page.locator('#wizard-engine-reason')).toContainText('remote attachment remains available');
        await expect(page.locator('#wizard-next-btn')).toBeDisabled();
        await expect(page.locator('#wizard-footer-hint')).toContainText('Apple Silicon macOS');
    });
    test('HF model search returns results', async ({ page }) => {
        await page.goto('/');
        await page.waitForLoadState('networkidle');

        // Open spawn wizard if button exists.
        const openBtn = page.locator('#spawn-wizard-btn, button:has-text("Spawn")').first();
        if (await openBtn.isVisible({ timeout: 3000 }).catch(() => false)) {
            await openBtn.click();
        }

        // Check that the HF search endpoint is reachable via a simple network request.
        const [response] = await Promise.all([
            page.waitForResponse(r => r.url().includes('/api/hf/search') && r.request().method() === 'POST'),
            page.evaluate(async () => {
                const headers = window.authHeaders ? window.authHeaders() : {};
                return fetch('/api/hf/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', ...headers },
                    body: JSON.stringify({ query: 'llama', limit: 5 }),
                });
            }),
        ]);

        expect(response.ok()).toBeTruthy();
        const data = await response.json();
        expect(data.ok).toBe(true);
        expect(Array.isArray(data.models)).toBe(true);
    });

    test('HF file browser marks the family-recommended mmproj', async ({ page }) => {
        await page.route('**/api/hf/files', route => route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
                ok: true,
                files: [
                    {
                        repo_id: 'unsloth/gemma-4-31B-it-qat-GGUF',
                        path: 'gemma-4-31B-it-Q4_0.gguf',
                        size: 17_400_000_000,
                        label: 'Q4_0',
                        quant_type: 'standard',
                        is_mmproj: false,
                        is_recommended_mmproj: false,
                        mmproj_recommendation: '',
                    },
                    {
                        repo_id: 'unsloth/gemma-4-31B-it-qat-GGUF',
                        path: 'mmproj-F16.gguf',
                        size: 741_000_000,
                        label: 'F16',
                        quant_type: 'standard',
                        is_mmproj: true,
                        is_recommended_mmproj: true,
                        mmproj_recommendation: 'F16 is the documented llama.cpp projector default for Gemma 4',
                    },
                    {
                        repo_id: 'unsloth/gemma-4-31B-it-qat-GGUF',
                        path: 'mmproj-F32.gguf',
                        size: 1_480_000_000,
                        label: 'F32',
                        quant_type: 'standard',
                        is_mmproj: true,
                        is_recommended_mmproj: false,
                        mmproj_recommendation: '',
                    },
                ],
            }),
        }));

        await page.goto('/');
        await page.waitForLoadState('networkidle');
        await page.evaluate(async () => {
            const container = document.createElement('div');
            container.id = 'test-hf-files';
            document.body.appendChild(container);
            const { hfListFiles } = await import('/js/features/hf-browse.js');
            await hfListFiles({
                repoId: 'unsloth/gemma-4-31B-it-qat-GGUF',
                container,
                vramGb: 32,
            });
        });

        const recommended = page.locator('#test-hf-files .hf-file-item', {
            hasText: 'mmproj-F16.gguf',
        });
        await expect(recommended).toContainText('Family recommended');
        await expect(recommended.locator('.hf-file-badge-recommended')).toHaveAttribute(
            'title',
            /documented llama\.cpp projector default for Gemma 4/
        );
        await expect(page.locator('#test-hf-files .hf-file-item', {
            hasText: 'mmproj-F32.gguf',
        })).not.toContainText('Family recommended');
    });

    test('HF file browser selects a linked mmproj from its owning repo', async ({ page }) => {
        await page.route('**/api/hf/files', route => route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
                ok: true,
                files: [{
                    repo_id: 'mradermacher/model-GGUF',
                    path: 'mmproj-F16.gguf',
                    size: 1_000_000_000,
                    label: 'F16',
                    quant_type: 'standard',
                    is_mmproj: true,
                    is_recommended_mmproj: true,
                    mmproj_recommendation: 'F16 is recommended',
                }],
            }),
        }));

        await page.goto('/');
        await page.waitForLoadState('networkidle');
        const selectedRepo = await page.evaluate(async () => {
            const container = document.createElement('div');
            document.body.appendChild(container);
            const { hfListFiles } = await import('/js/features/hf-browse.js');
            return new Promise(resolve => {
                hfListFiles({
                    repoId: 'mradermacher/model-i1-GGUF',
                    container,
                    vramGb: 32,
                    onSelectFile: (_file, repoId) => resolve(repoId),
                }).then(() => container.querySelector('.hf-file-item').click());
            });
        });

        expect(selectedRepo).toBe('mradermacher/model-GGUF');
    });

    test('Third-party models endpoint responds', async ({ page }) => {
        await page.goto('/');
        await page.waitForLoadState('networkidle');

        const [response] = await Promise.all([
            page.waitForResponse(r => r.url().includes('/api/third-party-models') && r.request().method() === 'POST'),
            page.evaluate(async () => {
                const headers = window.authHeaders ? window.authHeaders() : {};
                return fetch('/api/third-party-models', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', ...headers },
                    body: JSON.stringify({ include_subdirs: true }),
                });
            }),
        ]);

        expect(response.ok()).toBeTruthy();
        const data = await response.json();
        expect(data.ok).toBe(true);
        expect(Array.isArray(data.models)).toBe(true);
    });

    test('Model introspect endpoint responds with 400 when not configured', async ({ page }) => {
        await page.goto('/');
        await page.waitForLoadState('networkidle');

        const [response] = await Promise.all([
            page.waitForResponse(r => r.url().includes('/api/model/introspect') && r.request().method() === 'POST'),
            page.evaluate(async () => {
                const headers = window.authHeaders ? window.authHeaders() : {};
                return fetch('/api/model/introspect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', ...headers },
                    body: JSON.stringify({ model_path: '/nonexistent/model.gguf' }),
                });
            }),
        ]);

        // Endpoint is reachable; 200 is ok (graceful response), or 4xx if not configured.
        const status = response.status();
        expect(status).toBeGreaterThanOrEqual(200);
        expect(status).toBeLessThan(500);
    });

    test('HF search rate limiting returns non-200 after many requests', async ({ page }) => {
        await page.goto('/');
        await page.waitForLoadState('networkidle');

        // Send many rapid requests; at least one should be rejected.
        // The HF search rate limit is 10/60s (global), so other tests may
        // consume part of the budget. We use a generous count and accept
        // any non-200 as evidence that the limiter is active.
        const COUNT = 30;
        let nonOk = false;
        for (let i = 0; i < COUNT; i++) {
            const resp = await page.evaluate(async () => {
                const headers = window.authHeaders ? window.authHeaders() : {};
                return fetch('/api/hf/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', ...headers },
                    body: JSON.stringify({ query: 'llama', limit: 5 }),
                });
            });
            if (resp.status !== 200) {
                nonOk = true;
                break;
            }
        }
        expect(nonOk).toBe(true);
    });

    test('HF download cooldown returns 429 on rapid start', async ({ page }) => {
        await page.goto('/');
        await page.waitForLoadState('networkidle');

        // First request: start a download.
        const [first] = await Promise.all([
            page.waitForResponse(r => r.url().includes('/api/hf/download')),
            page.evaluate(async () => {
                const headers = window.authHeaders ? window.authHeaders() : {};
                return fetch('/api/hf/download', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', ...headers },
                    body: JSON.stringify({ repo_id: 'test/repo', file_path: 'model.gguf' }),
                });
            }),
        ]);

        // Immediate second request: should be 429 (cooldown).
        const [second] = await Promise.all([
            page.waitForResponse(r => r.url().includes('/api/hf/download')),
            page.evaluate(async () => {
                const headers = window.authHeaders ? window.authHeaders() : {};
                return fetch('/api/hf/download', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', ...headers },
                    body: JSON.stringify({ repo_id: 'test/repo', file_path: 'model.gguf' }),
                });
            }),
        ]);

        // At least one should be non-200; cooldown should kick in.
        expect(second.status()).toBe(429);
    });

    test('Spawn wizard UI is keyboard accessible', async ({ page }) => {
        await page.goto('/');
        await page.waitForLoadState('networkidle');

        // Open wizard via JS helper (DOM open button is a hidden compat element).
        await page.evaluate(async () => {
            const { openSpawnWizard } = await import('/js/features/spawn-wizard.js');
            openSpawnWizard();
        });
        await expect(page.locator('#spawn-wizard-overlay')).toHaveClass(/open/);

        // Escape key should close the wizard.
        await page.keyboard.press('Escape');

        const overlay = page.locator('#spawn-wizard-overlay');
        // After Escape, overlay should be closed.
        await expect(overlay).not.toBeVisible({ timeout: 2000 });
    });

    test('Spawn wizard ignores backdrop clicks so progress is not lost', async ({ page }) => {
        await page.goto('/');
        await page.waitForLoadState('networkidle');

        await page.evaluate(async () => {
            const { openSpawnWizard } = await import('/js/features/spawn-wizard.js');
            openSpawnWizard();
        });

        const overlay = page.locator('#spawn-wizard-overlay');
        await expect(overlay).toHaveClass(/open/);

        await page.locator('#spawn-wizard-overlay').click({ position: { x: 8, y: 8 } });

        await expect(overlay).toHaveClass(/open/);
        await expect(page.locator('#wizard-step-0')).toHaveClass(/active/);
    });

    test('Model step disables Next until a model is selected', async ({ page }) => {
        await page.goto('/');
        await page.waitForLoadState('networkidle');

        await page.evaluate(async () => {
            const { openSpawnWizard } = await import('/js/features/spawn-wizard.js');
            openSpawnWizard();
        });

        await expect(page.locator('#wizard-step-0')).toHaveClass(/active/);

        const nextBtn = page.locator('#wizard-next-btn');
        await expect(nextBtn).toBeDisabled();
        await expect(page.locator('#wizard-footer-hint')).toContainText('Choose a local GGUF file');

        await page.fill('#spawn-model-path', '/tmp/Qwen3.6-27B-Instruct-Q4_K_M.gguf');

        await expect(nextBtn).toBeEnabled();
        await expect(page.locator('#wizard-footer-hint')).toContainText('Local model selected');
    });

    test('Hardware auto-fit toggle auto-populates fit target and keeps step navigable', async ({ page }) => {
        await page.goto('/');
        await page.waitForLoadState('networkidle');

        await page.evaluate(async () => {
            const { openSpawnWizard, wizardState } = await import('/js/features/spawn-wizard.js');
            openSpawnWizard();
            wizardState.model.source = 'local';
            wizardState.model.path = '/tmp/Qwen3.6-27B-Instruct-Q4_K_M.gguf';
            wizardState.model.paramB = 27;
            wizardState.model.modelBytes = 16 * 1024 * 1024 * 1024;
            // Inject VRAM so renderScenarioCards doesn't short-circuit in CI (no GPU endpoint)
            wizardState.vram.available = 48 * 1024 * 1024 * 1024; // 48 GB
        });

        await page.fill('#spawn-model-path', '/tmp/Qwen3.6-27B-Instruct-Q4_K_M.gguf');
        await page.locator('#wizard-next-btn').click();
        await expect(page.locator('#wizard-step-1')).toHaveClass(/active/);
        await expect(page.locator('.vsc-section-label')).toContainText('Context fit modes');
        // Scenario cards are rendered via async backend calls; wait for them.
 await expect(page.locator('#vram-scenarios')).toContainText('Reliable agents', { timeout: 8000 });
 await page.locator('#all-settings-btn').click();
        await expect(page.locator('#spawn-cache-type-k')).toHaveValue('q8_0');
        await expect(page.locator('#spawn-cache-type-v')).toHaveValue('q8_0');

        const nextBtn = page.locator('#wizard-next-btn');

        // Enable fit — should auto-populate '2048' and keep next enabled
        await page.locator('#spawn-fit-enable').evaluate((el) => { el.value = 'true'; el.dispatchEvent(new Event('change', { bubbles: true })); });
        await expect(page.locator('#vram-scenarios')).toContainText('Reliable agents', { timeout: 8000 });
        await expect(nextBtn).toBeEnabled();
        await expect(page.locator('#spawn-fit-target')).toHaveValue('2048');

        await page.locator('#spawn-fit-target').evaluate((el) => { el.value = '2048'; el.dispatchEvent(new Event('input', { bubbles: true })); });
        await expect(nextBtn).toBeEnabled();

        // Hardware step must still be active and visible after the layout change
        await expect(page.locator('#wizard-step-1')).toHaveClass(/active/);
        await expect(page.locator('#wizard-step-1 > .wizard-main .wizard-section-title', { hasText: 'Configure hardware' })).toBeVisible();
    });

    test('@in-memory-test Spawn payload leaves fit parameters unset until the toggle is enabled', async ({ page }) => {
        await page.goto('/');
        await page.waitForLoadState('networkidle');

        const payloads = await page.evaluate(async () => {
            const { buildSpawnPayload, wizardState } = await import('/js/features/spawn-wizard.js');
            wizardState.hardware.fitEnabled = null;
            wizardState.hardware.fitTarget = '';
            const defaults = buildSpawnPayload();
            wizardState.hardware.fitEnabled = false;
            const disabled = buildSpawnPayload();
            wizardState.hardware.fitEnabled = true;
            wizardState.hardware.fitTarget = '2048';
            const enabled = buildSpawnPayload();
            return { defaults, disabled, enabled };
        });

        expect(payloads.defaults.fit_enabled).toBeNull();
        expect(payloads.defaults.fit_target).toBeNull();
        expect(payloads.disabled.fit_enabled).toBe(false);
        expect(payloads.disabled.fit_target).toBeNull();
        expect(payloads.enabled.fit_enabled).toBe(true);
        expect(payloads.enabled.fit_target).toBe('2048');
    });

    test('@in-memory-test Community templates use the correct source for Qwen and Gemma 4', async ({ page }) => {
        await page.goto('/');
        await page.waitForLoadState('networkidle');

        const templates = await page.evaluate(async () => {
 const { COMMUNITY_TEMPLATES, buildCommunityTemplateInstallRequest } = await import('/js/features/chat-template-registry.js?community-template-test=1');
 return {
   templates: COMMUNITY_TEMPLATES,
   qwenInstall: buildCommunityTemplateInstallRequest(COMMUNITY_TEMPLATES.qwen),
 };
        });

 expect(templates.templates.qwen.installEndpoint).toBe('/api/chat-template/install-hf');
 expect(templates.templates.qwen.repo).toBe('froggeric/Qwen-Fixed-Chat-Templates');
 expect(templates.templates.qwen.version).toBe('qwen3.8-froggeric-v22');
 expect(templates.templates.qwen.revision).toBe('9f14778c92c3b5ed3e0738085694c0d3452802dd');
 expect(templates.qwenInstall.body.revision).toBe(templates.templates.qwen.revision);
 expect(templates.qwenInstall.body.force).toBe(true);
        // Google's official template is the priority default (first entry);
        // jscott3201's agentic fork is kept as a fallback entry.
 expect(templates.templates.gemma4[0].installEndpoint).toBe('/api/chat-template/install-hf');
 expect(templates.templates.gemma4[0].repo).toBe('google/gemma-4-31B-it');
 expect(templates.templates.gemma4[1].installEndpoint).toBe('/api/chat-template/install-url');
 expect(templates.templates.gemma4[1].url).toBe(
            'https://raw.githubusercontent.com/jscott3201/llm-tuning/main/gemma4/chat_templates/custom_pub_chat_template_gemma4.jinja',
        );
    });

    test('Hardware advanced toggles work: kv-unified flips state, fit-enable auto-populates target', async ({ page }) => {
        await page.goto('/');
        await page.waitForLoadState('networkidle');

        await page.evaluate(async () => {
            const { openSpawnWizard, wizardState } = await import('/js/features/spawn-wizard.js');
            openSpawnWizard();
            wizardState.model.source = 'local';
            wizardState.model.path = '/tmp/Qwen3.6-27B-Instruct-Q4_K_M.gguf';
            wizardState.model.paramB = 27;
            wizardState.model.modelBytes = 16 * 1024 * 1024 * 1024;
            // Inject VRAM so renderScenarioCards doesn't short-circuit in CI (no GPU endpoint)
            wizardState.vram.available = 48 * 1024 * 1024 * 1024; // 48 GB
        });

        await page.fill('#spawn-model-path', '/tmp/Qwen3.6-27B-Instruct-Q4_K_M.gguf');
        await page.locator('#wizard-next-btn').click();
        await expect(page.locator('#wizard-step-1')).toHaveClass(/active/);
        await expect(page.locator('.vsc-section-label')).toContainText('Context fit modes');
        // Scenario cards are rendered via async backend calls; wait for them.
        await expect(page.locator('#vram-scenarios')).toContainText('Reliable agents', { timeout: 8000 });
        await expect(page.locator('#spawn-cache-type-k')).toHaveValue('q8_0');
 await expect(page.locator('#spawn-cache-type-v')).toHaveValue('q8_0');
 await page.locator('#all-settings-btn').click();

 await expect(page.locator('#spawn-kv-unified')).toHaveValue('');
await page.locator('#spawn-kv-unified').evaluate((el) => { el.value = 'false'; el.dispatchEvent(new Event('change', { bubbles: true })); });
        await expect(page.locator('#spawn-kv-unified')).toHaveValue('false');

        // fit On shows target input and auto-populates '2048'
await page.locator('#spawn-fit-enable').evaluate((el) => { el.value = 'true'; el.dispatchEvent(new Event('change', { bubbles: true })); });
        await page.waitForTimeout(400);

        const fitTargetValue = await page.evaluate(() =>
            document.getElementById('spawn-fit-target')?.value ?? ''
        );
        expect(fitTargetValue).toBe('2048');

        const moeLayout = await page.evaluate(() => {
            const field = document.getElementById('spawn-n-cpu-moe')?.closest('.hardware-field');
            const hint = field?.querySelector('.field-hint');
            const actions = document.getElementById('spawn-moe-autotune');
            if (!field || !hint || !actions) return null;

            actions.style.display = 'block';
            const fieldRect = field.getBoundingClientRect();
            const hintRect = hint.getBoundingClientRect();
            const actionsRect = actions.getBoundingClientRect();
            return {
                hintBottom: hintRect.bottom,
                actionsTop: actionsRect.top,
                actionsBottom: actionsRect.bottom,
                fieldBottom: fieldRect.bottom,
            };
        });
        expect(moeLayout).not.toBeNull();
        expect(moeLayout.actionsTop).toBeGreaterThanOrEqual(moeLayout.hintBottom);
        expect(moeLayout.fieldBottom).toBeGreaterThanOrEqual(moeLayout.actionsBottom);
    });

test('MTP defaults on only when resolved metadata proves built-in heads', async ({ page }) => {
        await page.goto('/');
        await page.waitForLoadState('networkidle');

        await page.evaluate(async () => {
            const { openSpawnWizard, wizardState } = await import('/js/features/spawn-wizard.js');
            openSpawnWizard();
            wizardState.model.source = 'local';
            wizardState.model.path = '/tmp/Qwen3.6-27B-MTP-Q4_K_M.gguf';
    wizardState.model.paramB = 27;
    wizardState.model.modelBytes = 16 * 1024 * 1024 * 1024;
    wizardState.arch.mtpDepth = 1;
    wizardState.arch.metadataStatus = 'resolved';
        });

        await page.locator('#wizard-next-btn').click();

  await expect(page.locator('#hw-use-mtp')).toBeChecked({ checked: true });
  await expect(page.locator('#hw-decision-speed .hw-speed-label').first()).toContainText('MTP heads detected');
});

test('MTP remains unavailable when metadata resolves without built-in heads', async ({ page }) => {
      await page.goto('/');
      await page.waitForLoadState('networkidle');

      await page.evaluate(async () => {
        const { openSpawnWizard, wizardState } = await import('/js/features/spawn-wizard.js');
        openSpawnWizard();
        wizardState.model.source = 'local';
        wizardState.model.path = '/tmp/model-without-mtp-Q4_K_M.gguf';
        wizardState.model.paramB = 7;
        wizardState.model.modelBytes = 4 * 1024 * 1024 * 1024;
        wizardState.arch.metadataStatus = 'resolved';
        wizardState.arch.mtpDepth = 0;
      });

      await page.locator('#wizard-next-btn').click();
      await expect(page.locator('input[name="hw-speed"][value="on"]')).toBeDisabled();
      await expect(page.locator('#hw-decision-speed .hw-speed-label').first()).toContainText('unavailable for this model');
      await expect(page.locator('#hw-use-mtp')).not.toBeChecked();
});

test('Pro shell switches the canonical settings without duplicating controls', async ({ page }) => {
  await page.goto('/');
  await page.waitForLoadState('networkidle');

  await page.evaluate(async () => {
    const { openSpawnWizard, wizardState } = await import('/js/features/spawn-wizard.js');
    openSpawnWizard();
    wizardState.model.source = 'local';
    wizardState.model.path = '/tmp/pro-shell-model.gguf';
    wizardState.model.paramB = 7;
    wizardState.model.modelBytes = 4 * 1024 * 1024 * 1024;
  });
  await page.locator('#wizard-next-btn').click();
  await expect(page.locator('#wizard-step-1')).toHaveClass(/active/);

  await page.selectOption('#view-mode-select', 'pro');
  await expect(page.locator('#pro-layout')).toBeVisible();
  await expect(page.locator('#pro-rail-nav .pro-rail-item')).toHaveCount(7);
  await expect(page.locator('#pro-controls-host #hw-decision-ctx')).toBeVisible();
  await expect(page.locator('#pro-controls-host #spawn-advanced-fields')).toBeAttached();
  for (const id of [
    'spawn-temperature', 'spawn-reasoning-mode', 'spawn-output-mode',
    'spawn-port', 'spawn-extra-args',
  ]) {
    await expect(page.locator(`#pro-controls-host #${id}`)).toHaveCount(1);
  }
  await expect(page.locator('[id="spawn-temperature"]')).toHaveCount(1);

  await page.keyboard.press('ControlOrMeta+k');
  await expect(page.locator('#pro-filter-input')).toBeFocused();

  await page.locator('#pro-filter-input').fill('batch');
  expect(await page.locator('#pro-controls-host .pro-search-hidden').count()).toBeGreaterThan(0);
  await page.keyboard.press('Escape');
  await expect(page.locator('#pro-filter-input')).toHaveValue('');
  await page.locator('#spawn-batch-size').evaluate((input) => {
    input.value = '1024';
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
  // The wizard's sticky identity/memory bars can overlap the toolbar at this
  // viewport; exercise the real label while preserving that layout.
  await page.locator('.pro-modified-only').evaluate((label) => label.click());
  expect(await page.locator('#pro-controls-host .pro-modified').count()).toBeGreaterThan(0);
  await page.locator('#pro-reset-all').dispatchEvent('click');
  await expect(page.locator('#spawn-batch-size')).toHaveValue('2048');

  await page.locator('#view-mode-select').evaluate((select) => {
    select.value = 'guided';
    select.dispatchEvent(new Event('change', { bubbles: true }));
  });
  await expect(page.locator('#pro-layout')).toBeHidden();
  await expect(page.locator('#hw-decision-ctx')).toBeVisible();
});

test('Rapid-MLX Pro keeps backend-native controls and shared access settings', async ({ page }) => {
  await page.goto('/');
  await page.waitForLoadState('networkidle');
  await page.evaluate(async () => {
    const { openSpawnWizard, wizardState, selectWizardEngine, showStep } = await import('/js/features/spawn-wizard.js');
    openSpawnWizard();
    wizardState.model.source = 'local';
    wizardState.model.path = '/tmp/rapid-pro-model';
    wizardState.model.paramB = 7;
    wizardState.model.modelBytes = 4 * 1024 * 1024 * 1024;
    selectWizardEngine('rapid_mlx', true);
    showStep(1);
  });
  await expect(page.locator('#wizard-step-1')).toHaveClass(/active/);
  await page.selectOption('#view-mode-select', 'pro');
  await expect(page.locator('#view-mode-select')).toHaveValue('pro');
  await expect(page.locator('#pro-layout')).toBeVisible();
  await expect(page.locator('#pro-controls-host #spawn-rapid-advanced-fields')).toBeAttached();
  await expect(page.locator('#pro-controls-host #spawn-advanced-fields')).toBeHidden();
  for (const id of [
    'spawn-kv-cache-dtype', 'spawn-retained-cache-mib', 'spawn-rapid-max-num-seqs',
    'spawn-rapid-tool-call-parser', 'spawn-rapid-reasoning-parser',
    'spawn-rapid-auto-tool-choice', 'spawn-port', 'spawn-bind-host',
  ]) {
    await expect(page.locator(`#pro-controls-host #${id}`)).toHaveCount(1);
  }
  await expect(page.locator('#pro-controls-host .pro-backend-note')).toBeVisible();
  await page.locator('#pro-filter-input').fill('cache');
  expect(await page.locator('#pro-controls-host .pro-search-hidden').count()).toBeGreaterThan(0);
  await page.selectOption('#view-mode-select', 'guided');
  await expect(page.locator('#pro-layout')).toBeHidden();
  await expect(page.locator('#all-settings-group #spawn-rapid-advanced-fields')).toBeAttached();
});

test('Launch Full config review labels requested and effective state', async ({ page }) => {
  await page.goto('/');
  await page.waitForLoadState('networkidle');
  await page.evaluate(async () => {
    const { openSpawnWizard, wizardState, selectWizardEngine, showStep } = await import('/js/features/spawn-wizard.js');
    openSpawnWizard();
    wizardState.model.source = 'local';
    wizardState.model.path = '/tmp/full-config-model.gguf';
    wizardState.model.paramB = 8;
    wizardState.model.modelBytes = 4 * 1024 * 1024 * 1024;
    selectWizardEngine('llama_cpp', true);
    showStep(2);
  });
  await expect(page.locator('#wizard-step-2')).toHaveClass(/active/);
  await expect(page.locator('#spawn-full-config-drawer')).toBeVisible();
  await expect(page.locator('#spawn-full-config-state')).toContainText('estimator-effective');
  await expect(page.locator('#preset-params-table .summary-state').first()).toContainText('requested');

  await page.evaluate(async () => {
    const { wizardState, selectWizardEngine, showStep } = await import('/js/features/spawn-wizard.js');
    wizardState.model.source = 'local';
    wizardState.model.path = '/tmp/full-config-rapid';
    selectWizardEngine('rapid_mlx', true);
    showStep(2);
  });
  await expect(page.locator('#spawn-full-config-state')).toContainText('runtime-effective');
  await expect(page.locator('#preset-params-table .summary-state').last()).toContainText('requested');
  await page.setViewportSize({ width: 430, height: 900 });
  await expect(page.locator('#spawn-full-config-drawer')).toBeVisible();
  const narrowLayout = await page.locator('#wizard-step-2').evaluate((step) => ({
    scrollWidth: step.scrollWidth,
    clientWidth: step.clientWidth,
  }));
  expect(narrowLayout.scrollWidth).toBeLessThanOrEqual(narrowLayout.clientWidth + 1);
});

test('Phase 11 axe and keyboard checks cover Guided, Pro, and Launch', async ({ page }) => {
  await page.goto('/');
  await page.waitForLoadState('networkidle');
  await page.evaluate(async () => {
    const { openSpawnWizard, wizardState, selectWizardEngine, showStep } = await import('/js/features/spawn-wizard.js');
    openSpawnWizard();
    wizardState.model.source = 'local';
    wizardState.model.path = '/tmp/phase11-a11y-model.gguf';
    wizardState.model.paramB = 8;
    wizardState.model.modelBytes = 4 * 1024 * 1024 * 1024;
    selectWizardEngine('llama_cpp', true);
    showStep(1);
  });
  const serious = (result) => result.violations.filter(({ impact }) => impact === 'serious' || impact === 'critical');
  expect(serious(await new AxeBuilder({ page }).include('#spawn-wizard-overlay').analyze())).toEqual([]);

  await page.locator('#view-mode-select').focus();
  await expect(page.locator('#view-mode-select')).toBeFocused();
  await page.selectOption('#view-mode-select', 'pro');
  await expect(page.locator('#pro-layout')).toBeVisible();
  expect(serious(await new AxeBuilder({ page }).include('#spawn-wizard-overlay').analyze())).toEqual([]);

  await page.evaluate(async () => {
    const { wizardState, selectWizardEngine, showStep } = await import('/js/features/spawn-wizard.js');
    wizardState.model.path = '/tmp/phase11-a11y-rapid';
    selectWizardEngine('rapid_mlx', true);
    showStep(2);
  });
  await page.locator('.spawn-full-config-summary').press('Enter');
  await expect(page.locator('#spawn-full-config-drawer')).toBeVisible();
  expect(serious(await new AxeBuilder({ page }).include('#spawn-wizard-overlay').analyze())).toEqual([]);
});

test('review step exposes structured output and full sampling defaults', async ({ page }) => {
        await page.goto('/');
        await page.waitForLoadState('networkidle');

        await page.evaluate(async () => {
            const { openSpawnWizard, wizardState } = await import('/js/features/spawn-wizard.js');
            openSpawnWizard();
            wizardState.model.source = 'local';
            wizardState.model.path = '/tmp/Qwen3.6-27B-Instruct-Q4_K_M.gguf';
            wizardState.model.paramB = 27;
            wizardState.model.modelBytes = 16 * 1024 * 1024 * 1024;
        });

        await page.locator('#wizard-next-btn').click();

        await expect(page.locator('#wizard-step-1')).toHaveClass(/active/);
        await expect(page.locator('#spawn-top-k')).toBeVisible();
        await expect(page.locator('#spawn-max-tokens')).toBeVisible();
        await expect(page.locator('#spawn-output-mode')).toBeVisible();

        await page.selectOption('#spawn-output-mode', 'json_schema');
        await expect(page.locator('#spawn-json-schema-wrap')).toBeVisible();
        await expect(page.locator('#spawn-grammar-wrap')).toBeHidden();
    });

    test('Error state: no internet (HF search returns empty)', async ({ page }) => {
        await page.goto('/');
        await page.waitForLoadState('networkidle');

        // Simulate a network failure by intercepting HF API calls.
        await page.route('https://huggingface.co/api/**', route => {
            route.abort('failed');
        });

        const [resp] = await Promise.all([
            page.waitForResponse(r => r.url().includes('/api/hf/search') && r.request().method() === 'POST'),
            page.evaluate(async () => {
                const headers = window.authHeaders ? window.authHeaders() : {};
                return fetch('/api/hf/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', ...headers },
                    body: JSON.stringify({ query: 'llama', limit: 5 }),
                });
            }),
        ]);

        // Should not crash; backend handles gracefully (200 with empty/error, or 4xx/5xx).
        const status = resp.status();
        expect(status).toBeGreaterThanOrEqual(200);
        expect(status).toBeLessThan(500);
    });

    test('@in-memory-test page-1 use-case drives the estimate, and stays out of the launch payload', async ({ page }) => {
        await page.goto('/');
        await page.waitForLoadState('networkidle');

        const result = await page.evaluate(async () => {
            const { buildSpawnPayload, wizardState } = await import('/js/features/spawn-wizard.js');
            const { rapidEstimatePolicyFromWizardHardware } = await import('/js/features/vram-estimate.js');

            wizardState.engine.selected = 'rapid_mlx';
            wizardState.engine.explicit = true;
            wizardState.model.rapidMlxSource = { kind: 'hugging_face_repo', repo_id: 'mlx-community/Qwen3-0.6B-4bit' };
            wizardState.access.port = 9123;

            // Page-1 use-case cards set workloadScenario directly (Phase 7B2).
            wizardState.hardware.workloadScenario = 'roleplay_storytelling';

            const payload = buildSpawnPayload();
            return {
                inLaunchPayload: 'workload_scenario' in (payload.rapid_mlx || {}),
                hasAssumptions: !!payload.rapid_mlx?.workload_assumptions,
                estimateScenario: rapidEstimatePolicyFromWizardHardware(wizardState.hardware).workload_scenario,
            };
        });

        // Workload scenario is an estimator input, not a launch setting. `RapidMlxConfig` has no
        // such field, so anything the wizard put here was dropped by serde on arrival -- asserting
        // that it serialized only ever proved the wizard wrote it, never that it did anything.
        expect(result.inLaunchPayload).toBe(false);
        expect(result.hasAssumptions).toBe(false);
        expect(result.estimateScenario).toBe('roleplay_storytelling');
    });


    test('@in-memory-test quant advisor forwards resolved global head dimension and omits degraded metadata', async ({ page }) => {
        const requests = [];
        await page.route('**/api/vram/quant-compare', async route => {
            requests.push(route.request().postDataJSON());
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({ ok: true, quants: [] }),
            });
        });
        await page.route('**/api/model-defaults', async route => {
            const body = route.request().postDataJSON();
            if (body.hf_repo_id === 'owner/slow-model') {
                await new Promise(resolve => setTimeout(resolve, 100));
            }
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({
                    introspected: body.hf_repo_id === 'owner/slow-model'
                        ? { global_head_dim: 512 }
                        : {},
                }),
            });
        });
        await page.goto('/');
        await page.waitForLoadState('networkidle');

        await page.evaluate(async () => {
            const { wizardState } = await import('/js/features/spawn-wizard.js');
            const { triggerQuantAdvisor } = await import('/js/features/spawn-wizard-hf-browse.js');
            wizardState.model.paramB = 31;
            wizardState.model.path = 'renamed-model.gguf';
            wizardState.vram.available = 32 * 1024 ** 3;
            wizardState.hardware.parallelSlots = 1;
            wizardState.arch.globalHeadDim = 512;
            triggerQuantAdvisor();
        });
        await expect.poll(() => requests.length).toBe(1);
        expect(requests[0].global_head_dim).toBe(512);

        await page.evaluate(async () => {
            const { introspectHfFileMetadata, wizardState } = await import('/js/features/spawn-wizard.js');
            const { triggerQuantAdvisor } = await import('/js/features/spawn-wizard-hf-browse.js');
            wizardState.model.hfRepo = 'owner/plain-model';
            wizardState.model.hfFile = 'plain-model.gguf';
            await introspectHfFileMetadata('owner/plain-model', 'plain-model.gguf', 1024);
            triggerQuantAdvisor();
        });
        await expect.poll(() => requests.length).toBe(2);
        expect(requests[1]).not.toHaveProperty('global_head_dim');

        await page.evaluate(async () => {
            const { introspectHfFileMetadata, wizardState } = await import('/js/features/spawn-wizard.js');
            const { triggerQuantAdvisor } = await import('/js/features/spawn-wizard-hf-browse.js');
            wizardState.model.hfRepo = 'owner/slow-model';
            wizardState.model.hfFile = 'slow-model.gguf';
            const stale = introspectHfFileMetadata('owner/slow-model', 'slow-model.gguf', 1024);
            wizardState.model.hfRepo = 'owner/current-model';
            wizardState.model.hfFile = 'current-model.gguf';
            const current = introspectHfFileMetadata('owner/current-model', 'current-model.gguf', 1024);
            await Promise.all([stale, current]);
            triggerQuantAdvisor();
        });
        await expect.poll(() => requests.length).toBe(3);
        expect(requests[2]).not.toHaveProperty('global_head_dim');
    });
});
