import { test, expect } from '@playwright/test';
import { dismissAuthShell } from '../helpers.js';

function preset(id, name, modelPath = `/models/${id}.gguf`) {
  return {
    id,
    name,
    model_path: modelPath,
    hf_repo: null,
    context_size: 8192,
    ctk: 'q8_0',
    ctv: 'q8_0',
    batch_size: 512,
    ubatch_size: 512,
    parallel_slots: 1,
    port: 8001,
    bind_host: '127.0.0.1',
  };
}

function rapidPreset(id = 'rapid', name = 'Rapid model') {
  return {
    id,
    name,
    backend: 'rapid_mlx',
    model_path: '',
    hf_repo: null,
    rapid_mlx: {
      model_path: '/models/mlx-community/Qwen3-4B-4bit',
      model_source: { kind: 'mlx_directory', path: '/models/mlx-community/Qwen3-4B-4bit' },
      model_source_view: {
        canonical_identity: '/models/mlx-community/Qwen3-4B-4bit',
        display_name: 'Qwen3-4B-4bit',
      },
      host: '127.0.0.1',
      port: 9123,
      served_model_name: 'qwen3-rapid',
      log_level: 'info',
    },
  };
}

// A v6 preset bundle: two weights artifacts (Q4/Q5) that differ only by
// quantization. The flat fields mirror what the server's
// `materialize_default_projection` would have written for `default_selection`,
// so the legacy one-artifact adapter renders from real data rather than a
// hand-tuned shape that only exists in this file.
function bundlePreset(id = 'bundled', name = 'Bundled preset', overrides = {}) {
  return {
    ...preset(id, name, '/models/qwen3-27b-q4_k_m.gguf'),
    context_size: 200704,
    ctk: 'q8_0',
    ctv: 'q8_0',
    batch_size: 2048,
    ubatch_size: 512,
    n_cpu_moe: 0,
    revision: 7,
    bundle: {
      identity: {
        bundle_id: 'qwen3-27b',
        tune_id: 'brainwaves-wfh',
        display_name: 'Qwen3.8 27B · Brainwaves WFH',
      },
      artifacts: [
        {
          id: 'art-q4',
          role: 'weights',
          display_name: 'Qwen3.8-27B-Q4_K_M.gguf',
          local_path: '/models/qwen3-27b-q4_k_m.gguf',
          hf_origin: null,
          size_bytes: 17179869184,
          digest: null,
          quantization: { value: 'q4_k_m', provenance: 'filename' },
          metadata: {},
          extra: {},
        },
        {
          id: 'art-q5',
          role: 'weights',
          display_name: 'Qwen3.8-27B-Q5_K_M.gguf',
          local_path: '/models/qwen3-27b-q5_k_m.gguf',
          hf_origin: null,
          size_bytes: 20401094656,
          digest: null,
          quantization: { value: 'q5_k_m', provenance: 'filename' },
          metadata: {},
          extra: {},
        },
      ],
      context_options: [160000, 200000, 262144],
      kv_policy_options: ['f16_f16', 'q8_0_q8_0', 'q4_0_q4_0'],
      performance_options: [
        { id: 'quality', label: 'Quality first', batch_size: 2048, ubatch_size: 512 },
        { id: 'balanced', label: 'Balanced', batch_size: 512, ubatch_size: 512 },
      ],
      n_cpu_moe_options: [0, 8, 16],
      selections: [],
      workload_policy: {},
      // No `intent_source`: this is an exact selection, not a fit intent.
      default_selection: {
        artifact_id: 'art-q4',
        context_size: 200704,
        kv_policy: 'q8_0_q8_0',
        performance_id: 'quality',
        n_cpu_moe: 0,
      },
      extra: {},
      ...(overrides.bundle || {}),
    },
    ...(overrides.preset || {}),
  };
}

async function installPresetMocks(page, options = {}) {
  const state = {
    presets: [...(options.presets || [preset('original', 'Original'), preset('other', 'Other')])],
    active: options.active || { status: 'Stopped', preset_id: '' },
    postCount: 0,
    putCount: 0,
    spawnPayloads: [],
    attachPayloads: [],
    resolvePayloads: [],
    selectionPayloads: [],
        cardsRequests: 0,
        selectionConflictUsed: false,
  };

  await page.route('**/api/settings', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ preset_id: options.savedPresetId || state.presets[0]?.id || '' }),
  }));

  await page.route('**/api/sessions/active/readiness', async route => {
    if (options.readinessDelay) await new Promise(resolve => setTimeout(resolve, options.readinessDelay));
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ ok: true, ready: true }),
    });
  });

  await page.route('**/api/sessions/active', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify(state.active),
  }));

  await page.route('**/api/sessions/recent', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ sessions: options.sessions || [], active_session_id: '' }),
  }));

  await page.route('**/api/sessions/remote-1', async route => {
    if (route.request().method() === 'DELETE') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ ok: true }),
      });
      return;
    }
    await route.continue();
  });

  await page.route('**/api/db/admin-token', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ token: 'admin-token' }),
  }));

  await page.route('**/api/kill-llama', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ ok: true }),
  }));

  await page.route('**/api/sessions/spawn', async route => {
    state.spawnPayloads.push(route.request().postDataJSON());
    if (options.spawnConflictOnce && state.spawnPayloads.length === 1) {
      // The saved selection moved under us; the server bumped the revision.
      state.presets = state.presets.map(p => (p.bundle ? { ...p, revision: (p.revision || 0) + 1 } : p));
      await route.fulfill({
        status: 409,
        contentType: 'application/json',
        body: JSON.stringify({
          ok: false,
          error: 'revision_conflict',
          current_revision: state.presets.find(p => p.bundle)?.revision || 0,
        }),
      });
      return;
    }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ok: true,
        session_id: 'spawned-session',
        backend: options.spawnBackend || 'llama_cpp',
        port: options.spawnPort || 8001,
      }),
    });
  });

  await page.route('**/api/attach', async route => {
    state.attachPayloads.push(route.request().postDataJSON());
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ ok: true }),
    });
  });

  const unifiedMemory = options.unifiedMemory !== false;

  await page.route('**/metrics/system', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ ram_total_gb: 128, ram_used_gb: 40, memory_reclaimable_gb: 8 }),
  }));

  await page.route('**/metrics/gpu', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify(unifiedMemory
      ? { gpus: [{ vram_total_mb: 131072, vram_used_mb: 40960, metal_gpu_limit_mb: 114688 }] }
      : { gpus: [{ vram_total_mb: 24576, vram_used_mb: 2048 }] }),
  }));

  await page.route('**/api/system/metal-gpu-limit', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ ok: true, limit_mb: unifiedMemory ? 114688 : 0 }),
  }));

  await page.route('**/api/llama-binary/platform-info', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ ok: true, auto_backend: unifiedMemory ? 'metal' : 'cuda' }),
  }));

  // Architecture invariant 19: the card's availability line must come from a
  // live read of this endpoint, never a value frozen at save time.
  await page.route('**/api/memory-availability', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({
      ok: true,
      snapshot: {
        current_safe_availability_bytes: options.availabilityBytes ?? 29_000_000_000,
        configured_ceiling_bytes: 120_259_084_288,
        total_unified_bytes: 137_438_953_472,
      },
    }),
  }));

  await page.route('**/api/vram-estimate', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({
      ok: true,
      total_bytes: 26_300_000_000,
      weights_bytes: 17_179_869_184,
      kv_cache_bytes: 8_000_000_000,
      overhead_bytes: 1_120_130_816,
    }),
  }));

  await page.route(/\/api\/(?:presets|preset-cards)/, async route => {
    const request = route.request();
    const url = new URL(request.url());
    const method = request.method();
    const parts = url.pathname.split('/').filter(Boolean);
    const id = parts.length === 3 ? decodeURIComponent(parts[2]) : '';

    if (url.pathname === '/api/preset-cards' && method === 'GET') {
      state.cardsRequests += 1;
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          cards: state.presets.map(p => ({
            id: p.id,
            name: p.name,
            revision: p.revision || 0,
            artifacts: (p.bundle?.artifacts || []).map(a => ({
              id: a.id,
              role: a.role,
              display_name: a.display_name,
              available: a.local_path != null,
              quantization: a.quantization.value,
            })),
          })),
          catalog_etag: 'catalog-v1:test',
          // Architecture invariant 16: the render kill-switch reaches the UI
          // only as a closed enum resolved server-side.
          preset_bundle_ui: options.presetBundleUi || 'bundled',
        }),
      });
      return;
    }

    if (url.pathname.endsWith('/resolve') && method === 'POST') {
      const body = request.postDataJSON() || {};
      state.resolvePayloads.push(body);
      const source = state.presets.find(p => p.id === parts[2]);
      const requestedSelection = body.selection || source?.bundle?.default_selection || null;
      const resolved = options.resolveResponse
        ? await options.resolveResponse({ body, source, selection: requestedSelection, state })
        : {};
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          ok: true,
          selection: resolved.selection || requestedSelection,
          changes: resolved.changes || [],
          estimate: resolved.estimate || null,
          capability_reasons: resolved.capability_reasons || [],
          evidence: resolved.evidence || null,
          selection_hash: resolved.selection_hash || 'sel-v1:test',
          resolved_config_hash: resolved.resolved_config_hash || 'cfg-v1:test',
          revision: source?.revision || 0,
          ...resolved,
        }),
      });
      return;
    }

        if (url.pathname.endsWith('/selection') && method === 'PATCH') {
            const body = request.postDataJSON();
            state.selectionPayloads.push(body);
            const index = state.presets.findIndex(p => p.id === parts[2]);
            if (index >= 0) {
                const current = state.presets[index];
                if (options.selectionConflictOnce && !state.selectionConflictUsed) {
                    state.selectionConflictUsed = true;
                    state.presets[index] = { ...current, revision: (current.revision || 0) + 1 };
                    await route.fulfill({
                        status: 409,
                        contentType: 'application/json',
                        body: JSON.stringify({ ok: false, code: 'revision_conflict', error: 'preset changed elsewhere' }),
                    });
                    return;
                }
                state.presets[index] = {
          ...current,
          revision: (current.revision || 0) + 1,
          bundle: {
            ...current.bundle,
            default_selection: { ...(body.selection || {}) },
          },
        };
      }
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          ok: true,
          preset: state.presets[index],
          revision: state.presets[index]?.revision || 0,
        }),
      });
      return;
    }

    if (url.pathname === '/api/presets' && method === 'GET') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(state.presets),
      });
      return;
    }

    if (url.pathname === '/api/presets' && method === 'POST') {
      state.postCount += 1;
      const body = request.postDataJSON();
      const created = { ...body, id: body.id || (body.name === 'Wizard preset' ? 'wizard-id' : `copy-${state.postCount}`) };
      state.presets.push(created);
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ ok: true, preset: created }),
      });
      return;
    }

    if (url.pathname.endsWith('/copy') && method === 'POST') {
      state.postCount += 1;
      const body = request.postDataJSON();
      const sourceId = parts[2];
      const source = state.presets.find(p => p.id === sourceId);
      const created = { ...source, name: body.new_name, id: `copy-${state.postCount}`, revision: 1 };
      state.presets.push(created);
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ ok: true, preset: created }),
      });
      return;
    }

    if (id && method === 'GET') {
      const found = state.presets.find(p => p.id === id);
      await route.fulfill({
        status: found ? 200 : 404,
        contentType: 'application/json',
        body: JSON.stringify(found ? { ok: true, preset: found } : { ok: false, error: 'preset not found' }),
      });
      return;
    }

    if (id && method === 'PUT') {
      state.putCount += 1;
      const body = { ...request.postDataJSON(), id };
      const index = state.presets.findIndex(p => p.id === id);
      if (index >= 0) state.presets[index] = body;
      else state.presets.push(body);
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ ok: true, preset: body }),
      });
      return;
    }

    await route.continue();
  });

  return state;
}

async function boot(page) {
  await page.goto('/');
  await page.waitForSelector('html.modules-ready');
  await dismissAuthShell(page);
  await expect(page.locator('#preset-select option')).not.toHaveCount(0);
}

test.describe('preset flow', () => {
  test('welcome recent list keeps remote endpoints only and supports dismissal', async ({ page }) => {
    await installPresetMocks(page, {
      sessions: [
        {
          id: 'remote-1',
          name: 'Remote endpoint',
          backend: 'llama_cpp',
          status: 'Disconnected',
          mode: { Attach: { endpoint: 'http://192.168.2.16:8001' } },
          last_connected_at: 1,
        },
        {
          id: 'local-1',
          name: 'Local preset launch',
          backend: 'rapid_mlx',
          status: 'Stopped',
          mode: { Spawn: { preset_id: 'original' } },
          preset_id: 'original',
          last_connected_at: 2,
        },
      ],
    });
    await boot(page);
    await expect(page.locator('#setup-filter-sort')).toHaveValue('last_launched');
    await expect(page.locator('#setup-endpoint-list .setup-endpoint-card')).toHaveCount(1);
    await expect(page.locator('.setup-endpoint-dismiss')).toHaveCount(1);
    await page.locator('.setup-endpoint-dismiss').click();
    await expect(page.locator('#setup-endpoint-list .setup-endpoint-card')).toHaveCount(0);
    await expect(page.locator('#setup-recent-endpoints')).toBeHidden();
  });

  test('duplicating from the preset editor selects and reopens the copy', async ({ page }) => {
    await installPresetMocks(page, {
      presets: [preset('original', 'Original'), preset('other', 'Other')],
      savedPresetId: 'original',
    });
    await boot(page);

    await page.evaluate(async () => {
      const { openPresetModal } = await import('/js/features/presets.js');
      openPresetModal('edit');
    });
    await expect(page.locator('#modal-name')).toHaveValue('Original');

    await page.locator('#preset-modal-duplicate').click();
    await expect(page.locator('#modal-name')).toHaveValue('Original (copy)');
    await expect(page.locator('#preset-select')).toHaveValue('copy-1');
    await expect(page.locator('#setup-preset-select')).toHaveValue('copy-1');

    await page.locator('#preset-modal-close').click();
    await page.evaluate(async () => {
      const { openPresetModal } = await import('/js/features/presets.js');
      openPresetModal('edit');
    });
    await expect(page.locator('#modal-name')).toHaveValue('Original (copy)');
  });

  test('selecting a different preset while running prompts and spawns the selected preset', async ({ page }) => {
    const state = await installPresetMocks(page, {
      presets: [preset('original', 'Original'), preset('other', 'Other')],
      active: { status: 'Running', preset_id: 'original' },
      savedPresetId: 'original',
    });
    page.on('dialog', dialog => dialog.accept());
    await boot(page);

    await page.locator('#preset-select').selectOption('other');

    // Wait for the restart confirmation toast and click "Restart Now"
    const restartBtn = page.locator('.toast-with-actions [data-action="restart"]');
    await expect(restartBtn).toBeVisible({ timeout: 5000 });
    await restartBtn.click();

    await expect.poll(() => state.spawnPayloads.length).toBe(1);
    expect(state.spawnPayloads[0].preset_id).toBe('other');
  });

  test('spawn wizard save records the created preset id and updates on the second save', async ({ page }) => {
    const state = await installPresetMocks(page, {
      presets: [preset('original', 'Original')],
      savedPresetId: 'original',
    });
    await boot(page);

    await page.evaluate(async () => {
      const { openSpawnWizard } = await import('/js/features/spawn-wizard.js');
      openSpawnWizard({ localPath: '/models/wizard.gguf' });
    });
    await expect.poll(() => page.evaluate(() => Boolean(
      window.wizardState
      && document.getElementById('spawn-preset-name-input')
      && document.getElementById('spawn-save-preset-btn'),
    ))).toBe(true);
    await page.evaluate(() => {
      const name = document.getElementById('spawn-preset-name-input');
      name.value = 'Wizard preset';
      name.dispatchEvent(new Event('input', { bubbles: true }));
      document.getElementById('spawn-save-preset-btn').dispatchEvent(
        new MouseEvent('click', { bubbles: true, cancelable: true }),
      );
    });
    await expect.poll(() => state.postCount).toBe(1);
    await expect(page.locator('#spawn-save-preset-btn')).toHaveText('Save as Preset');

    await page.evaluate(() => {
      document.getElementById('spawn-save-preset-btn').dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    });
    await expect.poll(() => state.putCount).toBe(1);
    expect(state.presets.filter(p => p.name === 'Wizard preset')).toHaveLength(1);
    expect(state.presets.find(p => p.name === 'Wizard preset')?.id).toBe('wizard-id');
  });

  test('Rapid welcome start sends preset identity only and shows the resolved server port', async ({ page }) => {
    const state = await installPresetMocks(page, {
      presets: [rapidPreset()],
      savedPresetId: 'rapid',
      spawnBackend: 'rapid_mlx',
      spawnPort: 9123,
      readinessDelay: 250,
    });
    await boot(page);

    await expect(page.locator('#preset-select')).toHaveValue('rapid');
    const start = page.evaluate(async () => {
      const { doStart } = await import('/js/features/attach-detach.js');
      await doStart();
    });

    await expect(page.locator('.toast').filter({ hasText: 'Loading model on port 9123' })).toBeVisible();
    await start;
    expect(state.spawnPayloads).toEqual([{ preset_id: 'rapid' }]);
  });

  test('Rapid editor save and duplicate preserve backend-owned configuration', async ({ page }) => {
    const original = rapidPreset();
    const rapidConfig = structuredClone(original.rapid_mlx);
    const state = await installPresetMocks(page, {
      presets: [original],
      savedPresetId: 'rapid',
    });
    await boot(page);

    await page.evaluate(async () => {
      const { openPresetModal } = await import('/js/features/presets.js');
      openPresetModal('edit');
    });
    await expect(page.locator('#modal-model-path')).toHaveValue(rapidConfig.model_path);
    // Naming the sections rather than counting them: the point is that the llama.cpp-only
    // sections stay hidden, and a bare count silently accepts the wrong three. Generation is
    // here deliberately -- the Rapid sampling defaults live in it, and while it was hidden the
    // editor rewrote them to nulls on every save because the user could not reach the inputs.
    await expect(page.locator('.preset-nav-item:visible')).toHaveCount(4);
    expect(
      await page.locator('.preset-nav-item:visible').evaluateAll(
        els => els.map(e => e.dataset.section),
      ),
    ).toEqual(['model', 'generation', 'context', 'advanced']);
    await page.locator('#modal-name').fill('Rapid renamed');
    await page.locator('.preset-nav-item[data-section="advanced"]').click();
    await page.locator('#modal-port').fill('9234');

    // Submit form → savePreset shows change summary → click Confirm Save via JS
    await page.evaluate(() => document.getElementById('preset-form')?.requestSubmit());
    // Change summary appears, saveBtn becomes "Confirm Save"
    await expect(page.locator('#btn-modal-save')).toHaveText('Confirm Save');
    // Force click via JS dispatch to bypass Playwright visibility checks
    await page.evaluate(() => document.getElementById('btn-modal-save').click());

    // Wait for save to complete (PUT to /api/presets/:id)
    await expect.poll(async () => state.putCount).toBe(1);
    expect(state.presets.find(p => p.id === 'rapid')?.rapid_mlx).toMatchObject({
      ...rapidConfig,
      port: 9234,
    });
    expect(state.presets.find(p => p.id === 'rapid')?.name).toBe('Rapid renamed');

    const editedRapidConfig = structuredClone(state.presets.find(p => p.id === 'rapid').rapid_mlx);

    await page.evaluate(async () => {
      const { openPresetModal } = await import('/js/features/presets.js');
      openPresetModal('edit');
    });
    await page.locator('#preset-modal-duplicate').click();
    await expect.poll(() => state.postCount).toBe(1);
    expect(state.presets.find(p => p.id === 'copy-1')?.rapid_mlx).toEqual(editedRapidConfig);
  });

  test('local spawn sessions are not duplicated in recent remote endpoints', async ({ page }) => {
    await installPresetMocks(page, {
      sessions: [{
        id: 'protected-session',
        name: 'Private model',
        backend: 'llama_cpp',
        launch_requires_api_key: true,
        launch: {
          backend: 'llama_cpp',
          config: { model_path: '/models/private.gguf', port: 8001 },
        },
        mode: { Spawn: { port: 8001 } },
        status: 'Stopped',
      }],
    });
    await boot(page);
    await expect(page.locator('.setup-spawn-restore-card')).toHaveCount(0);
    await expect(page.locator('#setup-recent-endpoints')).toBeHidden();
  });

  test('protected attached endpoint reconnects with a transient native prompt key', async ({ page }) => {
    const state = await installPresetMocks(page, {
      sessions: [{
        id: 'protected-attach',
        name: 'Protected endpoint',
        launch_requires_api_key: true,
        mode: { Attach: { endpoint: 'http://127.0.0.1:9000' } },
        status: 'Disconnected',
      }],
    });
    await boot(page);

    await page.locator('.setup-endpoint-card .setup-endpoint-connect').click();
    const prompt = page.getByRole('dialog', { name: 'Reconnect to protected endpoint' });
    await expect(prompt).toBeVisible();
    await prompt.locator('input[type="password"]').fill('attach-transient');
    await prompt.getByRole('button', { name: 'Reconnect' }).click();

    await expect.poll(() => state.attachPayloads.length).toBe(1);
    expect(state.attachPayloads[0]).toEqual({
      endpoint: 'http://127.0.0.1:9000',
      api_key: 'attach-transient',
      backend: 'llama_cpp',
    });
  });

  // ── Phase 8a: compact preset-bundle launch card ────────────────────────────

  test('a two-artifact bundle renders one card with the tune title and saved selection chips', async ({ page }) => {
    await installPresetMocks(page, { presets: [bundlePreset()] });
    await boot(page);

    // Q4 and Q5 are alternatives within one bundle, not two launchable presets.
    // The grid always appends a "New model" onboarding affordance (launch-card--new)
    // when few presets exist, so count only real preset cards.
    const cards = page.locator('#setup-launch-grid .launch-card:not(.launch-card--new)');
    await expect(cards).toHaveCount(1);

    const card = cards.first();
    await expect(card).toHaveAttribute('data-bundle-id', 'qwen3-27b');
    await expect(card).toHaveAttribute('data-revision', '7');
    await expect(card.locator('.launch-card-name')).toHaveText('Qwen3.8 27B · Brainwaves WFH');
    await expect(card.locator('.launch-card-tune')).toHaveText('Quality first');

    // Chips summarize the saved default selection, not the bundle's options.
    const chips = card.locator('.launch-card-chips .launch-chip');
    await expect(chips).toHaveText([/196k context/, /q8_0\/q8_0/, /Q4_K_M/]);

    // The collapsed card never exposes the resolved-config hash (drawer only).
    await expect(card).not.toContainText('cfg-v1:');
  });

  test('starting a bundle sends only the preset id and expected revision', async ({ page }) => {
    const state = await installPresetMocks(page, { presets: [bundlePreset()], spawnPort: 9123 });
    await boot(page);

    await page.locator('.launch-card[data-preset-id="bundled"] .launch-card-btn-start').click();

    await expect.poll(() => state.spawnPayloads.length).toBe(1);
    // Resolve-and-launch is server-owned: the client must not ship a selection.
    expect(state.spawnPayloads[0]).toEqual({ preset_id: 'bundled', expected_revision: 7 });
    expect(state.resolvePayloads).toHaveLength(0);
  });

  test('a revision conflict re-renders and re-asks instead of retrying', async ({ page }) => {
    const state = await installPresetMocks(page, {
      presets: [bundlePreset()],
      spawnPort: 9123,
      spawnConflictOnce: true,
    });
    await boot(page);

    const card = page.locator('.launch-card[data-preset-id="bundled"]');
    await card.locator('.launch-card-btn-start').click();

    await expect(page.locator('.toast').filter({ hasText: /changed/i })).toBeVisible();
    // One shot only — no silent retry with the stale revision.
    await expect.poll(() => state.spawnPayloads.length).toBe(1);
    await expect(card).toHaveAttribute('data-revision', '8');

    // The user re-asks explicitly, and the fresh revision goes out.
    await card.locator('.launch-card-btn-start').click();
    await expect.poll(() => state.spawnPayloads.length).toBe(2);
    expect(state.spawnPayloads[1]).toEqual({ preset_id: 'bundled', expected_revision: 8 });
  });

  test('a bundle whose selected artifact has no local file degrades to setup before spawning', async ({ page }) => {
    const state = await installPresetMocks(page, {
      presets: [bundlePreset('bundled', 'Bundled preset', {
        preset: { model_path: '' },
        bundle: {
          artifacts: [{
            id: 'art-q4',
            role: 'weights',
            display_name: 'Qwen3.8-27B-Q4_K_M.gguf',
            local_path: null,
            hf_origin: 'Qwen/Qwen3.8-27B-GGUF',
            size_bytes: 17179869184,
            digest: null,
            quantization: { value: 'q4_k_m', provenance: 'filename' },
            metadata: {},
            extra: {},
          }],
        },
      })],
    });
    await boot(page);

    const start = page.locator('.launch-card[data-preset-id="bundled"] .launch-card-btn-start');
    await expect(start).toHaveClass(/launch-card-btn-start--configure/);
    await expect(start).toHaveText(/Set up model/);

    await start.click();
    await expect(page).toHaveURL(/\/spawn/);
    expect(state.spawnPayloads).toHaveLength(0);
  });

  test('the availability line comes from a live read on unified memory', async ({ page }) => {
    await installPresetMocks(page, { presets: [bundlePreset()], unifiedMemory: true });
    await boot(page);

    const vram = page.locator('.launch-card[data-preset-id="bundled"] .launch-card-vram');
    await expect(vram).not.toHaveClass(/launch-card-vram--loading/);
    // Architecture invariant 19: the verdict carries the timestamp of the read
    // it was computed against, so a stale card is visibly stale.
    const readAt = await vram.getAttribute('data-availability-read-at');
    expect(readAt).toBeTruthy();
    expect(Number.isNaN(Date.parse(readAt))).toBe(false);
  });

  test('the availability line comes from the same live read on discrete VRAM', async ({ page }) => {
    await installPresetMocks(page, { presets: [bundlePreset()], unifiedMemory: false });
    await boot(page);

    const vram = page.locator('.launch-card[data-preset-id="bundled"] .launch-card-vram');
    await expect(vram).not.toHaveClass(/launch-card-vram--loading/);
    // The discrete branch used to derive availability from /metrics/gpu alone.
    const readAt = await vram.getAttribute('data-availability-read-at');
    expect(readAt).toBeTruthy();
    expect(Number.isNaN(Date.parse(readAt))).toBe(false);
  });

  test('a one-artifact bundle renders through the same card as a flat preset', async ({ page }) => {
    const state = await installPresetMocks(page, {
      presets: [
        bundlePreset('single', 'Single artifact', {
          bundle: {
            artifacts: [{
              id: 'art-q4',
              role: 'weights',
              display_name: 'Qwen3.8-27B-Q4_K_M.gguf',
              local_path: '/models/qwen3-27b-q4_k_m.gguf',
              hf_origin: null,
              size_bytes: 17179869184,
              digest: null,
              quantization: { value: 'q4_k_m', provenance: 'filename' },
              metadata: {},
              extra: {},
            }],
          },
        }),
        preset('flat', 'Flat preset'),
      ],
    });
    await boot(page);

    await expect(page.locator('#setup-launch-grid .launch-card:not(.launch-card--new)')).toHaveCount(2);
    const bundled = page.locator('.launch-card[data-preset-id="single"]');
    await expect(bundled.locator('.launch-card-btn-configure')).toBeVisible();

    // The flat preset keeps its legacy payload exactly.
    await page.locator('.launch-card[data-preset-id="flat"] .launch-card-btn-start').click();
    await expect.poll(() => state.spawnPayloads.length).toBe(1);
    expect(state.spawnPayloads[0]).toEqual({ preset_id: 'flat' });
  });

  test('the legacy render flag flattens a bundle through the legacy card path', async ({ page }) => {
    const state = await installPresetMocks(page, {
      presets: [bundlePreset()],
      presetBundleUi: 'legacy',
    });
    await boot(page);

    const card = page.locator('.launch-card[data-preset-id="bundled"]');
    // Legacy mode is the pre-bundle card: preset name, Edit, no bundle affordances.
    await expect(card.locator('.launch-card-name')).toHaveText('Bundled preset');
    await expect(card.locator('.launch-card-btn-edit')).toBeVisible();
    await expect(card.locator('.launch-card-btn-configure')).toHaveCount(0);
    await expect(card.locator('.launch-card-tune')).toHaveCount(0);

    // ...and the legacy start payload, with no revision handshake.
    await card.locator('.launch-card-btn-start').click();
    await expect.poll(() => state.spawnPayloads.length).toBe(1);
    expect(state.spawnPayloads[0]).toEqual({ preset_id: 'bundled' });
  });

  test('Configure opens the bundle drawer and restores focus on close', async ({ page }) => {
    const pageErrors = [];
    page.on('pageerror', error => pageErrors.push(error.message));
    await installPresetMocks(page, { presets: [bundlePreset()] });
    await boot(page);

    const configure = page.locator('.launch-card[data-preset-id="bundled"] .launch-card-btn-configure');
    await configure.click();
    expect(pageErrors, pageErrors.join('\n')).toEqual([]);
    const drawer = page.locator('#bundle-drawer');
    await expect(drawer).toHaveClass(/open/);
    await expect(drawer.locator('[role="dialog"]')).toHaveAttribute('aria-labelledby', 'bundle-drawer-title');
    await expect(drawer.locator('#bundle-drawer-title')).toHaveText('Configure Qwen3.8 27B · Brainwaves WFH');
    await expect.poll(() => page.evaluate(() => document.activeElement?.className)).toBe('bundle-drawer-close');

    await drawer.locator('.bundle-drawer-close').click();
    await expect(drawer).toBeHidden();
    await expect.poll(() => page.evaluate(() => document.activeElement?.classList.contains('launch-card-btn-configure'))).toBe(true);
  });

  test('drawer edits stay draft-only and Reset is the zero-friction discard', async ({ page }) => {
    await installPresetMocks(page, { presets: [bundlePreset()] });
    await boot(page);

    const card = page.locator('.launch-card[data-preset-id="bundled"]');
    await card.locator('.launch-card-btn-configure').click();
    const drawer = page.locator('#bundle-drawer');
    const q5 = drawer.locator('input[value="art-q5"]');
    await expect(q5).toBeVisible();
    await q5.check();
    await expect(drawer.locator('.bundle-quant[value="art-q5"]')).toBeChecked();
    await expect(card.locator('.launch-card-chips')).toContainText('Q4_K_M');
    await expect(card.locator('.launch-card-chips')).not.toContainText('Q5_K_M');
    await expect(drawer).toHaveClass(/is-dirty/);

    await drawer.locator('.bundle-reset').click();
    await expect(drawer).not.toHaveClass(/is-dirty/);
    await expect(drawer.locator('input[value="art-q4"]')).toBeChecked();
  });

  test('disabled bundle choices remain visible with an accessible backend reason', async ({ page }) => {
    await installPresetMocks(page, {
      presets: [bundlePreset()],
      resolveResponse: async () => ({
        capability_reasons: [{
          field: 'kv_policy',
          value: 'q4_0_q4_0',
          reason: 'Agentic / tool use requires the quality KV floor.',
        }],
      }),
    });
    await boot(page);

    await page.locator('.launch-card[data-preset-id="bundled"] .launch-card-btn-configure').click();
    const option = page.locator('.bundle-kv[value="q4_0_q4_0"]');
    await expect(option).toBeVisible();
    await expect(option).toBeDisabled();
    const describedBy = await option.getAttribute('aria-describedby');
    expect(describedBy).toBeTruthy();
    await expect(page.locator(`#${describedBy}`)).toHaveText('Agentic / tool use requires the quality KV floor.');
  });

  test('only the newest resolve response can update the draft', async ({ page }) => {
    await installPresetMocks(page, {
      presets: [bundlePreset()],
      resolveResponse: async ({ selection }) => {
        if (selection.context_size === 160000) await new Promise(resolve => setTimeout(resolve, 150));
        return { selection };
      },
    });
    await boot(page);

    await page.locator('.launch-card[data-preset-id="bundled"] .launch-card-btn-configure').click();
    const drawer = page.locator('#bundle-drawer');
    await drawer.locator('.bundle-ctx[data-context="160000"]').click();
    await drawer.locator('.bundle-ctx[data-context="200000"]').click();
    await expect(drawer.locator('.bundle-ctx[data-context="200000"]')).toHaveClass(/is-active/);
    await expect(drawer.locator('.bundle-ctx[data-context="160000"]')).not.toHaveClass(/is-active/);
  });

  test('Fit automatically is a draft probe result and headroom forwards fit_target_mib', async ({ page }) => {
    const moeArtifacts = bundlePreset().bundle.artifacts.map(artifact => ({
      ...artifact,
      metadata: artifact.id === 'art-q4' ? { model_kind: 'moe', moe_layer_count: 64 } : artifact.metadata,
    }));
    const state = await installPresetMocks(page, {
      presets: [bundlePreset('moe', 'MoE bundle', { bundle: { artifacts: moeArtifacts } })],
      resolveResponse: async ({ body, selection }) => body.fit_automatically
        ? {
          selection: { ...selection, n_cpu_moe: 12 },
          estimate: {
            status: 'available',
            estimate: {
              total_bytes: 26_300_000_000,
              weights_bytes: 17_179_869_184,
              kv_cache_bytes: 8_000_000_000,
              headroom_bytes: 1_500_000_000,
              probe_device_total_mib: 25_684,
              ram_bytes: 5_568 * 1024 * 1024,
            },
          },
        }
        : {},
    });
    await boot(page);

    await page.locator('.launch-card[data-preset-id="moe"] .launch-card-btn-configure').click();
    const drawer = page.locator('#bundle-drawer');
    await drawer.locator('.bundle-moe[data-moe="fit"]').click();
    await expect.poll(() => state.resolvePayloads.some(payload => payload.fit_automatically === true)).toBe(true);
    await expect(drawer.locator('.bundle-moe-probe')).toContainText('Device VRAM 25.1 GB');
    await expect(drawer.locator('.bundle-result-moe')).toContainText('slower generation');
    await expect(state.selectionPayloads).toHaveLength(0);

    await drawer.locator('.bundle-moe-headroom').fill('4');
    await expect.poll(() => state.resolvePayloads.at(-1)?.fit_target_mib).toBe(4096);
  });

  test('Start without saving sends the normalized draft and does not persist', async ({ page }) => {
    const state = await installPresetMocks(page, { presets: [bundlePreset()] });
    await boot(page);
    await page.locator('.launch-card[data-preset-id="bundled"] .launch-card-btn-configure').click();
    const drawer = page.locator('#bundle-drawer');
    const q5 = drawer.locator('input[value="art-q5"]');
    await expect(q5).toBeVisible();
    await q5.check();
    await drawer.locator('.bundle-start-draft').click();

    await expect.poll(() => state.spawnPayloads.length).toBe(1);
    expect(state.selectionPayloads).toHaveLength(0);
    expect(state.spawnPayloads[0]).toMatchObject({
      preset_id: 'bundled',
      expected_revision: 7,
      selection: { artifact_id: 'art-q5' },
    });
  });

  test('Save & Start persists before launching the returned revision', async ({ page }) => {
    const state = await installPresetMocks(page, { presets: [bundlePreset()] });
    await boot(page);
    await page.locator('.launch-card[data-preset-id="bundled"] .launch-card-btn-configure').click();
    const drawer = page.locator('#bundle-drawer');
    const q5 = drawer.locator('input[value="art-q5"]');
    await expect(q5).toBeVisible();
    await q5.check();
    await drawer.locator('.bundle-save-start').click();

    await expect.poll(() => state.selectionPayloads.length).toBe(1);
    await expect.poll(() => state.spawnPayloads.length).toBe(1);
    expect(state.spawnPayloads[0]).toMatchObject({
      preset_id: 'bundled',
      expected_revision: 8,
      selection: { artifact_id: 'art-q5' },
    });
  });

test('drawer revision conflict offers Reload without overwriting current state', async ({ page }) => {
    const state = await installPresetMocks(page, {
        presets: [bundlePreset()],
        selectionConflictOnce: true,
    });
    await boot(page);

    const drawer = page.locator('#bundle-drawer');
    await page.locator('.launch-card[data-preset-id="bundled"] .launch-card-btn-configure').click();
    await drawer.locator('input[value="art-q5"]').check();
    await drawer.locator('.bundle-save').click();

    const toast = page.locator('.toast-with-actions').filter({ hasText: 'Preset changed elsewhere' });
    await expect(toast).toBeVisible();
    const reload = toast.locator('[data-action="reload"]');
    await expect(reload).toHaveText('Reload');
    await reload.click();

    await expect(drawer).not.toHaveClass(/is-dirty/);
    await expect(drawer.locator('input[value="art-q4"]')).toBeChecked();
    await expect(drawer.locator('input[value="art-q5"]')).not.toBeChecked();
    expect(state.selectionPayloads).toHaveLength(1);
    expect(state.presets[0].revision).toBe(8);
});

test('dirty close control requires confirmation', async ({ page }) => {
    await installPresetMocks(page, { presets: [bundlePreset()] });
    await boot(page);
    const configure = page.locator('.launch-card[data-preset-id="bundled"] .launch-card-btn-configure');
    const drawer = page.locator('#bundle-drawer');
    await configure.click();
    const q5 = drawer.locator('input[value="art-q5"]');
    await expect(q5).toBeVisible();
    await q5.check();
    let promptCount = 0;
    page.once('dialog', async dialog => {
      promptCount += 1;
      await dialog.accept();
    });
    await drawer.locator('.bundle-drawer-close').click();
    await expect(drawer).toBeHidden();
    expect(promptCount).toBe(1);
  });

  test('dirty Escape requires confirmation', async ({ page }) => {
    await installPresetMocks(page, { presets: [bundlePreset()] });
    await boot(page);
    await page.locator('.launch-card[data-preset-id="bundled"] .launch-card-btn-configure').click();
    const drawer = page.locator('#bundle-drawer');
    const q5 = drawer.locator('input[value="art-q5"]');
    await expect(q5).toBeVisible();
    await q5.check();
    page.once('dialog', dialog => dialog.accept());
    await page.keyboard.press('Escape');
    await expect(drawer).toBeHidden();
  });

  test('dirty backdrop dismissal requires confirmation', async ({ page }) => {
    await installPresetMocks(page, { presets: [bundlePreset()] });
    await boot(page);
    await page.locator('.launch-card[data-preset-id="bundled"] .launch-card-btn-configure').click();
    const drawer = page.locator('#bundle-drawer');
    const q5 = drawer.locator('input[value="art-q5"]');
    await expect(q5).toBeVisible();
    await q5.check();
    page.once('dialog', dialog => dialog.accept());
    await drawer.locator('.bundle-drawer-backdrop').click({ position: { x: 8, y: 8 } });
    await expect(drawer).toBeHidden();
  });

  test('Custom is a derived indicator while workload is a selectable draft with a diff', async ({ page }) => {
    const state = await installPresetMocks(page, {
      presets: [bundlePreset('bundled', 'Bundled preset', { bundle: { workload_policy: 'general_chat' } })],
      resolveResponse: async ({ body, selection }) => body.selection?.intent_source
        ? { selection: { ...selection, performance_id: 'manual', batch_size: 1000, ubatch_size: 250 } }
        : { selection },
    });
    await boot(page);
    await page.locator('.launch-card[data-preset-id="bundled"] .launch-card-btn-configure').click();
    const drawer = page.locator('#bundle-drawer');

    await expect(drawer.locator('.bundle-row-intent .bundle-deriv-custom')).toBeVisible();
    await expect(drawer.locator('.bundle-row-perf .bundle-deriv-custom')).toBeHidden();
    await expect(drawer.locator('.bundle-row-intent .bundle-deriv-custom')).not.toHaveRole('button');
    await expect(drawer.locator('.bundle-row-perf .bundle-deriv-custom')).not.toHaveRole('button');

    await drawer.locator('.bundle-intent[data-intent="low_vram"]').click();
    await expect(drawer.locator('.bundle-row-intent .bundle-deriv-custom')).toBeHidden();
    await expect(drawer.locator('.bundle-row-perf .bundle-deriv-custom')).toBeVisible();

    const workload = drawer.locator('.bundle-workload');
    await expect(workload.locator('option')).toHaveCount(4);
    await workload.selectOption('roleplay_creative');
    await expect.poll(() => state.resolvePayloads.at(-1)?.workload_policy).toBe('roleplay_creative');
    expect(state.resolvePayloads.at(-1)?.selection?.workload_policy).toBeUndefined();
    await expect(drawer.locator('.bundle-diff-list')).toContainText('workload');
    expect(state.selectionPayloads).toHaveLength(0);
  });
});
