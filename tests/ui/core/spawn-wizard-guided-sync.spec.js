import { test, expect } from '@playwright/test';

// Keep the production wizard, its controls and event handlers real. Only the
// external model/memory services are replaced with deterministic responses.
test.beforeEach(async ({ page }) => {
  await page.route('**/api/**', route => route.abort());
  await page.route('**/api/model/introspect', route => route.fulfill({ json: {
    ok: true,
    file_size_bytes: 4 * 1024 ** 3,
    metadata: { n_layers: 32, n_ctx_train: 262144, n_kv_heads: 8, head_dim: 128, mtp_depth: 1 },
  } }));
  await page.route('**/api/vram-estimate', route => route.fulfill({ json: {
    ok: true, total_bytes: 5 * 1024 ** 3, model_bytes: 4 * 1024 ** 3,
    kv_cache_bytes: 512 * 1024 ** 2, headroom_bytes: 2 * 1024 ** 3,
    recommendation: 'fits',
  } }));
  await page.route('**/api/vram/auto-size', route => route.fulfill({ json: {
    ok: true,
    result: { context_size: 65536, kv_quant_k: 'q4_0', kv_quant_v: 'q4_0', ubatch_size: 512, warnings: [] },
  } }));
  await page.goto('/');
  await page.waitForLoadState('networkidle');
});

async function openGuided(page, templatePreset) {
  await page.evaluate(async preset => {
    const { openSpawnWizard, wizardState, showStep } = await import('/js/features/spawn-wizard.js');
    openSpawnWizard(preset ? { templatePreset: preset } : {});
    wizardState.model.source = 'local';
    wizardState.model.path = '/tmp/guided-sync-Q4_K_M.gguf';
    wizardState.model.paramB = 7;
    wizardState.model.modelBytes = 4 * 1024 ** 3;
    wizardState.vram.available = 64 * 1024 ** 3;
    wizardState.arch.metadataStatus = 'resolved';
    wizardState.arch.mtpDepth = 1;
    wizardState.viewMode = 'guided';
    document.getElementById('view-mode-select').value = 'guided';
    showStep(1);
  }, templatePreset);
  await expect(page.locator('#hw-decision-ctx')).toBeVisible();
}

async function launchChoices(page) {
  return page.evaluate(async () => {
    const { buildSpawnPayload } = await import('/js/features/spawn-wizard.js');
    const payload = buildSpawnPayload();
    return { context: payload.context_size, k: payload.ctk, v: payload.ctv };
  });
}

async function expectGuided(page, context, k, v = k) {
  await expect(page.locator('#spawn-context-size')).toHaveValue(String(context));
  await expect(page.locator('#hw-ctx-custom')).toHaveValue(String(context));
  await expect(page.locator('#spawn-cache-type-k')).toHaveValue(k);
  await expect(page.locator('#spawn-cache-type-v')).toHaveValue(v);
  await expect(page.locator('#hw-ctx-tiles .hw-decision-tile-active')).toHaveAttribute('data-ctx', String(context));
  if (k === v) {
    await expect(page.locator('#hw-kv-tiles .hw-decision-tile-active')).toHaveAttribute('data-kv', k);
    await expect(page.locator('#hw-sticky-kv')).toHaveText(`KV ${k}`);
  } else {
    await expect(page.locator('#hw-kv-tiles .hw-decision-tile-active')).toHaveCount(0);
    await expect(page.locator('#hw-sticky-kv')).toHaveText(`KV ${k}/${v}`);
  }
  expect(await launchChoices(page)).toEqual({ context, k, v });
}

test('@fake-data-bypass autosize refreshes Guided context, KV and sticky values without synthetic form events', async ({ page }) => {
  await openGuided(page);
  await page.locator('#vram-autosize-btn').dispatchEvent('click');
  await expect(page.locator('#vram-autosize-note')).toContainText('Set:');
  await expectGuided(page, 65536, 'q4_0');
  await expect(page.locator('#hw-sticky-ctx')).toHaveText('ctx 66k');
});

test('@fake-data-bypass a real scenario-card selection refreshes Guided KV at the retained context', async ({ page }) => {
  await openGuided(page, { context_size: 32768, ctk: 'q8_0', ctv: 'q8_0' });
  const fullPrecision = page.locator('#vram-scenarios .vram-scenario-card').filter({ has: page.locator('.vsc-mode-name', { hasText: 'Full precision' }) });
  await expect(fullPrecision).toHaveCount(1);
  await fullPrecision.dispatchEvent('click');
  await expectGuided(page, 32768, 'f16');
  await expect(page.locator('#hw-sticky-ctx')).toHaveText('ctx 33k');
});

test('@fake-data-bypass repeated Pro to Guided roundtrips preserve the same searchable canonical controls', async ({ page }) => {
  await openGuided(page, { context_size: 32768, ctk: 'q8_0', ctv: 'q8_0' });
  await page.evaluate(() => {
    window.guidedSyncControls = Object.fromEntries([
      'spawn-temperature', 'spawn-reasoning-mode', 'spawn-output-mode',
      'spawn-port', 'spawn-extra-args', 'spawn-context-size', 'spawn-cache-type-k',
    ].map(id => [id, document.getElementById(id)]));
  });
  for (let cycle = 0; cycle < 3; cycle += 1) {
    await page.locator('#view-mode-select').selectOption('pro');
    for (const id of ['spawn-temperature', 'spawn-reasoning-mode', 'spawn-output-mode', 'spawn-port', 'spawn-extra-args']) {
      await expect(page.locator(`#pro-controls-host #${id}`)).toHaveCount(1);
      await expect(page.locator(`[id="${id}"]`)).toHaveCount(1);
      const categories = {
        'spawn-temperature': 'Generation & reasoning',
        'spawn-reasoning-mode': 'Generation & reasoning',
        'spawn-output-mode': 'Tools & conversation formatting',
        'spawn-port': 'Network & observability',
        'spawn-extra-args': 'Advanced',
      };
      await expect(page.locator(`#${id}`).locator('xpath=ancestor::*[@data-pro-category][1]')).toHaveAttribute('data-pro-category', categories[id]);
    }
    await page.locator('#pro-filter-input').fill('temperature');
    await expect(page.locator('#spawn-temperature').locator('xpath=ancestor::*[@data-pro-category][1]')).not.toHaveClass(/pro-search-hidden/);
    await expect(page.locator('#spawn-extra-args').locator('xpath=ancestor::*[@data-pro-category][1]')).toHaveClass(/pro-search-hidden/);
    await page.locator('#pro-filter-input').fill('');
    await page.locator('#hw-ctx-custom').fill('65536');
    await page.locator('#hw-ctx-custom').dispatchEvent('change');
    await page.locator('#hw-kv-tiles [data-kv="q4_0"]').dispatchEvent('click');
    await page.locator('#view-mode-select').selectOption('guided');
    await expect(page.locator('#all-settings-group #spawn-sampling-block')).toHaveCount(1);
    await expect(page.locator('#all-settings-group #spawn-advanced-fields')).toHaveCount(1);
    await expectGuided(page, 65536, 'q4_0');
    expect(await page.evaluate(() => Object.entries(window.guidedSyncControls).every(([id, node]) => document.getElementById(id) === node))).toBe(true);
  }
});

test('@fake-data-bypass preset context and asymmetric KV restore on a later hardware visit', async ({ page }) => {
  await openGuided(page, { context_size: 32768, ctk: 'q4_0', ctv: 'q4_0' });
  await page.evaluate(async () => {
    const { closeSpawnWizard } = await import('/js/features/spawn-wizard.js');
    closeSpawnWizard();
  });
  await openGuided(page, { context_size: 131072, ctk: 'q8_0', ctv: 'f16' });
  await expectGuided(page, 131072, 'q8_0', 'f16');
});

test('@fake-data-bypass input-only and change-only canonical events both refresh Guided values', async ({ page }) => {
  await openGuided(page, { context_size: 32768, ctk: 'q8_0', ctv: 'q8_0' });
  for (const type of ['input', 'change']) {
    await page.evaluate(eventType => {
      document.getElementById('spawn-context-size').value = '65536';
      for (const id of ['spawn-context-size', 'spawn-cache-type-k', 'spawn-cache-type-v']) {
        const control = document.getElementById(id);
        if (id !== 'spawn-context-size') control.value = 'f16';
        control.dispatchEvent(new Event(eventType, { bubbles: true }));
      }
    }, type);
    await expectGuided(page, 65536, 'f16');
    await page.locator('#hw-ctx-tiles [data-ctx="32768"]').dispatchEvent('click');
    await page.locator('#hw-kv-tiles [data-kv="q8_0"]').dispatchEvent('click');
    await expectGuided(page, 32768, 'q8_0');
  }
});

test('@fake-data-bypass Guided vision and speed reflect production projector and MTP renders', async ({ page }) => {
  await openGuided(page);
  await page.evaluate(async () => {
    const { wizardState } = await import('/js/features/spawn-wizard.js');
    const { renderMmprojSection } = await import('/js/features/spawn-wizard-mmproj.js');
    wizardState.model.mmprojFiles = [
      { path: '/tmp/mmproj-first.gguf', size: 1024 },
      { path: '/tmp/mmproj-second.gguf', size: 2048 },
    ];
    wizardState.model.mmprojPath = '/tmp/mmproj-first.gguf';
    renderMmprojSection();
  });
  await expect(page.locator('#hw-vision-select')).toHaveValue('/tmp/mmproj-first.gguf');
  await page.evaluate(async () => {
    const { wizardState } = await import('/js/features/spawn-wizard.js');
    const { renderMmprojSection } = await import('/js/features/spawn-wizard-mmproj.js');
    const { renderMtpSection } = await import('/js/features/spawn-wizard-mtp-draft.js');
    wizardState.model.mmprojPath = '/tmp/mmproj-second.gguf';
    renderMmprojSection(); // same option list: only .value changes this time
    wizardState.hardware.mtpEnabled = false;
    renderMtpSection();
  });
  await expect(page.locator('#hw-vision-select')).toHaveValue('/tmp/mmproj-second.gguf');
  await expect(page.locator('#hw-use-mtp')).not.toBeChecked();
  await expect(page.locator('input[name="hw-speed"][value="off"]')).toBeChecked();
  await page.evaluate(async () => {
    const { wizardState } = await import('/js/features/spawn-wizard.js');
    const { renderMtpSection } = await import('/js/features/spawn-wizard-mtp-draft.js');
    wizardState.hardware.mtpEnabled = true;
    renderMtpSection();
  });
  await expect(page.locator('#hw-use-mtp')).toBeChecked();
  await expect(page.locator('input[name="hw-speed"][value="on"]')).toBeChecked();
});

test('@fake-data-bypass real Pro checkbox choices survive capability rediscovery and Guided roundtrips', async ({ page }) => {
  await openGuided(page);
  for (const checked of [false, true, false]) {
    await page.locator('#view-mode-select').selectOption('pro');
    await page.locator('#hw-use-mtp').evaluate((checkbox, value) => {
      checkbox.checked = value;
      checkbox.dispatchEvent(new Event('change', { bubbles: true }));
    }, checked);
    expect(await page.evaluate(async () => {
      const { _mtpUserConfigured } = await import('/js/features/spawn-wizard-mtp-draft.js');
      return _mtpUserConfigured;
    })).toBe(true);
    await page.evaluate(async () => {
      const { wizardState, doIntrospect } = await import('/js/features/spawn-wizard.js');
      const { refreshGuidedCapabilityCards } = await import('/js/features/spawn-wizard-guided.js');
      refreshGuidedCapabilityCards();
      await doIntrospect(wizardState.model.path);
    });
    await expect(page.locator('#hw-use-mtp')).toBeChecked({ checked });
    await page.locator('#view-mode-select').selectOption('guided');
    await expect(page.locator(`input[name="hw-speed"][value="${checked ? 'on' : 'off'}"]`)).toBeChecked();
    const launch = await page.evaluate(async () => {
      const { wizardState, buildSpawnPayload } = await import('/js/features/spawn-wizard.js');
      return { enabled: wizardState.hardware.mtpEnabled, spec: buildSpawnPayload().spec_type };
    });
    expect(launch.enabled).toBe(checked);
    expect(launch.spec).toBe(checked ? 'draft-mtp,ngram-mod' : '');
  }
});

test('@fake-data-bypass an explicit canonical speculative choice prevents a new built-in MTP default', async ({ page }) => {
  await openGuided(page);
  const result = await page.evaluate(async () => {
    const { wizardState, buildSpawnPayload } = await import('/js/features/spawn-wizard.js');
    const { refreshGuidedCapabilityCards } = await import('/js/features/spawn-wizard-guided.js');
    // A restored canonical selection is authoritative even before its user
    // listener has marked a checkbox or a Guided radio as configured.
    wizardState.hardware.mtpEnabled = false;
    document.getElementById('hw-use-mtp').checked = false;
    document.getElementById('spawn-spec-type').value = 'ngram-mod';
    document.getElementById('spawn-spec-type').dispatchEvent(new Event('change', { bubbles: true }));
    refreshGuidedCapabilityCards();
    return { enabled: wizardState.hardware.mtpEnabled, checked: document.getElementById('hw-use-mtp').checked, spec: buildSpawnPayload().spec_type };
  });
  expect(result).toEqual({ enabled: false, checked: false, spec: 'ngram-mod' });
});

test('@fake-data-bypass unrelated refreshes do not turn an explicit N-gram choice into MTP or Off', async ({ page }) => {
  await openGuided(page);
  await page.locator('input[name="hw-speed"][value="ngram"]').check();
  await page.locator('#hw-ctx-custom').fill('32768');
  await page.evaluate(async () => {
    const { refreshGuidedCapabilityCards } = await import('/js/features/spawn-wizard-guided.js');
    refreshGuidedCapabilityCards();
  });
  await expect(page.locator('input[name="hw-speed"][value="ngram"]')).toBeChecked();
  await expect(page.locator('#hw-use-mtp')).not.toBeChecked();
});

test('@fake-data-bypass backend roundtrips keep llama-only sampling out of Rapid Pro', async ({ page }) => {
  await openGuided(page);
  await page.locator('#view-mode-select').selectOption('pro');
  for (const backend of ['rapid_mlx', 'llama_cpp', 'rapid_mlx', 'llama_cpp']) {
    await page.evaluate(async loader => {
      const { selectWizardEngine, showStep } = await import('/js/features/spawn-wizard.js');
      selectWizardEngine(loader, true);
      showStep(1);
    }, backend);
    await expect(page.locator('#pro-controls-host #spawn-port')).toHaveCount(1);
    await expect(page.locator('[id="spawn-temperature"]')).toHaveCount(1);
    if (backend === 'rapid_mlx') {
      await expect(page.locator('#pro-controls-host #spawn-rapid-advanced-fields')).toHaveCount(1);
      await expect(page.locator('#pro-controls-host #spawn-temperature')).toHaveCount(0);
      await expect(page.locator('#spawn-sampling-block')).toBeHidden();
      await expect(page.locator('#pro-controls-host #hw-decision-vision')).toHaveCount(0);
      await page.locator('#view-mode-select').selectOption('guided');
      if (await page.locator('#all-settings-btn').getAttribute('aria-expanded') !== 'true') {
        await page.locator('#all-settings-btn').dispatchEvent('click');
      }
      for (const id of ['spawn-port', 'spawn-bind-host', 'spawn-api-key']) {
        await expect(page.locator(`#all-settings-group #${id}`)).toBeVisible();
      }
      expect(await page.locator('#spawn-sampling-block input, #spawn-sampling-block select')
        .evaluateAll(controls => controls.map(control => control.id).filter(id => ['spawn-port', 'spawn-bind-host', 'spawn-api-key'].includes(id))))
        .toEqual(['spawn-port', 'spawn-bind-host', 'spawn-api-key']);
      await page.locator('#view-mode-select').selectOption('pro');
    } else {
      await expect(page.locator('#pro-controls-host #spawn-temperature')).toHaveCount(1);
      await expect(page.locator('#pro-controls-host #hw-decision-vision')).toHaveCount(1);
    }
  }
  await page.locator('#view-mode-select').selectOption('guided');
  await expect(page.locator('#all-settings-group #spawn-sampling-block')).toHaveCount(1);
});
