import { test, expect } from '@playwright/test';

async function openVariant(page, backend, preset = null) {
  await page.evaluate(async ({ backend, preset }) => {
    const { openSpawnWizard, closeSpawnWizard, selectWizardEngine, wizardState, showStep } = await import('/js/features/spawn-wizard.js');
    closeSpawnWizard();
    openSpawnWizard({
      ...(preset ? { templatePreset: preset } : {}),
      ...(backend === 'llama_cpp' ? { localPath: '/tmp/test-fixtures/qwen.gguf' } : {}),
    });
    selectWizardEngine(backend, true);
    if (backend === 'rapid_mlx' && !preset) {
      wizardState.model.source = 'hf';
      wizardState.model.hfRepo = 'qwen3.8-27b-4bit';
      wizardState.model.rapidMlxSource = { kind: 'alias', value: 'qwen3.8-27b-4bit' };
    }
    showStep(2);
  }, { backend, preset });
  await expect(page.locator('#wizard-step-2.active .wizard-sidebar #spawn-alias')).toBeVisible();
}

for (const backend of ['rapid_mlx', 'llama_cpp']) {
  test(`${backend}: Guided served name follows shared state, preview and separate saved variants`, async ({ page }) => {
    const saved = [];
    const previews = [];
    await page.route('**/api/**', route => route.abort());
    await page.route('**/api/presets', async route => {
      if (route.request().method() === 'POST') {
        const preset = { ...route.request().postDataJSON(), id: `variant-${saved.length + 1}` };
        saved.push(preset);
        return route.fulfill({ json: { ok: true, preset } });
      }
      return route.fulfill({ json: { presets: saved } });
    });
    await page.route('**/api/rapid-mlx/command-preview', async route => {
      const config = route.request().postDataJSON();
      previews.push(config);
      return route.fulfill({ json: {
        argv: ['serve', config.model_source.value, '--served-model-name', config.served_model_name || config.model_source.value],
        redacted: true,
      } });
    });
    await page.goto('/');
    await page.waitForSelector('html.modules-ready');
    await openVariant(page, backend);
    const alias = page.locator('#spawn-alias');
    await expect(alias).toHaveCount(1);
    await alias.fill('qwen-205k');
    await expect(page.locator('#spawn-config-card')).toContainText('qwen-205k');
    if (backend === 'rapid_mlx') {
      await expect.poll(() => previews.at(-1)?.served_model_name).toBe('qwen-205k');
      expect(previews.at(-1).model_source).toEqual({ kind: 'alias', value: 'qwen3.8-27b-4bit' });
    }
    await page.locator('#spawn-preset-name-input').fill('205k variant');
    await page.locator('#spawn-save-preset-btn').click();
    await expect.poll(() => saved.length).toBe(1);
    expect(backend === 'rapid_mlx' ? saved[0].rapid_mlx.served_model_name : saved[0].alias).toBe('qwen-205k');

    // The same DOM control moves into Pro and returns to the Guided rail.
    await page.evaluate(async () => (await import('/js/features/spawn-wizard.js')).showStep(1));
    await page.locator('#view-mode-select').selectOption('pro');
    await expect(page.locator('#pro-controls-host #spawn-alias')).toHaveCount(1);
    await expect(alias).toHaveValue('qwen-205k');
    await alias.fill('qwen-32k');
    await page.locator('#view-mode-select').selectOption('guided');
    await page.evaluate(async () => (await import('/js/features/spawn-wizard.js')).showStep(2));
    await expect(alias).toHaveValue('qwen-32k');
    await expect(page.locator('#wizard-step-2 .wizard-sidebar #spawn-alias')).toHaveCount(1);
    await expect(page.locator('#spawn-config-card')).toContainText('qwen-32k');

    // Open a fresh wizard before saving a second independent preset.
    await openVariant(page, backend);
    await expect(alias).toHaveValue('');
    await alias.fill('qwen-32k');
    await page.locator('#spawn-preset-name-input').fill('32k variant');
    await page.locator('#spawn-save-preset-btn').click();
    await expect.poll(() => saved.length).toBe(2);
    expect(backend === 'rapid_mlx' ? saved[1].rapid_mlx.served_model_name : saved[1].alias).toBe('qwen-32k');
    for (let index = 0; index < saved.length; index++) {
      await openVariant(page, backend, saved[index]);
      await expect(alias).toHaveValue(index === 0 ? 'qwen-205k' : 'qwen-32k');
    }
    await alias.fill('');
    const payload = await page.evaluate(async () => (await import('/js/features/spawn-wizard.js')).buildSpawnPayload());
    expect(backend === 'rapid_mlx' ? payload.rapid_mlx.served_model_name : payload.alias).toBeNull();
    if (backend === 'rapid_mlx') expect(payload.rapid_mlx.model_source).toEqual({ kind: 'alias', value: 'qwen3.8-27b-4bit' });
  });
}
