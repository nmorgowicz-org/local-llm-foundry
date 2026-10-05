import { test, expect } from '@playwright/test';

// Page 2 of the Rapid-MLX wizard used to scroll into a large empty grey area when the
// All-settings drawer expanded (drawer layout overflow leaked into the step's scroll
// extent), and toggling Experimental MTP speculation chained the view to that grey
// region. The step now clips overflow, so only the two columns scroll.
test.describe('Rapid-MLX page-2 layout and protocol reference', () => {
  const GB = 1024 ** 3;
  const json = (route, body, status = 200) => route.fulfill({
    status, contentType: 'application/json', body: JSON.stringify(body),
  });

  // Two animation frames: layout and style recalculation triggered by the last change are done.
  const settle = (page) => page.evaluate(
    () => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))),
  );

  const sidecarHint = (page) => page.evaluate(
    () => document.getElementById('spawn-rapid-speculative-sidecars-list')?.textContent || '',
  );

  async function openHardwarePage(page) {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.route('**/api/**', route => {
      const url = new URL(route.request().url());
      if (url.pathname === '/api/memory-availability') {
        return json(route, { ok: true, snapshot: { current_safe_availability_bytes: 40 * GB, configured_ceiling_bytes: 48 * GB } });
      }
      if (url.pathname === '/api/metrics/gpu' || url.pathname === '/metrics/gpu') {
        return json(route, [{ name: 'Apple M-series', vendor: 'apple', backend: 'metal', vram_total_mb: 57344, metal_gpu_limit_mb: 57344 }]);
      }
      if (url.pathname === '/api/vram-estimate') {
        return json(route, { ok: true, weights_bytes: 15 * GB, kv_cache_bytes: 2 * GB, overhead_bytes: 1 * GB, total_bytes: 18 * GB, available_bytes: 40 * GB, headroom_bytes: 22 * GB, recommendation: 'fits', evidence: 'measured' });
      }
      if (url.pathname.startsWith('/api/rapid-mlx/models/')) {
        if (url.pathname.endsWith('/profile')) return json(route, { ok: true, profile: { tool_format: 'qwen3_coder_xml', reasoning_parser: 'qwen3', spec_decode: 'supported', extras: { mtp: true } } });
        return json(route, { ok: true, profile: { recommended: { tool_format: 'qwen3_coder_xml', reasoning_parser: 'qwen3', hybrid_mode: 'force' } } });
      }
      if (url.pathname === '/api/models/tags') return json(route, { ok: true, tags: {} });
      return route.abort();
    });
    await page.evaluate(async () => {
      const wiz = await import('/js/features/spawn-wizard.js');
      wiz.openSpawnWizard({ templatePreset: { backend: 'rapid_mlx', rapid_mlx: { model_source: { kind: 'mlx_directory', path: '/fake/hub/models--nightmedia--Qwen3.8-27B-MindMeld-AREX-mxfp4-mlx/snapshots/abc' } } } });
      wiz.showStep(1);
    });
    await expect(page.locator('#spawn-rapid-advanced-fields')).toBeAttached();
    await expect(page.locator('.wizard-body')).toBeVisible();
  }

  async function expandAllSettings(page) {
    // The drawer may still be collapsed when the wizard was entered
    // programmatically (the drawer toggle binds only after its renderer ran).
    // Force the drawer body and the advanced-fields chain open rather than
    // relying on the toggle button's handler being bound.
    await page.evaluate(() => {
      const drawer = document.getElementById('all-settings-drawer');
      if (drawer) drawer.style.display = '';
      const bodyEl = document.getElementById('all-settings-body');
      if (bodyEl) bodyEl.style.display = 'block';
      const rapidFields = document.getElementById('spawn-rapid-advanced-fields');
      if (rapidFields) {
        rapidFields.style.display = 'block';
        // The advanced group's inner grid carries an inline display:none from the
        // drawer relocation logic; clear it down the chain so the fields render.
        rapidFields.querySelectorAll('.hardware-grid').forEach(g => { g.style.display = 'grid'; });
      }
    });
    await expect(page.locator('#spawn-rapid-advanced-fields')).toBeVisible();
    await settle(page);
  }

  test('@in-memory-test expanded All settings does not create a grey scroll void', async ({ page }) => {
    await openHardwarePage(page);
    await expandAllSettings(page);
    const sh = await page.evaluate(() => {
      const body = document.querySelector('.wizard-body');
      return { scroll: body.scrollHeight, client: body.clientHeight, scrollTop: body.scrollTop };
    });
    expect(sh.scroll).toBeLessThanOrEqual(sh.client + 8);
    expect(sh.scrollTop).toBe(0);
  });

  test('@in-memory-test toggling MTP speculation does not scroll the page away', async ({ page }) => {
    await openHardwarePage(page);
    await expandAllSettings(page);
    await page.evaluate(() => {
      const toggle = document.getElementById('spawn-rapid-speculative-enabled');
      toggle.scrollIntoView({ block: 'center' });
      toggle.click();
      toggle.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await settle(page);
    // The old bug: the phantom scroll region below the step swallowed the view
    // (body.scrollTop jumped to ~2000 with the toggle far off screen). With the
    // step clipped, the outer scroller has nowhere to take the view.
    const state = await page.evaluate(() => {
      const body = document.querySelector('.wizard-body');
      return { scrollTop: body.scrollTop, scrollable: body.scrollHeight - body.clientHeight };
    });
    expect(state.scrollTop).toBe(0);
    expect(state.scrollable).toBeLessThanOrEqual(8);
  });

  test('@in-memory-test model profile auto-enables tool choice and reports MTP eligibility', async ({ page }) => {
    await openHardwarePage(page);
    await page.evaluate(async () => {
      const { scheduleRapidMlxProfileFetch } = await import('/js/features/spawn-wizard-rapid-mlx.js');
      scheduleRapidMlxProfileFetch('nightmedia/Qwen3.8-27B-MindMeld-AREX-mxfp4-mlx');
    });
    await expect(page.locator('#spawn-rapid-auto-tool-choice')).toBeChecked();
    await expect(page.locator('#spawn-rapid-mtp-eligibility')).toContainText('embedded MTP prediction heads');
  });

  test('@in-memory-test protocol modal resolves Gemma, not just Qwen', async ({ page }) => {
    await openHardwarePage(page);
    await expandAllSettings(page);
    await page.evaluate(() => {
      const path = document.getElementById('spawn-model-path');
      if (path) path.value = '/models/gemma-4-27b-it-qat-4bit';
    });
    await page.click('#spawn-rapid-advanced-fields [data-open-protocol-docs]');
    await expect(page.locator('#protocol-docs-overlay')).toBeVisible();
    await expect(page.locator('.protocol-docs-table')).toContainText('gemma4');
    await expect(page.locator('.protocol-docs-link')).toHaveAttribute('href', /families\/gemma/);
  });

  test('@in-memory-test official upstream MTP draft is suggested for a known tier', async ({ page }) => {
    await openHardwarePage(page);
    await page.route('**/api/hf/mtp-sidecars', route => json(route, { ok: true, sidecars: [] }));
    await page.evaluate(async () => {
      const wiz = await import('/js/features/spawn-wizard.js');
      const rapid = await import('/js/features/spawn-wizard-rapid-mlx.js');
      wiz.wizardState.model.path = '/fake/hub/models--nightmedia--Qwen3.8-27B-MindMeld-AREX-mxfp4-mlx/snapshots/abc';
      wiz.wizardState.hardware.speculativeEnabled = true;
      wiz.wizardState.hardware.speculativeSource = 'external';
      document.getElementById('spawn-rapid-speculative-enabled').checked = true;
      rapid.refreshRapidMlxSidecars();
    });
    await expect.poll(() => sidecarHint(page)).toContain('rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX');
    expect(await sidecarHint(page)).toContain('Preflight');
  });

  test('@in-memory-test fingerprint suggests a draft when the trunk name hides its family', async ({ page }) => {
    await openHardwarePage(page);
    // Scarlett-Opus-oQ4e-MLX: a Qwen3.8-27B finetune whose name carries no family hint.
    await page.route('**/api/hf/mtp-sidecars', route => json(route, { ok: true, sidecars: [] }));
    await page.route('**/api/rapid-mlx/mtp/draft-suggestion?*', route => json(route, {
      ok: true,
      suggestion: { repo: 'rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX', note: 'a bf16 variant (…-MTP-fp16-MLX) also exists', basis: 'architecture fingerprint' },
    }));
    await page.evaluate(async () => {
      const wiz = await import('/js/features/spawn-wizard.js');
      const rapid = await import('/js/features/spawn-wizard-rapid-mlx.js');
      wiz.wizardState.model.path = '/fake/models/mlx/native/Scarlett-Opus-oQ4e-MLX';
      wiz.wizardState.hardware.speculativeEnabled = true;
      wiz.wizardState.hardware.speculativeSource = 'external';
      document.getElementById('spawn-rapid-speculative-enabled').checked = true;
      rapid.refreshRapidMlxSidecars();
    });
    await expect.poll(() => sidecarHint(page)).toContain('rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX');
  });

  test('@in-memory-test Validate protocol opens the in-app reference modal', async ({ page }) => {
    await openHardwarePage(page);
    await expandAllSettings(page);
    await page.click('#spawn-rapid-advanced-fields [data-open-protocol-docs]');
    await expect(page.locator('#protocol-docs-overlay')).toBeVisible();
    await expect(page.locator('.protocol-docs-table')).toContainText('qwen3_coder_xml');
    await expect(page.locator('.protocol-docs-table')).toContainText('qwen3');
    await expect(page.locator('.protocol-docs-link')).toHaveAttribute('href', /rapidmlx\.com\/docs\/models\/families\/qwen/);
    await page.click('.protocol-docs-actions button');
    await expect(page.locator('#protocol-docs-overlay')).toHaveCount(0);
  });
});
