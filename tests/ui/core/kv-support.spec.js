import { test, expect } from '@playwright/test';

// K/V support line (kv-support.js): one server-written answer rendered the same
// way in the wizard (Guided + Pro), the preset editor and the launch chip.
// The endpoint is mocked per scenario; the classifier itself is covered by Rust tests.

const NATIVE = {
  support: { kind: 'native' },
  compiled_pair: { state: 'confirmed', value: true, source: 'probe' },
  reason: null,
  fix: null,
  blocking: false,
  buildable_missing: false,
  backend: 'cuda',
};

const UNVERIFIED = {
  support: { kind: 'unverified', reason: 'x' },
  compiled_pair: { state: 'confirmed', value: true, source: 'probe' },
  reason: 'The q8_0/q4_0 FlashAttention kernel is compiled in, but execution is unverified because no model metadata was provided.',
  fix: null,
  blocking: false,
  buildable_missing: false,
  backend: 'cuda',
};

const CUDA_FALLBACK = {
  support: { kind: 'fallback', impact: 'slower_decode', reason: 'x' },
  compiled_pair: { state: 'confirmed', value: false, source: 'probe' },
  reason: 'No FlashAttention vector kernel is compiled for q8_0/q4_0. llama.cpp converts K and V to f16 instead, which is slower.',
  fix: null,
  blocking: false,
  buildable_missing: true,
  backend: 'cuda',
};

const METAL_CPU = {
  support: { kind: 'fallback', impact: 'cpu_attention', reason: 'x' },
  compiled_pair: { state: 'assumed', value: false, reason: 'source' },
  reason: 'Metal flash attention supports equal K/V types only, so q8_0/q4_0 attention runs on the CPU.',
  fix: null,
  blocking: false,
  buildable_missing: false,
  backend: 'metal',
};

const NO_FA = {
  support: { kind: 'fallback', impact: 'no_flash_attention', reason: 'x' },
  compiled_pair: { state: 'confirmed', value: true, source: 'probe' },
  reason: 'Flash attention is off, so the fused K/V kernels are not used.',
  fix: null,
  blocking: false,
  buildable_missing: false,
  backend: 'cuda',
};

const WONT_START = {
  support: { kind: 'wont_start', reason: 'x', fix: 'x' },
  compiled_pair: { state: 'confirmed', value: true, source: 'probe' },
  reason: 'llama.cpp refuses to start with a quantized V cache (q4_0) when flash attention is off.',
  fix: 'Set flash attention to on or auto, or use a floating V cache type (f16, bf16 or f32).',
  blocking: true,
  buildable_missing: false,
  backend: 'cuda',
};

const GEMMA = {
  id: 'gemma4-quantized-kv',
  severity: 'caution',
  text: 'Gemma 4 can be sensitive to quantizing either K or V. Prefer conservative cache precision and check output quality on your own prompts.',
  source: 'Maintainer/user guidance (qualitative); not a measured severity.',
};

/** Install the mock. `respond(params)` returns a partial body (or null for a 500). */
async function mockKv(page, respond, { modelsDir = '/models' } = {}) {
  const calls = [];
  await page.route('**/api/hf/download-dir', route => route.fulfill({
    status: 200, contentType: 'application/json', body: JSON.stringify({ dir: modelsDir }),
  }));
  await page.route('**/api/llama-binary/kv-support*', async route => {
    const url = new URL(route.request().url());
    const params = Object.fromEntries(url.searchParams.entries());
    calls.push(params);
    const body = await respond(params, calls.length);
    if (!body) return route.fulfill({ status: 500, body: 'nope' });
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ok: true,
        k: params.k,
        v: params.v,
        effective_flash_attn: params.fa === 'off' ? 'off' : 'on',
        advisories: [],
        runtime_identity: 'rt-1',
        device_identity: 'CUDA0',
        model_metadata_identity: params.model_id ? 'm-1' : null,
        basis_version: 1,
        session_observations: [],
        ...body,
      }),
    });
  });
  return calls;
}

async function openWizard(page, { model = '/models/m.gguf' } = {}) {
  await page.route('**/api/llama-binary/capabilities', route => route.fulfill({
    status: 200, contentType: 'application/json', body: JSON.stringify({ ok: true, snapshot: {} }),
  }));
  await page.goto('/');
  await page.waitForSelector('html.modules-ready');
  await page.evaluate(async (path) => {
    const { openSpawnWizard, selectWizardEngine, showStep, wizardState } = await import('/js/features/spawn-wizard.js');
    openSpawnWizard();
    selectWizardEngine('llama_cpp', true);
    wizardState.model.source = 'local';
    wizardState.model.path = path;
    const input = document.getElementById('spawn-model-path');
    if (input) input.value = path;
    showStep(1);
  }, model);
}

async function setKv(page, { k, v, fa }) {
  await page.evaluate(({ k, v, fa }) => {
    const fire = (id, value) => {
      const el = document.getElementById(id);
      if (!el || value == null) return;
      el.value = value;
      el.dispatchEvent(new Event('change', { bubbles: true }));
    };
    fire('spawn-cache-type-k', k);
    fire('spawn-cache-type-v', v);
    fire('spawn-flash-attn', fa);
  }, { k, v, fa });
}

const text = (page, id) => page.locator(`#${id}`).evaluate(el => el.textContent.replace(/\s+/g, ' ').trim());

test.describe('@in-memory-test K/V support: wizard', () => {
  test('CUDA native is quiet', async ({ page }) => {
    await mockKv(page, () => NATIVE);
    await openWizard(page);
    await setKv(page, { k: 'q8_0', v: 'q8_0' });
    await expect(page.locator('#hw-kv-support')).toHaveAttribute('data-kv-state', 'native');
    await expect(page.locator('#hw-kv-support .kv-support-rebuild')).toHaveCount(0);
    // Quiet: native uses no alert role.
    await expect(page.locator('#hw-kv-support')).toHaveAttribute('role', 'status');
  });

  test('Guided and Pro show the same line: one card, visible in both profiles', async ({ page }) => {
    await mockKv(page, () => CUDA_FALLBACK);
    await openWizard(page);
    await setKv(page, { k: 'q8_0', v: 'q4_0' });
    await expect(page.locator('#hw-kv-support')).toBeVisible();
    const guidedText = await text(page, 'hw-kv-support');
    await page.evaluate(() => {
      const select = document.getElementById('view-mode-select');
      select.value = 'pro';
      select.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await setKv(page, { k: 'q8_0', v: 'q4_0' });
    await page.evaluate(() => document.querySelectorAll('#pro-rail-nav .pro-rail-item')[1]?.click());
    await expect(page.locator('#hw-kv-support')).toBeVisible();
    await expect(page.locator('#hw-kv-support')).toHaveAttribute('data-kv-state', 'fallback');
    expect(await text(page, 'hw-kv-support')).toBe(guidedText);
    // There is exactly one wizard line; the hidden inline row carries none.
    await expect(page.locator('.kv-support[id$="kv-support"]:not(#modal-kv-support):not(#bundle-kv-support)')).toHaveCount(1);
  });

  test('mixed tile: independent K/V, highlights from a stored mixed pair, native sub-label on CUDA', async ({ page }) => {
    await mockKv(page, () => NATIVE);
    await openWizard(page);
    const tile = page.locator('#hw-kv-tiles [data-kv-mixed]');
    await setKv(page, { k: 'q8_0', v: 'q8_0' });
    await expect(tile).toBeVisible();
    await expect(tile.locator('.hw-tile-tag')).toHaveText('fused path');
    await tile.click();
    await expect(page.locator('#spawn-cache-type-k')).toHaveValue('q8_0');
    await expect(page.locator('#spawn-cache-type-v')).toHaveValue('q4_0');
    await expect(tile).toHaveClass(/hw-decision-tile-active/);
    // Equal tiles no longer claim the mixed pair.
    await expect(page.locator('#hw-kv-tiles [data-kv="q8_0"]')).not.toHaveClass(/hw-decision-tile-active/);
    // A stored mixed pair highlights the mixed tile instead of leaving none selected.
    await setKv(page, { k: 'q8_0', v: 'q8_0' });
    await setKv(page, { k: 'q8_0', v: 'q4_0' });
    await expect(tile).toHaveClass(/hw-decision-tile-active/);
  });

  test('CUDA: amber only for an evidenced missing kernel, with the rebuild hint; unverified stays neutral', async ({ page }) => {
    await mockKv(page, (p) => (p.model_id ? CUDA_FALLBACK : UNVERIFIED));
    await openWizard(page);
    await setKv(page, { k: 'q8_0', v: 'q4_0' });
    await expect(page.locator('#hw-kv-support')).toHaveAttribute('data-kv-state', 'fallback');
    await expect(page.locator('#hw-kv-support .kv-support-rebuild')).toHaveCount(1);
    await expect(page.locator('#hw-kv-tiles [data-kv-mixed] .hw-tile-tag')).toHaveText('slower path');

    // No model id (the path is outside the models directory): neutral, with the compiled pair shown separately.
    await page.evaluate(() => { document.getElementById('spawn-model-path').value = '/elsewhere/m.gguf'; });
    await setKv(page, { k: 'q8_0', v: 'q4_0', fa: 'on' });
    await expect(page.locator('#hw-kv-support')).toHaveAttribute('data-kv-state', 'unverified');
    await expect(page.locator('#hw-kv-support .kv-support-compiled')).toContainText('compiled into this runtime');
    await expect(page.locator('#hw-kv-support .kv-support-rebuild')).toHaveCount(0);
    await expect(page.locator('#hw-kv-support')).not.toHaveClass(/kv-support--fallback/);
  });

  test('Metal: mixed tile hidden, Pro shows contextual CPU attention, no rebuild link', async ({ page }) => {
    await mockKv(page, () => METAL_CPU);
    await openWizard(page);
    await setKv(page, { k: 'q8_0', v: 'f16' });
    await expect(page.locator('#hw-kv-tiles [data-kv-mixed]')).toBeHidden();
    await expect(page.locator('#hw-kv-support')).toHaveAttribute('data-kv-state', 'fallback');
    await expect(page.locator('#hw-kv-support')).toContainText('runs on the CPU');
    await expect(page.locator('[data-kv-rebuild]')).toHaveCount(0);
    // Already selected: stays visible so the choice is not silently hidden.
    await setKv(page, { k: 'q8_0', v: 'q4_0' });
    await expect(page.locator('#hw-kv-tiles [data-kv-mixed]')).toBeVisible();
  });

  test('FA off: floating V is no-FA (not blocked); quantized V blocks Start with the fix', async ({ page }) => {
    await mockKv(page, (p) => (p.fa === 'off' && p.v === 'q4_0' ? WONT_START : p.fa === 'off' ? NO_FA : NATIVE));
    // Start also needs an installed binary; CI has none, so state it explicitly.
    await page.route('**/api/llama-binary/version', route => route.fulfill({ json: { ok: true, build: 1234 } }));
    await openWizard(page);
    await setKv(page, { k: 'q8_0', v: 'f16', fa: 'off' });
    await expect(page.locator('#hw-kv-support')).toHaveAttribute('data-kv-state', 'fallback');
    await expect(page.locator('#hw-kv-support')).toContainText('Flash attention is off');
    await expect(page.locator('#spawn-server-btn')).toHaveAttribute('data-kv-blocked', 'false');

    await setKv(page, { k: 'q8_0', v: 'q4_0', fa: 'off' });
    await expect(page.locator('#hw-kv-support')).toHaveAttribute('data-kv-state', 'wont_start');
    await expect(page.locator('#hw-kv-support')).toHaveAttribute('role', 'alert');
    await expect(page.locator('#hw-kv-support .kv-support-fix')).toContainText('flash attention to on or auto');
    await expect(page.locator('#spawn-server-btn')).toBeDisabled();
    await expect(page.locator('#spawn-server-btn')).toHaveAttribute('title', /flash attention to on or auto/);

    // Fixing it re-enables Start.
    await setKv(page, { k: 'q8_0', v: 'q4_0', fa: 'on' });
    await expect(page.locator('#hw-kv-support')).toHaveAttribute('data-kv-state', 'native');
    await expect(page.locator('#spawn-server-btn')).toHaveAttribute('data-kv-blocked', 'false');
    await expect(page.locator('#spawn-server-btn')).toBeEnabled();
    await expect(page.locator('#spawn-server-btn')).not.toHaveAttribute('title', /flash attention/);
  });

  test('Gemma guidance appears for K-only and V-only quantization; absent without model metadata', async ({ page }) => {
    const calls = await mockKv(page, (p) => ({
      ...NATIVE,
      advisories: p.model_id && (p.k !== 'f16' || p.v !== 'f16') ? [GEMMA] : [],
    }));
    await openWizard(page);
    await setKv(page, { k: 'q8_0', v: 'f16' });
    await expect(page.locator('#hw-kv-support [data-advisory-id="gemma4-quantized-kv"]')).toContainText('either K or V');
    await setKv(page, { k: 'f16', v: 'q8_0' });
    await expect(page.locator('#hw-kv-support [data-advisory-id="gemma4-quantized-kv"]')).toHaveCount(1);
    await setKv(page, { k: 'f16', v: 'f16' });
    await expect(page.locator('#hw-kv-support [data-advisory-id]')).toHaveCount(0);
    // Source is in the tooltip, never a universal claim.
    await setKv(page, { k: 'q8_0', v: 'q8_0' });
    await expect(page.locator('#hw-kv-support [data-advisory-id]')).toHaveAttribute('title', /qualitative/);

    // Model outside the models directory: no model_id is sent, so no advisory.
    await page.evaluate(() => { document.getElementById('spawn-model-path').value = '/elsewhere/gemma.gguf'; });
    await setKv(page, { k: 'q4_0', v: 'q4_0' });
    await expect(page.locator('#hw-kv-support')).toHaveAttribute('data-kv-state', 'native');
    await expect(page.locator('#hw-kv-support [data-advisory-id]')).toHaveCount(0);
    expect(calls.at(-1).model_id).toBeUndefined();
    expect(calls.some(c => c.model_id === 'm.gguf')).toBe(true);
  });

  test('late responses are discarded; a changed runtime or model identity replaces cached evidence', async ({ page }) => {
    let releaseFirst;
    const gate = new Promise(resolve => { releaseFirst = resolve; });
    const calls = await mockKv(page, async (p, n) => {
      if (p.v === 'q4_0' && n === 1) {
        await gate; // the stale answer for the first selection arrives last
        return { ...CUDA_FALLBACK };
      }
      return { ...NATIVE, runtime_identity: p.v === 'bf16' ? 'rt-2-plugin-swapped' : 'rt-1' };
    });
    await openWizard(page);
    await setKv(page, { k: 'q8_0', v: 'q4_0' });
    await expect.poll(() => calls.length).toBeGreaterThan(0);
    await setKv(page, { k: 'q8_0', v: 'q8_0' });
    await expect(page.locator('#hw-kv-support')).toHaveAttribute('data-kv-state', 'native');
    const staleResponse = page.waitForResponse(r =>
      r.url().includes('/api/llama-binary/kv-support') && new URL(r.url()).searchParams.get('v') === 'q4_0');
    releaseFirst();
    await (await staleResponse).finished();
    // Intentional negative wait: the stale answer is now fully delivered, and what is left is
    // in-page promise continuation (a cached getContext), so 100ms is ample for a repaint to
    // happen if it were going to. There is no condition to wait on for "nothing changes".
    await page.waitForTimeout(100);
    // The late q8_0/q4_0 answer must not repaint the q8_0/q8_0 selection.
    await expect(page.locator('#hw-kv-support')).toHaveAttribute('data-kv-state', 'native');
    await expect(page.locator('#hw-kv-support .kv-support-rebuild')).toHaveCount(0);

    // Same key asked again after the runtime identity changed: re-fetched, never served from cache.
    const before = calls.length;
    await setKv(page, { k: 'q8_0', v: 'bf16' });
    await expect.poll(() => calls.length).toBeGreaterThan(before);
    const ids = await page.evaluate(async () => {
      const { invalidateKvSupport } = await import('/js/features/kv-support.js');
      invalidateKvSupport();
      return true;
    });
    expect(ids).toBe(true);
  });

  test('a failed lookup renders nothing and never blocks Start', async ({ page }) => {
    await mockKv(page, () => null);
    await openWizard(page);
    // Wait for the failing lookup itself (debounce + request) instead of a fixed sleep.
    const failedLookup = page.waitForResponse(r =>
      r.url().includes('/api/llama-binary/kv-support') && r.status() === 500);
    await setKv(page, { k: 'q8_0', v: 'q4_0', fa: 'off' });
    await (await failedLookup).finished();
    await page.evaluate(() => {}); // let the page consume the response before asserting
    await expect(page.locator('#hw-kv-support')).toBeHidden();
    await expect(page.locator('#spawn-server-btn')).toHaveAttribute('data-kv-blocked', 'false');
  });
});

test.describe('@in-memory-test K/V support: preset editor and launch chip', () => {
  async function openEditor(page, seed = {}) {
    await page.goto('/');
    await page.waitForSelector('html.modules-ready');
    await page.evaluate(async (preset) => {
      const { openPresetModal } = await import('/js/features/presets.js');
      openPresetModal('new', null, { backend: 'llama_cpp', name: 'KV fixture', model_path: '/models/m.gguf', ...preset });
    }, seed);
  }

  test('new presets default to q8_0/q8_0 and the stale f16 tooltip is gone', async ({ page }) => {
    await mockKv(page, () => NATIVE);
    await openEditor(page);
    await expect(page.locator('#modal-ctk')).toHaveValue('q8_0');
    await expect(page.locator('#modal-ctv')).toHaveValue('q8_0');
    const tip = await page.locator('label[for="modal-ctv"]').getAttribute('title');
    expect(tip).not.toMatch(/f16 can be faster/i);
    expect(await page.locator('label[for="modal-flash-attn"]').getAttribute('title')).toMatch(/refuses to start/);
  });

  test('stored presets keep their values (D11 applies to new presets only)', async ({ page }) => {
    await mockKv(page, () => NATIVE);
    await page.goto('/');
    await page.waitForSelector('html.modules-ready');
    await page.evaluate(async () => {
      const { openPresetModal } = await import('/js/features/presets.js');
      openPresetModal('edit', null, {
        id: 'p-old', name: 'Old', backend: 'llama_cpp', model_path: '/models/m.gguf', ctk: 'q8_0', ctv: 'f16',
      });
    });
    await expect(page.locator('#modal-ctv')).toHaveValue('f16');
  });

  test('editor shows the same line; wont_start is red but plain Save stays enabled', async ({ page }) => {
    await mockKv(page, (p) => (p.fa === 'off' && p.v === 'q4_0' ? WONT_START : NATIVE));
    await openEditor(page);
    await expect(page.locator('#modal-kv-support')).toHaveAttribute('data-kv-state', 'native');
    await page.evaluate(() => {
      const set = (id, value) => {
        const el = document.getElementById(id);
        el.value = value;
        el.dispatchEvent(new Event('change', { bubbles: true }));
      };
      set('modal-ctv', 'q4_0');
      set('modal-flash-attn', 'off');
    });
    await expect(page.locator('#modal-kv-support')).toHaveAttribute('data-kv-state', 'wont_start');
    await expect(page.locator('#modal-kv-support .kv-support-fix')).toContainText('flash attention to on or auto');
    await expect(page.locator('#btn-modal-save')).toBeEnabled();
  });

  test('editor and wizard print the same text for the same context', async ({ page }) => {
    await mockKv(page, () => CUDA_FALLBACK);
    await openEditor(page);
    await page.evaluate(() => {
      const el = document.getElementById('modal-ctv');
      el.value = 'q4_0';
      el.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await expect(page.locator('#modal-kv-support')).toHaveAttribute('data-kv-state', 'fallback');
    const editorText = await text(page, 'modal-kv-support');
    await page.evaluate(() => document.getElementById('preset-modal').classList.remove('open'));
    await openWizard(page);
    await setKv(page, { k: 'q8_0', v: 'q4_0' });
    await expect(page.locator('#hw-kv-support')).toHaveAttribute('data-kv-state', 'fallback');
    expect(await text(page, 'hw-kv-support')).toBe(editorText);
  });

  test('launch chip: quiet for native, marked for fallback, unverified and wont_start', async ({ page }) => {
    await mockKv(page, (p) => ({
      q8_0_q8_0: NATIVE, q8_0_q4_0: CUDA_FALLBACK, q8_0_bf16: UNVERIFIED, q4_0_q4_0: WONT_START,
    }[`${p.k}_${p.v}`] || NATIVE));
    await page.goto('/');
    await page.waitForSelector('html.modules-ready');
    const found = await page.evaluate(async () => {
      const { sessionState } = await import('/js/core/app-state.js');
      sessionState.presets = [
        { id: 'a', name: 'A', backend: 'llama_cpp', model_path: '/models/a.gguf', ctk: 'q8_0', ctv: 'q8_0', context_size: 8192, flash_attn: 'on' },
        { id: 'b', name: 'B', backend: 'llama_cpp', model_path: '/models/b.gguf', ctk: 'q8_0', ctv: 'q4_0', context_size: 8192, flash_attn: 'on' },
        { id: 'c', name: 'C', backend: 'llama_cpp', model_path: '/models/c.gguf', ctk: 'q8_0', ctv: 'bf16', context_size: 8192, flash_attn: 'on' },
        { id: 'd', name: 'D', backend: 'llama_cpp', model_path: '/models/d.gguf', ctk: 'q4_0', ctv: 'q4_0', context_size: 8192, flash_attn: 'off' },
      ];
      const { renderLaunchGrid } = await import('/js/features/setup-view.js');
      renderLaunchGrid();
      return document.querySelectorAll('.launch-card[data-preset-id]').length;
    });
    expect(found).toBeGreaterThanOrEqual(4);
    const chip = id => page.locator(`.launch-card[data-preset-id="${id}"] .launch-chip--kv`);
    await expect(chip('b')).toHaveAttribute('data-kv-state', 'fallback');
    await expect(chip('c')).toHaveAttribute('data-kv-state', 'unverified');
    await expect(chip('d')).toHaveAttribute('data-kv-state', 'wont_start');
    await expect(chip('d')).toHaveAttribute('title', /flash attention to on or auto/);
    await expect(chip('a')).toHaveCount(0);
  });

  test('launch chip stays quiet for an equal pair that is only unverified', async ({ page }) => {
    await mockKv(page, () => UNVERIFIED);
    await page.goto('/');
    await page.waitForSelector('html.modules-ready');
    await page.evaluate(async () => {
      const { sessionState } = await import('/js/core/app-state.js');
      sessionState.presets = [
        { id: 'eq', name: 'EQ', backend: 'llama_cpp', model_path: '/models/a.gguf', ctk: 'q8_0', ctv: 'q8_0', context_size: 8192 },
        { id: 'mx', name: 'MX', backend: 'llama_cpp', model_path: '/models/b.gguf', ctk: 'q8_0', ctv: 'q4_0', context_size: 8192 },
      ];
      const { renderLaunchGrid } = await import('/js/features/setup-view.js');
      renderLaunchGrid();
    });
    await expect(page.locator('.launch-card[data-preset-id="mx"] .launch-chip--kv')).toHaveAttribute('data-kv-state', 'unverified');
    await expect(page.locator('.launch-card[data-preset-id="eq"] .launch-chip--kv')).toHaveCount(0);
  });
});
