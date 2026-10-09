import { test, expect } from '@playwright/test';

// The spawn wizard gained five Rapid-MLX throughput controls; the preset editor did not.
// A preset that cannot express a field does not merely omit it -- the editor rebuilds the
// rapid_mlx object on every save, so it writes its own defaults over whatever the wizard
// set. These tests cover both directions: existing values must load into the controls, and
// edited values must reach the request body.
test.describe('Rapid-MLX preset editor throughput fields', () => {
  const SEED = {
    name: 'throughput probe',
    backend: 'rapid_mlx',
    rapid_mlx: {
      port: 8080,
      model_source: '/tmp/Qwen3-8B-4bit',
      gpu_memory_utilization: 0.85,
      max_num_seqs: 8,
      max_concurrent_requests: 32,
      pflash_policy: 'auto',
    },
  };

  // Edit mode, not seeded-new: only the edit path loads stored values into the controls,
  // and editing an existing preset is the path where a dropped field overwrites real data.
  async function openSeeded(page, seed = SEED) {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.evaluate(async (seed) => {
      const { sessionState } = await import('/js/core/app-state.js');
      const mod = await import('/js/features/presets.js');
      sessionState.presets = [{ id: 'probe', ...seed }];
      const sel = document.getElementById('preset-select');
      sel.innerHTML = '<option value="probe">throughput probe</option>';
      sel.value = 'probe';
      mod.openPresetModal('edit');
    }, seed);
    await page.locator('#preset-modal .preset-nav-item[data-section="context"]').click();
  }

  test('@in-memory-test stored values load into the controls', async ({ page }) => {
    await openSeeded(page);
    await expect(page.locator('#modal-rapid-gpu-memory-utilization')).toHaveValue('0.85');
    await expect(page.locator('#modal-rapid-max-num-seqs')).toHaveValue('8');
    await expect(page.locator('#modal-rapid-max-concurrent-requests')).toHaveValue('32');
    await expect(page.locator('#modal-rapid-pflash-policy')).toHaveValue('auto');
  });

  for (const [label, concurrency] of [
    ['missing', {}],
    ['null', { max_num_seqs: null, max_concurrent_requests: null }],
  ]) {
    test(`@in-memory-test ${label} stored concurrency loads as Auto rather than new-preset defaults`, async ({ page }) => {
      await openSeeded(page, {
        ...SEED,
        rapid_mlx: { port: 8080, model_source: '/tmp/Qwen3-8B-4bit', ...concurrency },
      });
      await expect(page.locator('#modal-rapid-max-num-seqs')).toHaveValue('');
      await expect(page.locator('#modal-rapid-max-concurrent-requests')).toHaveValue('');
    });
  }

  test('@in-memory-test edited values reach the save request', async ({ page }) => {
    await openSeeded(page);

    let body = null;
    await page.route('**/api/presets/**', async (route, request) => {
      if (request.method() === 'PUT') body = request.postDataJSON();
      await route.fulfill({ status: 200, contentType: 'application/json', body: '{"id":"probe"}' });
    });

    await page.selectOption('#modal-rapid-gpu-memory-utilization', '0.95');
    await page.selectOption('#modal-rapid-max-num-seqs', '4');
    await page.selectOption('#modal-rapid-pflash-policy', 'on');
    // Editing an existing preset is a two-step save: the first call renders a change
    // summary and flips the button to "Confirm Save", the second issues the PUT.
    await page.evaluate(async () => {
      const mod = await import('/js/features/presets.js');
      await mod.savePreset(new Event('submit'));
    });
    await expect(page.locator('#preset-change-summary')).toBeVisible();
    await expect(page.locator('#preset-change-summary-list')).toContainText('0.95');
    await page.evaluate(async () => {
      const mod = await import('/js/features/presets.js');
      await mod.savePreset(new Event('submit'));
    });

    expect(body, 'save request was never issued').not.toBeNull();
    expect(body.rapid_mlx.gpu_memory_utilization).toBe(0.95);
    expect(body.rapid_mlx.max_num_seqs).toBe(4);
    expect(body.rapid_mlx.pflash_policy).toBe('on');
    // Untouched controls keep the seeded values rather than reverting to a default.
    expect(body.rapid_mlx.max_concurrent_requests).toBe(32);
  });

  // The reason Auto writes null instead of omitting the key: `out` is spread over the stored
  // rapid_mlx object, so an omitted key leaves the previous value untouched and selecting Auto
  // on a preset that already had a value would silently do nothing.
  test('@in-memory-test selecting Auto clears stored values and reloads as Auto', async ({ page }) => {
    await openSeeded(page);
    await expect(page.locator('#modal-rapid-gpu-memory-utilization')).toHaveValue('0.85');
    await expect(page.locator('#modal-rapid-max-num-seqs')).toHaveValue('8');
    await expect(page.locator('#modal-rapid-max-concurrent-requests')).toHaveValue('32');
    await page.selectOption('#modal-rapid-gpu-memory-utilization', '');
    await page.selectOption('#modal-rapid-max-num-seqs', '');
    await page.selectOption('#modal-rapid-max-concurrent-requests', '');

    let body = null;
    await page.route('**/api/presets/**', async (route, request) => {
      if (request.method() === 'PUT') body = request.postDataJSON();
      await route.fulfill({ status: 200, contentType: 'application/json', body: '{"id":"probe"}' });
    });
    for (const _ of [0, 1]) {
      await page.evaluate(async () => {
        const mod = await import('/js/features/presets.js');
        await mod.savePreset(new Event('submit'));
      });
    }

    expect(body, 'save request was never issued').not.toBeNull();
    expect(body.rapid_mlx.gpu_memory_utilization).toBeNull();
    expect(body.rapid_mlx.max_num_seqs).toBeNull();
    expect(body.rapid_mlx.max_concurrent_requests).toBeNull();

    // A saved Auto must remain Auto when reopened, not turn back into a safe default.
    await openSeeded(page, body);
    await expect(page.locator('#modal-rapid-gpu-memory-utilization')).toHaveValue('');
    await expect(page.locator('#modal-rapid-max-num-seqs')).toHaveValue('');
    await expect(page.locator('#modal-rapid-max-concurrent-requests')).toHaveValue('');
  });

  async function captureConfirmedSave(page) {
    let body = null;
    await page.route('**/api/presets/**', async (route, request) => {
      if (request.method() === 'PUT') body = request.postDataJSON();
      await route.fulfill({ status: 200, contentType: 'application/json', body: '{"id":"probe"}' });
    });
    await page.evaluate(async () => {
      const mod = await import('/js/features/presets.js');
      await mod.savePreset(new Event('submit'));
      if (document.getElementById('btn-modal-save').dataset.confirmed === 'yes') {
        await mod.savePreset(new Event('submit'));
      }
    });
    expect(body, 'save request was never issued').not.toBeNull();
    return body;
  }

  // The backend normalizes the nullable legacy mirror to zero and may omit the
  // unset Rapid option. Reopen that representation, not just the outgoing nulls.
  function persistedAutoContext(body) {
    const persisted = { ...body, context_size: 0, rapid_mlx: { ...body.rapid_mlx } };
    delete persisted.rapid_mlx.context_length;
    return persisted;
  }

  for (const [label, context] of [
    ['missing', {}],
    ['null', { context_length: null }],
  ]) {
    for (const [mirrorLabel, mirror] of [
      ['no legacy mirror', {}],
      ['stale legacy mirror', { context_size: 65536 }],
    ]) {
      test(`@in-memory-test name-only edit preserves ${label} Auto context with ${mirrorLabel}`, async ({ page }) => {
        await openSeeded(page, {
          ...SEED,
          ...mirror,
          rapid_mlx: { ...SEED.rapid_mlx, ...context },
        });
        await expect(page.locator('#modal-context-size')).toHaveValue('');
        await page.locator('#preset-modal .preset-nav-item[data-section="model"]').click();
        await page.locator('#modal-name').fill('renamed Auto context');

        const body = await captureConfirmedSave(page);
        expect(body.name).toBe('renamed Auto context');
        expect(body.rapid_mlx.context_length).toBeNull();
        expect(body.context_size).toBeNull();

        await openSeeded(page, persistedAutoContext(body));
        await expect(page.locator('#modal-context-size')).toHaveValue('');
      });
    }
  }

  test('@in-memory-test explicit Rapid context can be cleared to Auto without reviving its mirror', async ({ page }) => {
    await openSeeded(page, {
      ...SEED,
      context_size: 65536,
      rapid_mlx: { ...SEED.rapid_mlx, context_length: 32768 },
    });
    await expect(page.locator('#modal-context-size')).toHaveValue('32768');
    await page.locator('#modal-context-size').fill('');

    const body = await captureConfirmedSave(page);
    expect(body.rapid_mlx.context_length).toBeNull();
    expect(body.context_size).toBeNull();
    await openSeeded(page, body);
    await expect(page.locator('#modal-context-size')).toHaveValue('');
    await openSeeded(page, persistedAutoContext(body));
    await expect(page.locator('#modal-context-size')).toHaveValue('');
  });

  for (const [label, editedContext, expectedContext] of [
    ['untouched', null, 32768],
    ['edited', '49152', 49152],
  ]) {
    test(`@in-memory-test ${label} numeric Rapid context remains explicit and updates its mirror`, async ({ page }) => {
      await openSeeded(page, {
        ...SEED,
        context_size: 65536,
        rapid_mlx: { ...SEED.rapid_mlx, context_length: 32768 },
      });
      await expect(page.locator('#modal-context-size')).toHaveValue('32768');
      if (editedContext !== null) await page.locator('#modal-context-size').fill(editedContext);

      const body = await captureConfirmedSave(page);
      expect(body.rapid_mlx.context_length).toBe(expectedContext);
      expect(body.context_size).toBe(expectedContext);
      await openSeeded(page, body);
      await expect(page.locator('#modal-context-size')).toHaveValue(String(expectedContext));
    });
  }

  for (const [label, context, expectedContext] of [
    ['missing', {}, ''],
    ['null', { context_length: null }, ''],
    ['numeric', { context_length: 32768 }, '32768'],
  ]) {
    test(`@in-memory-test seeded-new Rapid ${label} context ignores the legacy mirror`, async ({ page }) => {
      await page.goto('/');
      await page.waitForLoadState('networkidle');
      await page.evaluate(async (seed) => {
        const mod = await import('/js/features/presets.js');
        mod.openPresetModal('new', 'context', seed);
      }, {
        ...SEED,
        context_size: 65536,
        rapid_mlx: { ...SEED.rapid_mlx, ...context },
      });
      await expect(page.locator('#modal-context-size')).toHaveValue(expectedContext);
    });
  }

  for (const { label, context, hydrated, edited, expected } of [
    { label: 'numeric', context: { context_size: 65536 }, hydrated: '65536', expected: 65536 },
    { label: 'edited numeric', context: { context_size: 65536 }, hydrated: '65536', edited: '49152', expected: 49152 },
    { label: 'missing default', context: {}, hydrated: '128000', expected: 128000 },
    { label: 'null default', context: { context_size: null }, hydrated: '128000', expected: 128000 },
    { label: 'cleared default', context: { context_size: 65536 }, hydrated: '65536', edited: '', expected: 128000 },
  ]) {
    test(`@in-memory-test llama.cpp ${label} context behavior is unchanged`, async ({ page }) => {
      await openSeeded(page, {
        name: 'llama context probe',
        backend: 'llama_cpp',
        model_path: '/tmp/context-probe.gguf',
        ...context,
      });
      await expect(page.locator('#modal-context-size')).toHaveValue(hydrated);
      if (edited !== undefined) await page.locator('#modal-context-size').fill(edited);

      const body = await captureConfirmedSave(page);
      expect(body.context_size).toBe(expected);
      expect(body.rapid_mlx).toBeUndefined();
      await openSeeded(page, body);
      await expect(page.locator('#modal-context-size')).toHaveValue(String(expected));
    });
  }

  // The generalisation of the bug above. Every Rapid control in the save path used the
  // `if (value) out.x = value` idiom, so "(unset)" and "Auto" were unreachable states on any
  // preset that already had a value -- the spread restored the old one and it kept reaching
  // argv. This walks the whole set rather than the four that were noticed first.
  test('@in-memory-test every Rapid control can be returned to its unset state', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.evaluate(async () => {
      const { sessionState } = await import('/js/core/app-state.js');
      const mod = await import('/js/features/presets.js');
      sessionState.presets = [{
        id: 'probe', name: 'fully set', backend: 'rapid_mlx',
        rapid_mlx: {
          port: 8080, model_source: '/tmp/Qwen3-8B-4bit',
          enable_thinking: true, kv_cache_dtype: 'int8', turboquant_mode: 'k8v4',
          tool_call_parser: 'hermes', reasoning_parser: 'deepseek_r1', sampling_mode: 'coding',
          default_temperature: 0.7, default_top_p: 0.9, default_top_k: 40, default_min_p: 0.05,
          default_repetition_penalty: 1.1, default_presence_penalty: 0.5, max_tokens: 4096,
        },
      }];
      const sel = document.getElementById('preset-select');
      sel.innerHTML = '<option value="probe">fully set</option>';
      sel.value = 'probe';
      mod.openPresetModal('edit');
    });
    // Return every control to the option that means "do not send this".
    await page.locator('#preset-modal .preset-nav-item[data-section="generation"]').click();
    await page.selectOption('#modal-rapid-enable-thinking', '');
    await page.selectOption('#modal-rapid-tool-call-parser', '');
    await page.selectOption('#modal-rapid-reasoning-parser', '');
    await page.selectOption('#modal-rapid-sampling-mode', 'auto');
    for (const id of ['modal-temperature', 'modal-top-p', 'modal-top-k', 'modal-min-p',
                      'modal-repeat-penalty', 'modal-presence-penalty']) {
      await page.fill(`#${id}`, '');
    }
    await page.locator('#preset-modal .preset-nav-item[data-section="context"]').click();
    await page.selectOption('#modal-rapid-kv-cache-dtype', '');
    await page.selectOption('#modal-rapid-turboquant-mode', 'auto');
    await page.locator('#preset-modal .preset-nav-item[data-section="generation"]').click();
    await page.fill('#modal-max-tokens', '');

    let body = null;
    await page.route('**/api/presets/**', async (route, request) => {
      if (request.method() === 'PUT') body = request.postDataJSON();
      await route.fulfill({ status: 200, contentType: 'application/json', body: '{"id":"probe"}' });
    });
    for (const _ of [0, 1]) {
      await page.evaluate(async () => {
        const mod = await import('/js/features/presets.js');
        await mod.savePreset(new Event('submit'));
      });
    }

    expect(body, 'save request was never issued').not.toBeNull();
    const stuck = Object.entries({
      enable_thinking: body.rapid_mlx.enable_thinking,
      kv_cache_dtype: body.rapid_mlx.kv_cache_dtype,
      turboquant_mode: body.rapid_mlx.turboquant_mode,
      tool_call_parser: body.rapid_mlx.tool_call_parser,
      reasoning_parser: body.rapid_mlx.reasoning_parser,
      sampling_mode: body.rapid_mlx.sampling_mode,
      default_temperature: body.rapid_mlx.default_temperature,
      default_top_p: body.rapid_mlx.default_top_p,
      default_top_k: body.rapid_mlx.default_top_k,
      default_min_p: body.rapid_mlx.default_min_p,
      default_repetition_penalty: body.rapid_mlx.default_repetition_penalty,
      default_presence_penalty: body.rapid_mlx.default_presence_penalty,
      max_tokens: body.rapid_mlx.max_tokens,
    }).filter(([, v]) => v !== null && v !== undefined);
    expect(stuck, 'controls cleared by the user but still carrying their old value').toEqual([]);
  });

  for (const { label, seedConcurrency, selectAuto, expectedMaxNumSeqs } of [
    {
      label: 'untouched new preset keeps the single-stream default',
      seedConcurrency: {}, selectAuto: false, expectedMaxNumSeqs: 1,
    },
    {
      label: 'explicitly selecting Auto on a new preset writes null',
      seedConcurrency: {}, selectAuto: true, expectedMaxNumSeqs: null,
    },
    {
      label: 'a new preset seeded with explicit Auto does not pin the single-stream default',
      seedConcurrency: { max_num_seqs: null }, selectAuto: false, expectedMaxNumSeqs: null,
    },
  ]) {
    test(`@in-memory-test ${label}`, async ({ page }) => {
      await page.goto('/');
      await page.waitForLoadState('networkidle');
      await page.evaluate(async (seedConcurrency) => {
        const mod = await import('/js/features/presets.js');
        mod.openPresetModal('new', null, {
          name: 'bare', backend: 'rapid_mlx',
          rapid_mlx: { port: 8080, model_source: '/tmp/Qwen3-8B-4bit', ...seedConcurrency },
        });
      }, seedConcurrency);
      await page.locator('#preset-modal .preset-nav-item[data-section="context"]').click();

      // Unlike the wizard's 1/4 defaults, the new preset editor defaults only the
      // sequence count; GPU memory utilization and concurrent requests start on Auto.
      await expect(page.locator('#modal-rapid-gpu-memory-utilization')).toHaveValue('');
      await expect(page.locator('#modal-rapid-max-num-seqs')).toHaveValue(
        seedConcurrency.max_num_seqs === null ? '' : '1',
      );
      await expect(page.locator('#modal-rapid-max-concurrent-requests')).toHaveValue('');
      if (selectAuto) {
        await page.selectOption('#modal-rapid-gpu-memory-utilization', '');
        await page.selectOption('#modal-rapid-max-num-seqs', '');
        await page.selectOption('#modal-rapid-max-concurrent-requests', '');
        await expect(page.locator('#modal-rapid-max-num-seqs')).toHaveValue('');
      }

      let body = null;
      await page.route('**/api/presets', async (route, request) => {
        if (request.method() === 'POST') body = request.postDataJSON();
        await route.fulfill({ status: 200, contentType: 'application/json', body: '{"id":"probe"}' });
      });
      await page.evaluate(async () => {
        const mod = await import('/js/features/presets.js');
        await mod.savePreset(new Event('submit'));
      });

      expect(body, 'save request was never issued').not.toBeNull();
      // Auto writes null so it can override stored values during the rapid_mlx merge.
      expect(body.rapid_mlx.gpu_memory_utilization).toBeNull();
      expect(body.rapid_mlx.max_num_seqs).toBe(expectedMaxNumSeqs);
      expect(body.rapid_mlx.max_concurrent_requests).toBeNull();
    });
  }
});
