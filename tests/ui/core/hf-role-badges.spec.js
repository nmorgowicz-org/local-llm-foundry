import { test, expect } from '@playwright/test';

// Role badges used to come from KNOWN_CONVERTER_PATTERNS, a hardcoded regex list in
// hf-browse.js with three roles. It classified `Qwen/` as a converter -- Qwen is the original
// author of Qwen -- and the user had no way to correct it, while the seven-role, editable
// CommunitySourceCatalog sat on the server with no route to reach it. These tests pin the
// badges to the catalog instead.
test.describe('HF role badges come from the community source catalog', () => {
  // Mirrors the server's wire format: snake_case role ids, camelCase field names.
  const CATALOG = {
    ok: true,
    catalog: {
      entries: [
        {
          username: 'bartowski', displayName: 'bartowski', description: 'Standard GGUF quants.',
          role: 'gguf_quantizer', bundled: true,
        },
        {
          username: 'mlx-community', displayName: 'MLX Community', description: 'MLX conversions.',
          role: 'mlx_converter', bundled: true,
        },
        {
          username: 'unsloth', displayName: 'Unsloth', description: 'UD quants and finetunes.',
          role: 'original_author', alsoKnownFor: ['gguf_quantizer'], bundled: true,
        },
      ],
      preferences: {},
      version: 1,
    },
    roles: [
      { id: 'original_author', label: 'Original author', description: 'Created the original model weights or first fine-tune.' },
      { id: 'gguf_quantizer', label: 'GGUF quantizer', description: 'Produced GGUF quantized weights from this model.' },
      { id: 'mlx_converter', label: 'MLX converter', description: 'Converted or produced native MLX weights from this model.' },
    ],
  };

  const MODELS = [
    { id: 'Qwen/Qwen3-8B', tags: ['text-generation'], param_b: 8, downloads: 100 },
    { id: 'bartowski/Qwen3-8B-GGUF', tags: ['gguf'], param_b: 8, downloads: 90 },
    { id: 'mlx-community/Qwen3-8B-4bit', tags: ['mlx'], param_b: 8, downloads: 80 },
    { id: 'unsloth/Qwen3-8B-GGUF', tags: ['gguf'], param_b: 8, downloads: 70 },
  ];

  async function search(page, { catalog = CATALOG, models = MODELS } = {}) {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    await page.route('**/api/hf/community-sources', async (route) => {
      if (catalog === null) {
        await route.fulfill({ status: 500, contentType: 'application/json', body: '{}' });
        return;
      }
      await route.fulfill({
        status: 200, contentType: 'application/json', body: JSON.stringify(catalog),
      });
    });
    await page.route('**/api/hf/search', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ models, next_cursor: null }),
      });
    });

    return page.evaluate(async () => {
      const mod = await import('/js/features/hf-browse.js');
      // Each test gets a fresh catalog fetch; the module caches it for the page lifetime.
      mod._resetCommunitySourceCatalog();
      const container = document.createElement('div');
      container.id = 'role-badge-probe';
      document.body.appendChild(container);
      await mod.hfSearch({ query: 'qwen3', container, allActive: true });
      return [...container.querySelectorAll('.hf-sg-variant')].map(v => ({
        id: v.querySelector('.hf-sg-variant-name')?.getAttribute('title'),
        role: v.querySelector('.hf-sg-role-badge')?.className || '',
        label: v.querySelector('.hf-sg-role-badge')?.textContent || '',
        title: v.querySelector('.hf-sg-role-badge')?.getAttribute('title') || '',
      }));
    });
  }

  test('@in-memory-test Qwen is the original author, not a converter', async ({ page }) => {
    const rows = await search(page);
    const qwen = rows.find(r => r.id === 'Qwen/Qwen3-8B');
    expect(qwen, 'Qwen/Qwen3-8B was not rendered').toBeTruthy();
    expect(qwen.label).toBe('Original author');
    expect(qwen.role).toContain('hf-sg-role-badge--original-author');
  });

  test('@in-memory-test catalog roles drive the badge, with the role description as its title', async ({ page }) => {
    const rows = await search(page);
    const byId = Object.fromEntries(rows.map(r => [r.id, r]));

    expect(byId['bartowski/Qwen3-8B-GGUF'].label).toBe('GGUF quantizer');
    expect(byId['mlx-community/Qwen3-8B-4bit'].label).toBe('MLX converter');
    // The tooltip is the role's own description from the Rust enum, not a string retyped in JS.
    expect(byId['mlx-community/Qwen3-8B-4bit'].title)
      .toBe('Converted or produced native MLX weights from this model.');
  });

  // Unsloth authors finetunes and quantizes them. A single-role lookup would label every
  // Unsloth repo the same way; the repo's own format tags pick which of their roles applies.
  test('@in-memory-test a multi-role owner is resolved by the repo tags', async ({ page }) => {
    const rows = await search(page);
    const unsloth = rows.find(r => r.id === 'unsloth/Qwen3-8B-GGUF');
    expect(unsloth.label).toBe('GGUF quantizer');
  });

  // The catalog is an enhancement to the badges, not a prerequisite for search working.
  test('@in-memory-test search still renders when the catalog cannot be loaded', async ({ page }) => {
    const rows = await search(page, { catalog: null });
    expect(rows.length).toBe(4);
    // Unknown owners fall back to the repo-name heuristic rather than to a wrong role.
    expect(rows.find(r => r.id === 'Qwen/Qwen3-8B').label).toBe('Original author');
    expect(rows.find(r => r.id === 'bartowski/Qwen3-8B-GGUF').label).toBe('');
  });

  test('@in-memory-test selection payload preserves the Hub commit instead of mutable main', async ({ page }) => {
    await page.goto('/');
    const selection = await page.evaluate(async () => {
      const { buildSelectionPayload } = await import('/js/features/hf-browse.js');
      return buildSelectionPayload({
        id: 'mlx-community/Qwen3-8B-4bit',
        revision: 'abcdef1234567890',
        format: 'mlx',
      }, null);
    });
    expect(selection).toMatchObject({
      repoId: 'mlx-community/Qwen3-8B-4bit',
      revision: 'abcdef1234567890',
      format: 'mlx',
    });
  });

  test('@fake-data-bypass qualification and identity feed the shared evidence drawer', async ({ page }) => {
    const requests = [];
    await page.route('**/api/hf/qualify', async route => {
      requests.push(await route.request().postDataJSON());
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({
        repoId: 'mlx-community/Qwen3-8B-4bit', revision: 'a'.repeat(40),
        backendHint: 'rapid_mlx', backendQualified: true, qualificationReason: 'MLX config is compatible.',
        qualifiedAt: 1785456000, config: { architecture: 'Qwen3ForCausalLM', contextLength: 32768 }, errors: [],
      }) });
    });
    await page.route('**/api/hf/identity', async route => {
      requests.push(await route.request().postDataJSON());
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({
        repoId: 'mlx-community/Qwen3-8B-4bit', revision: 'a'.repeat(40), resolutionConfidence: 'catalog',
        roles: [{ role: 'mlx_converter', username: 'mlx-community', confidence: 'catalog' }], errors: [],
      }) });
    });
    await page.goto('/');
    await page.waitForSelector('html.modules-ready');
    await page.evaluate(async sha => {
      const { openHfEvidence } = await import('/js/features/hf-browse.js');
      await openHfEvidence('mlx-community/Qwen3-8B-4bit', sha, 'rapid_mlx');
    }, 'a'.repeat(40));
    await expect(page.locator('#evidence-drawer')).toHaveClass(/open/);
    await expect(page.locator('.evidence-drawer-summary')).toContainText('provisionally qualified');
    await page.locator('.evidence-drawer-details > summary').click();
    await expect(page.locator('.evidence-drawer-technical')).toContainText('MLX config is compatible.');
    expect(requests).toHaveLength(2);
    expect(requests.every(request => !('configDir' in request))).toBe(true);
  });

  // A failed search used to leave a bare "Search failed." with no way forward.
  test('@in-memory-test a rate-limited search counts down, then Retry re-runs it', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    let calls = 0;
    await page.route('**/api/hf/community-sources', route => route.fulfill({
      status: 200, contentType: 'application/json', body: JSON.stringify(CATALOG),
    }));
    await page.route('**/api/hf/search', route => {
      calls += 1;
      if (calls === 1) {
        return route.fulfill({
          status: 429,
          contentType: 'application/json',
          body: JSON.stringify({ ok: false, error: 'Search rate limit reached. Try again in 2s.', retry_after_secs: 2 }),
        });
      }
      return route.fulfill({
        status: 200, contentType: 'application/json', body: JSON.stringify({ models: MODELS, next_cursor: null }),
      });
    });
    await page.evaluate(() => {
      const container = document.createElement('div');
      container.id = 'retry-probe';
      document.body.appendChild(container);
      import('/js/features/hf-browse.js').then(mod => mod.hfSearch({ query: 'qwen3', container, allActive: true }));
    });
    const failure = page.locator('#retry-probe .hf-search-failure');
    await expect(failure).toContainText('Search rate limit reached');
    const retry = failure.locator('.hf-search-retry-btn');
    await expect(retry).toBeDisabled();
    await expect(retry).toContainText('Retry in');
    await expect(retry).toBeEnabled({ timeout: 6000 });
    await expect(retry).toHaveText('Retry');
    await retry.click();
    await expect(page.locator('#retry-probe .hf-sg-variant')).toHaveCount(4);
    await expect(page.locator('#retry-probe .hf-search-failure')).toHaveCount(0);
    expect(calls).toBe(2);
  });

  test('@in-memory-test a network error offers an immediate Retry', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    let calls = 0;
    await page.route('**/api/hf/community-sources', route => route.fulfill({
      status: 200, contentType: 'application/json', body: JSON.stringify(CATALOG),
    }));
    await page.route('**/api/hf/search', route => {
      calls += 1;
      if (calls === 1) return route.abort('failed');
      return route.fulfill({
        status: 200, contentType: 'application/json', body: JSON.stringify({ models: MODELS, next_cursor: null }),
      });
    });
    await page.evaluate(() => {
      const container = document.createElement('div');
      container.id = 'retry-probe';
      document.body.appendChild(container);
      import('/js/features/hf-browse.js').then(mod => mod.hfSearch({ query: 'qwen3', container, allActive: true }));
    });
    const retry = page.locator('#retry-probe .hf-search-retry-btn');
    await expect(retry).toBeEnabled();
    await retry.click();
    await expect(page.locator('#retry-probe .hf-sg-variant')).toHaveCount(4);
    expect(calls).toBe(2);
  });

  test('@in-memory-test MLX quant siblings group as variants of one model', async ({ page }) => {
    const models = [
      { id: 'nightmedia/Qwen3.8-27B-MindMeld-AREX-mxfp4-mlx', tags: ['mlx'], param_b: 27, downloads: 5 },
      { id: 'nightmedia/Qwen3.8-27B-MindMeld-AREX-mxfp8-mlx', tags: ['mlx'], param_b: 27, downloads: 4 },
      { id: 'nightmedia/Qwen3.8-27B-MindMeld-AREX-qx86-hi-mlx', tags: ['mlx'], param_b: 27, downloads: 3 },
      { id: 'mlx-community/Qwen3-8B-4bit', tags: ['mlx'], param_b: 8, downloads: 2 },
      { id: 'mlx-community/Qwen3-8B-8bit', tags: ['mlx'], param_b: 8, downloads: 1 },
    ];
    await search(page, { models });
    const groups = await page.evaluate(() => [...document.querySelectorAll('#role-badge-probe .hf-sg-base-name')]
      .map(n => n.textContent));
    expect(groups.filter(g => g.startsWith('Qwen3.8-27b-Mindmeld-Arex'))).toHaveLength(1);
    expect(groups.filter(g => g.startsWith('Qwen3-8b'))).toHaveLength(1);
  });

  test('@in-memory-test selecting an MLX variant hands its sibling quants to the advisor', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    const models = [
      { id: 'nightmedia/Qwen3.8-27B-MindMeld-AREX-mxfp4-mlx', tags: ['mlx'], param_b: 27, downloads: 5, model_size_bytes: 15_200_000_000 },
      { id: 'nightmedia/Qwen3.8-27B-MindMeld-AREX-mxfp8-mlx', tags: ['mlx'], param_b: 27, downloads: 4, model_size_bytes: 28_700_000_000 },
    ];
    await page.route('**/api/hf/community-sources', route => route.fulfill({
      status: 200, contentType: 'application/json', body: JSON.stringify(CATALOG),
    }));
    await page.route('**/api/hf/search', route => route.fulfill({
      status: 200, contentType: 'application/json', body: JSON.stringify({ models, next_cursor: null }),
    }));
    const selection = await page.evaluate(async () => {
      const mod = await import('/js/features/hf-browse.js');
      const container = document.createElement('div');
      document.body.appendChild(container);
      let picked = null;
      await mod.hfSearch({ query: 'mindmeld', container, allActive: true, onSelectModel: s => { picked = s; } });
      container.querySelector('.hf-sg-variant').click();
      return picked;
    });
    expect(selection.siblings.map(s => s.label).sort()).toEqual(['mxfp4', 'mxfp8']);
    expect(selection.siblings.map(s => s.size).sort((a, b) => a - b)).toEqual([15_200_000_000, 28_700_000_000]);
  });

  test('@in-memory-test every MindMeld repo lands in one group with quant pills and edition tags', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    const models = [
      { id: 'nightmedia/Qwen3.8-27B-MindMeld-AREX-mxfp4-mlx', tags: ['mlx'], param_b: 27, downloads: 5, model_size_bytes: 15_200_000_000 },
      { id: 'nightmedia/Qwen3.8-27B-MindMeld-AREX-mxfp8-mlx', tags: ['mlx'], param_b: 27, downloads: 4, model_size_bytes: 28_700_000_000 },
      { id: 'nightmedia/Qwen3.8-27B-MindMeld', tags: ['mlx'], param_b: 28, downloads: 3, model_size_bytes: 55_500_000_000 },
      { id: 'nightmedia/Qwen3.8-27B-MindMeld-qx64-hi-mlx', tags: ['mlx'], param_b: 27, downloads: 2, model_size_bytes: 20_900_000_000 },
      { id: 'nightmedia/gemma-4-E4B-Chronos-q8-hi-mlx', tags: ['mlx'], param_b: 4, downloads: 1, model_size_bytes: 9_350_000_000 },
    ];
    await page.route('**/api/hf/community-sources', route => route.fulfill({
      status: 200, contentType: 'application/json', body: JSON.stringify(CATALOG),
    }));
    await page.route('**/api/hf/search', route => route.fulfill({
      status: 200, contentType: 'application/json', body: JSON.stringify({ models, next_cursor: null }),
    }));
    const result = await page.evaluate(async () => {
      const mod = await import('/js/features/hf-browse.js');
      const container = document.createElement('div');
      document.body.appendChild(container);
      let picked = null;
      await mod.hfSearch({ query: 'mindmeld', container, allActive: true, onSelectModel: s => { picked = s; } });
      const groups = [...container.querySelectorAll('.hf-search-group')].map(g => ({
        name: g.querySelector('.hf-sg-base-name')?.textContent,
        rows: [...g.querySelectorAll('.hf-sg-variant')].map(v => ({
          id: v.querySelector('.hf-sg-variant-name')?.getAttribute('title'),
          pill: v.querySelector('.hf-sg-quant-pill')?.textContent,
          edition: v.querySelector('.hf-sg-variant-name')?.textContent,
        })),
      }));
      [...container.querySelectorAll('.hf-sg-variant')]
        .find(v => v.querySelector('.hf-sg-variant-name')?.getAttribute('title').endsWith('qx64-hi-mlx')).click();
      return { groups, picked };
    });
    const mind = result.groups.filter(g => g.name === 'Qwen3.8-27b-Mindmeld');
    expect(mind).toHaveLength(1);
    expect(mind[0].rows.map(r => r.pill).sort()).toEqual(['bf16', 'mxfp4', 'mxfp8', 'qx64-hi']);
    expect(mind[0].rows.filter(r => r.edition === 'AREX')).toHaveLength(2);
    expect(result.groups).toHaveLength(2);
    expect(result.picked.siblings.map(s => s.label).sort())
      .toEqual(['AREX mxfp4', 'AREX mxfp8', 'bf16', 'qx64-hi']);
  });

  test('@in-memory-test a failed follow-up page keeps earlier results and retries the same page', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    const cursors = [];
    await page.route('**/api/hf/community-sources', route => route.fulfill({
      status: 200, contentType: 'application/json', body: JSON.stringify(CATALOG),
    }));
    await page.route('**/api/hf/search', route => {
      const body = route.request().postDataJSON();
      cursors.push(body.cursor || null);
      if (!body.cursor) {
        return route.fulfill({
          status: 200, contentType: 'application/json',
          body: JSON.stringify({ models: MODELS.slice(0, 2), next_cursor: 'page-2' }),
        });
      }
      if (cursors.filter(c => c === 'page-2').length === 1) {
        return route.fulfill({ status: 500, contentType: 'application/json', body: '{}' });
      }
      return route.fulfill({
        status: 200, contentType: 'application/json',
        body: JSON.stringify({ models: MODELS.slice(2), next_cursor: null }),
      });
    });
    await page.evaluate(() => {
      const container = document.createElement('div');
      container.id = 'retry-probe';
      document.body.appendChild(container);
      import('/js/features/hf-browse.js').then(mod => mod.hfSearch({ query: 'qwen3', container, allActive: true }));
    });
    // Two results is under the visible-row threshold, so page 2 loads automatically; its
    // failure must leave page 1 on screen and offer a Retry for that same cursor.
    const retry = page.locator('#retry-probe .hf-search-retry-btn');
    await expect(retry).toBeEnabled();
    await expect(page.locator('#retry-probe .hf-sg-variant')).toHaveCount(2);
    await retry.click();
    await expect(page.locator('#retry-probe .hf-sg-variant')).toHaveCount(4);
    await expect(page.locator('#retry-probe .hf-search-failure')).toHaveCount(0);
    expect(cursors).toEqual([null, 'page-2', 'page-2']);
  });
});
