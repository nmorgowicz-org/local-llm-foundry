import { test, expect } from '@playwright/test';

// A chat template tracked against its upstream repo used to announce "Chat template changes
// available" in a toast that offered nothing to do about it, never reached the notification
// bell, and came back on every tab focus. These tests pin the replacement: an update is
// announced once, with Update and Review actions in both the toast and the bell, applying it
// reinstalls from upstream's current `main`, and the retired no-JSON variant is left alone.
// Every /api/chat-template route is mocked and anything unlisted is refused, so no test can
// touch real templates or the network.
test.describe('Chat template updates', () => {
  const json = (route, body, status = 200) => route.fulfill({
    status, contentType: 'application/json', body: JSON.stringify(body),
  });

  const FROGGERIC = {
    name: 'qwen-froggeric-fixed',
    path: '/tmp/test-templates/qwen-froggeric-fixed.jinja',
    source_url: 'https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/blob/main/chat_template.jinja',
    fetch_url: 'https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/raw/23a40b0b/chat_template.jinja',
    legacy_variant: false,
  };
  const RETIRED = {
    name: 'qwen-froggeric-fixed-no_json',
    path: '/tmp/test-templates/qwen-froggeric-fixed-no_json.jinja',
    source_url: FROGGERIC.source_url,
    fetch_url: FROGGERIC.fetch_url,
    legacy_variant: true,
  };
  const UPDATE_AVAILABLE = {
    ok: true,
    changed: true,
    name: FROGGERIC.name,
    current_sha256: 'sha-v22-5',
    installed_sha256: 'sha-v21-3',
    installed_version: 'qwen3.6-froggeric-v21.3',
    upstream_version: 'qwen3.8-froggeric-v22.5',
    installed_revision: '23a40b0bd4d197c31d39e3c442fd2cd6100b3971',
    upstream_revision: '855bffc49448e299789730ff92c9b8d834d6cc14',
  };
  const EXPECTED_INSTALL = {
    repo: 'froggeric/Qwen-Fixed-Chat-Templates',
    file: 'chat_template.jinja',
    name: 'qwen-froggeric-fixed',
    force: true,
  };

  // Keeps the page's own scheduled check from running, so the test decides when a check happens.
  async function open(page) {
    await page.addInitScript(() => {
      localStorage.setItem('template_autoupdater_lastCheck', String(Date.now()));
    });
    await page.goto('/');
    await page.waitForLoadState('networkidle');
  }

  async function mockTemplateApi(page) {
    const seen = { checked: [], installs: [] };
    // Registered first, so every specific route below takes precedence over it.
    await page.route('**/api/chat-template/**', route => route.abort('blockedbyclient'));
    await page.route('**/api/chat-template/active', route => json(route, { ok: true, templates: [FROGGERIC, RETIRED] }));
    await page.route('**/api/chat-template/check-update', route => {
      seen.checked.push(route.request().postDataJSON().path);
      return json(route, UPDATE_AVAILABLE);
    });
    await page.route('**/api/chat-template/install-hf', route => {
      seen.installs.push(route.request().postDataJSON());
      return json(route, {
        ok: true, path: FROGGERIC.path, already_existed: false,
        revision: UPDATE_AVAILABLE.upstream_revision, source_url: FROGGERIC.source_url,
      });
    });
    return seen;
  }

  const runCheck = page => page.evaluate(async () => {
    const mod = await import('/js/features/template-autoupdater.js');
    return mod.checkTemplateUpdates();
  });

  test('@in-memory-test an update is announced with Update and Review, once, and Update applies it', async ({ page }) => {
    await open(page);
    const seen = await mockTemplateApi(page);

    const result = await runCheck(page);
    expect(result.changedTemplates.map(item => item.name)).toEqual(['qwen-froggeric-fixed']);
    // The retired variant has no upstream of its own, so it is never even checked.
    expect(seen.checked).toEqual([FROGGERIC.path]);

    const toast = page.locator('.toast-with-actions').filter({ hasText: 'Chat template update available' });
    await expect(toast).toBeVisible();
    await expect(toast).toContainText("froggeric's Fixed Template");
    await expect(toast).toContainText('v21.3 → v22.5');
    await expect(toast.locator('[data-action="update"]')).toBeVisible();
    await expect(toast.locator('[data-action="review"]')).toBeVisible();

    // The same announcement is in the bell, with the same actions, enabled.
    await page.locator('#nav-notifications-btn').click();
    const entry = page.locator('.nav-notification-item').filter({ hasText: 'Chat template update available' });
    await expect(entry).toHaveCount(1);
    await expect(entry.locator('.nav-notification-action', { hasText: 'Update' })).toBeEnabled();
    await expect(entry.locator('.nav-notification-action', { hasText: 'Review' })).toBeEnabled();

    // Checking again for the same upstream version stays quiet.
    await runCheck(page);
    await expect(toast).toHaveCount(1);
    await expect(entry).toHaveCount(1);

    // Update reinstalls from upstream's current main: never a pinned revision.
    await toast.locator('[data-action="update"]').click();
    await expect.poll(() => seen.installs.length).toBe(1);
    expect(seen.installs[0]).toEqual(EXPECTED_INSTALL);
    await expect(page.locator('.toast').filter({ hasText: "froggeric's Fixed Template updated" })).toBeVisible();

    // The bell entry is resolved and the pending status is forgotten.
    await expect(entry).toHaveCount(0);
    const pending = await page.evaluate(() => JSON.parse(localStorage.getItem('template_autoupdater_lastStatus')));
    expect(pending.templates_with_updates).toEqual([]);
  });

  test('@in-memory-test the bell Update button applies the update too', async ({ page }) => {
    await open(page);
    const seen = await mockTemplateApi(page);
    await runCheck(page);

    await page.locator('#nav-notifications-btn').click();
    const entry = page.locator('.nav-notification-item').filter({ hasText: 'Chat template update available' });
    await entry.locator('.nav-notification-action', { hasText: 'Update' }).click();

    await expect.poll(() => seen.installs.length).toBe(1);
    expect(seen.installs[0]).toEqual(EXPECTED_INSTALL);
    await expect(entry).toHaveCount(0);
  });

  test('@in-memory-test an up-to-date template announces nothing', async ({ page }) => {
    await open(page);
    await mockTemplateApi(page);
    await page.route('**/api/chat-template/check-update', route => json(route, { ok: true, changed: false, name: FROGGERIC.name }));

    const result = await runCheck(page);
    expect(result.changedTemplates).toEqual([]);
    await expect(page.locator('.toast-with-actions').filter({ hasText: 'Chat template update' })).toHaveCount(0);
  });

  test('@in-memory-test Manage template offers Update after a check, and has no Transform row', async ({ page }) => {
    await open(page);
    const seen = await mockTemplateApi(page);
    await page.route('**/api/chat-template/releases**', route => json(route, {
      ok: true,
      releases: [{
        sha256: 'sha-v21-3', revision: '23a40b0bd4d197c31d39e3c442fd2cd6100b3971', source_url: FROGGERIC.source_url,
        fetch_url: FROGGERIC.fetch_url, installed_at: '2026-08-06T11:00:10Z', file: 'qwen-froggeric-fixed-d203.jinja',
        template_version: 'qwen3.6-froggeric-v21.3',
      }],
      active_sha256: 'sha-v21-3',
    }));
    await page.route('**/api/chat-template/discussions**', route => json(route, { ok: true, discussions: [], source_repo: 'froggeric/Qwen-Fixed-Chat-Templates' }));
    await page.route('**/api/chat-template/upstream-history**', route => json(route, { ok: true, versions: [] }));

    await page.evaluate(async (path) => {
      const mod = await import('/js/features/chat-template-panel.js');
      void mod.openChatTemplateManageModal({
        tplName: 'qwen-froggeric-fixed',
        tplRepo: 'froggeric/Qwen-Fixed-Chat-Templates',
        currentPath: path,
        activePath: path,
      });
    }, FROGGERIC.path);

    const updates = page.locator('#chat-template-lifecycle-updates');
    await expect(page.locator('#chat-template-lifecycle-version')).toContainText('qwen3.6-froggeric-v21.3');
    await expect(page.locator('#chat-template-lifecycle-version')).not.toContainText('Transform');

    await updates.getByRole('button', { name: 'Check for updates' }).click();
    await expect(updates).toContainText('Update available (v21.3 → v22.5).');
    await updates.getByRole('button', { name: 'Update', exact: true }).click();

    await expect.poll(() => seen.installs.length).toBe(1);
    expect(seen.installs[0]).toEqual(EXPECTED_INSTALL);
    await expect(updates).toContainText('Updated.');
  });

  test('@in-memory-test Qwen offers froggeric and Sharp, both tracking main', async ({ page }) => {
    await page.goto('/');
    const registry = await page.evaluate(async () => {
      const mod = await import('/js/features/chat-template-registry.js');
      const qwen = mod.getTemplatesForFamily('qwen');
      return {
        names: qwen.map(tpl => tpl.name),
        defaultName: mod.getDefaultTemplateForFamily('qwen').name,
        pinned: qwen.filter(tpl => tpl.revision || tpl.transformed || tpl.version).map(tpl => tpl.name),
        sharp: mod.buildCommunityTemplateInstallRequest(mod.findTemplateByName('qwen-sharp')),
        qwenSuffix: mod.provenanceSuffix(qwen[0], 'qwen'),
        gemmaSuffix: mod.provenanceSuffix(mod.getTemplatesForFamily('gemma4')[0], 'gemma4'),
      };
    });
    expect(registry.names).toEqual(['qwen-froggeric-fixed', 'qwen-sharp']);
    expect(registry.defaultName).toBe('qwen-froggeric-fixed');
    expect(registry.pinned).toEqual([]);
    expect(registry.sharp).toEqual({
      endpoint: '/api/chat-template/install-hf',
      body: { repo: 'peculiar-ragdoll/Qwen-Sharp-Chat-Templates', file: 'chat_template.jinja', name: 'qwen-sharp' },
    });
    // Two community templates are told apart by name; a mixed family still gets a label.
    expect(registry.qwenSuffix).toBe('');
    expect(registry.gemmaSuffix).toBe(' (Official)');
  });

  test('@in-memory-test the preset editor lets you pick Sharp for a Qwen model', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.evaluate(async () => {
      const { openPresetModal } = await import('/js/features/presets.js');
      openPresetModal('new', undefined, { name: 'Qwen seed', family: 'qwen3_6', model_path: '', backend: 'llama_cpp' });
    });

    const choice = page.locator('#preset-chat-template-choice');
    await expect(choice).toBeVisible();
    await expect(choice.locator('option')).toHaveText(["froggeric's Fixed Template", 'Sharp Template (peculiar-ragdoll)']);

    // The chosen template is the one installed.
    const installs = [];
    await page.route('**/api/chat-template/install-hf', route => {
      installs.push(route.request().postDataJSON());
      return json(route, { ok: true, path: '/tmp/test-templates/qwen-sharp.jinja', already_existed: false });
    });
    await page.route('**/api/chat-template/releases**', route => json(route, { ok: false }));
    await choice.selectOption('qwen-sharp');
    await page.locator('#preset-recommended-chat-template-btn').click();
    await expect.poll(() => installs.length).toBe(1);
    expect(installs[0]).toMatchObject({ repo: 'peculiar-ragdoll/Qwen-Sharp-Chat-Templates', name: 'qwen-sharp' });
    await expect(page.locator('#modal-chat-template-file')).toHaveValue('/tmp/test-templates/qwen-sharp.jinja');
  });
});
