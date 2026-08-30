import { test, expect } from '@playwright/test';
import { dismissAuthShell, openSettings } from '../helpers.js';

async function enterMonitorView(page) {
  await dismissAuthShell(page);
  await page.evaluate(async () => {
    const { switchView } = await import('/js/features/setup-view.js');
    switchView('monitor');
  });
  await expect(page.locator('body')).not.toHaveClass(/setup-active/);
  await expect(page.locator('#view-monitor')).toBeVisible();
  await expect(page.locator('#endpoint-strip-monitor')).toBeVisible();
}

test.describe('app shell', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('html.modules-ready');
    await dismissAuthShell(page);
  });

  test('renders setup shell before monitor activation', async ({ page }) => {
    await expect(page.locator('body')).toHaveClass(/setup-active/);
    await expect(page.locator('.top-nav-bar')).toBeVisible();
    await expect(page.locator('.sidebar-nav')).toBeVisible();
    await expect(page.locator('#view-setup')).toBeVisible();
    // setup-pane-label elements are divs, not headings
    await expect(page.locator('.setup-pane-label').getByText('Connect to a running model')).toBeVisible();
    await expect(page.locator('.setup-pane-label').getByText('Local Server')).toBeVisible();
  });

  test('top status endpoint is read-only and edit control is in dashboard', async ({ page }) => {
    await enterMonitorView(page);
    await page.evaluate(() => {
      const endpointUrl = document.getElementById('endpoint-url');
      if (endpointUrl) endpointUrl.textContent = 'http://127.0.0.1:8001';
    });
    await expect(page.locator('.endpoint-url')).toBeVisible();
    await expect(page.locator('.endpoint-url')).not.toHaveJSProperty('tagName', 'INPUT');
    await expect(page.locator('#server-endpoint')).toBeEditable();
  });

  test('sidebar page tabs switch server, chat, and logs', async ({ page }) => {
    await enterMonitorView(page);

    await page.locator('.sidebar-btn[data-tab="chat"]').click();
    await expect(page.locator('#page-chat')).toBeVisible();
    await expect(page.locator('#page-server')).not.toBeVisible();

    await page.locator('.sidebar-btn[data-tab="logs"]').click();
    await expect(page.locator('#page-logs')).toBeVisible();
    await expect(page.locator('#page-chat')).not.toBeVisible();

    await page.locator('.sidebar-btn[data-tab="server"]').click();
    await expect(page.locator('#page-server')).toBeVisible();
  });

  test('log font controls resize the no-wrap console and persist the setting', async ({ page }) => {
    await enterMonitorView(page);
    await page.getByRole('button', { name: /logs/i }).click();
    await page.evaluate(() => {
      document.getElementById('page-logs')?.classList.remove('logs-empty-mode');
      const line = document.createElement('div');
      line.className = 'log-line';
      line.textContent = 'a long log line that should remain on one line';
      document.getElementById('log-panel')?.appendChild(line);
    });

    const panel = page.locator('#log-panel');
    const increase = page.getByRole('button', { name: 'Increase log font size' });
    await expect(page.locator('#log-font-size-btn')).toHaveText('13px');
    await increase.click();
    await expect(page.locator('#log-font-size-btn')).toHaveText('14px');
    await expect(panel).toHaveCSS('font-size', '14px');
    await expect(page.locator('.log-line')).toHaveCSS('white-space', 'pre');
    await expect.poll(() => page.evaluate(() => localStorage.getItem('llama-monitor-log-font-size'))).toBe('14');

    await page.reload();
    await page.waitForSelector('html.modules-ready');
    await dismissAuthShell(page);
    await expect(page.locator('#log-font-size-btn')).toHaveText('14px');
  });

  test('log console keeps updating when the fixed-size backend buffer rotates', async ({ page }) => {
    await enterMonitorView(page);
    await page.getByRole('button', { name: /logs/i }).click();

    await page.evaluate(async () => {
      const { updateLogs } = await import('/js/features/dashboard-ws.js');
      const first = Array.from({ length: 500 }, (_, i) => `log line ${i}`);
      updateLogs({ logs: first });
      updateLogs({ logs: [...first.slice(1), 'log line 500'] });
    });

    const lines = page.locator('#log-panel .log-line');
    await expect(lines).toHaveCount(500);
    await expect(lines.first()).toHaveText('log line 1');
    await expect(lines.last()).toHaveText('log line 500');
  });
});

test.describe('modals and menus', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('html.modules-ready');
    await dismissAuthShell(page);
    // Dismiss any modals that may have opened during initialization
    await page.evaluate(() => {
      document.querySelectorAll('.modal-overlay.open, .keyboard-shortcut-overlay.open').forEach(el => {
        el.classList.remove('open');
        el.setAttribute('aria-hidden', 'true');
        el.inert = true;
      });
    });
  });

  test('settings opens and secondary tabs switch', async ({ page }) => {
    await openSettings(page);
    await expect(page.locator('#settings-modal')).toHaveClass(/open/);

    // Default active pane is now Session
    await expect(page.locator('#settings-session')).toBeVisible();

    // Switch to Loaders tab and confirm Runtime Configuration button exists
    const loadersTab = page.locator('.settings-tab', { hasText: 'Loaders' });
    await loadersTab.click();
    await expect(page.locator('#settings-loaders')).toBeVisible();
    await expect(page.getByRole('button', { name: /open runtime configuration/i })).toBeVisible();
  });

  test('font scale applies the explicit root baseline', async ({ page }) => {
    await openSettings(page, 'appearance');
    const scale = page.locator('#settings-appearance-font-scale');
    await expect(scale).toBeVisible();

    for (const value of ['0.9', '1', '1.2']) {
      await scale.evaluate((element, nextValue) => {
        element.value = nextValue;
        element.dispatchEvent(new Event('input', { bubbles: true }));
      }, value);
      const expected = `${Number(value) * 16}px`;
      await expect.poll(() => page.evaluate(() => getComputedStyle(document.documentElement).fontSize)).toBe(expected);
      await expect.poll(() => page.evaluate(() => getComputedStyle(document.body).fontSize)).toBe(expected);
    }
  });

  test('models modal opens and lists model discovery state', async ({ page }) => {
    // No dedicated Models button; open via JS helper
    await page.evaluate(async () => {
      const { openModelsModal } = await import('/js/features/models.js');
      openModelsModal();
    });
    await expect(page.locator('#models-modal')).toHaveClass(/open/);
    await expect(page.locator('#models-summary')).toBeVisible();
    await expect(page.locator('#models-list')).toBeVisible();
  });

  test('preset editor preserves default and explicit off fit states', async ({ page }) => {
    let submittedPreset = null;
    await page.route('**/api/presets', async route => {
      if (route.request().method() !== 'POST') {
        await route.continue();
        return;
      }
      submittedPreset = route.request().postDataJSON();
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ id: 'test-fit-disabled' }),
      });
    });

    await page.evaluate(async () => {
      const { openPresetModal } = await import('/js/features/presets.js');
      openPresetModal('new');
    });
    await page.fill('#modal-name', 'Fit disabled');
    await page.fill('#modal-model-path', '/tmp/model.gguf');
    await page.click('.preset-nav-item[data-section="context"]');
    await page.locator('#modal-fit-target').evaluate(input => { input.value = '2048'; });
    await expect(page.locator('#modal-fit-enabled')).toHaveValue('');
    await page.selectOption('#modal-fit-enabled', 'false');
    await page.selectOption('#modal-kv-unified', 'false');
    await page.locator('#preset-form').evaluate(form => form.requestSubmit());
    await expect.poll(() => submittedPreset).not.toBeNull();

    expect(submittedPreset.fit_enabled).toBe(false);
    expect(submittedPreset.fit_target).toBeNull();
    expect(submittedPreset.kv_unified).toBe(false);
  });

  test('preset editor installs the recommended Gemma 4 chat template', async ({ page }) => {
    await page.route('**/api/models/gguf-meta', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ ok: true, architecture: 'gemma4' }),
      });
    });
    await page.route('**/api/chat-template/install-hf', async route => {
      expect(route.request().postDataJSON()).toEqual({
        repo: 'google/gemma-4-31B-it',
        file: 'chat_template.jinja',
        name: 'gemma4-google-official',
      });
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          ok: true,
          path: '/tmp/chat-templates/gemma4-google-official.jinja',
          already_existed: false,
        }),
      });
    });

    await page.evaluate(async () => {
      const { openPresetModal } = await import('/js/features/presets.js');
      openPresetModal('new');
    });
    await page.fill('#modal-model-path', '/models/Gemma-4-31B-it-Q4_K_M.gguf');
    await page.click('#preset-recommended-chat-template-btn');
    await expect(page.locator('#modal-chat-template-file')).toHaveValue(
      '/tmp/chat-templates/gemma4-google-official.jinja',
    );
  });

  test('user menu opens and shows options', async ({ page }) => {
    await page.locator('#nav-user-btn').click();
    await page.waitForSelector('#nav-user-menu-items', { state: 'visible', timeout: 5000 });
    await expect(page.locator('.nav-user-menu')).toHaveClass(/open/);
    await expect(page.getByRole('link', { name: 'Open Settings' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Help & Keyboard Shortcuts' })).toBeVisible();
  });

  test('user menu open settings clicks', async ({ page }) => {
    await page.locator('#nav-user-btn').click();
    await page.waitForSelector('#nav-user-menu-items', { state: 'visible' });
    await page.getByRole('link', { name: 'Open Settings' }).click();
    await expect(page.locator('.nav-user-menu')).not.toHaveClass(/open/);
  });

  test('remote agent fix opens runtime configuration', async ({ page }) => {
    await enterMonitorView(page);
    await page.evaluate(() => {
      document.getElementById('agent-status').style.display = '';
      document.querySelector('.btn-agent-fix').style.display = '';
    });
    await page.locator('.btn-agent-fix').click();
    await expect(page.locator('#remote-agent-setup-modal')).toBeVisible();
    await expect(page.locator('#btn-agent-setup-upgrade')).toBeHidden();
  });

test('remote agent update banner exposes upgrade action', async ({ page }) => {
  await enterMonitorView(page);
  await page.evaluate(async () => {
    const { setWsData } = await import('/js/core/app-state.js');
    const { openRemoteAgentSetup } = await import('/js/features/remote-agent.js');
    setWsData({
      remote_agent_connected: true,
      remote_agent_health_reachable: true,
      remote_agent_update_available: true,
      remote_agent_version: '2.0.11',
      session_mode: 'attach',
      endpoint_kind: 'Remote',
      system: { cpu_temp_available: true, cpu_temp: 45 },
    });
    openRemoteAgentSetup();
  });
  await expect(page.locator('#remote-agent-setup-modal')).toBeVisible();
  await expect(page.locator('#agent-setup-status-alert')).toHaveClass(/warning/);
  await expect(page.locator('#agent-setup-status-alert-title')).toHaveText('Agent Update Available');
  await expect(page.locator('#btn-agent-setup-upgrade')).toBeVisible();
});

test('configuration explains local executable, GPU, and explicit SSH flow', async ({ page }) => {
    // Open config modal directly via JS (avoids flaky Settings modal interactions)
    await page.evaluate(async () => {
      const { openConfigModal } = await import('/js/features/config.js');
      openConfigModal();
    });

    await expect(page.locator('#config-modal')).toHaveClass(/open/);
    await expect(page.getByText('Local llama-server executable')).toBeVisible();
    await expect(page.getByText('actual executable file')).toBeVisible();
    await expect(page.getByText('These checks run on this Mac or workstation only.')).toBeVisible();
    await expect(page.getByText('Remote agent actions are opt-in.')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Guided SSH Setup' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Check Host' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Check Release' })).toBeVisible();
  });

  test('guided SSH setup builds a structured target without contacting host', async ({ page }) => {
    let detectCalls = 0;

    await page.route('**/api/remote-agent/detect', async route => {
      detectCalls += 1;
      const payload = route.request().postDataJSON || {};
      // Only validate ssh_target on calls that include it (Check Host call)
      if (payload.ssh_target !== undefined) {
        expect(payload.ssh_target).toBe('ssh://user@192.0.2.16:2222');
        expect(payload.ssh_connection).toMatchObject({
          host: '192.0.2.16',
          username: 'user',
          port: 2222,
        });
      }
      await route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({
          ok: true,
          os: 'linux',
          arch: 'x86_64',
          installed: false,
          reachable: false,
          install_path: '~/.config/llama-monitor/bin/llama-monitor',
          matching_asset: { name: 'llama-monitor-linux-x86_64', archive: false },
          latest_release: { tag_name: 'v0.5.1' },
        }),
      });
    });

    // Open config modal directly via JS
    await page.evaluate(async () => {
      const { openConfigModal } = await import('/js/features/config.js');
      openConfigModal();
    });

    await expect(page.locator('#config-modal')).toHaveClass(/open/);
    await page.getByRole('button', { name: 'Guided SSH Setup' }).click();
    await page.locator('#ssh-guide-host').fill('192.0.2.16');
    await page.locator('#ssh-guide-user').fill('user');
    await page.locator('#ssh-guide-port').fill('2222');
    await page.getByRole('button', { name: 'Preview Plan' }).click();
    await expect(page.locator('#ssh-guide-plan')).toContainText('ssh://user@192.0.2.16:2222');
    await expect(page.getByRole('button', { name: 'Scan Host Key' })).toBeVisible();
    expect(detectCalls).toBe(0);

    await page.getByRole('button', { name: 'Use These Settings' }).click();
    await expect(page.locator('#set-remote-agent-ssh-target')).toHaveValue('ssh://user@192.0.2.16:2222');
    expect(detectCalls).toBe(0);

    await page.getByRole('button', { name: 'Check Host' }).click();
    await expect.poll(() => detectCalls).toBe(1);
  });

  test('typing SSH target does not auto-detect remote host', async ({ page }) => {
    let detectCalls = 0;

    await page.route('/api/remote-agent/detect', async route => {
      detectCalls += 1;
      await route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({
          ok: true,
          os: 'linux',
          arch: 'x86_64',
          installed: false,
          reachable: false,
          install_path: '~/.config/llama-monitor/bin/llama-monitor',
          matching_asset: { name: 'llama-monitor-linux-x86_64', archive: false },
          latest_release: { tag_name: 'v0.5.1' },
        }),
      });
    });

    // Open config modal directly via JS
    await page.evaluate(async () => {
      const { openConfigModal } = await import('/js/features/config.js');
      openConfigModal();
    });

    await expect(page.locator('#config-modal')).toHaveClass(/open/);
    const sshSummary = page.getByText('SSH and Agent Details');
    await sshSummary.scrollIntoViewIfNeeded();
    await sshSummary.click();
    await page.locator('#set-remote-agent-ssh-target').fill('user@192.0.2.16');
    await page.waitForTimeout(250);
    expect(detectCalls).toBe(0);

    await page.getByRole('button', { name: 'Check Host' }).click();
    await expect.poll(() => detectCalls).toBe(1);
  });
});

test.describe('responsive shell', () => {
  test('attach form reveals model identity only for Rapid-MLX', async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('html.modules-ready');

    const backend = page.locator('#setup-endpoint-backend');
    const model = page.locator('#setup-endpoint-model');
    await expect(backend).toHaveValue('llama_cpp');
    await expect(model).toBeHidden();

    await backend.selectOption('rapid_mlx');
    await expect(model).toBeVisible();
    await expect(model).toHaveAttribute('placeholder', /auto-detect/i);

    await backend.selectOption('llama_cpp');
    await expect(model).toBeHidden();
  });

  test('mobile layout keeps navigation and endpoint form usable', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto('/');
    await page.waitForSelector('html.modules-ready');

    await expect(page.locator('body')).toHaveClass(/setup-active/);
    await expect(page.locator('.sidebar-nav')).toBeVisible();
    await expect(page.locator('#setup-endpoint-url')).toBeEditable();
    await expect(page.locator('#setup-attach-btn')).toBeVisible();
  });
});

test.describe('inference metric rendering', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('html.modules-ready');
    await dismissAuthShell(page);
    await enterMonitorView(page);
  });

  test('renders active slot, request, speculative, and sampler state from slot snapshots', async ({ page }) => {
    await page.evaluate(async () => {
      const { renderSlotGrid, renderDecodingConfig, updateRequestActivity, renderActivityRail,
               renderGenerationDetailItems, setChipState } = await import('/js/features/dashboard-render.js');
      const l = {
        slots_processing: 1,
        slots_idle: 0,
        active_task_id: 6686,
        slot_generation_tokens: 690,
        slot_generation_remaining: 10,
        slot_generation_limit: 700,
        slot_generation_available: true,
        slot_generation_active: true,
        context_capacity_tokens: 212992,
        context_high_water_tokens: 121713,
        slots: [{
          id: 0,
          n_ctx: 212992,
          is_processing: true,
          id_task: 6686,
          output_tokens: 690,
          output_remaining: 10,
          output_limit: 700,
          output_active: true,
          output_available: true,
          speculative_enabled: true,
          speculative_type: 'ngram_map_k',
          speculative_config: [
            { label: 'type', value: 'ngram_map_k' },
            { label: 'n_max', value: '48' },
            { label: 'p_min', value: '0.75' },
          ],
          sampler_stack: ['penalties', 'dry', 'top_n_sigma', 'top_k', 'typ_p', 'top_p', 'min_p', 'xtc', 'temperature'],
          sampler_config: [
            { label: 'top_k', value: '20' },
            { label: 'top_p', value: '0.95' },
            { label: 'temp', value: '0.6' },
          ],
        }],
      };

      renderSlotGrid(l, true);
      renderDecodingConfig(l, true, true);
      updateRequestActivity(6686, true, 690, Date.now());
      renderActivityRail(true);
      renderGenerationDetailItems(document.getElementById('m-generation-details'), [
        'task 6686',
        '700 output budget',
        '10 output tokens remaining',
        '1 busy · 0 idle',
      ]);
      setChipState(document.getElementById('m-slots-state'), 'active', 'live');
      setChipState(document.getElementById('m-activity-state'), 'active', 'live');
    });

    await expect(page.locator('#m-slot-grid')).toContainText('task 6686');
    await expect(page.locator('#m-slot-grid')).toContainText('690 output');
    await expect(page.locator('#m-speculative-chip')).toContainText('Speculative · ngram_map_k · n_max 48');
    await expect(page.locator('#m-sampler-params-inline')).toContainText('top_k');
    await expect(page.locator('#m-sampler-params-inline')).toContainText('top_p');
    await expect(page.locator('#m-activity-rail .activity-segment.active')).toBeVisible();
    await expect(page.locator('#m-activity-rail .activity-phase.prompt')).toBeVisible();
    await expect(page.locator('#m-activity-rail .activity-phase.generation')).toBeVisible();
    await expect(page.locator('#m-generation-details .generation-detail-chip')).toHaveCount(4);
  });

  test('request rail leaves completion markers for finished tasks', async ({ page }) => {
    await page.evaluate(async () => {
      const { updateRequestActivity, renderActivityRail } = await import('/js/features/dashboard-render.js');
      const appState = await import('/js/core/app-state.js');
      appState.requestActivity.splice(0);
      const now = Date.now();
      updateRequestActivity(7001, true, 0, now - 4000);
      updateRequestActivity(7001, true, 40, now - 2500);
      updateRequestActivity(7001, false, 80, now - 500);
      renderActivityRail(false);
    });

    await expect(page.locator('#m-activity-rail .activity-segment.complete')).toBeVisible();
    await expect(page.locator('#m-activity-rail .activity-marker')).toBeVisible();
  });

  test('smooths live output estimate across recent polling samples', async ({ page }) => {
    const rate = await page.evaluate(async () => {
      const { updateLiveOutputEstimate } = await import('/js/features/dashboard-render.js');
      const appState = await import('/js/core/app-state.js');
      Object.assign(appState.liveOutputTracker, { taskId: null, previousDecoded: null, previousMs: null, latestRate: 0, rates: [] });
      appState.metricSeries.liveOutput = [];
      const now = Date.now();
      updateLiveOutputEstimate(123, 0, true, now - 3000);
      updateLiveOutputEstimate(123, 100, true, now - 2000);
      return updateLiveOutputEstimate(123, 170, true, now - 1000);
    });

    expect(rate).toBeGreaterThan(80);
    expect(rate).toBeLessThan(90);
  });

  test('capability popover renders correctly when capacity is known', async ({ page }) => {
    await page.evaluate(async () => {
      const { renderCapabilityPopover } = await import('/js/features/dashboard-render.js');
      renderCapabilityPopover({
        capabilities: { inference: true },
        host_metrics_available: false,
        remote_agent_connected: false,
      }, {
        slots_processing: 1,
        slots_idle: 0,
        context_capacity_tokens: 212992,
      }, true, true);
    });

    // Assert popover content directly — renderCapabilityPopover populates DOM synchronously
    await expect(page.locator('#capability-popover')).toContainText('Context usage');
    await expect(page.locator('#capability-popover')).toContainText('live');

    // Click handler toggles open state and re-renders with live data
    await page.locator('#endpoint-status').click();
    await expect(page.locator('#endpoint-status')).toHaveAttribute('aria-expanded', 'true');
  });
});
