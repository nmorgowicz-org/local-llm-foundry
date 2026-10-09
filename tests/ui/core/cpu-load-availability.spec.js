import { test, expect } from '@playwright/test';
import { dismissAuthShell } from '../helpers.js';

// cpu_load_available=false marks the first sysinfo sampling interval, where cpu_load=0 is a
// placeholder. Absent (older agents) or true means the value is real, including a real 0.
const CASES = [
  { name: 'flag false → unavailable', sys: { cpu_load: 0, cpu_load_available: false }, text: '—' },
  { name: 'flag true, real 0 → 0%', sys: { cpu_load: 0, cpu_load_available: true }, text: '0%' },
  { name: 'flag absent, real 0 → 0%', sys: { cpu_load: 0 }, text: '0%' },
  { name: 'flag true, 37 → 37%', sys: { cpu_load: 37, cpu_load_available: true }, text: '37%' },
  { name: 'flag false ignores stale value', sys: { cpu_load: 37, cpu_load_available: false }, text: '—' },
];

test.describe('compact CPU load availability', () => {
  for (const c of CASES) {
    test(c.name, async ({ page }) => {
      await page.routeWebSocket('**/ws', ws => {
        ws.send(JSON.stringify({ server_running: true, capabilities: { gpu: false, system: true },
          system: { ram_total_gb: 16, ram_used_gb: 8, ...c.sys }, gpu: {}, llama: {} }));
      });
      await page.goto('/compact');
      await expect(page.locator('#cpu-load')).toHaveText(c.text);
    });
  }
});

test.describe('dashboard system card CPU load availability', () => {
  test.beforeEach(async ({ page }) => {
    await page.routeWebSocket('**/ws', ws => ws.close());
    await page.route('**/api/**', route => route.abort());
    await page.goto('/');
    await dismissAuthShell(page);
  });

  async function render(page, samples) {
    return page.evaluate(async samples => {
      const { renderSystemCard } = await import('/js/features/dashboard-render.js');
      // The legacy system card has no static markup: provide the nodes it writes to.
      document.getElementById('cpu-test-root')?.remove();
      const root = document.createElement('div');
      root.id = 'cpu-test-root';
      root.innerHTML = '<div id="system-card"></div><div id="sys-load-viz"></div>'
        + '<span id="sys-load-value"></span><svg id="sys-load-spark"></svg>';
      document.body.appendChild(root);
      for (const sys of samples) renderSystemCard({ ram_total_gb: 16, ram_used_gb: 8, ...sys }, true, 'local');
      return {
        text: document.getElementById('sys-load-value').textContent,
        sparkVisible: document.getElementById('sys-load-spark').style.visibility !== 'hidden',
      };
    }, samples);
  }

  test('flag false shows unavailable and never pushes a history sample', async ({ page }) => {
    const out = await render(page, Array(3).fill({ cpu_load: 0, cpu_load_available: false }));
    expect(out.text).toBe('\u2014');
    // Sparkline needs ≥2 history points; placeholders must not have been recorded.
    expect(out.sparkVisible).toBe(false);
  });

  test('flag true records a real 0 and shows 0%', async ({ page }) => {
    const out = await render(page, Array(3).fill({ cpu_load: 0, cpu_load_available: true }));
    expect(out.text).toBe('0%');
    expect(out.sparkVisible).toBe(true);
  });

  test('flag absent keeps the legacy real-value behavior', async ({ page }) => {
    const out = await render(page, Array(3).fill({ cpu_load: 12 }));
    expect(out.text).toBe('12%');
    expect(out.sparkVisible).toBe(true);
  });
});
