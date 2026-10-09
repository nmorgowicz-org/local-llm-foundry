import { test, expect } from '@playwright/test';

const CASES = [
  { name: 'flag false, placeholder zero is unavailable', gpu: { load: 0, load_available: false }, text: '—', width: '0%' },
  { name: 'flag false ignores a stale high load', gpu: { load: 95, load_available: false }, text: '—', width: '0%' },
  { name: 'flag true preserves real zero', gpu: { load: 0, load_available: true }, text: '0%', width: '0%' },
  { name: 'old protocol preserves real zero', gpu: { load: 0 }, text: '0%', width: '0%' },
  { name: 'old protocol preserves measured load', gpu: { load: 37 }, text: '37%', width: '37%' },
  { name: 'missing load is unavailable', gpu: { load_available: true }, text: '—', width: '0%' },
  { name: 'null load is unavailable', gpu: { load: null, load_available: true }, text: '—', width: '0%' },
];

function frame(gpu) {
  return JSON.stringify({
    server_running: true,
    capabilities: { gpu: true, system: false },
    gpu: { 'Test GPU': { temp: 40, vram_used: 1024, vram_total: 8192, ...gpu } },
    llama: {},
  });
}

for (const c of CASES) {
  test(`compact GPU load: ${c.name}`, async ({ page }) => {
    await page.routeWebSocket('**/ws', ws => ws.send(frame(c.gpu)));
    await page.goto('/compact');
    const row = page.locator('.gpu-section .metric-row').first();
    await expect(row.locator('.metric-value')).toHaveText(c.text);
    await expect.poll(() => row.locator('.gpu-load').evaluate(el => el.style.width)).toBe(c.width);
    if (c.text === '—') await expect(row.locator('.gpu-load')).toHaveClass('metric-bar gpu-load');
    await expect(page.locator('.gpu-section .metric-row').nth(2).locator('.metric-value')).toHaveText('1.0 / 8.0 GB');
  });
}

test('compact GPU clears stale load and severity, then recovers on a fresh sample', async ({ page }) => {
  let socket;
  await page.routeWebSocket('**/ws', ws => {
    socket = ws;
    ws.send(frame({ load: 95, load_available: true }));
  });
  await page.goto('/compact');
  const row = page.locator('.gpu-section .metric-row').first();
  await expect(row.locator('.metric-value')).toHaveText('95%');
  await expect(row.locator('.gpu-load')).toHaveClass(/severity-/);

  socket.send(frame({ load: 95, load_available: false }));
  await expect(row.locator('.metric-value')).toHaveText('—');
  await expect(row.locator('.gpu-load')).toHaveClass('metric-bar gpu-load');
  await expect.poll(() => row.locator('.gpu-load').evaluate(el => el.style.width)).toBe('0%');

  socket.send(frame({ load: 0, load_available: true }));
  await expect(row.locator('.metric-value')).toHaveText('0%');
  socket.send(frame({ load: 37 }));
  await expect(row.locator('.metric-value')).toHaveText('37%');
});
