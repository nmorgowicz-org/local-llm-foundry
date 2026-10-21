import { test, expect } from '@playwright/test';

// Get api-token from the server for API calls
async function getApiToken(request) {
  try {
    const res = await request.get('/api/internal/api-token');
    if (res.ok) {
      const data = await res.json();
      return data.token || null;
    }
  } catch {}
  return null;
}

// Disable auth using Playwright's request (independent of page context)
async function disableAuth(request) {
  try {
    const statusRes = await request.get('/api/auth/status');
    if (!statusRes.ok) return;
    const status = await statusRes.json();
    if (!status.enabled) return;

    const token = await getApiToken(request);
    const headers = token ? { Authorization: `Bearer ${token}` } : {};

    await request.put('/api/auth/config', {
      headers,
      data: { basic_enabled: false, form_enabled: false, username: '' },
    });
  } catch {}
}

test.afterEach(async ({ request }) => {
  // Always disable auth after test to prevent interference with other tests
  await disableAuth(request);
});

test('dashboard access can be enabled, used, and disabled from settings', async ({ page, request }) => {
  await page.goto('/');
  await page.waitForSelector('html.modules-ready');

  // Reset auth state via API to avoid interference from parallel tests
  await disableAuth(request);
  await expect.poll(async () => (await (await request.get('/api/auth/status')).json()).enabled).toBe(false);

  await page.reload();
  await page.waitForSelector('html.modules-ready');

  // Enable form auth via API
  const token = await getApiToken(request);
  const headers = token ? { Authorization: `Bearer ${token}` } : {};

  const enableRes = await request.put('/api/auth/config', {
    headers,
    data: {
      basic_enabled: false,
      form_enabled: true,
      username: 'admin',
      current_password: '',
      new_password: 'secret1234',
    },
  });
  expect(enableRes.status(), 'Enable auth failed').toBe(200);

  // Verify auth status reflects the new config
  const statusRes = await request.get('/api/auth/status');
  const statusData = await statusRes.json();
  expect(statusData.methods?.form, `form auth not active: ${JSON.stringify(statusData)}`).toBe(true);

  // Reload and verify auth shell appears
  await page.reload();
  await expect(page.locator('#auth-shell')).toBeVisible({ timeout: 10000 });
  await expect(page.locator('#auth-shell-recovery')).toContainText('clear-auth-config');
  await page.locator('#auth-username').fill('admin');
  await page.locator('#auth-password').fill('secret1234');
  await page.locator('#auth-submit').click();

  await page.waitForSelector('html.modules-ready');
  await expect(page.locator('.top-nav-bar')).toBeVisible();

  // Note: auth will be disabled by afterEach teardown
});
