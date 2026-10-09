import { test, expect } from '@playwright/test';
import { dismissAuthShell, openSettings } from '../helpers.js';

test.describe('TLS / Certificates settings', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('html.modules-ready');
    await dismissAuthShell(page);
  });

  test('Certificates tab exists and is selectable', async ({ page }) => {
    await openSettings(page);
    await expect(page.locator('#settings-modal')).toHaveClass(/open/);

    const certsTab = page.locator('.settings-tab').getByText('Security');
    await expect(certsTab).toBeVisible();

    await certsTab.click();
    await expect(page.locator('#settings-security')).toBeVisible();
  });

  test('TLS mode controls exist', async ({ page }) => {
    await openSettings(page);
    await expect(page.locator('#settings-modal')).toHaveClass(/open/);

    const certsTab = page.locator('.settings-tab').getByText('Security');
    await certsTab.click();
    await expect(page.locator('#settings-security')).toBeVisible();

    // Wait for loadTlsConfig to complete (status text gets populated)
    await expect(page.locator('#tls-status-text')).toBeVisible({ timeout: 15000 });
    await expect(page.locator('#tls-status-text')).toContainText(/^\s*TLS:/);

    // Helper: switch cert mode using the authoritative settings function
    const setCertMode = (mode) =>
      page.evaluate(async (m) => {
        const mod = await import('/js/features/settings.js');
        mod.setActiveCertMode(m);
      }, mode);

    // Select "No HTTPS" mode
    await setCertMode('none');
    await expect(page.locator('#btn-disable-tls')).toBeVisible();

    // Select "Self-Signed" mode
    await setCertMode('self-signed');
    const selfSignedBtn = page.locator('#btn-generate-self-signed');
    await expect(selfSignedBtn).toBeVisible();

    // Select "Bring Your Own Key" mode
    await setCertMode('custom');
    await expect(page.locator('#btn-apply-custom-cert')).toBeVisible();
  });

  test('ACME section exists with required controls', async ({ page }) => {
    await openSettings(page);
    await expect(page.locator('#settings-modal')).toHaveClass(/open/);

    const certsTab = page.locator('.settings-tab').getByText('Security');
    await certsTab.click();
    await expect(page.locator('#settings-security')).toBeVisible();

    // Wait for loadTlsConfig to complete, then switch to ACME mode
    await expect(page.locator('#tls-status-text')).toBeVisible({ timeout: 15000 });
    await expect(page.locator('#tls-status-text')).toContainText(/^\s*TLS:/);

    await page.evaluate(async (m) => {
      const mod = await import('/js/features/settings.js');
      mod.setActiveCertMode(m);
    }, 'acme');
    await expect(page.locator('#cert-mode-acme')).toBeVisible();

    // Domain input
    const fqdnInput = page.locator('#acme-fqdn');
    await expect(fqdnInput).toBeVisible();
    await expect(fqdnInput).toHaveAttribute('placeholder', /foundry\.example\.com/i);

    // Environment controls - verify they exist in the ACME panel
    const stagingRadio = page.locator('#acme-env-staging');
    await expect(stagingRadio).toBeAttached();

    const prodRadio = page.locator('#acme-env-production');
    await expect(prodRadio).toBeAttached();

    // DNS provider dropdown with multiple options
    const providerSelect = page.locator('#acme-dns-provider');
    await expect(providerSelect).toBeAttached();
    const options = await providerSelect.locator('option').all();
    expect(options.length).toBeGreaterThan(3);

    // Credentials grid and add button
    const grid = page.locator('#acme-credentials-grid');
    await expect(grid).toBeAttached();
    const addBtn = page.locator('#acme-add-credential');
    await expect(addBtn).toBeAttached();
    await expect(addBtn).toContainText(/add field/i);
  });

  test('switching TLS mode updates UI', async ({ page }) => {
    await openSettings(page);
    await expect(page.locator('#settings-modal')).toHaveClass(/open/);

    const certsTab = page.locator('.settings-tab').getByText('Security');
    await certsTab.click();
    await expect(page.locator('#settings-security')).toBeVisible();

    // Wait for loadTlsConfig to complete
    const tlsStatusText = page.locator('#tls-status-text');
    await expect(tlsStatusText).toBeVisible({ timeout: 15000 });
    await expect(page.locator('#tls-status-text')).toContainText(/^\s*TLS:/);

    // Exercise the mode pills as well as the action buttons with real clicks.
    const setCertMode = (mode) =>
      page.locator(`#cert-mode-pills .cert-mode-pill[data-mode="${mode}"]`).click();

    // Select "No HTTPS" mode and click "Disable TLS"
    await setCertMode('none');
    const disableBtn = page.locator('#btn-disable-tls');
    await expect(disableBtn).toBeVisible();
    // Observe the refresh applying, even when the status was already disabled.
    const statusRefreshed = page.evaluate(() => new Promise(resolve => {
      const status = document.getElementById('tls-status-text');
      const observer = new MutationObserver(() => {
        observer.disconnect();
        resolve();
      });
      observer.observe(status, { childList: true, characterData: true, subtree: true });
    }));
    await Promise.all([statusRefreshed, disableBtn.click()]);
    await expect(tlsStatusText).toHaveText('TLS: Disabled (HTTP only)');

    // Select "Self-Signed" mode and click "Generate self-signed"
    await setCertMode('self-signed');
    const generateSelfSigned = page.locator('#btn-generate-self-signed');
    await expect(generateSelfSigned).toBeVisible();
    await generateSelfSigned.click();
    await expect(tlsStatusText).toHaveText('TLS: Enabled (Self-signed)');

    // Select ACME mode and confirm controls exist
    await setCertMode('acme');
    await expect(page.locator('#acme-fqdn')).toBeAttached();
    await expect(page.locator('#acme-dns-provider')).toBeAttached();
  });
});
