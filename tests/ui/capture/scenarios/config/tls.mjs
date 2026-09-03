// Scenario: tls
// Extracted from tests/ui/capture.mjs (Phase A3).
import { attachToServer } from '../../harness/attach.mjs';
import { gotoApp, switchTab } from '../../harness/browser.mjs';
import { sleep } from '../../harness/paths.mjs';
import { captureCloseUp, captureShot } from '../../harness/shot.mjs';

export default async function(ctx, options) {
    const { page, baseUrl } = ctx;
    await gotoApp(page, baseUrl);
    await attachToServer(page);

    await switchTab(page, 'chat');
    await sleep(500);

    // Open Settings modal
    try {
        await page.evaluate(() => { window.openSettingsModal?.(); });
        await page.waitForSelector('#settings-modal.open', { timeout: 5000 });
        await sleep(800);

        // Switch to Security tab
        const securityTab = await page.$('#settings-modal .settings-tab[data-tab="security"]');
        if (!securityTab) {
            console.log('[CAPTURE] Security tab not found; skipping TLS scenario');
            await page.keyboard.press('Escape');
            return;
        }

        await securityTab.click();
        await sleep(900);

        // Helper: log visibility of key Security elements
        const logCertsState = async () => {
            const state = await page.evaluate(() => {
                const pane = document.getElementById('settings-security');
                const tlsStatus = document.getElementById('tls-status-text');
                const pills = document.querySelectorAll('.cert-mode-pill');
                const acmeFqdn = document.getElementById('acme-fqdn');
                const acmeSection = document.getElementById('acme-credentials-section');
                const customCertPath = document.getElementById('tls-custom-cert-path');
                const customKeyPath = document.getElementById('tls-custom-key-path');
                const btnApplyCustom = document.getElementById('btn-apply-custom-cert');
                return {
                    paneExists: !!pane,
                    tlsStatusExists: !!tlsStatus,
                    pillsCount: pills.length,
                    acmeFqdnExists: !!acmeFqdn,
                    acmeSectionExists: !!acmeSection,
                    customCertPathExists: !!customCertPath,
                    customKeyPathExists: !!customKeyPath,
                    btnApplyCustomExists: !!btnApplyCustom,
                    paneScrollHeight: pane?.scrollHeight ?? null,
                };
            });
            console.log('[CAPTURE] TLS Certificates state:', JSON.stringify(state));
            return state;
        };

        // Helper: dismiss all toasts to keep screenshots clean
        const dismissToasts = async () => {
            await page.evaluate(() => {
                const toasts = document.querySelectorAll('[data-toast], .toast, .settings-toast, [role="status"]');
                toasts.forEach(t => {
                    const closeBtn = t.querySelector('[aria-label="Close"], button, .toast-close');
                    if (closeBtn) closeBtn.click();
                    else t.remove();
                });
            });
            await sleep(300);
        };

        // Helper: scroll a target element to the top of the pane so it appears as high as possible
        const scrollToTopOfPane = async (selector) => {
            await page.evaluate((sel) => {
                const pane = document.getElementById('settings-security');
                const target = document.querySelector(sel);
                if (!pane || !target) return;
                const paneRect = pane.getBoundingClientRect();
                const targetRect = target.getBoundingClientRect();
                const scrollDelta = targetRect.top - paneRect.top;
                pane.scrollTop += scrollDelta;
            }, selector);
            await sleep(500);
        };

        // Helper: select a certificate mode via pill
        const selectCertMode = async (mode) => {
            await page.evaluate((m) => {
                const pill = document.querySelector(`.cert-mode-pill[data-mode="${m}"]`);
                if (pill) pill.click();
            }, mode);
            await sleep(600);
        };

        await logCertsState();

        // 1) Security & Certificates tab overview (top area, default mode)
        await page.evaluate(() => {
            const pane = document.getElementById('settings-security');
            if (pane) pane.scrollTo({ top: 0, behavior: 'instant' });
        });
        await sleep(300);
        await dismissToasts();
        await captureShot(page, 'tls-certificates-tab.png', { fullPage: true });
        await captureCloseUp(page, '#settings-modal', 'tls-certificates-tab.png', options);

        // 2) No TLS mode: select "No HTTPS" pill, scroll to Certificates card, capture
        await selectCertMode('none');
        await scrollToTopOfPane('#cert-mode-none');
        await dismissToasts();
        await logCertsState();
        await captureShot(page, 'tls-mode-no-tls.png', { fullPage: true });

        // 3) Self-signed mode: select "Self-Signed" pill, scroll to Certificates card, capture
        await selectCertMode('self-signed');
        await scrollToTopOfPane('#cert-mode-self-signed');
        await dismissToasts();
        await logCertsState();
        await captureShot(page, 'tls-mode-self-signed.png', { fullPage: true });

        // 4) Custom certificate mode: select "Bring Your Own Key" pill, fill paths, apply, capture
        await selectCertMode('custom');
        const customCertPath = await page.$('#tls-custom-cert-path');
        const customKeyPath = await page.$('#tls-custom-key-path');
        const btnApplyCustom = await page.$('#btn-apply-custom-cert');
        if (customCertPath && customKeyPath && btnApplyCustom) {
            await customCertPath.type('/path/to/cert.pem', { delay: 10 });
            await customKeyPath.type('/path/to/key.pem', { delay: 10 });
            await sleep(400);
            await btnApplyCustom.click();
            await sleep(600);
            await scrollToTopOfPane('#tls-custom-cert-path');
            await dismissToasts();
            await logCertsState();
            await captureShot(page, 'tls-mode-custom.png', { fullPage: true });
        } else {
            console.log('[CAPTURE] Custom cert fields not fully present; skipping Custom cert shot');
        }

        // 5) ACME mode: select "Let's Encrypt (ACME)" pill, scroll so ACME is high, capture
        await selectCertMode('acme');

        // Wait for ACME FQDN field to be present and scroll it into view
        let acmeFound = false;
        for (let attempt = 0; attempt < 6; attempt++) {
            await sleep(500);
            acmeFound = await page.evaluate(() => {
                return !!document.getElementById('acme-fqdn');
            });
            if (acmeFound) break;
        }
        if (!acmeFound) {
            console.log('[CAPTURE] #acme-fqdn not found after retries, skipping ACME capture');
            await page.keyboard.press('Escape');
            return;
        }

        await page.evaluate(() => {
            const acmeFqdn = document.getElementById('acme-fqdn');
            if (acmeFqdn) {
                acmeFqdn.scrollIntoView({ behavior: 'instant', block: 'start' });
                const pane = document.getElementById('settings-security');
                if (pane) {
                    pane.scrollTop += 6;
                }
            }
        });
        await sleep(300);
        await dismissToasts();

        // Full view with ACME card high and title visible
        await captureShot(page, 'tls-mode-acme-full.png', { fullPage: true });

        // Show "Other" provider input, keep ACME high, capture
        await page.evaluate(() => {
            const select = document.getElementById('acme-dns-provider');
            const otherOption = Array.from(select?.options || [])
                .find(o => o.value === '__other__');
            if (otherOption) {
                select.value = '__other__';
                select.dispatchEvent(new Event('change'));
            }
        });
        await sleep(600);
        await page.evaluate(() => {
            const acmeFqdn = document.getElementById('acme-fqdn');
            if (acmeFqdn) {
                acmeFqdn.scrollIntoView({ behavior: 'instant', block: 'start' });
                const pane = document.getElementById('settings-security');
                if (pane) {
                    pane.scrollTop += 6;
                }
            }
        });
        await sleep(300);
        await dismissToasts();

        const customWrapVisible = await page.evaluate(() => {
            const el = document.getElementById('acme-provider-custom-wrap');
            if (!el) return false;
            const style = window.getComputedStyle(el);
            return style.display !== 'none';
        });
        if (customWrapVisible) {
            await captureShot(page, 'tls-acme-other-provider.png', { fullPage: true });
        }

        // 6) Database Administration section
        let dbFound = false;
        for (let attempt = 0; attempt < 6; attempt++) {
            await sleep(500);
            dbFound = await page.evaluate(() => {
                return !!document.getElementById('db-admin-panel');
            });
            if (dbFound) break;
        }
        if (!dbFound) {
            console.log('[CAPTURE] #db-admin-panel not found after retries, skipping DB admin capture');
            await page.keyboard.press('Escape');
            return;
        }

        await page.evaluate(() => {
            const dbPanel = document.getElementById('db-admin-panel');
            if (dbPanel) {
                dbPanel.scrollIntoView({ behavior: 'instant', block: 'start' });
                const pane = document.getElementById('settings-security');
                if (pane) {
                    pane.scrollTop += 6;
                }
            }
        });
        await sleep(300);
        await dismissToasts();

        await captureShot(page, 'tls-db-admin-section.png', { fullPage: true });

        await page.keyboard.press('Escape');
        await sleep(300);
    } catch (e) {
        console.log('[CAPTURE] TLS/Certificates scenario failed:', e.message);
    }
}
