// Browser/page lifecycle: launch, navigate, wait-for-app, tab switching.
// Extracted from tests/ui/capture.mjs (Phase A1).
import puppeteer from 'puppeteer';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DEFAULT_VIEWPORT, sleep } from './paths.mjs';

export async function launchBrowser(viewport = DEFAULT_VIEWPORT) {
    // Puppeteer's implicit Windows temp profile can retain a Chromium
    // `lockfile` after a failed launch, causing every subsequent capture to
    // report "browser is already running". An explicit unique profile avoids
    // that process-singleton race on Windows and remains isolated per launch.
    const userDataDir = mkdtempSync(join(tmpdir(), 'local-llm-foundry-capture-'));
    const browser = await puppeteer.launch({
        headless: 'new',
        userDataDir,
        args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage', '--disable-cache', '--disable-service-workers'],
    });
    const closeBrowser = browser.close.bind(browser);
    browser.close = async (...args) => {
        try {
            await closeBrowser(...args);
        } finally {
            try {
                rmSync(userDataDir, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 });
            } catch {
                // Chromium may release profile files asynchronously on Windows.
            }
        }
    };
    const page = await browser.newPage();
    await page.setCacheEnabled(false);
    // Migration toasts pop once per browser session and cover the bottom-right
    // of every capture. The notification entries stay; only the pop-ups are
    // marked as already shown, matching the dedicated migration scenario.
    await page.evaluateOnNewDocument(() => {
        try {
            sessionStorage.setItem('local-llm-foundry-migration-toast-seen', '1');
            sessionStorage.setItem('local-llm-foundry-model-move-toast-seen', '1');
        } catch {
            // Storage may be unavailable on non-app documents (about:blank).
        }
    });
    await page.setViewport(viewport);
    return { browser, page };
}

export async function waitForMonitor(page) {
    await page.waitForFunction(() => {
        const setup = document.getElementById('view-setup');
        const monitor = document.getElementById('view-monitor');
        if (!setup || !monitor) return false;
        return getComputedStyle(monitor).display !== 'none' && getComputedStyle(setup).display === 'none';
    }, { timeout: 30000 });
}

export async function switchTab(page, tabName) {
    await page.evaluate((name) => {
        document.querySelector(`button[data-tab="${name}"]`)?.click();
    }, tabName);
    await page.waitForFunction((name) => {
        const tab = document.querySelector(`button[data-tab="${name}"]`);
        const pageEl = document.getElementById(`page-${name}`);
        if (!tab || !pageEl) return false;
        const activeTab = tab.classList.contains('active') || tab.classList.contains('selected');
        const activePage = pageEl.classList.contains('active');
        return activeTab && activePage;
    }, { timeout: 10000 }, tabName);
}

export async function gotoApp(page, baseUrl, waitUntil = 'networkidle0') {
    await page.goto(baseUrl, { waitUntil });
    page.__fontDiagnostics = await assertDeterministicFonts(page);
    // See the scoped CSS rule in spawn-wizard.css. This marker prevents
    // Chromium's capture-only :focus-visible heuristic from outlining the
    // entire programmatically focused wizard step.
    await page.evaluate(() => { document.documentElement.dataset.screenshotCapture = 'true'; });
    await sleep(1500);
}

export async function assertDeterministicFonts(page) {
    const diagnostics = await page.evaluate(async () => {
        const faces = [
            ['Inter', 400],
            ['Inter', 500],
            ['Inter', 600],
            ['Inter', 700],
            ['Inter', 800],
            ['Fira Code', 400],
            ['Fira Code', 500],
            ['Fira Code', 600],
        ];
        await Promise.all(faces.map(([family, weight]) => document.fonts.load(`${weight} 16px "${family}"`)));
        await document.fonts.ready;
        const missing = faces.filter(([family, weight]) => !document.fonts.check(`${weight} 16px "${family}"`));
        const externalFontRequests = performance.getEntriesByType('resource')
            .map(entry => entry.name)
            .filter(url => /fonts\.(googleapis|gstatic)\.com/i.test(url));
        const root = getComputedStyle(document.documentElement);
        const body = getComputedStyle(document.body);
        return {
            status: document.fonts.status,
            missing,
            externalFontRequests,
            rootFontSize: root.fontSize,
            bodyFontFamily: body.fontFamily,
        };
    });
    if (diagnostics.status !== 'loaded' || diagnostics.missing.length || diagnostics.externalFontRequests.length) {
        throw new Error(`Deterministic font contract failed: ${JSON.stringify(diagnostics)}`);
    }
    return diagnostics;
}

export async function loadAppDocument(page, baseUrl) {
    try {
        await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 10000 });
    } catch (error) {
        console.log('[CAPTURE] app navigation fallback:', error.message);
    }

    const hasAppShell = await page.$('#page-server') !== null;
    if (hasAppShell) {
        page.__fontDiagnostics = await assertDeterministicFonts(page);
        await page.evaluate(() => { document.documentElement.dataset.screenshotCapture = 'true'; });
        await sleep(1500);
        return;
    }

    // Inject base tag via page.evaluate so relative URLs resolve
    // (avoids regex-based script stripping that CodeQL flags).
    await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 10000 });
    await page.evaluate((base) => {
        const baseTag = document.createElement('base');
        baseTag.href = base;
        document.head.insertBefore(baseTag, document.head.firstChild);
    }, baseUrl);
    await page.waitForSelector('#page-server', { timeout: 10000 });
    await page.evaluate(() => { document.documentElement.dataset.screenshotCapture = 'true'; });
    await page.evaluate(() => {
        document.getElementById('auth-shell')?.classList.add('hidden');
        document.body.classList.remove('auth-required');
    });
    page.__fontDiagnostics = await assertDeterministicFonts(page);
    await sleep(1500);
}

export async function openAppearancePaneForCapture(page) {
    await page.evaluate(() => {
        if (typeof window.openSettingsModal === 'function') {
            window.openSettingsModal();
            return;
        }
        const modal = document.getElementById('settings-modal');
        if (modal) {
            modal.classList.add('open');
            modal.removeAttribute('inert');
            modal.setAttribute('aria-hidden', 'false');
        }
    });
    await page.waitForSelector('#settings-modal.open', { timeout: 5000 });
    await page.evaluate(() => {
        document.querySelectorAll('.settings-tab').forEach(tab => {
            const active = tab.dataset.tab === 'appearance';
            tab.classList.toggle('active', active);
            tab.setAttribute('aria-selected', String(active));
        });
        document.querySelectorAll('.settings-pane').forEach(pane => {
            pane.classList.toggle('active', pane.id === 'settings-appearance');
        });
    });
}
