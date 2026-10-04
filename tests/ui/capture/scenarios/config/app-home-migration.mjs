import assert from 'node:assert/strict';
import { gotoApp } from '../../harness/browser.mjs';
import { captureShot } from '../../harness/shot.mjs';
import { createMigrationMock } from '../../../fixtures/app-home-migration.mjs';

// No attach or filesystem migration. Only the migration API namespace is
// intercepted; settings, auth/admin-token, and all other responses stay real.
export default async function scenarioAppHomeMigration(ctx) {
    const page = await ctx.browser.newPage();
    const state = createMigrationMock();
    const originalViewport = ctx.page.viewport();
    const dialogHandler = async dialog => {
        if (dialog.type() === 'confirm' && dialog.message().startsWith('Queue the Foundry migration')) {
            await dialog.accept();
        } else {
            await dialog.dismiss();
        }
    };
    const requestHandler = async request => {
        const response = state.respond(new URL(request.url()).pathname, request.method(), request.postData());
        if (response) {
            await request.respond({
                status: response.status, contentType: 'application/json',
                body: JSON.stringify(response.body),
            });
        } else {
            await request.continue();
        }
    };
    const shot = filename => captureShot(page, filename, {
        fullPage: false, expandSelector: '#settings-migration',
    });
    try {
        await page.setViewport(originalViewport);
        await page.setCacheEnabled(false);
        await page.setRequestInterception(true);
        page.on('request', requestHandler);
        page.on('dialog', dialogHandler);
        await page.evaluateOnNewDocument(() => {
            sessionStorage.setItem('local-llm-foundry-migration-toast-seen', '1');
        });
        await gotoApp(page, ctx.baseUrl);
        // Preserve harness font diagnostics from the dedicated scenario page.
        ctx.page.__fontDiagnostics = page.__fontDiagnostics;
        await page.waitForSelector('html.modules-ready');
        await page.evaluate(async () => {
            const Router = (await import('/js/features/router.js')).default;
            Router.navigate('/settings#migration');
        });
        await page.waitForSelector('#settings-modal.open', { visible: true });
        await page.waitForFunction(() => {
            const button = document.getElementById('app-home-migration-preview');
            return button && !button.disabled;
        });
        await page.evaluate(() => { document.documentElement.dataset.theme = 'dark'; });
        await page.click('#app-home-migration-preview');
        await page.waitForFunction(() => !document.getElementById('app-home-migration-queue').disabled);
        assert.equal(await page.$eval('#app-home-migration-copy-size', el => el.textContent), '771 MiB');
        assert.equal(await page.$eval('#app-home-migration-inventoried-count', el => el.textContent), '8');
        assert.equal(await page.$eval('#app-home-migration-retained-count', el => el.textContent), '4');
        assert.equal(await page.$$eval('#app-home-migration-card details[open]', els => els.length), 0);
        await shot('app-home-migration-preview-dark.png');

        await page.click('#app-home-migration-locations > summary');
        await page.click('#app-home-migration-technical > summary');
        await page.waitForSelector('#app-home-migration-locations[open]');
        await page.waitForSelector('#app-home-migration-technical[open]');
        await shot('app-home-migration-expanded-dark.png');

        await page.click('#app-home-migration-locations > summary');
        await page.click('#app-home-migration-technical > summary');
        await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
        await shot('app-home-migration-preview-light.png');

        await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }]);
        await page.setViewport({ width: 430, height: 900, deviceScaleFactor: 1 });
        await shot('app-home-migration-narrow-reduced-motion.png');

        await page.setViewport(originalViewport);
        await page.emulateMediaFeatures([]);
        await page.evaluate(() => { document.documentElement.dataset.theme = 'dark'; });
        await page.click('#app-home-migration-queue');
        await page.waitForFunction(() => {
            const text = document.getElementById('app-home-migration-state')?.textContent || '';
            return /Migration is queued/.test(text) && /Restart Foundry/.test(text);
        });
        assert.equal(await page.$eval('#app-home-migration-preview', el => el.disabled), true);
        assert.equal(await page.$eval('#app-home-migration-queue', el => el.disabled), true);
        assert.equal(state.queued, true);
        assert.equal(state.calls.filter(call => call.pathname === '/api/app-home-migration/queue').length, 1);
        assert.deepEqual(state.unexpected, []);
        await shot('app-home-migration-queued-success.png');
    } finally {
        // Keep interception active until this dedicated page is closed so a
        // delayed migration request cannot escape to the live server in cleanup.
        try {
            await page.close();
        } finally {
            page.off('request', requestHandler);
            page.off('dialog', dialogHandler);
        }
    }
}
