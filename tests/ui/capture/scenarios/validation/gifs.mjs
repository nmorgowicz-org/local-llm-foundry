// Scenario: gifs
// Extracted from tests/ui/capture.mjs (Phase A3).
import { join } from 'path';
import { attachToServer } from '../../harness/attach.mjs';
import { gotoApp, switchTab } from '../../harness/browser.mjs';
import { currentArtifactsDir, tagFilename, sleep } from '../../harness/paths.mjs';
import { captureFrames, cleanupFrames, framesToGif, startLiveGeneration, waitForRemoteIdle } from '../../harness/shot.mjs';

export default async function(ctx, options) {
    const { page, baseUrl } = ctx;
    const fps = 10;
    const durationSec = 6;
    const totalFrames = fps * durationSec;
    const inferenceDurationSec = 10;
    const inferenceTotalFrames = fps * inferenceDurationSec;

    await gotoApp(page, baseUrl);
    await attachToServer(page);

    if (!options.gpuOnly) {
        console.log('[CAPTURE] Capturing inference metrics GIF...');
        await switchTab(page, 'server');
        // Scroll a bit so more of the "Performance & metrics" section (generation
        // details, cache, etc. below the fold) is visible in the animated capture.
        await page.evaluate(() => {
            const section = document.getElementById('inference-section');
            const pg = section?.closest('.page') || document.querySelector('.page.active');
            if (pg && section) pg.scrollTop = section.offsetTop - 8;
        });
        await sleep(300);
        if (process.env.CAPTURE_WAIT_FOR_IDLE === '1') await waitForRemoteIdle();
        const generationPromise = startLiveGeneration();
        await sleep(1500);
        await captureFrames(page, 'inference', inferenceTotalFrames, fps);
        await generationPromise;
        framesToGif('inference', join(currentArtifactsDir(), tagFilename('performance-metrics.gif')), fps);
        cleanupFrames();
    }

    if (!options.inferenceOnly) {
        console.log('[CAPTURE] Capturing GPU/system metrics GIF...');
        await switchTab(page, 'server');
        // Wait for agent hardware data if we haven't already (gpuOnly path skips inference wait).
        if (options.gpuOnly) await sleep(3500);
        // GPU/system sections stay display:none (offsetTop 0) until hardware data
        // actually arrives — wait for visibility before scrolling to them, otherwise
        // the scroll silently no-ops and the GIF captures the wrong section.
        const hwVisible = await page.evaluate(() => new Promise(resolve => {
            const check = () => {
                const gpu = document.getElementById('gpu-section');
                const sys = document.getElementById('system-section');
                if ((gpu && gpu.style.display !== 'none') || (sys && sys.style.display !== 'none')) {
                    resolve(true);
                }
            };
            check();
            const obs = new MutationObserver(check);
            obs.observe(document.body, { attributes: true, subtree: true, attributeFilter: ['style'] });
            setTimeout(() => { obs.disconnect(); resolve(false); }, 6000);
        }));
        if (!hwVisible) {
            console.log('[CAPTURE] Hardware section not visible; capturing at current scroll position.');
        }
        await page.evaluate(() => {
            const target = document.getElementById('gpu-section') || document.getElementById('system-section');
            if (target && target.style.display !== 'none') {
                const pg = target.closest('.page') || document.querySelector('.page.active');
                if (pg) pg.scrollTop = target.offsetTop - 8;
            }
        });
        await sleep(1200);
        await captureFrames(page, 'gpu', totalFrames, fps);
        framesToGif('gpu', join(currentArtifactsDir(), tagFilename('gpu-metrics.gif')), fps);
        cleanupFrames();
    }
}

// Sidebar features capture: expanded panel, collapsed strip, FTS search, context menu
// ── Sidebar ─────────────────────────────────────────────────────────────────────
// Chat sidebar, FTS search flyout, context menu, title filter.
