// ── Sensor Bridge ─────────────────────────────────────────────────────────────
// Sensor bridge setup button handler.

import { escapeHtml } from '../core/format.js';

function _bindSensorBridgeSetup() {
    const btn = document.getElementById('btn-sensor-bridge-setup');
    if (!btn) return;
    btn.addEventListener('click', async () => {
        btn.disabled = true;
        btn.textContent = 'Installing...';
        const callout = document.getElementById('sensor-bridge-setup-callout');
        try {
            const res = await fetch('/api/sensor-bridge/install', {
                    method: 'POST',
                    headers: window.authHeaders ? window.authHeaders() : {},
                });
            const data = await res.json();
            if (!data.started) {
                btn.textContent = 'Setup';
                btn.disabled = false;
                if (callout) {
                    callout.innerHTML = '<span style="color:#bf616a;">Install failed: ' + escapeHtml(data.error || 'Unknown error') + '</span>';
                }
                return;
            }
            if (callout) {
                callout.innerHTML = '<span style="color:#a3be8c;">A UAC prompt will appear on your desktop \u2014 approve it to install the sensor service. This takes a few seconds.</span>';
            }
            let elapsed = 0;
            const poll = setInterval(async () => {
                elapsed += 2000;
                try {
                    const s = await fetch('/api/sensor-bridge/status', {
                            headers: window.authHeaders ? window.authHeaders() : {},
                        });
                    const sd = await s.json();
                    if (sd.running) {
                        clearInterval(poll);
                        // The bridge can run yet report no temperature when the PawnIO
                        // kernel driver is absent (normally installed during the same
                        // elevated setup step; can fail if winget is offline).
                        if (sd.pawnio === false) {
                            if (callout) {
                                callout.style.display = '';
                                callout.textContent =
                                    'Sensor service installed, but the PawnIO driver is missing — CPU temperature stays unavailable until it is present. It is normally installed automatically; if that failed (e.g. winget offline), install it from ';
                                const link = document.createElement('a');
                                link.href = 'https://pawnio.eu';
                                link.target = '_blank';
                                link.rel = 'noopener';
                                link.style.color = '#88c0d0';
                                link.textContent = 'pawnio.eu';
                                callout.appendChild(link);
                                callout.appendChild(document.createTextNode(' and click Setup again.'));
                            }
                        } else if (callout) {
                            callout.style.display = 'none';
                        }
                    } else if (elapsed >= 30000) {
                        clearInterval(poll);
                        btn.textContent = 'Setup';
                        btn.disabled = false;
                        if (callout) {
                            callout.innerHTML = 'CPU temperature requires a one-time service install. <button id="btn-sensor-bridge-setup" style="margin-left:8px; padding:3px 10px; background:#5e81ac; border:none; border-radius:4px; color:#eceff4; cursor:pointer; font-size:12px;">Setup</button><span style="color:#ebcb8b; margin-left:8px;">Timed out \u2014 did you approve the UAC prompt?</span>';
                            const newBtn = document.getElementById('btn-sensor-bridge-setup');
                            if (newBtn) newBtn.addEventListener('click', () => btn.click());
                        }
                    }
                } catch (_) {}
            }, 2000);
        } catch (_e) {
            btn.textContent = 'Setup';
            btn.disabled = false;
        }
    });
}

// ── Public API ────────────────────────────────────────────────────────────────

export function initSensorBridge() {
    _bindSensorBridgeSetup();
}
