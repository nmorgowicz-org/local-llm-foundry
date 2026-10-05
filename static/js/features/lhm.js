// ── LHM (LibreHardwareMonitor) ────────────────────────────────────────────────
// LHM notification, install/start/uninstall flow, UAC warning, and status check.

import { lhm } from '../core/app-state.js';
import { showToast, showConfirmDialog } from './toast.js';

// ── LHM Notification ──────────────────────────────────────────────────────────

export async function showLHMNotification() {
    return new Promise(async (resolve) => {
        lhm.resolve = resolve;
        const overlay = document.createElement('div');
        overlay.className = 'notification-container';
        overlay.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            width: 400px;
            background: #2e3440;
            border: 2px solid #ebcb8b;
            border-radius: 8px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            z-index: 9999;
            padding: 20px;
            color: #d8dee9;
            animation: fadeIn 0.3s ease-out;
        `;

        overlay.innerHTML = `
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <h3 style="margin:0 0 10px 0;color:#ebcb8b;">LibreHardwareMonitor Status</h3>
                <button id="lhm-cancel-btn" style="background:none;border:none;color:#d8dee9;cursor:pointer;font-size:20px;">&times;</button>
            </div>
            <p id="lhm-status-text" style="margin:0 0 15px 0;line-height:1.5;">Checking status...</p>
            <div id="lhm-buttons" style="display:flex;gap:10px;flex-direction:column;"></div>
        `;

        document.body.appendChild(overlay);

        // Bind cancel button
        document.getElementById('lhm-cancel-btn')?.addEventListener('click', () => {
            overlay.remove();
            lhm.resolve('cancel');
        });

        const lhmStatusEl = document.getElementById('lhm-status-text');
        const lhmButtonsEl = document.getElementById('lhm-buttons');

        try {
            const [statusResp, checkResp] = await Promise.all([
                fetch('/api/lhm/status').catch(() => null),
                fetch('/api/lhm/check').catch(() => null)
            ]);

            let isDisabled = false;
            let lhmAvailable = false;
            let lhmInstalled = false;

            if (statusResp && statusResp.ok) {
                const statusData = await statusResp.json();
                isDisabled = statusData.disabled || false;
            }

            if (checkResp && checkResp.ok) {
                const checkData = await checkResp.json();
                lhmAvailable = checkData.running || false;
                lhmInstalled = checkData.installed || false;
            }

            if (isDisabled) {
                lhmStatusEl.textContent = 'LibreHardwareMonitor is disabled. Enable it to monitor CPU temperatures.';
                lhmButtonsEl.innerHTML = `
                    <button id="btn-lhm-enable" style="flex:1;padding:10px;background:#a3be8c;border:none;border-radius:4px;cursor:pointer;font-weight:bold;">Enable Monitoring</button>
                `;
                lhmButtonsEl.querySelector('#btn-lhm-enable').onclick = async () => {
                    overlay.remove();
                    try {
                        const disableResp = await fetch('/api/lhm/disable', {
                            method: 'POST',
                            headers: window.authHeaders
                                ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
                                : { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ disabled: false })
                        });
                        if (disableResp.ok) {
                            showToast('LHM monitoring enabled', 'success');
                            setTimeout(() => location.reload(), 1500);
                        }
                    } catch (err) {
                        showToast('Failed to enable LHM: ' + err.message, 'error');
                    }
                };
            } else if (lhmAvailable) {
                lhmStatusEl.textContent = 'LibreHardwareMonitor is running. CPU temperature monitoring is active.';
                lhmButtonsEl.innerHTML = `
                    <button id="btn-lhm-uninstall" style="flex:1;padding:10px;background:#bf616a;border:none;border-radius:4px;cursor:pointer;font-weight:bold;">Uninstall LHM</button>
                `;
                lhmButtonsEl.querySelector('#btn-lhm-uninstall').onclick = async () => {
                    overlay.remove();
                    const uninstallConfirm = await showConfirmDialog(
                        'Uninstall LibreHardwareMonitor',
                        'Are you sure you want to uninstall LibreHardwareMonitor? This will disable CPU temperature monitoring.',
                        'Uninstall'
                    );
                    if (uninstallConfirm) {
                        try {
                            const uninstallResp = await fetch('/api/lhm/uninstall', {
                                method: 'POST',
                                headers: window.authHeaders ? window.authHeaders() : {},
                            });
                            if (uninstallResp.ok) {
                                showToast('LHM uninstalled successfully', 'success');
                                setTimeout(() => location.reload(), 1500);
                            }
                        } catch (err) {
                            showToast('Failed to uninstall LHM: ' + err.message, 'error');
                        }
                    }
                };
            } else if (lhmInstalled) {
                lhmStatusEl.textContent = 'LibreHardwareMonitor is installed but not running. Start it to enable CPU temperature monitoring.';
                lhmButtonsEl.innerHTML = `
                    <button id="btn-lhm-start" style="flex:1;padding:10px;background:#a3be8c;border:none;border-radius:4px;cursor:pointer;font-weight:bold;">Start LHM</button>
                    <button id="btn-lhm-uninstall" style="flex:1;padding:10px;background:#bf616a;border:none;border-radius:4px;cursor:pointer;">Uninstall LHM</button>
                `;

                lhmButtonsEl.querySelector('#btn-lhm-start').onclick = async () => {
                    overlay.remove();
                    const warningOverlay = createUACWarningOverlay();
                    const userConfirmed = await showWarningModal(warningOverlay);

                    if (!userConfirmed) return;

                    try {
                        const startResp = await fetch('/api/lhm/start', {
                                method: 'POST',
                                headers: window.authHeaders ? window.authHeaders() : {},
                            });
                        if (startResp.ok) {
                            showToast('LHM started successfully', 'success');
                            setTimeout(() => location.reload(), 2000);
                        } else {
                            const data = await startResp.json();
                            showToast('Failed to start LHM: ' + (data.error || 'Unknown error'), 'error');
                        }
                    } catch (err) {
                        showToast('Failed to start LHM: ' + err.message, 'error');
                    }
                };

                lhmButtonsEl.querySelector('#btn-lhm-uninstall').onclick = async () => {
                    overlay.remove();
                    const uninstallConfirm = await showConfirmDialog(
                        'Uninstall LibreHardwareMonitor',
                        'Are you sure you want to uninstall LibreHardwareMonitor?',
                        'Uninstall'
                    );
                    if (uninstallConfirm) {
                        try {
                            const uninstallResp = await fetch('/api/lhm/uninstall', {
                                method: 'POST',
                                headers: window.authHeaders ? window.authHeaders() : {},
                            });
                            if (uninstallResp.ok) {
                                showToast('LHM uninstalled successfully', 'success');
                                setTimeout(() => location.reload(), 1500);
                            }
                        } catch (err) {
                            showToast('Failed to uninstall LHM: ' + err.message, 'error');
                        }
                    }
                };
            } else {
                lhmStatusEl.textContent = 'CPU temperature monitoring requires LibreHardwareMonitor. Please install it to see CPU temperatures.';
                lhmButtonsEl.innerHTML = `
                    <button id="btn-lhm-install" style="flex:1;padding:10px;background:#a3be8c;border:none;border-radius:4px;cursor:pointer;font-weight:bold;">Install Automatically</button>
                    <button id="btn-lhm-cancel" style="flex:1;padding:10px;background:#bf616a;border:none;border-radius:4px;cursor:pointer;">Disable</button>
                `;

                lhmButtonsEl.querySelector('#btn-lhm-cancel').onclick = async () => {
                    overlay.remove();
                    try {
                        const disableResp = await fetch('/api/lhm/disable', {
                            method: 'POST',
                            headers: window.authHeaders
                                ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
                                : { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ disabled: true })
                        });
                        if (disableResp.ok) {
                            showToast('LHM monitoring disabled', 'success');
                            setTimeout(() => location.reload(), 1500);
                        }
                    } catch (err) {
                        showToast('Failed to disable LHM: ' + err.message, 'error');
                    }
                };

                lhmButtonsEl.querySelector('#btn-lhm-install').onclick = async () => {
                    console.log('[LHM UI] Install button clicked');
                    overlay.remove();

                    const warningOverlay = createUACWarningOverlay();
                    const userConfirmed = await showWarningModal(warningOverlay);

                    if (!userConfirmed) {
                        resolve('cancel');
                        return;
                    }

                    console.log('[LHM UI] User confirmed, starting installation...');

                    const progressOverlay = document.createElement('div');
                    progressOverlay.style.cssText = `
                        position: fixed;
                        top: 50%;
                        left: 50%;
                        transform: translate(-50%, -50%);
                        width: 400px;
                        background: #2e3440;
                        border: 2px solid #88c0d0;
                        border-radius: 12px;
                        box-shadow: 0 20px 60px rgba(0,0,0,0.7);
                        z-index: 99999;
                        padding: 30px;
                        color: #d8dee9;
                        text-align: center;
                    `;

                    progressOverlay.innerHTML = `
                        <div style="margin-bottom: 20px;">
                            <h3 style="margin: 0 0 10px 0; color: #88c0d0; font-size: 18px;">Installing LibreHardwareMonitor</h3>
                            <p style="margin: 0; color: #bf616a;">This will open a UAC prompt.</p>
                        </div>
                        <div id="progress-bar-container" style="width: 100%; height: 8px; background: #4c566a; border-radius: 4px; overflow: hidden; margin-bottom: 15px;">
                            <div id="progress-bar" style="width: 0%; height: 100%; background: #88c0d0; transition: width 0.3s ease;"></div>
                        </div>
                        <div id="progress-text" style="color: var(--color-error); font-size: 14px;">Waiting for UAC...</div>
                         <div style="margin-top: 15px; font-size: 12px; color: var(--color-text-muted);">
                             <span class="spinner" style="display: inline-block; width: 12px; height: 12px; border: 2px solid var(--color-text-muted); border-top: 2px solid var(--color-info); border-radius: 50%; animation: spin 1s linear infinite; margin-right: 8px;"></span>
                             Please wait...
                         </div>
                        <style>
                            @keyframes spin { to { transform: rotate(360deg); } }
                        </style>
                    `;

                    document.body.appendChild(progressOverlay);

                    console.log('[LHM UI] Calling /api/lhm/install...');
                    try {
                        const response = await fetch('/api/lhm/install', {
                            method: 'POST',
                            headers: window.authHeaders
                                ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
                                : { 'Content-Type': 'application/json' }
                        });
                        console.log('[LHM UI] /api/lhm/install response status:', response.status);

                        if (response.ok) {
                            const data = await response.json();
                            console.log('[LHM UI] /api/lhm/install response:', data);

                            const progressText = document.getElementById('progress-text');
                            const progressBar = document.getElementById('progress-bar');

                            let attempts = 0;
                            const maxAttempts = 60;

                            const checkProgress = async () => {
                                if (attempts >= maxAttempts) {
                                    if (progressOverlay) progressOverlay.remove();
                                    showToast('Installation timeout. Please check if LHM was installed.', 'error');
                                    return;
                                }

                                attempts++;
                                console.log(`[LHM UI] Checking progress (attempt ${attempts})...`);

                                try {
                                    const progressResp = await fetch('/api/lhm/progress');
                                    if (progressResp.ok) {
                                        const progressData = await progressResp.json();
                                        const progress = progressData.progress || '';
                                        console.log('[LHM UI] Progress:', progress);

                                        if (progressText) {
                                            let progressDisplay = progress;
                                            let progressBarWidth = '0%';

                                            if (progress.includes('downloading:')) {
                                                progressDisplay = 'Downloading...';
                                                const pct = progress.match(/(\d+)%/);
                                                if (pct) progressBarWidth = pct[1] + '%';
                                            } else if (progress.includes('extracting:')) {
                                                progressDisplay = progress;
                                                const pct = progress.match(/(\d+)%/);
                                                if (pct) progressBarWidth = pct[1] + '%';
                                           } else if (progress === 'completed') {
                                                 progressDisplay = 'Installation complete! LHM is now running.';
                                                 if (progressBar) {
                                                     progressBar.style.background = 'var(--color-success)';
                                                 }
                                             } else if (progress === 'failed') {
                                                 progressDisplay = 'Installation failed!';
                                                 if (progressBar) {
                                                     progressBar.style.background = 'var(--color-error)';
                                                 }
                                             }

                                            progressText.textContent = progressDisplay;
                                            if (progressBar && progressBarWidth !== '0%') {
                                                progressBar.style.width = progressBarWidth;
                                            }
                                        }

                                        if (progress === 'completed' || progress === 'failed') {
                                            setTimeout(() => {
                                                if (progressOverlay) progressOverlay.remove();
                                                showToast('Installation ' + (progress === 'completed' ? 'complete! Reloading...' : 'failed'), progress === 'completed' ? 'success' : 'error');
                                                if (progress === 'completed') {
                                                    setTimeout(() => window.location.reload(), 2000);
                                                }
                                            }, 1500);
                                        } else {
                                            setTimeout(checkProgress, 500);
                                        }
                                    }
                                } catch (err) {
                                    console.error('[LHM UI] Progress check error:', err);
                                    setTimeout(checkProgress, 500);
                                }
                            };

                            setTimeout(checkProgress, 1000);
                        } else {
                            const data = await response.json();
                            console.error('[LHM UI] /api/lhm/install failed:', data);
                            if (progressOverlay) progressOverlay.remove();
                            showToast(`Installation failed: ${data.error || 'Unknown error'}`, 'error');
                        }
                    } catch (err) {
                        console.error('[LHM UI] /api/lhm/install error:', err);
                        if (progressOverlay) progressOverlay.remove();
                        showToast(`Installation error: ${err.message}`, 'error');
                    }
                };
            }
        } catch (err) {
            console.error('[LHM UI] Error checking LHM status:', err);
            lhmStatusEl.textContent = 'Error checking LHM status. Please try again.';
        }
    });
}

// ── UAC Warning ───────────────────────────────────────────────────────────────

function createUACWarningOverlay() {
    const overlay = document.createElement('div');
    overlay.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 500px;
        background: #2e3440;
        border: 2px solid #ebcb8b;
        border-radius: 12px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.8);
        z-index: 99999;
        padding: 30px;
        color: #d8dee9;
    `;

    overlay.innerHTML = `
        <div style="display:flex;justify-content:center;align-items:center;margin-bottom:20px;">
            <div style="width:48px;height:48px;background:#3b4252;border-radius:50%;display:flex;align-items:center;justify-content:center;margin-right:20px;">
                <span style="font-size:24px;">⚠️</span>
            </div>
            <div>
                <h2 style="margin:0 0 5px 0;color:#ebcb8b;">Administrator Access Required</h2>
                <p style="margin:0;font-size:0.85rem;color:#a3be8c;">This will open a Windows security prompt</p>
            </div>
        </div>

        <div style="background:#3b4252;border-radius:8px;padding:15px;margin-bottom:20px;line-height:1.6;font-size:0.9rem;">
            <p style="margin:0 0 10px 0;"><strong>What will happen:</strong></p>
            <ul style="margin:0 0 10px 0;padding-left:20px;">
                <li>Windows will show a UAC prompt asking for admin permission</li>
                <li>LibreHardwareMonitor will be downloaded (~5MB)</li>
                <li>It will be installed silently to your AppData folder</li>
                <li>After installation, the window will minimize automatically</li>
            </ul>
        </div>

        <div style="display:flex;gap:10px;">
            <button id="btn-warning-yes" style="flex:1;padding:12px;background:#a3be8c;border:none;border-radius:6px;cursor:pointer;font-weight:bold;font-size:1rem;">Yes, Continue</button>
            <button id="btn-warning-no" style="flex:1;padding:12px;background:#bf616a;border:none;border-radius:6px;cursor:pointer;font-size:1rem;">Cancel</button>
        </div>

        <p style="margin-top:20px;font-size:0.75rem;color:#616e88;text-align:center;">
            LibreHardwareMonitor needs admin access to read hardware sensors.
        </p>
    `;

    return overlay;
}

function showWarningModal(overlay) {
    return new Promise((resolve) => {
        document.body.appendChild(overlay);

        overlay.querySelector('#btn-warning-yes').onclick = () => {
            overlay.remove();
            resolve(true);
        };

        overlay.querySelector('#btn-warning-no').onclick = () => {
            overlay.remove();
            resolve(false);
        };
    });
}

// ── Public API ────────────────────────────────────────────────────────────────

export function initLHM() {
    // Event delegation for dynamically generated LHM buttons
    document.addEventListener('click', (e) => {
        const btn = e.target.closest('[data-lhm-action]');
        if (btn && btn.dataset.lhmAction === 'show') {
            showLHMNotification();
        }
    });
}
