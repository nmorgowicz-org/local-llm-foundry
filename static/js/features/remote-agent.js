// ── Remote Agent ───────────────────────────────────────────────────────────────
// Remote agent setup modal, SSH guide, install/start/stop/update/remove,
// host key scanning, and status indicator.

import { remoteAgent } from '../core/app-state.js';
import { wsData } from '../core/app-state.js';
import { escapeHtml } from '../core/format.js';
import { showToast, showConfirmDialog } from './toast.js';
import { closeConfigModal, openConfigModal } from './config.js';
import { closeSettingsModal, saveSettings } from './settings.js';

let initialized = false;

// ── State ──────────────────────────────────────────────────────────────────────

const remoteAgentSetupState = {
    sshHost: '',
    sshPort: '22',
    sshAuth: 'agent',
    sshPassword: '',
    sshKeyPath: '',
    latestVersion: null,
    installedVersion: null,
    hostKey: null,
};

// ── Helpers ────────────────────────────────────────────────────────────────────

// Cached db-admin-token for install/remove endpoints.
let cachedDbAdminToken = null;

async function getDbAdminToken() {
    if (cachedDbAdminToken) return cachedDbAdminToken;
    try {
        const headers = window.authHeaders ? window.authHeaders() : {};
        const res = await fetch('/api/db/admin-token', { headers });
        if (res.ok) {
            const data = await res.json();
            if (data.token) {
                cachedDbAdminToken = data.token;
            }
        }
    } catch {
        // Non-critical
    }
    return cachedDbAdminToken;
}

// Returns headers with api-token (Bearer) for all remote-agent endpoints.
function remoteAgentHeaders(contentType = 'application/json') {
    const base = window.authHeaders
        ? window.authHeaders({ 'Content-Type': contentType })
        : { 'Content-Type': contentType };
    return base;
}

// Returns headers with db-admin-token for install/remove endpoints.
async function remoteAgentAdminHeaders(contentType = 'application/json') {
    const token = await getDbAdminToken();
    const headers = { 'Content-Type': contentType };
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }
    return headers;
}

// Wrap a fetch call and show a 401-friendly message if auth fails.
async function remoteAgentFetch(url, options) {
    try {
        const resp = await fetch(url, options);
        if (resp.status === 401) {
            showToast('Authentication required for this operation');
            return new Response(
                JSON.stringify({ ok: false, error: 'unauthorized' }),
                {
                    status: 401,
                    headers: { 'Content-Type': 'application/json' }
                }
            );
        }
        return resp;
    } catch (err) {
        throw err;
    }
}

function inferredAgentUrl() {
    const explicit = document.getElementById('set-remote-agent-url')?.value.trim();
    if (explicit) return explicit;
    const endpoint = document.getElementById('server-endpoint')?.value.trim();
    if (!endpoint) return '';
    try {
        const url = new URL(endpoint);
        return 'https://' + url.hostname + ':7779';
    } catch (_) {
        return '';
    }
}

function remoteEndpointHost() {
    const endpoint = document.getElementById('server-endpoint')?.value.trim();
    if (!endpoint) return '';
    try {
        const url = new URL(endpoint.includes('://') ? endpoint : 'http://' + endpoint);
        return url.hostname || '';
    } catch (_) {
        return '';
    }
}

function formatHostKey(keyHex) {
    return String(keyHex || '').match(/.{1,2}/g)?.join(':') || '';
}

function sshTargetFromConnection(connection) {
    const userHost = connection.username ? connection.username + '@' + connection.host : connection.host;
    return connection.port && connection.port !== 22 ? 'ssh://' + userHost + ':' + connection.port : userHost;
}

// ── Remote Agent Setup Modal ───────────────────────────────────────────────────

export function openRemoteAgentSetup() {
    closeConfigModal();
    closeSettingsModal();

    const modal = document.getElementById('remote-agent-setup-modal');
    if (!modal) return;
    prepareAgentSetupFromEndpoint();

    // Get current endpoint URL
    const endpointUrl = document.getElementById('endpoint-url')?.textContent || '';
    const endpointEl = document.getElementById('agent-setup-endpoint-url');
    if (endpointEl) endpointEl.textContent = endpointUrl;

    // Infer SSH host from endpoint
    let inferredHost = '';
    try {
        const url = endpointUrl.includes('://') ? endpointUrl : 'http://' + endpointUrl;
        const hostname = new URL(url).hostname;
        inferredHost = hostname;
    } catch (_) {}

    // Pre-fill fields
    const sshHostInput = document.getElementById('agent-setup-ssh-host');
    const agentUrlInput = document.getElementById('agent-setup-agent-url');
    if (sshHostInput && !sshHostInput.value && inferredHost) {
        sshHostInput.value = inferredHost;
    }
    if (agentUrlInput && !agentUrlInput.value && inferredHost) {
        agentUrlInput.value = 'https://' + inferredHost + ':7779';
    }

    // Reset state
    document.getElementById('agent-setup-host-key')?.style.setProperty('display', 'none');
    document.getElementById('btn-agent-setup-trust')?.style.setProperty('display', 'none');
    document.getElementById('agent-setup-details-section')?.style.setProperty('display', '');
    document.getElementById('agent-setup-install-section')?.style.setProperty('display', '');
    document.getElementById('agent-setup-progress')?.style.setProperty('display', 'none');
    document.getElementById('agent-setup-status')?.style.setProperty('display', 'none');
    document.getElementById('btn-agent-setup-done')?.style.setProperty('display', 'none');
    document.getElementById('btn-agent-setup-install')?.style.setProperty('display', '');
    document.getElementById('btn-agent-setup-start')?.style.setProperty('display', 'none');
    document.getElementById('btn-agent-setup-stop')?.style.setProperty('display', 'none');
    document.getElementById('btn-agent-setup-remove')?.style.setProperty('display', 'none');

    // Pre-fill version info from WebSocket data if available
    const installedEl = document.getElementById('agent-setup-installed-version');
    if (installedEl && wsData?.remote_agent_version) {
        installedEl.textContent = wsData.remote_agent_version;
    }

    // Check latest version
    checkRemoteAgentVersions();

    // Update status alert based on current agent state
    updateAgentSetupStatusAlert();

    modal.classList.add('open');
}

function closeRemoteAgentSetup() {
    document.getElementById('remote-agent-setup-modal')?.classList.remove('open');
}

function updateAgentSetupStatusAlert() {
    const alert = document.getElementById('agent-setup-status-alert');
    const icon = document.getElementById('agent-setup-status-alert-icon');
    const title = document.getElementById('agent-setup-status-alert-title');
    const message = document.getElementById('agent-setup-status-alert-message');

    if (!alert) return;

    if (!wsData) {
        alert.style.display = 'none';
        return;
    }

    const isConnected = wsData.remote_agent_connected;
    const isFirewallBlocked = wsData.remote_agent_connected && !wsData.remote_agent_health_reachable;
    const hasRemoteEndpoint = wsData.session_mode === 'attach' && wsData.endpoint_kind === 'Remote';
    const updateAvailable = wsData.remote_agent_update_available === true;
    const agentVersion = wsData.remote_agent_version || 'unknown';
    const sys = wsData.system || {};
    const hasCpuTemp = sys.cpu_temp_available && sys.cpu_temp > 0;

    if (!hasRemoteEndpoint) {
        alert.style.display = 'flex';
        alert.className = 'agent-setup-status-alert';
        icon.textContent = '\u2139\ufe0f';
        title.textContent = 'No Remote Endpoint';
        message.textContent = 'Configure a remote endpoint in Settings to enable agent management.';
        return;
    }

    // Update available takes priority
    if (isConnected && updateAvailable) {
        alert.style.display = 'flex';
        alert.className = 'agent-setup-status-alert warning';
        icon.textContent = '\u2191\ufe0f';
        title.textContent = 'Agent Update Available';
        message.textContent = 'Running v' + agentVersion + ' — a newer version is available. Click Upgrade to install it.';
        return;
    }

    if (isFirewallBlocked) {
        alert.style.display = 'flex';
        alert.className = 'agent-setup-status-alert warning';
        icon.textContent = '\u26a0\ufe0f';
        title.textContent = 'Firewall Blocking Agent';
        message.textContent = 'Agent running but HTTP port 7779 unreachable — check Windows Firewall inbound rules.';
        return;
    }

    if (!isConnected) {
        alert.style.display = 'flex';
        alert.className = 'agent-setup-status-alert';
        icon.textContent = '\ud83d\udd27';
        title.textContent = 'Agent Not Connected';
        message.textContent = 'Install or start the agent on the remote host to begin monitoring.';
        return;
    }

    // Connected — check for partial issues
    const issues = [];
    if (!hasCpuTemp) {
        issues.push(sys.cpu_temp_available
            ? 'CPU temp sensor returned no data'
            : 'sensor_bridge not installed (CPU temp unavailable)');
    }

    if (issues.length > 0) {
        alert.style.display = 'flex';
        alert.className = 'agent-setup-status-alert warning';
        icon.textContent = '\u26a0\ufe0f';
        title.textContent = 'Agent Running (with issues)';
        message.textContent = issues.join('. ') + '.';
        return;
    }

    // Fully healthy
    alert.style.display = 'flex';
    alert.className = 'agent-setup-status-alert success';
    icon.textContent = '\u2705';
    title.textContent = 'Agent Running';
    message.textContent = 'Remote agent v' + agentVersion + ' is connected and reporting all metrics.';
}

function prepareAgentSetupFromEndpoint() {
    const host = remoteEndpointHost();
    const sshHostInput = document.getElementById('agent-setup-ssh-host');
    const agentUrlInput = document.getElementById('agent-setup-agent-url');
    const configSshTarget = document.getElementById('set-remote-agent-ssh-target');
    const configAgentUrl = document.getElementById('set-remote-agent-url');

    if (sshHostInput && !sshHostInput.value.trim() && host) sshHostInput.value = host;
    if (agentUrlInput && !agentUrlInput.value.trim() && host) agentUrlInput.value = 'https://' + host + ':7779';
    if (configSshTarget && !configSshTarget.value.trim() && host) configSshTarget.value = host;
    if (configAgentUrl && !configAgentUrl.value.trim() && host) configAgentUrl.value = 'https://' + host + ':7779';
}

function updateSshSetupAuthFields() {
    const auth = document.getElementById('agent-setup-ssh-auth')?.value || 'agent';
    const passwordRow = document.getElementById('agent-setup-password-row');
    const keyRow = document.getElementById('agent-setup-key-row');
    if (passwordRow) passwordRow.style.display = auth === 'password' ? '' : 'none';
    if (keyRow) keyRow.style.display = auth === 'key' ? '' : 'none';
}

function collectRemoteAgentSetupConnection() {
    const hostInput = document.getElementById('agent-setup-ssh-host')?.value.trim() || '';
    const portInput = document.getElementById('agent-setup-ssh-port')?.value.trim() || '22';
    const auth = document.getElementById('agent-setup-ssh-auth')?.value || 'agent';
    const host = hostInput.includes('@') ? hostInput.split('@')[1] : hostInput;
    const username = hostInput.includes('@') ? hostInput.split('@')[0] : '';
    const connection = { host, username, port: parseInt(portInput, 10) };

    if (auth === 'password') {
        connection.password = document.getElementById('agent-setup-ssh-password')?.value || '';
    } else if (auth === 'key') {
        connection.private_key_path = document.getElementById('agent-setup-ssh-key-path')?.value.trim() || '';
    }

    return { auth, connection };
}

function sshTargetFromSetup() {
    const hostInput = document.getElementById('agent-setup-ssh-host')?.value.trim() || '';
    const portInput = document.getElementById('agent-setup-ssh-port')?.value.trim() || '22';
    const userHost = hostInput;
    const port = parseInt(portInput, 10);
    return port && port !== 22 ? 'ssh://' + userHost + ':' + port : userHost;
}

// ── Host Key Scanning (Setup Modal) ────────────────────────────────────────────

async function scanRemoteAgentHostKey() {
    const { connection } = collectRemoteAgentSetupConnection();
    if (!connection.host) {
        showAgentSetupStatus('Enter an SSH host first.', 'error');
        document.getElementById('agent-setup-ssh-host')?.focus();
        return;
    }

    const hostKeyEl = document.getElementById('agent-setup-host-key');
    const trustBtn = document.getElementById('btn-agent-setup-trust');
    if (hostKeyEl) {
        hostKeyEl.style.display = '';
        hostKeyEl.innerHTML = '<em>Scanning host key\u2026</em>';
    }
    if (trustBtn) trustBtn.style.display = 'none';

      try {
        const resp = await remoteAgentFetch('/api/remote-agent/ssh/host-key', {
            method: 'POST',
            headers: remoteAgentHeaders(),
            body: JSON.stringify({
                ssh_target: sshTargetFromSetup(),
                ssh_connection: connection,
            })
        });
        const data = await resp.json();
        if (!data.ok) {
            remoteAgentSetupState.hostKey = null;
            if (hostKeyEl) hostKeyEl.innerHTML = '<em style="color:#bf616a;">Scan failed: ' + escapeHtml(data.error || 'unknown') + '</em>';
            return;
        }

        remoteAgentSetupState.hostKey = data.host_key;
        const trusted = data.host_key.trusted ? 'trusted' : 'not trusted yet';
        if (hostKeyEl) {
            // eslint-disable-next-line no-unsanitized/property -- all server strings wrapped in escapeHtml(); trusted is a hardcoded string
            hostKeyEl.innerHTML = [
                '<strong>Key:</strong> ' + escapeHtml(data.host_key.key_type),
                '<strong>Host:</strong> ' + escapeHtml(data.host_key.host + ':' + data.host_key.port),
                '<strong>Fingerprint:</strong> ' + escapeHtml(formatHostKey(data.host_key.key_hex)),
                '<strong>Status:</strong> ' + trusted,
            ].join('<br>');
        }
        if (trustBtn) trustBtn.style.display = data.host_key.trusted ? 'none' : '';

        if (!data.host_key.trusted) {
            document.getElementById('agent-setup-details-section')?.style.setProperty('display', '');
        }
    } catch (err) {
        remoteAgentSetupState.hostKey = null;
        if (hostKeyEl) hostKeyEl.innerHTML = '<em style="color:#bf616a;">Scan failed: ' + escapeHtml(err.message) + '</em>';
    }
}

async function trustRemoteAgentHostKey() {
    const { connection } = collectRemoteAgentSetupConnection();
    if (!remoteAgentSetupState.hostKey?.key_hex) {
        showAgentSetupStatus('Scan the host key before trusting it.', 'error');
        return;
    }

    const resp = await remoteAgentFetch('/api/remote-agent/ssh/trust', {
        method: 'POST',
        headers: remoteAgentHeaders(),
        body: JSON.stringify({
            ssh_target: sshTargetFromSetup(),
            ssh_connection: connection,
            key_hex: remoteAgentSetupState.hostKey.key_hex,
        })
    });
    const data = await resp.json();
    if (!data.ok) {
        showAgentSetupStatus('Failed to trust host key: ' + (data.error || 'unknown'), 'error');
        return;
    }

    const hostKeyEl = document.getElementById('agent-setup-host-key');
    if (hostKeyEl) {
        hostKeyEl.innerHTML += '<br><strong style="color:#95bc7a;">\u2713 Trusted for future operations</strong>';
    }
    document.getElementById('btn-agent-setup-trust')?.style.setProperty('display', 'none');
    showAgentSetupStatus('Host key trusted. You can now install and start the agent.', 'ok');

    document.getElementById('agent-setup-install-section')?.style.setProperty('display', '');
}

// ── Version Checking ──────────────────────────────────────────────────────────

async function checkRemoteAgentVersions() {
    const latestEl = document.getElementById('agent-setup-latest-version');
    const installedEl = document.getElementById('agent-setup-installed-version');
    const installBtn = document.getElementById('btn-agent-setup-install');
    const startBtn = document.getElementById('btn-agent-setup-start');
    const stopBtn = document.getElementById('btn-agent-setup-stop');
    const removeBtn = document.getElementById('btn-agent-setup-remove');

    if (latestEl) latestEl.textContent = 'Checking\u2026';

     try {
        const resp = await remoteAgentFetch('/api/remote-agent/releases/latest', {
            headers: remoteAgentHeaders()
        });
        const data = await resp.json();
        if (data.ok && data.release?.tag_name) {
            remoteAgentSetupState.latestVersion = data.release.tag_name;
            if (latestEl) latestEl.textContent = data.release.tag_name;
        } else {
            if (latestEl) latestEl.textContent = 'Unavailable';
        }
    } catch (_) {
        if (latestEl) latestEl.textContent = 'Unavailable';
    }

    // If agent is connected via WebSocket, we already have version info
    const wsVersion = wsData?.remote_agent_version;
    const wsUpdateAvailable = wsData?.remote_agent_update_available === true;

    if (wsVersion) {
        remoteAgentSetupState.installedVersion = wsVersion;
        if (installedEl) installedEl.textContent = wsVersion;

        if (wsUpdateAvailable) {
            // Show upgrade button
            if (installBtn) {
                installBtn.style.display = '';
                installBtn.innerHTML = '<span class="btn-icon">\u2191</span> Upgrade Agent';
                installBtn.className = 'btn btn-agent-upgrade';
            }
            if (startBtn) startBtn.style.display = 'none';
            if (stopBtn) stopBtn.style.display = '';
            if (removeBtn) removeBtn.style.display = '';
        } else {
            if (installBtn) installBtn.style.display = 'none';
            if (startBtn) startBtn.style.display = 'none';
            if (stopBtn) stopBtn.style.display = '';
            if (removeBtn) removeBtn.style.display = '';
        }
        return;
    }

    // Check installed version via SSH detect
    const { connection } = collectRemoteAgentSetupConnection();
    if (!connection.host || !remoteAgentSetupState.hostKey) {
        if (installedEl) installedEl.textContent = '\u2014';
        return;
    }

    try {
        const resp = await remoteAgentFetch('/api/remote-agent/detect', {
            method: 'POST',
            headers: remoteAgentHeaders(),
            body: JSON.stringify({
                ssh_target: sshTargetFromSetup(),
                ssh_connection: connection,
                agent_url: document.getElementById('agent-setup-agent-url')?.value.trim() || null,
            })
        });
        const data = await resp.json();
        if (data.ok) {
            remoteAgentSetupState.installedVersion = data.installed_version || null;
            if (installedEl) installedEl.textContent = data.installed_version || 'Not installed';
            if (data.installed_version) {
                if (data.update_available) {
                    // Show upgrade button
                    if (installBtn) {
                        installBtn.style.display = '';
                        installBtn.innerHTML = '<span class="btn-icon">\u2191</span> Upgrade Agent';
                        installBtn.className = 'btn btn-agent-upgrade';
                    }
                } else {
                    if (installBtn) installBtn.style.display = 'none';
                }
                if (startBtn) startBtn.style.display = data.reachable ? 'none' : '';
                if (stopBtn) stopBtn.style.display = data.reachable ? '' : 'none';
                if (removeBtn) removeBtn.style.display = '';
            }
        } else {
            remoteAgentSetupState.installedVersion = null;
            if (installedEl) installedEl.textContent = 'Not installed';
        }
    } catch (_) {
        remoteAgentSetupState.installedVersion = null;
        if (installedEl) installedEl.textContent = 'Checking\u2026';
    }
}

// ── Progress / Status / Validation UI ──────────────────────────────────────────

function showAgentSetupProgress(message, percent) {
    const progressEl = document.getElementById('agent-setup-progress');
    const bar = document.getElementById('agent-setup-progress-bar');
    const text = document.getElementById('agent-setup-progress-text');

    progressEl.style.display = '';
    if (bar) bar.style.width = percent + '%';
    if (text) text.textContent = message;

    // Auto-scroll to keep progress visible
    setTimeout(() => {
        const statusEl = document.getElementById('agent-setup-status');
        if (statusEl && statusEl.offsetParent !== null) {
            statusEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
        } else if (progressEl && progressEl.offsetParent !== null) {
            progressEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }, 100);
}

function hideAgentSetupProgress() {
    document.getElementById('agent-setup-progress')?.style.setProperty('display', 'none');
}

function showAgentSetupStatus(message, kind) {
    const el = document.getElementById('agent-setup-status');
    el.style.display = '';
    el.className = 'agent-setup-status ' + kind;
    // eslint-disable-next-line no-unsanitized/property -- all call sites pass hardcoded strings or strings with server data wrapped in escapeHtml()
    el.innerHTML = message;
}

function remoteAgentSetupRequestPayload() {
    const { connection } = collectRemoteAgentSetupConnection();
    return {
        ssh_target: sshTargetFromSetup(),
        ssh_connection: connection,
        agent_url: document.getElementById('agent-setup-agent-url')?.value.trim() || inferredAgentUrl() || null,
    };
}

function renderManagedAgentStatus(data) {
    const installedEl = document.getElementById('agent-setup-installed-version');
    if (installedEl) installedEl.textContent = data.installed_version || (data.installed ? 'Unknown' : 'Not installed');

    const installBtn = document.getElementById('btn-agent-setup-install');
    const startBtn = document.getElementById('btn-agent-setup-start');
    const stopBtn = document.getElementById('btn-agent-setup-stop');
    const removeBtn = document.getElementById('btn-agent-setup-remove');

    if (installBtn) {
        installBtn.style.display = data.installed && data.managed_task_matches && !data.update_available ? 'none' : '';
        installBtn.querySelector('.btn-icon')?.replaceChildren(document.createTextNode(data.installed ? '\u21bb' : '\u2b07'));
    }
    if (startBtn) startBtn.style.display = data.running ? 'none' : '';
    if (stopBtn) stopBtn.style.display = data.running ? '' : 'none';
    if (removeBtn) removeBtn.style.display = data.installed || data.managed_task_installed ? '' : 'none';

    const managedLines = [
        '<strong>Install path:</strong> ' + escapeHtml(data.install_path || 'unknown'),
        '<strong>Installed:</strong> ' + (data.installed ? 'yes' : 'no'),
        '<strong>Running:</strong> ' + (data.running || data.reachable ? 'yes' : 'no'),
    ];
    if (data.installed_version) managedLines.push('<strong>Version:</strong> ' + escapeHtml(data.installed_version));
    if (data.managed_task_name) managedLines.push('<strong>Startup task:</strong> ' + escapeHtml(data.managed_task_name) + (data.managed_task_matches ? ' (healthy)' : ' (needs repair)'));
    if (data.managed_task_command && !data.managed_task_matches) managedLines.push('<strong>Task command:</strong> ' + escapeHtml(data.managed_task_command));
    showAgentSetupStatus(managedLines.join('<br>'), data.running || data.reachable ? 'ok' : 'info');
}

// ── Managed Agent Operations (Setup Modal) ─────────────────────────────────────

async function checkManagedRemoteAgent() {
    const { connection } = collectRemoteAgentSetupConnection();
    if (!connection.host) {
        prepareAgentSetupFromEndpoint();
    }
    if (!collectRemoteAgentSetupConnection().connection.host) {
        showAgentSetupStatus('Enter an SSH host first.', 'error');
        return { ok: false, error: 'No SSH host' };
    }

     showAgentSetupProgress('Checking managed agent\u2026', 20);
    try {
        const resp = await remoteAgentFetch('/api/remote-agent/status', {
            method: 'POST',
            headers: remoteAgentHeaders(),
            body: JSON.stringify(remoteAgentSetupRequestPayload())
        });
        const data = await resp.json();
        hideAgentSetupProgress();
        if (!data.ok) {
            showAgentSetupStatus('Status check failed: ' + escapeHtml(data.error || 'unknown'), 'error');
            return data;
        }
        renderManagedAgentStatus(data);
        return data;
    } catch (err) {
        hideAgentSetupProgress();
        showAgentSetupStatus('Status check failed: ' + escapeHtml(err.message), 'error');
        return { ok: false, error: err.message };
    }
}

async function installRemoteAgent() {
    const { connection } = collectRemoteAgentSetupConnection();
    if (!connection.host) {
        showAgentSetupStatus('Enter an SSH host first.', 'error');
        return;
    }

    // Auto-scan host key if not done
    if (!remoteAgentSetupState.hostKey) {
        showAgentSetupProgress('Scanning host key\u2026', 5);
        try {
            const resp = await remoteAgentFetch('/api/remote-agent/ssh/host-key', {
                method: 'POST',
                headers: remoteAgentHeaders(),
                body: JSON.stringify({
                    ssh_target: sshTargetFromSetup(),
                    ssh_connection: connection,
                })
            });
            const data = await resp.json();
            if (!data.ok) {
                showAgentSetupStatus('Failed to scan host key: ' + (data.error || 'unknown'), 'error');
                return;
            }
            remoteAgentSetupState.hostKey = data.host_key;
        } catch (err) {
            showAgentSetupStatus('Failed to scan host key: ' + err.message, 'error');
            return;
        }
    }

    // Auto-trust host key if not trusted
    if (!remoteAgentSetupState.hostKey.trusted) {
        showAgentSetupProgress('Trusting host key\u2026', 10);
        try {
            const resp = await remoteAgentFetch('/api/remote-agent/ssh/trust', {
                method: 'POST',
                headers: remoteAgentHeaders(),
                body: JSON.stringify({
                    ssh_target: sshTargetFromSetup(),
                    ssh_connection: connection,
                    key_hex: remoteAgentSetupState.hostKey.key_hex,
                })
            });
            const data = await resp.json();
            if (!data.ok) {
                showAgentSetupStatus('Failed to trust host key: ' + (data.error || 'unknown'), 'error');
                return;
            }
            remoteAgentSetupState.hostKey.trusted = true;
        } catch (err) {
            showAgentSetupStatus('Failed to trust host key: ' + err.message, 'error');
            return;
        }
    }

    showAgentSetupProgress('Detecting remote OS\u2026', 15);

    try {
        const detectResp = await remoteAgentFetch('/api/remote-agent/detect', {
            method: 'POST',
            headers: remoteAgentHeaders(),
            body: JSON.stringify({
                ssh_target: sshTargetFromSetup(),
                ssh_connection: connection,
                agent_url: document.getElementById('agent-setup-agent-url')?.value.trim() || null,
            })
        });
        const detectData = await detectResp.json();
        if (!detectData.ok) {
            showAgentSetupStatus('Failed to detect remote OS: ' + (detectData.error || 'unknown'), 'error');
            return;
        }

        // Auto-save token if agent is already running (repair scenario)
        maybeAutoSaveAgentToken(detectData.agent_token);

        const remoteOs = detectData.os || 'linux';
        const remoteArch = detectData.arch || 'x86_64';
        showAgentSetupProgress('Remote: ' + remoteOs + ' ' + remoteArch + '. Fetching release\u2026', 15);

        const asset = detectData.matching_asset;
        if (!asset) {
            showAgentSetupStatus('No compatible asset found for ' + remoteOs + ' ' + remoteArch + ': ' + (detectData.error || ''), 'error');
            return;
        }

        showAgentSetupProgress('Downloading ' + asset.name + '\u2026', 20);

        const installHeaders = await remoteAgentAdminHeaders();
        const resp = await remoteAgentFetch('/api/remote-agent/install', {
            method: 'POST',
            headers: installHeaders,
            body: JSON.stringify({
                ssh_target: sshTargetFromSetup(),
                ssh_connection: connection,
                asset: asset,
                install_path: detectData.install_path,
            })
        });

        if (!resp.ok) {
            const text = await resp.text();
            showAgentSetupStatus('Install failed: HTTP ' + resp.status + ' - ' + text, 'error');
            return;
        }

        const data = await resp.json();
        if (!data.ok) {
            showAgentSetupStatus('Install failed: ' + (data.error || 'unknown'), 'error');
            return;
        }

        remoteAgentSetupState.installedVersion = remoteAgentSetupState.latestVersion;
        document.getElementById('agent-setup-installed-version').textContent = remoteAgentSetupState.installedVersion;
        document.getElementById('btn-agent-setup-install')?.style.setProperty('display', 'none');
        document.getElementById('btn-agent-setup-start')?.style.setProperty('display', '');

        showAgentSetupStatus('Agent installed successfully. Starting managed agent\u2026', 'ok');
        await startRemoteAgent();
    } catch (err) {
        showAgentSetupStatus('Install failed: ' + err.message, 'error');
        hideAgentSetupProgress();
    }
}

async function upgradeRemoteAgent() {
    const { connection } = collectRemoteAgentSetupConnection();
    if (!connection.host) {
        showAgentSetupStatus('Enter an SSH host first.', 'error');
        return;
    }

    // Auto-scan host key if not done
    if (!remoteAgentSetupState.hostKey) {
        showAgentSetupProgress('Scanning host key\u2026', 5);
        try {
            const resp = await remoteAgentFetch('/api/remote-agent/ssh/host-key', {
                method: 'POST',
                headers: remoteAgentHeaders(),
                body: JSON.stringify({
                    ssh_target: sshTargetFromSetup(),
                    ssh_connection: connection,
                })
            });
            const data = await resp.json();
            if (!data.ok) {
                showAgentSetupStatus('Failed to scan host key: ' + (data.error || 'unknown'), 'error');
                return;
            }
            remoteAgentSetupState.hostKey = data.host_key;
        } catch (err) {
            showAgentSetupStatus('Failed to scan host key: ' + err.message, 'error');
            return;
        }
    }

    // Auto-trust host key if not trusted
    if (!remoteAgentSetupState.hostKey.trusted) {
        showAgentSetupProgress('Trusting host key\u2026', 10);
        try {
            const resp = await remoteAgentFetch('/api/remote-agent/ssh/trust', {
                method: 'POST',
                headers: remoteAgentHeaders(),
                body: JSON.stringify({
                    ssh_target: sshTargetFromSetup(),
                    ssh_connection: connection,
                    key_hex: remoteAgentSetupState.hostKey.key_hex,
                })
            });
            const data = await resp.json();
            if (!data.ok) {
                showAgentSetupStatus('Failed to trust host key: ' + (data.error || 'unknown'), 'error');
                return;
            }
            remoteAgentSetupState.hostKey.trusted = true;
        } catch (err) {
            showAgentSetupStatus('Failed to trust host key: ' + err.message, 'error');
            return;
        }
    }

    showAgentSetupProgress('Detecting remote OS\u2026', 15);

    try {
        const detectResp = await remoteAgentFetch('/api/remote-agent/detect', {
            method: 'POST',
            headers: remoteAgentHeaders(),
            body: JSON.stringify({
                ssh_target: sshTargetFromSetup(),
                ssh_connection: connection,
                agent_url: document.getElementById('agent-setup-agent-url')?.value.trim() || null,
            })
        });
        const detectData = await detectResp.json();
        if (!detectData.ok) {
            showAgentSetupStatus('Failed to detect remote OS: ' + (detectData.error || 'unknown'), 'error');
            return;
        }

        const asset = detectData.matching_asset;
        if (!asset) {
            showAgentSetupStatus('No compatible asset found for upgrade: ' + (detectData.error || ''), 'error');
            return;
        }

        showAgentSetupProgress('Upgrading agent\u2026', 30);

        const resp = await remoteAgentFetch('/api/remote-agent/update', {
            method: 'POST',
            headers: remoteAgentHeaders(),
            body: JSON.stringify({
                ssh_target: sshTargetFromSetup(),
                ssh_connection: connection,
                agent_url: document.getElementById('agent-setup-agent-url')?.value.trim() || null,
            })
        });

        const data = await resp.json();
        if (!data.ok) {
            showAgentSetupStatus('Upgrade failed: ' + (data.error || 'unknown'), 'error');
            return;
        }

        remoteAgentSetupState.installedVersion = data.new_version || remoteAgentSetupState.latestVersion;
        document.getElementById('agent-setup-installed-version').textContent = remoteAgentSetupState.installedVersion;

        // Reset install button to normal state
        const installBtn = document.getElementById('btn-agent-setup-install');
        if (installBtn) {
            installBtn.style.display = 'none';
            installBtn.className = 'btn btn-agent-install';
            installBtn.innerHTML = '<span class="btn-icon">\u2b07</span> Install / Repair';
        }

        maybeAutoSaveAgentToken(data.agent_token);
        showAgentSetupProgress('Agent upgraded successfully', 100);

        setTimeout(() => {
            hideAgentSetupProgress();
            showAgentSetupStatus('Agent upgraded to ' + (data.new_version || 'latest') + '. ' + (data.health_reachable ? 'Connected and reporting metrics.' : 'Started but HTTP not reachable.'), data.health_reachable ? 'ok' : 'warning');
            document.getElementById('btn-agent-setup-done').style.display = '';
        }, 800);
    } catch (err) {
        showAgentSetupStatus('Upgrade failed: ' + err.message, 'error');
        hideAgentSetupProgress();
    }
}

async function startRemoteAgent() {
    const { connection } = collectRemoteAgentSetupConnection();
    if (!connection.host) {
        showAgentSetupStatus('Enter an SSH host first.', 'error');
        return;
    }

    showAgentSetupProgress('Starting agent\u2026', 30);

    try {
        const resp = await remoteAgentFetch('/api/remote-agent/start', {
            method: 'POST',
            headers: remoteAgentHeaders(),
            body: JSON.stringify({
                ssh_target: sshTargetFromSetup(),
                ssh_connection: connection,
            })
        });
        const data = await resp.json();

        if (!data.ok) {
            showAgentSetupStatus('Start failed: ' + (data.error || 'unknown'), 'error');
            hideAgentSetupProgress();
            return;
        }

        // Auto-save the token from the start response (if the agent exposed it via HTTP)
        maybeAutoSaveAgentToken(data.agent_token);

        showAgentSetupProgress('Agent started\u2026 verifying\u2026', 80);

        await new Promise(r => setTimeout(r, 2000));

        const verifyResp = await remoteAgentFetch('/api/remote-agent/detect', {
            method: 'POST',
            headers: remoteAgentHeaders(),
            body: JSON.stringify({
                ssh_target: sshTargetFromSetup(),
                ssh_connection: connection,
                agent_url: document.getElementById('agent-setup-agent-url')?.value.trim() || null,
            })
        });
        const verifyData = await verifyResp.json();

        // Also auto-save from the detect/verify response
        maybeAutoSaveAgentToken(verifyData.agent_token);

        hideAgentSetupProgress();

        if (verifyData.ok && (verifyData.reachable || data.running)) {
            renderManagedAgentStatus({ ...verifyData, running: true });
            document.getElementById('btn-agent-setup-done').style.display = '';
        } else {
            showAgentSetupStatus('Agent started but verification failed. Check SSH logs.', 'error');
        }
    } catch (err) {
        showAgentSetupStatus('Start failed: ' + err.message, 'error');
        hideAgentSetupProgress();
    }
}

async function stopManagedRemoteAgent() {
    const { connection } = collectRemoteAgentSetupConnection();
    if (!connection.host) {
        prepareAgentSetupFromEndpoint();
    }
    if (!collectRemoteAgentSetupConnection().connection.host) {
        showAgentSetupStatus('Enter an SSH host first.', 'error');
        return;
    }

    showAgentSetupProgress('Stopping managed agent\u2026', 30);
    try {
        const resp = await remoteAgentFetch('/api/remote-agent/stop', {
            method: 'POST',
            headers: remoteAgentHeaders(),
            body: JSON.stringify(remoteAgentSetupRequestPayload())
        });
        const data = await resp.json();
        hideAgentSetupProgress();
        if (!data.ok) {
            showAgentSetupStatus('Stop failed: ' + escapeHtml(data.error || 'unknown'), 'error');
            return;
        }
        showAgentSetupStatus('Agent process stopped. The managed startup task remains installed.', 'ok');
        await checkManagedRemoteAgent();
    } catch (err) {
        hideAgentSetupProgress();
        showAgentSetupStatus('Stop failed: ' + escapeHtml(err.message), 'error');
    }
}

async function removeManagedRemoteAgent() {
    const { connection } = collectRemoteAgentSetupConnection();
    if (!connection.host) {
        prepareAgentSetupFromEndpoint();
    }
    if (!collectRemoteAgentSetupConnection().connection.host) {
        showAgentSetupStatus('Enter an SSH host first.', 'error');
        return;
    }

    const confirmed = await showConfirmDialog(
        'Remove remote agent',
        'Remove the managed remote agent from this host? This stops the process, deletes the startup task, and removes the managed binary.',
        'Remove'
    );
    if (!confirmed) return;

    showAgentSetupProgress('Removing managed agent\u2026', 30);
    try {
        const removeHeaders = await remoteAgentAdminHeaders();
        const resp = await remoteAgentFetch('/api/remote-agent/remove', {
            method: 'POST',
            headers: removeHeaders,
            body: JSON.stringify(remoteAgentSetupRequestPayload())
        });
        const data = await resp.json();
        hideAgentSetupProgress();
        if (!data.ok) {
            showAgentSetupStatus('Remove failed: ' + escapeHtml(data.error || 'unknown'), 'error');
            return;
        }
        showAgentSetupStatus('Managed agent removed from this host.', 'ok');
        document.getElementById('agent-setup-installed-version').textContent = 'Not installed';
        document.getElementById('btn-agent-setup-install')?.style.setProperty('display', '');
        document.getElementById('btn-agent-setup-start')?.style.setProperty('display', 'none');
        document.getElementById('btn-agent-setup-stop')?.style.setProperty('display', 'none');
        document.getElementById('btn-agent-setup-remove')?.style.setProperty('display', 'none');
    } catch (err) {
        hideAgentSetupProgress();
        showAgentSetupStatus('Remove failed: ' + escapeHtml(err.message), 'error');
    }
}

async function finishRemoteAgentSetup() {
    const agentUrlInput = document.getElementById('agent-setup-agent-url');
    const agentTokenInput = document.getElementById('agent-setup-agent-token');
    const sshHostInput = document.getElementById('agent-setup-ssh-host');
    const sshPortInput = document.getElementById('agent-setup-ssh-port');
    const sshAuthSelect = document.getElementById('agent-setup-ssh-auth');

    // Fetch current settings (full) to merge — avoids wiping unrelated settings
    // and preserves the real remote_agent_token (GET /api/settings masks it).
    let currentSettings = null;
    try {
        const headers = {};
        if (window.authHeaders) {
            Object.assign(headers, window.authHeaders());
        }
        const resp = await fetch('/api/settings/full', { headers });
        if (!resp.ok) {
            throw new Error('HTTP ' + resp.status);
        }
        currentSettings = await resp.json();
    } catch (err) {
        showAgentSetupStatus('Failed to load current settings before saving: ' + escapeHtml(err.message), 'error');
        return;
    }

    const settings = {
        ...currentSettings,
        remote_agent_url: agentUrlInput?.value.trim() || '',
        remote_agent_ssh_target: sshTargetFromSetup(),
        remote_agent_ssh_autostart: true,
    };

    // Only override the token if the user explicitly entered one.
    // If the token was auto-saved by maybeAutoSaveAgentToken, it's already
    // in currentSettings and will be preserved by the spread above.
    const explicitToken = agentTokenInput?.dataset.fullValue || agentTokenInput?.value.trim();
    if (explicitToken) {
        settings.remote_agent_token = explicitToken;
    } else if (agentTokenInput?.dataset.fullValue) {
        settings.remote_agent_token = agentTokenInput.dataset.fullValue;
    }

    const { auth, connection } = collectRemoteAgentSetupConnection();
    if (auth === 'password') {
        settings.remote_agent_ssh_password = connection.password;
    } else if (auth === 'key') {
        settings.remote_agent_ssh_key_path = connection.private_key_path;
    }

    try {
        await fetch('/api/settings', {
            method: 'PUT',
            headers: window.authHeaders
                ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
                : { 'Content-Type': 'application/json' },
            body: JSON.stringify(settings),
        });
    } catch (_) {}

    closeRemoteAgentSetup();
    window.location.reload();
}

// ── SSH Guide (Settings Panel) ────────────────────────────────────────────────

function inferSshGuideDefaults() {
    const hostInput = document.getElementById('ssh-guide-host');
    const userInput = document.getElementById('ssh-guide-user');
    const portInput = document.getElementById('ssh-guide-port');
    const existingTarget = document.getElementById('set-remote-agent-ssh-target')?.value.trim();
    const endpointHost = remoteEndpointHost();

    if (hostInput && !hostInput.value.trim()) {
        hostInput.value = endpointHost || '127.0.0.1';
    }

    if (portInput && !portInput.value.trim()) {
        portInput.value = '22';
    }

    if (existingTarget && existingTarget.includes('@') && userInput && !userInput.value.trim()) {
        const afterScheme = existingTarget.replace(/^ssh:\/\//, '');
        userInput.value = afterScheme.split('@')[0] || '';
    }
}

function updateSshGuideAuthFields() {
    const auth = document.getElementById('ssh-guide-auth')?.value || 'agent';
    const passwordRow = document.getElementById('ssh-guide-password-row');
    const keyRow = document.getElementById('ssh-guide-key-row');
    const passphraseRow = document.getElementById('ssh-guide-passphrase-row');

    if (passwordRow) passwordRow.style.display = auth === 'password' ? '' : 'none';
    if (keyRow) keyRow.style.display = auth === 'key' ? '' : 'none';
    if (passphraseRow) passphraseRow.style.display = auth === 'key' ? '' : 'none';
}

function openSshSetupGuide() {
    const guide = document.getElementById('ssh-setup-guide');
    if (!guide) return;

    inferSshGuideDefaults();
    updateSshGuideAuthFields();
    previewSshSetupGuide();
    guide.style.display = '';
    guide.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function closeSshSetupGuide() {
    const guide = document.getElementById('ssh-setup-guide');
    if (guide) guide.style.display = 'none';
}

function collectSshGuideConnection() {
    const host = document.getElementById('ssh-guide-host')?.value.trim() || '';
    const username = document.getElementById('ssh-guide-user')?.value.trim() || '';
    const port = parseInt(document.getElementById('ssh-guide-port')?.value, 10) || 22;
    const auth = document.getElementById('ssh-guide-auth')?.value || 'agent';
    const connection = { host, username, port };

    if (auth === 'password') {
        connection.password = document.getElementById('ssh-guide-password')?.value || '';
    } else if (auth === 'key') {
        connection.private_key_path = document.getElementById('ssh-guide-key-path')?.value.trim() || '';
        connection.private_key_passphrase = document.getElementById('ssh-guide-key-passphrase')?.value || '';
    }

    return { auth, connection };
}

function previewSshSetupGuide() {
    const plan = document.getElementById('ssh-guide-plan');
    if (!plan) return;

    const { auth, connection } = collectSshGuideConnection();
    if (!connection.host) {
        plan.textContent = 'Fill in the host details to preview the install/start plan.';
        return;
    }

    const target = sshTargetFromConnection(connection);
    const agentUrl = 'https://' + connection.host + ':7779';
    const authLabel = auth === 'password' ? 'password for this operation' : auth === 'key' ? 'private key file' : 'SSH agent or keychain';

    // eslint-disable-next-line no-unsanitized/property -- all server/computed strings wrapped in escapeHtml(); remaining items are hardcoded
    plan.innerHTML = [
        '<strong>SSH target:</strong> ' + escapeHtml(target),
        '<strong>Agent URL:</strong> ' + escapeHtml(agentUrl),
        '<strong>Auth:</strong> ' + escapeHtml(authLabel),
        '<strong>Install path:</strong> detected by OS; usually ~/.config/local-llm-foundry/bin/local-llm-foundry or %APPDATA%\\local-llm-foundry\\bin\\local-llm-foundry.exe',
        '<strong>Release source:</strong> latest Local LLM Foundry GitHub release asset matching remote OS/architecture',
        '<strong>Remote command:</strong> default OS-specific agent start command unless you override it below',
    ].join('<br>');
}

async function scanSshHostKey() {
    const { connection } = collectSshGuideConnection();
    if (!connection.host) {
        showRemoteAgentValidation('Enter a remote SSH host first.', 'error');
        document.getElementById('ssh-guide-host')?.focus();
        return;
    }

    const hostKeyEl = document.getElementById('ssh-guide-host-key');
    const trustBtn = document.getElementById('btn-ssh-guide-trust');
    if (hostKeyEl) {
        hostKeyEl.style.display = '';
        hostKeyEl.textContent = 'Scanning host key...';
    }
    if (trustBtn) trustBtn.style.display = 'none';

    try {
        const resp = await remoteAgentFetch('/api/remote-agent/ssh/host-key', {
            method: 'POST',
            headers: remoteAgentHeaders(),
            body: JSON.stringify({
                ssh_target: sshTargetFromConnection(connection),
                ssh_connection: connection,
            })
        });
        const data = await resp.json();
        if (!data.ok) {
            remoteAgent.latestHostKey = null;
            if (hostKeyEl) hostKeyEl.textContent = 'Host-key scan failed: ' + (data.error || 'unknown error');
            return;
        }

        remoteAgent.latestHostKey = data.host_key;
        if (hostKeyEl) {
            // eslint-disable-next-line no-unsanitized/property -- all server strings wrapped in escapeHtml(); trusted status is a hardcoded string
            hostKeyEl.innerHTML = [
                '<strong>Host key:</strong> ' + escapeHtml(data.host_key.key_type),
                '<strong>Host:</strong> ' + escapeHtml(data.host_key.host + ':' + data.host_key.port),
                '<strong>Fingerprint:</strong> ' + escapeHtml(formatHostKey(data.host_key.key_hex)),
                data.host_key.trusted ? '<strong>Status:</strong> trusted' : '<strong>Status:</strong> not trusted yet',
            ].join('<br>');
        }
        if (trustBtn) trustBtn.style.display = data.host_key.trusted ? 'none' : '';
    } catch (err) {
        remoteAgent.latestHostKey = null;
        if (hostKeyEl) hostKeyEl.textContent = 'Host-key scan failed: ' + err.message;
    }
}

async function trustSshHostKey() {
    const { connection } = collectSshGuideConnection();
    if (!remoteAgent.latestHostKey?.key_hex) {
        showRemoteAgentValidation('Scan the host key before trusting it.', 'error');
        return;
    }

    const resp = await remoteAgentFetch('/api/remote-agent/ssh/trust', {
        method: 'POST',
        headers: remoteAgentHeaders(),
        body: JSON.stringify({
            ssh_target: sshTargetFromConnection(connection),
            ssh_connection: connection,
            key_hex: remoteAgent.latestHostKey.key_hex,
        })
    });
    const data = await resp.json();
    if (!data.ok) {
        showRemoteAgentValidation('Failed to trust host key: ' + (data.error || 'unknown error'), 'error');
        return;
    }

    clearRemoteAgentValidation();
    document.getElementById('btn-ssh-guide-trust')?.style.setProperty('display', 'none');
    const hostKeyEl = document.getElementById('ssh-guide-host-key');
    if (hostKeyEl) {
        hostKeyEl.innerHTML += '<br><strong>Status:</strong> trusted for future SSH operations';
    }
    setRemoteAgentStatus('SSH host key trusted. You can now click <strong>Check Host</strong>, <strong>Install & Start</strong>, or <strong>Start Agent</strong>.', 'ok');
}

function applySshSetupGuide() {
    const { connection } = collectSshGuideConnection();

    if (!connection.host) {
        showRemoteAgentValidation('Enter a remote SSH host first.', 'error');
        document.getElementById('ssh-guide-host')?.focus();
        return;
    }

    const target = sshTargetFromConnection(connection);
    const targetInput = document.getElementById('set-remote-agent-ssh-target');
    const agentUrlInput = document.getElementById('set-remote-agent-url');

    if (targetInput) targetInput.value = target;
    if (agentUrlInput && !agentUrlInput.value.trim()) {
        agentUrlInput.value = 'https://' + connection.host + ':7779';
    }

    remoteAgent.sshConnection = connection;
    clearRemoteAgentValidation();
    setRemoteAgentStatus('Guided SSH settings are ready. Click <strong>Check Host</strong>, <strong>Install & Start</strong>, or <strong>Start Agent</strong> when you want to contact the remote machine.', 'info');
    saveSettings();
}

// ── Settings Panel UI Helpers ─────────────────────────────────────────────────

function remoteAgentSshPayload() {
    const sshTarget = document.getElementById('set-remote-agent-ssh-target')?.value.trim();
    const payload = { ssh_target: sshTarget };

    if (remoteAgent.sshConnection && sshTarget === sshTargetFromConnection(remoteAgent.sshConnection)) {
        payload.ssh_connection = remoteAgent.sshConnection;
    }

    return payload;
}

export function setRemoteAgentStatus(message, kind) {
    const el = document.getElementById('remote-agent-status');
    if (!el) return;

    el.style.color =
         kind === 'error' ? 'var(--color-error)' :
         kind === 'ok' ? 'var(--color-success)' :
         'var(--color-text-muted)';
    // eslint-disable-next-line no-unsanitized/property -- all call sites pass hardcoded strings or strings with server data wrapped in escapeHtml()
    el.innerHTML = message;
}

function showRemoteAgentValidation(message, type) {
    const el = document.getElementById('remote-agent-validation');
    const msgEl = document.getElementById('remote-agent-validation-message');
    if (!el || !msgEl) return;

    el.className = 'remote-agent-validation ' + type;
    msgEl.textContent = message;
    el.style.display = '';
}

function clearRemoteAgentValidation() {
    const el = document.getElementById('remote-agent-validation');
    if (el) el.style.display = 'none';
}

function showRemoteAgentProgress(message, percent, total) {
    const progressEl = document.getElementById('remote-agent-progress');
    if (!progressEl) return;

    const progressBarContainer = document.getElementById('remote-agent-progress-bar-container');
    const progressBar = document.getElementById('remote-agent-progress-bar');
    const progressText = document.getElementById('remote-agent-progress-text');

    progressEl.style.display = '';

    if (progressBarContainer && progressBar && progressText) {
        progressBar.style.width = percent + '%';
        progressText.textContent = message + (total ? ' (' + percent + '%)' : '');
    }
}

function hideRemoteAgentProgress() {
    const progressEl = document.getElementById('remote-agent-progress');
    if (progressEl) progressEl.style.display = 'none';
}

function setRemoteAgentButtonsDisabled(disabled) {
    const detectBtn = document.getElementById('btn-remote-agent-detect');
    const latestBtn = document.getElementById('btn-remote-agent-latest');
    const installBtn = document.getElementById('btn-remote-agent-install');
    const startBtn = document.getElementById('btn-remote-agent-start');
    const updateBtn = document.getElementById('btn-remote-agent-update');
    const stopBtn = document.getElementById('btn-remote-agent-stop');
    const restartBtn = document.getElementById('btn-remote-agent-restart');
    const removeBtn = document.getElementById('btn-remote-agent-remove');

    if (detectBtn) detectBtn.disabled = disabled;
    if (latestBtn) latestBtn.disabled = disabled;
    if (installBtn) installBtn.disabled = disabled;
    if (startBtn) startBtn.disabled = disabled;
    if (updateBtn) updateBtn.disabled = disabled;
    if (stopBtn) stopBtn.disabled = disabled;
    if (restartBtn) restartBtn.disabled = disabled;
    if (removeBtn) removeBtn.disabled = disabled;

    remoteAgent.inProgress = disabled;
}

// ── Status Indicator ──────────────────────────────────────────────────────────

function updateAgentStatusIndicator(connected, firewallBlocked) {
    const el = document.getElementById('agent-status');
    if (el) updateAgentStatusPill(el, connected, firewallBlocked);
}

function updateAgentStatusPill(el, connected, firewallBlocked) {
   if (!connected) {
       el.style.display = 'none';
       return;
   }
   el.style.display = 'flex';
   const fixBtn = el.querySelector('.btn-agent-fix');
   const tooltip = el.querySelector('.agent-status-tooltip');
   const statusEl = tooltip ? tooltip.querySelector('.agent-tooltip-status') : null;
   
   if (firewallBlocked) {
       el.className = 'agent-status firewall-blocked';
       const indicator = el.querySelector('.agent-indicator');
       const textEl = el.querySelector('.agent-text');
       if (indicator) indicator.textContent = '\u26a0\ufe0f';
       if (textEl) textEl.textContent = 'Firewall blocked';
       if (statusEl) {
           statusEl.textContent = 'Firewall blocked';
           statusEl.className = 'agent-tooltip-status warning';
       }
       if (fixBtn) fixBtn.style.display = '';
   } else {
       el.className = 'agent-status connected';
       const indicator = el.querySelector('.agent-indicator');
       const textEl = el.querySelector('.agent-text');
       if (indicator) indicator.textContent = '\u25cf';
       if (textEl) textEl.textContent = 'Remote Agent';
       if (statusEl) {
           statusEl.textContent = 'Connected';
           statusEl.className = 'agent-tooltip-status connected';
       }
       if (fixBtn) fixBtn.style.display = 'none';
   }
}

// ── Agent Status Badge Click ────────────────────────────────────────────────

function toggleAgentMenuFromBadge(event) {
   event.preventDefault();
   event.stopPropagation();
   openRemoteAgentSetup();
}

function openRemoteAgentSetupFromBadge(event) {
   event.preventDefault();
   event.stopPropagation();
   openRemoteAgentSetup();
}

// ── Agent Menu Actions ────────────────────────────────────────────────────────

function closeAgentMenu() {
   // No longer needed - agent-menu-panel removed
   // Kept for backward compatibility with action functions
}

async function agentMenuCheck() {
    closeAgentMenu();
    openRemoteAgentSetup();
    await checkManagedRemoteAgent();
}

async function agentMenuInstallRepair() {
    closeAgentMenu();
    openRemoteAgentSetup();
    await installRemoteAgent();
}

async function agentMenuStart() {
    closeAgentMenu();
    openRemoteAgentSetup();
    await startRemoteAgent();
}

async function agentMenuStop() {
    closeAgentMenu();
    openRemoteAgentSetup();
    await stopManagedRemoteAgent();
}

async function agentMenuRemove() {
    closeAgentMenu();
    openRemoteAgentSetup();
    await removeManagedRemoteAgent();
}

// ── Settings Panel Operations ─────────────────────────────────────────────────

function maybeAutoSaveAgentToken(token) {
    if (!token) return;
    let changed = false;

    for (const id of ['set-remote-agent-token', 'agent-setup-agent-token']) {
        const tokenInput = document.getElementById(id);
        if (!tokenInput) continue;

        const current = tokenInput.dataset.fullValue || tokenInput.value.trim();
        if (current === token) continue;

        tokenInput.dataset.fullValue = token;
        tokenInput.value = maskSecret(token);
        changed = true;
    }

    if (changed && document.getElementById('set-remote-agent-token')) {
        saveSettings();
    }
}

function maskSecret(value) {
    if (!value || value.length <= 8) {
        return '•'.repeat(value?.length || 0);
    }
    const start = value.slice(0, 4);
    const end = value.slice(-4);
    const mid = '•'.repeat(8);
    return start + mid + end;
}

async function remoteAgentLatestRelease() {
    showRemoteAgentProgress('Checking latest release...', 100, 100);
    setRemoteAgentButtonsDisabled(true);

    try {
        const resp = await remoteAgentFetch('/api/remote-agent/releases/latest', {
            headers: remoteAgentHeaders()
        });
        const data = await resp.json();

        if (!data.ok) {
            hideRemoteAgentProgress();
            setRemoteAgentButtonsDisabled(false);
            setRemoteAgentStatus('Release check failed: ' + escapeHtml(data.error || 'unknown error'), 'error');
            return;
        }

        const assets = (data.release.assets || []).map(asset => escapeHtml(asset.name)).join('<br>');
        const latestEl = document.getElementById('remote-agent-latest-version');
        if (latestEl) latestEl.textContent = data.release.tag_name || 'Unknown';

        setRemoteAgentStatus('<strong>Latest release:</strong> ' + escapeHtml(data.release.tag_name) + '<br>' + assets, 'ok');

        setTimeout(() => {
            hideRemoteAgentProgress();
            setRemoteAgentButtonsDisabled(false);
        }, 500);
    } catch (err) {
        hideRemoteAgentProgress();
        setRemoteAgentButtonsDisabled(false);
        setRemoteAgentStatus('Release check failed: ' + escapeHtml(String(err)), 'error');
    }
}

async function remoteAgentDetect(showProgress = false) {
    const sshTarget = document.getElementById('set-remote-agent-ssh-target')?.value.trim();

    if (!sshTarget) {
        clearRemoteAgentValidation();
        showRemoteAgentValidation('Enter an SSH target first.', 'error');
        document.getElementById('set-remote-agent-ssh-target')?.focus();
        return { ok: false, error: 'No SSH target' };
    }

    if (showProgress) {
        showRemoteAgentProgress('Detecting remote host...', 0, 100);
    } else {
        setRemoteAgentStatus('Detecting remote host...', 'info');
    }

    try {
        const resp = await remoteAgentFetch('/api/remote-agent/detect', {
            method: 'POST',
            headers: remoteAgentHeaders(),
            body: JSON.stringify({
                ...remoteAgentSshPayload(),
                agent_url: inferredAgentUrl() || null,
            }),
        });

        const data = await resp.json();

        maybeAutoSaveAgentToken(data.agent_token);

        const asset = data.matching_asset ? data.matching_asset.name : 'No matching asset';
        const archiveNote = data.matching_asset && data.matching_asset.archive ? ' (archive; extract before install)' : '';
        const installPath = data.install_path || 'unknown';

        const lines = [
            '<strong>' + escapeHtml(data.os) + ' / ' + escapeHtml(data.arch) + '</strong>',
            'Asset: ' + escapeHtml(asset) + escapeHtml(archiveNote),
            'Install: ' + escapeHtml(installPath),
            'Installed: ' + (data.installed ? 'yes' : 'no'),
            'Reachable: ' + (data.reachable ? 'yes' : 'no'),
        ];

        if (data.installed_version) {
            lines.push('Installed: v' + escapeHtml(data.installed_version));
        }
        if (data.managed_task_name) {
            lines.push('Startup task: ' + escapeHtml(data.managed_task_name) + (data.managed_task_matches ? ' (healthy)' : ' (needs repair)'));
        }
        if (data.update_available) {
            const latestVer = data.latest_release?.tag_name || 'unknown';
            lines.push('<strong style="color:#ebcb8b;">Update available: v' + escapeHtml(latestVer) + '</strong>');
        }
        if (data.error) lines.push('Issue: ' + escapeHtml(data.error));

        setRemoteAgentStatus(lines.join('<br>'), data.ok ? 'ok' : 'error');

        if (data.ok) {
            updateRemoteAgentPanelState(data);
        }

        clearRemoteAgentValidation();

        if (showProgress) {
            hideRemoteAgentProgress();
        }

        return data;
    } catch (err) {
        setRemoteAgentStatus('Detection failed: ' + escapeHtml(String(err)), 'error');
        if (showProgress) {
            hideRemoteAgentProgress();
        }
        return { ok: false, error: String(err) };
    }
}

async function remoteAgentInstall() {
    const sshTarget = document.getElementById('set-remote-agent-ssh-target')?.value.trim();

    if (!sshTarget) {
        clearRemoteAgentValidation();
        showRemoteAgentValidation('Enter an SSH target first.', 'error');
        document.getElementById('set-remote-agent-ssh-target')?.focus();
        return;
    }

    setRemoteAgentButtonsDisabled(true);
    clearRemoteAgentValidation();
    addTimelineItem('Installation started', 'pending');
    showRemoteAgentProgress('Detecting and installing agent...', 10, 100);

    try {
        const detectData = await remoteAgentDetect(true);

        if (!detectData.ok || !detectData.matching_asset) {
            addTimelineItem('Detection failed: ' + (detectData.error || 'unknown'), 'failed');
            hideRemoteAgentProgress();
            setRemoteAgentButtonsDisabled(false);

            if (detectData.error) {
                showRemoteAgentValidation('Detection failed: ' + detectData.error, 'error');
            } else {
                setRemoteAgentStatus('Install failed: Detection failed', 'error');
            }
            return;
        }

        const installHeaders = await remoteAgentAdminHeaders();
        const resp = await remoteAgentFetch('/api/remote-agent/install', {
            method: 'POST',
            headers: installHeaders,
            body: JSON.stringify({
                ...remoteAgentSshPayload(),
                asset: detectData.matching_asset,
                install_path: detectData.install_path,
            }),
        });

        const data = await resp.json();

        if (!data.ok) {
            addTimelineItem('Installation failed: ' + (data.error || 'unknown'), 'failed');
            hideRemoteAgentProgress();
            setRemoteAgentButtonsDisabled(false);
            setRemoteAgentStatus('Install failed: ' + escapeHtml(data.error || 'unknown'), 'error');
            return;
        }

        addTimelineItem('Installation completed', 'completed');
        showRemoteAgentProgress('Agent installed successfully', 100, 100);

        setTimeout(() => {
            hideRemoteAgentProgress();
            setRemoteAgentButtonsDisabled(false);
            setRemoteAgentStatus('Agent installed successfully at ' + escapeHtml(data.install_path || 'unknown'), 'ok');
            updateRemoteAgentPanelState(data);
            remoteAgentStart();
        }, 500);
    } catch (err) {
        addTimelineItem('Installation error: ' + escapeHtml(String(err)), 'failed');
        hideRemoteAgentProgress();
        setRemoteAgentButtonsDisabled(false);
        setRemoteAgentStatus('Install failed: ' + escapeHtml(String(err)), 'error');
    }
}

async function remoteAgentStart() {
    const sshTarget = document.getElementById('set-remote-agent-ssh-target')?.value.trim();

    if (!sshTarget) {
        clearRemoteAgentValidation();
        showRemoteAgentValidation('Enter an SSH target first.', 'error');
        document.getElementById('set-remote-agent-ssh-target')?.focus();
        return;
    }

    setRemoteAgentButtonsDisabled(true);
    addTimelineItem('Start command sent', 'pending');
    showRemoteAgentProgress('Detecting and starting agent...', 10, 100);

    try {
        const detectData = await remoteAgentDetect(true);

        if (!detectData.ok) {
            addTimelineItem('Detection failed: ' + (detectData.error || 'unknown'), 'failed');
            hideRemoteAgentProgress();
            setRemoteAgentButtonsDisabled(false);

            if (detectData.error) {
                showRemoteAgentValidation('Detection failed: ' + detectData.error, 'error');
            } else {
                setRemoteAgentStatus('Start failed: Detection failed', 'error');
            }
            return;
        }

        const installPath = detectData.install_path || '~/.config/local-llm-foundry/bin/local-llm-foundry';
        const startCommand = detectData.start_command || 'nohup ' + installPath + ' --agent --agent-host 0.0.0.0 --agent-port 7779 > ~/.config/local-llm-foundry/agent.log 2>&1 &';

        const resp = await remoteAgentFetch('/api/remote-agent/start', {
            method: 'POST',
            headers: remoteAgentHeaders(),
            body: JSON.stringify({
                ...remoteAgentSshPayload(),
                install_path: installPath,
                start_command: startCommand,
            }),
        });

        const data = await resp.json();

        if (!data.ok) {
            addTimelineItem('Start failed: ' + (data.error || 'unknown'), 'failed');
            hideRemoteAgentProgress();
            setRemoteAgentButtonsDisabled(false);
            setRemoteAgentStatus('Start failed: ' + escapeHtml(data.error || 'unknown'), 'error');
            return;
        }

        addTimelineItem('Agent started', 'completed');
        maybeAutoSaveAgentToken(data.agent_token);

        showRemoteAgentProgress('Agent started successfully', 100, 100);

        setTimeout(() => {
            hideRemoteAgentProgress();
            setRemoteAgentButtonsDisabled(false);

            let message = 'Agent started successfully';
            if (data.health_reachable) {
                message += ' and is reachable';
            } else {
                message += ', but HTTP is not reachable (firewall blocked)';
            }

            setRemoteAgentStatus(message, data.health_reachable ? 'ok' : 'warning');

            if (!data.health_reachable) {
                showRemoteAgentFirewall();
            }

            updateRemoteAgentPanelState(data);
        }, 500);
    } catch (err) {
        addTimelineItem('Start error: ' + escapeHtml(String(err)), 'failed');
        hideRemoteAgentProgress();
        setRemoteAgentButtonsDisabled(false);
        setRemoteAgentStatus('Start failed: ' + escapeHtml(String(err)), 'error');
    }
}

async function remoteAgentUpdate() {
    const sshTarget = document.getElementById('set-remote-agent-ssh-target')?.value.trim();

    if (!sshTarget) {
        clearRemoteAgentValidation();
        showRemoteAgentValidation('Enter an SSH target first.', 'error');
        document.getElementById('set-remote-agent-ssh-target')?.focus();
        return;
    }

    setRemoteAgentButtonsDisabled(true);
    addTimelineItem('Update started', 'pending');
    showRemoteAgentProgress('Checking for update...', 5, 100);

    try {
        const detectData = await remoteAgentDetect(true);

        if (!detectData.ok || !detectData.matching_asset) {
            addTimelineItem('Detection failed: ' + (detectData.error || 'unknown'), 'failed');
            hideRemoteAgentProgress();
            setRemoteAgentButtonsDisabled(false);

            if (detectData.error) {
                showRemoteAgentValidation('Detection failed: ' + detectData.error, 'error');
            } else {
                setRemoteAgentStatus('Update failed: Detection failed', 'error');
            }
            return;
        }

        showRemoteAgentProgress('Stopping, installing, and starting agent...', 20, 100);

        const resp = await remoteAgentFetch('/api/remote-agent/update', {
            method: 'POST',
            headers: remoteAgentHeaders(),
            body: JSON.stringify({
                ...remoteAgentSshPayload(),
                agent_url: inferredAgentUrl() || null,
            }),
        });

        const data = await resp.json();

        if (!data.ok) {
            addTimelineItem('Update failed: ' + (data.error || 'unknown'), 'failed');
            hideRemoteAgentProgress();
            setRemoteAgentButtonsDisabled(false);
            setRemoteAgentStatus('Update failed: ' + escapeHtml(data.error || 'unknown'), 'error');
            return;
        }

        addTimelineItem('Agent updated and started', 'completed');
        maybeAutoSaveAgentToken(data.agent_token);
        showRemoteAgentProgress('Agent updated successfully', 100, 100);

        setTimeout(() => {
            hideRemoteAgentProgress();
            setRemoteAgentButtonsDisabled(false);

            let message = 'Agent updated to ' + escapeHtml(data.new_version || 'latest');
            if (data.health_reachable) {
                message += ' and is reachable';
            } else {
                message += ', but HTTP is not reachable (firewall blocked)';
            }
            setRemoteAgentStatus(message, data.health_reachable ? 'ok' : 'warning');

            if (!data.health_reachable) {
                showRemoteAgentFirewall();
            }

            updateRemoteAgentPanelState({ ...detectData, ...data, installed: true });
        }, 500);
    } catch (err) {
        addTimelineItem('Update error: ' + escapeHtml(String(err)), 'failed');
        hideRemoteAgentProgress();
        setRemoteAgentButtonsDisabled(false);
        setRemoteAgentStatus('Update failed: ' + escapeHtml(String(err)), 'error');
    }
}

async function remoteAgentStop() {
    const sshTarget = document.getElementById('set-remote-agent-ssh-target')?.value.trim();

    if (!sshTarget) {
        clearRemoteAgentValidation();
        showRemoteAgentValidation('Enter an SSH target first.', 'error');
        document.getElementById('set-remote-agent-ssh-target')?.focus();
        return { ok: false, error: 'No SSH target' };
    }

    setRemoteAgentButtonsDisabled(true);
    addTimelineItem('Stop command sent', 'pending');
    showRemoteAgentProgress('Stopping agent...', 0, 100);

    try {
        const resp = await remoteAgentFetch('/api/remote-agent/stop', {
            method: 'POST',
            headers: remoteAgentHeaders(),
            body: JSON.stringify(remoteAgentSshPayload()),
        });

        const data = await resp.json();

        if (!data.ok) {
            addTimelineItem('Stop failed: ' + (data.error || 'unknown'), 'failed');
            hideRemoteAgentProgress();
            setRemoteAgentButtonsDisabled(false);
            setRemoteAgentStatus('Stop failed: ' + escapeHtml(data.error || 'unknown'), 'error');
            return data;
        }

        addTimelineItem('Agent stopped', 'completed');
        showRemoteAgentProgress('Agent stopped successfully', 100, 100);

        setTimeout(() => {
            hideRemoteAgentProgress();
            setRemoteAgentButtonsDisabled(false);
            setRemoteAgentStatus('Agent stopped successfully', 'ok');
            updateRemoteAgentPanelState(data);
        }, 500);

        return data;
    } catch (err) {
        addTimelineItem('Stop error: ' + escapeHtml(String(err)), 'failed');
        hideRemoteAgentProgress();
        setRemoteAgentButtonsDisabled(false);
        setRemoteAgentStatus('Stop failed: ' + escapeHtml(String(err)), 'error');
        return { ok: false, error: String(err) };
    }
}

async function remoteAgentRestart() {
    setRemoteAgentButtonsDisabled(true);
    addTimelineItem('Restart started', 'pending');

    const stopResult = await remoteAgentStop();

    if (!stopResult.ok) {
        setRemoteAgentButtonsDisabled(false);
        return;
    }

    addTimelineItem('Restarting agent...', 'pending');

    setTimeout(() => {
        remoteAgentStart();
    }, 1000);
}

async function remoteAgentRemove() {
    const sshTarget = document.getElementById('set-remote-agent-ssh-target')?.value.trim();

    if (!sshTarget) {
        clearRemoteAgentValidation();
        showRemoteAgentValidation('Enter an SSH target first.', 'error');
        document.getElementById('set-remote-agent-ssh-target')?.focus();
        return;
    }

    const ok = await showConfirmDialog(
        'Remove remote agent',
        'Remove the managed remote agent from this host? This stops the process, deletes the startup task, and removes the managed binary.',
        'Remove'
    );
    if (!ok) {
        return;
    }

    setRemoteAgentButtonsDisabled(true);
    addTimelineItem('Remove command sent', 'pending');
    showRemoteAgentProgress('Removing managed agent...', 0, 100);

    try {
        const removeHeaders = await remoteAgentAdminHeaders();
        const resp = await remoteAgentFetch('/api/remote-agent/remove', {
            method: 'POST',
            headers: removeHeaders,
            body: JSON.stringify(remoteAgentSshPayload()),
        });

        const data = await resp.json();

        if (!data.ok) {
            addTimelineItem('Remove failed: ' + (data.error || 'unknown'), 'failed');
            hideRemoteAgentProgress();
            setRemoteAgentButtonsDisabled(false);
            setRemoteAgentStatus('Remove failed: ' + escapeHtml(data.error || 'unknown'), 'error');
            return data;
        }

        addTimelineItem('Managed agent removed', 'completed');
        showRemoteAgentProgress('Managed agent removed', 100, 100);

        setTimeout(() => {
            hideRemoteAgentProgress();
            setRemoteAgentButtonsDisabled(false);
            setRemoteAgentStatus('Managed agent removed from this host.', 'ok');
            updateRemoteAgentPanelState({ installed: false, running: false, managed_task_installed: false });
        }, 500);

        return data;
    } catch (err) {
        addTimelineItem('Remove error: ' + escapeHtml(String(err)), 'failed');
        hideRemoteAgentProgress();
        setRemoteAgentButtonsDisabled(false);
        setRemoteAgentStatus('Remove failed: ' + escapeHtml(String(err)), 'error');
        return { ok: false, error: String(err) };
    }
}

// ── Panel State Rendering ─────────────────────────────────────────────────────

function updateRemoteAgentPanelState(data) {
    const versionsEl = document.getElementById('remote-agent-versions');
    if (!versionsEl) return;

    const latestVer = data.latest_release?.tag_name || data.release?.tag_name || 'Not checked';
    const installedVer = data.installed_version || (data.installed ? 'Unknown' : 'Not installed');

    document.getElementById('remote-agent-latest-version').textContent = latestVer;
    document.getElementById('remote-agent-installed-version').textContent = installedVer;
    versionsEl.style.display = '';

    const isInstalled = data.installed || false;
    const isRunning = data.running || false;
    const isUpdateAvailable = data.update_available || false;

    const updateIndicator = document.getElementById('remote-agent-update-indicator');
    if (updateIndicator) {
        updateIndicator.style.display = isUpdateAvailable ? 'flex' : 'none';
        updateIndicator.style.color = 'var(--color-warning)';
    }

    const buttonsEl = document.getElementById('remote-agent-buttons');
    if (buttonsEl) {
        const installBtn = document.getElementById('btn-remote-agent-install');
        const startBtn = document.getElementById('btn-remote-agent-start');
        const updateBtn = document.getElementById('btn-remote-agent-update');
        const stopBtn = document.getElementById('btn-remote-agent-stop');
        const restartBtn = document.getElementById('btn-remote-agent-restart');
        const removeBtn = document.getElementById('btn-remote-agent-remove');

        if (installBtn) installBtn.style.display = isInstalled ? 'none' : '';
        if (startBtn) startBtn.style.display = isRunning ? 'none' : '';
        if (updateBtn) {
            if (isUpdateAvailable) {
                updateBtn.style.display = '';
                updateBtn.textContent = 'Update Agent';
            } else if (isRunning) {
                updateBtn.textContent = 'Restart';
                updateBtn.style.display = '';
            } else {
                updateBtn.style.display = 'none';
            }
        }
        if (stopBtn) stopBtn.style.display = isRunning ? '' : 'none';
        if (restartBtn) restartBtn.style.display = isRunning ? '' : 'none';
        if (removeBtn) removeBtn.style.display = (isInstalled || data.managed_task_installed) ? '' : 'none';

        if (isRunning && isUpdateAvailable) {
            document.getElementById('remote-agent-status-indicator').textContent = '\u25cf Update available';
            document.getElementById('remote-agent-status-indicator').style.color = 'var(--color-warning)';
        } else if (isRunning) {
            document.getElementById('remote-agent-status-indicator').textContent = '\u25cf Ready';
            document.getElementById('remote-agent-status-indicator').style.color = 'var(--color-success)';
        } else {
            document.getElementById('remote-agent-status-indicator').textContent = '\u25cf Not running';
            document.getElementById('remote-agent-status-indicator').style.color = 'var(--color-text-muted)';
        }
    }
}

function showRemoteAgentFirewall(showAlert = true) {
    const firewallEl = document.getElementById('remote-agent-firewall');
    if (firewallEl) {
        firewallEl.style.display = '';
    }
    if (showAlert && typeof showToast === 'function') {
        showToast('Firewall blocked - Agent HTTP access is not reachable', 'error');
    }
}

function openFirewallHelp() {
    openConfigModal();

    const panel = document.getElementById('remote-agent-panel');
    if (panel) panel.open = true;

    const agentUrlInput = document.getElementById('set-remote-agent-url');
    if (agentUrlInput && !agentUrlInput.value.trim()) {
        agentUrlInput.value = inferredAgentUrl();
    }

    const sshTargetInput = document.getElementById('set-remote-agent-ssh-target');
    if (sshTargetInput && !sshTargetInput.value.trim()) {
        sshTargetInput.value = remoteEndpointHost();
    }

    const firewallEl = document.getElementById('remote-agent-firewall');
    if (firewallEl && firewallEl.style.display === 'none') {
        firewallEl.style.display = '';
    }

    setRemoteAgentStatus(
        'Configure the remote agent for this host, then use <strong>Install & Start</strong> or <strong>Start Agent</strong>. If the agent starts but remains unreachable, open TCP port <strong>7779</strong> on the remote machine.',
        'info'
    );

    setTimeout(() => {
        if (firewallEl) firewallEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
        sshTargetInput?.focus();
        sshTargetInput?.select();
    }, 50);
}

// ── Timeline ──────────────────────────────────────────────────────────────────

function addTimelineItem(message, status) {
    const timelineEl = document.getElementById('remote-agent-timeline');
    const itemsEl = document.getElementById('remote-agent-timeline-items');
    if (!timelineEl || !itemsEl) return;

    timelineEl.style.display = '';

    const timestamp = new Date().toLocaleTimeString();
    const item = document.createElement('div');
    item.className = 'remote-agent-timeline-item ' + status;
    // eslint-disable-next-line no-unsanitized/property -- timestamp is from Date(); message is passed by internal callers as hardcoded strings or escapeHtml()-wrapped server data; status is a hardcoded enum
    item.innerHTML = '<span class="timestamp">[' + timestamp + ']</span>' + message;

    itemsEl.appendChild(item);
    itemsEl.scrollTop = itemsEl.scrollHeight;
}

function clearTimeline() {
    const itemsEl = document.getElementById('remote-agent-timeline-items');
    if (itemsEl) {
        itemsEl.innerHTML = '';
    }
    const timelineEl = document.getElementById('remote-agent-timeline');
    if (timelineEl) {
        timelineEl.style.display = 'none';
    }
}

// ── Init ───────────────────────────────────────────────────────────────────────

export function initRemoteAgent() {
    if (initialized) return;
    initialized = true;

    // Bind agent status badge (opens setup modal)
    document.getElementById('agent-status')?.addEventListener('click', (e) => {
        if (!e.target.closest('.btn-agent-fix')) toggleAgentMenuFromBadge(e);
    });

    // Bind agent fix button
    document.getElementById('agent-fix-btn')?.addEventListener('click', (e) => {
        e.stopPropagation();
        openRemoteAgentSetupFromBadge(e);
    });

    // Bind remote agent setup modal buttons
    document.getElementById('remote-agent-close')?.addEventListener('click', closeRemoteAgentSetup);
    document.getElementById('remote-agent-cancel')?.addEventListener('click', closeRemoteAgentSetup);
    document.getElementById('btn-agent-setup-done')?.addEventListener('click', finishRemoteAgentSetup);
    document.getElementById('btn-agent-setup-scan')?.addEventListener('click', scanRemoteAgentHostKey);
    document.getElementById('btn-agent-setup-trust')?.addEventListener('click', trustRemoteAgentHostKey);
    document.getElementById('btn-agent-setup-status')?.addEventListener('click', checkManagedRemoteAgent);
    document.getElementById('btn-agent-setup-install')?.addEventListener('click', () => {
        const btn = document.getElementById('btn-agent-setup-install');
        if (btn && btn.classList.contains('btn-agent-upgrade')) {
            upgradeRemoteAgent();
        } else {
            installRemoteAgent();
        }
    });
    document.getElementById('btn-agent-setup-start')?.addEventListener('click', startRemoteAgent);
    document.getElementById('btn-agent-setup-stop')?.addEventListener('click', stopManagedRemoteAgent);
    document.getElementById('btn-agent-setup-remove')?.addEventListener('click', removeManagedRemoteAgent);

    // Bind remote agent buttons in config modal
    document.getElementById('btn-remote-agent-guide')?.addEventListener('click', openSshSetupGuide);
    document.getElementById('btn-remote-agent-detect')?.addEventListener('click', () => remoteAgentDetect(true));
    document.getElementById('btn-remote-agent-latest')?.addEventListener('click', remoteAgentLatestRelease);
    document.getElementById('btn-remote-agent-install')?.addEventListener('click', remoteAgentInstall);
    document.getElementById('btn-remote-agent-start')?.addEventListener('click', remoteAgentStart);
    document.getElementById('btn-remote-agent-update')?.addEventListener('click', remoteAgentUpdate);
    document.getElementById('btn-remote-agent-quick-update')?.addEventListener('click', e => {
        e.stopPropagation();
        document.getElementById('btn-remote-agent-update')?.click();
        setTimeout(() => {
            document.getElementById('remote-agent-progress')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }, 200);
    });
    document.getElementById('btn-remote-agent-stop')?.addEventListener('click', remoteAgentStop);
    document.getElementById('btn-remote-agent-restart')?.addEventListener('click', remoteAgentRestart);
    document.getElementById('btn-remote-agent-remove')?.addEventListener('click', remoteAgentRemove);
    document.getElementById('btn-firewall-help')?.addEventListener('click', openFirewallHelp);

    // Bind SSH guide buttons
    document.getElementById('ssh-guide-close')?.addEventListener('click', closeSshSetupGuide);
    document.getElementById('ssh-guide-preview')?.addEventListener('click', previewSshSetupGuide);
    document.getElementById('ssh-guide-scan')?.addEventListener('click', scanSshHostKey);
    document.getElementById('btn-ssh-guide-trust')?.addEventListener('click', trustSshHostKey);
    document.getElementById('ssh-guide-apply')?.addEventListener('click', applySshSetupGuide);

    // Agent setup modal overlay click
    const modal = document.getElementById('remote-agent-setup-modal');
    if (modal) {
        modal.addEventListener('click', e => {
            if (e.target === e.currentTarget) closeRemoteAgentSetup();
        });
    }

    // Auth selector change (setup modal)
    const authSelect = document.getElementById('agent-setup-ssh-auth');
    if (authSelect) {
        authSelect.addEventListener('change', updateSshSetupAuthFields);
    }

    // Auth selector change (SSH guide)
    const guideAuthSelect = document.getElementById('ssh-guide-auth');
    if (guideAuthSelect) {
        guideAuthSelect.addEventListener('change', updateSshGuideAuthFields);
    }

    // SSH guide input listeners for live preview
    ['ssh-guide-host', 'ssh-guide-user', 'ssh-guide-port', 'ssh-guide-password', 'ssh-guide-key-path', 'ssh-guide-key-passphrase'].forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('input', previewSshSetupGuide);
        }
    });

    // Close agent menu on outside click
    document.addEventListener('click', event => {
        if (!event.target.closest('.agent-menu')) {
            closeAgentMenu();
        }
    });

    // Sensor bridge setup button
    const sensorBtn = document.getElementById('btn-sensor-bridge-setup');
    if (sensorBtn) {
        sensorBtn.addEventListener('click', async () => {
            sensorBtn.disabled = true;
            sensorBtn.textContent = 'Installing...';
            const callout = document.getElementById('sensor-bridge-setup-callout');
            try {
                const res = await fetch('/api/sensor-bridge/install', {
                            method: 'POST',
                            headers: window.authHeaders ? window.authHeaders() : {},
                        });
                const data = await res.json();
                if (!data.started) {
                    sensorBtn.textContent = 'Setup';
                    sensorBtn.disabled = false;
                    if (callout) {
                        callout.innerHTML = '<span style="color:#bf616a;">Install failed: ' + escapeHtml(data.error || 'Unknown error') + '</span>';
                    }
                    return;
                }
                if (callout) {
                    callout.innerHTML = '<span style="color:#a3be8c;">A UAC prompt will appear on your desktop \u2014 approve it to install the sensor service. This takes a few seconds.</span>';
                }
                // Poll for running status up to 30 seconds
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
                            if (callout) callout.style.display = 'none';
                        } else if (elapsed >= 30000) {
                            clearInterval(poll);
                            sensorBtn.textContent = 'Setup';
                            sensorBtn.disabled = false;
                            if (callout) {
                                callout.innerHTML = 'CPU temperature requires a one-time service install. <button id="btn-sensor-bridge-setup" style="margin-left:8px; padding:3px 10px; background:#5e81ac; border:none; border-radius:4px; color:#eceff4; cursor:pointer; font-size:12px;">Setup</button><span style="color:#ebcb8b; margin-left:8px;">Timed out \u2014 did you approve the UAC prompt?</span>';
                                const newBtn = document.getElementById('btn-sensor-bridge-setup');
                                if (newBtn) newBtn.addEventListener('click', () => sensorBtn.click());
                            }
                        }
                    } catch (_) {}
                }, 2000);
            } catch (e) {
                sensorBtn.textContent = 'Setup';
                sensorBtn.disabled = false;
            }
        });
    }
}
