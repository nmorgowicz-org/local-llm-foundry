# Local LLM Foundry Remote Agent

Foundry 2.0 keeps the `llama-monitor` agent command, legacy token paths, and
existing protocol fields compatible through 2.x. New installs use the canonical
root and executable; detection preserves an existing custom or legacy install
path until the user explicitly migrates it.

Managed-path detection checks both product roots with both executable names. This
covers upgrades where the rebranded `local-llm-foundry.exe` was installed inside
the retained `%APPDATA%\llama-monitor\bin` root. Install, start, and SSH
autostart all use the same detected path, including its matching configuration
directory.

The remote agent adds host-level telemetry to remote llama.cpp endpoints. Without it, Llama Monitor can still attach to a remote server and read performance metrics, but GPU, CPU, RAM, and host-health data remain unavailable.

Implementation lives in `src/web/api/remote_agent.rs` (API endpoints) and the agent binary's own codebase. The endpoints are composed into the main router in `src/web/api/mod.rs`.

## What it adds

When a remote agent is installed and reachable, the dashboard can show:

- GPU temperature, utilization, VRAM, power, and clocks
- CPU load, temperature, RAM, and host/platform details
- Agent reachability, installed version, and update availability

This is what upgrades a remote attach session from **Basic** to **Full telemetry**.

![GPU & System telemetry from a remote agent](../screenshots/neutral--gpu-metrics.gif)

## Agent states and indicators

When attached to a remote endpoint, the header Agent button and Remote Agent panel reflect several states:

- **Connected**:
  - Agent process is reachable and HTTP health checks succeed.
  - Shows “Remote Agent” with a connected indicator.

- **Firewall blocked**:
  - Agent process is considered connected (e.g., started via SSH), but the dashboard cannot reach its HTTP endpoint.
  - Triggered when `remote_agent_connected` is true but `remote_agent_health_reachable` is false.
  - In practice, both `remote_agent_connected` and `remote_agent_health_reachable` are set and reset together:
    - On successful `/metrics` call: both become true.
    - On disconnect (`mark_disconnected`): both become false.
    - A steady state where only one is false (the “Firewall blocked” condition) is transient or UI-side and should not be assumed to be stable.
  - Header shows “Firewall blocked” with a **Fix** button.
  - Remote Agent panel shows “Agent Started, HTTP Blocked” with firewall guidance.

- **Update Available**:
  - The dashboard compares the agent’s version (from `/info`) against the latest release.
  - If the running version is older, it sets `remote_agent_update_available`.
  - Header shows “Update Available” with an **Upgrade** button.
  - Remote Agent panel shows an “Update available” indicator and an **Upgrade** / **Update Agent** button.

- **Agent tooltip**:
  - Hovering the Agent button shows a tooltip with:
    - Status line (Connected / Not connected / Firewall blocked / Update available).
    - Running version (e.g., “Running v1.2.3”) if known.
    - Agent URL if configured.

## Entry points in the UI

There are two current ways into remote-agent setup:

### Header Agent button

The **Agent** control in the top header is the fast path for remote telemetry. It is also where the dashboard surfaces status and a **Fix** action when a remote endpoint needs agent attention.

Use it when you want a guided setup flow focused on one remote host.

### Configuration modal

For manual control, open:

`Settings → Advanced → Open Runtime Configuration → Remote Agent`

This panel exposes the full runtime controls:

- Guided SSH setup
- Host check
- Release check
- Install and start
- Start, stop, restart, update, and remove
- Saved agent URL, token, SSH target, optional autostart, and optional custom SSH start command

## mTLS and trust

Remote agents and dashboards communicate over mutual TLS (mTLS). Both sides present certificates; both sides verify them.

### Per-device CA model

Each device (Mac, Android, etc.) generates its own CA (`ca.pem` / `ca.key`) and signs its own client certificate. Device CAs are kept independent — no device's key ever leaves that device.

Trust is established in two directions:

**Agent trusts clients** via the `cas/` directory:
- Each enrolled device has its own `cas/<instance-id>.pem` entry.
- On startup the agent loads all entries in this directory.
- A `tokio::sync::watch` channel allows new entries to take effect without a restart.
- A legacy single-CA fallback (`ca.pem`) is also supported for old installs.

**Dashboard trusts the agent** via the `remote-cas/` directory:
- Each agent host has its own `remote-cas/<hostname>.pem` entry.
- The dashboard loads all entries in this directory when building its HTTPS client.
- This allows connecting to multiple remote agents, each with its own CA.

### Automatic SSH bootstrap enrollment

When a dashboard fails to connect to a remote agent (mTLS rejection) and SSH is configured, it automatically bootstraps trust without any user interaction:

1. Reads the agent's CA cert (`ca.pem`) from the remote machine via SSH → saved to `remote-cas/<hostname>.pem`.
2. Reads the agent's api-token from the remote machine via SSH.
3. Ensures this device has a local CA + client cert (generates if missing).
4. POSTs this device's CA to the agent's enrollment endpoint, authenticated with the api-token.
5. Saves the api-token so ongoing API calls authenticate correctly.

The agent exposes a secondary enrollment port at `agent_port + 1` that does not require a client certificate (server-TLS only), to allow unenrolled dashboards to register their CA:

- **POST /api/agent/register-ca**:
  - Body: `{ "action": "register-ca", "ca_pem": "<base64 or PEM>", "instance_id": "<id>" }`
  - Auth: bearer token using the agent token.
  - On success, writes the CA into the agent's `cas/<instance_id>.pem` and reloads its TLS config so the new CA is trusted immediately.

On the next poll iteration, mTLS succeeds in both directions.

This flow requires only that SSH is configured (via Guided SSH Setup). No codes, QR codes, or manual credential management are needed.

### Multi-device independence

Each device that enrolls gets its own entry in the agent's `cas/` directory. Enrolling a new device does not affect existing enrolled devices.

## Agent tokens

There are two conceptually separate tokens:

- **--agent-token** (agent-mode bearer):
  - Set via `--agent-token` when running llama-monitor in agent mode (`--agent`).
  - Used by the agent process to authenticate its HTTP endpoints (e.g. `/info`, `/metrics`).
  - Dashboards or callers include it as `Authorization: Bearer <token>` when talking directly to the agent.

- **Dashboard-side remote_agent_token** (config / UI):
  - Stored in `ui-settings.json` as `remote_agent_token`.
  - Used by the dashboard (and its `/api/remote-agent/*` endpoints) to talk to the remote agent.
  - This is how the main app authenticates to the agent during polling and management.

The agent itself:

- The primary agent token is configured via --agent-token or the UI.
- The agent also supports multiple allowed tokens via agent-tokens.json:
  - File: ~/.config/llama-monitor/agent-tokens.json
  - Format: { "tokens": ["<token1>", "<token2>"] }
  - Any token in this list is accepted for authenticated agent endpoints.
- On startup, the primary token is automatically ensured in this file, enabling multi-client setups (for example, multiple dashboards polling the same agent).

## Agent-mode API surface

When llama-monitor runs with `--agent` (agent mode), it exposes a small HTTP API
(protected by `--agent-token`) used by the dashboard and remote clients.

Endpoints:

- **GET /health**
  - Unauthenticated health check; returns `{ "ok": true }`.
- **GET /info**
  - Authenticated; returns version, protocol_version, mode, platform, bind address.
- **GET /agent/info**
  - Same as `/info`, plus the current `agent_token` (for verification flows).
- **GET /metrics/system**
  - Returns live system metrics (CPU, RAM, platform details).
- **GET /metrics/gpu**
  - Returns live GPU metrics by cluster.
- **GET /metrics**
  - Combined endpoint returning both system and GPU metrics in a single payload.

### Idle polling behavior

The installed agent keeps its GPU and system metric workers idle until an
authenticated master request is received. Authenticated `/info`, `/agent/info`,
and `/metrics*` requests refresh the activity lease; `/health` remains available
for reachability checks but does not wake metric collection. After 180 seconds
without an authenticated request, workers return to idle and check for activity
at bounded five-second intervals. This applies consistently on macOS, Linux, and
Windows; it is especially important for Windows service installs, which may run
without a connected master monitor.

All authenticated endpoints require:
- Header: `Authorization: Bearer <token>` where `<token>` is the agent token.

## Protocol and versioning

- Agent endpoints:
  - GET /info: returns agent version and protocol_version.
  - GET /agent/info: same, plus agent_token for verification.
- The current protocol version is 1.0.0.
- The dashboard enforces a minimum protocol version when polling the agent.
- The dashboard reads `protocol_version` from the agent `/info` endpoint on each poll.
- Below minimum version: the dashboard enters degraded compatibility mode and sets the `protocol_too_old` flag. The telemetry grade reflects the degraded state in the UI.

## Version mismatch and degraded mode

- On each agent poll, the dashboard:
  - Reads the agent’s `protocol_version` from `/info`.
  - Compares it to its enforced minimum (currently 1.0.0).
- If `protocol_version` is:
  - Below 1.0.0, or
  - Missing (older agents that don’t report it),
  then:
  - The dashboard keeps the agent marked as connected.
  - It sets a “protocol too old” flag and logs a warning.
  - It runs in degraded compatibility mode: partial metrics may be available, advanced features may be limited.
- If metrics parsing fails after a successful HTTP response:
  - The dashboard treats this as degraded instead of disconnected,
    preserving partial telemetry when possible.

## Current setup flow

The real setup flow is now explicit and opt-in. Typing an endpoint or SSH target does not trigger SSH activity by itself.

### Guided SSH setup

In the Runtime Configuration panel, click **Guided SSH Setup**.

1. Enter the remote host, username, port, and auth method.
2. Click **Preview Plan** to review the inferred agent URL, SSH target, auth mode, and expected install path.
3. Click **Scan Host Key** to fetch the host fingerprint.
4. Click **Trust Host Key** if the fingerprint is correct.
5. Click **Use These Settings** to populate the saved SSH target and agent URL.
6. From the main Remote Agent controls, explicitly choose **Check Host**, **Install & Start**, or **Start Agent**.

The guide only prepares and validates settings. It does not contact the remote machine until you click one of the action buttons.

### Header setup modal

When the dashboard knows a remote endpoint needs an agent, the header flow opens a dedicated **Set Up Remote Agent** modal. That flow is step-based:

1. Enter the SSH host and port.
2. Choose authentication.
3. Scan and trust the host key.
4. Review the generated agent details.
5. Install/start the agent and save the resulting URL/token back into settings.

Use this flow when the app prompts you directly from a remote attach session.

## Saved runtime settings

Remote-agent connection details are persisted as runtime configuration, not ordinary UI preferences.

| Setting | Purpose |
|---------|---------|
| **Agent URL** | Explicit polling URL for the remote agent; if left blank, the app infers `https://<remote-host>:7779` from the attached endpoint host |
| **Agent Token** | Optional bearer token required by the agent |
| **SSH target** | Saved `user@host` target for manual agent actions or optional autostart |
| **After attach, try SSH start if the agent is unreachable** | Enables one autostart attempt after attaching to a remote endpoint |
| **Custom SSH start command** | Overrides the built-in OS-specific start command |

## SSH behavior and trust

Host-key verification is required before managed SSH operations.

1. The app scans the host key over SSH.
2. It shows the host, key type, and fingerprint.
3. Trusting the key stores it locally for future verification.
4. If the key changes later, the dashboard rejects the managed SSH operation until you re-scan and trust the new key.

The dedicated SSH backend supports:

- SSH agent / keychain
- Password for the current operation
- Private key path, with optional passphrase

## Autostart behavior

If **After attach, try SSH start if the agent is unreachable** is enabled, the dashboard will make one saved SSH start attempt after you attach to a remote endpoint and discover that the agent is unavailable.

This is not global background probing. It only applies after an attach, and only when you have explicitly enabled the setting.

When autostart is used, the agent token is persisted on the remote machine (`agent-token` file) and reused across restarts; SSH does not generate a new token on each start.

## Platform behavior

### Windows

On Windows, the managed install uses scheduled tasks for both the agent and `sensor_bridge.exe`, allowing startup without an interactive session. The installer resolves `%APPDATA%` to the remote user's absolute profile path before SCP, archive extraction, certificate installation, and task registration. It also searches nested archive directories and fails loudly if `local-llm-foundry.exe` is missing, rather than leaving a task that silently exits with `0x80070002`.

### Linux and macOS

The app chooses an OS-appropriate install/start path and command for the managed agent. The guided plan preview shows the expected install location before you run it.

## Troubleshooting

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| **Basic** status on a remote session | No agent is running or the agent URL is not reachable | Open the Agent flow or Runtime Configuration and start/install the agent |
| **Agent Started, HTTP Blocked** warning | The process started, but port `7779` is blocked or bound incorrectly | Open the remote firewall and confirm the agent is listening on `0.0.0.0` when remote access is required |
| Host-key mismatch | The remote server fingerprint changed | Re-scan the key and verify the host before trusting it again |
| Start/install buttons complain about missing SSH target | Guided settings were not applied or no target was saved | Run **Guided SSH Setup** and click **Use These Settings** |
| Agent is reachable but some host metrics are missing | The remote system cannot provide those sensors | Check the underlying platform tools such as `nvidia-smi`, `rocm-smi`, or OS temperature access |

## Stored secrets and security

Credentials used by the remote agent and internal APIs are encrypted at rest using AES-256-GCM and stored in `~/.config/llama-monitor/`.

Encryption:
- Llama-monitor:
  - Uses an encryption key:
    - From `LLAMA_MONITOR_ENCRYPTION_KEY` if set and ≥16 characters, or
    - Auto-generated and stored in `encryption-key` in the config directory.
  - Automatically encrypts:
    - `remote_agent_token`
    - `api-token`
    - `db-admin-token`
    - ACME `dns_config` values
- No manual key management is required.

Encrypted values:

- **Agent Token** (`remote_agent_token`):
  - Used as a bearer token when polling the remote agent.
  - Encrypted at rest.
  - Masked in `GET /api/settings`; real value available via `GET /api/settings/full` with `api-token` auth.
- **SSH password** (`remote_agent_ssh_password`):
  - Used for SSH operations when password auth is selected.
  - Stored in `ui-settings.json` and reused for subsequent operations; not considered transient.
- **Private key path** (`remote_agent_ssh_key_path`):
  - Path reference only; the key file itself is not copied into llama-monitor’s config.

Notes:
- These values are never logged in full (only “token generated”-style messages).
- They are not included in generic debug or health responses.
- For security-conscious setups:
  - Treat `~/.config/llama-monitor/` as sensitive.
  - Use file permissions and/or disk encryption where available.

## Endpoint authentication

All `/api/remote-agent/*` endpoints require a bearer token.

- **api-token** (standard operations):
  - `GET /api/remote-agent/releases/latest`
  - `POST /api/remote-agent/detect`
  - `POST /api/remote-agent/ssh/host-key`
  - `POST /api/remote-agent/ssh/trust`
  - `POST /api/remote-agent/status`
  - `POST /api/remote-agent/start`
  - `POST /api/remote-agent/update`
  - `POST /api/remote-agent/stop`
  - `GET /api/remote-agent/tls-status`
    - Returns:
      - `mtls_enforced`: true
      - `ca_present`: whether `ca.pem` exists
      - `server_cert_present`: whether `agent-server.pem` exists
      - `client_cert_present`: whether `agent-client.pem` exists

- **db-admin-token** (elevated operations):
  - `POST /api/remote-agent/install`
  - `POST /api/remote-agent/remove`

Requests must include:
- Header: `Authorization: Bearer <token>`
- If missing or invalid, the endpoint returns 401 with an error message.

The frontend automatically includes the appropriate token via `window.authHeaders()` (api-token) or by fetching the db-admin-token for install/remove.

## Remote agent config file

During install, the dashboard writes a `remote-agent-config.json` file next to the agent binary with:

- `api_token`: the dashboard's api-token, used by the agent (or SSH-managed operations) to authenticate to `/api/remote-agent/*` endpoints.

This file is written with restrictive permissions (0600 on Unix/macOS) so only the agent process can read it.

## Tokens and rotation

Three tokens are used by llama-monitor:

- **API Token**:
  - File: `api-token`
  - Used to protect sensitive endpoints (attach, DB queries, TLS/ACME, remote-agent endpoints, etc.).
  - Encrypted at rest.
- **DB Admin Token**:
  - File: `db-admin-token`
  - Used for advanced database operations and restricted queries, plus elevated remote-agent endpoints (install/remove).
  - Encrypted at rest.
- **Agent Token**:
  - Stored in `ui-settings.json` as `remote_agent_token`.
  - Used to authenticate with the remote agent.
  - Encrypted at rest.

All three can be rotated from the UI:

- Open `Settings → Security & Certificates`.
- Use one of:
  - `Rotate Agent Token`
  - `Rotate API Token`
  - `Rotate DB Admin Token`
- Confirm the prompt. A new token is generated and the previous one is immediately invalid.

After rotating the API Token or DB Admin Token, restart llama-monitor to fully apply the change.

## CLI equivalents

The dashboard-side CLI flags for remote-agent integration are documented in [CLI Flags](cli-flags.md). The agent itself can be launched directly with `--agent`, `--agent-host`, `--agent-port`, and `--agent-token`.
