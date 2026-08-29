use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use ssh2::Session;

const DEFAULT_SSH_PORT: u16 = 22;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(8);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SshConnection {
    pub host: String,
    #[serde(default)]
    pub username: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default)]
    pub password: Option<String>,
    #[serde(default)]
    pub private_key_path: Option<String>,
    #[serde(default)]
    pub private_key_passphrase: Option<String>,
    #[serde(default)]
    pub trusted_host_key: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SshCommandOutput {
    pub status: i32,
    pub stdout: String,
    pub stderr: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SshHostKeyInfo {
    pub host: String,
    pub port: u16,
    pub key_type: String,
    pub key_hex: String,
    pub trusted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct TrustedHostStore {
    #[serde(default)]
    hosts: BTreeMap<String, String>,
}

fn default_port() -> u16 {
    DEFAULT_SSH_PORT
}

impl SshConnection {
    pub fn from_target(target: &str) -> Self {
        let trimmed = target.trim();

        if let Ok(url) = reqwest::Url::parse(trimmed)
            && url.scheme() == "ssh"
        {
            return Self {
                host: url.host_str().unwrap_or("").to_string(),
                username: url.username().to_string(),
                port: url.port().unwrap_or(DEFAULT_SSH_PORT),
                password: None,
                private_key_path: None,
                private_key_passphrase: None,
                trusted_host_key: None,
            };
        }

        let (username, host) = trimmed
            .rsplit_once('@')
            .map(|(user, host)| (user.to_string(), host.to_string()))
            .unwrap_or_else(|| (String::new(), trimmed.to_string()));

        Self {
            host,
            username,
            port: DEFAULT_SSH_PORT,
            password: None,
            private_key_path: None,
            private_key_passphrase: None,
            trusted_host_key: None,
        }
    }

    pub fn target_label(&self) -> String {
        let user_host = if self.username.trim().is_empty() {
            self.host.clone()
        } else {
            format!("{}@{}", self.username.trim(), self.host.trim())
        };

        if self.port == DEFAULT_SSH_PORT {
            user_host
        } else {
            format!("ssh://{user_host}:{}", self.port)
        }
    }

    pub fn agent_url(&self, agent_port: u16) -> String {
        format!("https://{}:{agent_port}", self.host.trim())
    }

    fn username_or_default(&self) -> Result<String> {
        let username = self.username.trim();
        if !username.is_empty() {
            return Ok(username.to_string());
        }

        std::env::var("USER")
            .or_else(|_| std::env::var("USERNAME"))
            .map_err(|_| anyhow!("SSH username is required"))
    }

    fn connect_blocking(&self) -> Result<Session> {
        let addr = (self.host.trim(), self.port)
            .to_socket_addrs()
            .context("failed to resolve SSH host")?
            .next()
            .context("SSH host resolved to no addresses")?;

        let tcp = TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT)
            .with_context(|| format!("failed to connect to SSH host {}", self.host))?;
        let mut session = Session::new().context("failed to create SSH session")?;
        // Keep the socket blocking and let libssh2 own the timeout. Socket
        // SO_RCVTIMEO/SO_SNDTIMEO can make a partial USERAUTH exchange surface
        // as `Waiting for USERAUTH response` on Windows OpenSSH servers even
        // when the same key succeeds with the system OpenSSH client.
        session.set_timeout(CONNECT_TIMEOUT.as_millis() as u32);
        session.set_tcp_stream(tcp);
        session.handshake().context("SSH handshake failed")?;

        let actual_key = session
            .host_key()
            .map(|(key, _)| to_hex(key))
            .context("SSH server did not provide a host key")?;
        match self.trusted_host_key.as_deref() {
            Some(expected) if constant_time_eq(expected, &actual_key) => {}
            Some(_) => {
                return Err(anyhow!(
                    "SSH host key changed for {}. Review and re-trust this host before continuing.",
                    self.host
                ));
            }
            None => {
                return Err(anyhow!(
                    "SSH host key is not trusted for {}. Use Guided SSH Setup to review and trust the fingerprint.",
                    self.host
                ));
            }
        }

        let username = self.username_or_default()?;

        if let Some(password) = self.password.as_deref().filter(|value| !value.is_empty()) {
            session
                .userauth_password(&username, password)
                .context("SSH password authentication failed")?;
        } else if let Some(key_path) = self
            .private_key_path
            .as_deref()
            .filter(|value| !value.is_empty())
        {
            session
                .userauth_pubkey_file(
                    &username,
                    None,
                    Path::new(key_path),
                    self.private_key_passphrase
                        .as_deref()
                        .filter(|value| !value.is_empty()),
                )
                .context("SSH private key authentication failed")?;
        } else {
            // Try default key paths before falling back to agent
            if let Ok(home) = std::env::var("HOME") {
                let default_keys = [
                    format!("{home}/.ssh/id_ed25519"),
                    format!("{home}/.ssh/id_rsa"),
                    format!("{home}/.ssh/id_ecdsa"),
                ];
                for key in &default_keys {
                    if Path::new(key).exists() {
                        if let Err(e) =
                            session.userauth_pubkey_file(&username, None, Path::new(key), None)
                        {
                            eprintln!(
                                "[ssh] Key auth failed for {} (code {:?}): {}",
                                key,
                                e.code(),
                                e
                            );
                            continue;
                        }
                        if session.authenticated() {
                            return Ok(session);
                        }
                    }
                }
            }
            // Match OpenSSH's normal behavior by trying the running agent first.
            // This matters for Windows OpenSSH hosts: libssh2 can reject a valid
            // Ed25519 key-file exchange while the same key succeeds through the
            // agent.
            if session.userauth_agent(&username).is_ok() && session.authenticated() {
                return Ok(session);
            }

            // Fall back to SSH agent after trying default key paths.
            session
                .userauth_agent(&username)
                .context("SSH agent authentication failed")?;
        }

        if !session.authenticated() {
            return Err(anyhow!("SSH authentication failed"));
        }

        Ok(session)
    }
}

pub async fn scan_host_key(
    connection: SshConnection,
    trusted_hosts_file: PathBuf,
) -> Result<SshHostKeyInfo> {
    tokio::task::spawn_blocking(move || {
        let (key_type, key_hex) = scan_host_key_blocking(&connection)?;
        let trusted_key = load_trusted_host_key(&trusted_hosts_file, &connection)?;

        Ok(SshHostKeyInfo {
            host: connection.host.trim().to_string(),
            port: connection.port,
            key_type,
            trusted: trusted_key
                .as_deref()
                .is_some_and(|trusted| constant_time_eq(trusted, &key_hex)),
            key_hex,
        })
    })
    .await
    .context("SSH host-key scan task failed")?
}

pub fn with_trusted_host_key(
    mut connection: SshConnection,
    trusted_hosts_file: &Path,
) -> Result<SshConnection> {
    connection.trusted_host_key = load_trusted_host_key(trusted_hosts_file, &connection)?;
    Ok(connection)
}

pub fn trust_host_key(
    trusted_hosts_file: &Path,
    connection: &SshConnection,
    key_hex: &str,
) -> Result<()> {
    if key_hex.trim().is_empty() {
        return Err(anyhow!("host key fingerprint is required"));
    }

    let mut store = load_store(trusted_hosts_file)?;
    store.hosts.insert(
        trust_key_for(connection),
        key_hex.trim().to_ascii_lowercase(),
    );
    save_store(trusted_hosts_file, &store)
}

pub async fn exec(connection: SshConnection, command: String) -> Result<SshCommandOutput> {
    tokio::task::spawn_blocking(move || exec_blocking(&connection, &command))
        .await
        .context("SSH command task failed")?
}

pub async fn copy_to_remote(
    connection: SshConnection,
    local_path: String,
    remote_path: String,
    mode: i32,
) -> Result<()> {
    tokio::task::spawn_blocking(move || {
        let session = connection.connect_blocking()?;
        let bytes = std::fs::read(&local_path)
            .with_context(|| format!("failed to read local file {local_path}"))?;
        copy_bytes_via_sftp(&session, &bytes, &remote_path, mode)
            .with_context(|| format!("failed to copy to remote {remote_path}"))
    })
    .await
    .context("SSH copy task failed")?
}

/// Upload bytes to a remote path using the SFTP subsystem.
///
/// SFTP is used instead of SCP because libssh2's `scp_send` uses
/// LIBSSH2_ERROR_SCP_PROTOCOL (-28) when Windows OpenSSH rejects paths that
/// contain backslashes or drive-letter colons in the SCP C-command filename
/// field. SFTP avoids the SCP handshake entirely.
///
/// Windows OpenSSH SFTP uses `/C:/...` absolute paths (forward slashes,
/// leading slash before the drive letter). Unix SFTP uses plain `/...` paths.
fn copy_bytes_via_sftp(
    session: &ssh2::Session,
    bytes: &[u8],
    remote_path: &str,
    mode: i32,
) -> Result<()> {
    // Convert the path to SFTP format.
    //   Windows backslash paths  →  C:\Users\nick\...  →  /C:/Users/nick/...
    //   Unix paths               →  /home/user/...     →  /home/user/...  (unchanged)
    let sftp_path = to_sftp_path(remote_path);

    let sftp = session.sftp().context("failed to open SFTP subsystem")?;

    let mut file = sftp
        .open_mode(
            Path::new(&sftp_path),
            ssh2::OpenFlags::WRITE | ssh2::OpenFlags::CREATE | ssh2::OpenFlags::TRUNCATE,
            mode,
            ssh2::OpenType::File,
        )
        .with_context(|| format!("SFTP open failed for {sftp_path}"))?;

    file.write_all(bytes)
        .context("failed to write SFTP payload")?;

    Ok(())
}

/// Convert an OS-specific file path to the absolute path format expected by
/// Windows OpenSSH SFTP (`/C:/Users/...`) or leave Unix paths unchanged.
fn to_sftp_path(remote_path: &str) -> String {
    // Normalise separators first.
    let forward = remote_path.replace('\\', "/");

    // A Windows absolute path starts with a drive letter followed by ':'.
    // Windows OpenSSH SFTP expects a leading '/' before the drive letter.
    //   C:/Users/nick/... → /C:/Users/nick/...
    let is_windows_abs = forward.len() >= 2
        && forward.as_bytes()[0].is_ascii_alphabetic()
        && forward.as_bytes()[1] == b':';

    if is_windows_abs {
        format!("/{forward}")
    } else {
        forward
    }
}

fn exec_blocking(connection: &SshConnection, command: &str) -> Result<SshCommandOutput> {
    let session = connection.connect_blocking()?;
    let mut channel = session
        .channel_session()
        .context("failed to open SSH command channel")?;
    channel
        .exec(command)
        .with_context(|| format!("failed to execute remote command: {command}"))?;

    let mut stdout = String::new();
    channel.read_to_string(&mut stdout).ok();

    let mut stderr = String::new();
    channel.stderr().read_to_string(&mut stderr).ok();

    channel
        .wait_close()
        .context("failed to close SSH channel")?;
    let status = channel.exit_status().unwrap_or(255);

    Ok(SshCommandOutput {
        status,
        stdout,
        stderr,
    })
}

fn scan_host_key_blocking(connection: &SshConnection) -> Result<(String, String)> {
    let addr = (connection.host.trim(), connection.port)
        .to_socket_addrs()
        .context("failed to resolve SSH host")?
        .next()
        .context("SSH host resolved to no addresses")?;
    let tcp = TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT)
        .with_context(|| format!("failed to connect to SSH host {}", connection.host))?;
    let mut session = Session::new().context("failed to create SSH session")?;
    session.set_timeout(CONNECT_TIMEOUT.as_millis() as u32);
    session.set_tcp_stream(tcp);
    session.handshake().context("SSH handshake failed")?;
    let (key, key_type) = session
        .host_key()
        .context("SSH server did not provide a host key")?;

    Ok((format!("{key_type:?}"), to_hex(key)))
}

fn trust_key_for(connection: &SshConnection) -> String {
    format!(
        "{}:{}",
        connection.host.trim().to_ascii_lowercase(),
        connection.port
    )
}

fn load_trusted_host_key(path: &Path, connection: &SshConnection) -> Result<Option<String>> {
    Ok(load_store(path)?
        .hosts
        .get(&trust_key_for(connection))
        .cloned())
}

fn load_store(path: &Path) -> Result<TrustedHostStore> {
    if !path.exists() {
        return Ok(TrustedHostStore::default());
    }

    let contents = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&contents).unwrap_or_default())
}

fn save_store(path: &Path, store: &TrustedHostStore) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(store)?)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

fn constant_time_eq(left: &str, right: &str) -> bool {
    let left = left.trim().as_bytes();
    let right = right.trim().as_bytes();
    if left.len() != right.len() {
        return false;
    }

    left.iter()
        .zip(right.iter())
        .fold(0u8, |acc, (a, b)| acc | (a ^ b))
        == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_user_host_target() {
        let connection = SshConnection::from_target("user@example-host");

        assert_eq!(connection.username, "user");
        assert_eq!(connection.host, "example-host");
        assert_eq!(connection.port, 22);
        assert_eq!(connection.target_label(), "user@example-host");
    }

    #[test]
    fn parses_ssh_url_with_port() {
        let connection = SshConnection::from_target("ssh://user@example-host:2222");

        assert_eq!(connection.username, "user");
        assert_eq!(connection.host, "example-host");
        assert_eq!(connection.port, 2222);
        assert_eq!(connection.target_label(), "ssh://user@example-host:2222");
        assert_eq!(connection.agent_url(7779), "https://example-host:7779");
    }

    #[test]
    fn hex_encoding_is_stable() {
        assert_eq!(to_hex(&[0, 15, 16, 255]), "000f10ff");
    }
}
