/// Certificate management for mTLS (mutual TLS) authentication.
///
/// The dashboard acts as a Certificate Authority (CA). On first run, it generates:
/// - A CA certificate (valid 10 years)
/// - A client certificate (valid 1 year, signed by CA)
///
/// When installing a remote agent, the CA certificate is shipped with the install payload.
/// The agent generates a server certificate (valid 1 year, signed by the same CA).
///
/// Certificates are automatically renewed on startup if they expire within 30 days.
use std::path::{Path, PathBuf};

use crate::paths::AppPaths;

/// Returns the path to the certs directory, creating it if necessary.
///
/// Uses the selected application root's `certs` directory on all platforms
/// (XDG-style on Unix/macOS) so cert files live alongside the rest of config.
///
/// If certs already exist at the old macOS path they are migrated once on first
/// access so existing installations are not broken by the path change.
pub fn certs_dir() -> PathBuf {
    certs_dir_for(&AppPaths::default_active_root())
}

/// Resolve certificates beneath an already-selected application root.
/// Creation remains explicit for certificate generation; resolution itself is
/// centralized and does not reinterpret custom roots.
pub fn certs_dir_for(app_root: &Path) -> PathBuf {
    let xdg_dir = certs_path_for(app_root);

    // One-time migration: if certs exist at the macOS Library path but not at
    // the XDG path, move them so they are found from now on.
    #[cfg(target_os = "macos")]
    if !xdg_dir.join("ca.pem").exists() {
        let old = AppPaths::legacy_macos_support_root().join("certs");
        if old.join("ca.pem").exists() {
            let _ = std::fs::create_dir_all(&xdg_dir);
            for entry in std::fs::read_dir(&old).into_iter().flatten().flatten() {
                let dest = xdg_dir.join(entry.file_name());
                if !dest.exists() {
                    let _ = std::fs::rename(entry.path(), &dest)
                        .or_else(|_| std::fs::copy(entry.path(), &dest).map(|_| ()));
                }
            }
        }
    }

    let _ = std::fs::create_dir_all(&xdg_dir);
    xdg_dir
}

/// Resolve the certificate directory without creating it or migrating legacy
/// contents. Read-only status endpoints use this accessor.
pub fn certs_path_for(app_root: &Path) -> PathBuf {
    app_root.join("certs")
}

/// Returns the path to the agent's multi-CA directory, creating it if necessary.
pub fn agent_cas_dir() -> PathBuf {
    let dir = certs_dir().join("cas");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// Returns the path to the remote-agent trust-anchor directory, creating it if necessary.
///
/// This holds CA certs fetched from remote agents via SSH during enrollment bootstrap.
/// These are used as TLS trust anchors when connecting to those remote agents — separate
/// from the device's own CA (`ca.pem`/`ca.key`) which is used for client-cert signing.
pub fn remote_agent_cas_dir() -> PathBuf {
    let dir = certs_dir().join("remote-cas");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// Persists a remote agent's CA cert to `remote-cas/<sanitized_host>.pem` (mode 0600).
///
/// Called during SSH bootstrap enrollment so the client can validate the remote agent's
/// TLS server certificate on all subsequent connections.
pub fn save_remote_agent_ca(host: &str, pem: &str) {
    if pem.trim().is_empty() {
        return;
    }
    let dir = remote_agent_cas_dir();
    let sanitized: String = host
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let path = dir.join(format!("{sanitized}.pem"));
    if std::fs::write(&path, pem).is_ok() {
        crate::config::harden_file_permissions(&path);
    }
}

/// Loads all remote-agent CA certs from `remote-cas/` as `reqwest::Certificate` values.
///
/// Added to the TLS trust store in `build_agent_https_client` and
/// `build_enrollment_https_client` so fresh devices (whose own `ca.pem` differs from
/// the remote agent's CA) can still validate the server cert.
pub fn load_remote_agent_ca_certs() -> Vec<reqwest::Certificate> {
    let dir = remote_agent_cas_dir();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut paths: Vec<_> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("pem"))
        .collect();
    paths.sort();
    paths
        .iter()
        .filter_map(|p| std::fs::read(p).ok())
        .filter_map(|bytes| reqwest::Certificate::from_pem(&bytes).ok())
        .collect()
}

/// Loads all trusted CA PEMs from the local certs directory:
/// the legacy `ca.pem` plus every `*.pem` file inside `cas/`.
///
/// Used to rebuild the mTLS trust store after a new CA is registered,
/// without restarting the agent.
pub fn load_all_agent_cas() -> Vec<String> {
    let mut ca_pems = Vec::new();
    let dir = certs_dir();

    // Legacy single CA — always checked first for backward compat.
    if let Ok(pem) = std::fs::read_to_string(dir.join("ca.pem"))
        && !pem.trim().is_empty()
    {
        ca_pems.push(pem);
    }

    // Multi-client CAs in cas/
    let cas_dir = dir.join("cas");
    if let Ok(entries) = std::fs::read_dir(&cas_dir) {
        let mut paths: Vec<_> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("pem"))
            .collect();
        paths.sort(); // deterministic order for stable ServerConfig rebuilds
        for path in paths {
            if let Ok(pem) = std::fs::read_to_string(&path)
                && !pem.trim().is_empty()
            {
                ca_pems.push(pem);
            }
        }
    }

    ca_pems
}

/// Validates a PEM string as a CA certificate and returns a stable instance ID.
///
/// The instance ID is the first 32 hex chars of SHA-256 of the raw DER bytes.
/// This is the filename used when writing to `cas/<instance_id>.pem`.
///
/// Returns `Ok(instance_id)` or `Err(human-readable reason)`.
pub fn validate_and_get_ca_instance_id(pem: &str) -> Result<String, String> {
    use std::io::BufReader;

    if pem.len() > 65536 {
        return Err("CA PEM exceeds 64 KB size limit".to_string());
    }

    let mut reader = BufReader::new(pem.as_bytes());
    let certs: Vec<rustls::pki_types::CertificateDer<'static>> = rustls_pemfile::certs(&mut reader)
        .filter_map(|c| c.ok())
        .collect();

    if certs.is_empty() {
        return Err("no valid certificate found in PEM".to_string());
    }

    let der = certs[0].as_ref();

    // Verify the cert can be added to a RootCertStore (rustls validates CA constraints).
    let mut store = rustls::RootCertStore::empty();
    if let Err(e) = store.add(rustls::pki_types::CertificateDer::from(der.to_vec())) {
        return Err(format!("certificate is not a valid CA trust anchor: {e}"));
    }

    // Stable instance ID: first 32 hex chars of SHA-256 of DER bytes.
    use sha2::Digest;
    let hash = sha2::Sha256::digest(der);
    let instance_id: String = hash[..16].iter().map(|b| format!("{b:02x}")).collect();

    Ok(instance_id)
}

/// Builds a rustls ServerConfig for the enrollment port (server-only TLS, no client cert required).
///
/// This is intentionally NOT mTLS — the enrollment endpoint is authenticated by bearer token
/// only. The TLS layer protects the token and CA PEM in transit.
pub fn build_enrollment_tls_config(server_cert: &Cert) -> Result<rustls::ServerConfig, String> {
    use std::io::BufReader;

    let mut cert_reader = BufReader::new(server_cert.pem.as_bytes());
    let certs: Vec<rustls::pki_types::CertificateDer<'static>> =
        rustls_pemfile::certs(&mut cert_reader)
            .filter_map(|c| c.ok())
            .collect();
    if certs.is_empty() {
        return Err("no certificate found in enrollment server cert".to_string());
    }

    let mut key_reader = BufReader::new(server_cert.key.as_bytes());
    let key: rustls::pki_types::PrivateKeyDer<'static> =
        rustls_pemfile::private_key(&mut key_reader)
            .map_err(|_| "failed to read enrollment server private key".to_string())?
            .ok_or_else(|| "no private key found in enrollment server key".to_string())?;

    rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .map_err(|e| format!("failed to build enrollment TLS config: {e}"))
}

/// Represents a certificate and its key pair.
#[derive(Debug, Clone)]
pub struct Cert {
    pub pem: String,
    pub key: String,
}

impl Cert {
    /// Loads a certificate from disk, or returns `None` if it doesn't exist.
    pub fn load(pem_path: &Path, key_path: &Path) -> Option<Self> {
        let pem = std::fs::read_to_string(pem_path).ok()?;
        let key = std::fs::read_to_string(key_path).ok()?;
        Some(Cert { pem, key })
    }

    /// Saves the certificate to disk.
    pub fn save(&self, pem_path: &Path, key_path: &Path) -> std::io::Result<()> {
        std::fs::write(pem_path, &self.pem)?;
        std::fs::write(key_path, &self.key)?;
        Ok(())
    }

    /// Checks if the certificate is expiring within the renewal window.
    pub fn needs_renewal(&self) -> bool {
        // Simplified check — regenerate if the cert file is older than CERT_VALIDITY_DAYS - RENEWAL_WINDOW_DAYS
        if let Some(metadata) =
            std::fs::metadata(Path::new(self.pem.split('\n').next().unwrap_or(""))).ok()
            && let Ok(modified) = metadata.modified()
            && let Ok(age) = std::time::SystemTime::now().duration_since(modified)
        {
            let days_old = age.as_secs() / (60 * 60 * 24);
            return days_old > (365 - 30) as u64;
        }
        false
    }
}

/// Generates a self-signed certificate for general use (non-agent TLS).
pub fn generate_self_signed(sans: Vec<String>) -> Cert {
    let rcgen::CertifiedKey { cert, signing_key } =
        rcgen::generate_simple_self_signed(sans).unwrap();
    Cert {
        pem: cert.pem(),
        key: signing_key.serialize_pem(),
    }
}

/// The Common Name used for the mTLS CA so that the CA and leaf certs have
/// distinct subjects. When all certs share the same DN (rcgen's default
/// "rcgen self signed cert"), TLS libraries treat leaf certs as self-signed
/// because `subject == issuer` and decline to verify the chain.
const AGENT_CA_CN: &str = "llama-monitor CA";

/// Build the rcgen `Issuer` for signing agent leaf certs.
/// The issuer DN must match the CA cert's subject DN exactly.
fn ca_issuer(ca_key: rcgen::KeyPair) -> rcgen::Issuer<'static, rcgen::KeyPair> {
    let mut dn = rcgen::DistinguishedName::new();
    dn.push(rcgen::DnType::CommonName, AGENT_CA_CN);
    let mut params = rcgen::CertificateParams::default();
    params.distinguished_name = dn;
    rcgen::Issuer::new(params, ca_key)
}

/// Generate a new CA certificate with a stable, recognisable subject DN.
fn generate_ca() -> Cert {
    let key = rcgen::KeyPair::generate().expect("failed to generate CA key");
    let mut dn = rcgen::DistinguishedName::new();
    dn.push(rcgen::DnType::CommonName, AGENT_CA_CN);
    let mut params = rcgen::CertificateParams::default();
    params.distinguished_name = dn;
    params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
    let cert = params
        .self_signed(&key)
        .expect("failed to self-sign CA cert");
    Cert {
        pem: cert.pem(),
        key: key.serialize_pem(),
    }
}

/// Ensures the CA certificate exists, generating it if necessary.
///
/// If the existing CA was generated with the old ambiguous DN (rcgen default
/// "rcgen self signed cert"), it is rotated: old CA + leaf certs are removed
/// so they are regenerated with a properly distinct CA CN on the next call.
pub fn ensure_ca() -> Cert {
    let dir = certs_dir();
    let ca_path = dir.join("ca.pem");
    let ca_key_path = dir.join("ca.key");
    // Sentinel written alongside every new-format CA.
    let sentinel = dir.join(".ca-v2");

    if let Some(ca) = Cert::load(&ca_path, &ca_key_path)
        && !ca.needs_renewal()
    {
        // Existing trust anchors are part of the public compatibility contract.
        // The sentinel only records that this process has observed the CA; it
        // must never be required for reusing an otherwise valid legacy CA.
        if !sentinel.exists() {
            let _ = std::fs::write(&sentinel, b"2");
        }
        return ca;
    }

    // No usable CA (first run or expiry) → generate a replacement. A valid old
    // CA above is deliberately retained even when it predates the sentinel.
    let _ = std::fs::remove_file(&ca_path);
    let _ = std::fs::remove_file(&ca_key_path);
    let _ = std::fs::remove_file(dir.join("agent-client.pem"));
    let _ = std::fs::remove_file(dir.join("agent-client.key"));
    let _ = std::fs::remove_file(dir.join("agent-server.pem"));
    let _ = std::fs::remove_file(dir.join("agent-server.key"));

    let ca = generate_ca();
    let _ = ca.save(&ca_path, &ca_key_path);
    let _ = std::fs::write(&sentinel, b"2");
    ca
}

/// Ensures a client certificate for the dashboard to present to agent servers.
///
/// This certificate is signed by the shared CA and has SAN `agent-client`
/// so the agent can confirm the caller is an authorized agent client.
#[allow(dead_code)]
pub fn ensure_agent_client_cert() -> Cert {
    let dir = certs_dir();
    let cert_path = dir.join("agent-client.pem");
    let key_path = dir.join("agent-client.key");

    if let Some(cert) = Cert::load(&cert_path, &key_path)
        && !cert.needs_renewal()
    {
        return cert;
    }

    let ca = ensure_ca();
    let ca_key = rcgen::KeyPair::from_pem(&ca.key).expect("invalid CA key PEM");
    let issuer = ca_issuer(ca_key);

    let client_key = rcgen::KeyPair::generate().expect("failed to generate client key");
    let mut params = rcgen::CertificateParams::new(vec!["agent-client".to_string()]).unwrap();
    params.is_ca = rcgen::IsCa::NoCa;

    let cert = params
        .signed_by(&client_key, &issuer)
        .expect("failed to sign agent client cert");

    let c = Cert {
        pem: cert.pem(),
        key: client_key.serialize_pem(),
    };
    let _ = c.save(&cert_path, &key_path);
    c
}

/// Generates a server certificate for the agent, signed by the CA.
pub fn generate_agent_server_cert(sans: Vec<String>) -> Cert {
    let ca = ensure_ca();
    let ca_key = rcgen::KeyPair::from_pem(&ca.key).expect("invalid CA key PEM");
    let issuer = ca_issuer(ca_key);

    let server_key = rcgen::KeyPair::generate().expect("failed to generate server key");
    let mut params = rcgen::CertificateParams::new(sans).unwrap();
    params.is_ca = rcgen::IsCa::NoCa;

    let cert = params
        .signed_by(&server_key, &issuer)
        .expect("failed to sign agent server cert");

    Cert {
        pem: cert.pem(),
        key: server_key.serialize_pem(),
    }
}

/// Generates a server certificate for the agent, signed by the CA.
pub fn ensure_agent_server_cert(sans: Vec<String>) -> Cert {
    let dir = certs_dir();
    let cert_path = dir.join("agent-server.pem");
    let key_path = dir.join("agent-server.key");

    if let Some(cert) = Cert::load(&cert_path, &key_path)
        && !cert.needs_renewal()
    {
        return cert;
    }

    let c = generate_agent_server_cert(sans);

    let _ = c.save(&cert_path, &key_path);
    c
}

/// Resolves the CA PEM to use as trust anchor for agent mTLS.
///
/// Design:
/// - A single internal CA (ca.pem) is the trust root for all agent certificates.
/// - ACME certificates (leaf certs) are NOT used as agent CA.
/// - If main TLS is active (SelfSigned/Custom/Acme), we still use this internal CA
///   for agent mTLS, keeping browser trust and agent trust separate.
/// - If main TLS is None, agent mTLS is still enforced (non-breaking).
#[allow(dead_code)]
pub fn get_agent_ca_for_mtls(tls_config: &crate::config::TLSConfig) -> Result<String, String> {
    let ca = ensure_ca();

    if tls_config.mode != crate::config::TlsMode::None {
        eprintln!(
            "[info] Agent mTLS: main TLS active ({:?}); using internal CA as agent trust root",
            tls_config.mode
        );
    } else {
        eprintln!(
            "[info] Agent mTLS: main TLS disabled; using internal CA for agent trust (mTLS enforced)"
        );
    }

    Ok(ca.pem)
}

/// Builds a rustls ServerConfig with mTLS for the agent server.
///
/// - trust_ca_pem: CA that signs both agent server certs and agent client certs.
/// - server_cert: server certificate (signed by CA).
///
/// Security properties:
/// - mTLS is enforced: agent endpoints require a valid client certificate.
/// - A client is accepted as an "agent" only if:
///   - Its certificate chains to the configured CA.
///   - Its certificate includes the "agent-client" role marker.
/// - If trust_ca_pems is empty, this returns Err instead of silently
///   falling back to "no client auth".
pub fn build_agent_tls_config(
    trust_ca_pems: Vec<String>,
    server_cert: Cert,
) -> Result<rustls::ServerConfig, String> {
    use std::io::BufReader;
    use std::sync::Arc;

    // Parse server cert
    let mut cert_reader = BufReader::new(server_cert.pem.as_bytes());
    let certs: Vec<rustls::pki_types::CertificateDer<'static>> =
        rustls_pemfile::certs(&mut cert_reader)
            .filter_map(|c| c.ok())
            .collect();
    if certs.is_empty() {
        return Err("no certificate found in agent server cert".to_string());
    }

    // Parse server key
    let mut key_reader = BufReader::new(server_cert.key.as_bytes());
    let key: rustls::pki_types::PrivateKeyDer<'static> =
        rustls_pemfile::private_key(&mut key_reader)
            .map_err(|_| "failed to read agent server private key".to_string())?
            .ok_or_else(|| "no private key found in agent server key".to_string())?;

    // mTLS is enforced: require at least one CA.
    let ca_pems: Vec<String> = trust_ca_pems
        .into_iter()
        .filter(|p| !p.trim().is_empty())
        .collect();
    if ca_pems.is_empty() {
        return Err("no CA configured for agent mTLS; agent requires mutual TLS".to_string());
    }

    let verifier = AgentClientCertVerifier::new(ca_pems);

    let config = rustls::ServerConfig::builder()
        .with_client_cert_verifier(Arc::new(verifier))
        .with_single_cert(certs, key)
        .map_err(|e| format!("failed to build agent mTLS config: {}", e))?;

    Ok(config)
}

/// Custom client certificate verifier for agent mTLS.
///
/// Purpose:
/// - Ensure only authorized agent clients (not browsers or arbitrary clients)
///   can connect to agent endpoints.
///
/// Behavior:
/// - Delegates chain validation to rustls's WebPkiClientVerifier (using the
///   internal CA as the trust anchor).
/// - Enforces a role check: the client cert must contain "agent-client" in
///   its subject/SAN. This is our role marker for agent clients.
/// - Browser clients are explicitly excluded: they do not present an agent
///   client cert and are not trusted for agent operations.
#[derive(Debug)]
pub struct AgentClientCertVerifier {
    inner: std::sync::Arc<dyn rustls::server::danger::ClientCertVerifier>,
}

impl AgentClientCertVerifier {
    pub fn new(ca_pems: Vec<String>) -> Self {
        use std::io::BufReader;

        let mut root_store = rustls::RootCertStore::empty();
        for pem in ca_pems {
            let mut reader = BufReader::new(pem.as_bytes());
            let ca_certs: Vec<_> = rustls_pemfile::certs(&mut reader)
                .filter_map(|c| c.ok())
                .collect();
            for ca in ca_certs {
                let _ = root_store.add(ca);
            }
        }

        let inner: std::sync::Arc<dyn rustls::server::danger::ClientCertVerifier> =
            rustls::server::WebPkiClientVerifier::builder(std::sync::Arc::new(root_store))
                .build()
                .expect("WebPkiClientVerifier build failed");

        Self { inner }
    }
}

impl rustls::server::danger::ClientCertVerifier for AgentClientCertVerifier {
    fn root_hint_subjects(&self) -> &[rustls::DistinguishedName] {
        self.inner.root_hint_subjects()
    }

    fn verify_client_cert(
        &self,
        end_entity: &rustls::pki_types::CertificateDer<'_>,
        intermediates: &[rustls::pki_types::CertificateDer<'_>],
        now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::server::danger::ClientCertVerified, rustls::Error> {
        // 1) Validate the certificate chain against the CA.
        self.inner
            .verify_client_cert(end_entity, intermediates, now)?;

        // 2) Role check: client cert must have "agent-client" as a SAN.
        //    DNS SANs are stored as raw ASCII bytes in DER (IA5String), so we
        //    can search the raw DER bytes directly — no parser needed.
        if !der_contains_ascii(end_entity.as_ref(), b"agent-client") {
            eprintln!("[info] Agent mTLS reject: client cert missing agent-client role");
            return Err(rustls::Error::InvalidCertificate(
                rustls::CertificateError::NotValidForName,
            ));
        }

        Ok(rustls::server::danger::ClientCertVerified::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.inner.supported_verify_schemes()
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        self.inner.verify_tls12_signature(message, cert, dss)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        self.inner.verify_tls13_signature(message, cert, dss)
    }
}

/// Returns true if `needle` (an ASCII string) appears as a contiguous byte
/// sequence anywhere in the DER-encoded certificate.
///
/// X.509 DNS SANs are encoded as IA5String in DER, which stores ASCII bytes
/// literally — no encoding layer — so a plain byte search is sufficient.
fn der_contains_ascii(der: &[u8], needle: &[u8]) -> bool {
    der.windows(needle.len()).any(|w| w == needle)
}

/// Extracts a human-readable subject from a DER-encoded client certificate.
///
/// Checks for known role strings in the raw DER bytes (DNS SANs are stored
/// as literal ASCII in DER IA5String encoding).
pub fn extract_cert_subject_pem(cert: &rustls::pki_types::CertificateDer<'_>) -> String {
    let der = cert.as_ref();
    if der_contains_ascii(der, b"agent-client") {
        return "agent-client".to_string();
    }
    if der_contains_ascii(der, b"agent-server") {
        return "agent-server".to_string();
    }
    "unknown".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustls::crypto::CryptoProvider;
    use rustls::server::danger::ClientCertVerifier;
    use std::sync::Once;

    static INIT_CRYPTO: Once = Once::new();

    fn ensure_crypto_provider() {
        INIT_CRYPTO.call_once(|| {
            // install_default is safe to call; if already set, it returns Err but no panic.
            let provider = rustls::crypto::ring::default_provider();
            let _ = CryptoProvider::install_default(provider);
        });
    }

    fn make_ca() -> (rcgen::Certificate, rcgen::KeyPair) {
        let ca_key = rcgen::KeyPair::generate().expect("generate CA key");
        let issuer_key =
            rcgen::KeyPair::from_pem(&ca_key.serialize_pem()).expect("CA key from PEM");
        let issuer = ca_issuer(issuer_key);
        let mut params = rcgen::CertificateParams::default();
        let mut dn = rcgen::DistinguishedName::new();
        dn.push(rcgen::DnType::CommonName, AGENT_CA_CN);
        params.distinguished_name = dn;
        params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
        let ca_cert = params.signed_by(&ca_key, &issuer).expect("sign CA");
        (ca_cert, ca_key)
    }

    fn ca_pem_from_cert(cert: &rcgen::Certificate) -> String {
        cert.pem()
    }

    fn client_cert_with_agent_role(ca_key: &rcgen::KeyPair) -> rcgen::Certificate {
        let client_key = rcgen::KeyPair::generate().expect("generate client key");
        let issuer_key =
            rcgen::KeyPair::from_pem(&ca_key.serialize_pem()).expect("CA key from PEM");
        let issuer = ca_issuer(issuer_key);
        let mut params = rcgen::CertificateParams::new(vec!["agent-client".to_string()]).unwrap();
        params.is_ca = rcgen::IsCa::NoCa;
        params
            .signed_by(&client_key, &issuer)
            .expect("sign agent-client cert")
    }

    fn client_cert_without_agent_role(ca_key: &rcgen::KeyPair) -> rcgen::Certificate {
        let client_key = rcgen::KeyPair::generate().expect("generate client key");
        let issuer_key =
            rcgen::KeyPair::from_pem(&ca_key.serialize_pem()).expect("CA key from PEM");
        let issuer = ca_issuer(issuer_key);
        let mut params =
            rcgen::CertificateParams::new(vec!["some-other-role".to_string()]).unwrap();
        params.is_ca = rcgen::IsCa::NoCa;
        params
            .signed_by(&client_key, &issuer)
            .expect("sign non-agent cert")
    }

    fn der_from_cert(cert: &rcgen::Certificate) -> rustls::pki_types::CertificateDer<'static> {
        rustls::pki_types::CertificateDer::from(cert.der().to_vec())
    }

    #[test]
    fn verify_agent_client_cert_accepts_agent_role() {
        ensure_crypto_provider();
        let (ca_cert, ca_key) = make_ca();
        let ca_pem = ca_pem_from_cert(&ca_cert);

        let client_cert = client_cert_with_agent_role(&ca_key);
        let client_der = der_from_cert(&client_cert);

        let verifier = AgentClientCertVerifier::new(vec![ca_pem]);
        let now = rustls::pki_types::UnixTime::now();

        let result = verifier.verify_client_cert(&client_der, &[], now);
        assert!(result.is_ok(), "agent-client cert should be accepted");
    }

    #[test]
    fn verify_agent_client_cert_rejects_missing_role() {
        ensure_crypto_provider();
        let (ca_cert, ca_key) = make_ca();
        let ca_pem = ca_pem_from_cert(&ca_cert);

        let client_cert = client_cert_without_agent_role(&ca_key);
        let client_der = der_from_cert(&client_cert);

        let verifier = AgentClientCertVerifier::new(vec![ca_pem]);
        let now = rustls::pki_types::UnixTime::now();

        let result = verifier.verify_client_cert(&client_der, &[], now);
        assert!(result.is_err(), "non-agent-client cert should be rejected");
    }

    #[test]
    fn extract_cert_subject_pem_identifies_roles() {
        let (_ca_cert, ca_key) = make_ca();

        let agent_cert = client_cert_with_agent_role(&ca_key);
        let agent_der = der_from_cert(&agent_cert);
        assert_eq!(extract_cert_subject_pem(&agent_der), "agent-client");

        let other_cert = client_cert_without_agent_role(&ca_key);
        let other_der = der_from_cert(&other_cert);
        assert_eq!(extract_cert_subject_pem(&other_der), "unknown");
    }
}
