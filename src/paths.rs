//! Central application-root and resource path authority.
//!
//! This module is pure path resolution. It never creates, moves, deletes, or
//! migrates anything; mutation belongs to the restartable migration phase.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::identity::{LEGACY_PRODUCT_SLUG, PRODUCT_SLUG};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppPaths {
    pub root: PathBuf,
}

// Startup installs the explicitly selected root here before any background
// task or subsystem resolves a default path.  Keeping this override in the
// central authority means consumers that cannot cheaply receive `AppConfig`
// (HF token helpers, certificate helpers, runtime stores, and updater tasks)
// still honor `--config-dir` and migration state.
static ACTIVE_ROOT: OnceLock<PathBuf> = OnceLock::new();

#[allow(dead_code)] // resource accessors are consumed as phases route consumers
impl AppPaths {
    pub fn from_root(root: PathBuf) -> Self {
        Self { root }
    }

    /// Resolve the current application root without touching the filesystem.
    /// The legacy default remains selected until the restartable migration
    /// phase installs explicit root-state selection. An explicit
    /// `--config-dir` always wins, including for migration probes.
    pub fn resolve_root(config_dir: Option<PathBuf>) -> PathBuf {
        config_dir.unwrap_or_else(Self::default_active_root)
    }

    /// Select the default root for consumers that do not receive `AppConfig`.
    /// A populated canonical root wins; otherwise retain the legacy root until
    /// the explicit migration flow creates the new root. This is read-only and
    /// never creates either directory.
    pub fn default_active_root() -> PathBuf {
        if let Some(root) = ACTIVE_ROOT.get() {
            return root.clone();
        }
        let canonical = Self::canonical_default_root();
        let canonical_populated = canonical.is_dir()
            && std::fs::read_dir(&canonical)
                .ok()
                .and_then(|mut entries| entries.next())
                .is_some();
        let legacy = Self::legacy_default_root();
        let legacy_populated = legacy.is_dir()
            && std::fs::read_dir(&legacy)
                .ok()
                .and_then(|mut entries| entries.next())
                .is_some();
        if canonical_populated || !legacy_populated {
            canonical
        } else {
            legacy
        }
    }

    pub fn set_active_root(root: &Path) {
        let _ = ACTIVE_ROOT.set(root.to_path_buf());
    }

    pub fn canonical_default_root() -> PathBuf {
        platform_root(PRODUCT_SLUG)
    }

    pub fn legacy_default_root() -> PathBuf {
        platform_root(LEGACY_PRODUCT_SLUG)
    }

    /// Legacy macOS Application Support root used by pre-2.0 certificate
    /// installs. Kept in the central path authority so compatibility probing
    /// does not concatenate a product slug in a consumer.
    pub fn legacy_macos_support_root() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| {
                dirs::home_dir()
                    .unwrap_or_default()
                    .join("Library/Application Support")
            })
            .join(LEGACY_PRODUCT_SLUG)
    }

    pub fn config_file(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    pub fn binaries_dir(&self) -> PathBuf {
        self.root.join("binaries")
    }

    pub fn bin_dir(&self) -> PathBuf {
        self.root.join("bin")
    }

    /// Resolve the managed llama.cpp server while the application-root
    /// migration remains restartable. A healthy binary in either the canonical
    /// or legacy root must remain usable by every runtime consumer, not only by
    /// the updater API.
    pub fn default_llama_server_path(&self) -> PathBuf {
        let binary_name = if cfg!(windows) {
            "llama-server.exe"
        } else {
            "llama-server"
        };
        let configured = self.bin_dir().join(binary_name);
        if configured.is_file() {
            return configured;
        }

        for root in [
            self.root.clone(),
            Self::canonical_default_root(),
            Self::legacy_default_root(),
        ] {
            let candidate = root.join("bin").join(binary_name);
            if candidate.is_file() {
                return candidate;
            }
        }

        configured
    }

    pub fn models_dir(&self) -> PathBuf {
        if self.root == Self::canonical_default_root()
            && let Ok(bytes) = std::fs::read(self.model_root_selection_file())
            && let Ok(selection) = serde_json::from_slice::<ModelRootSelection>(&bytes)
        {
            return match selection.choice {
                ModelRootChoice::KeepLegacy => selection.source,
                ModelRootChoice::MoveIntoFoundry => selection.destination,
            };
        }
        self.root.join("models")
    }

    fn model_root_selection_file(&self) -> PathBuf {
        self.root.join(".local-llm-foundry-model-root.json")
    }

    pub fn scripts_dir(&self) -> PathBuf {
        self.root.join("scripts")
    }

    pub fn certs_dir(&self) -> PathBuf {
        self.root.join("certs")
    }

    pub fn chat_templates_dir(&self) -> PathBuf {
        self.root.join("chat-templates")
    }

    pub fn logs_dir(&self) -> PathBuf {
        self.root.join("logs")
    }

    pub fn calibrations_dir(&self) -> PathBuf {
        self.root.join("calibrations")
    }

    pub fn calibration_index_file(&self) -> PathBuf {
        self.calibrations_dir().join("index.json")
    }

    pub fn calibration_jobs_dir(&self) -> PathBuf {
        self.calibrations_dir().join("jobs")
    }

    pub fn calibration_receipts_dir(&self) -> PathBuf {
        self.calibrations_dir().join("receipts")
    }

    pub fn calibration_apply_backups_dir(&self) -> PathBuf {
        self.calibrations_dir().join("apply-backups")
    }

    /// Sibling to `calibration_receipts_dir`: exact launch-observation
    /// evidence (architecture 12), not a tuning-sweep receipt.
    pub fn launch_evidence_dir(&self) -> PathBuf {
        self.calibrations_dir().join("launch-evidence")
    }

    pub fn presets_file(&self) -> PathBuf {
        self.config_file("presets.json")
    }

    pub fn templates_file(&self) -> PathBuf {
        self.config_file("templates.json")
    }

    pub fn gpu_env_file(&self) -> PathBuf {
        self.config_file("gpu-env.json")
    }

    pub fn ui_settings_file(&self) -> PathBuf {
        self.config_file("ui-settings.json")
    }

    pub fn auth_config_file(&self) -> PathBuf {
        self.config_file("auth-config.json")
    }

    pub fn sessions_file(&self) -> PathBuf {
        self.config_file("sessions.json")
    }

    pub fn ssh_known_hosts_file(&self) -> PathBuf {
        self.config_file("ssh-known-hosts.json")
    }

    pub fn lhm_disabled_file(&self) -> PathBuf {
        self.config_file("lhm-disabled.json")
    }

    pub fn encryption_key_file(&self) -> PathBuf {
        self.config_file("encryption-key")
    }

    pub fn api_token_file(&self) -> PathBuf {
        self.config_file("api-token")
    }

    pub fn db_admin_token_file(&self) -> PathBuf {
        self.config_file("db-admin-token")
    }

    pub fn chat_db_file(&self) -> PathBuf {
        self.config_file("chat.db")
    }
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum ModelRootChoice {
    KeepLegacy,
    MoveIntoFoundry,
}

#[derive(serde::Deserialize)]
struct ModelRootSelection {
    choice: ModelRootChoice,
    source: PathBuf,
    destination: PathBuf,
}

fn platform_root(slug: &str) -> PathBuf {
    #[cfg(windows)]
    {
        let appdata = dirs::config_dir()
            .unwrap_or_else(|| dirs::home_dir().unwrap_or_default().join(".config"));
        windows_root(&appdata, slug)
    }
    #[cfg(not(windows))]
    {
        unix_root(
            &dirs::home_dir().unwrap_or_else(|| PathBuf::from(".")),
            slug,
        )
    }
}

#[allow(dead_code)] // exercised by cross-platform path tests and Unix builds
pub fn unix_root(home: &Path, slug: &str) -> PathBuf {
    home.join(".config").join(slug)
}

#[allow(dead_code)] // exercised by cross-platform path tests and Windows builds
pub fn windows_root(appdata: &Path, slug: &str) -> PathBuf {
    appdata.join(slug)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_root_is_side_effect_free_and_exact() {
        let custom = PathBuf::from("/tmp/foundry-test-root");
        assert_eq!(AppPaths::resolve_root(Some(custom.clone())), custom);
    }

    #[test]
    fn derived_resources_stay_below_root() {
        let paths = AppPaths::from_root(PathBuf::from("/tmp/foundry-test-root"));
        for path in [
            paths.binaries_dir(),
            paths.bin_dir(),
            paths.models_dir(),
            paths.certs_dir(),
            paths.logs_dir(),
            paths.chat_db_file(),
        ] {
            assert!(
                path.starts_with(&paths.root),
                "{} escaped root",
                path.display()
            );
        }
    }

    #[test]
    fn canonical_and_legacy_slugs_are_distinct() {
        assert_ne!(
            AppPaths::canonical_default_root(),
            AppPaths::legacy_default_root()
        );
    }

    #[test]
    fn injectable_platform_inputs_match_contract() {
        let home = PathBuf::from("/Users/example");
        let appdata = PathBuf::from(r"C:\Users\example\AppData\Roaming");
        assert_eq!(
            unix_root(&home, PRODUCT_SLUG),
            PathBuf::from("/Users/example/.config/local-llm-foundry")
        );
        let windows = windows_root(&appdata, PRODUCT_SLUG);
        assert!(windows.starts_with(&appdata));
        assert!(windows.ends_with(PRODUCT_SLUG));
    }
}
