//! Bounded, estimate-class `llama-fit-params` integration.
//!
//! The parser deliberately consumes both output streams.  The compact device
//! split is printed on stdout while the full memory table, including host and
//! device totals, is printed on stderr.

use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, SystemTime};

const MAX_CAPTURE_BYTES: u64 = 256 * 1024;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(8);

/// The existing llama.cpp fit-target default, shared by placement search.
pub const DEFAULT_FIT_RESERVE_MIB: u64 = 1024;

/// One probe reading at a single `n_cpu_moe`. All figures are in MiB.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FitReading {
    pub n_cpu_moe: u32,
    pub device_total_mib: u64,
    pub host_total_mib: u64,
    pub model_mib: u64,
    pub context_mib: u64,
    pub compute_mib: u64,
}

/// Errors are intentionally actionable: callers can render the reason beside
/// a disabled automatic-fit control without mistaking a failed probe for zero
/// memory.
#[derive(Debug)]
pub enum FitProbeError {
    Unavailable(String),
    Io(std::io::Error),
    Timeout,
    NonZeroExit(i32),
    OutputTooLarge,
    Parse(String),
    Identity(String),
}

impl fmt::Display for FitProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable(message) => write!(f, "probe_unavailable: {message}"),
            Self::Io(error) => write!(f, "probe_io_error: {error}"),
            Self::Timeout => write!(f, "probe_timeout"),
            Self::NonZeroExit(code) => write!(f, "probe_exit_status: {code}"),
            Self::OutputTooLarge => write!(f, "probe_output_too_large"),
            Self::Parse(message) => write!(f, "probe_parse_error: {message}"),
            Self::Identity(message) => write!(f, "probe_identity_error: {message}"),
        }
    }
}

impl std::error::Error for FitProbeError {}

impl From<std::io::Error> for FitProbeError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

/// The only capability Phase 4c may depend on.
pub trait FitReader {
    fn read(&mut self, n_cpu_moe: u32) -> Result<FitReading, FitProbeError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CompactRow {
    name: String,
    model_mib: u64,
    context_mib: u64,
    compute_mib: u64,
}

fn parse_compact(stdout: &str) -> Result<Vec<CompactRow>, FitProbeError> {
    let mut rows = Vec::new();
    for line in stdout.lines() {
        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields.len() != 4 {
            continue;
        }
        let values = fields[1..]
            .iter()
            .map(|value| {
                value.parse::<u64>().map_err(|_| {
                    FitProbeError::Parse(format!("compact row has non-numeric value: {line}"))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        rows.push(CompactRow {
            name: fields[0].to_string(),
            model_mib: values[0],
            context_mib: values[1],
            compute_mib: values[2],
        });
    }
    if rows.is_empty() {
        return Err(FitProbeError::Parse(
            "stdout contained no compact memory rows".into(),
        ));
    }
    Ok(rows)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TableRow {
    name: String,
    total_mib: u64,
}

fn parse_table(stderr: &str) -> Result<Vec<TableRow>, FitProbeError> {
    let mut rows = Vec::new();
    for line in stderr.lines() {
        let Some(after_marker) = line.split_once("|   - ").map(|(_, value)| value) else {
            continue;
        };
        let Some((name, values)) = after_marker.split_once(" |") else {
            continue;
        };
        let Some(total) = values
            .trim()
            .split_once('=')
            .and_then(|(value, _)| value.trim().parse::<u64>().ok())
        else {
            return Err(FitProbeError::Parse(format!(
                "memory table row has no total: {line}"
            )));
        };
        rows.push(TableRow {
            name: name.trim().to_string(),
            total_mib: total,
        });
    }
    if rows.is_empty() {
        return Err(FitProbeError::Parse(
            "stderr contained no memory table rows".into(),
        ));
    }
    Ok(rows)
}

/// Parse the two streams emitted by one fit-probe invocation.
pub fn parse_outputs(
    n_cpu_moe: u32,
    stdout: &str,
    stderr: &str,
) -> Result<FitReading, FitProbeError> {
    let compact = parse_compact(stdout)?;
    let table = parse_table(stderr)?;
    let devices = compact
        .iter()
        .filter(|row| row.name != "Host")
        .collect::<Vec<_>>();
    if devices.is_empty() {
        return Err(FitProbeError::Parse(
            "compact output contained no device row".into(),
        ));
    }

    let is_device_row = |row: &TableRow| {
        devices.iter().any(|device| {
            row.name == device.name
                || row
                    .name
                    .strip_prefix(&device.name)
                    .is_some_and(|suffix| suffix.starts_with(" ("))
        })
    };
    let table_device_total_mib = table
        .iter()
        .filter(|row| is_device_row(row))
        .map(|row| row.total_mib)
        .sum::<u64>();
    if table_device_total_mib == 0 {
        return Err(FitProbeError::Parse(
            "memory table contained no device total".into(),
        ));
    }
    // The table's first number is the device capacity, not this process's
    // allocation. The compact component row is the allocation budget used by
    // fit-target decisions and intentionally does not depend on live `free`.
    let device_total_mib = devices
        .iter()
        .map(|device| {
            device
                .model_mib
                .saturating_add(device.context_mib)
                .saturating_add(device.compute_mib)
        })
        .sum();
    let host_total_mib = table
        .iter()
        .filter(|row| !is_device_row(row))
        .map(|row| row.total_mib)
        .sum::<u64>();
    Ok(FitReading {
        n_cpu_moe,
        device_total_mib,
        host_total_mib,
        model_mib: devices.iter().map(|device| device.model_mib).sum(),
        context_mib: devices.iter().map(|device| device.context_mib).sum(),
        compute_mib: devices.iter().map(|device| device.compute_mib).sum(),
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FitBinaryIdentity {
    pub canonical_path: PathBuf,
    pub sha256: String,
    pub modified: SystemTime,
    pub version_line: String,
}

fn sha256_file(path: &Path) -> Result<String, FitProbeError> {
    let mut file = std::fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn read_version_line(path: &Path) -> Result<String, FitProbeError> {
    let mut child = Command::new(path)
        .arg("--version")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| FitProbeError::Identity("--version stdout was not piped".into()))?;
    let stdout_thread = std::thread::spawn(move || limited_read(stdout));
    let deadline = std::time::Instant::now() + DEFAULT_TIMEOUT;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout_thread.join();
            return Err(FitProbeError::Identity("--version timed out".into()));
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let stdout = stdout_thread
        .join()
        .map_err(|_| FitProbeError::Identity("--version reader panicked".into()))??;
    if !status.success() {
        return Err(FitProbeError::Identity(
            "--version returned a non-zero status".into(),
        ));
    }
    let line = String::from_utf8_lossy(&stdout)
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(str::to_string)
        .ok_or_else(|| FitProbeError::Identity("missing version line".into()))?;
    Ok(line)
}

pub fn binary_identity(path: &Path) -> Result<FitBinaryIdentity, FitProbeError> {
    let canonical_path = path
        .canonicalize()
        .map_err(|error| FitProbeError::Identity(format!("cannot canonicalize probe: {error}")))?;
    let modified = std::fs::metadata(&canonical_path)?.modified()?;
    Ok(FitBinaryIdentity {
        sha256: sha256_file(&canonical_path)?,
        version_line: read_version_line(&canonical_path)?,
        canonical_path,
        modified,
    })
}

pub fn verify_identity(
    path: &Path,
    expected: &FitBinaryIdentity,
) -> Result<FitBinaryIdentity, FitProbeError> {
    let actual = binary_identity(path)?;
    if actual.canonical_path != expected.canonical_path {
        return Err(FitProbeError::Identity("canonical path changed".into()));
    }
    if actual.sha256 != expected.sha256 {
        return Err(FitProbeError::Identity("SHA-256 changed".into()));
    }
    if actual.modified != expected.modified {
        return Err(FitProbeError::Identity("mtime changed".into()));
    }
    if actual.version_line.is_empty() || expected.version_line.is_empty() {
        return Err(FitProbeError::Identity("version line is missing".into()));
    }
    if actual.version_line != expected.version_line {
        return Err(FitProbeError::Identity("version line changed".into()));
    }
    Ok(actual)
}

#[derive(Debug, Clone)]
pub struct FitProbeConfig {
    pub executable: PathBuf,
    pub model_path: PathBuf,
    /// Full-file digest of the selected local weights artifact.
    pub artifact_digest: String,
    pub context_size: u64,
    pub ctk: String,
    pub ctv: String,
    pub draft_ctk: Option<String>,
    pub draft_ctv: Option<String>,
    pub batch_size: u32,
    pub ubatch_size: u32,
    pub timeout: Duration,
}

impl FitProbeConfig {
    pub fn new(executable: PathBuf, model_path: PathBuf) -> Self {
        Self {
            executable,
            model_path,
            artifact_digest: String::new(),
            context_size: 0,
            ctk: "f16".into(),
            ctv: "f16".into(),
            draft_ctk: None,
            draft_ctv: None,
            batch_size: 0,
            ubatch_size: 0,
            timeout: DEFAULT_TIMEOUT,
        }
    }
}

pub struct ProcessFitReader {
    config: FitProbeConfig,
    identity: FitBinaryIdentity,
    cache: HashMap<FitProbeCacheKey, FitReading>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct FitProbeCacheKey {
    artifact_digest: String,
    model_path: PathBuf,
    context_size: u64,
    ctk: String,
    ctv: String,
    draft_ctk: Option<String>,
    draft_ctv: Option<String>,
    batch_size: u32,
    ubatch_size: u32,
    probe_sha256: String,
    n_cpu_moe: u32,
}

impl ProcessFitReader {
    pub fn new(config: FitProbeConfig) -> Result<Self, FitProbeError> {
        if !config.executable.is_file() {
            return Err(FitProbeError::Unavailable(
                "llama_fit_params_path is not a file".into(),
            ));
        }
        let identity = binary_identity(&config.executable)?;
        Ok(Self {
            config,
            identity,
            cache: HashMap::new(),
        })
    }

    pub fn identity(&self) -> &FitBinaryIdentity {
        &self.identity
    }

    fn cache_key(&self, n_cpu_moe: u32) -> FitProbeCacheKey {
        FitProbeCacheKey {
            artifact_digest: self.config.artifact_digest.clone(),
            model_path: self.config.model_path.clone(),
            context_size: self.config.context_size,
            ctk: self.config.ctk.clone(),
            ctv: self.config.ctv.clone(),
            draft_ctk: self.config.draft_ctk.clone(),
            draft_ctv: self.config.draft_ctv.clone(),
            batch_size: self.config.batch_size,
            ubatch_size: self.config.ubatch_size,
            probe_sha256: self.identity.sha256.clone(),
            n_cpu_moe,
        }
    }

    fn command(&self, n_cpu_moe: u32) -> Command {
        let mut command = Command::new(&self.config.executable);
        command
            .arg("--fit")
            .arg("off")
            .arg("-lm")
            .arg("none")
            .arg("-lv")
            .arg("4")
            .arg("-fitp")
            .arg("on")
            .arg("-m")
            .arg(&self.config.model_path)
            .arg("-c")
            .arg(self.config.context_size.to_string())
            .arg("-ctk")
            .arg(&self.config.ctk)
            .arg("-ctv")
            .arg(&self.config.ctv)
            .arg("-b")
            .arg(self.config.batch_size.to_string())
            .arg("-ub")
            .arg(self.config.ubatch_size.to_string())
            .arg("--n-cpu-moe")
            .arg(n_cpu_moe.to_string());
        if let Some(value) = &self.config.draft_ctk {
            command.arg("-ctkd").arg(value);
        }
        if let Some(value) = &self.config.draft_ctv {
            command.arg("-ctvd").arg(value);
        }
        command
    }

    fn run(&self, n_cpu_moe: u32) -> Result<FitReading, FitProbeError> {
        let mut command = self.command(n_cpu_moe);
        command.stdout(Stdio::piped()).stderr(Stdio::piped());
        let mut child = command.spawn()?;
        let stdout = child.stdout.take().ok_or_else(|| {
            FitProbeError::Io(std::io::Error::other("probe stdout was not piped"))
        })?;
        let stderr = child.stderr.take().ok_or_else(|| {
            FitProbeError::Io(std::io::Error::other("probe stderr was not piped"))
        })?;
        let stdout_thread = std::thread::spawn(move || limited_read(stdout));
        let stderr_thread = std::thread::spawn(move || limited_read(stderr));
        let deadline = std::time::Instant::now() + self.config.timeout;
        loop {
            if let Some(status) = child.try_wait()? {
                let stdout = stdout_thread.join().map_err(|_| {
                    FitProbeError::Io(std::io::Error::other("stdout reader panicked"))
                })??;
                let stderr = stderr_thread.join().map_err(|_| {
                    FitProbeError::Io(std::io::Error::other("stderr reader panicked"))
                })??;
                if !status.success() {
                    return Err(FitProbeError::NonZeroExit(status.code().unwrap_or(-1)));
                }
                return parse_outputs(
                    n_cpu_moe,
                    &String::from_utf8_lossy(&stdout),
                    &String::from_utf8_lossy(&stderr),
                );
            }
            if std::time::Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                let _ = stdout_thread.join();
                let _ = stderr_thread.join();
                return Err(FitProbeError::Timeout);
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

fn limited_read<R: Read>(mut reader: R) -> Result<Vec<u8>, FitProbeError> {
    let mut output = Vec::new();
    reader
        .by_ref()
        .take(MAX_CAPTURE_BYTES + 1)
        .read_to_end(&mut output)?;
    if output.len() as u64 > MAX_CAPTURE_BYTES {
        return Err(FitProbeError::OutputTooLarge);
    }
    Ok(output)
}

impl FitReader for ProcessFitReader {
    fn read(&mut self, n_cpu_moe: u32) -> Result<FitReading, FitProbeError> {
        verify_identity(&self.config.executable, &self.identity)?;
        let key = self.cache_key(n_cpu_moe);
        if let Some(reading) = self.cache.get(&key) {
            return Ok(reading.clone());
        }
        let reading = self.run(n_cpu_moe)?;
        self.cache.insert(key, reading.clone());
        Ok(reading)
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
struct FixtureDocument {
    fixtures: Vec<FixtureEntry>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct FixtureEntry {
    n_cpu_moe: Option<i32>,
    exit_code: i32,
    stdout_compact: String,
    stderr_table: String,
}

/// First-class fixture reader used when no probe binary is installed and by
/// deterministic intent/probe tests.
pub struct FixtureFitReader {
    readings: HashMap<u32, Result<FitReading, String>>,
    errors: Vec<String>,
}

impl FixtureFitReader {
    pub fn from_json(contents: &str) -> Result<Self, FitProbeError> {
        let document: FixtureDocument = serde_json::from_str(contents)
            .map_err(|error| FitProbeError::Parse(format!("fixture JSON: {error}")))?;
        let mut readings = HashMap::new();
        let mut errors = Vec::new();
        for fixture in document.fixtures {
            let fixture_n = fixture.n_cpu_moe.unwrap_or(0);
            if fixture_n < 0 {
                errors.push(format!(
                    "n_cpu_moe={fixture_n}: probe exited with status {}",
                    fixture.exit_code
                ));
                continue;
            }
            let n_cpu_moe = fixture_n as u32;
            let reading = if fixture.exit_code != 0 {
                let error = format!("probe exited with status {}", fixture.exit_code);
                errors.push(format!("n_cpu_moe={n_cpu_moe}: {error}"));
                Err(error)
            } else {
                let reading =
                    parse_outputs(n_cpu_moe, &fixture.stdout_compact, &fixture.stderr_table)
                        .map_err(|error| error.to_string());
                if let Err(error) = &reading {
                    errors.push(format!("n_cpu_moe={n_cpu_moe}: {error}"));
                }
                reading
            };
            readings.insert(n_cpu_moe, reading);
        }
        Ok(Self { readings, errors })
    }

    pub fn embedded() -> Result<Self, FitProbeError> {
        Self::from_json(include_str!(
            "../../docs/plans/evidence/preset-bundles/phase-0/fit-probe-output-fixtures.json"
        ))
    }

    /// Fixture failures are retained as disabled-with-reason evidence rather
    /// than being converted into a zero or discarded during loading.
    pub fn errors(&self) -> &[String] {
        &self.errors
    }
}

impl FitReader for FixtureFitReader {
    fn read(&mut self, n_cpu_moe: u32) -> Result<FitReading, FitProbeError> {
        self.readings
            .get(&n_cpu_moe)
            .ok_or_else(|| FitProbeError::Unavailable(format!("fixture has no n={n_cpu_moe}")))?
            .clone()
            .map_err(FitProbeError::Parse)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const METAL_STDOUT: &str = "MTL0 15196 5182 832\nHost 515 0 641\n";
    const METAL_STDERR: &str = "|   - MTL0 (Apple M5 Max) | 57344 = 57280 + (21212 = 15196 + 5182 + 832) + -21148 |\n|   - Host | 1156 = 515 + 0 + 641 |\n|   - CPU_REPACK | 5568 = 5568 + 0 + 0 |\n";
    const CUDA_STDOUT: &str = "CUDA0 13276 2140 677\nHost 8037 0 102\n";
    const CUDA_STDERR: &str = "|   - CUDA0 (RTX 5090) | 32579 = 29860 + (16094 = 13276 + 2140 + 677) + -13375 |\n|   - Host | 8139 = 8037 + 0 + 102 |\n";

    #[test]
    fn parses_metal_cpu_repack_into_host_total() {
        let reading = parse_outputs(12, METAL_STDOUT, METAL_STDERR).unwrap();
        assert_eq!(reading.device_total_mib, 15_196 + 5_182 + 832);
        assert_eq!(reading.host_total_mib, 1_156 + 5_568);
        assert_eq!(reading.model_mib, 15_196);
    }

    #[test]
    fn parses_cuda_host_growth_without_cpu_repack_row() {
        let reading = parse_outputs(16, CUDA_STDOUT, CUDA_STDERR).unwrap();
        assert_eq!(reading.device_total_mib, 13_276 + 2_140 + 677);
        assert_eq!(reading.host_total_mib, 8_139);
        assert_eq!(reading.context_mib, 2_140);
    }

    #[test]
    fn fixture_reader_parses_all_successful_points() {
        let contents = include_str!(
            "../../docs/plans/evidence/preset-bundles/phase-0/fit-probe-output-fixtures.json"
        );
        let document: FixtureDocument = serde_json::from_str(contents).unwrap();
        for fixture in document.fixtures {
            if fixture.exit_code == 0 && fixture.n_cpu_moe.is_some_and(|n| n >= 0) {
                let n = fixture.n_cpu_moe.unwrap() as u32;
                parse_outputs(n, &fixture.stdout_compact, &fixture.stderr_table)
                    .unwrap_or_else(|error| panic!("fixture n={n}: {error}"));
            }
        }
    }

    #[test]
    fn parse_failure_is_not_zero() {
        let mut reader = FixtureFitReader::embedded().unwrap();
        assert!(
            reader
                .errors()
                .iter()
                .any(|error| error.contains("n_cpu_moe=-5"))
        );
        let error = reader.read(1_000).unwrap_err();
        assert!(matches!(error, FitProbeError::Unavailable(_)));
    }

    #[test]
    fn compact_parse_failure_is_actionable() {
        let error = parse_outputs(0, "garbage", METAL_STDERR).unwrap_err();
        assert!(matches!(error, FitProbeError::Parse(_)));
    }

    #[cfg(unix)]
    fn executable_fixture(contents: &str) -> (tempfile::TempDir, PathBuf) {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("llama-fit-params");
        std::fs::write(&path, contents).unwrap();
        let mut permissions = std::fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&path, permissions).unwrap();
        (directory, path)
    }

    #[cfg(unix)]
    #[test]
    fn identity_rejects_sha_mtime_and_missing_version_distinctly() {
        let (_directory, path) =
            executable_fixture("#!/bin/sh\nprintf '%s\\n' 'llama-fit-params test 1'\n");
        let identity = binary_identity(&path).unwrap();

        let mut wrong_sha = identity.clone();
        wrong_sha.sha256 = "0".repeat(64);
        assert!(matches!(
            verify_identity(&path, &wrong_sha),
            Err(FitProbeError::Identity(message)) if message == "SHA-256 changed"
        ));

        let mut wrong_mtime = identity.clone();
        wrong_mtime.modified = SystemTime::UNIX_EPOCH;
        assert!(matches!(
            verify_identity(&path, &wrong_mtime),
            Err(FitProbeError::Identity(message)) if message == "mtime changed"
        ));

        let (_directory, missing_version_path) = executable_fixture("#!/bin/sh\nexit 0\n");
        assert!(matches!(
            binary_identity(&missing_version_path),
            Err(FitProbeError::Identity(message)) if message == "missing version line"
        ));
    }

    #[cfg(unix)]
    #[test]
    fn process_reader_times_out_and_caches_by_n() {
        let (_directory, path) = executable_fixture(
            "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then printf '%s\\n' 'llama-fit-params test 1'; exit 0; fi\nsleep 1\n",
        );
        let model_path = tempfile::tempdir().unwrap().path().join("model.gguf");
        let mut config = FitProbeConfig::new(path, model_path);
        config.timeout = Duration::from_millis(20);
        let mut reader = ProcessFitReader::new(config).unwrap();
        assert!(matches!(reader.read(0), Err(FitProbeError::Timeout)));
        assert!(reader.cache.is_empty());
    }
}
