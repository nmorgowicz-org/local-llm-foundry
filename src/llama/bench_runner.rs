//! Offline `llama-bench` runner.
//!
//! Powers two capabilities surfaced in the Spawn Wizard / Preset Editor:
//! - **Depth sweep**: measure decode (tg) and prefill (pp) throughput at several
//!   context depths, exposing the long-context collapse that dominates agentic use.
//! - **Empirical `--n-cpu-moe` verify**: try a few offload values and report the
//!   fastest that actually runs, correcting the estimator's instant guess.
//!
//! All runs use `llama-bench -o json` so we parse structured output rather than
//! the human table. The binary is resolved as a sibling of the configured
//! `llama-server` (the llama.cpp release bundle ships both).

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

pub const BENCH_MAX_OUTPUT_BYTES: usize = 2 * 1024 * 1024;
/// Default repetitions for standalone benchmark helpers and Thorough runs.
pub const BENCH_REPETITIONS: u32 = 3;
/// Quick calibration intentionally samples each candidate once; finalists are
/// remeasured by the Balanced profile.
pub const QUICK_BENCH_REPETITIONS: u32 = 1;
/// Balanced calibration uses two samples for finalists, avoiding the old
/// three-sample cost on every candidate.
pub const BALANCED_BENCH_REPETITIONS: u32 = 2;

/// Telemetry is best-effort diagnostic context attached to a bench receipt,
/// not a measurement input. The underlying GPU-metrics probe (`mactop`) has
/// no timeout of its own and can take tens of seconds on some hosts (full
/// SMC/volume/IOReport enumeration on first sample) — bound it so a slow
/// probe never stalls a calibration trial.
const TELEMETRY_CAPTURE_TIMEOUT: Duration = Duration::from_millis(750);

static BENCH_LEASE: OnceLock<Arc<Semaphore>> = OnceLock::new();

fn bench_lease() -> Arc<Semaphore> {
    BENCH_LEASE
        .get_or_init(|| Arc::new(Semaphore::new(1)))
        .clone()
}

/// Reserve the one offline benchmark slot. Calibration, depth sweeps, and
/// other empirical probes share this lease so they cannot contend for GPU
/// memory or produce interleaved measurements.
pub fn try_acquire_bench_lease() -> Result<OwnedSemaphorePermit, String> {
    bench_lease()
        .try_acquire_owned()
        .map_err(|_| "another offline benchmark or calibration is already running".to_string())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchFailureKind {
    Launch,
    Timeout,
    Oom,
    NonZero,
    OutputLimit,
}

#[derive(Debug, Clone)]
pub struct BenchRunReceipt {
    pub stdout: String,
    pub stderr: String,
    pub wall_time: Duration,
    pub exit_code: Option<i32>,
    pub stdout_truncated: bool,
    pub stderr_truncated: bool,
    pub failure: Option<BenchFailureKind>,
    pub tool_sha256: Option<String>,
    pub telemetry_before: BenchTelemetry,
    pub telemetry_after: BenchTelemetry,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct BenchTelemetry {
    pub available: bool,
    pub gpu_count: u32,
    pub vram_used_mib: u64,
    pub vram_total_mib: u64,
    pub max_temperature_c: Option<f32>,
}

fn capture_telemetry() -> BenchTelemetry {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let backend = crate::gpu::detect_backend("auto");
        let _ = tx.send(backend.read_metrics());
    });
    let Ok(Ok(metrics)) = rx.recv_timeout(TELEMETRY_CAPTURE_TIMEOUT) else {
        return BenchTelemetry::default();
    };
    if metrics.is_empty() {
        return BenchTelemetry::default();
    }
    BenchTelemetry {
        available: true,
        gpu_count: metrics.len() as u32,
        vram_used_mib: metrics.values().map(|metric| metric.vram_used).sum(),
        vram_total_mib: metrics.values().map(|metric| metric.vram_total).sum(),
        max_temperature_c: metrics
            .values()
            .map(|metric| metric.temp)
            .filter(|temperature| temperature.is_finite() && *temperature > 0.0)
            .max_by(f32::total_cmp),
    }
}

async fn read_bounded<R: AsyncRead + Unpin>(mut reader: R) -> Result<(Vec<u8>, bool), String> {
    let mut output = Vec::new();
    let mut buffer = [0u8; 8192];
    let mut truncated = false;
    loop {
        let count = reader
            .read(&mut buffer)
            .await
            .map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        if output.len() < BENCH_MAX_OUTPUT_BYTES {
            let keep = count.min(BENCH_MAX_OUTPUT_BYTES - output.len());
            output.extend_from_slice(&buffer[..keep]);
            truncated |= keep < count;
        } else {
            truncated = true;
        }
    }
    Ok((output, truncated))
}

/// One measured point in a depth sweep.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SweepPoint {
    /// Context tokens already in the cache when the measurement was taken.
    pub depth: u64,
    /// Prefill throughput (tokens/s) at this depth (0.0 if not measured).
    pub pp_tps: f64,
    /// Decode/generation throughput (tokens/s) at this depth (0.0 if not measured).
    pub tg_tps: f64,
    /// Per-test standard deviation when emitted by the benchmark build.
    #[serde(default)]
    pub pp_stddev: f64,
    #[serde(default)]
    pub tg_stddev: f64,
    #[serde(default)]
    pub repetitions: u32,
    #[serde(default)]
    pub pp_samples: Vec<f64>,
    #[serde(default)]
    pub tg_samples: Vec<f64>,
}

/// Result of an empirical `--n-cpu-moe` verification sweep.
#[derive(Debug, Clone, serde::Serialize)]
pub struct NcpuMoeProbe {
    pub n_cpu_moe: i32,
    /// Short-context decode throughput (tokens/s); 0.0 means it failed to run/fit.
    pub tg_tps: f64,
}

/// Resolve the `llama-bench` binary that ships alongside `llama-server`.
pub fn llama_bench_path(server_path: &Path) -> PathBuf {
    crate::inference::llama_cpp_tools::sibling_tool_path(
        server_path,
        crate::inference::llama_cpp_tools::LlamaCppTool::Bench,
    )
}

/// Resolve the optional predictive-fit helper shipped by some llama.cpp
/// bundles. Its absence degrades predictive pruning only.
pub fn llama_fit_params_path(server_path: &Path) -> PathBuf {
    crate::inference::llama_cpp_tools::sibling_tool_path(
        server_path,
        crate::inference::llama_cpp_tools::LlamaCppTool::FitParams,
    )
}

fn fa_flag(flash_attn: &str) -> &'static str {
    match flash_attn.trim().to_ascii_lowercase().as_str() {
        "on" | "1" | "true" => "1",
        "off" | "0" | "false" => "0",
        _ => "auto",
    }
}

/// Parse a `llama-bench -o json` array into depth-keyed points.
fn parse_sweep_json(stdout: &str) -> Result<Vec<SweepPoint>, String> {
    let arr: serde_json::Value = serde_json::from_str(stdout.trim())
        .map_err(|e| format!("Failed to parse llama-bench JSON: {e}"))?;
    let rows = arr
        .as_array()
        .ok_or_else(|| "llama-bench JSON was not an array".to_string())?;

    use std::collections::BTreeMap;
    let mut by_depth: BTreeMap<u64, SweepPoint> = BTreeMap::new();

    for row in rows {
        // llama-bench emits numbers as JSON numbers; tolerate string forms too.
        let num = |k: &str| -> Result<f64, String> {
            let value = row
                .get(k)
                .and_then(|v| {
                    v.as_f64()
                        .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
                })
                .ok_or_else(|| format!("llama-bench row missing numeric field {k}"))?;
            if !value.is_finite() || value < 0.0 {
                return Err(format!("llama-bench row has invalid numeric field {k}"));
            }
            Ok(value)
        };
        let depth = num("n_depth")? as u64;
        let n_gen = num("n_gen")? as u64;
        let n_prompt = num("n_prompt")? as u64;
        let avg_ts = num("avg_ts")?;
        let stddev = row
            .get("stddev")
            .or_else(|| row.get("std_ts"))
            .and_then(|v| {
                v.as_f64()
                    .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
            })
            .unwrap_or(0.0);
        if !stddev.is_finite() || stddev < 0.0 {
            return Err("llama-bench row has invalid standard deviation".into());
        }
        let repetitions = row
            .get("n_rep")
            .or_else(|| row.get("repetitions"))
            .and_then(|v| {
                v.as_u64()
                    .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
            })
            .unwrap_or(BENCH_REPETITIONS as u64) as u32;

        let entry = by_depth.entry(depth).or_insert(SweepPoint {
            depth,
            pp_tps: 0.0,
            tg_tps: 0.0,
            pp_stddev: 0.0,
            tg_stddev: 0.0,
            repetitions,
            pp_samples: Vec::new(),
            tg_samples: Vec::new(),
        });
        if n_gen > 0 {
            entry.tg_tps = avg_ts;
            entry.tg_stddev = stddev;
            entry.repetitions = repetitions;
            entry.tg_samples = sample_values(row, "samples_ts")?;
        } else if n_prompt > 0 {
            entry.pp_tps = avg_ts;
            entry.pp_stddev = stddev;
            entry.repetitions = repetitions;
            entry.pp_samples = sample_values(row, "samples_ts")?;
        }
    }

    Ok(by_depth.into_values().collect())
}

fn sample_values(row: &serde_json::Value, key: &str) -> Result<Vec<f64>, String> {
    let Some(values) = row.get(key) else {
        return Ok(Vec::new());
    };
    let Some(values) = values.as_array() else {
        return Err(format!("llama-bench field {key} must be an array"));
    };
    values
        .iter()
        .map(|value| {
            let number = value
                .as_f64()
                .or_else(|| value.as_str().and_then(|text| text.parse().ok()))
                .ok_or_else(|| format!("llama-bench field {key} has a non-numeric sample"))?;
            if !number.is_finite() || number < 0.0 {
                return Err(format!("llama-bench field {key} has an invalid sample"));
            }
            Ok(number)
        })
        .collect()
}

/// Reject measurements whose reported throughput is physically incompatible
/// with the independent wall clock. The wide tolerance accommodates startup,
/// model loading, and telemetry overhead while still catching impossible
/// `NaN`/zero/garbage rows before they can become winners.
pub fn validate_wall_clock_plausibility(
    points: &[SweepPoint],
    prompt_tokens: u32,
    generation_tokens: u32,
    wall_time: Duration,
) -> Result<(), String> {
    if points.is_empty() || wall_time.is_zero() {
        return Err("benchmark produced no measurable points".into());
    }
    let wall_seconds = wall_time.as_secs_f64();
    for point in points {
        let prompt_rate = point.pp_tps;
        let generation_rate = point.tg_tps;
        if !prompt_rate.is_finite()
            || !generation_rate.is_finite()
            || prompt_rate <= 0.0
            || generation_rate <= 0.0
        {
            return Err(format!(
                "benchmark point at depth {} is not measurable",
                point.depth
            ));
        }
        let estimated_seconds =
            f64::from(prompt_tokens) / prompt_rate + f64::from(generation_tokens) / generation_rate;
        let lower = (wall_seconds / 100.0).max(0.000_001);
        let upper = wall_seconds * 100.0 + 1.0;
        if estimated_seconds < lower || estimated_seconds > upper {
            return Err(format!(
                "benchmark point at depth {} is implausible against wall clock",
                point.depth
            ));
        }
    }
    Ok(())
}

/// Build the base llama-bench argument vector shared by sweeps and probes.
#[allow(clippy::too_many_arguments)]
fn base_args(
    model_path: &str,
    ngl: i32,
    flash_attn: &str,
    ctk: &str,
    ctv: &str,
    batch_size: u32,
    ubatch_size: u32,
    n_cpu_moe: Option<i32>,
) -> Vec<String> {
    base_args_with_repetitions(
        model_path,
        ngl,
        flash_attn,
        ctk,
        ctv,
        batch_size,
        ubatch_size,
        n_cpu_moe,
        BENCH_REPETITIONS,
    )
}

#[allow(clippy::too_many_arguments)]
fn base_args_with_repetitions(
    model_path: &str,
    ngl: i32,
    flash_attn: &str,
    ctk: &str,
    ctv: &str,
    batch_size: u32,
    ubatch_size: u32,
    n_cpu_moe: Option<i32>,
    repetitions: u32,
) -> Vec<String> {
    let mut args: Vec<String> = vec![
        "-m".into(),
        model_path.into(),
        "-ngl".into(),
        ngl.to_string(),
        "-fa".into(),
        fa_flag(flash_attn).into(),
        "-ctk".into(),
        ctk.into(),
        "-ctv".into(),
        ctv.into(),
        "-b".into(),
        batch_size.to_string(),
        "-ub".into(),
        ubatch_size.to_string(),
        "-o".into(),
        "json".into(),
        "-r".into(),
        repetitions.to_string(),
    ];
    if let Some(n) = n_cpu_moe
        && n > 0
    {
        args.push("--n-cpu-moe".into());
        args.push(n.to_string());
    }
    args
}

async fn run_bench(
    bench_bin: &Path,
    cwd: &Path,
    args: &[String],
    timeout: Duration,
) -> Result<String, String> {
    if bench_bin.components().count() > 1 && !bench_bin.exists() {
        return Err(format!(
            "llama-bench not found at {}. It ships with the llama.cpp release alongside llama-server.",
            bench_bin.display()
        ));
    }

    let receipt = run_bench_receipt(bench_bin, cwd, args, timeout).await?;
    if let Some(failure) = receipt.failure {
        let tail: String = receipt
            .stderr
            .lines()
            .rev()
            .take(5)
            .collect::<Vec<_>>()
            .join(" | ");
        return Err(match failure {
            BenchFailureKind::Timeout => "llama-bench timed out".to_string(),
            BenchFailureKind::OutputLimit => {
                "llama-bench output exceeded the safety limit".to_string()
            }
            _ if tail.is_empty() => format!("llama-bench failed ({failure:?})"),
            _ => format!("llama-bench failed ({failure:?}): {tail}"),
        });
    }
    Ok(receipt.stdout)
}

/// Run a managed benchmark with bounded stdout/stderr and a structured
/// failure receipt. The existing sweep helpers retain their string API while
/// callers migrate to this safer primitive.
pub async fn run_bench_receipt(
    bench_bin: &Path,
    cwd: &Path,
    args: &[String],
    timeout: Duration,
) -> Result<BenchRunReceipt, String> {
    run_bench_receipt_with_deadline(bench_bin, cwd, args, Some(timeout)).await
}

/// Run a calibration-owned benchmark without an arbitrary wall-clock kill.
/// The executor still owns cancellation and `kill_on_drop` cleanup, while a
/// legitimate long-context trial is allowed to finish and emit JSON.
pub async fn run_bench_receipt_unbounded(
    bench_bin: &Path,
    cwd: &Path,
    args: &[String],
) -> Result<BenchRunReceipt, String> {
    run_bench_receipt_with_deadline(bench_bin, cwd, args, None).await
}

async fn run_bench_receipt_with_deadline(
    bench_bin: &Path,
    cwd: &Path,
    args: &[String],
    deadline: Option<Duration>,
) -> Result<BenchRunReceipt, String> {
    let _lease = bench_lease()
        .acquire_owned()
        .await
        .map_err(|_| "offline benchmark lease is unavailable".to_string())?;
    if !bench_bin.is_file() {
        return Err(format!(
            "llama-bench not found at {}. It ships with the llama.cpp release alongside llama-server.",
            bench_bin.display()
        ));
    }
    let telemetry_before = capture_telemetry();
    let tool_sha256 =
        crate::inference::llama_cpp_capabilities::ExecutableIdentity::from_path(bench_bin)
            .ok()
            .map(|identity| identity.file_hash);
    let started = Instant::now();
    let mut command = Command::new(bench_bin);
    command
        .current_dir(cwd)
        .args(args)
        .kill_on_drop(true)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    command.process_group(0);
    let mut child = command
        .spawn()
        .map_err(|error| format!("Failed to launch llama-bench: {error}"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "llama-bench stdout pipe unavailable".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "llama-bench stderr pipe unavailable".to_string())?;
    let capture = async move {
        let (out_result, err_result) = tokio::join!(read_bounded(stdout), read_bounded(stderr));
        let (out, out_truncated) = out_result?;
        let (err, err_truncated) = err_result?;
        Ok::<_, String>((out, err, out_truncated, err_truncated))
    };
    let capture_result = match deadline {
        Some(timeout) => tokio::time::timeout(timeout, capture).await,
        None => Ok(capture.await),
    };
    let (stdout, stderr, stdout_truncated, stderr_truncated) = match capture_result {
        Ok(result) => result?,
        Err(_) => {
            #[cfg(unix)]
            if let Some(pid) = child.id() {
                // The child owns a dedicated process group, so descendants
                // are terminated without broad name-based process kills.
                unsafe {
                    libc::kill(-(pid as i32), libc::SIGKILL);
                }
            }
            let _ = child.kill().await;
            return Ok(BenchRunReceipt {
                stdout: String::new(),
                stderr: String::new(),
                wall_time: started.elapsed(),
                exit_code: None,
                stdout_truncated: false,
                stderr_truncated: false,
                failure: Some(BenchFailureKind::Timeout),
                tool_sha256,
                telemetry_before,
                telemetry_after: capture_telemetry(),
            });
        }
    };
    let status = child
        .wait()
        .await
        .map_err(|error| format!("Failed waiting for llama-bench: {error}"))?;
    let stdout_text = String::from_utf8_lossy(&stdout).into_owned();
    let stderr_text = String::from_utf8_lossy(&stderr).into_owned();
    let stderr_lower = stderr_text.to_ascii_lowercase();
    let failure = if stdout_truncated || stderr_truncated {
        Some(BenchFailureKind::OutputLimit)
    } else if status.success() {
        None
    } else if stderr_lower.contains("out of memory") || stderr_lower.contains("oom") {
        Some(BenchFailureKind::Oom)
    } else {
        Some(BenchFailureKind::NonZero)
    };
    Ok(BenchRunReceipt {
        stdout: stdout_text,
        stderr: stderr_text,
        wall_time: started.elapsed(),
        exit_code: status.code(),
        stdout_truncated,
        stderr_truncated,
        failure,
        tool_sha256,
        telemetry_before,
        telemetry_after: capture_telemetry(),
    })
}

/// Run a depth sweep: prefill (512) + decode (64) at each requested depth.
#[allow(clippy::too_many_arguments)]
pub async fn run_sweep(
    bench_bin: &Path,
    cwd: &Path,
    model_path: &str,
    ngl: i32,
    flash_attn: bool,
    ctk: &str,
    ctv: &str,
    batch_size: u32,
    ubatch_size: u32,
    depths: &[u64],
    n_cpu_moe: Option<i32>,
) -> Result<Vec<SweepPoint>, String> {
    let flash_mode = if flash_attn { "on" } else { "off" };
    run_sweep_with_tokens(
        bench_bin,
        cwd,
        model_path,
        ngl,
        flash_mode,
        ctk,
        ctv,
        batch_size,
        ubatch_size,
        depths,
        n_cpu_moe,
        512,
        64,
    )
    .await
}

/// Run a bounded depth sweep with explicit prompt and generation lengths.
/// Calibration uses this form so the workload recorded in its receipt is the
/// workload actually measured.
#[allow(clippy::too_many_arguments)]
pub async fn run_sweep_with_tokens(
    bench_bin: &Path,
    cwd: &Path,
    model_path: &str,
    ngl: i32,
    flash_attn: &str,
    ctk: &str,
    ctv: &str,
    batch_size: u32,
    ubatch_size: u32,
    depths: &[u64],
    n_cpu_moe: Option<i32>,
    prompt_tokens: u32,
    generation_tokens: u32,
) -> Result<Vec<SweepPoint>, String> {
    run_sweep_with_tokens_repetitions(
        bench_bin,
        cwd,
        model_path,
        ngl,
        flash_attn,
        ctk,
        ctv,
        batch_size,
        ubatch_size,
        depths,
        n_cpu_moe,
        prompt_tokens,
        generation_tokens,
        BENCH_REPETITIONS,
    )
    .await
}

/// Run a token-shaped sweep with an explicit repetition profile.
/// Calibration uses this to screen once and only repeat finalists.
#[allow(clippy::too_many_arguments)]
pub async fn run_sweep_with_tokens_repetitions(
    bench_bin: &Path,
    cwd: &Path,
    model_path: &str,
    ngl: i32,
    flash_attn: &str,
    ctk: &str,
    ctv: &str,
    batch_size: u32,
    ubatch_size: u32,
    depths: &[u64],
    n_cpu_moe: Option<i32>,
    prompt_tokens: u32,
    generation_tokens: u32,
    repetitions: u32,
) -> Result<Vec<SweepPoint>, String> {
    if repetitions == 0 {
        return Err("Benchmark repetitions must be non-zero".into());
    }
    if depths.is_empty() {
        return Err("No depths requested".into());
    }
    let mut args = base_args_with_repetitions(
        model_path,
        ngl,
        flash_attn,
        ctk,
        ctv,
        batch_size,
        ubatch_size,
        n_cpu_moe,
        repetitions,
    );
    args.push("-p".into());
    args.push(prompt_tokens.to_string());
    args.push("-n".into());
    args.push(generation_tokens.to_string());
    args.push("-d".into());
    args.push(
        depths
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );

    // Each depth requires a prefill of that many tokens; scale the budget.
    // Long Thinking workloads spend most of their wall time in decode. The
    // previous depth-only timeout could kill a valid 8K generation before
    // llama-bench emitted JSON, which surfaced as a misleading EOF parse
    // error. Keep the short-workload behavior while reserving time for the
    // requested generation length.
    let receipt = run_bench_receipt_unbounded(bench_bin, cwd, &args).await?;
    let points = parse_sweep_json(&receipt.stdout)?;
    validate_wall_clock_plausibility(&points, prompt_tokens, generation_tokens, receipt.wall_time)?;
    Ok(points)
}

/// Empirically probe a set of `--n-cpu-moe` values (short-context decode only)
/// and return the throughput for each. The caller picks the fastest that ran.
#[allow(clippy::too_many_arguments)]
pub async fn probe_ncpumoe(
    bench_bin: &Path,
    cwd: &Path,
    model_path: &str,
    ngl: i32,
    flash_attn: bool,
    ctk: &str,
    ctv: &str,
    batch_size: u32,
    ubatch_size: u32,
    candidates: &[i32],
) -> Vec<NcpuMoeProbe> {
    let flash_mode = if flash_attn { "on" } else { "off" };
    let mut out = Vec::new();
    for &n in candidates {
        let mut args = base_args(
            model_path,
            ngl,
            flash_mode,
            ctk,
            ctv,
            batch_size,
            ubatch_size,
            Some(n),
        );
        args.push("-p".into());
        args.push("0".into());
        args.push("-n".into());
        args.push("64".into());
        let tg_tps = match run_bench(bench_bin, cwd, &args, Duration::from_secs(240)).await {
            Ok(stdout) => parse_sweep_json(&stdout)
                .ok()
                .and_then(|pts| pts.first().map(|p| p.tg_tps))
                .unwrap_or(0.0),
            Err(_) => 0.0, // failed to run/fit at this offload level
        };
        out.push(NcpuMoeProbe {
            n_cpu_moe: n,
            tg_tps,
        });
    }
    out
}

/// One measured point in a batch/ubatch sweep.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BatchProbe {
    pub batch_size: u32,
    pub ubatch_size: u32,
    /// Prefill throughput (tokens/s); 0.0 means the run failed or didn't fit.
    pub pp_tps: f64,
}

/// Probe a set of (batch_size, ubatch_size) pairs measuring PP throughput only
/// (no decode). `prompt_tokens` should be representative of the user's actual
/// prompt length — larger values expose batch-size effects more clearly.
#[allow(clippy::too_many_arguments)]
pub async fn probe_batch(
    bench_bin: &Path,
    cwd: &Path,
    model_path: &str,
    ngl: i32,
    flash_attn: bool,
    ctk: &str,
    ctv: &str,
    candidates: &[(u32, u32)],
    prompt_tokens: u32,
    n_cpu_moe: Option<i32>,
) -> Vec<BatchProbe> {
    let flash_mode = if flash_attn { "on" } else { "off" };
    let mut out = Vec::new();
    for &(batch, ubatch) in candidates {
        let mut args = base_args(
            model_path, ngl, flash_mode, ctk, ctv, batch, ubatch, n_cpu_moe,
        );
        args.push("-p".into());
        args.push(prompt_tokens.to_string());
        args.push("-n".into());
        args.push("0".into()); // PP only
        args.push("-r".into());
        args.push("2".into()); // 2 runs for stability without too much wall time
        let pp_tps = match run_bench(bench_bin, cwd, &args, Duration::from_secs(120)).await {
            Ok(stdout) => parse_sweep_json(&stdout)
                .ok()
                .and_then(|pts| pts.into_iter().find(|p| p.pp_tps > 0.0))
                .map(|p| p.pp_tps)
                .unwrap_or(0.0),
            Err(_) => 0.0,
        };
        out.push(BatchProbe {
            batch_size: batch,
            ubatch_size: ubatch,
            pp_tps,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_repetition_profiles_are_bounded() {
        assert_eq!(QUICK_BENCH_REPETITIONS, 1);
        assert_eq!(BALANCED_BENCH_REPETITIONS, 2);
        assert_eq!(BENCH_REPETITIONS, 3);
    }

    #[test]
    fn parses_pp_and_tg_by_depth() {
        let json = r#"[
          {"n_prompt":512,"n_gen":0,"n_depth":0,"avg_ts":1500.0},
          {"n_prompt":0,"n_gen":64,"n_depth":0,"avg_ts":50.0},
          {"n_prompt":0,"n_gen":64,"n_depth":32768,"avg_ts":30.0}
        ]"#;
        let pts = parse_sweep_json(json).unwrap();
        assert_eq!(pts.len(), 2);
        assert_eq!(pts[0].depth, 0);
        assert!((pts[0].pp_tps - 1500.0).abs() < 0.01);
        assert!((pts[0].tg_tps - 50.0).abs() < 0.01);
        assert!((pts[1].tg_tps - 30.0).abs() < 0.01);
    }

    #[test]
    fn parses_repetition_metadata_and_rejects_invalid_metrics() {
        let json = r#"[{"n_prompt":512,"n_gen":0,"n_depth":0,"avg_ts":1500.0,"stddev":2.5,"n_rep":3,"samples_ts":[1490.0,1500.0,1510.0]}]"#;
        let points = parse_sweep_json(json).expect("valid benchmark row");
        assert_eq!(points[0].repetitions, 3);
        assert!((points[0].pp_stddev - 2.5).abs() < f64::EPSILON);
        assert_eq!(points[0].pp_samples, [1490.0, 1500.0, 1510.0]);
        assert!(
            parse_sweep_json(r#"[{"n_prompt":1,"n_gen":0,"n_depth":0,"avg_ts":"NaN"}]"#).is_err()
        );
        assert!(parse_sweep_json(r#"[{"n_prompt":1,"n_gen":0,"n_depth":0}]"#).is_err());
    }

    #[test]
    fn wall_clock_plausibility_rejects_impossible_throughput() {
        let points = vec![SweepPoint {
            depth: 0,
            pp_tps: 100.0,
            tg_tps: 100.0,
            pp_stddev: 0.0,
            tg_stddev: 0.0,
            repetitions: 3,
            pp_samples: Vec::new(),
            tg_samples: Vec::new(),
        }];
        assert!(
            validate_wall_clock_plausibility(&points, 100, 100, Duration::from_secs(2)).is_ok()
        );
        assert!(
            validate_wall_clock_plausibility(&points, 100, 100, Duration::from_millis(1)).is_err()
        );
    }

    #[test]
    fn bench_path_is_sibling() {
        let p = llama_bench_path(Path::new("/opt/llama/bin/llama-server"));
        assert!(p.ends_with(if cfg!(windows) {
            "llama-bench.exe"
        } else {
            "llama-bench"
        }));
    }

    #[tokio::test]
    async fn bounded_receipt_classifies_missing_binary() {
        let result = run_bench_receipt(
            Path::new("/definitely/missing/llama-bench"),
            Path::new("."),
            &[],
            Duration::from_millis(20),
        )
        .await;
        assert!(result.is_err());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bounded_receipt_classifies_oom_without_unbounded_output() {
        use std::os::unix::fs::PermissionsExt;
        let temp = tempfile::tempdir().expect("tempdir");
        let script = temp.path().join("fake-bench");
        std::fs::write(&script, "#!/bin/sh\necho 'out of memory' >&2\nexit 1\n").expect("script");
        let mut permissions = std::fs::metadata(&script).expect("metadata").permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&script, permissions).expect("permissions");
        let receipt = run_bench_receipt(&script, temp.path(), &[], Duration::from_secs(5))
            .await
            .expect("receipt");
        assert_eq!(receipt.failure, Some(BenchFailureKind::Oom));
        assert!(!receipt.stdout_truncated);
        assert!(!receipt.stderr_truncated);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn timeout_terminates_process_group_and_descendant() {
        use std::os::unix::fs::PermissionsExt;
        let temp = tempfile::tempdir().expect("tempdir");
        let script = temp.path().join("tree-bench");
        let pid_file = temp.path().join("child.pid");
        std::fs::write(
            &script,
            format!(
                "#!/bin/sh\nsleep 30 &\necho $! > {}\nwait\n",
                pid_file.display()
            ),
        )
        .expect("script");
        let mut permissions = std::fs::metadata(&script).expect("metadata").permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&script, permissions).expect("permissions");
        let receipt = run_bench_receipt(&script, temp.path(), &[], Duration::from_secs(1))
            .await
            .expect("timeout receipt");
        assert_eq!(receipt.failure, Some(BenchFailureKind::Timeout));
        let child_pid = std::fs::read_to_string(pid_file)
            .expect("child pid")
            .trim()
            .parse::<i32>()
            .expect("pid number");
        std::thread::sleep(Duration::from_millis(100));
        let still_alive = unsafe { libc::kill(child_pid, 0) == 0 };
        assert!(!still_alive, "benchmark descendant survived timeout");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn output_limit_is_fail_closed() {
        use std::os::unix::fs::PermissionsExt;
        let temp = tempfile::tempdir().expect("tempdir");
        let script = temp.path().join("large-bench");
        std::fs::write(
            &script,
            "#!/bin/sh\ndd if=/dev/zero bs=2200000 count=1 2>/dev/null\n",
        )
        .expect("script");
        let mut permissions = std::fs::metadata(&script).expect("metadata").permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&script, permissions).expect("permissions");
        let receipt = run_bench_receipt(&script, temp.path(), &[], Duration::from_secs(5))
            .await
            .expect("receipt");
        assert_eq!(receipt.failure, Some(BenchFailureKind::OutputLimit));
        assert!(receipt.stdout_truncated);
    }
}
