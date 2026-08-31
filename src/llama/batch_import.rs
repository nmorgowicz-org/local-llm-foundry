use std::fs;
use std::path::Path;

use crate::inference::llama_cpp::{LlamaReasoningEffort, LlamaReasoningFormat, LoadMode};
use crate::presets::ModelPreset;

/// Parsed result from an imported launch script.
#[derive(Debug, Clone)]
pub struct ImportResult {
    pub preset: ModelPreset,
    pub warnings: Vec<String>,
}

/// Read a launch file from disk and parse it.
/// Detects OS from platform and file extension.
pub fn import_launch_file(file: &str) -> Result<ImportResult, String> {
    let content =
        fs::read_to_string(file).map_err(|e| format!("Failed to read file '{}': {}", file, e))?;

    let path = Path::new(file);
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    let os = if ext == "bat" || ext == "cmd" || cfg!(target_os = "windows") {
        "windows"
    } else {
        "linux"
    };

    parse_launch_script(&content, os)
}

/// Import a launch script (batch/sh) into a ModelPreset.
pub fn parse_launch_script(content: &str, os: &str) -> Result<ImportResult, String> {
    let (binary_path, args, warnings) = match os {
        "windows" => parse_windows_script(content),
        "macos" | "linux" => parse_unix_script(content),
        _ => return Err(format!("Unsupported OS: {os}")),
    };

    if binary_path.is_empty() {
        return Err("Could not detect llama-server binary path in script".into());
    }

    let preset = build_preset_from_args(&args);
    let mut warnings = warnings;
    // Precedence is stated here, not left implicit in emission order: when
    // both are set, emission at llama_cpp.rs prefers fit_target and silently
    // drops fit_ctx.
    if preset.fit_target.is_some() && preset.fit_ctx.is_some() {
        warnings.push(
            "Both --fit-target and --fit-ctx imported; at launch --fit-target takes precedence and --fit-ctx is ignored.".into(),
        );
    }
    Ok(ImportResult { preset, warnings })
}

fn parse_windows_script(content: &str) -> (String, Vec<String>, Vec<String>) {
    let mut warnings = Vec::new();
    let normalized = content
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace(" ^\n", " ")
        .replace(" ^\r\n", " ")
        .replace(" ^\r", " ")
        .replace(" \\\n", " ")
        .replace(" \\\r\n", " ")
        .replace(" \\\r", " ");

    let lines: Vec<&str> = normalized.lines().collect();
    let command_line: &str = lines
        .iter()
        .find(|l| {
            let s = l.trim();
            !s.is_empty() && !s.starts_with("::") && !s.starts_with("@echo")
        })
        .copied()
        .unwrap_or("");

    let tokens = tokenize_win(command_line);
    let mut binary_path = String::new();
    let mut args = Vec::new();

    if let Some(first) = tokens.first() {
        binary_path = first.clone();
        for t in tokens.iter().skip(1) {
            args.push(t.clone());
        }
    }

    if binary_path.contains("llama-server") || binary_path.ends_with(".exe") {
        // OK.
    } else {
        warnings.push("Binary path may not be llama-server; verify manually.".to_string());
    }

    (binary_path, args, warnings)
}

fn tokenize_win(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut in_quote = false;

    for ch in line.chars() {
        match ch {
            '"' if !in_quote => in_quote = true,
            '"' if in_quote => {
                in_quote = false;
                current.push('"');
            }
            ' ' | '\t' if !in_quote => {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
            }
            _ => current.push(ch),
        }
    }

    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn parse_unix_script(content: &str) -> (String, Vec<String>, Vec<String>) {
    let mut warnings = Vec::new();
    let normalized = content.replace("\\\n", " ").replace("\\\r\n", " ");

    let lines: Vec<&str> = normalized.lines().collect();

    let command_line: &str = lines
        .iter()
        .find(|l| {
            let s = l.trim();
            !s.is_empty() && !s.starts_with('#')
        })
        .copied()
        .unwrap_or("");

    let tokens: Vec<String> = shlex_like_split(command_line);
    let mut binary_path = String::new();
    let mut args = Vec::new();

    if let Some(first) = tokens.first() {
        binary_path = first.clone();
        for t in tokens.iter().skip(1) {
            args.push(t.clone());
        }
    }

    if binary_path.contains("llama-server") || binary_path.ends_with(".sh") {
        // OK.
    } else {
        warnings.push("Binary path may not be llama-server; verify manually.".to_string());
    }

    for line in &lines {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        if let Some(env) = parse_gpu_env_hint(line) {
            warnings.push(env);
        }
    }

    (binary_path, args, warnings)
}

fn parse_gpu_env_hint(line: &str) -> Option<String> {
    let line = line.trim();
    if line.starts_with('#') {
        return None;
    }
    if let Some((var, val)) = extract_env_var(line) {
        match var {
            "CUDA_VISIBLE_DEVICES" => {
                return Some(format!(
                    "CUDA_VISIBLE_DEVICES={val} — only these GPU(s) will be used"
                ));
            }
            "HSA_OVERRIDE_GFX_VERSION" => {
                return Some(format!(
                    "HSA_OVERRIDE_GFX_VERSION={val} — ROCm GPU override in effect"
                ));
            }
            "ROCR_VISIBLE_DEVICES" => {
                return Some(format!(
                    "ROCR_VISIBLE_DEVICES={val} — ROCm device selection"
                ));
            }
            "GGML_CUDA_FORCE_MMQ"
            | "GGML_CUDA_FA_DISABLE"
            | "GGML_HIP_BLAS_HANDLE"
            | "ZES_ENABLE_SYSMAN"
            | "SYCL_DEVICE_FILTER" => {
                return Some(format!("GPU env: {var}={val}"));
            }
            _ => {}
        }
    }
    None
}

fn extract_env_var(line: &str) -> Option<(&str, &str)> {
    let line = line.trim();
    if line.starts_with('#') || line.starts_with("//") {
        return None;
    }
    let rest = if let Some(s) = line.strip_prefix("export ") {
        s
    } else if line.starts_with("set ") {
        return None;
    } else {
        line
    };
    let rest = rest.trim();
    let (var, val) = rest.split_once('=')?;
    let var = var.trim();
    let val = val.trim().trim_matches('"').trim_matches('\'');
    if var.is_empty() || val.is_empty() || !var.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return None;
    }
    Some((var, val))
}

fn shlex_like_split(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut in_quote = None::<char>;

    for ch in line.chars() {
        match (ch, in_quote) {
            ('"' | '\'', Some(q)) if ch == q => {
                in_quote = None;
                current.push(ch);
            }
            ('"' | '\'', None) => {
                in_quote = Some(ch);
                current.push(ch);
            }
            (' ' | '\t', None) => {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
            }
            _ => current.push(ch),
        }
    }

    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn build_preset_from_args(args: &[String]) -> ModelPreset {
    let mut model_path = String::new();
    let mut context_size: u64 = 4096;
    let mut gpu_layers: Option<i32> = None;
    let mut no_mmap = false;
    let mut load_mode: Option<LoadMode> = None;
    let mut ngram_spec = false;
    let mut temperature: Option<f64> = None;
    let mut top_p: Option<f64> = None;
    let mut top_k: Option<i32> = None;
    let mut min_p: Option<f64> = None;
    let mut repeat_penalty: Option<f64> = None;
    let mut ctk: Option<String> = None;
    let mut ctv: Option<String> = None;
    let mut n_cpu_moe: Option<i32> = None;
    let mut spec_type: Option<String> = None;
    let mut spec_default = false;
    let draft_model = String::new();
    let mut spec_draft_n_max: Option<u32> = None;
    let mut spec_draft_n_min: Option<u32> = None;
    let mut spec_draft_p_split: Option<f32> = None;
    let mut spec_draft_p_min: Option<f32> = None;
    let mut spec_draft_ngl: Option<i32> = None;
    let mut spec_draft_device: Option<String> = None;
    let mut spec_draft_cpu_moe = false;
    let mut spec_draft_n_cpu_moe: Option<i32> = None;
    let mut spec_draft_type_k: Option<String> = None;
    let mut spec_draft_type_v: Option<String> = None;
    let mut spec_ngram_mod_n_min: Option<u32> = None;
    let mut spec_ngram_mod_n_max: Option<u32> = None;
    let mut spec_ngram_mod_n_match: Option<u32> = None;
    let mut spec_ngram_simple_size_n: Option<u32> = None;
    let mut spec_ngram_simple_size_m: Option<u32> = None;
    let mut spec_ngram_simple_min_hits: Option<u32> = None;
    let mut spec_ngram_map_k_size_n: Option<u32> = None;
    let mut spec_ngram_map_k_size_m: Option<u32> = None;
    let mut spec_ngram_map_k_min_hits: Option<u32> = None;
    let mut spec_ngram_map_k4v_size_n: Option<u32> = None;
    let mut spec_ngram_map_k4v_size_m: Option<u32> = None;
    let mut spec_ngram_map_k4v_min_hits: Option<u32> = None;
    let mut kv_unified: Option<bool> = None;
    let mut cache_idle_slots: Option<bool> = None;
    let mut fit_enabled: Option<bool> = None;
    let mut fit_ctx: Option<u32> = None;
    let mut fit_target: Option<String> = None;
    let mut fit_print: Option<bool> = None;
    let mut mmproj_offload: Option<bool> = None;
    let mut llama_reasoning_effort = LlamaReasoningEffort::Default;
    let mut llama_reasoning_format: Option<LlamaReasoningFormat> = None;
    let mut llama_reasoning_preserve: Option<bool> = None;
    let mut prio: Option<i32> = None;
    let mut prio_batch: Option<i32> = None;
    let mut extra_args = String::new();

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        match arg.as_str() {
            "-m" | "--model" => {
                if i + 1 < args.len() {
                    model_path = args[i + 1].clone();
                    i += 2;
                    continue;
                }
            }
            "-c" | "--ctx-size" | "--context-size" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<u64>() {
                        context_size = v;
                    }
                    i += 2;
                    continue;
                }
            }
            "-ngl" | "--gpu-layers" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<i32>() {
                        gpu_layers = Some(v);
                    }
                    i += 2;
                    continue;
                }
            }
            "--no-mmap" => {
                no_mmap = true;
                load_mode = Some(LoadMode::None);
                i += 1;
                continue;
            }
            "--load-mode" => {
                if i + 1 < args.len() {
                    load_mode = match args[i + 1].as_str() {
                        "mmap" => Some(LoadMode::Mmap),
                        "none" => Some(LoadMode::None),
                        "mlock" => Some(LoadMode::Mlock),
                        "mmap+mlock" => Some(LoadMode::MmapMlock),
                        "dio" => Some(LoadMode::Dio),
                        _ => None,
                    };
                    no_mmap = matches!(load_mode, Some(LoadMode::None));
                    i += 2;
                    continue;
                }
            }
            "--spec-type" => {
                if i + 1 < args.len() {
                    let v = args[i + 1].clone();
                    spec_type = Some(v.clone());
                    ngram_spec = v.contains("ngram");
                    i += 2;
                    continue;
                }
            }
            "--spec-default" => {
                spec_default = true;
                i += 1;
                continue;
            }
            "--spec-draft-n-max" => {
                if i + 1 < args.len() {
                    let _ = args[i + 1].parse::<u32>().map(|v| {
                        spec_draft_n_max = Some(v);
                    });
                    i += 2;
                    continue;
                }
            }
            "--spec-draft-n-min" => {
                if i + 1 < args.len() {
                    let _ = args[i + 1].parse::<u32>().map(|v| {
                        spec_draft_n_min = Some(v);
                    });
                    i += 2;
                    continue;
                }
            }
            "--spec-draft-p-split" => {
                if i + 1 < args.len() {
                    let _ = args[i + 1].parse::<f32>().map(|v| {
                        spec_draft_p_split = Some(v);
                    });
                    i += 2;
                    continue;
                }
            }
            "--spec-draft-p-min" => {
                if i + 1 < args.len() {
                    let _ = args[i + 1].parse::<f32>().map(|v| {
                        spec_draft_p_min = Some(v);
                    });
                    i += 2;
                    continue;
                }
            }
            "--spec-draft-ngl" => {
                if i + 1 < args.len() {
                    let _ = args[i + 1].parse::<i32>().map(|v| {
                        spec_draft_ngl = Some(v);
                    });
                    i += 2;
                    continue;
                }
            }
            "--spec-draft-device" => {
                if i + 1 < args.len() {
                    spec_draft_device = Some(args[i + 1].clone());
                    i += 2;
                    continue;
                }
            }
            "--spec-draft-cpu-moe" => {
                spec_draft_cpu_moe = true;
                i += 1;
                continue;
            }
            "--spec-draft-n-cpu-moe" => {
                if i + 1 < args.len() {
                    let _ = args[i + 1].parse::<i32>().map(|v| {
                        spec_draft_n_cpu_moe = Some(v);
                    });
                    i += 2;
                    continue;
                }
            }
            "--spec-draft-type-k" | "-ctkd" | "--cache-type-k-draft" => {
                if i + 1 < args.len() {
                    spec_draft_type_k = Some(args[i + 1].clone());
                    i += 2;
                    continue;
                }
            }
            "--spec-draft-type-v" | "-ctvd" | "--cache-type-v-draft" => {
                if i + 1 < args.len() {
                    spec_draft_type_v = Some(args[i + 1].clone());
                    i += 2;
                    continue;
                }
            }
            "--cache-type-k" | "-ctk" => {
                if i + 1 < args.len() {
                    ctk = Some(args[i + 1].clone());
                    i += 2;
                    continue;
                }
            }
            "--cache-type-v" | "-ctv" => {
                if i + 1 < args.len() {
                    ctv = Some(args[i + 1].clone());
                    i += 2;
                    continue;
                }
            }
            "--spec-ngram-mod-n-min" => {
                if i + 1 < args.len() {
                    let _ = args[i + 1].parse::<u32>().map(|v| {
                        spec_ngram_mod_n_min = Some(v);
                    });
                    i += 2;
                    continue;
                }
            }
            "--spec-ngram-mod-n-max" => {
                if i + 1 < args.len() {
                    let _ = args[i + 1].parse::<u32>().map(|v| {
                        spec_ngram_mod_n_max = Some(v);
                    });
                    i += 2;
                    continue;
                }
            }
            "--spec-ngram-mod-n-match" => {
                if i + 1 < args.len() {
                    let _ = args[i + 1].parse::<u32>().map(|v| {
                        spec_ngram_mod_n_match = Some(v);
                    });
                    i += 2;
                    continue;
                }
            }
            "--spec-ngram-simple-size-n" => {
                if i + 1 < args.len() {
                    let _ = args[i + 1].parse::<u32>().map(|v| {
                        spec_ngram_simple_size_n = Some(v);
                    });
                    i += 2;
                    continue;
                }
            }
            "--spec-ngram-simple-size-m" => {
                if i + 1 < args.len() {
                    let _ = args[i + 1].parse::<u32>().map(|v| {
                        spec_ngram_simple_size_m = Some(v);
                    });
                    i += 2;
                    continue;
                }
            }
            "--spec-ngram-simple-min-hits" => {
                if i + 1 < args.len() {
                    let _ = args[i + 1].parse::<u32>().map(|v| {
                        spec_ngram_simple_min_hits = Some(v);
                    });
                    i += 2;
                    continue;
                }
            }
            "--spec-ngram-map-k-size-n" => {
                if i + 1 < args.len() {
                    let _ = args[i + 1].parse::<u32>().map(|v| {
                        spec_ngram_map_k_size_n = Some(v);
                    });
                    i += 2;
                    continue;
                }
            }
            "--spec-ngram-map-k-size-m" => {
                if i + 1 < args.len() {
                    let _ = args[i + 1].parse::<u32>().map(|v| {
                        spec_ngram_map_k_size_m = Some(v);
                    });
                    i += 2;
                    continue;
                }
            }
            "--spec-ngram-map-k-min-hits" => {
                if i + 1 < args.len() {
                    let _ = args[i + 1].parse::<u32>().map(|v| {
                        spec_ngram_map_k_min_hits = Some(v);
                    });
                    i += 2;
                    continue;
                }
            }
            "--spec-ngram-map-k4v-size-n" => {
                if i + 1 < args.len() {
                    let _ = args[i + 1].parse::<u32>().map(|v| {
                        spec_ngram_map_k4v_size_n = Some(v);
                    });
                    i += 2;
                    continue;
                }
            }
            "--spec-ngram-map-k4v-size-m" => {
                if i + 1 < args.len() {
                    let _ = args[i + 1].parse::<u32>().map(|v| {
                        spec_ngram_map_k4v_size_m = Some(v);
                    });
                    i += 2;
                    continue;
                }
            }
            "--spec-ngram-map-k4v-min-hits" => {
                if i + 1 < args.len() {
                    let _ = args[i + 1].parse::<u32>().map(|v| {
                        spec_ngram_map_k4v_min_hits = Some(v);
                    });
                    i += 2;
                    continue;
                }
            }
            "--kv-unified" => {
                if i + 1 < args.len() {
                    let _ = args[i + 1].parse::<bool>().map(|v| {
                        kv_unified = Some(v);
                    });
                    i += 2;
                    continue;
                }
            }
            "--cache-idle-slots" => {
                if i + 1 < args.len() {
                    let _ = args[i + 1].parse::<bool>().map(|v| {
                        cache_idle_slots = Some(v);
                    });
                    i += 2;
                    continue;
                }
            }
            "--fit-enabled" => {
                if i + 1 < args.len() {
                    let _ = args[i + 1].parse::<bool>().map(|v| {
                        fit_enabled = Some(v);
                    });
                    i += 2;
                    continue;
                }
            }
            "--fit-ctx" => {
                if i + 1 < args.len() {
                    let _ = args[i + 1].parse::<u32>().map(|v| {
                        fit_ctx = Some(v);
                    });
                    i += 2;
                    continue;
                }
            }
            "--fit-target" => {
                if i + 1 < args.len() {
                    fit_target = Some(args[i + 1].clone());
                    i += 2;
                    continue;
                }
            }
            "--fit-print" => {
                if i + 1 < args.len() {
                    let _ = args[i + 1].parse::<bool>().map(|v| {
                        fit_print = Some(v);
                    });
                    i += 2;
                    continue;
                }
            }
            "--mmproj-offload" => {
                mmproj_offload = Some(true);
                i += 1;
                continue;
            }
            "--no-mmproj-offload" => {
                mmproj_offload = Some(false);
                i += 1;
                continue;
            }
            "--reasoning-effort" => {
                if i + 1 < args.len() {
                    llama_reasoning_effort = match args[i + 1].as_str() {
                        "default" => LlamaReasoningEffort::Default,
                        "minimal" => LlamaReasoningEffort::Minimal,
                        "low" => LlamaReasoningEffort::Low,
                        "medium" => LlamaReasoningEffort::Medium,
                        "high" => LlamaReasoningEffort::High,
                        "xhigh" => LlamaReasoningEffort::Xhigh,
                        "max" => LlamaReasoningEffort::Max,
                        value => LlamaReasoningEffort::Unknown(value.to_string()),
                    };
                    i += 2;
                    continue;
                }
            }
            "--reasoning-format" => {
                if i + 1 < args.len() {
                    llama_reasoning_format = Some(match args[i + 1].as_str() {
                        "none" => LlamaReasoningFormat::None,
                        "deepseek" => LlamaReasoningFormat::Deepseek,
                        "deepseek-legacy" => LlamaReasoningFormat::DeepseekLegacy,
                        value => LlamaReasoningFormat::Unknown(value.to_string()),
                    });
                    i += 2;
                    continue;
                }
            }
            "--reasoning-preserve" => {
                llama_reasoning_preserve = Some(true);
                i += 1;
                continue;
            }
            "--no-reasoning-preserve" => {
                llama_reasoning_preserve = Some(false);
                i += 1;
                continue;
            }
            "--prio" => {
                if i + 1 < args.len() {
                    let _ = args[i + 1].parse::<i32>().map(|v| {
                        prio = Some(v);
                    });
                    i += 2;
                    continue;
                }
            }
            "--prio-batch" => {
                if i + 1 < args.len() {
                    let _ = args[i + 1].parse::<i32>().map(|v| {
                        prio_batch = Some(v);
                    });
                    i += 2;
                    continue;
                }
            }
            "--temp" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<f64>() {
                        temperature = Some(v);
                    }
                    i += 2;
                    continue;
                }
            }
            "--top-p" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<f64>() {
                        top_p = Some(v);
                    }
                    i += 2;
                    continue;
                }
            }
            "--top-k" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<i32>() {
                        top_k = Some(v);
                    }
                    i += 2;
                    continue;
                }
            }
            "--min-p" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<f64>() {
                        min_p = Some(v);
                    }
                    i += 2;
                    continue;
                }
            }
            "--repeat-penalty" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<f64>() {
                        repeat_penalty = Some(v);
                    }
                    i += 2;
                    continue;
                }
            }
            "--n-cpu-moe" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<i32>() {
                        n_cpu_moe = Some(v);
                    }
                    i += 2;
                    continue;
                }
            }
            _ => {
                if i + 1 < args.len() && args[i + 1].chars().next().is_some_and(|c| c != '-') {
                    extra_args.push_str(arg);
                    extra_args.push(' ');
                    extra_args.push_str(&args[i + 1]);
                    extra_args.push(' ');
                    i += 2;
                    continue;
                } else {
                    extra_args.push_str(arg);
                    extra_args.push(' ');
                    i += 1;
                    continue;
                }
            }
        }
        i += 1;
    }

    let name = if model_path.is_empty() {
        "Imported preset".to_string()
    } else {
        let file = std::path::Path::new(&model_path)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "Imported preset".into());
        format!("Imported: {}", file)
    };

    ModelPreset {
        backend: crate::inference::InferenceBackend::LlamaCpp,
        rapid_mlx: None,
        id: crate::presets::next_id(),
        name,
        schema_version: None,
        revision: 0,
        model_path,
        context_size,
        ctk: ctk.unwrap_or_default(),
        ctv: ctv.unwrap_or_default(),
        tensor_split: String::new(),
        batch_size: 2048,
        ubatch_size: 2048,
        no_mmap,
        load_mode,
        verbosity: Some(4),
        no_cont_batching: false,
        swa_full: false,
        ctx_checkpoints: Some(32),
        checkpoint_min_step: Some(8192),
        cache_reuse: None,
        ngram_spec,
        parallel_slots: 1,
        temperature,
        top_p,
        top_k,
        min_p,
        repeat_penalty,
        repeat_last_n: None,
        presence_penalty: None,
        n_cpu_moe,
        gpu_layers,
        mlock: false,

        // Architecture metadata fields (populated later via ensure_gguf_metadata).
        architecture_kind: None,
        expert_count: None,
        expert_used_count: None,
        active_params_b: None,
        block_count: None,
        bytes_per_layer: None,
        expert_bytes_per_layer: None,
        flash_attn: String::new(),
        split_mode: String::new(),
        main_gpu: None,
        threads: None,
        threads_batch: None,
        prio: None,
        prio_batch: None,
        rope_scaling: String::new(),
        rope_freq_base: None,
        rope_freq_scale: None,
        draft_model,
        draft_min: None,
        draft_max: None,
        spec_ngram_size: None,
        spec_type,
        spec_default,
        spec_draft_n_max,
        spec_draft_n_min,
        spec_draft_p_split,
        spec_draft_p_min,
        spec_draft_ngl,
        spec_draft_device,
        spec_draft_cpu_moe,
        spec_draft_n_cpu_moe,
        spec_draft_type_k,
        spec_draft_type_v,
        spec_ngram_mod_n_min,
        spec_ngram_mod_n_max,
        spec_ngram_mod_n_match,
        spec_ngram_simple_size_n,
        spec_ngram_simple_size_m,
        spec_ngram_simple_min_hits,
        spec_ngram_map_k_size_n,
        spec_ngram_map_k_size_m,
        spec_ngram_map_k_min_hits,
        spec_ngram_map_k4v_size_n,
        spec_ngram_map_k4v_size_m,
        spec_ngram_map_k4v_min_hits,
        kv_unified,
        cache_idle_slots,
        cache_ram_mib: None,
        fit_enabled,
        fit_ctx,
        fit_target,
        fit_print,
        // Batch import produces flat legacy-shaped presets, not bundles.
        // Only `bundle::create_bundle_preset` is allowed to set `bundle`.
        bundle: None,
        mmproj_offload,
        llama_reasoning_effort,
        llama_reasoning_format,
        llama_reasoning_preserve,
        seed: None,
        system_prompt_file: String::new(),
        extra_args: extra_args.trim().to_string(),
        bind_host: None,
        port: None,
        hf_repo: None,
        chat_template_file: None,
        mmproj: None,
        image_min_tokens: None,
        image_max_tokens: None,
        grammar: None,
        json_schema: None,
        cache_type_k: None,
        cache_type_v: None,
        max_tokens: None,
        enable_thinking: None,
        preserve_thinking: None,
        tool_call_format: None,
        reasoning: None,
        reasoning_budget: None,
        reasoning_budget_message: None,
        api_key: None,
        api_key_configured: false,
        clear_api_key: false,
        alias: None,
        benchmark_mode: false,
        tags: Vec::new(),
        gguf_architecture: None,
        param_count: None,
        family: None,
        size_class: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_windows_script() {
        let script = r#"
@echo off
llama-server.exe -m models\my-model.gguf -c 4096 -ngl 99 --temp 0.8
"#;
        let result = parse_launch_script(script, "windows").unwrap();
        assert!(result.preset.model_path.contains("my-model.gguf"));
        assert_eq!(result.preset.context_size, 4096);
        assert_eq!(result.preset.gpu_layers, Some(99));
        assert_eq!(result.preset.temperature, Some(0.8));
    }

    #[test]
    fn test_parse_simple_unix_script() {
        let script = r#"
#!/bin/bash
./llama-server -m /models/my-model.gguf -c 8192 --top-p 0.95
"#;
        let result = parse_launch_script(script, "linux").unwrap();
        assert!(result.preset.model_path.contains("my-model.gguf"));
        assert_eq!(result.preset.context_size, 8192);
        assert_eq!(result.preset.top_p, Some(0.95));
    }

    #[test]
    fn test_parse_windows_line_continuation() {
        let script = r#"
llama-server.exe -m "models\my-model.gguf" -c 4096 ^
    -ngl 99 --temp 0.7
"#;
        let result = parse_launch_script(script, "windows").unwrap();
        assert!(result.preset.model_path.contains("my-model.gguf"));
        assert_eq!(result.preset.context_size, 4096);
        assert_eq!(result.preset.gpu_layers, Some(99));
        assert_eq!(result.preset.temperature, Some(0.7));
    }

    #[test]
    fn test_parse_unix_line_continuation() {
        let script = r#"
./llama-server -m /models/my-model.gguf -c 4096 \
    -ngl 99 --top-k 40
"#;
        let result = parse_launch_script(script, "linux").unwrap();
        assert!(result.preset.model_path.contains("my-model.gguf"));
        assert_eq!(result.preset.context_size, 4096);
        assert_eq!(result.preset.gpu_layers, Some(99));
        assert_eq!(result.preset.top_k, Some(40));
    }

    #[test]
    fn test_parse_unknown_flags_goes_to_extra_args() {
        let script = r#"
llama-server -m model.gguf --foo bar
"#;
        let result = parse_launch_script(script, "linux").unwrap();
        assert!(result.preset.extra_args.contains("--foo bar"));
    }

    // Phase 1b: importing a launch script that sets all four K/V options via
    // aliases must populate the four typed fields and leave extra_args empty
    // (the importer must not reintroduce K/V flags as raw text).
    #[test]
    fn test_parse_kv_flags_all_aliases_populate_typed_fields() {
        let script = r#"
llama-server -m model.gguf --cache-type-k q8_0 --cache-type-v q4_0 -ctkd q4_0 -ctvd q4_0
"#;
        let result = parse_launch_script(script, "linux").unwrap();
        assert_eq!(result.preset.ctk, "q8_0");
        assert_eq!(result.preset.ctv, "q4_0");
        assert_eq!(result.preset.spec_draft_type_k.as_deref(), Some("q4_0"));
        assert_eq!(result.preset.spec_draft_type_v.as_deref(), Some("q4_0"));
        assert!(
            result.preset.extra_args.trim().is_empty(),
            "extra_args must be empty after K/V import, got: {:?}",
            result.preset.extra_args
        );
        // The shared validator must find no KV-override issues for this preset.
        let issues = crate::presets::validation::validate_llama_launch_policy(&result.preset, None);
        assert!(
            !issues.iter().any(|i| i.code == "EXTRA_ARGS_KV_OVERRIDE"),
            "imported K/V must not trip EXTRA_ARGS_KV_OVERRIDE"
        );
    }

    #[test]
    fn test_parse_kv_flags_long_forms_only() {
        let script = r#"
llama-server -m model.gguf --cache-type-k q4_0 --cache-type-v f16 --spec-draft-type-k q5_0 --spec-draft-type-v q5_0
"#;
        let result = parse_launch_script(script, "linux").unwrap();
        assert_eq!(result.preset.ctk, "q4_0");
        assert_eq!(result.preset.ctv, "f16");
        assert_eq!(result.preset.spec_draft_type_k.as_deref(), Some("q5_0"));
        assert_eq!(result.preset.spec_draft_type_v.as_deref(), Some("q5_0"));
        assert!(
            result.preset.extra_args.trim().is_empty(),
            "extra_args must be empty, got: {:?}",
            result.preset.extra_args
        );
    }

    #[test]
    fn test_parse_kv_flags_absent_defaults_to_empty_not_f16() {
        let script = r#"
llama-server -m model.gguf -c 4096
"#;
        let result = parse_launch_script(script, "linux").unwrap();
        // When no K/V flag is present the canonical fields stay empty so the
        // runtime uses llama-server's own default (f16) rather than having the
        // importer hardcode a choice for the user.
        assert!(result.preset.ctk.trim().is_empty());
        assert!(result.preset.ctv.trim().is_empty());
    }

    #[test]
    fn test_parse_phase2_typed_llama_flags() {
        let script = r#"
llama-server -m model.gguf --mmproj-offload --reasoning-effort xhigh --reasoning-format deepseek-legacy --reasoning-preserve
"#;
        let result = parse_launch_script(script, "linux").unwrap();
        assert_eq!(result.preset.mmproj_offload, Some(true));
        assert_eq!(
            result.preset.llama_reasoning_effort,
            LlamaReasoningEffort::Xhigh
        );
        assert_eq!(
            result.preset.llama_reasoning_format,
            Some(LlamaReasoningFormat::DeepseekLegacy)
        );
        assert_eq!(result.preset.llama_reasoning_preserve, Some(true));
        assert!(result.preset.extra_args.trim().is_empty());
    }

    #[test]
    fn test_parse_phase2_negative_and_unknown_typed_flags() {
        let script = r#"
llama-server -m model.gguf --no-mmproj-offload --reasoning-effort future --reasoning-format future-format --no-reasoning-preserve
"#;
        let result = parse_launch_script(script, "linux").unwrap();
        assert_eq!(result.preset.mmproj_offload, Some(false));
        assert!(matches!(
            result.preset.llama_reasoning_effort,
            LlamaReasoningEffort::Unknown(value) if value == "future"
        ));
        assert!(matches!(
            result.preset.llama_reasoning_format,
            Some(LlamaReasoningFormat::Unknown(value)) if value == "future-format"
        ));
        assert_eq!(result.preset.llama_reasoning_preserve, Some(false));
        assert!(result.preset.extra_args.trim().is_empty());
    }

    #[test]
    fn test_fit_ctx_and_fit_target_precedence_warning_emitted() {
        let script = r#"
llama-server -m model.gguf --fit-target 4096 --fit-ctx 2048
"#;
        let result = parse_launch_script(script, "linux").unwrap();
        assert_eq!(result.preset.fit_target.as_deref(), Some("4096"));
        assert_eq!(result.preset.fit_ctx, Some(2048));
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.contains("--fit-target") && w.contains("precedence")),
            "expected a precedence warning, got: {:?}",
            result.warnings
        );
    }
}
