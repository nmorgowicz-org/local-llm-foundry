#[derive(Debug, Clone, serde::Serialize)]
pub struct EscapeFlagDescriptor {
    pub flag: &'static str,
    pub value_type: &'static str,
    pub description: &'static str,
    pub tooltip: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enum_options: Option<&'static [&'static str]>,
    /// The real `rapid-mlx serve --help` default for this flag, as display
    /// text (e.g. "32768", "0.20"). `None` when the CLI has no fixed default
    /// to show (e.g. a conditional default, or a bare on/off toggle flag).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<&'static str>,
}

pub const ALLOWED_ESCAPE_FLAGS: &[EscapeFlagDescriptor] = &[
    // ── PFlash family ──────────────────────────────────────────────────────────
    EscapeFlagDescriptor {
        flag: "pflash",
        value_type: "enum",
        description: "Prefilter cache mode",
        tooltip: "Controls how Rapid-MLX reuses KV cache across requests. 'auto' lets the runtime decide; 'always' maximizes reuse but uses more memory and may degrade context recall on long prompts (>60k tokens); 'off' disables it (default).",
        enum_options: Some(&["off", "auto", "always"]),
        default: Some("off (\"always\" for verified aliases)"),
    },
    EscapeFlagDescriptor {
        flag: "pflash-threshold",
        value_type: "int",
        description: "PFlash activation threshold",
        tooltip: "Minimum prompt length (tokens) at which PFlash reuse becomes active. Below this, a direct forward pass is used.",
        enum_options: None,
        default: Some("32768"),
    },
    EscapeFlagDescriptor {
        flag: "pflash-keep-ratio",
        value_type: "float",
        description: "PFlash keep ratio",
        tooltip: "Fraction of the prefix cache to preserve between requests. 0.0–1.0; higher keeps more history but consumes more memory.",
        enum_options: None,
        default: Some("0.20"),
    },
    EscapeFlagDescriptor {
        flag: "pflash-min-keep-tokens",
        value_type: "int",
        description: "PFlash minimum keep tokens",
        tooltip: "Absolute minimum number of tokens to retain in the prefix cache, regardless of keep-ratio.",
        enum_options: None,
        default: Some("2048"),
    },
    EscapeFlagDescriptor {
        flag: "pflash-sink-tokens",
        value_type: "int",
        description: "PFlash sink tokens",
        tooltip: "Leading prompt tokens always kept by PFlash. Higher values reduce false positives in cache matches.",
        enum_options: None,
        default: Some("256"),
    },
    EscapeFlagDescriptor {
        flag: "pflash-tail-tokens",
        value_type: "int",
        description: "PFlash tail tokens",
        tooltip: "Trailing prompt tokens always kept by PFlash. Adjusts how aggressively the cache can extend.",
        enum_options: None,
        default: Some("2048"),
    },
    EscapeFlagDescriptor {
        flag: "pflash-block-size",
        value_type: "int",
        description: "PFlash block size",
        tooltip: "Middle-token scoring block size, in tokens.",
        enum_options: None,
        default: Some("128"),
    },
    EscapeFlagDescriptor {
        flag: "pflash-query-window",
        value_type: "int",
        description: "PFlash query window",
        tooltip: "Trailing query window used to score middle blocks. Larger windows give more precise matches but slower lookups.",
        enum_options: None,
        default: Some("512"),
    },
    EscapeFlagDescriptor {
        flag: "pflash-stride-blocks",
        value_type: "int",
        description: "PFlash stride blocks",
        tooltip: "Keep every Nth middle block as an anchor during scoring (0 disables anchors).",
        enum_options: None,
        default: Some("8"),
    },
    EscapeFlagDescriptor {
        flag: "pflash-include-tools",
        value_type: "bool",
        description: "PFlash include tools",
        tooltip: "When true, tool definitions and tool-call history are included in PFlash cache keys. Disabling can improve reuse in agentic workloads with rotating tool sets.",
        enum_options: None,
        default: None,
    },
    // ── Hybrid force toggles ────────────────────────────────────────────────────
    EscapeFlagDescriptor {
        flag: "force-hybrid",
        value_type: "bool",
        description: "Force hybrid attention",
        tooltip: "Override the runtime's attention mode and force hybrid attention path. Useful for models where auto-detection is uncertain.",
        enum_options: None,
        default: None,
    },
    EscapeFlagDescriptor {
        flag: "no-hybrid",
        value_type: "bool",
        description: "Disable hybrid attention",
        tooltip: "Force pure attention path; disable hybrid attention mode even if the model supports it.",
        enum_options: None,
        default: None,
    },
];

/// Validate that all flags in a preset are from the allowlist. Returns the
/// subset of flags that failed validation, or empty if all are valid.
pub fn validate_escape_flags(flags: &[(String, serde_json::Value)]) -> Result<(), Vec<String>> {
    let allowlist: std::collections::HashSet<&str> =
        ALLOWED_ESCAPE_FLAGS.iter().map(|d| d.flag).collect();
    let mut invalid = Vec::new();
    for (name, _value) in flags {
        if !allowlist.contains(name.as_str()) {
            invalid.push(name.clone());
        }
    }
    if invalid.is_empty() {
        Ok(())
    } else {
        Err(invalid)
    }
}

/// Strings that reach argv must not be mistaken for another option, and must not carry
/// bytes that corrupt the argument list.
fn safe_argv_string(flag: &str, s: &str) -> Result<(), String> {
    if s.is_empty() || s.starts_with('-') || s.len() > 256 {
        return Err(format!(
            "escape flag --{flag}: value must be non-empty, at most 256 bytes and must not start with '-'"
        ));
    }
    if s.chars().any(|c| c == '\0' || c.is_control()) {
        return Err(format!(
            "escape flag --{flag}: value must not contain NUL, newline or other control characters"
        ));
    }
    Ok(())
}

/// Validate one escape-hatch value against its descriptor and return the argv fragment for
/// it (empty for a `false` switch). Objects, arrays and null are always rejected.
pub fn escape_flag_to_args(name: &str, value: &serde_json::Value) -> Result<Vec<String>, String> {
    use serde_json::Value;
    let descriptor = ALLOWED_ESCAPE_FLAGS
        .iter()
        .find(|d| d.flag == name)
        .ok_or_else(|| format!("escape flag --{name} is not in the allowlist"))?;
    let flag = format!("--{name}");
    match descriptor.value_type {
        "bool" => match value {
            Value::Bool(true) => Ok(vec![flag]),
            Value::Bool(false) => Ok(Vec::new()),
            _ => Err(format!(
                "escape flag {flag} is a switch and accepts only true or false"
            )),
        },
        "enum" => {
            let Value::String(s) = value else {
                return Err(format!("escape flag {flag} requires a string value"));
            };
            let options = descriptor.enum_options.unwrap_or(&[]);
            if !options.contains(&s.as_str()) {
                return Err(format!(
                    "escape flag {flag} must be one of [{}], got {s:?}",
                    options.join(", ")
                ));
            }
            Ok(vec![flag, s.clone()])
        }
        "int" => {
            let Value::Number(n) = value else {
                return Err(format!(
                    "escape flag {flag} requires a non-negative integer"
                ));
            };
            let Some(i) = n.as_u64() else {
                return Err(format!(
                    "escape flag {flag} requires a non-negative integer, got {n}"
                ));
            };
            Ok(vec![flag, i.to_string()])
        }
        "float" => {
            let Value::Number(n) = value else {
                return Err(format!("escape flag {flag} requires a number"));
            };
            match n.as_f64() {
                Some(f) if f.is_finite() && f >= 0.0 => Ok(vec![flag, format!("{f}")]),
                _ => Err(format!(
                    "escape flag {flag} requires a finite, non-negative number, got {n}"
                )),
            }
        }
        "string" => {
            let Value::String(s) = value else {
                return Err(format!("escape flag {flag} requires a string value"));
            };
            safe_argv_string(name, s)?;
            Ok(vec![flag, s.clone()])
        }
        other => Err(format!(
            "escape flag {flag} has unsupported descriptor type {other:?}"
        )),
    }
}

/// Validate the names and values of a whole preset's escape flags.
pub fn validate_escape_flag_values(flags: &[(String, serde_json::Value)]) -> Result<(), String> {
    validate_escape_flags(flags)
        .map_err(|invalid| format!("escape flags not in the allowlist: {}", invalid.join(", ")))?;
    for (name, value) in flags {
        escape_flag_to_args(name, value)?;
    }
    Ok(())
}

/// Ensure the two process-lifecycle-internal flags are never exposed in the
/// allowlist. This is a compile-time sanity check.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn watchdog_ppid_not_in_allowlist() {
        let names: Vec<&str> = ALLOWED_ESCAPE_FLAGS.iter().map(|d| d.flag).collect();
        assert!(
            !names.contains(&"watchdog-ppid"),
            "--watchdog-ppid must never be in the escape-hatch allowlist"
        );
    }

    #[test]
    fn listen_fd_not_in_allowlist() {
        let names: Vec<&str> = ALLOWED_ESCAPE_FLAGS.iter().map(|d| d.flag).collect();
        assert!(
            !names.contains(&"listen-fd"),
            "--listen-fd must never be in the escape-hatch allowlist"
        );
    }

    #[test]
    fn validate_rejects_non_allowlisted_flag() {
        let flags = vec![
            (
                "pflash".to_string(),
                serde_json::Value::String("auto".into()),
            ),
            ("unknown-flag".to_string(), serde_json::Value::Bool(true)),
        ];
        let err = validate_escape_flags(&flags).unwrap_err();
        assert_eq!(err, vec!["unknown-flag"]);
    }

    #[test]
    fn validate_accepts_all_allowlisted_flags() {
        let flags: Vec<_> = ALLOWED_ESCAPE_FLAGS
            .iter()
            .map(|d| {
                let value = match d.value_type {
                    "bool" => serde_json::Value::Bool(true),
                    "int" => serde_json::Value::Number(serde_json::Number::from(42)),
                    "float" => {
                        serde_json::Value::Number(serde_json::Number::from_f64(0.75).unwrap())
                    }
                    "enum" => serde_json::Value::String(
                        d.enum_options
                            .and_then(|opts| opts.first().copied())
                            .unwrap_or("auto")
                            .to_string(),
                    ),
                    _ => serde_json::Value::Null,
                };
                (d.flag.to_string(), value)
            })
            .collect();
        assert!(validate_escape_flags(&flags).is_ok());
    }

    /// Every non-boolean (int/float/enum) flag should surface a real
    /// `rapid-mlx serve --help` default so the wizard can show a
    /// placeholder instead of an empty input. Pure on/off toggle flags
    /// (bool) have no meaningful "default value" to display and are
    /// exempt.
    #[test]
    fn numeric_and_enum_flags_have_a_surfaced_default() {
        for d in ALLOWED_ESCAPE_FLAGS {
            if d.value_type == "bool" {
                continue;
            }
            assert!(
                d.default.is_some(),
                "flag '{}' (type {}) has no `default` surfaced for the wizard",
                d.flag,
                d.value_type
            );
        }
    }

    #[test]
    fn pflash_threshold_and_keep_ratio_defaults_match_real_cli_help() {
        let by_flag = |flag: &str| {
            ALLOWED_ESCAPE_FLAGS
                .iter()
                .find(|d| d.flag == flag)
                .unwrap_or_else(|| panic!("missing descriptor for {flag}"))
        };
        assert_eq!(by_flag("pflash-threshold").default, Some("32768"));
        assert_eq!(by_flag("pflash-keep-ratio").default, Some("0.20"));
        assert_eq!(by_flag("pflash-block-size").default, Some("128"));
    }

    #[test]
    fn pflash_block_size_tooltip_has_no_fabricated_power_of_two_claim() {
        let d = ALLOWED_ESCAPE_FLAGS
            .iter()
            .find(|d| d.flag == "pflash-block-size")
            .unwrap();
        assert!(
            !d.tooltip.to_lowercase().contains("power of two"),
            "pflash-block-size tooltip must not invent a power-of-two constraint absent from `serve --help`"
        );
    }

    #[test]
    fn pflash_sink_and_query_window_tooltips_match_real_cli_help_direction() {
        let sink = ALLOWED_ESCAPE_FLAGS
            .iter()
            .find(|d| d.flag == "pflash-sink-tokens")
            .unwrap();
        assert!(
            sink.tooltip.to_lowercase().contains("leading"),
            "pflash-sink-tokens are leading tokens per `serve --help`, tooltip must say so"
        );

        let query_window = ALLOWED_ESCAPE_FLAGS
            .iter()
            .find(|d| d.flag == "pflash-query-window")
            .unwrap();
        assert!(
            query_window.tooltip.to_lowercase().contains("trailing"),
            "pflash-query-window is a trailing window per `serve --help`, tooltip must say so"
        );
    }
}
