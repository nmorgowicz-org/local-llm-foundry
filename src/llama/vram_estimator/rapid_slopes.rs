//! Receipt-based active KV cache slopes for Rapid-MLX (stabilized at 130k+ tokens).
//!
//! All slopes derived from hardware receipts using ONLY ModelArch fields —
//! no model names, no repo IDs, no string matching.
//!
//! Formula: slope = base_factor(dtype) × effective_kv(arch)
//!
//! Calibration sources:
//! - Hybrid DeltaNet dense: Qwen3.5/3.6-27B (16 attn, 4 KV, BF16→132000)
//! - Hybrid DeltaNet MoE: Qwen3.5/3.6-35B-A3B (10 attn, 2 KV, BF16→43100, 4% variance)
//! - Hybrid DeltaNet: Qwen3.5-9B (8 attn, 4 KV, BF16→66000)
//! - Sliding Window: Gemma4-26B (5 global, 2 KV, BF16→45900)
//! - MLA: base_factor exists but no receipt-calibrated model yet (heuristic detection only)

use crate::llama::vram_estimator::ModelArch;
use crate::llama::vram_estimator::execution_policy::KvCacheDtype;

/// Active KV cache bytes per token from architecture fields + calibrated overhead.
///
/// Zero name matching — works for finetunes/distills with renamed filenames.
pub fn rapid_active_kv_bytes_per_token(arch: &ModelArch, dtype: KvCacheDtype) -> f64 {
    if arch.is_hybrid_attn() {
        // Hybrid DeltaNet: only n_attn_layers use KV cache.
        // base_bf16 = 132400 / (16*4) = 2062.5
        // base_int8 = 103300 / (16*4) = 1614.1
        // base_int4 = 86100  / (16*4) = 1345.3
        //
        // The bases embed the calibration geometry (head_dim 256). Re-verified
        // 2026-10-04 against rapid-mlx 0.15.5 on a 16-attn/4-kv/head_dim-256
        // checkpoint: a single 185,576-token prefill peaked at ~127 B/token
        // bf16 — the ~2× factor over the naive K+V analytic (65,536) is real
        // runtime allocation, not measurement error. Models with a different
        // head_dim scale linearly from the calibrated geometry.
        let effective = arch.n_attn_layers.max(1) as f64 * arch.n_kv_heads.max(1) as f64;
        let base = match dtype {
            KvCacheDtype::Bf16 => 2062.5,
            KvCacheDtype::Int8 => 1614.1,
            KvCacheDtype::Int4 => 1345.3,
        };
        let head_dim_scale = if arch.head_dim > 0 {
            arch.head_dim as f64 / 256.0
        } else {
            1.0
        };
        base * effective * head_dim_scale
    } else if arch.has_local_attn() {
        // Sliding window: only global layers grow at long context.
        // base_bf16 = 45900 / (5*2) = 4590.0
        // base_int8 = 36300 / (5*2) = 3630.0
        // base_int4 = 30700 / (5*2) = 3070.0
        let effective = arch.n_global_attn_layers.max(1) as f64 * arch.n_kv_heads.max(1) as f64;
        let base = match dtype {
            KvCacheDtype::Bf16 => 4590.0,
            KvCacheDtype::Int8 => 3630.0,
            KvCacheDtype::Int4 => 3070.0,
        };
        base * effective
    } else if is_mla_pattern(arch) {
        // MLA: latent KV representation with low KV heads.
        // base_bf16 = 66000 / (28*4) = 589.3
        // base_int8 = 52600 / (28*4) = 469.6
        // base_int4 = 44600 / (28*4) = 398.2
        let effective = arch.n_layers.max(1) as f64 * arch.n_kv_heads.max(1) as f64;
        let base = match dtype {
            KvCacheDtype::Bf16 => 589.3,
            KvCacheDtype::Int8 => 469.6,
            KvCacheDtype::Int4 => 398.2,
        };
        base * effective
    } else {
        // Standard transformer: theoretical only.
        let elem_bytes = match dtype {
            KvCacheDtype::Bf16 => 2.0,
            KvCacheDtype::Int8 => 1.0,
            KvCacheDtype::Int4 => 0.5,
        };
        arch.n_layers.max(1) as f64
            * arch.n_kv_heads.max(1) as f64
            * arch.head_dim.max(1) as f64
            * elem_bytes
            * 2.0
    }
}

/// Detect MLA (Multi-Head Latent Attention) pattern from ModelArch fields only.
///
/// MLA models use a compressed latent representation for KV, detectable by:
/// - Very low n_kv_heads (typically 2–4) relative to model size
/// - Many layers (≥20) with standard attention (not hybrid/swa)
/// - n_kv_heads << n_head (extreme GQA/MQA compression)
///
/// This is a heuristic: some standard models share this profile, but the
/// overhead factor (2.30) will be absorbed within normal estimation variance.
fn is_mla_pattern(arch: &ModelArch) -> bool {
    // Must be non-hybrid, non-sliding-window, using all layers for KV.
    // Low KV head count is the strongest MLA signal.
    !arch.is_hybrid_attn() && !arch.has_local_attn() && arch.n_kv_heads <= 4 && arch.n_layers >= 20
}

/// Prefix cache bytes per token from architecture fields (Hybrid DeltaNet only).
///
/// Only Hybrid DeltaNet models have prefix cache data from receipts.
/// Slope derived from Qwen3.5/3.6-35B-A3B (MoE): INT8 = 23400, INT4 = 12700.
/// Formula scales with n_attn_layers × n_kv_heads; applied to all Hybrid DeltaNet models.
// Not yet called from the estimator; its own tests are the only callers, so the
// expectation holds for the lib build and would be unfulfilled under
// --all-targets if applied unconditionally.
#[cfg_attr(not(test), expect(dead_code))]
pub fn rapid_prefix_cache_bytes_per_token(arch: &ModelArch, dtype: KvCacheDtype) -> Option<f64> {
    // Only Hybrid DeltaNet has measured prefix cache data.
    if !arch.is_hybrid_attn() {
        return None;
    }

    // Qwen3.6-35B-A3B reference: n_attn=10, n_kv=2, INT8=23400, INT4=12700
    // Base = slope / (n_attn × n_kv)
    let base_int8 = 23_400.0 / (10.0 * 2.0); // 1170
    let base_int4 = 12_700.0 / (10.0 * 2.0); // 635

    let effective = arch.n_attn_layers.max(1) as f64 * arch.n_kv_heads.max(1) as f64;

    Some(match dtype {
        KvCacheDtype::Bf16 => None?, // No BF16 prefix cache receipt
        KvCacheDtype::Int8 => effective * base_int8,
        KvCacheDtype::Int4 => effective * base_int4,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Calibrated base factors (from comments in main code).
    const HYBRID_BF16_BASE: f64 = 2062.5;
    const SW_BF16_BASE: f64 = 4590.0;

    fn qwen36_27b_arch() -> ModelArch {
        // Hybrid DeltaNet: 64 total, 16 attn layers, 4 KV heads, hd=256
        ModelArch {
            n_layers: 64,
            n_attn_layers: 16,
            n_kv_heads: 4,
            head_dim: 256,
            ..Default::default()
        }
    }

    fn qwen36_35b_a3b_arch() -> ModelArch {
        // Hybrid DeltaNet: 40 total, 10 attn layers, 2 KV heads, hd=256
        ModelArch {
            n_layers: 40,
            n_attn_layers: 10,
            n_kv_heads: 2,
            head_dim: 256,
            ..Default::default()
        }
    }

    fn qwen35_9b_arch() -> ModelArch {
        // Hybrid DeltaNet (NOT MLA): 32 layers, 8 attn (interval=4), 4 KV heads, hd=256.
        // HF config: qwen3_5.text_config with full_attention_interval=4.
        ModelArch {
            n_layers: 32,
            n_attn_layers: 8,
            n_kv_heads: 4,
            head_dim: 256,
            ..Default::default()
        }
    }

    fn gemma4_26b_a4b_arch() -> ModelArch {
        // Sliding window: 30 layers, 5 global attn, 2 KV heads, ghd=512, window=1024
        ModelArch {
            n_layers: 30,
            n_global_attn_layers: 5,
            n_kv_heads: 2,
            head_dim: 256,
            global_head_dim: 512,
            local_attn_window: 1024,
            local_kv_heads: 8,
            ..Default::default()
        }
    }

    fn gemma4_31b_arch() -> ModelArch {
        // Sliding window: 60 layers, 10 global attn, 4 KV heads, ghd=512, window=1024.
        // HF config: google/gemma-4-31B-it, text_config.num_global_key_value_heads=4,
        // text_config.num_key_value_heads=16 (total), layer_types: 10 full + 50 sliding.
        ModelArch {
            n_layers: 60,
            n_global_attn_layers: 10,
            n_kv_heads: 4,
            head_dim: 256,
            global_head_dim: 512,
            local_attn_window: 1024,
            local_kv_heads: 12,
            ..Default::default()
        }
    }

    fn llama70b_arch() -> ModelArch {
        // Standard transformer: no special attention
        ModelArch {
            n_layers: 80,
            n_kv_heads: 8,
            head_dim: 128,
            ..Default::default()
        }
    }

    // ── Hybrid DeltaNet: Qwen3.5/3.6-27B dense (calibration anchor) ────────────
    // Same arch: 64 layers, 16 attn, 4 KV heads, hd=256. Dense, not MoE.

    #[test]
    fn qwen3x_27b_dense_bf16_exact() {
        let bytes = rapid_active_kv_bytes_per_token(&qwen36_27b_arch(), KvCacheDtype::Bf16);
        // Calibrated: 2062.5 × 16 × 4 = 132000
        assert_eq!(bytes, 132_000.0, "BF16 calibration anchor must be exact");
    }

    #[test]
    fn qwen3x_27b_dense_int8_exact() {
        let bytes = rapid_active_kv_bytes_per_token(&qwen36_27b_arch(), KvCacheDtype::Int8);
        // Calibrated: 1614.1 × 16 × 4 = 103302.4
        assert_eq!(bytes, 103_302.4, "INT8 calibration anchor must be exact");
    }

    #[test]
    fn qwen3x_27b_dense_int4_exact() {
        let bytes = rapid_active_kv_bytes_per_token(&qwen36_27b_arch(), KvCacheDtype::Int4);
        // Calibrated: 1345.3 × 16 × 4 = 86099.2
        assert_eq!(bytes, 86_099.2, "INT4 calibration anchor must be exact");
    }

    // ── Hybrid DeltaNet: Qwen3.5/3.6-35B-A3B MoE ──────────────────────────────
    // MoE (256 experts, 8 active), NOT dense. 40 layers, 10 attn, 2 KV heads, hd=256.
    // Receipts: unsloth-qwen36-35b-source-5fc6556c-context-512/ (with KV dtype fix).
    // Same base_factor as dense 27B; MoE variance is within tolerance.

    #[test]
    fn qwen3x_35b_moe_bf16_scaled() {
        let bytes = rapid_active_kv_bytes_per_token(&qwen36_35b_a3b_arch(), KvCacheDtype::Bf16);
        // Formula: 2062.5 × 10 × 2 = 41250 vs receipt 43411 (5.2% under — MoE overhead)
        assert!(
            (bytes - 43_411.0).abs() / 43_411.0 < 0.07,
            "Qwen3x-35B MoE BF16: got {bytes:.0}, expected ~43411 (±7%)"
        );
    }

    #[test]
    fn qwen3x_35b_moe_int8_scaled() {
        let bytes = rapid_active_kv_bytes_per_token(&qwen36_35b_a3b_arch(), KvCacheDtype::Int8);
        // Formula: 1614.1 × 10 × 2 = 32282 vs receipt 33600 (3.9% under)
        assert!(
            (bytes - 33_600.0).abs() / 33_600.0 < 0.06,
            "Qwen3x-35B MoE INT8: got {bytes:.0}, expected ~33600 (±6%)"
        );
    }

    #[test]
    fn qwen3x_35b_moe_int4_scaled() {
        let bytes = rapid_active_kv_bytes_per_token(&qwen36_35b_a3b_arch(), KvCacheDtype::Int4);
        // Formula: 1345.3 × 10 × 2 = 26906 vs receipt 29755 (9.6% under — INT4 overhead)
        assert!(
            (bytes - 29_755.0).abs() / 29_755.0 < 0.12,
            "Qwen3x-35B MoE INT4: got {bytes:.0}, expected ~29755 (±12%)"
        );
    }

    // ── Hybrid DeltaNet: Qwen3.5-9B (calibration anchor) ──────────────────────

    #[test]
    fn qwen35_9b_bf16_exact() {
        let bytes = rapid_active_kv_bytes_per_token(&qwen35_9b_arch(), KvCacheDtype::Bf16);
        // Calibrated: 2062.5 × 8 × 4 = 66000 (receipt qwen35-9b-source-5fc6556c-context-512/)
        assert_eq!(bytes, 66_000.0, "BF16 calibration anchor must be exact");
    }

    #[test]
    fn qwen35_9b_int8_exact() {
        let bytes = rapid_active_kv_bytes_per_token(&qwen35_9b_arch(), KvCacheDtype::Int8);
        // Calibrated: 1614.1 × 8 × 4 = 51651.2 vs receipt 52600 (1.8% under)
        assert!(
            (bytes - 52_600.0).abs() / 52_600.0 < 0.03,
            "Qwen3.5-9B INT8: got {bytes:.0}, expected ~52600 (±3%)"
        );
    }

    #[test]
    fn qwen35_9b_int4_exact() {
        let bytes = rapid_active_kv_bytes_per_token(&qwen35_9b_arch(), KvCacheDtype::Int4);
        // Calibrated: 1345.3 × 8 × 4 = 43049.6 vs receipt 44600 (3.5% under)
        assert!(
            (bytes - 44_600.0).abs() / 44_600.0 < 0.04,
            "Qwen3.5-9B INT4: got {bytes:.0}, expected ~44600 (±4%)"
        );
    }

    // ── Sliding Window: Gemma4-26B-A4B (calibration anchor) ───────────────────

    #[test]
    fn gemma4_26b_a4b_bf16_exact() {
        let bytes = rapid_active_kv_bytes_per_token(&gemma4_26b_a4b_arch(), KvCacheDtype::Bf16);
        // Calibrated: 4590.0 × 5 × 2 = 45900
        assert_eq!(bytes, 45_900.0, "BF16 calibration anchor must be exact");
    }

    #[test]
    fn gemma4_26b_a4b_int8_exact() {
        let bytes = rapid_active_kv_bytes_per_token(&gemma4_26b_a4b_arch(), KvCacheDtype::Int8);
        // Calibrated: 3630.0 × 5 × 2 = 36300
        assert_eq!(bytes, 36_300.0, "INT8 calibration anchor must be exact");
    }

    #[test]
    fn gemma4_26b_a4b_int4_exact() {
        let bytes = rapid_active_kv_bytes_per_token(&gemma4_26b_a4b_arch(), KvCacheDtype::Int4);
        // Calibrated: 3070.0 × 5 × 2 = 30700
        assert_eq!(bytes, 30_700.0, "INT4 calibration anchor must be exact");
    }

    // ── Sliding Window: Gemma4-31B Dense (scales from 26B anchor) ─────────────
    // Dense: 60 layers, 10 global attn, 4 KV heads → slope = 4590 × 10 × 4 = 183600
    // 4× the 26B-A4B slope (2× more global layers × 2× more KV heads).
    // No receipts yet (model released 2026-07), but formula is calibrated on 26B.

    #[test]
    fn gemma4_31b_bf16_scaled_from_26b() {
        let bytes = rapid_active_kv_bytes_per_token(&gemma4_31b_arch(), KvCacheDtype::Bf16);
        // Formula: 4590.0 × 10 × 4 = 183600 (4× the 26B anchor)
        assert_eq!(bytes, 183_600.0, "BF16 must scale from 26B anchor");
    }

    #[test]
    fn gemma4_31b_int8_scaled_from_26b() {
        let bytes = rapid_active_kv_bytes_per_token(&gemma4_31b_arch(), KvCacheDtype::Int8);
        // Formula: 3630.0 × 10 × 4 = 145200 (4× the 26B anchor)
        assert_eq!(bytes, 145_200.0, "INT8 must scale from 26B anchor");
    }

    #[test]
    fn gemma4_31b_int4_scaled_from_26b() {
        let bytes = rapid_active_kv_bytes_per_token(&gemma4_31b_arch(), KvCacheDtype::Int4);
        // Formula: 3070.0 × 10 × 4 = 122800 (4× the 26B anchor)
        assert_eq!(bytes, 122_800.0, "INT4 must scale from 26B anchor");
    }

    // ── Finetune detection: renamed derivative uses arch, not name ────────────

    #[test]
    fn finetune_with_different_name_same_arch() {
        // A "Pantheon-27B" finetune based on Qwen3.5/3.6 has identical arch fields
        // but completely different name. Must match via arch, not name.
        let pantheon_arch = qwen36_27b_arch();
        let bytes = rapid_active_kv_bytes_per_token(&pantheon_arch, KvCacheDtype::Int4);
        // Same as dense 27B: 86099.2
        assert!(
            (bytes - 86_100.0).abs() < 100.0,
            "Finetune with 27B arch: got {bytes:.0}, expected ~86100"
        );
    }

    // ── Standard transformer: theoretical fallback ────────────────────────────

    #[test]
    fn standard_transformer_uses_theoretical() {
        let bytes = rapid_active_kv_bytes_per_token(&llama70b_arch(), KvCacheDtype::Int8);
        // Theoretical: 80*8*128*1*2 = 163840
        assert_eq!(bytes, 163_840.0);
    }

    // ── MLA detection logic (untested by receipts, heuristic only) ────────────

    #[test]
    fn mla_detection_low_kv_heads_many_layers() {
        // Synthetic MLA candidate: no hybrid/swa, low KV heads, many layers.
        let mla_candidate = ModelArch {
            n_layers: 28,
            n_kv_heads: 4,
            head_dim: 128,
            ..Default::default()
        };
        assert!(
            is_mla_pattern(&mla_candidate),
            "28 layers, 4 KV heads, no hybrid should be MLA pattern"
        );
    }

    #[test]
    fn mla_detection_not_hybrid() {
        let arch = qwen36_27b_arch();
        assert!(
            !is_mla_pattern(&arch),
            "Hybrid DeltaNet should NOT be detected as MLA"
        );
    }

    #[test]
    fn mla_detection_qwen35_9b_is_hybrid_not_mla() {
        // Qwen3.5-9B has full_attention_interval=4 → Hybrid DeltaNet, NOT MLA.
        let arch = qwen35_9b_arch();
        assert!(
            !is_mla_pattern(&arch),
            "Qwen3.5-9B is Hybrid DeltaNet (8 attn, 32 layers), not MLA"
        );
    }

    #[test]
    fn mla_detection_not_sliding_window() {
        let arch = gemma4_26b_a4b_arch();
        assert!(
            !is_mla_pattern(&arch),
            "Sliding window should NOT be detected as MLA"
        );
    }

    #[test]
    fn mla_detection_small_model_ignored() {
        // Small model with low KV heads is not MLA.
        let small_arch = ModelArch {
            n_layers: 8,
            n_kv_heads: 2,
            head_dim: 128,
            ..Default::default()
        };
        assert!(
            !is_mla_pattern(&small_arch),
            "8-layer model should not be MLA (too small)"
        );
    }

    // ── Prefix cache: Hybrid DeltaNet only ────────────────────────────────────

    #[test]
    fn prefix_cache_qwen36_35b_int8() {
        let slope = rapid_prefix_cache_bytes_per_token(&qwen36_35b_a3b_arch(), KvCacheDtype::Int8);
        assert_eq!(slope, Some(23_400.0));
    }

    #[test]
    fn prefix_cache_qwen36_35b_int4() {
        let slope = rapid_prefix_cache_bytes_per_token(&qwen36_35b_a3b_arch(), KvCacheDtype::Int4);
        assert_eq!(slope, Some(12_700.0));
    }

    #[test]
    fn prefix_cache_qwen36_35b_bf16_none() {
        let slope = rapid_prefix_cache_bytes_per_token(&qwen36_35b_a3b_arch(), KvCacheDtype::Bf16);
        assert!(slope.is_none());
    }

    #[test]
    fn prefix_cache_qwen36_27b_scaled() {
        // Qwen3.6-27B has n_attn=16, n_kv=4 → effective = 64 (vs 20 for 35B-A3B)
        // Scale factor: 64/20 = 3.2
        let slope_int8 =
            rapid_prefix_cache_bytes_per_token(&qwen36_27b_arch(), KvCacheDtype::Int8).unwrap();
        let slope_int4 =
            rapid_prefix_cache_bytes_per_token(&qwen36_27b_arch(), KvCacheDtype::Int4).unwrap();
        assert_eq!(slope_int8, 23_400.0 * 3.2);
        assert_eq!(slope_int4, 12_700.0 * 3.2);
    }

    #[test]
    fn prefix_cache_non_hybrid_none() {
        // Standard transformer: no prefix cache receipt
        let slope = rapid_prefix_cache_bytes_per_token(&llama70b_arch(), KvCacheDtype::Int8);
        assert!(slope.is_none());

        // Sliding window: no prefix cache receipt
        let slope = rapid_prefix_cache_bytes_per_token(&gemma4_26b_a4b_arch(), KvCacheDtype::Int4);
        assert!(slope.is_none());
    }

    #[test]
    fn prefix_cache_qwen35_9b_scaled() {
        // Qwen3.5-9B (Hybrid): n_attn=8, n_kv=4 → effective = 32 (vs 20 for 35B-A3B)
        // Scale factor: 32/20 = 1.6
        let slope_int8 =
            rapid_prefix_cache_bytes_per_token(&qwen35_9b_arch(), KvCacheDtype::Int8).unwrap();
        let slope_int4 =
            rapid_prefix_cache_bytes_per_token(&qwen35_9b_arch(), KvCacheDtype::Int4).unwrap();
        assert_eq!(slope_int8, 23_400.0 * 1.6);
        assert_eq!(slope_int4, 12_700.0 * 1.6);
    }

    #[test]
    fn prefix_cache_qwen36_8g_16g_capacity() {
        let int8_slope =
            rapid_prefix_cache_bytes_per_token(&qwen36_35b_a3b_arch(), KvCacheDtype::Int8).unwrap();
        let int4_slope =
            rapid_prefix_cache_bytes_per_token(&qwen36_35b_a3b_arch(), KvCacheDtype::Int4).unwrap();

        let gb = 1_000_000_000.0;

        let int8_8g = (8.0 * gb) / int8_slope;
        let int8_16g = (16.0 * gb) / int8_slope;
        let int4_8g = (8.0 * gb) / int4_slope;
        let int4_16g = (16.0 * gb) / int4_slope;

        assert!(
            (int8_8g - 342_000.0).abs() < 5_000.0,
            "INT8 8G capacity {int8_8g:.0} tokens, expected ~342K"
        );
        assert!(
            (int8_16g - 684_000.0).abs() < 5_000.0,
            "INT8 16G capacity {int8_16g:.0} tokens, expected ~684K"
        );
        assert!(
            (int4_8g - 630_000.0).abs() < 5_000.0,
            "INT4 8G capacity {int4_8g:.0} tokens, expected ~630K"
        );
        assert!(
            (int4_16g - 1_260_000.0).abs() < 5_000.0,
            "INT4 16G capacity {int4_16g:.0} tokens, expected ~1,260K"
        );
    }

    // ── Standard contexts × dtypes × architectures (DoD item 6) ──────────────

    fn standard_contexts() -> [u64; 6] {
        [32_768, 65_536, 131_072, 163_840, 200_000, 262_144]
    }

    #[test]
    fn hybrid_deltanet_all_contexts_all_dtypes_monotonic_kv() {
        let arch = qwen36_27b_arch();
        let contexts = standard_contexts();
        let prev = &mut [0.0f64; 3];

        for &ctx in &contexts {
            for (idx, dtype) in [KvCacheDtype::Bf16, KvCacheDtype::Int8, KvCacheDtype::Int4]
                .iter()
                .enumerate()
            {
                let slope = rapid_active_kv_bytes_per_token(&arch, *dtype);
                let kv_bytes = slope * ctx as f64;
                let pval = prev[idx];
                assert!(
                    kv_bytes > pval,
                    "KV for {:?} at {} tokens ({:.0}B) not > previous ({:.0}B)",
                    dtype,
                    ctx,
                    kv_bytes,
                    pval
                );
                prev[idx] = kv_bytes;
            }
        }
    }

    #[test]
    fn hybrid_deltanet_all_contexts_dtype_ratios_reasonable() {
        let arch = qwen36_27b_arch();
        let bf16_slope = rapid_active_kv_bytes_per_token(&arch, KvCacheDtype::Bf16);
        let int8_slope = rapid_active_kv_bytes_per_token(&arch, KvCacheDtype::Int8);
        let int4_slope = rapid_active_kv_bytes_per_token(&arch, KvCacheDtype::Int4);

        assert!(
            bf16_slope > int8_slope && int8_slope > int4_slope,
            "BF16 > INT8 > INT4 required"
        );
        let bf16_int8_ratio = bf16_slope / int8_slope;
        let int8_int4_ratio = int8_slope / int4_slope;
        // Receipt-based slopes: BF16/INT8 ~1.28, INT8/INT4 ~1.20 (overhead-dominated).
        assert!(
            bf16_int8_ratio > 1.2 && bf16_int8_ratio < 1.6,
            "BF16/INT8 ratio {:.3} out of range",
            bf16_int8_ratio
        );
        assert!(
            int8_int4_ratio > 1.1 && int8_int4_ratio < 1.5,
            "INT8/INT4 ratio {:.3} out of range",
            int8_int4_ratio
        );
    }

    #[test]
    fn hybrid_deltanet_context_262k_active_kv_scales_correctly() {
        let arch = qwen36_27b_arch();
        let bf16_slope = rapid_active_kv_bytes_per_token(&arch, KvCacheDtype::Bf16);
        let kv_262k = bf16_slope * 262_144.0;
        let gb = kv_262k / (1024.0 * 1024.0 * 1024.0);
        assert!(
            gb > 20.0 && gb < 60.0,
            "262K BF16 KV for Qwen3.6-27B: {gb:.2} GB"
        );
    }

    #[test]
    fn sliding_window_all_contexts_all_dtypes_monotonic_kv() {
        let arch = gemma4_26b_a4b_arch();
        let contexts = standard_contexts();
        let prev = &mut [0.0f64; 3];

        for &ctx in &contexts {
            for (idx, dtype) in [KvCacheDtype::Bf16, KvCacheDtype::Int8, KvCacheDtype::Int4]
                .iter()
                .enumerate()
            {
                let slope = rapid_active_kv_bytes_per_token(&arch, *dtype);
                let kv_bytes = slope * ctx as f64;
                let pval = prev[idx];
                assert!(
                    kv_bytes > pval,
                    "KV for {:?} at {} tokens ({:.0}B) not > previous ({:.0}B)",
                    dtype,
                    ctx,
                    kv_bytes,
                    pval
                );
                prev[idx] = kv_bytes;
            }
        }
    }

    #[test]
    fn sliding_window_all_contexts_dtype_ratios_reasonable() {
        let arch = gemma4_26b_a4b_arch();
        let bf16_slope = rapid_active_kv_bytes_per_token(&arch, KvCacheDtype::Bf16);
        let int8_slope = rapid_active_kv_bytes_per_token(&arch, KvCacheDtype::Int8);
        let int4_slope = rapid_active_kv_bytes_per_token(&arch, KvCacheDtype::Int4);

        assert!(
            bf16_slope > int8_slope && int8_slope > int4_slope,
            "BF16 > INT8 > INT4 required"
        );
        let bf16_int8_ratio = bf16_slope / int8_slope;
        let int8_int4_ratio = int8_slope / int4_slope;
        // Receipt-based slopes: similar to hybrid, overhead-dominated.
        assert!(
            bf16_int8_ratio > 1.2 && bf16_int8_ratio < 1.6,
            "BF16/INT8 ratio {:.3} out of range",
            bf16_int8_ratio
        );
        assert!(
            int8_int4_ratio > 1.1 && int8_int4_ratio < 1.5,
            "INT8/INT4 ratio {:.3} out of range",
            int8_int4_ratio
        );
    }

    #[test]
    fn sliding_window_context_262k_active_kv_scales_correctly() {
        let arch = gemma4_26b_a4b_arch();
        let bf16_slope = rapid_active_kv_bytes_per_token(&arch, KvCacheDtype::Bf16);
        let kv_262k = bf16_slope * 262_144.0;
        let gb = kv_262k / (1024.0 * 1024.0 * 1024.0);
        assert!(
            gb > 10.0 && gb < 40.0,
            "262K BF16 KV for Gemma4-26B: {gb:.2} GB"
        );
    }

    #[test]
    fn standard_transformer_all_contexts_all_dtypes_monotonic_kv() {
        let arch = llama70b_arch();
        let contexts = standard_contexts();
        let prev = &mut [0.0f64; 3];

        for &ctx in &contexts {
            for (idx, dtype) in [KvCacheDtype::Bf16, KvCacheDtype::Int8, KvCacheDtype::Int4]
                .iter()
                .enumerate()
            {
                let slope = rapid_active_kv_bytes_per_token(&arch, *dtype);
                let kv_bytes = slope * ctx as f64;
                let pval = prev[idx];
                assert!(
                    kv_bytes > pval,
                    "KV for {:?} at {} tokens ({:.0}B) not > previous ({:.0}B)",
                    dtype,
                    ctx,
                    kv_bytes,
                    pval
                );
                prev[idx] = kv_bytes;
            }
        }
    }

    #[test]
    fn standard_transformer_all_contexts_dtype_ratios_reasonable() {
        let arch = llama70b_arch();
        let bf16_slope = rapid_active_kv_bytes_per_token(&arch, KvCacheDtype::Bf16);
        let int8_slope = rapid_active_kv_bytes_per_token(&arch, KvCacheDtype::Int8);
        let int4_slope = rapid_active_kv_bytes_per_token(&arch, KvCacheDtype::Int4);

        assert!(
            bf16_slope > int8_slope && int8_slope > int4_slope,
            "BF16 > INT8 > INT4 required"
        );
        let bf16_int8_ratio = bf16_slope / int8_slope;
        let int8_int4_ratio = int8_slope / int4_slope;
        // Theoretical: BF16 uses 2 bytes, INT8 uses 1, INT4 uses 0.5 → exact 2:1:0.5.
        assert_eq!(
            bf16_int8_ratio, 2.0,
            "Standard BF16/INT8 should be exactly 2.0"
        );
        assert_eq!(
            int8_int4_ratio, 2.0,
            "Standard INT8/INT4 should be exactly 2.0"
        );
    }

    #[test]
    fn all_architectures_hybrid_uses_attn_layers_not_all_layers() {
        let arch = qwen36_27b_arch();
        assert!(
            arch.n_attn_layers < arch.n_layers,
            "Hybrid must have fewer attn layers"
        );
        let slope = rapid_active_kv_bytes_per_token(&arch, KvCacheDtype::Bf16);
        let expected = HYBRID_BF16_BASE * arch.n_attn_layers as f64 * arch.n_kv_heads as f64;
        assert!(
            (slope - expected).abs() < 1.0,
            "Hybrid slope {slope} should use n_attn_layers={}",
            arch.n_attn_layers
        );
    }

    #[test]
    fn all_architectures_sliding_window_uses_global_attn_layers() {
        let arch = gemma4_26b_a4b_arch();
        assert!(
            arch.n_global_attn_layers < arch.n_layers,
            "SW must have fewer global attn layers"
        );
        let slope = rapid_active_kv_bytes_per_token(&arch, KvCacheDtype::Bf16);
        let expected = SW_BF16_BASE * arch.n_global_attn_layers as f64 * arch.n_kv_heads as f64;
        assert!(
            (slope - expected).abs() < 1.0,
            "Sliding window slope {slope} should use n_global_attn_layers={}",
            arch.n_global_attn_layers
        );
    }

    #[test]
    fn hybrid_slope_scales_with_head_dim() {
        let mut arch = qwen36_27b_arch();
        assert!(
            arch.head_dim == 256,
            "calibration fixture must carry head_dim 256"
        );
        let calibrated = rapid_active_kv_bytes_per_token(&arch, KvCacheDtype::Bf16);
        // A same-family model with a doubled head width pays double per token.
        arch.head_dim = 512;
        let doubled = rapid_active_kv_bytes_per_token(&arch, KvCacheDtype::Bf16);
        assert!((doubled - calibrated * 2.0).abs() < 1.0);
        // Unknown head width falls back to the calibrated geometry.
        arch.head_dim = 0;
        assert!(
            (rapid_active_kv_bytes_per_token(&arch, KvCacheDtype::Bf16) - calibrated).abs() < 1.0
        );
    }
}
