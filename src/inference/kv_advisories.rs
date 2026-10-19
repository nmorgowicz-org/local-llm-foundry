//! Model-family K/V guidance. Information only: an advisory never blocks a
//! launch and never changes the classification. Every entry names where it
//! came from, and no entry claims more than its source supports.
//!
//! Plan: `docs/archive/llamacpp/20261019-llamacpp_compilation.md` §4.8.

use serde::Serialize;

use crate::inference::kv_support::KvType;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Info,
    Caution,
}

/// What the advisory matches on. Only facts read from the model itself or its
/// server-resolved file name; never a client-supplied string.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelFacts {
    /// GGUF `general.architecture`, when readable.
    pub architecture: Option<String>,
    /// File name (no directory) of the server-resolved model.
    pub file_name: Option<String>,
}

#[derive(Debug, Clone, Copy)]
enum Matcher {
    /// GGUF architecture starts with a prefix (e.g. `gemma4`).
    ArchitecturePrefix(&'static str),
    /// File name contains every listed fragment (case-insensitive).
    FileNameContainsAll(&'static [&'static str]),
}

#[derive(Debug, Clone, Copy)]
enum Predicate {
    /// K or V is quantized (not f32/f16/bf16).
    AnyQuantized,
    /// Exactly this pair.
    ExactPair(KvType, KvType),
}

struct Entry {
    id: &'static str,
    matchers: &'static [Matcher],
    applies_to: Predicate,
    severity: Severity,
    text: &'static str,
    source: &'static str,
}

const TABLE: &[Entry] = &[
    Entry {
        id: "gemma4-quantized-kv",
        // Either an architecture match or a Gemma 4 file name is enough.
        matchers: &[
            Matcher::ArchitecturePrefix("gemma4"),
            Matcher::FileNameContainsAll(&["gemma-4"]),
            Matcher::FileNameContainsAll(&["gemma4"]),
        ],
        applies_to: Predicate::AnyQuantized,
        severity: Severity::Caution,
        text: "Gemma 4 can be sensitive to quantizing either K or V. Prefer conservative cache precision and check output quality on your own prompts.",
        source: "Maintainer/user guidance (qualitative); not a measured severity and not advice about other Gemma generations.",
    },
    Entry {
        id: "qwen3.8-27b-q8-q4",
        matchers: &[Matcher::FileNameContainsAll(&["qwen3.8", "27b"])],
        applies_to: Predicate::ExactPair(KvType::Q8_0, KvType::Q4_0),
        severity: Severity::Info,
        text: "The maintainer reports K=q8_0 / V=q4_0 as a daily configuration on this model with acceptable quality. Validate it for your own workload.",
        source: "Maintainer observation, 2026-10; not a controlled quality measurement and not a claim about other Qwen models.",
    },
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct KvAdvisory {
    pub id: &'static str,
    pub severity: Severity,
    pub text: &'static str,
    pub source: &'static str,
}

fn matches(matchers: &[Matcher], facts: &ModelFacts) -> bool {
    let arch = facts.architecture.as_deref().map(str::to_ascii_lowercase);
    let name = facts.file_name.as_deref().map(str::to_ascii_lowercase);
    matchers.iter().any(|matcher| match matcher {
        Matcher::ArchitecturePrefix(prefix) => {
            arch.as_deref().is_some_and(|a| a.starts_with(prefix))
        }
        Matcher::FileNameContainsAll(fragments) => name
            .as_deref()
            .is_some_and(|n| fragments.iter().all(|f| n.contains(f))),
    })
}

fn applies(predicate: Predicate, k: KvType, v: KvType) -> bool {
    match predicate {
        Predicate::AnyQuantized => k.is_quantized() || v.is_quantized(),
        Predicate::ExactPair(ek, ev) => k == ek && v == ev,
    }
}

/// Advisories for a pair on a model. `None` facts (no model metadata) yield
/// nothing: guidance is never given without knowing the model.
pub fn advisories_for(k: KvType, v: KvType, facts: Option<&ModelFacts>) -> Vec<KvAdvisory> {
    let Some(facts) = facts else {
        return Vec::new();
    };
    TABLE
        .iter()
        .filter(|entry| matches(entry.matchers, facts) && applies(entry.applies_to, k, v))
        .map(|entry| KvAdvisory {
            id: entry.id,
            severity: entry.severity,
            text: entry.text,
            source: entry.source,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(arch: Option<&str>, name: Option<&str>) -> ModelFacts {
        ModelFacts {
            architecture: arch.map(Into::into),
            file_name: name.map(Into::into),
        }
    }

    fn ids(k: KvType, v: KvType, f: &ModelFacts) -> Vec<&'static str> {
        advisories_for(k, v, Some(f)).iter().map(|a| a.id).collect()
    }

    #[test]
    fn gemma4_warns_for_k_only_v_only_and_both_quantized() {
        let gemma = facts(Some("gemma4"), Some("gemma-4-26B-A4B-it-MTP-Q8_0.gguf"));
        assert_eq!(
            ids(KvType::Q8_0, KvType::F16, &gemma),
            ["gemma4-quantized-kv"]
        );
        assert_eq!(
            ids(KvType::F16, KvType::Q8_0, &gemma),
            ["gemma4-quantized-kv"]
        );
        assert_eq!(
            ids(KvType::Q8_0, KvType::Q4_0, &gemma),
            ["gemma4-quantized-kv"]
        );
        assert_eq!(
            ids(KvType::Q4_0, KvType::Q4_0, &gemma),
            ["gemma4-quantized-kv"]
        );
    }

    #[test]
    fn gemma4_is_quiet_for_floating_pairs() {
        let gemma = facts(Some("gemma4"), None);
        for (k, v) in [
            (KvType::F16, KvType::F16),
            (KvType::Bf16, KvType::F16),
            (KvType::F32, KvType::Bf16),
        ] {
            assert!(ids(k, v, &gemma).is_empty(), "{k:?}/{v:?}");
        }
    }

    #[test]
    fn gemma4_matches_by_architecture_or_file_name_but_not_other_generations() {
        let by_arch = facts(Some("gemma4_moe"), None);
        let by_name = facts(None, Some("My-Gemma-4-12B-QAT-Q4_0.gguf"));
        assert_eq!(ids(KvType::Q8_0, KvType::Q8_0, &by_arch).len(), 1);
        assert_eq!(ids(KvType::Q8_0, KvType::Q8_0, &by_name).len(), 1);
        for other in [
            facts(Some("gemma3"), Some("gemma-3-27b-it.gguf")),
            facts(Some("gemma2"), None),
            facts(Some("gemma"), Some("gemma-7b.gguf")),
        ] {
            assert!(
                ids(KvType::Q8_0, KvType::Q8_0, &other).is_empty(),
                "{other:?}"
            );
        }
    }

    #[test]
    fn qwen38_27b_observation_applies_to_that_model_and_that_pair_only() {
        let qwen = facts(Some("qwen35"), Some("Qwen3.8-27B-IQ4_XS-3.84bpw.gguf"));
        assert_eq!(
            ids(KvType::Q8_0, KvType::Q4_0, &qwen),
            ["qwen3.8-27b-q8-q4"]
        );
        assert!(ids(KvType::Q8_0, KvType::Q8_0, &qwen).is_empty());
        assert!(
            ids(KvType::Q4_0, KvType::Q8_0, &qwen).is_empty(),
            "reverse pair is not reported"
        );
    }

    #[test]
    fn no_broad_qwen_family_evidence_is_invented() {
        for other in [
            facts(Some("qwen35"), Some("Qwen3.6-35B-A3B-Q5_K_M.gguf")),
            facts(Some("qwen3"), Some("Qwen3-27B.gguf")),
            facts(Some("qwen35"), Some("Qwen3.8-35B-A3B.gguf")),
            facts(Some("qwen35"), None),
        ] {
            assert!(
                ids(KvType::Q8_0, KvType::Q4_0, &other).is_empty(),
                "{other:?}"
            );
        }
    }

    #[test]
    fn no_model_metadata_means_no_advisories() {
        assert!(advisories_for(KvType::Q8_0, KvType::Q4_0, None).is_empty());
        assert!(ids(KvType::Q8_0, KvType::Q4_0, &ModelFacts::default()).is_empty());
    }

    #[test]
    fn every_entry_states_its_source_and_qualifies_its_claim() {
        for entry in TABLE {
            assert!(!entry.source.is_empty(), "{}", entry.id);
            assert!(!entry.text.contains("always"), "{}", entry.id);
        }
        let gemma = facts(Some("gemma4"), None);
        let advisory = &advisories_for(KvType::Q8_0, KvType::Q8_0, Some(&gemma))[0];
        assert_eq!(advisory.severity, Severity::Caution);
    }
}
