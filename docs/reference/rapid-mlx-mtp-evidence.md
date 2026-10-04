# Rapid-MLX MTP Speculative Decoding — Evidence Record

| Field | Value |
| --- | --- |
| Status | Durable evidence record. Measurements only; no product decisions. |
| Runtime | rapid-mlx 0.11.1, pinned upstream commit [`206ed0e`](https://github.com/raullenchai/Rapid-MLX/tree/206ed0e03b7b6fc7b3b2e6f68a7b60467f6e5abe) |
| Stack | mlx 0.32.0, mlx_lm 0.31.3, Python 3.11 |
| Hardware | Apple Silicon M5 Max (single machine, single operator) |
| First captured | 2026-07-29 |
| Companions | Product plan, live state, and open items: Phase 6.5 of `docs/archive/rapid-mlx/20260718-final_rapidmlx_followups.md`. Cross-runtime audit: `docs/reference/apple-silicon-mtp-runtime-comparison.md`. The former working-handoff doc was folded into those two and deleted on 2026-07-30 (git `396644b`). |

This document holds what was **measured** and what the **source says**. It deliberately
contains no roadmap, no UI design, and no enablement policy. Those live in Phase 6.5.

> **Reading the percentages.** Every throughput figure below is a **speedup ratio**
> (`new/old - 1`), not a fraction of time saved. "+59.4% generation" means 1.594× the
> tokens per second, which corresponds to 37% less generation time. Mixing the two
> conventions is the easiest way to misquote this document.

> **State of Rapid-MLX speculative decoding, as of 2026-07-30 — read this before quoting
> any number below.** MTP is **not usable for real agent traffic on any shipped build**, and
> there is **no ETA on a build where it will be**. Rapid-MLX 0.11.1 engages the MTP path only
> for a greedy request carrying no logits processor, so a request at any temperature above
> zero, and any ordinary tool-enabled request (Rapid builds a grammar processor by default),
> falls through to plain autoregressive decode and records zero speculative attempts. This is
> an explicit contract in the installed source, unchanged on upstream `main`, and tracked as
> upstream issue `#1013` — see §4.6. Everything measured here is therefore an **artificial
> greedy research envelope**, not a workload the product can expose. The app ships with MTP
> off for everyone and does not advertise it.
>
> **What would change that, and the only thing that would:** run
> `scripts/rapid-mlx-requalify-spec-decode.mjs` (zero arguments) against a new build. Exit `0`
> means its `sampled`, `constrained`, and `parity` gates all passed and the state above is
> stale; exit `20` means still blocked and nothing here changes; exit `1` means the harness
> broke, which is not the same finding. Full procedure in §12. Because of this, Phase 6.5a of
> the product plan was **closed on 2026-07-30** rather than left open indefinitely; the lane is
> the reopening mechanism.

---

## 1. Root cause 1 — stale extractor produced a dead draft head

A pre-fix copy of upstream `scripts/extract_mtp_weights.py` selected norm tensors with
`endswith('norm.weight')`. That silently misses `pre_fc_norm_embedding.weight` and
`pre_fc_norm_hidden.weight`, because `"norm"` is mid-name in both. Those two never received
the `+1.0` RMSNorm convention shift (HF stores `x*(1+w)`; MLX `nn.RMSNorm` applies `x*w`),
leaving the MTP head's fc-input normalization inverted.

**Symptom profile — the reason this read as a capability limit rather than a bug:** ~0%
acceptance, no errors, no warnings, backbone entirely unaffected, tool calls still correct.

Upstream **has already fixed this**. Current `extract_mtp_weights.py:113-125` matches any
1-D norm weight and its comment names the symptom verbatim:

```python
# EVERY RMSNorm weight uses HF's ``(1 + w)`` convention; MLX's nn.RMSNorm
# applies ``x * w`` directly, so each norm weight needs +1.0. This MUST
# include the MTP-specific ``pre_fc_norm_embedding`` / ``pre_fc_norm_hidden``
# (the "norm" is mid-name, so an ``endswith('norm.weight')`` list silently
# missed them — leaving the MTP head's fc-input normalization inverted and
# producing ~0% draft acceptance). Match any 1-D norm weight instead.
for k in list(all_mtp_weights.keys()):
    if "norm" in k and k.endswith(".weight") and all_mtp_weights[k].ndim == 1:
        all_mtp_weights[k] = all_mtp_weights[k] + 1.0
```

**Contributing factor.** `qwen3_5_inject.py:725` calls
`mtp.load_weights(list(mtp_weights.items()), strict=False)`. A structurally valid but
semantically wrong head loads silently. There is no post-load sanity check anywhere on
the path.

**Cheapest detector:** `pre_fc_norm_embedding` mean should be ≈ **+0.56**, not ≈ **−0.44**.

---

## 2. Root cause 2 — sidecar inside the model directory double-shifts the trunk

Independent of root cause 1, and **not fixed upstream**.

`mlx_lm/utils.py:316` globs the trunk's weight shards:

```python
weight_files = glob.glob(str(model_path / "model*.safetensors"))
```

That pattern matches `model-mtp.safetensors`. `mlx_lm/models/qwen3_5.py:307-330` then
strips the `mtp.*` keys — so no stray keys survive — but their mere *presence* flips a
heuristic that shifts **every trunk norm**:

```python
has_mtp_weights = any("mtp." in k for k in weights)
should_shift_norm_weights = has_mtp_weights or has_unsanitized_conv1d
weights = {k: v for k, v in weights.items() if "mtp." not in k}
...
norm_keys = (".input_layernorm.weight", ".post_attention_layernorm.weight",
             "model.norm.weight", ".q_norm.weight", ".k_norm.weight")
for k, v in weights.items():
    if should_shift_norm_weights and any(k.endswith(sfx) for sfx in norm_keys):
        if v.ndim == 1:
            weights[k] = v + 1.0        # +1.0 on EVERY trunk norm
```

The heuristic is correct for a raw HF checkpoint, whose norms are stored `(1+w)` and do
need the shift. It is actively harmful on an **already-converted MLX** repo
(mlx-community, nightmedia, unsloth-MLX), whose norms are already in MLX convention: they
get shifted twice and the backbone degrades to gibberish.

**The trap ships in upstream's own tooling.** `extract_mtp_weights.py:188` writes
`output_file = mlx_dir / "model-mtp.safetensors"` — into the MLX model directory.
Following upstream's documented usage on a converted MLX checkpoint corrupts the trunk on
every subsequent load.

Same trunk shards, symlinked two ways, plain `mlx_lm.generate`, prompt
`"The capital of France is"`:

| Model dir contents | Output |
| --- | --- |
| trunk + `model-mtp.safetensors` | `'class.\n\n '` |
| trunk only | `'Paris.\n\n<think>\nThinking Process:\n\n1. **Analyze the input:**…'` |

`qwen3_next.py:455` performs the same strip. `qwen3_5_moe.py` does not.

---

## 3. Offline measurements — draft-vs-target agreement

**Method.** Draft-vs-target top-1 agreement at greedy: for each position, does the MTP
head's argmax match the backbone's own argmax two positions ahead. Computed in a single
model load, so it isolates head quality from the draft-k controller, batching, and server
plumbing. Probe: `mtp_probe2.py`, taking `<trunk_dir> <sidecar_path>`. Prompt: a 55-token
code-generation request, giving 54 comparable positions.

| Trunk | Sidecar | Agreement |
| --- | --- | --- |
| `unsloth/Qwen3.6-27B-MLX-8bit` | `mlx-community/Qwen3.6-27B-MTP-4bit` (official) | **59.26%** (32/54) |
| `unsloth/Qwen3.6-27B-MLX-8bit` | nightmedia bf16-extracted, current upstream script | **61.11%** (33/54) |
| nightmedia mxfp8 27B, sidecar **outside** model dir | nightmedia bf16-extracted | **61.11%** (33/54) |
| nightmedia mxfp8 27B, sidecar **inside** model dir | either | 3.70% (2/54) |

**Convention counter-check.** Applying `+1.0` a second time to an already-correct head
*lowers* agreement — 61.11% → 42.59% (all norms) → 50.00% (`pre_fc_norm_*` only). The
current extractor output is therefore the correct convention, not accidentally right.

The 3.70% row was never a measurement of the MTP head. It is the *target* logits coming
from a double-shifted backbone (§2).

---

## 4. Served measurements

Fresh server per cell. Single trial (**n=1**) per cell — screening evidence, not
qualification. 512-token completions, `--max-num-seqs 1`, `--max-concurrent-requests 1`,
auto-K controller enabled, `num_speculative_tokens=4` requested.

### 4.1 Acceptance and controller behaviour

Raw counter deltas from the receipts in `tmp/`:

| Cell | attempts | accepts | ratio | rounds | parks | K histogram |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| control: unsloth 8-bit + official 4-bit head | 255 | 248 | **0.9725** | 265 | 10 | K0=10, K1=255 |
| subject: nightmedia `qx64-hi` + extracted affine 8-bit head | 256 | 247 | **0.9648** | 266 | 10 | K0=10, K1=256 |
| subject: nightmedia MXFP8 trunk-only + same head | 258 | 245 | **0.9496** | 268 | 10 | K0=10, K1=258 |
| `qx64-hi`, prose workload | 292 | 211 | **0.7226** | 301 | 9 | K0=9, K1=292 |
| `qx64-hi`, 31,772-token code prompt | 257 | 245 | **0.9533** | 266 | 9 | K0=9, K1=257 |

**Acceptance is not inflated by a hidden denominator.** This was checked, because an
adaptive controller that parks on hard positions would bias conditional acceptance upward.
It does not here: parks are 3–4% of rounds, and the token accounting closes. Control:
265 rounds + 248 accepts = 513 against 514 measured completion tokens. So roughly **1.94
tokens per backbone forward** in the high-acceptance cells.

**Open observation — the controller looks nearly inert.** Park rate is flat at 3.0–3.8%
across the whole set and is *slightly higher* in the 96.5% cell than in the 72.3% cell.
An expected-value depth controller should park far more often on low-acceptance content.
On this workload set it does not discriminate. Unexplained; see Phase 6.5.

### 4.2 Throughput and latency

| Cell | accept | TG off | TG mtp | TG gain | TTFT off | TTFT mtp | End-to-end |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control, unsloth 8-bit | 97.3% | 14.47 | 17.63 | **+21.8%** | 10039 ms | 10217 ms | +15.7% |
| subject, `qx64-hi` | 96.5% | 21.87 | 32.82 | **+50.1%** | 10021 ms | 9891 ms | +31.2% |
| subject, MXFP8 | 95.0% | 17.04 | 27.17 | **+59.4%** | 10243 ms | 10254 ms | +38.4% |
| `qx64-hi`, prose | 72.3% | 23.68 | 30.57 | **+29.1%** | 3797 ms | 4098 ms | +22.0% |
| `qx64-hi`, 32k code | 95.3% | 16.47 | 21.58 | **+31.0%** | 52035 ms | 58124 ms | **+1.6%** |

End-to-end is `T_off / T_mtp - 1` where `T = TTFT + N / TG` at `N = 512`. All ten
throughput figures and all five end-to-end figures were recomputed from the raw receipt
fields and match.

**The 32k row is the product-relevant one.** Acceptance is excellent (95.3%) and
generation gains 31%, but TTFT regresses 11.7%, leaving **+1.6% end-to-end** for a
512-token completion. Acceptance alone does not predict user-visible benefit.

### 4.3 Unexplained anomaly — speedup inverts against acceptance

| Trunk | accept | TG gain | implied per-round cost |
| --- | ---: | ---: | ---: |
| unsloth 8-bit + 4-bit head (control) | 97.3% | +21.8% | **1.59×** |
| `qx64-hi` + 8-bit head | 96.5% | +50.1% | 1.29× |
| MXFP8 + 8-bit head | 95.0% | +59.4% | 1.20× |

All three produce ~1.94 tokens per round, so the differing gains are entirely per-round
overhead. The control has the **slowest** baseline and the **cheapest** sidecar (4-bit vs
8-bit) and should therefore gain the most. It gains the least, by a wide margin.

Candidate explanations, none tested: n=1 thermal or ordering effects; a genuinely more
expensive draft forward on the official head; an unrepresentative unsloth off-baseline.
**This is the single number most likely to invalidate a product claim** and it is
unresolved.

### 4.4 Receipt locations

| Path | Contents |
| --- | --- |
| `tmp/spec-decode-served-positive-control-v2/` | official-sidecar control |
| `tmp/spec-decode-served-off-v2/` | unsloth 8-bit off baseline |
| `tmp/spec-decode-qx64-subject-v2/`, `tmp/spec-decode-qx64-off-v2/` | `qx64-hi` pair |
| `tmp/spec-decode-mxfp8-subject-v2/`, `tmp/spec-decode-mxfp8-off-v2/` | MXFP8 pair |
| `tmp/spec-decode-qx64-prose-pair-v2/`, `tmp/spec-decode-qx64-code-32k-pair-v2/` | workload pairs |
| `tmp/spec-decode-trials-unsloth8bit-greedy-clean/` | clean 4-trial forced/greedy ABBA repeat |
| `tmp/spec-decode-trials-unsloth8bit-recommended-default/` | sampled-lane failure receipt and backend log |

**Void receipts.** `tmp/spec-decode-receipts/`, `tmp/spec-decode-warm-receipts/`, and
`tmp/spec-decode-control-receipts/` were all produced with a dead draft head (§1), and the
nightmedia ones additionally with a double-shifted trunk (§2). **They measure nothing about
MTP capability.** Retained only as a record of what the broken configuration looked like.

---

### 4.5 Clean repeated greedy lane — 2026-07-29

The unsloth 8-bit trunk was repeated with four counterbalanced ABBA trials, 45 seconds of
thermal settling, requested max K=2, the official 4-bit control, and the validated extracted
8-bit subject. The lane was explicitly `forced` and `greedy`, so it is research evidence,
not a naturally-eligible qualification result.

| Arm | Trials | Acceptance | TG gain, mean (range) | TTFT change, mean | End-to-end gain, mean (range) |
| --- | ---: | ---: | ---: | ---: | ---: |
| official 4-bit control | 4 | 97.25% | **+21.17%** (+20.66% to +21.72%) | −1.17% | **+16.10%** (+13.83% to +17.25%) |
| extracted 8-bit subject | 4 | 95.72% | **+20.01%** (+19.56% to +20.40%) | −1.98% | **+15.49%** (+14.52% to +16.26%) |

The suite completed all 12 receipts with no server errors. The positive control passed in
all four trials. Every speculative cell captured the SSM K=2→K=1 clamp. All four baseline
completions had one SHA-256 digest, and all eight off-vs-MTP comparisons matched it:
`greedy_parity.verdict="parity_held"`.

The earlier run's isolated trial-3 baseline collapse did not reproduce. Within this trunk,
the higher-acceptance official control also produced the slightly higher generation gain.
This does not resolve §4.3, which compares different trunks and still needs repeated qx64
and MXFP8 baselines if greedy-only research remains in scope.

### 4.6 Recommended sampled lane is unsupported — 2026-07-29

The next run resolved the unsloth vendor profile to temperature 1.0, top-p 0.95, top-k 20,
and reasoning enabled. Its baseline completed at 14.40 tok/s. The first official-control
request then produced zero speculative counter activity and the harness failed as designed.

The backend log records both facts:

- MTP installed as `single-request greedy K=1 ... falls through on ... non-greedy /
  logits-processors`.
- The request engaged `fused_top_p_sampler` at temperature 1.0/top-p 0.95/top-k 20.

Installed Rapid-MLX 0.11.1 source makes this an explicit contract: temperature >0 falls
through to autoregressive decode because residual-distribution rejection sampling is not
exercised by the MVP, and non-greedy support is a follow-up. Therefore sampled acceptance
and speedup are not merely unmeasured; they are unavailable in the pinned runtime. A
temperature-0.6 coding variant and tool-logits-processor workload hit the same boundary.

This is unchanged on current upstream `main`; it is also independently recorded in upstream
issue `#1013`. There are two separate real-agent blockers:

1. A sampling temperature above zero trips `_is_greedy_for_uid` and falls through.
2. Any truthy `GenerationBatch.logits_processors` entry also falls through. Rapid's
   default-on constrained-tool path builds a stateful `GrammarLogitsProcessor` for ordinary
   tool-enabled `auto` requests, so tool traffic reaches this gate without requiring
   `--enable-tool-logits-bias`.

The apparent bypass — force temperature 0 and set `RAPID_MLX_CONSTRAIN_TOOLS=0` — changes
both the sampling policy and tool-call reliability contract. It is suitable only for
diagnostics. It is not transparent acceleration of a coding-agent workload.

The implementation gap is narrower than a new speculative decoder but wider than deleting
two guards. `mtp_generate_step` already accepts temperature/top-p/top-k/min-p and contains
probabilistic acceptance plus residual-distribution sampling; the scheduler nevertheless
hardcodes `temp=0.0`. Sampled support must plumb the real per-request sampler and preserve
seed/EOS/cancellation behavior. Tool support is harder: the stateful grammar processor must
advance across accepted drafts and restore exactly on rejection rather than observe
hypothetical tokens permanently.

---

## 5. Draft depth is clamped to K=1 on Qwen3.5/3.6

`num_speculative_tokens` flows through the CLI and scheduler as `max_k`, but
`generator.py:~613` probes the real cache objects at loop start:

```python
_has_ssm_cache = any(hasattr(c, "rollback_state") for c in model_cache)
if not disable_auto_k:
    _max_k_hw = 1 if _has_ssm_cache else max(0, max_k)
```

Any model whose cache list contains an SSM slot is clamped to effective K=1. Observed
consequences:

- requested K=1, K=2, K=3, or K=4 with auto enabled all select only K∈{0,1};
- `disable_auto_k=true` bypasses the controller and fixes effective K=1 — it is **not**
  fixed K=4;
- `--force-spec-decode`, `--no-hybrid`, profile aliases, hidden legacy MTP flags,
  programmatic scheduler configuration, and direct generator `max_k` all reach the same
  clamp;
- there is no MTP K/SSM bypass environment variable.

**This is a correctness boundary, not conservative tuning.** The GatedDeltaNet patch stores
one `(conv_state, ssm_state)` snapshot. During K=2 verification of
`[target, draft1, draft2]`, that snapshot is after `[target, draft1]`. If `draft1` is
rejected, correct recovery needs the unavailable state after `[target]`. Rapid therefore
asserts when asked to roll SSM state back by more than one token. Removing the assert
restores the wrong recurrent state and silently violates lossless speculative decoding.
Safe hybrid K≥2 requires per-position snapshots or correct replay.

Sources:
[CLI mapping](https://github.com/raullenchai/Rapid-MLX/blob/206ed0e03b7b6fc7b3b2e6f68a7b60467f6e5abe/vllm_mlx/cli.py#L2012-L2038) ·
[scheduler mapping](https://github.com/raullenchai/Rapid-MLX/blob/206ed0e03b7b6fc7b3b2e6f68a7b60467f6e5abe/vllm_mlx/scheduler.py#L2626-L2637) ·
[clamp](https://github.com/raullenchai/Rapid-MLX/blob/206ed0e03b7b6fc7b3b2e6f68a7b60467f6e5abe/vllm_mlx/spec_decode/mtp/generator.py#L600-L641)

### Directly observed, 2026-07-29

No longer an inference. With backend stdout/stderr persisted per cell (harness commit
`8217f55`) and `--speculative-tokens 2` so the line can fire at all, the clamp was captured
verbatim on `unsloth/Qwen3.6-27B-MLX-8bit`:

```
[MTP-chain-of-K] … clamping max_k from 2 to 1
```

`runtime.backend_log.clamp_verdict` records `clamp_observed` with `effective_max_k: 1`.
This closes the ambiguity noted below: the K∈{0,1} histograms are the clamp, not an EV
controller preferring K=1.

> Superseded. Previously read: *"Not yet observed directly … The clamp is proven from
> source, inferred for these specific runs."* True when written — server stderr was not
> retained at that time.

### Unsloth's depth-2 guidance does not contradict this

[Qwen3.5](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF/blob/3221f178a6b842d04f1fb42f1c413534adcc0a6a/README.md#L995-L999)
and [Qwen3.6](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/blob/82d411acf4a06cfb8d9b073a5211bf410bfc29bf/README.md#L169-L173)
recommend two speculative tokens for vLLM's `qwen3_next_mtp`; the
[Qwen3.6 MTP GGUF card](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/blob/5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace/README.md#L48-L55)
recommends two for llama.cpp; the same cards recommend a different three-step/four-draft
SGLang setup. The number is a **backend-specific runtime width**, not the count of trained
MTP layers and not a portable model recommendation.

---

## 6. `aliases.json` is unreliable capability metadata

Two independent errors, both verified against source.

**6.1 The `supports_spec_decode` flag exists and does gate the MTP lane.** Every
`qwen3.6-27b-{4,6,8}bit` and `qwen3.6-27b-ud*` entry carries
`is_hybrid: false, is_hybrid_explicit: true, supports_spec_decode: false`.
`_mtp_path_label()` in `model_auto_config.py:2040+` reports `native` only when the family
ships a native MTP head **and** `supports_spec_decode=True`, else `disabled`. So the
mechanism claim holds.

**The inference does not.** "Upstream benchmarked this family and found MTP useless" is
unsupported by anything in the source and is directly contradicted by §3 and §4. Note that
`qwen3.6-35b-*` carries `is_hybrid: true` with the *same* `supports_spec_decode: false`,
consistent with the flag being applied family-wide from the hybrid lane rather than from a
27B-specific benchmark. **Treat the rationale as unknown.** Plausible readings, none
confirmed: a conservative default; blanket inheritance from the 35B hybrids; a decision
predating the §1 extractor fix.

**6.2 The `is_hybrid` value is factually wrong.** `unsloth/Qwen3.6-27B-MLX-8bit`
`config.json` `text_config.layer_types` has 64 entries: **48 `linear_attention`** and
16 `full_attention`, with `full_attention_interval: 4`. The 27B **is** a hybrid SSM model.
The alias says it is not.

This does not change §5 — the clamp keys off the runtime cache object, not the alias — so
both remain individually correct and never disagree at runtime. But it establishes the
general rule:

> **Never treat `aliases.json` as a source of architectural or capability truth.** It has
> now been shown wrong twice. Read `layer_types` from the model config instead.

---

## 7. Quantization and MoE support boundaries

### 7.1 Trunk and sidecar are independently packed

- The extractor reads only the destination config's top-level `bits` and `group_size` and
  calls affine `mx.quantize`. It does **not** reproduce `mode=mxfp*` or per-layer mixed
  overrides.
- Rapid infers the explicit sidecar's affine `(bits, group_size)` from its `fc.weight` /
  `fc.scales` (`qwen3_5_inject.py:520-557`), builds the MTP module at that packing, and
  validates all required sidecar shapes/dtypes before loading. `_detect_base_quantization`
  (the trunk reader) is used only when no explicit sidecar is passed.
- Supported explicit affine sidecar widths: **2/3/4/5/6/8 bits**, group sizes **32/64/128**.
- Therefore an MXFP4 target yields an affine 4-bit head by default, MXFP8 yields affine
  8-bit, and a `qx64-hi` target yields a uniform affine 6-bit head by default despite its
  per-layer 4/6-bit overrides. `--bits` / `--group-size` may deliberately choose another
  supported packing.
- An **MXFP-packed sidecar is not supported** by this injector; it expects full precision
  or uniformly affine `weight/scales/biases`. mxfp8 sidecars have no `biases`.
- `mlx_lm/utils.py:361` passes `mode=quantization.get("mode", "affine")`, so an mxfp8
  *trunk* is honoured correctly. **There is no "MTP requires an 8-bit trunk" rule** —
  §4 proves an affine 8-bit head on both a mixed 4/6-bit trunk and an MXFP8 trunk.

**Compatibility is structural, not provenance-safe.** The BF16 source should be the exact
unquantized parent/revision of the trunk. A same-shape head from a different fine-tune can
pass loading and still have poor acceptance, because it reuses the target trunk's
embeddings and LM head. The runtime checks structure, never training provenance.

### 7.2 MoE

Runtime support is explicit: both `qwen3_5` and `qwen3_5_moe` dispatch to the same
injector; the MTP head constructs `SparseMoeBlock` when `num_experts > 0`; the target
config must still declare `mtp_num_hidden_layers >= 1`.

Extractor support depends on source tensor layout:

- **Qwen3.6-35B-A3B is supported.** Its BF16 MTP tensors use fused `experts.gate_up_proj` /
  `experts.down_proj`; current upstream extraction converts these to canonical
  `switch_mlp.{gate,up,down}_proj`. Upstream commit `738a44e` reports an end-to-end
  Qwen3.6-35B-A3B result around 89% K=1 acceptance.
- **Official Qwen3.5-35B-A3B is not supported.** Its index uses 768 individually numbered
  expert tensors (`experts.<n>.{gate,up,down}_proj.weight`) and no fused `gate_up_proj`.
  The extractor does not stack these into `switch_mlp`; Rapid's required-key check will
  reject the resulting sidecar.

Capability detection must therefore distinguish family **and source layout**. There is no
single "MoE MTP supported" boolean.

---

## 8. Current extractor capabilities and gaps

Upstream `scripts/extract_mtp_weights.py` is useful but too narrow and unsafe to call
directly from a product workflow.

**Supported:** `--hf-model` repo ID with an indexed full-precision safetensors set; reads
`model.safetensors.index.json` and downloads every complete shard referenced by an `mtp.*`
key (shard-granular, may include substantial non-MTP tensors); applies the `+1.0` shift to
every 1-D `.weight`; converts Qwen3.6 fused MoE tensors to `switch_mlp`; affine-quantizes
every non-1-D MTP weight while leaving 1-D norms full precision; defaults 4-bit/group-64.

**Gaps and blockers:**

- no immutable `--revision`: a mutable repo default can silently change;
- no local authoritative-safetensors directory input;
- requires a sharded safetensors index — no unsharded safetensors, PyTorch, GGUF, or adapter input;
- no proof that the BF16 source is the parent of the target trunk;
- no source/config/tokenizer/embedding/LM-head identity check;
- no output provenance manifest or source/target hashes;
- reads only top-level target quantization; does not reproduce MXFP mode or per-layer overrides;
- official Qwen3.5 numbered-expert MoE layout is not stacked and must fail explicitly;
- shifts **any** 1-D `.weight` — safe for known Qwen norms, but an unexpected 1-D non-norm
  weight would be silently corrupted;
- **writes `model-mtp.safetensors` inside the target MLX directory** (§2);
- **may edit the target config in place** to add missing MTP depth.

The last two are unacceptable for llama-monitor. Never point the unwrapped script at a
user-owned trunk or an HF cache snapshot.

---

## 9. Artifact state

| Path | State |
| --- | --- |
| `~/.config/llama-monitor/models/mlx/native/nightmedia-27b-mxfp8-mlx` | The MXFP8 trunk, 27 GB of real files. Moved here from `/Users/nick/mlx-models/` on 2026-07-30. Its two in-dir heads (`model-mtp.safetensors` and `model-mtp.safetensors.shifted.bak`) were pulled out and quarantined; both failed the norm preflight (`pre_fc_norm_embedding` mean ≈ −0.44) and either one triggers the §2 trunk glob defect while present. Trunk weights themselves were never modified. |
| `/Users/nick/mlx-models/` | **Deleted 2026-07-30.** Held no model bytes of its own by then — two symlink farms over the trunk and the HF cache (`nightmedia-27b-mxfp8-mtp-fixed`, `control-unsloth8bit-official-mtp`, neither holding a head), plus the quarantine directory below. The `-mtp-fixed` farm carried a correctly extracted head **in-dir**, so it had the §2 defect; the `control-unsloth8bit-official-mtp` head was built from the stale extractor and was **invalid — never reuse it**. Both are superseded by the standalone official control. |
| `~/.config/llama-monitor/models/rapid-mlx/mtp-sidecars/.quarantine-in-trunk-sidecars/` | The four quarantined heads (1.9 GB) plus the README explaining each. Moved out of `~/mlx-models/` on 2026-07-30 when that directory was deleted. Kept as negative-control fixtures: a stale-extractor head and a wrong-parent head are exactly what a sidecar preflight has to refuse. |
| `scratchpad/nm-trunk-only` | Symlinked trunk, MTP files excluded. Generates correctly. Used as the MXFP8 subject. |
| `scratchpad/nm-mtp-fixed.safetensors` | Validated external sidecar. SHA-256 `5088ad4339b6d669b51ee66ab0e1ad214e5ca17d62bf643970ff09f09f25717a`. Served successfully. |
| `scratchpad/mtp_probe2.py` | The offline agreement probe. |
| `scratchpad/extract_mtp_weights.py` | Current upstream extractor, re-pulled 2026-07-29. |

**On `.shifted.bak`:** byte-level preflight shows both the live in-dir file *and* the
`.bak` carry the stale-extractor signature. Do not infer convention from the filename.
`nm-mtp-fixed.safetensors` is the only validated artifact (mean ≈ +0.56).

`mlx-community/Qwen3.6-27B-MTP-4bit` is a standalone **239 MB** sidecar
(`model_type: qwen3_5_mtp`), not a full model — the cheapest known-good positive control.

---

## 10. Install integrity and cleared suspects

- rapid-mlx 0.11.1: **267/267** `vllm_mlx` files hash-match `RECORD`. No local edits to
  `head.py`, `qwen3_5_inject.py`, or `generator.py`.
- Acceptance logic in `generator.py:730-879` was read and is **correct**. The greedy path
  compares `toks[:k_len] == drafts_i32`; `record_attempt()` is called `k_len` times per
  round. Not a suspect.
- Source of record: `~/.local/share/uv/tools/rapid-mlx/lib/python3.11/site-packages/vllm_mlx/`
  and `.../mlx_lm/`.

---

## 10b. The `family` metric label is path-derived, not fact-derived

Audited in `routes/metrics.py`. The label on every `rapid_mlx_spec_decode_*` series is:

```python
family = getattr(cfg, "model_alias", None) or _derive_mtp_family(cfg)
```

Neither branch reports a model family:

- **With an alias set**, the alias is used *verbatim*. The upstream docstring's own
  example is `"qwen3.5-9b-4bit" → "qwen3.5-9b-4bit"` — a quantization-specific alias,
  not a family.
- **Without an alias**, `_derive_mtp_family` substring-matches `model_name` /
  `model_path` / `model` against a six-row hint table (`gemma-4`, `gemma4`, `qwen3.5`,
  `qwen3_5`, `qwen3.6`, `qwen3_6`) and returns `"unknown"` on no match.

So the label is a function of **the served path string**, and a directory rename changes it.

**The `family="unknown"` in the MXFP8 receipts is not a model property.** Audited across
every receipt on disk:

| `family` | Served path | Receipts |
| --- | --- | --- |
| `qwen3.6` | `…/models--unsloth--Qwen3.6-27B-MLX-8bit/snapshots/…` | served-off, positive-control, fixed-k4 |
| `qwen3.6` | `…/models--nightmedia--Qwen3.6-…-qx64-hi-mlx/snapshots/…` | qx64 off, subject, prose-pair, code-32k-pair |
| `unknown` | `…/scratchpad/nm-trunk-only` | mxfp8 off, mxfp8 subject |

Both HF-cache paths carry `Qwen3.6` in the repo directory and sniff correctly. The MXFP8
cells were served through a sanitized scratchpad directory whose name contains no family
substring — the harness caused the label, the model did not.

**Consequence.** Never key an aggregation, alert, or capability decision on `family`. It
splits on operator path layout and collapses to `unknown` for any locally-managed model
directory, which is precisely the layout Phase 6.5b's managed sidecar store will use. The
benchmark harness's control gate therefore sums `rapid_mlx_spec_decode_{accepts,attempts}_total`
by **metric-name prefix** across all label sets, and is immune by construction.

---

## 11. Known-unknowns

Carried forward to Phase 6.5. Nothing below has been measured.

1. ~~Repeated-trial variance on the unsloth greedy lane.~~ **Closed 2026-07-29** with four
   counterbalanced trials; qx64 and MXFP8 remain n=1.
2. The §4.3 speedup inversion.
3. Why the depth controller barely parks at 72% acceptance (§4.1).
4. ~~Direct observation of the K=1 clamp log (§5).~~ **Closed 2026-07-29** — observed,
   `clamping max_k from 2 to 1`. See §5.
5. Tool-call warm/repeat/extension behaviour under MTP.
6. 65k and 131k prompt tiers; completion lengths other than 512.
7. Whether a same-shape wrong-parent sidecar is detectable before serving.
8. ~~Greedy-lossless token parity between off and MTP runs.~~ **Closed 2026-07-29** on the
   unsloth lane: deterministic baseline and 8/8 speculative comparisons matched.
9. Sampled MTP behavior. **Blocked in Rapid-MLX 0.11.1:** non-greedy requests deliberately
   fall through to plain autoregressive decode.

Items 5 and 9, plus the constrained-tool case, are now the three gates of
`scripts/rapid-mlx-requalify-spec-decode.mjs`. Run it against a new upstream build rather than
re-deriving them by hand; it exits `20` while upstream is still blocked, and only a full sweep
with every gate passing may promote `spec_decode` out of `Unavailable`. If it ever exits `0`,
record the measurements here **before** adding a `SchedulerEvidence::Engages` entry to `SPEC_DECODE_VERSION_PRIORS` in
`src/inference/rapid_mlx/capabilities.rs` — the version list is a claim about this document.

## 12. Procedure — requalifying a future build

**Short version — this is the whole procedure on an already-set-up box:**

```
node scripts/rapid-mlx-requalify-spec-decode.mjs
```

No arguments. Models, heads, parsers and port come from `scripts/spec-decode-recipe.json`;
receipts land in `tmp/requalify-<version>-<date>`; local paths are checked before anything is
served; and the verdict is recorded against the installed runtime automatically, so
`spec_decode` — and therefore MTP — reflects the run without a second step. `--help` lists the
overrides. Exit `0` qualified, `20` still blocked, `1` uninterpretable.

A passing sweep needs one more thing that a local run cannot do for you: add the version to
`SPEC_DECODE_VERSION_PRIORS` in `src/inference/rapid_mlx/capabilities.rs` as
`SchedulerEvidence::Engages`, so users who never run this lane get the fix too. The lane prints
that reminder on exit `0`.

The rest of this section is the long version: what each artifact is, why each argument exists,
and how to rebuild the inputs if the recipe's paths no longer resolve.

Everything below is runnable without re-deriving any of the analysis above. Two scripts: one
builds a draft head, one measures whether the runtime will use it.

### 12.1 Build a draft head from a BF16 source

Needed only if you do not already have a head for the trunk. Skip it when requalifying with
the official control (§12.3), which ships its own.

```
python3 scripts/build-mtp-head.py \
  --bf16-source nightmedia/Qwen3.6-27B-Architect-Polaris2-Fable-B-F451-Tess \
  --mlx-model ~/.config/llama-monitor/models/mlx/native/nightmedia-27b-mxfp8-mlx
```

`--bf16-source` is the repo (or local path) carrying the `mtp.*` tensors; `--mlx-model` is the
quantized MLX trunk, which supplies the quantization config and **is not modified**. Output
defaults to `~/.config/llama-monitor/models/rapid-mlx/mtp-sidecars/<trunk-slug>/mtp.safetensors`
with a `provenance.json` beside it.

Tensor extraction is upstream's **MTPSplitter framework**, run inside the managed
rapid-mlx venv the server already uses (`rapid_mlx.models.mlx_vlm_vendored.speculative.
drafters.qwen3_5_mtp.split`). Per-family tensor selection, the RMSNorm shift convention,
quantization metadata, and tokenizer copying are upstream's live code, so upstream fixes
land the moment the managed install updates — there is no vendored snapshot to re-pull
and no extractor sha pin to maintain. The wrapper picks the dense vs MoE splitter by the
source checkpoint's `num_experts` and installs the standalone drafter under the managed
sidecar root as `mtp.safetensors`, with `config.json` and tokenizer files beside it.

The wrapper owns only what upstream gets wrong for us, which is placement, not math:

- Upstream writes `model-mtp.safetensors` **into** `--mlx-model` and rewrites that
  `config.json` in place. `mlx_lm` globs `model*.safetensors` when loading a trunk, so an
  in-trunk sidecar is read as a trunk shard, which sets `should_shift_norm_weights` and
  double-shifts every trunk RMSNorm weight into gibberish, silently (§2). The wrapper moves the
  sidecar out, restores the config, and refuses an `--out` inside the trunk.
- It re-reads the built head and refuses to certify one whose `pre_fc_norm_*` means are not
  positive. A valid head reads ~`+0.56`; the stale extractor's read ~`-0.44` and gave ~0%
  acceptance (§1). This is the check that would have caught the original defect.

A passing norm check is an offline sanity check, not a qualification. Only §12.2 measures
whether the head is used.

### 12.2 Measure whether the runtime engages it

```
node scripts/rapid-mlx-requalify-spec-decode.mjs
```

**No arguments.** Every input — trunk, subject head, positive control, parsers, port — is
pinned in `scripts/spec-decode-recipe.json`, and `--out` defaults to a timestamped directory.
That is deliberate: the arguments are not guessable months later and the wrong ones do not
fail loudly, so a run with no tool parser would install no tool grammar and then report that
speculation survives tool use. Any field is still overridable by flag, and flags win over the
recipe field by field.

Exit codes: `0` qualified, `20` gates ran cleanly but the scheduler still does not engage,
`1` uninterpretable (control failed or a run errored). A failing positive control is always
`fail`, never `blocked`: "this build is still limited" and "the harness is broken" are
different findings, and conflating them is what produced the void receipts.

`--profile-alias` is not optional in practice. `rapid-mlx info` resolves the tool and reasoning
parsers only for HF repo aliases; for a **local model directory it reports the literal
`(none)`** for both. Without the alias the `constrained` gate would install no tool grammar and
then report that speculation survives tool use — from a request that never constrained
anything. The lane refuses to run that gate rather than produce the number. Pass
`--tool-call-parser` / `--reasoning-parser` explicitly if no alias fits.

Gate names are duplicated in `SPEC_DECODE_GATES` in `src/inference/rapid_mlx/capabilities.rs`,
because the post-upgrade probe message names the outstanding gates. Keep the two in step.

### 12.3 Artifacts this needs, and where they are

| Artifact | Location | Note |
| --- | --- | --- |
| Trunk | `~/.config/llama-monitor/models/mlx/native/nightmedia-27b-mxfp8-mlx` | 27 GB, real files |
| Positive control | `mlx-community/Qwen3.6-27B-MTP-4bit` | HF cache; a standalone 228 MB / 31-tensor head (`model_type: qwen3_5_mtp`), not a full model — nothing to extract |
| Validated subject head | `~/.config/llama-monitor/models/rapid-mlx/mtp-sidecars/qwen3.6-27b-nightmedia-f451-tess-8bit/` | 478 MB + `provenance.json` |
| Quarantined heads | `~/.config/llama-monitor/models/rapid-mlx/mtp-sidecars/.quarantine-in-trunk-sidecars/` | Pulled out of trunks; mostly stale-extractor output. Kept as negative-control fixtures. See its README |

All three lane artifacts live under `~/.config/llama-monitor/models/` as of 2026-07-30, so
model management stays inside the app: the trunk at `models/mlx/native/` (already an
inventory root, no import step needed), the subject head at `models/rapid-mlx/mtp-sidecars/`,
and the positive control in the app's own HF cache at `models/cache/huggingface/hub/`. The
lane exports `HF_HUB_CACHE` to that cache, matching what the resolver does when the app
launches rapid-mlx, so a repo-id control resolves to a model the app can actually serve.

`~/mlx-models/` no longer exists — it was deleted on 2026-07-30, once the trunk had moved and
6.5a closed. What it held by then was two symlink farms over the trunk and the user-wide HF
cache (`-mtp-fixed` and `control-unsloth8bit-official-mtp`, each with its own `config.json`
and neither holding a head) plus the quarantine directory, which was moved rather than
deleted; see §9. Any older receipt or log citing an `~/mlx-models/…` path is describing the
pre-move layout, not a directory a re-runner should expect to find.

⚠️ **The `tmp/…` receipt paths cited throughout this record are untracked** and will not survive
a clean checkout. Treat a missing receipt directory as absent evidence, not as a contradiction of
a recorded number; re-running the lane regenerates it.

### 12.3a Harness invocation prerequisites

Facts a re-runner needs before invoking the fuller benchmark suite directly rather than through
the requalification lane. Folded here 2026-07-30 from the working-handoff doc, which was then
deleted; the per-commit history of how the harness reached this state is in `git log
scripts/rapid-mlx-benchmark-suite.mjs`.

- **`--spec-decode-lane forced|natural` is required and has no default** on any spec-decode
  invocation. The lane is verified against the backend log and a mismatch fails the run. This
  exists because the suite passes `--force-spec-decode` unconditionally whenever a speculative
  config is present — the tested aliases advertise `supports_spec_decode=false` — which is a
  legitimate research override and an illegitimate qualification basis. A forced-lane number must
  never reach an enablement decision.
- **`--trials N`** runs counterbalanced ABBA ordering with `--settle-seconds` between cells and
  records `trial_protocol` in the suite index. A single-trial number is not a qualification
  result.
- **`--sampling greedy|recommended|explicit`** with `--sampling-variant` resolves through ordered
  sources (operator flags → checkpoint → curated vendor profile) and records which source
  answered. "No published settings" is a legal state that clears `performance_claim_eligible`;
  it is not an error. The unsloth trunk ships no `generation_config.json` at all, so nothing may
  assume a model publishes its settings.
- **`--spec-completion-tokens`** defaults to 8192 with a 4096 floor. 512 tokens cannot produce a
  stable accept ratio, and with reasoning on the model can spend the entire budget inside
  `<think>` and fail the completion floor — which reads as a gate failure rather than the harness
  limitation it is.
- The sidecar must resolve to the **snapshot** path, not the realpath: HF blobs have no extension
  and `mx.load` dispatches on it, so a realpath sidecar makes rapid-mlx refuse to boot.

### 12.4 Zero speculative activity is an answer, not a crash

The first two live executions of the lane both exited `1` / `uninterpretable` on a build whose
correct verdict is `20` / `still-blocked`. Two independent guards caused it, and both were
right for the qualification matrix they were written for:

1. `assertAttemptQualified` in `model-runtime-benchmark.mjs` calls `die()` when a speculative
   cell records no attempts, parks, or K-chosen rounds — so no receipt was ever written.
2. `assertPositiveControl` treats a control with no attempts as a control *failure*, which the
   requalification lane maps to `fail`.

Between them, all three gates' own `attempts <= 0 → blocked` branches were unreachable dead
code on 0.11.1: the lane could not report the one finding it exists to report. Fixed with an
opt-in, `speculative_zero_activity: 'required' | 'observed'` (suite flag
`--spec-zero-activity`, default `required`):

- `required` keeps the old behaviour. A speculative cell that never speculated measured
  nothing, which is the correct default for a qualification run.
- `observed` records `fidelity.speculative_activity_observed` (set on **both** outcomes, so an
  absent field never reads as "not checked") and lets the caller's predicate decide.

The positive control now returns a `code` — `cleared-floor`, `no-activity`, `failed`, or
`missing` — because "never engaged" and "engaged but below floor" are different findings.
`no-activity` returns `ok: true` so it cannot be misread as a broken harness, but carries a
`reason` and prints `Positive control: NO ACTIVITY`, because nothing about it licenses a
positive claim either. The requalification lane additionally refuses to promote capability when
a gate returns `pass` while its control recorded `no-activity` — that combination means the
predicate and the control disagree, and neither reading is then trustworthy.

⚠️ Do not "simplify" this by defaulting `--spec-zero-activity` to `observed`. A qualification
matrix that tolerates zero activity will happily record a full sweep of cells that never
speculated, which is precisely how the original void receipts happened.

### 12.5 The scheduler states its own constraints in the backend log

Observed directly on 0.11.1 while first executing the lane (`--gate sampled`, nightmedia MXFP8
trunk, official 4-bit control):

```
[MTP-vendored] installed on GenerationBatch._step (single-request greedy K=1 chain-of-1;
falls through on B>1 / non-greedy / logits-processors)
```

This is the two blocking conditions stated by the runtime itself, in one line, at install time —
not inferred from acceptance counts. It is also exactly the definition of
`MtpConcurrencyState::SingleActiveGreedy`, which `derive_mtp_concurrency` in
`src/inference/rapid_mlx/capabilities.rs` still returns `Unknown` for because it only sees
`serve --help`. The line is captured in every cell's `*.server.log`, so the requalification lane
is the natural place to promote that state from a measurement rather than a help-text guess.

Also worth noting from the same run: injection itself is healthy. `[mtp.inject]` loaded 31/31
expected tensors from the control sidecar and reported `Quantized MTP: 4-bit, group_size=64
(from sidecar tensors)`, and `[MTP-chain-of-K]` clamped `max_k` 2→1 on the SSM cache as in §5.
Nothing about the sidecar path is broken; the fall-through is a scheduler policy.

### 12.6 Generation length

The spec-decode suites generate `--spec-completion-tokens` (default 8192, floor half that).
It was 512, which broke the `sampled` gate outright: the recommended lane resolves this family
to temperature 1.0 with reasoning **on**, so the model spent the whole 512-token budget inside
`<think>`, emitted empty content, and the run failed its completion floor — a failure that says
nothing about speculation. The `sampled` gate therefore pins `--sampling explicit
--temperature 0.6` and leaves reasoning off, isolating the one axis it is named for. The cap is
only a ceiling; the floor is what costs wall-clock, so lower it if a per-bump check needs to be
cheaper than a stable acceptance ratio.
