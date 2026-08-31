# Inference Performance & Tuning Reference

How to pick a model, quantization, KV-cache type, and llama-server flags for the
best real-world throughput on a given machine. Covers **Apple Silicon (unified
memory)** and **discrete NVIDIA GPUs (5080 / 5090)**, dense vs MoE, and the
`--n-cpu-moe` offload strategy that makes large MoE models usable on small GPUs.

> **TL;DR for Apple Silicon:** dense models above ~14B are memory-bandwidth bound
> and there is no software trick that fixes that — run **MoE** instead. The
> runtime (llama.cpp vs MLX) barely matters at 27B+; the model *architecture*
> does. See [The two regimes](#the-two-regimes) and [Apple Silicon profiles](#apple-silicon-unified-memory).

When configuring models, use the Setup wizard or Preset Editor to stay within VRAM-safe recommendations.
These tools tie directly into the VRAM estimator, so you get a configuration that
respects your hardware limits instead of guessing from model size alone.

The VRAM estimator is hardware-aware: it uses your detected memory and backend
to suggest context size, KV cache types, and MoE settings that fit.

---

## The two regimes

Everything below follows from one distinction: **where the bottleneck is.**

| | Unified memory (Apple Silicon) | Discrete GPU (NVIDIA) |
|---|---|---|
| VRAM | = system RAM (minus OS reserve) | fixed, separate from system RAM |
| Decode bottleneck | **memory bandwidth** (~410 GB/s M*-Pro → ~614 GB/s M5 Max → ~820 GB/s M*-Ultra) | usually **VRAM capacity**, then bandwidth (5090 ≈ 1790 GB/s) |
| Prefill bottleneck | GPU compute (M5 adds matmul accelerators, MLX-only today) | GPU compute (abundant on 5080/5090) |
| Large MoE strategy | fit it in unified RAM; no offload needed | offload expert FFNs to system RAM via `--n-cpu-moe` |
| System RAM role | *is* the VRAM | a **spillover pool** for MoE experts |

### The physics (why this is non-negotiable)

- **Decode speed ≈ (memory bandwidth × efficiency) ÷ (bytes streamed per token).**
  Every generated token must read the active weights from memory once.
  - **Dense:** active bytes = the *entire* model. A 27B at Q5 (~18 GB) on a
    614 GB/s M5 Max tops out near `614 × 0.5 / 18 ≈ 17 t/s`. No loader beats this.
  - **MoE:** active bytes = attention/shared weights + only the *active* experts
    (e.g. 3B of 35B). That is why a 35B-A3B MoE runs ~3× faster than a dense 27B
    on the same chip.
- **Prefill speed ≈ GPU compute throughput** and is far less bandwidth-sensitive.
  This is why an M5 Max can do ~1600 t/s prefill on an MoE but only ~17 t/s decode
  on a dense model — different bottlenecks.
- **Metal efficiency** is roughly **0.5–0.65** of theoretical bandwidth; **CUDA**
  is roughly **0.7–0.85**. Use those as the multiplier in the estimate above.

---

## Measured anchors (Apple M5 Max, 64 GB, 40-core GPU, ~614 GB/s)

These were measured with `llama-bench`, build 9542, `-fa 1 -ngl 99`. Treat them as
ground truth for this class of chip; scale by the bandwidth ratio for other Macs.

| Model | Quant | KV | Context | Prefill (pp) | Decode (tg) |
|---|---|---|---|---|---|
| Qwen3.6-27B **dense** | Q5_K_S | q8_0 | short | 529 t/s | **16.9 t/s** |
| Qwen3.6-27B **dense** | Q5_K_S | f16  | short | 478 t/s | 17.7 t/s |
| Qwen3.6-27B **dense** | Q5_K_S | q8_0 | @16k | — | **6.9 t/s** |
| Qwen3.6-27B **dense** | Q5_K_S | f16  | @16k | — | 7.7 t/s |
| Qwen3.6-**35B-A3B MoE** | Q4_K_M | q8_0 | short | 1587 t/s | **50.5 t/s** |
| Qwen3.6-**35B-A3B MoE** | Q4_K_M | q8_0 | @32k | — | **72.9 t/s** |
| Qwen3.6-**35B-A3B MoE** | Q4_K_M | q8_0 | @98k | — | **30.2 t/s** |

Takeaways:
1. **Dense 27B ≈ 17 t/s is the hardware ceiling**, not a misconfiguration. It
   collapses to ~7 t/s by 16k context.
2. **MoE is ~3× faster and degrades far more gracefully** — still 30 t/s at 98k.
3. **q8_0 KV is slightly *slower* than f16 at depth on Metal** (6.9 vs 7.7). On
   Apple Silicon, prefer f16 KV unless you need q8 to fit the context (see
   [KV cache](#kv-cache-the-second-biggest-lever)).

---

## Quantization reference

File size ≈ `params(B) × bits-per-weight ÷ 8` GB. Decode scales inversely with
size, so a smaller quant is the only loader-independent way to speed up dense decode
— at a quality cost.

| Quant | bits/wt | Quality | When to use |
|---|---|---|---|
| Q8_0 | ~8.5 | ~lossless | Only if it fits and you don't care about speed |
| Q6_K | ~6.6 | excellent | Best quality/size on roomy hardware |
| Q5_K_M | ~5.7 | very good | Good default when VRAM allows |
| **Q4_K_M** | ~4.8 | **good** | **The default sweet spot for most models** |
| Q4_K_S | ~4.5 | good- | Squeeze a model into tight VRAM |
| Q4_0 | ~4.5 | acceptable normally | **Preferred for official Gemma 4 QAT checkpoints** |
| IQ3 / Q3_K | ~3.4 | noticeable loss | Last resort to fit |
| **MXFP4** | ~4.25 | native for gpt-oss | Use the official gpt-oss MXFP4 builds |

**MLX quant note:** MLX 4-bit ≈ llama.cpp Q4 in bytes, therefore ≈ the same
decode speed. The *format* is not a speedup; only fewer *bits* are. See
[Loader choice](#loader-choice-llamacpp-vs-mlx).

### Gemma 4 QAT

Google's Gemma 4 QAT checkpoints were trained to preserve near-BF16 quality when
converted to **Q4_0**. This changes the quality/size trade-off, not the runtime
architecture:

- Prefer the official `*-qat-q4_0-gguf` file when choosing a compact Gemma 4.
  The quant advisor recognizes `gemma-4` + `qat` names and rates Q4_0 as the
  high-quality target instead of applying the generic Q4_0 quality penalty.
- Unsloth's QAT GGUF repositories currently expose a QAT-derived
  `UD-Q4_K_XL` model plus separate multimodal projectors. Use the exact file size
  for VRAM planning; no QAT-specific llama-server flag is required.
- Prefer the matching **F16 mmproj** for Gemma 4 when it is available. This is
  Unsloth's documented llama.cpp default for both normal and QAT repositories;
  the QAT model does not require a different projector precision. Compatibility
  still comes first: use the projector produced for the same Gemma 4 variant and
  revision rather than borrowing an F16 projector from another model.
- QAT applies to **model weights**, not `-ctk` / `-ctv`. Choose KV precision from
  context and measured throughput exactly as for the non-QAT model.
- Gemma 4's MTP assistant is a separate matching draft checkpoint. A base QAT
  GGUF does not by itself mean MTP is active; enable speculative decoding only
  after supplying the compatible draft model.

Approximate official Q4_0 load footprints, including Google's 20% loading
allowance but excluding context KV and runtime-specific extras:

| Model | Q4_0 memory | Architecture |
|---|---:|---|
| Gemma 4 E2B | 2.9 GB | dense, 128K context |
| Gemma 4 E4B | 4.5 GB | dense, 128K context |
| Gemma 4 12B | 6.7 GB | dense unified multimodal, 256K context |
| Gemma 4 26B-A4B | 14.4 GB | MoE, 256K context |
| Gemma 4 31B | 17.5 GB | dense, 256K context |

---

## KV cache: the second-biggest lever

KV cache grows linearly with context and is often the deciding factor for whether
a long context fits. Per-token KV size:

```
bytes/token = 2 (K+V) × n_layers × n_kv_heads × head_dim × bytes_per_elem
```

- **f16** = 2 bytes/elem, **q8_0** = ~1, **q4_0** = ~0.5.
- **Quantized KV requires flash attention** (`-fa on`) and **matching K/V types**
  (`-ctk q8_0 -ctv q8_0`) to stay on the fast fused kernel — mixed types silently
  fall back to a slow path.
- Presets store KV types in the canonical `ctk`/`ctv` fields (the `-ctk`/`-ctv`
  launch flags). Launch validation rejects the one build-gated mixed pair,
  `q8_0/q4_0`, against the binary's capability snapshot, and refuses to let
  `extra_args` reintroduce KV flags that would bypass the typed fields.

### Architecture caveats that change the math

- **Qwen3.5 / 3.6 have `head_dim = 256`** (double the usual 128). This *doubles*
  the KV cache and puts quantized KV on a less-optimized Metal path — q8_0 KV can
  be **slower** than f16 here, and exotic rotation-based KV-quant schemes lose
  38–50% throughput. **On Apple Silicon use f16 KV when it fits; only drop to q8_0
  to make a long context fit at all.**
- **Gemma 4 uses alternating local/global attention** (sliding-window local layers
  + a few full-context global layers, `global_head_dim = 512`). Its KV cache is
  much smaller than a same-size dense model at long context — Gemma is unusually
  cheap to run at high context. The app's [VRAM estimator](vram-estimator.md)
  models this directly.

### Rule of thumb

| Context | Apple Silicon (64 GB) | Apple Silicon (128 GB) | Discrete |
|---|---|---|---|
| ≤ 32k | **f16 KV** (faster) | f16 KV | f16 KV |
| 64k–131k | q8_0 KV (to fit) | f16 if it fits, else q8_0 | f16 if VRAM allows, else q8_0 |

---

## Architecture labels (Dense / MoE / Hybrid MoE)

Wherever llama-monitor shows architecture info (welcome screen cards, preset editor, spawn wizard), you may see:

- **Dense • X B**
  - All X B parameters active per token.
  - Heavy on memory bandwidth; slower on Apple Silicon when large.
- **MoE • X B (Y B active)**
  - X B total, but only Y B actually computed per token via expert routing.
  - Often much faster than a dense model of similar total size, especially on:
    - Apple Silicon (less memory traffic)
    - Discrete GPUs with MoE offload (--n-cpu-moe).
- **Hybrid MoE • X B (Y B active)**
  - MoE plus hybrid attention (fewer full-KV layers).
  - More efficient at long context: smaller KV cache, faster decode.

Use these labels as a quick guide:
- For shared or laptop systems: prefer MoE or Hybrid MoE at the same quality tier.
- For dedicated desktops/GPU rigs: dense is acceptable and often fine with high priority.

## The throughput levers, in priority order

1. **Model architecture (MoE > dense)** — the only ~3× lever on unified memory.
2. **Quantization** — Q4_K_M default; smaller = faster decode, lower quality.
3. **KV cache type & context size** — don't allocate 131k if you use 16k; the
   over-allocation forces slow q8 KV and wastes memory.
4. **MTP / speculative decoding** — see below; ~1.4–2× on top, no quality loss.
5. **Flash attention** — `-fa on`, always, on both platforms.
6. **Threads** — set `-t` to the **performance-core count** (e.g. 6 on M5 Max),
   not 1. Mostly affects prefill and sampling overhead.
7. **`--n-cpu-moe`** — discrete-GPU MoE offload, see [its section](#-n-cpu-moe-large-moe-on-small-gpus).

### MTP (Multi-Token Prediction)

Qwen3.5/3.6 ship MTP draft heads built into dedicated GGUFs. They give **~1.4–2×
faster decode with zero quality change** (the main model verifies the drafts).

- Requires **llama.cpp ≥ b9180** (the app bundles a newer build).
- Flag (renamed May 2026): **`--spec-type draft-mtp --spec-draft-n-max 3`**.
  `n-max 3` is optimal for most quants; 4 only helps at F16.
- Helps **decode**, can slightly slow **prefill** (extra embedding transfers).
- Bigger relative win on **dense** (predictable drafts) than MoE, but worthwhile on
  both. ngram speculation (`--spec-type ngram-*`) has near-zero acceptance on
  reasoning/code — don't rely on it.

---

## Apple Silicon (unified memory)

Budget rule used by this app: usable VRAM ≈ `total RAM − in-use RAM − ~6 GB OS
reserve`, then keep ~10% headroom (Metal burst). Exceeding it does **not** spill
gracefully — it pages to SSD and collapses. See [VRAM estimator](vram-estimator.md).

### 64 GB Mac

| Model | Fits? | Recommended | Expected decode | Notes |
|---|---|---|---|---|
| Qwen3.6-35B-A3B (MoE) | ✅ | **Q4_K_M, q8 KV @131k / f16 KV @≤32k** | 30–70 t/s | **Best all-round pick.** MTP on. |
| gpt-oss-20b (MoE) | ✅ easily | MXFP4 (native) | very fast | Great light/coding model. |
| Gemma 4 12B (dense) | ✅ easily | **QAT Q4_0**, f16 KV | fast | Strong compact multimodal/agentic option. |
| Gemma 4 26B-A4B (MoE) | ✅ | **QAT Q4_0**, f16 KV | fast | Cheap KV; ample unified-memory headroom. |
| Qwen3-Coder (30B-A3B class, MoE) | ✅ | Q4_K_M | fast | Coder-tuned MoE. |
| Qwen3.6-27B (dense) | ✅ | Q5_K_S/Q4_K_M, **f16 KV ≤32k** | ~17 t/s (≤7 at depth) | Quality model, but slow; MTP helps. |
| Gemma 4 31B (dense) | ✅ | **QAT Q4_0** | ~12–16 t/s | Dense; QAT improves 4-bit quality and cheap KV softens long ctx. |
| Llama 3.3 70B (dense) | ✅ Q4 (~40 GB) | Q4_K_M, q8 KV | ~8–10 t/s | Batch only; painful interactive. |
| gpt-oss-120b (MoE) | ❌ (~63 GB + KV) | — | — | No room for KV/OS on 64 GB. |
| Qwen3.5-122B-A10B (MoE) | ❌ at Q4 (~65 GB) | — | — | Needs 128 GB. |

**Interactive/agentic (OpenCode) on 64 GB:** Qwen3.6-35B-A3B, Q4_K_M, `-c 131072
-ctk q8_0 -ctv q8_0 -fa on -ngl 99 -t 6 --spec-type draft-mtp --spec-draft-n-max 3`.

### 128 GB Mac

Everything above, plus the big MoEs become the headline:

| Model | Fits? | Recommended | Expected decode | Notes |
|---|---|---|---|---|
| gpt-oss-120b (MoE, ~5B active) | ✅ MXFP4 (~63 GB) | MXFP4, f16/q8 KV | ~30–50 t/s | Flagship local model on a 128 GB Mac. |
| Qwen3.5-122B-A10B (MoE, 10B active) | ✅ Q4 (~65 GB) | Q4_K_M, q8 KV @long | ~20–30 t/s | 10B active → heavier than gpt-oss. |
| Llama 3.3 70B (dense) | ✅ | Q5_K_M / Q6_K | ~9–12 t/s | Quality-first, batch. |
| Qwen3.6-35B-A3B (MoE) | ✅ | Q5/Q6, f16 KV @131k | 40–70 t/s | Run a higher quant than on 64 GB. |

Decode figures for the 120B-class MoEs are **estimates from the bandwidth model**
(`614 × 0.5 / active-GB`); confirm with `llama-bench` on your unit.

---

## Discrete NVIDIA (CUDA)

Here **VRAM capacity is the gate**, and **system RAM is a spillover pool for MoE
experts** via `--n-cpu-moe`. CUDA decode efficiency (~0.7–0.85) is high, so anything
that fits entirely in VRAM is fast.

### What fits fully in VRAM (no offload)

| GPU | VRAM | Dense fully on-GPU | MoE fully on-GPU |
|---|---|---|---|
| 5080 | 16 GB | ≤14B Q4–Q5, or 8B Q6/Q8 | ≤~20B-total MoE at Q4 (e.g. gpt-oss-20b MXFP4) |
| 5090 | 32 GB | ≤32B Q4–Q5 (e.g. Qwen3.6-27B → 35–70 t/s) | 35B-A3B Q4 (~20 GB) → 100+ t/s |

A dense 70B (~40 GB Q4) exceeds even a 5090 — it must partially offload to CPU and
will be slow (dense layers on CPU are brutal). Prefer a large **MoE** instead, which
offloads cleanly.

### `--n-cpu-moe`: large MoE on small GPUs

MoE weights are dominated by **expert FFN tensors** that are mostly idle per token.
`--n-cpu-moe N` keeps the **expert tensors of N layers on the CPU/system RAM** while
attention + the rest stay on the GPU. This lets a 16 GB card run a 120B MoE — decode
is then limited by how many active experts must be fetched over PCIe each token.

**Tuning procedure:**
1. Start with `--cpu-moe` (all experts on CPU). Confirm it loads and runs.
2. Decrease `--n-cpu-moe N` (move more layers' experts onto the GPU) until you are
   just under OOM. Every layer moved onto the GPU speeds decode.
3. Keep KV + compute buffers in the VRAM budget — leave ~1–2 GB headroom.

**Sizing math:**
```
VRAM_used ≈ attention/shared weights + KV cache + compute buffers
          + (total_moe_layers − n_cpu_moe) × per_layer_expert_size
```
Solve for the largest `(total_moe_layers − n_cpu_moe)` that fits. The app's
[VRAM estimator](vram-estimator.md) can do this for you.

### Discrete profiles by system RAM

System RAM must hold the **offloaded experts** plus the OS. For a ~63 GB MXFP4
gpt-oss-120b, the CPU-resident portion needs to fit in RAM.

| System RAM | gpt-oss-120b / 122B-A10B (offloaded) | Practical guidance |
|---|---|---|
| 32 GB | ❌ can't hold the offloaded experts | Stick to models that fit VRAM + a small spill (≤~30B-total MoE, dense ≤VRAM). |
| 64 GB | ⚠️ tight for 120B-class | Possible with aggressive `--n-cpu-moe`, low context; expect 5–15 t/s. |
| 128 GB | ✅ comfortable | Run 120B-class MoE; push more onto GPU as VRAM allows. Best discrete experience for big MoE. |

**5080 (16 GB) + 128 GB RAM, gpt-oss-120b:** `--n-cpu-moe` high (most experts in
RAM), small-to-moderate context, `-fa on`. Decode is PCIe/RAM-bandwidth bound,
typically **single-digit to low-teens t/s** — usable for non-interactive work.

**5090 (32 GB) + 128 GB RAM, gpt-oss-120b:** lower `--n-cpu-moe` (more experts on
GPU) → meaningfully faster than the 5080. Dense ≤32B and 35B-A3B-class MoE run
entirely on-GPU and are very fast.

---

## Loader choice: llama.cpp vs MLX

### llama.cpp model load mode

The Pro Spawn Wizard and Preset Editor expose llama.cpp's typed `load_mode` policy:

- `mmap` is the default and keeps normal memory-mapped loading behavior.
- `none` replaces the legacy `--no-mmap` spelling for users who explicitly need it.
- `mlock` and `mmap+mlock` pin mapped pages; use only with sufficient RAM.
- `dio` is an opt-in direct-I/O mode supported only by compatible llama.cpp builds.

Older presets containing `no_mmap` are migrated to an equivalent typed mode and
retain the legacy field for compatibility. Non-default modes are launch policy,
not benchmark recommendations, and should be qualified separately with a
matching no-spec control.

- **Dense ≥27B:** llama.cpp ≈ MLX (both bandwidth-bound; the gap is near zero above
  27B — MLX only wins big *under ~14B*). Quant choice moves throughput ~20%; loader
  choice moves it a couple t/s.
- **Prefill / TTFT:** MLX can exploit the **M5's matmul accelerators** (up to ~4×
  TTFT) which llama.cpp does **not** use yet — but at **long context** MLX prefill
  is often *slower* than llama.cpp + flash attention, and MLX's long-context/FA
  story is weaker. For 131k agentic use, **llama.cpp is the better bet.**
- **MTP on MLX:** native Qwen3.5/3.6 MTP exists in MLX (mlx-lm PR #990, and the
  third-party **MTPLX** runtime), reaching ~18 t/s on a dense 27B — roughly the same
  ceiling llama.cpp's dense MTP hits. Not a reason to leave the llama-server stack.

**Bottom line:** keep using llama.cpp/llama-server; choose **MoE + MTP + the right
KV/context**, not a different loader, to go faster.

---

## Per-family notes

Recommendations above are model-specific; these are the family-level rules that
generalize to new releases.

- **Qwen3.5 / 3.6 (dense & MoE):** `head_dim = 256` → KV cache is double-sized and
  quantized-KV is a slow path on Metal (prefer f16 KV when it fits). All ship MTP
  GGUFs — always enable `--spec-type draft-mtp`. MoE variants (A3B/A10B) are the
  right pick on every platform for speed.
- **Gemma 4 (E2B, E4B, dense 12B/31B, and 26B-A4B MoE):** alternating **local/global attention**
  (sliding-window local layers + sparse full-context global layers,
  `global_head_dim = 512`) makes the KV cache **much cheaper at long context** than
  a same-size Qwen/Llama. The 12B, 26B-A4B, and 31B support 256K context; E2B/E4B
  support 128K. The QAT releases make Q4_0 the preferred quality/size choice.
  The dense 31B is more tolerable at depth than a dense Qwen 27B even though raw
  decode is similar (~12–16 t/s on M5 Max, fast on a 5090). The 26B-A4B MoE is
  the fast option, but its 14.4 GB Q4_0 load estimate leaves limited room for KV
  and multimodal overhead on a 16 GB GPU. The app's
  [VRAM estimator](vram-estimator.md) models Gemma's two attention regimes
  explicitly — trust its long-context numbers over the generic formula.
- **gpt-oss (20b & 120b, MoE, native MXFP4):** always use the official **MXFP4**
  builds (don't re-quantize). 20b runs anywhere (fits a 5080 / any 64 GB Mac);
  120b needs a 128 GB Mac to run in-memory, or a discrete GPU + ≥128 GB system RAM
  with `--n-cpu-moe`. ~5B active → strong decode when it fits.
- **Llama 3.3 70B (dense):** the cautionary case. ~40 GB at Q4 exceeds a 5090 and
  is bandwidth-painful on Mac (~8–12 t/s). Use only for quality-first batch work;
  for interactive, a large MoE beats it on every axis.
- **Qwen3-Coder (30B-A3B class, MoE) / Qwen3.5-122B-A10B (MoE):** treat like other
  MoEs — fit in unified RAM on Mac, or offload experts with `--n-cpu-moe` on
  discrete. The 122B-A10B has 10B active, so it's heavier per token than gpt-oss-120b
  (≈5B active) despite similar total size — expect lower decode.

### Discrete-GPU quick fit (Gemma & large MoE)

| Model | 5080 (16 GB) | 5090 (32 GB) |
|---|---|---|
| Gemma 4 12B (dense) | ✅ QAT Q4_0, roomy | ✅ QAT Q4_0, very fast |
| Gemma 4 26B-A4B (MoE) | ⚠️ QAT Q4_0 is tight after KV/mmproj | ✅ QAT Q4_0, very fast |
| Gemma 4 31B (dense) | ❌ full GPU fit | ✅ QAT Q4_0 (~17.5 GB), fast |
| gpt-oss-20b (MoE) | ✅ MXFP4 | ✅ MXFP4, very fast |
| gpt-oss-120b (MoE) | `--n-cpu-moe` high + ≥128 GB RAM | `--n-cpu-moe` moderate + ≥128 GB RAM |
| Qwen3.5-122B-A10B (MoE) | `--n-cpu-moe` high + ≥128 GB RAM | `--n-cpu-moe` + ≥128 GB RAM |
| Llama 3.3 70B (dense) | ❌ | ❌ full (heavy CPU offload only) |

---

## Benchmark any model yourself

The bundled **`llama-bench`** sits next to `llama-server` in the app's binary dir
(macOS: `~/.config/llama-monitor/bin/llama-bench`; on Windows/Linux it's beside the
`llama-server` path shown in **Settings → Server**). Run these against *your* file
to replace any estimate in this doc with a measured number.

**1. Baseline (short context):**
```bash
llama-bench -m <model.gguf> -ngl 99 -fa 1 -ctk f16 -ctv f16 -p 512 -n 128 -r 2
```
Read the two output rows: `pp512` = prefill t/s, `tg128` = decode t/s.

**2. KV type comparison (run twice, matched K/V):**
```bash
llama-bench -m <model.gguf> -ngl 99 -fa 1 -ctk f16  -ctv f16  -p 512 -n 128 -r 2
llama-bench -m <model.gguf> -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 512 -n 128 -r 2
```

**3. Depth sweep (the important one — exposes the long-context collapse):**
```bash
llama-bench -m <model.gguf> -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -n 64 -d 0,16384,32768,98304 -r 1
```
`tg64 @ dN` is decode speed with N tokens already in the cache. This is what your
agentic/coding workload actually feels — short-context benches hide it.

**4. Discrete MoE offload sweep (`--n-cpu-moe`):** start high, lower until just
before OOM; each step that moves experts onto the GPU should raise decode.
```bash
llama-bench -m <moe.gguf> -ngl 99 -fa 1 --n-cpu-moe 48 -n 64 -r 1   # then 40, 32, ...
```

**Caveats:**
- `llama-bench` measures **base** decode — it does **not** exercise MTP /
  speculative decoding. To measure MTP, run `llama-server` with `--spec-type
  draft-mtp` and read the real tokens/sec from the app's chat telemetry (or
  `llama-cli` with the spec flags), comparing against the same request without it.
- Always match `-d` to your real working context; quote the depth with every number.
- `-r` is repetitions (higher = less noise, slower). Use `-r 1` for quick depth
  sweeps, `-r 2+` for numbers you'll publish.

### Calibration (bounded v1)

Phase 7 adds an optional `server_qualification` request to the calibration
start payload. The default lane is `parallel_requests: 1` with the
`latency_memory` track. `tool_correctness` is independently selectable;
concurrency requires explicit `allow_concurrency: true`. MTP and n-gram tracks
run only when managed-runtime capability evidence is present and include an
explicit no-spec baseline; DFLASH remains capability-gated and fail-closed
until a verified llama.cpp signal is available; this only omits the
DFLASH-specific receipt and does not block a DFLASH-capable preset from launch.
MTP qualification is likewise a bounded compatibility/smoke check: it preserves
the preset's existing `n_max`/`p_min` values and does not attempt to optimize
speculative decoding across workload-specific prompts.

Calibration is the evidence-first path for llama.cpp tuning. The current
release boundary exposes bounded, explicitly confirmed Quick and Balanced
candidate plans; Thorough search is not part of the 2.0 release contract. It
does not stop an active server or emit Rapid-MLX settings. The job records
durable journal transitions and a receipt under the active application home,
and an interrupted trial is surfaced as a suspected crash instead of being
retried silently.

The authenticated API is:

- **POST `/api/calibrations/preflight`** with `{ "preset_id": "...",
  "budget": "quick" | "balanced", "workload": { ... } }` — validates a
  local llama.cpp preset, configured managed `llama-bench`, model library
  membership, and returns a redacted fingerprint plus the bounded candidate
  and verification-trial plan.
- **POST `/api/calibrations`** — starts the bounded trial. The request must
  include the preflight fingerprint, a supported `budget` (`quick` or
  `balanced`), and the exact `CALIBRATE` confirmation string returned by the
  application; active-server stop/restart is not permitted in this version.
- **GET `/api/calibrations`** — lists durable job snapshots from the active
  application home. Protected receipt contents and manifests are never
  returned by this endpoint.
- **GET `/api/calibrations/{id}`** — polls the durable job snapshot.
- **GET `/api/calibrations/{id}/receipt`** — reads the authenticated measured
  receipt and apply history.
- **POST `/api/calibrations/{id}/cancel`** — requests cancellation and cleans up
  the owned benchmark process.
- **POST `/api/calibrations/{id}/resume`** — explicitly resumes a job recovered
  as a suspected crash. It requires the `RESUME_CALIBRATION` confirmation;
  finished trial results are loaded from the protected journal and are not
  silently repeated.
- **POST `/api/calibrations/{id}/apply`** — requires `db-admin-token`, the
  `APPLY_CALIBRATION` confirmation, and the expected target fingerprint. It
  creates a derived preset by default; updating the source preset is an
  explicit `create_derived: false` choice. Applying never changes the active
  session, runs one bounded post-apply `llama-bench` validation by default,
  and records before/after fingerprints plus validation status in the receipt.
  A failed validation immediately restores the source or removes the derived
  preset.
- **POST `/api/calibrations/{id}/rollback`** — requires `db-admin-token`, the
  `ROLLBACK_CALIBRATION` confirmation, and the applied preset fingerprint. It
  restores the private pre-apply snapshot (or removes the derived preset) only
  when the preset is unchanged since apply; conflicting edits fail closed.
- **POST `/api/calibrations/{id}/forget`** — requires `db-admin-token` and
  `FORGET_CALIBRATION`; removes a terminal job, its receipt, and private
  rollback backups. Active jobs must be cancelled first.

Quick and Balanced Calibration report measured candidate results, baseline
deltas, Pareto tradeoffs, and confidence/noise warnings. They are not a claim
that one run has found an optimal configuration: Balanced verification remains
bounded and Thorough refinement is deferred. Rapid-MLX benchmark output remains
informational-only until a separate backend-owned factor catalog and receipt
qualification exists.

#### Spawn Wizard receipt reuse

Spawn Wizard calls `POST /api/calibrations/match` with a saved llama.cpp preset,
workload, and budget. Results are tiered: exact receipts require the complete
artifact/runtime fingerprint; compatible evidence requires the same
introspected family/shape, GGUF weight-quantization signature,
hardware/workload, and normalized runtime capabilities; related evidence may
have a different weight quantization and is review-only. Runtime build changes
are disclosed rather than treated as a user-facing version pin. Rapid-MLX
receipts never cross into this llama.cpp evidence path.

### Built-in tuning and benchmark endpoints

The app exposes several POST endpoints (all require api-token auth) that automate
parts of this process. These are internal endpoints used by the UI, Preset Editor,
and Spawn Wizard.

- **POST /api/benchmark**
  - Sends a short test prompt to the currently running llama-server and measures
    prompt tokens/sec, gen tokens/sec, and TTFT.
  - Normal use: rate-limited to once every 15 seconds.
  - Tuning mode: include `{ "tuning": true }` in the body to skip the cooldown
    while iteratively adjusting settings.
  - Returns verdict plus hints/suggestions.

- **POST /api/advise**
  - Config-time performance advisor: given model name/params, context size, KV
    types, platform, and speculative decoding config, returns suggestions for
    MoE vs dense, KV type, MTP, etc.
  - Used by the Spawn Wizard and Preset Editor before running any benchmark.

- **POST /api/model-defaults**
  - Returns recommended chat parameters (temperature, top_p, reasoning, etc.) for
    a given model based on name, size, and tags.
  - Ensures tuned defaults align with the model family.

- **POST /api/moe-tune**
  - Suggests a starting `--n-cpu-moe` value for a MoE model given
    `model_size_bytes`, `available_vram_bytes`, and `n_moe_layers`.
  - Helps you fit MoE experts into limited VRAM by offloading some to system RAM.

- **POST /api/tune/ncpumoe**
  - Advanced `--n-cpu-moe` auto-tuner.
  - Without `verify`: returns an estimate based on VRAM fit.
  - With `verify: true`: runs llama-bench against several candidate `--n-cpu-moe`
    values and selects the one that gives the best real decode speed.
  - Requires the llama-server to be stopped when verifying (llama-bench needs the GPU).

- **POST /api/bench/sweep**
  - Offline depth sweep via llama-bench: benchmarks decode at multiple context
    depths (default 0, 16k, 32k).
  - Requires llama-server stopped; returns per-depth points (`points` array).

- **POST /api/bench/batch-sweep**
  - Offline batch/ubatch sweep: probes a matrix of (batch, ubatch) pairs and
    returns the best prefill throughput along with all probes.
  - Requires llama-server stopped.

- **POST /api/bench/mtp-sweep**
  - Online MTP n-max sweep: for a locally-spawned server with speculative decoding,
    iterates over given `spec-draft-n-max` values, restarting the server between
    probes, and selects the n_max that maximizes generation tokens/sec.
  - Requires local server and active MTP/draft config.

---

## Quick-pick cheat sheet

| Hardware | Best speed/quality balance | Flags |
|---|---|---|
| Mac 64 GB | Qwen3.6-35B-A3B Q4_K_M | `-c 131072 -ctk q8_0 -ctv q8_0 -fa on -ngl 99 -t 6 --spec-type draft-mtp --spec-draft-n-max 3` |
| Mac 128 GB | gpt-oss-120b MXFP4 | `-fa on -ngl 99 -t <P-cores>` (f16 KV at moderate ctx) |
| 5090 32 GB | Qwen3.6-27B dense Q5 *or* 35B-A3B Q4 | `-fa on -ngl 99` (+ `--spec-type draft-mtp` for MTP GGUFs) |
| 5080 16 GB | gpt-oss-20b MXFP4 (fits) | `-fa on -ngl 99` |
| 5080/5090 + 128 GB RAM, big MoE | gpt-oss-120b | `--n-cpu-moe <tuned>` `-fa on` (tune N down to OOM-1) |

Always confirm with `llama-bench -m <model> -fa 1 -ngl 99 -ctk <k> -ctv <v> -d <depth>`
at your **real** context depth — short-context benchmarks hide the long-context
collapse that dominates agentic use.

---

## Wizard control disclosure (Guided and Pro)

The wizard now presents Guided recommendations and a searchable Pro view over one canonical
state registry. Tier metadata controls Guided disclosure and default resolution; it never
removes a control, and Pro modified-only/reset actions operate on the same state.

Every hardware-step control in the Spawn Wizard is declared once, per loader, in
`static/js/features/spawn-wizard-groups.js` as a `{ id, tier, quickValue? }` row. The tier
governs two things and nothing else: whether the control is disabled with a wizard-written
value on Quick, and whether its group starts open or collapsed. **No tier ever removes a
control from the DOM** — everything is reachable at every profile, just closed by default
above Quick.

### llama.cpp

| Tier | Controls |
|---|---|
| Quick (disabled, wizard-written) | `spawn-context-size`, `spawn-batch-size`, `spawn-gpu-layers` (→ `auto`) |
| Balanced (editable, visible) | the three above, plus `spawn-cache-type-k/v`, `spawn-kv-unified`, `hw-quant-select`, `hw-mmproj-select`, `hw-use-mtp`, `spawn-parallel-slots` |
| Advanced (auto-opened) | `spawn-ubatch-size`, `spawn-flash-attn`, `spawn-prio`, `spawn-threads`, `spawn-threads-batch`, `spawn-n-cpu-moe`, `spawn-tensor-split`, `spawn-cache-mode`, `spawn-cache-ram`, `spawn-fit-enable`/`spawn-fit-target`, `spawn-mlock` |
| Advanced, nested collapse | `#spawn-spec-details`: `spawn-spec-type`, `spawn-spec-draft-type-k/v`, `spawn-draft-model`, `spawn-spec-draft-n-min`, `spawn-spec-draft-p-min` |

### Rapid-MLX

| Group | Control | Tier | Rationale |
|---|---|---|---|
| Generation — thinking | `spawn-rapid-reasoning-mode` | Quick, read-only readout | Effective value is always `on` — see [rapid-mlx-runtime.md](rapid-mlx-runtime.md#reasoning-profile-and-kv-cache-dtype). Editable only at Advanced, with the requested→effective diff shown inline. |
| Generation — sampling | `spawn-sampling-mode` | Balanced | MLX peer of KV-quant-as-quality-dial; client sampling params still win, so it's low risk. |
| Generation — protocol | `spawn-rapid-tool-call-parser` | Balanced | Auto-detected; override is a "my finetune is unusual" case, same weight as picking a chat template. |
| Generation — protocol | `spawn-rapid-reasoning-parser` | Balanced | Same as above. |
| Generation — protocol | `spawn-rapid-hybrid-mode` | Advanced | No llama.cpp equivalent; can disagree with runtime introspection on hybrid/linear-attention layers — wrong answer changes the memory model, not just speed. Peer of `spawn-tensor-split`. |
| Cache & Perf — active memory | `spawn-kv-cache-dtype` | Advanced, annotated | Peer of `spawn-cache-type-k/v`, but inert — pinned to int8 by the reasoning profile. |
| Cache & Perf — active memory | `spawn-turboquant-mode` | Advanced, badged "not applied" | Withheld pending per-model qualification receipts — see [rapid-mlx-runtime.md](rapid-mlx-runtime.md#turboquant-and-pflash). |
| Cache & Perf — active memory | `spawn-rapid-prefill-step-size` | Advanced | Peer of `spawn-ubatch-size`. |
| Cache & Perf — retained cache | `spawn-rapid-cache-mode` (Auto/Off/Custom) | Balanced | Peer of `spawn-cache-ram`, but promoted: retained cache is a context-fit scenario axis (see [vram-estimator.md](vram-estimator.md#mlx-context-fit-cards)), so it must be reachable at the tier those cards live at. Numeric `spawn-retained-cache-mib` field only appears on Custom. |
| Cache & Perf — retained cache | `spawn-rapid-hybrid-cache-entries` | Advanced | Working-set count; no llama.cpp peer. |
| Cache & Perf — scheduler | `spawn-rapid-max-num-seqs` | Balanced | Peer of `spawn-parallel-slots` — also a context-fit scenario axis, same reasoning as retained cache. |
| Cache & Perf — scheduler | `spawn-rapid-max-concurrent-requests` | Advanced | Admission limit distinct from sequence count; no llama.cpp peer. |
| Cache & Perf — scheduler | `spawn-rapid-gpu-memory-utilization` | Advanced | Peer of `spawn-fit-target` — a memory-budget ceiling. |
| Cache & Perf — scheduler | `spawn-rapid-pflash-policy` | Advanced, badged, default off | Ruled out at Rapid-MLX 0.11.0 — see [rapid-mlx-runtime.md](rapid-mlx-runtime.md#turboquant-and-pflash). |
| Cache & Perf — scheduler | `spawn-rapid-prefill-batch-size` | Advanced | Peer of `spawn-batch-size`. |
| Cache & Perf — scheduler | `spawn-rapid-completion-batch-size` | Advanced | No direct peer; batching internals. |
| Server & Safety — tool integration | `spawn-rapid-auto-tool-choice` | Balanced | User-facing capability toggle gated on parser compatibility; peer in spirit to `hw-use-mtp`. |
| Server & Safety — companions | `spawn-rapid-speculative-*` (8 controls) | Advanced, nested collapse | Structural peer of `#spawn-spec-details`. |

Every Quick-tier control must carry a `quickValue` the wizard writes when it disables the
control (`spawn-rapid-reasoning-mode` → `on`, `spawn-gpu-layers` → `auto`) — a Quick-disabled
control with nothing to write into it would be a dead control. This is enforced at load time
by `assertQuickValueCoverage()` and offline by `npm run validate-wizard-groups`. See
[spawn-wizard.md](spawn-wizard.md#control-tier-registry-quickbalancedadvanced-disclosure) for
the full I1/I2/I3 invariant writeup.

---

## Sources & further reading

Apple Silicon / M5:
- [M5 Max Local AI guide (llmcheck.net)](https://llmcheck.net/blog/apple-silicon-m5-max-local-ai-guide/)
- [Apple Silicon LLM benchmarks by chip/quant (llmcheck.net)](https://llmcheck.net/benchmarks)
- [Apple M5 Max 128 GB local LLM guide (aiproductivity.ai)](https://aiproductivity.ai/blog/apple-m5-max-local-llm-guide/)
- [Apple Silicon LLM inference optimization guide (starmorph)](https://blog.starmorph.com/blog/apple-silicon-llm-inference-optimization-guide)
- [Tuning llama.cpp on Apple Silicon — 7 flags (Hannecke)](https://medium.com/@michael.hannecke/tuning-llama-cpp-on-apple-silicon-843f37a6c3dc)

Loader comparison:
- [MLX vs llama.cpp on Apple Silicon (Groundy)](https://groundy.com/articles/mlx-vs-llamacpp-on-apple-silicon-which-runtime-to-use-for-local-llm-inference/)
- [Choosing an on-device runtime — decision framework (Hannecke)](https://medium.com/@michael.hannecke/choosing-an-on-device-llm-runtime-on-apple-silicon-a-decision-framework-beyond-benchmarks-2449067b8b67)
- [Comparative study: MLX, MLC-LLM, Ollama, llama.cpp (arXiv 2511.05502)](https://arxiv.org/pdf/2511.05502)

KV cache & quantization:
- [TurboQuant / KV cache quantization discussion (llama.cpp #20969)](https://github.com/ggml-org/llama.cpp/discussions/20969)
- [Symmetric KV quant enables fused flash-attention (llama.cpp #22411)](https://github.com/ggml-org/llama.cpp/discussions/22411)
- [KV cache quantization: Q8 vs FP16 (TechPlained)](https://www.techplained.com/kv-cache-quantization)

MTP / speculative decoding:
- [Qwen3.6 — run locally (Unsloth docs)](https://unsloth.ai/docs/models/qwen3.6)
- [Run Qwen3.6 MTP in llama.cpp — flag change writeup (mer.vin)](https://mer.vin/2026/05/run-qwen-3-6-mtp-in-llama-cpp-faster-local-inference-with-built-in-speculative-decoding/)
- [MTP + llama.cpp with Qwen3.6-27B step-by-step (Dre Dyson)](https://dredyson.com/mtp-llama-cpp-with-qwen3-6-27b-a-complete-beginners-step-by-step-guide-to-speculative-decoding-turboquant-and-running-multiple-models-on-limited-gpu-vram/)
- [MLX + native MTP speculative decoding, 18 t/s on a 27B (vinoth12940)](https://vinoth12940.github.io/blog/articles/genai-20260519-local-mtp-speculative-decoding/)

Gemma 4 QAT:
- [Gemma 4 QAT launch (Google)](https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/)
- [Gemma 4 model and QAT deployment overview (Google AI)](https://ai.google.dev/gemma/docs/core)
- [Gemma 4 QAT guide (Unsloth)](https://unsloth.ai/docs/models/gemma-4/qat)

Related internal docs: [vram-estimator.md](vram-estimator.md) · [cli-flags.md](cli-flags.md) · [setup-wizard.md](setup-wizard.md)

---

## Monitoring your system under load

### Memory pressure (macOS)

The dashboard includes macOS memory-pressure telemetry (via `vm_stat`):

- **Memory Pressure** sparkline and metric in the system card show:
  - Free GB, compressor GB, swap activity.
- A **nav-pill** appears (warning or critical) when:
  - Free memory < 1.5 GB, or compressor uses ≥ 18% of total RAM → warning
  - Free memory < 0.5 GB, or compressor uses ≥ 30% of total RAM → critical

If you see memory pressure while running a large model:
- Reduce context size.
- Stop active downloads.
- Disable mlock if enabled and keep mmap enabled so macOS can reclaim file-backed model pages instead of compressing/swapping the entire system.
- Consider MoE over dense for the same "smartness" with far less decode-time pressure.

### mlock and system responsiveness (macOS)

The preset editor and spawn wizard warn when mlock is enabled and the VRAM estimate is tight.

- mlock pins model memory so the OS cannot page it out. On unified-memory Macs, that removes it from the general pool.
- If the estimated VRAM usage is already close to your available memory, mlock can push macOS into heavy compression and swap, making the desktop feel unresponsive.
- Rule of thumb: if the VRAM estimator says Tight/Risk, prefer **no mlock** and mmap enabled unless you're doing long, memory-intensive runs and know your system has headroom.

### Sleep modes

The monitoring chip in the top nav supports three modes:

- **Monitoring** – full telemetry active.
- **Logs only** – only the live log tail is active; GPU, system, and sparkline updates are paused to save resources.
- **Paused** – all telemetry and logs paused; llama-server keeps running.

Use **Logs only** when you're watching a long generation and want to minimize overhead without losing visibility into the log stream.
