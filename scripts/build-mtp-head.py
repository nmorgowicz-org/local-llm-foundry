#!/usr/bin/env python3
"""Build a managed, out-of-trunk MTP draft sidecar from a BF16 source.

Extraction is upstream's MTPSplitter framework, run inside the same managed
rapid-mlx venv the server uses, so per-family tensor selection, the RMSNorm
shift convention, quantization metadata, and tokenizer copying are all
upstream's code — never a vendored snapshot that silently rots. A Qwen3.5/3.8
stock checkpoint carries a native ``mtp.*`` block; the splitter extracts it
into a standalone drafter directory (weights + config.json + tokenizer).

What this wrapper adds on top of ``split()``:

1. Managed placement. The drafter lands under
   ``~/.config/local-llm-foundry/models/rapid-mlx/mtp-sidecars/<trunk-slug>/``
   as ``mtp.safetensors`` — never inside the trunk, where ``mlx_lm`` would
   glob it up as a trunk shard and double-shift every trunk RMSNorm weight.
2. Provenance. ``provenance.json`` records the bf16 source, revision, trunk,
   splitter used, and sha256, which is what the app's sidecar inventory and
   trunk auto-matching read.
3. The norm sanity check. A valid head reads ``pre_fc_norm_*`` means ~ +0.56;
   an inverted head (~ -0.44) serves ~0% draft acceptance with no error
   anywhere. The check is cheap, so it is not optional. It is a sanity check,
   not a qualification: served acceptance is established only by
   scripts/rapid-mlx-requalify-spec-decode.mjs.

Usage
-----
    python3 scripts/build-mtp-head.py \
        --bf16-source mlx-community/Qwen3.8-27B-bf16 \
        --mlx-model ~/.config/local-llm-foundry/models/mlx/native/Scarlett-Opus-oQ4e-MLX

See docs/reference/rapid-mlx-mtp-evidence.md for the requalification procedure
this feeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn

DEFAULT_SIDECAR_ROOT = (
    Path.home()
    / ".config"
    / "local-llm-foundry"
    / "models"
    / "rapid-mlx"
    / "mtp-sidecars"
)

# The splitter writes a standalone drafter as `model.safetensors`; the app's
# managed sidecar layout uses `mtp.safetensors` (see sidecar_inventory.rs).
SPLITTER_OUTPUT_NAME = "model.safetensors"
SIDECAR_NAME = "mtp.safetensors"

# A valid head reads ~ +0.56 here; a stale/inverted extraction reads ~ -0.44.
# We only assert the sign, because the magnitude is model-specific and
# asserting it would make this check fail on models it should pass.
NORM_MARKER = "pre_fc_norm"
STALE_EXTRACTION_MEAN = -0.44

# Qwen3.5/3.8/3.6-Next family splitters, imported from the managed rapid-mlx
# install — the same code rapid-mlx itself splits with. Dense and MoE trunk
# backbones use different expert layouts, so the driver picks by num_experts.
SPLITTER_DRIVER = """
import json, sys
from pathlib import Path
from rapid_mlx.models.mlx_vlm_vendored.speculative.drafters.qwen3_5_mtp.split import (
    Qwen3NextMTPSplitter, Qwen3_5MTPSplitter,
)
source, output, revision, q_bits, q_group_size = (
    sys.argv[1], sys.argv[2], sys.argv[3],
    int(sys.argv[4]) if sys.argv[4] != "None" else None,
    int(sys.argv[5]) if sys.argv[5] != "None" else None,
)
import mlx_vlm.utils as _vlu
source_path = Path(_vlu.get_model_path(source, revision=revision or None))
config = json.loads((source_path / "config.json").read_text())
text = config.get("text_config") or config
num_experts = int(text.get("num_experts") or 0)
splitter = Qwen3NextMTPSplitter() if num_experts > 0 else Qwen3_5MTPSplitter()
print(splitter.output_model_type)
kwargs = {}
if revision:
    kwargs["revision"] = revision
if q_bits is not None:
    kwargs["q_bits"] = q_bits
if q_group_size is not None:
    kwargs["q_group_size"] = q_group_size
splitter.split(source=source, output=str(output), **kwargs)
"""


def die(message: str) -> NoReturn:
    sys.stderr.write(f"error: {message}\n")
    raise SystemExit(1)


def resolve_interpreter(explicit: str | None) -> str:
    """Find a Python that can import mlx and the rapid-mlx package.

    The splitter needs mlx, safetensors, and the rapid-mlx install. On this
    machine rapid-mlx is a uv tool install, so its interpreter has them and
    the system python3 does not.
    """
    if explicit:
        return explicit
    probe = subprocess.run(
        ["which", "rapid-mlx"],
        capture_output=True,
        text=True,
    )
    candidate = Path(probe.stdout.strip()) if probe.returncode == 0 else None
    if candidate and candidate.is_file():
        shebang = candidate.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        if shebang.startswith("#!"):
            return shebang[2:].strip()
    return sys.executable


def check_imports(interpreter: str) -> None:
    probe = subprocess.run(
        [
            interpreter,
            "-c",
            "import mlx.core, safetensors, rapid_mlx"
            ".models.mlx_vlm_vendored.speculative.drafters"
            ".qwen3_5_mtp.split",
        ],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        die(
            f"{interpreter} cannot import the rapid-mlx MTP splitter framework:\n"
            f"{probe.stderr.strip()}\nPoint --python at the interpreter rapid-mlx "
            "runs under (uv tool installs keep it next to the rapid-mlx entrypoint)."
        )


def slugify(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in text.lower()).strip("-") or "trunk"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def norm_means(interpreter: str, sidecar: Path) -> dict[str, float]:
    """Read the pre_fc_norm_* means out of the built sidecar.

    Done in a subprocess with the mlx-capable interpreter so this wrapper itself
    stays runnable under a bare system python3.
    """
    snippet = (
        "import json,sys;import mlx.core as mx;"
        "w=mx.load(sys.argv[1]);"
        f"print(json.dumps({{k:float(v.mean()) for k,v in w.items() if '{NORM_MARKER}' in k}}))"
    )
    probe = subprocess.run(
        [interpreter, "-c", snippet, str(sidecar)],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        die(f"Could not read norms back from {sidecar}:\n{probe.stderr.strip()}")
    try:
        return json.loads(probe.stdout)
    except json.JSONDecodeError as exc:
        die(f"Norm probe returned malformed JSON: {exc}")


def validate_norms(means: dict[str, float]) -> dict[str, Any]:
    """Refuse a head whose fc-input normalization is inverted.

    Absence of the markers is reported, not treated as a pass: a future
    architecture may not have them, and that is a reason to look, not to ship.
    """
    if not means:
        die(
            f"No '{NORM_MARKER}*' tensors found in the built head, so the norm-shift "
            "check could not run. Refusing to certify a head that cannot be "
            "validated. Inspect the sidecar's tensor names before using it; if this "
            "architecture genuinely lacks these tensors, that fact belongs in "
            "docs/reference/rapid-mlx-mtp-evidence.md first."
        )
    inverted = {name: mean for name, mean in means.items() if mean <= 0}
    if inverted:
        detail = ", ".join(f"{name}={mean:+.4f}" for name, mean in sorted(inverted.items()))
        die(
            f"Norm shift is missing or inverted: {detail}.\nA valid head reads a "
            f"positive mean (~+0.56); a stale extraction reads "
            f"~{STALE_EXTRACTION_MEAN}. This head would give ~0% draft acceptance, "
            "which is exactly the failure that produced the void receipts. Update "
            "the managed rapid-mlx install and rebuild; do not use this file."
        )
    return {
        "pre_fc_norm_means": means,
        "all_positive": True,
        "expected_mean_stale_extractor": STALE_EXTRACTION_MEAN,
        "method": "mean of every pre_fc_norm_* tensor in the built sidecar",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a managed, out-of-trunk MTP draft sidecar.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bf16-source",
        required=True,
        help="HF repo id or local path of the BF16 source that carries the mtp.* tensors.",
    )
    parser.add_argument(
        "--mlx-model",
        required=True,
        help="Quantized MLX trunk directory. Supplies the quantization config; is NOT modified.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Immutable Hugging Face commit/revision for the BF16 source.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=f"Sidecar output directory. Default: {DEFAULT_SIDECAR_ROOT}/<trunk-slug>/",
    )
    parser.add_argument("--bits", type=int, default=None, help="Override quantization bits.")
    parser.add_argument("--group-size", type=int, default=None, help="Override group size.")
    parser.add_argument(
        "--python",
        default=None,
        help="Interpreter that can import mlx and rapid_mlx. Default: the one rapid-mlx runs under.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing sidecar at --out.",
    )
    args = parser.parse_args()

    trunk = Path(args.mlx_model).expanduser().resolve()
    if not trunk.is_dir():
        die(f"--mlx-model is not a directory: {trunk}")

    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else DEFAULT_SIDECAR_ROOT / slugify(trunk.name)
    )
    # The whole point of managed placement. An in-trunk sidecar double-shifts
    # the trunk's norms on the next load, so refusing here is not pedantry.
    if out_dir == trunk or trunk in out_dir.parents:
        die(
            f"--out would place the sidecar inside the trunk ({out_dir}).\nmlx_lm globs "
            "model*.safetensors when loading a trunk, so an in-trunk sidecar is read as "
            "a trunk shard and double-shifts every trunk RMSNorm weight. Choose a "
            "directory outside the model."
        )

    sidecar_path = out_dir / SIDECAR_NAME
    if sidecar_path.exists() and not args.force:
        die(f"{sidecar_path} already exists. Pass --force to rebuild it.")

    interpreter = resolve_interpreter(args.python)
    check_imports(interpreter)

    # The splitter's trunk-quantization default: take the trunk's own quant
    # config so the drafter matches the quantization convention the trunk
    # serves with, unless the caller overrides either knob explicitly.
    if args.bits is None:
        trunk_config = trunk / "config.json"
        if trunk_config.exists():
            try:
                config = json.loads(trunk_config.read_text())
                quant = config.get("quantization") or {}
                if isinstance(quant, dict) and quant.get("bits"):
                    args.bits = int(quant["bits"])
            except (json.JSONDecodeError, ValueError, OSError):
                pass  # trunk quantization stays splitter-default

    staging = Path(
        tempfile.mkdtemp(prefix=f"mtp-split-{slugify(trunk.name)}-")
    )
    command = [
        interpreter,
        "-c",
        SPLITTER_DRIVER,
        args.bf16_source,
        str(staging),
        args.revision or "",
        str(args.bits) if args.bits is not None else "None",
        str(args.group_size) if args.group_size is not None else "None",
    ]

    sys.stderr.write("Splitting with upstream MTPSplitter (managed rapid-mlx install)\n")
    sys.stderr.write(f"  interpreter: {interpreter}\n")
    result = subprocess.run(command)

    try:
        if result.returncode != 0:
            die(f"Splitter exited {result.returncode}; nothing was written to {out_dir}.")
        staged_weights = staging / SPLITTER_OUTPUT_NAME
        if not staged_weights.exists():
            die(
                f"Splitter reported success but {staged_weights} does not exist. "
                "Upstream may have changed its output filename; check "
                "SPLITTER_OUTPUT_NAME in this wrapper against mtp_split.py."
            )

        validation = validate_norms(norm_means(interpreter, staged_weights))

        out_dir.mkdir(parents=True, exist_ok=True)
        # Install the whole standalone drafter: weights under the managed name,
        # plus the splitter's config.json and tokenizer files the runtime needs.
        for item in staging.iterdir():
            target = out_dir / (SIDECAR_NAME if item.name == SPLITTER_OUTPUT_NAME else item.name)
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    # Which splitter ran is the last line the driver printed.
    splitter_name = "qwen3_5_mtp"
    if result.stdout:
        last = result.stdout.strip().splitlines()[-1].strip()
        if last:
            splitter_name = last

    provenance = {
        "schema_version": 2,
        "kind": "mtp_sidecar",
        "status": "built_unvalidated_online",
        "repair_mode": "direct_parent",
        "note": (
            "Built by scripts/build-mtp-head.py using upstream's MTPSplitter. The "
            "norm check below is an offline sanity check, not a qualification: "
            "served acceptance is only established by "
            "scripts/rapid-mlx-requalify-spec-decode.mjs."
        ),
        "file": SIDECAR_NAME,
        "sha256": sha256_of(sidecar_path),
        "source": {
            "bf16_source": args.bf16_source,
            "revision": args.revision,
            "trunk": str(trunk),
            "extracted_with": f"rapid-mlx MTPSplitter ({splitter_name})",
        },
        "validation": validation,
        "known_good_positive_control": "mlx-community/Qwen3.6-27B-MTP-4bit",
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "provenance.json").write_text(f"{json.dumps(provenance, indent=2)}\n")

    means_map: dict[str, float] = validation["pre_fc_norm_means"]
    means = ", ".join(
        f"{name.split('.')[-2]}={mean:+.4f}" for name, mean in sorted(means_map.items())
    )
    sys.stderr.write(f"\nSidecar: {sidecar_path}\n")
    sys.stderr.write(f"Splitter: {splitter_name}\n")
    sys.stderr.write(f"Norm check passed ({means})\n")
    sys.stderr.write(f"Provenance: {out_dir / 'provenance.json'}\n")
    sys.stderr.write(
        "\nThis head is not qualified yet. Measure it served:\n"
        f"  node scripts/rapid-mlx-requalify-spec-decode.mjs \\\n"
        f"    --model {trunk} \\\n"
        f"    --speculative-control-model mlx-community/Qwen3.6-27B-MTP-4bit \\\n"
        f"    --speculative-model {out_dir} \\\n"
        f"    --profile-alias <hf-alias-for-this-family> \\\n"
        f"    --out tmp/requalify-$(date +%Y%m%d)\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
