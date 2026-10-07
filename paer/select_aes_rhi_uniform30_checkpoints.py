#!/usr/bin/env python3
"""Rerun RHI checkpoint selection with a uniform 30-step attack budget.

This script never modifies historical selection directories or training
checkpoints.  It selects Mixed-AT-RHI, PAER-RHI, and the two retrained PAER
ablations into new ``*_uniform30_*`` directories, then writes a compact
comparison telling the caller whether full evaluations must be repeated.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paer.rhi_experiment_utils import read_json, save_json


TARGETS = {
    "mixed_at_rhi": (
        "aes_mixed_at_rhi_seed{seed}",
        "aes_mixed_at_rhi_checkpoint_selection_uniform30_seed{seed}",
    ),
    "paer_rhi_v3": (
        "aes_paer_rhi_v3_seed{seed}",
        "aes_paer_rhi_v3_checkpoint_selection_uniform30_seed{seed}",
    ),
    "without_token_localization": (
        "aes_paer_rhi_v3_without_token_localization_seed{seed}",
        "aes_paer_rhi_v3_without_token_localization_checkpoint_selection_uniform30_seed{seed}",
    ),
    "without_routing": (
        "aes_paer_rhi_v3_without_routing_seed{seed}",
        "aes_paer_rhi_v3_without_routing_checkpoint_selection_uniform30_seed{seed}",
    ),
}


def _selection_cli(
    reference_arguments: dict,
    *,
    training_dir: Path,
    selection_dir: Path,
    training_inputs: Path,
) -> list[str]:
    """Serialize a frozen RHI protocol while changing only the step budget."""
    arguments = dict(reference_arguments)
    arguments.pop("attack", None)  # internal default, not a public CLI option
    arguments.update(
        defense_output_dir=str(training_dir),
        selection_output_dir=str(selection_dir),
        subset_ids_path=str(selection_dir / "subset_ids.json"),
        training_inputs=str(training_inputs),
        selection_steps=30,
        rudimentary_selection_steps=30,
        injection_selection_steps=30,
    )
    command: list[str] = []
    for key, value in arguments.items():
        if value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        command.extend([flag] if value is True else [flag, str(value)])
    return command


def _find_historical_ablation_selection(outputs: Path, variant: str) -> Path | None:
    candidates = sorted(
        outputs.glob(
            f"aes_paer_rhi_v3_retrained_ablations_seed*/{variant}/selection/"
            "best_checkpoint.json"
        )
    ) + sorted(
        outputs.glob(
            f"aes_paer_rhi_v3_retrained_ablations_seed*_run*/{variant}/selection/"
            "best_checkpoint.json"
        )
    )
    return candidates[-1] if candidates else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outputs-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    outputs = args.outputs_root
    old_mixed = outputs / f"aes_mixed_at_rhi_checkpoint_selection_seed{args.seed}"
    old_paer = outputs / f"aes_paer_rhi_v3_checkpoint_selection_seed{args.seed}"
    for path in (old_mixed, old_paer):
        if not (path / "rhi_run_binding.json").is_file():
            raise FileNotFoundError(f"Reference selection binding not found: {path}")

    reference_bindings = {
        "mixed_at_rhi": read_json(old_mixed / "rhi_run_binding.json"),
        "paer_rhi_v3": read_json(old_paer / "rhi_run_binding.json"),
    }
    reference_bindings["without_token_localization"] = reference_bindings[
        "paer_rhi_v3"
    ]
    reference_bindings["without_routing"] = reference_bindings["paer_rhi_v3"]

    historical_best: dict[str, Path | None] = {
        "mixed_at_rhi": old_mixed / "best_checkpoint.json",
        "paer_rhi_v3": old_paer / "best_checkpoint.json",
        "without_token_localization": _find_historical_ablation_selection(
            outputs, "without_token_localization"
        ),
        "without_routing": _find_historical_ablation_selection(
            outputs, "without_routing"
        ),
    }
    rows = []
    for name, (training_pattern, selection_pattern) in TARGETS.items():
        training_dir = outputs / training_pattern.format(seed=args.seed)
        if not training_dir.is_dir():
            raise FileNotFoundError(f"Training output not found: {training_dir}")
        is_ablation = name.startswith("without_")
        training_inputs = training_dir / (
            "reference_rhi_training_inputs.json" if is_ablation
            else "rhi_training_inputs.json"
        )
        if not training_inputs.is_file():
            raise FileNotFoundError(f"Training provenance not found: {training_inputs}")
        selection_dir = outputs / selection_pattern.format(seed=args.seed)
        command = [
            sys.executable,
            str(ROOT / "paer/select_aes_rhi_checkpoint.py"),
            *_selection_cli(
                reference_bindings[name]["arguments"],
                training_dir=training_dir,
                selection_dir=selection_dir,
                training_inputs=training_inputs,
            ),
        ]
        if args.no_progress:
            command.append("--no-progress")
        print(f"[UNIFORM30:{name}] {subprocess.list2cmdline(command)}", flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, check=True)
        selected = read_json(selection_dir / "best_checkpoint.json")
        binding = read_json(selection_dir / "rhi_run_binding.json")
        for key in (
            "selection_steps",
            "rudimentary_selection_steps",
            "injection_selection_steps",
        ):
            if int(binding["arguments"][key]) != 30:
                raise ValueError(f"Non-uniform selection budget persisted: {key}")
        previous_path = historical_best[name]
        previous = read_json(previous_path) if previous_path and previous_path.is_file() else None
        rows.append(
            {
                "model": name,
                "historical_selection": str(previous_path) if previous else None,
                "historical_checkpoint": previous.get("checkpoint_name") if previous else None,
                "uniform30_selection": str(selection_dir / "best_checkpoint.json"),
                "uniform30_checkpoint": selected["checkpoint_name"],
                "checkpoint_changed": (
                    previous is None
                    or previous.get("checkpoint_path") != selected.get("checkpoint_path")
                ),
                "clean_qwk": selected["clean_qwk"],
                "rhi_macro_subset_asr": selected["rhi_macro_subset_asr"],
            }
        )

    if args.dry_run:
        return 0
    summary_dir = outputs / f"aes_rhi_uniform30_checkpoint_selection_seed{args.seed}"
    summary_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        summary_dir / "selection_comparison.json",
        {
            "protocol": "uniform_30_step_rhi_checkpoint_selection",
            "seed": args.seed,
            "full_reevaluation_required": any(
                row["checkpoint_changed"] for row in rows
            ),
            "models": rows,
        },
    )
    lines = [
        "# Uniform 30-step RHI checkpoint selection",
        "",
        "Rudimentary, HotFlip, and Injection each use a 30-step selection ceiling.",
        "",
        "| Model | Historical checkpoint | Uniform-30 checkpoint | Changed | Clean QWK | RHI subset ASR |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['historical_checkpoint'] or 'NA'} | "
            f"{row['uniform30_checkpoint']} | {str(row['checkpoint_changed']).lower()} | "
            f"{row['clean_qwk']:.4f} | {row['rhi_macro_subset_asr']:.4f} |"
        )
    (summary_dir / "selection_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"[SAVED] {summary_dir / 'selection_comparison.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
