#!/usr/bin/env python3
"""Select a Mixed-AT/PAER checkpoint using clean QWK and RH macro ASR.

MLM-guided is deliberately absent from this script so it cannot leak into
training, validation, hyperparameter tuning, or checkpoint selection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from whitebox.select_aes_hotflip_defense_checkpoint import (
    _evaluate_clean_if_needed,
    _evaluate_subset_if_needed,
    _load_clean_qwk,
    _load_subset_summary,
    _validate_args,
    build_parser as build_single_attack_parser,
    create_or_load_debug_subset,
    discover_checkpoint_candidates,
)


def build_parser():
    parser = build_single_attack_parser("hotflip")
    parser.description = (
        "Select an RH-trained checkpoint by clean QWK gate and the mean of "
        "Rudimentary/HotFlip subset ASR; MLM is forbidden."
    )
    parser.set_defaults(
        defense_output_dir=REPO_ROOT / "outputs" / "aes_paer_rh_seed42",
        selection_output_dir=(
            REPO_ROOT / "outputs" / "aes_paer_rh_checkpoint_selection_seed42"
        ),
        selection_steps=10,
        # A 1024-token DeBERTa/PAER forward at the inherited batch size of 16
        # nearly exhausts a 24 GB RTX 3090 in float32.  This one value is
        # propagated to clean, original, and attack-candidate scoring.
        batch_size=4,
    )
    parser.add_argument(
        "--rudimentary-selection-steps",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--max-checkpoint-step",
        type=int,
        default=1400,
        help=(
            "Maximum explicit gstep checkpoint eligible for selection. "
            "best/final aliases are excluded when this cap is active because "
            "their training step is not encoded in the directory name. Use "
            "0 only when every completed checkpoint may be compared."
        ),
    )
    return parser


def restrict_candidates_to_common_budget(
    candidates: list[Path],
    max_checkpoint_step: int,
) -> tuple[list[Path], list[str]]:
    """Keep only unambiguous checkpoints within a shared training budget."""
    if max_checkpoint_step < 0:
        raise ValueError("max_checkpoint_step must be zero or greater")
    if max_checkpoint_step == 0:
        return candidates, []

    eligible: list[Path] = []
    excluded: list[str] = []
    for checkpoint in candidates:
        name = checkpoint.name
        if name.startswith("gstep") and name[5:].isdigit():
            if int(name[5:]) <= max_checkpoint_step:
                eligible.append(checkpoint)
            else:
                excluded.append(name)
        else:
            # The alias may have been written after the common budget. Without
            # saved-step metadata, accepting it would make the protocol
            # impossible to audit.
            excluded.append(name)
    if not eligible:
        raise RuntimeError(
            "No explicit gstep checkpoint is within max_checkpoint_step="
            f"{max_checkpoint_step}"
        )
    return eligible, excluded


def _run_attack_subset(
    *,
    attack: str,
    steps: int,
    checkpoint: Path,
    subset_csv: Path,
    output_dir: Path,
    args,
) -> dict:
    previous_attack = args.attack
    previous_steps = args.selection_steps
    try:
        args.attack = attack
        args.selection_steps = steps
        _evaluate_subset_if_needed(
            checkpoint=checkpoint,
            subset_csv=subset_csv,
            out_dir=output_dir,
            args=args,
        )
    finally:
        args.attack = previous_attack
        args.selection_steps = previous_steps
    if args.dry_run:
        return {"asr": None, "avg_delta": None}
    return _load_subset_summary(output_dir / "asr_summary.json")


def main() -> int:
    args = build_parser().parse_args()
    _validate_args(args)
    if args.rudimentary_selection_steps <= 0:
        raise ValueError("rudimentary_selection_steps must be greater than zero")
    if not args.c0_checkpoint.is_dir():
        raise FileNotFoundError(f"C0 checkpoint not found: {args.c0_checkpoint}")
    if not args.valid_csv.is_file():
        raise FileNotFoundError(f"Validation CSV not found: {args.valid_csv}")
    discovered = discover_checkpoint_candidates(args.defense_output_dir)
    candidates, excluded_candidates = restrict_candidates_to_common_budget(
        discovered,
        args.max_checkpoint_step,
    )
    print(
        f"[CANDIDATES] {len(candidates)} eligible checkpoints; "
        f"excluded={excluded_candidates}",
        flush=True,
    )

    subset_csv = args.selection_output_dir / f"debug_subset_{args.subset_size}.csv"
    if not args.dry_run:
        args.selection_output_dir.mkdir(parents=True, exist_ok=True)
        create_or_load_debug_subset(
            args.valid_csv,
            args.subset_ids_path,
            subset_csv,
            subset_size=args.subset_size,
            seed=args.subset_seed,
        )

    c0_output = args.selection_output_dir / "c0_reference"
    _evaluate_clean_if_needed(
        checkpoint=args.c0_checkpoint,
        valid_csv=args.valid_csv,
        out_dir=c0_output,
        args=args,
    )
    if args.dry_run:
        print("[DRY-RUN] MLM-guided is excluded from checkpoint selection.")
        return 0
    c0_qwk = _load_clean_qwk(c0_output / "clean_qwk.json")
    minimum_qwk = c0_qwk - args.qwk_tolerance
    print(f"[QWK GATE] C0={c0_qwk:.4f}; minimum={minimum_qwk:.4f}")

    rows = []
    for checkpoint in candidates:
        candidate_root = args.selection_output_dir / "candidates" / checkpoint.name
        clean_output = candidate_root / "clean"
        _evaluate_clean_if_needed(
            checkpoint=checkpoint,
            valid_csv=args.valid_csv,
            out_dir=clean_output,
            args=args,
        )
        clean_qwk = _load_clean_qwk(clean_output / "clean_qwk.json")
        row = {
            "checkpoint_name": checkpoint.name,
            "checkpoint_path": str(checkpoint.resolve()),
            "clean_qwk": clean_qwk,
            "minimum_qwk": minimum_qwk,
            "eligible": clean_qwk >= minimum_qwk,
            "rudimentary_subset_asr": None,
            "hotflip_subset_asr": None,
            "rh_macro_subset_asr": None,
        }
        if row["eligible"]:
            rudimentary = _run_attack_subset(
                attack="rudimentary",
                steps=args.rudimentary_selection_steps,
                checkpoint=checkpoint,
                subset_csv=subset_csv,
                output_dir=candidate_root / "rudimentary",
                args=args,
            )
            hotflip = _run_attack_subset(
                attack="hotflip",
                steps=args.selection_steps,
                checkpoint=checkpoint,
                subset_csv=subset_csv,
                output_dir=candidate_root / "hotflip",
                args=args,
            )
            row["rudimentary_subset_asr"] = float(rudimentary["asr"])
            row["hotflip_subset_asr"] = float(hotflip["asr"])
            row["rudimentary_subset_avg_delta"] = float(rudimentary["avg_delta"])
            row["hotflip_subset_avg_delta"] = float(hotflip["avg_delta"])
            row["rh_macro_subset_asr"] = 0.5 * (
                row["rudimentary_subset_asr"] + row["hotflip_subset_asr"]
            )
        rows.append(row)

    eligible = [
        row for row in rows if row["eligible"] and row["rh_macro_subset_asr"] is not None
    ]
    if not eligible:
        raise RuntimeError("No checkpoint passed the clean QWK gate")
    selected = min(
        eligible,
        key=lambda row: (
            row["rh_macro_subset_asr"],
            -row["clean_qwk"],
            row["checkpoint_name"],
        ),
    )
    summary = {
        "protocol": {
            "selection_attacks": ["rudimentary", "hotflip"],
            "held_out_attack": "mlm_guided",
            "mlm_used_for_selection": False,
            "c0_qwk": c0_qwk,
            "qwk_tolerance": args.qwk_tolerance,
            "minimum_qwk": minimum_qwk,
            "subset_size": args.subset_size,
            "subset_seed": args.subset_seed,
            "attack_seed": args.seed,
            "evaluation_batch_size": args.batch_size,
            "evaluation_dtype": args.dtype,
            "hotflip_selection_steps": args.selection_steps,
            "rudimentary_selection_steps": args.rudimentary_selection_steps,
            "max_checkpoint_step": args.max_checkpoint_step or None,
            "explicit_gstep_only": bool(args.max_checkpoint_step),
            "excluded_checkpoint_names": excluded_candidates,
        },
        "selected_checkpoint": selected,
        "candidates": rows,
    }
    (args.selection_output_dir / "checkpoint_selection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    best = {
        "checkpoint_name": selected["checkpoint_name"],
        "checkpoint_path": selected["checkpoint_path"],
        "clean_qwk": selected["clean_qwk"],
        "rudimentary_subset_asr": selected["rudimentary_subset_asr"],
        "hotflip_subset_asr": selected["hotflip_subset_asr"],
        "rh_macro_subset_asr": selected["rh_macro_subset_asr"],
        "mlm_used_for_selection": False,
        "max_checkpoint_step": args.max_checkpoint_step or None,
    }
    best_path = args.selection_output_dir / "best_checkpoint.json"
    best_path.write_text(
        json.dumps(best, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[SELECTED] {selected['checkpoint_name']} "
        f"QWK={selected['clean_qwk']:.4f} "
        f"RH macro ASR={selected['rh_macro_subset_asr']:.4f}"
    )
    print(f"[SAVED] {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
