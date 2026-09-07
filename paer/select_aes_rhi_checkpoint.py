#!/usr/bin/env python3
"""R/H/I checkpoint selection with immutable inputs and resumable evaluations."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paer.select_aes_rh_checkpoint import build_parser as rh_parser, restrict_candidates_to_common_budget
from whitebox.select_aes_hotflip_defense_checkpoint import (
    _validate_args, _evaluate_clean_if_needed, _evaluate_subset_if_needed,
    _load_clean_qwk, _load_subset_summary, create_or_load_debug_subset,
    discover_checkpoint_candidates,
)
from paer.rhi_experiment_utils import bind_directory, checkpoint_identity, read_json, save_json, sha256, rhi_macro


def build_parser():
    parser = rh_parser()
    parser.description = "Select an RHI model with three equally weighted families; MLM forbidden."
    parser.set_defaults(defense_output_dir=ROOT / "outputs/aes_paer_rhi_v3_seed42",
                        selection_output_dir=ROOT / "outputs/aes_paer_rhi_v3_checkpoint_selection_seed42",
                        subset_ids_path=None, batch_size=4)
    parser.add_argument("--injection-selection-steps", type=int, default=30)
    return parser


def evaluate_completed(*, checkpoint, csv, out, args, attack=None):
    """Only reuse successful child runs with all output files, not a partial summary."""
    marker = out / "completed_result_hashes.json"
    if marker.exists():
        for name, digest in read_json(marker).items():
            if sha256(out / name) != digest:
                raise ValueError(f"Selection result was modified: {out / name}")
        print(f"[REUSE] {out}", flush=True)
        return
    previous_force = args.force
    args.force = True  # rerun only unfinished output owned by this bound RHI run
    try:
        if attack is None:
            _evaluate_clean_if_needed(checkpoint=checkpoint, valid_csv=csv, out_dir=out, args=args)
            required = ["clean_qwk.json"]
        else:
            _evaluate_subset_if_needed(checkpoint=checkpoint, subset_csv=csv, out_dir=out, args=args)
            required = ["asr_summary.json", "run_manifest.json"]
            if attack == "injection_family":
                required += ["injection_family_summary.json", "injection_external_details.json", "injection_self_dup_details.json"]
            else:
                required += [f"{attack}_details.json"]
        save_json(marker, {name: sha256(out / name) for name in required})
    finally:
        args.force = previous_force


def main():
    args = build_parser().parse_args()
    _validate_args(args)
    if args.force:
        raise ValueError("RHI selector does not overwrite; use a new selection-output-dir")
    if args.injection_selection_steps <= 0 or args.rudimentary_selection_steps <= 0:
        raise ValueError("Selection steps must be positive")
    if args.subset_ids_path is None:
        args.subset_ids_path = args.selection_output_dir / "subset_ids.json"
    if not args.subset_ids_path.resolve().is_relative_to(args.selection_output_dir.resolve()):
        raise ValueError("Use subset IDs within this new selection directory; historical IDs stay untouched")
    inputs = read_json(args.defense_output_dir / "rhi_training_inputs.json")
    if inputs["valid_sha256"] != sha256(args.valid_csv):
        raise ValueError("Selection CSV differs from the recorded training development split")
    candidates, excluded = restrict_candidates_to_common_budget(
        discover_checkpoint_candidates(args.defense_output_dir), args.max_checkpoint_step)
    protocol = {
        "protocol": "rhi_selection_v1", "selection_attacks": ["rudimentary", "hotflip", "injection_family"],
        "held_out_attack": "mlm_guided", "mlm_used_for_selection": False,
        "aggregation": "(R + H + (external + self_dup)/2)/3",
        "arguments": {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
                      if k not in ("dry_run", "no_progress", "online", "force")},
        "valid_sha256": sha256(args.valid_csv), "bank_sha256": sha256(args.injection_sentence_bank),
        "training_inputs": inputs, "c0_weights": checkpoint_identity(args.c0_checkpoint),
        "candidate_weights": {p.name: checkpoint_identity(p) for p in candidates},
        "evaluation_source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(
            list((ROOT / "text_scoring_adv_training/evaluation/aes").rglob("*.py"))
            + [ROOT / "paer/modeling_paer.py", ROOT / "paer/modeling_paer_v3.py",
               ROOT / "whitebox/eval_hotflip_defended.py", Path(__file__)])},
        "evaluation_status": "development checkpoint selection, not independent test",
    }
    print(f"[RHI] {len(candidates)} candidates; excluded={excluded}; batch={args.batch_size}", flush=True)
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return 0
    bind_directory(args.selection_output_dir, protocol)
    best_path = args.selection_output_dir / "best_checkpoint.json"
    if best_path.exists():
        print(f"[COMPLETE] {best_path}")
        return 0
    subset = args.selection_output_dir / f"selection_subset_{args.subset_size}.csv"
    create_or_load_debug_subset(args.valid_csv, args.subset_ids_path, subset,
                               subset_size=args.subset_size, seed=args.subset_seed)
    c0_dir = args.selection_output_dir / "c0_reference"
    evaluate_completed(checkpoint=args.c0_checkpoint, csv=args.valid_csv, out=c0_dir, args=args)
    c0_qwk = _load_clean_qwk(c0_dir / "clean_qwk.json")
    minimum_qwk = c0_qwk - args.qwk_tolerance
    rows = []
    steps = {"rudimentary": args.rudimentary_selection_steps, "hotflip": args.selection_steps,
             "injection_family": args.injection_selection_steps}
    for checkpoint in candidates:
        root = args.selection_output_dir / "candidates" / checkpoint.name
        evaluate_completed(checkpoint=checkpoint, csv=args.valid_csv, out=root / "clean", args=args)
        qwk = _load_clean_qwk(root / "clean/clean_qwk.json")
        row = {"checkpoint_name": checkpoint.name, "checkpoint_path": str(checkpoint.resolve()),
               "clean_qwk": qwk, "eligible": qwk >= minimum_qwk, "rhi_macro_subset_asr": None}
        if row["eligible"]:
            for attack, budget in steps.items():
                args.attack, args.selection_steps = attack, budget
                out = root / attack
                evaluate_completed(checkpoint=checkpoint, csv=subset, out=out, args=args, attack=attack)
                metric = _load_subset_summary(out / "asr_summary.json", attack)
                row[f"{attack}_subset_asr"] = float(metric["asr"])
                row[f"{attack}_subset_avg_delta"] = float(metric["avg_delta"])
                if attack == "injection_family":
                    parts = {m["attack"]: m for m in read_json(out / "asr_summary.json")}
                    external = parts["injection_external"]
                    duplicate = parts["injection_self_dup"]
                    if external["n_essays"] != duplicate["n_essays"]:
                        raise ValueError("Injection subattacks must evaluate identical sample counts")
                    row["rhi_macro_subset_asr"] = rhi_macro(
                        row["rudimentary_subset_asr"], row["hotflip_subset_asr"],
                        external["asr"], duplicate["asr"])
            print(f"[CANDIDATE] {checkpoint.name} QWK={qwk:.4f} RHI={row['rhi_macro_subset_asr']:.4f}", flush=True)
        rows.append(row)
        save_json(args.selection_output_dir / "checkpoint_selection_progress.json",
                  {"candidates": rows, "minimum_qwk": minimum_qwk})
    eligible = [r for r in rows if r["eligible"]]
    summary = {"protocol": protocol, "c0_qwk": c0_qwk, "minimum_qwk": minimum_qwk, "candidates": rows}
    if not eligible:
        save_json(args.selection_output_dir / "checkpoint_selection_summary.json", summary)
        raise RuntimeError("No RHI checkpoint passed the clean gate; no fallback model was selected")
    best = min(eligible, key=lambda r: (r["rhi_macro_subset_asr"], -r["clean_qwk"], r["checkpoint_name"]))
    best = {**best, "mlm_used_for_selection": False, "max_checkpoint_step": args.max_checkpoint_step,
            "selection_attacks": list(steps), "selection_csv_sha256": sha256(args.valid_csv)}
    save_json(args.selection_output_dir / "checkpoint_selection_summary.json", {**summary, "selected_checkpoint": best})
    save_json(best_path, best)
    print(f"[SELECTED] {best_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
