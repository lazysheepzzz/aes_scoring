#!/usr/bin/env python3
"""Re-optimize R/H/I attacks against frozen PAER-v3 with routing disabled.

Only the in-memory correction_scale is zeroed. Token evidence, attention,
global scoring and checkpoint files are preserved. Candidate scores AND
HotFlip gradients therefore target route-off, never the original routed score.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paer.rhi_experiment_utils import bind_directory, read_json, save_json, sha256
from paer.routing_experiment_utils import DEFAULT_EVALUATION, code_identity, frozen_reference


def disable_routing(model):
    """Runtime-only ablation. Do not remove token evidence or reload B0."""
    from paer.modeling_paer_v3 import PAERV3ForEssayScoring
    if not isinstance(model, PAERV3ForEssayScoring):
        raise TypeError("Route-off evaluation requires PAERV3ForEssayScoring")
    original_scale = model.correction_scale
    model.correction_scale = 0.0
    return original_scale


def worker(config_path):
    """One subprocess per family; release all model/CUDA allocations on exit."""
    import torch
    from whitebox.eval_hotflip_defended import build_parser, _configure_environment, _evaluate_clean
    from text_scoring_adv_training.evaluation.aes import scorer as scorer_module
    from text_scoring_adv_training.evaluation.aes import run_attacks

    config = read_json(config_path)
    args = build_parser().parse_args([])
    for name, value in config.items():
        setattr(args, name, value)
    for name in ("checkpoint", "valid", "out", "injection_sentence_bank", "hf_home"):
        setattr(args, name, Path(getattr(args, name)))
    _configure_environment(args)
    original_class, original_attack_class = scorer_module.AESScorer, run_attacks.AESScorer

    class RouteOffScorer(original_class):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            # Verify equality against the original forward's architectural
            # counterfactual, not against the global-only pretrained head.
            probe = self.tokenizer("This is a short scoring check.", return_tensors="pt")
            probe = {k: v.to(self.device) for k, v in probe.items() if k in ("input_ids", "attention_mask")}
            with torch.no_grad():
                before = self.model(**probe).base_logits
                disable_routing(self.model)
                after = self.model(**probe)
                if not torch.allclose(after.logits, before, atol=1e-5, rtol=1e-5):
                    raise ValueError("Route-off score does not equal base_logits")
                if torch.count_nonzero(after.correction).item() != 0:
                    raise ValueError("Routing correction is not zero")

    # Scoped in-process substitution: shared historical evaluator files stay
    # untouched. Both its clean path and its attack factory receive this scorer.
    scorer_module.AESScorer = run_attacks.AESScorer = RouteOffScorer
    try:
        clean = _evaluate_clean(args)
        save_json(args.out / "clean_qwk.json", {**clean, "scoring_mode": "route_off"})
        attack_args = argparse.Namespace(**vars(args))
        attack_args.victim = str(args.checkpoint)
        attack_args.data = str(args.valid)
        attack_args.dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
        run_attacks.run(attack_args)
        manifest_path = args.out / "run_manifest.json"
        save_json(manifest_path, {**read_json(manifest_path),
                                 "scoring_mode": "route_off", "adaptive_route_off_attack": True,
                                 "intervention": "in-memory correction_scale=0; signed token aggregation retained",
                                 "checkpoint_files_modified": False})
    finally:
        scorer_module.AESScorer = original_class
        run_attacks.AESScorer = original_attack_class
    return 0


def run_config(reference, checkpoint, source_manifest, family, data, n, out, no_progress=False):
    params = source_manifest["attack_parameters"]
    common = {"n_steps": reference["steps"], "beam_size": 1, "n_sample_pos": 8,
              "top_k_per_pos": 2, "max_candidates_per_step": 16,
              "max_token_edit_rate": 0.1, "success_threshold": reference["success_threshold"]}
    for key in common:
        if key in params and params[key] is not None:
            common[key] = params[key]
    if common["n_steps"] != reference["steps"] or common["success_threshold"] != reference["success_threshold"]:
        raise ValueError("Frozen summary and attack budgets differ")
    if source_manifest["seed"] != reference["seed"] or source_manifest["n_essays"] != reference["n_essays"]:
        raise ValueError("Frozen attack seed/sample count differs")
    if Path(source_manifest["victim"]).resolve() != checkpoint.resolve():
        raise ValueError("Frozen attack victim differs")
    return {**common, "checkpoint": str(checkpoint), "valid": str(data), "out": str(out),
            "attack": family, "n_essays": n, "seed": reference["seed"],
            "dtype": reference["dtype"], "batch_size": reference["batch_size"],
            "device": reference["device"], "max_length": reference["max_length"],
            "injection_sentence_bank": str(ROOT / "injection/wikipedia_sentences_100.txt"),
            "hf_home": str(ROOT / ".cache/huggingface"), "online": False,
            "no_progress": no_progress, "current_python": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/aes_paer_rhi_v3_route_off_adaptive_seed42")
    parser.add_argument("--n-essays", type=int, help="First-N smoke only; use a separate smoke output directory")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_config:
        return worker(args.worker_config)
    reference, checkpoint = frozen_reference(args.evaluation_dir)
    n = args.n_essays if args.n_essays is not None else reference["n_essays"]
    if n <= 0 or n > reference["n_essays"]:
        parser.error("n-essays must be within the frozen evaluation sample count")
    if reference["max_length"] != 1024:
        raise ValueError("Existing scorer truncation is fixed at 1024")
    # A bounded first-N CSV ensures clean and attacked metrics use identical rows.
    with Path(reference["data"]).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)[:n]
    if len(rows) != n:
        raise ValueError("Frozen dataset has too few rows")
    data = args.output_dir / "evaluation_rows.csv"
    configs = {}
    for family in ("rudimentary", "hotflip", "injection_family"):
        src = read_json(args.evaluation_dir / "paer" / family / "run_manifest.json")
        configs[family] = run_config(reference, checkpoint, src, family, data, n,
                                     args.output_dir / family, args.no_progress)
    threshold_path = checkpoint.parent / "best_thresholds.json"
    protocol = {"experiment": "frozen_paer_v3_adaptive_route_off_v1", "reference": reference,
                "reference_directory": str(args.evaluation_dir.resolve()), "n_essays": n,
                "smoke": n < reference["n_essays"], "mlm_used": False,
                "intervention": "correction_scale=0 in memory; token evidence retained; no retraining/reselection",
                "thresholds_sha256": sha256(threshold_path) if threshold_path.exists() else None,
                "configs": {f: {k: v for k, v in c.items() if k != "no_progress"} for f, c in configs.items()},
                "code": code_identity("paer/evaluate_aes_paer_route_off.py", "paer/routing_experiment_utils.py"),
                "interpretation_limit": "Inference intervention on jointly trained PAER; not an independently trained ablation"}
    if args.dry_run:
        print(json.dumps({"n_essays": n, "checkpoint": str(checkpoint), "configs": configs}, indent=2))
        return 0
    bind_directory(args.output_dir, protocol)
    if not data.exists():
        with data.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        with data.open(encoding="utf-8", newline="") as handle:
            if list(csv.DictReader(handle)) != rows:
                raise ValueError("Saved evaluation subset changed")
    summaries = []
    for family, config in configs.items():
        out = Path(config["out"])
        marker = out / "completed_result_hashes.json"
        files = ["clean_qwk.json", "asr_summary.json", "run_manifest.json"]
        files += (["injection_family_summary.json", "injection_external_details.json", "injection_self_dup_details.json"]
                  if family == "injection_family" else [f"{family}_details.json"])
        if marker.exists():
            hashes = read_json(marker)
            if set(hashes) != set(files) or any(sha256(out / p) != h for p, h in hashes.items()):
                raise ValueError(f"Completed route-off result changed: {out}")
            print(f"[REUSE] {out}", flush=True)
        else:
            config_path = args.output_dir / f"{family}_worker_config.json"
            save_json(config_path, config)
            command = [sys.executable, str(Path(__file__).resolve()), "--worker-config", str(config_path)]
            print(f"[RUN route-off] {family}; batch={reference['batch_size']}; n={n}", flush=True)
            subprocess.run(command, check=True, env=os.environ.copy())
            result_manifest = read_json(out / "run_manifest.json")
            if result_manifest.get("scoring_mode") != "route_off" or result_manifest["n_essays"] != n:
                raise ValueError("Worker did not produce the requested route-off result")
            save_json(marker, {p: sha256(out / p) for p in files})
        filename = "injection_family_summary.json" if family == "injection_family" else "asr_summary.json"
        result = read_json(out / filename)
        if isinstance(result, list):
            result = result[0]
        routed = read_json(args.evaluation_dir / "paer" / family / filename)
        if isinstance(routed, list):
            routed = routed[0]
        summaries.append({"attack": family, "route_off": result,
                          "route_off_clean": read_json(out / "clean_qwk.json"),
                          "routed_reference_clean": read_json(args.evaluation_dir / "paer" / family / "clean_qwk.json") if not protocol["smoke"] else None,
                          "routed_reference": routed if not protocol["smoke"] else None})
    save_json(args.output_dir / "adaptive_route_off_summary.json", {"protocol": protocol, "attacks": summaries,
              "route_off_rhi_macro_asr": sum(s["route_off"]["asr"] for s in summaries) / 3})
    lines = ["# Adaptive route-off evaluation", "", f"Role: {reference['evaluation_role']}; n={n}; smoke={protocol['smoke']}",
             "Each mode is attacked separately. This is not a retrained ablation.", "",
             "| Attack | Route-off Clean QWK | Routed Clean QWK | Route-off ASR | Routed ASR (own attack) | Route-off mean delta |",
             "| --- | --- | --- | --- | --- | --- |"]
    for s in summaries:
        routed_asr = f"{s['routed_reference']['asr']:.4f}" if s["routed_reference"] else "NA (smoke)"
        routed_qwk = f"{s['routed_reference_clean']['qwk']:.4f}" if s["routed_reference_clean"] else "NA (smoke)"
        lines.append(f"| {s['attack']} | {s['route_off_clean']['qwk']:.4f} | {routed_qwk} | {s['route_off']['asr']:.4f} | {routed_asr} | {s['route_off']['avg_delta']:.4f} |")
    lines += ["", f"Route-off RHI macro ASR: {sum(s['route_off']['asr'] for s in summaries) / 3:.4f}"]
    (args.output_dir / "adaptive_route_off_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
