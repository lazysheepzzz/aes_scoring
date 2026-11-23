"""Frozen seed-42 RHI development evaluation: one search to +0.3, two ASRs."""
from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paer.rhi_experiment_utils import (
    RHI_ATTACKS, bind_directory, checkpoint_identity, read_json, rhi_macro, save_json, sha256,
)

MODELS = {"mixed_at_rhi": "gstep1000", "paer_rhi_v3": "gstep400"}
THRESHOLDS = (0.2, 0.3)
DATA = ROOT / "data/valid_fold0.csv"
BANK = ROOT / "injection/wikipedia_sentences_100.txt"
ENTRY = ROOT / "whitebox/evaluate_aes_checkpoint.py"
SHARED = ("num_epochs per_device_train_batch_size gradient_accumulation_steps learning_rate "
          "weight_decay warmup_ratio max_length seed precision eval_every save_every adam_beta1 "
          "adam_beta2 adam_epsilon clean_loss_weight adversarial_loss_weight inflation_tolerance "
          "relative_loss_power max_train_samples max_valid_samples training_attacks").split()


def frozen_inputs():
    ids, paths, pair_checks = {}, {}, []
    for model, step in MODELS.items():
        run = ROOT / "outputs" / f"aes_{model}_seed42"
        sel = run.parent / f"aes_{model}_checkpoint_selection_uniform30_seed42"
        choice = read_json(sel / "best_checkpoint.json")
        b = read_json(sel / "rhi_run_binding.json")
        cp = run / step
        if (choice["checkpoint_name"] != step or Path(choice["checkpoint_path"]).resolve() != cp.resolve()
                or choice["mlm_used_for_selection"] is not False or not choice["eligible"]):
            raise ValueError(f"Unexpected frozen selection for {model}")
        weights = checkpoint_identity(cp)
        if (weights != b["candidate_weights"][step] or b["valid_sha256"] != sha256(DATA)
                or b["bank_sha256"] != sha256(BANK)):
            raise ValueError(f"Changed checkpoint, development CSV or sentence bank: {model}")
        args = b["arguments"]
        if any(args[k] != 30 for k in ("selection_steps", "rudimentary_selection_steps", "injection_selection_steps")):
            raise ValueError("Expected uniform-30 selection")
        train = read_json(run / "rhi_training_inputs.json")
        if train != b["training_inputs"]:
            raise ValueError(f"Training inputs changed since selection: {model}")
        cfg = read_json(run / "launcher_config.json")
        ignore = {"training_inputs", "defense_output_dir", "selection_output_dir", "subset_ids_path"}
        pair_checks.append((
            {k: v for k, v in args.items() if k not in ignore},
            [train[k] for k in ("checkpoint", "train_sha256", "trace_sha256", "valid_sha256", "essay_counts_by_attack")],
            {k: cfg[k] for k in SHARED if k in cfg}, read_json(sel / "subset_ids.json")["essay_ids"], b["c0_weights"],
        ))
        paths[model] = cp
        ids[model] = {"checkpoint": str(cp), "files": weights, "selection": sha256(sel / "best_checkpoint.json"),
                      "binding": sha256(sel / "rhi_run_binding.json"), "training_config": sha256(run / "launcher_config.json")}
    if pair_checks[0] != pair_checks[1]:
        raise ValueError("Mixed/PAER training or selection protocols do not match")
    return paths, ids


def counts(payload, expected_ids, attack):
    rows, summary = payload["details"], payload["summary"]
    if (summary["attack"] != attack or summary["success_threshold"] != 0.3
            or [str(r["essay_id"]) for r in rows] != expected_ids):
        raise ValueError(f"{attack}: wrong threshold or unpaired essays")
    deltas = []
    for r in rows:
        d = float(r["delta"])
        if (not r["perturbed_text"] or not math.isfinite(d)
                or not math.isclose(d, float(r["perturbed_score"]) - float(r["original_score"]), abs_tol=1e-9)
                or bool(r["success"]) != (d >= 0.3)):
            raise ValueError(f"{attack}/{r['essay_id']}: missing output or inconsistent raw score")
        deltas.append(d)
    result = [sum(d >= t for d in deltas) for t in THRESHOLDS]
    if round(result[1] / len(rows), 4) != summary["asr"]:
        raise ValueError(f"{attack}: ASR@0.3 disagrees with evaluator")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true", help="First two essays, separate output directory")
    p.add_argument("--dry-run", action="store_true", help="Check inputs only; no writes or GPU")
    args = p.parse_args()
    with DATA.open(encoding="utf-8-sig", newline="") as f:
        expected = [row["essay_id"] for row in csv.DictReader(f)]
    if len(expected) != 1154 or len(set(expected)) != len(expected):
        raise ValueError("Unexpected development essay IDs")
    if args.smoke:
        expected = expected[:2]
    n = len(expected)
    paths, ids = frozen_inputs()
    out = ROOT / "outputs" / (("smoke_" if args.smoke else "") + "aes_rhi_thresholds_020_030_seed42")
    code = [Path(__file__), ENTRY, ROOT / "whitebox/eval_hotflip_defended.py",
            ROOT / "paer/modeling_paer.py", ROOT / "paer/modeling_paer_v3.py",
            ROOT / "paer/rhi_experiment_utils.py",
            *sorted((ROOT / "text_scoring_adv_training/evaluation/aes").rglob("*.py"))]
    protocol = {"role": "development", "smoke": args.smoke, "n": n, "seed": 42, "stop": 0.3,
                "thresholds": list(THRESHOLDS), "batch_size": 4, "dtype": "float32",
                "steps": 30, "candidates": 16, "beam": 1, "max_length": 1024,
                "sample_positions": 8, "top_k_per_position": 2, "token_edit_rate_rh": 0.1,
                "valid_sha256": sha256(DATA), "bank_sha256": sha256(BANK), "models": ids,
                "code": {str(f.relative_to(ROOT)): sha256(f) for f in code}}
    print(f"[CHECKED] frozen Mixed/PAER; n={n}; thresholds={THRESHOLDS}; out={out}", flush=True)
    if args.dry_run:
        return
    bind_directory(out, protocol)
    results = {}
    for model, cp in paths.items():
        results[model] = raw = {}
        for attack in RHI_ATTACKS:
            parent = out / model / attack
            done = parent / "completed.json"
            name = f"{attack}_details.json"
            if done.exists():
                record = read_json(done)
                attempt = parent / record["attempt"]
                if any(sha256(attempt / f) != digest for f, digest in record["files"].items()):
                    raise ValueError(f"Changed completed evaluation: {parent}")
                print(f"[REUSE] {parent}", flush=True)
            else:
                parent.mkdir(parents=True, exist_ok=True)
                attempt = Path(tempfile.mkdtemp(prefix="attempt_", dir=parent))
                command = [sys.executable, str(ENTRY), "--checkpoint", str(cp), "--attack", attack,
                           "--valid", str(DATA), "--out", str(attempt), "--current-python", "--skip-clean",
                           "--n-essays", str(n), "--seed", "42", "--device", "cuda", "--dtype", "float32",
                           "--batch-size", "4", "--max-length", "1024", "--n-steps", "30",
                           "--beam-size", "1", "--max-candidates-per-step", "16", "--n-sample-pos", "8",
                           "--top-k-per-pos", "2", "--max-token-edit-rate", "0.1", "--success-threshold", "0.3",
                           "--injection-sentence-bank", str(BANK)]
                print(f"[RUN] {model} {attack}", flush=True)
                subprocess.run(command, check=True)
            manifest = read_json(attempt / "run_manifest.json")
            if (manifest["attack"] != attack or manifest["n_essays"] != n or manifest["seed"] != 42
                    or manifest["success_threshold"] != 0.3 or Path(manifest["victim"]).resolve() != cp.resolve()
                    or Path(manifest["data"]).resolve() != DATA.resolve()):
                raise ValueError(f"Child protocol mismatch: {attempt}")
            raw[attack] = counts(read_json(attempt / name), expected, attack)
            if not done.exists():
                save_json(done, {"attempt": attempt.name, "files": {f: sha256(attempt / f)
                          for f in (name, "run_manifest.json", "asr_summary.json")}})
    lines = [f"# Development ASR | seed 42 | n={n} | smoke={args.smoke}", "",
             "One adaptive search per essay/model/subattack; stop at +0.3. ASR is in percent.",
             "", "| Model | Threshold | R | H | I-external | I-self | I-family | RHI macro |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model, raw in results.items():
        for j, t in enumerate(THRESHOLDS):
            r, h, e, s = (100 * raw[k][j] / n for k in RHI_ATTACKS)
            lines.append(f"| {model} | {t:.1f} | {r:.2f} | {h:.2f} | {e:.2f} | {s:.2f} | "
                         f"{(e+s)/2:.2f} | {rhi_macro(r,h,e,s):.2f} |")
    save_json(out / "threshold_counts.json", {"n": n, "thresholds": list(THRESHOLDS), "counts": results})
    report = out / "threshold_results.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), f"\n[SAVED] {report}")


if __name__ == "__main__":
    main()
