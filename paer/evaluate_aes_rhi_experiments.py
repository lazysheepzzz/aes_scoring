#!/usr/bin/env python3
"""Sequential RHI evaluation and missing specialist MLM tests, using frozen selections."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paer.prepare_aes_rhi_training_traces import load_rows
from paer.rhi_experiment_utils import bind_directory, checkpoint_identity, read_json, save_json, sha256, rhi_macro


def verify_paired_training(mixed_selection, paer_selection):
    roots = [Path(read_json(s)["checkpoint_path"]).parent for s in (mixed_selection, paer_selection)]
    inputs = [read_json(r / "rhi_training_inputs.json") for r in roots]
    for field in ("train_sha256", "valid_sha256", "trace_sha256", "checkpoint", "essay_counts_by_attack"):
        if inputs[0][field] != inputs[1][field]:
            raise ValueError(f"Mixed/PAER training provenance differs: {field}")
    configs = [read_json(r / "launcher_config.json") for r in roots]
    shared = ("num_epochs", "per_device_train_batch_size", "gradient_accumulation_steps",
              "learning_rate", "weight_decay", "warmup_ratio", "max_length", "seed", "precision",
              "eval_every", "save_every", "adam_beta1", "adam_beta2", "adam_epsilon",
              "clean_loss_weight", "adversarial_loss_weight", "inflation_tolerance",
              "relative_loss_power", "max_train_samples", "max_valid_samples", "training_attacks")
    for field in shared:
        if configs[0][field] != configs[1][field]:
            raise ValueError(f"Mixed/PAER common training setting differs: {field}")
    bindings = [read_json(s.parent / "rhi_run_binding.json") for s in (mixed_selection, paer_selection)]
    for selection, binding in zip((mixed_selection, paer_selection), bindings):
        chosen = read_json(selection)
        checkpoint = Path(chosen["checkpoint_path"])
        if checkpoint_identity(checkpoint) != binding["candidate_weights"].get(checkpoint.name):
            raise ValueError(f"Selected checkpoint changed since selection: {checkpoint}")
    ignored = {"defense_output_dir", "selection_output_dir", "subset_ids_path"}
    for field, value in bindings[0]["arguments"].items():
        if field not in ignored and bindings[1]["arguments"].get(field) != value:
            raise ValueError(f"Mixed/PAER selection setting differs: {field}")
    for field in ("valid_sha256", "bank_sha256", "c0_weights", "evaluation_source_hashes"):
        if bindings[0][field] != bindings[1][field]:
            raise ValueError(f"Mixed/PAER selection provenance differs: {field}")


def main(default_suite=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=("rhi", "specialists-mlm"), default=default_suite or "rhi")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data", type=Path, default=ROOT / "data/valid_fold0.csv")
    p.add_argument("--train-csv", type=Path, default=ROOT / "data/train_fold0.csv")
    p.add_argument("--development-csv", type=Path, default=ROOT / "data/valid_fold0.csv")
    p.add_argument("--evaluation-role", choices=("development", "independent-test"), default="development")
    p.add_argument("--test-provenance", help="For independent-test: explain where never-used labeled data came from")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--include-mlm", action="store_true", help="RHI: test MLM only after freezing method and selections")
    p.add_argument("--n-essays", type=int, help="Smoke limit only; default is ALL rows, not hard-coded 1154")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    p.add_argument("--device", default="cuda")
    p.add_argument("--online", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    for model in ("mixed", "paer", "hotflip", "rudimentary", "injection"):
        p.add_argument(f"--{model}-selection", type=Path, help="Path to the frozen best_checkpoint.json")
    args = p.parse_args()
    if args.batch_size <= 0 or (args.n_essays is not None and args.n_essays <= 0):
        raise ValueError("batch-size and n-essays must be positive")
    suffixes = {"mixed": "mixed_at_rhi", "paer": "paer_rhi_v3", "hotflip": "hotflip",
                "rudimentary": "rudimentary_defense_v2", "injection": "injection"}
    selections = {m: getattr(args, f"{m}_selection") or ROOT / "outputs" /
                  f"aes_{s}_checkpoint_selection_seed{args.seed}" / "best_checkpoint.json"
                  for m, s in suffixes.items()}
    models = ("mixed", "paer") if args.suite == "rhi" else ("hotflip", "rudimentary", "injection")
    checkpoints = {m: Path(read_json(selections[m])["checkpoint_path"]) for m in models}
    if args.suite == "rhi":
        verify_paired_training(selections["mixed"], selections["paer"])
    data = load_rows(args.data)
    if args.evaluation_role == "independent-test":
        if not args.test_provenance:
            raise ValueError("Independent test needs --test-provenance; development results remain development")
        texts = {r["text"] for r in data}
        for source in (args.train_csv, args.development_csv):
            if texts & {r["text"] for r in load_rows(source)}:
                raise ValueError(f"Test texts overlap {source}; cannot claim an independent test")
    attacks = ["rudimentary", "hotflip", "injection_family"] if args.suite == "rhi" else ["mlm_guided"]
    if args.include_mlm and args.suite == "rhi":
        attacks.append("mlm_guided")
    if args.output_dir is None:
        args.output_dir = ROOT / "outputs" / f"aes_{args.suite.replace('-', '_')}_evaluation_{args.evaluation_role}_seed{args.seed}"
    n = min(args.n_essays or len(data), len(data))
    protocol = {"protocol": "frozen_checkpoint_evaluation_v1", "suite": args.suite,
                "data": str(args.data.resolve()), "data_sha256": sha256(args.data), "n_essays": n,
                "evaluation_role": args.evaluation_role, "test_provenance": args.test_provenance,
                "smoke": n < len(data), "seed": args.seed, "batch_size": args.batch_size,
                "dtype": args.dtype, "device": args.device, "attacks": attacks,
                "success_threshold": 0.1, "steps": 30, "max_length": 1024,
                "selection_hashes": {m: sha256(selections[m]) for m in models},
                "checkpoints": {m: {"path": str(c.resolve()), "files": checkpoint_identity(c)}
                                for m, c in checkpoints.items()},
                "bank_sha256": sha256(ROOT / "injection/wikipedia_sentences_100.txt"),
                "code_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(
                    list((ROOT / "text_scoring_adv_training/evaluation/aes").rglob("*.py"))
                    + [ROOT / "paer/modeling_paer.py", ROOT / "paer/modeling_paer_v3.py",
                       ROOT / "whitebox/eval_hotflip_defended.py", Path(__file__)])}}
    if not args.dry_run:
        bind_directory(args.output_dir, protocol)
    evaluation_csv = args.data
    if n < len(data):
        evaluation_csv = args.output_dir / "smoke_evaluation_subset.csv"
        if not args.dry_run:
            import pandas as pd
            pd.read_csv(args.data).head(n).to_csv(evaluation_csv, index=False)
    results = []
    for model in models:
        for attack in attacks:
            out = args.output_dir / model / attack
            done = out / "completed_result_hashes.json"
            if not args.dry_run and done.exists():
                if any(sha256(out / name) != digest for name, digest in read_json(done).items()):
                    raise ValueError(f"Saved evaluation modified: {out}")
                print(f"[REUSE] {out}", flush=True)
            else:
                command = [sys.executable, str(ROOT / "whitebox/evaluate_aes_checkpoint.py"),
                           "--checkpoint", str(checkpoints[model]), "--attack", attack,
                           "--valid", str(evaluation_csv), "--out", str(out), "--current-python",
                           "--seed", str(args.seed), "--batch-size", str(args.batch_size),
                           "--dtype", args.dtype, "--device", args.device, "--n-essays", str(n),
                           "--n-steps", "30", "--success-threshold", "0.1", "--max-length", "1024",
                           "--injection-sentence-bank", str(ROOT / "injection/wikipedia_sentences_100.txt")]
                if args.online:
                    command.append("--online")
                if args.no_progress:
                    command.append("--no-progress")
                print(f"[RUN] {subprocess.list2cmdline(command)}", flush=True)
                if args.dry_run:
                    continue
                # One child at a time; model/MLM CUDA allocations die with the child.
                subprocess.run(command, check=True, env=os.environ.copy())
                required = ["clean_qwk.json", "asr_summary.json", "run_manifest.json"]
                if attack == "injection_family":
                    required += ["injection_family_summary.json", "injection_external_details.json", "injection_self_dup_details.json"]
                else:
                    required += [f"{attack}_details.json"]
                save_json(done, {name: sha256(out / name) for name in required})
            if not args.dry_run:
                metrics = read_json(out / ("injection_family_summary.json" if attack == "injection_family" else "asr_summary.json"))
                if isinstance(metrics, list):
                    metrics = metrics[0]
                expected_n = metrics.get("n_essays", metrics.get("n_essays_per_subattack"))
                if expected_n != n:
                    raise ValueError(f"Evaluation sample count differs from requested {n}: {out}")
                results.append({"model": model, "attack": attack,
                                "clean": read_json(out / "clean_qwk.json"), "metrics": metrics})
    if args.dry_run:
        return 0
    macro = {}
    for model in models:
        scores = {r["attack"]: r["metrics"]["asr"] for r in results if r["model"] == model}
        if all(a in scores for a in ("rudimentary", "hotflip", "injection_family")):
            macro[model] = (scores["rudimentary"] + scores["hotflip"] + scores["injection_family"]) / 3
    save_json(args.output_dir / "experiment_results.json", {"protocol": protocol, "results": results, "rhi_macro_asr": macro})
    lines = ["# Completed frozen-checkpoint evaluations", "", f"Evaluation role: {args.evaluation_role}; n={n}; seed={args.seed}", "",
             "| Model | Attack | Clean QWK | ASR | Mean delta |", "| --- | --- | --- | --- | --- |"]
    for row in results:
        lines.append(f"| {row['model']} | {row['attack']} | {row['clean']['qwk']:.4f} | {row['metrics']['asr']:.4f} | {row['metrics']['avg_delta']:.4f} |")
    for model, value in macro.items():
        lines.append(f"\n{model} RHI macro ASR: {value:.4f}")
    (args.output_dir / "experiment_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[SAVED] {args.output_dir / 'experiment_results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
