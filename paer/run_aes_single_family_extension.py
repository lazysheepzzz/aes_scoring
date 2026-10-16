#!/usr/bin/env python3
"""Matched single-family AT/PAER study reusing frozen RHI training traces.

No attack generation or Judge. One child runs at a time, using the existing
sequential-backward trainer. Historical models and output directories stay intact.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paer.prepare_aes_rhi_training_traces import load_rows
from paer.rhi_experiment_utils import (
    bind_directory, checkpoint_identity, read_json, save_json, sha256,
    validate_trace_groups,
)

FAMILIES = {
    "rudimentary": ("rudimentary",),
    "hotflip": ("hotflip",),
    "injection": ("injection_external", "injection_self_dup"),
}
MODES = {"at": "mixed_at_rh", "paer": "paer_rh_v3"}
EVALUATION_ATTACKS = ("rudimentary", "hotflip", "injection_family")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-training-dir", type=Path,
                        default=ROOT / "outputs/aes_paer_rhi_v3_seed42")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42, help="Training/smoke sampling seed")
    parser.add_argument("--attack-seed", type=int, default=42)
    parser.add_argument("--families", nargs="+", choices=tuple(FAMILIES), default=list(FAMILIES))
    parser.add_argument("--models", nargs="+", choices=tuple(MODES), default=list(MODES))
    parser.add_argument("--stage", choices=("prepare", "train", "select", "evaluate", "all"), default="train")
    parser.add_argument("--include-mlm", action="store_true", help="Final evaluation only; never selection")
    parser.add_argument("--c0-checkpoint", type=Path,
                        default=ROOT / "outputs/aes_clean_continuation_seed42/best")
    parser.add_argument("--sentence-bank", type=Path, default=ROOT / "injection/wikipedia_sentences_100.txt")
    parser.add_argument("--subset-size", type=int, default=256)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--qwk-tolerance", type=float, default=0.02)
    parser.add_argument("--max-checkpoint-step", type=int, default=1400)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-train-samples", type=int, default=160)
    parser.add_argument("--smoke-valid-samples", type=int, default=32)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def immutable_text(path, text):
    """Idempotently create an artifact; never replace different saved data."""
    path = Path(path)
    if path.exists():
        if path.read_bytes() != text.encode("utf-8"):
            raise ValueError(f"Saved input differs; use a new output-dir: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def immutable_json(path, payload):
    immutable_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def smoke_rows(groups, n_rows, count, seed):
    """Include each family's traces instead of an unreliable first-N slice."""
    rng = random.Random(seed)
    pools = {attack: sorted(i for i, g in groups.items() if g[0]["attack"] == attack)
             for attacks in FAMILIES.values() for attack in attacks}
    per_family = min(16, count // 3, len(pools["rudimentary"]), len(pools["hotflip"]),
                     2 * len(pools["injection_external"]), 2 * len(pools["injection_self_dup"]))
    per_family -= per_family % 2
    if count > n_rows or per_family < 2:
        raise ValueError("Smoke needs at least six source rows with balanced R/H/I traces")
    chosen = set()
    for attack, pool in pools.items():
        size = per_family // 2 if attack.startswith("injection_") else per_family
        chosen.update(rng.sample(pool, size))
    selected = set(chosen)
    selected.update(rng.sample(sorted(set(range(n_rows)) - selected), count - len(selected)))
    return sorted(selected), chosen


def family_config(reference, *, family, model, seed, train, valid, traces, output, smoke):
    config = dict(reference)
    config.update(training_mode=MODES[model], experiment_name=f"single_{family}_{model}",
                  training_attacks=list(FAMILIES[family]), seed=seed,
                  train_csv=str(train), valid_csv=str(valid), trace_jsonl=str(traces),
                  output_dir=str(output), max_trace_records=None,
                  max_train_samples=None, max_valid_samples=None)
    if smoke:
        config.update(num_epochs=1, eval_every=1, save_every=1)
    return config


def prepare(args):
    reference_dir = args.reference_training_dir.resolve()
    reference = read_json(reference_dir / "launcher_config.json")
    inputs = read_json(reference_dir / "rhi_training_inputs.json")
    if reference["training_mode"] != "paer_rh_v3" or set(reference["training_attacks"]) != set(sum(FAMILIES.values(), ())):
        raise ValueError("Reference must be the full, unablated PAER-RHI-v3 run")
    if any(reference.get(key) is not None for key in ("max_train_samples", "max_valid_samples", "max_trace_records")):
        raise ValueError("Reference must be a full run, not smoke/truncated training")
    for key in ("correction_scale", "localization_loss_weight", "attention_alignment_loss_weight",
                "correction_calibration_weight", "route_lift_loss_weight", "clean_correction_weight"):
        if reference[key] <= 0:
            raise ValueError(f"Reference is already ablated: {key}")
    paths = {key: Path(reference[key]).resolve() for key in
             ("train_csv", "valid_csv", "trace_jsonl", "checkpoint_path")}
    for field, key in (("train_sha256", "train_csv"), ("valid_sha256", "valid_csv"), ("trace_sha256", "trace_jsonl")):
        if inputs[field] != sha256(paths[key]):
            raise ValueError(f"Reference source changed: {key}")
    if inputs["checkpoint"] != checkpoint_identity(paths["checkpoint_path"]):
        raise ValueError("B0 weights differ from the reference initialization")
    manifest_path = paths["trace_jsonl"].with_suffix(".manifest.json")
    manifest = read_json(manifest_path)
    if manifest["trace_sha256"] != inputs["trace_sha256"] or not manifest.get("mlm_excluded"):
        raise ValueError("Invalid source trace manifest or MLM exposure")
    if reference["max_length"] != manifest["arguments"]["max_length"]:
        raise ValueError("Trace/training max_length differs")
    if sha256(args.sentence_bank) != manifest["source_hashes"]["bank"]:
        raise ValueError("Sentence bank differs from the frozen RHI protocol")
    rows = load_rows(paths["train_csv"])
    valid_rows = load_rows(paths["valid_csv"])
    if {r["text"] for r in rows} & {r["text"] for r in valid_rows}:
        raise ValueError("Training and development texts overlap")
    if args.smoke and args.smoke_valid_samples > len(valid_rows):
        raise ValueError("Smoke validation limit exceeds source CSV")
    with paths["trace_jsonl"].open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    groups, source_counts = validate_trace_groups(records, rows)
    if inputs["essay_counts_by_attack"] != source_counts:
        raise ValueError("Recorded source essay exposure differs from the trace pool")
    if args.output_dir is None:
        prefix = "smoke_" if args.smoke else ""
        args.output_dir = ROOT / "outputs" / f"{prefix}aes_single_family_extension_seed{args.seed}"
    args.output_dir = args.output_dir.resolve()
    for protected in (reference_dir, paths["checkpoint_path"], paths["trace_jsonl"].parent,
                      paths["train_csv"].parent, paths["valid_csv"].parent):
        if args.output_dir == protected or args.output_dir.is_relative_to(protected):
            raise ValueError(f"Output must be outside historical inputs: {protected}")
    code_paths = set((ROOT / "text_scoring_adv_training/evaluation/aes").rglob("*.py")) | {
        Path(__file__).resolve(), ROOT / "paer/aes_rh_trainer.py",
        ROOT / "paer/rhi_experiment_utils.py", ROOT / "paer/select_aes_rhi_checkpoint.py",
        ROOT / "paer/modeling_paer.py", ROOT / "paer/modeling_paer_v3.py",
        ROOT / "whitebox/eval_hotflip_defended.py", ROOT / "whitebox/evaluate_aes_checkpoint.py",
    }
    protocol = {
        "protocol": "aes_single_family_matched_at_paer_v1", "smoke": args.smoke,
        "seed": args.seed, "attack_seed": args.attack_seed,
        "reference_config": reference, "reference_inputs": inputs,
        "reference_manifest_sha256": sha256(manifest_path),
        "sentence_bank_sha256": sha256(args.sentence_bank),
        "smoke_samples": [args.smoke_train_samples, args.smoke_valid_samples] if args.smoke else None,
        "selection": {"attacks": list(EVALUATION_ATTACKS), "steps": 2 if args.smoke else 30,
                      "subset_size": min(8, args.smoke_valid_samples) if args.smoke else args.subset_size,
                      "subset_seed": args.subset_seed, "qwk_tolerance": args.qwk_tolerance,
                      "max_checkpoint_step": 1 if args.smoke else args.max_checkpoint_step,
                      "c0_checkpoint": str(args.c0_checkpoint.resolve()),
                      "c0_weights": checkpoint_identity(args.c0_checkpoint)},
        "evaluation": {"batch_size": args.batch_size, "dtype": "float32", "device": args.device,
                       "role": "development", "threshold": 0.1},
        "mlm_used_for_training_or_selection": False,
        "legacy_D_models_are_matched_controls": False,
        "code_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(code_paths)},
    }
    if args.dry_run:
        print(json.dumps({"output_dir": str(args.output_dir), "protocol": protocol,
                          "families": args.families, "models": args.models,
                          "stage": args.stage, "include_mlm": args.include_mlm}, indent=2))
        return None
    bind_directory(args.output_dir, protocol)
    train_path, valid_path = paths["train_csv"], paths["valid_csv"]
    if args.smoke:
        import pandas as pd
        indices, chosen = smoke_rows(groups, len(rows), args.smoke_train_samples, args.seed)
        remap = {old: new for new, old in enumerate(indices)}
        train_path, valid_path = args.output_dir / "data/train.csv", args.output_dir / "data/valid.csv"
        immutable_text(train_path, pd.read_csv(paths["train_csv"]).iloc[indices].to_csv(index=False))
        immutable_text(valid_path, pd.read_csv(paths["valid_csv"]).head(args.smoke_valid_samples).to_csv(index=False))
        records = [{**record, "source_row_index": int(record["row_index"]),
                    "row_index": remap[int(record["row_index"])]}
                   for record in records if int(record["row_index"]) in chosen]
        immutable_json(args.output_dir / "data/source_row_indices.json", indices)
    plans = {}
    for family, attacks in FAMILIES.items():
        filtered = [record for record in records if record["attack"] in attacks]
        trace_path = args.output_dir / "data" / f"{family}_training_traces.jsonl"
        immutable_text(trace_path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in filtered))
        counts = {attack: len({int(r["row_index"]) for r in filtered if r["attack"] == attack})
                  for attack in attacks}
        n_train = len(load_rows(train_path))
        provenance = {
            "experiment_name": f"single_family_{family}",
            "train_sha256": sha256(train_path), "valid_sha256": sha256(valid_path),
            "trace_sha256": sha256(trace_path), "checkpoint": inputs["checkpoint"],
            "essay_counts_by_attack": counts, "trace_records": len(filtered),
            "n_clean_essays": n_train, "n_attacked_essays": sum(counts.values()),
            "actual_attack_fraction": sum(counts.values()) / n_train,
            "training_attacks": list(attacks), "mlm_used_for_training": False,
            "source_trace_sha256": inputs["trace_sha256"], "smoke": args.smoke,
            "evaluation_status": "development only; other RHI families are used for selection",
        }
        immutable_json(trace_path.with_suffix(".manifest.json"), provenance)
        for model in MODES:
            directory = args.output_dir / family / model
            training_dir = directory / "training"
            config = family_config(reference, family=family, model=model, seed=args.seed,
                                   train=train_path, valid=valid_path, traces=trace_path,
                                   output=training_dir, smoke=args.smoke)
            config["show_progress"] = not args.no_progress
            immutable_json(training_dir / "launcher_config.json", config)
            immutable_json(training_dir / "rhi_training_inputs.json", provenance)
            plans[(family, model)] = {"directory": directory, "config": config,
                                      "inputs": provenance, "training_dir": training_dir}
        print(f"[POOL:{family}] clean={n_train}, attacked={sum(counts.values())}, traces={len(filtered)}, counts={counts}", flush=True)
    return plans, protocol, valid_path


def run_child(command, args):
    print(f"[RUN] {subprocess.list2cmdline([str(v) for v in command])}", flush=True)
    env = os.environ.copy()
    if not args.online:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    else:
        env.pop("HF_HUB_OFFLINE", None)
        env.pop("TRANSFORMERS_OFFLINE", None)
    subprocess.run([str(v) for v in command], check=True, env=env)


def train_model(plan, args):
    directory = plan["training_dir"]
    marker = directory / "training_complete.json"
    final = directory / "final"
    metrics = directory / "final_clean_metrics.json"
    config_path = directory / "launcher_config.json"
    if marker.exists():
        saved = read_json(marker)
        if (saved["config_sha256"] != sha256(config_path)
                or saved["metrics_sha256"] != sha256(metrics)
                or saved["final_weights"] != checkpoint_identity(final)):
            raise ValueError(f"Completed training changed: {directory}")
        print(f"[REUSE TRAINING] {directory}", flush=True)
        return
    # Recover a completed child if the parent stopped before writing its marker.
    completed = final.is_dir() and metrics.is_file()
    if completed:
        actual = read_json(directory / "training_config.json")
        if any(actual.get(k) != v for k, v in plan["config"].items()):
            raise ValueError(f"Completed training config differs: {directory}")
    elif (directory / "training_diagnostics.jsonl").exists() or any(directory.glob("gstep*")):
        raise RuntimeError(f"Partial training preserved (no optimizer resume): {directory}. "
                           "Use a new output-dir and --families/--models for this job.")
    else:
        run_child([sys.executable, ROOT / "paer/aes_rh_trainer.py", "--config", config_path], args)
    metric = read_json(metrics)
    if not math.isfinite(float(metric["qwk"])):
        raise ValueError(f"Non-finite final clean QWK: {metrics}")
    save_json(marker, {"config_sha256": sha256(config_path), "metrics_sha256": sha256(metrics),
                       "final_weights": checkpoint_identity(final)})


def select_model(plan, protocol, valid, args):
    if not (plan["training_dir"] / "training_complete.json").is_file():
        raise FileNotFoundError(f"Complete training first: {plan['training_dir']}")
    selection = protocol["selection"]
    command = [sys.executable, ROOT / "paer/select_aes_rhi_checkpoint.py",
               "--defense-output-dir", plan["training_dir"],
               "--training-inputs", plan["training_dir"] / "rhi_training_inputs.json",
               "--selection-output-dir", plan["directory"] / "selection",
               "--valid-csv", valid, "--c0-checkpoint", args.c0_checkpoint,
               "--injection-sentence-bank", args.sentence_bank,
               "--subset-size", selection["subset_size"], "--subset-seed", args.subset_seed,
               "--seed", args.attack_seed, "--qwk-tolerance", args.qwk_tolerance,
               "--max-checkpoint-step", selection["max_checkpoint_step"],
               "--batch-size", args.batch_size, "--dtype", "float32", "--device", args.device,
               "--selection-steps", selection["steps"],
               "--rudimentary-selection-steps", selection["steps"],
               "--injection-selection-steps", selection["steps"]]
    if args.online:
        command.append("--online")
    if args.no_progress:
        command.append("--no-progress")
    run_child(command, args)
    subset = plan["directory"] / "selection" / f"selection_subset_{selection['subset_size']}.csv"
    expected = {"sha256": sha256(subset), "valid_sha256": plan["inputs"]["valid_sha256"]}
    immutable_json(args.output_dir / "selection_subset_hash.json", expected)


def result_files(attack):
    names = ["clean_qwk.json", "asr_summary.json", "run_manifest.json"]
    if attack == "injection_family":
        names += ["injection_family_summary.json", "injection_external_details.json", "injection_self_dup_details.json"]
    else:
        names += [f"{attack}_details.json"]
    return names


def evaluate_model(plan, protocol, valid, args):
    selection_path = plan["directory"] / "selection/best_checkpoint.json"
    selected = read_json(selection_path)
    checkpoint = Path(selected["checkpoint_path"])
    selected_weights = checkpoint_identity(checkpoint)
    selection_binding = read_json(selection_path.parent / "rhi_run_binding.json")
    if selected_weights != selection_binding["candidate_weights"].get(checkpoint.name):
        raise ValueError(f"Selected weights changed: {checkpoint}")
    n = len(load_rows(valid))
    attacks = list(EVALUATION_ATTACKS) + (["mlm_guided"] if args.include_mlm else [])
    for attack in attacks:
        out = plan["directory"] / "evaluation" / attack
        binding = {"protocol": "single_family_frozen_evaluation_v1", "smoke": args.smoke,
                   "checkpoint": selected_weights, "selection_sha256": sha256(selection_path),
                   "data_sha256": sha256(valid), "n": n, "attack": attack,
                   "attack_seed": args.attack_seed, "steps": 2 if args.smoke else 30,
                   "evaluation": protocol["evaluation"], "code_hashes": protocol["code_hashes"],
                   "sentence_bank_sha256": protocol["sentence_bank_sha256"]}
        bind_directory(out, binding)
        marker = out / "completed_result_hashes.json"
        if marker.exists():
            if any(sha256(out / name) != digest for name, digest in read_json(marker).items()):
                raise ValueError(f"Saved evaluation changed: {out}")
            print(f"[REUSE EVALUATION] {out}", flush=True)
            continue
        command = [sys.executable, ROOT / "whitebox/evaluate_aes_checkpoint.py",
                   "--checkpoint", checkpoint, "--attack", attack,
                   "--valid", valid, "--out", out, "--current-python",
                   "--n-essays", n, "--n-steps", binding["steps"],
                   "--seed", args.attack_seed, "--batch-size", args.batch_size,
                   "--dtype", "float32", "--device", args.device,
                   "--success-threshold", "0.1", "--max-length", plan["config"]["max_length"],
                   "--injection-sentence-bank", args.sentence_bank]
        if args.online:
            command.append("--online")
        if args.no_progress:
            command.append("--no-progress")
        run_child(command, args)
        summary = read_json(out / ("injection_family_summary.json" if attack == "injection_family" else "asr_summary.json"))
        if isinstance(summary, list):
            summary = summary[0]
        clean, manifest = read_json(out / "clean_qwk.json"), read_json(out / "run_manifest.json")
        if (summary.get("n_essays", summary.get("n_essays_per_subattack")) != n
                or clean["n"] != n or manifest["n_essays"] != n):
            raise ValueError(f"Unexpected evaluation sample count: {out}")
        if (Path(manifest["victim"]).resolve() != checkpoint.resolve()
                or manifest["seed"] != args.attack_seed
                or manifest["attack_parameters"]["n_steps"] != binding["steps"]):
            raise ValueError(f"Unexpected victim/seed/budget: {out}")
        save_json(marker, {name: sha256(out / name) for name in result_files(attack)})


def summarize(plans, args):
    rows = []
    for (family, model), plan in plans.items():
        metrics = {}
        for attack in (*EVALUATION_ATTACKS, "mlm_guided"):
            out = plan["directory"] / "evaluation" / attack
            marker = out / "completed_result_hashes.json"
            if not marker.exists():
                continue
            if any(sha256(out / name) != digest for name, digest in read_json(marker).items()):
                raise ValueError(f"Completed summary source changed: {out}")
            metric = read_json(out / ("injection_family_summary.json" if attack == "injection_family" else "asr_summary.json"))
            if isinstance(metric, list):
                metric = metric[0]
            metrics[attack] = {"clean": read_json(out / "clean_qwk.json"), "metrics": metric}
        if metrics:
            chosen = read_json(plan["directory"] / "selection/best_checkpoint.json")
            row = {"training_family": family, "model": model, "checkpoint": chosen,
                   "training_exposure": plan["inputs"], "attacks": metrics}
            if all(a in metrics for a in EVALUATION_ATTACKS):
                row["rhi_macro_asr"] = sum(metrics[a]["metrics"]["asr"] for a in EVALUATION_ATTACKS) / 3
            rows.append(row)
    save_json(args.output_dir / "single_family_results.json", {"smoke": args.smoke, "evaluation_role": "development", "results": rows})
    lines = ["# Single-family AT / PAER extension", "",
             f"Training seed: {args.seed}; attack seed: {args.attack_seed}; smoke: {args.smoke}.", "",
             "Selection uses equal-weight R/H/I, all with the same step ceiling. MLM is evaluation-only.",
             "Other RHI families are absent from single-family training but participate in selection.", "",
             "| Training family | Model | Checkpoint | Clean QWK | R ASR | H ASR | I ASR | RHI macro | MLM ASR |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        metrics = row["attacks"]
        clean_values = {float(v["clean"]["qwk"]) for v in metrics.values()}
        if len(clean_values) != 1:
            raise ValueError(f"Clean metrics differ across attacks: {row['training_family']}/{row['model']}")
        clean = next(iter(clean_values))
        values = [f"{metrics[a]['metrics']['asr']:.4f}" if a in metrics else "NA"
                  for a in (*EVALUATION_ATTACKS, "mlm_guided")]
        macro = f"{row['rhi_macro_asr']:.4f}" if "rhi_macro_asr" in row else "NA"
        lines.append(f"| {row['training_family']} | {row['model']} | {row['checkpoint']['checkpoint_name']} | "
                     f"{clean:.4f} | {values[0]} | {values[1]} | {values[2]} | {macro} | {values[3]} |")
    lines += ["", "The two Injection subtypes have one family weight. Historical D models are not the matched controls.",
              "Single-family pools have fewer attacked essays than the full RHI pool; do not attribute all mixed/single differences to attack diversity."]
    (args.output_dir / "single_family_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = build_parser().parse_args()
    if min(args.batch_size, args.subset_size, args.smoke_train_samples, args.smoke_valid_samples) <= 0:
        raise ValueError("Batch and sample sizes must be positive")
    if args.max_checkpoint_step <= 0 or args.qwk_tolerance < 0:
        raise ValueError("Use a positive common checkpoint cap and nonnegative QWK tolerance")
    prepared = prepare(args)
    if prepared is None:
        return 0
    plans, protocol, valid = prepared
    if args.stage == "prepare":
        return 0
    for family in args.families:
        for model in args.models:
            plan = plans[(family, model)]
            if args.stage in ("train", "all"):
                train_model(plan, args)
            if args.stage in ("select", "all"):
                select_model(plan, protocol, valid, args)
            if args.stage in ("evaluate", "all"):
                evaluate_model(plan, protocol, valid, args)
                summarize(plans, args)
    print(f"[DONE] {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
