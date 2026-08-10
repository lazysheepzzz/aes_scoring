#!/usr/bin/env python3
"""Readable launchers for the paired Mixed-AT-RH and PAER-RH runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MIXED_AT_RH = "mixed_at_rh"
PAER_RH = "paer_rh"
PAER_RH_V2 = "paer_rh_v2"


def build_parser(training_mode: str) -> argparse.ArgumentParser:
    if training_mode not in (MIXED_AT_RH, PAER_RH, PAER_RH_V2):
        raise ValueError(training_mode)
    parser = argparse.ArgumentParser(
        description=f"Train {training_mode} on shared offline RH traces."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=REPO_ROOT / "deberta_checkpoints" / "fold0_best",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=REPO_ROOT / "data" / "train_fold0.csv",
    )
    parser.add_argument(
        "--trace-jsonl",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "paer"
            / "rh_counterfactual_training_traces_seed42.jsonl"
        ),
    )
    parser.add_argument(
        "--valid-csv",
        type=Path,
        default=REPO_ROOT / "data" / "valid_fold0.csv",
    )
    default_names = {
        MIXED_AT_RH: "aes_mixed_at_rh_seed42",
        PAER_RH: "aes_paer_rh_seed42",
        PAER_RH_V2: "aes_paer_rh_v2_seed42",
    }
    default_name = default_names[training_mode]
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / default_name,
    )
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument(
        "--precision", choices=("bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--clean-loss-weight", type=float, default=1.0)
    parser.add_argument("--adversarial-loss-weight", type=float, default=1.0)
    parser.add_argument("--inflation-tolerance", type=float, default=0.0)
    parser.add_argument("--relative-loss-power", type=float, default=1.0)
    parser.add_argument("--localization-loss-weight", type=float, default=0.25)
    parser.add_argument("--clean-false-suppression-weight", type=float, default=0.10)
    parser.add_argument("--localization-positive-weight", type=float, default=8.0)
    parser.add_argument("--attribution-gain-scale", type=float, default=0.1)
    parser.add_argument("--correction-scale", type=float, default=1.0)
    parser.add_argument("--paer-head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--correction-calibration-weight", type=float, default=1.0)
    parser.add_argument("--clean-correction-weight", type=float, default=1.0)
    parser.add_argument("--correction-loss-beta", type=float, default=0.02)
    parser.add_argument("--max-correction-target", type=float, default=1.0)
    parser.add_argument("--routing-top-k", type=int, default=8)
    parser.add_argument("--routing-risk-bias-init", type=float, default=-5.0)
    parser.add_argument("--max-trace-records", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_config(args: argparse.Namespace, training_mode: str) -> dict:
    return {
        "training_mode": training_mode,
        "checkpoint_path": str(args.checkpoint_path),
        "train_csv": str(args.train_csv),
        "trace_jsonl": str(args.trace_jsonl),
        "valid_csv": str(args.valid_csv),
        "output_dir": str(args.output_dir),
        "num_epochs": args.num_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_length": args.max_length,
        "seed": args.seed,
        "eval_every": args.eval_every,
        "save_every": args.save_every,
        "precision": args.precision,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "adam_epsilon": args.adam_epsilon,
        "clean_loss_weight": args.clean_loss_weight,
        "adversarial_loss_weight": args.adversarial_loss_weight,
        "inflation_tolerance": args.inflation_tolerance,
        "relative_loss_power": args.relative_loss_power,
        "localization_loss_weight": args.localization_loss_weight,
        "clean_false_suppression_weight": args.clean_false_suppression_weight,
        "localization_positive_weight": args.localization_positive_weight,
        "attribution_gain_scale": args.attribution_gain_scale,
        "correction_scale": args.correction_scale,
        "paer_head_learning_rate": args.paer_head_learning_rate,
        "correction_calibration_weight": args.correction_calibration_weight,
        "clean_correction_weight": args.clean_correction_weight,
        "correction_loss_beta": args.correction_loss_beta,
        "max_correction_target": args.max_correction_target,
        "routing_top_k": args.routing_top_k,
        "routing_risk_bias_init": args.routing_risk_bias_init,
        "show_progress": not args.no_progress,
        "max_trace_records": args.max_trace_records,
        "max_train_samples": args.max_train_samples,
    }


def main(training_mode: str) -> int:
    args = build_parser(training_mode).parse_args()
    config = build_config(args, training_mode)
    config_path = args.output_dir / "launcher_config.json"
    trainer_path = REPO_ROOT / "paer" / "aes_rh_trainer.py"
    command = [sys.executable, str(trainer_path), "--config", str(config_path)]
    print("[PAIRED RH] Resolved training configuration:")
    print(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"[RUN] {subprocess.list2cmdline(command)}")
    if args.dry_run:
        return 0
    for required_path, kind in (
        (args.checkpoint_path, "dir"),
        (args.train_csv, "file"),
        (args.trace_jsonl, "file"),
        (args.valid_csv, "file"),
    ):
        exists = required_path.is_dir() if kind == "dir" else required_path.is_file()
        if not exists:
            raise FileNotFoundError(required_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    if not args.online:
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
    completed = subprocess.run(command, env=environment, check=False)
    return completed.returncode
