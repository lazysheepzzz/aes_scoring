#!/usr/bin/env python3
"""Shared launcher for paired AES stage-two training runs.

Both the clean-continuation control (C0) and HotFlip defense use this module so
their optimization settings cannot silently drift apart.  The only intended
training difference is ``use_hotflip_swaps``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


SERVER_ROOT = Path("/root/autodl-tmp")
SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_REPO_ROOT = SERVER_ROOT / "robust_text_scoring"
REPO_ROOT = (
    SERVER_REPO_ROOT
    if (SERVER_REPO_ROOT / "text_scoring_adv_training").is_dir()
    else SCRIPT_REPO_ROOT
)
ON_SERVER = SERVER_ROOT.exists()

CLEAN_CONTINUATION = "clean_continuation"
HOTFLIP_DEFENSE = "hotflip_defense"
TRAINING_MODES = (CLEAN_CONTINUATION, HOTFLIP_DEFENSE)


def _default_path(server_path: str, local_path: Path) -> Path:
    return Path(server_path) if ON_SERVER else local_path


def _python_prefix(conda_env: str | None, current_python: bool) -> list[str]:
    if current_python or not conda_env:
        return [sys.executable]
    if os.environ.get("CONDA_DEFAULT_ENV") == conda_env:
        return [sys.executable]

    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    server_conda = Path("/root/miniconda3/bin/conda")
    if not conda and server_conda.is_file():
        conda = str(server_conda)
    if conda:
        return [conda, "run", "--no-capture-output", "-n", conda_env, "python"]

    print(
        f"[WARN] Conda environment {conda_env!r} was requested but conda was "
        f"not found; using {sys.executable}.",
        file=sys.stderr,
    )
    return [sys.executable]


def _format_command(command: Sequence[str]) -> str:
    return (
        subprocess.list2cmdline(list(command))
        if os.name == "nt"
        else shlex.join(command)
    )


def _configure_environment(args: argparse.Namespace) -> None:
    if args.online:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
    else:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(args.hf_home)


def _default_output_dir(training_mode: str) -> Path:
    if training_mode == CLEAN_CONTINUATION:
        return _default_path(
            "/root/autodl-tmp/aes_clean_continuation",
            REPO_ROOT / "outputs" / "aes_clean_continuation",
        )
    return _default_path(
        "/root/autodl-tmp/aes_hotflip_defense",
        REPO_ROOT / "outputs" / "aes_hotflip_defense",
    )


def build_parser(training_mode: str) -> argparse.ArgumentParser:
    if training_mode not in TRAINING_MODES:
        raise ValueError(f"Unknown training mode: {training_mode}")
    display_name = (
        "C0 clean-continuation"
        if training_mode == CLEAN_CONTINUATION
        else "D_HOTFLIP adversarial"
    )
    parser = argparse.ArgumentParser(
        description=f"Run paired AES stage-two training: {display_name}."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=_default_path(
            "/root/autodl-tmp/victim/fold0_best",
            REPO_ROOT / "deberta_checkpoints" / "fold0_best",
        ),
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=_default_path(
            "/root/autodl-tmp/data/train_fold0.csv",
            REPO_ROOT / "data" / "train_fold0.csv",
        ),
    )
    parser.add_argument(
        "--valid-csv",
        type=Path,
        default=_default_path(
            "/root/autodl-tmp/data/valid_fold0.csv",
            REPO_ROOT / "data" / "valid_fold0.csv",
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(training_mode),
    )

    # Parameters shared by C0 and every stage-two defense.
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument(
        "--precision",
        choices=("bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--clean-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable interactive training and validation progress bars.",
    )

    # Kept in both configs; ignored by C0 because use_hotflip_swaps is false.
    parser.add_argument("--hotflip-weight", type=float, default=1.0)
    parser.add_argument(
        "--hotflip-tolerance",
        "--hotflip-margin",
        dest="hotflip_tolerance",
        type=float,
        default=0.05,
    )
    parser.add_argument("--hotflip-fraction", type=float, default=0.5)
    parser.add_argument("--hotflip-n-sample-pos", type=int, default=8)
    parser.add_argument("--hotflip-top-k-per-pos", type=int, default=2)
    parser.add_argument("--hotflip-max-candidates", type=int, default=16)

    parser.add_argument(
        "--hf-home",
        type=Path,
        default=_default_path(
            "/root/autodl-tmp/hf_cache",
            REPO_ROOT / ".cache" / "huggingface",
        ),
    )
    parser.add_argument(
        "--conda-env",
        default="aes" if ON_SERVER else None,
        help="Conda environment used to run the trainer. Server default: aes.",
    )
    parser.add_argument("--current-python", action="store_true")
    parser.add_argument("--online", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the paired-stage configuration without training.",
    )
    return parser


def build_config(
    args: argparse.Namespace,
    training_mode: str,
) -> dict[str, Any]:
    if training_mode not in TRAINING_MODES:
        raise ValueError(f"Unknown training mode: {training_mode}")
    return {
        "training_mode": training_mode,
        "checkpoint_path": str(args.checkpoint_path),
        "train_csv": str(args.train_csv),
        "valid_csv": str(args.valid_csv),
        "output_dir": str(args.output_dir),
        "num_epochs": args.num_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
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
        "show_progress": not args.no_progress,
        "use_hotflip_swaps": training_mode == HOTFLIP_DEFENSE,
        "hotflip_weight": args.hotflip_weight,
        "hotflip_n_sample_pos": args.hotflip_n_sample_pos,
        "hotflip_top_k_per_pos": args.hotflip_top_k_per_pos,
        "hotflip_max_candidates": args.hotflip_max_candidates,
        "hotflip_fraction": args.hotflip_fraction,
        "hotflip_tolerance": args.hotflip_tolerance,
    }


def validate_config(config: dict[str, Any]) -> None:
    positive_integer_fields = (
        "num_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_length",
        "eval_every",
        "save_every",
        "hotflip_n_sample_pos",
        "hotflip_top_k_per_pos",
        "hotflip_max_candidates",
    )
    for field in positive_integer_fields:
        if config[field] <= 0:
            raise ValueError(f"{field} must be greater than zero")
    if config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be greater than zero")
    if config["weight_decay"] < 0:
        raise ValueError("weight_decay must be non-negative")
    if not 0.0 <= config["warmup_ratio"] <= 1.0:
        raise ValueError("warmup_ratio must be between zero and one")
    if not 0.0 <= config["hotflip_fraction"] <= 1.0:
        raise ValueError("hotflip_fraction must be between zero and one")
    if config["clean_loss_weight"] <= 0:
        raise ValueError("clean_loss_weight must be greater than zero")
    if config["hotflip_weight"] < 0:
        raise ValueError("hotflip_weight must be non-negative")
    if config["hotflip_tolerance"] < 0:
        raise ValueError("hotflip_tolerance must be non-negative")
    if not 0.0 < config["adam_beta1"] < 1.0:
        raise ValueError("adam_beta1 must be in (0, 1)")
    if not 0.0 < config["adam_beta2"] < 1.0:
        raise ValueError("adam_beta2 must be in (0, 1)")
    if config["adam_epsilon"] <= 0:
        raise ValueError("adam_epsilon must be greater than zero")


def main(training_mode: str) -> int:
    args = build_parser(training_mode).parse_args()
    _configure_environment(args)
    config = build_config(args, training_mode)
    validate_config(config)

    config_path = args.output_dir / "launcher_config.json"
    trainer = (
        REPO_ROOT
        / "text_scoring_adv_training"
        / "training"
        / "aes_trainer.py"
    )
    command = [
        *_python_prefix(args.conda_env, args.current_python),
        str(trainer),
        "--config",
        str(config_path),
    ]

    print("[STAGE2] Resolved paired training configuration:")
    print(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"[RUN] {_format_command(command)}")
    if args.dry_run:
        return 0

    if not args.checkpoint_path.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {args.checkpoint_path}"
        )
    if not args.train_csv.is_file():
        raise FileNotFoundError(f"Training CSV not found: {args.train_csv}")
    if not args.valid_csv.is_file():
        raise FileNotFoundError(f"Validation CSV not found: {args.valid_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(command, env=os.environ.copy(), check=False)
    if completed.returncode == 0:
        print(
            f"[DONE] {training_mode} complete. "
            f"Best checkpoint: {args.output_dir / 'best'}"
        )
    return completed.returncode
