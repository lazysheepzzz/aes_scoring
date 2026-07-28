#!/usr/bin/env python3
"""Launch the v4 HotFlip adversarial training run.

The current trainer accepts a JSON configuration through ``--config``. This
launcher records the former shell parameters in that supported format and then
starts the trainer with the selected Python environment.
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
        f"[WARN] Conda environment {conda_env!r} was requested but conda was not found; "
        f"using {sys.executable}.",
        file=sys.stderr,
    )
    return [sys.executable]


def _format_command(command: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(command)) if os.name == "nt" else shlex.join(command)


def _configure_environment(args: argparse.Namespace) -> None:
    if args.online:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
    else:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(args.hf_home)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AES HotFlip adversarial training v4.")
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
        default=_default_path(
            "/root/autodl-tmp/aes_adv_v4",
            REPO_ROOT / "outputs" / "aes_adv_v4",
        ),
    )
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--hotflip-weight", type=float, default=2.0)
    parser.add_argument("--hotflip-margin", type=float, default=0.1)
    parser.add_argument("--hotflip-fraction", type=float, default=1.0)
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
    parser.add_argument(
        "--current-python",
        action="store_true",
        help="Use the current Python interpreter instead of conda run.",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Allow Hugging Face network access. Offline mode is the default.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved configuration and command without training.",
    )
    return parser


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
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
        "use_hotflip_swaps": True,
        "hotflip_weight": args.hotflip_weight,
        "hotflip_n_sample_pos": args.hotflip_n_sample_pos,
        "hotflip_top_k_per_pos": args.hotflip_top_k_per_pos,
        "hotflip_max_candidates": args.hotflip_max_candidates,
        "hotflip_fraction": args.hotflip_fraction,
        "hotflip_margin": args.hotflip_margin,
    }


def _validate_config(config: dict[str, Any]) -> None:
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
    if not 0.0 <= config["hotflip_fraction"] <= 1.0:
        raise ValueError("hotflip_fraction must be between zero and one")


def main() -> int:
    args = build_parser().parse_args()
    _configure_environment(args)
    config = build_config(args)
    _validate_config(config)

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

    print("[V4] Resolved training configuration:")
    print(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"[RUN] {_format_command(command)}")
    if args.dry_run:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(command, env=os.environ.copy(), check=False)
    if completed.returncode == 0:
        print(f"[DONE] v4 training complete. Checkpoint: {args.output_dir / 'final'}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
