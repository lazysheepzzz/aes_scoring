#!/usr/bin/env python3
"""Evaluate the v4 defended model on clean essays and HotFlip attacks."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


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


def _find_conda() -> str | None:
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    server_conda = Path("/root/miniconda3/bin/conda")
    if not conda and server_conda.is_file():
        conda = str(server_conda)
    return conda


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
    parser = argparse.ArgumentParser(
        description="Evaluate clean QWK/MAE and HotFlip ASR for the v4 model."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_default_path(
            "/root/autodl-tmp/aes_adv_v4/final",
            REPO_ROOT / "outputs" / "aes_adv_v4" / "final",
        ),
    )
    parser.add_argument(
        "--valid",
        type=Path,
        default=_default_path(
            "/root/autodl-tmp/data/valid_fold0.csv",
            REPO_ROOT / "data" / "valid_fold0.csv",
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_default_path(
            "/root/autodl-tmp/aes_v4_run",
            REPO_ROOT / "outputs" / "aes_v4_run",
        ),
    )
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--n-essays", type=int, default=1154)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
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
        help="Conda environment used to run evaluation. Server default: aes.",
    )
    parser.add_argument(
        "--current-python",
        action="store_true",
        help="Use the current Python interpreter instead of conda run.",
    )
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--skip-attack", action="store_true")
    parser.add_argument(
        "--online",
        action="store_true",
        help="Allow Hugging Face network access. Offline mode is the default.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved settings without loading a model.",
    )
    return parser


def _relaunch_in_conda(args: argparse.Namespace) -> int | None:
    if args.current_python or not args.conda_env:
        return None
    if os.environ.get("CONDA_DEFAULT_ENV") == args.conda_env:
        return None

    conda = _find_conda()
    if conda is None:
        print(
            f"[WARN] Conda environment {args.conda_env!r} was requested but conda "
            f"was not found; using {sys.executable}.",
            file=sys.stderr,
        )
        return None

    command = [
        conda,
        "run",
        "--no-capture-output",
        "-n",
        args.conda_env,
        "python",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
        "--current-python",
    ]
    print(f"[ENV] Relaunching: {_format_command(command)}")
    return subprocess.run(command, env=os.environ.copy(), check=False).returncode


def _evaluate_clean(args: argparse.Namespace) -> dict[str, float | int]:
    import numpy as np
    import torch
    from sklearn.metrics import cohen_kappa_score
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(REPO_ROOT))
    from text_scoring_adv_training.evaluation.aes.scorer import AESScorer
    from text_scoring_adv_training.training.aes_trainer import (
        AESCollator,
        KaggleEssayDataset,
    )

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    scorer = AESScorer(args.checkpoint, device=args.device, dtype=dtype)
    dataset = KaggleEssayDataset(
        str(args.valid),
        scorer.tokenizer,
        max_length=args.max_length,
    )
    collator = AESCollator(scorer.tokenizer, max_length=args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    predictions: list[float] = []
    labels: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            logits = scorer.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits.squeeze(-1)
            predictions.extend(logits.float().cpu().numpy().tolist())
            labels.extend(batch["labels"].numpy().tolist())

    prediction_array = np.asarray(predictions)
    label_array = np.asarray(labels)
    y_true = np.clip(np.round(label_array).astype(int), 0, 5)
    y_pred = np.clip(np.round(prediction_array).astype(int), 0, 5)
    result: dict[str, float | int] = {
        "qwk": round(
            float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
            4,
        ),
        "mae": round(float(np.mean(np.abs(prediction_array - label_array))), 4),
        "n": len(labels),
    }

    del loader, collator, dataset, scorer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _attack_command(args: argparse.Namespace) -> list[str]:
    entrypoint = (
        REPO_ROOT
        / "text_scoring_adv_training"
        / "evaluation"
        / "aes"
        / "run_attacks.py"
    )
    command = [
        sys.executable,
        str(entrypoint),
        "--victim",
        str(args.checkpoint),
        "--data",
        str(args.valid),
        "--attack",
        "hotflip",
        "--n-essays",
        str(args.n_essays),
        "--out",
        str(args.out),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--batch-size",
        str(args.batch_size),
    ]
    if args.thresholds is not None:
        command.extend(["--thresholds", str(args.thresholds)])
    return command


def main() -> int:
    args = build_parser().parse_args()
    if args.n_essays <= 0:
        raise ValueError("n_essays must be greater than zero")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    _configure_environment(args)

    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "valid": str(args.valid),
                "out": str(args.out),
                "n_essays": args.n_essays,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "device": args.device,
                "dtype": args.dtype,
                "clean": not args.skip_clean,
                "hotflip": not args.skip_attack,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"[RUN] HotFlip command: {_format_command(_attack_command(args))}")
    if args.dry_run:
        return 0

    relaunched = _relaunch_in_conda(args)
    if relaunched is not None:
        return relaunched

    if not args.checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint}")
    if not args.valid.is_file():
        raise FileNotFoundError(f"Validation CSV not found: {args.valid}")

    args.out.mkdir(parents=True, exist_ok=True)
    if not args.skip_clean:
        print("=== Step 1: QWK on clean validation set ===", flush=True)
        clean_result = _evaluate_clean(args)
        clean_path = args.out / "clean_qwk.json"
        clean_path.write_text(
            json.dumps(clean_result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"Clean QWK={clean_result['qwk']:.4f} "
            f"MAE={clean_result['mae']:.4f}"
        )
        print(f"Saved: {clean_path}")

    if args.skip_attack:
        return 0

    print("=== Step 2: HotFlip ASR on validation set ===", flush=True)
    completed = subprocess.run(
        _attack_command(args),
        env=os.environ.copy(),
        check=False,
    )
    if completed.returncode == 0:
        print(f"[DONE] Results in {args.out}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
