#!/usr/bin/env python3
"""Evaluate an AES checkpoint on clean essays and a supported attack."""

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


def build_parser(default_attack: str = "hotflip") -> argparse.ArgumentParser:
    if default_attack not in ("hotflip", "rudimentary", "mlm_guided"):
        raise ValueError(f"Unsupported default attack: {default_attack}")
    parser = argparse.ArgumentParser(
        description="Evaluate clean QWK/MAE and adversarial ASR for an AES checkpoint."
    )
    parser.add_argument(
        "--attack",
        choices=("hotflip", "rudimentary", "mlm_guided"),
        default=default_attack,
        help=f"Attack evaluated after clean metrics (default: {default_attack}).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_default_path(
            "/root/autodl-tmp/aes_hotflip_defense/best",
            REPO_ROOT / "outputs" / "aes_hotflip_defense" / "best",
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
            "/root/autodl-tmp/aes_checkpoint_evaluation",
            REPO_ROOT / "outputs" / "aes_checkpoint_evaluation",
        ),
    )
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--n-essays", type=int, default=1154)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--success-threshold", type=float, default=0.1)
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--n-sample-pos", type=int, default=8)
    parser.add_argument("--top-k-per-pos", type=int, default=2)
    parser.add_argument("--max-candidates-per-step", type=int, default=16)
    parser.add_argument("--max-token-edit-rate", type=float, default=0.1)
    parser.add_argument("--mlm-max-token-edit-rate", type=float, default=0.05)
    parser.add_argument("--mlm-model-name", default="answerdotai/ModernBERT-large")
    parser.add_argument(
        "--similarity-model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--minimum-cosine-similarity", type=float, default=0.90)
    parser.add_argument("--mlm-max-length", type=int, default=8192)
    parser.add_argument(
        "--mlm-dtype",
        choices=("float32", "bfloat16"),
        default="bfloat16",
    )
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
        "--no-progress",
        action="store_true",
        help="Disable interactive progress bars.",
    )
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
    paer_corrections: list[float] = []
    with torch.inference_mode():
        from tqdm.auto import tqdm

        for batch in tqdm(
            loader,
            desc="Clean evaluation",
            unit="batch",
            dynamic_ncols=True,
            disable=True if args.no_progress else None,
        ):
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            model_output = scorer.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = model_output.logits.squeeze(-1)
            predictions.extend(logits.float().cpu().numpy().tolist())
            labels.extend(batch["labels"].numpy().tolist())
            correction = getattr(model_output, "correction", None)
            if correction is not None:
                paer_corrections.extend(
                    correction.float().cpu().numpy().tolist()
                )

    prediction_array = np.asarray(predictions)
    label_array = np.asarray(labels)
    y_true = np.clip(np.round(label_array).astype(int), 0, 5)
    y_pred = np.clip(np.round(prediction_array).astype(int), 0, 5)
    rounded_qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    result: dict[str, float | int] = {
        "qwk": round(rounded_qwk, 4),
        "rounded_qwk": round(rounded_qwk, 4),
        "mae": round(float(np.mean(np.abs(prediction_array - label_array))), 4),
        "rmse": round(
            float(np.sqrt(np.mean((prediction_array - label_array) ** 2))),
            4,
        ),
        "n": len(labels),
    }
    if paer_corrections:
        correction_array = np.asarray(paer_corrections, dtype=float)
        result["mean_clean_correction"] = round(
            float(np.mean(correction_array)), 4
        )
        result["p95_clean_correction"] = round(
            float(np.quantile(correction_array, 0.95)), 4
        )
        result["clean_correction_rate_ge_0_05"] = round(
            float(np.mean(correction_array >= 0.05)), 4
        )

    threshold_path = args.thresholds
    if threshold_path is None:
        candidate = args.checkpoint.parent / "best_thresholds.json"
        if candidate.is_file():
            threshold_path = candidate
    if threshold_path is not None:
        threshold_data = json.loads(threshold_path.read_text(encoding="utf-8"))
        if isinstance(threshold_data, dict):
            threshold_values = threshold_data.get("thresholds_label_space", [])
            if not threshold_values:
                label_offset = float(threshold_data.get("label_offset", 1))
                threshold_values = [
                    float(value) - label_offset
                    for value in threshold_data.get("thresholds_score_space", [])
                ]
        else:
            threshold_values = threshold_data
        if threshold_values:
            threshold_predictions = np.digitize(
                prediction_array,
                np.asarray(threshold_values, dtype=float),
            )
            result["shared_threshold_qwk"] = round(
                float(
                    cohen_kappa_score(
                        y_true,
                        threshold_predictions,
                        weights="quadratic",
                    )
                ),
                4,
            )

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
        args.attack,
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
        "--seed",
        str(args.seed),
        "--success-threshold",
        str(args.success_threshold),
        "--n-steps",
        str(args.n_steps),
        "--beam-size",
        str(args.beam_size),
        "--n-sample-pos",
        str(args.n_sample_pos),
        "--top-k-per-pos",
        str(args.top_k_per_pos),
        "--max-candidates-per-step",
        str(args.max_candidates_per_step),
        "--max-token-edit-rate",
        str(args.max_token_edit_rate),
        "--mlm-max-token-edit-rate",
        str(args.mlm_max_token_edit_rate),
        "--mlm-model-name",
        args.mlm_model_name,
        "--similarity-model-name",
        args.similarity_model_name,
        "--minimum-cosine-similarity",
        str(args.minimum_cosine_similarity),
        "--mlm-max-length",
        str(args.mlm_max_length),
        "--mlm-dtype",
        args.mlm_dtype,
    ]
    if args.thresholds is not None:
        command.extend(["--thresholds", str(args.thresholds)])
    if args.no_progress:
        command.append("--no-progress")
    return command


def main(default_attack: str = "hotflip") -> int:
    args = build_parser(default_attack).parse_args()
    if args.n_essays <= 0:
        raise ValueError("n_essays must be greater than zero")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    positive_fields = (
        "n_steps",
        "beam_size",
        "n_sample_pos",
        "top_k_per_pos",
        "max_candidates_per_step",
    )
    for field in positive_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"{field} must be greater than zero")
    if args.success_threshold < 0:
        raise ValueError("success_threshold must be non-negative")
    if not 0 < args.max_token_edit_rate <= 1:
        raise ValueError("max_token_edit_rate must be in (0, 1]")
    if not 0 < args.mlm_max_token_edit_rate <= 1:
        raise ValueError("mlm_max_token_edit_rate must be in (0, 1]")
    if not -1 <= args.minimum_cosine_similarity <= 1:
        raise ValueError("minimum_cosine_similarity must be in [-1, 1]")
    if args.mlm_max_length <= 0:
        raise ValueError("mlm_max_length must be greater than zero")
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
                "seed": args.seed,
                "success_threshold": args.success_threshold,
                "n_steps": args.n_steps,
                "beam_size": args.beam_size,
                "n_sample_pos": args.n_sample_pos,
                "top_k_per_pos": args.top_k_per_pos,
                "max_candidates_per_step": args.max_candidates_per_step,
                "max_token_edit_rate": args.max_token_edit_rate,
                "mlm_max_token_edit_rate": args.mlm_max_token_edit_rate,
                "mlm_model_name": args.mlm_model_name,
                "similarity_model_name": args.similarity_model_name,
                "minimum_cosine_similarity": args.minimum_cosine_similarity,
                "mlm_max_length": args.mlm_max_length,
                "mlm_dtype": args.mlm_dtype,
                "clean": not args.skip_clean,
                "attack": None if args.skip_attack else args.attack,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"[RUN] {args.attack} command: {_format_command(_attack_command(args))}")
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
            f"MAE={clean_result['mae']:.4f} "
            f"RMSE={clean_result['rmse']:.4f}"
        )
        print(f"Saved: {clean_path}")

    if args.skip_attack:
        return 0

    print(
        f"=== Step 2: {args.attack} ASR on validation set ===",
        flush=True,
    )
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
