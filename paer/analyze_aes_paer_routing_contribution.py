#!/usr/bin/env python3
"""Measure PAER routing activity on already generated adversarial texts.

This is a fixed-adversarial-set diagnostic, not a replacement for an adaptive
route-off attack.  It requires only forward passes and never regenerates or
changes an attack.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from text_scoring_adv_training.evaluation.aes.scorer import AESScorer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay saved adversarial texts through PAER and its uncorrected "
            "base branch to quantify routing contribution."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--attack-details",
        type=Path,
        nargs="+",
        required=True,
        help="One or more *_details.json files from completed AES attacks.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("float32", "bfloat16"), default="float32"
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser


def _load_attack_details(path: Path) -> tuple[str, float, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in {path}")
    summary = payload.get("summary")
    details = payload.get("details")
    if not isinstance(summary, dict) or not isinstance(details, list):
        raise ValueError(f"Missing summary/details in {path}")
    attack = str(summary.get("attack", path.stem.removesuffix("_details")))
    threshold = float(summary.get("success_threshold", 0.1))
    usable = [
        row
        for row in details
        if isinstance(row, dict)
        and isinstance(row.get("original_text"), str)
        and isinstance(row.get("perturbed_text"), str)
    ]
    if not usable:
        raise ValueError(f"No usable original/perturbed text pairs in {path}")
    return attack, threshold, usable


@torch.inference_mode()
def _score_unique_texts(
    scorer: AESScorer,
    texts: list[str],
    *,
    batch_size: int,
    max_length: int,
    show_progress: bool,
) -> dict[str, tuple[float, float, float]]:
    unique_texts = list(dict.fromkeys(texts))
    scored: dict[str, tuple[float, float, float]] = {}
    for start in tqdm(
        range(0, len(unique_texts), batch_size),
        desc="PAER routing replay",
        unit="batch",
        dynamic_ncols=True,
        disable=not show_progress,
    ):
        batch_texts = unique_texts[start : start + batch_size]
        encoded = scorer.tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        output = scorer.model(
            input_ids=encoded["input_ids"].to(scorer.device),
            attention_mask=encoded["attention_mask"].to(scorer.device),
        )
        base_logits = getattr(output, "base_logits", None)
        corrections = getattr(output, "correction", None)
        if base_logits is None or corrections is None:
            raise TypeError(
                "Checkpoint is not a PAER checkpoint with base_logits/correction"
            )
        routed = output.logits.squeeze(-1).float().cpu().numpy()
        base = base_logits.squeeze(-1).float().cpu().numpy()
        correction = corrections.float().cpu().numpy()
        for text, routed_value, base_value, correction_value in zip(
            batch_texts, routed, base, correction
        ):
            scored[text] = (
                float(routed_value),
                float(base_value),
                float(correction_value),
            )
    return scored


def _qwk(true_scores: np.ndarray, predictions: np.ndarray) -> float | None:
    if len(true_scores) < 2:
        return None
    true_bins = np.clip(np.round(true_scores - 1.0).astype(int), 0, 5)
    if len(np.unique(true_bins)) < 2:
        return None
    prediction_bins = np.clip(np.round(predictions).astype(int), 0, 5)
    return float(cohen_kappa_score(true_bins, prediction_bins, weights="quadratic"))


def compute_routing_metrics(
    records: list[dict[str, Any]],
    scored: dict[str, tuple[float, float, float]],
    *,
    success_threshold: float,
) -> dict[str, Any]:
    routed_original = np.asarray(
        [scored[row["original_text"]][0] for row in records], dtype=float
    )
    base_original = np.asarray(
        [scored[row["original_text"]][1] for row in records], dtype=float
    )
    correction_original = np.asarray(
        [scored[row["original_text"]][2] for row in records], dtype=float
    )
    routed_adversarial = np.asarray(
        [scored[row["perturbed_text"]][0] for row in records], dtype=float
    )
    base_adversarial = np.asarray(
        [scored[row["perturbed_text"]][1] for row in records], dtype=float
    )
    correction_adversarial = np.asarray(
        [scored[row["perturbed_text"]][2] for row in records], dtype=float
    )
    saved_original = np.asarray(
        [float(row["original_score"]) for row in records], dtype=float
    )
    saved_adversarial = np.asarray(
        [float(row["perturbed_score"]) for row in records], dtype=float
    )
    routed_delta = routed_adversarial - routed_original
    base_delta = base_adversarial - base_original
    correction_lift = correction_adversarial - correction_original
    routed_asr = float(np.mean(routed_delta >= success_threshold))
    base_asr = float(np.mean(base_delta >= success_threshold))
    true_scores = np.asarray(
        [float(row["true_score"]) for row in records], dtype=float
    )

    return {
        "n_pairs": len(records),
        "success_threshold": success_threshold,
        "saved_score_replay_max_abs_error": float(
            max(
                np.max(np.abs(routed_original - saved_original)),
                np.max(np.abs(routed_adversarial - saved_adversarial)),
            )
        ),
        "mean_original_correction": float(np.mean(correction_original)),
        "mean_adversarial_correction": float(np.mean(correction_adversarial)),
        "p95_adversarial_correction": float(
            np.quantile(correction_adversarial, 0.95)
        ),
        "mean_adversarial_minus_original_correction": float(
            np.mean(correction_lift)
        ),
        "fraction_positive_correction_lift": float(
            np.mean(correction_lift > 0.0)
        ),
        "fraction_adversarial_correction_ge_0_05": float(
            np.mean(correction_adversarial >= 0.05)
        ),
        "routed_fixed_set_asr": routed_asr,
        "base_branch_fixed_set_asr": base_asr,
        "fixed_set_asr_reduction_from_routing": base_asr - routed_asr,
        "routed_mean_delta": float(np.mean(routed_delta)),
        "base_branch_mean_delta": float(np.mean(base_delta)),
        "mean_delta_reduction_from_routing": float(
            np.mean(base_delta - routed_delta)
        ),
        "routed_original_qwk": _qwk(true_scores, routed_original),
        "routed_adversarial_qwk": _qwk(true_scores, routed_adversarial),
        "base_branch_original_qwk": _qwk(true_scores, base_original),
        "base_branch_adversarial_qwk": _qwk(true_scores, base_adversarial),
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.max_length <= 0:
        raise ValueError("batch_size and max_length must be greater than zero")
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    paer_config_path = args.checkpoint / "paer_config.json"
    paer_config = (
        json.loads(paer_config_path.read_text(encoding="utf-8"))
        if paer_config_path.is_file()
        else {}
    )
    if not (args.checkpoint / "paer_config.json").is_file():
        raise FileNotFoundError(
            f"PAER configuration not found: {args.checkpoint / 'paer_config.json'}"
        )

    loaded = []
    all_texts: list[str] = []
    for path in args.attack_details:
        if not path.is_file():
            raise FileNotFoundError(f"Attack details not found: {path}")
        attack, threshold, records = _load_attack_details(path)
        loaded.append((path, attack, threshold, records))
        for row in records:
            all_texts.extend((row["original_text"], row["perturbed_text"]))

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    scorer = AESScorer(
        args.checkpoint,
        device=args.device,
        dtype=dtype,
    )
    scored = _score_unique_texts(
        scorer,
        all_texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        show_progress=not args.no_progress,
    )

    attack_rows = []
    for path, attack, threshold, records in loaded:
        metrics = compute_routing_metrics(
            records,
            scored,
            success_threshold=threshold,
        )
        attack_rows.append(
            {
                "attack": attack,
                "details_path": str(path.resolve()),
                **metrics,
            }
        )

    macro_fields = (
        "routed_fixed_set_asr",
        "base_branch_fixed_set_asr",
        "fixed_set_asr_reduction_from_routing",
        "mean_adversarial_minus_original_correction",
        "mean_delta_reduction_from_routing",
    )
    payload = {
        "protocol": {
            "checkpoint": str(args.checkpoint.resolve()),
            "paer_model_type": paer_config.get("model_type", "unknown"),
            "base_logits_semantics": paer_config.get(
                "base_logits_semantics",
                "global scorer before scalar PAER correction",
            ),
            "diagnostic": "fixed_adversarial_set_routing_replay",
            "adaptive_route_off_attack": False,
            "interpretation_limit": (
                "Measures routing contribution on saved adversarial texts; "
                "it does not measure an attacker re-optimized against the "
                "base branch."
            ),
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
        },
        "attacks": attack_rows,
        "macro_average": {
            field: float(np.mean([row[field] for row in attack_rows]))
            for field in macro_fields
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["macro_average"], indent=2, ensure_ascii=False))
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
