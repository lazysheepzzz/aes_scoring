"""
Unified evaluation framework for AES attacks.
Computes ASR, avg_delta, band_asr per attack.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

__all__ = ["evaluate_attack", "AttackResult", "print_summary"]


class AttackResult:
    """Container for a single attack evaluation."""

    def __init__(self, attack_name: str, success_threshold: float = 0.1):
        self.attack_name = attack_name
        self.success_threshold = success_threshold
        self.original_scores: List[float] = []
        self.perturbed_scores: List[List[float]] = []
        self.original_band: List[int] = []
        self.perturbed_band: List[int] = []
        self.details: List[Dict] = []
        self.n_essays: int = 0
        self.n_successful: int = 0
        self.delta_sum: float = 0.0
        self.n_band_cross: int = 0

    def add_essay(
        self,
        orig_score: float,
        pert_scores: List[float],
        thresholds: Optional[List[float]] = None,
        *,
        essay_id: str | None = None,
        true_score: float | None = None,
        prompt: str | None = None,
        original_text: str | None = None,
        perturbed_texts: Optional[List[str]] = None,
        histories: Optional[List[List[Dict]]] = None,
    ):
        self.original_scores.append(orig_score)
        self.perturbed_scores.append(pert_scores)
        self.n_essays += 1

        # Compute band
        def score_to_band(s: float, thresh) -> int:
            if thresh is None:
                return int(np.clip(np.round(s), 0, 5))
            # thresholds may be a dict (best_thresholds.json format) or a plain list
            thresh_list: List[float]
            if isinstance(thresh, dict):
                # AES logits use label space (0-5).  Convert score-space
                # thresholds (1-6) only when label-space values are absent.
                thresh_list = thresh.get("thresholds_label_space", [])
                if not thresh_list:
                    label_offset = float(thresh.get("label_offset", 1))
                    thresh_list = [
                        float(value) - label_offset
                        for value in thresh.get("thresholds_score_space", [])
                    ]
            else:
                thresh_list = thresh
            for i, t in enumerate(thresh_list):
                if s < float(t):
                    return i
            return len(thresh_list)

        orig_band = score_to_band(orig_score, thresholds)
        self.original_band.append(orig_band)

        best_index = int(np.argmax(pert_scores)) if pert_scores else -1
        best_pert = pert_scores[best_index] if best_index >= 0 else orig_score
        pert_band = score_to_band(best_pert, thresholds)
        self.perturbed_band.append(pert_band)
        delta = best_pert - orig_score

        if pert_scores and delta >= self.success_threshold:
            self.n_successful += 1

        if pert_scores:
            self.delta_sum += delta

        # Score-inflation attacks only count upward crossings.
        if pert_band > orig_band:
            self.n_band_cross += 1

        best_text = None
        best_history: List[Dict] = []
        if best_index >= 0 and perturbed_texts and best_index < len(perturbed_texts):
            best_text = perturbed_texts[best_index]
        if best_index >= 0 and histories and best_index < len(histories):
            best_history = histories[best_index] or []
        self.details.append(
            {
                "essay_id": essay_id,
                "true_score": true_score,
                "prompt": prompt,
                "original_text": original_text,
                "perturbed_text": best_text,
                "original_score": orig_score,
                "perturbed_score": best_pert,
                "delta": delta,
                "success": bool(
                    best_index >= 0 and delta >= self.success_threshold
                ),
                "original_band": orig_band,
                "perturbed_band": pert_band,
                "history": best_history,
            }
        )

    def summary(self) -> Dict:
        n = self.n_essays or 1
        return {
            "attack": self.attack_name,
            "n_essays": self.n_essays,
            "success_threshold": self.success_threshold,
            "asr": round(self.n_successful / n, 4),
            "avg_delta": round(self.delta_sum / n, 4),
            "avg_perturbed_score": round(
                np.mean([max(s) if s else 0 for s in self.perturbed_scores]), 4
            ),
            "avg_original_score": round(np.mean(self.original_scores), 4),
            "band_asr": round(self.n_band_cross / n, 4),
            "upward_band_asr": round(self.n_band_cross / n, 4),
        }


def evaluate_attack(
    attack_name: str,
    attack_fn: Callable[[str], List[str]],
    scorer,
    essays: List[Tuple[str, float]],
    thresholds: Optional[List[float]] = None,
    batch_size: int = 32,
    success_threshold: float = 0.1,
) -> AttackResult:
    """
    Evaluate a single attack on a list of essays.

    Args:
        attack_name: name of the attack for reporting.
        attack_fn: function that takes (text) → list of perturbed variants.
        scorer: AESScorer instance.
        essays: list of (text, original_score) tuples.
        thresholds: optional score thresholds for band computation.
        batch_size: batch size for scorer.

    Returns:
        AttackResult with computed metrics.
    """
    if success_threshold < 0:
        raise ValueError("success_threshold must be non-negative")
    result = AttackResult(attack_name, success_threshold=success_threshold)

    for i in range(0, len(essays), batch_size):
        batch = essays[i : i + batch_size]
        records: List[Dict] = []
        for offset, essay in enumerate(batch):
            if isinstance(essay, dict):
                record = dict(essay)
            else:
                record = {
                    "text": essay[0],
                    "score": essay[1] if len(essay) > 1 else None,
                }
            record.setdefault("essay_id", f"idx_{i + offset}")
            records.append(record)
        texts = [str(record["text"]) for record in records]

        # Score originals in batch
        orig_scores = scorer.score_batch(texts, batch_size=batch_size)

        # Generate perturbations
        all_perturbed: List[List[str]] = []
        all_histories: List[List[List[Dict]]] = []
        for text in texts:
            try:
                raw = attack_fn(text)
            except Exception as exc:
                print(f"  [WARN] attack_fn failed on essay {i}: {exc}", file=sys.stderr)
                raw = []
            # Normalise: some attacks return (text, history) tuples, others return [text, ...]
            if isinstance(raw, tuple):
                variants = [raw[0]] if raw[0] else []
                histories = [raw[1] if len(raw) > 1 else []] if variants else []
            elif isinstance(raw, str):
                variants = [raw]
                histories = [[]]
            else:
                variants = list(raw) if raw else []
                histories = [[] for _ in variants]
            all_perturbed.append(variants)
            all_histories.append(histories)

        # Score perturbations
        pert_scores: List[List[float]] = []
        for variants in all_perturbed:
            if variants:
                pert_scores.append(scorer.score_batch(variants, batch_size=len(variants)))
            else:
                pert_scores.append([])

        # Record results
        for record, orig_text, orig_score, p_scores, variants, histories in zip(
            records,
            texts,
            orig_scores,
            pert_scores,
            all_perturbed,
            all_histories,
        ):
            result.add_essay(
                orig_score,
                p_scores,
                thresholds,
                essay_id=str(record.get("essay_id")),
                true_score=record.get("score"),
                prompt=record.get("prompt"),
                original_text=orig_text,
                perturbed_texts=variants,
                histories=histories,
            )

    return result


def print_summary(results: List[AttackResult], out_path: Optional[Path] = None):
    """Print a formatted ASR table."""
    rows = [r.summary() for r in results]

    header = f"{'Attack':<20} {'N':>6} {'ASRΔ':>8} {'avgΔ':>8} {'up_band':>10} {'avg_orig':>10} {'avg_pert':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['attack']:<20} "
            f"{row['n_essays']:>6} "
            f"{row['asr']:>8.4f} "
            f"{row['avg_delta']:>8.4f} "
            f"{row['band_asr']:>10.4f} "
            f"{row['avg_original_score']:>10.4f} "
            f"{row['avg_perturbed_score']:>10.4f}"
        )

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nResults saved to {out_path}")

    return rows
