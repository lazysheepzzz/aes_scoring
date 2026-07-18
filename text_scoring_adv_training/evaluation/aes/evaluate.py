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

    def __init__(self, attack_name: str):
        self.attack_name = attack_name
        self.original_scores: List[float] = []
        self.perturbed_scores: List[List[float]] = []
        self.original_band: List[int] = []
        self.perturbed_band: List[List[int]] = []
        self.n_essays: int = 0
        self.n_successful: int = 0
        self.delta_sum: float = 0.0
        self.n_band_cross: int = 0

    def add_essay(
        self,
        orig_score: float,
        pert_scores: List[float],
        thresholds: Optional[List[float]] = None,
    ):
        self.original_scores.append(orig_score)
        self.perturbed_scores.append(pert_scores)
        self.n_essays += 1

        # Compute band
        def score_to_band(s: float, thresh) -> int:
            if thresh is None:
                return int(s)
            # thresholds may be a dict (best_thresholds.json format) or a plain list
            thresh_list: List[float]
            if isinstance(thresh, dict):
                # Prefer score_space thresholds; fall back to label_space
                thresh_list = thresh.get("thresholds_score_space") or thresh.get("thresholds_label_space", [])
            else:
                thresh_list = thresh
            for i, t in enumerate(thresh_list):
                if s < float(t):
                    return i
            return len(thresh_list)

        orig_band = score_to_band(orig_score, thresholds)
        self.original_band.append(orig_band)

        best_pert = max(pert_scores) if pert_scores else orig_score
        pert_band = score_to_band(best_pert, thresholds)
        self.perturbed_band.append(pert_band)

        # Attack success: any perturbed score > original
        if pert_scores and max(pert_scores) > orig_score:
            self.n_successful += 1

        # Delta: original - perturbed (positive = score went up = attack succeeded)
        if pert_scores:
            self.delta_sum += max(pert_scores) - orig_score

        # Band crossing
        if orig_band != pert_band:
            self.n_band_cross += 1

    def summary(self) -> Dict:
        n = self.n_essays or 1
        return {
            "attack": self.attack_name,
            "n_essays": self.n_essays,
            "asr": round(self.n_successful / n, 4),
            "avg_delta": round(self.delta_sum / n, 4),
            "avg_perturbed_score": round(
                np.mean([max(s) if s else 0 for s in self.perturbed_scores]), 4
            ),
            "avg_original_score": round(np.mean(self.original_scores), 4),
            "band_asr": round(self.n_band_cross / n, 4),
        }


def evaluate_attack(
    attack_name: str,
    attack_fn: Callable[[str], List[str]],
    scorer,
    essays: List[Tuple[str, float]],
    thresholds: Optional[List[float]] = None,
    batch_size: int = 32,
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
    result = AttackResult(attack_name)

    for i in range(0, len(essays), batch_size):
        batch = essays[i : i + batch_size]
        texts = [e[0] for e in batch]

        # Score originals in batch
        orig_scores = scorer.score_batch(texts, batch_size=batch_size)

        # Generate perturbations
        all_perturbed: List[List[str]] = []
        for text in texts:
            try:
                raw = attack_fn(text)
            except Exception as exc:
                print(f"  [WARN] attack_fn failed on essay {i}: {exc}", file=sys.stderr)
                raw = []
            # Normalise: some attacks return (text, history) tuples, others return [text, ...]
            if isinstance(raw, tuple):
                variants = [raw[0]] if raw[0] else []
            else:
                variants = raw
            all_perturbed.append(variants)

        # Score perturbations
        pert_scores: List[List[float]] = []
        for variants in all_perturbed:
            if variants:
                pert_scores.append(scorer.score_batch(variants, batch_size=len(variants)))
            else:
                pert_scores.append([])

        # Record results
        for orig_text, orig_score, p_scores in zip(texts, orig_scores, pert_scores):
            result.add_essay(orig_score, p_scores, thresholds)

    return result


def print_summary(results: List[AttackResult], out_path: Optional[Path] = None):
    """Print a formatted ASR table."""
    rows = [r.summary() for r in results]

    header = f"{'Attack':<20} {'N':>6} {'ASR':>8} {'avgΔ':>8} {'band_ASR':>10} {'avg_orig':>10} {'avg_pert':>10}"
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
