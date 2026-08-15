#!/usr/bin/env python3
"""Paired essay-level bootstrap for PAER-RH-v3 versus Mixed-AT-RH.

The attacks are independently optimized against each model, but the validation
essays are identical. Resampling essay identities therefore preserves the
paired experimental unit. Positive reductions mean lower ASR for PAER.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


DETAIL_PATHS = {
    "mixed_at_rh": {
        "rudimentary": "eval_rudimentary_mixed_at_rh_seed42/rudimentary_details.json",
        "hotflip": "eval_hotflip_mixed_at_rh_seed42/hotflip_details.json",
        "injection_external": (
            "eval_injection_mixed_at_rh_seed42/injection_external_details.json"
        ),
        "injection_self_dup": (
            "eval_injection_mixed_at_rh_seed42/injection_self_dup_details.json"
        ),
        "mlm_guided": "eval_mlm_mixed_at_rh_seed42/mlm_guided_details.json",
    },
    "paer_rh_v3": {
        "rudimentary": "eval_rudimentary_paer_rh_v3_seed42/rudimentary_details.json",
        "hotflip": "eval_hotflip_paer_rh_v3_seed42/hotflip_details.json",
        "injection_external": (
            "eval_injection_paer_rh_v3_seed42/injection_external_details.json"
        ),
        "injection_self_dup": (
            "eval_injection_paer_rh_v3_seed42/injection_self_dup_details.json"
        ),
        "mlm_guided": "eval_mlm_paer_rh_v3_seed42/mlm_guided_details.json",
    },
}


def _essay_key(row: dict[str, Any]) -> str:
    essay_id = row.get("essay_id")
    if essay_id is not None:
        return f"id:{essay_id}"
    original_text = row.get("original_text")
    if not isinstance(original_text, str):
        raise ValueError("Attack detail is missing essay_id and original_text")
    return f"text:{original_text}"


def load_success_by_essay(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("details"), list):
        raise ValueError(f"Invalid attack-details payload: {path}")
    threshold = float(payload.get("summary", {}).get("success_threshold", 0.1))
    values: dict[str, float] = {}
    for row in payload["details"]:
        if not isinstance(row, dict):
            continue
        key = _essay_key(row)
        if key in values:
            raise ValueError(f"Duplicate essay key {key!r} in {path}")
        delta = float(row["perturbed_score"]) - float(row["original_score"])
        values[key] = float(delta >= threshold)
    if not values:
        raise ValueError(f"No usable attack details in {path}")
    return values


def aligned_vectors(
    baseline: dict[str, float], candidate: dict[str, float]
) -> tuple[list[str], np.ndarray, np.ndarray]:
    baseline_keys = set(baseline)
    candidate_keys = set(candidate)
    if baseline_keys != candidate_keys:
        raise ValueError(
            "Baseline/candidate essay identities differ: "
            f"baseline_only={len(baseline_keys - candidate_keys)}, "
            f"candidate_only={len(candidate_keys - baseline_keys)}"
        )
    keys = sorted(baseline_keys)
    return (
        keys,
        np.asarray([baseline[key] for key in keys], dtype=np.float64),
        np.asarray([candidate[key] for key in keys], dtype=np.float64),
    )


def bootstrap_reduction(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    if baseline.shape != candidate.shape or baseline.ndim != 1:
        raise ValueError("baseline and candidate must be aligned 1-D arrays")
    if baseline.size < 2:
        raise ValueError("At least two paired essays are required")
    rng = np.random.default_rng(seed)
    reductions = np.empty(n_bootstrap, dtype=np.float64)
    chunk_size = 1000
    paired_reduction = baseline - candidate
    for start in range(0, n_bootstrap, chunk_size):
        stop = min(start + chunk_size, n_bootstrap)
        indices = rng.integers(
            0,
            baseline.size,
            size=(stop - start, baseline.size),
        )
        reductions[start:stop] = paired_reduction[indices].mean(axis=1)
    observed_reduction = float(paired_reduction.mean())
    probability_improvement = float(np.mean(reductions > 0.0))
    two_sided_p = float(
        min(1.0, 2.0 * min(np.mean(reductions <= 0.0), np.mean(reductions >= 0.0)))
    )
    return {
        "n_essays": int(baseline.size),
        "baseline_asr": float(baseline.mean()),
        "paer_asr": float(candidate.mean()),
        "absolute_asr_reduction": observed_reduction,
        "reduction_95ci": [
            float(np.quantile(reductions, 0.025)),
            float(np.quantile(reductions, 0.975)),
        ],
        "bootstrap_probability_reduction_gt_0": probability_improvement,
        "two_sided_bootstrap_p": two_sided_p,
    }


def _load_all_vectors(outputs_dir: Path) -> dict[str, tuple[list[str], np.ndarray, np.ndarray]]:
    vectors = {}
    for attack in DETAIL_PATHS["mixed_at_rh"]:
        baseline = load_success_by_essay(
            outputs_dir / DETAIL_PATHS["mixed_at_rh"][attack]
        )
        candidate = load_success_by_essay(
            outputs_dir / DETAIL_PATHS["paer_rh_v3"][attack]
        )
        vectors[attack] = aligned_vectors(baseline, candidate)
    return vectors


def _mean_family_vectors(
    vectors: dict[str, tuple[list[str], np.ndarray, np.ndarray]],
    attacks: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    reference_keys = vectors[attacks[0]][0]
    for attack in attacks[1:]:
        if vectors[attack][0] != reference_keys:
            raise ValueError(f"Essay order differs for family component {attack}")
    baseline = np.mean([vectors[attack][1] for attack in attacks], axis=0)
    candidate = np.mean([vectors[attack][2] for attack in attacks], axis=0)
    return baseline, candidate


def compute_report(
    outputs_dir: Path,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    vectors = _load_all_vectors(outputs_dir)
    comparisons: dict[str, Any] = {}
    for offset, attack in enumerate(("rudimentary", "hotflip", "mlm_guided")):
        comparisons[attack] = bootstrap_reduction(
            vectors[attack][1],
            vectors[attack][2],
            n_bootstrap=n_bootstrap,
            seed=seed + offset,
        )

    injection_baseline, injection_candidate = _mean_family_vectors(
        vectors, ("injection_external", "injection_self_dup")
    )
    comparisons["injection_family"] = bootstrap_reduction(
        injection_baseline,
        injection_candidate,
        n_bootstrap=n_bootstrap,
        seed=seed + 3,
    )
    rhi_baseline = np.mean(
        [vectors["rudimentary"][1], vectors["hotflip"][1], injection_baseline],
        axis=0,
    )
    rhi_candidate = np.mean(
        [vectors["rudimentary"][2], vectors["hotflip"][2], injection_candidate],
        axis=0,
    )
    comparisons["rhi_macro"] = bootstrap_reduction(
        rhi_baseline,
        rhi_candidate,
        n_bootstrap=n_bootstrap,
        seed=seed + 4,
    )
    return {
        "protocol": {
            "comparison": "PAER-RH-v3 minus data-matched Mixed-AT-RH",
            "paired_unit": "validation essay identity",
            "n_bootstrap": n_bootstrap,
            "bootstrap_seed": seed,
            "positive_reduction_means": "lower ASR for PAER-RH-v3",
            "injection_aggregation": "equal mean of external and self-duplication",
            "rhi_aggregation": "equal mean of Rudimentary, HotFlip, and Injection family",
        },
        "comparisons": comparisons,
    }


def _format_probability(value: float) -> str:
    return f"{value:.4f}"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    labels = {
        "rudimentary": "Rudimentary",
        "hotflip": "HotFlip",
        "injection_family": "Injection family",
        "rhi_macro": "RHI Macro",
        "mlm_guided": "MLM-guided (attack-only)",
    }
    order = (
        "rudimentary",
        "hotflip",
        "injection_family",
        "rhi_macro",
        "mlm_guided",
    )
    lines = [
        "# PAER-RH-v3 vs Mixed-AT-RH paired bootstrap (seed 42)",
        "",
        "Positive reduction means lower ASR for PAER-RH-v3.",
        "",
        "| Attack | Mixed ASR | PAER ASR | ASR reduction | 95% CI | P(reduction > 0) | Two-sided p |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in order:
        row = report["comparisons"][name]
        low, high = row["reduction_95ci"]
        lines.append(
            "| "
            + " | ".join(
                (
                    labels[name],
                    f"{row['baseline_asr']:.4f}",
                    f"{row['paer_asr']:.4f}",
                    f"{row['absolute_asr_reduction']:.4f}",
                    f"[{low:.4f}, {high:.4f}]",
                    _format_probability(row["bootstrap_probability_reduction_gt_0"]),
                    _format_probability(row["two_sided_bootstrap_p"]),
                )
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Paired bootstrap PAER-RH-v3 versus Mixed-AT-RH ASR."
    )
    parser.add_argument("--outputs-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "outputs" / "paper_results"
    )
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be greater than zero")
    report = compute_report(
        args.outputs_dir,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "aes_paer_v3_vs_mixed_at_bootstrap_seed42.json"
    markdown_path = args.out_dir / "aes_paer_v3_vs_mixed_at_bootstrap_seed42.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(markdown_path, report)
    print(f"Saved: {json_path}")
    print(f"Saved: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
