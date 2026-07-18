#!/usr/bin/env python
"""
run_attacks.py — CLI entry point for AES robustness evaluation.

Usage:
    python run_attacks.py --victim /path/to/fold0_best \
                          --data /path/to/valid_fold0.csv \
                          --n-essays 200 \
                          --attack rudimentary \
                          --out results.json

    python run_attacks.py --victim /path/to/fold0_best \
                          --data /path/to/valid_fold0.csv \
                          --n-essays 200 \
                          --attack all \
                          --out results/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from text_scoring_adv_training.evaluation.aes.scorer import AESScorer
from text_scoring_adv_training.evaluation.aes.evaluate import evaluate_attack, print_summary, AttackResult
from text_scoring_adv_training.evaluation.aes.attacks.rudimentary import RudimentaryAttack
from text_scoring_adv_training.evaluation.aes.attacks.injection import InjectionAttack
from text_scoring_adv_training.evaluation.aes.attacks.hotflip import HotFlipAttack
from text_scoring_adv_training.evaluation.aes.attacks.mlm_guided import MLMGuidedAttack


AVAILABLE_ATTACKS = ["rudimentary", "injection", "hotflip", "mlm_guided", "all"]


def load_essays(csv_path: str | Path, n: int | None = None, score_col: str = "score"):
    """Load essays from CSV. Expects columns: 'full_text'/'text'/'essay', 'score'."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    # Resolve text column — try known names in order of specificity
    text_col = (
        "full_text" if "full_text" in df.columns
        else "text" if "text" in df.columns
        else "essay" if "essay" in df.columns
        else [c for c in df.columns if 'text' in c.lower() or 'essay' in c.lower()][0]
    )
    essays = list(zip(df[text_col].tolist(), df["score"].tolist()))
    if n is not None:
        essays = essays[:n]
    return essays


def build_attack(attack_name: str, scorer: AESScorer, device: str = "cuda"):
    """Instantiate an attack by name. v1 parameters."""
    if attack_name == "rudimentary":
        return RudimentaryAttack(n_variants=1)
    elif attack_name == "injection":
        return InjectionAttack(mode="injection", position="random", n_variants=1)
    elif attack_name == "hotflip":
        return HotFlipAttack(
            scorer,
            n_steps=20,
            beam_size=4,
            n_sample_pos=16,
            top_k_per_pos=8,
        )
    elif attack_name == "mlm_guided":
        return MLMGuidedAttack(scorer, sim_threshold=0.9, n_variants=3)
    else:
        raise ValueError(f"Unknown attack: {attack_name!r}")


def run(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load scorer
    print(f"[1/4] Loading scorer from {args.victim} ...")
    scorer = AESScorer(
        checkpoint_path=args.victim,
        thresholds_path=args.thresholds,
        device=args.device,
        dtype=args.dtype,
    )

    # Load essays
    print(f"[2/4] Loading essays from {args.data} ...")
    essays = load_essays(args.data, n=args.n_essays)
    print(f"    Loaded {len(essays)} essays")

    # Load thresholds
    thresholds = None
    if args.thresholds:
        with open(args.thresholds) as f:
            thresholds = json.load(f)
    elif (Path(args.victim).parent / "best_thresholds.json").exists():
        with open(Path(args.victim).parent / "best_thresholds.json") as f:
            thresholds = json.load(f)
        print(f"    Using thresholds: {thresholds}")

    # Run attacks
    if args.attack == "all":
        attack_names = ["rudimentary", "injection", "hotflip", "mlm_guided"]
    else:
        attack_names = [args.attack]

    results: list[AttackResult] = []
    for atk_name in attack_names:
        print(f"\n[3/4] Running attack: {atk_name}")
        attack = build_attack(atk_name, scorer, device=args.device)

        def attack_fn(text):
            return attack.attack(text)

        result = evaluate_attack(
            attack_name=atk_name,
            attack_fn=attack_fn,
            scorer=scorer,
            essays=essays,
            thresholds=thresholds,
            batch_size=args.batch_size,
        )
        results.append(result)

    # Print and save summary
    print(f"\n[4/4] Summary:")
    rows = print_summary(results, out_path=out_dir / "asr_summary.json")

    # Save per-essay details
    for r in results:
        out_file = out_dir / f"{r.attack_name}_details.json"
        detail = {
            "summary": r.summary(),
            "original_scores": r.original_scores,
            "perturbed_scores": r.perturbed_scores,
        }
        with open(out_file, "w") as f:
            json.dump(detail, f)
        print(f"  Detail saved to {out_file}")

    return rows


def main():
    parser = argparse.ArgumentParser(description="AES Attack Evaluation")
    parser.add_argument("--victim", required=True, help="Path to fold0_best checkpoint")
    parser.add_argument("--data", required=True, help="Path to valid_fold0.csv")
    parser.add_argument(
        "--thresholds", default=None, help="Path to best_thresholds.json"
    )
    parser.add_argument(
        "--attack",
        required=True,
        choices=AVAILABLE_ATTACKS,
        help="Which attack to run, or 'all'",
    )
    parser.add_argument(
        "--n-essays", type=int, default=200, help="Number of essays to evaluate"
    )
    parser.add_argument(
        "--out", default="aes_results/", help="Output directory for results"
    )
    parser.add_argument(
        "--device", default="cuda", help="Device (cuda or cpu)"
    )
    parser.add_argument(
        "--dtype", default="float32", help="Model dtype (float32 or bfloat16)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Scoring batch size"
    )
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
