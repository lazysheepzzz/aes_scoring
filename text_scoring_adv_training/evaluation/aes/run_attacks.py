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
import random
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
        else next(
            (
                c
                for c in df.columns
                if "text" in c.lower() or "essay" in c.lower()
            ),
            None,
        )
    )
    if text_col is None:
        raise ValueError(f"No essay text column found in {csv_path}")
    if score_col not in df.columns:
        raise ValueError(f"Score column {score_col!r} not found in {csv_path}")

    id_col = next(
        (column for column in ("essay_id", "id") if column in df.columns),
        None,
    )
    prompt_col = next(
        (
            column
            for column in ("prompt_name", "prompt_id", "prompt")
            if column in df.columns
        ),
        None,
    )
    essays = []
    for index, row in df.iterrows():
        essays.append(
            {
                "essay_id": str(row[id_col]) if id_col else f"idx_{index}",
                "text": str(row[text_col]),
                "score": float(row[score_col]),
                "prompt": str(row[prompt_col]) if prompt_col else None,
            }
        )
    if n is not None:
        essays = essays[:n]
    return essays


def build_attack(
    attack_name: str,
    scorer: AESScorer,
    device: str = "cuda",
    *,
    n_steps: int = 30,
    beam_size: int = 1,
    n_sample_pos: int = 8,
    top_k_per_pos: int = 2,
    max_candidates_per_step: int = 16,
    success_threshold: float = 0.1,
    max_token_edit_rate: float = 0.1,
):
    """Instantiate an attack using the unified AES evaluation protocol."""
    if attack_name == "rudimentary":
        return RudimentaryAttack(n_variants=1)
    elif attack_name == "injection":
        return InjectionAttack(mode="injection", position="random", n_variants=1)
    elif attack_name == "hotflip":
        return HotFlipAttack(
            scorer,
            n_steps=n_steps,
            beam_size=beam_size,
            n_sample_pos=n_sample_pos,
            top_k_per_pos=top_k_per_pos,
            max_candidates_per_step=max_candidates_per_step,
            threshold=success_threshold,
            max_token_edit_rate=max_token_edit_rate,
        )
    elif attack_name == "mlm_guided":
        return MLMGuidedAttack(scorer, sim_threshold=0.9, n_variants=3)
    else:
        raise ValueError(f"Unknown attack: {attack_name!r}")


def run(args):
    import numpy as np
    import torch

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"\n[3/4] Running attack: {atk_name}")
        attack = build_attack(
            atk_name,
            scorer,
            device=args.device,
            n_steps=args.n_steps,
            beam_size=args.beam_size,
            n_sample_pos=args.n_sample_pos,
            top_k_per_pos=args.top_k_per_pos,
            max_candidates_per_step=args.max_candidates_per_step,
            success_threshold=args.success_threshold,
            max_token_edit_rate=args.max_token_edit_rate,
        )

        def attack_fn(text):
            return attack.attack(text)

        result = evaluate_attack(
            attack_name=atk_name,
            attack_fn=attack_fn,
            scorer=scorer,
            essays=essays,
            thresholds=thresholds,
            batch_size=args.batch_size,
            success_threshold=args.success_threshold,
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
            "details": r.details,
        }
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(detail, f, indent=2, ensure_ascii=False)
        print(f"  Detail saved to {out_file}")

    manifest = {
        "victim": str(args.victim),
        "data": str(args.data),
        "attack": args.attack,
        "n_essays": len(essays),
        "seed": args.seed,
        "success_threshold": args.success_threshold,
        "hotflip": {
            "n_steps": args.n_steps,
            "beam_size": args.beam_size,
            "n_sample_pos": args.n_sample_pos,
            "top_k_per_pos": args.top_k_per_pos,
            "max_candidates_per_step": args.max_candidates_per_step,
            "max_token_edit_rate": args.max_token_edit_rate,
        },
    }
    with open(out_dir / "run_manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--success-threshold", type=float, default=0.1)
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--n-sample-pos", type=int, default=8)
    parser.add_argument("--top-k-per-pos", type=int, default=2)
    parser.add_argument("--max-candidates-per-step", type=int, default=16)
    parser.add_argument("--max-token-edit-rate", type=float, default=0.1)
    args = parser.parse_args()

    positive_fields = (
        "n_essays",
        "batch_size",
        "n_steps",
        "beam_size",
        "n_sample_pos",
        "top_k_per_pos",
        "max_candidates_per_step",
    )
    for field in positive_fields:
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be greater than zero")
    if args.success_threshold < 0:
        parser.error("--success-threshold must be non-negative")
    if not 0 < args.max_token_edit_rate <= 1:
        parser.error("--max-token-edit-rate must be in (0, 1]")

    run(args)


if __name__ == "__main__":
    main()
