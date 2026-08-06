#!/usr/bin/env python3
"""Prepare Rudimentary+HotFlip counterfactual training traces for PAER-AES.

Only accepted, score-increasing edits are written.  Each JSONL row represents
one accepted edit and stores its before/after text plus the victim score gain.
The same file is consumed by Mixed-AT-RH and PAER-RH so the comparison uses
identical adversarial data.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from text_scoring_adv_training.evaluation.aes.attacks.hotflip import HotFlipAttack
from text_scoring_adv_training.evaluation.aes.attacks.rudimentary import (
    IterativeRudimentaryAttack,
)
from text_scoring_adv_training.evaluation.aes.scorer import AESScorer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create the shared RH counterfactual trace dataset."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "deberta_checkpoints" / "fold0_best",
        help="Frozen victim used to attribute accepted step-wise score gains.",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=REPO_ROOT / "data" / "train_fold0.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "paer"
            / "rh_counterfactual_training_traces_seed42.jsonl"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--attack-fraction",
        type=float,
        default=0.5,
        help="Fraction of training essays attacked once; families are balanced.",
    )
    parser.add_argument("--max-essays", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--max-candidates-per-step", type=int, default=16)
    parser.add_argument("--n-sample-pos", type=int, default=8)
    parser.add_argument("--top-k-per-pos", type=int, default=2)
    parser.add_argument("--success-threshold", type=float, default=0.1)
    parser.add_argument("--max-token-edit-rate", type=float, default=0.1)
    parser.add_argument("--score-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_training_rows(csv_path: Path) -> list[dict]:
    frame = pd.read_csv(csv_path)
    text_col = "full_text" if "full_text" in frame.columns else "text"
    if text_col not in frame.columns or "score" not in frame.columns:
        raise ValueError("Training CSV must contain full_text/text and score")
    id_col = next((name for name in ("essay_id", "id") if name in frame.columns), None)
    rows = []
    for index, row in frame.iterrows():
        rows.append(
            {
                "row_index": int(index),
                "essay_id": str(row[id_col]) if id_col else f"train_{index}",
                "text": str(row[text_col]),
                "score": float(row["score"]),
            }
        )
    return rows


def _select_attacks(
    n_rows: int,
    fraction: float,
    seed: int,
) -> dict[int, str]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("attack_fraction must be in (0, 1]")
    n_selected = max(1, round(n_rows * fraction))
    selected = random.Random(seed).sample(range(n_rows), min(n_selected, n_rows))
    selected.sort()
    return {
        row_index: ("hotflip" if order % 2 == 0 else "rudimentary")
        for order, row_index in enumerate(selected)
    }


def _load_progress(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(value) for value in payload.get("processed_row_indices", [])}


def _save_progress(path: Path, processed: set[int]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {"processed_row_indices": sorted(processed)},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = build_parser().parse_args()
    rows = _load_training_rows(args.train_csv)
    if args.max_essays is not None:
        if args.max_essays <= 0:
            raise ValueError("max_essays must be greater than zero")
        rows = rows[: args.max_essays]
    attack_for_row = _select_attacks(len(rows), args.attack_fraction, args.seed)

    resolved = {
        "checkpoint": str(args.checkpoint.resolve()),
        "train_csv": str(args.train_csv.resolve()),
        "output": str(args.output),
        "seed": args.seed,
        "n_input_essays": len(rows),
        "n_selected_essays": len(attack_for_row),
        "attack_counts": dict(Counter(attack_for_row.values())),
        "n_steps": args.n_steps,
        "max_candidates_per_step": args.max_candidates_per_step,
        "n_sample_pos": args.n_sample_pos,
        "top_k_per_pos": args.top_k_per_pos,
        "success_threshold": args.success_threshold,
        "max_token_edit_rate": args.max_token_edit_rate,
        "score_batch_size": args.score_batch_size,
        "mlm_excluded": True,
    }
    print("[PAER TRACE] Resolved configuration:")
    print(json.dumps(resolved, indent=2, ensure_ascii=False))
    if args.dry_run:
        return 0
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)
    if not args.train_csv.is_file():
        raise FileNotFoundError(args.train_csv)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    progress_path = args.output.with_suffix(".progress.json")
    manifest_path = args.output.with_suffix(".manifest.json")
    run_config_path = args.output.with_suffix(".run_config.json")
    if args.overwrite:
        args.output.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
        run_config_path.unlink(missing_ok=True)
    if run_config_path.is_file():
        previous_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        if previous_config != resolved:
            raise ValueError(
                "Existing trace progress was created with a different "
                "configuration. Use a new readable output name or pass "
                "--overwrite to restart intentionally."
            )
    else:
        run_config_path.write_text(
            json.dumps(resolved, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    processed = _load_progress(progress_path)

    print(f"Loading frozen attribution victim: {args.checkpoint}", flush=True)
    scorer = AESScorer(
        args.checkpoint,
        device=args.device,
        dtype=torch.float32,
    )
    hotflip = HotFlipAttack(
        scorer,
        n_steps=args.n_steps,
        beam_size=1,
        n_sample_pos=args.n_sample_pos,
        top_k_per_pos=args.top_k_per_pos,
        max_candidates_per_step=args.max_candidates_per_step,
        batch_size=args.score_batch_size,
        threshold=args.success_threshold,
        max_token_edit_rate=args.max_token_edit_rate,
        record_intermediate_texts=True,
    )
    rudimentary = IterativeRudimentaryAttack(
        scorer,
        n_steps=args.n_steps,
        beam_size=1,
        candidates_per_step=args.max_candidates_per_step,
        batch_size=args.score_batch_size,
        threshold=args.success_threshold,
        max_token_edit_rate=args.max_token_edit_rate,
        record_intermediate_texts=True,
    )

    selected_rows = [row for row in rows if row["row_index"] in attack_for_row]
    written_records = 0
    successful_essays = 0
    with args.output.open("a", encoding="utf-8") as output_file:
        progress = tqdm(
            selected_rows,
            desc="Preparing RH traces",
            unit="essay",
            dynamic_ncols=True,
            disable=True if args.no_progress else None,
        )
        for row in progress:
            row_index = row["row_index"]
            if row_index in processed:
                continue
            per_essay_seed = args.seed * 1_000_003 + row_index
            random.seed(per_essay_seed)
            np.random.seed(per_essay_seed % (2**32 - 1))
            torch.manual_seed(per_essay_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(per_essay_seed)

            attack_name = attack_for_row[row_index]
            attack = hotflip if attack_name == "hotflip" else rudimentary
            _, history = attack.attack(row["text"])
            if history:
                successful_essays += 1
            for accepted_order, entry in enumerate(history, start=1):
                score_after = float(entry["score"])
                step_gain = float(entry["step_gain"])
                output_file.write(
                    json.dumps(
                        {
                            "record_version": 1,
                            "record_id": (
                                f"{row['essay_id']}:{attack_name}:{accepted_order}"
                            ),
                            "essay_id": row["essay_id"],
                            "row_index": row_index,
                            "attack": attack_name,
                            "label_score_space": row["score"],
                            "original_text": row["text"],
                            "before_text": entry["before_text"],
                            "adversarial_text": entry["after_text"],
                            "victim_score_before": score_after - step_gain,
                            "victim_score_after": score_after,
                            "step_gain": step_gain,
                            "cumulative_delta": float(entry["delta"]),
                            "accepted_edit_order": accepted_order,
                            "attack_step": int(entry["step"]),
                            "attribution_seed": per_essay_seed,
                        },
                        ensure_ascii=False,
                    ) + "\n"
                )
                written_records += 1
            output_file.flush()
            processed.add(row_index)
            _save_progress(progress_path, processed)
            progress.set_postfix(records=written_records)

    total_trace_records = 0
    trace_records_by_attack: Counter[str] = Counter()
    if args.output.is_file():
        with args.output.open(encoding="utf-8") as input_file:
            for line in input_file:
                if not line.strip():
                    continue
                total_trace_records += 1
                trace_records_by_attack[
                    str(json.loads(line)["attack"])
                ] += 1
    resolved.update(
        {
            "processed_essays": len(processed),
            "successful_essays_this_run": successful_essays,
            "records_written_this_run": written_records,
            "total_trace_records": total_trace_records,
            "trace_records_by_attack": dict(trace_records_by_attack),
            "trace_semantics": (
                "soft supervision for victim-induced positive score gain; "
                "not a human quality label"
            ),
        }
    )
    manifest_path.write_text(
        json.dumps(resolved, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved trace dataset: {args.output}")
    print(f"Saved manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
