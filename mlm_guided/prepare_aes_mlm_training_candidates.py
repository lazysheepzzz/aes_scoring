#!/usr/bin/env python3
"""Batch-generate and cache model-independent D_MLM training candidates."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from text_scoring_adv_training.evaluation.aes.attacks.mlm_guided import (
    MLMGuidedCandidateGenerator,
    _text_sha256,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the fixed semantic candidate pool for D_MLM training."
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
            / "mlm_guided"
            / "training_candidate_pool_seed42.jsonl"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mlm-model-name", default="answerdotai/ModernBERT-large")
    parser.add_argument(
        "--similarity-model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--minimum-cosine-similarity", type=float, default=0.90)
    parser.add_argument("--mlm-max-length", type=int, default=8192)
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=REPO_ROOT / ".cache" / "huggingface",
    )
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def _existing_hashes(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    hashes: set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                hashes.add(json.loads(line)["text_sha256"])
            except Exception as exc:
                raise ValueError(
                    f"Invalid cache line {line_number} in {path}; use --force"
                ) from exc
    return hashes


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.mlm_max_length <= 0:
        raise ValueError("batch_size and mlm_max_length must be greater than zero")
    if not -1 <= args.minimum_cosine_similarity <= 1:
        raise ValueError("minimum_cosine_similarity must be in [-1, 1]")
    if not args.train_csv.is_file():
        raise FileNotFoundError(f"Training CSV not found: {args.train_csv}")

    if args.online:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
    else:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(args.hf_home)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataframe = pd.read_csv(args.train_csv)
    text_column = "full_text" if "full_text" in dataframe.columns else "text"
    texts = [str(value) for value in dataframe[text_column].tolist()]
    if args.force and args.output.exists():
        args.output.unlink()
    completed = _existing_hashes(args.output)
    pending = [
        index
        for index, text in enumerate(texts)
        if _text_sha256(text) not in completed
    ]
    # Length bucketing substantially reduces padding in the ModernBERT batch.
    pending.sort(key=lambda index: len(texts[index]))
    print(
        f"[CACHE] total={len(texts)} complete={len(completed)} "
        f"pending={len(pending)} batch_size={args.batch_size}",
        flush=True,
    )
    if not pending:
        return 0

    generator = MLMGuidedCandidateGenerator(
        mlm_model_name=args.mlm_model_name,
        similarity_model_name=args.similarity_model_name,
        device=args.device,
        dtype=(
            torch.bfloat16
            if str(args.device).startswith("cuda")
            else torch.float32
        ),
        n_sample_pos=1,
        top_k_per_pos=16,
        max_candidates=16,
        minimum_similarity=args.minimum_cosine_similarity,
        mlm_max_length=args.mlm_max_length,
        training_position_seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    progress = tqdm(total=len(pending), desc="MLM candidate cache", unit="essay")
    with args.output.open("a", encoding="utf-8", newline="\n") as output_file:
        for start in range(0, len(pending), args.batch_size):
            indices = pending[start : start + args.batch_size]
            batch_texts = [texts[index] for index in indices]
            candidate_groups = generator.generate_batch_for_training(batch_texts)
            for index, text, candidates in zip(indices, batch_texts, candidate_groups):
                record = {
                    "row_index": index,
                    "text_sha256": _text_sha256(text),
                    "mlm_model_name": args.mlm_model_name,
                    "similarity_model_name": args.similarity_model_name,
                    "minimum_cosine_similarity": args.minimum_cosine_similarity,
                    "mlm_max_length": args.mlm_max_length,
                    "position_seed": args.seed,
                    "replacements": [
                        {
                            "mlm_position": candidate["mlm_position"],
                            "mlm_old_id": candidate["mlm_old_id"],
                            "mlm_new_id": candidate["mlm_new_id"],
                            "cosine_similarity": candidate["cosine_similarity"],
                        }
                        for candidate in candidates
                    ],
                }
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()
            progress.update(len(indices))
    progress.close()

    manifest = {
        "train_csv": str(args.train_csv.resolve()),
        "output": str(args.output.resolve()),
        "n_rows": len(texts),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "mlm_model_name": args.mlm_model_name,
        "similarity_model_name": args.similarity_model_name,
        "minimum_cosine_similarity": args.minimum_cosine_similarity,
        "mlm_max_length": args.mlm_max_length,
        "training_protocol": "one sampled position, top-16, one attack step",
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
