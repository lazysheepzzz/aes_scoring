#!/usr/bin/env python3
"""Prepare fixed offline Injection pairs for the AES defense baseline.

This mirrors the original repository's pre-generated Injection-data design:
candidate construction happens before training and does not query the victim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _split_sentences(text: str) -> list[str]:
    return [
        sentence
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
        if sentence.strip()
    ]


def _external_pair(
    text: str,
    sentence_bank: list[str],
    rng: random.Random,
) -> dict[str, Any] | None:
    sentences = _split_sentences(text)
    if not sentences:
        return None
    injected_sentence = rng.choice(sentence_bank)
    destination = rng.randrange(len(sentences) + 1)
    adversarial = list(sentences)
    adversarial.insert(destination, injected_sentence)
    return {
        "adversarial_text": " ".join(adversarial),
        "injected_sentence": injected_sentence,
        "source_sentence_index": None,
        "destination_index": destination,
    }


def _self_duplication_pair(
    text: str,
    rng: random.Random,
) -> dict[str, Any] | None:
    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return None
    valid_pairs = [
        (source, destination)
        for source in range(len(sentences))
        for destination in range(len(sentences) + 1)
        if destination not in {source, source + 1}
    ]
    if not valid_pairs:
        return None
    source, destination = rng.choice(valid_pairs)
    injected_sentence = sentences[source]
    adversarial = list(sentences)
    adversarial.insert(destination, injected_sentence)
    return {
        "adversarial_text": " ".join(adversarial),
        "injected_sentence": injected_sentence,
        "source_sentence_index": source,
        "destination_index": destination,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare fixed external/self-duplication AES Injection pairs."
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=REPO_ROOT / "data" / "train_fold0.csv",
    )
    parser.add_argument(
        "--sentence-bank",
        type=Path,
        default=REPO_ROOT / "injection" / "wikipedia_sentences_100.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "injection"
            / "aes_injection_training_pairs_seed42.jsonl"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--training-fraction", type=float, default=0.5)
    parser.add_argument(
        "--max-input-essays",
        type=int,
        help="Optional first-N smoke-test limit; omit for the formal pool.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0 < args.training_fraction <= 1:
        raise ValueError("training_fraction must be in (0, 1]")
    if args.max_input_essays is not None and args.max_input_essays <= 0:
        raise ValueError("max_input_essays must be greater than zero")
    if not args.train_csv.is_file():
        raise FileNotFoundError(f"Training CSV not found: {args.train_csv}")
    if not args.sentence_bank.is_file():
        raise FileNotFoundError(f"Sentence bank not found: {args.sentence_bank}")
    manifest_path = args.output.with_suffix(".manifest.json")
    if (args.output.exists() or manifest_path.exists()) and not args.force:
        raise FileExistsError(
            f"Output already exists: {args.output}. Use --force to replace it."
        )

    dataframe = pd.read_csv(args.train_csv)
    if args.max_input_essays is not None:
        dataframe = dataframe.head(args.max_input_essays).copy()
    text_column = "full_text" if "full_text" in dataframe.columns else "text"
    sentence_bank = [
        line.strip()
        for line in args.sentence_bank.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not sentence_bank:
        raise ValueError("Sentence bank is empty")

    rng = random.Random(args.seed)
    requested = round(len(dataframe) * args.training_fraction)
    selected_indices = sorted(rng.sample(range(len(dataframe)), requested))
    modes = [
        "external" if index % 2 == 0 else "self_duplication"
        for index in range(requested)
    ]
    rng.shuffle(modes)

    records: list[dict[str, Any]] = []
    skipped = Counter()
    iterator = tqdm(
        zip(selected_indices, modes),
        total=requested,
        desc="Preparing Injection pairs",
        unit="essay",
        dynamic_ncols=True,
        disable=True if args.no_progress else None,
    )
    for source_index, mode in iterator:
        row = dataframe.iloc[source_index]
        original_text = str(row[text_column])
        generated = (
            _external_pair(original_text, sentence_bank, rng)
            if mode == "external"
            else _self_duplication_pair(original_text, rng)
        )
        if generated is None:
            skipped[mode] += 1
            continue
        records.append(
            {
                "source_index": source_index,
                "essay_id": (
                    str(row["essay_id"])
                    if "essay_id" in dataframe.columns
                    else None
                ),
                "score": float(row["score"]),
                "injection_mode": mode,
                "original_text": original_text,
                "original_text_sha256": _text_sha256(original_text),
                **generated,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(args.output)

    counts = Counter(record["injection_mode"] for record in records)
    manifest = {
        "protocol": "offline_pre_generated_injection_pairs",
        "generation_uses_victim_scores": False,
        "train_csv": str(args.train_csv.resolve()),
        "train_csv_sha256": _sha256(args.train_csv),
        "sentence_bank": str(args.sentence_bank.resolve()),
        "sentence_bank_sha256": _sha256(args.sentence_bank),
        "sentence_bank_content_sha256": hashlib.sha256(
            "\n".join(sentence_bank).encode("utf-8")
        ).hexdigest(),
        "output": str(args.output),
        "seed": args.seed,
        "n_input_essays": len(dataframe),
        "max_input_essays": args.max_input_essays,
        "training_fraction": args.training_fraction,
        "n_requested_pairs": requested,
        "n_written_pairs": len(records),
        "pair_counts": dict(counts),
        "skipped_counts": dict(skipped),
        "injection_unit": "one_sentence",
        "training_objective": "squared_one_sided_hinge(injected-clean)",
        "evaluation_search_is_separate": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[SAVED] {args.output} ({len(records)} pairs)", flush=True)
    print(f"[SAVED] {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
