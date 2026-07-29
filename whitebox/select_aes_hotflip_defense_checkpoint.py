#!/usr/bin/env python3
"""Select a D_HOTFLIP checkpoint using clean QWK and subset HotFlip ASR.

The selection protocol follows plan.md:

1. Build or reuse a fixed 256-essay subset stratified by prompt and score.
2. Evaluate every saved D_HOTFLIP checkpoint on the full clean benchmark.
3. Keep checkpoints whose QWK is at least C0 QWK minus 0.02.
4. Run a 10-step HotFlip attack on the fixed subset for eligible checkpoints.
5. Select the lowest-ASR checkpoint, breaking ties by higher clean QWK.

Per-checkpoint results are reusable, so an interrupted selection run can be
started again without repeating completed evaluations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ENTRYPOINT = REPO_ROOT / "whitebox" / "evaluate_aes_checkpoint.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_sort_key(path: Path) -> tuple[int, int, str]:
    if path.name.startswith("gstep") and path.name[5:].isdigit():
        return (0, int(path.name[5:]), path.name)
    if path.name == "best":
        return (1, 0, path.name)
    if path.name == "final":
        return (2, 0, path.name)
    return (3, 0, path.name)


def discover_checkpoint_candidates(output_dir: Path) -> list[Path]:
    """Return saved model directories in deterministic training order."""
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Defense output directory not found: {output_dir}")
    candidates = [
        child
        for child in output_dir.iterdir()
        if child.is_dir()
        and (child / "model.safetensors").is_file()
        and (
            child.name in {"best", "final"}
            or (
                child.name.startswith("gstep")
                and child.name[5:].isdigit()
            )
        )
    ]
    candidates.sort(key=_checkpoint_sort_key)
    if not candidates:
        raise FileNotFoundError(
            f"No gstep*/best/final checkpoints found in {output_dir}"
        )
    return candidates


def deduplicate_checkpoints(
    candidates: Iterable[Path],
) -> tuple[list[Path], dict[str, str]]:
    """Avoid evaluating byte-identical model checkpoints more than once."""
    unique: list[Path] = []
    first_by_hash: dict[str, Path] = {}
    duplicates: dict[str, str] = {}
    for candidate in candidates:
        model_hash = _sha256(candidate / "model.safetensors")
        canonical = first_by_hash.get(model_hash)
        if canonical is None:
            first_by_hash[model_hash] = candidate
            unique.append(candidate)
        else:
            duplicates[candidate.name] = canonical.name
    return unique, duplicates


def stratified_sample_indices(
    dataframe,
    *,
    stratify_columns: list[str],
    sample_size: int,
    seed: int,
) -> list[int]:
    """Sample exact-size strata without sklearn's two-members restriction.

    When the sample can cover every stratum, one row is reserved for each
    stratum (including singleton strata). Remaining slots are allocated
    proportionally to each stratum's remaining capacity using largest
    remainders. Rows are then sampled deterministically within each stratum.
    """
    import numpy as np

    grouped_indices: dict[tuple[str, ...], list[int]] = {}
    for index, row in dataframe.iterrows():
        key = tuple(str(row[column]) for column in stratify_columns)
        grouped_indices.setdefault(key, []).append(int(index))

    groups = sorted(grouped_indices.items(), key=lambda item: item[0])
    if not groups:
        raise ValueError("Cannot stratify an empty dataframe")
    if not 0 < sample_size <= len(dataframe):
        raise ValueError(
            f"sample_size must be in [1, {len(dataframe)}], got {sample_size}"
        )

    cover_every_stratum = sample_size >= len(groups)
    allocations = [1 if cover_every_stratum else 0 for _ in groups]
    remaining_slots = sample_size - sum(allocations)
    capacities = [
        len(indices) - allocation
        for allocation, (_, indices) in zip(allocations, groups)
    ]

    while remaining_slots > 0:
        total_capacity = sum(capacities)
        if total_capacity <= 0:
            raise RuntimeError("Stratified allocation ran out of capacity")
        quotas = [
            remaining_slots * capacity / total_capacity
            for capacity in capacities
        ]
        floor_additions = [
            min(capacity, int(quota))
            for capacity, quota in zip(capacities, quotas)
        ]
        added = sum(floor_additions)
        for index, addition in enumerate(floor_additions):
            allocations[index] += addition
            capacities[index] -= addition
        remaining_slots -= added
        if remaining_slots == 0:
            break

        ranked = sorted(
            range(len(groups)),
            key=lambda index: (
                -(quotas[index] - int(quotas[index])),
                groups[index][0],
            ),
        )
        for index in ranked:
            if remaining_slots == 0:
                break
            if capacities[index] <= 0:
                continue
            allocations[index] += 1
            capacities[index] -= 1
            remaining_slots -= 1

    generator = np.random.default_rng(seed)
    selected: list[int] = []
    for allocation, (_, indices) in zip(allocations, groups):
        if allocation == 0:
            continue
        choices = generator.choice(indices, size=allocation, replace=False)
        selected.extend(int(value) for value in choices)
    return sorted(selected)


def create_or_load_debug_subset(
    valid_csv: Path,
    subset_ids_path: Path,
    subset_csv_path: Path,
    *,
    subset_size: int,
    seed: int,
) -> dict[str, Any]:
    """Create or reuse the fixed prompt+score stratified debugging subset."""
    import pandas as pd

    dataframe = pd.read_csv(valid_csv)
    required = {"essay_id", "prompt_name", "score"}
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(
            f"Validation CSV is missing subset columns: {', '.join(missing)}"
        )
    if not 0 < subset_size <= len(dataframe):
        raise ValueError(
            f"subset_size must be in [1, {len(dataframe)}], got {subset_size}"
        )

    dataframe = dataframe.copy()
    dataframe["essay_id"] = dataframe["essay_id"].astype(str)
    if dataframe["essay_id"].duplicated().any():
        raise ValueError("essay_id must be unique in the validation CSV")

    if subset_ids_path.is_file():
        stored = json.loads(subset_ids_path.read_text(encoding="utf-8"))
        essay_ids = stored["essay_ids"] if isinstance(stored, dict) else stored
        essay_ids = [str(value) for value in essay_ids]
        if len(essay_ids) != subset_size:
            raise ValueError(
                f"Stored subset has {len(essay_ids)} IDs, expected {subset_size}: "
                f"{subset_ids_path}"
            )
        row_by_id = dataframe.set_index("essay_id", drop=False)
        missing_ids = sorted(set(essay_ids) - set(row_by_id.index))
        if missing_ids:
            raise ValueError(
                f"Stored subset contains IDs absent from validation CSV: "
                f"{missing_ids[:5]}"
            )
        subset = row_by_id.loc[essay_ids].reset_index(drop=True)
        metadata = (
            stored
            if isinstance(stored, dict)
            else {
                "seed": seed,
                "subset_size": subset_size,
                "stratify_columns": ["prompt_name", "score"],
                "essay_ids": essay_ids,
            }
        )
    else:
        selected_indices = stratified_sample_indices(
            dataframe,
            stratify_columns=["prompt_name", "score"],
            sample_size=subset_size,
            seed=seed,
        )
        subset = dataframe.loc[selected_indices].reset_index(drop=True)
        essay_ids = subset["essay_id"].tolist()
        metadata = {
            "source_csv": str(valid_csv.resolve()),
            "source_sha256": _sha256(valid_csv),
            "seed": seed,
            "subset_size": subset_size,
            "stratify_columns": ["prompt_name", "score"],
            "essay_ids": essay_ids,
        }
        subset_ids_path.parent.mkdir(parents=True, exist_ok=True)
        subset_ids_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    subset_csv_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(subset_csv_path, index=False)
    return metadata


def select_best_checkpoint(
    rows: list[dict[str, Any]],
    *,
    minimum_qwk: float,
) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if row["clean_qwk"] >= minimum_qwk and row.get("subset_asr") is not None
    ]
    if not eligible:
        raise RuntimeError(
            "No checkpoint passed the clean QWK gate and completed subset attack"
        )
    return min(
        eligible,
        key=lambda row: (
            row["subset_asr"],
            -row["clean_qwk"],
            row["checkpoint_name"],
        ),
    )


def _load_clean_qwk(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload.get("rounded_qwk", payload["qwk"]))


def _load_subset_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(f"Expected one attack summary in {path}")
    return payload[0]


def _format_command(command: list[str]) -> str:
    return (
        subprocess.list2cmdline(command)
        if os.name == "nt"
        else " ".join(command)
    )


def _run(command: list[str], *, dry_run: bool) -> None:
    print(f"[RUN] {_format_command(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def _base_evaluation_command(
    *,
    checkpoint: Path,
    valid_csv: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        str(EVALUATION_ENTRYPOINT),
        "--checkpoint",
        str(checkpoint.resolve()),
        "--valid",
        str(valid_csv.resolve()),
        "--out",
        str(out_dir.resolve()),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--current-python",
    ]
    if args.no_progress:
        command.append("--no-progress")
    return command


def _evaluate_clean_if_needed(
    *,
    checkpoint: Path,
    valid_csv: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    result_path = out_dir / "clean_qwk.json"
    if result_path.is_file() and not args.force:
        print(f"[REUSE] {result_path}", flush=True)
        return
    command = _base_evaluation_command(
        checkpoint=checkpoint,
        valid_csv=valid_csv,
        out_dir=out_dir,
        args=args,
    )
    command.append("--skip-attack")
    _run(command, dry_run=args.dry_run)


def _evaluate_subset_if_needed(
    *,
    checkpoint: Path,
    subset_csv: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    result_path = out_dir / "asr_summary.json"
    if result_path.is_file() and not args.force:
        print(f"[REUSE] {result_path}", flush=True)
        return
    command = _base_evaluation_command(
        checkpoint=checkpoint,
        valid_csv=subset_csv,
        out_dir=out_dir,
        args=args,
    )
    command.extend(
        [
            "--skip-clean",
            "--n-essays",
            str(args.subset_size),
            "--success-threshold",
            str(args.success_threshold),
            "--n-steps",
            str(args.selection_steps),
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
        ]
    )
    _run(command, dry_run=args.dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select D_HOTFLIP by clean QWK gate and subset HotFlip ASR."
    )
    parser.add_argument(
        "--defense-output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "aes_hotflip_defense_seed42",
    )
    parser.add_argument(
        "--c0-checkpoint",
        type=Path,
        default=(
            REPO_ROOT
            / "outputs"
            / "aes_clean_continuation_seed42"
            / "best"
        ),
    )
    parser.add_argument(
        "--valid-csv",
        type=Path,
        default=REPO_ROOT / "data" / "valid_fold0.csv",
    )
    parser.add_argument(
        "--selection-output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "outputs"
            / "aes_hotflip_checkpoint_selection_seed42"
        ),
    )
    parser.add_argument(
        "--subset-ids-path",
        type=Path,
        default=REPO_ROOT / "artifacts" / "data" / "debug_subset_ids.json",
    )
    parser.add_argument("--subset-size", type=int, default=256)
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=42,
        help="Fixed seed used only to construct the stratified subset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by checkpoint attack evaluation.",
    )
    parser.add_argument("--qwk-tolerance", type=float, default=0.02)
    parser.add_argument("--selection-steps", type=int, default=10)
    parser.add_argument("--success-threshold", type=float, default=0.1)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--n-sample-pos", type=int, default=8)
    parser.add_argument("--top-k-per-pos", type=int, default=2)
    parser.add_argument("--max-candidates-per-step", type=int, default=16)
    parser.add_argument("--max-token-edit-rate", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
    )
    parser.add_argument(
        "--keep-duplicate-checkpoints",
        action="store_true",
        help="Evaluate byte-identical best/gstep/final models separately.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute results that already exist in the selection directory.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars in clean and subset evaluations.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive_fields = (
        "subset_size",
        "selection_steps",
        "beam_size",
        "n_sample_pos",
        "top_k_per_pos",
        "max_candidates_per_step",
        "batch_size",
    )
    for field in positive_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"{field} must be greater than zero")
    if args.qwk_tolerance < 0:
        raise ValueError("qwk_tolerance must be non-negative")
    if args.success_threshold < 0:
        raise ValueError("success_threshold must be non-negative")
    if not 0 < args.max_token_edit_rate <= 1:
        raise ValueError("max_token_edit_rate must be in (0, 1]")


def main() -> int:
    args = build_parser().parse_args()
    _validate_args(args)

    if not args.c0_checkpoint.is_dir():
        raise FileNotFoundError(f"C0 checkpoint not found: {args.c0_checkpoint}")
    if not args.valid_csv.is_file():
        raise FileNotFoundError(f"Validation CSV not found: {args.valid_csv}")

    candidates = discover_checkpoint_candidates(args.defense_output_dir)
    duplicates: dict[str, str] = {}
    if args.keep_duplicate_checkpoints:
        unique_candidates = candidates
    else:
        unique_candidates, duplicates = deduplicate_checkpoints(candidates)

    print(
        f"[CANDIDATES] {len(candidates)} saved, "
        f"{len(unique_candidates)} unique model weights",
        flush=True,
    )
    for candidate in candidates:
        duplicate_note = (
            f" (same weights as {duplicates[candidate.name]})"
            if candidate.name in duplicates
            else ""
        )
        print(f"  - {candidate.name}{duplicate_note}", flush=True)

    subset_csv = (
        args.selection_output_dir
        / f"debug_subset_{args.subset_size}.csv"
    )
    if args.dry_run:
        print(
            f"[DRY-RUN] subset IDs: {args.subset_ids_path}\n"
            f"[DRY-RUN] subset CSV: {subset_csv}"
        )
    else:
        args.selection_output_dir.mkdir(parents=True, exist_ok=True)
        metadata = create_or_load_debug_subset(
            args.valid_csv,
            args.subset_ids_path,
            subset_csv,
            subset_size=args.subset_size,
            seed=args.subset_seed,
        )
        print(
            f"[SUBSET] {len(metadata['essay_ids'])} fixed essays -> {subset_csv}",
            flush=True,
        )

    c0_out = args.selection_output_dir / "c0_reference"
    _evaluate_clean_if_needed(
        checkpoint=args.c0_checkpoint,
        valid_csv=args.valid_csv,
        out_dir=c0_out,
        args=args,
    )
    if args.dry_run:
        for checkpoint in unique_candidates:
            candidate_out = args.selection_output_dir / "candidates" / checkpoint.name
            _evaluate_clean_if_needed(
                checkpoint=checkpoint,
                valid_csv=args.valid_csv,
                out_dir=candidate_out,
                args=args,
            )
            _evaluate_subset_if_needed(
                checkpoint=checkpoint,
                subset_csv=subset_csv,
                out_dir=candidate_out,
                args=args,
            )
        return 0

    c0_qwk = _load_clean_qwk(c0_out / "clean_qwk.json")
    minimum_qwk = c0_qwk - args.qwk_tolerance
    print(
        f"[QWK GATE] C0={c0_qwk:.4f}; minimum={minimum_qwk:.4f}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    for checkpoint in unique_candidates:
        candidate_out = args.selection_output_dir / "candidates" / checkpoint.name
        _evaluate_clean_if_needed(
            checkpoint=checkpoint,
            valid_csv=args.valid_csv,
            out_dir=candidate_out,
            args=args,
        )
        clean_qwk = _load_clean_qwk(candidate_out / "clean_qwk.json")
        eligible = clean_qwk >= minimum_qwk
        row: dict[str, Any] = {
            "checkpoint_name": checkpoint.name,
            "checkpoint_path": str(checkpoint.resolve()),
            "clean_qwk": clean_qwk,
            "minimum_qwk": minimum_qwk,
            "eligible": eligible,
            "subset_asr": None,
            "subset_avg_delta": None,
        }
        if eligible:
            _evaluate_subset_if_needed(
                checkpoint=checkpoint,
                subset_csv=subset_csv,
                out_dir=candidate_out,
                args=args,
            )
            subset_summary = _load_subset_summary(
                candidate_out / "asr_summary.json"
            )
            row["subset_asr"] = float(subset_summary["asr"])
            row["subset_avg_delta"] = float(subset_summary["avg_delta"])
        else:
            print(
                f"[SKIP ATTACK] {checkpoint.name}: QWK={clean_qwk:.4f} "
                f"< {minimum_qwk:.4f}",
                flush=True,
            )
        rows.append(row)

    selected = select_best_checkpoint(rows, minimum_qwk=minimum_qwk)
    equivalent_names = [
        name
        for name, canonical_name in duplicates.items()
        if canonical_name == selected["checkpoint_name"]
    ]
    selection = {
        "protocol": {
            "c0_qwk": c0_qwk,
            "qwk_tolerance": args.qwk_tolerance,
            "minimum_qwk": minimum_qwk,
            "subset_size": args.subset_size,
            "subset_seed": args.subset_seed,
            "attack_seed": args.seed,
            "selection_hotflip_steps": args.selection_steps,
            "success_threshold": args.success_threshold,
        },
        "selected_checkpoint": selected,
        "equivalent_duplicate_names": equivalent_names,
        "duplicate_checkpoints": duplicates,
        "candidates": rows,
    }
    summary_path = args.selection_output_dir / "checkpoint_selection_summary.json"
    best_path = args.selection_output_dir / "best_checkpoint.json"
    summary_path.write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    best_path.write_text(
        json.dumps(
            {
                "checkpoint_name": selected["checkpoint_name"],
                "checkpoint_path": selected["checkpoint_path"],
                "clean_qwk": selected["clean_qwk"],
                "subset_asr": selected["subset_asr"],
                "equivalent_duplicate_names": equivalent_names,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"[SELECTED] {selected['checkpoint_name']} "
        f"QWK={selected['clean_qwk']:.4f} "
        f"subset ASR={selected['subset_asr']:.4f}",
        flush=True,
    )
    print(f"[SAVED] {best_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
