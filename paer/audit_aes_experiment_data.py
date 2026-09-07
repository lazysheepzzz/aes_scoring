#!/usr/bin/env python3
"""Report split overlaps without moving data or creating a supposed unseen test set."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def inspect_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        text_key = "full_text" if "full_text" in fields else "text"
        if text_key not in fields:
            raise ValueError(f"No text column: {path}")
        hashes = []
        ids = []
        labeled = 0
        for row in reader:
            normalized = " ".join(row[text_key].split())
            hashes.append(hashlib.sha256(normalized.encode("utf-8")).hexdigest())
            ids.append(row.get("essay_id", row.get("id", "")))
            if row.get("score", "").strip():
                labeled += 1
    return {"path": str(path.resolve()), "n": len(hashes), "labeled_rows": labeled,
            "duplicate_text_rows": len(hashes) - len(set(hashes)),
            "nonempty_unique_ids": len(set(ids) - {""})}, set(hashes)


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, default=root / "data/train_fold0.csv")
    p.add_argument("--development", type=Path, default=root / "data/valid_fold0.csv")
    p.add_argument("--test", type=Path)
    args = p.parse_args()
    sources = {"train": args.train, "development": args.development}
    if args.test:
        sources["test"] = args.test
    report, texts = {}, {}
    for name, path in sources.items():
        report[name], texts[name] = inspect_csv(path)
    report["overlaps_normalized_text"] = {
        f"{a}__{b}": len(texts[a] & texts[b]) for a in sources for b in sources if a < b}
    report["independent_test_verified"] = False
    report["note"] = "Disjoint text is necessary but does not establish absence of historical training/development exposure."
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
