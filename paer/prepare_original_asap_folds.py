"""Prepare original ASAP-AES (NOT ASAP 2.0) from the user-supplied labeled TSV.

Splits are within prompt, with identical normalized texts grouped. The raw
domain1_score remains authoritative for QWK; targets on 0..5 are continuous.
"""
import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paer.rhi_experiment_utils import sha256

RANGES = {1:(2,12), 2:(1,6), 3:(0,3), 4:(0,3), 5:(0,4), 6:(0,4), 7:(0,30), 8:(0,60)}


def target(prompt, raw):
    low, high = RANGES[prompt]
    if not low <= raw <= high:
        raise ValueError(f"Score {raw} outside prompt {prompt}'s range [{low}, {high}]")
    return 5*(raw-low)/(high-low)


def prepare(rows, seed):
    groups, ids, seen = defaultdict(lambda: defaultdict(list)), set(), {}
    for row in rows:
        prompt, raw = int(row["essay_set"]), int(row["domain1_score"])
        essay, text = row["essay_id"].strip(), row["essay"]
        if not essay or not text.strip() or essay in ids or prompt not in RANGES:
            raise ValueError(f"Invalid/duplicate essay ID, empty text or invalid prompt: {essay}")
        ids.add(essay)
        h = hashlib.sha256(" ".join(text.split()).encode()).hexdigest()
        if seen.setdefault(h, prompt) != prompt:
            raise ValueError("Identical essay occurs in multiple prompts; inspect before splitting")
        groups[prompt][h].append(dict(essay_id=essay, prompt_id=prompt, full_text=text,
                                      domain1_score=raw, model_target_0_5=target(prompt, raw)))
    if set(groups) != set(RANGES):
        raise ValueError("Expected all eight original ASAP-AES prompts")
    splits = {k: [] for k in ("train","development","test")}
    for prompt, group in sorted(groups.items()):
        keys = sorted(group)
        n = max(1, round(.15*len(keys)))
        if 2*n >= len(keys):
            raise ValueError(f"Too few unique essays at prompt {prompt}")
        random.Random(seed*100+prompt).shuffle(keys)
        for split, chosen in (("test",keys[:n]), ("development",keys[n:2*n]), ("train",keys[2*n:])):
            splits[split].extend(row for k in chosen for row in group[k])
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--encoding", default="cp1252", help="Set to utf-8-sig for a UTF-8 export")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"artifacts/original_asap/folds_seed42")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    with args.source.open(encoding=args.encoding, newline="") as f:
        splits = prepare(csv.DictReader(f, delimiter="\t"), 42)
    manifest = dict(dataset="original ASAP-AES", score_field="domain1_score", ranges=RANGES,
                    seed=42, source_sha256=sha256(args.source), encoding=args.encoding,
                    split="70/15/15 by normalized-text group within prompt", n={k:len(v) for k,v in splits.items()})
    print(json.dumps(manifest, indent=2))
    if args.dry_run:
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest["files"] = {}
    for name, rows in splits.items():
        path = args.output_dir/f"{name}.csv"
        with path.open("x", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        manifest["files"][name] = sha256(path)
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
