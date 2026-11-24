"""Compute original-ASAP clean QWK per prompt from saved continuous 0..5 predictions."""
import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from paer.prepare_original_asap_folds import RANGES, target


def evaluate(rows, predictions):
    from sklearn.metrics import cohen_kappa_score
    grouped = defaultdict(list)
    ids = set()
    for row in rows:
        i, p, raw = row["essay_id"], int(row["prompt_id"]), int(row["domain1_score"])
        if i in ids or not math.isclose(target(p,raw), float(row["model_target_0_5"]), abs_tol=1e-7):
            raise ValueError(f"Duplicate ID or inconsistent label: {i}")
        ids.add(i)
        value = predictions[i]
        if not math.isfinite(value):
            raise ValueError(f"Invalid prediction: {i}")
        lo,hi = RANGES[p]
        predicted_grade = round(lo + min(5.,max(0.,value))*(hi-lo)/5)
        grouped[p].append((raw, predicted_grade))
    if ids != predictions.keys() or set(grouped) != set(RANGES):
        raise ValueError("Prediction IDs or prompt coverage differ from the fold")
    results = {}
    for p, pairs in sorted(grouped.items()):
        gold, pred = zip(*pairs)
        lo, hi = RANGES[p]
        q = float(cohen_kappa_score(gold, pred, labels=list(range(lo,hi+1)), weights="quadratic"))
        if not math.isfinite(q):
            raise ValueError(f"Undefined QWK for prompt {p}")
        results[p] = q
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True, help="CSV: essay_id,prediction_0_5")
    args = ap.parse_args()
    preds = {}
    with args.predictions.open(encoding="utf-8-sig",newline="") as f:
        for r in csv.DictReader(f):
            if r["essay_id"] in preds:
                raise ValueError("Duplicate prediction")
            preds[r["essay_id"]] = float(r["prediction_0_5"])
    with args.fold.open(encoding="utf-8-sig",newline="") as f:
        scores = evaluate(csv.DictReader(f), preds)
    for p, q in scores.items():
        print(f"prompt {p}: {q:.4f}")
    print(f"Equal-prompt macro QWK: {mean(scores.values()):.4f}")


if __name__ == "__main__":
    main()
