#!/usr/bin/env python3
"""Reuse RH traces and generate Injection trajectories in a separate RHI pool.

Balance is by unique essay, not number of accepted edits. Failed Injection
searches remain clean-only; successful RH groups are downsampled to match.
Per-essay atomic shards allow an interrupted generation run to resume safely.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paer.rhi_experiment_utils import (
    bind_directory, checkpoint_identity, read_json, save_json, sha256,
    validate_trace_groups,
)


def load_rows(path):
    import pandas as pd
    frame = pd.read_csv(path)
    text_col = "full_text" if "full_text" in frame else "text"
    if text_col not in frame or "score" not in frame:
        raise ValueError("Training CSV requires full_text/text and score")
    id_col = next((k for k in ("essay_id", "id") if k in frame), None)
    return [{"text": str(row[text_col]), "score": float(row["score"]),
             "essay_id": str(row[id_col]) if id_col else f"train_{index}"}
            for index, row in frame.iterrows()]


def allocate_rhi(rows, rh_records, fraction, seed):
    """Disjoint essay assignments; balance Injection subtypes within its third."""
    if not 0 < fraction <= 1:
        raise ValueError("attack-fraction must be in (0, 1]")
    groups = {}
    for record in rh_records:
        if record["attack"] not in ("rudimentary", "hotflip"):
            raise ValueError("RH source contains an unexpected attack")
        index = int(record["row_index"])
        if index >= len(rows):
            continue  # first-N smoke test
        if index < 0 or record["original_text"] != rows[index]["text"]:
            raise ValueError(f"RH source differs from training CSV row {index}")
        if float(record["label_score_space"]) != rows[index]["score"]:
            raise ValueError(f"RH label differs at row {index}")
        if float(record["step_gain"]) <= 0:
            continue
        group = groups.setdefault(index, [])
        if group and group[0]["attack"] != record["attack"]:
            raise ValueError("RH source assigns multiple attacks to one essay")
        if not any(r["record_id"] == record["record_id"] for r in group):
            group.append(record)
    rng = random.Random(seed)
    pools = {a: sorted(i for i, g in groups.items() if g[0]["attack"] == a)
             for a in ("rudimentary", "hotflip")}
    per_family = min(int(len(rows) * fraction) // 3, *(len(v) for v in pools.values()))
    per_family -= per_family % 2
    if per_family < 2:
        raise ValueError("Too few RH essays; increase smoke --max-essays (try 256)")
    chosen = {}
    for attack, indices in pools.items():
        rng.shuffle(indices)
        chosen.update({i: groups[i] for i in indices[:per_family]})
    available = sorted(set(range(len(rows))) - set(chosen))
    injection_rows = rng.sample(available, per_family)
    jobs = {i: ("injection_external" if n % 2 == 0 else "injection_self_dup")
            for n, i in enumerate(injection_rows)}
    return chosen, jobs


def balance_successful_groups(rh_groups, injection_groups, seed):
    pools = {a: [] for a in ("rudimentary", "hotflip", "injection_external", "injection_self_dup")}
    for group in list(rh_groups.values()) + list(injection_groups.values()):
        if group:
            pools[group[0]["attack"]].append(group)
    half = min(len(pools["rudimentary"]) // 2, len(pools["hotflip"]) // 2,
               len(pools["injection_external"]), len(pools["injection_self_dup"]))
    if half == 0:
        raise ValueError("No balanced successful pool; increase --max-essays or generation budget in a NEW directory")
    rng = random.Random(seed)
    selected = []
    for attack, groups in pools.items():
        groups.sort(key=lambda g: int(g[0]["row_index"]))
        rng.shuffle(groups)
        selected.extend(groups[:half * (2 if attack in ("rudimentary", "hotflip") else 1)])
    return [r for g in sorted(selected, key=lambda g: int(g[0]["row_index"]))
            for r in sorted(g, key=lambda r: int(r["accepted_edit_order"]))]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-csv", type=Path, default=ROOT / "data/train_fold0.csv")
    p.add_argument("--rh-traces", type=Path, default=ROOT / "artifacts/paer/rh_counterfactual_training_traces_seed42.jsonl")
    p.add_argument("--checkpoint", type=Path, default=ROOT / "deberta_checkpoints/fold0_best")
    p.add_argument("--sentence-bank", type=Path, default=ROOT / "injection/wikipedia_sentences_100.txt")
    p.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/paer/rhi_training_pool_seed42")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attack-fraction", type=float, default=0.5)
    p.add_argument("--max-essays", type=int)
    p.add_argument("--n-steps", type=int, default=3)
    p.add_argument("--max-candidates-per-step", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--device", default="cuda")
    p.add_argument("--online", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if min(args.n_steps, args.max_candidates_per_step, args.batch_size, args.max_length) <= 0:
        raise ValueError("Generation budgets must be positive")
    if args.max_essays is not None and args.max_essays <= 0:
        raise ValueError("max-essays must be positive")
    if args.max_length != 1024:
        raise ValueError("The reused RH scorer protocol is fixed at 1024 tokens")
    rows = load_rows(args.train_csv)
    if args.max_essays:
        rows = rows[:args.max_essays]
    source_manifest = read_json(args.rh_traces.with_suffix(".manifest.json"))
    if int(source_manifest["n_steps"]) != args.n_steps:
        raise ValueError("Use the same n-steps as the reused RH manifest")
    if int(source_manifest["max_candidates_per_step"]) != args.max_candidates_per_step:
        raise ValueError("Candidate budget must match reused RH manifest")
    if Path(source_manifest["checkpoint"]).resolve() != args.checkpoint.resolve():
        raise ValueError("Injection attribution checkpoint must match RH source manifest")
    with args.rh_traces.open(encoding="utf-8") as f:
        source_records = [json.loads(line) for line in f if line.strip()]
    rh_groups, jobs = allocate_rhi(rows, source_records, args.attack_fraction, args.seed)
    protocol = {"protocol": "balanced_rhi_positive_trajectories_v1",
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                              if k not in ("dry_run", "no_progress", "online")},
                "source_hashes": {"train": sha256(args.train_csv), "rh": sha256(args.rh_traces),
                                  "rh_manifest": sha256(args.rh_traces.with_suffix(".manifest.json")),
                                  "bank": sha256(args.sentence_bank)},
                "checkpoint": checkpoint_identity(args.checkpoint),
                "generation_code": {str(p.relative_to(ROOT)): sha256(p) for p in (
                    Path(__file__), ROOT / "paer/rhi_experiment_utils.py",
                    ROOT / "text_scoring_adv_training/evaluation/aes/scorer.py",
                    ROOT / "text_scoring_adv_training/evaluation/aes/attacks/injection.py")},
                "injection_jobs": {str(k): v for k, v in jobs.items()},
                "rh_rows": sorted(rh_groups), "mlm_excluded": True}
    print(json.dumps({"RH_essays": len(rh_groups), "Injection_searches": len(jobs),
                      "output_dir": str(args.output_dir)}, indent=2), flush=True)
    if args.dry_run:
        return 0
    bind_directory(args.output_dir, protocol)
    target = args.output_dir / "rhi_counterfactual_training_traces.jsonl"
    manifest_path = target.with_suffix(".manifest.json")
    if manifest_path.exists():
        if read_json(manifest_path)["trace_sha256"] != sha256(target):
            raise ValueError("Completed pool was modified")
        print(f"[COMPLETE] {target}")
        return 0
    if not args.online:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import torch
    from tqdm.auto import tqdm
    from text_scoring_adv_training.evaluation.aes.scorer import AESScorer
    from text_scoring_adv_training.evaluation.aes.attacks.injection import IterativeInjectionAttack
    scorer = None
    bank = [s.strip() for s in args.sentence_bank.read_text(encoding="utf-8").splitlines() if s.strip()]
    injection_groups = {}
    for index, attack_name in tqdm(sorted(jobs.items()), desc="Preparing RHI Injection traces",
                                   unit="essay", disable=args.no_progress, dynamic_ncols=True):
        shard = args.output_dir / "injection_essay_progress" / f"row_{index}.json"
        if shard.exists():
            injection_groups[index] = read_json(shard)["records"]
            continue
        if scorer is None:
            scorer = AESScorer(args.checkpoint, device=args.device, dtype=torch.float32)
        local_seed = args.seed * 1_000_003 + index
        random.seed(local_seed)
        np.random.seed(local_seed % (2**32 - 1))
        torch.manual_seed(local_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(local_seed)
        attacker = IterativeInjectionAttack(
            scorer, mode="external" if attack_name == "injection_external" else "self_duplication",
            sentence_bank=bank, n_steps=args.n_steps, candidates_per_step=args.max_candidates_per_step,
            batch_size=args.batch_size, threshold=float(source_manifest["success_threshold"]),
            record_intermediate_texts=True)
        _, history = attacker.attack(rows[index]["text"])
        records = []
        for order, entry in enumerate(history, 1):
            records.append({"record_version": 1, "record_id": f"rhi:{index}:{attack_name}:{order}",
                            "essay_id": rows[index]["essay_id"], "row_index": index,
                            "attack": attack_name, "label_score_space": rows[index]["score"],
                            "original_text": rows[index]["text"], "before_text": entry["before_text"],
                            "adversarial_text": entry["after_text"], "step_gain": float(entry["step_gain"]),
                            "cumulative_delta": float(entry["delta"]), "accepted_edit_order": order,
                            "victim_score_before": float(entry["score"] - entry["step_gain"]),
                            "victim_score_after": float(entry["score"]), "attack_step": int(entry["step"]),
                            "attribution_seed": local_seed})
        save_json(shard, {"records": records})
        injection_groups[index] = records
    records = balance_successful_groups(rh_groups, injection_groups, args.seed)
    _, counts = validate_trace_groups(records, rows)
    temporary = target.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(target)
    save_json(manifest_path, {**protocol, "trace_sha256": sha256(target),
                             "essay_counts_by_attack": counts,
                             "trace_counts_by_attack": dict(Counter(r["attack"] for r in records)),
                             "n_clean_rows": len(rows), "n_attacked_rows": sum(counts.values()),
                             "actual_attack_fraction": sum(counts.values()) / len(rows),
                             "supervision": "victim score gain proxy; not human quality or causal attribution",
                             "sampling": "one family per essay; one rotating accepted state per epoch",
                             "RH_checkpoint_identity_limit": "Historical RH manifest contains a path, not a weight hash; verify B0 was not replaced."})
    print(f"[SAVED] {target}\nEssay counts: {counts}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
