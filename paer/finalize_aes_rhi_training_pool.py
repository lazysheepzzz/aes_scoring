#!/usr/bin/env python3
"""CPU-only finalization of a fully generated, bound RHI pool.

Preserves generation provenance and all source/shard files. Only the validation
utility may differ from the recorded generation code; search code, inputs,
weights and essay assignments must still match. Never generates missing shards.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paer.prepare_aes_rhi_training_traces import allocate_rhi, balance_successful_groups, load_rows
from paer.rhi_experiment_utils import checkpoint_identity, read_json, save_json, sha256, validate_trace_groups


def finalize(output_dir):
    output_dir = Path(output_dir)
    binding_path = output_dir / "rhi_run_binding.json"
    protocol = read_json(binding_path)
    if protocol["protocol"] != "balanced_rhi_positive_trajectories_v1" or protocol["mlm_excluded"] is not True:
        raise ValueError("Unsupported generation protocol")
    args = protocol["arguments"]
    train, rh, bank, checkpoint = (Path(args[k]) for k in
                                  ("train_csv", "rh_traces", "sentence_bank", "checkpoint"))
    sources = {"train": train, "rh": rh, "rh_manifest": rh.with_suffix(".manifest.json"), "bank": bank}
    for name, path in sources.items():
        if sha256(path) != protocol["source_hashes"][name]:
            raise ValueError(f"Generation input changed: {name}")
    if checkpoint_identity(checkpoint) != protocol["checkpoint"]:
        raise ValueError("Generation checkpoint changed")
    checked_code = set()
    for name, expected in protocol["generation_code"].items():
        normalized = name.replace("\\", "/")
        checked_code.add(normalized)
        if normalized == "paer/rhi_experiment_utils.py":
            continue  # Explicit validator repair; old and new hashes retained.
        if sha256(ROOT / normalized) != expected:
            raise ValueError(f"Generation code changed: {name}; refusing shard adoption")
    required = {"paer/prepare_aes_rhi_training_traces.py", "paer/rhi_experiment_utils.py",
                "text_scoring_adv_training/evaluation/aes/scorer.py",
                "text_scoring_adv_training/evaluation/aes/attacks/injection.py"}
    if not required.issubset(checked_code):
        raise ValueError("Incomplete generation code provenance")

    target = output_dir / "rhi_counterfactual_training_traces.jsonl"
    manifest_path = target.with_suffix(".manifest.json")
    if manifest_path.exists():
        if read_json(manifest_path)["trace_sha256"] != sha256(target):
            raise ValueError("Completed pool was modified")
        print(f"[COMPLETE] {target}", flush=True)
        return 0
    if target.exists():
        raise FileExistsError(f"Unmanifested pool exists; preserve it for inspection: {target}")

    rows = load_rows(train)
    if args.get("max_essays"):
        rows = rows[:args["max_essays"]]
    with rh.open(encoding="utf-8") as handle:
        source_records = [json.loads(line) for line in handle if line.strip()]
    rh_groups, jobs = allocate_rhi(rows, source_records, args["attack_fraction"], args["seed"])
    if sorted(rh_groups) != protocol["rh_rows"] or {str(k): v for k, v in jobs.items()} != protocol["injection_jobs"]:
        raise ValueError("Essay assignments changed; refusing shard adoption")
    injection_groups, shard_hashes = {}, {}
    for index, attack in sorted(jobs.items()):
        shard = output_dir / "injection_essay_progress" / f"row_{index}.json"
        if not shard.exists():
            raise FileNotFoundError(f"Generation incomplete; missing shard: {shard}")
        records = read_json(shard)["records"]
        for record in records:
            if (int(record["row_index"]) != index or record["attack"] != attack
                    or record["original_text"] != rows[index]["text"]
                    or float(record["label_score_space"]) != rows[index]["score"]):
                raise ValueError(f"Shard assignment/text/label mismatch: {shard}")
        injection_groups[index] = records
        shard_hashes[shard.name] = sha256(shard)
    records = balance_successful_groups(rh_groups, injection_groups, args["seed"])
    _, counts = validate_trace_groups(records, rows)
    signed_counts = dict(Counter(r["attack"] for r in records if float(r["cumulative_delta"]) <= 0))
    temporary = target.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(target)
    save_json(manifest_path, {
        **protocol, "trace_sha256": sha256(target), "essay_counts_by_attack": counts,
        "trace_counts_by_attack": dict(Counter(r["attack"] for r in records)),
        "n_clean_rows": len(rows), "n_attacked_rows": sum(counts.values()),
        "actual_attack_fraction": sum(counts.values()) / len(rows),
        "supervision": "victim score gain proxy; not human quality or causal attribution",
        "sampling": "one family per essay; one rotating accepted state per epoch",
        "RH_checkpoint_identity_limit": "Historical RH manifest contains a path, not a weight hash; verify B0 was not replaced.",
        "finalization": {
            "mode": "cpu_only_saved_shards", "binding_sha256": sha256(binding_path),
            "shard_sha256": shard_hashes,
            "code": {str(p.relative_to(ROOT)): sha256(p) for p in
                     (Path(__file__), ROOT / "paer/rhi_experiment_utils.py")},
            "nonpositive_cumulative_trace_counts_by_attack": signed_counts,
            "gain_semantics": "step_gain > 0; cumulative_delta is finite and signed; no records or gains rewritten",
        },
    })
    print(f"[SAVED] {target}\nEssay counts: {counts}\n"
          f"Nonpositive cumulative records retained: {signed_counts}\n"
          f"[REUSED] {len(jobs)} Injection shards; no GPU/search invoked", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/paer/rhi_training_pool_seed42")
    return finalize(parser.parse_args().output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
