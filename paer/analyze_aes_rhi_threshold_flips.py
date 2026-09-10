#!/usr/bin/env python3
"""Cache forward-only PAER replay, then count paired threshold flips on CPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paer.rhi_experiment_utils import bind_directory, read_json, save_json, sha256
from paer.routing_experiment_utils import ATTACK_FILES, DEFAULT_EVALUATION, code_identity, equal_family_average, frozen_reference


def paired_flips(records, scores, threshold=0.1, near_threshold=5e-5):
    """Success uses the existing exact >= rule; tolerance only flags fragility."""
    if not records or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Nonempty pairs and a finite nonnegative threshold required")
    pairs = []
    for index, row in enumerate(records):
        original, adversarial = scores[row["original_text"]], scores[row["perturbed_text"]]
        if not all(math.isfinite(float(v)) for v in (*original, *adversarial)):
            raise ValueError(f"Nonfinite replay score at {index}")
        routed_delta = adversarial[0] - original[0]
        base_delta = adversarial[1] - original[1]
        lift = adversarial[2] - original[2]
        routed_success, base_success = routed_delta >= threshold, base_delta >= threshold
        transition = ("both_success" if routed_success else "routing_prevents_success") if base_success else (
            "routing_induces_success" if routed_success else "both_fail")
        pairs.append({
            "row_index": index, "essay_id": row.get("essay_id"), "transition": transition,
            "original_text_sha256": hashlib.sha256(row["original_text"].encode()).hexdigest(),
            "base_original": original[1], "base_adversarial": adversarial[1],
            "routed_original": original[0], "routed_adversarial": adversarial[0],
            "original_correction": original[2], "adversarial_correction": adversarial[2],
            "base_delta": base_delta, "routed_delta": routed_delta, "correction_lift": lift,
            "base_success": base_success, "routed_success": routed_success,
            "base_margin": base_delta - threshold, "routed_margin": routed_delta - threshold,
            "near_threshold": min(abs(base_delta - threshold), abs(routed_delta - threshold)) <= near_threshold,
            "identity_abs_error": abs(base_delta - routed_delta - lift),
            "replay_abs_error": max(abs(original[0] - row["original_score"]),
                                    abs(adversarial[0] - row["perturbed_score"])),
        })
    counts = Counter(p["transition"] for p in pairs)
    n = len(pairs)
    mean = lambda key: sum(p[key] for p in pairs) / n
    summary = {
        "n_pairs": n, "success_threshold": threshold,
        "transition_counts": {k: counts[k] for k in
                              ("both_fail", "both_success", "routing_prevents_success", "routing_induces_success")},
        "base_branch_fixed_set_asr": mean("base_success"), "routed_fixed_set_asr": mean("routed_success"),
        "fixed_set_asr_reduction_from_routing": (counts["routing_prevents_success"] - counts["routing_induces_success"]) / n,
        "mean_correction_lift": mean("correction_lift"),
        "negative_correction_lift_count": sum(p["correction_lift"] < 0 for p in pairs),
        "near_threshold_count": sum(p["near_threshold"] for p in pairs),
        "near_threshold_flip_count": sum(p["near_threshold"] and p["base_success"] != p["routed_success"] for p in pairs),
        "replay_max_abs_error": max(p["replay_abs_error"] for p in pairs),
        "identity_max_abs_error": max(p["identity_abs_error"] for p in pairs),
    }
    return summary, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/aes_paer_rhi_v3_threshold_flips_seed42")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cpu-summary-only", action="store_true", help="Require existing complete replay cache; never import torch")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    reference, checkpoint = frozen_reference(args.evaluation_dir)
    payloads = {attack: read_json(args.evaluation_dir / "paer" / path) for attack, path in ATTACK_FILES.items()}
    texts = []
    alignment = None
    for attack, payload in payloads.items():
        records = payload["details"]
        if payload["summary"]["attack"] != attack or len(records) != reference["n_essays"]:
            raise ValueError(f"Attack/sample mismatch: {attack}")
        if payload["summary"]["success_threshold"] != reference["success_threshold"]:
            raise ValueError(f"Threshold mismatch: {attack}")
        keys = [(r["essay_id"], r["original_text"]) for r in records]
        if alignment is not None and keys != alignment:
            raise ValueError("Attack sets have different essay identities/order")
        alignment = keys
        for row in records:
            texts.extend((row["original_text"], row["perturbed_text"]))
    texts = list(dict.fromkeys(texts))
    protocol = {"diagnostic": "fixed_set_paired_threshold_flips_v1", "reference": reference,
                "inputs": {a: sha256(args.evaluation_dir / "paer" / p) for a, p in ATTACK_FILES.items()},
                "batch_size": args.batch_size, "device": args.device, "dtype": reference["dtype"],
                "near_threshold_epsilon": 5e-5,
                "code": code_identity("paer/analyze_aes_rhi_threshold_flips.py", "paer/routing_experiment_utils.py",
                                      "paer/analyze_aes_paer_routing_contribution.py"),
                "limit": "Fixed adversarial texts; not adaptive route-off evaluation or training ablation"}
    if args.cpu_summary_only and not (args.output_dir / "rhi_run_binding.json").is_file():
        raise FileNotFoundError("No replay cache; run the forward replay first")
    bind_directory(args.output_dir, protocol)
    scores, scorer = {}, None
    for start in range(0, len(texts), 256):
        chunk = texts[start:start + 256]
        shard = args.output_dir / "replay_cache" / f"texts_{start:06d}.json"
        marker = shard.with_suffix(".sha256.json")
        if shard.exists() and marker.exists():
            if sha256(shard) != read_json(marker)["sha256"]:
                raise ValueError(f"Replay cache changed: {shard}")
            cached = read_json(shard)
            if list(cached) != chunk:
                raise ValueError("Replay text order changed")
        else:
            if args.cpu_summary_only:
                raise FileNotFoundError(f"Replay incomplete: {shard}")
            if scorer is None:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                import torch
                from paer.analyze_aes_paer_routing_contribution import AESScorer, _score_unique_texts
                dtype = torch.float32 if reference["dtype"] == "float32" else torch.bfloat16
                scorer = AESScorer(checkpoint, device=args.device, dtype=dtype)
            cached = _score_unique_texts(scorer, chunk, batch_size=args.batch_size,
                                        max_length=reference["max_length"], show_progress=True)
            save_json(shard, cached)
            save_json(marker, {"sha256": sha256(shard)})
        scores.update(cached)
    summaries = []
    for attack, payload in payloads.items():
        summary, pairs = paired_flips(payload["details"], scores, reference["success_threshold"])
        if summary["replay_max_abs_error"] > 1e-4 or summary["identity_max_abs_error"] > 1e-4:
            raise ValueError(f"Replay mismatch for {attack}; inspect checkpoint/precision before interpretation")
        summaries.append({"attack": attack, **summary})
        save_json(args.output_dir / f"{attack}_paired_scores.json", pairs)
    fields = ("base_branch_fixed_set_asr", "routed_fixed_set_asr", "fixed_set_asr_reduction_from_routing", "mean_correction_lift")
    macro = equal_family_average(summaries, fields)
    save_json(args.output_dir / "threshold_flip_summary.json", {"protocol": protocol, "attacks": summaries, "rhi_macro": macro})
    lines = ["# Fixed-set routing threshold flips", "", "Not adaptive attacks. Injection subtypes share one family weight.", "",
             "| Attack | n | Prevented successes | Induced successes | Route-off ASR | Routed ASR | Mean correction lift |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for s in summaries:
        c = s["transition_counts"]
        lines.append(f"| {s['attack']} | {s['n_pairs']} | {c['routing_prevents_success']} | {c['routing_induces_success']} | "
                     f"{s['base_branch_fixed_set_asr']:.4f} | {s['routed_fixed_set_asr']:.4f} | {s['mean_correction_lift']:.6f} |")
    lines += ["", f"RHI macro: route-off={macro['base_branch_fixed_set_asr']:.4f}; routed={macro['routed_fixed_set_asr']:.4f}"]
    (args.output_dir / "threshold_flip_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
