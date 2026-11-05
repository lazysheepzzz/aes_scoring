#!/usr/bin/env python3
"""Inventory uniform30 results and analyze saved AES evidence on CPU only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paer.rhi_experiment_utils import checkpoint_identity, read_json, sha256
from paer.routing_experiment_utils import ATTACK_FILES
from paer.select_aes_rhi_uniform30_checkpoints import TARGETS

GROUPS = {group: [k for k, rel in ATTACK_FILES.items() if Path(rel).parent.name == group]
          for group in ("rudimentary", "hotflip", "injection_family")}
GROUPS["mlm_guided"] = ["mlm_guided"]
MIXED, PAER = "mixed_at_rhi", "paer_rhi_v3"
THRESHOLD = 0.1
READ_ERRORS = (OSError, ValueError, KeyError, TypeError, AttributeError)
SHARED_TRAINING = (
    "num_epochs", "per_device_train_batch_size", "gradient_accumulation_steps",
    "learning_rate", "weight_decay", "warmup_ratio", "max_length", "seed", "precision",
    "eval_every", "save_every", "adam_beta1", "adam_beta2", "adam_epsilon",
    "clean_loss_weight", "adversarial_loss_weight", "inflation_tolerance",
    "relative_loss_power", "max_train_samples", "max_valid_samples", "training_attacks",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolved(path, project):
    return (project / path).resolve()


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_data(path):
    rows = {}
    semantic = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for r in csv.DictReader(stream):
            identifier = r.get("essay_id", r.get("id"))
            body = r.get("full_text", r.get("text", r.get("essay")))
            label = float(r["score"])
            require(identifier and identifier not in rows and body and label in range(1, 7),
                    "Expected unique essay IDs, text and integer scores 1–6")
            rows[identifier] = (text_hash(body), label)
            semantic.append([body, r["score"]])
    require(rows, "Empty development CSV")
    return rows, text_hash(json.dumps(semantic, ensure_ascii=False, separators=(",", ":")))


def read_selection(directory, data_hash, project):
    path = directory / "best_checkpoint.json"
    chosen = read_json(path)
    binding = read_json(directory / "rhi_run_binding.json")
    args = binding["arguments"]
    require(binding["protocol"] == "rhi_selection_v1" and
            all(args[k] == 30 for k in ("selection_steps", "rudimentary_selection_steps",
                                         "injection_selection_steps")) and
            binding["mlm_used_for_selection"] is False and
            chosen["mlm_used_for_selection"] is False, "Not an RHI-only uniform30 selection")
    require(binding["valid_sha256"] == chosen["selection_csv_sha256"] == data_hash,
            "Development CSV does not match selection byte hash")
    checkpoint = resolved(chosen["checkpoint_path"], project)
    name = checkpoint.name
    require(name == chosen["checkpoint_name"] and re.fullmatch(r"gstep\d+", name) and
            checkpoint.parent == resolved(args["defense_output_dir"], project),
            "Checkpoint path/name does not match the selection")
    files = binding["candidate_weights"][name]
    require(files, "Selection has no recorded checkpoint identity")
    weights_checked = checkpoint.is_dir()
    if weights_checked:
        require(checkpoint_identity(checkpoint) == files, "Selected checkpoint files changed")
    subset = directory / f"selection_subset_{args['subset_size']}.csv"
    config = checkpoint.parent / "launcher_config.json"
    return {"path": str(path), "sha256": sha256(path),
            "binding_sha256": sha256(directory / "rhi_run_binding.json"),
            "checkpoint": str(checkpoint), "clean_qwk_at_selection": chosen["clean_qwk"],
            "files": files, "weights_checked": weights_checked,
            "subset_sha256": sha256(subset) if subset.is_file() else None,
            "launcher_config_sha256": sha256(config) if config.is_file() else None,
            "binding": binding}


def verified_marker(directory, name, required):
    marker = directory / name
    if not marker.is_file():
        return None
    hashes = read_json(marker)
    require(required <= hashes.keys(), f"Incomplete completion marker: {marker}")
    for member, digest in hashes.items():
        require(Path(member).name == member and member not in (".", "..") and
                sha256(directory / member) == digest, f"Unsafe path or changed artifact: {member}")
    return sha256(marker)


def run_binding(directory, outputs, selection, data_hash, n, seed):
    for parent in [directory, *directory.parents]:
        if not parent.is_relative_to(outputs):
            break
        path = parent / "rhi_run_binding.json"
        if not path.is_file():
            continue
        binding = read_json(path)
        if binding.get("protocol") != "frozen_checkpoint_evaluation_v1":
            continue
        require(binding["evaluation_role"] == "development" and binding["smoke"] is False and
                (binding["data_sha256"], binding["n_essays"], binding["seed"], binding["steps"],
                 binding["success_threshold"], binding["max_length"], binding["dtype"]) ==
                (data_hash, n, seed, 30, THRESHOLD, 1024, "float32"), "Incompatible evaluation binding")
        # A historical selection is reusable when the checkpoint identity is unchanged.
        require(any(c["path"] == selection["checkpoint"] and c["files"] == selection["files"]
                    for c in binding["checkpoints"].values()), "Evaluation weights do not match selection")
        if directory.name == "injection_family":
            require(binding["bank_sha256"] == selection["binding"]["bank_sha256"],
                    "Injection evaluation binding bank differs from selection")
        return {"sha256": sha256(path), "code_hashes": binding["code_hashes"]}
    return None


def read_attack(path, attack, dataset):
    artifact = read_json(path)
    summary, rows = artifact["summary"], artifact["details"]
    require(summary["attack"] == attack and summary["success_threshold"] == THRESHOLD and
            summary["n_essays"] == len(rows) == len(dataset), f"Incorrect summary/count: {path}")
    out = {}
    missing = 0
    for row in rows:
        essay = str(row["essay_id"])
        require(essay not in out and dataset.get(essay) ==
                (text_hash(row["original_text"]), float(row["true_score"])),
                f"Missing, duplicate or changed essay ID/text/label: {essay}")
        before, after = float(row["original_score"]), float(row["perturbed_score"])
        delta = after - before
        require(all(map(math.isfinite, (before, after, delta))) and
                math.isclose(delta, float(row["delta"]), abs_tol=1e-8) and
                row["success"] is (delta >= THRESHOLD), f"Inconsistent score or success: {essay}")
        missing += not bool(row.get("perturbed_text"))
        out[essay] = (before, delta)
    asr = mean(d >= THRESHOLD for _, d in out.values())
    avg = mean(d for _, d in out.values())
    for field, value in (("asr", asr), ("avg_delta", avg)):
        require(math.isclose(value, summary[field], abs_tol=5.1e-5),
                f"Recomputed {field} does not match saved summary: {path}")
    return out, {"asr": asr, "avg_delta": avg, "missing_perturbed_text": missing,
                 "details_sha256": sha256(path)}


def check_run(path, manifest, selection, dataset, csv_path, data_hash, outputs, project, seed):
    group = manifest["attack"]
    params = manifest.get("attack_parameters", manifest.get("hotflip") if group == "hotflip" else None)
    require(isinstance(params, dict) and params.get("n_steps") == 30 and
            params.get("beam_size") == 1 and params.get("max_candidates_per_step") == 16 and
            manifest["success_threshold"] == THRESHOLD and manifest["seed"] == seed and
            manifest["n_essays"] == len(dataset) and resolved(manifest["data"], project) == csv_path,
            "Incompatible or unknown evaluation settings")
    expected = {"max_token_edit_rate": 0.1} if group in ("hotflip", "rudimentary") else {}
    if group in ("hotflip", "mlm_guided"):
        expected.update(n_sample_pos=8, top_k_per_pos=2)
    if group == "mlm_guided":
        expected.update(max_token_edit_rate=0.05, minimum_cosine_similarity=0.9)
    if group == "injection_family":
        expected["sentence_bank_sha256"] = selection["binding"]["bank_sha256"]
    require(all(params.get(k) == v for k, v in expected.items()),
            "Attack-specific recorded settings differ from the audited protocol")
    directory = path.parent
    required = {"run_manifest.json", "clean_qwk.json", "asr_summary.json"}
    required.update(f"{attack}_details.json" for attack in GROUPS[group])
    if group == "injection_family":
        required.add("injection_family_summary.json")
    require(all((directory / f).is_file() for f in required), "Partial result directory")
    marker = verified_marker(directory, "completed_result_hashes.json", required)
    binding = run_binding(directory, outputs, selection, data_hash, len(dataset), seed)
    details, metrics = {}, {}
    for attack in GROUPS[group]:
        details[attack], metrics[attack] = read_attack(directory / f"{attack}_details.json", attack, dataset)
    clean = read_json(directory / "clean_qwk.json")
    require(clean["n"] == len(dataset) and all(math.isfinite(clean[k]) for k in ("qwk", "mae", "rmse")),
            "Invalid clean evaluation summary")
    saved_summary = {r["attack"]: r for r in read_json(directory / "asr_summary.json")}
    for attack, values in metrics.items():
        require(saved_summary[attack]["n_essays"] == len(dataset) and
                all(math.isclose(values[k], saved_summary[attack][k], abs_tol=5.1e-5)
                    for k in ("asr", "avg_delta")), f"ASR summary differs from per-essay details: {attack}")
    if group == "injection_family":
        family = read_json(directory / "injection_family_summary.json")
        require(family["n_essays_per_subattack"] == len(dataset) and
                all(math.isclose(mean(v[k] for v in metrics.values()), family[k], abs_tol=1.1e-4)
                    for k in ("asr", "avg_delta")), "Injection family summary differs from details")
    return ({"path": str(directory), "manifest_sha256": sha256(path), "completion_marker_sha256": marker,
             "evaluation_binding": binding,
             "status": "BOUND" if marker and binding else "LEGACY_NO_FULL_BINDING",
             "attack_parameters": params, "clean_summary": clean,
             "metrics": metrics}, details)


def discover(outputs, project, selections, dataset, csv_path, data_hash, seed):
    found = defaultdict(list)
    rejected, unreadable = [], []
    if not outputs.is_dir():
        return found, rejected, unreadable
    for path in sorted(outputs.rglob("run_manifest.json")):
        try:
            manifest = read_json(path)
            group = manifest.get("attack")
            if group not in GROUPS:
                continue
            victim = resolved(manifest["victim"], project)
        except READ_ERRORS as exc:
            unreadable.append({"path": str(path), "reason": str(exc)})
            continue
        for model, selection in selections.items():
            if victim != Path(selection["checkpoint"]):
                continue
            if any(p in ("candidates", "c0_reference") for p in path.relative_to(outputs).parts):
                continue  # Selection-subset artifacts are not full evaluations.
            try:
                found[model, group].append(check_run(path, manifest, selection, dataset, csv_path,
                                                     data_hash, outputs, project, seed))
            except READ_ERRORS as exc:
                rejected.append({"model": model, "attack": group, "path": str(path), "reason": str(exc)})
    return found, rejected, unreadable


def choose(candidates):
    if not candidates:
        return "NOT_FOUND_OR_NOT_VERIFIED", None
    # Prefer fully bound evidence, but never pick between conflicting reruns.
    if any(meta["status"] == "BOUND" for meta, _ in candidates):
        candidates = [r for r in candidates if r[0]["status"] == "BOUND"]
    fingerprints = {tuple(item["details_sha256"] for item in meta["metrics"].values())
                    for meta, _ in candidates}
    if len(fingerprints) > 1:
        return "AMBIGUOUS", None
    return candidates[0][0]["status"], candidates[0]


def compare_pair(selections, selected, dataset, allow_legacy, n_bootstrap, stat_seed):
    models = (MIXED, PAER)
    groups = list(GROUPS)[:3]
    require(all((model, group) in selected for model in models for group in groups),
            "Need full R/H/I details for both selected checkpoints")
    bindings = [selections[m]["binding"] for m in models]
    keys = ("checkpoint", "train_sha256", "valid_sha256", "trace_sha256", "essay_counts_by_attack")
    require(all(bindings[0]["training_inputs"][k] == bindings[1]["training_inputs"][k] for k in keys),
            "Training provenance differs")
    configs = [read_json(Path(selections[m]["checkpoint"]).parent / "launcher_config.json") for m in models]
    require(all(configs[0][k] == configs[1][k] for k in SHARED_TRAINING),
            "Shared training hyperparameters differ")
    require(all(bindings[0][k] == bindings[1][k] for k in
                ("bank_sha256", "c0_weights", "evaluation_source_hashes")),
            "Selection bank, C0 weights or evaluation source hashes differ")
    subset = [selections[m]["subset_sha256"] for m in models]
    require(subset[0] is not None and subset[0] == subset[1], "Selection subset hashes unavailable or different")
    ignored_paths = {"defense_output_dir", "selection_output_dir", "subset_ids_path", "training_inputs"}
    require({k: v for k, v in bindings[0]["arguments"].items() if k not in ignored_paths} ==
            {k: v for k, v in bindings[1]["arguments"].items() if k not in ignored_paths},
            "Selection arguments differ after excluding only per-model paths")
    # MLM is outside the primary RHI analysis and does not gate its completeness.
    legacy = []
    clean = {}
    deltas = {}
    for group in groups:
        (meta_m, rows_m), (meta_p, rows_p) = (selected[m, group] for m in models)
        excluded = {"sentence_bank"}  # Content hash, not the bank path, establishes identity.
        require({k: v for k, v in meta_m["attack_parameters"].items() if k not in excluded} ==
                {k: v for k, v in meta_p["attack_parameters"].items() if k not in excluded},
                f"Recorded attack parameters differ: {group}")
        bound = [meta_m["evaluation_binding"], meta_p["evaluation_binding"]]
        if all(bound):
            require(bound[0]["code_hashes"] == bound[1]["code_hashes"],
                    f"Recorded evaluation code hashes differ: {group}")
        for m, meta, rows in zip(models, (meta_m, meta_p), (rows_m, rows_p)):
            if meta["status"] != "BOUND":
                legacy.append(meta["path"])
            for attack, values in rows.items():
                require(meta["metrics"][attack]["missing_perturbed_text"] == 0,
                        f"Missing perturbed text; cannot distinguish attack errors: {m}/{attack}")
                before = [values[i][0] for i in dataset]
                if m in clean:
                    require(max(abs(a - b) for a, b in zip(before, clean[m])) <= 1e-4,
                            f"Clean original scores differ across attack runs: {m}/{attack}")
                else:
                    clean[m] = before
        for attack in rows_m:
            deltas[attack] = [[rows[i][1] for i in dataset] for rows in (rows_m[attack], rows_p[attack])]
    require(not legacy or allow_legacy, "Legacy data require explicit --allow-legacy for provisional statistics")
    from paper_results.bootstrap_paer_v3_vs_mixed_at import paired_statistics

    result = paired_statistics([label - 1 for _, label in dataset.values()], [clean[m] for m in models],
                               deltas, threshold=THRESHOLD, n_bootstrap=n_bootstrap, seed=stat_seed)
    result.update(status="PROVISIONAL_LEGACY" if legacy else "BOUND",
                  legacy_sources=legacy, clean_predictions_source="Original scores in Rudimentary details",
                  excluded_optional_groups=[g for g in GROUPS if g not in groups])
    return result


def summarize_replay(directory, source, selection, data_hash):
    from paer import analyze_aes_paer_score_paths as replay

    binding_path = directory / "rhi_run_binding.json"
    binding = read_json(binding_path)
    saved = read_json(directory / "score_path_summary.json")
    ref_path = source / "rhi_run_binding.json"
    require(binding["diagnostic"] == "paer_v3_fixed_set_score_paths_v1" and
            saved["protocol"] == binding and saved["quality_check"]["status"] in
            ("PASS", "PASS_WITH_THRESHOLD_FLIPS") and
            binding["reference_binding_sha256"] == sha256(ref_path), "Replay binding/QC mismatch")
    reference, checkpoint, _, payloads, hashes, _ = replay.load_reference(source)
    require(reference == binding["reference"] and hashes == binding["saved_inputs_sha256"] and
            str(checkpoint) == selection["checkpoint"] and
            reference["checkpoints"]["paer"]["files"] == selection["files"] and
            reference["data_sha256"] == data_hash, "Replay source differs from selected model or split")
    texts = list(dict.fromkeys(r[key] for p in payloads.values() for r in p["details"]
                              for key in ("original_text", "perturbed_text")))
    require(len(texts) == binding["n_unique_texts"], "Replay unique text count differs")
    scores = {}
    cache_hashes = {}
    size = binding["cache_shard_size"]
    require(isinstance(size, int) and size > 0, "Invalid shard size")
    for start in range(0, len(texts), size):
        path = directory / "replay_cache" / f"texts_{start:06d}.json"
        digest = sha256(path)
        require(digest == read_json(path.with_suffix(".sha256.json"))["sha256"], "Replay shard changed")
        cache_hashes[path.name] = digest
        chunk = read_json(path)
        require(list(chunk) == [text_hash(t) for t in texts[start:start + size]], "Replay shard IDs differ")
        require(all(replay.validate_score(v) <= replay.REPLAY_TOLERANCE for v in chunk.values()),
                "Replay score identity failed")
        scores.update(chunk)
    # The tolerance below is numerical only, not a threshold for research value.
    def distribution(values):
        values = sorted(values)
        return {"n": len(values), "mean": mean(values), "median": median(values),
                "p95": values[math.ceil(.95 * len(values)) - 1], "max": values[-1],
                "positive_n_gt_1e_6": sum(x > 1e-6 for x in values)}

    cohorts = {}
    clean = {text_hash(r["original_text"]) for p in payloads.values() for r in p["details"]}
    cohorts["clean_unique_texts"] = {k: distribution([scores[i][k] for i in clean])
                                     for k in ("negative", "positive", "correction")}
    for attack, artifact in payloads.items():
        qc, pairs = replay.decompose_pairs(artifact["details"], scores, reference["success_threshold"])
        require(max(qc["replay_max_abs_error"], qc["identity_max_abs_error"]) <= replay.REPLAY_TOLERANCE,
                f"Recomputed replay QC failed: {attack}")
        cohorts[attack] = {
            "perturbed": {k: distribution([r["perturbed_components"][k] for r in pairs])
                          for k in ("negative", "positive", "correction")},
            "correction_lift": distribution([r["correction_lift"] for r in pairs]),
            "success_disagreements": qc["saved_vs_replay_success_disagreements"],
        }
    return {"status": "CACHE_VERIFIED", "cohorts": cohorts, "cache_sha256": cache_hashes,
            "diagnostic_binding_sha256": sha256(binding_path),
            "scope": "Absolute weighted evidence. Not quality labels or symmetric routing results."}


def trace_target_audit(project, selection):
    config_path = Path(selection["checkpoint"]).parent / "launcher_config.json"
    config = read_json(config_path)
    require(config["max_train_samples"] is None and config.get("max_trace_records") is None,
            "Trace pool is filtered by training settings; full-pool counts are not comparable")
    traces = resolved(config["trace_jsonl"], project)
    digest = sha256(traces)
    require(digest == selection["binding"]["training_inputs"]["trace_sha256"], "Trace pool hash differs")
    model = read_json(Path(selection["checkpoint"]) / "paer_config.json")
    ceiling = float(model["max_token_evidence"]) * float(model["correction_scale"])
    clip = float(config["max_correction_target"])
    require(math.isfinite(ceiling) and ceiling >= 0 and math.isfinite(clip) and clip > 0,
            "Invalid correction or target bound")
    counts = defaultdict(lambda: {"n_states": 0, "nonpositive_gain": 0, "target_above_ceiling": 0})
    with traces.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            gain = float(row["cumulative_delta"])
            require(math.isfinite(gain), "Nonfinite cumulative training gain")
            count = counts[row["attack"]]
            count["n_states"] += 1
            count["nonpositive_gain"] += gain <= 0
            count["target_above_ceiling"] += min(max(gain, 0), clip) > ceiling
    require(counts, "No saved training states")
    return {"status": "TRACE_VERIFIED", "trace_sha256": digest, "absolute_correction_ceiling": ceiling,
            "target_clip_max": clip, "by_attack": counts,
            "scope": "Counts over saved states, not weighted by epoch exposure"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--evaluation-root", type=Path, help="Restrict scan if conflicting runs exist")
    parser.add_argument("--seed", type=int, default=42, help="Training/attack seed for this audit")
    parser.add_argument("--stat-seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--allow-legacy", action="store_true")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--replay-dir", type=Path)
    parser.add_argument("--replay-source", type=Path)
    parser.add_argument("--report", type=Path, help="New JSON report; no overwrite")
    args = parser.parse_args(argv)
    project = args.project_root.resolve()
    outputs = (args.evaluation_root or project / "outputs").resolve()
    csv_path = project / "data/valid_fold0.csv"
    dataset, semantic_hash = load_data(csv_path)
    data_hash = sha256(csv_path)
    selections, unavailable = {}, {}
    for model, (_, pattern) in TARGETS.items():
        path = project / "outputs" / pattern.format(seed=args.seed)
        try:
            selections[model] = read_selection(path, data_hash, project)
        except READ_ERRORS as exc:
            unavailable[model] = {"selection_dir": str(path), "reason": str(exc)}
    found, rejected, unreadable = discover(outputs, project, selections, dataset, csv_path,
                                          data_hash, args.seed)
    selected, coverage = {}, {}
    for model in TARGETS:
        coverage[model] = {}
        for group in GROUPS:
            candidates = found[model, group]
            status, chosen = choose(candidates)
            if chosen:
                selected[model, group] = chosen
            coverage[model][group] = {"status": status if model in selections else "SELECTION_UNAVAILABLE",
                                      "candidates": [meta for meta, _ in candidates]}
    report = {"evaluation_role": "development", "seed": args.seed, "project_root": str(project),
              "evaluation_root": str(outputs), "n_essays": len(dataset), "data_sha256": data_hash,
              "ordered_exact_text_score_sha256": semantic_hash, "selections": selections,
              "selection_unavailable": unavailable, "coverage": coverage,
              "rejected": rejected, "unreadable_manifests": unreadable}
    if not args.inventory_only:
        try:
            report["statistics"] = compare_pair(selections, selected, dataset, args.allow_legacy,
                                                args.n_bootstrap, args.stat_seed)
        except READ_ERRORS as exc:
            report["statistics"] = {"status": "UNAVAILABLE", "reason": str(exc)}
        try:
            report["trace_targets"] = trace_target_audit(project, selections[PAER])
        except READ_ERRORS as exc:
            report["trace_targets"] = {"status": "UNAVAILABLE", "reason": str(exc)}
        directory = args.replay_dir or project / "outputs" / f"aes_paer_rhi_v3_score_path_diagnostic_seed{args.seed}_run01"
        sources = [args.replay_source] if args.replay_source else [
            project / "outputs" / f"aes_rhi_evaluation_development_seed{args.seed}",
            project / "outputs" / f"aes_rhi_uniform30_full_evaluation_seed{args.seed}",
        ]
        failures = []
        for source in sources:
            try:
                report["replay"] = summarize_replay(directory, source, selections[PAER], data_hash)
                break
            except READ_ERRORS as exc:
                failures.append({"source": str(source), "reason": str(exc)})
        else:
            report["replay"] = {"status": "UNAVAILABLE", "diagnostic_dir": str(directory), "checks": failures}
    for model, groups in coverage.items():
        checkpoint = Path(selections[model]["checkpoint"]).name if model in selections else "unavailable"
        print(f"{model} ({checkpoint}): " + ", ".join(f"{g}={v['status']}" for g, v in groups.items()))
    for section in ("statistics", "replay", "trace_targets"):
        if section in report:
            print(f"{section}: {report[section]['status']}")
    stats = report.get("statistics", {})
    if "contrasts" in stats:
        macro = stats["contrasts"]["rhi_macro.asr"]
        print(f"RHI ASR: Mixed={macro['mixed']:.4f}, PAER={macro['paer']:.4f}, "
              f"PAER-Mixed 95% CI={macro['difference_ci95']}")
    for name, values in report.get("replay", {}).get("cohorts", {}).items():
        neg = values.get("perturbed", values)["negative"]
        print(f"Negative evidence {name}: mean={neg['mean']:.6g}, max={neg['max']:.6g}, "
              f"positive={neg['positive_n_gt_1e_6']}/{neg['n']}")
    if args.report:
        # Serialize before opening; reject non-finite values instead of writing invalid JSON.
        encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
        print(f"Report: {args.report.resolve()}")
    else:
        print(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
