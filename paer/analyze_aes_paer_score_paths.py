#!/usr/bin/env python3
"""Decompose PAER-v3 score changes on EXISTING, frozen RHI attack texts.

No attacks, training, API calls, new essays, or quality labels.  Components are
an exact algebraic decomposition of a forward pass, NOT causal attributions.
Only this diagnostic's new, input-bound output directory may be written.
Torch is imported lazily: --dry-run and cached CPU summaries need no GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paer.rhi_experiment_utils import (
    bind_directory, checkpoint_identity, read_json, save_json, sha256,
)
from paer.routing_experiment_utils import ATTACK_FILES, code_identity

DEFAULT_EVALUATION = ROOT / "outputs/aes_rhi_uniform30_full_evaluation_seed42"
DEFAULT_OUTPUT = ROOT / "outputs/aes_paer_rhi_v3_score_path_diagnostic_seed42_run01"
TERMS = ("global_delta", "positive_evidence_delta",
         "negative_evidence_contribution", "correction_contribution")
SCORE_FIELDS = ("routed", "base", "global", "positive", "negative", "correction")
REPLAY_TOLERANCE = 1e-4  # numerical agreement only; never changes ASR's >= rule


def text_id(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_reference(directory):
    """Lock weights and saved inputs; disclose code drift and require replay.

    Old attack-generation code is not executed here. Its hash drift is recorded,
    not silently treated as the original code. Current scoring code is bound in
    the replay cache, and matching old scalar scores is a mandatory QC gate.
    """
    directory = Path(directory).resolve()
    reference = read_json(directory / "rhi_run_binding.json")
    if (reference.get("protocol") != "frozen_checkpoint_evaluation_v1"
            or reference.get("suite") != "rhi"):
        raise ValueError("Expected a completed frozen RHI evaluation directory")
    checkpoint = Path(reference["checkpoints"]["paer"]["path"]).resolve()
    if checkpoint_identity(checkpoint) != reference["checkpoints"]["paer"]["files"]:
        raise ValueError("PAER checkpoint weights/config differ from frozen evaluation")
    config = read_json(checkpoint / "paer_config.json")
    if config.get("model_type") != "paer_aes_v3":
        raise ValueError("Requires original PAER-v3, not an expert or v1/v2 checkpoint")
    if reference["dtype"] != "float32":
        raise ValueError("This diagnostic requires an existing float32 evaluation")
    if sha256(reference["data"]) != reference["data_sha256"]:
        raise ValueError("Frozen evaluation CSV changed")
    if not 1 <= int(reference["max_length"]) <= 1024:
        raise ValueError("Expected the existing <=1024-token AES protocol")
    threshold = float(reference["success_threshold"])
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Invalid frozen attack success threshold")

    inputs, payloads, alignment = {}, {}, None
    for family in ("rudimentary", "hotflip", "injection_family"):
        result = directory / "paer" / family
        hashes = read_json(result / "completed_result_hashes.json")
        required = {"run_manifest.json", "clean_qwk.json", "asr_summary.json"}
        required.update(Path(p).name for p in ATTACK_FILES.values()
                        if Path(p).parent.name == family)
        if family == "injection_family":
            required.add("injection_family_summary.json")
        if not required.issubset(hashes):
            raise ValueError(f"Incomplete saved evaluation: {family}")
        for name, digest in hashes.items():
            if Path(name).name != name:
                raise ValueError("Result marker must contain simple file names")
            if sha256(result / name) != digest:
                raise ValueError(f"Saved evaluation changed: {result / name}")

    for attack, relative in ATTACK_FILES.items():
        path = directory / "paer" / relative
        payload = read_json(path)
        summary, rows = payload["summary"], payload["details"]
        if (summary["attack"] != attack or not isinstance(rows, list)
                or len(rows) != reference["n_essays"] or not rows):
            raise ValueError(f"Attack/sample mismatch: {attack}")
        if float(summary["success_threshold"]) != threshold:
            raise ValueError(f"Threshold mismatch: {attack}")
        identities = []
        for row in rows:
            for key in ("original_text", "perturbed_text"):
                if not isinstance(row.get(key), str) or not row[key]:
                    raise ValueError(f"Missing saved text: {attack}/{key}")
            for key in ("original_score", "perturbed_score"):
                if not math.isfinite(float(row[key])):
                    raise ValueError(f"Nonfinite saved score: {attack}/{key}")
            identities.append((str(row["essay_id"]), row["original_text"]))
        if len({x[0] for x in identities}) != len(identities):
            raise ValueError(f"Duplicate essay identities: {attack}")
        if alignment is not None and identities != alignment:
            raise ValueError("RHI attacks do not contain the same essays in the same order")
        alignment = identities
        inputs[attack] = sha256(path)
        payloads[attack] = payload

    drift = {}
    for name, expected in reference.get("code_hashes", {}).items():
        source = ROOT / name.replace("\\", "/")
        actual = sha256(source) if source.is_file() else None
        if actual != expected:
            drift[name] = {"saved_sha256": expected, "current_sha256": actual}
    return reference, checkpoint, config, payloads, inputs, drift


def guard_output(output, evaluation, checkpoint, protocol):
    output = Path(output).resolve()
    for protected in (Path(evaluation).resolve(), Path(checkpoint).resolve(),
                      ROOT / "data", ROOT / "deberta_checkpoints"):
        protected = protected.resolve()
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError(f"Output must be separate from protected inputs: {protected}")
    marker = output / "rhi_run_binding.json"
    if marker.is_file():
        if read_json(marker) != protocol:
            raise ValueError("Inputs/code/settings changed; use a NEW diagnostic output directory")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to adopt or overwrite a nonempty unbound output directory")


def validate_score(value):
    if not isinstance(value, dict):
        raise ValueError("Malformed replay score")
    if not all(math.isfinite(float(value[k])) for k in SCORE_FIELDS):
        raise ValueError("Nonfinite replay component")
    if not isinstance(value.get("visible_input_sha256"), str):
        raise ValueError("Missing tokenized-input fingerprint")
    if (min(value["positive"], value["negative"], value["correction"]) < -REPLAY_TOLERANCE
            or value["correction"] > value["positive"] + REPLAY_TOLERANCE):
        raise ValueError("Invalid PAER-v3 signed-evidence/correction bounds")
    return max(abs(value["base"] - value["global"] - value["positive"] + value["negative"]),
               abs(value["routed"] - value["base"] + value["correction"]))


def cohort_summary(pairs):
    if not pairs:
        return {"n_pairs": 0}
    n = len(pairs)
    mean = lambda key: math.fsum(row[key] for row in pairs) / n
    masses = {term: math.fsum(max(row[term], 0.0) for row in pairs) for term in TERMS}
    total = math.fsum(masses.values())
    # Ratio of summed positive components, NOT a ratio to a possibly tiny net delta.
    shares = {term: masses[term] / total if total > 0 else None for term in TERMS}
    largest = max(masses, key=masses.get) if total > 0 else None
    ties = [term for term in TERMS if total > 0 and masses[term] == masses[largest]]
    return {
        "n_pairs": n,
        **{f"mean_{key}": mean(key) for key in (*TERMS, "routed_delta", "local_net_delta")},
        "positive_component_mass_shares": shares,
        "largest_positive_component": largest if len(ties) == 1 else None,
        "tied_largest_positive_components": ties if len(ties) > 1 else [],
        "global_delta_ge_success_threshold_fraction": mean("global_ge_threshold"),
        "routed_fixed_set_asr": mean("replay_success"),
        "mean_correction_lift": mean("correction_lift"),
        "negative_correction_lift_count": sum(row["correction_lift"] < 0 for row in pairs),
        "identical_model_input_pairs": sum(row["identical_model_inputs"] for row in pairs),
    }


def decompose_pairs(records, scores, threshold):
    if not records or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Require nonempty records and a finite nonnegative threshold")
    pairs = []
    for index, row in enumerate(records):
        original_id, adversarial_id = text_id(row["original_text"]), text_id(row["perturbed_text"])
        before, after = scores[original_id], scores[adversarial_id]
        identity_error = max(validate_score(before), validate_score(after))
        delta = lambda key: after[key] - before[key]
        parts = {
            "global_delta": delta("global"),
            "positive_evidence_delta": delta("positive"),
            "negative_evidence_contribution": -delta("negative"),
            "correction_contribution": -delta("correction"),
        }
        routed_delta = delta("routed")
        saved_delta = float(row["perturbed_score"]) - float(row["original_score"])
        pairs.append({
            "row_index": index, "essay_id": row["essay_id"],
            "original_text_sha256": original_id, "perturbed_text_sha256": adversarial_id,
            "original_components": before, "perturbed_components": after,
            **parts, "local_net_delta": math.fsum(parts[t] for t in TERMS[1:]),
            "correction_lift": delta("correction"), "routed_delta": routed_delta,
            "saved_delta": saved_delta,
            "saved_success": saved_delta >= threshold,
            "replay_success": routed_delta >= threshold,
            "global_ge_threshold": parts["global_delta"] >= threshold,
            "near_success_threshold": min(abs(saved_delta - threshold),
                                          abs(routed_delta - threshold)) <= REPLAY_TOLERANCE,
            "identical_model_inputs": before["visible_input_sha256"] == after["visible_input_sha256"],
            "replay_abs_error": max(abs(before["routed"] - float(row["original_score"])),
                                    abs(after["routed"] - float(row["perturbed_score"]))),
            "identity_abs_error": max(identity_error, abs(routed_delta - math.fsum(parts.values()))),
        })
    summary = {
        "n_pairs": len(pairs), "success_threshold": threshold,
        "replay_max_abs_error": max(row["replay_abs_error"] for row in pairs),
        "identity_max_abs_error": max(row["identity_abs_error"] for row in pairs),
        "saved_vs_replay_success_disagreements": sum(row["saved_success"] != row["replay_success"] for row in pairs),
        "near_threshold_disagreements": sum(row["near_success_threshold"] and
                                            row["saved_success"] != row["replay_success"] for row in pairs),
        "all_pairs": cohort_summary(pairs),
        "saved_attack_successes": cohort_summary([row for row in pairs if row["saved_success"]]),
        "replay_attack_successes": cohort_summary([row for row in pairs if row["replay_success"]]),
    }
    return summary, pairs


def rhi_macro(rows):
    by_attack = {row["attack"]: row for row in rows}
    if len(by_attack) != len(rows) or set(by_attack) != set(ATTACK_FILES):
        raise ValueError("Require R, H and both Injection subattacks")
    if len({row["n_pairs"] for row in rows}) != 1:
        raise ValueError("Unequal essay counts")
    fields = [f"mean_{t}" for t in (*TERMS, "routed_delta", "local_net_delta")]
    fields.append("routed_fixed_set_asr")
    get = lambda attack, key: by_attack[attack]["all_pairs"][key]
    return {key: (get("rudimentary", key) + get("hotflip", key)
                  + (get("injection_external", key) + get("injection_self_dup", key)) / 2) / 3
            for key in fields}


def replay_chunk(scorer, texts, batch_size, max_length, progress):
    import torch
    from tqdm.auto import tqdm

    result = {}
    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), batch_size), desc="Score-path replay",
                          unit="batch", dynamic_ncols=True, disable=not progress):
            batch = texts[start:start + batch_size]
            encoded = scorer.tokenizer(batch, return_tensors="pt", truncation=True,
                                       max_length=max_length, padding=True)
            output = scorer.model(input_ids=encoded["input_ids"].to(scorer.device),
                                  attention_mask=encoded["attention_mask"].to(scorer.device))
            required = ("global_logits", "base_logits", "correction", "attention_weights",
                        "positive_evidence", "negative_evidence")
            if any(getattr(output, key, None) is None for key in required):
                raise TypeError("Model does not expose PAER-v3 score-path components")
            vectors = {
                "routed": output.logits.reshape(-1), "base": output.base_logits.reshape(-1),
                "global": output.global_logits.reshape(-1), "correction": output.correction.reshape(-1),
                "positive": (output.attention_weights.float() * output.positive_evidence.float()).sum(1),
                "negative": (output.attention_weights.float() * output.negative_evidence.float()).sum(1),
            }
            values = {key: value.float().cpu().tolist() for key, value in vectors.items()}
            for i, text in enumerate(batch):
                length = int(encoded["attention_mask"][i].sum())
                token_ids = encoded["input_ids"][i, :length].tolist()
                value = {key: values[key][i] for key in SCORE_FIELDS}
                value["visible_input_sha256"] = text_id(json.dumps(token_ids))
                if validate_score(value) > REPLAY_TOLERANCE:
                    raise ValueError("PAER component identity failed; no structural conclusion allowed")
                result[text_id(text)] = value
            del output, encoded, vectors, values
    return result


def render_report(payload):
    lines = ["# PAER-v3 score-path diagnostic", "",
             "Existing frozen attack texts only. No attacks, training, API calls or new essays.",
             "Descriptive algebraic decomposition, NOT causal attribution or a quality judgment.",
             "RHI = (R + H + (I-external + I-self-dup)/2)/3. MLM is not used.",
             f"Replay quality check: {payload['quality_check']['status']}",
             f"Max saved-score replay error: {max(row['replay_max_abs_error'] for row in payload['attacks']):.8g}",
             f"Max component identity error: {max(row['identity_max_abs_error'] for row in payload['attacks']):.8g}",
             f"Saved/replayed success disagreements: {payload['quality_check']['saved_vs_replay_success_disagreements']}",
             f"Historical code files with changed/missing hashes: {len(payload['protocol']['historical_code_differences'])}; see JSON.", "",
             "Delta final = Delta global + Delta positive - Delta negative - Delta correction.",
             "A positive -Delta negative/correction means that a deduction became smaller.", "",
             "## All saved pairs", "",
             "| Attack | n | Delta global | Delta positive | -Delta negative | -Delta correction | Delta final |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in payload["attacks"]:
        s = row["all_pairs"]
        values = [s[f"mean_{t}"] for t in (*TERMS, "routed_delta")]
        lines.append(f"| {row['attack']} | {s['n_pairs']} | " + " | ".join(f"{v:.6f}" for v in values) + " |")
    macro = payload["rhi_macro"]
    lines.append("| RHI equal-family macro | - | " + " | ".join(
        f"{macro[f'mean_{t}']:.6f}" for t in (*TERMS, "routed_delta")) + " |")
    lines += ["", "## Remaining successes in the ORIGINAL saved evaluation", "",
              "Positive-mass share = sum(max(Delta global,0)) / sum of positive component changes.",
              "This is bounded descriptive accounting, not a fraction of causal responsibility.",
              "It is not divided by a possibly near-zero final delta.", "",
              "| Attack | n successes | Mean Delta global | Mean net local change | Global positive-mass share | Largest positive source |",
              "| --- | ---: | ---: | ---: | ---: | --- |"]
    for row in payload["attacks"]:
        s = row["saved_attack_successes"]
        if not s["n_pairs"]:
            lines.append(f"| {row['attack']} | 0 | NA | NA | NA | NA |")
            continue
        share = s["positive_component_mass_shares"]["global_delta"]
        formatted = f"{share:.4f}" if share is not None else "NA"
        lines.append(f"| {row['attack']} | {s['n_pairs']} | {s['mean_global_delta']:.6f} | "
                     f"{s['mean_local_net_delta']:.6f} | {formatted} | {s['largest_positive_component']} |")
    lines += ["", "## Research decision boundary", "",
              "If residual gains consistently arise mainly in the global component, this motivates",
              "testing routing on the main scoring path. It does not prove that this redesign works.",
              "If local terms or reduced corrections drive the gains, address those mechanisms first.",
              "No automatic training approval, new ASR-improvement threshold, or quality label is created.",
              "Fixed-set component scores are not independently attacked branch baselines.",
              "Small numerical replay errors can flip exact-threshold cases; inspect disagreement counts.",
              "Full model remains adversarially trained; a global contribution is not an undefended model.", ""]
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=2, help="Forward-only batch; 2 is conservative on RTX3090 24GB")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="Read/hash inputs only; no files, Torch, GPU or network")
    parser.add_argument("--cpu-summary-only", action="store_true", help="Require complete verified replay cache; never load model")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    evaluation, output = args.evaluation_dir.resolve(), args.output_dir.resolve()
    reference, checkpoint, config, payloads, inputs, drift = load_reference(evaluation)
    texts = list(dict.fromkeys(text for p in payloads.values() for r in p["details"]
                               for text in (r["original_text"], r["perturbed_text"])))
    diagnostic_code = code_identity("paer/analyze_aes_paer_score_paths.py", "paer/rhi_experiment_utils.py",
                                    "paer/routing_experiment_utils.py", "paer/modeling_paer_v3.py",
                                    "paer/modeling_paer.py", "text_scoring_adv_training/evaluation/aes/scorer.py")
    protocol = {"diagnostic": "paer_v3_fixed_set_score_paths_v1", "reference": reference,
                "reference_binding_sha256": sha256(evaluation / "rhi_run_binding.json"),
                "saved_inputs_sha256": inputs, "current_code_sha256": diagnostic_code,
                "historical_code_differences": drift, "batch_size": args.batch_size,
                "device": args.device, "dtype": "float32", "replay_tolerance": REPLAY_TOLERANCE,
                "cache_shard_size": 128, "n_unique_texts": len(texts)}
    guard_output(output, evaluation, checkpoint, protocol)
    if args.dry_run:
        print(json.dumps({"mode": "READ_ONLY_NO_NETWORK_NO_GPU_NO_TRAINING", "checkpoint": str(checkpoint),
                          "evaluation_dir": str(evaluation), "output_dir": str(output),
                          "n_essays_per_attack": reference["n_essays"], "n_unique_texts": len(texts),
                          "batch_size": args.batch_size, "dtype": "float32", "model_type": config["model_type"],
                          "historical_code_differences": drift, "files_created": 0}, indent=2))
        return 0
    if args.cpu_summary_only and not (output / "rhi_run_binding.json").is_file():
        raise FileNotFoundError("No diagnostic replay cache. Run forward replay first.")
    bind_directory(output, protocol)
    scores, scorer, torch_module = {}, None, None
    for start in range(0, len(texts), 128):
        chunk = texts[start:start + 128]
        path = output / "replay_cache" / f"texts_{start:06d}.json"
        marker = path.with_suffix(".sha256.json")
        if path.exists() and marker.exists():
            if sha256(path) != read_json(marker)["sha256"]:
                raise ValueError(f"Replay cache changed: {path}")
            cached = read_json(path)
            if list(cached) != [text_id(text) for text in chunk]:
                raise ValueError("Replay cache text identities/order differ")
            for value in cached.values():
                if validate_score(value) > REPLAY_TOLERANCE:
                    raise ValueError("Cached score identity failed")
        else:
            if args.cpu_summary_only:
                raise FileNotFoundError(f"Replay cache incomplete: {path}")
            if scorer is None:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                import torch
                from text_scoring_adv_training.evaluation.aes.scorer import AESScorer
                torch_module = torch
                scorer = AESScorer(checkpoint, device=args.device, dtype=torch.float32)
                if str(args.device).startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats(scorer.device)
            print(f"[REPLAY] Unique texts {start + 1}-{start + len(chunk)}/{len(texts)}", flush=True)
            cached = replay_chunk(scorer, chunk, args.batch_size, int(reference["max_length"]), not args.no_progress)
            save_json(path, cached)
            save_json(marker, {"sha256": sha256(path)})
        scores.update(cached)

    attacks = []
    for attack, payload in payloads.items():
        summary, pairs = decompose_pairs(payload["details"], scores, float(reference["success_threshold"]))
        attacks.append({"attack": attack, **summary})
        save_json(output / f"{attack}_score_path_pairs.json", pairs)
    if checkpoint_identity(checkpoint) != reference["checkpoints"]["paer"]["files"]:
        raise ValueError("Checkpoint changed during replay; do NOT interpret these cached scores")
    for attack, relative in ATTACK_FILES.items():
        if sha256(evaluation / "paer" / relative) != inputs[attack]:
            raise ValueError("Saved attack input changed during replay; do NOT interpret the output")
    ok = all(row["replay_max_abs_error"] <= REPLAY_TOLERANCE
             and row["identity_max_abs_error"] <= REPLAY_TOLERANCE for row in attacks)
    gpu = None
    if torch_module is not None and str(args.device).startswith("cuda"):
        gpu = {"peak_allocated_mib": torch_module.cuda.max_memory_allocated(scorer.device) / 2**20,
               "peak_reserved_mib": torch_module.cuda.max_memory_reserved(scorer.device) / 2**20,
               "scope": "This process only; not nvidia-smi total device usage"}
    disagreements = sum(row["saved_vs_replay_success_disagreements"] for row in attacks)
    status = ("PASS_WITH_THRESHOLD_FLIPS" if disagreements else "PASS") if ok else "FAIL_DO_NOT_INTERPRET"
    report = {"protocol": protocol, "quality_check": {"status": status,
              "saved_vs_replay_success_disagreements": disagreements,
              "replay_tolerance": REPLAY_TOLERANCE}, "gpu_memory": gpu,
              "attacks": attacks, "rhi_macro": rhi_macro(attacks)}
    save_json(output / "score_path_summary.json", report)
    markdown = render_report(report)
    (output / "score_path_summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Saved: {output / 'score_path_summary.md'}")
    if not ok:
        raise ValueError("Replay/QC mismatch. Do NOT use this report to justify an architecture change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
