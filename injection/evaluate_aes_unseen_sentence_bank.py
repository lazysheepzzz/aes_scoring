"""Evaluate frozen Mixed/PAER on a new external bank; never select checkpoints."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paer.rhi_experiment_utils import bind_directory, checkpoint_identity, read_json, save_json, sha256


def normalized_sentence(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def audit_banks(training_bank, evaluation_bank):
    banks = [[line.strip() for line in Path(p).read_text(encoding="utf-8-sig").splitlines()
              if line.strip()] for p in (training_bank, evaluation_bank)]
    normalized = [{normalized_sentence(s) for s in bank} for bank in banks]
    if not banks[0] or len(banks[0]) != len(banks[1]):
        raise ValueError("Use nonempty banks with equal sentence counts to preserve candidate-pool size")
    if any(len(bank) != len(keys) for bank, keys in zip(banks, normalized)):
        raise ValueError("Duplicate sentences within bank after Unicode/case/whitespace normalization")
    if normalized[0] & normalized[1]:
        raise ValueError("Evaluation bank overlaps the training bank")
    return {"sentences_per_bank": len(banks[0]), "normalized_exact_overlap": 0,
            "limitation": "Exact normalization checks do not establish semantic novelty or historical non-use"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-evaluation", type=Path, default=ROOT / "outputs/aes_rhi_evaluation_development_seed42")
    p.add_argument("--training-pool-manifest", type=Path, default=ROOT / "artifacts/paer/rhi_training_pool_seed42/rhi_counterfactual_training_traces.manifest.json")
    p.add_argument("--training-bank", type=Path, default=ROOT / "injection/wikipedia_sentences_100.txt")
    p.add_argument("--sentence-bank", type=Path, required=True)
    p.add_argument("--bank-provenance", required=True, help="Source and collection method; confirm never used for training/selection")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n-essays", type=int, help="Smoke subset only; otherwise retain the source evaluation size")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if not args.bank_provenance.strip():
        raise ValueError("Nonempty bank provenance is required")
    audit = audit_banks(args.training_bank, args.sentence_bank)
    source = read_json(args.source_evaluation / "rhi_run_binding.json")
    bank_hash = sha256(args.training_bank)
    if bank_hash != read_json(args.training_pool_manifest)["source_hashes"]["bank"] or bank_hash != source["bank_sha256"]:
        raise ValueError("Reference bank does not match training and original evaluation")
    data = Path(source["data"])
    if sha256(data) != source["data_sha256"]:
        raise ValueError("Source evaluation data changed")
    n = args.n_essays if args.n_essays is not None else source["n_essays"]
    if not 0 < n <= source["n_essays"]:
        raise ValueError("n-essays must be positive and no larger than the source evaluation")
    attack_parameters = []
    for model in ("mixed", "paer"):
        identity = source["checkpoints"][model]
        if checkpoint_identity(identity["path"]) != identity["files"]:
            raise ValueError(f"Frozen checkpoint changed: {model}")
        manifest = read_json(args.source_evaluation / model / "injection_family/run_manifest.json")
        attack_parameters.append(manifest["attack_parameters"])
    budget_keys = ("n_steps", "beam_size", "max_candidates_per_step", "success_threshold")
    budget = {key: attack_parameters[0][key] for key in budget_keys}
    if any(attack_parameters[1][key] != value for key, value in budget.items()):
        raise ValueError("Original Mixed/PAER injection attack budgets differ")
    if budget["n_steps"] != source["steps"] or budget["success_threshold"] != source["success_threshold"]:
        raise ValueError("Original injection manifest differs from evaluation binding")
    protocol = {"protocol": "unseen_external_bank_v1", "source": source,
                "training_pool_manifest_sha256": sha256(args.training_pool_manifest),
                "sentence_bank": str(args.sentence_bank.resolve()), "bank_sha256": sha256(args.sentence_bank),
                "bank_provenance": args.bank_provenance, "bank_audit": audit,
                "n_essays": n, "smoke": n < source["n_essays"], "attack_budget": budget,
                "evaluation_role": source["evaluation_role"],
                "code_hashes": {str(f.relative_to(ROOT)): sha256(f) for f in sorted(
                    list((ROOT / "text_scoring_adv_training/evaluation/aes").rglob("*.py")) +
                    list((ROOT / "paer").glob("modeling_paer*.py")) +
                    [Path(__file__), ROOT / "whitebox/evaluate_aes_checkpoint.py", ROOT / "whitebox/eval_hotflip_defended.py"])}}
    print(json.dumps({"bank_audit": audit, "evaluation_role": source["evaluation_role"], "n_essays": n}, indent=2))
    if not args.dry_run:
        bind_directory(args.output_dir, protocol)
    results = {}
    for model in ("mixed", "paer"):
        out = args.output_dir / model
        done = out / "completed_result_hashes.json"
        if not args.dry_run and done.exists():
            if any(sha256(out / name) != digest for name, digest in read_json(done).items()):
                raise ValueError(f"Completed result changed: {out}")
        else:
            if out.exists() and any(out.iterdir()):
                raise FileExistsError(f"Incomplete/nonempty model output: choose a new output directory: {out}")
            command = [sys.executable, str(ROOT / "whitebox/evaluate_aes_checkpoint.py"),
                       "--checkpoint", source["checkpoints"][model]["path"], "--attack", "injection_external",
                       "--valid", str(data), "--out", str(out), "--current-python",
                       "--injection-sentence-bank", str(args.sentence_bank), "--n-essays", str(n),
                       "--seed", str(source["seed"]), "--batch-size", str(source["batch_size"]),
                       "--dtype", source["dtype"], "--device", source["device"],
                       "--n-steps", str(source["steps"]), "--max-length", str(source["max_length"]),
                       "--beam-size", str(budget["beam_size"]),
                       "--max-candidates-per-step", str(budget["max_candidates_per_step"]),
                       "--success-threshold", str(source["success_threshold"])]
            print("[RUN] " + subprocess.list2cmdline(command), flush=True)
            if args.dry_run:
                continue
            subprocess.run(command, check=True)
            metrics = read_json(out / "asr_summary.json")
            if len(metrics) != 1 or metrics[0]["attack"] != "injection_external" or metrics[0]["n_essays"] != n:
                raise ValueError("Unexpected attack or sample count in result")
            save_json(done, {name: sha256(out / name) for name in
                            ("asr_summary.json", "clean_qwk.json", "run_manifest.json", "injection_external_details.json")})
        if not args.dry_run:
            results[model] = read_json(out / "asr_summary.json")[0]
    if not args.dry_run:
        save_json(args.output_dir / "unseen_sentence_bank_results.json", {"protocol": protocol, "results": results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
