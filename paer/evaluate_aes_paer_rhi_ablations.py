"""Select and evaluate retrained ablations; no training and no MLM selection."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paer.rhi_experiment_utils import read_json, save_json, sha256, bind_directory, checkpoint_identity
from paer.run_aes_paer_rhi_ablation_training import ABLATIONS, ablated_config, validate_inputs


def selection_arguments(reference, training_dir, output_dir):
    args = dict(reference)
    args.update(defense_output_dir=str(training_dir), selection_output_dir=str(output_dir),
                subset_ids_path=str(output_dir / "subset_ids.json"),
                training_inputs=str(training_dir / "reference_rhi_training_inputs.json"))
    result = []
    for key, value in args.items():
        if value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        result.extend([flag] if value is True else [flag, str(value)])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-selection", type=Path, default=ROOT / "outputs/aes_paer_rhi_v3_checkpoint_selection_seed42")
    p.add_argument("--reference-evaluation", type=Path, default=ROOT / "outputs/aes_rhi_evaluation_development_seed42")
    p.add_argument("--training-root", type=Path, default=ROOT / "outputs")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/aes_paer_rhi_v3_retrained_ablations_seed42")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    selection = read_json(args.reference_selection / "rhi_run_binding.json")
    evaluation = read_json(args.reference_evaluation / "rhi_run_binding.json")
    source_dir = Path(selection["arguments"]["defense_output_dir"])
    reference_config = read_json(source_dir / "launcher_config.json")
    seed = reference_config["seed"]
    inputs = read_json(source_dir / "rhi_training_inputs.json")
    validate_inputs(reference_config, inputs)
    if evaluation["selection_hashes"]["paer"] != sha256(args.reference_selection / "best_checkpoint.json"):
        raise ValueError("Reference evaluation and selection do not match")
    for key, path in (("valid_sha256", selection["arguments"]["valid_csv"]), ("bank_sha256", selection["arguments"]["injection_sentence_bank"])):
        if sha256(path) != selection[key]:
            raise ValueError(f"Selection input changed: {key}")
    if sha256(evaluation["data"]) != evaluation["data_sha256"]:
        raise ValueError("Evaluation data changed")
    if checkpoint_identity(selection["arguments"]["c0_checkpoint"]) != selection["c0_weights"]:
        raise ValueError("C0 changed")
    training = {}
    for variant in ABLATIONS:
        directory = args.training_root / f"aes_paer_rhi_v3_{variant}_seed{seed}"
        protocol = read_json(directory / "ablation_protocol.json")
        config = read_json(directory / "launcher_config.json")
        if protocol["smoke"] or protocol["reference_config_sha256"] != sha256(source_dir / "launcher_config.json"):
            raise ValueError("Ablation is smoke or uses another reference")
        if config != ablated_config(reference_config, variant, config["output_dir"]):
            raise ValueError(f"Unexpected training configuration: {variant}")
        if read_json(directory / "reference_rhi_training_inputs.json") != inputs:
            raise ValueError("Ablation training provenance differs")
        training[variant] = directory
    evaluation_parameters = {a: read_json(args.reference_evaluation / "paer" / a / "run_manifest.json")["attack_parameters"]
                             for a in ("rudimentary", "hotflip", "injection_family")}
    code_files = list((ROOT / "text_scoring_adv_training/evaluation/aes").rglob("*.py")) + [
        ROOT / "paer/select_aes_rhi_checkpoint.py", ROOT / "paer/modeling_paer_v3.py",
        ROOT / "whitebox/eval_hotflip_defended.py", ROOT / "whitebox/evaluate_aes_checkpoint.py"]
    binding = {"reference_selection": selection, "reference_evaluation": evaluation,
               "evaluation_parameters": evaluation_parameters,
               "code_hashes": {str(f.relative_to(ROOT)): sha256(f) for f in code_files},
               "training": {v: {"config": sha256(d / "launcher_config.json"),
                                "weights": {c.name: checkpoint_identity(c) for c in d.glob("gstep*") if c.is_dir()}}
                            for v, d in training.items()}, "script_sha256": sha256(Path(__file__))}
    if not args.dry_run:
        bind_directory(args.output_dir, binding)
    rows = []
    for variant, directory in training.items():
        selected_dir = args.output_dir / variant / "selection"
        command = [sys.executable, str(ROOT / "paer/select_aes_rhi_checkpoint.py")] + selection_arguments(selection["arguments"], directory, selected_dir)
        print(subprocess.list2cmdline(command), flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, check=True)
        # Verify actual generated selection subset, not merely its seed/size.
        subset_name = f"selection_subset_{selection['arguments']['subset_size']}.csv"
        if sha256(selected_dir / subset_name) != sha256(args.reference_selection / subset_name):
            raise ValueError("Actual selection subset differs from full PAER")
        chosen = read_json(selected_dir / "best_checkpoint.json")
        for attack in ("rudimentary", "hotflip", "injection_family"):
            out = args.output_dir / variant / "evaluation" / attack
            marker = out / "completed_result_hashes.json"
            if marker.exists():
                if any(sha256(out / name) != digest for name, digest in read_json(marker).items()):
                    raise ValueError("Completed evaluation changed")
            else:
                if out.exists() and any(out.iterdir()):
                    raise FileExistsError(f"Partial evaluation preserved; use a new output-dir: {out}")
                command = [sys.executable, str(ROOT / "whitebox/evaluate_aes_checkpoint.py"),
                           "--checkpoint", chosen["checkpoint_path"], "--attack", attack,
                           "--valid", evaluation["data"], "--out", str(out), "--current-python",
                           "--injection-sentence-bank", selection["arguments"]["injection_sentence_bank"]]
                for flag, key in (("seed", "seed"), ("batch-size", "batch_size"), ("dtype", "dtype"),
                                  ("device", "device"), ("n-essays", "n_essays"), ("n-steps", "steps"),
                                  ("max-length", "max_length"), ("success-threshold", "success_threshold")):
                    command += ["--" + flag, str(evaluation[key])]
                params = evaluation_parameters[attack]
                for key in ("beam_size", "max_candidates_per_step", "n_sample_pos", "top_k_per_pos", "max_token_edit_rate"):
                    if params.get(key) is not None:
                        command += ["--" + key.replace("_", "-"), str(params[key])]
                subprocess.run(command, check=True)
                required = ["clean_qwk.json", "asr_summary.json", "run_manifest.json"]
                required += ["injection_family_summary.json", "injection_external_details.json", "injection_self_dup_details.json"] if attack == "injection_family" else [f"{attack}_details.json"]
                save_json(marker, {name: sha256(out / name) for name in required})
            metric = read_json(out / "injection_family_summary.json") if attack == "injection_family" else read_json(out / "asr_summary.json")[0]
            if metric.get("n_essays", metric.get("n_essays_per_subattack")) != evaluation["n_essays"]:
                raise ValueError("Evaluation sample count mismatch")
            rows.append({"variant": variant, "attack": attack, "checkpoint": chosen["checkpoint_name"],
                         "clean": read_json(out / "clean_qwk.json"), "metrics": metric})
        save_json(args.output_dir / "ablation_results.json", {"evaluation_role": evaluation["evaluation_role"], "results": rows})
        lines = ["# Retrained PAER ablations", "", f"Evaluation role: {evaluation['evaluation_role']}", "",
                 "| Variant | Checkpoint | Attack | Clean QWK | ASR | Mean delta |",
                 "| --- | --- | --- | --- | --- | --- |"]
        for row in rows:
            lines.append(f"| {row['variant']} | {row['checkpoint']} | {row['attack']} | {row['clean']['qwk']:.4f} | {row['metrics']['asr']:.4f} | {row['metrics']['avg_delta']:.4f} |")
        (args.output_dir / "ablation_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
