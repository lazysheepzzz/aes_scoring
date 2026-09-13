"""Retrain controlled PAER-v3 ablations from the full model's frozen inputs."""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paer.rhi_experiment_utils import checkpoint_identity, read_json, save_json, sha256

ABLATIONS = {
    "without_token_localization": {
        "localization_loss_weight": 0.0,
        "attention_alignment_loss_weight": 0.0,
    },
    "without_routing": {
        "correction_scale": 0.0,
        "correction_calibration_weight": 0.0,
        "route_lift_loss_weight": 0.0,
        "clean_correction_weight": 0.0,
    },
}


def ablated_config(reference, variant, output_dir, smoke=False):
    if reference.get("training_mode") != "paer_rh_v3" or reference.get("experiment_name") != "paer_rhi_v3":
        raise ValueError("Reference must be the full PAER-RHI-v3 training configuration")
    if variant not in ABLATIONS:
        raise ValueError("Unknown ablation")
    for key in ("correction_scale", "localization_loss_weight", "attention_alignment_loss_weight"):
        if reference.get(key, 0) <= 0:
            raise ValueError(f"Reference is not a full model: {key}")
    result = copy.deepcopy(reference)
    result.update(ABLATIONS[variant])
    result["output_dir"] = str(output_dir)
    result["experiment_name"] = f"paer_rhi_v3_{variant}"
    if smoke:
        result.update(max_train_samples=160, max_valid_samples=32, num_epochs=1)
    return result


def validate_inputs(config, inputs):
    for path_key, hash_key in (("train_csv", "train_sha256"), ("valid_csv", "valid_sha256"),
                               ("trace_jsonl", "trace_sha256")):
        if sha256(config[path_key]) != inputs[hash_key]:
            raise ValueError(f"Reference training input changed: {path_key}")
    if checkpoint_identity(config["checkpoint_path"]) != inputs["checkpoint"]:
        raise ValueError("B0 initialization changed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", choices=tuple(ABLATIONS), required=True)
    parser.add_argument("--reference-output-dir", type=Path, default=ROOT / "outputs/aes_paer_rhi_v3_seed42")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="160 training / 32 validation rows, 1 epoch; not a paper result")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Will not overwrite existing output: {args.output_dir}")
    reference = read_json(args.reference_output_dir / "launcher_config.json")
    inputs = read_json(args.reference_output_dir / "rhi_training_inputs.json")
    config = ablated_config(reference, args.ablation, args.output_dir, args.smoke)
    validate_inputs(reference, inputs)
    differences = {k: {"full": reference.get(k), "ablation": v} for k, v in config.items() if reference.get(k) != v}
    protocol = {"ablation": args.ablation, "smoke": args.smoke,
                "reference_output_dir": str(args.reference_output_dir.resolve()),
                "reference_config_sha256": sha256(args.reference_output_dir / "launcher_config.json"),
                "differences": differences, "inputs": inputs,
                "retrained_from_b0": True, "mlm_used_for_selection": False,
                "code_hashes": {p.name: sha256(p) for p in (
                    Path(__file__), ROOT / "paer/aes_rh_trainer.py", ROOT / "paer/modeling_paer_v3.py")}}
    print(json.dumps(protocol, indent=2), flush=True)
    if args.dry_run:
        return 0
    save_json(args.output_dir / "ablation_protocol.json", protocol)
    save_json(args.output_dir / "launcher_config.json", config)
    # Reference provenance is retained separately rather than pretending smoke counts are full counts.
    save_json(args.output_dir / "reference_rhi_training_inputs.json", inputs)
    env = os.environ.copy()
    if not args.online:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    command = [sys.executable, str(ROOT / "paer/aes_rh_trainer.py"), "--config",
               str(args.output_dir / "launcher_config.json")]
    return subprocess.run(command, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
