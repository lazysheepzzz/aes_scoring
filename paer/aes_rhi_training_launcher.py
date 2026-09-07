"""Launch paired RHI runs with the existing 24 GB-safe sequential trainer."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from paer.aes_rh_training_launcher import build_parser as rh_parser, build_config, MIXED_AT_RH, PAER_RH_V3
from paer.prepare_aes_rhi_training_traces import load_rows
from paer.rhi_experiment_utils import ROOT, RHI_ATTACKS, read_json, save_json, sha256, checkpoint_identity, validate_trace_groups


def build_parser(experiment):
    if experiment not in ("mixed_at_rhi", "paer_rhi_v3"):
        raise ValueError(experiment)
    backend = MIXED_AT_RH if experiment == "mixed_at_rhi" else PAER_RH_V3
    parser = rh_parser(backend)
    parser.description = f"Train {experiment}: shared balanced R/H/I data; MLM excluded."
    parser.set_defaults(output_dir=None,
                        trace_jsonl=ROOT / "artifacts/paer/rhi_training_pool_seed42/rhi_counterfactual_training_traces.jsonl")
    return parser, backend


def main(experiment):
    parser, backend = build_parser(experiment)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = ROOT / "outputs" / f"aes_{experiment}_seed{args.seed}"
    config = build_config(args, backend)
    # Backend mode identifies the existing architecture/loss implementation.
    # experiment_name and training_attacks identify the new experiment exposure.
    config.update(experiment_name=experiment, training_attacks=list(RHI_ATTACKS))
    print(json.dumps(config, indent=2), flush=True)
    if args.dry_run:
        return 0
    if args.max_trace_records is not None:
        raise ValueError("RHI forbids prefix trace truncation; use a balanced smoke pool instead")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Training output is not empty; use a NEW path: {args.output_dir}")
    rows = load_rows(args.train_csv)
    manifest = read_json(args.trace_jsonl.with_suffix(".manifest.json"))
    if manifest["source_hashes"]["train"] != sha256(args.train_csv):
        raise ValueError("Training CSV differs from the pool source")
    if manifest["trace_sha256"] != sha256(args.trace_jsonl):
        raise ValueError("Trace pool differs from its manifest")
    if manifest["checkpoint"] != checkpoint_identity(args.checkpoint_path):
        raise ValueError("Training initialization differs from the frozen attribution B0")
    if args.max_length != manifest["arguments"]["max_length"]:
        raise ValueError("Training max-length differs from trace generation")
    with args.trace_jsonl.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if args.max_train_samples:
        rows = rows[:args.max_train_samples]
        records = [r for r in records if int(r["row_index"]) < len(rows)]
    _, counts = validate_trace_groups(records, rows)
    valid_rows = load_rows(args.valid_csv)
    if {r["text"] for r in rows} & {r["text"] for r in valid_rows}:
        raise ValueError("Training and development CSV contain overlapping texts")
    save_json(args.output_dir / "launcher_config.json", config)
    save_json(args.output_dir / "rhi_training_inputs.json", {
        "experiment_name": experiment, "backend_training_mode": backend,
        "train_sha256": sha256(args.train_csv), "valid_sha256": sha256(args.valid_csv),
        "trace_sha256": sha256(args.trace_jsonl), "checkpoint": manifest["checkpoint"],
        "essay_counts_by_attack": counts, "mlm_used_for_training": False,
        "evaluation_status": "development data; no independent test claim",
    })
    env = os.environ.copy()
    if not args.online:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    command = [sys.executable, str(ROOT / "paer/aes_rh_trainer.py"), "--config",
               str(args.output_dir / "launcher_config.json")]
    print(f"[RUN] {subprocess.list2cmdline(command)}", flush=True)
    return subprocess.run(command, env=env, check=False).returncode
