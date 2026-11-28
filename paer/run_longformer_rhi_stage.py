"""One explicit stage at a time. Default prints commands; --run executes that stage only."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paer.rhi_experiment_utils import checkpoint_identity, read_json, save_json, sha256
OUT = ROOT / "outputs"
BASE = OUT / "longformer_asap2_b0_seed42/best"
ART = ROOT / "artifacts/paer/longformer_asap2_seed42"
RH = ART / "rh_training_traces.jsonl"
POOL = ART / "rhi_training_pool"
C0 = OUT / "longformer_asap2_c0_resumable_seed42"
MODELS = {k: OUT / f"longformer_asap2_{k}_seed42" for k in ("mixed", "paer")}
SEL = {k: OUT / f"longformer_asap2_{k}_selection_uniform30_seed42" for k in MODELS}
SMALL = ["--per-device-train-batch-size", "2", "--gradient-accumulation-steps", "16",
         "--per-device-eval-batch-size", "2"]


def commands(stage):
    if stage in ("c0", "resume-c0"):
        return [["paer/train_resumable_longformer_c0.py", *(["--resume"] if stage == "resume-c0" else [])]]
    if stage == "rh-pool":
        return [["paer/prepare_aes_rh_training_traces.py", "--checkpoint", BASE, "--output", RH,
                 "--score-batch-size", "2"]]
    if stage == "rhi-pool":
        return [["paer/prepare_aes_rhi_training_traces.py", "--checkpoint", BASE, "--rh-traces", RH,
                 "--batch-size", "2", "--output-dir", POOL]]
    if stage in MODELS:
        name = "mixed_at" if stage == "mixed" else "paer"
        return [[f"paer/run_aes_{name}_rhi_training.py", "--checkpoint-path", BASE, *SMALL,
                 "--trace-jsonl", POOL/"rhi_counterfactual_training_traces.jsonl",
                 "--output-dir", MODELS[stage]]]
    if stage.startswith("select-"):
        k = stage.removeprefix("select-")
        return [["paer/select_aes_rhi_checkpoint.py", "--defense-output-dir", MODELS[k],
                 "--selection-output-dir", SEL[k], "--c0-checkpoint", C0/"best",
                 "--batch-size", "2", "--max-checkpoint-step", "1400"]]
    if stage == "evaluate-b0":
        return [["whitebox/evaluate_aes_checkpoint.py", "--current-python", "--checkpoint", BASE,
                 "--attack", a, "--seed", "42", "--batch-size", "2", "--dtype", "float32",
                 "--out", OUT/"longformer_asap2_b0_evaluation_seed42"/a]
                for a in ("rudimentary", "hotflip", "injection_family", "mlm_guided")]
    return [["paer/evaluate_aes_rhi_experiments.py", "--include-mlm", "--batch-size", "2",
             "--mixed-selection", SEL["mixed"]/"best_checkpoint.json",
             "--paer-selection", SEL["paer"]/"best_checkpoint.json",
             "--output-dir", OUT/"longformer_asap2_paired_evaluation_seed42"]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("c0","resume-c0","rh-pool","rhi-pool","mixed","paer","select-mixed",
                                     "select-paer","evaluate-b0","evaluate-pair"))
    p.add_argument("--run", action="store_true")
    args = p.parse_args()
    for script, *opts in commands(args.stage):
        cmd = [sys.executable, str(ROOT/script), *map(str, opts)]
        print(subprocess.list2cmdline(cmd), flush=True)
        if args.run:
            if args.stage == "c0" and C0.exists():
                raise FileExistsError(f"Refusing to overwrite: {C0}; use resume-c0 --run")
            if args.stage.startswith("select-") and not (C0/"completed.json").is_file():
                raise ValueError(f"Longformer C0 has not finished: {C0}")
            if args.stage == "evaluate-b0":
                out = Path(opts[-1])
                marker = out/"longformer_b0_completed.json"
                provenance = dict(checkpoint=checkpoint_identity(BASE), seed=42, attack=out.name,
                                  valid_sha256=sha256(ROOT/"data/valid_fold0.csv"))
                if marker.exists():
                    rec = read_json(marker)
                    if rec["inputs"] != provenance or any(sha256(out/k) != v for k,v in rec["files"].items()):
                        raise ValueError(f"Changed completed B0 evaluation: {out}")
                    print(f"[REUSE] {out}")
                    continue
                if out.exists():
                    raise FileExistsError(f"Incomplete B0 evaluation; preserve it under another name before retrying: {out}")
            subprocess.run(cmd, check=True, cwd=ROOT)
            if args.stage == "evaluate-b0":
                names = ["asr_summary.json","run_manifest.json","clean_qwk.json"]
                if out.name == "injection_family":
                    names += ["injection_family_summary.json","injection_external_details.json","injection_self_dup_details.json"]
                else:
                    names += [f"{out.name}_details.json"]
                save_json(marker,dict(inputs=provenance,files={k:sha256(out/k) for k in names}))


if __name__ == "__main__":
    main()
