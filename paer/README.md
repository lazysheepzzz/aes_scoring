# PAER-AES main experiment

This directory contains the new AES contribution.  It does not replace the
original paper's shared attack primitives under
`text_scoring_adv_training/evaluation/robustness_tests/common/`.

## Experimental boundary

- Training attacks: Rudimentary and HotFlip only.
- Held-out attack: MLM-guided.  It must not be used for training, validation,
  hyperparameter tuning, or checkpoint selection.
- Attribution victim: the frozen B0 checkpoint by default.
- Training data: `train_fold0.csv` only.
- Checkpoint selection: clean-QWK gate plus the mean subset ASR of
  Rudimentary and HotFlip.
- Mixed-AT-RH and PAER-RH consume the exact same trace JSONL.
- Every training epoch still visits every clean training essay exactly once;
  only the fixed selected subset receives an additional offline attack pair.
  This keeps clean-data coverage and optimizer-step count aligned with C0 and
  the existing attack-specific stage-two runs.

Each accepted attack step produces one counterfactual trace row containing
the before/after text and its true victim-score gain.  The gain is a soft
`victim-induced score inflation` target, not a human writing-quality label.

## Model

PAER keeps the original DeBERTa global score and adds:

1. a token-level score-inflation risk head;
2. a token-level positive-evidence head;
3. a fixed-form correction that subtracts only evidence that is both positive
   and predicted to be inflationary.

The correction can only lower suspicious positive evidence.  It never masks
negative quality evidence, so spelling, grammar, and semantic damage can still
reduce an essay's score.

## Remote run order (PowerShell)

Run every dry-run first after pulling the repository.

### 1. Prepare the shared counterfactual trace dataset

```powershell
python .\paer\prepare_aes_rh_training_traces.py --seed 42 --dry-run

python .\paer\prepare_aes_rh_training_traces.py `
  --seed 42 `
  --n-steps 3 `
  --output .\artifacts\paer\rh_counterfactual_training_traces_seed42.jsonl
```

`n_steps=3` is the formal training-trace protocol: training lasts three epochs
and rotates at most one accepted edit per attacked essay per epoch.  The
30-step setting remains reserved for final robustness evaluation; using it
here creates many traces that never enter training.

The generator writes a progress file after every essay and resumes by default.
Use `--overwrite` only when intentionally starting the trace generation again.

### 2. Train the same-data Mixed-AT-RH baseline

```powershell
python .\paer\run_aes_mixed_at_rh_training.py `
  --seed 42 `
  --output-dir .\outputs\aes_mixed_at_rh_seed42 `
  --dry-run

python .\paer\run_aes_mixed_at_rh_training.py `
  --seed 42 `
  --output-dir .\outputs\aes_mixed_at_rh_seed42
```

### 3. Train PAER-RH

```powershell
python .\paer\run_aes_paer_rh_training.py `
  --seed 42 `
  --output-dir .\outputs\aes_paer_rh_seed42 `
  --dry-run

python .\paer\run_aes_paer_rh_training.py `
  --seed 42 `
  --output-dir .\outputs\aes_paer_rh_seed42
```

### 4. Select checkpoints without MLM leakage

Run once for Mixed-AT and once for PAER.  The selector uses the same fixed
validation subset and evaluates both training attacks.  The formal comparison
uses explicit checkpoints up to `gstep1400`, the last common saved training
budget available for both runs.  `best` and `final` aliases are deliberately
excluded because their step is not encoded in the directory name.  Candidate
scoring defaults to batch size 4 so float32 evaluation leaves safe headroom on
the 24 GB RTX 3090; the same batch size must be used for Mixed-AT and PAER.

```powershell
python .\paer\select_aes_rh_checkpoint.py `
  --defense-output-dir .\outputs\aes_mixed_at_rh_seed42 `
  --selection-output-dir .\outputs\aes_mixed_at_rh_checkpoint_selection_seed42 `
  --max-checkpoint-step 1400 `
  --batch-size 4

python .\paer\select_aes_rh_checkpoint.py `
  --defense-output-dir .\outputs\aes_paer_rh_seed42 `
  --selection-output-dir .\outputs\aes_paer_rh_checkpoint_selection_seed42 `
  --max-checkpoint-step 1400 `
  --batch-size 4
```

Read the selected path:

```powershell
$mixed = (Get-Content `
  .\outputs\aes_mixed_at_rh_checkpoint_selection_seed42\best_checkpoint.json `
  | ConvertFrom-Json).checkpoint_path

$paer = (Get-Content `
  .\outputs\aes_paer_rh_checkpoint_selection_seed42\best_checkpoint.json `
  | ConvertFrom-Json).checkpoint_path
```

### 5. Evaluate seen attacks first

Use the existing evaluator.  It loads PAER checkpoints automatically, and
HotFlip differentiates through the complete PAER correction module.

```powershell
python .\whitebox\evaluate_aes_checkpoint.py `
  --checkpoint $paer --attack rudimentary `
  --out .\outputs\eval_rudimentary_paer_rh_seed42 --seed 42

python .\whitebox\evaluate_aes_checkpoint.py `
  --checkpoint $paer --attack hotflip --skip-clean `
  --out .\outputs\eval_hotflip_paer_rh_seed42 --seed 42
```

Repeat the two commands with `$mixed` for the Mixed-AT baseline.

### 5.1 Diagnose PAER routing contribution without regenerating attacks

Before freezing the method, replay the saved PAER adversarial texts through
both the routed score and its uncorrected base branch.  This is a cheap
fixed-adversarial-set diagnostic, not an adaptive route-off attack.

```powershell
python .\paer\analyze_aes_paer_routing_contribution.py `
  --checkpoint $paer `
  --attack-details `
    .\outputs\eval_rudimentary_paer_rh_seed42\rudimentary_details.json `
    .\outputs\eval_hotflip_paer_rh_seed42\hotflip_details.json `
  --out .\outputs\aes_paer_rh_routing_diagnostic_seed42.json `
  --batch-size 4 --dtype float32
```

### 5.2 PAER-v2 after a collapsed v1 routing diagnostic

PAER-v1 is retained for reproducibility.  If its adversarial correction does
not exceed its clean correction, train the separately named v2 entrypoint.
V2 reuses the exact same RH trace file and B0 initialization.  It changes only
the PAER module/training: balanced edited-vs-unchanged localization, direct
calibration to the stored cumulative victim-score inflation, sparse top-k sum
routing, and a separate learning rate for the randomly initialized heads.
MLM remains excluded.

```powershell
python .\paer\run_aes_paer_rh_v2_training.py `
  --seed 42 `
  --trace-jsonl .\artifacts\paer\rh_counterfactual_training_traces_seed42.jsonl `
  --output-dir .\outputs\aes_paer_rh_v2_seed42 `
  --dry-run

python .\paer\run_aes_paer_rh_v2_training.py `
  --seed 42 `
  --trace-jsonl .\artifacts\paer\rh_counterfactual_training_traces_seed42.jsonl `
  --output-dir .\outputs\aes_paer_rh_v2_seed42

python .\paer\select_aes_rh_checkpoint.py `
  --defense-output-dir .\outputs\aes_paer_rh_v2_seed42 `
  --selection-output-dir .\outputs\aes_paer_rh_v2_checkpoint_selection_seed42 `
  --max-checkpoint-step 1400 --batch-size 4 --dtype float32
```

### 6. Freeze the method, then run MLM once

Do not change architecture, weights, loss coefficients, or checkpoint after
this evaluation.

```powershell
python .\mlm_guided\evaluate_aes_mlm_guided.py `
  --checkpoint $mixed `
  --out .\outputs\eval_mlm_mixed_at_rh_seed42 --seed 42

python .\mlm_guided\evaluate_aes_mlm_guided.py `
  --checkpoint $paer `
  --out .\outputs\eval_mlm_paer_rh_seed42 --seed 42
```

## Small smoke run

Use separate readable files so smoke artifacts cannot be mistaken for formal
results:

```powershell
python .\paer\prepare_aes_rh_training_traces.py `
  --max-essays 20 --attack-fraction 1.0 --n-steps 2 `
  --output .\artifacts\paer\smoke_rh_traces_seed42.jsonl --overwrite

python .\paer\run_aes_paer_rh_training.py `
  --trace-jsonl .\artifacts\paer\smoke_rh_traces_seed42.jsonl `
  --output-dir .\outputs\smoke_aes_paer_rh_seed42 `
  --num-epochs 1 --max-train-samples 20 `
  --per-device-train-batch-size 1 `
  --gradient-accumulation-steps 1 `
  --eval-every 1 --save-every 1
```

Smoke outputs are for execution validation only and must not enter tables.
