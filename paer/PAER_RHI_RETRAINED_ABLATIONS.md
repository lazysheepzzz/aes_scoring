# PAER-RHI-v3 retrained mechanism ablations

These experiments test contributions, not a new PAER version. Existing full-model,
Mixed-AT, and inference-only route-off results are preserved.

| Variant | Changes from the full reference | Interpretation |
| --- | --- | --- |
| without_token_localization | localization and edited-token attention alignment weights = 0 | Contribution of token-level trajectory supervision; score-level gain calibration and clean regularizers remain |
| without_routing | correction scale = 0; calibration, lift, and clean correction loss weights = 0 | Contribution of routing during training/scoring, together with its dependent objectives; signed token branch and token supervision remain |

The second variant is NOT a pure one-loss ablation or a comparison of directional
versus symmetric routing. Constant correction objectives are disabled because
their output is identically zero. Raw diagnostic loss values can remain nonzero
even when their objective weight is zero. No inference-only switch substitutes
for either retraining experiment. A positive result would support these specific
contributions, not prove that edited tokens are human-invalid evidence.

## Training on remote C

The launcher reads the existing full model's `launcher_config.json` and verifies
CSV, trace and B0 hashes against its `rhi_training_inputs.json`. Learning rates,
seeds, batch sizes, accumulation, precision, data, and epoch budget are inherited.
It uses the unchanged sequential clean/adversarial trainer for the 24 GB GPU.
All output directories must be empty/nonexistent, even for dry-run. No resume or
overwrite of a partial training run is attempted. Do not run both jobs concurrently.

```powershell
python -m unittest discover -s tests -p test_aes_paer_rhi_ablations.py -v

foreach ($variant in @('without_token_localization', 'without_routing')) {
    python .\paer\run_aes_paer_rhi_ablation_training.py --ablation $variant --output-dir ".\outputs\smoke_aes_paer_rhi_v3_${variant}_seed42" --smoke
    if ($LASTEXITCODE -ne 0) { throw "Smoke training failed: $variant" }
}
```

Smoke uses the first 160 training rows, first 32 validation rows and one epoch;
it is not an accuracy comparison. Inspect diagnostics to ensure adversarial rows
actually occurred, and check `best/paer_heads.pt` and `final_clean_metrics.json`.
Route-off correction should be zero; small-sample QWK is not an acceptance gate.

After both smoke runs succeed:

```powershell
foreach ($variant in @('without_token_localization', 'without_routing')) {
    python .\paer\run_aes_paer_rhi_ablation_training.py --ablation $variant --output-dir ".\outputs\aes_paer_rhi_v3_${variant}_seed42"
    if ($LASTEXITCODE -ne 0) { throw "Training failed: $variant" }
}
```

For a different training seed, pass the corresponding full-model
`--reference-output-dir`; do not merely rename the output.

## Selection and evaluation protocol

After full training, the sequential orchestration command is:

```powershell
python .\paer\evaluate_aes_paer_rhi_ablations.py --dry-run
python .\paer\evaluate_aes_paer_rhi_ablations.py
```

It reads the original full-model selection arguments and full evaluation binding,
validates ablation inputs/configurations, then selects and evaluates each variant.
It does not train. Default outputs are isolated under
`outputs/aes_paer_rhi_v3_retrained_ablations_seed42`; the top-level summary is
`ablation_results.md`. Dry-run validates inputs and prints selection commands,
but cannot determine future selected checkpoints or execute GPU evaluations.
Completed outputs are hash-checked and reused. Partial full evaluations are
preserved and rejected; use a new output directory if such a run is interrupted.
Unchanged selector-owned partial selection work can be resumed.

Reuse the existing RHI selector with NEW defense/selection directories. Match
the full model's original selection manifest exactly: same C0 gate, data/subset,
attack seed, R/H/I budgets, dtype, and maximum checkpoint step (1400 in the
current experiment). Do not choose a more favorable checkpoint using MLM.

Evaluate each selected checkpoint independently under R/H/I attacks using the
existing per-checkpoint entrypoints, with the same full evaluation budgets.
Keep outputs under ablation-specific names. The paired Mixed/full-model batch
evaluator is not an ablation aggregator; do not replace its frozen selections.
MLM is optional only after methods/selections are frozen, not required for the
first mechanism check. Report clean QWK and attack ASR/delta together. Current
essay evaluation remains development evaluation, not independent-test evidence.

Start with seed42. Do not launch new seeds or redesign the model based solely on
smoke results. Subsequent full results determine whether the mechanism claim is
supported; no improvement or acceptance threshold is assumed.
