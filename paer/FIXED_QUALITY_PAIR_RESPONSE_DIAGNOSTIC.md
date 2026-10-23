# Fixed-text quality-response diagnostic

Purpose: compare B0, Mixed-AT-RHI and PAER-RHI-v3 on the SAME existing A/B texts.
This is an in-sample mechanism diagnostic, not an independent test or a new adaptive attack.
It creates no essay text, does not change labels, calls no API and trains nothing.

## Run on the 24-GB experiment server

Sync the new code; from `E:\xjj\aes_scoring` run:

```powershell
python .\paer\evaluate_aes_fixed_quality_pairs.py --dry-run
python .\paer\evaluate_aes_fixed_quality_pairs.py
Get-Content -Encoding UTF8 .\outputs\quality_assessment\fixed_training_quality_pair_response_seed42_run01\fixed_quality_pair_summary.md
```

Dry-run verifies files, vote bindings and checkpoint hashes but does not load models,
use the GPU or write files. Reading checkpoint hashes can take time.
Actual scoring defaults to float32, batch size 2, max length 1024; progress is shown.
It loads and releases ONE model at a time, using local checkpoints with downloads disabled.
No DeepSeek key or budget ledger writes are required.

## Frozen inputs and models

- Texts: `artifacts/quality_assessment/training_edit_relations_seed42_preview01/`.
- Votes: `artifacts/quality_assessment/training_edit_relations_seed42_judge01/`.
- B0: `deberta_checkpoints/fold0_best`.
- Mixed: existing `aes_mixed_at_rhi_checkpoint_selection_uniform30_seed42/best_checkpoint.json`.
- PAER: existing `aes_paer_rhi_v3_checkpoint_selection_uniform30_seed42/best_checkpoint.json`.

The selector bindings must record 30 steps for all RHI families, no MLM selection,
and the selected weights must still match their selection-time hashes. No re-selection.
The generation-time B0 identity is also checked when present in the frozen pool manifest;
the preview explicitly says if it could not be verified.
The prepared files and paid judge code remain unchanged.

All 60 texts pairs are retained. Primary cohort: original forward/reverse judgments
are consistent AND the normalized relation is worsened (50 pairs in the reported run).
Majority-after-conflict and uncertain/error samples are not promoted into that cohort.
The script recomputes decisions from verified saved votes, not an editable report table.

## Outputs and interpretation

New directory: `outputs/quality_assessment/fixed_training_quality_pair_response_seed42_run01/`.
It contains a bound protocol, three hashed model-score caches, per-pair details,
`fixed_quality_pair_results.json` and `fixed_quality_pair_summary.md`.
Repeated identical runs reuse verified caches; changed inputs/settings/checkpoints require
a NEW `--out`. Unbound pre-existing directories are refused. No overwrite of old results.

- Delta always means score(B) minus score(A); A is an existing intermediate step,
  not necessarily the original human-scored essay.
- Report mean delta, positive-delta rates and paired PAER-minus-Mixed deltas.
- Numerical tolerance 1e-5 is not a writing-quality threshold. Rates at 0.05/0.1 are
  descriptive thresholds, not human quality judgments or adaptive ASR.
- Report pooled values and equal-family RHI macro; Injection subtypes each get half
  an Injection-family weight. A missing family produces NA, never an invented zero.
- Actual encoded-input fingerprints are checked across models. Pairs identical after
  truncation/tokenization remain reported in the full cohort; a second, common-visible
  cohort excludes them. Visibility does not validate the judge's quality assessment.
- No modified-text QWK: original human scores are NOT assigned to either edited state.
- LLM order consistency is not human validation. B0-positive-step selection biases
  the sample, and these parents were in training. Do not claim independent generalization.
- With no stable improved pairs, preservation of legitimate score gains is NOT evaluable.

If memory is insufficient, use batch size 1 with a NEW `--out`; do not change settings
inside an already bound run. Do not run the judge concurrently with this diagnostic.
