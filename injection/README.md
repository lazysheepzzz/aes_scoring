# AES Injection experiments

The historical scripts in this directory are retained as experiment-source
references.  They contain hard-coded server paths and should not be used for
the current formal runs.  The readable current entrypoint is:

```text
injection/evaluate_aes_injection_family.py
```

## Frozen protocol

Injection is a peer attack--defense family alongside Rudimentary and HotFlip.
It is not an auxiliary or lower-level experiment merely because its AES
implementation was completed later. All three have dedicated attacks and
defenses. MLM-guided has a different role: it is attack-only transfer
evaluation and has no dedicated defense in the source protocol. “Seen/unseen”
labels elsewhere refer only to a particular model's training exposure.

Injection is evaluated as two equally weighted subattacks:

1. `injection_external`: scorer-guided insertion from the fixed 100-sentence
   Wikipedia bank in `wikipedia_sentences_100.txt`;
2. `injection_self_dup`: scorer-guided duplication of an existing essay
   sentence at a non-adjacent location.

Both use greedy search (`beam=1`), at most 16 candidates per step, at most 30
steps, strict score-improvement acceptance, and early stopping at score gain
`>=0.1`.  No token-edit-rate cap is imposed because a sentence insertion is
the atomic edit in the source Injection protocol.  External and Self-Dup ASR
are reported separately; their equal-weight mean is `Injection family ASR`.

The shared `text_scoring_adv_training/evaluation/robustness_tests/common/`
files from the original repository are not modified.

## Smoke test

```powershell
python .\injection\evaluate_aes_injection_family.py `
  --checkpoint .\deberta_checkpoints\fold0_best `
  --out .\outputs\smoke_injection_b0_seed42 `
  --n-essays 2 --skip-clean `
  --seed 42 --batch-size 4 --dtype float32
```

## Formal baseline attacks

```powershell
python .\injection\evaluate_aes_injection_family.py `
  --checkpoint .\deberta_checkpoints\fold0_best `
  --out .\outputs\eval_injection_b0_seed42 `
  --skip-clean --seed 42 --batch-size 4 --dtype float32

python .\injection\evaluate_aes_injection_family.py `
  --checkpoint .\outputs\aes_clean_continuation_seed42\best `
  --out .\outputs\eval_injection_c0_seed42 `
  --skip-clean --seed 42 --batch-size 4 --dtype float32
```

## Injection adversarial-training baseline

The original repository also trains an Injection-specific defense from
pre-generated injected texts.  The AES adaptation therefore includes
`D_INJECTION`.  Pair generation is offline and does not score candidates with
the victim; training uses the original-paper objective
`max(injected_score - clean_score, 0)^2` in addition to clean MSE.

The fixed pool covers 50% of the training fold and is balanced between
External and Self-Duplication.  Generate it first (this is CPU-only and should
finish quickly):

```powershell
python .\injection\prepare_aes_injection_training_pairs.py --seed 42
```

Then inspect the resolved configuration and train:

```powershell
python .\injection\run_aes_injection_adv_training.py `
  --seed 42 `
  --output-dir .\outputs\aes_injection_defense_seed42 `
  --dry-run

python .\injection\run_aes_injection_adv_training.py `
  --seed 42 `
  --output-dir .\outputs\aes_injection_defense_seed42
```

The 3090-safe default is microbatch 2 with 16-step accumulation, preserving
the same effective batch size 32 used by C0/HotFlip/Rudimentary, and validation
batch size 4.

Select using the frozen 256-essay subset, clean-QWK gate `C0 - 0.02`, and the
equal-weight External/Self-Duplication family ASR:

```powershell
python .\injection\select_aes_injection_defense_checkpoint.py `
  --defense-output-dir .\outputs\aes_injection_defense_seed42 `
  --selection-output-dir .\outputs\aes_injection_checkpoint_selection_seed42
```

`D_INJECTION` is an original-paper-style attack-specific baseline.  It should
eventually be evaluated on Injection, Rudimentary, HotFlip, and MLM-guided for
the cross-robustness matrix.

Do not use Injection pairs or results to revise the already frozen
PAER-RH-v3 method after looking at its Injection evaluation. PAER-RHI-v3 is an
optional additional experiment in which Injection is included in training and
checkpoint selection on the same footing as Rudimentary and HotFlip. If it is
run, Mixed-AT-RHI must use the identical three-family training pool.
MLM-guided remains attack-only transfer evaluation.
