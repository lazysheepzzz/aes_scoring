# AES Injection experiments

The historical scripts in this directory are retained as experiment-source
references.  They contain hard-coded server paths and should not be used for
the current formal runs.  The readable current entrypoint is:

```text
injection/evaluate_aes_injection_family.py
```

## Frozen protocol

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

Do not train on Injection or use it for PAER checkpoint selection.  Evaluate
the final selected PAER-v2 checkpoint only after its RH-based specification
and checkpoint have been frozen.  This keeps Injection, like MLM-guided, as a
held-out structural attack.
