# Frozen-model external Injection evaluation with an unseen bank

This supplement does not replace any existing RHI results or train/select models.
External Injection alone is rerun because Self-duplication does not use the bank.
Do not label external-only ASR as Injection-family ASR or RHI macro ASR.

## Bank requirements

- Supply a UTF-8 text file with one sentence per line, 100 nonempty unique lines
  (the same size as the existing bank).
- Document source, collection date/method, and previous use in `--bank-provenance`.
- Do not split the old bank or rewrite its sentences as a purported unseen bank.
- Freeze the bank before evaluation and do not select sentences based on model scores.
- Match the original bank's language, domain, and approximate sentence length where
  possible. Otherwise report the distribution change as an additional difference.
- Automatic checking rejects Unicode/case/whitespace-normalized exact duplicates
  within/between banks. It does not prove semantic disjointness or historical non-use.
- A new bank on existing development essays is still development evaluation.

## Remote PowerShell example

Place the independently collected bank at
`artifacts/injection/unseen_sentence_bank_v1.txt`. This file is deliberately not
generated or populated with invented source claims by the launcher.

```powershell
python .\injection\evaluate_aes_unseen_sentence_bank.py `
  --sentence-bank .\artifacts\injection\unseen_sentence_bank_v1.txt `
  --bank-provenance "REPLACE with actual source, collection method/date, and non-use history" `
  --output-dir .\outputs\smoke_injection_unseen_bank_v1_trainseed42_attackseed42 `
  --n-essays 2 `
  --dry-run
```

Replace the provenance text, inspect the dry-run, then remove `--dry-run` to smoke
test. For the full run, remove `--n-essays 2` and use a DIFFERENT output directory,
e.g. `outputs/eval_injection_unseen_bank_v1_trainseed42_attackseed42`.
Defaults use the completed seed42 RHI development evaluation and seed42 pool.
For another frozen run pass `--source-evaluation` explicitly; seed, precision,
batch size, data, and checkpoints are inherited from its binding.

Mixed and PAER run sequentially in separate child processes (no concurrent GPU
models). Beam size and candidate budget are read from the original Injection
manifests and must agree across models. Evaluator source hashes are recorded.
This is not a bitwise reproduction of historical code.
No override to training or checkpoint selection is performed.

Results: `unseen_sentence_bank_results.json`, with per-model details underneath.
Nonempty foreign directories and changed bindings are rejected. Completed unchanged
model outputs can be reused. An interrupted partial model output is NOT overwritten:
keep it for diagnosis and choose a new directory to retry.

## CPU checks

```powershell
python -m unittest discover -s tests -p test_aes_unseen_sentence_bank.py -v
```
