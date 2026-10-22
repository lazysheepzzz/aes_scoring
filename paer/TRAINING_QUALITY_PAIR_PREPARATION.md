# Existing training edits: offline preparation

This stage prepares a feasibility check, not a new trained model or a completed quality audit.
It copies existing accepted RHI training steps without generating or rewriting essay text.
Preparation does not call an API, load a GPU model, start attacks/training, or change old audit files.
The separate, explicit paid runner below appends charges to the SAME original budget ledger.

## On the experiment server

After syncing the code, run from `E:\xjj\aes_scoring`:

```powershell
python -m unittest discover -b -s tests -p test_aes_training_quality_pairs.py
python .\paer\prepare_aes_training_quality_pairs.py --dry-run
```

Inspect `selected_by_attack`, `budget_snapshot` and both reservation estimates.
Defaults select 60 existing parent essays: R=20, H=20, I-external=10, I-self-dup=10.
The two injection subtypes together have the same quota as R or H. MLM is excluded.
Selection is seeded and stratified by prompt/score, not by maximum score gain.
Each pair is ONE recorded accepted step (`before_text` to `adversarial_text`),
not necessarily the original essay and final attack result.

To save that preparation after inspecting the preview:

```powershell
python .\paer\prepare_aes_training_quality_pairs.py
python .\paer\preview_aes_training_quality_judgments.py
```

The default NEW directory is
`artifacts/quality_assessment/training_edit_relations_seed42_preview01/`:

- `blind_pairs.json`: exact texts and mechanically aligned edit spans.
- `private_pair_records.json`: provenance and recorded model gains; NOT judge input.
- `review_protocol.json`: the planned single-relation prompt, not executed requests.
- `preparation_preview.json`: counts and offline cost reservation estimates.
- `source_manifest.json`: input/output hashes and protocol binding.

Existing output directories are refused, even if empty. To repeat extraction, use a new
`--output-dir`; pass that same directory to the preview script with `--input-dir`.
The preview script only reads prepared files and the current old budget ledger.

## Judgment and interpretation boundaries

The planned prompt judges ONE edit-quality relation in full context: improved,
unchanged or worsened. A small supported benefit need not change the overall grade.
Ambiguous/mixed edits remain unknown rather than being forced to unchanged.
Model scores, attack names, original human scores and parent IDs are excluded
from the constructed judge request. No desired score increment is supplied.
These are generic writing criteria, not an official rubric or human ground truth.
The accepted-positive-gain pool is selected data, not quality-change prevalence.
Token-window visibility has not been checked; exported pairs are NOT approved
training supervision. Preparation itself creates no quality labels and enables no API calls.

## Separate paid judge (after preparing the inputs)

Do NOT repeat preparation if the directory has already been saved. Do not use
the old `run_aes_quality_judge.py`: its holistic-quality protocol is different.
Set `DEEPSEEK_API_KEY` locally if it is not already available; never paste it into chat
or put it in a committed file. The new runner requires explicit `--allow-upload`.

```powershell
# No uploads, no writes, no key required: preview the first six already prepared pairs.
python .\paer\run_aes_training_quality_judge.py --max-pairs 6 --dry-run

# PAID: first six pairs = R2/H2/I-external1/I-self-dup1; same run/cache for later continuation.
python .\paer\run_aes_training_quality_judge.py --max-pairs 6 --allow-upload

Get-Content -Encoding UTF8 .\outputs\quality_assessment\training_edit_relations_seed42_judge01\training_edit_quality_summary.md
```

After inspecting those responses, continue the SAME run for all 60, if appropriate:

```powershell
python .\paer\run_aes_training_quality_judge.py --allow-upload
```

Successful votes are reused. A request has at most two charged attempts; interrupted
unknown requests retain their reservations. Do not change directories to retry.
Forward/reverse judgments are normalized; agreeing labels are stable. A disagreement
between two judged labels permits one additional forward vote, recorded as a conflict.
Uncertain/mixed judgments abstain rather than triggering retries for a preferred label.
Non-verbatim/empty quotes, malformed responses and incomplete completions are errors,
not unchanged. Exact quotes are checked automatically, but do not prove the judgment
is correct or relevant; inspect reasons before any supervision decision.

The NEW response cache and budget-status snapshot are under
`artifacts/quality_assessment/training_edit_relations_seed42_judge01/`.
Original responses/reports and prepared texts are not modified. The original cumulative
ledger receives new reserve/settlement entries; its prior entries remain intact.
Missing/replaced/truncated original ledger blocks execution rather than resetting spend.

Each run automatically writes a partial/final report under
`outputs/quality_assessment/training_edit_relations_seed42_judge01/`:
`training_edit_quality_summary.md`, `training_edit_quality_results.json`, `pair_decisions.json`.
Rows not yet judged remain unknown. Reports are counts of quality relations on selected
TRAINING steps, not a new ASR result or a claim that PAER improved. Labels remain
unapproved for training. Stable-only sensitivity counts and conflict flags are retained.

To refresh only the offline report from saved responses:

```powershell
python .\paer\summarize_aes_training_quality_judgments.py
```

## Budget

The original cumulative 30 CNY scope is retained. The old ledger is read-only:
`artifacts/quality_assessment/seed42_deepseek_flash_run01/budget_ledger.jsonl`.
Missing ledger means UNKNOWN prior spend, never zero. Outstanding reservations count.
Base estimates cover both presentation orders; maximum protocol estimates also
reserve a possible third vote and two attempts per order. These are conservative
estimates, not invoices or an account-side spending cap. Prices were checked on
2026-10-22 against the [official pricing page](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)
with a 25% margin over peak uncached input/output rates. Recheck before any future paid run.
