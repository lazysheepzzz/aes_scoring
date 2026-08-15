# AES paper-result collection

After completed remote outputs have been synchronized or while working on the
remote repository, build the current result matrix with:

```powershell
python .\paper_results\build_aes_main_experiment_matrix.py
```

The command writes JSON, CSV, and Markdown files under
`outputs/paper_results/`. Missing experiments remain blank. Rudimentary,
HotFlip, and Injection are treated as the three peer attack--defense families;
MLM-guided is shown separately as attack-only transfer evaluation.

In addition to the complete internal tracker, it produces three no-gap tables
with distinct paper roles:

- `aes_primary_attack_defense_table_seed42.md`: primary R/H/I comparison;
- `aes_mlm_transfer_table_seed42.md`: attack-only MLM transfer comparison;
- `aes_paer_ablation_table_seed42.md`: data-aligned RH method ablation.

## Paired bootstrap confidence intervals

After the Mixed-AT-RH and PAER-RH-v3 per-essay detail files are complete, run:

```powershell
python .\paper_results\bootstrap_paer_v3_vs_mixed_at.py
```

This resamples the 1,154 validation essay identities and reports PAER's ASR
reduction relative to the data-matched Mixed-AT baseline for each attack, the
R/H/I macro, and MLM attack-only transfer. It uses existing results only and
does not require a GPU. The interval describes essay-sampling uncertainty for
the already selected checkpoints; it does not measure training-seed variance.
