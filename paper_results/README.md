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
