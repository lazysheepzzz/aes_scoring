# Complete the five missing MLM transfer cells

This entrypoint does not train or select checkpoints. It evaluates D-HotFlip
(gstep600), D-Rudimentary-v2 (gstep400), D-Injection (gstep1200), and both retrained
PAER-RHI ablations (gstep400) in sequence. All use attack seed42 and the completed
full PAER-RHI MLM reference protocol. ModernBERT stays bfloat16 if that is what the
reference used; victim precision/batch size also come from the reference.

Defaults read ablation selections from the completed `run02` directory and output
under `outputs/aes_seed42_missing_mlm_transfers`. Old results are not overwritten.
Completed results in this new directory can be reused if binding/file hashes match.
An interrupted partial child output is preserved and rejected; choose a new
output directory to retry. Do not change checkpoints based on MLM results.

Remote C PowerShell:

```powershell
python -m unittest discover -s tests -p test_aes_missing_mlm_transfers.py -v
python .\mlm_guided\evaluate_aes_seed42_missing_transfers.py --dry-run
python .\mlm_guided\evaluate_aes_seed42_missing_transfers.py
Get-Content .\outputs\aes_seed42_missing_mlm_transfers\mlm_transfer_results.md
```

Dry-run validates paths, recorded reference data, and freezes weight identities,
then prints all five commands without writes or GPU inference. It cannot validate
runtime GPU memory or model downloads. Local tests cover CPU orchestration only.
For optional two-essay smoke, pass `--n-essays 2` and a separate `--output-dir`.
Run one GPU workflow at a time on the 24 GB RTX3090.

All outputs retain the reference evaluation role (currently development). The
script does not claim independent test results or silently update results.xlsx.
Summary is refreshed after each model; check `completed=5/5` before calling it full.
