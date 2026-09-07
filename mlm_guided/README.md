# AES MLM-guided attack-only transfer evaluation

## 补齐三个已训练专用防御的 MLM 列

运行 `python .\mlm_guided\evaluate_aes_specialist_defenses_mlm.py`，顺序评估
已选定的 D-HotFlip、D-Rudimentary-v2、D-Injection。默认 batch=4，输出到新的
`outputs/aes_specialists_mlm_evaluation_development_seed42/`，不覆盖旧结果。
可先加 `--dry-run` 检查 checkpoint。详细规则见
[RHI 运行说明](../paer/RHI_EXPERIMENT_RUNBOOK.md)。

MLM-guided has no dedicated adversarial-training defense in the source
protocol. It is evaluated after model and checkpoint selection as an unseen
attack-side transfer test. Rudimentary, HotFlip, and Injection remain the
three peer attack--defense families.

For the frozen PAER-RH-v3 seed-42 checkpoint:

```powershell
python .\mlm_guided\evaluate_aes_mlm_guided.py `
  --checkpoint .\outputs\aes_paer_rh_v3_seed42\gstep400 `
  --out .\outputs\eval_mlm_paer_rh_v3_seed42 `
  --skip-clean `
  --seed 42 `
  --batch-size 4 `
  --dtype float32 `
  --mlm-dtype bfloat16
```

This uses the same formal attack defaults as the completed B0/C0 MLM-guided
evaluations. Do not use the result to change PAER-RH-v3 or reselect its
checkpoint.
