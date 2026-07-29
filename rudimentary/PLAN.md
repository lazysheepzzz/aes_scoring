# Rudimentary 主实验运行说明

## 当前状态

Rudimentary 的正式攻击、D_RUDIMENTARY 在线对抗训练、checkpoint 选择和统一
评估入口已经接通。当前实现保留论文原始的字符/词级随机编辑，只在 AES 专用层
补充搜索、真实模型打分、预算、记录和训练逻辑。

论文原始源码保持不修改：

- `text_scoring_adv_training/evaluation/robustness_tests/common/rudimentary_edits.py`

AES 专用新增逻辑：

- `text_scoring_adv_training/evaluation/aes/attacks/rudimentary.py`
- `text_scoring_adv_training/training/aes_trainer.py`
- `whitebox/run_aes_rudimentary_adv_training.py`
- `whitebox/select_aes_rudimentary_defense_checkpoint.py`
- `whitebox/evaluate_aes_checkpoint.py`

## 正式攻击协议

| 参数 | 值 |
|---|---:|
| 最大搜索步数 | 30 |
| 每步候选数 | 16 |
| beam size | 1 |
| 成功阈值 | `score(perturbed) - score(original) >= 0.1` |
| 编辑预算 | 最多接受 `floor(original_token_count * 0.1)` 次编辑，同时不超过 30 |
| 随机种子 | 42 |

每一步都使用论文原始函数生成字符/词级候选，再用当前 victim 对完整候选文本
做真实前向评分。只接受分数至少提高 `1e-6` 的最佳候选；victim tokenizer
看来完全相同的候选会被过滤。输出包含 essay ID、原文、最终扰动文本、分数、
delta、是否成功、band crossing 和逐步编辑记录。

## 旧结果是否可用

`rudimentary_unified_thresh_result.json` 的历史记录是 1154 篇、ASR 94.45%
（1090/1154）、mean delta 0.1156，可用作 B0 的 sanity check。

它不能直接进入正式主表，因为缺少固定 seed、10% 编辑预算、essay ID、原文/
扰动文本和运行清单。`rudimentary_0.1.md` 中另有 1134 篇的旧口径记录，也只
保留作历史追溯。

## D_RUDIMENTARY 训练设计

D_RUDIMENTARY 与 C0、D_HOTFLIP 都从同一个 B0 初始化，共享 epochs、batch、
gradient accumulation、学习率、AdamW、warmup、bf16、seed、评估和保存频率。
唯一实质差异是训练时的攻击。

每个 micro-batch 随机选择 50% 样本；每个样本用论文原始编辑生成 16 个
one-step 候选，当前模型真实评分后选择最高且确实增分的候选。损失为 clean
MSE 加权单边分数膨胀损失，weight=1.0、tolerance=0.05，与 D_HOTFLIP 对齐。

## 训练机 C 的主实验顺序

以下命令在 `E:\xjj\aes_scoring` 的 PowerShell、`xjj_aes` 环境运行。

### 1. dry-run

```powershell
python .\whitebox\run_aes_rudimentary_adv_training.py --seed 42 --output-dir .\outputs\aes_rudimentary_defense_seed42 --dry-run
```

确认 `training_mode` 为 `rudimentary_defense`，`use_rudimentary_edits` 为
`true`，checkpoint 和 CSV 都指向当前仓库中的正确路径。

### 2. 训练 D_RUDIMENTARY

```powershell
python .\whitebox\run_aes_rudimentary_adv_training.py --seed 42 --output-dir .\outputs\aes_rudimentary_defense_seed42
```

不要与其他训练或攻击在同一张 RTX 3090 上并行运行。

### 3. 选择 checkpoint

```powershell
python .\whitebox\select_aes_rudimentary_defense_checkpoint.py
```

选择器复用固定的 256 篇 `prompt_name + score` 分层子集。先要求完整 clean
QWK 不低于 `C0 QWK - 0.02`，再以 10-step Rudimentary subset ASR 最低为
主要标准。结果写入：

```text
outputs\aes_rudimentary_checkpoint_selection_seed42\best_checkpoint.json
```

### 4. 正式评估

```powershell
$hotflipSelected = (Get-Content .\outputs\aes_hotflip_checkpoint_selection_seed42\best_checkpoint.json -Raw | ConvertFrom-Json).checkpoint_path
$rudimentarySelected = (Get-Content .\outputs\aes_rudimentary_checkpoint_selection_seed42\best_checkpoint.json -Raw | ConvertFrom-Json).checkpoint_path

python .\whitebox\evaluate_aes_checkpoint.py --attack rudimentary --checkpoint .\deberta_checkpoints\fold0_best --out .\outputs\eval_rudimentary_b0_seed42 --seed 42
python .\whitebox\evaluate_aes_checkpoint.py --attack rudimentary --checkpoint .\outputs\aes_clean_continuation_seed42\best --out .\outputs\eval_rudimentary_c0_seed42 --seed 42
python .\whitebox\evaluate_aes_checkpoint.py --attack rudimentary --checkpoint $hotflipSelected --out .\outputs\eval_rudimentary_hotflip_defense_seed42 --seed 42
python .\whitebox\evaluate_aes_checkpoint.py --attack rudimentary --checkpoint $rudimentarySelected --out .\outputs\eval_rudimentary_defense_seed42 --seed 42
```

四个目录都会生成 `clean_qwk.json`、`asr_summary.json`、
`rudimentary_details.json` 和 `run_manifest.json`，运行时默认显示进度条和
ETA。

## 主表应报告

至少报告 B0、C0、D_HOTFLIP 和 D_RUDIMENTARY 的：

- clean QWK、MAE；
- Rudimentary ASR@0.10；
- mean delta；
- upward band ASR。

seed 42 完成并确认流程无误后，再补 seed 43/44；这不阻塞当前主实验。
