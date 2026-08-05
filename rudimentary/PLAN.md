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
- `rudimentary/run_aes_rudimentary_adv_training.py`
- `rudimentary/select_aes_rudimentary_defense_checkpoint.py`
- `rudimentary/evaluate_aes_rudimentary.py`

目录职责：

| 位置 | 职责 |
|---|---|
| `rudimentary/` | Rudimentary 的当前用户入口、运行说明和历史资产 |
| `text_scoring_adv_training/evaluation/aes/attacks/` | 可复用的 AES 攻击实现 |
| `text_scoring_adv_training/training/` | 多种防御共用的训练实现 |
| `text_scoring_adv_training/evaluation/robustness_tests/common/` | 论文原始通用源码，保持不修改 |
| `whitebox/` | HotFlip 白盒实验入口及跨方法共享的旧兼容入口 |

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

## seed 42 正式结果与 v1 结论

| 模型 | Clean QWK | Rudimentary ASR | Mean delta | Upward band ASR |
|---|---:|---:|---:|---:|
| B0 | 0.8377 | 0.9419 | 0.1164 | 0.1075 |
| C0 | 0.8350 | 0.9965 | 0.1249 | 0.1066 |
| D_HOTFLIP | 0.8216 | 0.9896 | 0.1182 | 0.1265 |
| D_RUDIMENTARY-v1 | 0.8396 | 0.9411 | 0.1152 | 0.1075 |

v1 相对 C0 的 ASR 下降 5.54 个百分点，相对 B0 下降 0.08 个百分点，效果
较弱。v1 保留为训练强度消融，不作为主要 Rudimentary 防御结果。

## D_RUDIMENTARY-v2 训练设计

D_RUDIMENTARY 与 C0、D_HOTFLIP 都从同一个 B0 初始化，共享 epochs、batch、
gradient accumulation、学习率、AdamW、warmup、bf16、seed、评估和保存频率。
优化设置保持一致，攻击专用目标按 v1 结果修正。

每个 micro-batch 随机选择 50% 样本；每个样本用论文原始编辑生成 16 个
候选。v2 的每个候选连续执行 3 次原始编辑，再由当前模型真实评分并选择最高且
确实增分的候选。模型前向候选数仍为16，因此相对 v1 基本不增加 GPU 候选评分
开销。损失为 clean MSE 加权单边分数膨胀损失，weight=1.0；Rudimentary
相对增分 tolerance 从0.05改为0.0，并将相对虚高项从平方惩罚改为线性单边
惩罚，避免小增分在平方后缩小到近乎零。HotFlip 的既有平方损失不变。

## 训练机 C 的主实验顺序

以下命令在 `E:\xjj\aes_scoring` 的 PowerShell、`xjj_aes` 环境运行。

### 1. dry-run

```powershell
python .\rudimentary\run_aes_rudimentary_adv_training.py --seed 42 --output-dir .\outputs\aes_rudimentary_defense_v2_seed42 --dry-run
```

确认 `training_mode` 为 `rudimentary_defense`，`use_rudimentary_edits` 为
`true`，`rudimentary_edits_per_candidate` 为3，`rudimentary_tolerance` 为0，
`rudimentary_relative_loss_power` 为1，checkpoint 和 CSV 都指向当前仓库中的
正确路径。

### 2. 训练 D_RUDIMENTARY-v2

```powershell
python .\rudimentary\run_aes_rudimentary_adv_training.py --seed 42 --output-dir .\outputs\aes_rudimentary_defense_v2_seed42
```

不要与其他训练或攻击在同一张 RTX 3090 上并行运行。

### 3. 选择 checkpoint

```powershell
python .\rudimentary\select_aes_rudimentary_defense_checkpoint.py
```

选择器复用固定的 256 篇 `prompt_name + score` 分层子集。先要求完整 clean
QWK 不低于 `C0 QWK - 0.02`，再以30-step Rudimentary subset ASR最低为
主要标准。结果写入：

```text
outputs\aes_rudimentary_defense_v2_checkpoint_selection_seed42\best_checkpoint.json
```

### 4. 正式评估

```powershell
$rudimentaryV2Selected = (Get-Content .\outputs\aes_rudimentary_defense_v2_checkpoint_selection_seed42\best_checkpoint.json -Raw | ConvertFrom-Json).checkpoint_path

python .\rudimentary\evaluate_aes_rudimentary.py --checkpoint $rudimentaryV2Selected --out .\outputs\eval_rudimentary_defense_v2_seed42 --seed 42
```

已有 B0、C0、D_HOTFLIP 和 v1 结果无需重跑。v2 评估目录会生成
`clean_qwk.json`、`asr_summary.json`、`rudimentary_details.json` 和
`run_manifest.json`，运行时默认显示进度条和 ETA。

## 主表应报告

至少报告 B0、C0、D_HOTFLIP 和 D_RUDIMENTARY 的：

- clean QWK、MAE；
- Rudimentary ASR@0.10；
- mean delta；
- upward band ASR。

seed 42 完成并确认流程无误后，再补 seed 43/44；这不阻塞当前主实验。
