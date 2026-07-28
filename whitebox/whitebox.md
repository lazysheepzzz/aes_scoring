# HotFlip 白盒攻击 — AES 鲁棒性评估

## 概述

HotFlip 是基于梯度引导的 token 替换攻击。对 essay 中每个 token 位置，用一阶近似找出能让模型打分最高（分数提升最大）的替换词，迭代进行直到分数提升达到阈值。

## 攻击参数（统一配置）

| 参数 | 值 | 说明 |
|------|-----|------|
| `n_steps` | 30 | 最大迭代步数 |
| `beam_size` | 1 | 贪心搜索 |
| `n_sample_pos` | 8 | 每步采样位置数 |
| `top_k_per_pos` | 2 | 每位置 top_k 候选 |
| `max_candidates_per_step` | 16 | 每步最多候选数 |
| `threshold` | 0.1 | 成功阈值（delta >= 0.1 才算攻击成功） |
| `max_token_edit_rate` | 0.1 | 最多修改原始有效 token 的 10% |
| `seed` | 42 | Python、NumPy 和 PyTorch 统一随机种子 |

## 历史结果

以下结果由修复前的旧损失训练和旧入口生成，只作为探索性基线保留。两组
ASR 都使用逐篇 `score_single` 和相同的 `delta >= 0.1` 口径，可以彼此比较；
旧文档中的 clean QWK 使用了错误的左填充 batch，必须重新计算。

### Undefended（无防御 baseline，历史）

| 指标 | 值 |
|------|-----|
| Victim | `/root/autodl-tmp/victim/fold0_best` |
| 测试集 | valid_fold0.csv（1154 条） |
| **ASRΔ@0.10** | **87.78%** |
| avg_steps | 7.70（成功攻击平均迭代步数） |
| 攻击脚本 | `run_hotflip_asr_undefended.py` |
| 结果文件 | `hotflip_asr_undefended.json` |

### Defended（旧 v4 对抗训练后，历史）

| 指标 | 值 |
|------|-----|
| 模型 | 旧 v4 pairwise 对抗训练（weight=2.0, margin=0.1, epochs=5） |
| 测试集 | valid_fold0.csv（1154 条） |
| **ASRΔ@0.10** | **67.50%** |
| **旧 QWK** | 0.8222（受左填充 batch 问题影响，不再采用） |
| avg_steps | 7.35（成功攻击平均迭代步数） |
| 结果文件 | `hotflip_asr_defended.json` |

### 防御效果

- ASR 从 87.78% 降到 67.50%，下降 **20.3 个百分点**
- 该结果证明旧式对抗训练有探索性收益，但修复后需要重新训练和评估

---

## 文件说明

### 攻击脚本

| 文件 | 作用 |
|------|------|
| `run_hotflip_asr_undefended.py` | 对 **undefended** victim 跑统一参数 HotFlip |
| `eval_hotflip_defended.py` | 重新计算 defended clean QWK、MAE 和 HotFlip ASR |
| `generate_hotflip_train_adv_data.py` | 生成对抗训练用的 HotFlip 数据（JSONL 格式），用于离线对抗训练 |

### Python 启动脚本

| 文件 | 作用 |
|------|------|
| `run_aes_attacks.py` | 通用攻击入口（调用 `evaluation/aes/run_attacks.py`） |
| `run_aes_clean_continuation_training.py` | C0 干净续训入口 |
| `run_aes_hotflip_adv_training.py` | D_HOTFLIP 对抗训练入口 |
| `aes_stage2_training_launcher.py` | C0 与 D_HOTFLIP 共用的第二阶段参数、校验和启动逻辑 |
| `evaluate_aes_checkpoint.py` | 对任意 B0/C0/D checkpoint 计算 clean 指标和 HotFlip ASR |
| `eval_hotflip_defended.py` | 旧名称兼容入口；由 `evaluate_aes_checkpoint.py` 调用 |

训练和评估入口均支持 `--help` 和 `--dry-run`。在训练机 C 上先激活 `xjj_aes`
Conda 环境，入口随后使用当前 Python 解释器。路径参数均可通过命令行覆盖。

### 第二阶段成对训练

C0 和 D_HOTFLIP 都从同一个 B0 checkpoint 初始化，并共享 epochs、batch、
gradient accumulation、学习率、warmup、bf16、AdamW 参数分组、seed 和
checkpoint 选择规则。唯一的训练差异是：

| 配置 | C0 | D_HOTFLIP |
|------|----|-----------|
| `training_mode` | `clean_continuation` | `hotflip_defense` |
| `use_hotflip_swaps` | `false` | `true` |

在当前 Windows 训练机上，默认输出分别位于
`outputs/aes_clean_continuation/` 和 `outputs/aes_hotflip_defense/`。

### 结果 JSON

| 文件 | 内容 |
|------|------|
| `hotflip_asr_undefended.json` | undefended 模型历史 HotFlip ASR 结果（1154 条） |
| `hotflip_asr_undefended_progress.json` | 同上，进度记录 |
| `hotflip_asr_defended.json` | 旧 v4 防御后历史 HotFlip ASR 结果（1154 条） |
| `hotflip_asr_defended_progress.json` | 同上，进度记录 |

### 核心代码（不在 whitebox/ 内）

| 文件 | 作用 |
|------|------|
| `text_scoring_adv_training/evaluation/aes/attacks/hotflip.py` | AES 专用 HotFlipAttack；候选文本重新编码并真实评分 |
| `text_scoring_adv_training/evaluation/aes/scorer.py` | AES DeBERTa 评分封装，统一使用 right padding |
| `text_scoring_adv_training/training/aes_trainer.py` | AES 专用在线对抗训练和单边虚高损失 |
| `text_scoring_adv_training/evaluation/robustness_tests/common/hotflip.py` | 原论文通用 HotFlip 辅助函数；保持不修改 |
| `text_scoring_adv_training/training/losses.py` | 原论文通用损失文件；保持不修改 |

---

## 当前 A → B → C 工作流

| 项目 | 值 |
|------|-----|
| A（开发机） | 修改、测试代码并 push 到 GitHub；不要求有 CUDA |
| B（代码中转） | GitHub 仓库 |
| C（训练机） | 从 GitHub pull 后执行训练与评估 |
| C 的 GPU | RTX 3090 24GB |
| C 的代码路径 | `E:\xjj\aes_scoring` |
| C 的 Conda 环境 | `xjj_aes` |
| B0 checkpoint | `E:\xjj\aes_scoring\deberta_checkpoints\fold0_best` |
| 训练集 | `E:\xjj\aes_scoring\data\train_fold0.csv`（16153 条） |
| benchmark | `E:\xjj\aes_scoring\data\valid_fold0.csv`（1154 条） |

`deberta_checkpoints/` 已被 `.gitignore` 排除，因此 B0 不会随 A → B → C
的 Git 流程传输。C 上需要事先保留上述 B0 目录。训练输出也只保留在 C 上，
不要把模型权重提交到 GitHub。

---

## 使用方法

以下命令全部在训练机 C 的 PowerShell 中执行。

### 1. 拉取代码并检查运行环境

```powershell
Set-Location E:\xjj\aes_scoring
git pull
conda activate xjj_aes
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0)); print('bf16:', torch.cuda.is_bf16_supported())"
Test-Path .\deberta_checkpoints\fold0_best
Test-Path .\data\train_fold0.csv
Test-Path .\data\valid_fold0.csv
```

GPU 名称应显示 RTX 3090，`bf16` 和三个 `Test-Path` 都应为 `True`。

### 2. dry-run 检查成对配置

```powershell
python .\whitebox\run_aes_clean_continuation_training.py --seed 42 --output-dir .\outputs\aes_clean_continuation_seed42 --dry-run
python .\whitebox\run_aes_hotflip_adv_training.py --seed 42 --output-dir .\outputs\aes_hotflip_defense_seed42 --dry-run
```

检查两份输出中的共享训练超参数完全一致；预期只有
`training_mode`、`use_hotflip_swaps` 和 `output_dir` 不同。

### 3. 先训练 C0，再训练 D_HOTFLIP

```powershell
python .\whitebox\run_aes_clean_continuation_training.py --seed 42 --output-dir .\outputs\aes_clean_continuation_seed42
python .\whitebox\run_aes_hotflip_adv_training.py --seed 42 --output-dir .\outputs\aes_hotflip_defense_seed42
```

不要在同一张 RTX 3090 上并行启动这两个训练。后续评估使用各目录下的
`best/`，不使用 `final/`。

### 4. 使用同一入口评估 B0、C0 和 D_HOTFLIP

```powershell
python .\whitebox\evaluate_aes_checkpoint.py --checkpoint .\deberta_checkpoints\fold0_best --out .\outputs\eval_b0_seed42 --seed 42
python .\whitebox\evaluate_aes_checkpoint.py --checkpoint .\outputs\aes_clean_continuation_seed42\best --out .\outputs\eval_c0_seed42 --seed 42
python .\whitebox\evaluate_aes_checkpoint.py --checkpoint .\outputs\aes_hotflip_defense_seed42\best --out .\outputs\eval_hotflip_defense_seed42 --seed 42
```

每个评估目录会包含 `clean_qwk.json`、`asr_summary.json`、
`hotflip_details.json` 和 `run_manifest.json`。

`run_hotflip_asr_undefended.py`、`generate_hotflip_train_adv_data.py` 和旧 v4
结果保留作历史追溯，不属于当前 B0/C0/D_HOTFLIP 主运行顺序。

### 5. 多随机种子复现

seed 42 验收通过后，再按相同顺序运行 seed 43 和 44。每个 seed 使用独立的
输出目录，并在同一个 seed 内保持 C0 与 D_HOTFLIP 配对。
