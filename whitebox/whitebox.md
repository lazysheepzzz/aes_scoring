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

## 结果

### Undefended（无防御 baseline）

| 指标 | 值 |
|------|-----|
| Victim | `/root/autodl-tmp/victim/fold0_best` |
| 测试集 | valid_fold0.csv（1154 条） |
| **ASR** | **87.78%** |
| avg_steps | 7.70（成功攻击平均迭代步数） |
| 攻击脚本 | `run_hotflip_valid_1154.py` |
| 结果文件 | `hotflip_valid_1154_result.json` |

### Defended（v4 对抗训练后）

| 指标 | 值 |
|------|-----|
| 模型 | v4 对抗训练（weight=2.0, margin=0.1, epochs=5） |
| 测试集 | valid_fold0.csv（1154 条） |
| **ASR** | **67.50%** |
| **QWK** | **0.8222**（undefended 约 0.854） |
| avg_steps | 7.35（成功攻击平均迭代步数） |
| 攻击脚本 | `eval_defended.py` |
| 结果文件 | `hotflip_defended_result.json` |
.
### 防御效果

- ASR 从 87.78% 降到 67.50%，下降 **20.3 个百分点**
- QWK 仅下降 0.032，评分准确性损失可接受
- 结论：对抗训练有效降低了 HotFlip 攻击成功率，评分系统未崩溃

---

## 文件说明

### 攻击脚本

| 文件 | 作用 |
|------|------|
| `run_hotflip_valid_1154.py` | 对 **undefended** victim 跑 HotFlip，输出 ASR 结果 |
| `eval_defended.py` | 对 **defended** 模型跑 HotFlip ASR 评估 |
| `generate_hotflip_train_adv_data.py` | 生成对抗训练用的 HotFlip 数据（JSONL 格式），用于离线对抗训练 |

### Python 启动脚本

| 文件 | 作用 |
|------|------|
| `run_aes_attacks.py` | 通用攻击入口（调用 `evaluation/aes/run_attacks.py`） |
| `run_aes_adv_v4.py` | v4 对抗训练入口；生成训练器支持的 JSON 配置 |
| `run_aes_eval_v4.py` | v4 clean QWK、MAE 和 HotFlip ASR 评估入口 |

三个入口均支持 `--help` 和 `--dry-run`。服务器上默认使用 `aes` Conda
环境，本地默认使用当前 Python 解释器。路径参数均可通过命令行覆盖。

### 结果 JSON

| 文件 | 内容 |
|------|------|
| `hotflip_valid_1154_result.json` | undefended 模型 HotFlip ASR 结果（1154 条） |
| `hotflip_valid_1154_progress.json` | 同上，进度记录 |
| `hotflip_defended_result.json` | v4 防御后 HotFlip ASR 结果（1154 条） |
| `hotflip_defended_progress.json` | 同上，进度记录 |

### 核心代码（不在 whitebox/ 内）

| 文件 | 作用 |
|------|------|
| `text_scoring_adv_training/evaluation/aes/attacks/hotflip.py` | HotFlipAttack 类（被 eval_defended.py 调用） |
| `text_scoring_adv_training/evaluation/robustness_tests/common/hotflip.py` | hotflip_pointwise 函数（被 aes_trainer.py 在线对抗训练调用） |
| `text_scoring_adv_training/training/aes_trainer.py` | 对抗训练主代码 |
| `text_scoring_adv_training/training/losses.py` | hinge_loss 定义 |

---

## 服务器环境

| 项目 | 值 |
|------|-----|
| 主机 | `ssh aes-gpu` → connect.nmb1.seetacloud.com:47837 |
| GPU | RTX 4090 24GB |
| Conda 环境 | aes |
| 代码路径 | `/root/autodl-tmp/robust_text_scoring/` |
| 数据路径 | `/root/autodl-tmp/data/valid_fold0.csv`（1154 条） |
| 对抗训练数据 | `/root/autodl-tmp/data/train_fold0.csv`（16153 条） |

---

## 使用方法

### 1. 跑 undefended ASR

```bash
cd /root/autodl-tmp/robust_text_scoring
conda run --no-capture-output -n aes python whitebox/run_hotflip_valid_1154.py
```

### 2. 跑 v4 对抗训练

```bash
cd /root/autodl-tmp/robust_text_scoring
python whitebox/run_aes_adv_v4.py
```

### 3. 跑 defended clean 指标和 HotFlip ASR

```bash
cd /root/autodl-tmp/robust_text_scoring
python whitebox/run_aes_eval_v4.py
```

只检查命令和参数，不启动训练与评估：

```bash
python whitebox/run_aes_adv_v4.py --dry-run
python whitebox/run_aes_eval_v4.py --dry-run
```
