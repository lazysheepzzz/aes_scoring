# Rudimentary 攻击 — AES 鲁棒性评估计划

## 概述

Rudimentary 是基于字符/词级随机编辑的攻击（拼写错误、词序乱、词重复等），属于规则型攻击，不需要梯度。参考论文实现，与 HotFlip 对齐形成完整攻防体系。

## 攻击参数（统一配置）

| 参数 | 值 | 说明 |
|------|-----|------|
| `n_steps` | 30 | 最大迭代步数 |
| `beam_size` | 1 | 贪心搜索 |
| `n_candidates` | 16 | 每步候选数 |
| `threshold` | 0.1 | 成功阈值（delta >= 0.1 才算攻击成功） |

操作类型（参考论文）：
- 字符级：随机替换、随机插入、随机删除、相邻交换
- 词级：词重复、词删除、相邻词交换

## 工作流程

### Phase 1: Undefended ASR（1154 全量）

**目标**：获得 undefended baseline 的 Rudimentary ASR

- 测试集：`valid_fold0.csv`（1154 条）
- Victim：`/root/autodl-tmp/victim/fold0_best`
- 脚本：直接使用服务器已有代码 `run_rudimentary_full1134_thresh.py`，仅修改 `N_ESSAYS = 1134 → 1154`
- 结果：`rudimentary_valid_1154_result.json`

> **说明**：已有代码逻辑正确无需改动，只需把样本数从 1134 改为 1154 跑全量即可。

**历史结果（1134条）**：ASR = 94.97%

### Phase 2: 对抗训练

**目标**：训练抵御 Rudimentary 攻击的模型

- 参考论文原版对抗训练方案（与 HotFlip v4 对齐）
- 训练脚本：`run_rudimentary_adv_v4.sh`（需新建，参考 HotFlip v4 训练框架）
- 对抗训练数据生成：`generate_rudimentary_train_adv_data.py`（需新建）
- 配置参数：weight=2.0, margin=0.1, epochs=5（与 HotFlip v4 对齐）
- 输出：`/root/autodl-tmp/rudimentary_adv_v4/`

> **说明**：对抗训练框架复用 `aes_trainer.py`，将 HotFlip 攻击替换为 Rudimentary 攻击。训练逻辑与 HotFlip v4 完全一致，仅攻击方式不同。

### Phase 3: Defended ASR（1154 全量）

**目标**：评估对抗训练后的 Rudimentary ASR

- 脚本：`eval_rudimentary_defended.py`（需新建，参考 `eval_defended.py`）
- 结果：`rudimentary_defended_result.json`

## 预期结果对比

| 模型 | QWK | Rudimentary ASR | avg_steps |
|------|-----|-----------------|-----------|
| undefended baseline | ~0.854 | ? | ? |
| **rudimentary defended** | **?** | **?** | **?** |

目标：ASR 显著下降，QWK 不崩溃。

---

## 文件清单（rudimentary/ 目录）

### 需新建的脚本（Phase 2-3）

| 文件 | 作用 |
|------|------|
| `generate_rudimentary_train_adv_data.py` | 生成 Rudimentary 对抗训练数据 |
| `run_rudimentary_adv_v4.sh` | Rudimentary 对抗训练启动脚本 |
| `eval_rudimentary_defended.py` | 对 defended 模型跑 Rudimentary ASR |

### 已有文件（服务器，已有代码逻辑正确）

| 文件 | 状态 | 说明 |
|------|------|------|
| `run_rudimentary_full1134_thresh.py` | 直接使用，仅改 N_ESSAYS=1154 | 攻击代码逻辑正确无需改动 |
| `rudimentary_unified_thresh_result.json` | 已有 1134 条结果 | ASR=94.97% |

### 结果 JSON（跑完后）

| 文件 | 内容 |
|------|------|
| `rudimentary_valid_1154_result.json` | undefended Rudimentary ASR |
| `rudimentary_valid_1154_progress.json` | 进度记录 |
| `rudimentary_defended_result.json` | defended Rudimentary ASR |
| `rudimentary_defended_progress.json` | 进度记录 |

---

## 服务器环境

| 项目 | 值 |
|------|-----|
| 主机 | `ssh aes-gpu` → connect.nmb1.seetacloud.com:47837 |
| GPU | RTX 4090 24GB |
| Conda 环境 | aes |
| 代码路径 | `/root/autodl-tmp/robust_text_scoring/` |
| 数据路径 | `/root/autodl-tmp/data/valid_fold0.csv`（1154 条） |

---

## 使用方法

### Phase 1: 跑 undefended ASR

```bash
# 直接修改服务器已有脚本的 N_ESSAYS，然后运行
source /root/miniconda3/etc/profile.d/conda.sh && conda activate aes
cd /root/autodl-tmp/robust_text_scoring
# 修改 run_rudimentary_full1134_thresh.py 中 N_ESSAYS=1134 → 1154
python rudimentary/run_rudimentary_full1134_thresh.py
# 结果输出到 /root/autodl-tmp/aes_final_run/rudimentary_valid_1154_result.json
```

### Phase 2: 对抗训练

```bash
bash /root/autodl-tmp/rudimentary_adv_v4.sh
```

### Phase 3: 跑 defended ASR

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate aes
cd /root/autodl-tmp/robust_text_scoring
python whitebox/rudimentary/eval_rudimentary_defended.py
```

---

## 注意事项

1. **Rudimentary 攻击不需要梯度**：与 HotFlip 不同，Rudimentary 是纯规则型攻击，不需要模型梯度信息
2. **对抗训练可复用 HotFlip v4 框架**：只需将 `hotflip_pointwise` 替换为 `RudimentaryAttack`
3. **对齐 HotFlip 实验设计**：保证训练数据、参数一致，便于对比

---

## 关键代码参考

### RudimentaryAttack 类（已有）

- 路径：`text_scoring_adv_training/evaluation/aes/attacks/rudimentary.py`
- 接口：`RudimentaryAttack(scorer, n_steps=30, n_candidates=16, threshold=0.1)`
- 方法：`attack(text) -> perturbed_text or None`

### 对抗训练改造要点

在 `aes_trainer.py` 中，将 HotFlip 替换为 Rudimentary：

```python
# 原来（HotFlip）
from text_scoring_adv_training.evaluation.robustness_tests.common.hotflip import hotflip_pointwise

# 改为（Rudimentary）
from text_scoring_adv_training.evaluation.aes.attacks.rudimentary import RudimentaryAttack

# train_step 中生成对抗文本
atk = RudimentaryAttack(scorer, n_steps=30, n_candidates=16, threshold=0.1)
perturbed = atk.attack(text)  # 返回扰动后的文本
```

### 训练配置 JSON（rudimentary_v4_config.json）

```json
{
  "checkpoint_path": "/root/autodl-tmp/victim/fold0_best",
  "train_csv": "/root/autodl-tmp/data/train_fold0.csv",
  "valid_csv": "/root/autodl-tmp/data/valid_fold0.csv",
  "output_dir": "/root/autodl-tmp/rudimentary_adv_v4",
  "num_epochs": 5,
  "per_device_train_batch_size": 4,
  "gradient_accumulation_steps": 8,
  "learning_rate": 2e-5,
  "max_length": 1024,
  "seed": 42,
  "eval_every": 200,
  "save_every": 1000,
  "use_rudimentary_swaps": true,
  "rudimentary_weight": 2.0,
  "rudimentary_n_candidates": 16,
  "rudimentary_n_steps": 30,
  "rudimentary_threshold": 0.1,
  "rudimentary_margin": 0.1
}
```

> 注意：`aes_trainer.py` 目前只有 `use_hotflip_swaps`，需要新增 `use_rudimentary_swaps` 选项，复用同一套训练框架。
