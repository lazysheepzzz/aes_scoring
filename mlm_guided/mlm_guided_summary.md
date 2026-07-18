# MLM-Guided Attack — Full 1134 Essays (统一搜索策略 + 阈值0.1)

## 候选生成层

基于 ModernBERT-large 的 MLM 概率引导替换攻击。核心修复：DeBERTa tokenizer 词表（128003）与 ModernBERT 词表（50368）不兼容，在传入 MLM 前将所有 >= 50368 的 token ID 替换为 pad_id。

| 参数 | 值 | 说明 |
|------|-----|------|
| `n_sample_pos` | 8 | 每步采样 8 个位置（与 HotFlip 一致）|
| `top_k_per_pos` | 2 | 每位置 top-2 候选（与 HotFlip 一致）|
| `max_candidates_per_step` | 16 | 每步最多候选数（8×2=16，与统一搜索策略一致）|
| `n_steps` | 30 | 最大迭代步数 |
| `threshold` | 0.1 | 成功阈值（delta >= 0.1 才停）|
| MLM 骨干 | ModernBERT-large | 论文推荐，float32 |
| MLM_VOCAB | 50368 | ModernBERT-large 词表大小 |

## 搜索策略层（统一）

| 参数 | 三个攻击一致 |
|------|------------|
| beam_size | 1（贪心）|
| max_candidates_per_step | 16 |
| n_steps | 30 |
| threshold | 0.1 |

- **Batch scoring**：每步候选分批评分（batch_size=32）
- **停止条件**：达到阈值 0.1 才停，记录首次达到阈值的步数

## 最终结果

| 指标 | 值 |
|------|-----|
| **ASR** | **75.66% (858/1134)** |
| avg Δ score | +0.108 |
| 耗时 | 54.1 min |

## 三攻击对比

| 攻击 | ASR | 成功/总数 | avg Δ | 耗时 |
|------|-----|---------|-------|------|
| HotFlip | 88.62% | 1005/1134 | +0.129 | 30.2min |
| Rudimentary | 94.97% | 1077/1134 | +0.116 | 37.5min |
| **MLM-guided** | **75.66%** | 858/1134 | +0.108 | 54.1min |

## 候选生成层参数对比

| 参数 | HotFlip | Rudimentary | MLM-guided |
|------|---------|------------|-----------|
| n_sample_pos | 8 | —（随机生成）| 8 |
| top_k_per_pos | 2 | — | 2 |
| 实际候选/步 | 16 | 16 | 16 ✅ |

## 文件

- 运行脚本: `mlm_guided/run_mlm_guided_full1134_thresh.py`
- 结果文件: `mlm_guided/mlm_guided_unified_thresh_result.json`
- 攻击类: `text_scoring_adv_training/evaluation/aes/attacks/mlm_guided.py`

## 服务器环境

| 项目 | 值 |
|------|-----|
| 主机 | root@connect.nmb1.seetacloud.com:47837 |
| GPU | RTX 4090 24GB |
| Conda 环境 | aes |
| Victim | /root/autodl-tmp/victim/fold0_best |
| 数据 | /root/autodl-tmp/data/valid_fold0.csv |
