# Rudimentary Attack — Full 1134 Essays (统一搜索策略 + 阈值0.1)

> 历史记录：本文件不是当前正式协议。当前代码库中的旧 JSON 实际为 1154
> 篇，但缺少固定 seed、10% 编辑预算、攻击文本和运行清单。正式方案与命令见
> `rudimentary/PLAN.md`。

## 参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `n_steps` | 30 | 最大迭代步数 |
| `n_candidates` | 16 | 每步生成候选数 |
| `threshold` | 0.1 | 成功阈值（delta >= 0.1 才停） |
| 停止条件 | `best_score - original_score >= threshold` | 达到阈值即停 |
| 评分方式 | **Batch scoring**（batch_size=32） | 每步候选批量评分，GPU 利用率高 |

## 搜索策略

- **统一为贪心搜索**（beam=1），与 HotFlip 一致
- **Batch scoring**：每步 16 个候选一次性批量评分
- **停止条件**：达到阈值 0.1 才停，记录首次达到阈值的步数

## 最终结果

| 指标 | 值 |
|------|-----|
| **ASR** | **94.97% (1077/1134)** |
| avg Δ score | +0.116 |
| 耗时 | 37.5 min |

## 文件

- 运行脚本: `rudimentary/run_rudimentary_full1134_thresh.py`
- 结果文件: `rudimentary_unified_thresh_result.json`
- 攻击类: `text_scoring_adv_training/evaluation/aes/attacks/rudimentary.py`

## 服务器环境

| 项目 | 值 |
|------|-----|
| 主机 | root@connect.nmb1.seetacloud.com:47837 |
| GPU | RTX 4090 24GB |
| Conda 环境 | aes |
| Victim | /root/autodl-tmp/victim/fold0_best |
| 数据 | /root/autodl-tmp/data/valid_fold0.csv |
