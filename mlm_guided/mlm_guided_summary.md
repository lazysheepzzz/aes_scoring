# AES MLM-guided 主实验说明

## 状态

旧的 `mlm_guided_unified_thresh_result.json`（1,134 篇，ASR 75.66%）仅作
错误分析记录，不能进入主实验表。旧实现把 DeBERTa token ID 直接送入
ModernBERT，再把 ModernBERT ID 写回 DeBERTa 序列；不同 tokenizer 中相同的
整数不表示相同 token。

正式实现位于：

- 攻击：`text_scoring_adv_training/evaluation/aes/attacks/mlm_guided.py`
- 正式评估：`mlm_guided/evaluate_aes_mlm_guided.py`
- 训练候选缓存：`mlm_guided/prepare_aes_mlm_training_candidates.py`
- D_MLM 训练：`mlm_guided/run_aes_mlm_guided_adv_training.py`
- checkpoint 选择：`mlm_guided/select_aes_mlm_guided_defense_checkpoint.py`

原论文通用 MLM 源码
`text_scoring_adv_training/evaluation/robustness_tests/common/mlm.py` 保持只读；
AES 专用代码只调用它生成 ModernBERT 候选。

## 正式协议

```text
作文文本
→ ModernBERT tokenizer
→ mask 并生成 ModernBERT candidate IDs
→ ModernBERT decode 为完整候选文本
→ all-MiniLM-L6-v2 语义过滤（cosine >= 0.90）
→ DeBERTa tokenizer 独立重新编码
→ AES victim 真实打分
```

| 参数 | 值 |
|---|---:|
| MLM | `answerdotai/ModernBERT-large` |
| 语义模型 | `sentence-transformers/all-MiniLM-L6-v2` |
| 最大步数 | 30 |
| 每步采样位置 | 8 |
| 每位置候选 | 2 |
| 每步最多真实评分候选 | 16 |
| beam size | 1 |
| 成功阈值 | delta >= 0.10 |
| 最大编辑预算 | 原始 DeBERTa token 数的 5% |
| 随机种子 | 42 |
| 正式样本数 | 1,154 |

训练候选只做 1 步攻击：每篇作文固定采样 1 个位置，取 MLM top-16，经语义
过滤后缓存替换规格。该候选生成与 victim 参数无关，因此只需离线执行一次。
训练的每次前向仍由当前 DeBERTa 在候选池中选择真实分数最高的候选，保持
model-aware。训练时不加载 ModernBERT 或 MiniLM。

优化 `plan.md` 中预先规定的质量保持损失：对抗文本贴近真实标签，同时惩罚其
相对 clean prediction 的分数虚高。训练数据、优化器、epoch、batch size、学习率
和 C0/其他防御保持对齐。

## 第一次运行

先联网下载并缓存 ModernBERT 和 MiniLM，同时只跑 2 篇 smoke test：

```powershell
python .\mlm_guided\evaluate_aes_mlm_guided.py `
  --checkpoint .\deberta_checkpoints\fold0_best `
  --out .\outputs\smoke_mlm_b0_seed42 `
  --n-essays 2 --seed 42 --skip-clean --online
```

smoke 成功后，后续命令不再带 `--online`，使用本地缓存。

## D_MLM 运行顺序

先一次性批量生成训练候选。该过程支持中断续跑；不要对续跑命令使用
`--force`：

```powershell
python .\mlm_guided\prepare_aes_mlm_training_candidates.py `
  --seed 42 --batch-size 8
```

缓存完成后训练；训练阶段只加载 DeBERTa 和 tokenizer：

```powershell
python .\mlm_guided\run_aes_mlm_guided_adv_training.py `
  --seed 42 `
  --output-dir .\outputs\aes_mlm_guided_defense_cached_seed42
```
