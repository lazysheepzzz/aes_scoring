# 英语作文自动评分鲁棒性研究执行计划

## 0. 文档信息

- 项目名称：英语作文自动评分中的虚高攻击、跨攻击迁移与组合对抗训练
- 基础框架：《Unifying Adversarial Robustness and Training Across Text Scoring Models》
- 当前版本：1.0
- 建立日期：2026-07-27
- 维护方式：每次修改实验协议、参数、数据划分、指标定义后，先更新本文件，再运行实验
- 当前结论：仓库内已有结果统一标记为预实验；修复 padding、MLM tokenizer、训练循环、评估口径后产生的结果标记为正式实验

---

## 1. 研究目标

本项目研究低质量英语作文通过词汇替换、字符编辑、关键词注入、模板注入、无关内容注入、句子复写获得虚高分数的问题。

正式研究主线固定为：

```text
未防御评分模型
→ 六类作文虚高攻击
→ 单攻击对抗训练
→ 跨攻击迁移评估
→ 组合对抗训练
→ 正常评分性能评估
→ 低分作文专项分析
```

核心研究问题：

1. 未防御 DeBERTa 对六类虚高攻击的连续分数敏感性、等级提升风险、超过真实等级风险、排序翻转风险有多高。
2. 针对单一攻击训练的防御能否迁移到其他攻击。
3. 组合对抗训练能否获得更低的宏平均攻击成功率。
4. 防御能否在保持 clean QWK、MAE、RMSE 的前提下保护真实分数为 1–3 的低质量作文。

---

## 2. 当前工作定位

### 2.1 可保留的预实验资产

- `data/train.csv`：17,307 篇作文。
- `data/train_fold0.csv`：16,153 篇。
- `data/valid_fold0.csv`：1,154 篇。
- `deberta_checkpoints/fold0_best/`：当前未防御 DeBERTa checkpoint。
- Rudimentary 预实验：1,134 篇，`ASRΔ@0.1 = 94.97%`。
- HotFlip 预实验：1,154 篇，`ASRΔ@0.1 = 87.78%`。
- MLM-guided 预实验：结果仅保留作问题记录，不进入正式论文表格。
- HotFlip defended 预实验：结果仅保留作问题记录，不进入正式论文主结论。

### 2.2 必须重跑的实验

- MLM-guided 未防御攻击。
- Injection 未防御攻击。
- 修复后统一协议下的全部六类攻击。
- Clean continued control。
- HotFlip 单攻击防御。
- 其他单攻击防御。
- 跨攻击迁移矩阵。
- 组合对抗训练。
- 所有正式 clean performance 评估。

---

## 3. 正式数据协议

### 3.1 数据源

使用以下两个带完整元数据的文件合并为正式全量数据：

```text
data/train_fold0.csv
data/valid_fold0.csv
```

合并后总数固定为 17,307。合并时执行：

1. 按 `essay_id` 检查唯一性。
2. 按 `full_text` 检查重复正文。
3. 检查 `score` 范围为 1–6。
4. 检查 `prompt_name` 非空。
5. 输出数据检查报告 `reports/data_split_report.json`。

### 3.2 正式划分

使用：

```python
StratifiedKFold(
    n_splits=8,
    shuffle=True,
    random_state=42,
)
```

分层标签固定为：

```text
prompt_name + "|" + score
```

划分定义：

- fold 0：test。
- fold 1：dev。
- fold 2–7：train。

固定规模：

- train：12,979。
- dev：2,164。
- test：2,164。

划分生成后保存：

```text
data/formal/train.csv
data/formal/dev.csv
data/formal/test.csv
data/formal/split_manifest.json
```

正式 test 在所有模型训练、超参数选择、阈值选择完成前不得用于模型选择。

### 3.3 低质量作文定义

固定分组：

- 低质量：真实分数 1–2。
- 中等质量：真实分数 3–4。
- 高质量：真实分数 5–6。

所有攻击结果必须报告总体结果以及三个质量分组结果。

---

## 4. 模型组

### 4.1 第一阶段基线模型

模型名称：

```text
B0_BASE
```

从 `microsoft/deberta-v3-base` 训练，使用 clean MSE。B0 用于：

- 生成未防御攻击基线。
- 作为第二阶段全部模型的共同初始化 checkpoint。
- 在 dev 上生成共享等级阈值。

### 4.2 第二阶段对照与防御模型

全部模型从同一个 B0 checkpoint 初始化：

| 模型名 | 训练内容 |
|---|---|
| C0_CLEAN_CONT | 仅继续 clean MSE 训练 |
| D_RUDI | Rudimentary 对抗训练 |
| D_HOTFLIP | HotFlip 对抗训练 |
| D_MLM | MLM-guided 对抗训练 |
| D_INJECT | 无关内容与自我复写对抗训练 |
| D_KEYWORD | 关键词注入对抗训练 |
| D_TEMPLATE | 模板注入对抗训练 |
| D_COMBINED | 六类攻击的组合对抗训练 |

主要防御比较使用：

```text
C0_CLEAN_CONT vs D_*
```

次要部署比较使用：

```text
B0_BASE vs D_*
```

这样可以分离“继续训练带来的变化”和“对抗损失带来的变化”。

### 4.3 随机种子

正式训练运行三个随机种子：

```text
42
43
44
```

数据划分始终使用 seed 42。模型初始化、dataloader、攻击采样分别使用当前训练 seed。

---

## 5. 模型训练参数

### 5.1 B0 基线训练

| 参数 | 固定值 |
|---|---:|
| model_name | microsoft/deberta-v3-base |
| max_length | 1024 |
| padding_side | right |
| label | score - 1 |
| label_range | 0–5 |
| loss | MSE |
| epochs | 3 |
| train_batch_size | 4 |
| gradient_accumulation_steps | 4 |
| effective_batch_size | 16 |
| learning_rate | 2e-5 |
| weight_decay | 0.01 |
| warmup_ratio | 0.0 |
| scheduler | linear |
| bf16 | true |
| fp16 | false |
| attention_dropout | 0.0 |
| hidden_dropout | 0.0 |
| classifier_dropout | 0.0 |
| gradient_clip_norm | 1.0 |
| eval_every_optimizer_steps | 100 |
| metric_for_best_model | clean dev QWK |

### 5.2 第二阶段统一训练参数

以下参数对 C0 和全部 D_* 模型完全一致：

| 参数 | 固定值 |
|---|---:|
| initialization | B0 best checkpoint |
| max_length | 1024 |
| padding_side | right |
| epochs | 3 |
| train_batch_size | 4 |
| gradient_accumulation_steps | 8 |
| effective_batch_size | 32 |
| learning_rate | 1e-5 |
| weight_decay | 0.01 |
| warmup_ratio | 0.1 |
| scheduler | linear |
| bf16 | true |
| gradient_clip_norm | 1.0 |
| eval_every_optimizer_steps | 200 |
| save_every_optimizer_steps | 200 |
| adversarial_fraction | 0.5 |
| clean_loss_weight | 1.0 |
| adversarial_loss_weight | 1.0 |

每个 batch 含 4 篇作文，其中固定选择 2 篇生成 adversarial example。C0 仍使用相同 batch、累积步数、epoch、优化器与 scheduler。

---

## 6. 损失函数

### 6.1 Clean loss

模型输出为 0–5 连续回归分数：

\[
s_c = f_\theta(x)
\]

\[
L_{clean} = \operatorname{MSE}(s_c, y)
\]

其中：

```text
y = score - 1
```

### 6.2 统一容忍阈值

连续回归允许微小变化，固定容忍阈值：

\[
\tau = 0.05
\]

所有相对分数损失中的 clean score 使用：

\[
\operatorname{stopgrad}(s_c)
\]

避免相对损失直接把 clean score 向上推高。

### 6.3 质量保持攻击损失

MLM-guided 候选通过语义约束后视为质量保持攻击。

\[
L_{same}
=
\operatorname{SmoothL1}(s_a,y;\beta=0.1)
+
0.5\left[\max(0,s_a-\operatorname{stopgrad}(s_c)-\tau)\right]^2
\]

该损失要求：

- 攻击作文仍接近真实分数。
- 攻击作文不能相对 clean prediction 明显上涨。

### 6.4 质量下降攻击损失

Rudimentary、HotFlip、Injection、Keyword、Template 统一使用单边虚高惩罚：

\[
L_{degrade}
=
\left[\max(0,s_a-y-\tau)\right]^2
+
0.5\left[\max(0,s_a-\operatorname{stopgrad}(s_c)-\tau)\right]^2
\]

该损失不强制攻击作文获得更低的固定标签，只惩罚超过真实分数以及超过 clean prediction 的虚高行为。

### 6.5 单攻击训练总损失

MLM：

\[
L = L_{clean} + L_{same}
\]

其他五类攻击：

\[
L = L_{clean} + L_{degrade}
\]

### 6.6 组合训练总损失

D_COMBINED 每个 optimizer step 只启用一种攻击，攻击顺序固定循环：

```text
Rudimentary
HotFlip
MLM-guided
Injection
Keyword
Template
```

单个 step 的总损失沿用对应攻击的单攻击总损失。六个连续 optimizer step 构成一个完整攻击周期。该设计保持总 adversarial loss 权重为 1.0，防止组合模型仅因损失项数量增加而获得更大梯度。

### 6.7 损失函数消融

主实验完成后执行以下消融：

| 消融名 | 设置 |
|---|---|
| ABL_WEIGHT_05 | adversarial_loss_weight = 0.5 |
| ABL_WEIGHT_10 | adversarial_loss_weight = 1.0 |
| ABL_WEIGHT_20 | adversarial_loss_weight = 2.0 |
| ABL_TAU_00 | tau = 0.0 |
| ABL_TAU_05 | tau = 0.05 |
| ABL_TAU_10 | tau = 0.1 |
| ABL_PAIRWISE_OLD | 当前 clean-vs-adv squared hinge |
| ABL_NO_RELATIVE | 移除相对分数惩罚，仅保留 gold anchored loss |

消融先在 seed 42 上运行。确定主设置后，再运行 seed 43、44。

---

## 7. 正式攻击定义

### 7.1 统一搜索规则

所有攻击遵守：

- scorer 使用 `model.eval()`。
- tokenizer 使用 right padding。
- beam size 固定为 1。
- 每步候选数固定为 16。
- 保存每一步当前最高分。
- 正式评估不在 `delta = 0.1` 时提前停止。
- 达到攻击预算后返回全程最高分文本。
- 候选文本先规范化为字符串，再由 DeBERTa tokenizer 重新编码和评分。
- 结果保存作文 ID、真实分数、prompt、原文、攻击文本、编辑记录、每步分数。

### 7.2 Rudimentary

攻击操作：

- 字符替换。
- 字符插入。
- 字符删除。
- 相邻字符交换。
- 单词重复。
- 单词删除。
- 相邻单词交换。

评估参数：

| 参数 | 值 |
|---|---:|
| max_steps | 30 |
| candidates_per_step | 16 |
| beam_size | 1 |
| max_token_edit_rate | 0.10 |

训练参数：

| 参数 | 值 |
|---|---:|
| attack_steps_per_sample | 1 |
| candidates | 16 |

### 7.3 HotFlip

评估参数：

| 参数 | 值 |
|---|---:|
| max_steps | 30 |
| sampled_positions | 8 |
| top_k_per_position | 2 |
| candidates_per_step | 16 |
| beam_size | 1 |
| max_token_edit_rate | 0.10 |

每一步对 16 个候选做真实模型评分，选择实际分数最高的候选，不直接采用梯度近似排序的第一个候选。

训练参数：

| 参数 | 值 |
|---|---:|
| attack_steps_per_sample | 1 |
| sampled_positions | 8 |
| top_k_per_position | 2 |
| actual_scoring_candidates | 16 |

### 7.4 MLM-guided

MLM 模型固定为：

```text
answerdotai/ModernBERT-large
```

正确处理流程：

```text
文本
→ ModernBERT tokenizer
→ mask candidate
→ ModernBERT candidate IDs
→ ModernBERT decode
→ 完整候选文本
→ DeBERTa tokenizer
→ AES score
```

禁止在 ModernBERT 和 DeBERTa 之间直接共享 token ID。

评估参数：

| 参数 | 值 |
|---|---:|
| max_steps | 30 |
| sampled_positions | 8 |
| top_k_per_position | 2 |
| candidates_per_step | 16 |
| sentence_similarity_model | sentence-transformers/all-MiniLM-L6-v2 |
| minimum_cosine_similarity | 0.90 |
| max_token_edit_rate | 0.05 |

训练参数：

| 参数 | 值 |
|---|---:|
| attack_steps_per_sample | 1 |
| candidates | 16 |
| minimum_cosine_similarity | 0.90 |

### 7.5 Injection

包含两个固定子类：

```text
INJECT_EXTERNAL
INJECT_SELF_DUP
```

External 句子来自版本化资源：

```text
data/attack_resources/wikipedia_sentences_100.txt
```

使用 `all-MiniLM-L6-v2` 计算句子与作文 prompt、作文正文的相似度。候选必须同时满足：

```text
similarity(sentence, prompt) <= 0.20
similarity(sentence, essay) <= 0.20
```

评估参数：

| 参数 | 值 |
|---|---:|
| max_inserted_sentences | 3 |
| candidates_per_step | 16 |
| positions | start, middle, end |
| max_added_token_ratio | 0.20 |

训练时每篇 adversarial essay 只插入 1 个句子。

### 7.6 Keyword injection

关键词只从 formal train 中提取。每个 prompt 分别计算高分作文与低分作文的 TF-IDF 差异：

```text
高分组：score 5–6
低分组：score 1–2
```

每个 prompt 保存排名前 20 的高分关联关键词：

```text
data/attack_resources/keyword_bank.json
```

评估参数：

| 参数 | 值 |
|---|---:|
| max_inserted_keywords | 5 |
| candidates_per_step | 16 |
| positions | start, middle, end |
| duplicate_keyword_allowed | false |

每一步插入一个尚未出现的关键词。攻击文本不得增加与题目有关的新论据，只增加关键词表面信号。

训练时每篇 adversarial essay 插入 1 个关键词。

### 7.7 Template injection

建立 20 条人工审核的通用结构模板，资源文件：

```text
data/attack_resources/template_bank.json
```

模板只包含组织性表达，不包含具体论点和事实。模板类型固定为：

- 引言模板。
- 论点过渡模板。
- 举例模板。
- 结论模板。

评估参数：

| 参数 | 值 |
|---|---:|
| max_inserted_templates | 3 |
| candidates_per_step | 16 |
| positions | start, paragraph_boundary, end |
| max_added_token_ratio | 0.20 |

训练时每篇 adversarial essay 插入 1 条模板。

---

## 8. 评估指标

### 8.1 连续分数指标

定义：

\[
\Delta=s_{adv}-s_{clean}
\]

固定报告：

```text
Mean Δ
Median Δ
P90 Δ
ASRΔ@0.05
ASRΔ@0.10
ASRΔ@0.20
ASRΔ@0.50
```

已有的 `delta >= 0.1` 结果统一命名为 `ASRΔ@0.10`。

### 8.2 等级阈值

使用 B0 seed 42 在 clean dev 上优化五个单调阈值，将 0–5 回归输出映射到 1–6 等级。该阈值保存为：

```text
artifacts/calibration/shared_grade_thresholds.json
```

所有模型、所有攻击、三个训练 seed 使用同一套共享阈值计算正式等级攻击指标。

### 8.3 Grade Promotion ASR

\[
ASR_{grade}
=
P(g(s_{adv})>g(s_{clean}))
\]

只统计攻击导致的向上跨级。向下跨级不计为攻击成功。

### 8.4 Overgrade ASR

eligible 样本定义：

\[
g(s_{clean})\le y_{grade}
\]

成功定义：

\[
g(s_{adv})>y_{grade}
\]

\[
ASR_{overgrade}
=
\frac{\#success}{\#eligible}
\]

该指标必须单独报告真实分数 1–2 的结果。

### 8.5 Rank-flip ASR

在相同 `prompt_name` 内构造作文对：

```text
low essay：score 1–2
high essay：score 4–6
```

配对规则：

1. 在相同 prompt 内匹配。
2. 选择 word count 最接近的 high essay。
3. 每篇 high essay 只使用一次。
4. 只保留 clean 模型排序正确的 pair。

成功定义：

\[
s(x_{low}^{adv})\ge s(x_{high})
\]

该指标与原论文的排序翻转失败条件直接对应。

### 8.6 正常评分性能

每个 checkpoint 在 clean test 上报告：

```text
QWK
MAE
RMSE
Rounded QWK
Shared-threshold QWK
score 1–2 MAE
score 3–4 MAE
score 5–6 MAE
```

### 8.7 文本质量与攻击预算

每类攻击报告：

```text
平均字符编辑距离
平均 token edit rate
平均新增 token 比例
平均语义相似度
平均成功编辑步数
平均完整搜索步数
```

每类攻击从 test 中固定抽取 50 个成功样本做人工检查。人工检查记录：

- 语义是否保持。
- 语法质量是否改善。
- 语法质量是否下降。
- 是否引入新论据。
- 是否仍属于低质量作文。

---

## 9. 实验矩阵

### 9.1 未防御攻击矩阵

对 B0 运行：

```text
Rudimentary
HotFlip
MLM-guided
Injection External
Injection Self-Dup
Keyword
Template
```

Injection 两个子类分别报告，同时汇总为 Injection family。

### 9.2 跨攻击迁移矩阵

行是模型，列是测试攻击：

| Model | Rudi | HotFlip | MLM | Inject | Keyword | Template | Clean |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0_BASE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| C0_CLEAN_CONT | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| D_RUDI | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| D_HOTFLIP | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| D_MLM | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| D_INJECT | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| D_KEYWORD | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| D_TEMPLATE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| D_COMBINED | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

每个单元格至少报告：

```text
ASRΔ@0.10
Grade-ASR
Overgrade-ASR
Rank-flip ASR
Mean Δ
```

### 9.3 汇总指标

对六类攻击计算：

```text
Macro ASRΔ@0.10
Macro Grade-ASR
Macro Overgrade-ASR
Macro Rank-flip ASR
Worst-case ASRΔ@0.10
```

Injection family 使用两个子类的平均值后再进入六类宏平均，防止 Injection 因子类数量多而获得更高权重。

---

## 10. 模型选择规则

### 10.1 C0

C0 使用 clean dev QWK 最高的 checkpoint。

### 10.2 防御模型

为每个训练 seed 固定一个 256 篇的 dev robustness subset，按 `prompt_name + score` 分层抽样，抽样 seed 为 42。

每个保存 checkpoint 执行：

1. 计算完整 clean dev QWK。
2. 在 robustness subset 上执行对应训练攻击，评估预算为正式预算的三分之一。
3. 只保留 `QWK >= C0_QWK - 0.02` 的 checkpoint。
4. 在合格 checkpoint 中选择 `ASRΔ@0.10` 最低者。
5. 出现相同 ASR 时选择 QWK 更高者。

D_COMBINED 在 subset 上计算六类攻击的 Macro ASRΔ@0.10，并使用相同规则选择 checkpoint。

test 只在 checkpoint 选择完成后运行一次。

---

## 11. 预期结果与验收标准

以下数值是项目验收阈值，不代表提前承诺实验结果。

### 11.1 基线复现

| 指标 | 验收标准 |
|---|---:|
| B0 clean QWK | ≥ 0.83 |
| B0 MAE | ≤ 0.42 |
| 三个 seed 的 QWK 标准差 | ≤ 0.015 |

### 11.2 Clean continued control

| 指标 | 验收标准 |
|---|---:|
| C0 相对 B0 QWK 下降 | ≤ 0.01 |
| C0 相对 B0 Macro ASRΔ@0.10 变化 | 绝对值 ≤ 5 个百分点 |

### 11.3 单攻击防御

| 指标 | 验收标准 |
|---|---:|
| 目标攻击 ASRΔ@0.10 下降 | ≥ 15 个百分点 |
| 目标攻击 Grade-ASR 下降 | ≥ 10 个百分点 |
| 非目标攻击 Macro ASRΔ@0.10 下降 | ≥ 5 个百分点 |
| clean QWK 相对 C0 下降 | ≤ 0.02 |

### 11.4 组合防御

| 指标 | 验收标准 |
|---|---:|
| Macro ASRΔ@0.10 相对 C0 下降 | ≥ 15 个百分点 |
| Macro Grade-ASR 相对 C0 下降 | ≥ 10 个百分点 |
| 低质量 Overgrade-ASR 下降 | ≥ 15 个百分点 |
| Macro Rank-flip ASR 下降 | ≥ 10 个百分点 |
| clean QWK 相对 C0 下降 | ≤ 0.02 |
| 六类攻击中出现明显退化的数量 | 0 |

### 11.5 预期方向

- Rudimentary 单攻击防御在 Rudimentary 上取得最大单项收益。
- HotFlip 单攻击防御对 HotFlip 有明显收益，对 Injection 的迁移收益较小。
- MLM 单攻击防御主要改善词汇替换鲁棒性。
- Injection、Keyword、Template 防御对长度和表面结构信号具有互补作用。
- D_COMBINED 获得最低的宏平均攻击成功率。
- C0 证明继续训练本身不能解释全部鲁棒性收益。
- 低质量作文的防御收益小于总体收益时，继续调整 gold anchored loss 权重。

---

## 12. 代码修复任务

### P0：正式实验前必须完成

- [ ] 将 AES DeBERTa tokenizer 改为 right padding。
- [ ] 新增测试：同一短文本单独评分与混合长度 batch 评分差异小于 `1e-5`。
- [ ] 修复 MLM tokenizer 流程，禁止跨 tokenizer 共享 ID。
- [ ] 新增 MLM 测试：候选文本 decode 后可由 DeBERTa 独立重新编码。
- [ ] 统一 `aes_trainer.py` CLI 与 JSON config。
- [ ] 修复梯度累积尾部 optimizer step。
- [ ] 防止同一 global step 重复 eval 和 save。
- [ ] 修复 HotFlip 左 padding span。
- [ ] 让 `hotflip_max_candidates` 实际生效。
- [ ] HotFlip 候选必须经过真实评分后再选择。
- [ ] 统一 `ASRΔ@0.10`、Grade-ASR、Overgrade-ASR、Rank-flip ASR。
- [ ] band crossing 只统计向上跨级。
- [ ] 所有攻击设置随机种子。
- [ ] 所有输出目录由程序创建。

### P1：攻击闭环

- [ ] 完成正式数据划分脚本。
- [ ] 完成共享等级阈值优化脚本。
- [ ] 完成 Keyword bank 生成脚本。
- [ ] 完成 Template bank。
- [ ] 完成 Injection 相关性过滤。
- [ ] 统一六类攻击的结果 schema。
- [ ] 保存攻击后的完整文本和编辑历史。

### P2：训练闭环

- [ ] 实现 C0 clean continued control。
- [ ] 实现六个单攻击 trainer config。
- [ ] 实现 D_COMBINED 平衡攻击调度器。
- [ ] 实现 dev robustness subset checkpoint 选择。
- [ ] 保存 optimizer、scheduler、training state。

### P3：评估与报告

- [ ] 运行 B0 六攻击基线。
- [ ] 运行单攻击防御矩阵。
- [ ] 运行跨攻击迁移矩阵。
- [ ] 运行组合防御矩阵。
- [ ] 运行三个 seed。
- [ ] 计算置信区间。
- [ ] 完成低分作文专项表。
- [ ] 完成人工质量检查。

---

## 13. 结果与产物目录规范

正式代码和产物使用：

```text
configs/aes/
├─ base.json
├─ clean_cont.json
├─ adv_rudimentary.json
├─ adv_hotflip.json
├─ adv_mlm.json
├─ adv_injection.json
├─ adv_keyword.json
├─ adv_template.json
└─ adv_combined.json

data/formal/
data/attack_resources/
artifacts/calibration/

outputs/
└─ <model_name>/
   └─ seed_<seed>/
      ├─ checkpoints/
      ├─ train_log.jsonl
      ├─ config_resolved.json
      ├─ environment.json
      └─ best_checkpoint.json

results/
└─ <model_name>/
   └─ seed_<seed>/
      ├─ clean_metrics.json
      ├─ attack_summary.json
      ├─ attack_details.jsonl
      ├─ low_score_summary.json
      └─ run_manifest.json
```

每个 `run_manifest.json` 必须记录：

```text
git commit
model checkpoint
data split hash
attack resource hash
training seed
attack seed
完整参数
Python 版本
PyTorch 版本
Transformers 版本
CUDA 版本
GPU 型号
开始时间
结束时间
运行状态
```

---

## 14. 执行顺序

### Phase 0：修复与测试

1. 完成 P0。
2. 在本地 CPU 上通过 scorer batch 一致性测试。
3. 在 GPU 上用 16 篇作文完成六类攻击 smoke test。
4. 检查所有结果字段和文本记录。

完成标准：六类攻击均可运行，batch 分数一致，MLM 候选可读，结果 schema 相同。

### Phase 1：数据与 B0

1. 生成 formal train/dev/test。
2. 训练 B0 seed 42。
3. 达到 QWK 验收标准后训练 seed 43、44。
4. 在 B0 seed 42 dev 上生成共享等级阈值。
5. 冻结数据与阈值文件 hash。

### Phase 2：未防御攻击

1. 在 B0 三个 seed 上运行六类攻击。
2. 生成连续、等级、overgrade、rank-flip 指标。
3. 完成低质量作文分组。
4. 完成每类 50 个样本的人工检查。

### Phase 3：对照与单攻击防御

1. 训练 C0。
2. 训练 D_RUDI。
3. 训练 D_HOTFLIP。
4. 训练 D_MLM。
5. 训练 D_INJECT。
6. 训练 D_KEYWORD。
7. 训练 D_TEMPLATE。
8. 对每个模型运行完整跨攻击矩阵。

### Phase 4：组合训练

1. 训练 D_COMBINED seed 42。
2. 完成损失权重与 tau 消融。
3. 固定组合训练主配置。
4. 训练 D_COMBINED seed 43、44。
5. 运行完整评估矩阵。

### Phase 5：统计与论文表格

1. 计算三个 seed 的均值和标准差。
2. 对 ASR 差异执行 paired bootstrap，重复 10,000 次。
3. 输出 95% 置信区间。
4. 生成 clean performance 表。
5. 生成 attack baseline 表。
6. 生成 cross-attack transfer 矩阵。
7. 生成 combined training 表。
8. 生成低质量作文专项表。

---

## 15. 每次实验前检查清单

- [ ] 当前实现使用 right padding。
- [ ] 当前运行使用 formal split。
- [ ] test 未参与模型选择。
- [ ] config 已复制到输出目录。
- [ ] seed 已设置到 Python、NumPy、PyTorch、CUDA、dataloader。
- [ ] 模型处于正确的 train/eval 状态。
- [ ] 攻击没有跨 tokenizer 共享 token ID。
- [ ] 攻击预算与本文件一致。
- [ ] 正式攻击没有在 delta 0.1 提前停止。
- [ ] 结果保存完整文本和作文 ID。
- [ ] 等级指标使用共享阈值。
- [ ] 低质量分组结果已生成。
- [ ] clean QWK、MAE、RMSE 已生成。
- [ ] run manifest 已生成。

---

## 16. 长期更新记录

| 日期 | 版本 | 修改内容 | 修改人 |
|---|---|---|---|
| 2026-07-27 | 1.0 | 建立正式数据、攻击、损失、训练、评估与验收协议 | Codex |

后续更新规则：

1. 参数变化必须增加版本记录。
2. 已运行实验使用的参数不得覆盖，只新增 config。
3. 正式结果变化必须记录对应 commit 和 run manifest。
4. 预实验与正式实验始终分目录保存。
5. 论文表格只读取 `results/` 中状态为 completed 的正式运行。
