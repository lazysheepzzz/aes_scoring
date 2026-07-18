# AES Robustness Evaluation Plan

## 目标

在 `D:\here\robust_text_scoring-main/` 下，参考论文源码结构，新增 AES 鲁棒性评估框架。对 DeBERTa-v3-base AES victim（`D:\deberta_last\reproduction_yekenot_deberta_reg\outputs_kaggle_4090_1024\checkpoints\fold0_best`）进行四类攻击评测。

---

## 论文四类攻击 → AES 映射

| 论文攻击 | 机制 | AES 映射 | 是否白盒 |
|---------|------|---------|---------|
| **Rudimentary** | 字符/词随机编辑 | 拼写错误、词序乱、词重复 | ❌ 规则 |
| **HotFlip** | 梯度引导 token 替换 | 找能让 DeBERTa 打更高分的词替换 | ✅ 白盒（用 victim 梯度） |
| **MLM-guided** | MLM 候选 + scorer 选最优 | WordNet 同义词替换 + scorer 验证 | ❌ 规则 + scorer |
| **Injection** | 插入无关内容 | 插入无关句子、复写自身句子 | ❌ 规则 |

---

## 目录结构

```
D:\here\robust_text_scoring-main\text_scoring_adv_training\
├── evaluation/
│   └── aes/                          ← 新建
│       ├── attacks/
│       │   ├── __init__.py
│       │   ├── injection.py          # 无关句子插入、整句复写、结构模板
│       │   ├── rudimentary.py        # 字符/词级编辑（复制论文）
│       │   ├── hotflip.py           # 梯度引导词替换
│       │   └── mlm_guided.py        # WordNet 同义词 + scorer 验证
│       ├── scorer.py                 # DeBERTa AES victim 封装
│       ├── evaluate.py               # 统一评测框架
│       └── run_attacks.py           # 入口脚本
```

---

## 四个攻击详细设计

### 1. Rudimentary（基础参考攻击）

**参考论文** `text_scoring_adv_training/data/collators.py` 的 `_apply_rudimentary_manipulations()`。

```python
# 字符级
- 随机替换: 随机字母替换（保持可读）
- 随机插入: 随机字母插入
- 随机删除: 随机字母删除
- 相邻交换: 相邻字母交换

# 词级
- 词重复: 随机选词重复
- 词删除: 随机删虚词/形容词
- 词交换: 相邻词交换
```

**不做搜索**，每次随机一个位置一个操作，测 ASR。

---

### 2. HotFlip（白盒梯度攻击）

**直接用 victim 模型梯度**。参考 `text_scoring_adv_training/training/reward_trainer.py` 的 `_hotflip_swap()`。

```python
# 对 essay 中每个 token 位置:
# 1. forward → score
# 2. backward → d(score)/d(embedding[position])
# 3. 在 vocab 上找能让 score 最大的替换 token
#    用一阶近似: best_token = argmax_vocab (grad @ W @ v_token)
# 4. 迭代 beam search（简化: 4 beams × 4 candidates × max 50 steps）
```

**需要**：victim 模型参数 + tokenizer。

---

### 3. MLM-guided（语义候选 + scorer 选）

**不用 LLM，用 WordNet + DeBERTa scorer 验证**。

```python
# 步骤
# 1. 对 essay 中每个实义词（noun/verb/adj/adv），查 WordNet 同义词集
# 2. 过滤：只保留 WordNet similarity ≥ 0.9 的替换对
#    （语义高度等价，人类认为质量相同）
# 3. 对每个候选替换 → scorer(text) → 记录分数变化
# 4. 如果替换后分数显著上升（> 某阈值）→ 攻击成功
```

**为什么用 WordNet sim ≥ 0.9**：解决 "good→beneficial" 这类"人类觉得涨分"的问题。sim < 0.9 说明词义偏离，替换后分数变化可能是合理的，不算模型被骗。

---

### 4. Injection（插入攻击）

**两种形式，都用规则**：

```python
# 形式 A: 无关句子插入
# - 从预制"无关句子库"（语料库采样，过滤得到）随机选一句
# - 随机插入位置: start / middle / end
# - 测试: 插入后 scorer 分数是否上升

# 形式 B: 整句复写（复制 essay 已有句子到新位置）
# - 随机选 essay 中一句
# - 复制到: start / middle / end（与原文不同位置）
# - 零新信息，但文本变长
```

**无关句子库**：从 Wikipedia 或其他语料随机采样，验证与 essay 主题无关（不需要额外判断，随机采样的句子默认与任何 essay 无关）。

---

## 评测指标

每个攻击跑 **N essays**，报告：

```python
metrics = {
    "asr": 攻击成功率（分数上升的 essay 比例）,
    "avg_delta": 平均分数变化（原始分 - 攻击后分）,
    "band_asr": band 跨级率（分数 band 变化的 essay 比例）,
    "avg_steps": 平均编辑步数（如果做了搜索）,
}
```

**额外记录**：每个 essay 的原始分数分布（验证 victim 在各 band 上的脆弱性是否不同）。

---

## 数据

- **Essay 数据**：`D:\deberta_last\reproduction_yekenot_deberta_reg\data\processed\valid_fold0.csv`
- **Victim**：`D:\deberta_last\reproduction_yekenot_deberta_reg\outputs_kaggle_4090_1024\checkpoints\fold0_best`
- **阈值**：`D:\deberta_last\reproduction_yekenot_deberta_reg\outputs_kaggle_4090_1024\reports\best_thresholds.json`

---

## 实施顺序

```
Phase 1: 基础框架
├── scorer.py（加载 DeBERTa victim）✅
├── evaluate.py（统一评测接口）✅
└── rudimentary.py（最快，今天能跑）✅

Phase 2: Injection
├── injection.py（无关句子 + 整句复写）✅
└── 预制无关句子库✅

Phase 3: MLM-guided
└── mlm_guided.py（WordNet 同义词 + scorer 验证）

Phase 4: HotFlip
└── hotflip.py（梯度引导替换，需要白盒）
```

---

## 成功标准

四类攻击各跑 200 essays 后，得到：
- 各攻击 ASR 表格
- 跨攻击比较（哪个最强）
- 分数 band 分布（低分 essay 是否更脆弱）

这是后续对抗训练（防御）的 target。
