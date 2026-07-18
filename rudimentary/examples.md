# Rudimentary Attack — Text Examples

## Example 1: High Delta (idx=482)

| 指标 | 值 |
|------|-----|
| orig_score | 1.8482 |
| pert_score | 2.2913 |
| delta | +0.443 |
| steps | 7 |

**原文**:
The Challenge of Exploring Venus

In "The Challenge of Exploring Venus", the author suggests that studying Venus is a worthy pursuit despite the high pressure and extremely hot surface. The author suggests that us human can survive on Venus if there are some innovations that would allow our machines to last long enough to bring more knowledge about Venus.

The National Aeronautics and Space Administration (NASA) has some promising ideas for sending human to study Venus. One of their idea was tha

**攻击后**:
The Challenge of Exploring Venus

In "The Challenge of Exploring Venus", the author suggests that studying Venus is a worthy pursuit despite the high pressure and extremely hot surface. The author suggesst that us human can survive on Venus if there are some innovations that would allow our machines to last long enough to bring more knowledge about Venus.

The National Aeronautics and Space Administration (NASA) has some promising ideas for sending human to study Venus. One of their idea was tha

**改动分析**:
- "suggests" → "suggesst"（拼写错误）
- 仅 7 步迭代，从 1.85 提至 2.29（+0.44）

---

## Example 2: Low Delta (idx=863)

| 指标 | 值 |
|------|-----|
| orig_score | 2.5211 |
| pert_score | 2.6212 |
| delta | +0.100 |
| steps | 24 |

**原文**:
Reducing the Toll on the Human Race with Driverless Cars

The development of driverless cars would be an amazing thing.

So many bad situations could be avoided with driverless cars because the human drving a car is an enormous variable.

The payoff of using driverless cars would be giant as well.

Granted, it would be expensive and the period of time it would take to completely phase out regular cars would be lengthy, it would be a worthwhile investment into the future.

The development of driv

**攻击后**:
Reducing the Toll on the Human Race with Driverless Cars

The development of driverless cars would be an amazing thing.

So many bad situations could be avoided with driverless cars because the human drving a car is an enormous variable.

The payoff of using driverless cars would be giant as well.

Granted, it would be expensive and period the of time it would take to completely phase out regular cars would be lengthy, it would be a a worthwhile investment into the future.

The development of dr

**改动分析**:
- "would be expensive and the period" → "would be expensive and period the of"（语序混乱）
- 24 步迭代才达到 +0.10，分数边际提升
