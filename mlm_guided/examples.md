# MLM-Guided Attack — Text Examples

## Example 1: High Delta (idx=234)

| 指标 | 值 |
|------|-----|
| orig_score | 1.767 |
| pert_score | 2.197 |
| delta | +0.430 |
| steps | 3 |

**原文**:
Many people think that the Face that was discovered on Mars might have been created by aliens. But thats far from the truth. There are many reasons why that face could have shown up there. A natural lanform maybe? Mesa? Anything but aliens.

**攻击后**（3步达到阈值）:
Many people think that the Face that was discovered on Mars might have been created by aliens. But thats far from the truth. There are many reasons why that face could have shown up there. A natural landform maybe? Mesa? Anything but aliens.

**改动分析**:
- "lanform" → "landform"（拼写纠正），分数+0.43
- 仅 3 步即达到阈值，MLM-guided 精准定位有效替换位置

---

## Example 2: Low Delta Success (idx=772)

| 指标 | 值 |
|------|-----|
| orig_score | 1.940 |
| pert_score | 2.040 |
| delta | +0.100 |
| steps | 6 |

**原文**:
Computers can help shape the world. With the help of computers, we can tell the emotions of mona lisa. Imagine all the things the world can do with that technology. Children can be helped if there emotions can tell if there bored so, lesson plans can be more exciting to them.

**攻击后**（6步刚好达到阈值）:
Computers can help shape the world. With the help of computers, we can tell the emotions of mona lisa. Imagine all the things the world can do with that technology. Children can be helped if there emotions can tell if there board so, lesson plans can be more exciting to them.

**改动分析**:
- "bored" → "board"，细微用词替换
- 6 步才达到 +0.10 阈值，MLM 引导的替换增益较小

---

## Example 3: Failed (idx=82)

| 指标 | 值 |
|------|-----|
| orig_score | 0.212 |
| pert_score | 0.312 |
| delta | +0.100 |
| steps | 7 |
| 结果 | **失败**（delta 刚好 0.1 但评测阈值严格卡边界）|

**原文**:
It said the cowboys played baseball,volleyball,table tennis tornaments,fencing,boxing,reading,whittling,and games that help pass time so i think he was like showing people and telling people about its so fun and it was like about world war 2 and like that and told him it would be awesome and fun and

**改动分析**:
- 原 essay 拼写错误多（tornaments, bored 等）
- MLM-guided 尝试拼写纠正但收益有限
- 30 步内无法显著提升分数
