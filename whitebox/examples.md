# HotFlip Attack — Text Examples

## Example 1: High Delta (idx=728)

| 指标 | 值 |
|------|-----|
| orig_score | 3.6205 |
| pert_score | 4.0909 |
| delta | +0.470 |
| steps | 11 |

**原文**:
Being in a classroom with a large amount of students can be intimidating. This makes it easier for a student to fall behind in a classroom setting, since they may be too scared to ask a question. One may be thinking, ¨what if they laugh at me if I raise my hand?¨ Or maybe that student may need a different teaching approach, but they do not know how to convey this to them. There could be a solution to this problem. New technology uses a computer that ¨constructs a 3-D computer model of the face¨

**攻击后**:
Being in a classroom with a large amount of students can be intimidating. This makes it easier for a student to fall behind in a classroom setting, since they may be too scared to ask a question. One may be thinking, what if they laugh at me if I raise my hand? Or maybe that student may need a different teaching approach, but they do not know how to convey this to them. There could be a solution to this problem. New technology uses a computer that constructs a 3-D computer model of the face (Mak

**改动分析**:
- 删除特殊字符 ¨
- 原文质量高但分数仅 3.62，攻击后 4.09（+0.47）

---

## Example 2: Low Delta (idx=1116)

| 指标 | 值 |
|------|-----|
| orig_score | 2.7516 |
| pert_score | 2.8516 |
| delta | +0.100 |
| steps | 10 |

**原文**:
Limiting car usuage is an advantage to all citizens.  All around the world people are trying pass new laws to band cars.  When people do not use their cars the world becomes less polluted, clear skies would be nice.  Also this will promote a safe lifestyle as a pedestrian. Creating people to walk everyday will be a good habit to the society.

Countries all around the world are realizing that reducing the use of cars is a big success.  " When i had a car i always was tense.  I'm much happier this

**攻击后**:
prohibited car usuage is an unaided to all citizens. All around the world people are trying pass new laws to band cars. When people do not use their cars the world becomes less polluted, clear skiesNikki be nice Barge Also this will promote a safe lifestyle as a pedestrian. Creating people to walk everyday will be a good habit to the society.

 Countries all around the world are realizing that reducing the use of cars is a big success. " When i had a car i always was tense. I'm much happier this

**改动分析**:
- "Limiting" → "prohibited"，"an advantage" → "an unaided"（词被替换，质量下降）
- 中间插入乱码 "Nikki" "Barge"
- 分数反而涨 +0.10
