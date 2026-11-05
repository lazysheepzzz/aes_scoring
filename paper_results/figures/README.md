# AES 论文四图：来源与复现

只制作当前 PAER-v3 的框架和结果图，不训练、不重新攻击、不调用 API。
PDF 为 TeX 矢量图；SVG 保留文字便于编辑；PNG 用于预览。

## 当前数据边界

`seed42_reported_results.json` 是对用户已回传观测数值的逐项转录，
不是新实验、不是伪造数据，也不是本地原始产物核验结果。
它明确区分统一30步选模的 Mixed-AT gstep1000 和历史 gstep400。
PAER 为 RHI-v3 gstep400，不是 RH 版本或低秩专家版本。
结果角色为 development、单训练种子，不制作虚假的标准差或独立测试标记。

## 本地 A：生成已有三张图

```powershell
python .\paper_results\figures\build_aes_paper_figures.py --dry-run
python .\paper_results\figures\build_aes_paper_figures.py
```

需要 matplotlib、numpy。生成目录默认为 `outputs/paper_figures/seed42_run01`。
已存在的非空输出不会覆盖；重画必须使用新的可读目录名。
仓库只跟踪此脚本、数据快照与本说明，旧 paper_results 内容仍保持忽略。

## 远程 C：只读检查现有 history，再绘制预算曲线

先通过 A -> GitHub B -> C 同步这三个文件，再运行：

```powershell
python .\paper_results\figures\build_aes_paper_figures.py `
  --results-root .\outputs `
  --output-dir .\outputs\paper_figures\seed42_remote_run01 `
  --dry-run

python .\paper_results\figures\build_aes_paper_figures.py `
  --results-root .\outputs `
  --output-dir .\outputs\paper_figures\seed42_remote_run01

Get-Content -Encoding UTF8 `
  .\outputs\paper_figures\seed42_remote_run01\search_budget_data_status.json
```

自动扫描 run_manifest.json，要求 victim checkpoint、攻击种子、作文数、
干净 QWK、30轮 greedy、最终 ASR 与目标版本一致。
必须同时找到 Mixed-AT 与 PAER 的 R/H/Injection 两子类真实 history。
B0、C0 历史齐全时额外绘制，否则不影响两个配对主模型的预算曲线。
历史缺失、不一致或有多个不同候选时不生成曲线，原因保留在状态文件。
这里只核对元数据及历史一致性，不宣称完成原始权重和数据的全面哈希审计。

预算轴是搜索轮次，不是已接受编辑数或等价查询成本。
始终以全部1154篇为分母；早停成功在后续预算继续计入；失败不当作第30轮成功。
Injection 曲线是 External 与 Self-Dup 两条成功率曲线的等权平均。
不得从最终ASR插值曲线、填造 history、平滑为不存在的过程，或将三步训练轨迹代替30轮评估历史。

## 四张图的文件

1. `aes_paer_framework.*`：真实 v3 结构，保留未直接门控的全局分支。
2. `aes_cross_attack_heatmap.*`：7模型 x R/H/I/MLM；MLM 单独分隔；干净 QWK 独立点列。
3. `aes_attack_budget_curves.*`：仅有兼容真实历史时生成。
4. `aes_routing_diagnostics.*`：固定文本回放与分别自适应攻击并列，保留 R 固定集恶化结果。

重训消融仍用表格，不与推理开关诊断混淆。
未获编辑后独立质量标签，不绘制模型分数与“真实质量变化”的对照。
