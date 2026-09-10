# 冻结 RHI 模型后的路由诊断

保留既有模型、选模、攻击评估、旧 routing diagnostic 文件。
所有命令在远程 C 的项目根目录 `E:\xjj\aes_scoring`、`xjj_aes` 环境运行。
这一轮不训练、不调参、不重新选模，不用 MLM 指导开发。

## 两个不同问题

1. **固定攻击集阈值翻转**：原攻击文本不变，比较同一 PAER 的 routed logits 和
   base_logits。得到路由诱发/阻止多少次越过 0.1 阈值的攻击成功。
2. **自适应关闭路由评估**：以关闭路由后的模型重新搜索 R/H/I 攻击。
   和已有完整 PAER 各自被攻击的结果对照，不把固定集回放当作自适应攻击。

PAER-v3 的 base_logits 是“global score + signed token evidence”，不是 B0，
也不是独立训练的 Mixed-AT。关闭路由仅在内存中设 `correction_scale=0.0`，
保留 token evidence/attention/negative evidence 和同一个 checkpoint 的全部权重。
HotFlip 的梯度、候选评分和最终评分均使用这一前向分数。
不保存修改后的模型，不修改 `paer_config.json`，不改原论文/现有共享攻击文件。

## 新文件与输出

| 文件 | 用途 |
| --- | --- |
| `analyze_aes_rhi_threshold_flips.py` | 前向分数缓存、逐样本四格统计、等权家族汇总 |
| `evaluate_aes_paer_route_off.py` | 继承冻结预算的三家族自适应 route-off 攻击 |
| `routing_experiment_utils.py` | 冻结来源核对与 RHI 家族权重 |
| `outputs/aes_paer_rhi_v3_threshold_flips_seed42/` | 新回放缓存、逐篇分数和简短汇总 |
| `outputs/smoke_aes_paer_rhi_v3_route_off_seed42/` | 仅 2 篇端到端检查 |
| `outputs/aes_paer_rhi_v3_route_off_adaptive_seed42/` | 全量重新搜索的攻击和汇总 |

默认从 `outputs/aes_rhi_evaluation_development_seed42/rhi_run_binding.json`
读取真正已评估的 PAER checkpoint、数据、精度、batch、攻击种子和预算；
不依赖易丢失的 PowerShell `$selected`。
会核对模型/数据/搜索代码/攻击文件的哈希，输入变化时拒绝续跑。
不要通过删绑定文件或完成标记绕过保护。

## 1. 远程 CPU 测试

```powershell
python -m unittest discover -s tests -p test_aes_routing_followup.py
```

其中真实 Torch 测试不下载模型，在 CPU 上验证关闭路由后的分数和嵌入梯度
与原 base_logits 一致、与 routed 分支不同，并检查 state_dict 未变化。
本机 A 缺少 Torch/Transformers，此项在本机跳过；远程应运行而不是跳过。
其余测试检查阈值等号、翻转方向、家族权重、缓存、输出保护、mock 中断续跑。

## 2. 固定攻击集逐篇诊断

旧诊断只存汇总，无法从汇总倒推出逐样本翻转。
首次需要再做一次前向评分；默认 CUDA、batch=4，不生成攻击、不训练。

```powershell
python .\paer\analyze_aes_rhi_threshold_flips.py
Get-Content .\outputs\aes_paer_rhi_v3_threshold_flips_seed42\threshold_flip_summary.md
```

如果中断，同一命令复用完成的每 256 个不同文本的缓存块。
缓存齐全后，只重新计算 CPU 统计、不导入 Torch：

```powershell
python .\paer\analyze_aes_rhi_threshold_flips.py --cpu-summary-only
```

`--cpu-summary-only` 不是改变评分 device；它继续核对原缓存绑定。
如首次想完全使用 CPU 评分，可用 `--device cpu --output-dir <新目录>`，但更慢。
逐篇分数位于 `*_paired_scores.json`，不用向对话粘贴大文件。

四类转移以关闭路由为起点：both_fail、both_success、routing_prevents_success
（成功变失败）、routing_induces_success（失败变成功）。
ASR 改善=(阻止次数−诱发次数)/n。
RHI 汇总=(R+H+(External+Self-Dup)/2)/3，不再将 Injection 双倍计权。
距离 0.1 阈值 5e-5 内的样本单独标记，以提示浮点敏感性；不更改成功判定规则。
重放分数或 `base_delta-routed_delta=correction_lift` 误差超过 1e-4 时拒绝解读。

## 3. 自适应攻击：先 smoke，再全量

```powershell
python .\paer\evaluate_aes_paer_route_off.py `
  --n-essays 2 `
  --output-dir .\outputs\smoke_aes_paer_rhi_v3_route_off_seed42

Get-Content .\outputs\smoke_aes_paer_rhi_v3_route_off_seed42\adaptive_route_off_summary.md
```

smoke 的 clean/attack 都只用相同的前 2 篇；不会拿它和 1154 篇全量结果比较。
每次加载真实模型还会检查 route-off logits 等于原 base_logits，correction 为零。
smoke 成功且远程 CPU 梯度测试通过后，执行：

```powershell
python .\paer\evaluate_aes_paer_route_off.py
Get-Content .\outputs\aes_paer_rhi_v3_route_off_adaptive_seed42\adaptive_route_off_summary.md
```

各家族一个子进程，依次运行，batch/dtype/device 沿用已冻结的评估设置
（本次为 3090、batch=4、float32）。不同时加载多个模型，也不实例化 MLM 模型。
已完成的家族会跳过；中途失败的家族会重新搜索，不承诺逐作文续跑。
默认 R/H/I 各 30 步，具体参数从旧 run_manifest 继承，不借此修改攻击预算。

## 解释边界

- 两类诊断都是对当前共同训练模型的推理干预，不是“重新训练一个去路由模型”的消融。
- 自适应攻击仍是有限预算搜索，不是最坏情况鲁棒性证明；搜索路径和早停可能因模型改变。
- MLM 已作冻结后评估，不用其结果再次优化路由或挑选 checkpoint。
- 当前仍是开发集；没有新增独立测试数据，也没有产生多种子证据。
- 回放中基线较好，不自动推出关闭路由对自适应攻击更鲁棒；以重新搜索结果补充判断。
