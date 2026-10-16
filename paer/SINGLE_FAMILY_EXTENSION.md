# 单攻击家族 AT / PAER 内部扩展实验

目标：分别只用 Rudimentary、HotFlip、Injection 的离线轨迹训练，检验 PAER
相对同数据、同训练预算的普通对抗训练是否仍有收益。不加入 Judge，不生成新攻击。

## 配对设计

| 训练家族 | 普通 AT | PAER |
| --- | --- | --- |
| Rudimentary | 现有普通评分器 + 单向增分损失 | 同数据 + PAER-v3 结构及配套目标 |
| HotFlip | 同上 | 同上 |
| Injection | 外部插入/自身重复的作文数各半 | 同样两个子类型，作文数各半 |

旧 D-HotFlip / D-Rudimentary / D-Injection 保留为历史结果。本扩展的严格成对
基线另训：旧入口使用在线候选或另一套离线样本，并非 PAER 的同一固定轨迹池。
尤其旧 HotFlip 默认 tolerance=0.05，当前离线 PAER 协议为 0。
这些是代码默认配置的差异，不替代对历史运行配置的归档。

从已完成 `aes_paer_rhi_v3_seed42` 的配置和训练来源记录继承超参数。
每个家族过滤同一个 RHI 池，保留全部干净训练作文，逐 epoch 轮换轨迹的规则不变。
不将含更多编辑的作文重复计数；不改写正的单步增量或有符号累计增量。
如果源池为之前的 902/902/451/451，每个单家族均有 902 篇带轨迹作文。
这低于混合 RHI 的 2706 篇，故单家族/混合差异不能全部归因于攻击多样性。

选择协议：干净 QWK 门槛为 C0−0.02；256 篇相同分层子集；R/H/I 各30步；
三家族等权宏平均，Injection 内两个子类型各半；候选限显式 gstep≤1400。
未训练的其他 RHI 家族仍参与选模，不能称为完全未见测试。MLM 仅最终评估。
评价集仍为 development fold，不将其称为独立测试集。

## 运行位置与前提

本地 A 修改并验证代码 → A push 到 GitHub B → 远程 C pull → C 的
`E:\xjj\aes_scoring` 中运行。脚本不会自行连接远程或执行 git push/pull。

需要保留：B0、C0、完整 PAER-RHI-v3 的 `launcher_config.json` 与
`rhi_training_inputs.json`、原训练/验证 CSV、RHI 轨迹及 manifest、原句库。
所有路径从远程既有配置读取；来源哈希不一致时停止，不能静默采用新输入。

默认每次一个子进程；沿用训练器的顺序反向传播、bf16、训练4×累积8。
选模与评价默认为 float32、batch=4，适配现有24GB机器；无新增在线候选生成。
实际训练参数以被继承的配置为准，准备阶段会打印各家族的干净/攻击作文数。

## 1. CPU 检查和正式准备

先验证代码（不下载模型，不占GPU）：

```powershell
python -m unittest discover -b -s tests -p test_aes_single_family_extension.py
```

核对正式数据来源与配置，但不写文件：

```powershell
python .\paer\run_aes_single_family_extension.py --stage prepare --dry-run
```

## 2. Smoke 训练

```powershell
python .\paer\run_aes_single_family_extension.py --smoke --stage train
```

这是运行验证，单独使用 `outputs/smoke_aes_single_family_extension_seed42/`。
160篇训练作文、32篇验证作文、1 epoch；显式抽取每个家族的带轨迹作文，
不使用可能碰不到任何轨迹的 first-N 训练切片。抽样比例与正式实验不同，
smoke 指标不进入论文结果。该命令默认顺序验证全部六个模型。

读取六个模型的结果：

```powershell
$root = '.\outputs\smoke_aes_single_family_extension_seed42'
Get-ChildItem $root -Recurse -Filter final_clean_metrics.json | ForEach-Object {
    Write-Host "`n=== $($_.FullName) ==="
    Get-Content -LiteralPath $_.FullName
}
```

## 3. 完整训练、选模与交叉评估

Smoke 通过后执行：

```powershell
python .\paer\run_aes_single_family_extension.py --stage all --include-mlm
```

自动顺序完成 R-AT/R-PAER、H-AT/H-PAER、I-AT/I-PAER 的训练、选模与 RHI+MLM
评价；不会并行加载多个模型。正式结果在
`outputs/aes_single_family_extension_seed42/single_family_results.md` 和 `.json`。
每次完成一个模型评价后更新汇总，尚未完成的结果不伪装成完整结果。

也可以分阶段执行 `--stage train`、`--stage select`、`--stage evaluate --include-mlm`。
`--families hotflip` 仅执行H组；`--models paer` 仅执行PAER分支。
这些过滤器不会改变同一运行中的配对数据和配置。

## 输出和继续运行

```text
outputs/aes_single_family_extension_seed42/
  rhi_run_binding.json
  data/
    rudimentary_training_traces.jsonl (+ manifest)
    hotflip_training_traces.jsonl (+ manifest)
    injection_training_traces.jsonl (+ manifest)
  rudimentary/at/{training,selection,evaluation}/
  rudimentary/paer/{training,selection,evaluation}/
  hotflip/at/{training,selection,evaluation}/
  hotflip/paer/{training,selection,evaluation}/
  injection/at/{training,selection,evaluation}/
  injection/paer/{training,selection,evaluation}/
  single_family_results.json
  single_family_results.md
```

同命令重跑会检查来源、模型和结果哈希，并复用已完成的训练/评估；既有历史
目录不会被接管。参数、数据或代码改变时需要新输出目录。
训练器没有 optimizer-state 恢复：若单个训练任务中断，保留部分结果并停止，
使用新目录配合 `--families` / `--models` 重跑该任务。评估和选模可以恢复。
MLM 可在六个模型/选择均冻结后，通过 `--stage evaluate --include-mlm` 单独补齐。
