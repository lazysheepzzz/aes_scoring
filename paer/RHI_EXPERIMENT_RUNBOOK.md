# RHI 联合攻防实验：远程 24 GB RTX 3090 运行说明

## 当前范围与实验角色

本轮补齐 Mixed-AT-RHI、PAER-RHI-v3 的联合训练、三家族选模、完整评估，
以及三个已完成专用防御的 MLM 迁移测试。现有 RH、v1/v2/v3、单攻击防御结果均保留。
原论文共享攻击实现不变。新增模型继续使用当前 v3 结构，不更改方法名称。

- Rudimentary、HotFlip、Injection 都参与两个 RHI 模型的训练与选模。
- Injection 下 External/Self-Duplication 各占家族一半。
- MLM-guided 只参与冻结模型后的测试。禁止据 MLM 结果回头选 checkpoint。
- 当前默认评估数据仍是已使用的 `valid_fold0.csv`，属于开发集结果。
  代码不会将其重命名为独立测试集；独立测试来源尚待确认。
- 多种子可以用同一冻结数据池加 `--seed 43/44` 运行，输出目录分别命名。
  本轮不同时改变模型结构、添加新消融目标或重训三个专用防御。

本机 A 修改代码 -> GitHub B -> 远程 C pull。以下命令只在远程
`(xjj_aes) PS E:\xjj\aes_scoring>` 执行。同一时间只运行一个 GPU 命令。

## 文件与输出管理

### 生成 100% 后因 cumulative_delta 校验失败的恢复

`step_gain` 是相对上一步的增分；`cumulative_delta` 是相对原作文的
有符号分差。HotFlip 搜索可先降分后回升，所以前者为正并不要求后者为正。
校验现要求 step_gain 有限且为正、cumulative_delta 有限；不删除或改写原始记录。
训练器已有的非负 correction target 截断保持不变。

旧版本的目录绑定包含旧校验器哈希，修复后不要直接重跑生成命令，也不要删除目录。
同步代码后在远程项目根目录运行：

```powershell
python .\paer\finalize_aes_rhi_training_pool.py `
  --output-dir .\artifacts\paer\rhi_training_pool_seed42

Get-Content .\artifacts\paer\rhi_training_pool_seed42\rhi_counterfactual_training_traces.manifest.json
```

该入口验证原训练数据、RH 轨迹、句子库、模型权重、搜索代码和逐篇任务分配，
要求所有 Injection 分片存在；不加载 GPU 或补跑缺失任务。
仅允许此次校验工具变更，保留原目录绑定，并在新 manifest 中记录合并工具及分片哈希。
`finalization.nonpositive_cumulative_trace_counts_by_attack` 给出保留记录的实际统计。
已完成且哈希匹配的训练池再次调用会直接返回，不覆盖旧结果。
若提示输入、代码变化或缺少分片，应保留目录并检查报错，不绕过保护。
Mixed-AT-RHI 和 PAER-RHI-v3 继续读取同一个最终训练池，家族配比不变。

| 位置 | 作用 |
| --- | --- |
| `paer/prepare_aes_rhi_training_traces.py` | RH 复用与新的 Injection 轨迹生成 |
| `paer/finalize_aes_rhi_training_pool.py` | 已完成逐篇生成后，仅用 CPU 校验并合并训练池 |
| `paer/run_aes_mixed_at_rhi_training.py` | 普通三家族联合训练 |
| `paer/run_aes_paer_rhi_training.py` | 当前 PAER-v3 结构的三家族联合训练 |
| `paer/select_aes_rhi_checkpoint.py` | 两个模型共用的 RHI 选模 |
| `paer/evaluate_aes_rhi_experiments.py` | 顺序评估、完成标记、结果汇总 |
| `mlm_guided/evaluate_aes_specialist_defenses_mlm.py` | 三个专用防御的 MLM 缺失列 |
| `paer/audit_aes_experiment_data.py` | 只读检查训练/开发/可选测试文本重叠 |
| `artifacts/paer/rhi_training_pool_seed42/` | 新训练池、manifest、逐篇生成进度 |
| `outputs/aes_mixed_at_rhi_seed42/` | 新 Mixed 模型 |
| `outputs/aes_paer_rhi_v3_seed42/` | 新 PAER 模型 |

训练拒绝非空输出目录，不提供覆盖旧 checkpoint 的开关。训练中断后本轮入口不支持
恢复优化器状态；先保留失败目录，再用新目录名重启。数据生成和评估支持相同配置下续跑。
模型、输入或配置变化时，续跑保护会拒绝旧目录，需使用新的可读名称。
选模与评估只复用完成标记齐全的结果，失败的子任务会重做。

`launcher_config.json` 中的 `training_mode=mixed_at_rh/paer_rh_v3` 是复用的
底层结构/损失实现标识；实际实验由 `experiment_name` 和 `training_attacks` 明确记录。
RHI 的 `training_attacks` 包含四个子攻击名，训练日志及 `training_config.json`
会记录三家族 seen、MLM unseen。不能仅用底层实现标识判断训练暴露。

## 0. 检查现有输入

```powershell
python .\paer\audit_aes_experiment_data.py
Test-Path .\artifacts\paer\rh_counterfactual_training_traces_seed42.jsonl
Test-Path .\artifacts\paer\rh_counterfactual_training_traces_seed42.manifest.json
Test-Path .\deberta_checkpoints\fold0_best\model.safetensors
Test-Path .\outputs\aes_clean_continuation_seed42\best\model.safetensors
```

默认只读审计显示是否有文本重复，不能自动证明某数据从未参与历史模型开发。
如果后来提供新的独立测试集，可以追加 `--test <路径>` 先检查重叠。

## 1. 小规模数据和训练检查

先检查前 256 篇；它们来自现有训练集，不创建新划分。小样本仍要求 R/H/I 平衡。
如果某 Injection 子类没有成功轨迹，会明确失败；将 `--max-essays` 增至 512，
并换一个新的 smoke 输出目录，同时相应增大训练的 `--max-train-samples`。

```powershell
python .\paer\prepare_aes_rhi_training_traces.py --max-essays 256 --output-dir .\artifacts\paer\smoke_rhi_training_pool_seed42

python .\paer\run_aes_mixed_at_rhi_training.py --trace-jsonl .\artifacts\paer\smoke_rhi_training_pool_seed42\rhi_counterfactual_training_traces.jsonl --output-dir .\outputs\smoke_aes_mixed_at_rhi_seed42 --max-train-samples 256 --max-valid-samples 32 --num-epochs 1

python .\paer\run_aes_paer_rhi_training.py --trace-jsonl .\artifacts\paer\smoke_rhi_training_pool_seed42\rhi_counterfactual_training_traces.jsonl --output-dir .\outputs\smoke_aes_paer_rhi_v3_seed42 --max-train-samples 256 --max-valid-samples 32 --num-epochs 1

Get-Content .\artifacts\paer\smoke_rhi_training_pool_seed42\rhi_counterfactual_training_traces.manifest.json
Get-Content .\outputs\smoke_aes_paer_rhi_v3_seed42\final_clean_metrics.json
Get-Content .\outputs\smoke_aes_paer_rhi_v3_seed42\training_diagnostics.jsonl -Tail 5
Test-Path .\outputs\smoke_aes_paer_rhi_v3_seed42\best\paer_heads.pt
```

Smoke 的 QWK 仅验证运行，不能和 1,154 篇的全量 QWK 比较。
可给训练命令追加 `--dry-run`，只打印配置，不创建目录。

## 2. 正式 RHI 训练池

```powershell
python .\paer\prepare_aes_rhi_training_traces.py --seed 42
Get-Content .\artifacts\paer\rhi_training_pool_seed42\rhi_counterfactual_training_traces.manifest.json
```

默认最多选择约 50% 训练作文，一篇只分配一个攻击家族。R/H 从现有正增分轨迹中
采样，Injection 分配到其他作文上，用同一 B0 进行最多 3 步、每步最多 16 个候选的搜索。
原先专用 Injection 的随机单句 pair 没有分步分数信息，因此保留原文件，另外生成
用于 PAER 的轨迹。RHI 训练仍完全离线，不在训练显存里加载额外攻击模型。

Injection 搜索失败的作文只参加干净训练。生成完成后，按成功的 Injection 子类数量
下采样其他家族，保证唯一作文数 R:H:External:Self-Dup = 2:2:1:1。
因此实际对抗作文占比可能低于 50%，manifest 记录实际值，禁止写成必定 8,076 篇。
每个 epoch 所有干净作文仍访问一次；每个受攻击作文轮换一个已接受轨迹状态。
两个模型的输入文本、标签、采样次序和训练预算一致。

源 RH manifest 缺少旧 B0 权重哈希；工具能校验其 checkpoint 路径，并为新流程记录权重
哈希，但不能倒推出旧轨迹生成时的权重内容。请确保该 B0 没有被替换。

## 3. 正式联合训练，顺序运行

```powershell
python .\paer\run_aes_mixed_at_rhi_training.py --seed 42
python .\paer\run_aes_paer_rhi_training.py --seed 42
```

默认三轮，学习率 1e-5，物理 batch=4，累积=8，有效 batch=32，长度 1024，bf16。
沿用干净分支先 backward、再建立对抗分支的 24 GB 节省显存实现。
默认生成与评估也使用 batch=4。不要并行运行这两个训练。
若实际需要降低物理 batch，可两个模型共同改为 batch=2/累积=16，并记录配置。

```powershell
Test-Path .\outputs\aes_mixed_at_rhi_seed42\final\model.safetensors
Test-Path .\outputs\aes_paer_rhi_v3_seed42\final\model.safetensors
Test-Path .\outputs\aes_paer_rhi_v3_seed42\final\paer_heads.pt
Get-Content .\outputs\aes_mixed_at_rhi_seed42\final_clean_metrics.json
Get-Content .\outputs\aes_paer_rhi_v3_seed42\final_clean_metrics.json
```

## 4. 三家族共同选模

```powershell
python .\paer\select_aes_rhi_checkpoint.py --defense-output-dir .\outputs\aes_mixed_at_rhi_seed42 --selection-output-dir .\outputs\aes_mixed_at_rhi_checkpoint_selection_seed42

python .\paer\select_aes_rhi_checkpoint.py --defense-output-dir .\outputs\aes_paer_rhi_v3_seed42 --selection-output-dir .\outputs\aes_paer_rhi_v3_checkpoint_selection_seed42
```

两者均采用相同 seed=42 的 256 篇分层开发子集；显式 gstep<=1400；
排除 best/final 别名；干净 QWK >= 开发集 C0 QWK - 0.02。
在合格 checkpoint 中最小化 `(R + H + (External + Self-Dup)/2)/3`，
平局按干净 QWK、名称依次打破。没有合格模型时报告失败，不自动放宽门槛。

选模默认 HotFlip=10 步、Rudimentary=30 步、Injection=30 步，两模型设置一致；
这是低成本开发搜索预算，家族等级由联合参与和等权汇总决定。全量评估各攻击均为 30 步。
需要改变搜索预算时必须在查看新结果前固定，并对两个模型一致应用。

```powershell
Get-Content .\outputs\aes_mixed_at_rhi_checkpoint_selection_seed42\best_checkpoint.json
Get-Content .\outputs\aes_paer_rhi_v3_checkpoint_selection_seed42\best_checkpoint.json
```

## 5. 联合模型 R/H/I 和 MLM 评估

方法与两个 checkpoint 都冻结后运行：

```powershell
python .\paer\evaluate_aes_rhi_experiments.py --include-mlm
Get-Content .\outputs\aes_rhi_evaluation_development_seed42\experiment_results.md
```

如暂时只跑 R/H/I，省略 `--include-mlm`。以后要追加 MLM，需带 `--include-mlm`
并选择新 `--output-dir`，不能更改已绑定目录的攻击列表；为避免重复计算推荐一开始
冻结方法后直接使用上面的完整命令。中断后原命令重跑会跳过已有完成标记的任务。
不依赖 PowerShell 的 `$selected` 临时变量。

运行前检查两个模型共同训练设置、训练数据哈希、选模协议及代码版本是否一致。
依次加载模型，不将多个模型同时放进 GPU。输出 JSON 和 Markdown 汇总。
默认评估 CSV 全部行，不硬编码 1,154；仍记录开发集身份。

对正式独立测试需明确指定 `--data <新测试CSV> --evaluation-role independent-test`
和 `--test-provenance "来源及未参与开发的说明"`，并指定新的输出目录。
程序检查与 `--train-csv`、`--development-csv` 文本重叠；历史使用情况仍需人工确认。

## 6. 三个专用防御的 MLM 缺失列

这个步骤不依赖新 RHI 训练，GPU 空闲时可优先完成：

```powershell
python .\mlm_guided\evaluate_aes_specialist_defenses_mlm.py --dry-run
python .\mlm_guided\evaluate_aes_specialist_defenses_mlm.py
Get-Content .\outputs\aes_specialists_mlm_evaluation_development_seed42\experiment_results.md
```

默认读取现有 HotFlip、Rudimentary-v2、Injection 的 `best_checkpoint.json`，
不是训练目录中的 `best` 别名。若文件位置不同，用 `--hotflip-selection`、
`--rudimentary-selection`、`--injection-selection` 分别指定 JSON 文件。
测试预算与已有 MLM 评估入口一致：30 步、最多 16 候选、编辑率 0.05、
ModernBERT-large bf16、MiniLM 语义相似度门槛 0.90。

## 7. 后续种子与组件分析

多种子继续使用 seed42 的固定训练池，只改变训练随机种子；选模的 subset-seed
和 attack seed 仍可固定为 42，以隔离训练随机性。路径必须显式指定到对应种子。
评估的 `--seed` 表示攻击种子；使用 `--mixed-selection`/`--paer-selection`
指定不同训练种子的模型，并给新输出路径。

现有 `analyze_aes_paer_routing_contribution.py` 可继续读取新 PAER checkpoint 与
新评估目录下的详情文件，输出路径另取新名。这仍是固定攻击样本诊断，不是独立训练消融。
严格组件消融、人工质量复核及独立测试数据确认尚不算本轮已经完成的实验。
