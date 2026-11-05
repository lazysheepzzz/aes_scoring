# Shuffled Localization：等量随机位置监督对照

目的：检验完整 PAER 相比等量随机位置监督的收益，是否依赖真实编辑位置。对照从相同 B0 重新训练，不在原 checkpoint 上继续训练。除输出标识外，训练配置只新增固定置乱种子 42001；模型、训练 seed、轮转轨迹、样本、损失权重和分数级目标不变。

在每条实际看到的攻击状态上，按训练行号和攻击文本固定抽样，在可见非特殊 token 中不放回抽取与原目标相同数量的位置。定位损失和注意力对齐共用置乱后的累计目标。允许与真实位置偶然重合；非正累计增分、无可见编辑等无目标状态保持全零。这个设计是“等量随机位置”对照，不匹配真实编辑片段的连续性或位置分布；有差异时不要单独归因于定位损失。

## 1. 同步到 C 后，先做 CPU 预检

```powershell
Set-Location E:\xjj\aes_scoring
python -X utf8 E:\xjj\aes_scoring\paer\run_aes_paer_rhi_ablation_training.py --ablation shuffled_localization --output-dir E:\xjj\aes_scoring\outputs\aes_paer_rhi_v3_shuffled_localization_seed42 --dry-run --check-shuffle
```

确认 `shuffle_preflight.status = PASS`。脚本先核对原训练配置、B0 权重与 train/valid/trace 哈希，再按攻击子类从原训练轮转可触及的状态中各抽最多 32 条。检查 token 输入、分数、掩码和非位置目标完全一致，且每条状态的正例数和目标强度不变。看 `active_states`、`moved_states`、`positive_tokens`、`overlap_tokens` 判断这个对照实际改变了多少位置。预检只加载本地 tokenizer 和输入数据，不加载模型、不启动 GPU、也不创建输出目录。

## 2. 预检通过后，正式训练一个模型

```powershell
python -X utf8 E:\xjj\aes_scoring\paer\run_aes_paer_rhi_ablation_training.py --ablation shuffled_localization --output-dir E:\xjj\aes_scoring\outputs\aes_paer_rhi_v3_shuffled_localization_seed42
```

正式入口会再次预检并保存 `ablation_protocol.json`；拒绝非空训练输出目录。不改 batch 或梯度累积；在 RTX 3090 上不要与别的训练/评价并行。训练完成应有 `final_clean_metrics.json` 及与完整 PAER 的选模预算匹配的 gstep checkpoints。

## 3. 训练完成后，只选模和评价这个对照

```powershell
$shuffleEvalArgs = @(
  '--ablation', 'shuffled_localization',
  '--reference-selection', 'E:\xjj\aes_scoring\outputs\aes_paer_rhi_v3_checkpoint_selection_uniform30_seed42',
  '--reference-evaluation', 'E:\xjj\aes_scoring\outputs\aes_rhi_evaluation_development_seed42',
  '--output-dir', 'E:\xjj\aes_scoring\outputs\aes_paer_rhi_v3_shuffled_localization_evaluation_seed42'
)
python -X utf8 E:\xjj\aes_scoring\paer\evaluate_aes_paer_rhi_ablations.py @shuffleEvalArgs --dry-run
# 上一步核验通过后才去掉 --dry-run：
python -X utf8 E:\xjj\aes_scoring\paer\evaluate_aes_paer_rhi_ablations.py @shuffleEvalArgs
```

旧的完整 PAER 冻结评价只有在 checkpoint 路径与权重哈希和 uniform30 选中模型一致时才用作协议参考。选择沿用原 QWK 门槛、相同 gstep 候选预算、同一 256 篇子集、R/H/I 各 30 步。子集哈希不一致则不做全量攻击评价。只对该对照的选中 checkpoint 跑 1,154 篇 R/H/I；不重跑完整 PAER 和历史消融，不新增 MLM 评价。读取新目录的 `ablation_results.md`：主要比较 Clean QWK、R/H/I ASR 和等权 RHI 宏平均，再解释各家族变化。单 seed、单置乱的开发集结果不等于跨种子结论。
