# ASAP 2.0 第二骨干：Longformer-base，seed 42

在 C（3090，24 GiB）的仓库根目录运行；一次只开一个 GPU 任务。先同步代码。
原版阈值实验已经完成，不需要再次启动，也不要改旧绑定以让历史结果“通过”新代码的哈希。

1. CPU：`python paper_results/summarize_aes_upward_band_crossings.py` 从历史 seed42、早停 +0.1 的审计清单统计主表的内部向上跨档率。不用 GPU，不写文件。若历史记录与默认分档不符，请核实当时的门槛文件后用 `--thresholds PATH`，不要调整门槛来匹配结果。不同种子的审计和 +0.3 搜索不能混入。
2. `python paer/train_longformer_aes_baseline.py --dry-run` 验证本地划分，不联网。
3. `python paer/train_longformer_aes_baseline.py --smoke --online` 首次可下载 allenai/longformer-base-4096；只做两次更新。已下载可用 `--model 本地目录` 代替联网。
4. `python paer/check_longformer_paer_compatibility.py` 用 **smoke** B0 对齐原生分数，检查可见 token 编辑和 2×1024 的前向/反向显存。它不更新模型；显存读数不含正式训练优化器状态。
5. 通过后，`python paer/train_longformer_aes_baseline.py` 用缓存的预训练模型重新开始正式 B0（不接着 smoke 训练）。清洁训练有效 batch 16；后续成对训练均为 2×16，有效 batch 32。
6. 下列每个阶段先执行 `python paer/run_longformer_rhi_stage.py 阶段名` 核对命令，再加 `--run` 执行。按顺序，失败先停止：

   `c0 → rh-pool → rhi-pool → mixed → paer → select-mixed → select-paer → evaluate-pair → evaluate-b0`

   C0 仅用于本骨干的 clean QWK 门槛，保持与原实验相同的选模定义。RHI 使用本次 Longformer-B0 的轨迹；不能用 DeBERTa 轨迹增分替代。Mixed/PAER 共享轨迹池、统一 30 步选模；模型冻结后才评 MLM。完整表读 `outputs/longformer_asap2_paired_evaluation_seed42/experiment_results.md`，B0 参考在 `outputs/longformer_asap2_b0_evaluation_seed42`。MLM 使用远程已有评估模型缓存。不同骨干 tokenizer、轨迹池样本和参数量可能不同；主要看每个骨干内部 PAER 对 Mixed 的差值。

## 正式 C0：启动与恢复

**C0 与 B0 使用不同的断点格式。** 新 C0 通过 `python paer/run_longformer_rhi_stage.py c0 --run`
从正式 B0 开始，独立目录为 `outputs/longformer_asap2_c0_resumable_seed42`，不采用旧的
`gstep600` 或 `gstep200` 权重。每 200 个 optimizer step 写完整断点（含优化器、调度器、
RNG 和 epoch 内数据游标），先写临时文件再发布。请预留约 15–20 GiB 磁盘给新 C0。
重启后用 `python paer/run_longformer_rhi_stage.py resume-c0 --run`，自动取最近已提交断点；
不足 200 步则从原 B0 重新开始。全部结束后看该新目录的 `completed.json`；后续两组选模已切换到这里的 best。

## 正式 B0 中断后续训

保留原训练目录（包括当前最佳权重和最近保存的断点）。例如在仓库根目录：

```powershell
$cp = '.\outputs\longformer_asap2_b0_seed42\intermediate\checkpoint-576'
python .\paer\train_longformer_aes_baseline.py --resume-from-checkpoint $cp --dry-run
if ($LASTEXITCODE -ne 0) { throw "断点预检未通过，停止。" }
python .\paer\train_longformer_aes_baseline.py --resume-from-checkpoint $cp
```

换成实际最新断点。预检仅读取文件，不反序列化优化器，也不启动 GPU；实际续训由 Trainer 恢复状态，继续原有保存与最佳模型选择策略。

## 原版 ASAP-AES：独立的数据准备阶段

需要原版有标签 TSV（字段 essay_id、essay_set、essay、domain1_score），不是已有 ASAP 2.0 的 train.csv。
`python paer/prepare_original_asap_folds.py --source PATH --dry-run` 核对后去掉 dry-run 生成新目录；UTF-8 副本加 `--encoding utf-8-sig`。按题目、按归一化文本分组划分；保留原始分与 0–5 连续目标。各题档位不能直接混用。未来有逐篇预测后，可用 `python paer/evaluate_original_asap_prompt_qwk.py --fold PATH --predictions PATH` 计算逐题原量表 QWK 与等权均值。尚未将这些拆分接入 ASAP 2.0 旧的六档选模流程；不能把旧训练脚本改一个 CSV 路径当作跨数据集结果。
