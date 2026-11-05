# PAER-v3 补充实验：第一批 CPU 审计

在远程实验机 C（`E:\xjj\aes_scoring`）同步本轮脚本后运行。无需复制模型权重到本机；入口不会加载模型、联网、启动攻击或训练。旧产物只读，报告路径必须是新文件。

```powershell
Set-Location E:\xjj\aes_scoring
python -X utf8 .\paer\audit_aes_supplementary_evidence.py --report .\outputs\aes_paer_v3_supplementary_seed42_run01\protocol_audit\remote_cpu_audit.json
```

它先检查 uniform30 选中的四个 checkpoint，再查匹配的 full R/H/I 与 MLM 明细。按 manifest 的实际 victim、数据、攻击设置匹配，不按文件夹名猜 checkpoint。MLM 仅核验覆盖情况，不纳入本轮 RHI 配对统计。对完全绑定的主比较输出配对 95% CI、逐攻击 McNemar/Holm 和人工原分分层；读已存在的重放缓存，统计绝对负向证据和修正量；绑定训练池存在时统计校准目标超过结构上界的状态数。

- `BOUND`：评价绑定、明细哈希及协议通过检查。
- `LEGACY_NO_FULL_BINDING`：有匹配的明细与清单，但缺完整绑定或完成标记，不能当成完全核验结果。
- `NOT_FOUND_OR_NOT_VERIFIED`、`SELECTION_UNAVAILABLE`：当前扫描无法核验，不代表远程从未跑过。看 JSON 中的 `selection_unavailable`、`rejected`。
- `AMBIGUOUS`：有冲突的候选，不自动挑某次结果。可用 `--evaluation-root` 缩小扫描范围。
- `statistics: UNAVAILABLE` 和 `replay: UNAVAILABLE` 互不阻塞，各自原因写在 JSON。重放如用非默认源，可显式指定 `--replay-dir` 和 `--replay-source`；必须与历史绑定匹配。

若主比较确实只能使用旧清单、逐篇数据和其他协议均通过，但缺完整绑定，可以加 `--allow-legacy` 并换一个新报告路径，生成明确标记为 `PROVISIONAL_LEGACY` 的统计。不要补写历史 manifest 使其伪装成新产物。

首先回传命令的简短终端输出即可；如果有 `UNAVAILABLE`，再回传相应 JSON 的 `reason` 或 `checks`。不需要传学生全文、模型权重或 API 密钥。在核验无路由 gstep1000 的覆盖情况之前，不启动补攻击。
