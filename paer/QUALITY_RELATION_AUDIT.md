# DeepSeek 三档作文质量关系评估

本流程只审计已有攻击文本，不训练、生成新攻击、修改评分器或覆盖历史结果。
标签为 improved / unchanged / worsened，含义是修改后相对原文的整体写作质量变化。
使用固定的通用写作准则，不声称采用官方考试细则、人工真值或实现因果识别。
缺少题目/阅读材料时，不判断切题性、来源忠实性或无法核实的外部事实。
本轮不生成正常改进对照，因此不能据此证明保留了所有合法涨分能力。

## 固定范围

- B0：deberta_checkpoints/fold0_best。
- Mixed：统一30步选模目录记录的 checkpoint（seed42 当前为 gstep1000）。
- PAER：统一30步选模目录记录的 checkpoint（seed42 当前为 gstep400）。
- 仅 R/H/I，不使用 MLM；不纳入新单攻击 PAER，或旧选模检查点。
- 默认从开发集按题目/分数分层轮转抽40篇共同作文，不根据攻击是否成功抽样。
- 每模型 R40、H40、I40（external20、self_dup20），共360条来源记录。
- 相同原文/修改文/上下文按内容去重，可由多个来源复用判断。完全未改动文本直接标为 unchanged。
- 分层诊断样本不是全1154篇的质量分布估计，指标必须附上实际分母。
- 不同模型受到各自的攻击，攻击文本不相同。判断比例不能当作同一固定文本集上的配对效应。

## 价格和预算

2026-10-19核查官方中文价格页：
https://api-docs.deepseek.com/zh-cn/quick_start/pricing/
deepseek-flash 当前别名对应 DeepSeek-V4.1-Flash-1017；别名可能更新，并非不可变模型版本。
峰值每百万token：未缓存输入2元、输出8元。代码按输入2.5元、输出10元保守计账。
显式关闭默认的思考模式；不启用工具、搜索、流式或后台并发。

预算总额固定30元，试调用和后续正式请求共用一个目录和账本。
每次请求前按UTF-8字节保守上界、消息余量和输出上限预留费用；收到可靠usage后，
按高峰价格加25%余量结算。忽略缓存和非高峰折扣，因此不是官方实际账单。
解析失败也计费；超时、断线、缺失usage和中断的请求保留全部预留。
预算不足不再发新请求。中断后继续使用同一目录，不能换目录或删除账本来清零预算。
每个方向最多尝试两次，所有尝试留档；完成的有效判断不重发。

这是客户端保守停止机制，不是账户侧扣费封顶，不能控制其他程序调用、计价变动、
服务端异常或未预期的token计量。若需账户级限制，应同时使用服务商可用的额度措施。
运行前再次核查价格；价格变化时暂停，不要静默修改已绑定运行的计价规则。

## 输出与隐私

默认私有数据目录：artifacts/quality_assessment/seed42_deepseek_flash_run01。
默认汇总：outputs/quality_assessment/seed42_deepseek_flash_run01。
以上目录均由现有.gitignore排除。源结果只读，实验来源/输入和评审协议进行哈希绑定。

- source_manifest.json：抽样ID、文件哈希、检查点和来源。
- private_pair_records.json：来源模型、攻击类型、原分、改后分、文本对ID，不发给API。
- blind_pairs.json：完整原文/修改文及可选上下文，不含模型名、分数或攻击名称。
- judge_binding.json：提示词、请求参数、模型别名、计价及代码哈希。
- votes/：每个方向的最近结果；attempts/：保留每次尝试及API实际返回的模型标识、usage等。
- budget_ledger.jsonl：请求前落盘的链式预算记录；budget_status.json：当前用量/预留。

只有显式给出 --allow-upload 且设置环境变量 DEEPSEEK_API_KEY 才会发送文本。
执行此选项前应确认作文可传给该API，并按服务商条款处理隐私/保留问题。
脚本只连接 https://api.deepseek.com/chat/completions，拒绝重定向，不记录密钥。
不展示完整API错误正文以免泄漏敏感信息。不要把密钥写入代码、聊天或Git。

## 远程 PowerShell 操作

在远程项目 E:\xjj\aes_scoring 的 xjj_aes 环境运行，先同步A到B、C拉取B。

### 1 导出与预算预览 均不调用API

```powershell
python .\paer\prepare_aes_quality_judgment_pairs.py
python .\paer\run_aes_quality_judge.py --dry-run
```

会只读搜索各个模型已有的全量攻击details和run_manifest，验证对应checkpoint、数据、
seed42、30步、beam1和阈值0.1。若发现多个不同的匹配结果，会拒绝自动挑最好的一份。
缺文件时只报错，不会自动补跑训练或攻击。请保留原错误供核验。
如确需显式指定来源，可提供 --sources-json，结构如下（总共三个模型、每模型四类）：

```json
{
  "b0": {
    "rudimentary": "outputs/实际目录/rudimentary_details.json",
    "hotflip": "outputs/实际目录/hotflip_details.json",
    "injection_external": "outputs/实际目录/injection_external_details.json",
    "injection_self_dup": "outputs/实际目录/injection_self_dup_details.json"
  },
  "mixed": {"rudimentary": "...", "hotflip": "...", "injection_external": "...", "injection_self_dup": "..."},
  "paer": {"rudimentary": "...", "hotflip": "...", "injection_external": "...", "injection_self_dup": "..."}
}
```

可选 --context-json 提供 essay_id 到 {task_prompt_text, source_material} 的完整文本映射，
不要只用题目名称冒充题目。没有上下文也可运行，但结论限定于内在写作质量。

### 2 本地输入密钥并做12对接口检查

```powershell
$judgeSecret = Read-Host "DeepSeek API key" -AsSecureString
$env:DEEPSEEK_API_KEY = [System.Net.NetworkCredential]::new("", $judgeSecret).Password
Remove-Variable judgeSecret

python .\paer\run_aes_quality_judge.py --max-pairs 12 --allow-upload
python .\paer\summarize_aes_quality_judgments.py

Get-Content .\artifacts\quality_assessment\seed42_deepseek_flash_run01\budget_status.json
Get-Content .\outputs\quality_assessment\seed42_deepseek_flash_run01\quality_relation_summary.md
```

12对仅用于验证接口、格式和用量，不据此作论文主结论。此阶段已经发送文本并消耗预算。

### 3 检查通过后 在同目录完成余下判断

```powershell
python .\paer\run_aes_quality_judge.py --allow-upload
python .\paer\summarize_aes_quality_judgments.py
```

已完成方向复用；无需删除试调用结果，也不再生成第二份收费缓存。
若预算停止，保留全部文件并检查完成率，不自动加钱或更改样本数。
正式开始后不要更新核心脚本/提示词/输入。新代码会拒绝混用旧缓存，而不是静默兼容。
如需结束当前会话的密钥使用：Remove-Item Env:DEEPSEEK_API_KEY。

## 标签可靠性与结果阅读

每对进行正反两个独立请求。第二次结果反向换算后必须一致；冲突再做一次正向判断。
第三次只是同一LLM的辅助复核，不是独立人类评审；无多数时保留冲突，不增加第四个质量标签。
JSON错误、空结果、截断、虚构证据span不当成unchanged，也不无限重试。
低置信判断仍保留，并报告计数，不仅展示筛选后更好看的结果。
不同请求间的别名版本可能变化；汇总保存实际返回模型标识，混合标识需审计。

主汇总展示每组总数、完成数、三档计数、顺序冲突，以及：
ASR_on_LLM_nonimproved = (LLM判断不变/下降且模型增分>=0.1的数量) / (LLM判断不变/下降的数量)。
另在JSON中报告仅顺序一致样本的敏感性分析。不确定/失败样本不会偷偷放进分母。
这些标签不用于训练或checkpoint选择，MLM保持排除。原始ASR结果不覆盖。

## 离线测试

```powershell
python -m unittest discover -b -s tests -p test_aes_quality_relation_audit.py
```

所有API响应使用模拟函数，测试不消费额度、不联网、不用GPU。
