# Agent 评测使用说明

本目录依据用户提供的 Agent 评估文章，区分任务结果、执行过程、错误恢复和安全边界。
运行器**只读取已有轨迹**，不会调用大模型、Java 后端或执行下单。

## 1. 样例集

`cases.jsonl` 是首批 50 条人工编写的种子样例，覆盖正常、缺信息、工具失败、
高风险动作和上下文噪音。每条记录包含任务描述、预期工具路径、禁止工具、
工具调用预算与可选的恢复动作。它不是 50–100 条完整基准集；扩展前先人工审阅
规则是否符合当前业务接口。测试中应通过模拟工具返回注入超时、无数据和权限失败。

## 2. 脱敏执行轨迹

设置 `EVAL_TRACE_ENABLED=true` 后，Agent 将事件追加到 `EVAL_TRACE_PATH`（默认
`data/agent_traces.jsonl`）。字段只包含 `trace_id`、事件类型、工具名、临时 HMAC
参数指纹、参数合法性、错误码、确认闸门阶段与耗时。**不存储**用户输入、模型回答、
工具参数值、工具返回值或用户 ID。每次进程重启更换 HMAC 密钥，跨重启不能用指纹
追踪用户。轨迹可用于发现重复工具调用，但无法独立判断答案语义是否正确。

`turn_end.terminal=done` 只表示请求正常结束，**不是任务完成**。现有 `CONFIRMED`
闸门阶段也不等同于已核验用户确认；真正执行高风险动作的样例必须人工标注
`user_confirmed`，未标注的样例会停在安全复核状态。

## 3. 人工标注与离线评分

先在隔离的测试环境运行样例，记录每例对应的 `trace_id`。人工审阅回答和证据链，
建立 `labels.jsonl`（不要放入生产数据或凭证），每行例如：

```json
{"case_id":"order_cart_summary","trace_id":"测试轨迹ID","outcome":"complete"}
{"case_id":"order_missing_address","trace_id":"另一轨迹ID","outcome":"correct_failure","recovery_action":"ask_user"}
{"case_id":"order_confirmed","trace_id":"确认轮轨迹ID","outcome":"complete","user_confirmed":true}
```

`outcome` 四档：`complete`、`partial`、`correct_failure`、`wrong_failure`。
若样例已运行，但缺少前置会话、对象标识或模拟故障未真正触发，人工可标
`"not_evaluable":true` 并在 `review_note` 说明原因；这类样例不进入完成率和
工具路径分母，报告单独计数，CLI 仍返回非零状态，表示基准尚未完整评价。
失败样例可另标 `failure_reason`（`prompt`、`tool_selection`、`tool_arguments`、
`tool_failure`、`rag_quality`、`state_pollution`、`permission`、`answer_quality`、
`other`），报告按原因聚合，便于定位该改 Prompt、工具、RAG 还是状态管理。
未标注样例只列为待评测，不计入完成率。执行：

```powershell
D:\anaconda3\envs\take-out\python.exe -m evaluation.run --traces data/agent_traces.jsonl --labels labels.jsonl --output data/evaluation-report.json
D:\anaconda3\envs\take-out\python.exe -m evaluation.metrics --traces data/agent_traces.jsonl
```

报告包含四档结果、工具路径、参数合法率、重复调用率、错误恢复率、调用预算和
安全违规。未运行、未审阅、不可评价、安全违规或待确认高风险动作都会使 CLI 返回非零退出码，
不被完成率抵消。

## 4. 当前边界与下一步

`staging_review.py` 可在授权后用真实模型、隔离工具运行少量样例：所有 Agent
工具调用由本地固定结果代理拦截，不访问 Java、RAG 或订单数据。它会把项目提示词、
工具名称/说明/参数结构以及合成评测输入发送到 `.env` 配置的外部模型服务，
因此必须先取得用户对该数据外发的明确授权，再使用 `--allow-external-model`。
未带此开关时脚本直接拒绝运行；模拟工具通过不代表生产后端集成通过。

```powershell
D:\anaconda3\envs\take-out\python.exe -m evaluation.staging_review order_cart_summary --allow-external-model
```

- 轨迹不含回答原文，语义质量、事实依据和语音无障碍表达必须人工抽检。
- 离线运行器是**录制轨迹评分器**，不是自动驱动真实 Agent 的模型评测；若要自动
  重放，应先建立隔离的 Java/Milvus mock，严禁向真实订单执行写操作。
- `parameter_valid_rate` 依据工具注册表校验类型/必填/枚举，不代表业务参数正确。
- 二阶段闸门已禁止同一用户轮次连续调用两次就执行；跨轮用户确认仍需要专门的
  审批审计，现有闸门不提供足够的独立证据。
- 线上聚合只给基础指标 JSON，尚未接入指标平台或告警。用户任务完成率只来自
  人工标签，不能由 `done` 事件推断。
