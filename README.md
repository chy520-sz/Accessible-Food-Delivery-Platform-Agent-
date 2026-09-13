# 小鹿 AI Agent

面向无障碍外卖场景的独立 Agent 服务，提供文字对话、语音识别、统一语音合成和 RAG 知识检索。

## 技术栈

- Python 3.11
- FastAPI + Uvicorn
- LangChain 1.2 + LangGraph
- DeepSeek Chat
- Milvus + `pymilvus`
- `qwen3.7-text-embedding-flash`，1024 维
- 阿里云智能语音 ASR
- Edge TTS

## 目录

```text
take-out Agent/
├─ main.py                    # FastAPI 入口（只做应用装配：中间件/异常/挂载路由）
├─ routers/                   # APIRouter 路由分组
│  ├─ __init__.py             # 汇总 api_router
│  ├─ chat.py                 # 文字 / 流式文字 / 语音合成 / 语音对话
│  ├─ session.py              # 登录态同步 / 会话删除
│  ├─ health.py               # 健康检查
│  ├─ deps.py                 # 限流器与服务密钥鉴权依赖
│  └─ schemas.py              # 请求/响应 Pydantic 模型
├─ agent.py                   # Agent、工具、会话管理与流式对话
├─ llm_client.py              # 共享 LLM 单例（streaming=True）
├─ config.py                  # 环境变量
├─ knowledge_base.py          # Milvus 检索
├─ embedding_client.py        # Qwen Embedding 客户端
├─ build_knowledge_base.py    # 知识库构建
├─ check_rag.py               # RAG 自检
├─ speech.py                  # ASR 与 TTS
├─ data/                      # JSON 知识源（dish/dietary/faq）
├─ tests/                     # 单元测试
└─ docker-compose.milvus.yml  # Milvus Standalone v2.6.0（三容器）
```

`main.py` 不再包含任何路由实现，只保留 CORS、限流中间件、全局异常与 lifespan；
新增接口时在 `routers/` 下按职责添加模块，并在 `routers/__init__.py` 注册即可。

## 环境准备

项目使用 Conda 环境 `D:\anaconda3\envs\take-out`。

```powershell
cd "D:\java code\takeout\take-out Agent"
D:\anaconda3\envs\take-out\python.exe -m pip install -r requirements-dev.txt
D:\anaconda3\envs\take-out\python.exe -m pip check
Copy-Item .env.example .env
```

`.env` 至少填写：

```env
DEEPSEEK_API_KEY=你的DeepSeek密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

DASHSCOPE_API_KEY=你的百炼密钥
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/api/v1
RAG_EMBEDDING_MODEL=qwen3.7-text-embedding-flash
RAG_EMBEDDING_DIMENSIONS=1024

MILVUS_URI=http://localhost:19530
MILVUS_DB_NAME=default
```

启用语音识别时填写：

```env
ALIYUN_ACCESS_KEY_ID=你的AccessKeyId
ALIYUN_ACCESS_KEY_SECRET=你的AccessKeySecret
ALIYUN_ASR_APP_KEY=你的Appkey
```

语音合成可选配置：

```env
TTS_VOICE=zh-CN-XiaoxiaoNeural
TTS_RATE=-10%
```

不要提交包含真实密钥的 `.env`。

## 启动 Milvus

本项目使用 `docker-compose.milvus.yml` 编排的 **Milvus v2.6.0**（etcd + MinIO + standalone 三容器），
与 `requirements.txt` 中的 `pymilvus>=2.6,<2.7` 严格配套。请勿混用其他 Milvus 版本。

```powershell
cd "D:\java code\takeout\take-out Agent"
docker compose -f docker-compose.milvus.yml up -d
docker compose -f docker-compose.milvus.yml ps
```

默认连接地址为 `http://localhost:19530`，端口只绑定 `127.0.0.1`。

数据持久化在三个命名卷中，容器重建不会丢数据：

| 卷 | 内容 |
|---|---|
| `takeout-milvus_milvus_data` | Milvus 运行数据与消息队列 |
| `takeout-milvus_milvus_etcd` | 集合元数据（schema、段信息） |
| `takeout-milvus_milvus_minio` | 向量与段数据（`storageType: remote`） |

> **注意**
> - 停止请用 `docker compose down`（不带 `-v`）；`down -v` 会删除数据卷，需重新构建知识库。
> - 三个服务默认重启策略为 `no`，电脑重启后需手动 `up -d` 恢复。
> - 升级或更换 Milvus 版本（如 v3.x）会改变存储布局，旧集合不可直接复用，必须重建。

## 构建 RAG 知识库

首次运行或知识源发生变化时执行：

```powershell
cd "D:\java code\takeout\take-out Agent"

# 只检查知识源，不连接 Milvus、不调用 Embedding
D:\anaconda3\envs\take-out\python.exe build_knowledge_base.py --dry-run

# 构建全部集合并自检
D:\anaconda3\envs\take-out\python.exe build_knowledge_base.py
D:\anaconda3\envs\take-out\python.exe check_rag.py
```

只构建指定集合：

```powershell
D:\anaconda3\envs\take-out\python.exe build_knowledge_base.py --only dish
D:\anaconda3\envs\take-out\python.exe build_knowledge_base.py --only dietary
D:\anaconda3\envs\take-out\python.exe build_knowledge_base.py --only faq
```

| 类型 | 默认集合 |
|---|---|
| 菜品 | `takeout_dish_qwen37_1024_v1` |
| 饮食健康 | `takeout_dietary_qwen37_1024_v1` |
| 常见问题 | `takeout_faq_qwen37_1024_v1` |

## 启动 Agent

```powershell
cd "D:\java code\takeout\take-out Agent"
D:\anaconda3\envs\take-out\python.exe main.py
```

- API 文档：<http://localhost:8000/docs>
- 健康检查：<http://localhost:8000/agent/health>

端口被占用时停止旧进程：

```powershell
$agentPid = (Get-NetTCPConnection -LocalPort 8000 -State Listen).OwningProcess
Stop-Process -Id $agentPid
```

## API

| 方法 | 路径 | 用途 |
|---|---|---|
| POST | `/agent/sync` | 同步或清除登录态 |
| POST | `/agent/text` | 文字对话（同步返回完整 JSON） |
| POST | `/agent/text/stream` | 文字对话（SSE 流式增量输出） |
| POST | `/agent/tts` | 将文字合成为小鹿音色 MP3 |
| POST | `/agent/voice` | ASR → Agent → TTS 语音对话 |
| GET | `/agent/health` | 服务、模型和 RAG 状态 |
| DELETE | `/agent/session/{session_id}` | 删除会话与历史 |

本地开发可将 `AGENT_SERVICE_API_KEY` 留空。生产环境应配置该值，并由可信调用方携带 `X-Agent-Key`。

### 流式输出（SSE）

`POST /agent/text/stream` 与 `/agent/text` 使用完全相同的请求体和会话/鉴权语义，
区别在于响应以 `text/event-stream` 按 token 增量返回，首字延迟更低。
事件序列如下，`delta` 可出现 0..N 次：

```text
event: session
data: {"session_id":"abc123","is_new_session":true,"trace_id":"a1b2c3d4"}

event: delta
data: {"text":"您"}

event: done
data: {"session_id":"abc123","reply":"您好，想吃点什么？","tts_text":"您好，想吃点什么？","is_new_session":true,"trace_id":"a1b2c3d4"}
```

出错时以 `error` 事件替代 `done`（HTTP 状态仍是 200，调用方需按事件类型处理）：

```text
event: error
data: {"message":"服务繁忙，请稍后再试","trace_id":"a1b2c3d4"}
```

前端可先按 `delta` 实时渲染，收到 `done` 后用完整 `reply` 覆盖，并调用 `/agent/tts`
或在收到完整文本后自行播报；语音接口 `/agent/voice` 保持“聚合完整文本再合成”的原行为。

> 实现提示：SSE 增量**不能用 axios**（浏览器端基于 XHR，拿不到响应体增量），
> 需用原生 `fetch` + `response.body.getReader()` 逐块解析。用户端 `frontend/client`
> 已封装 `agentChatStream(sessionId, text, onDelta)` 供参考。

## RAG 日志

菜品搜索固定执行 Milvus RAG 与实时检索。同一次调用使用相同 ID 串联日志：

```text
[RAG tool][b8fc73cf] type=dish query='牛肉'
[RAG][b8fc73cf] START
[RAG][b8fc73cf] EMBEDDING_DONE dim=1024 elapsed_ms=1038
[RAG][b8fc73cf] MILVUS_DONE candidates=8 elapsed_ms=334
[RAG][b8fc73cf] FILTER candidates=8 accepted=8 scores=[...]
[RAG][b8fc73cf] FINISH elapsed_ms=1486
```

日志不会输出 API Key 或向量原值。

## RAG 可用性保障

Milvus 异常（如集合卡在 `Loading`）时，同步调用若无超时会无限阻塞，并拖满整轮对话预算。
检索链路对此做了两层保护，可在 `.env` 中调整：

```env
RAG_CALL_TIMEOUT=10       # 单次 Milvus RPC（加载集合/检索）超时（秒）
RAG_FAILURE_COOLDOWN=30   # 检索失败后的快速失败冷却时间（秒）
```

- **调用限时**：`load_collection` / `search` / `has_collection` 等均带显式超时；超时抛
  `KnowledgeStoreError`，工具层快速返回“知识库不可用”，不再阻塞整轮。
- **失败熔断**：某集合检索失败后进入冷却期，期间直接快速失败，避免每个请求都重复撞超时；
  冷却结束自动给一次恢复机会。“集合尚未构建”不算故障，不进入冷却，构建后即可立即恢复。

## 会话历史自愈

请求在工具节点执行期间被超时/中断时，checkpoint 可能留下“带 `tool_calls` 的 AI 消息
却没有对应工具结果”，OpenAI 兼容接口会直接返回 400，导致该会话后续对话全部失败。
每轮对话前会执行：

- **悬空工具调用修复**：为缺失结果的 `tool_call` 补一条合成 `ToolMessage`（内容提示该调用被中断），
  必要时整段重建消息；不丢历史。
- **窗口裁剪**：长对话按 HumanMessage 边界裁剪到 `AGENT_MESSAGE_WINDOW`，保证工具调用/结果成对完整。

上述状态更新显式指定 `as_node`，避免 LangGraph 的 `Ambiguous update` 静默失败。

## 语音

- `/agent/voice` 使用阿里云 ASR 识别 WAV/PCM，再由 Agent 回复并生成 MP3。
- `/agent/tts` 与语音对话共用 `TTS_VOICE`，用于统一系统播报和文字对话音色。
- Edge TTS 不可用时，调用方可降级到浏览器语音。

## 验证

```powershell
cd "D:\java code\takeout\take-out Agent"
D:\anaconda3\envs\take-out\python.exe -m pytest -q
D:\anaconda3\envs\take-out\python.exe -m ruff check .
D:\anaconda3\envs\take-out\python.exe -m pip check
```

当前基线：76 个单元测试通过，RAG 自检全部通过。

## 常见问题

- `Port 8000 is already in use`：停止旧 Agent 进程后重启。
- `AGENT_SERVICE_API_KEY 未设置`：本地开发提示，不影响启动。
- Milvus 集合为空：执行 `build_knowledge_base.py` 后再运行 `check_rag.py`。
- 集合卡在 `Loading` / 日志出现 `invalid local path`：存储布局损坏，需清理数据卷后重建集合
  （`docker compose down -v` 后 `up -d`，再执行 `build_knowledge_base.py`）。请确认使用的是
  compose 编排的 v2.6.0，而非其他手工容器。
- 会话开始报 400 `insufficient tool messages following tool_calls message`：历史存在悬空工具调用，
  下一轮对话会自愈；若持续出现请检查是否有外部代码直接改写 checkpoint。
- Embedding 维度不匹配：确认模型与集合均为 1024 维，然后重新构建集合。
- 阿里云 Token TLS 错误：代码会自动重试；持续失败时检查网络、系统时间和 `SSL_VERIFY`。
- TTS 返回 403：确认已安装 `edge-tts==7.2.8`。
