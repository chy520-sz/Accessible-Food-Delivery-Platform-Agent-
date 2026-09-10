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
├─ main.py                    # FastAPI 入口
├─ agent.py                   # Agent、工具与会话管理
├─ config.py                  # 环境变量
├─ knowledge_base.py          # Milvus 检索
├─ embedding_client.py        # Qwen Embedding 客户端
├─ build_knowledge_base.py    # 知识库构建
├─ check_rag.py               # RAG 自检
├─ speech.py                  # ASR 与 TTS
├─ knowledge/                 # JSON 知识源
├─ tests/                     # 单元测试
└─ docker-compose.milvus.yml  # Milvus Standalone
```

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

```powershell
docker ps --filter name=milvus-standalone
```

如果未显示 `healthy`：

```powershell
cd "D:\java code\takeout\take-out Agent"
docker compose -f docker-compose.milvus.yml up -d
```

默认连接地址为 `http://localhost:19530`。

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
| POST | `/agent/text` | 文字对话 |
| POST | `/agent/tts` | 将文字合成为小鹿音色 MP3 |
| POST | `/agent/voice` | ASR → Agent → TTS 语音对话 |
| GET | `/agent/health` | 服务、模型和 RAG 状态 |
| DELETE | `/agent/session/{session_id}` | 删除会话与历史 |

本地开发可将 `AGENT_SERVICE_API_KEY` 留空。生产环境应配置该值，并由可信调用方携带 `X-Agent-Key`。

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

当前基线：51 个单元测试通过，RAG 自检全部通过。

## 常见问题

- `Port 8000 is already in use`：停止旧 Agent 进程后重启。
- `AGENT_SERVICE_API_KEY 未设置`：本地开发提示，不影响启动。
- Milvus 集合为空：执行 `build_knowledge_base.py` 后再运行 `check_rag.py`。
- Embedding 维度不匹配：确认模型与集合均为 1024 维，然后重新构建集合。
- 阿里云 Token TLS 错误：代码会自动重试；持续失败时检查网络、系统时间和 `SSL_VERIFY`。
- TTS 返回 403：确认已安装 `edge-tts==7.2.8`。
