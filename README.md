# 小鹿 AI Agent

面向无障碍外卖场景的独立 Agent 服务，提供文字对话、语音识别、统一语音合成和 RAG 知识检索。

## 技术栈

- Python 3.11
- FastAPI + Uvicorn
- LangChain 1.2 + LangGraph
- DeepSeek Chat
- Milvus 2.6.0 + `pymilvus`（稠密向量 + BM25 稀疏向量双路混合检索）
- `qwen3.7-text-embedding-flash`，1024 维（语义嵌入）
- `gte-rerank-v2` Cross-Encoder（精排重排序）
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
├─ tools.py                   # 35 个工具函数（菜品/套餐/分类/商家/订单/评分/RAG等）
├─ prompts.py                 # Agent 系统提示词与能力清单
├─ llm_client.py              # 共享 LLM 单例（streaming=True）
├─ config.py                  # 环境变量（含 RAG 全链路配置）
├─ knowledge_base.py          # Milvus 检索核心（混合检索+RRF+Rerank+元数据过滤+Query改写）
├─ long_term_memory.py        # 分类型长期记忆、写入策略、任务检索与清理闭环
├─ memory_middleware.py       # 按当前任务筛选并注入相关记忆
├─ execution_trace.py        # 脱敏执行事件轨迹（供评测与运维聚合）
├─ evaluation/              # 种子评测集、离线评分器、线上指标聚合
├─ embedding_client.py        # Qwen Embedding 客户端（带 SQLite 磁盘缓存）
├─ rerank_client.py           # 百炼 gte-rerank-v2 Cross-Encoder 精排客户端
├─ query_rewriter.py          # Query 改写（多 Query 扩展 / HyDE 假设文档嵌入）
├─ build_knowledge_base.py    # 知识库构建（稠密+BM25稀疏双路集合）
├─ sync_dish_knowledge.py     # 从 Java 后端同步菜品到知识库 JSON
├─ fill_dish_fields.py        # 菜品知识库字段补全（口味/食材/营养/辣度等）
├─ check_rag.py               # RAG 自检（嵌入/Milvus/集合/检索/阈值过滤）
├─ speech.py                  # ASR 与 TTS
├─ data/                      # JSON 知识源（dish 1146条 / dietary 18条 / faq 20条）
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

知识库由三个 JSON 源构建，当前规模：

| 类型 | 知识源 | 记录数 | 集合 |
|---|---|---|---|
| 菜品 | `data/dish_knowledge.json` | 1146 | `takeout_dish_qwen37_1024_v1` |
| 饮食健康 | `data/dietary_knowledge.json` | 18 | `takeout_dietary_qwen37_1024_v1` |
| 常见问题 | `data/faq_knowledge.json` | 20 | `takeout_faq_qwen37_1024_v1` |

每个集合同时存储**稠密向量**（语义检索）和 **BM25 稀疏向量**（关键词检索），由 Milvus BM25 Function 从 `text` 字段自动生成。

### 从后端同步菜品知识

菜品知识库与 Java 后端数据库是两套独立数据。后端菜品变更后，同步到知识库：

```powershell
# 1. 从后端拉取实时菜品，追加到 dish_knowledge.json（需后端运行 + 测试账号）
D:\anaconda3\envs\take-out\python.exe sync_dish_knowledge.py --phone 13900000000 --password 123456 --write

# 2. 补全新增菜品的口味/食材/营养/辣度等字段
D:\anaconda3\envs\take-out\python.exe fill_dish_fields.py

# 3. 重建 Milvus 向量库
D:\anaconda3\envs\take-out\python.exe build_knowledge_base.py
```

### 构建命令

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

构建流水线：校验 JSON → 清洗拼接文本 → 分块（≤300字，overlap50）→ 批量 Embedding（带 SQLite 缓存）→ 写入 Milvus（稠密+BM25稀疏+元数据）→ 数量/维度/检索冒烟校验。

## RAG 检索链路

在线检索采用**混合检索 + Rerank 精排**的完整链路，兼顾语义召回和关键词精确匹配：

```
用户 Query
  │
  ├─ ① Query 改写（可选，默认关闭）
  │    ├─ multi_query：LLM 扩展为 N 个语义等价查询，分别稠密检索后合并去重
  │    └─ hyde：LLM 生成假设答案文档，用文档 embedding 替代 query embedding
  │
  ├─ ② 元数据预过滤（可选，默认开启）
  │    Milvus 标量过滤：category / shop / spicy_level / taste / price 等
  │
  ├─ ③ 双路召回（recall_k=20，粗排多召回）
  │    ├─ 稠密路：Query → qwen Embedding(1024维) → Milvus COSINE 检索
  │    └─ BM25路：原始 Query → 中文分词 → 稀疏向量倒排检索
  │
  ├─ ④ RRF 融合（k=60，只依赖排名，融合 COSINE 与 BM25 两种量纲）
  ├─ ⑤ Rerank 精排（gte-rerank-v2 Cross-Encoder，query-document 联合编码打分）
  ├─ ⑥ 阈值过滤（rerank ≥ 0.05）→ 截取最终 top_k（默认 8）
  │
  └─ ⑦ 格式化 Context（带来源标记【菜名】[分类]和相关性分数）
       → LangChain ToolMessage 注入 → DeepSeek LLM 生成 → 带来源的答案
```

### 三级降级策略

| 故障点 | 降级行为 |
|---|---|
| BM25 路异常 | 降级为纯稠密检索 |
| Rerank API 异常 | 降级为 RRF 粗排（稠密阈值 + BM25 补充通道） |
| Milvus 异常 | 熔断器快速失败，工具层返回"知识库不可用"，绝不伪装成"没找到" |
| Query 改写异常 | 降级为原始 Query 检索 |

### 元数据过滤用法

调用 `search_knowledge()` 时传入 `filters` 参数，在 Milvus 检索阶段预过滤：

```python
from knowledge_base import search_knowledge

# 等值过滤
search_knowledge("鸡肉", collection, filters={"category": "川菜"})

# 组合过滤：分类 IN + 辣度 >= 2
search_knowledge("辣", collection, filters={
    "category__in": ["川菜", "湘菜"],
    "spicy_level__gte": 2,
})

# 店铺 + 价格上限
search_knowledge("套餐", collection, filters={
    "shop": "湘味人家",
    "price__lte": 50,
})
```

支持的操作符：`__eq`(默认) / `__ne` / `__in` / `__nin` / `__contains` / `__gt` / `__gte` / `__lt` / `__lte`。

### RAG 工具（Agent 层）

LLM 通过三个 `@tool` 自主调用知识库：

| 工具 | 用途 | 集合 |
|---|---|---|
| `search_food_knowledge(query)` | 菜品口味、烹饪方法、食材搭配、菜系文化 | dish |
| `search_dietary_knowledge(query)` | 疾病饮食限制、营养建议、食物禁忌、过敏原 | dietary |
| `search_faq(query)` | 点餐流程、订单修改、配送时间、支付方式 | faq |

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
| GET | `/agent/memories?session_id=...` | 查看当前登录用户的长期记忆 |
| POST | `/agent/memories` | 按策略写入/更新记忆，外部引用自动进入 RAG 收件箱 |
| DELETE | `/agent/memories/{id}?session_id=...` | 删除指定长期记忆 |
| DELETE | `/agent/memories?session_id=...` | 删除当前用户的全部长期记忆 |

## Agent 评估

已接入脱敏执行轨迹，并提供 50 条种子样例、离线规则评分器和基础运行指标聚合。
评分将完整完成、部分完成、正确失败、错误失败分开统计；安全违规是独立门禁，
不能由其他高分抵消。轨迹中的 `done` 只表示一轮请求结束，不代表任务完成。
样例标注、运行命令、隐私约束和当前边界见 [evaluation/README.md](evaluation/README.md)。

## 长期记忆

长期记忆与 LangGraph 短期消息历史分离，并按登录用户隔离：

- `user_preference`：用户明确表达的长期偏好；
- `user_goal`：可更新的长期目标；
- `project_background`：完成任务所需的稳定背景；
- `task_state`：只存于 `session_task_states`，会话删除或过期即清理；
- `historical_conclusion`：只接受用户明确陈述或工具验证结果；
- `external_knowledge_reference`：不进入个人记忆，写到 RAG 审核收件箱。

写入策略拒绝密码、令牌与支付凭证；模型推断和敏感个人信息必须先经用户确认。
检索先识别当前任务，只加载对应类型的少量记忆，并记录使用次数。被新证据推翻的
历史结论会标记为 `superseded` 并降权。定时维护会清理过期任务状态、降低长期未用
记忆的权重、自动淘汰极低价值记忆，并最终清除删除标记。

### 排查 state（会话级，不写入长期记忆）

`update_investigation` 将结构化调查板写入 LangGraph 的 `investigation` state，
模型每次调用前会看到当前进展。可用结构例如：

```json
{
  "goal": "定位支付接口变慢原因",
  "current_hypothesis": "第三方支付回调超时可能导致接口变慢",
  "confirmed_facts": ["22:10 后 /api/pay P95 从 300ms 升到 3.8s", "22:05 发布了 payment-service v2.3.1"],
  "rejected_hypotheses": ["数据库慢查询导致接口变慢"],
  "next_actions": ["检查 v2.3.1 是否改动回调重试逻辑", "查询第三方支付回调错误码分布"],
  "status": "investigating",
  "conclusion": ""
}
```

同一 `goal` 的已确认事实和已排除假设会增量合并、去重；传入新的 `next_actions`
会替换旧待办。更换 `goal` 会开启新的调查，避免串案；换账号清空会话时也会清除
调查板。`confirmed_facts` 必须来自实际工具结果或用户明确提供的信息，模型猜测只放在
`current_hypothesis`。上例是结构示意，不表示本项目能查询支付服务日志。

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

菜品搜索固定执行 Milvus RAG 与实时检索。同一次调用使用相同 8 位 ID 串联日志，完整链路示例：

```text
[RAG tool][73ac0dd6] type=dish query='鸡肉'
[RAG][73ac0dd6] START type=dish final_top_k=8 recall_k=20 hybrid=True rerank=True
[RAG][73ac0dd6] EMBEDDING_DONE dim=1024 elapsed_ms=1201
[RAG][73ac0dd6] DENSE_DONE candidates=8 elapsed_ms=924          # 稠密语义检索
[RAG][73ac0dd6] BM25_DONE candidates=1 elapsed_ms=82             # BM25 关键词检索
[RAG][73ac0dd6] FUSE mode=HYBRID rrf_k=60 dense=8 bm25=1 fused=8   # RRF 融合
[RAG][73ac0dd6] DENSE_DIST candidates=8 max=0.590 min=0.564 avg=0.574 | ≥0.60=0 0.40-0.60=8 ...
[RAG][73ac0dd6] DENSE_TOP_K   D#1 ✓ cosine=0.590 【鸡块(5块)】[热销推荐] ...
[RAG][73ac0dd6] BM25_TOP_K    B#1 🔑 bm25=8.462 【辣子鸡】[川菜] ...
[RAG][73ac0dd6] FUSED_RANK    F#1 rrf=0.01639 [D#1 B#-] ...
[RAG][73ac0dd6] RERANK_DONE model=gte-rerank-v2 pre_candidates=8 elapsed_ms=318
[RAG][73ac0dd6] RERANK_DIST candidates=8 max=0.189 min=0.027 | ≥0.50=0 0.15-0.50=2 ...
[RAG][73ac0dd6] RERANK_DETAIL R#1 ✓ rerank=0.189 (RRF#1→#1) 【辣子鸡】[川菜] ...
[RAG][73ac0dd6] FILTER mode=RERANK reranked=8 accepted=5 rejected=3 threshold=0.050
[RAG][73ac0dd6] FINISH mode=RERANK accepted=5 elapsed_ms=2285
```

关键日志阶段：`METADATA_FILTER`（标量过滤）→ `QUERY_REWRITE`（Query改写）→ `EMBEDDING_DONE` → `DENSE_DONE`/`BM25_DONE`（双路召回）→ `FUSE`（RRF融合）→ `DENSE_DIST`/`BM25_DIST`（分数分布）→ `RERANK_DONE`/`RERANK_DIST`（精排）→ `FILTER`（阈值过滤）→ `FINISH`。

日志不会输出 API Key 或向量原值。

## RAG 可用性保障

Milvus 异常（如集合卡在 `Loading`）时，同步调用若无超时会无限阻塞，并拖满整轮对话预算。
检索链路对此做了多层保护，可在 `.env` 中调整：

```env
RAG_CALL_TIMEOUT=10       # 单次 Milvus RPC（加载集合/检索）超时（秒）
RAG_FAILURE_COOLDOWN=30   # 检索失败后的快速失败冷却时间（秒）
```

- **调用限时**：`load_collection` / `search` / `has_collection` 等均带显式超时；超时抛
  `KnowledgeStoreError`，工具层快速返回"知识库不可用"，不再阻塞整轮。
- **失败熔断**：某集合检索失败后进入冷却期，期间直接快速失败，避免每个请求都重复撞超时；
  冷却结束自动给一次恢复机会。"集合尚未构建"不算故障，不进入冷却，构建后即可立即恢复。
- **三级降级**：BM25 路故障 → 纯稠密；Rerank API 故障 → RRF 粗排；Query 改写故障 → 原始 Query。
  任何单路故障不影响整体检索可用性。
- **Embedding 缓存**：SQLite 磁盘缓存按 `模型|维度|版本|text_type|文本哈希` 去重，
  重建知识库时未变化的文本不重复计费。

## RAG 配置参考

所有 RAG 相关配置均在 `config.py` 中管理，可通过环境变量覆盖：

### 基础检索

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `RAG_TOP_K` | `8` | 最终返回条数 |
| `RAG_MIN_SCORE` | `0.25` | 稠密 COSINE 相似度下限（粗排阈值） |
| `RAG_METRIC_TYPE` | `COSINE` | 稠密向量度量类型 |
| `RAG_CALL_TIMEOUT` | `10` | 单次 Milvus RPC 超时（秒） |
| `RAG_FAILURE_COOLDOWN` | `30` | 检索失败冷却时间（秒） |

### 混合检索（稠密 + BM25）

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `RAG_HYBRID_SEARCH_ENABLED` | `true` | 混合检索总开关（集合需具备 sparse_vector 字段） |
| `RAG_HYBRID_RRF_K` | `60` | RRF 融合参数 k（越大排名差异越平） |
| `RAG_BM25_TOP_K` | `8` | BM25 路召回条数 |
| `RAG_BM25_DROP_RATIO` | `0.2` | BM25 忽略低 IDF term 比例 |
| `RAG_BM25_RESCUE_RANK` | `3` | BM25 精确命中补充通道排名阈值 |
| `RAG_SPARSE_FIELD` | `sparse_vector` | BM25 稀疏向量字段名 |

### Rerank 精排

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `RAG_RERANK_ENABLED` | `true` | Rerank 总开关 |
| `RAG_RERANK_MODEL` | `gte-rerank-v2` | 百炼 Cross-Encoder 模型 |
| `RAG_RERANK_CANDIDATES` | `20` | 精排候选数（召回阶段扩大到此值） |
| `RAG_RERANK_BATCH_SIZE` | `32` | 单次 rerank 请求最大文档数 |
| `RAG_RERANK_MIN_SCORE` | `0.05` | rerank 相关性分数下限（精排阈值） |
| `RAG_RERANK_TIMEOUT` | `15` | Rerank API 超时（秒） |

### Query 改写（默认关闭）

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `RAG_QUERY_REWRITE_ENABLED` | `false` | Query 改写总开关（启用会增加 LLM 调用延迟） |
| `RAG_QUERY_REWRITE_MODE` | `multi_query` | 改写模式：`multi_query` / `hyde` |
| `RAG_QUERY_REWRITE_NUM` | `3` | 多 Query 扩展数量 |
| `RAG_QUERY_REWRITE_TIMEOUT` | `10` | Query 改写 LLM 调用超时（秒） |

### 元数据过滤

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `RAG_METADATA_FILTER_ENABLED` | `true` | 元数据过滤总开关（`search_knowledge(filters=...)` 生效） |

### Embedding

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `RAG_EMBEDDING_MODEL` | `qwen3.7-text-embedding-flash` | 嵌入模型 |
| `RAG_EMBEDDING_DIMENSIONS` | `1024` | 向量维度 |
| `RAG_EMBEDDING_BATCH_SIZE` | `25` | 批量嵌入大小 |
| `RAG_EMBEDDING_TIMEOUT` | `30` | Embedding API 超时（秒） |
| `RAG_EMBEDDING_CACHE_DIR` | `./data/embedding_cache` | SQLite 缓存目录 |

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
