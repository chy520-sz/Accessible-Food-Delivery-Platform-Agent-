"""
配置文件 —— 集中管理所有可调参数和环境变量。
请复制 .env.example 为 .env 并填入您的真实凭据。
"""
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def _enable_utf8_stdout() -> None:
    """Windows 控制台默认 GBK 编码，直接 print 中文/emoji 可能崩溃；
    统一切换为 UTF-8 容错编码。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


_enable_utf8_stdout()

# SSL 证书校验默认开启；仅在确有证书问题（如内网自签证书）时显式设置 SSL_VERIFY=false
SSL_VERIFY = os.getenv("SSL_VERIFY", "true").lower() in ("true", "1", "yes")


# ==================== Java 后端连接 ====================
JAVA_BASE_URL = os.getenv("JAVA_BASE_URL", "http://localhost:3000")
JAVA_API_PREFIX = "/api"

# ==================== 大模型配置 ====================
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
# 修复[LLM超时]：增加 LLM API 调用超时配置，防止工具链长调用被默认短超时截断导致 500
LLM_REQUEST_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", "120"))

# ==================== 阿里云智能语音交互 ====================
# 请按以下步骤获取凭据：
# 1. 注册阿里云账号并实名认证
# 2. 登录 RAM 控制台创建 AccessKey（含 ID 和 Secret）
# 3. 开通智能语音交互服务（新用户 3 个月免费试用）
# 4. 在智能语音交互控制台创建项目，获取 Appkey
# 参考文档: https://help.aliyun.com/document_detail/72138.html

ALIYUN_ACCESS_KEY_ID = os.getenv("ALIYUN_ACCESS_KEY_ID", "")
ALIYUN_ACCESS_KEY_SECRET = os.getenv("ALIYUN_ACCESS_KEY_SECRET", "")
ALIYUN_ASR_APP_KEY = os.getenv("ALIYUN_ASR_APP_KEY", "")

# 语音识别 API 地址（一句话识别 WebSocket 协议）
ASR_WS_URL = os.getenv(
    "ASR_WS_URL",
    "wss://nls-gateway-cn-shanghai.aliyuncs.com/ws/v1",
)

# ==================== Edge TTS 配置 ====================
TTS_VOICE = os.getenv("TTS_VOICE", "zh-CN-XiaoxiaoNeural")  # 温暖女声
TTS_RATE = os.getenv("TTS_RATE", "-10%")  # 语速稍慢，适合视障用户
TTS_OUTPUT_DIR = os.getenv("TTS_OUTPUT_DIR", "./tts_output")

# ==================== Agent 服务配置 ====================
AGENT_PORT = int(os.getenv("AGENT_PORT", "8000"))
AGENT_HOST = os.getenv("AGENT_HOST", "127.0.0.1")

# ==================== Agent 运行预算 ====================
# LangGraph 图递归上限（一次请求内 model/tool 节点交替的总步数上限）。
# 注意：它不等价于旧 AgentExecutor 的 max_iterations=10。
AGENT_RECURSION_LIMIT = int(os.getenv("AGENT_RECURSION_LIMIT", "25"))
# 一次对话内最多允许的工具调用次数
AGENT_MAX_TOOL_CALLS = int(os.getenv("AGENT_MAX_TOOL_CALLS", "10"))
# 一次对话内最多允许的模型调用次数
AGENT_MAX_MODEL_CALLS = int(os.getenv("AGENT_MAX_MODEL_CALLS", "12"))
# 单次对话总耗时预算（秒），覆盖模型+工具链路
AGENT_TOTAL_TIMEOUT_SECONDS = float(os.getenv("AGENT_TOTAL_TIMEOUT_SECONDS", "115"))
# 长对话消息窗口：checkpoint 中最多保留的最近消息条数（工具调用/结果成对裁剪）
AGENT_MESSAGE_WINDOW = int(os.getenv("AGENT_MESSAGE_WINDOW", "24"))

# ==================== 会话管理 ====================
SESSION_EXPIRE_SECONDS = int(os.getenv("SESSION_EXPIRE_SECONDS", "1800"))
MAX_LLM_TOKENS = int(os.getenv("MAX_LLM_TOKENS", "2000"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# ==================== Agent API Key ====================
# 用于 Agent 调用 Java 后端专用接口（如模拟商家接单/配送），请设置为复杂随机字符串
# 注意：该值必须与 Java 后端 application.yml 的 agent.api-key（或环境变量 AGENT_API_KEY）一致
AGENT_API_KEY = os.getenv("AGENT_API_KEY", "change-me-in-production")

# ==================== Agent 服务对外鉴权 ====================
# 客户端调用 /agent/* 时需携带 X-Agent-Key 请求头。
# 留空表示开发模式（不校验）；生产环境请务必设置，并同步配置前端 VITE_AGENT_KEY。
AGENT_SERVICE_API_KEY = os.getenv("AGENT_SERVICE_API_KEY", "")

# CORS 允许来源（逗号分隔）。不再使用 "*"。
_DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000,http://localhost:5173,http://localhost:5174,"
    "http://localhost:5175,http://127.0.0.1:3000,http://127.0.0.1:5173,"
    "http://127.0.0.1:5174,http://127.0.0.1:5175"
)
AGENT_CORS_ORIGINS = [
    o.strip() for o in os.getenv("AGENT_CORS_ORIGINS", _DEFAULT_CORS_ORIGINS).split(",")
    if o.strip()
]

# 语音上传大小上限（默认 10MB）
AGENT_MAX_AUDIO_BYTES = int(os.getenv("AGENT_MAX_AUDIO_BYTES", str(10 * 1024 * 1024)))
# 单次文字转语音的最大字符数，避免超长文本拖慢播报服务
AGENT_MAX_TTS_CHARS = int(os.getenv("AGENT_MAX_TTS_CHARS", "2000"))

# 每个来源 IP 每分钟最多请求数（覆盖 /agent/*，健康检查除外）
AGENT_RATE_LIMIT_PER_MINUTE = int(os.getenv("AGENT_RATE_LIMIT_PER_MINUTE", "120"))

# 启动时是否开启 uvicorn reload（生产环境应关闭）
AGENT_RELOAD = os.getenv("AGENT_RELOAD", "false").lower() in ("1", "true", "yes")

# ==================== 重试配置 ====================
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "0.5"))

# ==================== 健康探活与会话清理 ====================
# LLM 健康状态缓存有效期（秒），期间不重复真调大模型
LLM_HEALTH_TTL = int(os.getenv("LLM_HEALTH_TTL", "300"))
# 过期会话清理任务的执行间隔（秒）
SESSION_CLEANUP_INTERVAL = int(os.getenv("SESSION_CLEANUP_INTERVAL", "60"))

# ==================== RAG 知识库配置（DashScope 嵌入 + Milvus） ====================
# ---------- 百炼 DashScope 文本嵌入 ----------
# 百炼 API Key（与语音识别的 AccessKey/AppKey 是两套凭据，需分别填写）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
# 按百炼控制台的地域和业务空间填写，例如 https://<业务空间ID>.cn-beijing.maas.aliyuncs.com/api/v1
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"
)
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "qwen3.7-text-embedding-flash")
# Flash 支持 1024/768/512/256，本项目固定 1024 维
RAG_EMBEDDING_DIMENSIONS = int(os.getenv("RAG_EMBEDDING_DIMENSIONS", "1024"))
# 单批最多 20 条，首批保守按 10 条发送
RAG_EMBEDDING_BATCH_SIZE = int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "10"))
RAG_EMBEDDING_TIMEOUT = float(os.getenv("RAG_EMBEDDING_TIMEOUT", "30"))
# 嵌入结果本地磁盘缓存目录（按 模型/维度/处理版本/文本哈希 去重，减少重复计费）
RAG_EMBEDDING_CACHE_DIR = os.getenv("RAG_EMBEDDING_CACHE_DIR", "./.embedding_cache")
# 文本处理版本：改变分块/清洗逻辑时递增，使旧缓存与旧向量自然失效
RAG_CONTENT_VERSION = os.getenv("RAG_CONTENT_VERSION", "v1")

# ---------- Milvus Standalone ----------
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME", "default")
# 第一阶段数据量小，使用 FLAT + COSINE；后续压测后再改 HNSW
RAG_INDEX_TYPE = os.getenv("RAG_INDEX_TYPE", "FLAT")
RAG_METRIC_TYPE = os.getenv("RAG_METRIC_TYPE", "COSINE")

# 每次检索返回条数（起点）
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "8"))
# COSINE 相似度下限（越大越相似）；低于该值视为无关，允许返回“未找到”。
# 该阈值需用项目评测问题校准，这里仅给保守默认值。
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.25"))
# 知识库集合名称（新版本集合，维度/模型体现在名字里）
RAG_COLLECTION_FOOD = os.getenv("RAG_COLLECTION_FOOD", "takeout_dish_qwen37_1024_v1")
RAG_COLLECTION_DIETARY = os.getenv("RAG_COLLECTION_DIETARY", "takeout_dietary_qwen37_1024_v1")
RAG_COLLECTION_FAQ = os.getenv("RAG_COLLECTION_FAQ", "takeout_faq_qwen37_1024_v1")

# 逻辑知识类型 -> 物理集合名（格式化逻辑按逻辑类型判断，不再硬编码物理集合名）
RAG_KNOWLEDGE_COLLECTIONS = {
    "dish": RAG_COLLECTION_FOOD,
    "dietary": RAG_COLLECTION_DIETARY,
    "faq": RAG_COLLECTION_FAQ,
}

# ==================== 营养估算（千卡/100g 近似值） ====================
CALORIE_ESTIMATE_MAP = {
    "鸡": 167, "鸭": 240, "鱼": 110, "虾": 93, "蟹": 95,
    "猪肉": 395, "牛肉": 125, "羊肉": 203, "蛋": 144,
    "蔬菜": 30, "青菜": 25, "白菜": 13, "菠菜": 28,
    "番茄": 20, "土豆": 81, "茄子": 21, "豆腐": 82,
    "米饭": 116, "面": 110, "馒头": 223, "粥": 46,
    "汤": 35, "炒": 120, "炸": 280, "蒸": 80, "烤": 180,
    "辣": 80, "酸": 40, "甜": 150, "沙拉": 50, "水果": 50,
    "牛奶": 54, "豆浆": 31, "茶": 1, "咖啡": 2,
}

# 修复[启动校验]：在服务启动时检查关键环境变量，若未配置则打印明确警告
def _validate_config():
    """启动时校验关键配置，打印缺失项警告而非静默失败。"""
    warnings = []
    if not DEEPSEEK_API_KEY:
        warnings.append("DEEPSEEK_API_KEY 未设置 —— LLM 调用将全部失败")
    if AGENT_API_KEY == "change-me-in-production":
        warnings.append(
            "AGENT_API_KEY 仍为默认值 —— 请设置为与 Java 后端 agent.api-key 一致的复杂随机字符串"
        )
    if not AGENT_SERVICE_API_KEY:
        warnings.append(
            "AGENT_SERVICE_API_KEY 未设置 —— /agent/* 接口处于开发模式（不校验调用方身份），生产环境请务必配置"
        )
    if not ALIYUN_ACCESS_KEY_ID:
        warnings.append("ALIYUN_ACCESS_KEY_ID 未设置 —— 语音识别将不可用（纯文本联调可忽略）")
    if not ALIYUN_ACCESS_KEY_SECRET:
        warnings.append("ALIYUN_ACCESS_KEY_SECRET 未设置 —— 语音识别将不可用（纯文本联调可忽略）")
    if not DASHSCOPE_API_KEY:
        warnings.append("DASHSCOPE_API_KEY 未设置 —— 知识库嵌入与 RAG 检索将不可用")
    if warnings:
        print("[config] [WARN] 配置警告:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("[config] [OK] 关键配置检查通过")

_validate_config()
