"""
配置文件 —— 集中管理所有可调参数和环境变量。
请复制 .env.example 为 .env 并填入您的真实凭据。
"""
import os

from dotenv import load_dotenv

load_dotenv()

# Python 3.14 + httpcore SSL 兼容性问题，本地开发环境关闭 SSL 验证
# (DeepSeek API 和 Java 后端都是本地/内网环境)
SSL_VERIFY = os.getenv("SSL_VERIFY", "false").lower() in ("true", "1", "yes")


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
AGENT_HOST = os.getenv("AGENT_HOST", "0.0.0.0")

# ==================== 会话管理 ====================
SESSION_EXPIRE_SECONDS = int(os.getenv("SESSION_EXPIRE_SECONDS", "1800"))
MAX_LLM_TOKENS = int(os.getenv("MAX_LLM_TOKENS", "2000"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# ==================== Agent API Key ====================
# 用于 Agent 调用 Java 后端专用接口（如模拟商家接单/配送），请设置为复杂随机字符串
AGENT_API_KEY = os.getenv("AGENT_API_KEY", "change-me-in-production")

# ==================== 重试配置 ====================
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "0.5"))

# ==================== RAG 知识库配置 ====================
# 向量数据库持久化目录
RAG_PERSIST_DIR = os.getenv("RAG_PERSIST_DIR", "./chroma_db")
# 嵌入模型（BAAI/bge-small-zh-v1.5: 本地免费，中文效果优秀，~100MB）
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
# 每次检索返回条数
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "8"))
# 知识库集合名称
RAG_COLLECTION_FOOD = os.getenv("RAG_COLLECTION_FOOD", "dish_knowledge")
RAG_COLLECTION_DIETARY = os.getenv("RAG_COLLECTION_DIETARY", "dietary_knowledge")
RAG_COLLECTION_FAQ = os.getenv("RAG_COLLECTION_FAQ", "faq")

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
    if not ALIYUN_ACCESS_KEY_ID:
        warnings.append("ALIYUN_ACCESS_KEY_ID 未设置 —— 语音识别将不可用")
    if not ALIYUN_ACCESS_KEY_SECRET:
        warnings.append("ALIYUN_ACCESS_KEY_SECRET 未设置 —— 语音识别将不可用")
    if warnings:
        print("[config] ⚠️  配置警告:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("[config] ✅ 关键配置检查通过")

_validate_config()
