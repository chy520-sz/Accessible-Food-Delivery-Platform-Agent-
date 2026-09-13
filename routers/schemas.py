"""HTTP 请求 / 响应模型 —— 与原有接口字段保持完全一致。"""

from typing import Optional

from pydantic import BaseModel, Field

from config import AGENT_MAX_TTS_CHARS


class SyncRequest(BaseModel):
    """登录态同步请求体"""
    session_id: Optional[str] = None
    auth_token: Optional[str] = None


class TextRequest(BaseModel):
    """文本对话请求体"""
    session_id: Optional[str] = None  # 会话ID，新会话传 null
    text: str  # 用户输入的文本
    auth_token: Optional[str] = None  # JWT token（从外卖平台前端传递）


class TtsRequest(BaseModel):
    """统一文字转语音请求体。"""
    text: str = Field(min_length=1, max_length=AGENT_MAX_TTS_CHARS)


class TextResponse(BaseModel):
    """文本对话响应体"""
    code: int = 200
    session_id: str
    reply: str
    tts_text: str = ""
    is_new_session: bool = False
    # 每次请求生成唯一 trace_id，便于前后端联合排障
    trace_id: str = ""


class HealthResponse(BaseModel):
    """健康检查响应体"""
    status: str
    model: str
    active_sessions: int
    llm_connected: bool
    java_backend: str
    # RAG（Milvus + 嵌入）就绪状态
    milvus_up: bool = False
    rag_ready: bool = False
    embedding_ok: bool = False
    rag_collections: dict = {}
