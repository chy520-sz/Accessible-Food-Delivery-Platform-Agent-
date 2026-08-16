"""
FastAPI 主入口 —— 小鹿 AI 语音Agent服务（端口 8000）。

接口列表:
  POST /agent/text     — 文本对话（JSON in / JSON out）
  POST /agent/voice    — 语音对话（音频 in / MP3 out）
  GET  /agent/health   — 服务健康检查

安全与运维:
  - 配置 AGENT_SERVICE_API_KEY 后，所有 /agent/* 接口要求 X-Agent-Key 请求头
  - 按来源 IP 限流（/agent/health 除外）
  - CORS 白名单由 AGENT_CORS_ORIGINS 控制（默认仅本地开发端口）
  - 服务启动后后台定时清理过期会话，关闭时释放 LLM/HTTP 资源

启动方式:
  python main.py
  或:
  uvicorn main:app --host 0.0.0.0 --port 8000
"""
import asyncio
import logging
import time
import traceback as tb
import uuid
from collections import deque
from contextlib import asynccontextmanager
from typing import Optional

from urllib.parse import quote

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from agent import (
    chat,
    cleanup_sessions,
    get_agent_status,
    get_or_create_session,
    shutdown as agent_shutdown,
    sync_login_token,
    _sessions,
)
from config import (
    AGENT_CORS_ORIGINS,
    AGENT_HOST,
    AGENT_MAX_AUDIO_BYTES,
    AGENT_PORT,
    AGENT_RATE_LIMIT_PER_MINUTE,
    AGENT_RELOAD,
    AGENT_SERVICE_API_KEY,
    SESSION_CLEANUP_INTERVAL,
)
from speech import recognize_speech, synthesize_speech

# 修复[错误分级]：配置日志，记录完整错误上下文
logger = logging.getLogger("main")


# ==================== 限流器 ====================

class _RateLimiter:
    """按 key（来源 IP）的滑动窗口限流器。"""

    def __init__(self, limit: int, window: float = 60.0):
        self.limit = limit
        self.window = window
        self._hits: dict[str, deque] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        q = self._hits.setdefault(key, deque())
        while q and now - q[0] > self.window:
            q.popleft()
        if len(q) >= self.limit:
            return False
        q.append(now)
        return True


_rate_limiter = _RateLimiter(AGENT_RATE_LIMIT_PER_MINUTE)


# ==================== FastAPI 应用初始化 ====================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动后台会话清理任务，关闭时释放全局资源。"""
    cleanup_task = asyncio.create_task(_session_cleanup_loop())
    logger.info("[main] Agent 服务启动，限流=%d/min，鉴权=%s",
                AGENT_RATE_LIMIT_PER_MINUTE, "开启" if AGENT_SERVICE_API_KEY else "关闭(开发模式)")
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        agent_shutdown()
        logger.info("[main] Agent 服务关闭，资源已释放")


app = FastAPI(
    title="小鹿 AI 语音Agent服务",
    description="面向视障人群的智能语音外卖平台 —— Agent服务",
    version="1.1.0",
    lifespan=lifespan,
)

# CORS 配置：白名单来自环境变量，不使用 "*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=AGENT_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["x-session-id", "x-recognized-text", "x-reply-text"],
)


async def _session_cleanup_loop() -> None:
    """周期清理过期会话，避免进程内存无限增长。"""
    while True:
        await asyncio.sleep(SESSION_CLEANUP_INTERVAL)
        try:
            cleanup_sessions()
        except Exception:
            logger.exception("[main] 会话清理任务异常")


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """对 /agent/* 请求按来源 IP 限流（健康检查除外）。"""
    path = request.url.path
    if path.startswith("/agent") and not path.startswith("/agent/health"):
        key = request.client.host if request.client else "unknown"
        if not _rate_limiter.allow(key):
            trace_id = uuid.uuid4().hex[:12]
            logger.warning("[rate_limit] 429 ip=%s path=%s trace=%s", key, path, trace_id)
            return JSONResponse(
                status_code=429,
                content={
                    "code": 429,
                    "error_code": "CLT-1-429",
                    "message": "请求过于频繁，请稍后再试",
                    "trace_id": trace_id,
                },
            )
    return await call_next(request)


def require_service_key(x_agent_key: str = Header(default="")) -> None:
    """服务密钥鉴权：配置了 AGENT_SERVICE_API_KEY 时强制校验 X-Agent-Key。"""
    if AGENT_SERVICE_API_KEY and x_agent_key != AGENT_SERVICE_API_KEY:
        raise HTTPException(status_code=401, detail="无效的服务密钥")


# ==================== 全局异常处理 ====================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """兜底异常处理器：返回带 trace_id 的脱敏 500。"""
    trace_id = uuid.uuid4().hex[:12]
    logger.error(
        "[global_error] trace=%s path=%s method=%s exc_type=%s msg=%s\n%s",
        trace_id,
        request.url.path,
        request.method,
        type(exc).__name__,
        str(exc),
        tb.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content={
            "code": 500,
            "error_code": "SRV-3-001",
            "message": f"服务内部错误。请将此追踪ID反馈给技术支持：{trace_id}",
            "trace_id": trace_id,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP 异常处理 —— 区分 4xx 客户端错误和 5xx 服务端错误。"""
    trace_id = uuid.uuid4().hex[:12]
    if exc.status_code < 500:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "code": exc.status_code,
                "error_code": f"CLT-1-{exc.status_code:03d}",
                "message": exc.detail,
                "trace_id": trace_id,
            },
        )
    logger.error(
        "[http_error] trace=%s status=%d detail=%s",
        trace_id, exc.status_code, exc.detail,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": exc.status_code,
            "error_code": "SRV-3-002",
            "message": f"服务异常。请将此追踪ID反馈给技术支持：{trace_id}",
            "trace_id": trace_id,
        },
    )


# ==================== Pydantic 请求/响应模型 ====================

class SyncRequest(BaseModel):
    """登录态同步请求体"""
    session_id: Optional[str] = None
    auth_token: Optional[str] = None


class TextRequest(BaseModel):
    """文本对话请求体"""
    session_id: Optional[str] = None  # 会话ID，新会话传 null
    text: str  # 用户输入的文本
    auth_token: Optional[str] = None  # JWT token（从外卖平台前端传递）


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


# ==================== 接口实现 ====================

@app.post("/agent/sync", response_model=TextResponse, dependencies=[Depends(require_service_key)])
async def agent_sync(req: SyncRequest):
    """登录态同步接口。

    前端在登录/登出后调用此接口，让 Agent 会话同步前端登录态。
    这样就无需用户在与 Agent 对话时手动登录。

    请求示例:
      { "session_id": null, "auth_token": "eyJhbG..." }
      { "session_id": "abc123", "auth_token": null }

    响应示例:
      { "code": 200, "session_id": "abc123", "reply": "登录态已同步", "is_new_session": false }
    """
    sess = get_or_create_session(req.session_id)
    logged_in, username = sync_login_token(sess.session_id, req.auth_token)
    if req.auth_token and logged_in:
        msg = f"已同步登录态，欢迎 {username}"
    elif req.auth_token:
        msg = "登录态同步失败，请检查 Token 是否有效"
    else:
        msg = "已清除登录态"
    return TextResponse(
        session_id=sess.session_id,
        reply=msg,
        tts_text=msg,
        is_new_session=req.session_id is None,
    )


@app.post("/agent/text", response_model=TextResponse, dependencies=[Depends(require_service_key)])
async def agent_text(req: TextRequest):
    """文本对话接口。

    用户发送文字消息，Agent 返回文字回复。
    支持多轮对话，通过 session_id 维持上下文。

    请求示例:
      { "session_id": null, "text": "我想吃辣的" }
      { "session_id": "abc123", "text": "帮我下单" }

    响应示例:
      { "code": 200, "session_id": "abc123", "reply": "...", "is_new_session": true, "trace_id": "a1b2c3d4" }
    """
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="消息内容不能为空")

    # 对话请求附加系统毫秒级日志时间戳，便于性能分析
    _start = time.time()

    result = await chat(req.session_id, req.text.strip(), req.auth_token)

    _elapsed = (time.time() - _start) * 1000
    logger.info(
        "[agent_text] session=%s elapsed=%.0fms reply_len=%d",
        result["session_id"], _elapsed, len(result["reply"]),
    )

    return TextResponse(
        code=200,
        session_id=result["session_id"],
        reply=result["reply"],
        tts_text=result.get("tts_text", result["reply"]),
        is_new_session=result["is_new_session"],
        trace_id=result.get("trace_id", ""),
    )


@app.post("/agent/voice", dependencies=[Depends(require_service_key)])
async def agent_voice(
    audio_file: UploadFile = File(...),
    audio_format: str = Form(default="wav"),
    session_id: Optional[str] = Form(default=None),
):
    """语音对话接口。

    接收用户录音（WAV/PCM），经过 ASR → LLM → TTS 链路，
    返回 MP3 音频和识别/回复文本。

    请求格式: multipart/form-data
      - audio_file: 音频文件（16kHz, 16bit, 单声道 WAV/PCM，最大 10MB）
      - audio_format: 音频格式，"wav" 或 "pcm"（默认 wav）
      - session_id: 会话ID（可选，用于多轮对话）

    响应:
      - Headers: x-session-id, x-recognized-text, x-reply-text
      - Body: MP3 音频二进制数据
    """
    # 1. 读取上传的音频数据（限制大小与格式）
    audio_bytes = await audio_file.read()
    if not audio_bytes or len(audio_bytes) < 1600:
        raise HTTPException(status_code=400, detail="音频数据过短，请重新录音")
    if len(audio_bytes) > AGENT_MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"音频文件过大（最大 {AGENT_MAX_AUDIO_BYTES // 1024 // 1024}MB）",
        )
    if audio_format not in ("wav", "pcm"):
        raise HTTPException(status_code=400, detail="audio_format 仅支持 wav 或 pcm")
    logger.info(
        "[voice] 收到音频: %d 字节, 格式: %s, content_type: %s",
        len(audio_bytes), audio_format, audio_file.content_type,
    )

    # 2. ASR 语音识别
    recognized = await recognize_speech(audio_bytes, audio_format)
    logger.info("[voice] ASR 识别结果: '%s'", recognized)
    if not recognized:
        raise HTTPException(status_code=422, detail="语音识别失败，请重试或使用文字输入")

    # 3. LLM 文本对话
    result = await chat(session_id, recognized)
    reply = result["reply"]
    tts_text = result.get("tts_text", reply)
    new_session_id = result["session_id"]
    logger.info("[voice] LLM 回复: '%s'", reply[:100])

    # 4. TTS 语音合成 —— 使用清理后的 tts_text
    mp3_bytes = await synthesize_speech(tts_text)
    logger.info("[voice] TTS 合成: %d 字节", len(mp3_bytes))

    if not mp3_bytes:
        logger.warning("[voice] TTS 返回空音频，降级返回文本")

    return Response(
        content=mp3_bytes,
        media_type="audio/mpeg",
        headers={
            "x-session-id": new_session_id,
            "x-recognized-text": quote(recognized, safe=""),
            "x-reply-text": quote(reply, safe=""),
            "x-tts-text": quote(tts_text, safe=""),
        },
    )


@app.get("/agent/health", response_model=HealthResponse)
async def agent_health():
    """健康检查接口。

    返回 Agent 服务的运行状态、模型连接情况、活跃会话数等。
    供 Java 前端或运维监控调用。

    响应示例:
      { "status": "ok", "model": "deepseek-chat", "active_sessions": 3, ... }
    """
    status = await get_agent_status()
    return HealthResponse(**status)


@app.delete("/agent/session/{session_id}", dependencies=[Depends(require_service_key)])
async def agent_session_delete(session_id: str):
    """清除指定会话（主要用于调试）。"""
    if session_id in _sessions:
        del _sessions[session_id]
        return JSONResponse({"code": 200, "message": f"会话 {session_id} 已清除"})
    return JSONResponse({"code": 404, "message": f"会话 {session_id} 不存在"})


# ==================== 启动入口 ====================

if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("   🦌  小鹿 AI语音Agent服务")
    print("   面向视障人群的智能外卖平台")
    print("=" * 50)
    print(f"   监听地址: http://{AGENT_HOST}:{AGENT_PORT}")
    print(f"   API文档: http://localhost:{AGENT_PORT}/docs")
    print(f"   reload: {AGENT_RELOAD}")
    print("=" * 50)
    uvicorn.run(
        "main:app",
        host=AGENT_HOST,
        port=AGENT_PORT,
        reload=AGENT_RELOAD,
        log_level="info",
    )
