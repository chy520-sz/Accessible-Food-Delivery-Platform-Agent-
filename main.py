"""
FastAPI 主入口 —— 小鹿 AI 语音Agent服务（端口 8000）。

接口列表:
  POST /agent/text     — 文本对话（JSON in / JSON out）
  POST /agent/voice    — 语音对话（音频 in / MP3 out）
  GET  /agent/health   — 服务健康检查

启动方式:
  python main.py
  或:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
import logging
import traceback as tb
import uuid
from typing import Optional

from urllib.parse import quote

from fastapi import FastAPI, HTTPException, File, Request, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from agent import chat, get_agent_status, get_or_create_session, _sessions, sync_login_token
from config import AGENT_HOST, AGENT_PORT
from speech import recognize_speech, synthesize_speech

# 修复[错误分级]：配置日志，记录完整错误上下文
logger = logging.getLogger("main")

# ==================== FastAPI 应用初始化 ====================

app = FastAPI(
    title="小鹿 AI 语音Agent服务",
    description="面向视障人群的智能语音外卖平台 —— Agent服务",
    version="1.0.0",
)

# CORS 配置：允许 Java 前端（3000）跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "*",  # 比赛演示环境宽松配置
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["x-session-id", "x-recognized-text", "x-reply-text"],
)


# ==================== 全局异常处理 ====================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """兜底异常处理器。

    修复[根因]：此前未捕获异常直接由 FastAPI 默认 handler 处理，返回无 trace_id 的
    通用 500 错误（如 agent.py 中 get_or_create_session 在 try 外抛异常时）。
    现在生成 UUID trace_id，记录含完整堆栈的结构化日志，返回脱敏后含追踪 ID 的响应。
    """
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
    """HTTP 异常处理 —— 区分 4xx 客户端错误和 5xx 服务端错误。

    修复[错误分级]：4xx 错误直接透传具体原因（参数缺失、格式错误等）；
    5xx 错误附加 trace_id 供排查，不再笼统返回"系统出了错误"。
    """
    trace_id = uuid.uuid4().hex[:12]
    if exc.status_code < 500:
        # 4xx 客户端错误：精确返回校验失败原因
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "code": exc.status_code,
                "error_code": f"CLT-1-{exc.status_code:03d}",
                "message": exc.detail,
                "trace_id": trace_id,
            },
        )
    # 5xx 服务端错误：脱敏后附加 trace_id
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
    # 修复[错误分级]：每次请求生成唯一 trace_id，便于前后端联合排障
    trace_id: str = ""


class HealthResponse(BaseModel):
    """健康检查响应体"""
    status: str
    model: str
    active_sessions: int
    llm_connected: bool
    java_backend: str


# ==================== 接口实现 ====================

@app.post("/agent/sync", response_model=TextResponse)
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


@app.post("/agent/text", response_model=TextResponse)
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

    # 修复[错误分级]：对话请求附加系统毫秒级日志时间戳，便于性能分析
    import time as _time
    _start = _time.time()

    result = await chat(req.session_id, req.text.strip(), req.auth_token)

    _elapsed = (_time.time() - _start) * 1000
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


@app.post("/agent/voice")
async def agent_voice(
    audio_file: UploadFile = File(...),
    audio_format: str = Form(default="wav"),
    session_id: Optional[str] = Form(default=None),
):
    """语音对话接口。

    接收用户录音（WAV/PCM），经过 ASR → LLM → TTS 链路，
    返回 MP3 音频和识别/回复文本。

    请求格式: multipart/form-data
      - audio_file: 音频文件（16kHz, 16bit, 单声道 WAV/PCM）
      - audio_format: 音频格式，"wav" 或 "pcm"（默认 wav）
      - session_id: 会话ID（可选，用于多轮对话）

    响应:
      - Headers: x-session-id, x-recognized-text, x-reply-text
      - Body: MP3 音频二进制数据
    """
    # 1. 读取上传的音频数据
    audio_bytes = await audio_file.read()
    print(f"[Voice] 收到音频: {len(audio_bytes)} 字节, 格式: {audio_format}")
    if not audio_bytes or len(audio_bytes) < 1600:
        raise HTTPException(status_code=400, detail="音频数据过短，请重新录音")

    # 2. ASR 语音识别
    recognized = await recognize_speech(audio_bytes, audio_format)
    print(f"[Voice] ASR 识别结果: '{recognized}'")
    if not recognized:
        raise HTTPException(status_code=422, detail="语音识别失败，请重试或使用文字输入")

    # 3. LLM 文本对话
    result = await chat(session_id, recognized)
    reply = result["reply"]
    tts_text = result.get("tts_text", reply)
    new_session_id = result["session_id"]
    print(f"[Voice] LLM 回复: '{reply[:100]}{'...' if len(reply) > 100 else ''}'")

    # 4. TTS 语音合成 —— 使用清理后的 tts_text
    mp3_bytes = await synthesize_speech(tts_text)
    print(f"[Voice] TTS 合成: {len(mp3_bytes)} 字节")

    if not mp3_bytes:
        print("[Voice] TTS 返回空音频，降级返回文本")
        return Response(
            content=b"",
            media_type="audio/mpeg",
            headers={
                "x-session-id": new_session_id,
                "x-recognized-text": quote(recognized, safe=""),
                "x-reply-text": quote(reply, safe=""),
                "x-tts-text": quote(tts_text, safe=""),
            },
        )

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


@app.delete("/agent/session/{session_id}")
async def agent_session_delete(session_id: str):
    """清除指定会话（主要用于调试）。

    参数:
        session_id: 要清除的会话ID
    """
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
    print(f"   LLM模型: deepseek-chat (DeepSeek)")
    print(f"   Java后端: http://localhost:3000")
    print(f"   API文档: http://localhost:{AGENT_PORT}/docs")
    print("=" * 50)
    uvicorn.run(
        "main:app",
        host=AGENT_HOST,
        port=AGENT_PORT,
        reload=True,
        log_level="info",
    )
