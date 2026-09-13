"""
FastAPI 主入口 —— 小鹿 AI 语音Agent服务（端口 8000）。

本文件只负责“应用装配”，接口实现按职责拆分到 routers/ 各子模块：
  POST /agent/sync              → routers/session.py   登录态同步
  DELETE /agent/session/{id}    → routers/session.py   删除会话
  POST /agent/text              → routers/chat.py      文本对话（JSON in / JSON out）
  POST /agent/text/stream       → routers/chat.py      文本对话（SSE 流式输出）
  POST /agent/tts               → routers/chat.py      统一文字播报（JSON in / MP3 out）
  POST /agent/voice             → routers/chat.py      语音对话（音频 in / MP3 out）
  GET  /agent/health            → routers/health.py    服务健康检查

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
import traceback as tb
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent import cleanup_sessions, shutdown as agent_shutdown
from config import (
    AGENT_CORS_ORIGINS,
    AGENT_HOST,
    AGENT_PORT,
    AGENT_RATE_LIMIT_PER_MINUTE,
    AGENT_RELOAD,
    AGENT_SERVICE_API_KEY,
    SESSION_CLEANUP_INTERVAL,
)
from routers import api_router
from routers.deps import rate_limiter

# 修复[错误分级]：配置日志，记录完整错误上下文
logger = logging.getLogger("main")


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
        await agent_shutdown()
        logger.info("[main] Agent 服务关闭，资源已释放")


app = FastAPI(
    title="小鹿 AI 语音Agent服务",
    description="面向视障人群的智能外卖平台 —— Agent服务",
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
        if not rate_limiter.allow(key):
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


# ==================== 路由挂载 ====================

app.include_router(api_router)


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
