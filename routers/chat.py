"""对话相关路由 —— 文字对话、流式文字对话、语音合成、语音对话。"""

import json
import logging
import time
from typing import AsyncIterator, Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse

from agent import chat, stream_chat
from config import AGENT_MAX_AUDIO_BYTES
from routers.deps import require_service_key
from routers.schemas import TextRequest, TextResponse, TtsRequest
from speech import recognize_speech, synthesize_speech
from text_utils import clean_text_for_tts

logger = logging.getLogger("main")

router = APIRouter(tags=["对话"])


def _sse_event(event: str, data: dict) -> str:
    """把一条结构化事件编码成 SSE 帧（JSON，保留中文）。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/agent/text", response_model=TextResponse, dependencies=[Depends(require_service_key)])
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


@router.post("/agent/text/stream", dependencies=[Depends(require_service_key)])
async def agent_text_stream(req: TextRequest):
    """流式文本对话接口（Server-Sent Events）。

    与 /agent/text 使用同一套会话、鉴权与错误语义，只是把大模型输出
    按 token 增量推送，首字延迟更低。事件序列：

      event: session  data: {"session_id","is_new_session","trace_id"}
      event: delta    data: {"text":"增量片段"}            # 可多次
      event: done     data: {"session_id","reply","tts_text","is_new_session","trace_id"}
      event: error    data: {"message","trace_id"}          # 出错时替代 done

    前端可先渲染 delta 文本，收到 done 后再用完整 reply 覆盖并播报 TTS。
    """
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="消息内容不能为空")

    user_text = req.text.strip()
    session_id = req.session_id
    auth_token = req.auth_token

    async def _event_stream() -> AsyncIterator[str]:
        started = time.time()
        reply_len = 0
        try:
            async for ev in stream_chat(session_id, user_text, auth_token):
                etype = ev.get("type")
                if etype == "session":
                    yield _sse_event("session", {
                        "session_id": ev["session_id"],
                        "is_new_session": ev["is_new_session"],
                        "trace_id": ev["trace_id"],
                    })
                elif etype == "delta":
                    text = ev.get("text", "")
                    reply_len += len(text)
                    yield _sse_event("delta", {"text": text})
                elif etype == "done":
                    yield _sse_event("done", {
                        "session_id": ev["session_id"],
                        "reply": ev["reply"],
                        "tts_text": ev.get("tts_text", ev["reply"]),
                        "is_new_session": ev["is_new_session"],
                        "trace_id": ev["trace_id"],
                    })
                elif etype == "error":
                    yield _sse_event("error", {
                        "message": ev["message"],
                        "trace_id": ev.get("trace_id", ""),
                    })
        except Exception:
            # 兜底：流已经开始，无法再改 HTTP 状态码，只能以 error 事件收尾
            logger.exception("[agent_text_stream] 流式输出异常")
            yield _sse_event("error", {"message": "服务内部错误，请稍后重试", "trace_id": ""})
        finally:
            logger.info(
                "[agent_text_stream] elapsed=%.0fms reply_len=%d",
                (time.time() - started) * 1000, reply_len,
            )

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/agent/tts", dependencies=[Depends(require_service_key)])
async def agent_tts(req: TtsRequest):
    """将系统提示或文字对话统一合成为小鹿音色的 MP3。"""
    tts_text = clean_text_for_tts(req.text).strip()
    if not tts_text:
        raise HTTPException(status_code=400, detail="播报文本不能为空")
    started = time.perf_counter()
    mp3_bytes = await synthesize_speech(tts_text)
    if not mp3_bytes:
        raise HTTPException(status_code=503, detail="语音合成暂时不可用")
    logger.info(
        "[tts] 合成完成 chars=%d bytes=%d elapsed_ms=%.0f",
        len(tts_text), len(mp3_bytes), (time.perf_counter() - started) * 1000,
    )
    return Response(content=mp3_bytes, media_type="audio/mpeg")


@router.post("/agent/voice", dependencies=[Depends(require_service_key)])
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

    # 3. LLM 文本对话（语音需要完整文本才能合成，这里聚合完整回复）
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
