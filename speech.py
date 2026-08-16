"""
语音处理模块 —— 阿里云一句话识别 (ASR) + Edge TTS 语音合成。

ASR: 使用阿里云智能语音交互 WebSocket API — 先通过 POP API 获取 Token，
     再通过 WebSocket 协议发送音频，返回识别文本。
TTS: 使用 edge-tts 免费合成，无需 API Key，支持丰富的中文音色。

参考文档:
  - 一句话识别 WebSocket API: https://help.aliyun.com/document_detail/43878.html
  - Token 获取: https://help.aliyun.com/document_detail/450514.html
"""
import base64
import asyncio
import hashlib
import hmac
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from urllib.parse import quote

import httpx

from config import (
    ALIYUN_ACCESS_KEY_ID,
    ALIYUN_ACCESS_KEY_SECRET,
    ALIYUN_ASR_APP_KEY,
    ASR_WS_URL,
    TTS_VOICE,
    TTS_RATE,
    MAX_RETRIES,
    RETRY_DELAY,
)

logger = logging.getLogger("speech")

# ==================== 凭据校验 ====================

_MISSING_CREDENTIALS = []
if not ALIYUN_ACCESS_KEY_ID:
    _MISSING_CREDENTIALS.append("ALIYUN_ACCESS_KEY_ID")
if not ALIYUN_ACCESS_KEY_SECRET:
    _MISSING_CREDENTIALS.append("ALIYUN_ACCESS_KEY_SECRET")
if not ALIYUN_ASR_APP_KEY:
    _MISSING_CREDENTIALS.append("ALIYUN_ASR_APP_KEY")


def _check_credentials():
    """在首次调用 ASR 时校验凭据，给出明确的配置指引。"""
    if _MISSING_CREDENTIALS:
        missing = ", ".join(_MISSING_CREDENTIALS)
        raise RuntimeError(
            f"阿里云语音服务配置不完整，缺少: {missing}\n"
            "请按以下步骤获取凭据：\n"
            "  1. 注册阿里云账号并完成个人实名认证\n"
            "  2. 登录 RAM 控制台 (https://ram.console.aliyun.com) 创建 AccessKey\n"
            "  3. 开通智能语音交互服务 (https://ai.aliyun.com/nls)\n"
            "  4. 在智能语音交互控制台创建项目，获取 Appkey\n"
            "  5. 将凭据写入 .env 文件（参考 .env.example）：\n"
            "       ALIYUN_ACCESS_KEY_ID=你的AccessKeyId\n"
            "       ALIYUN_ACCESS_KEY_SECRET=你的AccessKeySecret\n"
            "       ALIYUN_ASR_APP_KEY=你的Appkey\n"
            "参考文档: https://help.aliyun.com/document_detail/72138.html"
        )


# ==================== 阿里云 Token 获取 ====================
# Token 通过阿里云 POP API 获取，使用 AK/SK 签名鉴权。
# 有效期通常为 24 小时，这里做简单缓存。

_token_cache: dict = {"token": "", "expire_time": 0}

# Token 获取 API 地址
_TOKEN_API_URL = "https://nls-meta.cn-shanghai.aliyuncs.com/pop/2019-02-28/tokens"


def _get_aliyun_token() -> str:
    """获取阿里云智能语音交互服务的临时访问 Token。

    通过 AK/SK 进行 POP API 签名，向 nls-meta 服务换取 Token，
    并缓存至过期时间。

    参考文档: https://help.aliyun.com/document_detail/450514.html
    """
    _check_credentials()

    now = int(datetime.now(timezone.utc).timestamp())
    if _token_cache["token"] and now < _token_cache["expire_time"]:
        return _token_cache["token"]

    # RPC 风格 API：Signature 在 query string 中，不在 Authorization 头
    query_string, signature = _build_token_request()
    url = f"{_TOKEN_API_URL}?{query_string}&Signature={_pop_encode(signature)}"

    try:
        with httpx.Client(timeout=10.0, verify=False) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()
            token_info = data.get("Token", {})
            token = token_info.get("Id", "")
            expire = token_info.get("ExpireTime", now + 86400)
            if not token:
                raise RuntimeError("Token 响应为空，请检查 AccessKey 是否有效")
            _token_cache["token"] = token
            _token_cache["expire_time"] = expire
            return token
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            # 打印完整响应用于调试
            resp_body = e.response.text[:500] if e.response.text else "(空)"
            raise RuntimeError(
                "获取阿里云语音 Token 失败 (404)。\n"
                f"响应内容: {resp_body}\n"
                f"请求 AccessKeyId: {ALIYUN_ACCESS_KEY_ID[:8]}***\n"
                "\n"
                "请按以下顺序逐一排查：\n"
                "  1. 智能语音交互服务是否已开通？\n"
                "     访问 https://ai.aliyun.com/nls 点击「开通并购买」\n"
                "     （未开通时 API 网关直接返回 404，不会有其他错误码）\n"
                "\n"
                "  2. 该 AccessKey 对应的 RAM 用户是否已授权？\n"
                "     登录 https://ram.console.aliyun.com/users\n"
                "     找到该用户 → 添加权限 → 选择 AliyunNLSFullAccess\n"
                "     （没有此权限时，API 同样返回 404 而非 403）\n"
                "\n"
                "  3. AccessKey ID / Secret 是否正确？\n"
                "     Secret 在创建时仅显示一次，如果丢失需要重新创建\n"
                "\n"
                "验证方法：在控制台「智能语音交互 → 全部项目 → 点击项目名」\n"
                "  页面右侧有「Token 生成器」，如果能成功生成 Token，\n"
                "  则说明服务和权限没问题，再检查代码中的凭据是否复制正确。"
            )
        raise RuntimeError(
            f"获取阿里云语音 Token 失败 (HTTP {e.response.status_code}): {e.response.text[:200]}"
        )
    except Exception as e:
        raise RuntimeError(f"获取阿里云语音 Token 失败: {e}")


def _build_token_request() -> tuple[str, str]:
    """构建阿里云 POP API 的请求参数和 HMAC-SHA1 签名。

    CreateToken 是 RPC 风格接口，签名作为 Signature 参数拼接在 URL 中。

    返回:
        (query_string, signature)
        例如: ("AccessKeyId=xxx&Action=CreateToken&...", "base64signature...")

    签名流程参考: https://help.aliyun.com/document_detail/315526.html
    """
    params = {
        "AccessKeyId": ALIYUN_ACCESS_KEY_ID,
        "Action": "CreateToken",
        "Format": "JSON",
        "RegionId": "cn-shanghai",
        "SignatureMethod": "HMAC-SHA1",
        "SignatureNonce": str(uuid.uuid4()),
        "SignatureVersion": "1.0",
        "Timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "Version": "2019-02-28",
    }
    sorted_params = sorted(params.items(), key=lambda x: x[0])
    query_string = "&".join([
        f"{_pop_encode(k)}={_pop_encode(str(v))}" for k, v in sorted_params
    ])
    string_to_sign = f"GET&{_pop_encode('/')}&{_pop_encode(query_string)}"
    key = f"{ALIYUN_ACCESS_KEY_SECRET}&"
    signature = base64.b64encode(
        hmac.new(key.encode("utf-8"), string_to_sign.encode("utf-8"), hashlib.sha1).digest()
    ).decode("utf-8")
    return query_string, signature


def _pop_encode(s: str) -> str:
    """阿里云 POP 签名专用 URL 编码。"""
    encoded = quote(s, safe="")
    return encoded.replace("+", "%20").replace("*", "%2A").replace("%7E", "~")


# ==================== 一句话识别 (ASR) — WebSocket 协议 ====================

async def recognize_speech(audio_bytes: bytes, audio_format: str = "pcm") -> str:
    """使用阿里云一句话识别 WebSocket API 将音频转为文本。

    WebSocket 协议（官方推荐）：
      1. 连接到 wss://nls-gateway-cn-shanghai.aliyuncs.com/ws/v1?appkey=...&token=...
      2. 发送 StartTranscription 指令（JSON）
      3. 发送音频数据（binary frame）
      4. 发送 StopTranscription 指令（JSON）
      5. 接收 RecognitionCompleted 结果（JSON）

    参数:
        audio_bytes: 音频文件的二进制数据（16kHz, 16bit, 单声道）
        audio_format: 音频格式，"pcm" 或 "wav"

    返回:
        识别出的文字文本。识别失败返回空字符串。
    """
    import websockets

    _check_credentials()

    token = _get_aliyun_token()
    app_key = ALIYUN_ASR_APP_KEY

    ws_url = (
        f"{ASR_WS_URL}?appkey={app_key}&token={token}"
        f"&format={audio_format}&sample_rate=16000"
    )

    result_text = ""

    async def _attempt() -> str:
        nonlocal result_text
        result_text = ""
        try:
            async with websockets.connect(
                ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            ) as ws:
                task_id = uuid.uuid4().hex

                # ======== 第 1 步：发送 StartRecognition ========
                start_msg = json.dumps({
                    "header": {
                        "name": "StartRecognition",
                        "namespace": "SpeechRecognizer",
                        "message_id": uuid.uuid4().hex,
                        "task_id": task_id,
                        "appkey": app_key,
                        "status": 1,
                    },
                    "payload": {
                        "format": audio_format,
                        "sample_rate": 16000,
                        "enable_intermediate_result": False,
                        "enable_punctuation_prediction": True,
                    },
                })
                await ws.send(start_msg)

                # ======== 第 2 步：等待 RecognitionStarted 确认 ========
                started = False
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    header = msg.get("header", {})
                    name = header.get("name", "")
                    if name == "RecognitionStarted":
                        started = True
                        break
                    elif name == "TaskFailed":
                        status_text = header.get("status_text", "未知错误")
                        logger.warning("[ASR] 启动失败: %s", status_text)
                        return ""

                if not started:
                    return ""

                # ======== 第 3 步：发送音频数据（binary）========
                CHUNK = 8192
                for offset in range(0, len(audio_bytes), CHUNK):
                    await ws.send(audio_bytes[offset:offset + CHUNK])

                # ======== 第 4 步：发送 StopRecognition ========
                stop_msg = json.dumps({
                    "header": {
                        "name": "StopRecognition",
                        "namespace": "SpeechRecognizer",
                        "message_id": uuid.uuid4().hex,
                        "task_id": task_id,
                        "appkey": app_key,
                        "status": 1,
                    },
                })
                await ws.send(stop_msg)

                # ======== 第 5 步：等待 RecognitionCompleted 最终结果 ========
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    header = msg.get("header", {})
                    name = header.get("name", "")
                    status = header.get("status", -1)

                    if name == "RecognitionCompleted":
                        if status == 20000000:
                            result_text = msg.get("payload", {}).get("result", "")
                        else:
                            status_text = header.get("status_text", "未知错误")
                            logger.warning("[ASR] 识别失败 (status=%s): %s", status, status_text)
                        break
                    elif name == "RecognitionResultChanged":
                        pass
                    elif name == "TaskFailed":
                        status_text = header.get("status_text", "未知错误")
                        logger.warning("[ASR] 任务失败: %s", status_text)
                        break

            return result_text

        except websockets.exceptions.ConnectionClosed as e:
            logger.warning("[ASR] WebSocket 连接关闭: %s", e)
            return ""
        except (OSError, ConnectionError) as e:
            logger.warning("[ASR] 连接失败: %s", e)
            return ""
        except Exception as e:
            logger.warning("[ASR] WebSocket 异常: %s", e)
            return ""

    for attempt in range(1, MAX_RETRIES + 1):
        text = await _attempt()
        if text:
            return text.strip()
        if attempt < MAX_RETRIES:
            await asyncio.sleep(RETRY_DELAY)

    return ""


# ==================== 语音合成 (TTS) ====================

async def synthesize_speech(text: str) -> bytes:
    """使用 edge-tts 将文本合成为 MP3 音频。

    edge-tts 使用微软免费的 Azure TTS 引擎，提供自然流畅的中文语音。
    选用 zh-CN-XiaoxiaoNeural 音色（温暖活泼女声），语速稍慢适配视障用户。
    使用 asyncio.create_subprocess_exec 异步执行，避免阻塞事件循环。

    参数:
        text: 要合成的文本内容

    返回:
        MP3 格式的音频二进制数据。合成失败返回空字节。
    """
    import tempfile

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                output_file = tmp.name

            cmd = [
                "edge-tts",
                "--voice", TTS_VOICE,
                "--rate=" + TTS_RATE,
                "--text", text,
                "--write-media", output_file,
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                logger.warning("[TTS] 合成超时 (第%d次)", attempt)
                _safe_remove(output_file)
                if attempt >= MAX_RETRIES:
                    return b""
                continue
            if proc.returncode != 0:
                err = (stderr or b"").decode("utf-8", errors="replace").strip()
                logger.warning("[TTS] 合成失败 (第%d次): %s", attempt, err)
                _safe_remove(output_file)
                if attempt < MAX_RETRIES:
                    continue
                return b""

            if os.path.exists(output_file):
                with open(output_file, "rb") as f:
                    audio_data = f.read()
                _safe_remove(output_file)
                return audio_data
            else:
                logger.warning("[TTS] 输出文件未生成 (第%d次)", attempt)
                if attempt >= MAX_RETRIES:
                    return b""

        except FileNotFoundError:
            raise RuntimeError(
                "未找到 edge-tts 命令。请先安装: pip install edge-tts\n"
                "参考: https://github.com/rany2/edge-tts"
            )
        except Exception as e:
            logger.warning("[TTS] 异常: %s (第%d次)", e, attempt)
            if attempt >= MAX_RETRIES:
                return b""
    return b""


def _safe_remove(path: str):
    """安全删除临时文件，忽略文件不存在等异常。"""
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass
