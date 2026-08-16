"""
LLM 客户端单例管理。

所有调用方（主 Agent、饮食计划生成器）共享同一套 httpx 客户端，
避免每个会话/每次调用都新建连接池导致资源泄漏。
同时维护 LLM 健康状态缓存，健康检查不再每次真调大模型。
"""

import logging
import time

import httpx
from langchain_openai import ChatOpenAI

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    LLM_MODEL,
    LLM_REQUEST_TIMEOUT,
    LLM_TEMPERATURE,
    MAX_LLM_TOKENS,
    SSL_VERIFY,
)

logger = logging.getLogger("llm_client")

_llm = None
_http_client = None
_async_http_client = None

# LLM 健康状态：由真实对话调用更新，健康检查只读缓存
_probe = {"checked": False, "ok": False, "last_ok_ts": 0.0}
_last_probe_ts = 0.0


def get_http_clients() -> tuple[httpx.Client, httpx.AsyncClient]:
    """获取（并懒创建）共享的同步/异步 httpx 客户端。"""
    global _http_client, _async_http_client
    if _http_client is None:
        timeout = httpx.Timeout(LLM_REQUEST_TIMEOUT, connect=10.0)
        _http_client = httpx.Client(verify=SSL_VERIFY, timeout=timeout)
        _async_http_client = httpx.AsyncClient(verify=SSL_VERIFY, timeout=timeout)
    return _http_client, _async_http_client


def get_llm() -> ChatOpenAI:
    """获取共享 LLM 单例（复用同一套 httpx 客户端）。"""
    global _llm
    if _llm is None:
        http_client, async_http_client = get_http_clients()
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=MAX_LLM_TOKENS,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            streaming=False,
            request_timeout=LLM_REQUEST_TIMEOUT,
            http_client=http_client,
            http_async_client=async_http_client,
        )
        logger.info("[llm] 已创建共享 LLM 实例 model=%s", LLM_MODEL)
    return _llm


def close_llm() -> None:
    """关闭共享 LLM 及 httpx 客户端（服务关闭时调用）。"""
    global _llm, _http_client, _async_http_client
    for client in (_http_client, _async_http_client):
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
    _llm = None
    _http_client = None
    _async_http_client = None
    logger.info("[llm] 已关闭共享 LLM 客户端")


def mark_llm_success() -> None:
    """记录一次成功的 LLM 调用。"""
    _probe["checked"] = True
    _probe["ok"] = True
    _probe["last_ok_ts"] = time.time()


def mark_llm_failure() -> None:
    """记录一次失败的 LLM 调用。"""
    _probe["checked"] = True
    _probe["ok"] = False


def llm_healthy(ttl: float) -> bool:
    """返回缓存内的 LLM 健康状态（ttl 秒内最近一次成功视为健康）。"""
    if not _probe["checked"]:
        return False
    if _probe["ok"]:
        return (time.time() - _probe["last_ok_ts"]) <= ttl
    return False


async def probe_llm(ttl: float = 300.0) -> bool:
    """主动探测一次 LLM（带节流：ttl 秒内最多真调一次）。

    仅健康检查在从未探测或缓存状态不可用时调用。
    """
    global _last_probe_ts
    now = time.time()
    if _probe["checked"] and now - _last_probe_ts < ttl:
        return _probe["ok"] and (now - _probe["last_ok_ts"] <= ttl)
    _last_probe_ts = now
    try:
        llm = get_llm()
        resp = await llm.ainvoke("回复'OK'")
        ok = isinstance(resp.content, str) and "OK" in resp.content
    except Exception as e:
        logger.warning("[llm_probe] 探测失败: %s", type(e).__name__)
        ok = False
    if ok:
        mark_llm_success()
    else:
        mark_llm_failure()
    return ok
