"""
统一后端 HTTP 客户端 —— tools / health_tools 共用。

职责:
  - 会话级 JWT 存储（session_id -> {token, user_id, username, expires_at}）
  - 统一请求头、超时、错误提取
  - 重试策略: 仅 GET 查询可安全重试（超时/连接失败/5xx）；
    下单、加购、评分等 POST/PUT/DELETE 变更操作**绝不自动重试**，
    避免"服务端已提交但响应丢失"时重复下单/重复扣减。
  - validate_user_token: 通过 Java 后端真实校验 JWT（验签 + 有效性），
    不信任客户端直接传来的 payload。
"""

import logging
import time
from typing import Optional

import httpx

from config import JAVA_BASE_URL, MAX_RETRIES, RETRY_DELAY, SSL_VERIFY

logger = logging.getLogger("backend_client")


# ==================== 会话级 JWT 存储 ====================

_session_store: dict[str, dict] = {}


def set_session(session_id: str, data: dict) -> None:
    """将 JWT 登录信息存入会话。"""
    _session_store[session_id] = data


def get_session(session_id: str) -> Optional[dict]:
    """获取会话信息，若过期则自动清除并返回 None。"""
    sess = _session_store.get(session_id)
    if not sess:
        return None
    if time.time() > sess.get("expires_at", 0):
        _session_store.pop(session_id, None)
        return None
    return sess


def purge_expired_sessions(now: Optional[float] = None) -> int:
    """清理所有已过期的会话 token 记录，返回清理条数。"""
    now = time.time() if now is None else now
    expired = [
        sid for sid, data in _session_store.items()
        if now > data.get("expires_at", 0)
    ]
    for sid in expired:
        _session_store.pop(sid, None)
    if expired:
        logger.info("[session_purge] 清理过期 token %d 条", len(expired))
    return len(expired)


def get_auth_headers(session_id: str) -> dict[str, str]:
    """构造带 Bearer token 的 Authorization 请求头。"""
    sess = get_session(session_id)
    if not sess:
        raise PermissionError("用户未登录或登录已过期，请先登录。")
    return {"Authorization": f"Bearer {sess['token']}", "Content-Type": "application/json"}


# ==================== 共享 HTTP 客户端 ====================

_client: Optional[httpx.Client] = None


def _get_client() -> httpx.Client:
    """懒加载共享 httpx 客户端（线程安全，进程内单例）。"""
    global _client
    if _client is None:
        _client = httpx.Client(
            verify=SSL_VERIFY,
            timeout=httpx.Timeout(15.0, connect=10.0),
        )
    return _client


def close_client() -> None:
    """关闭共享客户端（服务关闭时调用）。"""
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
        _client = None


def _url(path: str) -> str:
    return f"{JAVA_BASE_URL}{path}"


def _is_retryable(exception: BaseException) -> bool:
    """判断异常是否可重试（超时/连接失败/5xx 可重试，4xx 不可重试）。"""
    if isinstance(exception, httpx.TimeoutException):
        return True
    if isinstance(exception, httpx.ConnectError):
        return True
    if isinstance(exception, httpx.HTTPStatusError):
        return 500 <= exception.response.status_code < 600
    return False


def _retry_get(url: str, headers: dict, params: Optional[dict]) -> dict:
    """GET 请求：安全重试，最多 MAX_RETRIES 次。"""
    last_error: BaseException | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _get_client().get(url, headers=headers, params=params)
            resp.raise_for_status()
            return resp.json()
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as e:
            if not _is_retryable(e) or attempt >= MAX_RETRIES:
                raise
            last_error = e
            logger.warning(
                "[http_retry] GET %s 第 %d/%d 次失败: %s",
                url, attempt, MAX_RETRIES, type(e).__name__,
            )
            time.sleep(RETRY_DELAY)
    raise last_error  # 理论上不可达


def _no_retry(method: str, url: str, headers: dict, body: Optional[dict] = None) -> dict:
    """变更类请求：单次执行，不重试（防重复下单/重复扣减）。"""
    client = _get_client()
    resp = client.request(method, url, headers=headers, json=body)
    resp.raise_for_status()
    return resp.json()


# ==================== 对外请求函数 ====================

def get(
    path: str,
    session_id: str,
    params: Optional[dict] = None,
    extra_headers: Optional[dict[str, str]] = None,
) -> dict:
    """携带用户 JWT 的 GET 请求（可重试）。"""
    headers = get_auth_headers(session_id)
    headers.pop("Content-Type", None)
    if extra_headers:
        headers.update(extra_headers)
    return _retry_get(_url(path), headers, params)


def post(
    path: str,
    session_id: str,
    body: Optional[dict] = None,
    extra_headers: Optional[dict[str, str]] = None,
) -> dict:
    """携带用户 JWT 的 POST 请求（不重试，防止重复提交）。"""
    headers = get_auth_headers(session_id)
    if extra_headers:
        headers.update(extra_headers)
    return _no_retry("POST", _url(path), headers, body)


def put(
    path: str,
    session_id: str,
    body: Optional[dict] = None,
    extra_headers: Optional[dict[str, str]] = None,
) -> dict:
    """携带用户 JWT 的 PUT 请求（不重试）。"""
    headers = get_auth_headers(session_id)
    if extra_headers:
        headers.update(extra_headers)
    return _no_retry("PUT", _url(path), headers, body)


def delete(path: str, session_id: str, extra_headers: Optional[dict[str, str]] = None) -> dict:
    """携带用户 JWT 的 DELETE 请求（不重试）。"""
    headers = get_auth_headers(session_id)
    headers.pop("Content-Type", None)
    if extra_headers:
        headers.update(extra_headers)
    return _no_retry("DELETE", _url(path), headers)


def post_public(path: str, body: Optional[dict] = None, timeout: float = 10.0) -> dict:
    """无需认证的 POST（如登录、店铺状态），单次执行。"""
    client = _get_client()
    resp = client.post(
        _url(path),
        json=body,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def get_public(path: str, params: Optional[dict] = None, timeout: float = 5.0) -> dict:
    """无需认证的 GET（如店铺状态探活），单次执行。"""
    client = _get_client()
    resp = client.get(_url(path), params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ==================== 统一响应解析 ====================

def extract_data(result: dict) -> object:
    """从 Java 统一响应 {code, message, data} 中提取 data 字段。"""
    code = result.get("code", -1)
    if code != 200:
        msg = result.get("message", "未知错误")
        raise RuntimeError(f"Java 后端返回错误 (code={code}): {msg}")
    return result.get("data")


def extract_records(data: object) -> list:
    """兼容 Java 后端直接返回列表或分页对象两种结构。"""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        records = data.get("records")
        return records if isinstance(records, list) else []
    return []


# ==================== JWT 后端验证 ====================

def validate_user_token(token: str) -> bool:
    """通过 Java 后端真实校验 JWT（验签 + 黑名单 + 过期检查）。

    使用一个轻量需鉴权接口（收货地址列表）探测 token 有效性：
    200 + code=200 视为有效；401/403 视为无效；其他异常记录日志并视为无效。
    """
    url = _url("/api/user/addresses")
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = _get_client().get(url, headers=headers, timeout=10.0)
        if resp.status_code in (401, 403):
            logger.warning("[token_validate] 后端拒绝该 token (HTTP %d)", resp.status_code)
            return False
        resp.raise_for_status()
        result = resp.json()
        return result.get("code") == 200
    except (httpx.TimeoutException, httpx.ConnectError) as e:
        logger.warning("[token_validate] 后端不可达，无法验证 token: %s", type(e).__name__)
        return False
    except Exception as e:
        logger.warning("[token_validate] token 验证异常: %s", type(e).__name__)
        return False
