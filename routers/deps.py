"""路由公共依赖 —— 限流器与服务密钥鉴权。

这些对象同时被 main.py（中间件）和各个路由模块引用，
集中放在这里避免主入口与路由模块互相 import 造成循环依赖。
"""

import time
from collections import deque

from fastapi import Header, HTTPException

from config import AGENT_RATE_LIMIT_PER_MINUTE, AGENT_SERVICE_API_KEY


class RateLimiter:
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


rate_limiter = RateLimiter(AGENT_RATE_LIMIT_PER_MINUTE)


def require_service_key(x_agent_key: str = Header(default="")) -> None:
    """服务密钥鉴权：配置了 AGENT_SERVICE_API_KEY 时强制校验 X-Agent-Key。"""
    if AGENT_SERVICE_API_KEY and x_agent_key != AGENT_SERVICE_API_KEY:
        raise HTTPException(status_code=401, detail="无效的服务密钥")
