"""
百炼 DashScope Rerank 精排客户端 —— Cross-Encoder 相关性重排。

定位（RAG 链路中的角色）:
  召回阶段用双塔 Bi-Encoder（embedding）快速从全库召回候选，query 与 document
  独立编码，速度快但交互不充分；精排阶段用 Cross-Encoder（rerank）把 query 与
  每个候选 document 拼接后联合编码，二者充分注意力交互，相关性判断更精准。
  只对召回的少量候选（默认 20 条）做精排，计算量可控。

能力:
  - rerank(query, documents, top_n) 返回按相关性降序的 [(原始下标, 分数), ...]
  - 401/403/模型不存在/参数错误：抛不可重试的明确异常
  - 网络错误/429/5xx：有限退避重试
  - 进程内共享单例，与 embedding/聊天客户端独立管理超时
  - 不做磁盘缓存：rerank 分数依赖 query-document 对，查询间复用率极低

参考:
  https://help.aliyun.com/zh/model-studio/rerank-api
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    MAX_RETRIES,
    RAG_RERANK_BATCH_SIZE,
    RAG_RERANK_MODEL,
    RAG_RERANK_TIMEOUT,
)

logger = logging.getLogger("rerank_client")

# 原生同步文本重排接口路径
_RERANK_PATH = "/services/rerank/text-rerank/text-rerank"


# ==================== 异常分类 ====================

class RerankError(RuntimeError):
    """Rerank 调用基础异常。"""


class RerankAuthError(RerankError):
    """401/403 等鉴权失败，不可重试。"""


class RerankConfigError(RerankError):
    """模型不存在 / 参数错误等客户端问题，不可重试。"""


class RerankTransientError(RerankError):
    """网络错误 / 429 / 5xx，可有限重试。"""


# ==================== Rerank 客户端 ====================

class DashScopeReranker:
    """DashScope 原生文本重排客户端。"""

    def __init__(
        self,
        model: str = RAG_RERANK_MODEL,
        timeout: float = RAG_RERANK_TIMEOUT,
        batch_size: int = RAG_RERANK_BATCH_SIZE,
        api_key: str = DASHSCOPE_API_KEY,
        base_url: str = DASHSCOPE_BASE_URL,
    ) -> None:
        if not api_key:
            raise RerankAuthError(
                "DASHSCOPE_API_KEY 未配置，无法调用百炼 Rerank。"
            )
        self.model = model
        self.timeout = timeout
        self.batch_size = batch_size
        self._api_key = api_key
        self._url = base_url.rstrip("/") + _RERANK_PATH
        self._client: Optional[httpx.Client] = None
        self._health = {"checked": False, "ok": False, "ts": 0.0}

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=httpx.Timeout(self.timeout, connect=10.0)
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # ---------- HTTP 错误分类 ----------

    def _classify_http_error(self, resp: httpx.Response) -> None:
        code = resp.status_code
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        if code in (401, 403):
            raise RerankAuthError(
                f"百炼 Rerank 鉴权失败(HTTP {code})，请检查 DASHSCOPE_API_KEY：{detail}"
            )
        if code == 429 or 500 <= code < 600:
            raise RerankTransientError(f"百炼 Rerank 暂时不可用(HTTP {code})：{detail}")
        raise RerankConfigError(
            f"百炼 Rerank 参数/模型错误(HTTP {code})，请核对模型名与接口地址：{detail}"
        )

    # ---------- 核心 HTTP 调用（单批） ----------

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(RerankTransientError),
        reraise=True,
    )
    def _post_batch(
        self, query: str, documents: list[str], top_n: int
    ) -> list[tuple[int, float]]:
        """对单批文档（不超过 batch_size）调用 rerank，返回 [(原始下标, 分数), ...]。"""
        payload = {
            "model": self.model,
            "input": {"query": query, "documents": documents},
            "parameters": {"top_n": top_n, "return_documents": False},
        }
        try:
            resp = self._get_client().post(
                self._url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
            raise RerankTransientError(f"百炼 Rerank 网络错误：{type(e).__name__}: {e}") from e

        if resp.status_code != 200:
            self._classify_http_error(resp)

        data = resp.json()
        try:
            results = data["output"]["results"]
        except (KeyError, TypeError) as e:
            raise RerankConfigError(
                f"百炼 Rerank 响应结构异常，缺少 output.results：{data}"
            ) from e

        parsed: list[tuple[int, float]] = []
        for item in results:
            idx = item.get("index")
            score = item.get("relevance_score")
            if not isinstance(idx, int) or not (0 <= idx < len(documents)):
                raise RerankConfigError(f"百炼 Rerank 返回非法下标：{idx}")
            if not isinstance(score, (int, float)):
                raise RerankConfigError(f"百炼 Rerank 返回非法分数：{score}")
            parsed.append((idx, float(score)))
        return parsed

    # ---------- 对外主方法 ----------

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: Optional[int] = None,
    ) -> list[tuple[int, float]]:
        """对 documents 按与 query 的相关性精排。

        参数:
            query: 检索问题原文
            documents: 候选文档文本列表（下标即文档在候选列表中的位置）
            top_n: 只返回前 N 条；None 表示返回全部（仍按相关性降序）

        返回:
            [(文档在 documents 中的原始下标, rerank 相关性分数), ...]，按分数降序。
        """
        if not documents:
            return []
        if top_n is None:
            top_n = len(documents)
        top_n = min(top_n, len(documents))

        t0 = time.perf_counter()
        # 文档数超过单批上限时分批：每批各自精排后，再按分数全局归并取 top_n
        if len(documents) <= self.batch_size:
            ranked = self._post_batch(query, documents, top_n)
        else:
            partials: list[tuple[int, float]] = []
            for start in range(0, len(documents), self.batch_size):
                batch = documents[start:start + self.batch_size]
                # 每批先取 min(top_n, 批大小)，归并后再截全局 top_n
                batch_ranked = self._post_batch(
                    query, batch, min(top_n, len(batch))
                )
                for local_idx, score in batch_ranked:
                    partials.append((start + local_idx, score))
            ranked = sorted(partials, key=lambda x: x[1], reverse=True)[:top_n]

        self._mark_ok()
        logger.info(
            "[rerank] model=%s candidates=%d returned=%d elapsed_ms=%.0f",
            self.model, len(documents), len(ranked),
            (time.perf_counter() - t0) * 1000,
        )
        return ranked

    # ---------- 健康状态 ----------

    def _mark_ok(self) -> None:
        self._health.update(checked=True, ok=True, ts=time.time())

    def health(self, ttl: float = 60.0) -> dict:
        fresh = (time.time() - self._health["ts"]) <= ttl
        return {
            "checked": self._health["checked"],
            "ok": bool(self._health["checked"] and self._health["ok"] and fresh),
            "model": self.model,
        }


# ==================== 单例 ====================

_reranker: Optional[DashScopeReranker] = None
_singleton_lock = threading.Lock()


def get_reranker() -> DashScopeReranker:
    """获取进程内共享的 Rerank 客户端单例。"""
    global _reranker
    if _reranker is None:
        with _singleton_lock:
            if _reranker is None:
                _reranker = DashScopeReranker()
                logger.info(
                    "[rerank] 已创建 DashScope Rerank 单例 model=%s",
                    _reranker.model,
                )
    return _reranker


def close_reranker() -> None:
    """同步关闭 Rerank 单例（服务关闭兜底）。"""
    global _reranker
    if _reranker is not None:
        _reranker.close()
        _reranker = None
