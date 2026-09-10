"""
百炼 DashScope 文本嵌入客户端 —— 实现 LangChain Embeddings 接口。

为什么不用通用 OpenAI 兼容封装：
  DashScope 原生同步接口支持区分入库文档（text_type=document）与检索问题
  （text_type=query），并支持查询任务指令 instruct；这些能力只在原生接口
  提供，因此这里直接用现有 httpx 调用原生 HTTP API。

能力:
  - embed_documents / embed_query 及对应异步版本
  - 按 RAG_EMBEDDING_BATCH_SIZE 分批，解析 text_index 恢复输入顺序
  - 校验返回数量、向量维度与数值有效性
  - 401/403/模型不存在/参数错误：直接抛清晰的不可重试错误
  - 网络错误/429/5xx：有限退避重试
  - 磁盘缓存：按 模型/维度/处理版本/文本哈希 去重，减少重复计费
  - 与聊天客户端分开管理超时、并发与健康状态

参考:
  https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api
  https://help.aliyun.com/zh/model-studio/embedding
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import sqlite3
import threading
import time
from typing import Optional

import httpx
from langchain_core.embeddings import Embeddings
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
    RAG_CONTENT_VERSION,
    RAG_EMBEDDING_BATCH_SIZE,
    RAG_EMBEDDING_CACHE_DIR,
    RAG_EMBEDDING_DIMENSIONS,
    RAG_EMBEDDING_MODEL,
    RAG_EMBEDDING_TIMEOUT,
)

logger = logging.getLogger("embedding_client")

# 原生同步文本嵌入接口路径
_EMBED_PATH = "/services/embeddings/text-embedding/text-embedding"


# ==================== 异常分类 ====================

class EmbeddingError(RuntimeError):
    """嵌入调用基础异常。"""


class EmbeddingAuthError(EmbeddingError):
    """401/403 等鉴权失败，不可重试。"""


class EmbeddingConfigError(EmbeddingError):
    """模型不存在 / 参数错误等客户端问题，不可重试。"""


class EmbeddingTransientError(EmbeddingError):
    """网络错误 / 429 / 5xx，可有限重试。"""


# ==================== 磁盘缓存 ====================

class _VectorCache:
    """以 sqlite 持久化的向量缓存，key=模型|维度|版本|text_type|文本哈希。"""

    def __init__(self, cache_dir: str):
        os.makedirs(cache_dir, exist_ok=True)
        self._path = os.path.join(cache_dir, "embedding_cache.sqlite")
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "cache_key TEXT PRIMARY KEY, vector TEXT NOT NULL)"
        )
        self._conn.commit()

    @staticmethod
    def _key(model: str, dim: int, version: str, text_type: str, text: str) -> str:
        raw = f"{model}|{dim}|{version}|{text_type}|{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get_many(self, model: str, dim: int, version: str,
                 text_type: str, texts: list[str]) -> dict[int, list[float]]:
        """返回 {texts 中的下标: 向量}，未命中的不在字典中。"""
        out: dict[int, list[float]] = {}
        with self._lock:
            for idx, text in enumerate(texts):
                key = self._key(model, dim, version, text_type, text)
                row = self._conn.execute(
                    "SELECT vector FROM cache WHERE cache_key=?", (key,)
                ).fetchone()
                if row is not None:
                    out[idx] = json.loads(row[0])
        return out

    def put_many(self, model: str, dim: int, version: str,
                 text_type: str, texts: list[str], vectors: list[list[float]]) -> None:
        with self._lock:
            for idx, text in enumerate(texts):
                key = self._key(model, dim, version, text_type, text)
                self._conn.execute(
                    "INSERT OR REPLACE INTO cache(cache_key, vector) VALUES(?,?)",
                    (key, json.dumps(vectors[idx], separators=(",", ":"))),
                )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ==================== 嵌入客户端 ====================

class DashScopeTextEmbeddings(Embeddings):
    """DashScope 原生文本嵌入的 LangChain 适配。"""

    def __init__(
        self,
        model: str = RAG_EMBEDDING_MODEL,
        dimensions: int = RAG_EMBEDDING_DIMENSIONS,
        batch_size: int = RAG_EMBEDDING_BATCH_SIZE,
        timeout: float = RAG_EMBEDDING_TIMEOUT,
        api_key: str = DASHSCOPE_API_KEY,
        base_url: str = DASHSCOPE_BASE_URL,
        content_version: str = RAG_CONTENT_VERSION,
        query_instruct: str = "",
        enable_cache: bool = True,
    ) -> None:
        if not api_key:
            raise EmbeddingAuthError(
                "DASHSCOPE_API_KEY 未配置，无法调用百炼文本嵌入。"
            )
        if dimensions not in (1024, 768, 512, 256):
            raise EmbeddingConfigError(
                f"Flash 嵌入仅支持 1024/768/512/256 维，当前为 {dimensions}"
            )
        if not 1 <= batch_size <= 20:
            raise EmbeddingConfigError(
                f"单批文本数量必须在 1-20 之间，当前为 {batch_size}"
            )
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.timeout = timeout
        self._api_key = api_key
        self._url = base_url.rstrip("/") + _EMBED_PATH
        self.content_version = content_version
        self.query_instruct = query_instruct
        self._sync_client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None
        self._async_semaphore: Optional[asyncio.Semaphore] = None
        self._cache: Optional[_VectorCache] = (
            _VectorCache(RAG_EMBEDDING_CACHE_DIR) if enable_cache else None
        )
        # 健康状态缓存（与聊天客户端独立）
        self._health = {"checked": False, "ok": False, "ts": 0.0}

    # ---------- httpx 客户端（与 LLM 客户端分离） ----------

    def _get_sync_client(self) -> httpx.Client:
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                timeout=httpx.Timeout(self.timeout, connect=10.0)
            )
        return self._sync_client

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=10.0)
            )
        return self._async_client

    def close(self) -> None:
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None
        if self._async_client is not None:
            # 同步兜底关闭；异步场景优先使用 aclose
            try:
                self._async_client.close()
            except Exception:
                logger.debug("[embedding] async client close 异常", exc_info=True)
            self._async_client = None
        if self._cache is not None:
            self._cache.close()
            self._cache = None

    async def aclose(self) -> None:
        """在 FastAPI 生命周期中异步关闭，避免事件循环告警。"""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None
        if self._cache is not None:
            self._cache.close()
            self._cache = None

    # ---------- 核心 HTTP 调用 ----------

    def _build_payload(self, texts: list[str], text_type: str,
                       instruct: str = "") -> dict:
        parameters: dict = {"dimension": self.dimensions, "text_type": text_type}
        if text_type == "query" and instruct:
            parameters["instruct"] = instruct
        return {
            "model": self.model,
            "input": {"texts": texts},
            "parameters": parameters,
        }

    def _classify_http_error(self, resp: httpx.Response) -> None:
        """把非 2xx 响应映射为明确的异常类型。"""
        code = resp.status_code
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        code_str = str(code)
        if code in (401, 403):
            raise EmbeddingAuthError(
                f"百炼嵌入鉴权失败(HTTP {code})，请检查 DASHSCOPE_API_KEY：{detail}"
            )
        if code == 429 or 500 <= code < 600:
            raise EmbeddingTransientError(f"百炼嵌入暂时不可用(HTTP {code})：{detail}")
        # 400/404 等通常是模型名错误或参数错误
        raise EmbeddingConfigError(
            f"百炼嵌入参数/模型错误(HTTP {code_str})，请核对模型名、维度与接口地址：{detail}"
        )

    def _parse_embeddings(self, data: dict, expect: int) -> list[list[float]]:
        """解析响应，按返回下标恢复顺序，并做数量/维度/数值校验。

        百炼原生接口通常返回 ``text_index``，部分兼容网关会在相同
        ``output.embeddings`` 结构中返回 OpenAI 风格的 ``index``。
        两者语义相同，因此同时接受。
        """
        try:
            items = data["output"]["embeddings"]
        except (KeyError, TypeError) as e:
            raise EmbeddingConfigError(f"百炼嵌入响应结构异常，缺少 output.embeddings：{data}") from e
        if not isinstance(items, list) or len(items) != expect:
            raise EmbeddingConfigError(
                f"百炼嵌入返回数量不符：期望 {expect}，实际 {0 if not isinstance(items, list) else len(items)}"
            )
        ordered: list[Optional[list[float]]] = [None] * expect
        for it in items:
            idx = it.get("text_index")
            if idx is None:
                idx = it.get("index")
            vec = it.get("embedding")
            if not isinstance(idx, int) or not (0 <= idx < expect):
                raise EmbeddingConfigError(f"百炼嵌入返回非法索引：{idx}")
            if not isinstance(vec, list) or len(vec) != self.dimensions:
                raise EmbeddingConfigError(
                    f"百炼嵌入维度不符：期望 {self.dimensions}，实际 "
                    f"{len(vec) if isinstance(vec, list) else '非向量'}"
                )
            for v in vec:
                if not isinstance(v, (int, float)) or not math.isfinite(v):
                    raise EmbeddingConfigError("百炼嵌入向量包含非有限数值")
            ordered[idx] = vec
        if any(v is None for v in ordered):
            raise EmbeddingConfigError("百炼嵌入返回存在缺失索引，无法还原顺序")
        return ordered  # type: ignore[return-value]

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(EmbeddingTransientError),
        reraise=True,
    )
    def _post_batch_sync(self, texts: list[str], text_type: str,
                         instruct: str = "") -> list[list[float]]:
        payload = self._build_payload(texts, text_type, instruct)
        try:
            resp = self._get_sync_client().post(
                self._url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
            raise EmbeddingTransientError(f"百炼嵌入网络错误：{type(e).__name__}: {e}") from e
        if resp.status_code != 200:
            self._classify_http_error(resp)  # 内部按类型 raise
        return self._parse_embeddings(resp.json(), len(texts))

    async def _post_batch_async(self, texts: list[str], text_type: str,
                                instruct: str = "") -> list[list[float]]:
        if self._async_semaphore is None:
            # 限制嵌入并发，避免大批量时打爆单连接/触发限流
            self._async_semaphore = asyncio.Semaphore(4)

        @retry(
            stop=stop_after_attempt(MAX_RETRIES),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(EmbeddingTransientError),
            reraise=True,
        )
        async def _do() -> list[list[float]]:
            payload = self._build_payload(texts, text_type, instruct)
            async with self._async_semaphore:  # type: ignore[union-attr]
                try:
                    resp = await self._get_async_client().post(
                        self._url,
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        },
                    )
                except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
                    raise EmbeddingTransientError(
                        f"百炼嵌入网络错误：{type(e).__name__}: {e}"
                    ) from e
            if resp.status_code != 200:
                self._classify_http_error(resp)
            return self._parse_embeddings(resp.json(), len(texts))

        return await _do()

    # ---------- 分批 + 缓存 ----------

    def _batches(self, texts: list[str]) -> list[tuple[int, int]]:
        return [
            (i, min(i + self.batch_size, len(texts)))
            for i in range(0, len(texts), self.batch_size)
        ]

    def _embed(self, texts: list[str], text_type: str,
               instruct: str = "") -> list[list[float]]:
        if not texts:
            return []
        results: list[Optional[list[float]]] = [None] * len(texts)

        # 1. 读缓存
        cached = self._cache.get_many(
            self.model, self.dimensions, self.content_version, text_type, texts
        ) if self._cache else {}
        for idx, vec in cached.items():
            results[idx] = vec

        # 2. 对未命中项按连续区间分批请求
        for start, end in self._batches(texts):
            miss_idx = [i for i in range(start, end) if results[i] is None]
            if not miss_idx:
                continue
            miss_texts = [texts[i] for i in miss_idx]
            vectors = self._post_batch_sync(miss_texts, text_type, instruct)
            for local_i, global_i in enumerate(miss_idx):
                results[global_i] = vectors[local_i]
            if self._cache:
                self._cache.put_many(
                    self.model, self.dimensions, self.content_version,
                    text_type, miss_texts, vectors,
                )
        return results  # type: ignore[return-value]

    async def _aembed(self, texts: list[str], text_type: str,
                      instruct: str = "") -> list[list[float]]:
        if not texts:
            return []
        results: list[Optional[list[float]]] = [None] * len(texts)
        cached = self._cache.get_many(
            self.model, self.dimensions, self.content_version, text_type, texts
        ) if self._cache else {}
        for idx, vec in cached.items():
            results[idx] = vec

        tasks = []
        task_index_map: list[list[int]] = []
        for start, end in self._batches(texts):
            miss_idx = [i for i in range(start, end) if results[i] is None]
            if not miss_idx:
                continue
            miss_texts = [texts[i] for i in miss_idx]
            tasks.append(self._post_batch_async(miss_texts, text_type, instruct))
            task_index_map.append(miss_idx)
        if tasks:
            batch_results = await asyncio.gather(*tasks)
            for miss_idx, vectors in zip(task_index_map, batch_results):
                for local_i, global_i in enumerate(miss_idx):
                    results[global_i] = vectors[local_i]
                if self._cache:
                    self._cache.put_many(
                        self.model, self.dimensions, self.content_version,
                        text_type, [texts[i] for i in miss_idx], vectors,
                    )
        return results  # type: ignore[return-value]

    # ---------- LangChain Embeddings 接口 ----------

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化入库文档，text_type=document。"""
        t0 = time.time()
        vecs = self._embed(list(texts), "document")
        self._mark_ok()
        logger.info(
            "[embedding] embed_documents %d 条，耗时 %.2fs", len(texts), time.time() - t0
        )
        return vecs

    def embed_query(self, text: str) -> list[float]:
        """向量化检索问题，text_type=query（可带 instruct）。"""
        vecs = self._embed([text], "query", self.query_instruct)
        self._mark_ok()
        return vecs[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        t0 = time.time()
        vecs = await self._aembed(list(texts), "document")
        self._mark_ok()
        logger.info(
            "[embedding] aembed_documents %d 条，耗时 %.2fs", len(texts), time.time() - t0
        )
        return vecs

    async def aembed_query(self, text: str) -> list[float]:
        vecs = await self._aembed([text], "query", self.query_instruct)
        self._mark_ok()
        return vecs[0]

    # ---------- 健康状态 ----------

    def _mark_ok(self) -> None:
        self._health.update(checked=True, ok=True, ts=time.time())

    def health(self, ttl: float = 60.0) -> dict:
        """返回最近一次嵌入调用的缓存健康状态。"""
        fresh = (time.time() - self._health["ts"]) <= ttl
        return {
            "checked": self._health["checked"],
            "ok": bool(self._health["checked"] and self._health["ok"] and fresh),
            "model": self.model,
            "dimensions": self.dimensions,
        }


# ==================== 单例 ====================

_embeddings: Optional[DashScopeTextEmbeddings] = None
_singleton_lock = threading.Lock()


def get_embeddings() -> DashScopeTextEmbeddings:
    """获取进程内共享的嵌入客户端单例。"""
    global _embeddings
    if _embeddings is None:
        with _singleton_lock:
            if _embeddings is None:
                _embeddings = DashScopeTextEmbeddings()
                logger.info(
                    "[embedding] 已创建 DashScope 嵌入单例 model=%s dim=%d",
                    _embeddings.model, _embeddings.dimensions,
                )
    return _embeddings


def close_embeddings() -> None:
    """同步关闭嵌入单例（服务关闭兜底）。"""
    global _embeddings
    if _embeddings is not None:
        _embeddings.close()
        _embeddings = None


async def aclose_embeddings() -> None:
    """异步关闭嵌入单例（FastAPI 生命周期使用）。"""
    global _embeddings
    if _embeddings is not None:
        await _embeddings.aclose()
        _embeddings = None
