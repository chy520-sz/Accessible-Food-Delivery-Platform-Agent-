"""
知识库检索模块 —— 基于 Milvus + 百炼 Qwen 文本嵌入（COSINE 稠密检索）。

职责:
  - 进程内共享官方 MilvusClient（懒加载）
  - 查询进程只连接“已构建”的集合，绝不在每次启动/查询时自动建库
  - COSINE 度量：相似度越大越相关（旧 Chroma 的“分数越小越相关”阈值已废弃）
  - 按“逻辑知识类型”(dish/dietary/faq) 决定格式化，不再硬编码物理集合名
  - top_k 起步、最低相关阈值过滤；没有可靠命中时允许返回“未找到”，不硬拼答案
  - 基础设施错误（Milvus 停机/维度不匹配）抛类型化异常，不伪装成“没有知识”

使用:
  from knowledge_base import search_knowledge
  search_knowledge("川菜有什么特点？", RAG_COLLECTION_FOOD, logical_type="dish")
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Optional

from langchain_core.documents import Document
from pymilvus import MilvusClient
from pymilvus.exceptions import MilvusException
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import (
    MAX_RETRIES,
    MILVUS_DB_NAME,
    MILVUS_TOKEN,
    MILVUS_URI,
    RAG_KNOWLEDGE_COLLECTIONS,
    RAG_METRIC_TYPE,
    RAG_MIN_SCORE,
    RAG_TOP_K,
)
from embedding_client import get_embeddings

logger = logging.getLogger("knowledge_base")


# ==================== 异常类型 ====================

class KnowledgeStoreError(RuntimeError):
    """知识库基础设施错误（连接失败、维度不匹配等），不应被当成“没找到知识”。"""


class KnowledgeNotBuilt(KnowledgeStoreError):
    """目标集合尚不存在或为空（尚未构建向量库）。"""


# ==================== 逻辑知识类型元数据 ====================
# 物理集合名 -> 逻辑类型
_COLLECTION_TO_LOGICAL = {phys: logical for logical, phys in RAG_KNOWLEDGE_COLLECTIONS.items()}

_LOGICAL_META = {
    "dish": {
        "label": "菜品知识",
        "empty_name": "菜品知识库",
        "hint_keys": ("name", "category"),
    },
    "dietary": {
        "label": "饮食知识",
        "empty_name": "饮食健康知识库",
        "hint_keys": ("condition",),
    },
    "faq": {
        "label": "常见问题",
        "empty_name": "常见问题知识库",
        "hint_keys": ("category",),
    },
}


def logical_type_of(collection: str, metadata: Optional[dict] = None) -> str:
    """根据物理集合名（优先用元数据里的 knowledge_type）推断逻辑知识类型。"""
    if metadata and metadata.get("knowledge_type") in _LOGICAL_META:
        return metadata["knowledge_type"]
    return _COLLECTION_TO_LOGICAL.get(collection, "unknown")


# ==================== Milvus 连接管理 ====================

_client_lock = threading.Lock()
_milvus_client: Optional[MilvusClient] = None
_store_cache: set[str] = set()


def connection_args() -> dict:
    """pymilvus 连接参数。"""
    args = {"uri": MILVUS_URI, "db_name": MILVUS_DB_NAME}
    if MILVUS_TOKEN:
        args["token"] = MILVUS_TOKEN
    return args


def get_milvus_client() -> MilvusClient:
    """获取进程内共享的 MilvusClient（连接失败抛 KnowledgeStoreError）。"""
    global _milvus_client
    if _milvus_client is not None:
        return _milvus_client
    with _client_lock:
        if _milvus_client is None:
            try:
                _milvus_client = MilvusClient(**connection_args())
            except Exception as e:  # 连接参数错误/服务不可达
                raise KnowledgeStoreError(f"无法连接 Milvus（{MILVUS_URI}）：{type(e).__name__}: {e}") from e
    return _milvus_client


def ping_milvus() -> bool:
    """轻量探活：是否能列出集合。"""
    try:
        get_milvus_client().list_collections()
        return True
    except KnowledgeStoreError:
        return False
    except MilvusException:
        return False


def collection_exists(collection_name: str) -> bool:
    """集合是否存在（查询路径用它判断，避免 langchain 自动建集合）。"""
    try:
        return bool(get_milvus_client().has_collection(collection_name))
    except KnowledgeStoreError:
        raise
    except Exception as e:
        raise KnowledgeStoreError(f"检查集合 {collection_name} 是否存在失败：{e}") from e


def collection_count(collection_name: str) -> int:
    """返回集合实体数（集合不存在返回 0）。"""
    try:
        client = get_milvus_client()
        if not client.has_collection(collection_name):
            return 0
        stats = client.get_collection_stats(collection_name)
        return int(stats.get("row_count", 0))
    except Exception as e:
        raise KnowledgeStoreError(f"读取集合 {collection_name} 计数失败：{e}") from e


def reset_store_cache() -> None:
    """构建/删除集合后清空向量存储缓存。"""
    _store_cache.clear()


def close_milvus_client() -> None:
    """关闭共享 MilvusClient，并允许后续按需重新连接。"""
    global _milvus_client
    with _client_lock:
        client = _milvus_client
        _milvus_client = None
        _store_cache.clear()
    if client is not None:
        try:
            client.close()
        except Exception:
            logger.debug("关闭 MilvusClient 失败", exc_info=True)


def _get_vector_store(collection_name: str) -> Optional[MilvusClient]:
    """获取已构建集合使用的 MilvusClient；不存在或为空返回 None。

    严格不自动建集合：先 has_collection 判断，再构造只读用途的 Milvus 对象。
    """
    if collection_name in _store_cache:
        return get_milvus_client()
    if not collection_exists(collection_name):
        return None
    try:
        store = get_milvus_client()
        if collection_count(collection_name) == 0:
            return None
        store.load_collection(collection_name)
        _store_cache.add(collection_name)
        return store
    except Exception as e:
        raise KnowledgeStoreError(f"加载集合 {collection_name} 失败：{type(e).__name__}: {e}") from e


# ==================== 结果格式化 ====================

def _relevance_stars(score: float) -> str:
    """COSINE 相似度（越大越相关）转易读标记。阈值可随评测校准。"""
    if score >= 0.6:
        return "★★★ 高度相关"
    if score >= 0.4:
        return "★★☆ 相关"
    return "★☆☆ 低相关"


def _format_meta_hint(logical_type: str, metadata: dict) -> str:
    """按逻辑知识类型格式化标题提示，而不是比较物理集合名。"""
    if logical_type == "dish":
        name = metadata.get("name", "")
        category = metadata.get("category", "")
        hint = f"【{name}】" if name else ""
        if category:
            hint += f"[{category}] "
        return hint
    if logical_type == "dietary":
        condition = metadata.get("condition", "")
        return f"【{condition}】" if condition else ""
    if logical_type == "faq":
        category = metadata.get("category", "")
        return f"【{category}】" if category else ""
    return ""


# ==================== 核心检索 ====================

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=6),
    retry=retry_if_exception_type((MilvusException, ConnectionError, OSError)),
    reraise=True,
)
def _similarity_search(
    store: MilvusClient,
    collection: str,
    query: str,
    top_k: int,
    rag_id: str = "-",
):
    embedding_started = time.perf_counter()
    query_vector = get_embeddings().embed_query(query)
    logger.info(
        "[RAG][%s] EMBEDDING_DONE dim=%d elapsed_ms=%.0f",
        rag_id,
        len(query_vector),
        (time.perf_counter() - embedding_started) * 1000,
    )

    search_started = time.perf_counter()
    rows = store.search(
        collection_name=collection,
        data=[query_vector],
        anns_field="vector",
        limit=top_k,
        output_fields=["*"],
        search_params={"metric_type": RAG_METRIC_TYPE, "params": {}},
    )
    hits = rows[0] if rows else []
    logger.info(
        "[RAG][%s] MILVUS_DONE collection=%s candidates=%d elapsed_ms=%.0f",
        rag_id,
        collection,
        len(hits),
        (time.perf_counter() - search_started) * 1000,
    )
    out = []
    for hit in hits:
        entity = dict(hit.get("entity") or {})
        text = str(entity.pop("text", ""))
        entity.setdefault("pk", hit.get("id"))
        score = hit.get("distance", hit.get("score", 0.0))
        out.append((Document(page_content=text, metadata=entity), float(score)))
    return out


def search_knowledge(
    query: str,
    collection: str,
    top_k: int = RAG_TOP_K,
    logical_type: Optional[str] = None,
    min_score: float = RAG_MIN_SCORE,
    trace_id: Optional[str] = None,
) -> str:
    """语义检索并格式化为 LLM 可读文本。

    参数:
        query: 自然语言查询
        collection: 物理集合名
        top_k: 候选条数
        logical_type: 逻辑知识类型 dish/dietary/faq；缺省时按集合名推断
        min_score: COSINE 相似度下限，低于此值视为无关被过滤

    返回:
        格式化文本（可能是“未找到足够相关内容”）。
        集合未构建抛 KnowledgeNotBuilt；Milvus 故障抛 KnowledgeStoreError，
        交由工具层/健康检查区分处理，绝不伪装成“没有知识”。
    """
    rag_id = trace_id or uuid.uuid4().hex[:8]
    lt = logical_type or logical_type_of(collection)
    safe_query = " ".join(query.split())[:120]
    started = time.perf_counter()
    logger.info(
        "[RAG][%s] START type=%s collection=%s top_k=%d min_score=%.3f query=%r",
        rag_id, lt, collection, top_k, min_score, safe_query,
    )

    store = _get_vector_store(collection)
    if store is None:
        cn = _LOGICAL_META.get(logical_type or logical_type_of(collection), {}).get(
            "empty_name", collection
        )
        raise KnowledgeNotBuilt(
            f"知识库（{cn}，集合 {collection}）尚未构建或为空。"
            f"请先运行 python build_knowledge_base.py 并确认 Milvus 已启动。"
        )

    try:
        hits = _similarity_search(store, collection, query, top_k, rag_id=rag_id)
    except KnowledgeStoreError:
        logger.exception("[RAG][%s] FAILED collection=%s", rag_id, collection)
        raise
    except Exception as e:
        logger.exception("[RAG][%s] FAILED collection=%s", rag_id, collection)
        raise KnowledgeStoreError(f"检索集合 {collection} 失败：{type(e).__name__}: {e}") from e

    # COSINE：越大越相似；过滤低相关结果
    scored = [(doc, float(score)) for doc, score in hits if float(score) >= min_score]
    scored.sort(key=lambda x: x[1], reverse=True)
    candidate_scores = ",".join(f"{float(score):.3f}" for _, score in hits) or "none"
    logger.info(
        "[RAG][%s] FILTER candidates=%d accepted=%d threshold=%.3f scores=[%s]",
        rag_id, len(hits), len(scored), min_score, candidate_scores,
    )

    if not scored:
        logger.info(
            "[RAG][%s] FINISH accepted=0 elapsed_ms=%.0f",
            rag_id, (time.perf_counter() - started) * 1000,
        )
        return "未在知识库中找到与该问题足够相关的内容，不要据此编造答案，可改用实时查询或提示用户换个问法。"

    label = _LOGICAL_META.get(lt, {}).get("label", "知识")
    lines = [f"为您找到以下{label}（共{len(scored)}条，按相关度排序）："]
    for i, (doc, score) in enumerate(scored, 1):
        metadata = doc.metadata or {}
        hint = _format_meta_hint(lt, metadata)
        lines.append(
            f"  {i}. {hint}（{_relevance_stars(score)}，相似度{score:.3f}）\n"
            f"     {doc.page_content.strip()[:300]}"
        )
    logger.info(
        "[RAG][%s] FINISH accepted=%d elapsed_ms=%.0f",
        rag_id, len(scored), (time.perf_counter() - started) * 1000,
    )
    return "\n".join(lines)


# ==================== 健康状态 ====================

def rag_status() -> dict:
    """供健康接口：Milvus 连通性、三个集合是否存在及计数。"""
    collections = list(RAG_KNOWLEDGE_COLLECTIONS.values())
    up = ping_milvus()
    detail = {}
    if up:
        for name in collections:
            try:
                detail[name] = collection_count(name) if collection_exists(name) else -1
            except KnowledgeStoreError:
                detail[name] = -1
    return {
        "milvus_up": up,
        "milvus_uri": MILVUS_URI,
        "metric_type": RAG_METRIC_TYPE,
        "collections": detail,
        "ready": up and all(c > 0 for c in detail.values()) if up else False,
    }
