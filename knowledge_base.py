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
    RAG_BM25_DROP_RATIO,
    RAG_BM25_RESCUE_RANK,
    RAG_BM25_TOP_K,
    RAG_CALL_TIMEOUT,
    RAG_FAILURE_COOLDOWN,
    RAG_HYBRID_RRF_K,
    RAG_HYBRID_SEARCH_ENABLED,
    RAG_KNOWLEDGE_COLLECTIONS,
    RAG_METRIC_TYPE,
    RAG_MIN_SCORE,
    RAG_METADATA_FILTER_ENABLED,
    RAG_QUERY_REWRITE_ENABLED,
    RAG_RERANK_BATCH_SIZE,
    RAG_RERANK_CANDIDATES,
    RAG_RERANK_ENABLED,
    RAG_RERANK_MIN_SCORE,
    RAG_RERANK_MODEL,
    RAG_RERANK_TIMEOUT,
    RAG_SPARSE_FIELD,
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
        return bool(get_milvus_client().has_collection(
            collection_name, timeout=RAG_CALL_TIMEOUT
        ))
    except KnowledgeStoreError:
        raise
    except Exception as e:
        raise KnowledgeStoreError(f"检查集合 {collection_name} 是否存在失败：{e}") from e


def collection_count(collection_name: str) -> int:
    """返回集合实体数（集合不存在返回 0）。"""
    try:
        client = get_milvus_client()
        if not client.has_collection(collection_name, timeout=RAG_CALL_TIMEOUT):
            return 0
        stats = client.get_collection_stats(collection_name, timeout=RAG_CALL_TIMEOUT)
        return int(stats.get("row_count", 0))
    except Exception as e:
        raise KnowledgeStoreError(f"读取集合 {collection_name} 计数失败：{e}") from e


def reset_store_cache() -> None:
    """构建/删除集合后清空向量存储缓存。"""
    _store_cache.clear()
    _bm25_capable_cache.clear()
    _failure_breaker.clear()


# ==================== 失败快速熔断 ====================
# Milvus 故障（如集合长期卡在 Loading）时，若每个请求都完整等一次超时，
# 工具链会被反复拖慢，还会连带产生被中断的悬空工具调用。
# 这里记录失败时间，冷却期内直接快速失败，冷却期后再给一次恢复机会。

_failure_breaker: dict[str, float] = {}
_breaker_lock = threading.Lock()


def _breaker_blocked(collection: str) -> Optional[float]:
    """返回剩余冷却秒数；未处于冷却期返回 None。"""
    if RAG_FAILURE_COOLDOWN <= 0:
        return None
    now = time.monotonic()
    with _breaker_lock:
        failed_at = _failure_breaker.get(collection)
        if failed_at is None:
            return None
        remaining = RAG_FAILURE_COOLDOWN - (now - failed_at)
        if remaining <= 0:
            _failure_breaker.pop(collection, None)
            return None
        return remaining


def _breaker_record_failure(collection: str) -> None:
    if RAG_FAILURE_COOLDOWN <= 0:
        return
    with _breaker_lock:
        _failure_breaker[collection] = time.monotonic()


def _breaker_record_success(collection: str) -> None:
    with _breaker_lock:
        _failure_breaker.pop(collection, None)


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
        # 必须限时：集合卡在 Loading 时 load_collection 会无限等待，
        # 进而拖满整轮对话预算，并让中断的 run 在 checkpoint 留下悬空工具调用。
        store.load_collection(collection_name, timeout=RAG_CALL_TIMEOUT)
        _store_cache.add(collection_name)
        return store
    except Exception as e:
        raise KnowledgeStoreError(
            f"加载集合 {collection_name} 失败（超时 {RAG_CALL_TIMEOUT}s）："
            f"{type(e).__name__}: {e}"
        ) from e


# ==================== 混合检索：BM25 能力检测 + 双路检索 + RRF 融合 ====================

# 集合是否具备 BM25 稀疏向量字段的缓存（重建集合后随 reset_store_cache 清空）
_bm25_capable_cache: dict[str, bool] = {}


def collection_has_bm25(collection: str) -> bool:
    """检测集合 schema 是否包含 BM25 稀疏向量字段（决定能否走混合检索）。"""
    if collection in _bm25_capable_cache:
        return _bm25_capable_cache[collection]
    try:
        client = get_milvus_client()
        desc = client.describe_collection(collection, timeout=RAG_CALL_TIMEOUT)
        field_names = [f.get("name") for f in desc.get("fields", [])]
        capable = RAG_SPARSE_FIELD in field_names
    except Exception:
        logger.debug("检测集合 %s 的 BM25 能力失败，按不支持处理", collection, exc_info=True)
        capable = False
    _bm25_capable_cache[collection] = capable
    return capable


def _parse_hit(hit: dict):
    """统一解析稠密/BM25/hybrid 搜索结果的单条 hit，兼容主键在 id/pk/entity.pk 的差异。"""
    entity = dict(hit.get("entity") or {})
    text = str(entity.pop("text", ""))
    pk = hit.get("id") or hit.get("pk") or entity.get("pk")
    if pk is not None:
        entity.setdefault("pk", pk)
    # 稀疏向量字段体积大且检索用不到，从元数据中剔除避免污染上下文
    entity.pop(RAG_SPARSE_FIELD, None)
    entity.pop("vector", None)
    score = float(hit.get("distance", hit.get("score", 0.0)) or 0.0)
    return Document(page_content=text, metadata=entity), score


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=6),
    retry=retry_if_exception_type((MilvusException, ConnectionError, OSError)),
    reraise=True,
)
def _bm25_search(
    store: MilvusClient,
    collection: str,
    query: str,
    top_k: int,
    rag_id: str = "-",
    filter_expr: Optional[str] = None,
):
    """BM25 全文检索（稀疏向量路）：直接传原始文本，Milvus 端用中文分析器分词。"""
    started = time.perf_counter()
    search_kwargs = {
        "collection_name": collection,
        "data": [query],
        "anns_field": RAG_SPARSE_FIELD,
        "limit": top_k,
        "output_fields": ["*"],
        "search_params": {
            "metric_type": "BM25",
            "params": {"drop_ratio_search": RAG_BM25_DROP_RATIO},
        },
        "timeout": RAG_CALL_TIMEOUT,
    }
    if filter_expr:
        search_kwargs["filter"] = filter_expr
    rows = store.search(**search_kwargs)
    hits = rows[0] if rows else []
    logger.info(
        "[RAG][%s] BM25_DONE collection=%s candidates=%d elapsed_ms=%.0f filter=%s",
        rag_id,
        collection,
        len(hits),
        (time.perf_counter() - started) * 1000,
        filter_expr or "none",
    )
    return [_parse_hit(hit) for hit in hits]


def _rrf_fuse(
    dense_hits: list,
    bm25_hits: list,
    k: int = RAG_HYBRID_RRF_K,
    top_k: int = RAG_TOP_K,
) -> list[dict]:
    """RRF（Reciprocal Rank Fusion）融合稠密与 BM25 两路排名。

    融合分 = Σ 1/(k + rank)，rank 从 1 开始。只依赖排名、不依赖两路原始分数量纲，
    因此能稳健融合 COSINE（-1~1）与 BM25（无上界）两种尺度。

    返回按融合分降序的 dict 列表，每项含：
      doc / rrf / dense_score / bm25_score / dense_rank / bm25_rank
    """
    fused: dict = {}

    def _slot(pk, doc):
        if pk not in fused:
            fused[pk] = {
                "doc": doc,
                "rrf": 0.0,
                "dense_score": 0.0,
                "bm25_score": 0.0,
                "dense_rank": None,
                "bm25_rank": None,
            }
        return fused[pk]

    for rank, (doc, score) in enumerate(dense_hits, 1):
        slot = _slot(doc.metadata.get("pk"), doc)
        slot["rrf"] += 1.0 / (k + rank)
        slot["dense_score"] = float(score)
        slot["dense_rank"] = rank
    for rank, (doc, score) in enumerate(bm25_hits, 1):
        slot = _slot(doc.metadata.get("pk"), doc)
        slot["rrf"] += 1.0 / (k + rank)
        slot["bm25_score"] = float(score)
        slot["bm25_rank"] = rank

    return sorted(fused.values(), key=lambda x: x["rrf"], reverse=True)[:top_k]


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


# ==================== 元数据过滤 ====================

def _build_filter_expr(filters: dict) -> str:
    """把 dict 格式的过滤条件转成 Milvus 标量过滤表达式。

    支持的 key 后缀：
      无后缀 / __eq   等值：category == "川菜"
      __in            IN：category in ["川菜", "湘菜"]
      __nin           NOT IN：category not in ["川菜"]
      __contains      数组包含：tags contains "辣"
      __gt / __gte    > / >=：spicy_level >= 2
      __lt / __lte    < / <=：price <= 50
      __ne            !=：category != "川菜"

    多条件用 && 连接。字符串值自动加双引号；数值不加。
    """
    if not filters:
        return ""
    parts = []
    for key, value in filters.items():
        # 解析字段名和操作符
        if "__" in key:
            field, op = key.rsplit("__", 1)
        else:
            field, op = key, "eq"

        def _quote(v):
            if isinstance(v, (int, float)):
                return str(v)
            if isinstance(v, bool):
                return "true" if v else "false"
            return f'"{v}"'

        if op == "eq":
            parts.append(f'{field} == {_quote(value)}')
        elif op == "ne":
            parts.append(f'{field} != {_quote(value)}')
        elif op == "in":
            vals = ", ".join(_quote(v) for v in value)
            parts.append(f'{field} in [{vals}]')
        elif op == "nin":
            vals = ", ".join(_quote(v) for v in value)
            parts.append(f'{field} not in [{vals}]')
        elif op == "contains":
            parts.append(f'{field} contains {_quote(value)}')
        elif op == "gt":
            parts.append(f'{field} > {_quote(value)}')
        elif op == "gte":
            parts.append(f'{field} >= {_quote(value)}')
        elif op == "lt":
            parts.append(f'{field} < {_quote(value)}')
        elif op == "lte":
            parts.append(f'{field} <= {_quote(value)}')
        else:
            # 未知操作符，退化为等值
            parts.append(f'{field} == {_quote(value)}')
    return " && ".join(parts)


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
    filter_expr: Optional[str] = None,
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
    search_kwargs = {
        "collection_name": collection,
        "data": [query_vector],
        "anns_field": "vector",
        "limit": top_k,
        "output_fields": ["*"],
        "search_params": {"metric_type": RAG_METRIC_TYPE, "params": {}},
        "timeout": RAG_CALL_TIMEOUT,
    }
    if filter_expr:
        search_kwargs["filter"] = filter_expr
    rows = store.search(**search_kwargs)
    hits = rows[0] if rows else []
    logger.info(
        "[RAG][%s] DENSE_DONE collection=%s candidates=%d elapsed_ms=%.0f filter=%s",
        rag_id,
        collection,
        len(hits),
        (time.perf_counter() - search_started) * 1000,
        filter_expr or "none",
    )
    return [_parse_hit(hit) for hit in hits]


def _score_distribution(scores: list[float], threshold: float) -> str:
    """把一组分数汇总成分布字符串（max/min/avg/median + 区间占比）。"""
    if not scores:
        return "candidates=0（无召回）"
    ordered = sorted(scores, reverse=True)
    n = len(ordered)
    median = (
        ordered[n // 2]
        if n % 2 == 1
        else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    )
    b_high = sum(1 for s in ordered if s >= 0.60)
    b_mid = sum(1 for s in ordered if 0.40 <= s < 0.60)
    b_low = sum(1 for s in ordered if threshold <= s < 0.40)
    b_rej = sum(1 for s in ordered if s < threshold)
    return (
        f"candidates={n} max={ordered[0]:.3f} min={ordered[-1]:.3f} "
        f"avg={sum(ordered) / n:.3f} median={median:.3f} | "
        f"≥0.60={b_high}({b_high * 100 // n}%) "
        f"0.40-0.60={b_mid}({b_mid * 100 // n}%) "
        f"{threshold:.2f}-0.40={b_low}({b_low * 100 // n}%) "
        f"<{threshold:.2f}(reject)={b_rej}({b_rej * 100 // n}%)"
    )


def _log_retrieval_detail(
    rag_id: str,
    lt: str,
    use_hybrid: bool,
    dense_hits: list,
    bm25_hits: list,
    fused: list[dict],
    min_score: float,
) -> None:
    """输出检索全链路日志：稠密/BM25 两路分数分布、各 chunk 排名、RRF 融合结果。"""
    # ---- 稠密路（语义）----
    dense_sorted = sorted(dense_hits, key=lambda x: x[1], reverse=True)
    dense_scores = [float(s) for _, s in dense_sorted]
    logger.info("[RAG][%s] DENSE_DIST %s", rag_id, _score_distribution(dense_scores, min_score))
    logger.info("[RAG][%s] DENSE_TOP_K type=%s (✓过阈值 ✗被过滤)", rag_id, lt)
    for i, (doc, score) in enumerate(dense_sorted, 1):
        meta = doc.metadata or {}
        hint = _format_meta_hint(lt, meta)
        passed = "✓" if float(score) >= min_score else "✗"
        snippet = doc.page_content.strip()[:120].replace("\n", " ")
        logger.info(
            "[RAG][%s]   D#%d %s cosine=%.3f %s %s",
            rag_id, i, passed, float(score), hint, snippet,
        )

    if not use_hybrid:
        return

    # ---- BM25 路（关键词）----
    bm25_sorted = sorted(bm25_hits, key=lambda x: x[1], reverse=True)
    bm25_scores = [float(s) for _, s in bm25_sorted]
    if bm25_scores:
        bm25_max, bm25_min = max(bm25_scores), min(bm25_scores)
        bm25_avg = sum(bm25_scores) / len(bm25_scores)
        logger.info(
            "[RAG][%s] BM25_DIST candidates=%d max=%.3f min=%.3f avg=%.3f（BM25分无上界，仅按排名融合）",
            rag_id, len(bm25_scores), bm25_max, bm25_min, bm25_avg,
        )
    else:
        logger.info("[RAG][%s] BM25_DIST candidates=0（无关键词命中）", rag_id)
    logger.info("[RAG][%s] BM25_TOP_K type=%s", rag_id, lt)
    for i, (doc, score) in enumerate(bm25_sorted, 1):
        meta = doc.metadata or {}
        hint = _format_meta_hint(lt, meta)
        rescue = "🔑" if i <= RAG_BM25_RESCUE_RANK else " "
        snippet = doc.page_content.strip()[:120].replace("\n", " ")
        logger.info(
            "[RAG][%s]   B#%d %s bm25=%.3f %s %s",
            rag_id, i, rescue, float(score), hint, snippet,
        )

    # ---- RRF 融合结果 ----
    logger.info("[RAG][%s] FUSED_RANK type=%s（RRF 融合后最终排序）", rag_id, lt)
    for i, item in enumerate(fused, 1):
        doc = item["doc"]
        meta = doc.metadata or {}
        hint = _format_meta_hint(lt, meta)
        dr = f"D#{item['dense_rank']}" if item["dense_rank"] is not None else "D#-"
        br = f"B#{item['bm25_rank']}" if item["bm25_rank"] is not None else "B#-"
        snippet = doc.page_content.strip()[:100].replace("\n", " ")
        logger.info(
            "[RAG][%s]   F#%d rrf=%.5f [%s %s] cosine=%.3f bm25=%.3f %s %s",
            rag_id, i, item["rrf"], dr, br,
            item["dense_score"], item["bm25_score"], hint, snippet,
        )


def _rerank_stars(score: float) -> str:
    """gte-rerank 相关性分数（0~1）转易读标记。阈值按实测校准。"""
    if score >= 0.5:
        return "★★★ 高度相关"
    if score >= 0.15:
        return "★★☆ 相关"
    return "★☆☆ 低相关"


def _log_rerank_detail(
    rag_id: str,
    lt: str,
    candidates: list[dict],
    rerank_results: list[tuple[int, float]],
    min_rerank_score: float,
    top_k: int,
) -> None:
    """输出 Rerank 精排日志：精排前后排名变化、rerank 分数分布、阈值过滤。"""
    # 建立 fused 候选下标 -> RRF 排名 的映射
    rrf_rank_map = {idx: idx + 1 for idx in range(len(candidates))}

    scores = [s for _, s in rerank_results]
    if scores:
        logger.info(
            "[RAG][%s] RERANK_DIST model=%s candidates=%d max=%.4f min=%.4f avg=%.4f "
            "| ≥0.50=%d 0.15-0.50=%d %.2f-0.15=%d <%.2f(reject)=%d",
            rag_id, RAG_RERANK_MODEL, len(scores),
            max(scores), min(scores), sum(scores) / len(scores),
            sum(1 for s in scores if s >= 0.50),
            sum(1 for s in scores if 0.15 <= s < 0.50),
            min_rerank_score,
            sum(1 for s in scores if min_rerank_score <= s < 0.15),
            min_rerank_score,
            sum(1 for s in scores if s < min_rerank_score),
        )

    logger.info("[RAG][%s] RERANK_DETAIL type=%s (RRF排名→Rerank排名，✓保留 ✗过滤)", rag_id, lt)
    for new_rank, (cand_idx, rr_score) in enumerate(rerank_results, 1):
        item = candidates[cand_idx]
        doc = item["doc"]
        meta = doc.metadata or {}
        hint = _format_meta_hint(lt, meta)
        old_rank = rrf_rank_map.get(cand_idx, "-")
        passed = "✓" if rr_score >= min_rerank_score and new_rank <= top_k else "✗"
        snippet = doc.page_content.strip()[:100].replace("\n", " ")
        logger.info(
            "[RAG][%s]   R#%d %s rerank=%.4f (RRF#%s→#%d) cosine=%.3f %s %s",
            rag_id, new_rank, passed, rr_score, old_rank, new_rank,
            item["dense_score"], hint, snippet,
        )


def search_knowledge(
    query: str,
    collection: str,
    top_k: int = RAG_TOP_K,
    logical_type: Optional[str] = None,
    min_score: float = RAG_MIN_SCORE,
    trace_id: Optional[str] = None,
    filters: Optional[dict] = None,
) -> str:
    """语义检索并格式化为 LLM 可读文本。

    参数:
        query: 自然语言查询
        collection: 物理集合名
        top_k: 候选条数
        logical_type: 逻辑知识类型 dish/dietary/faq；缺省时按集合名推断
        min_score: COSINE 相似度下限，低于此值视为无关被过滤
        filters: 元数据预过滤条件（Milvus 标量过滤），例如：
            {"category": "川菜"}
            {"category__in": ["川菜", "湘菜"], "spicy_level__gte": 2}
            {"tags__contains": "辣", "price__lte": 50}

    返回:
        格式化文本（可能是"未找到足够相关内容"）。
        集合未构建抛 KnowledgeNotBuilt；Milvus 故障抛 KnowledgeStoreError，
        交由工具层/健康检查区分处理，绝不伪装成"没有知识"。
    """
    rag_id = trace_id or uuid.uuid4().hex[:8]
    lt = logical_type or logical_type_of(collection)
    safe_query = " ".join(query.split())[:120]
    started = time.perf_counter()

    # 元数据过滤：构建 Milvus filter expression（在检索阶段预过滤，减少候选池）
    filter_expr = ""
    if RAG_METADATA_FILTER_ENABLED and filters:
        filter_expr = _build_filter_expr(filters)
        logger.info(
            "[RAG][%s] METADATA_FILTER expr=%s raw_filters=%s",
            rag_id, filter_expr, filters,
        )

    # 是否启用 Rerank 精排：启用后召回阶段扩大候选池（粗排多召回、精排少而准）
    use_rerank = RAG_RERANK_ENABLED
    recall_k = max(RAG_RERANK_CANDIDATES, top_k) if use_rerank else top_k
    logger.info(
        "[RAG][%s] START type=%s collection=%s final_top_k=%d recall_k=%d "
        "min_score=%.3f hybrid=%s rerank=%s filter=%s query=%r",
        rag_id, lt, collection, top_k, recall_k, min_score,
        RAG_HYBRID_SEARCH_ENABLED, use_rerank, filter_expr or "none", safe_query,
    )

    remaining = _breaker_blocked(collection)
    if remaining is not None:
        logger.warning(
            "[RAG][%s] SKIP collection=%s 处于失败冷却中（剩余 %.0fs），快速失败",
            rag_id, collection, remaining,
        )
        raise KnowledgeStoreError(
            f"知识库集合 {collection} 最近检索失败，正在冷却（剩余约 {remaining:.0f}s）。"
        )

    # 检索中间状态（try 外初始化，供后续过滤/格式化使用）
    dense_hits: list = []
    bm25_hits: list = []
    fused: list[dict] = []
    use_hybrid = False

    try:
        store = _get_vector_store(collection)
        if store is None:
            cn = _LOGICAL_META.get(logical_type or logical_type_of(collection), {}).get(
                "empty_name", collection
            )
            raise KnowledgeNotBuilt(
                f"知识库（{cn}，集合 {collection}）尚未构建或为空。"
                f"请先运行 python build_knowledge_base.py 并确认 Milvus 已启动。"
            )

        # ---- Query 改写（多 Query 扩展 / HyDE 假设文档嵌入）----
        dense_queries = [query]
        bm25_query = query
        rewrite_mode = "none"
        if RAG_QUERY_REWRITE_ENABLED:
            try:
                from query_rewriter import rewrite_query
                rw = rewrite_query(query)
                dense_queries = rw["queries"]
                bm25_query = rw["bm25_query"]
                rewrite_mode = rw["mode"]
                logger.info(
                    "[RAG][%s] QUERY_REWRITE mode=%s dense_queries=%d "
                    "bm25_query=%r hyde_doc_len=%d",
                    rag_id, rewrite_mode, len(dense_queries), bm25_query,
                    len(rw["hyde_doc"]) if rw.get("hyde_doc") else 0,
                )
            except Exception as rw_err:
                logger.warning(
                    "[RAG][%s] Query 改写失败，降级为原始 Query: %s: %s",
                    rag_id, type(rw_err).__name__, rw_err,
                )

        # ---- 第 1 路：稠密向量语义检索（COSINE）----
        # 多 Query 模式：对每个改写后的 query 分别检索，合并去重（保留最高分）后按分数降序
        if len(dense_queries) == 1:
            dense_hits = _similarity_search(
                store, collection, dense_queries[0], recall_k,
                rag_id=rag_id, filter_expr=filter_expr,
            )
        else:
            merged: dict = {}
            for qi, q in enumerate(dense_queries, 1):
                q_hits = _similarity_search(
                    store, collection, q, recall_k,
                    rag_id=f"{rag_id}#q{qi}", filter_expr=filter_expr,
                )
                for doc, score in q_hits:
                    pk = doc.metadata.get("pk")
                    if pk not in merged or float(score) > merged[pk][1]:
                        merged[pk] = (doc, float(score))
            dense_hits = sorted(merged.values(), key=lambda x: x[1], reverse=True)
            logger.info(
                "[RAG][%s] MULTI_QUERY_MERGE queries=%d raw_hits=%d merged_unique=%d",
                rag_id, len(dense_queries),
                sum(1 for _ in dense_queries) * recall_k, len(dense_hits),
            )

        # ---- 第 2 路：BM25 全文关键词检索（集合具备 sparse_vector 时启用）----
        # BM25 始终用原始 Query（关键词精确匹配，改写后的语义查询不适合关键词检索）
        use_hybrid = RAG_HYBRID_SEARCH_ENABLED and collection_has_bm25(collection)
        if use_hybrid:
            try:
                bm25_hits = _bm25_search(
                    store, collection, bm25_query, recall_k,
                    rag_id=rag_id, filter_expr=filter_expr,
                )
            except Exception as bm_err:
                # BM25 路故障不应拖垮语义检索：记录后降级为纯稠密
                logger.warning(
                    "[RAG][%s] BM25 路检索失败，降级为纯稠密检索: %s: %s",
                    rag_id, type(bm_err).__name__, bm_err,
                )
                use_hybrid = False
                bm25_hits = []
            if use_hybrid:
                fused = _rrf_fuse(
                    dense_hits, bm25_hits, k=RAG_HYBRID_RRF_K, top_k=recall_k
                )
                logger.info(
                    "[RAG][%s] FUSE mode=HYBRID rrf_k=%d dense_candidates=%d bm25_candidates=%d fused=%d",
                    rag_id, RAG_HYBRID_RRF_K, len(dense_hits), len(bm25_hits), len(fused),
                )
        else:
            logger.info(
                "[RAG][%s] FUSE mode=DENSE_ONLY reason=%s",
                rag_id,
                "config_disabled" if not RAG_HYBRID_SEARCH_ENABLED else "collection_without_bm25",
            )
            # 纯稠密模式：把 dense_hits 包装成 fused 结构，统一后续处理
            fused = [
                {
                    "doc": doc,
                    "rrf": 0.0,
                    "dense_score": float(score),
                    "bm25_score": 0.0,
                    "dense_rank": rank,
                    "bm25_rank": None,
                    "rerank_score": None,
                    "rerank_rank": None,
                }
                for rank, (doc, score) in enumerate(
                    sorted(dense_hits, key=lambda x: x[1], reverse=True), 1
                )
            ]

        _breaker_record_success(collection)

        # ---- 召回链路日志：分数分布 + 每个 chunk 详情 ----
        _log_retrieval_detail(rag_id, lt, use_hybrid, dense_hits, bm25_hits, fused, min_score)
    except KnowledgeNotBuilt:
        # 集合尚未构建不是故障，不进入冷却，便于构建后立即恢复
        raise
    except KnowledgeStoreError:
        _breaker_record_failure(collection)
        logger.exception("[RAG][%s] FAILED collection=%s", rag_id, collection)
        raise
    except Exception as e:
        _breaker_record_failure(collection)
        logger.exception("[RAG][%s] FAILED collection=%s", rag_id, collection)
        raise KnowledgeStoreError(f"检索集合 {collection} 失败：{type(e).__name__}: {e}") from e

    # 给混合模式的 fused 项补齐 rerank 字段
    for item in fused:
        item.setdefault("rerank_score", None)
        item.setdefault("rerank_rank", None)

    # ==================== Rerank 精排 ====================
    rerank_active = False
    if use_rerank and fused:
        # Rerank 前的粗筛：启用精排时尽量保留候选交给 Cross-Encoder 判断，
        # 只做极宽松的预筛（稠密过低且 BM25 未命中的明显噪声剔除），避免浪费调用。
        pre_candidates = [
            item for item in fused
            if item["dense_score"] >= min_score * 0.5
            or (item["bm25_rank"] is not None and item["bm25_rank"] <= RAG_BM25_RESCUE_RANK * 2)
        ]
        # 粗筛后若候选过少，回退为全部 fused，避免精排无米下锅
        if len(pre_candidates) < min(3, len(fused)):
            pre_candidates = list(fused)

        try:
            from rerank_client import get_reranker
            reranker = get_reranker()
            doc_texts = [item["doc"].page_content for item in pre_candidates]
            rerank_started = time.perf_counter()
            rerank_pairs = reranker.rerank(query, doc_texts, top_n=len(pre_candidates))
            logger.info(
                "[RAG][%s] RERANK_DONE model=%s pre_candidates=%d elapsed_ms=%.0f",
                rag_id, RAG_RERANK_MODEL, len(pre_candidates),
                (time.perf_counter() - rerank_started) * 1000,
            )

            # 用 rerank 分数重建排序结果
            reranked: list[dict] = []
            for new_rank, (cand_idx, rr_score) in enumerate(rerank_pairs, 1):
                item = pre_candidates[cand_idx]
                item["rerank_score"] = rr_score
                item["rerank_rank"] = new_rank
                reranked.append(item)

            _log_rerank_detail(
                rag_id, lt, pre_candidates, rerank_pairs,
                RAG_RERANK_MIN_SCORE, top_k,
            )

            # 精排阈值过滤 + 截取最终 top_k
            scored = [
                item for item in reranked
                if item["rerank_score"] is not None
                and item["rerank_score"] >= RAG_RERANK_MIN_SCORE
            ][:top_k]
            rerank_active = True
            rejected = len(reranked) - len(scored)
            logger.info(
                "[RAG][%s] FILTER mode=RERANK reranked=%d accepted=%d rejected=%d "
                "rerank_threshold=%.3f",
                rag_id, len(reranked), len(scored), rejected, RAG_RERANK_MIN_SCORE,
            )
        except Exception as rr_err:
            # Rerank 故障降级：回退到稠密阈值 + BM25 补充通道的粗排结果
            logger.warning(
                "[RAG][%s] Rerank 精排失败，降级为 RRF 粗排: %s: %s",
                rag_id, type(rr_err).__name__, rr_err,
            )
            rerank_active = False

    # ==================== 粗排阈值过滤（未启用 / Rerank 降级时使用）====================
    if not rerank_active:
        if use_hybrid:
            # 混合模式双通道保留：
            #   通道A（语义）：稠密 COSINE 分 >= min_score
            #   通道B（关键词精确命中补充）：BM25 排名进入前 RAG_BM25_RESCUE_RANK
            scored = []
            rescue_count = 0
            for item in fused:
                by_dense = item["dense_score"] >= min_score and item["dense_rank"] is not None
                by_bm25 = item["bm25_rank"] is not None and item["bm25_rank"] <= RAG_BM25_RESCUE_RANK
                if by_dense or by_bm25:
                    if by_bm25 and not by_dense:
                        rescue_count += 1
                    scored.append(item)
            scored = scored[:top_k]
            logger.info(
                "[RAG][%s] FILTER mode=HYBRID fused=%d accepted=%d (dense通道 + bm25补充%d) threshold=%.3f",
                rag_id, len(fused), len(scored), rescue_count, min_score,
            )
        else:
            # 纯稠密模式：COSINE 阈值过滤
            scored = [
                item for item in fused if item["dense_score"] >= min_score
            ][:top_k]
            logger.info(
                "[RAG][%s] FILTER mode=DENSE_ONLY candidates=%d accepted=%d threshold=%.3f",
                rag_id, len(fused), len(scored), min_score,
            )

    if not scored:
        logger.info(
            "[RAG][%s] FINISH accepted=0 elapsed_ms=%.0f",
            rag_id, (time.perf_counter() - started) * 1000,
        )
        return "未在知识库中找到与该问题足够相关的内容，不要据此编造答案，可改用实时查询或提示用户换个问法。"

    label = _LOGICAL_META.get(lt, {}).get("label", "知识")
    lines = [f"为您找到以下{label}（共{len(scored)}条，按相关度排序）："]
    for i, item in enumerate(scored, 1):
        doc = item["doc"]
        metadata = doc.metadata or {}
        hint = _format_meta_hint(lt, metadata)

        if rerank_active and item.get("rerank_score") is not None:
            # Rerank 精排结果：展示 Cross-Encoder 相关性分数
            rr = item["rerank_score"]
            lines.append(
                f"  {i}. {hint}（{_rerank_stars(rr)}，相关度{rr:.3f}）\n"
                f"     {doc.page_content.strip()[:300]}"
            )
        else:
            # 粗排结果：展示稠密 COSINE 分数（+ BM25 命中标记）
            score = item["dense_score"]
            match_note = ""
            if use_hybrid:
                if item["bm25_rank"] is not None and item["dense_rank"] is None:
                    match_note = f"，关键词精确命中#BM25第{item['bm25_rank']}"
                elif item["bm25_rank"] is not None and score < min_score:
                    match_note = f"，关键词命中#BM25第{item['bm25_rank']}"
            lines.append(
                f"  {i}. {hint}（{_relevance_stars(score) if score > 0 else '关键词匹配'}，"
                f"相似度{score:.3f}{match_note}）\n"
                f"     {doc.page_content.strip()[:300]}"
            )
    mode = "RERANK" if rerank_active else ("HYBRID" if use_hybrid else "DENSE_ONLY")
    logger.info(
        "[RAG][%s] FINISH mode=%s accepted=%d elapsed_ms=%.0f",
        rag_id, mode, len(scored), (time.perf_counter() - started) * 1000,
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
