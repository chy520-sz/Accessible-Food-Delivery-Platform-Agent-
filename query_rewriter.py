"""
Query 改写模块：提升模糊/口语化查询的召回率。

两种模式：
  - multi_query：用 LLM 把用户 Query 扩展成 N 个语义等价查询，分别检索后合并。
  - hyde：用 LLM 生成一个假设答案文档（Hypothetical Document），用文档 embedding
    替代原始 Query embedding 去检索（假设文档通常比短 Query 语义更丰富）。

设计原则：
  - 改写失败/超时不阻塞检索：降级为原始 Query，记录 warn 日志。
  - 同步调用（search_knowledge 是同步链路），用 LLM.invoke()。
  - 输出严格结构化（JSON 行 / 纯文本），避免 LLM 自由发挥导致解析失败。
"""

import json
import logging
import time
from typing import Optional

from config import (
    RAG_QUERY_REWRITE_MODE,
    RAG_QUERY_REWRITE_NUM,
    RAG_QUERY_REWRITE_TIMEOUT,
)

logger = logging.getLogger("query_rewriter")

# 多 Query 扩展的系统提示：要求输出严格 JSON 数组，每行一个查询
_MULTI_QUERY_SYSTEM = """你是外卖点餐助手的查询改写专家。用户会输入一个自然语言查询，
请把它改写成 {num} 个语义等价但表述不同的查询，用于向量检索召回。

要求：
1. 每个查询都是独立的、可单独用于检索的完整句子或短语。
2. 改写方向包括：同义词替换、口语化↔书面化、补充隐含意图、拆分复合需求。
3. 不要添加用户没有提到的菜品名、店铺名或分类。
4. 输出严格 JSON 数组，例如：["查询1", "查询2", "查询3"]，不要输出任何其他文字。"""

# HyDE 的系统提示：要求生成一个假设的答案文档
_HYDE_SYSTEM = """你是外卖点餐知识库的文档撰写专家。用户会输入一个查询，
请假设知识库中存在一篇完美回答该查询的菜品知识文档，写出这篇文档的正文。

要求：
1. 文档内容是菜品知识（名称、分类、口味、食材、描述、适合人群等），不是对话回复。
2. 用中文，200~400 字，信息密度高，包含具体的菜品特征词。
3. 不要编造用户没有提到的具体店铺名；可以用通用描述。
4. 直接输出文档正文，不要输出标题、解释或其他文字。"""


def _call_llm_sync(system_prompt: str, user_query: str, timeout: float) -> str:
    """同步调用 LLM，带超时控制。失败抛异常由调用方降级。"""
    from llm_client import get_llm

    llm = get_llm()
    # LangChain ChatOpenAI.invoke 支持 timeout 参数（覆盖默认）
    resp = llm.invoke(
        [
            ("system", system_prompt),
            ("human", user_query),
        ],
        config={"timeout": timeout},
    )
    content = resp.content if hasattr(resp, "content") else str(resp)
    return content.strip()


def rewrite_multi_query(query: str, num: int = RAG_QUERY_REWRITE_NUM) -> list[str]:
    """多 Query 扩展：生成 num 个语义等价查询。

    返回列表包含原始 Query + 扩展 Query（去重后）。失败时返回 [query]。
    """
    started = time.perf_counter()
    try:
        system = _MULTI_QUERY_SYSTEM.format(num=num)
        raw = _call_llm_sync(system, query, RAG_QUERY_REWRITE_TIMEOUT)
        # 解析 JSON 数组（容错：去掉可能的 markdown 代码块标记）
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # 去掉 ```json ... ``` 包裹
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        parsed = json.loads(cleaned)
        if not isinstance(parsed, list):
            raise ValueError(f"LLM 返回不是数组: {type(parsed)}")
        expanded = [str(q).strip() for q in parsed if str(q).strip()]
        # 合并原始 Query，去重（保持顺序）
        seen = set()
        result = []
        for q in [query] + expanded:
            key = q.lower()
            if key not in seen:
                seen.add(key)
                result.append(q)
        logger.info(
            "[rewrite] MULTI_QUERY done original=%r expanded=%d total=%d elapsed_ms=%.0f",
            query, len(expanded), len(result), (time.perf_counter() - started) * 1000,
        )
        return result
    except Exception as e:
        logger.warning(
            "[rewrite] MULTI_QUERY 失败，降级为原始 Query: %s: %s",
            type(e).__name__, e,
        )
        return [query]


def rewrite_hyde(query: str) -> str:
    """HyDE：生成假设答案文档，用于替代原始 Query 做 embedding 检索。

    返回假设文档文本。失败时返回原始 Query。
    """
    started = time.perf_counter()
    try:
        doc = _call_llm_sync(_HYDE_SYSTEM, query, RAG_QUERY_REWRITE_TIMEOUT)
        if not doc or len(doc) < 20:
            raise ValueError(f"生成的假设文档过短: {len(doc)} chars")
        logger.info(
            "[rewrite] HYDE done original=%r doc_len=%d elapsed_ms=%.0f",
            query, len(doc), (time.perf_counter() - started) * 1000,
        )
        return doc
    except Exception as e:
        logger.warning(
            "[rewrite] HYDE 失败，降级为原始 Query: %s: %s",
            type(e).__name__, e,
        )
        return query


def rewrite_query(
    query: str,
    mode: str = RAG_QUERY_REWRITE_MODE,
    num: int = RAG_QUERY_REWRITE_NUM,
) -> dict:
    """统一入口：按配置模式改写 Query。

    返回 dict：
      - mode: 实际使用的模式（multi_query / hyde / none）
      - queries: 用于稠密检索的查询列表（multi_query 模式有多个，hyde/原始模式1个）
      - bm25_query: 用于 BM25 检索的查询（始终用原始 Query，关键词精确匹配）
      - hyde_doc: hyde 模式下生成的假设文档（其他模式为 None）
    """
    if mode == "multi_query":
        queries = rewrite_multi_query(query, num=num)
        return {
            "mode": "multi_query",
            "queries": queries,
            "bm25_query": query,
            "hyde_doc": None,
        }
    if mode == "hyde":
        hyde_doc = rewrite_hyde(query)
        return {
            "mode": "hyde",
            "queries": [hyde_doc],
            "bm25_query": query,
            "hyde_doc": hyde_doc,
        }
    # 未启用或未知模式：不改写
    return {
        "mode": "none",
        "queries": [query],
        "bm25_query": query,
        "hyde_doc": None,
    }
