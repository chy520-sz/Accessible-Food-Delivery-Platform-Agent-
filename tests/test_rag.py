"""RAG 纯逻辑测试：集合/逻辑类型映射、COSINE 相关度、稳定主键、嵌入参数校验。

这些测试不连接 Milvus、不调用云端嵌入。
"""

import logging

import pytest
from langchain_core.documents import Document

import knowledge_base as kb
from config import (
    RAG_COLLECTION_DIETARY,
    RAG_COLLECTION_FAQ,
    RAG_COLLECTION_FOOD,
)


def test_logical_type_mapping():
    assert kb.logical_type_of(RAG_COLLECTION_FOOD) == "dish"
    assert kb.logical_type_of(RAG_COLLECTION_DIETARY) == "dietary"
    assert kb.logical_type_of(RAG_COLLECTION_FAQ) == "faq"
    # 元数据中的 knowledge_type 优先
    assert kb.logical_type_of("whatever", {"knowledge_type": "faq"}) == "faq"


def test_cosine_relevance_monotonic():
    # COSINE 越大越相关
    high = kb._relevance_stars(0.8)
    low = kb._relevance_stars(0.3)
    assert "高度相关" in high
    assert "低相关" in low


def test_format_meta_hint_by_logical_type():
    assert "宫保鸡丁" in kb._format_meta_hint("dish", {"name": "宫保鸡丁", "category": "川菜"})
    assert "糖尿病" in kb._format_meta_hint("dietary", {"condition": "糖尿病"})
    assert "配送" in kb._format_meta_hint("faq", {"category": "配送"})
    # 不再依赖物理集合名：未知类型返回空串而不是报错
    assert kb._format_meta_hint("unknown", {"name": "x"}) == ""


def test_stable_pk_deterministic():
    import build_knowledge_base as b
    pk1 = b.stable_pk("dish", "42", 0, "同一段内容")
    pk2 = b.stable_pk("dish", "42", 0, "同一段内容")
    pk3 = b.stable_pk("dish", "42", 1, "同一段内容")
    assert pk1 == pk2
    assert pk1 != pk3
    assert pk1.startswith("dish_")


def test_build_documents_unique_and_metadata():
    import build_knowledge_base as b
    spec = b.BuildSpec("faq", "faq_knowledge.json", RAG_COLLECTION_FAQ)
    docs = b.build_documents(spec)
    assert docs, "FAQ 应构建出文档"
    ids = [d.id for d in docs]
    assert len(ids) == len(set(ids)), "主键必须唯一（幂等构建不产生重复向量）"
    for d in docs:
        assert d.metadata["knowledge_type"] == "faq"
        assert d.metadata["embedding_dim"] == 1024
        assert "content_hash" in d.metadata


def test_embedding_requires_api_key(monkeypatch):
    # 未配置 Key 必须直接报清晰错误
    import embedding_client as ec
    with pytest.raises(ec.EmbeddingAuthError):
        ec.DashScopeTextEmbeddings(api_key="")


def test_embedding_rejects_bad_dimension():
    import embedding_client as ec
    with pytest.raises(ec.EmbeddingConfigError):
        ec.DashScopeTextEmbeddings(api_key="k", dimensions=513)


def test_embedding_parse_restores_order():
    import embedding_client as ec
    emb = ec.DashScopeTextEmbeddings(api_key="k", dimensions=256, enable_cache=False)
    data = {"output": {"embeddings": [
        {"text_index": 1, "embedding": [0.1] * 256},
        {"text_index": 0, "embedding": [0.2] * 256},
    ]}}
    out = emb._parse_embeddings(data, 2)
    assert out[0] == [0.2] * 256  # text_index=0 还原到首位
    assert out[1] == [0.1] * 256


def test_embedding_parse_accepts_openai_style_index():
    """兼容百炼兼容网关在 output.embeddings 中返回 index。"""
    import embedding_client as ec
    emb = ec.DashScopeTextEmbeddings(api_key="k", dimensions=256, enable_cache=False)
    data = {"output": {"embeddings": [
        {"index": 1, "embedding": [0.1] * 256},
        {"index": 0, "embedding": [0.2] * 256},
    ]}}
    out = emb._parse_embeddings(data, 2)
    assert out[0] == [0.2] * 256
    assert out[1] == [0.1] * 256


def test_embedding_parse_validates_count_and_dim():
    import embedding_client as ec
    emb = ec.DashScopeTextEmbeddings(api_key="k", dimensions=256, enable_cache=False)
    bad_count = {"output": {"embeddings": [{"text_index": 0, "embedding": [0.1] * 256}]}}
    with pytest.raises(ec.EmbeddingConfigError):
        emb._parse_embeddings(bad_count, 2)
    bad_dim = {"output": {"embeddings": [{"text_index": 0, "embedding": [0.1] * 128}]}}
    with pytest.raises(ec.EmbeddingConfigError):
        emb._parse_embeddings(bad_dim, 1)


def test_search_logs_rag_pipeline(monkeypatch, caplog):
    monkeypatch.setattr(kb, "_get_vector_store", lambda _collection: object())
    # Keep this a pure dense-path unit test regardless of local .env feature flags.
    monkeypatch.setattr(kb, "RAG_HYBRID_SEARCH_ENABLED", False)
    monkeypatch.setattr(kb, "RAG_RERANK_ENABLED", False)
    monkeypatch.setattr(
        kb,
        "_similarity_search",
        lambda _store, _collection, _query, _top_k, rag_id="-", filter_expr="": [
            (Document(page_content="命中内容", metadata={"name": "测试菜"}), 0.81),
            (Document(page_content="低分内容", metadata={}), 0.10),
        ],
    )

    with caplog.at_level(logging.INFO, logger="knowledge_base"):
        result = kb.search_knowledge(
            "测试 RAG 日志",
            RAG_COLLECTION_FOOD,
            min_score=0.3,
            trace_id="test-log",
        )

    assert "命中内容" in result
    assert "[RAG][test-log] START" in caplog.text
    assert "[RAG][test-log] FILTER mode=DENSE_ONLY candidates=2 accepted=1" in caplog.text
    assert "[RAG][test-log] FINISH mode=DENSE_ONLY accepted=1" in caplog.text
