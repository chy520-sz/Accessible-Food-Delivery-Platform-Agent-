"""RAG 抗阻塞测试：Milvus 调用限时 + 失败快速熔断。

不连接真实 Milvus；用假 client 模拟“集合卡在 Loading、load_collection 一直不返回”。
"""

import pytest

import knowledge_base as kb
from config import RAG_COLLECTION_FOOD


class _FakeClient:
    """模拟卡死的 MilvusClient：记录收到的 timeout，可选择直接抛错。"""

    def __init__(self, *, load_raises=None, search_raises=None, rows=None):
        self.load_calls = []
        self.search_calls = []
        self._load_raises = load_raises
        self._search_raises = search_raises
        self._rows = rows if rows is not None else []

    def has_collection(self, name, timeout=None):
        return True

    def get_collection_stats(self, name, timeout=None):
        return {"row_count": 3}

    def load_collection(self, name, timeout=None):
        self.load_calls.append(timeout)
        if self._load_raises is not None:
            raise self._load_raises

    def search(self, **kwargs):
        self.search_calls.append(kwargs.get("timeout"))
        if self._search_raises is not None:
            raise self._search_raises
        return self._rows


@pytest.fixture(autouse=True)
def _clean_state():
    """每个用例前后清空集合缓存与熔断状态。"""
    kb.reset_store_cache()
    yield
    kb.reset_store_cache()


def test_load_collection_passes_timeout(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(kb, "get_milvus_client", lambda: fake)
    # 直接走加载路径
    store = kb._get_vector_store(RAG_COLLECTION_FOOD)
    assert store is fake
    assert fake.load_calls == [kb.RAG_CALL_TIMEOUT]


def test_load_timeout_raises_and_does_not_cache(monkeypatch):
    fake = _FakeClient(load_raises=TimeoutError("load timed out"))
    monkeypatch.setattr(kb, "get_milvus_client", lambda: fake)

    with pytest.raises(kb.KnowledgeStoreError):
        kb._get_vector_store(RAG_COLLECTION_FOOD)
    # 失败不得写入缓存，否则后续请求会误以为集合可用
    assert RAG_COLLECTION_FOOD not in kb._store_cache


def test_breaker_fails_fast_after_failure(monkeypatch):
    """首次检索失败后，冷却期内第二次应立即抛错而不再次触达 Milvus。"""
    fake = _FakeClient(load_raises=TimeoutError("load timed out"))
    monkeypatch.setattr(kb, "get_milvus_client", lambda: fake)

    # 第一次：真实失败路径触发熔断
    with pytest.raises(kb.KnowledgeStoreError):
        kb.search_knowledge("鸡蛋面", RAG_COLLECTION_FOOD, logical_type="dish")
    calls_before = len(fake.load_calls)
    assert calls_before > 0

    # 第二次：冷却期内快速失败，不再触达 Milvus
    with pytest.raises(kb.KnowledgeStoreError, match="冷却"):
        kb.search_knowledge("鸡蛋面", RAG_COLLECTION_FOOD, logical_type="dish")
    assert len(fake.load_calls) == calls_before


def test_breaker_recovers_after_success(monkeypatch):
    kb._breaker_record_failure(RAG_COLLECTION_FOOD)
    assert kb._breaker_blocked(RAG_COLLECTION_FOOD) is not None
    kb._breaker_record_success(RAG_COLLECTION_FOOD)
    assert kb._breaker_blocked(RAG_COLLECTION_FOOD) is None


def test_not_built_does_not_trip_breaker(monkeypatch):
    """集合未构建属于配置问题，不应进入冷却（构建后要能立即恢复）。"""
    monkeypatch.setattr(kb, "_get_vector_store", lambda name: None)
    with pytest.raises(kb.KnowledgeNotBuilt):
        kb.search_knowledge("鸡蛋面", RAG_COLLECTION_FOOD, logical_type="dish")
    assert kb._breaker_blocked(RAG_COLLECTION_FOOD) is None


def test_search_passes_timeout(monkeypatch):
    fake = _FakeClient(rows=[[{"id": 1, "distance": 0.9, "entity": {"text": "鸡蛋面"}}]])
    monkeypatch.setattr(kb, "get_milvus_client", lambda: fake)
    monkeypatch.setattr(kb, "_get_vector_store", lambda name: fake)
    # 跳过真实嵌入调用
    monkeypatch.setattr(kb, "get_embeddings", lambda: type(
        "E", (), {"embed_query": staticmethod(lambda q: [0.0] * 1024)}
    )())

    out = kb.search_knowledge("鸡蛋面", RAG_COLLECTION_FOOD, logical_type="dish")
    assert fake.search_calls == [kb.RAG_CALL_TIMEOUT]
    assert "鸡蛋面" in out
