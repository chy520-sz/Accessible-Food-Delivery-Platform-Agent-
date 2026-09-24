"""Long-term memory policy, retrieval, lifecycle and RAG routing tests."""

import json

import pytest

from long_term_memory import MemoryPolicyError, MemoryService, classify_task


@pytest.fixture
def service(tmp_path):
    return MemoryService(str(tmp_path / "memory.db"), str(tmp_path / "rag-inbox.jsonl"))


def test_types_are_stored_and_same_key_updates(service):
    first = service.write(
        owner_id="u1", session_id="s1", memory_type="user_preference",
        key="spicy_level", content="喜欢微辣", importance=0.7,
    )
    second = service.write(
        owner_id="u1", session_id="s2", memory_type="user_preference",
        key="spicy_level", content="现在喜欢中辣", importance=0.8,
    )
    assert first.action == "created"
    assert second.action == "updated"
    assert first.memory.id == second.memory.id
    assert service.list_memories("u1")[0].content == "现在喜欢中辣"


def test_secrets_are_never_written(service):
    with pytest.raises(MemoryPolicyError, match="禁止写入"):
        service.write(
            owner_id="u1", session_id="s1", memory_type="project_background",
            key="api_key", content="API_KEY=abc123", user_confirmed=True,
        )
    assert service.list_memories("u1") == []

    with pytest.raises(MemoryPolicyError, match="metadata"):
        service.write(
            owner_id="u1", session_id="s1", memory_type="user_preference",
            key="safe", content="普通内容", metadata={"access_token": "secret"},
        )


def test_inferred_and_sensitive_memory_require_confirmation(service):
    with pytest.raises(MemoryPolicyError, match="明确确认"):
        service.write(
            owner_id="u1", session_id="s1", memory_type="user_goal",
            key="career", content="模型猜测用户想找 Java 工作", source="model_inferred",
        )
    with pytest.raises(MemoryPolicyError, match="敏感个人信息"):
        service.write(
            owner_id="u1", session_id="s1", memory_type="project_background",
            key="病史", content="用户有高血压", source="user_explicit",
        )
    saved = service.write(
        owner_id="u1", session_id="s1", memory_type="project_background",
        key="病史", content="用户确认有高血压", source="user_explicit", user_confirmed=True,
    )
    assert saved.memory is not None


def test_task_state_is_separate_and_cleared_with_session(service):
    result = service.write(
        owner_id="u1", session_id="s1", memory_type="task_state",
        key="checkout", content="等待用户确认地址",
    )
    assert result.action == "temporary"
    assert service.list_memories("u1") == []
    context = service.retrieve("u1", "继续下单", session_id="s1")
    assert any(item.memory_type == "task_state" for item in context.memories)
    assert service.clear_session_state("s1") == 1
    context = service.retrieve("u1", "继续下单", session_id="s1")
    assert not any(item.memory_type == "task_state" for item in context.memories)


def test_external_reference_goes_to_rag_inbox(service):
    result = service.write(
        owner_id="u1", session_id="s1", memory_type="external_knowledge_reference",
        key="配送规范", content="https://example.test/delivery", metadata={"topic": "faq"},
    )
    assert result.action == "routed_to_rag"
    assert result.memory is None
    assert service.list_memories("u1") == []
    lines = service.rag_inbox_path.read_text(encoding="utf-8").splitlines()
    doc = json.loads(lines[0])
    assert doc["title"] == "配送规范"
    assert doc["review_status"] == "pending"


def test_retrieval_filters_types_by_current_task_and_tracks_use(service):
    preference = service.write(
        owner_id="u1", session_id="s1", memory_type="user_preference",
        key="口味", content="喜欢清淡", importance=0.9,
    ).memory
    conclusion = service.write(
        owner_id="u1", session_id="s1", memory_type="historical_conclusion",
        key="事故结论", content="上次下单失败是因为购物车为空", source="verified_tool",
    ).memory
    service.write(
        owner_id="u1", session_id="s1", memory_type="project_background",
        key="服务背景", content="Java 后端连接本地数据库",
    )

    food = service.retrieve("u1", "帮我点几道菜", limit=10)
    assert food.task_type == "food_ordering"
    assert preference.id in {item.id for item in food.memories}
    assert all(item.memory_type != "project_background" for item in food.memories)

    incident = service.retrieve("u1", "排查下单失败故障", limit=10)
    assert incident.task_type == "troubleshooting"
    assert conclusion.id in {item.id for item in incident.memories}
    used = {item.id: item for item in service.list_memories("u1")}
    assert used[conclusion.id].access_count >= 1


def test_delete_is_owner_scoped_and_maintenance_expires_task_state(service):
    record = service.write(
        owner_id="u1", session_id="s1", memory_type="user_goal",
        key="goal", content="保持健康",
    ).memory
    assert service.delete("u2", record.id) == 0
    assert service.delete("u1", record.id) == 1
    assert service.list_memories("u1") == []

    service.write(
        owner_id="u1", session_id="s1", memory_type="task_state",
        key="temp", content="临时", task_ttl_seconds=60,
    )
    stats = service.maintain(now=10**12)
    assert stats["expired_task_states"] == 1


def test_superseded_conclusion_is_downgraded_and_not_retrieved(service):
    record = service.write(
        owner_id="u1", session_id="s1", memory_type="historical_conclusion",
        key="root_cause", content="旧结论", source="verified_tool",
        confidence=0.9, importance=0.8,
    ).memory
    assert service.supersede("u2", record.id, "越权") == 0
    assert service.supersede("u1", record.id, "新日志证明旧结论错误") == 1
    all_records = service.list_memories("u1", include_inactive=True)
    assert all_records[0].status == "superseded"
    assert all_records[0].confidence == pytest.approx(0.2)
    assert service.retrieve("u1", "排查故障", limit=10).memories == []


def test_maintenance_auto_deletes_stale_low_value_memory(service):
    result = service.write(
        owner_id="u1", session_id="s1", memory_type="user_preference",
        key="minor", content="低价值偏好", importance=0.1,
    )
    stats = service.maintain(now=result.memory.created_at + 365 * 86400, stale_days=180)
    assert stats["auto_deleted_memories"] == 1
    records = service.list_memories("u1", include_inactive=True)
    assert records[0].id == result.memory.id
    assert records[0].status == "deleted"


@pytest.mark.parametrize(
    ("query", "expected"),
    [("优化我的简历", "writing"), ("排查登录不上", "troubleshooting"),
     ("制定减肥饮食计划", "health_management"), ("今天吃什么菜", "food_ordering")],
)
def test_task_classifier(query, expected):
    assert classify_task(query) == expected
