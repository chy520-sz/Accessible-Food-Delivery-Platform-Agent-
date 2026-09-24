from fastapi.testclient import TestClient

import routers.memory as memory_router
from long_term_memory import MemoryService
from main import app


def test_memory_api_write_list_delete(monkeypatch, tmp_path):
    service = MemoryService(str(tmp_path / "memory.db"), str(tmp_path / "rag.jsonl"))
    monkeypatch.setattr(memory_router, "get_memory_service", lambda: service)
    monkeypatch.setattr(memory_router, "get_session", lambda sid: {"user_id": 7} if sid == "s1" else None)
    client = TestClient(app)

    created = client.post("/agent/memories", json={
        "session_id": "s1",
        "memory_type": "user_preference",
        "key": "taste",
        "content": "喜欢微辣",
    })
    assert created.status_code == 200
    memory_id = created.json()["memory"]["id"]

    listed = client.get("/agent/memories", params={"session_id": "s1"})
    assert listed.status_code == 200
    assert listed.json()["items"][0]["content"] == "喜欢微辣"

    deleted = client.delete(f"/agent/memories/{memory_id}", params={"session_id": "s1"})
    assert deleted.status_code == 200
    assert client.get("/agent/memories", params={"session_id": "s1"}).json()["items"] == []


def test_memory_api_requires_authenticated_session(monkeypatch):
    monkeypatch.setattr(memory_router, "get_session", lambda _sid: None)
    response = TestClient(app).get("/agent/memories", params={"session_id": "missing"})
    assert response.status_code == 401


def test_memory_api_policy_rejection(monkeypatch, tmp_path):
    service = MemoryService(str(tmp_path / "memory.db"), str(tmp_path / "rag.jsonl"))
    monkeypatch.setattr(memory_router, "get_memory_service", lambda: service)
    monkeypatch.setattr(memory_router, "get_session", lambda _sid: {"user_id": 7})
    response = TestClient(app).post("/agent/memories", json={
        "session_id": "s1",
        "memory_type": "project_background",
        "key": "API key",
        "content": "abc",
        "user_confirmed": True,
    })
    assert response.status_code == 422
    assert "禁止写入" in response.json()["message"]
