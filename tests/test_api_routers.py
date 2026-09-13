"""APIRouter 拆分与流式接口测试。

覆盖：
  - 路由分组后原有路径/方法保持兼容
  - /agent/text 请求/响应字段不变
  - /agent/text/stream 产出标准 SSE 事件序列
  - /agent/sync 登录态同步响应结构
测试通过 monkeypatch 路由模块内的 agent 函数，不触达真实 LLM。
"""

import main
import routers.chat as chat_router
import routers.health as health_router
import routers.session as session_router
from fastapi.testclient import TestClient

client = TestClient(main.app)


def _route_map():
    routes = {}
    for r in main.app.routes:
        methods = getattr(r, "methods", None)
        if methods:
            routes[r.path] = set(methods)
    return routes


def test_original_endpoints_preserved():
    routes = _route_map()
    assert "POST" in routes["/agent/text"]
    assert "POST" in routes["/agent/tts"]
    assert "POST" in routes["/agent/voice"]
    assert "POST" in routes["/agent/sync"]
    assert "DELETE" in routes["/agent/session/{session_id}"]
    assert "GET" in routes["/agent/health"]
    # 新增流式端点
    assert "POST" in routes["/agent/text/stream"]


def test_text_endpoint_response_contract(monkeypatch):
    async def fake_chat(session_id, text, auth_token=None):
        return {
            "session_id": "sess-1",
            "reply": "好的呀",
            "tts_text": "好的呀",
            "is_new_session": True,
            "trace_id": "trace-1",
        }

    monkeypatch.setattr(chat_router, "chat", fake_chat)
    resp = client.post("/agent/text", json={"session_id": None, "text": "你好"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {
        "code", "session_id", "reply", "tts_text", "is_new_session", "trace_id",
    }
    assert body["code"] == 200
    assert body["session_id"] == "sess-1"
    assert body["reply"] == "好的呀"
    assert body["is_new_session"] is True


def test_text_endpoint_rejects_blank(monkeypatch):
    async def fake_chat(session_id, text, auth_token=None):  # pragma: no cover
        raise AssertionError("空文本不应进入 agent")

    monkeypatch.setattr(chat_router, "chat", fake_chat)
    resp = client.post("/agent/text", json={"session_id": None, "text": "   "})
    assert resp.status_code == 400


def test_text_stream_sse_sequence(monkeypatch):
    async def fake_stream(session_id, user_input, auth_token=None):
        yield {"type": "session", "session_id": "s1", "is_new_session": True, "trace_id": "t1"}
        yield {"type": "delta", "text": "你"}
        yield {"type": "delta", "text": "好"}
        yield {
            "type": "done",
            "session_id": "s1",
            "reply": "你好",
            "tts_text": "你好",
            "is_new_session": True,
            "trace_id": "t1",
        }

    monkeypatch.setattr(chat_router, "stream_chat", fake_stream)
    resp = client.post("/agent/text/stream", json={"session_id": None, "text": "打招呼"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert "event: session" in body
    assert "event: delta" in body
    assert "event: done" in body
    assert '"text": "你"' in body and '"text": "好"' in body
    assert '"reply": "你好"' in body


def test_text_stream_error_event(monkeypatch):
    async def fake_stream(session_id, user_input, auth_token=None):
        yield {"type": "session", "session_id": "s2", "is_new_session": True, "trace_id": "t2"}
        yield {"type": "error", "message": "出错了", "session_id": "s2", "trace_id": "t2"}

    monkeypatch.setattr(chat_router, "stream_chat", fake_stream)
    resp = client.post("/agent/text/stream", json={"session_id": None, "text": "hi"})
    assert resp.status_code == 200
    assert "event: error" in resp.text
    assert "出错了" in resp.text


def test_sync_endpoint_contract(monkeypatch):
    class FakeSession:
        session_id = "sync-1"

    monkeypatch.setattr(session_router, "get_or_create_session", lambda sid: FakeSession())

    async def fake_sync(session_id, auth_token):
        return True, "小明"

    monkeypatch.setattr(session_router, "sync_login_token", fake_sync)

    resp = client.post("/agent/sync", json={"session_id": None, "auth_token": "tok"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "sync-1"
    assert "小明" in body["reply"]


def test_health_endpoint_contract(monkeypatch):
    async def fake_status():
        return {
            "status": "ok",
            "model": "deepseek-chat",
            "active_sessions": 0,
            "llm_connected": True,
            "java_backend": "up",
            "milvus_up": True,
            "rag_ready": True,
            "embedding_ok": True,
            "rag_collections": {"dish": 10},
        }

    monkeypatch.setattr(health_router, "get_agent_status", fake_status)
    resp = client.get("/agent/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
