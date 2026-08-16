"""agent 会话清理测试（langchain 可用时运行）。"""

import time

import pytest

langchain = pytest.importorskip("langchain")

try:
    import agent
    import backend_client as bc
except Exception as exc:  # 本地/CI 依赖不完整时优雅跳过
    pytest.skip(f"依赖不可用，跳过 agent 会话测试: {exc}", allow_module_level=True)


def test_cleanup_sessions(monkeypatch):
    class FakeSession:
        def __init__(self):
            self.last_active = time.time()
            self.agent_executor = object()

    s1 = FakeSession()
    s2 = FakeSession()
    s2.last_active = time.time() - 9999
    agent._sessions["alive"] = s1
    agent._sessions["dead"] = s2
    bc.set_session("alive", {"expires_at": time.time() + 100})
    bc.set_session("dead", {"expires_at": time.time() - 10})

    removed = agent.cleanup_sessions()
    assert removed == 1
    assert "alive" in agent._sessions
    assert "dead" not in agent._sessions

    # 清理测试残留
    agent._sessions.clear()
    bc.purge_expired_sessions(time.time() + 9999)
