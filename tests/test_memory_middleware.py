from langchain_core.messages import HumanMessage, SystemMessage

from long_term_memory import MemoryService
from memory_middleware import LongTermMemoryMiddleware, render_memory_context


class _Request:
    def __init__(self, messages, system_message=None):
        self.messages = messages
        self.system_message = system_message

    def override(self, **kwargs):
        return _Request(self.messages, kwargs.get("system_message", self.system_message))


def test_middleware_injects_only_current_owner_and_relevant_type(tmp_path):
    service = MemoryService(str(tmp_path / "m.db"), str(tmp_path / "rag.jsonl"))
    service.write(owner_id="u1", session_id="s", memory_type="user_preference",
                  key="口味", content="喜欢微辣")
    service.write(owner_id="u1", session_id="s", memory_type="project_background",
                  key="项目", content="这是内部项目")
    service.write(owner_id="u2", session_id="s", memory_type="user_preference",
                  key="口味", content="不吃辣")
    middleware = LongTermMemoryMiddleware(service, lambda: "u1", "s", top_k=10)
    req = _Request([HumanMessage(content="帮我点餐")], SystemMessage(content="BASE"))

    out = middleware.wrap_model_call(req, lambda value: value)

    assert out is not req
    assert out.system_message.content.startswith("BASE")
    assert "喜欢微辣" in out.system_message.content
    assert "这是内部项目" not in out.system_message.content
    assert "不吃辣" not in out.system_message.content


def test_no_identity_means_no_injection(tmp_path):
    service = MemoryService(str(tmp_path / "m.db"), str(tmp_path / "rag.jsonl"))
    middleware = LongTermMemoryMiddleware(service, lambda: "", "s")
    req = _Request([HumanMessage(content="你好")], SystemMessage(content="BASE"))
    assert middleware.wrap_model_call(req, lambda value: value) is req


def test_render_empty_context(tmp_path):
    service = MemoryService(str(tmp_path / "m.db"), str(tmp_path / "rag.jsonl"))
    context = service.retrieve("missing", "你好")
    assert render_memory_context(context) == ""

