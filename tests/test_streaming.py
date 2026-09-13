"""流式对话聚合与消息文本提取测试（不触达真实 LLM）。"""

import asyncio

import agent
import llm_client


def test_chat_aggregates_stream_deltas(monkeypatch):
    async def fake_stream(session_id, user_input, auth_token=None):
        yield {"type": "session", "session_id": "s", "is_new_session": True, "trace_id": "t"}
        yield {"type": "delta", "text": "你"}
        yield {"type": "delta", "text": "好"}
        yield {
            "type": "done",
            "session_id": "s",
            "reply": "你好",
            "tts_text": "你好",
            "is_new_session": True,
            "trace_id": "t",
        }

    monkeypatch.setattr(agent, "stream_chat", fake_stream)
    result = asyncio.run(agent.chat(None, "hi"))
    assert result["reply"] == "你好"
    assert result["tts_text"] == "你好"
    assert result["session_id"] == "s"
    assert result["is_new_session"] is True
    assert result["trace_id"] == "t"


def test_chat_maps_stream_error_to_reply(monkeypatch):
    async def fake_stream(session_id, user_input, auth_token=None):
        yield {"type": "session", "session_id": "s", "is_new_session": False, "trace_id": "t"}
        yield {"type": "error", "message": "服务繁忙", "session_id": "s", "trace_id": "t"}

    monkeypatch.setattr(agent, "stream_chat", fake_stream)
    result = asyncio.run(agent.chat("s", "hi"))
    assert result["reply"] == "服务繁忙"
    assert result["tts_text"]  # 兜底 TTS 文本非空


def test_extract_chunk_text_skips_tool_messages():
    class ToolChunk:
        type = "tool"
        content = "工具结果不应被播报"

    class AiChunk:
        type = "AIMessageChunk"
        content = "你好"

    assert agent._extract_chunk_text(ToolChunk()) == ""
    assert agent._extract_chunk_text(AiChunk()) == "你好"


def test_content_to_text_handles_blocks():
    assert agent._content_to_text("直接文本") == "直接文本"
    assert agent._content_to_text([{"type": "text", "text": "a"}, {"type": "tool_use", "id": "x"}]) == "a"


def test_llm_is_configured_for_streaming():
    """共享 LLM 必须开启 streaming，保证 astream 逐步输出。"""
    import inspect

    src = inspect.getsource(llm_client.get_llm)
    assert "streaming=True" in src
