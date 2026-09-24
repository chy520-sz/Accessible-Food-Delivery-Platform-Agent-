"""Inject only task-relevant user memories into each model call."""

from __future__ import annotations

from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage

from long_term_memory import MemoryService, RetrievalContext


def render_memory_context(context: RetrievalContext) -> str:
    if not context.memories:
        return ""
    lines = [
        "## 与当前任务相关的已保存记忆",
        f"- 当前任务类型：{context.task_type}",
        "以下内容可能过期；若与用户本轮表达冲突，以本轮表达为准，并更新旧记忆。",
    ]
    for rec in context.memories:
        lines.append(f"- [id={rec.id}][{rec.memory_type}][{rec.memory_key}] {rec.content}")
    lines.append("不要向用户声称记忆绝对正确，也不要泄露其他用户或其他会话的内容。")
    return "\n".join(lines)


class LongTermMemoryMiddleware(AgentMiddleware):
    def __init__(self, service: MemoryService, owner_resolver: Callable[[], str],
                 session_id: str, top_k: int = 6):
        self.service = service
        self.owner_resolver = owner_resolver
        self.session_id = session_id
        self.top_k = top_k

    @staticmethod
    def _latest_user_text(request: Any) -> str:
        messages = getattr(request, "messages", None) or []
        for message in reversed(messages):
            if getattr(message, "type", "") == "human":
                content = getattr(message, "content", "")
                return content if isinstance(content, str) else str(content)
        return ""

    def _with_memories(self, request: Any) -> Any:
        owner_id = self.owner_resolver()
        query = self._latest_user_text(request)
        if not owner_id or not query:
            return request
        context = self.service.retrieve(owner_id, query, self.session_id, self.top_k)
        block = render_memory_context(context)
        if not block:
            return request
        system_message = getattr(request, "system_message", None)
        base = getattr(system_message, "content", "") if system_message else ""
        if not isinstance(base, str):
            base = ""
        return request.override(system_message=SystemMessage(content=f"{base}\n\n{block}".strip()))

    def wrap_model_call(self, request, handler):
        return handler(self._with_memories(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._with_memories(request))

