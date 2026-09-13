"""历史状态更新测试：清空与裁剪必须真正生效。

回归背景：create_agent 图里 model/tools 节点都会写 messages，
aupdate_state 不带 as_node 会抛 "Ambiguous update"；
旧实现用 except+debug 吞掉了异常，导致换号隔离历史与长对话裁剪静默失效。
这些测试需要真实 LLM 图结构，但不调用大模型。
"""

import asyncio

import agent
from langchain_core.messages import AIMessage, HumanMessage

from config import AGENT_MESSAGE_WINDOW


def _run(coro):
    return asyncio.run(coro)


def test_clear_conversation_empties_history():
    async def scenario():
        sess = agent.get_or_create_session(None)
        sess.logged_in = True
        ag = sess.ensure_agent()
        cfg = sess.config()
        await ag.aupdate_state(cfg, {"messages": [
            HumanMessage(content="h1", id="h1"),
            AIMessage(content="a1", id="a1"),
        ]})
        assert len((await ag.aget_state(cfg)).values["messages"]) == 2

        await agent._clear_conversation(sess)
        assert len((await ag.aget_state(cfg)).values["messages"]) == 0
        agent.delete_session(sess.session_id)

    _run(scenario())


def test_trim_history_respects_window():
    async def scenario():
        sess = agent.get_or_create_session(None)
        sess.logged_in = True
        ag = sess.ensure_agent()
        cfg = sess.config()
        total = AGENT_MESSAGE_WINDOW + 10
        msgs = [HumanMessage(content=f"h{i}", id=f"h{i}") for i in range(total)]
        await ag.aupdate_state(cfg, {"messages": msgs}, as_node="model")

        await agent._trim_history(sess)
        remaining = (await ag.aget_state(cfg)).values["messages"]
        # 裁剪到窗口以内，且剩余的必须是最新的消息
        assert len(remaining) <= AGENT_MESSAGE_WINDOW
        assert remaining[-1].id == f"h{total - 1}"
        agent.delete_session(sess.session_id)

    _run(scenario())
