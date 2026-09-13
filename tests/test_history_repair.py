"""悬空工具调用修复测试。

背景：请求在工具节点执行期间被超时/中断时，checkpoint 会留下
“带 tool_calls 的 AIMessage 却缺少对应 ToolMessage”的历史。
OpenAI 兼容接口对此直接返回 400，导致该会话后续所有对话都失败。
这里的修复在每轮对话前补齐合成 ToolMessage。
"""

import agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def _ai_with_calls(*call_ids):
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "get_cart", "args": {}, "id": cid, "type": "tool_call"}
            for cid in call_ids
        ],
    )


def test_sanitize_appends_missing_tool_result():
    msgs = [HumanMessage(content="看看购物车"), _ai_with_calls("call_1")]
    sanitized, repairs = agent._sanitize_tool_call_messages(msgs)
    assert repairs == 1
    assert [type(m).__name__ for m in sanitized] == ["HumanMessage", "AIMessage", "ToolMessage"]
    assert sanitized[-1].tool_call_id == "call_1"
    assert sanitized[-1].status == "error"
    # 原始消息对象未被替换
    assert sanitized[0] is msgs[0]
    assert sanitized[1] is msgs[1]


def test_sanitize_keeps_existing_tool_result():
    msgs = [
        HumanMessage(content="看看购物车"),
        _ai_with_calls("call_1"),
        ToolMessage(content="购物车为空", tool_call_id="call_1"),
    ]
    sanitized, repairs = agent._sanitize_tool_call_messages(msgs)
    assert repairs == 0
    assert [m for m in sanitized] == msgs


def test_sanitize_fills_only_missing_call_in_parallel():
    """并行工具调用只补齐缺席项，已有结果保留原样。"""
    msgs = [
        HumanMessage(content="并行调用"),
        _ai_with_calls("call_a", "call_b"),
        ToolMessage(content="A 结果", tool_call_id="call_a"),
    ]
    sanitized, repairs = agent._sanitize_tool_call_messages(msgs)
    assert repairs == 1
    # 保留原顺序后追加缺失的 call_b
    assert sanitized[2].tool_call_id == "call_a"
    assert sanitized[3].tool_call_id == "call_b"


def test_sanitize_ignores_trailing_tool_calls_in_middle():
    """悬空调用夹在中间时也要补齐，且不吞掉后续消息。"""
    msgs = [
        HumanMessage(content="第一轮"),
        _ai_with_calls("call_x"),
        HumanMessage(content="第二轮"),
    ]
    sanitized, repairs = agent._sanitize_tool_call_messages(msgs)
    assert repairs == 1
    assert [type(m).__name__ for m in sanitized] == [
        "HumanMessage", "AIMessage", "ToolMessage", "HumanMessage",
    ]
    # 合成结果紧跟其 AI 消息之后，后续轮次消息不丢
    assert sanitized[2].tool_call_id == "call_x"
    assert sanitized[3] is msgs[2]


def test_sanitize_no_tool_calls_is_noop():
    msgs = [HumanMessage(content="你好"), AIMessage(content="您好")]
    sanitized, repairs = agent._sanitize_tool_call_messages(msgs)
    assert repairs == 0
    assert sanitized == msgs
