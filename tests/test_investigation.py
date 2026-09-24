# -*- coding: utf-8 -*-
"""问题排查调查板测试：构造/渲染、middleware 注入、工具写 state。"""

import pytest

pytest.importorskip("langgraph")

from pydantic import Field
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.types import Command

import agent
from agent import AgentSession
from investigation import (
    InvestigationMiddleware,
    build_board,
    merge_board,
    render_board,
)


# ---------- build_board 规范化 ----------

def test_build_board_normalizes_lists():
    b = build_board(
        goal="定位下单失败",
        confirmed_facts=["购物车为空", "购物车为空", "  ", "地址未选"],
        rejected_hypotheses=["数据库慢查询"],
        next_actions=["查地址"],
        status="weird",  # 非法值回退
    )
    assert b["goal"] == "定位下单失败"
    assert b["confirmed_facts"] == ["购物车为空", "地址未选"]  # 去空去重保序
    assert b["status"] == "investigating"
    assert b["rejected_hypotheses"] == ["数据库慢查询"]
    assert "updated_at" in b


def test_build_board_resolved():
    b = build_board(goal="g", status="resolved", conclusion="购物车为空导致无法下单")
    assert b["status"] == "resolved"
    assert b["conclusion"] == "购物车为空导致无法下单"


def test_payment_investigation_incremental_state():
    board = merge_board(None, goal="定位支付接口变慢原因",
                        current_hypothesis="第三方支付回调超时可能导致接口变慢",
                        confirmed_facts=["22:10 后 /api/pay P95 从 300ms 升到 3.8s"],
                        next_actions=["检查 v2.3.1 是否改动回调重试逻辑"])
    updated = merge_board(board, goal="定位支付接口变慢原因",
                          confirmed_facts=["22:05 发布了 payment-service v2.3.1"],
                          rejected_hypotheses=["数据库慢查询导致接口变慢"],
                          next_actions=["查询第三方支付回调错误码分布"])
    assert updated["current_hypothesis"] == board["current_hypothesis"]
    assert updated["confirmed_facts"] == [
        "22:10 后 /api/pay P95 从 300ms 升到 3.8s",
        "22:05 发布了 payment-service v2.3.1",
    ]
    assert updated["rejected_hypotheses"] == ["数据库慢查询导致接口变慢"]
    assert updated["next_actions"] == ["查询第三方支付回调错误码分布"]
    assert updated["status"] == "investigating"


def test_new_goal_resets_previous_investigation():
    old = merge_board(None, goal="定位支付接口变慢原因", confirmed_facts=["支付 P95 上升"])
    new = merge_board(old, goal="定位登录失败原因", next_actions=["检查认证日志"])
    assert new["confirmed_facts"] == []
    assert new["next_actions"] == ["检查认证日志"]


def test_invalid_state_does_not_accept_string_as_fact_list():
    with pytest.raises(ValueError, match="字符串列表"):
        merge_board(None, goal="排查支付", confirmed_facts="未验证的猜测")
    with pytest.raises(ValueError, match="必须填写结论"):
        merge_board(None, goal="排查支付", status="resolved")


# ---------- render_board ----------

def test_render_board_contains_sections():
    b = build_board(
        goal="定位配送卡住",
        current_hypothesis="商家未接单",
        confirmed_facts=["订单状态 pending"],
        rejected_hypotheses=["骑手问题"],
        next_actions=["查 query_order_status"],
    )
    text = render_board(b)
    for kw in ["排查目标", "定位配送卡住", "当前最可能原因", "商家未接单",
               "已确认事实", "订单状态 pending", "已排除", "骑手问题",
               "下一步", "query_order_status"]:
        assert kw in text


def test_render_empty_board():
    assert render_board(None) == ""
    assert render_board({}) == ""


# ---------- middleware 注入 ----------

class _FakeRequest:
    def __init__(self, state, system_prompt="SYS_PROMPT", system_message=None):
        self.state = state
        self.system_prompt = system_prompt
        self.system_message = system_message

    def override(self, **kwargs):
        return _FakeRequest(
            self.state,
            system_prompt=self.system_prompt,
            system_message=kwargs.get("system_message", self.system_message),
        )


def test_middleware_injects_board():
    req = _FakeRequest({"investigation": build_board(goal="排查登录失败")}, "BASE")
    out = InvestigationMiddleware().wrap_model_call(req, lambda r: r)
    # 返回的是 override 后的新 request，原 request 不变
    assert out is not req
    assert out.system_message.content.startswith("BASE")
    assert "排查登录失败" in out.system_message.content


def test_middleware_no_board_leaves_prompt():
    req = _FakeRequest({}, "BASE")
    out = InvestigationMiddleware().wrap_model_call(req, lambda r: r)
    assert out is req  # 无调查板原样返回
    assert out.system_message is None

    req2 = _FakeRequest({"investigation": None}, "BASE")
    out2 = InvestigationMiddleware().wrap_model_call(req2, lambda r: r)
    assert out2 is req2


# ---------- 工具写 state ----------

def test_update_investigation_tool_returns_command():
    sess = AgentSession("inv-tool")
    tools = {t.name: t for t in agent._create_tools(sess)}
    assert "update_investigation" in tools

    ut = tools["update_investigation"]
    # 注入参数 tool_call_id 不应暴露给 LLM（args / tool_call_schema 已过滤）
    assert "tool_call_id" not in ut.args
    assert "tool_call_id" not in ut.tool_call_schema.model_fields
    assert "state" not in ut.tool_call_schema.model_fields

    out = ut.invoke({
        "name": "update_investigation",
        "type": "tool_call",
        "id": "call_inv_1",
        "args": {
            "goal": "定位下单失败原因",
            "state": {},
            "current_hypothesis": "购物车为空",
            "confirmed_facts": ["get_cart 返回空购物车"],
            "rejected_hypotheses": [],
            "next_actions": ["引导用户先加菜"],
            "status": "resolved",
            "conclusion": "购物车为空，无法下单",
        },
    })
    assert isinstance(out, Command)
    board = out.update["investigation"]
    assert board["goal"] == "定位下单失败原因"
    assert board["status"] == "resolved"
    # 必须回 ToolMessage，避免悬空 tool_call
    msgs = out.update["messages"]
    assert msgs and getattr(msgs[0], "tool_call_id")
    assert "调查板已更新" in msgs[0].content


# ---------- 端到端：调工具 -> 写 state -> 下一轮注入系统提示 ----------

class _ScriptedChatModel(BaseChatModel):
    """按调用次数脚本化返回，并记录每轮收到的系统提示文本。"""
    call_systems: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self):
        return "scripted-investigation-test"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        sys_text = " ".join(
            m.content for m in messages
            if m.__class__.__name__ == "SystemMessage" and isinstance(m.content, str)
        )
        self.call_systems.append(sys_text)
        if len(self.call_systems) == 1:
            ai = AIMessage(content="", tool_calls=[{
                "name": "update_investigation",
                "args": {
                    "goal": "定位下单失败",
                    "confirmed_facts": ["购物车为空"],
                    "status": "resolved",
                    "conclusion": "购物车为空导致无法下单",
                },
                "id": "call_e2e_1",
                "type": "tool_call",
            }])
        else:
            ai = AIMessage(content="已定位：购物车为空，请先加菜后再下单")
        return ChatResult(generations=[ChatGeneration(message=ai)])


def test_agent_end_toend_board_persisted_and_injected():
    sess = AgentSession("inv-e2e")
    tool = {t.name: t for t in agent._create_tools(sess)}["update_investigation"]

    model = _ScriptedChatModel()
    graph = create_agent(
        model=model,
        tools=[tool],
        system_prompt="SYS_BASE",
        middleware=[InvestigationMiddleware()],
    )
    result = graph.invoke({"messages": [HumanMessage(content="我下单失败了")]})

    # state 里写入了调查板
    board = result["investigation"]
    assert board["goal"] == "定位下单失败"
    assert board["status"] == "resolved"
    assert board["confirmed_facts"] == ["购物车为空"]
    # 第二轮系统提示已注入调查板
    assert len(model.call_systems) >= 2
    assert "定位下单失败" in model.call_systems[1]
    # 工具结果消息存在，无悬空 tool_call
    kinds = [m.__class__.__name__ for m in result["messages"]]
    assert "ToolMessage" in kinds
