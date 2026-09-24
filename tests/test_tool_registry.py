"""工具治理新增测试：registry 完整性 + 通用草稿闸门两阶段行为。"""

import pytest

import agent
import tool_registry as reg
from agent import AgentSession, OrderConfirmationGate


# ---------- registry 完整性 ----------

def test_all_tools_have_spec():
    """agent._create_tools 返回的每个工具都必须在 registry 中有元数据。"""
    tools = {t.name for t in agent._create_tools(AgentSession("reg-check"))}
    missing = tools - set(reg.SPECS)
    assert not missing, f"缺少元数据: {missing}"


def test_high_risk_tools_are_draft_required():
    """所有 high_risk() 工具都必须 draft_required=True，且都属于 mutation。"""
    for name, spec in reg.high_risk().items():
        assert spec.kind == "mutation", f"{name} 应归类为 mutation"
        assert spec.draft_required is True, f"{name} 必须走草稿闸门"
        assert "DRAFT_REQUIRED" in spec.errors


def test_queries_are_readonly():
    """所有 query 工具不得声明副作用。"""
    for name, spec in reg.queries().items():
        assert spec.side_effect == "", f"{name} 是查询类，不应有 side_effect"


# ---------- 通用草稿闸门 ----------

def test_gate_evaluate_action_two_phase(monkeypatch):
    gate = OrderConfirmationGate()
    monkeypatch.setattr(agent, "get_session", lambda sid: {"user_id": 7})

    k1, p1 = gate.evaluate_action("s", "clear_cart", ())
    assert p1 == "NEED_CONFIRM"
    k2, p2 = gate.evaluate_action("s", "clear_cart", ())
    assert p2 == "CONFIRMED"
    assert k1 == k2

    gate.store_result(k2, "购物车已清空。")
    _, p3 = gate.evaluate_action("s", "clear_cart", ())
    assert p3 == "ALREADY_DONE"


def test_gate_different_actions_different_keys(monkeypatch):
    gate = OrderConfirmationGate()
    monkeypatch.setattr(agent, "get_session", lambda sid: {"user_id": 7})
    k_clear, _ = gate.evaluate_action("s", "clear_cart", ())
    k_accept, _ = gate.evaluate_action("s", "merchant_accept_order", (1, 9))
    assert k_clear != k_accept


def test_gate_different_params_different_keys(monkeypatch):
    gate = OrderConfirmationGate()
    monkeypatch.setattr(agent, "get_session", lambda sid: {"user_id": 7})
    k1, _ = gate.evaluate_action("s", "merchant_accept_order", (1, 9))
    k2, _ = gate.evaluate_action("s", "merchant_accept_order", (2, 9))
    assert k1 != k2


# ---------- clear_cart 两阶段行为 ----------

def test_clear_cart_first_call_is_draft_only(monkeypatch):
    """第一次调 clear_cart 只生成草稿，绝不真正清空；第二次才执行。"""
    sess = AgentSession("draft-clear")
    called = {"n": 0}

    def fake_clear(session_id):
        called["n"] += 1
        return "购物车已清空。"

    monkeypatch.setattr(agent, "_clear_cart", fake_clear)
    monkeypatch.setattr(agent, "get_session", lambda sid: {"user_id": 7})

    tools = {t.name: t for t in agent._create_tools(sess)}
    first = tools["clear_cart"].invoke({})
    assert called["n"] == 0, "首次调用不得产生副作用"
    assert "DRAFT_REQUIRED" in first
    assert "确认" in first

    second = tools["clear_cart"].invoke({})
    assert called["n"] == 1, "确认后才真正执行"
    assert "已清空" in second

    # 第三次幂等复用，不重复执行
    third = tools["clear_cart"].invoke({})
    assert called["n"] == 1
    assert third == second
