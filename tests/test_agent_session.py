"""agent 会话清理 / 下单确认闸门测试。

必需模块（agent 及其 LangChain 依赖）属于本项目核心依赖，
导入失败必须直接让测试失败，而不是 importorskip/skip 掩盖问题。
"""

import time

import agent
import backend_client as bc
from agent import AgentSession, OrderConfirmationGate


def test_cleanup_sessions():
    s1 = AgentSession("alive")
    s1.last_active = time.time()
    s2 = AgentSession("dead")
    s2.last_active = time.time() - 9999
    agent._sessions["alive"] = s1
    agent._sessions["dead"] = s2
    bc.set_session("alive", {"expires_at": time.time() + 100})
    bc.set_session("dead", {"expires_at": time.time() - 10})

    removed = agent.cleanup_sessions()
    assert removed == 1
    assert "alive" in agent._sessions
    assert "dead" not in agent._sessions
    # JWT 也应被联动清理
    assert bc.get_session("dead") is None

    agent._sessions.clear()
    bc.purge_expired_sessions(time.time() + 9999)


def test_delete_session_clears_jwt_and_history():
    sess = AgentSession("to-delete")
    agent._sessions["to-delete"] = sess
    bc.set_session("to-delete", {"token": "t", "expires_at": time.time() + 100})
    sess.order_gate._pending["x"] = {"ts": time.time()}

    assert agent.delete_session("to-delete") is True
    assert "to-delete" not in agent._sessions
    assert bc.get_session("to-delete") is None
    # 重复删除返回 False，但不抛异常
    assert agent.delete_session("to-delete") is False


def test_order_gate_two_phase_and_idempotent(monkeypatch):
    """第一次只待确认，第二次才提交，重复提交复用同一幂等键。"""
    gate = OrderConfirmationGate()
    monkeypatch.setattr(agent, "get_session", lambda sid: {"user_id": 7})

    key1, phase1 = gate.evaluate("s", 1, "少辣")
    assert phase1 == "NEED_CONFIRM"
    key2, phase2 = gate.evaluate("s", 1, "少辣")
    assert phase2 == "CONFIRMED"
    assert key1 == key2  # 同一确认动作派生稳定幂等键

    gate.store_result(key2, "下单成功！")
    _, phase3 = gate.evaluate("s", 1, "少辣")
    assert phase3 == "ALREADY_DONE"

    # 不同备注应得到不同幂等键，避免错误复用
    other, _ = gate.evaluate("s", 1, "多醋")
    assert other != key1


def test_order_gate_different_user_different_key(monkeypatch):
    gate = OrderConfirmationGate()
    monkeypatch.setattr(agent, "get_session", lambda sid: {"user_id": 7})
    key_a, _ = gate.evaluate("s", 1, "")
    monkeypatch.setattr(agent, "get_session", lambda sid: {"user_id": 8})
    key_b, _ = gate.evaluate("s", 1, "")
    assert key_a != key_b


def test_add_to_cart_invalidates_pending_order(monkeypatch):
    """购物车成功变化后，之前确认过的订单参数不能继续提交。"""
    sess = AgentSession("cart-change")
    sess.order_gate._pending["old-order"] = {"ts": time.time()}
    monkeypatch.setattr(agent, "_add_to_cart", lambda *_args, **_kwargs: "添加成功")

    tools = {item.name: item for item in agent._create_tools(sess)}
    assert tools["add_to_cart"].invoke({"dish_id": 1, "quantity": 2}) == "添加成功"
    assert sess.order_gate._pending == {}


def test_every_dish_search_calls_rag_then_realtime(monkeypatch):
    calls = []

    def fake_rag(query, collection, **kwargs):
        calls.append(("rag", query, collection, kwargs.get("logical_type")))
        return "知识库命中"

    def fake_realtime(session_id, keyword, category_id):
        calls.append(("realtime", session_id, keyword, category_id))
        return "实时菜品命中"

    monkeypatch.setattr(agent, "search_knowledge", fake_rag)
    monkeypatch.setattr(agent, "_search_dishes", fake_realtime)
    tools = {item.name: item for item in agent._create_tools(AgentSession("dish-search"))}

    result = tools["search_dishes"].invoke({"keyword": "牛肉", "category_id": None})

    assert [call[0] for call in calls] == ["rag", "realtime"]
    assert calls[0][1] == "牛肉"
    assert "知识库命中" in result
    assert "实时菜品命中" in result
