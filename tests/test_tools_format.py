"""tools.py 输出格式化与桥接解析测试（httpx 可用时运行）。"""

import pytest

pytest.importorskip("httpx")

import backend_client as bc
import tools


def test_search_dishes_empty(monkeypatch):
    monkeypatch.setattr(bc, "get", lambda *a, **k: {"code": 200, "data": []})
    out = tools.search_dishes("s", keyword="辣")
    assert "没有找到" in out


def test_search_dishes_formatting(monkeypatch):
    monkeypatch.setattr(
        bc, "get",
        lambda *a, **k: {"code": 200, "data": [
            {"id": 1, "name": "宫保鸡丁", "price": 22, "monthlySales": 100,
             "description": "经典川菜", "stock": 10, "categoryName": "川菜",
             "shopName": "暖心食堂"},
        ]},
    )
    out = tools.search_dishes("s", keyword="宫保")
    assert "宫保鸡丁" in out
    assert "暖心食堂" in out


def test_agent_get_order_status(monkeypatch):
    monkeypatch.setattr(
        bc, "get",
        lambda *a, **k: {"code": 200, "data": {
            "orderNo": "ORD001", "status": "delivering",
            "deliveryStatus": "配送中", "riderAssigned": True,
            "riderPhone": "138****0000", "scheduledDeliveryTime": "2026-08-03T12:00:00",
        }},
    )
    out = tools.agent_get_order_status("s", 1)
    assert "ORD001" in out
    assert "配送中" in out
    assert "138****0000" in out


def test_agent_start_delivery_no_rider(monkeypatch):
    monkeypatch.setattr(
        bc, "post",
        lambda *a, **k: {"code": 200, "data": {
            "orderNo": "ORD002", "status": "delivering",
            "deliveryStatus": "配送中", "message": "订单已进入配送状态，正在等待骑手接单",
        }},
    )
    out = tools.agent_start_delivery("s", 2, 99)
    assert "ORD002" in out
    assert "等待骑手接单" in out
    assert "N/A" not in out


def test_place_order_success(monkeypatch):
    monkeypatch.setattr(
        bc, "post",
        lambda *a, **k: {"code": 200, "data": {"orderNo": "ORD003", "totalPrice": 38.0, "status": "pending"}},
    )
    out = tools.place_order("s", 1, "少辣")
    assert "ORD003" in out
    assert "38.0" in out
