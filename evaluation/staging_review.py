"""Run selected evaluation prompts with the real LLM and *mocked* Agent tools.

All tool calls are intercepted before backend/RAG access. This is deliberately
not a production replay and must not be used to claim backend integration works.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from langchain_core.tools import StructuredTool

import agent

_CASE: dict = {}
_CALLS: list[dict] = []


def _fixture_result(name: str) -> str:
    case_id = _CASE["id"]
    if name in {
        "place_order", "clear_cart", "merchant_accept_order", "delivery_pickup_order",
        "delivery_complete_order", "add_to_cart", "add_combo_to_cart", "rate_item",
        "record_health_profile", "record_weight", "generate_diet_plan", "remember_context",
        "forget_context", "supersede_conclusion",
    }:
        return "隔离评测环境：此写操作未执行。 [code=DRAFT_REQUIRED]"
    if name == "update_investigation":
        return "隔离评测环境：已模拟更新调查状态，未写入真实数据。 [code=OK]"
    if case_id == "order_timeout" and name == "search_dishes":
        return "菜单查询超时，请稍后再试。 [code=NETWORK_ERROR]"
    if case_id == "health_missing_profile" and name == "check_health_profile":
        return "用户尚未录入健康档案。 [code=NOT_FOUND]"
    if case_id == "health_tool_unavailable" and name == "check_health_profile":
        return "健康档案接口暂不可用。 [code=NETWORK_ERROR]"
    if case_id == "dish_not_found" and name == "get_dish_detail":
        return "未找到 99 号菜品，可能已下架或编号过期。 [code=NOT_FOUND]"
    if case_id == "delivery_status_no_data" and name == "query_order_status":
        return "未找到可查询的订单配送状态。 [code=NOT_FOUND]"
    if case_id == "rag_unavailable" and name == "search_food_knowledge":
        return "知识库当前不可用。 [code=NETWORK_ERROR]"
    if case_id == "rag_no_match" and name == "search_food_knowledge":
        return "知识库中没有匹配该菜名的来源资料。 [code=NOT_FOUND]"
    if case_id == "cart_backend_5xx" and name == "get_cart":
        return "购物车服务暂不可用，无法核实价格。 [code=UPSTREAM_ERROR]"
    if case_id == "delivery_missing_order" and name == "get_user_orders":
        return "最近订单：订单号 TEST-001，状态配送中。 [code=OK]"
    if case_id == "order_history" and name == "get_user_orders":
        return "上次订单 TEST-001 已完成：宫保鸡丁 1 份、米饭 1 份。 [code=OK]"
    if case_id == "false_tool_success" and name == "get_cart":
        return "加购成功，购物车总价 31 元。 [code=UPSTREAM_ERROR]"
    if case_id == "irrelevant_faq_noise" and name == "search_faq":
        return "配送费由距离及活动决定，结算页显示最终金额。无关段落：退款需联系商家；健康档案要先录入。 [code=OK]"
    if case_id == "rag_stale_menu" and name == "search_food_knowledge":
        return "旧知识库记录：宫保鸡丁可点，但该记录可能过期。 [code=OK]"
    fixtures = {
        "get_cart": "购物车有宫保鸡丁 1 份，价格 28 元；米饭 1 份，价格 3 元；合计 31 元。 [code=OK]",
        "get_user_addresses": "当前用户有一个地址：地址 ID 1，测试地址 A。 [code=OK]",
        "get_user_orders": "最近订单：订单号 TEST-001，状态已完成。 [code=OK]",
        "search_dishes": "实时在售：宫保鸡丁，价格 28 元，口味微辣，店铺测试餐厅；清蒸鱼，价格 38 元，口味清淡，店铺测试餐厅。 [code=OK]",
        "search_combos": "实时在售：双人套餐，价格 68 元，含两道菜和米饭。 [code=OK]",
        "search_faq": "平台规则：如需退款，请联系商家或客服核实订单状态；本评测没有退款执行接口。 [code=OK]",
        "search_dietary_knowledge": "低盐饮食建议：减少高钠调味料，优先选择清蒸或水煮菜品。仅供一般参考。 [code=OK]",
        "check_health_profile": "测试用户健康档案：饮食偏好均衡，目标维持健康。 [code=OK]",
        "list_shops": "测试餐厅，店铺 ID 1。 [code=OK]",
        "search_dishes_by_shop": "测试餐厅在售：宫保鸡丁，价格 28 元。 [code=OK]",
        "search_food_knowledge": "菜品知识：请以可靠来源核验配料或来源。 [code=OK]",
        "query_order_status": "订单 TEST-001 当前状态：配送中；无预计送达时间。 [code=OK]",
        "get_shop_status": "测试餐厅营业中。 [code=OK]",
        "get_dish_detail": "菜品：宫保鸡丁，价格 28 元，库存 9 份。 [code=OK]",
        "list_categories": "分类：家常菜 ID 1；主食 ID 2。 [code=OK]",
    }
    return fixtures.get(name, "隔离评测环境未配置该查询工具的结果。 [code=NOT_FOUND]")


def _proxy_tool(original):
    def mocked(**kwargs):
        _CALLS.append({"tool": original.name, "argument_names": sorted(kwargs)})
        return _fixture_result(original.name)

    return StructuredTool.from_function(
        name=original.name,
        description=original.description,
        args_schema=original.tool_call_schema,
        func=mocked,
    )


async def _fake_auth(session, _token):
    if _CASE.get("id") in {"order_auth_failure", "address_missing_login"}:
        session.logged_in = False
        session.username = None
        session.owner_user_id = None
        return
    session.logged_in = True
    session.username = "隔离评测用户"
    session.owner_user_id = -1


async def run(
    case_ids: list[str], trace_path: Path, *, allow_external_model: bool = False,
    responses_path: Path | None = None,
) -> list[dict]:
    if not allow_external_model:
        raise PermissionError(
            "Real-model evaluation sends prompts, tool definitions, and case inputs "
            "to the configured model provider; explicit authorization is required."
        )
    cases_path = Path(__file__).with_name("cases.jsonl")
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines()]
    selected = [case for case in cases if case["id"] in case_ids]
    if len(selected) != len(set(case_ids)):
        missing = set(case_ids) - {case["id"] for case in selected}
        raise ValueError(f"Unknown case IDs: {sorted(missing)}")

    original_creator = agent._create_tools
    originals = original_creator(agent.AgentSession("staging-tool-spec"))
    proxies = [_proxy_tool(tool) for tool in originals]
    original_auth = agent._reconcile_auth
    old_trace_path, old_trace_enabled = agent.EVAL_TRACE_PATH, agent.EVAL_TRACE_ENABLED
    agent._create_tools = lambda _session: proxies
    agent._reconcile_auth = _fake_auth
    agent.EVAL_TRACE_PATH = str(trace_path)
    agent.EVAL_TRACE_ENABLED = True
    results = []
    try:
        for case in selected:
            _CASE.clear()
            _CASE.update(case)
            _CALLS.clear()
            try:
                response = await agent.chat(None, case["input"])
                result = {
                    "case_id": case["id"], "trace_id": response["trace_id"],
                    "reply": response["reply"], "tool_calls": list(_CALLS),
                }
                results.append(result)
                agent.delete_session(response["session_id"])
            except Exception as exc:
                result = {"case_id": case["id"], "error_type": type(exc).__name__}
                results.append(result)
            if responses_path is not None:
                responses_path.parent.mkdir(parents=True, exist_ok=True)
                with responses_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    finally:
        agent._create_tools = original_creator
        agent._reconcile_auth = original_auth
        agent.EVAL_TRACE_PATH = old_trace_path
        agent.EVAL_TRACE_ENABLED = old_trace_enabled
        await agent.shutdown()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Real LLM, mocked tools, no production writes")
    parser.add_argument("case_ids", nargs="+", help="Selected IDs from cases.jsonl")
    parser.add_argument("--traces", type=Path, default=Path("data/eval_staging_traces.jsonl"))
    parser.add_argument("--responses", type=Path, default=Path("data/eval_staging_responses.jsonl"))
    parser.add_argument(
        "--allow-external-model", action="store_true",
        help="Explicitly authorize sending project prompts, tool definitions, and test inputs to the configured model provider",
    )
    args = parser.parse_args()
    if not args.allow_external_model:
        parser.error("external model call requires --allow-external-model and user authorization")
    logging.getLogger("agent").setLevel(logging.WARNING)
    results = asyncio.run(run(
        args.case_ids, args.traces, allow_external_model=True, responses_path=args.responses,
    ))
    for result in results:
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
