# -*- coding: utf-8 -*-
"""
问题排查调查板（Investigation Board）。

给 Agent 增加一块结构化的"根因排查工作记忆"，用于处理用户反馈的异常类问题：
下单失败、订单卡住、配送异常、扣款/金额疑问、登录失败等。

为什么需要它：
  默认 AgentState 只有自由文本 messages，Agent 跨多轮排查时容易：
  重复怀疑已排除的原因、忘记已确认的事实、没有明确的下一步。
  调查板把推理过程显式拆成五块，随 LangGraph state 一起持久化（进 checkpointer），
  每轮模型调用前由 InvestigationMiddleware 注入到系统提示。

字段：
  goal                  排查目标（一句话）
  status                investigating（排查中）/ resolved（已定位）
  current_hypothesis    当前最可能的原因
  confirmed_facts       已被工具结果坐实的事实
  rejected_hypotheses   已排除的猜测（防止反复怀疑）
  next_actions          下一步要验证什么
  conclusion            定位后的结论（status=resolved 时填写）
  updated_at            最近一次更新的时间戳
"""

from __future__ import annotations

import time
from typing import Any, Optional

from typing_extensions import TypedDict

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage


# ==================== state schema ====================

class InvestigationBoard(TypedDict):
    """会话内排查状态；与长期用户记忆严格分离。"""
    goal: str
    current_hypothesis: str
    confirmed_facts: list[str]
    rejected_hypotheses: list[str]
    next_actions: list[str]
    status: str
    conclusion: str
    updated_at: float


class InvestigationState(TypedDict, total=False):
    """create_agent 自定义 state 扩展：新增 investigation 字段。"""
    investigation: Optional[InvestigationBoard]


STATE_KEY = "investigation"

VALID_STATUS = ("investigating", "resolved")
MAX_LIST_ITEMS = 12
MAX_ITEM_CHARS = 200


# ==================== 构造 / 规范化 ====================

def _clean_list(items: Optional[list[str]]) -> list[str]:
    """去空、去重（保序）、截断、限长，避免模型塞垃圾或超长内容。"""
    if not items:
        return []
    if not isinstance(items, list):
        raise ValueError("排查事实和行动必须是字符串列表")
    seen, out = set(), []
    for it in items:
        if not isinstance(it, str):
            raise ValueError("排查事实和行动必须是字符串列表")
        text = str(it).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text[:MAX_ITEM_CHARS])
        if len(out) >= MAX_LIST_ITEMS:
            break
    return out


def build_board(
    goal: str,
    current_hypothesis: str = "",
    confirmed_facts: Optional[list[str]] = None,
    rejected_hypotheses: Optional[list[str]] = None,
    next_actions: Optional[list[str]] = None,
    status: str = "investigating",
    conclusion: str = "",
) -> InvestigationBoard:
    """构造一块规范化的调查板。status 非法时回退 investigating。"""
    if status not in VALID_STATUS:
        status = "investigating"
    return {
        "goal": str(goal or "").strip()[:MAX_ITEM_CHARS],
        "status": status,
        "current_hypothesis": str(current_hypothesis or "").strip()[:MAX_ITEM_CHARS],
        "confirmed_facts": _clean_list(confirmed_facts),
        "rejected_hypotheses": _clean_list(rejected_hypotheses),
        "next_actions": _clean_list(next_actions),
        "conclusion": str(conclusion or "").strip()[:500],
        "updated_at": time.time(),
    }


def merge_board(
    current: Optional[InvestigationBoard], *, goal: str,
    current_hypothesis: Optional[str] = None,
    confirmed_facts: Optional[list[str]] = None,
    rejected_hypotheses: Optional[list[str]] = None,
    next_actions: Optional[list[str]] = None,
    status: Optional[str] = None,
    conclusion: Optional[str] = None,
) -> InvestigationBoard:
    """增量更新同一目标；目标变化则开启全新调查，避免旧事实串案。"""
    clean_goal = str(goal or "").strip()[:MAX_ITEM_CHARS]
    if not clean_goal:
        raise ValueError("排查目标不能为空")
    old = current if isinstance(current, dict) and current.get("goal") == clean_goal else {}
    if status is not None and status not in VALID_STATUS:
        raise ValueError("排查状态必须是 investigating 或 resolved")

    def merged_items(field: str, incoming: Optional[list[str]]) -> list[str]:
        previous = old.get(field, []) if old else []
        if incoming is not None and not isinstance(incoming, list):
            raise ValueError("排查事实和行动必须是字符串列表")
        return _clean_list([*previous, *incoming]) if incoming is not None else _clean_list(previous)

    facts = merged_items("confirmed_facts", confirmed_facts)
    rejected = merged_items("rejected_hypotheses", rejected_hypotheses)
    # 如果假设后来被证实或证伪，不再让它同时出现在两个类别中。
    rejected = [item for item in rejected if item not in facts]
    new_status = status or old.get("status", "investigating")
    new_conclusion = conclusion if conclusion is not None else old.get("conclusion", "")
    if new_status == "resolved" and not str(new_conclusion).strip():
        raise ValueError("标记已定位时必须填写结论")
    if new_status == "investigating":
        new_conclusion = ""
    return build_board(
        goal=clean_goal,
        current_hypothesis=(current_hypothesis if current_hypothesis is not None
                            else old.get("current_hypothesis", "")),
        confirmed_facts=facts,
        rejected_hypotheses=rejected,
        next_actions=(next_actions if next_actions is not None else old.get("next_actions", [])),
        status=new_status,
        conclusion=new_conclusion,
    )


# ==================== 渲染给模型 ====================

_STATUS_CN = {"investigating": "排查中", "resolved": "已定位"}


def render_board(board: Optional[dict]) -> str:
    """把调查板渲染成注入系统提示的文本；无调查板返回空串。"""
    if not board:
        return ""

    def _bullets(items: list[str]) -> list[str]:
        return [f"  {i}. {x}" for i, x in enumerate(items, 1)] or ["  （暂无）"]

    lines = [
        "## 当前问题排查进度（调查板）",
        f"- 排查目标：{board.get('goal') or '（未明确）'}",
        f"- 状态：{_STATUS_CN.get(board.get('status'), '排查中')}",
        f"- 当前最可能原因：{board.get('current_hypothesis') or '（尚未形成假设）'}",
        "- 已确认事实：",
        *_bullets(board.get("confirmed_facts", [])),
        "- 已排除的猜测（不要再重复怀疑）：",
        *_bullets(board.get("rejected_hypotheses", [])),
        "- 下一步要验证：",
        *_bullets(board.get("next_actions", [])),
    ]
    if board.get("conclusion"):
        lines.append(f"- 结论：{board['conclusion']}")
    lines.append(
        "已确认事实只能来自用户明确陈述或实际工具结果，不能把猜测当事实。"
        "请严格基于调查板推进：用工具验证 next_actions 中的事项，"
        "把坐实的结果加入 confirmed_facts、被证伪的加入 rejected_hypotheses，"
        "每有进展就调用 update_investigation 更新；定位根因后把 status 置为 resolved 并填写 conclusion。"
    )
    return "\n".join(lines)


# ==================== middleware：每轮注入调查板 ====================

def _with_board(request: Any) -> Any:
    """返回一个把调查板追加进系统提示的新 request；无调查板则原样返回。"""
    try:
        state = getattr(request, "state", None)
        board = state.get(STATE_KEY) if isinstance(state, dict) else None
    except Exception:
        board = None
    block = render_board(board)
    if not block:
        return request

    # 取现有系统提示文本（system_message 优先，回退到 system_prompt 字符串）
    base = ""
    sys_msg = getattr(request, "system_message", None)
    if sys_msg is not None and isinstance(getattr(sys_msg, "content", None), str):
        base = sys_msg.content
    else:
        sys_prompt = getattr(request, "system_prompt", None)
        if isinstance(sys_prompt, str):
            base = sys_prompt
    combined = f"{base}\n\n{block}" if base else block
    return request.override(system_message=SystemMessage(content=combined))


class InvestigationMiddleware(AgentMiddleware):
    """每轮模型调用前，把当前调查板注入系统提示。"""

    state_schema = InvestigationState

    def wrap_model_call(self, request, handler):
        return handler(_with_board(request))

    async def awrap_model_call(self, request, handler):
        return await handler(_with_board(request))
