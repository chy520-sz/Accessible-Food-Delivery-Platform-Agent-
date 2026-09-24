"""
统一工具错误码 —— 所有工具对 LLM / 上层暴露的错误分类标准。

设计原则：
  - 错误码是**机器可读**的稳定标识，不因文案变化而变化；LLM 和测试都可以断言它。
  - 工具返回文本里同时携带 `code=XXX` 片段，便于 LLM 区分"没数据"和"系统挂了"，
    也便于审计/监控聚合。
  - 可重试性是显式标注的：查询类网络抖动可重试，写操作类绝不自动重试（防重复提交）。

错误码一览：
  OK                成功
  AUTH_REQUIRED     未登录或 JWT 过期（对应 PermissionError）
  BAD_PARAM         入参非法/越界（由 LLM 修正后重试，非服务端故障）
  NOT_FOUND         资源不存在（菜品/套餐/订单/地址 ID 无效或已删除）
  CONFLICT          状态冲突（库存不足、订单状态机不允许、购物车为空）
  DRAFT_REQUIRED    高风险操作仅生成了草稿，等待用户确认后再提交
  DRAFT_EXPIRED     草稿已过期（默认 5 分钟），需重新发起
  UPSTREAM_ERROR    Java 后端返回非 200 业务错误（code != 200，对应 RuntimeError）
  NETWORK_ERROR     连接失败/超时/5xx（查询可重试，写操作不重试）
  INTERNAL_ERROR    未预期的异常（兜底）
"""

from __future__ import annotations

from dataclasses import dataclass


# ==================== 错误码常量 ====================
OK = "OK"
AUTH_REQUIRED = "AUTH_REQUIRED"
BAD_PARAM = "BAD_PARAM"
NOT_FOUND = "NOT_FOUND"
CONFLICT = "CONFLICT"
DRAFT_REQUIRED = "DRAFT_REQUIRED"
DRAFT_EXPIRED = "DRAFT_EXPIRED"
UPSTREAM_ERROR = "UPSTREAM_ERROR"
NETWORK_ERROR = "NETWORK_ERROR"
INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass(frozen=True)
class ErrorSpec:
    code: str
    meaning: str          # 一句话含义
    retryable: bool       # LLM/调用方是否可以原样重试
    user_action: str      # 期望 LLM 引导用户做什么


ERRORS: dict[str, ErrorSpec] = {
    OK: ErrorSpec(OK, "成功", False, "直接把结果呈现给用户"),
    AUTH_REQUIRED: ErrorSpec(
        AUTH_REQUIRED, "未登录或登录已过期", False,
        "引导用户先调用 login 登录，再重试原操作",
    ),
    BAD_PARAM: ErrorSpec(
        BAD_PARAM, "入参非法或超出取值范围", False,
        "修正参数后重试（如 score 必须在 1-5、ID 必须是数字）",
    ),
    NOT_FOUND: ErrorSpec(
        NOT_FOUND, "目标资源不存在或已下架", False,
        "提示用户换一个关键词/ID，或先搜索获取最新 ID",
    ),
    CONFLICT: ErrorSpec(
        CONFLICT, "当前状态不允许该操作", False,
        "向用户解释冲突原因（如售罄、订单已完成），不要原样重试",
    ),
    DRAFT_REQUIRED: ErrorSpec(
        DRAFT_REQUIRED, "高风险操作已生成草稿，等待用户确认", False,
        "把草稿内容完整复述给用户，等待用户明确确认后再以相同参数调用一次",
    ),
    DRAFT_EXPIRED: ErrorSpec(
        DRAFT_EXPIRED, "草稿已过期", False,
        "提示用户重新发起该操作，不要复用旧草稿",
    ),
    UPSTREAM_ERROR: ErrorSpec(
        UPSTREAM_ERROR, "Java 后端返回业务错误", False,
        "把后端 message 转达给用户；写操作失败不要自动重试",
    ),
    NETWORK_ERROR: ErrorSpec(
        NETWORK_ERROR, "网络/超时/服务端 5xx", True,
        "查询类可自动重试；写操作类提示用户稍后再试",
    ),
    INTERNAL_ERROR: ErrorSpec(
        INTERNAL_ERROR, "未预期的内部错误", False,
        "记录日志并提示用户稍后再试，不要编造结果",
    ),
}


# ==================== 异常 → 错误码 映射 ====================

def classify(exc: BaseException) -> str:
    """把工具捕获到的异常映射成统一错误码。"""
    if isinstance(exc, PermissionError):
        return AUTH_REQUIRED
    if isinstance(exc, ValueError):
        # 参数校验类（如 health_validator / 数值范围）
        return BAD_PARAM
    if isinstance(exc, RuntimeError):
        # backend_client.extract_data 对 code != 200 抛出
        return UPSTREAM_ERROR
    name = type(exc).__name__
    if name in {"TimeoutException", "ConnectError", "HTTPStatusError",
                "ReadTimeout", "ConnectTimeout"}:
        return NETWORK_ERROR
    # 其余一律兜底
    return INTERNAL_ERROR


def with_code(text: str, code: str) -> str:
    """在 LLM 可读的文本结果尾部附加机器可读错误码。

    形式：`...原文...\n[code=AUTH_REQUIRED]`
    保持原文不变，只在末尾追加，向后兼容旧测试对文本片段的断言。
    """
    if code == OK:
        return text
    return f"{text}\n[code={code}]"
