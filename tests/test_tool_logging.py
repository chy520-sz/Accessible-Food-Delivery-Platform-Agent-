# -*- coding: utf-8 -*-
"""失败工具日志：脱敏与失败判定辅助函数测试。"""

from langchain_core.messages import ToolMessage

import agent


def _tm(content, is_error=False):
    return ToolMessage(content=content, tool_call_id="x", is_error=is_error)


# ---------- 脱敏 ----------

def test_mask_sensitive_fields():
    masked = agent._mask_tool_args({"dish_name": "水煮鱼", "password": "123456", "phone": "13800000000"})
    assert masked["dish_name"] == "水煮鱼"
    assert masked["password"] == "***"
    assert masked["phone"] == "***"


def test_mask_token_and_secret():
    masked = agent._mask_tool_args({"token": "abc.def", "api_secret": "s3", "remark": "少辣"})
    assert masked["token"] == "***"
    assert masked["api_secret"] == "***"
    assert masked["remark"] == "少辣"


def test_mask_non_dict():
    assert "_raw" in agent._mask_tool_args("not-a-dict")


# ---------- 失败判定 ----------

def test_exception_tool_message_is_failure():
    assert agent._tool_failure_code(_tm("boom", is_error=True)) == "EXCEPTION"


def test_exception_prefers_code():
    msg = _tm("下单失败 [code=NETWORK_ERROR]", is_error=True)
    assert agent._tool_failure_code(msg) == "NETWORK_ERROR"


def test_code_in_text_is_failure():
    assert agent._tool_failure_code(_tm("购物车为空 [code=CONFLICT]")) == "CONFLICT"


def test_code_ok_is_not_failure():
    assert agent._tool_failure_code(_tm("操作成功 [code=OK]")) == ""


def test_plain_success_text_is_not_failure():
    assert agent._tool_failure_code(_tm("已加入购物车，共 2 道菜")) == ""
