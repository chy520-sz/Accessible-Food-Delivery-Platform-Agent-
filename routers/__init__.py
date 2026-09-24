"""路由分组包 —— 用 APIRouter 把 Agent 接口拆成独立小模块。

每个子模块只负责一类职责：
  - chat    : 文字 / 流式文字 / 语音合成 / 语音对话
  - session : 登录态同步 / 会话删除
  - health  : 健康检查

main.py 只需 include 这里汇总的 api_router，不再堆积路由实现。
"""

from fastapi import APIRouter

from routers import chat, health, memory, session

api_router = APIRouter()
api_router.include_router(chat.router)
api_router.include_router(session.router)
api_router.include_router(health.router)
api_router.include_router(memory.router)

__all__ = ["api_router"]
