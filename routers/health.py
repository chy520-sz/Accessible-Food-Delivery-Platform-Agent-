"""健康检查路由 —— 无需服务密钥，供前端与运维监控调用。"""

from fastapi import APIRouter

from agent import get_agent_status
from routers.schemas import HealthResponse

router = APIRouter(tags=["运维"])


@router.get("/agent/health", response_model=HealthResponse)
async def agent_health():
    """健康检查接口。

    返回 Agent 服务的运行状态、模型连接情况、活跃会话数等。
    供 Java 前端或运维监控调用。

    响应示例:
      { "status": "ok", "model": "deepseek-chat", "active_sessions": 3, ... }
    """
    status = await get_agent_status()
    return HealthResponse(**status)
