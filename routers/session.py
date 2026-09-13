"""会话相关路由 —— 登录态同步与会话删除。"""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from agent import delete_session, get_or_create_session, sync_login_token
from routers.deps import require_service_key
from routers.schemas import SyncRequest, TextResponse

router = APIRouter(tags=["会话"])


@router.post("/agent/sync", response_model=TextResponse, dependencies=[Depends(require_service_key)])
async def agent_sync(req: SyncRequest):
    """登录态同步接口。

    前端在登录/登出后调用此接口，让 Agent 会话同步前端登录态。
    这样就无需用户在与 Agent 对话时手动登录。

    请求示例:
      { "session_id": null, "auth_token": "eyJhbG..." }
      { "session_id": "abc123", "auth_token": null }

    响应示例:
      { "code": 200, "session_id": "abc123", "reply": "登录态已同步", "is_new_session": false }
    """
    sess = get_or_create_session(req.session_id)
    logged_in, username = await sync_login_token(sess.session_id, req.auth_token)
    if req.auth_token and logged_in:
        msg = f"已同步登录态，欢迎 {username}"
    elif req.auth_token:
        msg = "登录态同步失败，请检查 Token 是否有效"
    else:
        msg = "已清除登录态"
    return TextResponse(
        session_id=sess.session_id,
        reply=msg,
        tts_text=msg,
        is_new_session=req.session_id is None,
    )


@router.delete("/agent/session/{session_id}", dependencies=[Depends(require_service_key)])
async def agent_session_delete(session_id: str):
    """清除指定会话：联动清理 JWT、checkpoint 历史、锁与待确认订单。"""
    existed = delete_session(session_id)
    if existed:
        return JSONResponse({"code": 200, "message": f"会话 {session_id} 及其登录态/历史已清除"})
    return JSONResponse(
        status_code=404,
        content={"code": 404, "message": f"会话 {session_id} 不存在"},
    )
