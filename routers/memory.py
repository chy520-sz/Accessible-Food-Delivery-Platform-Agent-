"""Long-term memory API; delegates all persistence and policy to MemoryService."""

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from agent import get_memory_service
from routers.deps import require_service_key
from tools import get_session
from long_term_memory import MemoryPolicyError, record_to_dict

router = APIRouter(prefix="/agent/memories", tags=["长期记忆"])


MemoryType = Literal[
    "user_preference", "user_goal", "project_background", "task_state",
    "historical_conclusion", "external_knowledge_reference",
]


class MemoryWriteRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    memory_type: MemoryType
    key: str = Field(min_length=1, max_length=80)
    content: str = Field(min_length=1, max_length=2000)
    source: Literal["user_explicit", "verified_tool", "model_inferred"] = "user_explicit"
    user_confirmed: bool = False
    importance: float = Field(default=0.5, ge=0, le=1)
    confidence: float = Field(default=1.0, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _identity(session_id: str) -> str:
    session = get_session(session_id)
    owner_id = session.get("user_id") if session else None
    if owner_id in (None, ""):
        raise HTTPException(status_code=401, detail="请先登录后再管理长期记忆")
    return str(owner_id)


@router.get("", dependencies=[Depends(require_service_key)])
async def list_memories(session_id: str = Query(min_length=1, max_length=128)):
    owner_id = _identity(session_id)
    records = get_memory_service().list_memories(owner_id)
    return {"code": 200, "items": [record_to_dict(record) for record in records]}


@router.post("", dependencies=[Depends(require_service_key)])
async def write_memory(req: MemoryWriteRequest):
    owner_id = _identity(req.session_id)
    try:
        result = get_memory_service().write(
            owner_id=owner_id, session_id=req.session_id, memory_type=req.memory_type,
            key=req.key, content=req.content, source=req.source,
            user_confirmed=req.user_confirmed, importance=req.importance,
            confidence=req.confidence, metadata=req.metadata,
        )
    except MemoryPolicyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "code": 200,
        "action": result.action,
        "message": result.message,
        "memory": record_to_dict(result.memory) if result.memory else None,
        "rag_reference_id": result.rag_reference_id,
    }


@router.delete("/{memory_id}", dependencies=[Depends(require_service_key)])
async def delete_memory(memory_id: str, session_id: str = Query(min_length=1, max_length=128)):
    count = get_memory_service().delete(_identity(session_id), memory_id)
    if not count:
        raise HTTPException(status_code=404, detail="记忆不存在或不属于当前用户")
    return {"code": 200, "message": "记忆已删除"}


@router.delete("", dependencies=[Depends(require_service_key)])
async def delete_all_memories(session_id: str = Query(min_length=1, max_length=128)):
    count = get_memory_service().delete(_identity(session_id))
    return {"code": 200, "message": "用户长期记忆已删除", "deleted": count}
