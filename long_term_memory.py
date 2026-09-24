"""Typed, policy-governed long-term memory for the Agent.

Short-lived conversation messages remain in LangGraph.  This module only owns
cross-session user memory and session task state.  External knowledge is routed
to a RAG review inbox instead of being mixed into personal memory.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("long_term_memory")

LONG_TERM_TYPES = {
    "user_preference",
    "user_goal",
    "project_background",
    "historical_conclusion",
}
TASK_STATE = "task_state"
EXTERNAL_REFERENCE = "external_knowledge_reference"
ALL_MEMORY_TYPES = LONG_TERM_TYPES | {TASK_STATE, EXTERNAL_REFERENCE}

VALID_SOURCES = {"user_explicit", "verified_tool", "model_inferred"}
_SECRET_RE = re.compile(
    r"(?:password|passwd|口令|密码|api[_ -]?key|access[_ -]?token|refresh[_ -]?token|"
    r"bearer\s+[a-z0-9._-]+|银行卡|信用卡|cvv)", re.IGNORECASE
)
_SENSITIVE_RE = re.compile(
    r"(?:身份证|护照|家庭住址|详细地址|手机号|电话号码|诊断|病史|疾病)", re.IGNORECASE
)


class MemoryPolicyError(ValueError):
    """Raised when a memory write violates a deterministic policy."""


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    owner_id: str
    memory_type: str
    memory_key: str
    content: str
    importance: float
    confidence: float
    status: str
    source: str
    metadata: dict[str, Any]
    created_at: float
    updated_at: float
    last_accessed_at: Optional[float]
    access_count: int


@dataclass(frozen=True)
class WriteResult:
    action: str
    memory: Optional[MemoryRecord] = None
    message: str = ""
    rag_reference_id: Optional[str] = None


@dataclass(frozen=True)
class RetrievalContext:
    task_type: str
    memories: list[MemoryRecord]


TASK_PROFILES: dict[str, dict[str, Any]] = {
    "food_ordering": {
        "keywords": ("点餐", "菜", "吃", "口味", "套餐", "商家", "购物车", "下单"),
        "types": ("user_preference", "user_goal", "historical_conclusion"),
    },
    "health_management": {
        "keywords": ("健康", "减肥", "增肌", "体重", "营养", "饮食计划", "忌口", "过敏"),
        "types": ("user_preference", "user_goal", "project_background"),
    },
    "troubleshooting": {
        "keywords": ("失败", "异常", "没反应", "没送到", "不对", "登录不上", "排查", "故障"),
        "types": ("project_background", "historical_conclusion", "user_goal"),
    },
    "writing": {
        "keywords": ("简历", "润色", "写作", "改写", "文案"),
        "types": ("user_preference", "user_goal", "project_background"),
    },
    "general": {
        "keywords": (),
        "types": ("user_preference", "user_goal"),
    },
}


def classify_task(text: str) -> str:
    normalized = (text or "").lower()
    best_name, best_score = "general", 0
    for name, profile in TASK_PROFILES.items():
        if name == "general":
            continue
        score = sum(1 for word in profile["keywords"] if word.lower() in normalized)
        if score > best_score:
            best_name, best_score = name, score
    return best_name


class MemoryService:
    """SQLite-backed service; routers and agent tools never access SQL directly."""

    def __init__(self, db_path: str, rag_inbox_path: str):
        self.db_path = Path(db_path)
        self.rag_inbox_path = Path(rag_inbox_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.rag_inbox_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS long_term_memories (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL CHECK(memory_type IN (
                        'user_preference','user_goal','project_background','historical_conclusion')),
                    memory_key TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 0.5,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','superseded','deleted')),
                    source TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    last_accessed_at REAL,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(owner_id, memory_type, memory_key)
                );
                CREATE INDEX IF NOT EXISTS idx_ltm_owner_status
                    ON long_term_memories(owner_id, status, memory_type);
                CREATE TABLE IF NOT EXISTS session_task_states (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    state_key TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    UNIQUE(session_id, state_key)
                );
                CREATE INDEX IF NOT EXISTS idx_task_state_expiry ON session_task_states(expires_at);
                CREATE TABLE IF NOT EXISTS memory_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT,
                    owner_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL
                );
                """
            )

    @staticmethod
    def _row(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"], owner_id=row["owner_id"], memory_type=row["memory_type"],
            memory_key=row["memory_key"], content=row["content"],
            importance=float(row["importance"]), confidence=float(row["confidence"]),
            status=row["status"], source=row["source"],
            metadata=json.loads(row["metadata_json"] or "{}"),
            created_at=float(row["created_at"]), updated_at=float(row["updated_at"]),
            last_accessed_at=row["last_accessed_at"], access_count=int(row["access_count"]),
        )

    @staticmethod
    def _validate_common(memory_type: str, key: str, content: str, source: str) -> tuple[str, str]:
        if memory_type not in ALL_MEMORY_TYPES:
            raise MemoryPolicyError(f"不支持的记忆类型: {memory_type}")
        if source not in VALID_SOURCES:
            raise MemoryPolicyError(f"不支持的来源: {source}")
        clean_key, clean_content = key.strip()[:80], content.strip()[:2000]
        if not clean_key or not clean_content:
            raise MemoryPolicyError("记忆 key 和内容不能为空")
        if _SECRET_RE.search(clean_key + " " + clean_content):
            raise MemoryPolicyError("密码、令牌、支付凭证等秘密信息禁止写入记忆")
        return clean_key, clean_content

    def write(
        self, *, owner_id: str, session_id: str, memory_type: str, key: str,
        content: str, source: str = "user_explicit", user_confirmed: bool = False,
        importance: float = 0.5, confidence: float = 1.0,
        metadata: Optional[dict[str, Any]] = None, task_ttl_seconds: int = 1800,
    ) -> WriteResult:
        owner_id = str(owner_id or "").strip()
        if not owner_id:
            raise MemoryPolicyError("长期记忆必须绑定已认证用户")
        key, content = self._validate_common(memory_type, key, content, source)
        metadata = metadata or {}
        try:
            metadata_json = json.dumps(metadata, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise MemoryPolicyError("metadata 必须是可序列化的 JSON 对象") from exc
        if len(metadata_json) > 8000:
            raise MemoryPolicyError("metadata 内容过长")
        if _SECRET_RE.search(metadata_json):
            raise MemoryPolicyError("metadata 中禁止写入密码、令牌或支付凭证")

        if memory_type == EXTERNAL_REFERENCE:
            return self._route_to_rag(owner_id, key, content, metadata)
        if memory_type == TASK_STATE:
            return self._write_task_state(
                owner_id, session_id, key, content, metadata, task_ttl_seconds
            )
        if source == "model_inferred" and not user_confirmed:
            raise MemoryPolicyError("模型推断的信息需要用户明确确认后才能长期保存")
        if _SENSITIVE_RE.search(key + " " + content) and not user_confirmed:
            raise MemoryPolicyError("敏感个人信息需要用户明确确认后才能长期保存")
        if memory_type == "historical_conclusion" and source not in {"verified_tool", "user_explicit"}:
            raise MemoryPolicyError("历史结论只能来自用户明确陈述或已验证工具结果")

        now = time.time()
        record_id = uuid.uuid4().hex
        importance = max(0.0, min(1.0, float(importance)))
        confidence = max(0.0, min(1.0, float(confidence)))
        with self._write_lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM long_term_memories WHERE owner_id=? AND memory_type=? AND memory_key=?",
                (owner_id, memory_type, key),
            ).fetchone()
            action = "updated" if existing else "created"
            if existing:
                record_id = existing["id"]
                conn.execute(
                    """UPDATE long_term_memories SET content=?, importance=?, confidence=?, status='active',
                       source=?, metadata_json=?, updated_at=? WHERE id=?""",
                    (content, importance, confidence, source, metadata_json, now, record_id),
                )
            else:
                conn.execute(
                    """INSERT INTO long_term_memories
                       (id,owner_id,memory_type,memory_key,content,importance,confidence,status,source,
                        metadata_json,created_at,updated_at) VALUES (?,?,?,?,?,?,?,'active',?,?,?,?)""",
                    (record_id, owner_id, memory_type, key, content, importance, confidence, source,
                     metadata_json, now, now),
                )
            self._audit(conn, record_id, owner_id, action, "policy_accepted")
            row = conn.execute("SELECT * FROM long_term_memories WHERE id=?", (record_id,)).fetchone()
        return WriteResult(action=action, memory=self._row(row), message="记忆已保存")

    def _write_task_state(self, owner_id: str, session_id: str, key: str, content: str,
                          metadata: dict[str, Any], ttl: int) -> WriteResult:
        if not session_id:
            raise MemoryPolicyError("任务状态必须绑定会话")
        now, record_id = time.time(), uuid.uuid4().hex
        expires_at = now + max(60, min(int(ttl), 86400))
        with self._write_lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO session_task_states
                   (id,session_id,owner_id,state_key,content,metadata_json,created_at,updated_at,expires_at)
                   VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(session_id,state_key) DO UPDATE SET
                   content=excluded.content, metadata_json=excluded.metadata_json,
                   updated_at=excluded.updated_at, expires_at=excluded.expires_at""",
                (record_id, session_id, owner_id, key, content,
                 json.dumps(metadata, ensure_ascii=False), now, now, expires_at),
            )
            self._audit(conn, None, owner_id, "task_state_upsert", key)
        return WriteResult(action="temporary", message="任务状态仅在当前会话临时保存")

    def _route_to_rag(self, owner_id: str, key: str, content: str,
                      metadata: dict[str, Any]) -> WriteResult:
        ref_id = uuid.uuid4().hex
        doc = {
            "id": ref_id, "title": key, "content": content, "metadata": metadata,
            "submitted_by": owner_id, "submitted_at": time.time(), "review_status": "pending",
        }
        with self._write_lock:
            with self.rag_inbox_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
        return WriteResult(
            action="routed_to_rag", message="外部知识引用已进入 RAG 审核收件箱，未写入用户记忆",
            rag_reference_id=ref_id,
        )

    @staticmethod
    def _audit(conn: sqlite3.Connection, memory_id: Optional[str], owner_id: str,
               action: str, reason: str = "") -> None:
        conn.execute(
            "INSERT INTO memory_audit_log(memory_id,owner_id,action,reason,created_at) VALUES (?,?,?,?,?)",
            (memory_id, owner_id, action, reason[:300], time.time()),
        )

    def retrieve(self, owner_id: str, query: str, session_id: str = "", limit: int = 6) -> RetrievalContext:
        task_type = classify_task(query)
        allowed = TASK_PROFILES[task_type]["types"]
        placeholders = ",".join("?" for _ in allowed)
        now = time.time()
        bounded_limit = max(0, min(limit, 20))
        with self._connect() as conn:
            rows = conn.execute(
                f"""SELECT * FROM long_term_memories WHERE owner_id=? AND status='active'
                    AND memory_type IN ({placeholders})""",
                (str(owner_id), *allowed),
            ).fetchall()
            records = [self._row(row) for row in rows]
            terms = set(re.findall(r"[\w\u4e00-\u9fff]+", (query or "").lower()))

            def score(rec: MemoryRecord) -> float:
                haystack = f"{rec.memory_key} {rec.content}".lower()
                lexical = sum(1 for term in terms if term in haystack) / max(1, len(terms))
                age_days = max(0.0, (now - rec.updated_at) / 86400)
                recency = 1.0 / (1.0 + age_days / 30.0)
                return lexical * 0.5 + rec.importance * 0.25 + rec.confidence * 0.15 + recency * 0.1

            records.sort(key=score, reverse=True)
            task_records: list[MemoryRecord] = []
            if session_id and bounded_limit:
                task_rows = conn.execute(
                    "SELECT state_key,content FROM session_task_states WHERE session_id=? AND owner_id=? AND expires_at>?",
                    (session_id, str(owner_id), now),
                ).fetchall()
                # Temporary state has priority for continuation of the current task.
                for row in task_rows[:bounded_limit]:
                    task_records.append(MemoryRecord(
                        id=f"task:{row['state_key']}", owner_id=str(owner_id), memory_type=TASK_STATE,
                        memory_key=row["state_key"], content=row["content"], importance=1.0,
                        confidence=1.0, status="active", source="verified_tool", metadata={},
                        created_at=now, updated_at=now, last_accessed_at=now, access_count=1,
                    ))
            selected = records[:max(0, bounded_limit - len(task_records))]
            if selected:
                ids = [rec.id for rec in selected]
                marks = ",".join("?" for _ in ids)
                conn.execute(
                    f"UPDATE long_term_memories SET last_accessed_at=?, access_count=access_count+1 WHERE id IN ({marks})",
                    (now, *ids),
                )
        return RetrievalContext(task_type=task_type, memories=task_records + selected)

    def list_memories(self, owner_id: str, include_inactive: bool = False) -> list[MemoryRecord]:
        where = "owner_id=?" if include_inactive else "owner_id=? AND status='active'"
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM long_term_memories WHERE {where} ORDER BY updated_at DESC", (str(owner_id),)
            ).fetchall()
        return [self._row(row) for row in rows]

    def delete(self, owner_id: str, memory_id: Optional[str] = None) -> int:
        with self._write_lock, self._connect() as conn:
            if memory_id:
                cur = conn.execute(
                    "UPDATE long_term_memories SET status='deleted',updated_at=? WHERE id=? AND owner_id=? AND status!='deleted'",
                    (time.time(), memory_id, str(owner_id)),
                )
            else:
                cur = conn.execute(
                    "UPDATE long_term_memories SET status='deleted',updated_at=? WHERE owner_id=? AND status!='deleted'",
                    (time.time(), str(owner_id)),
                )
            count = cur.rowcount
            self._audit(conn, memory_id, str(owner_id), "deleted", "user_request")
        return count

    def supersede(self, owner_id: str, memory_id: str, reason: str) -> int:
        """Downgrade an overturned conclusion without erasing its audit history."""
        clean_reason = (reason or "旧结论已被新信息推翻").strip()[:300]
        with self._write_lock, self._connect() as conn:
            cur = conn.execute(
                """UPDATE long_term_memories SET status='superseded',
                   confidence=MIN(confidence,0.2), importance=MIN(importance,0.2), updated_at=?
                   WHERE id=? AND owner_id=? AND memory_type='historical_conclusion'
                   AND status='active'""",
                (time.time(), memory_id, str(owner_id)),
            )
            if cur.rowcount:
                self._audit(conn, memory_id, str(owner_id), "superseded", clean_reason)
        return cur.rowcount

    def clear_session_state(self, session_id: str) -> int:
        with self._write_lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM session_task_states WHERE session_id=?", (session_id,))
        return cur.rowcount

    def maintain(self, now: Optional[float] = None, stale_days: int = 180) -> dict[str, int]:
        """Expire task state, decay unused low-value memory, purge old tombstones."""
        now = time.time() if now is None else now
        stale_before = now - max(1, stale_days) * 86400
        purge_before = now - 30 * 86400
        with self._write_lock, self._connect() as conn:
            expired = conn.execute("DELETE FROM session_task_states WHERE expires_at<=?", (now,)).rowcount
            auto_deleted = conn.execute(
                """UPDATE long_term_memories SET status='deleted',updated_at=?
                   WHERE status='active' AND importance<0.2 AND access_count=0
                   AND COALESCE(last_accessed_at,updated_at)<?""",
                (now, stale_before),
            ).rowcount
            decayed = conn.execute(
                """UPDATE long_term_memories SET importance=MAX(0.1,importance*0.8),updated_at=?
                   WHERE status='active' AND importance>=0.2 AND importance<0.5 AND access_count=0
                   AND COALESCE(last_accessed_at,updated_at)<?""",
                (now, stale_before),
            ).rowcount
            purged = conn.execute(
                "DELETE FROM long_term_memories WHERE status='deleted' AND updated_at<?", (purge_before,)
            ).rowcount
        return {
            "expired_task_states": expired,
            "decayed_memories": decayed,
            "auto_deleted_memories": auto_deleted,
            "purged_memories": purged,
        }


def record_to_dict(record: MemoryRecord) -> dict[str, Any]:
    return asdict(record)
