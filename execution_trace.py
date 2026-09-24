"""Privacy-minimized Agent execution events for evaluation and operations.

Never persist prompts, replies, tool arguments/results, tokens, or user IDs.
An ephemeral HMAC fingerprint permits same-process duplicate-call detection.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("execution_trace")
_SECRET = secrets.token_bytes(32)
_WRITE_LOCK = threading.Lock()
_EVENTS = {"turn_start", "model_step", "tool_call", "tool_result", "guard_action", "turn_end"}


def fingerprint(value: Any) -> str:
    """Non-reversible across process restarts; never log the original value."""
    try:
        raw = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    except (TypeError, ValueError):
        raw = b"unserializable"
    return hmac.new(_SECRET, raw, hashlib.sha256).hexdigest()[:20]


class TurnTrace:
    def __init__(self, trace_id: str, path: str, enabled: bool = True):
        self.trace_id = trace_id
        self.path = Path(path)
        self.enabled = enabled
        self.started = time.monotonic()
        self.finished = False
        self.record("turn_start")

    def record(self, event: str, **fields: Any) -> None:
        if not self.enabled or event not in _EVENTS:
            return
        # Explicit allow-list prevents a future caller from adding raw private text.
        allowed = {
            "tool", "tool_call_id", "args_fingerprint", "error_code", "duration_ms",
            "phase", "executed", "terminal", "model_steps", "tool_calls", "args_valid",
        }
        safe = {key: value for key, value in fields.items() if key in allowed}
        payload = {
            "schema_version": 1,
            "trace_id": self.trace_id,
            "event": event,
            "timestamp_ms": int(time.time() * 1000),
            **safe,
        }
        try:
            with _WRITE_LOCK:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        except OSError:
            # Telemetry must not break ordering, login, or user-facing replies.
            logger.warning("Agent trace write failed: %s", type(self.path).__name__, exc_info=True)

    def finish(self, terminal: str, model_steps: int = 0, tool_calls: int = 0) -> None:
        if self.finished:
            return
        self.finished = True
        self.record(
            "turn_end", terminal=terminal,
            duration_ms=round((time.monotonic() - self.started) * 1000),
            model_steps=model_steps, tool_calls=tool_calls,
        )
