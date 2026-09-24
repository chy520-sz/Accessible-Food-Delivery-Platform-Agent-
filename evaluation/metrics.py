"""Operational metrics from privacy-minimized event traces (not task-success labels)."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def summarize(events: list[dict]) -> dict:
    grouped = defaultdict(list)
    for event in events:
        if event.get("trace_id"):
            grouped[event["trace_id"]].append(event)
    terminal = [event for trace in grouped.values() for event in trace if event.get("event") == "turn_end"]
    calls = [event for trace in grouped.values() for event in trace if event.get("event") == "tool_call"]
    results = [event for trace in grouped.values() for event in trace if event.get("event") == "tool_result"]
    errors = [event for event in results if event.get("error_code") not in (None, "", "OK")]
    guards = [event for trace in grouped.values() for event in trace if event.get("event") == "guard_action"]
    signatures = Counter((event.get("trace_id"), event.get("tool"), event.get("args_fingerprint"))
                         for event in calls)
    duplicates = sum(count - 1 for count in signatures.values() if count > 1)
    return {
        "traces": len(grouped),
        "completed_turns": sum(event.get("terminal") == "done" for event in terminal),
        "timeouts": sum(event.get("terminal") == "timeout" for event in terminal),
        "tool_calls": len(calls),
        "tool_failures": len(errors),
        "tool_failure_rate": len(errors) / len(results) if results else None,
        "duplicate_call_rate": duplicates / len(calls) if calls else 0,
        "average_tool_calls": len(calls) / len(grouped) if grouped else None,
        "p95_duration_ms": _percentile([event["duration_ms"] for event in terminal
                                        if isinstance(event.get("duration_ms"), (int, float))], 0.95),
        "guarded_action_attempts": sum(bool(event.get("executed")) for event in guards),
        "drafts_required": sum(event.get("phase") == "NEED_CONFIRM" for event in guards),
        "error_codes": dict(Counter(event.get("error_code") for event in errors)),
        "note": "completed_turns is transport completion, not task completion; confirmation needs human review",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize safe Agent operational trace events")
    parser.add_argument("--traces", type=Path, required=True)
    args = parser.parse_args()
    events = [json.loads(line) for line in args.traces.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    print(json.dumps(summarize(events), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

