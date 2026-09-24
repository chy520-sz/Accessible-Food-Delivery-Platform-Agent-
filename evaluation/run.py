"""Score recorded traces against curated cases: python -m evaluation.run --help."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from evaluation.scoring import aggregate, score_case, validate_case


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON") from exc
    return rows


def evaluate(cases_path: Path, traces_path: Path, labels_path: Path) -> dict:
    cases = _read_jsonl(cases_path)
    traces = _read_jsonl(traces_path)
    labels = _read_jsonl(labels_path)
    by_case = {row["case_id"]: row for row in labels}
    by_trace = defaultdict(list)
    for event in traces:
        by_trace[event.get("trace_id")].append(event)
    results = []
    pending = []
    for case in cases:
        validate_case(case)
        label = by_case.get(case["id"])
        if not label or not label.get("trace_id") or label["trace_id"] not in by_trace:
            pending.append(case["id"])
            continue
        results.append(score_case(case, by_trace[label["trace_id"]], label))
    return {"summary": aggregate(results), "pending_cases": pending, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline Agent trace evaluator; never calls production tools")
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.jsonl"))
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True,
                        help="JSONL case_id/trace_id/outcome/recovery_action; human reviewed")
    parser.add_argument("--output", type=Path, help="Optional report JSON path")
    args = parser.parse_args()
    report = evaluate(args.cases, args.traces, args.labels)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 1 if (report["pending_cases"] or report["summary"]["unreviewed_cases"] or
                 report["summary"]["not_evaluable_cases"] or
                 report["summary"]["safety_violations"] or
                 report["summary"]["safety_unreviewed"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
