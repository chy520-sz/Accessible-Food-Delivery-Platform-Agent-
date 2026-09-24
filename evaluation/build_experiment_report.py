"""Build a case-by-case Markdown archive from the reviewed isolated run."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "evaluation"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def render() -> str:
    cases = read_jsonl(EVALUATION / "cases.jsonl")
    labels = {row["case_id"]: row for row in read_jsonl(
        EVALUATION / "labels_staging_2026-09-22.jsonl"
    )}
    responses: dict[str, dict] = {}
    for row in read_jsonl(ROOT / "data" / "eval_staging_responses.jsonl"):
        responses[row["case_id"]] = row
    for row in read_jsonl(EVALUATION / "seed_responses_staging_2026-09-22.jsonl"):
        responses.setdefault(row["case_id"], row)
    traces = read_jsonl(ROOT / "data" / "eval_staging_traces.jsonl")
    events_by_trace: dict[str, list[dict]] = {}
    for event in traces:
        events_by_trace.setdefault(event["trace_id"], []).append(event)
    scores = json.loads((ROOT / "data" / "evaluation-report.json").read_text(encoding="utf-8"))
    score_by_case = {row["case_id"]: row for row in scores["results"]}

    missing = {case["id"] for case in cases} - responses.keys()
    if missing:
        raise ValueError(f"Missing saved model responses: {sorted(missing)}")
    mismatched = [case_id for case_id, label in labels.items()
                  if responses[case_id].get("trace_id") != label.get("trace_id")]
    if mismatched:
        raise ValueError(f"Response/label trace IDs differ: {sorted(mismatched)}")

    summary = scores["summary"]
    lines = [
        "# Agent 隔离评测逐例数据（2026-09-22）",
        "",
        "## 实验范围",
        "",
        "- 模型：`.env` 配置的 DeepSeek `deepseek-chat`。",
        "- 样例：`evaluation/cases.jsonl` 中全部 50 条合成输入。",
        "- 工具环境：所有工具调用被本地固定响应代理拦截，没有访问 Java 后端、Milvus 或执行业务写操作。",
        "- 输出、工具轨迹、人工标签按 `trace_id` 关联。轨迹只记录工具名、参数指纹、错误码、参数校验标记和耗时，不保存参数值。",
        "- 逐例人工标签和理由：`evaluation/labels_staging_2026-09-22.jsonl`。原始工具轨迹：`data/eval_staging_traces.jsonl`。",
        "",
        "## 汇总",
        "",
        f"50 条运行；可评价 {summary['cases_labelled']} 条；不可评价 {summary['not_evaluable_cases']} 条。",
        f"完成 {summary['outcomes']['complete']}，正确失败 {summary['outcomes']['correct_failure']}，部分完成 {summary['outcomes']['partial']}，错误失败 {summary['outcomes']['wrong_failure']}。",
        f"预期工具路径符合率（可评价样例）{summary['required_tool_path_rate']:.1%}；观察到的禁止工具/高风险闸门违规 {summary['safety_violations']}。",
        "参数合法率受流式参数轨迹采集缺陷影响，详见复核报告，不应作为模型质量结论。",
        "",
        "## 逐例记录",
        "",
    ]

    for case in cases:
        case_id = case["id"]
        response = responses[case_id]
        label = labels[case_id]
        score = score_by_case[case_id]
        trace_id = response["trace_id"]
        events = events_by_trace.get(trace_id, [])
        tool_events = [event for event in events if event.get("event") == "tool_call"]
        tool_results = [event for event in events if event.get("event") == "tool_result"]
        terminal = next((event for event in reversed(events)
                         if event.get("event") == "turn_end"), {})
        calls = response.get("tool_calls", [])
        call_descriptions = [
            f"`{call['tool']}`({', '.join(call.get('argument_names', [])) or '无参数'})"
            for call in calls
        ]
        result_descriptions = []
        for index, event in enumerate(tool_results):
            parts = [f"`{event.get('tool', '?')}` 错误码 `{event.get('error_code', '未知')}`"]
            if "args_valid" in event:
                parts.append(f"参数校验 `{event['args_valid']}`")
            result_descriptions.append("，".join(parts))
        forbidden = score.get("forbidden_tools_used") or []
        evidence = [
            f"trace `{trace_id}`",
            f"工具调用 {score.get('tool_calls', len(tool_events))} 次，预算 {case['max_tool_calls']} 次",
            f"工具路径预期 `{score.get('required_tools_ok')}`",
            f"终态 `{terminal.get('terminal', '缺失')}`，耗时 {terminal.get('duration_ms', '未知')} ms",
            f"禁止工具 `{', '.join(forbidden) if forbidden else '未观察到'}`",
        ]
        if score.get("over_tool_budget"):
            evidence.append("超过工具调用预算")
        lines.extend([
            f"### {case_id}（{case['category']}）",
            "",
            f"**结论：** {'不可评价' if label.get('not_evaluable') else label.get('outcome', '未标注')}。{label.get('review_note', '')}",
            "",
            f"**输入：** {case['input']}",
            "",
            "**完整模型输出：**",
            "",
            "```text",
            response.get("reply", "[本轮没有回复文本]"),
            "```",
            "",
            f"**调用工具：** {'；'.join(call_descriptions) if call_descriptions else '无'}。",
            f"**工具返回证据：** {'；'.join(result_descriptions) if result_descriptions else '没有工具结果事件'}。",
            "**轨迹证据：**",
            "",
        ])
        lines.extend(f"- {item}" for item in evidence)
        lines.extend(["", "---", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    destination = EVALUATION / "EXPERIMENT_DATA_2026-09-22.md"
    destination.write_text(render(), encoding="utf-8")
    print(f"Wrote {destination}")
