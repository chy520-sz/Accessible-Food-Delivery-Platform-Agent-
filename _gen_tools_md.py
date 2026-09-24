# -*- coding: utf-8 -*-
"""从 tool_registry 生成 TOOLS.md，保证文档与代码单一事实来源一致。"""
import io
import tool_registry as reg
from tool_errors import ERRORS

out = []
A = out.append

A("# 工具清单与治理规范（TOOLS.md）")
A("")
A("> 本文档由 `tool_registry.py` 自动生成，是 Agent 工具的单一事实来源。")
A("> 修改工具元数据请改 `tool_registry.py`，再重新运行 `python _gen_tools_md.py`。")
A("")
A("## 一、设计原则")
A("")
A("1. **查询类（query）与执行类（mutation）严格分离**：query 只读、无副作用、可安全重试；")
A("   mutation 会写后端状态，写操作**绝不自动重试**（防重复提交）。")
A("2. **高风险操作两步确认**：凡花钱、不可逆、或推进交易状态机的工具，第一次调用只生成")
A("   草稿（返回 `[code=DRAFT_REQUIRED]`），必须向用户复述影响并获明确确认后，再以相同")
A("   参数调用一次才真正执行；同参数重复提交幂等去重。草稿默认 5 分钟过期。")
A("3. **统一错误码**：所有工具错误以 `[code=XXX]` 形式追加在返回文本末尾，机器可读、稳定不变。")
A("")
A("## 二、错误码总表")
A("")
A("| 错误码 | 含义 | 可重试 | 期望动作 |")
A("|---|---|---|---|")
for code, spec in ERRORS.items():
    if code == "OK":
        continue
    A(f"| `{code}` | {spec.meaning} | {'是' if spec.retryable else '否'} | {spec.user_action} |")
A("")
A("## 三、工具分类总览")
A("")
A(f"- 查询类（query）：**{len(reg.queries())}** 个")
A(f"- 执行类（mutation）：**{len(reg.mutations())}** 个")
hr = reg.high_risk()
A(f"  - 其中高风险（草稿闸门）：**{len(hr)}** 个 —— {', '.join(f'`{n}`' for n in hr)}")
A("")

# 详情分组
groups = [
    ("四、查询类工具（只读，无副作用）", reg.queries()),
    ("五、执行类工具 · 低风险（直接生效）",
     {n: s for n, s in reg.mutations().items() if n not in hr}),
    ("六、执行类工具 · 高风险（先生成草稿，确认后提交）", hr),
]

for title, specs in groups:
    A(f"## {title[2:]}")
    A("")
    for name in sorted(specs):
        s = specs[name]
        auth = "需登录" if s.auth_required else "公开/无需登录"
        A(f"### `{name}`")
        A("")
        A(f"- **用途**：{s.description}")
        A(f"- **鉴权**：{auth}　|　**类型**：{s.kind}　|　**风险**：{s.risk}"
          + ("　|　**草稿确认**：是" if s.draft_required else ""))
        A(f"- **适用场景**：")
        for w in s.when_to_use:
            A(f"  - {w}")
        A(f"- **不适用场景**：")
        for w in s.when_not_to_use:
            A(f"  - {w}")
        A(f"- **参数 Schema**：")
        if s.params:
            A("")
            A("  | 参数 | 类型 | 必填 | 说明 |")
            A("  |---|---|---|---|")
            for p in s.params:
                enum = f"，取值 {p.enum}" if p.enum else ""
                defv = f"，默认 {p.default!r}" if not p.required and p.default is not None else ""
                A(f"  | `{p.name}` | {p.type} | {'是' if p.required else '否'} | {p.description}{enum}{defv} |")
        else:
            A("  （无参数）")
        A(f"- **返回结构**：{s.returns}")
        if s.side_effect:
            A(f"- **副作用**：{s.side_effect}")
        A(f"- **可能错误码**：{', '.join(f'`{e}`' for e in s.errors)}")
        A("")

content = "\n".join(out) + "\n"
io.open("TOOLS.md", "w", encoding="utf-8", newline="\n").write(content)
print("TOOLS.md written, chars:", len(content))
