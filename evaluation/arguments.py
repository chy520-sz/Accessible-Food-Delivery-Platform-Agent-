"""Validate tool arguments in memory; emit only a boolean, never argument values."""

from __future__ import annotations

from typing import Any

from tool_registry import get


def arguments_valid(tool_name: str, args: Any) -> bool | None:
    spec = get(tool_name)
    if spec is None or not isinstance(args, dict):
        return None
    for field in spec.params:
        if field.name not in args:
            if field.required:
                return False
            continue
        value = args[field.name]
        if value is None and not field.required:
            continue
        kind = field.type
        if kind == "string" and not isinstance(value, str):
            return False
        if kind == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
            return False
        if kind == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
            return False
        if kind == "boolean" and not isinstance(value, bool):
            return False
        if kind == "array" and (not isinstance(value, list) or
                                not all(isinstance(item, str) for item in value)):
            return False
        if field.enum is not None and value not in field.enum:
            return False
    return True
