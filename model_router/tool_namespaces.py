"""Reversible Responses tool namespaces for protocols with flat tool names."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from typing import Any

from .common import RouterError


def flatten_namespaces(
    payload: dict[str, Any], *, origins: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, tuple[str, str]]]:
    """Flatten function/custom groups, history and choices; retain original paths."""
    payload = deepcopy(payload)
    names: dict[str, tuple[str, str]] = {}
    reserved = {tool["name"] for tool in (payload.get("tools") or [])
                if tool.get("type") != "namespace" and isinstance(tool.get("name"), str)}
    if isinstance(payload.get("input"), list):
        reserved.update(item["name"] for item in payload["input"]
                        if isinstance(item, dict) and not item.get("namespace") and isinstance(item.get("name"), str))

    def alias(namespace: Any, name: Any) -> str:
        if not isinstance(namespace, str) or not namespace or not isinstance(name, str) or not name:
            raise RouterError(400, "Tool namespaces and function names must be nonempty strings")
        key = "ns_" + sha256((namespace + "\0" + name).encode()).hexdigest()[:56]
        if key in reserved or (key in names and names[key] != (namespace, name)):
            raise RouterError(400, "Tool namespace alias collides with another function")
        names[key] = (namespace, name)
        return key

    tools = []
    for index, tool in enumerate(payload.get("tools") or []):
        if tool.get("type") != "namespace":
            if origins is not None:
                origins[f"tools[{len(tools)}]"] = f"tools[{index}]"
            tools.append(tool)
            continue
        children = tool.get("tools")
        if not isinstance(children, list):
            raise RouterError(400, "A tool namespace must contain a function tools array")
        for child_index, child in enumerate(children):
            if not isinstance(child, dict):
                raise RouterError(400, "A tool namespace must contain tool objects")
            if origins is not None:
                origins[f"tools[{len(tools)}]"] = f"tools[{index}].tools[{child_index}]"
            if child.get("type") not in ("function", "custom"):
                # Not a function: surface it as an ordinary tool so the translation
                # policy accepts or rejects it individually.
                tools.append(child)
                continue
            tools.append({**child, "name": alias(tool.get("name"), child.get("name"))})
    if "tools" in payload:
        payload["tools"] = tools
    for item in payload.get("input", []) if isinstance(payload.get("input"), list) else []:
        if item.get("type") in ("function_call", "custom_tool_call") and item.get("namespace"):
            item["name"] = alias(item.pop("namespace"), item.get("name"))
    choice = payload.get("tool_choice")
    def flatten_choice(choice: Any) -> None:
        if not isinstance(choice, dict):
            return
        if choice.get("type") in ("function", "custom") and choice.get("namespace"):
            choice["name"] = alias(choice.pop("namespace"), choice.get("name"))
        if choice.get("type") == "allowed_tools":
            for member in choice.get("tools", []):
                flatten_choice(member)

    flatten_choice(choice)
    return payload, names


def restore_namespaces(value: Any, names: dict[str, tuple[str, str]]) -> Any:
    """Restore tool identities on JSON responses and individual Responses SSE events."""
    if isinstance(value, list):
        return [restore_namespaces(item, names) for item in value]
    if not isinstance(value, dict):
        return value
    result = dict(value)
    if (value.get("type") in ("function_call", "custom_tool_call") and isinstance(value.get("name"), str)
            and value["name"] in names):
        result["namespace"], result["name"] = names[value["name"]]
    for key in ("item", "output", "response"):
        if key in result:
            result[key] = restore_namespaces(result[key], names)
    return result
