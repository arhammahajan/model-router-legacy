"""Reversible Responses function namespaces for protocols with flat tool names."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from typing import Any

from .common import RouterError


def flatten_namespaces(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, tuple[str, str]]]:
    """Flatten function groups and prior calls without changing argument schemas."""
    payload = deepcopy(payload)
    names: dict[str, tuple[str, str]] = {}
    reserved = {tool["name"] for tool in payload.get("tools", [])
                if tool.get("type") != "namespace" and isinstance(tool.get("name"), str)}

    def alias(namespace: Any, name: Any) -> str:
        if not isinstance(namespace, str) or not namespace or not isinstance(name, str) or not name:
            raise RouterError(400, "Tool namespaces and function names must be nonempty strings")
        key = "ns_" + sha256((namespace + "\0" + name).encode()).hexdigest()[:56]
        if key in reserved or (key in names and names[key] != (namespace, name)):
            raise RouterError(400, "Tool namespace alias collides with another function")
        names[key] = (namespace, name)
        return key

    tools = []
    for tool in payload.get("tools", []):
        if tool.get("type") != "namespace":
            tools.append(tool)
            continue
        children = tool.get("tools")
        if not isinstance(children, list):
            raise RouterError(400, "A tool namespace must contain a function tools array")
        for child in children:
            if not isinstance(child, dict) or child.get("type") != "function":
                raise RouterError(400, "Only function tools can be translated from a namespace")
            tools.append({**child, "name": alias(tool.get("name"), child.get("name"))})
    if "tools" in payload:
        payload["tools"] = tools
    for item in payload.get("input", []) if isinstance(payload.get("input"), list) else []:
        if item.get("type") == "function_call" and item.get("namespace"):
            item["name"] = alias(item.pop("namespace"), item.get("name"))
    choice = payload.get("tool_choice")
    if isinstance(choice, dict) and choice.get("type") == "function" and choice.get("namespace"):
        choice["name"] = alias(choice.pop("namespace"), choice.get("name"))
    return payload, names


def restore_namespaces(value: Any, names: dict[str, tuple[str, str]]) -> Any:
    """Restore tool identities on JSON responses and individual Responses SSE events."""
    if isinstance(value, list):
        return [restore_namespaces(item, names) for item in value]
    if not isinstance(value, dict):
        return value
    result = dict(value)
    if (value.get("type") == "function_call" and isinstance(value.get("name"), str)
            and value["name"] in names):
        result["namespace"], result["name"] = names[value["name"]]
    for key in ("item", "output", "response"):
        if key in result:
            result[key] = restore_namespaces(result[key], names)
    return result
