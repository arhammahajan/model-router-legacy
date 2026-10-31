"""Request-local tool decisions and reversible custom-tool protocol translation.

The router translates tool contracts; execution stays with the client or upstream.
Schema functions are portable. Freeform tools can be emulated through a string
field, with an explicit approximation. Hosted and versioned vendor tools need
reviewed mappings and otherwise follow the strict/lossy policy.
"""
from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping

from .common import RouterError, _jsonable
from .constants import DIALECT_ANTHROPIC, DIALECT_OPENAI_CHAT, DIALECT_OPENAI_RESPONSES
from .tool_namespaces import flatten_namespaces, restore_namespaces

# Bound buffering independently of upstream chunk sizes and the request limit.
MAX_TOOL_ARGUMENTS = 1_000_000
MAX_BUFFERED_TOOL_ARGUMENTS = 8_000_000
MAX_BUFFERED_TOOL_CALLS = 1024
_SAFE_LABEL = re.compile(r"[A-Za-z0-9_.:-]{1,64}\Z")
_NATIVE_RESPONSES_HISTORY = frozenset({
    "web_search_call", "file_search_call", "code_interpreter_call", "image_generation_call",
    "computer_call", "computer_call_output", "local_shell_call", "local_shell_call_output",
    "shell_call", "shell_call_output", "apply_patch_call", "apply_patch_call_output",
    "mcp_call", "mcp_list_tools", "mcp_approval_request", "mcp_approval_response", "tool_search_call",
})


def _label(value: Any) -> str:
    return value if isinstance(value, str) and _SAFE_LABEL.fullmatch(value) else "unknown"


def _invalid(field: str = "tools") -> RouterError:
    return RouterError(400, f"Invalid {field} for tool translation", code="invalid_tool")


def _bad_call() -> RouterError:
    # Tool arguments and names are prompt data, never diagnostic text.
    return RouterError(502, "Upstream returned an invalid or oversized translated tool call",
                       code="invalid_tool_call")


def function_name(tool: Mapping[str, Any], ingress: str) -> str | None:
    kind = tool.get("type")
    if ingress == DIALECT_ANTHROPIC:
        name = tool.get("name") if kind in (None, "custom") else None
    elif kind in (None, "function"):
        spec = tool.get("function") if ingress == DIALECT_OPENAI_CHAT else tool
        name = spec.get("name") if isinstance(spec, Mapping) else None
    else:
        name = None
    return name if isinstance(name, str) else None


@dataclass(frozen=True)
class ToolDecision:
    path: str
    kind: str
    execution: str
    outcome: str                    # exact / approximated / unsupported
    reason: str = ""

    def loss(self) -> dict[str, str] | None:
        if self.outcome == "exact":
            return None
        return {"field": self.path, "reason": self.reason,
                "action": "approximated" if self.outcome == "approximated" else "dropped"}


def decide_tool(ingress: str, target: str, kind: Any, path: str, *,
                function: bool = False) -> ToolDecision:
    """One reviewed decision table, also used before Gemini's Chat normalization."""
    label = _label(kind)
    custom = kind == "custom" and ingress in (DIALECT_OPENAI_CHAT, DIALECT_OPENAI_RESPONSES)
    typed_client = (ingress == DIALECT_ANTHROPIC and isinstance(kind, str)
                    and kind.startswith(("bash_", "text_editor_", "computer_", "memory_")))
    execution = "client" if function or custom or typed_client else "upstream_or_unknown"
    if ingress == target or function:
        return ToolDecision(path, label, execution, "exact")
    if custom:
        if ingress == DIALECT_OPENAI_CHAT and target == DIALECT_OPENAI_RESPONSES:
            return ToolDecision(path, label, execution, "exact")
        return ToolDecision(path, label, execution, "approximated",
                            "custom tool input is emulated by a string-valued function; "
                            "native freeform/grammar generation is not preserved")
    return ToolDecision(path, label, execution, "unsupported",
                        f"native tool type '{label}' has no equivalent on this target "
                        "and requires a native route")


def _choice_name(choice: Mapping[str, Any]) -> str | None:
    spec = choice.get("function") or choice.get("custom") or choice
    return spec.get("name") if isinstance(spec, Mapping) else None


def _encode(text: Any) -> str:
    if not isinstance(text, str) or len(text) > MAX_TOOL_ARGUMENTS:
        raise _invalid("custom tool input")
    encoded = json.dumps({"content": text}, ensure_ascii=False, separators=(",", ":"))
    if len(encoded) > MAX_TOOL_ARGUMENTS:
        raise _invalid("custom tool input")
    return encoded


def _decode(arguments: Any) -> str:
    if not isinstance(arguments, str) or len(arguments) > MAX_TOOL_ARGUMENTS:
        raise _bad_call()
    try:
        value = json.loads(arguments)
    except (ValueError, RecursionError):
        raise _bad_call() from None
    if (not isinstance(value, dict) or set(value) != {"content"}
            or not isinstance(value["content"], str)):
        raise _bad_call()
    return value["content"]


def _custom_spec(tool: Mapping[str, Any], ingress: str) -> dict[str, Any]:
    if ingress == DIALECT_OPENAI_CHAT and tool.keys() - {"type", "custom"}:
        raise _invalid("custom tool definition")
    spec = tool.get("custom") if ingress == DIALECT_OPENAI_CHAT else tool
    allowed_spec = {"name", "description", "format"} | ({"type"} if ingress == DIALECT_OPENAI_RESPONSES else set())
    if (not isinstance(spec, dict) or not isinstance(spec.get("name"), str) or not spec["name"]
            or spec.keys() - allowed_spec
            or ("description" in spec and not isinstance(spec["description"], str))):
        raise _invalid("custom tool definition")
    spec = deepcopy(spec)
    fmt = spec.get("format")
    if fmt is not None:
        if not isinstance(fmt, dict) or fmt.get("type") not in ("text", "grammar"):
            raise _invalid("custom tool format")
        if fmt["type"] == "grammar" and ingress == DIALECT_OPENAI_CHAT:
            grammar = fmt.get("grammar")
            if (fmt.keys() - {"type", "grammar"} or not isinstance(grammar, dict)
                    or grammar.keys() - {"syntax", "definition"}):
                raise _invalid("custom tool format")
            fmt = spec["format"] = {"type": "grammar", **grammar}
        allowed = {"type"} if fmt["type"] == "text" else {"type", "syntax", "definition"}
        if fmt.keys() - allowed or (fmt["type"] == "grammar" and (
            fmt.get("syntax") not in ("lark", "regex") or not isinstance(fmt.get("definition"), str)
        )):
            raise _invalid("custom tool format")
    return spec


def _function_tool(spec: Mapping[str, Any], ingress: str) -> dict[str, Any]:
    description = spec.get("description", "")
    fmt = spec.get("format") or {}
    if fmt.get("type") == "grammar":
        description += "\n\nReturn content following this " + fmt["syntax"] + " grammar:\n" + fmt["definition"]
    function = {
        "name": spec["name"], "description": description,
        "parameters": {"type": "object", "properties": {"content": {"type": "string"}},
                       "required": ["content"], "additionalProperties": False},
    }
    return ({"type": "function", "function": function} if ingress == DIALECT_OPENAI_CHAT
            else {"type": "function", **function})


@dataclass
class ToolPlan:
    ingress: str
    target: str
    request: dict[str, Any]
    decisions: list[ToolDecision] = field(default_factory=list)
    losses: list[dict[str, str]] = field(default_factory=list)
    namespaces: dict[str, tuple[str, str]] = field(default_factory=dict)
    namespace_definitions: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Upstream function identity -> original, flat custom definition.
    custom: dict[str, dict[str, Any]] = field(default_factory=dict)
    native_custom: set[str] = field(default_factory=set)
    origins: dict[str, str] = field(default_factory=dict)
    # Exact reviewed body fields that the pinned Chat facade cannot validate.
    body: dict[str, Any] = field(default_factory=dict)

    def response(self, value: Any) -> Any:
        """Restore only protocol envelopes; never traverse user schemas/arguments."""
        data = _jsonable(value)
        if not isinstance(data, dict):
            return data
        if self.ingress == DIALECT_OPENAI_CHAT:
            data = deepcopy(data)
            for choice in data.get("choices") or []:
                message = choice.get("message") or {}
                for index, call in enumerate(message.get("tool_calls") or []):
                    function = call.get("function") or {}
                    name = function.get("name")
                    if call.get("type") == "function" and (name in self.custom or name in self.native_custom):
                        text = function.get("arguments")
                        if name in self.custom:
                            text = _decode(text)
                        elif not isinstance(text, str) or len(text) > MAX_TOOL_ARGUMENTS:
                            raise _bad_call()
                        message["tool_calls"][index] = {
                            **{k: v for k, v in call.items() if k != "function"}, "type": "custom",
                            "custom": {"name": name, "input": text},
                        }
            return data
        return restore_namespaces(self._responses_value(data), self.namespaces)

    def _responses_value(self, data: dict[str, Any]) -> dict[str, Any]:
        data = dict(data)
        if data.get("type") == "function_call" and data.get("name") in self.custom:
            data["type"] = "custom_tool_call"
            data["input"] = _decode(data.pop("arguments", None))
            data["id"] = _custom_id(data.get("id"))
        for key in ("response", "item"):
            if isinstance(data.get(key), dict):
                data[key] = self._responses_value(data[key])
        if isinstance(data.get("output"), list):
            data["output"] = [self._responses_value(item) if isinstance(item, dict) else item
                              for item in data["output"]]
        if isinstance(data.get("tools"), list):
            tools, groups = [], {}
            for raw in data["tools"]:
                tool = deepcopy(raw)
                name = tool.get("name")
                if tool.get("type") == "function" and name in self.custom:
                    tool = deepcopy(self.custom[name])
                if name in self.namespaces:
                    namespace, tool["name"] = self.namespaces[name]
                    if namespace not in groups:
                        groups[namespace] = {**self.namespace_definitions.get(namespace, {
                            "type": "namespace", "name": namespace}), "tools": []}
                        tools.append(groups[namespace])
                    groups[namespace]["tools"].append(tool)
                else:
                    tools.append(tool)
            data["tools"] = tools
        choice = data.get("tool_choice")
        if isinstance(choice, dict) and choice.get("type") == "function":
            name = choice.get("name")
            choice = dict(choice)
            if name in self.custom:
                choice["type"] = "custom"
            if name in self.namespaces:
                choice["namespace"], choice["name"] = self.namespaces[name]
            data["tool_choice"] = choice
        return data

    def responses_stream(self, source: Iterable[Any]) -> Iterator[dict[str, Any]]:
        """Buffer emulated arguments; emit the custom-input event contract."""
        calls: dict[str, dict[str, Any]] = {}
        budget = _BufferBudget()
        completed = False
        for raw in source:
            event = _jsonable(raw)
            if not isinstance(event, dict):
                yield event
                continue
            kind = event.get("type")
            if kind in ("error", "response.failed", "response.incomplete"):
                yield restore_namespaces(event, self.namespaces)
                return
            item = event.get("item") or {}
            if (kind == "response.output_item.added" and item.get("type") == "function_call"
                    and item.get("name") in self.custom):
                item_id = item.get("id")
                if (not isinstance(item_id, str) or not item_id or item_id in calls
                        or not isinstance(item.get("call_id"), str) or not item["call_id"]
                        or type(event.get("output_index")) is not int):
                    raise _bad_call()
                budget.start()
                budget.append("", item_id)
                budget.append("", item["call_id"])
                calls[item_id] = {"arguments": "", "done": False, "name": item["name"],
                                  "call_id": item.get("call_id"), "output_index": event.get("output_index")}
                item = {k: v for k, v in item.items() if k != "arguments"}
                item.update(type="custom_tool_call", id=_custom_id(item_id), input="")
                yield restore_namespaces({**event, "item": item}, self.namespaces)
                continue
            item_id = event.get("item_id")
            call = calls.get(item_id) if isinstance(item_id, str) else None
            if call is not None and kind == "response.function_call_arguments.delta":
                if call["done"] or event.get("output_index") != call["output_index"]:
                    raise _bad_call()
                delta = event.get("delta")
                budget.append(call["arguments"], delta)
                call["arguments"] += delta
                continue
            if call is not None and kind == "response.function_call_arguments.done":
                if call["done"] or event.get("output_index") != call["output_index"]:
                    raise _bad_call()
                arguments = event.get("arguments", call["arguments"])
                if not call["arguments"]:
                    budget.append("", arguments)
                text = _decode(arguments)
                if call["arguments"] and arguments != call["arguments"]:
                    raise _bad_call()
                call["done"] = True
                call["input"] = text
                common = {k: v for k, v in event.items() if k not in ("type", "arguments")}
                common["item_id"] = _custom_id(item_id)
                yield {**common, "type": "response.custom_tool_call_input.delta", "delta": text}
                yield {**common, "type": "response.custom_tool_call_input.done", "input": text}
                continue
            if kind == "response.output_item.done" and item.get("id") in calls:
                if event.get("output_index") != calls[item["id"]]["output_index"]:
                    raise _bad_call()
                self._check_completed_item(item, calls[item["id"]])
            if kind == "response.completed":
                final = event.get("response") or {}
                items = {item.get("id"): item for item in final.get("output") or [] if isinstance(item, dict)}
                if any(item.get("type") == "function_call" and item.get("name") in self.custom
                       and key not in calls for key, item in items.items()):
                    raise _bad_call()
                for key, call in calls.items():
                    if key not in items:
                        raise _bad_call()
                    self._check_completed_item(items[key], call)
                completed = True
            yield self.response(event)
        if calls and not completed:
            raise _bad_call()

    def _check_completed_item(self, item: Mapping[str, Any], call: Mapping[str, Any]) -> None:
        if (not call["done"] or item.get("name") != call["name"] or item.get("call_id") != call["call_id"]
                or item.get("status") not in (None, "completed")
                or _decode(item.get("arguments")) != call["input"]):
            raise _bad_call()

    def chat_stream(self, source: Iterable[Any]) -> Iterator[dict[str, Any]]:
        """Buffer tool calls when emulating custom tools; text remains incremental."""
        if not self.custom and not self.native_custom:
            for value in source:
                yield _jsonable(value)
            return
        calls: dict[tuple[int, int], dict[str, Any]] = {}
        budget = _BufferBudget()
        for raw in source:
            chunk = deepcopy(_jsonable(raw))
            if not isinstance(chunk, dict) or chunk.get("error") is not None:
                yield chunk
                return
            choices = []
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                ci = choice.get("index", 0)
                for tool in delta.pop("tool_calls", []) or []:
                    ti = tool.get("index")
                    if type(ti) is not int or type(ci) is not int:
                        raise _bad_call()
                    key = (ci, ti)
                    if key not in calls:
                        budget.start()
                        calls[key] = {"index": ti, "type": "function", "function": {"name": "", "arguments": ""}}
                    call = calls[key]
                    if tool.get("id"):
                        if call.get("id") not in (None, tool["id"]):
                            raise _bad_call()
                        if "id" not in call:
                            budget.append("", tool["id"])
                        call["id"] = tool["id"]
                    function = tool.get("function") or {}
                    for field in ("name", "arguments"):
                        fragment = function.get(field) or ""
                        budget.append(call["function"][field], fragment)
                        call["function"][field] += fragment
                if choice.get("finish_reason") is not None:
                    pending = [calls.pop(key) for key in sorted(calls) if key[0] == ci]
                    if pending:
                        if any(not isinstance(call.get("id"), str) or not call["id"]
                               or not call["function"]["name"] for call in pending):
                            raise _bad_call()
                        restored = self.response({"choices": [{"message": {"tool_calls": pending}}]})
                        yield {**{k: v for k, v in chunk.items() if k != "usage"}, "choices": [{
                            "index": ci, "delta": {"tool_calls": restored["choices"][0]["message"]["tool_calls"]},
                            "finish_reason": None,
                        }]}
                if delta or choice.get("finish_reason") is not None:
                    choices.append({**choice, "delta": delta})
            if choices or not chunk.get("choices"):
                yield {**chunk, "choices": choices}
        if calls:
            raise _bad_call()


def _custom_id(value: Any) -> Any:
    return "ctc_" + value[3:] if isinstance(value, str) and value.startswith("fc_") else value


class _BufferBudget:
    def __init__(self) -> None:
        self.total = 0
        self.calls = 0

    def start(self) -> None:
        self.calls += 1
        if self.calls > MAX_BUFFERED_TOOL_CALLS:
            raise _bad_call()

    def append(self, previous: str, fragment: Any) -> None:
        if not isinstance(fragment, str):
            raise _bad_call()
        self.total += len(fragment)
        if len(previous) + len(fragment) > MAX_TOOL_ARGUMENTS or self.total > MAX_BUFFERED_TOOL_ARGUMENTS:
            raise _bad_call()


def plan_tools(payload: Mapping[str, Any], ingress: str, target: str) -> ToolPlan:
    plan = ToolPlan(ingress, target, deepcopy(dict(payload)))
    if ingress == target:
        return plan
    if ingress == DIALECT_OPENAI_RESPONSES:
        plan.request, plan.namespaces = flatten_namespaces(plan.request, origins=plan.origins)
        plan.namespace_definitions = {
            tool["name"]: {key: deepcopy(value) for key, value in tool.items() if key != "tools"}
            for tool in payload.get("tools") or []
            if tool.get("type") == "namespace" and isinstance(tool.get("name"), str)
        }
    request = plan.request
    kept, removed_names, declared = [], set(), set()
    emulated = ingress in (DIALECT_OPENAI_CHAT, DIALECT_OPENAI_RESPONSES) and not (
        ingress == DIALECT_OPENAI_CHAT and target == DIALECT_OPENAI_RESPONSES)
    for index, tool in enumerate(request.get("tools") or []):
        if not isinstance(tool, dict):
            raise _invalid()
        path = plan.origins.get(f"tools[{index}]", f"tools[{index}]")
        name = function_name(tool, ingress)
        if (name is None and (tool.get("type") in (None, "function")
                or (ingress == DIALECT_ANTHROPIC and tool.get("type") == "custom"))):
            raise _invalid("function tool definition")
        decision = decide_tool(ingress, target, tool.get("type"), path, function=name is not None)
        plan.decisions.append(decision)
        loss = decision.loss()
        if loss is not None:
            plan.losses.append(loss)
        if decision.outcome == "unsupported":
            if isinstance(tool.get("name"), str):
                removed_names.add(tool["name"])
            continue
        if tool.get("type") == "custom" and ingress != DIALECT_ANTHROPIC:
            spec = _custom_spec(tool, ingress)
            name = spec["name"]
            if emulated:
                plan.custom[name] = deepcopy(tool)
                tool = _function_tool(spec, ingress)
            else:
                plan.native_custom.add(name)
        if name is not None:
            if not name or name in declared:
                raise _invalid("duplicate tool identity")
            declared.add(name)
        kept.append(tool)
    if "tools" in request:
        if kept:
            request["tools"] = kept
        else:
            request.pop("tools")
    _history(plan, removed_names, declared, emulated)
    _choice(plan, declared, removed_names)
    return plan


def _history(plan: ToolPlan, removed: set[str], declared: set[str], emulated: bool) -> None:
    request, ingress = plan.request, plan.ingress
    if ingress == DIALECT_OPENAI_RESPONSES and isinstance(request.get("input"), list):
        custom_call_ids = {item.get("call_id") for item in request["input"]
                           if item.get("type") == "custom_tool_call" and isinstance(item.get("call_id"), str)}
        for index, item in enumerate(request["input"]):
            kind = item.get("type")
            if kind in _NATIVE_RESPONSES_HISTORY:
                raise RouterError(400, "Provider-native tool history requires a native route",
                                  code="unsupported_tool_history")
            if kind in ("function_call", "custom_tool_call") and item.get("name") in removed:
                raise RouterError(400, "Conversation history uses a removed tool; use a native route",
                                  code="unsupported_tool_history")
            if kind != "custom_tool_call" or not emulated:
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name:
                raise _invalid("custom tool history")
            if name not in plan.custom:
                if name in declared:
                    raise _invalid("custom tool history identity")
                plan.custom[name] = {"type": "custom", "name": name}
                decision = decide_tool(ingress, plan.target, "custom", f"input[{index}]")
                plan.decisions.append(decision)
                plan.losses.append(decision.loss())
            item["type"] = "function_call"
            item["arguments"] = _encode(item.pop("input", None))
        if emulated:
            for item in request["input"]:
                if item.get("type") == "custom_tool_call_output":
                    if not isinstance(item.get("call_id"), str) or item["call_id"] not in custom_call_ids:
                        raise _invalid("custom tool result history")
                    item["type"] = "function_call_output"
    elif ingress in (DIALECT_OPENAI_CHAT, DIALECT_ANTHROPIC):
        for mi, message in enumerate(request.get("messages") or []):
            for call in message.get("tool_calls") or []:
                spec = call.get("custom") if call.get("type") == "custom" else call.get("function")
                if isinstance(spec, dict) and spec.get("name") in removed:
                    raise RouterError(400, "Conversation history uses a removed tool; use a native route",
                                      code="unsupported_tool_history")
                if call.get("type") != "custom" or not emulated:
                    if call.get("type") == "custom" and isinstance(spec, dict) and isinstance(spec.get("name"), str):
                        plan.native_custom.add(spec["name"])
                    continue
                if not isinstance(spec, dict) or not isinstance(spec.get("name"), str):
                    raise _invalid("custom tool history")
                name = spec["name"]
                if name not in plan.custom:
                    if name in declared:
                        raise _invalid("custom tool history identity")
                    plan.custom[name] = {"type": "custom", "custom": {"name": name}}
                    decision = decide_tool(ingress, plan.target, "custom", f"messages[{mi}].tool_calls")
                    plan.decisions.append(decision)
                    plan.losses.append(decision.loss())
                call.pop("custom")
                call.update(type="function", function={"name": name, "arguments": _encode(spec.get("input"))})
            if isinstance(message.get("content"), list):
                for block in message["content"]:
                    if (block.get("type") == "server_tool_use"
                            or str(block.get("type", "")).endswith("_tool_result")):
                        raise RouterError(400, "Provider-native tool history requires a native route",
                                          code="unsupported_tool_history")
                    if block.get("type") == "tool_use" and block.get("name") in removed:
                        raise RouterError(400, "Conversation history uses a removed tool; use a native route",
                                          code="unsupported_tool_history")


def _choice(plan: ToolPlan, names: set[str], removed: set[str]) -> None:
    request = plan.request
    choice = request.get("tool_choice")
    if (plan.ingress == DIALECT_ANTHROPIC and isinstance(choice, dict)
            and "disable_parallel_tool_use" in choice and type(choice["disable_parallel_tool_use"]) is not bool):
        raise _invalid("tool_choice.disable_parallel_tool_use")
    if not request.get("tools"):
        for key in ("tool_choice", "parallel_tool_calls"):
            if request.get(key) is not None:
                request.pop(key)
                plan.losses.append({"field": key, "action": "dropped",
                                    "reason": "every tool definition was removed"})
        return
    if not isinstance(choice, dict):
        return
    kind = choice.get("type")
    if kind in (None, "auto", "none", "required", "any"):
        return
    if kind == "allowed_tools":
        # There is no reviewed adapter contract for a list of allowed tools.
        # Never let an unexamined nested custom choice disappear in an adapter.
        request.pop("tool_choice")
        plan.losses.append({"field": "tool_choice", "action": "dropped",
                            "reason": "allowed-tools choice has no reviewed bridge mapping"})
        return
    name = _choice_name(choice)
    if name is not None and not isinstance(name, str):
        raise _invalid("tool_choice identity")
    if (plan.ingress in (DIALECT_OPENAI_CHAT, DIALECT_OPENAI_RESPONSES)
            and kind in ("function", "custom") and name in names
            and (kind == "custom") != (name in plan.custom or name in plan.native_custom)):
        raise _invalid("tool_choice type")
    if (kind == "custom" and name in names and plan.ingress == DIALECT_OPENAI_CHAT
            and plan.target == DIALECT_OPENAI_RESPONSES):
        # LiteLLM's Chat validator rejects custom choices before its Responses
        # converter runs. Send the reviewed Responses shape as a body field.
        request.pop("tool_choice")
        plan.body["tool_choice"] = {"type": "custom", "name": name}
        return
    if kind == "custom" and name in names and name in plan.custom:
        request["tool_choice"] = ({"type": "function", "function": {"name": name}}
                                  if plan.ingress == DIALECT_OPENAI_CHAT else {"type": "function", "name": name})
        return
    if kind in ("function", "tool", "custom") and name in names:
        return
    if name not in removed and kind in ("function", "tool", "custom"):
        raise _invalid("tool_choice identity")
    if plan.ingress == DIALECT_ANTHROPIC and "disable_parallel_tool_use" in choice:
        request["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": choice["disable_parallel_tool_use"]}
        action = "approximated"
    else:
        request.pop("tool_choice")
        action = "dropped"
    plan.losses.append({"field": "tool_choice", "reason": "forced tool was removed", "action": action})
