"""Ingress validation and strict/lossy translation policy, independent of HTTP handlers."""
from __future__ import annotations

import math
import re
from typing import Any, Mapping

import router_metadata

from .adapters import adapter
from .common import RouterError, _thaw
from .config import Provider, Route
from .constants import (
    _GENERATION_FIELDS,
    DIALECT_ANTHROPIC,
    DIALECT_GEMINI,
    DIALECT_OPENAI_CHAT,
    DIALECT_OPENAI_RESPONSES,
)


def _generation_error(field: str) -> RouterError:
    # Do not echo untrusted values (or arbitrary field names) into diagnostics.
    return RouterError(400, f"Unsupported or invalid {field} for translation",
                       code="invalid_generation_parameter")


def _validate_ingress(
    ingress: str, payload: Mapping[str, Any], *, translated: bool = False,
) -> None:
    """Validate protocol shapes without closing native vendor JSON namespaces.

    Translation has a separate closed generation boundary in _clean_request_params.
    This layer never promotes vendor fields to adapter execution settings.
    """
    def invalid(field: str, expectation: str) -> None:
        raise RouterError(400, f"'{field}' {expectation}", code="invalid_request")

    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise RouterError(400, "A model ID is required", code="model_required")

    def finite(value: Any) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            invalid("body", "must contain finite JSON numbers")
        if isinstance(value, Mapping):
            for item in value.values():
                finite(item)
        elif isinstance(value, list):
            for item in value:
                finite(item)

    finite(payload)
    fields: dict[str, tuple[type, ...]] = {
        "stream": (bool,), "temperature": (int, float), "top_p": (int, float),
        "tools": (list,), "tool_choice": (str, dict), "metadata": (dict,),
    }
    if ingress == DIALECT_OPENAI_CHAT:
        fields.update({
            "messages": (list,), "audio": (dict,), "frequency_penalty": (int, float),
            "function_call": (str, dict), "functions": (list,), "logit_bias": (dict,),
            "logprobs": (bool,), "max_completion_tokens": (int,), "max_tokens": (int,),
            "modalities": (list,), "n": (int,), "parallel_tool_calls": (bool,),
            "prediction": (dict,), "presence_penalty": (int, float),
            "reasoning_effort": (str,), "response_format": (dict,), "seed": (int,),
            "stop": (str, list), "store": (bool,), "stream_options": (dict,),
            "top_logprobs": (int,), "verbosity": (str,), "web_search_options": (dict,),
        })
    elif ingress == DIALECT_OPENAI_RESPONSES:
        fields.update({
            "background": (bool,), "conversation": (str, dict), "include": (list,),
            "input": (str, list), "instructions": (str,), "max_output_tokens": (int,),
            "max_tool_calls": (int,), "parallel_tool_calls": (bool,),
            "previous_response_id": (str,), "prompt": (dict,), "reasoning": (dict,),
            "store": (bool,), "stream_options": (dict,), "text": (dict,),
            "top_logprobs": (int,), "truncation": (str,),
        })
    else:
        fields.update({
            "messages": (list,), "max_tokens": (int,), "system": (str, list),
            "top_k": (int,), "stop_sequences": (list,), "thinking": (dict,),
            "tool_choice": (dict,), "output_format": (dict,), "output_config": (dict,),
            "context_management": (dict,),
        })
    if ingress != DIALECT_ANTHROPIC:
        fields.update({name: (str,) for name in (
            "prompt_cache_key", "prompt_cache_retention", "safety_identifier", "service_tier", "user",
        )})
    for field, types in fields.items():
        if field not in payload:
            continue
        value = payload[field]
        # OpenAI optional fields are generally nullable; stream is deliberately
        # an explicit boolean for every ingress, never a truthiness switch.
        if value is None and field != "stream" and ingress != DIALECT_ANTHROPIC:
            continue
        if type(value) not in types:
            invalid(field, "must be " + "/".join(t.__name__ for t in types))
        if field in {"max_tokens", "max_completion_tokens", "max_output_tokens", "max_tool_calls", "n"}:
            if value <= 0:
                invalid(field, "must be a positive integer")
        if field == "top_p" and not 0 <= value <= 1:
            invalid(field, "must be between 0 and 1")
        if field == "temperature" and not 0 <= value <= (1 if ingress == DIALECT_ANTHROPIC else 2):
            invalid(field, "is outside the supported numeric range")

    if ingress != DIALECT_OPENAI_RESPONSES:
        messages = payload.get("messages")
        if not isinstance(messages, list):
            invalid("messages", "must be an array")
        for message in messages:
            if not isinstance(message, dict) or not isinstance(message.get("role"), str):
                invalid("messages", "must contain objects with string roles")
            content = message.get("content")
            if content is not None and not isinstance(content, (str, list)):
                invalid("messages.content", "must be a string, array or null")
            if isinstance(content, list) and any(not isinstance(block, dict) for block in content):
                invalid("messages.content", "must contain content block objects")
    elif isinstance(payload.get("input"), list):
        for item in payload["input"]:
            if not isinstance(item, dict):
                invalid("input", "must contain input item objects")
            for field in ("type", "role"):
                if field in item and not isinstance(item[field], str):
                    invalid(f"input.{field}", "must be a string")
            if "content" in item and not isinstance(item["content"], (str, list)):
                invalid("input.content", "must be a string or array")
            if isinstance(item.get("content"), list) and any(
                not isinstance(block, dict) for block in item["content"]
            ):
                invalid("input.content", "must contain content block objects")

    for field in ("tools", "functions"):
        if isinstance(payload.get(field), list) and any(not isinstance(item, dict) for item in payload[field]):
            invalid(field, "must contain objects")
    for field in ("include", "modalities", "stop_sequences", "stop"):
        if isinstance(payload.get(field), list) and any(not isinstance(item, str) for item in payload[field]):
            invalid(field, "must contain strings")
    options = payload.get("stream_options")
    if isinstance(options, dict):
        for field in ("include_usage", "include_obfuscation"):
            if field in options and type(options[field]) is not bool:
                invalid(f"stream_options.{field}", "must be bool")

    if ingress == DIALECT_OPENAI_RESPONSES:
        if payload.get("previous_response_id") is not None and payload.get("conversation") is not None:
            invalid("previous_response_id", "cannot be combined with conversation")
        if translated:
            if (any(payload.get(field) is not None for field in (
                "previous_response_id", "conversation", "prompt",
            )) or payload.get("background") is True):
                raise RouterError(400, "Stateful Responses, stored prompts and background execution "
                                  "require a native Responses route. For translation, send the "
                                  "complete conversation in 'input'.", code="unsupported_responses_state")
            if not isinstance(payload.get("input"), (str, list)):
                invalid("input", "is required for translated Responses requests (string or array)")


def _closed_generation_object(
    value: Any, fields: Mapping[str, type], label: str
) -> None:
    if not isinstance(value, dict) or value.keys() - fields.keys():
        raise _generation_error(label)
    for key, item in value.items():
        # bool is an int subclass, but is not a generation budget.
        if type(item) is not fields[key]:
            raise _generation_error(label)


def _validate_thinking(value: Any) -> None:
    _closed_generation_object(value, {
        "type": str, "budget_tokens": int, "display": str, "summary": str,
    }, "thinking")


def _validated_generation_extensions(value: Any) -> dict[str, Any]:
    """A closed, typed body-extension API, never arbitrary LiteLLM kwargs.

    No generic dict/list extension is admitted: adapters sometimes flatten
    extra_body into other argument dictionaries, including on bridge paths.
    """
    if not isinstance(value, dict) or value.keys() - {
        "top_k", "min_p", "repetition_penalty", "thinking",
    }:
        raise _generation_error("extra_body generation extension")
    for key, item in value.items():
        if key == "thinking":
            _validate_thinking(item)
        elif key == "top_k":
            if type(item) is not int or item < 0:
                raise _generation_error("extra_body.top_k")
        elif (type(item) not in (int, float)
              or (type(item) is float and not math.isfinite(item))
              or (key == "min_p" and not 0 <= item <= 1)
              or (key == "repetition_penalty" and item <= 0)):
            raise _generation_error("extra_body sampling parameter")
    return dict(value)


def _clean_request_params(
    payload: Mapping[str, Any], ingress: str
) -> dict[str, Any]:
    allowed = _GENERATION_FIELDS[ingress]
    if "metadata" in payload and ingress != DIALECT_ANTHROPIC:
        # Even body metadata can be promoted to LiteLLM execution context by
        # protocol bridges. Until there is a body-only adapter contract, fail
        # closed rather than letting model_group/guardrails affect execution.
        raise RouterError(400, "OpenAI metadata requires a same-protocol route",
                          code="invalid_generation_parameter")
    if payload.keys() - allowed - {"model", "extra_body"}:
        raise _generation_error("request field")
    params = {key: value for key, value in payload.items() if key in allowed}
    extensions = _validated_generation_extensions(payload.get("extra_body", {}))

    # These objects are interpreted by adapters, rather than opaque prompt/tool
    # schema data. Keep their members generation-only as well.
    for field in ("reasoning", "reasoning_effort"):
        value = params.get(field)
        if value is not None and not (field == "reasoning_effort" and isinstance(value, str)):
            _closed_generation_object(value, {
                "effort": str, "summary": str, "generate_summary": str,
            }, field)
    if params.get("thinking") is not None:
        _validate_thinking(params["thinking"])
    if params.get("output_config") is not None:
        _closed_generation_object(params["output_config"], {
            "effort": str, "format": dict,
        }, "output_config")

    metadata = params.get("metadata")
    if metadata is not None:
        if (not isinstance(metadata, dict)
            or any(not isinstance(key, str) or not isinstance(item, str)
                   for key, item in metadata.items())
            or (ingress == DIALECT_ANTHROPIC and metadata.keys() - {"user_id"})):
            raise _generation_error("metadata")
    if extensions:
        params["extra_body"] = extensions
    return params


_SAFE_LABEL = re.compile(r"[A-Za-z0-9_.:-]{1,64}")


def _label(value: Any) -> str:
    # Tool types and names are client data; only echo plain identifiers.
    return value if isinstance(value, str) and _SAFE_LABEL.fullmatch(value) else "unknown"


def _function_name(tool: Mapping[str, Any], ingress: str) -> str | None:
    """The function name when this tool is a translatable function, else None."""
    kind = tool.get("type")
    if ingress == DIALECT_ANTHROPIC:
        # Typed Anthropic tools (bash_*, text_editor_*, web_search_*, ...) are
        # provider-native; only plain/custom tools are JSON-schema functions.
        name = tool.get("name") if kind in (None, "custom") else None
    elif kind in (None, "function"):
        spec = tool.get("function") if ingress == DIALECT_OPENAI_CHAT else tool
        name = spec.get("name") if isinstance(spec, Mapping) else None
    else:
        name = None
    return name if isinstance(name, str) else None


def _choice_names(choice: Mapping[str, Any]) -> list[str | None] | None:
    """Tool names a dict tool_choice refers to; None for choices that name no tool.

    A None entry stands for a custom, hosted or unknown tool.
    """
    kind = choice.get("type")
    if kind in (None, "auto", "none", "required", "any", "allowed_tools"):
        # allowed_tools has no adapter mapping with or without native tools; not ours to rewrite.
        return None
    inner = choice.get("function") if isinstance(choice.get("function"), Mapping) else choice
    if kind in ("function", "tool") and isinstance(inner.get("name"), str):
        return [inner["name"]]
    return [None]


def _reconcile_tools(params: dict[str, Any], ingress: str) -> list[dict[str, str]]:
    """Keep translatable function tools, remove each native one, fix tool_choice.

    Returns one loss record per removed tool. tool_choice is only touched when
    it would refer to a tool that was removed; parallel-tool controls are kept
    while any function tool remains.
    """
    tools = params["tools"]
    kept: list[Any] = []
    names: set[str] = set()
    losses: list[dict[str, str]] = []
    for index, tool in enumerate(tools):
        name = _function_name(tool, ingress) if isinstance(tool, Mapping) else None
        if name is not None:
            kept.append(tool)
            names.add(name)
            continue
        kind = tool.get("type") if isinstance(tool, Mapping) else None
        losses.append({"field": f"tools[{index}]", "action": "dropped",
                       "reason": f"native tool type '{_label(kind)}' has no equivalent on this target "
                                 "and requires a native route"})
    if not losses:
        return []

    def lose_choice(reason: str, action: str = "dropped") -> None:
        losses.append({"field": "tool_choice", "reason": reason, "action": action})

    choice = params.get("tool_choice")
    if not kept:
        params.pop("tools")
        for key in ("tool_choice", "parallel_tool_calls"):
            if params.get(key) is not None:
                params.pop(key)
                losses.append({"field": key, "action": "dropped",
                               "reason": "every tool definition was removed"})
        return losses
    params["tools"] = kept
    if isinstance(choice, dict):
        refs = _choice_names(choice)
        if refs is None or all(ref in names for ref in refs):
            return losses
        if ingress == DIALECT_ANTHROPIC and "disable_parallel_tool_use" in choice:
            # Keep the parallel constraint; a forced call to a removed tool cannot be honored.
            params["tool_choice"] = {"type": "auto",
                                     "disable_parallel_tool_use": choice["disable_parallel_tool_use"]}
            lose_choice("forced tool was removed", "approximated")
        else:
            params.pop("tool_choice")
            lose_choice("forced tool was removed")
    return losses


def _prepare_translation(
    provider: Provider, route: Route, ingress: str, payload: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Conservative, body-only bridge contract; unknown model facts are neutral.

    An adapter's supported-parameter list describes its mapping, not proof of
    model support. Fields outside the reviewed bridge surface cannot silently
    disappear, even when LiteLLM accepts them through **kwargs.
    """
    params = _thaw(_clean_request_params(payload, ingress))
    losses: list[dict[str, str]] = []

    def value(path: str) -> Any:
        node: Any = contract
        for key in path.split("."):
            node = node.get(key, {})
        return node.get("value")

    def lose(path: str, reason: str, *, approximate: bool = False) -> None:
        losses.append({"field": path, "reason": reason,
                       "action": "approximated" if approximate else "dropped"})
        if not approximate:
            node = params
            parts = path.split(".")
            for key in parts[:-1]:
                node = node.get(key, {})
            node.pop(parts[-1], None)

    if route.dialect == DIALECT_GEMINI:
        # Anthropic cache hints make LiteLLM create Gemini cachedContents via
        # additional API calls. The generation bridge does not expose that
        # service (and gateways commonly do not implement it). Keep the prompt
        # and tools intact, and diagnose the lost caching optimization.
        cache_removed = False

        def strip_cache(node: Any) -> None:
            nonlocal cache_removed
            if isinstance(node, list):
                for item in node:
                    strip_cache(item)
            elif isinstance(node, dict):
                if "cache_control" in node:
                    node.pop("cache_control")
                    cache_removed = True
                # Only protocol content blocks; never traverse tool schemas or
                # user-supplied tool arguments, which may name cache_control.
                if "content" in node:
                    strip_cache(node["content"])

        for field in ("messages", "system", "tools"):
            strip_cache(params.get(field))
        if cache_removed:
            lose("cache_control", "target cache service is outside the generation bridge")

    # These bridge fields have an explicit mapping in the installed adapters.
    # Native fields outside this set require their native endpoint.
    if ingress == DIALECT_OPENAI_CHAT:
        mapped = set(_GENERATION_FIELDS[ingress])
        if route.dialect == DIALECT_OPENAI_RESPONSES:
            mapped &= {"messages", "max_tokens", "max_completion_tokens", "stream",
                       "stream_options", "tools", "tool_choice", "parallel_tool_calls",
                       "response_format", "reasoning_effort", "verbosity", "temperature",
                       "top_p", "user", "store", "service_tier", "safety_identifier",
                       "prompt_cache_key", "prompt_cache_retention", "web_search_options"}
    elif ingress == DIALECT_OPENAI_RESPONSES:
        mapped = {"input", "instructions", "max_output_tokens", "stream", "tools",
                  "tool_choice", "parallel_tool_calls", "reasoning", "text",
                  "temperature", "top_p", "user", "service_tier", "safety_identifier"}
    else:
        mapped = {"messages", "system", "max_tokens", "stream", "temperature", "top_p",
                  "top_k", "stop_sequences", "tools", "tool_choice", "thinking",
                  "metadata", "output_format", "output_config"}
        if route.dialect == DIALECT_OPENAI_RESPONSES:
            mapped -= {"top_k", "stop_sequences"}
    for key, item in list(params.items()):
        if item is not None and key not in mapped | {"extra_body"}:
            lose(key, "no reviewed bridge mapping")

    # Tri-state facts: only explicit false/known enums reject a request. Unknown
    # metadata never prevents an otherwise available adapter mapping.
    for key in list(params):
        if key in router_metadata.SAMPLING and value(f"sampling.{key}.supported") is False:
            lose(key, "model capability is unsupported")
    for key in list(params.get("extra_body", {})):
        fact_key = "repeat_penalty" if key == "repetition_penalty" else key
        if fact_key in router_metadata.SAMPLING and value(f"sampling.{fact_key}.supported") is False:
            lose(f"extra_body.{key}", "model capability is unsupported")

    for key in ("tools", "tool_choice", "functions", "function_call"):
        if params.get(key) is not None and value("tools.supported") is False:
            lose(key, "model capability is unsupported")
    if params.get("parallel_tool_calls") is True and value("tools.parallel") is False:
        lose("parallel_tool_calls", "model capability is unsupported")
    # Provider-native tools have no equivalent on the target; function tools do.
    # Reconcile them one by one so a single native tool never discards the rest.
    if params.get("tools") and (route.dialect != DIALECT_OPENAI_RESPONSES
                                or ingress == DIALECT_ANTHROPIC):
        losses.extend(_reconcile_tools(params, ingress))
    if (params.get("tool_choice") is not None or params.get("parallel_tool_calls") is not None
            ) and not params.get("tools"):
        for key in ("tool_choice", "parallel_tool_calls"):
            if params.get(key) is not None:
                lose(key, "bridge requires tools for this control")
    if (ingress == DIALECT_ANTHROPIC and route.dialect == DIALECT_OPENAI_RESPONSES
            and isinstance(params.get("tool_choice"), dict)
            and "disable_parallel_tool_use" in params["tool_choice"]):
        lose("tool_choice.disable_parallel_tool_use", "parallel tool constraint has no bridge mapping")

    reasoning_key = "reasoning_effort" if ingress == DIALECT_OPENAI_CHAT else "reasoning"
    reasoning = params.get(reasoning_key)
    effort = reasoning.get("effort") if isinstance(reasoning, dict) else reasoning
    if reasoning is not None:
        if value("reasoning.supported") is False or value("reasoning.effort.supported") is False:
            lose(reasoning_key, "model capability is unsupported")
        elif effort is not None and value("reasoning.effort.values") is not None and effort not in value("reasoning.effort.values"):
            lose(reasoning_key, "effort is outside the model's known values")
        elif effort == "none" and route.dialect == DIALECT_ANTHROPIC:
            lose(reasoning_key, "adapter omits this control instead of explicitly disabling thinking")
        elif isinstance(reasoning, dict) and route.dialect != DIALECT_OPENAI_RESPONSES:
            for key in list(reasoning):
                if key != "effort":
                    lose(f"{reasoning_key}.{key}", "reasoning detail has no reviewed bridge mapping")

    thinking_path = "thinking" if "thinking" in params else "extra_body.thinking"
    thinking = params.get("thinking", params.get("extra_body", {}).get("thinking"))
    if thinking is not None:
        if value("reasoning.supported") is False:
            lose(thinking_path, "model capability is unsupported")
        elif (thinking.get("budget_tokens") is not None
              and value("reasoning.budget.supported") is False):
            lose(thinking_path, "model budget control is unsupported")
        elif (thinking.get("budget_tokens") is not None
              and ((value("reasoning.budget.min_tokens") is not None
                    and thinking["budget_tokens"] < value("reasoning.budget.min_tokens"))
                   or (value("reasoning.budget.max_tokens") is not None
                       and thinking["budget_tokens"] > value("reasoning.budget.max_tokens")))):
            lose(thinking_path, "budget is outside the model's known bounds")
        elif route.dialect != DIALECT_ANTHROPIC:
            if value("reasoning.effort.supported") is False:
                lose(thinking_path, "target effort control is unsupported")
            elif ingress == DIALECT_ANTHROPIC and thinking.get("type") == "enabled":
                # LiteLLM maps a token budget to an effort bucket. Keep that
                # useful mapping only when the operator explicitly accepts loss.
                lose(thinking_path, "token budget becomes an approximate effort", approximate=True)
            else:
                lose(thinking_path, "native thinking control requires a native route")
        elif route.dialect == DIALECT_ANTHROPIC and thinking.get("type") == "adaptive":
            lose(thinking_path, "adaptive thinking may become a token budget", approximate=True)

    formats = [("response_format", params.get("response_format")),
               ("text.format", (params.get("text") or {}).get("format")),
               ("output_format", params.get("output_format")),
               ("output_config.format", (params.get("output_config") or {}).get("format"))]
    for path, fmt in formats:
        if fmt is None:
            continue
        if not isinstance(fmt, dict):
            raise _generation_error("output format")
        if fmt.get("type") == "text":
            continue
        kind = fmt.get("type")
        if (value("structured_output.supported") is False
                or value(f"structured_output.{kind}") is False):
            lose(path, "model capability is unsupported")
        elif route.dialect == DIALECT_ANTHROPIC:
            # LiteLLM can emulate JSON via a synthetic tool on some models;
            # that is not equivalent to a native constrained output contract.
            lose(path, "native structured output is not guaranteed by this bridge")
        elif kind not in ("json_object", "json_schema"):
            lose(path, "output format has no reviewed bridge mapping")
    if params.get("text"):
        for key in list(params["text"]):
            if key != "format":
                # Unknown nested keys must never become diagnostic/header data.
                lose("text", "text controls beyond format have no reviewed bridge mapping")
                break
    if (params.get("output_config") or {}).get("effort") is not None:
        lose("output_config.effort", "native effort control has no reviewed bridge mapping")

    # A provider's Chat mapping table also applies after Responses/Messages
    # normalization. Responses-native calls do not go through this table.
    if route.dialect != DIALECT_OPENAI_RESPONSES:
        supported = adapter.supported_openai_params(
            model=route.upstream_model, custom_llm_provider=provider.litellm_provider)
        aliases = {"max_output_tokens": "max_tokens", "stop_sequences": "stop",
                   "reasoning": "reasoning_effort", "output_format": "response_format",
                   "text": "response_format"}
        special = {"messages", "input", "instructions", "system", "extra_body",
                   "metadata", "output_config", "top_k"}
        # include_usage is implemented by the common streaming wrapper even
        # when the target has no native stream_options field.
        if (ingress == DIALECT_OPENAI_CHAT and isinstance(params.get("stream_options"), dict)
                and params["stream_options"].keys() <= {"include_usage"}):
            special.add("stream_options")
        for key, item in list(params.items()):
            if item is not None and key not in special and supported is not None:
                mapped_key = aliases.get(key, key)
                if key == "thinking" and ingress == DIALECT_ANTHROPIC:
                    mapped_key = "reasoning_effort"
                if mapped_key not in supported:
                    lose(key, "target adapter has no mapping for this control")
        if ("thinking" in params.get("extra_body", {}) and supported is not None
                and "thinking" not in supported):
            lose("extra_body.thinking", "target adapter has no mapping for this control")

    if losses and not provider.drop_params:
        detail = "; ".join(dict.fromkeys(
            f"{loss['field']} ({loss['reason']})" for loss in losses))
        raise RouterError(400, f"Translation would lose or approximate: {detail}. "
                          "Use a same-protocol route, remove these controls, or ask the operator "
                          "to configure compatibility = 'lossy' (with loss diagnostics).",
                          code="translation_loss")
    return params, losses
