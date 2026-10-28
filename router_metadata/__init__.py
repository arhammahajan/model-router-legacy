"""Version 1 advisory capability metadata; never used to select a transport."""
from __future__ import annotations

import copy
import json
import math
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, TypeGuard

MODALITIES = ("text", "image", "audio", "video", "pdf", "embedding")
PROTOCOLS = ("openai-chat", "openai-responses", "anthropic", "gemini")
SAMPLING = (
    "temperature", "top_p", "top_k", "min_p", "typical_p", "seed",
    "frequency_penalty", "presence_penalty", "repeat_penalty", "repeat_last_n",
    "dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n",
    "mirostat", "mirostat_tau", "mirostat_eta", "dynatemp_range", "dynatemp_exponent",
    "xtc_probability", "xtc_threshold", "samplers",
    "top_n_sigma", "adaptive_target", "adaptive_decay", "dry_sequence_breakers",
)
# Effort levels the router maps to llama.cpp thinking controls; see
# model_router._apply_llama_thinking. "max" is absent: llama.cpp has no maximum
# level, so a translated request for it is reported instead of sending nothing.
LLAMA_EFFORT_VALUES = ("none", "minimal", "low", "medium", "high", "xhigh")
# Canonical native levels in PolyStrata's serve/frontend.py effort_kwargs
# (reviewed against 0.1.45). Other accepted spellings are aliases, not levels.
STRATA_EFFORT_VALUES = ("none", "low", "medium", "high")
LIMITS = ("context_window","trained_context_window", "max_input_tokens",
          "max_output_tokens", "default_output_tokens")


def _boolean(value: Any) -> bool:
    return type(value) is bool


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _strings(value: Any) -> TypeGuard[list[str] | tuple[str, ...]]:
    return (isinstance(value, (list, tuple)) and all(isinstance(v, str) and v for v in value)
            and len(set(value)) == len(value))


def _native(value: Any) -> bool:
    return (isinstance(value, Mapping) and not value.keys() - {"protocol", "path", "values"}
            and value.get("protocol") in PROTOCOLS
            and isinstance(value.get("path"), str) and bool(value["path"])
            and ("values" not in value or (isinstance(value["values"], Mapping)
                 and all(isinstance(k, str) and type(v) in (str, bool, int)
                         for k, v in value["values"].items()))))


# Every leaf is a fact, with the same value/source/kind shape as prompt 08.
FIELDS: dict[str, Any] = {
    "reasoning.supported": _boolean,
    "reasoning.toggle.supported": _boolean,
    "reasoning.toggle.default": _boolean,
    "reasoning.toggle.native_parameter": _native,
    "reasoning.effort.supported": _boolean,
    "reasoning.effort.values": _strings,
    "reasoning.effort.default": lambda v: isinstance(v, str) and bool(v),
    "reasoning.effort.native_parameter": _native,
    "reasoning.budget.supported": _boolean,
    "reasoning.budget.native_parameter": _native,
    "tools.supported": _boolean,
    "tools.parallel": _boolean,
    "structured_output.supported": _boolean,
    "structured_output.json_object": _boolean,
    "structured_output.json_schema": _boolean,
    "streaming": _boolean,
    "protocols.known_lossy_features": _strings,
}
for _key in ("min_tokens", "max_tokens", "default_tokens"):
    FIELDS[f"reasoning.budget.{_key}"] = lambda v: type(v) is int and v >= 0
for _direction in ("input", "output"):
    for _modality in MODALITIES:
        FIELDS[f"modalities.{_direction}.{_modality}"] = _boolean
for _protocol in PROTOCOLS:
    FIELDS[f"protocols.native.{_protocol}"] = _boolean
    FIELDS[f"protocols.translated.{_protocol}"] = _boolean
for _key in SAMPLING:
    FIELDS[f"sampling.{_key}.supported"] = _boolean
    FIELDS[f"sampling.{_key}.default"] = _strings if _key in ("samplers", "dry_sequence_breakers") else _number
    FIELDS[f"sampling.{_key}.native_parameter"] = _native
    FIELDS[f"sampling.{_key}.constraints"] = _strings


def _parent(obj: dict[str, Any], path: str) -> tuple[dict[str, Any], str]:
    parts = path.split(".")
    for part in parts[:-1]:
        obj = obj.setdefault(part, {})
    return obj, parts[-1]


def fact(value: Any = None, source: str | None = None, kind: str = "unknown") -> dict[str, Any]:
    return {"value": copy.deepcopy(value), "source": source, "kind": kind}


def put(contract: dict[str, Any], path: str, value: Any, source: str, kind: str) -> None:
    """Keep displaced evidence for conflicts, including explicit null overrides."""
    parent, key = _parent(contract, path)
    previous = parent.get(key, fact())
    result = fact(value, source, kind)
    conflicts = copy.deepcopy(previous.get("conflicts", []))
    if previous["value"] is not None and previous["value"] != value:
        conflicts.append({k: v for k, v in previous.items() if k != "conflicts"})
    if conflicts:
        result["conflicts"] = conflicts
    parent[key] = result


def empty(dialect: str) -> dict[str, Any]:
    result: dict[str, Any] = {"schema_version": 1, "sources": {}, "limits": {}}
    for path in FIELDS:
        parent, key = _parent(result, path)
        parent[key] = fact()
    result["limits"] = {key: fact() for key in LIMITS}
    # This is the actual configured native transport, not a claim about every
    # upstream endpoint or every feature surviving translation.
    put(result, f"protocols.native.{dialect}", True, "routing.dialect", "configured")
    for protocol in PROTOCOLS:
        put(result, f"protocols.translated.{protocol}",
            protocol != dialect, "routing.adapters", "configured")
    result["sources"]["routing"] = {"verification": "configured transport and adapter paths only"}
    return result


def _flatten(raw: Any, prefix: str = "") -> dict[str, Any]:
    if prefix in FIELDS:
        if raw is not None and not FIELDS[prefix](raw):
            raise ValueError(f"{prefix}: invalid capability value")
        return {prefix: raw}
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"{prefix or 'override'}: expected a nonempty capability object")
    result = {}
    for key, value in raw.items():
        if not isinstance(key, str) or "." in key:
            raise ValueError(f"{prefix}: capability keys must be simple field names")
        path = f"{prefix}.{key}" if prefix else key
        if path not in FIELDS and not any(p.startswith(path + ".") for p in FIELDS):
            raise ValueError(f"{path}: unknown capability field")
        result.update(_flatten(value, path))
    return result


def validate_overrides(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, Mapping):
        raise SystemExit("capabilities.overrides: must be a table/object")
    result = {}
    for model, value in raw.items():
        if not isinstance(model, str) or not model.strip() or model != model.strip():
            raise SystemExit("capabilities.overrides: expected exact nonempty public model IDs")
        try:
            flattened = _flatten(value)
            # Validate related values supplied together; partial overrides are
            # also checked against the final resolved contract by normalize().
            result[model] = flattened
            probe = empty("openai-chat")
            apply_overrides(probe, flattened)
            _check_relations(probe)
        except ValueError as exc:
            raise SystemExit(f"capabilities.overrides[{model!r}]: {exc}") from exc
    return result


def apply_overrides(contract: dict[str, Any], values: Mapping[str, Any]) -> None:
    for path, value in values.items():
        put(contract, path, value, "override." + path, "override")
    if values:
        contract["sources"]["override"] = {"verification": "operator assertion"}


def _get(contract: Mapping[str, Any], path: str) -> Any:
    value = contract
    for key in path.split("."):
        value = value[key]
    return value["value"]


def _check_relations(contract: Mapping[str, Any]) -> None:
    lo, hi = (_get(contract, f"reasoning.budget.{key}_tokens") for key in ("min", "max"))
    if lo is not None and hi is not None and lo > hi:
        raise ValueError("reasoning.budget: min_tokens exceeds max_tokens")
    default = _get(contract, "reasoning.budget.default_tokens")
    if default is not None and ((lo is not None and default < lo) or (hi is not None and default > hi)):
        raise ValueError("reasoning.budget.default_tokens: outside the budget range")
    values, default = (_get(contract, f"reasoning.effort.{key}") for key in ("values", "default"))
    if values is not None and default is not None and default not in values:
        raise ValueError("reasoning.effort.default: absent from effort values")


def normalize(contract: dict[str, Any]) -> None:
    """An explicit negative parent suppresses dependent positive claims.

    Overrides are sparse; disabling reasoning must not leave inherited effort
    values, defaults or mappings active. Conflicting evidence remains visible.
    """
    for root in ("reasoning", "reasoning.toggle", "reasoning.effort", "reasoning.budget",
                 "tools", "structured_output", *(f"sampling.{key}" for key in SAMPLING)):
        supported = _get(contract, root + ".supported")
        if supported is not False:
            continue
        parent, key = _parent(contract, root + ".supported")
        evidence = parent[key]
        for path in FIELDS:
            if not path.startswith(root + ".") or path == root + ".supported":
                continue
            negative = False if path.endswith(".supported") or path in (
                "tools.parallel", "structured_output.json_object", "structured_output.json_schema") else None
            put(contract, path, negative, evidence["source"], evidence["kind"])
    # A sparse override can invalidate an inherited default or bound. Prefer
    # the explicit assertion and withdraw the conflicting derived claim. Bad
    # combinations within the override itself were rejected at config time.
    lo, hi = (_get(contract, f"reasoning.budget.{key}_tokens") for key in ("min", "max"))
    if lo is not None and hi is not None and lo > hi:
        low_fact = contract["reasoning"]["budget"]["min_tokens"]
        key = "max_tokens" if low_fact["kind"] == "override" else "min_tokens"
        put(contract, "reasoning.budget." + key, None, "validation.budget_range", "conflict")
    default = _get(contract, "reasoning.budget.default_tokens")
    lo, hi = (_get(contract, f"reasoning.budget.{key}_tokens") for key in ("min", "max"))
    if default is not None:
        for bound, invalid in (("min_tokens", lo is not None and default < lo),
                               ("max_tokens", hi is not None and default > hi)):
            if invalid:
                key = bound if contract["reasoning"]["budget"]["default_tokens"]["kind"] == "override" else "default_tokens"
                put(contract, "reasoning.budget." + key, None, "validation.budget_default", "conflict")
    values, default = (_get(contract, f"reasoning.effort.{key}") for key in ("values", "default"))
    if values is not None and default is not None and default not in values:
        key = "values" if contract["reasoning"]["effort"]["default"]["kind"] == "override" else "default"
        put(contract, "reasoning.effort." + key, None, "validation.effort_default", "conflict")


@lru_cache(maxsize=1)
def _go_review() -> dict[str, Any]:
    return json.loads(Path(__file__).with_name("opencode-go-v1.json").read_text())


def go(model: str, dialect: str) -> dict[str, Any]:
    """Join exact Go IDs to a frozen provider-specific review, never the network."""
    result = empty(dialect)
    review = _go_review()
    row = review["models"].get(model)
    if row is None:
        return result
    result["sources"]["go.review"] = {
        **review["provenance"], "model_source": row["source"],
        "retrieved_at": review["retrieved_at"],
        "stale_after_seconds": review["stale_after_seconds"],
    }
    # Reviewing capabilities for one native endpoint doesn't establish them on
    # a different operator-selected endpoint. Preserve routing truth and leave
    # capabilities unknown in that case.
    if row["dialect"] != dialect:
        result["sources"]["go.review"]["applicable"] = False
        return result
    result["sources"]["go.review"]["applicable"] = True
    for path, value in row["facts"].items():
        if path.startswith("limits."):
            if value is None or type(value) is not int or value <= 0:
                continue
        elif path not in FIELDS or (value is not None and not FIELDS[path](value)):
            raise ValueError(f"invalid bundled Go review field: {path}")
        put(result, path, value, "go.review." + path, "reviewed")
    return result


def local_engine(props: Any) -> str:
    """Identify Strata from its /props build report, never from a model name."""
    build = props.get("build_info") if isinstance(props, Mapping) else None
    name = build.split(maxsplit=1)[0].lower() if isinstance(build, str) and build.strip() else ""
    if name in ("strata", "polystrata"):
        return "strata"
    return "llama.cpp"


def llama(row: Mapping[str, Any], props: Any, mode: str, limits: Mapping[str, Any],
          retrieved_at: float, *, per_model_props: bool = False,
          engine: str = "llama.cpp") -> dict[str, Any]:
    result = empty("openai-chat")
    result["limits"] = copy.deepcopy(dict(limits))
    source = "llama.props"
    usable_props = per_model_props or mode in ("single", "legacy")
    props = props if usable_props and isinstance(props, Mapping) else {}
    result["sources"]["llama.models"] = {"retrieved_at": retrieved_at, "verification": "upstream report"}
    result["sources"][source] = {
        "retrieved_at": retrieved_at if props else None,
        "endpoint": "/props?model=<id>&autoload=false" if per_model_props else "/props?autoload=false",
        "available": bool(props), "verification": "upstream report",
        "server_build": props.get("build_info") if isinstance(props.get("build_info"), str) else None,
    }

    def report(path: str, value: Any, field: str, origin: str = source, kind: str = "reported") -> None:
        if value is not None and FIELDS[path](value):
            put(result, path, value, origin + "." + field, kind)

    architecture = row.get("architecture")
    if isinstance(architecture, Mapping):
        for direction in ("input", "output"):
            values = architecture.get(direction + "_modalities")
            if _strings(values):
                for modality in MODALITIES:
                    report(f"modalities.{direction}.{modality}", modality in values,
                           "architecture." + direction + "_modalities", "llama.models")
    # /props flags are partial: a missing flag is unknown, never false. These
    # describe input only. A vision projector says nothing about image output.
    modalities = props.get("modalities")
    if isinstance(modalities, Mapping):
        for flag, modality in (("vision", "image"), ("audio", "audio"), ("video", "video")):
            report(f"modalities.input.{modality}", modalities.get(flag), "modalities." + flag)
    if mode == "legacy" and isinstance(row.get("capabilities"), (list, tuple)):
        if "multimodal" in row["capabilities"]:
            report("modalities.input.image", True, "capabilities", "llama.models", "inferred")

    caps = props.get("chat_template_caps")
    if isinstance(caps, Mapping):
        tools = [caps[k] for k in ("supports_tools", "supports_tool_calls") if type(caps.get(k)) is bool]
        if tools:
            report("tools.supported", all(tools), "chat_template_caps.supports_tools/supports_tool_calls")
        report("tools.parallel", caps.get("supports_parallel_tool_calls"),
               "chat_template_caps.supports_parallel_tool_calls")
        effort = caps.get("supports_reasoning_effort")
        report("reasoning.effort.supported", effort, "chat_template_caps.supports_reasoning_effort")
        if effort is True:
            report("reasoning.supported", True, "chat_template_caps.supports_reasoning_effort")
            report("reasoning.effort.native_parameter", {"protocol": "openai-chat", "path": "reasoning_effort"},
                   "chat_template_caps.supports_reasoning_effort", kind="inferred")
        # No effort enum, toggle, budget, or default follows from this boolean.
        # supports_preserve_reasoning concerns history, not a thinking toggle.

    # Independent of the chat template: llama.cpp uses the router's budget map;
    # Strata accepts native effort levels. Neither proves a model produces thinking.
    strata = engine == "strata"
    source_name = "strata.compat" if strata else "router.llama_thinking_map"
    kind = "inferred" if strata else "configured"
    report("reasoning.effort.supported", True, "supported", source_name, kind)
    report("reasoning.effort.values", list(STRATA_EFFORT_VALUES if strata else LLAMA_EFFORT_VALUES),
           "values", source_name, kind)
    report("reasoning.effort.native_parameter", {"protocol": "openai-chat", "path": "reasoning_effort"},
           "native_parameter", source_name, kind)
    result["sources"][source_name] = {
        "verification": ("PolyStrata 0.1.45 serve/frontend.py effort_kwargs; identified by /props.build_info"
                         if strata else "router request mapping, not an upstream report"),
    }
    if strata:
        report("reasoning.toggle.supported", True, "none", source_name, kind)
        report("reasoning.toggle.native_parameter", {
            "protocol": "openai-chat", "path": "reasoning_effort", "values": {"off": "none"},
        }, "none", source_name, kind)

    defaults = props.get("default_generation_settings")
    if isinstance(defaults, Mapping):
        params = defaults.get("params")
        params = params if isinstance(params, Mapping) else defaults
        location = "default_generation_settings." + ("params." if params is not defaults else "")
        for key in SAMPLING:
            if key in params and FIELDS[f"sampling.{key}.default"](params[key]):
                report(f"sampling.{key}.supported", True, location + key, kind="inferred")
                report(f"sampling.{key}.default", params[key], location + key)
                report(f"sampling.{key}.native_parameter", {"protocol": "openai-chat", "path": key},
                       location + key, kind="inferred")
        # The value of stream is a default, not a support flag. Presence of this
        # setting is evidence of streaming, even when the default is false.
        if type(params.get("stream")) is bool:
            report("streaming", True, location + "stream", kind="inferred")
    return result


def render(contract: dict[str, Any], overrides: Mapping[str, Any], discovery: Mapping[str, Any],
           now: float) -> dict[str, Any]:
    """Render a private copy, computing ages at read time without network I/O."""
    result = copy.deepcopy(contract)
    apply_overrides(result, overrides)
    normalize(result)
    result["freshness"] = {"discovery": dict(discovery)}
    for source in result["sources"].values():
        retrieved = source.get("retrieved_at")
        if isinstance(retrieved, str):
            retrieved = datetime.fromisoformat(retrieved.replace("Z", "+00:00")).replace(tzinfo=timezone.utc).timestamp()
        age = max(0, now - retrieved) if type(retrieved) in (int, float) else None
        ttl = source.get("stale_after_seconds")
        source["age_seconds"] = age
        # Static/configured assertions have no observed freshness. The review
        # TTL is independent of how recently availability was refreshed.
        source["stale"] = (age >= ttl if ttl is not None and age is not None else
                           discovery.get("stale") if age is not None else None)
    return result
