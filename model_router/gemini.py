"""Gemini API ingress: SDK paths, request validation, and Gemini <-> Chat translation.

Same-protocol (Gemini -> Gemini) requests never pass through this translation;
they are forwarded as JSON. For every other upstream dialect a Gemini request is
rewritten into OpenAI Chat Completions parameters, which the existing bridge
sends to the destination, and the Chat result is rewritten back into Gemini
response shapes. Anything without an established counterpart is reported as a
loss record (the strict/lossy policy decides) or rejected outright when
dropping it would silently change what the request means.
"""
from __future__ import annotations

import base64
import binascii
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping
from urllib.parse import unquote

from .tools import decide_tool
from .common import RouterError
from .constants import (
    DIALECT_ANTHROPIC,
    DIALECT_OPENAI_CHAT,
    DIALECT_OPENAI_RESPONSES,
)

# -- paths -------------------------------------------------------------------

_API_VERSIONS = ("/v1beta", "/v1")
# method -> streams
GENERATION_METHODS = {"generateContent": False, "streamGenerateContent": True}
# Methods of the public API the router recognizes but does not implement.
KNOWN_UNSUPPORTED_METHODS = frozenset({
    "countTokens", "countMessageTokens", "embedContent", "batchEmbedContents",
    "batchGenerateContent", "asyncBatchEmbedContent", "predict", "predictLongRunning",
    "bidiGenerateContent", "generateAnswer",
})
GENERATION_METHODS_LISTED = ["generateContent", "streamGenerateContent"]
_NAME_OK = re.compile(r"[^\x00-\x1f\x7f]+")


@dataclass(frozen=True)
class GeminiPath:
    kind: str      # "models" or "tunedModels"
    name: str      # percent-decoded resource id; may contain "/" and ":"
    method: str = ""


def _split_version(path: str) -> str | None:
    for prefix in _API_VERSIONS:
        if path == prefix or path.startswith(prefix + "/"):
            return path[len(prefix):]
    return path if path.startswith(("/models/", "/tunedModels/")) else None


def _resource(path: str) -> tuple[str, str] | None:
    rest = _split_version(path)
    if rest is None:
        return None
    for kind in ("models", "tunedModels"):
        head = f"/{kind}/"
        if rest.startswith(head) and len(rest) > len(head):
            return kind, rest[len(head):]
    return None


def parse_method_path(path: str) -> GeminiPath | None:
    """``/v1beta/models/{name}:{method}``; the method follows the *last* colon.

    The SDKs leave "/" in the model unescaped and never escape ":" (the SDK-level
    ``models/`` prefix is the only encoding), but percent-encoded names are
    accepted as well.
    """
    found = _resource(path)
    if found is None:
        return None
    kind, raw = found
    name, sep, method = raw.rpartition(":")
    if not sep or not name or not method or not re.fullmatch(r"[A-Za-z]+", method):
        return None
    name = unquote(name)
    if not _NAME_OK.fullmatch(name):
        return None
    return GeminiPath(kind, name, method)


def parse_model_path(path: str) -> GeminiPath | None:
    """``/v1beta/models/{name}`` (models.get); a trailing ":x" is part of the name."""
    found = _resource(path)
    if found is None:
        return None
    kind, raw = found
    name = unquote(raw)
    return GeminiPath(kind, name) if _NAME_OK.fullmatch(name) else None


def is_list_path(path: str) -> bool:
    return path == "/v1beta/models"


def log_label(method: str, path: str) -> str | None:
    """A fixed, non-identifying label for access logs; never the model or query."""
    if method == "POST":
        parsed = parse_method_path(path)
        if parsed is not None and (parsed.method in GENERATION_METHODS
                                   or parsed.method in KNOWN_UNSUPPORTED_METHODS):
            return f"/v1beta/{parsed.kind}/*:{parsed.method}"
    elif method == "GET":
        if is_list_path(path):
            return "/v1beta/models"
        if path.startswith("/v1beta/") and parse_model_path(path) is not None:
            return "/v1beta/models/*"
    return None


# -- request validation helpers ------------------------------------------------

_LABEL = re.compile(r"[A-Za-z0-9_.:-]{1,64}")


def _label(value: Any) -> str:
    return value if isinstance(value, str) and _LABEL.fullmatch(value) else "unknown"


def _bad(message: str) -> RouterError:
    return RouterError(400, message, code="invalid_request")


def _snake_to_camel(name: str) -> str:
    return re.sub(r"_([a-z0-9])", lambda m: m.group(1).upper(), name)


def _object(value: Any, keys: frozenset[str], where: str) -> dict[str, Any]:
    """A JSON object with camelCase keys; snake_case spellings are accepted, as on the API."""
    if not isinstance(value, Mapping):
        raise _bad(f"'{where}' must be an object")
    result: dict[str, Any] = {}
    for key, item in value.items():
        camel = _snake_to_camel(key) if isinstance(key, str) else key
        if camel not in keys:
            raise _bad(f"Invalid JSON payload received. Unknown name \"{_label(key)}\" "
                       f"at '{where}': Cannot find field.")
        if camel in result:
            raise _bad(f"'{where}' sets {camel} more than once (camelCase and snake_case)")
        result[camel] = item
    return result


def _typed(value: Any, types: tuple[type, ...], where: str) -> Any:
    if type(value) not in types:
        raise _bad(f"'{where}' must be " + "/".join(t.__name__ for t in types))
    return value


def _number(value: Any, where: str, *, low: float | None = None, high: float | None = None) -> float:
    _typed(value, (int, float), where)
    if isinstance(value, float) and not math.isfinite(value):
        raise _bad(f"'{where}' must be finite")
    if (low is not None and value < low) or (high is not None and value > high):
        raise _bad(f"'{where}' is outside the supported numeric range")
    return value


def _int(value: Any, where: str, *, low: int | None = None) -> int:
    _typed(value, (int,), where)
    if low is not None and value < low:
        raise _bad(f"'{where}' must be at least {low}")
    return value


_REQUEST_KEYS = frozenset({
    "contents", "tools", "toolConfig", "safetySettings", "systemInstruction",
    "generationConfig", "cachedContent", "labels", "serviceTier", "store",
    "continuationToken",
})
_CONTENT_KEYS = frozenset({"role", "parts"})
_PART_KEYS = frozenset({
    "text", "inlineData", "fileData", "functionCall", "functionResponse", "thought",
    "thoughtSignature", "executableCode", "codeExecutionResult", "videoMetadata",
    "mediaResolution", "partMetadata", "toolCall", "toolResponse",
    "audioTranscription", "mediaProcessing", "speechMetadata",
})
_DATA_KEYS = ("text", "inlineData", "fileData", "functionCall", "functionResponse",
              "executableCode", "codeExecutionResult", "toolCall", "toolResponse")
_INLINE_KEYS = frozenset({"mimeType", "data", "displayName"})
_FUNCTION_CALL_KEYS = frozenset({"id", "name", "args", "partialArgs", "willContinue"})
_FUNCTION_RESPONSE_KEYS = frozenset({"id", "name", "response", "parts", "willContinue",
                                     "scheduling"})
_GENERATION_KEYS = frozenset({
    "stopSequences", "responseMimeType", "responseSchema", "responseJsonSchema",
    "responseModalities", "candidateCount", "maxOutputTokens", "temperature", "topP",
    "topK", "seed", "presencePenalty", "frequencyPenalty", "responseLogprobs", "logprobs",
    "enableEnhancedCivicAnswers", "speechConfig", "thinkingConfig", "mediaResolution",
    "imageConfig", "audioTimestamp",
})
_THINKING_KEYS = frozenset({"includeThoughts", "thinkingBudget", "thinkingLevel"})
_TOOL_KEYS = frozenset({
    "functionDeclarations", "googleSearchRetrieval", "codeExecution", "googleSearch",
    "computerUse", "urlContext", "fileSearch", "mcpServers", "googleMaps",
})
_DECLARATION_KEYS = frozenset({
    "name", "description", "behavior", "parameters", "parametersJsonSchema", "response",
    "responseJsonSchema",
})
_TOOL_CONFIG_KEYS = frozenset({"functionCallingConfig", "retrievalConfig"})
_FUNCTION_CALLING_KEYS = frozenset({"mode", "allowedFunctionNames"})
_SAFETY_KEYS = frozenset({"category", "threshold", "method"})
_SCHEMA_KEYS = frozenset({
    "type", "format", "title", "description", "nullable", "enum", "maxItems", "minItems",
    "properties", "required", "minProperties", "maxProperties", "minLength", "maxLength",
    "pattern", "example", "anyOf", "propertyOrdering", "default", "items", "minimum",
    "maximum",
})
_SCHEMA_INTEGERS = {"maxItems", "minItems", "minProperties", "maxProperties",
                    "minLength", "maxLength"}


def check_shape(body: Mapping[str, Any]) -> None:
    """Shape checks that apply to every route, including same-protocol passthrough.

    Native requests are otherwise left to the upstream; vendor fields stay open.
    """
    def finite(value: Any) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise _bad("'body' must contain finite JSON numbers")
        if isinstance(value, Mapping):
            for item in value.values():
                finite(item)
        elif isinstance(value, list):
            for item in value:
                finite(item)

    finite(body)
    contents = body.get("contents")
    if not isinstance(contents, list) or not contents:
        raise _bad("'contents' must be a non-empty array")
    for index, content in enumerate(contents):
        if not isinstance(content, Mapping):
            raise _bad(f"'contents[{index}]' must be an object")
        parts = content.get("parts")
        if not isinstance(parts, list) or any(not isinstance(p, Mapping) for p in parts):
            raise _bad(f"'contents[{index}].parts' must be an array of objects")
        role = content.get("role")
        if role is not None and not isinstance(role, str):
            raise _bad(f"'contents[{index}].role' must be a string")
    for name, kind in (("systemInstruction", Mapping), ("generationConfig", Mapping),
                       ("toolConfig", Mapping), ("tools", list), ("safetySettings", list)):
        for key in (name, re.sub(r"([A-Z])", lambda m: "_" + m.group(1).lower(), name)):
            if key in body and not isinstance(body[key], kind):
                raise _bad(f"'{name}' must be " + ("an array" if kind is list else "an object"))


# -- translation: Gemini request -> Chat Completions ---------------------------

@dataclass
class Translation:
    """A Chat Completions request plus what is needed to report and convert back."""

    params: dict[str, Any]
    losses: list[dict[str, str]] = field(default_factory=list)
    # Chat parameter path -> Gemini request path, used to name later-stage losses.
    origins: dict[str, str] = field(default_factory=dict)
    # Gemini path of each entry in params["tools"].
    tool_paths: list[str] = field(default_factory=list)
    include_thoughts: bool = False

    def rename(self, name: str) -> str:
        match = re.fullmatch(r"tools\[(\d+)\](.*)", name)
        if match and int(match[1]) < len(self.tool_paths):
            return self.tool_paths[int(match[1])] + match[2]
        for chat, gemini in self.origins.items():
            if name == chat or name.startswith(chat + "."):
                return gemini + name[len(chat):]
        return name


_MIME = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]*/[A-Za-z0-9][A-Za-z0-9.+-]*")
_IMAGE_MIMES = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})
_AUDIO_FORMATS = {"audio/wav": "wav", "audio/x-wav": "wav", "audio/wave": "wav",
                  "audio/mpeg": "mp3", "audio/mp3": "mp3"}
# Input kinds each destination protocol can carry through the Chat bridge.
_MEDIA = {
    DIALECT_OPENAI_CHAT: frozenset({"image", "pdf", "audio"}),
    DIALECT_OPENAI_RESPONSES: frozenset({"image", "pdf"}),
    DIALECT_ANTHROPIC: frozenset({"image", "pdf"}),
}
_TOOL_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
_FILENAME = re.compile(r"[\w. -]{1,100}")
_MAX_SCHEMA_DEPTH = 32


class _Translator:
    def __init__(self, body: Mapping[str, Any], target: str) -> None:
        self.target = target
        self.request = _object(body, _REQUEST_KEYS, "request")
        self.losses: list[dict[str, str]] = []
        self.origins: dict[str, str] = {}
        self.tool_paths: list[str] = []
        self.calls = 0
        self.ids: dict[str, str] = {}

    def lose(self, path: str, reason: str, *, approximate: bool = False) -> None:
        self.losses.append({"field": path, "reason": reason,
                            "action": "approximated" if approximate else "dropped"})

    # ---- top level ----------------------------------------------------------

    def run(self) -> Translation:
        request = self.request
        for key in ("cachedContent", "continuationToken"):
            if request.get(key) is not None:
                raise RouterError(
                    400, f"'{key}' refers to state held by Gemini and requires a native Gemini "
                         "route. For translation, send the complete conversation in 'contents'.",
                    code="unsupported_gemini_state")
        params: dict[str, Any] = {}
        messages: list[dict[str, Any]] = []
        system = request.get("systemInstruction")
        if system is not None:
            messages.append({"role": "system", "content": self.system_text(system)})
        messages += self.messages(request.get("contents"))
        if not any(m["role"] != "system" for m in messages):
            raise _bad("'contents' has nothing that can be translated to this destination")
        params["messages"] = messages

        thinking = self.generation(params)
        self.tools(params)
        for key, reason in (
            ("labels", "request labels are Gemini billing metadata with no equivalent"),
            ("serviceTier", "service tiers are provider-specific"),
            ("store", "the logging preference is provider-specific"),
        ):
            if request.get(key) is not None:
                self.lose(key, reason)
        safety = request.get("safetySettings")
        if safety:
            for index, item in enumerate(_typed(safety, (list,), "safetySettings")):
                _object(item, _SAFETY_KEYS, f"safetySettings[{index}]")
            self.lose("safetySettings",
                      "safety thresholds are Gemini-specific; the destination applies its own "
                      "moderation")
        return Translation(params=params, losses=self.losses, origins=self.origins,
                           tool_paths=self.tool_paths, include_thoughts=thinking)

    # ---- contents -----------------------------------------------------------

    def system_text(self, value: Any) -> str:
        content = _object(value, _CONTENT_KEYS, "systemInstruction")
        texts = []
        for index, part in enumerate(_typed(content.get("parts"), (list,), "systemInstruction.parts")):
            part = _object(part, _PART_KEYS, f"systemInstruction.parts[{index}]")
            if not isinstance(part.get("text"), str):
                raise _bad("'systemInstruction' accepts text parts only")
            texts.append(part["text"])
        return "\n".join(texts)

    def call_id(self, original: Any) -> str:
        if isinstance(original, str) and original:
            if original not in self.ids:
                self.ids[original] = original if _TOOL_ID.fullmatch(original) else self.fresh_id()
            return self.ids[original]
        return self.fresh_id()

    def fresh_id(self) -> str:
        self.calls += 1
        return f"call_{self.calls}"

    def messages(self, contents: Any) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        pending: list[tuple[str, str]] = []   # (id, name) of the previous model turn's calls
        for ci, raw in enumerate(_typed(contents, (list,), "contents")):
            content = _object(raw, _CONTENT_KEYS, f"contents[{ci}]")
            role = content.get("role") or "user"
            if role not in ("user", "model", "function"):
                raise _bad(f"'contents[{ci}].role' must be 'user' or 'model'")
            parts = _typed(content.get("parts"), (list,), f"contents[{ci}].parts")
            if role == "model":
                message, pending = self.model_turn(ci, parts)
                if message is not None:
                    messages.append(message)
            else:
                messages += self.user_turn(ci, parts, pending)
                pending = []
        return messages

    def part(self, ci: int, pi: int, raw: Any) -> tuple[dict[str, Any], str | None, str]:
        path = f"contents[{ci}].parts[{pi}]"
        part = _object(raw, _PART_KEYS, path)
        kinds = [key for key in _DATA_KEYS if part.get(key) is not None]
        if len(kinds) > 1:
            raise _bad(f"'{path}' sets more than one of {', '.join(kinds)}")
        return part, kinds[0] if kinds else None, path

    def model_turn(self, ci: int, parts: list[Any]) -> tuple[dict[str, Any] | None, list[tuple[str, str]]]:
        texts: list[str] = []
        calls: list[dict[str, Any]] = []
        pending: list[tuple[str, str]] = []
        for pi, raw in enumerate(parts):
            part, kind, path = self.part(ci, pi, raw)
            if part.get("thought") is True:
                # Thought summaries and signatures are provider-private reasoning
                # state a different destination can neither use nor verify.
                continue
            if kind == "text":
                texts.append(_typed(part["text"], (str,), path + ".text"))
            elif kind == "functionCall":
                call = _object(part["functionCall"], _FUNCTION_CALL_KEYS, path + ".functionCall")
                name = _typed(call.get("name"), (str,), path + ".functionCall.name")
                args = call.get("args", {})
                if args is None:
                    args = {}
                _typed(args, (dict,), path + ".functionCall.args")
                identity = self.call_id(call.get("id"))
                calls.append({"id": identity, "type": "function", "function": {
                    "name": name, "arguments": json.dumps(args, ensure_ascii=False,
                                                          separators=(",", ":"))}})
                pending.append((identity, name))
            elif kind is not None:
                self.lose(path, f"a model-produced '{kind}' part has no equivalent in a "
                                "destination conversation")
        if not texts and not calls:
            return None, pending
        message: dict[str, Any] = {"role": "assistant",
                                   "content": "".join(texts) if texts else None}
        if calls:
            message["tool_calls"] = calls
        return message, pending

    def user_turn(self, ci: int, parts: list[Any],
                  pending: list[tuple[str, str]]) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        blocks: list[dict[str, Any]] = []
        for pi, raw in enumerate(parts):
            part, kind, path = self.part(ci, pi, raw)
            if part.get("thought") is True:
                continue
            if kind == "text":
                # An empty text part carries nothing; several destinations reject it.
                if _typed(part["text"], (str,), path + ".text"):
                    blocks.append({"type": "text", "text": part["text"]})
            elif kind == "inlineData":
                block = self.media(path + ".inlineData", part["inlineData"])
                if block is not None:
                    blocks.append(block)
            elif kind == "fileData":
                self.lose(path, "file references resolve only on Gemini; send the data inline")
            elif kind == "functionResponse":
                tools.append(self.function_response(path + ".functionResponse",
                                                    part["functionResponse"], pending))
            elif kind is not None:
                self.lose(path, f"'{kind}' input has no equivalent on this destination")
            elif part.get("videoMetadata") is not None:
                self.lose(path, "video input has no equivalent on this destination")
        messages = list(tools)
        if blocks:
            only_text = len(blocks) == 1 and blocks[0]["type"] == "text"
            messages.append({"role": "user",
                             "content": blocks[0]["text"] if only_text else blocks})
        return messages

    def function_response(self, path: str, raw: Any,
                          pending: list[tuple[str, str]]) -> dict[str, Any]:
        response = _object(raw, _FUNCTION_RESPONSE_KEYS, path)
        name = _typed(response.get("name"), (str,), path + ".name")
        if response.get("parts"):
            self.lose(path + ".parts", "multimodal function results have no equivalent")
        value = response.get("response", {})
        original = response.get("id")
        if isinstance(original, str) and original:
            identity = self.call_id(original)
            pending[:] = [item for item in pending if item[0] != identity]
        else:
            match = next((item for item in pending if item[1] == name), None)
            if match is None:
                raise _bad(f"'{path}' has no matching functionCall named '{_label(name)}' "
                           "in the preceding model turn")
            pending.remove(match)
            identity = match[0]
        return {"role": "tool", "tool_call_id": identity,
                "content": json.dumps(value, ensure_ascii=False, separators=(",", ":"))}

    def media(self, path: str, raw: Any) -> dict[str, Any] | None:
        inline = _object(raw, _INLINE_KEYS, path)
        mime, data = inline.get("mimeType"), inline.get("data")
        if not isinstance(mime, str) or _MIME.fullmatch(mime) is None:
            raise _bad(f"'{path}.mimeType' must be a media type")
        if not isinstance(data, str) or not data:
            raise _bad(f"'{path}.data' must be non-empty base64")
        try:
            base64.b64decode(data, validate=False)
            if re.search(r"[^A-Za-z0-9+/=_\s-]", data):
                raise binascii.Error
        except (binascii.Error, ValueError):
            raise _bad(f"'{path}.data' must be base64") from None
        lower = mime.lower()
        if lower in _IMAGE_MIMES:
            kind = "image"
        elif lower == "application/pdf":
            kind = "pdf"
        elif lower in _AUDIO_FORMATS:
            kind = "audio"
        else:
            self.lose(path, f"media type '{lower}' has no established equivalent on this "
                            "destination")
            return None
        if kind not in _MEDIA[self.target]:
            self.lose(path, f"{kind} input is not carried to this destination protocol")
            return None
        if kind == "image":
            return {"type": "image_url", "image_url": {"url": f"data:{lower};base64,{data}"}}
        if kind == "pdf":
            name = inline.get("displayName")
            name = name if isinstance(name, str) and _FILENAME.fullmatch(name) else "document.pdf"
            return {"type": "file", "file": {
                "file_data": f"data:application/pdf;base64,{data}", "filename": name}}
        return {"type": "input_audio", "input_audio": {"data": data,
                                                      "format": _AUDIO_FORMATS[lower]}}

    # ---- tools ----------------------------------------------------------------

    def schema(self, node: Any, where: str, depth: int = 0) -> Any:
        """OpenAPI-subset Schema -> JSON Schema (lowercase types, nullable, integers)."""
        if depth > _MAX_SCHEMA_DEPTH:
            raise _bad(f"'{where}' is nested too deeply")
        schema = _object(node, _SCHEMA_KEYS, where)
        result: dict[str, Any] = {}
        for key, value in schema.items():
            if key == "propertyOrdering" or value is None:
                continue
            if key == "type":
                kind = _typed(value, (str,), where + ".type").lower()
                if kind != "type_unspecified":
                    result["type"] = kind
            elif key == "properties":
                _typed(value, (dict,), where + ".properties")
                result["properties"] = {name: self.schema(item, f"{where}.properties.{_label(name)}",
                                                          depth + 1)
                                        for name, item in value.items()}
            elif key == "items":
                result["items"] = self.schema(value, where + ".items", depth + 1)
            elif key == "anyOf":
                result["anyOf"] = [self.schema(item, f"{where}.anyOf[{i}]", depth + 1)
                                   for i, item in enumerate(_typed(value, (list,), where + ".anyOf"))]
            elif key in _SCHEMA_INTEGERS:
                result[key] = int(value) if isinstance(value, str) and value.isdigit() else value
            elif key == "nullable":
                continue
            else:
                result[key] = value
        if schema.get("nullable") is True:
            kind = result.get("type")
            if isinstance(kind, str):
                result["type"] = [kind, "null"]
            elif "anyOf" in result:
                result["anyOf"] = [*result["anyOf"], {"type": "null"}]
        return result

    def tools(self, params: dict[str, Any]) -> None:
        raw_tools = self.request.get("tools")
        declared: list[tuple[str, dict[str, Any]]] = []
        names: set[str] = set()
        for ti, raw in enumerate(raw_tools or []):
            tool = _object(raw, _TOOL_KEYS, f"tools[{ti}]")
            for key, value in tool.items():
                if value is None:
                    continue
                if key != "functionDeclarations":
                    decision = decide_tool("gemini", self.target, key, f"tools[{ti}].{key}")
                    loss = decision.loss()
                    if loss is not None:
                        self.losses.append(loss)
                    continue
                for fi, raw_decl in enumerate(_typed(value, (list,), f"tools[{ti}].{key}")):
                    path = f"tools[{ti}].functionDeclarations[{fi}]"
                    decl = _object(raw_decl, _DECLARATION_KEYS, path)
                    name = _typed(decl.get("name"), (str,), path + ".name")
                    if name in names:
                        raise _bad(f"'{path}' repeats function name '{_label(name)}'")
                    names.add(name)
                    if decl.get("parameters") is not None and decl.get("parametersJsonSchema") is not None:
                        raise _bad(f"'{path}' sets both parameters and parametersJsonSchema")
                    if decl.get("behavior") is not None:
                        self.lose(path + ".behavior", "non-blocking function behavior has no equivalent")
                    for key2 in ("response", "responseJsonSchema"):
                        if decl.get(key2) is not None:
                            self.lose(f"{path}.{key2}", "function response schemas have no equivalent")
                    if decl.get("parametersJsonSchema") is not None:
                        schema = _typed(decl["parametersJsonSchema"], (dict,), path + ".parametersJsonSchema")
                    elif decl.get("parameters") is not None:
                        schema = self.schema(decl["parameters"], path + ".parameters")
                    else:
                        schema = {"type": "object", "properties": {}}
                    function: dict[str, Any] = {"name": name, "parameters": schema}
                    if decl.get("description") is not None:
                        function["description"] = _typed(decl["description"], (str,), path + ".description")
                    declared.append((path, {"type": "function", "function": function}))

        choice: Any = None
        config = self.request.get("toolConfig")
        if config is not None:
            config = _object(config, _TOOL_CONFIG_KEYS, "toolConfig")
            if config.get("retrievalConfig") is not None:
                self.lose("toolConfig.retrievalConfig", "retrieval settings apply to Gemini grounding only")
            calling = config.get("functionCallingConfig")
            if calling is not None:
                calling = _object(calling, _FUNCTION_CALLING_KEYS, "toolConfig.functionCallingConfig")
                where = "toolConfig.functionCallingConfig"
                mode = calling.get("mode")
                if mode not in (None, "MODE_UNSPECIFIED", "AUTO", "ANY", "NONE", "VALIDATED"):
                    raise _bad(f"'{where}.mode' is not a function calling mode")
                allowed = calling.get("allowedFunctionNames")
                if allowed is not None:
                    _typed(allowed, (list,), where + ".allowedFunctionNames")
                    if any(not isinstance(name, str) for name in allowed):
                        raise _bad(f"'{where}.allowedFunctionNames' must contain strings")
                    unknown = [name for name in allowed if name not in names]
                    if unknown and declared:
                        raise _bad(f"'{where}.allowedFunctionNames' names a function that is not declared")
                choice = {"AUTO": "auto", "ANY": "required", "NONE": "none"}.get(mode or "")
                if mode == "VALIDATED":
                    choice = "auto"
                    self.lose(where + ".mode", "schema-validated calling has no equivalent; "
                              "automatic calling is used", approximate=True)
                if allowed:
                    if len(allowed) == 1 and mode == "ANY":
                        choice = {"type": "function", "function": {"name": allowed[0]}}
                    elif mode != "NONE":
                        keep = set(allowed)
                        declared = [item for item in declared if item[1]["function"]["name"] in keep]
                        self.lose(where + ".allowedFunctionNames",
                                  "the destination has no allowed-tools control; the tool list "
                                  "was narrowed instead", approximate=True)
        if declared:
            params["tools"] = [item[1] for item in declared]
            self.tool_paths = [item[0] for item in declared]
        if choice is not None:
            params["tool_choice"] = choice
            self.origins["tool_choice"] = "toolConfig.functionCallingConfig"

    # ---- generationConfig -------------------------------------------------------

    def generation(self, params: dict[str, Any]) -> bool:
        raw = self.request.get("generationConfig")
        if raw is None:
            return False
        gc = _object(raw, _GENERATION_KEYS, "generationConfig")
        extra: dict[str, Any] = {}

        def put(key: str, chat: str, value: Any) -> None:
            params[chat] = value
            self.origins[chat] = f"generationConfig.{key}"

        if gc.get("maxOutputTokens") is not None:
            put("maxOutputTokens", "max_tokens", _int(gc["maxOutputTokens"], "generationConfig.maxOutputTokens", low=1))
        if gc.get("temperature") is not None:
            put("temperature", "temperature", _number(gc["temperature"], "generationConfig.temperature", low=0, high=2))
        if gc.get("topP") is not None:
            put("topP", "top_p", _number(gc["topP"], "generationConfig.topP", low=0, high=1))
        if gc.get("seed") is not None:
            put("seed", "seed", _int(gc["seed"], "generationConfig.seed"))
        for key, chat in (("presencePenalty", "presence_penalty"), ("frequencyPenalty", "frequency_penalty")):
            if gc.get(key) is not None:
                put(key, chat, _number(gc[key], f"generationConfig.{key}", low=-2, high=2))
        stops = gc.get("stopSequences")
        if stops is not None:
            _typed(stops, (list,), "generationConfig.stopSequences")
            if any(not isinstance(item, str) or not item for item in stops):
                raise _bad("'generationConfig.stopSequences' must contain non-empty strings")
            if stops:
                put("stopSequences", "stop", list(stops))
        if gc.get("topK") is not None:
            top_k = _int(gc["topK"], "generationConfig.topK", low=0)
            if self.target == DIALECT_OPENAI_RESPONSES:
                self.lose("generationConfig.topK", "no reviewed bridge mapping")
            else:
                extra["top_k"] = top_k
                self.origins["extra_body.top_k"] = "generationConfig.topK"
        if gc.get("candidateCount") not in (None, 1):
            _int(gc["candidateCount"], "generationConfig.candidateCount", low=1)
            self.lose("generationConfig.candidateCount",
                      "multiple candidates have no equivalent; one is generated")
        if gc.get("responseLogprobs") is True or gc.get("logprobs") is not None:
            self.lose("generationConfig.logprobs", "log probabilities are not translated")
        modalities = gc.get("responseModalities")
        if modalities is not None:
            _typed(modalities, (list,), "generationConfig.responseModalities")
            if any(item not in ("TEXT", "MODALITY_UNSPECIFIED") for item in modalities):
                self.lose("generationConfig.responseModalities",
                          "non-text output modalities are not produced by this destination")
        for key in ("speechConfig", "imageConfig", "mediaResolution", "enableEnhancedCivicAnswers",
                    "audioTimestamp"):
            if gc.get(key) is not None:
                self.lose(f"generationConfig.{key}", "this control has no equivalent on the destination")
        self.response_format(gc, params)
        include = self.thinking(gc.get("thinkingConfig"), params, extra)
        if extra:
            params["extra_body"] = extra
        return include

    def response_format(self, gc: Mapping[str, Any], params: dict[str, Any]) -> None:
        mime = gc.get("responseMimeType")
        schema, json_schema = gc.get("responseSchema"), gc.get("responseJsonSchema")
        if schema is not None and json_schema is not None:
            raise _bad("'generationConfig' sets both responseSchema and responseJsonSchema")
        if mime is not None:
            _typed(mime, (str,), "generationConfig.responseMimeType")
        if (schema is not None or json_schema is not None) and mime != "application/json":
            if mime == "text/x.enum":
                self.lose("generationConfig.responseMimeType", "enum output has no equivalent")
                return
            raise _bad("'generationConfig.responseSchema' requires responseMimeType 'application/json'")
        if mime in (None, "text/plain"):
            return
        if mime != "application/json":
            self.lose("generationConfig.responseMimeType", "this output type has no equivalent")
            return
        if schema is None and json_schema is None:
            params["response_format"] = {"type": "json_object"}
        else:
            body = (self.schema(schema, "generationConfig.responseSchema") if schema is not None
                    else _typed(json_schema, (dict,), "generationConfig.responseJsonSchema"))
            params["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "response", "schema": body}}
        self.origins["response_format"] = "generationConfig.responseMimeType"

    def thinking(self, raw: Any, params: dict[str, Any], extra: dict[str, Any]) -> bool:
        if raw is None:
            return False
        config = _object(raw, _THINKING_KEYS, "generationConfig.thinkingConfig")
        where = "generationConfig.thinkingConfig"
        include = config.get("includeThoughts")
        if include is not None:
            _typed(include, (bool,), where + ".includeThoughts")
        budget, level = config.get("thinkingBudget"), config.get("thinkingLevel")
        if budget is not None and level is not None:
            raise _bad(f"'{where}' sets both thinkingBudget and thinkingLevel")
        if budget is not None:
            _int(budget, where + ".thinkingBudget", low=-1)
            if budget == 0:
                params["reasoning_effort"] = "none"
                self.origins["reasoning_effort"] = where + ".thinkingBudget"
            elif budget > 0:
                extra["thinking"] = {"type": "enabled", "budget_tokens": budget}
                self.origins["extra_body.thinking"] = where + ".thinkingBudget"
            # -1 asks for the model's own dynamic budget: the destination default.
        if level is not None:
            _typed(level, (str,), where + ".thinkingLevel")
            if level not in ("THINKING_LEVEL_UNSPECIFIED", "MINIMAL", "LOW", "MEDIUM", "HIGH"):
                raise _bad(f"'{where}.thinkingLevel' is not a thinking level")
            if level != "THINKING_LEVEL_UNSPECIFIED":
                params["reasoning_effort"] = level.lower()
                self.origins["reasoning_effort"] = where + ".thinkingLevel"
        return include is True


def to_chat(body: Mapping[str, Any], target: str, *, stream: bool) -> Translation:
    """Translate a Gemini request body into Chat Completions parameters.

    Raises RouterError(400) for malformed requests and for requests that cannot be
    translated without changing their meaning; recoverable differences are returned
    as loss records.
    """
    translation = _Translator(body, target).run()
    if stream:
        translation.params["stream"] = True
        translation.params["stream_options"] = {"include_usage": True}
    return translation


# -- translation: Chat Completions result -> Gemini ------------------------------

_FINISH = {"stop": "STOP", "length": "MAX_TOKENS", "tool_calls": "STOP",
           "function_call": "STOP", "content_filter": "SAFETY"}


def _map(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _usage(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    prompt = raw.get("prompt_tokens") or 0
    completion = raw.get("completion_tokens") or 0
    thoughts = _map(raw.get("completion_tokens_details")).get("reasoning_tokens") or 0
    cached = _map(raw.get("prompt_tokens_details")).get("cached_tokens")
    usage: dict[str, Any] = {
        "promptTokenCount": prompt,
        # Chat completion tokens include reasoning; Gemini reports them separately.
        "candidatesTokenCount": max(completion - thoughts, 0),
        "totalTokenCount": prompt + completion,
    }
    if thoughts:
        usage["thoughtsTokenCount"] = thoughts
    if cached:
        usage["cachedContentTokenCount"] = cached
    return usage


def _function_call(call: Mapping[str, Any]) -> dict[str, Any] | None:
    """A Gemini functionCall part, or None when the call is not well-formed."""
    function = _map(call.get("function"))
    name = function.get("name")
    text = function.get("arguments") or "{}"
    try:
        args = json.loads(text) if isinstance(text, str) else text
    except ValueError:
        return None
    if not isinstance(name, str) or not name or not isinstance(args, dict):
        return None
    part: dict[str, Any] = {"name": name, "args": args}
    if isinstance(call.get("id"), str) and call["id"]:
        part["id"] = call["id"]
    return {"functionCall": part}


def _candidate(parts: list[dict[str, Any]], finish: str | None, *, malformed: bool = False) -> dict[str, Any]:
    candidate: dict[str, Any] = {"content": {"role": "model", "parts": parts}, "index": 0}
    if malformed:
        candidate["finishReason"] = "MALFORMED_FUNCTION_CALL"
    elif finish is not None:
        candidate["finishReason"] = _FINISH.get(finish, "OTHER")
    return candidate


def chat_to_gemini(data: Mapping[str, Any], translation: Translation, upstream_model: str) -> dict[str, Any]:
    """A Chat Completions response as a GenerateContentResponse."""
    choices = data.get("choices")
    choice = _map(choices[0] if isinstance(choices, list) and choices else None)
    message = _map(choice.get("message"))
    parts: list[dict[str, Any]] = []
    reasoning = message.get("reasoning_content")
    if translation.include_thoughts and isinstance(reasoning, str) and reasoning:
        parts.append({"text": reasoning, "thought": True})
    if isinstance(message.get("content"), str) and message["content"]:
        parts.append({"text": message["content"]})
    malformed = False
    for call in message.get("tool_calls") or []:
        part = _function_call(call if isinstance(call, Mapping) else {})
        if part is None:
            malformed = True
        else:
            parts.append(part)
    response: dict[str, Any] = {
        "candidates": [_candidate(parts, choice.get("finish_reason"), malformed=malformed)],
        "modelVersion": upstream_model,
    }
    if isinstance(data.get("id"), str):
        response["responseId"] = data["id"]
    usage = _usage(data.get("usage"))
    if usage is not None:
        response["usageMetadata"] = usage
    return response


def gemini_frames(chunks: Iterable[Mapping[str, Any]], translation: Translation,
                  upstream_model: str) -> Iterator[dict[str, Any]]:
    """Chat stream chunks -> GenerateContentResponse frames.

    Text and thought deltas are forwarded as they arrive. Function calls are
    buffered, because Gemini delivers each call whole, and are emitted with the
    final frame together with the finish reason and usage.
    """
    calls: dict[int, dict[str, Any]] = {}
    finish: str | None = None
    usage: dict[str, Any] | None = None
    response_id: str | None = None

    def frame(parts: list[dict[str, Any]], *, final: bool = False, malformed: bool = False) -> dict[str, Any]:
        candidate = _candidate(parts, finish if final else None, malformed=malformed and final)
        if final and finish is None and not malformed:
            candidate["finishReason"] = "OTHER"
        result: dict[str, Any] = {"candidates": [candidate], "modelVersion": upstream_model}
        if response_id:
            result["responseId"] = response_id
        if final and usage is not None:
            result["usageMetadata"] = usage
        return result

    for chunk in chunks:
        if not isinstance(chunk, Mapping):
            continue
        if chunk.get("error") is not None:
            error = chunk["error"]
            raise RouterError(502, error.get("message", "Upstream stream failed")
                              if isinstance(error, Mapping) else str(error))
        if isinstance(chunk.get("id"), str):
            response_id = chunk["id"]
        found = _usage(chunk.get("usage"))
        if found is not None:
            usage = found
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, Mapping) or choice.get("index", 0) != 0:
                continue
            delta = _map(choice.get("delta"))
            parts: list[dict[str, Any]] = []
            reasoning = delta.get("reasoning_content")
            if translation.include_thoughts and isinstance(reasoning, str) and reasoning:
                parts.append({"text": reasoning, "thought": True})
            if isinstance(delta.get("content"), str) and delta["content"]:
                parts.append({"text": delta["content"]})
            for fragment in delta.get("tool_calls") or []:
                if not isinstance(fragment, Mapping):
                    continue
                index = fragment.get("index")
                if not isinstance(index, int):
                    index = max(calls) if calls and not fragment.get("id") else len(calls)
                entry = calls.setdefault(index, {"id": None, "function": {"name": "", "arguments": ""}})
                if fragment.get("id"):
                    entry["id"] = fragment["id"]
                function = _map(fragment.get("function"))
                if function.get("name"):
                    entry["function"]["name"] = function["name"]
                if isinstance(function.get("arguments"), str):
                    entry["function"]["arguments"] += function["arguments"]
            if parts:
                yield frame(parts)
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]

    parts, malformed = [], False
    for index in sorted(calls):
        part = _function_call(calls[index])
        if part is None:
            malformed = True
        else:
            parts.append(part)
    yield frame(parts, final=True, malformed=malformed)


def sse_frame(payload: bytes) -> bytes:
    return b"data: " + payload + b"\n\n"


def json_array_frames(frames: Iterator[bytes]) -> Iterator[bytes]:
    """The stream without ``alt=sse``: one JSON array, written incrementally."""
    first = True
    for frame in frames:
        yield (b"[" if first else b",\r\n") + frame
        first = False
    yield b"]" if not first else b"[]"


# -- models.list / models.get --------------------------------------------------

def _fact(entry: Mapping[str, Any], *path: str) -> Any:
    node: Any = entry.get("metadata", {}).get("router", {})
    for key in path:
        node = node.get(key, {}) if isinstance(node, Mapping) else {}
    return node.get("value") if isinstance(node, Mapping) else None


def model_resource(entry: Mapping[str, Any]) -> dict[str, Any]:
    """One route as a Gemini Model resource; unknown limits are omitted, never invented."""
    metadata = entry.get("metadata", {})
    model: dict[str, Any] = {
        "name": f"models/{entry['id']}",
        "baseModelId": entry["id"],
        "displayName": entry.get("display_name") or entry["id"],
        "description": f"Routed to provider '{metadata.get('provider')}' "
                       f"({metadata.get('dialect')}) by model-router.",
        "supportedGenerationMethods": list(GENERATION_METHODS_LISTED),
    }
    input_limit = _fact(entry, "limits", "max_input_tokens") or _fact(entry, "limits", "context_window")
    output_limit = _fact(entry, "limits", "max_output_tokens")
    if isinstance(input_limit, int):
        model["inputTokenLimit"] = input_limit
    if isinstance(output_limit, int):
        model["outputTokenLimit"] = output_limit
    thinking = _fact(entry, "reasoning", "supported")
    if isinstance(thinking, bool):
        model["thinking"] = thinking
    return model


def models_page(entries: list[Mapping[str, Any]], page_size: str | None,
                page_token: str | None) -> dict[str, Any]:
    size = 50
    if page_size not in (None, ""):
        if not re.fullmatch(r"[0-9]{1,6}", page_size or "") or not 0 <= int(page_size) <= 1000:
            raise _bad("'pageSize' must be an integer between 0 and 1000")
        size = int(page_size) or 50
    offset = 0
    if page_token:
        try:
            decoded = base64.urlsafe_b64decode(page_token.encode("ascii") + b"=" * (-len(page_token) % 4))
            match = re.fullmatch(rb"offset:([0-9]{1,9})", decoded)
        except (binascii.Error, UnicodeError, ValueError):
            match = None
        if match is None:
            raise _bad("'pageToken' is invalid")
        offset = int(match[1])
    page = entries[offset:offset + size]
    result: dict[str, Any] = {"models": [model_resource(entry) for entry in page]}
    if offset + size < len(entries):
        token = base64.urlsafe_b64encode(b"offset:%d" % (offset + size)).decode("ascii").rstrip("=")
        result["nextPageToken"] = token
    return result
