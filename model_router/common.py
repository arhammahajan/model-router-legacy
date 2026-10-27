"""Shared JSON, immutable data and sanitized error/diagnostic helpers."""
from __future__ import annotations

import json
import logging
import re
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping
from urllib.parse import quote, quote_plus

import httpx

from .constants import (
    _ANTHROPIC_ERROR_TYPES,
    DIALECT_ANTHROPIC,
    DIALECT_OPENAI_CHAT,
    DIALECT_OPENAI_RESPONSES,
    MAX_ERROR_MESSAGE,
)

if TYPE_CHECKING:
    from .streams import StreamState


class RouterError(Exception):
    """An error that should be returned to the caller with a specific status."""

    def __init__(
        self,
        status: int,
        message: str,
        *,
        code: str | None = None,
        error_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code
        self.error_type = error_type


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if hasattr(value, "value"):
        return value.value
    return str(value)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), default=_json_default
    ).encode("utf-8")


class ErrorSanitizer:
    """Application-owned secrets, with no process-wide credential registry."""

    def __init__(self, secrets: Any = ()) -> None:
        values: set[str] = set()
        for secret in secrets:
            if not isinstance(secret, str) or not secret:
                continue
            values.update((secret, quote(secret, safe=""), quote_plus(secret),
                           json.dumps(secret, ensure_ascii=False)[1:-1],
                           json.dumps(secret, ensure_ascii=True)[1:-1]))
        self.secrets = tuple(sorted(values, key=len, reverse=True))

    def __call__(self, message: str) -> str:
        # Replace whole known values before limiting, including provider keys
        # with arbitrary formats and secrets that str(exc) has JSON/URL encoded.
        for secret in self.secrets:
            message = message.replace(secret, "<redacted>")
        message = re.sub(r"(?i)(bearer\s+)[^\s\"',;<>\[\]{}]+",
                         r"\1<redacted>", message)
        # Do not assume a provider's keys start with sk-. Handle credentials in
        # header/dict/query diagnostics even when their values are unknown.
        message = re.sub(
            r"(?i)((?:[\w-]*(?:key|token|secret|password|credential)|"
            r"authorization|cookie)[\"']?\s*[:=]\s*[\"']?)"
            r"[^\s\"',;&<>\[\]{}]+", r"\1<redacted>", message,
        )
        message = re.sub(r"(?i)(https?://)[^/\s@]+@", r"\1<redacted>@", message)
        message = re.sub(r"\b(?:sk-[\w.\-]+|AIza[\w\-]{20,}|gsk_[\w]{20,})",
                         "<redacted>", message)
        # Avoid log forging/terminal controls as well as oversized diagnostics.
        message = "".join(c if c.isprintable() else " " for c in message)
        return message[:MAX_ERROR_MESSAGE]


_DEFAULT_SANITIZER = ErrorSanitizer()


def _error_message(
    exc: BaseException, sanitize: ErrorSanitizer = _DEFAULT_SANITIZER,
) -> str:
    message = getattr(exc, "message", None)
    if not isinstance(message, str) or not message:
        message = str(exc) or exc.__class__.__name__
    return sanitize(message)


def _status_for(exc: BaseException) -> int:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and 400 <= status <= 599:
        return 504 if status == 408 else status
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        return 504
    if isinstance(exc, RouterError):
        return exc.status
    return 502


def _error_payload(
    ingress: str, exc: BaseException,
    sanitize: ErrorSanitizer = _DEFAULT_SANITIZER,
) -> dict[str, Any]:
    status = _status_for(exc)
    message = _error_message(exc, sanitize)
    code = getattr(exc, "code", None)
    error_type = getattr(exc, "error_type", None)
    code = sanitize(str(code)) if code is not None else None
    error_type = sanitize(str(error_type)) if error_type is not None else None

    if ingress == DIALECT_ANTHROPIC:
        if error_type is None:
            error_type = _ANTHROPIC_ERROR_TYPES.get(
                status, "api_error" if status >= 500 else "invalid_request_error"
            )
        return {"type": "error", "error": {"type": error_type, "message": message}}

    if error_type is None:
        error_type = "invalid_request_error" if status < 500 else "api_error"
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": code,
        }
    }


def _upstream_error_headers(exc: BaseException, sanitize: ErrorSanitizer) -> dict[str, str]:
    """Recover only retry/rate-limit hints from adapter exception wrappers.

    LiteLLM may replace the outer response with an empty synthetic response;
    the real HTTP headers remain on the chained provider/SDK exception.
    """
    headers: dict[str, str] = {}
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(seen) < 16:
        seen.add(id(current))
        for source in (getattr(current, "headers", None),
                       getattr(getattr(current, "response", None), "headers", None)):
            if isinstance(source, Mapping):
                for name, value in source.items():
                    key = str(name).lower()
                    if (key == "retry-after" or "ratelimit" in key) and re.fullmatch(
                            r"[a-z0-9-]+", key):
                        headers.setdefault(key, sanitize(str(value)).replace("\r", " ").replace("\n", " "))
        current = current.__cause__ or current.__context__
    return headers


def _stream_error_payload(
    ingress: str, exc: BaseException,
    sanitize: ErrorSanitizer = _DEFAULT_SANITIZER,
) -> dict[str, Any]:
    if ingress == DIALECT_ANTHROPIC:
        return _error_payload(ingress, exc, sanitize)
    if ingress == DIALECT_OPENAI_RESPONSES:
        payload = _error_payload(ingress, exc, sanitize)["error"]
        return {
            "type": "error",
            "code": payload.get("code") or "upstream_error",
            "message": payload.get("message"),
            "param": payload.get("param"),
        }
    return _error_payload(ingress, exc, sanitize)


def _stream_error_bytes(
    ingress: str, exc: BaseException,
    sanitize: ErrorSanitizer = _DEFAULT_SANITIZER,
    state: StreamState | None = None,
) -> bytes:
    payload = _stream_error_payload(ingress, exc, sanitize)
    if ingress == DIALECT_OPENAI_RESPONSES and state is not None:
        payload = state.number(payload)
    if ingress == DIALECT_OPENAI_CHAT:
        return b"data: " + _json_bytes(payload) + b"\n\n"
    if ingress == DIALECT_OPENAI_RESPONSES:
        return b"event: error\ndata: " + _json_bytes(payload) + b"\n\n"
    return b"event: error\ndata: " + _json_bytes(payload) + b"\n\n"


def _sanitize_generated_errors(value: Any, sanitize: ErrorSanitizer) -> Any:
    """Adapters can emit error objects instead of raising an exception."""
    def diagnostic(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {sanitize(str(key)): diagnostic(part) for key, part in item.items()}
        if isinstance(item, list):
            return [diagnostic(part) for part in item]
        if item is None or isinstance(item, (bool, int, float)):
            return item
        return sanitize(str(item))

    if not isinstance(value, dict):
        return value
    if value.get("type") == "error":
        return diagnostic(value)
    result = value
    if value.get("error") is not None:
        result = {**value, "error": diagnostic(value["error"])}
    response = result.get("response")
    if isinstance(response, dict) and response.get("error") is not None:
        result = {**result, "response": {**response, "error": diagnostic(response["error"])}}
    return result


def _sanitize_adapter_sse(chunk: bytes, sanitize: ErrorSanitizer) -> bytes:
    # The Anthropic adapters emit complete SSE frames. Native/pass-through
    # streams never use this path; their successful bytes remain opaque.
    frames: list[bytes] = []
    for frame in chunk.split(b"\n\n"):
        lines = frame.split(b"\n")
        for index, line in enumerate(lines):
            if not line.startswith(b"data: "):
                continue
            try:
                payload = json.loads(line[6:])
            except (ValueError, UnicodeError):
                if b"event: error" in frame:
                    raise RouterError(502, "Malformed translated error event") from None
                continue
            cleaned = _sanitize_generated_errors(payload, sanitize)
            if cleaned is not payload:
                lines[index] = b"data: " + _json_bytes(cleaned)
        frames.append(b"\n".join(lines))
    return b"\n\n".join(frames)


def _freeze(value: Any) -> Any:
    """Copy JSON-like registry data into immutable containers."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    """Return independently mutable JSON data for adapters and public payloads."""
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


class _DiagnosticLogFilter(logging.Filter):
    """Sanitize diagnostics before handlers (including dependency handlers)."""

    def __init__(self, sanitize: ErrorSanitizer) -> None:
        super().__init__()
        self.sanitize = sanitize

    def filter(self, record: logging.LogRecord) -> bool:
        if hasattr(record, "router_event"):
            # Sanitize values before serialization, retaining valid JSON even
            # when an individual diagnostic reaches the string length limit.
            record.router_event = _sanitize_details(record.router_event, self.sanitize)
            record.msg = json.dumps(record.router_event,
                                    separators=(",", ":"), ensure_ascii=True)
            record.args = ()
            return True
        message = record.getMessage()
        if record.exc_info and record.exc_info[1] is not None:
            message += ": " + _error_message(record.exc_info[1], self.sanitize)
        record.msg = self.sanitize(message)
        record.args = ()
        # Formatters must not append an unredacted traceback/exception again.
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        return True


def _sanitize_details(value: Any, sanitize: ErrorSanitizer) -> Any:
    if isinstance(value, Mapping):
        return {sanitize(str(key)): _sanitize_details(part, sanitize)
                for key, part in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_details(part, sanitize) for part in value]
    return sanitize(value) if isinstance(value, str) else value


def _close_quietly(closable: Any) -> None:
    if closable is None:
        return
    try:
        closable()
    except Exception:  # noqa: BLE001
        pass
