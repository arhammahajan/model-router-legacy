#!/usr/bin/env python3
"""A small LiteLLM-backed model router with bidirectional API translation.

Ingress dialects (what clients speak):

    POST /v1/chat/completions   OpenAI Chat Completions
    POST /v1/responses          OpenAI Responses
    POST /v1/messages           Anthropic Messages

Egress dialects (what each configured provider speaks):

    openai-chat, openai-responses, anthropic, gemini

Every ingress can target every egress. Same-dialect requests preserve vendor
JSON fields and protocol headers. Request bodies are decoded and reserialized
with the upstream model name and fresh representation headers; credentials
are server-owned. Successful response bodies pass through without protocol
translation. HTTP error bodies are replaced with sanitized ingress errors.
Cross-dialect requests use LiteLLM's maintained adapters in both directions,
including streaming.

Running llama.cpp servers (single, router, legacy and llama-swap modes) can be
discovered automatically with a [llama_cpp] config section: their models,
aliases, context sizes and capabilities are probed from /health, /v1/models and
/props and exposed as ordinary router routes.

Configuration is TOML or JSON; see config.example.toml and README.md.
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import io
import json
import logging
import math
import os
import random
import re
import socket
import sys
import threading
import time
import uuid
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Any, AsyncIterator, Callable, Iterator, Mapping
from urllib.parse import parse_qsl, quote, quote_plus, unquote, urlsplit

# Keep LiteLLM self-contained and quiet.  Credentials always come from the
# router config; ambient provider keys are never used.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import httpx
import litellm
import router_metadata

litellm.telemetry = False
litellm.suppress_debug_info = True
litellm.drop_params = False

try:
    # LiteLLM's Anthropic Messages adapters.  These are the same code paths the
    # LiteLLM proxy uses for /v1/messages <-> chat/responses translation.
    from litellm.llms.anthropic.pass_through.adapters.transformation import (
        AnthropicAdapter,
    )
    from litellm.llms.anthropic.pass_through.responses_adapters.handler import (
        LiteLLMMessagesToResponsesAPIHandler,
    )
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
except ImportError as exc:  # pragma: no cover - startup guard
    raise SystemExit(
        "This router needs litellm>=1.106.4,<2. The installed version is missing "
        "LiteLLM's Anthropic translation adapters. Run: pip install -U "
        "'litellm>=1.106.4,<2'"
    ) from exc

__version__ = "0.3.0"

LOG = logging.getLogger("model-router")

DIALECT_ANTHROPIC = "anthropic"
DIALECT_OPENAI_CHAT = "openai-chat"
DIALECT_OPENAI_RESPONSES = "openai-responses"
DIALECT_GEMINI = "gemini"
DIALECTS = {
    DIALECT_ANTHROPIC,
    DIALECT_OPENAI_CHAT,
    DIALECT_OPENAI_RESPONSES,
    DIALECT_GEMINI,
}

ENDPOINTS = {
    "/v1/chat/completions": DIALECT_OPENAI_CHAT,
    "/chat/completions": DIALECT_OPENAI_CHAT,
    "/v1/responses": DIALECT_OPENAI_RESPONSES,
    "/responses": DIALECT_OPENAI_RESPONSES,
    "/v1/messages": DIALECT_ANTHROPIC,
    "/messages": DIALECT_ANTHROPIC,
}

DEFAULT_TIMEOUT = 600.0
DEFAULT_MAX_BODY = 64 * 1024 * 1024
DEFAULT_MAX_CONNECTIONS = 32
DEFAULT_MAX_GENERATIONS = 8
DEFAULT_HEADER_READ_TIMEOUT = 10.0
DEFAULT_BODY_READ_TIMEOUT = 30.0
DEFAULT_WRITE_TIMEOUT = 30.0
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
# Empty keys are not safe: several adapters interpret them as permission to use
# environment/global credentials. Local unauthenticated servers must accept this
# public placeholder, sent by both the native and translated transports.
UNAUTHENTICATED_API_KEY = "model-router-no-auth"
MAX_ERROR_MESSAGE = 4000
# Bound framing overhead independently of the payload limit.
MAX_CHUNK_LINE = 1024
MAX_CHUNK_METADATA = 64 * 1024
MAX_CHUNK_TRAILERS = 100
_HTTP_TOKEN = rb"[!#$%&'*+.^_`|~0-9A-Za-z-]+"
_HTTP_QUOTED = rb'"(?:[\t\x20\x21\x23-\x5b\x5d-\x7e\x80-\xff]|\\[\t\x20-\x7e\x80-\xff])*"'
_CHUNK_LINE = re.compile(
    rb"([0-9A-Fa-f]+)(?:[ \t]*;[ \t]*" + _HTTP_TOKEN
    + rb"(?:[ \t]*=[ \t]*(?:" + _HTTP_TOKEN + rb"|" + _HTTP_QUOTED
    + rb"))?)*\r\n"
)
_TRAILER_LINE = re.compile(_HTTP_TOKEN + rb":[\t\x20-\x7e\x80-\xff]*\r\n")

# llama.cpp auto-discovery (mirrors pi-llama-cpp's defaults).
DEFAULT_LLAMA_URL = "http://127.0.0.1:9931"
DEFAULT_LLAMA_CONTEXT = 128000
DEFAULT_LLAMA_REFRESH = 30.0
DEFAULT_LLAMA_PROBE_TIMEOUT = 2.0
LLAMA_LIMIT_FIELDS = (
    "context_window", "trained_context_window", "max_input_tokens",
    "max_output_tokens", "default_output_tokens",
)
REFRESH_BACKOFF_BASE = 5.0
REFRESH_BACKOFF_MAX = 300.0

_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
# Requests are decoded and reserialized as JSON. Original representation
# metadata, integrity checks and signatures cannot describe the new bytes.
_REQUEST_REPRESENTATION_HEADERS = {
    "content-type", "content-encoding", "content-length", "content-md5",
    "content-digest", "repr-digest", "digest", "content-range",
    "content-location", "content-language", "etag", "last-modified",
    "signature", "signature-input",
}
# Never forward client credentials or transport headers upstream.
_DROP_REQUEST_HEADERS = _HOP_BY_HOP | {
    "host",
    "content-length",
    "authorization",
    "x-api-key",
    "api-key",
    "x-goog-api-key",
    "cookie",
    "user-agent",
} | _REQUEST_REPRESENTATION_HEADERS
# Response bodies are decoded by httpx, so encoding/length headers no longer
# describe what we send.
_DROP_RESPONSE_HEADERS = _HOP_BY_HOP | {
    "content-length",
    "content-encoding",
    "set-cookie",
}

# This is a protocol boundary, deliberately independent of LiteLLM's signatures
# and **kwargs. New LiteLLM administrative options must never become client API.
_GENERATION_FIELDS = {
    DIALECT_OPENAI_CHAT: frozenset({
        "messages", "audio", "frequency_penalty", "function_call", "functions",
        "logit_bias", "logprobs", "max_completion_tokens", "max_tokens",
        "modalities", "n", "parallel_tool_calls", "prediction",
        "presence_penalty", "prompt_cache_key", "prompt_cache_retention",
        "reasoning_effort", "response_format", "safety_identifier", "seed",
        "service_tier", "stop", "store", "stream", "stream_options",
        "temperature", "tool_choice", "tools", "top_logprobs", "top_p",
        "user", "verbosity", "web_search_options",
    }),
    DIALECT_OPENAI_RESPONSES: frozenset({
        "background", "conversation", "include", "input", "instructions",
        "max_output_tokens", "max_tool_calls", "parallel_tool_calls",
        "previous_response_id", "prompt", "prompt_cache_key",
        "prompt_cache_retention", "reasoning", "safety_identifier", "service_tier",
        "store", "stream", "stream_options", "temperature", "text",
        "tool_choice", "tools", "top_logprobs", "top_p", "truncation", "user",
    }),
    DIALECT_ANTHROPIC: frozenset({
        "messages", "max_tokens", "system", "temperature", "top_p", "top_k",
        "stop_sequences", "stream", "thinking", "tool_choice", "tools",
        "metadata", "output_format", "output_config", "context_management",
    }),
}

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_ANTHROPIC_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    529: "overloaded_error",
}

_ANTHROPIC_ADAPTER = AnthropicAdapter()


# ---------------------------------------------------------------------------
# Errors and JSON helpers
# ---------------------------------------------------------------------------


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
) -> bytes:
    payload = _stream_error_payload(ingress, exc, sanitize)
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            return os.environ.get(match.group(1), "")

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def _load_config_file(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"Cannot read config {path}: {exc}") from exc

    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            data = json.loads(text)
        elif suffix == ".toml":
            import tomllib

            data = tomllib.loads(text)
        elif suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore[import-not-found]
            except ImportError as exc:
                raise SystemExit(
                    "YAML configs need PyYAML. Use TOML/JSON or install pyyaml."
                ) from exc
            data = yaml.safe_load(text)
        else:
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                import tomllib

                data = tomllib.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"Invalid config {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise SystemExit(f"Config {path} must contain an object/table at the top level")
    return _expand_env(data)


def _as_str(raw: Mapping[str, Any], key: str, *, where: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"{where}: '{key}' must be a non-empty string")
    return value.strip()


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


@dataclass(frozen=True)
class Provider:
    name: str
    dialect: str
    base_url: str
    api_key: str
    headers: Mapping[str, str]
    timeout: float
    litellm_provider: str
    litellm_kwargs: Mapping[str, Any]
    drop_params: bool
    session_header: str = ""
    auth_header: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "headers", _freeze(self.headers))
        object.__setattr__(self, "litellm_kwargs", _freeze(self.litellm_kwargs))

    @property
    def effective_api_key(self) -> str:
        return self.api_key.strip() or UNAUTHENTICATED_API_KEY

    def auth_headers(self) -> dict[str, str]:
        """Upstream credential headers, honoring an explicit auth_header."""
        key = self.effective_api_key
        if self.auth_header == "bearer":
            return {"Authorization": "Bearer " + key}
        if self.auth_header:
            return {self.auth_header: key}
        if self.dialect == DIALECT_ANTHROPIC:
            return {"x-api-key": key}
        if self.dialect == DIALECT_GEMINI:
            return {"x-goog-api-key": key}
        return {"Authorization": "Bearer " + key}

    def litellm_call_kwargs(
        self, *, skip_mcp: bool = False, session_id: str | None = None
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = _thaw(self.litellm_kwargs)
        api_base = self.base_url
        if self.litellm_provider == "anthropic":
            # LiteLLM appends /v1/messages itself, so an origin-style base is
            # what it wants. Accept the base_url forms the pass-through path
            # allows and normalize them here.
            for suffix in ("/v1/messages", "/v1"):
                if api_base.endswith(suffix):
                    api_base = api_base[: -len(suffix)]
                    break
        kwargs.setdefault("api_base", api_base or self.base_url)
        # Always truthy, including unauthenticated mode. Never permit a second
        # credential source in litellm_kwargs to change this selection.
        kwargs["api_key"] = self.effective_api_key
        extra_headers = dict(kwargs.pop("headers", {}) or {})
        for name, value in dict(kwargs.pop("extra_headers", {}) or {}).items():
            _set_header(extra_headers, name, value)
        for name, value in self.headers.items():
            _set_header(extra_headers, name, value)
        for name, value in self.auth_headers().items():
            _set_header(extra_headers, name, value)
        if self.session_header and session_id:
            if not any(
                key.lower() == self.session_header.lower() for key in extra_headers
            ):
                extra_headers[self.session_header] = session_id
        extra_headers = _serialized_json_headers(extra_headers)
        # Keep both truthy: adapters otherwise consult litellm.headers.
        kwargs["headers"] = dict(extra_headers)
        kwargs["extra_headers"] = extra_headers
        kwargs.setdefault("timeout", self.timeout)
        # Loss is handled before calling adapters, including in explicit lossy
        # mode. Never let LiteLLM silently remove additional parameters.
        kwargs["drop_params"] = False
        if skip_mcp:
            kwargs.setdefault("_skip_mcp_handler", True)
        return kwargs


@dataclass(frozen=True)
class Route:
    name: str
    provider_name: str
    upstream_model: str
    dialect: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    enabled: bool = True
    # Resolution pins the transport identity even after a newer registry is
    # published. Exclude credentials from repr/equality.
    _provider: Provider | None = field(default=None, repr=False, compare=False)
    _capabilities: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        object.__setattr__(self, "_capabilities", _freeze(self._capabilities))


@dataclass(frozen=True)
class RefreshState:
    # Monotonic process-local times; None also works with clocks starting at 0.
    last_success: float | None = None
    last_attempt: float | None = None
    next_attempt: float | None = None
    consecutive_failures: int = 0


@dataclass(frozen=True)
class CatalogSnapshot:
    routes: Mapping[str, Route] = field(default_factory=lambda: _freeze({}))
    refresh: RefreshState = field(default_factory=RefreshState)
    error: str | None = None
    unmapped: frozenset[str] = frozenset()


@dataclass(frozen=True)
class LlamaSnapshot:
    providers: Mapping[str, Provider] = field(default_factory=lambda: _freeze({}))
    routes: Mapping[str, Route] = field(default_factory=lambda: _freeze({}))
    public: tuple[Route, ...] = ()
    by_server: Mapping[str, tuple[Provider, Mapping[str, Route], tuple[Route, ...]]] = field(
        default_factory=lambda: _freeze({})
    )
    servers: tuple[Mapping[str, Any], ...] = ()
    refresh: RefreshState = field(default_factory=RefreshState)
    error: str | None = None


@dataclass(frozen=True)
class RegistrySnapshot:
    providers: Mapping[str, Provider]
    routes: Mapping[str, Route]
    public: tuple[Route, ...]
    catalog: CatalogSnapshot
    llama: LlamaSnapshot


@dataclass(frozen=True)
class CatalogRule:
    pattern: re.Pattern[str]
    provider_name: str


@dataclass(frozen=True)
class CatalogConfig:
    url: str
    headers: dict[str, str]
    timeout: float
    refresh_seconds: float
    rules: list[CatalogRule]
    # Exact IDs take precedence over optional regex rules. Values reference
    # trusted providers; catalog responses cannot supply endpoints or keys.
    model_providers: dict[str, str] = field(default_factory=dict)
    metadata_source: str | None = None


@dataclass(frozen=True)
class LlamaCppServer:
    """One llama.cpp / llama-swap endpoint to probe."""

    url: str
    name: str = ""
    api_key: str = ""
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LlamaCppConfig:
    servers: list[LlamaCppServer]
    headers: dict[str, str]
    probe_timeout: float
    refresh_seconds: float
    context_fallback: int
    prefix: str
    expose_aliases: bool
    limit_overrides: Mapping[str, Mapping[str, int | None]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "limit_overrides", _freeze(self.limit_overrides))


@dataclass
class Config:
    host: str
    port: int
    api_keys: list[str]
    max_body_bytes: int
    providers: dict[str, Provider]
    routes: dict[str, Route]
    public_routes: list[Route]
    catalog: CatalogConfig | None = None
    llama_cpp: LlamaCppConfig | None = None
    max_connections: int = DEFAULT_MAX_CONNECTIONS
    max_generations: int = DEFAULT_MAX_GENERATIONS
    header_read_timeout: float = DEFAULT_HEADER_READ_TIMEOUT
    body_read_timeout: float = DEFAULT_BODY_READ_TIMEOUT
    write_timeout: float = DEFAULT_WRITE_TIMEOUT
    capability_overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Also validate Config instances constructed directly by embedders.
        self.capability_overrides = _freeze(self.capability_overrides)
        _llama_provider_names(self)
        for name in ("max_connections", "max_generations"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise SystemExit(f"server.{name} must be a positive integer")
        if self.max_generations > self.max_connections:
            raise SystemExit("server.max_generations must not exceed max_connections")
        for name in ("header_read_timeout", "body_read_timeout", "write_timeout"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not 0 < value <= threading.TIMEOUT_MAX):
                raise SystemExit(
                    f"server.{name} must be a finite positive number "
                    f"<= {threading.TIMEOUT_MAX}"
                )


def _config_secrets(config: Config) -> list[str]:
    secrets = list(config.api_keys)

    def credentials(value: Any, sensitive: bool = False) -> None:
        if isinstance(value, Mapping):
            for name, item in value.items():
                credentials(item, sensitive or bool(re.search(
                    r"(?i)key|token|secret|password|credential|headers", str(name)
                )))
        elif isinstance(value, (list, tuple)):
            for item in value:
                credentials(item, sensitive)
        elif sensitive and isinstance(value, str) and value:
            secrets.append(value)
            if value.lower().startswith("bearer "):
                secrets.append(value[7:])

    def url_secrets(url: str) -> None:
        parsed = urlsplit(url)
        if parsed.password:
            secrets.append(unquote(parsed.password))
        credentials(dict(parse_qsl(parsed.query, keep_blank_values=True)))

    for provider in config.providers.values():
        secrets.append(provider.api_key)
        url_secrets(provider.base_url)
        credentials(provider.headers, True)
        credentials(provider.litellm_kwargs)
    if config.catalog:
        url_secrets(config.catalog.url)
        credentials(config.catalog.headers, True)
    if config.llama_cpp:
        credentials(config.llama_cpp.headers, True)
        for server in config.llama_cpp.servers:
            secrets.append(server.api_key)
            url_secrets(server.url)
            credentials(server.headers, True)
    return secrets


class _DiagnosticLogFilter(logging.Filter):
    """Sanitize diagnostics before handlers (including dependency handlers)."""

    def __init__(self, sanitize: ErrorSanitizer) -> None:
        super().__init__()
        self.sanitize = sanitize

    def filter(self, record: logging.LogRecord) -> bool:
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


def _resolve_credential(raw: Mapping[str, Any], *, where: str) -> str:
    """Read api_key / api_key_env / api_key_file from a config mapping."""
    api_key = raw.get("api_key")
    if api_key is None and raw.get("api_key_env"):
        env_name = raw["api_key_env"]
        if not isinstance(env_name, str):
            raise SystemExit(f"{where}: 'api_key_env' must be a string")
        api_key = os.environ.get(env_name)
        if api_key is None:
            raise SystemExit(
                f"{where}: environment variable {env_name!r} is not set"
            )
    if api_key is None and raw.get("api_key_file"):
        key_path = Path(str(raw["api_key_file"])).expanduser()
        try:
            api_key = key_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise SystemExit(f"{where}: cannot read api_key_file: {exc}") from exc
    if api_key is None:
        return ""
    if not isinstance(api_key, str):
        raise SystemExit(f"{where}: 'api_key' must be a string")
    api_key = api_key.strip()
    if api_key.startswith("os.environ/"):
        raise SystemExit(f"{where}: use api_key_env for environment references")
    return api_key


def _build_provider(name: str, raw: Mapping[str, Any]) -> Provider:
    where = f"providers.{name}"
    if not isinstance(raw, Mapping):
        raise SystemExit(f"{where}: must be a table/object")

    dialect = raw.get("dialect")
    if dialect not in DIALECTS:
        raise SystemExit(
            f"{where}: 'dialect' must be one of {sorted(DIALECTS)}, got {dialect!r}"
        )

    base_url = _as_str(raw, "base_url", where=where).rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise SystemExit(f"{where}: 'base_url' must start with http:// or https://")

    api_key = _resolve_credential(raw, where=where)

    headers_raw = raw.get("headers") or {}
    if not isinstance(headers_raw, Mapping):
        raise SystemExit(f"{where}: 'headers' must be a table/object")
    headers = {str(key): str(value) for key, value in headers_raw.items()}

    try:
        timeout = float(raw.get("timeout", DEFAULT_TIMEOUT))
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{where}: 'timeout' must be a number") from exc
    if timeout <= 0:
        raise SystemExit(f"{where}: 'timeout' must be positive")

    default_litellm_provider = {
        DIALECT_ANTHROPIC: "anthropic",
        DIALECT_GEMINI: "gemini",
    }.get(dialect, "openai")
    litellm_provider = str(raw.get("litellm_provider") or default_litellm_provider)

    litellm_kwargs_raw = raw.get("litellm_kwargs") or {}
    if not isinstance(litellm_kwargs_raw, Mapping):
        raise SystemExit(f"{where}: 'litellm_kwargs' must be a table/object")
    compatibility = raw.get("compatibility", "lossy" if raw.get("drop_params") is True else "strict")
    if compatibility not in ("strict", "lossy"):
        raise SystemExit(f"{where}: 'compatibility' must be 'strict' or 'lossy'")
    if "drop_params" in raw and (type(raw["drop_params"]) is not bool
                                 or raw["drop_params"] != (compatibility == "lossy")):
        raise SystemExit(f"{where}: 'drop_params' must be boolean and agree with 'compatibility'")

    session_header = raw.get("session_header") or ""
    if not isinstance(session_header, str):
        raise SystemExit(f"{where}: 'session_header' must be a string")
    session_header = session_header.strip()

    auth_header = raw.get("auth_header") or ""
    if not isinstance(auth_header, str):
        raise SystemExit(f"{where}: 'auth_header' must be a string")
    auth_header = auth_header.strip()
    if auth_header and auth_header.lower() not in ("bearer", "x-api-key", "api-key"):
        raise SystemExit(
            f"{where}: 'auth_header' must be 'bearer', 'x-api-key', 'api-key', "
            f"or empty, got {auth_header!r}"
        )
    auth_header = auth_header.lower()

    return Provider(
        name=name,
        dialect=dialect,
        base_url=base_url,
        api_key=api_key,
        headers=headers,
        timeout=timeout,
        litellm_provider=litellm_provider,
        litellm_kwargs=dict(litellm_kwargs_raw),
        drop_params=compatibility == "lossy",
        session_header=session_header,
        auth_header=auth_header,
    )


def _split_server_urls(value: str) -> list[str]:
    """Split LLAMA_SERVER_URL-style values (semicolon or comma separated)."""
    return [part.strip() for part in re.split(r"[;,]", value) if part.strip()]


def _normalize_llama_url(url: str, *, where: str) -> str:
    url = url.strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        raise SystemExit(f"{where}: server URL must start with http:// or https://")
    return url


def _build_llama_cpp(raw: Mapping[str, Any]) -> LlamaCppConfig:
    """Validate the [llama_cpp] table used for automatic model discovery."""
    where = "llama_cpp"
    if not isinstance(raw, Mapping):
        raise SystemExit(f"{where}: must be a table/object")

    headers_raw = raw.get("headers") or {}
    if not isinstance(headers_raw, Mapping):
        raise SystemExit(f"{where}: 'headers' must be a table/object")
    global_headers = {str(key): str(value) for key, value in headers_raw.items()}
    global_api_key = _resolve_credential(raw, where=where)

    try:
        probe_timeout = float(raw.get("timeout", DEFAULT_LLAMA_PROBE_TIMEOUT))
        refresh_seconds = float(
            raw.get("refresh_seconds", DEFAULT_LLAMA_REFRESH)
        )
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"{where}: 'timeout' and 'refresh_seconds' "
            "must be numbers"
        ) from exc
    if not math.isfinite(probe_timeout) or not 0 < probe_timeout <= threading.TIMEOUT_MAX:
        raise SystemExit(f"{where}: 'timeout' must be finite, positive and <= threading.TIMEOUT_MAX")
    if not math.isfinite(refresh_seconds) or not 0 <= refresh_seconds <= threading.TIMEOUT_MAX:
        raise SystemExit(f"{where}: 'refresh_seconds' must be finite, >= 0 and <= threading.TIMEOUT_MAX")
    context_fallback = raw.get("context_fallback", DEFAULT_LLAMA_CONTEXT)
    if type(context_fallback) is not int or context_fallback <= 0:
        raise SystemExit(f"{where}: 'context_fallback' must be a positive integer")

    overrides_raw = raw.get("limit_overrides", {})
    if not isinstance(overrides_raw, Mapping):
        raise SystemExit(f"{where}.limit_overrides: must be a table/object")
    limit_overrides: dict[str, dict[str, int | None]] = {}
    for model_id, limits in overrides_raw.items():
        label = f"{where}.limit_overrides[{model_id!r}]"
        if not isinstance(model_id, str) or not model_id.strip() or model_id != model_id.strip():
            raise SystemExit(f"{label}: model ID must be a nonempty string without surrounding whitespace")
        if not isinstance(limits, Mapping):
            raise SystemExit(f"{label}: must be a table/object")
        for key, value in limits.items():
            if key not in LLAMA_LIMIT_FIELDS:
                raise SystemExit(f"{label}: unknown limit {key!r}")
            minimum = 0 if key == "default_output_tokens" else 1
            if value is not None and (type(value) is not int or value < minimum):
                raise SystemExit(f"{label}.{key}: must be an integer >= {minimum} or null")
        limit_overrides[model_id] = dict(limits)

    prefix = raw.get("prefix") or ""
    if not isinstance(prefix, str):
        raise SystemExit(f"{where}: 'prefix' must be a string")

    servers_raw = raw.get("servers") or []
    if isinstance(servers_raw, (str, Mapping)):
        servers_raw = [servers_raw]
    if not isinstance(servers_raw, (list, tuple)):
        raise SystemExit(f"{where}: 'servers' must be an array")

    ports_raw = raw.get("ports") or []
    if isinstance(ports_raw, int):
        ports_raw = [ports_raw]
    if not isinstance(ports_raw, (list, tuple)):
        raise SystemExit(f"{where}: 'ports' must be an array of integers")

    entries: list[Any] = list(servers_raw)
    for index, port in enumerate(ports_raw):
        try:
            port_number = int(port)
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"{where}.ports[{index}]: port must be an integer"
            ) from exc
        if not 1 <= port_number <= 65535:
            raise SystemExit(
                f"{where}.ports[{index}]: port must be between 1 and 65535"
            )
        entries.append(f"http://127.0.0.1:{port_number}")

    if not entries:
        # Same fallback chain as pi-llama-cpp: environment first, then the
        # conventional llama-server port.
        entries = _split_server_urls(os.environ.get("LLAMA_SERVER_URL", ""))
        if not entries:
            entries = [DEFAULT_LLAMA_URL]

    servers: list[LlamaCppServer] = []
    seen: set[str] = set()
    explicit_names: dict[str, int] = {}
    for index, entry in enumerate(entries):
        entry_where = f"{where}.servers[{index}]"
        if isinstance(entry, str):
            url = _normalize_llama_url(entry, where=entry_where)
            name = ""
            api_key = global_api_key
            headers = dict(global_headers)
        elif isinstance(entry, Mapping):
            url = _normalize_llama_url(
                _as_str(entry, "url", where=entry_where), where=entry_where
            )
            name_raw = entry.get("name") or ""
            if not isinstance(name_raw, str):
                raise SystemExit(f"{entry_where}: 'name' must be a string")
            name = name_raw.strip()
            has_server_key = any(
                key in entry for key in ("api_key", "api_key_env", "api_key_file")
            )
            api_key = (
                _resolve_credential(entry, where=entry_where)
                if has_server_key
                else global_api_key
            )
            entry_headers_raw = entry.get("headers") or {}
            if not isinstance(entry_headers_raw, Mapping):
                raise SystemExit(
                    f"{entry_where}: 'headers' must be a table/object"
                )
            headers = dict(global_headers)
            headers.update(
                {str(key): str(value) for key, value in entry_headers_raw.items()}
            )
        else:
            raise SystemExit(
                f"{entry_where}: must be a URL string or a table with 'url'"
            )
        if name:
            if name in explicit_names:
                raise SystemExit(
                    f"{entry_where}.name: duplicate explicit provider name {name!r}; "
                    f"already used by llama_cpp.servers[{explicit_names[name]}]. "
                    "Assign each server a unique name or omit names to generate them."
                )
            explicit_names[name] = index
        if url in seen:
            continue
        seen.add(url)
        servers.append(
            LlamaCppServer(
                url=url, name=name, api_key=api_key, headers=headers
            )
        )

    return LlamaCppConfig(
        servers=servers,
        headers=global_headers,
        probe_timeout=probe_timeout,
        refresh_seconds=refresh_seconds,
        context_fallback=context_fallback,
        prefix=prefix,
        expose_aliases=bool(raw.get("aliases", True)),
        limit_overrides=limit_overrides,
    )


def build_config(raw: Mapping[str, Any]) -> Config:
    """Validate a config mapping and return a Config."""

    capabilities_raw = raw.get("capabilities", {})
    if not isinstance(capabilities_raw, Mapping) or capabilities_raw.keys() - {"overrides"}:
        raise SystemExit("capabilities: expected a table containing only 'overrides'")
    capability_overrides = router_metadata.validate_overrides(capabilities_raw.get("overrides", {}))
    server_raw = raw.get("server") or {}
    if not isinstance(server_raw, Mapping):
        raise SystemExit("server: must be a table/object")

    host = str(server_raw.get("host", "127.0.0.1"))
    try:
        port = int(server_raw.get("port", 4000))
    except (TypeError, ValueError) as exc:
        raise SystemExit("server.port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise SystemExit("server.port must be between 1 and 65535")

    keys_raw = server_raw.get("api_keys") or []
    if isinstance(keys_raw, str):
        keys_raw = [keys_raw]
    if not isinstance(keys_raw, (list, tuple)):
        raise SystemExit("server.api_keys must be an array of strings")
    api_keys = [str(key).strip() for key in keys_raw if str(key).strip()]

    env_keys = os.environ.get("MODEL_ROUTER_API_KEYS", "")
    if env_keys:
        api_keys.extend(part.strip() for part in env_keys.split(",") if part.strip())
    env_key = os.environ.get("MODEL_ROUTER_API_KEY", "").strip()
    if env_key:
        api_keys.append(env_key)
    # De-duplicate while preserving order.
    api_keys = list(dict.fromkeys(api_keys))

    try:
        max_body = int(server_raw.get("max_body_bytes", DEFAULT_MAX_BODY))
    except (TypeError, ValueError) as exc:
        raise SystemExit("server.max_body_bytes must be an integer") from exc
    if max_body <= 0:
        raise SystemExit("server.max_body_bytes must be positive")

    llama_cpp_raw = raw.get("llama_cpp")
    llama_cpp = (
        _build_llama_cpp(llama_cpp_raw) if llama_cpp_raw is not None else None
    )

    providers_raw = raw.get("providers") or {}
    if not isinstance(providers_raw, Mapping):
        raise SystemExit("providers: must be a table/object")
    if not providers_raw and llama_cpp is None:
        raise SystemExit(
            "providers: at least one provider is required (or configure "
            "[llama_cpp] for automatic local discovery)"
        )
    providers = {
        str(name): _build_provider(str(name), value)
        for name, value in providers_raw.items()
    }

    catalog_raw = raw.get("catalog")
    catalog: CatalogConfig | None = None
    if catalog_raw is not None:
        if not isinstance(catalog_raw, Mapping):
            raise SystemExit("catalog: must be a table/object")
        catalog_url = _as_str(catalog_raw, "url", where="catalog")
        if not catalog_url.startswith(("http://", "https://")):
            raise SystemExit("catalog: 'url' must start with http:// or https://")
        catalog_headers_raw = catalog_raw.get("headers") or {}
        if not isinstance(catalog_headers_raw, Mapping):
            raise SystemExit("catalog: 'headers' must be a table/object")
        try:
            catalog_timeout = float(catalog_raw.get("timeout", 30.0))
            catalog_refresh = float(catalog_raw.get("refresh_seconds", 300.0))
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                "catalog: 'timeout' and 'refresh_seconds' must be numbers"
            ) from exc
        if (not math.isfinite(catalog_timeout) or not math.isfinite(catalog_refresh)
                or not 0 < catalog_timeout <= threading.TIMEOUT_MAX
                or not 0 <= catalog_refresh <= threading.TIMEOUT_MAX):
            raise SystemExit(
                "catalog: 'timeout' must be positive and 'refresh_seconds' >= 0; "
                "both must be finite and <= threading.TIMEOUT_MAX"
            )
        model_providers_raw = catalog_raw.get("model_providers", {})
        if not isinstance(model_providers_raw, Mapping):
            raise SystemExit("catalog.model_providers: must be a table/object")
        model_providers: dict[str, str] = {}
        for model_id, provider_name in model_providers_raw.items():
            if (not isinstance(model_id, str) or not model_id.strip()
                    or model_id != model_id.strip()):
                raise SystemExit(
                    "catalog.model_providers: model IDs must be nonempty strings "
                    "without surrounding whitespace"
                )
            if not isinstance(provider_name, str) or provider_name not in providers:
                raise SystemExit(
                    f"catalog.model_providers[{model_id!r}]: unknown provider {provider_name!r}"
                )
            model_providers[model_id] = provider_name
        rules_raw = catalog_raw.get("rules", [])
        if not isinstance(rules_raw, (list, tuple)):
            raise SystemExit("catalog.rules: must be an array")
        if not rules_raw and not model_providers:
            raise SystemExit(
                "catalog: at least one model_providers entry or [[catalog.rules]] is required"
            )
        rules: list[CatalogRule] = []
        for index, rule_raw in enumerate(rules_raw):
            where = f"catalog.rules[{index}]"
            if not isinstance(rule_raw, Mapping):
                raise SystemExit(f"{where}: must be a table/object")
            match = _as_str(rule_raw, "match", where=where)
            rule_provider = _as_str(rule_raw, "provider", where=where)
            if rule_provider not in providers:
                raise SystemExit(
                    f"{where}: unknown provider {rule_provider!r}; "
                    f"known providers: {sorted(providers)}"
                )
            try:
                pattern = re.compile(match)
            except re.error as exc:
                raise SystemExit(f"{where}: invalid 'match' regex: {exc}") from exc
            rules.append(CatalogRule(pattern=pattern, provider_name=rule_provider))
        metadata_source = catalog_raw.get("metadata_source")
        if metadata_source not in (None, "opencode-go-v1"):
            raise SystemExit("catalog.metadata_source: expected 'opencode-go-v1' or null")
        catalog = CatalogConfig(
            url=catalog_url,
            headers={
                str(key): str(value)
                for key, value in catalog_headers_raw.items()
            },
            timeout=catalog_timeout,
            refresh_seconds=catalog_refresh,
            rules=rules,
            model_providers=model_providers,
            metadata_source=metadata_source,
        )

    models_raw = raw.get("models") or []
    if not isinstance(models_raw, (list, tuple)):
        raise SystemExit("models: must be an array")
    if not models_raw and catalog is None and llama_cpp is None:
        raise SystemExit(
            "models: at least one model route is required (or configure "
            "[catalog] / [llama_cpp])"
        )

    routes: dict[str, Route] = {}
    public_routes: list[Route] = []

    for index, model_raw in enumerate(models_raw):
        where = f"models[{index}]"
        if not isinstance(model_raw, Mapping):
            raise SystemExit(f"{where}: must be a table/object")
        name = _as_str(model_raw, "name", where=where)
        provider_name = _as_str(model_raw, "provider", where=where)
        provider = providers.get(provider_name)
        if provider is None:
            raise SystemExit(
                f"{where}: unknown provider {provider_name!r}; "
                f"known providers: {sorted(providers)}"
            )

        upstream_model = model_raw.get("model")
        if upstream_model is None:
            upstream_model = (
                name.split("/", 1)[1]
                if name.startswith(provider_name + "/") and "/" in name
                else name
            )
        if not isinstance(upstream_model, str) or not upstream_model.strip():
            raise SystemExit(f"{where}: 'model' must be a non-empty string")

        dialect = model_raw.get("dialect", provider.dialect)
        if dialect not in DIALECTS:
            raise SystemExit(
                f"{where}: 'dialect' must be one of {sorted(DIALECTS)}, got {dialect!r}"
            )

        metadata = model_raw.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise SystemExit(f"{where}: 'metadata' must be a table/object")

        route = Route(
            name=name,
            provider_name=provider_name,
            upstream_model=upstream_model.strip(),
            dialect=str(dialect),
            metadata=dict(metadata),
            enabled=bool(model_raw.get("enabled", True)),
        )

        aliases_raw = model_raw.get("aliases") or []
        if isinstance(aliases_raw, str):
            aliases_raw = [aliases_raw]
        if not isinstance(aliases_raw, (list, tuple)):
            raise SystemExit(f"{where}: 'aliases' must be an array of strings")

        for public_name in [name, *[str(alias) for alias in aliases_raw]]:
            public_name = public_name.strip()
            if not public_name:
                continue
            if public_name in routes:
                raise SystemExit(
                    f"{where}: duplicate model name/alias {public_name!r}"
                )
            routes[public_name] = route

        public_routes.append(route)

    return Config(
        host=host,
        port=port,
        api_keys=api_keys,
        max_body_bytes=max_body,
        providers=providers,
        routes=routes,
        public_routes=public_routes,
        catalog=catalog,
        llama_cpp=llama_cpp,
        capability_overrides=capability_overrides,
        **{name: server_raw.get(name, default) for name, default in (
            ("max_connections", DEFAULT_MAX_CONNECTIONS),
            ("max_generations", DEFAULT_MAX_GENERATIONS),
            ("header_read_timeout", DEFAULT_HEADER_READ_TIMEOUT),
            ("body_read_timeout", DEFAULT_BODY_READ_TIMEOUT),
            ("write_timeout", DEFAULT_WRITE_TIMEOUT),
        )},
    )


def load_config(path: Path) -> Config:
    return build_config(_load_config_file(path))


# ---------------------------------------------------------------------------
# Credentials / dotenv loading
# ---------------------------------------------------------------------------

_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def default_env_files() -> list[Path]:
    """Credential files loaded automatically before the config is read."""
    candidates = []
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        candidates.append(Path(xdg) / "ai" / "credentials.env")
    candidates.append(Path.home() / ".config" / "ai" / "credentials.env")
    seen: set[Path] = set()
    unique = []
    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def load_env_file(path: Path, *, override: bool = False) -> int:
    """Load KEY=value lines from a shell-style env file.

    Supports blank lines, # comments, an optional leading `export`, and single
    or double quoted values.  Existing environment variables win unless
    override is true.  Returns the number of keys seen.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"Cannot read env file {path}: {exc}") from exc

    count = 0
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, raw = line.partition("=")
        key = key.strip()
        if not separator or not _ENV_KEY.match(key):
            LOG.warning("Skipping malformed line %d in %s", line_number, path)
            continue
        value = raw.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        count += 1
        if override or key not in os.environ:
            os.environ[key] = value
    return count


def load_default_credentials() -> list[Path]:
    loaded = []
    for path in default_env_files():
        if path.is_file():
            load_env_file(path)
            loaded.append(path)
    return loaded


# ---------------------------------------------------------------------------
# Response plans
# ---------------------------------------------------------------------------


@dataclass
class Plan:
    """A prepared HTTP response.  Exactly one of body/chunks/async_chunks is set."""

    status: int = 200
    content_type: str = "application/json"
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes | None = None
    chunks: Iterator[bytes] | None = None
    async_chunks: AsyncIterator[bytes] | None = None
    closer: Callable[[], None] | None = None


def _chat_sse(
    iterator: Iterator[Any], sanitize: ErrorSanitizer = _DEFAULT_SANITIZER,
) -> Iterator[bytes]:
    for chunk in iterator:
        payload = _sanitize_generated_errors(_jsonable(chunk), sanitize)
        yield b"data: " + _json_bytes(payload) + b"\n\n"
    yield b"data: [DONE]\n\n"


def _responses_sse(
    iterator: Iterator[Any], sanitize: ErrorSanitizer = _DEFAULT_SANITIZER,
) -> Iterator[bytes]:
    sequence = 0
    for event in iterator:
        payload = _sanitize_generated_errors(_jsonable(event), sanitize)
        if not isinstance(payload, dict):
            payload = {"type": "response.event", "data": payload}
        # LiteLLM's translated Responses streams do not fill sequence_number;
        # OpenAI clients expect it to be present and monotonic.
        number = payload.get("sequence_number")
        if not isinstance(number, int):
            number = sequence
        payload["sequence_number"] = number
        sequence = number + 1
        event_type = str(payload.get("type") or "message")
        yield (
            b"event: "
            + event_type.encode("utf-8")
            + b"\ndata: "
            + _json_bytes(payload)
            + b"\n\n"
        )


# ---------------------------------------------------------------------------
# Router application
# ---------------------------------------------------------------------------


def _header_value(headers: Mapping[str, str], name: str) -> str | None:
    value = headers.get(name)
    if value:
        return value
    lower = name.lower()
    for key, candidate in headers.items():
        if key.lower() == lower and candidate:
            return candidate
    return None


def _session_id(
    provider: Provider, payload: Mapping[str, Any], inbound: Mapping[str, str]
) -> str | None:
    """An upstream session id, taken from the client or generated per request."""
    if not provider.session_header:
        return None
    value = _header_value(inbound, provider.session_header)
    if not value:
        cache_key = payload.get("prompt_cache_key")
        if isinstance(cache_key, str):
            value = cache_key
    if not value or not str(value).strip():
        value = "router-" + uuid.uuid4().hex
    return str(value).strip()


# ---------------------------------------------------------------------------
# llama.cpp discovery helpers (ported from pi-llama-cpp)
# ---------------------------------------------------------------------------


def _llama_mode(props: Any, rows: list[Mapping[str, Any]]) -> str:
    """Classify a llama.cpp endpoint the same way pi-llama-cpp does."""
    if isinstance(props, Mapping):
        if props.get("src") == "llama-swap":
            return "llama-swap"
        if props.get("role") == "router":
            return "router"
    if rows and "max_model_len" in rows[0]:
        return "legacy"
    return "single"


def _llama_arg(args: Any, flags: tuple[str, ...]) -> int | None:
    """Read the last occurrence, including equals syntax; never execute args.

    A final invalid/auto value must not revive an earlier positive value.
    """
    if not isinstance(args, (list, tuple)):
        return None
    result = None
    for index, arg in enumerate(args):
        if not isinstance(arg, str):
            continue
        flag, sep, value = arg.partition("=")
        if flag not in flags:
            continue
        if not sep:
            value = args[index + 1] if index + 1 < len(args) else None
        try:
            result = int(value) if isinstance(value, str) and re.fullmatch(r"-?[0-9]+", value) else None
        except ValueError:  # also rejects integer strings beyond Python's size guard
            result = None
    return result


def _llama_context_from_args(args: Any) -> int | None:
    value = _llama_arg(args, ("--ctx-size", "-c"))
    return value if value is not None and value > 0 else None


def _limit_number(value: Any, *, minimum: int = 1) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        return None
    return int(value) if value >= minimum else None


def _llama_limits(
    row: Mapping[str, Any], props: Any, mode: str, fallback: int,
    overrides: Mapping[str, int | None], *, per_model_props: bool = False,
) -> dict[str, Any]:
    """Keep reported runtime capacity, training metadata and defaults distinct.

    Router/swap global props cannot describe any particular model. The caller
    must explicitly identify props fetched for that model with autoload=false.
    """
    def mapping(value: Any) -> Mapping[str, Any]:
        return value if isinstance(value, Mapping) else {}

    meta = mapping(row.get("meta"))
    metadata = mapping(row.get("metadata"))
    props = mapping(props) if mode in ("single", "legacy") or per_model_props else {}
    props_source = "/props?model=<id>" if per_model_props else "/props"
    defaults = mapping(props.get("default_generation_settings"))
    params = mapping(defaults.get("params"))
    args = mapping(row.get("status")).get("args")
    result: dict[str, Any] = {
        key: {"value": None, "source": None, "kind": "unknown"}
        for key in LLAMA_LIMIT_FIELDS
    }

    def choose(key: str, candidates: list[tuple[Any, str]], *, minimum: int = 1) -> None:
        for raw, source in candidates:
            value = _limit_number(raw, minimum=minimum)
            if value is not None:
                result[key] = {"value": value, "source": source, "kind": "reported"}
                return

    context = [(meta.get("n_ctx"), "/v1/models.meta.n_ctx")]
    if mode == "legacy":
        context.insert(0, (row.get("max_model_len"), "/v1/models.max_model_len"))
    choose("context_window", context + [
        (defaults.get("n_ctx"), props_source + ".default_generation_settings.n_ctx"),
        (props.get("n_ctx"), props_source + ".n_ctx"),
    ])
    if result["context_window"]["value"] is None:
        requested = _llama_context_from_args(args)
        result["context_window"] = (
            {"value": requested, "source": "/v1/models.status.args.ctx-size", "kind": "inferred"}
            if requested is not None else
            {"value": fallback, "source": "llama_cpp.context_fallback", "kind": "fallback"}
        )
    choose("trained_context_window", [
        (meta.get("n_ctx_train"), "/v1/models.meta.n_ctx_train"),
        # llama.cpp router listings use context_length for trained capacity.
        (row.get("context_length") if mode == "router" else None, "/v1/models.context_length"),
        (props.get("n_ctx_train"), props_source + ".n_ctx_train"),
    ])
    # These names explicitly denote hard limits. Ambiguous max_tokens and
    # n_predict are never promoted to maximum output capacity.
    for key in ("max_input_tokens", "max_output_tokens"):
        choose(key, [(meta.get(key), "/v1/models.meta." + key),
                     (row.get(key), "/v1/models." + key),
                     (metadata.get(key), "/v1/models.metadata." + key),
                     (props.get(key), props_source + "." + key)])

    for raw, source in [
        (params.get("n_predict"), props_source + ".default_generation_settings.params.n_predict"),
        (defaults.get("n_predict"), props_source + ".default_generation_settings.n_predict"),
        (params.get("max_tokens"), props_source + ".default_generation_settings.params.max_tokens"),
        (defaults.get("max_tokens"), props_source + ".default_generation_settings.max_tokens"),
        (_llama_arg(args, ("--predict", "--n-predict", "-n")), "/v1/models.status.args.n-predict"),
    ]:
        value = _limit_number(raw, minimum=-2)
        if value is None:
            continue
        result["default_output_tokens"] = {
            "value": value if value >= 0 else None, "source": source,
            "kind": "inferred" if source.startswith("/v1/models.status.args") else "reported",
        }
        if value < 0:
            result["default_output_tokens"]["special"] = "unlimited" if value == -1 else "until_context"
        break

    for key, value in overrides.items():
        result[key] = {"value": value, "source": "llama_cpp.limit_overrides", "kind": "override"}

    fit = _llama_arg(args, ("--fit-ctx", "-fitc"))
    result["constraints"] = {}
    if fit is not None and fit > 0:
        result["constraints"]["fit_context_min"] = {
            "value": fit, "source": "/v1/models.status.args.fit-ctx", "kind": "constraint",
        }
    return result


def _llama_capabilities(row: Mapping[str, Any], mode: str) -> list[str]:
    architecture = row.get("architecture")
    if isinstance(architecture, Mapping):
        modalities = architecture.get("input_modalities")
        if isinstance(modalities, (list, tuple)):
            return ["text", "image"] if "image" in modalities else ["text"]
    if mode == "legacy":
        capabilities = row.get("capabilities")
        if isinstance(capabilities, (list, tuple)) and any(
            str(item).lower() == "multimodal" for item in capabilities
        ):
            return ["text", "image"]
    return ["text"]


def _llama_model_aliases(row: Mapping[str, Any]) -> list[str]:
    """Aliases reported by llama.cpp, preferring llama-swap's nested list."""
    meta = row.get("meta")
    candidates: Any = None
    if isinstance(meta, Mapping):
        llamaswap = meta.get("llamaswap")
        if isinstance(llamaswap, Mapping):
            candidates = llamaswap.get("aliases")
    if not isinstance(candidates, (list, tuple)):
        candidates = row.get("aliases")
    if isinstance(candidates, str):
        candidates = [candidates]
    if not isinstance(candidates, (list, tuple)):
        return []
    return [str(alias).strip() for alias in candidates if str(alias).strip()]


def _llama_provider_name(server: LlamaCppServer, taken: set[str]) -> str:
    """Choose a generated name against identities reserved before probing."""
    if server.name:
        base = server.name
    else:
        parsed = urlsplit(server.url)
        base = "llama-" + (parsed.netloc + parsed.path).strip("/")
        base = re.sub(r"[^A-Za-z0-9_.-]+", "-", base).strip("-") or "llama"
    name = base
    index = 2
    while name in taken:
        name = f"{base}-{index}"
        index += 1
    return name


def _llama_provider_names(config: Config) -> dict[str, str]:
    """Reserve explicit names first, then generated names in config order."""
    if config.llama_cpp is None:
        return {}
    taken = set(config.providers)
    explicit: dict[str, int] = {}
    for index, server in enumerate(config.llama_cpp.servers):
        if not server.name:
            continue
        where = f"llama_cpp.servers[{index}].name"
        if server.name in explicit:
            raise SystemExit(
                f"{where}: duplicate explicit provider name {server.name!r}; "
                f"already used by llama_cpp.servers[{explicit[server.name]}]. "
                "Assign each server a unique name or omit names to generate them."
            )
        if server.name in config.providers:
            raise SystemExit(
                f"{where}: provider name {server.name!r} is already configured "
                "in providers; choose a unique server name or omit it."
            )
        explicit[server.name] = index
        taken.add(server.name)
    names: dict[str, str] = {}
    for server in config.llama_cpp.servers:
        name = server.name or _llama_provider_name(server, taken)
        names[server.url] = name
        taken.add(name)
    return names


def _llama_base_url(url: str) -> str:
    """The OpenAI-compatible base URL of a llama.cpp server."""
    return url if url.endswith("/v1") else url + "/v1"


class App:
    def __init__(
        self, config: Config, *, clock: Callable[[], float] = time.monotonic,
        jitter: Callable[[float, float], float] = random.uniform,
    ) -> None:
        self.config = config
        self._clock = clock
        self._jitter = jitter
        self._closed = threading.Event()
        self._close_lock = threading.Lock()
        self._worker_lock = threading.Lock()
        self._worker_stop = threading.Event()
        self._refresh_wakeup = threading.Event()
        self._refresh_worker: threading.Thread | None = None
        self.sanitize = ErrorSanitizer(_config_secrets(config))
        self._log_filter = _DiagnosticLogFilter(self.sanitize)
        # Logger filters do not propagate to children. Include the dependency
        # loggers themselves, which can log HTTP URLs and adapter exceptions.
        names = {"model-router", "LiteLLM", "LiteLLM Router", "LiteLLM Proxy",
                 "httpx", "httpcore", "openai"}
        names.update(name for name in logging.Logger.manager.loggerDict
                     if name.startswith(("LiteLLM", "litellm", "httpx", "httpcore", "openai")))
        self._diagnostic_loggers = [logging.getLogger(name) for name in names]
        for logger in self._diagnostic_loggers:
            logger.addFilter(self._log_filter)
        self.http = httpx.Client(
            follow_redirects=False,
            timeout=httpx.Timeout(DEFAULT_TIMEOUT, connect=20.0),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
        )
        self.verbose = False
        self._registry_lock = threading.Lock()
        self._llama_names = _llama_provider_names(config)
        self._registry = RegistrySnapshot(
            providers=_freeze({}), routes=_freeze({}), public=(),
            catalog=CatalogSnapshot(), llama=LlamaSnapshot(),
        )
        self._publish_registry()
        self._catalog_lock = threading.Lock()
        self._llama_lock = threading.Lock()
        self._llama_unreachable: set[str] = set()
        self._llama_counts: dict[str, int] = {}
        # One synchronous initial inventory preserves startup/--check behavior.
        # No background thread is owned until start() (or serve()) is called.
        try:
            self.refresh_llama_cpp(force=True)
            self.refresh_catalog(force=True)
        except BaseException:
            self.close()
            raise

    def start(self) -> None:
        """Start scheduled discovery. Idempotent; serve() calls this after bind."""
        with self._worker_lock:
            if self._closed.is_set():
                raise RuntimeError("Cannot start a closed router")
            if self._refresh_worker is not None:
                return
            if not any(source and source.refresh_seconds > 0
                       for source in (self.config.catalog, self.config.llama_cpp)):
                return
            self._worker_stop.clear()
            self._refresh_worker = threading.Thread(
                target=self._run_refresh_worker, name="model-router-discovery", daemon=False,
            )
            try:
                self._refresh_worker.start()
            except BaseException:
                self._refresh_worker = None
                raise

    def stop(self) -> None:
        """Stop scheduling and join active discovery before returning.

        An active HTTP probe finishes under its configured timeout. start() can
        resume scheduling after stop(); close() permanently releases resources.
        """
        with self._worker_lock:
            self._worker_stop.set()
            self._refresh_wakeup.set()
            if self._refresh_worker is not None:
                self._refresh_worker.join()
                self._refresh_worker = None

    def _run_refresh_worker(self) -> None:
        while not self._worker_stop.is_set():
            self._refresh_wakeup.clear()
            for refresh in (self.refresh_llama_cpp, self.refresh_catalog):
                if self._worker_stop.is_set():
                    return
                refresh()
            snapshot = self._registry
            # A manual refresh may own a source. Its completion wakes us;
            # exclude its old deadline so an overdue attempt cannot spin here.
            deadlines = [state.refresh.next_attempt
                         for state, lock in ((snapshot.catalog, self._catalog_lock),
                                             (snapshot.llama, self._llama_lock))
                         if state.refresh.next_attempt is not None and not lock.locked()]
            delay = max(0.0, min(deadlines) - self._clock()) if deadlines else None
            self._refresh_wakeup.wait(
                min(delay, threading.TIMEOUT_MAX) if delay is not None else None
            )

    def _next_refresh(
        self, previous: RefreshState, interval: float, attempted: float, *, failed: bool,
    ) -> RefreshState:
        completed = self._clock()
        failures = previous.consecutive_failures + 1 if failed else 0
        if failed:
            ceiling = min(REFRESH_BACKOFF_MAX,
                          REFRESH_BACKOFF_BASE * 2 ** min(failures - 1, 16))
            delay = self._jitter(ceiling / 2, ceiling)
        else:
            delay = interval
        return RefreshState(
            last_success=previous.last_success if failed else completed,
            last_attempt=attempted,
            next_attempt=completed + delay if interval > 0 else None,
            consecutive_failures=failures,
        )

    def _refresh_status(
        self, state: RefreshState, interval: float, error: str | None, now: float,
    ) -> dict[str, Any]:
        age = max(0.0, now - state.last_success) if state.last_success is not None else None
        return {
            "last_success": state.last_success,
            "last_attempt": state.last_attempt,
            "next_attempt": state.next_attempt,
            "consecutive_failures": state.consecutive_failures,
            "age_seconds": age,
            "stale": state.last_success is None or error is not None
                     or (interval > 0 and age is not None and age >= interval),
            "stale_age_seconds": (max(0.0, age - interval) if interval > 0 else 0.0)
                                 if age is not None else None,
        }

    def _publish_registry(
        self, *, catalog: CatalogSnapshot | None = None,
        llama: LlamaSnapshot | None = None,
    ) -> None:
        """Merge the latest source generations and publish one immutable value.

        Discovery source locks serialize their own network work. This short
        lock prevents simultaneous catalog/local publications losing updates;
        request dispatch never holds it, including during upstream I/O.
        """
        with self._registry_lock:
            previous = self._registry
            catalog = catalog if catalog is not None else previous.catalog
            llama = llama if llama is not None else previous.llama
            providers = {**self.config.providers, **llama.providers}
            bound: dict[int, Route] = {}

            def bind(route: Route) -> Route:
                if id(route) not in bound:
                    bound[id(route)] = replace(
                        route, _provider=providers[route.provider_name]
                    )
                return bound[id(route)]

            routes = {**catalog.routes, **llama.routes, **self.config.routes}
            public: list[Route] = []
            # Explicit aliases also reserve names against public discovery.
            names: set[str] = set()
            for source, entries in (
                (self.config.routes, self.config.public_routes),
                (llama.routes, llama.public),
                (catalog.routes, catalog.routes.values()),
            ):
                for route in entries:
                    if route.enabled and route.name not in names:
                        public.append(bind(route))
                names.update(source)
            self._registry = RegistrySnapshot(
                providers=_freeze(providers),
                routes=_freeze({name: bind(route) for name, route in routes.items()}),
                public=tuple(public), catalog=catalog, llama=llama,
            )

    # -- auth / inventory -------------------------------------------------

    def authorized(self, headers: Mapping[str, str]) -> bool:
        if not self.config.api_keys:
            return True
        candidates: list[str] = []
        authorization = headers.get("Authorization", "")
        if authorization.lower().startswith("bearer "):
            candidates.append(authorization[7:].strip())
        for name in ("x-api-key", "api-key"):
            value = headers.get(name)
            if value:
                candidates.append(value.strip())
        if not candidates:
            return False
        return any(
            hmac.compare_digest(candidate, key)
            for candidate in candidates
            for key in self.config.api_keys
        )

    def _catalog_is_stale(self) -> bool:
        deadline = self._registry.catalog.refresh.next_attempt
        return deadline is not None and self._clock() >= deadline

    def refresh_catalog(self, *, force: bool = False) -> None:
        """Explicit refresh; overlapping calls return immediately, even if forced."""
        catalog = self.config.catalog
        if catalog is None or not self._catalog_lock.acquire(blocking=False):
            return
        try:
            if self._closed.is_set() or (not force and not self._catalog_is_stale()):
                return
            previous = self._registry.catalog
            attempted = self._clock()
            try:
                self._load_catalog(catalog, previous, attempted)
            except Exception as exc:  # noqa: BLE001 - discovery is best-effort
                error = _error_message(exc, self.sanitize)
                self._publish_registry(catalog=replace(
                    previous, error=error,
                    refresh=self._next_refresh(previous.refresh, catalog.refresh_seconds,
                                               attempted, failed=True),
                ))
                LOG.warning("Model catalog refresh failed: %s", error)
        finally:
            self._catalog_lock.release()
            if threading.current_thread() is not self._refresh_worker:
                self._refresh_wakeup.set()

    def _load_catalog(
        self, catalog: CatalogConfig, previous: CatalogSnapshot, attempted: float,
    ) -> None:
        request_headers = {"User-Agent": f"litellm-model-router/{__version__}"}
        request_headers.update(catalog.headers)
        response = self.http.get(
            catalog.url,
            headers=request_headers,
            timeout=httpx.Timeout(catalog.timeout, connect=catalog.timeout),
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("data") if isinstance(payload, Mapping) else payload
        if not isinstance(rows, list):
            raise ValueError("catalog response has no model list")

        routes: dict[str, Route] = {}
        unmapped: set[str] = set()
        for row in rows:
            if isinstance(row, Mapping):
                model_id = row.get("id")
                metadata = row.get("metadata")
            else:
                model_id = row
                metadata = None
            if not isinstance(model_id, str) or not model_id.strip():
                continue
            model_id = model_id.strip()
            provider_name = catalog.model_providers.get(model_id)
            if provider_name is None:
                for rule in catalog.rules:
                    if rule.pattern.search(model_id):
                        provider_name = rule.provider_name
                        break
            if provider_name is None:
                unmapped.add(model_id)
                continue
            provider = self.config.providers[provider_name]
            capabilities = (router_metadata.go(model_id, provider.dialect)
                            if catalog.metadata_source == "opencode-go-v1"
                            else router_metadata.empty(provider.dialect))
            capabilities["sources"]["catalog"] = {
                "retrieved_at": time.time(), "verification": "availability only",
            }
            routes[model_id] = Route(
                name=model_id,
                provider_name=provider_name,
                upstream_model=model_id,
                dialect=provider.dialect,
                metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
                _capabilities=capabilities,
            )

        # A valid catalog containing only unreviewed IDs is a successful
        # empty routing inventory, not a reason to retain removed routes.
        self._publish_registry(catalog=CatalogSnapshot(
            routes=_freeze(routes), unmapped=frozenset(unmapped),
            refresh=self._next_refresh(previous.refresh, catalog.refresh_seconds,
                                       attempted, failed=False),
        ))
        unmapped = set(self.catalog_status()["unmapped_models"])
        if unmapped:
            LOG.warning(
                "Catalog has %d model(s) with unknown endpoint mapping; "
                "omitted from /v1/models: %s. Configure catalog.model_providers "
                "or an explicit [[models]] route after reviewing the endpoint.",
                len(unmapped), ", ".join(sorted(unmapped)),
            )
        LOG.info(
            "Model catalog loaded %d models from %s", len(routes), catalog.url
        )

    def catalog_status(self, snapshot: RegistrySnapshot | None = None) -> dict[str, Any]:
        snapshot = snapshot if snapshot is not None else self._registry
        state = snapshot.catalog
        catalog = self.config.catalog
        return {
            "enabled": catalog is not None,
            "url": self.sanitize(catalog.url) if catalog else None,
            "models": len(state.routes),
            "loaded": state.refresh.last_success is not None,
            "error": state.error,
            "refresh": self._refresh_status(state.refresh, catalog.refresh_seconds if catalog else 0,
                                            state.error, self._clock()),
            "unmapped_models": sorted(
                state.unmapped - snapshot.routes.keys()
            ),
        }

    # -- llama.cpp auto-discovery -----------------------------------------

    def provider(self, name: str) -> Provider:
        """Look up a provider in the current snapshot (resolved routes pin theirs)."""
        return self._registry.providers[name]

    def provider_count(self) -> int:
        return len(self._registry.providers)

    def _llama_is_stale(self) -> bool:
        deadline = self._registry.llama.refresh.next_attempt
        return deadline is not None and self._clock() >= deadline

    def _probe_llama_server(
        self, llama: LlamaCppConfig, server: LlamaCppServer
    ) -> dict[str, Any]:
        """Fetch /health, /v1/models and /props from one llama.cpp server."""
        headers = {"User-Agent": f"litellm-model-router/{__version__}"}
        headers.update(llama.headers)
        headers.update(server.headers)
        if server.api_key:
            headers["Authorization"] = "Bearer " + server.api_key
        timeout = httpx.Timeout(
            llama.probe_timeout, connect=llama.probe_timeout
        )
        base = server.url.rstrip("/")
        try:
            health = self.http.get(base + "/health", headers=headers, timeout=timeout)
            if health.status_code >= 400:
                raise RuntimeError(f"/health returned HTTP {health.status_code}")
            response = self.http.get(
                base + "/v1/models", headers=headers, timeout=timeout
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # noqa: BLE001 - discovery is best-effort
            return {"server": server, "error": _error_message(exc, self.sanitize)}

        try:
            props_response = self.http.get(
                base + "/props",
                params={"autoload": "false"},
                headers=headers,
                timeout=timeout,
            )
            props = (
                props_response.json()
                if props_response.status_code < 400
                else {}
            )
        except Exception:  # noqa: BLE001 - props is optional
            props = {}

        if isinstance(payload, Mapping):
            rows = payload.get("data")
            if not isinstance(rows, list):
                rows = payload.get("models")
        else:
            rows = payload
        if not isinstance(rows, list):
            return {"server": server, "error": "response has no model list"}

        rows = [row for row in rows if isinstance(row, Mapping)]
        model_props: dict[str, Mapping[str, Any]] = {}
        if _llama_mode(props, rows) == "router":
            for row in rows:
                model_id = row.get("id")
                status = row.get("status")
                if (not isinstance(model_id, str) or not model_id.strip()
                        or not isinstance(status, Mapping)
                        or status.get("value") not in ("loaded", "sleeping")):
                    continue
                try:
                    response = self.http.get(
                        base + "/props", params={"model": model_id, "autoload": "false"},
                        headers=headers, timeout=timeout,
                    )
                    if response.status_code < 400:
                        value = response.json()
                        # Older routers may ignore the model query and return
                        # their global router props. Those are not model data.
                        if (isinstance(value, Mapping)
                                and value.get("role") != "router"
                                and value.get("src") != "llama-swap"):
                            model_props[model_id.strip()] = value
                except Exception:  # noqa: BLE001 - older servers may lack per-model props
                    pass

        return {
            "server": server,
            "rows": rows,
            "props": props,
            "model_props": model_props,
        }

    def _log_llama_probe(
        self, server: LlamaCppServer, error: str | None, models: int
    ) -> None:
        """Log reachability changes without spamming every refresh."""
        if error:
            if server.url not in self._llama_unreachable:
                LOG.warning("llama.cpp server %s unavailable: %s", server.url, error)
            else:
                LOG.debug("llama.cpp server %s still unavailable: %s", server.url, error)
            self._llama_unreachable.add(server.url)
            return
        was_unreachable = server.url in self._llama_unreachable
        previous = self._llama_counts.get(server.url)
        if was_unreachable or previous != models:
            LOG.info(
                "llama.cpp server %s ready with %d model%s",
                server.url,
                models,
                "" if models == 1 else "s",
            )
        self._llama_unreachable.discard(server.url)
        self._llama_counts[server.url] = models

    def refresh_llama_cpp(self, *, force: bool = False) -> None:
        """Probe every configured llama.cpp server and rebuild its routes."""
        llama = self.config.llama_cpp
        if llama is None:
            return
        if not self._llama_lock.acquire(blocking=False):
            return
        try:
            if self._closed.is_set() or (not force and not self._llama_is_stale()):
                return
            attempted = self._clock()
            previous = self._registry.llama
            try:
                self._load_llama_cpp(llama, previous, attempted)
            except Exception as exc:  # noqa: BLE001 - retain last usable inventory
                error = _error_message(exc, self.sanitize)
                self._publish_registry(llama=replace(
                    previous, error=error,
                    refresh=self._next_refresh(previous.refresh, llama.refresh_seconds,
                                               attempted, failed=True),
                ))
                LOG.warning("llama.cpp discovery failed: %s", error)
        finally:
            self._llama_lock.release()
            if threading.current_thread() is not self._refresh_worker:
                self._refresh_wakeup.set()

    def _load_llama_cpp(
        self, llama: LlamaCppConfig, previous_state: LlamaSnapshot, attempted: float,
    ) -> None:
        outcomes: list[dict[str, Any] | None] = [None] * len(llama.servers)
        if llama.servers:
            workers = min(8, len(llama.servers))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(self._probe_llama_server, llama, server): index
                    for index, server in enumerate(llama.servers)
                }
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        outcomes[index] = future.result()
                    except Exception as exc:  # noqa: BLE001
                        outcomes[index] = {
                            "server": llama.servers[index],
                            "error": _error_message(exc, self.sanitize),
                        }

        explicit_names = set(self.config.routes)
        providers: dict[str, Provider] = {}
        routes: dict[str, Route] = {}
        public: list[Route] = []
        by_server: dict[
            str, tuple[Provider, Mapping[str, Route], tuple[Route, ...]]
        ] = {}
        states: list[dict[str, Any]] = []
        errors: list[str] = []
        reachable = 0
        previous_servers = {server["url"]: server for server in previous_state.servers}

        for index, server in enumerate(llama.servers):
            outcome = outcomes[index]
            if outcome is None or outcome.get("error"):
                error = self.sanitize(
                    outcome["error"]
                    if outcome and outcome.get("error")
                    else "probe failed"
                )
                # Keep the last good routes for this server during a
                # transient failure instead of dropping them.
                previous = previous_state.by_server.get(server.url)
                if previous is not None:
                    old_provider, old_routes, old_public = previous
                    providers[old_provider.name] = old_provider
                    for public_name, route in old_routes.items():
                        if (
                            public_name not in explicit_names
                            and public_name not in routes
                        ):
                            routes[public_name] = route
                    known = {route.name for route in public}
                    for route in old_public:
                        if (
                            route.name not in explicit_names
                            and route.name not in known
                        ):
                            public.append(route)
                            known.add(route.name)
                    by_server[server.url] = previous
                    model_count = len(old_public)
                else:
                    model_count = 0
                self._log_llama_probe(server, error, model_count)
                errors.append(self.sanitize(f"{server.url}: {error}"))
                states.append(
                    {
                        "url": server.url,
                        "reachable": False,
                        "mode": None,
                        "provider": self._llama_names[server.url],
                        "models": model_count,
                        "stale": previous is not None,
                        "error": error,
                        "last_success": previous_servers.get(server.url, {}).get("last_success"),
                        "last_attempt": attempted,
                    }
                )
                continue

            reachable += 1
            rows = outcome["rows"]
            props = outcome["props"]
            mode = _llama_mode(props, rows)
            provider_name = self._llama_names[server.url]
            provider = Provider(
                name=provider_name,
                dialect=DIALECT_OPENAI_CHAT,
                base_url=_llama_base_url(server.url),
                api_key=server.api_key,
                headers={**llama.headers, **server.headers},
                timeout=DEFAULT_TIMEOUT,
                litellm_provider="openai",
                litellm_kwargs={},
                drop_params=False,
            )
            providers[provider_name] = provider
            server_routes: dict[str, Route] = {}
            server_public: list[Route] = []
            alias_candidates: list[tuple[list[str], Route]] = []

            for row in rows:
                model_id = row.get("id")
                if not isinstance(model_id, str) or not model_id.strip():
                    continue
                model_id = model_id.strip()
                aliases = _llama_model_aliases(row)
                display_name = aliases[0] if aliases else model_id
                model_props = outcome.get("model_props", {})
                per_model_props = model_id in model_props
                limits = _llama_limits(
                    row, model_props[model_id] if per_model_props else props,
                    mode, llama.context_fallback, llama.limit_overrides.get(model_id, {}),
                    per_model_props=per_model_props,
                )
                context = limits["context_window"]["value"]
                capabilities = _llama_capabilities(row, mode)
                status = row.get("status")
                status_value = (
                    status.get("value") if isinstance(status, Mapping) else None
                )
                if status_value is None and mode in ("single", "legacy"):
                    # These servers only ever run one loaded model.
                    status_value = "loaded"
                metadata = (
                    dict(row["metadata"])
                    if isinstance(row.get("metadata"), Mapping)
                    else {}
                )
                metadata.update(
                    {
                        "display_name": display_name,
                        "context_window": context,
                        # Compatibility alias: maximum generation length only.
                        # null means unknown, never the context or a default.
                        "max_tokens": limits["max_output_tokens"]["value"],
                        "limits": limits,
                        "capabilities": capabilities,
                        "llama_cpp": {
                            "server": server.url,
                            "mode": mode,
                            "status": status_value,
                            "n_ctx": context,
                        },
                    }
                )
                for key in LLAMA_LIMIT_FIELDS:
                    metadata[key] = limits[key]["value"]
                public_name = llama.prefix + model_id
                if (
                    public_name in explicit_names
                    or public_name in routes
                    or public_name in server_routes
                ):
                    continue
                route = Route(
                    name=public_name,
                    provider_name=provider_name,
                    upstream_model=model_id,
                    dialect=DIALECT_OPENAI_CHAT,
                    metadata=metadata,
                    _capabilities=router_metadata.llama(
                        row, model_props[model_id] if per_model_props else props,
                        mode, limits, time.time(), per_model_props=per_model_props,
                    ),
                )
                server_routes[public_name] = route
                routes[public_name] = route
                public.append(route)
                server_public.append(route)
                if llama.expose_aliases and aliases:
                    alias_candidates.append((aliases, route))

            for aliases, route in alias_candidates:
                for alias in aliases:
                    public_alias = llama.prefix + alias
                    if (
                        public_alias in explicit_names
                        or public_alias in routes
                        or public_alias in server_routes
                    ):
                        continue
                    server_routes[public_alias] = route
                    routes[public_alias] = route

            by_server[server.url] = (
                provider,
                server_routes,
                tuple(server_public),
            )
            self._log_llama_probe(server, None, len(server_public))
            states.append(
                {
                    "url": server.url,
                    "reachable": True,
                    "mode": mode,
                    "provider": provider_name,
                    "models": len(server_public),
                    "stale": False,
                    "error": None,
                    "last_success": self._clock(),
                    "last_attempt": attempted,
                }
            )

        self._publish_registry(llama=LlamaSnapshot(
            providers=_freeze(providers), routes=_freeze(routes),
            public=tuple(public), by_server=_freeze(by_server),
            servers=_freeze(states),
            refresh=self._next_refresh(previous_state.refresh, llama.refresh_seconds,
                                       attempted, failed=bool(errors)),
            error=self.sanitize("; ".join(errors)) if errors else None,
        ))
        LOG.debug(
            "llama.cpp discovery: %d server(s) reachable, %d model route(s)",
            reachable,
            len(public),
        )

    def llama_cpp_status(self, snapshot: RegistrySnapshot | None = None) -> dict[str, Any]:
        snapshot = snapshot if snapshot is not None else self._registry
        state = snapshot.llama
        llama = self.config.llama_cpp
        now = self._clock()
        interval = llama.refresh_seconds if llama else 0

        def server_status(server: Mapping[str, Any]) -> dict[str, Any]:
            freshness = self._refresh_status(
                RefreshState(last_success=server["last_success"]), interval, server["error"], now,
            )
            return {**server, "url": self.sanitize(server["url"]),
                    "age_seconds": freshness["age_seconds"],
                    "stale_age_seconds": freshness["stale_age_seconds"]}

        return {
            "enabled": llama is not None,
            "servers": [server_status(server) for server in state.servers],
            "models": len(state.public),
            "loaded": bool(state.by_server) or state.refresh.last_success is not None,
            "error": state.error,
            "refresh": self._refresh_status(state.refresh, interval, state.error, now),
        }

    def _route_entry(self, route: Route, now: int, snapshot: RegistrySnapshot) -> dict[str, Any]:
        provider = route._provider
        assert provider is not None, "public routes must be bound to their provider"
        if route.provider_name in snapshot.llama.providers:
            server = next((s for s in snapshot.llama.servers if s["provider"] == route.provider_name), {})
            interval = self.config.llama_cpp.refresh_seconds if self.config.llama_cpp else 0
            freshness = self._refresh_status(
                RefreshState(last_success=server.get("last_success")), interval,
                server.get("error"), self._clock(),
            )
        elif route.name in snapshot.catalog.routes and route.name not in self.config.routes:
            interval = self.config.catalog.refresh_seconds if self.config.catalog else 0
            freshness = self._refresh_status(snapshot.catalog.refresh, interval,
                                             snapshot.catalog.error, self._clock())
        else:
            freshness = {"stale": None, "age_seconds": None}
        # Upstream metadata.router is never the router's own capability
        # contract. Only reviewed reports and trusted sparse overrides feed it.
        contract = router_metadata.render(
            _thaw(route._capabilities) if route._capabilities else router_metadata.empty(route.dialect),
            _thaw(self.config.capability_overrides.get(route.name, {})), freshness, now,
        )
        entry: dict[str, Any] = {
            "id": route.name,
            "object": "model",
            "type": "model",
            "created": now,
            "created_at": now,
            "owned_by": route.provider_name,
            "display_name": _thaw(route.metadata.get("display_name", route.name)),
            "metadata": {
                **_thaw(route.metadata),
                "provider": route.provider_name,
                "upstream_model": route.upstream_model,
                "dialect": route.dialect,
                "provider_dialect": provider.dialect,
                "router": contract,
            },
        }
        for key in ("context_window", "max_tokens"):
            if key in route.metadata:
                entry[key] = _thaw(route.metadata[key])
        return entry

    def models_payload(self) -> dict[str, Any]:
        return self._models_payload(self._registry)

    def _models_payload(self, snapshot: RegistrySnapshot) -> dict[str, Any]:
        now = int(time.time())
        data = [self._route_entry(route, now, snapshot) for route in snapshot.public]
        return {
            "object": "list",
            "data": data,
            "has_more": False,
            "first_id": data[0]["id"] if data else None,
            "last_id": data[-1]["id"] if data else None,
        }

    def health_payload(self) -> dict[str, Any]:
        # Liveness and inventory/status reads never perform or schedule I/O.
        snapshot = self._registry
        return {
            "status": "ok",
            "providers": len(snapshot.providers),
            "models": len(snapshot.public),
            "catalog": self.catalog_status(snapshot),
            "llama_cpp": self.llama_cpp_status(snapshot),
        }

    # -- routing ----------------------------------------------------------

    def resolve(self, model: str) -> Route | None:
        snapshot = self._registry
        route = snapshot.routes.get(model)
        if route is not None:
            return route if route.enabled else None
        if model in snapshot.catalog.unmapped:
            raise RouterError(
                404,
                f"Model {model!r} is listed by the catalog but its endpoint compatibility "
                "is unknown. Configure catalog.model_providers or an explicit "
                "[[models]] route after reviewing the provider endpoint.",
                code="model_endpoint_unknown",
            )
        # Dynamic "provider/model" routing is a convenience for people who do
        # not want to list every model explicitly.
        if "/" in model:
            provider_name, _, upstream = model.partition("/")
            provider = snapshot.providers.get(provider_name)
            if provider is not None and upstream.strip():
                return Route(
                    name=model,
                    provider_name=provider_name,
                    upstream_model=upstream.strip(),
                    dialect=provider.dialect,
                    _provider=provider,
                )
        return None

    def dispatch(
        self,
        ingress: str,
        route: Route,
        payload: dict[str, Any],
        inbound_headers: Mapping[str, str],
    ) -> Plan:
        provider = route._provider or self.provider(route.provider_name)
        session_id = _session_id(provider, payload, inbound_headers)
        if ingress == route.dialect:
            return self._passthrough(
                ingress, provider, route, payload, inbound_headers, session_id
            )
        contract = self._route_entry(route if route._provider else replace(route, _provider=provider),
                                     int(time.time()), self._registry)["metadata"]["router"]
        params, losses = _prepare_translation(provider, route, ingress, payload, contract)
        provider = _translation_header_provider(provider, inbound_headers)
        if provider.dialect == DIALECT_ANTHROPIC:
            # Its adapter rebuilds version/beta headers after kwargs merging.
            # Reapply the selected opt-ins at the actual HTTP boundary using
            # our server-owned shared client, without touching global state.
            kwargs = _thaw(provider.litellm_kwargs)
            kwargs["client"] = _AnthropicHeaderTransport(self.http, provider)
            provider = replace(provider, litellm_kwargs=kwargs)
        try:
            if ingress == DIALECT_OPENAI_CHAT:
                plan = self._from_openai_chat(provider, route, params, session_id)
            elif ingress == DIALECT_OPENAI_RESPONSES:
                plan = self._from_openai_responses(provider, route, params, session_id)
            else:
                plan = self._from_anthropic(provider, route, params, session_id)
        except litellm.UnsupportedParamsError as exc:
            raise RouterError(400, "The target adapter cannot translate these generation controls. "
                              "Use a native route or review the provider compatibility policy.",
                              code="translation_loss") from exc
        if losses:
            plan.headers["x-model-router-loss"] = json.dumps(losses, separators=(",", ":"))
        return plan

    # -- pass-through -----------------------------------------------------

    def _passthrough(
        self,
        ingress: str,
        provider: Provider,
        route: Route,
        payload: dict[str, Any],
        inbound_headers: Mapping[str, str],
        session_id: str | None = None,
    ) -> Plan:
        url = _upstream_url(provider, ingress)
        body = dict(payload)
        body["model"] = route.upstream_model

        request = self.http.build_request(
            "POST",
            url,
            json=body,
            headers=_upstream_request_headers(
                provider, ingress, inbound_headers, session_id
            ),
            timeout=httpx.Timeout(provider.timeout, connect=min(20.0, provider.timeout)),
        )
        response = self.http.send(request, stream=True)

        headers = _filtered_response_headers(response.headers)
        status = response.status_code
        content_type = response.headers.get("content-type", "application/json")

        if status >= 400 or not payload.get("stream"):
            try:
                data = response.read()
            finally:
                response.close()
            if status >= 400:
                # Error bodies are untrusted diagnostics, not successful vendor
                # output. Deliberately trade byte fidelity for redaction.
                exc = RouterError(status, data.decode("utf-8", errors="replace")
                                  or f"Upstream returned HTTP {status}")
                return Plan(
                    status=status,
                    headers={name: self.sanitize(value) for name, value in headers.items()
                             if name.lower() == "retry-after"
                             or "ratelimit" in name.lower()},
                    body=_json_bytes(_error_payload(ingress, exc, self.sanitize)),
                )
            return Plan(
                status=status,
                content_type=content_type,
                headers=headers,
                body=data,
            )

        def chunks() -> Iterator[bytes]:
            try:
                yield from response.iter_bytes()
            finally:
                response.close()

        return Plan(
            status=status,
            content_type=content_type,
            headers=headers,
            chunks=chunks(),
            closer=response.close,
        )

    # -- OpenAI Chat Completions ingress ---------------------------------

    def _from_openai_chat(
        self,
        provider: Provider,
        route: Route,
        payload: dict[str, Any],
        session_id: str | None = None,
    ) -> Plan:
        params = _clean_request_params(payload, DIALECT_OPENAI_CHAT)
        stream = bool(params.get("stream"))

        if route.dialect == DIALECT_OPENAI_RESPONSES:
            # LiteLLM's "responses/" model prefix bridges Chat Completions
            # ingress calls through an upstream Responses endpoint and converts
            # the result back to Chat Completions.
            model = (
                f"{provider.litellm_provider}/responses/{route.upstream_model}"
            )
        else:
            model = f"{provider.litellm_provider}/{route.upstream_model}"

        result = litellm.completion(**_translation_call_kwargs(
            provider, params, model=model, session_id=session_id
        ))

        if stream:
            return Plan(
                content_type="text/event-stream",
                chunks=_chat_sse(result, self.sanitize),
                closer=getattr(result, "close", None),
            )
        return Plan(body=_json_bytes(_sanitize_generated_errors(_jsonable(result), self.sanitize)))

    # -- OpenAI Responses ingress ----------------------------------------

    def _from_openai_responses(
        self,
        provider: Provider,
        route: Route,
        payload: dict[str, Any],
        session_id: str | None = None,
    ) -> Plan:
        params = _clean_request_params(payload, DIALECT_OPENAI_RESPONSES)
        stream = bool(params.get("stream"))

        model = f"{provider.litellm_provider}/{route.upstream_model}"
        call_kwargs = _translation_call_kwargs(
            provider, params, model=model, session_id=session_id
        )
        if route.dialect == DIALECT_OPENAI_CHAT:
            call_kwargs["use_chat_completions_api"] = True
        result = litellm.responses(**call_kwargs)

        if stream:
            return Plan(
                content_type="text/event-stream",
                chunks=_responses_sse(result, self.sanitize),
                closer=getattr(result, "close", None),
            )
        return Plan(body=_json_bytes(_sanitize_generated_errors(_jsonable(result), self.sanitize)))

    # -- Anthropic Messages ingress --------------------------------------

    def _from_anthropic(
        self,
        provider: Provider,
        route: Route,
        payload: dict[str, Any],
        session_id: str | None = None,
    ) -> Plan:
        payload = _clean_request_params(payload, DIALECT_ANTHROPIC)
        if payload.get("max_tokens") is None:
            raise RouterError(400, "'max_tokens' is required", code="max_tokens")
        if route.dialect in (DIALECT_OPENAI_CHAT, DIALECT_GEMINI):
            return self._anthropic_to_chat(provider, route, payload, session_id)
        if route.dialect == DIALECT_OPENAI_RESPONSES:
            return self._anthropic_to_responses(provider, route, payload, session_id)
        raise RouterError(
            500, f"No Anthropic translation for dialect {route.dialect!r}"
        )

    def _anthropic_to_chat(
        self,
        provider: Provider,
        route: Route,
        payload: dict[str, Any],
        session_id: str | None = None,
    ) -> Plan:
        request = dict(payload)
        extensions = request.pop("extra_body", {})
        request.pop("stream", None)
        request.pop("model", None)
        request["model"] = route.upstream_model
        request.setdefault("messages", [])

        completion_params, tool_name_mapping = (
            _ANTHROPIC_ADAPTER.translate_completion_input_params_with_tool_mapping(
                request, custom_llm_provider=provider.litellm_provider
            )
        )
        if completion_params is None:
            raise RouterError(400, "Could not translate the Anthropic request")

        completion_params = {
            key: value for key, value in completion_params.items() if value is not None
        }
        # The adapter copies untranslated fields into its result. Revalidate its
        # output as well so a new adapter merge cannot expand this boundary.
        adapter_extensions = {
            key: completion_params.pop(key)
            for key in ("top_k", "thinking") if key in completion_params
        }
        completion_params = _clean_request_params(
            completion_params, DIALECT_OPENAI_CHAT
        )
        completion_params["extra_body"] = {
            **extensions, **_validated_generation_extensions(adapter_extensions)
        }

        stream = bool(payload.get("stream"))
        if stream:
            completion_params["stream"] = True
            completion_params.setdefault("stream_options", {"include_usage": True})

        result = litellm.completion(**_translation_call_kwargs(
            provider, completion_params,
            model=f"{provider.litellm_provider}/{route.upstream_model}",
            session_id=session_id,
        ))

        if stream:
            # Resolve lazy HTTP failures before the adapter synthesizes its
            # message_start. Its sync iterator can swallow later exceptions;
            # retain those so the router can emit a sanitized terminal error.
            try:
                first = next(result)
            except StopIteration:
                source = iter(())
            except Exception:
                _close_quietly(getattr(result, "close", None))
                raise
            else:
                source = chain((first,), result)
            failures: list[Exception] = []

            def observed() -> Iterator[Any]:
                try:
                    yield from source
                except Exception as exc:
                    failures.append(exc)
                    raise

            transformed = _ANTHROPIC_ADAPTER.translate_completion_output_params_streaming(
                observed(),
                model=route.upstream_model,
                tool_name_mapping=tool_name_mapping,
                is_async=False,
            )
            if transformed is None:
                raise RouterError(502, "Failed to transform the streaming response")

            def chunks() -> Iterator[bytes]:
                for chunk in transformed:
                    yield _sanitize_adapter_sse(chunk, self.sanitize)
                if failures:
                    raise failures[0]

            return Plan(
                content_type="text/event-stream",
                chunks=chunks(),
                closer=getattr(result, "close", None),
            )

        translated = _ANTHROPIC_ADAPTER.translate_completion_output_params(
            result, tool_name_mapping=tool_name_mapping
        )
        if translated is None:
            raise RouterError(502, "Failed to transform the upstream response")
        return Plan(body=_json_bytes(_sanitize_generated_errors(translated, self.sanitize)))

    def _anthropic_to_responses(
        self,
        provider: Provider,
        route: Route,
        payload: dict[str, Any],
        session_id: str | None = None,
    ) -> Plan:
        handler_kwargs = _translation_call_kwargs(
            provider, payload,
            model=f"{provider.litellm_provider}/{route.upstream_model}",
            session_id=session_id,
        )

        result = LiteLLMMessagesToResponsesAPIHandler.anthropic_messages_handler(
            **handler_kwargs
        )

        if payload.get("stream"):
            async def chunks() -> AsyncIterator[bytes]:
                async for chunk in result:
                    yield _sanitize_adapter_sse(chunk, self.sanitize)

            return Plan(content_type="text/event-stream", async_chunks=chunks())
        return Plan(body=_json_bytes(_sanitize_generated_errors(_jsonable(result), self.sanitize)))

    def close(self) -> None:
        """Join discovery (including manual refreshes), then release transports."""
        with self._close_lock:
            if self._closed.is_set():
                return
            self._closed.set()
            self.stop()
            with self._catalog_lock, self._llama_lock:
                try:
                    self.http.close()
                finally:
                    for logger in self._diagnostic_loggers:
                        logger.removeFilter(self._log_filter)


# ---------------------------------------------------------------------------
# URL/header helpers
# ---------------------------------------------------------------------------


def _upstream_url(provider: Provider, ingress: str) -> str:
    base = provider.base_url.rstrip("/")
    if ingress == DIALECT_ANTHROPIC:
        if base.endswith("/v1/messages"):
            return base
        if base.endswith("/v1"):
            return base + "/messages"
        return base + "/v1/messages"
    if ingress == DIALECT_OPENAI_CHAT:
        if base.endswith("/chat/completions"):
            return base
        return base + "/chat/completions"
    if ingress == DIALECT_OPENAI_RESPONSES:
        if base.endswith("/responses"):
            return base
        return base + "/responses"
    if ingress == DIALECT_GEMINI:
        if base.endswith("/v1beta"):
            return base
        return base + "/v1beta"
    raise RouterError(500, f"Unknown dialect {ingress!r}")


def _set_header(headers: dict[str, str], name: str, value: str) -> None:
    lower = name.lower()
    for existing in [key for key in headers if key.lower() == lower]:
        del headers[existing]
    headers[name] = value


_TRANSLATION_HEADERS = {
    DIALECT_ANTHROPIC: {"anthropic-beta", "anthropic-version"},
    DIALECT_OPENAI_CHAT: {"openai-beta"},
    DIALECT_OPENAI_RESPONSES: {"openai-beta"},
    DIALECT_GEMINI: set(),
}


class _AnthropicHeaderTransport(HTTPHandler):
    """Request-scoped header selection; App owns the shared HTTP connection pool."""

    def __init__(self, client: httpx.Client, provider: Provider) -> None:
        super().__init__(client=client)
        self.selected = {key.lower(): val for key, val in provider.headers.items()
                         if key.lower() in _TRANSLATION_HEADERS[DIALECT_ANTHROPIC]}
        self.selected.update(provider.auth_headers())

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        headers = dict(kwargs.pop("headers", {}) or {})
        for key, val in self.selected.items():
            _set_header(headers, key, val)
        return super().post(url, headers=headers, **kwargs)


def _translation_header_provider(provider: Provider, inbound: Mapping[str, str]) -> Provider:
    """Only destination-dialect opt-ins are client-controlled on a bridge.

    Defaults < inbound < trusted kwargs.headers < kwargs.extra_headers <
    provider.headers < auth. No credential/transport header comes from inbound.
    """
    headers = ({"anthropic-version": DEFAULT_ANTHROPIC_VERSION}
               if provider.dialect == DIALECT_ANTHROPIC else {})
    for source in (inbound, provider.litellm_kwargs.get("headers", {}),
                   provider.litellm_kwargs.get("extra_headers", {}), provider.headers):
        for name, value in source.items():
            if source is not inbound or name.lower() in _TRANSLATION_HEADERS[provider.dialect]:
                # LiteLLM merges native defaults with ordinary Python dicts;
                # canonical names prevent differently cased defaults surviving.
                _set_header(headers, name.lower(), value)
    return replace(provider, headers=headers)


def _serialized_json_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Let the transport frame JSON and discard metadata for original bytes."""
    result = {name: value for name, value in headers.items()
              if name.lower() not in _REQUEST_REPRESENTATION_HEADERS
              and name.lower() not in _HOP_BY_HOP}
    result["content-type"] = "application/json"
    return result


def _upstream_request_headers(
    provider: Provider,
    ingress: str,
    inbound: Mapping[str, str],
    session_id: str | None = None,
) -> dict[str, str]:
    headers = {
        name: value
        for name, value in inbound.items()
        if name.lower() not in _DROP_REQUEST_HEADERS
    }
    headers["accept-encoding"] = "identity"
    headers.setdefault("user-agent", f"litellm-model-router/{__version__}")
    for name, value in provider.headers.items():
        _set_header(headers, name, value)

    for name, value in provider.auth_headers().items():
        _set_header(headers, name, value)

    if provider.session_header and session_id:
        _set_header(headers, provider.session_header, session_id)

    if ingress == DIALECT_ANTHROPIC:
        lower = {name.lower() for name in headers}
        if "anthropic-version" not in lower:
            headers["anthropic-version"] = DEFAULT_ANTHROPIC_VERSION
    # Trusted static headers also cannot override the serialized body's framing.
    return _serialized_json_headers(headers)


def _filtered_response_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        name: value
        for name, value in headers.items()
        if name.lower() not in _DROP_RESPONSE_HEADERS
    }


def _generation_error(field: str) -> RouterError:
    # Do not echo untrusted values (or arbitrary field names) into diagnostics.
    return RouterError(400, f"Unsupported or invalid {field} for translation",
                       code="invalid_generation_parameter")


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
    if (params.get("tool_choice") is not None or params.get("parallel_tool_calls") is not None
            ) and not params.get("tools"):
        for key in ("tool_choice", "parallel_tool_calls"):
            if params.get(key) is not None:
                lose(key, "bridge requires tools for this control")
    if params.get("tools") and route.dialect != DIALECT_OPENAI_RESPONSES:
        function_tools = all(
            tool.get("type", "function") == "function" for tool in params["tools"]
        )
        if not function_tools:
            lose("tools", "native tool type requires a native route")
            if "tool_choice" in params:
                lose("tool_choice", "tool definitions cannot be translated")
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
        supported = litellm.get_supported_openai_params(
            model=route.upstream_model, custom_llm_provider=provider.litellm_provider)
        aliases = {"max_output_tokens": "max_tokens", "stop_sequences": "stop",
                   "reasoning": "reasoning_effort", "output_format": "response_format",
                   "text": "response_format"}
        special = {"messages", "input", "instructions", "system", "extra_body",
                   "metadata", "output_config", "top_k"}
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
        fields = ", ".join(sorted({loss["field"] for loss in losses}))
        raise RouterError(400, f"Translation would lose or approximate: {fields}. "
                          "Use a same-protocol route, remove these controls, or ask the operator "
                          "to configure compatibility = 'lossy' (with loss diagnostics).",
                          code="translation_loss")
    return params, losses


def _translation_call_kwargs(
    provider: Provider, params: Mapping[str, Any], *, model: str,
    session_id: str | None,
) -> dict[str, Any]:
    """Merge only validated generation data, then trusted server settings."""
    # These globals can inject auth after argument merging or via a client's
    # default auth/headers. Refuse them instead of swapping shared state.
    for setting in ("proxy_auth", "client_session", "aclient_session"):
        if getattr(litellm, setting, None) is not None:
            raise RouterError(500, f"Global LiteLLM {setting} is incompatible with credential isolation")
    generation = dict(params)
    body = dict(generation.pop("extra_body", {}))
    # Thinking is a reviewed generation extension with a native LiteLLM mapping.
    # Never promote arbitrary extra_body members into keyword arguments.
    if "thinking" in body:
        generation["thinking"] = body.pop("thinking")
    trusted = provider.litellm_call_kwargs(skip_mcp=True, session_id=session_id)
    if trusted.get("additional_drop_params") or litellm.drop_params:
        raise RouterError(500, "LiteLLM parameter dropping bypasses compatibility diagnostics; "
                          "use the provider compatibility policy instead")
    body.update(trusted.pop("extra_body", {}) or {})
    result = {**generation, **trusted, "model": model}
    if body:
        result["extra_body"] = body
    # The router forwards tool definitions; it must not execute MCP locally.
    result["_skip_mcp_handler"] = True
    return result


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class _ReadDeadlineExceeded(Exception):
    pass


class _ClientWriteTimeout(OSError):
    pass


class _ResponseWriter(io.BufferedIOBase):
    def __init__(self, connection: socket.socket, timeout: float) -> None:
        super().__init__()
        self.connection = connection
        self.timeout = timeout

    def writable(self) -> bool:
        return True

    def write(self, data: bytes) -> int:
        self.connection.settimeout(self.timeout)
        try:
            self.connection.sendall(data)
        except TimeoutError as exc:
            raise _ClientWriteTimeout from exc
        return len(data)


class _DeadlineReader(io.RawIOBase):
    """Check the absolute phase deadline before *each* socket receive.

    A socket inactivity timeout alone lets a slow trickle hold a handler forever.
    BufferedReader may perform several receives inside one read()/readline().
    """

    def __init__(self, connection: socket.socket) -> None:
        super().__init__()
        self.connection = connection
        self.deadline: float | None = None

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        if self.deadline is not None:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise _ReadDeadlineExceeded
            self.connection.settimeout(remaining)
        try:
            return self.connection.recv_into(buffer)
        except TimeoutError as exc:
            raise _ReadDeadlineExceeded from exc


class RouterHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 16

    def __init__(self, address: tuple[str, int], handler: type[Handler]) -> None:
        self.app = handler.app
        self.config = handler.app.config
        self._capacity_lock = threading.Lock()
        self._connections: set[socket.socket] = set()
        self._generations: set[socket.socket] = set()
        self._closing = False
        super().__init__(address, handler)

    def process_request(self, request: socket.socket, client_address: Any) -> None:
        # Admission happens in the accept loop, before ThreadingMixIn can spawn.
        with self._capacity_lock:
            admitted = not self._closing and len(self._connections) < self.config.max_connections
            if admitted:
                self._connections.add(request)
        if not admitted:
            try:
                # Never stall the accept loop on an overloaded, non-reading peer.
                request.setblocking(False)
                request.send(b"HTTP/1.1 503 Service Unavailable\r\n"
                             b"Connection: close\r\nContent-Length: 0\r\n"
                             b"Retry-After: 1\r\n\r\n")
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._release_connection(request)
            raise

    def process_request_thread(self, request: socket.socket, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._release_connection(request)

    def _release_connection(self, request: socket.socket) -> None:
        with self._capacity_lock:
            self._generations.discard(request)
            self._connections.discard(request)

    def acquire_generation(self, request: socket.socket) -> bool:
        with self._capacity_lock:
            if self._closing or len(self._generations) >= self.config.max_generations:
                return False
            self._generations.add(request)
            return True

    def release_generation(self, request: socket.socket) -> None:
        with self._capacity_lock:
            self._generations.discard(request)

    def _stop_connections(self) -> None:
        with self._capacity_lock:
            self._closing = True
            connections = list(self._connections)
            self._connections.clear()
            self._generations.clear()
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()

    def shutdown(self) -> None:
        self._stop_connections()
        super().shutdown()
        self.app.stop()

    def server_close(self) -> None:
        self._stop_connections()
        try:
            super().server_close()
        finally:
            self.app.stop()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"litellm-model-router/{__version__}"
    app: App  # provided by serve()/main()

    def setup(self) -> None:
        super().setup()
        self.rfile.close()
        self.wfile.close()
        self._reader = _DeadlineReader(self.connection)
        self.rfile = io.BufferedReader(self._reader)
        self.wfile = _ResponseWriter(self.connection, self.app.config.write_timeout)
        self.connection.settimeout(self.app.config.write_timeout)

    def handle_one_request(self) -> None:
        self._headers_sent = False
        self.requestline = ""
        self.request_version = "HTTP/1.1"
        self.command = None
        self._reader.deadline = time.monotonic() + self.app.config.header_read_timeout
        try:
            # The stdlib catches TimeoutError without sending a response. Our
            # distinct read exception lets us return an observable HTTP 408.
            super().handle_one_request()
        except _ReadDeadlineExceeded:
            try:
                self._send_error(DIALECT_OPENAI_CHAT, RouterError(
                    408, "Request headers read deadline exceeded", code="header_read_timeout"
                ))
            except OSError:
                pass
        except OSError:
            pass
        finally:
            self.close_connection = True
            self._reader.deadline = None

    def parse_request(self) -> bool:
        parsed = super().parse_request()
        self._reader.deadline = None
        self.connection.settimeout(self.app.config.write_timeout)
        return parsed

    # -- logging ---------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:
        if getattr(self, "_verbose", False):
            LOG.debug("%s - %s", self.address_string(), fmt % args)

    # -- entry points ----------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        path = urlsplit(self.path).path.rstrip("/") or "/"
        try:
            if path in ("/health", "/healthz"):
                return self._send_bytes(
                    200,
                    "application/json",
                    {},
                    _json_bytes(self.app.health_payload()),
                )
            if path in ("/v1/models", "/models"):
                if not self.app.authorized(self.headers):
                    return self._send_error(
                        DIALECT_OPENAI_CHAT,
                        RouterError(401, "Invalid API key"),
                    )
                return self._send_bytes(
                    200, "application/json", {}, _json_bytes(self.app.models_payload())
                )
            if path == "/":
                return self._send_bytes(
                    200,
                    "application/json",
                    {},
                    _json_bytes(
                        {
                            "name": "litellm-model-router",
                            "version": __version__,
                            "endpoints": sorted(ENDPOINTS),
                        }
                    ),
                )
            self._send_error(
                DIALECT_OPENAI_CHAT,
                RouterError(404, f"Unknown route: {path}"),
            )
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        started = time.monotonic()
        path = urlsplit(self.path).path
        ingress = ENDPOINTS.get(path)
        self._headers_sent = False
        self._verbose = self.app.verbose

        if ingress is None:
            return self._send_error(
                DIALECT_OPENAI_CHAT, RouterError(404, f"Unknown route: {path}")
            )

        if not self.app.authorized(self.headers):
            return self._send_error(
                ingress,
                RouterError(401, "Invalid API key"),
            )

        status_for_log = 500
        model_for_log = "-"
        route_for_log = "-"
        self._streamed_request = False
        generation_acquired = False
        try:
            self._reader.deadline = time.monotonic() + self.app.config.body_read_timeout
            try:
                raw = self._read_body()
            except _ReadDeadlineExceeded as exc:
                raise RouterError(408, "Request body read deadline exceeded",
                                  code="body_read_timeout") from exc
            finally:
                self._reader.deadline = None
                self.connection.settimeout(self.app.config.write_timeout)
            if not raw:
                raise RouterError(400, "A JSON request body is required")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RouterError(400, f"Invalid JSON body: {exc}") from exc
            if not isinstance(payload, dict):
                raise RouterError(400, "Request body must be a JSON object")

            model = payload.get("model")
            if not isinstance(model, str) or not model.strip():
                raise RouterError(
                    400, "A model ID is required", code="model_required"
                )
            model_for_log = model
            self._streamed_request = bool(payload.get("stream"))

            self._validate(ingress, payload)
            # No waiting queue and no discovery lock is held during admission.
            # Include route resolution so discovery cannot accumulate an
            # unbounded set of generation callers either.
            generation_acquired = self.server.acquire_generation(self.connection)
            if not generation_acquired:
                raise RouterError(503, "Generation capacity exhausted",
                                  code="generation_overloaded")
            route = self.app.resolve(model)
            if route is None:
                raise RouterError(
                    404,
                    f"Model {model!r} is not configured",
                    code="model_not_found",
                )
            route_for_log = f"{route.provider_name}/{route.upstream_model}"

            plan = self.app.dispatch(ingress, route, payload, self.headers)
            status_for_log = plan.status
            self._send_plan(plan, ingress)
        except (BrokenPipeError, ConnectionResetError, _ClientWriteTimeout):
            pass
        except RouterError as exc:
            status_for_log = exc.status
            self._send_error(ingress, exc)
        except Exception as exc:  # noqa: BLE001 - a router must not crash
            status_for_log = _status_for(exc)
            if LOG.isEnabledFor(logging.DEBUG):
                LOG.debug("Request failed: %s", _error_message(exc, self.app.sanitize))
            self._send_error(ingress, exc)
        finally:
            if generation_acquired:
                self.server.release_generation(self.connection)
            elapsed_ms = (time.monotonic() - started) * 1000
            LOG.info(
                "%s %s model=%s route=%s stream=%s status=%d %.0fms",
                self.command,
                path,
                model_for_log,
                route_for_log,
                "yes" if self._streamed_request else "no",
                status_for_log,
                elapsed_ms,
            )

    # -- request parsing --------------------------------------------------

    def _validate(self, ingress: str, payload: Mapping[str, Any]) -> None:
        if ingress == DIALECT_OPENAI_CHAT:
            if not isinstance(payload.get("messages"), list):
                raise RouterError(400, "'messages' must be an array")
        elif ingress == DIALECT_ANTHROPIC:
            if not isinstance(payload.get("messages"), list):
                raise RouterError(400, "'messages' must be an array")
        # Responses accepts input, instructions, prompt or previous_response_id;
        # LiteLLM validates the combination.

    def _read_body(self) -> bytes:
        max_bytes = self.app.config.max_body_bytes
        transfers = self.headers.get_all("Transfer-Encoding", [])
        lengths = self.headers.get_all("Content-Length", [])
        encodings = self.headers.get_all("Content-Encoding", [])
        if transfers and lengths:
            raise RouterError(400, "Transfer-Encoding conflicts with Content-Length")
        if len(transfers) > 1 or (transfers and transfers[0].strip(" \t").lower() != "chunked"):
            raise RouterError(400, "Only a single chunked Transfer-Encoding is supported")
        if len(lengths) > 1:
            raise RouterError(400, "Multiple Content-Length headers are not supported")
        encoding = encodings[0].strip(" \t").lower() if encodings else ""
        if len(encodings) > 1 or encoding not in ("", "identity", "gzip"):
            raise RouterError(415, "Unsupported Content-Encoding")

        if transfers:
            raw = self._read_chunked(max_bytes)
        else:
            length_text = lengths[0].strip(" \t") if lengths else "0"
            if not re.fullmatch(r"[0-9]+", length_text):
                raise RouterError(400, "Invalid Content-Length")
            # Compare before int() so arbitrarily long decimal values stay bounded.
            significant = length_text.lstrip("0") or "0"
            maximum = str(max_bytes)
            if (len(significant) > len(maximum)
                    or (len(significant) == len(maximum) and significant > maximum)):
                raise RouterError(413, "Request body is too large")
            raw = self._read_exact(int(significant))

        if encoding == "gzip":
            raw = _gunzip_limited(raw, max_bytes)

        if len(raw) > max_bytes:
            raise RouterError(413, "Request body is too large")
        return raw

    def _read_exact(self, length: int) -> bytes:
        if length < 0:
            raise RouterError(400, "Invalid body read length")
        data = bytearray()
        while len(data) < length:
            part = self.rfile.read(length - len(data))
            if not part:
                raise RouterError(400, "Truncated request body")
            data.extend(part)
        return bytes(data)

    def _read_chunked(self, max_bytes: int) -> bytes:
        data = bytearray()
        metadata = 0

        def read_line() -> bytes:
            nonlocal metadata
            if metadata >= MAX_CHUNK_METADATA:
                raise RouterError(413, "Chunk metadata is too large")
            line = self.rfile.readline(min(MAX_CHUNK_LINE, MAX_CHUNK_METADATA - metadata) + 1)
            metadata += len(line)
            if len(line) > MAX_CHUNK_LINE or metadata > MAX_CHUNK_METADATA:
                raise RouterError(413, "Chunk metadata is too large")
            if not line.endswith(b"\r\n"):
                raise RouterError(400, "Invalid or truncated chunk framing")
            return line

        while True:
            match = _CHUNK_LINE.fullmatch(read_line())
            if match is None:
                raise RouterError(400, "Invalid chunk size or extension")
            size = int(match[1], 16)
            if size == 0:
                # Validate and discard trailers; they never affect request headers.
                for count in range(MAX_CHUNK_TRAILERS + 1):
                    trailer = read_line()
                    if trailer == b"\r\n":
                        break
                    if count == MAX_CHUNK_TRAILERS:
                        raise RouterError(413, "Too many chunk trailers")
                    if _TRAILER_LINE.fullmatch(trailer) is None:
                        raise RouterError(400, "Invalid chunk trailer")
                    name = trailer.split(b":", 1)[0].decode("ascii").lower()
                    if name in _DROP_REQUEST_HEADERS:
                        raise RouterError(400, "Forbidden chunk trailer")
                break
            if size > max_bytes - len(data):
                raise RouterError(413, "Request body is too large")
            data.extend(self._read_exact(size))
            if self._read_exact(2) != b"\r\n":
                raise RouterError(400, "Invalid chunk delimiter")
            metadata += 2
        return bytes(data)

    # -- response sending -------------------------------------------------

    def _send_plan(self, plan: Plan, ingress: str) -> None:
        if plan.body is not None:
            self._send_bytes(
                plan.status, plan.content_type, plan.headers, plan.body
            )
            return
        if plan.async_chunks is not None:
            asyncio.run(self._relay_async(plan, ingress))
            return
        self._relay_sync(plan, ingress)

    def _relay_sync(self, plan: Plan, ingress: str) -> None:
        iterator = plan.chunks
        if iterator is None:
            self._send_bytes(plan.status, plan.content_type, plan.headers, b"")
            return
        try:
            try:
                first = next(iterator)
            except StopIteration:
                self._send_stream_headers(plan)
                return
            except (BrokenPipeError, ConnectionResetError, _ClientWriteTimeout):
                return
            except Exception as exc:  # noqa: BLE001
                self._send_error(ingress, exc)
                return

            self._send_stream_headers(plan)
            try:
                self.wfile.write(first)
                self.wfile.flush()
                for chunk in iterator:
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, _ClientWriteTimeout):
                pass
            except Exception as exc:  # noqa: BLE001
                self._write_stream_error(ingress, exc)
        finally:
            _close_quietly(getattr(iterator, "close", None))
            _close_quietly(plan.closer)

    async def _relay_async(self, plan: Plan, ingress: str) -> None:
        iterator = plan.async_chunks
        if iterator is None:
            self._send_bytes(plan.status, plan.content_type, plan.headers, b"")
            return
        try:
            try:
                first = await anext(iterator)
            except StopAsyncIteration:
                self._send_stream_headers(plan)
                return
            except (BrokenPipeError, ConnectionResetError, _ClientWriteTimeout):
                return
            except Exception as exc:  # noqa: BLE001
                self._send_error(ingress, exc)
                return

            self._send_stream_headers(plan)
            try:
                self.wfile.write(first)
                self.wfile.flush()
                async for chunk in iterator:
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, _ClientWriteTimeout):
                pass
            except Exception as exc:  # noqa: BLE001
                self._write_stream_error(ingress, exc)
        finally:
            aclose = getattr(iterator, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:  # noqa: BLE001
                    pass
            _close_quietly(plan.closer)

    def _send_stream_headers(self, plan: Plan) -> None:
        self.send_response(plan.status)
        self.send_header(
            "Content-Type", plan.content_type or "text/event-stream"
        )
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self._send_extra_headers(plan.headers)
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self._headers_sent = True

    def _write_stream_error(self, ingress: str, exc: BaseException) -> None:
        try:
            self.wfile.write(_stream_error_bytes(ingress, exc, self.app.sanitize))
            self.wfile.flush()
            if ingress == DIALECT_OPENAI_CHAT:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, _ClientWriteTimeout):
            pass

    def _send_extra_headers(self, headers: Mapping[str, str]) -> None:
        """Send forwarded upstream headers without duplicating managed ones."""
        managed = {"content-type", "content-length", "cache-control", "connection"}
        for name, value in headers.items():
            if name.lower() not in managed:
                self.send_header(name, value)

    def _send_error(self, ingress: str, exc: BaseException) -> None:
        status = _status_for(exc)
        if getattr(self, "_headers_sent", False):
            self._write_stream_error(ingress, exc)
            return
        body = _json_bytes(_error_payload(ingress, exc, self.app.sanitize))
        extra = {}
        if status == 401:
            extra["WWW-Authenticate"] = 'Bearer realm="model-router"'
        if status == 503:
            extra["Retry-After"] = "1"
        self._send_bytes(status, "application/json", extra, body)

    def _send_bytes(
        self,
        status: int,
        content_type: str,
        headers: Mapping[str, str],
        body: bytes,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self._send_extra_headers(headers)
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self._headers_sent = True
        if body:
            self.wfile.write(body)
            self.wfile.flush()


def _close_quietly(closable: Any) -> None:
    if closable is None:
        return
    try:
        closable()
    except Exception:  # noqa: BLE001
        pass


def _gunzip_limited(raw: bytes, limit: int) -> bytes:
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        data = decompressor.decompress(raw, limit + 1)
    except zlib.error as exc:
        raise RouterError(400, "Invalid gzip request body") from exc
    if len(data) > limit or decompressor.unconsumed_tail:
        raise RouterError(413, "Decompressed request body is too large")
    if not decompressor.eof:
        raise RouterError(400, "Incomplete gzip request body")
    if decompressor.unused_data:
        raise RouterError(400, "Gzip request body must contain exactly one member and no trailing bytes")
    return data


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def serve(app: App, host: str, port: int) -> RouterHTTPServer:
    handler_class = type("BoundHandler", (Handler,), {"app": app})
    server = RouterHTTPServer((host, port), handler_class)
    try:
        app.start()
    except BaseException:
        server.server_close()
        raise
    return server


def _find_default_config() -> Path | None:
    for name in (
        "router.toml",
        "router.json",
        "config.toml",
        "config.json",
        "router.yaml",
        "config.yaml",
    ):
        path = Path(name)
        if path.exists():
            return path
    return None


def _print_summary(
    config: Config,
    host: str,
    port: int,
    *,
    listening: bool = True,
    app: App | None = None,
) -> None:
    if listening:
        LOG.info("model-router %s listening on http://%s:%d", __version__, host, port)
    else:
        LOG.info("model-router %s config OK for http://%s:%d", __version__, host, port)
    for route in config.public_routes:
        LOG.info(
            "  %-28s -> %-18s %-40s [%s]",
            route.name,
            route.provider_name,
            route.upstream_model,
            route.dialect,
        )
    if app is not None and config.catalog is not None:
        status = app.catalog_status()
        detail = f" (error: {status['error']})" if status["error"] else ""
        LOG.info(
            "catalog: %d models from %s%s",
            status["models"],
            config.catalog.url,
            detail,
        )
        if not listening:
            for name in sorted(app._registry.catalog.routes):
                LOG.info("  %s", name)
    if app is not None and config.llama_cpp is not None:
        status = app.llama_cpp_status()
        LOG.info(
            "llama.cpp: %d models from %d server(s)",
            status["models"],
            len(config.llama_cpp.servers),
        )
        for server in status["servers"]:
            LOG.info(
                "  %s: %s%s",
                server["url"],
                "reachable" if server["reachable"] else "unavailable",
                (
                    f" ({server['mode']}, {server['models']} models)"
                    if server["reachable"]
                    else f" ({server['error']})"
                ),
            )
        if not listening:
            for route in sorted(app._registry.llama.public, key=lambda item: item.name):
                LOG.info(
                    "  llama: %s (context %s)",
                    route.name,
                    route.metadata.get("context_window"),
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="LiteLLM-backed multi-dialect model router."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="TOML/JSON config file (default: ./router.toml or ./router.json)",
    )
    parser.add_argument("--host", help="Override server.host")
    parser.add_argument("--port", type=int, help="Override server.port")
    parser.add_argument(
        "--api-key",
        action="append",
        default=None,
        help="Require this router API key (repeatable). Overrides config.",
    )
    parser.add_argument(
        "--env-file",
        action="append",
        type=Path,
        default=[],
        help=(
            "Load KEY=value credentials from this file before reading the config "
            "(repeatable). Existing environment variables win."
        ),
    )
    parser.add_argument(
        "--no-default-env",
        action="store_true",
        help="Do not auto-load ~/.config/ai/credentials.env",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the config, print the routes, and exit",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    if not args.verbose:
        for noisy in ("LiteLLM", "litellm", "httpx", "httpcore"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    for env_file in args.env_file:
        load_env_file(env_file)
        LOG.info("Loaded credentials from %s", env_file)
    if not args.no_default_env:
        for env_file in load_default_credentials():
            LOG.info("Loaded credentials from %s", env_file)

    config_path = args.config or _find_default_config()
    if config_path is None:
        parser.error("No config found; pass --config /path/to/router.toml")

    config = load_config(config_path)
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.api_key:
        config.api_keys = list(dict.fromkeys(args.api_key))

    if args.check:
        app = App(config)
        try:
            _print_summary(
                config, config.host, config.port, listening=False, app=app
            )
        finally:
            app.close()
        return 0

    app = App(config)
    app.verbose = args.verbose
    httpd = None
    try:
        httpd = serve(app, config.host, config.port)
        _print_summary(config, config.host, config.port, app=app)
        httpd.serve_forever()
    except KeyboardInterrupt:
        LOG.info("Shutting down")
    finally:
        if httpd is not None:
            httpd.server_close()
        app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
