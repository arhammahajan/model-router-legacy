"""Credential, session and representation-header policy shared by transports."""
from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Mapping
from urllib.parse import quote

from .common import RouterError
from .constants import (
    _DROP_REQUEST_HEADERS,
    _DROP_RESPONSE_HEADERS,
    _HOP_BY_HOP,
    _NATIVE_SESSION_HEADERS,
    _REQUEST_REPRESENTATION_HEADERS,
    _SESSION_VALUE,
    _TRANSLATION_HEADERS,
    DEFAULT_ANTHROPIC_VERSION,
    DIALECT_ANTHROPIC,
    DIALECT_GEMINI,
    DIALECT_OPENAI_CHAT,
    DIALECT_OPENAI_RESPONSES,
    __version__,
)

if TYPE_CHECKING:
    from .config import Provider


def _connection_options(headers: Mapping[str, str]) -> set[str]:
    """Collect hop-by-hop names, including repeated Connection fields."""
    return {
        token.strip().lower()
        for name, value in headers.items() if name.lower() == "connection"
        for token in value.split(",") if token.strip()
    }


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
    """Select a client-owned identity; never infer one from prompts or credentials."""
    if not provider.session_header:
        return None
    candidates: list[str] = []

    def validate(value: Any) -> str:
        if not isinstance(value, str) or _SESSION_VALUE.fullmatch(value) is None:
            raise RouterError(400, "Conversation identity must be 1-256 ASCII characters: "
                              "letters, digits, '.', '_', ':', '/', '-' with a leading "
                              "letter or digit", code="invalid_session_id")
        return value

    # Validate even losing/disabled native headers, which passthrough otherwise
    # forwards. Duplicates (including case variants) must not depend on map order.
    names = dict.fromkeys((provider.session_header, *provider.session_source_headers,
                           *_NATIVE_SESSION_HEADERS))
    enabled = {provider.session_header, *provider.session_source_headers}
    hop_headers = _connection_options(inbound)
    for name in names:
        if name in hop_headers:
            continue
        values = [value for key, value in inbound.items() if key.lower() == name]
        if len(values) > 1:
            raise RouterError(400, "Duplicate conversation identity header",
                              code="invalid_session_id")
        if values:
            value = validate(values[0])
            if name in enabled:
                candidates.append(value)
    if provider.session_prompt_cache_key and "prompt_cache_key" in payload:
        candidates.append(validate(payload["prompt_cache_key"]))
    if candidates:
        return candidates[0]
    if provider.session_missing == "reject":
        raise RouterError(400, f"Conversation identity required: send {provider.session_header} "
                          "with the same value on every turn", code="missing_session_id")
    return None


def _session_request_headers(
    provider: Provider, headers: Mapping[str, str], session_id: str | None,
) -> dict[str, str]:
    """Consume native identity headers and emit only the selected upstream identity."""
    if not provider.session_header:
        return dict(headers)
    consumed = {provider.session_header.lower(), "x-opencode-session", *_NATIVE_SESSION_HEADERS}
    result = {name: value for name, value in headers.items() if name.lower() not in consumed}
    if session_id is not None:
        result[provider.session_header] = session_id
    return result


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
        if base.endswith(("/v1beta", "/v1")):
            return base
        return base + "/v1beta"
    raise RouterError(500, f"Unknown dialect {ingress!r}")


def _gemini_url(provider: Provider, upstream_model: str, method: str, alt: str | None) -> str:
    """The upstream generateContent URL for a resolved model.

    The model is client-influenced on dynamic ``provider/model`` routes and is
    placed in the URL path, so it must not be able to change the path structure,
    add a query or reach another resource. The SDKs reject the same characters.
    """
    name = upstream_model if upstream_model.startswith(("models/", "tunedModels/")) \
        else "models/" + upstream_model
    segments = name.split("/")
    if (re.search(r"[\x00-\x20\x7f?#%&\\]", name) or any(s in ("", ".", "..") for s in segments)
            or len(segments) < 2):
        raise RouterError(400, "The upstream model name cannot be used in a Gemini request path",
                          code="invalid_model")
    url = f"{_upstream_url(provider, DIALECT_GEMINI)}/{quote(name, safe='/:')}:{method}"
    # Only the response-format selector is forwarded; never `key` or anything else.
    return url + "?alt=sse" if alt == "sse" else url


def _count_tokens_url(provider: Provider) -> str:
    """Anthropic's token-count endpoint, accepting the same base forms as messages."""
    messages = _upstream_url(provider, DIALECT_ANTHROPIC)
    return messages + "/count_tokens"


def _set_header(headers: dict[str, str], name: str, value: str) -> None:
    lower = name.lower()
    for existing in [key for key in headers if key.lower() == lower]:
        del headers[existing]
    headers[name] = value


def _translation_header_provider(provider: Provider, inbound: Mapping[str, str]) -> Provider:
    """Only destination-dialect opt-ins are client-controlled on a bridge.

    Defaults < inbound < trusted kwargs.headers < kwargs.extra_headers <
    provider.headers < auth. No credential/transport header comes from inbound.
    """
    headers = ({"anthropic-version": DEFAULT_ANTHROPIC_VERSION}
               if provider.dialect == DIALECT_ANTHROPIC else {})
    hop_headers = _connection_options(inbound)
    for source in (inbound, provider.litellm_kwargs.get("headers", {}),
                   provider.litellm_kwargs.get("extra_headers", {}), provider.headers):
        for name, value in source.items():
            if source is not inbound or (
                name.lower() in _TRANSLATION_HEADERS[provider.dialect]
                and name.lower() not in hop_headers
            ):
                # LiteLLM merges native defaults with ordinary Python dicts;
                # canonical names prevent differently cased defaults surviving.
                _set_header(headers, name.lower(), value)
    return replace(provider, headers=headers)


def _serialized_json_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Let the transport frame JSON and discard metadata for original bytes."""
    blocked = _REQUEST_REPRESENTATION_HEADERS | _HOP_BY_HOP | _connection_options(headers)
    result = {name: value for name, value in headers.items()
              if name.lower() not in blocked}
    result["content-type"] = "application/json"
    return result


def _upstream_request_headers(
    provider: Provider,
    ingress: str,
    inbound: Mapping[str, str],
    session_id: str | None = None,
) -> dict[str, str]:
    blocked = _DROP_REQUEST_HEADERS | _connection_options(inbound)
    headers = {
        name: value
        for name, value in inbound.items()
        if name.lower() not in blocked
    }
    headers["accept-encoding"] = "identity"
    headers.setdefault("user-agent", f"litellm-model-router/{__version__}")
    for name, value in provider.headers.items():
        _set_header(headers, name, value)

    for name, value in provider.auth_headers().items():
        _set_header(headers, name, value)

    headers = _session_request_headers(provider, headers, session_id)

    if ingress == DIALECT_ANTHROPIC:
        lower = {name.lower() for name in headers}
        if "anthropic-version" not in lower:
            headers["anthropic-version"] = DEFAULT_ANTHROPIC_VERSION
    # Trusted static headers also cannot override the serialized body's framing.
    return _serialized_json_headers(headers)


def _filtered_response_headers(headers: Mapping[str, str]) -> dict[str, str]:
    blocked = _DROP_RESPONSE_HEADERS | _connection_options(headers)
    return {
        name: value
        for name, value in headers.items()
        if name.lower() not in blocked
    }
