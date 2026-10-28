"""POST /v1/messages/count_tokens: exact upstream counting or an explicit refusal.

Anthropic's endpoint takes the Messages request shape and returns
``{"input_tokens": n}`` without generating anything (it is free, and billed
separately from messages). The router forwards to a provider that is known or
declared to implement it and refuses everywhere else. It never estimates, never
translates a request into another dialect to "count" it, and never calls a
generation endpoint.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

import httpx

from .common import RouterError, _error_payload, _json_bytes
from .config import Provider, Route
from .constants import DIALECT_ANTHROPIC, MAX_COUNT_RESPONSE_BYTES
from .headers import _count_tokens_url, _filtered_response_headers, _upstream_request_headers
from .streams import Plan


def unsupported(route: Route, provider: Provider) -> RouterError:
    """The route cannot be counted exactly; say why and what would change that."""
    if route.dialect != DIALECT_ANTHROPIC:
        reason = (f"Model {route.name!r} is served through the {route.dialect!r} protocol, "
                  "which has no Anthropic token-count endpoint, and the router does not "
                  "estimate token counts.")
    elif provider.count_tokens is False:
        reason = f"Token counting is disabled for provider {provider.name!r}."
    else:
        reason = (f"Provider {provider.name!r} is not known to implement Anthropic's "
                  "token-count endpoint. Set count_tokens = true on the provider after "
                  "confirming it does.")
    return RouterError(400, "Exact token counting is unavailable for this model. " + reason,
                       code="count_tokens_unsupported", error_type="invalid_request_error")


def count_tokens(
    route: Route, provider: Provider, payload: Mapping[str, Any],
    inbound_headers: Mapping[str, str], client: httpx.Client, sanitize: Any,
) -> Plan:
    if not provider.count_tokens_enabled or route.dialect != DIALECT_ANTHROPIC:
        raise unsupported(route, provider)

    # Everything but the model is forwarded untouched so the upstream applies its
    # own contract to tools, thinking, caching and multimodal blocks.
    body = dict(payload)
    body["model"] = route.upstream_model
    request = client.build_request(
        "POST", _count_tokens_url(provider), json=body,
        # Conversation identity is a generation affinity concern, not a count one.
        headers=_upstream_request_headers(provider, DIALECT_ANTHROPIC, inbound_headers, None),
        timeout=httpx.Timeout(provider.timeout, connect=min(20.0, provider.timeout)),
    )
    response = client.send(request, stream=True)
    try:
        data = _read_bounded(response)
    finally:
        response.close()

    status = response.status_code
    headers = _filtered_response_headers(response.headers)
    if status >= 400:
        # Error bodies are untrusted diagnostics: redact rather than preserve bytes.
        exc = RouterError(status, data.decode("utf-8", errors="replace")
                          or f"Upstream returned HTTP {status}")
        return Plan(
            status=status,
            headers={name: sanitize(value) for name, value in headers.items()
                     if name.lower() == "retry-after" or "ratelimit" in name.lower()},
            body=_json_bytes(_error_payload(DIALECT_ANTHROPIC, exc, sanitize)),
        )
    # A 2xx that is not a count must never reach the caller as one.
    tokens = _input_tokens(data) if status == 200 else None
    if tokens is None:
        raise RouterError(502, "The upstream did not return a valid token count",
                          code="invalid_count_response")
    return Plan(status=status, content_type="application/json", headers=headers, body=data)


def _read_bounded(response: httpx.Response) -> bytes:
    data = bytearray()
    for chunk in response.iter_bytes():
        data.extend(chunk)
        if len(data) > MAX_COUNT_RESPONSE_BYTES:
            raise RouterError(502, "The upstream token-count response is too large",
                              code="count_response_too_large")
    return bytes(data)


def _input_tokens(data: bytes) -> int | None:
    try:
        value = json.loads(data)
    except (ValueError, RecursionError):
        return None
    tokens = value.get("input_tokens") if isinstance(value, dict) else None
    # bool is an int subclass; True is not a count.
    return tokens if type(tokens) is int and tokens >= 0 else None
