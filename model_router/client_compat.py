"""Opt-in session compatibility for bearer-only and Gemini clients."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from .common import RouterError

_SESSION_TOKEN = re.compile(r"router-session-[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
_SESSION = re.compile(r"Bearer (" + _SESSION_TOKEN.pattern + r")")


def prepare_gemini(
    payload: dict[str, Any], headers: Mapping[str, str], query_keys: Sequence[str] = (),
) -> tuple[dict[str, Any], str | None]:
    """Consume Antigravity conversation metadata and explicit public API-key tokens.

    Called only with bearer_session_compat enabled. The ACP gateway session ID wins
    over a per-process token; native session headers take priority in _session_id.
    Headers stay intact so duplicate identity headers still reach validation.
    Credential headers and query keys are already excluded from upstream requests.
    """
    tokens = set()
    for name, value in headers.items():
        if name.lower() == "authorization" and value.startswith("Bearer "):
            value = value[7:]
        elif name.lower() not in {"x-goog-api-key", "x-api-key"}:
            continue
        if _SESSION_TOKEN.fullmatch(value):
            tokens.add(value)
    tokens.update(value for value in query_keys if _SESSION_TOKEN.fullmatch(value))
    if len(tokens) > 1:
        raise RouterError(400, "Conflicting public session tokens", code="invalid_session_id")

    session = payload.get("sessionId")
    if session is not None and not isinstance(session, str):
        raise RouterError(400, "'sessionId' must be a string or null", code="invalid_session_id")
    if "sessionId" in payload:
        payload = {key: value for key, value in payload.items() if key != "sessionId"}
    if session:
        # Preserve the identity used by the former Antigravity ACP proxy.
        return payload, "agy-acp-" + hashlib.sha256(
            json.dumps(session, sort_keys=True).encode()
        ).hexdigest()
    return payload, next(iter(tokens), None)


def prepare_chat(payload: dict[str, Any], headers: Mapping[str, str]):
    """Consume an explicit public session token and combine plain system messages.

    Called only when bearer_session_compat is enabled, after ingress validation.
    Ordinary credential tokens and unmarked requests retain their behavior.
    """
    authorization = next((v for k, v in headers.items() if k.lower() == "authorization"), "")
    session = _SESSION.fullmatch(authorization)
    if session is None:
        return payload, headers
    # Explicit native headers retain priority; use the token only as fallback.
    from .constants import _NATIVE_SESSION_HEADERS
    native = {"x-opencode-session", *_NATIVE_SESSION_HEADERS}
    headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}
    if not any(k.lower() in native for k in headers):
        headers["x-session-id"] = session[1]
    messages = payload.get("messages", [])
    count = 0
    for message in messages:
        if (message.get("role") != "system" or not isinstance(message.get("content"), str)
                or set(message) != {"role", "content"}):
            break
        count += 1
    if count > 1:
        payload = dict(payload, messages=[{
            "role": "system", "content": "\n\n".join(m["content"] for m in messages[:count]),
        }, *messages[count:]])
    return payload, headers
