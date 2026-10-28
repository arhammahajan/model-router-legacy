"""Opt-in compatibility for clients that can only supply bearer tokens."""
from __future__ import annotations

import re
from typing import Any, Mapping

_SESSION = re.compile(r"Bearer (router-session-[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})")


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
