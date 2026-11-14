"""Protocol names, defaults and closed field/header sets."""
from __future__ import annotations

import logging
import re

__version__ = "0.3.0"
TESTED_LITELLM_VERSION = "1.106.4"

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

# Token counting is not generation and has its own routes. Keep these out of
# ENDPOINTS, which names generation ingress dialects.
COUNT_TOKENS_ENDPOINTS = frozenset({"/v1/messages/count_tokens", "/messages/count_tokens"})
# The only documented success shape is {"input_tokens": n}; anything near this
# size is not a token count. Applies to success and error bodies alike.
MAX_COUNT_RESPONSE_BYTES = 64 * 1024
# The one host whose token-count endpoint is documented. Other Anthropic-protocol
# providers must opt in with `count_tokens = true`.
COUNT_TOKENS_NATIVE_HOSTS = frozenset({"api.anthropic.com"})

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
# llama.cpp has no reasoning_effort control; it takes a token budget
# (thinking_budget_tokens) or chat_template_kwargs.enable_thinking instead. The
# router maps the standard OpenAI effort onto those native fields. "max" is
# deliberately absent: it means "no budget", i.e. the server default.
LLAMA_THINKING_BUDGETS = {
    "minimal": 1024, "low": 2048, "medium": 8192, "high": 16384, "xhigh": 32768,
}
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
    "x-model-router-reasoning-effort",
    # Quota/billing attribution is the operator's choice, never the client's.
    "x-goog-user-project",
    "x-goog-quota-user",
    "cookie",
    "user-agent",
} | _REQUEST_REPRESENTATION_HEADERS
# Reviewed native conversation headers; order is the default extraction priority.
# Sources and review date are recorded in README's conversation affinity policy.
_NATIVE_SESSION_HEADERS = (
    "x-claude-code-session-id", "session-id", "session_id", "thread-id",
    "conversation_id", "x-session-id", "x-mavis-session-id",
)
_SESSION_VALUE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}\Z")
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
        "background", "client_metadata", "conversation", "include", "input", "instructions",
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

# google.rpc.Code names for the HTTP statuses the router can produce.
_GEMINI_ERROR_STATUSES = {
    400: "INVALID_ARGUMENT",
    401: "UNAUTHENTICATED",
    403: "PERMISSION_DENIED",
    404: "NOT_FOUND",
    408: "DEADLINE_EXCEEDED",
    409: "ABORTED",
    413: "INVALID_ARGUMENT",
    429: "RESOURCE_EXHAUSTED",
    499: "CANCELLED",
    500: "INTERNAL",
    501: "UNIMPLEMENTED",
    502: "UNAVAILABLE",
    503: "UNAVAILABLE",
    504: "DEADLINE_EXCEEDED",
}

DISCOVERY_CACHE_VERSION = 1
DISCOVERY_CACHE_MAX_BYTES = 8 * 1024 * 1024

_TRANSLATION_HEADERS = {
    DIALECT_ANTHROPIC: {"anthropic-beta", "anthropic-version"},
    DIALECT_OPENAI_CHAT: {"openai-beta"},
    DIALECT_OPENAI_RESPONSES: {"openai-beta"},
    DIALECT_GEMINI: set(),
}
