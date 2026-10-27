"""Pinned LiteLLM integration; all private adapter APIs stay in this module."""
from __future__ import annotations

import os
from importlib import metadata as importlib_metadata
from typing import Any, Iterator, Mapping

from .common import RouterError, _close_quietly
from .config import Provider
from .constants import _TRANSLATION_HEADERS, DIALECT_OPENAI_CHAT, TESTED_LITELLM_VERSION
from .headers import _set_header

# Keep LiteLLM self-contained and quiet.  Credentials always come from the
# router config; ambient provider keys are never used.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def _require_litellm_version() -> str:
    """Private adapters require an explicitly tested release, not a major range."""
    try:
        installed = importlib_metadata.version("litellm")
    except importlib_metadata.PackageNotFoundError:
        installed = "not installed"
    if installed != TESTED_LITELLM_VERSION:
        raise SystemExit(
            f"Incompatible LiteLLM version: {installed}; this router requires "
            f"litellm=={TESTED_LITELLM_VERSION} for its internal translation adapters. "
            f'Restore with: python -m pip install -c constraints.txt "litellm[mcp]=={TESTED_LITELLM_VERSION}". '
            "Contributors: see CONTRIBUTING.md before testing a dependency upgrade."
        )
    return installed


_require_litellm_version()

try:
    import httpx
    import litellm

    # LiteLLM's Anthropic Messages adapters.  These are the same code paths the
    # LiteLLM proxy uses for /v1/messages <-> chat/responses translation.
    from litellm.llms.anthropic.pass_through.adapters.transformation import (
        AnthropicAdapter,
    )
    from litellm.llms.anthropic.pass_through.responses_adapters.handler import (
        LiteLLMMessagesToResponsesAPIHandler,
    )
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    # 1.106.4 imports this even when skip_mcp_handler=True. Require its extra
    # at installation/startup instead of failing on the first Responses call.
    from litellm.responses.mcp.litellm_proxy_mcp_handler import (
        LiteLLM_Proxy_MCP_Handler,
    )
    from openai import OpenAI

    _ANTHROPIC_ADAPTER = AnthropicAdapter()
    for adapter, methods in (
        (_ANTHROPIC_ADAPTER, (
            "translate_completion_input_params_with_tool_mapping",
            "translate_completion_output_params",
            "translate_completion_output_params_streaming",
        )),
        (LiteLLMMessagesToResponsesAPIHandler, ("anthropic_messages_handler",)),
        (HTTPHandler, ("post", "close")),
        (LiteLLM_Proxy_MCP_Handler, ("should_use_litellm_mcp_gateway",)),
    ):
        for method in methods:
            if not callable(getattr(adapter, method, None)):
                raise AttributeError(f"Missing adapter method: {method}")
except (ImportError, AttributeError, TypeError) as exc:
    raise SystemExit(
        f"LiteLLM {TESTED_LITELLM_VERSION} adapter initialization failed "
        f"({type(exc).__name__}); the installation is incomplete or incompatible. "
        f'Restore with: python -m pip install -c constraints.txt --force-reinstall "litellm[mcp]=={TESTED_LITELLM_VERSION}". '
        "Use constraints.txt for the tested transitive dependencies; see CONTRIBUTING.md."
    ) from exc


litellm.telemetry = False
litellm.suppress_debug_info = True
litellm.drop_params = False


class _TranslationHTTPTransport(HTTPHandler):
    """Request-scoped header selection; App owns the shared HTTP connection pool."""

    def __init__(self, client: httpx.Client, provider: Provider) -> None:
        super().__init__(client=client)
        self.responses: list[httpx.Response] = []
        self.selected = {key.lower(): val for key, val in provider.headers.items()
                         if key.lower() in _TRANSLATION_HEADERS[provider.dialect]}
        self.selected.update(provider.auth_headers())

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        headers = dict(kwargs.pop("headers", {}) or {})
        for key, val in self.selected.items():
            _set_header(headers, key, val)
        response = super().post(url, headers=headers, **kwargs)
        self.responses.append(response)
        return response

    def close(self) -> None:
        responses, self.responses = self.responses, []
        for response in responses:
            _close_quietly(response.close)


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
    # No SDK, wrapper, or fallback layer owns retry policy. In particular the
    # pinned sync wrapper treats num_retries=0 as "use global", so reject globals
    # rather than temporarily mutating process-wide state during concurrent calls.
    for setting in ("num_retries", "retry_policy", "fallbacks",
                    "context_window_fallback_dict"):
        if getattr(litellm, setting, None):
            raise RouterError(500, f"Global LiteLLM {setting} bypasses router retry policy")
    result.update(num_retries=0, max_retries=0, retry_policy=None,
                  fallbacks=[], context_window_fallback_dict={})
    if body:
        result["extra_body"] = body
    # The router forwards tool definitions; it must not execute MCP locally.
    result["_skip_mcp_handler"] = True
    return result


class LiteLLMAdapter:
    """The sole interface to LiteLLM translation and its private wrappers.

    Request validation and route selection belong to the router. This boundary
    owns version checks, SDK clients, adapter calls and hidden stream resources.
    """

    unsupported_params_error = litellm.UnsupportedParamsError

    def completion(self, **kwargs: Any) -> Any:
        return litellm.completion(**kwargs)

    def responses(self, **kwargs: Any) -> Any:
        return litellm.responses(**kwargs)

    def supported_openai_params(self, **kwargs: Any) -> Any:
        return litellm.get_supported_openai_params(**kwargs)

    def messages_input(self, *args: Any, **kwargs: Any) -> Any:
        return _ANTHROPIC_ADAPTER.translate_completion_input_params_with_tool_mapping(*args, **kwargs)

    def messages_output(self, *args: Any, **kwargs: Any) -> Any:
        return _ANTHROPIC_ADAPTER.translate_completion_output_params(*args, **kwargs)

    def messages_stream(self, *args: Any, **kwargs: Any) -> Any:
        return _ANTHROPIC_ADAPTER.translate_completion_output_params_streaming(*args, **kwargs)

    def messages_to_responses(self, **kwargs: Any) -> Any:
        return LiteLLMMessagesToResponsesAPIHandler.anthropic_messages_handler(**kwargs)

    def call_kwargs(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return _translation_call_kwargs(*args, **kwargs)

    def transport(self, client: httpx.Client, provider: Provider,
                  session_id: str | None) -> Any:
        if provider.dialect == DIALECT_OPENAI_CHAT:
            trusted = provider.litellm_call_kwargs(session_id=session_id)
            return OpenAI(api_key=provider.effective_api_key,
                          base_url=trusted["api_base"], http_client=client,
                          default_headers=trusted["extra_headers"], max_retries=0)
        return _TranslationHTTPTransport(client, provider)

    def stream_resources(self, resource: Any) -> Iterator[Any]:
        # Only request resources; clients/pools are owned explicitly by Plan.
        if not type(resource).__module__.startswith("litellm."):
            return
        for name in (
            "completion_stream", "litellm_custom_stream_wrapper",
            "response", "http_response", "responses_stream",
            "streaming_response", "stream_iterator", "_wrapper",
            "_anthropic_wrapper",
        ):
            yield getattr(resource, name, None)


adapter = LiteLLMAdapter()
