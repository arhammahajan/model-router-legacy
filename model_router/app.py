"""Generation dispatch joining a resolved route, protocol policy and provider adapters."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from itertools import chain
from typing import Any, AsyncIterator, Iterator, Mapping

import httpx

from .adapters import adapter
from .common import (
    RouterError,
    _apply_llama_thinking,
    _error_payload,
    _json_bytes,
    _jsonable,
    _sanitize_generated_errors,
    _thaw,
)
from .config import Provider, Route
from .constants import (
    DIALECT_ANTHROPIC,
    DIALECT_GEMINI,
    DIALECT_OPENAI_CHAT,
    DIALECT_OPENAI_RESPONSES,
)
from .discovery import DiscoveryRegistry
from .headers import (
    _filtered_response_headers,
    _session_id,
    _translation_header_provider,
    _upstream_request_headers,
    _upstream_url,
)
from .protocol import (
    _clean_request_params,
    _prepare_translation,
    _validate_ingress,
    _validated_generation_extensions,
)
from .streams import Plan, StreamState, _chat_sse, _responses_sse, _translated_sse
from .transport import _GenerationTransport, _RequestBudget


class App(DiscoveryRegistry):
    """Dispatch a pinned registry route through native or translated transport."""

    def dispatch(
        self,
        ingress: str,
        route: Route,
        payload: dict[str, Any],
        inbound_headers: Mapping[str, str],
    ) -> Plan:
        budget = _RequestBudget(self)
        provider = route._provider or self.provider(route.provider_name)
        transport = _GenerationTransport(budget, provider)
        # Each generation owns its pool. The inner transport selects environment
        # proxies; outer proxy mounts must not bypass the policy boundary.
        client = httpx.Client(transport=transport, trust_env=False, follow_redirects=False,
                              timeout=provider.timeout)
        plan = None
        try:
            plan = self._dispatch(ingress, route, payload, inbound_headers, client)
            budget.timeout()
        except BaseException as exc:
            if plan is not None:
                plan.close()
            client.close()
            if isinstance(exc, Exception):
                if budget.failure is not None:
                    raise budget.failure from exc
                if budget.clock() >= budget.end:
                    raise budget.error() from exc
            raise
        plan.resources += (client,)
        # These headers are also available on a 200 SSE response; headers cannot
        # be changed if a later stream error occurs.
        for name, value in budget.headers.items():
            plan.headers.setdefault(name, self.sanitize(value))
        if plan.body is not None:
            # Nonstream plans contain fully materialized bytes and need no pool.
            client.close()
        else:
            self._limit_plan(plan, budget)
        return plan

    def _limit_plan(self, plan: Plan, budget: _RequestBudget) -> None:
        # Adapters can buffer events or synthesize framing without socket IO.
        # Check those against the same deadline, including between async pulls.
        if plan.chunks is not None:
            source = plan.chunks

            def chunks() -> Iterator[bytes]:
                iterator = iter(source)
                while True:
                    budget.timeout()
                    try:
                        chunk = next(iterator)
                    except StopIteration:
                        budget.timeout()
                        return
                    except Exception:
                        if budget.failure is not None:
                            raise budget.failure
                        budget.timeout()
                        raise
                    budget.timeout()
                    # Some adapters defer opening HTTP until the first pull.
                    for name, value in budget.headers.items():
                        plan.headers.setdefault(name, self.sanitize(value))
                    # This includes adapter-synthesized framing emitted before
                    # a lazy source opens HTTP. Even that output forbids replay.
                    budget.output_started = True
                    yield chunk

            plan.resources += (source,)
            plan.chunks = chunks()
        if plan.async_chunks is not None:
            async_source = plan.async_chunks

            async def async_chunks() -> AsyncIterator[bytes]:
                iterator = async_source.__aiter__()
                while True:
                    try:
                        remaining = budget.timeout()
                        chunk = await asyncio.wait_for(anext(iterator), remaining)
                    except StopAsyncIteration:
                        budget.timeout()
                        return
                    except asyncio.TimeoutError as exc:
                        raise budget.error() from exc
                    except Exception:
                        if budget.failure is not None:
                            raise budget.failure
                        budget.timeout()
                        raise
                    budget.timeout()
                    for name, value in budget.headers.items():
                        plan.headers.setdefault(name, self.sanitize(value))
                    budget.output_started = True
                    yield chunk

            plan.resources += (async_source,)
            plan.async_chunks = async_chunks()

    def _dispatch(
        self, ingress: str, route: Route, payload: dict[str, Any],
        inbound_headers: Mapping[str, str], client: httpx.Client,
    ) -> Plan:
        _validate_ingress(ingress, payload, translated=ingress != route.dialect)
        provider = route._provider or self.provider(route.provider_name)
        session_id = _session_id(provider, payload, inbound_headers)
        if ingress == route.dialect:
            return self._passthrough(
                ingress, provider, route, payload, inbound_headers, session_id, client
            )
        expected_adapter = {
            DIALECT_OPENAI_CHAT: "openai", DIALECT_OPENAI_RESPONSES: "openai",
            DIALECT_ANTHROPIC: "anthropic", DIALECT_GEMINI: "gemini",
        }[provider.dialect]
        if (provider.litellm_provider != expected_adapter
                or provider.litellm_kwargs.get("custom_llm_provider") not in (None, expected_adapter)):
            raise RouterError(500, "Router retry/transport policy requires the reviewed "
                              f"{expected_adapter!r} adapter for {provider.dialect!r} translation. "
                              "Use a native route for other adapters.", code="retry_adapter_unsupported")
        if provider.session_header and provider.session_prompt_cache_key:
            # Explicit affinity metadata is consumed before protocol validation.
            # Native passthrough keeps the provider's original cache field.
            payload = {key: value for key, value in payload.items() if key != "prompt_cache_key"}
        contract = self._route_entry(route if route._provider else replace(route, _provider=provider),
                                     int(time.time()), self._registry)["metadata"]["router"]
        params, losses = _prepare_translation(provider, route, ingress, payload, contract)
        provider = _translation_header_provider(provider, inbound_headers)
        transport = None
        kwargs = _thaw(provider.litellm_kwargs)
        transport = adapter.transport(client, provider, session_id)
        kwargs["client"] = transport
        provider = replace(provider, litellm_kwargs=kwargs)
        try:
            if ingress == DIALECT_OPENAI_CHAT:
                plan = self._from_openai_chat(provider, route, params, session_id)
            elif ingress == DIALECT_OPENAI_RESPONSES:
                plan = self._from_openai_responses(provider, route, params, session_id)
            else:
                plan = self._from_anthropic(provider, route, params, session_id)
        except adapter.unsupported_params_error as exc:
            if transport is not None:
                transport.close()
            raise RouterError(400, "The target adapter cannot translate these generation controls. "
                              "Use a native route or review the provider compatibility policy.",
                              code="translation_loss") from exc
        except BaseException:
            if transport is not None:
                transport.close()
            raise
        if transport is not None:
            plan.resources += (transport,)
        if losses:
            plan.headers["x-model-router-loss"] = json.dumps(losses, separators=(",", ":"))
        return plan

    def _passthrough(
        self,
        ingress: str,
        provider: Provider,
        route: Route,
        payload: dict[str, Any],
        inbound_headers: Mapping[str, str],
        session_id: str | None = None,
        client: httpx.Client | None = None,
    ) -> Plan:
        assert client is not None
        url = _upstream_url(provider, ingress)
        body = dict(payload)
        body["model"] = route.upstream_model
        if provider.llama_cpp and ingress == DIALECT_OPENAI_CHAT:
            _apply_llama_thinking(body, body.pop("reasoning_effort", None))

        request = client.build_request(
            "POST",
            url,
            json=body,
            headers=_upstream_request_headers(
                provider, ingress, inbound_headers, session_id
            ),
            timeout=httpx.Timeout(provider.timeout, connect=min(20.0, provider.timeout)),
        )
        response = client.send(request, stream=True)

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

        return Plan(
            status=status,
            content_type=content_type,
            headers=headers,
            chunks=response.iter_bytes(),
            resources=(response,),
        )

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

        result = adapter.completion(**adapter.call_kwargs(
            provider, params, model=model, session_id=session_id
        ))

        if stream:
            state = StreamState()
            return Plan(
                content_type="text/event-stream",
                chunks=_chat_sse(result, self.sanitize, state),
                resources=(result,), stream_state=state,
            )
        return Plan(body=_json_bytes(_sanitize_generated_errors(_jsonable(result), self.sanitize)))

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
        call_kwargs = adapter.call_kwargs(
            provider, params, model=model, session_id=session_id
        )
        if route.dialect == DIALECT_OPENAI_CHAT:
            call_kwargs["use_chat_completions_api"] = True
        result = adapter.responses(**call_kwargs)

        if stream:
            state = StreamState()
            return Plan(
                content_type="text/event-stream",
                chunks=_responses_sse(result, self.sanitize, state),
                resources=(result,), stream_state=state,
            )
        return Plan(body=_json_bytes(_sanitize_generated_errors(_jsonable(result), self.sanitize)))

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
            adapter.messages_input(
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

        result = adapter.completion(**adapter.call_kwargs(
            provider, completion_params,
            model=f"{provider.litellm_provider}/{route.upstream_model}",
            session_id=session_id,
        ))

        if stream:
            # Resolve lazy HTTP failures before the adapter synthesizes its
            # message_start. Its sync iterator can swallow later exceptions;
            # retain those so the router can emit a sanitized terminal error.
            owner = Plan(resources=(result,))
            try:
                first = next(result)
            except StopIteration:
                source = iter(())
            except BaseException:
                owner.close()
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

            observed_source = observed()
            owner.resources += (source, observed_source)
            try:
                transformed = adapter.messages_stream(
                    observed_source,
                    model=route.upstream_model,
                    tool_name_mapping=tool_name_mapping,
                    is_async=False,
                )
                if transformed is None:
                    raise RouterError(502, "Failed to transform the streaming response")
            except BaseException:
                owner.close()
                raise
            state = StreamState()

            def chunks() -> Iterator[bytes]:
                for chunk in transformed:
                    # The sync adapter swallows upstream exceptions and may
                    # synthesize a success terminator. Never forward it.
                    if failures:
                        raise failures[0]
                    yield _translated_sse(chunk, self.sanitize, state)
                    if state.error is not None:
                        return
                if failures:
                    raise failures[0]

            owner.content_type = "text/event-stream"
            owner.chunks = chunks()
            owner.resources += (transformed,)
            owner.stream_state = state
            return owner

        translated = adapter.messages_output(
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
        handler_kwargs = adapter.call_kwargs(
            provider, payload,
            model=f"{provider.litellm_provider}/{route.upstream_model}",
            session_id=session_id,
        )

        result = adapter.messages_to_responses(
            **handler_kwargs
        )

        if payload.get("stream"):
            state = StreamState()

            async def chunks() -> AsyncIterator[bytes]:
                async for chunk in result:
                    yield _translated_sse(chunk, self.sanitize, state)
                    if state.error is not None:
                        return

            return Plan(content_type="text/event-stream", async_chunks=chunks(),
                        resources=(result,), stream_state=state)
        return Plan(body=_json_bytes(_sanitize_generated_errors(_jsonable(result), self.sanitize)))
