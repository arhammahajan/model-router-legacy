"""Generation dispatch joining a resolved route, protocol policy and provider adapters."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from itertools import chain
from typing import Any, AsyncIterator, Iterator, Mapping

import httpx

from . import gemini
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
from .client_compat import prepare_chat
from .count_tokens import count_tokens as _count_tokens
from .constants import (
    DIALECT_ANTHROPIC,
    DIALECT_GEMINI,
    DIALECT_OPENAI_CHAT,
    DIALECT_OPENAI_RESPONSES,
)
from .discovery import DiscoveryRegistry
from .headers import (
    _filtered_response_headers,
    _gemini_url,
    _session_id,
    _translation_header_provider,
    _upstream_request_headers,
    _upstream_url,
)
from .protocol import (
    _clean_request_params,
    _prepare_gemini_translation,
    _prepare_translation,
    _validate_ingress,
    _validated_generation_extensions,
)
from .streams import Plan, StreamState, _chat_sse, _responses_sse, _translated_sse
from .transport import _GenerationTransport, _RequestBudget
from .tools import ToolPlan, plan_tools


def _upstream_error(status: int, data: bytes, ingress: str) -> RouterError:
    """The untrusted upstream error body as a RouterError.

    A Gemini ingress keeps an upstream google.rpc message and status name when the
    body has them; everything is sanitized again when the response is rendered.
    """
    text = data.decode("utf-8", errors="replace") or f"Upstream returned HTTP {status}"
    if ingress != DIALECT_GEMINI:
        return RouterError(status, text)
    try:
        error = json.loads(data).get("error")
    except (ValueError, AttributeError):
        error = None
    if isinstance(error, dict) and isinstance(error.get("message"), str) and error["message"]:
        status_name = error.get("status")
        return RouterError(status, error["message"],
                           error_type=status_name if isinstance(status_name, str) else None)
    return RouterError(status, text)


class App(DiscoveryRegistry):
    """Dispatch a pinned registry route through native or translated transport."""

    def dispatch(
        self,
        ingress: str,
        route: Route,
        payload: dict[str, Any],
        inbound_headers: Mapping[str, str],
        *,
        count_tokens: bool = False,
        alt: str | None = None,
    ) -> Plan:
        # A token count shares the generation transport's deadline, retry policy
        # and proxy/redirect rules but never reaches a generation path.
        budget = _RequestBudget(self, counting=count_tokens)
        provider = route._provider or self.provider(route.provider_name)
        transport = _GenerationTransport(budget, provider)
        # Each generation owns its pool. The inner transport selects environment
        # proxies; outer proxy mounts must not bypass the policy boundary.
        client = httpx.Client(transport=transport, trust_env=False, follow_redirects=False,
                              timeout=provider.timeout)
        plan = None
        try:
            if count_tokens:
                plan = _count_tokens(route, provider, payload, inbound_headers, client,
                                     self.sanitize)
            else:
                plan = self._dispatch(ingress, route, payload, inbound_headers, client, alt)
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
        alt: str | None = None,
    ) -> Plan:
        _validate_ingress(ingress, payload, translated=ingress != route.dialect)
        if self.config.bearer_session_compat and ingress == DIALECT_OPENAI_CHAT:
            payload, inbound_headers = prepare_chat(payload, inbound_headers)
        provider = route._provider or self.provider(route.provider_name)
        session_id = _session_id(provider, payload, inbound_headers)
        if ingress == route.dialect:
            return self._passthrough(
                ingress, provider, route, payload, inbound_headers, session_id, client, alt
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
        tool_plan = None
        contract = self._route_entry(route if route._provider else replace(route, _provider=provider),
                                     int(time.time()), self._registry)["metadata"]["router"]
        translation = None
        if ingress == DIALECT_GEMINI:
            # Gemini has no LiteLLM ingress adapter. Its request becomes Chat
            # parameters (with losses named by their Gemini paths), and the Chat
            # result is converted back below.
            translation = gemini.to_chat(
                {k: v for k, v in payload.items() if k not in ("model", "stream")},
                route.dialect, stream=bool(payload.get("stream")))
            params, losses, native = _prepare_gemini_translation(
                provider, route, translation, contract)
        else:
            tool_plan = plan_tools(_clean_request_params(payload, ingress), ingress, route.dialect)
            params, losses, native = _prepare_translation(
                provider, route, ingress, payload, contract, tool_plan=tool_plan)
        provider = _translation_header_provider(provider, inbound_headers)
        transport = None
        kwargs = _thaw(provider.litellm_kwargs)
        transport = adapter.transport(client, provider, session_id)
        kwargs["client"] = transport
        provider = replace(provider, litellm_kwargs=kwargs)
        try:
            if ingress == DIALECT_OPENAI_CHAT:
                plan = self._from_openai_chat(provider, route, params, session_id, native, tool_plan)
            elif ingress == DIALECT_OPENAI_RESPONSES:
                plan = self._from_openai_responses(provider, route, params, session_id, tool_plan, native)
            elif translation is not None:
                plan = self._from_gemini(provider, route, params, session_id, native,
                                         translation, alt)
            else:
                plan = self._from_anthropic(provider, route, params, session_id, native)
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
        alt: str | None = None,
    ) -> Plan:
        assert client is not None
        streaming = bool(payload.get("stream"))
        if ingress == DIALECT_GEMINI:
            # The model and the streaming mode live in the path, not the body.
            url = _gemini_url(provider, route.upstream_model,
                              "streamGenerateContent" if streaming else "generateContent", alt)
            body = {k: v for k, v in payload.items() if k not in ("model", "stream")}
        else:
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

        if status >= 400 or not streaming:
            try:
                data = response.read()
            finally:
                response.close()
            if status >= 400:
                # Error bodies are untrusted diagnostics, not successful vendor
                # output. Deliberately trade byte fidelity for redaction.
                exc = _upstream_error(status, data, ingress)
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

    def _chat_completion(
        self,
        provider: Provider,
        route: Route,
        payload: dict[str, Any],
        session_id: str | None,
        native: Mapping[str, Mapping[str, Any]] | None,
    ) -> tuple[Any, bool]:
        """Run Chat Completions parameters against the route; (result, streaming)."""
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
            provider, params, model=model, session_id=session_id, native=native
        ))
        return result, stream

    def _from_openai_chat(
        self,
        provider: Provider,
        route: Route,
        payload: dict[str, Any],
        session_id: str | None = None,
        native: Mapping[str, Mapping[str, Any]] | None = None,
        tools: ToolPlan | None = None,
    ) -> Plan:
        result, stream = self._chat_completion(provider, route, payload, session_id, native)
        if stream:
            state = StreamState()
            return Plan(
                content_type="text/event-stream",
                chunks=_chat_sse(tools.chat_stream(result) if tools else result, self.sanitize, state),
                resources=(result,), stream_state=state,
            )
        data = tools.response(result) if tools else _jsonable(result)
        return Plan(body=_json_bytes(_sanitize_generated_errors(data, self.sanitize)))

    def _from_gemini(
        self,
        provider: Provider,
        route: Route,
        payload: dict[str, Any],
        session_id: str | None,
        native: Mapping[str, Mapping[str, Any]] | None,
        translation: gemini.Translation,
        alt: str | None,
    ) -> Plan:
        """Gemini ingress on a non-Gemini route: Chat out, GenerateContentResponse back."""
        result, stream = self._chat_completion(provider, route, payload, session_id, native)
        if not stream:
            data = _sanitize_generated_errors(_jsonable(result), self.sanitize)
            return Plan(body=_json_bytes(gemini.chat_to_gemini(data, translation, route.upstream_model)))

        chunks = (_sanitize_generated_errors(_jsonable(chunk), self.sanitize) for chunk in result)
        encoded = (_json_bytes(frame) for frame in
                   gemini.gemini_frames(chunks, translation, route.upstream_model))
        if alt == "sse":
            return Plan(content_type="text/event-stream",
                        chunks=(gemini.sse_frame(frame) for frame in encoded),
                        resources=(result,), stream_state=StreamState())
        return Plan(content_type="application/json", chunks=gemini.json_array_frames(encoded),
                    resources=(result,), stream_state=StreamState())

    def _from_openai_responses(
        self,
        provider: Provider,
        route: Route,
        payload: dict[str, Any],
        session_id: str | None = None,
        tools: ToolPlan | None = None,
        native: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> Plan:
        params = _clean_request_params(payload, DIALECT_OPENAI_RESPONSES)
        stream = bool(params.get("stream"))

        model = f"{provider.litellm_provider}/{route.upstream_model}"
        call_kwargs = adapter.call_kwargs(
            provider, params, model=model, session_id=session_id, native=native
        )
        if route.dialect == DIALECT_OPENAI_CHAT:
            call_kwargs["use_chat_completions_api"] = True
        result = adapter.responses(**call_kwargs)

        def restore(value: Any) -> Any:
            return tools.response(value) if tools else _jsonable(value)

        if stream:
            state = StreamState()
            return Plan(
                content_type="text/event-stream",
                chunks=_responses_sse(tools.responses_stream(result) if tools else result, self.sanitize, state),
                resources=(result,), stream_state=state,
            )
        return Plan(body=_json_bytes(_sanitize_generated_errors(restore(result), self.sanitize)))

    def _from_anthropic(
        self,
        provider: Provider,
        route: Route,
        payload: dict[str, Any],
        session_id: str | None = None,
        native: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> Plan:
        payload = _clean_request_params(payload, DIALECT_ANTHROPIC)
        if payload.get("max_tokens") is None:
            raise RouterError(400, "'max_tokens' is required", code="max_tokens")
        if route.dialect in (DIALECT_OPENAI_CHAT, DIALECT_GEMINI):
            return self._anthropic_to_chat(provider, route, payload, session_id, native)
        if route.dialect == DIALECT_OPENAI_RESPONSES:
            return self._anthropic_to_responses(provider, route, payload, session_id, native)
        raise RouterError(
            500, f"No Anthropic translation for dialect {route.dialect!r}"
        )

    def _anthropic_to_chat(
        self,
        provider: Provider,
        route: Route,
        payload: dict[str, Any],
        session_id: str | None = None,
        native: Mapping[str, Mapping[str, Any]] | None = None,
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
        choice = payload.get("tool_choice")
        if (route.dialect == DIALECT_OPENAI_CHAT and isinstance(choice, dict)
                and type(choice.get("disable_parallel_tool_use")) is bool):
            completion_params["parallel_tool_calls"] = not choice["disable_parallel_tool_use"]
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
            session_id=session_id, native=native,
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
        native: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> Plan:
        handler_kwargs = adapter.call_kwargs(
            provider, payload,
            model=f"{provider.litellm_provider}/{route.upstream_model}",
            session_id=session_id, native=native,
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
