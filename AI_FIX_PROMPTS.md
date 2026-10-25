# Focused implementation prompts

Give an AI one numbered prompt at a time. The prompts assume access to this repository and `AUDIT.md`. Run them sequentially in one checkout: most touch `model_router.py`. If previous work changes the implementation, use the current behavior and symbols rather than old audit line numbers.

You can invoke a task with: **“Read `AI_FIX_PROMPTS.md` and implement prompt 01, following its working rules.”**

## Working rules for every prompt

1. Implement only the selected prompt. The other prompts provide context, not authorization to expand the task.
2. Read applicable `AGENTS.md` instructions, the relevant audit findings, and current implementation/tests first. Treat audit findings as hypotheses to verify against the current code.
3. Make the scoped changes and add meaningful regression tests. Preserve unrelated behavior and edits. Refactor only what the selected task needs.
4. Use fake credentials and local test upstreams. Do not read personal credential files, print real secrets, or make paid inference requests. Consult current official documentation when provider behavior matters; do not guess.
5. Run targeted tests, then `.venv/bin/python -m pytest`. Fix failures caused by the changes. Do not weaken assertions or disable tests to obtain a passing result.
6. Update affected documentation/examples and record compatibility changes. Keep `AUDIT.md` as historical evidence; describe resolutions in the completion report or a separate tracking note.
7. Finish with changed behavior, files changed, test results, and unresolved limitations. Complete the implementation rather than returning only a plan. Do not commit, push, publish, or deploy unless separately requested.

## 01 — Prevent client control of LiteLLM routing and transport

```text
Follow the working rules in AI_FIX_PROMPTS.md. Implement prompt 01 only.

Fix the credential escape described in audit finding 1. Client JSON currently flows into LiteLLM keyword arguments after only a small denylist filter; client-supplied fallback dictionaries can change api_base while inheriting the server's provider key.

Build an explicit boundary between protocol generation parameters and trusted administrative settings. Apply it to all translated ingresses. Clients must not control destinations, credentials, provider selection, fallbacks, model lists, retries, callbacks, clients/transports, or mock execution. Review nested extension data and argument-merging paths as well as top-level keys. Preserve legitimate generation extensions through a separately validated mechanism, and preserve same-protocol vendor-field behavior where safe.

Add local HTTP regression tests using two upstreams and dummy keys. Prove that client-supplied settings cannot redirect a request or send the configured key to the second upstream. Cover Chat and Responses translation, inspect the Anthropic adapters for equivalent entry points, and verify ordinary tools/reasoning requests still work. Do not solve this by adding only a few names to the existing denylist.
```

## 02 — Enforce credential isolation and consistent redaction

```text
Follow the working rules in AI_FIX_PROMPTS.md. Implement prompt 02 only.

Fix audit findings 3 and 4. An uncredentialed Anthropic provider can inherit ambient credentials during translation, and _error_payload bypasses sanitization when an exception has a nonempty message attribute.

Define explicit behavior for configured credentials and intentionally unauthenticated upstreams. Make translated and passthrough calls consistent, without mutating process-wide environment variables around concurrent requests. Ensure unrelated environment/global LiteLLM credentials cannot be selected accidentally.

Route generated JSON and SSE error messages through one sanitizer and length limit. Cover configured secret values, bearer tokens, arbitrary provider-key formats, exception.message, and exception stringification. Apply the policy to relevant diagnostic surfaces without logging raw secrets. Keep successful passthrough response fidelity; document a deliberate policy for raw upstream error bodies rather than silently promising both redaction and byte fidelity.

Test with dummy ambient keys and configured keys across dialects, plus synthetic and real adapter errors. Prove that neither credentials nor oversized exception messages escape through generated responses.
```

## 03 — Correct HTTP body framing, gzip decoding, and forwarding

```text
Follow the working rules in AI_FIX_PROMPTS.md. Implement prompt 03 only.

Fix audit findings 5 and 6 and the gzip-decoding cases in finding 15.

Validate chunk-size syntax before reading; reject signed/negative sizes and malformed framing. Enforce exact read lengths, CRLF delimiters, body limits, and bounded chunk metadata/trailers. Handle conflicting or unsupported transfer framing explicitly. Never pass an unchecked negative length to a stream read.

For gzip, enforce the decompressed limit and verify a complete valid stream. Choose and document a policy for concatenated members and trailing bytes. After decoding and serializing a request, ensure outgoing Content-Encoding, Content-Length, and other representation headers match the transmitted bytes. Preserve ordinary gzip and chunked requests.

Use raw-socket or faithful reader tests for malformed framing and truncated bodies. Make the fake upstream honor Content-Encoding so it detects plain JSON incorrectly advertised as gzip. Verify large malformed input cannot bypass a small configured limit. Keep connection admission and timeout redesign outside this task.
```

## 04 — Bound connections and generation concurrency

```text
Follow the working rules in AI_FIX_PROMPTS.md. Implement prompt 04 only.

Address the connection/resource portion of audit finding 5. ThreadingHTTPServer currently accepts unbounded handler threads, and clients can hold a connection while sending incomplete headers or bodies.

Add configurable, validated limits for accepted connections and in-flight generations, with explicit header/body read deadlines and bounded waiting or immediate overload rejection. A semaphore acquired only after spawning unlimited threads is insufficient. Separate inbound read deadlines from long-running upstream generation and streaming write timeouts. Avoid holding discovery locks while waiting for generation capacity.

Ensure capacity is released on success, validation failures, exceptions, timeout, disconnect, and shutdown. Preserve useful long-lived streaming behavior and the lightweight local-server deployment.

Add deterministic local tests with stalled readers/writers and held generations. Verify excess work is bounded, ordinary requests recover after cancellation, and tests terminate promptly. Document defaults and observable overload/timeout behavior.
```

## 05 — Correct OpenCode Go endpoint selection

```text
Follow the working rules in AI_FIX_PROMPTS.md. Implement prompt 05 only.

Fix audit finding 2. The bundled configuration routes every non-Claude/non-Gemini catalog model to OpenAI Responses, but OpenCode Go documents different native endpoints for individual model families and exceptions.

Check current official Go endpoint documentation and the public catalog. Add an OpenAI Chat upstream and a reviewed model-to-dialect mapping covering Chat, Responses, Messages, and Gemini. Support explicit overrides and a clear policy for unmapped new models; do not silently treat every new model as Responses. Keep provider credentials in trusted configuration. Treat endpoint compatibility that was not verified as unknown.

Use checked-in fixtures for normal tests. Test the actual bundled configuration's route selection for representative models and exceptions, including Qwen/MiniMax cases where documented. Validate unknown-model behavior. Update README examples and stale comments/user-agent versions. Make no authenticated cloud inference requests. Leave capability enrichment and conversation-session handling to their separate prompts.
```

## 06 — Stabilize discovered provider identities and registry snapshots

```text
Follow the working rules in AI_FIX_PROMPTS.md. Implement prompt 06 only.

Fix audit finding 8. Provider names are allocated according to reachability, stale providers can overwrite newly allocated identities, and requests can observe routes and providers from different refresh generations.

Assign deterministic identities to all configured llama.cpp servers before probing. Reject ambiguous duplicate explicit names with an actionable configuration error, and resolve generated-name collisions predictably. Health changes must not reassign an existing server's identity.

Publish providers, routes, public model entries, and their status as one consistent registry snapshot. A request must retain a compatible route/provider pair throughout dispatch, even while refresh removes or changes entries. Preserve documented explicit/discovered/catalog route precedence and stale-route behavior.

Add deterministic tests for generated-name collisions and server recovery/failure transitions. Reproduce the audit's duplicate-name case to prove it now fails validation or remains correctly isolated. Use barriers/events to exercise refresh between resolution and dispatch; avoid timing-dependent sleep tests. Background scheduling is a separate task.
```

## 07 — Move discovery refresh off request paths and add backoff

```text
Follow the working rules in AI_FIX_PROMPTS.md. Implement prompt 07 only; build on the stable registry work if available.

Fix audit finding 7. After a successful catalog becomes stale, failures can leave refresh immediately eligible on every request. Model listing, resolution, and health checks perform synchronous network refreshes.

Track last success, last attempt, and next attempt independently for catalog and llama.cpp discovery. Add bounded exponential backoff with jitter after failures, reset it on recovery, and prevent concurrent duplicate refreshes. Serve last-known usable routes without waiting for network refresh. Use a managed background worker with explicit startup, shutdown, and cleanup; preserve a documented startup-only refresh mode.

Publish refresh state atomically, including stale age and sanitized errors. Keep liveness network-free. Define what happens when discovery has never succeeded.

Test success-to-failure-to-recovery, partial server failures, startup failure, concurrent callers, and worker shutdown using a fake clock and controlled local responses. Verify a burst of requests cannot cause a burst of refresh attempts. Persistent on-disk caching is outside this task.
```

## 08 — Make discovered context and generation limits accurate

```text
Follow the working rules in AI_FIX_PROMPTS.md. Implement prompt 08 only.

Fix the limit-reporting portion of audit finding 9. Separate effective runtime context, trained context, input limits, maximum generation length, and default generation length. Stop advertising max_tokens equal to context size unless that meaning is explicitly established and documented for the field.

Read usable runtime context from model metadata and /props, including default_generation_settings.n_ctx for single-server mode. Handle supported split and equals forms of launch flags. Treat fitting constraints as constraints, not measured runtime limits. Use per-model information correctly in router mode and preserve legacy/llama-swap compatibility.

Mark fallback and inferred values with provenance; unknown output limits must remain unknown. Define precedence among measured values, explicit user overrides, and fallbacks. Preserve existing public API fields where possible, and document corrected semantics where compatibility requires a change.

Test conflicting sources, absent/invalid data, zero/auto values, context arguments, legacy mode, and configured overrides. Do not fabricate limits from model names.
```

## 09 — Expose a versioned model capability contract

```text
Follow the working rules in AI_FIX_PROMPTS.md. Implement prompt 09 only; reuse the corrected limits from prompt 08 if available.

Implement the capability proposal in AUDIT.md for both OpenCode Go and llama.cpp. Preserve the existing model-list envelope and add a documented versioned extension, such as metadata.router.

Represent reasoning support, toggles, known effort values, budgets, defaults, and native parameter mappings; input/output modalities; tools and parallel tool calls; structured output; supported sampling controls; native/translated protocols; and provenance/freshness. Distinguish unsupported from unknown. Do not assume every thinking model supports the same effort enum.

For llama.cpp, consume appropriate /props defaults, modalities, and template capabilities, with per-model queries where required and no accidental loading of unloaded models. For Go, join availability with a reviewed metadata source and explicit overrides; keep ordinary tests offline. Metadata overrides should enrich discovered routes without duplicating their routing configuration.

Test known, unknown, stale, conflicting, and overridden metadata. Document the schema and how a client uses it; extra model fields alone do not automatically configure SDKs or UIs. Do not claim reasoning levels that the source cannot establish.
```

## 10 — Make translation loss explicit and preserve necessary headers

```text
Follow the working rules in AI_FIX_PROMPTS.md. Implement prompt 10 only; integrate capability metadata when available.

Fix audit finding 10. Important generation controls can be silently dropped by LiteLLM, and cross-protocol calls discard needed beta/version headers.

Define a documented strict-versus-lossy compatibility policy. Meaningful unsupported controls, including reasoning, tools, and structured output, must produce an actionable error or an explicit loss diagnostic. Unknown capability metadata must not automatically mean unsupported. Preserve the protection established against client-controlled LiteLLM administrative settings.

Forward only appropriate dialect-specific beta/version headers during translation, with case-insensitive merging and clear precedence. Credentials and transport settings remain server-owned. Preserve same-protocol vendor fields where allowed by the established security policy.

Test supported reasoning mappings, unsupported and unknown settings, explicit lossy mode, beta/version forwarding, and credential-header protection across relevant ingresses. Use upstream request assertions to prove what was actually transmitted. Avoid promising that all native fields can be translated without loss.
```

## 11 — Preserve conversation-stable OpenCode sessions

```text
Follow the working rules in AI_FIX_PROMPTS.md. Implement prompt 11 only.

Fix audit finding 11. A fresh x-opencode-session UUID per request does not establish stable conversation affinity.

Define a configurable, documented session extraction policy. Support the explicit upstream session header, selected client-native conversation headers verified against current provider documentation, and prompt_cache_key as an explicit fallback. Preserve the identifier consistently across translated and passthrough calls. Specify precedence and behavior when no conversation identity is supplied; do not pretend a per-request fallback is a stable conversation ID.

Never use one global session for all conversations or derive identity solely from a provider/API key. Avoid hashing mutable prompt content into a supposed stable conversation identifier. Validate header values and keep credentials separate.

Test repeated turns, two independent conversations, conflicting identifier sources, absent identifiers, and all ingress dialects. Document how clients supply an identity and how the fallback affects affinity.
```

## 12 — Correct streaming cleanup, sequencing, and terminal failures

```text
Follow the working rules in AI_FIX_PROMPTS.md. Implement prompt 12 only.

Fix audit findings 12 and 13. The synchronous relay passes the iterator itself to a callable cleanup helper, so iterator.close() is skipped. Translated Responses events can preserve duplicate/decreasing sequence numbers, and synthesized errors do not share the stream counter.

Define iterator and upstream-transport ownership for synchronous and asynchronous plans. Close/aclose resources on completion, early failure, downstream disconnect, and cancellation. Stop upstream consumption promptly where supported. Make cleanup idempotent.

Own a monotonic sequence counter for translated Responses streams, including synthesized events. Preserve same-protocol streaming fidelity. Verify terminal error behavior for each ingress and distinguish an error before the first event from failure after HTTP headers have been sent. Record stream failure separately from HTTP status if response headers are already committed.

Test with close-tracking iterators and real local HTTP streams: empty output, first-event failure, midstream failure, duplicate/decreasing/missing sequence values, disconnect, and repeat requests. Assert event structure and resource cleanup, not just text output.
```

## 13 — Tighten request/config validation and reconcile Python support

```text
Follow the working rules in AI_FIX_PROMPTS.md. Implement prompt 13 only.

Fix audit finding 14 and the request/config validation portions of finding 15.

Validate ingress field types, particularly stream booleans, before dispatch. Distinguish malformed input or unsupported stateful Responses behavior from upstream failures; do not expose internal missing-argument errors as 502. Preserve controlled vendor extensions instead of indiscriminately rejecting unknown protocol fields.

Validate configuration types, finite numeric ranges, unknown fields, and environment substitution. An explicitly configured authentication secret that is missing must not silently disable auth. Provide a clear intentional unauthenticated configuration and migrate existing examples. Make dynamic provider/model routing configurable and document that disabled aliases alone are not a model access-control policy.

Either support Python 3.10 TOML loading with a conditional tomli dependency or raise the documented/package minimum consistently. Prefer a small dependency footprint.

Add focused tests for invalid types, missing secrets, typos, auth mode, dynamic-route policy, and the selected Python compatibility strategy. Keep framing/gzip fixes in their dedicated task.
```

## 14 — Separate health, readiness, diagnostics, and CLI checks

```text
Follow the working rules in AI_FIX_PROMPTS.md. Implement prompt 14 only; use cached discovery state from prior fixes.

Address the operational portions of audit finding 15. Define network-free liveness, readiness with documented criteria, and authenticated diagnostics containing sanitized provider/discovery details. Decide how existing /health and /healthz clients migrate. A transient unavailable provider need not make every otherwise usable route unready; specify the policy.

Separate config validation from connectivity/readiness checking in the CLI. Define useful exit codes and handle offline startup-only checks explicitly. Preserve --check compatibility where practical, with clear help text.

Add request IDs and structured outcome logging for provider/model selection, translation mode, elapsed time, and stream outcome. Surface discovery staleness and failure counts without logging prompts, completions, or secrets. Respect the distinction between an HTTP 200 stream and a later stream failure.

Test auth boundaries, sanitized diagnostics, partial outages, no successful discovery, zero network work during liveness, and CLI exit behavior. Avoid adding an external metrics service dependency.
```

## 15 — Complete protocol contract regression coverage

```text
Follow the working rules in AI_FIX_PROMPTS.md. Implement prompt 15 only, after the targeted fixes have landed.

Strengthen the coverage described in AUDIT.md using real installed LiteLLM adapters and local fake HTTP upstreams. Add Gemini to all three ingress paths, streaming and non-streaming, so the matrix covers all twelve ingress/upstream combinations in both modes. Do not replace the adapter call with a stub for these integration tests.

Add focused contracts for multi-turn tool calls and tool results, fragmented streamed tool arguments, reasoning settings/content where supported, image input, usage/cache/reasoning counters, and relevant provider usage extensions. Cover actual upstream 4xx/5xx, rate-limit headers, first-event failures, midstream errors, and cancellation. Fix the misleading upstream-error test that currently exercises only local model lookup.

Keep fixtures faithful to documented provider schemas and tests deterministic/offline. Assert outgoing requests, response structures, SSE state/order, and termination. Preserve prior adversarial security tests. If a test reveals a small correctness bug, fix it; report substantial new feature work separately. Update README coverage claims to match the resulting suite.
```

## 16 — Make installation, dependencies, and CI reproducible

```text
Follow the working rules in AI_FIX_PROMPTS.md. Implement prompt 16 only.

Improve release reliability around the internal LiteLLM adapter imports. Inspect the supported Python policy, packaging, requirements.txt, and private adapter usage. Establish a reproducible tested dependency set and a documented upgrade procedure; avoid an unexplained broad upgrade or a dependency-tool migration.

Provide clear startup errors for incompatible adapter versions. Consolidate dependency declarations or verify that they remain consistent. Add a CI configuration suitable for the intended repository host, using supported Python versions and local tests only. Include package installation, entry-point/version checks, the full suite, and scoped lint/type checks with an achievable baseline.

Validate a clean temporary installation and the default configuration-loading path without reading real credentials or starting paid inference. Document how contributors reproduce checks and intentionally test a dependency upgrade. Do not require every future LiteLLM 1.x release to work merely because it shares a major version. Broad module refactoring is separate.
```

## Optional follow-up 17 — Unify retry behavior

```text
Follow the working rules in AI_FIX_PROMPTS.md. Implement optional prompt 17 only.

Make retry policy explicit and consistent across direct httpx forwarding and LiteLLM translation. Inspect actual adapter defaults first. Define configurable attempt limits, retryable failures, Retry-After handling, backoff, and an overall request deadline. Prevent retry amplification across nested layers.

Generation requests may have been accepted before a timeout; account for duplicate work and billing. Never replay a request after downstream output has begun. Preserve useful upstream rate-limit and retry information in client errors, with ingress-appropriate formatting. Keep client request data unable to override retry policy.

Test 429/503 recovery, Retry-After, nonretryable 4xx, connection failures, ambiguous post-send failures, exhausted deadlines, and streaming failures using controlled local upstreams and a fake clock. Document the policy and its duplication limits. Do not add model/provider fallback as part of retries.
```

## Optional follow-up 18 — Persist last-known-good discovery safely

```text
Follow the working rules in AI_FIX_PROMPTS.md. Implement optional prompt 18 only, after registry and background refresh fixes.

Add an optional versioned on-disk cache for last-known-good model discovery so the router can restart while catalogs or local servers are temporarily unavailable. Cache availability and metadata, not credentials or private request content. Reconstruct providers only from current trusted configuration.

Define cache identity, expiration, stale-route retention, configuration-change invalidation, collision precedence, and behavior for corrupt/incompatible files. Use atomic writes and appropriate file permissions. Cached routes must not revive disabled models or override current endpoint/credential choices. Expose cached/stale provenance to diagnostics and model metadata.

Test offline restart, corruption, schema changes, expired entries, changed server configuration, removed/disabled models, and recovery to live discovery. Make the feature opt-in and document its limits. Avoid persisting credentials embedded in URLs or headers.
```

## Optional follow-up 19 — Refactor around tested interfaces

```text
Follow the working rules in AI_FIX_PROMPTS.md. Implement optional prompt 19 only, after correctness fixes and regression coverage.

Split the implementation into a small package with explicit boundaries for configuration, provider transports/adapters, discovery/registry, HTTP handling, and CLI. Isolate LiteLLM private APIs behind one adapter interface. Keep the model-router entry point and existing documented python model_router.py invocation compatible through a thin shim.

Make this a behavior-preserving refactor. Retain routing precedence, credential ownership, metadata shape, errors, streaming, and configuration semantics established by earlier fixes. Avoid unrelated features, an automatic framework rewrite, or new runtime dependencies without a concrete need.

Update package metadata and imports. Verify editable and clean package installation, entry points, and the full suite. Add only tests needed to cover packaging/import boundaries; do not replace existing behavioral assertions. Explain the final module responsibilities and any unavoidable compatibility change.
```

## Suggested execution order

- **Immediate fixes:** 01–06. Prompt 05 can be handled before 04 if provider routing is the immediate blocker.
- **Reliable discovery and model controls:** 07–11, with 08 before 09 and 09 before 10.
- **Protocol and release quality:** 12–16. Prompt 13's Python fix can be done earlier if installation is blocked.
- **Optional improvements:** 17–19. Refactor last so functional changes remain reviewable.

Prompts are scoped to reduce accidental expansion, not to guarantee independent edits. Separate AI sessions should read the latest files and previous completion notes before making changes.
