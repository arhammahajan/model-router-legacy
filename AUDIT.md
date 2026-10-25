# Project audit — 2026-10-25

The router has a useful local-gateway design and a working translation core. Its highest-priority weaknesses are the boundary between untrusted request data and LiteLLM configuration, the bundled OpenCode Go endpoint rules, and discovery failure handling. Capability metadata needs a more explicit contract before clients can reliably configure context, reasoning, and tools from model discovery.

This audit changes documentation only. Application code and configuration were not changed.

## Scope and validation

- Read the 2,622-line implementation, 1,614-line test module, README, example configuration, deployed configuration, and packaging files.
- Ran `.venv/bin/python -m pytest`: **53 passed, 4 warnings, 32.08 seconds**.
- Environment: Python 3.14.8, LiteLLM 1.106.4, httpx 0.28.1, OpenAI Python 2.54.0.
- Ran additional isolated probes using fake credentials and local upstreams. These were audit probes, not additions to the committed test suite.
- Verified basic text generation through Gemini for all three ingresses, both streaming and non-streaming: all six combinations passed against a fake Gemini HTTP endpoint using the real LiteLLM adapters.
- Read the public OpenCode Go catalog: 38 models; every entry contained only `id`, `object`, `created`, and `owned_by`.
- Checked current upstream documentation. Authenticated cloud generation and a live llama.cpp instance were outside this validation; neither configured local service was listening during the audit.
- The supplied directory has no `.git` metadata or CI configuration. Findings refer to this filesystem snapshot.

Severity describes impact within the router's intended local deployment. Authentication limits which clients can exploit the request-boundary issue; it does not remove the issue for an authorized client.

## Findings

### 1. Critical: client-controlled LiteLLM fallbacks can send the provider key to another server

**Code:** [reserved parameters](model_router.py#L138), [request filtering](model_router.py#L2061), [Chat translation](model_router.py#L1797).

`_clean_request_params()` removes a short list of administrative names and passes the remaining request fields directly into LiteLLM. Fields including `fallbacks`, `model_list`, `custom_llm_provider`, and mock controls remain accessible to a client.

The installed LiteLLM accepts dictionary fallback entries that override `api_base` while inheriting the original API key. An isolated HTTP probe supplied a client fallback, forced the primary call to fail with a mock error, and received HTTP 200 from a second local upstream. The configured upstream received zero calls; the alternate upstream received the configured dummy key in `Authorization`.

**Change:** introduce a clear request boundary. Allow only protocol request parameters into LiteLLM's top-level arguments. Keep routing, endpoints, credentials, retry policies, callbacks, model lists, transport controls, and mock settings exclusively in trusted server configuration. Put explicitly supported vendor generation extensions through a separately validated mechanism. Apply the policy to every translated ingress, including nested data and merge behavior; merely adding `fallbacks` to the current denylist is insufficient. Add a regression test that asserts no request can change the destination or inherit credentials at a client-selected destination.

### 2. High: bundled OpenCode Go rules conflict with the provider's documented endpoints

**Code:** [catalog rules and providers](router.toml#L25).

The configuration has no OpenAI Chat upstream and sends all models except `claude-*` and `gemini-*` to Responses. Current [OpenCode Go documentation](https://opencode.ai/v2/docs/console/go/#endpoints) assigns GLM, Kimi, and DeepSeek models to Chat Completions, and some Qwen and MiniMax models to Messages. A model-name prefix is therefore insufficient to choose a dialect.

This is a confirmed configuration/documentation mismatch. Whether undocumented compatibility endpoints also accept these models was not tested with authenticated generation.

**Change:** add a Chat provider and maintain explicit model-to-endpoint metadata, with reviewed overrides for exceptions. Use provider metadata when available and a versioned local mapping otherwise. Validate known catalog entries against that mapping, surface unknown mappings, and avoid silently selecting Responses for every new model. Test the actual bundled rules and update the README examples with them.

### 3. High: translated Anthropic calls can use ambient credentials

**Code:** [LiteLLM credential arguments](model_router.py#L365).

For an Anthropic provider without a configured key, the router passes an empty string to LiteLLM. LiteLLM treats that as absent and falls back to `ANTHROPIC_API_KEY` and global credentials. This contradicts the documented credential-isolation promise.

An isolated probe set a fake ambient key and configured an uncredentialed local Anthropic upstream. Same-protocol forwarding sent no key; Chat-to-Anthropic translation sent the ambient key.

**Change:** define explicit credential modes, such as configured credentials and intentionally unauthenticated upstreams, and enforce them consistently in both transports. Ensure the adapter receives no usable ambient secret when unauthenticated mode is selected. Test all dialects with unrelated fake ambient provider keys present.

### 4. High: error redaction is bypassed for exceptions with a message attribute

**Code:** [sanitizer](model_router.py#L207), [error construction](model_router.py#L228).

`_error_payload()` uses `exc.message` directly and invokes `_error_message()` only when that attribute is absent or empty. Many provider exceptions have a message attribute. A synthetic exception containing a dummy bearer credential was redacted by `_error_message()` but returned verbatim by `_error_payload()`. The 4,000-character limit is bypassed too.

**Change:** route every error message through one sanitizer and length limit, including stream errors and diagnostic surfaces. Redact known configured secret values as well as common token patterns. Treat raw upstream error passthrough as a separate policy: preserving its bytes does not guarantee that upstream error text is safe.

### 5. High: malformed chunk sizes bypass the body-read limit

**Code:** [body reader](model_router.py#L2228), [chunk parser](model_router.py#L2254).

`int(size_text, 16)` accepts `-1`. The parser adds that negative size to the accumulated total and calls `read(-1)`, which reads until EOF. A probe with `max_body_bytes=64` consumed 10,004 bytes before reporting an error. On a socket the read can also remain blocked while a client continues sending.

Chunk delimiters and exact read lengths are not checked: a chunk using `xx` instead of the required trailing CRLF was accepted. Accepted sockets also have no explicit request-read timeout, and `ThreadingHTTPServer` creates unbounded handler threads.

**Change:** validate chunk-size syntax and non-negativity before any read; enforce exact lengths, CRLF, framing/header limits, and trailer bounds. Add header/body deadlines and bounded admission for connections and in-flight generations. Keep request-size and connection limits independent.

### 6. High: gzip passthrough forwards an incorrect content encoding

**Code:** [request header filter](model_router.py#L119), [body decompression](model_router.py#L2243), [passthrough serialization](model_router.py#L1754).

The router decompresses a client's gzip body, parses it, and sends newly serialized JSON upstream, but forwards the original `Content-Encoding: gzip` header. An HTTP probe confirmed that the fake upstream received plain JSON advertised as gzip. An upstream that honors the encoding will reject or misread it. The existing gzip test passes because its fake upstream ignores content encoding.

**Change:** remove the original content encoding after decoding, or deliberately re-encode the outgoing body and set matching headers. Add an upstream test double that validates encoding before parsing JSON. Review all forwarded representation headers whenever the body is transformed.

### 7. High: discovery failures after an initial success can cause repeated blocking refreshes

**Code:** [catalog staleness](model_router.py#L1221), [llama.cpp staleness](model_router.py#L1327), [resolution](model_router.py#L1692), [health](model_router.py#L2092).

After a successful discovery becomes stale, refresh eligibility is calculated from the last successful update. If a subsequent refresh fails, updating the last-attempt time does not postpone the next attempt. Probes confirmed that both discovery mechanisms remain eligible for an immediate retry in this state; llama.cpp has this behavior when no server succeeds.

Refreshes run synchronously during model listing and route resolution, with a lock held across network operations. Waiting callers can therefore queue and repeat failed network refreshes. `/health` also calls model listing, so a liveness probe can block on discovery.

**Change:** track independent success time, attempt time, and next-attempt deadline. Use bounded exponential backoff with jitter and retain the last usable catalog. Refresh in a background worker and publish a complete snapshot. Serve known routes and liveness from cached state. Make stale age and errors observable.

### 8. High: llama.cpp provider identities can collide during recovery

**Code:** [name allocation](model_router.py#L1147), [stale provider retention](model_router.py#L1468), [snapshot publication](model_router.py#L1615).

Provider names are allocated anew according to which servers are reachable. With two servers both named `local`, a probe reproduced this sequence: the first is initially down, the second receives identity `local`; the first recovers and the second becomes unreachable; the retained second provider overwrites the first provider under `local`. Both models then resolve to the second server.

Related concurrency risk: providers and routes are assigned separately, and requests resolve a route and look up its provider in separate steps. A refresh between those operations can mix generations of registry state.

**Change:** reserve stable identities for all configured servers before probing, validate duplicate explicit names, and retain identity independently of health. Publish one immutable registry snapshot and keep that snapshot, or a resolved route/provider pair, attached to each request. Add the recovery reproduction and concurrent refresh tests.

### 9. Medium: advertised local limits can be inaccurate, and model capabilities are incomplete

**Code:** [context detection](model_router.py#L1063), [capability detection](model_router.py#L1115), [discovered metadata](model_router.py#L1550), [public model shape](model_router.py#L1643).

Local discovery assigns `max_tokens = context_window`, although it has not discovered a distinct generation limit. The default 128,000 context fallback is presented like a measured value. Single-server context detection ignores `default_generation_settings.n_ctx` from `/props`. A probe reporting 4,096 there advertised the 128,000 fallback; `--ctx-size=8192` was also missed by argument parsing. `--fit-ctx` is a fitting constraint and should not be treated as a measured runtime context size.

OpenCode discovery imports only model IDs and nested `metadata`; it has no enrichment source. Local discovery extracts basic text/image flags, but not a comprehensive reasoning, tools, structured output, or sampling contract.

Current llama.cpp [server properties](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) include generation defaults, modalities, and template capabilities. Its [capability structure](https://github.com/ggml-org/llama.cpp/blob/master/common/jinja/caps.h) includes tool, parallel-tool, and reasoning-effort support. These signals should be consumed with server-version and per-model awareness.

**Change:** separate trained context, effective runtime context, input limits, and generation limits. Mark inferred values and fallbacks with provenance. Preserve unknown values as unknown. Add metadata overrides without replacing an entire discovered route. See the proposed capability contract below.

### 10. Medium: translated settings can disappear without feedback

**Code:** [default parameter policy](model_router.py#L403), [Chat translation](model_router.py#L1797), [Responses translation](model_router.py#L1833), [request headers](model_router.py#L2018).

`drop_params` defaults to true. In a fake-upstream probe, Chat-to-Anthropic returned HTTP 200 but removed a requested `reasoning_effort=high` for the synthetic model. This demonstrates silent loss for a model the adapter does not recognize; it does not establish that all named Anthropic models lose that parameter. Responses-to-Chat and Anthropic-to-Chat did forward mapped reasoning effort in separate probes.

Cross-protocol paths also discard inbound headers except the configured session extraction; an inbound `anthropic-beta` header did not reach an Anthropic upstream during Chat translation.

**Change:** validate settings against effective model capabilities. Prefer a clear unsupported-parameter error for meaningful controls such as reasoning, tools, and output schemas. Make lossy translation an explicit compatibility policy and expose diagnostics for dropped parameters. Forward a dialect-specific allowlist of necessary beta/version headers while retaining server ownership of credentials.

### 11. Medium: generated OpenCode session IDs are not stable across a conversation

**Code:** [session selection](model_router.py#L1030).

When neither the configured session header nor `prompt_cache_key` is present, every request receives a new UUID. OpenCode Go [asks clients to maintain a stable session ID per conversation](https://opencode.ai/v2/docs/console/go/#where-can-i-use-it). A present but ever-changing header does not preserve conversation affinity. Cross-protocol paths also drop other client-native session headers that Go may recognize.

**Change:** document an explicit conversation identifier, preserve supported native session identifiers through a controlled mapping, and support a configurable session extraction policy. Keep `prompt_cache_key` as an explicit fallback. Do not combine all users or all conversations into one global session.

### 12. Medium: synchronous stream iterator cleanup is incorrect

**Code:** [relay cleanup](model_router.py#L2321), [cleanup helper](model_router.py#L2423).

`_close_quietly(iterator)` attempts to call the iterator itself. Its `TypeError` is swallowed, so an iterator's `close()` is not invoked. A close-tracking iterator confirmed this. Some paths separately close the underlying transport through `plan.closer`, which reduces the impact but does not fulfill iterator cleanup.

**Change:** call the iterator's `close` method explicitly, use `aclose` for async iterators, and define ownership and cleanup once for every plan. Test cancellation, downstream disconnects, early upstream failures, empty streams, and completion. Ensure aborted requests stop consuming upstream generation where the adapter supports it.

### 13. Medium: Responses sequence numbers are not guaranteed to be monotonic

**Code:** [Responses SSE conversion](model_router.py#L991).

The converter preserves every integer supplied by upstream. Events numbered 2 and 1 were emitted as 2 and 1, contradicting the documented monotonic guarantee. Missing sequence values are filled, but duplicates or decreasing values are accepted. Error events generated by the generic stream-error path do not share this counter.

**Change:** own one sequence counter for every translated Responses stream and apply it consistently to synthesized events. Preserve same-protocol passthrough under its existing fidelity policy. Test sequence and event-state invariants, including terminal failures.

### 14. Medium: advertised Python 3.10 support does not cover the default configuration

**Code:** [Python requirement](pyproject.toml#L10), [TOML loading](model_router.py#L307).

`requires-python` permits 3.10, but TOML loading imports `tomllib`, introduced in Python 3.11, without a fallback dependency. The bundled default configuration is TOML. This was established from the import path; the installed interpreter was 3.14, so a separate 3.10 run was not performed.

**Change:** require Python 3.11+, or conditionally depend on `tomli` and use it on 3.10. Check installation and the default startup path in the supported Python matrix.

### 15. Medium: validation and operational status need clearer contracts

**Code:** [configuration expansion](model_router.py#L283), [configuration validation](model_router.py#L702), [request validation](model_router.py#L2217), [gzip decoding](model_router.py#L2432), [CLI check](model_router.py#L2590).

Several distinct cases need attention:

- `stream="false"` is treated as true and returns SSE; `stream=[]` is accepted as false. Validate boolean fields before routing.
- A translated Responses request without `input` returned 502 with an internal missing-argument message. Distinguish malformed requests and unsupported stateful features from upstream failures, and document the supported request combinations.
- Gzip decoding accepted a missing trailer and silently ignored a concatenated gzip member. Verify complete decoding and adopt an explicit policy for additional members/trailing bytes.
- Unknown configuration fields are ignored; string booleans are coerced with `bool()`, so `"false"` becomes true. Reject mistakes rather than silently altering behavior.
- Missing environment substitutions become empty strings. In an explicit auth configuration, this can leave no keys and disable auth. Support an explicit unauthenticated mode and distinguish it from missing configured credentials.
- `/health` exposes diagnostic upstream details without router authentication, always reports `status=ok`, and may refresh discovery. Split inexpensive liveness, readiness, and authenticated diagnostic detail.
- `--check` succeeds even when configured discovery fails. Separate configuration validation from a connectivity/readiness check with appropriate exit codes.

**Change:** introduce typed configuration and ingress validation, explicit unknown-field and missing-variable policies, and documented health/check semantics. Maintain passthrough extensibility through a controlled vendor extension policy.

## Proposed capability contract

Keep `/v1/models` compatible with existing clients and put a versioned, documented extension under `metadata.router`, with a model-detail endpoint if needed. Extra fields alone will not make every SDK or UI consume them; client integration must use the contract explicitly.

| Group | Suggested fields |
| --- | --- |
| Limits | Effective context, trained context if known, maximum input, maximum output; separate default generation length |
| Reasoning | Supported/unknown, toggle support, supported effort values when known, token-budget range when known, defaults, native parameter mapping |
| Modalities | Input and output modalities separately |
| Tools and output | Tool calls, parallel tool calls, JSON mode, JSON Schema, streaming; represent unknown explicitly |
| Sampling | Supported parameters and generation defaults; model-specific constraints where established |
| Protocols | Native endpoints, translated endpoints, and known lossy features |
| Provenance | Source for each derived group, retrieval time, server build/version where available, stale/fallback flags |

For llama.cpp, enrich from model listings and `/props`; use per-model queries in router mode and avoid loading unloaded models simply to obtain metadata. Exact supported effort values can depend on the template and require explicit overrides when not exposed. For Go, join the live availability list with an authoritative or reviewed metadata mapping and user overrides. Do not guess effort enums or context limits from model-name substrings.

Current llama.cpp also documents native Responses and Messages endpoints. An endpoint-aware route can preserve native features when a particular server build supports them; probe or configure support rather than assuming it for every discovered server.

## Test coverage assessment

The existing integration tests exercise real LiteLLM adapters, which is valuable. Their claims are broader than their coverage:

- The primary translation matrix has three upstream dialects, not four: [TARGETS](tests/test_model_router.py#L617) omits Gemini. The Gemini test replaces `litellm.completion` with a stub, so it verifies argument construction only. The six additional audit smoke checks passed and should become regression coverage.
- Existing tool tests check input schema translation. They do not cover a complete tool-call/result conversation or fragmented streamed tool arguments.
- The test named `test_streaming_upstream_error_becomes_ingress_error` actually requests an unknown local route; it does not simulate an upstream streaming failure.
- No tests cover the confirmed fallback credential escape, ambient keys, redaction bypass, negative chunks, backoff after a successful discovery, colliding recovery identities, or iterator cleanup. The gzip test does not check upstream encoding consistency.
- Reasoning controls, image input, usage and cache/reasoning counters, Go allowance fields, beta headers, stateful Responses behavior, rate-limit headers, timeouts, and concurrent refreshes need targeted tests.

Add focused adversarial tests first, then protocol contract tests. Assert structure, event order, final status, and preservation of semantically important fields, not just concatenated text. Keep paid real-provider smoke tests opt-in and separate from normal CI.

## Maintainability and operations improvements

- Isolate the internal LiteLLM imports and translation shims behind a small adapter module. `>=1.106.4,<2` is a broad range for private interfaces; record a reproducible dependency set and test a deliberate upgrade matrix.
- Split configuration, provider transports/adapters, discovery/registry, HTTP handling, and CLI into modules as fixes land. Preserve the simple installation and local deployment experience.
- Add CI for supported Python versions, package installation, the test suite, linting, and focused type checks. Consolidate the duplicated dependency declarations or verify they stay synchronized.
- Attach request IDs to client responses and logs. Record HTTP status, stream outcome, selected provider/model, translation mode, refresh duration, stale age, and dropped-parameter diagnostics without prompt/completion text.
- Fix stream outcome accounting: an error handled inside the relay can still be logged as the plan's HTTP 200 by `do_POST`.
- Configure retries explicitly across passthrough and LiteLLM paths; honor rate-limit information, bound attempts, and avoid replay once streamed output has been emitted.
- Support a persistent last-known-good discovery cache if startup during provider outages matters. Provide explicit stale-route expiration and collision policies.
- Make dynamic `provider/model` routing configurable. A disabled public route is not an access-control rule for the corresponding upstream model while dynamic routes remain allowed.
- The HTTP header filters should honor headers named by `Connection`, not just a static hop-by-hop list, and should avoid case-variant duplicates when merging configured headers.
- Correct documentation drift: the catalog user agent still identifies 0.2.0; `LLAMA_SERVER_URL` is documented as overriding the explicit server list, but is used only when no servers/ports are configured; the README overstates Gemini regression coverage and the monotonic sequence guarantee.

## Recommended implementation order

1. **Credential and request boundary:** block client transport/routing overrides, enforce explicit credential modes, fix redaction, chunk parsing, and gzip forwarding, add regression tests, and correct Go endpoint mapping.
2. **Discovery reliability:** stable server identities, one immutable registry snapshot, background refresh with backoff, cheap liveness, and bounded connection/request handling.
3. **Model capability support:** accurate limits and provenance, per-model reasoning/tool metadata, explicit parameter policies, and conversation-stable Go sessions.
4. **Protocol and release quality:** fix stream cleanup and sequencing, extend all-dialect/tool/reasoning/error coverage, reconcile Python support, add reproducible dependencies and CI, and refactor around the verified interfaces.

Acceptance should include both ordinary generation and the failure cases reproduced here. Passing the current 53 tests alone does not establish the credential boundary, provider mapping, or discovery recovery behavior.
