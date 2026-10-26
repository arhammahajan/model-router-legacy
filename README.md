# litellm-model-router

A lightweight, LiteLLM-backed HTTP router that accepts **OpenAI Chat Completions**, **OpenAI Responses**, and **Anthropic Messages** requests, and speaks any of those protocols (plus Gemini) upstream. Successful same-protocol responses are passed through untouched; cross-protocol calls are translated by LiteLLM. Local **llama.cpp** servers are discovered automatically — models, aliases, context size and capabilities included.

```
                 ingress                                     egress
   /v1/chat/completions ─┐                 ┌─ openai-chat      (pass-through)
   /v1/responses         ─┼─> model-router ─┼─ openai-responses (translated)
   /v1/messages          ─┘                 ├─ anthropic         (translated)
                                            └─ gemini            (translated)
```

## Translation matrix

Every ingress can reach every upstream dialect — all twelve combinations are covered, non-streaming and streaming. There is no Gemini ingress; Gemini is an upstream-only dialect.

| Ingress ↓ / Upstream → | OpenAI Chat | OpenAI Responses | Anthropic | Gemini |
| --- | --- | --- | --- | --- |
| **OpenAI Chat** `/v1/chat/completions` | pass-through | LiteLLM bridge | LiteLLM translate | LiteLLM translate |
| **OpenAI Responses** `/v1/responses` | LiteLLM bridge | pass-through | LiteLLM translate | LiteLLM translate |
| **Anthropic Messages** `/v1/messages` | LiteLLM adapter | LiteLLM adapter | pass-through | LiteLLM adapter |

Pass-through is byte-faithful for successful response bodies. HTTP error bodies use the redaction policy below. On the request side it rewrites only `model` and upstream auth/session headers, so provider-specific and beta fields (`cache_control`, `mcp_servers`, `reasoning`, `encrypted_content`, `previous_response_id`, custom vendor fields, …) survive a same-protocol hop. The one exception is discovered llama.cpp servers, where OpenAI reasoning effort is mapped to native thinking controls (see [Thinking control](#llama-cpp-thinking-control)).

Proxy forwarding removes hop-by-hop headers, including fields nominated by any `Connection` header. Client hop-only beta or session headers are not promoted into translated requests. Trusted upstream credentials and configured header defaults remain server-owned; outgoing representation headers describe the reserialized JSON body.

Translated requests use explicit per-protocol generation field allowlists. Unknown top-level fields return HTTP 400 instead of becoming LiteLLM keyword arguments. Destinations, credentials, provider selection, fallbacks, model lists, retries, callbacks, HTTP clients/transports and mock settings belong exclusively to trusted provider configuration. This applies to Chat, Responses and both Anthropic translation adapters. The router forwards tool definitions but disables LiteLLM's local MCP execution.

For translated vendor generation extensions, use a validated `extra_body` object. It accepts only `top_k` (nonnegative integer), `min_p` (finite number from 0 to 1), `repetition_penalty` (finite positive number), and `thinking`. Thinking accepts only `type`, `budget_tokens` (integer), `display` and `summary` (strings); the router uses LiteLLM's thinking mapping. For example:

```json
{"model":"my-model","messages":[{"role":"user","content":"Hello"}],"extra_body":{"min_p":0.05,"top_k":40}}
```

Extension support still depends on the target adapter/provider. Arbitrary nested extensions, `extra_headers`, and nested administrative settings are rejected. Tool parameters and JSON schemas remain ordinary data, so schema properties can have any name. Trusted `litellm_kwargs` take precedence when they overlap generation settings, including individual `extra_body` fields; the selected route owns `model`.

Compatibility changes: arbitrary top-level vendor fields and LiteLLM controls are no longer accepted on translated routes. Move supported generation extensions into `extra_body`, or use a same-protocol route for other native vendor fields. OpenAI `metadata` requires a same-protocol route because LiteLLM bridges can reinterpret it as execution context; translated Anthropic metadata accepts only a string `user_id`. Reasoning/configuration objects reject unrecognized members. Same-protocol JSON remains vendor-compatible and is sent directly to the configured URL without LiteLLM execution; these restrictions do not filter its body.

## Using it with OpenCode Go

The repository ships a ready `router.toml` for an OpenCode Go subscription:

```bash
.venv/bin/python model_router.py            # listens on 127.0.0.1:4000
```

Credentials are loaded **automatically** from `~/.config/ai/credentials.env` (or `$XDG_CONFIG_HOME/ai/credentials.env`) before the config is read. The file is expected to contain `OPENCODE_API_KEY=...`; nothing needs to be exported in your shell. Existing environment variables take precedence. Use `--env-file PATH` for another file or `--no-default-env` to disable this.

The router fetches the [public catalog](https://opencode.ai/inference/go/v1/models) at startup and every five minutes. The catalog supplies availability, but no endpoint compatibility information. `router.toml` contains an exact-ID mapping (revision 1, reviewed 2026-10-26) from the [official Go endpoint table](https://opencode.ai/v2/docs/console/go/#endpoints):

| Native upstream | Representative reviewed IDs |
| --- | --- |
| OpenAI Chat | `glm-5.3`, `kimi-k3`, `deepseek-v4.1-flash`, `mimo-v2.6-pro`, `minimax-m3.1-flash` |
| OpenAI Responses | `gpt-6-luna`, `gpt-5.6-luna`, `grok-4.7`, `muse-spark-1.3-contributor` |
| Anthropic Messages | `claude-haiku-5-5`, `minimax-m3`, `qwen3.8-max`, `qwen3.8-flash`, `qwen3.7-plus` |
| Gemini | `gemini-4-argon` |

These are exact IDs, not family-prefix rules: MiniMax M3.1 Flash uses Chat while M3 uses Messages. The checked-in snapshot contains 38 catalog IDs and 29 documented mappings. The other nine (`glm-5.1`, `grok-4.6`, `iceman-contributor-free`, `kimi-k2.6`, `minimax-m2.5`, `minimax-m2.7`, `omen-alpha`, `qwen3.6-plus`, `qwen3.7-max`) have unknown endpoint compatibility in this review. They and unmapped new IDs are omitted from `/v1/models`, reported in authenticated `/diagnostics` under `catalog.unmapped_models` and in refresh warnings, and return HTTP 404 with an actionable message when requested (`model_endpoint_unknown` on OpenAI ingresses). An ID absent from the catalog and explicit/local routes returns HTTP 404 (`model_not_found` on OpenAI ingresses).

Override an exact entry in `[catalog.model_providers]`, or add an explicit `[[models]]` route after reviewing the intended endpoint. A mapping references a trusted provider; URLs, credentials and auth headers stay in `[providers]`. Documented native routing does not establish compatibility with other endpoints. No authenticated cloud inference was performed; undocumented compatibility remains unknown.

```bash
curl http://127.0.0.1:4000/v1/models

curl http://127.0.0.1:4000/v1/chat/completions \
  -H 'Authorization: Bearer unused' \
  -H 'Content-Type: application/json' \
  -d '{"model":"claude-haiku-5-5","messages":[{"role":"user","content":"Hi"}]}'
```

OpenCode Go asks clients to supply a stable `x-opencode-session` per conversation. The router maps explicit and reviewed native conversation headers to it. The bundled configuration also enables `prompt_cache_key` as a fallback; without identity it omits the session header, so affinity is unavailable and Go may reject the request. See the conversation affinity policy below. The upstream subscription key never reaches clients.

`.venv/bin/python model_router.py --validate-config` validates configuration without networking. `--check-readiness` (also `--check`) additionally attempts discovery once, prints routes/readiness, and exits 0 when ready or 3 when unready.

## Install

```bash
python -m venv .venv
.venv/bin/python -m pip install pip==26.2.2
.venv/bin/python -m pip install -c constraints.txt -r requirements.txt

# or, to get the `model-router` command:
.venv/bin/python -m pip install -c constraints.txt .
```

Supports CPython **3.10–3.14**, with LiteLLM **1.106.4** and httpx **0.28.1**. Python 3.10 uses the conditional `tomli` dependency for TOML; newer supported Python versions use standard-library `tomllib`. Package metadata excludes Python 3.15+, matching the selected LiteLLM release. `requirements.txt` and package runtime declarations are checked for consistency. `constraints.txt` records the full runtime/test/build/check dependency set for Linux, including Python 3.10 compatibility pins. Use it for repeatable installations; installing the package alone pins direct dependencies but does not freeze transitives.

The router uses private `AnthropicAdapter`, `LiteLLMMessagesToResponsesAPIHandler` and `HTTPHandler` interfaces as well as LiteLLM's public Chat/Responses functions. LiteLLM's `mcp` extra is required because this release imports its gateway module before checking `skip_mcp_handler`, even when no MCP tools are used. The extra supplies import dependencies; the router still disables MCP execution. Startup rejects other LiteLLM versions before importing adapters and reports missing adapter imports/methods or constructor incompatibility with recovery instructions. **Compatibility change (prompt 16):** the previous `>=1.106.4,<2` range is replaced by an explicitly tested release with the required extra; sharing the 1.x major version is insufficient. See [CONTRIBUTING.md](CONTRIBUTING.md) for local CI reproduction, clean wheel installation and deliberate upgrade testing.

## Endpoints

| Endpoint | Ingress protocol |
| --- | --- |
| `POST /v1/chat/completions` (also `/chat/completions`) | OpenAI Chat Completions |
| `POST /v1/responses` (also `/responses`) | OpenAI Responses |
| `POST /v1/messages` (also `/messages`) | Anthropic Messages |
| `GET /v1/models` | OpenAI-style list, plus Anthropic-style `has_more`/`first_id`/`last_id` |
| `GET /livez` (also `/health`, `/healthz`) | Public, network-free liveness; HTTP 200 with `{"status":"ok"}` |
| `GET /readyz` | Public, cached readiness; HTTP 200 ready or HTTP 503 not ready |
| `GET /diagnostics` | Authenticated, sanitized provider inventory, readiness and discovery details |

## Client examples

Point any SDK's base URL at the router. The endpoint path selects the ingress dialect; the model name selects the upstream.

### OpenAI Chat Completions SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:4000/v1", api_key="unused")
resp = client.chat.completions.create(
    model="claude-haiku-5-5",           # routed to OpenCode's Anthropic endpoint
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=256,
)
print(resp.choices[0].message.content)
```

### OpenAI Responses SDK

```python
resp = client.responses.create(
    model="deepseek-v4.1-flash",        # translated to OpenCode's Chat endpoint
    input="Hello",
    max_output_tokens=256,
)
print(resp.output_text)
```

### Anthropic SDK

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://127.0.0.1:4000", api_key="unused")
msg = client.messages.create(
    model="gemini-4-argon",             # routed to OpenCode's Gemini endpoint
    max_tokens=256,                     # required for the Anthropic ingress
    messages=[{"role": "user", "content": "Hello"}],
)
print(msg.content[0].text)
```

### Streaming

Set `"stream": true` on any ingress. The router emits SSE in the **ingress** dialect:

- OpenAI Chat: `data: {...}\n\n` … `data: [DONE]\n\n`
- OpenAI Responses: `event: <type>\ndata: {...}\n\n` with a monotonic `sequence_number`
- Anthropic: `event: <type>\ndata: {...}\n\n`

Upstream errors that happen before the first event are returned as normal JSON errors with the ingress dialect's error shape; errors after streaming has begun are emitted as an SSE error event followed by the dialect's terminator.

Generated JSON/SSE errors, translated adapter error objects/events, discovery error status, and router/dependency diagnostics use the same sanitizer and a **4,000-character limit per diagnostic string**, applied after redaction. It removes configured router/provider/discovery keys and header values (including JSON/URL-escaped values), bearer tokens, recognizable token prefixes, and credential assignments such as `x-api-key: ...` regardless of the key's format. Both `exception.message` and exception stringification follow this path. Exception codes/types are sanitized too; raw exception tracebacks are not appended to logs. Arbitrary unknown strings with no credential context cannot reliably be identified as secrets.

**Raw upstream errors:** HTTP statuses 400–599, including same-protocol calls and requests with `stream=true`, return a sanitized ingress-shaped JSON error rather than the raw upstream body. The HTTP status is preserved; only sanitized Retry-After/rate-limit response headers are retained, including those carried on chained translated-adapter exceptions. Upstream Retry-After takes precedence over the router’s generic 503 hint. This deliberately changes error body/header fidelity. Successful same-protocol bodies, including opaque SSE bytes under HTTP 2xx, remain unchanged: embedded vendor error events or secret-looking generated content in such streams are consequently outside the redaction guarantee. Use a translated route when router-controlled error-event redaction is required.

## Configuration

TOML and JSON are supported (YAML too if PyYAML is installed). Every string value supports `${ENV_VAR}` expansion. Missing or whitespace-only variables and malformed substitutions are configuration errors; substitution is a single pass and never evaluates shell expressions. Lookup order when `--config` is omitted: `router.toml`, `router.json`, `config.toml`, `config.json`, `router.yaml`, `config.yaml`.

Malformed YAML is a configuration error (CLI exit code 2); parser diagnostics do not echo source lines that could contain credentials.

Configuration rejects unknown fields in router-owned tables, invalid types (including string booleans/numeric strings), nonfinite numbers, invalid ports, and nonpositive limits/timeouts. Refresh intervals may be zero for startup-only discovery. Header values must be strings. Model `metadata`, capability overrides (their own versioned schema), and trusted provider `litellm_kwargs` are deliberate extension points, not arbitrary router settings. `api_keys` and model `aliases` must be arrays of nonempty strings.

```toml
[server]
host = "127.0.0.1"
port = 4000
auth_mode = "none"                     # intentional local unauthenticated mode
# auth_mode = "required"               # use instead of "none" for client auth
# api_keys = ["${MODEL_ROUTER_API_KEY}"] # missing/empty => startup error
allow_dynamic_routes = false
max_body_bytes = 67108864
max_connections = 32
max_generations = 8
header_read_timeout = 10
body_read_timeout = 30
write_timeout = 30

[providers.anthropic]
dialect = "anthropic"
base_url = "https://api.anthropic.com"
api_key_env = "ANTHROPIC_API_KEY"
# headers = { "anthropic-beta" = "..." }
# timeout = 600
# auth_header = "x-api-key"             # bearer | x-api-key | api-key
# session_header = "x-opencode-session" # stable per-conversation identity
# session_prompt_cache_key = true       # explicitly enable body fallback
# session_missing = "omit"              # or "reject" to require identity
# litellm_kwargs = { ... }              # trusted server-side LiteLLM settings

[providers.openai]
dialect = "openai-responses"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"

[providers.local]
dialect = "openai-chat"
base_url = "http://127.0.0.1:11434/v1"
api_key = ""                            # intentional unauthenticated local upstream

[[models]]
name = "claude-sonnet"                  # public name clients send
provider = "anthropic"
model = "claude-sonnet-4-5-20250929"    # upstream model id
aliases = ["claude"]
metadata = { display_name = "Claude Sonnet 4.5", context_window = 200000 }
```

### Connection and generation limits

The built-in server uses bounded handler threads without extra services or runtime dependencies. `[server]` accepts these settings:

| Key | Default | Meaning |
| --- | --- | --- |
| `max_connections` | `32` | Maximum admitted client sockets/handler threads, including incomplete requests and streams. Admission occurs before creating a thread. |
| `max_generations` | `8` | Maximum requests resolving a route, calling an upstream, or sending a generation response. A stream holds its slot until completion/cleanup. |
| `header_read_timeout` | `10` | Seconds for the complete request line and headers. |
| `body_read_timeout` | `30` | Seconds to receive the complete encoded body, including chunk framing/trailers, after headers. |
| `write_timeout` | `30` | Seconds allowed for each downstream socket write, including response headers and SSE chunks. |

Counts must be positive integers (booleans and numeric strings are rejected); `max_generations` must not exceed `max_connections`. Timeouts must be finite positive numbers within Python's platform timeout range (`threading.TIMEOUT_MAX`). Header/body limits are **absolute read deadlines**, so a trickle of bytes does not extend them. They do not limit JSON processing, discovery, or upstream generation. Provider `timeout` (default 600 seconds) independently controls upstream operations; `write_timeout` applies only while writing to the client. Streams may pause between events, subject to the provider timeout, the overall `[retry].deadline`, and available capacity.

Both admission limits reject immediately; there is no application waiting queue. Connection overload attempts a minimal HTTP **503**, `Retry-After: 1`, and closes the socket without spawning a handler. That response is best effort using a nonblocking write: a peer that cannot receive it may observe only a disconnect. The kernel listen backlog is separately bounded to 16. Generation overload returns an ingress-shaped HTTP **503** (`generation_overloaded` for OpenAI dialects), `Retry-After: 1`, and closes the connection. Validation precedes generation admission; admission precedes route resolution and never waits under discovery locks. GET requests use connection capacity but no generation slot; reserve spare connections if model listing/health must remain accessible during generation load.

Incomplete request lines/headers and bodies return HTTP **408** and close, with `header_read_timeout` or `body_read_timeout` error codes for OpenAI dialects. Before headers identify an ingress, timeout errors use the OpenAI error shape. A downstream write timeout or disconnect closes the stream/connection and releases its generation slot after cleanup; an error cannot reliably be delivered to a stalled reader. Validation errors and upstream exceptions/timeouts also release capacity. All responses close the connection, as before.

Shutdown stops admission, closes client sockets, and clears reservations without waiting for client read/write deadlines. Synchronous upstream calls cannot be forcibly interrupted safely: a disconnect is observed on the next downstream write, and an adapter already waiting on its upstream may run until its provider timeout or next event. **Its slot stays reserved until it unwinds during normal operation**, preventing cancellation storms from exceeding upstream concurrency. During shutdown such already-running calls can finish in daemon handlers; shutdown does not wait for them or promise to cancel remote inference. The overall `[retry].deadline` also bounds upstream stream consumption; there is no background cancellation worker.

Compatibility change (prompt 04): previously unbounded connections/generations now use these defaults. Increase the limits or inbound read deadlines explicitly for larger local workloads/uploads; increase `write_timeout` for slow receiving clients. These admission settings do not shorten upstream timeouts; the separate `[retry].deadline` caps the generation budget.

### Generation retries and deadlines

One router-owned HTTP transport applies the same policy to direct forwarding and all LiteLLM translations. The compatibility default is **one attempt**, including the initial send, so retries require an explicit operator choice. Discovery keeps its separate scheduling policy.

```toml
[retry]
max_attempts = 1        # 1..20; set 3 to enable up to two replays
statuses = [429, 503]   # [] disables status retries; only 429 or HTTP 5xx allowed
connect_errors = true  # ConnectError / ConnectTimeout before sending HTTP
backoff_base = 0.5      # seconds
backoff_max = 8.0       # seconds
deadline = 600.0       # total seconds from dispatch, including streaming
```

For retry number `n` starting at 1, the exponential cap is `min(backoff_max, backoff_base * 2**(n-1))`; sleep is uniformly jittered between half that cap and the cap. A valid `Retry-After` (nonnegative integer seconds or timezone-bearing HTTP date) is a **minimum wait**, even beyond `backoff_max`. Past dates impose no extra delay; invalid hints use backoff. If the wait cannot fit the remaining deadline, the router stops immediately with HTTP 504 and `request_deadline_exceeded`, retaining the upstream retry/rate-limit headers. Otherwise it checks the deadline again after sleeping. Backoff values and deadline must be finite positive numbers, with base no greater than max. Boolean/numeric-string coercions and unknown retry fields are rejected.

By default, only explicit **429/503 response headers** and **connection establishment failures** are eligible. Other HTTP errors, including nonretryable 4xx, are returned immediately; operators may select other 5xx in `statuses` after considering their acceptance/billing semantics. Error responses are closed before waiting without consuming their bodies. The last exhausted HTTP error keeps its status, sanitized message and useful `Retry-After`/rate-limit headers in the ingress's JSON error shape. The router never switches model, provider, key or destination as part of a retry.

**Ambiguous outcomes are never retried.** Read/write timeouts, read/write connection failures, truncated HTTP responses, failures while parsing generation output and SSE failures may follow an accepted request. Transport failures after sending may have begun return `upstream_outcome_unknown` (HTTP 504 for timeouts, 502 otherwise), with an explicit acceptance/billing notice. After the router returns any final upstream response, including HTTP 200 headers before the first event, that request's transport refuses further sends. This is stricter than merely waiting for downstream output: even a first-event failure cannot restart a generation. Adapter-synthesized framing also counts as output: if a lazy source opens HTTP afterward, even connection failures or 429/503 cannot trigger a replay. Once downstream headers/output exist, failure uses the ingress's terminal SSE error format, retaining HTTP 200; there is no replay. Retry/rate-limit hints known before the first downstream event are forwarded as HTTP headers; later stream errors cannot change committed headers.

The monotonic deadline covers translation preparation, all attempts, connection/TLS setup, request writes, upstream header/body reads, backoff and stream consumption. Each socket operation uses the smaller of its normal timeout and the remaining budget; stream activity does not reset it. Provider `timeout` remains a per-operation limit (connect is also capped at 20 seconds). Buffered adapter events and async pulls are checked against the same deadline. Inbound header/body deadlines and downstream per-write timeouts remain separate. Synchronous DNS resolution and adapter CPU work cannot be forcibly interrupted safely; results are discarded when control returns after the deadline. A blocked downstream write can run until its separate `write_timeout`, then cleanup releases resources. The deadline does not promise cancellation of remote inference.

The installed defaults were inspected before this policy was added: httpx 0.28.1 transport retries default to zero; LiteLLM 1.106.4's OpenAI Chat adapter can create an SDK client with two retries, and its completion/Responses wrappers also honor global or call-level retry settings. Native sync Anthropic/Gemini and Responses adapters use `HTTPHandler`; the generic adapter layer also has a provider-specific error-repair retry hook. See the official [httpx transport documentation](https://www.python-httpx.org/advanced/transports/), [OpenAI SDK retry documentation](https://github.com/openai/openai-python#retries), and [LiteLLM retry documentation](https://docs.litellm.ai/docs/completion/reliable_completions). The router now supplies request-owned clients, explicitly sets SDK/LiteLLM retries to zero, clears fallback/retry-policy call settings and enforces the shared budget at the HTTP boundary. Nonempty global LiteLLM retry/fallback overrides fail closed for translations; no process-global settings are swapped around concurrent calls. Trusted `litellm_kwargs` can no longer replace retry/transport ownership. Translation requires the reviewed `openai`, `anthropic`, or `gemini` prefix matching its dialect; an unreviewed prefix or conflicting `custom_llm_provider` fails closed with `retry_adapter_unsupported`. Native forwarding still accepts any provider prefix metadata. Client JSON cannot change this policy: translated administrative controls are rejected, while native vendor fields remain ordinary request data and never configure the router transport.

Compatibility changes (prompt 17): translated requests no longer inherit hidden SDK retries; ambiguous transport failures consistently use HTTP 502/504 and `upstream_outcome_unknown` instead of adapter-specific 500/503 wrapping; long streams now default to a 600-second overall budget, independently configurable through `retry.deadline`. Generation connection pools are request-owned and closed on completion/cancellation; discovery retains App's pool. Environment proxy/CA selection remains supported, with proxy connections also going through the retry/deadline boundary. The deadline implementation wraps the pinned httpx/httpcore network backend; dependency upgrades must retain its socket/TLS and stream tests.

**Duplication limit:** no retry policy supplies exactly-once generation or billing. An upstream or proxy may return 429/503 after scheduling work, lose a response, keep working after a connection closes, or retry internally. Clients and intervening proxies may also retry independently. The router does not invent an idempotency key or claim provider deduplication across translated protocols. Keep `max_attempts = 1` when duplication is unacceptable and coordinate client/proxy retry settings. Enabling retries accepts this residual risk; disabling ambiguous replays only reduces it.

### Providers

| Key | Meaning |
| --- | --- |
| `dialect` | `openai-chat`, `openai-responses`, `anthropic`, or `gemini` — what the endpoint speaks |
| `base_url` | OpenAI dialects: include `/v1`. Anthropic: use the origin or a prefix (`/v1/messages` is appended). Gemini: include `/v1beta`. A base already ending in the provider's final path is used as-is. |
| `api_key` | Literal key |
| `api_key_env` | Read the key from this environment variable (error if unset/empty) |
| `api_key_file` | Read a key from a file (whitespace stripped; error if unreadable/empty) |
| `headers` | Extra upstream headers, e.g. beta flags |
| `auth_header` | Force `bearer`, `x-api-key`, or `api-key` auth. Empty uses the provider default. |
| `session_header` | Upstream conversation identity header, e.g. `"x-opencode-session"`. Empty (default) disables extraction. Must be separate from credential, protocol and transport headers. |
| `session_source_headers` | Ordered subset of reviewed native headers; defaults to the list below. `[]` accepts only the explicit upstream header and optional body fallback. |
| `session_prompt_cache_key` | Explicitly use top-level `prompt_cache_key` as the last identity fallback (default `false`; enabled for bundled Go providers). |
| `session_missing` | `"omit"` (default) sends no session header without identity; `"reject"` returns HTTP 400 before dispatch. Requires `session_header`. |
| `timeout` | Per-operation upstream timeout in seconds (default 600), clamped to remaining `[retry].deadline` |
| `litellm_provider` | LiteLLM provider prefix (default `anthropic`, `gemini`, or `openai` by dialect) |
| `litellm_kwargs` | Trusted adapter/generation arguments; credentials, client/transport, and retries remain router-owned. Configure retries in `[retry]`. |
| `compatibility` | Translation policy: `"strict"` (default) or explicit `"lossy"`; see below |
| `drop_params` | Legacy alias: `false` selects strict, `true` selects diagnosed lossy mode; must agree with `compatibility` if both are set |

Upstream credential selection has two behaviors:

- A nonempty `api_key`, `api_key_env`, or `api_key_file` selects that configured key. Surrounding whitespace is stripped. The top-level provider key owns LiteLLM's `api_key` even if `litellm_kwargs` repeats it; the selected auth header also wins case-insensitive header conflicts. Other explicitly configured headers remain trusted. Adapters may add their native credential header as well as an `auth_header` override, using the same selected key.
- An omitted credential or a literal empty/whitespace-only `api_key` intentionally selects the public placeholder **`model-router-no-auth`**. Explicit `api_key_env`/`api_key_file` references must resolve to a nonempty key. Both passthrough and translation send the placeholder using the provider's native auth mechanism (or configured `auth_header`). An unauthenticated server must accept/ignore this placeholder; servers that reject any auth header are not supported. No environment or LiteLLM global provider key is used as a fallback.

This avoids [LiteLLM's environment/global credential sources](https://docs.litellm.ai/docs/set_keys) without clearing or swapping process-wide environment variables during requests. Only named `api_key_env` settings opt into environment credentials. LiteLLM's `os.environ/NAME` key indirection is rejected; use `api_key_env` instead. Global LiteLLM `proxy_auth`, `client_session`, or `aclient_session`, which can inject auth after argument merging or through client defaults, cause translated calls to fail closed without modifying those globals. Client/transport entries in `litellm_kwargs` are replaced by the router-owned generation transport. Client-visible router auth is separate and never forwarded.

Compatibility change (prompt 02): keyless passthrough calls now send the same placeholder as translation, replacing absent auth headers or the old `not-needed` placeholder. Existing nonempty configured keys continue to work. Raw HTTP error bodies are replaced as described above. These isolation regressions are tested against installed LiteLLM 1.106.4 with local upstreams and dummy environment/global keys for Anthropic, OpenAI Chat/Responses, and Gemini.

### Conversation affinity

Create a distinct opaque ID when a conversation starts and reuse it on **every turn**, including tool results, retries and streaming requests. For every ingress (`/v1/chat/completions`, `/v1/responses`, `/v1/messages`), supply it as an HTTP header, for example `x-opencode-session: conversation-123`. SDKs can set it through their per-call extra-header option. Different conversations must use different values, even when they share an API key or start with identical prompts. This is routing/cache affinity metadata, not authentication or server-side conversation storage: clients still supply the required conversation history.

When `session_header` is configured, selection is case-insensitive and follows this priority:

1. The inbound header named by `session_header` always wins.
2. The first present header in `session_source_headers` wins. The default order is `x-claude-code-session-id`, `session-id`, `session_id`, `thread-id`, `conversation_id`, `x-session-id`. Configure a reordered subset to change native priority; arbitrary header names are rejected.
3. If `session_prompt_cache_key = true`, top-level JSON `prompt_cache_key` is the last fallback.
4. Otherwise, `session_missing = "omit"` sends no identity, or `"reject"` returns HTTP 400 with instructions to supply the header (OpenAI code `missing_session_id`; Anthropic type `invalid_request_error`).

Native names were reviewed on 2026-10-26 against [Claude Code's gateway compatibility guide](https://code.claude.com/docs/en/llm-gateway-protocol#request-headers) (`x-claude-code-session-id`) and [LiteLLM's native session header documentation](https://docs.litellm.ai/docs/proxy/cache_aware_routing#3-send-a-stable-session-id) (Codex `session-id`/`thread-id`, legacy `session_id`/`conversation_id`, and OpenCode `x-session-id`). These describe client identities; they are not universal OpenAI/Anthropic API requirements. Parent, agent, request and trace IDs are deliberately excluded. The router maps the chosen value into its configured upstream header itself; it does not depend on LiteLLM's proxy detection or User-Agent rules. [OpenCode Go documents stable per-conversation identity and recognizes native Claude Code/Codex sessions](https://opencode.ai/v2/docs/console/go/#where-can-i-use-it). Client versions or intermediary proxies may omit native headers; use the explicit header in that case.

```toml
[providers.go]
dialect = "openai-responses"
base_url = "https://opencode.ai/inference/go/openai/v1"
api_key_env = "OPENCODE_API_KEY"
session_header = "x-opencode-session"
session_source_headers = ["session-id", "session_id", "thread-id", "x-claude-code-session-id", "x-session-id"]
session_prompt_cache_key = true
session_missing = "reject"
```

The selected identifier is forwarded unchanged in the target header on both native and translated calls to every upstream dialect, including Gemini. With extraction enabled, reviewed native session headers (including disabled sources) and any inbound `x-opencode-session` are consumed; only the selected target header is sent, so conflicting native values cannot change the upstream's choice. Different source headers may disagree: the priority above resolves that conflict. Repeated occurrences of the same identity header, including identical duplicates or case variants, return HTTP 400. All present reviewed native values and the explicit target are validated, including lower-priority and disabled sources. Values must be 1–256 ASCII characters, start with a letter/digit, and contain only letters, digits, `.`, `_`, `:`, `/`, `-`. Empty, whitespace, control characters, non-ASCII, comma-joined and oversized values return HTTP 400 (OpenAI code `invalid_session_id`; Anthropic type `invalid_request_error`); error messages do not echo values. An enabled, present `prompt_cache_key` must satisfy the same validation, even when a header wins. Static session headers in provider `headers` or `litellm_kwargs.headers`/`extra_headers` are configuration errors, preventing a global identity from combining conversations. Credential headers cannot be selected as identity sources or targets; authentication remains server-owned.

The body fallback is useful only when the client keeps `prompt_cache_key` stable **per conversation**. Keys shared across unrelated chats combine their affinity; keys changed each turn lose affinity. Choose an explicit conversation header if the client's cache key has different semantics. On translated calls with fallback enabled, the router consumes this field as affinity metadata before protocol validation, including on Anthropic ingress; it is not also sent as a generation/cache control. Native calls retain the original JSON field. With fallback disabled, existing dialect-specific generation validation applies to the field. With `session_header = ""`, extraction is disabled, native passthrough retains its ordinary filtered-header behavior, and translation does not infer an identity.

Compatibility change (prompt 11): no per-request UUID is generated. An absent identifier establishes no affinity; Go may require identity, so use `session_missing = "reject"` for a local actionable error. The router cannot reconstruct a stable identity when clients omit it. It does not use a global session, provider/API keys, mutable prompt hashes, or in-memory guesses to combine turns. Existing configurations relying on automatic `prompt_cache_key` selection must set `session_prompt_cache_key = true`. Static session headers must move to client requests. The bundled Go providers explicitly enable the cache fallback and use the default omit behavior.

### Translation compatibility and protocol headers

Translated calls default to **strict** compatibility. Before dispatch, the router checks the reviewed bridge surface, LiteLLM's target Chat parameter mappings where applicable, and the route's effective `metadata.router` facts (including trusted capability overrides). Known unsupported reasoning, tools, structured output and sampling controls, known invalid effort values/budget bounds, unmapped native fields, and known approximations return HTTP **400** (`translation_loss` on OpenAI ingresses). The error identifies field paths and suggests removing the controls, selecting a native route, or changing the operator policy. Unknown capability facts and unknown effort enums do **not** mean unsupported: a control with an available mapping is still sent for the upstream to validate. An unknown model with no adapter mapping can still fail translation; that is a mapping limitation, not a claim that the model lacks the capability. Stale facts retain their reported values; use a capability override to correct or clear them.

For a provider that explicitly accepts loss:

```toml
[providers.local]
dialect = "openai-chat"
base_url = "http://127.0.0.1:8080/v1"
compatibility = "lossy"
```

Lossy mode removes identified unsupported/unmapped controls and reports each loss in the **`x-model-router-loss` response header**, a JSON array of `{field, reason, action}` objects. It is present on JSON and SSE responses when loss occurs, before stream events. No request values are included. For example, `[{"field":"reasoning","reason":"model capability is unsupported","action":"dropped"}]`. Clients that need fidelity should treat any diagnostic as a failed compatibility check. A native Anthropic token budget becomes an effort bucket on the Messages-to-OpenAI bridges: strict mode rejects that approximation; lossy mode keeps the mapping with `action: "approximated"`. Client JSON cannot select compatibility or LiteLLM administrative settings, even in lossy mode. Unknown request fields and unsafe extensions remain errors under the existing security boundary.

Effort-to-effort Chat/Responses mappings, effort-to-Anthropic thinking budgets on recognized models, function tools, and Schema output between OpenAI dialects and from Messages have local upstream regression coverage. JSON output targeting Anthropic is conservatively rejected/dropped on bridges because an adapter's synthetic-tool emulation does not guarantee native constrained output. Native state, reasoning summaries, adaptive thinking, provider-hosted tools, stop controls targeting Responses, and other unmapped fields may require a native route. This is a bounded compatibility contract, not a promise to translate every native field without loss. Same-protocol bodies keep their existing vendor-field passthrough behavior and bypass translation capability checks.

LiteLLM's own parameter dropping is disabled in both modes, following its [documented strict behavior](https://docs.litellm.ai/docs/completion/drop_params). Additional adapter refusals remain actionable errors even in lossy mode; the router does not retry a possibly accepted generation to recover from a parameter error. Global `litellm.drop_params=true` and nonempty trusted `additional_drop_params` fail closed on translated calls because they bypass diagnostics. Trusted generation overrides retain their existing precedence at call merging; compatibility preflight validates client controls, not arbitrary operator adapter customizations.

Translated requests forward only these **destination-dialect** client opt-ins:

| Destination | Allowed client headers |
| --- | --- |
| Anthropic Messages | `anthropic-beta`, `anthropic-version` |
| OpenAI Chat / Responses | `openai-beta` |
| Gemini | none (the API version is part of the configured URL) |

Headers are matched/merged case-insensitively. Precedence is native version default < inbound allowlisted header < trusted `litellm_kwargs.headers` < trusted `litellm_kwargs.extra_headers` < provider `headers` < configured authentication. Values replace earlier values; beta lists are not concatenated. Anthropic defaults to `2023-06-01` when no version is supplied, preserving existing behavior; see its [version header](https://platform.claude.com/docs/en/api/versioning) and [beta opt-ins](https://platform.claude.com/docs/en/api/beta-headers). A beta header enables only the upstream feature; it does not give an unmapped request field a bridge mapping. Client credentials, cookies, destinations, transport/representation headers and arbitrary headers cannot enter translated calls. Conversation identity follows the separate extraction policy above. Same-protocol forwarding keeps its broader filtered header policy; configured headers now also replace inbound headers case-insensitively. Authentication and serialized JSON framing remain server-owned on both paths.

**Compatibility change (prompt 10):** omitted `drop_params` used to enable silent dropping; the default is now strict. Explicit legacy `drop_params=true` selects lossy mode with diagnostics rather than enabling LiteLLM dropping. Review affected clients and configure loss acceptance deliberately.

### Models

`name` is the public route; `aliases` are additional public names. `dialect` overrides the provider default if one endpoint exposes both protocols. `enabled = false` hides a route and makes requests for it resolve as not found.

With `server.allow_dynamic_routes = true`, a request naming an unlisted model as `provider/model` forwards dynamically using the provider's default dialect. The compatibility default is `true`; bundled configurations explicitly set `false` to allow only registered explicit/catalog/discovered routes:

```json
{"model": "local/qwen2.5-coder:7b", "messages": [...]}
```

Disabled aliases alone are **not a model access-control policy**. Another alias, a catalog/discovered route, or a dynamic provider/model ID can reach the same upstream model. To restrict the inventory, disable dynamic routing and review every explicit and discovery/catalog source. This is a global inventory policy, not a per-client authorization system.

Ingress validates known field types before dispatch, including strict `stream` booleans, generation numbers, arrays/objects, message envelopes and tool lists. Invalid input returns HTTP 400 in the ingress error format, before contacting an upstream. Native routes preserve unknown vendor JSON fields and nested vendor extensions; translated routes retain the closed generation boundary and typed `extra_body` extensions described above. Validation does not claim to implement every provider's full schema.

Translated Responses requires a non-null string or array `input` containing the complete conversation. `instructions` can accompany it. `previous_response_id`, `conversation`, stored `prompt` references, and `background=true` require a native Responses route and return HTTP 400 (`unsupported_responses_state`) on translation. Missing translated `input` returns 400 (`invalid_request`) instead of an internal adapter error/502. Native Responses permits input-omitting combinations and passes state/prompt behavior to the upstream, after type checks; `previous_response_id` and `conversation` cannot be combined. See the [official Responses request contract](https://developers.openai.com/api/reference/resources/responses/methods/create). The router does not store conversation state.

### Dynamic catalogs

A catalog fetches model IDs from an OpenAI-style `/v1/models` endpoint and registers mapped IDs under their upstream names. For example, using the providers from `router.toml`:

```toml
[catalog]
url = "https://opencode.ai/inference/go/v1/models"
refresh_seconds = 300          # 0 = startup only
headers = { "User-Agent" = "litellm-model-router/0.3.0" }

# Exact entries win over optional regex rules. See router.toml for the full map.
[catalog.model_providers]
"deepseek-v4.1-flash" = "opencode-chat"
"gpt-6-luna" = "opencode-openai"
"claude-haiku-5-5" = "opencode-anthropic"
"gemini-4-argon" = "opencode-gemini"
"minimax-m3" = "opencode-anthropic"
"minimax-m3.1-flash" = "opencode-chat"
"qwen3.8-max" = "opencode-anthropic"
```

`catalog.model_providers` maps exact model IDs to configured provider names, which select the dialect and trusted endpoint. Entries do not create routes unless the ID is present in the fetched catalog. Explicit `[[models]]` routes and aliases retain precedence and can also expose models absent from the catalog. An explicit route for the same public ID can override a catalog mapping:

```toml
[[models]]
name = "deepseek-v4.1-flash"
provider = "opencode-chat"
model = "deepseek-v4.1-flash"
```

Existing `[[catalog.rules]]` regex rules remain supported as an operator opt-in: exact mappings win, then the first matching rule. At least one exact mapping or rule is required. A broad prefix or catch-all rule explicitly accepts responsibility for routing future matching IDs; the bundled Go configuration has neither. Catalog data cannot choose a provider, URL or credential. Unmapped IDs without an explicit/local route are omitted, diagnosed and rejected as described above. When dynamic routing is enabled, `provider/model` requests are a manual endpoint choice using the named provider's dialect; they do not establish verified compatibility.

Catalog fetch/parse failures retain existing routes and are logged and retried in the background, subject to the refresh policy below. A successful empty or entirely unmapped catalog replaces the previous catalog routes with an empty inventory and still reports `loaded = true`.

### Local llama.cpp discovery

The router can probe running [llama.cpp](https://github.com/ggml-org/llama.cpp) servers and expose every model they report. This ports the discovery behavior of [pi-llama-cpp](https://github.com/gsanhueza/pi-llama-cpp): single-model, router (`--models-preset`), legacy (`ik_llama.cpp`) and [llama-swap](https://github.com/mostlygeek/llama-swap) servers are recognized.

A config needs explicit `[[models]]`, a `[catalog]`, a `[llama_cpp]`, or any combination. Explicit routes win when a name collides, then discovered llama.cpp models, then catalog models. Explicit aliases and disabled explicit names also reserve their names against discovery; an overridden catalog entry is not separately advertised in `/v1/models`.

```toml
[llama_cpp]
servers = ["http://127.0.0.1:9931"]   # URLs or { url, name, api_key, headers }
# ports = [9931, 8080]                 # convenience: localhost URLs
refresh_seconds = 30                   # 0 = startup only
timeout = 2                            # per-probe timeout
context_fallback = 128000              # labeled estimate when context is unavailable
prefix = ""                            # e.g. "llama/" to namespace model IDs
aliases = true                         # expose llama.cpp aliases as routes
api_key_env = "LLAMA_API_KEY"          # for servers started with --api-key
```

If both `servers` and `ports` are omitted, the router uses `LLAMA_SERVER_URL` (semicolon- or comma-separated URLs) and then `http://127.0.0.1:9931`, matching pi-llama-cpp. Servers started after the router are picked up on the next refresh. A discovered model is routed exactly like an explicit model on an `openai-chat` provider, so any ingress (Chat, Responses, Anthropic) can call it:

```bash
curl http://127.0.0.1:4000/v1/models

curl http://127.0.0.1:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-8B-Q4_K_M","messages":[{"role":"user","content":"Hi"}]}'
```

### llama.cpp thinking control

llama.cpp has no `reasoning_effort` parameter; it takes a token budget or a chat-template toggle. For discovered llama.cpp models the router maps the standard OpenAI effort onto those native fields, so any client can use ordinary thinking levels:

| `reasoning_effort` (Chat) / `reasoning.effort` (Responses) | Sent to llama.cpp |
| --- | --- |
| `none` | `chat_template_kwargs: {"enable_thinking": false}` |
| `minimal` / `low` / `medium` / `high` / `xhigh` | `thinking_budget_tokens`: 1024 / 2048 / 8192 / 16384 / 32768 |
| `max`, or any other value | nothing added; the server default applies |

The effort field itself is removed from the request. A client-supplied `thinking_budget_tokens` or `chat_template_kwargs.enable_thinking` always wins over the mapping. This applies to the Chat pass-through and to translated Responses requests; Anthropic `thinking` still requires a native route. Models report `reasoning.effort.supported` with kind `configured` (source `router.llama_thinking_map`): that states the control exists, not that a given model produces thinking, so `reasoning.supported` stays whatever the upstream reported.

Server provider identities are reserved for **all configured servers before probing**, including unreachable servers. Explicit `name` values must be unique and must not match a static `[providers]` name; duplicate names now fail configuration validation with the conflicting entries and a suggested fix. Omit `name` for a generated identity: the router normalizes the URL into `llama-…`, reserves all static and explicit names first, then resolves collisions with `-2`, `-3`, etc. in server configuration order. Reachability and probe completion order never change identities. Changing server order or configuration can change generated suffixes; use unique explicit names when identities must survive configuration edits.

Discovery reads:

| Endpoint | Used for |
| --- | --- |
| `/health` | reachability (the server is skipped when this fails) |
| `/v1/models` | model IDs, aliases, per-model limits, `status`, `architecture` |
| `/props?autoload=false` | server mode and single/legacy runtime context and generation defaults |
| `/props?model=<id>&autoload=false` | limits/defaults for each loaded or sleeping router model; optional on older servers |

Each discovered model has `metadata.limits`, separating these token counts:

| Field | Meaning |
| --- | --- |
| `context_window` | Effective runtime context when reported; otherwise a labeled launch estimate or fallback |
| `trained_context_window` | Model's trained context, independent of runtime capacity |
| `max_input_tokens` | Explicit input maximum, otherwise unknown |
| `max_output_tokens` | Explicit maximum generation length, otherwise unknown |
| `default_output_tokens` | Default generation length, independent of any maximum |

Each field is an object with `value` (integer or `null`), `source` (field/config path or `null`) and `kind` (`reported`, `override`, `inferred`, `fallback`, or `unknown`). For example, a single server reporting only `/props.default_generation_settings.n_ctx = 4096` yields `context_window = {"value": 4096, "source": "/props.default_generation_settings.n_ctx", "kind": "reported"}` and an unknown output maximum. Flat numeric copies also appear in `metadata`.

Precedence is explicit `llama_cpp.limit_overrides` first, then usable per-model reports, then launch-argument estimates, then the configured context fallback (default 128000). Within reports, runtime context prefers `/v1/models.meta.n_ctx`, then `/props.default_generation_settings.n_ctx`, then `/props.n_ctx`. Legacy mode prefers `max_model_len` before those sources. Global `/props` values apply only in single/legacy mode: router mode queries loaded/sleeping models individually, always with `autoload=false`; unloaded models are never loaded to collect metadata. Unsupported/failed per-model queries leave listing metadata and estimates available. llama-swap retains per-model `meta.n_ctx` and nested aliases, without applying its global properties to every model.

Trained context uses `meta.n_ctx_train`, router-mode `context_length`, then model-specific `/props.n_ctx_train`. Input/output maxima use only explicit `max_input_tokens` / `max_output_tokens` fields, preferring model `meta`, the model row, model `metadata`, then applicable `/props`. Input capacity is not calculated from context or by subtracting a generation budget. Generation defaults prefer `/props.default_generation_settings.params.n_predict`, then the older direct `n_predict`, then nested/direct default `max_tokens`, then launch `--predict` / `--n-predict` / `-n`. A default is never a hard maximum. Special defaults `-1` and `-2` have `value: null` with `special: "unlimited"` or `"until_context"`; zero is a valid no-generation default.

Launch `--ctx-size` / `-c` and generation flags accept split (`-c 8192`) and equals (`--ctx-size=8192`) forms; the last occurrence wins, including zero/auto/invalid values that clear an earlier estimate. Context estimates describe the requested size, which can differ from usable per-request capacity because of slots, fitting, or server-version behavior; use `kind: reported` for runtime-aware decisions. `--fit-ctx` / `-fitc` is recorded separately under `metadata.limits.constraints.fit_context_min` with `kind: constraint` and never establishes runtime context. Context zero/negative/auto, booleans, fractional/nonfinite numbers and malformed data cannot establish a positive limit. Model names are never used to infer limits.

Overrides enrich discovered limits without duplicating routing configuration. Keys match the exact upstream model ID **before** `prefix` or aliases and apply to that ID on all configured local servers:

```toml
[llama_cpp.limit_overrides."my-model-id"]
context_window = 8192
trained_context_window = 131072
max_input_tokens = 8000
max_output_tokens = 2048
default_output_tokens = 256
```

Omit fields to retain discovery. Values must be positive integers, except `default_output_tokens` also accepts zero; `context_fallback` itself must also be a positive integer. JSON configs can explicitly set a field to `null` to report it as unknown. Overrides and fallback sizes only change advertised metadata; they do not reconfigure or enforce limits on the upstream server.

**Compatibility change (prompt 08):** top-level `context_window` and `metadata.llama_cpp.n_ctx` remain aliases of the resolved context, including labeled estimates. Top-level/metadata `max_tokens` now mean **maximum output tokens** and are `null` unless explicitly known or overridden; they no longer equal context size automatically. Explicit static-route metadata keeps its existing user-defined meaning. Clients should read `metadata.limits` and its provenance before selecting budgets. Listings still include display name (first alias), capabilities (`text`, `image`), server URL, detected mode and load status; aliases share the same upstream route unless `aliases = false`.

Upstream semantics were checked against the [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) and [model/property response implementation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-context.cpp). Older versions may report less information; missing data remains unknown or explicitly estimated.

If a server becomes unreachable during a refresh, its last known routes are kept and the failure is reported in authenticated `/diagnostics` (`llama_cpp.servers[].reachable = false`, `stale = true`). A server that was never reachable has no routes or active provider, but its reserved identity is reported as `llama_cpp.servers[].provider`. A successful empty model list removes that server's previous model routes. Failed catalog refreshes likewise retain the last successful catalog routes. The router itself keeps serving.

Providers, routes, public model entries and discovery status are published as one immutable registry snapshot. Each model listing and diagnostics response uses one snapshot, and a resolved request retains its route and provider through dispatch even if refresh removes the entry or changes provider settings. Later requests use the new snapshot.

### Versioned model capabilities

Model listings now include `metadata.router.schema_version = 1` while preserving the existing list envelope and legacy fields. The extension describes reasoning toggles, exact known effort values, budgets, defaults and native mappings; separate input/output modalities; tools and parallel calls; structured output; sampling controls; native/translated protocols; and field provenance, conflicts and freshness. Boolean false means unsupported; null means unknown. llama.cpp's effort-support flag does not establish any effort values.

Local enrichment uses applicable `/props` defaults, modality flags, template capabilities and build information, with per-model router queries using `autoload=false` only for loaded/sleeping models. The bundled Go configuration enables `catalog.metadata_source = "opencode-go-v1"`, joining availability with a frozen, provider-specific Models.dev review. Its review freshness expires after 30 days independently of live catalog refreshes; there are no runtime Models.dev requests. Reviewed facts are not authenticated inference verification. Exact model/dialect matches are required, and missing controls or enums remain unknown.

Sparse `[capabilities.overrides."public-model-id"]` tables enrich existing or subsequently discovered routes without copying routing configuration. They cannot create routes, select transports, or configure SDKs/UIs. Clients must consume the extension explicitly and map controls for their selected ingress. See [CAPABILITIES.md](CAPABILITIES.md) for the complete schema, evidence/precedence rules, override examples, source review procedure, and client integration example. Prompt 08's corrected limits are reused under `metadata.router.limits`; their existing fields and overrides retain their semantics.

### Discovery refresh and lifecycle

The router makes **one synchronous discovery attempt per configured source before listening**, preserving initial inventory and `--check` behavior. After startup, a managed background worker refreshes catalog and llama.cpp discovery on independent schedules. Model listing, resolution, readiness and diagnostics only read cached state: they never initiate or wait for discovery network work. Liveness continues to return `status = "ok"` during discovery outages; it does not certify upstream readiness.

After a successful round, the next attempt is due `refresh_seconds` after completion. After a failed round, retries use exponential backoff: the ceiling starts at 5 seconds, doubles with each consecutive failure, and caps at 300 seconds. Each delay is chosen uniformly between half the ceiling and the ceiling. Retries are measured from completion, independently of the last success. Recovery resets the failure count and restores the configured refresh interval. Overlapping refresh calls for the same source return immediately, including explicit forced calls. Request bursts cannot trigger extra attempts.

For llama.cpp, servers are probed concurrently (up to eight); successful servers publish their new inventories while failed servers retain their own last usable routes. Any server failure counts as a failed discovery round for scheduling, even if others succeed. The source's `last_success` records the last **entirely successful round**; each server also reports its own success and attempt times. Optional `/props` remains best effort. The single scheduling worker processes llama.cpp and catalog in turn, so an active source can delay the other until its probes finish under the configured HTTP timeouts.

Set a source's `refresh_seconds = 0` for **startup-only** mode. This now means exactly one startup attempt, even when that attempt fails; there are no automatic retries for that source. Set a positive interval to discover servers started later or recover automatically from startup failure. If discovery has never succeeded and no usable disk cache is enabled, its inventory is empty: explicit routes and other successful sources remain available, unknown models return 404, and an unreachable llama.cpp server has no active provider for `provider/model` routing. A partially successful first llama.cpp round exposes the successful servers and reports `loaded = true` even though the round's `last_success` is still null.

Authenticated diagnostics include `catalog.refresh` and `llama_cpp.refresh`, atomically associated with their inventories:

| Field | Meaning |
| --- | --- |
| `last_success`, `last_attempt`, `next_attempt` | Process-local monotonic timestamps; null means no success, no attempt, or no scheduled retry respectively. They are not Unix timestamps and cannot be compared across restarts. |
| `consecutive_failures` | Failed rounds since the last complete success. |
| `age_seconds` | Time since last success; null if never successful. |
| `stale` | No successful round, a failed round, or the normal refresh interval has elapsed. A successful startup-only inventory does not expire. |
| `stale_age_seconds` | Time beyond the normal refresh interval since last success, floored at zero; null if never successful. Startup-only inventories report zero. |

Errors remain sanitized and length-limited. Per-server llama.cpp status includes `last_success`, `last_attempt`, `age_seconds`, `stale_age_seconds`, and `consecutive_failures` (reset on recovery); its existing `stale` flag identifies retained routes after a failed probe. Freshness ages are derived at read time from the captured immutable snapshot, so they keep increasing during a blocked refresh. New timestamps, retry state, errors and inventories publish together after the attempt completes; readers keep the previous complete generation while it is in progress.

For embedders, `App(config)` performs the initial attempt; `app.start()` starts scheduling and is idempotent. `serve()` starts it after binding, and the HTTP server's `shutdown()` / `server_close()` stop it. `app.stop()` joins the worker and permits a later `start()`; `app.close()` is idempotent, permanently stops scheduling, waits for active discovery, closes HTTP resources, and removes diagnostic filters. Always call `close()` in a `finally` block. Shutdown waits for in-flight probes to finish under the configured timeouts, then skips remaining sources; it does not abandon probe threads. `--check` starts no background worker. Refresh intervals and discovery timeouts must be finite and within `threading.TIMEOUT_MAX` (timeouts positive; intervals nonnegative).

**Compatibility:** requests no longer discover a new model synchronously; it becomes available after the next scheduled round. Failed startup-only discovery no longer retries automatically. Last usable routes are retained in memory without an expiry, so generation can still fail against an unavailable upstream. Discovery state is persisted across restarts only when the optional cache below is configured.

### Optional discovery cache

Persistence is off by default. Opt in with a private file on a local POSIX filesystem:

```toml
[discovery_cache]
path = "/var/lib/model-router/discovery.json"
max_age_seconds = 86400  # positive, finite; default 24 hours
```

Relative paths resolve against the process working directory; `~` expands to the current user's home. The cache is read before the usual bounded startup discovery attempts, which still happen before listening. A failed initial probe can retain the cached inventory; this does not skip startup timeouts or make an unavailable generation endpoint work. Startup-only mode still makes one attempt and starts no retry worker. Removing this table disables all cache I/O; delete the file separately if desired.

The version 1 JSON file stores successful routable catalog availability and each successful local server's model IDs/aliases, selected numeric limits, display names, basic modalities and typed capability facts with observation times. It stores **no provider objects, endpoints, headers, API keys, requests or completions**. Unmapped catalog IDs, arbitrary upstream metadata, raw `/props`, server URLs/build strings, freeform stop strings and capability conflict histories are omitted. Strings containing full HTTP URLs or recognized/configured secrets are omitted; an unsafe model ID/alias causes its entry to be omitted rather than rewritten. Cache fidelity is intentionally narrower than a live model listing.

A SHA-256 digest identifies the complete configured discovery/routing namespace: provider identities/settings, catalog URL/headers/mappings/rules/review source, ordered local server URLs/names/headers/keys, prefix/alias policy, limit overrides/fallback, and explicit model/alias targets and enabled flags. Sensitive configuration contributes only to the digest, never as plaintext. **Any change to these inputs discards the entire file**, including credential rotation, removed providers/servers/models/aliases and newly disabled names. Discovery intervals, ingress authentication and retry policy do not invalidate it. Use one cache path per configuration and one writing process; concurrent processes sharing a file are unsupported. The digest detects configuration drift, not malicious edits: the file/directory must remain trusted.

`max_age_seconds` is a **restart admission limit**, measured in wall-clock seconds from each source's last successful observation. Entries expire at the exact age limit; future timestamps are rejected. Catalog and individual local servers expire independently; failures and other sources' successful writes never advance their observation times. After admission, routes follow the existing in-memory retention policy without a hard runtime expiry. A successful discovery replaces that source's inventory, including removing routes on a successful empty list; a failed probe retains it. Models removed while a source is offline cannot be detected until discovery succeeds. Disabled explicit names and aliases still block cached discovery. As before, disabling an alias alone is not access control when dynamic routing is enabled.

Providers are reconstructed only from current trusted configuration. Cached data cannot choose a provider endpoint, dialect, credential or header. Collision precedence remains explicit routes/aliases (including disabled reservations), then local servers in configured order, then catalog entries, regardless of cached/live origin. A cached local route therefore retains precedence over a live catalog entry. Catalog IDs are mapped again using current trusted rules; cached local providers use reserved deterministic identities.

Writes use a private temporary file in the destination directory, flush/fsync, atomic replace, and directory fsync. Files are mode `0600`; newly created cache directories are `0700`. Pre-existing directories must be managed by the operator. Reads require a regular, non-symlink file owned by the current user with no group/other permissions; files larger than 8 MiB are rejected. Missing, unreadable, corrupt, incompatible or unsafe files never prevent startup: discovery proceeds normally. Parsing is all-or-nothing (apart from independent source expiration); successful discovery can replace the rejected file. Write failures leave live in-memory discovery usable and, before replacement, preserve the old file. This feature depends on POSIX ownership/permissions and local atomic replacement semantics; it is not an encrypted backup, a multi-process database or a connectivity guarantee.

Authenticated `/diagnostics.discovery_cache` reports enablement, load outcome (`missing`, `loaded`, `empty`, `configuration_changed`, `incompatible`, `invalid`), expired source count when applicable, and a generic write error. No file contents or parser exceptions are logged. `catalog.cached` and `llama_cpp.servers[].cached` identify retained disk observations, with `succeeded_at` as Unix seconds. Model `metadata.router.freshness.discovery` also includes `cached` and `succeeded_at`; imported observations are always stale until a successful live replacement. Ages survive restart by reconstructing process-local monotonic timestamps, which can be negative for earlier observations. Cached local providers are unverified/unreachable for readiness until a successful probe; catalog availability retains its existing readiness policy. Live recovery clears the corresponding cached flag and resets normal refresh/backoff state.

### Liveness, readiness and diagnostics

All operational GET endpoints are local reads: none initiates, schedules, or waits for discovery I/O. `/livez` returns HTTP 200 with only `{"status":"ok"}` while the handler can respond, regardless of upstream outages or empty inventory. `/health` and `/healthz` remain public liveness aliases with the same minimal body. Existing status-only probes can keep using them. **Compatibility change (prompt 14):** clients reading provider/discovery details from those aliases must move to `/diagnostics` and supply a router API key. Use `/readyz` for traffic readiness instead of relying on an always-200 liveness endpoint.

`/readyz` returns only `{"status":"ready"}` (HTTP 200) or `{"status":"not_ready"}` (HTTP 503). Readiness requires at least one enabled named route in the current registry whose provider has no cached reachability failure. Disabled routes do not count. Dynamic `provider/model` convenience routing alone does not establish a named inventory and does not satisfy readiness. Explicit routes are considered usable without probing their endpoints; catalog availability also does not prove generation connectivity. This is an inventory/admission policy, not a guarantee that a particular generation or model works, nor a check of momentary generation-slot availability.

A failed llama.cpp probe excludes only that provider's routes from readiness, even though its stale routes remain dispatchable for clients choosing them. Other local servers and explicit routes can keep the router ready. Failed catalog refreshes retain eligible last-known routes: a catalog outage says nothing conclusive about generation endpoint availability. Staleness alone has no expiry that makes routes unready. With no successful discovery, enabled explicit routes can still satisfy readiness; a discovery-only router with no successful usable inventory is unready. A successful empty inventory is likewise unready. Recovery is reflected after the next successful publication. Consult per-source/server status to assess partial failures even while the overall router is ready.

`/diagnostics` uses the same router API keys and header formats as generation/model listing. Missing/incorrect credentials return 401. If router auth is intentionally disabled (`auth_mode = "none"`, or no configured keys), diagnostics are disabled with HTTP 403; an arbitrary supplied key cannot enable them. Configure router authentication to use this surface. It reports the readiness decision/counts, provider name/dialect/route counts, reachability (`available` from local probes, `unavailable`, or `unprobed` for static endpoints), and catalog/llama.cpp discovery status. Never-successful local providers appear under `llama_cpp.servers` even without an active provider. Credentials, headers, transport kwargs and full model metadata are omitted; all included strings, errors and URLs pass through the shared sanitizer and per-string limit. No health request probes static providers.

For embedders, `app.liveness_payload()`, `app.readiness_payload()` and `app.diagnostics_payload()` expose these local reads. The legacy Python `app.health_payload()` remains a diagnostic alias for compatibility; it is not the public HTTP liveness body. Embedders are responsible for authenticating any diagnostic surface they expose.

### Request IDs and operational logging

Each admitted HTTP request receives a fresh server-generated UUID hex ID in `X-Request-ID`, including auth/validation errors, operational GETs and streaming responses. Client and upstream `X-Request-ID` values cannot replace it. Connection overload rejection occurs before a handler exists, so its best-effort 503 has no request ID or per-request record.

The router emits one JSON `request_outcome` log per handled request, after streaming/cleanup: `request_id`, method, recognized path, `http_status`, `provider`, public `model`, `upstream_model`, `translation_mode` (`passthrough` or `translated`), ingress/egress dialects when selected, `elapsed_ms`, stream flag and `stream_outcome`. Auth/validation failures have null selection fields. Dynamic model IDs supplied by clients are omitted from logs. Unknown paths are recorded as `unknown`; queries, prompts, completions and raw upstream request exceptions are never included. Configured secret values are sanitized before serialization, preserving valid JSON. The CLI retains a timestamp/level prefix around the JSON message.

`discovery` on outcome records summarizes cached staleness, age, stale age and consecutive failure counts per enabled source. Each completed discovery attempt also emits a JSON `discovery_refresh` record with source, elapsed milliseconds and the same summary, so idle routers still report changes. Detailed errors remain in authenticated diagnostics with the existing redaction/length policy. This requires no external metrics service.

Example JSON log message for a stream that failed after its headers were sent:

```json
{"event":"request_outcome","request_id":"0123456789abcdef0123456789abcdef","method":"POST","path":"/v1/responses","http_status":200,"provider":"local","model":"assistant","upstream_model":"model-a","translation_mode":"translated","ingress":"openai-responses","egress":"openai-chat","stream":true,"stream_outcome":"failed","elapsed_ms":412.5,"discovery":{"llama_cpp":{"stale":true,"age_seconds":120,"stale_age_seconds":60,"consecutive_failures":2}}}
```

### Router auth

Use `server.auth_mode = "required"` with `server.api_keys` (or `MODEL_ROUTER_API_KEY` / comma-separated `MODEL_ROUTER_API_KEYS`). Required mode without a nonempty key fails startup. Explicit key entries must be nonempty strings; configured missing/empty environment substitutions are errors. Clients may send `Authorization: Bearer <key>`, `x-api-key: <key>`, or `api-key: <key>`.

For intentional unauthenticated operation, use `server.auth_mode = "none"` and omit `api_keys` (or use `[]`). This ignores the auth environment variables and rejects conflicting nonempty configured keys. Bundled examples use this mode on localhost; migrate an old optional `${MODEL_ROUTER_API_KEY}` entry by either removing it and selecting `none`, or selecting `required` and supplying the secret. Repeated `--api-key` overrides the configured mode/keys and enables authentication. For compatibility, omitting `auth_mode` preserves key-based inference: configured/environment keys enable auth, otherwise auth is disabled. Prefer an explicit mode for new configurations.

Upstream credential references (`api_key_env`/`api_key_file`) must also resolve to a nonempty secret. Omit the credential or use a literal empty `api_key` for an intentional unauthenticated upstream (the public placeholder policy above still applies).

## Behavior notes and limits

- **Translation** uses LiteLLM adapters with the strict/lossy policy below. Chat `stream_options.include_usage` is supported by the shared adapter streaming wrapper even for upstreams without that native request field. Native features do not all have equivalent mappings. Token *counts* and error wording differ between dialects; the response shape follows the ingress dialect.
- **Anthropic ingress requires `max_tokens`**, as the Anthropic API does.
- **Reasoning / encrypted content:** same-protocol responses preserve native fields. Supported translations can re-encode thinking text into ingress reasoning fields; exact signatures, encrypted content and arbitrary reasoning extensions are not guaranteed across dialects.
- **`sequence_number`** on translated Responses streams is assigned by the router: contiguous integers starting at zero for each request, including synthesized errors. Adapter-supplied values are replaced even when present, duplicated, or decreasing. Native Responses SSE bytes and sequence values remain unchanged.
- Request bodies accept one unsigned decimal `Content-Length` or a single `Transfer-Encoding: chunked`. Conflicting framing, duplicate length/transfer headers (even identical ones), transfer-coding lists, signed sizes, malformed chunk extensions, missing CRLF delimiters, and truncated bodies return HTTP 400. Chunk extensions are validated and ignored; valid trailers are discarded and cannot override framing, representation, or credential headers.
- `max_body_bytes` bounds both the encoded payload and the decompressed body. Oversized bodies or chunk metadata return HTTP 413. Each chunk-size/extension or trailer line is limited to 1,024 bytes including CRLF; all chunk metadata (size lines, extensions, delimiters, and trailers) is limited to 64 KiB per request, with at most 100 trailer fields. These framing limits also apply to small-chunk requests.
- Gzip (`Content-Encoding: gzip`) must contain **exactly one complete member**, with a valid checksum and length trailer. Concatenated members and all trailing bytes (including zero padding) return HTTP 400. Other content encodings or encoding lists return HTTP 415. Ordinary gzip and chunked gzip requests are supported.
- Compatibility change (prompt 03): requests are decoded, parsed, and serialized as UTF-8 JSON with the upstream model ID. Outgoing `Content-Type` is `application/json`; the transport computes `Content-Length`. Original encoding, range/location/language metadata, validators, digests, and signatures are discarded because they describe the original bytes. Static provider headers and translation header overrides cannot override this framing policy. Vendor JSON fields and unrelated protocol headers retain their existing behavior.
- The router is synchronous per request (bounded `ThreadingHTTPServer` + LiteLLM's sync APIs); long streams hold one admitted thread and generation slot each. It is built for a handful of local clients, not as a multi-tenant gateway.
- Request outcome logs contain no prompts, completions, request headers, arbitrary URL queries or raw upstream exceptions. `--verbose` enables router debug diagnostics; dependency debug logging stays disabled because it can expose generation bodies.

### Streaming ownership and failures

A response `Plan` owns its output iterator, adapter source iterators, and request-specific upstream HTTP responses. The relay calls `close()` / `aclose()` on completion (including empty output), first-event failure, later failure, downstream write failure/disconnect, and async task cancellation. Closing a plan repeatedly is safe; one failing cleanup does not skip other resources. Sources are registered independently of serializer generators, so even an unstarted plan can be closed. Generation clients and their pools belong to the request and are closed with the plan; App retains its separate discovery pool. Embedders that dispatch without the HTTP handler must call `plan.close()` outside an event loop or `await plan.aclose()` inside one.

After a downstream write fails, the relay stops pulling events and closes upstream responses rather than draining generation. Async cancellation also closes the source and transport. A sync adapter already blocked in an upstream read cannot always be interrupted immediately: client disconnect detection resumes at its next downstream write, or the smaller of the configured upstream timeout and remaining request deadline ends the read. Closing the HTTP response asks the upstream to stop; whether its backend cancels generation is provider-dependent.

Failures **before the first downstream event** return a sanitized ingress JSON error. Deadlines and ambiguous transport failures use the retry policy above; other errors retain their adapter HTTP status mapping. Once streaming headers have been sent, the HTTP status is fixed. A later exception emits a sanitized terminal event: Chat uses a `data: {"error": ...}` frame followed by `[DONE]`; Responses uses `event: error` with `type`, `code`, `message`, and `param` (plus the stream's next `sequence_number` on translated streams); Messages uses `event: error` with an Anthropic error object, without a success `message_stop`. Translated adapter-emitted error events (including `response.failed`) also end consumption and suppress subsequent success events. The sync Messages adapter's swallowed exceptions are surfaced before any synthesized success terminator.

The first event means the first **downstream** event. In particular, the installed Messages-from-Responses adapter emits a synthetic `message_start` before reading upstream; failures discovered afterward are SSE errors under HTTP 200. Empty iterators finish without a generated error; adapter behavior may instead produce protocol events or reject a native Responses stream missing its terminal event. No successful content or completion event is fabricated by the relay itself.

JSON request logs record the actual `http_status` separately from `stream_outcome` (`none`, `completed`, `failed`, `disconnected`, or `cancelled`). A failed stream can therefore have `"http_status":200,"stream_outcome":"failed"`. Native successful HTTP streams remain opaque, including provider error events embedded in SSE: bytes, comments, vendor fields, and ordering are preserved, and only transport failures are classified by the router. This preserves the existing fidelity policy; it does not promise redaction or semantic validation of native SSE error frames.

**Compatibility change (prompt 12):** translated Responses numbering is now router-owned, and translated terminal failures stop consumption. Same-protocol stream bytes retain their existing fidelity. Error event shapes follow the [Responses streaming event reference](https://developers.openai.com/api/reference/resources/responses/streaming-events) and [Messages streaming error format](https://platform.claude.com/docs/en/build-with-claude/streaming#errors).

## Package layout

The implementation lives in the `model_router` package. The repository's
`model_router.py` is a thin launcher; `python model_router.py`, the installed
`model-router` command, and `python -m model_router` use the same CLI.

| Module | Responsibility |
| --- | --- |
| `config` | Configuration validation, credential loading, immutable provider/route records and defaults |
| `discovery` | Catalog/llama.cpp probes, refresh worker, persistent cache, atomic registry snapshots and cached diagnostics |
| `transport` | Request-owned HTTP transports, retry policy and upstream deadlines |
| `headers` | Credential ownership, session selection, protocol headers and representation-header policy |
| `adapters` | One `LiteLLMAdapter` interface containing pinned LiteLLM imports, private APIs, SDK transport creation and hidden stream resources |
| `protocol` | Ingress validation, allowed generation fields and strict/lossy translation policy |
| `streams` | Response plans, resource cleanup, translated SSE serialization and event sequencing |
| `app` | Generation dispatch using a pinned route/provider and the protocol/transport interfaces |
| `http` | Bounded HTTP server, request framing, endpoints, admission and response/stream relay |
| `cli` | Arguments, config selection, checks and server lifecycle |
| `common`, `constants` | Shared JSON, immutable data, sanitized errors/logging and protocol defaults |

`router_metadata` retains the packaged capability schema and reviewed Go JSON.
No runtime dependencies or HTTP framework changed in this extraction. Routing
precedence, configuration/credential semantics, public metadata, errors and stream
behavior retain their existing contracts.

Public embedding imports such as `from model_router import App, Config, Plan,
build_config, serve, main` remain available. Configuration and discovery imports
can now load without initializing LiteLLM; the pinned adapter checks still run
when the application or CLI loads, including `--version` and `--validate-config`.
Python implementation details have new module locations: private helpers and
patch targets belong to their owning modules, and class `__module__` values now
refer to those modules. The old internal namespace and serialized Python class
paths are not compatibility guarantees. CLI arguments and wire formats have no
compatibility change.

## Tests

```bash
.venv/bin/python -m pip install -c constraints.txt '.[test,dev]'
.venv/bin/python -m pytest
.venv/bin/python -m ruff check model_router.py model_router router_metadata scripts tests
.venv/bin/python -m mypy
```

GitHub Actions builds an sdist and wheel, installs the wheel, checks the installed entry point/version and bundled metadata, validates default `router.toml` loading using only a dummy key, and runs the full local suite on Python 3.10–3.14. Lint checks syntax/undefined names and related correctness errors across the package, launcher, metadata, scripts and tests; type checks cover `router_metadata` and the release smoke script. This is an initial achievable baseline, not a claim that the whole router is strictly typed. [Contributor checks](CONTRIBUTING.md) explain the scope and how to expand it.

`tests/test_package.py` checks public import identity, configuration/discovery
without the translation backend, the private-import boundary, and version/help/
offline validation through all three CLI launchers. Existing behavioral tests
retain their assertions and patch globals in the modules that now own them.

The suite starts local fake HTTP upstreams, a fake llama.cpp server and the real router in-process. Its generation matrix exercises all **12 ingress→upstream combinations in both modes (24 cases)**: Chat, Responses and Messages ingress against Chat, Responses, Messages and Gemini upstreams, through installed LiteLLM adapters for every translated route. It checks outgoing paths, model IDs, credentials, text/response structures, SSE lifecycle order, terminal events and Responses sequence numbers. It also covers pass-through fidelity, tool-schema translation, auth, session-header forwarding, catalog discovery, llama.cpp discovery across single/router/legacy modes (context sizes, aliases, prefixes, stale routes), gzip/chunked bodies, dynamic routing, and model listing. Bundled Go routing tests use checked-in public catalog/endpoint fixtures, fake credentials and a blocked network transport; tests never fetch the live catalog or perform cloud inference. See `tests/fixtures/README.md` for the mapping review procedure. The fake upstream honors request content encoding. Registry tests cover duplicate explicit names, generated-name collisions through recovery/failure, route precedence and stale inventories. Events and barriers pause requests between resolution and dispatch, pause model/health reads across refresh, and coordinate concurrent catalog/local publications without sleep-based races. Raw-socket and bounded-reader tests cover malformed framing, truncated bodies, chunk metadata limits, gzip integrity/completion, and encoded/decompressed size limits. Capacity tests use local sockets, controlled upstreams, and held generations to cover admission before thread creation, absolute read deadlines, stalled response readers, sync/async stream disconnects, overload recovery, and shutdown.

`tests/test_protocol_contracts.py` adds complete tool-call/result round trips and inline image requests for all twelve combinations, plus tool argument reassembly from fragmented Chat/Responses/Messages SSE (Gemini emits complete function-call objects). Usage contracts check non-streaming and streaming counts across the same matrix: prompt/output totals, cached input, exact reasoning subcounts where the upstream reports them, Anthropic cache creation and native server-tool counts. Focused reasoning cases assert actual outgoing effort/budget settings and returned thinking content/signatures on supported routes. Native Chat/Responses/Messages retain OpenCode `cost`, allowance and billed-token extensions in JSON and final usage events. Translation does not guarantee preservation of arbitrary provider billing extensions or encrypted/signed reasoning across dialects.

Actual upstream HTTP 400/429/503 responses are tested for every combination in both modes, with zero adapter retries, ingress error shapes, retry/rate-limit headers and exclusion of unrelated diagnostic headers. Provider SSE errors before content and midstream are also tested, including suppression of success events after a translated failure. The installed Responses adapter can replace an upstream SSE error message with `Response API in-stream error`; exact upstream wording is not a contract. The former upstream-error test is explicitly named as a local unknown-model test. First-event/midstream socket failures and disconnect recovery include Gemini on all three ingresses; cancelling an embedder between pulls is tested with real open adapter/HTTP resources on all nine translated paths, followed by upstream socket closure and a successful repeat request. Separate sync/async relay cancellation tests remain in the lifecycle suite.

Protocol fixtures are synthetic and based on the official schemas linked in `tests/fixtures/README.md`. Tests use fake credentials, inline images, local HTTP and LiteLLM's installed cost map, with telemetry disabled and external socket connections blocked. They make no paid inference calls. These contracts establish adapter/relay behavior for the installed dependency, not live model quality, provider acceptance of every control, or compatibility with all future LiteLLM versions. Prior credential-escape, ambient-key, redaction, malformed-body and other adversarial tests remain in the suite.

Conversation affinity tests assert actual outgoing headers with installed adapters across all three ingresses and four upstream dialects, streaming and non-streaming. They interleave two conversations with shared credentials and initially identical prompts, then change prompt content on repeated turns. Additional cases cover conflicting sources, configured priority, opt-in cache fallback, absent/required identity, duplicate and invalid values, custom targets, disabled sources and static-session configuration rejection.

`tests/test_discovery_refresh.py` uses a fake monotonic clock and controlled local HTTP responses to cover success/failure/recovery, bounded jitter and backoff, partial outages, startup-only failures, independent source schedules, cached liveness/listing/resolution during request bursts, overlapping refresh calls, and worker/transport cleanup. Events hold responses open and observe worker idle boundaries without sleep-based races.

`tests/test_discovery_cache.py` covers offline restart/recovery, per-source expiration without renewal on failure, corrupt/incompatible/unsafe files, configuration and credential changes, removed/disabled models and aliases, collision precedence, successful empty inventories, atomic replacement failures, and omission of credentials/URLs/freeform metadata from disk. All credentials are fake and all discovery uses local test servers.

`tests/test_stream_lifecycle.py` uses close-tracking sync/async iterators and real local SSE upstreams to verify empty output, first-event and midstream failures, terminal event structure, missing/duplicate/decreasing sequence values, idempotent ownership, setup failures, async cancellation, disconnect-driven upstream socket closure, and repeat requests. Native SSE fidelity assertions include comments, vendor events, and unchanged nonmonotonic sequence values.

`tests/test_operations.py` verifies public probe/auth boundaries, sanitized diagnostics, cached partial/total outages, never-successful and empty discovery, network-free probes, correlated JSON outcomes and CLI exit/offline behavior. Stream lifecycle tests assert HTTP status independently from stream outcome and correlate response request IDs.

## CLI

```
python model_router.py [-c CONFIG] [--host HOST] [--port PORT]
                       [--api-key KEY ...] [--env-file PATH] [--no-default-env]
                       [--validate-config | --check-readiness | --check] [--offline]
                       [--verbose] [--version]
```

`--validate-config` loads credentials, expands environment substitutions and validates configuration/overrides without constructing an App, HTTP client, server or discovery worker. It needs locally available configured credentials, including files explicitly referenced by the config; offline does not mean missing required credentials are accepted. Use `--no-default-env` to avoid loading the default credential file.

`--check-readiness` makes exactly one initial attempt for each configured discovery source, including sources with `refresh_seconds = 0`, prints routes to the log and a JSON readiness summary to stdout, then closes HTTP resources. It never listens or starts a background worker. It applies the same cached readiness policy as `/readyz`; it does not send paid inference requests or probe static generation endpoints. A partial discovery outage can exit 0 when another usable route remains. No successful discovery and no explicit enabled route exits 3. A successful empty inventory also exits 3.

`--check` remains an alias for `--check-readiness`, retaining initial discovery and route printing. **Compatibility change (prompt 14):** it previously always exited 0 after config validation, including failed discovery; it now reports readiness. Scripts that require only config validity should migrate to `--validate-config`.

`--offline` is accepted only with a check. `--validate-config --offline` remains config validation. `--check-readiness --offline` (or `--check --offline`) validates configuration but skips App construction, disk cache loading and all network work, reports `status = "indeterminate"`, and exits 4. This applies even in startup-only mode or with persistence enabled: a disk inventory does not establish connectivity. Run the online check to load any configured cache and make its one startup attempt, or select config validation if network evidence is unnecessary.

| Exit code | Meaning |
| --- | --- |
| 0 | Config valid (`--validate-config`), or readiness satisfied (`--check-readiness`/`--check`) |
| 1 | Unexpected runtime/check failure |
| 2 | Invalid arguments, unreadable/invalid config or missing required credentials |
| 3 | Valid config, readiness not satisfied after discovery |
| 4 | Valid config, offline readiness indeterminate |

Examples (pass `-c PATH` as needed):

```bash
model-router --no-default-env --validate-config
model-router --no-default-env --check-readiness
model-router --no-default-env --check --offline  # exits 4 after validation
```
