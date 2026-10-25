# litellm-model-router

A single-file, LiteLLM-backed HTTP router that accepts **OpenAI Chat Completions**, **OpenAI Responses**, and **Anthropic Messages** requests, and speaks any of those protocols (plus Gemini) upstream. Successful same-protocol responses are passed through untouched; cross-protocol calls are translated by LiteLLM. Local **llama.cpp** servers are discovered automatically — models, aliases, context size and capabilities included.

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

Pass-through is byte-faithful for successful response bodies. HTTP error bodies use the redaction policy below. On the request side it rewrites only `model` and upstream auth/session headers, so provider-specific and beta fields (`cache_control`, `mcp_servers`, `reasoning`, `encrypted_content`, `previous_response_id`, custom vendor fields, …) survive a same-protocol hop.

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

These are exact IDs, not family-prefix rules: MiniMax M3.1 Flash uses Chat while M3 uses Messages. The checked-in snapshot contains 38 catalog IDs and 29 documented mappings. The other nine (`glm-5.1`, `grok-4.6`, `iceman-contributor-free`, `kimi-k2.6`, `minimax-m2.5`, `minimax-m2.7`, `omen-alpha`, `qwen3.6-plus`, `qwen3.7-max`) have unknown endpoint compatibility in this review. They and unmapped new IDs are omitted from `/v1/models`, reported in `/health` and `/healthz` under `catalog.unmapped_models` and in refresh warnings, and return HTTP 404 with an actionable message when requested (`model_endpoint_unknown` on OpenAI ingresses). An ID absent from the catalog and explicit/local routes returns HTTP 404 (`model_not_found` on OpenAI ingresses).

Override an exact entry in `[catalog.model_providers]`, or add an explicit `[[models]]` route after reviewing the intended endpoint. A mapping references a trusted provider; URLs, credentials and auth headers stay in `[providers]`. Documented native routing does not establish compatibility with other endpoints. No authenticated cloud inference was performed; undocumented compatibility remains unknown.

```bash
curl http://127.0.0.1:4000/v1/models

curl http://127.0.0.1:4000/v1/chat/completions \
  -H 'Authorization: Bearer unused' \
  -H 'Content-Type: application/json' \
  -d '{"model":"claude-haiku-5-5","messages":[{"role":"user","content":"Hi"}]}'
```

OpenCode Go requires an `x-opencode-session` header. The router forwards a client-supplied one, otherwise reuses `prompt_cache_key`, otherwise generates a fresh ID per request. The upstream subscription key never reaches clients.

`.venv/bin/python model_router.py --check` validates the config, fetches the catalog, and prints every discovered model.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# or, to get the `model-router` command:
.venv/bin/pip install -e .
```

Requires Python 3.10+ and LiteLLM `>=1.106.4,<2`. The router imports LiteLLM's Anthropic pass-through adapters, which are internal and not covered by LiteLLM's compatibility promise — the version pin matters.

## Endpoints

| Endpoint | Ingress protocol |
| --- | --- |
| `POST /v1/chat/completions` (also `/chat/completions`) | OpenAI Chat Completions |
| `POST /v1/responses` (also `/responses`) | OpenAI Responses |
| `POST /v1/messages` (also `/messages`) | Anthropic Messages |
| `GET /v1/models` | OpenAI-style list, plus Anthropic-style `has_more`/`first_id`/`last_id` |
| `GET /health`, `GET /healthz` | Liveness, provider count, catalog and llama.cpp status |

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

**Raw upstream errors:** HTTP statuses 400–599, including same-protocol calls and requests with `stream=true`, return a sanitized ingress-shaped JSON error rather than the raw upstream body. The HTTP status is preserved; only sanitized Retry-After/rate-limit response headers are retained. This deliberately changes error body/header fidelity. Successful same-protocol bodies, including opaque SSE bytes under HTTP 2xx, remain unchanged: embedded vendor error events or secret-looking generated content in such streams are consequently outside the redaction guarantee. Use a translated route when router-controlled error-event redaction is required.

## Configuration

TOML and JSON are supported (YAML too if PyYAML is installed). Every string supports `${ENV_VAR}` expansion. Lookup order when `--config` is omitted: `router.toml`, `router.json`, `config.toml`, `config.json`, `router.yaml`, `config.yaml`.

```toml
[server]
host = "127.0.0.1"
port = 4000
api_keys = ["${MODEL_ROUTER_API_KEY}"]   # empty values are ignored
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
# session_header = "x-opencode-session" # per-request session id
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

Counts must be positive integers (booleans and numeric strings are rejected); `max_generations` must not exceed `max_connections`. Timeouts must be finite positive numbers within Python's platform timeout range (`threading.TIMEOUT_MAX`). Header/body limits are **absolute read deadlines**, so a trickle of bytes does not extend them. They do not limit JSON processing, discovery, or upstream generation. Provider `timeout` (default 600 seconds) independently controls upstream operations; `write_timeout` applies only while writing to the client. A stream may last indefinitely and pause between events, subject to the upstream timeout and available capacity.

Both admission limits reject immediately; there is no application waiting queue. Connection overload attempts a minimal HTTP **503**, `Retry-After: 1`, and closes the socket without spawning a handler. That response is best effort using a nonblocking write: a peer that cannot receive it may observe only a disconnect. The kernel listen backlog is separately bounded to 16. Generation overload returns an ingress-shaped HTTP **503** (`generation_overloaded` for OpenAI dialects), `Retry-After: 1`, and closes the connection. Validation precedes generation admission; admission precedes route resolution and never waits under discovery locks. GET requests use connection capacity but no generation slot; reserve spare connections if model listing/health must remain accessible during generation load.

Incomplete request lines/headers and bodies return HTTP **408** and close, with `header_read_timeout` or `body_read_timeout` error codes for OpenAI dialects. Before headers identify an ingress, timeout errors use the OpenAI error shape. A downstream write timeout or disconnect closes the stream/connection and releases its generation slot after cleanup; an error cannot reliably be delivered to a stalled reader. Validation errors and upstream exceptions/timeouts also release capacity. All responses close the connection, as before.

Shutdown stops admission, closes client sockets, and clears reservations without waiting for client read/write deadlines. Synchronous upstream calls cannot be forcibly interrupted safely: a disconnect is observed on the next downstream write, and an adapter already waiting on its upstream may run until its provider timeout or next event. **Its slot stays reserved until it unwinds during normal operation**, preventing cancellation storms from exceeding upstream concurrency. During shutdown such already-running calls can finish in daemon handlers; shutdown does not wait for them or promise to cancel remote inference. This remains a lightweight local server, with no total stream duration limit or background cancellation worker.

Compatibility change (prompt 04): previously unbounded connections/generations now use these defaults. Increase the limits or inbound read deadlines explicitly for larger local workloads/uploads; increase `write_timeout` for slow receiving clients. None of these settings shortens the configured upstream timeout.

### Providers

| Key | Meaning |
| --- | --- |
| `dialect` | `openai-chat`, `openai-responses`, `anthropic`, or `gemini` — what the endpoint speaks |
| `base_url` | OpenAI dialects: include `/v1`. Anthropic: use the origin or a prefix (`/v1/messages` is appended). Gemini: include `/v1beta`. A base already ending in the provider's final path is used as-is. |
| `api_key` | Literal key |
| `api_key_env` | Read the key from this environment variable (error if unset) |
| `api_key_file` | Read a key from a file (trailing whitespace stripped) |
| `headers` | Extra upstream headers, e.g. beta flags |
| `auth_header` | Force `bearer`, `x-api-key`, or `api-key` auth. Empty uses the provider default. |
| `session_header` | Header carrying a per-request session ID. Client value wins; otherwise the router uses `prompt_cache_key` or a generated UUID. |
| `timeout` | Upstream timeout in seconds (default 600) |
| `litellm_provider` | LiteLLM provider prefix (default `anthropic`, `gemini`, or `openai` by dialect) |
| `litellm_kwargs` | Trusted administrative/generation keyword arguments passed to LiteLLM calls; never supplied by client JSON |
| `compatibility` | Translation policy: `"strict"` (default) or explicit `"lossy"`; see below |
| `drop_params` | Legacy alias: `false` selects strict, `true` selects diagnosed lossy mode; must agree with `compatibility` if both are set |

Upstream credential selection has two behaviors:

- A nonempty `api_key`, `api_key_env`, or `api_key_file` selects that configured key. Surrounding whitespace is stripped. The top-level provider key owns LiteLLM's `api_key` even if `litellm_kwargs` repeats it; the selected auth header also wins case-insensitive header conflicts. Other explicitly configured headers remain trusted. Adapters may add their native credential header as well as an `auth_header` override, using the same selected key.
- An omitted, empty, or whitespace-only key intentionally selects the public placeholder **`model-router-no-auth`**. Both passthrough and translation send it using the provider's native auth mechanism (or configured `auth_header`). An unauthenticated server must accept/ignore this placeholder; servers that reject any auth header are not supported. No environment or LiteLLM global provider key is used as a fallback.

This avoids [LiteLLM's environment/global credential sources](https://docs.litellm.ai/docs/set_keys) without clearing or swapping process-wide environment variables during requests. Only named `api_key_env` settings opt into environment credentials. LiteLLM's `os.environ/NAME` key indirection is rejected; use `api_key_env` instead. Global LiteLLM `proxy_auth`, `client_session`, or `aclient_session`, which can inject auth after argument merging or through client defaults, cause translated calls to fail closed without modifying those globals. Keep custom credential-bearing clients/transports out of `litellm_kwargs`; they are trusted administrative code outside this native-adapter credential contract. Client-visible router auth is separate and never forwarded.

Compatibility change (prompt 02): keyless passthrough calls now send the same placeholder as translation, replacing absent auth headers or the old `not-needed` placeholder. Existing nonempty configured keys continue to work. Raw HTTP error bodies are replaced as described above. These isolation regressions are tested against installed LiteLLM 1.106.4 with local upstreams and dummy environment/global keys for Anthropic, OpenAI Chat/Responses, and Gemini.

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

Headers are matched/merged case-insensitively. Precedence is native version default < inbound allowlisted header < trusted `litellm_kwargs.headers` < trusted `litellm_kwargs.extra_headers` < provider `headers` < configured authentication. Values replace earlier values; beta lists are not concatenated. Anthropic defaults to `2023-06-01` when no version is supplied, preserving existing behavior; see its [version header](https://platform.claude.com/docs/en/api/versioning) and [beta opt-ins](https://platform.claude.com/docs/en/api/beta-headers). A beta header enables only the upstream feature; it does not give an unmapped request field a bridge mapping. Client credentials, cookies, destinations, transport/representation headers and arbitrary headers cannot enter translated calls. The configured session-header mechanism is unchanged. Same-protocol forwarding keeps its broader filtered header policy; configured headers now also replace inbound headers case-insensitively. Authentication and serialized JSON framing remain server-owned on both paths.

**Compatibility change (prompt 10):** omitted `drop_params` used to enable silent dropping; the default is now strict. Explicit legacy `drop_params=true` selects lossy mode with diagnostics rather than enabling LiteLLM dropping. Review affected clients and configure loss acceptance deliberately.

### Models

`name` is the public route; `aliases` are additional public names. `dialect` overrides the provider default if one endpoint exposes both protocols. `enabled = false` hides a route and makes requests for it resolve as not found.

If a request names an unlisted model as `provider/model`, the router forwards it to that provider dynamically using the provider's default dialect — handy for trying models without editing config:

```json
{"model": "local/qwen2.5-coder:7b", "messages": [...]}
```

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

Existing `[[catalog.rules]]` regex rules remain supported as an operator opt-in: exact mappings win, then the first matching rule. At least one exact mapping or rule is required. A broad prefix or catch-all rule explicitly accepts responsibility for routing future matching IDs; the bundled Go configuration has neither. Catalog data cannot choose a provider, URL or credential. Unmapped IDs without an explicit/local route are omitted, diagnosed and rejected as described above. Explicit `provider/model` requests remain a manual endpoint choice using the named provider's dialect; they do not establish verified compatibility.

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

If a server becomes unreachable during a refresh, its last known routes are kept and the failure is reported in `/health` (`llama_cpp.servers[].reachable = false`, `stale = true`). A server that was never reachable has no routes or active provider, but its reserved identity is reported as `llama_cpp.servers[].provider`. A successful empty model list removes that server's previous model routes. Failed catalog refreshes likewise retain the last successful catalog routes. The router itself keeps serving.

Providers, routes, public model entries and discovery status are published as one immutable registry snapshot. Each model listing and health response uses one snapshot, and a resolved request retains its route and provider through dispatch even if refresh removes the entry or changes provider settings. Later requests use the new snapshot.

### Versioned model capabilities

Model listings now include `metadata.router.schema_version = 1` while preserving the existing list envelope and legacy fields. The extension describes reasoning toggles, exact known effort values, budgets, defaults and native mappings; separate input/output modalities; tools and parallel calls; structured output; sampling controls; native/translated protocols; and field provenance, conflicts and freshness. Boolean false means unsupported; null means unknown. llama.cpp's effort-support flag does not establish any effort values.

Local enrichment uses applicable `/props` defaults, modality flags, template capabilities and build information, with per-model router queries using `autoload=false` only for loaded/sleeping models. The bundled Go configuration enables `catalog.metadata_source = "opencode-go-v1"`, joining availability with a frozen, provider-specific Models.dev review. Its review freshness expires after 30 days independently of live catalog refreshes; there are no runtime Models.dev requests. Reviewed facts are not authenticated inference verification. Exact model/dialect matches are required, and missing controls or enums remain unknown.

Sparse `[capabilities.overrides."public-model-id"]` tables enrich existing or subsequently discovered routes without copying routing configuration. They cannot create routes, select transports, or configure SDKs/UIs. Clients must consume the extension explicitly and map controls for their selected ingress. See [CAPABILITIES.md](CAPABILITIES.md) for the complete schema, evidence/precedence rules, override examples, source review procedure, and client integration example. Prompt 08's corrected limits are reused under `metadata.router.limits`; their existing fields and overrides retain their semantics.

### Discovery refresh and lifecycle

The router makes **one synchronous discovery attempt per configured source before listening**, preserving initial inventory and `--check` behavior. After startup, a managed background worker refreshes catalog and llama.cpp discovery on independent schedules. Model listing, resolution, and `/health` / `/healthz` only read cached state: they never initiate or wait for discovery network work. Liveness continues to return `status = "ok"` during discovery outages; it does not certify upstream readiness.

After a successful round, the next attempt is due `refresh_seconds` after completion. After a failed round, retries use exponential backoff: the ceiling starts at 5 seconds, doubles with each consecutive failure, and caps at 300 seconds. Each delay is chosen uniformly between half the ceiling and the ceiling. Retries are measured from completion, independently of the last success. Recovery resets the failure count and restores the configured refresh interval. Overlapping refresh calls for the same source return immediately, including explicit forced calls. Request bursts cannot trigger extra attempts.

For llama.cpp, servers are probed concurrently (up to eight); successful servers publish their new inventories while failed servers retain their own last usable routes. Any server failure counts as a failed discovery round for scheduling, even if others succeed. The source's `last_success` records the last **entirely successful round**; each server also reports its own success and attempt times. Optional `/props` remains best effort. The single scheduling worker processes llama.cpp and catalog in turn, so an active source can delay the other until its probes finish under the configured HTTP timeouts.

Set a source's `refresh_seconds = 0` for **startup-only** mode. This now means exactly one startup attempt, even when that attempt fails; there are no automatic retries for that source. Set a positive interval to discover servers started later or recover automatically from startup failure. If discovery has never succeeded, its inventory is empty: explicit routes and other successful sources remain available, unknown models return 404, and an unreachable llama.cpp server has no active provider for `provider/model` routing. A partially successful first llama.cpp round exposes the successful servers and reports `loaded = true` even though the round's `last_success` is still null.

The existing health responses include `catalog.refresh` and `llama_cpp.refresh`, atomically associated with their inventories:

| Field | Meaning |
| --- | --- |
| `last_success`, `last_attempt`, `next_attempt` | Process-local monotonic timestamps; null means no success, no attempt, or no scheduled retry respectively. They are not Unix timestamps and cannot be compared across restarts. |
| `consecutive_failures` | Failed rounds since the last complete success. |
| `age_seconds` | Time since last success; null if never successful. |
| `stale` | No successful round, a failed round, or the normal refresh interval has elapsed. A successful startup-only inventory does not expire. |
| `stale_age_seconds` | Time beyond the normal refresh interval since last success, floored at zero; null if never successful. Startup-only inventories report zero. |

Errors remain sanitized and length-limited. Per-server llama.cpp status includes `last_success`, `last_attempt`, `age_seconds`, and `stale_age_seconds`; its existing `stale` flag identifies retained routes after a failed probe. Freshness ages are derived at read time from the captured immutable snapshot, so they keep increasing during a blocked refresh. New timestamps, retry state, errors and inventories publish together after the attempt completes; readers keep the previous complete generation while it is in progress.

For embedders, `App(config)` performs the initial attempt; `app.start()` starts scheduling and is idempotent. `serve()` starts it after binding, and the HTTP server's `shutdown()` / `server_close()` stop it. `app.stop()` joins the worker and permits a later `start()`; `app.close()` is idempotent, permanently stops scheduling, waits for active discovery, closes HTTP resources, and removes diagnostic filters. Always call `close()` in a `finally` block. Shutdown waits for in-flight probes to finish under the configured timeouts, then skips remaining sources; it does not abandon probe threads. `--check` starts no background worker. Refresh intervals and discovery timeouts must be finite and within `threading.TIMEOUT_MAX` (timeouts positive; intervals nonnegative).

**Compatibility:** requests no longer discover a new model synchronously; it becomes available after the next scheduled round. Failed startup-only discovery no longer retries automatically. Last usable routes are retained in memory without an expiry, so generation can still fail against an unavailable upstream. No discovery state is persisted across restarts.

### Router auth

Set `server.api_keys` (or `MODEL_ROUTER_API_KEY` / comma-separated `MODEL_ROUTER_API_KEYS`, or repeat `--api-key`). Clients may send `Authorization: Bearer <key>`, `x-api-key: <key>`, or `api-key: <key>`. With no keys configured, auth is disabled — bind to localhost or put the router behind something that authenticates.

## Behavior notes and limits

- **Translation** uses LiteLLM adapters with the strict/lossy policy below. Native features do not all have equivalent mappings. Token *counts* and error wording differ between dialects; the response shape follows the ingress dialect.
- **Anthropic ingress requires `max_tokens`**, as the Anthropic API does.
- **Reasoning / encrypted content** survives only on a same-dialect pass-through. Crossing protocols necessarily re-encodes it (or drops it, if the target has no equivalent).
- **`sequence_number`** is added to translated Responses streams so OpenAI clients see a monotonic counter.
- Request bodies accept one unsigned decimal `Content-Length` or a single `Transfer-Encoding: chunked`. Conflicting framing, duplicate length/transfer headers (even identical ones), transfer-coding lists, signed sizes, malformed chunk extensions, missing CRLF delimiters, and truncated bodies return HTTP 400. Chunk extensions are validated and ignored; valid trailers are discarded and cannot override framing, representation, or credential headers.
- `max_body_bytes` bounds both the encoded payload and the decompressed body. Oversized bodies or chunk metadata return HTTP 413. Each chunk-size/extension or trailer line is limited to 1,024 bytes including CRLF; all chunk metadata (size lines, extensions, delimiters, and trailers) is limited to 64 KiB per request, with at most 100 trailer fields. These framing limits also apply to small-chunk requests.
- Gzip (`Content-Encoding: gzip`) must contain **exactly one complete member**, with a valid checksum and length trailer. Concatenated members and all trailing bytes (including zero padding) return HTTP 400. Other content encodings or encoding lists return HTTP 415. Ordinary gzip and chunked gzip requests are supported.
- Compatibility change (prompt 03): requests are decoded, parsed, and serialized as UTF-8 JSON with the upstream model ID. Outgoing `Content-Type` is `application/json`; the transport computes `Content-Length`. Original encoding, range/location/language metadata, validators, digests, and signatures are discarded because they describe the original bytes. Static provider headers and translation header overrides cannot override this framing policy. Vendor JSON fields and unrelated protocol headers retain their existing behavior.
- The router is synchronous per request (bounded `ThreadingHTTPServer` + LiteLLM's sync APIs); long streams hold one admitted thread and generation slot each. It is built for a handful of local clients, not as a multi-tenant gateway.
- No request logging of prompts or completions; `--verbose` enables LiteLLM/httpx debug logging.

## Tests

```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest
```

The suite starts a fake upstream, a fake llama.cpp server and the real router in-process. It exercises every ingress→upstream combination (streaming and non-streaming), pass-through fidelity, tool-schema translation, auth, session-header forwarding, catalog discovery, llama.cpp discovery across single/router/legacy modes (context sizes, aliases, prefixes, stale routes), gzip/chunked bodies, dynamic routing, and model listing. Bundled Go routing tests use checked-in public catalog/endpoint fixtures, fake credentials and a blocked network transport; tests never fetch the live catalog or perform cloud inference. See `tests/fixtures/README.md` for the mapping review procedure. The fake upstream honors request content encoding. Registry tests cover duplicate explicit names, generated-name collisions through recovery/failure, route precedence and stale inventories. Events and barriers pause requests between resolution and dispatch, pause model/health reads across refresh, and coordinate concurrent catalog/local publications without sleep-based races. Raw-socket and bounded-reader tests cover malformed framing, truncated bodies, chunk metadata limits, gzip integrity/completion, and encoded/decompressed size limits. Capacity tests use local sockets, controlled upstreams, and held generations to cover admission before thread creation, absolute read deadlines, stalled response readers, sync/async stream disconnects, overload recovery, and shutdown.

`tests/test_discovery_refresh.py` uses a fake monotonic clock and controlled local HTTP responses to cover success/failure/recovery, bounded jitter and backoff, partial outages, startup-only failures, independent source schedules, cached liveness/listing/resolution during request bursts, overlapping refresh calls, and worker/transport cleanup. Events hold responses open and observe worker idle boundaries without sleep-based races.

## CLI

```
python model_router.py [-c CONFIG] [--host HOST] [--port PORT]
                       [--api-key KEY ...] [--env-file PATH] [--no-default-env]
                       [--check] [--verbose] [--version]
```

`--check` validates the config, makes one initial attempt for each configured discovery source, and prints the routes without listening or starting a background worker.
