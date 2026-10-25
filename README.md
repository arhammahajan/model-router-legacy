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

The router fetches `https://opencode.ai/inference/go/v1/models` at startup, registers every model under its upstream ID, and refreshes the list every five minutes. Routing rules send:

- `claude-*` to OpenCode's Anthropic endpoint,
- `gemini-*` to its Google/Gemini endpoint,
- everything else to its OpenAI Responses endpoint.

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
    model="deepseek-v4.1-flash",        # routed to OpenCode's Responses endpoint
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
| `drop_params` | Drop parameters the target does not support instead of erroring (default `true`) |

Upstream credential selection has two behaviors:

- A nonempty `api_key`, `api_key_env`, or `api_key_file` selects that configured key. Surrounding whitespace is stripped. The top-level provider key owns LiteLLM's `api_key` even if `litellm_kwargs` repeats it; the selected auth header also wins case-insensitive header conflicts. Other explicitly configured headers remain trusted. Adapters may add their native credential header as well as an `auth_header` override, using the same selected key.
- An omitted, empty, or whitespace-only key intentionally selects the public placeholder **`model-router-no-auth`**. Both passthrough and translation send it using the provider's native auth mechanism (or configured `auth_header`). An unauthenticated server must accept/ignore this placeholder; servers that reject any auth header are not supported. No environment or LiteLLM global provider key is used as a fallback.

This avoids [LiteLLM's environment/global credential sources](https://docs.litellm.ai/docs/set_keys) without clearing or swapping process-wide environment variables during requests. Only named `api_key_env` settings opt into environment credentials. LiteLLM's `os.environ/NAME` key indirection is rejected; use `api_key_env` instead. Global LiteLLM `proxy_auth`, `client_session`, or `aclient_session`, which can inject auth after argument merging or through client defaults, cause translated calls to fail closed without modifying those globals. Keep custom credential-bearing clients/transports out of `litellm_kwargs`; they are trusted administrative code outside this native-adapter credential contract. Client-visible router auth is separate and never forwarded.

Compatibility change (prompt 02): keyless passthrough calls now send the same placeholder as translation, replacing absent auth headers or the old `not-needed` placeholder. Existing nonempty configured keys continue to work. Raw HTTP error bodies are replaced as described above. These isolation regressions are tested against installed LiteLLM 1.106.4 with local upstreams and dummy environment/global keys for Anthropic, OpenAI Chat/Responses, and Gemini.

### Models

`name` is the public route; `aliases` are additional public names. `dialect` overrides the provider default if one endpoint exposes both protocols. `enabled = false` hides a route and makes requests for it resolve as not found.

If a request names an unlisted model as `provider/model`, the router forwards it to that provider dynamically using the provider's default dialect — handy for trying models without editing config:

```json
{"model": "local/qwen2.5-coder:7b", "messages": [...]}
```

### Dynamic catalogs

A catalog fetches model IDs from an OpenAI-style `/v1/models` endpoint and registers each one under its upstream ID:

```toml
[catalog]
url = "https://opencode.ai/inference/go/v1/models"
refresh_seconds = 300          # 0 = startup only
headers = { "User-Agent" = "litellm-model-router/0.2.0" }

# Regex rules matched against each model ID; the first match wins.
[[catalog.rules]]
match = "^claude-"
provider = "opencode-anthropic"

[[catalog.rules]]
match = "^gemini-"
provider = "opencode-gemini"

[[catalog.rules]]
match = "."
provider = "opencode-openai"
```

Catalog failures are logged and retried without stopping the server; existing routes keep working.

### Local llama.cpp discovery

The router can probe running [llama.cpp](https://github.com/ggml-org/llama.cpp) servers and expose every model they report. This ports the discovery behavior of [pi-llama-cpp](https://github.com/gsanhueza/pi-llama-cpp): single-model, router (`--models-preset`), legacy (`ik_llama.cpp`) and [llama-swap](https://github.com/mostlygeek/llama-swap) servers are recognized.

A config needs explicit `[[models]]`, a `[catalog]`, a `[llama_cpp]`, or any combination. Explicit routes win when a name collides, then discovered llama.cpp models, then catalog models.

```toml
[llama_cpp]
servers = ["http://127.0.0.1:9931"]   # URLs or { url, name, api_key, headers }
# ports = [9931, 8080]                 # convenience: localhost URLs
refresh_seconds = 30                   # 0 = startup only
timeout = 2                            # per-probe timeout
context_fallback = 128000              # when the server reports no context size
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

Discovery reads:

| Endpoint | Used for |
| --- | --- |
| `/health` | reachability (the server is skipped when this fails) |
| `/v1/models` | model IDs, aliases, `meta.n_ctx`, `status`, `architecture` |
| `/props?autoload=false` | server mode (`role: "router"` / `src: "llama-swap"`) |

Context size follows pi-llama-cpp's rules:

- **Router mode** — `meta.n_ctx` when loaded; otherwise `--ctx-size` / `--fit-ctx` / `-c` from the model's launch arguments.
- **Single and llama-swap** — `meta.n_ctx`.
- **Legacy** — `max_model_len`, falling back to `n_ctx` from `/props`.
- Otherwise `context_fallback` (default 128000).

Each model appears in `/v1/models` with its display name (first alias), `context_window`, `max_tokens`, `capabilities` (`text`, `image`), and a `metadata.llama_cpp` object containing the server URL, detected mode and load status. Aliases resolve to the same upstream route unless `aliases = false`.

If a server becomes unreachable during a refresh, its last known routes are kept and the failure is reported in `/health` (`llama_cpp.servers[].reachable = false`, `stale = true`). A server that was never reachable is simply absent. The router itself keeps serving.

### Router auth

Set `server.api_keys` (or `MODEL_ROUTER_API_KEY` / comma-separated `MODEL_ROUTER_API_KEYS`, or repeat `--api-key`). Clients may send `Authorization: Bearer <key>`, `x-api-key: <key>`, or `api-key: <key>`. With no keys configured, auth is disabled — bind to localhost or put the router behind something that authenticates.

## Behavior notes and limits

- **Translation** is LiteLLM's, not a hand-rolled field mapper. Parameters unsupported by the target are dropped when `drop_params` is true (the default) and rejected otherwise. Token *counts* and error wording differ slightly between dialects; the response shape is always the ingress dialect's.
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

The suite starts a fake upstream, a fake llama.cpp server and the real router in-process. It exercises every ingress→upstream combination (streaming and non-streaming), pass-through fidelity, tool-schema translation, auth, session-header forwarding, catalog discovery, llama.cpp discovery across single/router/legacy modes (context sizes, aliases, prefixes, stale routes), gzip/chunked bodies, dynamic routing, and model listing. The fake upstream honors request content encoding. Raw-socket and bounded-reader tests cover malformed framing, truncated bodies, chunk metadata limits, gzip integrity/completion, and encoded/decompressed size limits. Capacity tests use local sockets, controlled upstreams, and held generations to cover admission before thread creation, absolute read deadlines, stalled response readers, sync/async stream disconnects, overload recovery, and shutdown.

## CLI

```
python model_router.py [-c CONFIG] [--host HOST] [--port PORT]
                       [--api-key KEY ...] [--env-file PATH] [--no-default-env]
                       [--check] [--verbose] [--version]
```

`--check` validates the config, refreshes the catalog, and prints the routes without listening.
