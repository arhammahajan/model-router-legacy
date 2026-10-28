# litellm-model-router

A small HTTP router that accepts **OpenAI Chat**, **OpenAI Responses**, **Anthropic Messages** and **Gemini API** (`generateContent`) requests and forwards them to any upstream that speaks one of those protocols. Local **llama.cpp** servers are discovered automatically.

```
   /v1/chat/completions ─────┐                 ┌─ openai-chat
   /v1/responses         ─────┤                 ├─ openai-responses
   /v1/messages          ─────┼─> model-router ─┼─ anthropic
   /v1beta/models/*:generateContent ─┘          └─ gemini
```

When the request protocol matches the upstream's, the request and response pass through untouched. Otherwise the request is translated, and the router refuses to silently drop fields it can't carry over (set `compatibility = "lossy"` on a provider to allow it). Any ingress can reach any upstream, streaming or not. OpenAI and Anthropic requests are translated by LiteLLM; Gemini requests by the router itself (see [Gemini API ingress](#gemini-api-ingress)).

## Quick start

```bash
python -m venv .venv
.venv/bin/python -m pip install -c constraints.txt -r requirements.txt
.venv/bin/python -m model_router          # http://127.0.0.1:4000
```

Python 3.10–3.14. To get a `model-router` command instead, run `pip install -c constraints.txt .`.

The bundled `router.toml` targets an [OpenCode Go](https://opencode.ai) subscription. Put `OPENCODE_API_KEY=...` in `~/.config/ai/credentials.env` (loaded automatically; `--env-file PATH` or `--no-default-env` override this). For your own providers, copy `config.example.toml` and edit it.

## Using it

Point any SDK at the router. The URL path picks the request protocol and the model name picks the upstream.

```bash
curl http://127.0.0.1:4000/v1/models

curl http://127.0.0.1:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"claude-haiku-5-5","messages":[{"role":"user","content":"Hi"}]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:4000/v1", api_key="unused")
client.chat.completions.create(model="gpt-6-luna", messages=[{"role": "user", "content": "Hi"}])
```

```python
from anthropic import Anthropic
client = Anthropic(base_url="http://127.0.0.1:4000", api_key="unused")
client.messages.create(model="glm-5.3", max_tokens=256, messages=[{"role": "user", "content": "Hi"}])
```

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI Chat ingress |
| `POST /v1/responses` | OpenAI Responses ingress |
| `POST /v1/messages` | Anthropic Messages ingress |
| `POST /v1beta/models/{model}:generateContent` | Gemini ingress (also under `/v1/` or with no version prefix) |
| `POST /v1beta/models/{model}:streamGenerateContent` | Gemini streaming ingress (`?alt=sse` for SSE, otherwise a JSON array) |
| `GET /v1beta/models`, `GET /v1beta/models/{model}` | Gemini-format model discovery (`models.list` / `models.get`) |
| `POST /v1/messages/count_tokens` | Exact Anthropic token count, for providers that implement it (see below) |
| `GET /v1/models` | Model list with capability and limit metadata |
| `GET /v1/muse-code/models` (also `/muse-code/models`) | Muse-compatible full model catalog and limits |
| `GET /livez` | Liveness |
| `GET /readyz` | Readiness (503 until a usable route exists) |
| `GET /diagnostics` | Providers and discovery status; needs router auth |

### Token counting

`POST /v1/messages/count_tokens` takes Anthropic's request shape (`model`, `messages`, `system`, `tools`, `tool_choice`, `thinking`, ...; `max_tokens` is not required) and returns `{"input_tokens": n}`. It uses the same router auth, body limits, deadline and retry policy as generation, but it is not generation: it takes no generation slot and never calls a generation endpoint.

- **Supported:** a route whose provider speaks `dialect = "anthropic"` and is known to implement the endpoint. The request is forwarded to `<base_url>/v1/messages/count_tokens` with the resolved upstream model ID and the provider's own credential, and the upstream's answer is returned as-is. Every other field is forwarded untouched, so tools, thinking, `cache_control` and base64 images/PDFs follow the upstream's contract.
- **Which providers:** Anthropic-compatible does not imply token counting. Only `https://api.anthropic.com` is assumed to support it; any other Anthropic-protocol provider must opt in with `count_tokens = true` after you confirm it has the endpoint. `count_tokens = false` turns it off. The setting is rejected on other dialects.
- **Unsupported:** OpenAI Chat, OpenAI Responses and Gemini routes, and undeclared Anthropic providers, get HTTP 400 `invalid_request_error` ("Exact token counting is unavailable for this model...") and the upstream is not contacted. The router has no tokenizer for these models, so it does not estimate; clients that need a number must estimate themselves and treat it as one.
- **Upstream limits pass through:** Anthropic rejects server tools, the MCP connector and `url`/`file` image or document sources on this endpoint, and documents the count as an estimate that can differ slightly from billed usage. A 2xx that is not `{"input_tokens": <int>}` becomes a 502, and responses over 64 KiB are refused.

### Gemini API ingress

Point a Gemini SDK at the router and use a router model name as the model:

```python
from google import genai
client = genai.Client(api_key="router-key-or-anything",
                      http_options={"base_url": "http://127.0.0.1:4000"})
client.models.generate_content(model="claude-haiku-5-5", contents="Hi")
```

- **Paths and names.** `/v1beta/models/{model}:generateContent` and `:streamGenerateContent`, as the SDKs call them. A `models/` prefix is optional, `provider/model` names may be sent with an unescaped or percent-encoded `/`, and `/v1/...` or an unversioned path also work. `tunedModels/...`, `countTokens`, `embedContent` and other methods are not implemented (404/501 in Gemini's error shape). Discovery is `GET /v1beta/models` (`pageSize`, `pageToken`) and `GET /v1beta/models/{model}`; token limits are reported only when known. `GET /v1/models` keeps its OpenAI/Anthropic shape.
- **Same-protocol routes** (`dialect = "gemini"`) are forwarded as JSON: the body is untouched, the model and streaming mode move into the upstream URL, and an `alt=sse` selector is the only query parameter forwarded. Upstream errors keep their status and Google error status name.
- **Other routes** translate through the Chat Completions bridge and answer with `GenerateContentResponse` / SSE frames (`data: {...}`, no terminator; function calls are delivered whole; `usageMetadata` and `finishReason` arrive on the final frame). Requests may use camelCase or snake_case, as on the public API. Unknown request fields are rejected.
- **Authentication.** The router's own `api_keys` are accepted as `x-goog-api-key`, `?key=`, `Authorization: Bearer` or `x-api-key`, and failures are Gemini `UNAUTHENTICATED` errors. A client key is never forwarded: the upstream credential always comes from the provider config, `key=` never reaches an upstream URL or a log, and `x-goog-user-project` / `x-goog-quota-user` are dropped. This is API-key access to the Gemini *Developer API* surface only. It does not speak Google OAuth, Vertex AI paths or Antigravity's account-based protocol, so Antigravity cannot be moved onto it by supplying an API key.

What translation does with Gemini features (strict providers refuse anything in the *reported* column; lossy providers apply it and send the `x-model-router-loss` header):

| Gemini feature | Translated | Reported |
| --- | --- | --- |
| text, images (png/jpeg/webp/gif), PDFs, wav/mp3 audio inline | to each protocol's native form (audio: Chat only) | other media types, `fileData` references, video |
| `systemInstruction`, history, `maxOutputTokens`, `temperature`, `topP` | exact | |
| `stopSequences`, `topK`, `seed`, penalties | where the destination defines them | elsewhere |
| function declarations, calls and responses | exact; OpenAPI-subset schemas become JSON Schema | `behavior`, function response schemas |
| `toolConfig` AUTO / ANY / NONE, one allowed name | exact | several allowed names and `VALIDATED` (approximated) |
| `googleSearch`, `codeExecution`, `urlContext`, `googleMaps`, ... | | dropped per tool; function tools next to them survive |
| `responseMimeType` + `responseSchema` / `responseJsonSchema` | JSON mode / JSON schema | on destinations without a native structured-output contract |
| `thinkingBudget` > 0 | exact budget where the destination has one | elsewhere |
| `thinkingBudget` 0, `thinkingLevel` | disable / the named level, approximated by the destination | levels it cannot express |
| `thinkingBudget` -1 | the destination's default (no control sent) | |
| `includeThoughts` | reasoning text returned as `thought` parts | |
| `safetySettings` | | always (no equivalent exists) |
| `candidateCount` > 1, logprobs, non-text modalities | | always |
| `cachedContent`, `continuationToken` | | rejected even in lossy mode: they point at state held by Gemini |

Thought summaries and `thoughtSignature` values in sent history are provider-private reasoning state and are removed without a loss record.

### Conversation affinity

Providers with `session_header` use an explicit upstream session header first,
then these client-owned headers in order: `x-claude-code-session-id`, `session-id`,
`session_id`, `thread-id`, `conversation_id`, `x-session-id`, and
`x-mavis-session-id`. A configured `session_source_headers` list can narrow or
reorder that set. `session_prompt_cache_key = true` enables a final body fallback.
No identity is derived from credentials or prompt text.

The MCode header was verified on 2026-10-27 against installed MCode 0.6.13's
`LocalModelResolver`: it sets `X-Mavis-Session-Id` from the durable session ID
for custom providers. Pi's integration sends its durable session ID as
`x-session-id`. These identities are consumed and remapped to the configured
upstream header rather than forwarded as unrelated request headers.

## Configuration

A config file is TOML. Strings support `${ENV_VAR}`; missing variables are errors. The main tables:

| Table | Purpose |
| --- | --- |
| `[server]` | Host, port, client auth (`auth_mode = "none"` or `"required"` with `api_keys`), body and connection limits |
| `[providers.NAME]` | An upstream: `dialect`, `base_url`, `api_key`, optional headers and timeout |
| `[[models]]` | An explicit route from a public model name to a provider |
| `[catalog]` | Pull a model list from a remote endpoint and map each model to a provider |
| `[llama_cpp]` | Discover local llama.cpp servers |
| `[retry]` | Off by default; opt in with `max_attempts` |
| `[capabilities]` | Override reported model capabilities |
| `[discovery_cache]` | Optional on-disk cache of discovered models |

Names resolve in this order: explicit routes, then llama.cpp models, then catalog models. `config.example.toml` documents every option.

```bash
model-router --validate-config     # check the config, no network
model-router --check-readiness     # one discovery pass; exit 0 ready, 3 not ready
```

## llama.cpp

Running servers are probed (default `http://127.0.0.1:9931`, or `LLAMA_SERVER_URL`), and every model they report is routed on its own, with aliases, context size and capabilities. Single-model, router-mode, `ik_llama.cpp` and [llama-swap](https://github.com/mostlygeek/llama-swap) servers are recognized. Servers started later are picked up on the next refresh.

```toml
[llama_cpp]
servers = ["http://127.0.0.1:9931"]
refresh_seconds = 30
context_fallback = 128000     # used only when the server reports no context size
```

Each model's `metadata.limits` separates `context_window`, `trained_context_window`, `max_input_tokens`, `max_output_tokens` and `default_output_tokens`. Every field carries its value, source and kind (`reported`, `override`, `inferred`, `fallback` or `unknown`). Pin a value with `[llama_cpp.limit_overrides."model-id"]`. If a server goes down, its last known routes stay and `/diagnostics` reports it as stale.

### Thinking control

llama.cpp has no OpenAI-style effort ladder, so the router maps to its native fields:

| Request | Sent to llama.cpp | Kind |
| --- | --- | --- |
| effort `none`, or Anthropic `thinking: {type: disabled}` | `chat_template_kwargs.enable_thinking = false` | exact |
| Anthropic `thinking: {type: enabled, budget_tokens: N}` | `thinking_budget_tokens = N` | exact |
| effort `minimal` `low` `medium` `high` `xhigh` | `thinking_budget_tokens` = 1024 / 2048 / 8192 / 16384 / 32768 | router-defined table |
| effort `max` or any other level | nothing (see below) | unsupported |

The effort table is the router's own convention, not a llama.cpp or OpenAI equivalence. On native Chat requests `max` or an unknown level sends nothing and the server default applies. On translated requests it is rejected, or dropped and reported under `compatibility = "lossy"`. A client-supplied `thinking_budget_tokens` or `enable_thinking` always wins on native Chat requests.

## Reasoning translation

Reasoning controls are mapped as a unit, never by letting an adapter pick its own table. Each control is **exact** (same meaning, applied silently), **approximated** (a mapping exists but means something different) or **unsupported** (the destination has no such control). Strict providers reject the last two before any network call; lossy providers apply them and list each in `x-model-router-loss` as `approximated` or `dropped`. Level names are never renamed (`minimal` never becomes `low`), a value outside the route's published effort values is never clamped, and an explicit choice to disable thinking is carried or reported, never lost. Same-protocol routes pass through untouched.

| Control | Destination | Result |
| --- | --- | --- |
| effort `none` / `thinking: disabled` | Anthropic, Chat, Responses, llama.cpp | exact (`thinking: disabled`, `reasoning_effort: none`, `enable_thinking: false`) |
| | Gemini | exact when the route's `reasoning.toggle.supported` is true; otherwise approximated (some models cannot fully disable thinking) |
| Chat effort ↔ Responses effort | Responses ↔ Chat | exact, one enum |
| Anthropic `budget_tokens` | llama.cpp, Gemini (`thinkingBudget`) | exact |
| | Chat, Responses | approximated: bucketed into an effort level |
| effort level | Anthropic | approximated: a token budget or adaptive effort, depending on the model |
| | Gemini `low` `medium` `high` | exact when the route publishes the level and the adapter sends it as a native `thinkingLevel` (models newer than Gemini 2); otherwise approximated as a thinking budget |
| | Gemini `minimal` | approximated: the adapter turns it into a budget, or into `low` on models without a `minimal` level; `xhigh` and `max` are unsupported |
| Anthropic `output_config.effort` | Chat, Responses | approximated, and only for an identically named level in the route's values (`low` `medium` `high` when none are published) |
| | Gemini, llama.cpp | unsupported |
| `thinking: adaptive` | anything but Anthropic | unsupported |
| `thinking.display`, `reasoning.summary`, `reasoning.generate_summary` | anything without an equivalent | unsupported |
| `store`, `include` (Responses) | Chat, Anthropic, Gemini, llama.cpp | unsupported: retention and returned fields follow the upstream |
| `store: false` (Chat) | Anthropic, Gemini, llama.cpp | exact: Chat stores nothing unless asked, so there is nothing to carry. `store: true` is unsupported |

Known capability facts decide: `reasoning.supported = false` makes a disable request already satisfied and any other control a loss, `reasoning.toggle.supported = false` rejects a disable, and `reasoning.effort.values` or `reasoning.budget.*` bounds reject what falls outside them. Unknown facts stay neutral. Malformed `thinking` objects (missing or unknown `type`, a budget on a disabled or adaptive object, a missing budget) are rejected rather than defaulted, and an effort cannot be combined with a `thinking` object in one request. Diagnostics name fields and reasons only, never prompt text or credentials.

Limits: the router cannot send Anthropic `output_config.effort` to a model LiteLLM does not know supports it, so effort for such Anthropic-protocol routes stays an approximation. Whether a llama.cpp build honours `thinking_budget_tokens` is not verified by the router; the current llama.cpp server documents per-request `reasoning_effort` and `chat_template_kwargs` but not that field.

## Model capabilities

`/v1/models` includes `metadata.router`: reasoning and effort levels, input and output modalities, tools, structured output, sampling controls, limits, and where each fact came from. `false` means unsupported and `null` means unknown. See [docs/capabilities.md](docs/capabilities.md).

## Harness integrations

[`integrations/`](integrations) holds client-side adapters that point a coding harness at the router. Currently there is a pi extension; see [docs/integrations.md](docs/integrations.md).

Muse connects directly to `http://127.0.0.1:4000/v1` and needs no adapter. The
router serves a Muse-compatible catalog at `/muse-code/models` and
`/v1/muse-code/models` from the same authenticated, cached inventory as
`/v1/models`, and generation uses the existing `/v1/responses` endpoint.

The bundled `router.toml` opts its providers and discovered llama.cpp servers into
`compatibility = "lossy"` so harnesses can use every model. Nonportable controls
are dropped or approximated with `x-model-router-loss` diagnostics (see
[Reasoning translation](#reasoning-translation)); native calls pass through unchanged. Custom configurations still default to strict translation.
Model visibility does not guarantee every upstream-native feature exists in all
protocols: stored Responses state and provider-native built-in tools remain
protocol-specific. Ordinary function tools translate across the protocols,
including Responses function namespaces.

## Development

```bash
.venv/bin/python -m pytest
```

See [docs/contributing.md](docs/contributing.md) for CI, dependency pins and upgrade testing. The code is in `model_router/`, with capability metadata in `router_metadata/`.
