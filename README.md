# litellm-model-router

A small HTTP router that accepts **OpenAI Chat**, **OpenAI Responses** and **Anthropic Messages** requests and forwards them to any upstream that speaks one of those protocols or Gemini. Local **llama.cpp** servers are discovered automatically.

```
   /v1/chat/completions ─┐                 ┌─ openai-chat
   /v1/responses         ─┼─> model-router ─┼─ openai-responses
   /v1/messages          ─┘                 ├─ anthropic
                                            └─ gemini
```

When the request protocol matches the upstream's, the request and response pass through untouched. Otherwise LiteLLM translates, and the router refuses to silently drop fields it can't carry over (set `compatibility = "lossy"` on a provider to allow it). Any ingress can reach any upstream, streaming or not. Gemini is upstream-only.

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
| `GET /v1/models` | Model list with capability and limit metadata |
| `GET /livez` | Liveness |
| `GET /readyz` | Readiness (503 until a usable route exists) |
| `GET /diagnostics` | Providers and discovery status; needs router auth |

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

llama.cpp has no `reasoning_effort`, so the router maps it to the native fields:

| Effort | Sent to llama.cpp |
| --- | --- |
| `none` | `chat_template_kwargs.enable_thinking = false` |
| `minimal` `low` `medium` `high` `xhigh` | `thinking_budget_tokens` = 1024 / 2048 / 8192 / 16384 / 32768 |
| `max` or anything else | nothing; the server default applies |

A client-supplied `thinking_budget_tokens` or `enable_thinking` always wins. This covers Chat and Responses requests.

## Model capabilities

`/v1/models` includes `metadata.router`: reasoning and effort levels, input and output modalities, tools, structured output, sampling controls, limits, and where each fact came from. `false` means unsupported and `null` means unknown. See [docs/capabilities.md](docs/capabilities.md).

## Harness integrations

[`integrations/`](integrations) holds client-side adapters that point a coding harness at the router. Currently there is a pi extension; see [docs/integrations.md](docs/integrations.md).

## Development

```bash
.venv/bin/python -m pytest
```

See [docs/contributing.md](docs/contributing.md) for CI, dependency pins and upgrade testing. The code is in `model_router/`, with capability metadata in `router_metadata/`.
