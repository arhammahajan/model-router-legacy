# Harness integrations

`integrations/` holds client-side adapters that point a coding harness at a running router. They read `/v1/models` and its `metadata` (see [capabilities](capabilities.md)), so change them in the same commit as any change to that contract. They aren't part of the Python package. Each harness gets its own directory.

## Muse

Muse needs no adapter: point it directly at the router's `/v1` base URL. `GET /muse-code/models` and `GET /v1/muse-code/models` serve a Muse-compatible catalog from the same authenticated, cached inventory as `/v1/models`. Entries add `metadata["muse-code"]` with a display label and context/output limits from the router's capability metadata (unknown limits fall back to 128000/16384 tokens), and list every available route regardless of upstream protocol. Generation uses the existing `/v1/responses` endpoint.

## pi

`integrations/pi/model-router.ts` registers the router's models as a pi provider, discovered at startup and on refresh. Each model uses its upstream's native protocol, so requests pass through untouched.

Install by symlinking, so edits take effect:

```bash
ln -s "$PWD/integrations/pi/model-router.ts" ~/.pi/agent/extensions/model-router.ts
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_ROUTER_URL` | `http://127.0.0.1:4000` | Router origin |
| `MODEL_ROUTER_API_KEY` | none | API key, if the router requires one |

It reads `metadata.dialect`, `metadata.llama_cpp`, `metadata.context_window` and `metadata.max_tokens` (llama.cpp), and under `metadata.router`: `limits`, `modalities.input.image`, `reasoning.supported`, `reasoning.effort.supported` and `reasoning.effort.values`. Unknown limits fall back to pi's defaults (128000 context, 16384 output).

What the extension still does, and why the router cannot do it instead:

| Extension responsibility | Why it stays |
| --- | --- |
| Registers models and sets `api`, `baseUrl`, `reasoning`, `input`, limits | pi builds its model list only from `registerProvider`; a catalog endpoint is read by nothing in pi unless something registers it. Native protocols (`anthropic-messages`, `openai-responses`) are chosen per model so those requests pass through. |
| `thinkingLevelMap` from `reasoning.effort.values` | pi offers `minimal`..`high` unless a level is `null`, and `xhigh`/`max` only when mapped. The router can refuse a level, never hide one, so the levels pi offers must match the published values. This covers llama.cpp (`none` → off, `xhigh`) without a llama-specific branch. |
| Gemini routes without published `effort.values` stay non-reasoning | Effort is exact only as a native `thinkingLevel`, which the router knows only for routes that publish their levels; otherwise it is a token-budget estimate that a strict router refuses. The router publishes no policy fact a client could use instead. |
| `forceAdaptiveThinking` for `claude-*` | pi encodes Anthropic thinking before the request exists, and Anthropic routes are pass-through by design. |
| `x-session-id` from the pi session | Conversation identity belongs to the client and is never derived from prompts or credentials. pi's Chat adapter sends none by default, and its `sendSessionAffinityHeaders` compat would have to be set on every model instead. |
| Reading `/v1/models` at startup and on refresh | pi has no discovery of its own for this provider. |

What the router now handles, so the extension no longer has to: Chat `store: false` (the Chat default) needs no mapping on Gemini, Anthropic or llama.cpp; `stream_options.include_usage` is implemented for every destination; and `low` / `medium` / `high` reach a published, thinking-level Gemini route as native levels without a loss report.

`tests/test_pi_integration.py` runs the installed pi client's own request builders against an in-process router and fake upstreams (it skips when node or pi is missing; `PI_AI_DIR` overrides the pi location).

Requests include pi's durable session ID as `x-session-id`; the router maps it to the upstream's configured conversation-affinity header.
