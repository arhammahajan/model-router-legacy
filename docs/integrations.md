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

It reads `metadata.dialect`, `metadata.llama_cpp`, `metadata.context_window` and `metadata.max_tokens` (llama.cpp), and under `metadata.router`: `limits`, `modalities.input.image`, `reasoning.supported` and `reasoning.effort.supported`. Unknown limits fall back to pi's defaults (128000 context, 16384 output). llama.cpp thinking levels are sent as `reasoning_effort` and mapped by the router, so no thinking logic lives in the extension.

Requests include pi's durable session ID as `x-session-id`; the router maps it to the upstream's configured conversation-affinity header.
