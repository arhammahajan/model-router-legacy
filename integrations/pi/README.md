# pi extension

`model-router.ts` discovers models from the router's `/v1/models` and registers
them as the `model-router` provider, replacing a hand-written `models.json`
entry. Each model uses its upstream's native protocol, so requests pass through
untouched.

## Install

Symlink it so edits here take effect:

```bash
ln -s "$PWD/integrations/pi/model-router.ts" ~/.pi/agent/extensions/model-router.ts
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_ROUTER_URL` | `http://127.0.0.1:4000` | Router origin |
| `MODEL_ROUTER_API_KEY` | none | Sent as the API key if the router requires one |

## Metadata it relies on

`metadata.dialect`, `metadata.llama_cpp`, `metadata.context_window` and
`metadata.max_tokens` (llama.cpp), and under `metadata.router`: `limits`,
`modalities.input.image`, `reasoning.supported` and `reasoning.effort.supported`.
Unknown limits fall back to pi's defaults (128000 context, 16384 output).

llama.cpp thinking levels are sent as `reasoning_effort` and mapped to native
fields by the router, so no thinking logic lives here.
