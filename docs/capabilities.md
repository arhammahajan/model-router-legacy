# Model capabilities

`GET /v1/models` lists each model with `metadata.router`, a versioned description of what the model supports. It is **advisory**: nothing here configures an SDK, enables a control or enforces a limit. Clients read it and decide. The list envelope (`object`, `data`, `has_more`, `first_id`, `last_id`) is unchanged, and `metadata.router` is reserved, so upstream metadata can't impersonate it.

## Facts

Every leaf is a fact:

```json
{"value": true, "source": "llama.props.chat_template_caps.supports_reasoning_effort", "kind": "reported"}
```

- Booleans: `true` supported, `false` unsupported, `null` unknown. Don't treat `null` as `false`.
- `kind` is `reported`, `reviewed`, `configured`, `inferred`, `override`, `fallback`, `unknown` or `conflict`. Inferred is weaker than reported, and a context fallback is not a measured limit.
- `conflicts` lists displaced facts when sources disagree. The fact's own `value` is the resolved one.
- The schema can gain fields. Ignore unknown ones; if `schema_version` isn't `1`, treat the whole block as unknown.

## Fields

Paths are relative to `metadata.router`.

| Path | Meaning |
| --- | --- |
| `limits.context_window` | Runtime context, or a labeled estimate or fallback |
| `limits.trained_context_window` | Training context |
| `limits.max_input_tokens`, `limits.max_output_tokens` | Explicit maxima; unknown stays `null` |
| `limits.default_output_tokens` | Default generation length |
| `reasoning.supported` | The model reasons; implies no controls |
| `reasoning.toggle.supported` / `.default` | Thinking can be switched on or off |
| `reasoning.effort.supported` / `.values` / `.default` | Effort control and its exact values |
| `reasoning.budget.supported` / `.min_tokens` / `.max_tokens` / `.default_tokens` | Token-budget control |
| `reasoning.*.native_parameter` | Where the native request field lives |
| `modalities.input.<m>`, `modalities.output.<m>` | `text`, `image`, `audio`, `video`, `pdf`, `embedding` |
| `tools.supported`, `tools.parallel` | Tool calls; several per turn |
| `structured_output.supported` / `.json_object` / `.json_schema` | Structured output, by format |
| `streaming` | Streaming support |
| `sampling.<param>.supported` / `.default` / `.native_parameter` / `.constraints` | Sampling controls |
| `protocols.native.<dialect>` | Which upstream endpoints the model supports |
| `protocols.translated.<dialect>` | Whether the router can translate from that ingress |
| `protocols.known_lossy_features` | Established translation losses; `null` means unknown, not lossless |
| `freshness`, `sources` | Age and evidence for the above |

Dialects are `openai-chat`, `openai-responses`, `anthropic` and `gemini`. A same-dialect route is pass-through, so its `translated` value is `false`; every other ingress, Gemini included, is `true`.

A native mapping has `protocol`, a dotted JSON `path`, and optionally `values`:

```json
{"protocol": "anthropic", "path": "thinking.type", "values": {"enabled": "adaptive", "disabled": "disabled"}}
```

It describes the native request body, not a portable SDK argument. Overrides never change which transport a route uses.

## Evidence and precedence

- Each entry in `sources` has `age_seconds` and `stale` (either can be `null` when there's no observation).
- A failed discovery keeps the old facts and marks them stale.
- Facts restored from the discovery cache are always stale until live discovery replaces them.
- Disabling a parent (for example `reasoning.supported = false`) suppresses its dependent claims. Override the parent explicitly to re-enable them.
- If an override contradicts an inherited bound or default, the override wins and the weaker claim becomes `unknown` with `kind: conflict`.

### llama.cpp

Facts come from `/v1/models`, `/props`, `chat_template_caps` and build info. Per-model `/props` outranks model-list flags, and overrides outrank both. Router mode queries only loaded or sleeping models, with `autoload=false`. Missing flags stay unknown, and model names are never used to infer controls.

The router maps reasoning effort to llama.cpp's native fields (see the README), so it reports `reasoning.effort` as supported with kind `configured` and source `router.llama_thinking_map`. Its values are the levels the router maps (`none` to `xhigh`); `max` is not listed because llama.cpp has no maximum level. That says the control exists, not that the loaded model produces thinking. `reasoning.supported` stays whatever the server reported.

### OpenCode Go

Set `catalog.metadata_source = "opencode-go-v1"` (the bundled `router.toml` does) to join live model IDs to a frozen review of [Models.dev](https://models.dev) in `router_metadata/opencode-go-v1.json`. It applies only to an exact ID on its reviewed native dialect, and it is reviewed evidence, not verified inference. Routing still follows Go's [endpoint table](https://opencode.ai/v2/docs/console/go/#endpoints). The review expires after 30 days, and refreshing live availability does not renew it.

To update it, fetch the public API without credentials, review the exact Go IDs and their provider-specific records, keep only source-established facts, update the revision, date, checksum and links, and run the offline tests. Don't infer missing values from similar models.

## Overrides

`[capabilities.overrides."public-model-id"]` enriches a discovered or static model. It matches the advertised ID, including any prefix, and can't create routes or hold credentials. Unknown keys fail validation. It uses the same paths as the facts, with raw values:

```toml
[capabilities.overrides."local/my-model".reasoning.effort]
supported = true
values = ["small", "large"]     # only after checking the model's template
default = "small"
native_parameter = { protocol = "openai-chat", path = "reasoning_effort" }

[capabilities.overrides."local/my-model".tools]
parallel = false
```

Omitted leaves keep their discovered evidence. TOML has no `null`; JSON configs can use it to withdraw a claim. Limits use `[llama_cpp.limit_overrides]` instead.

## Using it from a client

Read the raw JSON, since typed SDK models often drop extension fields:

```python
import httpx

models = httpx.get("http://127.0.0.1:4000/v1/models").json()["data"]
model = next(m for m in models if m["id"] == "local/my-model")
cap = model["metadata"].get("router", {})
if cap.get("schema_version") == 1:
    effort = cap["reasoning"]["effort"]
    values = effort["values"]["value"]
    fresh = cap["freshness"]["discovery"]["stale"] is False
    if fresh and effort["supported"]["value"] is True and values:
        ...  # offer `values` as the effort choices
```

Build a dropdown only from known `values`. Hide controls that are explicitly unsupported, and decide how to present unknown or stale ones.
