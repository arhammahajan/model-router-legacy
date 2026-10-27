# Model capability contract, version 1

`GET /v1/models` and its existing aliases keep the same list envelope:
`object`, `data`, `has_more`, `first_id`, and `last_id`. Each model now includes
`metadata.router.schema_version = 1`. Existing model IDs, routes, aliases,
other metadata, and prompt 08 limit fields retain their meanings.
`metadata.router` is reserved for the router; upstream or static free-form
metadata with that name cannot impersonate this contract.

This is **advisory discovery data**. Extra fields do not automatically configure
SDKs, enable UI controls, translate native parameters, or enforce limits.
Clients must read this contract explicitly. Request validation and translation
loss policy are separate concerns. Translation preflight now consumes explicit
unsupported facts, effort enums and budget bounds under the [strict/lossy
policy](README.md#translation-compatibility-and-protocol-headers); null facts
remain neutral. Same-protocol calls bypass these checks. A reported native capability does not
promise it survives every adapter.

## Facts and unknown values

Every capability leaf and limit is a fact:

```json
{"value": true, "source": "llama.props.chat_template_caps.supports_reasoning_effort", "kind": "reported"}
```

- Boolean `value: true` means supported; `false` means unsupported;
  `null` means unknown. Test these values explicitly; do not coerce null to false.
- For other types, `null` means unknown. An empty effort list means the source
  established no accepted effort values; it is different from an unknown list.
- `source` identifies the field/config path. Observed/configured evidence is
  described by the corresponding prefix in `sources`; `validation.*` names a
  schema consistency rule that withdrew an inherited claim. Prompt 08 limit
  sources retain their original `/props`,
  `/v1/models`, or `llama_cpp` paths and refer to the corresponding local
  discovery/config evidence.
- `kind` is `reported`, `reviewed`, `configured`, `inferred`, `override`,
  `fallback`, `unknown`, or `conflict`. Inferred support is weaker evidence
  than a direct report. A context fallback is not a measured runtime limit.
- Optional `conflicts` contains displaced facts (`value`, `source`, `kind`),
  preserving their order. The fact's current value is the resolved value.
  Conflict records are evidence, not additional supported alternatives.

Version 1 can gain additive fields. Clients should ignore unfamiliar fields;
if the version is unrecognized or absent, treat this extension as unknown.
Arrays and maps are JSON values, never strings containing serialized JSON.

## Fields

All paths below are relative to `metadata.router`. Each terminal path is a
fact unless identified as a container.

| Path | Value and meaning |
| --- | --- |
| `limits.context_window` | Effective runtime context, or labeled estimate/fallback |
| `limits.trained_context_window` | Training context, when known |
| `limits.max_input_tokens`, `limits.max_output_tokens` | Explicit maxima; unknown stays null |
| `limits.default_output_tokens` | Default generation length, independent of maxima |
| `reasoning.supported` | Model reasoning support; no controls follow automatically |
| `reasoning.toggle.supported`, `.default` | Whether thinking can be switched; default enabled state |
| `reasoning.effort.supported`, `.values`, `.default` | Effort control, exact known string values, default value |
| `reasoning.budget.supported` | Token-budget control |
| `reasoning.budget.min_tokens`, `.max_tokens`, `.default_tokens` | Nonnegative token counts; either bound can be unknown |
| `reasoning.toggle.native_parameter`, `reasoning.effort.native_parameter`, `reasoning.budget.native_parameter` | Native parameter description, or null |
| `modalities.input.<modality>`, `modalities.output.<modality>` | Separate booleans for `text`, `image`, `audio`, `video`, `pdf`, `embedding` |
| `tools.supported`, `tools.parallel` | Tool calls and multiple tool calls in one turn |
| `structured_output.supported` | Broad structured-output support; does not establish either specific format |
| `structured_output.json_object`, `.json_schema` | JSON mode and JSON Schema support separately |
| `streaming` | Streaming support, independent of whether enabled by default |
| `sampling.<parameter>.supported`, `.default`, `.native_parameter`, `.constraints` | Support, default, native mapping, and known textual constraints |
| `protocols.native.<dialect>` | Native upstream endpoint support; configured route dialect is true with `kind: configured`, other endpoints unknown without evidence/override |
| `protocols.translated.<dialect>` | Whether this router has an adapter path from that ingress to the configured native dialect |
| `protocols.known_lossy_features` | List of established losses; null means not established, not lossless |

Dialects are `openai-chat`, `openai-responses`, `anthropic`, and `gemini`.
There is no Gemini ingress, so its translated value is false. A same-dialect
path has translated=false because it uses passthrough. Adapter availability
is not per-feature verification. Metadata overrides never change which
transport dispatch uses; declaring an additional native endpoint does not
switch a route to that endpoint.

Sampling parameters currently represented are `temperature`, `top_p`, `top_k`,
`min_p`, `typical_p`, `seed`, `frequency_penalty`, `presence_penalty`,
`repeat_penalty`, `repeat_last_n`, `dry_multiplier`, `dry_base`,
`dry_allowed_length`, `dry_penalty_last_n`, `dry_sequence_breakers`, `mirostat`,
`mirostat_tau`, `mirostat_eta`, `dynatemp_range`, `dynatemp_exponent`,
`xtc_probability`, `xtc_threshold`, `top_n_sigma`, `adaptive_target`,
`adaptive_decay`, and `samplers`. Defaults are numbers except `samplers` and
`dry_sequence_breakers` (string arrays). Constraints are string arrays.
Unlisted or unreported controls are unknown.

A native mapping value has `protocol` and a dotted JSON `path`, optionally
`values` mapping semantic selections to native values. For example, the
reviewed MiniMax M3 toggle is:

```json
{"protocol":"anthropic","path":"thinking.type","values":{"enabled":"adaptive","disabled":"disabled"}}
```

A mapping describes the **native request body**, not a portable SDK argument.
It does not imply that a cross-protocol request can send that field through
LiteLLM unchanged. Native fields can be supplied on a same-protocol route
using that SDK's extension-body facility, or with raw JSON, subject to its
schema and the upstream's other requirements.

## Evidence, precedence, and freshness

`sources` is a map of evidence descriptions. Entries can include
`retrieved_at` (Unix seconds for live discovery; ISO 8601 UTC for the frozen
review), `reviewed_on`, `source`, `model_source`, `repository_revision`,
`server_build`, `verification`, `applicable`, and `stale_after_seconds`.
Every rendered source includes `age_seconds` and `stale`; either can be null
when there is no observation (for example, operator assertions).
`llama.props.available=false` and a null retrieval time mean that no applicable
model properties were obtained.

`freshness.discovery` reports the captured inventory's refresh state and age.
Its monotonic process-local timestamps have the same meaning as the health
refresh fields, not wall-clock dates. Local freshness is per server, so one
server's outage does not mark another server's properties stale. A discovery
failure retains previously collected facts and their retrieval times while
marking them stale; age also increases without network activity when a refresh
is blocked or overdue. Startup-only mode does not expire by interval, but a
forced failed refresh still marks retained facts stale.

With optional `[discovery_cache]` persistence, discovered entries also include
`freshness.discovery.cached` and `succeeded_at` (Unix seconds of the successful
observation). Imported inventory is always stale until live discovery replaces
it. Its age includes time before restart; reconstructed monotonic timestamps can
be negative. Local provenance is per server. Persistence keeps typed facts but
omits freeform stop strings, source URLs/build strings, conflict histories and
arbitrary upstream metadata. Current configuration supplies overrides and
transport identity. `max_age_seconds` controls admission on restart; admitted
routes retain the existing in-memory stale-route policy. See README for cache
identity, invalidation, permissions and readiness limits.

For local capabilities, applicable per-model `/props` reports take precedence
over model-list architecture flags, then sparse operator overrides win.
Conflicting lower-priority evidence remains in `conflicts`. Limits are copied
from prompt 08's resolver with its existing precedence and provenance; they
are also still available under `metadata.limits`. Go uses the frozen reviewed
metadata, then sparse overrides. Live availability is evidence of availability
only, not a source of unverified capability assertions.

Disabling a parent (`reasoning.supported`, a reasoning control's `supported`,
`tools.supported`, `structured_output.supported`, or a sampling control's
`supported`) suppresses dependent
claims and retains displaced evidence. Override that parent explicitly as
well when re-enabling a control. Internally inconsistent budget bounds/defaults
and effort defaults are rejected when configured together. If a sparse
override contradicts an inherited bound/default, the explicit value wins and
the weaker claim becomes unknown with `kind: conflict`.

### llama.cpp

The parser follows the [server property documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md),
[model/property implementation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-context.cpp),
and [template capability structure](https://github.com/ggml-org/llama.cpp/blob/master/common/jinja/caps.h).
It consumes model-list input/output architecture, `/props.modalities` input
flags, `chat_template_caps`, generation defaults, and `build_info`. Missing
modality flags stay unknown; input vision never implies image output.
A recognized sampling default establishes inferred support and a native Chat
mapping. Unreported controls and constraints stay unknown.

Router mode queries only models reported loaded or sleeping, always using
`/props?model=<id>&autoload=false`. It never probes unloaded, loading, failed,
or unknown-status models to obtain properties. A server that ignores the
model query and returns router/swap global properties cannot spread those
properties to individual models. Global properties are used for single/legacy
servers only; llama-swap retains its listing metadata and aliases.

`supports_reasoning_effort=true` establishes effort support but **no enum**.
False establishes unsupported effort control, not absence of reasoning.
`supports_preserve_reasoning` describes history preservation and does not
establish a thinking toggle. Template text and model-name substrings are
never used to invent controls or effort values. Toggle and budget details,
reasoning defaults, specific structured-output formats, and additional native
endpoints remain unknown when the server does not report them; use reviewed
operator assertions for your particular build/template.

### OpenCode Go

Set `catalog.metadata_source = "opencode-go-v1"` to join live IDs to the
bundled [review snapshot](router_metadata/opencode-go-v1.json), revision 1,
reviewed 2026-10-26. The shipped `router.toml` enables it. It was extracted
from the **Go-specific** records in [Models.dev](https://models.dev/api.json),
with repository revision, input checksum, and per-model source links recorded
in the snapshot. This is reviewed registry evidence, not authenticated Go
inference verification. It is not imported from a generic model family or
another provider's metadata. Normal startup and tests never fetch Models.dev.

The [official Go endpoint table](https://opencode.ai/v2/docs/console/go/#endpoints)
continues to govern routing via the existing exact mappings. Metadata is only
applied to an exact ID with the reviewed native dialect. An operator choosing
a different native endpoint gets unknown capabilities (`applicable=false`)
until they supply overrides. IDs without metadata remain routable when their
endpoint mapping exists. Metadata alone cannot expose an unavailable or
unmapped model.

The snapshot preserves exact effort options from its source: for example,
GLM-5.3-Flash lists `low/high/max`, Qwen3.8 Max lists `low/medium/xhigh`, and
MiniMax M3 lists a toggle without an established effort enum. An absent option
is unknown, not evidence of unsupported controls. Native parameter paths are
included only where separately reviewed in provider-specific records; other
mappings stay unknown. Broad structured-output support does not fabricate
separate JSON mode/Schema claims. No reasoning defaults are guessed.

The review expires after 30 days for advisory freshness. Refreshing live
availability cannot refresh this review date; stale facts remain visible.
To update deliberately, retrieve the public API without credentials, review
exact Go IDs and their provider-specific records and native parameter paths,
retain only source-established facts, update the revision/date/checksum/links,
and run offline tests. Do not infer missing enums from neighboring models.

## Sparse overrides

Overrides match the **canonical advertised public model ID**, including a
local prefix. Aliases share that route's contract and are not separately
listed. Unlike `[[models]]`, an override contains no provider, model name,
URL, credentials, or route declaration. It applies when the model is
discovered; it cannot create a route. Static public models also accept these
overrides. Unknown keys and invalid types fail configuration validation.

```toml
[capabilities.overrides."local/my-model".reasoning]
supported = true

[capabilities.overrides."local/my-model".reasoning.effort]
supported = true
values = ["small", "large"] # only after checking this model's template
default = "small"
native_parameter = { protocol = "openai-chat", path = "reasoning_effort" }

[capabilities.overrides."local/my-model".structured_output]
supported = true
json_schema = true

[capabilities.overrides."local/my-model".tools]
parallel = false
```

Overrides use the same nested paths as capability facts, but contain raw
values instead of fact wrappers. Omitted leaves retain discovered evidence.
JSON configuration can use null to explicitly withdraw a claim; TOML has no
null literal. Limits continue to use `[llama_cpp.limit_overrides]` as described
in the README, keyed by upstream ID before prefix/aliases. This capability
override namespace does not accept limits or rewrite legacy free-form metadata.

## Client use

Fetch raw JSON if an SDK's typed model object hides extension fields. This
example prepares a reasoning control for a Chat-native model only after
checking evidence and freshness:

```python
import httpx

models = httpx.get("http://127.0.0.1:4000/v1/models").json()["data"]
model = next(m for m in models if m["id"] == "local/my-model")
cap = model.get("metadata", {}).get("router", {})
body = {"model": model["id"], "messages": [{"role": "user", "content": "Hi"}]}
if cap.get("schema_version") == 1:
    effort = cap["reasoning"]["effort"]
    values = effort["values"]["value"]
    mapping = effort["native_parameter"]["value"]
    # Review source age/kind too; your UI chooses how to handle stale/unknown.
    fresh = cap["freshness"]["discovery"]["stale"] is False
    if (fresh and effort["supported"]["value"] is True and values
            and mapping == {"protocol": "openai-chat", "path": "reasoning_effort"}):
        body["reasoning_effort"] = values[0]
# httpx.post("http://127.0.0.1:4000/v1/chat/completions", json=body)
```

For Go also inspect `sources["go.review"].stale` and its verification label.
Only populate effort dropdowns from known `values`; a supported effort control
with unknown values cannot establish a dropdown. Hide or disable explicitly
unsupported controls; explain unknown/stale controls according to the client's
policy. Choose the ingress deliberately and map fields for that protocol.
An SDK/UI integration must implement this logic; listing enrichment alone
does not provide automatic configuration.
