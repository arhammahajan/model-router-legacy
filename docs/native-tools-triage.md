# Native tool handling: investigation and recommendation

Reviewed 2026-10-31 against the working tree and pinned LiteLLM 1.106.4.
The original investigation below proposed changes without altering generation.

Implementation update: the first phase is now implemented. The original
findings below describe the investigated baseline; current behavior and
remaining work are documented in [Tool translation](tool-translation.md).
The shared decision layer, custom-tool emulation, namespace restoration,
stream normalization and Anthropic → Chat parallel constraint mapping are
included. Versioned client-tool mappings, hosted mappings and independent
tool-dropping policy remain future work.

## Finding

The router deliberately removes non-function tools during most cross-protocol
translations. The bundled `router.toml` sets every provider, including local
discovery, to `compatibility = "lossy"`, so these requests can return HTTP 200
after losing their tools. The loss is reported in `x-model-router-loss`; the
standard request outcome log does not include these loss records.

Same-protocol requests take the passthrough branch before translation planning
and preserve tool definitions. Ordinary JSON-schema functions translate across
Chat Completions, Responses, Messages, and Gemini. The problem is the policy's
broad classification of everything else as an unsupported native tool.

The main policy is in `model_router/protocol.py`:

- `_function_name` recognizes schema functions, including Anthropic's plain or
  `type: "custom"` schema tools. OpenAI's `type: "custom"` means a different,
  freeform tool and is not recognized.
- `_reconcile_tools` removes every other tool, keeps functions, and adjusts
  dependent `tool_choice` and parallel controls.
- `_prepare_translation` invokes this filter when the destination is not
  Responses, or when the ingress is Anthropic. Consequently Chat → Responses
  is an exception: it can carry custom tools through the pinned adapter.
- `_enforce_policy` rejects reported losses with HTTP 400 `translation_loss`
  in strict mode and allows them in lossy mode.

Gemini ingress has an additional earlier filter in `gemini.Translation.tools`:
anything other than `functionDeclarations` is reported as a loss and omitted.
Removing only `_reconcile_tools` would therefore leave Gemini ingress broken
and allow unreviewed adapter behavior elsewhere.

These are established behaviors, explicitly asserted by `test_mixed_tools.py`
and `test_gemini_ingress.py`. Commit `98c4684` introduced per-tool reconciliation
so a native tool would no longer discard neighboring functions. This
investigation does not establish when a particular client first encountered it.

## Reproduction

Local fake upstreams and the real router/adapters were used. No authenticated
cloud inference or live model behavior was tested.

Each request contained a function plus one native tool: OpenAI freeform
`custom`, Anthropic `bash_20250124`, or Gemini `googleSearch`. The matrix was
run once in strict mode and once in lossy mode, for 32 requests total.

| Ingress → destination | Chat | Responses | Messages | Gemini |
| --- | --- | --- | --- | --- |
| Chat, function + custom | Preserved | Preserved by bridge | Rejected / dropped | Rejected / dropped |
| Responses, function + custom | Rejected / dropped | Preserved | Rejected / dropped | Rejected / dropped |
| Messages, function + typed bash | Rejected / dropped | Rejected / dropped | Preserved | Rejected / dropped |
| Gemini, function + Google Search | Rejected / dropped | Rejected / dropped | Rejected / dropped | Preserved |

“Rejected / dropped” means HTTP 400 before inference in strict mode, or HTTP
200 with only the function remaining and a loss header in lossy mode.
“Preserved” establishes transport fidelity, not that every upstream supports
the requested tool. The existing mixed-tool tests additionally cover Responses
web search and MCP removal on Chat/Messages destinations.

## Triage

### High: emulatable client tools are discarded

The filter conflates three different capabilities:

1. Schema functions executed by the client.
2. Other client-executed tools: OpenAI freeform tools and versioned Anthropic
   bash/editor/computer tools.
3. Hosted services: web/file search, hosted code execution, remote MCP
   connectors, and Gemini grounding tools.

These categories require different decisions. A local model can call a client
function without providing a hosted search service. A wire-compatible endpoint
also does not prove that a model implements a particular hosted tool.

The pinned LiteLLM already contains a Responses custom-tool bridge. Bypassing
only the router filter in a local probe produced a function with a single
`content: string` parameter upstream and restored the returned call to:

```json
{"type":"custom_tool_call","name":"patch","call_id":"call_patch","input":"*** Begin Patch\n*** End Patch"}
```

Grammar-constrained tools need special treatment: LiteLLM puts the grammar in
the function description, which does not preserve constrained decoding. That
conversion must be reported as an approximation. The official OpenAI
[function calling documentation](https://developers.openai.com/api/docs/guides/function-calling)
distinguishes schema functions, arbitrary-string custom tools, and grammar
constraints.

### High: dropping definitions leaves inconsistent conversation state

A lossy Responses → Chat probe removed the `patch` custom-tool definition but
still forwarded a previous `custom_tool_call` and its result as Chat function
history. If the upstream then called `patch`, the router returned
`function_call` with JSON arguments instead of `custom_tool_call` with string
input. The adapter no longer had the removed definition needed for restoration.

Tool definitions, selected tools, historical calls/results, returned calls,
and streamed events must share one identity/conversion plan. Removing only
definitions does not provide a coherent continuation contract.

### High for the proposed fix: custom-tool streaming needs repair

A direct probe of the pinned adapter, with the custom definition retained,
restored the final custom call correctly but emitted
`response.function_call_arguments.delta` and `.done` events for its JSON
wrapper. OpenAI defines separate
[`response.custom_tool_call_input.delta` and `.done` events](https://developers.openai.com/api/reference/resources/responses/streaming-events)
containing the custom string input.

The router's `_responses_sse` serializes these events without converting their
tool semantics. Merely allowing custom definitions through would expose this
latent adapter defect to streaming clients.

### Medium: namespace restoration supports functions only

`flatten_namespaces` aliases function members and historical function calls;
non-function members are lifted out unchanged. Two namespaces containing a
custom tool named `patch` become two identically named top-level tools, with
no reverse identity mapping. `restore_namespaces` also only handles
`function_call`. Custom tools need the same collision-safe identity handling.
Loss paths should retain the original nested tool location.

### Medium: hosted tool fidelity is broader than request definitions

The Gemini response encoder carries text, reasoning text and function calls,
but does not carry Chat citation annotations into Gemini grounding metadata.
A unit probe confirmed the annotations disappear. Chat → Responses can already
pass some native definitions through; a local web-search response probe kept
answer annotations but omitted the separate search-call item in the Chat
response. Any new hosted-tool mapping must review response metadata, events,
usage and continuation, as well as input configuration.

Hosted tools cannot be made universal by renaming their types. Even OpenAI
documents different behavior for
[Chat search and Responses web search](https://developers.openai.com/api/docs/guides/tools-web-search).
Provider resource IDs, containers, vector stores and approval state also
cannot be assumed to exist at another provider.

## Recommended design

Add a router-owned `ToolPlan` in a focused `model_router/tools.py`, following
the existing `ReasoningPlan` pattern. Keep the pinned adapter boundary in
`adapters.py`, existing transport ownership in `Plan`, and same-protocol
passthrough ahead of translation planning.

The plan should classify tools by source protocol, tool kind/version,
execution owner (client/upstream), original request path, and identity
(namespace/name). Resolve each against the actual route, adapter mapping, and
known model/provider capability facts. Use the existing exact / approximated /
unsupported policy model. Do not infer hosted services from `tools.supported`,
model names, or the endpoint dialect.

The plan should contain validated request rewrites, loss records, reversible
tool identities and codecs for calls/results and response events. One
request-local plan must be used by definitions, choice controls, history,
JSON responses, and streaming responses. Gemini's planner must operate before
native declarations are discarded by its Chat normalization step.

### Behavior by tool category

- **Schema functions:** retain current mappings across all four formats and
  collision-safe namespace restoration.
- **Freeform custom tools:** preserve exact native Chat ↔ Responses forms
  where the destination/model supports them. For schema-only destinations,
  offer reviewed emulation using a single string field and restore the
  original client-facing call/result form. Reuse pinned LiteLLM functionality
  where verified; fill gaps inside the router adapter boundary. Conservatively
  report emulation as approximated; grammar guarantees are always a loss
  unless the destination can enforce the same grammar.
- **Typed client tools:** add explicit, versioned schema/prompt mappings for
  bash/editor/computer tools only where their input/output contract can be
  preserved. Treat injected tool instructions or lost controls as
  approximations. Unknown versions must be refused or explicitly dropped.
- **Hosted tools:** preserve them on native routes. Add individual mappings
  only for verified destination services with compatible options, result
  metadata, events and state. Otherwise reject with the specific tool and
  destination identified. Explicit opt-in dropping can remain available.

For custom streaming, normalize event types, payloads and item identities as
one unit. Initially buffer each emulated tool's bounded JSON arguments, decode
the string, and emit a valid custom-input delta/done sequence before the final
item. This trades incremental tool-input delivery for a correct contract and
matches the router's existing willingness to buffer Gemini function calls.
Malformed arguments or upstream failures must terminate without fabricating a
successful tool call.

This is a protocol planner, not a tool executor. Keep client-side execution
with the client and hosted execution with a verified upstream. Keep the
router's MCP gateway execution disabled; forwarding or emulating a tool must
not accidentally start executing it in the router process.

### Policy and capability reporting

Separate tool-loss acceptance from general sampling/reasoning compatibility.
Recommended behavior is to refuse unsupported tool removal unless a dedicated
operator policy explicitly permits it. This prevents the bundled broad
`compatibility = "lossy"` setting from turning an agent request into plain text
generation. Make this migration explicit for existing deployments.

Expose per-tool execution/format support and effective ingress-to-route mapping
status in capability metadata, using the existing tri-state facts and
provenance conventions. Preserve `tools.supported` as the general function-call
fact. Add sanitized loss counts/kinds to request outcome logs; retain the
existing loss header. Do not change native response bodies to add router data.

Keep endpoint selection tied to reviewed routes. Where a provider exposes
multiple verified native endpoints for the same model, explicit routes can
offer them; a later route-set extension could choose among those declared
endpoints. Do not guess alternate endpoints or silently switch models or
providers to satisfy a tool.

## Delivery and acceptance

1. Add contract tests for freeform definitions, custom history/results, forced
   choices, duplicate names/namespaces, and the custom streaming mismatch.
2. Implement the shared plan and freeform emulation first. Verify it for each
   of the four ingresses against every destination, including continuations.
3. Add reviewed typed client-tool mappings and independent tool-loss policy.
4. Add hosted mappings individually, gated on established upstream capability
   and complete response fidelity. Unsupported combinations stay explicit.

Use real pinned adapters and local HTTP fixtures for the 4 × 4 matrix,
streaming and nonstreaming, strict and explicit lossy policies. Test fragmented
JSON/Unicode, concurrent calls, stable call IDs and namespaces, grammar loss,
required/none/named/allowed choices, parallel constraints, malformed calls,
terminal errors and cleanup. Preserve byte-for-byte native response/SSE
contracts. The existing Anthropic → Chat parallel-control gap documented in
`test_mixed_tools.py` should also become an enforced diagnostic or a supported
mapping rather than remaining an unreported adapter loss.

Immediate operational mitigation: use the selected model's native ingress for
native tools, or use strict translation to receive an explicit error. Strict
mode changes visibility of the problem; it does not implement missing tools.

## Validation performed

Result: **293 passed**, with 94 dependency warnings, in 201.49 seconds. Passing
tests confirm the current contracts, including intentional native-tool removal;
they do not demonstrate support for the proposed mappings.

The targeted existing suites were run with ambient router authentication
variables removed because those variables otherwise make unauthenticated
local test requests return 401:

```sh
env -u MODEL_ROUTER_API_KEYS -u MODEL_ROUTER_API_KEY \
  .venv/bin/python -m pytest \
  tests/test_mixed_tools.py tests/test_gemini_ingress.py \
  tests/test_tool_namespaces.py tests/test_translation_policy.py \
  tests/test_protocol_contracts.py --tb=short
```

Additional local probes exercised the 32 native-tool matrix requests, direct
adapter custom-tool JSON/stream translation, definition/history consistency,
custom namespace collisions, and response annotation handling. These establish
router and adapter wire behavior; live provider support and model adherence to
emulated instructions remain unverified.
