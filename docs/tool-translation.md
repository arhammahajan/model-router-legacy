# Tool translation

The router makes a request-local decision for each tool: preserve its contract,
emulate it with an explicit approximation, or report it as unsupported. The
same decision covers definitions, forced choices, conversation history, JSON
responses and streamed calls. Tool execution stays with the caller or upstream.

Same-protocol calls bypass the translation layer. The router does not enable
features that the upstream itself does not support.

## Supported contracts

| Tool | Cross-protocol behavior |
| --- | --- |
| JSON-schema function | Existing exact mappings among Chat, Responses, Messages and Gemini |
| Chat freeform `custom` → Responses | Native custom definition, grammar, choice, history and returned call contract |
| Responses freeform `custom` → Chat, Messages or Gemini | Emulated function taking `{"content": "<tool input>"}`; returned calls restored as Responses custom calls |
| Chat freeform `custom` → Messages or Gemini | Same emulation; returned calls restored as Chat custom calls |
| Responses function/custom namespaces | Reversible identities, including duplicate tool names in different namespaces |
| Anthropic plain tools or `type: "custom"` | JSON-schema functions; this Anthropic type is different from OpenAI freeform `custom` |
| Versioned Anthropic bash/editor/computer tools | Unsupported across protocols until a version-specific mapping is reviewed |
| Hosted search, MCP, code execution and other vendor tools | Require native routes; no hosted execution is added by the router |

An unsupported entry does not remove neighboring supported tools. Loss records
retain the original location, such as `tools[0].tools[1]` for a namespace child.
Chat tools with a Responses-only type, such as `web_search`, are not assumed to
be portable just because the target speaks Responses. Chat's separately
documented `web_search_options` mapping is unchanged.

## Strict and lossy policies

The existing provider `compatibility` policy applies:

- `strict` rejects unsupported tools and emulation before inference, with HTTP
  400 `translation_loss`.
- `lossy` applies reviewed emulation and reports `action: "approximated"` in
  `x-model-router-loss`. Unsupported definitions are removed with `action:
  "dropped"`, and choices referring to them are reconciled.

Freeform emulation is always an approximation, even when the string can make a
lossless round trip through the wire protocol. Asking a model for a JSON string
field changes generation behavior. A custom grammar is included as instructions
in the emulated function description; it does not constrain the destination's
decoder. Native Chat → Responses mapping preserves the actual grammar field.

The router refuses provider-native tool history that it cannot translate, even
in lossy mode (`unsupported_tool_history`). It also refuses history that refers
to a tool definition removed by the plan. Dropping that state would change the
meaning of the conversation. Historical custom calls remain translatable when
their definitions are no longer supplied; they retain a reversible identity and
receive their own approximation diagnostic.

Forced custom-tool choices are translated with the definitions. Named choices
must reference retained definitions. An `allowed_tools` choice has no reviewed
bridge contract yet and is rejected by strict policy or explicitly dropped by
lossy policy. Parallel-call controls keep their existing protocol mappings;
Anthropic's `disable_parallel_tool_use` now reaches Chat as
`parallel_tool_calls`. The unsupported nested control is diagnosed on Responses
and Gemini destinations.

## Streaming and failure behavior

Text remains incremental. For translated requests with custom tools, Chat tool
calls are buffered until their arguments are complete. Responses custom calls
emit `response.custom_tool_call_input.delta` and `.done` with the original string
input, followed by the matching completed item. JSON wrappers never become
client-facing custom input. Namespaces, call IDs, item IDs, indexes and event
sequence numbers remain consistent.

Buffering is bounded to 1,000,000 characters per accumulated field, 8,000,000
characters across a stream, and 1,024 calls. Invalid JSON, a non-string custom
input, inconsistent final arguments, oversized calls or an unfinished tool
stream produce `invalid_tool_call`. A stream that has already started receives
a terminal protocol error rather than a fabricated successful completion.
Native streams remain opaque passthrough and do not use these translation
buffers.

## Architecture and validation

`model_router/tools.py` owns `ToolDecision`, `ToolPlan`, the reviewed mapping
table, custom codecs and tool stream normalization. `protocol.py` combines its
losses with model capability and reasoning decisions, then applies compatibility
policy. Gemini uses the same decision table before normalizing tools to Chat.
`tool_namespaces.py` provides collision-safe reversible aliases.

The existing pinned LiteLLM adapters still handle the protocol bridges and the
router still owns transport deadlines, retries and resource cleanup. A reviewed
native Responses choice is supplied through the existing router-owned body-field
path because LiteLLM's Chat facade rejects custom choices before its Responses
converter runs. MCP gateway execution remains disabled.

`tests/test_tool_translation.py` exercises custom definitions, choices,
continuations, namespaces, grammar loss, JSON and SSE through real pinned
adapters with local HTTP upstreams. It also covers malformed calls, interleaved
tools, metadata restoration, stream interruption and buffer limits. Existing
function and native-fidelity tests continue to cover all four API formats.
These are wire-contract tests, not live verification that a model follows
emulated tool instructions.

Implementation verification: 345 tool/protocol regression tests passed, as did
lint, scoped type checks, dependency checks, the sdist/wheel build and an
installed-wheel smoke test using the existing pinned dependencies. The full
suite also exposed four existing Muse/Pi failures; they reproduced on a HEAD
snapshot carrying only the pre-existing catalog/Pi edits. The former opaque
Chat-to-Responses MCP forwarding assertion was updated to verify the explicit
strict/lossy decision and that the router never executes MCP.

Further work can add reviewed versioned client-tool mappings, individual hosted
service mappings, more detailed capability metadata and a separate policy for
tool dropping. They should extend the decision table and round-trip contracts.

Protocol sources:

- [OpenAI custom tools and grammars](https://developers.openai.com/api/docs/guides/function-calling)
- [OpenAI Responses custom-tool streaming events](https://developers.openai.com/api/reference/resources/responses/streaming-events)
- Installed OpenAI 2.54.0 Chat/Responses schemas, pinned by `constraints.txt`
