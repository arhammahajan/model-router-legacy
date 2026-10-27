# OpenCode Go routing review — revision 1, 2026-10-26

`opencode_go_catalog.json` is the unauthenticated JSON response from
https://opencode.ai/inference/go/v1/models, retrieved during this review.
It contains availability only: `id`, `object`, `created`, and `owned_by`.

`opencode_go_endpoints.json` transcribes the 29 model ID/endpoint rows from
https://opencode.ai/v2/docs/console/go/#endpoints on the same date. `dialect`
labels normalize the documented paths. Gemini's documented model resource
uses the configured Google `/v1beta` base; LiteLLM adds the generation action.
This fixture records documented native endpoints, not live verification or
support for other endpoint dialects. No authenticated inference was performed.

Nine catalog IDs have no row in the current endpoint table: `glm-5.1`,
`grok-4.6`, `iceman-contributor-free`, `kimi-k2.6`, `minimax-m2.5`, `minimax-m2.7`,
`omen-alpha`, `qwen3.6-plus`, and `qwen3.7-max`. They remain unmapped/unknown.
Do not infer a dialect from their family name or from catalog availability.
In particular, the table documents MiniMax M3 as Messages and M3.1 Flash as
Chat; all three currently documented Qwen IDs use Messages.

For a deliberate update, read the current official endpoint table and fetch
the public catalog without credentials. Review each exact ID and path, update
both fixtures and `[catalog.model_providers]` in `router.toml`, and increment
the review revision/date in the fixture, configuration and README. Leave IDs
without endpoint evidence unknown. Run the offline routing and full suites.
Normal tests must read these files and must not fetch current cloud data.

Prompt 09's separate capability review is packaged in
`router_metadata/opencode-go-v1.json` so installed routers and offline tests use
the same evidence. It contains exact Go-provider facts from Models.dev, source
links pinned to the reviewed repository revision, the original API checksum,
retrieval/review dates, and an independent 30-day freshness policy. Availability
and endpoint fixtures do not establish model capabilities. See `CAPABILITIES.md`
for the schema and deliberate review procedure; do not copy effort enums from
other providers or infer controls from model names. No authenticated inference
was performed in either review.

## Protocol contract fixtures — prompt 15, reviewed 2026-10-27

`test_protocol_contracts.py` builds small synthetic HTTP JSON/SSE fixtures from
these official schemas; they are not recordings of paid inference:

- [OpenAI Responses and usage](https://developers.openai.com/api/reference/resources/responses/methods/create)
  and [streaming events](https://developers.openai.com/api/reference/resources/responses/streaming-events):
  function-call IDs, argument delta/done events, input/output token details,
  inline images and response lifecycle events.
- [OpenAI Chat streaming](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create):
  indexed tool-call deltas, JSON argument strings, a separate final usage chunk
  with empty `choices`, and `[DONE]` termination.
- [Anthropic streaming](https://platform.claude.com/docs/en/build-with-claude/streaming),
  [tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use),
  and [prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching):
  indexed content blocks, `input_json_delta`, thinking/signature deltas, cumulative
  output usage, cache creation/read counts, and server-tool usage.
- [Gemini GenerateContent](https://ai.google.dev/api/generate-content):
  `contents`/`parts`, complete `functionCall`/`functionResponse` objects, inline
  image data, thought parts, and `usageMetadata`. There are no invented partial
  function-argument string deltas. Gemini 2.5's candidate count excludes thoughts;
  the fixture has 12 prompt + 3 candidate + 2 thought = 17 total tokens, including
  4 cached prompt tokens. LiteLLM emits snake_case request keys (`function_call`,
  `inline_data`, `max_output_tokens`); these are valid original proto field names
  under [ProtoJSON's field-name rules](https://protobuf.dev/programming-guides/json/#field-names-as-json-keys).
- [OpenCode inference cost extensions](https://opencode.ai/v2/docs/console/api/inference#cost):
  `cost`, `opencode.billing_source`, `list_cost`, `allowance`, and billed token
  breakdown live inside native `usage` (Gemini uses `usageMetadata`). Stream
  billing is attached to the final usage event. Native Chat/Responses/Messages
  fidelity is tested; cross-protocol preservation of arbitrary billing extensions
  is not promised.

All credentials, signatures, IDs, token counts, billing amounts and model output
are synthetic. The signatures test opaque transport, not cryptographic validity.
The image is a one-pixel inline PNG, so no image URL needs fetching. Split tool
arguments include Unicode and escaped quotes. Responses fixtures include ordered
lifecycle/content/function events and explicit sequence numbers. Tests select a
known Responses model to exercise upstream SSE rather than LiteLLM's simulated
streaming fallback for unknown models. Terminal-error tests deliberately append
an invalid success continuation after the error to detect accidental consumption
or success fabrication; native passthrough remains opaque to that continuation.

`tests/conftest.py` uses the installed LiteLLM cost map, disables telemetry, and
rejects non-loopback socket connections. Adapter functions remain real. A few
lifecycle tests wrap HTTP sends solely to observe response closure while still
performing the real HTTP request. Explicit zero adapter retries make refusal
contracts single-attempt; this does not define a new production retry policy.
