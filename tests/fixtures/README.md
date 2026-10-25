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
