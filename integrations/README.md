# Harness integrations

Client-side adapters that point a coding harness at a running router. Each
harness gets its own directory. They read `/v1/models` and its `metadata`
(see `CAPABILITIES.md`), so a change to that contract can break them. Update
them in the same change.

| Directory | Harness | What it does |
| --- | --- | --- |
| `pi/` | [pi](https://github.com/earendil-works/pi-coding-agent) | Registers the router's models as a pi provider, discovered at startup and on refresh. |

These are not part of the Python package or its sdist.
