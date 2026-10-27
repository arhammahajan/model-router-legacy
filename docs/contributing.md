# Contributing

Use CPython 3.10–3.14. CI (`.github/workflows/ci.yml`) runs each minor on Linux. Python 3.15+ is excluded because the pinned LiteLLM release excludes it.

## Run the checks

No credentials needed:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -c constraints.txt -e '.[test,dev]'
.venv/bin/python -m pip check
.venv/bin/python -m pytest
.venv/bin/python -m ruff check model_router.py model_router router_metadata scripts tests
.venv/bin/python -m mypy
```

Always install with `-c constraints.txt`. Tests check that the installed versions match it.

Tests use synthetic keys and local HTTP upstreams with the real LiteLLM adapters. `tests/conftest.py` disables telemetry and blocks external network access, so no live llama.cpp, cloud discovery or paid calls are involved. Don't use `--check-readiness` to reproduce offline validation, because it performs discovery. Use `--validate-config` instead.

Ruff selects `E9,F63,F7,F82` (syntax and undefined names). Mypy covers `router_metadata` and `scripts/check_release.py`.

## Check a clean install

Builds the wheel from the sdist, so missing package data fails before release:

```bash
repo="$PWD"; tmp="$(mktemp -d)"
.venv/bin/python -m build --outdir "$tmp/dist"
python3 -m venv "$tmp/venv"
"$tmp/venv/bin/python" -m pip install -c "$repo/constraints.txt" "$tmp"/dist/*.whl
"$tmp/venv/bin/python" "$repo/scripts/check_release.py"
rm -rf "$tmp"
```

`check_release.py` validates the installed package, console command and bundled data using a dummy key and the default `router.toml`. It starts no server and makes no network calls.

## Dependencies

- `pyproject.toml` holds package metadata. `requirements.txt` is the runtime-only list, and `tests/test_release.py` checks the two agree.
- `constraints.txt` pins the whole runtime, test, build and check closure. These are version pins, not hashes.
- The baseline is LiteLLM 1.106.4, httpx 0.28.1 and OpenAI 2.54.0. LiteLLM is pinned exactly because the router uses its private interfaces (`AnthropicAdapter`, `LiteLLMMessagesToResponsesAPIHandler`, `HTTPHandler`). The `mcp` extra is needed because this release imports its MCP gateway before checking `skip_mcp_handler`; MCP execution stays disabled.
- All LiteLLM imports live in `model_router/adapters.py`. The pinned version is `TESTED_LITELLM_VERSION` in `model_router/constants.py`. Startup rejects any other version.
- When patching in tests, patch the owning module, not the lazy exports in `model_router/__init__.py`.

## Upgrading LiteLLM

1. Use a branch and a fresh venv. Pick one candidate version and note why (fix, required change) and any private-interface changes.
2. Change only that direct pin, in `pyproject.toml`, `requirements.txt`, `TESTED_LITELLM_VERSION` and a temporary copy of `constraints.txt`. Never loosen `==` or disable the version guard.
3. Resolve without installing: `pip install --dry-run --ignore-installed -c candidate-constraints.txt '.[test,dev]' --report candidate.json`. Relax only justified transitive pins, then record exact resolved versions. Repeat on each supported Python minor.
4. Install in a fresh venv and run `pip check`, the full suite, lint and types. Keep the real-adapter ingress × upstream matrix, tools, reasoning, credential isolation, SSE errors and stream cleanup. Don't stub adapters or weaken assertions.
5. Rebuild and run the clean-install check, then run the CI matrix.
6. Promote the candidate constraints only when everything passes. Commit the pins, any adapter fixes and a note of what changed. Delete the candidate files.

To reject a candidate, revert the pins and recreate the venv from the baseline.
