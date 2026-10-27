# Contributor checks and dependency upgrades

Use CPython 3.10–3.14. The GitHub remote uses GitHub Actions; `.github/workflows/ci.yml` runs each supported minor on Linux. Python 3.15+ is outside the current package policy because the selected LiteLLM release excludes it. Other operating systems are not covered by this dependency baseline.

## Reproduce checks

From the repository root, with no credentials required:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install pip==26.2.2
.venv/bin/python -m pip install -c constraints.txt '.[test,dev]'
.venv/bin/python -m pip check
.venv/bin/python -m pytest tests/test_release.py tests/test_validation.py
.venv/bin/python -m pytest
.venv/bin/python -m ruff check model_router.py model_router router_metadata scripts tests
.venv/bin/python -m mypy
```

An editable install (`-e '.[test,dev]'`) is convenient for development. Use a regular wheel install for the release check below. CI runs the suite from outside the checkout so imports exercise the installed package. Tests still read checked-in config/fixtures/metadata declarations via their absolute repository paths.

Tests use synthetic keys and local HTTP upstreams with the real installed LiteLLM adapters. `tests/conftest.py` selects the installed cost map, disables telemetry and blocks external socket connections/DNS during tests. They do not need personal credential files, cloud discovery, a live llama.cpp server or paid inference. Do not use `--check`/`--check-readiness` to reproduce offline validation: those options intentionally perform discovery.

The lint baseline selects Ruff `E9,F63,F7,F82`: syntax and selected correctness errors, including undefined names. It covers the `model_router` package, the `model_router.py` launcher, `router_metadata`, scripts and tests without imposing a mass style rewrite. Mypy checks all of `router_metadata` and `scripts/check_release.py`, including untyped function bodies, with Python 3.10 semantics and imported modules skipped. No blanket error suppression or legacy error-count allowance is used. Add modules/rules as their existing errors are fixed; broader router typing remains separate work.

## Check a clean installation

This builds the wheel **from the sdist**, so missing package data can fail before release. Run these Bash commands from the repository root after installing the development tools above:

```bash
repo="$PWD"
release_dir="$(mktemp -d)"
.venv/bin/python -m build --outdir "$release_dir/dist"
python3 -m venv "$release_dir/venv"
release_python="$release_dir/venv/bin/python"
"$release_python" -m pip install pip==26.2.2
wheels=("$release_dir"/dist/*.whl)
"$release_python" -m pip install -c "$repo/constraints.txt" "${wheels[0]}"
"$release_python" -m pip check
"$release_python" "$repo/scripts/check_release.py"
"$release_python" -m pip install -c "$repo/constraints.txt" "${wheels[0]}[test,dev]"
"$release_python" -m pip check
(cd "$release_dir" && "$release_python" -m pytest -c "$repo/pyproject.toml" "$repo/tests")
"$release_python" -m ruff check model_router.py model_router router_metadata scripts tests
"$release_python" -m mypy
# Remove the temporary environment/artifacts when finished:
rm -rf "$release_dir"
```

`check_release.py` runs before test/check extras are installed, so those tools cannot hide a missing runtime dependency. It uses a fresh temporary directory with a copy of the repository's default `router.toml`, only `OPENCODE_API_KEY=dummy-release-smoke-key`, and an isolated Python import path. It verifies installed module locations, package/runtime/console versions and packaged Go review JSON. It exercises default config discovery with no explicit config argument, using `--no-default-env --validate-config --offline`. Its guarded in-process check forbids credential loaders, sockets/DNS and App construction. The actual console command is checked separately with the same dummy environment. It neither starts a server nor runs inference.

## Dependency policy

`pyproject.toml` owns package metadata and optional test/development tools. `requirements.txt` retains the small runtime-only installation path. `tests/test_release.py` compares their complete normalized runtime declarations and the runtime LiteLLM guard, checks exact constraints for every supported Python minor, and verifies the active runtime transitive closure. The Python 3.10 TOML marker is separately tested in `tests/test_validation.py`.

`constraints.txt` pins the runtime, test, build and check tool closure; it is a [pip constraints file](https://pip.pypa.io/en/stable/user_guide/#constraints-files), so it restricts versions without installing unused extras. The build backend is pinned in `pyproject.toml` because isolated build environments do not inherit normal install constraints. Pip is also pinned in the bootstrap commands. These are version pins, not artifact hashes or an offline wheel mirror; downloads still require an index and available platform wheels.

The baseline retains the audited LiteLLM 1.106.4, httpx 0.28.1 and OpenAI 2.54.0 stack and the existing environment's runtime transitive versions where Python policy permits. A clean base-only installation exposed LiteLLM 1.106.4's unconditional MCP gateway import on Responses calls, before `skip_mcp_handler` is checked. `litellm[mcp]` therefore supplies those import dependencies; startup checks the gateway import too. MCP execution remains disabled. This adds the required extra, not the much larger proxy/enterprise extras present in the original development environment. Python 3.10 uses marker-specific older transitive releases where newer baseline versions require Python 3.11; async-timeout and exceptiongroup are conditional compatibility dependencies. Test versions come from the existing suite environment; the build/lint/type tools are newly pinned tooling, not a broad runtime upgrade.

Private interfaces in use:

| Interface | Router use |
| --- | --- |
| `AnthropicAdapter` | Input/tool-name conversion, output conversion, streaming output conversion |
| `LiteLLMMessagesToResponsesAPIHandler.anthropic_messages_handler` | Messages to Responses, including async streams |
| `HTTPHandler` | Request-scoped translation transport with a generation-owned pool |
| httpx transport pools / httpcore network backend | Request-owned retry boundary; clamp each socket/TLS operation to the overall deadline |
| Adapter stream wrapper resources | Cleanup of upstream responses/iterators in `Plan` |

All LiteLLM imports and private wrapper inspection live in
`model_router/adapters.py`, behind `LiteLLMAdapter`. Application dispatch,
translation policy and stream ownership call that interface. The pinned version
constant lives in `model_router/constants.py`. Configuration and discovery can
be imported independently of LiteLLM; CLI/application startup still checks its
version and adapter surface. See README's package layout for the module owners.
When testing implementation details, patch the owning module's globals rather
than the lazy public exports in `model_router.__init__`.

The startup version/import/method/constructor checks catch obvious incompatible installations. They do not prove semantic compatibility; local protocol, security, header, error and lifecycle contracts provide that evidence. A future 1.x release is a candidate to test, not an accepted dependency by default.

## Intentionally test an upgrade

1. Use a branch and a new temporary virtual environment. Choose one explicit LiteLLM candidate version and record the reason (bug/security fix, required adapter change), release notes and relevant private-interface changes. Keep the current baseline available for comparison.
2. Change **only the candidate's direct pin** in `pyproject.toml`, `requirements.txt`, `model_router.constants.TESTED_LITELLM_VERSION` and a temporary copy of `constraints.txt`. Do not disable the version guard, replace `==` with a major-version range, or run an unbounded `pip install -U`. The temporary constraints should retain every other baseline pin initially.
3. Resolve without installing first, using the candidate constraints and a report:

   ```bash
   # Run in the candidate environment; candidate-constraints.txt starts as a
   # copy of constraints.txt with only the selected LiteLLM pin changed.
   python -m pip install --dry-run --ignore-installed \
     -c candidate-constraints.txt '.[test,dev]' --report candidate-resolution.json
   ```

   If the candidate cannot resolve, review each conflicting transitive requirement. Relax only justified pins in the temporary constraints, inspect the report, and replace relaxed ranges with the resolved exact versions before testing. Repeat resolution on the actual supported Python minors; preserve marker-specific pins. Cross-Python `pip --python-version` resolution alone does not validate environment markers or execution.
4. Install with those exact candidate constraints in a fresh environment, then run `pip check`, the focused release/validation tests, the full suite, lint and types. In particular retain the real-adapter Chat/Responses/Messages × four-upstream matrix in both modes, tools, reasoning, usage, credential isolation, actual HTTP/SSE errors, stream cleanup and cancellation. Do not stub adapters, weaken assertions or use live paid requests to get a pass. Fix small adapter compatibility issues within scope; report larger feature/refactor needs separately.
5. Rebuild the sdist/wheel and repeat the clean install/default-config check above with candidate constraints. Run the GitHub Actions Python matrix (or the same commands locally on each minor). Review frozen environments or installation reports for unexpected additions/version changes; do not copy an unrelated development environment's whole `pip freeze` into the baseline.
6. Promote the reviewed candidate constraints to `constraints.txt` only when all supported matrix jobs pass. Commit the synchronized direct pins/guard, any adapter contract fixes, updated constraints and a note with the reason, changed transitives, Python compatibility and check results. Update README's tested versions. Delete candidate files rather than leaving a second dependency source. If a Python minor must be dropped, make that a deliberate documented metadata/CI compatibility change.

To reject a candidate, revert its pins and code changes and recreate the environment from the retained baseline. Production startup has no opt-out that silently accepts untested LiteLLM versions.
