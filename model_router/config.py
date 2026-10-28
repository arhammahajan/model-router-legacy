"""Validated configuration, trusted provider/route records and credential loading."""
from __future__ import annotations

import json
import math
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qsl, unquote, urlsplit

import router_metadata

from .common import _freeze, _thaw
from .constants import (
    _DROP_REQUEST_HEADERS,
    _ENV_PATTERN,
    _HTTP_TOKEN,
    _NATIVE_SESSION_HEADERS,
    COUNT_TOKENS_NATIVE_HOSTS,
    DEFAULT_BODY_READ_TIMEOUT,
    DEFAULT_HEADER_READ_TIMEOUT,
    DEFAULT_LLAMA_CONTEXT,
    DEFAULT_LLAMA_PROBE_TIMEOUT,
    DEFAULT_LLAMA_REFRESH,
    DEFAULT_LLAMA_URL,
    DEFAULT_MAX_BODY,
    DEFAULT_MAX_CONNECTIONS,
    DEFAULT_MAX_GENERATIONS,
    DEFAULT_TIMEOUT,
    DEFAULT_WRITE_TIMEOUT,
    DIALECT_ANTHROPIC,
    DIALECT_GEMINI,
    DIALECTS,
    LLAMA_LIMIT_FIELDS,
    LOG,
    UNAUTHENTICATED_API_KEY,
)
from .headers import _serialized_json_headers, _session_request_headers, _set_header


def _expand_env(value: Any, where: str = "config") -> Any:
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            result = os.environ.get(name)
            if result is None or not result.strip():
                raise SystemExit(f"{where}: environment variable {name!r} is missing or empty")
            return result

        if "${" in _ENV_PATTERN.sub("", value):
            raise SystemExit(f"{where}: invalid environment substitution; use ${{ENV_VAR}}")
        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand_env(item, f"{where}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, dict):
        return {key: _expand_env(item, f"{where}.{key}") for key, item in value.items()}
    return value


def _toml_loads(text: str) -> dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10; dependency is conditional.
        import tomli as tomllib
    return tomllib.loads(text)


def _load_config_file(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"Cannot read config {path}: {exc}") from exc

    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            data = json.loads(text)
        elif suffix == ".toml":
            data = _toml_loads(text)
        elif suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore[import-not-found]
            except ImportError as exc:
                raise SystemExit(
                    "YAML configs need PyYAML. Use TOML/JSON or install pyyaml."
                ) from exc
            try:
                data = yaml.safe_load(text)
            except yaml.YAMLError as exc:
                # Parser diagnostics can include literal credential-bearing
                # source lines; report the format failure without echoing them.
                raise SystemExit(f"Invalid config {path}: malformed YAML") from exc
        else:
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = _toml_loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"Invalid config {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise SystemExit(f"Config {path} must contain an object/table at the top level")
    return _expand_env(data)


def _as_str(raw: Mapping[str, Any], key: str, *, where: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"{where}: '{key}' must be a non-empty string")
    return value.strip()


def _config_table(raw: Any, fields: Mapping[str, tuple[type, ...]], where: str) -> None:
    """Reject typos and coercions before applying defaults or truthiness."""
    if not isinstance(raw, Mapping):
        raise SystemExit(f"{where}: must be a table/object")
    for key, value in raw.items():
        if key not in fields:
            raise SystemExit(f"{where}: unknown field {key!r}")
        if type(value) not in fields[key]:
            expected = "/".join(t.__name__ for t in fields[key])
            raise SystemExit(f"{where}.{key}: must be {expected}")


def _config_number(value: Any, where: str, *, zero: bool = False) -> None:
    if (type(value) not in (int, float) or (type(value) is float and not math.isfinite(value))
            or not (0 <= value if zero else 0 < value) or value > threading.TIMEOUT_MAX):
        raise SystemExit(f"{where}: must be a finite {'nonnegative' if zero else 'positive'} "
                         "number <= threading.TIMEOUT_MAX")


def _config_strings(value: Any, where: str) -> None:
    if type(value) is not list or any(type(item) is not str or not item.strip() for item in value):
        raise SystemExit(f"{where}: must be an array of non-empty strings")


def _config_headers(value: Any, where: str) -> None:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) or not isinstance(item, str)
        or re.fullmatch(_HTTP_TOKEN.decode("ascii"), key) is None
        or any(ord(c) < 32 or ord(c) > 126 for c in item)
        for key, item in value.items()
    ):
        raise SystemExit(f"{where}: must contain HTTP header names and string values")


def _config_finite(value: Any, where: str = "config") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise SystemExit(f"{where}: numbers must be finite")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SystemExit(f"{where}: field names must be strings")
            _config_finite(item, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for item in value:
            _config_finite(item, where)


_CREDENTIAL_FIELDS = {"api_key": (str, type(None)), "api_key_env": (str,), "api_key_file": (str,)}


_PROVIDER_FIELDS = {
    **_CREDENTIAL_FIELDS, **{name: (str,) for name in (
        "dialect", "base_url", "litellm_provider", "compatibility", "session_header",
        "session_missing", "auth_header",
    )}, "headers": (dict,), "timeout": (int, float), "litellm_kwargs": (dict,),
    "drop_params": (bool,), "session_source_headers": (list,),
    "session_prompt_cache_key": (bool,), "count_tokens": (bool,),
}


@dataclass(frozen=True)
class Provider:
    name: str
    dialect: str
    base_url: str
    api_key: str
    headers: Mapping[str, str]
    timeout: float
    litellm_provider: str
    litellm_kwargs: Mapping[str, Any]
    drop_params: bool
    session_header: str = ""
    auth_header: str = ""
    session_source_headers: tuple[str, ...] = _NATIVE_SESSION_HEADERS
    session_prompt_cache_key: bool = False
    session_missing: str = "omit"
    # Discovered llama.cpp server: reasoning effort is mapped to its native
    # thinking controls instead of being forwarded as reasoning_effort.
    llama_cpp: bool = False
    # Whether this upstream has an Anthropic token-count endpoint. None means
    # "decide from the base URL"; True/False is an operator assertion.
    count_tokens: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "headers", _freeze(self.headers))
        object.__setattr__(self, "litellm_kwargs", _freeze(self.litellm_kwargs))

    @property
    def count_tokens_enabled(self) -> bool:
        """Only an Anthropic-protocol provider known or declared to count tokens.

        Anthropic-compatible does not imply a /count_tokens endpoint. Absent an
        explicit setting, only the documented first-party host qualifies; the
        exact netloc comparison rejects userinfo, ports and look-alike hosts.
        """
        if self.dialect != DIALECT_ANTHROPIC or self.count_tokens is False:
            return False
        if self.count_tokens:
            return True
        parts = urlsplit(self.base_url)
        return parts.scheme == "https" and parts.netloc.lower() in COUNT_TOKENS_NATIVE_HOSTS

    @property
    def effective_api_key(self) -> str:
        return self.api_key.strip() or UNAUTHENTICATED_API_KEY

    def auth_headers(self) -> dict[str, str]:
        """Upstream credential headers, honoring an explicit auth_header."""
        key = self.effective_api_key
        if self.auth_header == "bearer":
            return {"Authorization": "Bearer " + key}
        if self.auth_header:
            return {self.auth_header: key}
        if self.dialect == DIALECT_ANTHROPIC:
            return {"x-api-key": key}
        if self.dialect == DIALECT_GEMINI:
            return {"x-goog-api-key": key}
        return {"Authorization": "Bearer " + key}

    def litellm_call_kwargs(
        self, *, skip_mcp: bool = False, session_id: str | None = None
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = _thaw(self.litellm_kwargs)
        api_base = self.base_url
        if self.litellm_provider == "anthropic":
            # LiteLLM appends /v1/messages itself, so an origin-style base is
            # what it wants. Accept the base_url forms the pass-through path
            # allows and normalize them here.
            for suffix in ("/v1/messages", "/v1"):
                if api_base.endswith(suffix):
                    api_base = api_base[: -len(suffix)]
                    break
        kwargs.setdefault("api_base", api_base or self.base_url)
        # Always truthy, including unauthenticated mode. Never permit a second
        # credential source in litellm_kwargs to change this selection.
        kwargs["api_key"] = self.effective_api_key
        extra_headers = dict(kwargs.pop("headers", {}) or {})
        for name, value in dict(kwargs.pop("extra_headers", {}) or {}).items():
            _set_header(extra_headers, name, value)
        for name, value in self.headers.items():
            _set_header(extra_headers, name, value)
        for name, value in self.auth_headers().items():
            _set_header(extra_headers, name, value)
        extra_headers = _session_request_headers(self, extra_headers, session_id)
        extra_headers = _serialized_json_headers(extra_headers)
        # Keep both truthy: adapters otherwise consult litellm.headers.
        kwargs["headers"] = dict(extra_headers)
        kwargs["extra_headers"] = extra_headers
        kwargs.setdefault("timeout", self.timeout)
        # Loss is handled before calling adapters, including in explicit lossy
        # mode. Never let LiteLLM silently remove additional parameters.
        kwargs["drop_params"] = False
        if skip_mcp:
            kwargs.setdefault("_skip_mcp_handler", True)
        return kwargs


@dataclass(frozen=True)
class Route:
    name: str
    provider_name: str
    upstream_model: str
    dialect: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    enabled: bool = True
    # Resolution pins the transport identity even after a newer registry is
    # published. Exclude credentials from repr/equality.
    _provider: Provider | None = field(default=None, repr=False, compare=False)
    _capabilities: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    _dynamic: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        object.__setattr__(self, "_capabilities", _freeze(self._capabilities))


@dataclass(frozen=True)
class CatalogRule:
    pattern: re.Pattern[str]
    provider_name: str


@dataclass(frozen=True)
class CatalogConfig:
    url: str
    headers: dict[str, str]
    timeout: float
    refresh_seconds: float
    rules: list[CatalogRule]
    # Exact IDs take precedence over optional regex rules. Values reference
    # trusted providers; catalog responses cannot supply endpoints or keys.
    model_providers: dict[str, str] = field(default_factory=dict)
    metadata_source: str | None = None


@dataclass(frozen=True)
class LlamaCppServer:
    """One llama.cpp / llama-swap endpoint to probe."""

    url: str
    name: str = ""
    api_key: str = ""
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LlamaCppConfig:
    servers: list[LlamaCppServer]
    headers: dict[str, str]
    probe_timeout: float
    refresh_seconds: float
    context_fallback: int
    prefix: str
    expose_aliases: bool
    limit_overrides: Mapping[str, Mapping[str, int | None]] = field(default_factory=dict)
    compatibility: str = "strict"

    def __post_init__(self) -> None:
        object.__setattr__(self, "limit_overrides", _freeze(self.limit_overrides))


@dataclass(frozen=True)
class DiscoveryCacheConfig:
    path: str
    max_age_seconds: float = 86400

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise SystemExit("discovery_cache.path must be a nonempty path")
        _config_number(self.max_age_seconds, "discovery_cache.max_age_seconds")


@dataclass(frozen=True)
class RetryPolicy:
    # Attempts include the initial send. Replays are deliberately opt-in.
    max_attempts: int = 1
    statuses: tuple[int, ...] = (429, 503)
    connect_errors: bool = True
    backoff_base: float = 0.5
    backoff_max: float = 8.0
    deadline: float = DEFAULT_TIMEOUT

    def __post_init__(self) -> None:
        if type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 20:
            raise SystemExit("retry.max_attempts must be an integer between 1 and 20")
        if type(self.connect_errors) is not bool:
            raise SystemExit("retry.connect_errors must be boolean")
        if (not isinstance(self.statuses, (tuple, list)) or any(
                type(status) is not int or not (status == 429 or 500 <= status <= 599)
                for status in self.statuses)):
            raise SystemExit("retry.statuses must contain only 429 or HTTP 5xx integers")
        object.__setattr__(self, "statuses", tuple(self.statuses))
        for name in ("backoff_base", "backoff_max", "deadline"):
            _config_number(getattr(self, name), f"retry.{name}")
        if self.backoff_base > self.backoff_max:
            raise SystemExit("retry.backoff_base must not exceed retry.backoff_max")


@dataclass
class Config:
    host: str
    port: int
    api_keys: list[str]
    max_body_bytes: int
    providers: dict[str, Provider]
    routes: dict[str, Route]
    public_routes: list[Route]
    catalog: CatalogConfig | None = None
    llama_cpp: LlamaCppConfig | None = None
    max_connections: int = DEFAULT_MAX_CONNECTIONS
    max_generations: int = DEFAULT_MAX_GENERATIONS
    header_read_timeout: float = DEFAULT_HEADER_READ_TIMEOUT
    body_read_timeout: float = DEFAULT_BODY_READ_TIMEOUT
    write_timeout: float = DEFAULT_WRITE_TIMEOUT
    capability_overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    bearer_session_compat: bool = False
    allow_dynamic_routes: bool = True
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    discovery_cache: DiscoveryCacheConfig | None = None

    def __post_init__(self) -> None:
        # Also validate Config instances constructed directly by embedders.
        if not isinstance(self.retry, RetryPolicy):
            raise SystemExit("retry must be a RetryPolicy")
        if self.discovery_cache is not None and not isinstance(self.discovery_cache, DiscoveryCacheConfig):
            raise SystemExit("discovery_cache must be a DiscoveryCacheConfig")
        self.capability_overrides = _freeze(self.capability_overrides)
        if type(self.allow_dynamic_routes) is not bool:
            raise SystemExit("server.allow_dynamic_routes must be boolean")
        if type(self.bearer_session_compat) is not bool:
            raise SystemExit("server.bearer_session_compat must be boolean")
        if self.bearer_session_compat and self.api_keys:
            raise SystemExit("server.bearer_session_compat requires unauthenticated client access")
        _llama_provider_names(self)
        for name in ("max_connections", "max_generations"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise SystemExit(f"server.{name} must be a positive integer")
        if self.max_generations > self.max_connections:
            raise SystemExit("server.max_generations must not exceed max_connections")
        for name in ("header_read_timeout", "body_read_timeout", "write_timeout"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not 0 < value <= threading.TIMEOUT_MAX):
                raise SystemExit(
                    f"server.{name} must be a finite positive number "
                    f"<= {threading.TIMEOUT_MAX}"
                )


def _config_secrets(config: Config) -> list[str]:
    secrets = list(config.api_keys)

    def credentials(value: Any, sensitive: bool = False) -> None:
        if isinstance(value, Mapping):
            for name, item in value.items():
                credentials(item, sensitive or bool(re.search(
                    r"(?i)key|token|secret|password|credential|headers", str(name)
                )))
        elif isinstance(value, (list, tuple)):
            for item in value:
                credentials(item, sensitive)
        elif sensitive and isinstance(value, str) and value:
            secrets.append(value)
            if value.lower().startswith("bearer "):
                secrets.append(value[7:])

    def url_secrets(url: str) -> None:
        parsed = urlsplit(url)
        if parsed.password:
            secrets.append(unquote(parsed.password))
        credentials(dict(parse_qsl(parsed.query, keep_blank_values=True)))

    for provider in config.providers.values():
        secrets.append(provider.api_key)
        url_secrets(provider.base_url)
        credentials(provider.headers, True)
        credentials(provider.litellm_kwargs)
    if config.catalog:
        url_secrets(config.catalog.url)
        credentials(config.catalog.headers, True)
    if config.llama_cpp:
        credentials(config.llama_cpp.headers, True)
        for server in config.llama_cpp.servers:
            secrets.append(server.api_key)
            url_secrets(server.url)
            credentials(server.headers, True)
    return secrets


def _resolve_credential(raw: Mapping[str, Any], *, where: str) -> str:
    """Read api_key / api_key_env / api_key_file from a config mapping."""
    for field in ("api_key_env", "api_key_file"):
        if field in raw and (type(raw[field]) is not str or not raw[field].strip()):
            raise SystemExit(f"{where}: '{field}' must be a non-empty string")
    api_key = raw.get("api_key")
    if "api_key" in raw and api_key is None and not any(
        field in raw for field in ("api_key_env", "api_key_file")
    ):
        raise SystemExit(f"{where}: 'api_key' cannot be null; omit it for an unauthenticated upstream")
    if api_key is None and raw.get("api_key_env"):
        env_name = raw["api_key_env"]
        if not isinstance(env_name, str):
            raise SystemExit(f"{where}: 'api_key_env' must be a string")
        api_key = os.environ.get(env_name)
        if api_key is None or not api_key.strip():
            raise SystemExit(
                f"{where}: environment variable {env_name!r} is not set or is empty"
            )
    if api_key is None and raw.get("api_key_file"):
        key_path = Path(str(raw["api_key_file"])).expanduser()
        try:
            api_key = key_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise SystemExit(f"{where}: cannot read api_key_file: {exc}") from exc
        if not api_key:
            raise SystemExit(f"{where}: api_key_file is empty")
    if api_key is None:
        return ""
    if not isinstance(api_key, str):
        raise SystemExit(f"{where}: 'api_key' must be a string")
    api_key = api_key.strip()
    if api_key.startswith("os.environ/"):
        raise SystemExit(f"{where}: use api_key_env for environment references")
    return api_key


def _build_provider(name: str, raw: Mapping[str, Any]) -> Provider:
    where = f"providers.{name}"
    _config_table(raw, _PROVIDER_FIELDS, where)
    if not isinstance(raw, Mapping):
        raise SystemExit(f"{where}: must be a table/object")

    dialect = raw.get("dialect")
    if not isinstance(dialect, str) or dialect not in DIALECTS:
        raise SystemExit(
            f"{where}: 'dialect' must be one of {sorted(DIALECTS)}, got {dialect!r}"
        )

    base_url = _as_str(raw, "base_url", where=where).rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise SystemExit(f"{where}: 'base_url' must start with http:// or https://")

    api_key = _resolve_credential(raw, where=where)
    _config_headers(raw.get("headers", {}), f"{where}.headers")
    _config_number(raw.get("timeout", DEFAULT_TIMEOUT), f"{where}.timeout")
    for field in ("headers", "extra_headers"):
        if field in raw.get("litellm_kwargs", {}):
            _config_headers(raw["litellm_kwargs"][field], f"{where}.litellm_kwargs.{field}")
    if "litellm_provider" in raw:
        _as_str(raw, "litellm_provider", where=where)

    headers_raw = raw.get("headers") or {}
    if not isinstance(headers_raw, Mapping):
        raise SystemExit(f"{where}: 'headers' must be a table/object")
    headers = {str(key): str(value) for key, value in headers_raw.items()}

    try:
        timeout = float(raw.get("timeout", DEFAULT_TIMEOUT))
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{where}: 'timeout' must be a number") from exc
    if timeout <= 0:
        raise SystemExit(f"{where}: 'timeout' must be positive")

    default_litellm_provider = {
        DIALECT_ANTHROPIC: "anthropic",
        DIALECT_GEMINI: "gemini",
    }.get(dialect, "openai")
    litellm_provider = str(raw.get("litellm_provider") or default_litellm_provider)

    litellm_kwargs_raw = raw.get("litellm_kwargs") or {}
    if not isinstance(litellm_kwargs_raw, Mapping):
        raise SystemExit(f"{where}: 'litellm_kwargs' must be a table/object")
    compatibility = raw.get("compatibility", "lossy" if raw.get("drop_params") is True else "strict")
    if compatibility not in ("strict", "lossy"):
        raise SystemExit(f"{where}: 'compatibility' must be 'strict' or 'lossy'")
    if "drop_params" in raw and (type(raw["drop_params"]) is not bool
                                 or raw["drop_params"] != (compatibility == "lossy")):
        raise SystemExit(f"{where}: 'drop_params' must be boolean and agree with 'compatibility'")

    session_header = raw.get("session_header", "")
    if not isinstance(session_header, str):
        raise SystemExit(f"{where}: 'session_header' must be a string")
    session_header = session_header.lower()
    if session_header and (
        re.fullmatch(_HTTP_TOKEN.decode("ascii"), session_header) is None
        or session_header in _DROP_REQUEST_HEADERS
        or session_header in {"accept", "accept-encoding", "anthropic-beta",
                              "anthropic-version", "openai-beta"}
    ):
        raise SystemExit(f"{where}: 'session_header' must be a dedicated identity header, "
                         "separate from credentials, protocol and transport headers")
    source_headers = raw.get("session_source_headers", list(_NATIVE_SESSION_HEADERS))
    if (not isinstance(source_headers, list)
        or any(not isinstance(name, str) or name.lower() not in _NATIVE_SESSION_HEADERS
               for name in source_headers)):
        raise SystemExit(f"{where}: 'session_source_headers' must be an ordered list "
                         f"selected from {_NATIVE_SESSION_HEADERS}")
    source_headers = tuple(name.lower() for name in source_headers)
    if len(set(source_headers)) != len(source_headers):
        raise SystemExit(f"{where}: 'session_source_headers' contains duplicate headers")
    cache_fallback = raw.get("session_prompt_cache_key", False)
    if type(cache_fallback) is not bool:
        raise SystemExit(f"{where}: 'session_prompt_cache_key' must be boolean")
    session_missing = raw.get("session_missing", "omit")
    if session_missing not in ("omit", "reject"):
        raise SystemExit(f"{where}: 'session_missing' must be 'omit' or 'reject'")
    if not session_header and any(key in raw for key in (
        "session_source_headers", "session_prompt_cache_key", "session_missing",
    )):
        raise SystemExit(f"{where}: session policy requires 'session_header'")
    reserved_sessions = {"x-opencode-session", *_NATIVE_SESSION_HEADERS}
    if session_header:
        reserved_sessions.add(session_header)
    for header_source in (headers, litellm_kwargs_raw.get("headers", {}),
                          litellm_kwargs_raw.get("extra_headers", {})):
        if isinstance(header_source, Mapping) and any(
            str(key).lower() in reserved_sessions for key in header_source
        ):
            raise SystemExit(f"{where}: static session headers are forbidden; "
                             "supply a conversation identity on each request")

    auth_header = raw.get("auth_header") or ""
    if not isinstance(auth_header, str):
        raise SystemExit(f"{where}: 'auth_header' must be a string")
    auth_header = auth_header.strip()
    if auth_header and auth_header.lower() not in ("bearer", "x-api-key", "api-key"):
        raise SystemExit(
            f"{where}: 'auth_header' must be 'bearer', 'x-api-key', 'api-key', "
            f"or empty, got {auth_header!r}"
        )
    auth_header = auth_header.lower()

    count_tokens = raw.get("count_tokens")
    if count_tokens and dialect != DIALECT_ANTHROPIC:
        raise SystemExit(f"{where}: 'count_tokens' can only be enabled for the "
                         f"{DIALECT_ANTHROPIC!r} dialect")

    return Provider(
        name=name,
        dialect=dialect,
        base_url=base_url,
        api_key=api_key,
        headers=headers,
        timeout=timeout,
        litellm_provider=litellm_provider,
        litellm_kwargs=dict(litellm_kwargs_raw),
        drop_params=compatibility == "lossy",
        session_header=session_header,
        auth_header=auth_header,
        session_source_headers=source_headers,
        session_prompt_cache_key=cache_fallback,
        session_missing=session_missing,
        count_tokens=count_tokens,
    )


def _split_server_urls(value: str) -> list[str]:
    """Split LLAMA_SERVER_URL-style values (semicolon or comma separated)."""
    return [part.strip() for part in re.split(r"[;,]", value) if part.strip()]


def _normalize_llama_url(url: str, *, where: str) -> str:
    url = url.strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        raise SystemExit(f"{where}: server URL must start with http:// or https://")
    return url


def _build_llama_cpp(raw: Mapping[str, Any]) -> LlamaCppConfig:
    """Validate the [llama_cpp] table used for automatic model discovery."""
    where = "llama_cpp"
    _config_table(raw, {
        **_CREDENTIAL_FIELDS, "headers": (dict,), "servers": (list, str, dict),
        "ports": (list, int), "timeout": (int, float), "refresh_seconds": (int, float),
        "context_fallback": (int,), "limit_overrides": (dict,), "prefix": (str,),
        "aliases": (bool,), "compatibility": (str,),
    }, where)
    compatibility = raw.get("compatibility", "strict")
    if compatibility not in ("strict", "lossy"):
        raise SystemExit(f"{where}: 'compatibility' must be 'strict' or 'lossy'")
    _config_headers(raw.get("headers", {}), f"{where}.headers")
    _config_number(raw.get("timeout", DEFAULT_LLAMA_PROBE_TIMEOUT), f"{where}.timeout")
    _config_number(raw.get("refresh_seconds", DEFAULT_LLAMA_REFRESH),
                   f"{where}.refresh_seconds", zero=True)
    if not isinstance(raw, Mapping):
        raise SystemExit(f"{where}: must be a table/object")

    headers_raw = raw.get("headers") or {}
    if not isinstance(headers_raw, Mapping):
        raise SystemExit(f"{where}: 'headers' must be a table/object")
    global_headers = {str(key): str(value) for key, value in headers_raw.items()}
    global_api_key = _resolve_credential(raw, where=where)

    try:
        probe_timeout = float(raw.get("timeout", DEFAULT_LLAMA_PROBE_TIMEOUT))
        refresh_seconds = float(
            raw.get("refresh_seconds", DEFAULT_LLAMA_REFRESH)
        )
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"{where}: 'timeout' and 'refresh_seconds' "
            "must be numbers"
        ) from exc
    if not math.isfinite(probe_timeout) or not 0 < probe_timeout <= threading.TIMEOUT_MAX:
        raise SystemExit(f"{where}: 'timeout' must be finite, positive and <= threading.TIMEOUT_MAX")
    if not math.isfinite(refresh_seconds) or not 0 <= refresh_seconds <= threading.TIMEOUT_MAX:
        raise SystemExit(f"{where}: 'refresh_seconds' must be finite, >= 0 and <= threading.TIMEOUT_MAX")
    context_fallback = raw.get("context_fallback", DEFAULT_LLAMA_CONTEXT)
    if type(context_fallback) is not int or context_fallback <= 0:
        raise SystemExit(f"{where}: 'context_fallback' must be a positive integer")

    overrides_raw = raw.get("limit_overrides", {})
    if not isinstance(overrides_raw, Mapping):
        raise SystemExit(f"{where}.limit_overrides: must be a table/object")
    limit_overrides: dict[str, dict[str, int | None]] = {}
    for model_id, limits in overrides_raw.items():
        label = f"{where}.limit_overrides[{model_id!r}]"
        if not isinstance(model_id, str) or not model_id.strip() or model_id != model_id.strip():
            raise SystemExit(f"{label}: model ID must be a nonempty string without surrounding whitespace")
        if not isinstance(limits, Mapping):
            raise SystemExit(f"{label}: must be a table/object")
        for key, value in limits.items():
            if key not in LLAMA_LIMIT_FIELDS:
                raise SystemExit(f"{label}: unknown limit {key!r}")
            minimum = 0 if key == "default_output_tokens" else 1
            if value is not None and (type(value) is not int or value < minimum):
                raise SystemExit(f"{label}.{key}: must be an integer >= {minimum} or null")
        limit_overrides[model_id] = dict(limits)

    prefix = raw.get("prefix") or ""
    if not isinstance(prefix, str):
        raise SystemExit(f"{where}: 'prefix' must be a string")

    servers_raw = raw.get("servers") or []
    if isinstance(servers_raw, (str, Mapping)):
        servers_raw = [servers_raw]
    if not isinstance(servers_raw, (list, tuple)):
        raise SystemExit(f"{where}: 'servers' must be an array")

    ports_raw = raw.get("ports") or []
    if isinstance(ports_raw, int):
        ports_raw = [ports_raw]
    if not isinstance(ports_raw, (list, tuple)):
        raise SystemExit(f"{where}: 'ports' must be an array of integers")

    entries: list[Any] = list(servers_raw)
    for index, port in enumerate(ports_raw):
        if type(port) is not int:
            raise SystemExit(f"{where}.ports[{index}]: port must be an integer")
        try:
            port_number = int(port)
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"{where}.ports[{index}]: port must be an integer"
            ) from exc
        if not 1 <= port_number <= 65535:
            raise SystemExit(
                f"{where}.ports[{index}]: port must be between 1 and 65535"
            )
        entries.append(f"http://127.0.0.1:{port_number}")

    if not entries:
        # Same fallback chain as pi-llama-cpp: environment first, then the
        # conventional llama-server port.
        entries = _split_server_urls(os.environ.get("LLAMA_SERVER_URL", ""))
        if not entries:
            entries = [DEFAULT_LLAMA_URL]

    servers: list[LlamaCppServer] = []
    seen: set[str] = set()
    explicit_names: dict[str, int] = {}
    for index, entry in enumerate(entries):
        entry_where = f"{where}.servers[{index}]"
        if isinstance(entry, str):
            url = _normalize_llama_url(entry, where=entry_where)
            name = ""
            api_key = global_api_key
            headers = dict(global_headers)
        elif isinstance(entry, Mapping):
            _config_table(entry, {**_CREDENTIAL_FIELDS, "url": (str,), "name": (str,),
                                  "headers": (dict,)}, entry_where)
            _config_headers(entry.get("headers", {}), f"{entry_where}.headers")
            url = _normalize_llama_url(
                _as_str(entry, "url", where=entry_where), where=entry_where
            )
            name_raw = entry.get("name") or ""
            if not isinstance(name_raw, str):
                raise SystemExit(f"{entry_where}: 'name' must be a string")
            name = name_raw.strip()
            has_server_key = any(
                key in entry for key in ("api_key", "api_key_env", "api_key_file")
            )
            api_key = (
                _resolve_credential(entry, where=entry_where)
                if has_server_key
                else global_api_key
            )
            entry_headers_raw = entry.get("headers") or {}
            if not isinstance(entry_headers_raw, Mapping):
                raise SystemExit(
                    f"{entry_where}: 'headers' must be a table/object"
                )
            headers = dict(global_headers)
            headers.update(
                {str(key): str(value) for key, value in entry_headers_raw.items()}
            )
        else:
            raise SystemExit(
                f"{entry_where}: must be a URL string or a table with 'url'"
            )
        if name:
            if name in explicit_names:
                raise SystemExit(
                    f"{entry_where}.name: duplicate explicit provider name {name!r}; "
                    f"already used by llama_cpp.servers[{explicit_names[name]}]. "
                    "Assign each server a unique name or omit names to generate them."
                )
            explicit_names[name] = index
        if url in seen:
            continue
        seen.add(url)
        servers.append(
            LlamaCppServer(
                url=url, name=name, api_key=api_key, headers=headers
            )
        )

    return LlamaCppConfig(
        servers=servers,
        headers=global_headers,
        probe_timeout=probe_timeout,
        refresh_seconds=refresh_seconds,
        context_fallback=context_fallback,
        prefix=prefix,
        expose_aliases=bool(raw.get("aliases", True)),
        limit_overrides=limit_overrides,
        compatibility=compatibility,
    )


def build_config(raw: Mapping[str, Any]) -> Config:
    """Validate a config mapping and return a Config."""
    _config_table(raw, {name: (dict,) for name in (
        "server", "providers", "catalog", "llama_cpp", "capabilities", "retry", "discovery_cache",
    )} | {"models": (list,)}, "config")
    _config_finite(raw)
    cache_raw = raw.get("discovery_cache")
    if cache_raw is not None:
        _config_table(cache_raw, {"path": (str,), "max_age_seconds": (int, float)}, "discovery_cache")
        if "path" not in cache_raw:
            raise SystemExit("discovery_cache.path is required")
    discovery_cache = DiscoveryCacheConfig(**cache_raw) if cache_raw is not None else None
    retry_raw = raw.get("retry", {})
    _config_table(retry_raw, {
        "max_attempts": (int,), "statuses": (list,), "connect_errors": (bool,),
        "backoff_base": (int, float), "backoff_max": (int, float),
        "deadline": (int, float),
    }, "retry")
    retry = RetryPolicy(**retry_raw)

    capabilities_raw = raw.get("capabilities", {})
    if not isinstance(capabilities_raw, Mapping) or capabilities_raw.keys() - {"overrides"}:
        raise SystemExit("capabilities: expected a table containing only 'overrides'")
    capability_overrides = router_metadata.validate_overrides(capabilities_raw.get("overrides", {}))
    server_raw = raw.get("server", {})
    _config_table(server_raw, {
        "host": (str,), "port": (int,), "api_keys": (list,), "auth_mode": (str,),
        "max_body_bytes": (int,), "max_connections": (int,), "max_generations": (int,),
        "header_read_timeout": (int, float), "body_read_timeout": (int, float),
        "write_timeout": (int, float), "allow_dynamic_routes": (bool,),
        "bearer_session_compat": (bool,),
    }, "server")
    if not isinstance(server_raw, Mapping):
        raise SystemExit("server: must be a table/object")

    host = _as_str({"host": server_raw.get("host", "127.0.0.1")}, "host", where="server")
    try:
        port = int(server_raw.get("port", 4000))
    except (TypeError, ValueError) as exc:
        raise SystemExit("server.port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise SystemExit("server.port must be between 1 and 65535")

    keys_raw = server_raw.get("api_keys", [])
    _config_strings(keys_raw, "server.api_keys")
    api_keys = [key.strip() for key in keys_raw]
    auth_mode = server_raw.get("auth_mode")
    if auth_mode not in (None, "none", "required"):
        raise SystemExit("server.auth_mode must be 'none' or 'required'")
    if auth_mode == "none" and keys_raw:
        raise SystemExit("server.auth_mode='none' conflicts with server.api_keys")

    env_keys = os.environ.get("MODEL_ROUTER_API_KEYS", "")
    if env_keys and auth_mode != "none":
        api_keys.extend(part.strip() for part in env_keys.split(",") if part.strip())
    env_key = os.environ.get("MODEL_ROUTER_API_KEY", "").strip()
    if env_key and auth_mode != "none":
        api_keys.append(env_key)
    # De-duplicate while preserving order.
    api_keys = list(dict.fromkeys(api_keys))
    if auth_mode == "required" and not api_keys:
        raise SystemExit("server.auth_mode='required' requires at least one non-empty API key")

    try:
        max_body = int(server_raw.get("max_body_bytes", DEFAULT_MAX_BODY))
    except (TypeError, ValueError) as exc:
        raise SystemExit("server.max_body_bytes must be an integer") from exc
    if max_body <= 0:
        raise SystemExit("server.max_body_bytes must be positive")

    llama_cpp_raw = raw.get("llama_cpp")
    llama_cpp = (
        _build_llama_cpp(llama_cpp_raw) if llama_cpp_raw is not None else None
    )

    providers_raw = raw.get("providers") or {}
    if not isinstance(providers_raw, Mapping):
        raise SystemExit("providers: must be a table/object")
    if not providers_raw and llama_cpp is None:
        raise SystemExit(
            "providers: at least one provider is required (or configure "
            "[llama_cpp] for automatic local discovery)"
        )
    providers = {
        name: _build_provider(name, value)
        for name, value in providers_raw.items()
    }
    if any(not name.strip() or name != name.strip() or "/" in name for name in providers):
        raise SystemExit("providers: names must be non-empty without whitespace padding or '/'")

    catalog_raw = raw.get("catalog")
    catalog: CatalogConfig | None = None
    if catalog_raw is not None:
        _config_table(catalog_raw, {
            "url": (str,), "headers": (dict,), "timeout": (int, float),
            "refresh_seconds": (int, float), "model_providers": (dict,),
            "rules": (list,), "metadata_source": (str, type(None)),
        }, "catalog")
        _config_headers(catalog_raw.get("headers", {}), "catalog.headers")
        _config_number(catalog_raw.get("timeout", 30.0), "catalog.timeout")
        _config_number(catalog_raw.get("refresh_seconds", 300.0),
                       "catalog.refresh_seconds", zero=True)
        if not isinstance(catalog_raw, Mapping):
            raise SystemExit("catalog: must be a table/object")
        catalog_url = _as_str(catalog_raw, "url", where="catalog")
        if not catalog_url.startswith(("http://", "https://")):
            raise SystemExit("catalog: 'url' must start with http:// or https://")
        catalog_headers_raw = catalog_raw.get("headers") or {}
        if not isinstance(catalog_headers_raw, Mapping):
            raise SystemExit("catalog: 'headers' must be a table/object")
        try:
            catalog_timeout = float(catalog_raw.get("timeout", 30.0))
            catalog_refresh = float(catalog_raw.get("refresh_seconds", 300.0))
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                "catalog: 'timeout' and 'refresh_seconds' must be numbers"
            ) from exc
        if (not math.isfinite(catalog_timeout) or not math.isfinite(catalog_refresh)
                or not 0 < catalog_timeout <= threading.TIMEOUT_MAX
                or not 0 <= catalog_refresh <= threading.TIMEOUT_MAX):
            raise SystemExit(
                "catalog: 'timeout' must be positive and 'refresh_seconds' >= 0; "
                "both must be finite and <= threading.TIMEOUT_MAX"
            )
        model_providers_raw = catalog_raw.get("model_providers", {})
        if not isinstance(model_providers_raw, Mapping):
            raise SystemExit("catalog.model_providers: must be a table/object")
        model_providers: dict[str, str] = {}
        for model_id, provider_name in model_providers_raw.items():
            if (not isinstance(model_id, str) or not model_id.strip()
                    or model_id != model_id.strip()):
                raise SystemExit(
                    "catalog.model_providers: model IDs must be nonempty strings "
                    "without surrounding whitespace"
                )
            if not isinstance(provider_name, str) or provider_name not in providers:
                raise SystemExit(
                    f"catalog.model_providers[{model_id!r}]: unknown provider {provider_name!r}"
                )
            model_providers[model_id] = provider_name
        rules_raw = catalog_raw.get("rules", [])
        if not isinstance(rules_raw, (list, tuple)):
            raise SystemExit("catalog.rules: must be an array")
        if not rules_raw and not model_providers:
            raise SystemExit(
                "catalog: at least one model_providers entry or [[catalog.rules]] is required"
            )
        rules: list[CatalogRule] = []
        for index, rule_raw in enumerate(rules_raw):
            where = f"catalog.rules[{index}]"
            _config_table(rule_raw, {"match": (str,), "provider": (str,)}, where)
            if not isinstance(rule_raw, Mapping):
                raise SystemExit(f"{where}: must be a table/object")
            match = _as_str(rule_raw, "match", where=where)
            rule_provider = _as_str(rule_raw, "provider", where=where)
            if rule_provider not in providers:
                raise SystemExit(
                    f"{where}: unknown provider {rule_provider!r}; "
                    f"known providers: {sorted(providers)}"
                )
            try:
                pattern = re.compile(match)
            except re.error as exc:
                raise SystemExit(f"{where}: invalid 'match' regex: {exc}") from exc
            rules.append(CatalogRule(pattern=pattern, provider_name=rule_provider))
        metadata_source = catalog_raw.get("metadata_source")
        if metadata_source not in (None, "opencode-go-v1"):
            raise SystemExit("catalog.metadata_source: expected 'opencode-go-v1' or null")
        catalog = CatalogConfig(
            url=catalog_url,
            headers={
                str(key): str(value)
                for key, value in catalog_headers_raw.items()
            },
            timeout=catalog_timeout,
            refresh_seconds=catalog_refresh,
            rules=rules,
            model_providers=model_providers,
            metadata_source=metadata_source,
        )

    models_raw = raw.get("models") or []
    if not isinstance(models_raw, (list, tuple)):
        raise SystemExit("models: must be an array")
    if not models_raw and catalog is None and llama_cpp is None:
        raise SystemExit(
            "models: at least one model route is required (or configure "
            "[catalog] / [llama_cpp])"
        )

    routes: dict[str, Route] = {}
    public_routes: list[Route] = []

    for index, model_raw in enumerate(models_raw):
        where = f"models[{index}]"
        _config_table(model_raw, {
            "name": (str,), "provider": (str,), "model": (str,), "dialect": (str,),
            "metadata": (dict,), "enabled": (bool,), "aliases": (list,),
        }, where)
        _config_strings(model_raw.get("aliases", []), f"{where}.aliases")
        if not isinstance(model_raw, Mapping):
            raise SystemExit(f"{where}: must be a table/object")
        name = _as_str(model_raw, "name", where=where)
        provider_name = _as_str(model_raw, "provider", where=where)
        provider = providers.get(provider_name)
        if provider is None:
            raise SystemExit(
                f"{where}: unknown provider {provider_name!r}; "
                f"known providers: {sorted(providers)}"
            )

        upstream_model = model_raw.get("model")
        if upstream_model is None:
            upstream_model = (
                name.split("/", 1)[1]
                if name.startswith(provider_name + "/") and "/" in name
                else name
            )
        if not isinstance(upstream_model, str) or not upstream_model.strip():
            raise SystemExit(f"{where}: 'model' must be a non-empty string")

        dialect = model_raw.get("dialect", provider.dialect)
        if dialect not in DIALECTS:
            raise SystemExit(
                f"{where}: 'dialect' must be one of {sorted(DIALECTS)}, got {dialect!r}"
            )

        metadata = model_raw.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise SystemExit(f"{where}: 'metadata' must be a table/object")

        route = Route(
            name=name,
            provider_name=provider_name,
            upstream_model=upstream_model.strip(),
            dialect=str(dialect),
            metadata=dict(metadata),
            enabled=bool(model_raw.get("enabled", True)),
        )

        aliases_raw = model_raw.get("aliases") or []
        if isinstance(aliases_raw, str):
            aliases_raw = [aliases_raw]
        if not isinstance(aliases_raw, (list, tuple)):
            raise SystemExit(f"{where}: 'aliases' must be an array of strings")

        for public_name in [name, *[str(alias) for alias in aliases_raw]]:
            public_name = public_name.strip()
            if not public_name:
                continue
            if public_name in routes:
                raise SystemExit(
                    f"{where}: duplicate model name/alias {public_name!r}"
                )
            routes[public_name] = route

        public_routes.append(route)

    return Config(
        host=host,
        port=port,
        api_keys=api_keys,
        max_body_bytes=max_body,
        providers=providers,
        routes=routes,
        public_routes=public_routes,
        catalog=catalog,
        llama_cpp=llama_cpp,
        capability_overrides=capability_overrides,
        allow_dynamic_routes=server_raw.get("allow_dynamic_routes", True),
        bearer_session_compat=server_raw.get("bearer_session_compat", False),
        retry=retry,
        discovery_cache=discovery_cache,
        **{name: server_raw.get(name, default) for name, default in (
            ("max_connections", DEFAULT_MAX_CONNECTIONS),
            ("max_generations", DEFAULT_MAX_GENERATIONS),
            ("header_read_timeout", DEFAULT_HEADER_READ_TIMEOUT),
            ("body_read_timeout", DEFAULT_BODY_READ_TIMEOUT),
            ("write_timeout", DEFAULT_WRITE_TIMEOUT),
        )},
    )


def load_config(path: Path) -> Config:
    return build_config(_load_config_file(path))


_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def default_env_files() -> list[Path]:
    """Credential files loaded automatically before the config is read."""
    candidates = []
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        candidates.append(Path(xdg) / "ai" / "credentials.env")
    candidates.append(Path.home() / ".config" / "ai" / "credentials.env")
    seen: set[Path] = set()
    unique = []
    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def load_env_file(path: Path, *, override: bool = False) -> int:
    """Load KEY=value lines from a shell-style env file.

    Supports blank lines, # comments, an optional leading `export`, and single
    or double quoted values.  Existing environment variables win unless
    override is true.  Returns the number of keys seen.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"Cannot read env file {path}: {exc}") from exc

    count = 0
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, raw = line.partition("=")
        key = key.strip()
        if not separator or not _ENV_KEY.match(key):
            LOG.warning("Skipping malformed line %d in %s", line_number, path)
            continue
        value = raw.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        count += 1
        if override or key not in os.environ:
            os.environ[key] = value
    return count


def load_default_credentials() -> list[Path]:
    loaded = []
    for path in default_env_files():
        if path.is_file():
            load_env_file(path)
            loaded.append(path)
    return loaded


def _llama_provider_name(server: LlamaCppServer, taken: set[str]) -> str:
    """Choose a generated name against identities reserved before probing."""
    if server.name:
        base = server.name
    else:
        parsed = urlsplit(server.url)
        base = "llama-" + (parsed.netloc + parsed.path).strip("/")
        base = re.sub(r"[^A-Za-z0-9_.-]+", "-", base).strip("-") or "llama"
    name = base
    index = 2
    while name in taken:
        name = f"{base}-{index}"
        index += 1
    return name


def _llama_provider_names(config: Config) -> dict[str, str]:
    """Reserve explicit names first, then generated names in config order."""
    if config.llama_cpp is None:
        return {}
    taken = set(config.providers)
    explicit: dict[str, int] = {}
    for index, server in enumerate(config.llama_cpp.servers):
        if not server.name:
            continue
        where = f"llama_cpp.servers[{index}].name"
        if server.name in explicit:
            raise SystemExit(
                f"{where}: duplicate explicit provider name {server.name!r}; "
                f"already used by llama_cpp.servers[{explicit[server.name]}]. "
                "Assign each server a unique name or omit names to generate them."
            )
        if server.name in config.providers:
            raise SystemExit(
                f"{where}: provider name {server.name!r} is already configured "
                "in providers; choose a unique server name or omit it."
            )
        explicit[server.name] = index
        taken.add(server.name)
    names: dict[str, str] = {}
    for server in config.llama_cpp.servers:
        name = server.name or _llama_provider_name(server, taken)
        names[server.url] = name
        taken.add(name)
    return names


def _llama_base_url(url: str) -> str:
    """The OpenAI-compatible base URL of a llama.cpp server."""
    return url if url.endswith("/v1") else url + "/v1"
