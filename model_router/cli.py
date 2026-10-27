"""Command-line parsing, configuration selection and server lifecycle."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .app import App
from .common import ErrorSanitizer, _error_message
from .config import (
    Config,
    _as_str,
    _config_secrets,
    _config_strings,
    load_config,
    load_default_credentials,
    load_env_file,
)
from .constants import LOG, __version__
from .http import serve


def _find_default_config() -> Path | None:
    for name in (
        "router.toml",
        "router.json",
        "config.toml",
        "config.json",
        "router.yaml",
        "config.yaml",
    ):
        path = Path(name)
        if path.exists():
            return path
    return None


def _print_summary(
    config: Config,
    host: str,
    port: int,
    *,
    listening: bool = True,
    app: App | None = None,
) -> None:
    if listening:
        LOG.info("model-router %s listening on http://%s:%d", __version__, host, port)
    else:
        LOG.info("model-router %s config OK for http://%s:%d", __version__, host, port)
    for route in config.public_routes:
        LOG.info(
            "  %-28s -> %-18s %-40s [%s]",
            route.name,
            route.provider_name,
            route.upstream_model,
            route.dialect,
        )
    if app is not None and config.catalog is not None:
        status = app.catalog_status()
        detail = f" (error: {status['error']})" if status["error"] else ""
        LOG.info(
            "catalog: %d models from %s%s",
            status["models"],
            config.catalog.url,
            detail,
        )
        if not listening:
            for name in sorted(app._registry.catalog.routes):
                LOG.info("  %s", name)
    if app is not None and config.llama_cpp is not None:
        status = app.llama_cpp_status()
        LOG.info(
            "llama.cpp: %d models from %d server(s)",
            status["models"],
            len(config.llama_cpp.servers),
        )
        for server in status["servers"]:
            LOG.info(
                "  %s: %s%s",
                server["url"],
                "reachable" if server["reachable"] else "unavailable",
                (
                    f" ({server['mode']}, {server['models']} models)"
                    if server["reachable"]
                    else f" ({server['error']})"
                ),
            )
        if not listening:
            for route in sorted(app._registry.llama.public, key=lambda item: item.name):
                LOG.info(
                    "  llama: %s (context %s)",
                    route.name,
                    route.metadata.get("context_window"),
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="LiteLLM-backed multi-dialect model router."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="TOML/JSON config file (default: ./router.toml or ./router.json)",
    )
    parser.add_argument("--host", help="Override server.host")
    parser.add_argument("--port", type=int, help="Override server.port")
    parser.add_argument(
        "--api-key",
        action="append",
        default=None,
        help="Require this router API key (repeatable). Overrides config.",
    )
    parser.add_argument(
        "--env-file",
        action="append",
        type=Path,
        default=[],
        help=(
            "Load KEY=value credentials from this file before reading the config "
            "(repeatable). Existing environment variables win."
        ),
    )
    parser.add_argument(
        "--no-default-env",
        action="store_true",
        help="Do not auto-load ~/.config/ai/credentials.env",
    )
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--validate-config", action="store_true",
                        help="Validate config/credentials without networking or creating an App; exit 0/2")
    checks.add_argument(
        "--check-readiness", "--check", dest="check", action="store_true",
        help="Validate, attempt discovery once (also in startup-only mode), print routes/readiness, "
             "exit 0 ready/3 unready; --check is a compatibility alias. Static endpoints are unprobed",
    )
    parser.add_argument("--offline", action="store_true",
                        help="With a check only: forbid network work; readiness is indeterminate (exit 4). "
                             "Use --validate-config for offline validation (exit 0/2)")
    parser.add_argument("--verbose", action="store_true", help="Enable router debug logs (no dependency body logging)")
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args(argv)
    if args.offline and not (args.check or args.validate_config):
        parser.error("--offline requires --validate-config or --check-readiness/--check")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    # Dependency debug output may include full generation bodies. Verbose is
    # restricted to router diagnostics rather than enabling payload logging.
    dependency_names = {"LiteLLM", "LiteLLM Router", "LiteLLM Proxy", "litellm", "httpx", "httpcore", "openai"}
    dependency_names.update(name for name in logging.Logger.manager.loggerDict
                            if name.startswith(("LiteLLM", "litellm", "httpx", "httpcore", "openai")))
    for noisy in dependency_names:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    try:
        for env_file in args.env_file:
            load_env_file(env_file)
            LOG.info("Loaded credentials from %s", env_file)
        if not args.no_default_env:
            for env_file in load_default_credentials():
                LOG.info("Loaded credentials from %s", env_file)

        config_path = args.config or _find_default_config()
        if config_path is None:
            raise SystemExit("No config found; pass --config /path/to/router.toml")

        config = load_config(config_path)
        if args.host is not None:
            config.host = _as_str({"host": args.host}, "host", where="server")
        if args.port is not None:
            if not 1 <= args.port <= 65535:
                raise SystemExit("--port must be between 1 and 65535")
            config.port = args.port
        if args.api_key:
            _config_strings(args.api_key, "--api-key")
            config.api_keys = list(dict.fromkeys(key.strip() for key in args.api_key))
    except SystemExit as exc:
        # Configuration loaders also serve embedders and raise text errors;
        # the CLI consistently maps those errors to usage/config exit code 2.
        parser.error(str(exc))

    if args.validate_config:
        # No HTTP client, probes, worker, server, or initial discovery attempt.
        print(json.dumps({"status": "valid", "network_checked": False}))
        return 0

    if args.check:
        if args.offline:
            print(json.dumps({"status": "indeterminate", "network_checked": False,
                              "reason": "offline_readiness_requires_discovery"}))
            return 4
        app = None
        try:
            app = App(config)
            _print_summary(
                config, config.host, config.port, listening=False, app=app
            )
            readiness = app.readiness_payload()
            print(json.dumps({**readiness, "discovery": app.discovery_summary()}))
            return 0 if readiness["ready"] else 3
        except Exception as exc:
            LOG.error("Readiness check failed: %s", _error_message(exc, ErrorSanitizer(_config_secrets(config))))
            return 1
        finally:
            if app is not None:
                app.close()

    app = App(config)
    app.verbose = args.verbose
    httpd = None
    try:
        httpd = serve(app, config.host, config.port)
        _print_summary(config, config.host, config.port, app=app)
        httpd.serve_forever()
    except KeyboardInterrupt:
        LOG.info("Shutting down")
    finally:
        if httpd is not None:
            httpd.server_close()
        app.close()
    return 0
