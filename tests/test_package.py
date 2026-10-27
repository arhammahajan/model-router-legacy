"""Import and distribution boundaries added by the package extraction."""
from __future__ import annotations

import ast
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import model_router


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("module,names", [
    ("config", ("Config", "Provider", "Route", "RetryPolicy", "build_config", "load_config")),
    ("discovery", ("RegistrySnapshot", "CatalogSnapshot", "LlamaSnapshot")),
    ("app", ("App",)),
    ("http", ("Handler", "RouterHTTPServer", "serve")),
    ("streams", ("Plan", "StreamState")),
    ("cli", ("main",)),
])
def test_public_embedding_imports_preserve_object_identity(module, names):
    owner = importlib.import_module(f"model_router.{module}")
    for name in names:
        assert getattr(model_router, name) is getattr(owner, name)


def test_configuration_and_discovery_import_without_translation_backend():
    # A separate interpreter prevents previously imported adapters from hiding
    # a cycle or a newly introduced dependency on LiteLLM's private modules.
    subprocess.run([sys.executable, "-c", """
import builtins
import sys

original = builtins.__import__
def importing(name, *args, **kwargs):
    assert name.split('.')[0] not in {'litellm', 'openai'}, name
    return original(name, *args, **kwargs)
builtins.__import__ = importing

from model_router.config import build_config
from model_router.discovery import RegistrySnapshot
import model_router
config = build_config({
    'server': {'auth_mode': 'none'},
    'providers': {'local': {'dialect': 'openai-chat', 'base_url': 'http://127.0.0.1:1/v1',
                            'api_key': 'dummy-package-test-key'}},
    'models': [{'name': 'test', 'provider': 'local', 'model': 'test'}],
})
assert set(config.providers) == {'local'}
assert model_router.Config is type(config)
assert 'model_router.adapters' not in sys.modules
assert 'model_router.app' not in sys.modules
"""], check=True, capture_output=True, text=True)


def test_only_adapter_imports_litellm_or_openai():
    package = Path(model_router.__file__).parent
    for file in package.glob("*.py"):
        if file.name == "adapters.py":
            continue
        for node in ast.walk(ast.parse(file.read_text())):
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imports = [node.module or ""]
            else:
                continue
            assert not any(name.split(".")[0] in {"litellm", "openai"}
                           for name in imports), file


@pytest.mark.parametrize("launcher", ["script", "module", "console"])
def test_cli_launchers_share_version_help_and_offline_validation(tmp_path, launcher):
    if launcher == "script":
        command = [sys.executable, str(ROOT / "model_router.py")]
    elif launcher == "module":
        command = [sys.executable, "-m", "model_router"]
    else:
        console = Path(sys.executable).with_name("model-router")
        if sys.platform == "win32":
            console = console.with_suffix(".exe")
        command = [str(console)]
    config = tmp_path / "router.json"
    config.write_text(json.dumps({
        "server": {"auth_mode": "none"},
        "providers": {"local": {"dialect": "openai-chat", "base_url": "http://127.0.0.1:1/v1",
                                "api_key": "dummy-package-test-key"}},
        "models": [{"name": "test", "provider": "local", "model": "test"}],
    }))
    # No ambient credentials or dotenv paths; no provider needs probing.
    env = {key: os.environ[key] for key in ("PATH", "SYSTEMROOT") if key in os.environ}
    env.update(LITELLM_LOCAL_MODEL_COST_MAP="True", DO_NOT_TRACK="True")

    def run(*args):
        return subprocess.check_output(command + list(args), cwd=tmp_path, env=env,
                                       text=True).strip()

    assert run("--version") == model_router.__version__
    help_text = run("--help")
    assert "--check-readiness" in help_text
    assert "--no-default-env" in help_text
    assert json.loads(run("--no-default-env", "-c", str(config),
                          "--validate-config", "--offline")) == {
        "status": "valid", "network_checked": False,
    }
