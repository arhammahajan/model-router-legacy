"""Smoke-test the installed wheel outside the checkout, using dummy credentials."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    # Deliberately omit ambient provider credentials, dotenv and PYTHONPATH.
    env = {key: os.environ[key] for key in ("PATH", "SYSTEMROOT") if key in os.environ}
    env.update(LITELLM_LOCAL_MODEL_COST_MAP="True", DO_NOT_TRACK="True",
               OPENCODE_API_KEY="dummy-release-smoke-key")
    console = Path(sys.executable).with_name("model-router")
    if sys.platform == "win32":
        console = console.with_suffix(".exe")
    with tempfile.TemporaryDirectory(prefix="model-router-smoke-") as directory:
        cwd = Path(directory)
        shutil.copyfile(ROOT / "router.toml", cwd / "router.toml")

        def run(command: list[str]) -> str:
            return subprocess.check_output(command, cwd=cwd, env=env, text=True).strip()

        installed = run([sys.executable, "-I", "-c", """
import importlib.metadata as metadata
import importlib
import json
from pathlib import Path
import socket
import sys

def forbidden(*args, **kwargs):
    raise AssertionError('Release smoke must not read credentials or use the network/App')

socket.socket.connect = forbidden
socket.getaddrinfo = forbidden
import model_router as router
import model_router.cli as cli
import router_metadata
assert not Path(router.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
assert not Path(router_metadata.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
assert metadata.version('litellm-model-router') == router.__version__
assert metadata.version('litellm') == router.TESTED_LITELLM_VERSION
entry, = [entry for entry in metadata.distribution('litellm-model-router').entry_points
          if entry.group == 'console_scripts' and entry.name == 'model-router']
assert entry.value == 'model_router.cli:main'
assert entry.load() is cli.main is router.main
for name in ('config', 'headers', 'transport', 'adapters', 'discovery', 'protocol',
             'streams', 'app', 'http', 'cli', 'constants', 'common'):
    module = importlib.import_module('model_router.' + name)
    assert Path(module.__file__).parent == Path(router.__file__).parent
assert router_metadata.go('gpt-6-luna', 'openai-responses')['sources']['go.review']['applicable']
cli.load_default_credentials = forbidden
cli.load_env_file = forbidden
cli.App = forbidden
assert cli._find_default_config() == Path('router.toml')
config = router.load_config(Path('router.toml'))
assert config.providers['opencode-chat'].api_key == 'dummy-release-smoke-key'
assert router.main(['--no-default-env', '--validate-config', '--offline']) == 0
print(json.dumps({'version': router.__version__, 'module': router.__file__}))
""", str(ROOT)])
        details = json.loads(installed.splitlines()[-1])
        assert run([str(console), "--version"]) == details["version"]
        assert run([sys.executable, "-I", "-m", "model_router", "--version"]) == details["version"]
        result = json.loads(run([str(console), "--no-default-env", "--validate-config", "--offline"]))
        assert result == {"status": "valid", "network_checked": False}
        print(f"Installed wheel, console version, bundled metadata and default TOML: OK ({details['version']})")


if __name__ == "__main__":
    main()
