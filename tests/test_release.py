"""Release contracts: exact adapters, synchronized manifests and offline startup."""
from __future__ import annotations

import builtins
from importlib import metadata
from pathlib import Path
import runpy

from packaging.markers import default_environment
from packaging.requirements import Requirement
import pytest

import model_router as mr
import model_router.adapters as router_adapters
import model_router.cli as router_cli
import model_router.config as router_configuration


ROOT = Path(__file__).resolve().parents[1]


def requirements(path):
    return [Requirement(line) for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")]


def test_runtime_manifests_and_adapter_guard_agree():
    project = router_configuration._toml_loads((ROOT / "pyproject.toml").read_text())["project"]
    packaged = {str(Requirement(raw)) for raw in project["dependencies"]}
    assert packaged == {str(dep) for dep in requirements(ROOT / "requirements.txt")}
    litellm, = [Requirement(raw) for raw in project["dependencies"]
                if Requirement(raw).name == "litellm"]
    assert str(litellm.specifier) == "==" + mr.TESTED_LITELLM_VERSION
    assert litellm.extras == {"mcp"}
    assert project["version"] == mr.__version__


@pytest.mark.parametrize("python", ["3.10", "3.11", "3.12", "3.13", "3.14"])
def test_declared_dependencies_have_exact_constraints_for_each_supported_python(python):
    config = router_configuration._toml_loads((ROOT / "pyproject.toml").read_text())
    project = config["project"]
    declared = (project["dependencies"] + project["optional-dependencies"]["test"]
                + project["optional-dependencies"]["dev"] + config["build-system"]["requires"])
    environment = {**default_environment(), "python_version": python,
                   "python_full_version": python + ".0"}
    pins = {dep.name: dep for dep in requirements(ROOT / "constraints.txt")
            if not dep.marker or dep.marker.evaluate(environment)}
    for raw in declared:
        dep = Requirement(raw)
        if dep.marker and not dep.marker.evaluate(environment):
            continue
        pin, = pins[dep.name].specifier
        assert pin.operator == "=="
        assert pin.version in dep.specifier


def test_runtime_transitive_dependencies_are_constrained():
    """No fresh resolver choices may sneak into the tested runtime closure."""
    pins = {dep.name.lower().replace("_", "-"): dep
            for dep in requirements(ROOT / "constraints.txt")
            if not dep.marker or dep.marker.evaluate()}
    visited = set()

    def check(raw):
        dep = Requirement(raw)
        key = (dep.name, tuple(sorted(dep.extras)))
        if key in visited:
            return
        visited.add(key)
        pinned = pins[dep.name.lower().replace("_", "-")]
        installed = metadata.distribution(dep.name)
        assert installed.version in pinned.specifier
        for child in installed.requires or ():
            requirement = Requirement(child)
            if not requirement.marker or any(requirement.marker.evaluate({"extra": extra})
                                              for extra in ("", *dep.extras)):
                check(child)

    for dep in requirements(ROOT / "requirements.txt"):
        if not dep.marker or dep.marker.evaluate():
            check(str(dep))


@pytest.mark.parametrize("version", ["1.106.3", "1.106.5", "1.999.0", "2.0.0", None])
def test_incompatible_litellm_fails_before_import(monkeypatch, version):
    def installed(name):
        if version is None:
            raise metadata.PackageNotFoundError(name)
        return version

    original = builtins.__import__

    def importing(name, *args, **kwargs):
        assert name != "litellm", "Unsupported adapter must fail before importing LiteLLM"
        return original(name, *args, **kwargs)

    monkeypatch.setattr(metadata, "version", installed)
    monkeypatch.setattr(builtins, "__import__", importing)
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(router_adapters.__file__, run_name="model_router.release_startup_probe")
    message = str(caught.value)
    assert "Incompatible LiteLLM version" in message
    assert (version or "not installed") in message
    assert "litellm==" + mr.TESTED_LITELLM_VERSION in message
    assert "litellm[mcp]==" + mr.TESTED_LITELLM_VERSION in message
    assert "-c constraints.txt" in message
    assert "docs/contributing.md" in message




def test_default_config_cli_never_reads_credentials_or_starts_app(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "router.toml").write_text((ROOT / "router.toml").read_text())
    monkeypatch.setenv("OPENCODE_API_KEY", "dummy-release-test-key")

    def forbidden(*args, **kwargs):
        pytest.fail("Config validation must not read credential files or create the App")

    monkeypatch.setattr(router_cli, "load_default_credentials", forbidden)
    monkeypatch.setattr(router_cli, "load_env_file", forbidden)
    monkeypatch.setattr(router_cli, "App", forbidden)
    assert router_cli._find_default_config() == Path("router.toml")
    assert mr.main(["--no-default-env", "--validate-config", "--offline"]) == 0
    assert '"network_checked": false' in capsys.readouterr().out
