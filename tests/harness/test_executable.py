from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agency.configs.agconfig import agconfig, agentconfig, harnessadapterconfig
from agency.harness import executable, pins
from agency.harness.adapters.base import HarnessAdapter
from agency.harness.adapters.pty.driver import driver_for


def config_for(harness="claude_code", binary="/custom/cli"):
    return agconfig(agentconfig(harness=harness), harnessadapterconfig(binary_path=binary))


def install(path, contents=b"executable"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    path.chmod(0o755)
    return path


@pytest.mark.parametrize("harness", ["claude_code", "codex", "grok", "kimi", "opencode"])
def test_default_is_the_pinned_install_in_the_agency_directory(harness, tmp_path, monkeypatch):
    monkeypatch.setenv(pins.HOME_ENV, str(tmp_path))
    config = config_for(harness, None)
    with pytest.raises(pins.HarnessNotInstalled, match="agency install-harnesses"):
        executable.external_binary(harness, config)
    binary = install(pins.install_dir(harness) / pins.BINARY_RELPATH[harness])
    assert executable.external_binary(harness, config) == str(binary)
    assert executable.external_binary(harness, config_for(harness, "/custom/cli")) == "/custom/cli"


@pytest.mark.parametrize("harness", ["claude_code", "codex", "grok", "kimi", "opencode"])
def test_adapters_and_pty_drivers_run_the_managed_install_never_path(
    harness, tmp_path, monkeypatch
):
    monkeypatch.setenv(pins.HOME_ENV, str(tmp_path / "home"))
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    adapter = HarnessAdapter.for_config(harness, agconfig())
    install(tmp_path / "bin" / adapter._DEFAULT_BINARY)  # on PATH, but must be ignored
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    runtime = SimpleNamespace(
        harness_base_url="http://127.0.0.1:1",
        token="t",
        agconfig=adapter.agconfig,
        model="m",
        has_sandbox_mcp_tools=False,
    )
    with pytest.raises(pins.HarnessNotInstalled):
        adapter.binary()
    with pytest.raises(pins.HarnessNotInstalled):
        driver_for(adapter, runtime, first, None, None, 4)
    binary = install(pins.install_dir(harness) / pins.BINARY_RELPATH[harness])
    assert adapter.binary() == str(binary)
    driver = driver_for(adapter, runtime, second, None, None, 4)
    assert driver.argv[0] == str(binary)


def test_native_does_not_discover_or_mount_external_files():
    config = config_for("native")
    assert executable.harness_installation_mounts(config) == {}
    assert executable.resolve_harness_binary("native", config) is None


def test_mounts_complete_npm_tree_including_hoisted_native_package(monkeypatch, tmp_path):
    modules = tmp_path / "lib" / "node_modules"
    cli = install(modules / "@vendor" / "cli" / "bin" / "cli.js", b"#!/usr/bin/env node\n")
    (cli.parent.parent / "package.json").write_text('{"type":"module"}')
    native = install(modules / "@vendor" / "cli-linux" / "bin" / "cli")
    entry = tmp_path / "bin" / "cli"
    entry.parent.mkdir()
    entry.symlink_to(Path("../lib/node_modules/@vendor/cli/bin/cli.js"))
    mounts = executable.harness_installation_mounts(config_for("codex", str(entry)))
    assert list(mounts.values()) == [(str(modules), str(modules), "ro")]
    assert entry.is_symlink()
    assert native.is_relative_to(modules)


def test_mounts_standalone_install_directory_without_copying(monkeypatch, tmp_path):
    root = tmp_path / "versions"
    binary = install(root / "1.0")
    entry = tmp_path / "cli"
    entry.symlink_to(binary)
    mounts = executable.harness_installation_mounts(config_for(binary=str(entry)))
    assert list(mounts.values()) == [(str(root), str(root), "ro")]
    install(root / "2.0")
    assert (Path(next(iter(mounts.values()))[0]) / "2.0").exists()
    assert entry.readlink() == binary


def test_app_bin_mount_includes_sibling_data_and_internal_symlinks(monkeypatch, tmp_path):
    root = tmp_path / "app"
    binary = install(root / "bin" / "cli")
    data = install(root / "lib" / "data")
    link = root / "bin" / "data"
    link.symlink_to("../lib/data")
    assert list(
        executable.harness_installation_mounts(config_for(binary=str(binary))).values()
    ) == [(str(root), str(root), "ro")]
    assert link.readlink() == Path("../lib/data")
    assert link.resolve() == data


def test_external_package_symlinks_keep_targets_mounted(monkeypatch, tmp_path):
    modules = tmp_path / "node_modules"
    cli = install(modules / "cli" / "bin" / "cli.js")
    linked = tmp_path / "linked-package"
    install(linked / "data")
    (modules / "linked").symlink_to(linked, target_is_directory=True)
    # A reverse link verifies discovery terminates on cycles.
    (linked / "modules").symlink_to(modules, target_is_directory=True)
    mounts = executable.harness_installation_mounts(config_for("codex", str(cli)))
    assert set(mounts.values()) == {
        (str(modules), str(modules), "ro"),
        (str(linked), str(linked), "ro"),
    }


def test_missing_host_installation_is_an_error_not_an_image_fallback(tmp_path):
    # Nothing is assumed to exist in the sandbox image, and PATH is never
    # consulted: a binary the host doesn't have fails before the container starts.
    config = config_for(binary=str(tmp_path / "not-there" / "cli"))
    with pytest.raises(pins.HarnessNotInstalled, match="does not exist on this host"):
        executable.harness_installation_mounts(config)
    with pytest.raises(pins.HarnessNotInstalled, match="does not exist on this host"):
        executable.resolve_harness_binary("claude_code", config)


@pytest.mark.parametrize("harness", ["claude_code", "codex", "grok", "kimi", "opencode"])
def test_binary_path_must_be_absolute_and_path_is_never_searched(harness, tmp_path, monkeypatch):
    install(tmp_path / "cli")
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ValueError, match="absolute path"):
        executable.external_binary(harness, config_for(harness, "cli"))


def test_prepared_host_symlink_resolves_to_mounted_package(monkeypatch, tmp_path):
    binary = install(tmp_path / "app" / "bin" / "cli")
    entry = tmp_path / "cli"
    entry.symlink_to(binary)
    assert executable.resolve_harness_binary("codex", config_for("codex", str(entry))) == str(
        binary
    )


def test_prepare_local_validates_the_resolved_binary_runs(monkeypatch, tmp_path):
    binary = install(tmp_path / "app" / "bin" / "cli")
    config = config_for("codex", str(binary))
    monkeypatch.setattr(
        executable.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout="codex-cli 0.147.0", stderr=""),
    )
    assert executable.prepare_harness_executable_local("codex", config) == str(binary)


def test_prepare_local_rejects_codex_versions_other_than_the_pinned_one(monkeypatch, tmp_path):
    """0.154.0 hangs waiting for native prompt acknowledgment on large
    bracketed pastes (see executable.py); fail closed instead of a
    confusing mid-run submit timeout."""
    binary = install(tmp_path / "app" / "bin" / "cli")
    config = config_for("codex", str(binary))
    monkeypatch.setattr(
        executable.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout="codex-cli 0.154.0", stderr=""),
    )
    with pytest.raises(RuntimeError, match="expected"):
        executable.prepare_harness_executable_local("codex", config)


def test_missing_mount_fails_before_launch(tmp_path):
    missing = tmp_path / "not-there" / "cli"
    config = config_for("grok", str(missing))
    with pytest.raises(FileNotFoundError, match="before container creation"):
        executable.prepare_harness_executable_local("grok", config)


@pytest.mark.parametrize(
    "error", ["node: No such file", "Missing optional dependency", "libc.so: not found"]
)
def test_prepare_local_with_missing_runtime_dependencies_fails(monkeypatch, tmp_path, error):
    binary = install(tmp_path / "app" / "cli")
    config = config_for("codex", str(binary))
    monkeypatch.setattr(
        executable.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=127, stdout="", stderr=error),
    )
    with pytest.raises(RuntimeError, match=error):
        executable.prepare_harness_executable_local("codex", config)


def test_does_not_mount_entire_system_prefix(monkeypatch):
    monkeypatch.setattr(Path, "is_file", lambda self: str(self) == "/usr/bin/cli")
    monkeypatch.setattr(Path, "resolve", lambda self, **kwargs: self)
    with pytest.raises(ValueError, match="dedicated installation"):
        executable.harness_installation_mounts(config_for(binary="/usr/bin/cli"))


def test_external_symlink_cannot_expand_mount_to_host_home(monkeypatch, tmp_path):
    binary = install(tmp_path / "app" / "cli")
    (binary.parent / "home").symlink_to(Path.home(), target_is_directory=True)
    with pytest.raises(ValueError, match="broad host directory"):
        executable.harness_installation_mounts(config_for(binary=str(binary)))


def test_facade_registers_installation_before_backend_creation(monkeypatch, tmp_path):
    from agency.sandbox.agsandbox import agSandbox

    binary = install(tmp_path / "app" / "cli")
    factory = Mock()
    monkeypatch.setattr("agency.sandbox.agsandbox.agsandbox_backend.for_config", factory)
    sandbox = agSandbox("mount-order", agconfig=config_for("codex", str(binary)))
    try:
        mounts = factory.call_args.kwargs["mounts"]
        assert (str(binary.parent), str(binary.parent), "ro") in mounts.values()
        factory.return_value.exec.assert_not_called()
    finally:
        sandbox.destroy()


def test_agent_harness_override_reaches_sandbox_configuration(monkeypatch):
    import importlib

    module = importlib.import_module("agency.agent")
    factory = Mock()
    monkeypatch.setattr(module, "agSandbox", factory)
    config = config_for("native")
    owner = SimpleNamespace(
        sandbox=None, agconfig=config, harness="codex", output_path=None, agname="override"
    )
    assert module.agent._ensure_sandbox(owner) is factory.return_value
    assert factory.call_args.kwargs["agconfig"].agent.harness == "codex"
    assert config.agent.harness == "native"


def test_restored_harness_reaches_checkpoint_sandbox_configuration(monkeypatch, tmp_path):
    import importlib
    import io
    import json
    import tarfile

    module = importlib.import_module("agency.agent")
    factory = Mock()
    monkeypatch.setattr(module, "agSandbox", factory)
    config = config_for("native")
    config.llm.api_key = "test-key"
    config.llm.model = "test-model"
    checkpoint = tmp_path / "agent.ckpt"
    state = json.dumps({"agname": "restored-harness", "harness": "codex"}).encode()
    with tarfile.open(checkpoint, "w:gz") as archive:
        for name, data in [("state.json", state), ("container.tar", b"fake image")]:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))

    restored = module.agent.load(checkpoint, agconfig=config)
    try:
        assert restored.harness == "codex"
        assert factory.call_args.kwargs["agconfig"].agent.harness == "codex"
        assert factory.call_args.kwargs["checkpoint_image"]
        assert config.agent.harness == "native"
    finally:
        restored.sandbox.destroy()
