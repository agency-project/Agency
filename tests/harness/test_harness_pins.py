import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agency.configs.agconfig import agconfig, agentconfig, harnessadapterconfig
from agency.harness import executable, install, pins

ROOT = Path(__file__).resolve().parents[2]
HARNESSES = ["claude_code", "codex", "grok", "kimi", "opencode"]


def test_every_external_harness_is_pinned_and_installable():
    assert sorted(pins.harness_pins()) == HARNESSES
    assert sorted(pins.BINARY_RELPATH) == HARNESSES
    assert sorted(install._INSTALLERS) == HARNESSES


def test_ci_installs_the_pinned_versions_from_pyproject():
    workflow = (ROOT / ".github/workflows/tests.yml").read_text()
    assert "agency install-harnesses" in workflow
    assert not re.search(r"claude.ai/install|codex/install|opencode.ai/install|x.ai/cli", workflow)


@pytest.mark.parametrize(
    ("output", "version", "expected"),
    [
        ("codex-cli 0.147.0", "0.147.0", True),
        ("2.1.311 (Claude Code)", "2.1.311", True),
        ("grok 1.0.77 (2fe064390d07)", "1.0.77", True),
        ("1.19.5\n", "1.19.5", True),
        ("grok 1.0.777", "1.0.77", False),
        ("grok 11.0.77", "1.0.77", False),
        ("grok 1.0.77.1", "1.0.77", False),
        ("codex-cli 0.154.0", "0.147.0", False),
        ("", "0.147.0", False),
    ],
)
def test_reports_version_names_exactly_the_pinned_version(output, version, expected):
    assert pins.reports_version(output, version) is expected


def test_managed_binary_needs_an_executable_file_at_the_pinned_version(tmp_path):
    assert pins.managed_binary("codex", tmp_path) is None
    binary = pins.install_dir("codex", tmp_path) / pins.BINARY_RELPATH["codex"]
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n")
    assert pins.managed_binary("codex", tmp_path) is None
    binary.chmod(0o755)
    assert pins.managed_binary("codex", tmp_path) == binary
    assert pins.install_dir("codex", tmp_path).name == f"codex-{pins.harness_pins()['codex']}"
    assert pins.managed_binary("native", tmp_path) is None


def test_home_defaults_to_dot_agency_and_can_be_moved(monkeypatch, tmp_path):
    monkeypatch.delenv(pins.HOME_ENV, raising=False)
    assert pins.harness_home() == Path.home() / ".agency" / "harnesses"
    monkeypatch.setenv(pins.HOME_ENV, str(tmp_path))
    assert pins.harness_home() == tmp_path


def _fake_installer(version_output):
    def installer(version, work, destination):
        binary = destination / pins.BINARY_RELPATH["grok"]
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_text(
            f"#!{sys.executable}\nprint({version_output.format(version=version)!r})\n"
        )

    return installer


def test_install_puts_the_pinned_version_where_agency_looks(monkeypatch, tmp_path):
    monkeypatch.setitem(install._INSTALLERS, "grok", _fake_installer("grok {version}"))
    monkeypatch.setattr(install.platform, "system", lambda: "Linux")
    monkeypatch.setattr(install.platform, "machine", lambda: "x86_64")
    binary = install.install_harness("grok", tmp_path)
    assert binary == pins.managed_binary("grok", tmp_path)
    assert pins.reports_version(
        install.subprocess.run([binary, "--version"], capture_output=True, text=True).stdout,
        pins.harness_pins()["grok"],
    )
    # Installed already: not fetched again, unless forced.
    calls = []

    def counting(version, work, destination):
        calls.append(version)
        _fake_installer("grok {version}")(version, work, destination)

    monkeypatch.setitem(install._INSTALLERS, "grok", counting)
    assert install.install_harness("grok", tmp_path) == binary
    assert calls == []
    assert install.install_harness("grok", tmp_path, force=True) == binary
    assert calls == [pins.harness_pins()["grok"]]


def test_install_refuses_a_binary_that_reports_another_version(monkeypatch, tmp_path):
    monkeypatch.setitem(install._INSTALLERS, "grok", _fake_installer("grok 0.0.1"))
    monkeypatch.setattr(install.platform, "system", lambda: "Linux")
    monkeypatch.setattr(install.platform, "machine", lambda: "x86_64")
    with pytest.raises(RuntimeError, match="another version"):
        install.install_harness("grok", tmp_path)
    assert pins.managed_binary("grok", tmp_path) is None
    assert [p.name for p in tmp_path.iterdir()] == []


def test_install_rejects_an_unknown_harness(tmp_path):
    with pytest.raises(ValueError, match="no pinned harness named nope"):
        install.install_harnesses(["nope"], tmp_path)


@pytest.mark.parametrize("harness", HARNESSES)
def test_a_binary_reporting_another_version_is_refused_for_every_harness(
    harness, monkeypatch, tmp_path
):
    binary = tmp_path / "cli"
    binary.write_text("x")
    binary.chmod(0o755)
    config = agconfig(agentconfig(harness=harness), harnessadapterconfig(binary_path=str(binary)))
    monkeypatch.setattr(
        executable.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout="9.9.9", stderr=""),
    )
    with pytest.raises(RuntimeError, match=f"expected {pins.harness_pins()[harness]}"):
        executable.prepare_harness_executable_local(harness, config)
    monkeypatch.setattr(
        executable.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            returncode=0, stdout=f"v {pins.harness_pins()[harness]}", stderr=""
        ),
    )
    assert executable.prepare_harness_executable_local(harness, config) == str(binary)
