"""The harness CLI versions Agency is verified with, and where it keeps them.

``[tool.agency.harnesses]`` in ``pyproject.toml`` names one version per
external harness. ``agency install-harnesses`` (``harness/install.py``) puts
exactly those versions under ``harness_home()``, and Agency runs the binaries
from there unless ``harness_adapter.binary_path`` names another one. Whichever
binary is used must report the pinned version (``executable.py``).
"""

from __future__ import annotations

import os
import re
import tomllib
from functools import cache
from pathlib import Path

# Where each harness's executable lies inside its install directory.
BINARY_RELPATH = {
    "claude_code": "package/claude",
    "codex": "package/vendor/x86_64-unknown-linux-musl/bin/codex",
    "grok": "grok",
    "kimi": "kimi",
    "opencode": "package/bin/opencode",
}

HOME_ENV = "AGENCY_HARNESS_HOME"


class HarnessNotInstalled(FileNotFoundError):
    """The pinned version of a harness is not in the agency-managed location."""


@cache
def harness_pins() -> "dict[str, str]":
    """Harness name -> pinned version, from ``[tool.agency.harnesses]``."""
    # A wheel carries a copy of pyproject.toml; a source checkout has it at the root.
    here = Path(__file__).resolve()
    for path in (here.with_name("_pyproject.toml"), here.parents[2] / "pyproject.toml"):
        try:
            data = tomllib.loads(path.read_text())
        except (OSError, tomllib.TOMLDecodeError):
            continue
        pins = data.get("tool", {}).get("agency", {}).get("harnesses")
        if data.get("project", {}).get("name") == "agency" and pins:
            return {str(name): str(version) for name, version in pins.items()}
    raise RuntimeError("[tool.agency.harnesses] is missing from Agency's pyproject.toml")


def harness_home() -> Path:
    """The directory holding the managed installs (``$AGENCY_HARNESS_HOME``,
    else ``~/.agency/harnesses``)."""
    return Path(os.environ.get(HOME_ENV) or Path.home() / ".agency" / "harnesses")


def install_dir(harness: str, home: "Path | None" = None) -> Path:
    return (home or harness_home()) / f"{harness}-{harness_pins()[harness]}"


def managed_binary(harness: str, home: "Path | None" = None) -> "Path | None":
    """The pinned harness's executable, or None if it is not installed."""
    if harness not in BINARY_RELPATH or harness not in harness_pins():
        return None
    binary = install_dir(harness, home) / BINARY_RELPATH[harness]
    return binary if binary.is_file() and os.access(binary, os.X_OK) else None


def reports_version(output: str, version: str) -> bool:
    """Whether the ``--version`` output of a harness names exactly *version*
    (``1.0.6`` is not ``1.0.67``)."""
    return re.search(rf"(?<![\d.]){re.escape(version)}(?![\d]|\.\d)", output) is not None
