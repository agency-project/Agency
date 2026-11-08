"""Host-side discovery and read-only installation mounts for external CLIs,
plus the daemon-local startup check for the resolved binary.

Mounts are chosen before container creation. Keeping installation directories at
their original absolute paths preserves package-relative lookups and symlinks;
the daemon receives a path, never responsibility for copying host files.
System interpreters and shared libraries belong in the sandbox image.

`resolve_harness_binary()` finds the binary's real absolute path from the
host, ahead of the daemon's launch. `prepare_harness_executable_local()`
validates that resolved path from inside the daemon itself: file exists, is
executable, actually runs.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

from .adapters.base import HarnessAdapter
from .pins import HarnessNotInstalled, harness_home, harness_pins, managed_binary, reports_version


HARNESS_PATH = "/usr/local/bin:/usr/bin:/bin"

# Why another version is refused, where it is known.
_VERSION_NOTES = {
    # 0.154.0 hangs waiting for native prompt acknowledgment on large
    # (>1000 char) bracketed pastes -- the PTY driver's normal submission
    # path -- because codex routes those through a placeholder/expand-on-
    # submit pipeline that was heavily refactored between the two versions.
    "codex": (
        "Newer builds have hung waiting for native prompt acknowledgment on real "
        "(large) task prompts. "
    ),
}


def external_binary(harness, config) -> str | None:
    """The harness CLI to run, always as an absolute path: the explicit
    ``harness_adapter.binary_path`` if set, else the pinned version in Agency's
    own install directory (harness/pins.py). Nothing is looked up on PATH and
    nothing is assumed to exist in the sandbox image."""
    if not HarnessAdapter.for_config(harness, config)._DEFAULT_BINARY:
        return None
    configured = config.harness_adapter.binary_path
    if configured:
        if not Path(configured).is_absolute():
            raise ValueError(
                f"harness_adapter.binary_path must be an absolute path, got {configured!r}; "
                "Agency never looks harness CLIs up on PATH"
            )
        return configured
    managed = managed_binary(harness)
    if managed is None:
        raise HarnessNotInstalled(
            f"{harness} {harness_pins().get(harness)} is not installed in {harness_home()}; "
            "run `agency install-harnesses`, or set harness_adapter.binary_path"
        )
    return str(managed)


def _installation_root(executable: Path) -> Path:
    # npm can hoist platform packages beside the CLI package. Include the
    # outer node_modules tree, not just the entry script's immediate package.
    node_modules = [parent for parent in executable.parents if parent.name == "node_modules"]
    if node_modules:
        return node_modules[-1]
    for parent in executable.parents:
        if (parent / "package.json").is_file():
            return parent
    # A conventional app/bin layout may have sibling lib/data directories.
    root = executable.parent
    if root.name == "bin":
        root = root.parent
    return root


def host_binary(harness, config) -> str | None:
    """The harness binary's real absolute path on this host, or None for a
    harness with no external CLI. Raises if the file is not there: it is
    mounted into the sandbox from the host, never taken from the image."""
    binary = external_binary(harness, config)
    if binary is None:
        return None
    path = Path(binary)
    if not path.is_file():
        raise HarnessNotInstalled(
            f"{harness} executable {binary!r} does not exist on this host; "
            "run `agency install-harnesses`, or fix harness_adapter.binary_path"
        )
    return str(path.resolve(strict=True))


def harness_installation_mounts(config) -> dict[str, tuple[str, str, str]]:
    """Discover the selected harness without starting or modifying a container."""
    host_path = host_binary(config.agent.harness, config)
    if host_path is None:
        return {}
    root = _installation_root(Path(host_path))
    roots = [root]
    mounts = {}
    while roots:
        root = roots.pop()
        if root in {
            Path("/"),
            Path("/usr"),
            Path("/usr/local"),
            Path("/opt"),
            Path("/home"),
            Path.home(),
            Path.home() / ".local",
        }:
            raise ValueError(
                f"Cannot mount broad host directory {root} as a harness installation; "
                "use a dedicated installation directory"
            )
        if str(root) in {mount[0] for mount in mounts.values()}:
            continue
        name = "_harness_install_" + hashlib.sha256(str(root).encode()).hexdigest()[:16]
        mounts[name] = (str(root), str(root), "ro")
        # Linked workspace packages can live outside node_modules. Mount their
        # complete directories too, at the original paths the links reference.
        for directory, dirs, files in os.walk(root, followlinks=False):
            for entry in dirs + files:
                path = Path(directory) / entry
                if not path.is_symlink():
                    continue
                target = path.resolve()
                if target.is_relative_to(root) or not target.exists():
                    continue
                linked_root = target if target.is_dir() else _installation_root(target)
                if linked_root not in roots:
                    roots.append(linked_root)
    return mounts


def resolve_harness_binary(harness, config) -> str | None:
    """The harness binary's real absolute path on the host (None for native).
    Its install directory is mounted at the same path in the sandbox. Raises
    if it is missing; there is no PATH lookup and no image-provided fallback."""
    return host_binary(harness, config)


def prepare_harness_executable_local(harness: str, config) -> "str | None":
    """Validate the harness binary from inside the sandbox: file exists, is
    executable, actually runs. The path is the absolute one the host resolved
    and mounted; it is never looked up on PATH."""
    binary = external_binary(harness, config)
    if binary is None:
        return None

    resolved_path = Path(binary)
    if not (resolved_path.is_file() and os.access(resolved_path, os.X_OK)):
        raise FileNotFoundError(
            f"{harness} executable {binary!r} is missing or not executable in the "
            "sandbox. Its installation directory is mounted from the host before "
            "container creation; check harness_adapter.binary_path."
        )
    resolved = str(resolved_path.resolve(strict=True))

    proc = subprocess.run([resolved, "--version"], capture_output=True, text=True, timeout=30)
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(
            f"{harness} executable {resolved!r} cannot start (exit {proc.returncode}). "
            "Its complete installation must provide its system interpreter and shared "
            f"libraries. Output: {out[-2000:]}"
        )
    pin = harness_pins().get(harness)
    if pin is not None and not reports_version(out, pin):
        note = _VERSION_NOTES.get(harness, "")
        raise RuntimeError(
            f"{harness} executable {resolved!r} reports version {out.strip()!r}, expected "
            f"{pin}, the only version Agency is verified with. {note}Install it with "
            "`agency install-harnesses`."
        )
    return resolved
