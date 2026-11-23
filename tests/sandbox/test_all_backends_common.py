"""Shared harness for the tier-3 sandbox-wide ``agSandbox`` contract tests.

The contract tests live in test_all_backends.py and
 test_all_backends_checkpoints.py (split for CI wall time); both import
this module's ``_Box`` factory, fixtures, and helpers so the two files
exercise identical setup. Each test is parametrized over ``sandbox_kind``
(chroot, docker, podman) and only uses the public ``agSandbox`` facade.
"""

from __future__ import annotations

import contextlib
import hashlib
import time
import uuid
from pathlib import Path

import pytest

from tests.sandbox.conftest import (
    backend_class,
    make_sandbox,
    requires,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A busy loop that stamps the time into a file: lets a test observe, from
# inside the sandbox only, whether a background process is still running.
_HEARTBEAT_LOOP = "while true; do date +%s%N > /workspace/{name}; sleep 0.2; done"


class _Box:
    """Per-test factory: builds sandboxes forced onto one backend and removes
    every sandbox and checkpoint tag it handed out, even when the test fails."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._sandboxes: list = []
        self._tags: list[str] = []

    def requires(self, *capabilities: str) -> None:
        requires(self.kind, *capabilities)

    def config(self, *mounts, gpu: bool = False, **resources):
        """An agconfig with ``(name, host_path, container_path, mode)`` mounts,
        optional GPU passthrough and ``resources`` overrides (idle_cpus...)."""
        from agency.configs.agconfig import agconfig as agconfig_cls

        cfg = agconfig_cls()
        cfg.sandbox.gpu_passthrough = gpu
        for name, host, container, mode in mounts:
            cfg.sandbox.add_mount(name, host, container, mode)
        for key, value in resources.items():
            setattr(cfg.resources, key, value)
        return cfg

    def sandbox(self, *, agconfig=None, **kwargs):
        sb = make_sandbox(self.kind, agconfig=agconfig, **kwargs)
        self._sandboxes.append(sb)
        return sb

    def adopt(self, sb):
        """Take ownership of a sandbox built elsewhere (e.g. by ``fork()``)."""
        self._sandboxes.append(sb)
        return sb

    def tag(self, prefix: str = "ckpt") -> str:
        tag = f"agency/test-{prefix}-{uuid.uuid4().hex[:12]}"
        self._tags.append(tag)
        return tag

    def close(self) -> None:
        for sb in reversed(self._sandboxes):
            with contextlib.suppress(Exception):
                sb.destroy()
        for tag in self._tags:
            with contextlib.suppress(Exception):
                backend_class(self.kind).delete_image(tag, force=True)


@pytest.fixture
def box(sandbox_kind):
    factory = _Box(sandbox_kind)
    yield factory
    factory.close()


@pytest.fixture
def sb(box):
    """A default sandbox on the backend under test (CPU-only, no mounts)."""
    return box.sandbox()


def ok(sandbox, cmd: str, **kwargs) -> str:
    """Run *cmd*, require exit 0, return stripped output. (chroot keeps the
    trailing newline of the output, containers strip it, so callers compare
    stripped text.)"""
    out, rc = sandbox.exec(cmd, **kwargs)
    assert rc == 0, f"{cmd!r} exited {rc}: {out}"
    return out.strip()


def wait_until(predicate, *, timeout: float = 30.0, interval: float = 0.25):
    """Poll *predicate* until truthy or *timeout*; return its last value."""
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value or time.monotonic() >= deadline:
            return value
        time.sleep(interval)


def read_heartbeat(sandbox, name: str = "hb") -> str:
    out, rc = sandbox.exec(f"cat /workspace/{name}")
    return out.strip() if rc == 0 else ""


def heartbeat_advances(sandbox, name: str = "hb") -> bool:
    """True if the heartbeat file changes over ~1s (its writer is alive)."""
    before = wait_until(lambda: read_heartbeat(sandbox, name), timeout=15)
    assert before, f"heartbeat file {name} was never written"
    time.sleep(1.0)
    return read_heartbeat(sandbox, name) != before


def backend_view(sandbox):
    """A second backend object with the SAME identity as *sandbox*'s own, the
    way a worker-process copy of it looks. Built with ``for_config()`` rather
    than another agSandbox(...): the facade de-duplicates agnames on every
    construction, so it would produce a different identity instead."""
    from agency.sandbox import agsandbox_backend

    return agsandbox_backend.for_config(
        sandbox.agconfig,
        agname=sandbox._backend._agname,
        name=sandbox._backend._name,
        checkpoint_image=None,
        base_image=sandbox.agconfig.sandbox.base_image,
        mounts={},
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
