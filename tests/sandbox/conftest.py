"""Shared fixtures for the three tiers of sandbox tests.

Tier 1 -- backend-specific:   test_docker_backend.py, test_podman_backend.py,
                              test_chroot_backend.py (mechanisms and arguments
                              that exist in exactly one backend).
Tier 2 -- family-specific:    test_container_backends.py (docker + podman,
                              everything in ``_ContainerBackendBase``) and the
                              non-container family (chroot, the only member,
                              so its behaviors live in test_chroot_backend.py).
Tier 3 -- sandbox-wide:       test_all_backends.py (the ``agSandbox`` contract:
                              exec, file I/O, mounts, isolation, checkpoints,
                              process tracking, resources...). Runs on every
                              backend available on the host.

Fixtures
--------
``sandbox_kind``      chroot | docker | podman. Real-backend runs skip
                      themselves when that backend is unusable on this host.
``container_kind``    docker | podman, with the same skipping.
``container_runtime`` docker | podman and NEVER skips: for mock-based tier-2
                      tests that never touch a daemon.
``backend_kind``      chroot | docker | podman and NEVER skips: for mock-based
                      tests of base-class logic that must hold for every
                      backend (test_base*.py).

Helpers (import from ``tests.sandbox.conftest``)
------------------------------------------------
``backend_available(kind)``  cached live availability probe.
``make_sandbox(kind, ...)``  real ``agSandbox`` forced onto ``kind``.
``make_backend(kind, ...)``  bare backend object, no daemon probe, for mocks.
``backend_class(kind)``      the backend class itself (static image helpers).
``checkpoint_exists(kind, tag)``  does a committed checkpoint *tag* still exist?
``started_run_cmd(backend)``  drive ``_ensure_started`` with every external
                             call mocked; returns the ``run`` argv.
``make_cow_backend(runtime)``  container backend wired for cow_zfs + CRIU
                             fast-resume, runtime/storage calls mocked.
``requires(kind, *caps)``    skip with a visible reason when *kind* lacks a
                             capability (see ``BACKEND_CAPABILITIES``).
"""

from __future__ import annotations

import functools
import shutil
import subprocess
import uuid
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

ALL_BACKENDS = ("chroot", "docker", "podman")
CONTAINER_BACKENDS = ("docker", "podman")

# What each backend can honour. A tier-3 test whose contract only some
# backends can satisfy calls ``requires(kind, "<capability>")`` so the gap
# shows up as a skip with a reason in the test output, rather than the test
# being dropped or written for one backend only.
CAPABILITY_DESCRIPTIONS = {
    "network_namespace": "own network namespace (host loopback unreachable from the sandbox)",
    "pid_namespace": "own PID namespace (host processes invisible to the sandbox)",
    "cgroup_limits": "cgroup CPU/memory limits that update_limits()/resources config enforce",
    "checkpoint_handle": "checkpoint()/delete_checkpoint() handle API (not just commit()/restore())",
    "ptrace_ingest": "ingest_ptrace_pids() feeds get_live_pids()",
    "pid_elapsed_summary": "pid_status_summary() reports per-PID elapsed time",
    "detached_pid_tracking": "exec_detached() daemons show up in get_live_pids()",
    "env_isolation": "host environment variables are not inherited by sandboxed commands",
}
BACKEND_CAPABILITIES: "dict[str, frozenset[str]]" = {
    # chroot: shares the host network and PID namespaces, has no cgroup of
    # its own, tracks background work by process group only (so, unlike the
    # containers, it does see exec_detached daemons), inherits the host
    # environment, and only has the commit()/restore() snapshot API.
    "chroot": frozenset({"detached_pid_tracking"}),
    "docker": frozenset(CAPABILITY_DESCRIPTIONS) - {"detached_pid_tracking"},
    "podman": frozenset(CAPABILITY_DESCRIPTIONS) - {"detached_pid_tracking"},
}


def has_capability(kind: str, capability: str) -> bool:
    if capability not in CAPABILITY_DESCRIPTIONS:
        raise KeyError(f"unknown capability {capability!r}")
    return capability in BACKEND_CAPABILITIES[kind]


def requires(kind: str, *capabilities: str) -> None:
    """Skip the calling test (with a reason naming the backend and the
    missing capability) unless *kind* has every one of *capabilities*."""
    for capability in capabilities:
        if not has_capability(kind, capability):
            pytest.skip(
                f"{kind} backend lacks capability {capability!r}: "
                f"{CAPABILITY_DESCRIPTIONS[capability]}"
            )


@functools.lru_cache(maxsize=None)
def backend_available(kind: str) -> bool:
    """True if *kind* can actually run a sandbox on this host."""
    if kind == "chroot":
        from agency.sandbox.chroot import chroot_available

        return chroot_available()
    if kind in CONTAINER_BACKENDS:
        if shutil.which(kind) is None:
            return False
        try:
            return subprocess.run([kind, "info"], capture_output=True, timeout=10).returncode == 0
        except Exception:
            return False
    raise ValueError(f"unknown sandbox backend {kind!r}")


def _skip_unavailable(kinds):
    return [
        pytest.param(
            kind,
            id=kind,
            marks=pytest.mark.skipif(
                not backend_available(kind), reason=f"{kind} backend not usable on this host"
            ),
        )
        for kind in kinds
    ]


def make_sandbox(kind: str, *, agconfig=None, probe: bool = True, **kwargs):
    """Build a real ``agSandbox`` forced onto backend *kind*.

    ``probe=False`` skips the live-daemon check done at backend selection, for
    configuration-only tests that never run a command. ``gpu_passthrough`` is
    off unless the caller's config says otherwise: filesystem, lifecycle and
    process tests do not need GPU device setup.
    """
    from agency.configs.agconfig import agconfig as agconfig_cls
    from agency.sandbox.agsandbox import agSandbox

    cfg = agconfig.clone() if agconfig is not None else agconfig_cls()
    if agconfig is None:
        cfg.sandbox.gpu_passthrough = False
    cfg.sandbox.backend = kind
    sandbox_id = str(uuid.uuid4())
    if probe or kind == "chroot":
        return agSandbox(sandbox_id, agconfig=cfg, **kwargs)
    with patch("agency.sandbox.base.shutil.which", return_value=f"/usr/bin/{kind}"):
        with patch("agency.sandbox.container._runtime_works", return_value=True):
            return agSandbox(sandbox_id, agconfig=cfg, **kwargs)


def make_backend(kind: str, **kwargs):
    """Build a bare backend object directly, bypassing ``for_config()``'s live
    daemon check. For unit tests that mock ``_run``/``_container_running``/
    etc. and never issue a real ``docker``/``podman`` call."""
    name = f"{kind}-test-{uuid.uuid4().hex[:8]}"
    defaults = dict(agname=name, name=name, checkpoint_image=None, mounts={}, agconfig=None)
    if kind == "chroot":
        from agency.sandbox.chroot import _ChrootBackend as cls
    else:
        defaults["base_image"] = "docker.io/library/python:3.12-slim"
        if kind == "docker":
            from agency.sandbox.docker import _DockerBackend as cls
        elif kind == "podman":
            from agency.sandbox.podman import _PodmanBackend as cls
        else:
            raise ValueError(f"unknown sandbox backend {kind!r}")
    defaults.update(kwargs)
    return cls(defaults.pop("agname"), **defaults)


def backend_class(kind: str):
    """The backend class for *kind* (for its static image helpers)."""
    if kind == "chroot":
        from agency.sandbox.chroot import _ChrootBackend

        return _ChrootBackend
    if kind == "docker":
        from agency.sandbox.docker import _DockerBackend

        return _DockerBackend
    if kind == "podman":
        from agency.sandbox.podman import _PodmanBackend

        return _PodmanBackend
    raise ValueError(f"unknown sandbox backend {kind!r}")


def checkpoint_exists(kind: str, tag: str) -> bool:
    """True if a checkpoint committed under *tag* still exists on this host
    (an image for docker/podman, a snapshot directory for chroot)."""
    if kind == "chroot":
        from agency.sandbox.chroot import _CHROOT_SNAPSHOTS_DIR, _sanitize_tag

        return (_CHROOT_SNAPSHOTS_DIR / _sanitize_tag(tag)).is_dir()
    proc = subprocess.run([kind, "image", "inspect", tag], capture_output=True, timeout=60)
    return proc.returncode == 0


def started_run_cmd(sb, *, status=(False, ""), extra=(), cfs=False, cpu_count=None):
    """Drive a container backend's ``_ensure_started`` with every external call
    mocked and return the ``run`` argv it asked ``_run_with_conflict_retry`` to
    execute."""
    import agency.sandbox.container as _container

    with ExitStack() as stack:
        stack.enter_context(patch.object(sb, "_inspect_container_state", return_value=status))
        stack.enter_context(patch.object(sb, "_acquire_runtime_slot"))
        stack.enter_context(patch.object(sb, "_release_runtime_slot"))
        stack.enter_context(patch.object(sb, "_cfs_supported", return_value=cfs))
        stack.enter_context(patch.object(sb, "_snapshot_pids_started", return_value=set()))
        stack.enter_context(patch.object(sb, "_run"))
        stack.enter_context(patch.object(_container, "_gpu_flags", return_value=[]))
        if cpu_count is not None:
            stack.enter_context(patch.object(_container.os, "cpu_count", return_value=cpu_count))
        for cm in extra:
            stack.enter_context(cm)
        retry = stack.enter_context(patch.object(sb, "_run_with_conflict_retry"))
        sb._ensure_started()
    assert retry.call_count == 1, "expected exactly one run invocation"
    return retry.call_args.args[0]


def make_cow_backend(runtime: str = "podman"):
    """A bare container backend configured for ``cow_zfs`` checkpoints with CRIU
    fast resume, whose runtime and storage calls are mocks. Returns
    ``(backend, state)``; ``state["running"]`` tracks the fake container."""
    from agency.configs.agconfig import agconfig as agconfig_cls
    from agency.sandbox import checkpoint as cp
    from agency.sandbox.container import _ContainerBackendBase

    cfg = agconfig_cls()
    cfg.sandbox.backend = runtime
    cfg.sandbox.checkpoint_backend = "cow_zfs"
    cfg.sandbox.checkpoint_zfs_parent = "tank/agency"
    cfg.sandbox.checkpoint_fast_resume = True
    state = {"running": True}
    backend = _ContainerBackendBase.__new__(_ContainerBackendBase)
    backend._agconfig = cfg
    backend._runtime = runtime
    backend._name = "sandbox"
    backend._virtual_gpu_count_requested = 0
    backend._checkpointer = cp.CowZfsCheckpoint()
    backend._normalized_checkpoint_manifest = Mock(return_value={"schema_version": 1})
    backend._container_running = lambda: state["running"]
    backend._stop_image = Mock(side_effect=lambda: state.update(running=False))
    backend._ensure_started_profiled = Mock(side_effect=lambda: state.update(running=True))
    backend._register_prof_container = Mock()
    backend._checkpoint_storage = SimpleNamespace(
        prepared=True,
        dataset="tank/agency/private",
        prepare=Mock(),
        capture_dataset=Mock(return_value="tank/agency/private"),
        snapshot=Mock(),
        snapshot_stats=Mock(return_value={"zfs_snapshot_used_bytes": 0}),
        rollback=Mock(),
        delete=Mock(),
        criu_ready=True,
        criu_setup_error=None,
        fast_root=Path("/zfs/agency-fast-resume"),
        init_path="/zfs/agency-fast-resume/init",
        criu_config="/zfs/agency-fast-resume/criu.conf",
        prepare_restore_mounts=Mock(),
        runtime_files_restored=Mock(),
    )
    return backend, state


@pytest.fixture(params=_skip_unavailable(ALL_BACKENDS))
def sandbox_kind(request) -> str:
    """Every sandbox backend; one that is unusable on this host skips."""
    return request.param


@pytest.fixture(params=_skip_unavailable(CONTAINER_BACKENDS))
def container_kind(request) -> str:
    """docker and podman; one that is unusable on this host skips."""
    return request.param


@pytest.fixture(params=ALL_BACKENDS)
def backend_kind(request) -> str:
    """chroot, docker and podman, never skipped -- for mock-based tests of logic
    every backend inherits from the base class."""
    return request.param


@pytest.fixture(params=CONTAINER_BACKENDS)
def container_runtime(request) -> str:
    """docker and podman, never skipped -- for mock-based tests that do not
    need a daemon."""
    return request.param


@pytest.fixture(autouse=True)
def _clean_chroot_state_root():
    """Chroot jails and snapshots live under one per-run directory; remove it
    around every sandbox test so runs never see each other's state."""
    from agency.sandbox.chroot import _CHROOT_STATE_ROOT

    shutil.rmtree(_CHROOT_STATE_ROOT, ignore_errors=True)
    yield
    shutil.rmtree(_CHROOT_STATE_ROOT, ignore_errors=True)
