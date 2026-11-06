"""Tier 2 -- the container family: everything ``_ContainerBackendBase`` implements.

docker and podman share one implementation (``agency/sandbox/container.py``);
only a handful of values differ between them (the ``top`` syntax, how bare
image names are qualified, NVIDIA GPU flags, ``--memory-swap``, the
``--cgroup-parent`` flag, storage-driver layout, rootless id-map discovery,
cgroup scope names). Every test in this file therefore runs once per runtime
and the runtime-specific expectations live in ONE data table (``RUNTIMES``
below) instead of in docker/podman copies of the same test.

Two fixture families are used (see conftest.py):

``container_runtime``  docker | podman, never skipped. Mock-based tests: they
                       patch ``_run``/``subprocess.run``/``/proc`` and never
                       need a daemon, so they run on every host.
``container_kind``     docker | podman, skipping when that daemon is unusable.
                       Real-daemon tests: the user runs these on a host with
                       the runtimes installed.

What does NOT belong here (and lives in the tier above or below):
* behaviour of the whole ``agSandbox`` contract on every backend (exec, file
  I/O, mounts, isolation, checkpoints, process tracking...):
  ``test_all_backends.py``.
* mechanisms that exist in one runtime only (docker's containerd ``ctr``
  helpers and overlay2/overlayfs layout, podman's ``layers.json``, docker
  rootless ``dockerd`` discovery...): ``test_docker_backend.py`` / ``test_podman_backend.py``.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import tarfile
import threading
import time
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock, call, patch

import pytest

import agency.sandbox.container as _container
from agency.configs.agconfig import agconfig as agconfig_cls
from tests.sandbox.conftest import (
    backend_class,
    make_backend,
    make_sandbox,
    started_run_cmd,
)

PYTHON_IMAGE = "docker.io/library/python:3.12-slim"
ALPINE_IMAGE = "docker.io/library/alpine:latest"  # fully qualified: podman prefixes bare names
CONTAINER_ID = "d" * 64


# ---------------------------------------------------------------------------
# Per-runtime data table
# ---------------------------------------------------------------------------


def _docker_fake_root(tmp_path: Path) -> Path:
    root = tmp_path / "docker-root"
    (root / "image" / "overlay2" / "layerdb" / "sha256").mkdir(parents=True)
    (root / "overlay2").mkdir(parents=True)
    return root


def _docker_add_layer(root: Path, diff_id: str, layer_id: str, *, with_content: bool = True):
    entry = root / "image" / "overlay2" / "layerdb" / "sha256" / f"entry-{layer_id}"
    entry.mkdir(parents=True)
    (entry / "diff").write_text(diff_id)
    (entry / "cache-id").write_text(layer_id)
    _maybe_add_diff(root / "overlay2" / layer_id / "diff", with_content)
    return root / "overlay2" / layer_id / "diff"


def _podman_fake_root(tmp_path: Path) -> Path:
    root = tmp_path / "podman-root"
    (root / "overlay-layers").mkdir(parents=True)
    (root / "overlay").mkdir(parents=True)
    (root / "overlay-layers" / "layers.json").write_text("[]")
    return root


def _podman_add_layer(root: Path, diff_id: str, layer_id: str, *, with_content: bool = True):
    layers_path = root / "overlay-layers" / "layers.json"
    layers = json.loads(layers_path.read_text())
    layers.append({"id": layer_id, "diff-digest": diff_id})
    layers_path.write_text(json.dumps(layers))
    _maybe_add_diff(root / "overlay" / layer_id / "diff", with_content)
    return root / "overlay" / layer_id / "diff"


def _maybe_add_diff(diff_dir: Path, with_content: bool) -> None:
    if with_content:
        diff_dir.mkdir(parents=True)
        (diff_dir / "marker").write_text("x")


# Rootless id mapping every runtime must translate identically: container uid
# 0 is the invoking user (host 1000), container uids 1..65536 start at host
# 100000.
ROOTLESS_UID_MAP = [(0, 1000, 1), (1, 100000, 65536)]
ROOTLESS_GID_MAP = [(0, 1000, 1), (1, 100000, 65536)]


@dataclass(frozen=True)
class RuntimeSpec:
    name: str
    # `<rt> top <name> <tail>` and the header line that command prints.
    top_tail: tuple
    top_header: bytes
    # Flags `_gpu_flags()` returns for an NVIDIA host.
    nvidia_gpu_flags: tuple
    # Flags that must never appear for this runtime's NVIDIA GPUs.
    forbidden_gpu_flag: str
    # `update --memory=X` also pins `--memory-swap=X` (docker only).
    pins_memory_swap: bool
    # `agprof.container_cgroup_parent()` becomes `--cgroup-parent=` (docker only).
    honours_cgroup_parent: bool
    # What a bare image name becomes when the sandbox starts it.
    bare_image: Callable[[str], str]
    # Name of the cgroup scope the runtime creates under systemd, and the
    # parent slices of a rootful and a rootless install.
    scope: Callable[[str], str]
    rootful_slice: str
    rootless_slice: str
    # `agprof` daemon kind recorded for a systemd scope.
    daemon_kind: str
    # Storage introspection: name of the `(root, driver)` method, a driver
    # that supports the fast squash path and one that must be refused.
    storage_probe: str
    good_driver: str
    bad_driver: str
    fake_root: Callable[[Path], Path]
    add_layer: Callable
    # Wall-clock budget of the real end-to-end fast squash.
    fast_squash_budget_s: int
    # Order destroy() uses for private-storage cleanup vs container removal.
    destroy_order: tuple
    # CLI flags the real-GPU probe uses to attach NVIDIA GPUs.
    gpu_probe_flags: tuple


RUNTIMES = {
    "docker": RuntimeSpec(
        name="docker",
        top_tail=("-eo", "pid"),
        top_header=b"PID\n",
        nvidia_gpu_flags=("--gpus", "all"),
        forbidden_gpu_flag="nvidia.com/gpu=all",
        pins_memory_swap=True,
        honours_cgroup_parent=True,
        bare_image=lambda name: name,
        scope=lambda cid: f"docker-{cid}.scope",
        rootful_slice="/system.slice",
        rootless_slice="/user.slice/user-1000.slice/user@1000.service/user.slice",
        daemon_kind="dockerd",
        storage_probe="_docker_data_root_and_driver",
        good_driver="overlay2",
        bad_driver="devicemapper",
        fake_root=_docker_fake_root,
        add_layer=_docker_add_layer,
        fast_squash_budget_s=45,
        destroy_order=("storage.destroy", "rm_container"),
        gpu_probe_flags=("--gpus", "all"),
    ),
    "podman": RuntimeSpec(
        name="podman",
        top_tail=("hpid",),
        top_header=b"HPID\n",
        nvidia_gpu_flags=("--device", "nvidia.com/gpu=all"),
        forbidden_gpu_flag="--gpus",
        pins_memory_swap=False,
        honours_cgroup_parent=False,
        bare_image=lambda name: f"localhost/{name}",
        scope=lambda cid: f"libpod-{cid}.scope",
        rootful_slice="/machine.slice",
        rootless_slice="/user.slice/user-1000.slice/user@1000.service/user.slice",
        daemon_kind="conmon",
        storage_probe="_podman_graph_root_and_driver",
        good_driver="overlay",
        bad_driver="vfs",
        fake_root=_podman_fake_root,
        add_layer=_podman_add_layer,
        fast_squash_budget_s=20,
        destroy_order=("rm_container", "storage.destroy"),
        gpu_probe_flags=("--device", "nvidia.com/gpu=all"),
    ),
}


@pytest.fixture
def spec(container_runtime) -> RuntimeSpec:
    return RUNTIMES[container_runtime]


@pytest.fixture
def backend(container_runtime):
    return make_backend(container_runtime)


@pytest.fixture
def kind_spec(container_kind) -> RuntimeSpec:
    return RUNTIMES[container_kind]


class _Done:
    """A CompletedProcess stand-in."""

    def __init__(self, stdout=b"", returncode=0, stderr=b""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def _cli(kind: str, *args: str, check: bool = False, text: bool = True, input=None):
    return subprocess.run(
        [kind, *args], capture_output=True, text=text, check=check, input=input, timeout=300
    )


def _fn_run(handler):
    """Wrap ``handler(args, check, input, timeout)`` as the unbound ``_run``."""

    def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
        return handler(list(args), check, input, timeout)

    return fake_run


@contextmanager
def _committing(sb, handler, *, max_depth=None, extra=()):
    """The mocks every mocked ``commit()`` test shares: a patched ``_run`` on
    the backend's class, a running container, and no virtual GPU."""
    with ExitStack() as stack:
        if max_depth is not None:
            stack.enter_context(
                patch.object(sb._agconfig.sandbox, "checkpoint_squash_max_depth", max_depth)
            )
        stack.enter_context(patch.object(type(sb), "_run", _fn_run(handler)))
        stack.enter_context(patch.object(sb, "_container_status", return_value="running"))
        stack.enter_context(patch.object(sb, "_virtual_gpu_count_requested", 0))
        for cm in extra:
            stack.enter_context(cm)
        yield


LAYERS = "--format={{json .RootFS.Layers}}"
IMAGE_ID = "--format={{.Id}}"


# ===========================================================================
# Module-level helpers of container.py (runtime independent)
# ===========================================================================


class TestPidAlive:
    def test_own_pid_is_alive(self):
        assert _container._pid_alive(os.getpid()) is True

    def test_nonexistent_pid_is_not_alive(self):
        assert _container._pid_alive(2**31 - 1) is False

    def test_permission_error_counts_as_alive(self):
        with patch("os.kill", side_effect=PermissionError):
            assert _container._pid_alive(1) is True


class TestKeyringQuotaDiagnostics:
    def test_keyring_container_limit_uses_kernel_maxkeys_minus_buffer(self):
        with patch("pathlib.Path.read_text", return_value="500\n"):
            limit = _container._keyring_container_limit()
        assert limit == 500 - _container.agconfig_cls().sandbox.container_limit_buffer

    def test_keyring_container_limit_never_below_floor(self):
        with patch("pathlib.Path.read_text", return_value="1\n"):
            limit = _container._keyring_container_limit()
        assert limit == _container.agconfig_cls().sandbox.container_limit_floor

    def test_keyring_container_limit_falls_back_when_proc_unreadable(self):
        with patch("pathlib.Path.read_text", side_effect=OSError("no such file")):
            limit = _container._keyring_container_limit()
        fields = _container.agconfig_cls().sandbox
        assert limit == fields.container_limit_fallback - fields.container_limit_buffer

    def test_keyring_quota_reports_used_max_and_free(self):
        def fake_read_text(self):
            return "200\n" if "maxkeys" in str(self) else "a\nb\nc\n"

        with patch("pathlib.Path.read_text", fake_read_text):
            assert _container.keyring_quota() == {"used": 3, "max": 200, "free": 197}

    def test_keyring_quota_reports_minus_one_when_proc_unreadable(self):
        with patch("pathlib.Path.read_text", side_effect=OSError("no such file")):
            assert _container.keyring_quota() == {"used": -1, "max": -1, "free": -1}

    def test_semaphore_held_count_reflects_acquired_slots(self):
        """held goes up by exactly one per acquire(), checked as a delta since
        ``_container_semaphore`` is a real process-wide semaphore."""
        before_held, limit = _container._semaphore_held_count().split("/")
        _container._container_semaphore.acquire()
        try:
            after_held, limit_after = _container._semaphore_held_count().split("/")
        finally:
            _container._container_semaphore.release()
        assert limit_after == limit
        if before_held == "?" or after_held == "?":
            pytest.skip("sem_getvalue is unavailable on this platform")
        assert int(after_held) == int(before_held) + 1

    def test_semaphore_held_count_falls_back_to_unknown_on_error(self):
        fake_semlock = MagicMock()
        fake_semlock._get_value.side_effect = Exception("boom")
        with patch.object(_container, "_keyring_container_limit", return_value=10):
            with patch.object(_container._container_semaphore, "_semlock", fake_semlock):
                assert _container._semaphore_held_count() == "?/10"


# ===========================================================================
# Runtime detection / orphan reaper
# ===========================================================================


class TestReapOrphanedContainers:
    @pytest.fixture(autouse=True)
    def _reset_once_per_process_guard(self):
        _container._reap_done = False
        yield
        _container._reap_done = False

    @staticmethod
    def _ps(lines):
        result = MagicMock()
        result.returncode = 0
        result.stdout = ("\n".join(lines) + "\n" if lines else "").encode()
        return result

    def _reap(self, runtime, ps_result):
        with patch("agency.sandbox.container.get_container_runtime", return_value=runtime):
            with patch("subprocess.run", return_value=ps_result) as mock_run:
                _container.reap_orphaned_containers()
        return mock_run

    def test_removes_container_owned_by_dead_pid_with_the_right_runtime(self, container_runtime):
        dead = 2**31 - 1
        mock_run = self._reap(
            container_runtime, self._ps([f"abc123\t{dead}\tsandbox-rXXXXXXX-someagent"])
        )
        argvs = [c.args[0] for c in mock_run.call_args_list]
        assert all(a[0] == container_runtime for a in argvs), argvs
        assert any(a[1:3] == ["rm", "-f"] and a[-1] == "abc123" for a in argvs)
        assert any(
            a[1] == "rmi" and a[-1] == "agency/lifecycle-sandbox-rxxxxxxx-someagent" for a in argvs
        )

    def test_skips_container_owned_by_live_pid(self, container_runtime):
        mock_run = self._reap(
            container_runtime, self._ps([f"abc123\t{os.getpid()}\tsandbox-rXXXXXXX-someagent"])
        )
        # container ps, then the lifecycle-image scan -- no rm/rmi.
        assert mock_run.call_count == 2
        assert not any(c.args[0][1] in ("rm", "rmi") for c in mock_run.call_args_list)

    def test_no_containers_found_is_a_noop(self, container_runtime):
        assert self._reap(container_runtime, self._ps([])).call_count == 2

    def test_unparseable_rows_are_skipped(self, container_runtime):
        mock_run = self._reap(
            container_runtime, self._ps(["only-two\tfields", "id\tnot-a-pid\tname"])
        )
        assert not any(c.args[0][1] in ("rm", "rmi") for c in mock_run.call_args_list)

    def test_failed_ps_stops_before_the_image_scan(self, container_runtime):
        failed = MagicMock(returncode=1, stdout=b"")
        assert self._reap(container_runtime, failed).call_count == 1

    def test_runs_at_most_once_per_process(self, container_runtime):
        with patch(
            "agency.sandbox.container.get_container_runtime", return_value=container_runtime
        ):
            with patch("subprocess.run", return_value=self._ps([])) as mock_run:
                for _ in range(3):
                    _container.reap_orphaned_containers()
        assert mock_run.call_count == 2

    def test_no_usable_runtime_is_a_noop(self):
        with patch(
            "agency.sandbox.container.get_container_runtime",
            side_effect=RuntimeError("no docker/podman"),
        ):
            with patch("subprocess.run") as mock_run:
                _container.reap_orphaned_containers()
        mock_run.assert_not_called()

    def test_exception_during_ps_call_is_swallowed(self, container_runtime):
        """A failure here must never block real sandbox construction."""
        with patch(
            "agency.sandbox.container.get_container_runtime", return_value=container_runtime
        ):
            with patch("subprocess.run", side_effect=OSError("binary vanished")):
                _container.reap_orphaned_containers()  # must not raise
        assert _container._reap_done is True


class TestReapOrphanedLifecycleImages:
    """The image-side half of the reaper: lifecycle images whose owning
    container is already gone. Three subprocess shapes are mocked by
    dispatching on the subcommand: ``images --filter``, ``inspect`` (the
    label), and ``ps --filter ancestor=`` (the safety check)."""

    TAG = "agency/lifecycle-someagent:latest"

    @staticmethod
    def _result(returncode=0, stdout=""):
        result = MagicMock()
        result.returncode = returncode
        result.stdout = stdout.encode() if isinstance(stdout, str) else stdout
        return result

    def _dispatcher(self, *, images=None, labels=None, ancestors=None, images_rc=0, ps_rc=0):
        labels = labels or {}

        def fake_run(args, **kwargs):
            if "images" in args:
                if images_rc != 0:
                    return self._result(returncode=images_rc)
                lines = images or []
                return self._result(stdout="\n".join(lines) + ("\n" if lines else ""))
            if "inspect" in args:
                pid = labels.get(args[-1])
                return self._result(stdout="" if pid is None else str(pid))
            if "ps" in args:
                if ps_rc != 0:
                    return self._result(returncode=ps_rc)
                lines = ancestors or []
                return self._result(stdout="\n".join(lines) + ("\n" if lines else ""))
            if "rmi" in args:
                return self._result()
            raise AssertionError(f"unexpected subprocess.run call: {args}")

        return fake_run

    def _reap(self, runtime, own_pid, **dispatch):
        with patch("subprocess.run", side_effect=self._dispatcher(**dispatch)) as mock_run:
            _container._reap_orphaned_lifecycle_images(runtime, own_pid)
        return mock_run

    @staticmethod
    def _rmis(mock_run):
        return [c.args[0] for c in mock_run.call_args_list if c.args[0][1] == "rmi"]

    def test_removes_lifecycle_image_owned_by_dead_pid(self, container_runtime):
        mock_run = self._reap(
            container_runtime,
            os.getpid(),
            images=[self.TAG],
            labels={self.TAG: 2**31 - 1},
        )
        rmis = self._rmis(mock_run)
        assert rmis and rmis[0][0] == container_runtime and self.TAG in rmis[0]
        assert all(c.args[0][0] == container_runtime for c in mock_run.call_args_list)

    def test_skips_image_owned_by_live_pid(self, container_runtime):
        mock_run = self._reap(
            container_runtime,
            os.getpid() + 1,
            images=[self.TAG],
            labels={self.TAG: os.getpid()},
        )
        assert mock_run.call_count == 2  # images list + one inspect
        assert not self._rmis(mock_run)

    def test_skips_image_owned_by_self(self, container_runtime):
        mock_run = self._reap(
            container_runtime, os.getpid(), images=[self.TAG], labels={self.TAG: os.getpid()}
        )
        assert mock_run.call_count == 2

    def test_skips_image_with_missing_or_unparseable_label(self, container_runtime):
        mock_run = self._reap(
            container_runtime, os.getpid(), images=[self.TAG], labels={self.TAG: None}
        )
        assert mock_run.call_count == 2
        assert not self._rmis(mock_run)

    def test_skips_non_lifecycle_prefixed_image_even_with_dead_label(self, container_runtime):
        tag = "agency/ckpt-restore-abc123:latest"
        mock_run = self._reap(container_runtime, os.getpid(), images=[tag], labels={tag: 2**31 - 1})
        assert mock_run.call_count == 1, "only agency/lifecycle-* images are in scope"

    def test_skips_image_still_in_use_by_a_container_despite_dead_label(self, container_runtime):
        """A stale label (e.g. from a checkpoint restored by agent.load()) must
        not get an image a live container runs from deleted."""
        mock_run = self._reap(
            container_runtime,
            os.getpid(),
            images=[self.TAG],
            labels={self.TAG: 2**31 - 1},
            ancestors=["somecontainerid"],
        )
        assert not self._rmis(mock_run)

    def test_ancestor_check_failure_skips_rather_than_deletes(self, container_runtime):
        mock_run = self._reap(
            container_runtime,
            os.getpid(),
            images=[self.TAG],
            labels={self.TAG: 2**31 - 1},
            ps_rc=1,
        )
        assert not self._rmis(mock_run)

    def test_images_scan_failure_is_a_noop(self, container_runtime):
        self._reap(container_runtime, os.getpid(), images_rc=1)  # must not raise

    def test_no_images_found_is_a_noop(self, container_runtime):
        assert self._reap(container_runtime, os.getpid(), images=[]).call_count == 1

    def test_real_scan_finds_no_broken_template(self, container_kind):
        """Regression: the real ``images --filter label=... --format
        {{.Repository}}:{{.Tag}}`` call must be well formed -- a bad template
        used to make returncode != 0 look like "no orphans found"."""
        result = _cli(
            container_kind,
            "images",
            "--filter",
            "label=agency.owner_pid",
            "--format",
            "{{.Repository}}:{{.Tag}}",
            text=False,
        )
        assert result.returncode == 0, result.stderr.decode(errors="replace")


class TestDetectedRuntimeFallback:
    """``get_container_runtime()`` itself: podman preferred, caching, errors."""

    @pytest.fixture(autouse=True)
    def _fresh_cache(self):
        saved = _container._RUNTIME
        _container._RUNTIME = None
        yield
        _container._RUNTIME = saved

    @staticmethod
    def _detect(which, works):
        with patch.object(_container.shutil, "which", side_effect=which):
            with patch.object(_container, "_runtime_works", side_effect=works):
                return _container.get_container_runtime()

    def test_prefers_podman_when_both_work(self):
        assert self._detect(lambda n: f"/usr/bin/{n}", lambda rt: True) == "podman"

    def test_uses_docker_when_podman_is_unusable(self):
        assert self._detect(lambda n: f"/usr/bin/{n}", lambda rt: rt == "docker") == "docker"

    def test_result_is_cached(self):
        assert self._detect(lambda n: f"/usr/bin/{n}", lambda rt: rt == "docker") == "docker"
        with patch.object(_container.shutil, "which", side_effect=AssertionError("not cached")):
            assert _container.get_container_runtime() == "docker"

    def test_installed_but_unreachable_runtimes_are_named_in_the_error(self):
        with pytest.raises(RuntimeError, match="docker is installed but not reachable"):
            self._detect(lambda n: "/usr/bin/docker" if n == "docker" else None, lambda rt: False)

    def test_nothing_installed(self):
        with pytest.raises(RuntimeError, match="Neither docker nor podman"):
            self._detect(lambda n: None, lambda rt: False)


# ===========================================================================
# Static image helpers use the backend's OWN runtime (bug fix)
# ===========================================================================


class TestStaticImageHelpersUseTheirOwnRuntime:
    """tag/delete/export/import/relabel are classmethods: a docker backend must
    run ``docker ...`` even when the process-wide detected runtime (podman is
    preferred) is the other one. Only the runtime-agnostic base class (used by
    ``agent.load()``) falls back to detection."""

    @staticmethod
    def _recording(stdout=b""):
        calls = []

        def fake_run(args, **kwargs):
            calls.append((list(args), kwargs))
            return subprocess.CompletedProcess(args, 0, stdout, b"")

        return calls, fake_run

    @pytest.fixture
    def other_runtime_detected(self, container_runtime):
        other = "podman" if container_runtime == "docker" else "docker"
        with patch.object(_container, "get_container_runtime", return_value=other):
            yield other

    def test_tag_image(self, container_runtime, other_runtime_detected):
        calls, fake = self._recording()
        with patch("subprocess.run", side_effect=fake):
            backend_class(container_runtime).tag_image("src", "dst")
        assert calls[0][0] == [container_runtime, "tag", "src", "dst"]

    @pytest.mark.parametrize("force", [False, True])
    def test_delete_image(self, container_runtime, other_runtime_detected, force):
        calls, fake = self._recording()
        with patch("subprocess.run", side_effect=fake):
            backend_class(container_runtime).delete_image("img", force=force)
        expected = [container_runtime, "rmi"] + (["-f"] if force else []) + ["img"]
        assert calls[0][0] == expected

    def test_export_image(self, container_runtime, other_runtime_detected):
        calls, fake = self._recording(stdout=b"TAR")
        with patch("subprocess.run", side_effect=fake):
            assert backend_class(container_runtime).export_image("img", 7) == b"TAR"
        assert calls[0][0] == [container_runtime, "save", "img"]
        assert calls[0][1]["timeout"] == 7

    def test_import_image(self, container_runtime, other_runtime_detected):
        calls, fake = self._recording()
        with patch("subprocess.run", side_effect=fake):
            backend_class(container_runtime).import_image(b"TAR", 9)
        assert calls[0][0] == [container_runtime, "load"]
        assert calls[0][1]["input"] == b"TAR"

    def test_relabel_owner_pid(self, container_runtime, other_runtime_detected):
        calls, fake = self._recording(stdout=b"cid\n")
        with patch("subprocess.run", side_effect=fake):
            backend_class(container_runtime).relabel_owner_pid("img", 5, 3)
        assert {c[0][0] for c in calls} == {container_runtime}

    @pytest.mark.parametrize("detected", ["docker", "podman"])
    def test_base_class_uses_the_detected_runtime(self, detected):
        calls, fake = self._recording()
        with patch.object(_container, "get_container_runtime", return_value=detected):
            with patch("subprocess.run", side_effect=fake):
                _container._ContainerBackendBase.tag_image("src", "dst")
        assert calls[0][0][0] == detected

    def test_image_helper_holds_a_semaphore_slot(self, container_runtime):
        calls, fake = self._recording()
        with patch("subprocess.run", side_effect=fake):
            with patch.object(_container, "_docker_semaphore_slot") as slot:
                backend_class(container_runtime).tag_image("a", "b")
        slot.assert_called_once_with()


class TestRelabelOwnerPid:
    """``relabel_owner_pid`` scrubs/restamps the owner label on a checkpoint's
    embedded image (agent.save()/load()) via a throwaway ``create`` + commit."""

    @staticmethod
    def _run(runtime, owner_pid, *, fail_commit=False):
        calls = []

        def fake_run(args, **kwargs):
            calls.append(list(args))
            if "create" in args:
                return _Done(stdout=b"fakecontainerid123\n")
            if fail_commit and "commit" in args:
                raise subprocess.CalledProcessError(1, args)
            return _Done()

        with patch("subprocess.run", side_effect=fake_run):
            if fail_commit:
                with pytest.raises(subprocess.CalledProcessError):
                    backend_class(runtime).relabel_owner_pid("agency/ckpt-foo", owner_pid, 30)
            else:
                backend_class(runtime).relabel_owner_pid("agency/ckpt-foo", owner_pid, 30)
        return calls

    def test_stamps_given_pid_via_create_then_commit(self, container_runtime):
        calls = self._run(container_runtime, 4242)
        create = [c for c in calls if "create" in c]
        commit = [c for c in calls if "commit" in c]
        rm = [c for c in calls if c[1:3] == ["rm", "-f"]]
        assert create and create[0][-1] == "agency/ckpt-foo"
        assert commit[0][commit[0].index("--change") + 1] == "LABEL agency.owner_pid=4242"
        assert commit[0][-2:] == ["fakecontainerid123", "agency/ckpt-foo"]
        assert rm and rm[0][-1] == "fakecontainerid123"

    def test_none_pid_clears_the_label(self, container_runtime):
        commit = [c for c in self._run(container_runtime, None) if "commit" in c]
        assert commit[0][commit[0].index("--change") + 1] == "LABEL agency.owner_pid="

    def test_temp_container_removed_even_when_commit_fails(self, container_runtime):
        calls = self._run(container_runtime, 1, fail_commit=True)
        rm = [c for c in calls if c[1:3] == ["rm", "-f"]]
        assert rm and rm[0][-1] == "fakecontainerid123"


# ===========================================================================
# Concurrency slot and session-keyring quota hooks
# ===========================================================================


class TestQuotaHooks:
    """Both runtimes are subject to the same kernel session-keyring quota, so
    ``_ContainerBackendBase``'s hooks are concrete and identical."""

    def test_matches_session_key_message(self, backend):
        assert backend._is_quota_exhaustion_error(
            "unable to create session key: disk quota exceeded"
        )

    def test_matches_disk_quota_exceeded_with_keyring(self, backend):
        assert backend._is_quota_exhaustion_error("disk quota exceeded for keyring")

    def test_plain_disk_quota_exceeded_does_not_match(self, backend):
        """A full overlay volume also says "disk quota exceeded"."""
        assert not backend._is_quota_exhaustion_error("disk quota exceeded")

    def test_unrelated_stderr_does_not_match(self, backend):
        assert not backend._is_quota_exhaustion_error("no such image: docker.io/library/python")

    def test_wait_for_quota_slot_polls_until_a_slot_frees_up(self, backend):
        quotas = iter([{"free": 0}, {"free": 0}, {"free": 1}])
        with patch.object(_container, "keyring_quota", side_effect=lambda: next(quotas)):
            with patch.object(_container.time, "sleep") as sleep_mock:
                backend._wait_for_quota_slot()
        assert sleep_mock.call_count == 2

    def test_wait_for_quota_slot_gives_up_once_deadline_passes(self, backend):
        moments = iter([0.0, 10_000.0])
        with patch.object(_container, "keyring_quota", return_value={"free": 0}):
            with patch.object(_container.time, "monotonic", side_effect=lambda: next(moments)):
                with patch.object(_container.time, "sleep") as sleep_mock:
                    backend._wait_for_quota_slot()
        sleep_mock.assert_not_called()

    def test_quota_diagnostics_format(self, backend):
        with patch.object(_container, "keyring_quota", return_value={"used": 5, "max": 200}):
            with patch.object(_container, "_semaphore_held_count", return_value="3/195"):
                assert (
                    backend._quota_diagnostics()
                    == "[keyring: 5/200 used, framework semaphore: 3/195 held]"
                )

    def test_acquire_and_release_runtime_slot_use_the_shared_semaphore(self, backend):
        with patch.object(_container._container_semaphore, "acquire") as acquire_mock:
            backend._acquire_runtime_slot()
        acquire_mock.assert_called_once()
        with patch.object(_container._container_semaphore, "release") as release_mock:
            backend._release_runtime_slot()
        release_mock.assert_called_once()

    def test_both_runtimes_share_one_semaphore(self):
        a, b = make_backend("docker"), make_backend("podman")
        with patch.object(_container._container_semaphore, "acquire") as acquire_mock:
            a._acquire_runtime_slot()
            b._acquire_runtime_slot()
        assert acquire_mock.call_count == 2


# ===========================================================================
# _run(): the CLI wrapper
# ===========================================================================


class TestRun:
    """``_run`` survives a CLI child stuck in uninterruptible sleep: SIGKILL
    cannot preempt a D-state process, so ``subprocess.run(timeout=)`` is not a
    reliable ceiling (see ``run_with_unkillable_child_grace``). These mock
    ``subprocess.run`` directly so the real threading/timeout logic executes."""

    def test_fast_path_returns_the_real_completed_process(self, backend, container_runtime):
        fake = subprocess.CompletedProcess([container_runtime, "x"], 0, b"out", b"")
        with patch("subprocess.run", return_value=fake) as mock_run:
            assert backend._run([container_runtime, "x"], timeout=5) is fake
        mock_run.assert_called_once()

    def test_check_true_failure_raises_runtime_error_with_existing_message_format(
        self, backend, container_runtime
    ):
        err = subprocess.CalledProcessError(1, [container_runtime, "x"], output=b"", stderr=b"boom")
        with patch("subprocess.run", side_effect=err):
            with pytest.raises(
                RuntimeError, match=rf"{container_runtime} x failed \(exit 1\): boom"
            ):
                backend._run([container_runtime, "x"], check=True, timeout=5)

    def test_ordinary_fast_timeout_still_raises_timeout_expired(self, backend, container_runtime):
        with patch(
            "subprocess.run", side_effect=subprocess.TimeoutExpired([container_runtime, "x"], 5)
        ):
            with pytest.raises(subprocess.TimeoutExpired):
                backend._run([container_runtime, "x"], timeout=5)

    def test_gives_up_instead_of_hanging_when_child_outlives_timeout_and_grace(
        self, backend, container_runtime
    ):
        with patch.object(backend._agconfig.sandbox, "unkillable_child_grace_s", 0.05):
            with patch("subprocess.run", side_effect=lambda *a, **k: time.sleep(10)):
                start = time.monotonic()
                with pytest.raises(subprocess.TimeoutExpired):
                    backend._run([container_runtime, "x"], timeout=0.05)
                elapsed = time.monotonic() - start
        assert elapsed < 5

    def test_semaphore_is_released_immediately_on_give_up_not_leaked(
        self, backend, container_runtime
    ):
        """A wedged call must not starve every OTHER sandbox's docker/podman
        calls by holding the shared concurrency semaphore forever."""
        sem = _container._get_docker_semaphore()
        with patch.object(backend._agconfig.sandbox, "unkillable_child_grace_s", 0.05):
            with patch("subprocess.run", side_effect=lambda *a, **k: time.sleep(10)):
                with pytest.raises(subprocess.TimeoutExpired):
                    backend._run([container_runtime, "x"], timeout=0.05)
        acquired = sem.acquire(blocking=False)
        assert acquired, "docker semaphore slot was not released on give-up"
        sem.release()

    def test_semaphore_is_released_after_a_normal_failure(self, backend, container_runtime):
        sem = _container._get_docker_semaphore()
        with patch("subprocess.run", side_effect=OSError("exec failed")):
            with pytest.raises(OSError):
                backend._run([container_runtime, "x"], timeout=5)
        assert sem.acquire(blocking=False)
        sem.release()

    def test_private_storage_never_contacts_the_shared_store_before_provisioning(
        self, backend, container_runtime
    ):
        backend._checkpoint_storage = MagicMock(prepared=False)
        with patch("subprocess.run") as mock_run:
            result = backend._run([container_runtime, "ps"], timeout=5)
        mock_run.assert_not_called()
        assert result.returncode == 1


# ===========================================================================
# Small CLI helpers
# ===========================================================================


class TestCommandHelpers:
    """``_rm_container`` / ``_rmi`` / state inspection / lifecycle tag."""

    @staticmethod
    def _capture(backend, stdout=b"", returncode=0):
        calls = []

        def handler(args, check, input, timeout):
            calls.append((args, check, timeout))
            return _Done(stdout=stdout, returncode=returncode)

        return calls, patch.object(type(backend), "_run", _fn_run(handler))

    def test_rm_container_sends_rm_force_args(self, backend, container_runtime):
        calls, patcher = self._capture(backend)
        with patcher:
            backend._rm_container("my-container")
        [(args, check, _timeout)] = calls
        assert args == [container_runtime, "rm", "-f", "my-container"]
        assert check is True

    def test_rm_container_raises_on_failure(self, backend):
        with patch.object(type(backend), "_run", side_effect=RuntimeError("rm failed")):
            with pytest.raises(RuntimeError, match="rm failed"):
                backend._rm_container("bad-container")

    def test_rmi_sends_rmi_args_without_force(self, backend, container_runtime):
        calls, patcher = self._capture(backend)
        with patcher:
            backend._rmi("sha256:abc123")
        [(args, check, _timeout)] = calls
        assert args == [container_runtime, "rmi", "sha256:abc123"]
        assert check is True

    def test_rmi_force_adds_dash_f(self, backend, container_runtime):
        calls, patcher = self._capture(backend)
        with patcher:
            backend._rmi("myimage:tag", force=True)
        assert calls[0][0] == [container_runtime, "rmi", "-f", "myimage:tag"]

    def test_rmi_raises_on_failure(self, backend):
        with patch.object(type(backend), "_run", side_effect=RuntimeError("rmi failed")):
            with pytest.raises(RuntimeError, match="rmi failed"):
                backend._rmi("sha256:deadbeef")

    @pytest.mark.parametrize(
        "stdout,rc,expected",
        [(b"true\n", 0, True), (b"false\n", 0, False), (b"true", 1, False)],
    )
    def test_container_running_parses_inspect(self, backend, stdout, rc, expected):
        _calls, patcher = self._capture(backend, stdout=stdout, returncode=rc)
        with patcher:
            assert backend._container_running() is expected

    @pytest.mark.parametrize(
        "stdout,rc,expected",
        [(b"exited\n", 0, "exited"), (b"created", 0, "created"), (b"", 1, "")],
    )
    def test_container_status_parses_inspect(self, backend, stdout, rc, expected):
        _calls, patcher = self._capture(backend, stdout=stdout, returncode=rc)
        with patcher:
            assert backend._container_status() == expected

    @pytest.mark.parametrize(
        "stdout,rc,expected",
        [
            (b"true|running\n", 0, (True, "running")),
            (b"false|exited\n", 0, (False, "exited")),
            (b"false|created", 0, (False, "created")),
            (b"", 1, (False, "")),
        ],
    )
    def test_inspect_container_state_is_one_merged_call(
        self, backend, container_runtime, stdout, rc, expected
    ):
        calls, patcher = self._capture(backend, stdout=stdout, returncode=rc)
        with patcher:
            assert backend._inspect_container_state() == expected
        assert len(calls) == 1
        assert calls[0][0][:2] == [container_runtime, "inspect"]

    def test_lifecycle_tag_is_lowercase(self, backend):
        """Both runtimes reject uppercase repository names."""
        backend._name = "GenerationAgent_4816622_0000"
        tag = backend._lifecycle_tag()
        assert tag == tag.lower() and "generationagent" in tag

    def test_lifecycle_tag_format(self, backend):
        backend._name = "myagent_0000"
        assert backend._lifecycle_tag() == "agency/lifecycle-myagent_0000"

    def test_resolve_image_qualifies_bare_names_per_runtime(self, backend, spec):
        assert backend._resolve_image("python-base") == spec.bare_image("python-base")

    @pytest.mark.parametrize(
        "ref", [PYTHON_IMAGE, "registry.example.com/team/img:1", "sha256:" + "a" * 64, "f" * 64]
    )
    def test_resolve_image_leaves_qualified_names_and_ids_alone(self, backend, ref):
        assert backend._resolve_image(ref) == ref


# ===========================================================================
# `top`: host pids inside the container
# ===========================================================================


class TestOwnHostPids:
    """``_own_host_pids`` uses each runtime's own `top` syntax. They are NOT
    interchangeable: ps(1)-style flags on podman silently run a real ps INSIDE
    the container and report container-local pids; podman's plain `pid`
    descriptor is container-local too, only `hpid` maps to host pids."""

    @staticmethod
    def _top(backend, stdout=b"", returncode=0):
        ok = MagicMock(returncode=returncode, stdout=stdout)
        return patch("subprocess.run", return_value=ok)

    def test_uses_the_runtimes_own_top_syntax(self, backend, spec, container_runtime):
        with self._top(backend, spec.top_header + b"100\n") as run:
            backend._own_host_pids()
        args = run.call_args.args[0]
        assert args[:2] == [container_runtime, "top"]
        assert args[2] == backend._container_name()
        assert tuple(args[3:]) == spec.top_tail

    def test_podman_never_uses_ps_style_flags(self):
        sb = make_backend("podman")
        with self._top(sb, b"HPID\n100\n") as run:
            sb._own_host_pids()
        args = run.call_args.args[0]
        assert "-eo" not in args and "-o" not in args

    def test_parses_pids_skipping_header(self, backend, spec):
        with self._top(backend, spec.top_header + b"100\n101\n102\n"):
            assert backend._own_host_pids() == {100, 101, 102}

    def test_includes_exec_spawned_sibling_pids(self, backend, spec):
        """init(100) and an unrelated exec'd sibling(200) come back together,
        with no ancestry between them -- the original regression."""
        with self._top(backend, spec.top_header + b"100\n200\n"):
            assert backend._own_host_pids() == {100, 200}

    def test_returns_empty_when_top_fails(self, backend):
        with self._top(backend, returncode=1):
            assert backend._own_host_pids() == set()

    def test_returns_empty_when_only_header_present(self, backend, spec):
        with self._top(backend, spec.top_header):
            assert backend._own_host_pids() == set()

    def test_header_is_skipped_by_position_not_by_text(self, backend):
        with self._top(backend, b"Some Other Header Text\n100\n101\n"):
            assert backend._own_host_pids() == {100, 101}

    def test_non_numeric_rows_are_ignored(self, backend, spec):
        with self._top(backend, spec.top_header + b"100\n<defunct>\n\n 7 \n"):
            assert backend._own_host_pids() == {100, 7}

    def test_real_container_includes_a_separately_execd_sibling_process(self, container_kind):
        """Against a REAL container started with `tail -f /dev/null` as init and
        the work spawned by a SEPARATE `exec -d` (how every real command runs):
        that process is a sibling under the runtime supervisor, not a /proc
        descendant of init, which is what the old /proc-walk missed."""
        spec_ = RUNTIMES[container_kind]
        name = f"own-host-pids-real-{container_kind}-{uuid.uuid4().hex[:8]}"
        try:
            _cli(
                container_kind,
                "run",
                "-d",
                "--name",
                name,
                ALPINE_IMAGE,
                "tail",
                "-f",
                "/dev/null",
                check=True,
            )
            init_pid = int(
                _cli(
                    container_kind, "inspect", "--format", "{{.State.Pid}}", name, check=True
                ).stdout.strip()
            )
            _cli(container_kind, "exec", "-d", name, "sh", "-c", "sleep 60", check=True)
            time.sleep(0.5)
            children = Path(f"/proc/{init_pid}/task/{init_pid}/children").read_text().split()
            assert children == [], "the exec'd process IS a /proc child of init on this host"

            result = make_backend(container_kind, name=name, agname="agent")._own_host_pids()
            top = _cli(container_kind, "top", name, *spec_.top_tail, check=True)
            expected = {
                int(line.strip()) for line in top.stdout.splitlines()[1:] if line.strip().isdigit()
            }
            assert len(expected) >= 2, "expected both init and the exec'd sleep in `top`"
            assert result == expected and init_pid in result
        finally:
            _cli(container_kind, "rm", "-f", name)


# ===========================================================================
# CPU quota capability and cgroup layouts
# ===========================================================================


class TestCfsSupported:
    """``_cfs_supported`` is a staticmethod on the shared base, so docker and
    podman are the same code; it runs through each backend class anyway."""

    @pytest.mark.parametrize(
        "v1_quota_file_exists, v2_controllers, expected",
        [
            (True, None, True),
            (False, "cpuset cpu io memory pids", True),
            (False, "cpuset io memory pids", False),
            (False, None, False),
            (True, "cpuset io memory pids", True),  # hybrid: v1 quota wins
            (False, "", False),  # v2 mounted but empty controller list
            (False, "cpuset cpuset-cpu io", False),  # substring must not match
        ],
        ids=[
            "cgroup-v1",
            "cgroup-v2-with-cpu",
            "cgroup-v2-without-cpu",
            "no-cgroup-files",
            "hybrid",
            "v2-empty-controllers",
            "v2-cpu-substring-only",
        ],
    )
    def test_cfs_supported_follows_the_cgroup_files(
        self,
        tmp_path,
        monkeypatch,
        container_runtime,
        v1_quota_file_exists,
        v2_controllers,
        expected,
    ):
        v1_quota_file = tmp_path / "cfs_quota_us"
        if v1_quota_file_exists:
            v1_quota_file.write_text("-1")
        v2_controllers_file = tmp_path / "cgroup.controllers"
        if v2_controllers is not None:
            v2_controllers_file.write_text(v2_controllers + "\n")
        monkeypatch.setattr(_container, "_CGROUP_V1_CFS_QUOTA", str(v1_quota_file))
        monkeypatch.setattr(_container, "_CGROUP_V2_CONTROLLERS", str(v2_controllers_file))
        assert type(make_backend(container_runtime))._cfs_supported() is expected
        assert _container._ContainerBackendBase._cfs_supported() is expected


class TestProfilerContainerRegistration:
    """``_register_prof_container`` resolves the container's cgroup from the
    runtime-reported init PID. Kernel and runtime surfaces are mocked, so this
    runs without a daemon over cgroup v1/v2/hybrid/missing layouts, systemd vs
    cgroupfs scope naming and rootful vs rootless parents, for both runtimes."""

    LABEL = "sandbox_research_agent_a1b2"

    def _sb(self, runtime, agname=LABEL):
        return make_backend(runtime, agname=agname, name="profiler-registration-test")

    def _register(self, sb, *, cgroup_text=None, read_error=None, isdirs=(), inspect=None):
        """Run the registration with the profiler on. Returns (started mock)."""
        if inspect is None:
            inspect = MagicMock(returncode=0, stdout=f"{CONTAINER_ID}|4242\n".encode())
        read = (
            patch.object(_container.Path, "read_text", side_effect=read_error)
            if read_error is not None
            else patch.object(_container.Path, "read_text", return_value=cgroup_text)
        )
        with patch.object(_container.agprof, "enabled", return_value=True):
            with patch.object(_container.agprof, "container_registered", return_value=False):
                with patch.object(sb, "_run", return_value=inspect) as run:
                    with read:
                        with patch.object(
                            _container.os.path, "isdir", side_effect=lambda p: p in isdirs
                        ):
                            with patch.object(_container.agprof, "container_started") as started:
                                self.run_mock, self.started = run, started
                                sb._register_prof_container()
        return started

    # -- identity and no-op paths

    def test_profiler_label_preserves_unique_sandbox_identity(self, container_runtime):
        assert self._sb(container_runtime)._prof_container_label() == self.LABEL
        assert self._sb(container_runtime, "plain_agent")._prof_container_label() == "plain_agent"

    def test_registration_is_a_noop_without_an_active_profiler(self, container_runtime):
        sb = self._sb(container_runtime)
        with patch.object(_container.agprof, "enabled", return_value=False):
            with patch.object(sb, "_run") as run:
                sb._register_prof_container()
        run.assert_not_called()

    def test_already_registered_container_skips_runtime_and_proc_probes(self, container_runtime):
        sb = self._sb(container_runtime)
        with patch.object(_container.agprof, "enabled", return_value=True):
            with patch.object(_container.agprof, "container_registered", return_value=True):
                with patch.object(sb, "_run") as run:
                    with patch.object(_container.Path, "read_text") as read_text:
                        sb._register_prof_container()
        run.assert_not_called()
        read_text.assert_not_called()

    def test_inspect_asks_the_runtime_for_id_and_init_pid(self, container_runtime, spec):
        sb = self._sb(container_runtime)
        cgroup = f"0::{spec.rootful_slice}/{spec.scope(CONTAINER_ID)}\n"
        self._register(sb, cgroup_text=cgroup)
        assert self.run_mock.call_args.args[0] == [
            container_runtime,
            "inspect",
            "--format",
            "{{.Id}}|{{.State.Pid}}",
            sb._name,
        ]

    # -- systemd scopes: rootful, rootless, with and without the `container` leaf

    @pytest.mark.parametrize("rootless", [False, True], ids=["rootful", "rootless"])
    @pytest.mark.parametrize(
        "payload_suffix", ["", "/container"], ids=["scope-leaf", "container-child"]
    )
    def test_systemd_scope_is_registered_under_its_cgroup_dir(
        self, container_runtime, spec, rootless, payload_suffix
    ):
        """The payload may sit in a `container` child of the runtime's scope
        (podman's systemd manager); the ID-bearing component identifies the
        container wherever it is in the path."""
        sb = self._sb(container_runtime)
        parent = spec.rootless_slice if rootless else spec.rootful_slice
        rel = f"{parent}/{spec.scope(CONTAINER_ID)}{payload_suffix}"
        started = self._register(sb, cgroup_text=f"0::{rel}\n")
        assert started.call_count >= 1
        for registration in started.call_args_list:
            assert registration.args[0] == self.LABEL
            assert registration.args[1] == f"/sys/fs/cgroup{rel}"
            assert registration.args[3] == spec.daemon_kind

    def test_podman_registers_conmon_scope_next_to_the_libpod_scope(self):
        sb = self._sb("podman")
        rel = f"/user.slice/libpod-{CONTAINER_ID}.scope/container"
        conmon = f"/sys/fs/cgroup/user.slice/libpod-conmon-{CONTAINER_ID}.scope"
        started = self._register(sb, cgroup_text=f"0::{rel}\n", isdirs={conmon})
        started.assert_called_once_with(self.LABEL, f"/sys/fs/cgroup{rel}", conmon, "conmon")

    def test_podman_without_a_conmon_scope_registers_no_daemon_dir(self):
        sb = self._sb("podman")
        rel = f"/user.slice/libpod-{CONTAINER_ID}.scope"
        started = self._register(sb, cgroup_text=f"0::{rel}\n", isdirs=())
        started.assert_called_once_with(self.LABEL, f"/sys/fs/cgroup{rel}", None, "conmon")

    def test_docker_registers_the_daemon_service_cgroups_it_finds(self):
        sb = self._sb("docker")
        rel = f"/system.slice/docker-{CONTAINER_ID}.scope"
        service = "/sys/fs/cgroup/system.slice/containerd.service"
        started = self._register(sb, cgroup_text=f"0::{rel}\n", isdirs={service})
        assert (
            call(self.LABEL, f"/sys/fs/cgroup{rel}", service, "dockerd") in started.call_args_list
        )

    # -- cgroupfs naming (no systemd scope)

    def test_docker_cgroupfs_layout_is_accepted_by_bare_container_id(self):
        """With the cgroupfs driver docker's cgroup is `/docker/<id>`: the bare
        ID component identifies it, and there is no scope to derive a daemon
        cgroup from."""
        sb = self._sb("docker")
        started = self._register(sb, cgroup_text=f"0::/docker/{CONTAINER_ID}\n")
        started.assert_called_once_with(
            self.LABEL, f"/sys/fs/cgroup/docker/{CONTAINER_ID}", None, "conmon"
        )

    def test_podman_cgroupfs_layout_is_rejected_as_unrecognised(self):
        """Pins current behaviour: podman's cgroupfs layout
        (`/libpod_parent/libpod-<id>`) has neither the bare ID nor a
        `libpod-<id>.scope` component, so profiling refuses it instead of
        recording a cgroup it cannot tie to the container."""
        sb = self._sb("podman")
        with pytest.raises(RuntimeError, match="does not match container"):
            self._register(sb, cgroup_text=f"0::/libpod_parent/libpod-{CONTAINER_ID}\n")

    # -- cgroup versions and unreadable cgroup

    def test_cgroup_v1_only_is_rejected(self, container_runtime):
        sb = self._sb(container_runtime)
        text = "12:cpu,cpuacct:/legacy\n2:cpu:/legacy\n1:name=systemd:/legacy\n"
        with pytest.raises(RuntimeError, match="cgroup v2"):
            self._register(sb, cgroup_text=text)

    def test_hybrid_layout_uses_the_unified_entry(self, container_runtime, spec):
        sb = self._sb(container_runtime)
        rel = f"{spec.rootful_slice}/{spec.scope(CONTAINER_ID)}"
        text = f"12:cpu,cpuacct:/legacy\n1:name=systemd:/legacy\n0::{rel}\n"
        started = self._register(sb, cgroup_text=text)
        assert started.call_args.args[1] == f"/sys/fs/cgroup{rel}"

    def test_unreadable_cgroup_file_is_reported(self, container_runtime):
        sb = self._sb(container_runtime)
        with pytest.raises(RuntimeError, match="cannot read cgroup of container PID 4242"):
            self._register(sb, read_error=OSError("no such process"))

    def test_remote_daemon_pid_with_unrelated_local_cgroup_is_rejected(self, container_runtime):
        sb = self._sb(container_runtime)
        unrelated = "0::/user.slice/user-501.slice/session-7.scope\n"
        with pytest.raises(RuntimeError, match="remote or VM-backed"):
            self._register(sb, cgroup_text=unrelated)
        self.started.assert_not_called()

    @pytest.mark.parametrize(
        "stdout,rc",
        [
            (b"", 1),  # inspect failed
            (b"no-separator", 0),
            (f"{CONTAINER_ID}|0\n".encode(), 0),  # pid 0
            (f"{CONTAINER_ID}|abc\n".encode(), 0),  # pid not a number
            (b"shortid|4242\n", 0),  # not a 64-hex container id
        ],
        ids=["inspect-failed", "no-separator", "pid-zero", "pid-not-int", "short-id"],
    )
    def test_unresolvable_id_or_pid_is_reported(self, container_runtime, stdout, rc):
        sb = self._sb(container_runtime)
        inspect = MagicMock(returncode=rc, stdout=stdout)
        with pytest.raises(RuntimeError, match="cannot resolve ID and init PID"):
            self._register(sb, cgroup_text="0::/x\n", inspect=inspect)

    # -- lifecycle hooks

    def test_ensure_started_registers_after_every_successful_start_path(self, container_runtime):
        sb = self._sb(container_runtime)
        with patch.object(sb, "_ensure_started_profiled") as ensure:
            with patch.object(sb, "_register_prof_container") as register:
                sb._ensure_started()
        ensure.assert_called_once_with()
        register.assert_called_once_with()

    def test_stop_deregisters_after_container_is_confirmed_stopped(self, container_runtime):
        sb = self._sb(container_runtime)
        with patch.object(sb, "_container_running", side_effect=[True, False]):
            with patch.object(sb, "_run"):
                with patch.object(sb, "_release_runtime_slot"):
                    with patch.object(_container.agprof, "container_stopped") as stopped:
                        sb.stop()
        stopped.assert_called_once_with(self.LABEL)

    def test_failed_stop_keeps_running_container_registered(self, container_runtime):
        sb = self._sb(container_runtime)
        with patch.object(sb, "_container_running", side_effect=[True, True]):
            with patch.object(sb, "_run", side_effect=RuntimeError("stop failed")):
                with patch.object(_container.agprof, "container_stopped") as stopped:
                    with pytest.raises(RuntimeError, match="stop failed"):
                        sb.stop()
        stopped.assert_not_called()

    def test_rm_deregisters_only_after_successful_removal(self, container_runtime):
        sb = self._sb(container_runtime)
        with patch.object(sb, "_container_status", return_value="running"):
            with patch.object(sb, "_container_running", side_effect=[True, False]):
                with patch.object(sb, "_rm_container"):
                    with patch.object(sb, "_release_runtime_slot"):
                        with patch.object(_container.agprof, "container_stopped") as stopped:
                            sb.rm_container()
        stopped.assert_called_once_with(self.LABEL)

    def test_failed_rm_keeps_a_still_running_container_registered(self, container_runtime):
        sb = self._sb(container_runtime)
        with patch.object(sb, "_container_status", return_value="running"):
            with patch.object(sb, "_container_running", side_effect=[True, True, True]):
                with patch.object(sb, "_rm_container", side_effect=RuntimeError("rm failed")):
                    with patch.object(_container.time, "sleep"):
                        with patch.object(_container.agprof, "container_stopped") as stopped:
                            with pytest.raises(RuntimeError, match="rm failed"):
                                sb.rm_container()
        stopped.assert_not_called()

    def test_rm_of_a_missing_container_still_deregisters(self, container_runtime):
        sb = self._sb(container_runtime)
        with patch.object(sb, "_container_status", return_value=""):
            with patch.object(_container.agprof, "container_stopped") as stopped:
                sb.rm_container()
        stopped.assert_called_once_with(self.LABEL)


# ===========================================================================
# GPU flags
# ===========================================================================


class TestGpuFlags:
    """``_gpu_flags(runtime)``. Regression: docker's ``--gpus all`` was used
    for podman too, which accepts it without mounting the driver, so the
    container silently had no GPU. Every GPU is attached to every container at
    ``run`` time (no reserve_gpu ordering dependency)."""

    @pytest.fixture(autouse=True)
    def _clear_caches(self):
        _container._gpu_kind_cache.clear()
        _container._amd_render_node_paths_cache = None
        yield
        _container._gpu_kind_cache.clear()
        _container._amd_render_node_paths_cache = None

    def test_no_gpu_detected_returns_empty(self, container_runtime):
        with patch.object(_container, "detect_gpus", return_value=[]):
            assert _container._gpu_flags(container_runtime) == []

    def test_nvidia_flags_follow_the_runtimes_syntax(self, container_runtime, spec):
        with patch.object(_container, "detect_gpus", return_value=[0, 1]):
            with patch.object(_container.shutil, "which", return_value="/usr/bin/nvidia-smi"):
                flags = _container._gpu_flags(container_runtime)
        assert tuple(flags) == spec.nvidia_gpu_flags
        assert spec.forbidden_gpu_flag not in flags

    def test_nvidia_flags_do_not_depend_on_how_many_gpus_there_are(self, container_runtime):
        with patch.object(_container.shutil, "which", return_value="/usr/bin/nvidia-smi"):
            with patch.object(_container, "detect_gpus", return_value=[0]):
                one = _container._gpu_flags(container_runtime)
            with patch.object(_container, "detect_gpus", return_value=[0, 1, 2, 3]):
                many = _container._gpu_flags(container_runtime)
        assert one == many

    def test_runtimes_differ_for_nvidia(self):
        with patch.object(_container, "detect_gpus", return_value=[0]):
            with patch.object(_container.shutil, "which", return_value="/usr/bin/nvidia-smi"):
                assert _container._gpu_flags("docker") != _container._gpu_flags("podman")

    @staticmethod
    def _rocm(tmp_path, nodes, *, pci_order=None, gpus=(0, 1)):
        dri = tmp_path / "dri"
        dri.mkdir()
        for node in nodes:
            (dri / node).touch()
        (dri / "card0").touch()  # not a render node -- never selected
        return dri, [str(dri / n) for n in (pci_order or [])] or None

    def _flags(self, runtime, dri, pci, gpus):
        with patch.object(_container, "detect_gpus", return_value=list(gpus)):
            with patch.object(_container.shutil, "which", return_value=None):
                with patch.object(_container, "amd_render_node_paths_by_pci_bus", return_value=pci):
                    with patch.object(
                        _container, "Path", lambda p: dri if p == "/dev/dri" else Path(p)
                    ):
                        return _container._gpu_flags(runtime)

    def test_rocm_attaches_kfd_and_every_render_node_identically_for_both_runtimes(
        self, container_runtime, tmp_path
    ):
        dri, pci = self._rocm(tmp_path, ["renderD128", "renderD129"])
        flags = self._flags(container_runtime, dri, pci, (0, 1))
        assert flags == [
            "--device", "/dev/kfd",
            "--device", str(dri / "renderD128"),
            "--device", str(dri / "renderD129"),
        ]  # fmt: skip

    def test_rocm_single_render_node_present(self, container_runtime, tmp_path):
        dri, pci = self._rocm(tmp_path, ["renderD128"])
        flags = self._flags(container_runtime, dri, pci, (0,))
        assert flags == ["--device", "/dev/kfd", "--device", str(dri / "renderD128")]

    def test_rocm_follows_pci_bus_ordering_when_available(self, container_runtime, tmp_path):
        dri, pci = self._rocm(
            tmp_path, ["renderD128", "renderD129"], pci_order=["renderD129", "renderD128"]
        )
        flags = self._flags(container_runtime, dri, pci, (0, 1))
        assert flags[3::2] == [str(dri / "renderD129"), str(dri / "renderD128")]

    def test_backend_construction_does_not_eagerly_cache_gpu_flags(self, container_runtime):
        with patch.object(_container, "_gpu_flags", return_value=["sentinel"]) as gpu_flags_mock:
            sb = make_backend(container_runtime)
        gpu_flags_mock.assert_not_called()
        assert "_gpu_flags" not in vars(sb)


# ===========================================================================
# `run` command construction (the argv _ensure_started builds)
# ===========================================================================


_started_run_cmd = started_run_cmd


class TestOwnerPidLabel:
    """The owner-PID label lets the orphan reaper tell a dead owner from
    another process operating on a serialized copy of a live owner's backend."""

    @staticmethod
    def _label_value(run_cmd):
        label = run_cmd[run_cmd.index("--label") + 1]
        assert label.startswith("agency.owner_pid=")
        return label.split("=", 1)[1]

    def test_label_defaults_to_constructing_processs_own_pid(self, backend):
        assert self._label_value(_started_run_cmd(backend)) == str(os.getpid())

    def test_label_reflects_owner_pid_attribute_not_live_process(self, backend):
        """Simulates a cloudpickle-to-a-worker copy: the label tracks the
        stored ``_owner_pid``, not a fresh ``os.getpid()``."""
        backend._owner_pid = 424242
        assert self._label_value(_started_run_cmd(backend)) == "424242"


class TestRunCommand:
    def test_base_shape_and_runtime_binary(self, backend, container_runtime):
        cmd = _started_run_cmd(backend)
        assert cmd[:6] == [container_runtime, "run", "-d", "--init", "--name", backend._name]
        assert cmd[-3:] == ["tail", "-f", "/dev/null"]

    def test_bare_base_image_is_qualified_per_runtime(self, container_runtime, spec):
        sb = make_backend(container_runtime, base_image="my-base")
        assert _started_run_cmd(sb)[-4] == spec.bare_image("my-base")

    def test_qualified_base_image_is_used_verbatim(self, backend):
        assert _started_run_cmd(backend)[-4] == PYTHON_IMAGE

    def test_checkpoint_image_replaces_the_base_image_verbatim(self, container_runtime):
        sb = make_backend(container_runtime, checkpoint_image="agency/lifecycle-x")
        cmd = _started_run_cmd(sb)
        assert cmd[-4] == "agency/lifecycle-x"
        assert PYTHON_IMAGE not in cmd

    def test_mounts_become_volume_flags_in_order(self, container_runtime):
        sb = make_backend(
            container_runtime,
            mounts={"a": ("/host/a", "/mnt/a", "rw"), "b": ("/host/b", "/mnt/b", "ro")},
        )
        cmd = _started_run_cmd(sb)
        vols = [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "-v"]
        assert vols == ["/host/a:/mnt/a:rw", "/host/b:/mnt/b:ro"]

    def test_no_mounts_means_no_volume_flags(self, backend):
        assert "-v" not in _started_run_cmd(backend)

    def test_extra_sandbox_flags_are_passed_through_before_the_image(self, backend):
        backend._agconfig.sandbox.flags = ["--network=none", "--shm-size=1g"]
        cmd = _started_run_cmd(backend)
        assert cmd.index("--network=none") < cmd.index("--shm-size=1g") < cmd.index(PYTHON_IMAGE)

    def test_memory_limit(self, backend):
        backend._agconfig.resources.idle_memory = "2g"
        assert "--memory=2g" in _started_run_cmd(backend)

    def test_no_memory_flag_without_a_limit(self, backend):
        backend._agconfig.resources.idle_memory = None
        assert not any(a.startswith("--memory") for a in _started_run_cmd(backend))

    def test_cpu_limit_is_clamped_to_the_host_cpu_count(self, backend):
        backend._agconfig.resources.idle_cpus = 64
        assert "--cpus=4" in _started_run_cmd(backend, cfs=True, cpu_count=4)

    def test_cpu_limit_below_the_cpu_count_is_kept(self, backend):
        backend._agconfig.resources.idle_cpus = 2
        assert "--cpus=2" in _started_run_cmd(backend, cfs=True, cpu_count=8)

    def test_cpu_limit_is_dropped_when_the_kernel_cannot_enforce_a_quota(self, backend):
        backend._agconfig.resources.idle_cpus = 2
        assert not any(a.startswith("--cpus") for a in _started_run_cmd(backend, cfs=False))

    def test_cpuset_flags(self, backend):
        backend._agconfig.sandbox.cpuset_cpus = "0-3"
        backend._agconfig.sandbox.cpuset_mems = "0"
        cmd = _started_run_cmd(backend)
        assert "--cpuset-cpus=0-3" in cmd and "--cpuset-mems=0" in cmd

    def test_gpu_flags_attach_even_before_any_gpu_was_reserved(self, backend):
        """A container created by an early exec/read_file, before
        reserve_resource() ran, must already have device access: neither
        runtime can hot-attach a device later."""
        assert backend._virtual_gpu_count_requested == 0
        gf = MagicMock(return_value=["--device", "sentinel"])
        cmd = _started_run_cmd(backend, extra=[patch.object(_container, "_gpu_flags", gf)])
        gf.assert_called_once_with(backend._runtime)
        assert "--device" in cmd and "sentinel" in cmd
        # Native harness supervision must also follow children that change UID
        # (for example apt's _apt download worker). Parent/child ptrace alone
        # does not permit reading their memory after the credential change.
        assert "--cap-add=SYS_PTRACE" in cmd

    def test_cpu_only_policy_omits_gpu_flags_on_a_gpu_host(self, container_runtime):
        cfg = agconfig_cls()
        cfg.sandbox.gpu_passthrough = False
        sb = make_backend(container_runtime, agconfig=cfg)
        gf = MagicMock(return_value=["--gpus", "all"])
        cmd = _started_run_cmd(sb, extra=[patch.object(_container, "_gpu_flags", gf)])
        gf.assert_not_called()
        assert "--gpus" not in cmd and "--device" not in cmd

    def test_cpu_only_policy_rejects_a_gpu_reservation(self, container_runtime):
        cfg = agconfig_cls()
        cfg.sandbox.gpu_passthrough = False
        sb = make_backend(container_runtime, agconfig=cfg)
        sb._virtual_gpu_count_requested = 1
        with patch.object(sb, "_inspect_container_state", return_value=(False, "")):
            with pytest.raises(RuntimeError, match="CPU-only policy"):
                sb._ensure_started()

    def test_profiler_cgroup_parent_is_a_docker_only_flag(self, backend, spec):
        with patch.object(
            _container.agprof, "container_cgroup_parent", return_value="agprof-ab.slice"
        ):
            cmd = _started_run_cmd(backend)
        assert ("--cgroup-parent=agprof-ab.slice" in cmd) is spec.honours_cgroup_parent

    def test_no_cgroup_parent_without_an_active_profiler(self, backend):
        with patch.object(_container.agprof, "container_cgroup_parent", return_value=None):
            assert not any(a.startswith("--cgroup-parent") for a in _started_run_cmd(backend))

    def test_run_is_attempted_with_the_conflict_retry_wrapper(self, backend):
        _started_run_cmd(backend)  # asserts exactly one retry-wrapped run

    def test_workspace_dir_is_created_after_run(self, backend, container_runtime):
        with patch.object(backend, "_ensure_workspace_dir") as mkdir:
            _started_run_cmd(backend)
        mkdir.assert_called_once_with()


# ===========================================================================
# _ensure_started: resume / recreate / slot accounting
# ===========================================================================


class TestEnsureStarted:
    @staticmethod
    def _start(sb, *, state, start=None, run_retry=None, mkdir=None):
        """Run ``_ensure_started`` with mocks; returns (acquire, release, run_cmds)."""
        calls = []

        def handler(args, check, input, timeout):
            calls.append((args, check))
            return _Done()

        with ExitStack() as stack:
            stack.enter_context(patch.object(type(sb), "_run", _fn_run(handler)))
            stack.enter_context(patch.object(sb, "_inspect_container_state", return_value=state))
            acquire = stack.enter_context(patch.object(sb, "_acquire_runtime_slot"))
            release = stack.enter_context(patch.object(sb, "_release_runtime_slot"))
            stack.enter_context(patch.object(sb, "_snapshot_pids_started", return_value={1}))
            stack.enter_context(patch.object(sb, "_cfs_supported", return_value=False))
            stack.enter_context(patch.object(_container, "_gpu_flags", return_value=[]))
            if start is not None:
                stack.enter_context(patch.object(sb, "_start_with_quota_retry", start))
            if run_retry is not None:
                stack.enter_context(patch.object(sb, "_run_with_conflict_retry", run_retry))
            else:
                stack.enter_context(patch.object(sb, "_run_with_conflict_retry"))
            if mkdir is not None:
                stack.enter_context(patch.object(sb, "_ensure_workspace_dir", mkdir))
            error = None
            try:
                sb._ensure_started()
            except Exception as e:  # noqa: BLE001
                error = e
        return acquire, release, calls, error

    def test_no_rm_when_there_is_no_leftover_container(self, backend):
        _acq, _rel, calls, error = self._start(backend, state=(False, ""))
        assert error is None
        assert not [a for a, _ in calls if "rm" in a], calls

    @pytest.mark.parametrize("status", ["exited", "created", "paused"])
    def test_a_non_running_container_is_resumed_in_place_not_recreated(
        self, backend, container_runtime, status
    ):
        """Hibernating (stop()ed, not removed) or stuck in 'created': the
        container under this exact name is unambiguously this backend's own
        (names embed the per-process run id), so it is `start`ed -- never
        removed and never `run` again. `_run_with_conflict_retry` is left
        unmocked-by-contract: it must not be reached."""
        retry = MagicMock(side_effect=AssertionError("must not run a new container"))
        acquire, release, calls, error = self._start(
            backend, state=(False, status), run_retry=retry
        )
        assert error is None
        assert not [a for a, _ in calls if "rm" in a]
        starts = [a for a, _ in calls if "start" in a]
        assert starts == [[container_runtime, "start", backend._name]]
        acquire.assert_called_once_with()
        release.assert_not_called()

    def test_resume_makes_sure_the_workspace_exists_and_captures_a_baseline(self, backend):
        mkdir = MagicMock()
        _a, _r, _c, error = self._start(backend, state=(False, "exited"), mkdir=mkdir)
        assert error is None
        mkdir.assert_called_once_with()
        assert backend._baseline_pids == {1}

    def test_resume_failure_releases_the_slot_it_took(self, backend):
        start = MagicMock(side_effect=RuntimeError("start failed"))
        acquire, release, _calls, error = self._start(backend, state=(False, "exited"), start=start)
        assert isinstance(error, RuntimeError) and "start failed" in str(error)
        acquire.assert_called_once_with()
        release.assert_called_once_with()

    def test_running_container_takes_no_second_slot(self, backend):
        acquire, release, calls, error = self._start(backend, state=(True, "running"))
        assert error is None
        acquire.assert_not_called()
        release.assert_not_called()
        assert backend._baseline_pids == {1}
        assert not [a for a, _ in calls if "start" in a]

    def test_baseline_is_captured_once(self, backend):
        backend._baseline_pids = {99}
        self._start(backend, state=(True, "running"))
        assert backend._baseline_pids == {99}

    def test_fresh_run_keeps_its_slot_on_success(self, backend):
        acquire, release, _calls, error = self._start(backend, state=(False, ""))
        assert error is None
        acquire.assert_called_once_with()
        release.assert_not_called()

    def test_failed_run_releases_the_slot(self, backend):
        retry = MagicMock(side_effect=RuntimeError("no such image"))
        acquire, release, _c, error = self._start(backend, state=(False, ""), run_retry=retry)
        assert isinstance(error, RuntimeError)
        acquire.assert_called_once_with()
        release.assert_called_once_with()

    def test_failed_workspace_mkdir_after_run_releases_the_slot(self, backend):
        mkdir = MagicMock(side_effect=RuntimeError("mkdir failed"))
        acquire, release, _c, error = self._start(backend, state=(False, ""), mkdir=mkdir)
        assert isinstance(error, RuntimeError)
        release.assert_called_once_with()

    def test_losing_a_start_race_releases_our_slot_and_adopts_the_container(self, backend):
        """Another process started the container while this one retried: it
        owns the slot, so ours is released and start-up still succeeds."""
        retry = MagicMock(side_effect=_container._ContainerAlreadyRunning())
        acquire, release, _c, error = self._start(backend, state=(False, ""), run_retry=retry)
        assert error is None
        release.assert_called_once_with()
        assert backend._baseline_pids == {1}


# ===========================================================================
# _run_with_conflict_retry / _start_with_quota_retry
# ===========================================================================


class _FakeResult:
    def __init__(self, returncode: int, stderr: bytes = b""):
        self.returncode = returncode
        self.stderr = stderr


class TestRunWithConflictRetry:
    def _argv(self, rt):
        return [rt, "run", "--name", "x"]

    def test_quota_branch_waits_then_cleans_up_stale_container_and_retries(
        self, backend, container_runtime
    ):
        results = iter([_FakeResult(1, b"quota exhausted"), _FakeResult(0)])
        with patch.object(backend, "_run", side_effect=lambda *a, **k: next(results)):
            with patch.object(backend, "_is_quota_exhaustion_error", return_value=True):
                with patch.object(backend, "_wait_for_quota_slot") as wait_mock:
                    with patch.object(backend, "_container_status", return_value="created"):
                        with patch.object(backend, "_rm_container") as rm_mock:
                            backend._run_with_conflict_retry(self._argv(container_runtime), "x")
        wait_mock.assert_called_once()
        rm_mock.assert_called_once_with("x")

    def test_quota_branch_does_not_remove_when_nothing_was_created(
        self, backend, container_runtime
    ):
        results = iter([_FakeResult(1, b"quota exhausted"), _FakeResult(0)])
        with patch.object(backend, "_run", side_effect=lambda *a, **k: next(results)):
            with patch.object(backend, "_is_quota_exhaustion_error", return_value=True):
                with patch.object(backend, "_wait_for_quota_slot"):
                    with patch.object(backend, "_container_status", return_value=""):
                        with patch.object(backend, "_rm_container") as rm_mock:
                            backend._run_with_conflict_retry(self._argv(container_runtime), "x")
        rm_mock.assert_not_called()

    def test_persistent_quota_exhaustion_gives_up_with_diagnostics(
        self, backend, container_runtime
    ):
        result = _FakeResult(1, b"unable to create session key: disk quota exceeded")
        with patch.object(backend, "_run", return_value=result) as run:
            with patch.object(backend, "_wait_for_quota_slot"):
                with patch.object(backend, "_container_status", return_value=""):
                    with patch.object(backend, "_quota_diagnostics", return_value="[diag: 9/9]"):
                        with pytest.raises(RuntimeError) as exc_info:
                            backend._run_with_conflict_retry(self._argv(container_runtime), "x")
        assert run.call_count == backend._agconfig.sandbox.conflict_retry_max_attempts
        msg = str(exc_info.value)
        assert f"{container_runtime} run --name x failed after retries" in msg
        assert "[diag: 9/9]" in msg and "session key" in msg

    def test_conflict_branch_raises_already_running_when_container_is_up(
        self, backend, container_runtime
    ):
        result = _FakeResult(1, b"Conflict. The container name is already in use")
        with patch.object(backend, "_run", return_value=result):
            with patch.object(backend, "_container_running", return_value=True):
                with pytest.raises(_container._ContainerAlreadyRunning):
                    backend._run_with_conflict_retry(self._argv(container_runtime), "x")

    def test_conflict_branch_removes_stale_container_and_waits_for_quota_each_attempt(
        self, backend, container_runtime
    ):
        result = _FakeResult(1, b"Conflict. The container name is already in use")
        with patch.object(backend, "_run", return_value=result):
            with patch.object(backend, "_container_running", return_value=False):
                with patch.object(backend, "_container_status", return_value=""):
                    with patch.object(backend, "_rm_container") as rm_mock:
                        with patch.object(backend, "_wait_for_quota_slot") as wait_mock:
                            with patch("time.sleep"):
                                with pytest.raises(RuntimeError, match="failed after retries"):
                                    backend._run_with_conflict_retry(
                                        self._argv(container_runtime), "x"
                                    )
        attempts = backend._agconfig.sandbox.conflict_retry_max_attempts
        assert rm_mock.call_count == attempts
        assert wait_mock.call_count == attempts

    def test_conflict_cleared_after_one_retry_succeeds(self, backend, container_runtime):
        results = iter([_FakeResult(1, b"Conflict: already in use"), _FakeResult(0)])
        with patch.object(backend, "_run", side_effect=lambda *a, **k: next(results)):
            with patch.object(backend, "_container_running", return_value=False):
                with patch.object(backend, "_container_status", return_value=""):
                    with patch.object(backend, "_rm_container") as rm_mock:
                        with patch.object(backend, "_wait_for_quota_slot"):
                            with patch("time.sleep"):
                                backend._run_with_conflict_retry(self._argv(container_runtime), "x")
        rm_mock.assert_called_once_with("x")

    def test_final_error_names_the_live_runtime_and_includes_diagnostics(
        self, backend, container_runtime
    ):
        result = _FakeResult(1, b"Conflict. The container name is already in use")
        with patch.object(backend, "_run", return_value=result):
            with patch.object(backend, "_container_running", return_value=False):
                with patch.object(backend, "_container_status", return_value=""):
                    with patch.object(backend, "_rm_container"):
                        with patch.object(backend, "_wait_for_quota_slot"):
                            with patch.object(
                                backend, "_quota_diagnostics", return_value="[diag: 1/2]"
                            ):
                                with patch("time.sleep"):
                                    with pytest.raises(RuntimeError) as exc_info:
                                        backend._run_with_conflict_retry(
                                            self._argv(container_runtime), "x"
                                        )
        msg = str(exc_info.value)
        assert f"{container_runtime} run --name x failed after retries" in msg
        assert "[diag: 1/2]" in msg and "already in use" in msg

    def test_non_conflict_non_quota_failure_raises_immediately_without_retry(
        self, backend, container_runtime
    ):
        result = _FakeResult(1, b"no such image: docker.io/library/python:3.12-slim")
        run_mock = MagicMock(return_value=result)
        with patch.object(backend, "_run", run_mock):
            with pytest.raises(RuntimeError, match="no such image"):
                backend._run_with_conflict_retry(self._argv(container_runtime), "x")
        assert run_mock.call_count == 1


class TestStartWithQuotaRetry:
    """Resuming a hibernating container re-takes a keyring slot exactly like
    `run` does, so `start` retries on quota exhaustion too."""

    def test_success_on_first_attempt(self, backend, container_runtime):
        with patch.object(backend, "_run", return_value=_FakeResult(0)) as run:
            backend._start_with_quota_retry("x")
        assert run.call_args.args[0] == [container_runtime, "start", "x"]

    def test_quota_exhaustion_waits_then_retries(self, backend):
        results = iter([_FakeResult(1, b"disk quota exceeded keyring"), _FakeResult(0)])
        with patch.object(backend, "_run", side_effect=lambda *a, **k: next(results)):
            with patch.object(backend, "_wait_for_quota_slot") as wait_mock:
                with patch.object(_container.time, "sleep"):
                    backend._start_with_quota_retry("x")
        wait_mock.assert_called_once()

    def test_quota_never_clearing_gives_up_with_diagnostics(self, backend, container_runtime):
        quota = _FakeResult(1, b"disk quota exceeded keyring")
        with patch.object(backend, "_run", return_value=quota) as run:
            with patch.object(backend, "_wait_for_quota_slot"):
                with patch.object(backend, "_quota_diagnostics", return_value="[diag]"):
                    with patch.object(_container.time, "sleep"):
                        with pytest.raises(RuntimeError) as exc_info:
                            backend._start_with_quota_retry("x")
        assert run.call_count == backend._agconfig.sandbox.conflict_retry_max_attempts
        msg = str(exc_info.value)
        assert f"{container_runtime} start x failed after retries (runtime quota)" in msg
        assert "[diag]" in msg

    def test_other_failures_raise_immediately(self, backend, container_runtime):
        failure = _FakeResult(125, b"no such container")
        with patch.object(backend, "_run", return_value=failure) as run:
            with pytest.raises(
                RuntimeError,
                match=rf"{container_runtime} start x failed \(exit 125\): no such container",
            ):
                backend._start_with_quota_retry("x")
        assert run.call_count == 1


# ===========================================================================
# exec argv
# ===========================================================================


class TestContainerExec:
    @staticmethod
    def _capture(backend):
        calls = []

        def handler(args, check, input, timeout):
            calls.append({"args": args, "check": check, "input": input, "timeout": timeout})
            return _Done(stdout=b"out", stderr=b"err")

        return calls, patch.object(type(backend), "_run", _fn_run(handler))

    def test_exec_argv(self, backend, container_runtime):
        calls, patcher = self._capture(backend)
        with patcher, patch.object(backend, "_ensure_started"):
            out, rc = backend._container_exec("echo hi", workdir="/w", timeout=9)
        assert (out, rc) == ("outerr", 0)
        assert calls[0]["args"] == [
            container_runtime, "exec", "-w", "/w", backend._name, "bash", "-c", "echo hi"
        ]  # fmt: skip
        assert calls[0]["timeout"] == 9

    def test_stdin_adds_interactive_flag_and_input(self, backend, container_runtime):
        calls, patcher = self._capture(backend)
        with patcher, patch.object(backend, "_ensure_started"):
            backend._container_exec("cat", stdin=b"data", shell="sh")
        assert calls[0]["args"][:3] == [container_runtime, "exec", "-i"]
        assert calls[0]["args"][-3:-1] == ["sh", "-c"]
        assert calls[0]["input"] == b"data"

    def test_exec_starts_the_container_first(self, backend):
        calls, patcher = self._capture(backend)
        order = []
        with (
            patcher,
            patch.object(backend, "_ensure_started", side_effect=lambda: order.append("start")),
        ):
            backend._container_exec("true")
        assert order == ["start"] and calls

    def test_exec_started_does_not_start_the_container(self, backend):
        calls, patcher = self._capture(backend)
        with patcher, patch.object(backend, "_ensure_started") as ensure:
            backend._container_exec_started("true")
        ensure.assert_not_called()

    def test_timeout_is_reported_as_minus_one(self, backend):
        with patch.object(type(backend), "_run", side_effect=subprocess.TimeoutExpired("x", 3)):
            assert backend._container_exec_started("sleep 9", timeout=3) == (
                "Command timed out after 3s",
                -1,
            )

    def test_launch_failure_is_reported_as_minus_one(self, backend):
        with patch.object(type(backend), "_run", side_effect=RuntimeError("no such container")):
            assert backend._container_exec_started("true") == ("no such container", -1)

    def test_detached_exec_argv(self, backend, container_runtime):
        calls, patcher = self._capture(backend)
        with patcher, patch.object(backend, "_ensure_started"):
            backend._container_exec_detached("serve", workdir="/w", shell="sh")
        assert calls[0]["args"] == [
            container_runtime, "exec", "-d", "-w", "/w", backend._name, "sh", "-c", "serve"
        ]  # fmt: skip
        assert calls[0]["check"] is True

    def test_detached_exec_failure_raises(self, backend):
        with patch.object(backend, "_ensure_started"):
            with patch.object(type(backend), "_run", side_effect=RuntimeError("gone")):
                with pytest.raises(RuntimeError, match="gone"):
                    backend._container_exec_detached("serve")

    def test_detached_exec_under_fast_resume_double_forks_instead_of_exec_d(self, backend):
        backend._agconfig.sandbox.checkpoint_fast_resume = True
        calls, patcher = self._capture(backend)
        with patcher, patch.object(backend, "_ensure_started"):
            backend._container_exec_detached("serve it")
        args = calls[0]["args"]
        assert "-d" not in args
        assert args[-1].startswith("nohup bash -c ") and args[-1].endswith("&")


# ===========================================================================
# update_limits
# ===========================================================================


class TestUpdateLimits:
    def _cmds(self, backend, *, running=True, cfs=True, cpu_count=None, **limits):
        commands = []
        with ExitStack() as stack:
            stack.enter_context(patch.object(backend, "_container_running", return_value=running))
            stack.enter_context(patch.object(backend, "_cfs_supported", return_value=cfs))
            if cpu_count is not None:
                stack.enter_context(
                    patch.object(_container.os, "cpu_count", return_value=cpu_count)
                )
            stack.enter_context(
                patch.object(backend, "_run", side_effect=lambda cmd, **kw: commands.append(cmd))
            )
            backend.update_limits(**limits)
        return commands

    def test_memory_swap_is_pinned_only_where_the_runtime_supports_it(self, backend, spec):
        (cmd,) = self._cmds(backend, cpus=2.0, memory="8192m")
        assert "--memory=8192m" in cmd
        assert ("--memory-swap=8192m" in cmd) is spec.pins_memory_swap

    def test_memory_only_command(self, backend, container_runtime):
        (cmd,) = self._cmds(backend, memory="1g")
        assert cmd[:2] == [container_runtime, "update"]
        assert cmd[-1] == backend._name
        assert not any(a.startswith("--cpus") for a in cmd)

    def test_cpus_only_sets_no_memory_flags(self, backend):
        (cmd,) = self._cmds(backend, cpus=2.0)
        assert "--cpus=2.0" in cmd
        assert not any(arg.startswith("--memory") for arg in cmd)

    def test_cpus_are_clamped_to_the_host_cpu_count(self, backend):
        (cmd,) = self._cmds(backend, cpus=64.0, cpu_count=4)
        assert "--cpus=4" in cmd

    def test_cpus_are_dropped_when_no_quota_can_be_enforced(self, backend):
        assert self._cmds(backend, cfs=False, cpus=2.0) == []

    def test_not_running_is_a_noop(self, backend):
        assert self._cmds(backend, running=False, cpus=2.0, memory="1g") == []

    def test_nothing_to_update_is_a_noop(self, backend):
        assert self._cmds(backend) == []


# ===========================================================================
# stop / rm_container / destroy: slot and GPU release
# ===========================================================================


class TestStopSlotAndGpu:
    @staticmethod
    def _reserved(backend, gpu_ids=None):
        released = []
        backend._virtual_gpu_count_requested = 1
        backend._real_gpu_acquire_fn = lambda count: [0]
        backend._real_gpu_release_fn = lambda ids: released.append(list(ids))
        backend._real_gpu_ids = list(gpu_ids or [])
        return released

    def test_stop_argv_clears_pid_tracking_and_releases_the_slot_once(
        self, backend, container_runtime
    ):
        backend._watched_pids = {5: 0.0}
        backend._daemon_pids = {6}
        backend._ptrace_managed_pids = {7}
        backend._baseline_pids = {1}
        with patch.object(backend, "_container_running", side_effect=[True, False]):
            with patch.object(backend, "_run") as run:
                with patch.object(backend, "_release_runtime_slot") as release:
                    backend._stop_image()
        grace = str(backend._agconfig.sandbox.docker_stop_grace_s)
        assert run.call_args.args[0] == [container_runtime, "stop", "-t", grace, backend._name]
        assert run.call_args.kwargs["check"] is True
        release.assert_called_once_with()
        assert backend._watched_pids == {} and backend._daemon_pids == set()
        assert backend._ptrace_managed_pids == set() and backend._baseline_pids is None

    def test_stop_of_an_already_stopped_container_runs_nothing_and_keeps_the_slot_count(
        self, backend
    ):
        with patch.object(backend, "_container_running", return_value=False):
            with patch.object(backend, "_run") as run:
                with patch.object(backend, "_release_runtime_slot") as release:
                    backend._stop_image()
        run.assert_not_called()
        release.assert_not_called()

    def test_failed_stop_that_leaves_the_container_running_releases_nothing(self, backend):
        released = self._reserved(backend, [0])
        with patch.object(backend, "_container_running", side_effect=[True, True]):
            with patch.object(backend, "_run", side_effect=RuntimeError("stop failed")):
                with patch.object(backend, "_release_runtime_slot") as release:
                    with pytest.raises(RuntimeError, match="stop failed"):
                        backend._stop_image()
        release.assert_not_called()
        assert released == [] and backend._real_gpu_ids == [0]

    def test_stop_error_is_still_raised_when_the_container_turns_out_to_be_gone(self, backend):
        with patch.object(backend, "_container_running", side_effect=[True, False]):
            with patch.object(backend, "_run", side_effect=RuntimeError("client error")):
                with patch.object(backend, "_release_runtime_slot") as release:
                    with pytest.raises(RuntimeError, match="client error"):
                        backend._stop_image()
        release.assert_called_once_with()

    def test_stopping_the_container_gives_the_held_gpus_back(self, backend):
        released = self._reserved(backend)
        backend._hold_real_gpus()
        with patch.object(backend, "_container_running", side_effect=[True, False]):
            with patch.object(backend, "_run"), patch.object(backend, "_release_runtime_slot"):
                backend._stop_image()
        assert released == [[0]]
        assert backend._real_gpu_ids == []

    def test_an_already_stopped_container_still_gives_the_held_gpus_back(self, backend):
        released = self._reserved(backend)
        backend._hold_real_gpus()
        with patch.object(backend, "_container_running", return_value=False):
            with (
                patch.object(backend, "_run") as run,
                patch.object(backend, "_release_runtime_slot") as release_slot,
            ):
                backend._stop_image()
        assert released == [[0]]
        run.assert_not_called()
        release_slot.assert_not_called()


class TestDestroySlot:
    """destroy() / rm_container() runtime-slot accounting (the slot is only
    released once the container is CONFIRMED gone, and never twice)."""

    @staticmethod
    def _failing_rm(args, check, input, timeout):
        if "rm" in args:
            raise RuntimeError("rm exploded")
        return _Done()

    def _destroy(self, backend, running_answers, handler=None):
        """destroy() with `_container_running` answering from *running_answers*
        (then False); returns (released_count, error)."""
        released = []
        answers = list(running_answers)
        error = None
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(type(backend), "_run", _fn_run(handler or self._failing_rm))
            )
            stack.enter_context(
                patch.object(
                    backend,
                    "_container_running",
                    side_effect=lambda: answers.pop(0) if answers else False,
                )
            )
            stack.enter_context(patch.object(backend, "_container_status", return_value="running"))
            stack.enter_context(
                patch.object(
                    _container._container_semaphore,
                    "release",
                    side_effect=lambda: released.append(1),
                )
            )
            stack.enter_context(patch.object(_container.time, "sleep"))
            try:
                backend.destroy()
            except RuntimeError as e:
                error = e
        return len(released), error

    def test_slot_is_not_released_while_the_container_still_runs_after_a_failed_rm(self, backend):
        """Releasing would over-credit the semaphore (one more container than
        the keyring quota allows) for a container that never went away."""
        released, error = self._destroy(backend, [True] * 10)
        assert released == 0
        assert error is not None and "rm exploded" in str(error)

    def test_slot_is_released_when_the_container_is_confirmed_gone_despite_an_rm_error(
        self, backend
    ):
        released, error = self._destroy(backend, [True, False])
        assert released == 1
        assert error is not None and "rm exploded" in str(error)

    def test_destroy_skips_rm_when_the_container_is_absent(self, backend):
        calls = []

        def handler(args, check, input, timeout):
            calls.append(args)
            return _Done()

        with patch.object(type(backend), "_run", _fn_run(handler)):
            with patch.object(backend, "_container_running", return_value=False):
                with patch.object(backend, "_container_status", return_value=""):
                    backend.destroy()
        assert not [a for a in calls if "rm" in a and "rmi" not in a], calls

    def test_destroy_removes_the_checkpoint_image_and_marks_the_backend_destroyed(
        self, container_runtime
    ):
        sb = make_backend(container_runtime, checkpoint_image="agency/lifecycle-x")
        with patch.object(sb, "rm_container"):
            with patch.object(sb, "_rmi") as rmi:
                sb.destroy()
        rmi.assert_called_once_with("agency/lifecycle-x", force=True)
        assert sb._checkpoint_image is None and sb._destroyed is True

    def test_destroy_attempts_image_cleanup_even_when_rm_fails_then_raises(self, container_runtime):
        sb = make_backend(container_runtime, checkpoint_image="agency/lifecycle-x")
        with patch.object(sb, "rm_container", side_effect=RuntimeError("rm failed")):
            with patch.object(sb, "_rmi") as rmi:
                with pytest.raises(RuntimeError, match="rm failed"):
                    sb.destroy()
        rmi.assert_called_once()
        assert sb._destroyed is False  # stays retryable

    def test_image_cleanup_failure_is_best_effort(self, container_runtime, capsys):
        sb = make_backend(container_runtime, checkpoint_image="agency/lifecycle-x")
        with patch.object(sb, "rm_container"):
            with patch.object(sb, "_rmi", side_effect=RuntimeError("in use")):
                sb.destroy()  # must not raise
        assert "checkpoint image cleanup failed" in capsys.readouterr().out
        assert sb._destroyed is True

    def test_destroy_is_idempotent(self, backend):
        with patch.object(backend, "rm_container") as rm:
            backend.destroy()
            backend.destroy()
        rm.assert_called_once_with()

    def test_destroy_signals_tracked_pids_before_removing_the_container(self, backend):
        backend._watched_pids = {11: 0.0, 12: 0.0}
        order = []
        with patch.object(
            backend, "_container_exec", side_effect=lambda cmd, **kw: order.append(cmd) or ("", 0)
        ):
            with patch.object(backend, "rm_container", side_effect=lambda: order.append("rm")):
                backend.destroy()
        assert order[0].startswith("kill 11 12") and order[1] == "rm"

    def test_failure_to_signal_tracked_pids_does_not_stop_destroy(self, backend, capsys):
        backend._watched_pids = {11: 0.0}
        with patch.object(backend, "_container_exec", side_effect=RuntimeError("gone")):
            with patch.object(backend, "rm_container") as rm:
                backend.destroy()
        rm.assert_called_once_with()
        assert "failed to kill PIDs" in capsys.readouterr().out

    def test_private_storage_cleanup_order_differs_per_runtime(self, backend, spec):
        """docker tears the private storage down first and keeps it available
        (prepared) through container removal; podman removes the container
        before the storage."""
        order = []
        storage = MagicMock(prepared=True, dataset="tank/x/writable")
        storage.destroy.side_effect = lambda: order.append("storage.destroy")

        def remove():
            assert storage.prepared is True, "private runtime must stay available through rm"
            order.append("rm_container")

        backend._checkpoint_storage = storage
        with patch.object(backend, "rm_container", side_effect=remove):
            backend.destroy()
        assert order == list(spec.destroy_order)
        assert backend._destroyed is True
        if spec.name == "docker":
            storage.destroy.assert_called_once_with()
            assert storage.prepared is False and storage.dataset is None


class TestGpuReleaseGating:
    """The real GPU lease is released only once the container is CONFIRMED
    torn down -- releasing it while the container may still be running would
    let another sandbox acquire the same physical GPU concurrently. stop()
    (hibernate) is covered in TestStopSlotAndGpu; here rm_container()/destroy()."""

    @staticmethod
    def _lease_gpu(sb, gpu_id=3):
        released = []
        sb._virtual_gpu_count_requested = 1
        sb._real_gpu_ids = [gpu_id]
        sb._real_gpu_release_fn = lambda ids: released.extend(ids)
        return released

    @staticmethod
    def _ok(args, check, input, timeout):
        return _Done()

    @staticmethod
    def _rm_explodes(args, check, input, timeout):
        if "rm" in args:
            raise RuntimeError("rm exploded")
        return _Done()

    @staticmethod
    def _answers(values):
        values = list(values)
        return lambda: values.pop(0) if values else False

    def test_rm_container_releases_gpu_when_already_confirmed_gone(self, backend):
        released = self._lease_gpu(backend)
        with patch.object(type(backend), "_run"):
            with patch.object(backend, "_container_status", return_value=""):
                backend.rm_container()
        assert released == [3] and backend._real_gpu_ids == []

    def test_rm_container_releases_gpu_via_main_teardown_path_after_successful_rm(self, backend):
        released = self._lease_gpu(backend)
        with patch.object(type(backend), "_run", _fn_run(self._ok)):
            with patch.object(backend, "_container_status", return_value="running"):
                with patch.object(
                    backend, "_container_running", side_effect=self._answers([False, False])
                ):
                    backend.rm_container()
        assert released == [3] and backend._real_gpu_ids == []

    def test_rm_container_does_not_release_gpu_when_rm_fails_and_container_still_running(
        self, backend
    ):
        released = self._lease_gpu(backend)
        with patch.object(type(backend), "_run", _fn_run(self._rm_explodes)):
            with patch.object(backend, "_container_status", return_value="running"):
                with patch.object(backend, "_container_running", return_value=True):
                    with patch.object(_container.time, "sleep"):
                        with pytest.raises(RuntimeError, match="rm exploded"):
                            backend.rm_container()
        assert released == [] and backend._real_gpu_ids == [3]

    def test_rm_container_releases_gpu_when_rm_failed_but_container_is_gone(self, backend):
        released = self._lease_gpu(backend)
        with patch.object(type(backend), "_run", _fn_run(self._rm_explodes)):
            with patch.object(backend, "_container_status", return_value="running"):
                with patch.object(
                    backend, "_container_running", side_effect=self._answers([True, False, False])
                ):
                    with patch.object(_container.time, "sleep"):
                        with pytest.raises(RuntimeError, match="rm exploded"):
                            backend.rm_container()
        assert released == [3] and backend._real_gpu_ids == []

    def test_rm_container_retries_the_removal_before_giving_up(self, backend):
        attempts = []

        def flaky(args, check, input, timeout):
            if "rm" in args:
                attempts.append(1)
                if len(attempts) == 1:
                    raise RuntimeError("transient")
            return _Done()

        with patch.object(type(backend), "_run", _fn_run(flaky)):
            with patch.object(backend, "_container_status", return_value="running"):
                with patch.object(
                    backend, "_container_running", side_effect=self._answers([True, False, False])
                ):
                    with patch.object(_container.time, "sleep"):
                        backend.rm_container()
        assert len(attempts) == 2

    def test_rm_container_raises_after_every_attempt_failed(self, backend):
        attempts = []

        def always(args, check, input, timeout):
            if "rm" in args:
                attempts.append(1)
                raise RuntimeError("persistent")
            return _Done()

        with patch.object(type(backend), "_run", _fn_run(always)):
            with patch.object(backend, "_container_status", return_value="running"):
                with patch.object(backend, "_container_running", return_value=True):
                    with patch.object(_container.time, "sleep"):
                        with pytest.raises(RuntimeError, match="persistent"):
                            backend.rm_container()
        assert len(attempts) == backend._agconfig.sandbox.rm_retry_attempts

    def test_rm_container_on_an_already_hibernating_container_does_not_release_the_slot(
        self, backend
    ):
        """stop() already released the slot; rm_container() must not credit
        the (non-bounded) semaphore a second time."""
        with patch.object(type(backend), "_run", _fn_run(self._ok)):
            with patch.object(backend, "_container_status", return_value="exited"):
                with patch.object(backend, "_container_running", return_value=False):
                    with patch.object(backend, "_release_runtime_slot") as release:
                        backend.rm_container()
        release.assert_not_called()

    def test_rm_container_of_a_running_container_releases_the_slot_once(self, backend):
        with patch.object(type(backend), "_run", _fn_run(self._ok)):
            with patch.object(backend, "_container_status", return_value="running"):
                with patch.object(
                    backend, "_container_running", side_effect=self._answers([True, False, False])
                ):
                    with patch.object(backend, "_release_runtime_slot") as release:
                        backend.rm_container()
        release.assert_called_once_with()

    def test_rm_container_clears_pid_tracking(self, backend):
        backend._watched_pids = {1: 0.0}
        backend._daemon_pids = {2}
        backend._ptrace_managed_pids = {3}
        backend._baseline_pids = {4}
        with patch.object(type(backend), "_run", _fn_run(self._ok)):
            with patch.object(backend, "_container_status", return_value="running"):
                with patch.object(
                    backend, "_container_running", side_effect=self._answers([True, False, False])
                ):
                    backend.rm_container()
        assert backend._watched_pids == {} and backend._daemon_pids == set()
        assert backend._ptrace_managed_pids == set() and backend._baseline_pids is None

    def test_destroy_releases_gpu_when_container_confirmed_gone_despite_rm_error(self, backend):
        released = self._lease_gpu(backend)
        with patch.object(type(backend), "_run", _fn_run(self._rm_explodes)):
            with patch.object(
                backend, "_container_running", side_effect=self._answers([True, False])
            ):
                with patch.object(backend, "_container_status", return_value="running"):
                    with patch.object(_container.time, "sleep"):
                        with pytest.raises(RuntimeError, match="rm exploded"):
                            backend.destroy()
        assert released == [3] and backend._real_gpu_ids == []

    def test_destroy_does_not_release_gpu_when_container_still_running_after_rm_failure(
        self, backend
    ):
        released = self._lease_gpu(backend)
        with patch.object(type(backend), "_run", _fn_run(self._rm_explodes)):
            with patch.object(backend, "_container_running", return_value=True):
                with patch.object(backend, "_container_status", return_value="running"):
                    with patch.object(_container.time, "sleep"):
                        with pytest.raises(RuntimeError, match="rm exploded"):
                            backend.destroy()
        assert released == [] and backend._real_gpu_ids == [3]

    def test_gpu_released_exactly_once_across_rm_container_then_destroy(self, backend):
        released = self._lease_gpu(backend)
        with patch.object(type(backend), "_run"):
            with patch.object(backend, "_container_status", return_value=""):
                with patch.object(backend, "_container_running", return_value=False):
                    backend.rm_container()
                    backend.destroy()
        assert released == [3], "GPU must be released exactly once, not once per call"


# ===========================================================================
# commit(): dangling images and squashing
# ===========================================================================


class TestDanglingImageEagerCleanup:
    """commit() reclaims the image that previously held the lifecycle tag."""

    @staticmethod
    def _handler(calls, *, old_id=b"sha256:deadbeef0000", old_rc=0, rmi_error=None, order=None):
        def handler(args, check, input, timeout):
            calls.append(args)
            if LAYERS in args:
                return _Done(stdout=b'["sha256:layer0"]')
            if IMAGE_ID in args:
                return _Done(stdout=old_id if old_rc == 0 else b"", returncode=old_rc)
            if order is not None:
                for verb in ("rm", "ps", "rmi"):
                    if verb in args:
                        order.append(verb)
            if rmi_error is not None and "rmi" in args:
                raise rmi_error
            return _Done()

        return handler

    def test_squash_commit_deletes_the_old_image(self, backend):
        """Only a squash cycle can delete it: a plain commit's result is a
        child of the old image and the runtime refuses. checkpoint_squash_max_depth
        is patched to 1 to force the squash."""
        calls = []
        with _committing(backend, self._handler(calls), max_depth=1):
            backend.commit()
        rmis = [a for a in calls if "rmi" in a]
        assert rmis, "expected an rmi for the old image"
        assert any("sha256:deadbeef0000" in " ".join(a) for a in rmis)
        assert all(a[0] == backend._runtime for a in calls)

    def test_squash_commit_skips_rmi_when_no_old_image(self, backend):
        calls = []
        with _committing(backend, self._handler(calls, old_rc=1), max_depth=1):
            backend.commit()
        assert not [a for a in calls if "rmi" in a]

    def test_plain_commit_with_no_previous_image_skips_ps_and_rmi(self, backend):
        """Step 0 still looks up the tag's current image every cycle, but with
        no previous image there is nothing to check or delete."""
        calls = []
        with _committing(backend, self._handler(calls, old_rc=1)):
            backend.commit()
        assert [c for c in calls if IMAGE_ID in c], "step 0 must still look the tag up"
        assert not any("ps" in c for c in calls), calls
        assert not any("rmi" in c for c in calls), calls

    def test_rmi_failure_is_best_effort(self, backend, capsys):
        """A race with another agent's rmi (or a fork still using the image)
        leaves a dangling image rather than crashing commit(), which runs once
        per skill."""
        calls = []
        handler = self._handler(
            calls, old_id=b"sha256:cafebabe1234", rmi_error=RuntimeError("image in use")
        )
        with _committing(backend, handler, max_depth=1):
            backend.commit()  # must not raise
        err = capsys.readouterr().err
        assert "WARNING" in err and "sha256:cafebabe1234" in err

    def test_ancestor_checks_run_via_ps_then_rmi_and_never_rm(self, backend):
        """Twice per squash cycle: step 0's previous-tag cleanup, then the
        squash's own superseded-commit cleanup. commit() never removes the
        container, so no `rm` appears."""
        calls, order = [], []
        with _committing(backend, self._handler(calls, order=order), max_depth=1):
            backend.commit()
        assert "rm" not in order, order
        assert order == ["ps", "rmi", "ps", "rmi"], order

    def test_an_image_still_in_use_by_a_container_is_not_removed(self, backend):
        calls = []

        def handler(args, check, input, timeout):
            calls.append(args)
            if LAYERS in args:
                return _Done(stdout=b'["sha256:layer0"]')
            if IMAGE_ID in args:
                return _Done(stdout=b"sha256:deadbeef0000")
            if "ps" in args:
                return _Done(stdout=b"some-container-id\n")
            return _Done()

        with _committing(backend, handler):
            backend.commit()
        assert not [a for a in calls if "rmi" in a]

    def test_real_plain_commit_cycle_cleans_up_previous_sibling(self, container_kind):
        """Two plain commit() calls on the same never-recreated container each
        re-snapshot its writable layer as a fresh base+1 image (siblings, never
        a chain), so checkpoint 1's image is deleted once the tag moves on."""
        sb = make_sandbox(container_kind)
        sb._backend._base_image = PYTHON_IMAGE
        try:
            sb.exec("echo one")
            sb.commit()
            first = _cli(
                container_kind, "inspect", "--format={{.Id}}", sb._backend._lifecycle_tag()
            ).stdout.strip()
            assert first
            sb.exec("echo two")
            sb.commit()
            assert _cli(container_kind, "image", "inspect", first).returncode != 0, (
                "checkpoint 1's image is a sibling of checkpoint 2's: commit() must clean it up"
            )
        finally:
            sb.destroy()

    def test_real_squash_cycle_reclaims_its_own_superseded_commit(self, container_kind):
        sb = make_sandbox(container_kind)
        sb._backend._base_image = PYTHON_IMAGE

        def dangling():
            r = _cli(container_kind, "images", "-f", "dangling=true", "-q")
            return {ln.strip() for ln in r.stdout.splitlines() if ln.strip()}

        base_layers = _cli(
            container_kind, "inspect", "--format={{json .RootFS.Layers}}", PYTHON_IMAGE
        ).stdout
        try:
            sb.exec("echo one")
            before = dangling()
            with patch.object(sb._backend._agconfig.sandbox, "checkpoint_squash_max_depth", 1):
                sb.commit()
            assert dangling() - before == set(), "squashing left its superseded commit dangling"
            # Either squash path is a correct outcome: flattened (1) or the
            # fast path's base layers + 1 merged layer.
            layers = int(
                _cli(
                    container_kind,
                    "inspect",
                    "--format={{len .RootFS.Layers}}",
                    sb._backend._lifecycle_tag(),
                ).stdout.strip()
            )
            assert layers in (1, len(json.loads(base_layers)) + 1)
        finally:
            sb.destroy()

    def test_real_repeated_commits_leave_no_dangling_images(self, container_kind):
        name = f"test-eager-{container_kind}-{uuid.uuid4().hex[:8]}"
        tag = f"agency/lifecycle-{name}"

        def dangling():
            r = _cli(container_kind, "images", "-f", "dangling=true", "-q")
            return {ln.strip() for ln in r.stdout.splitlines() if ln.strip()}

        _cli(
            container_kind,
            "run",
            "-d",
            "--name",
            name,
            PYTHON_IMAGE,
            "tail",
            "-f",
            "/dev/null",
            check=True,
        )
        try:
            before = dangling()
            for _ in range(3):
                old = _cli(container_kind, "inspect", "--format={{.Id}}", tag)
                old_id = old.stdout.strip() if old.returncode == 0 else None
                _cli(container_kind, "commit", name, tag, check=True)
                if old_id:
                    _cli(container_kind, "rmi", old_id)
            assert len(dangling() - before) == 0
        finally:
            _cli(container_kind, "rm", "-f", name)
            _cli(container_kind, "rmi", "-f", tag)


class TestCheckpointSquash:
    """Squashing is triggered purely by the chain's real depth
    (``checkpoint_squash_max_depth``). The depth-check inspect is mocked to
    return a chain of the wanted length, or the max depth is patched down to
    force a squash without a deep chain."""

    @staticmethod
    def _chain(layers):
        return lambda: _Done(stdout=json.dumps(layers).encode())

    def _commit(self, sb, handler, *, max_depth=None):
        with _committing(sb, handler, max_depth=max_depth):
            sb.commit()

    def test_warns_when_fast_path_fails_even_if_fallback_succeeds(self, backend, capsys):
        """The fast accumulator path failing must always be visible, even when
        the export/import fallback then succeeds."""
        assert backend._accumulated_diff_path is None

        def handler(args, check, input, timeout):
            if "export" in args:
                return _Done(stdout=b"FAKE_TAR")
            if LAYERS in args:
                return _Done(stdout=b'["sha256:layer0"]')
            return _Done()

        self._commit(backend, handler, max_depth=1)
        err = capsys.readouterr().err
        assert "WARNING" in err
        assert "fast squash path failed" in err and "falling back to export/import" in err

    def test_squashes_when_depth_at_or_above_max_depth(self, backend, container_runtime):
        max_depth = backend._agconfig.sandbox.checkpoint_squash_max_depth
        deep = [f"sha256:layer{i}" for i in range(max_depth)]
        calls = []

        def handler(args, check, input, timeout):
            calls.append(args)
            if LAYERS in args:
                return _Done(stdout=json.dumps(deep).encode())
            return _Done()

        self._commit(backend, handler)
        assert any(c[:2] == [container_runtime, "commit"] for c in calls), calls
        assert any("export" in c for c in calls), "expected the fallback's export"
        assert any("import" in c for c in calls), "expected the fallback's import"

    def test_does_not_squash_when_depth_is_below_max_depth(self, backend, container_runtime):
        calls = []

        def handler(args, check, input, timeout):
            calls.append(args)
            if LAYERS in args:
                return _Done(stdout=b'["sha256:layer0", "sha256:layer1"]')
            return _Done()

        self._commit(backend, handler)
        assert any(c[:2] == [container_runtime, "commit"] for c in calls)
        assert not any("export" in c for c in calls), calls

    def test_squash_check_failure_does_not_raise_and_skips_squash(self, backend):
        calls = []

        def handler(args, check, input, timeout):
            calls.append(args)
            if LAYERS in args:
                return _Done(stdout=b"not json")
            return _Done()

        self._commit(backend, handler)  # must not raise
        assert not any("export" in c for c in calls)

    def test_squash_pipes_export_stdout_into_import_stdin(self, backend):
        """The flatten round-trips the container's real export bytes into
        import's stdin (no shell pipe) and re-applies agency.owner_pid via
        --change, since export/import does not preserve labels -- without it a
        fallback-squashed image would be unreapable."""
        captured = {}

        def handler(args, check, input, timeout):
            if "export" in args:
                return _Done(stdout=b"FAKE_EXPORTED_FILESYSTEM_BYTES")
            if "import" in args:
                captured["input"], captured["args"] = input, list(args)
                return _Done()
            if LAYERS in args:
                return _Done(stdout=b'["sha256:layer0"]')
            return _Done()

        self._commit(backend, handler, max_depth=1)
        assert captured["input"] == b"FAKE_EXPORTED_FILESYSTEM_BYTES"
        assert captured["args"][-1] == backend._lifecycle_tag()
        assert captured["args"][-2] == "-"
        change = captured["args"].index("--change")
        assert captured["args"][change + 1] == (
            f"LABEL {_container._AGENCY_OWNER_PID_LABEL}={backend._owner_pid}"
        )

    def test_squash_commit_records_post_import_diff_ids_for_rebaseline(self, backend):
        """After a successful fallback, `_squash_base_diff_ids` is the
        flattened image's own chain, so the NEXT squash can use the fast path."""
        assert backend._squash_base_diff_ids is None

        def handler(args, check, input, timeout):
            if "export" in args:
                return _Done(stdout=b"FAKE_TAR")
            if LAYERS in args:
                return _Done(stdout=b'["sha256:flattened-single-layer"]')
            return _Done()

        self._commit(backend, handler, max_depth=1)
        assert backend._squash_base_diff_ids == ["sha256:flattened-single-layer"]

    def test_squash_commit_degrades_gracefully_when_recording_diff_ids_fails(self, backend, capsys):
        """A failed read-back of the flattened chain loses only the
        re-baseline optimisation: the squash itself already succeeded."""
        n = {"n": 0}

        def handler(args, check, input, timeout):
            if "export" in args:
                return _Done(stdout=b"FAKE_TAR")
            if LAYERS in args:
                n["n"] += 1
                return _Done(stdout=b'["sha256:layer0"]' if n["n"] <= 2 else b"not json")
            return _Done()

        self._commit(backend, handler, max_depth=1)  # must not raise
        err = capsys.readouterr().err
        assert backend._squash_base_diff_ids is None
        assert "WARNING" in err and "squash failed" not in err

    def test_second_squash_after_fallback_uses_fast_path(
        self, backend, container_runtime, tmp_path
    ):
        """A sandbox that fell back to export/import once must use the FAST
        path on its next squash instead of paying export/import forever."""
        calls = []

        def cycle1(args, check, input, timeout):
            calls.append(args)
            if "export" in args:
                return _Done(stdout=b"FAKE_TAR")
            if LAYERS in args:
                return _Done(stdout=b'["sha256:flattened-1"]')
            return _Done()

        self._commit(backend, cycle1, max_depth=1)
        assert any("export" in c for c in calls), "cycle 1 must have fallen back"
        assert backend._squash_base_diff_ids == ["sha256:flattened-1"]

        diff_dir = tmp_path / "diff2"
        diff_dir.mkdir()
        (diff_dir / "f").write_text("x")
        calls.clear()

        def cycle2(args, check, input, timeout):
            calls.append(args)
            if args[1] == "save" and "-o" in args:
                Path(args[args.index("-o") + 1]).write_bytes(
                    (tmp_path / "fake-base-save.tar").read_bytes()
                )
            if LAYERS in args:
                return _Done(stdout=b'["sha256:flattened-1", "sha256:new-commit"]')
            if args[1] == "inspect":
                return _Done(
                    stdout=json.dumps(
                        [{"Config": {}, "Architecture": "amd64", "Os": "linux"}]
                    ).encode()
                )
            return _Done()

        fake_save = tmp_path / "fake-base-save.tar"
        import tarfile as _tarfile

        with _tarfile.open(fake_save, "w") as _t:
            _entry = _tarfile.TarInfo("sha256:flattened-1.tar")
            _entry.size = 0
            _t.addfile(_entry)

        with _committing(
            backend,
            cycle2,
            max_depth=2,
            extra=[patch.object(type(backend), "_locate_layer_diff_dir", return_value=diff_dir)],
        ):
            backend.commit()
        assert not any("export" in c or "import" in c for c in calls), calls
        assert any(c[:2] == [container_runtime, "load"] for c in calls), calls

    def test_squash_failure_is_best_effort(self, backend, capsys):
        """The plain commit already succeeded: a failed squash only means the
        chain keeps growing until the next attempt."""

        def handler(args, check, input, timeout):
            if "export" in args or "import" in args:
                raise RuntimeError("simulated export/import failure")
            if LAYERS in args:
                return _Done(stdout=b'["sha256:layer0"]')
            return _Done()

        self._commit(backend, handler, max_depth=1)  # must not raise
        err = capsys.readouterr().err
        assert "WARNING" in err and "squash failed" in err

    def test_plain_commit_retries_on_first_failure(self, backend, container_runtime):
        attempts = []

        def handler(args, check, input, timeout):
            if "commit" in args:
                attempts.append(1)
                if len(attempts) == 1:
                    raise RuntimeError("simulated commit failure")
            if LAYERS in args:
                return _Done(stdout=b'["sha256:layer0"]')
            return _Done()

        with patch.object(_container.time, "sleep"):
            self._commit(backend, handler)
        assert len(attempts) == 2

    def test_real_squash_flattens_real_layer_depth(self, container_kind):
        """A real `export | import` resets RootFS.Layers to 1, i.e. it really
        flattens rather than re-tagging the same growing chain. Uses a tiny
        image: this exercises the mechanism, not the sandbox base."""

        def layers(ref):
            return int(
                _cli(
                    container_kind, "inspect", "--format={{len .RootFS.Layers}}", ref
                ).stdout.strip()
            )

        name = f"test-squash-{container_kind}-{uuid.uuid4().hex[:8]}"
        tag = f"agency/lifecycle-{name}"
        _cli(
            container_kind,
            "run",
            "-d",
            "--name",
            name,
            ALPINE_IMAGE,
            "tail",
            "-f",
            "/dev/null",
            check=True,
        )
        try:
            for i in range(3):
                _cli(
                    container_kind, "exec", name, "sh", "-c", f"echo {i} > /marker-{i}", check=True
                )
                _cli(container_kind, "commit", name, tag, check=True)
            assert layers(tag) > 1, "expected stacked layers before squashing"
            exported = _cli(container_kind, "export", name, check=True, text=False)
            _cli(container_kind, "import", "-", tag, check=True, text=False, input=exported.stdout)
            assert layers(tag) == 1
        finally:
            _cli(container_kind, "rm", "-f", name)
            _cli(container_kind, "rmi", "-f", tag)


# ===========================================================================
# Fast squash: locating layer diffs, rootless id mapping, the accumulator
# ===========================================================================


class TestLocateLayerDiffDir:
    """The shared contract of ``_locate_layer_diff_dir``: find the on-disk diff
    directory of a layer digest, and degrade to ``None`` (with a visible
    WARNING) for an unsupported driver, a missing lookup or missing content.
    The two runtimes' on-disk layouts are built by the table's ``fake_root`` /
    ``add_layer`` (docker overlay2 layerdb vs podman layers.json); mechanisms
    unique to one runtime stay in test_docker_backend.py / test_podman_backend.py."""

    @staticmethod
    def _patch_storage(backend, spec, value):
        return patch.object(type(backend), spec.storage_probe, return_value=value)

    def test_finds_diff_dir_matching_digest(self, backend, spec, tmp_path):
        root = spec.fake_root(tmp_path)
        expected = spec.add_layer(root, "sha256:target123", "layer-target")
        spec.add_layer(root, "sha256:other456", "layer-other")
        with self._patch_storage(backend, spec, (root, spec.good_driver)):
            result = backend._locate_layer_diff_dir("sha256:target123")
        assert result == expected and (result / "marker").read_text() == "x"

    def test_returns_none_and_warns_when_digest_not_found(self, backend, spec, tmp_path, capsys):
        root = spec.fake_root(tmp_path)
        spec.add_layer(root, "sha256:other456", "layer-other")
        with self._patch_storage(backend, spec, (root, spec.good_driver)):
            assert backend._locate_layer_diff_dir("sha256:nonexistent") is None
        err = capsys.readouterr().err
        assert "WARNING" in err and "sha256:nonexistent" in err

    def test_returns_none_and_names_an_unsupported_driver(self, backend, spec, tmp_path, capsys):
        root = spec.fake_root(tmp_path)
        spec.add_layer(root, "sha256:target123", "layer-target")
        with self._patch_storage(backend, spec, (root, spec.bad_driver)):
            assert backend._locate_layer_diff_dir("sha256:target123") is None
        err = capsys.readouterr().err
        assert "WARNING" in err and spec.bad_driver in err

    def test_returns_none_when_info_lookup_fails(self, backend, spec):
        with self._patch_storage(backend, spec, None):
            assert backend._locate_layer_diff_dir("sha256:anything") is None

    def test_returns_none_and_warns_when_the_entry_exists_but_its_content_is_missing(
        self, backend, spec, tmp_path, capsys
    ):
        """A matching entry whose diff dir was already cleaned up must degrade
        to None, not raise."""
        root = spec.fake_root(tmp_path)
        spec.add_layer(root, "sha256:target123", "layer-target", with_content=False)
        with self._patch_storage(backend, spec, (root, spec.good_driver)):
            assert backend._locate_layer_diff_dir("sha256:target123") is None
        assert "WARNING" in capsys.readouterr().err

    def test_info_failure_is_reported_once_and_cached(self, backend, container_runtime, capsys):
        """`<rt> info` failing disables the fast path for this backend's whole
        lifetime, with one visible warning."""
        with patch.object(type(backend), "_run", side_effect=RuntimeError("daemon down")):
            assert backend._locate_layer_diff_dir("sha256:x") is None
            assert backend._locate_layer_diff_dir("sha256:y") is None
        err = capsys.readouterr().err
        assert err.count(f"`{container_runtime} info` failed") == 1


def _rootless_patches(backend, runtime: str, mode: str):
    """Patches that make *backend* look like *mode* to its rootless-detection:
    not_rootless | info_unavailable | rootless_no_maps | rootless_maps."""
    cls = type(backend)
    if runtime == "docker":
        info = {
            "not_rootless": {"SecurityOptions": []},
            "info_unavailable": None,
            "rootless_no_maps": {"SecurityOptions": ["name=rootless"]},
            "rootless_maps": {"SecurityOptions": ["name=rootless"]},
        }[mode]
        patches = [patch.object(cls, "_docker_info", return_value=info)]
        if mode == "rootless_no_maps":
            patches.append(patch.object(cls, "_find_dockerd_pid", return_value=None))
        if mode == "rootless_maps":
            patches.append(
                patch.object(
                    cls, "_rootless_id_maps", return_value=(ROOTLESS_UID_MAP, ROOTLESS_GID_MAP)
                )
            )
        return patches

    def entries(rows):
        return [{"container_id": c, "host_id": h, "size": s} for c, h, s in rows]

    info = {
        "not_rootless": {"host": {"security": {"rootless": False}}},
        "info_unavailable": None,
        "rootless_no_maps": {"host": {"security": {"rootless": True}, "idMappings": {}}},
        "rootless_maps": {
            "host": {
                "security": {"rootless": True},
                "idMappings": {
                    "uidmap": entries(ROOTLESS_UID_MAP),
                    "gidmap": entries(ROOTLESS_GID_MAP),
                },
            }
        },
    }[mode]
    return [patch.object(cls, "_podman_info", return_value=info)]


class TestHostToContainerId:
    """``_host_to_container_id`` translates host ownership of an overlay diff
    directory into the ids the container sees (identity unless the runtime is
    rootless AND its id maps are readable). How each runtime discovers rootless
    mode and its maps differs (``docker info`` + the dockerd process' uid_map vs
    ``podman info``'s idMappings); the contract below is shared."""

    @staticmethod
    def _translate(backend, container_runtime, mode, uid, gid):
        with ExitStack() as stack:
            for p in _rootless_patches(backend, container_runtime, mode):
                stack.enter_context(p)
            return backend._host_to_container_id(uid, gid)

    def test_identity_when_not_rootless(self, backend, container_runtime):
        assert self._translate(backend, container_runtime, "not_rootless", 1000, 1000) == (
            1000,
            1000,
        )

    def test_identity_when_runtime_info_is_unavailable(self, backend, container_runtime):
        assert self._translate(backend, container_runtime, "info_unavailable", 1000, 1000) == (
            1000,
            1000,
        )

    def test_identity_when_rootless_but_maps_are_unreadable(self, backend, container_runtime):
        assert self._translate(backend, container_runtime, "rootless_no_maps", 1000, 1000) == (
            1000,
            1000,
        )

    def test_translates_through_the_rootless_id_maps(self, backend, container_runtime):
        """container uid 0 is host 1000; container 1..65536 starts at host 100000."""
        assert self._translate(backend, container_runtime, "rootless_maps", 1000, 1000) == (0, 0)
        assert self._translate(backend, container_runtime, "rootless_maps", 100001, 100001) == (
            2,
            2,
        )

    def test_uid_and_gid_use_their_own_maps(self, backend, container_runtime):
        assert self._translate(backend, container_runtime, "rootless_maps", 1000, 100005) == (0, 6)

    def test_unmapped_ids_pass_through_unchanged(self, backend, container_runtime):
        assert self._translate(backend, container_runtime, "rootless_maps", 999999, 999999) == (
            999999,
            999999,
        )

    def test_translate_id_helpers_agree(self):
        import agency.sandbox.docker as docker_mod
        import agency.sandbox.podman as podman_mod

        for translate in (docker_mod._translate_id, podman_mod._translate_id):
            assert translate(999999, [(0, 200682, 1)]) == 999999
            assert translate(5, [(0, 5, 1)]) == 0
            assert translate(100000, ROOTLESS_UID_MAP) == 1
            assert translate(165536, ROOTLESS_UID_MAP) == 165536  # one past the range


class TestCheckpointAccumulator:
    """Lazy ``_build_accumulator_for_squash`` and ``_accumulator_squash_commit``
    (the fast squash fed by ``_locate_layer_diff_dir``). The locate hook is
    mocked so the shared merge/load logic is what is tested."""

    BASE = PYTHON_IMAGE

    @staticmethod
    def _diff_dir(tmp_path, name, files):
        d = tmp_path / name
        d.mkdir()
        for rel, content in files.items():
            p = d / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        return d

    @staticmethod
    def _inspect_config():
        return _Done(
            stdout=json.dumps([{"Config": {}, "Architecture": "amd64", "Os": "linux"}]).encode()
        )

    def _layers_handler(self, base_layers, current_layers, base_save_tar=None):
        """`_run` answering the base image's chain with *base_layers* and every
        other image's with *current_layers*; plain `inspect` returns a config.
        *base_save_tar* (a real archive path) is served for `save` so the
        Podman squash path can stream the base blobs it needs."""

        def handler(args, check, input, timeout):
            if LAYERS in args:
                chain = base_layers if args[-1] == self.BASE else current_layers
                return _Done(stdout=json.dumps(chain).encode())
            if args[1] == "inspect":
                return self._inspect_config()
            if args[1] == "save" and base_save_tar is not None:
                if "-o" in args:
                    Path(args[args.index("-o") + 1]).write_bytes(base_save_tar.read_bytes())
                return _Done(stdout=base_save_tar.read_bytes())
            return _Done()

        return handler

    def test_build_accumulator_from_overlay_diff_dir(self, backend, tmp_path):
        backend._squash_base_diff_ids = []
        diff_dir = self._diff_dir(tmp_path, "diff1", {"workspace/f1": "one"})
        handler = self._layers_handler([], ["sha256:layer1"])
        with patch.object(type(backend), "_run", _fn_run(handler)):
            with patch.object(type(backend), "_locate_layer_diff_dir", return_value=diff_dir):
                backend._build_accumulator_for_squash("some-tag")
        try:
            assert backend._accumulated_layer_count == 1
            with tarfile.open(backend._accumulated_diff_path, "r") as tf:
                assert tf.extractfile("workspace/f1").read() == b"one"
        finally:
            backend._reset_accumulator()

    def test_build_accumulator_raises_when_diff_dir_not_found(self, backend):
        """Raising makes commit() fall back visibly: a silent miss is what made
        a real production squash fallback undiagnosable."""
        backend._squash_base_diff_ids = []
        handler = self._layers_handler([], ["sha256:layer1"])
        with patch.object(type(backend), "_run", _fn_run(handler)):
            with patch.object(type(backend), "_locate_layer_diff_dir", return_value=None):
                with pytest.raises(RuntimeError, match="could not locate on-disk diff"):
                    backend._build_accumulator_for_squash("some-tag")
        assert backend._accumulated_diff_path is None
        assert backend._accumulated_layer_count == 0

    def test_build_accumulator_merges_all_layers_since_reference(self, backend, tmp_path):
        """One lazy build over a multi-layer gap folds every new layer, not
        just the tip."""
        backend._squash_base_diff_ids = ["sha256:base1"]
        diffs = {
            "sha256:layer1": self._diff_dir(tmp_path, "diff1", {"a": "1"}),
            "sha256:layer2": self._diff_dir(tmp_path, "diff2", {"b": "2"}),
        }
        chain = ["sha256:base1", "sha256:layer1", "sha256:layer2"]
        handler = self._layers_handler([], chain)

        def fake_locate(self_inner, diff_id, *, diff_ids=None):
            return diffs.get(diff_id)

        with patch.object(type(backend), "_run", _fn_run(handler)):
            with patch.object(type(backend), "_locate_layer_diff_dir", fake_locate):
                backend._build_accumulator_for_squash("tag")
        try:
            assert backend._accumulated_layer_count == 2
            with tarfile.open(backend._accumulated_diff_path, "r") as tf:
                assert {m.name for m in tf.getmembers()} == {"a", "b"}
        finally:
            backend._reset_accumulator()

    def test_reset_accumulator_clears_state_and_removes_the_directory(self, backend, tmp_path):
        backend._accumulator_dir = tmp_path / "acc"
        backend._accumulator_dir.mkdir()
        backend._accumulated_diff_path = backend._accumulator_dir / "x.tar"
        backend._accumulated_layer_count = 3
        backend._reset_accumulator()
        assert backend._accumulated_diff_path is None and backend._accumulated_layer_count == 0
        assert backend._accumulator_dir is None and not (tmp_path / "acc").exists()

    def test_ordinary_commit_does_not_build_accumulator(self, backend):
        """Plain commits below the squash depth must not touch TMPDIR."""

        def handler(args, check, input, timeout):
            if LAYERS in args:
                return _Done(stdout=b'["sha256:layer0", "sha256:layer1"]')
            return _Done()

        boom = patch.object(
            type(backend),
            "_build_accumulator_for_squash",
            side_effect=AssertionError("must not build accumulator"),
        )
        with _committing(backend, handler, extra=[boom]):
            backend.commit()
        assert backend._accumulated_diff_path is None and backend._accumulator_dir is None

    def test_accumulator_squash_raises_when_layer_count_mismatched(self, backend, tmp_path):
        """Post-fork: the fresh accumulator (0 layers) does not match the real
        gap (2): must raise so the caller falls back, never squash an
        incomplete diff."""
        backend._accumulated_diff_path = tmp_path / "fake.tar"
        backend._accumulated_diff_path.write_bytes(b"")
        backend._accumulated_layer_count = 0
        handler = self._layers_handler(["sha256:base1"], ["sha256:base1", "sha256:c1", "sha256:c2"])
        with patch.object(type(backend), "_run", _fn_run(handler)):
            with pytest.raises(RuntimeError, match="accumulator tracks"):
                backend._accumulator_squash_commit("some-tag")

    def test_accumulator_squash_raises_when_base_not_a_prefix(self, backend, tmp_path):
        backend._accumulated_diff_path = tmp_path / "fake.tar"
        backend._accumulated_diff_path.write_bytes(b"")
        backend._accumulated_layer_count = 1
        handler = self._layers_handler(["sha256:base1"], ["sha256:different", "sha256:c1"])
        with patch.object(type(backend), "_run", _fn_run(handler)):
            with pytest.raises(RuntimeError):
                backend._accumulator_squash_commit("some-tag")

    def test_accumulator_squash_raises_when_no_accumulator(self, backend):
        assert backend._accumulated_diff_path is None
        with pytest.raises(RuntimeError, match="no checkpoint diff accumulator"):
            backend._accumulator_squash_commit("some-tag")

    def _accumulator_with(self, backend, tmp_path, files=None):
        from agency.sandbox._layer_squash import overlay_diff_to_tar

        diff_dir = self._diff_dir(tmp_path, "diff1", files or {"a": "1"})
        tar = tmp_path / "accum.tar"
        overlay_diff_to_tar(diff_dir, tar)
        backend._accumulated_diff_path = tar
        backend._accumulated_layer_count = 1

    def test_accumulator_squash_uses_squash_base_diff_ids_when_already_set(self, backend, tmp_path):
        """Once a squash has succeeded, `_squash_base_diff_ids` -- not the
        original base image -- is the reference chain: the point of
        re-baselining."""
        self._accumulator_with(backend, tmp_path)
        backend._squash_base_diff_ids = ["sha256:prior-squash-layer"]
        calls = []
        base = self._layers_handler(
            ["sha256:prior-squash-layer"],
            ["sha256:prior-squash-layer", "sha256:c1"],
            base_save_tar=self._tiny_base_save_tar(tmp_path),
        )

        def handler(args, check, input, timeout):
            calls.append(args)
            return base(args, check, input, timeout)

        with patch.object(type(backend), "_run", _fn_run(handler)):
            backend._accumulator_squash_commit("some-tag")
        # No diff_ids inspect of the base image; the Podman save of it is
        # expected (blob streaming), the Docker path touches it not at all.
        layers_inspects = [c for c in calls if LAYERS in c]
        assert all(c[-1] != self.BASE for c in layers_inspects), calls
        assert backend._squash_base_diff_ids[0] == "sha256:prior-squash-layer"
        assert len(backend._squash_base_diff_ids) == 2

    def test_accumulator_squash_records_new_diff_ids_on_success(self, backend, tmp_path):
        """A fresh backend's first squash validates against its base image,
        then records base chain + exactly one flattened layer."""
        assert backend._squash_base_diff_ids is None
        self._accumulator_with(backend, tmp_path)
        handler = self._layers_handler(
            ["sha256:base1"],
            ["sha256:base1", "sha256:c1"],
            base_save_tar=self._tiny_base_save_tar(tmp_path),
        )
        with patch.object(type(backend), "_run", _fn_run(handler)):
            backend._accumulator_squash_commit("some-tag")
        assert backend._squash_base_diff_ids[0] == "sha256:base1"
        assert len(backend._squash_base_diff_ids) == 2

    def test_accumulator_squash_loads_and_tags_the_merged_image_with_this_runtime(
        self, backend, container_runtime, tmp_path
    ):
        self._accumulator_with(backend, tmp_path)
        calls = []
        base = self._layers_handler(
            ["sha256:base1"],
            ["sha256:base1", "sha256:c1"],
            base_save_tar=self._tiny_base_save_tar(tmp_path),
        )

        def handler(args, check, input, timeout):
            calls.append(args)
            return base(args, check, input, timeout)

        with patch.object(type(backend), "_run", _fn_run(handler)):
            backend._accumulator_squash_commit("some-tag")
        loads = [c for c in calls if c[1] == "load"]
        assert loads and loads[0][0] == container_runtime
        assert all(c[0] == container_runtime for c in calls)

    @staticmethod
    def _tiny_base_save_tar(tmp_path):
        """A minimal docker-archive for the `save` call in the Podman squash
        path: one legacy-layout layer blob, which the archive builder
        streams through."""
        import io
        import tarfile

        path = tmp_path / "base-save.tar"
        with tarfile.open(path, "w") as out:
            data = b"layer"
            entry = tarfile.TarInfo("sha256:base1.tar")
            entry.size = len(data)
            out.addfile(entry, io.BytesIO(data))
        return path

    def test_real_end_to_end_fast_squash(self, container_kind, kind_spec):
        """Several plain-commit cycles on the real base image, each folded into
        the accumulator via the REAL storage lookup, then a real squash: it
        must be fast, correct, and leave the base image's own layers
        untouched. rm_container() between cycles makes each commit a genuinely
        incremental layer on the previous checkpoint."""
        pytest.importorskip("agency.sandbox._layer_squash")
        sb = make_sandbox(container_kind)
        probe = getattr(sb._backend, kind_spec.storage_probe)()
        if probe is None or probe[1] not in {kind_spec.good_driver, "overlayfs"}:
            sb.destroy()
            pytest.skip(f"fast layer accumulator needs {kind_spec.good_driver} storage: {probe}")
        data_root, driver = probe
        if driver in ("overlay2", "overlay"):
            readable = (
                data_root / "image" / "overlay2" / "layerdb" / "sha256"
                if driver == "overlay2"
                else data_root / "overlay-layers"
            )
            try:
                next(readable.iterdir(), None)
            except OSError:
                sb.destroy()
                pytest.skip(
                    f"test process cannot read {readable}: fast path structurally unavailable"
                )

        base_layers_before = _cli(
            container_kind, "inspect", "--format={{json .RootFS.Layers}}", PYTHON_IMAGE
        ).stdout
        sb.exec("mkdir -p /workspace/proj && echo one > /workspace/proj/f1")
        sb.commit()
        sb.rm_container()
        sb.exec("echo two > /workspace/proj/f2 && rm /workspace/proj/f1")
        sb.commit()
        sb.rm_container()
        sb.exec("mkdir -p /workspace/proj/sub && echo three > /workspace/proj/sub/f3")
        t0 = time.time()
        with patch.object(sb._backend._agconfig.sandbox, "checkpoint_squash_max_depth", 1):
            sb.commit()  # commits AND squashes
        elapsed = time.time() - t0
        try:
            assert elapsed < kind_spec.fast_squash_budget_s, f"fast squash took {elapsed:.1f}s"
            assert (
                _cli(
                    container_kind, "inspect", "--format={{json .RootFS.Layers}}", PYTHON_IMAGE
                ).stdout
                == base_layers_before
            )
            tag = sb._backend._lifecycle_tag()
            layers = int(
                _cli(
                    container_kind, "inspect", "--format={{len .RootFS.Layers}}", tag
                ).stdout.strip()
            )
            assert layers == len(json.loads(base_layers_before)) + 1
            result = _cli(
                container_kind,
                "run",
                "--rm",
                tag,
                "sh",
                "-c",
                "ls /workspace/proj/f1 2>&1; cat /workspace/proj/f2; cat /workspace/proj/sub/f3",
            )
            assert "No such file" in result.stdout + result.stderr
            assert "two" in result.stdout and "three" in result.stdout
        finally:
            sb.destroy()


# ===========================================================================
# Facade-level container lifecycle against a REAL daemon
# ===========================================================================


def _container_names(kind: str, name: str, *, all_states: bool = False) -> str:
    args = ["ps", "-a"] if all_states else ["ps"]
    return _cli(kind, *args, "--filter", f"name={name}", "--format", "{{.Names}}").stdout


class TestRealContainerLifecycle:
    """Real containers: CLI-visible state, container identity across resume,
    retries, and semaphore accounting. Requires the runtime's daemon."""

    def test_container_starts_lazily_and_destroy_removes_it(self, container_kind):
        sb = make_sandbox(container_kind)
        name = sb._backend._container_name()
        sb.exec("true")
        assert name in _container_names(container_kind, name)
        sb.destroy()
        assert name not in _container_names(container_kind, name, all_states=True)

    def test_rm_container_removes_the_container_after_commit(self, container_kind):
        """rm_container() is a separate, explicit step from commit()."""
        sb = make_sandbox(container_kind)
        name = sb._backend._container_name()
        tag = sb._backend._lifecycle_tag()
        try:
            sb.write_file("/workspace/marker.txt", "lifecycle\n")
            sb.commit()
            sb.rm_container()
            assert name not in _container_names(container_kind, name, all_states=True)
        finally:
            _cli(container_kind, "rmi", "-f", tag)
            sb.destroy()

    def test_rm_container_without_a_commit_creates_no_image(self, container_kind):
        sb = make_sandbox(container_kind)
        name = sb._backend._container_name()
        tag = sb._backend._lifecycle_tag()
        try:
            sb.write_file("/workspace/dirty.txt", "dirty\n")
            previous = sb._checkpoint_image
            sb.rm_container()
            assert name not in _container_names(container_kind, name, all_states=True)
            assert sb._checkpoint_image == previous
            assert _cli(container_kind, "images", "-q", tag).stdout.strip() == ""
        finally:
            _cli(container_kind, "rmi", "-f", tag)
            sb.destroy()

    def test_stop_then_exec_resumes_the_same_container_with_the_workspace_intact(
        self, container_kind
    ):
        """stop() (hibernate: no commit, no image, no rm) then exec() resumes
        the SAME container via `start` -- the ordinary per-tool-call path."""
        sb = make_sandbox(container_kind)
        name = sb._backend._container_name()
        try:
            sb.write_file("/workspace/persistent.txt", "saved\n")
            before = _cli(container_kind, "inspect", "--format", "{{.Id}}", name).stdout.strip()
            sb.stop()
            assert not sb._backend._container_running()
            assert sb._checkpoint_image is None
            out, rc = sb.exec("cat /workspace/persistent.txt")
            assert rc == 0 and "saved" in out
            assert (
                _cli(container_kind, "inspect", "--format", "{{.Id}}", name).stdout.strip()
                == before
            )
        finally:
            sb.destroy()

    def test_rm_container_retries_rm_on_first_failure(self, container_kind):
        sb = make_sandbox(container_kind)
        sb.write_file("/workspace/x.txt", "x\n")
        name = sb._backend._container_name()
        count = [0]
        real_run = sb._backend._run

        def flaky_run(cmd, **kwargs):
            if "rm" in cmd and "-f" in cmd and name in cmd:
                count[0] += 1
                if count[0] == 1:
                    raise RuntimeError("simulated rm -f failure")
            return real_run(cmd, **kwargs)

        sb._backend._run = flaky_run
        try:
            sb.rm_container()
            assert count[0] == 2, "expected one failure then one success"
            assert not sb._backend._container_running()
            assert name not in _container_names(container_kind, name, all_states=True)
        finally:
            sb._backend._run = real_run
            sb.destroy()

    def test_rm_container_raises_after_all_retries_fail(self, container_kind):
        sb = make_sandbox(container_kind)
        sb.write_file("/workspace/x.txt", "x\n")
        real_run = sb._backend._run

        def always_fail_rm(cmd, **kwargs):
            if "rm" in cmd and "-f" in cmd:
                raise RuntimeError("simulated persistent failure")
            return real_run(cmd, **kwargs)

        sb._backend._run = always_fail_rm
        try:
            with pytest.raises(RuntimeError, match="simulated persistent failure"):
                sb.rm_container()
        finally:
            sb._backend._run = real_run
            sb.destroy()

    def test_failed_rm_then_destroy_releases_the_slot_exactly_once(self, container_kind):
        """If rm_container() exhausts its retries the slot is still held;
        destroy() releases it, once -- never both (double release over-credits
        the semaphore past the keyring quota)."""
        sb = make_sandbox(container_kind)
        sb.write_file("/workspace/x.txt", "x\n")
        real_run = sb._backend._run

        def always_fail_rm(cmd, **kwargs):
            if "rm" in cmd and "-f" in cmd:
                raise RuntimeError("simulated persistent failure")
            return real_run(cmd, **kwargs)

        held = _container._container_semaphore._semlock._get_value()
        sb._backend._run = always_fail_rm
        try:
            with pytest.raises(RuntimeError, match="simulated persistent failure"):
                sb.rm_container()
        finally:
            sb._backend._run = real_run
        assert _container._container_semaphore._semlock._get_value() == held
        sb.destroy()
        assert _container._container_semaphore._semlock._get_value() == held + 1

    def test_rm_container_on_an_already_hibernating_container_releases_the_slot_once(
        self, container_kind
    ):
        sb = make_sandbox(container_kind)
        sb.write_file("/workspace/x.txt", "x\n")
        sem = _container._container_semaphore
        held = sem._semlock._get_value()
        sb.stop()
        assert sem._semlock._get_value() == held + 1, "stop() must release the slot exactly once"
        sb.rm_container()
        assert sem._semlock._get_value() == held + 1
        sb.destroy()
        assert sem._semlock._get_value() == held + 1

    def test_concurrent_runtime_calls_are_gated_by_the_semaphore(self, container_kind):
        """Every CLI call goes through `_run`, which holds the shared semaphore."""
        sem = _container._get_docker_semaphore()
        sandboxes = [make_sandbox(container_kind) for _ in range(4)]
        tags = [sb._backend._lifecycle_tag() for sb in sandboxes]
        for sb in sandboxes:
            sb.write_file("/workspace/x.txt", "x\n")
        concurrent, peak, lock = [0], [0], threading.Lock()
        real_acquire, real_release = sem.acquire, sem.release

        def counting_acquire(*a, **kw):
            real_acquire(*a, **kw)
            with lock:
                concurrent[0] += 1
                peak[0] = max(peak[0], concurrent[0])

        def counting_release(*a, **kw):
            with lock:
                concurrent[0] -= 1
            real_release(*a, **kw)

        sem.acquire, sem.release = counting_acquire, counting_release
        try:
            threads = [threading.Thread(target=sb.commit) for sb in sandboxes]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        finally:
            sem.acquire, sem.release = real_acquire, real_release
            for tag in tags:
                _cli(container_kind, "rmi", "-f", tag)
            for sb in sandboxes:
                sb.destroy()
        assert peak[0] <= _container.DOCKER_SEMAPHORE_LIMIT

    def test_a_container_stuck_in_created_state_is_started_not_recreated(self, container_kind):
        sb = make_sandbox(container_kind)
        name = sb._backend._container_name()
        try:
            _cli(
                container_kind,
                "create",
                "--name",
                name,
                sb._backend._resolve_image(sb.agconfig.sandbox.base_image),
                "tail",
                "-f",
                "/dev/null",
                check=True,
            )
            status = _cli(container_kind, "inspect", "--format", "{{.State.Status}}", name)
            assert status.stdout.strip() == "created"
            sb._backend._ensure_started()
            assert sb._backend._container_running() is True
            out, rc = sb.exec("echo ok")
            assert rc == 0 and "ok" in out
        finally:
            sb.destroy()

    def test_commit_retries_on_first_failure(self, container_kind):
        sb = make_sandbox(container_kind)
        sb.write_file("/workspace/x.txt", "x\n")
        tag = sb._backend._lifecycle_tag()
        count = [0]
        real_run = sb._backend._run

        def flaky_run(cmd, **kwargs):
            if "commit" in cmd:
                count[0] += 1
                if count[0] == 1:
                    raise RuntimeError("simulated commit failure")
            return real_run(cmd, **kwargs)

        sb._backend._run = flaky_run
        try:
            assert sb.commit() is True
            assert count[0] == 2, "expected one failure then one success"
            assert sb._checkpoint_image == tag
            assert sb._backend._container_running(), "commit() never touches the container"
            assert _cli(container_kind, "images", "-q", tag).stdout.strip() != ""
        finally:
            sb._backend._run = real_run
            _cli(container_kind, "rmi", "-f", tag)
            sb.destroy()

    def test_commit_raises_after_all_retries_fail_and_the_container_keeps_running(
        self, container_kind
    ):
        sb = make_sandbox(container_kind)
        sb.write_file("/workspace/x.txt", "x\n")
        previous = sb._checkpoint_image
        real_run = sb._backend._run

        def always_fail_commit(cmd, **kwargs):
            if "commit" in cmd:
                raise RuntimeError("simulated persistent commit failure")
            return real_run(cmd, **kwargs)

        sb._backend._run = always_fail_commit
        try:
            with pytest.raises(RuntimeError, match="simulated persistent commit failure"):
                sb.commit()
        finally:
            sb._backend._run = real_run
        assert sb._checkpoint_image == previous
        assert sb._backend._container_running()
        sb.destroy()

    def test_an_externally_stopped_container_is_resumed_with_its_id_and_files(self, container_kind):
        """A container `stop`ped behind the sandbox's back (exited, not
        removed) is resumed via `start`, not recreated."""
        sb = make_sandbox(container_kind)
        name = sb._backend._container_name()
        try:
            sb.write_file("/workspace/exited.txt", "still-here\n")
            before = _cli(container_kind, "inspect", "--format", "{{.Id}}", name).stdout.strip()
            _cli(container_kind, "stop", "-t", "0", name)
            sb._backend._ensure_started()
            assert sb._backend._container_running() is True
            after = _cli(container_kind, "inspect", "--format", "{{.Id}}", name).stdout.strip()
            assert after == before
            assert sb.read_file("/workspace/exited.txt") == "still-here\n"
        finally:
            sb.destroy()

    def test_the_run_command_carries_the_owner_pid_label(self, container_kind):
        sb = make_sandbox(container_kind)
        name = sb._backend._container_name()
        try:
            sb.exec("true")
            label = _cli(
                container_kind,
                "inspect",
                "--format",
                '{{index .Config.Labels "agency.owner_pid"}}',
                name,
            ).stdout.strip()
            assert label == str(os.getpid())
        finally:
            sb.destroy()

    def test_memory_and_cpu_limits_reach_the_runtime(self, container_kind):
        cfg = agconfig_cls()
        cfg.sandbox.gpu_passthrough = False
        cfg.resources.idle_memory = "256m"
        sb = make_sandbox(container_kind, agconfig=cfg)
        name = sb._backend._container_name()
        try:
            sb.exec("true")
            memory = _cli(container_kind, "inspect", "--format", "{{.HostConfig.Memory}}", name)
            assert memory.stdout.strip() == str(256 * 1024 * 1024)
        finally:
            sb.destroy()


# ===========================================================================
# Base-class logic exercised without a runtime
# ===========================================================================


class TestReadFileUnit:
    """read_file's base64-decode/error-mapping lives on the shared
    ``_ContainerBackendBase``; exercised directly, with no runtime."""

    @staticmethod
    def _sb():
        sb = _container._ContainerBackendBase.__new__(_container._ContainerBackendBase)
        sb._agconfig = agconfig_cls()
        return sb

    def test_read_file_returns_text_content(self):
        sb = self._sb()
        b64 = base64.b64encode(b"hello world\n").decode()
        with patch.object(sb, "_container_exec", return_value=(b64, 0)):
            assert sb.read_file("/workspace/hello.txt") == "hello world\n"

    def test_read_file_missing_path_raises_file_not_found(self):
        sb = self._sb()
        sb._container_exec = MagicMock(side_effect=[("", 1), ("", 1)])
        with pytest.raises(FileNotFoundError):
            sb.read_file("/workspace/missing.txt")

    def test_read_file_directory_raises_is_a_directory_error(self):
        sb = self._sb()
        sb._container_exec = MagicMock(side_effect=[("", 1), ("", 0)])
        with pytest.raises(IsADirectoryError):
            sb.read_file("/workspace/outputs")

    def test_read_file_binary_raises_unicode_decode_error(self):
        sb = self._sb()
        b64 = base64.b64encode(bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])).decode()
        with patch.object(sb, "_container_exec", return_value=(b64, 0)):
            with pytest.raises(UnicodeDecodeError):
                sb.read_file("/workspace/image.png")

    def test_read_file_valid_utf8_succeeds(self):
        sb = self._sb()
        content = "def main():\n    pass\n"
        b64 = base64.b64encode(content.encode("utf-8")).decode()
        with patch.object(sb, "_container_exec", return_value=(b64, 0)):
            assert sb.read_file("/workspace/core.py") == content


class TestIngestPtracePidsUnit:
    """Pure-logic PID bookkeeping with ``_container_exec`` mocked to report no
    processes (the container family's own ``get_live_pids``)."""

    def test_ingest_populates_watched_and_managed_sets(self, backend):
        backend.ingest_ptrace_pids(spawned={111, 222})
        assert 111 in backend._watched_pids and 222 in backend._watched_pids
        assert backend._ptrace_managed_pids == {111, 222}

    def test_ingest_skips_baseline_and_daemon_pids(self, backend):
        backend._baseline_pids = {111}
        backend._daemon_pids.add(222)
        backend.ingest_ptrace_pids(spawned={111, 222, 333})
        assert 111 not in backend._watched_pids and 222 not in backend._watched_pids
        assert 333 in backend._watched_pids

    def test_get_live_pids_trusts_ptrace_managed_over_empty_proc_scan(self, backend):
        backend.ingest_ptrace_pids(spawned={555})
        with patch.object(backend, "_container_exec", return_value=("", 0)):
            live = backend.get_live_pids()
        assert 555 in live and 555 in backend._watched_pids

    def test_get_live_pids_prunes_non_ptrace_pid_absent_from_proc_scan(self, backend):
        backend._watched_pids[777] = 0.0
        with patch.object(backend, "_container_exec", return_value=("", 0)):
            live = backend.get_live_pids()
        assert 777 not in live and 777 not in backend._watched_pids

    def test_ingest_exited_removes_from_both_sets(self, backend):
        backend.ingest_ptrace_pids(spawned={888})
        backend.ingest_ptrace_pids(exited={888})
        assert 888 not in backend._watched_pids and 888 not in backend._ptrace_managed_pids


@pytest.mark.parametrize("runtime", ["docker", "podman"])
def test_backend_destroy_remains_retryable_after_removal_failure(runtime, tmp_path):
    """A failed rm_container() leaves ``_destroyed`` False so destroy() can be
    retried; once it succeeds further calls are no-ops."""
    cls = _container._ContainerBackendBase
    backend = MagicMock()
    backend._destroyed = False
    backend._watched_pids = {}
    backend._checkpoint_image = None
    backend._accumulator_dir = None
    backend._runtime = runtime
    backend._checkpoint_storage = None
    backend.rm_container.side_effect = [RuntimeError("removal failed"), None]
    with pytest.raises(RuntimeError, match="removal failed"):
        cls.destroy(backend)
    assert not backend._destroyed
    cls.destroy(backend)
    cls.destroy(backend)
    assert backend._destroyed
    assert backend.rm_container.call_count == 2


# ===========================================================================
# Real GPU passthrough (needs GPU hardware and a daemon)
# ===========================================================================


def _host_gpu_available() -> bool:
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


def _runtime_gpu_usable(kind: str) -> bool:
    """Probe that *kind* can really attach the host's NVIDIA GPUs, with
    known-good flags and independently of ``_gpu_flags`` (so a bug there
    cannot hide itself): `<rt> run --device ...`, `--pull=never`, nvidia-smi."""
    if not _host_gpu_available():
        return False
    try:
        result = subprocess.run(
            [
                kind, "run", "--rm", "--pull=never",
                *RUNTIMES[kind].gpu_probe_flags,
                PYTHON_IMAGE, "nvidia-smi", "-L",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )  # fmt: skip
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "GPU" in result.stdout


@pytest.fixture
def gpu_kind(container_kind) -> str:
    if not _runtime_gpu_usable(container_kind):
        pytest.skip(
            f"{container_kind} cannot attach an NVIDIA GPU (or the local {PYTHON_IMAGE} is missing)"
        )
    return container_kind


class TestGpuProbe:
    @pytest.mark.parametrize(
        "probe,usable",
        [
            (subprocess.CompletedProcess([], 0, stdout="GPU 0: fixture"), True),
            (subprocess.CompletedProcess([], 126, stdout="", stderr="unresolvable"), False),
            (subprocess.CompletedProcess([], 0, stdout=""), False),
            (OSError("missing runtime"), False),
            (subprocess.TimeoutExpired("rt", 30), False),
        ],
        ids=["usable", "nonzero-exit", "no-gpu-listed", "oserror", "timeout"],
    )
    def test_probe_checks_independent_runtime_prerequisites(
        self, monkeypatch, container_runtime, spec, probe, usable
    ):
        monkeypatch.setitem(globals(), "_host_gpu_available", lambda: True)
        run = MagicMock()
        if isinstance(probe, Exception):
            run.side_effect = probe
        else:
            run.return_value = probe
        monkeypatch.setattr(subprocess, "run", run)
        assert _runtime_gpu_usable(container_runtime) is usable
        command = run.call_args.args[0]
        assert command[0] == container_runtime and "--pull=never" in command
        assert tuple(command[command.index("--rm") + 2 : command.index(PYTHON_IMAGE)]) == (
            spec.gpu_probe_flags
        )


class TestRealGpuPassthrough:
    def test_nvidia_smi_inside_a_real_container_sees_the_gpus(self, gpu_kind):
        """Starts a REAL container with exactly the flags ``_gpu_flags`` returns
        and asserts nvidia-smi inside it sees the host's GPUs: the check that
        would have caught `--gpus all` being accepted-but-useless on podman."""
        _container._gpu_kind_cache.clear()
        flags = _container._gpu_flags(gpu_kind)
        assert flags, "expected non-empty GPU flags on a host with a real GPU"
        result = _cli(gpu_kind, "run", "--rm", *flags, PYTHON_IMAGE, "nvidia-smi", "-L")
        assert result.returncode == 0, (
            f"nvidia-smi failed (exit {result.returncode}): {result.stderr!r}"
        )
        assert "GPU" in result.stdout

    def test_agent_cannot_hijack_cuda_visible_devices(self, gpu_kind):
        """An agent's inline ``CUDA_VISIBLE_DEVICES=999 cmd`` prefix must not
        shadow the harness-leased id (base.py's `readonly` export), else compute
        lands on a GPU the pool's bookkeeping does not know the sandbox holds.
        Goes through the real agSandbox/agResourcePool/reserve path."""
        import functools

        from agency.orchestrator.agresources import agResourcePool

        sb = make_sandbox(gpu_kind, agconfig=_gpu_config())
        pool = agResourcePool(mark_gpus=False)
        assert pool.gpus, "expected at least one real GPU on this host"
        try:
            sb._virtual_gpu_count_requested = 1
            sb._real_gpu_acquire_fn = functools.partial(pool.acquire_real_gpus, sb)
            sb._real_gpu_release_fn = functools.partial(pool.release_real_gpus, sb)
            leased_out, rc = sb.exec("echo $CUDA_VISIBLE_DEVICES")
            assert rc == 0 and leased_out.strip().isdigit(), leased_out
            leased = leased_out.strip()
            out, rc = sb.exec(
                "CUDA_VISIBLE_DEVICES=999 python3 -c "
                "\"import os; print(os.environ['CUDA_VISIBLE_DEVICES'])\""
            )
            assert rc == 0, out
            assert out.strip().splitlines()[-1] == leased, (
                f"inline override took effect: saw {out!r}, harness leased {leased!r}"
            )
        finally:
            sb.destroy()


def _gpu_config():
    cfg = agconfig_cls()
    cfg.sandbox.gpu_passthrough = True
    return cfg


# ===========================================================================
# Real AMD hardware (no daemon needed; ROCm host only)
# ===========================================================================


def _host_rocm_available() -> bool:
    try:
        return (
            subprocess.run(["rocm-smi", "--showid"], capture_output=True, timeout=10).returncode
            == 0
        )
    except Exception:
        return False


rocm_hardware = pytest.mark.skipif(
    not _host_rocm_available(), reason="No real ROCm/AMD GPU hardware available"
)


class TestAmdRenderNodeLiveHardware:
    """Runs the real (unmocked) AMD GPU-to-render-node scoping against
    actual ROCm hardware -- no mocks anywhere, unlike TestGpuFlagsPerRuntime
    above. Skipped automatically without a real rocm-smi + AMD GPU(s).

    This is the check that actually caught the bug being regression-tested
    here: on an 8x MI350X host, naive sorted /dev/dri/renderD* order did
    NOT correspond to rocm-smi's GPU index for ANY of the 8 GPUs (each GPU
    there exposes itself plus 7 XCD/compute-partition sibling render nodes
    -- 64 nodes total -- and even the primary node's number doesn't sort in
    GPU-index order; e.g. GPU 3's real node was the numerically LOWEST of
    the 64 present, not the 4th). These tests independently recompute the
    "ground truth" GPU-to-render-node mapping from rocm-smi --showbus and
    /sys/class/drm/*/device -- without going through
    amd_render_node_paths_by_pci_bus itself -- and assert the real
    _gpu_flags()/_amd_render_node_paths() output agrees with it for every
    real GPU on this host.
    """

    _PCI_BUS_RE = re.compile(r"([0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f])$")

    def _ground_truth_bus_by_gpu_id(self) -> "dict[int, str]":
        result = subprocess.run(
            ["rocm-smi", "--showbus", "--csv"], capture_output=True, text=True, timeout=10
        )
        assert result.returncode == 0, f"rocm-smi --showbus failed: {result.stderr!r}"
        mapping = {}
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or line.lower().startswith("device"):
                continue
            card, bus = (c.strip() for c in line.split(",")[:2])
            mapping[int(card[4:])] = bus.lower()
        return mapping

    def _real_render_node_bus(self, path: str) -> "str | None":
        name = os.path.basename(path)
        target = os.path.realpath(f"/sys/class/drm/{name}/device")
        match = self._PCI_BUS_RE.search(target)
        return match.group(1) if match else None

    @rocm_hardware
    def test_amd_render_node_paths_match_gpu_pci_bus_on_real_hardware(self):
        _container._gpu_kind_cache.clear()
        _container._amd_render_node_paths_cache = None
        bus_by_gpu_id = self._ground_truth_bus_by_gpu_id()
        assert bus_by_gpu_id, "expected at least one real AMD GPU"

        render_nodes = _container._amd_render_node_paths()
        assert len(render_nodes) == len(bus_by_gpu_id), (
            f"expected one resolved render node per real GPU ({len(bus_by_gpu_id)}), "
            f"got {len(render_nodes)}: {render_nodes}"
        )

        for gpu_id, expected_bus in bus_by_gpu_id.items():
            actual_bus = self._real_render_node_bus(render_nodes[gpu_id])
            assert actual_bus == expected_bus, (
                f"gpu_id={gpu_id}: _amd_render_node_paths() picked "
                f"{render_nodes[gpu_id]} (bus={actual_bus}), but rocm-smi says "
                f"GPU {gpu_id} is actually on bus {expected_bus}"
            )

        # No two GPUs should ever be scoped to the same render node.
        assert len(set(render_nodes)) == len(render_nodes)

    @rocm_hardware
    def test_gpu_flags_attach_every_real_render_node(self, container_runtime):
        """_gpu_flags() attaches every render node _amd_render_node_paths()
        resolves unconditionally -- both so the hibernate model can hand a
        resumed container a different physical GPU without recreating it,
        and so a container created before reserve_gpu() is ever called
        still gets device access (see _gpu_flags()'s docstring in
        container.py)."""
        _container._gpu_kind_cache.clear()
        _container._amd_render_node_paths_cache = None
        bus_by_gpu_id = self._ground_truth_bus_by_gpu_id()
        assert bus_by_gpu_id, "expected at least one real AMD GPU"

        flags = _container._gpu_flags(container_runtime)
        assert flags[:2] == ["--device", "/dev/kfd"]
        device_flags = flags[2:]
        assert device_flags[::2] == ["--device"] * len(bus_by_gpu_id), (
            f"expected one --device pair per real GPU ({len(bus_by_gpu_id)}): {flags}"
        )
        device_paths = device_flags[1::2]
        assert len(set(device_paths)) == len(device_paths), (
            f"no two GPUs should ever resolve to the same render node: {device_paths}"
        )
        for gpu_id, expected_bus in bus_by_gpu_id.items():
            actual_bus = self._real_render_node_bus(device_paths[gpu_id])
            assert actual_bus == expected_bus, (
                f"gpu_id={gpu_id}: _gpu_flags() attached {device_paths[gpu_id]} "
                f"(bus={actual_bus}), but rocm-smi says GPU {gpu_id} is on bus {expected_bus}"
            )


# ===========================================================================
# Private ZFS (cow_zfs) checkpoint transactions over a container backend
# ===========================================================================


@pytest.fixture
def cow_backend(container_runtime):
    """A container backend wired for filesystem-only ZFS checkpoints, with the
    runtime and storage calls mocked."""
    from types import SimpleNamespace

    from agency.sandbox import checkpoint as cp

    cfg = agconfig_cls()
    cfg.sandbox.backend = container_runtime
    cfg.sandbox.checkpoint_backend = "cow_zfs"
    cfg.sandbox.checkpoint_zfs_parent = "tank/agency"
    state = {"running": True, "slots": 1}
    events = []
    sb = _container._ContainerBackendBase.__new__(_container._ContainerBackendBase)
    sb._agconfig = cfg
    sb._runtime = container_runtime
    sb._name = "test-container"
    sb._checkpointer = cp.CowZfsCheckpoint()
    sb._normalized_checkpoint_manifest = MagicMock(return_value={"schema_version": 1})
    sb._container_running = lambda: state["running"]
    sb._register_prof_container = MagicMock()

    def stop():
        events.append("stop")
        state["running"] = False
        state["slots"] -= 1

    def start():
        events.append("start")
        state["running"] = True
        state["slots"] += 1

    sb._stop_image = MagicMock(side_effect=stop)
    sb._ensure_started_profiled = MagicMock(side_effect=start)
    sb._checkpoint_storage = SimpleNamespace(
        prepared=True,
        dataset="tank/agency/private",
        prepare=MagicMock(),
        capture_dataset=MagicMock(return_value="tank/agency/private"),
        snapshot=MagicMock(side_effect=lambda _ref: events.append("snapshot")),
        snapshot_stats=MagicMock(return_value={"zfs_snapshot_used_bytes": 0}),
        rollback=MagicMock(side_effect=lambda _ref: events.append("rollback")),
        delete=MagicMock(),
    )
    return sb, state, events


class TestCowZfsCheckpoint:
    def test_checkpoint_stops_all_processes_before_snapshot_and_restore_starts_fresh(
        self, cow_backend
    ):
        sb, state, events = cow_backend
        handle = sb.checkpoint()
        assert handle.backend == "cow_zfs"
        assert events == ["stop", "snapshot"]
        assert state == {"running": False, "slots": 0}
        sb.stop()  # the engine's commit+stop path must not checkpoint twice
        assert events == ["stop", "snapshot"]
        sb._ensure_started()
        assert events == ["stop", "snapshot", "rollback", "start"]
        assert state == {"running": True, "slots": 1}

    def test_repeated_checkpoints_keep_only_latest_snapshot(self, cow_backend):
        sb, _state, _events = cow_backend
        first = sb.checkpoint()
        sb._ensure_started()
        second = sb.checkpoint()
        sb._checkpoint_storage.delete.assert_called_once_with(first.reference)
        assert sb._checkpointer.latest == second

    def test_snapshot_failure_does_not_publish_checkpoint_or_require_criu(self, cow_backend):
        sb, state, events = cow_backend
        sb._checkpoint_storage.snapshot.side_effect = RuntimeError("ZFS failed")
        with pytest.raises(RuntimeError, match="ZFS failed"):
            sb.checkpoint()
        assert sb._checkpointer.latest is None
        assert state == {"running": False, "slots": 0}
        assert events == ["stop"]

    def test_profiler_has_filesystem_only_component_spans(self, cow_backend, monkeypatch):
        from agency.sandbox import checkpoint as cp

        sb, _state, _events = cow_backend
        spans = []

        @contextmanager
        def span(name):
            spans.append(name)
            yield

        monkeypatch.setattr(cp.agprof, "span", span)
        sb._ensure_started_profiled = MagicMock()
        sb._checkpointer.restore(sb, sb.checkpoint())
        assert {"checkpoint.total", "checkpoint.quiesce", "checkpoint.fs_snapshot"} <= set(spans)
        assert {"restore.total", "restore.fs"} <= set(spans)
        assert not any("criu" in name or "process" in name for name in spans)


# ===========================================================================
# agutil.ensure_python_packages_locally run INSIDE a container (docker and
# podman). Container family because the scenarios use runtime flags
# (``--network none``) and the python image's ``/usr/local/bin/pip3``; the
# function itself has no backend dependence (its mock-level tests are in
# tests/test_agutil.py).
# ===========================================================================

_MISSING_TEST_PACKAGE = "cowsay"  # a real, tiny PyPI package outside agency's own dependency set


_dereferenced_site_packages_dir = None


def _dereferenced_site_packages() -> str:
    """This process's site-packages, with every symlink resolved to a real
    file. uv (this repo's own venv, and CI's) hardlinks or symlinks in
    packages from its cache depending on whether the venv and cache happen
    to share a filesystem in a given environment -- a symlink's target isn't
    reachable once only site-packages itself is bind-mounted into a
    container, so a package uv happened to symlink (rather than hardlink)
    imports fine on one machine and fails on another for a reason with
    nothing to do with the package itself. Copied once per test session
    (cached), not per test -- site-packages can be a couple hundred MB."""
    global _dereferenced_site_packages_dir
    if _dereferenced_site_packages_dir is None:
        import shutil
        import sysconfig
        import tempfile

        src = sysconfig.get_paths()["purelib"]
        dst = tempfile.mkdtemp(prefix="agency-test-site-packages-")
        shutil.copytree(src, dst, dirs_exist_ok=True, symlinks=False)
        _dereferenced_site_packages_dir = dst
    return _dereferenced_site_packages_dir


def _cfg_with_host_site_packages(cfg=None):
    from agency.configs.agconfig import agconfig as agconfig_cls

    cfg = cfg if cfg is not None else agconfig_cls()
    cfg.sandbox.gpu_passthrough = False
    cfg.sandbox.add_mount(
        "host_site_packages", _dereferenced_site_packages(), "/opt/host_site_packages", "ro"
    )
    return cfg


def _run_ensure_packages_locally(sb, packages, *, timeout_s=180, install_missing=True):
    from agency.utils.agutil import AGENCY_PACKAGE_CONTAINER_MOUNT

    script = (
        "from agency.utils.agutil import ensure_python_packages_locally\n"
        f"ensure_python_packages_locally({packages!r}, timeout_s={timeout_s!r}, "
        f"install_missing={install_missing!r})\n"
    )
    sb.write_file("/workspace/_ensure_packages_probe.py", script)
    return sb.exec(
        f"PYTHONPATH={AGENCY_PACKAGE_CONTAINER_MOUNT}:/opt/host_site_packages "
        "python3 /workspace/_ensure_packages_probe.py",
        timeout=timeout_s + 30,
    )


class TestEnsurePythonPackagesInContainer:
    def test_installs_a_missing_package(self, container_kind):
        sb = make_sandbox(container_kind, agconfig=_cfg_with_host_site_packages())
        try:
            _, rc = sb.exec(f'python3 -c "import {_MISSING_TEST_PACKAGE}"')
            assert rc != 0, (
                f"{_MISSING_TEST_PACKAGE} must NOT already be present -- test assumes the gap"
            )

            _, rc = _run_ensure_packages_locally(sb, [_MISSING_TEST_PACKAGE], timeout_s=120)
            assert rc == 0

            _, rc = sb.exec(f'python3 -c "import {_MISSING_TEST_PACKAGE}"')
            assert rc == 0, (
                f"{_MISSING_TEST_PACKAGE} must be importable after ensure_python_packages_locally"
            )
        finally:
            sb.destroy()

    def test_noop_when_already_present(self, container_kind):
        """An already-importable package must not trigger any pip install at
        all -- verified by making pip unusable and confirming that doesn't
        matter. Uses a stdlib module, present in every image."""
        sb = make_sandbox(container_kind, agconfig=_cfg_with_host_site_packages())
        try:
            _, rc = sb.exec('python3 -c "import json"')
            assert rc == 0, "stdlib import must work -- test assumes this baseline"

            # Break pip so any real install attempt would fail loudly.
            sb.exec("mv /usr/local/bin/pip3 /usr/local/bin/pip3.disabled")

            _, rc = _run_ensure_packages_locally(sb, ["json"], timeout_s=30)
            assert rc == 0
        finally:
            sb.destroy()

    def test_raises_on_a_nonexistent_package(self, container_kind):
        sb = make_sandbox(container_kind, agconfig=_cfg_with_host_site_packages())
        try:
            _, rc = _run_ensure_packages_locally(
                sb, ["this_package_definitely_does_not_exist_xyz"], timeout_s=30
            )
            assert rc != 0
        finally:
            sb.destroy()

    def test_raises_before_pip_when_network_unreachable(self, container_kind):
        """A missing package plus no network must raise a diagnosable error.

        Without the probe, pip's own failure is "Could not find a version
        that satisfies the requirement X (from versions: none)", which reads
        like a bad package name rather than a deliberately isolated sandbox.
        The daemon calls this on every launch, so that message would surface
        as every agent failing to start for no visible reason.
        """
        cfg = _cfg_with_host_site_packages()
        cfg.sandbox.flags = ["--network", "none"]
        sb = make_sandbox(container_kind, agconfig=cfg)
        try:
            out, rc = _run_ensure_packages_locally(sb, [_MISSING_TEST_PACKAGE], timeout_s=60)
            assert rc != 0 and "no outbound network" in out
        finally:
            sb.destroy()

    def test_no_network_needed_when_nothing_is_missing(self, container_kind):
        """The no-network path must stay usable when the image already carries
        everything -- this is what makes a --network none sandbox workable at
        all, rather than the probe simply moving the failure earlier.

        pip is broken here too, so reaching either the probe or the install
        would fail rather than silently pass.
        """
        cfg = _cfg_with_host_site_packages()
        cfg.sandbox.flags = ["--network", "none"]
        sb = make_sandbox(container_kind, agconfig=cfg)
        try:
            sb.exec("mv /usr/local/bin/pip3 /usr/local/bin/pip3.disabled")
            _, rc = _run_ensure_packages_locally(sb, ["json", "os"], timeout_s=30)
            assert rc == 0
        finally:
            sb.destroy()

    def test_probe_reports_reachable_with_network(self, container_kind):
        """The probe must not be a blanket "always unreachable" -- otherwise
        the raise above would fire on every ordinary networked sandbox."""
        from agency.utils.agutil import AGENCY_PACKAGE_CONTAINER_MOUNT

        sb = make_sandbox(container_kind, agconfig=_cfg_with_host_site_packages())
        try:
            script = (
                "from agency.utils.agutil import _can_reach_pypi_locally\n"
                "raise SystemExit(0 if _can_reach_pypi_locally() else 1)\n"
            )
            sb.write_file("/workspace/_reach_pypi_probe.py", script)
            _, rc = sb.exec(
                f"PYTHONPATH={AGENCY_PACKAGE_CONTAINER_MOUNT}:/opt/host_site_packages "
                "python3 /workspace/_reach_pypi_probe.py"
            )
            assert rc == 0
        finally:
            sb.destroy()

    def test_installs_only_the_missing_subset(self, container_kind):
        """Mixed request (one present, one missing) must only pip-install
        the missing one -- verified indirectly via the end state (both
        importable afterward), plus the noop-when-present test above
        already covers the "don't touch what's already there" contract
        directly."""
        from agency.utils.agutil import AGENCY_PACKAGE_CONTAINER_MOUNT

        sb = make_sandbox(container_kind, agconfig=_cfg_with_host_site_packages())
        try:
            _, rc = _run_ensure_packages_locally(
                sb, ["httpx", _MISSING_TEST_PACKAGE], timeout_s=120
            )
            assert rc == 0
            # httpx is only ever reachable via the mounted host site-packages
            # (it's never actually pip-installed into the container -- that's
            # the whole point of this test), so this check needs the same
            # PYTHONPATH _run_ensure_packages_locally used, unlike the
            # genuinely-pip-installed _MISSING_TEST_PACKAGE check below.
            _, rc_httpx = sb.exec(
                f"PYTHONPATH={AGENCY_PACKAGE_CONTAINER_MOUNT}:/opt/host_site_packages "
                'python3 -c "import httpx"'
            )
            _, rc_missing = sb.exec(f'python3 -c "import {_MISSING_TEST_PACKAGE}"')
            assert rc_httpx == 0
            assert rc_missing == 0
        finally:
            sb.destroy()
