"""Base-class tier: logic in sandbox/base.py that is not about any one backend.

Backend selection (agsandbox_backend.for_config(), _auto_detect_runtime()),
checkpoint-kind routing (backend_for_image_kind) and serialization of a backend
for a worker process. No chroot/docker execution required. The concrete
backends these route to are tested in test_docker_backend.py,
test_podman_backend.py and test_chroot_backend.py (plus test_container_backends.py
for what docker and podman share); the GPU-lease bookkeeping every backend
inherits is in test_base_gpu_lease.py and background-process liveness in
test_hibernation_liveness.py.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from agency.sandbox import agsandbox_backend


@pytest.fixture(autouse=True)
def _reset_runtime_cache():
    """get_container_runtime()'s `_RUNTIME` is a module-level, process-wide
    cache (sandbox/container.py) -- set once, reused forever
    after. Constructing a real backend below (test_explicit_docker_builds_
    docker_backend/test_explicit_podman_builds_podman_backend) triggers
    reap_orphaned_containers() -> get_container_runtime() internally,
    which can populate this cache with a value derived from THIS test's
    own mocks. Without resetting it, that value leaks into every other
    test in the whole pytest process afterward -- confirmed to actually
    happen (bisected): a too-broad `shutil.which` mock here made podman
    look available too, caching _RUNTIME='podman' even though this host
    only has docker, breaking unrelated tests elsewhere with
    `FileNotFoundError: podman` far later in the same run. Reset both
    before AND after so a stale value from some earlier, unrelated test
    can't affect these tests' own assertions either.
    """
    import agency.sandbox.container as _container_mod

    _container_mod._RUNTIME = None
    yield
    _container_mod._RUNTIME = None


class TestBackendSelection:
    def test_unknown_backend_raises_value_error(self):
        from agency.configs.agconfig import agconfig, sandboxconfig

        cfg = agconfig(sandboxconfig(backend="not-a-real-backend"))
        with pytest.raises(ValueError, match="Unknown agsandbox_backend.backend"):
            agsandbox_backend.for_config(
                cfg,
                agname="a",
                name="a",
                checkpoint_image=None,
                base_image="x",
                mounts={},
            )

    def test_explicit_docker_raises_when_unusable(self):
        from agency.configs.agconfig import agconfig, sandboxconfig

        cfg = agconfig(sandboxconfig(backend="docker"))
        with patch("agency.sandbox.base.shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="docker"):
                agsandbox_backend.for_config(
                    cfg,
                    agname="a",
                    name="a",
                    checkpoint_image=None,
                    base_image="x",
                    mounts={},
                )

    def test_explicit_chroot_raises_when_unavailable(self):
        from agency.configs.agconfig import agconfig, sandboxconfig

        cfg = agconfig(sandboxconfig(backend="chroot"))
        with patch("agency.sandbox.chroot.chroot_available", return_value=False):
            with pytest.raises(RuntimeError, match="chroot"):
                agsandbox_backend.for_config(
                    cfg,
                    agname="a",
                    name="a",
                    checkpoint_image=None,
                    base_image="x",
                    mounts={},
                )

    def test_explicit_chroot_builds_chroot_backend(self):
        from agency.configs.agconfig import agconfig, sandboxconfig
        from agency.sandbox.chroot import _ChrootBackend

        cfg = agconfig(sandboxconfig(backend="chroot"))
        with patch("agency.sandbox.chroot.chroot_available", return_value=True):
            backend = agsandbox_backend.for_config(
                cfg,
                agname="a",
                name="chroot-select-test",
                checkpoint_image=None,
                base_image="x",
                mounts={},
            )
        assert isinstance(backend, _ChrootBackend)
        backend.destroy()

    def test_explicit_docker_builds_docker_backend(self):
        from agency.configs.agconfig import agconfig, sandboxconfig
        from agency.sandbox.docker import _DockerBackend

        cfg = agconfig(sandboxconfig(backend="docker"))
        with patch("agency.sandbox.container._runtime_works", return_value=True):
            # shutil.which is a single shared module object (both base.py and
            # container.py do plain `import shutil`) -- a return_value= mock
            # here would make EVERY call, for ANY binary name, look truthy,
            # including get_container_runtime()'s own internal "is podman
            # also available?" check (triggered by the real backend
            # construction below, via reap_orphaned_containers()). Argument-
            # aware so only "docker" resolves, matching what this test
            # actually claims to verify.
            with patch(
                "agency.sandbox.base.shutil.which",
                side_effect=lambda name: "/usr/bin/docker" if name == "docker" else None,
            ):
                backend = agsandbox_backend.for_config(
                    cfg,
                    agname="a",
                    name="docker-select-test",
                    checkpoint_image=None,
                    base_image="x",
                    mounts={},
                )
        assert isinstance(backend, _DockerBackend)

    def test_explicit_podman_builds_podman_backend(self):
        from agency.configs.agconfig import agconfig, sandboxconfig
        from agency.sandbox.podman import _PodmanBackend

        cfg = agconfig(sandboxconfig(backend="podman"))
        with patch("agency.sandbox.container._runtime_works", return_value=True):
            # See test_explicit_docker_builds_docker_backend's comment --
            # argument-aware for the same reason (shutil.which is one
            # shared module object, not scoped to base.py).
            with patch(
                "agency.sandbox.base.shutil.which",
                side_effect=lambda name: "/usr/bin/podman" if name == "podman" else None,
            ):
                backend = agsandbox_backend.for_config(
                    cfg,
                    agname="a",
                    name="podman-select-test",
                    checkpoint_image=None,
                    base_image="x",
                    mounts={},
                )
        assert isinstance(backend, _PodmanBackend)

    def test_auto_prefers_podman_then_docker_then_chroot(self):
        from agency.sandbox.base import _auto_detect_runtime

        with patch("agency.sandbox.container.get_container_runtime", return_value="podman"):
            assert _auto_detect_runtime() == "podman"

    def test_auto_falls_back_to_chroot_when_no_container_runtime(self):
        from agency.sandbox.base import _auto_detect_runtime

        with patch(
            "agency.sandbox.container.get_container_runtime",
            side_effect=RuntimeError("no docker/podman"),
        ):
            with patch("agency.sandbox.chroot.chroot_available", return_value=True):
                assert _auto_detect_runtime() == "chroot"

    def test_auto_raises_when_nothing_usable(self):
        from agency.sandbox.base import _auto_detect_runtime

        with patch(
            "agency.sandbox.container.get_container_runtime",
            side_effect=RuntimeError("no docker/podman"),
        ):
            with patch("agency.sandbox.chroot.chroot_available", return_value=False):
                with pytest.raises(RuntimeError, match="No usable sandbox backend"):
                    _auto_detect_runtime()


class TestBackendForImageKind:
    def test_container_kind_returns_container_backend_base(self):
        from agency.sandbox import backend_for_image_kind
        from agency.sandbox.container import _ContainerBackendBase

        assert backend_for_image_kind("container") is _ContainerBackendBase

    def test_chroot_kind_returns_chroot_backend(self):
        from agency.sandbox import backend_for_image_kind
        from agency.sandbox.chroot import _ChrootBackend

        assert backend_for_image_kind("chroot") is _ChrootBackend

    def test_unknown_kind_raises_value_error(self):
        from agency.sandbox import backend_for_image_kind

        with pytest.raises(ValueError, match="Unknown sandbox checkpoint image kind"):
            backend_for_image_kind("not-a-real-kind")

    def test_container_kind_routes_to_the_recorded_runtime(self):
        from agency.sandbox import backend_for_image_kind
        from agency.sandbox.docker import _DockerBackend
        from agency.sandbox.podman import _PodmanBackend

        assert backend_for_image_kind("container", "docker") is _DockerBackend
        assert backend_for_image_kind("container", "podman") is _PodmanBackend

    @pytest.mark.parametrize("runtime", [None, "", "unknown"])
    def test_container_kind_without_a_known_runtime_uses_the_detecting_base(self, runtime):
        from agency.sandbox import backend_for_image_kind
        from agency.sandbox.container import _ContainerBackendBase

        assert backend_for_image_kind("container", runtime) is _ContainerBackendBase

    def test_the_runtime_is_irrelevant_for_a_chroot_checkpoint(self):
        from agency.sandbox import backend_for_image_kind
        from agency.sandbox.chroot import _ChrootBackend

        assert backend_for_image_kind("chroot", "docker") is _ChrootBackend

    def test_the_facade_forwards_the_runtime(self):
        from agency.sandbox.agsandbox import agSandbox
        from agency.sandbox.podman import _PodmanBackend

        assert agSandbox.backend_for_image_kind("container", "podman") is _PodmanBackend

    def test_every_backend_class_declares_its_image_kind(self):
        from agency.sandbox import backend_for_image_kind

        assert backend_for_image_kind("chroot").IMAGE_KIND == "chroot"
        for runtime in ("docker", "podman"):
            assert backend_for_image_kind("container", runtime).IMAGE_KIND == "container"
            assert backend_for_image_kind("container", runtime)._runtime == runtime


class TestBackendSerialization:
    """A backend may be serialized and re-created in a worker process: it must
    pickle (the GPU-lease lock and idle thread are process-local) and the copy
    must have its own working lock."""

    @staticmethod
    def _roundtrip(backend):
        import pickle

        return pickle.loads(pickle.dumps(backend))

    def test_the_copy_keeps_identity_and_has_a_fresh_lock(self, backend_kind):
        from tests.sandbox.conftest import make_backend

        original = make_backend(backend_kind, checkpoint_image="agency/ckpt-x")
        original._real_gpu_holders = {"7"}
        copy = self._roundtrip(original)
        assert copy is not original
        assert (copy._name, copy._agname, copy._checkpoint_image) == (
            original._name,
            original._agname,
            "agency/ckpt-x",
        )
        assert copy._real_gpu_holders == {"7"}
        assert copy._real_gpu_lock is not original._real_gpu_lock
        with copy._real_gpu_lock:
            with copy._real_gpu_lock:  # re-entrant, like the original
                pass

    def test_a_running_idle_release_thread_is_not_carried_over(self, backend_kind):
        import threading

        from tests.sandbox.conftest import make_backend

        original = make_backend(backend_kind)
        original._real_gpu_idle_thread = threading.Thread(target=lambda: None)
        assert self._roundtrip(original)._real_gpu_idle_thread is None
        assert original._real_gpu_idle_thread is not None, "serializing must not mutate the source"

    def test_cloudpickle_round_trips_too(self, backend_kind):
        import cloudpickle

        from tests.sandbox.conftest import make_backend

        copy = cloudpickle.loads(cloudpickle.dumps(make_backend(backend_kind)))
        with copy._real_gpu_lock:
            pass
