"""CRIU fast-resume (``checkpoint_fast_resume``) over ZFS checkpoints: the parts
that behave identically for docker and podman, run for both runtimes.

The runtime-specific halves of the flow are in test_docker_backend.py
(``docker checkpoint create`` / ``start --checkpoint``, containerd publish-race
retry, ``--network=host`` requirement, runtime bind-file repair) and
test_podman_backend.py (``podman container checkpoint/restore``, catatonit
init, stats keys). Opt-in tests against a real ZFS pool and CRIU are in
test_cow_zfs_linux.py and test_cow_zfs_criu_linux.py.
"""

from unittest.mock import Mock

import pytest

from agency.configs.agconfig import agconfig
from agency.sandbox import checkpoint as cp
from agency.sandbox.base import agsandbox_backend
from tests.sandbox.conftest import make_cow_backend


def test_fast_resume_is_optional_and_requires_cow_zfs():
    cfg = agconfig()
    cfg.sandbox.checkpoint_fast_resume = True
    with pytest.raises(ValueError, match="requires checkpoint_backend='cow_zfs'"):
        agsandbox_backend()._validate_config(cfg)


def test_successful_process_checkpoint_remains_an_acceleration_layer(
    monkeypatch, container_runtime
):
    backend, state = make_cow_backend(container_runtime)
    artifact = {"name": "agency-test", "daemon_count": 1, "live_pty": True}

    def dump(_sandbox):
        state["running"] = False
        return artifact

    restore = Mock(side_effect=lambda _sandbox, _artifact: state.update(running=True))
    monkeypatch.setattr(backend._checkpointer, "_runtime_checkpoint", dump)
    monkeypatch.setattr(backend._checkpointer, "_runtime_restore", restore)

    handle = backend.checkpoint()
    assert handle.stats["fast_resume_available"] is True
    assert handle.stats["fast_resume_live_pty"] is True
    assert backend._checkpoint_storage.snapshot.call_count == 1

    backend._checkpointer.restore(backend, handle)
    restore.assert_called_once_with(backend, artifact)
    assert handle.stats["fast_resume_used"] is True
    assert state["running"] is True


def test_process_dump_failure_still_publishes_filesystem_checkpoint(monkeypatch, container_runtime):
    backend, state = make_cow_backend(container_runtime)
    monkeypatch.setattr(
        backend._checkpointer,
        "_runtime_checkpoint",
        Mock(side_effect=cp.CheckpointCapabilityError("CRIU unavailable")),
    )

    handle = backend.checkpoint()

    assert handle.stats["fast_resume_available"] is False
    assert handle.stats["fast_resume_checkpoint_error"] == "CheckpointCapabilityError"
    backend._stop_image.assert_called_once_with()
    assert state["running"] is False


def test_process_restore_failure_rolls_back_again_for_fresh_start(monkeypatch, container_runtime):
    backend, state = make_cow_backend(container_runtime)
    artifact = {"name": "agency-test", "daemon_count": 1, "live_pty": True}

    def dump(_sandbox):
        state["running"] = False
        return artifact

    monkeypatch.setattr(backend._checkpointer, "_runtime_checkpoint", dump)
    handle = backend.checkpoint()
    monkeypatch.setattr(
        backend._checkpointer,
        "_runtime_restore",
        Mock(side_effect=RuntimeError("restore rejected")),
    )

    backend._checkpointer.restore(backend, handle)

    assert backend._checkpoint_storage.rollback.call_count == 2
    assert handle.stats["fast_resume_used"] is False
    assert handle.stats["fast_resume_restore_error"] == "RuntimeError"
    assert backend._checkpointer.fast_artifact is None
    assert state["running"] is False
