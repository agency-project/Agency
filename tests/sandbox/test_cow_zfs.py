"""Unit coverage for the runtime-independent parts of filesystem-only ZFS
(``cow_zfs``) checkpoints: the direct ``zfs`` calls, config validation and the
benchmark's accepted layouts.

Runtime-specific storage details (docker's native zfs driver and dataset
capture, podman's private ZFS-clone store and ``--root`` prefix) are in
test_docker_backend.py / test_podman_backend.py; the checkpoint transaction
over a container backend is test_container_backends.py::TestCowZfsCheckpoint;
the CRIU fast-resume flow is test_cow_zfs_fast_resume.py plus the per-runtime
sections of the backend files.
"""

from pathlib import Path
from unittest.mock import Mock

import pytest

from agency.configs.agconfig import agconfig
from agency.sandbox import checkpoint as cp
from agency.sandbox.base import agsandbox_backend


def test_zfs_calls_are_direct_snapshot_and_rollback(monkeypatch, container_runtime):
    storage = cp.ZfsRuntimeStorage.__new__(cp.ZfsRuntimeStorage)
    storage.runtime = container_runtime
    storage.dataset = "tank/private"
    storage.fast_root = Path("/nonexistent/agency-fast-resume")  # docker rollback probes it
    command = Mock()
    monkeypatch.setattr(cp, "_command", command)
    reference = "tank/private@agency-123"
    storage.snapshot(reference)
    storage.rollback(reference)
    storage.delete(reference)
    assert [call.args[0] for call in command.call_args_list] == [
        ["zfs", "snapshot", reference],
        ["zfs", "rollback", reference],
        ["zfs", "destroy", reference],
    ]


def test_config_requires_zfs_parent_for_cow_backend():
    cfg = agconfig()
    cfg.sandbox.checkpoint_backend = "cow_zfs"
    with pytest.raises(ValueError, match="checkpoint_zfs_parent"):
        agsandbox_backend()._validate_config(cfg)


def test_benchmark_accepts_docker_and_podman_cow_layouts():
    runner = pytest.importorskip(
        "benchmarks.checkpoint_size_microbenchmark.runner",
        reason="benchmark sources are maintained outside the Agency repository",
    )
    validate_checkpoint_comparison = runner.validate_checkpoint_comparison

    for runtime in ("docker", "podman"):
        validate_checkpoint_comparison(
            {
                "checkpoint_backend": "cow_zfs",
                "backend": runtime,
                "checkpoint_zfs_parent": "tank/runtime",
            }
        )
