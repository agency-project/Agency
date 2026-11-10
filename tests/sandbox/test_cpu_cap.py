"""The per-sandbox CPU cap (resources.idle_cpus) must reach the container and actually limit it."""

from __future__ import annotations

import os
import subprocess
import uuid
from unittest.mock import patch

import pytest

from agency.configs.agconfig import agconfig, resourcesconfig, sandboxconfig
from agency.sandbox import container
from agency.sandbox.agsandbox import agSandbox


def _docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.mark.parametrize(
    "v1_quota_file_exists, v2_controllers, expected",
    [
        (True, None, True),
        (False, "cpuset cpu io memory pids", True),
        (False, "cpuset io memory pids", False),
        (False, None, False),
    ],
    ids=["cgroup-v1", "cgroup-v2-with-cpu", "cgroup-v2-without-cpu", "no-cgroup-files"],
)
def test_cfs_supported_follows_the_cgroup_files(
    tmp_path, monkeypatch, v1_quota_file_exists, v2_controllers, expected
):
    v1_quota_file = tmp_path / "cfs_quota_us"
    if v1_quota_file_exists:
        v1_quota_file.write_text("-1")
    v2_controllers_file = tmp_path / "cgroup.controllers"
    if v2_controllers is not None:
        v2_controllers_file.write_text(v2_controllers + "\n")
    monkeypatch.setattr(container, "_CGROUP_V1_CFS_QUOTA", str(v1_quota_file))
    monkeypatch.setattr(container, "_CGROUP_V2_CONTROLLERS", str(v2_controllers_file))
    assert container._ContainerBackendBase._cfs_supported() is expected


# Starts busy loops for a few seconds and prints the CPU time the container's cgroup used meanwhile.
_BURN_AND_MEASURE = r"""
usage_usec() { awk '/^usage_usec/ {print $2}' /sys/fs/cgroup/cpu.stat; }
before=$(usage_usec)
for _ in $(seq 8); do timeout 4 sh -c 'while :; do :; done' & done
wait
after=$(usage_usec)
echo "average_cpus_used=$(awk -v b=$before -v a=$after 'BEGIN {printf "%.2f", (a-b)/4000000}')"
"""
_HOST_MEMORY_BYTES = int(open("/proc/meminfo", encoding="utf-8").readline().split()[1]) * 1024
_MEMORY_LIMITS = {"0.5gb": ("512m", 512 * 2**20), "1gb": ("1g", 2**30), "3gb": ("3g", 3 * 2**30)}


def _sandbox(**resources):
    config = agconfig(sandboxconfig(backend="docker"), resourcesconfig(**resources))
    with patch("agency.sandbox.base.shutil.which", return_value="/usr/bin/docker"):
        with patch("agency.sandbox.container._runtime_works", return_value=True):
            sandbox = agSandbox(str(uuid.uuid4()), agconfig=config)
    sandbox._backend._base_image = "docker.io/library/python:3.12-slim"
    return sandbox


needs_docker = pytest.mark.skipif(not _docker_available(), reason="Docker daemon not reachable")


@needs_docker
@pytest.mark.parametrize("cpu_cap", [0.5, 1, 3])
def test_container_cpu_cap_is_applied_and_enforced(cpu_cap):
    if cpu_cap > (os.cpu_count() or 0):
        pytest.skip(f"host has fewer than {cpu_cap} CPUs")
    sandbox = _sandbox(idle_cpus=cpu_cap)
    try:
        limit_text, _ = sandbox.exec("cat /sys/fs/cgroup/cpu.max")
        quota_microseconds, period_microseconds = limit_text.split()
        assert int(quota_microseconds) / int(period_microseconds) == cpu_cap

        measurement, _ = sandbox.exec(_BURN_AND_MEASURE)
        average_cpus_used = float(measurement.split("average_cpus_used=")[1].split()[0])
        assert cpu_cap * 0.3 < average_cpus_used <= cpu_cap * 1.15  # busy, yet never above the cap
    finally:
        sandbox.destroy()


@needs_docker
@pytest.mark.parametrize("memory_name", list(_MEMORY_LIMITS))
def test_container_memory_cap_is_applied(memory_name):
    memory, expected_bytes = _MEMORY_LIMITS[memory_name]
    if expected_bytes > _HOST_MEMORY_BYTES:
        pytest.skip(f"host has less than {memory_name} of memory")
    sandbox = _sandbox(idle_memory=memory)
    try:
        limit_text, _ = sandbox.exec("cat /sys/fs/cgroup/memory.max")
        assert int(limit_text.strip()) == expected_bytes
    finally:
        sandbox.destroy()


@needs_docker
def test_cpu_cap_above_the_host_cpu_count_is_clamped_to_it():
    sandbox = _sandbox(idle_cpus=(os.cpu_count() or 1) + 4)  # the daemon refuses a larger quota
    try:
        limit_text, _ = sandbox.exec("cat /sys/fs/cgroup/cpu.max")
        quota_microseconds, period_microseconds = limit_text.split()
        assert int(quota_microseconds) / int(period_microseconds) == os.cpu_count()
    finally:
        sandbox.destroy()
