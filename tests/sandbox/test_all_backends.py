"""Tier 3: the sandbox-wide ``agSandbox`` contract, run on every backend.

Everything here is behaviour that callers rely on regardless of how isolation
is achieved: exec, file I/O, mounts, isolation, resource handling,
checkpoints, process tracking, networking. Each test is parametrized over
``sandbox_kind`` (chroot, docker, podman) and only uses the public
``agSandbox`` facade (plus the few backend hooks the facade itself exposes,
e.g. ``_backend._lifecycle_tag()``), so the same assertion has to hold for all
of them. A backend that is not usable on the host skips itself.

Where a backend genuinely cannot honour part of the contract (chroot has no
network/PID namespace and no cgroup, for instance) the test calls
``box.requires("<capability>")`` -- see ``BACKEND_CAPABILITIES`` in conftest --
so the gap shows up as a skip with a reason (``pytest -rs``) instead of the
test being dropped or written for one backend only.

Backend- and family-internal mechanisms (docker CLI argument lists, the runtime
semaphore, image GC, process-group bookkeeping...) live in the tier-1/tier-2
files, not here.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import os
import pickle
import socket
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agency.orchestrator.agresources import MIN_CPUS, MIN_MEMORY_MB, agResourcePool
from tests.sandbox.conftest import (
    BACKEND_CAPABILITIES,
    CAPABILITY_DESCRIPTIONS,
    CONTAINER_BACKENDS,
    backend_class,
    checkpoint_exists,
    make_backend,
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


# ---------------------------------------------------------------------------
# The capability table itself
# ---------------------------------------------------------------------------


class TestCapabilityTable:
    def test_every_backend_has_a_well_formed_entry(self):
        assert set(BACKEND_CAPABILITIES) == {"chroot", "docker", "podman"}
        for kind, capabilities in BACKEND_CAPABILITIES.items():
            assert capabilities <= set(CAPABILITY_DESCRIPTIONS), kind

    def test_container_backends_are_one_family(self):
        assert BACKEND_CAPABILITIES["docker"] == BACKEND_CAPABILITIES["podman"]
        assert set(CONTAINER_BACKENDS) == {"docker", "podman"}

    def test_requires_skips_with_a_reason_naming_backend_and_capability(self):
        with pytest.raises(pytest.skip.Exception, match=r"chroot.*network_namespace"):
            requires("chroot", "network_namespace")
        requires("docker", "network_namespace")  # present: must not skip

    def test_unknown_capability_is_an_error_not_a_silent_pass(self):
        with pytest.raises(KeyError):
            requires("chroot", "no_such_capability")


# ---------------------------------------------------------------------------
# exec
# ---------------------------------------------------------------------------


class TestExec:
    def test_simple_command(self, sb):
        out, rc = sb.exec("echo hello")
        assert rc == 0
        assert out.strip() == "hello"

    def test_nonzero_exit_code(self, sb):
        _, rc = sb.exec("exit 42")
        assert rc == 42

    def test_stderr_is_captured_in_the_output(self, sb):
        out, rc = sb.exec("echo err >&2; exit 3")
        assert "err" in out
        assert rc == 3

    def test_stdout_and_stderr_both_reach_the_output(self, sb):
        out, _ = sb.exec("echo to-out; echo to-err >&2")
        assert "to-out" in out and "to-err" in out

    def test_default_workdir_is_workspace(self, sb):
        assert ok(sb, "pwd") == "/workspace"

    def test_workdir_argument(self, sb):
        ok(sb, "mkdir -p /tmp/mydir")
        assert ok(sb, "pwd", workdir="/tmp/mydir") == "/tmp/mydir"

    def test_timeout_returns_minus_one_and_the_sandbox_stays_usable(self, sb):
        started = time.monotonic()
        out, rc = sb.exec("sleep 60", timeout=2)
        assert rc == -1
        assert "timed out" in out.lower()
        assert time.monotonic() - started < 30
        assert ok(sb, "echo still-alive") == "still-alive"

    def test_each_exec_is_a_fresh_shell(self, sb):
        ok(sb, "export AGENCY_TEST_VAR=1; cd /tmp")
        assert ok(sb, "echo [$AGENCY_TEST_VAR]; pwd") == "[]\n/workspace"

    def test_gpus_are_hidden_when_none_is_reserved(self, sb):
        assert ok(sb, "echo $CUDA_VISIBLE_DEVICES") == "NoDevFiles"
        assert ok(sb, "echo $HIP_VISIBLE_DEVICES") == "NoDevFiles"

    def test_quotes_dollars_and_backslashes_survive(self, sb):
        out = ok(sb, "echo 'it'\"'\"'s \"quoted\" $NOT_EXPANDED \\\\ done'")
        assert out == 'it\'s "quoted" $NOT_EXPANDED \\\\ done'

    def test_unicode_output(self, sb):
        assert ok(sb, "echo 'héllo ✓ 日本語'") == "héllo ✓ 日本語"

    def test_multiline_script_and_heredoc(self, sb):
        script = "a=2\nb=3\necho $((a*b))\ncat <<'EOF'\n$literal 'text' \"here\"\nEOF"
        assert ok(sb, script) == "6\n$literal 'text' \"here\""

    def test_pipes_redirects_and_subshells(self, sb):
        out = ok(
            sb, "printf 'b\\na\\nc\\n' | sort > /workspace/sorted.txt; cat /workspace/sorted.txt"
        )
        assert out == "a\nb\nc"
        assert ok(sb, "echo $(echo nested)") == "nested"

    def test_large_output_is_returned_whole(self, sb):
        out = ok(sb, "head -c 300000 /dev/zero | tr '\\0' x")
        assert out == "x" * 300000

    def test_python_is_usable(self, sb):
        assert ok(sb, "python3 -c 'print(1+1)'") == "2"

    def test_dev_null_and_urandom_are_usable(self, sb):
        assert ok(sb, "echo hi > /dev/null && echo OK") == "OK"
        assert ok(sb, "head -c 8 /dev/urandom | wc -c") == "8"

    def test_a_failing_command_does_not_break_later_ones(self, sb):
        _, rc = sb.exec("definitely-not-a-command-xyz")
        assert rc != 0
        assert ok(sb, "echo recovered") == "recovered"

    def test_interactive_harnesses_can_allocate_a_pty(self, sb):
        out = ok(
            sb,
            "python3 -c 'import os; master, slave = os.openpty(); "
            'assert os.isatty(slave); os.close(master); os.close(slave); print("ok")\'',
        )
        assert out == "ok"

    def test_python_semaphores_can_be_created(self, sb):
        out = ok(
            sb,
            "python3 -c 'import multiprocessing; "
            "semaphore = multiprocessing.BoundedSemaphore(1); "
            'assert semaphore.acquire(False); semaphore.release(); print("ok")\'',
        )
        assert out == "ok"

    def test_cuda_env_prefix_for_a_leased_gpu(self, sb):
        sb._real_gpu_ids = [3]
        assert ok(sb, "echo $CUDA_VISIBLE_DEVICES") == "3"
        sb._real_gpu_ids = []

    def test_cuda_env_prefix_for_several_leased_gpus(self, sb):
        sb._real_gpu_ids = [3, 5]
        assert ok(sb, "echo $CUDA_VISIBLE_DEVICES") == "3,5"
        sb._real_gpu_ids = []


# ---------------------------------------------------------------------------
# exec_detached
# ---------------------------------------------------------------------------


class TestExecDetached:
    def test_returns_immediately(self, sb):
        # Warm up: the cold start happens lazily on the first call of any
        # kind and would otherwise be conflated with exec_detached()'s own latency.
        ok(sb, "true")
        started = time.monotonic()
        sb.exec_detached("sleep 8 && touch /workspace/detached_marker.txt")
        elapsed = time.monotonic() - started
        assert elapsed < 4, f"exec_detached() blocked for {elapsed:.2f}s"
        # The command is still sleeping, which proves the call did not wait for it.
        _, rc = sb.exec("test -f /workspace/detached_marker.txt")
        assert rc != 0

    def test_process_runs_to_completion(self, sb):
        sb.exec_detached("sleep 1 && touch /workspace/detached_marker2.txt")
        assert wait_until(
            lambda: sb.exec("test -f /workspace/detached_marker2.txt")[1] == 0, timeout=20
        ), "detached process never created its marker file"

    def test_long_lived_process_outlives_the_call(self, sb):
        sb.exec_detached(_HEARTBEAT_LOOP.format(name="detached_hb"))
        assert heartbeat_advances(sb, "detached_hb")

    def test_workdir(self, sb):
        sb.exec_detached("pwd > /tmp/detached_workdir.txt", workdir="/tmp")
        content = wait_until(
            lambda: sb.exec("cat /tmp/detached_workdir.txt")[0].strip(), timeout=20
        )
        assert content == "/tmp"

    def test_a_failing_detached_command_does_not_raise(self, sb):
        assert sb.exec_detached("exit 3") is None
        assert ok(sb, "echo fine") == "fine"


# ---------------------------------------------------------------------------
# file I/O
# ---------------------------------------------------------------------------


class TestFileIO:
    def test_write_and_read_round_trip(self, sb):
        sb.write_file("/workspace/hello.txt", "hello world\n")
        assert sb.read_file("/workspace/hello.txt") == "hello world\n"

    def test_write_replaces_existing_content(self, sb):
        sb.write_file("/workspace/f.txt", "a long first version\n")
        sb.write_file("/workspace/f.txt", "short\n")
        assert sb.read_file("/workspace/f.txt") == "short\n"

    def test_empty_file(self, sb):
        sb.write_file("/workspace/empty.txt", "")
        assert sb.read_file("/workspace/empty.txt") == ""

    def test_content_without_trailing_newline_is_preserved(self, sb):
        sb.write_file("/workspace/nonl.txt", "no newline")
        assert sb.read_file("/workspace/nonl.txt") == "no newline"

    def test_write_creates_parent_directories(self, sb):
        sb.write_file("/workspace/a/b/c.txt", "deep\n")
        assert sb.read_file("/workspace/a/b/c.txt") == "deep\n"

    def test_write_special_characters(self, sb):
        payload = "line1\nline2\ttab\n$VAR 'quotes' \"double\" `tick` \\back\n"
        sb.write_file("/workspace/special.txt", payload)
        assert sb.read_file("/workspace/special.txt") == payload

    def test_unicode_content(self, sb):
        payload = "héllo ✓ 日本語 🚀\n"
        sb.write_file("/workspace/unicode.txt", payload)
        assert sb.read_file("/workspace/unicode.txt") == payload

    def test_path_with_spaces(self, sb):
        sb.write_file("/workspace/dir with space/file name.txt", "spaced\n")
        assert sb.read_file("/workspace/dir with space/file name.txt") == "spaced\n"
        sb.write_file_bytes("/workspace/other dir/b.bin", b"\x00\x01")
        assert sb.read_file_bytes("/workspace/other dir/b.bin") == b"\x00\x01"

    def test_bytes_round_trip_covers_every_byte_value(self, sb):
        data = bytes(range(256)) * 4
        sb.write_file_bytes("/workspace/bin.dat", data)
        assert sb.read_file_bytes("/workspace/bin.dat") == data

    def test_large_bytes_round_trip(self, sb):
        data = os.urandom(1024 * 1024)
        sb.write_file_bytes("/workspace/big.dat", data)
        assert sb.read_file_bytes("/workspace/big.dat") == data

    def test_write_file_bytes_creates_parent_directories(self, sb):
        sb.write_file_bytes("/workspace/x/y/z.bin", b"\xff\xfe")
        assert sb.read_file_bytes("/workspace/x/y/z.bin") == b"\xff\xfe"

    def test_read_missing_file_raises_file_not_found(self, sb):
        with pytest.raises(FileNotFoundError):
            sb.read_file("/workspace/does_not_exist.txt")
        with pytest.raises(FileNotFoundError):
            sb.read_file_bytes("/workspace/does_not_exist.bin")

    def test_read_directory_raises_is_a_directory_error(self, sb):
        with pytest.raises(IsADirectoryError):
            sb.read_file("/workspace")
        with pytest.raises(IsADirectoryError):
            sb.read_file_bytes("/workspace")

    def test_read_binary_file_as_text_raises_unicode_decode_error(self, sb):
        sb.write_file_bytes("/workspace/binary.bin", bytes([0x89, 0x50, 0x4E, 0x47]))
        with pytest.raises(UnicodeDecodeError):
            sb.read_file("/workspace/binary.bin")
        assert sb.read_file_bytes("/workspace/binary.bin") == bytes([0x89, 0x50, 0x4E, 0x47])

    def test_file_written_by_the_api_is_visible_to_exec(self, sb):
        sb.write_file("/workspace/from_api.txt", "api\n")
        assert ok(sb, "cat /workspace/from_api.txt") == "api"

    def test_file_written_by_exec_is_visible_to_the_api(self, sb):
        ok(sb, "printf 'from-exec' > /workspace/from_exec.txt")
        assert sb.read_file("/workspace/from_exec.txt") == "from-exec"

    def test_remove_files_deletes_files_and_ignores_missing_ones(self, sb):
        sb.write_file("/workspace/gone.txt", "x")
        sb.remove_files(["/workspace/gone.txt", "/workspace/never_existed.txt"])
        with pytest.raises(FileNotFoundError):
            sb.read_file("/workspace/gone.txt")


# ---------------------------------------------------------------------------
# workspace persistence
# ---------------------------------------------------------------------------


class TestWorkspace:
    def test_state_persists_across_exec_calls(self, sb):
        ok(sb, "mkdir -p /workspace/proj && echo one > /workspace/proj/a.txt")
        ok(sb, "echo two >> /workspace/proj/a.txt")
        assert ok(sb, "cat /workspace/proj/a.txt") == "one\ntwo"

    def test_tmp_is_writable_and_persists_across_exec_calls(self, sb):
        ok(sb, "echo scratch > /tmp/agency_scratch.txt")
        assert ok(sb, "cat /tmp/agency_scratch.txt") == "scratch"

    def test_stop_keeps_the_workspace_and_exec_resumes(self, sb):
        sb.write_file("/workspace/persistent.txt", "saved\n")
        sb.stop()
        assert sb._checkpoint_image is None, "stop() must not create a checkpoint"
        assert ok(sb, "cat /workspace/persistent.txt") == "saved"
        assert sb.read_file("/workspace/persistent.txt") == "saved\n"

    def test_repeated_stop_and_resume(self, sb):
        for i in range(3):
            sb.write_file(f"/workspace/round{i}.txt", f"{i}\n")
            sb.stop()
        for i in range(3):
            assert sb.read_file(f"/workspace/round{i}.txt") == f"{i}\n"

    def test_stop_before_anything_started_is_harmless(self, sb):
        sb.stop()
        assert ok(sb, "echo ok") == "ok"


# ---------------------------------------------------------------------------
# mounts
# ---------------------------------------------------------------------------


class TestMounts:
    def test_rw_mount_is_visible_both_ways(self, box, tmp_path):
        host = tmp_path / "shared"
        host.mkdir()
        (host / "from_host.txt").write_text("host-written\n")
        sb = box.sandbox(agconfig=box.config(("shared", host, "/mnt/shared", "rw")))
        assert ok(sb, "cat /mnt/shared/from_host.txt") == "host-written"
        ok(sb, "echo sandbox-written > /mnt/shared/from_sandbox.txt")
        assert (host / "from_sandbox.txt").read_text().strip() == "sandbox-written"
        # And later host edits are live, not a copy taken at start.
        (host / "late.txt").write_text("late\n")
        assert ok(sb, "cat /mnt/shared/late.txt") == "late"

    def test_rw_mount_works_through_the_file_api(self, box, tmp_path):
        host = tmp_path / "shared"
        sb = box.sandbox(agconfig=box.config(("shared", host, "/mnt/shared", "rw")))
        sb.write_file("/mnt/shared/api.txt", "via-api\n")
        assert (host / "api.txt").read_text() == "via-api\n"
        assert sb.read_file("/mnt/shared/api.txt") == "via-api\n"

    def test_missing_host_directory_is_created(self, box, tmp_path):
        host = tmp_path / "does" / "not" / "exist" / "yet"
        sb = box.sandbox(agconfig=box.config(("m", host, "/mnt/m", "rw")))
        ok(sb, "echo hi > /mnt/m/f.txt")
        assert (host / "f.txt").read_text().strip() == "hi"

    def test_ro_mount_is_readable_and_rejects_writes(self, box, tmp_path):
        host = tmp_path / "readonly"
        host.mkdir()
        (host / "data.txt").write_text("protected\n")
        sb = box.sandbox(agconfig=box.config(("ro", host, "/mnt/ro", "ro")))
        assert ok(sb, "cat /mnt/ro/data.txt") == "protected"
        _, rc = sb.exec("echo overwrite > /mnt/ro/data.txt")
        assert rc != 0, "write to a read-only mount must fail"
        _, rc = sb.exec("touch /mnt/ro/new.txt")
        assert rc != 0
        _, rc = sb.exec("rm /mnt/ro/data.txt")
        assert rc != 0
        assert (host / "data.txt").read_text() == "protected\n"
        assert not (host / "new.txt").exists()

    def test_ro_mount_rejects_the_file_api_too(self, box, tmp_path):
        host = tmp_path / "readonly"
        host.mkdir()
        sb = box.sandbox(agconfig=box.config(("ro", host, "/mnt/ro", "ro")))
        with pytest.raises(OSError):
            sb.write_file("/mnt/ro/x.txt", "nope")
        assert not (host / "x.txt").exists()

    def test_mount_into_a_nested_path(self, box, tmp_path):
        host = tmp_path / "nested"
        host.mkdir()
        (host / "n.txt").write_text("nested\n")
        sb = box.sandbox(agconfig=box.config(("n", host, "/mnt/deep/er/path", "rw")))
        assert ok(sb, "cat /mnt/deep/er/path/n.txt") == "nested"
        ok(sb, "echo back > /mnt/deep/er/path/back.txt")
        assert (host / "back.txt").read_text().strip() == "back"

    def test_several_mounts_with_different_modes(self, box, tmp_path):
        rw, ro = tmp_path / "rw", tmp_path / "ro"
        ro.mkdir()
        (ro / "r.txt").write_text("r\n")
        cfg = box.config(("rw", rw, "/mnt/rw", "rw"), ("ro", ro, "/mnt/ro", "ro"))
        sb = box.sandbox(agconfig=cfg)
        ok(sb, "cp /mnt/ro/r.txt /mnt/rw/copied.txt")
        assert (rw / "copied.txt").read_text() == "r\n"
        _, rc = sb.exec("touch /mnt/ro/x")
        assert rc != 0

    def test_mounts_survive_stop_and_resume(self, box, tmp_path):
        host = tmp_path / "shared"
        sb = box.sandbox(agconfig=box.config(("shared", host, "/mnt/shared", "rw")))
        ok(sb, "echo before > /mnt/shared/before.txt")
        sb.stop()
        ok(sb, "echo after > /mnt/shared/after.txt")
        assert (host / "before.txt").read_text().strip() == "before"
        assert (host / "after.txt").read_text().strip() == "after"

    def test_mounts_survive_rm_container_and_restart(self, box, tmp_path):
        host = tmp_path / "shared"
        sb = box.sandbox(agconfig=box.config(("shared", host, "/mnt/shared", "rw")))
        ok(sb, "echo before > /mnt/shared/before.txt")
        sb.rm_container()
        assert ok(sb, "cat /mnt/shared/before.txt") == "before"

    def test_mounted_data_is_not_part_of_a_checkpoint(self, box, tmp_path):
        host = tmp_path / "shared"
        sb = box.sandbox(agconfig=box.config(("shared", host, "/mnt/shared", "rw")))
        ok(sb, "echo in-mount > /mnt/shared/m.txt; echo in-workspace > /workspace/w.txt")
        assert sb.commit(box.tag()) is True
        sb.rm_container()
        # The mount is a live view of the host dir; the workspace comes from the checkpoint.
        assert ok(sb, "cat /mnt/shared/m.txt") == "in-mount"
        assert ok(sb, "cat /workspace/w.txt") == "in-workspace"


# ---------------------------------------------------------------------------
# isolation
# ---------------------------------------------------------------------------


class TestIsolation:
    def test_host_files_outside_mounts_are_invisible(self, sb, tmp_path):
        sentinel = tmp_path / "host_only.txt"
        sentinel.write_text("TOPSECRET-CONTENT\n")
        _, rc = sb.exec(f"test -e {sentinel}")
        assert rc != 0, f"sandbox can see host file {sentinel}"
        out, _ = sb.exec(f"cat {sentinel} 2>&1")
        assert "TOPSECRET-CONTENT" not in out

    def test_host_checkout_is_invisible(self, sb):
        repo_root = Path(__file__).resolve().parents[2]
        _, rc = sb.exec(f"test -e {repo_root}/pyproject.toml")
        assert rc != 0, f"sandbox can see the host checkout at {repo_root}"

    def test_host_home_directory_contents_are_invisible(self, sb):
        home = Path.home()
        entries = [entry.name for entry in home.iterdir()] if home.is_dir() else []
        # Only dotfile-free, distinctive names make a meaningful probe: a
        # container legitimately has its own /root/.bashrc, for example.
        probes = [name for name in entries if not name.startswith(".")][:5]
        if not probes:
            pytest.skip("host home directory has no distinctive entries to probe for")
        for name in probes:
            _, rc = sb.exec(f"test -e {home / name}")
            assert rc != 0, f"sandbox can see host home entry {home / name}"

    def test_writes_outside_mounts_do_not_reach_the_host(self, sb):
        marker = f"agency_escape_{uuid.uuid4().hex}"
        sb.exec(f"echo x > /tmp/{marker}; echo x > /usr/{marker}; echo x > /{marker}")
        assert not Path("/tmp", marker).exists()
        assert not Path("/usr", marker).exists()
        assert not Path("/", marker).exists()

    def test_workspace_writes_do_not_reach_the_host_workspace_path(self, sb):
        marker = f"agency_ws_{uuid.uuid4().hex}.txt"
        sb.write_file(f"/workspace/{marker}", "x")
        assert not (Path("/workspace") / marker).exists()
        assert not (Path.cwd() / marker).exists()

    def test_two_sandboxes_do_not_share_a_workspace(self, box):
        sb1, sb2 = box.sandbox(), box.sandbox()
        sb1.write_file("/workspace/only-in-1.txt", "secret\n")
        ok(sb1, "echo more > /workspace/also-in-1.txt")
        out, _ = sb2.exec("ls /workspace")
        assert "only-in-1.txt" not in out and "also-in-1.txt" not in out
        with pytest.raises(FileNotFoundError):
            sb2.read_file("/workspace/only-in-1.txt")

    def test_two_sandboxes_can_use_the_same_paths_independently(self, box):
        sb1, sb2 = box.sandbox(), box.sandbox()
        sb1.write_file("/workspace/same.txt", "one\n")
        sb2.write_file("/workspace/same.txt", "two\n")
        ok(sb1, "echo 1 > /tmp/same_tmp.txt")
        ok(sb2, "echo 2 > /tmp/same_tmp.txt")
        assert sb1.read_file("/workspace/same.txt") == "one\n"
        assert sb2.read_file("/workspace/same.txt") == "two\n"
        assert ok(sb1, "cat /tmp/same_tmp.txt") == "1"
        assert ok(sb2, "cat /tmp/same_tmp.txt") == "2"

    def test_system_directories_cannot_be_used_to_modify_the_host(self, sb):
        marker = f"agency_bin_{uuid.uuid4().hex}"
        sb.exec(f"touch /bin/{marker} /etc/{marker}")
        assert not Path("/bin", marker).exists()
        assert not Path("/etc", marker).exists()

    def test_host_environment_variables_are_not_inherited(self, box, monkeypatch):
        box.requires("env_isolation")
        monkeypatch.setenv("AGENCY_TEST_HOST_SECRET", "s3cret-value")
        sb = box.sandbox()
        out = ok(sb, "echo [$AGENCY_TEST_HOST_SECRET]")
        assert out == "[]", "host environment variable leaked into the sandbox"

    def test_host_processes_are_invisible(self, box):
        box.requires("pid_namespace")
        marker = f"agencyhostproc{uuid.uuid4().hex}"
        host_proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)", marker])
        try:
            sb = box.sandbox()
            # "[a]..." keeps grep's own cmdline from matching its own pattern.
            out, _ = sb.exec(f"grep -la '[a]{marker[1:]}' /proc/[0-9]*/cmdline; true")
            assert out.strip() == "", f"sandbox can see a host process: {out}"
        finally:
            host_proc.kill()
            host_proc.wait()


# ---------------------------------------------------------------------------
# output_dir sharing
# ---------------------------------------------------------------------------


class TestOutputDir:
    def test_agent_can_write_and_read_its_output_dir(self, box, tmp_path):
        out_dir = tmp_path / "agent_output" / "test-agent"
        sb = box.sandbox(agconfig=box.config(("agent_output", out_dir, "/agent_output", "rw")))
        _, rc = sb.exec("echo hello > /agent_output/result.txt")
        assert rc == 0
        assert (out_dir / "result.txt").read_text().strip() == "hello"
        assert ok(sb, "cat /agent_output/result.txt") == "hello"

    def test_per_agent_output_dirs_are_isolated(self, box, tmp_path):
        out1 = tmp_path / "agent_output" / "brave-fox"
        out2 = tmp_path / "agent_output" / "swift-hawk"
        sb1 = box.sandbox(agconfig=box.config(("agent_output", out1, "/agent_output", "rw")))
        sb2 = box.sandbox(agconfig=box.config(("agent_output", out2, "/agent_output", "rw")))
        ok(sb1, "echo from_agent1 > /agent_output/out.txt")
        assert ok(sb1, "cat /agent_output/out.txt") == "from_agent1"
        assert (out1 / "out.txt").read_text().strip() == "from_agent1"
        _, rc = sb2.exec("test -e /agent_output/out.txt")
        assert rc != 0
        assert not (out2 / "out.txt").exists()

    def test_agents_mounting_the_same_output_dir_share_files(self, box, tmp_path):
        shared = tmp_path / "agent_output" / "shared"
        mount = ("agent_output", shared, "/agent_output", "rw")
        sb1 = box.sandbox(agconfig=box.config(mount))
        sb2 = box.sandbox(agconfig=box.config(mount))
        ok(sb1, "echo from-1 > /agent_output/a.txt")
        assert ok(sb2, "cat /agent_output/a.txt") == "from-1"
        ok(sb2, "echo from-2 > /agent_output/b.txt")
        assert ok(sb1, "cat /agent_output/b.txt") == "from-2"
        assert (shared / "a.txt").read_text().strip() == "from-1"
        assert (shared / "b.txt").read_text().strip() == "from-2"

    def test_output_survives_destroy_of_the_writing_sandbox(self, box, tmp_path):
        out_dir = tmp_path / "agent_output" / "durable"
        sb = box.sandbox(agconfig=box.config(("agent_output", out_dir, "/agent_output", "rw")))
        ok(sb, "echo durable > /agent_output/keep.txt")
        sb.destroy()
        assert (out_dir / "keep.txt").read_text().strip() == "durable"


# ---------------------------------------------------------------------------
# the agency package mount
# ---------------------------------------------------------------------------


class TestAgencyPackageMount:
    def test_package_is_mounted_and_matches_the_host(self, sb):
        from agency.utils.agutil import AGENCY_PACKAGE_CONTAINER_MOUNT, agency_package_dir

        marker_path = f"{AGENCY_PACKAGE_CONTAINER_MOUNT}/agency/agskill.py"
        out, rc = sb.exec(f"test -f {marker_path} && echo yes")
        assert rc == 0 and "yes" in out, f"agskill.py not found at {marker_path}: {out}"

        host_content = (agency_package_dir() / "agency" / "agskill.py").read_text()
        assert sb.read_file(marker_path) == host_content, (
            "the mounted package must be byte-identical to the host's own copy"
        )

    def test_package_mount_is_read_only(self, sb):
        from agency.utils.agutil import AGENCY_PACKAGE_CONTAINER_MOUNT, agency_package_dir

        host_file = agency_package_dir() / "agency" / "agskill.py"
        before = sha256(host_file)
        mount = AGENCY_PACKAGE_CONTAINER_MOUNT
        _, rc = sb.exec(f"touch {mount}/agency/should_not_exist.txt")
        assert rc != 0, "agency package mount must be read-only inside the sandbox"
        _, rc = sb.exec(f"echo tampered >> {mount}/agency/agskill.py")
        assert rc != 0
        assert not (agency_package_dir() / "agency" / "should_not_exist.txt").exists()
        assert sha256(host_file) == before

    def test_package_mount_survives_stop_and_resume(self, sb):
        from agency.utils.agutil import AGENCY_PACKAGE_CONTAINER_MOUNT

        sb.stop()
        _, rc = sb.exec(f"test -f {AGENCY_PACKAGE_CONTAINER_MOUNT}/agency/agskill.py")
        assert rc == 0


# ---------------------------------------------------------------------------
# lifecycle: commit / restore / rm_container / stop / destroy / fork
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_commit_with_an_explicit_tag_creates_a_checkpoint(self, box, sb):
        tag = box.tag()
        sb.write_file("/workspace/marker.txt", "committed\n")
        assert sb.commit(tag) is True
        assert checkpoint_exists(box.kind, tag)
        assert sb._checkpoint_image == tag

    def test_commit_without_a_tag_uses_the_lifecycle_tag_and_keeps_running(self, box, sb):
        sb.write_file("/workspace/marker.txt", "lifecycle\n")
        assert sb.commit() is True
        lifecycle_tag = sb._backend._lifecycle_tag()
        assert sb._checkpoint_image == lifecycle_tag
        assert checkpoint_exists(box.kind, lifecycle_tag)
        # commit() checkpoints in place: the live sandbox is untouched.
        assert sb.read_file("/workspace/marker.txt") == "lifecycle\n"
        assert ok(sb, "echo still-running") == "still-running"

    def test_commit_returns_false_when_nothing_was_ever_started(self, box, sb):
        tag = box.tag()
        assert sb.commit(tag) is False
        assert not checkpoint_exists(box.kind, tag)

    def test_checkpoint_restores_into_a_new_sandbox(self, box):
        tag = box.tag()
        sb1 = box.sandbox()
        sb1.write_file("/workspace/data.txt", "restored\n")
        sb1.commit(tag)
        # sb2 must start from the tag BEFORE sb1.destroy(): destroy() removes
        # the checkpoint sb1 committed.
        sb2 = box.sandbox(checkpoint_image=tag)
        assert sb2.read_file("/workspace/data.txt") == "restored\n"

    def test_checkpoint_does_not_capture_later_changes(self, box):
        tag = box.tag()
        sb1 = box.sandbox()
        sb1.write_file("/workspace/before.txt", "before\n")
        sb1.commit(tag)
        sb1.write_file("/workspace/after.txt", "after\n")
        sb2 = box.sandbox(checkpoint_image=tag)
        assert sb2.read_file("/workspace/before.txt") == "before\n"
        with pytest.raises(FileNotFoundError):
            sb2.read_file("/workspace/after.txt")

    def test_a_checkpoint_can_seed_several_sandboxes_independently(self, box):
        tag = box.tag()
        sb1 = box.sandbox()
        sb1.write_file("/workspace/base.txt", "base\n")
        sb1.commit(tag)
        sb2 = box.sandbox(checkpoint_image=tag)
        sb3 = box.sandbox(checkpoint_image=tag)
        sb2.write_file("/workspace/only2.txt", "2\n")
        assert sb3.read_file("/workspace/base.txt") == "base\n"
        with pytest.raises(FileNotFoundError):
            sb3.read_file("/workspace/only2.txt")

    def test_restore_reverts_to_the_checkpoint(self, box, sb):
        tag = box.tag()
        sb.write_file("/workspace/a.txt", "aaa\n")
        sb.commit(tag)
        sb.write_file("/workspace/b.txt", "bbb\n")
        sb.restore(tag)
        assert sb.read_file("/workspace/a.txt") == "aaa\n"
        with pytest.raises(FileNotFoundError):
            sb.read_file("/workspace/b.txt")

    def test_rm_container_discards_uncommitted_state(self, sb):
        sb.write_file("/workspace/dirty.txt", "dirty\n")
        sb.rm_container()
        with pytest.raises(FileNotFoundError):
            sb.read_file("/workspace/dirty.txt")
        assert ok(sb, "echo restarted") == "restarted"

    def test_rm_container_does_not_create_or_change_a_checkpoint(self, box, sb):
        sb.write_file("/workspace/dirty.txt", "dirty\n")
        assert sb._checkpoint_image is None
        sb.rm_container()
        assert sb._checkpoint_image is None
        assert not checkpoint_exists(box.kind, sb._backend._lifecycle_tag())

    def test_commit_then_rm_container_restores_the_workspace_on_next_start(self, sb):
        sb.write_file("/workspace/persistent.txt", "saved\n")
        sb.commit()
        sb.rm_container()
        assert ok(sb, "cat /workspace/persistent.txt") == "saved"

    def test_rm_container_reverts_to_the_last_checkpoint(self, sb):
        sb.write_file("/workspace/good.txt", "good\n")
        sb.commit()
        sb.rm_container()
        sb.exec("true")  # restarts from the checkpoint
        sb.write_file("/workspace/dirty.txt", "dirty\n")
        sb.rm_container()
        assert sb.read_file("/workspace/good.txt") == "good\n"
        _, rc = sb.exec("test -f /workspace/dirty.txt")
        assert rc != 0, "dirty file must not exist after rm_container()"

    def test_commit_after_commit_keeps_only_the_latest_state(self, sb):
        sb.write_file("/workspace/v.txt", "v1\n")
        sb.commit()
        sb.write_file("/workspace/v.txt", "v2\n")
        sb.commit()
        sb.rm_container()
        assert sb.read_file("/workspace/v.txt") == "v2\n"

    def test_destroy_removes_the_checkpoint(self, box, sb):
        sb.write_file("/workspace/x.txt", "x\n")
        sb.commit()
        lifecycle_tag = sb._backend._lifecycle_tag()
        assert checkpoint_exists(box.kind, lifecycle_tag)
        sb.destroy()
        assert not checkpoint_exists(box.kind, lifecycle_tag)

    def test_destroy_is_idempotent(self, sb):
        sb.write_file("/workspace/x.txt", "x\n")
        sb.destroy()
        sb.destroy()

    def test_destroy_of_a_sandbox_that_never_started_is_harmless(self, sb):
        sb.destroy()
        sb.destroy()

    def test_destroy_after_stop_and_after_rm_container(self, sb):
        sb.write_file("/workspace/x.txt", "x\n")
        sb.stop()
        sb.rm_container()
        sb.destroy()
        sb.destroy()

    def test_destroy_does_not_leave_the_workspace_for_the_next_sandbox(self, box):
        sb1 = box.sandbox()
        sb1.write_file("/workspace/leftover.txt", "x\n")
        sb1.destroy()
        sb2 = box.sandbox()
        with pytest.raises(FileNotFoundError):
            sb2.read_file("/workspace/leftover.txt")

    # -- a second backend object with the same identity (a worker-process copy)

    def test_second_view_reuses_the_workspace_instead_of_wiping_it(self, sb):
        sb.write_file("/workspace/persist.txt", "still-here\n")
        view = backend_view(sb)
        view._ensure_started()
        assert sb.read_file("/workspace/persist.txt") == "still-here\n"
        assert view.read_file("/workspace/persist.txt") == "still-here\n"

    def test_commit_works_from_a_view_that_never_started_the_sandbox_itself(self, box, sb):
        tag = box.tag()
        sb.write_file("/workspace/marker.txt", "worker-written\n")
        view = backend_view(sb)
        assert view.commit(tag) is True, "commit() must find the already-running sandbox"
        assert checkpoint_exists(box.kind, tag)

    def test_ensure_started_does_not_wipe_workspace_across_worker_processes(self, sandbox_kind):
        """Serialized backend copies must reuse the same committed workspace."""
        orig = make_backend(sandbox_kind)
        try:
            worker1 = pickle.loads(pickle.dumps(orig))
            worker1.write_file("/workspace/inputs/full_text_123.txt", "important content\n")
            worker2 = pickle.loads(pickle.dumps(orig))
            content = worker2.read_file("/workspace/inputs/full_text_123.txt")
            assert content == "important content\n"
        finally:
            orig.destroy()

    def test_commit_commits_even_when_this_process_never_started(self, sandbox_kind):
        """commit() on a sandbox whose workspace was written entirely by a
        worker-process copy must still checkpoint that work."""
        orig = make_backend(sandbox_kind)
        restored = None
        try:
            worker = pickle.loads(pickle.dumps(orig))
            worker.write_file("/workspace/data.txt", "from-worker\n")

            assert orig.commit() is True, "commit() must have committed"
            assert orig._checkpoint_image is not None

            restored = make_backend(sandbox_kind, checkpoint_image=orig._checkpoint_image)
            assert restored.read_file("/workspace/data.txt") == "from-worker\n"
        finally:
            if restored is not None:
                restored.destroy()
            if orig._checkpoint_image:
                backend_class(sandbox_kind).delete_image(orig._checkpoint_image, force=True)
            orig.destroy()

    # -- checkpoint() handle API

    def test_checkpoint_handle_round_trip(self, box, sb):
        box.requires("checkpoint_handle")
        sb.write_file("/workspace/kept.txt", "kept\n")
        handle = sb.checkpoint()
        assert handle is not None
        sb.write_file("/workspace/after.txt", "after\n")
        sb.restore(handle)
        assert sb.read_file("/workspace/kept.txt") == "kept\n"
        with pytest.raises(FileNotFoundError):
            sb.read_file("/workspace/after.txt")
        sb.delete_checkpoint(handle)
        assert not checkpoint_exists(box.kind, handle.reference)

    # -- fork

    def test_fork_preserves_checkpoint_content(self, box, sb):
        sb.write_file("/workspace/parent.txt", "parent-data\n")
        sb.commit()
        fork_sb = box.adopt(sb.fork(str(uuid.uuid4())))
        assert type(fork_sb._backend) is type(sb._backend)
        assert fork_sb.read_file("/workspace/parent.txt") == "parent-data\n"
        assert fork_sb._checkpoint_image == fork_sb._backend._lifecycle_tag()

    def test_fork_and_parent_diverge_independently(self, box, sb):
        sb.write_file("/workspace/shared.txt", "base\n")
        sb.commit()
        fork_sb = box.adopt(sb.fork(str(uuid.uuid4())))
        fork_sb.write_file("/workspace/fork_only.txt", "fork\n")
        sb.write_file("/workspace/parent_only.txt", "parent\n")
        assert fork_sb.read_file("/workspace/shared.txt") == "base\n"
        with pytest.raises(FileNotFoundError):
            fork_sb.read_file("/workspace/parent_only.txt")
        with pytest.raises(FileNotFoundError):
            sb.read_file("/workspace/fork_only.txt")

    def test_fork_without_a_checkpoint_starts_fresh(self, box, sb):
        sb.write_file("/workspace/uncommitted.txt", "x\n")
        fork_sb = box.adopt(sb.fork(str(uuid.uuid4())))
        with pytest.raises(FileNotFoundError):
            fork_sb.read_file("/workspace/uncommitted.txt")
        assert ok(fork_sb, "echo ok") == "ok"

    def test_fork_survives_destroying_the_parent(self, box, sb):
        sb.write_file("/workspace/parent.txt", "parent-data\n")
        sb.commit()
        fork_sb = box.adopt(sb.fork(str(uuid.uuid4())))
        sb.destroy()
        assert fork_sb.read_file("/workspace/parent.txt") == "parent-data\n"

    # -- checkpoint image helpers (tag / export / import)

    def test_tag_image_copies_a_checkpoint(self, box, sb):
        src_tag, dest_tag = box.tag("src"), box.tag("dest")
        sb.write_file("/workspace/x.txt", "x\n")
        sb.commit(src_tag)
        backend_class(box.kind).tag_image(src_tag, dest_tag)
        sb2 = box.sandbox(checkpoint_image=dest_tag)
        assert sb2.read_file("/workspace/x.txt") == "x\n"

    def test_export_then_import_round_trips_a_checkpoint(self, box, sb):
        tag = box.tag("export")
        cls = backend_class(box.kind)
        sb.write_file("/workspace/e.txt", "exported\n")
        sb.commit(tag)
        blob = cls.export_image(tag, 300)
        assert isinstance(blob, bytes) and len(blob) > 0
        cls.delete_image(tag, force=True)
        assert not checkpoint_exists(box.kind, tag)
        cls.import_image(blob, 300)
        assert checkpoint_exists(box.kind, tag)
        sb2 = box.sandbox(checkpoint_image=tag)
        assert sb2.read_file("/workspace/e.txt") == "exported\n"

    def test_image_kind_routes_back_to_the_backend_that_wrote_it(self, sb):
        from agency.sandbox.agsandbox import agSandbox

        backend_cls = agSandbox.backend_for_image_kind(sb.image_kind)
        assert isinstance(sb._backend, backend_cls)

    def test_facade_construction_is_lazy(self, box):
        sb = box.sandbox()
        assert not sb._destroyed
        # Nothing was run yet: committing a never-started sandbox is a no-op.
        assert sb.commit(box.tag()) is False


# ---------------------------------------------------------------------------
# agent.save() / agent.load() round trip
# ---------------------------------------------------------------------------


class TestAgentSaveLoad:
    def _agconfig(self, kind):
        from agency.configs.agconfig import agconfig, llmconfig, sandboxconfig

        return agconfig(sandboxconfig(backend=kind), llmconfig(api_key="k", model="m"))

    def test_save_records_the_checkpoint_image_kind(self, box, tmp_path):
        import json
        import tarfile

        from agency.agent import agent
        from agency.agname import agname
        from agency.sandbox.agsandbox import agSandbox

        cfg = self._agconfig(box.kind)
        ag = agent(agconfig=cfg)
        try:
            ag.sandbox = box.adopt(agSandbox(ag.agname, agconfig=cfg))
            ag.sandbox.write_file("/workspace/marker.txt", "data\n")
            ag.sandbox.commit()

            ckpt = tmp_path / "agent.ckpt"
            ag.save(ckpt)

            with tarfile.open(ckpt, "r:gz") as tar:
                state = json.loads(tar.extractfile("state.json").read())
            assert state["sandbox_image_kind"] == ag.sandbox.image_kind
            assert state["sandbox_image_kind"] == (
                "chroot" if box.kind == "chroot" else "container"
            )
            # docker/podman checkpoints name the runtime that produced them
            # so load() restores through it; chroot has no runtime.
            assert state.get("sandbox_runtime") == (None if box.kind == "chroot" else box.kind)
        finally:
            agname._allocated.discard(str(ag.agname))

    def test_save_then_load_restores_the_workspace(self, box, tmp_path):
        from agency.agent import agent
        from agency.agname import agname
        from agency.sandbox.agsandbox import agSandbox

        cfg = self._agconfig(box.kind)
        ag = agent(agconfig=cfg)
        ag2 = None
        try:
            ag.sandbox = box.adopt(agSandbox(ag.agname, agconfig=cfg))
            ag.sandbox.write_file("/workspace/marker.txt", "checkpointed-via-agent\n")
            ag.sandbox.commit()

            ckpt = tmp_path / "agent.ckpt"
            ag.save(ckpt)
            saved_agname = str(ag.agname)
            ag.sandbox.destroy()
            agname._allocated.discard(saved_agname)

            ag2 = agent.load(ckpt, agconfig=cfg)
            box.adopt(ag2.sandbox)
            assert type(ag2.sandbox._backend) is type(ag.sandbox._backend)
            assert ag2.sandbox.read_file("/workspace/marker.txt") == "checkpointed-via-agent\n"
        finally:
            if ag2 is not None:
                agname._allocated.discard(str(ag2.agname))


# ---------------------------------------------------------------------------
# background-process tracking
# ---------------------------------------------------------------------------


class TestProcessTracking:
    def test_background_pid_is_tracked(self, sb):
        sb.exec("sleep 30 &")
        assert len(sb.get_live_pids()) > 0

    def test_foreground_spawned_child_is_tracked(self, sb):
        # A foreground command that forks a child and exits: the child escapes
        # the shell's job table but must still be tracked.
        sb.write_file(
            "/workspace/spawner.py",
            "import subprocess\nsubprocess.Popen(['sleep', '30'])\n",
        )
        sb.exec("python3 /workspace/spawner.py")
        assert len(sb.get_live_pids()) > 0

    def test_child_of_an_exited_parent_is_still_tracked(self, sb):
        # The orphaned child is reparented to init and escapes any walk down
        # from the original pid.
        sb.write_file(
            "/workspace/spawner.py",
            "import subprocess, os\nsubprocess.Popen(['sleep', '30'])\nos._exit(0)\n",
        )
        sb.exec("python3 /workspace/spawner.py")
        assert len(sb.get_live_pids()) > 0

    def test_process_tree_descendants_are_tracked(self, sb):
        sb.write_file(
            "/workspace/parent.py",
            (
                "import subprocess, time\n"
                "time.sleep(0.2)\n"
                "subprocess.Popen(['sleep', '30'])\n"
                "subprocess.Popen(['sleep', '30'])\n"
                "time.sleep(30)\n"
            ),
        )
        sb.exec("python3 /workspace/parent.py &")
        # parent + 2 children, however long the interpreter takes to start.
        assert wait_until(lambda: len(sb.get_live_pids()) >= 3, timeout=20)

    def test_multiple_children_spawned_within_one_call_are_tracked(self, sb):
        sb.write_file(
            "/workspace/parent.py",
            (
                "import subprocess\n"
                "subprocess.Popen(['sleep', '30'])\n"
                "subprocess.Popen(['sleep', '30'])\n"
            ),
        )
        sb.exec("python3 /workspace/parent.py")
        assert len(sb.get_live_pids()) >= 2

    def test_double_forked_daemon_is_tracked(self, sb):
        # Classic double fork: the grandchild is reparented to init and detached
        # from the shell's job table.
        sb.write_file(
            "/workspace/daemon.py",
            (
                "import os, time\n"
                "if os.fork() == 0:\n"
                "    if os.fork() == 0:\n"
                "        time.sleep(30)\n"
                "    os._exit(0)\n"
                "os.wait()\n"
            ),
        )
        sb.exec("python3 /workspace/daemon.py")
        assert len(sb.get_live_pids()) > 0

    def test_get_live_pids_empty_when_nothing_runs_in_the_background(self, sb):
        sb.exec("echo hi")
        assert sb.get_live_pids() == set()
        assert sb._has_pending_background_work() is False

    def test_exited_background_process_stops_being_live(self, sb):
        # 10s is sized for get_live_pids() itself, not exec(): on chroot the
        # scan walks every process on the host, and a process that exits
        # before the scan reaches it reads as "not tracked" rather than as slow.
        sb.exec("sleep 10 >/dev/null 2>&1 &")
        assert sb.get_live_pids(), "expected the background sleep to be tracked"
        assert sb._has_pending_background_work() is True
        assert wait_until(lambda: not sb.get_live_pids(), timeout=60, interval=1.0), (
            f"background sleep never stopped being live: {sb.get_live_pids()}"
        )
        assert sb._has_pending_background_work() is False

    def test_released_daemon_is_no_longer_monitored(self, sb):
        sb.exec("sleep 30 &")
        live = sb.get_live_pids()
        assert len(live) > 0
        for pid in list(live):
            sb.release_daemon(pid)
        assert sb.get_live_pids() == set()
        assert sb._has_pending_background_work() is False

    def test_children_of_a_released_daemon_are_excluded_too(self, sb):
        # Release the parent before its child even exists; a child it spawns
        # later must inherit the daemon status.
        sb.write_file(
            "/workspace/daemon_parent.py",
            (
                "import subprocess, time\n"
                "time.sleep(0.5)\n"
                "subprocess.Popen(['sleep', '30'])\n"
                "time.sleep(30)\n"
            ),
        )
        sb.exec("python3 /workspace/daemon_parent.py &")
        parent_pids = sb.get_live_pids()
        assert parent_pids, "expected the backgrounded parent to be tracked"
        for pid in parent_pids:
            sb.release_daemon(pid)
        time.sleep(1.5)  # let the child spawn, now that its parent is already a daemon
        assert sb.get_live_pids() == set()

    def test_pid_status_summary_with_no_processes(self, sb):
        assert "no background" in sb.pid_status_summary()

    def test_pid_status_summary_lists_a_running_process(self, sb):
        sb.exec("sleep 30 &")
        assert "PID" in sb.pid_status_summary()

    def test_pid_status_summary_reports_elapsed_time(self, box, sb):
        box.requires("pid_elapsed_summary")
        sb.exec("sleep 30 &")
        summary = sb.pid_status_summary()
        assert "PID" in summary and "running" in summary

    def test_stop_kills_background_processes_but_keeps_the_workspace(self, sb):
        sb.write_file("/workspace/kept.txt", "kept\n")
        sb.exec(f"({_HEARTBEAT_LOOP.format(name='hb')}) >/dev/null 2>&1 &")
        assert heartbeat_advances(sb)
        sb.stop()
        assert sb.get_live_pids() == set()
        # After resuming, the writer must be gone for good while files remain.
        assert sb.read_file("/workspace/kept.txt") == "kept\n"
        assert not heartbeat_advances(sb), "background process survived stop()"

    def test_rm_container_kills_background_processes(self, sb):
        sb.exec(f"({_HEARTBEAT_LOOP.format(name='hb')}) >/dev/null 2>&1 &")
        assert heartbeat_advances(sb)
        sb.commit()  # so the heartbeat file survives the rm_container() below
        sb.rm_container()
        assert sb.get_live_pids() == set()
        assert not heartbeat_advances(sb), "background process survived rm_container()"

    # -- wait_for_processes (the facade's "is the sandbox clean?" check)

    def test_wait_for_processes_returns_none_when_clean(self, sb):
        sb.exec("echo hi")
        assert sb.wait_for_processes("skill", None, ping_interval_s=1, poll_interval_s=0.2) is None

    def test_wait_for_processes_reports_a_still_running_process(self, sb):
        sb.exec("sleep 60 &")
        msg = sb.wait_for_processes("skill", None, ping_interval_s=1, poll_interval_s=0.2)
        assert msg is not None
        assert "still running" in msg
        assert "daemon_release" in msg

    def test_wait_for_processes_notices_a_process_finishing(self, sb):
        sb.exec("sleep 3 >/dev/null 2>&1 &")
        msg = sb.wait_for_processes("skill", None, ping_interval_s=60, poll_interval_s=0.5)
        assert msg is not None
        assert "completed" in msg

    def test_wait_for_processes_ignores_released_daemons(self, sb):
        sb.exec("sleep 60 &")
        for pid in list(sb.get_live_pids()):
            sb.release_daemon(pid)
        assert sb.wait_for_processes("skill", None, ping_interval_s=1, poll_interval_s=0.2) is None

    def test_wait_for_processes_sees_a_child_spawned_after_exec_returned(self, sb):
        sb.write_file(
            "/workspace/parent.py",
            (
                "import subprocess, os, time\n"
                "time.sleep(0.5)\n"  # spawn after exec()'s own bookkeeping already ran
                "subprocess.Popen(['sleep', '30'])\n"
                "os._exit(0)\n"  # parent exits at once, the child detaches
            ),
        )
        sb.exec("python3 /workspace/parent.py &")
        time.sleep(1.5)
        assert sb.get_live_pids(), "expected the delayed child to still be found"
        assert sb._has_pending_background_work() is True
        msg = sb.wait_for_processes("skill", None, ping_interval_s=1, poll_interval_s=0.2)
        assert msg is not None, "reported clean while a background process was still running"


# ---------------------------------------------------------------------------
# ptrace pid ingest (the alternate pid-population path for harness agents)
# ---------------------------------------------------------------------------


class TestIngestPtracePids:
    """``ingest_ptrace_pids()`` pids are trusted regardless of what the
    sandbox's own /proc scan shows, since a harness's ptrace events are
    authoritative and generally describe a different PID namespace."""

    @pytest.fixture(autouse=True)
    def _require_capability(self, box):
        box.requires("ptrace_ingest")

    def test_ingested_pid_appears_live(self, sb):
        sb._backend.ingest_ptrace_pids(spawned={999001})
        assert 999001 in sb.get_live_pids()

    def test_ingested_pid_that_exited_is_removed(self, sb):
        sb._backend.ingest_ptrace_pids(spawned={999002})
        assert 999002 in sb.get_live_pids()
        sb._backend.ingest_ptrace_pids(exited={999002})
        assert 999002 not in sb.get_live_pids()

    def test_ptrace_pid_is_not_pruned_by_proc_scan_absence(self, sb):
        sb._backend.ingest_ptrace_pids(spawned={999003})
        for _ in range(3):
            assert 999003 in sb.get_live_pids()

    def test_ingest_coexists_with_a_real_exec_tracked_pid(self, sb):
        sb.exec("sleep 30 &")
        sb._backend.ingest_ptrace_pids(spawned={999004})
        live = sb.get_live_pids()
        assert 999004 in live
        assert len(live) >= 2  # the real sleep + the injected ptrace pid

    def test_release_daemon_also_clears_a_ptrace_managed_pid(self, sb):
        sb._backend.ingest_ptrace_pids(spawned={999005})
        sb.release_daemon(999005)
        assert 999005 not in sb.get_live_pids()


# ---------------------------------------------------------------------------
# resource handling
# ---------------------------------------------------------------------------


class TestResourceLimits:
    def test_update_limits_cpu_before_start(self, sb):
        sb.update_limits(cpus=2.0)

    def test_update_limits_on_a_running_sandbox(self, sb):
        ok(sb, "true")
        sb.update_limits(cpus=2.0)
        sb.update_limits(memory="256m")
        sb.update_limits(cpus=1.0, memory="128m")
        assert ok(sb, "echo still-works") == "still-works"

    def test_update_limits_with_nothing_to_change(self, sb):
        ok(sb, "true")
        sb.update_limits()

    def test_resources_config_is_accepted_and_the_sandbox_runs(self, box):
        sb = box.sandbox(agconfig=box.config(idle_cpus=1, idle_memory="512m"))
        assert ok(sb, "echo configured") == "configured"

    @pytest.mark.parametrize("cpu_cap", [0.5, 1, 3])
    def test_cpu_cap_is_applied_and_enforced(self, box, cpu_cap):
        box.requires("cgroup_limits")
        if cpu_cap > (os.cpu_count() or 0):
            pytest.skip(f"host has fewer than {cpu_cap} CPUs")
        sb = box.sandbox(agconfig=box.config(idle_cpus=cpu_cap))
        limit_text = ok(sb, "cat /sys/fs/cgroup/cpu.max")
        quota_microseconds, period_microseconds = limit_text.split()
        assert int(quota_microseconds) / int(period_microseconds) == cpu_cap

        measurement = ok(sb, _BURN_AND_MEASURE)
        average_cpus_used = float(measurement.split("average_cpus_used=")[1].split()[0])
        assert cpu_cap * 0.3 < average_cpus_used <= cpu_cap * 1.15  # busy, yet never above the cap

    def test_update_limits_reaches_the_cgroup_cpu_quota(self, box, sb):
        box.requires("cgroup_limits")
        if not backend_class(box.kind)._cfs_supported():
            pytest.skip("kernel cannot enforce a CPU quota here (no cgroup cpu controller)")
        ok(sb, "true")
        sb.update_limits(cpus=1.0)
        quota, period = ok(sb, "cat /sys/fs/cgroup/cpu.max").split()
        assert int(quota) / int(period) == min(1.0, os.cpu_count() or 1)

    def test_update_limits_reaches_the_cgroup_memory_limit(self, box, sb):
        box.requires("cgroup_limits")
        ok(sb, "true")
        sb.update_limits(memory="256m")
        assert int(ok(sb, "cat /sys/fs/cgroup/memory.max")) == 256 * 2**20

    @pytest.mark.parametrize(
        ("memory", "expected_bytes"),
        [("512m", 512 * 2**20), ("1g", 2**30), ("3g", 3 * 2**30)],
        ids=["0.5gb", "1gb", "3gb"],
    )
    def test_memory_cap_is_applied(self, box, memory, expected_bytes):
        box.requires("cgroup_limits")
        host_memory = int(open("/proc/meminfo", encoding="utf-8").readline().split()[1]) * 1024
        if expected_bytes > host_memory:
            pytest.skip(f"host has less than {memory} of memory")
        sb = box.sandbox(agconfig=box.config(idle_memory=memory))
        assert int(ok(sb, "cat /sys/fs/cgroup/memory.max")) == expected_bytes

    def test_cpu_cap_above_the_host_cpu_count_is_clamped_to_it(self, box):
        box.requires("cgroup_limits")
        cap = (os.cpu_count() or 1) + 4  # the daemon refuses a larger quota
        sb = box.sandbox(agconfig=box.config(idle_cpus=cap))
        quota_microseconds, period_microseconds = ok(sb, "cat /sys/fs/cgroup/cpu.max").split()
        assert int(quota_microseconds) / int(period_microseconds) == os.cpu_count()

    def test_cpu_and_memory_bookkeeping_through_the_pool(self, sb):
        pool = agResourcePool(gpus=[0], idle_cpus=1.0, idle_memory="1024m")
        ok(sb, "true")
        pool.acquire_cpu_mem(sb, 2.0, 256)
        assert sb._cpu_acquired == max(2.0, MIN_CPUS)
        assert sb._memory_acquired_mb == max(256, MIN_MEMORY_MB)
        pool.release_cpu_mem(sb, cpu=True, memory=True)
        assert sb._cpu_acquired == 0.0
        assert sb._memory_acquired_mb == 0
        assert ok(sb, "echo fine") == "fine"

    def test_release_resources_without_a_pool_does_not_raise(self, sb):
        sb.release_virtual_and_real_resources(None)

    def test_release_resources_returns_a_leased_gpu_to_the_pool(self, box):
        sb = box.sandbox(agconfig=box.config(gpu=True))
        pool = agResourcePool(gpus=[0])
        pool.acquire_real_gpus(sb, 1)
        sb._virtual_gpu_count_requested = 1
        assert pool._gpus_acquired == 1
        sb.release_virtual_and_real_resources(pool)
        assert sb._real_gpu_ids == []
        assert sb._virtual_gpu_count_requested == 0
        assert pool._gpus_acquired == 0

    def test_release_resources_without_a_reservation_does_not_raise(self, box):
        sb = box.sandbox(agconfig=box.config(gpu=True))
        pool = agResourcePool(gpus=[0, 1])
        sb.release_virtual_and_real_resources(pool)
        assert sb._virtual_gpu_count_requested == 0
        assert sb._real_gpu_ids == []

    def test_cpu_only_sandbox_forbids_gpu_reservation(self, sb):
        with pytest.raises(RuntimeError, match="CPU-only"):
            sb._virtual_gpu_count_requested = 1
        assert sb._virtual_gpu_count_requested == 0


# Starts busy loops for a few seconds and prints the CPU time the container's cgroup used meanwhile.
_BURN_AND_MEASURE = r"""
usage_usec() { awk '/^usage_usec/ {print $2}' /sys/fs/cgroup/cpu.stat; }
before=$(usage_usec)
for _ in $(seq 8); do timeout 4 sh -c 'while :; do :; done' & done
wait
after=$(usage_usec)
echo "average_cpus_used=$(awk -v b=$before -v a=$after 'BEGIN {printf "%.2f", (a-b)/4000000}')"
"""


class TestGpuLeasing:
    """Physical-GPU leasing through the facade, against a fake pool (no real
    GPU needed: only the lease bookkeeping and the visible-devices env var are
    observable)."""

    @pytest.fixture
    def pool(self):
        return agResourcePool(gpus=[0, 1], idle_cpus=1.0, idle_memory="1024m")

    @pytest.fixture
    def gsb(self, box):
        """A sandbox with GPU passthrough enabled (the default config)."""
        return box.sandbox(agconfig=box.config(gpu=True))

    @staticmethod
    def reserve(sandbox, pool, count: int = 1) -> None:
        """A virtual-only reservation, no physical GPU claimed yet -- the same
        thing agskill._reserve_virtual_resource does."""
        sandbox._virtual_gpu_count_requested = count
        sandbox._real_gpu_acquire_fn = functools.partial(pool.acquire_real_gpus, sandbox)
        sandbox._real_gpu_release_fn = functools.partial(pool.release_real_gpus, sandbox)

    def test_exec_leases_a_physical_gpu_when_requested(self, gsb, pool):
        self.reserve(gsb, pool)
        assert pool._gpus_acquired == 0
        assert ok(gsb, "echo $CUDA_VISIBLE_DEVICES").isdigit()

    def test_exec_sets_both_visible_devices_variables(self, gsb, pool):
        self.reserve(gsb, pool)
        out = ok(gsb, "echo $CUDA_VISIBLE_DEVICES-$HIP_VISIBLE_DEVICES")
        cuda, hip = out.split("-")
        assert cuda.isdigit() and cuda == hip

    def test_exec_lists_every_leased_gpu(self, gsb, pool):
        self.reserve(gsb, pool, count=2)
        ids = ok(gsb, "echo $CUDA_VISIBLE_DEVICES").split(",")
        assert sorted(int(i) for i in ids) == [0, 1]

    def test_foreground_exec_releases_the_physical_gpu_when_it_ends(self, gsb, pool):
        self.reserve(gsb, pool)
        gsb.exec("echo hello")
        assert gsb._real_gpu_ids == []
        assert pool._gpus_acquired == 0
        assert gsb._virtual_gpu_count_requested > 0

    def test_consecutive_foreground_execs_each_lease_and_release(self, gsb, pool):
        self.reserve(gsb, pool)
        for _ in range(3):
            assert ok(gsb, "echo $CUDA_VISIBLE_DEVICES").isdigit()
            assert gsb._real_gpu_ids == []
            assert pool._gpus_acquired == 0

    def test_virtual_reservation_persists_across_execs(self, gsb, pool):
        self.reserve(gsb, pool)
        gsb.exec("echo first")
        assert gsb._virtual_gpu_count_requested > 0
        assert gsb._real_gpu_ids == []
        assert ok(gsb, "echo $CUDA_VISIBLE_DEVICES").isdigit()

    def test_physical_gpu_is_held_while_a_background_process_runs(self, gsb, pool):
        self.reserve(gsb, pool)
        gsb.exec("sleep 30 &")
        assert len(gsb.get_live_pids()) > 0
        assert gsb._real_gpu_ids
        assert pool._gpus_acquired == 1

    def test_the_same_gpu_serves_later_execs_during_a_background_process(self, gsb, pool):
        self.reserve(gsb, pool)
        gsb.exec("sleep 30 &")
        gsb.get_live_pids()
        first_gpu_ids = list(gsb._real_gpu_ids)
        assert first_gpu_ids
        gsb.exec("echo checking")
        assert gsb._real_gpu_ids == first_gpu_ids  # same physical GPU, not re-acquired

    def test_physical_gpu_is_released_after_the_background_process_finishes(self, gsb, pool):
        gsb.exec("true")
        self.reserve(gsb, pool)
        gsb.exec("sleep 15 >/dev/null 2>&1 &")
        assert gsb._real_gpu_ids
        assert wait_until(lambda: not gsb._real_gpu_ids, timeout=90, interval=0.5)
        assert gsb._real_gpu_ids == []
        assert pool._gpus_acquired == 0
        assert gsb._virtual_gpu_count_requested > 0

    def test_exec_blocks_until_a_pool_gpu_is_freed(self, box):
        pool = agResourcePool(gpus=[0])
        holder = MagicMock()
        holder._name = "holder"
        holder._real_gpu_ids = []
        pool.acquire_real_gpus(holder, 1)  # exhaust the only GPU

        sb2 = box.sandbox(agconfig=box.config(gpu=True))
        self.reserve(sb2, pool)
        exec_done = threading.Event()

        def _run():
            sb2.exec("echo hello")  # blocks inside exec() until a GPU is freed
            exec_done.set()

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        time.sleep(1.0)
        assert not exec_done.is_set()  # still waiting
        pool.release_real_gpus(holder, [0])
        assert exec_done.wait(timeout=90), "exec() never unblocked after the GPU was freed"
        worker.join(timeout=10)

    def test_release_resources_clears_the_flag_and_returns_a_held_gpu(self, gsb, pool):
        self.reserve(gsb, pool)
        gsb.exec("sleep 30 &")
        gsb.get_live_pids()
        assert gsb._real_gpu_ids
        gsb.release_virtual_and_real_resources(pool)
        assert gsb._virtual_gpu_count_requested == 0
        assert gsb._real_gpu_ids == []
        assert pool._gpus_acquired == 0

    def test_stop_returns_a_held_gpu_to_the_pool(self, gsb, pool):
        self.reserve(gsb, pool)
        gsb.exec("sleep 30 &")
        gsb.get_live_pids()
        assert pool._gpus_acquired == 1
        gsb.stop()
        assert pool._gpus_acquired == 0
        assert gsb._real_gpu_ids == []

    def test_destroy_returns_a_held_gpu_to_the_pool(self, gsb, pool):
        self.reserve(gsb, pool)
        gsb.exec("sleep 30 &")
        gsb.get_live_pids()
        assert pool._gpus_acquired == 1
        gsb.destroy()
        assert pool._gpus_acquired == 0


# ---------------------------------------------------------------------------
# concurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_concurrent_sandboxes_do_not_interfere(self, box):
        sandboxes = [box.sandbox() for _ in range(4)]

        def work(index_and_sb):
            index, sandbox = index_and_sb
            sandbox.write_file("/workspace/id.txt", f"sandbox-{index}\n")
            time.sleep(0.2)
            return index, sandbox.read_file("/workspace/id.txt"), ok(sandbox, "ls /workspace")

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(work, enumerate(sandboxes)))
        for index, content, listing in results:
            assert content == f"sandbox-{index}\n"
            assert listing == "id.txt"

    def test_concurrent_execs_on_one_sandbox_all_complete_correctly(self, sb):
        ok(sb, "true")

        def run(i):
            return ok(sb, f"echo job-{i}")

        with ThreadPoolExecutor(max_workers=6) as pool:
            outputs = list(pool.map(run, range(12)))
        assert outputs == [f"job-{i}" for i in range(12)]

    def test_concurrent_file_writes_on_one_sandbox(self, sb):
        ok(sb, "true")

        def write(i):
            sb.write_file(f"/workspace/concurrent/{i}.txt", f"{i}\n")

        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(write, range(12)))
        for i in range(12):
            assert sb.read_file(f"/workspace/concurrent/{i}.txt") == f"{i}\n"

    def test_concurrent_commits_of_different_sandboxes(self, box):
        sandboxes = [box.sandbox() for _ in range(3)]
        tags = [box.tag() for _ in sandboxes]
        for index, sandbox in enumerate(sandboxes):
            sandbox.write_file("/workspace/x.txt", f"{index}\n")
        with ThreadPoolExecutor(max_workers=3) as pool:
            committed = list(pool.map(lambda pair: pair[0].commit(pair[1]), zip(sandboxes, tags)))
        assert committed == [True, True, True]
        for index, tag in enumerate(tags):
            assert box.sandbox(checkpoint_image=tag).read_file("/workspace/x.txt") == f"{index}\n"


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------

_LOOPBACK_ROUNDTRIP = """python3 - <<'PY'
import socket
server = socket.socket()
server.bind(("127.0.0.1", 0))
server.listen(1)
client = socket.create_connection(("127.0.0.1", server.getsockname()[1]), timeout=5)
peer, _ = server.accept()
client.sendall(b"ping")
print(peer.recv(4).decode())
PY"""


class TestNetwork:
    def test_loopback_works_inside_the_sandbox(self, sb):
        assert ok(sb, _LOOPBACK_ROUNDTRIP) == "ping"

    def test_two_sandboxes_each_have_a_working_loopback(self, box):
        sb1, sb2 = box.sandbox(), box.sandbox()
        assert ok(sb1, _LOOPBACK_ROUNDTRIP) == "ping"
        assert ok(sb2, _LOOPBACK_ROUNDTRIP) == "ping"

    def test_host_loopback_services_are_unreachable_from_the_sandbox(self, box):
        box.requires("network_namespace")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            sb = box.sandbox()
            _, rc = sb.exec(
                'python3 -c "import socket; '
                f"socket.create_connection(('127.0.0.1', {port}), timeout=3)\""
            )
        assert rc != 0, "the sandbox reached a service listening on the host's loopback"

    def test_a_sandbox_loopback_service_is_unreachable_from_the_host(self, box):
        box.requires("network_namespace")
        port = 18000 + uuid.uuid4().int % 2000
        sb = box.sandbox()
        sb.exec_detached(
            'python3 -c "import socket, time; s = socket.socket(); '
            f"s.bind(('127.0.0.1', {port})); s.listen(1); time.sleep(60)\""
        )
        # Wait until the listener is really up inside the sandbox, then probe from the host.
        up = wait_until(
            lambda: (
                sb.exec(
                    'python3 -c "import socket; '
                    f"socket.create_connection(('127.0.0.1', {port}), timeout=1)\""
                )[1]
                == 0
            ),
            timeout=30,
        )
        assert up, "the in-sandbox listener never came up"
        with pytest.raises(OSError), socket.create_connection(("127.0.0.1", port), timeout=2):
            pass


# ---------------------------------------------------------------------------
# ptrace inside the sandbox (what agency's agproxy_ptrace relies on)
# ---------------------------------------------------------------------------

# agproxy_ptrace (agency/harness/ptrace) runs inside the sandbox next to the
# harness it supervises. It needs: PTRACE_TRACEME in a forked child, attaching
# to a child, seccomp filters returning SECCOMP_RET_TRACE with
# PTRACE_O_TRACESECCOMP stops (also how clock reads are trapped for
# time_control), and PTRACE_GETSIGMASK. This probe exercises exactly those with
# ctypes only, so it needs nothing but python3 in the sandbox. x86_64 only,
# like the supervisor itself.
_PTRACE_PROBE = r"""
import ctypes, os, platform, signal, struct, sys
if platform.machine() != "x86_64":
    sys.exit(77)
libc = ctypes.CDLL(None, use_errno=True)
libc.ptrace.restype = ctypes.c_long
libc.ptrace.argtypes = [ctypes.c_long, ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p]
TRACEME, CONT, ATTACH, DETACH, SETOPTIONS, GETSIGMASK = 0, 7, 16, 17, 0x4200, 0x420A
O_TRACESECCOMP, EVENT_SECCOMP = 0x80, 7
mode = sys.argv[1]

def fail(what):
    print("FAIL %s errno=%s" % (what, os.strerror(ctypes.get_errno())))
    sys.exit(1)

def install_trace_filter(nr):
    code = b"".join(struct.pack("HBBI", *ins) for ins in [
        (0x20, 0, 0, 0),            # load syscall number
        (0x15, 0, 1, nr),           # if nr == target: fallthrough, else skip one
        (0x06, 0, 0, 0x7FF00000),   # SECCOMP_RET_TRACE
        (0x06, 0, 0, 0x7FFF0000),   # SECCOMP_RET_ALLOW
    ])
    buf = ctypes.create_string_buffer(code)
    class Prog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]
    prog = Prog(4, ctypes.addressof(buf))
    if libc.prctl(38, 1, 0, 0, 0) != 0:
        fail("PR_SET_NO_NEW_PRIVS")
    if libc.prctl(22, 2, ctypes.byref(prog), 0, 0) != 0:
        fail("seccomp filter install")

def spawn_traced(child_body):
    pid = os.fork()
    if pid == 0:
        if libc.ptrace(TRACEME, 0, None, None) != 0:
            os._exit(2)
        os.kill(os.getpid(), signal.SIGSTOP)
        child_body()
        os._exit(7)
    _, status = os.waitpid(pid, 0)
    if not os.WIFSTOPPED(status):
        fail("child did not stop after PTRACE_TRACEME+SIGSTOP")
    return pid

def finish(pid):
    if libc.ptrace(CONT, pid, None, None) != 0:
        fail("PTRACE_CONT")
    _, status = os.waitpid(pid, 0)
    if not (os.WIFEXITED(status) and os.WEXITSTATUS(status) == 7):
        fail("child did not exit cleanly, status=%#x" % status)

if mode == "traceme":
    finish(spawn_traced(lambda: None))
elif mode == "getsigmask":
    pid = spawn_traced(lambda: None)
    mask = ctypes.c_ulong(0)
    if libc.ptrace(GETSIGMASK, pid, 8, ctypes.byref(mask)) != 0:
        fail("PTRACE_GETSIGMASK")
    finish(pid)
elif mode == "attach":
    child = os.fork()
    if child == 0:
        signal.pause()
        os._exit(0)
    if libc.ptrace(ATTACH, child, None, None) != 0:
        os.kill(child, signal.SIGKILL)
        fail("PTRACE_ATTACH to own child")
    _, status = os.waitpid(child, 0)
    if not os.WIFSTOPPED(status):
        fail("attached child did not stop")
    if libc.ptrace(DETACH, child, None, None) != 0:
        fail("PTRACE_DETACH")
    os.kill(child, signal.SIGKILL)
    os.waitpid(child, 0)
elif mode.startswith("seccomp:"):
    nr = int(mode.split(":")[1])
    buf = ctypes.create_string_buffer(16)
    def body():
        install_trace_filter(nr)
        libc.syscall(nr, 1, buf) if nr == 228 else libc.syscall(nr)
    pid = spawn_traced(body)
    if libc.ptrace(SETOPTIONS, pid, None, O_TRACESECCOMP) != 0:
        fail("PTRACE_SETOPTIONS TRACESECCOMP")
    if libc.ptrace(CONT, pid, None, None) != 0:
        fail("PTRACE_CONT")
    _, status = os.waitpid(pid, 0)
    if not (os.WIFSTOPPED(status) and status >> 16 == EVENT_SECCOMP):
        fail("no SECCOMP event stop, status=%#x" % status)
    finish(pid)
else:
    sys.exit(2)
print("ptrace-ok")
"""


class TestPtraceInSandbox:
    """The ptrace/seccomp primitives agproxy_ptrace needs must work inside the
    sandbox: docker/podman seccomp profiles and capabilities, or the chroot's
    user namespace plus Yama, must not block them."""

    def _probe(self, sb, mode: str) -> None:
        sb.write_file("/workspace/ptrace_probe.py", _PTRACE_PROBE)
        out, rc = sb.exec(f"python3 /workspace/ptrace_probe.py {mode}")
        if rc == 77:
            pytest.skip("the ptrace supervisor (and so this probe) is x86_64-only")
        assert rc == 0 and "ptrace-ok" in out, f"ptrace probe {mode!r} failed (rc={rc}): {out}"

    def test_a_forked_child_can_ptrace_traceme(self, sb):
        self._probe(sb, "traceme")

    def test_a_process_can_attach_to_its_own_child(self, sb):
        self._probe(sb, "attach")

    def test_ptrace_getsigmask_works(self, sb):
        self._probe(sb, "getsigmask")

    def test_seccomp_ret_trace_stops_are_delivered_to_the_tracer(self, sb):
        self._probe(sb, "seccomp:110")  # getppid

    def test_clock_gettime_can_be_trapped_the_way_time_control_does(self, sb):
        self._probe(sb, "seccomp:228")  # clock_gettime


# ---------------------------------------------------------------------------
# daemons (exec_detached)
# ---------------------------------------------------------------------------


class TestDaemons:
    def test_stdout_and_stderr_can_be_redirected_to_a_log_inside_the_sandbox(self, sb):
        sb.exec_detached("( echo to-stdout; echo to-stderr >&2 ) > /workspace/daemon.log 2>&1")
        log = wait_until(lambda: sb.exec("cat /workspace/daemon.log")[0].strip(), timeout=20)
        assert "to-stdout" in log and "to-stderr" in log
        assert "to-stdout" in sb.read_file("/workspace/daemon.log")

    def test_a_daemon_keeps_appending_to_its_log_while_it_lives(self, sb):
        sb.exec_detached(
            "i=0; while true; do echo tick-$i >> /workspace/ticks.log; i=$((i+1)); sleep 0.2; done"
        )

        def lines():
            out, rc = sb.exec("wc -l < /workspace/ticks.log")
            return int(out.strip()) if rc == 0 and out.strip().isdigit() else 0

        first = wait_until(lambda: lines() >= 2 and lines(), timeout=20)
        assert first
        assert wait_until(lambda: lines() > first, timeout=20)

    def test_a_daemon_outlives_the_exec_calls_that_follow_it(self, sb):
        sb.exec_detached(_HEARTBEAT_LOOP.format(name="daemon_hb"))
        for _ in range(3):
            ok(sb, "echo unrelated-call")
        assert heartbeat_advances(sb, "daemon_hb")

    def test_a_detached_daemon_is_tracked_by_the_sandbox(self, box, sb):
        box.requires("detached_pid_tracking")
        sb.exec_detached("sleep 60")
        assert sb.get_live_pids()

    def test_stop_tears_the_daemon_down_but_keeps_files(self, sb):
        sb.write_file("/workspace/kept.txt", "kept\n")
        sb.exec_detached(_HEARTBEAT_LOOP.format(name="daemon_hb"))
        assert heartbeat_advances(sb, "daemon_hb")
        sb.stop()
        assert sb.read_file("/workspace/kept.txt") == "kept\n"
        assert not heartbeat_advances(sb, "daemon_hb"), "daemon survived stop()"

    def test_rm_container_tears_the_daemon_down(self, sb):
        sb.exec_detached(_HEARTBEAT_LOOP.format(name="daemon_hb"))
        assert heartbeat_advances(sb, "daemon_hb")
        sb.commit()
        sb.rm_container()
        assert not heartbeat_advances(sb, "daemon_hb"), "daemon survived rm_container()"

    def test_destroy_tears_the_daemon_down(self, box, tmp_path):
        # Observed from the host through a mount: nothing inside the destroyed
        # sandbox can be asked, but its last write to the shared directory
        # must be the last.
        host = tmp_path / "beat"
        sb = box.sandbox(agconfig=box.config(("beat", host, "/mnt/beat", "rw")))
        sb.exec_detached(
            "while true; do date +%s%N > /mnt/beat/tick.tmp && mv /mnt/beat/tick.tmp /mnt/beat/tick;"
            " sleep 0.2; done"
        )
        assert wait_until(lambda: (host / "tick").exists(), timeout=30)
        before = (host / "tick").read_text()
        assert wait_until(lambda: (host / "tick").read_text() != before, timeout=10)
        sb.destroy()
        time.sleep(0.7)
        settled = (host / "tick").read_text()
        time.sleep(1.5)
        assert (host / "tick").read_text() == settled, "daemon kept running after destroy()"


# ---------------------------------------------------------------------------
# pause and resume (SIGSTOP/SIGCONT; time freezing)
# ---------------------------------------------------------------------------


class TestPauseAndResume:
    """A paused agent tree is a SIGSTOPped process tree. The ptrace supervisor's
    time_control / time_realtime freezing needs the seccomp and ptrace
    primitives probed in TestPtraceInSandbox (clock_gettime trapping); driving
    the full supervisor needs agency's own dependencies, which a bare sandbox
    image does not carry, so that part stays host-side
    (tests/harness/test_time_ctrl_linux.py, test_pause_ack.py)."""

    def test_a_running_process_can_be_stopped_and_continued(self, sb):
        sb.exec_detached("echo $$ > /workspace/loop.pid; " + _HEARTBEAT_LOOP.format(name="hb"))
        pid = wait_until(lambda: sb.exec("cat /workspace/loop.pid")[0].strip(), timeout=20)
        assert pid.isdigit()
        assert heartbeat_advances(sb)
        ok(sb, f"kill -STOP {pid}")
        time.sleep(0.6)  # let an iteration already in flight finish
        assert not heartbeat_advances(sb), "process kept running after SIGSTOP"
        ok(sb, f"kill -CONT {pid}")
        assert heartbeat_advances(sb), "process did not resume after SIGCONT"

    def test_a_stopped_process_can_still_be_killed(self, sb):
        sb.exec_detached("echo $$ > /workspace/loop.pid; " + _HEARTBEAT_LOOP.format(name="hb"))
        pid = wait_until(lambda: sb.exec("cat /workspace/loop.pid")[0].strip(), timeout=20)
        assert pid.isdigit()
        ok(sb, f"kill -STOP {pid}")
        sb.stop()
        assert not heartbeat_advances(sb), "stopped process survived stop()"

    def test_the_time_control_config_flags_exist_with_their_documented_defaults(self):
        from agency.configs.agconfig import ptraceconfig

        cfg = ptraceconfig()
        assert cfg.time_control is True
        assert cfg.time_realtime is False


# ---------------------------------------------------------------------------
# unix domain sockets across the mount boundary
# (mirrors harness_daemon_launcher: a host dir bind-mounted rw, one socket
# served from each side, and the in-sandbox path computed from the host path)
# ---------------------------------------------------------------------------

_UDS_SERVER = """
import os, socket, sys
path = sys.argv[1]
server = socket.socket(socket.AF_UNIX)
server.bind(path)
os.chmod(path, 0o777)
server.listen(4)
while True:
    conn, _ = server.accept()
    with conn:
        conn.recv(64)
        conn.sendall(b"pong-from-sandbox")
"""

_UDS_CLIENT = (
    'python3 -c "import socket, sys; s = socket.socket(socket.AF_UNIX); s.settimeout(10); '
    "s.connect(sys.argv[1]); s.sendall(b'ping'); print(s.recv(64).decode())\" "
)


@contextlib.contextmanager
def _host_uds_server(path: Path, reply: bytes = b"pong-from-host", mode: int = 0o777):
    server = socket.socket(socket.AF_UNIX)
    server.bind(str(path))
    os.chmod(path, mode)
    server.listen(4)
    server.settimeout(60)
    received: list[bytes] = []

    def serve():
        try:
            while True:
                conn, _ = server.accept()
                with conn:
                    received.append(conn.recv(64))
                    conn.sendall(reply)
        except OSError:
            return

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    try:
        yield received
    finally:
        server.close()
        worker.join(timeout=5)


def _uds_layout(tmp_root: Path, *, long_paths: bool):
    """(host_dir, mount_point, socket file name). With *long_paths*, both the
    host-side and the in-sandbox socket paths are exactly 100 characters, close
    to the 108-byte sun_path limit."""
    name = "agency.sock"
    if not long_paths:
        return tmp_root / "uds", "/mnt/uds", name
    pad_host = 100 - len(str(tmp_root)) - 2 - len(name)
    pad_mount = 100 - len("/mnt/") - 1 - len(name)
    return tmp_root / ("h" * pad_host), "/mnt/" + "m" * pad_mount, name


@pytest.fixture
def uds_root():
    import shutil
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="agu", dir="/tmp"))
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.mark.parametrize("long_paths", [False, True], ids=["short-path", "100-char-path"])
class TestUnixSockets:
    def _sandbox(self, box, uds_root, long_paths):
        host_dir, mount_point, name = _uds_layout(uds_root, long_paths=long_paths)
        host_dir.mkdir(parents=True, exist_ok=True)
        if long_paths:
            assert len(str(host_dir / name)) == 100 and len(f"{mount_point}/{name}") == 100
        sb = box.sandbox(agconfig=box.config(("uds", host_dir, mount_point, "rw")))
        return sb, host_dir / name, f"{mount_point}/{name}"

    def test_sandbox_client_reaches_a_host_server(self, box, uds_root, long_paths):
        sb, host_sock, sandbox_sock = self._sandbox(box, uds_root, long_paths)
        with _host_uds_server(host_sock) as received:
            _, rc = sb.exec(f"test -S {sandbox_sock}")
            assert rc == 0, "the host socket is not visible as a socket inside the sandbox"
            assert ok(sb, _UDS_CLIENT + sandbox_sock) == "pong-from-host"
        assert received == [b"ping"]

    def test_host_client_reaches_a_sandbox_server(self, box, uds_root, long_paths):
        sb, host_sock, sandbox_sock = self._sandbox(box, uds_root, long_paths)
        sb.write_file("/workspace/uds_server.py", _UDS_SERVER)
        sb.exec_detached(f"python3 /workspace/uds_server.py {sandbox_sock}")

        def connect():
            try:
                with socket.socket(socket.AF_UNIX) as client:
                    client.settimeout(5)
                    client.connect(str(host_sock))
                    client.sendall(b"ping")
                    return client.recv(64)
            except OSError:
                return b""

        assert wait_until(connect, timeout=60, interval=0.5) == b"pong-from-sandbox"

    def test_an_owner_only_host_socket_is_still_reachable_from_the_sandbox(
        self, box, uds_root, long_paths
    ):
        # Rootless podman maps the sandbox's root onto the host user, docker's
        # root bypasses the check, chroot maps root onto the user too: a 0600
        # socket owned by the host user must work in every case.
        sb, host_sock, sandbox_sock = self._sandbox(box, uds_root, long_paths)
        with _host_uds_server(host_sock, mode=0o600):
            assert ok(sb, f"stat -c %a {sandbox_sock}") == "600"
            assert ok(sb, _UDS_CLIENT + sandbox_sock) == "pong-from-host"

    def test_a_socket_survives_the_sandbox_restarting(self, box, uds_root, long_paths):
        sb, host_sock, sandbox_sock = self._sandbox(box, uds_root, long_paths)
        with _host_uds_server(host_sock):
            assert ok(sb, _UDS_CLIENT + sandbox_sock) == "pong-from-host"
            sb.stop()
            assert ok(sb, _UDS_CLIENT + sandbox_sock) == "pong-from-host"
