"""Tier 3: the sandbox-wide ``agSandbox`` contract, run on every backend
(exec, file I/O, mounts, isolation, lifecycle, process tracking).

Split from the original single file for CI wall time; the shared harness
(_Box, fixtures, helpers) lives in test_all_backends_common.py. Each test
is parametrized over ``sandbox_kind`` (chroot, docker, podman) and only
uses the public ``agSandbox`` facade, so the same assertion holds for all
backends. A backend that is not usable on the host skips itself. Where a
backend genuinely cannot honour part of the contract the test calls
``box.requires("<capability>")`` so the gap shows up as a skip with a
reason instead of the test being dropped.
"""

from tests.sandbox.test_all_backends_common import (  # noqa: F401
    _HEARTBEAT_LOOP,
    _Box,
    backend_view,
    box,
    heartbeat_advances,
    ok,
    read_heartbeat,
    sb,
    sha256,
    wait_until,
)

import contextlib  # noqa: F401
import functools  # noqa: F401
import hashlib  # noqa: F401
import os  # noqa: F401
import pickle  # noqa: F401
import socket  # noqa: F401
import subprocess  # noqa: F401
import sys  # noqa: F401
import threading  # noqa: F401
import time  # noqa: F401
import uuid  # noqa: F401
from concurrent.futures import ThreadPoolExecutor  # noqa: F401
from pathlib import Path  # noqa: F401
from unittest.mock import MagicMock  # noqa: F401

import pytest  # noqa: F401

from agency.orchestrator.agresources import (  # noqa: F401
    MIN_CPUS,
    MIN_MEMORY_MB,
    agResourcePool,
)
from tests.sandbox.conftest import (  # noqa: F401
    BACKEND_CAPABILITIES,
    CAPABILITY_DESCRIPTIONS,
    CONTAINER_BACKENDS,
    backend_class,  # noqa: F401
    checkpoint_exists,  # noqa: F401
    make_backend,  # noqa: F401
    make_sandbox,  # noqa: F401
    requires,  # noqa: F401
)


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
