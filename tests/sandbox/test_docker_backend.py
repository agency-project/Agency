"""Tier 1 (docker-only) tests for agency.sandbox.docker._DockerBackend and the
docker halves of the shared modules.

Only mechanisms that exist for docker alone live here: ``docker info`` parsing
(root dir, storage driver, rootless detection, containerd address), the
classic ``overlay2`` layerdb and the containerd ``overlayfs`` snapshotter
lookups (``ctr`` argv, ChainID, ``snapshots view/mounts``), rootless ``dockerd``
discovery through ``/proc/<pid>/uid_map``, docker's CRIU ``checkpoint create``
/ ``start --checkpoint`` flow and its native-ZFS-driver storage.

Everything docker shares with podman through ``_ContainerBackendBase`` (run
flags, lifecycle, retries, quota, slot/GPU release, commit/squash/accumulator,
the generic diff-dir lookup contract, rootless id translation) is tested once
for both runtimes in test_container_backends.py; behaviour every backend shares
is in test_all_backends.py. The podman counterpart of this file is
test_podman_backend.py.

Tests that need a real Docker daemon skip automatically when it is unreachable.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import agency.sandbox.docker as _mod
from agency.sandbox import checkpoint as cp
from tests.sandbox.conftest import (
    backend_available,
    make_backend,
    make_cow_backend,
    started_run_cmd,
)

docker_daemon = pytest.mark.skipif(
    not backend_available("docker"), reason="Docker daemon not reachable"
)


class _FakeCompleted:
    def __init__(self, stdout=b"", returncode=0, stderr=b""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def _info_backend(info, **kwargs):
    """A docker backend whose ``docker info`` JSON is *info* (None = failing)."""
    backend = make_backend("docker", **kwargs)
    patch.object(type(backend), "_docker_info", return_value=info).start()
    return backend


@pytest.fixture(autouse=True)
def _stop_info_patches():
    yield
    patch.stopall()


# ===========================================================================
# containerd overlayfs snapshotter, ctr, ChainID
# ===========================================================================


class TestLocateLayerDiffDir:
    """The containerd-snapshotter (``overlayfs`` driver) branch of
    ``_DockerBackend._locate_layer_diff_dir``. The overlay2/shared contract
    (digest found/not found, unsupported driver, info failure, missing
    content) is in test_container_backends.py::TestLocateLayerDiffDir."""

    def _sb(self):
        return make_backend("docker")

    def _fake_docker_root(self, tmp_path, *, driver="overlay2"):
        root = tmp_path / "docker-root"
        (root / "image" / "overlay2" / "layerdb" / "sha256").mkdir(parents=True)
        (root / "overlay2").mkdir(parents=True)
        return root

    def _add_layerdb_entry(self, root, diff_id, cache_id, *, with_content=True):
        entry = root / "image" / "overlay2" / "layerdb" / "sha256" / f"entry-{cache_id}"
        entry.mkdir(parents=True)
        (entry / "diff").write_text(diff_id)
        (entry / "cache-id").write_text(cache_id)
        if with_content:
            diff_dir = root / "overlay2" / cache_id / "diff"
            diff_dir.mkdir(parents=True)
            (diff_dir / "marker").write_text("x")
        return entry

    def test_overlayfs_requires_diff_ids_chain_ending_at_diff_id(self, tmp_path):
        """Containerd snapshotter path keys layers by ChainID -- calling
        without a matching RootFS.Layers prefix must return None rather
        than guessing from the tip DiffID alone."""
        import agency.sandbox.docker as _mod

        sb = make_backend("docker")
        root = self._fake_docker_root(tmp_path)
        with patch.object(
            _mod._DockerBackend, "_docker_data_root_and_driver", return_value=(root, "overlayfs")
        ):
            assert sb._locate_layer_diff_dir("sha256:tip") is None
            assert (
                sb._locate_layer_diff_dir("sha256:tip", diff_ids=["sha256:base", "sha256:not-tip"])
                is None
            )

    def test_overlayfs_uses_ctr_mounts_top_fs(self, tmp_path):
        """Mocked ctr view+mounts: the first lowerdir component is the
        layer's own fs directory (confirmed against real containerd)."""
        import agency.sandbox.docker as _mod

        sb = make_backend("docker")
        root = self._fake_docker_root(tmp_path)
        snap_fs = tmp_path / "snapshots" / "9" / "fs"
        snap_fs.mkdir(parents=True)
        (snap_fs / "marker").write_text("ok")
        diff_ids = ["sha256:" + "a" * 64, "sha256:" + "b" * 64]
        mounts_out = (
            f"mount -t overlay overlay /tmp/x -o "
            f"lowerdir={snap_fs}:{tmp_path}/snapshots/8/fs,index=off\n"
        )

        class _Done:
            def __init__(self, returncode=0, stdout="", stderr=b""):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr

        calls = []

        def fake_run(args, **kwargs):
            calls.append(list(args))
            if "mounts" in args:
                return _Done(stdout=mounts_out)
            return _Done()

        with patch.object(
            _mod._DockerBackend, "_docker_data_root_and_driver", return_value=(root, "overlayfs")
        ):
            with patch.object(_mod._DockerBackend, "_ctr_argv", return_value=["ctr"]):
                with patch.object(_mod.subprocess, "run", side_effect=fake_run):
                    result = sb._locate_layer_diff_dir(diff_ids[-1], diff_ids=diff_ids)

        assert result == snap_fs
        assert any("view" in c for c in calls)
        assert any("mounts" in c for c in calls)
        assert any(c[-2:] == ["snapshots", "rm"] or "rm" in c for c in calls)

    def test_chain_id_matches_containerd_definition(self):
        from agency.sandbox.docker import _chain_id_for_diff_ids
        import hashlib

        d0 = "sha256:" + "a" * 64
        d1 = "sha256:" + "b" * 64
        assert _chain_id_for_diff_ids([d0]) == d0
        expected = "sha256:" + hashlib.sha256(f"{d0} {d1}".encode()).hexdigest()
        assert _chain_id_for_diff_ids([d0, d1]) == expected

    def test_parse_ctr_mounts_top_fs(self):
        from agency.sandbox.docker import _parse_ctr_mounts_top_fs
        from pathlib import Path

        out = (
            "mount -t overlay overlay /tmp/x -o "
            "lowerdir=/var/lib/containerd/.../snapshots/1049/fs:"
            "/var/lib/containerd/.../snapshots/1046/fs,index=off\n"
        )
        assert _parse_ctr_mounts_top_fs(out) == Path("/var/lib/containerd/.../snapshots/1049/fs")
        assert _parse_ctr_mounts_top_fs(
            "mount --bind /var/lib/containerd/.../snapshots/1/fs /tmp/x\n"
        ) == Path("/var/lib/containerd/.../snapshots/1/fs")

    @docker_daemon
    def test_real_containerd_overlayfs_commit_layer_locates(self):
        """End-to-end against live Docker+containerd: commit a marker file
        and resolve its RootFS tip to a readable snapshot fs/ directory
        containing that marker. Skips when this host isn't on the
        overlayfs snapshotter or the snapshot tree isn't readable."""
        info = json.loads(
            subprocess.run(
                ["docker", "info", "--format", "{{json .}}"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
        if info.get("Driver") != "overlayfs":
            pytest.skip(f"Docker Driver is {info.get('Driver')!r}, not overlayfs")

        # Rootful containerd stores are often 0700; make the snapshotter
        # tree traversable/readable for this process when passwordless
        # sudo is available so overlay_diff_to_tar can use the path.
        snap_root = Path("/var/lib/containerd/io.containerd.snapshotter.v1.overlayfs")
        if not os.access(snap_root, os.R_OK | os.X_OK):
            chmod = subprocess.run(
                ["sudo", "-n", "chmod", "-R", "a+rX", str(snap_root.parent), str(snap_root)],
                capture_output=True,
            )
            if chmod.returncode != 0 or not os.access(snap_root, os.R_OK | os.X_OK):
                pytest.skip("containerd snapshotter storage not readable")

        name = f"test-ctrd-locate-{uuid.uuid4().hex[:8]}"
        tag = f"agency-test-ctrd-locate-{uuid.uuid4().hex[:8]}:latest"
        sb = make_backend("docker")
        try:
            subprocess.run(
                ["docker", "run", "-d", "--name", name, "alpine:latest", "sleep", "3600"],
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["docker", "exec", name, "sh", "-c", "echo ctrd-locate > /tmp/ctrd-locate.txt"],
                capture_output=True,
                check=True,
            )
            subprocess.run(["docker", "commit", name, tag], capture_output=True, check=True)
            layers = json.loads(
                subprocess.run(
                    ["docker", "inspect", "--format={{json .RootFS.Layers}}", tag],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
            )
            diff_dir = sb._locate_layer_diff_dir(layers[-1], diff_ids=layers)
            assert diff_dir is not None, f"failed to locate diff dir for {layers[-1]}"
            assert (diff_dir / "tmp" / "ctrd-locate.txt").read_text() == "ctrd-locate\n"
        finally:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
            subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)


class TestCtrArgv:
    """Tests for _DockerBackend._ctr_argv()/_containerd_address() -- the
    fix for rootless Docker's containerd-snapshotter mode reaching for
    the wrong socket. Rootless Docker runs its own private per-user
    containerd (reported via `docker info`'s Containerd.Address field),
    never the system-wide /run/containerd/containerd.sock bare `ctr`
    defaults to -- confirmed empirically: on a real rootless host, the
    system socket stayed root-owned and permission-denied while the
    daemon's own private socket (owned by the invoking user) worked.
    """

    def _sb(self):
        return make_backend("docker")

    def test_containerd_address_reads_docker_info_field(self):
        import agency.sandbox.docker as _mod

        sb = make_backend("docker")
        info = {"Containerd": {"Address": "/run/user/1000/docker/containerd/containerd.sock"}}
        with patch.object(_mod._DockerBackend, "_docker_info", return_value=info):
            assert sb._containerd_address() == "/run/user/1000/docker/containerd/containerd.sock"

    def test_containerd_address_none_when_field_missing(self):
        import agency.sandbox.docker as _mod

        sb = make_backend("docker")
        with patch.object(_mod._DockerBackend, "_docker_info", return_value={}):
            assert sb._containerd_address() is None
        with patch.object(_mod._DockerBackend, "_docker_info", return_value=None):
            assert sb._containerd_address() is None

    def test_ctr_argv_passes_address_when_available(self):
        """The probed argv must target the daemon's own containerd
        socket via --address, not bare `ctr` (which would silently
        default to the root-owned system socket)."""
        import agency.sandbox.docker as _mod

        sb = make_backend("docker")
        address = "/run/user/1000/docker/containerd/containerd.sock"
        calls = []

        def fake_run(args, **kwargs):
            calls.append(list(args))
            return _FakeCompleted(returncode=0)

        with patch.object(_mod._DockerBackend, "_containerd_address", return_value=address):
            with patch.object(_mod.subprocess, "run", side_effect=fake_run):
                result = sb._ctr_argv()

        assert result == ["ctr", "--address", address]
        assert calls == [["ctr", "--address", address, "version"]]

    def test_ctr_argv_falls_back_to_sudo_with_address_when_bare_fails(self):
        import agency.sandbox.docker as _mod

        sb = make_backend("docker")
        address = "/run/containerd/containerd.sock"

        def fake_run(args, **kwargs):
            if args[:2] == ["sudo", "-n"]:
                return _FakeCompleted(returncode=0)
            return _FakeCompleted(returncode=1, stderr=b"permission denied")

        with patch.object(_mod._DockerBackend, "_containerd_address", return_value=address):
            with patch.object(_mod.subprocess, "run", side_effect=fake_run):
                result = sb._ctr_argv()

        assert result == ["sudo", "-n", "ctr", "--address", address]

    def test_ctr_argv_plain_ctr_when_no_address_reported(self):
        """Older Docker or a runtime that doesn't expose Containerd.Address
        must still fall back to bare `ctr` rather than erroring."""
        import agency.sandbox.docker as _mod

        sb = make_backend("docker")
        calls = []

        def fake_run(args, **kwargs):
            calls.append(list(args))
            return _FakeCompleted(returncode=0)

        with patch.object(_mod._DockerBackend, "_containerd_address", return_value=None):
            with patch.object(_mod.subprocess, "run", side_effect=fake_run):
                result = sb._ctr_argv()

        assert result == ["ctr"]
        assert calls == [["ctr", "version"]]

    def test_ctr_argv_none_when_nothing_works(self):
        import agency.sandbox.docker as _mod

        sb = make_backend("docker")
        with patch.object(_mod._DockerBackend, "_containerd_address", return_value=None):
            with patch.object(
                _mod.subprocess, "run", return_value=_FakeCompleted(returncode=1, stderr=b"nope")
            ):
                assert sb._ctr_argv() is None

    def test_ctr_argv_cached(self):
        import agency.sandbox.docker as _mod

        sb = make_backend("docker")
        calls = []

        def fake_run(args, **kwargs):
            calls.append(list(args))
            return _FakeCompleted(returncode=0)

        with patch.object(_mod._DockerBackend, "_containerd_address", return_value=None):
            with patch.object(_mod.subprocess, "run", side_effect=fake_run):
                first = sb._ctr_argv()
                second = sb._ctr_argv()

        assert first == second == ["ctr"]
        assert len(calls) == 1


class TestRootlessDockerdHelpers:
    """Docker's way of finding the rootless daemon's id maps: locate the
    ``dockerd`` process owned by this user and read its /proc uid_map/gid_map.
    (The translation contract itself is shared with podman: see
    test_container_backends.py::TestHostToContainerId.)"""

    def _sb(self):
        return make_backend("docker")

    def test_find_dockerd_pid_matches_process_by_name_and_owner(self, tmp_path, monkeypatch):
        """Fakes /proc as a tmp_path tree: PID 111 is "dockerd" owned by
        our own uid (the match), PID 222 is some other process owned by
        a different uid (must be skipped)."""
        import agency.sandbox.docker as _mod

        sb = make_backend("docker")
        (tmp_path / "111").mkdir()
        (tmp_path / "111" / "comm").write_text("dockerd\n")
        (tmp_path / "222").mkdir()
        (tmp_path / "222" / "comm").write_text("bash\n")

        my_uid = os.getuid()
        real_stat = os.stat

        def fake_listdir(path):
            assert path == "/proc"
            return ["111", "222"]

        def fake_stat(path):
            if path == "/proc/111":
                return real_stat(tmp_path / "111")
            return real_stat(tmp_path)  # some other uid (our own test process's cwd)

        real_open = open

        def fake_open(path, *a, **kw):
            if str(path).startswith("/proc/"):
                pid = str(path).split("/")[2]
                return real_open(tmp_path / pid / "comm", *a, **kw)
            return real_open(path, *a, **kw)

        monkeypatch.setattr(_mod.os, "listdir", fake_listdir)
        monkeypatch.setattr(_mod.os, "stat", fake_stat)
        monkeypatch.setattr(_mod.os, "getuid", lambda: my_uid)
        monkeypatch.setattr(_mod, "open", fake_open, raising=False)

        assert real_stat(tmp_path / "111").st_uid == my_uid

        result = sb._find_dockerd_pid()
        assert result == 111

    def test_find_dockerd_pid_returns_none_when_absent(self, monkeypatch):
        import agency.sandbox.docker as _mod

        sb = make_backend("docker")
        monkeypatch.setattr(_mod.os, "listdir", lambda p: [])
        assert sb._find_dockerd_pid() is None

    def test_parse_id_map_reads_real_proc_style_content(self, tmp_path):
        import agency.sandbox.docker as _mod

        p = tmp_path / "uid_map"
        p.write_text("         0     200682          1\n         1 1355350016      65536\n")
        assert _mod._DockerBackend._parse_id_map(p) == [(0, 200682, 1), (1, 1355350016, 65536)]

    def test_parse_id_map_returns_none_for_missing_file(self, tmp_path):
        import agency.sandbox.docker as _mod

        assert _mod._DockerBackend._parse_id_map(tmp_path / "nonexistent") is None


class TestParseCtrMounts:
    """``ctr snapshots mounts`` output comes in an overlay and a bind form."""

    @pytest.mark.parametrize(
        ("output", "expected"),
        [
            ("mount -t overlay overlay /x -o lowerdir=/a/1/fs:/a/0/fs,index=off\n", "/a/1/fs"),
            ("mount -t overlay overlay /x -o index=off,lowerdir=/only/fs\n", "/only/fs"),
            ("mount --bind /snap/1/fs /tmp/x\n", "/snap/1/fs"),
            ("mount -t bind /snap/2/fs /tmp/x -o bind\n", None),
            ("-o bind /snap/3/fs /tmp/x\n", "/snap/3/fs"),
            ("", None),
            ("some unrelated output\n", None),
        ],
    )
    def test_top_layer_directory_is_extracted(self, output, expected):
        result = _mod._parse_ctr_mounts_top_fs(output)
        assert (str(result) if result is not None else None) == expected

    def test_chain_id_of_nothing_is_none(self):
        assert _mod._chain_id_for_diff_ids([]) is None

    def test_chain_id_folds_the_whole_prefix(self):
        a, b, c = (f"sha256:{ch * 64}" for ch in "abc")
        ab = "sha256:" + hashlib.sha256(f"{a} {b}".encode()).hexdigest()
        abc = "sha256:" + hashlib.sha256(f"{ab} {c}".encode()).hexdigest()
        assert _mod._chain_id_for_diff_ids([a, b, c]) == abc


class TestContainerdOverlayfsLookup:
    """``_locate_containerd_overlayfs_diff_dir``: every failure degrades to
    None (the base class then squashes by export/import) and the temporary
    ``ctr`` view snapshot is always removed."""

    DIFF_IDS = ["sha256:" + "a" * 64, "sha256:" + "b" * 64]

    @staticmethod
    def _locate(backend, responder, ctr=("ctr",), diff_ids=None):
        calls = []

        def fake_run(args, **kwargs):
            calls.append(list(args))
            return responder(list(args))

        with patch.object(
            _mod._DockerBackend, "_ctr_argv", return_value=list(ctr) if ctr else None
        ):
            with patch.object(_mod.subprocess, "run", side_effect=fake_run):
                result = backend._locate_containerd_overlayfs_diff_dir(
                    diff_ids if diff_ids is not None else TestContainerdOverlayfsLookup.DIFF_IDS
                )
        return result, calls

    @staticmethod
    def _mounts_ok(snap):
        def respond(args):
            if "mounts" in args:
                return _FakeCompleted(stdout=f"mount -t overlay x /t -o lowerdir={snap}\n")
            return _FakeCompleted()

        return respond

    def test_empty_chain_is_rejected_before_any_ctr_call(self, capsys):
        result, calls = self._locate(
            make_backend("docker"), lambda a: _FakeCompleted(), diff_ids=[]
        )
        assert result is None and calls == []
        assert "empty diff_ids" in capsys.readouterr().err

    def test_no_usable_ctr_means_no_fast_path_and_no_calls(self):
        result, calls = self._locate(make_backend("docker"), lambda a: _FakeCompleted(), ctr=None)
        assert result is None and calls == []

    def test_view_uses_the_moby_namespace_and_the_chain_id(self, tmp_path):
        snap = tmp_path / "fs"
        snap.mkdir()
        result, calls = self._locate(
            make_backend("docker"), self._mounts_ok(snap), ctr=("ctr", "--address", "/s.sock")
        )
        assert result == snap
        chain = _mod._chain_id_for_diff_ids(self.DIFF_IDS)
        view = calls[0]
        assert view[:5] == ["ctr", "--address", "/s.sock", "-n", "moby"]
        assert view[5:7] == ["snapshots", "view"] and view[-1] == chain
        assert view[7].startswith("agency-locate-")
        assert calls[-1][-3:-1] == ["snapshots", "rm"] and calls[-1][-1] == view[7]

    def test_view_failure_is_reported_and_still_cleans_up(self, capsys):
        def respond(args):
            return _FakeCompleted(returncode=1 if "view" in args else 0, stderr=b"no such chain")

        result, calls = self._locate(make_backend("docker"), respond)
        assert result is None
        assert "snapshots view` failed" in capsys.readouterr().err
        assert "rm" in calls[-1] and not any("mounts" in c for c in calls)

    def test_mounts_failure_is_reported_and_still_cleans_up(self, capsys):
        def respond(args):
            if "mounts" in args:
                return _FakeCompleted(returncode=1, stdout="", stderr="denied")
            return _FakeCompleted()

        result, calls = self._locate(make_backend("docker"), respond)
        assert result is None
        assert "snapshots mounts` failed" in capsys.readouterr().err
        assert "rm" in calls[-1]

    def test_unparseable_mounts_output_is_reported(self, capsys):
        def respond(args):
            return _FakeCompleted(stdout="garbage" if "mounts" in args else "")

        result, _calls = self._locate(make_backend("docker"), respond)
        assert result is None
        assert "could not parse a diff directory" in capsys.readouterr().err

    def test_a_snapshot_directory_that_does_not_exist_is_reported(self, tmp_path, capsys):
        result, calls = self._locate(make_backend("docker"), self._mounts_ok(tmp_path / "missing"))
        assert result is None
        assert "does not exist" in capsys.readouterr().err
        assert "rm" in calls[-1]

    def test_unexpected_exception_degrades_to_none_and_still_cleans_up(self, capsys):
        def respond(args):
            if "mounts" in args:
                raise subprocess.TimeoutExpired(args, 1)
            return _FakeCompleted()

        result, calls = self._locate(make_backend("docker"), respond)
        assert result is None
        assert "containerd overlayfs layer lookup failed" in capsys.readouterr().err
        assert "rm" in calls[-1]

    def test_snapshot_removal_failure_is_tolerated(self, tmp_path, capsys):
        snap = tmp_path / "fs"
        snap.mkdir()
        ok = self._mounts_ok(snap)

        def respond(args):
            if "rm" in args:
                raise OSError("cannot rm")
            return ok(args)

        result, _calls = self._locate(make_backend("docker"), respond)
        assert result == snap
        assert "could not remove temporary ctr snapshot view" in capsys.readouterr().err

    def test_sudo_ctr_also_opens_the_one_snapshot_for_the_current_user(self, tmp_path):
        snap = tmp_path / "snapshots" / "9" / "fs"
        snap.mkdir(parents=True)
        result, calls = self._locate(
            make_backend("docker"), self._mounts_ok(snap), ctr=("sudo", "-n", "ctr")
        )
        assert result == snap
        assert ["sudo", "-n", "chmod", "-R", "a+rX", str(snap.parent)] in calls

    def test_plain_ctr_never_runs_chmod(self, tmp_path):
        snap = tmp_path / "fs"
        snap.mkdir()
        _result, calls = self._locate(make_backend("docker"), self._mounts_ok(snap))
        assert not any("chmod" in c for c in calls)


class TestDockerCtrProbeFailures:
    def test_a_ctr_that_cannot_launch_falls_through_to_sudo(self, capsys):
        backend = make_backend("docker")

        def fake_run(args, **kwargs):
            if args[0] == "ctr":
                raise FileNotFoundError("ctr")
            return _FakeCompleted(returncode=0)

        with patch.object(_mod._DockerBackend, "_containerd_address", return_value=None):
            with patch.object(_mod.subprocess, "run", side_effect=fake_run):
                assert backend._ctr_argv() == ["sudo", "-n", "ctr"]
        assert "ctr probe ctr failed" in capsys.readouterr().err

    def test_nothing_working_is_cached_and_warned_once(self, capsys):
        backend = make_backend("docker")
        run = Mock(return_value=_FakeCompleted(returncode=1, stderr=b"no"))
        with patch.object(_mod._DockerBackend, "_containerd_address", return_value=None):
            with patch.object(_mod.subprocess, "run", run):
                assert backend._ctr_argv() is None
                assert backend._ctr_argv() is None
        assert run.call_count == 2, "plain and sudo probed once each, then cached"
        assert capsys.readouterr().err.count("neither `ctr` nor `sudo -n ctr`") == 1


class TestDockerInfo:
    def test_info_is_asked_for_as_json_and_cached(self):
        backend = make_backend("docker")
        run = Mock(return_value=SimpleNamespace(stdout=b'{"Driver": "overlay2"}'))
        with patch.object(type(backend), "_run", run):
            assert backend._docker_info() == {"Driver": "overlay2"}
            assert backend._docker_info() == {"Driver": "overlay2"}
        run.assert_called_once()
        assert run.call_args.args[0] == ["docker", "info", "--format", "{{json .}}"]

    def test_malformed_info_is_none_and_not_retried(self, capsys):
        backend = make_backend("docker")
        run = Mock(return_value=SimpleNamespace(stdout=b"not json"))
        with patch.object(type(backend), "_run", run):
            assert backend._docker_info() is None
            assert backend._docker_info() is None
        run.assert_called_once()
        assert "`docker info` failed" in capsys.readouterr().err

    def test_data_root_and_driver_come_from_info(self):
        backend = _info_backend({"DockerRootDir": "/var/lib/docker", "Driver": "overlay2"})
        assert backend._docker_data_root_and_driver() == (Path("/var/lib/docker"), "overlay2")

    def test_missing_driver_reads_as_empty_string(self):
        backend = _info_backend({"DockerRootDir": "/var/lib/docker"})
        assert backend._docker_data_root_and_driver() == (Path("/var/lib/docker"), "")

    def test_missing_root_dir_is_none_with_a_warning(self, capsys):
        backend = _info_backend({"Driver": "overlay2"})
        assert backend._docker_data_root_and_driver() is None
        assert "missing DockerRootDir" in capsys.readouterr().err

    def test_unavailable_info_is_none(self):
        assert _info_backend(None)._docker_data_root_and_driver() is None

    @pytest.mark.parametrize(
        ("info", "rootless"),
        [
            ({"SecurityOptions": ["name=seccomp,profile=builtin", "name=rootless"]}, True),
            ({"SecurityOptions": ["name=seccomp,profile=builtin"]}, False),
            ({"SecurityOptions": None}, False),
            ({}, False),
            (None, False),
        ],
    )
    def test_rootless_is_the_documented_security_option(self, info, rootless):
        assert _info_backend(info)._is_rootless() is rootless

    @pytest.mark.parametrize(
        "info",
        [
            {"Containerd": None},
            {"Containerd": {}},
            {"Containerd": {"Address": ""}},
            {"Containerd": {"Address": None}},
        ],
    )
    def test_no_containerd_address_when_not_exposed(self, info):
        assert _info_backend(info)._containerd_address() is None


class TestRootlessIdMaps:
    """Rootless docker has no id-map API: read the dockerd process' maps."""

    @staticmethod
    def _maps(backend, pid, files):
        def parse(path):
            return files.get(str(path))

        with patch.object(type(backend), "_find_dockerd_pid", return_value=pid) as find:
            with patch.object(type(backend), "_parse_id_map", side_effect=parse):
                first = backend._rootless_id_maps()
                second = backend._rootless_id_maps()
        return first, second, find

    UID = [(0, 1000, 1), (1, 100000, 65536)]
    GID = [(0, 1000, 1), (1, 100000, 65536)]

    def test_both_maps_of_the_dockerd_process_are_used_and_cached(self):
        backend = make_backend("docker")
        files = {"/proc/42/uid_map": self.UID, "/proc/42/gid_map": self.GID}
        first, second, find = self._maps(backend, 42, files)
        assert first == second == (self.UID, self.GID)
        find.assert_called_once()

    def test_no_dockerd_process_gives_none_with_a_warning(self, capsys):
        first, _second, _find = self._maps(make_backend("docker"), None, {})
        assert first is None
        assert "dockerd's own PID could not be found" in capsys.readouterr().err

    @pytest.mark.parametrize("missing", ["uid_map", "gid_map"])
    def test_one_unreadable_map_gives_none(self, missing):
        files = {"/proc/42/uid_map": self.UID, "/proc/42/gid_map": self.GID}
        files[f"/proc/42/{missing}"] = None
        first, _second, _find = self._maps(make_backend("docker"), 42, files)
        assert first is None

    def test_ids_are_translated_end_to_end_for_a_rootless_daemon(self):
        backend = _info_backend({"SecurityOptions": ["name=rootless"]})
        files = {"/proc/42/uid_map": self.UID, "/proc/42/gid_map": [(0, 1000, 1)]}
        with patch.object(type(backend), "_find_dockerd_pid", return_value=42):
            with patch.object(type(backend), "_parse_id_map", side_effect=lambda p: files[str(p)]):
                assert backend._host_to_container_id(100005, 1000) == (6, 0)

    def test_a_rootful_daemon_never_reads_any_map(self):
        backend = _info_backend({"SecurityOptions": ["name=seccomp"]})
        with patch.object(type(backend), "_find_dockerd_pid") as find:
            assert backend._host_to_container_id(1234, 5678) == (1234, 5678)
        find.assert_not_called()


class TestParseIdMap:
    def test_short_or_blank_lines_are_skipped(self, tmp_path):
        path = tmp_path / "uid_map"
        path.write_text("0 1000 1\n\nbogus line\n1 100000\n")
        assert _mod._DockerBackend._parse_id_map(path) == [(0, 1000, 1)]

    def test_an_empty_file_is_none(self, tmp_path):
        path = tmp_path / "uid_map"
        path.write_text("")
        assert _mod._DockerBackend._parse_id_map(path) is None

    def test_non_numeric_fields_give_none_with_a_warning(self, tmp_path, capsys):
        path = tmp_path / "uid_map"
        path.write_text("0 abc 1\n")
        assert _mod._DockerBackend._parse_id_map(path) is None
        assert "could not parse" in capsys.readouterr().err


class TestOverlay2Layerdb:
    """The classic graphdriver layout (``<root>/image/overlay2/layerdb``). The
    found / not-found / missing-content contract is shared (see
    test_container_backends.py); these are the layerdb-only failure modes."""

    @staticmethod
    def _root(tmp_path):
        root = tmp_path / "docker-root"
        (root / "image" / "overlay2" / "layerdb" / "sha256").mkdir(parents=True)
        (root / "overlay2").mkdir()
        return root

    @staticmethod
    def _locate(backend, root, diff_id):
        with patch.object(
            _mod._DockerBackend, "_docker_data_root_and_driver", return_value=(root, "overlay2")
        ):
            return backend._locate_layer_diff_dir(diff_id)

    def test_missing_layerdb_is_none_with_a_warning(self, tmp_path, capsys):
        root = tmp_path / "empty-root"
        root.mkdir()
        assert self._locate(make_backend("docker"), root, "sha256:x") is None
        assert "could not list" in capsys.readouterr().err

    def test_an_unreadable_entry_is_skipped_and_a_later_match_is_found(self, tmp_path, capsys):
        root = self._root(tmp_path)
        layerdb = root / "image" / "overlay2" / "layerdb" / "sha256"
        (layerdb / "a-broken").mkdir()  # no diff / cache-id files at all
        good = layerdb / "z-good"
        good.mkdir()
        (good / "diff").write_text("sha256:want\n")
        (good / "cache-id").write_text("cache-1\n")
        (root / "overlay2" / "cache-1" / "diff").mkdir(parents=True)
        result = self._locate(make_backend("docker"), root, "sha256:want")
        assert result == root / "overlay2" / "cache-1" / "diff"
        assert "could not read layerdb entry" in capsys.readouterr().err

    def test_whitespace_around_the_stored_digest_and_cache_id_is_ignored(self, tmp_path):
        root = self._root(tmp_path)
        entry = root / "image" / "overlay2" / "layerdb" / "sha256" / "e"
        entry.mkdir()
        (entry / "diff").write_text("  sha256:ws \n")
        (entry / "cache-id").write_text("\ncache-ws\n")
        (root / "overlay2" / "cache-ws" / "diff").mkdir(parents=True)
        assert self._locate(make_backend("docker"), root, "sha256:ws") == (
            root / "overlay2" / "cache-ws" / "diff"
        )


class TestDockerRunFlags:
    """Flags only docker's run command gets (the podman-only ones are in
    test_podman_backend.py)."""

    @staticmethod
    def _fast_resume(backend):
        backend._agconfig.sandbox.checkpoint_fast_resume = True
        backend._checkpoint_storage = SimpleNamespace(
            prepare=Mock(),
            criu_ready=True,
            init_path="/zfs/init",
            criu_config="/zfs/criu.conf",
            restore_runtime_files_after_start=Mock(),
        )
        return backend

    def test_fast_resume_adds_ptrace_and_the_criu_annotation_but_no_init_path(self):
        cmd = started_run_cmd(self._fast_resume(make_backend("docker")))
        assert "--cap-add=SYS_PTRACE" in cmd
        assert cmd[cmd.index("--annotation") + 1] == "org.criu.config=/zfs/criu.conf"
        assert not any(arg.startswith("--init-path") for arg in cmd)

    def test_no_criu_flags_without_fast_resume(self):
        backend = make_backend("docker")
        backend._checkpoint_storage = SimpleNamespace(prepare=Mock(), criu_ready=True)
        cmd = started_run_cmd(backend)
        assert "--cap-add=SYS_PTRACE" not in cmd and "--annotation" not in cmd


class TestDockerNativeZfsStorage:
    """``ZfsRuntimeStorage`` for docker: the daemon itself owns a per-container
    ZFS dataset (native zfs storage driver); there is no private store."""

    @staticmethod
    def _storage(**attrs):
        storage = cp.ZfsRuntimeStorage.__new__(cp.ZfsRuntimeStorage)
        storage.runtime = "docker"
        storage.parent = "tank/docker"
        storage.dataset = None
        storage.path = None
        storage.fast_root = None
        storage.prepared = False
        storage.criu_ready = False
        storage.criu_setup_error = None
        storage._fast_resume_id = "abc"
        for key, value in attrs.items():
            setattr(storage, key, value)
        return storage

    @staticmethod
    def _inspect_backend(dataset):
        payload = [{"GraphDriver": {"Data": {"Dataset": dataset}}}]
        return SimpleNamespace(
            _name="sandbox",
            _run=Mock(return_value=SimpleNamespace(stdout=json.dumps(payload).encode())),
        )

    def test_the_command_prefix_is_plain_docker(self):
        assert self._storage().prefix == ["docker"]

    def test_the_container_owned_dataset_is_captured_from_inspect(self):
        backend = self._inspect_backend("tank/docker/container-layer")
        assert self._storage().capture_dataset(backend) == "tank/docker/container-layer"
        assert backend._run.call_args.args[0] == ["docker", "inspect", "sandbox"]

    def test_a_dataset_outside_the_configured_parent_is_rejected(self):
        with pytest.raises(cp.CheckpointCapabilityError, match="container-owned"):
            self._storage().capture_dataset(self._inspect_backend("tank/unrelated/layer"))

    def test_the_parent_dataset_itself_is_not_a_container_dataset(self):
        with pytest.raises(cp.CheckpointCapabilityError, match="container-owned"):
            self._storage().capture_dataset(self._inspect_backend("tank/docker"))

    def test_a_missing_dataset_field_is_rejected(self):
        backend = SimpleNamespace(
            _name="sandbox",
            _run=Mock(return_value=SimpleNamespace(stdout=b'[{"GraphDriver": {"Data": null}}]')),
        )
        with pytest.raises(cp.CheckpointCapabilityError):
            self._storage().capture_dataset(backend)

    def test_the_writable_dataset_identity_may_not_change(self):
        storage = self._storage()
        storage.capture_dataset(self._inspect_backend("tank/docker/one"))
        assert (
            storage.capture_dataset(self._inspect_backend("tank/docker/one")) == "tank/docker/one"
        )
        with pytest.raises(cp.CheckpointCapabilityError, match="identity changed"):
            storage.capture_dataset(self._inspect_backend("tank/docker/two"))

    @staticmethod
    def _prepare(storage, info, mount="/tank/docker", *, fast_resume=False):
        sandbox = SimpleNamespace(
            _runtime="docker",
            _agconfig=SimpleNamespace(
                sandbox=SimpleNamespace(
                    checkpoint_backend="cow_zfs", checkpoint_fast_resume=fast_resume
                )
            ),
        )

        def command(args, timeout=120):
            if args[0] == "zfs":
                return mount.encode() + b"\n"
            return json.dumps(info).encode()

        with patch.object(cp, "_command", side_effect=command), patch.object(cp, "_annotate"):
            storage.prepare(sandbox)

    def test_only_the_native_zfs_driver_is_accepted(self, tmp_path):
        info = {"DockerRootDir": str(tmp_path), "Driver": "overlay2"}
        with pytest.raises(cp.CheckpointCapabilityError, match="native zfs storage driver"):
            self._prepare(self._storage(), info, mount=str(tmp_path))

    def test_the_docker_root_must_be_the_configured_zfs_mountpoint(self, tmp_path):
        info = {"DockerRootDir": str(tmp_path / "elsewhere"), "Driver": "zfs"}
        with pytest.raises(cp.CheckpointCapabilityError, match="must be the configured"):
            self._prepare(self._storage(), info, mount=str(tmp_path))

    def test_prepare_records_the_root_and_a_per_instance_fast_resume_directory(self, tmp_path):
        storage = self._storage()
        self._prepare(storage, {"DockerRootDir": str(tmp_path), "Driver": "zfs"}, str(tmp_path))
        assert storage.prepared and storage.path == tmp_path.resolve()
        assert storage.fast_root == tmp_path.resolve() / "agency-fast-resume" / "abc"
        assert not storage.criu_ready

    def test_a_failing_criu_setup_keeps_zfs_checkpoints_enabled(self, tmp_path):
        storage = self._storage()
        with patch.object(
            cp.ZfsRuntimeStorage,
            "_prepare_criu",
            side_effect=cp.CheckpointCapabilityError("no criu"),
        ):
            self._prepare(
                storage,
                {"DockerRootDir": str(tmp_path), "Driver": "zfs"},
                str(tmp_path),
                fast_resume=True,
            )
        assert storage.prepared and not storage.criu_ready
        assert "no criu" in storage.criu_setup_error

    def test_destroy_removes_only_this_sandboxes_agency_snapshots(self):
        storage = self._storage(dataset="tank/docker/layer")
        listing = b"tank/docker/layer@agency-1\ntank/docker/layer@user-keep\n"
        with patch.object(cp, "_command", return_value=listing) as command:
            storage.destroy()
        destroyed = [
            c.args[0] for c in command.call_args_list if c.args[0][:2] == ["zfs", "destroy"]
        ]
        assert destroyed == [["zfs", "destroy", "tank/docker/layer@agency-1"]]

    def test_destroy_refuses_an_unexpected_fast_resume_path(self, tmp_path):
        fast = tmp_path / "not-the-expected-parent" / "abc"
        fast.mkdir(parents=True)
        storage = self._storage(path=tmp_path, fast_root=fast)
        with pytest.raises(cp.CheckpointCapabilityError, match="unexpected fast-resume path"):
            storage.destroy()
        assert fast.exists()

    def test_destroy_removes_its_own_fast_resume_directory_and_stays_prepared(self, tmp_path):
        fast = tmp_path / "agency-fast-resume" / "abc"
        fast.mkdir(parents=True)
        storage = self._storage(path=tmp_path, fast_root=fast, prepared=True)
        storage.destroy()
        assert not fast.exists()
        assert storage.prepared, "docker still needs the runtime to remove the container"

    @staticmethod
    def _criu(storage, info, tmp_path, *, start_help="--checkpoint", run_help="--annotation"):
        def command(args, timeout=120):
            if args[:2] == ["docker", "info"]:
                return json.dumps(info).encode()
            if args[:2] == ["docker", "start"]:
                return start_help.encode()
            if args[:2] == ["docker", "run"]:
                return run_help.encode()
            return b""

        storage.fast_root = tmp_path / "fast"
        storage.irmap_paths = []
        with patch.object(cp, "_command", side_effect=command):
            with patch.object(cp.shutil, "which", return_value="/usr/bin/x"):
                storage._prepare_criu(SimpleNamespace(_runtime="docker"))

    GOOD = {"ExperimentalBuild": True, "CgroupVersion": "2", "CgroupDriver": "systemd"}

    def test_criu_needs_an_experimental_daemon(self, tmp_path):
        with pytest.raises(cp.CheckpointCapabilityError, match="experimental"):
            self._criu(self._storage(), {**self.GOOD, "ExperimentalBuild": False}, tmp_path)

    def test_criu_needs_systemd_managed_cgroup_v2(self, tmp_path):
        with pytest.raises(cp.CheckpointCapabilityError, match="cgroup v2 and systemd"):
            self._criu(self._storage(), {**self.GOOD, "CgroupDriver": "cgroupfs"}, tmp_path)
        with pytest.raises(cp.CheckpointCapabilityError, match="cgroup v2 and systemd"):
            self._criu(self._storage(), {**self.GOOD, "CgroupVersion": "1"}, tmp_path)

    def test_criu_needs_checkpoint_restore_and_annotation_support(self, tmp_path):
        with pytest.raises(cp.CheckpointCapabilityError, match="checkpoint restore"):
            self._criu(self._storage(), self.GOOD, tmp_path, start_help="--other")
        with pytest.raises(cp.CheckpointCapabilityError, match="annotation"):
            self._criu(self._storage(), self.GOOD, tmp_path, run_help="--label")

    def test_a_capable_daemon_gets_private_restore_hooks(self, tmp_path):
        storage = self._storage()
        self._criu(storage, self.GOOD, tmp_path)
        assert oct(storage.fast_root.stat().st_mode & 0o777) == "0o700"
        assert (storage.fast_root / "restore-runtime-mounts").exists()
        assert "action-script" in (storage.fast_root / "criu.conf").read_text()


# ===========================================================================
# CRIU fast resume through ``docker checkpoint`` / ``docker start --checkpoint``
# (the podman flow is ``podman container checkpoint/restore``, see
# test_podman_backend.py)
# ===========================================================================


def test_docker_restore_retries_containerd_checkpoint_publish_race(monkeypatch):
    backend, state = make_cow_backend("docker")
    backend._agency_harness_daemon_handles = {}
    backend._acquire_runtime_slot = Mock()
    backend._release_runtime_slot = Mock()
    backend._infrastructure_pids = {}
    starts = 0

    def run(args, **_kwargs):
        nonlocal starts
        starts += 1
        if starts < 3:
            raise RuntimeError(
                "failed to upload checkpoint to containerd: commit failed: "
                "content sha256:abc: already exists"
            )
        state["running"] = True
        return subprocess.CompletedProcess(args, 0, b"sandbox\n", b"")

    backend._run = Mock(side_effect=run)
    monkeypatch.setattr(cp.time, "sleep", Mock())
    monkeypatch.setattr(cp.CowZfsCheckpoint, "_resume_daemons", Mock())
    from agency.sandbox import _criu_restore_mounts

    monkeypatch.setattr(_criu_restore_mounts, "thaw_retained_tasks", Mock())
    artifact = {
        "name": "agency-checkpoint",
        "runtime": "docker",
        "daemon_count": 0,
        "pid_tracking": {},
    }

    backend._checkpointer._runtime_restore(backend, artifact)

    assert starts == 3
    assert state["running"] is True


def test_docker_checkpoint_and_restore_use_private_checkpoint_directory(monkeypatch):
    backend, state = make_cow_backend("docker")
    backend._agency_harness_daemon_handles = {}
    backend._mark_runtime_checkpoint_stopped = Mock(side_effect=lambda: state.update(running=False))
    backend._acquire_runtime_slot = Mock()
    backend._release_runtime_slot = Mock()
    backend._infrastructure_pids = {}
    commands = []

    def run(args, **_kwargs):
        commands.append(args)
        if args[1] == "inspect":
            payload = [
                {
                    "Id": "a" * 64,
                    "Config": {"Tty": False},
                    "HostConfig": {"NetworkMode": "host"},
                    "ExecIDs": None,
                }
            ]
            return subprocess.CompletedProcess(args, 0, json.dumps(payload).encode(), b"")
        if args[1] == "checkpoint":
            return subprocess.CompletedProcess(args, 0, b"", b"")
        if args[1] == "start":
            state["running"] = True
            return subprocess.CompletedProcess(args, 0, b"sandbox\n", b"")
        raise AssertionError(args)

    backend._run = Mock(side_effect=run)
    monkeypatch.setattr(cp, "check_process_resources", lambda _sandbox: 123)
    monkeypatch.setattr(cp.time, "monotonic", Mock(side_effect=range(10)))
    monkeypatch.setattr(cp.CowZfsCheckpoint, "_resume_daemons", Mock())
    from agency.sandbox import _criu_restore_mounts

    monkeypatch.setattr(_criu_restore_mounts, "thaw_retained_tasks", Mock())

    artifact = backend._checkpointer._runtime_checkpoint(backend)
    assert artifact["runtime"] == "docker"
    assert commands[-1][:3] == [
        "docker",
        "checkpoint",
        "create",
    ]
    assert commands[-1][-2:] == ["sandbox", artifact["name"]]
    backend._checkpoint_storage.prepare_restore_mounts.assert_called_once()
    assert state["running"] is False

    backend._checkpointer._runtime_restore(backend, artifact)
    assert commands[-1] == [
        "docker",
        "start",
        "--checkpoint",
        artifact["name"],
        "sandbox",
    ]
    assert artifact["restore_stats"] == {"docker_criu_restore_seconds": 1}
    _criu_restore_mounts.thaw_retained_tasks.assert_called_once_with(
        Path("/zfs/agency-fast-resume")
    )
    assert state["running"] is True


def test_docker_fast_checkpoint_rejects_managed_network_namespace(monkeypatch):
    backend, _state = make_cow_backend("docker")
    backend._agency_harness_daemon_handles = {}
    inspection = [
        {
            "Id": "a" * 64,
            "Config": {"Tty": False},
            "HostConfig": {"NetworkMode": "bridge"},
            "ExecIDs": None,
        }
    ]
    backend._run = Mock(
        return_value=subprocess.CompletedProcess(
            ["docker", "inspect", "sandbox"], 0, json.dumps(inspection).encode(), b""
        )
    )
    monkeypatch.setattr(cp, "check_process_resources", lambda _sandbox: 123)

    with pytest.raises(cp.CheckpointCapabilityError, match="--network=host"):
        backend._checkpointer._runtime_checkpoint(backend)


def test_docker_fast_artifact_cleanup_uses_private_directory():
    backend, _state = make_cow_backend("docker")
    backend._run = Mock()
    artifact = {
        "name": "agency-checkpoint",
    }

    backend._checkpointer._delete_fast_artifact(backend, artifact)

    backend._run.assert_called_once_with(
        [
            "docker",
            "checkpoint",
            "rm",
            "sandbox",
            "agency-checkpoint",
        ],
        check=False,
        timeout=30,
    )


def test_docker_runtime_bind_files_are_repaired_from_private_backup(tmp_path):
    from agency.sandbox._criu_restore_mounts import restore_runtime_files

    runtime_root = tmp_path / "docker"
    fast_root = runtime_root / "agency-fast-resume" / "sandbox"
    container_root = runtime_root / "containers" / ("a" * 64)
    container_root.mkdir(parents=True)
    fast_root.mkdir(parents=True)
    paths = {}
    for name in ("hosts", "hostname", "resolv.conf"):
        path = container_root / name
        path.write_text("checkpoint-" + name)
        paths[name] = path
    storage = cp.ZfsRuntimeStorage.__new__(cp.ZfsRuntimeStorage)
    storage.runtime = "docker"
    storage.path = runtime_root
    storage.fast_root = fast_root
    storage.prepare_restore_mounts(
        {
            "Id": "a" * 64,
            "HostsPath": str(paths["hosts"]),
            "HostnamePath": str(paths["hostname"]),
            "ResolvConfPath": str(paths["resolv.conf"]),
        },
        "tank/docker/container@agency-checkpoint",
        [],
    )
    for path in paths.values():
        path.write_text("regenerated")

    restore_runtime_files(fast_root)

    for name, path in paths.items():
        assert path.read_text() == "checkpoint-" + name


def test_docker_restore_does_not_retry_unrelated_failures(monkeypatch):
    backend, _state = make_cow_backend("docker")
    backend._agency_harness_daemon_handles = {}
    backend._acquire_runtime_slot = Mock()
    backend._release_runtime_slot = Mock()
    backend._infrastructure_pids = {}
    backend._run = Mock(side_effect=RuntimeError("start failed: no such checkpoint"))
    monkeypatch.setattr(cp.time, "sleep", Mock())
    artifact = {"name": "agency-checkpoint", "runtime": "docker", "daemon_count": 0}
    with pytest.raises(RuntimeError, match="no such checkpoint"):
        backend._checkpointer._runtime_restore(backend, artifact)
    assert backend._run.call_count == 1


def test_docker_restore_gives_up_after_three_publish_races(monkeypatch):
    backend, _state = make_cow_backend("docker")
    backend._agency_harness_daemon_handles = {}
    backend._acquire_runtime_slot = Mock()
    backend._release_runtime_slot = Mock()
    backend._infrastructure_pids = {}
    backend._run = Mock(
        side_effect=RuntimeError("commit failed: content sha256:abc: already exists")
    )
    monkeypatch.setattr(cp.time, "sleep", Mock())
    artifact = {"name": "agency-checkpoint", "runtime": "docker", "daemon_count": 0}
    with pytest.raises(RuntimeError, match="already exists"):
        backend._checkpointer._runtime_restore(backend, artifact)
    assert backend._run.call_count == 3


def test_docker_restore_rejects_an_artifact_made_by_another_runtime():
    backend, _state = make_cow_backend("docker")
    backend._agency_harness_daemon_handles = {}
    with pytest.raises(cp.CheckpointCapabilityError, match="runtime changed"):
        backend._checkpointer._runtime_restore(
            backend, {"name": "x", "runtime": "podman", "daemon_count": 0}
        )
