"""Tier 1 (podman-only) tests for agency.sandbox.podman._PodmanBackend and
the podman halves of the shared modules.

Only mechanisms that exist for podman alone live here: ``localhost/`` image
qualification of bare names, ``podman info`` parsing (graph root, storage
driver, ``host.security.rootless``, ``host.idMappings``), the containers/storage
``layers.json`` lookup used by the fast squash path, podman's CRIU flow
(``podman container checkpoint/restore`` with the catatonit ``--init-path``) and
its private ZFS-clone storage.

Everything podman shares with docker through ``_ContainerBackendBase`` (run
flags, lifecycle, retries, quota, slot/GPU release, commit/squash/accumulator,
CDI-vs-``--gpus`` GPU flags, ``podman top`` syntax, conmon scope registration,
rootless id translation, the generic diff-dir lookup contract) is tested once
for both runtimes in test_container_backends.py; behaviour every backend shares
is in test_all_backends.py. The docker counterpart is test_docker_backend.py.

Tests that need a real Podman daemon skip automatically when it is unreachable.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import agency.sandbox.podman as _mod
from agency.sandbox import checkpoint as cp
from tests.sandbox.conftest import (
    backend_available,
    make_backend,
    make_cow_backend,
    started_run_cmd,
)

podman_daemon = pytest.mark.skipif(
    not backend_available("podman"), reason="Podman daemon not reachable"
)


def _info_backend(info, **kwargs):
    """A podman backend whose ``podman info`` JSON is *info* (None = failing)."""
    backend = make_backend("podman", **kwargs)
    patch.object(type(backend), "_podman_info", return_value=info).start()
    return backend


@pytest.fixture(autouse=True)
def _stop_info_patches():
    yield
    patch.stopall()


def _entries(rows):
    return [{"container_id": c, "host_id": h, "size": s} for c, h, s in rows]


# ===========================================================================
# containers/storage layers.json
# ===========================================================================


class TestLocateLayerDiffDir:
    """podman-only parts of ``_PodmanBackend._locate_layer_diff_dir``
    (``layers.json`` is append-ordered and may be missing). The shared
    found/not-found/unsupported-driver contract is in
    test_container_backends.py::TestLocateLayerDiffDir."""

    def _fake_podman_root(self, tmp_path, *, driver="overlay"):
        root = tmp_path / "podman-root"
        (root / "overlay-layers").mkdir(parents=True)
        (root / "overlay").mkdir(parents=True)
        (root / "overlay-layers" / "layers.json").write_text("[]")
        return root

    def _add_layer_entry(self, root, diff_id, layer_id, *, with_content=True):
        layers_path = root / "overlay-layers" / "layers.json"
        layers = json.loads(layers_path.read_text())
        layers.append({"id": layer_id, "diff-digest": diff_id})
        layers_path.write_text(json.dumps(layers))
        if with_content:
            diff_dir = root / "overlay" / layer_id / "diff"
            diff_dir.mkdir(parents=True)
            (diff_dir / "marker").write_text("x")
        return layer_id

    def test_prefers_last_matching_entry_when_digest_duplicated(self, tmp_path):
        """Empty layers commonly share a digest; layers.json is
        append-ordered, so the last match is the newest."""
        import agency.sandbox.podman as _mod

        sb = make_backend("podman")
        root = self._fake_podman_root(tmp_path)
        self._add_layer_entry(root, "sha256:shared", "layer-older")
        self._add_layer_entry(root, "sha256:shared", "layer-newer")

        with patch.object(
            _mod._PodmanBackend, "_podman_graph_root_and_driver", return_value=(root, "overlay")
        ):
            result = sb._locate_layer_diff_dir("sha256:shared")

        assert result == root / "overlay" / "layer-newer" / "diff"

    def test_returns_none_when_layers_json_missing(self, tmp_path):
        import agency.sandbox.podman as _mod

        sb = make_backend("podman")
        root = self._fake_podman_root(tmp_path)
        (root / "overlay-layers" / "layers.json").unlink()

        with patch.object(
            _mod._PodmanBackend, "_podman_graph_root_and_driver", return_value=(root, "overlay")
        ):
            result = sb._locate_layer_diff_dir("sha256:anything")

        assert result is None

    @podman_daemon
    def test_real_commit_layer_locates_via_live_storage(self):
        """End-to-end against a real podman commit: the top RootFS.Layers
        digest must resolve to a real overlay diff dir containing the
        file we wrote (no mocking of info/storage)."""
        name = f"test-locate-{uuid.uuid4().hex[:8]}"
        tag = f"localhost/agency-test-locate-{uuid.uuid4().hex[:8]}:latest"
        subprocess.run(
            ["podman", "run", "-d", "--name", name, "alpine:latest", "sleep", "3600"],
            capture_output=True,
            check=True,
        )
        sb = make_backend("podman")
        try:
            subprocess.run(
                ["podman", "exec", name, "sh", "-c", "echo locate-marker > /tmp/locate.txt"],
                capture_output=True,
                check=True,
            )
            subprocess.run(["podman", "commit", name, tag], capture_output=True, check=True)
            layers = json.loads(
                subprocess.run(
                    ["podman", "inspect", "--format={{json .RootFS.Layers}}", tag],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
            )
            diff_dir = sb._locate_layer_diff_dir(layers[-1])
            assert diff_dir is not None, f"failed to locate diff dir for {layers[-1]}"
            assert (diff_dir / "tmp" / "locate.txt").is_file()
            assert (diff_dir / "tmp" / "locate.txt").read_text() == "locate-marker\n"
        finally:
            subprocess.run(["podman", "rm", "-f", name], capture_output=True)
            subprocess.run(["podman", "rmi", "-f", tag], capture_output=True)


def test_dump_and_restore_criu_stats_use_distinct_keys():
    raw = '{"podman_checkpoint_duration": 123}'

    assert cp._criu_stats(raw) == {"podman_criu_stats": {"podman_checkpoint_duration": 123}}
    assert cp._criu_stats(raw, "podman_criu_restore_stats") == {
        "podman_criu_restore_stats": {"podman_checkpoint_duration": 123}
    }


class TestLayersJsonEdgeCases:
    """Shapes ``<graphRoot>/overlay-layers/layers.json`` can take in the wild."""

    @staticmethod
    def _root(tmp_path, content):
        root = tmp_path / "podman-root"
        (root / "overlay-layers").mkdir(parents=True)
        (root / "overlay").mkdir()
        layers = root / "overlay-layers" / "layers.json"
        if isinstance(content, bytes):
            layers.write_bytes(content)
        else:
            layers.write_text(content if isinstance(content, str) else json.dumps(content))
        return root

    @staticmethod
    def _locate(root, diff_id="sha256:want", driver="overlay", **kwargs):
        backend = make_backend("podman")
        with patch.object(
            _mod._PodmanBackend, "_podman_graph_root_and_driver", return_value=(root, driver)
        ):
            return backend._locate_layer_diff_dir(diff_id, **kwargs)

    @staticmethod
    def _content(root, layer_id):
        (root / "overlay" / layer_id / "diff").mkdir(parents=True)
        return root / "overlay" / layer_id / "diff"

    def test_invalid_json_is_none_with_a_warning(self, tmp_path, capsys):
        assert self._locate(self._root(tmp_path, "{not json")) is None
        assert "could not read/parse" in capsys.readouterr().err

    def test_undecodable_bytes_are_none_with_a_warning(self, tmp_path, capsys):
        assert self._locate(self._root(tmp_path, b"\xff\xfe\x00bad")) is None
        assert "could not read/parse" in capsys.readouterr().err

    def test_json_null_is_none_with_a_warning(self, tmp_path, capsys):
        assert self._locate(self._root(tmp_path, "null")) is None
        assert "WARNING" in capsys.readouterr().err

    @pytest.mark.parametrize("content", [{"a": 1}, [1, 2, 3], ["text"]])
    def test_an_unexpected_shape_is_none_with_a_warning(self, tmp_path, capsys, content):
        assert self._locate(self._root(tmp_path, content)) is None
        assert "unexpected shape" in capsys.readouterr().err

    def test_entries_without_a_digest_are_ignored(self, tmp_path):
        root = self._root(tmp_path, [{"id": "no-digest"}, {"id": "other", "diff-digest": "x"}])
        assert self._locate(root) is None

    def test_a_matching_entry_without_an_id_is_none(self, tmp_path, capsys):
        root = self._root(tmp_path, [{"diff-digest": "sha256:want"}])
        assert self._locate(root) is None
        assert "no layers.json entry found" in capsys.readouterr().err

    def test_a_matching_entry_with_an_empty_id_is_none(self, tmp_path):
        assert (
            self._locate(self._root(tmp_path, [{"diff-digest": "sha256:want", "id": ""}])) is None
        )

    def test_the_newest_duplicate_wins_even_if_an_older_one_has_content(self, tmp_path, capsys):
        """The last matching entry is chosen; an older sibling is not a fallback."""
        root = self._root(
            tmp_path,
            [
                {"diff-digest": "sha256:want", "id": "older"},
                {"diff-digest": "sha256:want", "id": "newer"},
            ],
        )
        self._content(root, "older")
        assert self._locate(root) is None
        assert "doesn't exist" in capsys.readouterr().err

    def test_extra_fields_and_unrelated_entries_are_ignored(self, tmp_path):
        root = self._root(
            tmp_path,
            [
                {"id": "base", "diff-digest": "sha256:base", "parent": None, "created": "t"},
                {"id": "mine", "diff-digest": "sha256:want", "parent": "base", "size": 7},
                {"id": "later", "diff-digest": "sha256:later"},
            ],
        )
        expected = self._content(root, "mine")
        assert self._locate(root) == expected

    def test_the_diff_ids_chain_is_accepted_and_ignored(self, tmp_path):
        """Docker needs the chain for ChainID; podman keys layers by DiffID."""
        root = self._root(tmp_path, [{"id": "mine", "diff-digest": "sha256:want"}])
        expected = self._content(root, "mine")
        assert self._locate(root, diff_ids=["sha256:a", "sha256:want"]) == expected
        assert self._locate(root, diff_ids=["unrelated"]) == expected

    @pytest.mark.parametrize("driver", ["vfs", "btrfs", "zfs", ""])
    def test_only_the_overlay_driver_is_supported(self, tmp_path, capsys, driver):
        root = self._root(tmp_path, [{"id": "mine", "diff-digest": "sha256:want"}])
        self._content(root, "mine")
        assert self._locate(root, driver=driver) is None
        assert "unsupported podman storage driver" in capsys.readouterr().err

    def test_docker_style_overlayfs_driver_is_not_accepted_by_podman(self, tmp_path):
        root = self._root(tmp_path, [{"id": "mine", "diff-digest": "sha256:want"}])
        self._content(root, "mine")
        assert self._locate(root, driver="overlayfs") is None


class TestPodmanInfo:
    def test_info_is_asked_for_as_json_and_cached(self):
        backend = make_backend("podman")
        run = Mock(return_value=SimpleNamespace(stdout=b'{"store": {}}'))
        with patch.object(type(backend), "_run", run):
            assert backend._podman_info() == {"store": {}}
            assert backend._podman_info() == {"store": {}}
        run.assert_called_once()
        assert run.call_args.args[0] == ["podman", "info", "--format", "{{json .}}"]

    def test_malformed_info_is_none_and_not_retried(self, capsys):
        backend = make_backend("podman")
        run = Mock(return_value=SimpleNamespace(stdout=b"<html>"))
        with patch.object(type(backend), "_run", run):
            assert backend._podman_info() is None
            assert backend._podman_info() is None
        run.assert_called_once()
        assert "`podman info` failed" in capsys.readouterr().err

    def test_graph_root_and_driver_come_from_the_store_section(self):
        backend = _info_backend(
            {"store": {"graphRoot": "/var/lib/containers/storage", "graphDriverName": "overlay"}}
        )
        assert backend._podman_graph_root_and_driver() == (
            _mod.Path("/var/lib/containers/storage"),
            "overlay",
        )

    def test_a_missing_driver_name_reads_as_empty(self):
        backend = _info_backend({"store": {"graphRoot": "/g"}})
        assert backend._podman_graph_root_and_driver() == (_mod.Path("/g"), "")

    @pytest.mark.parametrize("info", [{}, {"store": {}}, {"store": None}, {"host": {}}])
    def test_a_missing_store_or_graph_root_is_none_with_a_warning(self, info, capsys):
        assert _info_backend(info)._podman_graph_root_and_driver() is None
        assert "missing store.graphRoot" in capsys.readouterr().err

    def test_unavailable_info_is_none(self):
        assert _info_backend(None)._podman_graph_root_and_driver() is None

    @pytest.mark.parametrize(
        ("info", "rootless"),
        [
            ({"host": {"security": {"rootless": True}}}, True),
            ({"host": {"security": {"rootless": False}}}, False),
            ({"host": {"security": {}}}, False),
            ({"host": {}}, False),
            ({"host": {"security": None}}, False),
            ({}, False),
            (None, False),
        ],
    )
    def test_rootless_is_host_security_rootless(self, info, rootless):
        assert _info_backend(info)._is_rootless() is rootless


class TestPodmanIdMaps:
    """Podman publishes the rootless id maps itself, in ``host.idMappings``."""

    UID = [(0, 1000, 1), (1, 100000, 65536)]

    @staticmethod
    def _info(mappings, rootless=True):
        return {"host": {"security": {"rootless": rootless}, "idMappings": mappings}}

    def test_entries_are_converted_to_namespace_host_length_tuples(self):
        backend = _info_backend(
            self._info({"uidmap": _entries(self.UID), "gidmap": _entries([(0, 1000, 1)])})
        )
        assert backend._rootless_id_maps() == (self.UID, [(0, 1000, 1)])

    def test_numeric_strings_are_accepted(self):
        rows = [{"container_id": "0", "host_id": "1000", "size": "1"}]
        backend = _info_backend(self._info({"uidmap": rows, "gidmap": rows}))
        assert backend._rootless_id_maps() == ([(0, 1000, 1)], [(0, 1000, 1)])

    @pytest.mark.parametrize(
        "mappings",
        [
            {},
            {"uidmap": _entries([(0, 1, 1)])},
            {"gidmap": _entries([(0, 1, 1)])},
            {"uidmap": [], "gidmap": _entries([(0, 1, 1)])},
            {"uidmap": None, "gidmap": None},
        ],
    )
    def test_missing_or_empty_maps_give_none_with_a_warning(self, mappings, capsys):
        assert _info_backend(self._info(mappings))._rootless_id_maps() is None
        assert "no usable uidmap/gidmap" in capsys.readouterr().err

    def test_a_malformed_entry_gives_none_with_a_warning(self, capsys):
        bad = [{"container_id": 0, "host_id": "x", "size": 1}]
        backend = _info_backend(self._info({"uidmap": bad, "gidmap": bad}))
        assert backend._rootless_id_maps() is None
        assert "could not parse" in capsys.readouterr().err

    def test_missing_idmappings_key_gives_none_with_a_warning(self, capsys):
        backend = _info_backend({"host": {"security": {"rootless": True}}})
        assert backend._rootless_id_maps() is None
        assert "could not parse" in capsys.readouterr().err

    def test_the_maps_are_computed_once(self):
        info = self._info({"uidmap": _entries(self.UID), "gidmap": _entries(self.UID)})
        backend = make_backend("podman")
        info_mock = Mock(return_value=info)
        with patch.object(type(backend), "_podman_info", info_mock):
            first = backend._rootless_id_maps()
            assert backend._rootless_id_maps() is first
        assert info_mock.call_count == 1

    def test_ids_are_translated_end_to_end_for_a_rootless_podman(self):
        backend = _info_backend(
            self._info({"uidmap": _entries(self.UID), "gidmap": _entries([(0, 1000, 1)])})
        )
        assert backend._host_to_container_id(100005, 1000) == (6, 0)

    def test_a_rootful_podman_translates_nothing_even_with_maps_present(self):
        backend = _info_backend(
            self._info({"uidmap": _entries(self.UID), "gidmap": _entries(self.UID)}, rootless=False)
        )
        assert backend._host_to_container_id(100005, 100005) == (100005, 100005)


class TestResolveImage:
    @pytest.mark.parametrize(
        ("name", "resolved"),
        [
            ("alpine", "localhost/alpine"),
            ("agency/lifecycle-x", "agency/lifecycle-x"),
            ("docker.io/library/python:3.12", "docker.io/library/python:3.12"),
            ("localhost/already:latest", "localhost/already:latest"),
            ("sha256:" + "a" * 64, "sha256:" + "a" * 64),
            ("a" * 64, "a" * 64),
        ],
    )
    def test_only_bare_names_get_the_localhost_prefix(self, name, resolved):
        assert make_backend("podman")._resolve_image(name) == resolved

    def test_a_short_hex_string_is_a_name_not_an_image_id(self):
        assert make_backend("podman")._resolve_image("abc123") == "localhost/abc123"

    def test_a_sixty_four_char_name_that_is_not_hex_is_still_prefixed(self):
        assert make_backend("podman")._resolve_image("z" * 64) == "localhost/" + "z" * 64


class TestPodmanRunFlags:
    @staticmethod
    def _fast_resume(backend):
        backend._agconfig.sandbox.checkpoint_fast_resume = True
        backend._checkpoint_storage = SimpleNamespace(
            prepare=Mock(),
            criu_ready=True,
            init_path="/zfs/catatonit",
            criu_config="/zfs/criu.conf",
        )
        return backend

    def test_fast_resume_adds_ptrace_the_catatonit_init_and_the_criu_annotation(self):
        cmd = started_run_cmd(self._fast_resume(make_backend("podman")))
        assert "--cap-add=SYS_PTRACE" in cmd
        assert "--init-path=/zfs/catatonit" in cmd
        assert cmd[cmd.index("--annotation") + 1] == "org.criu.config=/zfs/criu.conf"

    def test_the_cgroup_parent_flag_is_never_passed_to_podman(self):
        with patch(
            "agency.sandbox.container.agprof.container_cgroup_parent", return_value="p.slice"
        ):
            cmd = started_run_cmd(make_backend("podman"))
        assert not any(arg.startswith("--cgroup-parent") for arg in cmd)

    def test_the_run_uses_a_fully_qualified_image(self):
        backend = make_backend("podman", base_image="python:3.12-slim")
        assert "localhost/python:3.12-slim" in started_run_cmd(backend)


class TestPodmanPrivateZfsStorage:
    """``ZfsRuntimeStorage`` for podman: a private ZFS clone per sandbox holds a
    whole podman store (graphroot/runroot/tmp), so every call carries
    ``--root`` flags."""

    @staticmethod
    def _storage(**attrs):
        storage = cp.ZfsRuntimeStorage.__new__(cp.ZfsRuntimeStorage)
        storage.runtime = "podman"
        storage.parent = "tank/agency"
        storage.dataset = "tank/agency/agency-1234"
        storage.path = None
        storage.fast_root = None
        storage.prepared = False
        storage.created = False
        storage.criu_ready = False
        storage.criu_setup_error = None
        storage.irmap_paths = []
        for key, value in attrs.items():
            setattr(storage, key, value)
        return storage

    def test_the_private_dataset_is_known_without_asking_the_runtime(self):
        backend = SimpleNamespace(_run=Mock())
        assert self._storage().capture_dataset(backend) == "tank/agency/agency-1234"
        backend._run.assert_not_called()

    def test_commands_need_a_provisioned_store(self):
        with pytest.raises(cp.CheckpointCapabilityError, match="not been provisioned"):
            self._storage().prefix

    def test_the_prefix_points_podman_at_the_private_store(self, tmp_path):
        prefix = self._storage(path=tmp_path).prefix
        assert prefix[:2] == ["podman", "--remote=false"]
        assert prefix[prefix.index("--root") + 1] == str(tmp_path / "graphroot")
        assert prefix[prefix.index("--runroot") + 1] == str(tmp_path / "runroot")
        assert prefix[prefix.index("--tmpdir") + 1] == str(tmp_path / "tmp")
        assert "--storage-driver=overlay" in prefix and "--runtime=runc" in prefix
        assert "--storage-opt=overlay.mount_program=" in prefix, "FUSE overlay is rejected"

    @staticmethod
    def _criu(
        storage,
        tmp_path,
        *,
        cgroups="systemd v2",
        help_text="--keep --file-locks --tcp-established --print-stats",
        init=True,
    ):
        init_path = tmp_path / "catatonit"
        if init:
            init_path.write_text("")
        storage.init_path = str(init_path)
        storage.fast_root = tmp_path / "fast"

        def command(args, timeout=120):
            if args[:2] == ["podman", "info"]:
                return cgroups.encode()
            if args[:3] == ["podman", "container", "checkpoint"] or args[:3] == [
                "podman",
                "container",
                "restore",
            ]:
                return help_text.encode()
            return b""

        with patch.object(cp, "_command", side_effect=command):
            with patch.object(cp.shutil, "which", return_value="/usr/bin/x"):
                storage._prepare_criu(SimpleNamespace(_runtime="podman"))

    def test_criu_needs_the_catatonit_init_binary(self, tmp_path):
        with pytest.raises(cp.CheckpointCapabilityError, match="catatonit"):
            self._criu(self._storage(), tmp_path, init=False)

    @pytest.mark.parametrize("cgroups", ["cgroupfs v2", "systemd v1", "cgroupfs v1"])
    def test_criu_needs_systemd_managed_cgroup_v2(self, tmp_path, cgroups):
        with pytest.raises(cp.CheckpointCapabilityError, match="cgroup v2 and systemd"):
            self._criu(self._storage(), tmp_path, cgroups=cgroups)

    @pytest.mark.parametrize(
        "flag", ["--keep", "--file-locks", "--tcp-established", "--print-stats"]
    )
    def test_criu_needs_every_checkpoint_restore_flag(self, tmp_path, flag):
        flags = "--keep --file-locks --tcp-established --print-stats".replace(flag, "")
        with pytest.raises(cp.CheckpointCapabilityError, match=f"lacks {flag}"):
            self._criu(self._storage(), tmp_path, help_text=flags)

    def test_a_capable_podman_gets_private_restore_hooks(self, tmp_path):
        storage = self._storage()
        self._criu(storage, tmp_path)
        assert (storage.fast_root / "restore-runtime-mounts").exists()
        assert "action-script" in (storage.fast_root / "criu.conf").read_text()

    def test_destroy_retries_only_a_busy_dataset(self):
        storage = self._storage(created=True, prepared=True)
        busy = cp.CheckpointCapabilityError("cannot destroy: dataset is busy")
        with patch.object(cp, "_command", side_effect=[busy, busy, b""]) as command:
            with patch.object(cp.time, "sleep"):
                storage.destroy()
        assert command.call_count == 3
        assert command.call_args.args[0] == ["zfs", "destroy", "-r", "tank/agency/agency-1234"]
        assert not storage.created and not storage.prepared

    def test_destroy_does_not_retry_other_errors(self):
        storage = self._storage(created=True, prepared=True)
        with patch.object(
            cp, "_command", side_effect=cp.CheckpointCapabilityError("permission denied")
        ) as command:
            with pytest.raises(cp.CheckpointCapabilityError, match="permission denied"):
                storage.destroy()
        assert command.call_count == 1

    def test_destroy_is_a_noop_for_a_dataset_that_was_never_created(self):
        with patch.object(cp, "_command") as command:
            self._storage().destroy()
        command.assert_not_called()


# ===========================================================================
# CRIU fast resume through ``podman container checkpoint`` / ``restore``
# (docker's flow is ``docker checkpoint create`` / ``start --checkpoint``, see
# test_docker_backend.py)
# ===========================================================================

_PODMAN_CHECKPOINT = [
    "podman",
    "container",
    "checkpoint",
    "--keep",
    "--file-locks",
    "--tcp-established",
    "--print-stats",
    "sandbox",
]
_PODMAN_RESTORE = [
    "podman",
    "container",
    "restore",
    "--keep",
    "--file-locks",
    "--tcp-established",
    "--print-stats",
    "sandbox",
]


def _criu_backend(monkeypatch, *, network="bridge"):
    backend, state = make_cow_backend("podman")
    backend._agency_harness_daemon_handles = {}
    backend._mark_runtime_checkpoint_stopped = Mock(side_effect=lambda: state.update(running=False))
    backend._acquire_runtime_slot = Mock()
    backend._release_runtime_slot = Mock()
    backend._infrastructure_pids = {}
    commands = []

    def run(args, **_kwargs):
        commands.append(list(args))
        if args[1] == "inspect":
            payload = [
                {
                    "Id": "a" * 64,
                    "Config": {"Tty": False},
                    "HostConfig": {"NetworkMode": network},
                    "ExecIDs": None,
                }
            ]
            return subprocess.CompletedProcess(args, 0, json.dumps(payload).encode(), b"")
        if args[:3] == ["podman", "container", "checkpoint"]:
            return subprocess.CompletedProcess(args, 0, b'{"podman_checkpoint_duration": 7}', b"")
        if args[:3] == ["podman", "container", "restore"]:
            state["running"] = True
            return subprocess.CompletedProcess(args, 0, b'{"restore_duration": 3}', b"")
        raise AssertionError(args)

    backend._run = Mock(side_effect=run)
    monkeypatch.setattr(cp, "check_process_resources", lambda _sandbox: 123)
    monkeypatch.setattr(cp.CowZfsCheckpoint, "_resume_daemons", Mock())
    from agency.sandbox import _criu_restore_mounts

    monkeypatch.setattr(_criu_restore_mounts, "thaw_retained_tasks", Mock())
    return backend, state, commands


def test_podman_checkpoint_and_restore_use_the_podman_container_commands(monkeypatch):
    backend, state, commands = _criu_backend(monkeypatch)
    artifact = backend._checkpointer._runtime_checkpoint(backend)
    assert commands[-1] == _PODMAN_CHECKPOINT
    assert artifact["runtime"] == "podman"
    assert artifact["podman_criu_stats"] == {"podman_checkpoint_duration": 7}
    backend._checkpoint_storage.prepare_restore_mounts.assert_called_once()
    assert state["running"] is False

    backend._checkpointer._runtime_restore(backend, artifact)
    assert commands[-1] == _PODMAN_RESTORE
    assert artifact["restore_stats"] == {"podman_criu_restore_stats": {"restore_duration": 3}}
    assert state["running"] is True


def test_podman_does_not_need_host_networking_for_fast_checkpoints(monkeypatch):
    """Only docker's managed network namespace is rejected."""
    backend, _state, commands = _criu_backend(monkeypatch, network="bridge")
    backend._checkpointer._runtime_checkpoint(backend)
    assert commands[-1] == _PODMAN_CHECKPOINT


def test_podman_restore_does_not_retry_a_containerd_style_publish_race(monkeypatch):
    backend, _state, _commands = _criu_backend(monkeypatch)
    backend._run = Mock(side_effect=RuntimeError("content sha256:abc: already exists"))
    monkeypatch.setattr(cp.time, "sleep", Mock())
    with pytest.raises(RuntimeError, match="already exists"):
        backend._checkpointer._runtime_restore(
            backend, {"name": "n", "runtime": "podman", "daemon_count": 0, "pid_tracking": {}}
        )
    assert backend._run.call_count == 1


def test_podman_restore_rejects_an_artifact_made_by_another_runtime(monkeypatch):
    backend, _state, _commands = _criu_backend(monkeypatch)
    with pytest.raises(cp.CheckpointCapabilityError, match="runtime changed"):
        backend._checkpointer._runtime_restore(
            backend, {"name": "n", "runtime": "docker", "daemon_count": 0}
        )


def test_podman_has_no_runtime_side_artifact_to_delete():
    """docker keeps a named checkpoint in the daemon; podman's lives in the
    private ZFS store, so removing the fast artifact is a no-op."""
    backend, _state = make_cow_backend("podman")
    backend._run = Mock()
    backend._checkpointer._delete_fast_artifact(backend, {"name": "agency-checkpoint"})
    backend._run.assert_not_called()


def test_a_failed_podman_dump_leaves_the_container_resumed(monkeypatch):
    backend, state, _commands = _criu_backend(monkeypatch)
    inner = backend._run.side_effect

    def run(args, **kwargs):
        if args[:3] == ["podman", "container", "checkpoint"]:
            raise RuntimeError("criu dump failed")
        return inner(args, **kwargs)

    backend._run = Mock(side_effect=run)
    with pytest.raises(RuntimeError, match="criu dump failed"):
        backend._checkpointer._runtime_checkpoint(backend)
    assert state["running"] is True
    backend._mark_runtime_checkpoint_stopped.assert_not_called()
