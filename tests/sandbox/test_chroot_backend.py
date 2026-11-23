"""Tests for agency.sandbox.chroot._ChrootBackend (tier 1 and the
non-container family tier 2).

Chroot is the only member of the non-container family, so this one file holds
both the behaviours that family implements itself (section 1: lifecycle,
process-group tracking, GPU release gating, snapshot-directory images) and the
mechanisms unique to chroot (section 2: ``unshare``/``rootlesskit`` probing,
the jail setup script, rbind/devpts/per-file ``/dev`` mounts, GPU device
paths, environment inheritance...). Anything the container backends share
with chroot (exec, file I/O, mounts, isolation, checkpoints...) is asserted
for every backend in test_all_backends.py instead.

Tests that actually enter a jail are marked ``@chroot`` and skip when
unprivileged user namespaces are not usable (``chroot_available()``). The rest
are mock- or filesystem-only and always run.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import agency.sandbox.chroot as _chroot_mod
from agency.sandbox.chroot import (
    _CHROOT_RO_BASE_DIRS,
    _CHROOT_SNAPSHOTS_DIR,
    _ChrootBackend,
    _chroot_gpu_dev_paths,
    _sanitize_tag,
    chroot_available,
)
from tests.sandbox.conftest import make_backend

chroot = pytest.mark.skipif(
    not chroot_available(), reason="unprivileged user namespaces not usable"
)


def _make_backend(**kwargs):
    return make_backend("chroot", **kwargs)


# ===========================================================================
# 1. NON-CONTAINER FAMILY BEHAVIOURS
#    What a backend that is not a container does for the agSandbox contract,
#    done here with process groups and directory copies.
# ===========================================================================


class TestDetachedExecution:
    # pty allocation and python semaphores inside a jail are covered for every
    # backend by test_all_backends.py::TestExec.
    def test_launch_failure_is_reported(self, monkeypatch):
        sb = _make_backend()
        monkeypatch.setattr(sb, "_exec_with_pid_tracking", lambda *args: ("unshare failed", 1))
        with pytest.raises(RuntimeError, match="unshare failed"):
            sb.exec_detached("sleep 20")

    @chroot
    def test_daemon_returns_immediately_and_is_owned_for_teardown(self):
        sb = _make_backend()
        try:
            started = time.monotonic()
            sb.exec_detached("sleep 20")
            assert time.monotonic() - started < 5
            assert sb.get_live_pids()
            groups = set(sb._invocation_pgids)
            sb.stop()
            sb._invocation_pgids = groups
            deadline = time.monotonic() + 3
            while sb.get_live_pids() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert not sb.get_live_pids()
        finally:
            sb.destroy()


# ---------------------------------------------------------------------------
# _own_host_pids -- chroot processes run directly on the host (no PID
# namespace to translate through, unlike _ContainerBackendBase's version),
# and now delegates straight to get_live_pids() (PGID-matched -- see
# chroot.py's module docstring), so a fake /proc table via monkeypatched
# _read_proc_table() is enough to test this, no real chroot jail needed.
# GPU release itself no longer waits on anything backend-specific at all:
# release_gpu() just releases immediately, called after stop()/destroy()'s
# own kill+teardown has already run synchronously (see
# agresources.release_gpu()'s docstring).
# ---------------------------------------------------------------------------


class TestOwnHostPids:
    def test_matches_pgid_matched_pids(self, monkeypatch):
        sb = _make_backend()
        sb._invocation_pgids = {4242}
        monkeypatch.setattr(
            sb,
            "_read_proc_table",
            lambda script, timeout: ("111 1 4242 S\n222 1 4242 S\n", 0),
        )
        assert sb._own_host_pids() == {111, 222}

    def test_empty_when_nothing_tracked(self):
        sb = _make_backend()
        assert sb._own_host_pids() == set()


# ---------------------------------------------------------------------------
# /dev -- individual per-file bind mounts (not a whole-directory bind, which
# silently fails identically to /proc -- see module docstring), verified
# against a real jail rather than just checking the setup script's text.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _ChrootBackend -- functional (requires unprivileged userns + chroot)
# ---------------------------------------------------------------------------


@chroot
class TestChrootBackendLifecycle:
    def test_destroy_removes_jail_directory(self):
        sb = _make_backend()
        sb.exec("true")
        root = sb._root
        assert root.exists()
        sb.destroy()
        assert not root.exists()

    @staticmethod
    def _find_marked_pid(pids, marker):
        """Identify the specific pid matching *marker* among *pids* --
        blindly grabbing an arbitrary tracked pid (e.g. next(iter(...)))
        is unsafe: a single tracked process group can contain more than one
        pid (the wrapping shell alongside the actual backgrounded command),
        so the marker is what pins down the specific one being tested."""
        for pid in pids:
            try:
                cmdline = open(f"/proc/{pid}/cmdline", "rb").read()
            except OSError:
                continue
            if marker.encode() in cmdline:
                return pid
        return None

    def test_destroy_kills_background_process_before_removing_root(self):
        """Regression: destroy() used to just clear _watched_pids and
        rmtree the jail root, never actually sending a background process
        any signal -- it would keep running independently, chrooted into a
        directory tree that had just been deleted out from under it. Unlike
        the container backend, nothing here does this implicitly (there's
        no container-removal-equivalent hard kill for a chroot jail), so
        destroy() must explicitly terminate anything still running first."""
        sb = _make_backend()
        marker = f"agencytest{uuid.uuid4().hex[:8]}"
        try:
            sb.exec(f"exec -a {marker} sleep 30 &")
            tracked_pid = self._find_marked_pid(sb.get_live_pids(), marker)
            assert tracked_pid is not None, (
                f"expected the marked sleep to be tracked, got {sb.get_live_pids()}"
            )
        finally:
            sb.destroy()

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and os.path.exists(f"/proc/{tracked_pid}"):
            time.sleep(0.1)
        assert not os.path.exists(f"/proc/{tracked_pid}"), (
            "backgrounded process must not survive destroy()"
        )


# ---------------------------------------------------------------------------
# GPU semaphore release in stop()/destroy() -- unlike the container backend,
# chroot has no daemon process to inspect for "confirmed torn down", so there
# is no such gate here: the kill attempt in _kill_all_sandbox_processes() IS
# the confirmation (see its docstring), and release always follows it
# unconditionally. No real chroot required --
# _kill_all_sandbox_processes()/_materialize_workspace() aren't invoked with
# a real jail here, only stop()/destroy()'s own release-ordering logic is
# exercised.
# ---------------------------------------------------------------------------


class TestChrootGpuReleaseGating:
    def _lease_gpu(self, sb, gpu_ids=(3,)):
        released = []
        sb._virtual_gpu_count_requested = len(gpu_ids)
        sb._real_gpu_ids = list(gpu_ids)
        sb._real_gpu_release_fn = lambda ids: released.extend(ids)
        return released

    def test_stop_releases_gpu_after_kill_attempt(self, monkeypatch):
        sb = _make_backend()
        released = self._lease_gpu(sb)
        monkeypatch.setattr(sb, "_kill_all_sandbox_processes", lambda: None)

        sb.stop()

        assert released == [3]
        assert sb._real_gpu_ids == []

    def test_kill_runs_before_release(self, monkeypatch):
        order = []
        sb = _make_backend()
        sb._virtual_gpu_count_requested = 1
        sb._real_gpu_ids = [3]
        sb._real_gpu_release_fn = lambda ids: order.append(("release", ids))
        monkeypatch.setattr(sb, "_kill_all_sandbox_processes", lambda: order.append(("kill",)))

        sb.stop()

        assert order == [("kill",), ("release", [3])]

    def test_gpu_released_exactly_once_across_stop_then_destroy(self, monkeypatch):
        sb = _make_backend()
        released = self._lease_gpu(sb)
        monkeypatch.setattr(sb, "_kill_all_sandbox_processes", lambda: None)

        sb.stop()
        sb.destroy()

        assert released == [3], "GPU must be released exactly once, not once per call"

    def test_destroy_releases_gpu_exactly_once(self, monkeypatch):
        """destroy() calls rm_container() internally -- confirm that single
        call chain still only releases once, not via some hidden double
        invocation."""
        sb = _make_backend()
        released = self._lease_gpu(sb)
        monkeypatch.setattr(sb, "_kill_all_sandbox_processes", lambda: None)

        sb.destroy()

        assert released == [3]

    def test_stop_releases_multiple_held_gpu_ids(self, monkeypatch):
        sb = _make_backend()
        released = self._lease_gpu(sb, gpu_ids=(2, 4))
        monkeypatch.setattr(sb, "_kill_all_sandbox_processes", lambda: None)

        sb.stop()

        assert released == [2, 4]
        assert sb._real_gpu_ids == []


class TestChrootDestroyRetry:
    def test_destroy_remains_retryable_after_removal_failure(self, tmp_path):
        """A failed ``rm_container()`` must not mark the backend destroyed, so
        the caller can retry; once it succeeds destroy() is idempotent. (The
        container family's twin is in test_container_backends.py.)"""
        backend = MagicMock()
        backend._destroyed = False
        backend._watched_pids = {}
        backend._checkpoint_image = None
        backend._accumulator_dir = None
        backend._root = tmp_path / "sandbox"
        backend.rm_container.side_effect = [RuntimeError("removal failed"), None]
        with pytest.raises(RuntimeError, match="removal failed"):
            _ChrootBackend.destroy(backend)
        assert not backend._destroyed
        _ChrootBackend.destroy(backend)
        _ChrootBackend.destroy(backend)
        assert backend._destroyed
        assert backend.rm_container.call_count == 2


class TestChrootStopAndRemove:
    def test_stop_forgets_tracked_process_groups_but_keeps_the_workspace(self):
        sb = _make_backend()
        sb._workspace.mkdir(parents=True)
        (sb._workspace / "keep.txt").write_text("x")
        sb._invocation_pgids = {4242}
        sb._watched_pids = {7: 1.0}
        with patch("os.killpg"):
            sb.stop()
        assert sb._invocation_pgids == set() and sb._watched_pids == {}
        assert (sb._workspace / "keep.txt").read_text() == "x"

    def test_rm_container_discards_the_workspace_but_not_committed_snapshots(self):
        sb = _make_backend()
        sb._workspace.mkdir(parents=True)
        (sb._workspace / "f.txt").write_text("committed")
        assert sb.commit("agency/keep") is True
        (sb._workspace / "f.txt").write_text("uncommitted")
        sb.rm_container()
        assert not sb._workspace.exists()
        assert (_CHROOT_SNAPSHOTS_DIR / _sanitize_tag("agency/keep") / "f.txt").read_text() == (
            "committed"
        )

    def test_stop_kills_every_tracked_process_group(self):
        sb = _make_backend()
        sb._invocation_pgids = {11, 22}
        killed = []
        with patch("os.killpg", side_effect=lambda pgid, sig: killed.append(pgid)):
            sb.stop()
        assert sorted(killed) == [11, 22]


class TestChrootSnapshots:
    """Checkpoints are plain directory copies under ``_CHROOT_SNAPSHOTS_DIR``
    (``cp -a --reflink=auto``), not image layers. Needs no jail, only a
    filesystem."""

    @staticmethod
    def _started(**kwargs):
        sb = _make_backend(**kwargs)
        sb._workspace.mkdir(parents=True, exist_ok=True)
        return sb

    @staticmethod
    def _template():
        from agency.sandbox.chroot_template import chroot_template_dir

        template = chroot_template_dir()
        if template is None:
            pytest.skip("chroot filesystem template not available on this host")
        return template

    @staticmethod
    def _snapshot(tag):
        return _CHROOT_SNAPSHOTS_DIR / _sanitize_tag(tag)

    def test_commit_before_the_workspace_exists_returns_false(self):
        assert _make_backend().commit("agency/nothing") is False
        assert not self._snapshot("agency/nothing").exists()

    def test_commit_copies_the_workspace_into_a_sanitized_snapshot_directory(self):
        sb = self._started()
        (sb._workspace / "sub").mkdir()
        (sb._workspace / "sub" / "f.txt").write_text("payload")
        assert sb.commit("agency/lifecycle-x:v1") is True
        snap = self._snapshot("agency/lifecycle-x:v1")
        assert (
            snap.parent == _CHROOT_SNAPSHOTS_DIR and "/" not in snap.name and ":" not in snap.name
        )
        assert (snap / "sub" / "f.txt").read_text() == "payload"
        assert sb._checkpoint_image == "agency/lifecycle-x:v1"

    def test_default_tag_is_the_lifecycle_tag(self):
        sb = self._started()
        assert sb.commit() is True
        assert self._snapshot(sb._lifecycle_tag()).is_dir()

    def test_snapshot_is_a_real_copy_not_a_hardlink_clone(self):
        """An in-place write after the commit must not change the snapshot."""
        sb = self._started()
        f = sb._workspace / "f.txt"
        f.write_text("before")
        sb.commit("agency/copy")
        with open(f, "r+") as handle:
            handle.write("AFTER!")
        assert (self._snapshot("agency/copy") / "f.txt").read_text() == "before"

    def test_recommit_replaces_the_snapshot_and_leaves_no_temp_directories(self):
        sb = self._started()
        (sb._workspace / "f.txt").write_text("one")
        sb.commit("agency/twice")
        (sb._workspace / "f.txt").write_text("two")
        (sb._workspace / "new.txt").write_text("n")
        sb.commit("agency/twice")
        snap = self._snapshot("agency/twice")
        assert (snap / "f.txt").read_text() == "two" and (snap / "new.txt").exists()
        assert [p.name for p in _CHROOT_SNAPSHOTS_DIR.iterdir()] == [snap.name]

    def test_a_new_backend_materializes_its_workspace_from_the_checkpoint(self):
        self._template()
        first = self._started()
        (first._workspace / "f.txt").write_text("carried")
        first.commit("agency/seed")
        second = _make_backend(checkpoint_image="agency/seed")
        second._ensure_started()
        assert (second._workspace / "f.txt").read_text() == "carried"
        (second._workspace / "f.txt").write_text("diverged")
        assert (self._snapshot("agency/seed") / "f.txt").read_text() == "carried"

    def test_an_unknown_checkpoint_tag_gives_an_empty_workspace(self):
        self._template()
        sb = _make_backend(checkpoint_image="agency/never-committed")
        sb._ensure_started()
        assert sb._workspace.is_dir() and list(sb._workspace.iterdir()) == []

    def test_an_existing_workspace_is_never_overwritten_by_ensure_started(self):
        sb = _make_backend(checkpoint_image="agency/whatever")
        sb._workspace.mkdir(parents=True)
        (sb._workspace / "live.txt").write_text("live")
        sb._ensure_started()
        assert (sb._workspace / "live.txt").read_text() == "live"

    def test_restore_replaces_the_workspace_and_stops_tracked_processes(self):
        self._template()
        sb = self._started()
        (sb._workspace / "f.txt").write_text("snap")
        sb.commit("agency/r")
        (sb._workspace / "f.txt").write_text("later")
        (sb._workspace / "extra.txt").write_text("later")
        sb._invocation_pgids = {4242}
        killed = []
        with patch("os.killpg", side_effect=lambda pgid, sig: killed.append(pgid)):
            sb.restore("agency/r")
        assert killed == [4242] and sb._invocation_pgids == set()
        assert (sb._workspace / "f.txt").read_text() == "snap"
        assert not (sb._workspace / "extra.txt").exists()
        assert sb._checkpoint_image == "agency/r"

    def test_destroy_removes_the_backends_own_checkpoint_snapshot(self):
        sb = self._started()
        sb.commit("agency/mine")
        sb.destroy()
        assert not self._snapshot("agency/mine").exists()
        assert not sb._root.exists()


class TestChrootSnapshotHelpers:
    """The static image helpers agent.save()/load()/fork() call generically."""

    def test_tag_image_copies_the_snapshot_and_overwrites_the_destination(self):
        src = _CHROOT_SNAPSHOTS_DIR / _sanitize_tag("agency/src")
        dst = _CHROOT_SNAPSHOTS_DIR / _sanitize_tag("agency/dst")
        src.mkdir(parents=True)
        (src / "a.txt").write_text("a")
        dst.mkdir(parents=True)
        (dst / "stale.txt").write_text("stale")
        _ChrootBackend.tag_image("agency/src", "agency/dst")
        assert (dst / "a.txt").read_text() == "a" and not (dst / "stale.txt").exists()
        assert (src / "a.txt").exists(), "tagging copies, it does not move"

    def test_tag_image_of_a_missing_snapshot_is_a_noop(self):
        _ChrootBackend.tag_image("agency/nope", "agency/dst")
        assert not (_CHROOT_SNAPSHOTS_DIR / _sanitize_tag("agency/dst")).exists()

    def test_delete_image_removes_the_snapshot_and_tolerates_a_missing_one(self):
        snap = _CHROOT_SNAPSHOTS_DIR / _sanitize_tag("agency/gone")
        snap.mkdir(parents=True)
        _ChrootBackend.delete_image("agency/gone", force=True)
        assert not snap.exists()
        _ChrootBackend.delete_image("agency/gone")

    def test_export_then_import_restores_the_same_tag(self):
        snap = _CHROOT_SNAPSHOTS_DIR / _sanitize_tag("agency/ckpt:1")
        (snap / "dir").mkdir(parents=True)
        (snap / "dir" / "f.txt").write_text("exported")
        blob = _ChrootBackend.export_image("agency/ckpt:1", 30)
        shutil.rmtree(snap)
        _ChrootBackend.import_image(blob, 30)
        assert (snap / "dir" / "f.txt").read_text() == "exported"

    def test_export_of_a_missing_tag_raises(self):
        with pytest.raises(FileNotFoundError):
            _ChrootBackend.export_image(f"agency/no-such-{uuid.uuid4().hex[:8]}", 30)

    def test_relabel_owner_pid_is_a_noop(self):
        """A chroot snapshot has no label/metadata concept; the method exists
        for API parity so agent.save()/load() can call it for any backend."""
        _ChrootBackend.relabel_owner_pid(f"agency/no-such-{uuid.uuid4().hex[:8]}", 12345, 30)
        _ChrootBackend.relabel_owner_pid(f"agency/no-such-{uuid.uuid4().hex[:8]}", None, 30)

    def test_the_image_kind_is_chroot(self):
        assert _ChrootBackend.IMAGE_KIND == "chroot"


# ===========================================================================
# 2. CHROOT-ONLY MECHANISMS
#    unshare/rootlesskit probing, the jail setup script, process-group
#    tracking, GPU device-node scoping.
# ===========================================================================


class TestSanitizeTag:
    def test_replaces_slash(self):
        assert "/" not in _sanitize_tag("agency/lifecycle-foo")

    def test_replaces_colon(self):
        assert ":" not in _sanitize_tag("agency/lifecycle-foo:v1")

    def test_distinct_tags_stay_distinct(self):
        assert _sanitize_tag("a/b") != _sanitize_tag("a-b")


class TestChrootDoesNotAdoptUnwatchedLivePids:
    def test_flag_disabled(self):
        assert _make_backend()._adopt_unwatched_live_pids is False

    def test_get_live_pids_excludes_unrelated_process_groups(self, monkeypatch):
        """A live process outside every tracked _invocation_pgids must never
        appear in get_live_pids() -- PGID matching, not host-wide presence,
        decides membership here."""
        sb = _make_backend()
        sb._invocation_pgids = {4242}
        # Fake /proc table: tracked-group member 111 alive, unrelated stranger
        # 999999 (a different, untracked pgid) also alive.
        monkeypatch.setattr(
            sb,
            "_read_proc_table",
            lambda script, timeout: ("111 1 4242 S\n999999 1 555 S\n", 0),
        )
        live = sb.get_live_pids()
        assert live == {111}


# ---------------------------------------------------------------------------
# _has_pending_background_work / _live_pgid_matched_pids /
# _kill_all_sandbox_processes -- this backend tracks background work purely
# via process groups (see chroot.py's module docstring). Each exec() call's
# underlying unshare invocation is its own new process-group leader (see
# _run_unshared()), and that pgid is recorded into _invocation_pgids. A
# process-group match is a real, kernel-tracked relationship (not a timing
# guess the way an earlier "non-baseline since jail start" scan was) --
# immune to host churn, and confirmed empirically to survive a child
# outliving its exited parent's reparenting, as well as to catch a child
# spawned well after the exec() call that backgrounded its parent already
# returned (something an earlier before/after-diff mechanism, since
# removed, could never see). It's also verified enough to act on
# destructively via os.killpg(), unlike a baseline scan, which could never
# confirm a candidate pid's ownership well enough to justify SIGKILLing it.
#
# (GPU release itself doesn't need any of this: release_gpu() no longer
# waits/polls on anything at all -- see agresources.release_gpu()'s
# docstring for why the caller's own kill+teardown sequence, which already
# ran before release_gpu() is even called, is sufficient confirmation.)
#
# The accepted trade-off: a process that calls setsid()/setpgid() to detach
# into its own new group (nohup/setsid/disown and similar daemonizing
# idioms) escapes this tracking entirely -- confirmed empirically (see
# TestChrootDelayedChildSafety.test_setsid_detached_daemon_is_not_tracked
# below) and accepted deliberately: the alternative (a baseline scan) traded
# that same blind spot for a *worse* one, making
# _has_pending_background_work() never resolve to "done" quickly on any host
# with background churn -- confirmed empirically too.
# ---------------------------------------------------------------------------


class TestChrootInvocationPgidTracking:
    def _sb(self, **kwargs):
        sb = _make_backend()
        sb._daemon_pids = set()
        sb._watched_pids = {}
        sb._invocation_pgids = {4242}
        for k, v in kwargs.items():
            setattr(sb, k, v)
        return sb

    def test_no_tracked_groups_no_pending_work_without_scanning(self, monkeypatch):
        """A fast path: with nothing in _invocation_pgids, don't even bother
        running a /proc scan."""
        sb = self._sb(_invocation_pgids=set())

        def _fail(*a, **kw):
            raise AssertionError("must not scan /proc when there are no tracked groups")

        monkeypatch.setattr(sb, "_read_proc_table", _fail)
        assert sb._has_pending_background_work() is False

    def test_matching_pgid_flags_pending_work(self, monkeypatch):
        """The exact regression: a pid that was NEVER in _watched_pids (a
        delayed child of an already-exited, already-untracked parent) must
        still be found via its inherited pgid and flag pending work."""
        sb = self._sb()
        monkeypatch.setattr(sb, "_read_proc_table", lambda script, timeout: ("999 1 4242 S\n", 0))
        assert sb._has_pending_background_work() is True
        assert sb._invocation_pgids == {4242}, "still-alive group must not be pruned"

    def test_no_matching_pgid_prunes_group(self, monkeypatch):
        """Once a scan finds nothing alive in a tracked group, that group is
        confirmed permanently gone (a pgid ceases to exist once its last
        member exits) and safe to forget."""
        sb = self._sb()
        monkeypatch.setattr(
            sb,
            "_read_proc_table",
            lambda script, timeout: ("999 1 555 S\n", 0),  # unrelated pgid
        )
        assert sb._has_pending_background_work() is False
        assert sb._invocation_pgids == set()

    def test_zombie_in_matching_group_excluded(self, monkeypatch):
        sb = self._sb()
        monkeypatch.setattr(sb, "_read_proc_table", lambda script, timeout: ("999 1 4242 Z\n", 0))
        assert sb._has_pending_background_work() is False

    def test_daemon_descendant_in_matching_group_excluded(self, monkeypatch):
        """A child of an explicitly-released daemon must not count as
        pending work, matching get_live_pids()'s own daemon-propagation --
        even though it shares a tracked invocation's pgid."""
        sb = self._sb(_daemon_pids={500})
        monkeypatch.setattr(
            sb, "_read_proc_table", lambda script, timeout: ("500 1 4242 S\n999 500 4242 S\n", 0)
        )
        assert sb._has_pending_background_work() is False

    def test_kill_all_sandbox_processes_kills_process_groups(self):
        sb = self._sb()
        killed_pgids = []
        with patch("os.killpg", side_effect=lambda pgid, sig: killed_pgids.append(pgid)):
            sb._kill_all_sandbox_processes()
        assert killed_pgids == [4242]

    def test_kill_all_sandbox_processes_ignores_already_dead_groups(self):
        sb = self._sb()

        def _raise_killpg(pgid, sig):
            raise ProcessLookupError()

        with patch("os.killpg", side_effect=_raise_killpg):
            sb._kill_all_sandbox_processes()  # must not raise


# ---------------------------------------------------------------------------
# Live background-process tracking -- the chroot analogue of
# the container family's PID tracking in test_container_backends.py. Regression coverage
# for two real, previously-silent bugs, both found and fixed against a real
# jail rather than assumed from reading the code:
#
# 1. Before the /proc-visibility fix in sandbox/base.py + chroot.py,
#    mounting (fresh OR bind) a procfs inside the jail silently failed (kernel
#    EINVAL -- see chroot.py's module docstring), so `/proc` inside every jail
#    was always empty and _watched_pids/get_live_pids() never tracked a
#    single background process, no matter what a command spawned.
#
# 2. _exec_with_pid_tracking()'s chroot sub-invocation used to capture output
#    via `$(...)` command substitution -- a real OS pipe, whose EOF (and so
#    $(...)'s own return) blocks until EVERY process holding the write end
#    closes it, including a backgrounded descendant of the chrooted command.
#    That silently made exec() block for the ENTIRE runtime of any `cmd &`
#    before ever reaching its own after-diff -- exactly the case this
#    mechanism exists to detect without waiting on. Confirmed empirically
#    (a `sleep 3 &` made exec() itself take ~3.8s) and fixed by capturing
#    output via a temp file instead, whose reads don't block on other
#    processes' open write handles the way a pipe's do.
#
# Unlike docker/podman, chroot has no isolated PID namespace of its own, so
# get_live_pids() here is PGID-matched rather than baseline-diffed (see
# chroot.py's module docstring): every PID sharing a tracked invocation's
# process group counts, including one spawned well after the exec() call
# that started that invocation already returned. get_live_pids() and
# _has_pending_background_work() are backed by the exact same scan, so they
# can never disagree with each other the way an earlier, since-removed
# before/after-diff `get_live_pids()` and PGID-based
# `_has_pending_background_work()` sometimes did.
# ---------------------------------------------------------------------------


@chroot
class TestChrootDelayedChildSafety:
    """End-to-end reproduction, against a real jail, of the bug PGID-based
    tracking was introduced to fix: a child spawned by an already-tracked
    process AFTER that process's own exec() call already returned used to
    be invisible to the old before/after-diff `_watched_pids` mechanism,
    but not to `_has_pending_background_work()`'s pgid scan -- a real gap
    since closed by making get_live_pids() itself PGID-based too (see
    chroot.py's module docstring), so both now agree the child is alive."""

    def test_setsid_detached_daemon_is_not_tracked(self):
        """The accepted, deliberate blind spot (see chroot.py's module
        docstring and TestChrootInvocationPgidTracking's class comment): a
        process that detaches into its own new process group via setsid
        (the same mechanism `nohup`/`disown`/many "daemonize" library
        patterns use) is invisible to _invocation_pgids, since it no longer
        shares its invocation's pgid. Pinned here explicitly so this is a
        visible, intentional decision rather than an implicit gap -- confirm
        it stays this way rather than silently regressing to "worse" (a
        crash) or silently improving to "better" (which would need a design
        change to rely on, not just an incidental fix)."""
        sb = _make_backend()
        try:
            sb.exec("setsid sleep 20 < /dev/null > /dev/null 2>&1 &")
            time.sleep(0.5)
            assert sb._has_pending_background_work() is False, (
                "documenting the accepted gap: a setsid-detached process is "
                "not caught by pgid-based tracking, unlike a plain background job"
            )
        finally:
            sb.destroy()


# ---------------------------------------------------------------------------
# _chroot_gpu_dev_paths(gpu_id) -- scopes the jail's GPU device bind-mounts to
# the single leased gpu_id (mirrors container.py's _gpu_flags(runtime,
# gpu_id) after this session's fix there) so a sandboxed process cannot open
# another leased jail's GPU device node directly by guessing its path. No
# real chroot/GPU required -- _all_chroot_gpu_dev_paths() (the raw host
# detection step) is mocked, isolating the scoping logic tested here from
# nvidia-smi/rocm-smi/real /dev contents.
# ---------------------------------------------------------------------------


class TestChrootGpuDevPaths:
    def test_gpu_ids_empty_returns_empty_even_with_gpus_on_host(self):
        with patch.object(
            _chroot_mod,
            "_all_chroot_gpu_dev_paths",
            return_value=["/dev/nvidiactl", "/dev/nvidia0"],
        ):
            assert _chroot_gpu_dev_paths([]) == []

    def test_no_gpu_devices_on_host_returns_empty(self):
        with patch.object(_chroot_mod, "_all_chroot_gpu_dev_paths", return_value=[]):
            assert _chroot_gpu_dev_paths([0]) == []

    def test_nvidia_control_devices_plus_single_indexed_device(self):
        all_paths = [
            "/dev/nvidiactl",
            "/dev/nvidia-uvm",
            "/dev/nvidia-uvm-tools",
            "/dev/nvidia0",
            "/dev/nvidia1",
            "/dev/nvidia2",
        ]
        with patch.object(_chroot_mod, "_all_chroot_gpu_dev_paths", return_value=all_paths):
            with patch.object(_chroot_mod.shutil, "which", return_value="/usr/bin/nvidia-smi"):
                paths = _chroot_gpu_dev_paths([1])
        assert "/dev/nvidia1" in paths
        assert "/dev/nvidia0" not in paths
        assert "/dev/nvidia2" not in paths
        for control in ("/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools"):
            assert control in paths

    def test_nvidia_multiple_indexed_devices(self):
        all_paths = ["/dev/nvidiactl", "/dev/nvidia0", "/dev/nvidia1", "/dev/nvidia2"]
        with patch.object(_chroot_mod, "_all_chroot_gpu_dev_paths", return_value=all_paths):
            with patch.object(_chroot_mod.shutil, "which", return_value="/usr/bin/nvidia-smi"):
                paths = _chroot_gpu_dev_paths([0, 2])
        assert paths == ["/dev/nvidiactl", "/dev/nvidia0", "/dev/nvidia2"]

    def test_nvidia_gpu_id_out_of_range_returns_only_control_devices(self):
        all_paths = ["/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia0"]
        with patch.object(_chroot_mod, "_all_chroot_gpu_dev_paths", return_value=all_paths):
            with patch.object(_chroot_mod.shutil, "which", return_value="/usr/bin/nvidia-smi"):
                paths = _chroot_gpu_dev_paths([5])
        assert paths == ["/dev/nvidiactl", "/dev/nvidia-uvm"]

    def test_amd_control_device_plus_single_render_node(self):
        """amd_render_node_paths_by_pci_bus() is explicitly forced to None
        here (no rocm-smi/real sysfs in this test), which exercises the
        naive-sorted-order FALLBACK path specifically -- see
        test_amd_uses_pci_bus_ordering_when_available below for the mapping
        actually being applied, and TestChrootAmdRenderNodeLiveHardware for
        the real-hardware check that found sorted order alone is wrong on a
        multi-GPU AMD host."""
        all_paths = ["/dev/kfd", "/dev/dri/card0", "/dev/dri/renderD128", "/dev/dri/renderD129"]
        with patch.object(_chroot_mod, "_all_chroot_gpu_dev_paths", return_value=all_paths):
            with patch.object(_chroot_mod.shutil, "which", return_value=None):
                with patch.object(
                    _chroot_mod, "amd_render_node_paths_by_pci_bus", return_value=None
                ):
                    paths = _chroot_gpu_dev_paths([1])
        assert paths == ["/dev/kfd", "/dev/dri/renderD129"]

    def test_amd_gpu_id_out_of_range_returns_only_control_device(self):
        all_paths = ["/dev/kfd", "/dev/dri/renderD128"]
        with patch.object(_chroot_mod, "_all_chroot_gpu_dev_paths", return_value=all_paths):
            with patch.object(_chroot_mod.shutil, "which", return_value=None):
                with patch.object(
                    _chroot_mod, "amd_render_node_paths_by_pci_bus", return_value=None
                ):
                    paths = _chroot_gpu_dev_paths([5])
        assert paths == ["/dev/kfd"]

    def test_amd_uses_pci_bus_ordering_when_available(self):
        """When amd_render_node_paths_by_pci_bus() successfully builds a
        mapping, _chroot_gpu_dev_paths() must use ITS ordering, not naive
        sorted /dev/dri order -- this is the actual fix: on real 8x MI350X
        hardware the two disagree for every GPU (see agresources.py's
        amd_render_node_paths_by_pci_bus docstring)."""
        all_paths = ["/dev/kfd", "/dev/dri/renderD128", "/dev/dri/renderD129"]
        pci_ordered = ["/dev/dri/renderD129", "/dev/dri/renderD128"]  # reversed
        with patch.object(_chroot_mod, "_all_chroot_gpu_dev_paths", return_value=all_paths):
            with patch.object(_chroot_mod.shutil, "which", return_value=None):
                with patch.object(
                    _chroot_mod, "amd_render_node_paths_by_pci_bus", return_value=pci_ordered
                ):
                    paths_0 = _chroot_gpu_dev_paths([0])
                    paths_1 = _chroot_gpu_dev_paths([1])
        assert paths_0 == ["/dev/kfd", "/dev/dri/renderD129"]
        assert paths_1 == ["/dev/kfd", "/dev/dri/renderD128"]

    def test_amd_multiple_render_nodes_via_pci_bus_ordering(self):
        all_paths = ["/dev/kfd", "/dev/dri/renderD128", "/dev/dri/renderD129"]
        pci_ordered = ["/dev/dri/renderD129", "/dev/dri/renderD128"]  # reversed
        with patch.object(_chroot_mod, "_all_chroot_gpu_dev_paths", return_value=all_paths):
            with patch.object(_chroot_mod.shutil, "which", return_value=None):
                with patch.object(
                    _chroot_mod, "amd_render_node_paths_by_pci_bus", return_value=pci_ordered
                ):
                    paths = _chroot_gpu_dev_paths([0, 1])
        assert paths == ["/dev/kfd", "/dev/dri/renderD129", "/dev/dri/renderD128"]


class TestChrootSetupLinesGpuScoping:
    """Integration check that _setup_lines() actually wires
    _chroot_gpu_dev_paths(self._real_gpu_ids) into the jail's /dev bind mounts,
    rather than every host GPU device -- no real chroot required, this only
    inspects the generated shell lines."""

    def test_only_the_scoped_gpu_device_is_bind_mounted(self, tmp_path, monkeypatch):
        scoped_dev = tmp_path / "nvidia1"
        scoped_dev.touch()
        sb = _make_backend()
        sb._real_gpu_ids = [1]
        monkeypatch.setattr(
            _chroot_mod,
            "_chroot_gpu_dev_paths",
            lambda gpu_ids: [str(scoped_dev)] if gpu_ids == [1] else [],
        )
        joined = "\n".join(sb._setup_lines())
        assert str(scoped_dev) in joined
        assert "nvidia0" not in joined

    def test_multiple_scoped_gpu_devices_are_bind_mounted(self, tmp_path, monkeypatch):
        dev1 = tmp_path / "nvidia1"
        dev1.touch()
        dev3 = tmp_path / "nvidia3"
        dev3.touch()
        sb = _make_backend()
        sb._real_gpu_ids = [1, 3]
        monkeypatch.setattr(
            _chroot_mod,
            "_chroot_gpu_dev_paths",
            lambda gpu_ids: [str(dev1), str(dev3)] if gpu_ids == [1, 3] else [],
        )
        joined = "\n".join(sb._setup_lines())
        assert str(dev1) in joined
        assert str(dev3) in joined

    def test_no_gpu_leased_bind_mounts_no_gpu_devices(self, monkeypatch):
        sb = _make_backend()
        sb._real_gpu_ids = []
        monkeypatch.setattr(_chroot_mod, "_chroot_gpu_dev_paths", lambda gpu_ids: [])
        joined = "\n".join(sb._setup_lines())
        assert "nvidia" not in joined
        assert "renderD" not in joined
        assert "kfd" not in joined


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


class TestChrootAmdRenderNodeLiveHardware:
    """Runs the real (unmocked) AMD GPU-to-render-node scoping against
    actual ROCm hardware -- no mocks anywhere. Skipped automatically without
    a real rocm-smi + AMD GPU(s). Mirrors
    test_container_backends.py's TestAmdRenderNodeLiveHardware.

    This is the check that actually caught the bug being regression-tested
    here: on an 8x MI350X host, naive sorted /dev/dri/renderD* order did
    NOT correspond to rocm-smi's GPU index for ANY of the 8 GPUs (each GPU
    there exposes itself plus 7 XCD/compute-partition sibling render nodes
    -- 64 nodes total -- and even the primary node's number doesn't sort in
    GPU-index order). This test independently recomputes the "ground truth"
    GPU-to-render-node mapping from rocm-smi --showbus and
    /sys/class/drm/*/device -- without going through
    amd_render_node_paths_by_pci_bus itself -- and asserts the real
    _chroot_gpu_dev_paths() output agrees with it for every real GPU on
    this host.
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
    def test_chroot_gpu_dev_paths_match_gpu_pci_bus_on_real_hardware(self):
        _chroot_mod._gpu_dev_paths_cache = None
        bus_by_gpu_id = self._ground_truth_bus_by_gpu_id()
        assert bus_by_gpu_id, "expected at least one real AMD GPU"

        seen_devices = set()
        for gpu_id, expected_bus in bus_by_gpu_id.items():
            paths = _chroot_gpu_dev_paths(gpu_id)
            assert paths and os.path.basename(paths[0]) == "kfd"
            assert len(paths) == 2, f"expected a scoped render node for gpu_id={gpu_id}: {paths}"
            render_path = paths[1]
            actual_bus = self._real_render_node_bus(render_path)
            assert actual_bus == expected_bus, (
                f"gpu_id={gpu_id}: _chroot_gpu_dev_paths() picked {render_path} "
                f"(bus={actual_bus}), but rocm-smi says GPU {gpu_id} is actually "
                f"on bus {expected_bus}"
            )
            assert render_path not in seen_devices, (
                f"gpu_id={gpu_id} was scoped to {render_path}, already used by another gpu_id"
            )
            seen_devices.add(render_path)


# ---------------------------------------------------------------------------
# Which namespace-creating prefix works on this host: plain ``unshare``, else
# ``rootlesskit --net=none unshare`` (distros whose AppArmor denies
# CLONE_NEWUSER to unconfined binaries but exempts rootlesskit's profile).
# ---------------------------------------------------------------------------

_ROOTLESSKIT = ["rootlesskit", "--net=none", "unshare"]


@pytest.fixture
def fresh_probe(monkeypatch):
    """Empty probe caches and no userns sysctl, so the probe really runs."""
    monkeypatch.setattr(_chroot_mod, "_chroot_unshare_prefix_cache", None)
    monkeypatch.setattr(_chroot_mod, "_chroot_available_cache", None)

    class _NoSysctl:
        def __init__(self, *_a):
            pass

        def exists(self):
            return False

    monkeypatch.setattr(_chroot_mod, "Path", _NoSysctl)


def _which(*present):
    return lambda name: f"/usr/bin/{name}" if name in present else None


class TestUnsharePrefixCandidates:
    def test_only_plain_unshare_without_rootlesskit(self):
        with patch.object(_chroot_mod.shutil, "which", _which("unshare", "chroot")):
            assert _chroot_mod._unshare_prefix_candidates() == [["unshare"]]

    def test_rootlesskit_is_the_second_candidate_when_installed(self):
        with patch.object(_chroot_mod.shutil, "which", _which("unshare", "rootlesskit")):
            assert _chroot_mod._unshare_prefix_candidates() == [["unshare"], _ROOTLESSKIT]


@pytest.mark.usefixtures("fresh_probe")
class TestChrootAvailabilityProbe:
    @staticmethod
    def _probe(which, run):
        with patch.object(_chroot_mod.shutil, "which", which):
            with patch.object(_chroot_mod.subprocess, "run", run):
                return _chroot_mod._probe_chroot_available()

    def test_missing_unshare_or_chroot_binary_means_unavailable(self):
        run = MagicMock()
        assert self._probe(_which("chroot"), run) is False
        assert self._probe(_which("unshare"), run) is False
        run.assert_not_called()

    def test_sysctl_disabling_userns_means_unavailable_without_probing(self, monkeypatch):
        class _Sysctl:
            def __init__(self, *_a):
                pass

            def exists(self):
                return True

            def read_text(self):
                return "0\n"

        monkeypatch.setattr(_chroot_mod, "Path", _Sysctl)
        run = MagicMock()
        assert self._probe(_which("unshare", "chroot"), run) is False
        run.assert_not_called()

    def test_unreadable_sysctl_falls_through_to_the_live_probe(self, monkeypatch):
        class _Broken:
            def __init__(self, *_a):
                pass

            def exists(self):
                raise OSError("no /proc")

        monkeypatch.setattr(_chroot_mod, "Path", _Broken)
        run = MagicMock(return_value=subprocess.CompletedProcess([], 0))
        assert self._probe(_which("unshare", "chroot"), run) is True

    def test_plain_unshare_is_preferred_and_probed_with_the_real_flags(self):
        run = MagicMock(return_value=subprocess.CompletedProcess([], 0))
        assert self._probe(_which("unshare", "chroot", "rootlesskit"), run) is True
        assert _chroot_mod._chroot_unshare_prefix_cache == ["unshare"]
        assert [c.args[0] for c in run.call_args_list] == [
            ["unshare", "--user", "--map-root-user", "--mount", "true"]
        ]

    def test_rootlesskit_is_used_when_plain_unshare_is_denied(self):
        def run(argv, **_kw):
            return subprocess.CompletedProcess(argv, 1 if argv[0] == "unshare" else 0)

        run = MagicMock(side_effect=run)
        assert self._probe(_which("unshare", "chroot", "rootlesskit"), run) is True
        assert _chroot_mod._chroot_unshare_prefix_cache == _ROOTLESSKIT
        assert [c.args[0][0] for c in run.call_args_list] == ["unshare", "rootlesskit"]
        assert run.call_args_list[1].args[0][:3] == _ROOTLESSKIT

    def test_a_candidate_that_cannot_even_launch_does_not_abort_detection(self, capsys):
        def run(argv, **_kw):
            if argv[0] == "unshare":
                raise OSError("exec failed")
            return subprocess.CompletedProcess(argv, 0)

        assert self._probe(_which("unshare", "chroot", "rootlesskit"), run) is True
        assert _chroot_mod._chroot_unshare_prefix_cache == _ROOTLESSKIT
        assert "capability probe" in capsys.readouterr().out

    def test_every_candidate_failing_means_unavailable_and_no_cached_prefix(self):
        run = MagicMock(return_value=subprocess.CompletedProcess([], 1))
        assert self._probe(_which("unshare", "chroot", "rootlesskit"), run) is False
        assert _chroot_mod._chroot_unshare_prefix_cache is None
        assert run.call_count == 2


class TestChrootUnsharePrefix:
    def test_the_probed_prefix_is_returned(self, monkeypatch):
        monkeypatch.setattr(_chroot_mod, "_chroot_available_cache", True)
        monkeypatch.setattr(_chroot_mod, "_chroot_unshare_prefix_cache", list(_ROOTLESSKIT))
        assert _chroot_mod._chroot_unshare_prefix() == _ROOTLESSKIT

    def test_plain_unshare_is_the_default_when_nothing_was_probed(self, monkeypatch):
        monkeypatch.setattr(_chroot_mod, "_chroot_available_cache", False)
        monkeypatch.setattr(_chroot_mod, "_chroot_unshare_prefix_cache", None)
        assert _chroot_mod._chroot_unshare_prefix() == ["unshare"]


class _FakeProc:
    def __init__(self, pid=4321, rc=0, out=b"out", err=b"", hang=False):
        self.pid, self.returncode, self._out, self._err, self._hang = pid, rc, out, err, hang
        self.killed = False

    def communicate(self, input=None, timeout=None):
        if self._hang and not self.killed:
            raise subprocess.TimeoutExpired("x", timeout)
        return self._out, self._err

    def kill(self):
        self.killed = True


class TestRunUnshared:
    """The lowest-level launch: argv shape, process-group leadership, timeout."""

    @staticmethod
    def _run(proc, prefix=None, **kwargs):
        sb = _make_backend()
        seen = {}

        def popen(args, **kw):
            seen["args"], seen["kw"] = args, kw
            return proc

        with patch.object(
            _chroot_mod, "_chroot_unshare_prefix", return_value=prefix or ["unshare"]
        ):
            with patch.object(_chroot_mod.subprocess, "Popen", popen):
                result = sb._run_unshared("echo hi", timeout=kwargs.pop("timeout", 5), **kwargs)
        return result, seen

    def test_argv_is_prefix_then_user_mount_namespace_flags_then_bash(self):
        (output, rc, pgid), seen = self._run(_FakeProc(out=b"o", err=b"e"), prefix=_ROOTLESSKIT)
        assert seen["args"] == [
            *_ROOTLESSKIT,
            "--user",
            "--map-root-user",
            "--mount",
            "--",
            "bash",
            "-c",
            "echo hi",
        ]
        assert (output, rc) == ("oe", 0)

    def test_the_launcher_leads_a_new_session_whose_pgid_is_its_pid(self):
        (_o, _rc, pgid), seen = self._run(_FakeProc(pid=777))
        assert seen["kw"]["start_new_session"] is True
        assert pgid == 777

    def test_stdin_is_piped_only_when_given(self):
        _r, seen = self._run(_FakeProc())
        assert seen["kw"]["stdin"] is None
        _r, seen = self._run(_FakeProc(), stdin=b"data")
        assert seen["kw"]["stdin"] == subprocess.PIPE

    def test_nonzero_exit_status_is_reported(self):
        (output, rc, _pgid), _seen = self._run(_FakeProc(rc=3, out=b"", err=b"boom"))
        assert (output, rc) == ("boom", 3)

    def test_timeout_kills_the_launcher_and_reports_minus_one_with_the_pgid(self):
        proc = _FakeProc(pid=99, hang=True)
        (output, rc, pgid), _seen = self._run(proc, timeout=2)
        assert proc.killed
        assert (output, rc, pgid) == ("Command timed out after 2s", -1, 99)

    def test_a_launch_failure_is_reported_not_raised(self):
        sb = _make_backend()
        with patch.object(_chroot_mod.subprocess, "Popen", side_effect=OSError("no unshare")):
            output, rc, pgid = sb._run_unshared("true", timeout=5)
        assert (output, rc, pgid) == ("no unshare", -1, None)


class TestExecWithPidTracking:
    def test_the_invocations_process_group_is_recorded_and_output_goes_through_a_file(self):
        sb = _make_backend()
        sb._ensure_started = lambda: None
        captured = {}

        def run_unshared(script, *, stdin=None, timeout):
            captured["script"] = script
            return "result", 0, 777

        sb._run_unshared = run_unshared
        assert sb._exec_with_pid_tracking("export A=1; ", "sleep 1 &", "/workspace", 5) == (
            "result",
            0,
        )
        assert sb._invocation_pgids == {777}
        script = captured["script"]
        assert script.startswith("{ mkdir -p ") and ">/dev/null 2>&1" in script.splitlines()[0]
        assert "mktemp" in script and "$(chroot" not in script
        assert f"chroot {sb._root} bash -c" in script and "export A=1; sleep 1 &" in script
        assert script.rstrip().endswith("exit $__AGENCY_RC")

    def test_an_invocation_without_a_pgid_records_nothing(self):
        sb = _make_backend()
        sb._ensure_started = lambda: None
        sb._run_unshared = lambda script, *, stdin=None, timeout: ("x", 1, None)
        sb._exec_with_pid_tracking("", "true", "/workspace", 5)
        assert sb._invocation_pgids == set()


# ---------------------------------------------------------------------------
# The jail setup script (_setup_lines / _build_jail_script). The jail owns a
# full filesystem copied from the shared template; setup only binds in
# device files and caller-configured mounts. Mounts use --rbind: a bare
# ``mount --bind`` of a directory failed with "wrong fs type" when the host
# root itself was an overlayfs (a sandbox started inside a container), while
# --rbind works there. Setup output and failures must never reach the caller
# or abort the command.
# ---------------------------------------------------------------------------


def _lines(sb=None):
    sb = sb or _make_backend()
    return sb, sb._setup_lines()


class TestJailSetupLines:
    def test_no_host_system_directory_is_mounted_into_the_jail(self):
        """The jail owns a full filesystem copied from the shared template;
        host system directories must never be mounted over it."""
        _sb, lines = _lines()
        system_mounts = [l for l in lines if re.match(r"mount --rbind /(?!dev/)", l)]
        assert system_mounts == []

    def test_a_bare_bind_is_never_used_for_directories(self):
        """The overlay-root regression: ``mount --bind`` on a directory."""
        _sb, lines = _lines()
        directory_binds = [l for l in lines if re.match(r"mount --bind /(?!dev/)", l)]
        assert directory_binds == []

    def test_the_read_only_remount_may_fail_without_failing_setup(self):
        _sb, lines = _lines()
        remounts = [l for l in lines if "remount" in l]
        user_ro_mounts = [l for l in lines if "remount,bind,ro" in l]
        assert all(l.endswith("2>/dev/null || true") for l in user_ro_mounts)
        assert (remounts == user_ro_mounts) or (remounts == [])

    def test_user_mounts_are_rbind_mounted(self, tmp_path):
        rw, ro = tmp_path / "rw", tmp_path / "ro"
        sb = _make_backend(mounts={"a": (str(rw), "/data", "rw"), "b": (str(ro), "/ro/x", "ro")})
        _sb, lines = _lines(sb)
        root = str(sb._root)
        i_rw = lines.index(f"mount --rbind {rw} {root}/data")
        i_ro = lines.index(f"mount --rbind {ro} {root}/ro/x")
        assert i_rw < i_ro
        assert f"mkdir -p {rw}" in lines[:i_rw] and f"mkdir -p {root}/ro/x" in lines[:i_ro]
        assert not any(l.startswith(f"mount -o remount,bind,ro {root}/data") for l in lines)
        assert lines[i_ro + 1] == f"mount -o remount,bind,ro {root}/ro/x 2>/dev/null || true"

    def test_dev_gets_per_file_binds_and_never_a_whole_directory_or_tmpfs(self):
        sb, lines = _lines()
        root, joined = str(sb._root), "\n".join(lines)
        for name in _chroot_mod._CHROOT_DEV_FILES:
            if os.path.exists(f"/dev/{name}"):
                assert f"touch {root}/dev/{name}" in lines
                assert f"mount --bind /dev/{name} {root}/dev/{name}" in lines
        assert "mount --rbind /dev " not in joined and "--bind /dev " not in joined
        assert "tmpfs" not in joined

    def test_dev_pts_is_a_private_devpts_instance_with_ptmx_symlink(self):
        sb, lines = _lines()
        root = str(sb._root)
        assert (
            f"mount -t devpts -o newinstance,ptmxmode=0666,mode=0620 devpts {root}/dev/pts" in lines
        )
        assert f"ln -sfn pts/ptmx {root}/dev/ptmx" in lines

    def test_dev_shm_is_private_and_sticky_world_writable(self):
        sb, lines = _lines()
        root = str(sb._root)
        assert f"mkdir -p {root}/dev/shm" in lines and f"chmod 1777 {root}/dev/shm" in lines

    def test_proc_is_left_as_an_empty_directory_and_never_mounted(self):
        sb, lines = _lines()
        joined = "\n".join(lines)
        assert f"mkdir -p {sb._root}/proc" in lines
        assert "-t proc" not in joined and "/proc " not in joined.replace(f"{sb._root}/proc", "")

    def test_workspace_and_tmp_exist_in_every_jail(self):
        sb, lines = _lines()
        assert f"mkdir -p {sb._root}/workspace" in lines and f"mkdir -p {sb._root}/tmp" in lines


class TestBuildJailScript:
    def test_setup_is_one_silenced_group_whose_failures_do_not_abort(self):
        sb = _make_backend()
        script = sb._build_jail_script("echo hi", workdir="/workspace", shell="bash")
        setup, exec_line = script.split("\n", 1)
        assert setup.startswith("{ ") and setup.endswith("; } >/dev/null 2>&1")
        assert "&&" not in setup, (
            "a failing mount must not stop the later setup lines or the command"
        )
        assert exec_line.startswith(f"exec chroot {sb._root} bash -c ")

    def test_the_command_runs_chrooted_from_its_workdir_with_the_requested_shell(self):
        sb = _make_backend()
        script = sb._build_jail_script("echo hi", workdir="/work dir", shell="sh")
        exec_line = script.split("\n", 1)[1]
        assert shlex.split(exec_line)[:4] == ["exec", "chroot", str(sb._root), "sh"]
        assert shlex.split(exec_line)[-1] == "cd '/work dir' 2>/dev/null; echo hi"

    @chroot
    def test_a_failing_setup_line_neither_breaks_nor_pollutes_the_command(self, monkeypatch):
        """Mount failures (the overlay-root case) are swallowed: the command
        still runs and its output has no mount diagnostics in it."""
        sb = _make_backend()
        real = sb._setup_lines
        broken = [
            "mount --rbind /definitely/not/a/dir /nowhere",
            "mount -t nosuchfs none /nowhere",
            "false",
        ]
        monkeypatch.setattr(sb, "_setup_lines", lambda: broken + real())
        try:
            output, rc = sb.exec("echo still-works")
            assert (rc, output.strip()) == (0, "still-works")
        finally:
            sb.destroy()


@chroot
class TestJailFilesystem:
    """What the setup script produces, observed from inside a real jail."""

    @pytest.fixture
    def sb(self):
        sb = _make_backend()
        yield sb
        sb.destroy()

    @pytest.mark.parametrize("base", _CHROOT_RO_BASE_DIRS)
    def test_system_directories_are_writable_own_copies_not_host_mounts(self, sb, base):
        """The jail owns its filesystem (a template copy), so system
        directories accept writes and stay isolated from the host: a file
        created inside the jail must not appear on the host."""
        if not os.path.isdir(f"/{base}"):
            pytest.skip(f"/{base} does not exist in the template")
        marker = f"agency-write-probe-{uuid.uuid4().hex[:8]}"
        output, rc = sb.exec(f"touch /{base}/{marker} && echo rc=0")
        assert rc == 0, f"write to /{base} failed inside the jail: {output}"
        assert "rc=0" in output, output
        assert not os.path.exists(f"/{base}/{marker}"), (
            f"file written inside the jail leaked to the host /{base}"
        )

    def test_an_existing_system_file_is_isolated_from_the_host(self, sb):
        output, _rc = sb.exec("echo jail-only-line >> /etc/passwd; echo rc=$?")
        assert "rc=0" in output, output
        assert "jail-only-line" not in Path("/etc/passwd").read_text()

    def test_the_templates_interpreters_are_visible_and_executable(self, sb):
        output, rc = sb.exec("test -x /usr/bin/bash && /usr/bin/bash -c 'echo visible'")
        assert (rc, output.strip()) == (0, "visible")
        output, rc = sb.exec("ls /usr/bin | wc -l")
        assert rc == 0 and int(output.strip()) > 10

    def test_the_template_provides_the_basic_tool_set(self, sb):
        # ps is installed but cannot run inside the jail (no procfs by
        # design, unlike container sandboxes); its presence is checked
        # instead of its exit.
        output, rc = sb.exec(
            "python3 --version >/dev/null && python --version >/dev/null && "
            "git --version >/dev/null && curl --version >/dev/null && "
            "test -x /usr/bin/ps && echo tools-ok"
        )
        assert (rc, output.strip()) == (0, "tools-ok"), output

    def test_the_template_provides_the_harness_python_packages(self, sb):
        output, rc = sb.exec(
            "python3 -c 'import fastapi, uvicorn, openai, httpx, mcp, "
            "pyseccomp, cloudpickle, pyte; print(\"packages-ok\")'"
        )
        assert (rc, output.strip()) == (0, "packages-ok"), output

    def test_dev_has_the_standard_device_files_and_they_open(self, sb):
        output, rc = sb.exec(
            "echo discarded > /dev/null && head -c 4 /dev/urandom | wc -c && "
            "head -c 4 /dev/zero | wc -c && test -c /dev/null && echo chardev"
        )
        assert rc == 0, output
        assert output.split() == ["4", "4", "chardev"]

    def test_dev_shm_is_private_to_the_jail(self, sb):
        name = f"agency-shm-{uuid.uuid4().hex[:8]}"
        _out, rc = sb.exec(f"touch /dev/shm/{name}")
        assert rc == 0
        assert not os.path.exists(f"/dev/shm/{name}")

    def test_proc_is_an_empty_directory_inside_the_jail(self, sb):
        output, rc = sb.exec("ls -A /proc | wc -l")
        assert (rc, output.strip()) == (0, "0")

    def test_inside_root_is_the_host_user_so_files_are_owned_by_the_host_user(self, sb):
        output, rc = sb.exec("id -u && echo data > /workspace/owned.txt")
        assert rc == 0 and output.strip() == "0"
        assert (sb._workspace / "owned.txt").stat().st_uid == os.getuid()

    def test_the_host_environment_is_inherited_by_jailed_commands(self, sb, monkeypatch):
        """Pinned (known isolation gap, see conftest's ``env_isolation``
        capability): the jail is a plain subprocess of the host process."""
        monkeypatch.setenv("AGENCY_CHROOT_TEST_HOST_VAR", "from-the-host")
        output, rc = sb.exec("printenv AGENCY_CHROOT_TEST_HOST_VAR")
        assert (rc, output.strip()) == (0, "from-the-host")


class TestChrootHasNoProcessCheckpointApi:
    def test_there_is_no_checkpoint_handle_api(self):
        """chroot snapshots only support commit()/restore(); the
        ``checkpoint()`` / ``delete_checkpoint()`` handle API is container-only
        (conftest capability ``checkpoint_handle``)."""
        sb = _make_backend()
        assert not hasattr(sb, "checkpoint") and not hasattr(sb, "delete_checkpoint")

    def test_update_limits_is_a_noop(self):
        assert _make_backend().update_limits(cpus=1.0, memory="1g") is None


class TestChrootGpuDeviceDetection:
    def test_detection_is_cached_for_the_process_lifetime(self, monkeypatch):
        monkeypatch.setattr(_chroot_mod, "_gpu_dev_paths_cache", None)
        detect = MagicMock(return_value=["/dev/nvidiactl"])
        monkeypatch.setattr(_chroot_mod, "_detect_chroot_gpu_dev_paths", detect)
        assert _chroot_mod._all_chroot_gpu_dev_paths() == ["/dev/nvidiactl"]
        assert _chroot_mod._all_chroot_gpu_dev_paths() == ["/dev/nvidiactl"]
        detect.assert_called_once()

    def test_no_detected_gpus_means_no_device_paths(self, monkeypatch):
        monkeypatch.setattr("agency.utils.agutil.detect_gpus", lambda: [])
        assert _chroot_mod._detect_chroot_gpu_dev_paths() == []
