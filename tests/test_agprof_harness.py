"""Cross-cutting interruption and thread-parentage guards.

The deterministic HTTP adapter/profiler and shared tool-contract golden tests
live in test_profiler_native.py. They exercise all five wire adapters with a
fake provider; real CLI binaries, Linux ptrace, cgroups and GPU devices still
require their environment-specific integration runs.
"""

from __future__ import annotations

import ast
import json
import threading
from collections import Counter
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_AGENCY_ROOT = _REPO_ROOT / "agency"


# A raw Thread is correct for plumbing that must not inherit a task span.  The
# two agprof.py calls are the reviewed implementation of spawn_traced itself
# (including its profiling-disabled fast path).  An identity is
# (repository-relative path, lexical class/function scope); counts distinguish
# multiple reviewed calls in one scope without relying on brittle line numbers.
_BARE_THREAD_ALLOWLIST = {
    # Harness service plumbing.
    (
        "agency/harness/daemon.py",
        "_HarnessApiServer.start",
    ): (1, "harness API server"),
    (
        "agency/harness/servers/harness_interaction_server.py",
        "HarnessInteractionServer.start",
    ): (1, "sandbox interaction server"),
    # ptrace supervision and subprocess pipe plumbing.
    (
        "agency/harness/ptrace/_tracer_loop.py",
        "TracerLoop.start",
    ): (1, "tracer loop"),
    (
        "agency/harness/ptrace/_tracer_loop.py",
        "TracerLoop._fork_and_exec",
    ): (3, "subprocess pipe readers/writer"),
    (
        "agency/harness/ptrace/_tracer_loop.py",
        "TracerLoop._fork_pty",
    ): (1, "subprocess terminal reader"),
    # General background I/O and UI maintenance.
    ("agency/utils/agutil.py", "_iter_batched"): (1, "stream iterator drainer"),
    (
        "agency/observability/agwebui/__init__.py",
        "agwebui.run",
    ): (2, "UI command relay + periodic disk-flush loop"),
    ("agency/observability/profiler/agprof.py", "spawn_traced"): (2, "spawn_traced implementation"),
}

# A raw Popen launches a process the profiler does not observe by
# construction (unlike subprocess.run, which blocks the calling, traced
# thread and reaps before returning).  Every site must either run inside a
# profiler-observed path or carry a reviewed reason here.  Same identity
# scheme as the thread allowlist.
_BARE_POPEN_ALLOWLIST = {
    # Sandbox process launch: chroot's own exec (no tracer available).
    (
        "agency/sandbox/chroot.py",
        "_ChrootBackend._run_unshared._call",
    ): (1, "chroot sandbox exec"),
    # Opens the profiled subprocess and records its own launch-to-reap span.
    ("agency/native_harness/profiling.py", "run_process"): (
        1,
        "profiler-observed subprocess launch",
    ),
    # The webui's own HTTP server; UI plumbing, not agent work.
    (
        "agency/observability/agwebui/__init__.py",
        "agwebui.run",
    ): (1, "webui server process"),
}


class _BareThreadVisitor(ast.NodeVisitor):
    def __init__(
        self,
        relative_path: str,
        *,
        threading_aliases: set[str],
        thread_aliases: set[str],
        subprocess_aliases: "set[str] | None" = None,
        popen_aliases: "set[str] | None" = None,
    ) -> None:
        self.relative_path = relative_path
        self.threading_aliases = threading_aliases
        self.thread_aliases = thread_aliases
        self.subprocess_aliases = subprocess_aliases or set()
        self.popen_aliases = popen_aliases or set()
        self.scope: list[str] = []
        self.calls: Counter[tuple[str, str]] = Counter()
        self.popen_calls: Counter[tuple[str, str]] = Counter()

    def _visit_scope(self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scope(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scope(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scope(node)

    def visit_Call(self, node: ast.Call) -> None:
        scope = ".".join(self.scope) or "<module>"
        if self._is_bare_thread(node):
            self.calls[(self.relative_path, scope)] += 1
        if self._is_bare_popen(node):
            self.popen_calls[(self.relative_path, scope)] += 1
        self.generic_visit(node)

    def _is_bare_thread(self, node: ast.Call) -> bool:
        if isinstance(node.func, ast.Attribute):
            owner = node.func.value
            return (
                node.func.attr == "Thread"
                and isinstance(owner, ast.Name)
                and owner.id in self.threading_aliases
            )
        if isinstance(node.func, ast.Name):
            return node.func.id in self.thread_aliases
        return False

    def _is_bare_popen(self, node: ast.Call) -> bool:
        if isinstance(node.func, ast.Attribute):
            owner = node.func.value
            return (
                node.func.attr == "Popen"
                and isinstance(owner, ast.Name)
                and owner.id in self.subprocess_aliases
            )
        if isinstance(node.func, ast.Name):
            return node.func.id in self.popen_aliases
        return False


def _thread_import_aliases(tree: ast.AST) -> tuple[set[str], set[str]]:
    """Return aliases that can construct ``threading.Thread`` directly.

    Collect imports in a separate pass so a conditional or function-local
    import cannot evade the guard merely by appearing after a call in AST
    traversal order.  A rare alias later rebound to an unrelated object is
    intentionally conservative: it must still be reviewed and classified.
    """
    threading_aliases: set[str] = set()
    thread_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name == "threading":
                    threading_aliases.add(imported.asname or "threading")
        elif isinstance(node, ast.ImportFrom) and node.module == "threading":
            for imported in node.names:
                if imported.name == "Thread":
                    thread_aliases.add(imported.asname or "Thread")
    return threading_aliases, thread_aliases


def _popen_import_aliases(tree: ast.AST) -> tuple[set[str], set[str]]:
    """Return aliases that can construct ``subprocess.Popen`` directly.

    Same conservative alias collection as the thread guard: ``import
    subprocess as sp`` and ``from subprocess import Popen as Launch`` are
    both caught, whatever their position in the file.
    """
    subprocess_aliases: set[str] = set()
    popen_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name == "subprocess":
                    subprocess_aliases.add(imported.asname or "subprocess")
        elif isinstance(node, ast.ImportFrom) and node.module == "subprocess":
            for imported in node.names:
                if imported.name == "Popen":
                    popen_aliases.add(imported.asname or "Popen")
    return subprocess_aliases, popen_aliases


def _bare_thread_calls() -> Counter[tuple[str, str]]:
    calls: Counter[tuple[str, str]] = Counter()
    for path in _AGENCY_ROOT.rglob("*.py"):
        # macOS rsync/tar exports can leave AppleDouble sidecars such as
        # ``._module.py``. They are binary metadata, not Python sources or
        # executable thread sites.
        if path.name.startswith("._"):
            continue
        # The guard audits threads shipped by the library, not concurrency
        # intentionally created by the library's own tests.
        if "tests" in path.relative_to(_AGENCY_ROOT).parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        threading_aliases, thread_aliases = _thread_import_aliases(tree)
        subprocess_aliases, popen_aliases = _popen_import_aliases(tree)
        visitor = _BareThreadVisitor(
            path.relative_to(_REPO_ROOT).as_posix(),
            threading_aliases=threading_aliases,
            thread_aliases=thread_aliases,
            subprocess_aliases=subprocess_aliases,
            popen_aliases=popen_aliases,
        )
        visitor.visit(tree)
        calls.update(visitor.calls)
    return calls


def _bare_popen_calls() -> Counter[tuple[str, str]]:
    calls: Counter[tuple[str, str]] = Counter()
    for path in _AGENCY_ROOT.rglob("*.py"):
        if path.name.startswith("._"):
            continue
        if "tests" in path.relative_to(_AGENCY_ROOT).parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        subprocess_aliases, popen_aliases = _popen_import_aliases(tree)
        visitor = _BareThreadVisitor(
            path.relative_to(_REPO_ROOT).as_posix(),
            threading_aliases=set(),
            thread_aliases=set(),
            subprocess_aliases=subprocess_aliases,
            popen_aliases=popen_aliases,
        )
        visitor.visit(tree)
        calls.update(visitor.popen_calls)
    return calls


def test_bare_thread_visitor_recognizes_import_aliases():
    tree = ast.parse(
        """
import threading as th
from threading import Thread as WorkerThread

def launch():
    th.Thread(target=lambda: None)
    WorkerThread(target=lambda: None)
"""
    )
    threading_aliases, thread_aliases = _thread_import_aliases(tree)
    visitor = _BareThreadVisitor(
        "agency/example.py",
        threading_aliases=threading_aliases,
        thread_aliases=thread_aliases,
    )
    visitor.visit(tree)

    assert visitor.calls == Counter({("agency/example.py", "launch"): 2})


def test_bare_popen_visitor_recognizes_import_aliases():
    tree = ast.parse(
        """
import subprocess as sp
from subprocess import Popen as Launch

def run():
    sp.Popen(["true"])
    Launch(["true"])
"""
    )
    subprocess_aliases, popen_aliases = _popen_import_aliases(tree)
    visitor = _BareThreadVisitor(
        "agency/example.py",
        threading_aliases=set(),
        thread_aliases=set(),
        subprocess_aliases=subprocess_aliases,
        popen_aliases=popen_aliases,
    )
    visitor.visit(tree)

    assert visitor.popen_calls == Counter({("agency/example.py", "run"): 2})


def test_session_stop_preserves_an_inflight_run_as_interrupted(monkeypatch, tmp_path):
    """An interrupted harness run must not disappear or look successful."""
    pytest.importorskip("opentelemetry.sdk.trace")
    from agency.observability.profiler import agprof

    monkeypatch.setattr(agprof, "_require_linux", lambda: None)
    started = threading.Event()
    release = threading.Event()

    def run() -> None:
        with agprof.span("run0:claude_code:golden-probe"):
            started.set()
            release.wait(timeout=10)

    agprof.start(tmp_path, sample_hz=0, sample_gpu=False)
    worker = agprof.spawn_traced(run)
    worker.start()
    assert started.wait(timeout=10)
    agprof.stop()
    release.set()
    worker.join(timeout=10)

    summary = json.loads((tmp_path / "summary.json").read_text())
    runs = summary["run_metrics"]
    assert runs["started"] == 1
    assert runs["completed"] == 0
    assert runs["succeeded"] == 0
    assert runs["failed"] == 0
    assert runs["interrupted"] == 1
    assert runs["p50_ms"] is None
    assert summary["incomplete_spans"][0]["label"] == "run0:claude_code:golden-probe"
    assert summary["incomplete_spans"][0]["outcome"] == "interrupted"


def test_bare_thread_sites_match_reviewed_allowlist():
    """A new bare task thread must be routed through agprof.spawn_traced()."""
    expected = Counter(
        {identity: count for identity, (count, _classification) in _BARE_THREAD_ALLOWLIST.items()}
    )

    assert _bare_thread_calls() == expected


def test_bare_popen_sites_match_reviewed_allowlist():
    """A new unobserved process launch must run under a profiler-observed
    path or carry a reviewed reason in the allowlist."""
    expected = Counter(
        {identity: count for identity, (count, _classification) in _BARE_POPEN_ALLOWLIST.items()}
    )

    assert _bare_popen_calls() == expected
