"""Facade-level tests for agSandbox with mock backends, plus the host GPU/CPU/
memory detection and agResourcePool unit tests.

The three tiers of real-backend sandbox tests live in tests/sandbox/ (see the
README there): nothing here starts a container or a jail. ``agSandbox`` is built
around a ``MagicMock`` backend so the facade's own behaviour (laziness, forking,
locking, config handling) is asserted in isolation.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import uuid

import pytest
from unittest.mock import MagicMock, patch

from agency.orchestrator.agresources import agResourcePool


def _facade(agconfig=None):
    """An agSandbox over a MagicMock backend (never touches a runtime)."""
    from agency.sandbox.agsandbox import agSandbox

    with patch("agency.sandbox.agsandbox.agsandbox_backend.for_config", return_value=MagicMock()):
        return agSandbox(str(uuid.uuid4()), agconfig=agconfig)


def test_facade_construction_does_not_start_backend():
    from agency.configs.agconfig import agconfig as agconfig_cls
    from agency.sandbox.agsandbox import agSandbox

    backend = MagicMock()
    with patch(
        "agency.sandbox.agsandbox.agsandbox_backend.for_config",
        return_value=backend,
    ):
        sandbox = agSandbox(str(uuid.uuid4()), agconfig=agconfig_cls())

    try:
        backend._ensure_started.assert_not_called()
        backend.exec.assert_not_called()
    finally:
        sandbox.destroy()


def test_fork_without_checkpoint_keeps_lazy_fresh_backend():
    from agency.sandbox.agsandbox import agSandbox

    parent_backend = MagicMock()
    child_backend = MagicMock()
    parent_backend._checkpoint_image = None
    child_backend._checkpoint_image = None
    with patch(
        "agency.sandbox.agsandbox.agsandbox_backend.for_config",
        side_effect=[parent_backend, child_backend],
    ):
        parent = agSandbox("fresh-parent")
        child = parent.fork("fresh-child")
    try:
        assert child._checkpoint_image is None
        child_backend._ensure_started.assert_not_called()
    finally:
        parent.destroy()
        child.destroy()


def test_destroy_can_retry_a_failed_backend_cleanup():
    from agency.sandbox.agsandbox import agSandbox

    backend = MagicMock()
    backend.destroy.side_effect = [RuntimeError("removal failed"), None]
    with patch("agency.sandbox.agsandbox.agsandbox_backend.for_config", return_value=backend):
        sandbox = agSandbox("retry-cleanup")
    with pytest.raises(RuntimeError, match="removal failed"):
        sandbox.destroy()
    sandbox.destroy()
    sandbox.destroy()
    assert backend.destroy.call_count == 2


@pytest.mark.parametrize(
    ("operation", "args"),
    [
        ("exec", ("true",)),
        ("exec_detached", ("true",)),
        ("read_file", ("/workspace/file.txt",)),
        ("write_file", ("/workspace/file.txt", "content")),
        ("commit", ()),
        ("stop", ()),
        ("destroy", ()),
    ],
)
def test_public_facade_operation_uses_shared_lock(operation, args):
    from agency.sandbox.agsandbox import agSandbox

    sandbox = agSandbox.__new__(agSandbox)
    sandbox._agname = "lock-test"
    sandbox._destroyed = operation != "destroy"
    sandbox._lock = threading.RLock()
    sandbox._backend = MagicMock()

    attempted = threading.Event()
    entered_backend = threading.Event()
    errors = []
    getattr(sandbox._backend, operation).side_effect = lambda *_args, **_kwargs: (
        entered_backend.set()
    )

    def call_operation():
        attempted.set()
        try:
            getattr(sandbox, operation)(*args)
        except Exception as exc:
            errors.append(exc)

    with sandbox._lock:
        worker = threading.Thread(target=call_operation)
        worker.start()
        assert attempted.wait(timeout=1)
        assert not entered_backend.wait(timeout=0.05)

    worker.join(timeout=1)
    assert not worker.is_alive()
    assert entered_backend.is_set()
    assert errors == []


# ---------------------------------------------------------------------------
# detect_gpus
# ---------------------------------------------------------------------------


class TestDetectGpus:
    def test_returns_list(self):
        from agency.utils.agutil import detect_gpus

        gpus = detect_gpus()
        assert isinstance(gpus, list)
        assert all(isinstance(g, int) for g in gpus)

    def test_nvidia_smi_unavailable_returns_empty(self, monkeypatch):
        from agency.utils.agutil import detect_gpus

        monkeypatch.setattr(
            "subprocess.run", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError())
        )
        assert detect_gpus() == []

    def test_nvidia_smi_nonzero_exit_returns_empty(self, monkeypatch):
        from unittest.mock import MagicMock
        from agency.utils.agutil import detect_gpus

        mock = MagicMock()
        mock.returncode = 1
        mock.stdout = ""
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock)
        assert detect_gpus() == []

    def test_nvidia_smi_parses_indices(self, monkeypatch):
        from unittest.mock import MagicMock
        from agency.utils.agutil import detect_gpus

        mock = MagicMock()
        mock.returncode = 0
        mock.stdout = "0\n1\n2\n"
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock)
        # Isolate from whatever CUDA_VISIBLE_DEVICES happens to be set to on
        # the host running this test -- this test is about nvidia-smi output
        # parsing, not _cvd_filter (see TestCvdFilter for that).
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        assert detect_gpus() == [0, 1, 2]

    def test_cvd_filter_applied_to_nvidia_smi_output(self, monkeypatch):
        from unittest.mock import MagicMock
        from agency.utils.agutil import detect_gpus

        mock = MagicMock()
        mock.returncode = 0
        mock.stdout = "0\n1\n2\n3\n"
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")
        assert detect_gpus() == [1, 3]

    def test_cvd_unset_returns_all_from_nvidia_smi(self, monkeypatch):
        from unittest.mock import MagicMock
        from agency.utils.agutil import detect_gpus

        mock = MagicMock()
        mock.returncode = 0
        mock.stdout = "0\n1\n2\n"
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        assert detect_gpus() == [0, 1, 2]


# ---------------------------------------------------------------------------
# _cvd_filter
# ---------------------------------------------------------------------------


class TestCvdFilter:
    def test_no_env_var_passes_all(self, monkeypatch):
        from agency.utils.agutil import _cvd_filter

        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        assert _cvd_filter([0, 1, 2, 3]) == [0, 1, 2, 3]

    def test_filters_to_allowed_subset(self, monkeypatch):
        from agency.utils.agutil import _cvd_filter

        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,3,5,7")
        assert _cvd_filter([0, 1, 2, 3, 4, 5, 6, 7]) == [0, 3, 5, 7]

    def test_single_gpu(self, monkeypatch):
        from agency.utils.agutil import _cvd_filter

        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
        assert _cvd_filter([0, 1, 2, 3]) == [3]

    def test_empty_string_passes_all(self, monkeypatch):
        from agency.utils.agutil import _cvd_filter

        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        assert _cvd_filter([0, 1, 2]) == [0, 1, 2]

    def test_nodevfiles_passes_all(self, monkeypatch):
        from agency.utils.agutil import _cvd_filter

        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "NoDevFiles")
        assert _cvd_filter([0, 1, 2]) == [0, 1, 2]

    def test_none_string_passes_all(self, monkeypatch):
        from agency.utils.agutil import _cvd_filter

        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "none")
        assert _cvd_filter([0, 1, 2]) == [0, 1, 2]

    def test_cvd_id_not_in_pool_ignored(self, monkeypatch):
        from agency.utils.agutil import _cvd_filter

        # CVD says GPU 9 is allowed but nvidia-smi only reported [0,1,2]
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,9")
        assert _cvd_filter([0, 1, 2]) == [0]

    def test_preserves_order_from_pool_list(self, monkeypatch):
        from agency.utils.agutil import _cvd_filter

        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,3,1")
        # Order follows the pool list, not CVD order
        assert _cvd_filter([0, 1, 2, 3, 4, 5]) == [1, 3, 5]

    def test_pool_auto_detect_respects_cvd(self, monkeypatch):
        from unittest.mock import MagicMock

        mock = MagicMock()
        mock.returncode = 0
        mock.stdout = "0\n1\n2\n3\n"
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,3,5,7")
        pool = agResourcePool()
        # Only GPUs 0 and 3 overlap between [0,1,2,3] and {0,3,5,7}
        assert pool.gpus == [0, 3]


class TestDetectCpus:
    def test_returns_positive_int(self):
        from agency.utils.agutil import detect_cpus

        cpus = detect_cpus()
        assert isinstance(cpus, int)
        assert cpus >= 1

    def test_os_cpu_count_none_returns_one(self, monkeypatch):
        from agency.utils.agutil import detect_cpus

        monkeypatch.setattr("os.cpu_count", lambda: None)
        assert detect_cpus() == 1


class TestDetectMemoryMb:
    def test_returns_positive_int(self):
        from agency.utils.agutil import detect_memory_mb

        mb = detect_memory_mb()
        assert isinstance(mb, int)
        assert mb > 0

    def test_fallback_when_proc_missing(self, monkeypatch, tmp_path):
        from unittest.mock import MagicMock
        from agency.utils.agutil import detect_memory_mb

        # Point /proc/meminfo to a non-existent path and make sysctl fail
        monkeypatch.setattr("builtins.open", lambda *a, **kw: (_ for _ in ()).throw(OSError()))
        mock = MagicMock()
        mock.returncode = 1
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock)
        assert detect_memory_mb() == 4096  # safe fallback


class TestPoolAutoDetect:
    def test_pool_auto_detects_gpus(self, monkeypatch):
        from unittest.mock import MagicMock

        mock = MagicMock()
        mock.returncode = 0
        mock.stdout = "0\n1\n"
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock)
        # Isolate from the host's real CUDA_VISIBLE_DEVICES -- this test is
        # about auto-detection wiring, not CVD filtering (see
        # test_pool_auto_detect_respects_cvd for that).
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        pool = agResourcePool()
        assert pool.gpus == [0, 1]

    def test_pool_auto_detects_cpus(self, monkeypatch):
        monkeypatch.setattr("os.cpu_count", lambda: 16)
        pool = agResourcePool(gpus=[])
        assert pool.total_cpus == 16

    def test_pool_explicit_overrides_detection(self):
        pool = agResourcePool(gpus=[0], total_cpus=4, total_memory_mb=8192)
        assert pool.gpus == [0]
        assert pool.total_cpus == 4
        assert pool.total_memory_mb == 8192

    def test_orchestrator_has_default_pool(self):
        from agency.orchestrator import get_orchestrator

        pool = get_orchestrator().agresource_pool
        assert pool is not None
        assert isinstance(pool.total_cpus, int)
        assert isinstance(pool.total_memory_mb, int)


# ---------------------------------------------------------------------------
# agResourcePool
# ---------------------------------------------------------------------------


def _resource_sandbox(name="test-sandbox"):
    sb = MagicMock()
    sb._name = name
    sb._real_gpu_ids = []
    sb._cpu_acquired = 0.0
    sb._memory_acquired_mb = 0
    return sb


class TestAgResourcePool:
    def test_single_gpu_acquire_release(self):
        pool = agResourcePool(gpus=[0])
        sandbox = _resource_sandbox()
        gids = pool.acquire_real_gpus(sandbox, 1)
        assert gids == [0]
        pool.release_real_gpus(sandbox, gids)

    def test_two_gpus_both_acquired(self):
        pool = agResourcePool(gpus=[0, 1])
        sandbox = _resource_sandbox()
        g1 = pool.acquire_real_gpus(sandbox, 1)
        g2 = pool.acquire_real_gpus(sandbox, 1)
        assert set(g1) | set(g2) == {0, 1}
        pool.release_real_gpus(sandbox, g1)
        pool.release_real_gpus(sandbox, g2)

    def test_acquire_blocks_until_released(self):
        pool = agResourcePool(gpus=[0])
        sandbox = _resource_sandbox()
        pool.acquire_real_gpus(sandbox, 1)  # hold the only GPU

        acquired: list[int] = []

        def _waiter():
            acquired.extend(pool.acquire_real_gpus(_resource_sandbox(), 1))

        t = threading.Thread(target=_waiter)
        t.start()
        time.sleep(0.1)
        assert acquired == []  # still blocked
        pool.release_real_gpus(sandbox, [0])
        t.join(timeout=2)
        assert acquired == [0]

    def test_acquire_timeout_raises(self):
        pool = agResourcePool(gpus=[0])
        pool.acquire_real_gpus(_resource_sandbox(), 1)  # exhaust pool
        with pytest.raises(TimeoutError):
            pool.acquire_real_gpus(_resource_sandbox(), 1, timeout=0.2)

    def test_release_unowned_gpu_is_safe(self):
        pool = agResourcePool(gpus=[0])
        pool.release_real_gpus(_resource_sandbox(), [0])  # never acquired — should not raise

    def test_release_unknown_gpu_is_safe(self):
        pool = agResourcePool(gpus=[0])
        pool.release_real_gpus(_resource_sandbox(), [99])  # not in pool — should not raise

    def test_repr(self):
        pool = agResourcePool(gpus=[0, 1], idle_cpus=1.0, idle_memory="1g")
        r = repr(pool)
        assert "agResourcePool" in r
        assert "[0, 1]" in r
        assert "total_cpus" in r
        assert "total_memory_mb" in r


# ---------------------------------------------------------------------------
# GPU presence markers
# ---------------------------------------------------------------------------


class TestGpuMarkers:
    """GPU markers are now allocated in-process via ctypes (no subprocesses)."""

    def test_mark_gpus_false_does_not_call_allocate(self):
        from agency.orchestrator import agresources

        with patch.object(agresources, "_allocate_gpu_markers") as mock_alloc:
            agResourcePool(gpus=[0, 1], mark_gpus=False)
        mock_alloc.assert_not_called()

    def test_mark_gpus_true_empty_gpu_list_does_not_call_allocate(self):
        from agency.orchestrator import agresources

        with patch.object(agresources, "_allocate_gpu_markers") as mock_alloc:
            agResourcePool(gpus=[], mark_gpus=True)
        mock_alloc.assert_not_called()

    def test_mark_gpus_true_calls_allocate_with_gpu_list(self):
        from agency.orchestrator import agresources

        with patch.object(agresources, "_allocate_gpu_markers") as mock_alloc:
            agResourcePool(gpus=[0, 1], mark_gpus=True)
        mock_alloc.assert_called_once_with([0, 1])

    def test_mark_gpus_true_single_gpu_calls_allocate(self):
        from agency.orchestrator import agresources

        with patch.object(agresources, "_allocate_gpu_markers") as mock_alloc:
            agResourcePool(gpus=[2], mark_gpus=True)
        mock_alloc.assert_called_once_with([2])

    def test_no_marker_procs_attribute(self):
        pool = agResourcePool(gpus=[0], mark_gpus=False)
        assert not hasattr(pool, "_marker_procs")

    def test_no_stop_gpu_markers_method(self):
        pool = agResourcePool(gpus=[0], mark_gpus=False)
        assert not hasattr(pool, "_stop_gpu_markers")

    def test_allocate_gpu_markers_skips_on_no_libcuda(self):
        from agency.utils.agutil import _allocate_gpu_markers
        import ctypes

        with patch.object(ctypes, "CDLL", side_effect=OSError("libcuda.so.1 not found")):
            _allocate_gpu_markers([0, 1])  # must not raise

    def test_allocate_gpu_markers_skips_on_cuinit_failure(self):
        from agency.utils.agutil import _allocate_gpu_markers
        import ctypes

        mock_cuda = MagicMock()
        mock_cuda.cuInit.return_value = 1  # CUDA_ERROR_NOT_INITIALIZED
        with patch.object(ctypes, "CDLL", return_value=mock_cuda):
            _allocate_gpu_markers([0])  # must not raise
        mock_cuda.cuCtxCreate_v2.assert_not_called()

    def test_allocate_gpu_markers_remaps_cuda_device_indices_with_cvd(self):
        """When CUDA_VISIBLE_DEVICES=0,3,5,7, physical IDs must be remapped to
        CUDA device indices 0-3 before calling cuCtxCreate_v2.  This is the
        exact bug that caused markers to be missing on GPUs 3 and 5."""
        from agency.utils.agutil import _allocate_gpu_markers
        import ctypes

        mock_cuda = MagicMock()
        mock_cuda.cuInit.return_value = 0  # success
        mock_cuda.cuCtxCreate_v2.return_value = 0
        mock_cuda.cuMemAlloc_v2.return_value = 0
        with patch.object(ctypes, "CDLL", return_value=mock_cuda):
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,3,5,7"}):
                _allocate_gpu_markers([0, 3, 5, 7])
        # Extract the device argument (3rd positional arg) from each call
        called_devs = [call.args[2] for call in mock_cuda.cuCtxCreate_v2.call_args_list]
        assert called_devs == [0, 1, 2, 3], (
            f"Expected CUDA device indices [0,1,2,3], got {called_devs}. "
            "Physical GPU IDs were passed directly instead of being remapped."
        )

    def test_allocate_gpu_markers_falls_back_to_rocm_when_no_libcuda(self):
        """On an AMD-only host (no libcuda.so.1 at all), markers must be
        allocated via ROCm/HIP instead of silently doing nothing."""
        from agency.utils.agutil import _allocate_gpu_markers
        import ctypes

        mock_hip = MagicMock()
        mock_hip.hipInit.return_value = 0
        mock_hip.hipSetDevice.return_value = 0
        mock_hip.hipMalloc.return_value = 0

        def _cdll(name, *a, **kw):
            if name == "libcuda.so.1":
                raise OSError("libcuda.so.1 not found")
            assert name == "libamdhip64.so"
            return mock_hip

        with patch.object(ctypes, "CDLL", side_effect=_cdll):
            _allocate_gpu_markers([0, 1])
        assert mock_hip.hipSetDevice.call_count == 2
        assert mock_hip.hipMalloc.call_count == 2

    def test_allocate_gpu_markers_skips_rocm_on_hipinit_failure(self):
        from agency.utils.agutil import _allocate_gpu_markers
        import ctypes

        mock_hip = MagicMock()
        mock_hip.hipInit.return_value = 1  # failure

        def _cdll(name, *a, **kw):
            if name == "libcuda.so.1":
                raise OSError("libcuda.so.1 not found")
            return mock_hip

        with patch.object(ctypes, "CDLL", side_effect=_cdll):
            _allocate_gpu_markers([0])  # must not raise
        mock_hip.hipSetDevice.assert_not_called()

    def test_allocate_gpu_markers_skips_entirely_when_neither_cuda_nor_rocm_present(self):
        from agency.utils.agutil import _allocate_gpu_markers
        import ctypes

        with patch.object(ctypes, "CDLL", side_effect=OSError("not found")):
            _allocate_gpu_markers([0, 1])  # must not raise

    def test_allocate_gpu_markers_remaps_hip_device_indices_with_hip_visible_devices(self):
        """When HIP_VISIBLE_DEVICES=1,3, physical IDs must be remapped to HIP
        device indices 0-1 before calling hipSetDevice -- same remap bug class
        as the CUDA/CUDA_VISIBLE_DEVICES case above, for the ROCm path."""
        from agency.utils.agutil import _allocate_gpu_markers
        import ctypes

        mock_hip = MagicMock()
        mock_hip.hipInit.return_value = 0
        mock_hip.hipSetDevice.return_value = 0
        mock_hip.hipMalloc.return_value = 0

        def _cdll(name, *a, **kw):
            if name == "libcuda.so.1":
                raise OSError("libcuda.so.1 not found")
            return mock_hip

        with patch.object(ctypes, "CDLL", side_effect=_cdll):
            with patch.dict(os.environ, {"HIP_VISIBLE_DEVICES": "1,3"}):
                _allocate_gpu_markers([1, 3])
        called_devs = [call.args[0] for call in mock_hip.hipSetDevice.call_args_list]
        assert called_devs == [0, 1], (
            f"Expected HIP device indices [0,1], got {called_devs}. "
            "Physical GPU IDs were passed directly instead of being remapped."
        )

    def test_allocate_gpu_markers_does_not_try_rocm_when_cuda_available(self):
        """CUDA present and working -- ROCm/HIP must never be attempted."""
        from agency.utils.agutil import _allocate_gpu_markers
        import ctypes

        mock_cuda = MagicMock()
        mock_cuda.cuInit.return_value = 0
        mock_cuda.cuCtxCreate_v2.return_value = 0
        mock_cuda.cuMemAlloc_v2.return_value = 0
        with patch.object(ctypes, "CDLL", return_value=mock_cuda) as cdll:
            _allocate_gpu_markers([0])
        cdll.assert_called_once_with("libcuda.so.1")

    def test_non_main_process_name_blocks_allocation(self):
        """The MainProcess guard must block _allocate_gpu_markers in worker processes."""
        from agency.orchestrator import agresources

        mock_proc = MagicMock()
        mock_proc.name = "ForkPoolWorker-1"
        with patch("multiprocessing.current_process", return_value=mock_proc):
            with patch.object(agresources, "_allocate_gpu_markers") as mock_alloc:
                agResourcePool(gpus=[0], mark_gpus=True)
        mock_alloc.assert_not_called()

    def test_subprocess_import_does_not_call_allocate(self):
        """Importing agent in a subprocess must not call _allocate_gpu_markers."""
        script = (
            "import sys; "
            "from unittest.mock import patch; "
            "from agency.orchestrator import agresources; "
            "calls = []; "
            "original = agresources._allocate_gpu_markers; "
            "agresources._allocate_gpu_markers = lambda ids: calls.append(ids) or original(ids); "
            "from agency.agent import agent; "
            "assert calls == [], f'allocate called: {calls}'"
        )
        child = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            timeout=15,
        )
        assert child.returncode == 0, child.stderr.decode()


# ---------------------------------------------------------------------------
# agSandbox — change_config / get_config_copy
# ---------------------------------------------------------------------------


class TestAgSandboxChangeConfigAndGetConfigCopy:
    def test_change_config_replaces_agconfig(self):
        from agency.configs.agconfig import agconfig as agconfig_cls, llmconfig

        sb = _facade(agconfig_cls(llmconfig(temperature=0.7)))
        sb.change_config(agconfig_cls(llmconfig(temperature=0.2)))
        assert sb.agconfig.llm.temperature == 0.2

    def test_change_config_clones_given_agconfig(self):
        from agency.configs.agconfig import agconfig as agconfig_cls, llmconfig

        sb = _facade(agconfig_cls())
        new_cfg = agconfig_cls(llmconfig(temperature=0.2))
        sb.change_config(new_cfg)
        new_cfg.llm.temperature = 0.9
        assert sb.agconfig.llm.temperature == 0.2

    def test_get_config_copy_returns_clone_not_same_object(self):
        from agency.configs.agconfig import agconfig as agconfig_cls, llmconfig

        cfg = agconfig_cls(llmconfig(temperature=0.7))
        sb = _facade(cfg)
        copy = sb.get_config_copy()
        assert copy is not sb.agconfig

    def test_get_config_copy_reflects_current_values(self):
        from agency.configs.agconfig import agconfig as agconfig_cls, llmconfig

        sb = _facade(agconfig_cls(llmconfig(temperature=0.7)))
        assert sb.get_config_copy().llm.temperature == 0.7

    def test_mutating_get_config_copy_does_not_affect_sandbox(self):
        from agency.configs.agconfig import agconfig as agconfig_cls, llmconfig

        sb = _facade(agconfig_cls(llmconfig(temperature=0.7)))
        copy = sb.get_config_copy()
        copy.llm.temperature = 0.1
        assert sb.agconfig.llm.temperature == 0.7

    def test_get_config_copy_defaults_when_no_agconfig_given(self):
        # agSandbox.agconfig is always a real agconfig instance (never None), so
        # constructing without one just means the sandbox falls back to defaults.
        from agency.configs.agconfig import agconfig as agconfig_cls

        sb = _facade()
        copy = sb.get_config_copy()
        assert copy is not None
        assert copy.sandbox.base_image == agconfig_cls().sandbox.base_image

    def test_change_config_none_resets_to_default_agconfig(self):
        # agSandbox.agconfig can never be None -- change_config(None) resets
        # it to a fresh default agconfig instead of clearing it.
        from agency.configs.agconfig import agconfig as agconfig_cls, llmconfig

        sb = _facade(agconfig_cls(llmconfig(temperature=0.7)))
        sb.change_config(None)
        assert sb.get_config_copy() is not None
        assert sb.get_config_copy().llm.temperature == agconfig_cls().llm.temperature
