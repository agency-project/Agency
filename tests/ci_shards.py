"""CI shard definitions for the sandbox/harness pytest job.

The CI workflow (.github/workflows/tests.yml) runs tests/sandbox and
tests/harness as 8 parallel matrix shards, each serial on its own runner,
because those tests are sensitive to CPU contention (cgroup throughput
assertions, wall-clock budgets, ptrace timing). Shards are balanced
greedily by test count; rebalance when large test files are added.

Everything outside tests/sandbox and tests/harness runs in the separate
"general" job with pytest-xdist and needs no shard list.
"""

from __future__ import annotations

# Greedy bin-packing of sandbox/harness test files into 8 shards, heaviest
# file first into the lightest shard. Counts are `def test` occurrences;
# backend parametrization multiplies real runtimes roughly equally across
# shards, so file-level counts stay a fair basis.
_SANDBOX_SHARD_FILES: "list[list[str]]" = [
    # Shard 1
    ["tests/sandbox/test_container_backends.py"],
    # Shard 2
    ["tests/sandbox/test_all_backends.py"],
    # Shard 3
    [
        "tests/sandbox/test_chroot_backend.py",
        "tests/sandbox/test_cow_zfs_criu_linux.py",
        "tests/harness/test_control_observation.py",
        "tests/harness/test_agharness.py",
        "tests/harness/test_native_lifecycle_boundary.py",
        "tests/sandbox/test_harness_checkpoint_matrix_linux.py",
        "tests/harness/test_kimi_pty_live.py",
    ],
    # Shard 4
    [
        "tests/sandbox/test_docker_backend.py",
        "tests/harness/adapters/test_chat_completions_protocol.py",
        "tests/sandbox/test_hibernation_liveness.py",
        "tests/harness/adapters/test_claude_code.py",
        "tests/sandbox/test_cow_zfs_fast_resume.py",
        "tests/harness/test_gpu_exec_gate_linux.py",
        "tests/harness/adapters/test_grok.py",
        "tests/harness/test_dual_uds_rpc.py",
        "tests/harness/test_mcp_transport.py",
    ],
    # Shard 5
    [
        "tests/harness/test_external_pty.py",
        "tests/sandbox/test_layer_squash.py",
        "tests/harness/test_harness_permission_hook.py",
        "tests/harness/adapters/test_native_llm.py",
        "tests/sandbox/test_pid_diagnostics.py",
        "tests/harness/adapters/test_adapter_routes.py",
        "tests/harness/test_claude_pty.py",
        "tests/sandbox/test_cow_zfs_linux.py",
        "tests/harness/test_claude_pty_live.py",
        "tests/harness/test_ptrace_read_edge_linux.py",
    ],
    # Shard 6
    [
        "tests/sandbox/test_podman_backend.py",
        "tests/sandbox/test_base.py",
        "tests/harness/test_kimi_pty.py",
        "tests/harness/adapters/test_daemon_seam.py",
        "tests/harness/test_native_hooks.py",
        "tests/harness/test_fast_resume.py",
        "tests/harness/test_protocol.py",
        "tests/harness/test_attribution_smoke.py",
        "tests/harness/test_executable_container.py",
        "tests/harness/adapters/test_codex.py",
    ],
    # Shard 7
    [
        "tests/harness/adapters/test_codex_llm.py",
        "tests/harness/test_daemon.py",
        "tests/harness/test_executable.py",
        "tests/harness/adapters/test_base.py",
        "tests/harness/test_harness_pins.py",
        "tests/harness/test_sandbox_mcp.py",
        "tests/harness/test_time_ctrl_linux.py",
        "tests/harness/test_checkpoint_handoff_linux.py",
        "tests/harness/test_external_pty_live.py",
        "tests/harness/adapters/test_opencode.py",
    ],
    # Shard 8
    [
        "tests/harness/test_agproxy_ptrace.py",
        "tests/harness/adapters/test_claude_code_llm.py",
        "tests/sandbox/test_base_gpu_lease.py",
        "tests/harness/test_host_client_routes.py",
        "tests/harness/test_pause_ack.py",
        "tests/harness/test_traced_pty.py",
        "tests/sandbox/test_cow_zfs.py",
        "tests/harness/test_initial_exec_authorization.py",
        "tests/harness/test_host_services_client_mount.py",
    ],
]

_SANDBOX_SHARD_COUNT = len(_SANDBOX_SHARD_FILES)


def sandbox_shard_files(shard_number: int) -> "list[str]":
    """The test files for *shard_number* (1-based). Raises on an out-of-range
    number so a matrix/shard-list mismatch fails loudly instead of silently
    skipping coverage."""
    if not 1 <= shard_number <= _SANDBOX_SHARD_COUNT:
        raise ValueError(f"shard number {shard_number} out of range 1..{_SANDBOX_SHARD_COUNT}")
    return _SANDBOX_SHARD_FILES[shard_number - 1]


def sandbox_shard_count() -> int:
    return _SANDBOX_SHARD_COUNT
