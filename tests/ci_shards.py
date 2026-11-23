"""CI shard definitions for the sandbox/harness pytest job.

The CI workflow (.github/workflows/tests.yml) runs tests/sandbox and
tests/harness as 3 parallel matrix shards, each serial on its own runner,
because those tests are sensitive to CPU contention (cgroup throughput
assertions, wall-clock budgets, ptrace timing). The split follows observed
CI wall time: the x3-backend parametrized contract file ran ~22 minutes
alone, so it was split into two files (contract basics vs checkpoints and
process tracking) that get one runner each; every other sandbox/harness
file together runs well under that and shares the third runner.

Everything outside tests/sandbox and tests/harness runs in the separate
"general" job with pytest-xdist and needs no shard list.
"""

from __future__ import annotations

_SANDBOX_SHARD_FILES: "dict[str, list[str]]" = {
    # The two halves of the split tier-3 contract file.
    "all_backends_contract": ["tests/sandbox/test_all_backends.py"],
    "all_backends_checkpoints": ["tests/sandbox/test_all_backends_checkpoints.py"],
    # All remaining sandbox/harness files.
    "everything_else": [
        "tests/sandbox/test_container_backends.py",
        "tests/sandbox/test_cpu_controller_probe.py",
        "tests/sandbox/test_chroot_backend.py",
        "tests/sandbox/test_docker_backend.py",
        "tests/sandbox/test_podman_backend.py",
        "tests/sandbox/test_layer_squash.py",
        "tests/sandbox/test_base.py",
        "tests/sandbox/test_base_gpu_lease.py",
        "tests/sandbox/test_hibernation_liveness.py",
        "tests/sandbox/test_pid_diagnostics.py",
        "tests/sandbox/test_cow_zfs_criu_linux.py",
        "tests/sandbox/test_cow_zfs_fast_resume.py",
        "tests/sandbox/test_cow_zfs_linux.py",
        "tests/sandbox/test_cow_zfs.py",
        "tests/harness/test_external_pty.py",
        "tests/harness/test_agproxy_ptrace.py",
        "tests/harness/test_daemon.py",
        "tests/harness/test_executable.py",
        "tests/harness/test_kimi_pty.py",
        "tests/harness/test_harness_permission_hook.py",
        "tests/harness/test_host_client_routes.py",
        "tests/harness/test_native_hooks.py",
        "tests/harness/test_harness_pins.py",
        "tests/harness/test_pause_ack.py",
        "tests/harness/test_control_observation.py",
        "tests/harness/test_traced_pty.py",
        "tests/harness/test_sandbox_mcp.py",
        "tests/harness/test_time_ctrl_linux.py",
        "tests/harness/test_claude_prompt_ack.py",
        "tests/harness/test_context_window.py",
        "tests/harness/test_context_window_live.py",
        "tests/harness/test_protocol.py",
        "tests/harness/test_gpu_exec_gate_linux.py",
        "tests/harness/test_fast_resume.py",
        "tests/harness/test_native_lifecycle_boundary.py",
        "tests/harness/test_initial_exec_authorization.py",
        "tests/harness/test_claude_pty.py",
        "tests/harness/test_checkpoint_handoff_linux.py",
        "tests/harness/test_attribution_smoke.py",
        "tests/harness/test_external_pty_live.py",
        "tests/harness/test_claude_pty_live.py",
        "tests/harness/test_kimi_pty_live.py",
        "tests/harness/test_host_services_client_mount.py",
        "tests/harness/test_executable_container.py",
        "tests/harness/test_dual_uds_rpc.py",
        "tests/harness/test_mcp_transport.py",
        "tests/harness/test_agharness.py",
        "tests/harness/adapters/test_codex_llm.py",
        "tests/harness/adapters/test_claude_code_llm.py",
        "tests/harness/adapters/test_chat_completions_protocol.py",
        "tests/harness/adapters/test_native_llm.py",
        "tests/harness/adapters/test_daemon_seam.py",
        "tests/harness/adapters/test_base.py",
        "tests/harness/adapters/test_claude_code.py",
        "tests/harness/adapters/test_adapter_routes.py",
        "tests/harness/adapters/test_grok.py",
        "tests/harness/adapters/test_opencode.py",
        "tests/harness/adapters/test_codex.py",
        "tests/harness/test_ptrace_read_edge_linux.py",
        "tests/sandbox/test_harness_checkpoint_matrix_linux.py",
    ],
}

_SANDBOX_SHARD_COUNT = len(_SANDBOX_SHARD_FILES)


def sandbox_shard_files(shard_name: str) -> "list[str]":
    """The test files for *shard_name*. Raises on an unknown name so a
    matrix/shard-list mismatch fails loudly instead of silently skipping
    coverage."""
    try:
        return _SANDBOX_SHARD_FILES[shard_name]
    except KeyError:
        raise ValueError(
            f"unknown sandbox shard {shard_name!r} (expected one of {sorted(_SANDBOX_SHARD_FILES)})"
        ) from None


def sandbox_shard_count() -> int:
    return _SANDBOX_SHARD_COUNT
