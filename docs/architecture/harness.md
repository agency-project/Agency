# harness — sandbox integration and supervision

`agency/harness/` adapts coding-agent implementations to Agency's execution protocol. It owns the sandbox daemon, harness-facing gateway, CLI-specific adapters and process supervision. The [engine](engine.md) sends it an attempt; the selected adapter determines how the coding agent runs.

## Internal responsibilities

| Source area | Responsibility |
| --- | --- |
| [protocol.py](../../agency/harness/protocol.py) | Shared prompt, attempt request and attempt result data structures. |
| [daemon.py](../../agency/harness/daemon.py) | Long-lived sandbox manager, loopback gateway, attempt routing and controls. |
| [adapters/](../../agency/harness/adapters/base.py) | Harness-specific launching, model protocol translation and session handling. |
| [adapters/pty/](../../agency/harness/adapters/pty/execution.py) | Interactive terminal execution and driver lifecycle for external CLIs. |
| [ptrace/](../../agency/harness/ptrace/supervisor.py) | Linux process tracing, syscall observation, policy and process controls. |
| [clients/](../../agency/harness/clients/host_services_client.py) and [servers/](../../agency/harness/servers/harness_interaction_server.py) | Callback clients to the host and the daemon's execution/control server. |
| [sandbox_mcp.py](../../agency/harness/sandbox_mcp.py) and [mcp_proxy.py](../../agency/harness/mcp_proxy.py) | Sandbox tool serving and MCP transport forwarding. |

## Two directions across the boundary

The host controls the daemon over a sandbox socket. Harness model, tool and policy traffic travels through the daemon's gateway to the host service socket. An attempt carries its prompt, request identity, fresh token and optional session state; host Python tool closures and provider credentials stay with host services. Explicit sandbox tools are serialized and reconstructed in the sandbox.

The adapters currently include native, Claude Code, Codex, opencode, Grok and Kimi. Each adapter configures its harness to use the gateway and translates its model-facing protocol. Shared configuration-home helpers isolate per-launch settings; CLI command dialects and session formats stay in the concrete adapters. This is what allows harness choice and model-backend choice to vary independently. How each harness is told the model's context limit, and given token usage it can compact on, is in [Context_Window.md](../Context_Window.md).

The ptrace supervisor observes and controls the launched process tree. External adapters use PTY drivers, while the native adapter launches the [standalone native loop](native_harness.md) with pipe output and additional semantic activity hooks. The process tracing implementation depends on local Linux x86-64 facilities.

## Completion and session continuity

Adapters return a common `HarnessAttemptResult`, but completion evidence remains harness-specific. PTY drivers inspect terminal and session events; native execution reports its loop result and progress files. Session bundles preserve selected files for later invocations, separately from sandbox filesystem checkpoints or retained process memory.

Inactivity paths currently allow PTY and native adapters to return partial or empty text with `ok=True`. The engine's output-schema checks are a later gate; raw-text output can pass without proving the task is solved. Harness substitution therefore shares transport and contracts while retaining differences in completion and control behavior.

Freezing the tree also freezes its clocks when `ptrace.time_control` is on ([time_ctrl.py](../../agency/harness/ptrace/time_ctrl.py)). While every task is parked, monotonic clock reads return real time minus the total frozen time, timed waits that carry an absolute deadline are translated from virtual to real time, and timers already armed in the kernel (`timerfd`, `setitimer`, POSIX timers) are disarmed and re-armed around the freeze. Interrupted relative waits return `EINTR`, so programs that retry on `EINTR` (nearly all) keep their virtual deadline. Clock reads trap, which costs roughly 16-25 microseconds each in the traced tree. Real-time clocks (`time()`, `Date.now()`) stay real unless `ptrace.time_realtime` is set; a connection to an outside peer can still be dropped by the peer's own timers.

An `execve` made after the agent reserved GPUs is admitted by the host. If a GPU is free the exec gets `CUDA_VISIBLE_DEVICES` and the process becomes a lease holder. If none is free the host answers `pending`: the daemon parks that exec, freezes the whole tree and its clocks, waits on the host without a timeout, then applies the environment and resumes. Descendants of a holder are holders, and the lease returns to the pool when the last holder exits (or at sandbox teardown). A task parked in a `vfork` wait counts as frozen.

Optional fast resume can retain a compatible idle PTY execution. The daemon binds retained local traffic to the current host attempt and reestablishes tracing after restoration. Filesystem and process checkpoint ownership remains in [sandbox](sandbox.md). See [adapter route tests](../../tests/harness/adapters/test_adapter_routes.py) and [PTY tests](../../tests/harness/test_external_pty.py) for integration behavior.
