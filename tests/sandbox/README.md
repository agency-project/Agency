# Sandbox tests

The sandbox has three backends: `chroot` (no daemon, a directory plus an
unprivileged user/mount namespace), `docker` and `podman` (the two container
runtimes, one shared implementation in `agency/sandbox/container.py`). The tests
are organised in three tiers so that every behaviour is asserted exactly once,
at the widest scope that is true for it.

| Tier | Scope | Files |
| --- | --- | --- |
| 3 | Sandbox-wide: the `agSandbox` contract, run on **every** backend | `test_all_backends.py` |
| 2 | Family: the container family (docker + podman, parametrized) and the non-container family (chroot is its only member) | `test_container_backends.py`, and section 1 of `test_chroot_backend.py` |
| 1 | Backend-specific mechanisms | `test_docker_backend.py`, `test_podman_backend.py`, section 2 of `test_chroot_backend.py` |

**The rule.** Behaviour shared by the backends goes in `test_all_backends.py`.
Behaviour shared by docker and podman goes in `test_container_backends.py`. Only
mechanisms that genuinely exist in one backend (docker's `ctr`/containerd
lookups, podman's `layers.json` and `--init-path`, chroot's `unshare` probing and
jail setup script...) go in the per-backend files. If you are about to write the
same test for a second backend, move it up a tier and parametrize it instead.

## Other files

| File | What it covers |
| --- | --- |
| `test_base.py` | Base-class tier: backend selection (`for_config`, auto-detection), checkpoint-kind routing (`backend_for_image_kind`), serialization of a backend for a worker process. |
| `test_base_gpu_lease.py` | Real-GPU lease bookkeeping every backend inherits from `agsandbox_backend`; each test runs for chroot, docker and podman (mocks only). |
| `test_hibernation_liveness.py` | Background-process liveness logic in the base class. |
| `test_pid_diagnostics.py`, `test_layer_squash.py` | Pure helper modules (`pid_diagnostics`, `_layer_squash`); no backend involved. |
| `test_cow_zfs.py`, `test_cow_zfs_fast_resume.py` | `cow_zfs` checkpoint units: the runtime-independent parts, parametrized over docker/podman. The per-runtime halves are in the backend files. |
| `test_cow_zfs_linux.py`, `test_cow_zfs_criu_linux.py`, `test_harness_checkpoint_matrix_linux.py` | Opt-in tests against real ZFS/CRIU; the runtime comes from `AGENCY_TEST_RUNTIME`. Gated by their own environment variables. |

Outside this directory: `tests/test_agsandbox.py` (facade tests over a mock
backend, plus GPU/CPU/memory detection and the resource pool),
`tests/test_sandbox_naming.py` (container naming, run id) and
`tests/harness/test_executable_container.py` (harness executables in a container,
docker and podman).

## Fixtures and capability gates (`conftest.py`)

| Fixture | Values | Skips? |
| --- | --- | --- |
| `sandbox_kind` | chroot, docker, podman | yes, when that backend is unusable on the host |
| `container_kind` | docker, podman | yes, likewise |
| `container_runtime` | docker, podman | never: for mock tests that never touch a daemon |
| `backend_kind` | chroot, docker, podman | never: for mock tests of base-class logic |

Helpers: `make_sandbox(kind)` (a real `agSandbox` forced onto a backend),
`make_backend(kind)` (a bare backend object, no daemon probe, for mocks),
`backend_available(kind)` (cached live probe), `checkpoint_exists(kind, tag)`,
`started_run_cmd(backend)` and `make_cow_backend(runtime)`.

A tier-3 test whose contract only some backends can honour calls
`requires(kind, "<capability>")`. The capability table (`BACKEND_CAPABILITIES`)
records what each backend can do (own network namespace, cgroup limits,
`checkpoint()` handles, environment isolation...), so a gap shows up as a skip
with a reason instead of the test being dropped or written for one backend only.

Every test must **collect** with docker and podman absent: availability is
decided at run time (fixture params carry a `skipif`), never at import time in a
way that hides a test.

## Running

```bash
uv run --locked --extra dev pytest tests/sandbox -q            # everything usable here
uv run --locked --extra dev pytest tests/sandbox -q -k chroot  # one backend
uv run --locked --extra dev pytest tests/sandbox -q -k docker
uv run --locked --extra dev pytest tests/sandbox -q -k podman
uv run --locked --extra dev pytest tests/sandbox/test_all_backends.py -q -rs   # show why anything skipped
uv run --locked --extra dev pytest --collect-only -q tests/sandbox             # collection needs no runtime
```

`-k docker` also selects the docker half of the container-family tests (their ids
end in `[docker]`) and the docker-only files. Real-daemon tests skip themselves
when the daemon is unreachable; CI installs all three backends and fails loudly
if one is unusable (see `.github/workflows/tests.yml`).
