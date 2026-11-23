"""Builds and caches the shared chroot filesystem template.

One complete minimal Debian root filesystem per host, created once and
reused by every chroot sandbox: each jail starts as a copy of it instead of
a read-only bind-mount view of the host. This is what makes the chroot a
full sandbox with its own system directories -- matching what container
sandboxes get from their base image, including the same basic tool set
(see agency/sandbox/sandbox_tools.py).

The template lives under the process's temp root (alongside chroot jail
state) and is keyed by OS release, tool-set content digest, and ownership
uid, so a changed tool set or a different owner rebuilds rather than
reusing a stale or inaccessible template.
"""

from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from ..utils.agutil import agency_tmp_dir
from .sandbox_tools import SANDBOX_PYTHON_PACKAGES, SANDBOX_SYSTEM_PACKAGES

_template_lock = threading.Lock()
_template_root_cache: "Path | None" = None

# Shell script installed as /usr/local/bin/python inside the template so
# "python" (unversioned) works exactly as it does in container images.
_PYTHON_WRAPPER = """#!/bin/sh
exec /usr/bin/python3 "$@"
"""


def _templates_dir() -> Path:
    # Keyed under the user's temp root directly (not a per-run directory):
    # the template is a machine-level cache meant to survive across runs,
    # like a container image -- rebuilding ~200MB per process would defeat
    # its purpose.
    return agency_tmp_dir() / "chroot_filesystem_templates"


def _template_marker_name() -> str:
    """Filename that stamps a finished template with everything that makes
    it this host's current template: OS release, tool-set content, and the
    owning uid (jail directories are copied per-jail and must stay readable
    by the sandbox process)."""
    try:
        release = (
            subprocess.run(
                ["sh", "-c", ". /etc/os-release && echo $ID$VERSION_ID"],
                capture_output=True,
                timeout=10,
                check=True,
            )
            .stdout.decode()
            .strip()
        )
    except Exception:
        release = "unknown"
    tool_content = ";".join(SANDBOX_SYSTEM_PACKAGES + SANDBOX_PYTHON_PACKAGES)
    digest = hashlib.sha256(f"{release};{tool_content}".encode()).hexdigest()[:16]
    return f".agency-chroot-template-ready-uid{os.getuid()}-{digest}"


def _sources_list_lines(os_release_id: str, version_codename: str) -> "list[str]":
    """Package-repository lines the template needs: the base repository plus
    the extra component that carries pip on Debian-family systems."""
    if os_release_id == "debian":
        return [
            f"deb http://deb.debian.org/debian {version_codename} main",
            f"deb http://deb.debian.org/debian {version_codename}-updates main",
        ]
    return [
        f"deb http://archive.ubuntu.com/ubuntu {version_codename} main universe",
        f"deb http://archive.ubuntu.com/ubuntu {version_codename}-updates main universe",
    ]


def _release_field(field: str) -> str:
    try:
        return (
            subprocess.run(
                ["sh", "-c", f". /etc/os-release && echo ${field}"],
                capture_output=True,
                timeout=10,
                check=True,
            )
            .stdout.decode()
            .strip()
        )
    except Exception:
        return ""


def _warn(message: str) -> None:
    print(f"[agsandbox_backend] WARNING: {message}", file=sys.stderr, flush=True)


def _progress(message: str) -> None:
    """User-visible progress line for long build steps, so a minutes-long
    one-time build never looks like a hang."""
    print(f"[agsandbox] {message}", file=sys.stderr, flush=True)


def chroot_template_dir(timeout_s: float = 1800) -> "Path | None":
    """The shared filesystem template directory, building it first if needed.

    Returns None when the template cannot be built on this host (no
    debootstrap, or no passwordless sudo to run it as a non-root process);
    callers then fail the sandbox start with a clear message rather than
    silently degrading to a broken jail. The build runs debootstrap under
    sudo when the calling process itself is not root -- the one privilege
    escalation the chroot backend needs, and only during the one-time
    template build.
    """
    global _template_root_cache
    with _template_lock:
        if _template_root_cache is not None:
            return _template_root_cache
        existing = _existing_template_dir()
        if existing is not None:
            _template_root_cache = existing
            return existing
        if shutil.which("debootstrap") is None:
            return None
        built = _build_template_dir(timeout_s)
        if built is None:
            return None
        _template_root_cache = built
        return built


def _existing_template_dir() -> "Path | None":
    """The finished template directory if one exists for this host and
    owner, else None."""
    candidate = _templates_dir() / "root"
    if (candidate / _template_marker_name()).is_file():
        return candidate
    return None


def _build_template_dir(timeout_s: float) -> "Path | None":
    """Build the template into a staging directory and publish it under the
    final name; None (with the staging directory removed) on any failure."""
    marker = _template_marker_name()
    templates_dir = _templates_dir()
    templates_dir.mkdir(parents=True, exist_ok=True)
    candidate = templates_dir / "root"
    staging = templates_dir / "staging"
    shutil.rmtree(staging, ignore_errors=True)

    release_id = _release_field("ID") or "ubuntu"
    codename = _release_field("VERSION_CODENAME") or "noble"
    prefix = [] if os.geteuid() == 0 else ["sudo", "-n"]

    _progress(
        f"First chroot sandbox on this host: building the shared filesystem "
        f"template ({codename} base system; one-time, a few minutes, cached "
        f"for all later sandboxes)..."
    )
    bootstrap = subprocess.run(
        [
            *prefix,
            "debootstrap",
            "--variant=minbase",
            "--include=python3",
            codename,
            str(staging),
        ],
        capture_output=True,
        timeout=timeout_s,
    )
    if bootstrap.returncode != 0 or not (staging / "usr" / "bin").is_dir():
        _warn(
            f"chroot filesystem template build failed (exit {bootstrap.returncode}): "
            f"{bootstrap.stderr.decode('utf-8', errors='replace')[-2000:]}"
        )
        shutil.rmtree(staging, ignore_errors=True)
        return None

    sources_lines = "\n".join(_sources_list_lines(release_id, codename))
    _progress("Base system downloaded; installing the sandbox tool set...")
    package_install_script = (
        "set -e; "
        f"printf '%s\\n' {shlex.quote(sources_lines)} > /etc/apt/sources.list; "
        "apt-get update -qq; "
        "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
        "--no-install-recommends python3-pip " + " ".join(SANDBOX_SYSTEM_PACKAGES) + "; "
        "rm -rf /var/lib/apt/lists/*; "
        "python3 -m pip install --quiet --no-cache-dir --break-system-packages "
        + " ".join(SANDBOX_PYTHON_PACKAGES)
    )
    configure = subprocess.run(
        [
            *prefix,
            "chroot",
            str(staging),
            "/bin/bash",
            "-c",
            package_install_script,
        ],
        capture_output=True,
        timeout=timeout_s,
    )
    if configure.returncode != 0:
        _warn(
            "sandbox tool installation into the chroot filesystem template failed "
            f"(exit {configure.returncode}): "
            f"{configure.stderr.decode('utf-8', errors='replace')[-2000:]}"
        )
        shutil.rmtree(staging, ignore_errors=True)
        return None

    try:
        python_wrapper = staging / "usr" / "local" / "bin" / "python"
        if os.geteuid() != 0:
            # debootstrap created the staging tree as root; hand it to the
            # sandbox process before writing into it or copying it per-jail.
            # a+rX on everything (plus u+rw on files) so per-jail copies by
            # the sandbox user can read root-owned system files like
            # /etc/shadow, and device nodes in /dev stay reproducible.
            subprocess.run(
                [*prefix, "chown", "-R", str(os.getuid()), str(staging)],
                capture_output=True,
                timeout=timeout_s,
                check=True,
            )
            subprocess.run(
                ["chmod", "-R", "u+rwX,a+rX", str(staging)],
                capture_output=True,
                timeout=timeout_s,
                check=True,
            )
        # The base system install populates /dev with device files; a
        # per-jail copy cannot recreate those unprivileged, and the jail
        # setup binds the host's own device files in at every start
        # anyway -- ship the template with an empty /dev.
        dev_dir = staging / "dev"
        if dev_dir.is_dir():
            for dev_entry in dev_dir.iterdir():
                if dev_entry.is_dir() and not dev_entry.is_symlink():
                    shutil.rmtree(dev_entry, ignore_errors=True)
                else:
                    dev_entry.unlink(missing_ok=True)
        python_wrapper.parent.mkdir(parents=True, exist_ok=True)
        python_wrapper.write_text(_PYTHON_WRAPPER)
        python_wrapper.chmod(0o755)
        (staging / marker).write_text("ready\n")
        staging.rename(candidate)
    except Exception as e:
        _warn(f"could not finalize the chroot filesystem template: {e}")
        shutil.rmtree(staging, ignore_errors=True)
        return None

    _progress("Shared filesystem template ready.")
    return candidate


def chroot_template_available() -> bool:
    """True if the shared filesystem template exists or can be built on
    this host."""
    return chroot_template_dir() is not None


_build_thread: "threading.Thread | None" = None
_build_thread_lock = threading.Lock()


def start_chroot_template_build() -> None:
    """Kick the template build off in a background thread (once per
    process), so the minutes-long first build overlaps with other startup
    work instead of blocking the first jail start. `chroot_template_dir()`
    blocks on the same lock if it is still running then. Uses
    `agprof.spawn_traced()` (not a bare Thread) so the thread is visible
    to the profiler's thread tracking."""
    global _build_thread
    with _build_thread_lock:
        if _build_thread is not None and _build_thread.is_alive():
            return
        from ..observability.profiler import agprof

        _build_thread = agprof.spawn_traced(chroot_template_dir, daemon=True)
        _build_thread.start()


def reset_chroot_template_cache_for_tests() -> None:
    global _template_root_cache
    with _template_lock:
        _template_root_cache = None
