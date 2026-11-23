"""The set of basic tools every sandbox starts with, regardless of backend.

Container backends build their prepared image from this list; the chroot
backend builds its filesystem template from it. Keeping the lists here means
both kinds of sandbox offer the same tooling, and a harness daemon never
needs to pip-install its own Python packages at startup (they are baked in
ahead of time, so the daemon also works when the sandbox has no network).
"""

from __future__ import annotations

# System packages installed with the base system's own package manager
# (apt for the Debian-family images and filesystem templates in use).
SANDBOX_SYSTEM_PACKAGES = (
    "ca-certificates",
    "curl",
    "git",
    "procps",
    "libseccomp2",
    "libstdc++6",
)

# Python packages baked into every sandbox, matching what the harness
# daemon would otherwise pip-install at startup (see
# agency/harness/daemon.py's required-package list).
SANDBOX_PYTHON_PACKAGES = (
    "fastapi",
    "uvicorn",
    "openai",
    "httpx",
    "mcp",
    "pyseccomp",
    "cloudpickle",
    "pyte",
)

__all__ = ["SANDBOX_SYSTEM_PACKAGES", "SANDBOX_PYTHON_PACKAGES"]
