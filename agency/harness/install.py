"""Install the pinned harness CLIs into Agency's own directory.

``agency install-harnesses`` downloads the versions of ``[tool.agency.harnesses]``
(``harness/pins.py``) into ``<harness_home()>/<harness>-<version>``, checks them
against the digest their publisher lists where there is one, and runs
``--version`` on the result. No Node.js and no shell installer is involved: each
harness is a self-contained Linux x86-64 executable, fetched from the same place
as its vendor's own installer fetches it.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from .pins import (
    BINARY_RELPATH,
    harness_home,
    harness_pins,
    install_dir,
    managed_binary,
    reports_version,
)

_NPM = "https://registry.npmjs.org"
_KIMI = "https://code.kimi.com/kimi-code/binaries"
_GROK = "https://x.ai/cli"
_ANTIGRAVITY = "https://storage.googleapis.com/antigravity-public/antigravity-cli"
# Antigravity's release manifest
# (https://antigravity-cli-auto-updater-974169037036.us-central1.run.app/manifests/linux_amd64.json)
# names only the latest build, so each pinned version records its build path
# and the SHA-512 that manifest published for it.
_ANTIGRAVITY_BUILDS = {
    "1.5.9": (
        "1.5.9-4515810895593472",
        "d01eb13ba7750eed18907f1ae8860864371e0ad9064415e8d033d22b4bd7b9b4"
        "e4c7f962058520b452d67db1c33163d9a889d55a7dffa159596ec6cd91a87b79",
    ),
}
_CHUNK = 1 << 20


def _fetch(url: str, destination: Path, digest=None) -> None:
    """Stream *url* to *destination*; *digest* is a hashlib object to feed."""
    with urllib.request.urlopen(url, timeout=60) as response, open(destination, "wb") as out:
        while chunk := response.read(_CHUNK):
            out.write(chunk)
            if digest is not None:
                digest.update(chunk)


def _fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def _extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive) as tar:
        tar.extractall(destination, filter="data")


def _npm(package: str, version: str, work: Path, destination: Path) -> None:
    """A package of the npm registry, checked against its published integrity."""
    metadata = _fetch_json(f"{_NPM}/{package.replace('/', '%2f')}/{version}")
    algorithm, expected = metadata["dist"]["integrity"].split("-", 1)
    digest = hashlib.new(algorithm)
    archive = work / "package.tgz"
    _fetch(metadata["dist"]["tarball"], archive, digest)
    if base64.b64encode(digest.digest()).decode() != expected:
        raise RuntimeError(f"{package}@{version}: the download does not match its integrity")
    _extract(archive, destination)


def _opencode_package() -> str:
    # OpenCode's own installer picks the baseline build for a CPU without AVX2.
    try:
        avx2 = " avx2" in Path("/proc/cpuinfo").read_text()
    except OSError:
        avx2 = True
    return "opencode-linux-x64" if avx2 else "opencode-linux-x64-baseline"


def _install_claude_code(version: str, work: Path, destination: Path) -> None:
    _npm("@anthropic-ai/claude-code-linux-x64", version, work, destination)


def _install_codex(version: str, work: Path, destination: Path) -> None:
    _npm("@openai/codex", f"{version}-linux-x64", work, destination)


def _install_opencode(version: str, work: Path, destination: Path) -> None:
    _npm(_opencode_package(), version, work, destination)


def _install_kimi(version: str, work: Path, destination: Path) -> None:
    manifest = _fetch_json(f"{_KIMI}/{version}/manifest.json")
    expected = manifest["platforms"]["linux-x64"]["checksum"]
    archive = work / "kimi.tar.gz"
    _fetch(f"{_KIMI}/{version}/kimi-code-linux-x64.tar.gz", archive)
    _extract(archive, destination)
    # The manifest's checksum is that of the executable, not of the archive.
    if hashlib.sha256((destination / BINARY_RELPATH["kimi"]).read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"kimi {version}: the download does not match its manifest")


def _install_grok(version: str, work: Path, destination: Path) -> None:
    # Grok publishes no digest; the transport is HTTPS, as in its own installer.
    archive = work / "grok.gz"
    _fetch(f"{_GROK}/grok-{version}-linux-x86_64.gz", archive)
    binary = destination / BINARY_RELPATH["grok"]
    with gzip.open(archive) as source, open(binary, "wb") as out:
        shutil.copyfileobj(source, out)
    binary.chmod(0o755)


def _install_antigravity(version: str, work: Path, destination: Path) -> None:
    if version not in _ANTIGRAVITY_BUILDS:
        raise RuntimeError(
            f"antigravity {version}: no recorded build; add it to _ANTIGRAVITY_BUILDS"
        )
    build, expected = _ANTIGRAVITY_BUILDS[version]
    digest = hashlib.sha512()
    archive = work / "agy.tar.gz"
    _fetch(f"{_ANTIGRAVITY}/{build}/linux-x64/cli_linux_x64.tar.gz", archive, digest)
    if digest.hexdigest() != expected:
        raise RuntimeError(f"antigravity {version}: the download does not match its manifest")
    with tarfile.open(archive) as tar:
        tar.extract("antigravity", destination, filter="data")


_INSTALLERS = {
    "antigravity": _install_antigravity,
    "claude_code": _install_claude_code,
    "codex": _install_codex,
    "grok": _install_grok,
    "kimi": _install_kimi,
    "opencode": _install_opencode,
}


def install_harness(harness: str, home: "Path | None" = None, *, force: bool = False) -> Path:
    """Install the pinned version of *harness*; returns its executable. An
    install that is already there, and reports its version, is left alone."""
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise RuntimeError("Agency's harness installer supports Linux x86-64 only")
    version = harness_pins()[harness]
    home = home or harness_home()
    if not force and (binary := managed_binary(harness, home)) is not None:
        if _reports_pinned_version(binary, version):
            return binary
    home.mkdir(parents=True, exist_ok=True)
    destination = install_dir(harness, home)
    with tempfile.TemporaryDirectory(prefix=f".{harness}-", dir=home) as work_dir:
        work = Path(work_dir)
        staged = work / "install"
        staged.mkdir()
        _INSTALLERS[harness](version, work, staged)
        binary = staged / BINARY_RELPATH[harness]
        if not binary.is_file():
            raise RuntimeError(
                f"{harness} {version}: {BINARY_RELPATH[harness]} is not in the download"
            )
        binary.chmod(binary.stat().st_mode | 0o755)
        if not _reports_pinned_version(binary, version):
            raise RuntimeError(f"{harness} {version}: the installed binary reports another version")
        shutil.rmtree(destination, ignore_errors=True)
        os.replace(staged, destination)
    return destination / BINARY_RELPATH[harness]


def _reports_pinned_version(binary: Path, version: str) -> bool:
    try:
        run = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return run.returncode == 0 and reports_version(run.stdout + run.stderr, version)


def install_harnesses(
    names: "list[str] | None" = None, home: "Path | None" = None, *, force: bool = False
) -> "dict[str, Path]":
    """Install *names* (default: every pinned harness)."""
    pins = harness_pins()
    names = list(names or pins)
    unknown = [name for name in names if name not in _INSTALLERS or name not in pins]
    if unknown:
        raise ValueError(f"no pinned harness named {', '.join(unknown)}; known: {', '.join(pins)}")
    installed = {}
    for name in names:
        print(f"{name} {pins[name]} ...", file=sys.stderr, flush=True)
        installed[name] = install_harness(name, home, force=force)
        print(f"  {installed[name]}", file=sys.stderr, flush=True)
    return installed
