"""Sandboxed tool execution.

The agent's shell/file tools are the dangerous surface — left unsandboxed they
run on the host. This routes them through a `Sandbox` backend instead. The strong
backend is a libkrun **microVM** (own kernel, hardware isolation) via microsandbox:
the project workdir is bind-mounted read-write at /workspace (so the agent can edit
the real repo) while everything else on the host is invisible to the VM. Verified:
a write inside the VM lands on the host workdir; a host file outside it is unreadable.

Backends:
  host      runs on the host (current behavior, the explicit opt-out)
  microvm   libkrun microVM via microsandbox (Linux+KVM / macOS Apple Silicon)

`make_sandbox` is **fail-closed**: asking for `microvm` when it isn't available
raises rather than silently dropping to the host — a security feature shouldn't
degrade into running unsandboxed without you knowing.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

VM_ROOT = "/workspace"  # where the workdir is mounted inside the microVM
DEFAULT_IMAGE = "python"  # debian-based: has /bin/bash + coreutils


class SandboxUnavailable(RuntimeError):
    """Raised (fail-closed) when a requested sandbox backend can't be provided."""


class Sandbox(ABC):
    kind: str = "host"

    def __init__(self, workdir: str | os.PathLike):
        self.workdir = os.path.abspath(workdir)

    @abstractmethod
    async def exec(self, command: str, timeout: int = 30) -> tuple[str, int]:
        """Run a shell command; return (combined_output, exit_code)."""

    # File ops — confined to the workdir root. Subclasses implement these so the
    # whole tool surface (not just bash) stays inside the sandbox.
    async def read_file(self, path: str, max_bytes: int = 65536) -> str:
        raise NotImplementedError

    async def write_file(self, path: str, content: str) -> str:
        raise NotImplementedError

    async def list_dir(self, path: str = ".") -> str:
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release any backend resources (stop the microVM, etc.)."""

    async def __aenter__(self) -> "Sandbox":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()


class HostSandbox(Sandbox):
    """Runs commands directly on the host — no isolation. The explicit opt-out."""

    kind = "host"

    async def exec(self, command: str, timeout: int = 30) -> tuple[str, int]:
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-lc",
            command,
            cwd=self.workdir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return f"error: command timed out after {timeout}s", 124
        return out.decode("utf-8", "replace"), proc.returncode or 0

    def _host_path(self, path: str) -> Path:
        root = Path(self.workdir).resolve()
        p = (root / path).resolve()
        if p != root and root not in p.parents:
            raise ValueError(f"path escapes the sandbox workdir: {path}")
        return p

    async def read_file(self, path: str, max_bytes: int = 65536) -> str:
        return self._host_path(path).read_bytes()[:max_bytes].decode("utf-8", "replace")

    async def write_file(self, path: str, content: str) -> str:
        p = self._host_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {len(content)} chars to {path}"

    async def list_dir(self, path: str = ".") -> str:
        return "\n".join(sorted(os.listdir(self._host_path(path)))) or "[empty]"


class MicrosandboxSandbox(Sandbox):
    """Runs commands inside a libkrun microVM with the workdir bind-mounted at
    /workspace. The VM is created lazily and reused for the session, so state
    persists across calls; `aclose()` stops it."""

    kind = "microvm"

    def __init__(
        self,
        workdir: str | os.PathLike,
        *,
        image: str = DEFAULT_IMAGE,
        cpus: int = 1,
        memory: int = 1024,
    ):
        super().__init__(workdir)
        self.image, self.cpus, self.memory = image, cpus, memory
        self._sb = None
        # a stable, valid name derived from the workdir (≤128 bytes)
        digest = hashlib.sha1(self.workdir.encode()).hexdigest()[:12]
        self._name = f"harness-{digest}"

    async def _ensure(self):
        if self._sb is None:
            from microsandbox import Sandbox as MS, Volume

            try:
                await MS.remove(self._name)  # clear a stale sandbox of the same name
            except Exception:
                pass
            self._sb = await MS.create(
                self._name,
                image=self.image,
                cpus=self.cpus,
                memory=self.memory,
                volumes={VM_ROOT: Volume.bind(self.workdir)},
            )
        return self._sb

    async def exec(self, command: str, timeout: int = 30) -> tuple[str, int]:
        try:
            sb = await self._ensure()
            r = await sb.exec(
                "/bin/bash", ["-lc", command], cwd=VM_ROOT, timeout=timeout
            )
        except Exception as e:  # surface the real reason; the turn shouldn't crash
            return f"error: sandbox exec failed: {type(e).__name__}: {e}", 1
        out = r.stdout_text
        if r.stderr_text:
            out = f"{out}\n{r.stderr_text}" if out else r.stderr_text
        return out, r.exit_code

    def _vm_path(self, path: str) -> str:
        import posixpath

        p = posixpath.normpath(posixpath.join(VM_ROOT, path))
        if p != VM_ROOT and not p.startswith(VM_ROOT + "/"):
            raise ValueError(f"path escapes the sandbox workdir: {path}")
        return p

    async def read_file(self, path: str, max_bytes: int = 65536) -> str:
        sb = await self._ensure()
        return (await sb.fs.read_text(self._vm_path(path)))[:max_bytes]

    async def write_file(self, path: str, content: str) -> str:
        import posixpath

        sb = await self._ensure()
        vmp = self._vm_path(path)
        parent = posixpath.dirname(vmp)
        if parent and parent != VM_ROOT:
            try:
                await sb.fs.mkdir(parent)
            except Exception:
                pass
        await sb.fs.write(vmp, content.encode())
        return f"wrote {len(content)} chars to {path}"

    async def list_dir(self, path: str = ".") -> str:
        import posixpath

        sb = await self._ensure()
        entries = await sb.fs.list(self._vm_path(path))
        return (
            "\n".join(sorted(posixpath.basename(e.path) for e in entries)) or "[empty]"
        )

    async def aclose(self) -> None:
        if self._sb is not None:
            try:
                await self._sb.stop()
            except Exception:
                pass
            self._sb = None


def microvm_ready() -> tuple[bool, str]:
    """(ready, reason-if-not) for the microVM backend on this host."""
    if os.name == "posix" and not Path("/dev/kvm").exists() and not _is_apple_silicon():
        return False, "no /dev/kvm (enable virtualization) and not macOS Apple Silicon"
    if (
        shutil.which("msb") is None
        and not Path.home().joinpath(".microsandbox/bin/msb").exists()
    ):
        return False, "msb runtime not installed — run: lo sandbox install"
    try:
        import microsandbox  # noqa: F401
    except Exception:
        return False, "microsandbox SDK not installed — run: lo sandbox install"
    return True, ""


def _is_apple_silicon() -> bool:
    import platform

    return platform.system() == "Darwin" and platform.machine() == "arm64"


def make_sandbox(
    kind: str | None, workdir: str | os.PathLike, *, fail_closed: bool = True, **kw
) -> Sandbox:
    """Build a sandbox. `kind` in {None|'host', 'microvm'/'microsandbox'}.

    Fail-closed: requesting microvm when it's unavailable raises SandboxUnavailable
    rather than silently running on the host (set fail_closed=False to downgrade
    with a warning instead)."""
    if kind in (None, "", "host"):
        return HostSandbox(workdir)
    if kind in ("microvm", "microsandbox", "vm"):
        ready, why = microvm_ready()
        if ready:
            return MicrosandboxSandbox(workdir, **kw)
        if fail_closed:
            raise SandboxUnavailable(
                f"microVM sandbox requested but unavailable: {why}"
            )
        warnings.warn(
            f"microVM sandbox unavailable ({why}); running UNSANDBOXED on host"
        )
        return HostSandbox(workdir)
    raise ValueError(f"unknown sandbox kind: {kind!r}")
