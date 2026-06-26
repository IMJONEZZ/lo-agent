#!/usr/bin/env bash
# Install the microVM sandbox that ships with local_harness.
#
# microsandbox runs each tool call in a libkrun microVM with its OWN kernel
# (hardware isolation, not a shared-kernel container), so the agent's bash/file
# tools can't touch the host. This script installs the `msb` runtime + the Python
# SDK into the harness, then proves it with a real microVM smoke test.
#
#   bash scripts/install-sandbox.sh
#
# Idempotent: re-running it skips anything already in place. It is intentionally
# the only place that runs microsandbox's remote installer, so the network step
# is auditable and gated behind prerequisite checks. Needs Linux+KVM (or macOS
# Apple Silicon). See docs.microsandbox.dev.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALLER_URL="https://install.microsandbox.dev"

say()  { printf '\033[1;34m▸ %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m✓ %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m! %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# --- 1. prerequisites -------------------------------------------------------
say "Checking prerequisites for microVM isolation"
OS="$(uname -s)"
case "$OS" in
  Linux)
    [ -e /dev/kvm ] || die "/dev/kvm missing — enable KVM/virtualization (VT-x/AMD-V) in BIOS, or this box can't run microVMs."
    if [ ! -r /dev/kvm ] || [ ! -w /dev/kvm ]; then
      warn "You lack read/write on /dev/kvm. Add yourself to the 'kvm' group:"
      warn "    sudo usermod -aG kvm \"$USER\"   # then log out/in"
    fi
    ok "Linux + KVM present" ;;
  Darwin)
    [ "$(uname -m)" = "arm64" ] || die "microsandbox on macOS needs Apple Silicon (arm64)."
    ok "macOS Apple Silicon" ;;
  *) die "Unsupported OS: $OS (need Linux+KVM or macOS Apple Silicon)" ;;
esac
command -v curl >/dev/null 2>&1 || die "curl is required."

# --- 2. install the msb runtime --------------------------------------------
ensure_path() { case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) export PATH="$HOME/.local/bin:$PATH" ;; esac; }
ensure_path
if command -v msb >/dev/null 2>&1; then
  ok "msb already installed ($(msb --version 2>/dev/null | head -n1 || echo present))"
else
  say "Installing the msb runtime from $INSTALLER_URL"
  warn "This downloads and runs microsandbox's official installer. Review it first if you like:"
  warn "    curl -fsSL $INSTALLER_URL | less"
  curl -fsSL "$INSTALLER_URL" | sh
  ensure_path
  command -v msb >/dev/null 2>&1 || die "msb not on PATH after install — add \$HOME/.local/bin to PATH and re-run."
  ok "msb installed"
fi

# --- 3. install the Python SDK into the harness ----------------------------
# Additive on purpose: `uv sync --extra sandbox` would PRUNE other extras (native
# torch/transformers/peft, dspy) out of the env. `uv pip install` adds the SDK
# without disturbing what's already there. The `sandbox` extra is still declared
# in pyproject for `uv sync --all-extras`.
say "Installing the microsandbox Python SDK into the harness (additive)"
if command -v uv >/dev/null 2>&1; then
  (cd "$REPO_ROOT" && uv pip install microsandbox) || die "uv pip install microsandbox failed."
  ok "SDK installed into the project env (other extras left intact)"
else
  warn "uv not found — install the SDK however you manage Python: pip install microsandbox"
fi

# --- 4. pull a base image ---------------------------------------------------
say "Pulling a base microVM image (one-time)"
msb pull microsandbox/python 2>/dev/null || msb pull python 2>/dev/null || \
  warn "Could not pre-pull an image; it will be pulled on first sandbox run."

# --- 5. smoke test: run code INSIDE a microVM ------------------------------
say "Smoke test: executing inside a microVM (must NOT touch the host)"
SMOKE="$(msb exec --image python -- python3 -c 'print("microvm-ok")' 2>/dev/null || true)"
if printf '%s' "$SMOKE" | grep -q "microvm-ok"; then
  ok "microVM executed code in isolation — sandbox is ready"
else
  warn "CLI smoke test inconclusive (CLI flags vary by msb version)."
  warn "Verifying via the Python SDK instead…"
  if command -v uv >/dev/null 2>&1 && (cd "$REPO_ROOT" && uv run python - <<'PY'
import asyncio
from microsandbox import Sandbox
async def main():
    sb = await Sandbox.create("harness-smoke", image="python", cpus=1, memory=512)
    out = await sb.exec("python3", ["-c", "print('microvm-ok')"])
    assert "microvm-ok" in out.stdout_text, out.stdout_text
    await sb.stop()
asyncio.run(main())
print("SDK_SMOKE_OK")
PY
  ); then
    ok "Python SDK ran code in a microVM — sandbox is ready"
  else
    die "Sandbox installed but the smoke test failed. Check 'msb --version' and docs.microsandbox.dev."
  fi
fi

ok "Done. Run the harness sandboxed with:  harness run --sandbox microvm  (once the backend lands)"
