#!/usr/bin/env bash
# Lo: the harness for local models — installer
#
#   curl -fsSL https://raw.githubusercontent.com/IMJONEZZ/lo-agent/main/install.sh | bash
#
# Installs the `lo` CLI with uv (bootstrapping uv itself if needed). No sudo,
# nothing outside ~/.local and uv's own directories. Safe to re-run; it
# upgrades in place.
#
# Uninstall (removes lo however it was installed; your data stays put):
#
#   curl -fsSL https://raw.githubusercontent.com/IMJONEZZ/lo-agent/main/install.sh | bash -s -- --uninstall

set -euo pipefail

REPO="https://github.com/IMJONEZZ/lo-agent"
LOG="$(mktemp -t lo-install-XXXXXX.log)"

# ── palette (only when stdout is a real terminal) ────────────────────────────
if [ -t 1 ] && command -v tput >/dev/null 2>&1 && [ "$(tput colors 2>/dev/null || echo 0)" -ge 8 ]; then
  TTY=1
  JADE="$(tput setaf 6)" GOLD="$(tput setaf 3)" DIM="$(tput dim)"
  BOLD="$(tput bold)" OK="$(tput setaf 2)" RED="$(tput setaf 1)" R="$(tput sgr0)"
else
  TTY=0
  JADE="" GOLD="" DIM="" BOLD="" OK="" RED="" R=""
fi

say()  { printf '%s\n' "$1"; }
fail() { printf '%s✗ %s%s\n' "$RED" "$1" "$R" >&2
         printf '%s  install log: %s%s\n' "$DIM" "$LOG" "$R" >&2
         tail -n 8 "$LOG" >&2 || true
         exit 1; }

banner() {
  printf '\n%s' "$JADE"
  cat <<'EOF'
   ██╗      ██████╗
   ██║     ██╔═══██╗
   ██║     ██║   ██║
   ██║     ██║   ██║
   ███████╗╚██████╔╝
   ╚══════╝ ╚═════╝
EOF
  printf '%s   %sthe harness for local models%s\n\n' "$R" "$BOLD" "$R"
}

# spin <pid> <message…> — a spinner that rotates through harness-flavored
# status lines while <pid> works. Plain single line when not a TTY.
spin() {
  local pid=$1; shift
  local msgs=("$@") frames='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏' i=0 m=0
  if [ "$TTY" -eq 0 ]; then
    printf '… %s\n' "${msgs[0]}"
    wait "$pid"; return $?
  fi
  while kill -0 "$pid" 2>/dev/null; do
    printf '\r%s%s%s %s%-42s%s' "$JADE" "${frames:i%10:1}" "$R" "$DIM" "${msgs[m]}" "$R"
    i=$((i + 1))
    [ $((i % 24)) -eq 0 ] && m=$(((m + 1) % ${#msgs[@]}))
    sleep 0.08
  done
  printf '\r%-50s\r' ''
  wait "$pid"
}

step_done() { printf '%s✓%s %s\n' "$OK" "$R" "$1"; }

# tier_unlock — the capability ladder, climbed one rung at a time.
tier_unlock() {
  local tiers=(
    "tier 0  agent loop · event log · validate-and-retry"
    "tier 1  bit-identical replay · logprob signals"
    "tier 2  grammar skills · sampler zoo · think budgets"
    "tier 3  KV tree search · fork & backtrack"
  )
  printf '\n%s%s%s\n' "$BOLD" "  unlocked with any OpenAI-compatible server:" "$R"
  for t in "${tiers[@]}"; do
    if [ "$TTY" -eq 1 ]; then
      printf '  %s🔒%s %s' "$DIM" "$R" "$t"; sleep 0.22
      printf '\r  %s🔓%s %s%s%s\n' "$GOLD" "$R" "$DIM" "$t" "$R"
    else
      printf '  🔓 %s\n' "$t"
    fi
  done
}

# tier_lock — the ladder, climbed back down.
tier_lock() {
  local tiers=(
    "tier 3  KV tree search · fork & backtrack"
    "tier 2  grammar skills · sampler zoo · think budgets"
    "tier 1  bit-identical replay · logprob signals"
    "tier 0  agent loop · event log · validate-and-retry"
  )
  printf '\n'
  for t in "${tiers[@]}"; do
    if [ "$TTY" -eq 1 ]; then
      printf '  %s🔓%s %s' "$GOLD" "$R" "$t"; sleep 0.18
      printf '\r  %s🔒%s %s%s%s\n' "$DIM" "$R" "$DIM" "$t" "$R"
    else
      printf '  🔒 %s\n' "$t"
    fi
  done
}

uninstall() {
  banner
  local removed=0
  if command -v uv >/dev/null 2>&1 && uv tool list 2>/dev/null | grep -q '^lo-agent '; then
    uv tool uninstall lo-agent >>"$LOG" 2>&1 &
    spin $! "unhooking the harness (uv)…" || fail "uv tool uninstall failed"
    step_done "uv tool removed"
    removed=1
  fi
  if command -v brew >/dev/null 2>&1 && brew list lo-agent >/dev/null 2>&1; then
    brew uninstall lo-agent >>"$LOG" 2>&1 &
    spin $! "unhooking the harness (brew)…" || fail "brew uninstall failed"
    step_done "brew formula removed  (drop the tap too: brew untap IMJONEZZ/lo-agent)"
    removed=1
  fi
  if [ "$removed" -eq 0 ]; then
    say "nothing to do — no uv-tool or brew install of lo found."
    say "${DIM}(a from-source checkout is just its directory — delete the clone)${R}"
    rm -f "$LOG"; exit 0
  fi
  tier_lock
  printf '\n%syour data was left alone:%s\n' "$BOLD" "$R"
  printf '  %s~/.lo/%s            config + memory — remove with: rm -rf ~/.lo\n' "$GOLD" "$R"
  printf '  %slo.db%s             per-project event logs, next to wherever you ran lo\n' "$GOLD" "$R"
  printf '\n%sthe models were local all along. so long. 🦙%s\n\n' "$DIM" "$R"
  rm -f "$LOG"
  exit 0
}

case "${1:-}" in
  --uninstall|uninstall) uninstall ;;
  "") ;;
  *) say "usage: install.sh [--uninstall]"; exit 2 ;;
esac

banner

# ── 1. uv (bootstrap if missing) ─────────────────────────────────────────────
if command -v uv >/dev/null 2>&1; then
  step_done "uv $(uv --version 2>/dev/null | awk '{print $2}') found"
else
  curl -fsSL https://astral.sh/uv/install.sh | sh >>"$LOG" 2>&1 &
  spin $! "fetching uv (the installer's installer)…" \
          "uv also brings its own Python — no system setup…" \
    || fail "couldn't install uv — see https://docs.astral.sh/uv/"
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || fail "uv installed but not on PATH yet — open a new shell and re-run"
  step_done "uv installed"
fi

# ── 2. lo itself ─────────────────────────────────────────────────────────────
uv tool install --force "git+${REPO}" >>"$LOG" 2>&1 &
spin $! "cloning the harness…" \
        "warming the KV cache…" \
        "teaching the sampler zoo new tricks…" \
        "convincing logits to sit still (determinism)…" \
        "banning the word 'tapestry' (anti-slop)…" \
        "event-sourcing everything, twice, identically…" \
  || fail "install failed"
step_done "lo installed"

# ── 3. verify ────────────────────────────────────────────────────────────────
LO_BIN="$(command -v lo || true)"
[ -z "$LO_BIN" ] && [ -x "$HOME/.local/bin/lo" ] && LO_BIN="$HOME/.local/bin/lo"
[ -n "$LO_BIN" ] || fail "installed, but 'lo' isn't on your PATH — try: uv tool update-shell"
VERSION="$("$LO_BIN" --version 2>/dev/null || true)"
step_done "${VERSION:-lo} responds"

tier_unlock

printf '\n%snext:%s\n' "$BOLD" "$R"
if ! command -v lo >/dev/null 2>&1; then
  printf '  %suv tool update-shell%s   # put ~/.local/bin on your PATH, then open a new shell\n' "$GOLD" "$R"
fi
printf '  %slo quickstart%s          # finds your model server (:8080 :8000 :1234 :11434) → TUI\n' "$GOLD" "$R"
printf '  %slo doctor%s              # if anything misbehaves — every ✗ comes with a fix\n' "$GOLD" "$R"
printf '\n%s%s%s\n\n' "$DIM" "docs: ${REPO}#readme" "$R"
rm -f "$LOG"
