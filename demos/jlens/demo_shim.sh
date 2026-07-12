#!/usr/bin/env bash
# Show-don't-tell: the LD_PRELOAD shim steers a STOCK llama.cpp binary's output.
# No fork, no sidecar — one env var interposes the user's own server's context
# creation and edits the residual stream live. You watch the token stream change.
set -u
GOLD='\033[1;38;5;220m'; GREEN='\033[38;5;42m'; DIM='\033[2m'; BOLD='\033[1m'; R='\033[0m'
STOCK=${STOCK:-/tmp/stock_gen}          # a stock llama.cpp client (sets NO callback)
MODEL=${MODEL:?set MODEL}
SHIM=${SHIM:-/tmp/lo_jlens_shim_cli.so}
STEER=${STEER:-/tmp/steer.bin}

sleep 0.6
printf "${DIM}\$ # a plain llama.cpp program — same one a stock llama-server is built from${R}\n"
sleep 0.8
printf "${BOLD}${GOLD}── baseline: no shim ──${R}\n\n"
sleep 0.5
printf "${DIM}\$ ${R}stock_gen model.gguf\n"
sleep 0.4
printf "  generated token ids:  ${BOLD}"; "$STOCK" "$MODEL" 2>/dev/null; printf "${R}\n"
sleep 1.4
printf "\n${BOLD}${GOLD}── live-steered: LD_PRELOAD the shim, point it at a steer file ──${R}\n\n"
sleep 0.5
printf "${DIM}\$ ${R}LD_PRELOAD=lo_jlens_shim.so LO_JLENS_STEER=steer.bin stock_gen model.gguf\n"
sleep 0.4
printf "  generated token ids:  ${BOLD}${GREEN}"
LD_PRELOAD="$SHIM" LO_JLENS_STEER="$STEER" "$STOCK" "$MODEL" 2>/dev/null
printf "${R}\n"
sleep 1.2
printf "\n${DIM}  same binary, same model, same prompt — the residual stream was edited${R}\n"
printf "${DIM}  mid-graph, so the tokens it actually sampled changed. One restart, one${R}\n"
printf "${DIM}  env var, your own server. ${R}${BOLD}${GOLD}No frontier API can do this.${R}\n\n"
sleep 1.5
