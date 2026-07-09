#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/Users/keynary/Code/xiandaobei/meta-main}"
RUN_ID="${RUN_ID:-r3.1b-fuseddefault-3bucket-$(date '+%Y%m%d-%H%M')}"

cd "$ROOT"
FUSED_MAX_SEQ_LEN=0 RUN_ID="$RUN_ID" \
  exec experiments/r3.1b-fused-ab-20260709/run_threshold16384_guard.sh
