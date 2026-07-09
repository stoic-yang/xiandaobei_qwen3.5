#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/Users/keynary/Code/xiandaobei/meta-main}"
OVERLAY_SOURCE_DIR="${OVERLAY_SOURCE_DIR:-/public/home/xdzs2026_c166/vllm_cscc_codex_r31b_threshold_20260709_173633}"
RUN_ID="${RUN_ID:-r3.1b-threshold16384-3bucket-$(date '+%Y%m%d-%H%M')}"
SERVER_PORT="${SERVER_PORT:-18001}"
PERSIST_MODEL_DIR="${PERSIST_MODEL_DIR:-/public/home/xdzs2026_c166/Qwen3.5-27B}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-/root/.xdb_guard_no_local_model_${RUN_ID}}"
FUSED_MAX_SEQ_LEN="${FUSED_MAX_SEQ_LEN:-16384}"
LOG_DIR="$ROOT/experiments/$RUN_ID"
LOG="$LOG_DIR/local_guard.log"

cd "$ROOT"
mkdir -p "$LOG_DIR"
trap 'rc=$?; printf "%s\n" "$rc" > "$LOG_DIR/.exit"' EXIT

set +e
preflight_remote=$(ssh -F ~/.ssh/xiandaobei.generated.conf xiandaobei-worker-auto 'bash -s' -- "$PERSIST_MODEL_DIR" "$LOCAL_MODEL_DIR" <<'REMOTE'
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
PERSIST_MODEL_DIR="$1"
LOCAL_MODEL_DIR="$2"

busy_procs=$(ps -eo pid=,args= | grep -E 'guard_bench|bench serve|vllm serve|vllm.entrypoints|EngineCore|run_throughput|run_accuracy' | grep -v grep || true)
if [ -n "$busy_procs" ]; then
  printf 'BUSY_PROCESS\n%s\n' "$busy_procs"
  exit 20
fi

python3 - "$PERSIST_MODEL_DIR" "$LOCAL_MODEL_DIR" <<'PY'
import json
import os
import sys

persist, local = sys.argv[1:3]
with open(os.path.join(persist, "config.json")) as f:
    cfg = json.load(f)
text = cfg.get("text_config") or cfg
layers = text.get("num_hidden_layers")
hidden = text.get("hidden_size")
heads = text.get("num_attention_heads")
kv_heads = text.get("num_key_value_heads")
head_dim = text.get("head_dim")
print(f"PERSIST_MODEL_CHECK layers={layers} hidden={hidden} heads={heads} kv_heads={kv_heads} head_dim={head_dim}")
if (layers, hidden, heads, kv_heads, head_dim) != (64, 5120, 24, 4, 256):
    raise SystemExit(f"persistent model is not expected 27B config: {persist}")
local_cfg = os.path.join(local, "config.json")
if os.path.exists(local_cfg):
    raise SystemExit(f"LOCAL_MODEL_DIR unexpectedly exists and would shadow persistent model: {local}")
PY

hy_smi_out=$(PATH=/opt/hyhal/bin:$PATH hy-smi -a 2>/dev/null || true)
mem_use=$(printf '%s\n' "$hy_smi_out" | awk -F': ' '/HCU memory use \(%\)/ {print $2}' | head -n 1 | tr -dc '0-9')
mem_use=${mem_use:-0}
if [ "$mem_use" -gt 10 ]; then
  printf 'BUSY_HCU_MEMORY=%s\n' "$mem_use"
  exit 21
fi

printf 'CLEAN mem=%s\n' "$mem_use"
REMOTE
)
preflight_rc=$?
set -e
printf '%s\n' "$preflight_remote" | tee "$LOG_DIR/preflight.txt"
if [ "$preflight_rc" -ne 0 ]; then
  printf 'preflight failed rc=%s; not starting guard\n' "$preflight_rc" | tee -a "$LOG_DIR/preflight.txt"
  exit "$preflight_rc"
fi

python3 -m py_compile scripts/guard_bench.py

{
  printf 'started_at=%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
  printf 'run_id=%s\n' "$RUN_ID"
  printf 'overlay_source_dir=%s\n' "$OVERLAY_SOURCE_DIR"
  printf 'fused_max_seq_len=%s\n' "$FUSED_MAX_SEQ_LEN"
  python3 scripts/guard_bench.py \
    --run-id "$RUN_ID" \
    --repo competition \
    --model-dir "$PERSIST_MODEL_DIR" \
    --local-model-dir "$LOCAL_MODEL_DIR" \
    --locked-start-script \
    --load-format runai_streamer \
    --overlay-source-dir "$OVERLAY_SOURCE_DIR" \
    --env VLLM_CACHE_ROOT=/public/home/xdzs2026_c166/vllm_cache/vllm_cache \
    --env TRITON_CACHE_DIR=/public/home/xdzs2026_c166/vllm_cache/triton_cache \
    --env XDB_R31_FLASH_ATTN_PREFILL=1 \
    --env XDB_R31_FLASH_ATTN_MIN_Q=2 \
    --env VLLM_TRITON_FUSED_PREFILL=1 \
    --env VLLM_TRITON_FUSED_PREFILL_MAX_SEQ_LEN="$FUSED_MAX_SEQ_LEN" \
    --server-port "$SERVER_PORT" \
    --stop-existing \
    --server-start-timeout 1800 \
    --remote-timeout 14400 \
    --poll-interval 300 \
    --buckets 4-8K,8-16K,16-32K \
    --num-prompts 3 \
    --repetitions 3 \
    --accuracy none
} >"$LOG" 2>&1
