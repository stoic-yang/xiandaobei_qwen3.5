#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/keynary/Code/xiandaobei/meta-main"
RUN_ROOT="$ROOT/experiments/r3.1b-fused-ab-20260709"
LOG="$RUN_ROOT/run_ab.log"

cd "$ROOT"
mkdir -p "$RUN_ROOT"

ts() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

run_guard() {
  local run_id="$1"
  local fused="$2"

  echo "=== $(ts) start $run_id fused=$fused ===" | tee -a "$LOG"
  python3 scripts/guard_bench.py \
    --run-id "$run_id" \
    --repo competition \
    --locked-start-script \
    --load-format runai_streamer \
    --env VLLM_CACHE_ROOT=/public/home/xdzs2026_c166/vllm_cache/vllm_cache \
    --env TRITON_CACHE_DIR=/public/home/xdzs2026_c166/vllm_cache/triton_cache \
    --env XDB_R31_FLASH_ATTN_PREFILL=1 \
    --env XDB_R31_FLASH_ATTN_MIN_Q=2 \
    --env VLLM_TRITON_FUSED_PREFILL="$fused" \
    --server-port 18001 \
    --stop-existing \
    --server-start-timeout 1800 \
    --remote-timeout 14400 \
    --poll-interval 300 \
    --buckets 16-32K \
    --num-prompts 3 \
    --repetitions 3 \
    --accuracy none 2>&1 | tee -a "$LOG"
  echo "=== $(ts) done $run_id ===" | tee -a "$LOG"
}

trap 'rc=$?; echo "$rc" > "$RUN_ROOT/.exit"; echo "finished_at=$(ts) rc=$rc" | tee -a "$LOG"; exit "$rc"' EXIT

echo "started_at=$(ts)" | tee -a "$LOG"
python3 -m py_compile scripts/guard_bench.py

run_guard "r3.1b-fused-A-default-16to32-20260709-1605" "1"
run_guard "r3.1b-fused-B-xdbvarlen-16to32-20260709-1605" "0"

python3 - "$ROOT" "$RUN_ROOT" <<'PY' | tee -a "$LOG"
import json, pathlib, statistics, sys

root = pathlib.Path(sys.argv[1])
run_root = pathlib.Path(sys.argv[2])
ids = [
    "r3.1b-fused-A-default-16to32-20260709-1605",
    "r3.1b-fused-B-xdbvarlen-16to32-20260709-1605",
]
out = {"runs": {}}
for run_id in ids:
    p = root / "experiments" / run_id / "summary.json"
    data = json.loads(p.read_text())
    bucket = data.get("throughput", {}).get("16-32K", {})
    metrics = bucket.get("median", {})
    out["runs"][run_id] = metrics

a = out["runs"][ids[0]].get("output_throughput")
b = out["runs"][ids[1]].get("output_throughput")
if a and b:
    out["delta_b_vs_a_pct"] = (b / a - 1.0) * 100.0

(run_root / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(out, indent=2, ensure_ascii=False))
PY
