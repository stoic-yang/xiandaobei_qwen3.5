#!/usr/bin/env bash
set -u
set -o pipefail

ROOT=/Users/keynary/Code/xiandaobei/meta-main
RUN_ID=r3.0-tunableop-coverage-20260708-133548
EXP="$ROOT/experiments/$RUN_ID"
REMOTE=/public/home/xdzs2026_c166/codex_runs/$RUN_ID

cd "$ROOT" || exit 2
rm -f "$EXP/local_guard_coverage.exit"

PYTHONUNBUFFERED=1 python3 scripts/guard_bench.py \
  --run-id "$RUN_ID" \
  --repo competition \
  --num-prompts 1 \
  --repetitions 1 \
  --buckets 8-16K \
  --accuracy none \
  --locked-start-script \
  --load-format runai_streamer \
  --stop-existing \
  --keep-server \
  --server-start-timeout 1800 \
  --poll-interval 300 \
  --remote-timeout 7200 \
  --env "PYTHONPATH=$REMOTE/sitecustomize:/usr/local" \
  --env XDB_TUNABLE_ENABLE=1 \
  --env "XDB_TUNABLE_FILE=$REMOTE/tunable/tunable.csv" \
  --env "XDB_TUNABLE_RESULTS_JSON=$REMOTE/tunable/results.json" \
  --env XDB_TUNABLE_DUMP_INTERVAL=20 \
  --env XDB_TUNABLE_MAX_ITERATIONS=1 \
  --env XDB_TUNABLE_MAX_DURATION_MS=100 \
  --env XDB_TUNABLE_RECORD_UNTUNED=1 \
  --env XDB_TUNABLE_TUNE=1 \
  > "$EXP/local_guard_coverage.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$EXP/local_guard_coverage.exit"
exit "$rc"
