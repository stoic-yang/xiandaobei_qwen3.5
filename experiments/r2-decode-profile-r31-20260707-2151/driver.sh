#!/usr/bin/env bash
set -euo pipefail
RUN_ID=r2-decode-profile-r31-20260707-2151
RUN_DIR=/public/home/xdzs2026_c166/codex_runs/$RUN_ID
REPO=/public/home/xdzs2026_c166/vllm_cscc_competition
TESTDATA=/public/home/xdzs2026_c166/testdata
SESSION=xdb_r20_2151
BASE_LD=/opt/ucx/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/mpi/lib:/opt/hwloc/lib
HIPPROF_LD=/opt/dtk-26.04-DCC2602-0317/dcc/lib:/opt/dtk-26.04-DCC2602-0317/hipprof_utils/lib:/opt/dtk-26.04-DCC2602-0317/lib:/opt/dtk-26.04-DCC2602-0317/.hyhal/rocm_smi/lib
exec > >(tee -a "$RUN_DIR/driver.log") 2>&1

echo "started_at=$(date -Is)"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy=127.0.0.1,localhost
export DTKROOT=/opt/dtk
export HIP_PATH=/opt/dtk/hip
export HYHAL_PATH=/opt/hyhal
export ROCM_PATH=/opt/dtk
export PYTHONPATH=/usr/local/${PYTHONPATH:+:$PYTHONPATH}
export PATH=/opt/ucx/bin:/opt/dtk/bin:/opt/dtk/llvm/bin:/opt/dtk/hip/bin:/opt/dtk/hip/bin/hipify:/opt/hyhal/bin:/opt/dtk/opencl/bin:/opt/mpi/bin:/opt/hwloc/bin:$PATH
export LD_LIBRARY_PATH=$HIPPROF_LD:$BASE_LD:${LD_LIBRARY_PATH:-}
export MODEL_DIR=/public/home/xdzs2026_c166/Qwen3.5-27B
export VLLM_CACHE_ROOT=/public/home/xdzs2026_c166/vllm_cache/vllm_cache
export TRITON_CACHE_DIR=/public/home/xdzs2026_c166/vllm_cache/triton_cache
export XDB_R31_FLASH_ATTN_PREFILL=1
export XDB_R31_FLASH_ATTN_MIN_Q=2

git config --global --add safe.directory "$REPO" >/dev/null 2>&1 || true
git -C "$REPO" status --short --branch > "$RUN_DIR/repo_status.txt" 2>&1 || true
git -C "$REPO" log --oneline --decorate -12 > "$RUN_DIR/repo_log.txt" 2>&1 || true
echo "repo_head=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"

SITE_ROOT=$(python3 -c "import pathlib, vllm; print(pathlib.Path(vllm.__file__).resolve().parent)")
echo "site_root=$SITE_ROOT"
: > "$RUN_DIR/overlay_manifest.txt"
for file in \
  vllm/model_executor/models/qwen3_5.py \
  vllm/model_executor/models/qwen3_next.py \
  vllm/model_executor/layers/activation.py \
  vllm/model_executor/layers/fla/ops/chunk.py \
  vllm/model_executor/layers/fla/ops/chunk_o.py \
  vllm/v1/attention/backends/triton_attn.py \
  vllm/v1/attention/ops/triton_unified_attention.py \
  vllm/version.py
do
  if [ -f "$REPO/$file" ]; then
    dest="$SITE_ROOT/${file#vllm/}"
    mkdir -p "$(dirname "$dest")"
    cp "$REPO/$file" "$dest"
    printf '%s  %s\n' "$(sha256sum "$dest" | awk '{print $1}')" "$file" >> "$RUN_DIR/overlay_manifest.txt"
  fi
done

python3 - "$RUN_DIR" <<"PY2"
import hashlib, json, pathlib, sys
run=pathlib.Path(sys.argv[1])
import vllm
site=pathlib.Path(vllm.__file__).resolve().parent
files=[
"vllm/model_executor/models/qwen3_5.py",
"vllm/model_executor/models/qwen3_next.py",
"vllm/model_executor/layers/activation.py",
"vllm/model_executor/layers/fla/ops/chunk.py",
"vllm/model_executor/layers/fla/ops/chunk_o.py",
"vllm/v1/attention/backends/triton_attn.py",
"vllm/v1/attention/ops/triton_unified_attention.py",
"vllm/version.py",
]
out={"site_root":str(site),"files":{}}
for f in files:
    p=site / f.removeprefix("vllm/")
    if p.exists():
        out["files"][f]={"site_path":str(p),"site_sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
(run/"runtime_fingerprints.json").write_text(json.dumps(out,indent=2))
PY2

cat > "$RUN_DIR/start_vllm_locked.sh" <<"START"
#!/usr/bin/env bash
set -u
set -o pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy=127.0.0.1,localhost
export MODEL_DIR=/public/home/xdzs2026_c166/Qwen3.5-27B
export VLLM_CACHE_ROOT=/public/home/xdzs2026_c166/vllm_cache/vllm_cache
export TRITON_CACHE_DIR=/public/home/xdzs2026_c166/vllm_cache/triton_cache
export XDB_R31_FLASH_ATTN_PREFILL=1
export XDB_R31_FLASH_ATTN_MIN_Q=2
vllm serve "$MODEL_DIR" \
  --served-model-name Qwen3.5-27B \
  --port 8001 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.95 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --load-format runai_streamer
START
chmod +x "$RUN_DIR/start_vllm_locked.sh"

pkill -TERM -f "vllm serve .*Qwen3.5-27B" 2>/dev/null || true
sleep 5
pkill -KILL -f "vllm serve .*Qwen3.5-27B" 2>/dev/null || true

echo "starting hipprof-wrapped service session=$SESSION at $(date -Is)"
nohup hipprof --hip-trace --hsa-trace --trace-off --session "$SESSION" --flush-interval 1000 --buffer-size 5000 --output-type 0 -o "$RUN_DIR/profile/vllm_decode" "$RUN_DIR/start_vllm_locked.sh" > "$RUN_DIR/profile/hipprof_service.log" 2>&1 &
echo $! > "$RUN_DIR/profile/hipprof.pid"

for i in $(seq 1 1200); do
  if curl -fsS --max-time 2 http://127.0.0.1:8001/health >/dev/null 2>&1; then
    echo "health_ok_at=$(date -Is) wait_s=$i"
    break
  fi
  if ! kill -0 "$(cat "$RUN_DIR/profile/hipprof.pid")" 2>/dev/null; then
    echo "hipprof service exited before health" >&2
    tail -200 "$RUN_DIR/profile/hipprof_service.log" >&2 || true
    exit 20
  fi
  sleep 1
  if [ "$i" = 1200 ]; then
    echo "health timeout" >&2
    tail -200 "$RUN_DIR/profile/hipprof_service.log" >&2 || true
    exit 21
  fi
done

cp "$RUN_DIR/profile/hipprof_service.log" "$RUN_DIR/vllm_server.log" || true
cd "$TESTDATA"
echo "warmup_start=$(date -Is)"
./run_throughput.sh 4-8K 1 > "$RUN_DIR/raw/warmup_4-8K.log" 2>&1 || true
echo "decode_trace_start=$(date -Is)"
python3 "$RUN_DIR/decode_stream_profile.py" \
  --prompt-file "$TESTDATA/8-16K_throughput.jsonl" \
  --prompt-row 0 \
  --session "$SESSION" \
  --hipprof hipprof \
  --max-tokens 96 \
  --trace-chunks 64 \
  --output "$RUN_DIR/profile/decode_stream.json" \
  > "$RUN_DIR/profile/decode_stream.stdout" \
  2> "$RUN_DIR/profile/decode_stream.stderr" || echo "decode_stream_failed=$?"

echo "session cleanup $(date -Is)"
hipprof --session-client "$SESSION" --stop >> "$RUN_DIR/profile/hipprof_ctrl.log" 2>&1 || true
hipprof --session-client "$SESSION" --flush >> "$RUN_DIR/profile/hipprof_ctrl.log" 2>&1 || true
hipprof --session-client "$SESSION" --exit >> "$RUN_DIR/profile/hipprof_ctrl.log" 2>&1 || true
sleep 5
pkill -TERM -f "vllm serve .*Qwen3.5-27B" 2>/dev/null || true
sleep 2
pkill -KILL -f "vllm serve .*Qwen3.5-27B" 2>/dev/null || true

if [ -f "$RUN_DIR/profile/vllm_decode.db" ]; then
  hipprof --db "$RUN_DIR/profile/vllm_decode.db" --output-type 0 -o "$RUN_DIR/profile/vllm_decode_export" > "$RUN_DIR/profile/hipprof_export.log" 2>&1 || true
fi
find "$RUN_DIR/profile" -maxdepth 2 -type f -printf "%p %s\n" | sort > "$RUN_DIR/profile_files.txt"
echo "finished_at=$(date -Is)"

