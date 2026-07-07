#!/usr/bin/env bash
set -euo pipefail
RUN_ID=runai-startup-probe-fresh-20260707-1259
RUN_DIR=/public/home/xdzs2026_c166/codex_runs/$RUN_ID
MODEL_DIR=/public/home/xdzs2026_c166/Qwen3.5-27B
CACHE_ROOT=/public/home/xdzs2026_c166/vllm_cache
WHEEL=/public/home/xdzs2026_c166/vllm_cscc_competition/dist/vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl
mkdir -p "$RUN_DIR" "$CACHE_ROOT/vllm_cache" "$CACHE_ROOT/triton_cache"
exec > >(tee -a "$RUN_DIR/driver.log") 2>&1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy=127.0.0.1,localhost
export DTKROOT=/opt/dtk
export HIP_PATH=/opt/dtk/hip
export HYHAL_PATH=/opt/hyhal
export ROCM_PATH=/opt/dtk
export PYTHONPATH=/usr/local/${PYTHONPATH:+:$PYTHONPATH}
export PATH=/opt/ucx/bin:/opt/dtk/bin:/opt/dtk/llvm/bin:/opt/dtk/hip/bin:/opt/dtk/hip/bin/hipify:/opt/hyhal/bin:/opt/dtk/opencl/bin:/opt/mpi/bin:/opt/hwloc/bin:$PATH
export LD_LIBRARY_PATH=/opt/ucx/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/mpi/lib:/opt/hwloc/lib:${LD_LIBRARY_PATH:-}
export MODEL_DIR
export VLLM_CACHE_ROOT=$CACHE_ROOT/vllm_cache
export TRITON_CACHE_DIR=$CACHE_ROOT/triton_cache

echo "=== runai fresh startup probe ==="
echo "run_id=$RUN_ID"
echo "host=$(hostname)"
echo "started_at=$(date -Is)"
echo "MODEL_DIR=$MODEL_DIR"
echo "WHEEL=$WHEEL"
echo "VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT"
echo "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
echo "load_format=runai_streamer"

pip_start=$(date +%s)
echo "pip_install_start=$(date -Is) epoch=$pip_start"
python3 -m pip install --no-deps "$WHEEL"
pip_end=$(date +%s)
echo "pip_install_done=$(date -Is) epoch=$pip_end elapsed_s=$((pip_end - pip_start))"

cat > "$RUN_DIR/start_vllm_runai.sh" <<'EOS'
#!/usr/bin/env bash
set -u
set -o pipefail
vllm serve "$MODEL_DIR" \
    --served-model-name Qwen3.5-27B \
    --port 8001 \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.95 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --load-format runai_streamer
EOS
chmod +x "$RUN_DIR/start_vllm_runai.sh"

start_epoch=$(date +%s)
echo "server_start_at=$(date -Is) epoch=$start_epoch"
(cd "$RUN_DIR" && nohup bash "$RUN_DIR/start_vllm_runai.sh" > "$RUN_DIR/vllm_server.log" 2>&1 & echo $! > "$RUN_DIR/vllm_server.pid")
pid=$(cat "$RUN_DIR/vllm_server.pid")
echo "server_pid=$pid"
health_ok() { curl -fsS --noproxy '*' http://127.0.0.1:8001/health >/dev/null 2>&1; }
deadline=$((SECONDS + 1200))
while ! health_ok; do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "server exited before health" >&2
    tail -n 240 "$RUN_DIR/vllm_server.log" >&2 || true
    exit 13
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "server health timeout" >&2
    tail -n 240 "$RUN_DIR/vllm_server.log" >&2 || true
    exit 14
  fi
  sleep 10
done
ready_epoch=$(date +%s)
elapsed=$((ready_epoch - start_epoch))
echo "server_ready_at=$(date -Is) epoch=$ready_epoch elapsed_s=$elapsed"
python3 - "$RUN_DIR" "$((pip_end - pip_start))" "$elapsed" <<'PY'
import json, re, sys, time
from pathlib import Path
run_dir = Path(sys.argv[1])
pip_elapsed = int(sys.argv[2])
health_elapsed = int(sys.argv[3])
log = (run_dir / 'vllm_server.log').read_text(encoding='utf-8', errors='ignore')
patterns = {
    'loading_weights_s': r'Loading weights took ([0-9.]+) seconds',
    'model_loading_s': r'Model loading took .* and ([0-9.]+) seconds',
    'initial_profiling_s': r'Initial profiling/warmup run took ([0-9.]+) s',
    'engine_init_s': r'init engine \(profile, create kv cache, warmup model\) took ([0-9.]+) seconds',
}
metrics = {}
for k, pat in patterns.items():
    m = re.search(pat, log)
    if m:
        metrics[k] = float(m.group(1))
summary = {
    'run_id': 'runai-startup-probe-fresh-20260707-1259',
    'status': 'pass',
    'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    'model_dir': '/public/home/xdzs2026_c166/Qwen3.5-27B',
    'load_format': 'runai_streamer',
    'vllm_cache_root': '/public/home/xdzs2026_c166/vllm_cache/vllm_cache',
    'triton_cache_dir': '/public/home/xdzs2026_c166/vllm_cache/triton_cache',
    'pip_install_elapsed_s': pip_elapsed,
    'health_elapsed_s': health_elapsed,
    'log_metrics': metrics,
}
(run_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
echo "completed_at=$(date -Is)"
