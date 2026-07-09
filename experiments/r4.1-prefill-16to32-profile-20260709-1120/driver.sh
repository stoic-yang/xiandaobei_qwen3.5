#!/usr/bin/env bash
set -euo pipefail

RUN_ID=r4.1-prefill-16to32-profile-20260709-1120
RUN_DIR=/public/home/xdzs2026_c166/codex_runs/$RUN_ID
REPO=/public/home/xdzs2026_c166/vllm_cscc_competition
TESTDATA=/public/home/xdzs2026_c166/testdata
PORT=18001
SESSION=xdb_r41_prefill16_1120

BASE_LD=/opt/ucx/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/mpi/lib:/opt/hwloc/lib
HIPPROF_LD=/opt/dtk-26.04-DCC2602-0317/dcc/lib:/opt/dtk-26.04-DCC2602-0317/hipprof_utils/lib:/opt/dtk-26.04-DCC2602-0317/lib:/opt/dtk-26.04-DCC2602-0317/.hyhal/rocm_smi/lib

mkdir -p "$RUN_DIR/profile" "$RUN_DIR/raw"
exec > >(tee -a "$RUN_DIR/driver.log") 2>&1
trap 'rc=$?; echo "$rc" > "$RUN_DIR/exit"; echo "finished_at=$(date -Is) rc=$rc"; exit "$rc"' EXIT

echo "started_at=$(date -Is)"
echo "run_id=$RUN_ID port=$PORT session=$SESSION"
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

WHEEL=$(ls -t "$REPO"/dist/*.whl | head -1)
echo "install_wheel=$WHEEL"
python3 -m pip install --no-deps --force-reinstall "$WHEEL"

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

cat > "$RUN_DIR/start_vllm_locked.sh" <<START
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
vllm serve "\$MODEL_DIR" \
  --served-model-name Qwen3.5-27B \
  --port $PORT \
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

pkill -TERM -f "vllm serve .*--port $PORT" 2>/dev/null || true
sleep 2
pkill -KILL -f "vllm serve .*--port $PORT" 2>/dev/null || true

echo "starting hipprof-wrapped service at $(date -Is)"
nohup hipprof --hip-trace --hsa-trace --trace-off --session "$SESSION" --flush-interval 1000 --buffer-size 5000 --output-type 0 -o "$RUN_DIR/profile/vllm_prefill16" "$RUN_DIR/start_vllm_locked.sh" > "$RUN_DIR/profile/hipprof_service.log" 2>&1 &
echo $! > "$RUN_DIR/profile/hipprof.pid"

for i in $(seq 1 1200); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
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

cat > "$RUN_DIR/prefill_request.py" <<'PY'
import json
import os
import pathlib
import time
import urllib.request

run = pathlib.Path(os.environ["RUN_DIR"])
port = int(os.environ["PORT"])
model_dir = os.environ["MODEL_DIR"]

def load_prompt(bucket: str) -> str:
    path = pathlib.Path("/public/home/xdzs2026_c166/testdata") / f"{bucket}_throughput.jsonl"
    return json.loads(path.read_text().splitlines()[0])["prompt"]

def token_counts(prompt: str) -> dict:
    out = {}
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        out["plain_tokens"] = len(tok(prompt, add_special_tokens=False).input_ids)
        messages = [{"role": "user", "content": prompt}]
        try:
            ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        out["chat_tokens"] = len(ids)
    except Exception as exc:
        out["tokenizer_error"] = repr(exc)
    return out

def send(prompt: str, bucket: str) -> dict:
    payload = {
        "model": "Qwen3.5-27B",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1,
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter_ns()
    with urllib.request.urlopen(req, timeout=900) as resp:
        body = resp.read()
    t1 = time.perf_counter_ns()
    info = {
        "bucket": bucket,
        "prompt_chars": len(prompt),
        "wall_ms": (t1 - t0) / 1e6,
        "response_bytes": len(body),
        "response_preview": body[:500].decode("utf-8", "replace"),
    }
    info.update(token_counts(prompt))
    return info

warm_prompt = load_prompt("4-8K")
(run / "profile" / "warmup_request.json").write_text(json.dumps(send(warm_prompt, "4-8K"), indent=2))
PY

export RUN_DIR PORT MODEL_DIR
echo "warmup_request_start=$(date -Is)"
python3 "$RUN_DIR/prefill_request.py" --warmup-only > "$RUN_DIR/profile/warmup_prefill.stdout" 2> "$RUN_DIR/profile/warmup_prefill.stderr" || true

cat > "$RUN_DIR/profile_request_only.py" <<'PY'
import json
import os
import pathlib
import time
import urllib.request

run = pathlib.Path(os.environ["RUN_DIR"])
port = int(os.environ["PORT"])
model_dir = os.environ["MODEL_DIR"]
path = pathlib.Path("/public/home/xdzs2026_c166/testdata/16-32K_throughput.jsonl")
prompt = json.loads(path.read_text().splitlines()[0])["prompt"]

counts = {}
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    counts["plain_tokens"] = len(tok(prompt, add_special_tokens=False).input_ids)
    messages = [{"role": "user", "content": prompt}]
    try:
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    counts["chat_tokens"] = len(ids)
except Exception as exc:
    counts["tokenizer_error"] = repr(exc)

payload = {
    "model": "Qwen3.5-27B",
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.0,
    "max_tokens": 1,
    "stream": False,
}
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(payload).encode(),
    method="POST",
    headers={"Content-Type": "application/json"},
)
t0 = time.perf_counter_ns()
with urllib.request.urlopen(req, timeout=900) as resp:
    body = resp.read()
t1 = time.perf_counter_ns()
info = {
    "bucket": "16-32K",
    "prompt_chars": len(prompt),
    "wall_ms": (t1 - t0) / 1e6,
    "response_bytes": len(body),
    "response_preview": body[:500].decode("utf-8", "replace"),
}
info.update(counts)
(run / "profile" / "prefill_request.json").write_text(json.dumps(info, indent=2))
PY

echo "prefill_trace_start=$(date -Is)"
hipprof --session-client "$SESSION" --start >> "$RUN_DIR/profile/hipprof_ctrl.log" 2>&1
python3 "$RUN_DIR/profile_request_only.py" > "$RUN_DIR/profile/prefill_request.stdout" 2> "$RUN_DIR/profile/prefill_request.stderr"
hipprof --session-client "$SESSION" --stop >> "$RUN_DIR/profile/hipprof_ctrl.log" 2>&1 || true
hipprof --session-client "$SESSION" --flush >> "$RUN_DIR/profile/hipprof_ctrl.log" 2>&1 || true

echo "session cleanup $(date -Is)"
hipprof --session-client "$SESSION" --exit >> "$RUN_DIR/profile/hipprof_ctrl.log" 2>&1 || true
sleep 5
cp "$RUN_DIR/profile/hipprof_service.log" "$RUN_DIR/vllm_server.log" || true
pkill -TERM -f "vllm serve .*--port $PORT" 2>/dev/null || true
sleep 2
pkill -KILL -f "vllm serve .*--port $PORT" 2>/dev/null || true

if [ -f "$RUN_DIR/profile/vllm_prefill16.db" ]; then
  hipprof --db "$RUN_DIR/profile/vllm_prefill16.db" --output-type 0 -o "$RUN_DIR/profile/vllm_prefill16_export" > "$RUN_DIR/profile/hipprof_export.log" 2>&1 || true
fi

python3 - "$RUN_DIR" <<'PY'
import csv
import json
import pathlib
import sys

run = pathlib.Path(sys.argv[1])
csvs = sorted((run / "profile").glob("*hipkernel.csv"))
request_path = run / "profile" / "prefill_request.json"
req = json.loads(request_path.read_text()) if request_path.exists() else {}

rows = []
for path in csvs:
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            row["source_csv"] = path.name
            row["calls"] = int(row.get("Calls") or 0)
            row["total_ns"] = int(row.get("TotalDurationNs") or 0)
            row["pct"] = float(row.get("Percentage") or 0)
            rows.append(row)

total_ns = sum(r["total_ns"] for r in rows)
cijk = [r for r in rows if r["Name"].startswith("Cijk_")]
flash = [r for r in rows if "flash_fwd" in r["Name"]]
gdn_core_names = ("chunk_gated_delta", "chunk_fwd_kernel_o", "recompute_w_u", "chunk_scaled_dot_kkt")
gdn_core = [r for r in rows if any(k in r["Name"] for k in gdn_core_names)]

def group(items):
    ns = sum(r["total_ns"] for r in items)
    return {
        "items": len(items),
        "calls": sum(r["calls"] for r in items),
        "total_ms": ns / 1e6,
        "kernel_pct": (ns / total_ns * 100) if total_ns else None,
    }

tokens = req.get("chat_tokens") or req.get("plain_tokens")
families = [
    ("linear_attn.in_proj_qkvz", 48, 16384, 5120),
    ("linear_attn.out_proj", 48, 5120, 6144),
    ("mlp.gate_up_proj", 64, 34816, 5120),
    ("mlp.down_proj", 64, 5120, 17408),
    ("self_attn.qkv_proj", 16, 14336, 5120),
    ("self_attn.o_proj", 16, 5120, 6144),
]
projection = {}
total_flops = None
if tokens:
    total_flops = 0
    for name, layers, n, k in families:
        flops = 2 * tokens * n * k * layers
        projection[name] = flops / 1e12
        total_flops += flops
    projection["total_included"] = total_flops / 1e12

cijk_ms = group(cijk)["total_ms"]
derived = {}
if total_flops and cijk_ms:
    tflops = total_flops / (cijk_ms / 1000) / 1e12
    derived = {
        "aggregate_cijk_tflops": tflops,
        "peak_fraction_vs_395": tflops / 395.0,
        "peak_fraction_pct_vs_395": tflops / 395.0 * 100,
    }

summary = {
    "exp_id": run.name,
    "request": req,
    "profile_csvs": [p.name for p in csvs],
    "kernel": {
        "total_ms": total_ns / 1e6,
        "groups": {
            "gemm_cijk": group(cijk),
            "flash_attention_prefill": group(flash),
            "gdn_core": group(gdn_core),
        },
        "top_kernels": [
            {
                "name": r["Name"],
                "calls": r["calls"],
                "total_ms": r["total_ns"] / 1e6,
                "avg_us": (r["total_ns"] / r["calls"] / 1e3) if r["calls"] else None,
                "kernel_pct": (r["total_ns"] / total_ns * 100) if total_ns else None,
            }
            for r in sorted(rows, key=lambda x: x["total_ns"], reverse=True)[:20]
        ],
    },
    "projection_flops_tflop": projection,
    "derived": derived,
    "verdict": "pending local review",
}
(run / "summary.remote.json").write_text(json.dumps(summary, indent=2))
PY

find "$RUN_DIR/profile" -maxdepth 1 -type f -printf "%p %s\n" | sort > "$RUN_DIR/profile_files.txt"
echo "done"
