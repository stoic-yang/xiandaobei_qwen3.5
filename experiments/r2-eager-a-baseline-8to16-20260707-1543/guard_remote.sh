set -euo pipefail

RUN_ID=r2-eager-a-baseline-8to16-20260707-1543
RUN_DIR=/public/home/xdzs2026_c166/codex_runs/$RUN_ID
REPO_PATH=/public/home/xdzs2026_c166/vllm_cscc_competition
WHEEL_GLOB='/public/home/xdzs2026_c166/vllm_cscc_competition/dist/*.whl'
TESTDATA=/public/home/xdzs2026_c166/testdata
PERSIST_MODEL_DIR=/public/home/xdzs2026_c166/Qwen3.5-27B
LOCAL_MODEL_DIR=/root/Qwen3.5-27B
COPY_MODEL_LOCAL=0
NUM_PROMPTS=3
REPETITIONS=3
BUCKETS=8-16K
REPO_KIND=competition
OVERLAY_REV=''
OVERLAY_SOURCE_DIR=''
ACCURACY=none
ACCURACY_ROWS=''
COPY_ACCURACY_OUTPUT=0
LOCKED_START_SCRIPT=1
LOAD_FORMAT=runai_streamer
ENFORCE_EAGER=0
REUSE_SERVER=0
KEEP_SERVER=1
STOP_EXISTING=1
INSTALL_WHEEL=1
SERVER_START_TIMEOUT=1200
export NUM_PROMPTS REPETITIONS BUCKETS REPO_KIND OVERLAY_REV OVERLAY_SOURCE_DIR LOCKED_START_SCRIPT LOAD_FORMAT ENFORCE_EAGER

mkdir -p "$RUN_DIR/raw" "$RUN_DIR/throughput" "$RUN_DIR/accuracy"
exec > >(tee -a "$RUN_DIR/driver.log") 2>&1

echo "=== guard benchmark ==="
echo "run_id=$RUN_ID"
echo "host=$(hostname)"
echo "started_at=$(date -Is)"
echo "repo_kind=competition"
echo "repo_path=$REPO_PATH"
echo "num_prompts=$NUM_PROMPTS repetitions=$REPETITIONS buckets=$BUCKETS accuracy=$ACCURACY accuracy_rows=$ACCURACY_ROWS overlay_rev=$OVERLAY_REV"
echo "locked_start_script=$LOCKED_START_SCRIPT load_format=$LOAD_FORMAT enforce_eager=$ENFORCE_EAGER"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy=127.0.0.1,localhost
export DTKROOT=/opt/dtk
export HIP_PATH=/opt/dtk/hip
export HYHAL_PATH=/opt/hyhal
export ROCM_PATH=/opt/dtk
export PYTHONPATH=/usr/local/${PYTHONPATH:+:$PYTHONPATH}
export PATH=/opt/ucx/bin:/opt/dtk/bin:/opt/dtk/llvm/bin:/opt/dtk/hip/bin:/opt/dtk/hip/bin/hipify:/opt/hyhal/bin:/opt/dtk/opencl/bin:/opt/mpi/bin:/opt/hwloc/bin:$PATH
export LD_LIBRARY_PATH=/opt/ucx/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/mpi/lib:/opt/hwloc/lib:${LD_LIBRARY_PATH:-}
export VLLM_CACHE_ROOT=/public/home/xdzs2026_c166/vllm_cache/vllm_cache
export TRITON_CACHE_DIR=/public/home/xdzs2026_c166/vllm_cache/triton_cache

if [ "$COPY_MODEL_LOCAL" = "1" ] && [ ! -f "$LOCAL_MODEL_DIR/config.json" ]; then
  echo "copying model to local disk: $PERSIST_MODEL_DIR -> $LOCAL_MODEL_DIR"
  rm -rf "${LOCAL_MODEL_DIR}.tmp"
  mkdir -p "$(dirname "$LOCAL_MODEL_DIR")"
  cp -a "$PERSIST_MODEL_DIR" "${LOCAL_MODEL_DIR}.tmp"
  mv "${LOCAL_MODEL_DIR}.tmp" "$LOCAL_MODEL_DIR"
  echo "model_copy_done_at=$(date -Is)"
fi

if [ -f "$LOCAL_MODEL_DIR/config.json" ]; then
  export MODEL_DIR="$LOCAL_MODEL_DIR"
  echo "model_dir_source=local MODEL_DIR=$MODEL_DIR"
else
  export MODEL_DIR="$PERSIST_MODEL_DIR"
  echo "model_dir_source=persistent MODEL_DIR=$MODEL_DIR"
fi

git config --global --add safe.directory "$REPO_PATH" >/dev/null 2>&1 || true
git_quick() {
  if command -v timeout >/dev/null 2>&1; then
    timeout 20 git -C "$REPO_PATH" "$@"
  else
    git -C "$REPO_PATH" "$@"
  fi
}
REPO_HEAD="$(git_quick rev-parse HEAD 2>/dev/null || echo unknown)"
REPO_BRANCH="$(git_quick rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
git_quick status --short --branch > "$RUN_DIR/repo_status.txt" 2>&1 || true
git_quick log --oneline --decorate -12 > "$RUN_DIR/repo_log.txt" 2>&1 || true
echo "repo_head=$REPO_HEAD branch=$REPO_BRANCH"

WHEEL="$(ls -t $WHEEL_GLOB 2>/dev/null | head -n 1 || true)"
if [ -z "$WHEEL" ]; then
  echo "no wheel matched: $WHEEL_GLOB" >&2
  exit 11
fi
WHEEL_SHA="$(sha256sum "$WHEEL" | awk '{print $1}')"
echo "wheel=$WHEEL"
echo "wheel_sha256=$WHEEL_SHA"
if [ "$INSTALL_WHEEL" = "1" ]; then
  python3 -m pip install --no-deps "$WHEEL"
fi

if [ -n "$OVERLAY_REV" ] || [ -n "$OVERLAY_SOURCE_DIR" ]; then
  echo "overlay_rev=$OVERLAY_REV"
  echo "overlay_source_dir=$OVERLAY_SOURCE_DIR"
  if [ -n "$OVERLAY_REV" ]; then
    git_quick cat-file -e "$OVERLAY_REV^{commit}"
  fi
  SITE_ROOT="$(python3 -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).resolve().parent)')"
  echo "overlay_site_root=$SITE_ROOT"
  : > "$RUN_DIR/overlay_manifest.txt"
  for file in             vllm/model_executor/models/qwen3_5.py             vllm/model_executor/models/qwen3_next.py             vllm/model_executor/layers/activation.py             vllm/model_executor/layers/fla/ops/chunk.py             vllm/model_executor/layers/fla/ops/chunk_o.py             vllm/v1/attention/ops/triton_unified_attention.py             vllm/version.py
  do
    src=""
    if [ -n "$OVERLAY_SOURCE_DIR" ] && [ -f "$OVERLAY_SOURCE_DIR/$file" ]; then
      src="$OVERLAY_SOURCE_DIR/$file"
    fi
    if [ -n "$src" ]; then
      dest="$SITE_ROOT/${file#vllm/}"
      mkdir -p "$(dirname "$dest")"
      cp "$src" "$dest"
      printf '%s  %s\n' "$(sha256sum "$dest" | awk '{print $1}')" "$file" >> "$RUN_DIR/overlay_manifest.txt"
      echo "overlay_file=$file"
    elif [ -n "$OVERLAY_REV" ] && git_quick cat-file -e "$OVERLAY_REV:$file" 2>/dev/null; then
      dest="$SITE_ROOT/${file#vllm/}"
      mkdir -p "$(dirname "$dest")"
      git_quick show "$OVERLAY_REV:$file" > "$dest"
      printf '%s  %s\n' "$(sha256sum "$dest" | awk '{print $1}')" "$file" >> "$RUN_DIR/overlay_manifest.txt"
      echo "overlay_file=$file"
    else
      echo "overlay_missing=$file"
    fi
  done
fi

python3 - "$RUN_DIR" "$WHEEL" <<'PY'
import hashlib, json, pathlib, sys, zipfile

run_dir = pathlib.Path(sys.argv[1])
wheel = pathlib.Path(sys.argv[2])
files = [
    "vllm/model_executor/models/qwen3_5.py",
    "vllm/model_executor/models/qwen3_next.py",
    "vllm/model_executor/layers/activation.py",
    "vllm/model_executor/layers/fla/ops/chunk.py",
    "vllm/model_executor/layers/fla/ops/chunk_o.py",
    "vllm/v1/attention/ops/triton_unified_attention.py",
    "vllm/version.py",
]

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

site_root = None
try:
    import vllm

    site_root = pathlib.Path(vllm.__file__).resolve().parent
except Exception:
    site_root = None

out = {
    "wheel": {"path": str(wheel), "sha256": sha(wheel.read_bytes())},
    "site_root": str(site_root) if site_root else None,
    "files": {},
}
with zipfile.ZipFile(wheel) as zf:
    names = set(zf.namelist())
    for file in files:
        item = {"wheel_sha256": None, "site_sha256": None, "site_path": None}
        if file in names:
            item["wheel_sha256"] = sha(zf.read(file))
        if site_root:
            site_path = site_root / file.removeprefix("vllm/")
            if site_path.exists():
                item["site_path"] = str(site_path)
                item["site_sha256"] = sha(site_path.read_bytes())
        out["files"][file] = item
(run_dir / "runtime_fingerprints.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY

health_ok() {
  curl -fsS http://127.0.0.1:8001/health >/dev/null 2>&1
}

START_SCRIPT="$TESTDATA/start_vllm.sh"
if [ "$LOCKED_START_SCRIPT" = "1" ]; then
  START_SCRIPT="$RUN_DIR/start_vllm_locked.sh"
  cat > "$START_SCRIPT" <<'EOS'
#!/usr/bin/env bash
set -u
set -o pipefail

load_format_args=()
if [ -n "${GUARD_LOAD_FORMAT:-}" ]; then
  load_format_args=(--load-format "$GUARD_LOAD_FORMAT")
fi
eager_args=()
if [ "${GUARD_ENFORCE_EAGER:-0}" = "1" ]; then
  eager_args=(--enforce-eager)
fi

vllm serve "$MODEL_DIR"             --served-model-name Qwen3.5-27B             --port 8001             --trust-remote-code             --dtype bfloat16             --tensor-parallel-size 1             --max-model-len 32768             --max-num-seqs 128             --max-num-batched-tokens 4096             --gpu-memory-utilization 0.95             --default-chat-template-kwargs '{"enable_thinking": false}'             --reasoning-parser qwen3             --enable-auto-tool-choice             --tool-call-parser qwen3_coder             "${eager_args[@]}"             "${load_format_args[@]}"
EOS
  chmod +x "$START_SCRIPT"
  export GUARD_LOAD_FORMAT="$LOAD_FORMAT"
  export GUARD_ENFORCE_EAGER="$ENFORCE_EAGER"
fi

SERVER_PID=""
if [ "$REUSE_SERVER" = "1" ] && health_ok; then
  echo "reuse_server=1 health=ok"
else
  if pgrep -af "vllm serve|start_vllm" | grep -v -E "pgrep -af|grep -v|guard_bench" >/tmp/guard_active_vllm.txt; then
    if [ "$STOP_EXISTING" = "1" ]; then
      echo "stopping existing server processes"
      awk '{print $1}' /tmp/guard_active_vllm.txt | xargs -r kill || true
      sleep 5
      awk '{print $1}' /tmp/guard_active_vllm.txt | xargs -r kill -9 || true
    else
      echo "existing vLLM processes found; use --reuse-server or --stop-existing" >&2
      cat /tmp/guard_active_vllm.txt >&2
      exit 12
    fi
  fi
  echo "starting vLLM server"
  (cd "$RUN_DIR" && nohup bash "$START_SCRIPT" > "$RUN_DIR/vllm_server.log" 2>&1 & echo $! > "$RUN_DIR/vllm_server.pid")
  SERVER_PID="$(cat "$RUN_DIR/vllm_server.pid")"
  deadline=$((SECONDS + SERVER_START_TIMEOUT))
  until health_ok; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "server exited before health check" >&2
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
  echo "server_ready_at=$(date -Is) pid=$SERVER_PID"
fi

cleanup() {
  rc=$?
  if [ "$KEEP_SERVER" != "1" ] && [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "stopping server pid=$SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 3
    kill -9 "$SERVER_PID" 2>/dev/null || true
  elif [ "$KEEP_SERVER" = "1" ]; then
    echo "keep_server=1"
  fi
  exit "$rc"
}
trap cleanup EXIT

cd "$TESTDATA"
echo "=== warmup ==="
./run_throughput.sh 4-8K 1 > "$RUN_DIR/raw/warmup_4-8K.log" 2>&1

for bucket in $BUCKETS; do
  for rep in $(seq 1 "$REPETITIONS"); do
    echo "=== throughput bucket=$bucket rep=$rep ==="
    rm -f "./test/${bucket}_throughput/result.json"
    ./run_throughput.sh "$bucket" "$NUM_PROMPTS" > "$RUN_DIR/raw/throughput_${bucket}_rep${rep}.log" 2>&1
    cp "./test/${bucket}_throughput/result.json" "$RUN_DIR/throughput/${bucket}_rep${rep}.json"
  done
done

if [ "$ACCURACY" != "none" ]; then
  echo "=== accuracy $ACCURACY ==="
  if [ -n "$ACCURACY_ROWS" ]; then
    ./run_accuracy.sh all "$ACCURACY_ROWS" > "$RUN_DIR/raw/accuracy.log" 2>&1
  else
    ./run_accuracy.sh all > "$RUN_DIR/raw/accuracy.log" 2>&1
  fi
  if [ "$COPY_ACCURACY_OUTPUT" = "1" ]; then
    if command -v timeout >/dev/null 2>&1; then
      timeout 20 cp -a ./accuracy_debug/output "$RUN_DIR/accuracy/output" 2>/dev/null || true
    else
      cp -a ./accuracy_debug/output "$RUN_DIR/accuracy/output" 2>/dev/null || true
    fi
  else
    echo "accuracy_output_copy=skipped"
  fi
fi

python3 - "$RUN_DIR" "$RUN_ID" "$REPO_HEAD" "$REPO_BRANCH" "$REPO_PATH" "$WHEEL" "$WHEEL_SHA" "$MODEL_DIR" "$ACCURACY" <<'PY'
import json, os, re, statistics, sys, time
from pathlib import Path

run_dir = Path(sys.argv[1])
run_id, repo_head, repo_branch, repo_path, wheel, wheel_sha, model_dir, accuracy_mode = sys.argv[2:]
buckets = tuple(os.environ.get("BUCKETS", "4-8K 8-16K 16-32K").split())
weights = {"4-8K": 0.2, "8-16K": 0.5, "16-32K": 0.3}

def metric(data, key):
    if key in data:
        return data[key]
    if key == "p99_ttft_ms" and "ttfts" in data:
        xs = sorted(float(x) * 1000 for x in data["ttfts"])
    elif key == "p99_tpot_ms" and "tpot" in data:
        xs = sorted(float(x) * 1000 for x in data["tpot"])
    else:
        return None
    if not xs:
        return None
    idx = min(len(xs) - 1, int(0.99 * (len(xs) - 1)))
    return xs[idx]

throughput = {}
for bucket in buckets:
    reps = []
    for path in sorted((run_dir / "throughput").glob(f"{bucket}_rep*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        reps.append({
            "path": str(path),
            "completed": data.get("completed"),
            "failed": data.get("failed"),
            "output_throughput": metric(data, "output_throughput"),
            "p99_ttft_ms": metric(data, "p99_ttft_ms"),
            "p99_tpot_ms": metric(data, "p99_tpot_ms"),
            "duration": data.get("duration"),
        })
    med = {}
    for key in ("output_throughput", "p99_ttft_ms", "p99_tpot_ms", "duration"):
        vals = [r[key] for r in reps if r.get(key) is not None]
        med[key] = statistics.median(vals) if vals else None
    throughput[bucket] = {"repetitions": reps, "median": med}

accuracy = {}
acc_log = run_dir / "raw" / "accuracy.log"
if acc_log.exists():
    text = acc_log.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        m = re.match(r"\|\s*(hotpotqa|gov_report)\s*\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|", line)
        if m:
            accuracy[m.group(1)] = {
                "version": m.group(2).strip(),
                "metric": m.group(3).strip(),
                "mode": m.group(4).strip(),
                "score": m.group(5).strip(),
            }
            continue
        m = re.match(r"(retrieval_multi_point|aggregation_keyword_aggregation)_recalculated:\s*([0-9.]+|NA)\s*(?:\(([^)]*)\))?", line)
        if m:
            accuracy[m.group(1)] = {"metric": "recalculated_acc", "score": m.group(2), "count": m.group(3) or ""}

weighted = None
if set(buckets) == set(weights) and all(throughput[b]["median"].get("output_throughput") is not None for b in buckets):
    weighted = sum(weights[b] * throughput[b]["median"]["output_throughput"] for b in buckets)

summary = {
    "run_id": run_id,
    "status": "pass",
    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "repo": {
        "kind": os.environ.get("REPO_KIND", ""),
        "path": repo_path,
        "branch": repo_branch,
        "head": repo_head,
        "status_path": str(run_dir / "repo_status.txt"),
        "log_path": str(run_dir / "repo_log.txt"),
    },
    "wheel": {"path": wheel, "sha256": wheel_sha},
    "runtime_fingerprints_path": str(run_dir / "runtime_fingerprints.json") if (run_dir / "runtime_fingerprints.json").exists() else None,
    "overlay": {
        "rev": os.environ.get("OVERLAY_REV", ""),
        "source_dir": os.environ.get("OVERLAY_SOURCE_DIR", ""),
        "manifest_path": str(run_dir / "overlay_manifest.txt") if (run_dir / "overlay_manifest.txt").exists() else None,
    },
    "model_dir": model_dir,
    "protocol": {
        "warmup": "run_throughput.sh 4-8K 1",
        "buckets": list(buckets),
        "repetitions": int(os.environ.get("REPETITIONS", "0") or 0),
        "num_prompts": int(os.environ.get("NUM_PROMPTS", "0") or 0),
        "accuracy": accuracy_mode,
        "locked_start_script": os.environ.get("LOCKED_START_SCRIPT", ""),
        "load_format": os.environ.get("LOAD_FORMAT", ""),
        "enforce_eager": os.environ.get("ENFORCE_EAGER", ""),
    },
    "throughput": throughput,
    "weighted_output_throughput": weighted,
    "accuracy": accuracy,
}
(run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("=== summary ===")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "completed_at=$(date -Is)"
