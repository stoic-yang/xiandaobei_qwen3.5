#!/usr/bin/env python3
"""Run the Xiandaobei warm-container guard benchmark protocol.

Protocol:
  1. attach to an already-running SCNet worker via generated SSH config;
  2. install the selected wheel, start or reuse one warm vLLM server;
  3. run an explicit warmup;
  4. run each throughput bucket N times and report medians;
  5. run the four accuracy tasks once and collect their scores.

The script does not edit vLLM source. It only installs a wheel into the
ephemeral container and writes run artifacts under codex_runs/<run_id>.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import statistics
import subprocess
import sys
import textwrap
import time
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_config(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def q(value: str | pathlib.Path | int | None) -> str:
    return shlex.quote("" if value is None else str(value))


def run(
    cmd: list[str],
    *,
    cwd: pathlib.Path | None = None,
    stdin: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        input=stdin,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def require_ok(proc: subprocess.CompletedProcess[str], context: str) -> None:
    if proc.returncode == 0:
        return
    sys.stderr.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    raise SystemExit(f"{context} failed with rc={proc.returncode}")


def now_run_id() -> str:
    return time.strftime("guard-%Y%m%d-%H%M%S")


def remote_script(args: argparse.Namespace, cfg: dict[str, Any]) -> str:
    bucket_list = [item.strip() for item in args.buckets.split(",") if item.strip()]
    valid_buckets = {"4-8K", "8-16K", "16-32K"}
    invalid = [item for item in bucket_list if item not in valid_buckets]
    if not bucket_list or invalid:
        raise SystemExit(f"--buckets must be comma-separated from {sorted(valid_buckets)}, got {args.buckets!r}")
    bucket_words = " ".join(bucket_list)

    dcu_env = {
        "DTKROOT": "/opt/dtk",
        "HIP_PATH": "/opt/dtk/hip",
        "HYHAL_PATH": "/opt/hyhal",
        "ROCM_PATH": "/opt/dtk",
        "PYTHONPATH": "/usr/local/",
        "PATH_PREFIX": "/opt/ucx/bin:/opt/dtk/bin:/opt/dtk/llvm/bin:/opt/dtk/hip/bin:/opt/dtk/hip/bin/hipify:/opt/hyhal/bin:/opt/dtk/opencl/bin:/opt/mpi/bin:/opt/hwloc/bin",
        "LD_LIBRARY_PATH_PREFIX": "/opt/ucx/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/mpi/lib:/opt/hwloc/lib",
    }
    dcu_env.update(cfg.get("dcu_env", {}))

    repo_path = cfg["competition_repo"] if args.repo == "competition" else cfg["baseline_repo"]
    wheel_glob = cfg["competition_wheel_glob"] if args.repo == "competition" else cfg["baseline_wheel_glob"]
    if args.source_repo:
        repo_path = args.source_repo
    if args.wheel:
        wheel_glob = args.wheel

    accuracy_rows = ""
    if args.accuracy == "smoke":
        accuracy_rows = str(args.accuracy_rows or 10)
    elif args.accuracy == "full" and args.accuracy_rows:
        accuracy_rows = str(args.accuracy_rows)

    extra_exports = "\n".join(
        f"export {item.split('=', 1)[0]}={q(item.split('=', 1)[1])}"
        for item in args.env
        if "=" in item and item.split("=", 1)[0].isidentifier()
    )
    if extra_exports:
        extra_exports = textwrap.indent(extra_exports, "        ")

    return textwrap.dedent(
        f"""\
        set -euo pipefail

        RUN_ID={q(args.run_id)}
        RUN_DIR={q(cfg['remote_runs_dir'])}/$RUN_ID
        REPO_PATH={q(repo_path)}
        WHEEL_GLOB={q(wheel_glob)}
        TESTDATA={q(cfg['testdata_dir'])}
        PERSIST_MODEL_DIR={q(args.model_dir or cfg.get('model_dir', '/public/home/xdzs2026_c166/Qwen3.5-27B'))}
        LOCAL_MODEL_DIR={q(args.local_model_dir)}
        COPY_MODEL_LOCAL={1 if args.copy_model_local else 0}
        NUM_PROMPTS={q(args.num_prompts)}
        REPETITIONS={q(args.repetitions)}
        BUCKETS={q(bucket_words)}
        REPO_KIND={q(args.repo)}
        OVERLAY_REV={q(args.overlay_rev)}
        OVERLAY_SOURCE_DIR={q(args.overlay_source_dir)}
        ACCURACY={q(args.accuracy)}
        ACCURACY_ROWS={q(accuracy_rows)}
        COPY_ACCURACY_OUTPUT={1 if args.copy_accuracy_output else 0}
        LOCKED_START_SCRIPT={1 if args.locked_start_script else 0}
        LOAD_FORMAT={q(args.load_format or "")}
        ENFORCE_EAGER={1 if args.enforce_eager else 0}
        REUSE_SERVER={1 if args.reuse_server else 0}
        KEEP_SERVER={1 if args.keep_server else 0}
        STOP_EXISTING={1 if args.stop_existing else 0}
        INSTALL_WHEEL={0 if args.no_install else 1}
        SERVER_START_TIMEOUT={q(args.server_start_timeout)}
        export NUM_PROMPTS REPETITIONS BUCKETS REPO_KIND OVERLAY_REV OVERLAY_SOURCE_DIR LOCKED_START_SCRIPT LOAD_FORMAT ENFORCE_EAGER

        mkdir -p "$RUN_DIR/raw" "$RUN_DIR/throughput" "$RUN_DIR/accuracy"
        exec > >(tee -a "$RUN_DIR/driver.log") 2>&1

        echo "=== guard benchmark ==="
        echo "run_id=$RUN_ID"
        echo "host=$(hostname)"
        echo "started_at=$(date -Is)"
        echo "repo_kind={args.repo}"
        echo "repo_path=$REPO_PATH"
        echo "num_prompts=$NUM_PROMPTS repetitions=$REPETITIONS buckets=$BUCKETS accuracy=$ACCURACY accuracy_rows=$ACCURACY_ROWS overlay_rev=$OVERLAY_REV"
        echo "locked_start_script=$LOCKED_START_SCRIPT load_format=$LOAD_FORMAT enforce_eager=$ENFORCE_EAGER"

        unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
        export no_proxy=127.0.0.1,localhost
        export DTKROOT={q(dcu_env['DTKROOT'])}
        export HIP_PATH={q(dcu_env['HIP_PATH'])}
        export HYHAL_PATH={q(dcu_env['HYHAL_PATH'])}
        export ROCM_PATH={q(dcu_env['ROCM_PATH'])}
        export PYTHONPATH={q(dcu_env['PYTHONPATH'])}${{PYTHONPATH:+:$PYTHONPATH}}
        export PATH={q(dcu_env['PATH_PREFIX'])}:$PATH
        export LD_LIBRARY_PATH={q(dcu_env['LD_LIBRARY_PATH_PREFIX'])}:${{LD_LIBRARY_PATH:-}}
{extra_exports}

        if [ "$COPY_MODEL_LOCAL" = "1" ] && [ ! -f "$LOCAL_MODEL_DIR/config.json" ]; then
          echo "copying model to local disk: $PERSIST_MODEL_DIR -> $LOCAL_MODEL_DIR"
          rm -rf "${{LOCAL_MODEL_DIR}}.tmp"
          mkdir -p "$(dirname "$LOCAL_MODEL_DIR")"
          cp -a "$PERSIST_MODEL_DIR" "${{LOCAL_MODEL_DIR}}.tmp"
          mv "${{LOCAL_MODEL_DIR}}.tmp" "$LOCAL_MODEL_DIR"
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
        git_quick() {{
          if command -v timeout >/dev/null 2>&1; then
            timeout 20 git -C "$REPO_PATH" "$@"
          else
            git -C "$REPO_PATH" "$@"
          fi
        }}
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
        WHEEL_SHA="$(sha256sum "$WHEEL" | awk '{{print $1}}')"
        echo "wheel=$WHEEL"
        echo "wheel_sha256=$WHEEL_SHA"
        if [ "$INSTALL_WHEEL" = "1" ]; then
          python3 -m pip install --no-deps "$WHEEL"
        fi

        if [ -n "$OVERLAY_REV" ] || [ -n "$OVERLAY_SOURCE_DIR" ]; then
          echo "overlay_rev=$OVERLAY_REV"
          echo "overlay_source_dir=$OVERLAY_SOURCE_DIR"
          if [ -n "$OVERLAY_REV" ]; then
            git_quick cat-file -e "$OVERLAY_REV^{{commit}}"
          fi
          SITE_ROOT="$(python3 -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).resolve().parent)')"
          echo "overlay_site_root=$SITE_ROOT"
          : > "$RUN_DIR/overlay_manifest.txt"
          for file in \
            vllm/model_executor/models/qwen3_5.py \
            vllm/model_executor/models/qwen3_next.py \
            vllm/model_executor/layers/activation.py \
            vllm/model_executor/layers/fla/ops/chunk.py \
            vllm/model_executor/layers/fla/ops/chunk_o.py \
            vllm/v1/attention/ops/triton_unified_attention.py \
            vllm/version.py
          do
            src=""
            if [ -n "$OVERLAY_SOURCE_DIR" ] && [ -f "$OVERLAY_SOURCE_DIR/$file" ]; then
              src="$OVERLAY_SOURCE_DIR/$file"
            fi
            if [ -n "$src" ]; then
              dest="$SITE_ROOT/${{file#vllm/}}"
              mkdir -p "$(dirname "$dest")"
              cp "$src" "$dest"
              printf '%s  %s\\n' "$(sha256sum "$dest" | awk '{{print $1}}')" "$file" >> "$RUN_DIR/overlay_manifest.txt"
              echo "overlay_file=$file"
            elif [ -n "$OVERLAY_REV" ] && git_quick cat-file -e "$OVERLAY_REV:$file" 2>/dev/null; then
              dest="$SITE_ROOT/${{file#vllm/}}"
              mkdir -p "$(dirname "$dest")"
              git_quick show "$OVERLAY_REV:$file" > "$dest"
              printf '%s  %s\\n' "$(sha256sum "$dest" | awk '{{print $1}}')" "$file" >> "$RUN_DIR/overlay_manifest.txt"
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

        out = {{
            "wheel": {{"path": str(wheel), "sha256": sha(wheel.read_bytes())}},
            "site_root": str(site_root) if site_root else None,
            "files": {{}},
        }}
        with zipfile.ZipFile(wheel) as zf:
            names = set(zf.namelist())
            for file in files:
                item = {{"wheel_sha256": None, "site_sha256": None, "site_path": None}}
                if file in names:
                    item["wheel_sha256"] = sha(zf.read(file))
                if site_root:
                    site_path = site_root / file.removeprefix("vllm/")
                    if site_path.exists():
                        item["site_path"] = str(site_path)
                        item["site_sha256"] = sha(site_path.read_bytes())
                out["files"][file] = item
        (run_dir / "runtime_fingerprints.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False) + "\\n",
            encoding="utf-8",
        )
        PY

        health_ok() {{
          curl -fsS http://127.0.0.1:8001/health >/dev/null 2>&1
        }}

        START_SCRIPT="$TESTDATA/start_vllm.sh"
        if [ "$LOCKED_START_SCRIPT" = "1" ]; then
          START_SCRIPT="$RUN_DIR/start_vllm_locked.sh"
          cat > "$START_SCRIPT" <<'EOS'
        #!/usr/bin/env bash
        set -u
        set -o pipefail

        load_format_args=()
        if [ -n "${{GUARD_LOAD_FORMAT:-}}" ]; then
          load_format_args=(--load-format "$GUARD_LOAD_FORMAT")
        fi
        eager_args=()
        if [ "${{GUARD_ENFORCE_EAGER:-0}}" = "1" ]; then
          eager_args=(--enforce-eager)
        fi

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
            --default-chat-template-kwargs '{{"enable_thinking": false}}' \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice \
            --tool-call-parser qwen3_coder \
            "${{eager_args[@]}}" \
            "${{load_format_args[@]}}"
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
              awk '{{print $1}}' /tmp/guard_active_vllm.txt | xargs -r kill || true
              sleep 5
              awk '{{print $1}}' /tmp/guard_active_vllm.txt | xargs -r kill -9 || true
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

        cleanup() {{
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
        }}
        trap cleanup EXIT

        cd "$TESTDATA"
        echo "=== warmup ==="
        ./run_throughput.sh 4-8K 1 > "$RUN_DIR/raw/warmup_4-8K.log" 2>&1

        for bucket in $BUCKETS; do
          for rep in $(seq 1 "$REPETITIONS"); do
            echo "=== throughput bucket=$bucket rep=$rep ==="
            rm -f "./test/${{bucket}}_throughput/result.json"
            ./run_throughput.sh "$bucket" "$NUM_PROMPTS" > "$RUN_DIR/raw/throughput_${{bucket}}_rep${{rep}}.log" 2>&1
            cp "./test/${{bucket}}_throughput/result.json" "$RUN_DIR/throughput/${{bucket}}_rep${{rep}}.json"
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
        weights = {{"4-8K": 0.2, "8-16K": 0.5, "16-32K": 0.3}}

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

        throughput = {{}}
        for bucket in buckets:
            reps = []
            for path in sorted((run_dir / "throughput").glob(f"{{bucket}}_rep*.json")):
                data = json.loads(path.read_text(encoding="utf-8"))
                reps.append({{
                    "path": str(path),
                    "completed": data.get("completed"),
                    "failed": data.get("failed"),
                    "output_throughput": metric(data, "output_throughput"),
                    "p99_ttft_ms": metric(data, "p99_ttft_ms"),
                    "p99_tpot_ms": metric(data, "p99_tpot_ms"),
                    "duration": data.get("duration"),
                }})
            med = {{}}
            for key in ("output_throughput", "p99_ttft_ms", "p99_tpot_ms", "duration"):
                vals = [r[key] for r in reps if r.get(key) is not None]
                med[key] = statistics.median(vals) if vals else None
            throughput[bucket] = {{"repetitions": reps, "median": med}}

        accuracy = {{}}
        acc_log = run_dir / "raw" / "accuracy.log"
        if acc_log.exists():
            text = acc_log.read_text(encoding="utf-8", errors="ignore")
            for line in text.splitlines():
                m = re.match(r"\\|\\s*(hotpotqa|gov_report)\\s*\\|\\s*([^|]+)\\|\\s*([^|]+)\\|\\s*([^|]+)\\|\\s*([^|]+)\\|", line)
                if m:
                    accuracy[m.group(1)] = {{
                        "version": m.group(2).strip(),
                        "metric": m.group(3).strip(),
                        "mode": m.group(4).strip(),
                        "score": m.group(5).strip(),
                    }}
                    continue
                m = re.match(r"(retrieval_multi_point|aggregation_keyword_aggregation)_recalculated:\\s*([0-9.]+|NA)\\s*(?:\\(([^)]*)\\))?", line)
                if m:
                    accuracy[m.group(1)] = {{"metric": "recalculated_acc", "score": m.group(2), "count": m.group(3) or ""}}

        weighted = None
        if set(buckets) == set(weights) and all(throughput[b]["median"].get("output_throughput") is not None for b in buckets):
            weighted = sum(weights[b] * throughput[b]["median"]["output_throughput"] for b in buckets)

        summary = {{
            "run_id": run_id,
            "status": "pass",
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "repo": {{
                "kind": os.environ.get("REPO_KIND", ""),
                "path": repo_path,
                "branch": repo_branch,
                "head": repo_head,
                "status_path": str(run_dir / "repo_status.txt"),
                "log_path": str(run_dir / "repo_log.txt"),
            }},
            "wheel": {{"path": wheel, "sha256": wheel_sha}},
            "runtime_fingerprints_path": str(run_dir / "runtime_fingerprints.json") if (run_dir / "runtime_fingerprints.json").exists() else None,
            "overlay": {{
                "rev": os.environ.get("OVERLAY_REV", ""),
                "source_dir": os.environ.get("OVERLAY_SOURCE_DIR", ""),
                "manifest_path": str(run_dir / "overlay_manifest.txt") if (run_dir / "overlay_manifest.txt").exists() else None,
            }},
            "model_dir": model_dir,
            "protocol": {{
                "warmup": "run_throughput.sh 4-8K 1",
                "buckets": list(buckets),
                "repetitions": int(os.environ.get("REPETITIONS", "0") or 0),
                "num_prompts": int(os.environ.get("NUM_PROMPTS", "0") or 0),
                "accuracy": accuracy_mode,
                "locked_start_script": os.environ.get("LOCKED_START_SCRIPT", ""),
                "load_format": os.environ.get("LOAD_FORMAT", ""),
                "enforce_eager": os.environ.get("ENFORCE_EAGER", ""),
            }},
            "throughput": throughput,
            "weighted_output_throughput": weighted,
            "accuracy": accuracy,
        }}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
        print("=== summary ===")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        PY

        echo "completed_at=$(date -Is)"
        """
    )


def collect_artifacts(args: argparse.Namespace, cfg: dict[str, Any], local_dir: pathlib.Path) -> None:
    ssh_config = pathlib.Path(os.path.expanduser(cfg["generated_ssh_config"]))
    alias = cfg.get("auto_worker_alias", "xiandaobei-worker-auto")
    remote_dir = f"{cfg['remote_runs_dir']}/{args.run_id}"
    local_dir.mkdir(parents=True, exist_ok=True)
    items = [
        "summary.json",
        "runtime_fingerprints.json",
        "overlay_manifest.txt",
        "driver.log",
        "repo_status.txt",
        "repo_log.txt",
        "raw",
        "throughput",
        "guard.pid",
        "guard_remote.sh",
        "launch.log",
        "vllm_server.log",
        "start_vllm_locked.sh",
    ]
    if args.copy_accuracy_output:
        items.append("accuracy")
    for item in items:
        target = local_dir / item
        if target.exists():
            continue
        proc = run([
            "scp",
            "-F",
            str(ssh_config),
            "-r",
            f"{alias}:{remote_dir}/{item}",
            str(local_dir),
        ])
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
    readme = local_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            textwrap.dedent(
                f"""\
                # {args.run_id}

                Guard benchmark run.

                - Intent: fixed warm-container guard protocol for Round 0 / Round 1 comparisons.
                - Method: warmup once, then three throughput buckets x {args.repetitions} repetitions, median summary, plus accuracy mode `{args.accuracy}`.
                - Buckets: `{args.buckets}`
                - Overlay rev: `{args.overlay_rev or ""}`
                - Locked start script: `{args.locked_start_script}`
                - Load format: `{args.load_format or ""}`
                - Enforce eager: `{args.enforce_eager}`
                - Remote run dir: `{remote_dir}`
                - Local summary: `summary.json`
                - Raw logs: `raw/`
                - Throughput result JSONs: `throughput/`
                """
            ),
            encoding="utf-8",
        )


def remote_run_dir(args: argparse.Namespace, cfg: dict[str, Any]) -> str:
    return f"{cfg['remote_runs_dir']}/{args.run_id}"


def ssh_run(
    ssh_config: pathlib.Path,
    alias: str,
    command: str,
    *,
    stdin: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return run(["ssh", "-F", str(ssh_config), alias, command], stdin=stdin)


def upload_remote_script(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    ssh_config: pathlib.Path,
    alias: str,
    script: str,
) -> subprocess.CompletedProcess[str]:
    remote_dir = remote_run_dir(args, cfg)
    remote_script_path = f"{remote_dir}/guard_remote.sh"
    command = (
        f"mkdir -p {q(remote_dir)} && "
        f"cat > {q(remote_script_path)} && "
        f"chmod +x {q(remote_script_path)}"
    )
    return ssh_run(ssh_config, alias, command, stdin=script)


def start_remote_script(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    ssh_config: pathlib.Path,
    alias: str,
) -> subprocess.CompletedProcess[str]:
    remote_dir = remote_run_dir(args, cfg)
    command = textwrap.dedent(
        f"""\
        cd {q(remote_dir)} &&
        (nohup bash guard_remote.sh > launch.log 2>&1 < /dev/null & echo $! > guard.pid) &&
        cat guard.pid
        """
    )
    return ssh_run(ssh_config, alias, command)


def poll_remote_run(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    ssh_config: pathlib.Path,
    alias: str,
    local_dir: pathlib.Path,
) -> None:
    remote_dir = remote_run_dir(args, cfg)
    poll_log = local_dir / "poll.log"
    deadline = time.monotonic() + args.remote_timeout
    last_status = ""
    while True:
        command = textwrap.dedent(
            f"""\
            cd {q(remote_dir)} 2>/dev/null || exit 2
            if [ -f summary.json ]; then
              echo status=done
              exit 0
            fi
            pid="$(cat guard.pid 2>/dev/null || true)"
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
              echo status=running pid=$pid
            else
              echo status=dead pid=$pid
            fi
            tail -n 60 driver.log 2>/dev/null || true
            """
        )
        proc = ssh_run(ssh_config, alias, command)
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with poll_log.open("a", encoding="utf-8") as f:
            f.write(f"=== poll {stamp} rc={proc.returncode} ===\n")
            f.write(proc.stdout)
            f.write(proc.stderr)
            if not proc.stdout.endswith("\n") and proc.stdout:
                f.write("\n")

        if proc.returncode == 0 and "status=done" in proc.stdout.splitlines()[:1]:
            return
        if proc.returncode == 0 and proc.stdout.splitlines():
            status = proc.stdout.splitlines()[0]
            if status != last_status:
                print(status, flush=True)
                last_status = status
            if status.startswith("status=dead"):
                collect_artifacts(args, cfg, local_dir)
                raise SystemExit("remote guard benchmark exited before summary.json was written")
        elif proc.returncode != 0:
            print(f"poll ssh failed rc={proc.returncode}; retrying", flush=True)

        if time.monotonic() >= deadline:
            collect_artifacts(args, cfg, local_dir)
            raise SystemExit(f"remote guard benchmark timed out after {args.remote_timeout}s")
        time.sleep(args.poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "automation" / "config.json"))
    parser.add_argument("--run-id", default=now_run_id())
    parser.add_argument("--repo", choices=["competition", "baseline"], default="competition")
    parser.add_argument("--source-repo", help="Override remote source repo path for repo metadata")
    parser.add_argument("--wheel", help="Override remote wheel path or glob")
    parser.add_argument("--model-dir")
    parser.add_argument("--local-model-dir", default="/root/Qwen3.5-27B")
    parser.add_argument("--copy-model-local", action="store_true")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--buckets", default="4-8K,8-16K,16-32K", help="Comma-separated throughput buckets to run")
    parser.add_argument("--overlay-rev", help="Overlay selected Python files from this remote git revision after installing the wheel")
    parser.add_argument("--overlay-source-dir", help="Overlay selected Python files from this remote directory instead of git show")
    parser.add_argument("--accuracy", choices=["none", "smoke", "full"], default="full")
    parser.add_argument("--accuracy-rows", type=int)
    parser.add_argument("--copy-accuracy-output", action="store_true")
    parser.add_argument("--locked-start-script", action="store_true", help="Start vLLM from a generated script that preserves the locked competition CLI, including --max-model-len 32768")
    parser.add_argument("--load-format", help="Optional vLLM --load-format to use with --locked-start-script, for example runai_streamer")
    parser.add_argument("--enforce-eager", action="store_true", help="Pass vLLM --enforce-eager through the generated locked start script; diagnostic only")
    parser.add_argument("--server-start-timeout", type=int, default=900)
    parser.add_argument("--reuse-server", action="store_true")
    parser.add_argument("--keep-server", action="store_true")
    parser.add_argument("--stop-existing", action="store_true")
    parser.add_argument("--no-install", action="store_true")
    parser.add_argument("--env", action="append", default=[], help="Export KEY=VALUE in the remote benchmark process")
    parser.add_argument("--foreground", action="store_true", help="Run the remote benchmark in the SSH foreground")
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--remote-timeout", type=int, default=14400)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.repetitions < 1:
        raise SystemExit("--repetitions must be >= 1")
    if args.load_format and not args.locked_start_script:
        raise SystemExit("--load-format is only supported with --locked-start-script")
    if args.enforce_eager and not args.locked_start_script:
        raise SystemExit("--enforce-eager is only supported with --locked-start-script")

    cfg = load_config(pathlib.Path(args.config))
    ssh_config = pathlib.Path(os.path.expanduser(cfg["generated_ssh_config"]))
    alias = cfg.get("auto_worker_alias", "xiandaobei-worker-auto")
    script = remote_script(args, cfg)
    if args.dry_run:
        print(script)
        return

    local_dir = ROOT / "experiments" / args.run_id
    local_dir.mkdir(parents=True, exist_ok=True)
    if args.foreground:
        proc = run(["ssh", "-F", str(ssh_config), alias, "bash", "-s"], stdin=script)
        (local_dir / "remote.stdout.log").write_text(proc.stdout, encoding="utf-8", errors="ignore")
        (local_dir / "remote.stderr.log").write_text(proc.stderr, encoding="utf-8", errors="ignore")
        if proc.returncode != 0:
            collect_artifacts(args, cfg, local_dir)
            require_ok(proc, "remote guard benchmark")
    else:
        proc = upload_remote_script(args, cfg, ssh_config, alias, script)
        (local_dir / "upload.stdout.log").write_text(proc.stdout, encoding="utf-8", errors="ignore")
        (local_dir / "upload.stderr.log").write_text(proc.stderr, encoding="utf-8", errors="ignore")
        require_ok(proc, "upload remote guard script")
        proc = start_remote_script(args, cfg, ssh_config, alias)
        (local_dir / "start.stdout.log").write_text(proc.stdout, encoding="utf-8", errors="ignore")
        (local_dir / "start.stderr.log").write_text(proc.stderr, encoding="utf-8", errors="ignore")
        require_ok(proc, "start remote guard script")
        print(f"remote_guard_pid={proc.stdout.strip()}", flush=True)
        poll_remote_run(args, cfg, ssh_config, alias, local_dir)
    collect_artifacts(args, cfg, local_dir)
    print(f"wrote {local_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
