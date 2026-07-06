#!/usr/bin/env python3
"""Local orchestrator for SCNet xiandaobei container experiments.

The script deliberately keeps browser/container creation separate from the SSH
experiment runner. It can fully drive a running container, and it emits a clear
machine-readable state when Chrome startup is needed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import textwrap
import time
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "automation" / "config.json"


class ScnetError(RuntimeError):
    def __init__(self, message: str, code: int = 1):
        super().__init__(message)
        self.code = code


def expand_path(value: str) -> pathlib.Path:
    return pathlib.Path(os.path.expandvars(os.path.expanduser(value)))


def load_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def now_id(prefix: str) -> str:
    return f"{prefix}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"


def run_local(
    argv: list[str],
    *,
    input_text: str | None = None,
    timeout: int | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(
            argv,
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        cmd = " ".join(shlex.quote(x) for x in argv)
        raise ScnetError(f"command timed out after {timeout}s: {cmd}", 124) from exc
    if check and proc.returncode != 0:
        cmd = " ".join(shlex.quote(x) for x in argv)
        raise ScnetError(
            f"command failed ({proc.returncode}): {cmd}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
            proc.returncode,
        )
    return proc


def ssh_cmd(
    alias: str,
    remote_cmd: str,
    *,
    timeout: int | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return run_local(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectionAttempts=1",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=1",
            "-o",
            f"ConnectTimeout={CONFIG['ssh_connect_timeout_s']}",
            alias,
            remote_cmd,
        ],
        timeout=timeout,
        check=check,
    )


def worker_ssh_argv(remote_cmd: str) -> list[str]:
    if CONFIG.get("_generated_ssh_config"):
        return [
            "ssh",
            "-F",
            CONFIG["_generated_ssh_config"],
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={CONFIG['ssh_connect_timeout_s']}",
            CONFIG["auto_worker_alias"],
            remote_cmd,
        ]
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={CONFIG['ssh_connect_timeout_s']}",
        CONFIG["worker_alias"],
        remote_cmd,
    ]


def worker_scp_argv(remote_path: str, local_path: pathlib.Path) -> list[str]:
    base = ["scp", "-q", "-o", f"ConnectTimeout={CONFIG['ssh_connect_timeout_s']}"]
    if CONFIG.get("_generated_ssh_config"):
        base += ["-F", CONFIG["_generated_ssh_config"]]
        alias = CONFIG["auto_worker_alias"]
    else:
        alias = CONFIG["worker_alias"]
    return base + ["-r", f"{alias}:{remote_path}", str(local_path)]


def transient_ssh_failure(proc: subprocess.CompletedProcess[str] | None, exc: Exception | None = None) -> bool:
    if isinstance(exc, ScnetError) and exc.code in {124, 255}:
        return True
    if proc is None:
        return False
    text = f"{proc.stdout}\n{proc.stderr}".lower()
    return proc.returncode == 255 or any(
        needle in text
        for needle in [
            "timed out",
            "banner exchange",
            "connection reset",
            "connection closed",
            "not responding",
        ]
    )


def worker_run(
    remote_cmd: str,
    *,
    input_text: str | None = None,
    timeout: int | None = None,
    attempts: int = 4,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    last_proc: subprocess.CompletedProcess[str] | None = None
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            proc = run_local(
                worker_ssh_argv(remote_cmd),
                input_text=input_text,
                timeout=timeout,
                check=False,
            )
            last_proc = proc
            if proc.returncode == 0 or not transient_ssh_failure(proc):
                if check and proc.returncode != 0:
                    raise ScnetError(
                        f"worker command failed ({proc.returncode}): {remote_cmd}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
                        proc.returncode,
                    )
                return proc
        except ScnetError as exc:
            last_exc = exc
            if not transient_ssh_failure(None, exc):
                raise
        if attempt < attempts:
            time.sleep(min(2 * attempt, 8))
    if last_proc is not None:
        if check:
            raise ScnetError(
                f"worker command failed after retries ({last_proc.returncode}): {remote_cmd}\nSTDOUT:\n{last_proc.stdout}\nSTDERR:\n{last_proc.stderr}",
                last_proc.returncode,
            )
        return last_proc
    if isinstance(last_exc, ScnetError):
        raise last_exc
    raise ScnetError(f"worker command failed after retries: {remote_cmd}", 1)


def scp_from(alias: str, remote_path: str, local_path: pathlib.Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    run_local(
        [
            "scp",
            "-q",
            "-o",
            f"ConnectTimeout={CONFIG['ssh_connect_timeout_s']}",
            "-r",
            f"{alias}:{remote_path}",
            str(local_path),
        ]
    )


def resolve_job() -> dict[str, Any] | None:
    user = CONFIG["user"]
    fmt = "%.18i|%.9P|%.60j|%.2t|%.10M|%.10l|%.30R"
    cmd = f"squeue -u {shlex.quote(user)} -h -o {shlex.quote(fmt)} || true"
    proc = None
    last_error: ScnetError | None = None
    for attempt in range(1, 5):
        try:
            proc = ssh_cmd(CONFIG["login_alias"], cmd, check=False, timeout=20)
            if proc.returncode == 0:
                break
            last_error = ScnetError(
                f"cannot query squeue:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
                proc.returncode,
            )
        except ScnetError as exc:
            last_error = exc
        if attempt < 4:
            time.sleep(2)
    if proc is None:
        raise last_error or ScnetError("cannot query squeue", 1)
    if proc.returncode != 0:
        raise last_error or ScnetError(f"cannot query squeue:\n{proc.stderr}", proc.returncode)

    jobs: list[dict[str, str]] = []
    for line in proc.stdout.splitlines():
        parts = line.split("|")
        if len(parts) != 7:
            continue
        job_id, partition, name, state, runtime, limit, node = [p.strip() for p in parts]
        if not job_id or node in {"", "None", "(None)", "ReqNodeNotAvail"}:
            continue
        jobs.append(
            {
                "job_id": job_id,
                "partition": partition,
                "name": name,
                "state": state,
                "runtime": runtime,
                "limit": limit,
                "node": node,
            }
        )
    running = [j for j in jobs if j["state"] == "R"]
    if not running:
        return None
    # Prefer the newest Slurm job id among running container jobs.
    return sorted(running, key=lambda j: int(re.sub(r"\D", "", j["job_id"]) or 0))[-1]


def resolve_container_ip(job: dict[str, Any]) -> dict[str, Any]:
    job_id = job["job_id"]
    node = job["node"]
    remote = textwrap.dedent(
        f"""
        set -e
        node={shlex.quote(node)}
        job={shlex.quote(job_id)}
        ssh -o BatchMode=yes -o ConnectTimeout=8 "$node" "docker inspect -f '{{{{range .NetworkSettings.Networks}}}}{{{{.IPAddress}}}}{{{{end}}}}' ${{job}}_${{node}} 2>/dev/null"
        """
    ).strip()
    proc = None
    last_error: ScnetError | None = None
    for attempt in range(1, 5):
        try:
            proc = ssh_cmd(CONFIG["login_alias"], remote, check=False, timeout=30)
            if proc.returncode == 0 and proc.stdout.strip():
                break
            last_error = ScnetError(
                f"cannot resolve container ip for job={job_id} node={node}\nSTDOUT:\n{proc.stdout if proc else ''}\nSTDERR:\n{proc.stderr if proc else ''}",
                proc.returncode if proc else 1,
            )
        except ScnetError as exc:
            last_error = exc
        if attempt < 4:
            time.sleep(2)
    if proc is None:
        raise last_error or ScnetError(f"cannot resolve container ip for job={job_id}", 1)
    ip = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if proc.returncode != 0 or not re.match(r"^\d+\.\d+\.\d+\.\d+$", ip):
        raise ScnetError(
            f"cannot resolve container ip for job={job_id} node={node}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
            proc.returncode or 1,
        )
    out = dict(job)
    out["container_name"] = f"{job_id}_{node}"
    out["container_ip"] = ip
    return out


def write_generated_ssh_config(resolved: dict[str, Any]) -> pathlib.Path:
    path = expand_path(CONFIG["generated_ssh_config"])
    path.parent.mkdir(parents=True, exist_ok=True)
    login_identity = CONFIG["login_identity_file"]
    worker_identity = CONFIG["worker_identity_file"]
    known_hosts = CONFIG["known_hosts_file"]
    content = f"""# Generated by scnetctl.py. Safe to overwrite.
Host {CONFIG['login_alias']}
  HostName {CONFIG['login_host']}
  Port {CONFIG['login_port']}
  User {CONFIG['user']}
  IdentityFile {login_identity}
  IdentitiesOnly yes
  ControlMaster auto
  ControlPath ~/.ssh/cm-xiandaobei-%C
  ControlPersist 10m
  StrictHostKeyChecking accept-new
  UserKnownHostsFile {known_hosts}
  HostKeyAlias xiandaobei-login-{CONFIG['login_port']}

Host {CONFIG['auto_worker_alias']}
  HostName {resolved['container_ip']}
  Port 22
  User root
  IdentityFile {worker_identity}
  IdentitiesOnly yes
  ControlMaster auto
  ControlPath ~/.ssh/cm-xiandaobei-%C
  ControlPersist 10m
  StrictHostKeyChecking accept-new
  UserKnownHostsFile {known_hosts}
  HostKeyAlias xiandaobei-worker-{resolved['job_id']}
  ProxyCommand ssh -F {path} {CONFIG['login_alias']} ssh -o BatchMode=yes -o ConnectTimeout=10 -W %h:%p {resolved['node']}
"""
    path.write_text(content, encoding="utf-8")
    os.chmod(path, 0o600)
    return path


def status(as_json: bool) -> dict[str, Any]:
    state: dict[str, Any] = {
        "checked_at": dt.datetime.now().isoformat(timespec="seconds"),
        "job": None,
        "container": None,
        "worker": None,
    }
    job = resolve_job()
    state["job"] = job
    if job:
        try:
            state["container"] = resolve_container_ip(job)
        except Exception as exc:  # noqa: BLE001 - status should be best effort.
            state["container_error"] = str(exc)

    worker_cmd = textwrap.dedent(
        f"""
        set +e
        echo HOST=$(hostname)
        echo DATE=$(date '+%Y-%m-%dT%H:%M:%S%z')
        echo ACTIVE_PROCESSES_BEGIN
        pgrep -af {shlex.quote(CONFIG['guard_process_pattern'])} | grep -v -E 'pgrep -af|bash -c|bash -s|scnetctl' || true
        echo ACTIVE_PROCESSES_END
        echo HEALTH_BEGIN
        unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
        curl --noproxy '*' -fsS --max-time 3 http://127.0.0.1:8001/health 2>&1
        echo
        echo HEALTH_END
        """
    ).strip()
    proc = worker_run(worker_cmd, check=False, timeout=20)
    state["worker"] = {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    if as_json:
        print(json.dumps(state, ensure_ascii=False, indent=2))
    else:
        print(f"checked_at: {state['checked_at']}")
        if job:
            print(
                "job: {job_id} {state} {runtime}/{limit} node={node} name={name}".format(
                    **job
                )
            )
        else:
            print("job: none")
        if state.get("container"):
            c = state["container"]
            print(f"container: {c['container_name']} ip={c['container_ip']}")
        if proc.returncode == 0:
            print("worker: reachable")
            print(proc.stdout.strip())
        else:
            print("worker: unreachable")
            print(proc.stderr.strip())
    return state


def ensure_attached(write_config: bool = True, test: bool = True) -> dict[str, Any]:
    job = resolve_job()
    if not job:
        raise ScnetError("no running SCNet container job; Chrome startup is required", 20)
    resolved = resolve_container_ip(job)
    config_path = None
    if write_config:
        config_path = write_generated_ssh_config(resolved)
        resolved["generated_ssh_config"] = str(config_path)
        CONFIG["_generated_ssh_config"] = str(config_path)
    if test:
        test_cmd = "hostname; date '+%Y-%m-%dT%H:%M:%S%z'"
        proc = run_local(worker_ssh_argv(test_cmd), check=False, timeout=20)
        resolved["worker_alias_test"] = {
            "alias": CONFIG.get("auto_worker_alias") if config_path else CONFIG["worker_alias"],
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    return resolved


def load_task(name: str) -> dict[str, Any]:
    path = ROOT / "automation" / "tasks" / f"{name}.json"
    if not path.exists():
        raise ScnetError(f"unknown task {name!r}: {path} not found", 2)
    return load_json(path)


def choose_wheel(repo: str) -> str:
    key = f"{repo}_wheel_glob"
    if key not in CONFIG:
        raise ScnetError(f"repo must be baseline or competition, got {repo!r}", 2)
    return CONFIG[key]


def remote_baseline_script(
    *,
    run_id: str,
    task: dict[str, Any],
    force: bool,
    reuse_server: bool,
) -> str:
    repo = task.get("repo", "baseline")
    wheel_glob = choose_wheel(repo)
    num_prompts = str(task.get("num_prompts", CONFIG["default_num_prompts"]))
    dataset = str(task.get("throughput_dataset", "all"))
    accuracy = str(task.get("accuracy", "none"))
    run_dir = f"{CONFIG['remote_runs_dir']}/{run_id}"
    guard_pattern = CONFIG["guard_process_pattern"]
    return textwrap.dedent(
        f"""\
        set -u
        set -o pipefail

        RUN_ID={shlex.quote(run_id)}
        RUN_DIR={shlex.quote(run_dir)}
        WORK="$RUN_DIR/work"
        TESTDATA={shlex.quote(CONFIG['testdata_dir'])}
        MODEL_DIR={shlex.quote(CONFIG['model_dir'])}
        WHEEL_GLOB={shlex.quote(wheel_glob)}
        DATASET={shlex.quote(dataset)}
        NUM_PROMPTS={shlex.quote(num_prompts)}
        ACCURACY={shlex.quote(accuracy)}
        FORCE={1 if force else 0}
        REUSE_SERVER={1 if reuse_server else 0}
        GUARD_PATTERN={shlex.quote(guard_pattern)}

        mkdir -p "$RUN_DIR" "$WORK"
        exec > >(tee -a "$RUN_DIR/driver.log") 2>&1

        echo "=== scnetctl baseline run ==="
        echo "run_id=$RUN_ID"
        echo "host=$(hostname)"
        echo "started_at=$(date -Is)"
        echo "repo={repo}"
        echo "dataset=$DATASET num_prompts=$NUM_PROMPTS accuracy=$ACCURACY"

        unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
        export no_proxy=127.0.0.1,localhost
        export NO_PROXY=127.0.0.1,localhost
        export MODEL_DIR

        active="$(pgrep -af "$GUARD_PATTERN" | grep -v -E 'pgrep -af|bash -c|bash -s|scnetctl' || true)"
        if [ -n "$active" ] && [ "$REUSE_SERVER" != "1" ] && [ "$FORCE" != "1" ]; then
          echo "GUARD_ACTIVE_PROCESSES"
          printf '%s\\n' "$active"
          cat > "$RUN_DIR/summary.json" <<JSON
        {{"run_id":"$RUN_ID","status":"guarded","reason":"active_processes","repo":"{repo}"}}
        JSON
          exit 41
        fi

        echo "=== install wheel ==="
        python3 -m pip install --no-deps $WHEEL_GLOB

        echo "=== prepare isolated workdir ==="
        cp "$TESTDATA/run_throughput.sh" "$WORK/run_throughput.sh"
        cp "$TESTDATA/run_accuracy.sh" "$WORK/run_accuracy.sh"
        chmod +x "$WORK/run_throughput.sh" "$WORK/run_accuracy.sh"
        for f in 4-8K_throughput.jsonl 8-16K_throughput.jsonl 16-32K_throughput.jsonl hotpotqa.jsonl gov_report.jsonl retrieval_multi_point.jsonl aggregation_keyword_aggregation.jsonl; do
          ln -sf "$TESTDATA/$f" "$WORK/$f"
        done

        server_pid=""
        cleanup() {{
          rc=$?
          if [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null; then
            echo "=== stopping server pid=$server_pid ==="
            kill "$server_pid" 2>/dev/null || true
            sleep 3
            kill -9 "$server_pid" 2>/dev/null || true
          fi
          echo "=== write summary rc=$rc ==="
          python3 - "$RUN_DIR" "$RUN_ID" "$rc" "{repo}" <<'PY'
        import glob, json, os, sys, time
        run_dir, run_id, rc, repo = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
        summary = {{
            "run_id": run_id,
            "repo": repo,
            "status": "pass" if rc == 0 else "fail",
            "returncode": rc,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "throughput": {{}},
            "weighted_output_throughput": None,
        }}
        weights = {{"4-8K": 0.2, "8-16K": 0.5, "16-32K": 0.3}}
        for path in glob.glob(os.path.join(run_dir, "work", "test", "*_throughput", "result.json")):
            label = os.path.basename(os.path.dirname(path)).replace("_throughput", "")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                summary["throughput"][label] = {{"error": str(exc), "path": path}}
                continue
            summary["throughput"][label] = {{
                "path": path,
                "completed": data.get("completed"),
                "failed": data.get("failed"),
                "output_throughput": data.get("output_throughput"),
                "p99_ttft_ms": data.get("p99_ttft_ms"),
                "p99_tpot_ms": data.get("p99_tpot_ms"),
                "duration": data.get("duration"),
            }}
        if all(k in summary["throughput"] and isinstance(summary["throughput"][k].get("output_throughput"), (int, float)) for k in weights):
            summary["weighted_output_throughput"] = sum(summary["throughput"][k]["output_throughput"] * w for k, w in weights.items())
        with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.write("\\n")
        PY
          exit "$rc"
        }}
        trap cleanup EXIT

        if [ "$REUSE_SERVER" = "1" ]; then
          echo "=== reuse existing server ==="
          curl --noproxy '*' -fsS --max-time 5 http://127.0.0.1:8001/health
        else
          echo "=== start vllm server ==="
          (cd "$RUN_DIR" && nohup bash "$TESTDATA/start_vllm.sh" > "$RUN_DIR/vllm_server.log" 2>&1 & echo $! > "$RUN_DIR/vllm_server.pid")
          server_pid="$(cat "$RUN_DIR/vllm_server.pid")"
          echo "server_pid=$server_pid"
          deadline=$((SECONDS + {int(CONFIG['server_start_timeout_s'])}))
          until curl --noproxy '*' -fsS --max-time 5 http://127.0.0.1:8001/health; do
            if ! kill -0 "$server_pid" 2>/dev/null; then
              echo "server died before health check"
              tail -120 "$RUN_DIR/vllm_server.log" || true
              exit 42
            fi
            if [ "$SECONDS" -ge "$deadline" ]; then
              echo "server health timeout"
              tail -120 "$RUN_DIR/vllm_server.log" || true
              exit 43
            fi
            sleep 20
          done
          echo
        fi

        echo "=== run throughput ==="
        (cd "$WORK" && bash ./run_throughput.sh "$DATASET" "$NUM_PROMPTS") > "$RUN_DIR/throughput.log" 2>&1

        if [ "$ACCURACY" = "smoke" ]; then
          echo "=== run accuracy smoke ==="
          (cd "$WORK" && bash ./run_accuracy.sh all 1) > "$RUN_DIR/accuracy.log" 2>&1
        elif [ "$ACCURACY" = "full" ]; then
          echo "=== run accuracy full ==="
          (cd "$WORK" && bash ./run_accuracy.sh all) > "$RUN_DIR/accuracy.log" 2>&1
        else
          echo "=== skip accuracy ==="
        fi

        echo "completed_at=$(date -Is)"
        """
    )


def collect_run(run_id: str, remote_run_dir: str) -> pathlib.Path:
    exp_dir = ROOT / "experiments" / run_id
    if exp_dir.exists():
        raise ScnetError(f"local experiment dir already exists: {exp_dir}", 2)
    exp_dir.mkdir(parents=True)
    run_local(worker_scp_argv(f"{remote_run_dir}/summary.json", exp_dir / "summary.json"))
    for name in ["driver.log", "throughput.log", "accuracy.log", "vllm_server.log"]:
        proc = run_local(worker_scp_argv(f"{remote_run_dir}/{name}", exp_dir / name), check=False)
        if proc.returncode != 0:
            continue
    summary = load_json(exp_dir / "summary.json")
    readme = [
        f"# {run_id}",
        "",
        f"- remote_run_dir: `{remote_run_dir}`",
        f"- status: `{summary.get('status')}`",
        f"- repo: `{summary.get('repo')}`",
        f"- weighted_output_throughput: `{summary.get('weighted_output_throughput')}`",
        "",
        "## Throughput",
        "",
    ]
    for label in ["4-8K", "8-16K", "16-32K"]:
        row = summary.get("throughput", {}).get(label, {})
        readme.append(
            f"- {label}: output={row.get('output_throughput')} p99_ttft_ms={row.get('p99_ttft_ms')} p99_tpot_ms={row.get('p99_tpot_ms')} completed={row.get('completed')} failed={row.get('failed')}"
        )
    (exp_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    return exp_dir


def upload_remote_driver(remote_run_dir: str, script: str) -> None:
    quoted_dir = shlex.quote(remote_run_dir)
    cmd = f"mkdir -p {quoted_dir} && cat > {quoted_dir}/driver.sh && chmod +x {quoted_dir}/driver.sh"
    worker_run(cmd, input_text=script, timeout=60)


def launch_remote_driver(remote_run_dir: str) -> str:
    quoted_dir = shlex.quote(remote_run_dir)
    cmd = (
        f"cd {quoted_dir} && "
        "nohup bash ./driver.sh > launcher.log 2>&1 < /dev/null & "
        "echo $! > driver.pid && cat driver.pid"
    )
    proc = worker_run(cmd, timeout=30)
    pid = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if not pid.isdigit():
        raise ScnetError(f"failed to launch remote driver; pid output={proc.stdout!r}", 1)
    return pid


def wait_remote_run(remote_run_dir: str, *, poll_interval: int, timeout_s: int) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last_status = ""
    while time.time() < deadline:
        quoted_dir = shlex.quote(remote_run_dir)
        cmd = textwrap.dedent(
            f"""
            set +e
            if [ -f {quoted_dir}/summary.json ]; then
              echo SUMMARY_READY
              cat {quoted_dir}/summary.json
              exit 0
            fi
            echo SUMMARY_PENDING
            [ -f {quoted_dir}/driver.pid ] && echo driver_pid=$(cat {quoted_dir}/driver.pid)
            [ -f {quoted_dir}/driver.log ] && tail -20 {quoted_dir}/driver.log
            [ -f {quoted_dir}/launcher.log ] && tail -20 {quoted_dir}/launcher.log
            """
        ).strip()
        proc = worker_run(cmd, timeout=45, attempts=4, check=False)
        output = proc.stdout.strip()
        if "SUMMARY_READY" in output:
            json_text = output.split("SUMMARY_READY", 1)[1].strip()
            return json.loads(json_text)
        if output and output != last_status:
            print(output, flush=True)
            last_status = output
        time.sleep(poll_interval)
    raise ScnetError(f"remote run timed out waiting for summary: {remote_run_dir}", 124)


def run_task(args: argparse.Namespace) -> None:
    task = load_task(args.task)
    if args.num_prompts is not None:
        task["num_prompts"] = args.num_prompts
    if args.repo is not None:
        task["repo"] = args.repo
    if args.accuracy is not None:
        task["accuracy"] = args.accuracy

    resolved = ensure_attached(write_config=True, test=True)
    print(json.dumps({"attached": resolved}, ensure_ascii=False, indent=2))

    run_id = args.run_id or now_id(task["name"])
    remote_run_dir = f"{CONFIG['remote_runs_dir']}/{run_id}"
    script = remote_baseline_script(
        run_id=run_id,
        task=task,
        force=args.force,
        reuse_server=args.reuse_server,
    )
    if args.dry_run:
        print(script)
        return

    print(f"remote_run_dir={remote_run_dir}", flush=True)
    upload_remote_driver(remote_run_dir, script)
    pid = launch_remote_driver(remote_run_dir)
    print(f"remote_driver_pid={pid}", flush=True)
    summary: dict[str, Any] | None = None
    if not args.no_monitor:
        summary = wait_remote_run(
            remote_run_dir,
            poll_interval=args.poll_interval,
            timeout_s=args.run_timeout,
        )
        print(json.dumps({"remote_summary": summary}, ensure_ascii=False, indent=2), flush=True)
    exp_dir = None
    try:
        exp_dir = collect_run(run_id, remote_run_dir)
        print(f"collected={exp_dir}")
    except Exception as exc:  # noqa: BLE001 - preserve remote failure first.
        print(f"collect_failed={exc}", file=sys.stderr)
    if summary and summary.get("returncode") not in {0, None}:
        raise ScnetError(f"remote run failed with rc={summary.get('returncode')}", int(summary.get("returncode") or 1))
    if exp_dir:
        print(f"experiment_dir={exp_dir}")


def cmd_attach(args: argparse.Namespace) -> None:
    resolved = ensure_attached(write_config=not args.no_write, test=not args.no_test)
    print(json.dumps(resolved, ensure_ascii=False, indent=2))


def cmd_start(args: argparse.Namespace) -> None:
    job = resolve_job()
    if job:
        print(json.dumps({"status": "already_running", "job": job}, ensure_ascii=False, indent=2))
        return
    print(
        json.dumps(
            {
                "status": "needs_chrome_start",
                "reason": "No running Slurm container job was found. Use the Codex Chrome startup adapter or start an existing SCNet instance in Chrome, then rerun scnetctl.py attach.",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    raise ScnetError("container startup requires Chrome adapter", 20)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SCNet xiandaobei automation controller")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("status")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=lambda a: status(a.json))

    p = sub.add_parser("start")
    p.set_defaults(func=cmd_start)

    p = sub.add_parser("attach")
    p.add_argument("--no-write", action="store_true")
    p.add_argument("--no-test", action="store_true")
    p.set_defaults(func=cmd_attach)

    p = sub.add_parser("run")
    p.add_argument("task", choices=["baseline-smoke", "baseline-full"])
    p.add_argument("--run-id")
    p.add_argument("--repo", choices=["baseline", "competition"])
    p.add_argument("--num-prompts", type=int)
    p.add_argument("--accuracy", choices=["none", "smoke", "full"])
    p.add_argument("--reuse-server", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-monitor", action="store_true")
    p.add_argument("--poll-interval", type=int, default=60)
    p.add_argument("--run-timeout", type=int, default=7200)
    p.set_defaults(func=run_task)

    p = sub.add_parser("auto")
    p.add_argument("task", choices=["baseline-smoke", "baseline-full"])
    p.add_argument("--run-id")
    p.add_argument("--repo", choices=["baseline", "competition"])
    p.add_argument("--num-prompts", type=int)
    p.add_argument("--accuracy", choices=["none", "smoke", "full"])
    p.add_argument("--reuse-server", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-monitor", action="store_true")
    p.add_argument("--poll-interval", type=int, default=60)
    p.add_argument("--run-timeout", type=int, default=7200)
    p.set_defaults(func=run_task)
    return parser


CONFIG: dict[str, Any] = {}


def main(argv: list[str] | None = None) -> int:
    global CONFIG
    parser = build_parser()
    args = parser.parse_args(argv)
    CONFIG = load_json(expand_path(args.config))
    generated = expand_path(CONFIG["generated_ssh_config"])
    if generated.exists():
        CONFIG["_generated_ssh_config"] = str(generated)
    try:
        args.func(args)
    except ScnetError as exc:
        print(f"scnetctl: {exc}", file=sys.stderr)
        return exc.code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
