# Xiandaobei Fast Iteration Protocol

Use this reference before running Xiandaobei remote experiments, benchmarks,
container work, or submission preflight. The point is to make speed automatic:
prefer tool defaults, generated config, locked startup, background jobs, and
fixed result formats over per-session memory.

## Source Of Truth

Start from `/Users/keynary/Code/xiandaobei/meta` locally and
`/public/home/xdzs2026_c166/meta` on SCNet. Before claiming current project
state, pull or otherwise verify the active git state, but do not merge over a
dirty worktree. Read these files when present:

- `AGENTS.md`: repo boundaries plus efficient-execution rules.
- `plans/roadmap.md`: round discipline and controlled comparison policy.
- `memory/10-project.md`: scoring formula and red/green lines.
- `memory/20-env.md`: SCNet connection, container lifetime, proxy, wheel, and
  model-directory traps.
- `memory/30-codeworkflow.md`: authoritative worktree, wheel, and guard口径
  corrections.
- `memory/50-arch-bottleneck.md`: architecture and bottleneck facts.

If a file exists only in another local worktree, remote clone, or branch, name
that fact explicitly instead of pretending the current checkout has it.

## A. SSH Reuse First

Always make `scripts/scnetctl.py attach` regenerate
`~/.ssh/xiandaobei.generated.conf`; do not hand-maintain worker IPs. The
generated login and worker hosts must include:

```sshconfig
ControlMaster auto
ControlPath ~/.ssh/cm-xiandaobei-%C
ControlPersist 10m
```

Acceptance evidence for connection work:

```bash
python3 scripts/scnetctl.py attach
ssh -F ~/.ssh/xiandaobei.generated.conf -O check xiandaobei-worker-auto
time ssh -F ~/.ssh/xiandaobei.generated.conf xiandaobei-worker-auto true
time ssh -F ~/.ssh/xiandaobei.generated.conf xiandaobei-worker-auto true
```

The control check should say `Master running`; the second timed `true` should
be under 1s. Batch multiple remote steps into one heredoc over the generated
alias when possible.

## B. Locked RunAI Startup For Rebuilt Containers

When `/root/Qwen3.5-27B` is absent, prefer the validated direct-from-home startup
over copying the 52G model into `/root`. The working mechanism is Run:ai Model
Streamer: `--load-format runai_streamer` parallel/streaming safetensors loading.
It is faster than the 40-60 minute explicit copy path, but it is not a 300s
full-ready guarantee.

Do not use the user-provided raw command by memory: it omitted
`--max-model-len 32768` and vLLM resolved `max_seq_len=262144`. Any such run is
invalid for competition timing.

Use `scripts/guard_bench.py` to generate the locked start script:

```bash
python3 scripts/guard_bench.py \
  --run-id guard-<rev>-locked-srcdir-fullsmoke10-<date> \
  --repo competition \
  --overlay-source-dir /root/overlay-<rev>-locked \
  --locked-start-script \
  --load-format runai_streamer \
  --env VLLM_CACHE_ROOT=/public/home/xdzs2026_c166/vllm_cache/vllm_cache \
  --env TRITON_CACHE_DIR=/public/home/xdzs2026_c166/vllm_cache/triton_cache \
  --accuracy smoke \
  --accuracy-rows 10 \
  --server-start-timeout 1200 \
  --stop-existing \
  --poll-interval 30 \
  --remote-timeout 14400
```

Acceptance evidence:

```bash
rg -n "max_model_len=32768|max_seq_len=32768|load_format=runai_streamer" \
  experiments/<run_id>/vllm_server.log experiments/<run_id>/README.md
```

Validated anchors:

- `experiments/runai-startup-probe-locked-20260707-1324/`: `/health` in `725s`
  from `vllm serve`; RunAI streamed `51.7 GiB` in `341.16s`.
- `experiments/guard-d29e9db3-locked-srcdir-fullsmoke10-20260707-1355/`:
  corrected locked startup with `max_seq_len=32768` and `load_format=runai_streamer`;
  stopped early after user reprioritized R1, so it is not a sign-table row.

## C. Overlay From Root, Not Shared Git, For Long Guards

For commit A/B overlays, avoid repeated `git show` from the shared
`/public/home` repository during long guards. Prepare a source overlay once into
`/root` and pass `--overlay-source-dir`:

```bash
ssh -F ~/.ssh/xiandaobei.generated.conf xiandaobei-worker-auto '
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
REPO=/public/home/xdzs2026_c166/vllm_cscc_competition
REV=<rev>
DEST=/root/overlay-$REV-locked
rm -rf "$DEST"; mkdir -p "$DEST"
FILES="vllm/model_executor/models/qwen3_5.py vllm/model_executor/models/qwen3_next.py vllm/model_executor/layers/activation.py vllm/model_executor/layers/fla/ops/chunk.py vllm/model_executor/layers/fla/ops/chunk_o.py vllm/v1/attention/ops/triton_unified_attention.py vllm/version.py"
timeout 180 git -C "$REPO" archive "$REV" -- $FILES | tar -x -C "$DEST"
find "$DEST" -type f -maxdepth 8 -print -exec sha256sum {} \;
'
```

This avoids the observed shared-NFS stall in
`experiments/guard-d29e9db3-locked-runai-fullsmoke10-20260707-1344/`.

## D. Container Pool Before Parallel Work

The maintained pool is viable only if container creation can be submitted from
the login node with `sbatch`. Probe from SCNet before filling or trusting
`submit_job()`:

```bash
scontrol show job <jobid>
sacct -j <jobid> --format=JobID,JobName,Partition,SubmitLine%200
squeue -u "$USER" -o "%.18i %.30j %.9P %.8T %R"
sacctmgr show assoc user="$USER" format=Partition,MaxJobs,GrpTRES,MaxSubmitJobs
```

If an equivalent `sbatch` command is found, put it in
`scripts/pool_manager.py::submit_job()` and run the manager on the login node.

Keep the default light: `POOL_K=1`, `POOL_B=2`. Pending is free queue pressure;
running burns GPU-hours. Raise `POOL_K` only while actively screening parallel
candidates. If only Chrome can create containers, report the downgrade as a real
blocker instead of faking an automatic pool.

## E. Smoke Vs Full Accuracy

Use precision tiers deliberately:

- `--accuracy smoke`: daily regression, 10 rows per accuracy class.
- `--accuracy full`: 109-row full accuracy, only at round close or before
  submission.
- `--accuracy none`: throughput-only diagnosis when accuracy is irrelevant.

If using `scripts/guard_bench.py`, the same `--accuracy smoke/full` meaning
applies. Record the wall time and the experiment anchor.

## F. Background Long Jobs

vLLM startup, wheel install, compile, benchmark, full accuracy, and pool manager
loops must not monopolize an interactive shell. Prefer existing tool behavior
that uses `nohup` and logs, or launch explicitly:

```bash
nohup <long-command> >logs/<name>.log 2>&1 &
```

Then poll meaningful artifacts: health endpoint, PID, log tail, `summary.json`,
or Feishu/job hooks. Do not sit in foreground sleep loops posting only waiting
updates.

## G. Controlled Comparison Format

Parallel screening is allowed only with a local baseline inside every
container. Report relative deltas, not cross-container absolute throughput:

```text
container=<id> bucket=8-16K baseline=7.2317 candidate=7.4880 delta=+3.54%
```

Local proxy values are for same-container A/B and sanity gates only. Never plug
local absolute values into an official score formula or compare container A
absolute throughput against container B absolute throughput.

## H. Lightweight Habits

- Validate a batch once (`py_compile`, `node --check`, `git status`), not after
  every microscopic edit.
- Stop at the declared acceptance criterion; record the anchor and move on.
- On macOS use `gtimeout` or PID+sleep+kill instead of GNU `timeout`.
- New worktrees may lack upstream; use `git push -u` or explicit remote branch.
- Read paths from `memory/` and `references/project-map.md`; do not rediscover
  stable paths with broad `find` loops.
