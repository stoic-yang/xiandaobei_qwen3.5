# guard-d29e9db3-hotserver-nooverlay-fullsmoke10-20260707-1011

> README stub written by opencode on 2026-07-07 (repo consolidation pass).
> The run was launched by Codex; Codex owns the final verdict header. This stub
> only records the partial/as-collected state so the directory is not untracked.

## Verdict (provisional)

**PARTIAL — remote run was still in progress when this snapshot was collected.**
No `summary.json` present locally. Final R1 sign-table row not derivable from this
snapshot alone; Codex should append the completed metrics section once the remote
benchmark finishes (or mark run-aborted if the container was reclaimed).

## What is here

- `poll.log` — `guard_bench.py` polling transcript. Last observed stage before
  snapshot ended: `throughput bucket=8-16K rep=3` at `2026-07-07T10:40:46+0800`,
  remote pid `13496` still alive.
- `start.stdout.log` / `start.stderr.log` — `start_remote_script` output.
- `upload.stdout.log` / `upload.stderr.log` — `upload_remote_script` output.
- No `raw/`, `throughput/`, `accuracy/` directories yet: the remote run had not
  written its final artifacts to the polled run dir at snapshot time.

## Run configuration (from poll.log)

- repo_kind=competition, repo_head=`d29e9db3ffa01b701346445c6e62fe963f6c17b1`
  branch=`contest-p1-ffn-pool-20260621`
- wheel=`.../dist/vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl`
  sha256=`a0f09295a60dc1e5f4f7e9a096f540f29165168047c3caaf37233b6e4cb8cfde`
- num_prompts=10 repetitions=3 buckets=4-8K 8-16K 16-32K accuracy=smoke accuracy_rows=10
- overlay_rev= (empty: this is the no-overlay / installed-wheel baseline path on a
  hot reused server)
- reuse_server=1, health=ok
- model_dir=/root/Qwen3.5-27B

## Intent

Same-container baseline for the `d29-revert333` and `a55-revert333` stop-loss
candidates prepared in the 2026-07-07 01:35–09:15 Codex R1 continuation block
(see `journal/2026-07-07.md`). Without a completed summary it cannot anchor R1
A/B; Codex needs to either finish it on a fresh container or re-run.