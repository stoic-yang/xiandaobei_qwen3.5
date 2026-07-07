---
name: xiandaobei-operator
description: Use when operating the Xiandaobei 2026 Qwen3.5/vLLM competition automation in /Users/keynary/Code/xiandaobei/meta: attaching SCNet containers, running guarded benchmarks, using the locked RunAI streamer startup path, interpreting local-vs-official score gaps, preparing guarded Xi GitLab submissions, or driving Chrome submission automation. Not for generic vLLM optimization outside this project.
---

# Xiandaobei Operator

## First Moves

Work from `/Users/keynary/Code/xiandaobei/meta` unless the user explicitly names
another worktree such as `meta-r0-r1`. Read `AGENTS.md` first, then read
`memory/00-index.md` and the current round files named by the user. Before any
remote experiment, benchmark, container-pool work, or submission preflight, read
`references/fast-iteration.md` so SSH reuse, locked startup, smoke/full tiers,
background jobs, and same-container A/B discipline are active by default.

For submission details, also read `automation/SUBMISSION.md` when present. For
daily context or concurrent agent state, read the latest `journal/YYYY-MM-DD.md`.

Do not edit shared SCNet source until the user asks for a concrete code change.
Do not save GitLab credentials. Do not change GitLab visibility. Do not use
Computer Use for browser submission when the Chrome connector can inspect and
drive the logged-in page.

## Workflow

Container state:

```bash
python3 scripts/scnetctl.py status
python3 scripts/scnetctl.py attach
ssh -F ~/.ssh/xiandaobei.generated.conf -O check xiandaobei-worker-auto
```

Guard benchmark with the validated locked RunAI startup path:

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

Acceptance check for any locked-start run:

```bash
rg -n "max_model_len=32768|max_seq_len=32768|load_format=runai_streamer" \
  experiments/<run_id>/vllm_server.log experiments/<run_id>/README.md
```

If the vLLM log shows `max_seq_len=262144`, mark the run invalid for competition
timing even if throughput numbers exist.

Legacy `scnetctl.py run` shortcuts are still useful for quick sanity checks, but
do not use remote `testdata/start_vllm.sh` as an official guard口径 unless the
log proves `max_seq_len=32768`.

For maintained container pools, smoke/full accuracy tiers, long-job
backgrounding, and controlled parallel A/B reporting, follow
`references/fast-iteration.md` instead of reconstructing policy from memory.

Submission preflight:

```bash
python3 scripts/submit_job.py repo-check
python3 scripts/submit_job.py manifest --test-result experiments/<run_id>/summary.json
```

GitLab push:

```bash
python3 scripts/submit_job.py push-gitlab --test-result experiments/<run_id>/summary.json --write-manifest
```

If GitLab cannot accept the vLLM history, use `submit_job.py push-snapshot`
instead of manually reconstructing `commit-tree` commands. If GitLab returns
502 or visibility is unknown, stop writes and keep preparing local artifacts.

## Score Discipline

The local 10-prompt proxy is a sanity gate, not proof of leaderboard score.
After an official submission, inspect the platform row before claiming baseline.
If an environment-switch experiment regresses the targeted bucket, do not submit
or bake it into defaults.

Parallel candidate screening must report per-container baseline, candidate, and
relative delta; do not compare cross-container absolute throughput.

## Artifacts

Write run outputs under `experiments/<run_id>/`, manifests and browser evidence
under `snapshots/`, daily timeline entries under `journal/YYYY-MM-DD.md`, and
slow-changing facts under `memory/` with a dated changelog note.
