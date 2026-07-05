# SCNet automation

This directory contains the local automation layer for the xiandaobei Qwen3.5
vLLM contest work. The goal is to let Codex drive a clean baseline run with no
manual terminal relay once a Chrome session is logged in to SCNet.

## Safety model

- Source edits stay out of shared vLLM checkouts unless explicitly requested.
- Runs write to `/public/home/xdzs2026_c166/codex_runs/<run_id>/`.
- Collected summaries write to `meta/experiments/<run_id>/`.
- `run` refuses to start when it sees active `vllm serve`, `vllm bench`,
  `run_throughput`, `run_accuracy`, or `opencompass` processes unless `--force`
  is supplied.
- `start` does not buy resources, delete containers, stop containers, save
  images, or bypass login/CAPTCHA prompts.

## Commands

```bash
python3 meta/scripts/scnetctl.py status
python3 meta/scripts/scnetctl.py start
python3 meta/scripts/scnetctl.py attach
python3 meta/scripts/scnetctl.py run baseline-smoke --dry-run
python3 meta/scripts/scnetctl.py run baseline-full
```

`attach` resolves the current Slurm job, asks the compute node for the container
IP, and writes `~/.ssh/xiandaobei.generated.conf`. Long-running commands use the
generated `xiandaobei-worker-auto` host so container IP churn does not require
hand editing `~/.ssh/config`.

## Baseline tasks

- `baseline-smoke`: install baseline wheel, start vLLM, run all three throughput
  buckets with one prompt each, collect logs.
- `baseline-full`: install baseline wheel, start vLLM, run all three throughput
  buckets with ten prompts each, collect logs. Accuracy is disabled by default
  for this task; pass `--accuracy smoke` or `--accuracy full` when needed.

## Current Chrome boundary

The SSH/experiment runner is a normal Python script. SCNet container creation is
controlled through the Codex Chrome extension because it depends on the user's
logged-in browser session. The current Chrome adapter can identify an already
running `qwen3.5-dtk26.04:0509` instance. The stopped-instance start click still
needs validation the next time the instance is stopped.
