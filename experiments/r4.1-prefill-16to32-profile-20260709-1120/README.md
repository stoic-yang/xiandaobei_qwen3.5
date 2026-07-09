# R4.1 16-32K Prefill Roofline Profile

## Intent

Run `plans/task-r4.1-prefill-gemm-roofline.md` Step 1: capture one locked `16-32K` prefill-only profile under R3.1 current source, then compute the `Cijk_*` GEMM roofline.

## Status

Prepared locally. Remote run is launched separately with `driver.sh` on the current SCNet container.

## Remote Plan

- remote run dir: `/public/home/xdzs2026_c166/codex_runs/r4.1-prefill-16to32-profile-20260709-1120`
- port: `18001`
- request: first `16-32K_throughput.jsonl` prompt, `max_tokens=1`
- locked serving: `--max-model-len 32768`, `--max-num-seqs 128`, `--max-num-batched-tokens 4096`, `--load-format runai_streamer`
- source: install competition wheel, then overlay current `/public/home/xdzs2026_c166/vllm_cscc_competition` Python files into site-packages

## Expected Outputs

- `driver.log`
- `exit`
- `vllm_server.log`
- `profile/prefill_request.json`
- `profile/vllm_prefill16*.hipkernel.csv`
- `summary.remote.json`
