# R3.0 baseline screen newcontainer 20260708-155959

Status: invalid.

Intent: same-container 8-16K throughput-only baseline for R3.0 Inductor GEMM autotune screening.

Method: `scripts/guard_bench.py --locked-start-script --load-format runai_streamer --num-prompts 3 --repetitions 3 --buckets 8-16K --accuracy none`.

Verdict: discard this run. The service reached `/health`, but the guard got stuck during warmup and the vLLM server entered shutdown before producing any throughput JSON or `summary.json`. A separate 0.8B smoke service from `/public/home/xdzs2026_c166/testdata/start_vllm.sh` was observed on the same container/port/GPU during diagnosis, so this run is not a clean competition-baseline measurement.

Useful anchors:

- Remote run dir: `/public/home/xdzs2026_c166/codex_runs/r3.0-baseline-screen-newcontainer-20260708-155959`
- Guard log: `driver.log` showed `server_ready_at=2026-07-08T16:12:19+08:00`, then only warmup.
- Server log: `vllm_server.log` showed locked `max-model-len 32768`, `load_format=runai_streamer`, decode CUDA graph capture, then API shutdown before benchmark completion.
- Contamination source found during cleanup: `/public/home/xdzs2026_c166/codex_logs/0p8b_linear_only_baseline_8_16k_20260708_162543/start_vllm.log`.

Replacement: `r3.0-baseline-screen-newcontainer-retry-*`.
