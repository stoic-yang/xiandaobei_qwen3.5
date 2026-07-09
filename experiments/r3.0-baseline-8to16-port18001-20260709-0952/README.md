# R3.0 baseline 8-16K port18001 20260709-0952

## Intent

Fresh same-container baseline for the R3.0 Inductor GEMM autotune A/B in user-started container `Instances_2607090940238205_0_0` / Slurm job `661607`.

## Method

- `scripts/guard_bench.py`
- `--repo competition`
- `--locked-start-script`
- `--load-format runai_streamer`
- `--server-port 18001`
- `--buckets 8-16K`
- `--num-prompts 3`
- `--repetitions 3`
- `--accuracy none`
- shared caches: `VLLM_CACHE_ROOT=/public/home/xdzs2026_c166/vllm_cache/vllm_cache`, `TRITON_CACHE_DIR=/public/home/xdzs2026_c166/vllm_cache/triton_cache`

The run used isolated copied testdata for non-8001 traffic and did not touch port 8001.

## Anchors

- Remote run: `/public/home/xdzs2026_c166/codex_runs/r3.0-baseline-8to16-port18001-20260709-0952`
- Local summary: `summary.json`
- Logs: `remote/driver.log`, `remote/vllm_server.log`
- Repo head: `4653a56d52719dc8617d31e556e62a670ed5e6ec`
- Wheel sha256: `973e64e8fb9bc54c94a0b30837c87cc94095dee60d08c7ff5208ba36d09ef964`

## Validity

Valid screening baseline. `vllm_server.log` shows `max_model_len=32768`, engine config `max_seq_len=32768`, and `load_format=runai_streamer`. Server was on port `18001`; cleanup left `8001/18001` closed and HCU memory use at `0%`.

## Metrics

8-16K median across 3 reps, 3 prompts per rep:

| metric | value |
|---|---:|
| output throughput | `8.096026242866404 tok/s` |
| TTFT-P99 | `12779.450334100984 ms` |
| TPOT-P99 | `69.74396645218654 ms` |
| duration | `78.31002284097485 s` |
