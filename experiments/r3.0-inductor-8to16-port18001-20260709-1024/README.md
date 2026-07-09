# R3.0 Inductor GEMM autotune 8-16K port18001 20260709-1024

## Intent

Same-container candidate for R3.0 Step2: enable PyTorch Inductor GEMM autotune and compare against `r3.0-baseline-8to16-port18001-20260709-0952`.

## Candidate switch

- `TORCHINDUCTOR_MAX_AUTOTUNE=1`
- `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1`
- `PYTHONPATH=/public/home/xdzs2026_c166/codex_runs/r3.0-inductor-8to16-port18001-20260709-1024/sitecustomize:/usr/local`
- `XDB_INDUCTOR_DUMP_CONFIG=1`
- `XDB_INDUCTOR_CONFIG_JSON=/public/home/xdzs2026_c166/codex_runs/r3.0-inductor-8to16-port18001-20260709-1024/config/config.json`

No TunableOp is enabled. The runtime GEMM backend list remains `ATEN,TRITON,CPP`.

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

The run used isolated copied testdata for non-8001 traffic and did not touch port 8001.

## Anchors

- Remote run: `/public/home/xdzs2026_c166/codex_runs/r3.0-inductor-8to16-port18001-20260709-1024`
- Local summary: `summary.json`
- Config dumps: `remote/config/`
- Logs: `remote/driver.log`, `remote/vllm_server.log`
- Repo head: `4653a56d52719dc8617d31e556e62a670ed5e6ec`
- Wheel sha256: `973e64e8fb9bc54c94a0b30837c87cc94095dee60d08c7ff5208ba36d09ef964`

## Validity

Valid screening candidate. The config dump proves `max_autotune=true` and `max_autotune_gemm=true`. `vllm_server.log` shows `max_seq_len=32768` and `load_format=runai_streamer`. Server was on port `18001`; cleanup left `8001/18001` closed and HCU memory use at `0%`.

## Metrics

8-16K median across 3 reps, 3 prompts per rep:

| metric | value |
|---|---:|
| output throughput | `8.091542528776012 tok/s` |
| TTFT-P99 | `12787.138073854148 ms` |
| TPOT-P99 | `70.01528968467291 ms` |
| duration | `78.35341626708396 s` |

## Verdict

Stop-loss. Against the same-container baseline, output throughput changed by `-0.055%`, TTFT-P99 by `+0.060%`, and TPOT-P99 by `+0.389%`. This is not a positive speed signal, so the candidate should not be expanded to three buckets, should not run accuracy, and should not enter defaults.
