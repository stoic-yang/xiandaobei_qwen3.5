# r3-fa-candidate-fullacc-20260707-2130

## Intent

R3.1 flash-attention prefill candidate 的本地 full accuracy 回归。GitLab/官方评测卡死时，用本地全量 accuracy 先排除明显精度崩塌。

## Method

- Reused the warm R3.1 candidate service from `r3-fa-candidate-accuracy-smoke-20260707-1948`.
- Because this run used `--reuse-server`, `summary.json` shows `locked_start_script=0`; locked口径 evidence is copied here as:
  - `reused_vllm_server.log`
  - `reused_start_vllm_locked.sh`
- Locked evidence:
  - `reused_vllm_server.log:13`: `Using max model len 32768`
  - `reused_vllm_server.log:20`: `max_seq_len=32768`, `load_format=runai_streamer`
  - `reused_vllm_server.log:33`: `XDB_R31_FLASH_ATTN_PREFILL enabled`
- Protocol: warmup `4-8K 1`, one 8-16K throughput request as guard script anchor, then `run_accuracy.sh all` full local set.

## Results

Throughput anchor, 8-16K x 1 prompt:

| output tok/s | TTFT-P99 | TPOT-P99 |
|---:|---:|---:|
| 8.621047 | 3275.072 ms | 70.269 ms |

Full local accuracy:

| metric | score |
|---|---:|
| hotpotqa | 77.96 |
| gov_report | 32.71 |
| retrieval_multi_point | 100.00 (30/30) |
| aggregation_keyword_aggregation | 100.00 (30/30) |

## Verdict

Pass as a local full-accuracy sanity check for R3.1: no class collapses to zero, retrieval and aggregation remain perfect on the local full set, and the candidate service remains on the locked `32768`/`runai_streamer`口径.

Open caveat: this is not a same-container baseline-vs-candidate full accuracy A/B, so `gov_report=32.71` should be treated as a regression watch item until either official scoring recovers or a same口径 baseline full accuracy is run.
