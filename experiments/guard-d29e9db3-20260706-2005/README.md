# guard-d29e9db3-20260706-2005

- Intent: R0.3 guard benchmark on current competition HEAD `d29e9db3ffa01b701346445c6e62fe963f6c17b1`.
- Method: local model copy, competition wheel install, warmup `4-8K 1`, then `4-8K` / `8-16K` / `16-32K` with 10 prompts x 3 repetitions.
- Status: throughput complete; full accuracy started but was manually interrupted after about 60 minutes to avoid burning the container window.
- Remote run dir: `/public/home/xdzs2026_c166/codex_runs/guard-d29e9db3-20260706-2005`
- Summary: `summary.json`
- Raw logs: `raw/`
- Throughput result JSONs: `throughput/`

## Runtime Fingerprint Correction

Later R1 fingerprinting in `experiments/r1-wheel-fingerprint-20260706-2320/` showed the installed competition wheel does **not** contain the `d29e9db3` `triton_unified_attention.py` source change. For that file, wheel/site-packages match `a55f3c3` and earlier, while source `d29e9db3` has a different SHA256. Interpret this run as a measurement of the installed competition wheel, not as proof of d29 source performance.

## Median Throughput

| bucket | output tok/s | TTFT-P99 ms | TPOT-P99 ms |
|---|---:|---:|---:|
| 4-8K | 12.227836 | 4535.416 | 69.270 |
| 8-16K | 7.229916 | 15629.511 | 70.574 |
| 16-32K | 4.651398 | 28698.031 | 72.138 |

Weighted output throughput: `7.455944844135703`.

## Accuracy

`run_accuracy.sh all` did not finish before the manual stop-loss. Accuracy is pending and must be rerun with either a longer window or a narrowed protocol.

Smoke rerun (`run_accuracy.sh all 1`) completed:

| dataset | score | rows |
|---|---:|---:|
| hotpotqa | 100.00 | 1 |
| gov_report | 30.51 | 1 |
| retrieval_multi_point | 100.00 | 1 |
| aggregation_keyword_aggregation | 100.00 | 1 |

This is a chain sanity check only, not a full accuracy delta.
