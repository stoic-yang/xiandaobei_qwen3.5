# guard-a55f3c3-overlay-8to16-20260706-2252

- Intent: R1 screening check for the 8-16K bucket after removing the d29 source attention path.
- Method: reinstall competition wheel, overlay selected Python files from `a55f3c316b9c88aae957555f8b3994cfc32bee9a`, run warmup `4-8K 1`, then `8-16K` with 10 prompts x 3 repetitions, plus `run_accuracy.sh all 1` smoke.
- Status: screening pass, not a full R0.3 guard protocol.
- Remote run dir: `/public/home/xdzs2026_c166/codex_runs/guard-a55f3c3-overlay-8to16-20260706-2252`
- Summary: `summary.json`
- Raw logs: `raw/`

## Median 8-16K

| bucket | output tok/s | TTFT-P99 ms | TPOT-P99 ms |
|---|---:|---:|---:|
| 8-16K | 7.245884 | 15620.899 | 70.370 |

Compared with `guard-d29e9db3-20260706-2005` 8-16K median `7.229916`, this is about `+0.22%`. Because `r1-wheel-fingerprint-20260706-2320` shows the competition wheel already matched `a55f3c3` for the d29 attention file, treat this difference as run noise / screening evidence only, not a validated commit delta.

## Smoke Accuracy

| dataset | score | rows |
|---|---:|---:|
| hotpotqa | 100.00 | 1 |
| gov_report | 30.51 | 1 |
| retrieval_multi_point | 100.00 | 1 |
| aggregation_keyword_aggregation | 100.00 | 1 |

