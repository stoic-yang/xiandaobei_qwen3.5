# guard-fde463d-overlay-fullsmoke10-20260707-0236

## Verdict

**PASS / small positive versus a55 in the same warm container.** This run used
the updated smoke tier (`run_accuracy.sh all 10`) and completed before SCNet job
`656013` timed out. Runtime fingerprints show the measured path is `fde463d`
for `qwen3_5.py`/`qwen3_next.py`, while
`triton_unified_attention.py` remains the wheel/a55 SHA `8e8d393c...`; this is
not a measurement of the d29 attention source path.

Compared with the same-container a55/runtime-wheel-equivalent guard
`guard-a55f3c3-overlay-fullsmoke-20260707-0010`:

| bucket | a55 output tok/s | fde output tok/s | delta |
|---|---:|---:|---:|
| 4-8K | 12.156717 | 12.230124 | +0.604% |
| 8-16K | 7.231679 | 7.232895 | +0.017% |
| 16-32K | 4.655501 | 4.655773 | +0.006% |

Weighted output throughput: `7.459204300202086`
(`+0.207%` versus a55's `7.443832877222891`).

Compared with the completed d29 source-overlay guard
`guard-d29e9db3-overlay-fullsmoke10-20260707-0122`, fde is also slightly
positive on throughput (`+0.035%` weighted), but d29 and fde differ in which
source files are active, so treat that as a sign-table reference, not an
isolated single-file delta.

## Accuracy Smoke

This uses `run_accuracy.sh all 10`.

| dataset | metric | score |
|---|---|---:|
| hotpotqa | score | 67.71 |
| gov_report | score | 35.00 |
| retrieval_multi_point | recalculated_acc | 100.00 (10/10) |
| aggregation_keyword_aggregation | recalculated_acc | 100.00 (10/10) |

The only smoke10 delta versus d29 is gov_report `35.09 -> 35.00`; full 109-row
accuracy is still required before using this as a final submission gate.

Guard benchmark run.

- Intent: fixed warm-container guard protocol for Round 0 / Round 1 comparisons.
- Method: warmup once, then three throughput buckets x 3 repetitions, median summary, plus accuracy mode `smoke`.
- Buckets: `4-8K,8-16K,16-32K`
- Overlay rev: `fde463d`
- Remote run dir: `/public/home/xdzs2026_c166/codex_runs/guard-fde463d-overlay-fullsmoke10-20260707-0236`
- Local summary: `summary.json`
- Raw logs: `raw/`
- Throughput result JSONs: `throughput/`
