# guard-a55f3c3-overlay-fullsmoke-20260707-0010

## Verdict

**PASS as a complete a55/runtime-wheel-equivalent guard run.** This is the
current best baseline-safe candidate for excluding the unbuilt d29 source
attention path, because the installed competition wheel fingerprint matches
`a55f3c3` for `triton_unified_attention.py`.

This run does **not** complete the full R1 6+1 sign table. It compares a55 to
the installed-wheel guard from `experiments/guard-d29e9db3-20260706-2005/`,
which later fingerprinting showed was already a55-equivalent for the d29
attention file. Cross-run delta is small; the 4-8K bucket is lower by 0.58%,
so the strict R1 exit rule ("all three buckets >= current competition branch")
still needs same-container A/B if we want a formal pass.

## Median Throughput

| bucket | output tok/s | TTFT-P99 ms | TPOT-P99 ms | vs installed-wheel guard |
|---|---:|---:|---:|---:|
| 4-8K | 12.156717 | 4536.688 | 69.731 | -0.582% |
| 8-16K | 7.231679 | 15616.262 | 70.654 | +0.024% |
| 16-32K | 4.655501 | 28667.305 | 72.115 | +0.088% |

Weighted output throughput: `7.443832877222891`
(`-0.162%` vs `7.455944844135703` installed-wheel guard).

## Accuracy Smoke

`raw/accuracy.log` reports:

| dataset | metric | score |
|---|---|---:|
| hotpotqa | score | 100.00 |
| gov_report | score | 30.51 |
| retrieval_multi_point | recalculated_acc | 100.00 (1/1) |
| aggregation_keyword_aggregation | recalculated_acc | 100.00 (1/1) |

Caveat: the OpenCompass result JSON for `aggregation_keyword_aggregation`
stores `accuracy: 0.0`, while the guard parser recalculates the prediction as
1/1 because the predicted top-10 set matches the gold set. Treat this as a
smoke-chain sanity check plus an accuracy-mouthpiece question, not a full
accuracy delta.

## Runtime Fingerprint

Runtime `triton_unified_attention.py` SHA256:
`8e8d393c4d551547de397859462ad7c3750230841458e658d73381dbf3f59005`.
That matches the competition wheel and `a55f3c3`, not source `d29e9db3`.

Guard benchmark run.

- Intent: fixed warm-container guard protocol for Round 0 / Round 1 comparisons.
- Method: warmup once, then three throughput buckets x 3 repetitions, median summary, plus accuracy mode `smoke`.
- Buckets: `4-8K,8-16K,16-32K`
- Overlay rev: `a55f3c3`
- Remote run dir: `/public/home/xdzs2026_c166/codex_runs/guard-a55f3c3-overlay-fullsmoke-20260707-0010`
- Local summary: `summary.json`
- Raw logs: `raw/`
- Throughput result JSONs: `throughput/`
