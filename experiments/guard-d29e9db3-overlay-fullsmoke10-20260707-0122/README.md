# guard-d29e9db3-overlay-fullsmoke10-20260707-0122

## Verdict

**PASS / neutral-to-slight-positive versus a55 in this warm container.** This
run replaces the earlier partial d29 overlay attempt. It proves the source
`d29e9db3` attention path was actually active: runtime
`triton_unified_attention.py` SHA256 is
`acf4b51ba9250014a08ae54f91c775d3764cbf350856a077e484a76b52cba3f8`, while the
wheel file remains `8e8d393c...`.

Compared with the same-container a55/runtime-wheel-equivalent guard
`guard-a55f3c3-overlay-fullsmoke-20260707-0010`, the d29 source overlay is not
the expected large regression:

| bucket | a55 output tok/s | d29 output tok/s | delta |
|---|---:|---:|---:|
| 4-8K | 12.156717 | 12.223197 | +0.547% |
| 8-16K | 7.231679 | 7.230687 | -0.014% |
| 16-32K | 4.655501 | 4.655347 | -0.003% |

Weighted output throughput: `7.456586594423519`
(`+0.171%` versus a55's `7.443832877222891`).

## Accuracy Smoke

This uses the updated smoke tier: `run_accuracy.sh all 10`.

| dataset | metric | score |
|---|---|---:|
| hotpotqa | score | 67.71 |
| gov_report | score | 35.09 |
| retrieval_multi_point | recalculated_acc | 100.00 (10/10) |
| aggregation_keyword_aggregation | recalculated_acc | 100.00 (10/10) |

This is a 10-row smoke gate, not full 109-row accuracy.

Guard benchmark run.

- Intent: fixed warm-container guard protocol for Round 0 / Round 1 comparisons.
- Method: warmup once, then three throughput buckets x 3 repetitions, median summary, plus accuracy mode `smoke`.
- Buckets: `4-8K,8-16K,16-32K`
- Overlay rev: `d29e9db3`
- Remote run dir: `/public/home/xdzs2026_c166/codex_runs/guard-d29e9db3-overlay-fullsmoke10-20260707-0122`
- Local summary: `summary.json`
- Raw logs: `raw/`
- Throughput result JSONs: `throughput/`
