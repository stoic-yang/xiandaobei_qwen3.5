# guard-d29e9db3-overlay-fullsmoke-20260706-2355

## Verdict

**PARTIAL / negative-suspicion only.** This run proves that the `d29e9db3`
runtime overlay was applied and the server reached `/health`, but it was
manually terminated before the full three-bucket guard completed.

Observed behavior:

- `triton_unified_attention.py` site SHA changed from the wheel/a55 hash
  `8e8d393c...` to the d29 source hash `acf4b51...`.
- vLLM reached `server_ready_at=2026-07-07T00:04:18+08:00`.
- The first 4-8K warmup request took `128.30s`, far outside the normal warm
  path, but the post-warm single-prompt main run reported TTFT `4338.56ms`,
  TPOT `69.53ms`, and output throughput `7.73 tok/s` for that one request.
- The run was terminated at `4-8K rep=1`; no full three-bucket median exists.

Use this as evidence that d29 has abnormal first-request/warmup cost under the
guard setup, not as a completed R1 sign table row. A full d29 post-warm A/B
still needs a bounded rerun if we want a strict throughput sign.

Guard benchmark run.

- Intent: fixed warm-container guard protocol for Round 0 / Round 1 comparisons.
- Method: warmup once, then three throughput buckets x 3 repetitions, median summary, plus accuracy mode `smoke`.
- Buckets: `4-8K,8-16K,16-32K`
- Overlay rev: `d29e9db3`
- Remote run dir: `/public/home/xdzs2026_c166/codex_runs/guard-d29e9db3-overlay-fullsmoke-20260706-2355`
- Local summary: `summary.json`
- Raw logs: `raw/`
- Throughput result JSONs: `throughput/`
